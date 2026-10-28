"""The calculated stage: derive what a household does not deliver.

Stage 4 of the pipeline. Reads the imputed households, works out per household
which catalog variables it can derive from what it has, derives them, fills the
gaps the derivation cannot reach, and writes the result as a hive shard tree --
one shard per (household, household batch).

This stage lives apart from aggregate.py, which holds the imputation and the
resample/aggregate stages. The three stages share almost nothing but the
options object, and reading one of them inside a file holding all three is
harder than it needs to be.

The expression layer is calculated_columns.py (how a formula becomes a column)
and the per-cell flags are calculated_flags.py (what the stage records about
each value it produced). This module is the orchestration: availability,
schema groups, provenance, writes, checkpoints.
"""

import logging
import os

import ibis
import pandas as pd

import etdtransform
from etdtransform.calculated_columns import _catalog_hash
from etdmap import ids


def _write_calc_provenance(output_path, plan_rows, hh_rows, catalog_hash,
                           fill_plan_frames=None):
    """Persist per-schema-group derivation provenance next to household_calculated.

    Two compact sidecars (factored by schema group, so size is groups x derived
    vars + one row per household, not households x vars):
      - derivation_plan.parquet:      schema_group_id -> (variable, rhs_text, rhs_vars)
      - household_derivation.parquet: ``ids.HOUSEHOLD_BATCH`` -> schema_group_id

    A variable absent from a group's plan was provided/measured (not derived).
    Both carry catalog_hash so the equations are reproducible.

    The gap fill's plan goes to a third, calculation_fill_plan.parquet, rather
    than into derivation_plan: its rows are keyed by the not-null map a group
    of rows shares, not by schema group, so one row per (map, variable) would
    not fit the other file's shape. Together they say which formula produced
    any calculated cell -- the plan for a column the stage derived, the fill
    plan for a gap it filled afterwards.
    """
    out_dir = os.path.dirname(output_path) or "."
    plan_df = pd.DataFrame(
        plan_rows, columns=["schema_group_id", "variable", "rhs_text", "rhs_vars"]
    ).astype({
        "schema_group_id": "Int64", "variable": "string",
        "rhs_text": "string", "rhs_vars": "string",
    })
    plan_df["catalog_hash"] = pd.array([catalog_hash] * len(plan_df), dtype="string")
    plan_df.to_parquet(os.path.join(out_dir, "derivation_plan.parquet"))

    hh_df = pd.DataFrame(
        hh_rows, columns=[ids.HOUSEHOLD_BATCH, "schema_group_id"]
    ).astype({ids.HOUSEHOLD_BATCH: "Int64", "schema_group_id": "Int64"})
    hh_df["catalog_hash"] = pd.array([catalog_hash] * len(hh_df), dtype="string")
    hh_df.to_parquet(os.path.join(out_dir, "household_derivation.parquet"))

    if fill_plan_frames:
        fill_df = pd.concat(fill_plan_frames, ignore_index=True)
        fill_df["catalog_hash"] = pd.array(
            [catalog_hash] * len(fill_df), dtype="string"
        )
        fill_df.to_parquet(os.path.join(out_dir, "calculation_fill_plan.parquet"))
        logging.info(
            f"[calc ibis] gap fill provenance: {len(fill_df)} "
            f"(map, variable) row(s), "
            f"{int(fill_df['cells_filled'].sum()):,} cell(s) filled, "
            f"{int(fill_df['cells_rejected'].sum()):,} rejected by their "
            f"row's rule bound"
        )

    logging.info(
        f"[calc ibis] derivation provenance: {len(plan_df)} plan rows over "
        f"{hh_df['schema_group_id'].nunique()} schema groups, "
        f"{len(hh_df)} household batches, catalog {catalog_hash}"
    )



def add_calculated_columns_to_hh_data_ibis(
    source_path: str,
    output_path: str,
    catalog_df=None,
    target_columns=None,
) -> None:
    """
    Ibis/DuckDB variant: derives calculated columns without a full pandas load.

    Reads source_path directly from disk, computes per-household column
    availability in one DuckDB GROUP BY pass, groups households by schema
    (frozenset of available catalog input columns), derives calculated columns
    per schema group via ibis mutate, and writes combined output to output_path.

    Phases:
      1. Availability analysis -- one DuckDB GROUP BY pass over the parquet.
      2. Planning -- DatasetAdapter.execution_plan_for_available per schema group.
      3. Mutate per schema group -- ibis filter + sequential mutate per derived col.
      4. Write -- one hive shard per (household, household batch).

    Parameters
    ----------
    source_path : str
        The imputed household artifact (input).
    output_path : str
        Directory for the calculated hive shards
        (HuisIdBSV=<n>/HuisBatchIdBSV=<p>/). Cleared before writing, so
        re-runs are idempotent. Requires ``ids.HOUSEHOLD_BATCH`` in the source
        (a sharded directory source provides it as a hive column).

    Notes
    -----
    source_path may be a parquet FILE or a hive-sharded DIRECTORY
    (HuisIdBSV=<n>/HuisBatchIdBSV=<p>/*.parquet); reads are format-detected.
    With a directory source the input is validated at entry: a household with
    more than one household batch over the same ReadingDate raises (overlapping household batch
    periods).
    """
    import shutil
    from etdmap.catalog import load_catalog
    from etdtransform.calculated_columns import undefined_where
    from etdtransform.calculated_flags import (
        calculated_flag_expressions,
        fill_gaps_in_table,
    )
    from etdtransform.checkpoints import Checkpoints

    checkpoints = Checkpoints(stage="household_calculated")

    # A calculated value is judged at the GRID interval: the derivation runs
    # after imputation, so every variable it reads sits on the grid and one
    # grid interval is what a calculated cell covers. An unreadable grid leaves
    # the marks off rather than failing the derivation.
    try:
        from etdtransform.bounds import dataset_grid_minutes

        _calc_grid_minutes = dataset_grid_minutes()
    except (FileNotFoundError, ValueError) as _grid_error:
        _calc_grid_minutes = None
        logging.warning(
            "[calc ibis] mapped dataset's grid unavailable (%s) -- CALCULATED_COLUMN "
            "and CALCULATED_THRESHOLD_VIOLATED will not be written",
            _grid_error,
        )
    from etdmap.data_model import all_performance_data_columns, required_performance_data_columns
    from etdtransform.catalog.compiled import compile_plan
    from etdtransform.catalog.query import DatasetAdapter
    from etdtransform.calculated_columns import sympy_to_ibis

    source_path = str(source_path)
    output_path = str(output_path)

    logging.info("[calc ibis] Loading catalog...")
    if catalog_df is None:
        catalog_df = load_catalog()
    adapter = DatasetAdapter(catalog_df)

    if target_columns is not None:
        all_perf_cols = set(target_columns)
        req_perf_cols = set()
    else:
        all_perf_cols = set(all_performance_data_columns)
        req_perf_cols = set(required_performance_data_columns)
    derivable_in_catalog = set(catalog_df["lhs"].unique())

    # Source is format-detected (file or hive shard directory) by the shared
    # read layer.
    from etdmap.storage import source_is_sharded
    from etdtransform.load_data import source_table

    source_is_shards = source_is_sharded(source_path)

    def _read_source_table(household_ids=None):
        """The source, restricted to *household_ids* when they are known.

        Scope has to be set BEFORE planning, not after. A read that globs the
        whole tree and filters afterwards costs more per unit the more units
        there are, because every unit re-resolves every shard. Naming only the
        unit's own files keeps the per-unit cost flat whatever the scale.
        """
        return source_table(source_path, household_ids=household_ids)

    # Phase 1: schema check + rename shim + availability analysis
    tbl = _read_source_table()
    parquet_col_set = set(tbl.columns)

    # Every cumulative variable should carry its interval column by now, and
    # the reverse: the mapping derives whichever the supplier did not deliver.
    # A missing half reads here as the variable being absent rather than as a
    # skipped step, so it has to stop the run instead of changing the plan.
    from etdtransform.impute import assert_cumulative_interval_pairs_complete

    assert_cumulative_interval_pairs_complete(parquet_col_set)

    if source_is_shards:
        # Batch-safety guard at this compute entry point: a timestamp twice
        # in one household batch raises.
        from etdtransform.impute import _assert_batch_safe_ibis
        _assert_batch_safe_ibis(tbl)
    if ids.HOUSEHOLD_BATCH not in parquet_col_set:
        raise ValueError(
            f"add_calculated_columns_to_hh_data_ibis: the output is written "
            f"one shard per (household, household batch) and the source has "
            f"no {ids.HOUSEHOLD_BATCH} column. A household artifact always "
            f"carries the household batch it was mapped as."
        )

    # Zon-opwekTotaalDiff -> ZonopwekBruto compatibility shim
    # (hyphenated name is unparseable as a SymPy symbol; rename before catalog use)
    has_zon_rename = (
        "Zon-opwekTotaalDiff" in parquet_col_set
        and "ZonopwekBruto" not in parquet_col_set
    )
    if has_zon_rename:
        tbl = tbl.rename({"ZonopwekBruto": "Zon-opwekTotaalDiff"})
        logging.info(
            "[calc ibis] Renamed Zon-opwekTotaalDiff -> ZonopwekBruto "
            "(compatibility shim)"
        )

    effective_cols = set(tbl.columns)

    # Candidate input columns: catalog rhs_vars + performance data cols, present in parquet
    all_rhs_vars = {v for vars_list in catalog_df["rhs_vars"] for v in vars_list}
    analysis_cols = sorted(effective_cols & (all_rhs_vars | all_perf_cols))

    logging.info(
        f"[calc ibis] Computing per-household-batch availability for "
        f"{len(analysis_cols)} candidate columns..."
    )

    agg_exprs = [tbl[col].notnull().mean().name(col) for col in analysis_cols]
    with checkpoints.phase("availability_scan") as mark:
        availability_df = tbl.group_by(ids.HOUSEHOLD_BATCH).agg(agg_exprs).execute()
        mark["rows"] = len(availability_df)
        mark["note"] = f"{len(analysis_cols)} candidate columns"

    # Each household batch's own formula set and bounds, for the gap fill at
    # the end of the stage. Resolved once for the whole run: household batches
    # sharing an ApparaatOntbrekend declaration share both objects, and the
    # presolve is the expensive half.
    _rule_sets = None
    from etdtransform.calculated_flags import KEY_COLUMNS as _fill_keys

    _missing_keys = [c for c in _fill_keys if c not in effective_cols]
    if _missing_keys:
        logging.warning(
            "[calc ibis] the source has no %s, so a filled value cannot be "
            "joined back to its row -- the gap fill will not run",
            ", ".join(_missing_keys),
        )
    elif _calc_grid_minutes is not None:
        try:
            from etdtransform.constraints import rule_sets_by_household_batch

            with checkpoints.phase("household_rule_sets") as mark:
                _rule_sets = rule_sets_by_household_batch(
                    availability_df[ids.HOUSEHOLD_BATCH], catalog_df,
                    _calc_grid_minutes,
                )
                mark["note"] = f"{len(_rule_sets)} apparaat-set(s)"
        except (FileNotFoundError, ValueError) as _rules_error:
            logging.warning(
                "[calc ibis] cannot resolve per-household rule sets (%s) -- "
                "the gap fill will not run and no cell will be filled from a "
                "formula or a rule bound",
                _rules_error,
            )

    # Build schema groups: frozenset(available_cols) -> household batch ids
    schema_groups: dict = {}
    for _, row in availability_df.iterrows():
        batch_id = int(row[ids.HOUSEHOLD_BATCH])
        avail = frozenset(
            col for col in analysis_cols
            # Available means the column holds at least one value. No fraction
            # test: pd.NA propagates, so a gap in an input becomes a gap in the
            # output rather than a silent zero.
            if not pd.isna(row.get(col, float("nan"))) and row[col] > 0
        )
        schema_groups.setdefault(avail, []).append(batch_id)

    logging.info(
        f"[calc ibis] {len(schema_groups)} unique schema groups "
        f"across {len(availability_df)} household batches"
    )

    # Phase 2 + 3: plan and mutate per schema group
    _written_units: list = []
    _PARTITION_KEYS = (ids.HOUSEHOLD, ids.HOUSEHOLD_BATCH)

    # Which household batch each household's shard belongs to, resolved in one pass
    # rather than per unit.
    #
    # THE UNIT IS THE SHARD: one (household, household batch) pair, not one
    # household. A household can appear in several household batches, and between them
    # it may have been renovated -- different devices, different metadata --
    # so its household batches are different household batches that happen to share an address.
    # Collapsing them to one key per household would file one household batch's rows
    # under another household batch's path.
    with checkpoints.phase("shard_units") as mark:
        _pairs = (
            tbl.select(*_PARTITION_KEYS).distinct().execute()
        )
        _shard_units = sorted(
            (int(row.HuisIdBSV), int(row.HuisBatchIdBSV))
            for row in _pairs.itertuples()
        )
        _households_with_many = len(
            {h for h, _ in _shard_units}
        ) != len(_shard_units)
        mark["rows"] = len(_shard_units)
        mark["note"] = (
            f"{len(_shard_units)} shard(s) over "
            f"{len({h for h, _ in _shard_units})} household(s)"
            + ("; some households span household batches"
               if _households_with_many else "")
        )

    # The writer owns and clears its output directory -- re-runs are
    # idempotent (same convention as the sharded impute).
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    # Derivation provenance accumulators (which equation derived each variable
    # per schema group, and which households belong to each group).
    _prov_plan_rows: list = []
    _prov_hh_rows: list = []
    _fill_plan_frames: list = []
    _cat_hash = _catalog_hash(catalog_df)

    for group_idx, (available_cols, group_ids) in enumerate(schema_groups.items()):
        targets = (all_perf_cols - available_cols) & derivable_in_catalog

        if not targets:
            plan: list = []
            not_derivable: list = []
        else:
            with checkpoints.phase("pass1_plan_formulas", note=f"group {group_idx}"):
                plan, not_derivable = adapter.execution_plan_for_available(
                    available_cols, targets
                )
                # The plan is fixed for the whole group, while _pass1_table
                # runs once per shard unit. Compiling here keeps lambdify out
                # of that loop.
                plan = compile_plan(plan)

        if not_derivable:
            # Find which RHS input columns failed the completeness threshold
            # (present in the parquet but below threshold -- not absent entirely).
            needed_rhs: set = set()
            for tgt in not_derivable:
                for _, rule_row in catalog_df[catalog_df["lhs"] == tgt].iterrows():
                    needed_rhs.update(rule_row["rhs_vars"])
            blocked_inputs = sorted((needed_rhs - available_cols) & set(analysis_cols))
            if blocked_inputs:
                group_avail = availability_df[
                    availability_df[ids.HOUSEHOLD_BATCH].isin(group_ids)
                ]
                avail_by_col = {
                    col: float(group_avail[col].mean())
                    for col in blocked_inputs
                    if col in group_avail.columns
                }
                blocked_str = ", ".join(
                    f"{c}={v:.1%}"
                    for c, v in sorted(avail_by_col.items(), key=lambda x: x[1])
                )
                logging.warning(
                    f"[calc ibis] Schema group {group_idx}: derivation blocked -- "
                    f"no usable data in this household: "
                    f"{blocked_str}. "
                    f"A column with no values at all cannot feed a formula."
                )

            not_req = set(not_derivable) & req_perf_cols
            hh_sample = sorted(group_ids)[:10]
            hh_note = (
                f"{len(group_ids)} HH: {hh_sample}"
                if len(group_ids) <= 10
                else f"{len(group_ids)} HH (first 10): {hh_sample}"
            )
            logging.error(
                f"[calc ibis] Schema group {group_idx} ({hh_note}): "
                f"could not derive {sorted(not_derivable)}"
                + (f" (required: {sorted(not_req)})" if not_req else "")
            )

        def _pass1_table(household_ids, batch_ids=None,
                         record_provenance=False):
            """PASS 1: the regular formula calculations, as a lazy table.

            Derives each column this schema group cannot get from its
            supplier, by evaluating the catalog formula the plan chose,
            and marks every derived cell. Nothing executes here -- the
            expressions are built and forced later, by whoever writes.
            """
            scoped = _read_source_table(household_ids=household_ids)
            scoped = scoped.filter(
                scoped[ids.HOUSEHOLD].isin(household_ids)
            )
            if batch_ids is not None:
                # A household's household batches are separate household batches; a unit is
                # one of them, never all of a household's rows.
                scoped = scoped.filter(
                    scoped[ids.HOUSEHOLD_BATCH].isin(batch_ids)
                )
            if has_zon_rename:
                scoped = scoped.rename({"ZonopwekBruto": "Zon-opwekTotaalDiff"})

            derived = []
            undefined = {}
            for col_name, compiled in plan:
                # The undefined mask reads the denominator, so it is built
                # against the table BEFORE the target column is added.
                _undefined = undefined_where(compiled, scoped)
                scoped = scoped.mutate(
                    **{col_name: sympy_to_ibis(compiled, scoped)}
                )
                if _undefined is not None:
                    undefined[col_name] = _undefined
                derived.append(col_name)
                if record_provenance:
                    _prov_plan_rows.append({
                        "schema_group_id": group_idx,
                        "variable": col_name,
                        "rhs_text": str(compiled.expr),
                        "rhs_vars": ",".join(compiled.symbols),
                    })
            # Mark the derived cells and flag the ones outside their
            # declared bound. Nothing else in the pipeline checks a
            # derived value against its threshold: the validate_ column
            # for a derived variable is written before this stage runs,
            # so it exists and is empty.
            if derived and _calc_grid_minutes is not None:
                scoped = scoped.mutate(
                    **calculated_flag_expressions(
                        scoped, derived, _calc_grid_minutes,
                        undefined=undefined,
                    )
                )
            return scoped

        def _pass2_fill(table, household_batch_ids):
            """PASS 2: fill the gaps pass 1 could not reach.

            Two routes, each recorded separately: cell filling evaluates a
            formula whose inputs happen to be present in a row whose
            target is not, and zero-width derivation takes the single
            value the rules narrow a row to. Integrated, not optional: a
            cell this stage can determine is a cell it should produce.
            """
            if _rule_sets is None:
                return table
            filled, report, fill_plan = fill_gaps_in_table(
                table, household_batch_ids, _rule_sets,
                checkpoints=checkpoints,
            )
            if not fill_plan.empty:
                _fill_plan_frames.append(
                    fill_plan.assign(schema_group_id=group_idx)
                )
            return filled

        for _bid in group_ids:
            _prov_hh_rows.append(
                {ids.HOUSEHOLD_BATCH: int(_bid), "schema_group_id": group_idx}
            )

        # ONE SHARD IS THE UNIT OF WORK: one (household, household batch) pair,
        # derived, filled and written on its own, so the amount of data in
        # flight never depends on how many units exist. A write whose
        # fan-out is proportional to the whole population is not a bulk
        # write: scattering every unit in one partitioned COPY costs
        # superlinearly in rows, in time and in peak memory, while writing
        # per unit keeps both flat.
        _group_id_set = {int(b) for b in group_ids}
        _group_units = [
            unit for unit in _shard_units if unit[1] in _group_id_set
        ]
        if not _group_units:
            raise ValueError(
                f"schema group {group_idx} has household batches "
                f"{sorted(group_ids)[:5]} with no shard in the "
                f"source; a unit path cannot be formed."
            )
        for _unit_index, (_household_id, _batch_id) in enumerate(
            _group_units
        ):
            with checkpoints.phase("pass1_build_expressions") as mark:
                _unit_tbl = _pass1_table(
                    [_household_id], batch_ids=[_batch_id],
                    record_provenance=(_unit_index == 0),
                )
                mark["note"] = f"{len(plan)} formula(s)"
            _unit_tbl = _pass2_fill(_unit_tbl, [_batch_id])
            with checkpoints.phase(
                "pass1_derive_and_write_shard",
                note=f"household {_household_id} batch {_batch_id}",
            ):
                _shard_dir = os.path.join(
                    output_path,
                    f"HuisIdBSV={_household_id}",
                    f"HuisBatchIdBSV={_batch_id}",
                )
                os.makedirs(_shard_dir, exist_ok=True)
                # Hive convention: the partition ids live in the path,
                # not in the file, as the imputed shards do.
                _unit_tbl.drop(*_PARTITION_KEYS).to_parquet(
                    os.path.join(_shard_dir, "part.parquet")
                )
        _written_units.extend(_group_units)
        logging.info(
            f"[calc ibis] Group {group_idx + 1}/{len(schema_groups)}: "
            f"{len(group_ids)} HH, {len(plan)} columns derived"
        )

    if not _written_units:
        raise RuntimeError(
            "[calc ibis] No shard units written -- no output produced."
        )
    # Every unit the source holds must be in the output. A household
    # whose household batches were collapsed to one key would pass a row count
    # and still have lost a household batch, so the units are compared as pairs.
    _missing_units = sorted(set(_shard_units) - set(_written_units))
    if _missing_units:
        raise RuntimeError(
            f"[calc ibis] {len(_missing_units)} source shard(s) were "
            f"not written: {_missing_units[:5]}. Every "
            f"({ids.HOUSEHOLD}, {ids.HOUSEHOLD_BATCH}) in the source must "
            f"appear in the output."
        )
    logging.info(
        f"[calc ibis] Wrote {len(_written_units)} shard "
        f"unit(s), one write per (household, household batch)"
    )
    logging.info(
        f"[calc ibis] Saved calculated hive shards -> {output_path}"
    )
    _write_calc_provenance(
        output_path, _prov_plan_rows, _prov_hh_rows, _cat_hash,
        fill_plan_frames=_fill_plan_frames,
    )
    checkpoints.log_summary()
    checkpoints.write(output_path)


