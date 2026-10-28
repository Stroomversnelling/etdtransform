import logging
import os
import re
from typing import Optional

import ibis
import numpy as np
import pandas as pd
from etdmap.data_model import (
    cumulative_column,
    cumulative_columns,
    get_aggregation_config,
    interval_column,
)
from etdmap.index_helpers import read_index

import etdtransform
from etdtransform.calculated_columns import _catalog_hash
# Re-exported so callers can import the calculated stage from here; the
# stage itself lives in calculated_stage.py.
from etdtransform.calculated_stage import (  # noqa: F401
    add_calculated_columns_to_hh_data_ibis,
)
from etdtransform.impute import read_diffs
from etdmap import ids

"""
Aggregating the data for a given time interval
Example intervals:
1 hour: '1h'
15 min: '15min'
5 min: '5min'
"""


def read_hh_data(interval="default", metadata_columns=None):
    """
    Read household data from a parquet file and optionally add index columns.

    Parameters
    ----------
    interval : str, optional
        The time interval of the data to read, by default "default"
    metadata_columns : list, optional
        Additional columns to include from the index, by default None

    Returns
    -------
    pd.DataFrame
        The household data with optional index columns added

    Notes
    -----
    This function reads parquet files from a predefined folder path.
    """
    if not metadata_columns:
        metadata_columns = []
    from etdmap.storage import read_source_frame, resolve_source_path
    df = read_source_frame(resolve_source_path(
        etdtransform.options.aggregate_folder_path, f"household_{interval}"
    ))
    return add_index_columns(df, columns=metadata_columns)


def add_index_columns(df: pd.DataFrame, columns: Optional[list] = None) -> pd.DataFrame:
    """
    Add index columns to the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame
    columns : list, optional
        Additional columns to include from the index, by default None

    Returns
    -------
    pd.DataFrame
        The DataFrame with added index columns

    Notes
    -----
    Merges the input DataFrame with the index on ``ids.HOUSEHOLD_BATCH``, the
    index's own key. A household occurs at most once in a project, so joining
    on the household together with its project reaches the same single row --
    the same key written in two columns, leaning on an invariant it does not
    state.
    """
    if columns:
        index_df, index_path = read_index()
        columns_to_select = [ids.HOUSEHOLD_BATCH, *columns]
        columns_to_select = list(set(columns_to_select))
        index_df = index_df[columns_to_select]
        df = df.merge(index_df, on=ids.HOUSEHOLD_BATCH, how="left")
        return df
    else:
        return df


def reconstruct_cumulative_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Rebuild cumulative columns from their imputed Diff columns in-place.

    For each col in cols:
      - col + "Original"  = original cumulative value (saved for audit)
      - col               = cumsum of its interval column per household batch (rebuilt)
      - col + "Check"     = per-household-batch diff of (rebuilt - original)

    The interval column is the one the data model links to the cumulative
    variable, which is not always its name plus "Diff".

    "Original" is the cumulative column as this stage received it. For a column
    the mapping stage built from delivered intervals
    (MappingFlag.CUMULATIVE_CONSTRUCTED) that is itself a running total starting
    from its own first interval, so its absolute level carries no meter reading
    and only differences across it mean anything. "Check" is a difference either
    way, so it remains the drift imputation introduced.

    Data must be sorted by household batch then ReadingDate before calling
    this. Mutates df in-place and returns it.
    """
    for col in cols:
        df[col + "Original"] = df[col]
        df[col] = (
            df.groupby(ids.HOUSEHOLD_BATCH, sort=False)[interval_column(col)]
            .cumsum()
        )
        df[col + "Check"] = (
            (df[col] - df[col + "Original"])
            .groupby(df[ids.HOUSEHOLD_BATCH]).diff()
        )
    return df


def reconstruct_cumulative_columns_ibis(tbl: "ibis.Table", cols: list) -> "ibis.Table":
    """
    Ibis/DuckDB variant of reconstruct_cumulative_columns.

    Builds lazy window expressions -- nothing is executed until the caller
    calls .to_parquet() or .execute().  All three derived columns per col
    (Original, rebuilt cumsum, Check) are added as Ibis mutations, and carry the
    same meaning as in reconstruct_cumulative_columns.

    Typical pipeline use:
        tbl = ibis.read_parquet(imputed_path)
        tbl = reconstruct_cumulative_columns_ibis(tbl, cum_cols)
        tbl.to_parquet(imputed_path)   # overwrites with reconstructed data
    """
    win_cum = ibis.window(
        group_by=ids.HOUSEHOLD_BATCH, order_by="ReadingDate",
        preceding=None, following=0,
    )
    win_lag = ibis.window(group_by=ids.HOUSEHOLD_BATCH, order_by="ReadingDate")

    # Pass 1: all orig_col saves + cumsum rewrites in one mutate().
    # Batching avoids building a 2*len(cols)-deep nested subquery chain --
    # each loop iteration would otherwise wrap tbl in another subquery level,
    # causing DuckDB query-planning time to grow quadratically with col count.
    # All cols are independent of each other so a single mutate is correct.
    first_pass = {}
    for col in cols:
        first_pass[col + "Original"] = tbl[col]
        first_pass[col] = tbl[interval_column(col)].sum().over(win_cum)
    tbl = tbl.mutate(**first_pass)

    # Pass 2: all check_col computations in one mutate().
    # Each check_col references the updated col and orig_col from pass 1 but
    # is independent of every other check_col, so batching is safe.
    second_pass = {}
    for col in cols:
        rmo = tbl[col] - tbl[col + "Original"]
        second_pass[col + "Check"] = rmo - rmo.lag(1).over(win_lag)
    tbl = tbl.mutate(**second_pass)

    return tbl


#: The in-run household summary carries only what the persisted outputs
#: cannot reproduce: keys, the in-run value stats (pre-threshold
#: diff_col_total, pre-reconstruction cum_col_min_max_diff), the factor
#: decision record and total_records. The counts half is derived and
#: rejoined by etdworkflow/postprocess_reports.py.
_SLIM_SUMMARY_KEYS = (ids.PROJECT, ids.HOUSEHOLD_BATCH, "column")
_SLIM_SUMMARY_STATS = ("diff_col_total", "cum_col_min_max_diff")
_SLIM_SUMMARY_FACTORS = (
    "house_impute_factor", "house_factor_is_fallback", "house_diff_median",
    "project_avg_median", "comparable_counts", "total_counts",
)


def _slim_summary_columns(frame) -> list:
    """The slim in-run summary columns present on *frame*, in canon order."""
    ordered = (
        list(_SLIM_SUMMARY_KEYS)
        + list(_SLIM_SUMMARY_STATS)
        + list(_SLIM_SUMMARY_FACTORS)
        + ["total_records"]
    )
    return [c for c in ordered if c in frame.columns]


_VOLUME_SUMS = (
    "total_routed",
    "total_final",
    "deviation",
    "redistributed_volume",
    "uniform_residual_volume",
    "unplaced_volume",
    "unanchored_removed_volume",
)
_VOLUME_COUNTS = (
    "cells_clipped",
    "cells_absorbed",
    "cells_residual",
    "cells_not_filled",
    "gap_length",
)



#: Per-chunk wall-time budget for the imputation shell, in seconds. A chunk
#: beyond this budget is announced loudly so a regression is visible on the
#: FIRST chunk of a run, not after it.
CHUNK_BUDGET_SECONDS = 720.0


class _ChunkTimer:
    """Times each chunk of the imputation shell and projects the whole run.

    Logs every chunk's wall time with a running projection, and warns --
    once, loudly -- as soon as the average chunk exceeds the budget, naming
    the projected total. The guard exists so a slow run announces itself at
    chunk one rather than being discovered from the outside half-way in.
    """

    def __init__(self, n_chunks, log_prefix, budget_seconds=None):
        import time as _time

        self._time = _time
        self.n_chunks = n_chunks
        self.log_prefix = log_prefix
        self.budget = (
            CHUNK_BUDGET_SECONDS if budget_seconds is None else budget_seconds
        )
        self.durations: list = []
        self._warned = False
        self._started = None

    def start(self):
        self._started = self._time.perf_counter()

    def finish(self):
        elapsed = self._time.perf_counter() - self._started
        self.durations.append(elapsed)
        done = len(self.durations)
        average = sum(self.durations) / done
        projected = average * self.n_chunks
        logging.info(
            f"{self.log_prefix} Chunk {done}/{self.n_chunks} took "
            f"{elapsed:.0f}s (avg {average:.0f}s/chunk, projected total "
            f"{projected / 60:.0f}m)."
        )
        if not self._warned and average > self.budget:
            self._warned = True
            logging.warning(
                f"{self.log_prefix} Chunk time {average:.0f}s exceeds the "
                f"{self.budget:.0f}s budget. Projected total "
                f"{projected / 60:.0f}m -- investigate before letting the run "
                f"finish."
            )


def _write_volume_reports(volume_reports, output_folder, log_prefix="[impute]"):
    """Write the fill's per-gap TRANSIENT accumulator.

    impute_volume_gap_accumulator.parquet holds one row per (household,
    column, gap) with only the state that exists nowhere in the persisted
    outputs: the pre-threshold total_final and per-gap bits (collected
    before apply_thresholds rewrites values and ORs its own flags), the
    routed/redistributed/unplaced/removed volumes from the fill's temp
    columns, and the span keys (gap_start, gap_end) the post-processor
    joins on. The FULL impute_volume_gap.parquet and its two rollups are
    assembled by the end-of-pipeline post-processing step
    (etdworkflow/postprocess_reports.py) from this accumulator plus the
    persisted flags and values -- the deliberately different name keeps any
    reader from mistaking the accumulator for the full artifact.

    Volume sums take min_count=1 throughout, so a grain with no values
    reports pd.NA rather than a zero.
    """
    non_empty = [
        frame for frame in (volume_reports or [])
        if frame is not None and not frame.empty
    ]
    if not non_empty:
        logging.info(f"{log_prefix} No gap volumes to report.")
        return
    gaps = pd.concat(non_empty, ignore_index=True)

    logging.info(
        f"{log_prefix} Saving gap volume accumulator: {len(gaps)} gap(s) over "
        f"{gaps['column'].nunique()} column(s); the full artifact and its "
        f"rollups are post-processing."
    )
    gaps.to_parquet(
        os.path.join(str(output_folder), "impute_volume_gap_accumulator.parquet"),
        engine="pyarrow",
        index=False,
    )


def _finalize_impute_outputs(
    all_gap_stats,
    total_records,
    output_folder,
    imputed_read_sql: str,
    imputed_schema_names,
    cum_cols,
    log_prefix: str = "[impute]",
    volume_reports=None,
) -> None:
    """
    Step-4+5 finalizer for the imputation shell: combine gap stats and write
    the slim household summary plus the per-gap volume artifact.

    The project summary, the counts half of the household summary and
    household_aggregated_diff are derived by the end-of-pipeline
    post-processing step (etdworkflow/postprocess_reports.py) from the
    persisted outputs. ``imputed_read_sql`` and ``imputed_schema_names``
    stay in the signature for the caller that passes them; nothing here reads
    the finished output.

    The combined gap stats stay in memory. Writing them separately would
    duplicate impute_summary_household.parquet at the same grain and on every
    shared column.
    """
    from etdtransform.vectorized_impute import methods_to_bitwise

    logging.info(f"{log_prefix} Combining gap stats and saving summaries...")
    non_empty = [g for g in all_gap_stats if not g.empty]
    if not non_empty:
        raise RuntimeError(
            f"{log_prefix} No gap stats produced across any chunk. "
            "This means no Diff columns were present after merging diffs. "
            "Ensure prepare_diff_averages_for_impute_ibis has been run before imputation."
        )
    imputation_gap_stats_df = pd.concat(non_empty, ignore_index=True)
    imputation_gap_stats_df["bitwise_methods"] = methods_to_bitwise(
        imputation_gap_stats_df["methods"]
    )

    _write_volume_reports(volume_reports, output_folder, log_prefix)

    total_records_df = pd.DataFrame(
        [
            {ids.HOUSEHOLD_BATCH: k, "total_records": v}
            for k, v in total_records.items()
        ]
    )
    # The household factor and its inputs travel with the summary: a factor that
    # scaled a fill has to be readable in the data it produced.
    _factor_columns = [
        name
        for name in (
            "house_impute_factor", "house_factor_is_fallback",
            "house_diff_median", "project_avg_median",
            "comparable_counts", "total_counts",
        )
        if name in imputation_gap_stats_df.columns
    ]
    summary_house = (
        imputation_gap_stats_df[[
            ids.PROJECT, ids.HOUSEHOLD_BATCH, "column", "diff_col_total",
            "cum_col_min_max_diff", "missing", "imputed", "imputed_na",
            "methods", "bitwise_methods",
        ] + _factor_columns]
        .merge(total_records_df, on=ids.HOUSEHOLD_BATCH)
    )
    summary_house["percentage_imputed"] = (
        summary_house["imputed"] / summary_house["total_records"] * 100
    )
    # The written file carries only what the persisted outputs cannot
    # reproduce (see _slim_summary_columns); the counts half derives exactly
    # from the persisted flags and is rejoined by the post-processing step,
    # which also writes impute_summary_project.
    summary_house[_slim_summary_columns(summary_house)].to_parquet(
        os.path.join(str(output_folder), "impute_summary_household.parquet"),
        engine="pyarrow",
    )

    # The >40% warning reads the in-memory counts, which the written file
    # does not carry.
    over_40 = summary_house[summary_house["percentage_imputed"] > 40]
    for _, row in over_40.iterrows():
        logging.warning(
            f"House {int(row[ids.HOUSEHOLD])}, Column {row['column']} has "
            f"{row['percentage_imputed']:.2f}% imputed values."
        )

    # household_aggregated_diff is a pure post-fill aggregation of the
    # finished output and is written by the end-of-pipeline post-processing
    # step (etdworkflow/postprocess_reports.py), not here, so the run makes no
    # extra pass over its own output.


def _chunk_ids_by_project(all_ids, project_of, chunk_size):
    """The household batch ids in chunks that never straddle a project.

    A chunk is a grouping for reading, not a unit of analysis -- it decides how
    many shards are held at once, not how any of their rows are imputed. But
    the fill reads a project average and reports per project, so a chunk
    holding halves of two projects holds two unrelated populations and nothing
    in the run is grouped the way the numbers are.

    The cap still applies inside each project, so a project larger than
    *chunk_size* still yields several chunks and the memory bound holds. A
    household batch whose project is unknown is kept, in a group of its own,
    rather than dropped.
    """
    groups: dict = {}
    for household_batch_id in all_ids:
        groups.setdefault(project_of.get(household_batch_id), []).append(
            household_batch_id
        )

    chunks = []
    # None last, so an unknown project never splits a real one's chunks.
    for project in sorted(groups, key=lambda p: (p is None, p)):
        household_batch_ids = sorted(groups[project])
        chunks.extend(
            household_batch_ids[i:i + chunk_size]
            for i in range(0, len(household_batch_ids), chunk_size)
        )
    return chunks


def _impute_chunk_core(
    chunk_df, cum_cols, diffs, volume_reports=None,
    constraints=None, coverage_duty=None, apparaat_sets=None,
):
    """
    Per-chunk imputation core: sort -> merge project diffs ->
    impute_and_normalize -> reconstruct cumulative columns.

    The imputation math lives here and only here, so the shell around it stays
    I/O and no second copy of the math can drift from this one.

    Returns (imputed_chunk_df, gap_stats_chunk).
    """
    from etdtransform.impute import sort_for_impute
    from etdtransform.vectorized_impute import (
        impute_and_normalize,
        without_tariff_splits,
    )

    chunk_df = sort_for_impute(chunk_df, ids.PROJECT)
    chunk_df = chunk_df.merge(diffs, on=[ids.PROJECT, "ReadingDate"], how="left")

    chunk_df, gap_stats_chunk, _ = impute_and_normalize(
        chunk_df, list(cum_cols), ids.PROJECT,
        volume_reports=volume_reports,
        constraints=constraints,
        coverage_duty=coverage_duty,
        apparaat_sets=apparaat_sets,
    )

    # Rebuild cumulative columns from imputed Diff columns per household batch.
    # chunk_df is already sorted by sort_for_impute, so cumsum is correct.
    # Tariff splits are not processed by the fill, so their delivered
    # cumulative registers must not be overwritten by a rebuilt cumsum.
    cols_to_reconstruct = [
        c for c in without_tariff_splits(cum_cols)
        if interval_column(c) in chunk_df.columns
    ]
    chunk_df = reconstruct_cumulative_columns(chunk_df, cols_to_reconstruct)
    return chunk_df, gap_stats_chunk



def _read_coverage_duty(folder):
    """The fill-order table beside avg_diffs, or None when *folder* has none."""
    path = os.path.join(str(folder), "avg_coverage_duty.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path, dtype_backend="numpy_nullable")
    return None


def _constraints_for_run(use_constraints):
    """The packaged rule table, or None for an explicit comparison run."""
    if not use_constraints:
        logging.warning(
            "Imputing WITHOUT constraints: comparison mode against "
            "pre-constraint checkpoints. Production runs narrow with the "
            "synced rules."
        )
        return None
    from etdtransform.constraints import packaged_constraints

    return packaged_constraints()


def impute_mapped_households_sharded(
    mapped_folder_path=None,
    aggregate_folder_path=None,
    cum_cols=None,
    household_batch_ids=None,
    chunk_size: int = 50,
    use_constraints: bool = True,
    apparaat_sets: dict | None = None,
) -> None:
    """
    The imputation stage (stage 3) over the sharded mapped data.

    I/O shell only -- the imputation math is _impute_chunk_core. The population
    is a set of household batches, which is the grain `Meenemen` is decided at;
    chunks are materialised with read_mapped_households (direct per-shard reads
    -- the measured-fastest strategy for household batches); the project is
    joined from the index; inclusion comes from the index at read time rather
    than from a materialised default file.

    Input read from ``aggregate_folder_path``: avg_diffs.parquet.

    Output: hive shards under
    ``<aggregate_folder_path>/sharded/household_imputed/HuisIdBSV=<n>/HuisBatchIdBSV=<p>/part.parquet``
    (the partition ids live in the path, not in the files). The writer owns and
    clears its output directory -- re-runs are idempotent.
    """
    import glob as glob_mod
    import shutil

    import pyarrow as pa
    import pyarrow.parquet as pq

    # Function-level import: load_data imports from aggregate at module level.
    from etdtransform.load_data import (
        included_household_batch_ids,
        read_mapped_households,
    )
    from etdmap.index_helpers import read_index

    if mapped_folder_path is None:
        mapped_folder_path = etdtransform.options.mapped_folder_path
    if aggregate_folder_path is None:
        raise ValueError(
            "impute_mapped_households_sharded: aggregate_folder_path is "
            "required and has no default (the config value points at the "
            "promoted production area; this writer clears its output "
            "directory before writing)."
        )

    if household_batch_ids is not None:
        all_ids = sorted(int(x) for x in household_batch_ids)
    else:
        all_ids = included_household_batch_ids(mapped_folder_path)

    index_df, _ = read_index(mapped_folder_path)
    if ids.HOUSEHOLD_BATCH not in index_df.columns:
        raise ValueError(
            "index.parquet has no household-batch column; "
            "re-run the index save to add the household-batch columns."
        )
    # Keyed on the household batch, NOT the household: a household's project
    # assignment is constant WITHIN a household batch but may differ BETWEEN
    # them (project definitions change over time; the household does not). So
    # household -> project is not a function over time, while
    # household batch -> project is. The household batch is also the primary key of
    # index.parquet (one row per household batch), so keying on the
    # household would additionally keep whichever of its rows happened to come
    # last.
    household_batch_projects = index_df[[ids.HOUSEHOLD_BATCH, ids.PROJECT]].dropna()
    project_by_household_batch = dict(zip(
        (int(x) for x in household_batch_projects[ids.HOUSEHOLD_BATCH]),
        (int(x) for x in household_batch_projects[ids.PROJECT]),
    ))

    # The household each household batch belongs to, which is the level the
    # shard path prunes on. From the index already read above, so the lookup
    # and the selection come from one registry read.
    batch_households = index_df[[ids.HOUSEHOLD_BATCH, ids.HOUSEHOLD]].dropna()
    household_of_household_batch = dict(zip(
        (int(b) for b in batch_households[ids.HOUSEHOLD_BATCH]),
        (int(h) for h in batch_households[ids.HOUSEHOLD]),
    ))
    unplaced = [b for b in all_ids if b not in household_of_household_batch]
    if unplaced:
        raise ValueError(
            f"[sharded impute] the index does not place {ids.HOUSEHOLD_BATCH} "
            f"{unplaced}, so the shards to read cannot be resolved."
        )

    chunks = _chunk_ids_by_project(all_ids, project_by_household_batch, chunk_size)
    logging.info(
        f"[sharded impute] {len(all_ids)} household batches -> {len(chunks)} "
        f"chunk(s) (cap {chunk_size}, no chunk straddles a project)"
    )

    # Estimation inputs from the SHARDED pipeline's artifact area (produced by
    # prepare_diff_averages_sharded) -- the sharded chain is self-contained under
    # <aggregate>/sharded/.
    sharded_agg = os.path.join(str(aggregate_folder_path), "sharded")
    diffs = read_diffs(sharded_agg)
    _constraints = _constraints_for_run(use_constraints)
    _coverage_duty = _read_coverage_duty(sharded_agg)

    out_root = os.path.join(str(aggregate_folder_path), "sharded", "household_imputed")
    if os.path.isdir(out_root):
        shutil.rmtree(out_root)

    all_gap_stats: list = []
    _volume_reports: list = []
    _chunk_timer = _ChunkTimer(len(chunks), "[sharded impute]")
    total_records: dict = {}
    for i, chunk_ids in enumerate(chunks):
        logging.info(
            f"[sharded impute] Chunk {i + 1}/{len(chunks)}: "
            f"{len(chunk_ids)} household batches"
        )
        _chunk_timer.start()
        # Prune the directories by household, then select the household batches
        # exactly: a household batch that is not included is not read even when
        # another of its household's is.
        chunk_households = sorted(
            {household_of_household_batch[b] for b in chunk_ids}
        )
        chunk_df = read_mapped_households(
            chunk_households,
            mapped_folder_path=mapped_folder_path,
            household_batch_ids=chunk_ids,
        )

        # Computational safety guard (independent of the index coexistence
        # guard): a household with several household batches, or with
        # overlapping periods, cannot be imputed correctly yet -- loud stop,
        # never silent blending.
        from etdtransform.impute import assert_imputation_input_batch_safe
        assert_imputation_input_batch_safe(chunk_df)

        chunk_df[ids.PROJECT] = pd.array(
            [project_by_household_batch.get(int(b)) for b in chunk_df[ids.HOUSEHOLD_BATCH]],
            dtype="Int64",
        )

        chunk_df, gap_stats_chunk = _impute_chunk_core(
            chunk_df, cum_cols, diffs, volume_reports=_volume_reports,
            constraints=_constraints, coverage_duty=_coverage_duty,
            apparaat_sets=apparaat_sets,
        )
        all_gap_stats.append(gap_stats_chunk)
        # Per household batch: the summary's percentage_imputed divides a household batch's
        # imputed count by this, so a household's two household batches must not share
        # one total.
        for batch_id, count in chunk_df.groupby(ids.HOUSEHOLD_BATCH).size().items():
            total_records[int(batch_id)] = int(count)

        for (household_id, household_batch_id), g in chunk_df.groupby(
            [ids.HOUSEHOLD, ids.HOUSEHOLD_BATCH], sort=True
        ):
            part_dir = os.path.join(
                out_root,
                f"HuisIdBSV={int(household_id)}",
                f"HuisBatchIdBSV={int(household_batch_id)}",
            )
            os.makedirs(part_dir, exist_ok=True)
            # Hive convention: the partition ids live in the path, not the file.
            g = g.drop(columns=[ids.HOUSEHOLD, ids.HOUSEHOLD_BATCH])
            pq.write_table(
                pa.Table.from_pandas(g, preserve_index=False),
                os.path.join(part_dir, "part.parquet"),
                compression="snappy",
            )
        del chunk_df
        _chunk_timer.finish()

    # Step 4+5 via the shared finalizer, into the sharded artifact area; the
    # aggregated diff reads the hive glob of the imputed shards.
    imputed_glob = os.path.join(out_root, "**", "*.parquet").replace("\\", "/")
    first_part = sorted(
        glob_mod.glob(os.path.join(out_root, "HuisIdBSV=*", "HuisBatchIdBSV=*", "*.parquet"))
    )
    schema_names = pq.read_schema(first_part[0]).names if first_part else []
    _finalize_impute_outputs(
        all_gap_stats,
        total_records,
        output_folder=sharded_agg,
        imputed_read_sql=(
            f"read_parquet('{imputed_glob}', hive_partitioning=true, union_by_name=true)"
        ),
        imputed_schema_names=schema_names,
        cum_cols=cum_cols,
        log_prefix="[sharded impute]",
        volume_reports=_volume_reports,
    )
    logging.info(f"[sharded impute] Done -> {out_root}")


def _write_derive_provenance(output_dir, level, rows, catalog_hash):
    """Persist which catalog equation produced each 'derive' (ratio) variable at
    each resample/aggregate scope, to derive_provenance_{level}.parquet
    (columns: scope, variable, rhs_text, rhs_vars, catalog_hash)."""
    if not rows:
        return
    df = pd.DataFrame(
        rows, columns=["scope", "variable", "rhs_text", "rhs_vars"]
    ).astype("string")
    df["catalog_hash"] = pd.array([catalog_hash] * len(df), dtype="string")
    path = os.path.join(output_dir, f"derive_provenance_{level}.parquet")
    df.to_parquet(path)
    logging.info(
        f"[derive] provenance: {len(df)} rows -> {os.path.basename(path)} "
        f"(catalog {catalog_hash})"
    )




def read_aggregate(name, interval):
    """
    Read an aggregate parquet file.

    Parameters
    ----------
    name : str
        The name of the aggregate
    interval : str
        The time interval of the aggregate

    Returns
    -------
    pd.DataFrame
        The aggregate data

    Notes
    -----
    This function reads a parquet file based on the provided name and interval.
    """
    safe_name = re.sub(r"\W+", "_", name.lower())
    from etdmap.storage import read_source_frame, resolve_source_path
    return read_source_frame(resolve_source_path(
        etdtransform.options.aggregate_folder_path, f"{safe_name}_{interval}"
    ))


def get_aggregate_table(name, interval):
    """
    Get an aggregate table as an ibis table.

    Parameters
    ----------
    name : str
        The name of the aggregate
    interval : str
        The time interval of the aggregate

    Returns
    -------
    ibis.Table
        The aggregate data as an ibis table

    Notes
    -----
    This function reads a parquet file and returns it as an ibis table.
    """
    safe_name = re.sub(r"\W+", "_", name.lower())
    from etdmap.storage import resolve_source_path
    from etdtransform.load_data import source_table
    return source_table(resolve_source_path(
        etdtransform.options.aggregate_folder_path, f"{safe_name}_{interval}"
    ))


def _sql_path(path) -> str:
    """Return path as a single-quoted SQL string literal with forward slashes."""
    return "'" + str(path).replace("\\", "/").replace("'", "''") + "'"


def _resample_sql_expr(col: str, method: str, min_count: int) -> str:
    """Return the DuckDB SQL expression for resampling a single column.

    Column names are double-quoted so hyphens (e.g. 'Zon-opwekTotaal') are
    not parsed as arithmetic by DuckDB. All aggregate operands are cast to
    DOUBLE so boolean columns (e.g. VentilatieFiltermelding) produce numeric
    output (True -> 1.0, False -> 0.0) rather than raising a type error.
    """
    qcol = f'"{col}"'
    num = f'CAST("{col}" AS DOUBLE)'
    if method == "sum":
        return f"CASE WHEN COUNT({qcol}) >= {min_count} THEN SUM({num}) ELSE NULL END AS {qcol}"
    if method == "max":
        return f"CASE WHEN COUNT({qcol}) >= {min_count} THEN MAX({num}) ELSE NULL END AS {qcol}"
    if method == "avg":
        return (
            f"CASE WHEN COUNT({qcol}) >= {min_count}"
            f" THEN SUM({num}) / COUNT({qcol}) ELSE NULL END AS {qcol}"
        )
    raise ValueError(
        f"Unknown resample_method '{method}' for column '{col}' -- "
        f"check the data model."
    )


def _parse_interval(interval: str):
    """Return (bucket_minutes, duckdb_interval_expr) for an interval string.

    Accepts Nmin (e.g. "5min", "15min", "60min") or Nh (e.g. "6h", "24h").
    """
    import re
    m = re.fullmatch(r"(\d+)min", interval)
    if m:
        mins = int(m.group(1))
        return mins, f"INTERVAL '{mins} minutes'"
    m = re.fullmatch(r"(\d+)h", interval)
    if m:
        hours = int(m.group(1))
        return hours * 60, f"INTERVAL '{hours} hours'"
    raise ValueError(f"Unsupported interval '{interval}'. Expected Nmin or Nh.")


def _finite_or_null(col_expr):
    """Map +/-inf and NaN in an ibis column to NULL.

    A 'derive' ratio (e.g. ZelfgebruikPercentage = Zelfgebruik / ZonopwekBruto)
    hits a zero denominator whenever there is no production: sympy_to_ibis
    fill_null(0)'s the inputs, so a night interval becomes 0 / 0 = NaN and a
    spurious non-zero numerator over zero production becomes +/-inf. Neither is a
    real value -- collapse both to NULL so the result is pd.NA (ADR-005), not a
    bogus number.
    """
    inf = ibis.literal(float("inf"))
    bad = (
        col_expr.isnull()
        | col_expr.isnan()
        | (col_expr == inf)
        | (col_expr == -inf)
    )
    return bad.ifelse(ibis.null(), col_expr)


def _derive_columns_in_parquet(path, derive_cols, catalog_df=None):
    """Recompute 'derive' columns in-place from their catalog rules.

    A 'derive' column is never summed or averaged as a stored value -- it is an
    intensive quantity (a ratio) that must be recomputed from its components
    AFTER they have been aggregated to the target resolution. This reads the
    just-written parquet, drops any stale copy of each derive column, derives it
    via the catalog (the same machinery as the calculated stage), masks
    non-finite results to NULL, and atomically replaces the file.

    Missing components are tolerated (skip-and-log): a derive column whose
    inputs are absent at this resolution -- e.g. a non-PV project without
    ZonopwekBruto -- is logged and left out rather than failing the pass.

    Parameters
    ----------
    path : str
        Parquet file to rewrite in place (a household_{interval} or
        project_{interval} file).
    derive_cols : list[str]
        Columns flagged with method 'derive' that are present in this file.
    catalog_df : pd.DataFrame, optional
        Loaded catalog; loaded via load_catalog() when None.
    """
    from etdmap.catalog import load_catalog
    from etdtransform.catalog.query import DatasetAdapter
    from etdtransform.calculated_columns import sympy_to_ibis

    if not derive_cols:
        return []
    if catalog_df is None:
        catalog_df = load_catalog()

    con = ibis.duckdb.connect()
    derived_now = []
    prov_rows = []  # {variable, rhs_text, rhs_vars} actually used
    tmp_path = None
    try:
        tbl = con.read_parquet(path)
        present = set(tbl.columns)
        targets = set(derive_cols)
        # Force recompute: treat the derive targets as NOT available so the
        # planner derives them from their components, and drop any stale copy
        # carried in from a finer resolution.
        available = present - targets
        stale = [c for c in derive_cols if c in present]
        if stale:
            tbl = tbl.drop(*stale)

        adapter = DatasetAdapter(catalog_df)
        plan, not_derivable = adapter.execution_plan_for_available(available, targets)

        if not_derivable:
            logging.warning(
                f"[derive] {os.path.basename(path)}: cannot derive "
                f"{sorted(not_derivable)} -- required components missing at this "
                f"resolution; leaving column(s) out (skip-and-log)."
            )

        for col, rhs_expr in plan:
            tbl = tbl.mutate(**{col: sympy_to_ibis(rhs_expr, tbl)})
            derived_now.append(col)
            prov_rows.append({
                "variable": col,
                "rhs_text": str(rhs_expr),
                "rhs_vars": ",".join(sorted(str(s) for s in rhs_expr.free_symbols)),
            })

        if derived_now:
            tbl = tbl.mutate(
                **{c: _finite_or_null(tbl[c]) for c in derived_now}
            )
            tmp_path = path + ".derivetmp.parquet"
            tbl.to_parquet(tmp_path)
    finally:
        con.disconnect()

    if tmp_path is not None:
        os.replace(tmp_path, path)
        logging.info(
            f"[derive] {os.path.basename(path)}: derived {sorted(derived_now)}"
        )
    return prov_rows


def resample_hh_data_duckdb(
    source_path: str,
    output_dir: str,
    intervals: tuple = ("60min", "15min", "5min"),
) -> None:
    """
    Resample the calculated household artifact to multiple time intervals using
    DuckDB.

    Reads resampling config from etdmap.data_model.get_aggregation_config() (ADR-004).
    Only processes columns present in the source parquet (guards against columns not
    yet computed for this run).

    For 5min: column selection only -- no aggregation.
    For 15min / 60min: TIME_BUCKET groupby for Diff/total columns, then window
    cumsum over the household batch and its project to reconstruct cumulative
    counterparts for every Diff column in the config.

    Type validation: every aggregation column must be float, int, or boolean
    (boolean is cast to 0.0/1.0 in SUM/AVG SQL). Any other type triggers a
    ValueError listing all offending columns -- per ADR-003, data-correctness
    errors fail loudly. Errors are also logged individually so the operator
    sees the full picture in one run.

    Parameters
    ----------
    source_path : str
        The calculated household artifact: a hive shard directory, or a single
        parquet file. Reads are format-detected.
    output_dir : str
        Directory under which a ``household_{interval}`` hive shard directory
        is written per interval, partitioned on (ids.HOUSEHOLD, ids.HOUSEHOLD_BATCH).
        Each is cleared before writing, so re-runs are idempotent.
    intervals : tuple
        Intervals to produce. Supported formats: Nmin (e.g. "5min", "15min", "60min")
        or Nh (e.g. "6h", "24h"). The "5min" interval is a column-selection passthrough
        with no aggregation; all others use TIME_BUCKET grouping.

    Raises
    ------
    ValueError
        If any column in the active aggregation config has an unsupported type
        (anything other than float, int, or boolean).
    """
    import duckdb
    import shutil
    import pyarrow as pa
    import pyarrow.parquet as pq
    from etdmap.data_model import get_aggregation_config

    config = get_aggregation_config()

    # Source is format-detected (file or hive shard directory) by the shared
    # read layer, which unions per-supplier schemas by name: suppliers deliver
    # different column sets, so a positional union would misalign them. The SQL
    # math below is identical for both layouts.
    from etdmap.storage import (
        source_is_sharded,
        source_schema_names as _schema_names_of,
        source_schema_types,
        source_sql,
    )
    from etdtransform.load_data import source_table

    source_is_shards = source_is_sharded(source_path)
    source_schema_names = _schema_names_of(source_path)
    _schema_types = source_schema_types(source_path)
    _src_sql = source_sql(source_path)
    if source_is_shards:
        # Batch-safety guard at this compute entry point: a timestamp twice
        # in one household batch raises.
        from etdtransform.impute import _assert_batch_safe_ibis
        _assert_batch_safe_ibis(source_table(source_path))

    if ids.HOUSEHOLD_BATCH not in source_schema_names:
        raise ValueError(
            "resample_hh_data_duckdb: the output partitions on HuisBatchIdBSV "
            "and the source has no such column. A household artifact always "
            "carries the household batch it was mapped in."
        )

    active_config = {
        col: cfg for col, cfg in config.items() if col in source_schema_names
    }

    # 'derive' columns are recomputed from their catalog rule AFTER the
    # components are resampled -- they must NOT go through the SUM/AVG/MAX
    # groupby. Load the catalog once if any are present.
    derive_cols = [
        col for col, cfg in active_config.items()
        if cfg["resample_method"] == "derive"
    ]
    catalog_df = None
    _derive_prov: list = []
    if derive_cols:
        from etdmap.catalog import load_catalog
        catalog_df = load_catalog()

    # Validate that every aggregation column has a supported type. The SUM/AVG
    # SQL casts each column to DOUBLE; that cast is meaningful only for
    # floating, integer, and boolean (boolean -> 0.0/1.0). Any other type
    # (string, timestamp, decimal, list, struct, ...) either errors at SQL
    # execution or produces nonsense, so we refuse to proceed and let the
    # operator either fix the source data or set AggregatieMeenemen=False in
    # etdmodel.csv. ADR-003: hard fail on data-correctness errors.
    type_errors: list[tuple[str, str]] = []
    for col in active_config:
        col_type = _schema_types.get(col)
        if col_type is None:
            continue
        if pa.types.is_floating(col_type) or pa.types.is_integer(col_type):
            continue
        if pa.types.is_boolean(col_type):
            continue  # intentional: boolean -> 0.0/1.0 in SUM/AVG SQL
        logging.error(
            f"resample_hh_data_duckdb: column '{col}' has unsupported type "
            f"'{col_type}' for aggregation."
        )
        type_errors.append((col, str(col_type)))
    if type_errors:
        raise ValueError(
            f"resample_hh_data_duckdb: {len(type_errors)} column(s) have "
            f"unsupported types for aggregation. Supported types are float, "
            f"int, and boolean. Either fix the source data types, or set "
            f"AggregatieMeenemen=False in etdmodel.csv. Offending columns: "
            + ", ".join(f"{c} ({t})" for c, t in type_errors)
        )

    # Diff columns whose cumulative counterpart should be rebuilt.
    diff_cols = [col for col in active_config if cumulative_column(col)]
    cumul_cols = [cumulative_column(col) for col in diff_cols]

    # The household batch id rides along in the grouping: it is a partition key
    # of the output, so it has to survive the GROUP BY. It does not change the
    # grain, which is already per household batch.
    _batch_id = [ids.HOUSEHOLD_BATCH]
    id_cols = [ids.HOUSEHOLD] + _batch_id + [ids.PROJECT, "ReadingDate"]
    _group_ids_sql = ", ".join([ids.HOUSEHOLD] + _batch_id + [ids.PROJECT])

    for interval in intervals:
        out_target = os.path.join(output_dir, f"household_{interval}")
        # The writer owns and clears its output directory (idempotent
        # re-runs; same convention as the other sharded writers).
        if os.path.isdir(out_target):
            shutil.rmtree(out_target)
        os.makedirs(out_target, exist_ok=True)
        _copy_opts = (
            "FORMAT PARQUET, COMPRESSION SNAPPY, "
            "PARTITION_BY (HuisIdBSV, HuisBatchIdBSV), OVERWRITE_OR_IGNORE TRUE"
        )
        logging.info(f"resample_hh_data_duckdb: writing {interval} -> {out_target}")

        if interval == "5min":
            # Column selection only -- include config columns + existing cumulative counterparts.
            select_cols = id_cols + list(active_config.keys())
            for c in cumul_cols:
                if c in source_schema_names and c not in select_cols:
                    select_cols.append(c)
            # Double-quote identifiers so hyphenated names (e.g. "Zon-opwekTotaal") are
            # not parsed as arithmetic expressions by DuckDB.
            quoted = ", ".join(f'"{c}"' for c in select_cols)
            sql = (
                f"COPY (SELECT {quoted} FROM {_src_sql})"
                f" TO {_sql_path(out_target)} ({_copy_opts})"
            )
            with duckdb.connect() as con:
                con.execute(sql)

        else:
            bucket_minutes, bucket_expr = _parse_interval(interval)
            min_count = bucket_minutes // 5

            resample_exprs = [
                _resample_sql_expr(col, cfg["resample_method"], min_count)
                for col, cfg in active_config.items()
                if cfg["resample_method"] != "derive"
            ]
            resample_clause = ",\n            ".join(resample_exprs)

            cumsum_exprs = [
                f'SUM("{diff_col}") OVER ('
                f"PARTITION BY HuisBatchIdBSV, ProjectIdBSV "
                f"ORDER BY ReadingDate "
                f"ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
                f') AS "{base_col}"'
                for diff_col, base_col in zip(diff_cols, cumul_cols)
            ]
            cumsum_clause = (",\n        " + ",\n        ".join(cumsum_exprs)) if cumsum_exprs else ""

            sql = f"""
            COPY (
                WITH resampled AS (
                    SELECT
                        {_group_ids_sql},
                        TIME_BUCKET({bucket_expr}, ReadingDate) AS ReadingDate,
                        {resample_clause}
                    FROM {_src_sql}
                    GROUP BY {_group_ids_sql},
                             TIME_BUCKET({bucket_expr}, ReadingDate)
                )
                SELECT *{cumsum_clause}
                FROM resampled
                ORDER BY HuisIdBSV, ProjectIdBSV, ReadingDate
            ) TO {_sql_path(out_target)} ({_copy_opts})
            """
            with duckdb.connect() as con:
                con.execute(sql)

            # Recompute derive columns (ratios) from their now-resampled
            # components. The 5min branch carries them as-is (already correct
            # per-interval from the calculated stage); only the grouping
            # intervals need re-derivation. The derivation runs in place per
            # shard file -- row-wise ratios, so identical math whatever the
            # shard split is -- and provenance rows are deduplicated across
            # shards.
            if derive_cols:
                import glob as glob_mod
                _seen_prov = set()
                for _shard in sorted(glob_mod.glob(
                    os.path.join(out_target, "**", "*.parquet"), recursive=True
                )):
                    for r in _derive_columns_in_parquet(_shard, derive_cols, catalog_df):
                        _key = tuple(sorted((k, str(v)) for k, v in r.items()))
                        if _key not in _seen_prov:
                            _seen_prov.add(_key)
                            _derive_prov.append(
                                {"scope": f"household_{interval}", **r}
                            )

        logging.info(f"resample_hh_data_duckdb: done {interval}")

    if _derive_prov:
        _write_derive_provenance(
            output_dir, "household", _derive_prov, _catalog_hash(catalog_df)
        )


def aggregate_project_data_duckdb(
    intervals: tuple = ("5min", "15min", "60min"),
    aggregate_folder_path=None,
) -> None:
    """
    Aggregate resampled household parquet files to project level using DuckDB.

    Reads aggregation config from etdmap.data_model.get_aggregation_config() (ADR-004).
    One DuckDB pass per interval. Produces project_{interval}.parquet in
    etdtransform.options.aggregate_folder_path.

    Cumulative counterparts of Diff columns are rebuilt via window cumsum
    (partitioned by project, ordered by ReadingDate) after the GROUP BY step,
    matching the convention in resample_hh_data_duckdb.

    Type validation: every aggregation column must be float, int, or boolean
    (same rule as resample_hh_data_duckdb). Any other type triggers a
    ValueError listing all offending columns for the current interval -- per
    ADR-003, data-correctness errors fail loudly.

    Parameters
    ----------
    intervals : tuple
        Intervals to aggregate; each element must match a household_{interval}.parquet
        file produced by a prior resample step.

    Raises
    ------
    ValueError
        If any column in the active aggregation config (per interval) has an
        unsupported type (anything other than float, int, or boolean).
    """
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq
    from etdmap.data_model import get_aggregation_config

    config = get_aggregation_config()
    output_dir = (
        aggregate_folder_path
        if aggregate_folder_path is not None
        else etdtransform.options.aggregate_folder_path
    )

    # Load the catalog once if any variable is recomputed post-aggregation.
    catalog_df = None
    _derive_prov: list = []
    if any(cfg["aggregate_method"] == "derive" for cfg in config.values()):
        from etdmap.catalog import load_catalog
        catalog_df = load_catalog()

    # Diff columns whose cumulative counterpart should be rebuilt.
    diff_cols_all = [col for col in config if col.endswith("Diff")]
    cumul_cols_all = [col[:-4] for col in diff_cols_all]

    for interval in intervals:
        out_path = os.path.join(output_dir, f"project_{interval}.parquet")
        logging.info(f"aggregate_project_data_duckdb: writing {interval} -> {out_path}")

        # Source is format-detected (file or hive shard directory) by the
        # shared read layer. Project output stays a single parquet file in
        # either case (project-level data is small).
        from etdmap.storage import (
            resolve_source_path,
            source_is_sharded,
            source_schema_names as _schema_names_of,
            source_schema_types,
            source_sql,
        )
        from etdtransform.load_data import source_table

        source_path = resolve_source_path(output_dir, f"household_{interval}")
        source_schema_names = _schema_names_of(source_path)
        _schema_types = source_schema_types(source_path)
        _src_sql = source_sql(source_path)
        if source_is_sharded(source_path):
            # Batch-safety guard: a timestamp twice in one household batch
            # would double-count project sums.
            from etdtransform.impute import _assert_batch_safe_ibis
            _assert_batch_safe_ibis(source_table(source_path))
        active_config = {col: cfg for col, cfg in config.items() if col in source_schema_names}

        # Validate types -- same rule as resample_hh_data_duckdb. Float, int,
        # and boolean are aggregable; anything else is a data-correctness
        # error and is refused per ADR-003.
        type_errors: list[tuple[str, str]] = []
        for col in active_config:
            col_type = _schema_types.get(col)
            if col_type is None:
                continue
            if pa.types.is_floating(col_type) or pa.types.is_integer(col_type):
                continue
            if pa.types.is_boolean(col_type):
                continue  # intentional: boolean -> 0.0/1.0 in SUM/AVG SQL
            logging.error(
                f"aggregate_project_data_duckdb [{interval}]: column '{col}' has "
                f"unsupported type '{col_type}' for aggregation."
            )
            type_errors.append((col, str(col_type)))
        if type_errors:
            raise ValueError(
                f"aggregate_project_data_duckdb [{interval}]: {len(type_errors)} "
                f"column(s) have unsupported types for aggregation. Supported "
                f"types are float, int, and boolean. Either fix the source data "
                f"types, or set AggregatieMeenemen=False in etdmodel.csv. "
                f"Offending columns: "
                + ", ".join(f"{c} ({t})" for c, t in type_errors)
            )

        diff_cols = [col for col in active_config if col.endswith("Diff")]
        cumul_cols = [col[:-4] for col in diff_cols]

        derive_cols = [
            col for col, cfg in active_config.items()
            if cfg["aggregate_method"] == "derive"
        ]
        agg_exprs = []
        for col, cfg in active_config.items():
            method = cfg["aggregate_method"]
            # CAST to DOUBLE handles boolean columns (True/False -> 1.0/0.0).
            if method == "derive":
                continue  # recomputed post-aggregation from its catalog rule
            if method == "avg":
                agg_exprs.append(f'AVG(CAST("{col}" AS DOUBLE)) AS "{col}"')
            elif method == "sum":
                agg_exprs.append(f'SUM(CAST("{col}" AS DOUBLE)) AS "{col}"')
            else:
                raise ValueError(
                    f"Unknown aggregate_method '{method}' for column '{col}' -- "
                    f"check the data model."
                )

        agg_clause = ",\n            ".join(agg_exprs)

        cumsum_exprs = [
            f'SUM("{diff_col}") OVER ('
            f"PARTITION BY ProjectIdBSV "
            f"ORDER BY ReadingDate "
            f"ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
            f') AS "{base_col}"'
            for diff_col, base_col in zip(diff_cols, cumul_cols)
        ]
        cumsum_clause = (",\n        " + ",\n        ".join(cumsum_exprs)) if cumsum_exprs else ""

        sql = f"""
        COPY (
            WITH aggregated AS (
                SELECT
                    ProjectIdBSV,
                    ReadingDate,
                    {agg_clause}
                FROM {_src_sql}
                GROUP BY ProjectIdBSV, ReadingDate
            )
            SELECT *{cumsum_clause}
            FROM aggregated
            ORDER BY ProjectIdBSV, ReadingDate
        ) TO {_sql_path(out_path)} (FORMAT PARQUET, COMPRESSION SNAPPY)
        """
        with duckdb.connect() as con:
            con.execute(sql)

        # Recompute derive columns (ratios) from their aggregated components.
        # Unlike the household resample, this applies at EVERY interval (incl.
        # 5min): a project ratio must be sum(numerator)/sum(denominator) over
        # the project's household batches, never the mean of their individual ratios.
        if derive_cols:
            _rows = _derive_columns_in_parquet(out_path, derive_cols, catalog_df)
            _derive_prov.extend(
                {"scope": f"project_{interval}", **r} for r in _rows
            )

        logging.info(f"aggregate_project_data_duckdb: done {interval}")

    if _derive_prov:
        _write_derive_provenance(
            output_dir, "project", _derive_prov, _catalog_hash(catalog_df)
        )


# def aggregate_weerstation_data(index_df):
#     group_column = ['Weerstation']
#     intervals = ['5min', '15min', '60min']
#     for interval in intervals:
#         logging.info(f'-- Starting {group_column} aggregation with {interval} intervals --')
#         df = read_hh_data(interval = interval, metadata_columns = ['Weerstation'])
#         aggregate_and_save(df, group_column, interval=interval)

