"""Flags the calculated stage writes, and the threshold scan behind them.

Three mechanisms here write a calculated value, and each names itself:
``CALCULATED_COLUMN`` for deriving a whole column for a household group,
``CALCULATED_FROM_ROW_FORMULA`` for filling one empty cell from a formula
chosen for that row, and ``CALCULATED_FROM_RULE_BOUND`` for writing the single
value the rules pin a row to. ``CALCULATED_VALUE`` matches any of the three.
The two ``CALCULATED_*`` violation flags mark a value as outside a bound. All
of them go into the same impute-type column the fill writes, so one decode
reads both stages.

Why a flag and not a sidecar. Which columns a household gets derived is
decided from what that household delivers, so two households in the same
project can have different derived columns. The derivation sidecars record
that per household; they cannot answer "is THIS cell measured or derived",
which is what filtering needs.

Why the scan has to be here. The threshold columns (``validate_<column>``) are
written before this stage runs, so a derived column's threshold column exists
and is empty -- nothing had the values to check. Nothing in the pipeline
compares a derived value against its declared minimum and maximum.
"""

import logging
from contextlib import contextmanager

import pandas as pd

from etdtransform.flags import ImputeType

#: Bound sources, and the flag each one raises. The thresholds table gives one
#: bound per column, scaled to the grid. The rules narrow a row using the other
#: values known in it, which is a different claim about the same value and so a
#: different flag.
THRESHOLD_SOURCE = "column threshold, scaled to the grid"
RULE_SOURCE = "rule bound from known values"


def grid_scaled_bounds(column: str, grid_minutes) -> tuple:
    """The declared threshold for *column*, scaled to the dataset grid.

    A calculated column is judged at the GRID cadence, never at a delivered
    column's own cadence and never at the threshold's declared unit. The
    derivation runs after imputation, by which point every variable it reads
    sits on the grid, so one grid interval is what a calculated cell covers --
    for every household, which is why there is one bound and not one per
    household. The cadence table does not enter into it: it records the
    interval a column's READINGS arrived at, which the fill has already
    resolved away.
    """
    from etdmap.data_model import load_thresholds_as_dict, scale_bounds

    return scale_bounds(load_thresholds_as_dict().get(column, {}), grid_minutes)


def impute_type_column(column: str) -> str:
    """The impute-type column that travels beside *column*."""
    return f"{column}_impute_type"


def scan_calculated_columns(
    df: pd.DataFrame,
    derived_columns,
    grid_minutes=None,
    package=None,
    mapped_folder_path=None,
) -> pd.DataFrame:
    """Mark every derived cell, and flag the ones outside their threshold.

    Writes three marks per derived column:

    - ``CALCULATED_COLUMN`` on every cell holding a value;
    - ``CALCULATED_THRESHOLD_VIOLATED`` where the value is outside the
      column's declared bound scaled to the GRID;
    - ``CALCULATED_OUTSIDE_RULE_BOUND`` where it is outside the bound the
      rules narrow its row to from the other values known in that row.

    A missing value is marked none of them: it was not calculated and cannot
    be out of range. Bits are OR-ed into any existing impute-type column,
    never assigned over it, so a cell the fill already described keeps that
    description.

    The two violation flags are separate because they are different claims.
    The threshold says the value is impossible for the variable; the rule
    bound says it contradicts the rest of its own row.

    A fourth mark, ``CALCULATED_RULE_BOUND_FROM_SOFT_KNOWNS``, says the rule bound
    rested partly on cells the fill wrote. It comes free: the narrowing
    already computes it, so no extra pass and no comparison across columns. A
    rule violation WITHOUT it stands on measurements alone and is the stronger
    signal -- the row contradicts itself using nothing but delivered readings,
    which cannot be explained away by the fill.

    Parameters
    ----------
    df : pd.DataFrame
        The calculated frame.
    derived_columns : iterable of str
        The columns this run derived. The caller knows them from the
        derivation plan; deriving them again here would be a second copy.
    grid_minutes : int, float or None
        The dataset grid. Read from *mapped_folder_path* when None.
    package : constraint package or None
        From :func:`etdtransform.constraints.build_constraint_package` at the
        grid cadence. When None the rule-bound flag is skipped and only the
        threshold flag is written, because a bound nobody computed is not the
        same as a bound nothing violates.
    mapped_folder_path : str or None
        Where to read the index from, when *grid_minutes* is None.

    Returns
    -------
    pd.DataFrame
        *df*, with the impute-type columns created or updated in place.
    """
    from etdmap.flags import MappingFlag, bound_flags

    derived_columns = [c for c in derived_columns if c in df.columns]
    if not derived_columns:
        logging.info("[calc flags] no derived columns in this frame")
        return df

    if grid_minutes is None:
        from etdtransform.bounds import dataset_grid_minutes

        grid_minutes = dataset_grid_minutes(mapped_folder_path)

    out_of_bounds = (
        MappingFlag.OUT_OF_BOUNDS_LOW.value | MappingFlag.OUT_OF_BOUNDS_HIGH.value
    )
    imputed = imputed_masks(df) if package is not None else None
    marked = violations = rule_violations = hard_violations = soft_bounds = 0
    for column in derived_columns:
        flag_column = impute_type_column(column)
        if flag_column not in df.columns:
            df[flag_column] = pd.array([0] * len(df), dtype="Int64")
        existing = df[flag_column].fillna(0).astype("Int64")

        present = df[column].notna()
        bits = present.astype("Int64") * ImputeType.CALCULATED_COLUMN.value

        lower, upper = grid_scaled_bounds(column, grid_minutes)
        outside = bound_flags(df[column], lower_bound=lower, upper_bound=upper)
        out_of_range = ((outside & out_of_bounds) != 0).fillna(False)
        bits = bits | (
            out_of_range.astype("Int64") * ImputeType.CALCULATED_THRESHOLD_VIOLATED.value
        )

        if package is not None:
            rule_lo, rule_hi, from_imputed = _rule_bounds(
                df, column, package, imputed=imputed
            )
            outside_rule = bound_flags(
                df[column], lower_bound=rule_lo, upper_bound=rule_hi
            )
            broke_rule = ((outside_rule & out_of_bounds) != 0).fillna(False)
            bits = bits | (
                broke_rule.astype("Int64")
                * ImputeType.CALCULATED_OUTSIDE_RULE_BOUND.value
            )
            # Recorded on every cell whose bound rests on a filled value, as
            # BOUND_FROM_IMPUTED is in the fill, so the two read the same way.
            # A violation WITHOUT this flag is one that stands on measurements
            # alone -- a bit test on one column, not a comparison across them.
            soft = pd.Series(from_imputed, index=df.index).fillna(False) & present
            bits = bits | (
                soft.astype("Int64")
                * ImputeType.CALCULATED_RULE_BOUND_FROM_SOFT_KNOWNS.value
            )
            rule_violations += int(broke_rule.sum())
            hard_violations += int((broke_rule & ~soft).sum())
            soft_bounds += int(soft.sum())

        df[flag_column] = existing | bits
        marked += int(present.sum())
        violations += int(out_of_range.sum())

    if package is None:
        rule_report = "not computed (no constraint package)"
    else:
        rule_report = (
            f"{rule_violations:,} ({hard_violations:,} on measurements alone, "
            f"{rule_violations - hard_violations:,} where the bound rests on a "
            f"filled cell); bound rests on a filled cell for {soft_bounds:,}"
        )
    logging.info(
        "[calc flags] grid %s min; CALCULATED_COLUMN on %s cell(s) across %d "
        "column(s); CALCULATED_THRESHOLD_VIOLATED on %s; "
        "CALCULATED_OUTSIDE_RULE_BOUND on %s",
        grid_minutes, f"{marked:,}", len(derived_columns),
        f"{violations:,}", rule_report,
    )
    return df


def imputed_masks(df: pd.DataFrame) -> dict:
    """Which cells the fill wrote, per column, from the ``_is_imputed`` columns.

    ``narrow_bounds_by_row`` needs this to report whether a row's bound was
    tightened using a filled cell. Without it every bound reads as resting on
    measurements alone, which is the stronger claim and the wrong one.
    """
    suffix = "_is_imputed"
    return {
        column[: -len(suffix)]: df[column].fillna(False).astype(bool)
        for column in df.columns
        if column.endswith(suffix)
    }


def _rule_bounds(df: pd.DataFrame, column: str, package, imputed=None):
    """The per-row rule bound for *column*, as two Float64 Series plus a mask.

    A thin seam over ``narrow_bounds_by_row`` so the scan does not depend on
    that module's import at call time, and so the bound can be supplied
    pre-computed by a caller that already has it.
    """
    from etdtransform.constraints import narrow_bounds_by_row

    return narrow_bounds_by_row(df, column, package, imputed=imputed)


def calculated_flag_expressions(table, derived_columns, grid_minutes, undefined=None):
    """``CALCULATED_COLUMN`` and the THRESHOLD flag as ibis expressions.

    For the ibis derivation path, which never materialises the frame. Returns
    ``{impute_type_column: expression}`` for the caller to pass to ``mutate``,
    OR-ed onto an existing impute-type column where one is present.

    One bound per column serves every household, because a calculated column
    is judged at the grid and the grid is the same for all of them.

    ``CALCULATED_OUTSIDE_RULE_BOUND`` is NOT written here: the rule bound is a
    per-row intersection over substituted values, which is numpy work on a
    materialised frame. Use :func:`scan_calculated_columns` with a package for
    that.
    """
    import ibis

    expressions = {}
    for column in derived_columns:
        if column not in table.columns:
            continue
        lower, upper = grid_scaled_bounds(column, grid_minutes)

        present = table[column].notnull()
        bits = present.cast("int64") * ImputeType.CALCULATED_COLUMN.value

        outside = ibis.literal(False)
        if pd.notna(lower):
            outside = outside | (present & (table[column] < float(lower)))
        if pd.notna(upper):
            outside = outside | (present & (table[column] > float(upper)))
        bits = bits | (
            outside.cast("int64") * ImputeType.CALCULATED_THRESHOLD_VIOLATED.value
        )

        # An undefined result is not a missing input. The cell holds pd.NA
        # either way, so without this mark the two are indistinguishable --
        # and only one of them means the formula does not apply to the row.
        condition = (undefined or {}).get(column)
        if condition is not None:
            bits = bits | (
                condition.cast("int64") * ImputeType.CALCULATED_UNDEFINED.value
            )

        flag_column = impute_type_column(column)
        if flag_column in table.columns:
            bits = table[flag_column].fill_null(0) | bits
        expressions[flag_column] = bits
    return expressions


def threshold_violation_summary(df: pd.DataFrame, derived_columns) -> pd.DataFrame:
    """Per column and household, how many derived cells break their threshold.

    Reads the flags :func:`scan_calculated_columns` wrote rather than
    recomputing the bounds, so the summary cannot disagree with the data.
    """
    rows = []
    sources = (
        (THRESHOLD_SOURCE, ImputeType.CALCULATED_THRESHOLD_VIOLATED.value, None),
        (RULE_SOURCE, ImputeType.CALCULATED_OUTSIDE_RULE_BOUND.value,
         ImputeType.CALCULATED_RULE_BOUND_FROM_SOFT_KNOWNS.value),
    )
    for column in derived_columns:
        flag_column = impute_type_column(column)
        if flag_column not in df.columns:
            continue
        flags = df[flag_column].fillna(0).astype("Int64")
        for source, bit, soft_bit in sources:
            hit = (flags & bit) != 0
            if not bool(hit.any()):
                continue
            # Split the rule-bound rows by whether the bound stood on
            # measurements alone: the two are different findings and summing
            # them would hide the stronger one.
            if soft_bit is None:
                parts = [("", hit)]
            else:
                soft = (flags & soft_bit) != 0
                parts = [
                    (" (measurements alone)", hit & ~soft),
                    (" (bound rests on a filled cell)", hit & soft),
                ]
            for label, mask in parts:
                if not bool(mask.any()):
                    continue
                frame = df.loc[mask, ["HuisIdBSV", column]]
                for household, group in frame.groupby("HuisIdBSV"):
                    rows.append({
                        "column": column,
                        "HuisIdBSV": household,
                        "cells": len(group),
                        "worst": group[column].min(),
                        "highest": group[column].max(),
                        "source": source + label,
                    })
    return pd.DataFrame(rows)


def constraint_package_for_grid(grid_minutes, apparaat_set=""):
    """The constraint package to pass to :func:`scan_calculated_columns`.

    Built at the GRID cadence, for the same reason the thresholds are scaled
    to it: a calculated cell covers one grid interval. Built once per
    apparaat-set, because the presolve is the expensive half and it does not
    depend on the column or the household.

    Selection is ``build_constraint_package``'s own job -- it takes
    ``apparaat_set`` and refuses a key the artifact does not hold. This is a
    two-line composition over the library and must not grow a second copy of
    that logic.
    """
    from etdtransform.constraints import build_constraint_package, packaged_constraints

    return build_constraint_package(
        packaged_constraints(),
        cadence_minutes=grid_minutes,
        apparaat_set=apparaat_set,
    )

#: Widest bound still treated as pinning a single value. The rules are
#: evaluated in floating point, so an exactly-determined cell can land a few
#: ulps apart rather than identical.
POINT_BOUND_TOLERANCE = 1e-9


def derive_from_rule_bound(df: pd.DataFrame, targets, package,
                           tolerance: float = POINT_BOUND_TOLERANCE):
    """Fill cells the rules narrow to a single value, and flag them.

    For a pd.NA cell, ``narrow_bounds_by_row`` can return a bound whose ends
    meet: the other values in that row determine the column exactly. Writing
    that value is a derivation, not a guess, and it reaches cells no catalog
    formula does -- measured on the estate, 452,631 cells of
    ``ElektriciteitsgebruikTotaalBruto`` against 431,173 from the formula
    route, on the same population and at a quarter of the cost.

    Only cells that are pd.NA are touched; a delivered or already-derived
    value is never overwritten. Each filled cell is flagged
    ``CALCULATED_FROM_RULE_BOUND``, which is an ACTION distinct from
    ``CALCULATED_FROM_ROW_FORMULA``: the value came from the constraint system
    rather than from evaluating a formula, and a reader needs to be able to
    tell those apart. ``CALCULATED_RULE_BOUND_FROM_SOFT_KNOWNS`` is added where the
    bound rested on a cell the fill wrote.

    Returns *df* with the values written and the flags OR-ed in, plus a
    per-column count of what was filled.
    """
    from etdtransform.constraints import narrow_bounds_by_row

    imputed = imputed_masks(df)
    filled = {}
    for column in [c for c in targets if c in df.columns]:
        missing = df[column].isna()
        if not bool(missing.any()):
            continue
        lo, hi, from_imputed = narrow_bounds_by_row(df, column, package,
                                                    imputed=imputed)
        pinned = (
            missing & lo.notna() & hi.notna() & ((hi - lo).abs() <= tolerance)
        ).fillna(False)
        if not bool(pinned.any()):
            continue

        df.loc[pinned, column] = lo[pinned]

        flag_column = impute_type_column(column)
        if flag_column not in df.columns:
            df[flag_column] = pd.array([0] * len(df), dtype="Int64")
        bits = pinned.astype("Int64") * ImputeType.CALCULATED_FROM_RULE_BOUND.value
        soft = (pinned & pd.Series(from_imputed, index=df.index).fillna(False))
        bits = bits | (
            soft.astype("Int64")
            * ImputeType.CALCULATED_RULE_BOUND_FROM_SOFT_KNOWNS.value
        )
        df[flag_column] = df[flag_column].fillna(0).astype("Int64") | bits
        filled[column] = int(pinned.sum())

    if filled:
        logging.info(
            "[calc flags] CALCULATED_FROM_RULE_BOUND filled %s cell(s): %s",
            f"{sum(filled.values()):,}",
            ", ".join(f"{c} {n:,}" for c, n in sorted(filled.items())),
        )
    return df, filled


def catalog_variables(catalog: pd.DataFrame) -> set:
    """Every variable the catalog names, as a target or as an input."""
    inputs = {str(v) for variables in catalog["rhs_vars"] for v in variables}
    return inputs | {str(lhs) for lhs in catalog["lhs"]}


def choose_formula(candidates, available, imputed_columns=frozenset()):
    """The one formula to evaluate for a target, from those *available* allows.

    Chosen before anything is evaluated, which is possible because every
    candidate has all its inputs present by construction -- membership of
    *available* is the whole test -- so the choice cannot depend on a value.

    Fewest inputs first: a shorter formula compounds fewer errors and reads
    fewer columns. Then fewest inputs that the fill wrote, so a formula
    standing on measurements alone wins over one standing on filled cells.
    Then the expression text, so the same mask always yields the same formula
    and the provenance of a column does not vary between runs.

    Returns None when no candidate is available.
    """
    usable = [
        row for row in candidates
        if set(row.rhs_vars) <= available and str(row.lhs) not in set(row.rhs_vars)
    ]
    if not usable:
        return None
    return min(usable, key=lambda row: (
        len(row.rhs_vars),
        sum(1 for v in row.rhs_vars if v in imputed_columns),
        str(row.rhs_text),
    ))


def fill_from_formula(df: pd.DataFrame, targets, catalog, package,
                      tolerance: float = POINT_BOUND_TOLERANCE):
    """Fill pd.NA cells a catalog formula reaches, and flag them.

    The second fill route. Where a supplier delivers A, B and C and the rules
    derive C from A and B, this fills C's gaps from the rows where A and B are
    both present -- which the derivation pass cannot do, because C is a
    delivered column and so was never a derivation target.

    Rows are grouped by their not-null map over the catalog's variables. That
    map decides which formulas are candidates, so rows sharing one share an
    answer, and the choice is made once per group rather than once per cell.
    An absent input is simply not in the map, so no formula naming it is a
    candidate and nothing here reasons about derivation order -- which is why
    a target derived from a composite that depends on the target is not a
    special case.

    Each written value is tested against the bound its own row narrows the
    target to, and dropped if it falls outside: the fill is an invented value
    and has to be plausible, not merely computable.

    Only pd.NA cells are written. A delivered or already-derived value is
    never overwritten, and the cells written are flagged
    ``CALCULATED_FROM_ROW_FORMULA``, which is how a later reader tells them
    both from measurements in the same column and from a cell produced by
    deriving the whole column, which carries ``CALCULATED_COLUMN`` instead.

    Returns *df*, a per-column count of what was filled, and a plan frame
    recording the formula used for each (mask, target) so the provenance of
    every filled cell is recoverable without storing it per row.
    """
    import sympy as sp

    from etdtransform.calculated_columns import evaluate_over_frame
    from etdtransform.constraints import narrow_bounds_by_row

    present_columns = set(df.columns)
    mask_columns = sorted(catalog_variables(catalog) & present_columns)
    targets = [column for column in targets if column in present_columns]
    if not mask_columns or not targets:
        return df, {}, pd.DataFrame()
    if not df.index.is_unique:
        # Rows are selected by label, per group of rows sharing a not-null
        # map, so a repeated label would write a value to the wrong row.
        raise ValueError(
            "the gap fill selects rows by index label; the frame's index "
            "has repeated labels"
        )

    imputed = imputed_masks(df)
    imputed_columns = frozenset(imputed)
    by_target = {
        target: list(catalog[catalog["lhs"] == target].itertuples())
        for target in targets
    }
    expressions: dict = {}
    filled: dict = {}
    plan_rows: list = []

    presence = df[mask_columns].notna()
    for key, positions in presence.groupby(mask_columns, sort=False).indices.items():
        bits = key if isinstance(key, tuple) else (key,)
        available = {column for column, bit in zip(mask_columns, bits) if bit}
        group_index = df.index[positions]

        for target in targets:
            gap = df.loc[group_index, target].isna()
            if not bool(gap.any()):
                continue
            chosen = choose_formula(by_target[target], available, imputed_columns)
            if chosen is None:
                continue

            rows = gap.index[gap]
            frame = df.loc[rows]
            if chosen.rhs_text not in expressions:
                expressions[chosen.rhs_text] = sp.sympify(chosen.rhs_text)
            values = evaluate_over_frame(expressions[chosen.rhs_text], frame)

            lower, upper, _ = narrow_bounds_by_row(
                frame, target, package,
                imputed={c: s.loc[rows] for c, s in imputed.items()},
            )
            # A bound that is pd.NA on one side is unbounded there, not
            # failed, so that side admits the value.
            within = (
                values.notna()
                & (values >= (lower - tolerance)).fillna(True)
                & (values <= (upper + tolerance)).fillna(True)
            )
            written = within.index[within]
            if len(written):
                df.loc[written, target] = values[written]

                flag_column = impute_type_column(target)
                if flag_column not in df.columns:
                    df[flag_column] = pd.array([0] * len(df), dtype="Int64")
                df[flag_column] = df[flag_column].fillna(0).astype("Int64")
                df.loc[written, flag_column] = (
                    df.loc[written, flag_column]
                    | ImputeType.CALCULATED_FROM_ROW_FORMULA.value
                )
                filled[target] = filled.get(target, 0) + len(written)

            # Recorded even when every value was rejected. A formula that was
            # a candidate and produced implausible values is the finding worth
            # keeping; dropping the row would make it indistinguishable from a
            # mask that had no candidate at all.
            plan_rows.append({
                "mask": "|".join(sorted(available)),
                "variable": target,
                "rhs_text": str(chosen.rhs_text),
                "rhs_vars": ",".join(sorted(str(v) for v in chosen.rhs_vars)),
                "inputs_from_fill": sum(
                    1 for v in chosen.rhs_vars if v in imputed_columns
                ),
                "cells_filled": len(written),
                "cells_rejected": int(len(rows) - len(written)),
            })

    if filled:
        logging.info(
            "[calc flags] CALCULATED_FROM_ROW_FORMULA filled %s gap cell(s) "
            "from formulas: %s",
            f"{sum(filled.values()):,}",
            ", ".join(f"{c} {n:,}" for c, n in sorted(filled.items())),
        )
    return df, filled, pd.DataFrame(plan_rows)


#: Suffixes the patch frame carries into the join. Dropped again before the
#: table is written, so they never reach an output schema.
FILL_SUFFIX = "__fill"
BITS_SUFFIX = "__bits"
KEY_COLUMNS = ("HuisIdBSV", "ReadingDate")


def fill_gaps_in_table(table, household_ids, rule_sets,
                       key_columns=KEY_COLUMNS, checkpoints=None):
    """Run both fill routes over *table* and return the patched table.

    The calculated stage's final pass, in the shape the stage can use: an ibis
    table in, an ibis table out, so the output writers are untouched and the
    stage still writes once.

    Only the rows with a gap are materialised, and only the columns the
    formulas and the rules name. That is what makes this affordable at estate
    scale -- the pass costs a fraction of the derivation it follows because
    almost every row has nothing missing to fill.

    The values are then joined back on *key_columns* and coalesced, so a cell
    the pass did not touch keeps the expression it already had. Flag bits are
    OR-ed, which is idempotent, so a cell already marked by the derivation
    pass keeps that mark and gains the fill's.

    Each household is given its own formula set and bounds via *rule_sets*
    from :func:`etdtransform.constraints.rule_sets_by_household`: a household
    never sees a formula naming a device it does not have.

    Returns ``(table, report, plan)``.
    """
    import ibis

    from etdtransform.impute import _to_nullable

    present = set(table.columns)
    keys = [column for column in key_columns if column in present]
    if len(keys) != len(key_columns):
        raise ValueError(
            f"the gap fill joins on {list(key_columns)}; the table has "
            f"{keys}"
        )

    report = {"from_formula": {}, "from_rule_bound": {}, "cells": 0}
    plans = []
    pieces: dict = {}
    members_by_set = {
        key: sorted({int(h) for h in household_ids} & set(rule_set.households))
        for key, rule_set in rule_sets.items()
    }

    for key, rule_set in rule_sets.items():
        members = members_by_set[key]
        if not members:
            continue
        variables = catalog_variables(rule_set.formulas) & present
        targets = sorted({str(lhs) for lhs in rule_set.formulas["lhs"]} & present)
        if not targets or not variables:
            continue

        needed = sorted(
            variables
            | set(keys)
            | {c for c in present if c.endswith("_is_imputed")}
            | ({impute_type_column(t) for t in targets} & present)
        )
        scope = table.filter(table["HuisIdBSV"].isin(members))
        gap = None
        for target in targets:
            missing = scope[target].isnull()
            gap = missing if gap is None else (gap | missing)
        with _maybe_phase(checkpoints, "pass2_read_gap_rows") as mark:
            frame = scope.filter(gap).select(*needed).execute()
            mark["rows"] = len(frame)
            mark["note"] = f"{len(needed)} columns the formulas and rules name"
        if frame.empty:
            continue

        frame = _to_nullable(frame, float_cols=sorted(variables))
        was_missing = {target: frame[target].isna() for target in targets}
        frame, set_report, plan = fill_calculated_gaps(
            frame, targets, rule_set.formulas, rule_set.package,
            checkpoints=checkpoints,
        )
        if not plan.empty:
            plan = plan.assign(apparaat_set=key)
            plans.append(plan)
        for route in ("from_formula", "from_rule_bound"):
            for column, count in set_report[route].items():
                report[route][column] = report[route].get(column, 0) + count
        report["cells"] += set_report["cells"]

        for target in targets:
            changed = was_missing[target] & frame[target].notna()
            if not bool(changed.any()):
                continue
            flag_column = impute_type_column(target)
            pieces.setdefault(target, []).append(pd.DataFrame({
                **{column: frame.loc[changed, column] for column in keys},
                f"{target}{FILL_SUFFIX}": frame.loc[changed, target].astype("Float64"),
                f"{target}{BITS_SUFFIX}": (
                    frame.loc[changed, flag_column].astype("Int64")
                ),
            }))

    if not pieces:
        logging.info("[calc flags] gap fill wrote nothing in this unit")
        return table, report, pd.DataFrame()

    # Recorded because it sat between the two fill routes and was therefore
    # invisible: assembling the patch and building the join is neither of the
    # two passes, and on an estate run the unrecorded remainder was a quarter
    # of the stage.
    with _maybe_phase(checkpoints, "pass2_patch_and_join") as mark:
        patch = None
        for target in sorted(pieces):
            combined = pd.concat(pieces[target], ignore_index=True)
            patch = combined if patch is None else patch.merge(
                combined, on=keys, how="outer"
            )
        mark["rows"] = len(patch)
        mark["note"] = f"{len(pieces)} column(s) patched back"
        joined = table.left_join(ibis.memtable(patch), keys)

    mutations = {}
    for target in sorted(pieces):
        mutations[target] = joined[target].coalesce(
            joined[f"{target}{FILL_SUFFIX}"]
        )
        flag_column = impute_type_column(target)
        bits = joined[f"{target}{BITS_SUFFIX}"].fill_null(0)
        mutations[flag_column] = (
            joined[flag_column].fill_null(0) | bits
            if flag_column in present
            else bits
        )
    patch_columns = [
        column for column in joined.columns
        if column.endswith((FILL_SUFFIX, BITS_SUFFIX, "_right"))
    ]
    return joined.mutate(**mutations).drop(*patch_columns), report, (
        pd.concat(plans, ignore_index=True) if plans else pd.DataFrame()
    )


@contextmanager
def _maybe_phase(checkpoints, name, **fields):
    """Record a phase when there is a recorder, otherwise just run the body."""
    if checkpoints is None:
        yield {}
    else:
        with checkpoints.phase(name, **fields) as mark:
            yield mark


def fill_calculated_gaps(df: pd.DataFrame, targets, catalog, package,
                         tolerance: float = POINT_BOUND_TOLERANCE,
                         checkpoints=None):
    """Both fill routes, in the order that lets the second use the first.

    The final pass of the calculated stage. The formula route runs first and
    the rule-bound route second, because a cell the formula fills is a known
    value in its row afterwards and can narrow another cell's bound to a
    point. The reverse order would leave those cells unfilled.

    Each route writes its own ACTION flag -- ``CALCULATED_FROM_ROW_FORMULA``
    and ``CALCULATED_FROM_RULE_BOUND`` -- so which route produced a value is
    readable per cell, and ``CALCULATED_VALUE`` matches either, as it also
    matches pass 1's ``CALCULATED_COLUMN``.

    Returns *df*, a report of the per-column counts for each route, and the
    formula route's plan frame.
    """
    with _maybe_phase(checkpoints, "pass2_cell_filling") as mark:
        df, from_formula, plan = fill_from_formula(
            df, targets, catalog, package, tolerance=tolerance
        )
        mark["rows"] = sum(from_formula.values())
        mark["note"] = "formula reaches a gap from the same row"

    with _maybe_phase(checkpoints, "pass2_zero_width_derivation") as mark:
        df, from_bound = derive_from_rule_bound(
            df, targets, package, tolerance=tolerance
        )
        mark["rows"] = sum(from_bound.values())
        mark["note"] = "rules narrow the row to a single value"
    report = {
        "from_formula": from_formula,
        "from_rule_bound": from_bound,
        "cells": sum(from_formula.values()) + sum(from_bound.values()),
    }
    logging.info(
        "[calc flags] gap fill wrote %s cell(s): %s from formulas, %s from "
        "rule bounds",
        f"{report['cells']:,}",
        f"{sum(from_formula.values()):,}",
        f"{sum(from_bound.values()):,}",
    )
    return df, report, plan
