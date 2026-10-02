"""Flags the calculated stage writes, and the threshold scan behind them.

Three members of ``ImputeType`` belong to this stage: ``CALCULATED`` marks a
value as produced by a catalog formula rather than measured, and the two
``CALCULATED_*`` violation flags mark it as outside a bound. They go into the
same impute-type column the fill writes, so one decode reads both stages.

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

    - ``CALCULATED`` on every cell holding a value;
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

    A fourth mark, ``CALCULATED_RULE_BOUND_FROM_IMPUTED``, says the rule bound
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
        bits = present.astype("Int64") * ImputeType.CALCULATED.value

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
                * ImputeType.CALCULATED_RULE_BOUND_FROM_IMPUTED.value
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
        "[calc flags] grid %s min; CALCULATED on %s cell(s) across %d "
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
    """``CALCULATED`` and the THRESHOLD flag as ibis expressions.

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
        bits = present.cast("int64") * ImputeType.CALCULATED.value

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
         ImputeType.CALCULATED_RULE_BOUND_FROM_IMPUTED.value),
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
