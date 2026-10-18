import logging

import numpy as np
import pandas as pd
from etdmap.data_model import interval_column
from etdmap.flags import (
    MappingFlag,
    bound_flags,
    frozen_spans,
)
from etdmap.record_validators import thresholds_dict

from etdtransform.bounds import (
    cadence_minutes_by_household,
    clip_to_bounds,
    column_bound,
    fill_bound,
    load_column_cadence,
    usable_readings,
)
from etdtransform.flags import ImputeType

__all__ = [
    'ImputeType',
    'apply_thresholds',
    'drop_temp_cols',
    'impute_and_normalize',
    'methods_to_bitwise',
    'process_gap_and_cumulative_groups',
    'process_imputation_vectorized'
]

#: ASSESSMENT scale for JITTER_CANDIDATE, pending a metadata decision.
#:
#: The data model carries no per-column metering-resolution field, so the
#: scale cannot yet be derived from metadata. 0.02 kWh is the measured p99 of
#: the known bound contradictions (phase 3 validation: median 0.002, p99
#: 0.021 kWh) -- an empty bound whose overshoot is at or below this is marked
#: a jitter CANDIDATE for assessment, nothing more: the flag is a signal and
#: changes no fill behaviour. Per-column overrides go in
#: ``JITTER_ASSESSMENT_SCALE_KWH``; the default applies to every energy Diff
#: column otherwise. When a metering-resolution field lands in the data
#: model, derivation replaces this table.
DEFAULT_JITTER_ASSESSMENT_SCALE_KWH = 0.02
JITTER_ASSESSMENT_SCALE_KWH: dict = {}


def _jitter_assessment_scale(diff_col: str) -> float:
    return JITTER_ASSESSMENT_SCALE_KWH.get(
        diff_col, DEFAULT_JITTER_ASSESSMENT_SCALE_KWH
    )


#: Gate for the whole-household no-known short-circuit in the narrowing.
#: Evaluating only the rows with known rule inputs is faster below this
#: evaluated-row share and slower above it, because the subset must be
#: materialised. The split path applies only when the evaluated share of a
#: (cadence, model)
#: group is below this.
MASKED_EVAL_MAX_EVALUATED_SHARE = 0.5

def methods_to_bitwise(methods_column):
    """
    Convert methods to bitwise representation.

    This function takes a column of methods and converts each method to a bitwise
    representation. Each method is represented by a bit in the resulting integer.

    Parameters
    ----------
    methods_column : array-like
        A column containing lists of method numbers.

    Returns
    -------
    numpy.ndarray
        An array of integers where each integer represents the bitwise
        representation of the methods for that row.

    Notes
    -----
    The values in ``methods_column`` are ``ImputeType`` members, which is an
    ``IntFlag`` built with ``auto()``. Combining them is therefore a
    plain OR of the values.

    """

    bitwise_values = np.zeros(len(methods_column), dtype=np.int64)

    for i, methods in enumerate(methods_column):
        bitwise_value = 0
        for method in methods:
            if method > 0:
                bitwise_value |= int(method)
        bitwise_values[i] = bitwise_value

    return bitwise_values


def apply_thresholds(
    df,
    lower_bound,
    upper_bound,
    diff_col,
    avg_col,
    impute_type_col,
    is_imputed_col,
    detection_bound=None,
):
    """
    Apply thresholds to difference column and update imputation flags.

    This function applies lower and upper bounds to a difference column in the
    DataFrame. Values outside these bounds are replaced with average values,
    and corresponding imputation flags are updated.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data.
    lower_bound : float
        The lower threshold for the difference column.
    upper_bound : float
        The upper threshold for the difference column.
    diff_col : str
        The name of the difference column to apply thresholds to.
    avg_col : str
        The name of the column containing average values to use for imputation.
    impute_type_col : str
        The name of the column indicating the imputation type.
    is_imputed_col : str
        The name of the column indicating whether a value is imputed.
    detection_bound : float, pandas.Series or None, optional
        The threshold used to decide whether a value is implausible, when it
        differs from *upper_bound*. A filled value is judged against the grid
        threshold while a delivered value is judged against the column
        threshold, so the two differ wherever a column does not report at the
        grid cadence. Defaults to *upper_bound*.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with thresholds applied and imputation flags updated.

    Notes
    -----
    This function modifies the input DataFrame in-place and also returns it.
    Values outside the thresholds are replaced with average values and marked
    as imputed.
    """
    # Out-of-bounds rows, judged against the column threshold.
    mask = (
        bound_flags(
            df[diff_col],
            lower_bound,
            upper_bound if detection_bound is None else detection_bound,
        )
        != MappingFlag.NONE.value
    )

    # Catch-up readings are excluded from the mask: their volume is already
    # accounted for by the cumulative variable.
    catchup = _catchup_positions(
        df, diff_col, upper_bound if detection_bound is None else detection_bound
    )
    if catchup.any():
        logging.info(
            f"Keeping {int(catchup.sum())} catch-up value(s) in {diff_col}: "
            f"above the bound, but the cumulative variable accounts for them."
        )
        mask = mask & ~catchup

    # The average is the replacement, so where it is absent the pass blanks the
    # value instead of substituting one. The two outcomes carry different
    # action flags; THRESHOLD_VIOLATED records the condition either way.
    replaced = mask & df[avg_col].notna()
    blanked = mask & df[avg_col].isna()

    df.loc[mask, diff_col] = df.loc[mask, avg_col]
    df.loc[mask, is_imputed_col] = True
    df.loc[replaced, impute_type_col] = (
        df[impute_type_col].fillna(ImputeType.NONE.value)
        | ImputeType.THRESHOLD_ADJUSTED.value
        | ImputeType.THRESHOLD_VIOLATED.value
        | ImputeType.UNSCALED_AVERAGE.value
    )
    df.loc[blanked, impute_type_col] = (
        df[impute_type_col].fillna(ImputeType.NONE.value)
        | ImputeType.THRESHOLD_ADJUSTED.value
        | ImputeType.THRESHOLD_VIOLATED.value
        | ImputeType.NULLED_POST_IMPUTATION.value
    )

    return df


def _catchup_positions(df, diff_col, upper_bound) -> pd.Series:
    """
    Rows where an over-bound value is the catch-up of a frozen span.

    Calls the mapping-stage detector, ``etdmap.flags.frozen_spans``. A value
    counts as a catch-up only when a flat run of non-advancing readings sits
    directly before it.

    Runs per household, since a span is a property of one household's series.
    *upper_bound* is a scalar or one value per row; the bound is the same for
    every row of a household, so the first of them is the household's bound.
    """
    found = pd.Series(False, index=df.index)
    if df.empty:
        return found

    per_row = isinstance(upper_bound, pd.Series)
    if not per_row and pd.isna(upper_bound):
        return found

    group_column = "HuisIdBSV" if "HuisIdBSV" in df.columns else None
    groups = (
        df.groupby(group_column, sort=False) if group_column else [(None, df)]
    )
    for _, group in groups:
        bound = upper_bound.loc[group.index].iloc[0] if per_row else upper_bound
        if pd.isna(bound):
            continue
        spans = frozen_spans(
            group[diff_col].reset_index(drop=True), upper_bound=bound
        )
        if not spans.empty:
            found.loc[group.index[spans["catchup_position"].astype(int)]] = True

    return found


def _row_bounds_per_cadence(
    df,
    diff_col,
    constraints,
    cadence,
    packages,
    imputed=None,
    physical_model_by_project=None,
    apparaat_sets_by_household=None,
):
    """Per-row narrowed bounds for *diff_col*, at each household's own cadence,
    physical model and apparaat-set.

    A constraint package fixes one cadence and one rule subset, and households
    differ on all three axes: they report at more than one cadence, their
    project's physical model decides which rules apply to them at all, and
    their declared-absent devices select which SPECIALIZED compile of those
    rules they narrow with (the sync substitutes 0 for absent variables per
    distinct set and writes the table keyed by ``apparaat_set``). So the
    frame is resolved in (cadence, model, apparaat-set) groups, each with its
    own package built once per call into *packages* -- selection only, no
    sympy here. A household absent from the cadence table takes the declared
    bounds unscaled -- the same resolution the column thresholds apply
    through ``grid_bound``. A project absent from
    *physical_model_by_project* raises: the model decides the rules, so an
    unregistered project is a sync gap, never a default. Likewise a
    household absent from *apparaat_sets_by_household*, or a set with no
    compiled table, raises as an unsynced state.

    A constraint table WITHOUT the ``apparaat_set`` column (the pre-split
    artifact) is treated as the empty-set table for every household, with a
    warning naming the re-sync -- tests and comparison runs on the committed
    artifact keep working until the next apply.

    Returns ``(lo, hi, from_imputed)`` aligned to ``df.index``, as
    :func:`etdtransform.constraints.narrow_bounds_by_row` returns them per
    group.
    """
    from etdtransform.constraints import (
        build_constraint_package,
        narrow_bounds_by_row,
    )

    if physical_model_by_project is None:
        from etdmap.data_model import physical_model_by_project

    models = df["ProjectIdBSV"].map(physical_model_by_project)
    unmapped = df["ProjectIdBSV"][models.isna()].dropna().unique()
    if len(unmapped):
        raise ValueError(
            f"No physical model registered for ProjectIdBSV "
            f"{sorted(int(p) for p in unmapped)}. The model decides which "
            f"constraint rules apply, so it cannot default. Set FysischModel "
            f"in the Grist Project table and re-run the sync, which writes "
            f"etdmap's projects.csv."
        )

    set_keyed = "apparaat_set" in constraints.columns
    if set_keyed:
        if apparaat_sets_by_household is None:
            from etdtransform.constraints import apparaat_set_by_household

            apparaat_sets_by_household = apparaat_set_by_household()
        if "HuisIdBSV" in df.columns:
            set_keys = df["HuisIdBSV"].map(apparaat_sets_by_household)
            unmapped_households = (
                df["HuisIdBSV"][set_keys.isna()].dropna().unique()
            )
            if len(unmapped_households):
                raise ValueError(
                    f"No apparaat-set registered for HuisIdBSV "
                    f"{sorted(int(h) for h in unmapped_households)}. The "
                    f"declared-absent set selects the specialized constraint "
                    f"table, so it cannot default. Rebuild the BSV metadata "
                    f"(sync_data_model.py / build_bsv_metadata.py)."
                )
            set_keys = set_keys.astype("string")
        else:
            # No household column: a caller that scoped its own frame. The
            # empty set is the unspecialized table and the only honest pick.
            set_keys = pd.Series("", index=df.index, dtype="string")
        available_sets = set(constraints["apparaat_set"].astype(str))
    else:
        if "_legacy_constraints_warned" not in packages:
            packages["_legacy_constraints_warned"] = True
            logging.warning(
                "[narrow] constraints table carries no apparaat_set column "
                "(pre-specialization artifact): treating it as the empty-set "
                "table for every household. Re-run the data model sync to "
                "write the per-set tables."
            )
        set_keys = pd.Series("", index=df.index, dtype="string")
        available_sets = {""}

    by_household = cadence_minutes_by_household(diff_col, cadence)
    if "HuisIdBSV" in df.columns:
        minutes = df["HuisIdBSV"].map(by_household)
    else:
        minutes = pd.Series(pd.NA, index=df.index, dtype="Float64")

    # The narrowing reads only the rule variables and their findings columns,
    # so hand it those rather than a row-subset copy of the whole chunk frame
    # -- which is wider by two orders of magnitude.
    from etdmap.flags import flag_column

    wanted = sorted(
        set(constraints.loc[constraints["lhs"] == diff_col, "variable"].astype(str))
    )
    narrow_columns = [
        c
        for name in wanted
        for c in (name, flag_column(name))
        if c in df.columns
    ]
    narrow_frame = df[narrow_columns] if narrow_columns else df[[]]

    # The whole-household no-known short-circuit: a household none of whose
    # rule inputs is ever known in this chunk gets, on every row, exactly the
    # package's presolved bound for the target -- the per-row narrowing
    # substitutes each unknown at its presolved endpoint, which the presolve
    # already intersected to fixpoint, so evaluating it per row recomputes a
    # constant. Knownness here is the SAME predicate the narrowing applies
    # (``usable_readings``), over the superset of rule variables, so a
    # household this counts as known is simply evaluated normally -- the
    # short-circuit can only skip work, never change a bound.
    known_any = pd.Series(False, index=df.index)
    for name in wanted:
        known_any = known_any | usable_readings(df, name)
    if "HuisIdBSV" in df.columns:
        known_household = known_any.groupby(df["HuisIdBSV"]).transform("any")
    else:
        known_household = pd.Series(bool(known_any.any()), index=df.index)

    lo = pd.Series(pd.NA, index=df.index, dtype="Float64")
    hi = pd.Series(pd.NA, index=df.index, dtype="Float64")
    from_imputed = pd.Series(False, index=df.index, dtype="boolean")

    _inf = float("inf")

    distinct = [None] if minutes.isna().any() else []
    distinct += sorted(float(m) for m in minutes.dropna().unique())
    for value in distinct:
        cadence_key = None if value is None else value
        cadence_mask = (
            minutes.isna() if cadence_key is None
            else (minutes == value).fillna(False)
        )
        for model in sorted(models[cadence_mask].dropna().unique()):
            model_mask = cadence_mask & (models == model).fillna(False)
            for set_key in sorted(
                str(s) for s in set_keys[model_mask].dropna().unique()
            ):
                key = (cadence_key, str(model), set_key)
                if key not in packages:
                    if set_key not in available_sets:
                        raise ValueError(
                            f"No compiled constraint table for apparaat-set "
                            f"'{set_key}'. The sync compiles one table per "
                            f"declared set; a set without a table is an "
                            f"unsynced state -- run the data model sync."
                        )
                    subset = (
                        constraints[
                            constraints["apparaat_set"].astype(str) == set_key
                        ]
                        if set_keyed
                        else constraints
                    )
                    packages[key] = build_constraint_package(
                        subset,
                        cadence_minutes=cadence_key,
                        physical_model=str(model),
                    )
                mask = model_mask & (set_keys == set_key).fillna(False)
                package = packages[key]

                # Split the group only below the measured crossover: with no
                # rules for the target every row is the presolve constant, and
                # with few known households evaluating only those wins; past
                # ~50% evaluated share the subset copy costs more than it saves,
                # so the group is evaluated whole as before (identical result --
                # no-known rows come out at the presolve constant either way).
                rules_for_target = package.rules_by_target.get(diff_col, ())
                group_rows = int(mask.sum())
                if rules_for_target and group_rows:
                    evaluated = mask & known_household.fillna(False)
                    share = float(evaluated.sum()) / group_rows
                else:
                    evaluated = pd.Series(False, index=df.index)
                    share = 0.0
                use_split = (
                    group_rows > 0
                    and share < MASKED_EVAL_MAX_EVALUATED_SHARE
                    and not bool(evaluated.equals(mask))
                )
                if not use_split:
                    evaluated = mask
                skipped = mask & ~evaluated

                if skipped.any():
                    start_lo, start_hi = package.bounds.get(
                        diff_col, (-_inf, _inf)
                    )
                    lo.loc[skipped] = (
                        pd.NA if start_lo == -_inf else float(start_lo)
                    )
                    hi.loc[skipped] = (
                        pd.NA if start_hi == _inf else float(start_hi)
                    )
                    # from_imputed stays False: nothing was substituted.
                    logging.info(
                        f"[narrow] {diff_col}: {int(skipped.sum())} row(s) in "
                        f"household(s) with no known rule input take the "
                        f"presolved bound without per-row evaluation "
                        f"(evaluated share {share:.0%})."
                    )

                if evaluated.any():
                    whole_frame = bool(evaluated.all())
                    part_frame = (
                        narrow_frame if whole_frame
                        else narrow_frame.loc[evaluated]
                    )
                    part_imputed_in = (
                        None if imputed is None
                        else (
                            imputed if whole_frame
                            else {
                                name: series.loc[evaluated]
                                for name, series in imputed.items()
                            }
                        )
                    )
                    part_lo, part_hi, part_imputed = narrow_bounds_by_row(
                        part_frame,
                        diff_col,
                        package,
                        imputed=part_imputed_in,
                    )
                    if whole_frame:
                        return part_lo, part_hi, part_imputed
                    lo.loc[evaluated] = part_lo
                    hi.loc[evaluated] = part_hi
                    from_imputed.loc[evaluated] = part_imputed
    return lo, hi, from_imputed


def without_tariff_splits(columns):
    """Drop Hoog/Laag tariff register columns from a column list.

    The combined root is delivered at mapping (``combine_tariff_registers``),
    and the discrete alternating splits are not physically averageable: only
    the continuous root is averaged and filled. The splits stay in the mapped
    data untouched -- they simply contribute no averages and receive no
    fills. Membership comes from the data model's own tariff pair structures
    (``tariff_partner_columns`` covers both the cumulative and Diff forms),
    never from name heuristics here.
    """
    from etdmap.data_model import tariff_partner_columns

    return [c for c in columns if c not in tariff_partner_columns]


def order_columns_for_fill(cumulative_columns, coverage_duty):
    """The fill order: coverage descending, then duty descending, then name.

    A later column's bounds rest on what earlier columns filled, so the
    columns whose averages exist most and move most go first. Ordered from
    ``avg_coverage_duty.parquet`` -- one row per project and column -- taking
    each column's median across its projects, so a re-run orders exactly as
    the run that produced the averages did. A column the table does not carry
    sorts after every one it does, alphanumerically.
    """
    medians = coverage_duty.groupby("column")[["coverage", "duty"]].median()

    def rank(cum_col):
        diff = interval_column(cum_col)
        if diff in medians.index:
            row = medians.loc[diff]
            return (0, -float(row["coverage"]), -float(row["duty"]), cum_col)
        return (1, 0.0, 0.0, cum_col)

    return sorted(cumulative_columns, key=rank)


def _allocate_gap_fills(df, diff_col, impute_type_col):
    """Clip the anchored fills to each row's bound and place the volume back.

    An anchored gap carries a measured jump, so cutting a cell to its bound
    (``CLIPPED``) leaves volume that still belongs in the gap -- or, cut at a
    floor, volume that must come back out. It moves in one pass, in proportion
    to each remaining cell's headroom::

        add = capacity * min(1, excess / total_capacity)

    The factor never exceeds one, so no cell can overshoot its own cap and
    nothing needs a second pass; wherever there is room, exactly the excess is
    placed (``CLIPPED_REDISTRIBUTION``). Cells whose profile weight is zero
    stay out of that pool -- in it they would quietly absorb volume and the
    reserve would never fire -- and instead take any remainder uniformly, up
    to their own caps (``UNIFORM_RESIDUAL``): the jump is a measurement, a
    zero profile only an estimate that the device was off. What survives the
    reserve stays unplaced in ``_gap_unplaced``, signed by direction; the gap
    total then cannot be met, which the infeasibility pass records.

    The excess is what clipping moved, not the jump minus the gap's content:
    a partially-filled gap (cells the route left empty for want of an
    average) keeps its pre-existing deficit untouched.

    Reads the row bounds from ``_row_bound_lo`` / ``_row_bound_hi`` and does
    nothing when they are absent.
    """
    if "_row_bound_lo" not in df.columns:
        return df

    INF = float("inf")
    lo = df["_row_bound_lo"]
    hi = df["_row_bound_hi"]

    findings = df[impute_type_col].fillna(ImputeType.NONE.value)
    anchored = (
        (findings & (ImputeType.UNIFORM_FILL.value | ImputeType.SCALED_FILL.value))
        != 0
    ) & df["cumulative_value_group"].notna()
    if not anchored.any():
        return df

    # An empty bound permits no value at all; that is the infeasibility
    # pass's case, not a clip, so those cells are left exactly as filled.
    feasible = ~(lo > hi).fillna(False)

    over = anchored & feasible & (df[diff_col] > hi).fillna(False)
    under = anchored & feasible & (df[diff_col] < lo).fillna(False)
    if not (over.any() or under.any()):
        return df

    # The volume each clip moves, taken as it happens -- no copy of the
    # column. Positive: cut down, volume wants back in. Negative: pushed up
    # to a floor, volume must come out.
    moved_by_clip = (
        (df[diff_col] - hi).where(over, 0.0)
        - (lo - df[diff_col]).where(under, 0.0)
    ).fillna(0.0)

    df.loc[over, diff_col] = hi[over]
    df.loc[under, diff_col] = lo[under]
    clipped = over | under
    df.loc[clipped, impute_type_col] = (
        df[impute_type_col].fillna(ImputeType.NONE.value)
        | ImputeType.CLIPPED.value
    )

    group = df["cumulative_value_group"]
    delta = moved_by_clip.groupby(group).transform("sum")

    # A uniform fill has no shape, so every one of its cells counts as active.
    weight = df["impute_values"].fillna(0.0).where(
        (findings & ImputeType.UNIFORM_FILL.value) == 0, 1.0
    )
    eligible = anchored & feasible & ~clipped & (weight > 0)
    reserve = anchored & feasible & ~clipped & (weight <= 0)

    df["_gap_unplaced"] = pd.Series(0.0, index=df.index, dtype="Float64")
    # Signed, like the clip delta: positive volume was placed back after
    # ceiling clips, negative was taken back after floor clips.
    df["_gap_redistributed"] = pd.Series(0.0, index=df.index, dtype="Float64")
    df["_gap_uniform_residual"] = pd.Series(0.0, index=df.index, dtype="Float64")

    for direction, bound in ((1.0, hi), (-1.0, lo)):
        direction_rows = (delta * direction) > 1e-12
        if not direction_rows.any():
            continue
        need = (delta * direction).where(direction_rows, 0.0)

        capacity = ((bound - df[diff_col]) * direction).fillna(INF)
        cap = capacity.where(eligible & direction_rows, 0.0)
        total = cap.groupby(group).transform("sum")
        ratio = (need / total).clip(upper=1.0)
        add = (cap * ratio).fillna(0.0)
        # An unbounded cell's headroom is infinite, so the proportional share
        # degenerates (ratio 0 against an infinite total). In that limit the
        # excess splits evenly across the unbounded cells, which absorb it all.
        unbounded = eligible & direction_rows & (capacity == INF)
        if unbounded.any():
            n_unbounded = unbounded.groupby(group).transform("sum")
            add = add.where(~unbounded, need / n_unbounded)

        moved = add > 1e-12
        df.loc[moved, diff_col] = round(df[diff_col] + direction * add, 10)[moved]
        df.loc[moved, impute_type_col] = (
            df[impute_type_col].fillna(ImputeType.NONE.value)
            | ImputeType.CLIPPED_REDISTRIBUTION.value
        )
        placed_total = add.groupby(group).transform("sum")
        df.loc[direction_rows, "_gap_redistributed"] = (
            direction * placed_total
        )[direction_rows]

        leftover = need - placed_total
        residual_rows = reserve & direction_rows & (leftover > 1e-12)
        placed_r = pd.Series(0.0, index=df.index, dtype="Float64")
        if residual_rows.any():
            cap_r = capacity.where(residual_rows, 0.0)
            n_reserve = residual_rows.groupby(group).transform("sum")
            share = (leftover / n_reserve).where(n_reserve > 0, 0.0)
            add_r = share.where(share <= cap_r, cap_r).where(residual_rows, 0.0)
            placed = add_r > 1e-12
            df.loc[placed, diff_col] = round(
                df[diff_col] + direction * add_r, 10
            )[placed]
            df.loc[placed, impute_type_col] = (
                df[impute_type_col].fillna(ImputeType.NONE.value)
                | ImputeType.UNIFORM_RESIDUAL.value
            )
            placed_r = add_r
        residual_total = placed_r.groupby(group).transform("sum")
        df.loc[direction_rows, "_gap_uniform_residual"] = (
            direction * residual_total
        )[direction_rows]
        unplaced = leftover - residual_total
        still_short = direction_rows & (unplaced > 1e-12)
        df.loc[still_short, "_gap_unplaced"] = (direction * unplaced)[still_short]

    return df


def _bits_or_by_group(values, group):
    """Bitwise OR of *values* per *group*, as an Int64 Series indexed by group.

    *group* holds one contiguous run of rows per group, so each run reduces
    with ``np.bitwise_or.reduceat`` over the run boundaries. Runs sharing a
    group are folded together, which leaves the result correct if a group
    ever appears in more than one run.
    """
    bits = values.to_numpy(dtype="int64")
    codes, uniques = pd.factorize(group, sort=False)
    run_start = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    run_bits = np.bitwise_or.reduceat(bits, run_start)
    run_codes = codes[run_start]

    if len(run_start) == len(uniques):
        return pd.Series(
            run_bits, index=pd.Index(uniques[run_codes], name=group.name),
            dtype="Int64",
        )

    folded = np.zeros(len(uniques), dtype="int64")
    np.bitwise_or.at(folded, run_codes, run_bits)
    return pd.Series(
        folded, index=pd.Index(uniques, name=group.name), dtype="Int64",
    )


def _gap_volume_report(df, diff_col, impute_type_col, routed_by_gap):
    """One accumulator row per gap: the fill's TRANSIENT state only.

    Collected inside the fill, before the threshold pass -- that pass records
    its own actions in the flags and is not this report's subject, which is
    exactly why these columns must be captured here: total_final and the
    per-gap bits are pre-threshold snapshots, the routed/redistributed/
    unplaced volumes read temp columns, and none of it exists in the
    persisted outputs. Everything DERIVABLE from the outputs (gap_length,
    the cells_* counts, deviation and its fraction, rule_expressions) is
    assembled by the end-of-pipeline post-processing step, which joins this
    accumulator to the persisted flags on (HuisIdBSV, column, gap_start,
    gap_end) -- the span columns exist for that join, so the post-processor
    never has to trust re-derived gap numbering.

    Data-volume sums take ``min_count=1``, so a gap holding no values
    reports pd.NA rather than a zero total. The signed convention
    throughout: positive volume was cut away or placed back downward-of-cap,
    negative the mirror.
    """
    in_gap = df["cumulative_value_group"].notna()
    if not in_gap.any():
        return pd.DataFrame()

    scoped = df.loc[in_gap]
    group = scoped["cumulative_value_group"]
    grouped = scoped.groupby(group)

    report = grouped.agg(
        ProjectIdBSV=("ProjectIdBSV", "first"),
        HuisIdBSV=("HuisIdBSV", "first"),
        gap_start=("ReadingDate", "first"),
        gap_end=("ReadingDate", "last"),
        gap_jump=("gap_jump", "first"),
    )
    report["total_final"] = grouped[diff_col].sum(min_count=1)
    report["total_routed"] = (
        routed_by_gap.reindex(report.index)
        if routed_by_gap is not None
        else pd.Series(pd.NA, index=report.index, dtype="Float64")
    )
    report["anchored"] = report["gap_jump"].notna()

    findings = scoped[impute_type_col].fillna(ImputeType.NONE.value)
    report["impute_type_bits"] = _bits_or_by_group(findings, group)

    for name, column in (
        ("redistributed_volume", "_gap_redistributed"),
        ("uniform_residual_volume", "_gap_uniform_residual"),
        ("unplaced_volume", "_gap_unplaced"),
        ("unanchored_removed_volume", "_gap_unanchored_removed"),
    ):
        if column in df.columns:
            report[name] = grouped[column].first().astype("Float64")
        else:
            report[name] = pd.Series(0.0, index=report.index, dtype="Float64")

    # The worst bound contradiction inside the gap, where any bound came up
    # empty -- transient like its neighbours (the bounds exist only in-run),
    # and the magnitude evidence behind the JITTER_CANDIDATE signal, so the
    # assessment is never locked to the announced scale.
    if "_bound_overshoot" in df.columns:
        report["bound_overshoot_max"] = (
            grouped["_bound_overshoot"].max().astype("Float64")
        )
    else:
        report["bound_overshoot_max"] = pd.Series(
            pd.NA, index=report.index, dtype="Float64"
        )

    report.index.name = "gap"
    return report.reset_index()


def impute_and_normalize(
    df: pd.DataFrame,
    cumulative_columns: list,
    project_id_column: str,
    cadence: pd.DataFrame | None = None,
    constraints: pd.DataFrame | None = None,
    coverage_duty: pd.DataFrame | None = None,
    physical_models: dict | None = None,
    apparaat_sets: dict | None = None,
    volume_reports: list | None = None,
):
    """
    Perform vectorized imputation and normalization on cumulative columns

    This function applies imputation techniques to fill missing values in
    cumulative columns and normalizes the data. It uses vectorized operations
    for improved performance.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data to be imputed and normalized.
    cumulative_columns : list
        A list of column names representing cumulative variables to be processed.
    project_id_column : str
        The name of the column containing project identifiers.
    cadence : pandas.DataFrame, optional
        The per-column cadence table. Every threshold is scaled to the cadence
        the column reports at, so it is read from the mapped folder when not
        given, and an absent sidecar is an error rather than a grid assumption.
    constraints : pandas.DataFrame, optional
        The constraint compiler's table (``load_constraints``). When given,
        each column's rows get bounds narrowed by the rules from the row's own
        known values, the average the fill reads is clipped into them
        (``CLIPPED_AVERAGE`` where a clipped average was used), and a delivered
        value outside its row's narrowed bound is marked
        ``DELIVERED_OUTSIDE_RULE_BOUND`` without being touched. ``None`` is the
        off switch: no narrowing and no per-row flags.
    coverage_duty : pandas.DataFrame, optional
        The ``avg_coverage_duty.parquet`` table. When given, the columns are
        filled in the order :func:`order_columns_for_fill` derives from it;
        ``None`` keeps the caller's order.
    physical_models : dict, optional
        ``ProjectIdBSV -> physical model`` for rule selection. Defaults to
        etdmap's synced project registry; only used when *constraints* is
        given, and a project the mapping does not carry raises rather than
        defaulting.
    apparaat_sets : dict, optional
        ``HuisIdBSV -> apparaat-set key`` selecting each household's
        specialized constraint table when *constraints* carries the
        ``apparaat_set`` column. Defaults to the BSV metadata's
        declarations (``apparaat_set_by_household``); ignored for a
        pre-specialization table, which is treated as the empty set with a
        warning.
    volume_reports : list, optional
        When given, one per-gap volume frame per column is appended, each
        carrying its ``column`` and the rule expressions that were in force
        for it. The caller concatenates and writes them; nothing is written
        here.

    Returns
    -------
    tuple
        A tuple containing three elements:

        - df : pandas.DataFrame
            The imputed and normalized DataFrame.

        - imputation_gap_stats_df : pandas.DataFrame
            Statistics about the imputation process for each gap.

        - imputation_reading_date_stats_df : None or pandas.DataFrame
            Statistics about imputation by reading date (if calculated).

    Notes
    -----
    This function applies various imputation methods based on the nature of
    the missing data and the available information. It handles different
    scenarios such as gaps in data, zero jumps, and negative jumps.

    The function also calculates and returns statistics about the imputation
    process, which can be useful for quality assessment.

    Warnings
    --------
    - The function may modify the input DataFrame in-place.
    - Imputation methods may introduce bias or affect the variance of the data.
    - Large amounts of imputed data may significantly affect analysis results.
    """

    logging.info("Starting to impute cumulative column diffs (vectorized).")

    # Tariff splits are never filled: the combined root carries the physics.
    # One membership check where the fill list is constructed, so every
    # shell and direct caller agrees.
    cumulative_columns = without_tariff_splits(cumulative_columns)

    # Read once per call: the bound for every column is scaled from this.
    _cadence = load_column_cadence() if cadence is None else cadence

    if coverage_duty is not None and not coverage_duty.empty:
        cumulative_columns = order_columns_for_fill(
            cumulative_columns, coverage_duty
        )
        logging.info(
            "Fill order by coverage and duty: "
            + ", ".join(cumulative_columns)
        )

    # One constraint package per cadence in play, built on first use and shared
    # across the columns of this call.
    _packages: dict = {}

    # Cells earlier columns of this loop filled. A later column's bound can
    # rest on them, and a bound fed by the fill's own output is a softer fact
    # than one fed by measurements alone -- the infeasibility flags split on
    # exactly that.
    _imputed_so_far: dict = {}

    imputation_gap_stats = []

    for cum_col in cumulative_columns:
        logging.info(f"Starting {cum_col} vectorized imputation")

        temp_cols = ["gap_length", "cumulative_value_group"]

        diff_col = interval_column(cum_col)
        old_diff_col = f"{cum_col}OldDiff"
        is_imputed_col = f"{diff_col}_is_imputed"
        impute_type_col = f"{diff_col}_impute_type"
        avg_col = f"{diff_col}_avg"

        if diff_col not in df.columns:
            logging.info(f"[impute] {diff_col} not in chunk columns, skipping.")
            continue

        df[old_diff_col] = df[diff_col]

        drop_temp_cols(df, logLeftoverError=True)

        # The no-delivered-no-fill rule: a (household, column) with zero
        # delivered readings is never filled -- there is nothing to anchor or
        # scale from, and the only thing the fill could do is invent the whole
        # column from peer averages (the broadcast bug class the zero-prefill
        # era papered over). Delivered means the usable-readings predicate on
        # the interval OR its cumulative form -- a column of pipeline-written
        # values carries NOT_A_MEASUREMENT bits and does not count -- and the
        # seeded first-row zero of add_diff_columns is excluded, mirroring the
        # interval-only test in the fill. One mask per column per chunk; the
        # hot loop never sees it.
        _first_in_house = (
            df["HuisIdBSV"] != df["HuisIdBSV"].shift(1)
            if "HuisIdBSV" in df.columns
            else pd.Series(False, index=df.index)
        )
        _delivered_rows = (
            (usable_readings(df, diff_col) & ~_first_in_house)
            | usable_readings(df, cum_col)
        )
        if "HuisIdBSV" in df.columns:
            _never_fillable = ~(
                _delivered_rows.groupby(df["HuisIdBSV"]).transform("any")
            )
        else:
            _never_fillable = pd.Series(
                not bool(_delivered_rows.any()), index=df.index
            )

        avg_use = avg_col
        if constraints is not None:
            row_lo, row_hi, _bound_from_imputed = _row_bounds_per_cadence(
                df, diff_col, constraints, _cadence, _packages,
                imputed=_imputed_so_far,
                physical_model_by_project=physical_models,
                apparaat_sets_by_household=apparaat_sets,
            )

            # Judged before the fill writes into the column: a delivered value
            # outside what the rules narrow its row to, while inside the bound
            # its own column declares, is a signal and is never touched.
            delivered = usable_readings(df, diff_col)
            declared_lo = column_bound(df, diff_col, _cadence, bound_key="Min")
            declared_hi = column_bound(df, diff_col, _cadence, bound_key="Max")
            inside_declared = ~(
                (df[diff_col] < declared_lo).fillna(False)
                | (df[diff_col] > declared_hi).fillna(False)
            )
            outside_rule = (
                (df[diff_col] < row_lo).fillna(False)
                | (df[diff_col] > row_hi).fillna(False)
            )
            df["_outside_rule_bound"] = delivered & outside_rule & inside_declared

            # The fill and the threshold pass read the average through this
            # bounded copy; the project average column itself stays as written.
            bounded, moved, _empty = clip_to_bounds(df[avg_col], row_lo, row_hi)
            avg_use = f"{avg_col}_bounded"
            df[avg_use] = bounded
            df["_avg_was_clipped"] = moved

            # The fill clips what it writes to these and places the clipped
            # volume back within each anchored gap.
            df["_row_bound_lo"] = row_lo
            df["_row_bound_hi"] = row_hi
            df["_row_bound_empty"] = _empty
            df["_bound_from_imputed"] = _bound_from_imputed

            # An empty bound at a cell with nothing delivered: no value is
            # permitted, nothing will be written, and the cause splits on
            # what fed the bound -- measurements alone contradict each other
            # (hard), while a cell this run filled makes it a contradiction
            # we made (soft). A never-fillable household is excluded: no
            # fill was ever possible there, so an empty bound withholds
            # nothing and the flags would be noise on untouched rows.
            _no_value_permitted = _empty & ~delivered & ~_never_fillable
            df["_infeasible_hard"] = (
                _no_value_permitted & ~_bound_from_imputed.fillna(False)
            )
            df["_infeasible_soft"] = (
                _no_value_permitted & _bound_from_imputed.fillna(False)
            )

            # How far apart the contradicting bounds are, where they are
            # empty. Transient in-run state (the bounds exist only here), so
            # the gap accumulator carries its per-gap maximum; the
            # JITTER_CANDIDATE signal reads it against the assessment scale.
            df["_bound_overshoot"] = (row_lo - row_hi).where(_empty)

        if df[diff_col].isna().sum() == 0:
            logging.info(
                f"No values to impute in {diff_col}. Only checking thresholds.",
            )
            df[is_imputed_col] = False
            df[impute_type_col] = pd.Series(pd.NA, dtype="Int64", index=df.index)
            df["gap_length"] = pd.Series(pd.NA, dtype="Int64", index=df.index)
            df["cumulative_value_group"] = pd.Series(
                pd.NA,
                dtype="Int64",
                index=df.index,
            )
            logging.info("Apply thresholds to remove physically impossible outliers")
            df = apply_thresholds(
                df,
                lower_bound=thresholds_dict[diff_col]["Min"],
                upper_bound=fill_bound(
                    df, diff_col, _cadence, is_imputed_col=is_imputed_col
                ),
                detection_bound=column_bound(df, diff_col, _cadence),
                diff_col=diff_col,
                avg_col=avg_use,
                impute_type_col=impute_type_col,
                is_imputed_col=is_imputed_col,
            )

        else:
            logging.info(f"Defining {diff_col} gap groups")
            df = process_gap_and_cumulative_groups(
                df,
                diff_col=diff_col,
                cum_col=cum_col,
            )

            # Never-fillable households leave the gap structures entirely:
            # their rows are not gaps, so no route writes, no flag is set,
            # no volume row is collected, and the cells stay NA.
            if _never_fillable.any():
                _n_house = (
                    int(df.loc[_never_fillable, "HuisIdBSV"].nunique())
                    if "HuisIdBSV" in df.columns else 1
                )
                logging.info(
                    f"{diff_col}: {_n_house} household(s) deliver no usable "
                    f"reading in this column (interval or cumulative); their "
                    f"{int(_never_fillable.sum())} row(s) are not treated as "
                    f"gaps and stay NA."
                )
                df.loc[_never_fillable, "cumulative_value_group"] = pd.NA
                df.loc[_never_fillable, "gap_length"] = pd.NA

            logging.info(f"Imputing {diff_col}")
            df = process_imputation_vectorized(
                df,
                diff_col=diff_col,
                cum_col=cum_col,
                avg_col=avg_use,
                impute_type_col=impute_type_col,
                is_imputed_col=is_imputed_col,
                cadence=_cadence,
                volume_reports=volume_reports,
            )
            if volume_reports:
                volume_reports[-1]["column"] = diff_col
                # rule_expressions (the rules in force for this column) is a
                # package fact, not fill state: the post-processing step
                # derives it from the packaged constraints when it assembles
                # the full gap artifact from this accumulator.

            # Every cell inside a gap should carry a value once imputation
            # has run.
            inside_a_gap = df["cumulative_value_group"].notna()
            remaining_na = int((df[diff_col].isna() & inside_a_gap).sum())

            if remaining_na > 0:
                without_average = int(
                    (df[diff_col].isna() & inside_a_gap & df[avg_use].isna()).sum()
                )
                logging.error(
                    f"{remaining_na} cell(s) inside a gap are still missing in "
                    f"column {diff_col} after imputation, {without_average} of "
                    f"them where no project average was available. Check masks "
                    f"and imputation logic.",
                )

        if constraints is not None:
            _imputed_so_far[diff_col] = df[is_imputed_col].fillna(False).copy()

        logging.info(f"Calculating {diff_col} imputation gap stats")

        # `imputed` is the cells this run wrote a value into; `imputed_na` is the
        # cells inside a gap it left empty. Counting non-null impute_type counts
        # every gap cell instead, because a cell left empty still carries
        # GAP_NOT_FILLED -- which is what made `imputed` equal `missing`.
        _in_gap = df["gap_length"].notna()
        df["_gap_written"] = _in_gap & df[diff_col].notna()
        df["_gap_left_empty"] = _in_gap & df[diff_col].isna()

        # The household factor and the figures it is built from, so a factor can
        # be audited in the data it produced rather than reconstructed by proxy.
        # Each is constant over a (household, column), so "first" reads it.
        _factor_stats = {
            name: (name, "first")
            for name in (
                "house_impute_factor",
                "house_factor_is_fallback",
                "house_diff_median",
                "project_avg_median",
                "comparable_counts",
                "total_counts",
            )
            if name in df.columns
        }

        _grp = df.groupby([project_id_column, "HuisIdBSV"])
        # One C-level pass for all scalar stats; apply only for the methods list.
        _stats = _grp.agg(
            diff_col_total=(diff_col, "sum"),
            _cum_max=(cum_col, "max"),
            _cum_min=(cum_col, "min"),
            missing=("gap_length", "count"),
            imputed=("_gap_written", "sum"),
            imputed_na=("_gap_left_empty", "sum"),
            **_factor_stats,
        ).reset_index()
        _stats["cum_col_min_max_diff"] = _stats["_cum_max"] - _stats["_cum_min"]
        _stats["deviation"] = _stats["diff_col_total"] - _stats["cum_col_min_max_diff"]
        _stats["column"] = diff_col
        _stats["methods"] = (
            _grp[impute_type_col]
            .apply(lambda x: sorted(set(v for v in x if pd.notna(v))))
            .values
        )
        _stats.drop(columns=["_cum_max", "_cum_min"], inplace=True)
        imputation_gap_stats.append(_stats)

        drop_temp_cols(
            df,
            temp_cols=[
                "gap_length",
                "cumulative_value_group",
                "_gap_written",
                "_gap_left_empty",
                # Read into the stats just above, and per column, so they must
                # not survive into the next column's pass.
                "house_impute_factor",
                "house_factor_is_fallback",
                "house_diff_median",
                "project_avg_median",
                "comparable_counts",
                "total_counts",
                "comparable_to_impute_mask",
                "_outside_rule_bound",
                "_avg_was_clipped",
                "_row_bound_lo",
                "_row_bound_hi",
                "_row_bound_empty",
                "_infeasible_hard",
                "_infeasible_soft",
                "_bound_from_imputed",
                "_bound_overshoot",
                "_gap_unplaced",
                "_gap_redistributed",
                "_gap_uniform_residual",
                "_gap_unanchored_removed",
                f"{avg_col}_bounded",
            ],
        )

        drop_temp_cols(df, logLeftoverError=True)

    if not imputation_gap_stats:
        raise RuntimeError(
            "No columns were imputed -- all Diff columns were absent from this chunk. "
            "Check that prepare_diff_averages_for_impute has run and that cum_cols matches the source data."
        )
    imputation_gap_stats_df = pd.concat(imputation_gap_stats, ignore_index=True)
    imputation_gap_stats_df["bitwise_methods"] = methods_to_bitwise(
        imputation_gap_stats_df["methods"],
    )

    imputation_reading_date_stats_df = None

    return df, imputation_gap_stats_df, imputation_reading_date_stats_df


def drop_temp_cols(
    df,
    temp_cols=None,
    logLeftoverError=False,
):
    """
    Drop temporary columns from the DataFrame.

    This function removes specified temporary columns from the DataFrame.
    If no columns are specified, it drops a predefined set of temporary columns.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame from which to drop columns.
    temp_cols : list, optional
        A list of column names to drop. If None, a default set of temporary
        columns will be used.
    logLeftoverError : bool, optional
        If True, log an error message for any leftover columns to be removed.

    Notes
    -----
    This function modifies the DataFrame in-place.

    The default set of temporary columns includes various intermediate
    calculation columns used in the imputation process.

    Warnings
    --------
    - If logLeftoverError is True and there are columns to be dropped, an error
      message will be logged, which might indicate unintended remnants in the
      data processing pipeline.
    """

    if temp_cols is None:
        temp_cols = [
            "gap_start",
            "gap_group",
            "gap_jump_is_na_mask",
            "gap_jump",
            "house_impute_factor",
            "avg_na",
            "impute_jump",
            "impute_na",
            "impute_values",
            "impute_na_ratio",
            "cum_value_encountered",
            "prev_cum_value",
            "end_cum_value",
            "impute_type_old",
            "gap_length",
            "cumulative_value_group",
            "no_diff_mask"
            ]


    cols_to_drop = [col for col in temp_cols if col in df.columns]

    if logLeftoverError and len(cols_to_drop) > 0:
        logging.error(
            f"There are some leftover columns to remove from the code: {cols_to_drop}",
        )

    df.drop(columns=cols_to_drop, inplace=True)


def process_gap_and_cumulative_groups(df, diff_col, cum_col):
    """
    Process gap and cumulative value groups in the DataFrame.

    This function identifies gaps in the data, creates gap groups, and
    establishes cumulative value groups based on the presence of NA values
    and transitions between households.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to process.
    diff_col : str
        The name of the difference column to analyze for gaps.
    cum_col : str
        The name of the cumulative column to use for value grouping.

    Returns
    -------
    pandas.DataFrame
        The processed DataFrame with added columns for gap and cumulative
        value grouping.

    Notes
    -----
    This function adds several temporary columns to the DataFrame:
    - 'gap_start': Identifies the start of a diff column gap or transition between households.
    - 'gap_group': Groups consecutive NA values in diff columns.
    - 'cum_value_encountered': Marks where a non-NA value is encountered in cumulative column.
    - 'cumulative_value_group': Groups gaps based on cumulative values.
    - 'gap_length': The length of each gap group.

    These columns are crucial for the subsequent imputation process. It returns only for further imputation:
    - 'cumulative_value_group'
    - 'gap_length'

    Warnings
    --------
    - This function modifies the input DataFrame in-place.
    - The added columns should be handled carefully in subsequent processing steps.
    """
    temp_cols = ["gap_start", "gap_group", "cum_value_encountered"]

    # Step 1: Identify NA values in diff_col
    is_na_mask = df[diff_col].isna()

    # Step 2: Identify gap start only for NA values and transitions between households
    df["gap_start"] = (is_na_mask & (~is_na_mask.shift(1, fill_value=False))) | (
        (df["HuisIdBSV"] != df["HuisIdBSV"].shift(1)) & is_na_mask
    )

    # Step 3: Create gap groups by cumulative sum, but now ensure it doesn't increment unnecessarily at house boundaries
    df["gap_group"] = df["gap_start"].cumsum()

    # Step 4: Ensure non-NA values are not included in any gap group
    df["gap_group"] = df["gap_group"].mask(~is_na_mask, pd.NA)

    # Step 5: Mark the end of a group where a non-NA value is encountered in cum_col
    df["cum_value_encountered"] = df[cum_col].notna() & is_na_mask

    # Step 6: Adjust cumulative group logic to handle gaps without cumulative values
    df["cumulative_value_group"] = (
        df["cum_value_encountered"].shift(1, fill_value=False) | df["gap_start"]
    ).cumsum()

    # Step 7: Ensure that the group continues when there is no cumulative value in the gap
    df["cumulative_value_group"] = df["cumulative_value_group"].ffill()

    # Step 8: Ensure non-NA values in diff_col don't belong to a group

    # Combine all conditions
    exclude_mask = ~is_na_mask

    df["cumulative_value_group"] = df["cumulative_value_group"].mask(
        exclude_mask,
        pd.NA,
    )

    # Step 7: Add the count of records in each group consistently for all rows in the group
    df["gap_length"] = df.groupby("cumulative_value_group").transform("size")

    drop_temp_cols(df, temp_cols=temp_cols)

    return df


def process_imputation_vectorized(
    df,
    diff_col,
    cum_col,
    avg_col,
    impute_type_col,
    is_imputed_col,
    cadence=None,
    volume_reports=None,
):
    """
    Perform vectorized imputation on the DataFrame.

    This function applies various imputation methods to fill missing values
    in the difference column based on cumulative values and average differences.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to impute.
    diff_col : str
        The name of the difference column to impute.
    cum_col : str
        The name of the cumulative column used for imputation.
    avg_col : str
        The name of the column containing average differences.
    impute_type_col : str
        The name of the column to store imputation type.
    is_imputed_col : str
        The name of the column to indicate whether a value is imputed.
    cadence : pandas.DataFrame, optional
        The per-column cadence table from
        :func:`etdtransform.bounds.load_column_cadence`, used to judge a
        delivered value against the interval it covers. Without it every value
        is judged against the grid threshold.
    volume_reports : list, optional
        When given, one frame per call is appended: per gap, what the routes
        wrote, what the bounds moved, and what could not be placed. Collected
        before the threshold pass, which accounts for itself separately.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with imputed values and additional columns indicating
        imputation types and statistics.

    Notes
    -----
    This function applies several imputation methods for diff columns:
    - Filling with zeros for flat or near-zero gaps
    - Linear filling for positive gaps with near-zero impute jumps
    - Scaled impute value filling for positive gaps with positive impute jumps
    - Handling cases with no gap jump (e.g., at the start or end of the dataset)

    The function also applies thresholds to remove physically impossible outliers.

    Warnings
    --------
    - This function modifies the input DataFrame in-place.
    - The imputation process may introduce bias, especially in cases with large gaps
      or when a significant portion of the data is imputed.
    - The function assumes that the input data has been properly prepared and sorted.
    """
    def setup_prev_value_columns(df, cum_col):
        # Shift by one to get the immediate previous time step value before the group and make sure no other time step is there
        df["prev_cum_value"] = df[cum_col].shift(1)

        first_in_group_mask = (
            df["cumulative_value_group"] != df["cumulative_value_group"].shift(1)
        ) & ~(df["cumulative_value_group"].isna())

        first_in_house = df["HuisIdBSV"] != df["HuisIdBSV"].shift(1)

        # remove all values that are not the first in the group or that are from another household
        df.loc[
            ~first_in_group_mask | df["cumulative_value_group"].isna() | first_in_house,
            "prev_cum_value",
        ] = pd.NA

        # Remove negative previous values, as they are not valid for imputation
        df.loc[df["prev_cum_value"] < 0, "prev_cum_value"] = pd.NA

        # fill for each group
        df["prev_cum_value"] = df.groupby("cumulative_value_group")[
            "prev_cum_value"
        ].ffill()

        return df

    def ensure_Float64(df, columns=["CumCol", "CumColDiff", "diff_avg"]):
        for col in columns:
            if df[col].dtype != "Float64":
                logging.warning(
                    f'{col} dtype is "{df[col].dtype}". Attempting to convert.',
                )
                try:
                    # Attempt to convert the column to Float64
                    df[col] = df[col].astype("Float64")
                    logging.info(f"{col} successfully converted to Float64.")
                except Exception as e:
                    logging.error(f"Failed to convert {col} to Float64: {e}")
                    raise e
        return df

    def setup_gap_stats(df):
        df["gap_jump"] = df["end_cum_value"] - df["prev_cum_value"]
        df["gap_jump_is_na_mask"] = (
            df["gap_jump"].isna() & df["cumulative_value_group"].notna()
        )

        # df['no_diff_mask'] = df['cumulative_value_group'].notna() & (df['prev_cum_value'].isna() | df['end_cum_value'].isna() | (df['gap_jump'] <= 0))

        return df

    def setup_impute_columns(df, avg_col, impute_type_col, is_imputed_col):
        # add impute jump and tracking of missing values
        df["avg_na"] = df[avg_col].isna()
        df["impute_na"] = df.groupby("cumulative_value_group")["avg_na"].transform(
            "sum",
        )
        df["impute_values"] = df[avg_col]  # pd.NA where avg unavailable; do not substitute 0

        if (df["impute_values"] < 0).any():
            raise Exception("Negative impute jump - not allowed - check averages")

        df["impute_na_ratio"] = df["impute_na"] / df["gap_length"]
        df["impute_jump"] = df.groupby("cumulative_value_group")[
            "impute_values"
        ].transform("sum")

        # How many cells of the gap the average has a value for. The fill
        # routes on this count, not on the summed average.
        df["impute_present"] = df["gap_length"] - df["impute_na"]

        # Cells of a partly-covered gap that have no average take the gap's own
        # mean. Both terms are reductions already computed above.
        partial = (df["impute_na"] > 0) & (df["impute_present"] > 0)
        gap_mean = df["impute_jump"] / df["impute_present"]
        df["impute_values"] = df["impute_values"].where(
            ~(partial & df["avg_na"]), gap_mean
        )
        # Recomputed after the substitution so it matches impute_values.
        df["impute_jump"] = df["impute_jump"].where(
            ~partial, df["impute_jump"] * df["gap_length"] / df["impute_present"]
        )

        # calculate cumulative total for household for scaling the averages (optional)
        df[is_imputed_col] = False
        df[impute_type_col] = pd.Series(pd.NA, dtype="Int64", index=df.index)

        return df

    def setup_proportional_adjustment_to_impute(df, diff_col, avg_col):
        """The household's level relative to its project's average.

        The median of the household's own column over the median of the project
        average for the same rows. 1.0 matches the average, 2.0 is twice it, and
        it multiplies the average to bring it to this household's level.

        A ratio of sums was used before, over a mask that admitted any household
        present in the project. A household without the device contributes
        legitimate zeros to the project average, so the denominator measured how
        many peers lacked the device rather than what the device consumes: on
        ElektriciteitsgebruikBoilervatDiff, one household of 120 had the device
        and its factor came out at 97.4x. Excluding zeros is what fixes that,
        and once they are gone mean and median agree, so the median is taken
        because it is the more robust of the two.
        """
        # Both medians are taken over the same rows, so the ratio compares like
        # with like. A zero is a real measurement but not evidence of the
        # household's level when the device is idle or absent, and a reading the
        # mapping stage disqualified is not a measurement at all.
        df["comparable_to_impute_mask"] = (
            usable_readings(df, diff_col)
            & ~df["avg_na"]
            & (df[diff_col] > 0)
            & (df[avg_col] > 0)
        )

        df["house_diff_median"] = (
            df[diff_col]
            .where(df["comparable_to_impute_mask"])
            .groupby(df["HuisIdBSV"])
            .transform("median")
        )
        df["project_avg_median"] = (
            df[avg_col]
            .where(df["comparable_to_impute_mask"])
            .groupby(df["HuisIdBSV"])
            .transform("median")
        )

        df["comparable_counts"] = df.groupby("HuisIdBSV")[
            "comparable_to_impute_mask"
        ].transform("sum")
        df["total_counts"] = df.groupby("HuisIdBSV")[diff_col].transform("size")
        not_enough_comparable = df["comparable_counts"] <= (df["total_counts"] / 2)

        # Either median at zero leaves nothing to compare, and dividing by it
        # would give an infinite factor rather than no answer.
        degenerate = (
            df["house_diff_median"].isna()
            | df["project_avg_median"].isna()
            | (df["house_diff_median"] == 0)
            | (df["project_avg_median"] == 0)
        )

        factor = df["house_diff_median"] / df["project_avg_median"]
        # True where the factor is the neutral 1.0 rather than a value computed
        # from the household's own rows. The fill branches read it to choose
        # between the scaled and unscaled impute types.
        df["house_factor_is_fallback"] = degenerate | not_enough_comparable
        df["house_impute_factor"] = factor.mask(
            df["house_factor_is_fallback"], 1.0
        ).astype("Float64")

        return df

    temp_cols = [
        "gap_jump_is_na_mask",
        "gap_jump",
        # house_impute_factor and the figures behind it are deliberately absent:
        # the caller reads them into the gap statistics and drops them there, so
        # dropping them here would put the factor beyond inspection again.
        "avg_na",
        "impute_jump",
        "impute_na",
        "impute_values",
        "impute_na_ratio",
        "cum_value_encountered",
        "prev_cum_value",
        "end_cum_value",
        "impute_type_old",
    ]

    logging.info(f"Ensuring {[cum_col, diff_col, avg_col]} are Float64")

    df = ensure_Float64(df, columns=[cum_col, diff_col, avg_col])

    logging.info("Setting up previous value column")
    df = setup_prev_value_columns(df, cum_col)

    df["end_cum_value"] = df.groupby("cumulative_value_group")[cum_col].transform(
        "last",
    )
    df.loc[df["end_cum_value"] < 0, "end_cum_value"] = pd.NA

    logging.info("Setting up gap calculations")
    df = setup_gap_stats(df)

    logging.info("Setting up impute columns")
    df = setup_impute_columns(
        df,
        avg_col=avg_col,
        impute_type_col=impute_type_col,
        is_imputed_col=is_imputed_col,
    )

    logging.info("Setting up proportional adjustment to mpute")
    df = setup_proportional_adjustment_to_impute(df, diff_col=diff_col, avg_col=avg_col)

    def impute_with_gap_jump(df, diff_col, is_imputed_col, impute_type_col):
        has_gap_jump_mask = (
            ~df["gap_jump_is_na_mask"] & df["cumulative_value_group"].notna()
        )

        # An empty row bound permits no value, so no route may write there.
        # The shaped routes are already stopped by their blanked average; the
        # zero and uniform routes need the exclusion stated. The cell then
        # falls through to the GAP_NOT_FILLED sweep, and the infeasibility flags
        # say why.
        permitted = (
            ~df["_row_bound_empty"].fillna(False)
            if "_row_bound_empty" in df.columns
            else True
        )

        ## if has gap jump
        # negative gap jump
        flat_gap_jump_mask = has_gap_jump_mask & (df["gap_jump"] < 0) & permitted
        df.loc[flat_gap_jump_mask, is_imputed_col] = True
        df.loc[flat_gap_jump_mask, diff_col] = 0
        # First write, and the routes are exclusive, so this assigns rather than OR-s.
        df.loc[flat_gap_jump_mask, impute_type_col] = (
            ImputeType.NEGATIVE_GAP_JUMP.value | ImputeType.ZERO_FILL.value
        )

        # gap jump near zero - fill with zeros
        flat_gap_jump_mask = (
            has_gap_jump_mask
            & (df["gap_jump"] >= 0)
            & (df["gap_jump"] < 1e-8)
            & permitted
        )
        df.loc[flat_gap_jump_mask, is_imputed_col] = True
        df.loc[flat_gap_jump_mask, diff_col] = 0
        # First write, and the routes are exclusive, so this assigns rather than OR-s.
        df.loc[flat_gap_jump_mask, impute_type_col] = (
            ImputeType.NEAR_ZERO_GAP_JUMP.value | ImputeType.ZERO_FILL.value
        )

        # No average for any cell of the gap, or an average that is present but
        # everywhere zero: no shape to distribute by, so the jump is spread
        # evenly.
        no_shape = (df["impute_present"] <= 0) | (df["impute_jump"] < 1e-8)
        positive_gap_linear_mask = (
            has_gap_jump_mask & (df["gap_jump"] >= 1e-8) & no_shape & permitted
        )
        df.loc[positive_gap_linear_mask, is_imputed_col] = True
        df.loc[positive_gap_linear_mask, diff_col] = round(
            df["gap_jump"] / df["gap_length"],
            10,
        )
        # First write, and the routes are exclusive, so this assigns rather than OR-s.
        df.loc[positive_gap_linear_mask, impute_type_col] = ImputeType.UNIFORM_FILL.value

        # A shape exists for at least one cell of the gap, so the jump is
        # distributed by it. Cells without an average carry the gap's own mean
        # by this point.
        positive_gap_scaled_mask = (
            has_gap_jump_mask & (df["gap_jump"] >= 1e-8) & ~no_shape
        )
        # permitted again: a cell whose average the bound blanked can get it
        # back as the gap's own mean, which must not resurrect the write.
        _can_scaled = (
            positive_gap_scaled_mask & df["impute_values"].notna() & permitted
        )
        df.loc[_can_scaled, is_imputed_col] = True
        df.loc[_can_scaled, diff_col] = round(
            df.loc[_can_scaled, "impute_values"]
            * (
                df.loc[_can_scaled, "gap_jump"]
                / df.loc[_can_scaled, "impute_jump"]
            ),
            10,
        )
        # First write, and the routes are exclusive, so this assigns rather than OR-s.
        df.loc[_can_scaled, impute_type_col] = ImputeType.SCALED_FILL.value

        return df

    logging.info("Impute with gap jump")
    df = impute_with_gap_jump(
        df,
        diff_col=diff_col,
        is_imputed_col=is_imputed_col,
        impute_type_col=impute_type_col,
    )
    # What the anchored routes wrote, per gap, before any bound acts on it.
    _routed_anchored = None
    if volume_reports is not None:
        _routed_anchored = df.groupby("cumulative_value_group")[diff_col].sum(
            min_count=1
        )
    df = _allocate_gap_fills(df, diff_col, impute_type_col)

    def impute_wo_gap_jump(df, diff_col, impute_type_col, is_imputed_col):
        wo_gap_jump_mask = (
            df["gap_jump_is_na_mask"] & df["cumulative_value_group"].notna()
        )

        # An empty row bound permits no value, so no route may write there;
        # the cell falls through to the GAP_NOT_FILLED sweep and the
        # infeasibility flags say why.
        permitted = (
            ~df["_row_bound_empty"].fillna(False)
            if "_row_bound_empty" in df.columns
            else True
        )

        ###
        ## else no gap jump:
        # no starting value for gap and no end value - throw an exception
        nogpjump_no_start_no_end_value_mask = (
            wo_gap_jump_mask & df["end_cum_value"].isna() & df["prev_cum_value"].isna()
        )
        if nogpjump_no_start_no_end_value_mask.any():
            # add_diff_columns seeds row 0 of each household with 0, so that row
            # is excluded when testing whether the household delivers intervals.
            _first_in_house = df["HuisIdBSV"] != df["HuisIdBSV"].shift(1)
            house_no_diff_values = (
                (df[diff_col].notna() & ~_first_in_house)
                .groupby(df["HuisIdBSV"])
                .transform("sum")
                == 0
            )

            # Neither end of the gap has a cumulative value. The household
            # delivers this variable as intervals only when the interval column
            # holds values and the cumulative does not.
            _interval_only = (
                nogpjump_no_start_no_end_value_mask
                & ~house_no_diff_values
                & df["impute_values"].notna()
                & permitted
            )
            # No cumulative reading bounds the gap, so house_impute_factor is
            # the only scale available. Fill where it was computed from the
            # household's own rows; leave the cells empty where it fell back.
            _fallback = df["house_factor_is_fallback"]
            _fill_io = _interval_only & ~_fallback
            _skip_io = _interval_only & _fallback

            if _fill_io.any():
                logging.info(
                    f"{diff_col}: filling {int(_fill_io.sum())} interval(s) from "
                    f"the project average scaled to the household -- the variable "
                    f"is delivered as intervals only for this household."
                )
                df.loc[_fill_io, is_imputed_col] = True
                df.loc[_fill_io, diff_col] = (
                    df.loc[_fill_io, "impute_values"]
                    * df.loc[_fill_io, "house_impute_factor"]
                )
                # First write, and the routes are exclusive, so this assigns rather than OR-s.
                df.loc[_fill_io, impute_type_col] = (
                    ImputeType.NO_CUMULATIVE_VALUE.value
                    | ImputeType.HOUSEHOLD_SCALED_AVERAGE.value
                )

            if _skip_io.any():
                logging.warning(
                    f"{diff_col}: leaving {int(_skip_io.sum())} interval(s) "
                    f"unfilled -- no cumulative reading bounds them and the "
                    f"household factor has too few comparable rows to scale with."
                )
                # First write, and the routes are exclusive, so this assigns rather than OR-s.
                df.loc[_skip_io, impute_type_col] = (
                    ImputeType.NO_CUMULATIVE_VALUE.value
                    | ImputeType.GAP_NOT_FILLED.value
                )

            _truly_empty = nogpjump_no_start_no_end_value_mask & house_no_diff_values
            if _truly_empty.any():
                logging.error(
                    f"{diff_col}: no cumulative and no interval values for "
                    f"{int(df.loc[_truly_empty, 'HuisIdBSV'].nunique())} household(s). "
                    f"Nothing to impute from."
                )

        ### has end value but no start value for gap
        nogpjump_has_end_value_mask = (
            wo_gap_jump_mask & ~df["end_cum_value"].isna() & df["prev_cum_value"].isna()
        )

        #### has end value is 0 - fill with 0 - type 6
        nogpjump_has_end_value_zero_mask = nogpjump_has_end_value_mask & (
            df["end_cum_value"] < 1e-8
        )
        # DISABLED, deliberately reversible. Filling zeros here says the meter
        # stood at zero for the whole gap, which follows only if it had never
        # advanced. With no opening reading there is nothing to support that:
        # the cumulative is pd.NA, not zero. Left in place so the previous
        # behaviour can be restored by uncommenting.
        #
        # df.loc[nogpjump_has_end_value_zero_mask, is_imputed_col] = True
        # df.loc[nogpjump_has_end_value_zero_mask, diff_col] = 0
        # df.loc[nogpjump_has_end_value_zero_mask, impute_type_col] = (
        #     ImputeType.ZERO_END_VALUE.value
        #     | ImputeType.ONLY_CLOSING_READING.value
        #     | ImputeType.ZERO_FILL.value
        # )
        if nogpjump_has_end_value_zero_mask.any():
            # First write, and the routes are exclusive, so this assigns rather than OR-s.
            df.loc[nogpjump_has_end_value_zero_mask, impute_type_col] = (
                ImputeType.ZERO_END_VALUE.value
                | ImputeType.ONLY_CLOSING_READING.value
                | ImputeType.GAP_NOT_FILLED.value
            )

        #### end value > 0 - fill with impute values - type 7 (only when avg available)
        nogpjump_has_end_value_positive_mask = nogpjump_has_end_value_mask & (
            df["end_cum_value"] > 1e-8
        )
        # Only a closing reading bounds this gap, which is the mirror of the
        # opening-reading case below and takes the same treatment: scale the
        # average to the household where the factor rests on enough of the
        # series, and write nothing where it does not.
        _can_fill_ev = (
            nogpjump_has_end_value_positive_mask
            & df["impute_values"].notna()
            & permitted
        )
        _ev_fallback = df["house_factor_is_fallback"]
        _fill_ev = _can_fill_ev & ~_ev_fallback
        _skip_ev = _can_fill_ev & _ev_fallback

        df.loc[_fill_ev, is_imputed_col] = True
        df.loc[_fill_ev, diff_col] = (
            df.loc[_fill_ev, "impute_values"]
            * df.loc[_fill_ev, "house_impute_factor"]
        )
        # First write, and the routes are exclusive, so this assigns rather than OR-s.
        df.loc[_fill_ev, impute_type_col] = (
            ImputeType.POSITIVE_END_VALUE.value
            | ImputeType.ONLY_CLOSING_READING.value
            | ImputeType.HOUSEHOLD_SCALED_AVERAGE.value
        )

        if _skip_ev.any():
            logging.warning(
                f"{diff_col}: leaving {int(_skip_ev.sum())} interval(s) unfilled "
                f"-- the gap has no opening reading and the household factor has "
                f"too few comparable rows to scale with."
            )
            # First write, and the routes are exclusive, so this assigns rather than OR-s.
            df.loc[_skip_ev, impute_type_col] = (
                ImputeType.POSITIVE_END_VALUE.value
                | ImputeType.ONLY_CLOSING_READING.value
                | ImputeType.GAP_NOT_FILLED.value
            )

        #### end value < 0 - raise exception
        if (df["end_cum_value"] < 0).any():
            raise Exception(
                "Negative next value at end of gap - that is not supposed to happen!",
            )

        ### start value but no end value for gap - fill with impute values (only when avg available)
        nogpjump_has_start_value_mask = (
            wo_gap_jump_mask & df["end_cum_value"].isna() & ~df["prev_cum_value"].isna()
        )
        # The gap opens on a reading and never closes, so again only
        # house_impute_factor scales it. Same rule as the no-cumulative case.
        _can_fill_sv = (
            nogpjump_has_start_value_mask
            & df["impute_values"].notna()
            & permitted
        )
        _sv_fallback = df["house_factor_is_fallback"]
        _fill_sv = _can_fill_sv & ~_sv_fallback
        _skip_sv = _can_fill_sv & _sv_fallback

        df.loc[_fill_sv, is_imputed_col] = True
        df.loc[_fill_sv, diff_col] = (
            df.loc[_fill_sv, "impute_values"]
            * df.loc[_fill_sv, "house_impute_factor"]
        )
        # First write, and the routes are exclusive, so this assigns rather than OR-s.
        df.loc[_fill_sv, impute_type_col] = (
            ImputeType.NO_END_VALUE.value
            | ImputeType.ONLY_OPENING_READING.value
            | ImputeType.HOUSEHOLD_SCALED_AVERAGE.value
        )

        if _skip_sv.any():
            logging.warning(
                f"{diff_col}: leaving {int(_skip_sv.sum())} interval(s) unfilled "
                f"-- the gap has no closing reading and the household factor has "
                f"too few comparable rows to scale with."
            )
            # First write, and the routes are exclusive, so this assigns rather than OR-s.
            df.loc[_skip_sv, impute_type_col] = (
                ImputeType.NO_END_VALUE.value
                | ImputeType.ONLY_OPENING_READING.value
                | ImputeType.GAP_NOT_FILLED.value
            )

        return df

    logging.info("Impute without gap jump")
    df = impute_wo_gap_jump(
        df,
        diff_col=diff_col,
        impute_type_col=impute_type_col,
        is_imputed_col=is_imputed_col,
    )

    # The unanchored routes' totals, before their clip; anchored gaps keep the
    # pre-allocation figure captured above.
    _routed_by_gap = None
    if volume_reports is not None:
        _routed_unanchored = df.groupby("cumulative_value_group")[
            diff_col
        ].sum(min_count=1)
        _routed_by_gap = _routed_anchored.fillna(_routed_unanchored)

    if "_row_bound_lo" in df.columns:
        # An unanchored fill has no measured total to give volume back to, so
        # the bound simply clips what was written: clip, mark, and move on.
        _findings = df[impute_type_col].fillna(ImputeType.NONE.value)
        _unanchored = (
            df[is_imputed_col].fillna(False)
            & (
                (
                    _findings
                    & (
                        ImputeType.ONLY_CLOSING_READING.value
                        | ImputeType.ONLY_OPENING_READING.value
                        | ImputeType.NO_CUMULATIVE_VALUE.value
                    )
                )
                != 0
            )
            & ((_findings & ImputeType.GAP_FILLED.value) != 0)
        )
        if _unanchored.any():
            _bounded, _moved, _ = clip_to_bounds(
                df[diff_col].where(_unanchored),
                df["_row_bound_lo"],
                df["_row_bound_hi"],
            )
            _cut = _unanchored & _moved
            # Signed: positive volume was cut away, negative was raised in.
            # There is no give-back, so this is what the report carries.
            _removed = (df[diff_col] - _bounded).where(_cut, 0.0).fillna(0.0)
            df["_gap_unanchored_removed"] = _removed.groupby(
                df["cumulative_value_group"]
            ).transform("sum")
            df.loc[_cut, diff_col] = _bounded[_cut]
            df.loc[_cut, impute_type_col] = (
                df[impute_type_col].fillna(ImputeType.NONE.value)
                | ImputeType.CLIPPED.value
            )

    if "_row_bound_lo" in df.columns:
        # A gap whose measured total cannot be met any more: the allocation
        # left volume unplaced, or a cell of it permits no value at all. The
        # condition belongs to the whole gap -- the allocation already drove
        # every cell to a bound, so no subset is at fault -- and flagging all
        # of them makes it visible on whichever row a reader lands.
        _in_gap = df["cumulative_value_group"].notna()
        _anchored_gap = (
            _in_gap & ~df["gap_jump_is_na_mask"] & (df["gap_jump"] >= 1e-8)
        )
        _unmet = pd.Series(False, index=df.index)
        if "_gap_unplaced" in df.columns:
            _unmet = df["_gap_unplaced"].fillna(0.0) != 0
        if "_row_bound_empty" in df.columns:
            _unmet = _unmet | (
                df["_row_bound_empty"].fillna(False) & _anchored_gap
            )
        if _unmet.any():
            _gap_unmet = (
                _unmet.groupby(df["cumulative_value_group"]).transform("max")
                .fillna(False)
                & _in_gap
            )
            df.loc[_gap_unmet, impute_type_col] = (
                df[impute_type_col].fillna(ImputeType.NONE.value)
                | ImputeType.INFEASIBLE_GAP.value
            )

    # Gap cells no branch above wrote to. Row 0 of a household bounds no gap,
    # so it is excluded.
    _first_in_house = df["HuisIdBSV"] != df["HuisIdBSV"].shift(1)
    _unfilled = (
        df[diff_col].isna()
        & df["cumulative_value_group"].notna()
        & ~_first_in_house
    )
    if _unfilled.any():
        df.loc[_unfilled, impute_type_col] = (
            df[impute_type_col].fillna(ImputeType.NONE.value)
            | ImputeType.GAP_NOT_FILLED.value
        )
        logging.warning(
            f"{diff_col}: {int(_unfilled.sum())} cell(s) inside a gap were not "
            f"filled and carry GAP_NOT_FILLED."
        )

    if "_row_bound_lo" in df.columns:
        # Signals recorded after the routes ran, so they OR onto whatever the
        # fill wrote -- and BEFORE the volume report collects the gap bits,
        # so the report sees the complete vocabulary.
        _reads_avg = (
            ImputeType.SCALED_FILL
            | ImputeType.UNSCALED_AVERAGE
            | ImputeType.HOUSEHOLD_SCALED_AVERAGE
        )
        _used_avg = (
            df[impute_type_col].fillna(ImputeType.NONE.value)
            & _reads_avg.value
        ) != 0
        _clipped_used = df["_avg_was_clipped"] & _used_avg
        df.loc[_clipped_used, impute_type_col] = (
            df[impute_type_col].fillna(ImputeType.NONE.value)
            | ImputeType.CLIPPED_AVERAGE.value
        )
        df.loc[df["_outside_rule_bound"], impute_type_col] = (
            df[impute_type_col].fillna(ImputeType.NONE.value)
            | ImputeType.DELIVERED_OUTSIDE_RULE_BOUND.value
        )
        df.loc[df["_infeasible_hard"], impute_type_col] = (
            df[impute_type_col].fillna(ImputeType.NONE.value)
            | ImputeType.INFEASIBLE_BOUNDS_HARD.value
        )
        df.loc[df["_infeasible_soft"], impute_type_col] = (
            df[impute_type_col].fillna(ImputeType.NONE.value)
            | ImputeType.INFEASIBLE_BOUNDS_SOFT.value
        )
        # A bound resting on a cell an earlier column's fill wrote inherits
        # that fill. Accepted, and recorded where the bound was in play.
        _bound_in_play = (
            df[is_imputed_col].fillna(False)
            | df["_outside_rule_bound"]
            | df["_infeasible_hard"]
            | df["_infeasible_soft"]
        )
        _inherited = df["_bound_from_imputed"].fillna(False) & _bound_in_play
        df.loc[_inherited, impute_type_col] = (
            df[impute_type_col].fillna(ImputeType.NONE.value)
            | ImputeType.BOUND_FROM_IMPUTED.value
        )

        # An empty bound whose contradiction is at metering resolution: a
        # jitter CANDIDATE, marked for assessment against the announced
        # scale. Signal only -- what fills or stays unfilled is unchanged --
        # and OR-ed here, before the accumulator collects the gap bits.
        if "_bound_overshoot" in df.columns:
            _jitter = (
                (
                    df["_infeasible_hard"].fillna(False)
                    | df["_infeasible_soft"].fillna(False)
                )
                & (
                    df["_bound_overshoot"]
                    <= _jitter_assessment_scale(diff_col)
                ).fillna(False)
            )
            df.loc[_jitter, impute_type_col] = (
                df[impute_type_col].fillna(ImputeType.NONE.value)
                | ImputeType.JITTER_CANDIDATE.value
            )

    if volume_reports is not None:
        volume_reports.append(
            _gap_volume_report(df, diff_col, impute_type_col, _routed_by_gap)
        )

    logging.info("Apply thresholds to remove physically impossible outliers")
    df = apply_thresholds(
        df,
        lower_bound=thresholds_dict[diff_col]["Min"],
        upper_bound=fill_bound(
            df, diff_col, cadence, is_imputed_col=is_imputed_col
        ),
        detection_bound=column_bound(df, diff_col, cadence),
        diff_col=diff_col,
        avg_col=avg_col,
        impute_type_col=impute_type_col,
        is_imputed_col=is_imputed_col,
    )

    if "_avg_was_clipped" in df.columns:
        # The threshold pass substitutes the bounded average after the volume
        # report collected; a substituted cell whose average was clipped
        # still carries the signal on the cell itself.
        _sub_used = (
            df[impute_type_col].fillna(ImputeType.NONE.value)
            & ImputeType.UNSCALED_AVERAGE.value
        ) != 0
        _sub_clipped = df["_avg_was_clipped"] & _sub_used
        df.loc[_sub_clipped, impute_type_col] = (
            df[impute_type_col].fillna(ImputeType.NONE.value)
            | ImputeType.CLIPPED_AVERAGE.value
        )

    # Households without a cumulative sum above 0 or all cum_col values are NA
    house_no_cum_sum = (
        df.groupby("HuisIdBSV")[cum_col].transform("sum").fillna(0) <= 0
    ) | (df.groupby("HuisIdBSV")[cum_col].transform("count") == 0)

    # Households where max - min cumulative sum is not > 0
    houses_cum_min_max = (
        df[cum_col].groupby(df["HuisIdBSV"]).transform("max")
        - df[cum_col].groupby(df["HuisIdBSV"]).transform("min")
    ).fillna(0) <= 0

    drop_temp_cols(df, temp_cols=temp_cols)

    return df


