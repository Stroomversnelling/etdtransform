import logging
import os
from math import floor, isclose, log10

import numpy as np
import pandas as pd
from etdmap.data_model import (
    complement_column,
    cumulative_column,
    interval_column,
    monotonic_interval_columns,
    scale_bounds,
    thresholds_dict,
)

from etdmap.flags import MappingFlag, flag_column

from etdtransform.presence import (
    MIN_WINDOW_COVERAGE,
    SUFFICIENCY_FRACTION,
    SUFFICIENCY_SHAPE,
    WINDOW_MINUTES,
    apply_presence_filter,
    window_minutes_for_column,
    window_presence_frame,
)

from etdtransform.bounds import (
    cadence_minutes_by_household,
    column_bound,
    column_bound_by_household,
    dataset_grid_minutes,
    load_column_cadence,
    resample_to_grid,
)

import etdtransform
from etdtransform.vectorized_impute import impute_and_normalize, without_tariff_splits

#: Findings that disqualify a value from shaping a project average: it was
#: written by the pipeline rather than measured, or it was measured over a
#: window rather than at the interval it sits on.
_NOT_MEASURED_HERE = MappingFlag.NOT_A_MEASUREMENT.value


def assert_no_negative_interval(column, smallest) -> None:
    """Stop if *smallest*, the minimum of *column*, is below zero.

    Mapping classifies and blanks a decreasing cumulative reading, so a negative
    interval reaching the average means those findings were not applied.
    """
    if smallest is None or pd.isna(smallest) or smallest >= 0:
        return
    raise ValueError(
        f"{column}: negative interval value(s) present, smallest {smallest}. "
        f"Mapping is expected to blank these before the average sees them."
    )


def assert_averageable(diff_columns) -> None:
    """Refuse to average an interval column whose cumulative source may decrease.

    The average and the gap fill both treat an interval as a volume: they sum
    it, scale it and redistribute it. That holds only where the variable
    accumulates, which the data model declares on the cumulative variable as
    Monotoon and etdmap resolves to the interval columns through
    ComplementVariabele.

    A column whose complement cannot be resolved is refused as well: an
    undeclared link is not a declaration that the variable accumulates.

    Metadata only, once per call: this costs nothing per row.
    """
    offenders = [
        column for column in diff_columns
        if column not in monotonic_interval_columns
    ]

    if offenders:
        listed = "\n  ".join(
            f"{column}: not the interval column of a variable declared Monotoon"
            for column in offenders
        )
        raise ValueError(
            f"These columns cannot be used for a project average:\n  {listed}\n"
            f"Check the data model: Monotoon and ComplementVariabele.\n"
        )


def assert_cumulative_interval_pairs_complete(columns) -> None:
    """Refuse a column set holding one half of a cumulative/interval pair.

    By the time a dataset reaches the stages after imputation, every cumulative
    variable should carry its interval column and every interval column its
    cumulative: the mapping derives whichever was not delivered. A set with one
    half missing means that conversion did not happen, and every later stage
    reads the gap as the variable being absent rather than as a pipeline step
    having been skipped.

    Both directions are checked at once: ``complement_column`` resolves the
    other half whichever side is held. The link comes from
    ``ComplementVariabele``, so a pair whose names do not follow the ``Diff``
    convention resolves correctly -- ``ZonopwekTotaal``'s interval is
    ``ZonopwekBruto``, and a guard built on the suffix would report it broken.

    Parameters
    ----------
    columns : iterable of str
        Column names. Not a DataFrame, so this serves a pandas frame and an
        ibis table equally -- pass ``df.columns`` or ``table.columns``.

    Raises
    ------
    ValueError
        Listing every column whose declared complement is absent, in both
        directions, rather than the first one found.
    """
    present = set(columns)
    missing = [
        f"{column}: declared complement {partner} is absent"
        for column in sorted(present)
        if (partner := complement_column(column)) is not None
        and partner not in present
    ]

    if missing:
        listed = "\n  ".join(missing)
        raise ValueError(
            f"Cumulative/interval pairs are incomplete:\n  {listed}\n"
            f"Both halves of a pair should exist by this stage -- the mapping "
            f"derives whichever the supplier did not deliver. A missing half "
            f"reads downstream as an absent variable rather than a skipped "
            f"step. Check the mapping stage and ComplementVariabele."
        )


def assert_bounds_not_inverted(diff_col, cadence) -> None:
    """Stop when the scaled lower bound of *diff_col* lands above its upper bound.

    The two bounds reach a reporting cadence by different rules: the floor in
    proportion to the interval, the peak by the shape the threshold declares. A
    lower bound above the upper one admits no value at all, so the column would
    be emptied wherever that cadence applies.

    Every cadence recorded for the column is checked, together with the declared
    pair, which is what a household with no recorded cadence is judged against.
    Metadata only, so this costs nothing per row.
    """
    declared = thresholds_dict.get(diff_col, {})
    cadences = sorted(
        set(cadence_minutes_by_household(diff_col, cadence).values())
    )

    for cadence_minutes in [None, *cadences]:
        lower, upper = scale_bounds(declared, cadence_minutes)
        if pd.isna(lower) or pd.isna(upper) or lower <= upper:
            continue

        at = (
            "the declared interval"
            if cadence_minutes is None
            else f"a cadence of {cadence_minutes} minute(s)"
        )
        raise ValueError(
            f"{diff_col}: scaled lower bound {lower} is above scaled upper "
            f"bound {upper} at {at}, so no value can satisfy both. "
            f"Check the variable, rule and physical-model metadata for an "
            f"inconsistency."
        )


def _coverage_duty_by_project(avg, col, project_id_column) -> pd.DataFrame:
    """One row per project: how much of the average exists, and how much of it moves.

    *coverage* is the share of the project's slots the average has a value for.
    *duty* is, of those, the share carrying consumption rather than a measured
    zero. A column can be fully covered and almost entirely idle, so the two
    answer different questions and the fill order needs both.

    Computed from the returned average frame rather than from either path's own
    intermediates, so the pandas and ibis paths cannot drift apart on it.
    """
    average = f"{col}_avg"
    frame = avg[[project_id_column, average]].copy()
    frame["_present"] = frame[average].notna()
    frame["_consuming"] = frame["_present"] & (frame[average] > 0)

    out = (
        frame.groupby(project_id_column, dropna=False)
        .agg(
            slots=(average, "size"),
            present=("_present", "sum"),
            consuming=("_consuming", "sum"),
        )
        .reset_index()
    )
    out["column"] = col
    out["coverage"] = (out["present"] / out["slots"]).astype("Float64")
    out["duty"] = (
        (out["consuming"] / out["present"]).where(out["present"] > 0, 0.0)
    ).astype("Float64")
    return out[
        [project_id_column, "column", "slots", "present", "consuming",
         "coverage", "duty"]
    ]


def calculate_average_diff(
    df: pd.DataFrame,
    project_id_column: str,
    diff_columns: list[str],
    bounds: dict | None = None,
    cadence: pd.DataFrame | None = None,
    grid_minutes: float | None = None,
) -> pd.DataFrame:
    """
    Calculate average differences for specified columns grouped by project and reading date.

    This function computes the average differences for the specified columns. It's used to prepare
    data for imputation of missing values. Criteria used to assess the inclusion of values is found in assert_averageable()

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data.
    project_id_column : str
        The name of the column containing project IDs.
    diff_columns : list[str]
        A list of column names for which to calculate average differences.
    bounds : dict, optional
        Column -> upper threshold. A value above its threshold is discarded
        before averaging. Defaults to the Max declared for the variable in the
        thresholds table; pass ``{}`` to disable, or a mapping to override.
    cadence : pd.DataFrame, optional
        The per-column cadence table. Households reporting coarser than the grid
        contribute to the level but not to the shape. Defaults to the sidecar
        written at mapping; an empty frame makes this the arithmetic mean.
    grid_minutes : float, optional
        The grid the data sits on. Resolved from the household batches'
        declared Gegevensfrequentie when not given.

    Returns
    -------
    dict
        A dictionary where keys are column names and values are dictionaries containing:
        - 'avg_diff': DataFrame with average differences
        - 'n_discarded': how many values were discarded for exceeding the
          column threshold
        - 'n_flagged': how many values were excluded as invented or mistimed
        - 'coverage': the share of slots the average has a value for
        - 'duty': of those, the share carrying consumption

    Warnings
    --------
    - Missing values in the resulting average columns will be logged as errors.


    """
    logging.info("Calculating Diff column averages.")
    assert_averageable(diff_columns)

    if bounds is None:
        bounds = {
            col: thresholds_dict[col]["Max"]
            for col in diff_columns
            if col in thresholds_dict
        }
    if cadence is None:
        cadence = load_column_cadence()
    if grid_minutes is None:
        grid_minutes = dataset_grid_minutes()

    avg_diff_dict = {}

    for col in diff_columns:
        logging.info(f"Handling column: {col}")

        logging.info(f"Filtering the dataframe for {col}.")
        # The cumulative variable supplies the volume where it has readings, so
        # it travels with the interval column, as do both findings columns.
        wanted = ["HuisIdBSV", project_id_column, "ReadingDate", col]
        for extra in (cumulative_column(col), flag_column(col)):
            if extra and extra in df.columns and extra not in wanted:
                wanted.append(extra)
        cum_findings = flag_column(cumulative_column(col) or "")
        if cum_findings in df.columns and cum_findings not in wanted:
            wanted.append(cum_findings)
        df_filtered = df[wanted]

        # any() on the mask is the cheap test; the minimum is only computed
        # when it has already failed.
        if (df_filtered[col] < 0).any():
            assert_no_negative_interval(col, df_filtered[col].min())

        df_filtered = df_filtered.copy()

        # Exclude values that were not measured at this slot: invented by the
        # pipeline, or recorded over a window rather than at this interval.
        # Identified by their findings flags, not by a threshold.
        n_flagged = 0
        findings = flag_column(col)
        if findings in df_filtered.columns:
            not_measured_here = (
                df_filtered[findings].fillna(0) & _NOT_MEASURED_HERE
            ) != 0
            n_flagged = int(not_measured_here.sum())
            if n_flagged:
                df_filtered.loc[not_measured_here, col] = pd.NA
                logging.info(
                    f"Excluded {n_flagged} value(s) from the {col} average: "
                    f"invented or recorded for a different window."
                )

        # Values above the column threshold are discarded, not clipped. The
        # threshold is scaled to each household's reporting cadence.
        n_discarded = 0
        bound = (bounds or {}).get(col, pd.NA)
        if bound is None or (not isinstance(bound, pd.Series) and pd.isna(bound)):
            bound = pd.NA
        else:
            bound = column_bound(df_filtered, col, cadence)
        if not (isinstance(bound, pd.Series) or pd.isna(bound)):
            bound = pd.Series(bound, index=df_filtered.index)
        if isinstance(bound, pd.Series):
            assert_bounds_not_inverted(col, cadence)
            out_of_bounds = (df_filtered[col] > bound).fillna(False)
            n_discarded = int(out_of_bounds.sum())
            if n_discarded:
                df_filtered.loc[out_of_bounds, col] = pd.NA
                logging.info(
                    f"Discarded {n_discarded} value(s) above the column bound "
                    f"for {col} before averaging."
                )

        logging.info(f"Calculating the average differences for {col}.")
        # Spread each reading evenly over the grid slots it covers, so every
        # row is a per-grid-interval rate and the average is a plain mean over
        # them. A household already at the grid cadence resamples to itself.
        df_filtered = df_filtered.assign(
            **{col: resample_to_grid(df_filtered, col, cadence, grid_minutes)}
        )

        # A household counts toward a slot only where it covered the window the
        # slot sits in, and a slot has an average only where enough of the
        # eligible households delivered for it.
        window_minutes = window_minutes_for_column(
            cadence_minutes_by_household(col, cadence).values()
        )
        values, presence_stats, window_presence = apply_presence_filter(
            df_filtered, col, project_id_column, window_minutes=window_minutes
        )
        df_filtered = df_filtered.assign(**{col: values})

        grouped = df_filtered.groupby([project_id_column, "ReadingDate"])[col]
        avg_diff = grouped.mean().reset_index()
        avg_diff.columns = [project_id_column, "ReadingDate", f"{col}_avg"]
        # Nanoseconds in both paths, so the two outputs merge cleanly.
        avg_diff = _match_reading_date_dtype(avg_diff)

        # coverage: slots the meters delivered anything for. duty: of those,
        # the slots that carry consumption.
        present = avg_diff[f"{col}_avg"].notna()
        coverage = float(present.mean()) if len(avg_diff) else 0.0
        duty = (
            float((avg_diff.loc[present, f"{col}_avg"] > 0).mean())
            if int(present.sum()) else 0.0
        )
        coverage_duty = _coverage_duty_by_project(avg_diff, col, project_id_column)

        avg_diff_dict[col] = {
            "avg_diff": avg_diff,
            "n_discarded": n_discarded,
            "n_flagged": n_flagged,
            "coverage": coverage,
            "duty": duty,
            "coverage_duty": coverage_duty,
            "presence": presence_stats,
            "window_presence": window_presence,
        }

    return avg_diff_dict


def calculate_average_diff_ibis(
    tbl,
    project_id_column: str,
    diff_columns: list[str],
    bounds: dict | None = None,
    cadence: pd.DataFrame | None = None,
    grid_minutes: float | None = None,
) -> dict:
    """
    Ibis/DuckDB variant of calculate_average_diff.

    Accepts an ibis.Table and performs all aggregations as SQL so the full
    dataset never materialises in Python memory.  Only the small result tables
    (one row per project, or one row per project x date) are executed, and the
    scaling step they feed is the one the pandas path uses.

    Returns the same dict structure as calculate_average_diff so that callers
    can substitute either function without downstream changes.

    Parameters
    ----------
    tbl : ibis.Table
        Household data table (e.g. from get_household_tables()["default"]).
    project_id_column : str
        Name of the project ID column.
    diff_columns : list[str]
        Diff column names to process.
    bounds : dict, optional
        Column -> upper threshold, as in calculate_average_diff.
    cadence : pd.DataFrame, optional
        The per-column cadence table, as in calculate_average_diff.
    grid_minutes : float, optional
        The grid the data sits on, as in calculate_average_diff.
    """
    import ibis

    logging.info("Calculating Diff column averages (Ibis).")
    assert_averageable(diff_columns)

    if bounds is None:
        bounds = {
            col: thresholds_dict[col]["Max"]
            for col in diff_columns
            if col in thresholds_dict
        }
    if cadence is None:
        cadence = load_column_cadence()
    if grid_minutes is None:
        grid_minutes = dataset_grid_minutes()

    # Properties of the frame rather than of any column, so they are computed
    # once here instead of once per column inside the loop below.
    _total_slots = int(
        tbl.select(project_id_column, "ReadingDate").distinct().count().execute()
    )
    _coarsest = (
        max(cadence["cadence_minutes"].dropna())
        if cadence is not None and not cadence.empty
        else grid_minutes
    )
    _max_slots = max(1, int(round(float(_coarsest) / float(grid_minutes))))

    # Every column's flagged count in one aggregate, so the loop below does not
    # make a round trip per column.
    _guard_agg = {}
    for i, c in enumerate(diff_columns):
        _guard_agg[f"_min_{i}"] = tbl[c].min()
        findings = flag_column(c)
        if findings in tbl.columns:
            _guard_agg[f"_flagged_{i}"] = (
                (tbl[findings].fill_null(0) & _NOT_MEASURED_HERE) != 0
            ).sum()
    _guards = (
        tbl.aggregate(**_guard_agg).execute().iloc[0].to_dict()
        if _guard_agg else {}
    )

    # Every household contributes, so this is the same list for every column and
    # is resolved once rather than per column.
    included_ids = tbl.select("HuisIdBSV").distinct().execute()["HuisIdBSV"].tolist()

    avg_diff_dict = {}

    for col in diff_columns:
        logging.info(f"Handling column (Ibis): {col}")
        assert_no_negative_interval(
            col, _guards.get(f"_min_{diff_columns.index(col)}")
        )
        n_discarded = 0
        n_flagged = 0
        coverage = 0.0
        duty = 0.0
        window_presence = None

        if not included_ids:
            # No households at all. Return an empty avg_diff with the correct
            # schema rather than calling isin([]), which is undefined behaviour
            # in some SQL backends.
            logging.warning(
                f"No households to average for column `{col}`; avg_diff will be empty."
            )
            avg_diff = pd.DataFrame(
                {
                    project_id_column: pd.array([], dtype="Int64"),
                    "ReadingDate": pd.array([], dtype="datetime64[us]"),
                    f"{col}_avg": pd.array([], dtype="Float64"),
                }
            )
        else:
            included = tbl.filter(tbl["HuisIdBSV"].isin(included_ids))
            value = included[col].cast("float64")

            # The interval each household's reading covers. Households absent
            # from the cadence table report at the grid cadence.
            minutes = _per_household(
                included["HuisIdBSV"],
                cadence_minutes_by_household(col, cadence),
                grid_minutes,
            )

            # Exclude values that were not measured at this slot: written by the
            # pipeline, or recorded for a window rather than for this interval.
            # A column with no findings column has nothing excluded here.
            findings = flag_column(col)
            if findings in included.columns:
                not_measured_here = (
                    included[findings].fill_null(0) & _NOT_MEASURED_HERE
                ) != 0
                n_flagged = int(
                    _guards.get(f"_flagged_{diff_columns.index(col)}") or 0
                )
                if n_flagged:
                    logging.info(
                        f"Excluded {n_flagged} value(s) from the {col} average: "
                        f"invented or recorded for a different window."
                    )
                    value = not_measured_here.ifelse(
                        ibis.null().cast("float64"), value
                    )

            # Values above the column threshold are discarded, not clipped.
            # The threshold is scaled to each household's cadence.
            bound = (bounds or {}).get(col, pd.NA)
            if pd.notna(bound):
                assert_bounds_not_inverted(col, cadence)
                scaled_bound = _per_household(
                    included["HuisIdBSV"],
                    column_bound_by_household(col, cadence),
                    bound,
                )
                over = value > scaled_bound
                n_discarded = int(included.filter(over).count().execute())
                if n_discarded:
                    logging.info(
                        f"Discarded {n_discarded} value(s) above the column "
                        f"bound for {col} before averaging."
                    )
                # NULL rather than filter: filtering would drop a
                # (project, timestamp) group whose values are all discarded,
                # where the group must survive with an NA average.
                value = over.ifelse(ibis.null().cast("float64"), value)

            # The cumulative variable supplies the volume wherever it holds
            # usable readings. Decided per household, so one column is behind
            # every window.
            cumulative = cumulative_column(col)
            cum_value = None
            on_cumulative_ids = []
            if cumulative and cumulative in included.columns:
                cum_value = included[cumulative].cast("float64")
                cum_findings = flag_column(cumulative)
                if cum_findings in included.columns:
                    cum_value = (
                        (included[cum_findings].fill_null(0) & _NOT_MEASURED_HERE)
                        != 0
                    ).ifelse(ibis.null().cast("float64"), cum_value)
                on_cumulative_ids = (
                    included.filter(cum_value.notnull())
                    .select("HuisIdBSV")
                    .distinct()
                    .execute()["HuisIdBSV"]
                    .tolist()
                )

            if on_cumulative_ids:
                on_cumulative = included["HuisIdBSV"].isin(on_cumulative_ids)
                source = on_cumulative.ifelse(cum_value, value)
            else:
                on_cumulative = ibis.literal(False)
                source = value

            prepared = included.mutate(
                _value=source, _minutes=minutes, _on_cumulative=on_cumulative
            )
            avg_diff, coverage, duty, window_presence = _resample_average_ibis(
                prepared, col, project_id_column, grid_minutes,
                max_slots=_max_slots, total_slots=_total_slots,
                window_minutes=window_minutes_for_column(
                    cadence_minutes_by_household(col, cadence).values()
                ),
            )

        avg_diff_dict[col] = {
            "avg_diff": avg_diff,
            "n_discarded": n_discarded,
            "n_flagged": n_flagged,
            "coverage": coverage,
            "duty": duty,
            "coverage_duty": _coverage_duty_by_project(
                avg_diff, col, project_id_column
            ),
            "window_presence": window_presence,
        }

    return avg_diff_dict


def _to_nullable(df: pd.DataFrame, float_cols: list[str] | None = None) -> pd.DataFrame:
    """Cast numpy int dtypes to Int64 and nominated float cols to Float64.

    int32/int64 cannot hold pd.NA so they are always promoted to Int64.
    float64 supports NaN but is promoted to Float64 only for explicitly
    nominated measurement columns, keeping non-measurement floats unchanged so
    that dtypes match the pandas path exactly.
    """
    for c in df.select_dtypes(include=["int32", "int64"]).columns:
        df[c] = df[c].astype("Int64")
    for c in float_cols or []:
        if c in df.columns:
            df[c] = df[c].astype("Float64")
    return df


def _per_household(household_column, by_household, default):
    """
    A per-household number as an ibis expression, defaulting where unknown.

    An empty mapping means nothing is known per household, which is the state
    when no cadence sidecar exists, so every row takes *default*.
    """
    import ibis

    known = {
        int(household): float(value)
        for household, value in by_household.items()
        if pd.notna(value)
    }
    if not known:
        return ibis.literal(float(default)).cast("float64")
    return household_column.substitute(known, else_=float(default)).cast("float64")

def _match_reading_date_dtype(frame):
    """Give ReadingDate the resolution pandas produces, preserving any timezone.

    DuckDB returns microsecond timestamps; the pandas path produces nanosecond
    ones. Both describe the same instant, but the two frames are then not
    directly comparable and a merge against a nanosecond frame is at the mercy
    of pandas version behaviour. Normalise here, at the boundary where the
    result leaves the engine, rather than making every caller cope.
    """
    if "ReadingDate" not in frame.columns:
        return frame
    column = frame["ReadingDate"]
    if isinstance(column.dtype, pd.DatetimeTZDtype):
        frame["ReadingDate"] = column.dt.as_unit("ns")
    elif pd.api.types.is_datetime64_any_dtype(column):
        frame["ReadingDate"] = column.astype("datetime64[ns]")
    return frame


def _resample_average_ibis(
    prepared, diff_col, project_id_column, grid_minutes, max_slots, total_slots,
    window_minutes=WINDOW_MINUTES,
):
    """
    The project average of *diff_col*, computed in SQL by resampling.

    The counterpart of :func:`etdtransform.bounds.resample_to_grid` followed by a
    plain mean. Two consecutive usable readings bound a window; its volume is
    spread evenly over the grid slots between them, ending at the later reading.

    ``_value`` arrives with every disqualified value already NULL, so a usable
    reading is simply a non-NULL one. The row number is taken over the full
    table before those rows are filtered out, so the count between two readings
    is a count of grid slots.

    Placement is a bounded cross join rather than a fill: each window is
    expanded into the slots it covers against a small table of offsets, at most
    as many as the coarsest cadence spans. DuckDB does not expose ``IGNORE
    NULLS`` through ibis 9.5, so a carry-backwards is not available.

    Parameters
    ----------
    prepared : ibis.Table
        Carrying ``_value`` (the value after exclusions) and ``_minutes`` (the
        interval each household's reading covers).
    diff_col : str
        Interval column being averaged.
    project_id_column : str
        Column identifying the project.
    grid_minutes : float
        Grid interval in minutes.
    window_minutes : float
        The window presence is judged over, as in
        :mod:`etdtransform.presence`.

    Returns
    -------
    tuple of (pd.DataFrame, float, float)
        The average with columns *project_id_column*, ``ReadingDate`` and
        ``<diff_col>_avg``; then coverage and duty.
    """
    import ibis
    import ibis.selectors as s_selectors
    from ibis import _

    order = ibis.window(group_by="HuisIdBSV", order_by="ReadingDate")
    slots = prepared["_minutes"] / float(grid_minutes)

    positioned = prepared.mutate(
        _pos=ibis.row_number().over(order), _slots=slots
    )
    readings = positioned.filter(positioned["_value"].notnull())
    readings = readings.mutate(
        _rise=_._value - _._value.lag().over(order),
        _gap=(_._pos - _._pos.lag().over(order)).cast("float64"),
    )

    # A cumulative reading states a running total, so a window's volume is the
    # rise since the previous reading and its length is the distance back to it.
    # A delivered interval value already is its own window's volume, and that
    # window is its cadence however long ago the last reading arrived.
    readings = readings.mutate(
        _volume=_._on_cumulative.ifelse(_._rise, _._value),
        _length=_._on_cumulative.ifelse(_._gap, _._slots),
    )

    # A cumulative window longer than one reporting interval spans a meter
    # outage. Spreading measured consumption across it is imputation's work, not
    # the average's. An interval-sourced window is its cadence and cannot span
    # one.
    readings = readings.filter(
        _._length.notnull()
        & (_._length > 0)
        & (~_._on_cumulative | (_._gap.notnull() & (_._gap <= _._slots)))
    )
    readings = readings.mutate(_per_slot=_._volume / _._length)

    # The offsets table is sized from the coarsest declared cadence rather than
    # by querying the longest window. Both give the same bound; the cadence
    # table is already in hand, and a query here would be one more round trip
    # per column. PROFILING_GUIDANCE.md: no per-column execute in a loop.
    span = int(max_slots)
    if span < 1:
        blank = _to_nullable(
            prepared.select(project_id_column, "ReadingDate").distinct().execute()
        )
        blank[f"{diff_col}_avg"] = pd.array([pd.NA] * len(blank), dtype="Float64")
        return _match_reading_date_dtype(blank), 0.0, 0.0, None

    offsets = ibis.memtable(
        pd.DataFrame({"_offset": pd.array(range(span), dtype="Int64")})
    )
    placed = readings.cross_join(offsets).filter(
        _._offset.cast("float64") < _._length
    )
    placed = placed.mutate(
        _slot=_.ReadingDate
        - (_._offset.cast("int64") * ibis.interval(minutes=int(grid_minutes)))
    )

    # The two presence gates, as in etdtransform.presence: a household counts
    # toward a slot only where it covered the window the slot sits in, and a
    # slot has an average only where enough of the window's eligible households
    # delivered for it.
    window = int(window_minutes)
    placed = placed.mutate(
        _window=(_._slot.epoch_seconds() // 60) // window
    )

    # The denominator is every grid row the household has in the window, so a
    # window clipped by the edge of the dataset is judged on what it held.
    grid = prepared.mutate(
        _window=(prepared["ReadingDate"].epoch_seconds() // 60) // window
    )
    held = grid.group_by([project_id_column, "HuisIdBSV", "_window"]).aggregate(
        _held=_.count()
    )
    delivered = placed.group_by(
        [project_id_column, "HuisIdBSV", "_window"]
    ).aggregate(_delivered=_._per_slot.count())

    coverage_keys = [project_id_column, "HuisIdBSV", "_window"]

    # The counts the eligibility gate below judges, exported per household and
    # window. One small execute per column: (household, window) rows, not data
    # rows. The judgements themselves are derived in window_presence_frame so
    # both averaging implementations share one definition.
    presence_counts = _to_nullable(
        held.left_join(delivered, coverage_keys)
        .select(~s_selectors.endswith("_right"))
        .execute()
    )
    presence_counts = presence_counts.rename(
        columns={"_held": "n_slots", "_delivered": "n_delivered"}
    )
    presence_counts["n_delivered"] = (
        presence_counts["n_delivered"].fillna(0).astype("Int64")
    )
    window_presence = window_presence_frame(
        presence_counts, diff_col, float(window), project_id_column
    )

    eligible = (
        delivered.inner_join(held, coverage_keys)
        .select(~s_selectors.endswith("_right"))
        .filter(
            _._delivered.cast("float64") / _._held.cast("float64")
            >= float(MIN_WINDOW_COVERAGE)
        )
        .select(*coverage_keys)
    )

    n_eligible = eligible.group_by([project_id_column, "_window"]).aggregate(
        _n_eligible=_.count()
    )

    placed = placed.inner_join(eligible, coverage_keys).select(
        ~s_selectors.endswith("_right")
    )

    covered = placed.group_by([project_id_column, "_slot"]).aggregate(
        **{f"{diff_col}_avg": _._per_slot.mean()},
        _contributors=_._per_slot.count(),
        # A slot belongs to one window, so every row of this group shares it.
        _window=_._window.max(),
    )
    covered = covered.left_join(
        n_eligible, [project_id_column, "_window"]
    ).select(~s_selectors.endswith("_right"))
    minimum = ibis.least(
        int(SUFFICIENCY_SHAPE),
        (
            float(SUFFICIENCY_FRACTION)
            * covered["_n_eligible"].fill_null(0).cast("float64")
        ).ceil(),
    )
    covered = covered.mutate(
        **{
            f"{diff_col}_avg": (covered["_contributors"] >= minimum).ifelse(
                covered[f"{diff_col}_avg"], ibis.null().cast("float64")
            )
        }
    )

    # Every (project, slot) the frame holds must appear, carrying NULL where no
    # window covers it. Grouping over the expanded windows alone drops those
    # slots entirely, and a missing row is not the same as a row with no value:
    # the pandas path keeps the group with an NA average, and the merge
    # downstream depends on that.
    slots = prepared.select(project_id_column, "ReadingDate").distinct()
    joined = slots.left_join(
        covered,
        [
            slots[project_id_column] == covered[project_id_column],
            slots["ReadingDate"] == covered["_slot"],
        ],
    )
    avg = _to_nullable(
        joined.select(
            slots[project_id_column], slots["ReadingDate"], covered[f"{diff_col}_avg"]
        ).execute(),
        float_cols=[f"{diff_col}_avg"],
    )
    avg = _match_reading_date_dtype(avg)
    avg = avg[[project_id_column, "ReadingDate", f"{diff_col}_avg"]].sort_values(
        [project_id_column, "ReadingDate"]
    ).reset_index(drop=True)

    # coverage over the slots the frame spans, not over the slots that survived.
    # total_slots is a property of the frame, not of the column, so the caller
    # counts it once rather than once per column.
    present = avg[f"{diff_col}_avg"].notna()
    coverage = float(int(present.sum()) / total_slots) if total_slots else 0.0
    duty = (
        float((avg.loc[present, f"{diff_col}_avg"] > 0).mean())
        if int(present.sum()) else 0.0
    )
    return avg, coverage, duty, window_presence


def concatenate_avg_diff_columns(avg_diff_dict, project_id_column):
    """
    Concatenate average difference columns for all variables.

    Each column may cover a different subset of (project, date) pairs
    (e.g. one supplier vs another). An outer merge on the key columns is used
    so that projects missing a column receive pd.NA rather than being
    silently assigned another column's values via positional alignment.
    """
    result_df = None
    for col, data in avg_diff_dict.items():
        df_col = data["avg_diff"]
        if result_df is None:
            result_df = df_col
        else:
            result_df = result_df.merge(
                df_col, on=[project_id_column, "ReadingDate"], how="outer"
            )
    return result_df


def equal_sig_fig(a, b, sig_figs):
    """
    Compare two numbers for equality up to a specified number of significant figures.

    This function rounds both numbers to the specified number of significant figures
    and then compares them for equality using a relative tolerance that scales with
    the magnitude of the numbers.

    Parameters
    ----------
    a : float
        The first number to compare.
    b : float
        The second number to compare.
    sig_figs : int
        The number of significant figures to consider for comparison.

    Returns
    -------
    bool
        True if the numbers are equal up to the specified number of significant figures,
        False otherwise.

    Notes
    -----
    This function uses the `isclose` function from the `math` module to compare the
    rounded numbers with a relative tolerance based on the number of significant figures.

    """
    # Define a helper function to scale the number to significant figures
    def round_to_sig_figs(x, sig_figs):
        if x == 0:
            return 0
        return round(x, sig_figs - int(floor(log10(abs(x)))) - 1)

    # Round both numbers to the specified significant figures
    a_rounded = round_to_sig_figs(a, sig_figs)
    b_rounded = round_to_sig_figs(b, sig_figs)

    # Apply a relative tolerance that scales with the magnitude of the numbers
    tolerance = 10 ** (-sig_figs)

    # Use isclose to compare with relative tolerance
    return isclose(a_rounded, b_rounded, rel_tol=tolerance)




def validate_household_column(household_df, cum_col, household_id):
    """
    Validate a household column for data quality and completeness.

    This function checks a specific column in a household DataFrame for missing values,
    zero values, and lack of change. It logs warnings and information about the data quality.

    Parameters
    ----------
    household_df : pd.DataFrame
        The DataFrame containing household data.
    cum_col : str
        The name of the cumulative column to validate.
    household_id : str
        The unique identifier for the household.

    Returns
    -------
    bool
        True if the column passes all checks, False otherwise.

    Notes
    -----
    This function is currently unused in the main processing pipeline.

    Warnings
    --------
    - Logs a warning if more than 40% of values in the column are missing.
    - Logs information about the number of missing values, zero values, and lack of change.

    """
    n_na = household_df[cum_col].isna().sum()
    len_df = len(household_df.index)

    if n_na == len_df:
        logging.info(
            f"HuisIdBSV {household_id} has all {n_na} missing values in {cum_col} of {len_df} records. Skipping column.",
        )
        return False
    elif n_na / len_df > 0.4:
        percent_na = 100 * n_na / len_df
        logging.error(
            f"HuisIdBSV {household_id} has {percent_na:.2f}% missing values in {cum_col}. Consider removing.",
        )
    else:
        logging.info(
            f"HuisIdBSV {household_id} has {n_na} missing values in {cum_col} of {len_df} records.",
        )

    if round(household_df[cum_col].sum(), 10) == 0:
        logging.info(
            f"HuisIdBSV {household_id} has no non-zero values in {cum_col}. Skipping column.",
        )
        return False
    if round(household_df[cum_col].max() - household_df[cum_col].min(), 10) == 0:
        logging.info(
            f"HuisIdBSV {household_id} has no change in {cum_col}. Skipping column.",
        )
        return False
    diff_col = interval_column(cum_col)
    if round(household_df[diff_col].sum(), 10) == 0:
        logging.warning(
            f"HuisIdBSV {household_id} has no non-zero values in {diff_col} before imputation.",
        )

    return True


# currently unused - apply as a sense check to ensure not too many values are missing
def get_reading_date_imputation_stats(df, project_id_column, cumulative_columns):
    """
    Calculate imputation statistics for each reading date and cumulative column.

    This function computes various statistics related to imputation for each reading date
    and cumulative column, including the number of imputed values, missing values, and
    original values.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    project_id_column : str
        The name of the column containing project IDs.
    cumulative_columns : list
        A list of cumulative column names to analyze.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing imputation statistics for each reading date and column.

    Notes
    -----
    This function is currently unused but can be applied as a sense check to ensure
    not too many values are missing.

    The resulting DataFrame includes the following columns:
    - project_id_column
    - ReadingDate
    - column
    - imputed
    - na
    - total_records
    - original
    - percent_imputed
    - percent_na
    - percent_original

    """
    grouped = df.groupby([project_id_column, "ReadingDate"])
    total_stats = grouped.size().rename("total_records")

    df_list = []
    for col in cumulative_columns:
        logging.info(f"Calculating imputation statistics by ReadingDate for {col}")

        diff_col = interval_column(col)
        is_imputed_col = f"{diff_col}_is_imputed"

        imputed_stats = grouped[is_imputed_col].sum().rename("imputed")
        na_stats = grouped[diff_col].apply(lambda x: x.isna().sum()).rename("na")

        stats_df = pd.concat(
            [imputed_stats, na_stats, total_stats],
            axis=1,
            ignore_index=False,
        )
        # stats_df = pd.concat([imputed_stats, na_stats, total_stats], axis=1).reset_index()

        stats_df["original"] = (
            stats_df["total_records"] - stats_df["imputed"] - stats_df["na"]
        )
        stats_df["percent_imputed"] = (
            stats_df["imputed"] / stats_df["total_records"]
        ) * 100
        stats_df["percent_na"] = (stats_df["na"] / stats_df["total_records"]) * 100
        stats_df["percent_original"] = (
            stats_df["original"] / stats_df["total_records"]
        ) * 100
        stats_df["column"] = col

        # Append the DataFrame to the list
        df_list.append(stats_df)

    # Concatenate all the DataFrames in the list
    logging.info(f"Concatenating the reading date statistics")
    imputation_reading_date_stats_df = pd.concat(
        df_list,
        ignore_index=False,
    ).reset_index()

    return imputation_reading_date_stats_df


def sort_for_impute(df: pd.DataFrame, project_id_column: str):
    """
    Sort the DataFrame to prepare for imputation.

    This function sorts the input DataFrame by project ID, household ID, and reading date.
    Sorting is necessary to ensure correct imputation of missing values.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to be sorted.
    project_id_column : str
        The name of the column containing project IDs.

    Returns
    -------
    pd.DataFrame
        The sorted DataFrame.

    Notes
    -----
    The sorting order is: project ID, household ID (HuisIdBSV), and reading date (ReadingDate).
    This order is crucial for the imputation process to work correctly.

    """
    logging.info("Sorting to prepare for imputation.")
    return df.sort_values(by=[project_id_column, "HuisIdBSV", "ReadingDate"])


def get_diff_columns(cumulative_columns: list):
    """
    Generate difference column names from cumulative column names.

    This function takes a list of cumulative column names and returns a list of
    corresponding difference column names by appending 'Diff' to each name.

    Parameters
    ----------
    cumulative_columns : list
        A list of cumulative column names.

    Returns
    -------
    list
        A list of difference column names.

    Notes
    -----
    This function is used to create names for columns that will store the differences
    between consecutive cumulative values.

    Tariff splits (Hoog/Laag) are dropped here: this is the single choke
    point the averaging twins build their diff set from, and the discrete
    alternating splits are not physically averageable -- only the combined
    root (delivered at mapping) is averaged. See
    :func:`etdtransform.vectorized_impute.without_tariff_splits`.
    """
    return [
        interval_column(col) for col in without_tariff_splits(cumulative_columns)
        if interval_column(col) is not None
    ]

def prepare_diff_averages_for_impute(
    df: pd.DataFrame,
    project_id_column: str,
    cumulative_columns: list,
    sorted=False,
):
    """
    Prepare difference columns for imputation.

    This function calculates average differences, combines them, and prepares
    household maximum and bound information for imputation.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data.
    project_id_column : str
        The name of the column containing project IDs.
    cumulative_columns : list
        A list of cumulative column names.
    sorted : bool, optional
        Whether the DataFrame is already sorted. Default is False.

    Returns
    -------
    tuple
        A tuple containing:
        - diff_columns: list of difference column names
        - diffs: DataFrame with average differences

    Notes
    -----
    This function performs the following steps:
    1. Sorts the DataFrame if not already sorted.
    2. Calculates average differences for each cumulative column.
    3. Saves the result to parquet for later use.

    The resulting file is saved in the directory specified by
    etdtransform.options.aggregate_folder_path.

    """
    if not sorted:
        df = sort_for_impute(df, project_id_column)

    diff_columns = get_diff_columns(cumulative_columns)

    logging.info("Starting to prepare diff averages.")
    avg_diff_dict = calculate_average_diff(df, project_id_column, diff_columns)
    logging.info("Combining average diff columns.")
    diffs = concatenate_avg_diff_columns(avg_diff_dict, project_id_column)

    logging.info("Saving average diff columns in avg_diffs.parquet")
    diffs.to_parquet(
        os.path.join(etdtransform.options.aggregate_folder_path, "avg_diffs.parquet"),
        engine="pyarrow",
    )
    write_coverage_duty(avg_diff_dict)
    write_window_presence(avg_diff_dict)

    return diff_columns, diffs


def prepare_diff_averages_for_impute_ibis(
    tbl,
    project_id_column: str,
    cumulative_columns: list,
):
    """
    Ibis variant of prepare_diff_averages_for_impute.

    Accepts an ibis.Table (e.g. from ibis.read_parquet) instead of a pandas
    DataFrame.  All heavy aggregations run as SQL inside DuckDB; only the small
    result tables (one row per project or per project x date) are materialised and
    saved as parquet artefacts.  The input table is never executed in full.

    Parameters
    ----------
    tbl : ibis.Table
        Lazy household table containing the Diff columns for each cumulative column.
    project_id_column : str
        Name of the project ID column.
    cumulative_columns : list
        List of cumulative column names (without "Diff" suffix).

    Returns
    -------
    tuple
        (diff_columns, diffs) -- same structure as
        prepare_diff_averages_for_impute.
    """
    diff_columns = get_diff_columns(cumulative_columns)

    logging.info("Starting to prepare diff averages (Ibis).")
    avg_diff_dict = calculate_average_diff_ibis(tbl, project_id_column, diff_columns)

    logging.info("Combining average diff columns.")
    diffs = concatenate_avg_diff_columns(avg_diff_dict, project_id_column)

    logging.info("Saving average diff columns in avg_diffs.parquet")
    diffs.to_parquet(
        os.path.join(etdtransform.options.aggregate_folder_path, "avg_diffs.parquet"),
        engine="pyarrow",
    )
    write_coverage_duty(avg_diff_dict)
    write_window_presence(avg_diff_dict)

    return diff_columns, diffs


def write_coverage_duty(avg_diff_dict, aggregate_folder_path=None) -> pd.DataFrame:
    """Write avg_coverage_duty.parquet beside avg_diffs.parquet.

    One row per project and column. The fill orders its columns by how much of
    the average exists and how much of it moves, and that ordering has to be the
    same on a re-run as it was on the run that produced the averages, so it is
    read from an artifact rather than recomputed from whatever is in memory.
    """
    frames = [
        data["coverage_duty"]
        for data in avg_diff_dict.values()
        if data.get("coverage_duty") is not None
    ]
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    if aggregate_folder_path is None:
        aggregate_folder_path = etdtransform.options.aggregate_folder_path
    path = os.path.join(str(aggregate_folder_path), "avg_coverage_duty.parquet")
    logging.info(
        f"Saving coverage and duty for {out['column'].nunique()} column(s) over "
        f"{len(out)} (project, column) pair(s) in avg_coverage_duty.parquet"
    )
    out.to_parquet(path, engine="pyarrow", index=False)
    return out


def write_window_presence(avg_diff_dict, aggregate_folder_path=None) -> pd.DataFrame:
    """Write avg_window_presence.parquet beside avg_diffs.parquet.

    One row per project, household, column and presence window: the counts the
    two presence gates judged and the judgements derived from them. The gates
    remove values without recording which household fell short where, so a
    thin average could not be traced to the households and windows behind it;
    this artifact is that record.
    """
    frames = [
        data["window_presence"]
        for data in avg_diff_dict.values()
        if data.get("window_presence") is not None
    ]
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    if aggregate_folder_path is None:
        aggregate_folder_path = etdtransform.options.aggregate_folder_path
    path = os.path.join(str(aggregate_folder_path), "avg_window_presence.parquet")
    logging.info(
        f"Saving window presence for {out['column'].nunique()} column(s) over "
        f"{len(out)} (project, household, window) row(s) in "
        f"avg_window_presence.parquet"
    )
    out.to_parquet(path, engine="pyarrow", index=False)
    return out


def read_diffs():
    """
    Read average differences from a parquet file.

    This function reads the average differences data from a parquet file
    located in the aggregate folder specified in the etdtransform options.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the average differences data.

    Notes
    -----
    The function assumes that the 'avg_diffs.parquet' file exists in the
    aggregate folder path specified in etdtransform.options.aggregate_folder_path.

    This function is typically used to load pre-calculated average differences
    for use in imputation processes.

    """
    return pd.read_parquet(
        os.path.join(etdtransform.options.aggregate_folder_path, "avg_diffs.parquet"),
        dtype_backend="numpy_nullable",
    )


def process_and_impute(
    df: pd.DataFrame,
    project_id_column: str,
    cumulative_columns: list,
    sorted=False,
    diffs_calculated=False,
    volume_reports=None,
    constraints=None,
    coverage_duty=None,
):
    """
    Process and impute missing values in the dataset.

    This function performs data processing and imputation on the input DataFrame.
    It can either calculate differences or load pre-calculated differences,
    and then applies imputation methods to fill missing values.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data to be processed and imputed.
    project_id_column : str
        The name of the column containing project IDs.
    cumulative_columns : list
        A list of cumulative column names to be processed.
    sorted : bool, optional
        Whether the DataFrame is already sorted. Default is False.
    diffs_calculated : bool, optional
        Whether differences have already been calculated. Default is False.

    Returns
    -------
    tuple
        A tuple containing:
        - df: The processed and imputed DataFrame
        - imputation_summary_house: Summary of imputation statistics per house
        - imputation_summary_project: Summary of imputation statistics per project
        - imputation_reading_date_stats_df: Statistics of imputation by reading date

    Notes
    -----
    This function performs the following steps:
    1. Sorts the DataFrame if not already sorted.
    2. Loads or calculates differences.
    3. Merges average differences into the household DataFrame.
    4. Applies imputation methods.
    5. Calculates and saves imputation statistics.
    6. Provides warnings for high imputation percentages.

    The function saves various statistics and summary files in the
    aggregate folder specified in etdtransform.options.aggregate_folder_path.

    Warnings
    --------
    - Logs warnings if any house or project has more than 40% imputed values.
    - Logs warnings if any reading date has more than 40% imputed values.

    """
    if not sorted:
        df = sort_for_impute(df, project_id_column)

    if diffs_calculated:
        logging.info("Loading average diffs from file...")
        diffs = read_diffs()
    else:
        diff_columns, diffs = prepare_diff_averages_for_impute(
            df=df,
            project_id_column=project_id_column,
            cumulative_columns=cumulative_columns,
            sorted=True,
        )

    logging.info(
        "Merging the average differences into the household dataframe for imputation.",
    )
    df = df.merge(diffs, on=[project_id_column, "ReadingDate"], how="left")

    logging.info("Merge completed.")

    df, imputation_gap_stats_df, imputation_reading_date_stats_df = (
        impute_and_normalize(
            df, cumulative_columns, project_id_column,
            volume_reports=volume_reports,
            constraints=constraints,
            coverage_duty=coverage_duty,
        )
    )

    logging.info("Summarizing imputation_gap_stats_df per house and column")
    imputation_summary_house = imputation_gap_stats_df[
        [
            project_id_column,
            "HuisIdBSV",
            "column",
            "diff_col_total",
            "cum_col_min_max_diff",
            "missing",
            "imputed",
            "imputed_na",
            "methods",
            "bitwise_methods",
        ]
        # The household factor and its inputs, so a factor that scaled a fill
        # can be read in the data it produced.
        + [
            name
            for name in (
                "house_impute_factor", "house_factor_is_fallback",
                "house_diff_median", "project_avg_median",
                "comparable_counts", "total_counts",
            )
            if name in imputation_gap_stats_df.columns
        ]
    ].reset_index()

    logging.info("Calculating the total records for each house")
    total_records_house = (
        df.groupby("HuisIdBSV").size().reset_index(name="total_records")
    )

    logging.info("Merging total records with house imputation summary")
    imputation_summary_house = imputation_summary_house.merge(
        total_records_house,
        on=["HuisIdBSV"],
    )
    imputation_summary_house["percentage_imputed"] = (
        imputation_summary_house["imputed"] / imputation_summary_house["total_records"]
    ) * 100

    logging.info("Summarizing imputation_gap_stats_df per project and column")
    imputation_summary_project = (
        imputation_gap_stats_df.groupby([project_id_column, "column"])
        .agg(
            {
                "bitwise_methods": lambda x: np.bitwise_or.reduce(x),
                "methods": lambda x: list(set().union(*x)),
                "missing": "sum",
                "imputed": "sum",
                "imputed_na": "sum",
            },
        )
        .reset_index()
    )

    logging.info(
        "Calculate the total records for each project and column from the original dataframe",
    )
    total_records_project = (
        df.groupby(project_id_column).size().reset_index(name="total_records")
    )

    logging.info("Merge total records with project imputation summary")
    imputation_summary_project = imputation_summary_project.merge(
        total_records_project,
        on=[project_id_column],
    )
    imputation_summary_project["percentage_imputed"] = (
        imputation_summary_project["imputed"]
        / imputation_summary_project["total_records"]
    ) * 100

    logging.info("Provide warnings if any house has > 40% imputed")
    over_40_percent_imputed_house = imputation_summary_house[
        imputation_summary_house["percentage_imputed"] > 40
    ]
    for _, row in over_40_percent_imputed_house.iterrows():
        logging.warning(
            f"House {row['HuisIdBSV']}, Column {row['column']} has {row['percentage_imputed']:.2f}% imputed values.",
        )

    logging.info("Provide warnings if any project has > 40% imputed")
    over_40_percent_imputed_project = imputation_summary_project[
        imputation_summary_project["percentage_imputed"] > 40
    ]
    for _, row in over_40_percent_imputed_project.iterrows():
        logging.warning(
            f"Project {row[project_id_column]}, Column {row['column']} has {row['percentage_imputed']:.2f}% imputed values.",
        )

    if imputation_reading_date_stats_df:
        logging.info("Provide warnings if any ReadingDates are over 40% imputed values")
        over_40_percent_imputed_dates = imputation_reading_date_stats_df[
            imputation_reading_date_stats_df["percent_imputed"] > 40
        ]
        for _, row in over_40_percent_imputed_dates.iterrows():
            logging.warning(
                f"ReadingDate {row['ReadingDate']}, Project {row[project_id_column]}, Column {row['column']} has {row['percent_imputed']:.2f}% imputed values.",
            )
    else:
        logging.warning("Not calculating reading date stats")

    return (
        df,
        imputation_summary_house,
        imputation_summary_project,
        imputation_reading_date_stats_df,
    )

def assert_imputation_input_batch_safe(df: pd.DataFrame) -> None:
    """
    Reject imputation input that cannot be imputed correctly.

    Expects one row per (HuisIdBSV, ReadingDate), and at most one
    HuisBatchIdBSV per HuisIdBSV. Call before averaging or imputing.

    Raises
    ------
    HouseBatchOverlapError
        A household has rows in more than one batch. Imputing the
        concatenated series would fill the between-batch gap, which is real
        non-delivery rather than missing data.
    ValueError
        Duplicate (HuisIdBSV, ReadingDate) rows. These double-weight the
        project averages and break the diff and sort order.
    """
    from etdmap.index_helpers import HouseBatchOverlapError

    if "HuisBatchIdBSV" in df.columns:
        counts = df.groupby("HuisIdBSV")["HuisBatchIdBSV"].nunique()
        multi = sorted(int(h) for h, n in counts.items() if n > 1)
        if multi:
            raise HouseBatchOverlapError(
                f"[imputation] Household(s) {multi} span more than one batch. "
                f"Imputing a concatenated multi-batch series would fill the "
                f"between-batch gap (real non-delivery). Batch-aware imputation "
                f"is not implemented yet; impute per batch or exclude."
            )
    dup_mask = df.duplicated(subset=["HuisIdBSV", "ReadingDate"])
    if bool(dup_mask.any()):
        bad = sorted(int(h) for h in df.loc[dup_mask, "HuisIdBSV"].dropna().unique())
        raise ValueError(
            f"[imputation] Duplicate (HuisIdBSV, ReadingDate) rows for household(s) "
            f"{bad} -- overlapping batch periods double-weight project averages and "
            f"break the diff math. Resolve the batch overlap first."
        )


def _assert_batch_safe_ibis(tbl) -> None:
    """Engine-side variant of assert_imputation_input_batch_safe for lazy tables
    (cheap aggregates; the table is never materialised in full)."""
    from etdmap.index_helpers import HouseBatchOverlapError

    if "HuisBatchIdBSV" in tbl.columns:
        per_hh = (
            tbl.group_by("HuisIdBSV")
            .aggregate(n_batches=tbl["HuisBatchIdBSV"].nunique())
        )
        multi = per_hh.filter(per_hh.n_batches > 1)["HuisIdBSV"].execute()
        if len(multi):
            raise HouseBatchOverlapError(
                f"[prepare_diff_averages] Household(s) {sorted(int(h) for h in multi)} span "
                f"more than one batch; project averages would mix batches. "
                f"Batch-aware averaging is not implemented yet."
            )
    n_rows = int(tbl.count().execute())
    n_keys = int(tbl[["HuisIdBSV", "ReadingDate"]].distinct().count().execute())
    if n_rows != n_keys:
        raise ValueError(
            f"[prepare_diff_averages] {n_rows - n_keys} duplicate (HuisIdBSV, ReadingDate) "
            f"row(s) -- overlapping batch periods would double-weight the project "
            f"averages. Resolve the batch overlap first."
        )


def prepare_diff_averages_sharded(
    mapped_folder_path=None,
    aggregate_folder_path=None,
    cumulative_columns: list = None,
    household_ids=None,
):
    """
    Sharded version of the avg-diffs stage: compute the imputation estimation
    input (avg_diffs.parquet) directly from the SHARDED mapped data.

    I/O shell only -- the computation is the SAME prepare_diff_averages_for_impute_ibis
    the legacy path uses, fed a lazy table built from mapped_household_table
    (DuckDB hive-glob, per the measured perf decision) with `ProjectIdBSV` joined
    from the index and `Meenemen` inclusion applied at read time.

    Outputs are written under ``<aggregate_folder_path>/sharded/`` (the sharded
    pipeline's artifact area), keeping legacy artifacts untouched.

    Batch safety: refuses multi-batch households and overlapping batch periods
    (see assert_imputation_input_batch_safe) -- the estimation inputs feed
    everything downstream.
    """
    import ibis as _ibis

    from etdmap.index_helpers import read_index
    from etdtransform.load_data import included_household_ids, mapped_household_table

    if mapped_folder_path is None:
        mapped_folder_path = etdtransform.options.mapped_folder_path
    if aggregate_folder_path is None:
        raise ValueError(
            "prepare_diff_averages_sharded: aggregate_folder_path is required and has "
            "no default (the config value points at the promoted production "
            "area)."
        )

    if household_ids is not None:
        ids = sorted(int(x) for x in household_ids)
    else:
        ids = included_household_ids(mapped_folder_path)

    index_df, _ = read_index(mapped_folder_path)
    if "HuisBatchIdBSV" not in index_df.columns:
        raise ValueError(
            "index.parquet has no HuisBatchIdBSV column; "
            "re-run the index save to add the household-batch columns."
        )
    # Keyed on the household batch, NOT the household: a household's project assignment is
    # constant WITHIN a batch but may differ BETWEEN batches (project
    # definitions change over time; the household does not). So
    # HuisIdBSV -> ProjectIdBSV is not a function over time, while
    # HuisBatchIdBSV -> ProjectIdBSV is. HuisBatchIdBSV is also the primary key
    # of index.parquet (one row per household batch), so keying on HuisIdBSV
    # would additionally keep whichever batch row happened to come last.

    household_batch_projects = index_df[["HuisBatchIdBSV", "ProjectIdBSV"]].dropna()
    project_map = _ibis.memtable(
        pd.DataFrame({
            "HuisBatchIdBSV": household_batch_projects["HuisBatchIdBSV"].astype("int64"),
            "ProjectIdBSV": household_batch_projects["ProjectIdBSV"].astype("int64"),
        })
    )

    tbl = mapped_household_table(mapped_folder_path)
    tbl = tbl.filter(tbl["HuisIdBSV"].isin(ids))
    tbl = tbl.join(project_map, tbl["HuisBatchIdBSV"] == project_map["HuisBatchIdBSV"])

    _assert_batch_safe_ibis(tbl)

    out_dir = os.path.join(str(aggregate_folder_path), "sharded")
    os.makedirs(out_dir, exist_ok=True)

    # prepare_diff_averages_for_impute_ibis writes via options.aggregate_folder_path;
    # scope it to the sharded artifact area for this call.
    previous = etdtransform.options.aggregate_folder_path
    etdtransform.options.aggregate_folder_path = out_dir
    try:
        return prepare_diff_averages_for_impute_ibis(
            tbl,
            project_id_column="ProjectIdBSV",
            cumulative_columns=cumulative_columns,
        )
    finally:
        etdtransform.options.aggregate_folder_path = previous
