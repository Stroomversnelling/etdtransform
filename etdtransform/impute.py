import logging
import os
import shutil
from math import floor, isclose, log10

import duckdb
import ibis
import numpy as np
import pandas as pd
from etdmap.storage import resolve_source_path
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
    MIN_ELIGIBLE_DATASETS,
    MIN_WINDOW_COVERAGE,
    SUFFICIENCY_FRACTION,
    SUFFICIENCY_SHAPE,
    WINDOW_MINUTES,
    window_minutes_for_column,
    window_presence_frame,
)

from etdtransform.bounds import (
    cadence_minutes_by_household_batch,
    column_bound,
    column_bound_by_household_batch,
    grid_minutes_by_project,
    load_column_cadence,
)

import etdtransform
from etdtransform.vectorized_impute import impute_and_normalize, without_tariff_splits
from etdmap import ids

#: Findings that disqualify a value from shaping a project average: it was
#: written by the pipeline rather than measured, or it was measured over a
#: window rather than at the interval it sits on.
_NOT_MEASURED_HERE = MappingFlag.NOT_A_MEASUREMENT.value

#: The averages artifact, named without a suffix because it is a directory of
#: (project, cadence) partitions. ``resolve_source_path`` takes this name and
#: falls back to ``avg_diffs.parquet`` for a run that predates the partitioning.
AVG_DIFFS_ARTIFACT = "avg_diffs"


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

    By the time a household batch reaches the stages after imputation, every cumulative
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

    Parameters
    ----------
    diff_col : str
        The column whose bounds are checked.
    cadence : pd.DataFrame or None
        The cadence table from :func:`etdtransform.bounds.load_column_cadence`.
        Every distinct ``cadence_minutes`` recorded for *diff_col* is one
        interval the column will be judged at, so each is checked here.
    """
    declared = thresholds_dict.get(diff_col, {})
    cadences = sorted(
        set(cadence_minutes_by_household_batch(diff_col, cadence).values())
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


def _coverage_duty_by_project(
    avg, col, project_id_column, discarded=None, flagged=None
) -> pd.DataFrame:
    """One row per project: how much of the average exists, and how much of it moves.

    *coverage* is the share of the project's slots the average has a value for.
    *duty* is, of those, the share carrying consumption rather than a measured
    zero. A column can be fully covered and almost entirely idle, so the two
    answer different questions and the fill order needs both.

    Computed from the returned average frame rather than from either path's own
    intermediates, so the pandas and ibis paths cannot drift apart on it.

    *discarded* and *flagged* are project -> count of readings the averaging
    threw out before taking the mean -- over the column's bound, and written by
    the pipeline rather than measured. They are computed to make that decision
    and then have nowhere to go, so a thin average could not be traced to the
    readings removed behind it. An absent mapping means nothing was removed,
    which at this grain is a measured zero rather than an unknown: the averaging
    looked at every row.
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
    for name, counts in (
        ("n_discarded", discarded), ("n_flagged", flagged),
    ):
        out[name] = pd.array(
            [(counts or {}).get(int(p), 0) for p in out[project_id_column]],
            dtype="Int64",
        )
    return out[
        [project_id_column, "column", "slots", "present", "consuming",
         "coverage", "duty", "n_discarded", "n_flagged"]
    ]


def _empty_average(project_id_column, col) -> pd.DataFrame:
    """The average frame of a column no household delivered.

    A frame with the right columns and no rows, rather than nothing: the merge
    in :func:`concatenate_avg_diff_columns` and every caller read these column
    names, and an absent column is a different failure from an empty one.
    """
    return pd.DataFrame(
        {
            project_id_column: pd.array([], dtype="Int64"),
            "ReadingDate": pd.array([], dtype="datetime64[us]"),
            f"{col}_avg": pd.array([], dtype="Float64"),
            f"{col}_contributors": pd.array([], dtype="Int64"),
            f"{col}_n_eligible": pd.array([], dtype="Int64"),
            "cadence_minutes": pd.array([], dtype="Int64"),
        }
    )


def _combine_cadence_groups(per_grid, project_id_column, diff_columns) -> dict:
    """One result per column out of the per-grid passes.

    A project sits on one grid, so the passes cover disjoint projects: the
    average frames concatenate rather than merge, the discard counts add, and
    the per-project coverage rows are already distinct. Sorting the result by
    (project, ReadingDate) leaves a single-grid run with exactly the frame one
    pass produced, plus ``cadence_minutes``.

    *per_grid* is ``(grid_minutes, result)`` pairs. An empty *per_grid* means no
    household at all, which still has to yield each column's empty frame.
    """
    combined = {}
    for col in diff_columns:
        frames = []
        presence = []
        coverage_duty = []
        n_discarded = 0
        n_flagged = 0
        for grid, result in per_grid:
            data = result[col]
            frame = data["avg_diff"].copy()
            # The interval one average covers, and the partition the artifact
            # is written under. Whole minutes: a sub-minute grid cannot be
            # recorded in the cadence table it is read from.
            frame["cadence_minutes"] = pd.array(
                [int(round(float(grid)))] * len(frame), dtype="Int64"
            )
            frames.append(frame)
            if data.get("window_presence") is not None:
                presence.append(data["window_presence"])
            if data.get("coverage_duty") is not None:
                coverage_duty.append(data["coverage_duty"])
            n_discarded += int(data["n_discarded"])
            n_flagged += int(data["n_flagged"])

        avg = (
            pd.concat(frames, ignore_index=True) if frames
            else _empty_average(project_id_column, col)
        )
        avg = avg.sort_values(
            [project_id_column, "ReadingDate"]
        ).reset_index(drop=True)

        duty_frame = (
            pd.concat(coverage_duty, ignore_index=True) if coverage_duty
            else _coverage_duty_by_project(avg, col, project_id_column)
        )
        # Recombined from the per-project rows rather than from each pass's own
        # figure: the rows partition the slots, so summing them gives the
        # household-batch-wide share exactly, with no weighting to get wrong.
        slots = int(duty_frame["slots"].sum()) if len(duty_frame) else 0
        present = int(duty_frame["present"].sum()) if len(duty_frame) else 0
        consuming = int(duty_frame["consuming"].sum()) if len(duty_frame) else 0

        combined[col] = {
            "avg_diff": avg,
            "n_discarded": n_discarded,
            "n_flagged": n_flagged,
            "coverage": float(present / slots) if slots else 0.0,
            "duty": float(consuming / present) if present else 0.0,
            "coverage_duty": duty_frame,
            "window_presence": (
                pd.concat(presence, ignore_index=True) if presence else None
            ),
        }
    return combined


def calculate_average_diff_ibis(
    tbl,
    project_id_column: str,
    diff_columns: list[str],
    bounds: dict | None = None,
    cadence: pd.DataFrame | None = None,
    grid_minutes: float | None = None,
    min_eligible: int = MIN_ELIGIBLE_DATASETS,
) -> dict:
    """
    The project average of each interval column, per project and reading date.

    This is what imputation fills from, and the only implementation: the
    aggregations run as SQL, so the full household batch never materialises in Python
    memory. Only the small result tables (one row per project, or one row per
    project x date) are executed.

    What may contribute at all is :func:`assert_averageable`. Of what remains, a
    reading above its column threshold is discarded rather than clipped -- it is
    a wrong reading, not a large one, and clipping would keep a fabricated
    number in the pool. A household reporting more coarsely than the grid
    contributes its level but not its shape: each reading is spread over the
    slots it covers before the mean is taken, so an hourly reading does not read
    as one slot of twelve hours' consumption.

    Parameters
    ----------
    tbl : ibis.Table
        Household data table (e.g. from get_household_tables()["default"]).
    project_id_column : str
        Name of the project ID column.
    diff_columns : list[str]
        Diff column names to process.
    bounds : dict, optional
        Column -> upper threshold. A value above its threshold is discarded
        before averaging. Defaults to the Max declared for the variable in the
        thresholds table; pass ``{}`` to disable, or a mapping to override.
    cadence : pd.DataFrame, optional
        The per-column cadence table. Households reporting coarser than the grid
        contribute to the level but not to the shape. Defaults to the sidecar
        written at mapping; an empty frame makes this the arithmetic mean.
    grid_minutes : float, optional
        Forces one grid on every project. Leave unset in the pipeline: the grid
        is then resolved per project from the cadence table, which is what makes
        a project's average sit on the interval that project reports at. Pass it
        only to state the grid of a frame built at a known spacing.
    min_eligible : int
        Eligible households a window needs before its average is published at
        all (:func:`etdtransform.presence.window_is_averageable`). Lower it only
        for a frame deliberately smaller than a project.

    Returns
    -------
    dict
        One entry per column in *diff_columns*, each a dict of:

        - ``avg_diff`` -- DataFrame of the average per (project, ReadingDate),
          carrying ``cadence_minutes`` -- the interval one average covers
        - ``n_discarded`` -- values dropped for exceeding the column threshold
        - ``n_flagged`` -- values excluded as invented or mistimed
        - ``coverage`` -- the share of slots the average has a value for
        - ``duty`` -- of those, the share carrying consumption
        - ``coverage_duty`` -- the same two per project
        - ``window_presence`` -- which household batches covered which window

    Warnings
    --------
    A slot every contributing household was discarded at has no average rather
    than an average of zero, and that is logged: no average is not the same
    fact as no consumption.
    """
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

    # Which project each household batch is in, in one round trip. The grid is
    # resolved per project and the projects grouped by it, so there is one pass
    # per distinct grid rather than one per project: a per-project pass would
    # rebuild every presence aggregate once per project per column, and the
    # number of distinct grids is bounded by the declarations while the number
    # of projects is not (PROFILING_GUIDANCE.md).
    membership = (
        tbl.select(ids.HOUSEHOLD_BATCH, project_id_column).distinct().execute()
    )
    project_by_household_batch = {}
    unassigned = []
    for household_batch, project in zip(
        membership[ids.HOUSEHOLD_BATCH], membership[project_id_column]
    ):
        if pd.isna(project):
            unassigned.append(int(household_batch))
        else:
            project_by_household_batch[int(household_batch)] = int(project)
    if unassigned:
        raise ValueError(
            f"{len(unassigned)} household batch(es) carry no "
            f"{project_id_column}, so their readings belong to no project "
            f"average: {sorted(unassigned)[:10]}. The project is joined from "
            f"the index before averaging; a dataset missing from it cannot "
            f"contribute."
        )

    if grid_minutes is None:
        grid_by_project = grid_minutes_by_project(cadence, project_by_household_batch)
        unresolved = sorted(
            set(project_by_household_batch.values()) - set(grid_by_project)
        )
        if unresolved:
            raise ValueError(
                f"Project(s) {unresolved} resolve no grid, so there is no "
                f"interval for their average to sit on. It is declared per "
                f"household batch as Gegevensfrequentie and recorded on the "
                f"ReadingDate row of the column cadence sidecar, which "
                f"detect_column_cadence writes at mapping."
            )
    else:
        grid_by_project = {
            project: float(grid_minutes)
            for project in set(project_by_household_batch.values())
        }

    batches_by_grid = {}
    for household_batch, project in project_by_household_batch.items():
        batches_by_grid.setdefault(grid_by_project[project], []).append(
            household_batch
        )

    per_grid = [
        (
            grid,
            _average_at_one_cadence(
                tbl, project_id_column, diff_columns, bounds, cadence,
                grid_minutes=grid, included_ids=sorted(household_batches),
                min_eligible=min_eligible,
            ),
        )
        for grid, household_batches in sorted(batches_by_grid.items())
    ]
    return _combine_cadence_groups(per_grid, project_id_column, diff_columns)


def _average_at_one_cadence(
    tbl,
    project_id_column: str,
    diff_columns: list[str],
    bounds: dict,
    cadence: pd.DataFrame | None,
    grid_minutes: float,
    included_ids: list[int],
    min_eligible: int = MIN_ELIGIBLE_DATASETS,
) -> dict:
    """
    One pass of the averaging, over the household batches that share one grid.

    Split out from :func:`calculate_average_diff_ibis` so the spreading
    denominator and the step-back can take a literal: within a pass there is one
    grid, and ``_resample_average_ibis`` needs the slot spacing as a number.

    *included_ids* is the pass's household batches, resolved by the caller from
    the (household batch, project) membership it already read, so this costs no
    round trip of its own.
    """
    tbl = (
        tbl.filter(tbl[ids.HOUSEHOLD_BATCH].isin(included_ids))
        if included_ids else tbl
    )

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
    # make a round trip per column. Grouped by project, which costs the same
    # round trip and is the grain the counts are published at -- a household-batch-wide
    # total cannot say which project lost readings.
    _guard_agg = {}
    for i, c in enumerate(diff_columns):
        _guard_agg[f"_min_{i}"] = tbl[c].min()
        findings = flag_column(c)
        if findings in tbl.columns:
            _guard_agg[f"_flagged_{i}"] = (
                (tbl[findings].fill_null(0) & _NOT_MEASURED_HERE) != 0
            ).sum()
    _guards = (
        tbl.group_by(project_id_column).aggregate(**_guard_agg).execute()
        if _guard_agg else pd.DataFrame()
    )

    def _by_project(field):
        """A per-project figure from the grouped guard aggregate."""
        if field not in _guards.columns:
            return {}
        return {
            int(project): value
            for project, value in zip(_guards[project_id_column], _guards[field])
            if pd.notna(project)
        }

    avg_diff_dict = {}

    for col in diff_columns:
        logging.info(f"Handling column (Ibis): {col}")
        # The smallest across the projects, since the guard is about the column.
        # NaN means a project delivered nothing of it, which is not a smallest.
        _smallest = [
            value
            for value in _by_project(f"_min_{diff_columns.index(col)}").values()
            if pd.notna(value)
        ]
        assert_no_negative_interval(col, min(_smallest) if _smallest else None)
        discarded_by_project = {}
        flagged_by_project = {}
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
            avg_diff = _empty_average(project_id_column, col)
        else:
            included = tbl
            value = included[col].cast("float64")

            # The interval each household batch's readings cover. A household batch absent
            # from the cadence table reports at the grid cadence.
            minutes = _per_dataset(
                included[ids.HOUSEHOLD_BATCH],
                cadence_minutes_by_household_batch(col, cadence),
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
                flagged_by_project = {
                    project: int(count)
                    for project, count in _by_project(
                        f"_flagged_{diff_columns.index(col)}"
                    ).items()
                    if pd.notna(count)
                }
                n_flagged = sum(flagged_by_project.values())
                if n_flagged:
                    logging.info(
                        f"Excluded {n_flagged} value(s) from the {col} average: "
                        f"invented or recorded for a different window."
                    )
                    value = not_measured_here.ifelse(
                        ibis.null().cast("float64"), value
                    )

            # Values above the column threshold are discarded, not clipped.
            # The threshold is scaled to each household batch's cadence.
            bound = (bounds or {}).get(col, pd.NA)
            if pd.notna(bound):
                assert_bounds_not_inverted(col, cadence)
                scaled_bound = _per_dataset(
                    included[ids.HOUSEHOLD_BATCH],
                    column_bound_by_household_batch(col, cadence),
                    bound,
                )
                over = value > scaled_bound
                # Per project in the same round trip the total already cost:
                # which project's readings were thrown away is the question a
                # reader of the average has, and a single total cannot answer it.
                _over = included.filter(over)
                _discarded = (
                    _over.group_by(project_id_column)
                    .aggregate(_n=_over.count())
                    .execute()
                )
                discarded_by_project = {
                    int(project): int(count)
                    for project, count in zip(
                        _discarded[project_id_column], _discarded["_n"]
                    )
                    if pd.notna(project) and pd.notna(count)
                }
                n_discarded = sum(discarded_by_project.values())
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
            # usable readings. Decided per household batch, so one column is behind
            # every reading of that household batch.
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
                    .select(ids.HOUSEHOLD_BATCH)
                    .distinct()
                    .execute()[ids.HOUSEHOLD_BATCH]
                    .tolist()
                )

            if on_cumulative_ids:
                on_cumulative = included[ids.HOUSEHOLD_BATCH].isin(on_cumulative_ids)
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
                    cadence_minutes_by_household_batch(col, cadence).values()
                ),
                min_eligible=min_eligible,
            )

        avg_diff_dict[col] = {
            "avg_diff": avg_diff,
            "n_discarded": n_discarded,
            "n_flagged": n_flagged,
            "coverage": coverage,
            "duty": duty,
            "coverage_duty": _coverage_duty_by_project(
                avg_diff, col, project_id_column,
                discarded=discarded_by_project, flagged=flagged_by_project,
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


def _per_dataset(dataset_column, by_dataset, default):
    """
    A per-household-batch number as an ibis expression, defaulting where unknown.

    An empty mapping means nothing is known per household batch, which is the state
    when no cadence sidecar exists, so every row takes *default*.
    """
    import ibis

    known = {
        int(dataset): float(value)
        for dataset, value in by_dataset.items()
        if pd.notna(value)
    }
    if not known:
        return ibis.literal(float(default)).cast("float64")
    return dataset_column.substitute(known, else_=float(default)).cast("float64")

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
    min_eligible=MIN_ELIGIBLE_DATASETS,
):
    """
    The project average of *diff_col*, computed in SQL by resampling.

    The spreading step, followed by a plain mean. Two consecutive usable
    readings of one household batch bound a window; its volume is spread evenly over
    the grid slots between them, ending at the later reading.

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
        interval each household batch's readings cover).
    diff_col : str
        Interval column being averaged.
    project_id_column : str
        Column identifying the project.
    grid_minutes : float
        Grid interval in minutes.
    window_minutes : float
        The window presence is judged over, as in
        :mod:`etdtransform.presence`.
    min_eligible : int
        Eligible households a window needs before its average is published.
        Lower it only for a frame deliberately smaller than a project, the same
        way a one-slot *window_minutes* neutralises the coverage gate.

    Returns
    -------
    tuple of (pd.DataFrame, float, float)
        The average with columns *project_id_column*, ``ReadingDate`` and
        ``<diff_col>_avg``; then coverage and duty.
    """
    import ibis
    import ibis.selectors as s_selectors
    from ibis import _

    order = ibis.window(group_by=ids.HOUSEHOLD_BATCH, order_by="ReadingDate")
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
    #
    # Marked as well as excluded: an average thinned because a meter was out
    # and one thinned because nobody delivered are different facts about the
    # project, and the surviving counts cannot tell them apart.
    _has_length = _._length.notnull() & (_._length > 0)
    marked = readings.mutate(
        _outage=(
            _has_length
            & _._on_cumulative
            & ~(_._gap.notnull() & (_._gap <= _._slots))
        )
    )
    readings = marked.filter(_has_length & ~_._outage)
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
        # pd.NA, not zero: no gate ran, so these counts were never measured
        # (ADR-022) -- unlike a slot the gates judged and found empty.
        for _count in ("_contributors", "_n_eligible"):
            blank[f"{diff_col}{_count}"] = pd.array(
                [pd.NA] * len(blank), dtype="Int64"
            )
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
    # window clipped by the edge of the household batch is judged on what it held.
    #
    # Which source fed the household rides out of the same aggregate. It is
    # decided per household batch per column and then used and forgotten, so without
    # this a reader cannot tell whether a household's volume came from its
    # cumulative readings or from its delivered intervals.
    grid = prepared.mutate(
        _window=(prepared["ReadingDate"].epoch_seconds() // 60) // window
    )
    held = grid.group_by(
        [project_id_column, ids.HOUSEHOLD_BATCH, "_window"]
    ).aggregate(
        _held=_.count(),
        _from_cumulative=_._on_cumulative.max(),
    )
    delivered = placed.group_by(
        [project_id_column, ids.HOUSEHOLD_BATCH, "_window"]
    ).aggregate(_delivered=_._per_slot.count())

    # The windows the outage term above removed, attributed to the window the
    # reading that closed the outage sits in. Joined into the presence read
    # rather than executed on its own: these are (household, window) rows, and
    # a separate execute would re-scan the data once more per column.
    outages = (
        marked.filter(_._outage)
        .mutate(_window=(_.ReadingDate.epoch_seconds() // 60) // window)
        .group_by([project_id_column, ids.HOUSEHOLD_BATCH, "_window"])
        .aggregate(
            _n_outage_windows=_.count(),
            _n_outage_slots=_._length.sum(),
        )
    )

    coverage_keys = [project_id_column, ids.HOUSEHOLD_BATCH, "_window"]

    # The counts the eligibility gate below judges, exported per household batch and
    # window. One small execute per column: (household, window) rows, not data
    # rows. The judgements themselves are derived in window_presence_frame so
    # both averaging implementations share one definition.
    presence_counts = _to_nullable(
        held.left_join(delivered, coverage_keys)
        .select(~s_selectors.endswith("_right"))
        .left_join(outages, coverage_keys)
        .select(~s_selectors.endswith("_right"))
        .execute()
    )
    presence_counts = presence_counts.rename(
        columns={
            "_held": "n_slots",
            "_delivered": "n_delivered",
            "_n_outage_windows": "n_outage_windows",
            "_n_outage_slots": "n_outage_slots",
        }
    )
    # Zero rather than pd.NA on all three: a left join leaves nothing where the
    # household delivered for no slot of the window or had no outage in it, and
    # both of those are measured zeros rather than absences of measurement.
    #
    # Numeric before fillna, because the join can hand back an object column and
    # filling that one downcasts (a pandas FutureWarning that becomes an error).
    for _count in ("n_delivered", "n_outage_windows", "n_outage_slots"):
        presence_counts[_count] = pd.array(
            pd.to_numeric(presence_counts[_count], errors="coerce").fillna(0),
            dtype="Int64",
        )
    # The source as a word rather than a flag, because "which of the two" is
    # the question and a boolean named after one of them answers it backwards
    # half the time.
    presence_counts["source"] = pd.array(
        [
            "cumulative" if bool(flag) else "interval"
            for flag in presence_counts["_from_cumulative"].fillna(False)
        ],
        dtype="string",
    )
    presence_counts = presence_counts.drop(columns=["_from_cumulative"])
    window_presence = window_presence_frame(
        presence_counts, diff_col, float(window), project_id_column,
        min_eligible=min_eligible,
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
    # Two gates, both required, and the second is not implied by the first: the
    # fraction scales with the count it judges, so at one eligible household it
    # demands one and that household's own readings are published as the
    # project's average. presence.window_is_averageable is the same rule on the
    # pandas side.
    enough_households = (
        covered["_n_eligible"].fill_null(0) >= int(min_eligible)
    )
    covered = covered.mutate(
        **{
            f"{diff_col}_avg": (
                enough_households & (covered["_contributors"] >= minimum)
            ).ifelse(
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
    # The two counts the gates judged come out with the average rather than
    # being dropped by this projection. Neither is recoverable afterwards: the
    # contributor count is per slot and depends on where each household's own
    # readings fell, which the per-window presence artifact cannot say, and
    # without the eligible count beside it a reader cannot tell a slot few
    # households delivered for from a project that has few.
    contributors = f"{diff_col}_contributors"
    n_eligible_col = f"{diff_col}_n_eligible"
    avg = _to_nullable(
        joined.select(
            slots[project_id_column],
            slots["ReadingDate"],
            covered[f"{diff_col}_avg"],
            covered["_contributors"].name(contributors),
            covered["_n_eligible"].name(n_eligible_col),
        ).execute(),
        float_cols=[f"{diff_col}_avg"],
    )
    avg = _match_reading_date_dtype(avg)
    for _count in (contributors, n_eligible_col):
        avg[_count] = avg[_count].astype("Int64")
    avg = avg[[
        project_id_column, "ReadingDate", f"{diff_col}_avg",
        contributors, n_eligible_col,
    ]].sort_values([project_id_column, "ReadingDate"]).reset_index(drop=True)

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

    ``cadence_minutes`` joins the key columns where it is present. It is a
    property of the project, so every column's frame carries the same value for
    a given project and merging on it changes no row -- but merging on only the
    other two would leave a suffixed pair of it per column.
    """
    keys = [project_id_column, "ReadingDate"]
    result_df = None
    for col, data in avg_diff_dict.items():
        df_col = data["avg_diff"]
        if result_df is None:
            result_df = df_col
            if "cadence_minutes" in df_col.columns:
                keys = [*keys, "cadence_minutes"]
        else:
            result_df = result_df.merge(df_col, on=keys, how="outer")
    return result_df


# currently unused - apply as a sense check to ensure not too many values are missing
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
    The sorting order is: project, household batch dataset, ReadingDate. Each
    household batch's rows then form one contiguous run in date order, which the
    imputation reads positionally. The household is not in the key: the
    household batch determines it, so naming it sorts on the same thing twice.

    """
    logging.info("Sorting to prepare for imputation.")
    return df.sort_values(
        by=[project_id_column, ids.HOUSEHOLD_BATCH, "ReadingDate"]
    )


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

def prepare_diff_averages_for_impute_ibis(
    tbl,
    project_id_column: str,
    cumulative_columns: list,
    min_eligible: int = MIN_ELIGIBLE_DATASETS,
):
    """
    Compute the project averages and save the artefacts imputation reads.

    All heavy aggregations run as SQL inside DuckDB; only the small result
    tables (one row per project or per project x date) are materialised and
    saved as parquet artefacts. The input table is never executed in full.

    Parameters
    ----------
    tbl : ibis.Table
        Lazy household table containing the Diff columns for each cumulative column.
    project_id_column : str
        Name of the project ID column.
    cumulative_columns : list
        List of cumulative column names (without "Diff" suffix).
    min_eligible : int
        Eligible households a window needs before its average is published
        (:func:`etdtransform.presence.window_is_averageable`). Lower it only for
        a household batch deliberately smaller than a project -- a ten-household sample
        is below the floor by construction, and leaving the floor there would
        publish no average at all and send every column down the no-average
        path, which looks like a passing run and tests almost nothing.

    Returns
    -------
    tuple
        ``(diff_columns, diffs)`` -- the interval column names, and one frame
        of every column's average keyed by (project, ReadingDate).
    """
    diff_columns = get_diff_columns(cumulative_columns)

    logging.info("Starting to prepare diff averages (Ibis).")
    avg_diff_dict = calculate_average_diff_ibis(
        tbl, project_id_column, diff_columns, min_eligible=min_eligible
    )

    logging.info("Combining average diff columns.")
    diffs = concatenate_avg_diff_columns(avg_diff_dict, project_id_column)

    write_diffs(diffs, project_id_column)
    write_coverage_duty(avg_diff_dict)
    write_window_presence(avg_diff_dict)

    return diff_columns, diffs


def write_diffs(
    diffs, project_id_column=ids.PROJECT, aggregate_folder_path=None
) -> str:
    """Write the averages, partitioned on project and cadence.

    One partition per (project, cadence). The cadence is the interval one
    average covers, so it is what tells a reader what the number means -- and
    putting it in the path rather than only in a column means a reader cannot
    take a quarter-hourly average for a five-minute one by dropping a column.

    Written as a directory named ``avg_diffs``, which is the partitioned
    spelling of the canonical ``avg_diffs.parquet``
    (``promote_aggregate_outputs._CANONICAL_STEMS``); promotion supersedes
    whichever form the previous run left. Read it back with :func:`read_diffs`,
    never with a bare ``pd.read_parquet`` -- the partition columns arrive
    dictionary-encoded and the nullable backend refuses them.

    Returns the path written.
    """
    if aggregate_folder_path is None:
        aggregate_folder_path = etdtransform.options.aggregate_folder_path
    target = os.path.join(str(aggregate_folder_path), AVG_DIFFS_ARTIFACT)

    missing = [
        key for key in (project_id_column, "cadence_minutes")
        if key not in diffs.columns
    ]
    if missing:
        raise ValueError(
            f"write_diffs: the averages carry no {missing}, so they cannot be "
            f"partitioned by project and cadence. Both are added by "
            f"calculate_average_diff_ibis."
        )

    # The writer owns and clears its output directory, the same convention the
    # other partitioned writers follow: a partition no longer produced would
    # otherwise be read back beside this run's.
    if os.path.isdir(target):
        shutil.rmtree(target)

    logging.info(
        f"Saving average diff columns for "
        f"{diffs[project_id_column].nunique()} project(s) over "
        f"{diffs['cadence_minutes'].nunique()} cadence(s) in "
        f"{AVG_DIFFS_ARTIFACT}"
    )
    with duckdb.connect() as con:
        con.register("_avg_diffs", diffs)
        con.execute(
            f"COPY (SELECT * FROM _avg_diffs) TO '{target}' "
            f"(FORMAT PARQUET, COMPRESSION SNAPPY, "
            f'PARTITION_BY ("{project_id_column}", cadence_minutes), '
            f"OVERWRITE_OR_IGNORE TRUE)"
        )
    return target


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


def read_diffs(
    aggregate_folder_path=None, with_cadence=False, with_counts=False
):
    """
    The project averages imputation fills from.

    The one reader for this artifact, in either layout: the ``avg_diffs``
    directory of (project, cadence) partitions :func:`write_diffs` produces, or
    the single ``avg_diffs.parquet`` a run predating the partitioning left.
    ``resolve_source_path`` decides which is present.

    A partitioned read goes through the engine, not ``pd.read_parquet``: the
    partition columns arrive dictionary-encoded, which the nullable backend
    raises on. Dtypes are restored on the way out, so both layouts return the
    same ADR-005 frame.

    By default the frame holds the keys and the averages and nothing else.
    The fill merges it onto a chunk on (project, ReadingDate), so every other
    column in it would ride into the written shards.

    Parameters
    ----------
    aggregate_folder_path : str, optional
        Where to read from. Defaults to the configured aggregate folder; the
        sharded chain passes its own ``sharded`` area.
    with_cadence : bool
        Keep ``cadence_minutes`` -- the interval one average covers.
    with_counts : bool
        Keep ``<column>_contributors`` and ``<column>_n_eligible`` -- how many
        households the slot's average rests on, and how many were eligible that
        window.

    Returns
    -------
    pd.DataFrame
        The average per (project, ReadingDate), one ``<column>_avg`` per
        averaged column.
    """
    if aggregate_folder_path is None:
        aggregate_folder_path = etdtransform.options.aggregate_folder_path
    path = resolve_source_path(aggregate_folder_path, AVG_DIFFS_ARTIFACT)

    if os.path.isdir(path):
        pattern = os.path.join(path, "**", "*.parquet").replace("\\", "/")
        frame = ibis.read_parquet(
            pattern, hive_partitioning=True, union_by_name=True
        ).execute()
        frame = _match_reading_date_dtype(
            _to_nullable(
                frame,
                float_cols=[c for c in frame.columns if c.endswith("_avg")],
            )
        )
    else:
        frame = pd.read_parquet(path, dtype_backend="numpy_nullable")

    dropped = []
    if not with_cadence:
        dropped.append("cadence_minutes")
    if not with_counts:
        dropped += [
            c for c in frame.columns
            if c.endswith(("_contributors", "_n_eligible"))
        ]
    present = [c for c in dropped if c in frame.columns]
    return frame.drop(columns=present) if present else frame


def assert_imputation_input_batch_safe(df: pd.DataFrame) -> None:
    """
    Reject imputation input that cannot be imputed correctly.

    Expects one row per (ids.HOUSEHOLD, ReadingDate), and at most one
    ids.HOUSEHOLD_BATCH per ids.HOUSEHOLD. Call before averaging or imputing.

    Raises
    ------
    HouseBatchOverlapError
        A household has rows in more than one household batch. Imputing the
        concatenated series would fill the between-household-batch gap, which is real
        non-delivery rather than missing data.
    ValueError
        Duplicate (ids.HOUSEHOLD, ReadingDate) rows. These double-weight the
        project averages and break the diff and sort order.
    """
    from etdmap.index_helpers import HouseBatchOverlapError

    if ids.HOUSEHOLD_BATCH in df.columns:
        counts = df.groupby(ids.HOUSEHOLD)[ids.HOUSEHOLD_BATCH].nunique()
        multi = sorted(int(h) for h, n in counts.items() if n > 1)
        if multi:
            raise HouseBatchOverlapError(
                f"[imputation] Household(s) {multi} span more than one batch. "
                f"Imputing a concatenated multi-batch series would fill the "
                f"between-batch gap (real non-delivery). Batch-aware imputation "
                f"is not implemented yet; impute per batch or exclude."
            )
    dup_mask = df.duplicated(subset=[ids.HOUSEHOLD_BATCH, "ReadingDate"])
    if bool(dup_mask.any()):
        bad = sorted(
            int(b) for b in df.loc[dup_mask, ids.HOUSEHOLD_BATCH].dropna().unique()
        )
        raise ValueError(
            f"[imputation] Duplicate ({ids.HOUSEHOLD_BATCH}, ReadingDate) rows "
            f"for dataset(s) {bad} -- a timestamp twice in one series "
            f"double-weights the project averages and breaks the diff math. "
            f"Resolve the duplication first."
        )


def _assert_batch_safe_ibis(tbl) -> None:
    """Engine-side variant of assert_imputation_input_batch_safe for lazy tables
    (cheap aggregates; the table is never materialised in full)."""
    from etdmap.index_helpers import HouseBatchOverlapError

    if ids.HOUSEHOLD_BATCH in tbl.columns:
        per_hh = (
            tbl.group_by(ids.HOUSEHOLD)
            .aggregate(n_batches=tbl[ids.HOUSEHOLD_BATCH].nunique())
        )
        multi = per_hh.filter(per_hh.n_batches > 1)[ids.HOUSEHOLD].execute()
        if len(multi):
            raise HouseBatchOverlapError(
                f"[prepare_diff_averages] Household(s) {sorted(int(h) for h in multi)} span "
                f"more than one batch; project averages would mix batches. "
                f"Batch-aware averaging is not implemented yet."
            )
    n_rows = int(tbl.count().execute())
    n_keys = int(tbl[[ids.HOUSEHOLD, "ReadingDate"]].distinct().count().execute())
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
    min_eligible: int = MIN_ELIGIBLE_DATASETS,
):
    """
    Sharded version of the avg-diffs stage: compute the imputation estimation
    input (avg_diffs.parquet) directly from the SHARDED mapped data.

    I/O shell only -- the computation is the SAME prepare_diff_averages_for_impute_ibis
    the legacy path uses, fed a lazy table built from mapped_household_table
    (DuckDB hive-glob, per the measured perf decision) with `ids.PROJECT` joined
    from the index and `Meenemen` inclusion applied at read time.

    Outputs are written under ``<aggregate_folder_path>/sharded/`` (the sharded
    pipeline's artifact area), keeping legacy artifacts untouched.

    Household-batch safety: refuses multi-household-batch households and overlapping household batch periods
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
        household_ids = sorted(int(x) for x in household_ids)
    else:
        household_ids = included_household_ids(mapped_folder_path)

    index_df, _ = read_index(mapped_folder_path)
    if ids.HOUSEHOLD_BATCH not in index_df.columns:
        raise ValueError(
            "index.parquet has no HuisBatchIdBSV column; "
            "re-run the index save to add the household-batch columns."
        )
    # Keyed on the household batch, NOT the household: a household's project assignment is
    # constant WITHIN a household batch but may differ BETWEEN household batches (project
    # definitions change over time; the household does not). So
    # ids.HOUSEHOLD -> ids.PROJECT is not a function over time, while
    # ids.HOUSEHOLD_BATCH -> ids.PROJECT is. ids.HOUSEHOLD_BATCH is also the primary key
    # of index.parquet (one row per household batch), so keying on ids.HOUSEHOLD
    # would additionally keep whichever household-batch row happened to come last.

    household_batch_projects = index_df[[ids.HOUSEHOLD_BATCH, ids.PROJECT]].dropna()
    project_map = _ibis.memtable(
        pd.DataFrame({
            ids.HOUSEHOLD_BATCH: household_batch_projects[ids.HOUSEHOLD_BATCH].astype("int64"),
            ids.PROJECT: household_batch_projects[ids.PROJECT].astype("int64"),
        })
    )

    tbl = mapped_household_table(mapped_folder_path)
    tbl = tbl.filter(tbl[ids.HOUSEHOLD].isin(household_ids))
    tbl = tbl.join(project_map, tbl[ids.HOUSEHOLD_BATCH] == project_map[ids.HOUSEHOLD_BATCH])

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
            project_id_column=ids.PROJECT,
            cumulative_columns=cumulative_columns,
            min_eligible=min_eligible,
        )
    finally:
        etdtransform.options.aggregate_folder_path = previous
