"""How much of a project has to be present before its average means anything.

A project average was the mean of whatever values existed at a slot, with no
minimum. The contributor set then changes as households come and go, so the
average changes shape for reasons that have nothing to do with consumption: a
household joining mid-month moves the level, and one dropping out moves it back.

Two gates close that:

Window eligibility
    Per (household, column, window). A household's readings count only where it
    was reporting through the window -- coverage at or above
    ``MIN_WINDOW_COVERAGE``. A household present for an hour of a day says
    nothing about that day.

Slot sufficiency
    Per (project, column, slot). A slot has an average only where enough of the
    eligible households delivered a value for it. The minimum is
    ``min(SUFFICIENCY_SHAPE, ceil(SUFFICIENCY_FRACTION * n_eligible))``: a
    fraction of the project, capped so a large project does not need an
    unreachable count.

Coverage is measured on the resampled column, after the outage windows have
been dropped, so it counts slots a meter actually delivered for rather than
slots a fill occupied.
"""

import logging
from math import ceil

import pandas as pd

#: The window eligibility is judged over, in minutes. A day is the shortest
#: period over which a household's presence is a property of the household
#: rather than of the hour.
WINDOW_MINUTES = 1440

#: The fraction of a window's slots a household must have delivered for.
MIN_WINDOW_COVERAGE = 0.95

#: The largest contributor count ever demanded of a slot.
SUFFICIENCY_SHAPE = 25

#: The fraction of the eligible households demanded of a slot below that cap.
SUFFICIENCY_FRACTION = 0.5


def window_minutes_for_column(cadence_minutes, base=WINDOW_MINUTES) -> float:
    """The window length for a column reporting every *cadence_minutes*.

    A window shorter than one reporting interval cannot be covered at all, so a
    column reporting less often than the base window is judged over its own
    interval instead.

    *cadence_minutes* may be a scalar, an iterable or empty. The coarsest
    cadence decides, since one window grid per column keeps the eligible count
    at a slot well defined.
    """
    if cadence_minutes is None:
        return float(base)
    if isinstance(cadence_minutes, (int, float)):
        values = [cadence_minutes]
    else:
        values = [v for v in cadence_minutes if pd.notna(v)]
    if not values:
        return float(base)
    return float(max(float(base), max(float(v) for v in values)))


def window_index(reading_dates: pd.Series, window_minutes: float) -> pd.Series:
    """Which window each timestamp falls in, counted from the epoch.

    Windows are aligned to 00:00 UTC rather than to the first reading, so the
    same calendar day is one window for every household and the eligible count
    at a slot is comparable across them.
    """
    minutes = pd.to_datetime(reading_dates).astype("int64") // (60 * 1_000_000_000)
    return (minutes // int(window_minutes)).astype("Int64")


def window_coverage(frame, column, window_column, household_column="HuisIdBSV"):
    """The fraction of its window's slots each household delivered a value for.

    Returned per row, so it aligns with *frame*.

    The denominator is the rows the household has in that window, not a full
    window's worth. A window clipped by the start or the end of the dataset is
    then judged on what it could have held, rather than counted short for
    sitting at the edge.
    """
    grouped = frame.groupby([household_column, window_column], sort=False)[column]
    present = grouped.transform("count")
    total = grouped.transform("size")
    return (present / total).astype("Float64")


def eligible_rows(
    frame,
    column,
    window_column,
    household_column="HuisIdBSV",
    min_coverage=MIN_WINDOW_COVERAGE,
) -> pd.Series:
    """Rows belonging to a (household, window) that met the coverage minimum."""
    coverage = window_coverage(frame, column, window_column, household_column)
    return (coverage >= min_coverage).fillna(False)


def slot_minimum(n_eligible, shape=SUFFICIENCY_SHAPE, fraction=SUFFICIENCY_FRACTION):
    """The contributor count a slot needs, given *n_eligible* households.

    A fraction of the project, capped at *shape*: a project of four needs two,
    a project of four hundred needs twenty-five rather than two hundred.
    Zero eligible households needs zero, which no slot can fail -- there is
    nothing to average either way.
    """
    if n_eligible is None or pd.isna(n_eligible) or int(n_eligible) <= 0:
        return 0
    return int(min(int(shape), ceil(float(fraction) * int(n_eligible))))


def apply_presence_filter(
    frame,
    column,
    project_id_column,
    window_minutes=WINDOW_MINUTES,
    household_column="HuisIdBSV",
    date_column="ReadingDate",
    min_coverage=MIN_WINDOW_COVERAGE,
    shape=SUFFICIENCY_SHAPE,
    fraction=SUFFICIENCY_FRACTION,
):
    """Blank the values a project average may not be taken over.

    Returns ``(values, stats)``: *column* with the excluded values replaced by
    pd.NA, and a dict describing what the two gates removed. *frame* is not
    modified.

    A value is kept when its (household, window) covered the window and its
    (project, slot) has enough eligible households delivering for it. Both
    gates remove values rather than substitute any: a slot no longer having an
    average is the finding, and an average of a handful of households is not a
    better answer than none.
    """
    work = frame[[household_column, project_id_column, date_column, column]].copy()
    work["_window"] = window_index(work[date_column], window_minutes)

    eligible = eligible_rows(
        work, column, "_window", household_column, min_coverage
    )

    # Contributors at a slot are the eligible households delivering for it; the
    # eligible count is over the whole window the slot sits in, so a slot is
    # judged against the households that were reporting that day rather than
    # against whoever happens to be present at that instant.
    contributing = eligible & work[column].notna()
    work["_contributors"] = (
        contributing.groupby([work[project_id_column], work[date_column]])
        .transform("sum")
        .astype("Int64")
    )

    eligible_households = (
        work.loc[eligible, [project_id_column, "_window", household_column]]
        .drop_duplicates()
        .groupby([project_id_column, "_window"], sort=False)
        .size()
        .rename("_n_eligible")
    )
    work = work.join(
        eligible_households, on=[project_id_column, "_window"], how="left"
    )
    work["_n_eligible"] = work["_n_eligible"].fillna(0).astype("Int64")
    work["_minimum"] = pd.array(
        [slot_minimum(n, shape, fraction) for n in work["_n_eligible"]],
        dtype="Int64",
    )

    sufficient = (work["_contributors"] >= work["_minimum"]).fillna(False)
    keep = eligible & sufficient

    delivered = work[column].notna()
    values = work[column].where(keep, pd.NA)

    stats = {
        "n_windows": int(work["_window"].nunique()),
        "window_minutes": float(window_minutes),
        "n_delivered": int(delivered.sum()),
        "n_dropped_ineligible": int((delivered & ~eligible).sum()),
        "n_dropped_insufficient": int((delivered & eligible & ~sufficient).sum()),
        "n_kept": int((delivered & keep).sum()),
        "eligible_household_windows": int(
            work.loc[eligible, [household_column, "_window"]]
            .drop_duplicates()
            .shape[0]
        ),
        "household_windows": int(
            work[[household_column, "_window"]].drop_duplicates().shape[0]
        ),
    }
    if stats["n_dropped_ineligible"] or stats["n_dropped_insufficient"]:
        logging.info(
            f"{column}: presence filter dropped "
            f"{stats['n_dropped_ineligible']} value(s) from households not "
            f"covering their window and {stats['n_dropped_insufficient']} "
            f"from slots with too few contributors; {stats['n_kept']} kept."
        )

    return values.astype("Float64"), stats
