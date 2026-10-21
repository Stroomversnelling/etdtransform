"""How much of a project has to be present before its average means anything.

A project average is the mean of the values at a slot. Without a minimum, the
contributor set changes as households come and go, so the average changes shape
for reasons that have nothing to do with consumption: a household joining
mid-month moves the level, and one dropping out moves it back.

Three gates close that:

Window eligibility
    Per (household, column, window). A household's readings count only where it
    was reporting through the window -- coverage at or above
    ``MIN_WINDOW_COVERAGE``. A household present for an hour of a day says
    nothing about that day.

Project sufficiency
    Per (project, column, window). The window is averaged at all only where at
    least ``MIN_ELIGIBLE_HOUSEHOLDS`` were eligible for it. The slot fraction
    below cannot carry this, because it scales with the count it is judging.

Slot sufficiency
    Per (project, column, slot). A slot has an average only where enough of the
    eligible households delivered a value for it. The minimum is
    ``min(SUFFICIENCY_SHAPE, ceil(SUFFICIENCY_FRACTION * n_eligible))``: a
    fraction of the project, capped so a large project does not need an
    unreachable count.

Coverage is measured on the resampled column, after the outage windows have
been dropped, so it counts slots a meter actually delivered for rather than
slots a fill occupied.

This module holds the constants and the judgements. The gates are applied by
the averaging engine (``impute.calculate_average_diff_ibis``), which supplies
the counts to :func:`window_presence_frame` and reads its verdicts.
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

#: The eligible households a window needs before its average is published at
#: all. A project average is a shape taken from the project, and the slot
#: fraction alone does not say there is a project behind it: at one eligible
#: household it demands one contributor, which that household always meets, so
#: its own readings would be published as the project's average.
MIN_ELIGIBLE_HOUSEHOLDS = 10


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


def window_is_averageable(n_eligible, minimum=MIN_ELIGIBLE_HOUSEHOLDS) -> bool:
    """Whether a window has enough eligible households to average at all.

    A separate question from :func:`slot_minimum`, which asks how many of them a
    slot needs, and both have to hold. A window of three households satisfies
    the contributor fraction with two and still does not describe a project, so
    the fraction cannot carry this on its own -- it scales with the count it is
    meant to be judging.

    An unknown count is not averageable: the gate exists to stop an average
    resting on households nobody counted.
    """
    if n_eligible is None or pd.isna(n_eligible):
        return False
    return int(n_eligible) >= int(minimum)


def window_presence_frame(
    counts,
    column,
    window_minutes,
    project_id_column,
    household_column="HuisIdBSV",
    min_coverage=MIN_WINDOW_COVERAGE,
    shape=SUFFICIENCY_SHAPE,
    fraction=SUFFICIENCY_FRACTION,
    min_eligible=MIN_ELIGIBLE_HOUSEHOLDS,
):
    """One row per (project, household, window): the counts the presence gates
    judged, and the judgements derived from them.

    *counts* carries *project_id_column*, *household_column*, ``_window`` (the
    :func:`window_index` value) and the two raw counts ``n_slots`` and
    ``n_delivered``. Everything else -- eligibility, the per-window household
    tallies, the slot minimum and whether the window is averageable at all --
    is derived here, so a caller supplies counts and reads verdicts rather than
    carrying its own copy of the rules.

    ``n_contributing`` tallies the eligible households that delivered at least
    one slot of the window. At the default coverage minimum it equals
    ``n_eligible``; the two part when the minimum is lowered.

    ``averageable`` is the household floor (:func:`window_is_averageable`).
    Recorded per row rather than left implicit, because a window with no average
    and a window with a thin one are different facts and the counts alone do not
    separate them.

    ``n_outage_windows``, ``n_outage_slots`` and ``source`` are carried through
    from *counts* when present. They are measurements of the spreading rather
    than judgements derivable here, so a caller that does not spread omits them.
    """
    out = counts.copy()
    out["n_slots"] = out["n_slots"].astype("Int64")
    out["n_delivered"] = out["n_delivered"].astype("Int64")
    out["eligible"] = (
        ((out["n_delivered"] / out["n_slots"]) >= min_coverage)
        .fillna(False)
        .astype("boolean")
    )
    window_keys = [project_id_column, "_window"]
    out["n_eligible"] = (
        out.groupby(window_keys, sort=False)["eligible"]
        .transform("sum")
        .astype("Int64")
    )
    contributing = out["eligible"] & (out["n_delivered"] > 0)
    out["n_contributing"] = (
        contributing.groupby([out[project_id_column], out["_window"]], sort=False)
        .transform("sum")
        .astype("Int64")
    )
    out["slot_minimum"] = pd.array(
        [slot_minimum(n, shape, fraction) for n in out["n_eligible"]],
        dtype="Int64",
    )
    out["averageable"] = pd.array(
        [window_is_averageable(n, min_eligible) for n in out["n_eligible"]],
        dtype="boolean",
    )
    # The inverse of window_index: windows count int(window_minutes)-minute
    # steps from the epoch.
    out["window_start"] = pd.to_datetime(
        out["_window"].astype("int64") * int(window_minutes), unit="m"
    )
    out["window_minutes"] = pd.array(
        [float(window_minutes)] * len(out), dtype="Float64"
    )
    out["column"] = column
    ordered = [
        "column",
        project_id_column,
        household_column,
        "window_start",
        "window_minutes",
        "n_slots",
        "n_delivered",
        "eligible",
        "n_eligible",
        "n_contributing",
        "slot_minimum",
        "averageable",
    ]
    # Measurements the caller made that this function cannot derive. Optional:
    # a caller that does not spread has none of them to offer.
    ordered += [
        name for name in ("n_outage_windows", "n_outage_slots", "source")
        if name in out.columns
    ]
    return (
        out[ordered]
        .sort_values([project_id_column, household_column, "window_start"])
        .reset_index(drop=True)
    )
