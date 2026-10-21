"""A project average requires enough of the project to be present.

Without a minimum the average is the mean of whoever happened to report. Three
gates fix that: a household counts toward a day only if it covered that day; a
day is averaged at all only if enough households were eligible for it; and a
slot has an average only if enough of that day's eligible households delivered
for it.

The household floor is not implied by the contributor fraction, because the
fraction scales with the count it is judging: at one eligible household it
demands one, so that household's own readings came out as the project's average
with nothing in the number saying so.

Every gate removes values. None substitutes one: a slot with too few
contributors has no average, which is a different statement from an average of
two households and is not the same as zero.

Frames here hold a handful of households so a case can be spelled out, which is
below the production floor. Those tests pass ``min_eligible=1`` to say so --
the floor itself is exercised at its production value in
``TestTheHouseholdFloor``.
"""

import ibis
import pandas as pd
import pytest

from conftest import project_averages
from etdtransform.impute import calculate_average_diff_ibis
from etdtransform.presence import (
    MIN_ELIGIBLE_HOUSEHOLDS,
    MIN_WINDOW_COVERAGE,
    SUFFICIENCY_FRACTION,
    SUFFICIENCY_SHAPE,
    WINDOW_MINUTES,
    slot_minimum,
    window_index,
    window_is_averageable,
    window_minutes_for_column,
)

pytestmark = pytest.mark.unit

CUM = "ElektriciteitNetgebruik"
COL = f"{CUM}Diff"
GRID = 5
SLOTS_PER_DAY = 288


def _frame(per_household, start="2026-01-01", freq="5min", project=1):
    """One frame from ``{household: [values]}``, all on the same slots."""
    rows = []
    for household, values in per_household.items():
        dates = pd.date_range(start, periods=len(values), freq=freq)
        for date, value in zip(dates, values):
            rows.append({
                "HuisIdBSV": household,
                "ProjectIdBSV": project,
                "ReadingDate": date,
                COL: value,
            })
    frame = pd.DataFrame(rows)
    frame["HuisIdBSV"] = frame["HuisIdBSV"].astype("Int64")
    frame["ProjectIdBSV"] = frame["ProjectIdBSV"].astype("Int64")
    frame[COL] = frame[COL].astype("Float64")
    return frame


def _day(value=1.0, missing=0):
    """One day of five-minute slots, *missing* of them absent."""
    return [pd.NA] * missing + [value] * (SLOTS_PER_DAY - missing)


def _cadence_for(frame, minutes=GRID):
    """A cadence table for *frame*'s households.

    Recorded under the cumulative name, which is where cadence is recorded and
    where the interval column resolves it from, plus the ``ReadingDate`` row
    carrying the grid.
    """
    households = sorted(int(h) for h in frame["HuisIdBSV"].unique())
    rows = [
        {"HuisIdBSV": household, "column": column, "cadence_minutes": minutes}
        for household in households
        for column in (CUM, "ReadingDate")
    ]
    return pd.DataFrame(rows).astype({
        "HuisIdBSV": "Int64", "column": "string", "cadence_minutes": "Int64",
    })


def _averaged(frame, **kwargs):
    """The averaging engine the pipeline runs, on *frame*.

    The gates are judged here rather than by the pandas twin that used to stand
    in for them: that twin lost its production caller when the pandas averaging
    engine was deleted, so what it decided was no longer what the pipeline
    decides.

    *min_eligible* defaults to 1 because these frames hold a handful of
    households, which is below the production floor by construction. The floor
    itself is exercised at its production value in ``TestTheHouseholdFloor``.
    """
    kwargs.setdefault("min_eligible", 1)
    return calculate_average_diff_ibis(
        ibis.memtable(frame), "ProjectIdBSV", [COL],
        cadence=_cadence_for(frame), grid_minutes=GRID, **kwargs
    )[COL]


def _presence_of(frame, **kwargs):
    """The per (household, window) counts and judgements the gates made."""
    return _averaged(frame, **kwargs)["window_presence"]


def _published(frame, **kwargs):
    """The average per slot, as published: pd.NA where a gate removed it."""
    avg = _averaged(frame, **kwargs)["avg_diff"]
    return avg.sort_values(["ProjectIdBSV", "ReadingDate"]).reset_index(
        drop=True
    )


def _by_household(frame, **kwargs):
    """The presence row per household, indexed by HuisIdBSV."""
    return _presence_of(frame, **kwargs).set_index("HuisIdBSV")


class TestTheWindowIsACalendarDay:
    def test_a_day_of_five_minute_slots_is_one_window(self):
        dates = pd.date_range("2026-01-01", periods=SLOTS_PER_DAY, freq="5min")
        assert window_index(pd.Series(dates), WINDOW_MINUTES).nunique() == 1

    def test_the_next_day_is_the_next_window(self):
        dates = pd.Series(pd.date_range("2026-01-01", periods=SLOTS_PER_DAY + 1,
                                        freq="5min"))
        windows = window_index(dates, WINDOW_MINUTES)
        assert windows.nunique() == 2
        assert windows.iloc[-1] == windows.iloc[0] + 1

    def test_windows_start_at_midnight_utc_not_at_the_first_reading(self):
        """Aligned to the clock, so the same day is one window for every
        household and the eligible count at a slot is comparable."""
        late = pd.Series(pd.date_range("2026-01-01 23:55", periods=2, freq="5min"))
        windows = window_index(late, WINDOW_MINUTES)
        assert windows.iloc[0] != windows.iloc[1]

    def test_two_households_starting_at_different_times_share_a_window(self):
        early = pd.Series(pd.date_range("2026-01-01 00:00", periods=1, freq="5min"))
        later = pd.Series(pd.date_range("2026-01-01 18:00", periods=1, freq="5min"))
        assert (
            window_index(early, WINDOW_MINUTES).iloc[0]
            == window_index(later, WINDOW_MINUTES).iloc[0]
        )


class TestTheWindowGrowsWithTheCadence:
    def test_a_column_at_the_grid_takes_the_base_window(self):
        assert window_minutes_for_column([5.0]) == float(WINDOW_MINUTES)

    def test_a_column_reporting_hourly_still_takes_the_base_window(self):
        assert window_minutes_for_column([60.0]) == float(WINDOW_MINUTES)

    def test_a_column_reporting_less_often_than_a_day_takes_its_own_interval(self):
        """A window shorter than one reporting interval cannot be covered at
        all, so every household would be ineligible."""
        assert window_minutes_for_column([10080.0]) == 10080.0

    def test_the_coarsest_cadence_decides(self):
        """One window grid per column keeps the eligible count at a slot well
        defined."""
        assert window_minutes_for_column([5.0, 60.0, 10080.0]) == 10080.0

    @pytest.mark.parametrize("cadence", [None, [], [pd.NA]])
    def test_no_known_cadence_takes_the_base_window(self, cadence):
        assert window_minutes_for_column(cadence) == float(WINDOW_MINUTES)

    def test_a_scalar_is_accepted(self):
        assert window_minutes_for_column(10080.0) == 10080.0


class TestCoverageIsMeasuredOverWhatTheWindowHeld:
    """The counts the engine measured, and the verdict it reached from them.

    Asserted as the engine's own ``n_delivered``, ``n_slots`` and ``eligible``.
    Dividing the two in the test and checking the quotient would be restating
    the gate's formula rather than testing it.
    """

    def test_a_full_window_is_delivered_for_every_slot_it_held(self):
        row = _by_household(_frame({1: _day()})).loc[1]
        assert int(row["n_slots"]) == SLOTS_PER_DAY
        assert int(row["n_delivered"]) == SLOTS_PER_DAY
        assert bool(row["eligible"])

    def test_a_half_empty_window_is_delivered_for_half_and_fails(self):
        half = SLOTS_PER_DAY // 2
        row = _by_household(_frame({1: _day(missing=half)})).loc[1]
        assert int(row["n_slots"]) == SLOTS_PER_DAY
        assert int(row["n_delivered"]) == SLOTS_PER_DAY - half
        assert not bool(row["eligible"])

    def test_a_partial_window_is_judged_on_what_it_could_have_held(self):
        """A window clipped by the start of the dataset is not counted short for
        sitting at the edge: a household present for every slot it has covered
        that window. So the denominator is the twelve slots it held, not a day."""
        row = _by_household(
            _frame({1: [1.0] * 12}, start="2026-01-01 23:00")
        ).loc[1]
        assert int(row["n_slots"]) == 12
        assert int(row["n_delivered"]) == 12
        assert bool(row["eligible"])

    def test_the_measurement_is_per_household(self):
        rows = _by_household(
            _frame({1: _day(), 2: _day(missing=SLOTS_PER_DAY // 2)})
        )
        assert int(rows.loc[1, "n_delivered"]) == SLOTS_PER_DAY
        assert int(rows.loc[2, "n_delivered"]) == SLOTS_PER_DAY // 2
        assert bool(rows.loc[1, "eligible"])
        assert not bool(rows.loc[2, "eligible"])


class TestEligibilityIsAllOrNothingPerWindow:
    def test_a_household_covering_the_window_is_eligible(self):
        assert bool(_presence_of(_frame({1: _day()}))["eligible"].all())

    def test_a_household_just_below_the_minimum_is_not(self):
        missing = int(SLOTS_PER_DAY * (1 - MIN_WINDOW_COVERAGE)) + 1
        presence = _presence_of(_frame({1: _day(missing=missing)}))
        assert not bool(presence["eligible"].any())

    def test_the_readings_it_did_deliver_are_excluded_too(self):
        """The gate is the household's presence over the window, not the
        presence of the individual value: a household reporting for one hour of
        a day says nothing about that day. So the hour it did deliver shapes
        no average either.
        """
        frame = _frame({1: [1.0] * 12 + [pd.NA] * (SLOTS_PER_DAY - 12)})
        assert not bool(_presence_of(frame)["eligible"].any())
        assert _published(frame)[f"{COL}_avg"].isna().all()

    def test_eligibility_is_decided_per_window_not_per_household(self):
        """A household that covers Monday and not Tuesday contributes to Monday."""
        frame = _frame({1: _day() + _day(missing=SLOTS_PER_DAY // 2)})
        by_window = (
            _presence_of(frame).sort_values("window_start")["eligible"].tolist()
        )
        assert by_window == [True, False]


class TestTheSlotMinimum:
    def test_it_is_a_fraction_of_the_eligible_households(self):
        assert slot_minimum(4) == 2
        assert slot_minimum(10) == 5

    def test_it_rounds_up(self):
        """Half of three is two, not one: rounding down would let a minority
        set the level."""
        assert slot_minimum(3) == 2

    def test_it_is_capped(self):
        """A large project should not need an unreachable count."""
        assert slot_minimum(400) == SUFFICIENCY_SHAPE
        assert slot_minimum(10_000) == SUFFICIENCY_SHAPE

    def test_the_cap_binds_from_twice_the_shape_upward(self):
        """Half of 48 is 24, so the fraction still decides there; from 49 up it
        reaches the cap and stays."""
        assert slot_minimum(2 * SUFFICIENCY_SHAPE - 2) == SUFFICIENCY_SHAPE - 1
        assert slot_minimum(2 * SUFFICIENCY_SHAPE - 1) == SUFFICIENCY_SHAPE
        assert slot_minimum(2 * SUFFICIENCY_SHAPE) == SUFFICIENCY_SHAPE

    @pytest.mark.parametrize("n", [0, None, pd.NA, -1])
    def test_no_eligible_households_demands_nothing(self, n):
        """There is nothing to average either way, so the gate has no work."""
        assert slot_minimum(n) == 0

    def test_one_eligible_household_needs_itself(self):
        assert slot_minimum(1) == 1

    def test_the_declared_fraction_is_what_is_applied(self):
        assert slot_minimum(8, fraction=SUFFICIENCY_FRACTION) == 4


class TestTheGatesRemoveRatherThanSubstitute:
    """What the published average looks like where a gate fired.

    Judged on the engine's own output. Every gate removes: a slot it refuses
    carries pd.NA, which is a different statement from an average of whoever
    happened to report, and is never a zero.
    """

    AVG = f"{COL}_avg"

    def _thin_household(self):
        """Three households covering the day, one covering only an hour."""
        return _frame({
            1: _day(), 2: _day(), 3: _day(),
            4: [1.0] * 12 + [pd.NA] * (SLOTS_PER_DAY - 12),
        })

    def test_a_project_where_everyone_covers_the_day_is_averaged(self):
        frame = _frame({h: _day() for h in range(1, 5)})
        presence = _presence_of(frame)
        assert bool(presence["eligible"].all())
        assert int(presence["n_eligible"].max()) == 4
        assert _published(frame)[self.AVG].notna().all()

    def test_a_household_that_did_not_cover_the_day_is_dropped(self):
        presence = _presence_of(self._thin_household()).set_index("HuisIdBSV")
        assert not bool(presence.loc[4, "eligible"])
        # It delivered twelve slots and none of them count.
        assert int(presence.loc[4, "n_delivered"]) == 12
        assert int(presence.loc[1, "n_eligible"]) == 3

    def test_the_others_still_have_their_average(self):
        """Dropping a household removes its contribution, not the slot."""
        assert _published(self._thin_household())[self.AVG].notna().all()

    def test_a_slot_with_too_few_contributors_is_dropped(self):
        """Four households cover the day, so a slot needs two. A slot only one
        of them delivered for has no average."""
        thin = _day()
        thin[100] = pd.NA
        frame = _frame({
            1: _day(), 2: list(thin), 3: list(thin), 4: list(thin),
        })
        published = _published(frame)
        at_slot = published["ReadingDate"] == published["ReadingDate"].iloc[100]
        assert published.loc[at_slot, self.AVG].isna().all()
        assert int(published[self.AVG].isna().sum()) == 1, (
            "only that slot loses its average"
        )

    def test_nothing_is_substituted_where_a_value_was_dropped(self):
        """Not zero, and not the surviving household's own reading carried
        across: the slot has no average at all."""
        thin = _day()
        thin[100] = pd.NA
        frame = _frame({
            1: _day(), 2: list(thin), 3: list(thin), 4: list(thin),
        })
        published = _published(frame)
        empty = published[self.AVG].isna()
        assert bool(empty.any())
        assert not (published.loc[empty, self.AVG] == 0).any()

    def test_the_input_frame_is_not_modified(self):
        frame = self._thin_household()
        before = frame[COL].copy()
        _published(frame)
        pd.testing.assert_series_equal(frame[COL], before)

    def test_the_published_average_is_a_nullable_float(self):
        assert _published(_frame({1: _day()}))[self.AVG].dtype == "Float64"

    def test_the_counts_account_for_every_slot_the_window_held(self):
        """The artifact has to be readable as an account of the window, so the
        slots a household delivered for and the slots it did not must add up to
        the slots the window held for it."""
        presence = _presence_of(self._thin_household())
        assert (presence["n_delivered"] <= presence["n_slots"]).all()
        assert int(presence["n_slots"].min()) == SLOTS_PER_DAY

    def test_the_counts_report_the_window_they_were_judged_over(self):
        presence = _presence_of(_frame({1: _day() + _day()}))
        assert presence["window_minutes"].unique().tolist() == [
            float(WINDOW_MINUTES)
        ]
        assert len(presence) == 2, "one row per household and window"

    def test_the_counts_report_how_many_household_windows_qualified(self):
        frame = _frame({
            1: _day(), 2: [1.0] * 12 + [pd.NA] * (SLOTS_PER_DAY - 12),
        })
        presence = _presence_of(frame)
        assert len(presence) == 2
        assert int(presence["eligible"].sum()) == 1


class TestProjectsAreJudgedSeparately:
    def test_one_projects_absence_does_not_disqualify_anothers_slot(self):
        one = _frame({h: _day() for h in range(1, 5)}, project=1)
        thin = _day()
        thin[50] = pd.NA
        two = _frame({5: thin, 6: thin, 7: thin, 8: thin}, project=2)
        frame = pd.concat([one, two], ignore_index=True)

        published = _published(frame)
        avg = f"{COL}_avg"
        at_slot = published["ReadingDate"] == sorted(
            published["ReadingDate"].unique()
        )[50]
        first = published["ProjectIdBSV"] == 1
        assert published.loc[at_slot & first, avg].notna().all()
        assert published.loc[at_slot & ~first, avg].isna().all()


class TestAThinProjectLosesItsAverage:
    AVG = f"{COL}_avg"

    def test_a_single_household_project_is_not_published_at_all(self):
        """A project of one has no project average. The contributor fraction
        cannot say so -- it demands one of one, which that household always
        meets -- so this is the household floor's job, and until it existed one
        household's own readings were published as its project's average."""
        frame = _frame({1: _day()})
        published = _published(frame, min_eligible=MIN_ELIGIBLE_HOUSEHOLDS)
        presence = _presence_of(frame, min_eligible=MIN_ELIGIBLE_HOUSEHOLDS)

        assert published[self.AVG].isna().all()
        assert not bool(presence["averageable"].any())
        # The fraction found nothing wrong, which is the point: one eligible
        # household needs one contributor and had it.
        assert int(presence["slot_minimum"].iloc[0]) == 1
        assert bool(presence["eligible"].all())

    def test_it_keeps_the_slots_it_delivered_when_the_floor_is_lowered(self):
        """The same frame with the floor lowered, so the removal above is
        attributable to the floor and not to the other two gates."""
        assert _published(_frame({1: _day()}))[self.AVG].notna().all()

    def test_a_single_household_project_that_stops_reporting_loses_the_day(self):
        frame = _frame({1: [1.0] * 12 + [pd.NA] * (SLOTS_PER_DAY - 12)})
        assert _published(frame)[self.AVG].isna().all()


class TestTheHouseholdFloor:
    """At the production value. Everything else in this file lowers the floor
    to isolate another gate, so this is the only place the shipped number is
    exercised."""

    def test_the_floor_is_ten(self):
        assert MIN_ELIGIBLE_HOUSEHOLDS == 10

    @pytest.mark.parametrize("n", [0, 1, 2, 9])
    def test_below_the_floor_is_not_averageable(self, n):
        assert window_is_averageable(n) is False

    @pytest.mark.parametrize("n", [10, 11, 400])
    def test_at_or_above_the_floor_is(self, n):
        assert window_is_averageable(n) is True

    @pytest.mark.parametrize("n", [None, pd.NA])
    def test_an_unknown_count_is_not_averageable(self, n):
        """The gate exists to stop an average resting on households nobody
        counted, so an uncounted window is exactly what it refuses."""
        assert window_is_averageable(n) is False

    def test_the_floor_carries_a_contributor_minimum_of_five(self):
        """No second constant is needed: ten eligible households already
        demand five contributors through the fraction."""
        assert slot_minimum(MIN_ELIGIBLE_HOUSEHOLDS) == 5

    def test_a_project_of_nine_loses_its_average(self):
        frame = _frame({h: _day() for h in range(1, 10)})
        published = _published(frame, min_eligible=MIN_ELIGIBLE_HOUSEHOLDS)
        presence = _presence_of(frame, min_eligible=MIN_ELIGIBLE_HOUSEHOLDS)
        assert published[f"{COL}_avg"].isna().all()
        assert not bool(presence["averageable"].any())
        assert int(presence["n_eligible"].max()) == 9

    def test_a_project_of_ten_keeps_it(self):
        frame = _frame({h: _day() for h in range(1, 11)})
        published = _published(frame, min_eligible=MIN_ELIGIBLE_HOUSEHOLDS)
        presence = _presence_of(frame, min_eligible=MIN_ELIGIBLE_HOUSEHOLDS)
        assert published[f"{COL}_avg"].notna().all()
        assert bool(presence["averageable"].all())
        assert int(presence["n_eligible"].max()) == 10

    def test_eligibility_is_what_is_counted_not_household_rows(self):
        """Eleven households of which two never covered the day leaves nine
        eligible, so the day is not averageable even though eleven are present.
        The floor is a statement about who was reporting, not about who exists
        in the frame."""
        absent = [1.0] * 12 + [pd.NA] * (SLOTS_PER_DAY - 12)
        per_household = {h: _day() for h in range(1, 10)}
        per_household[10] = list(absent)
        per_household[11] = list(absent)

        frame = _frame(per_household)
        presence = _presence_of(frame, min_eligible=MIN_ELIGIBLE_HOUSEHOLDS)
        assert len(presence) == 11, "every household has a row"
        assert int(presence["eligible"].sum()) == 9
        assert int(presence["n_eligible"].max()) == 9
        assert _published(
            frame, min_eligible=MIN_ELIGIBLE_HOUSEHOLDS
        )[f"{COL}_avg"].isna().all()

    def test_the_averaging_applies_the_floor_too(self):
        """The ibis path is what production runs. A floor in the pandas twin
        and not in it would make them two algorithms."""
        cum = "ElektriciteitNetgebruik"
        parts = []
        for household in range(1, 10):
            values = _day()
            part = pd.DataFrame({
                "HuisIdBSV": pd.array(
                    [household] * len(values), dtype="Int64"
                ),
                "ProjectIdBSV": pd.array([1] * len(values), dtype="Int64"),
                "ReadingDate": pd.date_range(
                    "2026-01-01", periods=len(values), freq="5min"
                ),
                COL: pd.array(values, dtype="Float64"),
            })
            part[cum] = part[COL].fillna(0).cumsum().astype("Float64")
            parts.append(part)
        frame = pd.concat(parts, ignore_index=True)
        households = sorted(int(h) for h in frame["HuisIdBSV"].unique())
        cadence = pd.DataFrame({
            "HuisIdBSV": pd.array(households, dtype="Int64"),
            "column": pd.array([cum] * len(households), dtype="string"),
            "cadence_minutes": pd.array([5] * len(households), dtype="Int64"),
        })

        avg = calculate_average_diff_ibis(
            ibis.memtable(frame), "ProjectIdBSV", [COL],
            cadence=cadence, grid_minutes=5,
        )[COL]["avg_diff"]
        assert avg[f"{COL}_avg"].isna().all(), "nine households is not a project"


class TestTheAveragingAppliesTheGates:
    """The ibis path is what production runs; the pandas path is what the tests
    reason about. A gate in one and not the other makes them two algorithms."""

    CUM = "ElektriciteitNetgebruik"

    def _frame(self, per_household, project=1):
        """Cumulative readings that difference to *per_household*'s values."""
        parts = []
        for household, values in per_household.items():
            frame = pd.DataFrame({
                "HuisIdBSV": pd.array([household] * len(values), dtype="Int64"),
                "ProjectIdBSV": pd.array([project] * len(values), dtype="Int64"),
                "ReadingDate": pd.date_range(
                    "2026-01-01", periods=len(values), freq="5min"
                ),
                COL: pd.array(values, dtype="Float64"),
            })
            frame[self.CUM] = frame[COL].fillna(0).cumsum().astype("Float64")
            parts.append(frame)
        return pd.concat(parts, ignore_index=True)

    def _cadence(self, frame):
        households = sorted(int(h) for h in frame["HuisIdBSV"].unique())
        return pd.DataFrame({
            "HuisIdBSV": pd.array(households, dtype="Int64"),
            "column": pd.array([self.CUM] * len(households), dtype="string"),
            "cadence_minutes": pd.array([5] * len(households), dtype="Int64"),
        })

    def _averaged(self, frame):
        return project_averages(
            frame, "ProjectIdBSV", [COL],
            cadence=self._cadence(frame), grid_minutes=5,
            # The two per-slot gates are what this class compares across the
            # paths; the floor is compared in TestTheHouseholdFloor.
            min_eligible=1,
        )[COL]["avg_diff"]

    def test_a_day_everyone_covers_is_averaged_throughout(self):
        """Slot 0 has no average: an interval series has no reading before its
        first, so there is nothing for that slot to average."""
        avg = self._averaged(
            self._frame({h: _day() for h in range(1, 5)})
        ).sort_values("ReadingDate")
        assert len(avg) == SLOTS_PER_DAY
        assert pd.isna(avg[f"{COL}_avg"].iloc[0])
        assert avg[f"{COL}_avg"].iloc[1:].notna().all()

    def test_a_thin_day_still_averages_the_slots_the_others_cover(self):
        """A household covering only the first twelve slots is excluded from the
        rest of the day rather than removing it: the three that did cover it
        still have an average."""
        frame = self._frame({
            1: _day(), 2: _day(), 3: _day(),
            4: [1.0] * 12 + [pd.NA] * (SLOTS_PER_DAY - 12),
        })
        avg = self._averaged(frame).sort_values("ReadingDate")
        covered = avg[f"{COL}_avg"].notna()
        assert bool(covered.iloc[1:12].all())
        assert int(covered.sum()) == SLOTS_PER_DAY - 1
