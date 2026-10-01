"""A project average requires enough of the project to be present.

Without a minimum the average is the mean of whoever happened to report. Two
gates fix that: a household counts toward a day only if it covered that day, and
a slot has an average only if enough of the day's eligible households delivered
for it.

Both gates remove values. Neither substitutes one: a slot with too few
contributors has no average, which is a different statement from an average of
two households and is not the same as zero.
"""

import pandas as pd
import pytest

from etdtransform.presence import (
    MIN_WINDOW_COVERAGE,
    SUFFICIENCY_FRACTION,
    SUFFICIENCY_SHAPE,
    WINDOW_MINUTES,
    apply_presence_filter,
    eligible_rows,
    slot_minimum,
    window_coverage,
    window_index,
    window_minutes_for_column,
)

pytestmark = pytest.mark.unit

COL = "ElektriciteitNetgebruikDiff"
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
    def test_a_full_window_covers_completely(self):
        frame = _frame({1: _day()})
        frame["_w"] = window_index(frame["ReadingDate"], WINDOW_MINUTES)
        assert float(window_coverage(frame, COL, "_w").iloc[0]) == 1.0

    def test_a_half_empty_window_covers_half(self):
        frame = _frame({1: _day(missing=SLOTS_PER_DAY // 2)})
        frame["_w"] = window_index(frame["ReadingDate"], WINDOW_MINUTES)
        assert float(window_coverage(frame, COL, "_w").iloc[0]) == pytest.approx(0.5)

    def test_a_partial_window_is_judged_on_what_it_could_have_held(self):
        """A window clipped by the start of the dataset is not counted short for
        sitting at the edge: a household present for every slot it has covered
        that window."""
        frame = _frame({1: [1.0] * 12}, start="2026-01-01 23:00")
        frame["_w"] = window_index(frame["ReadingDate"], WINDOW_MINUTES)
        assert float(window_coverage(frame, COL, "_w").iloc[0]) == 1.0

    def test_coverage_is_per_household(self):
        frame = _frame({1: _day(), 2: _day(missing=SLOTS_PER_DAY // 2)})
        frame["_w"] = window_index(frame["ReadingDate"], WINDOW_MINUTES)
        coverage = window_coverage(frame, COL, "_w")
        by_household = frame.assign(c=coverage).groupby("HuisIdBSV")["c"].first()
        assert float(by_household.loc[1]) == 1.0
        assert float(by_household.loc[2]) == pytest.approx(0.5)


class TestEligibilityIsAllOrNothingPerWindow:
    def test_a_household_covering_the_window_is_eligible(self):
        frame = _frame({1: _day()})
        frame["_w"] = window_index(frame["ReadingDate"], WINDOW_MINUTES)
        assert eligible_rows(frame, COL, "_w").all()

    def test_a_household_just_below_the_minimum_is_not(self):
        missing = int(SLOTS_PER_DAY * (1 - MIN_WINDOW_COVERAGE)) + 1
        frame = _frame({1: _day(missing=missing)})
        frame["_w"] = window_index(frame["ReadingDate"], WINDOW_MINUTES)
        assert not eligible_rows(frame, COL, "_w").any()

    def test_the_readings_it_did_deliver_are_excluded_too(self):
        """The gate is the household's presence over the window, not the
        presence of the individual value: a household reporting for one hour of
        a day says nothing about that day."""
        frame = _frame({1: [1.0] * 12 + [pd.NA] * (SLOTS_PER_DAY - 12)})
        frame["_w"] = window_index(frame["ReadingDate"], WINDOW_MINUTES)
        eligible = eligible_rows(frame, COL, "_w")
        assert not eligible[frame[COL].notna()].any()

    def test_eligibility_is_decided_per_window_not_per_household(self):
        """A household that covers Monday and not Tuesday contributes to Monday."""
        frame = _frame({1: _day() + _day(missing=SLOTS_PER_DAY // 2)})
        frame["_w"] = window_index(frame["ReadingDate"], WINDOW_MINUTES)
        eligible = eligible_rows(frame, COL, "_w")
        by_window = frame.assign(e=eligible).groupby("_w")["e"].first()
        assert bool(by_window.iloc[0])
        assert not bool(by_window.iloc[1])


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


class TestTheFilterRemovesRatherThanSubstitutes:
    def _filtered(self, frame, **kwargs):
        return apply_presence_filter(frame, COL, "ProjectIdBSV", **kwargs)

    def test_a_project_where_everyone_covers_the_day_keeps_everything(self):
        frame = _frame({h: _day() for h in range(1, 5)})
        values, stats, _presence = self._filtered(frame)
        assert values.notna().all()
        assert stats["n_dropped_ineligible"] == 0
        assert stats["n_dropped_insufficient"] == 0

    def test_a_household_that_did_not_cover_the_day_is_dropped(self):
        frame = _frame({1: _day(), 2: _day(), 3: _day(), 4: [1.0] * 12
                        + [pd.NA] * (SLOTS_PER_DAY - 12)})
        values, stats, _presence = self._filtered(frame)
        assert stats["n_dropped_ineligible"] == 12
        dropped = frame["HuisIdBSV"] == 4
        assert values[dropped].isna().all()

    def test_the_others_are_untouched(self):
        frame = _frame({1: _day(), 2: _day(), 3: _day(), 4: [1.0] * 12
                        + [pd.NA] * (SLOTS_PER_DAY - 12)})
        values, _, _presence = self._filtered(frame)
        assert values[frame["HuisIdBSV"] != 4].notna().all()

    def test_a_slot_with_too_few_contributors_is_dropped(self):
        """Four households cover the day, so a slot needs two. A slot only one
        of them delivered for has no average."""
        thin = _day()
        thin[100] = pd.NA
        thinner = _day()
        thinner[100] = pd.NA
        third = _day()
        third[100] = pd.NA
        frame = _frame({1: _day(), 2: thin, 3: thinner, 4: third})
        values, stats, _presence = self._filtered(frame)
        assert stats["n_dropped_insufficient"] == 1
        at_slot = frame["ReadingDate"] == frame["ReadingDate"].unique()[100]
        assert values[at_slot].isna().all()

    def test_nothing_is_filled_in_where_a_value_was_dropped(self):
        frame = _frame({1: _day(), 2: _day(), 3: _day(), 4: [1.0] * 12
                        + [pd.NA] * (SLOTS_PER_DAY - 12)})
        values, _, _presence = self._filtered(frame)
        assert values.isna().sum() >= frame[COL].isna().sum()

    def test_the_frame_is_not_modified(self):
        frame = _frame({1: _day(), 2: [1.0] * 12 + [pd.NA] * (SLOTS_PER_DAY - 12)})
        before = frame[COL].copy()
        self._filtered(frame)
        pd.testing.assert_series_equal(frame[COL], before)

    def test_the_result_is_a_nullable_float(self):
        frame = _frame({1: _day()})
        values, _, _presence = self._filtered(frame)
        assert values.dtype == "Float64"

    def test_the_stats_account_for_every_delivered_value(self):
        frame = _frame({1: _day(), 2: _day(), 3: _day(), 4: [1.0] * 12
                        + [pd.NA] * (SLOTS_PER_DAY - 12)})
        _, stats, _presence = self._filtered(frame)
        assert stats["n_delivered"] == (
            stats["n_kept"]
            + stats["n_dropped_ineligible"]
            + stats["n_dropped_insufficient"]
        )

    def test_the_stats_report_the_window(self):
        frame = _frame({1: _day() + _day()})
        _, stats, _presence = self._filtered(frame)
        assert stats["n_windows"] == 2
        assert stats["window_minutes"] == float(WINDOW_MINUTES)

    def test_the_stats_report_how_many_household_windows_qualified(self):
        frame = _frame({1: _day(), 2: [1.0] * 12 + [pd.NA] * (SLOTS_PER_DAY - 12)})
        _, stats, _presence = self._filtered(frame)
        assert stats["household_windows"] == 2
        assert stats["eligible_household_windows"] == 1


class TestProjectsAreJudgedSeparately:
    def test_one_projects_absence_does_not_disqualify_anothers_slot(self):
        one = _frame({h: _day() for h in range(1, 5)}, project=1)
        thin = _day()
        thin[50] = pd.NA
        two = _frame({5: thin, 6: thin, 7: thin, 8: thin}, project=2)
        frame = pd.concat([one, two], ignore_index=True)
        values, _, _presence = apply_presence_filter(frame, COL, "ProjectIdBSV")
        at_slot = frame["ReadingDate"] == frame["ReadingDate"].unique()[50]
        assert values[at_slot & (frame["ProjectIdBSV"] == 1)].notna().all()
        assert values[at_slot & (frame["ProjectIdBSV"] == 2)].isna().all()


class TestAThinProjectLosesItsAverage:
    def test_a_single_household_project_keeps_the_slots_it_delivered(self):
        """One eligible household needs itself, so a project of one is not
        silenced -- but it is also not padded out to look like more."""
        frame = _frame({1: _day()})
        values, stats, _presence = apply_presence_filter(frame, COL, "ProjectIdBSV")
        assert values.notna().all()
        assert stats["n_dropped_insufficient"] == 0

    def test_a_single_household_project_that_stops_reporting_loses_the_day(self):
        frame = _frame({1: [1.0] * 12 + [pd.NA] * (SLOTS_PER_DAY - 12)})
        values, _, _presence = apply_presence_filter(frame, COL, "ProjectIdBSV")
        assert values.isna().all()


class TestBothAveragingPathsApplyTheGates:
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

    def _both(self, frame):
        import ibis

        from etdtransform.impute import (
            calculate_average_diff,
            calculate_average_diff_ibis,
        )

        cadence = self._cadence(frame)
        pandas_result = calculate_average_diff(
            frame.copy(), "ProjectIdBSV", [COL], cadence=cadence, grid_minutes=5
        )[COL]["avg_diff"]
        ibis_result = calculate_average_diff_ibis(
            ibis.memtable(frame), "ProjectIdBSV", [COL],
            cadence=cadence, grid_minutes=5,
        )[COL]["avg_diff"]
        merged = pandas_result.merge(
            ibis_result, on=["ProjectIdBSV", "ReadingDate"],
            suffixes=("_pandas", "_ibis"),
        )
        assert len(merged) == len(pandas_result), "the two paths report different slots"
        return merged

    def test_they_agree_where_everyone_covers_the_day(self):
        merged = self._both(self._frame({h: _day() for h in range(1, 5)}))
        for _, row in merged.iterrows():
            a, b = row[f"{COL}_avg_pandas"], row[f"{COL}_avg_ibis"]
            assert (pd.isna(a) and pd.isna(b)) or float(a) == pytest.approx(float(b))

    def test_they_agree_on_the_slots_a_thin_day_removes(self):
        frame = self._frame({
            1: _day(), 2: _day(), 3: _day(),
            4: [1.0] * 12 + [pd.NA] * (SLOTS_PER_DAY - 12),
        })
        merged = self._both(frame)
        pandas_missing = merged[f"{COL}_avg_pandas"].isna()
        ibis_missing = merged[f"{COL}_avg_ibis"].isna()
        pd.testing.assert_series_equal(
            pandas_missing, ibis_missing, check_names=False
        )
