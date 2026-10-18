"""Placing a coarsely delivered interval reading on the grid slots it covers.

A supplier sending an hourly interval column puts an hour of consumption in one
five-minute cell. The reading is spread over the twelve slots it covers, so the
column reads as the grid-cadence column everything below it expects.

The property that matters most is what it does NOT do: an hour the meter never
reported stays unknown. The fill is limited to k-1 rows, so it cannot reach back
across a tick that carried no value and borrow from the hour after it.

Synthetic frames only -- no pipeline, no fixtures.
"""

import pandas as pd
import pytest

from etdtransform.bounds import resample_to_grid
from etdtransform.spread import (
    cadence_with_column_at_grid,
    spread_delivered_interval_to_grid,
    spread_steps_by_household,
)

pytestmark = pytest.mark.unit

GRID = 5
CUM = "ElektriciteitNetgebruik"
DIFF = "ElektriciteitNetgebruikDiff"
READING = 0.6


def _cadence(rows):
    """The sidecar as load_column_cadence returns it: (household, column) rows."""
    return pd.DataFrame({
        "HuisIdBSV": pd.array([r[0] for r in rows], dtype="Int64"),
        "column": pd.array([r[1] for r in rows], dtype="string"),
        "cadence_minutes": pd.array([r[2] for r in rows], dtype="Int64"),
        "n_readings": pd.array([r[3] for r in rows], dtype="Int64"),
    })


def _interval_only_cadence(household=1, reading_minutes=15, grid=GRID):
    """Intervals delivered, register never sent."""
    return _cadence([
        (household, "ReadingDate", grid, 1000),
        (household, CUM, grid, 0),
        (household, DIFF, reading_minutes, 100),
    ])


def _frame(values, household=1, n=None):
    n = len(values) if n is None else n
    return pd.DataFrame({
        "HuisIdBSV": pd.array([household] * n, dtype="Int64"),
        "ReadingDate": pd.date_range("2026-01-01", periods=n, freq="5min"),
        CUM: pd.array([pd.NA] * n, dtype="Float64"),
        DIFF: pd.array(values, dtype="Float64"),
    })


def _ticks(n, k, first=None, missing=()):
    """A reading every k rows, the first at *first* (default the k-th row)."""
    first = (k - 1) if first is None else first
    values = [pd.NA] * n
    for row in range(first, n, k):
        values[row] = READING
    for row in missing:
        values[row] = pd.NA
    return values


def _as_list(series):
    return [None if pd.isna(v) else round(float(v), 10) for v in series]


class TestTheSmokeTest:
    """One household, k = 3, 24 rows, and the tick at row 11 never delivered."""

    K = 3
    N = 24
    PER_SLOT = READING / 3

    @pytest.fixture
    def spread(self):
        df = _frame(_ticks(self.N, self.K, missing=(11,)))
        return spread_delivered_interval_to_grid(df, DIFF, {1: self.K})

    def test_the_first_windows_are_filled(self, spread):
        assert _as_list(spread[0:9]) == [
            pytest.approx(self.PER_SLOT)
        ] * 9

    def test_the_block_of_the_undelivered_tick_stays_unknown(self, spread):
        """The whole point: rows 9-11 were owed a reading that never came, so
        they stay pd.NA rather than taking a share of the next hour."""
        assert spread[9:12].isna().all()

    def test_the_fill_resumes_at_the_next_delivered_tick(self, spread):
        assert _as_list(spread[12:24]) == [
            pytest.approx(self.PER_SLOT)
        ] * 12

    def test_a_window_sums_back_to_its_delivered_reading(self, spread):
        assert float(spread[3:6].sum()) == pytest.approx(READING)

    def test_no_energy_is_created_or_lost(self, spread):
        """Seven delivered readings, one withheld."""
        delivered = _frame(_ticks(self.N, self.K, missing=(11,)))[DIFF]
        assert float(spread.sum()) == pytest.approx(float(delivered.sum()))

    def test_a_household_not_given_a_step_is_untouched(self):
        df = pd.concat(
            [
                _frame(_ticks(self.N, self.K), household=1),
                _frame(_ticks(self.N, self.K), household=2),
            ],
            ignore_index=True,
        )
        spread = spread_delivered_interval_to_grid(df, DIFF, {1: self.K})
        assert spread[df["HuisIdBSV"] == 2].isna().all()
        assert spread[df["HuisIdBSV"] == 1].notna().any()


class TestTheLeadingPartialWindow:
    def test_only_the_in_frame_share_is_placed(self):
        """A first reading on row 1 with k = 3 covers a slot before the frame
        starts. That share is consumption from before the dataset and is simply
        not placed."""
        df = _frame(_ticks(24, 3, first=1))
        spread = spread_delivered_interval_to_grid(df, DIFF, {1: 3})
        assert _as_list(spread[0:2]) == [pytest.approx(0.2)] * 2
        assert float(spread[0:2].sum()) == pytest.approx(0.4)


class TestTwoCadencesInOneCall:
    def test_each_household_is_spread_at_its_own_step(self):
        df = pd.concat(
            [
                _frame(_ticks(24, 3), household=1),
                _frame(_ticks(24, 12), household=2),
            ],
            ignore_index=True,
        )
        spread = spread_delivered_interval_to_grid(df, DIFF, {1: 3, 2: 12})
        first = spread[df["HuisIdBSV"] == 1].dropna()
        second = spread[df["HuisIdBSV"] == 2].dropna()
        assert float(first.iloc[0]) == pytest.approx(READING / 3)
        assert float(second.iloc[0]) == pytest.approx(READING / 12)


class TestAgreementWithTheAveraging:
    """The fill spreads with a limited backward fill; the averaging spreads with
    resample_to_grid. A household's own values and the project average it is
    scaled against have to sit on the same footing, because the interval branch
    multiplies one by the other."""

    @pytest.mark.parametrize("k", [3, 4, 12])
    @pytest.mark.parametrize("first", [None, 1])
    @pytest.mark.parametrize("missing", [(), (11,)])
    def test_the_two_agree_slot_for_slot(self, k, first, missing):
        df = _frame(_ticks(24, k, first=first, missing=missing))
        mine = spread_delivered_interval_to_grid(df, DIFF, {1: k})
        theirs = resample_to_grid(
            df, DIFF, _interval_only_cadence(reading_minutes=GRID * k), GRID
        )
        assert _as_list(mine) == _as_list(theirs)


class TestWhichHouseholdsAreSpread:
    def test_interval_only_delivery_is_spread(self):
        steps = spread_steps_by_household(
            CUM, DIFF, _interval_only_cadence(reading_minutes=60)
        )
        assert steps == {1: 12}

    def test_a_delivered_register_is_never_spread(self):
        """The register is authoritative and the cumulative logic reads it at
        its own cadence."""
        cadence = _cadence([
            (1, "ReadingDate", GRID, 1000),
            (1, CUM, 60, 100),
            (1, DIFF, 60, 100),
        ])
        assert spread_steps_by_household(CUM, DIFF, cadence) == {}

    def test_a_column_already_at_the_grid_is_not_spread(self):
        steps = spread_steps_by_household(
            CUM, DIFF, _interval_only_cadence(reading_minutes=GRID)
        )
        assert steps == {}

    def test_a_cadence_that_does_not_divide_the_grid_is_refused(self):
        """A declared cadence like this is refused at mapping, so one reaching
        here was measured -- detection noise on an irregular column. Refuse it
        quietly rather than failing the run."""
        steps = spread_steps_by_household(
            CUM, DIFF, _interval_only_cadence(reading_minutes=7)
        )
        assert steps == {}

    def test_nothing_delivered_is_not_spread(self):
        cadence = _cadence([
            (1, "ReadingDate", GRID, 1000),
            (1, CUM, GRID, 0),
            (1, DIFF, GRID, 0),
        ])
        assert spread_steps_by_household(CUM, DIFF, cadence) == {}

    def test_households_are_decided_one_by_one(self):
        """Form is per (household, column): one household's register absence
        does not make its neighbour's column interval-only."""
        cadence = _cadence([
            (1, "ReadingDate", GRID, 1000),
            (1, CUM, GRID, 0),
            (1, DIFF, 60, 100),
            (2, "ReadingDate", GRID, 1000),
            (2, CUM, 60, 100),
            (2, DIFF, 60, 100),
        ])
        assert spread_steps_by_household(CUM, DIFF, cadence) == {1: 12}

    def test_each_household_takes_its_own_grid(self):
        """A dataset spans batches, so a household on a coarser grid must not be
        stepped against the finest of them."""
        cadence = _cadence([
            (1, "ReadingDate", 5, 1000),
            (1, CUM, 5, 0),
            (1, DIFF, 60, 100),
            (2, "ReadingDate", 15, 1000),
            (2, CUM, 15, 0),
            (2, DIFF, 60, 100),
        ])
        assert spread_steps_by_household(CUM, DIFF, cadence) == {1: 12, 2: 4}


class TestTheCadenceAmendment:
    def test_a_spread_column_reads_at_the_grid(self):
        cadence = _interval_only_cadence(reading_minutes=60)
        amended = cadence_with_column_at_grid(cadence, DIFF, [1])
        row = amended[(amended["HuisIdBSV"] == 1) & (amended["column"] == DIFF)]
        assert int(row["cadence_minutes"].iloc[0]) == GRID

    def test_no_other_row_moves(self):
        cadence = _cadence([
            (1, "ReadingDate", GRID, 1000),
            (1, CUM, GRID, 0),
            (1, DIFF, 60, 100),
            (2, "ReadingDate", GRID, 1000),
            (2, DIFF, 60, 100),
        ])
        amended = cadence_with_column_at_grid(cadence, DIFF, [1])
        untouched = amended["HuisIdBSV"] != 1
        pd.testing.assert_frame_equal(
            amended[untouched].reset_index(drop=True),
            cadence[untouched].reset_index(drop=True),
        )

    def test_the_table_handed_in_is_not_modified(self):
        """It belongs to the caller of the fill, and what happened inside one
        pass is not a fact about the mapped data."""
        cadence = _interval_only_cadence(reading_minutes=60)
        before = cadence.copy()
        cadence_with_column_at_grid(cadence, DIFF, [1])
        pd.testing.assert_frame_equal(cadence, before)

    def test_naming_no_household_changes_nothing(self):
        cadence = _interval_only_cadence(reading_minutes=60)
        amended = cadence_with_column_at_grid(cadence, DIFF, [])
        pd.testing.assert_frame_equal(amended, cadence)
