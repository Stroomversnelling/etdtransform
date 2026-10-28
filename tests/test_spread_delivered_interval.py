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
from conftest import spread_average
from etdmap import ids
from etdmap.testing import (
    PAIRS_EMPTY,
    cadence_sidecar,
    household_batches_frame,
    one_to_one_test_household_batch_of,
    sidecar_frame,
)

from etdtransform.bounds import cadence_minutes_by_household_batch
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
    """The sidecar as load_column_cadence returns it: (household batch, column) rows.

    Each row is ``(household, column, cadence_minutes, n_readings)``. Cadence is
    a fact about one body of data, so the row is recorded under the household
    household batch that household's rows carry. Rows are kept in the order given, which
    is what lets a test state an order a lookup must not depend on.
    """
    return sidecar_frame("cadence", [
        {
            ids.HOUSEHOLD_BATCH: one_to_one_test_household_batch_of(household),
            "column": column,
            "cadence_minutes": cadence_minutes,
            "n_readings": n_readings,
        }
        for household, column, cadence_minutes, n_readings in rows
    ])


def _interval_only_cadence(household=1, reading_minutes=15, grid=GRID):
    """Intervals delivered, register never sent."""
    return cadence_sidecar(
        {CUM: grid, DIFF: reading_minutes},
        grid_minutes=grid,
        n_readings={"ReadingDate": 1000, CUM: 0, DIFF: 100},
        household_batch_id=one_to_one_test_household_batch_of(household),
    )


def _frame(values, household=1):
    """One household batch's rows, carrying *values* as the delivered interval column.

    The register is present and empty throughout, which is what keeps the
    interval column the volume source where the frame reaches the averaging.
    """
    return household_batches_frame(
        {one_to_one_test_household_batch_of(household): {DIFF: list(values)}},
        start="2026-01-01",
        freq="5min",
        tz=None,
        pairs=PAIRS_EMPTY,
    )


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
    """One household batch, k = 3, 24 rows, and the tick at row 11 never delivered."""

    K = 3
    N = 24
    PER_SLOT = READING / 3

    @pytest.fixture
    def spread(self):
        df = _frame(_ticks(self.N, self.K, missing=(11,)))
        return spread_delivered_interval_to_grid(
            df, DIFF, {one_to_one_test_household_batch_of(1): self.K}
        )

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
        """A step is given per household batch, so only the named household batch's rows move."""
        df = pd.concat(
            [
                _frame(_ticks(self.N, self.K), household=1),
                _frame(_ticks(self.N, self.K), household=2),
            ],
            ignore_index=True,
        )
        spread = spread_delivered_interval_to_grid(
            df, DIFF, {one_to_one_test_household_batch_of(1): self.K}
        )
        assert spread[
            df[ids.HOUSEHOLD_BATCH] == one_to_one_test_household_batch_of(2)
        ].isna().all()
        assert spread[
            df[ids.HOUSEHOLD_BATCH] == one_to_one_test_household_batch_of(1)
        ].notna().any()


class TestTheLeadingPartialWindow:
    def test_only_the_in_frame_share_is_placed(self):
        """A first reading on row 1 with k = 3 covers a slot before the frame
        starts. That share is consumption from before the household batch and is simply
        not placed."""
        df = _frame(_ticks(24, 3, first=1))
        spread = spread_delivered_interval_to_grid(
            df, DIFF, {one_to_one_test_household_batch_of(1): 3}
        )
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
        spread = spread_delivered_interval_to_grid(
            df, DIFF, {one_to_one_test_household_batch_of(1): 3, one_to_one_test_household_batch_of(2): 12}
        )
        first = spread[
            df[ids.HOUSEHOLD_BATCH] == one_to_one_test_household_batch_of(1)
        ].dropna()
        second = spread[
            df[ids.HOUSEHOLD_BATCH] == one_to_one_test_household_batch_of(2)
        ].dropna()
        assert float(first.iloc[0]) == pytest.approx(READING / 3)
        assert float(second.iloc[0]) == pytest.approx(READING / 12)


class TestAgreementWithTheAveraging:
    """The fill spreads with a limited backward fill; the averaging expands each
    window against a table of offsets in SQL. A household batch's own values and the
    project average they are scaled against have to sit on the same footing,
    because the interval branch multiplies one by the other.

    One household batch contributes, so the project average at a slot is that household batch's
    own spread value there."""

    @pytest.mark.parametrize("k", [3, 4, 12])
    @pytest.mark.parametrize("first", [None, 1])
    @pytest.mark.parametrize("missing", [(), (11,)])
    def test_the_two_agree_slot_for_slot(self, k, first, missing):
        df = _frame(_ticks(24, k, first=first, missing=missing))
        mine = spread_delivered_interval_to_grid(
            df, DIFF, {one_to_one_test_household_batch_of(1): k}
        )
        avg, _, _ = spread_average(
            df, _interval_only_cadence(reading_minutes=GRID * k), DIFF, GRID,
            drop_na=False,
        )
        theirs = avg.sort_values("ReadingDate")[f"{DIFF}_avg"]
        assert _as_list(mine) == _as_list(theirs)


class TestWhichHouseholdsAreSpread:
    def test_interval_only_delivery_is_spread(self):
        steps = spread_steps_by_household(
            CUM, DIFF, _interval_only_cadence(reading_minutes=60)
        )
        assert steps == {one_to_one_test_household_batch_of(1): 12}

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
        """Form is per (household batch, column): one household batch's register absence does
        not make its neighbour's column interval-only."""
        cadence = _cadence([
            (1, "ReadingDate", GRID, 1000),
            (1, CUM, GRID, 0),
            (1, DIFF, 60, 100),
            (2, "ReadingDate", GRID, 1000),
            (2, CUM, 60, 100),
            (2, DIFF, 60, 100),
        ])
        assert spread_steps_by_household(CUM, DIFF, cadence) == {
            one_to_one_test_household_batch_of(1): 12
        }

    def test_each_household_takes_its_own_grid(self):
        """An aggregated artifact spans household batches, so a household batch on a coarser grid
        must not be stepped against the finest of them."""
        cadence = _cadence([
            (1, "ReadingDate", 5, 1000),
            (1, CUM, 5, 0),
            (1, DIFF, 60, 100),
            (2, "ReadingDate", 15, 1000),
            (2, CUM, 15, 0),
            (2, DIFF, 60, 100),
        ])
        assert spread_steps_by_household(CUM, DIFF, cadence) == {
            one_to_one_test_household_batch_of(1): 12, one_to_one_test_household_batch_of(2): 4,
        }


class TestWhichCadenceEachStageReads:
    """A coarsely delivered interval is judged at its own cadence until this
    pass moves its values onto the grid. Two different questions, and the sidecar
    answers both -- so each has to read the row that applies to it."""

    def test_the_delivered_cadence_comes_from_the_columns_own_row(self):
        """The register of an interval-only variable still has a row, carrying
        the grid as a fallback. Letting it through would report the delivered
        interval as reporting every slot, and the averaging would stop spreading
        it."""
        cadence = _interval_only_cadence(reading_minutes=60)
        assert cadence_minutes_by_household_batch(DIFF, cadence) == {
            one_to_one_test_household_batch_of(1): 60.0
        }

    def test_the_winner_does_not_depend_on_row_order(self):
        reordered = _cadence([
            (1, DIFF, 60, 100),
            (1, CUM, GRID, 0),
            (1, "ReadingDate", GRID, 1000),
        ])
        assert cadence_minutes_by_household_batch(DIFF, reordered) == {
            one_to_one_test_household_batch_of(1): 60.0
        }

    def test_a_derived_column_still_inherits_from_its_register(self):
        """Cadence is measured on the columns as delivered, so a column the
        pipeline derives later has no row of its own. The pairing is what
        answers for it."""
        register_only = _cadence([
            (1, "ReadingDate", GRID, 1000),
            (1, CUM, 60, 100),
        ])
        assert cadence_minutes_by_household_batch(DIFF, register_only) == {
            one_to_one_test_household_batch_of(1): 60.0
        }

    def test_the_averaging_spreads_a_coarse_column_at_its_own_cadence(self):
        """Not at the grid: one hourly reading is twelve slots of consumption,
        so the averaging divides by twelve. Reading the grid here would place
        the whole hour in one slot."""
        df = _frame(_ticks(24, 12))
        avg, _, _ = spread_average(
            df, _interval_only_cadence(reading_minutes=60), DIFF, GRID
        )
        placed = avg[f"{DIFF}_avg"].dropna()
        assert float(placed.iloc[-1]) == pytest.approx(READING / 12)
        assert int(placed.count()) > 12


class TestTheCadenceAmendment:
    def test_a_spread_column_reads_at_the_grid(self):
        cadence = _interval_only_cadence(reading_minutes=60)
        amended = cadence_with_column_at_grid(
            cadence, DIFF, [one_to_one_test_household_batch_of(1)]
        )
        row = amended[
            (amended[ids.HOUSEHOLD_BATCH] == one_to_one_test_household_batch_of(1))
            & (amended["column"] == DIFF)
        ]
        assert int(row["cadence_minutes"].iloc[0]) == GRID

    def test_no_other_row_moves(self):
        cadence = _cadence([
            (1, "ReadingDate", GRID, 1000),
            (1, CUM, GRID, 0),
            (1, DIFF, 60, 100),
            (2, "ReadingDate", GRID, 1000),
            (2, DIFF, 60, 100),
        ])
        amended = cadence_with_column_at_grid(
            cadence, DIFF, [one_to_one_test_household_batch_of(1)]
        )
        untouched = amended[ids.HOUSEHOLD_BATCH] != one_to_one_test_household_batch_of(1)
        pd.testing.assert_frame_equal(
            amended[untouched].reset_index(drop=True),
            cadence[untouched].reset_index(drop=True),
        )

    def test_the_table_handed_in_is_not_modified(self):
        """It belongs to the caller of the fill, and what happened inside one
        pass is not a fact about the mapped data."""
        cadence = _interval_only_cadence(reading_minutes=60)
        before = cadence.copy()
        cadence_with_column_at_grid(cadence, DIFF, [one_to_one_test_household_batch_of(1)])
        pd.testing.assert_frame_equal(cadence, before)

    def test_naming_no_household_changes_nothing(self):
        cadence = _interval_only_cadence(reading_minutes=60)
        amended = cadence_with_column_at_grid(cadence, DIFF, [])
        pd.testing.assert_frame_equal(amended, cadence)
