"""The averaging stage on a column that reports more coarsely than the grid.

A supplier can send a variable coarsely in either form, and the two are handled
by different halves of the same function:

- a delivered **register** reporting hourly: the window's volume is the rise
  between two readings and its length is the distance back to the previous one;
- a delivered **interval** reporting hourly: the value already *is* its window's
  volume, and the window is its cadence however long ago the last reading came.

Both end up dividing by twelve on a five-minute grid, but by different routes,
so each needs its own case. The same split applies to the bound: a reading
covering an hour is judged against an hour's worth, not a slot's, or a plausible
value is rejected.

Everything here reads the cadence from the sidecar. The register of a variable
delivered as intervals only still has a row in it -- carrying the grid as a
fallback -- so a lookup that let that row answer for the interval column would
report the column as reporting every slot and stop spreading it.

Synthetic frames only -- no pipeline, no fixtures.
"""

import pandas as pd
import pytest
from etdmap import ids
from etdmap.data_model import thresholds_dict
from etdmap.testing import (
    PAIRS_EMPTY,
    datasets_frame,
    household_batch_of,
    sidecar_frame,
)

from etdtransform.bounds import (
    cadence_minutes_by_household_batch,
    column_bound,
    grid_bound,
)
from etdtransform.impute import calculate_average_diff_ibis

pytestmark = pytest.mark.unit

GRID = 5
K = 12  # hourly on a five-minute grid
CUM = "ElektriciteitNetgebruik"
DIFF = f"{CUM}Diff"

DECLARED_MAX = thresholds_dict[DIFF]["Max"]  # what one grid slot may hold
#: The same bound scaled to an hourly reading. A '5-minute' threshold scales
#: sub-linearly: an hour cannot sustain a five-minute peak.
HOURLY_MAX = DECLARED_MAX * K**0.5

#: Two full days. The presence gate asks whether a household covered the
#: window its readings sit in, and that window is at least a day, so a shorter
#: frame is excluded wholesale and there is no average left to assert on.
N = 288 * 2
RISE = 0.6  # consumption delivered in one hour

#: The household every frame below describes. Its dataset -- keyed on
#: `HuisBatchIdBSV`, which the cadence sidecar and every lookup built from it
#: are keyed on too -- is ``household_batch_of(HOUSEHOLD)``.
HOUSEHOLD = 1

#: Where every frame below starts.
START = "2026-01-01"


def _coarse_register_cadence():
    """A register delivered hourly. Its interval column is derived later, so the
    sidecar has no row for it and the pairing has to answer."""
    return sidecar_frame(
        "cadence",
        [
            {"column": ids.READING_DATE, "cadence_minutes": GRID,
             "n_readings": N},
            {"column": CUM, "cadence_minutes": GRID * K, "n_readings": N // K},
        ],
        household_batch_id=household_batch_of(HOUSEHOLD),
    )


def _coarse_interval_cadence():
    """Intervals delivered hourly, register never sent -- so the register's row
    records no readings and carries the grid."""
    return sidecar_frame(
        "cadence",
        [
            {"column": ids.READING_DATE, "cadence_minutes": GRID,
             "n_readings": N},
            {"column": CUM, "cadence_minutes": GRID, "n_readings": 0},
            {"column": DIFF, "cadence_minutes": GRID * K,
             "n_readings": N // K},
        ],
        household_batch_id=household_batch_of(HOUSEHOLD),
    )


def _coarse_register_frame(n=N, rise=RISE):
    """A register reading every K rows; the interval column derived from it is
    empty between readings, which is what a sparse register differences to."""
    total, values = 100.0, []
    for row in range(n):
        if row % K == K - 1:
            total += rise
            values.append(total)
        else:
            values.append(pd.NA)
    df = datasets_frame(
        {household_batch_of(HOUSEHOLD): {CUM: values}},
        start=START,
        freq=f"{GRID}min",
        tz=None,
        # Both halves of the pair are stated here: the register above, its
        # differences below.
        pairs=False,
    )
    df[DIFF] = df[CUM].diff().round(10)
    df.loc[0, DIFF] = 0
    return df


def _coarse_interval_frame(n=N, reading=RISE):
    """An interval reading every K rows, and no register at all."""
    values = [pd.NA] * n
    for row in range(K - 1, n, K):
        values[row] = reading
    return datasets_frame(
        {household_batch_of(HOUSEHOLD): {DIFF: values}},
        start=START,
        freq=f"{GRID}min",
        tz=None,
        # The register is present and empty: the stages refuse half a pair, and
        # the averaging takes a dataset's volume from the register wherever that
        # column holds any value, which would displace the intervals stated
        # here.
        pairs=PAIRS_EMPTY,
    )


def _placed(df, cadence):
    """The per-slot project average, through the averaging the pipeline runs.

    One household contributes, so the project average at a slot is that
    household's own value there -- which is what makes this readable as "where
    did the reading end up".
    """
    import ibis

    result = calculate_average_diff_ibis(
        ibis.memtable(df), ids.PROJECT, [DIFF],
        cadence=cadence, grid_minutes=GRID,
        # One household contributes, which is the point of this file and is
        # smaller than a project, so the household floor is lowered rather than
        # the frame padded -- padding it would hide which reading went where.
        min_eligible=1,
    )
    avg = result[DIFF]["avg_diff"].sort_values("ReadingDate")
    return avg[f"{DIFF}_avg"].reset_index(drop=True)


class TestSpreadingACoarseDeliveredRegister:
    """The common case: a register reporting hourly. The window's volume is the
    rise between two readings and its length is the distance back to the
    previous one."""

    def test_the_rise_is_spread_over_the_slots_it_covers(self):
        placed = _placed(_coarse_register_frame(), _coarse_register_cadence())
        covered = placed.iloc[K:2 * K]
        assert covered.notna().all()
        assert covered.astype(float).tolist() == pytest.approx(
            [RISE / K] * K
        )

    def test_the_window_keeps_exactly_what_the_meter_recorded(self):
        placed = _placed(_coarse_register_frame(), _coarse_register_cadence())
        assert float(placed.iloc[K:2 * K].sum()) == pytest.approx(RISE)

    def test_it_is_not_left_as_one_reading_in_one_slot(self):
        """The failure this guards: reading the grid cadence instead of the
        column's would make the window one slot long and put a whole hour of
        consumption into it."""
        placed = _placed(_coarse_register_frame(), _coarse_register_cadence())
        assert float(placed.max()) == pytest.approx(RISE / K)
        assert float(placed.max()) < RISE

    def test_the_interval_column_is_found_through_its_register(self):
        """Cadence is measured on the columns as delivered, so a column derived
        later has no row of its own."""
        assert cadence_minutes_by_household_batch(DIFF, _coarse_register_cadence()) == {
            household_batch_of(HOUSEHOLD): float(GRID * K)
        }


class TestSpreadingACoarseDeliveredInterval:
    """No register, so the delivered value already is its window's volume and the
    window is its cadence."""

    def test_the_reading_is_spread_over_the_slots_it_covers(self):
        placed = _placed(_coarse_interval_frame(), _coarse_interval_cadence())
        covered = placed.iloc[K:2 * K]
        assert covered.notna().all()
        assert covered.astype(float).tolist() == pytest.approx(
            [RISE / K] * K
        )

    def test_the_window_keeps_exactly_what_was_delivered(self):
        placed = _placed(_coarse_interval_frame(), _coarse_interval_cadence())
        assert float(placed.iloc[K:2 * K].sum()) == pytest.approx(RISE)

    def test_it_is_not_left_as_one_reading_in_one_slot(self):
        placed = _placed(_coarse_interval_frame(), _coarse_interval_cadence())
        assert float(placed.max()) == pytest.approx(RISE / K)

    def test_the_absent_registers_row_does_not_answer_for_it(self):
        """The register's row records no readings and carries the grid. Letting
        it through would report the delivered interval as reporting every slot,
        and the spreading would stop."""
        assert cadence_minutes_by_household_batch(DIFF, _coarse_interval_cadence()) == {
            household_batch_of(HOUSEHOLD): float(GRID * K)
        }


class TestBoundingACoarseDeliveredRegister:
    def test_the_bound_is_scaled_to_the_hour_it_reports(self):
        df = _coarse_register_frame()
        bound = column_bound(df, DIFF, _coarse_register_cadence())
        assert float(pd.Series(bound).max()) == pytest.approx(HOURLY_MAX)

    def test_a_plausible_hourly_reading_is_not_rejected(self):
        """The failure this guards: judging an hour's consumption against one
        slot's bound rejects a reading the meter legitimately delivered."""
        reading = DECLARED_MAX * 2  # above a slot's bound, below an hour's
        assert reading > DECLARED_MAX
        assert reading < HOURLY_MAX
        df = _coarse_register_frame(rise=reading)
        bound = column_bound(df, DIFF, _coarse_register_cadence())
        delivered = df[DIFF].dropna()
        assert (delivered <= float(pd.Series(bound).max())).all()

    def test_the_grid_bound_is_the_one_it_must_not_be_judged_by(self):
        assert float(grid_bound(DIFF)) == pytest.approx(DECLARED_MAX)
        assert float(grid_bound(DIFF)) < HOURLY_MAX


class TestBoundingACoarseDeliveredInterval:
    def test_the_bound_is_scaled_to_the_hour_it_reports(self):
        df = _coarse_interval_frame()
        bound = column_bound(df, DIFF, _coarse_interval_cadence())
        assert float(pd.Series(bound).max()) == pytest.approx(HOURLY_MAX)

    def test_a_plausible_hourly_reading_is_not_rejected(self):
        reading = DECLARED_MAX * 2
        df = _coarse_interval_frame(reading=reading)
        bound = column_bound(df, DIFF, _coarse_interval_cadence())
        delivered = df[DIFF].dropna()
        assert (delivered <= float(pd.Series(bound).max())).all()

    def test_a_reading_above_even_the_hourly_bound_is_still_caught(self):
        """Scaling widens the bound; it does not remove it."""
        reading = HOURLY_MAX * 2
        df = _coarse_interval_frame(reading=reading)
        bound = column_bound(df, DIFF, _coarse_interval_cadence())
        delivered = df[DIFF].dropna()
        assert not (delivered <= float(pd.Series(bound).max())).all()


class TestTheTwoFormsAgreeOnWhatTheyPlace:
    def test_the_same_hourly_consumption_spreads_the_same_way(self):
        """The two routes through the function differ, but a register rising by
        0.6 over an hour and an interval delivering 0.6 for that hour describe
        the same consumption, so they must place the same values."""
        register = _placed(
            _coarse_register_frame(), _coarse_register_cadence()
        )
        interval = _placed(
            _coarse_interval_frame(), _coarse_interval_cadence()
        )
        window = slice(K, 2 * K)
        assert register.iloc[window].astype(float).tolist() == pytest.approx(
            interval.iloc[window].astype(float).tolist()
        )
