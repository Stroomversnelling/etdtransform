"""The upspread as a step of the imputation pass.

The spread runs inside ``impute_and_normalize``: before the fill reads the
column for what is missing, because its values change which cells are empty, and
recorded afterwards, because the fill routes assign into a findings column
created all-NA between the two.

What the pass then judges the column by also changes. A column reporting every
grid row is bounded per slot, not per hour, so the amendment the spread makes to
the cadence it holds has to reach the threshold pass.

Synthetic frames only -- no pipeline, no fixtures.
"""

import logging

import pandas as pd
import pytest
from etdmap import ids
from etdmap.testing import (
    PAIRS_EMPTY,
    datasets_frame,
    household_batch_of,
    sidecar_columns,
    sidecar_frame,
)

from etdtransform.vectorized_impute import ImputeType, impute_and_normalize

pytestmark = pytest.mark.unit

GRID = 5
CUM = "ElektriciteitNetgebruik"
DIFF = f"{CUM}Diff"
MAX_PER_5MIN = 1.5  # thresholds.csv bound for DIFF
MAX_PER_HOUR = 1.5 * 12**0.5  # the same bound scaled to an hourly reading

K = 12  # an hourly column on a 5-minute grid
N = 48
#: The household every frame below belongs to.
HOUSEHOLD = 1


@pytest.fixture(autouse=True)
def _quiet_logs():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _interval_only_frame(reading=1.2, missing=(), n=N):
    """Hourly interval readings, no register, one project average per row."""
    values = [pd.NA] * n
    for row in range(K - 1, n, K):
        values[row] = reading
    for row in missing:
        values[row] = pd.NA

    # PAIRS_EMPTY: the register is the one that never arrived, so the pair is
    # present and holds nothing, leaving the interval column the only source.
    return datasets_frame(
        {household_batch_of(HOUSEHOLD): {DIFF: values}},
        start="2026-01-01",
        freq="5min",
        tz=None,
        pairs=PAIRS_EMPTY,
        **{f"{DIFF}_avg": reading / K},
    )


def _interval_only_cadence(household=HOUSEHOLD):
    return sidecar_frame(
        "cadence",
        [
            {"column": "ReadingDate", "cadence_minutes": GRID, "n_readings": N},
            {"column": CUM, "cadence_minutes": GRID, "n_readings": 0},
            {"column": DIFF, "cadence_minutes": GRID * K, "n_readings": N // K},
        ],
        household_batch_id=household_batch_of(household),
    )


def _impute(df, cadence):
    out, _, _ = impute_and_normalize(
        df.copy(), [CUM], ids.PROJECT, cadence=cadence
    )
    return out


def _types(out):
    col = f"{DIFF}_impute_type"
    return out[col].fillna(0).astype("Int64")


def _has(out, flag):
    return ((_types(out) & flag.value) != 0).tolist()


class TestTheSpreadRunsInThePass:
    def test_the_coarse_reading_is_placed_on_the_slots_it_covers(self):
        out = _impute(_interval_only_frame(), _interval_only_cadence())
        assert float(out[DIFF].iloc[0]) == pytest.approx(1.2 / K)
        assert int(out[DIFF].notna().sum()) == N

    def test_the_placed_cells_are_recorded(self):
        out = _impute(_interval_only_frame(), _interval_only_cadence())
        assert all(_has(out, ImputeType.RESAMPLED_TO_GRID))

    def test_the_branch_is_recorded_even_with_nothing_left_to_impute(self):
        """A complete coarse column leaves no gap, so the fill takes its
        nothing-to-impute path -- but the spread resolved those cells, and which
        logic resolved them is recorded there too."""
        out = _impute(_interval_only_frame(), _interval_only_cadence())
        assert all(_has(out, ImputeType.DELIVERED_INTERVAL_IMPUTATION))

    def test_the_record_survives_the_fill(self):
        """The routes assign into a findings column created all-NA, so a bit
        written before them would be replaced. One cell is left for the fill to
        resolve, so a route does run."""
        out = _impute(
            _interval_only_frame(missing=(11,)), _interval_only_cadence()
        )
        placed = _has(out, ImputeType.RESAMPLED_TO_GRID)
        assert sum(placed) == N - K
        assert placed[12] is True

    def test_a_window_the_meter_never_reported_is_left_to_the_fill(self):
        """Rows 0-11 hold no delivered reading, so the spread places nothing
        there and they reach the fill as missing."""
        out = _impute(
            _interval_only_frame(missing=(11,)), _interval_only_cadence()
        )
        assert not any(_has(out, ImputeType.RESAMPLED_TO_GRID)[0:12])

    def test_the_delivered_total_is_preserved(self):
        before = _interval_only_frame()
        out = _impute(before, _interval_only_cadence())
        assert float(out[DIFF].sum()) == pytest.approx(
            float(before[DIFF].sum())
        )

    def test_the_pre_spread_values_are_kept_for_comparison(self):
        """OldDiff is captured before the spread, so a comparison against a
        previous run shows this pass placing the energy."""
        out = _impute(_interval_only_frame(), _interval_only_cadence())
        assert int(out[f"{CUM}OldDiff"].notna().sum()) == N // K


class TestWhatTheColumnIsJudgedBy:
    def test_a_spread_column_is_bounded_per_slot(self):
        """An hourly reading of 24 is itself above its hourly bound, but what
        reaches the threshold pass is twelve cells of 2.0 -- each above the
        per-slot bound of 1.5 and flagged, which only happens if the cadence the
        pass holds followed the values onto the grid."""
        reading = 24.0
        assert reading / K > MAX_PER_5MIN
        out = _impute(
            _interval_only_frame(reading=reading), _interval_only_cadence()
        )
        assert any(_has(out, ImputeType.THRESHOLD_VIOLATED))

    def test_a_reading_plausible_per_slot_is_left_alone(self):
        reading = 1.2
        assert reading / K < MAX_PER_5MIN
        out = _impute(
            _interval_only_frame(reading=reading), _interval_only_cadence()
        )
        assert not any(_has(out, ImputeType.THRESHOLD_VIOLATED))


class TestOtherDeliveryFormsAreUntouched:
    def _register_frame(self, n=N):
        """A register delivered hourly; its interval column derived from it.

        The register's values are the subject here, so they are stated rather
        than taken from a builder, and the interval column is the difference
        between them.
        """
        rise = 1.2
        values, total = [], 100.0
        for row in range(n):
            if row % K == K - 1:
                total += rise
            values.append(total if row % K == K - 1 else pd.NA)
        # pairs=False: both halves are supplied here, the register directly and
        # the interval column by differencing it.
        df = datasets_frame(
            {household_batch_of(HOUSEHOLD): {CUM: values}},
            start="2026-01-01",
            freq="5min",
            tz=None,
            pairs=False,
            **{f"{DIFF}_avg": rise / K},
        )
        df[DIFF] = df[CUM].diff().round(10)
        df.loc[0, DIFF] = 0
        return df

    def _register_cadence(self):
        return sidecar_frame(
            "cadence",
            [
                {"column": "ReadingDate", "cadence_minutes": GRID,
                 "n_readings": N},
                {"column": CUM, "cadence_minutes": GRID * K,
                 "n_readings": N // K},
                {"column": DIFF, "cadence_minutes": GRID * K,
                 "n_readings": N // K},
            ],
            household_batch_id=household_batch_of(HOUSEHOLD),
        )

    def test_a_delivered_register_is_never_spread(self):
        out = _impute(self._register_frame(), self._register_cadence())
        assert not any(_has(out, ImputeType.RESAMPLED_TO_GRID))

    def test_nothing_delivered_is_never_spread(self):
        df = _interval_only_frame()
        df[DIFF] = pd.array([pd.NA] * N, dtype="Float64")
        cadence = sidecar_frame(
            "cadence",
            [
                {"column": "ReadingDate", "cadence_minutes": GRID,
                 "n_readings": N},
                {"column": CUM, "cadence_minutes": GRID, "n_readings": 0},
                {"column": DIFF, "cadence_minutes": GRID, "n_readings": 0},
            ],
            household_batch_id=household_batch_of(HOUSEHOLD),
        )
        out = _impute(df, cadence)
        assert not any(_has(out, ImputeType.RESAMPLED_TO_GRID))

    def test_a_cadence_table_without_the_delivery_count_spreads_nothing(self):
        """Older tables carry no n_readings. The form is then unknown, and an
        unknown form is not a licence to rewrite the column."""
        df = _interval_only_frame()
        # The subject: every cadence column but n_readings, which is the one
        # the delivered form is read from.
        cadence = sidecar_frame(
            "cadence",
            [
                {"column": CUM, "cadence_minutes": GRID},
                {"column": DIFF, "cadence_minutes": GRID * K},
            ],
            household_batch_id=household_batch_of(HOUSEHOLD),
            partial=[
                name for name in sidecar_columns("cadence")
                if name != "n_readings"
            ],
        )
        out = _impute(df, cadence)
        assert not any(_has(out, ImputeType.RESAMPLED_TO_GRID))
