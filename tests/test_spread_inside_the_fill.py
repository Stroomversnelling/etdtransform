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

from etdtransform.vectorized_impute import ImputeType, impute_and_normalize

pytestmark = pytest.mark.unit

GRID = 5
CUM = "ElektriciteitNetgebruik"
DIFF = f"{CUM}Diff"
MAX_PER_5MIN = 1.5  # thresholds.csv bound for DIFF
MAX_PER_HOUR = 1.5 * 12**0.5  # the same bound scaled to an hourly reading

K = 12  # an hourly column on a 5-minute grid
N = 48


@pytest.fixture(autouse=True)
def _quiet_logs():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _cadence(rows):
    return pd.DataFrame({
        "HuisIdBSV": pd.array([r[0] for r in rows], dtype="Int64"),
        "HuisBatchIdBSV": pd.array([value + 1000 for value in [r[0] for r in rows]], dtype="Int64"),
        "column": pd.array([r[1] for r in rows], dtype="string"),
        "cadence_minutes": pd.array([r[2] for r in rows], dtype="Int64"),
        "n_readings": pd.array([r[3] for r in rows], dtype="Int64"),
    })


def _interval_only_frame(reading=1.2, missing=(), n=N):
    """Hourly interval readings, no register, one project average per row."""
    values = [pd.NA] * n
    for row in range(K - 1, n, K):
        values[row] = reading
    for row in missing:
        values[row] = pd.NA

    df = pd.DataFrame({
        "HuisIdBSV": pd.array([1] * n, dtype="Int64"),
        "HuisBatchIdBSV": pd.array([value + 1000 for value in [1] * n], dtype="Int64"),
        "ProjectIdBSV": pd.array([1] * n, dtype="Int64"),
        "ReadingDate": pd.date_range("2026-01-01", periods=n, freq="5min"),
        CUM: pd.array([pd.NA] * n, dtype="Float64"),
        DIFF: pd.array(values, dtype="Float64"),
    })
    df[f"{DIFF}_avg"] = pd.array([reading / K] * n, dtype="Float64")
    return df


def _interval_only_cadence(household=1):
    return _cadence([
        (household, "ReadingDate", GRID, n)
        for n in [N]
    ] + [
        (household, CUM, GRID, 0),
        (household, DIFF, GRID * K, N // K),
    ])


def _impute(df, cadence):
    out, _, _ = impute_and_normalize(
        df.copy(), [CUM], "ProjectIdBSV", cadence=cadence
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
        """A register delivered hourly; its interval column derived from it."""
        rise = 1.2
        values, total = [], 100.0
        for row in range(n):
            if row % K == K - 1:
                total += rise
            values.append(total if row % K == K - 1 else pd.NA)
        df = pd.DataFrame({
            "HuisIdBSV": pd.array([1] * n, dtype="Int64"),
            "HuisBatchIdBSV": pd.array([value + 1000 for value in [1] * n], dtype="Int64"),
            "ProjectIdBSV": pd.array([1] * n, dtype="Int64"),
            "ReadingDate": pd.date_range("2026-01-01", periods=n, freq="5min"),
            CUM: pd.array(values, dtype="Float64"),
        })
        df[DIFF] = df[CUM].diff().round(10)
        df.loc[0, DIFF] = 0
        df[f"{DIFF}_avg"] = pd.array([rise / K] * n, dtype="Float64")
        return df

    def _register_cadence(self):
        return _cadence([
            (1, "ReadingDate", GRID, N),
            (1, CUM, GRID * K, N // K),
            (1, DIFF, GRID * K, N // K),
        ])

    def test_a_delivered_register_is_never_spread(self):
        out = _impute(self._register_frame(), self._register_cadence())
        assert not any(_has(out, ImputeType.RESAMPLED_TO_GRID))

    def test_nothing_delivered_is_never_spread(self):
        df = _interval_only_frame()
        df[DIFF] = pd.array([pd.NA] * N, dtype="Float64")
        cadence = _cadence([
            (1, "ReadingDate", GRID, N),
            (1, CUM, GRID, 0),
            (1, DIFF, GRID, 0),
        ])
        out = _impute(df, cadence)
        assert not any(_has(out, ImputeType.RESAMPLED_TO_GRID))

    def test_a_cadence_table_without_the_delivery_count_spreads_nothing(self):
        """Older tables carry no n_readings. The form is then unknown, and an
        unknown form is not a licence to rewrite the column."""
        df = _interval_only_frame()
        cadence = pd.DataFrame({
            "HuisIdBSV": pd.array([1, 1], dtype="Int64"),
            "HuisBatchIdBSV": pd.array([1001, 1001], dtype="Int64"),
            "column": pd.array([CUM, DIFF], dtype="string"),
            "cadence_minutes": pd.array([GRID, GRID * K], dtype="Int64"),
        })
        out = _impute(df, cadence)
        assert not any(_has(out, ImputeType.RESAMPLED_TO_GRID))
