"""
Imputation of frozen cumulative variables, interval columns delivered without
cumulative values,
and the project averages imputation fills from.

Mapping-stage column handling is tested in etdmap:
tests/test_frozen_cumulative_and_interval_columns.py
"""

import logging

import pandas as pd
import pytest

from etdtransform.vectorized_impute import ImputeType, impute_and_normalize
from conftest import cadence_table, project_averages

pytestmark = pytest.mark.unit

CUM = "WatergebruikWarmTapwater"
DIFF = f"{CUM}Diff"
MAX_PER_5MIN = 100.0  # thresholds.csv bound for DIFF

N = 40
FREEZE_FIRST, FREEZE_LAST = 10, 29  # stale repeats; catch-up at FREEZE_LAST + 1
NORMAL = 3.0  # litres per 5-min interval
BASE = 1000.0


@pytest.fixture(autouse=True)
def _quiet_logs():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _series(catchup: float) -> pd.DataFrame:
    """
    One household on a 5-min grid: the cumulative variable rises by NORMAL per interval,
    holds flat across FREEZE_FIRST..FREEZE_LAST, then jumps by `catchup`.

    The flat run spans FREEZE_FIRST-1..FREEZE_LAST. FREEZE_FIRST-1 holds the
    last genuine value; FREEZE_FIRST..FREEZE_LAST repeat it.
    """
    values = []
    current = BASE - FREEZE_FIRST * NORMAL
    for i in range(N):
        if i < FREEZE_FIRST:
            current += NORMAL
        elif i <= FREEZE_LAST:
            pass  # cumulative variable does not advance
        elif i == FREEZE_LAST + 1:
            current += catchup
        else:
            current += NORMAL
        values.append(current)

    df = pd.DataFrame(
        {
            "HuisIdBSV": pd.array([1] * N, dtype="Int64"),
            "ProjectIdBSV": pd.array([1] * N, dtype="Int64"),
            "ReadingDate": pd.date_range("2026-01-01", periods=N, freq="5min"),
            CUM: pd.array(values, dtype="Float64"),
        }
    )
    df[DIFF] = df[CUM].diff().round(10)
    df.loc[0, DIFF] = 0
    # Flat average: keeps the assertions independent of profile shape.
    df[f"{DIFF}_avg"] = pd.array([NORMAL] * N, dtype="Float64")
    return df


def _impute(df: pd.DataFrame) -> pd.DataFrame:
    out, _, _ = impute_and_normalize(
        df.copy(), [CUM], "ProjectIdBSV", cadence=cadence_table(df, [CUM])
    )
    return out


def _span(s: pd.Series) -> pd.Series:
    """The frozen run plus the catch-up reading."""
    return s.iloc[FREEZE_FIRST : FREEZE_LAST + 2]


#: Bits another pass ORs onto whatever a fill route recorded, masked off so an
#: assertion on a route's own value tests one thing. The branch signal says
#: which logic resolved the cell, which every cell here is resolved by.
_NOT_THE_ROUTE = (
    ImputeType.DELIVERED_CUMULATIVE_IMPUTATION.value
    | ImputeType.DELIVERED_INTERVAL_IMPUTATION.value
)


def _types(out: pd.DataFrame) -> set:
    return {
        int(v) & ~_NOT_THE_ROUTE
        for v in out[f"{DIFF}_impute_type"]
        if pd.notna(v)
    }


@pytest.mark.parametrize(
    "nulled",
    [
        pytest.param([FREEZE_LAST + 1], id="catchup_diff_only"),
        pytest.param(list(range(FREEZE_FIRST, FREEZE_LAST + 2)), id="frozen_and_catchup"),
    ],
)
def test_nulling_diff_with_cumulative_intact_is_a_noop(nulled):
    """
    Imputation rebuilds a missing Diff from the cumulative variable, so setting
    Diff values to NA while that variable is intact leaves the series unchanged.

    The catch-up is 50 L, below MAX_PER_5MIN, so threshold enforcement cannot
    fire and the result is reconstruction alone.
    """
    original = _series(catchup=50.0)
    df = original.copy()
    df.loc[nulled, DIFF] = pd.NA

    out = _impute(df)

    pd.testing.assert_series_equal(
        out[DIFF].astype(float).round(8),
        original[DIFF].astype(float).round(8),
        check_names=False,
    )


def test_nulling_cumulative_over_frozen_run_conserves_volume():
    """
    With the stale cumulative repeats set to NA, the freeze and the catch-up form
    one gap whose jump is the real volume. That volume is redistributed across
    the gap, every interval stays within the threshold, and enforcement is not
    triggered.
    """
    catchup = 600.0
    df = _series(catchup=catchup)
    df.loc[FREEZE_FIRST:FREEZE_LAST, CUM] = pd.NA
    df.loc[FREEZE_FIRST : FREEZE_LAST + 1, DIFF] = pd.NA

    out = _impute(df)
    span = _span(out[DIFF])

    assert float(span.sum()) == pytest.approx(catchup, abs=1e-6)
    assert float(span.max()) <= MAX_PER_5MIN
    assert _types(out) == {ImputeType.SCALED_FILL.value}


def test_nulling_first_reading_of_flat_run_loses_the_anchor():
    """
    Redistribution depends on the reading before the stale repeats: it supplies
    the gap's starting cumulative value. Nulling it too removes the anchor and the
    redistributed total no longer equals the real volume.
    """
    catchup = 600.0
    df = _series(catchup=catchup)
    df.loc[FREEZE_FIRST - 1 : FREEZE_LAST, CUM] = pd.NA
    df.loc[FREEZE_FIRST - 1 : FREEZE_LAST + 1, DIFF] = pd.NA

    out = _impute(df)
    total = float(out[DIFF].iloc[FREEZE_FIRST - 1 : FREEZE_LAST + 2].sum())

    assert total != pytest.approx(catchup, abs=1e-6)


def test_catchup_volume_is_conserved_not_clamped():
    """
    A catch-up above the threshold is real consumption recorded late. Its volume
    must survive imputation rather than being replaced by the project average.
    """
    catchup = 600.0
    df = _series(catchup=catchup)
    df.loc[FREEZE_LAST + 1, DIFF] = pd.NA

    out = _impute(df)
    span = _span(out[DIFF])

    assert float(span.sum()) == pytest.approx(catchup, abs=1e-6)
    assert ImputeType.THRESHOLD_ADJUSTED.value not in _types(out)



def test_interval_only_column_is_imputed():
    """
    A Diff column with no cumulative counterpart still has its gaps filled; it
    is the only source for that variable.
    """
    n = 20
    gap = [8, 9, 10, 11]  # .loc slices are label-inclusive on both ends
    df = pd.DataFrame(
        {
            "HuisIdBSV": pd.array([1] * n, dtype="Int64"),
            "ProjectIdBSV": pd.array([1] * n, dtype="Int64"),
            "ReadingDate": pd.date_range("2026-01-01", periods=n, freq="5min"),
            CUM: pd.array([None] * n, dtype="Float64"),
            DIFF: pd.array([2.0] * n, dtype="Float64"),
        }
    )
    df.loc[gap, DIFF] = pd.NA
    df[f"{DIFF}_avg"] = pd.array([2.0] * n, dtype="Float64")

    out = _impute(df)

    assert int(out[DIFF].isna().sum()) == 0


def test_household_level_outlier_filter_excludes_a_spiking_household():
    """
    A household whose maximum Diff reaches twice the project's 95th percentile
    of household maxima is excluded from the project average.
    """
    n = 6
    dates = pd.date_range("2026-01-01", periods=n, freq="5min")
    rows = [
        {"HuisIdBSV": hh, "ProjectIdBSV": 1, "ReadingDate": d, DIFF: 3.0}
        for hh in range(1, 21)
        for d in dates
    ]
    rows += [{"HuisIdBSV": 99, "ProjectIdBSV": 1, "ReadingDate": d, DIFF: 3.0} for d in dates]
    df = pd.DataFrame(rows)
    df.loc[(df.HuisIdBSV == 99) & (df.ReadingDate == dates[0]), DIFF] = 5000.0
    df[DIFF] = df[DIFF].astype("Float64")

    result = project_averages(
        df, "ProjectIdBSV", [DIFF],
        cadence=cadence_table(df, [CUM]), grid_minutes=5,
    )
    avg = result[DIFF]["avg_diff"]
    first = float(avg.loc[avg.ReadingDate == dates[0], f"{DIFF}_avg"].iloc[0])
    assert first == pytest.approx(3.0, abs=1e-9)


def test_project_average_rejects_physically_impossible_value():
    """
    The project average stays within the variable's threshold even when every
    household reports the same impossible reading, which a filter relative to
    the distribution of household maxima cannot detect.

    An out-of-bounds value is discarded, not clipped: it is a wrong reading, not
    a large one, and clipping would keep a fabricated number in the pool. When
    every contributing value is discarded the average is pd.NA -- there is no
    average, which is not the same as an average of zero and is not above the
    threshold either.

    That case is synthetic. Measured on the May build, no (project, timestamp)
    slot has all of its values out of bounds; the fewest surviving households at
    an affected slot is 94 of ~99.
    """
    n = 6
    dates = pd.date_range("2026-01-01", periods=n, freq="5min")
    rows = [
        {"HuisIdBSV": hh, "ProjectIdBSV": 1, "ReadingDate": d, DIFF: 3.0}
        for hh in range(1, 21)
        for d in dates
    ]
    df = pd.DataFrame(rows)
    df.loc[df.ReadingDate == dates[0], DIFF] = 5000.0
    df[DIFF] = df[DIFF].astype("Float64")

    result = project_averages(
        df, "ProjectIdBSV", [DIFF],
        cadence=cadence_table(df, [CUM]), grid_minutes=5,
    )
    avg = result[DIFF]["avg_diff"]
    first = avg.loc[avg.ReadingDate == dates[0], f"{DIFF}_avg"].iloc[0]

    assert pd.isna(first) or first <= MAX_PER_5MIN
    assert result[DIFF]["n_discarded"] == 20, "one per household, all discarded"

    # The remaining timestamps are untouched: discarding is per value, so a
    # household keeps contributing the readings that are fine.
    rest = avg.loc[avg.ReadingDate != dates[0], f"{DIFF}_avg"]
    assert (rest == 3.0).all()


class TestWhichThresholdAppliesWhere:
    """There is no single threshold. Each task uses the one that matches what it
    is asking about, and the wrong choice fails in opposite directions:

    - the grid threshold on a delivered hourly reading rejects a correct value;
    - the column threshold on a filled value writes twelve intervals' worth of
      consumption into one five-minute slot.
    """

    HOURLY, GRID = 1, 2  # household ids: reports hourly, reports every 5 min

    def _frame(self):
        return pd.DataFrame({
            "HuisIdBSV": pd.array([self.HOURLY, self.HOURLY, self.GRID, self.GRID],
                                  dtype="Int64"),
            DIFF: pd.array([50.0] * 4, dtype="Float64"),
            f"{DIFF}_is_imputed": pd.array([False, True, False, True], dtype="boolean"),
        })

    def _cadence(self):
        return pd.DataFrame({
            "HuisIdBSV": pd.array([self.HOURLY, self.GRID], dtype="Int64"),
            "column": pd.array([CUM, CUM], dtype="string"),
            "cadence_minutes": pd.array([60, 5], dtype="Int64"),
        })

    def test_the_grid_threshold_is_the_declared_bound(self):
        from etdtransform.bounds import grid_bound
        assert grid_bound(DIFF) == MAX_PER_5MIN

    def test_the_column_threshold_scales_with_how_often_the_column_reports(self):
        from etdmap.data_model import scale_upper_bound
        from etdtransform.bounds import column_bound
        bound = column_bound(self._frame(), DIFF, self._cadence())
        # Hourly: twelve five-minute intervals arrive in one reading, so the
        # bound grows -- by the declared shape, not in proportion, because a
        # five-minute peak is not plausible sustained for an hour.
        assert float(bound.iloc[0]) == pytest.approx(
            scale_upper_bound(MAX_PER_5MIN, 12.0)
        )
        assert MAX_PER_5MIN < float(bound.iloc[0]) < MAX_PER_5MIN * 12
        # Already on the grid: the two thresholds coincide.
        assert float(bound.iloc[2]) == MAX_PER_5MIN

    def test_the_column_threshold_ignores_whether_a_row_was_filled(self):
        """Detection asks whether the cumulative variable could have delivered this, which does
        not depend on how the value got there."""
        from etdtransform.bounds import column_bound
        bound = column_bound(self._frame(), DIFF, self._cadence())
        assert float(bound.iloc[0]) == float(bound.iloc[1])

    def test_the_fill_bound_drops_to_the_grid_threshold_on_filled_rows(self):
        from etdtransform.bounds import fill_bound
        bound = fill_bound(
            self._frame(), DIFF, self._cadence(), is_imputed_col=f"{DIFF}_is_imputed"
        )
        from etdmap.data_model import scale_upper_bound
        assert float(bound.iloc[0]) == pytest.approx(
            scale_upper_bound(MAX_PER_5MIN, 12.0)
        )  # delivered
        assert float(bound.iloc[1]) == MAX_PER_5MIN        # filled
        assert float(bound.iloc[3]) == MAX_PER_5MIN

    def test_frozen_span_detection_uses_the_column_threshold(self):
        """A catch-up is implausible relative to what the cumulative variable delivers in one
        of ITS reporting intervals, not relative to a grid slot it never reports
        on. Judged at the grid threshold, every normal hourly reading looks like
        a catch-up."""
        from etdmap.data_model import scale_upper_bound
        from etdmap.flags import frozen_spans

        hourly = scale_upper_bound(MAX_PER_5MIN, 12.0)
        # An hourly cumulative variable: flat, then one reading holding an
        # hour's worth -- more than a grid slot allows, less than an hour does.
        plausible = (MAX_PER_5MIN + hourly) / 2
        intervals = pd.Series(pd.array(
            [3.0] + [0.0] * 8 + [plausible] + [3.0], dtype="Float64"
        ))
        assert len(frozen_spans(intervals, upper_bound=MAX_PER_5MIN)) == 1, (
            "the grid threshold calls it impossible"
        )
        assert len(frozen_spans(intervals, upper_bound=hourly)) == 0, (
            "at its own cadence an hour of consumption is plausible"
        )

    def test_the_column_threshold_is_still_a_threshold(self):
        """The bound grows with the reporting interval but does not vanish: a
        value beyond what an hour can hold is still a catch-up."""
        from etdmap.data_model import scale_upper_bound
        from etdmap.flags import frozen_spans

        hourly = scale_upper_bound(MAX_PER_5MIN, 12.0)
        intervals = pd.Series(pd.array(
            [3.0] + [0.0] * 8 + [hourly * 2] + [3.0], dtype="Float64"
        ))
        assert len(frozen_spans(intervals, upper_bound=hourly)) == 1

    def test_no_cadence_table_raises(self):
        """Judging every household at the grid bound is what nulls a coarsely
        reporting one, so it is refused rather than defaulted to."""
        from etdtransform.bounds import column_bound, fill_bound
        with pytest.raises(ValueError, match="no cadence table"):
            column_bound(self._frame(), DIFF, pd.DataFrame())
        with pytest.raises(ValueError, match="no cadence table"):
            fill_bound(self._frame(), DIFF, pd.DataFrame())
