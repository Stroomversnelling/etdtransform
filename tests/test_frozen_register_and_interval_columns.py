"""
Imputation of frozen cumulative registers, interval columns with no register,
and the project averages imputation fills from.

Mapping-stage column handling is tested in etdmap:
tests/test_interval_and_register_columns.py
"""

import logging

import pandas as pd
import pytest

from etdtransform.impute import calculate_average_diff
from etdtransform.vectorized_impute import ImputeType, impute_and_normalize

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
    One household on a 5-min grid: the register rises by NORMAL per interval,
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
            pass  # register does not advance
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
    out, _, _ = impute_and_normalize(df.copy(), [CUM], "ProjectIdBSV", None)
    return out


def _span(s: pd.Series) -> pd.Series:
    """The frozen run plus the catch-up reading."""
    return s.iloc[FREEZE_FIRST : FREEZE_LAST + 2]


def _types(out: pd.DataFrame) -> set:
    return {int(v) for v in out[f"{DIFF}_impute_type"] if pd.notna(v)}


@pytest.mark.parametrize(
    "nulled",
    [
        pytest.param([FREEZE_LAST + 1], id="catchup_diff_only"),
        pytest.param(list(range(FREEZE_FIRST, FREEZE_LAST + 2)), id="frozen_and_catchup"),
    ],
)
def test_nulling_diff_with_register_intact_is_a_noop(nulled):
    """
    Imputation rebuilds a missing Diff from the register, so setting Diff values
    to NA while the register is intact leaves the series unchanged.

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


def test_nulling_register_over_frozen_run_conserves_volume():
    """
    With the stale register repeats set to NA, the freeze and the catch-up form
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
    the gap's starting register value. Nulling it too removes the anchor and the
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

    result = calculate_average_diff(df, "ProjectIdBSV", [DIFF])
    avg = result[DIFF]["avg_diff"]
    first = float(avg.loc[avg.ReadingDate == dates[0], f"{DIFF}_avg"].iloc[0])
    assert first == pytest.approx(3.0, abs=1e-9)

    bounds = result[DIFF]["household_max_with_bounds"]
    excluded = bounds.loc[
        bounds[f"{DIFF}_household_max"] >= bounds[f"{DIFF}_upper_bound"], "HuisIdBSV"
    ]
    assert 99 in set(excluded)


def test_project_average_rejects_physically_impossible_value():
    """
    The project average stays within the variable's threshold even when every
    household reports the same impossible reading, which a filter relative to
    the distribution of household maxima cannot detect.
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

    result = calculate_average_diff(df, "ProjectIdBSV", [DIFF])
    avg = result[DIFF]["avg_diff"]
    first = float(avg.loc[avg.ReadingDate == dates[0], f"{DIFF}_avg"].iloc[0])

    assert first <= MAX_PER_5MIN
