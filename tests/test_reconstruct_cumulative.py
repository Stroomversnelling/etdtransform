"""
Regression tests for reconstruct_cumulative_columns (pandas) and
reconstruct_cumulative_columns_ibis (Ibis/DuckDB).

Both must produce identical output to the original per-household loop + pd.concat
implementation that they replaced.
"""
import numpy as np
import pandas as pd
import pytest
import ibis

from etdmap import ids
from etdtransform.aggregate import (
    reconstruct_cumulative_columns,
    reconstruct_cumulative_columns_ibis,
)

pytestmark = pytest.mark.unit

COLS = ["ElektriciteitNetgebruikLaag", "ElektriciteitTerugleveringHoog"]
ROWS = 20


# ---------------------------------------------------------------------------
# Reference implementation (original loop + concat)
# ---------------------------------------------------------------------------

def _reference_reconstruct(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Original loop-based implementation used as ground truth.

    Per dataset, not per household: a cumulative total runs over one time
    series, and a household can have several.
    """
    modified = []
    for _, hh in df.groupby(ids.HOUSEHOLD_BATCH):
        for col in cols:
            hh = hh.copy()
            hh[col + "Original"] = hh[col]
            hh[col] = hh[col + "Diff"].cumsum()
            hh[col + "Check"] = (hh[col] - hh[col + "Original"]).diff()
        modified.append(hh)
    return pd.concat(modified, ignore_index=True)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def _make_df(datasets: list, seed: int = 42) -> pd.DataFrame:
    """Frames for *datasets*, each ``(HuisIdBSV, HuisBatchIdBSV)``.

    Not `conftest.dataset_frame`: that builds one household's rows and fills
    the cumulative counterpart with arbitrary values, while the relation
    ``cumulative == cumsum(interval) + 100`` is what reconstruction is checked
    against here, and these frames carry several datasets at once.

    The draw is numpy's because pandas has no generator; every column it feeds
    is stored as nullable Float64 (ADR-005).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(
        "2023-01-01", periods=ROWS, freq="5min"
    ).astype("datetime64[us]")
    rows = []
    for household_id, household_batch_id in datasets:
        diffs = rng.uniform(0.1, 5.0, (len(COLS), ROWS))
        cumulatives = [pd.Series(d, dtype="Float64").cumsum() + 100.0 for d in diffs]
        for i, dt in enumerate(dates):
            row = {
                ids.HOUSEHOLD: household_id,
                ids.HOUSEHOLD_BATCH: household_batch_id,
                ids.PROJECT: 1,
                "ReadingDate": dt,
            }
            for j, col in enumerate(COLS):
                row[col] = cumulatives[j][i]
                row[col + "Diff"] = diffs[j][i]
            rows.append(row)
    df = pd.DataFrame(rows)
    for id_column in (ids.HOUSEHOLD, ids.HOUSEHOLD_BATCH, ids.PROJECT):
        df[id_column] = df[id_column].astype("Int64")
    for col in COLS:
        df[col] = df[col].astype("Float64")
        df[col + "Diff"] = df[col + "Diff"].astype("Float64")
    return df.sort_values(
        [ids.HOUSEHOLD_BATCH, "ReadingDate"]
    ).reset_index(drop=True)


#: Three households, one dataset each, plus a SECOND dataset of household 1
#: over the same timestamps. Grouping the rebuild by household would run one
#: cumulative total across both of household 1's deliveries.
DATASETS = [(1, 1001), (2, 1002), (3, 1003), (1, 2001)]


@pytest.fixture(scope="module")
def base_df():
    return _make_df(DATASETS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sort(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values([ids.HOUSEHOLD_BATCH, "ReadingDate"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Tests -- pandas vectorised
# ---------------------------------------------------------------------------

def test_pandas_matches_reference_single_household():
    df = _make_df([(1, 1001)])
    ref = _reference_reconstruct(df.copy(), COLS)
    out = reconstruct_cumulative_columns(df.copy(), COLS)
    pd.testing.assert_frame_equal(_sort(out), _sort(ref), check_like=True, rtol=1e-9)


def test_pandas_matches_reference_multiple_households(base_df):
    ref = _reference_reconstruct(base_df.copy(), COLS)
    out = reconstruct_cumulative_columns(base_df.copy(), COLS)
    pd.testing.assert_frame_equal(_sort(out), _sort(ref), check_like=True, rtol=1e-9)


def test_pandas_original_column_preserved(base_df):
    out = reconstruct_cumulative_columns(base_df.copy(), COLS)
    for col in COLS:
        pd.testing.assert_series_equal(
            out[col + "Original"].reset_index(drop=True),
            base_df[col].reset_index(drop=True),
            check_names=False,
        )


def test_pandas_cumsum_is_per_dataset(base_df):
    out = reconstruct_cumulative_columns(base_df.copy(), COLS)
    for col in COLS:
        for _dataset, grp in out.groupby(ids.HOUSEHOLD_BATCH):
            expected = grp[col + "Diff"].cumsum().values
            np.testing.assert_allclose(grp[col].values, expected, rtol=1e-9)


def test_pandas_check_column_is_diff_of_residual(base_df):
    out = reconstruct_cumulative_columns(base_df.copy(), COLS)
    for col in COLS:
        for _, grp in out.groupby(ids.HOUSEHOLD_BATCH):
            residual = grp[col] - grp[col + "Original"]
            expected = residual.diff().values
            np.testing.assert_allclose(
                grp[col + "Check"].values, expected, rtol=1e-9, equal_nan=True
            )


def test_pandas_single_row_per_dataset():
    """Edge case: one row per dataset -- check should be NaN."""
    df = (
        _make_df([(10, 1010), (11, 1011)])
        .groupby(ids.HOUSEHOLD_BATCH).head(1).reset_index(drop=True)
    )
    out = reconstruct_cumulative_columns(df.copy(), COLS)
    for col in COLS:
        assert out[col + "Check"].isna().all(), "Check should be all-NaN for 1-row groups"


# ---------------------------------------------------------------------------
# Tests -- Ibis variant matches pandas
# ---------------------------------------------------------------------------

def test_ibis_matches_pandas_multiple_households(base_df):
    tbl = ibis.memtable(base_df)
    ibis_result = (
        reconstruct_cumulative_columns_ibis(tbl, COLS)
        .execute()
    )
    pandas_result = reconstruct_cumulative_columns(base_df.copy(), COLS)

    pd.testing.assert_frame_equal(
        _sort(ibis_result),
        _sort(pandas_result),
        check_like=True,
        rtol=1e-9,
        check_dtype=False,
    )


def test_ibis_cumsum_per_dataset(base_df):
    tbl = ibis.memtable(base_df)
    out = reconstruct_cumulative_columns_ibis(tbl, COLS).execute()
    out = _sort(out)
    for col in COLS:
        for _, grp in out.groupby(ids.HOUSEHOLD_BATCH):
            expected = grp[col + "Diff"].cumsum().values
            np.testing.assert_allclose(grp[col].values, expected, rtol=1e-9)


def test_ibis_returns_lazy_table(base_df):
    """reconstruct_cumulative_columns_ibis must return an ibis.Table, not a DataFrame."""
    tbl = ibis.memtable(base_df)
    result = reconstruct_cumulative_columns_ibis(tbl, COLS)
    assert isinstance(result, ibis.expr.types.Table)
