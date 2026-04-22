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

from etdtransform.aggregate import (
    reconstruct_cumulative_columns,
    reconstruct_cumulative_columns_ibis,
)

COLS = ["ElektriciteitNetgebruikLaag", "ElektriciteitTerugleveringHoog"]
ROWS = 20


# ---------------------------------------------------------------------------
# Reference implementation (original loop + concat)
# ---------------------------------------------------------------------------

def _reference_reconstruct(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Original loop-based implementation used as ground truth."""
    modified = []
    for _, hh in df.groupby("HuisIdBSV"):
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

def _make_df(huis_ids: list, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=ROWS, freq="5min").astype("datetime64[us]")
    rows = []
    for huis_id in huis_ids:
        diffs = rng.uniform(0.1, 5.0, (len(COLS), ROWS))
        cumulatives = [np.cumsum(d) + 100.0 for d in diffs]
        for i, dt in enumerate(dates):
            row = {
                "HuisIdBSV": huis_id,
                "ProjectIdBSV": 1,
                "ReadingDate": dt,
            }
            for j, col in enumerate(COLS):
                row[col] = cumulatives[j][i]
                row[col + "Diff"] = diffs[j][i]
            rows.append(row)
    df = pd.DataFrame(rows)
    df["HuisIdBSV"] = df["HuisIdBSV"].astype("Int64")
    df["ProjectIdBSV"] = df["ProjectIdBSV"].astype("Int64")
    return df.sort_values(["HuisIdBSV", "ReadingDate"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def base_df():
    return _make_df([1, 2, 3])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sort(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["HuisIdBSV", "ReadingDate"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Tests — pandas vectorised
# ---------------------------------------------------------------------------

def test_pandas_matches_reference_single_household():
    df = _make_df([1])
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


def test_pandas_cumsum_is_per_household(base_df):
    out = reconstruct_cumulative_columns(base_df.copy(), COLS)
    for col in COLS:
        for huis_id, grp in out.groupby("HuisIdBSV"):
            expected = grp[col + "Diff"].cumsum().values
            np.testing.assert_allclose(grp[col].values, expected, rtol=1e-9)


def test_pandas_check_column_is_diff_of_residual(base_df):
    out = reconstruct_cumulative_columns(base_df.copy(), COLS)
    for col in COLS:
        for _, grp in out.groupby("HuisIdBSV"):
            residual = grp[col] - grp[col + "Original"]
            expected = residual.diff().values
            np.testing.assert_allclose(
                grp[col + "Check"].values, expected, rtol=1e-9, equal_nan=True
            )


def test_pandas_single_row_per_household():
    """Edge case: one row per household — check should be NaN."""
    df = _make_df([10, 11]).groupby("HuisIdBSV").head(1).reset_index(drop=True)
    out = reconstruct_cumulative_columns(df.copy(), COLS)
    for col in COLS:
        assert out[col + "Check"].isna().all(), "Check should be all-NaN for 1-row groups"


# ---------------------------------------------------------------------------
# Tests — Ibis variant matches pandas
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


def test_ibis_cumsum_per_household(base_df):
    tbl = ibis.memtable(base_df)
    out = reconstruct_cumulative_columns_ibis(tbl, COLS).execute()
    out = _sort(out)
    for col in COLS:
        for _, grp in out.groupby("HuisIdBSV"):
            expected = grp[col + "Diff"].cumsum().values
            np.testing.assert_allclose(grp[col].values, expected, rtol=1e-9)


def test_ibis_returns_lazy_table(base_df):
    """reconstruct_cumulative_columns_ibis must return an ibis.Table, not a DataFrame."""
    tbl = ibis.memtable(base_df)
    result = reconstruct_cumulative_columns_ibis(tbl, COLS)
    assert isinstance(result, ibis.expr.types.Table)
