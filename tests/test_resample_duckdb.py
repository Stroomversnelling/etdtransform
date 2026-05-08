"""
Tests for resample_hh_data_duckdb and aggregate_project_data_duckdb.

Compares DuckDB outputs against the pandas path (resample_hh_data with df=)
on the shared ibis_pipeline fixture from test_pipeline_equivalence.py.

Config isolation: uses _TEST_CONFIG to cover both active method patterns
(Diff/sum-resample/avg-aggregate and non-Diff/sum-resample/avg-aggregate)
so tests remain stable across Grist syncs.
"""

import pytest
import pandas as pd
from unittest.mock import patch

import etdtransform
from etdtransform.aggregate import (
    resample_hh_data,
    resample_hh_data_duckdb,
    aggregate_project_data_duckdb,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SORT_COLS_HH = ("HuisIdBSV", "ProjectIdBSV", "ReadingDate")
SORT_COLS_PRJ = ("ProjectIdBSV", "ReadingDate")
REL_TOL = 1e-5

# Covers both active method patterns.
# Diff column: resample sum -> cumsum -> cumulative counterpart rebuilt.
# Non-Diff total: resample sum, no cumsum.
_TEST_CONFIG = {
    "ElektriciteitNetgebruikHoogDiff": {"resample_method": "sum", "aggregate_method": "avg"},
    "ZonopwekBruto": {"resample_method": "sum", "aggregate_method": "avg"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(path, sort_cols=None):
    df = pd.read_parquet(path, dtype_backend="numpy_nullable")
    if sort_cols:
        present = [c for c in sort_cols if c in df.columns]
        if present:
            df = df.sort_values(present).reset_index(drop=True)
    return df


def _shared_cols(df_a, df_b):
    return sorted(set(df_a.columns) & set(df_b.columns))


def _assert_frames_equal(df_a, df_b, label_a, label_b):
    shared = _shared_cols(df_a, df_b)
    pd.testing.assert_frame_equal(
        df_a[shared].reset_index(drop=True),
        df_b[shared].reset_index(drop=True),
        check_like=False,
        rtol=REL_TOL,
        obj=f"{label_a} vs {label_b}",
    )


# ---------------------------------------------------------------------------
# Session fixtures: reuse the shared `ibis_pipeline` fixture from conftest.py
# (its household_calculated.parquet is identical to what this file used to
# produce in its own _calculated_dir fixture -- ~128s of duplicated work
# eliminated).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def resample_duckdb_dirs(tmp_path_factory, ibis_pipeline):
    """
    Run DuckDB and pandas resample on the shared ibis_pipeline's
    household_calculated.parquet, each into its own temp directory.
    Returns (duckdb_dir, pandas_dir).
    """
    calculated_path = str(ibis_pipeline / "household_calculated.parquet")

    # -- DuckDB path (patched to use _TEST_CONFIG) --
    duckdb_dir = tmp_path_factory.mktemp("resample_duckdb")
    with patch("etdmap.data_model.get_aggregation_config", return_value=_TEST_CONFIG):
        resample_hh_data_duckdb(
            source_path=calculated_path,
            output_dir=str(duckdb_dir),
            intervals=("5min", "15min", "60min"),
        )

    # -- Pandas path via resample_hh_data(df=...) --
    pandas_dir = tmp_path_factory.mktemp("resample_pandas")
    old = etdtransform.options.aggregate_folder_path
    etdtransform.options.aggregate_folder_path = pandas_dir
    try:
        df_calc = pd.read_parquet(calculated_path, dtype_backend="numpy_nullable")
        resample_hh_data(df=df_calc.copy(), intervals=("5min", "15min", "60min"))
    finally:
        etdtransform.options.aggregate_folder_path = old

    return duckdb_dir, pandas_dir


@pytest.fixture(scope="session")
def aggregate_project_duckdb_dir(tmp_path_factory, resample_duckdb_dirs):
    """
    Run aggregate_project_data_duckdb on the DuckDB resample output.
    Returns the output directory.
    """
    duckdb_dir, _ = resample_duckdb_dirs
    out_dir = tmp_path_factory.mktemp("agg_project_duckdb")

    # Copy resample outputs to out_dir so aggregate reads them.
    import shutil
    for interval in ("5min", "15min", "60min"):
        shutil.copy(
            str(duckdb_dir / f"household_{interval}.parquet"),
            str(out_dir / f"household_{interval}.parquet"),
        )

    old = etdtransform.options.aggregate_folder_path
    etdtransform.options.aggregate_folder_path = out_dir
    try:
        with patch("etdmap.data_model.get_aggregation_config", return_value=_TEST_CONFIG):
            aggregate_project_data_duckdb(intervals=("5min", "15min", "60min"))
    finally:
        etdtransform.options.aggregate_folder_path = old

    return out_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestResampleDuckdb:
    def test_resample_5min_passthrough_row_count(self, resample_duckdb_dirs, ibis_pipeline):
        duckdb_dir, _ = resample_duckdb_dirs
        df_src = _load(str(ibis_pipeline / "household_calculated.parquet"))
        df_5min = _load(str(duckdb_dir / "household_5min.parquet"))
        # Row count must match the source (column selection only, no aggregation)
        assert len(df_5min) == len(df_src)

    def test_resample_5min_config_columns_present(self, resample_duckdb_dirs):
        duckdb_dir, _ = resample_duckdb_dirs
        df = _load(str(duckdb_dir / "household_5min.parquet"))
        for col in _TEST_CONFIG:
            assert col in df.columns, f"Column '{col}' missing from 5min output"

    def test_resample_15min_equivalence(self, resample_duckdb_dirs):
        duckdb_dir, pandas_dir = resample_duckdb_dirs
        df_duckdb = _load(str(duckdb_dir / "household_15min.parquet"), SORT_COLS_HH)
        df_pandas = _load(str(pandas_dir / "household_15min.parquet"), SORT_COLS_HH)
        # Only compare columns in _TEST_CONFIG that exist in both outputs
        cols = [c for c in _TEST_CONFIG if c in df_duckdb.columns and c in df_pandas.columns]
        assert cols, "No config columns found in both outputs"
        _assert_frames_equal(df_duckdb[cols], df_pandas[cols], "duckdb", "pandas")

    def test_resample_60min_equivalence(self, resample_duckdb_dirs):
        duckdb_dir, pandas_dir = resample_duckdb_dirs
        df_duckdb = _load(str(duckdb_dir / "household_60min.parquet"), SORT_COLS_HH)
        df_pandas = _load(str(pandas_dir / "household_60min.parquet"), SORT_COLS_HH)
        cols = [c for c in _TEST_CONFIG if c in df_duckdb.columns and c in df_pandas.columns]
        assert cols
        _assert_frames_equal(df_duckdb[cols], df_pandas[cols], "duckdb", "pandas")

    def test_cumulative_columns_rebuilt(self, resample_duckdb_dirs):
        """Diff columns must produce cumulative counterparts in non-5min outputs."""
        duckdb_dir, _ = resample_duckdb_dirs
        diff_col = "ElektriciteitNetgebruikHoogDiff"
        base_col = "ElektriciteitNetgebruikHoog"
        for interval in ("15min", "60min"):
            df = _load(str(duckdb_dir / f"household_{interval}.parquet"),
                       sort_cols=["HuisIdBSV", "ProjectIdBSV", "ReadingDate"])
            assert base_col in df.columns, f"{base_col} missing from {interval} output"
            # DuckDB uses SUM() OVER (UNBOUNDED PRECEDING) which skips NULLs.
            # pandas .expanding().sum() is the equivalent (also skips NAs).
            expected = (
                df.groupby(["HuisIdBSV", "ProjectIdBSV"], group_keys=False)[diff_col]
                .transform(lambda x: x.expanding().sum())
            )
            actual = df[base_col]
            pd.testing.assert_series_equal(
                actual.reset_index(drop=True).astype(float),
                expected.reset_index(drop=True).astype(float),
                rtol=REL_TOL,
                check_names=False,
                obj=f"cumsum {base_col} at {interval}",
            )


class TestAggregateProjectDuckdb:
    def test_project_files_created(self, aggregate_project_duckdb_dir):
        out_dir = aggregate_project_duckdb_dir
        for interval in ("5min", "15min", "60min"):
            assert (out_dir / f"project_{interval}.parquet").exists(), (
                f"project_{interval}.parquet not created"
            )

    def test_project_aggregation_5min_equivalence(self, aggregate_project_duckdb_dir, resample_duckdb_dirs):
        """Project-level average must equal the mean over households for each timestamp."""
        out_dir = aggregate_project_duckdb_dir
        duckdb_dir, _ = resample_duckdb_dirs
        hh = _load(str(duckdb_dir / "household_5min.parquet"), SORT_COLS_HH)
        prj = _load(str(out_dir / "project_5min.parquet"), SORT_COLS_PRJ)

        col = "ElektriciteitNetgebruikHoogDiff"
        if col not in hh.columns or col not in prj.columns:
            pytest.skip(f"{col} not in test data")

        manual_avg = (
            hh.groupby(["ProjectIdBSV", "ReadingDate"])[col]
            .mean()
            .reset_index()
            .sort_values(list(SORT_COLS_PRJ))
            .reset_index(drop=True)
        )
        prj_col = prj.sort_values(list(SORT_COLS_PRJ)).reset_index(drop=True)[col]

        pd.testing.assert_series_equal(
            prj_col.astype(float),
            manual_avg[col].astype(float),
            rtol=REL_TOL,
            obj="project AVG vs manual mean",
        )
