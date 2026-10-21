"""
Tests for resample_hh_data_duckdb and aggregate_project_data_duckdb.

Asserts what each function produces, on the calculated output of the canonical
chain (`sharded_calculated`): row count against the source at the passthrough
interval, the configured columns present, the cumulative counterpart rebuilt
from its Diff column, and the project average equal to the mean over
households.

Cross-FORMAT equivalence -- the same function over a monolith and over hive
shards -- is test_sharded_resample.py and test_sharded_project.py, not here.

Config isolation: uses _TEST_CONFIG to cover both active method patterns
(Diff/sum-resample/avg-aggregate and non-Diff/sum-resample/avg-aggregate)
so tests remain stable across data-model syncs.
"""

import pytest
import pandas as pd
from unittest.mock import patch

import etdtransform
from etdtransform.aggregate import (
    resample_hh_data_duckdb,
    aggregate_project_data_duckdb,
)

pytestmark = [pytest.mark.integration, pytest.mark.pipeline]

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



# ---------------------------------------------------------------------------
# Session fixtures: built on `sharded_calculated` from conftest.py -- the
# calculated output of the canonical chain -- rather than resampling again here.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def resample_duckdb_dir(tmp_path_factory, sharded_calculated):
    """Resample the canonical chain's calculated output into a temp directory."""
    out = tmp_path_factory.mktemp("resample_duckdb")
    with patch("etdmap.data_model.get_aggregation_config", return_value=_TEST_CONFIG):
        resample_hh_data_duckdb(
            source_path=str(sharded_calculated),
            output_dir=str(out),
            intervals=("5min", "15min", "60min"),
        )
    return out


@pytest.fixture(scope="session")
def aggregate_project_duckdb_dir(tmp_path_factory, resample_duckdb_dir):
    """
    Run aggregate_project_data_duckdb on the DuckDB resample output.
    Returns the output directory.
    """
    duckdb_dir = resample_duckdb_dir
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
    def test_resample_5min_passthrough_row_count(
        self, resample_duckdb_dir, sharded_calculated
    ):
        df_src = _load(str(sharded_calculated))
        df_5min = _load(str(resample_duckdb_dir / "household_5min.parquet"))
        # Row count must match the source (column selection only, no aggregation)
        assert len(df_5min) == len(df_src)

    def test_resample_5min_config_columns_present(self, resample_duckdb_dir):
        df = _load(str(resample_duckdb_dir / "household_5min.parquet"))
        for col in _TEST_CONFIG:
            assert col in df.columns, f"Column '{col}' missing from 5min output"

    def test_cumulative_columns_rebuilt(self, resample_duckdb_dir):
        """Diff columns must produce cumulative counterparts in non-5min outputs."""
        duckdb_dir = resample_duckdb_dir
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

    def test_project_average_is_the_mean_over_households(
        self, aggregate_project_duckdb_dir, resample_duckdb_dir
    ):
        """Project-level average must equal the mean over households for each timestamp."""
        out_dir = aggregate_project_duckdb_dir
        duckdb_dir = resample_duckdb_dir
        hh = _load(str(duckdb_dir / "household_5min.parquet"), SORT_COLS_HH)
        prj = _load(str(out_dir / "project_5min.parquet"), SORT_COLS_PRJ)

        col = "ElektriciteitNetgebruikHoogDiff"
        assert col in hh.columns, (
            f"{col} missing from household_5min.parquet -- the resample "
            f"fixture no longer produces it; fix the fixture or the test."
        )
        assert col in prj.columns, (
            f"{col} missing from project_5min.parquet -- project aggregation "
            f"dropped a column the household frame still has."
        )

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
