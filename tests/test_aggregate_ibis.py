"""
Tests for aggregate_hh_data_5min_ibis and aggregate_hh_data_duckdb.

All tests are self-contained: synthetic household parquet files are written to
tmp_path, etdtransform.options paths are redirected there, and update_meenemen
is patched to return a controlled index DataFrame.  No real data is read.
"""
import os

import ibis
import numpy as np
import pandas as pd
import pytest

import etdtransform
from etdtransform.aggregate import (
    aggregate_hh_data_5min,
    aggregate_hh_data_5min_ibis,
    aggregate_hh_data_duckdb,
    impute_hh_data_5min_chunked,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ROWS_PER_HH = 20
DIFF_COL = "ElektriciteitNetgebruikLaagDiff"


def _make_index(huis_ids: list, project_ids: list, meenemen: list | None = None) -> pd.DataFrame:
    """Return a minimal index DataFrame matching what update_meenemen returns."""
    if meenemen is None:
        meenemen = [True] * len(huis_ids)
    return pd.DataFrame({
        "HuisIdBSV":    pd.array(huis_ids, dtype="Int64"),
        "ProjectIdBSV": pd.array(project_ids, dtype="Int64"),
        "Meenemen":     meenemen,
    })


def _write_hh_files(mapped_dir: str, huis_ids: list) -> None:
    """Write one synthetic parquet per household into mapped_dir."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=ROWS_PER_HH, freq="5min").astype("datetime64[us]")
    for huis_id in huis_ids:
        df = pd.DataFrame({
            "ReadingDate": dates,
            DIFF_COL: rng.uniform(0.5, 10.0, ROWS_PER_HH),
        })
        df.to_parquet(os.path.join(mapped_dir, f"household_{huis_id}_table.parquet"), engine="pyarrow")


@pytest.fixture()
def dirs(tmp_path):
    """Create mapped/ and aggregate/ subdirs, redirect etdtransform.options."""
    mapped = tmp_path / "mapped"
    agg    = tmp_path / "aggregate"
    mapped.mkdir()
    agg.mkdir()

    old_mapped = etdtransform.options.mapped_folder_path
    old_agg    = etdtransform.options.aggregate_folder_path
    etdtransform.options.mapped_folder_path    = str(mapped)
    etdtransform.options.aggregate_folder_path = str(agg)

    yield {"mapped": str(mapped), "aggregate": str(agg)}

    etdtransform.options.mapped_folder_path    = old_mapped
    etdtransform.options.aggregate_folder_path = old_agg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basic_output_correctness(dirs, monkeypatch):
    """All households appear in the output with correct row counts and columns."""
    huis_ids    = list(range(1, 8))   # 7 households
    project_ids = [1, 1, 1, 2, 2, 2, 2]
    index_df = _make_index(huis_ids, project_ids)
    _write_hh_files(dirs["mapped"], huis_ids)

    monkeypatch.setattr("etdtransform.aggregate.read_index", lambda: (index_df, None))

    aggregate_hh_data_5min_ibis(batch_size=4)

    out = pd.read_parquet(os.path.join(dirs["aggregate"], "household_default.parquet"))

    assert len(out) == len(huis_ids) * ROWS_PER_HH
    assert set(out["HuisIdBSV"].unique()) == set(huis_ids)
    assert set(out["ProjectIdBSV"].unique()) == {1, 2}
    assert "ReadingDate" in out.columns
    assert DIFF_COL in out.columns


def test_batching_produces_same_result_as_single_batch(dirs, monkeypatch):
    """Output is identical whether batching splits the work or not."""
    huis_ids    = list(range(1, 13))  # 12 households
    project_ids = [1] * 6 + [2] * 6
    index_df = _make_index(huis_ids, project_ids)
    _write_hh_files(dirs["mapped"], huis_ids)

    monkeypatch.setattr("etdtransform.aggregate.read_index", lambda: (index_df, None))

    out_path = os.path.join(dirs["aggregate"], "household_default.parquet")

    aggregate_hh_data_5min_ibis(batch_size=3)   # forces 4 batches of 3
    out_small = pd.read_parquet(out_path).sort_values(["HuisIdBSV", "ReadingDate"]).reset_index(drop=True)

    aggregate_hh_data_5min_ibis(batch_size=100)  # all in one batch
    out_big   = pd.read_parquet(out_path).sort_values(["HuisIdBSV", "ReadingDate"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(out_small, out_big, check_like=True)


def test_batch_size_larger_than_n_files(dirs, monkeypatch):
    """batch_size > number of files should not cause errors."""
    huis_ids    = [10, 11]
    project_ids = [1, 1]
    index_df = _make_index(huis_ids, project_ids)
    _write_hh_files(dirs["mapped"], huis_ids)

    monkeypatch.setattr("etdtransform.aggregate.read_index", lambda: (index_df, None))

    aggregate_hh_data_5min_ibis(batch_size=500)

    out = pd.read_parquet(os.path.join(dirs["aggregate"], "household_default.parquet"))
    assert len(out) == 2 * ROWS_PER_HH


def test_stratified_sampling(dirs, monkeypatch):
    """sample_ratio keeps representation from each project."""
    huis_ids    = list(range(1, 21))           # 20 households
    project_ids = [1] * 10 + [2] * 10
    index_df = _make_index(huis_ids, project_ids)
    _write_hh_files(dirs["mapped"], huis_ids)

    monkeypatch.setattr("etdtransform.aggregate.read_index", lambda: (index_df, None))

    aggregate_hh_data_5min_ibis(sample_ratio=0.5, batch_size=4)

    out = pd.read_parquet(os.path.join(dirs["aggregate"], "household_default.parquet"))
    sampled_huis = out["HuisIdBSV"].unique()

    # 50% of 20 = 10 households total; both projects must be represented
    assert len(sampled_huis) == 10
    p1 = out[out["ProjectIdBSV"] == 1]["HuisIdBSV"].nunique()
    p2 = out[out["ProjectIdBSV"] == 2]["HuisIdBSV"].nunique()
    assert p1 > 0, "Project 1 not represented in sample"
    assert p2 > 0, "Project 2 not represented in sample"
    assert p1 + p2 == 10


def test_column_projection(dirs, monkeypatch):
    """When columns= is given, only id cols + requested cols appear in output."""
    huis_ids    = [1, 2, 3]
    project_ids = [1, 1, 2]
    index_df = _make_index(huis_ids, project_ids)

    # Write files with an extra column that should be dropped
    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", periods=ROWS_PER_HH, freq="5min").astype("datetime64[us]")
    for huis_id in huis_ids:
        df = pd.DataFrame({
            "ReadingDate":          dates,
            DIFF_COL:               rng.uniform(0.5, 10.0, ROWS_PER_HH),
            "Zon-opwekTotaalDiff":  rng.uniform(0.1, 5.0, ROWS_PER_HH),
        })
        df.to_parquet(
            os.path.join(dirs["mapped"], f"household_{huis_id}_table.parquet"),
            engine="pyarrow",
        )

    monkeypatch.setattr("etdtransform.aggregate.read_index", lambda: (index_df, None))

    aggregate_hh_data_5min_ibis(columns=[DIFF_COL], batch_size=2)

    out = pd.read_parquet(os.path.join(dirs["aggregate"], "household_default.parquet"))

    assert DIFF_COL in out.columns
    assert "Zon-opwekTotaalDiff" not in out.columns, "Extra column should have been projected away"
    assert "HuisIdBSV" in out.columns
    assert "ProjectIdBSV" in out.columns
    assert "ReadingDate" in out.columns


def test_missing_household_files_skipped(dirs, monkeypatch):
    """Households listed in the index but with no parquet file are silently skipped."""
    huis_ids    = [1, 2, 3, 4]
    project_ids = [1, 1, 2, 2]
    index_df = _make_index(huis_ids, project_ids)
    # Only write files for households 1 and 3; 2 and 4 are missing
    _write_hh_files(dirs["mapped"], [1, 3])

    monkeypatch.setattr("etdtransform.aggregate.read_index", lambda: (index_df, None))

    aggregate_hh_data_5min_ibis(batch_size=10)

    out = pd.read_parquet(os.path.join(dirs["aggregate"], "household_default.parquet"))
    assert set(out["HuisIdBSV"].unique()) == {1, 3}
    assert len(out) == 2 * ROWS_PER_HH


def test_all_files_missing_raises(dirs, monkeypatch):
    """ValueError is raised when no parquet files exist for any listed household."""
    index_df = _make_index([99, 100], [1, 1])
    # write nothing to mapped dir

    monkeypatch.setattr("etdtransform.aggregate.read_index", lambda: (index_df, None))

    with pytest.raises(ValueError, match="No household files found"):
        aggregate_hh_data_5min_ibis()


def test_meenemen_false_excluded(dirs, monkeypatch):
    """Households with Meenemen=False are excluded even if their files exist."""
    huis_ids    = [1, 2, 3]
    project_ids = [1, 1, 1]
    index_df = _make_index(huis_ids, project_ids, meenemen=[True, False, True])
    _write_hh_files(dirs["mapped"], huis_ids)

    monkeypatch.setattr("etdtransform.aggregate.read_index", lambda: (index_df, None))

    aggregate_hh_data_5min_ibis(batch_size=10)

    out = pd.read_parquet(os.path.join(dirs["aggregate"], "household_default.parquet"))
    assert set(out["HuisIdBSV"].unique()) == {1, 3}
    assert 2 not in out["HuisIdBSV"].values


def test_temp_dir_cleaned_up(dirs, monkeypatch, tmp_path):
    """No temp batch parquets should remain after the function returns."""
    huis_ids    = list(range(1, 6))
    project_ids = [1] * 5
    index_df = _make_index(huis_ids, project_ids)
    _write_hh_files(dirs["mapped"], huis_ids)

    monkeypatch.setattr("etdtransform.aggregate.read_index", lambda: (index_df, None))

    # Count parquet files in the temp root before and after
    before = set(tmp_path.rglob("batch_*.parquet"))
    aggregate_hh_data_5min_ibis(batch_size=2)
    after = set(tmp_path.rglob("batch_*.parquet"))

    assert after == before, f"Temp batch parquets were not cleaned up: {after - before}"


# ===========================================================================
# aggregate_hh_data_duckdb tests
# ===========================================================================

DATES = pd.date_range("2023-01-01", periods=ROWS_PER_HH, freq="5min", tz="UTC")

COL_SHARED = "ElektriciteitNetgebruikLaagCum"
COL_A_ONLY = "supplier_a_extra"
COL_B_ONLY = "SunCollectorHeatingWaterVolumeLCum"


def _write_mixed_schema_files(mapped_dir: str) -> None:
    """
    Write files with deliberately different column sets to simulate mixed-supplier data.

    Supplier A (huis 1-3): ReadingDate, COL_SHARED, COL_A_ONLY
    Supplier B (huis 4-6): ReadingDate, COL_SHARED, COL_B_ONLY
    """
    rng = np.random.default_rng(7)
    for huis_id in range(1, 7):
        extra_col = COL_A_ONLY if huis_id <= 3 else COL_B_ONLY
        df = pd.DataFrame({
            "ReadingDate": DATES,
            COL_SHARED:   pd.array(rng.uniform(0, 100, ROWS_PER_HH).tolist(), dtype="Float64"),
            extra_col:    pd.array(rng.uniform(0, 10, ROWS_PER_HH).tolist(), dtype="Float64"),
        })
        df.to_parquet(
            os.path.join(mapped_dir, f"household_{huis_id}_table.parquet"),
            engine="pyarrow",
        )


class TestAggregateDuckdb:

    def test_basic_output_correctness(self, dirs, monkeypatch):
        """All households appear in the output with correct row counts."""
        huis_ids    = list(range(1, 6))
        project_ids = [1, 1, 2, 2, 2]
        index_df = _make_index(huis_ids, project_ids)
        _write_hh_files(dirs["mapped"], huis_ids)
        monkeypatch.setattr("etdtransform.aggregate.read_index", lambda: (index_df, None))

        aggregate_hh_data_duckdb()

        out = pd.read_parquet(
            os.path.join(dirs["aggregate"], "household_default.parquet"),
            dtype_backend="numpy_nullable",
        )
        assert len(out) == len(huis_ids) * ROWS_PER_HH
        assert set(out["HuisIdBSV"].unique()) == set(huis_ids)
        assert set(out["ProjectIdBSV"].unique()) == {1, 2}

    def test_ids_not_in_source_files(self, dirs, monkeypatch):
        """HuisIdBSV and ProjectIdBSV are injected from the index, not read from files."""
        huis_ids    = [10, 20]
        project_ids = [3, 7]
        index_df = _make_index(huis_ids, project_ids)

        # Write files WITHOUT HuisIdBSV or ProjectIdBSV columns
        dates = pd.date_range("2023-01-01", periods=ROWS_PER_HH, freq="5min", tz="UTC")
        for huis_id in huis_ids:
            df = pd.DataFrame({
                "ReadingDate": dates,
                COL_SHARED:   pd.array([1.0] * ROWS_PER_HH, dtype="Float64"),
            })
            df.to_parquet(
                os.path.join(dirs["mapped"], f"household_{huis_id}_table.parquet"),
                engine="pyarrow",
            )

        monkeypatch.setattr("etdtransform.aggregate.read_index", lambda: (index_df, None))
        aggregate_hh_data_duckdb(columns=[COL_SHARED])

        out = pd.read_parquet(
            os.path.join(dirs["aggregate"], "household_default.parquet"),
            dtype_backend="numpy_nullable",
        )
        assert set(out["HuisIdBSV"].unique()) == {10, 20}
        assert set(out["ProjectIdBSV"].unique()) == {3, 7}
        assert out["HuisIdBSV"].dtype == "Int64"
        assert out["ProjectIdBSV"].dtype == "Int64"

    def test_mixed_schemas_union_by_name(self, dirs, monkeypatch):
        """Files with different column sets union cleanly; missing cols are NULL."""
        huis_ids    = list(range(1, 7))
        project_ids = [1, 1, 1, 2, 2, 2]
        index_df = _make_index(huis_ids, project_ids)
        _write_mixed_schema_files(dirs["mapped"])
        monkeypatch.setattr("etdtransform.aggregate.read_index", lambda: (index_df, None))

        aggregate_hh_data_duckdb(columns=[COL_SHARED, COL_A_ONLY, COL_B_ONLY])

        out = pd.read_parquet(
            os.path.join(dirs["aggregate"], "household_default.parquet"),
            dtype_backend="numpy_nullable",
        )
        assert len(out) == len(huis_ids) * ROWS_PER_HH

        # Supplier A rows: COL_A_ONLY present, COL_B_ONLY all-NA
        a_rows = out[out["HuisIdBSV"].isin([1, 2, 3])]
        assert a_rows[COL_A_ONLY].notna().all(), "COL_A_ONLY should be non-null for supplier A"
        assert a_rows[COL_B_ONLY].isna().all(), "COL_B_ONLY should be all-NA for supplier A"

        # Supplier B rows: COL_B_ONLY present, COL_A_ONLY all-NA
        b_rows = out[out["HuisIdBSV"].isin([4, 5, 6])]
        assert b_rows[COL_B_ONLY].notna().all(), "COL_B_ONLY should be non-null for supplier B"
        assert b_rows[COL_A_ONLY].isna().all(), "COL_A_ONLY should be all-NA for supplier B"

    def test_mixed_schemas_column_absent_from_all_files_ignored(self, dirs, monkeypatch):
        """Requesting a column that exists in no file produces no error — col is omitted."""
        huis_ids    = [1, 2]
        project_ids = [1, 1]
        index_df = _make_index(huis_ids, project_ids)
        _write_hh_files(dirs["mapped"], huis_ids)
        monkeypatch.setattr("etdtransform.aggregate.read_index", lambda: (index_df, None))

        aggregate_hh_data_duckdb(columns=["NonExistentColumn", DIFF_COL])

        out = pd.read_parquet(
            os.path.join(dirs["aggregate"], "household_default.parquet"),
            dtype_backend="numpy_nullable",
        )
        assert "NonExistentColumn" not in out.columns
        assert DIFF_COL in out.columns

    def test_column_projection_excludes_unrequested(self, dirs, monkeypatch):
        """Columns not in the requested list are absent from the output."""
        huis_ids    = [1, 2]
        project_ids = [1, 1]
        index_df = _make_index(huis_ids, project_ids)
        _write_mixed_schema_files(dirs["mapped"])
        monkeypatch.setattr("etdtransform.aggregate.read_index", lambda: (index_df, None))

        aggregate_hh_data_duckdb(columns=[COL_SHARED])

        out = pd.read_parquet(
            os.path.join(dirs["aggregate"], "household_default.parquet"),
            dtype_backend="numpy_nullable",
        )
        assert COL_SHARED in out.columns
        assert COL_A_ONLY not in out.columns
        assert COL_B_ONLY not in out.columns

    def test_stratified_sampling(self, dirs, monkeypatch):
        """sample_ratio keeps representation from each project."""
        huis_ids    = list(range(1, 21))
        project_ids = [1] * 10 + [2] * 10
        index_df = _make_index(huis_ids, project_ids)
        _write_hh_files(dirs["mapped"], huis_ids)
        monkeypatch.setattr("etdtransform.aggregate.read_index", lambda: (index_df, None))

        aggregate_hh_data_duckdb(sample_ratio=0.5)

        out = pd.read_parquet(
            os.path.join(dirs["aggregate"], "household_default.parquet"),
            dtype_backend="numpy_nullable",
        )
        assert out["HuisIdBSV"].nunique() == 10
        assert out[out["ProjectIdBSV"] == 1]["HuisIdBSV"].nunique() > 0
        assert out[out["ProjectIdBSV"] == 2]["HuisIdBSV"].nunique() > 0

    def test_missing_files_skipped(self, dirs, monkeypatch):
        """Households in the index with no parquet file are silently skipped."""
        huis_ids    = [1, 2, 3, 4]
        project_ids = [1, 1, 2, 2]
        index_df = _make_index(huis_ids, project_ids)
        _write_hh_files(dirs["mapped"], [1, 3])
        monkeypatch.setattr("etdtransform.aggregate.read_index", lambda: (index_df, None))

        aggregate_hh_data_duckdb()

        out = pd.read_parquet(
            os.path.join(dirs["aggregate"], "household_default.parquet"),
            dtype_backend="numpy_nullable",
        )
        assert set(out["HuisIdBSV"].unique()) == {1, 3}

    def test_all_files_missing_raises(self, dirs, monkeypatch):
        """ValueError raised when no parquet files exist for any listed household."""
        index_df = _make_index([99, 100], [1, 1])
        monkeypatch.setattr("etdtransform.aggregate.read_index", lambda: (index_df, None))

        with pytest.raises(ValueError):
            aggregate_hh_data_duckdb()

    def test_matches_pandas_baseline_shared_columns(self, dirs, monkeypatch):
        """
        DuckDB output matches the pandas baseline (aggregate_hh_data_5min) on the
        shared columns when all files have the same schema.

        This test guards against regressions where the two implementations diverge
        in row counts, ID assignment, or value accuracy.
        """
        huis_ids    = list(range(1, 8))
        project_ids = [1, 1, 1, 2, 2, 2, 2]
        index_df = _make_index(huis_ids, project_ids)
        _write_hh_files(dirs["mapped"], huis_ids)
        monkeypatch.setattr("etdtransform.aggregate.read_index", lambda: (index_df, None))

        out_path = os.path.join(dirs["aggregate"], "household_default.parquet")

        # pandas baseline
        aggregate_hh_data_5min()
        pandas_out = (
            pd.read_parquet(out_path, dtype_backend="numpy_nullable")
            .sort_values(["HuisIdBSV", "ReadingDate"])
            .reset_index(drop=True)
        )

        # duckdb
        aggregate_hh_data_duckdb(columns=[DIFF_COL])
        duckdb_out = (
            pd.read_parquet(out_path, dtype_backend="numpy_nullable")
            .sort_values(["HuisIdBSV", "ReadingDate"])
            .reset_index(drop=True)
        )

        assert len(duckdb_out) == len(pandas_out), "Row counts differ"
        assert set(duckdb_out["HuisIdBSV"].unique()) == set(pandas_out["HuisIdBSV"].unique())
        assert set(duckdb_out["ProjectIdBSV"].unique()) == set(pandas_out["ProjectIdBSV"].unique())

        # Values for the shared column must match
        pd.testing.assert_series_equal(
            duckdb_out[DIFF_COL].astype(float),
            pandas_out[DIFF_COL].astype(float),
            check_names=False,
            rtol=1e-6,
        )

    def test_matches_pandas_baseline_mixed_schemas(self, dirs, monkeypatch):
        """
        DuckDB handles mixed-schema files correctly; pandas baseline (which also uses
        pd.concat) agrees on the shared column and on row/ID counts.

        This specifically exercises the union_by_name=True path that has no pandas
        equivalent beyond pd.concat's natural behaviour.
        """
        huis_ids    = list(range(1, 7))
        project_ids = [1, 1, 1, 2, 2, 2]
        index_df = _make_index(huis_ids, project_ids)
        _write_mixed_schema_files(dirs["mapped"])
        monkeypatch.setattr("etdtransform.aggregate.read_index", lambda: (index_df, None))

        out_path = os.path.join(dirs["aggregate"], "household_default.parquet")

        aggregate_hh_data_5min()
        pandas_out = (
            pd.read_parquet(out_path, dtype_backend="numpy_nullable")
            .sort_values(["HuisIdBSV", "ReadingDate"])
            .reset_index(drop=True)
        )

        aggregate_hh_data_duckdb(columns=[COL_SHARED, COL_A_ONLY, COL_B_ONLY])
        duckdb_out = (
            pd.read_parquet(out_path, dtype_backend="numpy_nullable")
            .sort_values(["HuisIdBSV", "ReadingDate"])
            .reset_index(drop=True)
        )

        assert len(duckdb_out) == len(pandas_out)
        assert set(duckdb_out["HuisIdBSV"].unique()) == set(pandas_out["HuisIdBSV"].unique())

        # Shared column values must agree
        pd.testing.assert_series_equal(
            duckdb_out[COL_SHARED].astype(float),
            pandas_out[COL_SHARED].astype(float),
            check_names=False,
            rtol=1e-6,
        )


# ===========================================================================
# impute_hh_data_5min_chunked tests
# ===========================================================================

_IMPUTE_CUM_COL  = "ElektriciteitNetgebruikLaag"
_IMPUTE_DIFF_COL = "ElektriciteitNetgebruikLaagDiff"
_IMPUTE_AVG_COL  = "ElektriciteitNetgebruikLaagDiff_avg"
_IMPUTE_ROWS     = 20


def _write_impute_source(agg_dir: str, huis_ids: list, project_id: int = 1) -> None:
    """
    Write household_default.parquet with BOTH cumulative and Diff columns.

    Regression guard: the chunked imputer must read *Diff columns from the
    source parquet.  If needed_cols omits them, every column is skipped and
    impute_and_normalize raises RuntimeError.
    """
    rng   = np.random.default_rng(99)
    dates = pd.date_range("2023-01-01", periods=_IMPUTE_ROWS, freq="5min")
    frames = []
    for hid in huis_ids:
        diff_vals = rng.uniform(0.1, 1.0, _IMPUTE_ROWS)
        frames.append(pd.DataFrame({
            "HuisIdBSV":      pd.array([hid]     * _IMPUTE_ROWS, dtype="Int64"),
            "ProjectIdBSV":   pd.array([project_id] * _IMPUTE_ROWS, dtype="Int64"),
            "ReadingDate":    dates,
            _IMPUTE_CUM_COL:  pd.array(diff_vals.cumsum(), dtype="Float64"),
            _IMPUTE_DIFF_COL: pd.array(diff_vals, dtype="Float64"),
        }))
    pd.concat(frames, ignore_index=True).to_parquet(
        os.path.join(agg_dir, "household_default.parquet"), engine="pyarrow"
    )


def _write_impute_artifacts(agg_dir: str, dates, project_id: int = 1) -> None:
    """Write minimal avg_diffs.parquet and household_diff_max_bounds.parquet."""
    avg = pd.DataFrame({
        "ProjectIdBSV":   pd.array([project_id] * len(dates), dtype="Int64"),
        "ReadingDate":    dates,
        _IMPUTE_AVG_COL:  pd.array([0.5] * len(dates), dtype="Float64"),
    })
    avg.to_parquet(os.path.join(agg_dir, "avg_diffs.parquet"), engine="pyarrow")

    # max_bound is passed to impute_and_normalize but not read inside it.
    pd.DataFrame({"ProjectIdBSV": pd.array([], dtype="Int64")}).to_parquet(
        os.path.join(agg_dir, "household_diff_max_bounds.parquet"), engine="pyarrow"
    )


class TestChunkedImpute:
    def test_diff_columns_read_from_source(self, dirs):
        """
        Regression: impute_hh_data_5min_chunked must include *Diff columns in
        needed_cols when reading each chunk.  Before the fix, needed_cols only
        contained cumulative columns, so every column was skipped and
        impute_and_normalize raised RuntimeError('No columns were imputed').
        """
        huis_ids = list(range(1, 6))  # 5 households, chunk_size=3 forces 2 chunks
        dates    = pd.date_range("2023-01-01", periods=_IMPUTE_ROWS, freq="5min")
        _write_impute_source(dirs["aggregate"], huis_ids)
        _write_impute_artifacts(dirs["aggregate"], dates)

        # Must not raise
        impute_hh_data_5min_chunked(
            source_path=os.path.join(dirs["aggregate"], "household_default.parquet"),
            chunk_size=3,
            cum_cols=[_IMPUTE_CUM_COL],
        )

        out = pd.read_parquet(
            os.path.join(dirs["aggregate"], "household_imputed.parquet"),
            dtype_backend="numpy_nullable",
        )
        assert _IMPUTE_DIFF_COL in out.columns, "Diff column missing from imputed output"
        assert set(out["HuisIdBSV"].unique()) == set(huis_ids)
        assert len(out) == len(huis_ids) * _IMPUTE_ROWS

    def test_huis_ids_filter_restricts_output(self, dirs):
        """huis_ids parameter limits which households are imputed."""
        huis_ids   = list(range(1, 7))   # 6 in source
        keep_ids   = [1, 3, 5]
        dates      = pd.date_range("2023-01-01", periods=_IMPUTE_ROWS, freq="5min")
        _write_impute_source(dirs["aggregate"], huis_ids)
        _write_impute_artifacts(dirs["aggregate"], dates)

        impute_hh_data_5min_chunked(
            source_path=os.path.join(dirs["aggregate"], "household_default.parquet"),
            chunk_size=2,
            cum_cols=[_IMPUTE_CUM_COL],
            huis_ids=keep_ids,
        )

        out = pd.read_parquet(
            os.path.join(dirs["aggregate"], "household_imputed.parquet"),
            dtype_backend="numpy_nullable",
        )
        assert set(out["HuisIdBSV"].unique()) == set(keep_ids)
        assert len(out) == len(keep_ids) * _IMPUTE_ROWS


class TestAvgDiffsAlignment:
    def test_avg_diffs_column_alignment_across_projects(self):
        """
        Regression: concatenate_avg_diff_columns must use outer merge, not positional
        concat. Project 1 has ColADiff data only, project 2 has ColBDiff data only.
        After concatenation each project must get pd.NA for the other project's column.
        """
        from etdtransform.impute import concatenate_avg_diff_columns

        dates = pd.date_range("2023-01-01", periods=3, freq="5min")
        avg_A = pd.DataFrame({
            "ProjectIdBSV": pd.array([1, 1, 1], dtype="Int64"),
            "ReadingDate": dates,
            "ColADiff_avg": pd.array([0.1, 0.2, 0.3], dtype="Float64"),
        })
        avg_B = pd.DataFrame({
            "ProjectIdBSV": pd.array([2, 2, 2], dtype="Int64"),
            "ReadingDate": dates,
            "ColBDiff_avg": pd.array([0.4, 0.5, 0.6], dtype="Float64"),
        })
        avg_diff_dict = {
            "ColADiff": {
                "avg_diff": avg_A,
                "upper_bounds": pd.DataFrame(),
                "household_max_with_bounds": pd.DataFrame(),
            },
            "ColBDiff": {
                "avg_diff": avg_B,
                "upper_bounds": pd.DataFrame(),
                "household_max_with_bounds": pd.DataFrame(),
            },
        }
        result = concatenate_avg_diff_columns(avg_diff_dict, "ProjectIdBSV")

        proj1 = result[result["ProjectIdBSV"] == 1]
        proj2 = result[result["ProjectIdBSV"] == 2]
        assert len(proj1) == 3, "Project 1 must have 3 rows"
        assert len(proj2) == 3, "Project 2 must have 3 rows"
        assert proj1["ColADiff_avg"].notna().all(), "Project 1 ColA avg must be present"
        assert proj1["ColBDiff_avg"].isna().all(), "Project 1 ColB avg must be pd.NA"
        assert proj2["ColBDiff_avg"].notna().all(), "Project 2 ColB avg must be present"
        assert proj2["ColADiff_avg"].isna().all(), "Project 2 ColA avg must be pd.NA"

    def test_na_avg_no_end_value_leaves_diff_na(self):
        """
        Regression: POSITIVE_END_VALUE and NO_END_VALUE imputation paths use
        impute_values directly. Before the fix, fillna(0) substituted 0 for a
        missing avg, causing gaps to be filled with spurious zeros when avg was
        unavailable (e.g. due to cross-project concat misalignment).
        ADR-005: if avg is pd.NA and there is no end cumulative bookend to
        derive a gap_jump from, the diff must remain pd.NA.
        """
        from etdtransform.vectorized_impute import impute_and_normalize

        cum_col = _IMPUTE_CUM_COL
        diff_col = _IMPUTE_DIFF_COL
        avg_col = _IMPUTE_AVG_COL
        dates = pd.date_range("2023-01-01", periods=5, freq="5min")
        # Cumulative drops to NA and never recovers: no end bookend for the gap,
        # so no gap_jump is computable. Only the NO_END_VALUE path can fire,
        # and it must NOT fill with 0 when avg is NA.
        df = pd.DataFrame({
            "HuisIdBSV":    pd.array([1, 1, 1, 1, 1], dtype="Int64"),
            "ProjectIdBSV": pd.array([1, 1, 1, 1, 1], dtype="Int64"),
            "ReadingDate":  dates,
            cum_col:  pd.array([0.5, pd.NA, pd.NA, pd.NA, pd.NA], dtype="Float64"),
            diff_col: pd.array([0.5, pd.NA, pd.NA, pd.NA, pd.NA], dtype="Float64"),
            avg_col:  pd.array([pd.NA, pd.NA, pd.NA, pd.NA, pd.NA], dtype="Float64"),
        })
        max_bound = pd.DataFrame({
            "ProjectIdBSV":            pd.array([1], dtype="Int64"),
            "HuisIdBSV":               pd.array([1], dtype="Int64"),
            f"{diff_col}_huis_max":    pd.array([1.0], dtype="Float64"),
            f"{diff_col}_upper_bound": pd.array([2.0], dtype="Float64"),
        })
        result_df, _, _ = impute_and_normalize(df, [cum_col], "ProjectIdBSV", max_bound)

        for i in [1, 2, 3, 4]:
            assert pd.isna(result_df.loc[i, diff_col]), (
                f"Gap slot {i} with no end cumulative and NA avg must remain NA, got {result_df.loc[i, diff_col]}"
            )
        is_imputed_col = f"{diff_col}_is_imputed"
        if is_imputed_col in result_df.columns:
            for i in [1, 2, 3, 4]:
                assert not result_df.loc[i, is_imputed_col], f"is_imputed must be False when avg is NA (slot {i})"
