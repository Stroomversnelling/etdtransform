"""
Cross-check tests: pandas, ibis, and duckdb pipeline paths must produce
identical outputs on the test fixture (10 households, 2 projects).

NOTE: These are integration tests that use the local test fixture dataset
configured in config_test.yaml (see config_test_template.yaml for the
reference template). The fixture is a small anonymised dataset -- NOT
production data. Tests must pass on that fixture; any failure is a real bug.

Each session fixture runs one complete pipeline variant into an isolated temp
directory. Comparison tests load matching parquets from two paths and assert
numeric equality.

Covered pairs
-------------
Aggregation  : aggregate_hh_data_5min (pandas)
               aggregate_hh_data_5min_ibis (ibis/DuckDB batched)
               aggregate_hh_data_duckdb (DuckDB union_by_name)
               -> household_default.parquet

Diffs        : prepare_diffs_for_impute (pandas)
               prepare_diffs_for_impute_ibis (ibis)
               -> avg_diffs.parquet, household_diff_max_bounds.parquet

Imputation   : impute_hh_data_5min (pandas)
               impute_hh_data_5min_chunked (chunked pandas + parquet streaming)
               -> household_imputed.parquet

Calc columns : add_calculated_columns_to_hh_data (pandas)
               add_calculated_columns_to_hh_data_ibis (ibis/DuckDB)
               -> household_calculated.parquet

Column dtype requirements (ADR-005)
------------------------------------
All data columns must use pandas nullable dtypes (Float64, Int64, boolean,
string[python]). Parquets are always read with dtype_backend="numpy_nullable".

The authoritative column types are defined in etdmap.data_model.model_column_type.
"WarmtepompFoutmelding" is the only data model column typed as string. Its
parquet physical type must be utf8 (not large_utf8) so that pandas reads it
as string[python] rather than object. add_calculated_columns_to_hh_data_ibis()
enforces this via a PyArrow streaming passthrough after sink_parquet() -- see
etdtransform/docs/parquet-merge-strategy.md for details. If a dtype mismatch
appears in test_ibis_all_cols_match_pandas, fix the write path, not the test.
"""

import pandas as pd

# Pipeline fixtures (pandas_pipeline, ibis_pipeline, duckdb_pipeline) and their
# Meenemen / cum_cols dependencies live in tests/conftest.py so other test
# files (e.g. test_resample_duckdb) can share the same outputs without
# re-running the pipelines.


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REL_TOL = 1e-6
SORT_COLS = ("HuisIdBSV", "ReadingDate")

DERIVED_COLS = [
    "TerugleveringTotaalNetto",
    "ElektriciteitsgebruikTotaalNetto",
    "ElektriciteitsgebruikTotaalWarmtepomp",
    "ElektriciteitsgebruikTotaalGebouwgebonden",
    "ElektriciteitsgebruikTotaalHuishoudelijk",
    "Zelfgebruik",
    "ElektriciteitsgebruikTotaalBruto",
]


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


def _assert_frames_equal(df_a, df_b, label_a, label_b, cols=None):
    """Compare two DataFrames using pd.testing.assert_frame_equal.

    pd.NA is a first-class value: pd.NA in one path and 0.0 in the other is a
    failure, not a tolerance issue. assert_frame_equal handles this natively
    for nullable dtypes (Float64, Int64, boolean) without converting to float/nan.

    Columns are restricted to the shared set (or `cols` if provided) and
    sorted consistently so column order differences don't cause spurious failures.
    """
    shared = sorted(set(df_a.columns) & set(df_b.columns))
    if cols is not None:
        shared = [c for c in cols if c in shared]
    pd.testing.assert_frame_equal(
        df_a[shared].reset_index(drop=True),
        df_b[shared].reset_index(drop=True),
        check_like=False,
        rtol=REL_TOL,
        obj=f"{label_a} vs {label_b}",
    )


# ---------------------------------------------------------------------------
# Tests: Aggregation step
# ---------------------------------------------------------------------------

class TestAggregationEquivalence:
    """household_default.parquet: all three approaches must produce the same data."""

    def test_ibis_matches_pandas(self, pandas_pipeline, ibis_pipeline):
        p = _load(pandas_pipeline / "household_default.parquet", SORT_COLS)
        i = _load(ibis_pipeline / "household_default.parquet", SORT_COLS)
        _assert_frames_equal(p, i, "pandas", "ibis")

    def test_duckdb_matches_pandas(self, pandas_pipeline, duckdb_pipeline):
        p = _load(pandas_pipeline / "household_default.parquet", SORT_COLS)
        d = _load(duckdb_pipeline / "household_default.parquet", SORT_COLS)
        _assert_frames_equal(p, d, "pandas", "duckdb")

    def test_ibis_same_households_as_pandas(self, pandas_pipeline, ibis_pipeline):
        p_ids = sorted(_load(pandas_pipeline / "household_default.parquet")["HuisIdBSV"].dropna().unique().tolist())
        i_ids = sorted(_load(ibis_pipeline / "household_default.parquet")["HuisIdBSV"].dropna().unique().tolist())
        assert p_ids == i_ids

    def test_duckdb_same_households_as_pandas(self, pandas_pipeline, duckdb_pipeline):
        p_ids = sorted(_load(pandas_pipeline / "household_default.parquet")["HuisIdBSV"].dropna().unique().tolist())
        d_ids = sorted(_load(duckdb_pipeline / "household_default.parquet")["HuisIdBSV"].dropna().unique().tolist())
        assert p_ids == d_ids


# ---------------------------------------------------------------------------
# Tests: Diffs step
# ---------------------------------------------------------------------------

class TestDiffsEquivalence:
    """avg_diffs.parquet and household_diff_max_bounds.parquet: pandas vs ibis."""

    def test_avg_diffs_ibis_matches_pandas(self, pandas_pipeline, ibis_pipeline):
        p = _load(pandas_pipeline / "avg_diffs.parquet", ("ProjectIdBSV", "ReadingDate"))
        i = _load(ibis_pipeline / "avg_diffs.parquet", ("ProjectIdBSV", "ReadingDate"))
        _assert_frames_equal(p, i, "pandas", "ibis")

    def test_max_bounds_ibis_matches_pandas(self, pandas_pipeline, ibis_pipeline):
        p = _load(pandas_pipeline / "household_diff_max_bounds.parquet", ("HuisIdBSV",))
        i = _load(ibis_pipeline / "household_diff_max_bounds.parquet", ("HuisIdBSV",))
        _assert_frames_equal(p, i, "pandas", "ibis")


# ---------------------------------------------------------------------------
# Tests: Imputation step
# ---------------------------------------------------------------------------

class TestImputationEquivalence:
    """household_imputed.parquet: pandas vs ibis chunked impute."""

    def test_ibis_matches_pandas(self, pandas_pipeline, ibis_pipeline):
        p = _load(pandas_pipeline / "household_imputed.parquet", SORT_COLS)
        i = _load(ibis_pipeline / "household_imputed.parquet", SORT_COLS)
        _assert_frames_equal(p, i, "pandas", "ibis")

    def test_ibis_same_households_as_pandas(self, pandas_pipeline, ibis_pipeline):
        p_ids = sorted(_load(pandas_pipeline / "household_imputed.parquet")["HuisIdBSV"].dropna().unique().tolist())
        i_ids = sorted(_load(ibis_pipeline / "household_imputed.parquet")["HuisIdBSV"].dropna().unique().tolist())
        assert p_ids == i_ids


# ---------------------------------------------------------------------------
# Tests: Calculated columns step
# ---------------------------------------------------------------------------

class TestCalculatedColumnsEquivalence:
    """household_calculated.parquet: pandas vs ibis derived columns."""

    def test_ibis_derived_cols_present(self, pandas_pipeline, ibis_pipeline):
        p = _load(pandas_pipeline / "household_calculated.parquet")
        i = _load(ibis_pipeline / "household_calculated.parquet")
        for col in DERIVED_COLS:
            if col in p.columns:
                assert col in i.columns, f"ibis output missing derived column '{col}'"

    def test_ibis_derived_cols_match_pandas(self, pandas_pipeline, ibis_pipeline):
        p = _load(pandas_pipeline / "household_calculated.parquet", SORT_COLS)
        i = _load(ibis_pipeline / "household_calculated.parquet", SORT_COLS)
        present = [c for c in DERIVED_COLS if c in p.columns and c in i.columns]
        _assert_frames_equal(p, i, "pandas", "ibis", cols=present)

    def test_ibis_all_cols_match_pandas(self, pandas_pipeline, ibis_pipeline):
        p = _load(pandas_pipeline / "household_calculated.parquet", SORT_COLS)
        i = _load(ibis_pipeline / "household_calculated.parquet", SORT_COLS)
        _assert_frames_equal(p, i, "pandas", "ibis")
