"""
Tests that Ibis-based functions produce the same outputs as their pandas
equivalents.  All tests use a synthetic in-memory fixture — ibis.memtable()
wraps the fixture DataFrame for the Ibis path.

The fixture intentionally contains null (pd.NA) values in measurement columns
to verify that both paths handle missingness identically.  Null-in, null-out
is a correctness requirement throughout the ETD data science pipeline.
"""
import numpy as np
import pandas as pd
import pytest

import ibis

from etdtransform.impute import calculate_average_diff, calculate_average_diff_ibis

# Two projects: 20 normal + 1 outlier per project.
# Outlier values are ~15-20× larger than the normal range so the
# 95th-percentile upper-bound logic clearly excludes them.
_P1_NORMAL = list(range(101, 121))
_P1_OUTLIER = 199
_P2_NORMAL = list(range(201, 221))
_P2_OUTLIER = 299

DIFF_COLS = ["ElektriciteitNetgebruikLaagDiff", "Zon-opwekTotaalDiff"]

# Rows whose measurement values are set to pd.NA to verify null handling.
# We use (project_id, huis_id, timestamp_index) tuples to identify them.
_NULL_SLOTS = {(1, 105, 3), (1, 110, 7), (2, 207, 0), (2, 215, 12)}


@pytest.fixture(scope="module")
def diff_fixture_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=20, freq="5min").as_unit("us")
    rows = []

    for project_id, normal_ids, outlier_id in [
        (1, _P1_NORMAL, _P1_OUTLIER),
        (2, _P2_NORMAL, _P2_OUTLIER),
    ]:
        for huis_id in normal_ids:
            for t_idx, dt in enumerate(dates):
                null_this = (project_id, huis_id, t_idx) in _NULL_SLOTS
                rows.append({
                    "ProjectIdBSV": project_id,
                    "HuisIdBSV": huis_id,
                    "ReadingDate": dt,
                    "ElektriciteitNetgebruikLaagDiff": pd.NA if null_this else float(rng.uniform(1.0, 10.0)),
                    "Zon-opwekTotaalDiff": pd.NA if null_this else float(rng.uniform(0.5, 5.0)),
                })
        for dt in dates:
            rows.append({
                "ProjectIdBSV": project_id,
                "HuisIdBSV": outlier_id,
                "ReadingDate": dt,
                "ElektriciteitNetgebruikLaagDiff": float(rng.uniform(150.0, 300.0)),
                "Zon-opwekTotaalDiff": float(rng.uniform(150.0, 300.0)),
            })

    df = pd.DataFrame(rows).astype({
        "ProjectIdBSV": "Int64",
        "HuisIdBSV": "Int64",
        "ElektriciteitNetgebruikLaagDiff": "Float64",
        "Zon-opwekTotaalDiff": "Float64",
    })
    # Row-by-row construction reverts timestamps to ns; pin to us to match
    # DuckDB's output precision and the parquet files produced by the pipeline.
    df["ReadingDate"] = df["ReadingDate"].astype("datetime64[us]")
    return df


def _null_positions(df: pd.DataFrame, col: str) -> set:
    """Return the integer-location set of null positions in a column."""
    return set(df.index[df[col].isna()].tolist())


def test_ibis_memtable_preserves_nulls(diff_fixture_df):
    """ibis.memtable round-trip must preserve null positions and values exactly."""
    tbl = ibis.memtable(diff_fixture_df)
    result = tbl.execute().astype({c: diff_fixture_df[c].dtype for c in diff_fixture_df.columns})

    sort_cols = ["HuisIdBSV", "ReadingDate"]
    original = diff_fixture_df.sort_values(sort_cols).reset_index(drop=True)
    restored = result.sort_values(sort_cols).reset_index(drop=True)

    for col in DIFF_COLS:
        orig_nulls = _null_positions(original, col)
        rest_nulls = _null_positions(restored, col)
        assert orig_nulls == rest_nulls, (
            f"{col}: null positions changed after ibis round-trip. "
            f"original={orig_nulls}, restored={rest_nulls}"
        )

    pd.testing.assert_frame_equal(original, restored, check_like=True)


def test_calculate_average_diff_ibis_matches_pandas(diff_fixture_df):
    """Ibis and pandas paths agree on avg_diff values and null positions."""
    tbl = ibis.memtable(diff_fixture_df)
    result_pd = calculate_average_diff(diff_fixture_df, "ProjectIdBSV", DIFF_COLS)
    result_ibis = calculate_average_diff_ibis(tbl, "ProjectIdBSV", DIFF_COLS)

    sort_avg = ["ProjectIdBSV", "ReadingDate"]
    sort_ub = ["ProjectIdBSV"]

    for col in DIFF_COLS:
        avg_pd = result_pd[col]["avg_diff"].sort_values(sort_avg).reset_index(drop=True)
        avg_ib = result_ibis[col]["avg_diff"].sort_values(sort_avg).reset_index(drop=True)

        # Null positions in the avg column must match
        assert _null_positions(avg_pd, f"{col}_avg") == _null_positions(avg_ib, f"{col}_avg"), (
            f"{col}: avg_diff null positions differ between pandas and Ibis"
        )
        pd.testing.assert_frame_equal(avg_pd, avg_ib, check_like=True, rtol=1e-5)

        ub_pd = result_pd[col]["upper_bounds"].sort_values(sort_ub).reset_index(drop=True)
        ub_ib = result_ibis[col]["upper_bounds"].sort_values(sort_ub).reset_index(drop=True)
        pd.testing.assert_frame_equal(ub_pd, ub_ib, check_like=True, rtol=1e-5)


def test_outlier_households_excluded(diff_fixture_df):
    """Outlier households must sit above the upper bound and be excluded."""
    tbl = ibis.memtable(diff_fixture_df)
    result = calculate_average_diff_ibis(tbl, "ProjectIdBSV", DIFF_COLS)

    outlier_ids = [_P1_OUTLIER, _P2_OUTLIER]
    for col in DIFF_COLS:
        bounds = result[col]["household_max_with_bounds"]
        outlier_rows = bounds[bounds["HuisIdBSV"].isin(outlier_ids)]
        assert not outlier_rows.empty, f"Outlier rows missing for {col}"
        assert (
            outlier_rows[f"{col}_huis_max"] > outlier_rows[f"{col}_upper_bound"]
        ).all(), f"Outlier not above upper bound for {col}"


def test_nulls_excluded_from_mean(diff_fixture_df):
    """Null Diff values must not affect the mean (both paths skip them)."""
    # Build a minimal frame: one project, two households, three timestamps.
    # huis 1: readings [2.0, pd.NA, 4.0]  → mean = 3.0
    # huis 2: readings [6.0, 6.0,  6.0]  → mean = 6.0
    # project mean over both households = (3.0 + 6.0) / 2 = 4.5
    # (pandas and SQL both skip NULLs in mean, so average is over non-null pairs)
    dates = pd.date_range("2024-01-01", periods=3, freq="5min")
    mini = pd.DataFrame({
        "ProjectIdBSV": pd.array([1, 1, 1, 1, 1, 1], dtype="Int64"),
        "HuisIdBSV": pd.array([1, 1, 1, 2, 2, 2], dtype="Int64"),
        "ReadingDate": list(dates) * 2,
        "TestDiff": pd.array([2.0, pd.NA, 4.0, 6.0, 6.0, 6.0], dtype="Float64"),
    })

    result_pd = calculate_average_diff(mini, "ProjectIdBSV", ["TestDiff"])
    result_ibis = calculate_average_diff_ibis(ibis.memtable(mini), "ProjectIdBSV", ["TestDiff"])

    avg_pd = result_pd["TestDiff"]["avg_diff"]["TestDiff_avg"].sort_values().values
    avg_ib = result_ibis["TestDiff"]["avg_diff"]["TestDiff_avg"].sort_values().values

    np.testing.assert_allclose(avg_pd, avg_ib, rtol=1e-6)
    # Verify the mean correctly ignores the null
    assert not np.isnan(avg_pd).any(), "NaN leaked into pandas mean despite non-null peers"
    assert not avg_ib.isna().any() if hasattr(avg_ib, "isna") else not np.isnan(avg_ib).any()


def test_empty_project_edge_cases():
    """Edge cases for projects with no qualifying data.

    Three degenerate situations that must not crash and must return consistent
    structures across both paths:
      A) All diff values are zero (filtered out by the > 1e-8 guard) →
         upper_bound is null, avg_diff is empty.
      B) All diff values are pd.NA →
         same outcome as (A): no household max qualifies.
      C) One 'normal' project alongside a zero-only project →
         normal project processes correctly; zero project gets null upper_bound.
    """
    dates = pd.date_range("2024-06-01", periods=3, freq="5min")

    # Project 1: all zeros. Project 2: normal non-zero values.
    mini = pd.DataFrame({
        "ProjectIdBSV": pd.array([1, 1, 1, 2, 2, 2], dtype="Int64"),
        "HuisIdBSV":    pd.array([1, 1, 1, 2, 2, 2], dtype="Int64"),
        "ReadingDate":  list(dates) * 2,
        "TestDiff":     pd.array([0.0, 0.0, 0.0, 3.0, 5.0, 4.0], dtype="Float64"),
    })
    mini["ReadingDate"] = mini["ReadingDate"].astype("datetime64[us]")

    result_pd   = calculate_average_diff(mini, "ProjectIdBSV", ["TestDiff"])
    result_ibis = calculate_average_diff_ibis(ibis.memtable(mini), "ProjectIdBSV", ["TestDiff"])

    for result, label in [(result_pd, "pandas"), (result_ibis, "ibis")]:
        ub = result["TestDiff"]["upper_bounds"]
        hmb = result["TestDiff"]["household_max_with_bounds"]
        avg = result["TestDiff"]["avg_diff"]

        # Both projects must appear in upper_bounds
        assert set(ub["ProjectIdBSV"].dropna()) >= {2}, f"{label}: project 2 missing from upper_bounds"
        assert 1 in ub["ProjectIdBSV"].values, f"{label}: project 1 missing from upper_bounds"

        # Project 1 upper_bound must be null (no qualifying households)
        p1_ub = ub.loc[ub["ProjectIdBSV"] == 1, "TestDiff_upper_bound"]
        assert p1_ub.isna().all(), f"{label}: project 1 upper_bound should be null, got {p1_ub.values}"

        # Project 2 upper_bound must be non-null and positive
        p2_ub = ub.loc[ub["ProjectIdBSV"] == 2, "TestDiff_upper_bound"]
        assert p2_ub.notna().all() and (p2_ub > 0).all(), (
            f"{label}: project 2 upper_bound should be positive, got {p2_ub.values}"
        )

        # avg_diff must have rows for project 2 only (project 1 excluded)
        assert (avg["ProjectIdBSV"] == 2).all(), f"{label}: avg_diff should only contain project 2"
        assert len(avg) == 3, f"{label}: expected 3 avg rows for project 2, got {len(avg)}"

        # household_max_with_bounds must include all households
        assert set(hmb["HuisIdBSV"]) == {1, 2}, f"{label}: unexpected households in hmb: {set(hmb['HuisIdBSV'])}"

    # The two paths must agree exactly on upper_bounds and avg_diff
    sort_ub  = ["ProjectIdBSV"]
    sort_avg = ["ProjectIdBSV", "ReadingDate"]
    pd.testing.assert_frame_equal(
        result_pd["TestDiff"]["upper_bounds"].sort_values(sort_ub).reset_index(drop=True),
        result_ibis["TestDiff"]["upper_bounds"].sort_values(sort_ub).reset_index(drop=True),
        check_like=True, rtol=1e-5,
    )
    pd.testing.assert_frame_equal(
        result_pd["TestDiff"]["avg_diff"].sort_values(sort_avg).reset_index(drop=True),
        result_ibis["TestDiff"]["avg_diff"].sort_values(sort_avg).reset_index(drop=True),
        check_like=True, rtol=1e-5,
    )


def test_all_null_diffs_edge_case():
    """When all diff values for a project are pd.NA, behaviour mirrors all-zeros."""
    dates = pd.date_range("2024-06-01", periods=2, freq="5min")
    mini = pd.DataFrame({
        "ProjectIdBSV": pd.array([1, 1], dtype="Int64"),
        "HuisIdBSV":    pd.array([1, 1], dtype="Int64"),
        "ReadingDate":  list(dates),
        "TestDiff":     pd.array([pd.NA, pd.NA], dtype="Float64"),
    })
    mini["ReadingDate"] = mini["ReadingDate"].astype("datetime64[us]")

    result_pd   = calculate_average_diff(mini, "ProjectIdBSV", ["TestDiff"])
    result_ibis = calculate_average_diff_ibis(ibis.memtable(mini), "ProjectIdBSV", ["TestDiff"])

    for result, label in [(result_pd, "pandas"), (result_ibis, "ibis")]:
        ub = result["TestDiff"]["upper_bounds"]
        avg = result["TestDiff"]["avg_diff"]

        p1_ub = ub.loc[ub["ProjectIdBSV"] == 1, "TestDiff_upper_bound"]
        assert p1_ub.isna().all(), f"{label}: all-null project upper_bound should be null"
        assert len(avg) == 0, f"{label}: avg_diff should be empty for all-null project, got {len(avg)} rows"
