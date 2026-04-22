"""
Performance and regression tests for imputation gap stats calculation.

Three approaches are benchmarked:
  1. apply  -- original: one groupby().apply() call, Python function per group
  2. agg    -- current: one groupby().agg() for scalar stats + apply for methods list
  3. sep    -- alternative: 7 separate C-level groupby operations

Run with -s to see timing output:
  pytest tests/test_vectorized_impute.py -s -v
"""
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Re-implementations of the three approaches (self-contained, no etdtransform
# import needed — we are testing the algorithm, not the module integration)
# ---------------------------------------------------------------------------

def _gap_stats_apply(df, project_col, cum_col, diff_col, impute_type_col):
    """Original groupby().apply() approach."""
    def _calc(group):
        diff_total = group[diff_col].sum()
        cum_diff = group[cum_col].max() - group[cum_col].min()
        return pd.Series({
            "column": diff_col,
            "diff_col_total": diff_total,
            "cum_col_min_max_diff": cum_diff,
            "deviation": diff_total - cum_diff,
            "missing": (~group["gap_length"].isna()).sum(),
            "methods": sorted(set(v for v in group[impute_type_col] if pd.notna(v))),
            "imputed": group[impute_type_col].notna().sum(),
            "imputed_na": group["cumulative_value_group"].notna().sum() - group[impute_type_col].notna().sum(),
        })
    return (
        df.groupby([project_col, "HuisIdBSV"])
        .apply(_calc, include_groups=False)
        .reset_index()
    )


def _gap_stats_agg(df, project_col, cum_col, diff_col, impute_type_col):
    """Combined agg() for scalar stats + apply only for methods list."""
    grp = df.groupby([project_col, "HuisIdBSV"])
    stats = grp.agg(
        diff_col_total=(diff_col, "sum"),
        _cum_max=(cum_col, "max"),
        _cum_min=(cum_col, "min"),
        missing=("gap_length", "count"),
        imputed=(impute_type_col, "count"),
        _cvg_count=("cumulative_value_group", "count"),
    ).reset_index()
    stats["cum_col_min_max_diff"] = stats["_cum_max"] - stats["_cum_min"]
    stats["deviation"] = stats["diff_col_total"] - stats["cum_col_min_max_diff"]
    stats["imputed_na"] = stats["_cvg_count"] - stats["imputed"]
    stats["column"] = diff_col
    stats["methods"] = (
        grp[impute_type_col]
        .apply(lambda x: sorted(set(v for v in x if pd.notna(v))))
        .values
    )
    stats.drop(columns=["_cum_max", "_cum_min", "_cvg_count"], inplace=True)
    return stats


def _gap_stats_sep(df, project_col, cum_col, diff_col, impute_type_col):
    """7 separate C-level groupby operations."""
    grp = df.groupby([project_col, "HuisIdBSV"])
    diff_sum = grp[diff_col].sum()
    cum_max = grp[cum_col].max()
    cum_min = grp[cum_col].min()
    gap_count = grp["gap_length"].count()
    imputed_count = grp[impute_type_col].count()
    cvg_count = grp["cumulative_value_group"].count()
    methods_list = grp[impute_type_col].apply(
        lambda x: sorted(set(v for v in x if pd.notna(v)))
    )
    return pd.DataFrame({
        "column": diff_col,
        "diff_col_total": diff_sum,
        "cum_col_min_max_diff": cum_max - cum_min,
        "deviation": diff_sum - (cum_max - cum_min),
        "missing": gap_count,
        "methods": methods_list,
        "imputed": imputed_count,
        "imputed_na": cvg_count - imputed_count,
    }).reset_index()


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gap_stats_df():
    """
    Synthetic dataset mimicking df state during imputation gap stats calculation.
    2 projects x 50 households x 10000 rows = 1M rows, 1 cumulative column.
    """
    rng = np.random.default_rng(42)
    n_proj, n_hh, n_rows = 2, 50, 10_000
    n_total = n_proj * n_hh * n_rows

    proj = np.repeat(np.arange(1, n_proj + 1), n_hh * n_rows)
    huis = np.tile(np.repeat(np.arange(1, n_hh + 1), n_rows), n_proj) + (proj - 1) * 100

    diff_raw = rng.exponential(0.1, n_total)
    cum_vals = np.cumsum(diff_raw)

    gap_mask = rng.random(n_total) < 0.05
    imputed_mask = gap_mask & (rng.random(n_total) < 0.5)
    impute_types = rng.integers(1, 4, n_total).astype(float)
    cvg_groups = rng.integers(1, 100, n_total).astype(float)

    diff_arr = diff_raw.copy()
    diff_arr[gap_mask] = np.nan

    impute_type_arr = np.where(imputed_mask, impute_types, np.nan)
    gap_length_arr = np.where(gap_mask, 1.0, np.nan)
    cvg_arr = np.where(gap_mask, cvg_groups, np.nan)

    return pd.DataFrame({
        "ProjectIdBSV": pd.array(proj, dtype="Int64"),
        "HuisIdBSV": pd.array(huis, dtype="Int64"),
        "Diff": pd.array(diff_arr, dtype="Float64"),
        "Cum": pd.array(cum_vals, dtype="Float64"),
        "gap_length": pd.array(gap_length_arr, dtype="Float64"),
        "cumulative_value_group": pd.array(cvg_arr, dtype="Float64"),
        "impute_type": pd.array(impute_type_arr, dtype="Float64"),
    })


# ---------------------------------------------------------------------------
# Helper to normalise results for comparison
# ---------------------------------------------------------------------------

def _normalise(df, project_col):
    sort_cols = [project_col, "HuisIdBSV"]
    return (
        df.sort_values(sort_cols)
        .reset_index(drop=True)
        [[project_col, "HuisIdBSV", "diff_col_total", "cum_col_min_max_diff",
          "deviation", "missing", "imputed", "imputed_na", "methods"]]
    )


# ---------------------------------------------------------------------------
# Correctness: all three approaches must agree
# ---------------------------------------------------------------------------

def test_gap_stats_agg_matches_apply(gap_stats_df):
    """agg approach must produce identical scalar stats to the original apply."""
    proj, cum, diff, itype = "ProjectIdBSV", "Cum", "Diff", "impute_type"
    ref = _normalise(_gap_stats_apply(gap_stats_df, proj, cum, diff, itype), proj)
    new = _normalise(_gap_stats_agg(gap_stats_df, proj, cum, diff, itype), proj)

    scalar_cols = ["diff_col_total", "cum_col_min_max_diff", "deviation",
                   "missing", "imputed", "imputed_na"]
    for col in scalar_cols:
        pd.testing.assert_series_equal(
            ref[col].astype(float),
            new[col].astype(float),
            check_names=False,
            rtol=1e-5,
            obj=f"agg vs apply: {col}",
        )

    for apply_methods, agg_methods in zip(ref["methods"], new["methods"]):
        assert sorted(apply_methods) == sorted(agg_methods), (
            f"methods mismatch: apply={apply_methods}  agg={agg_methods}"
        )


def test_gap_stats_sep_matches_apply(gap_stats_df):
    """Separate-ops approach must produce identical scalar stats to the original apply."""
    proj, cum, diff, itype = "ProjectIdBSV", "Cum", "Diff", "impute_type"
    ref = _normalise(_gap_stats_apply(gap_stats_df, proj, cum, diff, itype), proj)
    sep = _normalise(_gap_stats_sep(gap_stats_df, proj, cum, diff, itype), proj)

    scalar_cols = ["diff_col_total", "cum_col_min_max_diff", "deviation",
                   "missing", "imputed", "imputed_na"]
    for col in scalar_cols:
        pd.testing.assert_series_equal(
            ref[col].astype(float),
            sep[col].astype(float),
            check_names=False,
            rtol=1e-5,
            obj=f"sep vs apply: {col}",
        )

