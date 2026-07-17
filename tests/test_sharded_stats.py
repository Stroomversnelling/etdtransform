"""
Wide-stage stats (etdtransform.data_stats.get_data_stats) work on both
storage layouts: a monolith file and a hive shard directory of the SAME data
must produce identical stats rows (HuisBatchIdBSV is excluded from stats by
default, like the other id columns, so the output schema matches across
formats -- cross-stage drift comparisons stay well-defined).
"""

from pathlib import Path

import pandas as pd

import etdtransform
from etdtransform.data_stats import get_data_stats

TESTDATA_AGG = Path(etdtransform.options.aggregate_folder_path)


def _split_monolith_to_shards(monolith_path: Path, shard_root: Path) -> None:
    df = pd.read_parquet(monolith_path, dtype_backend="numpy_nullable")
    for hid, g in df.groupby("HuisIdBSV", sort=True):
        d = shard_root / f"HuisIdBSV={int(hid)}" / f"HuisBatchIdBSV={int(hid)}"
        d.mkdir(parents=True)
        g.drop(columns=["HuisIdBSV"]).to_parquet(d / "part.parquet", index=False)


def test_variance_overflow_reports_na_not_crash(tmp_path, caplog):
    """A column whose values are large enough that variance overflows float64
    (found on ZelfgebruikPercentage: a ratio with near-zero denominators) must
    not kill the whole stats report: that column's stats become NA with a
    factual warning, counts are kept, and every other column reports normally
    (ADR-022: NA = 'we do not know', stated visibly)."""
    dates = pd.date_range("2024-01-01", periods=10, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "HuisIdBSV": pd.array([1] * 10, dtype="Int64"),
        "ReadingDate": dates,
        "Normal": pd.array([float(i) for i in range(10)], dtype="Float64"),
        "HugeRatio": pd.array([1e200, 2e200] * 5, dtype="Float64"),
    })
    p = tmp_path / "stage.parquet"
    df.to_parquet(p)

    stats = get_data_stats(str(p), precision="exact")

    assert {"Normal", "HugeRatio"}.issubset(set(stats["column"]))
    normal = stats[stats["column"] == "Normal"].iloc[0]
    assert float(normal["std"]) > 0
    huge = stats[stats["column"] == "HugeRatio"].iloc[0]
    assert int(huge["count"]) == 10  # counts survive
    assert pd.isna(huge["std"]) and pd.isna(huge["mean"])
    warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any("HugeRatio" in w for w in warnings)


def test_monolith_vs_sharded_stats_equivalence(tmp_path):
    source = TESTDATA_AGG / "household_imputed.parquet"
    shard_src = tmp_path / "shards"
    _split_monolith_to_shards(source, shard_src)

    # precision="exact": approximate quantiles (the default) use DuckDB
    # sketches whose result depends on scan order, which differs between a
    # shard glob and a monolith (observed 0.007 on a median). Exact quantiles
    # over the identical multiset are order-independent.
    a = get_data_stats(str(source), precision="exact")
    b = get_data_stats(str(shard_src), precision="exact")

    key = [c for c in ["column", "season"] if c in a.columns]
    a = a.sort_values(key).reset_index(drop=True)
    b = b.sort_values(key).reset_index(drop=True)
    assert set(a["column"]) == set(b["column"]), (
        sorted(set(a["column"]) ^ set(b["column"])))
    assert len(a) == len(b)
    for col in a.columns:
        av, bv = a[col], b[col]
        assert av.isna().equals(bv.isna()), f"null mask differs: {col}"
        try:
            # 1e-6, not exact: aggregating a shard glob sums in a different
            # order than one monolith file, so means/stds differ in the last
            # float bits (observed 2e-9). The underlying DATA is identical --
            # the per-stage value comparisons prove that exactly.
            diff = (av.astype("float64") - bv.astype("float64")).abs().fillna(0)
            assert diff.max() <= 1e-6, f"values differ: {col} (max {diff.max()})"
        except (ValueError, TypeError):
            assert av.astype("string").equals(bv.astype("string")), (
                f"values differ: {col}")
