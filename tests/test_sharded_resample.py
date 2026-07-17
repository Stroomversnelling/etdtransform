"""
Stage 5 (resampling) works on both storage layouts.

Same contract as stage 4 (test_sharded_calc): reads are format-detected (a
directory source is read as hive shards with union_by_name), the math is the
frozen legacy SQL, and sharded OUTPUT is an explicit opt-in
(partition_output=True) writing household_<interval>/ hive shard directories
instead of household_<interval>.parquet files.

Equivalence gate: the SAME fixture calculated data run as a monolith and as
hive shards must produce identical resampled values per interval.
"""

import glob as glob_mod
import shutil
from pathlib import Path

import pandas as pd
import pytest

import etdtransform
from etdtransform.aggregate import resample_hh_data_duckdb

TESTDATA_AGG = Path(etdtransform.options.aggregate_folder_path)
INTERVALS = ("5min", "60min")


def _split_monolith_to_shards(monolith_path: Path, shard_root: Path) -> None:
    df = pd.read_parquet(monolith_path, dtype_backend="numpy_nullable")
    for hid, g in df.groupby("HuisIdBSV", sort=True):
        d = shard_root / f"HuisIdBSV={int(hid)}" / f"HuisBatchIdBSV={int(hid)}"
        d.mkdir(parents=True)
        g.drop(columns=["HuisIdBSV"]).to_parquet(d / "part.parquet", index=False)


def _read_shards(shard_root: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(glob_mod.glob(str(shard_root / "**" / "*.parquet"), recursive=True)):
        hid = int(Path(p).parent.parent.name.split("=")[1])
        g = pd.read_parquet(p, dtype_backend="numpy_nullable")
        g.insert(0, "HuisIdBSV", pd.array([hid] * len(g), dtype="Int64"))
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def _assert_frames_equivalent(a: pd.DataFrame, b: pd.DataFrame) -> None:
    b = b.drop(columns=[c for c in ["HuisBatchIdBSV"] if c in b.columns])
    assert set(a.columns) == set(b.columns), sorted(set(a.columns) ^ set(b.columns))
    common = sorted(a.columns)
    key = ["HuisIdBSV", "ReadingDate"]
    a = a[common].sort_values(key).reset_index(drop=True)
    b = b[common].sort_values(key).reset_index(drop=True)
    assert len(a) == len(b)
    for col in common:
        av, bv = a[col], b[col]
        assert av.isna().equals(bv.isna()), f"null mask differs: {col}"
        try:
            diff = (av.astype("float64") - bv.astype("float64")).abs().fillna(0)
            assert diff.max() <= 1e-9, f"values differ: {col} (max {diff.max()})"
        except (ValueError, TypeError):
            assert av.astype("string").equals(bv.astype("string")), f"values differ: {col}"


class TestResampleCrossFormat:
    def test_monolith_vs_sharded_equivalence(self, tmp_path):
        source = TESTDATA_AGG / "household_calculated.parquet"
        shard_src = tmp_path / "calculated_shards"
        _split_monolith_to_shards(source, shard_src)

        out_flat = tmp_path / "out_flat"
        out_flat.mkdir()
        resample_hh_data_duckdb(str(source), str(out_flat), intervals=INTERVALS)

        out_shard = tmp_path / "out_shard"
        out_shard.mkdir()
        resample_hh_data_duckdb(
            str(shard_src), str(out_shard), intervals=INTERVALS,
            partition_output=True,
        )

        for interval in INTERVALS:
            a = pd.read_parquet(out_flat / f"household_{interval}.parquet",
                                dtype_backend="numpy_nullable")
            b = _read_shards(out_shard / f"household_{interval}")
            _assert_frames_equivalent(a, b)

    def test_rerun_overwrites_existing_shard_output(self, tmp_path):
        """Regression: the writer clears its existing output directory on a
        re-run. This cleanup only executes when output already exists -- a
        path that tests with freshly created directories never reach."""
        source = TESTDATA_AGG / "household_calculated.parquet"
        shard_src = tmp_path / "calculated_shards"
        _split_monolith_to_shards(source, shard_src)
        out = tmp_path / "out"
        out.mkdir()
        resample_hh_data_duckdb(str(shard_src), str(out), intervals=("60min",),
                                partition_output=True)
        first = _read_shards(out / "household_60min")
        resample_hh_data_duckdb(str(shard_src), str(out), intervals=("60min",),
                                partition_output=True)  # must not raise
        second = _read_shards(out / "household_60min")
        assert len(first) == len(second)

    def test_duplicate_reading_dates_in_sharded_source_raise(self, tmp_path):
        source = TESTDATA_AGG / "household_calculated.parquet"
        shard_src = tmp_path / "shards"
        _split_monolith_to_shards(source, shard_src)
        src1 = next((shard_src / "HuisIdBSV=1").glob("HuisBatchIdBSV=*"))
        shutil.copytree(src1, shard_src / "HuisIdBSV=1" / "HuisBatchIdBSV=999")

        with pytest.raises(ValueError):
            resample_hh_data_duckdb(
                str(shard_src), str(tmp_path / "out"), intervals=("60min",),
                partition_output=True,
            )

    def test_partition_output_requires_batch_column(self, tmp_path):
        source = TESTDATA_AGG / "household_calculated.parquet"
        with pytest.raises(ValueError, match="HuisBatchIdBSV"):
            resample_hh_data_duckdb(
                str(source), str(tmp_path / "out"), intervals=("60min",),
                partition_output=True,
            )
