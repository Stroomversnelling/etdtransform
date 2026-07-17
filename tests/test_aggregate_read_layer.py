"""
TDD: legacy stage-2 aggregation reads through the shared read layer.

Design rule: a legacy function's computation and WRITE behaviour stay
unchanged, but its READS of mapped household data go through the shared
format-detecting read layer -- no function builds mapped file paths inline.
This keeps the whole legacy pipeline runnable from a sharded-only mapped
store, which is what makes a like-for-like comparison of the two pipelines
possible on real data.

Equivalence gate for the migration of aggregate_hh_data_duckdb: the same
fixture households read from a FLAT-only folder and from a SHARDED-only
folder must produce an identical household_default.parquet.

Batch safety (guards live at compute entry points): the stage-2 aggregation
assumes one file per household; a household spanning more than one HuisBatch
must raise HuisBatchOverlapError, never silently combine two batches into
the single output file.
"""

import shutil
from pathlib import Path

import pandas as pd
import pytest

import etdmap
import etdtransform
from etdmap.index_helpers import HuisBatchOverlapError
from etdtransform.aggregate import aggregate_hh_data_duckdb

TESTDATA_MAPPED = Path(etdtransform.options.mapped_folder_path)


def _copy_index_reviewed(dst: Path) -> None:
    """Copy the fixture index with Meenemen set True: the fixture ships in
    the first-mapping state (Meenemen empty), but these tests represent a
    REVIEWED dataset -- the aggregation's Meenemen gate must pass."""
    idx = pd.read_parquet(TESTDATA_MAPPED / "index.parquet",
                          dtype_backend="numpy_nullable")
    idx["Meenemen"] = pd.array([True] * len(idx), dtype="boolean")
    idx.to_parquet(dst / "index.parquet")


def _copy_flat(dst: Path) -> None:
    dst.mkdir(parents=True)
    _copy_index_reviewed(dst)
    for f in TESTDATA_MAPPED.glob("household_*_table.parquet"):
        shutil.copy2(f, dst / f.name)


def _copy_sharded(dst: Path) -> None:
    dst.mkdir(parents=True)
    _copy_index_reviewed(dst)
    shutil.copytree(TESTDATA_MAPPED / "sharded", dst / "sharded")


def _run_aggregate(mapped: Path, out: Path) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    saved = (
        etdmap.options.mapped_folder_path,
        etdtransform.options.mapped_folder_path,
        etdtransform.options.aggregate_folder_path,
    )
    try:
        etdmap.options.mapped_folder_path = mapped
        etdtransform.options.mapped_folder_path = mapped
        etdtransform.options.aggregate_folder_path = out
        aggregate_hh_data_duckdb()
    finally:
        (
            etdmap.options.mapped_folder_path,
            etdtransform.options.mapped_folder_path,
            etdtransform.options.aggregate_folder_path,
        ) = saved
    df = pd.read_parquet(out / "household_default.parquet", dtype_backend="numpy_nullable")
    return (
        df.sort_values(["HuisIdBSV", "ReadingDate"])
        .reset_index(drop=True)
        .reindex(sorted(df.columns), axis=1)
    )


class TestStage2ReadLayer:
    def test_flat_vs_sharded_equivalence(self, tmp_path):
        flat_dir = tmp_path / "flat_mapped"
        shard_dir = tmp_path / "sharded_mapped"
        _copy_flat(flat_dir)
        _copy_sharded(shard_dir)

        df_flat = _run_aggregate(flat_dir, tmp_path / "out_flat")
        df_shard = _run_aggregate(shard_dir, tmp_path / "out_shard")

        assert len(df_flat) > 0
        pd.testing.assert_frame_equal(df_flat, df_shard)

    def test_multi_batch_household_raises(self, tmp_path):
        shard_dir = tmp_path / "sharded_mapped"
        _copy_sharded(shard_dir)
        # Duplicate household 1's shard under a second batch id.
        src = next((shard_dir / "sharded" / "HuisIdBSV=1").glob("HuisBatchIdBSV=*"))
        dup = shard_dir / "sharded" / "HuisIdBSV=1" / "HuisBatchIdBSV=999"
        shutil.copytree(src, dup)

        with pytest.raises(HuisBatchOverlapError):
            _run_aggregate(shard_dir, tmp_path / "out")
