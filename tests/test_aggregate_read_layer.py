"""
Stage-2 aggregation reads through the shared format-detecting read layer.

The single-file aggregation's computation and write behaviour are unchanged;
its reads of mapped household data go through the shared read layer, so it
runs from either storage layout.

Equivalence contract for aggregate_hh_data_duckdb: the same fixture
households read from a FLAT-only folder and from a SHARDED-only folder must
produce an identical household_default.parquet.

Batch safety (guards live at compute entry points): the stage-2 aggregation
assumes one file per household; a household spanning more than one HuisBatch
must raise HouseBatchOverlapError, never silently combine two batches into
the single output file.
"""

import shutil
from pathlib import Path

import pandas as pd
import pytest

import etdmap
import etdtransform
from etdmap.index_helpers import HouseBatchOverlapError
from etdtransform.aggregate import aggregate_hh_data_duckdb

pytestmark = pytest.mark.integration

def _copy_index_reviewed(mapped: Path, dst: Path) -> None:
    """Copy the fixture index with Meenemen set: the fixture ships in the
    first-mapping state (Meenemen empty), but these tests represent a
    REVIEWED dataset -- the aggregation's Meenemen gate must pass. A reviewed
    inclusion also declares its grid, so the fixture's undeclared-grid batch
    stays excluded rather than acquiring a frequency it never declared."""
    idx = pd.read_parquet(mapped / "index.parquet",
                          dtype_backend="numpy_nullable")
    idx["Meenemen"] = idx["Gegevensfrequentie"].notna().astype("boolean")
    idx.to_parquet(dst / "index.parquet")


def _copy_flat(mapped: Path, dst: Path) -> None:
    dst.mkdir(parents=True)
    _copy_index_reviewed(mapped, dst)
    for f in mapped.glob("household_*_table.parquet"):
        shutil.copy2(f, dst / f.name)


def _copy_sharded(mapped: Path, dst: Path) -> None:
    """Build the sharded layout FROM the flat fixture files.

    Derived rather than copied from a checked-in sharded directory, so the two
    layouts hold the same data by construction. A stored copy silently goes
    stale whenever mapping changes which columns it writes, which turns this
    equivalence test into a comparison of two different datasets.
    """
    dst.mkdir(parents=True)
    _copy_index_reviewed(mapped, dst)
    for f in sorted(mapped.glob("household_*_table.parquet")):
        household_id = int(f.stem.split("_")[1])
        shard_dir = (
            dst / "sharded" / f"HuisIdBSV={household_id}"
            / f"HuisBatchIdBSV={household_id}"
        )
        shard_dir.mkdir(parents=True)
        shutil.copy2(f, shard_dir / "part.parquet")


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
    def test_flat_vs_sharded_equivalence(self, tmp_path, mapped_dataset):
        flat_dir = tmp_path / "flat_mapped"
        shard_dir = tmp_path / "sharded_mapped"
        _copy_flat(mapped_dataset, flat_dir)
        _copy_sharded(mapped_dataset, shard_dir)

        df_flat = _run_aggregate(flat_dir, tmp_path / "out_flat")
        df_shard = _run_aggregate(shard_dir, tmp_path / "out_shard")

        assert len(df_flat) > 0
        pd.testing.assert_frame_equal(df_flat, df_shard)

    def test_multi_batch_household_raises(self, tmp_path, mapped_dataset):
        """A household with two INCLUDED batches must stop the run: the index
        is what includes a batch, so the overlap is declared there, and no
        stage may silently blend two batches of one household."""
        shard_dir = tmp_path / "sharded_mapped"
        _copy_sharded(mapped_dataset, shard_dir)
        src = next((shard_dir / "sharded" / "HuisIdBSV=2").glob("HuisBatchIdBSV=*"))
        dup = shard_dir / "sharded" / "HuisIdBSV=2" / "HuisBatchIdBSV=999"
        shutil.copytree(src, dup)
        idx = pd.read_parquet(
            shard_dir / "index.parquet", dtype_backend="numpy_nullable"
        )
        second = idx[idx["HuisIdBSV"] == 2].copy()
        second["HuisBatchIdBSV"] = pd.array([999], dtype="Int64")
        pd.concat([idx, second], ignore_index=True).to_parquet(
            shard_dir / "index.parquet"
        )

        with pytest.raises(HouseBatchOverlapError):
            _run_aggregate(shard_dir, tmp_path / "out")

    def test_a_disk_batch_the_index_does_not_include_stops_the_run(
        self, tmp_path, mapped_dataset
    ):
        """A second batch directory on disk for an included household, with
        no index row behind it, is a registry/disk mismatch. The run stops
        loudly rather than picking a directory or blending the two -- which
        of them is right is a reconciliation question, not the reader's."""
        shard_dir = tmp_path / "sharded_mapped"
        _copy_sharded(mapped_dataset, shard_dir)
        src = next((shard_dir / "sharded" / "HuisIdBSV=2").glob("HuisBatchIdBSV=*"))
        dup = shard_dir / "sharded" / "HuisIdBSV=2" / "HuisBatchIdBSV=999"
        shutil.copytree(src, dup)

        with pytest.raises(HouseBatchOverlapError):
            _run_aggregate(shard_dir, tmp_path / "out")
