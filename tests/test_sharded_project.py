"""
Stage 6 (project aggregation) works on both storage layouts.

Project-level outputs STAY single parquet files (design decision: they are
small); only the READ side is format-detected -- when
``household_<interval>`` is a hive shard directory it is read with
union_by_name, otherwise the legacy ``household_<interval>.parquet`` file is
read. The SQL math is frozen.

Equivalence gate: project aggregation over the SAME resampled data stored as
a monolith and as hive shards must produce identical project files.
"""

import shutil
from pathlib import Path

import pandas as pd
import pytest

import etdtransform
from etdtransform.aggregate import (
    aggregate_project_data_duckdb,
    resample_hh_data_duckdb,
)

pytestmark = [pytest.mark.integration, pytest.mark.pipeline]





class TestProjectAggregationCrossFormat:
    def test_monolith_vs_sharded_equivalence(self, tmp_path, aggregate_artifacts,
                                             calculated_shards):
        source = aggregate_artifacts / "household_calculated.parquet"

        flat_dir = tmp_path / "flat"
        flat_dir.mkdir()
        resample_hh_data_duckdb(str(source), str(flat_dir), intervals=("60min",))
        aggregate_project_data_duckdb(
            intervals=("60min",), aggregate_folder_path=str(flat_dir)
        )

        shard_src = calculated_shards
        shard_dir = tmp_path / "shard"
        shard_dir.mkdir()
        resample_hh_data_duckdb(
            str(shard_src), str(shard_dir), intervals=("60min",),
            partition_output=True,
        )
        aggregate_project_data_duckdb(
            intervals=("60min",), aggregate_folder_path=str(shard_dir)
        )

        a = pd.read_parquet(flat_dir / "project_60min.parquet",
                            dtype_backend="numpy_nullable")
        b = pd.read_parquet(shard_dir / "project_60min.parquet",
                            dtype_backend="numpy_nullable")
        assert set(a.columns) == set(b.columns), sorted(set(a.columns) ^ set(b.columns))
        key = ["ProjectIdBSV", "ReadingDate"]
        a = a[sorted(a.columns)].sort_values(key).reset_index(drop=True)
        b = b[sorted(b.columns)].sort_values(key).reset_index(drop=True)
        assert len(a) == len(b)
        for col in a.columns:
            av, bv = a[col], b[col]
            assert av.isna().equals(bv.isna()), f"null mask differs: {col}"
            try:
                diff = (av.astype("float64") - bv.astype("float64")).abs().fillna(0)
                assert diff.max() <= 1e-9, f"values differ: {col} (max {diff.max()})"
            except (ValueError, TypeError):
                assert av.astype("string").equals(bv.astype("string")), (
                    f"values differ: {col}")

    def test_duplicate_reading_dates_in_sharded_source_raise(self, tmp_path,
                                                            fresh_shards):
        # Copies a batch directory in, so it needs a tree of its own.
        shard_src = fresh_shards("household_calculated.parquet")
        shard_dir = tmp_path / "shard"
        shard_dir.mkdir()
        resample_hh_data_duckdb(
            str(shard_src), str(shard_dir), intervals=("60min",),
            partition_output=True,
        )
        # Duplicate one household's resampled shard under a second batch id.
        hh_dir = next((shard_dir / "household_60min").glob("HuisIdBSV=*"))
        src_batch = next(hh_dir.glob("HuisBatchIdBSV=*"))
        shutil.copytree(src_batch, hh_dir / "HuisBatchIdBSV=999")

        with pytest.raises(ValueError):
            aggregate_project_data_duckdb(
                intervals=("60min",), aggregate_folder_path=str(shard_dir)
            )
