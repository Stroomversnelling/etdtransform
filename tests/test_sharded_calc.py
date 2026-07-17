"""
Stage 4 (calculated columns) works on both storage layouts.

Design rule: a legacy function's READS go through format detection (a
directory source is read as shards, with per-supplier schemas unified by
column name), while its computation and default WRITE behaviour stay
unchanged. Sharded OUTPUT is an explicit opt-in (partition_output=True) that
writes a shard folder tree (HuisIdBSV=<n>/HuisBatchIdBSV=<p>/) instead of
one file -- the math is identical, only the reading and writing differ.

Equivalence gate: the SAME fixture imputed data run as a monolith and as hive
shards must produce identical calculated values.

Batch safety: with a directory source the function must refuse duplicate
(HuisIdBSV, ReadingDate) rows (overlapping batches) at entry.
"""

import glob as glob_mod
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

import etdtransform
from etdtransform.aggregate import add_calculated_columns_to_hh_data_ibis

TESTDATA_AGG = Path(etdtransform.options.aggregate_folder_path)
FILLNA_VARS = ["ElektriciteitsgebruikBoilervatDiff", "ElektriciteitsgebruikWTWDiff"]


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


class TestCalcCrossFormat:
    def test_monolith_vs_sharded_equivalence(self, tmp_path):
        source = TESTDATA_AGG / "household_imputed.parquet"
        shard_src = tmp_path / "household_imputed_shards"
        _split_monolith_to_shards(source, shard_src)

        out_file = tmp_path / "household_calculated.parquet"
        add_calculated_columns_to_hh_data_ibis(
            source_path=str(source), output_path=str(out_file),
            fillna_vars=FILLNA_VARS,
        )
        out_dir = tmp_path / "household_calculated_shards"
        add_calculated_columns_to_hh_data_ibis(
            source_path=str(shard_src), output_path=str(out_dir),
            fillna_vars=FILLNA_VARS, partition_output=True,
        )

        a = pd.read_parquet(out_file, dtype_backend="numpy_nullable")
        b = _read_shards(out_dir)
        assert "HuisBatchIdBSV" not in a.columns
        b = b.drop(columns=[c for c in ["HuisBatchIdBSV"] if c in b.columns])

        common = sorted(set(a.columns) & set(b.columns))
        assert set(a.columns) == set(b.columns), (
            sorted(set(a.columns) ^ set(b.columns)))
        key = ["HuisIdBSV", "ReadingDate"]
        a = a[common].sort_values(key).reset_index(drop=True)
        b = b[common].sort_values(key).reset_index(drop=True)
        for col in common:
            av, bv = a[col], b[col]
            assert av.isna().equals(bv.isna()), f"null mask differs: {col}"
            try:
                af = av.astype("float64")
                bf = bv.astype("float64")
                assert (af - bf).abs().fillna(0).max() <= 1e-9, f"values differ: {col}"
            except (ValueError, TypeError):
                assert av.astype("string").equals(bv.astype("string")), (
                    f"values differ: {col}")

    def test_duplicate_reading_dates_in_sharded_source_raise(self, tmp_path):
        source = TESTDATA_AGG / "household_imputed.parquet"
        shard_src = tmp_path / "shards"
        _split_monolith_to_shards(source, shard_src)
        # Duplicate household 1 under a second batch: overlapping periods.
        src1 = next((shard_src / "HuisIdBSV=1").glob("HuisBatchIdBSV=*"))
        shutil.copytree(src1, shard_src / "HuisIdBSV=1" / "HuisBatchIdBSV=999")

        with pytest.raises(ValueError):
            add_calculated_columns_to_hh_data_ibis(
                source_path=str(shard_src),
                output_path=str(tmp_path / "out"),
                fillna_vars=FILLNA_VARS, partition_output=True,
            )

    def test_partition_output_requires_batch_column(self, tmp_path):
        # Opting into hive output from a monolith source (no HuisBatchIdBSV)
        # is a caller error and must fail before writing anything.
        source = TESTDATA_AGG / "household_imputed.parquet"
        with pytest.raises(ValueError, match="HuisBatchIdBSV"):
            add_calculated_columns_to_hh_data_ibis(
                source_path=str(source),
                output_path=str(tmp_path / "out"),
                fillna_vars=FILLNA_VARS, partition_output=True,
            )
