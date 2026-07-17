"""
Contract for the sharded read layer.

Built per a measured comparison of shard read strategies:

- household-batch materialisation = DIRECT per-shard file reads (pandas,
  nullable dtypes) -- `read_mapped_households`;
- whole-dataset reads = DuckDB/Ibis hive-glob with union_by_name --
  `mapped_household_table`;
- polars not adopted.

Source transparency (ADR D): `read_mapped_households` works on BOTH layouts --
flat legacy files and hive shards -- via `detect_mapped_format`; callers never
pass a format. `mapped_household_table` is the sharded-forward whole-table view;
on a flat-only folder it raises with a pointer to the legacy stage-2 monolith
(which IS the legacy whole-table view).

Tests build their own tiny layouts in tmp dirs (no dependency on the etdmap
fixture or on etdmap itself). NONE of the three functions exists yet -> red.
"""

import os
from pathlib import Path

import pandas as pd
import pytest

from etdtransform.load_data import (
    detect_mapped_format,
    mapped_household_files,
    mapped_household_table,
    read_mapped_households,
)


def _hh_df(hid: int, n: int = 6) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({
        "ReadingDate": dates,
        "ElektriciteitNetgebruik": pd.array([float(hid) * 10 + i for i in range(n)], dtype="Float64"),
        "Gasgebruik": pd.array([pd.NA] * n, dtype="Float64"),
    })


def _write_flat(root: Path, huis_ids):
    root.mkdir(parents=True, exist_ok=True)
    for h in huis_ids:
        _hh_df(h).to_parquet(root / f"household_{h}_table.parquet")


def _write_sharded(root: Path, huis_ids):
    for h in huis_ids:
        d = root / "sharded" / f"HuisIdBSV={h}" / f"HuisBatchIdBSV={h}"
        d.mkdir(parents=True, exist_ok=True)
        _hh_df(h).to_parquet(d / "part.parquet")


class TestDetectMappedFormat:
    def test_sharded_wins_when_present(self, tmp_path):
        # Coexistence: flat legacy baseline + sharded -- sharded is the format
        # going forward, so it is preferred on detection.
        _write_flat(tmp_path, [1])
        _write_sharded(tmp_path, [1])
        assert detect_mapped_format(tmp_path) == "sharded"

    def test_flat_only(self, tmp_path):
        _write_flat(tmp_path, [1])
        assert detect_mapped_format(tmp_path) == "flat"

    def test_neither_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            detect_mapped_format(tmp_path)


class TestReadStageFrame:
    """Generic pandas reader for any stage artifact: file or shard folder."""

    def test_dir_and_file_reads(self, tmp_path):
        from etdtransform.load_data import read_source_frame
        _write_sharded(tmp_path, [1, 2])
        df = read_source_frame(tmp_path / "sharded")
        assert set(int(x) for x in df["HuisIdBSV"].unique()) == {1, 2}
        assert str(df["HuisIdBSV"].dtype) == "Int64"
        assert len(df) == 12
        # column selection with injected ids
        sub = read_source_frame(tmp_path / "sharded",
                               columns=["HuisIdBSV", "ReadingDate"])
        assert list(sub.columns) == ["HuisIdBSV", "ReadingDate"]
        # single-file passthrough
        f = tmp_path / "one.parquet"
        _hh_df(1).to_parquet(f)
        assert len(read_source_frame(f)) == 6


class TestMappedHouseholdFiles:
    """Shared read-layer resolver: no function builds mapped file paths
    inline. Returns [(path, HuisBatchIdBSV), ...]. It does NOT raise on a
    multi-batch household -- loaders return what exists; one-batch-per-
    household assumptions are guarded where the computation starts."""

    def test_sharded_single_batch(self, tmp_path):
        _write_sharded(tmp_path, [1, 2])
        files = mapped_household_files(1, mapped_folder_path=tmp_path)
        assert len(files) == 1
        path, hbid = files[0]
        assert hbid == 1
        assert str(path).endswith("part.parquet")
        assert "HuisIdBSV=1" in str(path)

    def test_flat_single_file(self, tmp_path):
        _write_flat(tmp_path, [1])
        files = mapped_household_files(1, mapped_folder_path=tmp_path)
        assert len(files) == 1
        path, hbid = files[0]
        assert hbid == 1  # flat predates batches: 1:1 by definition
        assert str(path).endswith("household_1_table.parquet")

    def test_sharded_multi_batch_returns_both(self, tmp_path):
        for hbid in (1, 9):
            d = tmp_path / "sharded" / "HuisIdBSV=1" / f"HuisBatchIdBSV={hbid}"
            d.mkdir(parents=True)
            _hh_df(1).to_parquet(d / "part.parquet")
        files = mapped_household_files(1, mapped_folder_path=tmp_path)
        assert [hbid for _, hbid in files] == [1, 9]  # sorted by batch id

    def test_missing_household_raises(self, tmp_path):
        _write_sharded(tmp_path, [1])
        with pytest.raises(FileNotFoundError) as exc:
            mapped_household_files(99, mapped_folder_path=tmp_path)
        assert "99" in str(exc.value)

    def test_unparseable_shard_path_raises(self, tmp_path, monkeypatch):
        _write_sharded(tmp_path, [1])
        import etdmap.storage as storage_mod  # the read layer's home
        monkeypatch.setattr(storage_mod, "_SHARD_PART_RE", __import__("re").compile(r"WILL_NOT_MATCH"))
        with pytest.raises(ValueError):
            mapped_household_files(1, mapped_folder_path=tmp_path)


class TestReadMappedHouseholds:
    def test_sharded_batch_read(self, tmp_path):
        _write_sharded(tmp_path, [1, 2, 3])
        df = read_mapped_households([1, 2], mapped_folder_path=tmp_path)
        assert set(df["HuisIdBSV"].unique()) == {1, 2}
        assert str(df["HuisIdBSV"].dtype) == "Int64"
        assert str(df["HuisBatchIdBSV"].dtype) == "Int64"
        assert (df["HuisBatchIdBSV"] == df["HuisIdBSV"]).all()
        # nullable dtypes (ADR-005)
        assert str(df["ElektriciteitNetgebruik"].dtype) == "Float64"
        assert len(df) == 12  # 2 households x 6 rows

    def test_flat_batch_read_equivalent(self, tmp_path):
        # Source transparency: same call, flat layout, same logical result
        # (ids injected; flat carries no id columns, like the shards).
        flat_root = tmp_path / "flat"
        shard_root = tmp_path / "shard"
        _write_flat(flat_root, [1, 2])
        _write_sharded(shard_root, [1, 2])
        df_flat = read_mapped_households([1, 2], mapped_folder_path=flat_root)
        df_shard = read_mapped_households([1, 2], mapped_folder_path=shard_root)
        pd.testing.assert_frame_equal(
            df_flat.sort_values(["HuisIdBSV", "ReadingDate"]).reset_index(drop=True),
            df_shard.sort_values(["HuisIdBSV", "ReadingDate"]).reset_index(drop=True),
        )

    def test_column_projection(self, tmp_path):
        _write_sharded(tmp_path, [1])
        df = read_mapped_households([1], columns=["ReadingDate", "Gasgebruik"],
                                    mapped_folder_path=tmp_path)
        assert set(df.columns) == {"HuisIdBSV", "HuisBatchIdBSV", "ReadingDate", "Gasgebruik"}

    def test_two_batches_yield_distinct_batch_ids(self, tmp_path):
        """Regression: HuisBatchIdBSV must come from the shard PATH per file.
        A silent fall-back to HuisBatchIdBSV=HuisIdBSV masks multi-batch data
        and blinds the imputation batch guard (a path pattern that matched
        only '/' once made every Windows path fall back that way)."""
        for hbid in (1, 9):
            d = tmp_path / "sharded" / "HuisIdBSV=1" / f"HuisBatchIdBSV={hbid}"
            d.mkdir(parents=True)
            _hh_df(1).to_parquet(d / "part.parquet")
        df = read_mapped_households([1], mapped_folder_path=tmp_path)
        assert set(int(x) for x in df["HuisBatchIdBSV"].unique()) == {1, 9}
        assert len(df) == 12

    def test_unparseable_shard_path_raises_not_defaults(self, tmp_path, monkeypatch):
        """In SHARDED mode a shard path that fails to parse must RAISE, never
        silently default HuisBatchIdBSV to the household id: a silent
        fallback masks multi-batch data and blinds the imputation batch
        guard."""
        d = tmp_path / "sharded" / "HuisIdBSV=1" / "HuisBatchIdBSV=1"
        d.mkdir(parents=True)
        _hh_df(1).to_parquet(d / "part.parquet")
        import etdmap.storage as storage_mod  # the read layer's home
        # Simulate a parse failure (e.g. an unexpected future layout change).
        monkeypatch.setattr(storage_mod, "_SHARD_PART_RE", __import__("re").compile(r"WILL_NOT_MATCH"))
        with pytest.raises(ValueError):
            read_mapped_households([1], mapped_folder_path=tmp_path)

    def test_missing_household_raises(self, tmp_path):
        _write_sharded(tmp_path, [1])
        with pytest.raises(FileNotFoundError) as exc:
            read_mapped_households([1, 99], mapped_folder_path=tmp_path)
        assert "99" in str(exc.value)


class TestMappedHouseholdTable:
    def test_sharded_lazy_table(self, tmp_path):
        _write_sharded(tmp_path, [1, 2, 3])
        tbl = mapped_household_table(mapped_folder_path=tmp_path)
        assert int(tbl.count().execute()) == 18  # 3 x 6 rows
        cols = set(tbl.columns)
        assert {"HuisIdBSV", "HuisBatchIdBSV", "ReadingDate"}.issubset(cols)
        # engine-side filter on the hive partition column
        n1 = int(tbl.filter(tbl.HuisIdBSV == 1).count().execute())
        assert n1 == 6

    def test_flat_only_raises_with_pointer_to_legacy(self, tmp_path):
        _write_flat(tmp_path, [1])
        with pytest.raises(ValueError) as exc:
            mapped_household_table(mapped_folder_path=tmp_path)
        assert "household_default" in str(exc.value)
