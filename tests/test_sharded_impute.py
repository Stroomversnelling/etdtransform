"""
The FULL sharded imputation chain + the batch-safety guards.

Chain parity: the sharded path must produce ALL the imputation artifacts the
legacy chain produces -- avg_diffs + max bounds (estimation inputs), the imputed
data itself, gap stats, household/project summaries, and the post-imputation
household_aggregated_diff -- so the planned imputation rewrite can be validated
on either path against promoted production. Math is never duplicated: the
sharded diffs function feeds the SAME prepare_diffs_for_impute_ibis; the
sharded impute function calls the SAME _impute_chunk_core; summaries come from
a shared finalizer extracted from the legacy shell.

Batch-safety guards (computational, independent of the registry coexistence
guard, which deliberately relaxes post-switch): the averaging/imputation chain
must refuse
  (a) a household spanning MULTIPLE batches -- concatenating batches would let
      the large-gap project-average method impute across the BETWEEN-batch gap
      (real non-delivery; the cross-batch continuity invariant), and
  (b) duplicate (HuisIdBSV, ReadingDate) -- overlapping batch periods double-
      weight project averages and break the sort/diff math.
Batch-aware imputation is native-resolution-phase work; until then: loud stop.
"""

import shutil
from pathlib import Path

import pandas as pd
import pytest

import etdtransform
from etdmap.index_helpers import HuisBatchOverlapError
from etdtransform.aggregate import impute_mapped_households_sharded
from etdtransform.impute import prepare_diffs_sharded
from etdtransform.load_data import included_household_ids


def _write_batch_index(folder: Path, rows):
    """rows: list of (HuisIdBSV, HuisBatchIdBSV, ProjectIdBSV, Meenemen)."""
    df = pd.DataFrame({
        "HuisIdBSV": pd.array([r[0] for r in rows], dtype="Int64"),
        "HuisBatchIdBSV": pd.array([r[1] for r in rows], dtype="Int64"),
        "BatchIdBSV": pd.array([1] * len(rows), dtype="Int64"),
        "ProjectIdBSV": pd.array([r[2] for r in rows], dtype="Int64"),
        "Meenemen": pd.array([r[3] for r in rows], dtype="boolean"),
        "Gegevensfrequentie": pd.array(["5-minute"] * len(rows), dtype="string"),
    })
    folder.mkdir(parents=True, exist_ok=True)
    df.to_parquet(folder / "batch_index.parquet")


class TestExplicitOutputFolderRequired:
    """The sharded stage-3 WRITERS must never default their output folder.

    The config default (etdtransform.options.aggregate_folder_path) points at
    the promoted production area, and impute_mapped_households_sharded even
    clears its output directory before writing -- a bare call must be a loud
    error, never a silent production write. Callers pass the output folder
    explicitly.
    """

    def test_prepare_diffs_sharded_requires_aggregate_folder(self, tmp_path):
        with pytest.raises(ValueError, match="aggregate_folder_path"):
            prepare_diffs_sharded(
                mapped_folder_path=tmp_path,
                aggregate_folder_path=None,
                cumulative_columns=[],
            )

    def test_impute_sharded_requires_aggregate_folder(self, tmp_path):
        with pytest.raises(ValueError, match="aggregate_folder_path"):
            impute_mapped_households_sharded(
                mapped_folder_path=tmp_path,
                aggregate_folder_path=None,
                cum_cols=[],
            )


class TestIncludedHouseholdIds:
    def test_meenemen_filter(self, tmp_path):
        _write_batch_index(tmp_path, [(1, 1, 1, True), (2, 2, 1, False), (3, 3, 2, True)])
        assert included_household_ids(mapped_folder_path=tmp_path) == [1, 3]

    def test_missing_batch_index_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            included_household_ids(mapped_folder_path=tmp_path)


@pytest.fixture(scope="module")
def sharded_fixture_mapped(tmp_path_factory, _meenemen_updated):
    """Sharded mapped copy of the CURRENT flat test fixture (10 HH), with the
    fixture's batch_index (Meenemen-populated -- depends on _meenemen_updated,
    which refreshes the registry). Built fresh so it cannot drift."""
    src = Path(etdtransform.options.mapped_folder_path)
    dst = tmp_path_factory.mktemp("sharded_mapped")
    n = 0
    for f in sorted(src.glob("household_*_table.parquet")):
        hid = int(f.stem.split("_")[1])
        d = dst / "sharded" / f"HuisIdBSV={hid}" / f"HuisBatchIdBSV={hid}"
        d.mkdir(parents=True)
        shutil.copyfile(f, d / "part.parquet")
        n += 1
    assert n == 10, f"expected the 10-HH fixture, found {n}"
    shutil.copyfile(src / "batch_index.parquet", dst / "batch_index.parquet")
    return dst


@pytest.fixture(scope="module")
def sharded_stage3(sharded_fixture_mapped, _cum_cols, tmp_path_factory):
    """Run the FULL sharded stage-3 chain once: diffs -> impute (+ summaries)."""
    agg = tmp_path_factory.mktemp("sharded_agg")
    prepare_diffs_sharded(
        mapped_folder_path=sharded_fixture_mapped,
        aggregate_folder_path=agg,
        cumulative_columns=_cum_cols,
    )
    impute_mapped_households_sharded(
        mapped_folder_path=sharded_fixture_mapped,
        aggregate_folder_path=agg,
        cum_cols=_cum_cols,
    )
    return agg / "sharded"


def _read_parquet(p):
    return pd.read_parquet(p, dtype_backend="numpy_nullable")


class TestShardedChainEquivalence:
    """Every artifact of the sharded chain equals its legacy counterpart on the
    fixture (same data, same math, different I/O shell)."""

    def test_avg_diffs_equal(self, sharded_stage3, ibis_pipeline):
        sharded = _read_parquet(sharded_stage3 / "avg_diffs.parquet")
        legacy = _read_parquet(ibis_pipeline / "avg_diffs.parquet")
        key = ["ProjectIdBSV", "ReadingDate"]
        pd.testing.assert_frame_equal(
            sharded.sort_values(key).reset_index(drop=True)[legacy.columns],
            legacy.sort_values(key).reset_index(drop=True),
        )

    def test_max_bounds_equal(self, sharded_stage3, ibis_pipeline):
        sharded = _read_parquet(sharded_stage3 / "household_diff_max_bounds.parquet")
        legacy = _read_parquet(ibis_pipeline / "household_diff_max_bounds.parquet")
        cols = list(legacy.columns)
        sort_by = [c for c in ("ProjectIdBSV", "column") if c in cols] or cols[:1]
        pd.testing.assert_frame_equal(
            sharded.sort_values(sort_by).reset_index(drop=True)[cols],
            legacy.sort_values(sort_by).reset_index(drop=True),
        )

    def test_imputed_values_equal(self, sharded_stage3, ibis_pipeline):
        parts = sorted((sharded_stage3 / "household_imputed").glob(
            "HuisIdBSV=*/HuisBatchIdBSV=*/part.parquet"))
        assert len(parts) == 10
        frames = []
        for p in parts:
            hid = int(p.parents[1].name.split("=")[1])
            df = _read_parquet(p)
            assert "HuisIdBSV" not in df.columns  # ids live in the path
            df.insert(0, "HuisIdBSV", pd.array([hid] * len(df), dtype="Int64"))
            frames.append(df)
        sharded = pd.concat(frames, ignore_index=True)
        legacy = _read_parquet(ibis_pipeline / "household_imputed.parquet")
        common = [c for c in legacy.columns if c in sharded.columns]
        key = ["HuisIdBSV", "ReadingDate"]
        pd.testing.assert_frame_equal(
            sharded[common].sort_values(key).reset_index(drop=True),
            legacy[common].sort_values(key).reset_index(drop=True),
            check_dtype=True,
        )

    def test_gap_stats_and_summaries_equal(self, sharded_stage3, ibis_pipeline):
        for name, key in [
            ("impute_gap_stats.parquet", ["HuisIdBSV", "column"]),
            ("impute_summary_household.parquet", ["HuisIdBSV", "column"]),
            ("impute_summary_project.parquet", ["ProjectIdBSV", "column"]),
        ]:
            sharded = _read_parquet(sharded_stage3 / name)
            legacy = _read_parquet(ibis_pipeline / name)
            cols = [c for c in legacy.columns if c != "methods"]  # set-valued lists
            pd.testing.assert_frame_equal(
                sharded[cols].sort_values(key).reset_index(drop=True),
                legacy[cols].sort_values(key).reset_index(drop=True),
                check_dtype=False,
            )

    def test_aggregated_diff_equal(self, sharded_stage3, ibis_pipeline):
        sharded = _read_parquet(sharded_stage3 / "household_aggregated_diff.parquet")
        legacy = _read_parquet(ibis_pipeline / "household_aggregated_diff.parquet")
        key = ["ProjectIdBSV", "ReadingDate"]
        pd.testing.assert_frame_equal(
            sharded.sort_values(key).reset_index(drop=True)[legacy.columns],
            legacy.sort_values(key).reset_index(drop=True),
            check_dtype=False,
        )


class TestBatchSafetyGuards:
    """The averaging/imputation chain refuses multi-batch households and
    overlapping batch periods until batch-aware imputation exists."""

    def _mapped_df(self, start, n=6):
        dates = pd.date_range(start, periods=n, freq="5min", tz="UTC")
        return pd.DataFrame({
            "ReadingDate": dates,
            "ElektriciteitNetgebruik": pd.array(range(n), dtype="Float64"),
            "ElektriciteitNetgebruikDiff": pd.array([0.1] * n, dtype="Float64"),
        })

    def _shard(self, root, hid, hbid, start):
        d = root / "sharded" / f"HuisIdBSV={hid}" / f"HuisBatchIdBSV={hbid}"
        d.mkdir(parents=True)
        self._mapped_df(start).to_parquet(d / "part.parquet")

    def _agg_with_artifacts(self, tmp_path, src_agg):
        agg = tmp_path / "agg"
        (agg / "sharded").mkdir(parents=True)
        for name in ("avg_diffs.parquet", "household_diff_max_bounds.parquet"):
            shutil.copyfile(src_agg / name, agg / "sharded" / name)
        return agg

    def test_multi_batch_household_raises_in_impute(self, tmp_path, ibis_pipeline, _cum_cols):
        # household 1 delivered in two batches (disjoint periods): imputing the
        # concatenated series would fill the BETWEEN-batch gap -> refuse.
        _write_batch_index(tmp_path, [(1, 1, 1, True), (1, 9, 1, True)])
        self._shard(tmp_path, 1, 1, "2024-01-01")
        self._shard(tmp_path, 1, 9, "2025-01-01")
        agg = self._agg_with_artifacts(tmp_path, ibis_pipeline)
        with pytest.raises(HuisBatchOverlapError):
            impute_mapped_households_sharded(
                mapped_folder_path=tmp_path, aggregate_folder_path=agg,
                cum_cols=_cum_cols, huis_ids=[1],
            )

    def test_duplicate_timestamps_raise_in_impute(self, tmp_path, ibis_pipeline, _cum_cols):
        # overlapping batch periods: duplicate (HuisIdBSV, ReadingDate).
        _write_batch_index(tmp_path, [(1, 1, 1, True), (1, 9, 1, True)])
        self._shard(tmp_path, 1, 1, "2024-01-01")
        self._shard(tmp_path, 1, 9, "2024-01-01")  # same period redelivered
        agg = self._agg_with_artifacts(tmp_path, ibis_pipeline)
        with pytest.raises((HuisBatchOverlapError, ValueError)):
            impute_mapped_households_sharded(
                mapped_folder_path=tmp_path, aggregate_folder_path=agg,
                cum_cols=_cum_cols, huis_ids=[1],
            )

    def test_multi_batch_household_raises_in_diffs(self, tmp_path, _cum_cols):
        # project AVERAGES must also refuse: a household in two batches feeds
        # the estimation inputs the imputer relies on.
        _write_batch_index(tmp_path, [(1, 1, 1, True), (1, 9, 1, True)])
        self._shard(tmp_path, 1, 1, "2024-01-01")
        self._shard(tmp_path, 1, 9, "2024-01-01")
        with pytest.raises((HuisBatchOverlapError, ValueError)):
            prepare_diffs_sharded(
                mapped_folder_path=tmp_path, aggregate_folder_path=tmp_path / "agg",
                cumulative_columns=_cum_cols, huis_ids=[1],
            )
