"""
Stage 3: the imputation chain, and the batch-safety guards.

What the chain must produce: the estimation inputs (`avg_diffs` and the
averaging reports), the imputed shards themselves, the gap statistics, and the
in-run household summary. A stage that ran but wrote one of those nowhere is a
run whose output cannot be read back or promoted.

The I/O shell is thin on purpose and the math lives in one place:
`prepare_diff_averages_sharded` feeds `prepare_diff_averages_for_impute_ibis`,
`impute_mapped_households_sharded` calls `_impute_chunk_core`, and the
summaries come from a shared finalizer.

Batch-safety guards (computational, at the compute entry points): the
averaging/imputation chain must refuse
  (a) a household spanning MULTIPLE batches -- concatenating batches would let
      the large-gap project-average method impute across the BETWEEN-batch gap
      (real non-delivery; the cross-batch continuity invariant), and
  (b) duplicate (HuisIdBSV, ReadingDate) -- overlapping batch periods double-
      weight project averages and break the sort/diff math.
Batch-aware imputation is native-resolution-phase work; until then: loud stop.
"""

import shutil
from pathlib import Path

import conftest
import pandas as pd
import pytest

import etdtransform
from etdmap.index_helpers import HouseBatchOverlapError
from etdmap.storage import resolve_source_path
from etdtransform.aggregate import impute_mapped_households_sharded
from etdtransform.impute import prepare_diff_averages_sharded, read_diffs
from etdtransform.load_data import included_household_ids


def _write_index(folder: Path, rows):
    """rows: list of (HuisIdBSV, HuisBatchIdBSV, ProjectIdBSV, Meenemen).

    Writes index.parquet (with the household-batch
    columns) as save_index_to_parquet produces it."""
    n = len(rows)
    df = pd.DataFrame({
        "HuisIdLeverancier": pd.array([f"L{r[0]}" for r in rows], dtype="string"),
        "HuisIdBSV": pd.array([r[0] for r in rows], dtype="Int64"),
        "HuisBatchIdBSV": pd.array([r[0] for r in rows], dtype="Int64"),
        "Meenemen": pd.array([r[3] for r in rows], dtype="boolean"),
        "ProjectIdLeverancier": pd.array(["P1"] * n, dtype="string"),
        "ProjectIdBSV": pd.array([r[2] for r in rows], dtype="Int64"),
        "Notities": pd.array([""] * n, dtype="string"),
        "Dataleverancier": pd.array(["test"] * n, dtype="string"),
        "HuisBatchIdBSV": pd.array([r[1] for r in rows], dtype="Int64"),
        "BatchIdBSV": pd.array([1] * n, dtype="Int64"),
        "Gegevensfrequentie": pd.array(["5min"] * n, dtype="string"),
    })
    folder.mkdir(parents=True, exist_ok=True)
    df.to_parquet(folder / "index.parquet")


@pytest.mark.unit
class TestExplicitOutputFolderRequired:
    """The sharded stage-3 WRITERS must never default their output folder.

    The config default (etdtransform.options.aggregate_folder_path) points at
    the promoted production area, and impute_mapped_households_sharded even
    clears its output directory before writing -- a bare call must be a loud
    error, never a silent production write. Callers pass the output folder
    explicitly.
    """

    def test_prepare_diff_averages_sharded_requires_aggregate_folder(self, tmp_path):
        with pytest.raises(ValueError, match="aggregate_folder_path"):
            prepare_diff_averages_sharded(
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


@pytest.mark.unit
class TestIncludedHouseholdIds:
    def test_meenemen_filter(self, tmp_path):
        _write_index(tmp_path, [(1, 1, 1, True), (2, 2, 1, False), (3, 3, 2, True)])
        assert included_household_ids(mapped_folder_path=tmp_path) == [1, 3]

    def test_missing_index_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            included_household_ids(mapped_folder_path=tmp_path)

    def test_household_with_two_included_batches_raises(self, tmp_path):
        _write_index(tmp_path, [(1, 1, 1, True), (1, 9, 1, True)])
        with pytest.raises(HouseBatchOverlapError):
            included_household_ids(mapped_folder_path=tmp_path)


def _read_parquet(p):
    return pd.read_parquet(p, dtype_backend="numpy_nullable")


def _copy_artifact(src_folder, dst_folder, name):
    """Copy a stage artifact in whichever layout the run that wrote it used.

    An artifact is a single parquet file or a directory of partitions, and
    which one a run produced is resolved by ``resolve_source_path`` rather than
    assumed here -- a hardcoded ``<name>.parquet`` silently stops finding the
    artifact the day it becomes partitioned.
    """
    src = Path(resolve_source_path(src_folder, name))
    dst = Path(dst_folder) / src.name
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copyfile(src, dst)
    return dst


@pytest.fixture(scope="module")
def averaged_input(tmp_path_factory):
    """An aggregate folder holding `avg_diffs`, from this file's own households.

    The guards below need imputation's one input artifact to exist before they
    can reach the check they are about. Built here from clean single-batch
    households at the production floor, rather than taken from the integration
    chain: these tests want an artifact, not that dataset, and borrowing it
    makes them pay for a chain they do not read.
    """
    from etdmap.testing import DEFAULT_HOUSEHOLDS_PER_PROJECT
    from conftest import household_frame, write_shard_tree

    root = tmp_path_factory.mktemp("averaged_input")
    mapped = root / "mapped"
    households = range(1, DEFAULT_HOUSEHOLDS_PER_PROJECT + 1)
    _write_index(mapped, [(h, h, 1, True) for h in households])
    write_shard_tree(
        mapped / "sharded",
        {
            h: household_frame(
                n_rows=6, start="2024-01-01", ElektriciteitNetgebruikDiff=0.1,
            )
            for h in households
        },
    )

    agg = root / "agg"
    prepare_diff_averages_sharded(
        mapped_folder_path=mapped,
        aggregate_folder_path=agg,
        cumulative_columns=["ElektriciteitNetgebruik"],
    )
    return agg / "sharded"


@pytest.mark.integration
@pytest.mark.pipeline
class TestBatchSafetyGuards:
    """The averaging/imputation chain refuses multi-batch households and
    overlapping batch periods until batch-aware imputation exists."""

    def _shard(self, root, hid, hbid, start):
        """One mapped shard for (hid, hbid), beginning at *start*.

        Through the shared helpers, so the frame carries whatever a household
        artifact is required to carry and the path is the one the readers
        resolve. The mapped sharded area has no artifact-name level, hence the
        `sharded` root.
        """
        from conftest import household_frame, write_shard_tree

        write_shard_tree(
            root / "sharded",
            {(hid, hbid): household_frame(
                n_rows=6, start=start, ElektriciteitNetgebruikDiff=0.1,
            )},
        )

    def _agg_with_artifacts(self, tmp_path, src_agg):
        agg = tmp_path / "agg"
        (agg / "sharded").mkdir(parents=True)
        for name in ("avg_diffs",):
            _copy_artifact(src_agg, agg / "sharded", name)
        return agg

    def test_multi_batch_household_raises_in_impute(self, tmp_path, averaged_input, _cum_cols):
        # household 1 delivered in two batches (disjoint periods): imputing the
        # concatenated series would fill the BETWEEN-batch gap -> refuse.
        _write_index(tmp_path, [(1, 1, 1, True), (1, 9, 1, True)])
        self._shard(tmp_path, 1, 1, "2024-01-01")
        self._shard(tmp_path, 1, 9, "2025-01-01")
        agg = self._agg_with_artifacts(tmp_path, averaged_input)
        with pytest.raises(HouseBatchOverlapError):
            impute_mapped_households_sharded(
                mapped_folder_path=tmp_path, aggregate_folder_path=agg,
                cum_cols=_cum_cols, household_ids=[1],
            )

    def test_duplicate_timestamps_raise_in_impute(self, tmp_path, averaged_input, _cum_cols):
        # overlapping batch periods: duplicate (HuisIdBSV, ReadingDate).
        _write_index(tmp_path, [(1, 1, 1, True), (1, 9, 1, True)])
        self._shard(tmp_path, 1, 1, "2024-01-01")
        self._shard(tmp_path, 1, 9, "2024-01-01")  # same period redelivered
        agg = self._agg_with_artifacts(tmp_path, averaged_input)
        with pytest.raises((HouseBatchOverlapError, ValueError)):
            impute_mapped_households_sharded(
                mapped_folder_path=tmp_path, aggregate_folder_path=agg,
                cum_cols=_cum_cols, household_ids=[1],
            )

    def test_multi_batch_household_raises_in_diffs(self, tmp_path, _cum_cols):
        # project AVERAGES must also refuse: a household in two batches feeds
        # the estimation inputs the imputer relies on.
        _write_index(tmp_path, [(1, 1, 1, True), (1, 9, 1, True)])
        self._shard(tmp_path, 1, 1, "2024-01-01")
        self._shard(tmp_path, 1, 9, "2024-01-01")
        with pytest.raises((HouseBatchOverlapError, ValueError)):
            prepare_diff_averages_sharded(
                mapped_folder_path=tmp_path, aggregate_folder_path=tmp_path / "agg",
                cumulative_columns=_cum_cols, household_ids=[1],
            )
