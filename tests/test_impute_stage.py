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


#: The spacing every frame in this file sits on, in minutes. The averaging
#: resolves a project's grid from the cadence sidecar, so the frames and the
#: sidecar are built from this one number.
GRID_MINUTES = 5


def _batch(household):
    """The dataset id (`HuisBatchIdBSV`) a frame for *household* carries.

    Never the household's own id: a household can have several datasets, so
    code reaching for `HuisIdBSV` where the dataset is meant finds no rows
    rather than the right ones.
    """
    return household + 1000


def _another_batch(household):
    """The id of a SECOND dataset of *household* -- another body of data for
    the same household, such as a period after a renovation."""
    return household + 2000


def _write_index(folder: Path, rows):
    """rows: list of (HuisIdBSV, HuisBatchIdBSV, ProjectIdBSV, Meenemen).

    Writes index.parquet (with the household-batch
    columns) as save_index_to_parquet produces it. One row per household
    batch, which is the key of the index; a household delivered in several
    batches appears once per batch."""
    n = len(rows)
    df = pd.DataFrame({
        "HuisIdLeverancier": pd.array([f"L{r[0]}" for r in rows], dtype="string"),
        "HuisIdBSV": pd.array([r[0] for r in rows], dtype="Int64"),
        "Meenemen": pd.array([r[3] for r in rows], dtype="boolean"),
        "ProjectIdLeverancier": pd.array(["P1"] * n, dtype="string"),
        "ProjectIdBSV": pd.array([r[2] for r in rows], dtype="Int64"),
        "Notities": pd.array([""] * n, dtype="string"),
        "Dataleverancier": pd.array(["test"] * n, dtype="string"),
        "HuisBatchIdBSV": pd.array([r[1] for r in rows], dtype="Int64"),
        "BatchIdBSV": pd.array([1] * n, dtype="Int64"),
        "Gegevensfrequentie": pd.array([f"{GRID_MINUTES}min"] * n, dtype="string"),
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
        _write_index(tmp_path, [
            (1, _batch(1), 1, True),
            (2, _batch(2), 1, False),
            (3, _batch(3), 2, True),
        ])
        assert included_household_ids(mapped_folder_path=tmp_path) == [1, 3]

    def test_missing_index_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            included_household_ids(mapped_folder_path=tmp_path)

    def test_household_with_two_included_batches_raises(self, tmp_path):
        _write_index(tmp_path, [
            (1, _batch(1), 1, True), (1, _another_batch(1), 1, True),
        ])
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
    from etdmap.mapping_helpers import (
        cadence_to_rows,
        detect_column_cadence,
        write_column_cadence,
    )
    from etdmap.testing import DEFAULT_HOUSEHOLDS_PER_PROJECT
    from conftest import household_frame, write_shard_tree

    root = tmp_path_factory.mktemp("averaged_input")
    mapped = root / "mapped"
    households = range(1, DEFAULT_HOUSEHOLDS_PER_PROJECT + 1)
    frames = {
        (h, _batch(h)): household_frame(
            n_rows=6, start="2024-01-01", freq=f"{GRID_MINUTES}min",
            ElektriciteitNetgebruikDiff=0.1,
        )
        for h in households
    }
    _write_index(mapped, [(h, _batch(h), 1, True) for h in households])
    write_shard_tree(mapped / "sharded", frames)

    # The averaging reads the interval a dataset reports at from the cadence
    # sidecar, one row per (dataset, column), and refuses a dataset it resolves
    # no grid for. Measured by the mapping's own detector and written by its own
    # writer, so the sidecar is keyed and typed the way a mapped dataset's is.
    cadence_rows = []
    for (_household, batch), frame in frames.items():
        cadence_rows += cadence_to_rows(
            batch, detect_column_cadence(frame, grid_minutes=GRID_MINUTES),
        )
    write_column_cadence(cadence_rows, mapped)

    agg = root / "agg"
    # The cadence sidecar is resolved through the configured mapped folder
    # rather than the one passed in, so the option names this dataset for the
    # duration of the call.
    previous = etdtransform.options.mapped_folder_path
    etdtransform.options.mapped_folder_path = mapped
    try:
        prepare_diff_averages_sharded(
            mapped_folder_path=mapped,
            aggregate_folder_path=agg,
            cumulative_columns=["ElektriciteitNetgebruik"],
        )
    finally:
        etdtransform.options.mapped_folder_path = previous
    return agg / "sharded"


@pytest.mark.integration
@pytest.mark.pipeline
class TestBatchSafetyGuards:
    """The averaging/imputation chain refuses multi-batch households and
    overlapping batch periods until batch-aware imputation exists."""

    def _shard(self, root, household_id, household_batch_id, start):
        """One mapped shard: the dataset *household_batch_id* of household
        *household_id*, beginning at *start*.

        Through the shared helpers, so the frame carries whatever a household
        artifact is required to carry and the path is the one the readers
        resolve. The mapped sharded area has no artifact-name level, hence the
        `sharded` root.
        """
        from conftest import household_frame, write_shard_tree

        write_shard_tree(
            root / "sharded",
            {(household_id, household_batch_id): household_frame(
                n_rows=6, start=start, freq=f"{GRID_MINUTES}min",
                ElektriciteitNetgebruikDiff=0.1,
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
        _write_index(tmp_path, [
            (1, _batch(1), 1, True), (1, _another_batch(1), 1, True),
        ])
        self._shard(tmp_path, 1, _batch(1), "2024-01-01")
        self._shard(tmp_path, 1, _another_batch(1), "2025-01-01")
        agg = self._agg_with_artifacts(tmp_path, averaged_input)
        with pytest.raises(HouseBatchOverlapError):
            impute_mapped_households_sharded(
                mapped_folder_path=tmp_path, aggregate_folder_path=agg,
                cum_cols=_cum_cols, household_ids=[1],
            )

    def test_duplicate_timestamps_raise_in_impute(self, tmp_path, averaged_input, _cum_cols):
        # overlapping batch periods: duplicate (HuisIdBSV, ReadingDate).
        _write_index(tmp_path, [
            (1, _batch(1), 1, True), (1, _another_batch(1), 1, True),
        ])
        self._shard(tmp_path, 1, _batch(1), "2024-01-01")
        # Same period redelivered as a second dataset of household 1.
        self._shard(tmp_path, 1, _another_batch(1), "2024-01-01")
        agg = self._agg_with_artifacts(tmp_path, averaged_input)
        with pytest.raises((HouseBatchOverlapError, ValueError)):
            impute_mapped_households_sharded(
                mapped_folder_path=tmp_path, aggregate_folder_path=agg,
                cum_cols=_cum_cols, household_ids=[1],
            )

    def test_multi_batch_household_raises_in_diffs(self, tmp_path, _cum_cols):
        # project AVERAGES must also refuse: a household in two batches feeds
        # the estimation inputs the imputer relies on.
        _write_index(tmp_path, [
            (1, _batch(1), 1, True), (1, _another_batch(1), 1, True),
        ])
        self._shard(tmp_path, 1, _batch(1), "2024-01-01")
        self._shard(tmp_path, 1, _another_batch(1), "2024-01-01")
        with pytest.raises((HouseBatchOverlapError, ValueError)):
            prepare_diff_averages_sharded(
                mapped_folder_path=tmp_path, aggregate_folder_path=tmp_path / "agg",
                cumulative_columns=_cum_cols, household_ids=[1],
            )
