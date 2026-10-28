"""
Stage 3: the imputation chain, and the household-batch-safety guards.

What the chain must produce: the estimation inputs (`avg_diffs` and the
averaging reports), the imputed shards themselves, the gap statistics, and the
in-run household summary. A stage that ran but wrote one of those nowhere is a
run whose output cannot be read back or promoted.

The I/O shell is thin on purpose and the math lives in one place:
`prepare_diff_averages_sharded` feeds `prepare_diff_averages_for_impute_ibis`,
`impute_mapped_households_sharded` calls `_impute_chunk_core`, and the
summaries come from a shared finalizer.

Batch safety, at the compute entry points: a timestamp twice in one household
batch double-weights the project averages and breaks the sort/diff math, so it
is a loud stop. One household batch is one series and nothing a stage computes
crosses them, so a household holding several household batches is ordinary
input -- the averaging keeps each in its own project.
"""

import shutil
from pathlib import Path

import pandas as pd
import pytest
from etdmap import ids
from etdmap.index_helpers import HouseBatchOverlapError
from etdmap.storage import resolve_source_path
from etdmap.testing import index_frame, one_to_one_test_household_batch_of

import etdtransform
from etdtransform.aggregate import impute_mapped_households_sharded
from etdtransform.impute import prepare_diff_averages_sharded
from etdtransform.presence import WINDOW_MINUTES
from etdtransform.load_data import included_household_batch_ids

#: The spacing every frame in this file sits on, in minutes. The averaging
#: resolves a project's grid from the cadence sidecar, so the frames and the
#: sidecar are built from this one number.
GRID_MINUTES = 5

#: A SECOND household batch of a household -- another body of data for the same
#: household, such as a period after a renovation. The household it belongs to
#: is not the one derived from the household batch id, so every frame carrying it
#: declares that household through `household_of`.
SECOND_HOUSEHOLD_BATCH = 2001


def _write_index(folder: Path, household_batch_ids, *, household_of=None,
                 **columns):
    """Write index.parquet covering *household_batch_ids*.

    One row per household batch, which is the key of the index: a household
    delivered as several household batches appears once per household batch. *household_of*
    names the household a household batch belongs to where it is not the one derived
    from the household batch id, and any further keyword overrides an index column
    with a scalar or a per-row list.
    """
    folder.mkdir(parents=True, exist_ok=True)
    index_frame(
        household_batch_ids,
        household_of=household_of,
        Gegevensfrequentie=f"{GRID_MINUTES}min",
        **columns,
    ).to_parquet(folder / "index.parquet")


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
class TestIncludedHouseholdBatchIds:
    def test_meenemen_filter(self, tmp_path):
        _write_index(
            tmp_path,
            [one_to_one_test_household_batch_of(1), one_to_one_test_household_batch_of(2),
             one_to_one_test_household_batch_of(3)],
            Meenemen=[True, False, True],
            **{ids.PROJECT: [1, 1, 2]},
        )
        assert included_household_batch_ids(mapped_folder_path=tmp_path) == [
            one_to_one_test_household_batch_of(1), one_to_one_test_household_batch_of(3)
        ]

    def test_missing_index_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            included_household_batch_ids(mapped_folder_path=tmp_path)

    def test_both_household_batches_of_one_household_are_selected(self, tmp_path):
        """`Meenemen` is decided per household batch, so a second delivery of
        one household is its own decision.

        Collapsing the two to the household used to raise, which meant the
        pipeline could not run this case at all.
        """
        _write_index(
            tmp_path, [one_to_one_test_household_batch_of(1), SECOND_HOUSEHOLD_BATCH],
            household_of={SECOND_HOUSEHOLD_BATCH: 1}, Meenemen=True,
            **{ids.PROJECT: [1, 2], ids.BATCH: [1, 2]},
        )
        assert included_household_batch_ids(mapped_folder_path=tmp_path) == sorted(
            [one_to_one_test_household_batch_of(1), SECOND_HOUSEHOLD_BATCH]
        )


def _pooled(frames_by_household_batch, *, household_id):
    """The frames concatenated with their ids, as a chunk read would hold them."""
    pieces = []
    for household_batch_id, frame in frames_by_household_batch.items():
        piece = frame.copy()
        piece.insert(
            0, ids.HOUSEHOLD,
            pd.array([household_id] * len(piece), dtype="Int64"),
        )
        piece.insert(
            1, ids.HOUSEHOLD_BATCH,
            pd.array([household_batch_id] * len(piece), dtype="Int64"),
        )
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def _read_parquet(p):
    return pd.read_parquet(p, dtype_backend="numpy_nullable")


@pytest.mark.unit
class TestBatchSafetyGuards:
    """A timestamp twice in one household batch is refused, in both engines."""

    def test_a_concatenated_multi_batch_chunk_raises(self):
        """The guard the imputation chunk is checked with, on a frame holding
        one household's two household batches.

        Asserted against the guard rather than through the stage: the chunker
        groups by project and the read selects household batches exactly, so a
        chunk can no longer be built that holds both. The guard stays as the
        defence that keeps it that way.
        """
        from conftest import household_batch_frame

        from etdtransform.impute import assert_imputation_input_batch_safe

        first = household_batch_frame(
            n_rows=6, start="2024-01-01", freq=f"{GRID_MINUTES}min",
            ElektriciteitNetgebruikDiff=0.1,
        )
        second = household_batch_frame(
            n_rows=6, start="2025-01-01", freq=f"{GRID_MINUTES}min",
            ElektriciteitNetgebruikDiff=0.1,
        )
        pooled = _pooled(
            {one_to_one_test_household_batch_of(1): first, SECOND_HOUSEHOLD_BATCH: second},
            household_id=1,
        )
        with pytest.raises(HouseBatchOverlapError):
            assert_imputation_input_batch_safe(pooled)

    def test_a_timestamp_twice_in_one_household_batch_raises(self):
        """The other half of the guard: a redelivered period."""
        from conftest import household_batch_frame

        from etdtransform.impute import assert_imputation_input_batch_safe

        twice = household_batch_frame(
            n_rows=6, start="2024-01-01", freq=f"{GRID_MINUTES}min",
            ElektriciteitNetgebruikDiff=0.1,
        )
        pooled = _pooled(
            {one_to_one_test_household_batch_of(1): pd.concat([twice, twice], ignore_index=True)},
            household_id=1,
        )
        with pytest.raises(ValueError):
            assert_imputation_input_batch_safe(pooled)

@pytest.mark.integration
@pytest.mark.pipeline
class TestAHouseholdInTwoProjects:
    """A household delivered twice is two series, and each project averages
    its own.

    The legal shape: a household occurs at most once within a project and at
    most once within a delivery batch, so its second household batch carries
    its own project and its own delivery batch. `avg_diffs` is partitioned on
    (project, cadence) and the averaging groups on the project, so neither
    household batch can reach the other's average.
    """

    #: Household batches per project. Below the production floor by
    #: construction, which is what ``min_eligible=1`` below says; the floor
    #: itself is exercised at its production value in
    #: test_minimum_household_presence.py.
    PER_PROJECT = 2

    #: One value per project, far enough apart that any mixing of the shared
    #: household's two household batches moves an average off its project's
    #: own number.
    FIRST_VALUE = 0.1
    SECOND_VALUE = 0.5

    SHARED_HOUSEHOLD = 1

    #: One whole averaging window per household batch. A household batch must
    #: deliver MIN_WINDOW_COVERAGE of the window to be eligible at all, and the
    #: window is derived from the column's cadence rather than passed in, so a
    #: shorter frame makes every project withhold its average and the test
    #: could not tell the two keys apart.
    ROWS = int(WINDOW_MINUTES // GRID_MINUTES)

    def _population(self):
        """(household, household batch, project, delivery batch, value) per row.

        The shared household is in both projects: its first household batch
        with the rest of project 1, its second with the rest of project 2.
        """
        rows = [
            (h, one_to_one_test_household_batch_of(h), 1, 1, self.FIRST_VALUE)
            for h in range(1, self.PER_PROJECT + 1)
        ]
        second_project_households = range(
            self.PER_PROJECT + 1, 2 * self.PER_PROJECT
        )
        rows += [
            (h, one_to_one_test_household_batch_of(h), 2, 2, self.SECOND_VALUE)
            for h in second_project_households
        ]
        rows.append(
            (self.SHARED_HOUSEHOLD, SECOND_HOUSEHOLD_BATCH, 2, 2,
             self.SECOND_VALUE)
        )
        return rows

    def _averages(self, tmp_path):
        """Run the averaging over that population and read `avg_diffs` back."""
        from conftest import household_batch_frame, write_shard_tree
        from etdmap.mapping_helpers import (
            cadence_to_rows,
            detect_column_cadence,
            write_column_cadence,
        )
        from etdmap.testing import PAIRS_EMPTY

        from etdtransform.impute import read_diffs

        population = self._population()
        mapped = tmp_path / "mapped"
        frames = {
            (household, batch): household_batch_frame(
                n_rows=self.ROWS, start="2024-01-01",
                freq=f"{GRID_MINUTES}min",
                # The project comes from the index, per household batch -- a
                # project column on the frame would make every shard claim
                # the builder's default. PAIRS_EMPTY because the averaging
                # prefers a cumulative column holding any value over the
                # interval one, and these are the interval values under test.
                project=None, pairs=PAIRS_EMPTY,
                ElektriciteitNetgebruikDiff=value,
            )
            for household, batch, _project, _delivery, value in population
        }
        _write_index(
            mapped,
            [batch for _h, batch, _p, _d, _v in population],
            household_of={
                batch: household for household, batch, _p, _d, _v in population
            },
            Meenemen=True,
            **{
                ids.PROJECT: [p for _h, _b, p, _d, _v in population],
                ids.BATCH: [d for _h, _b, _p, d, _v in population],
            },
        )
        write_shard_tree(mapped / "sharded", frames)

        cadence_rows = []
        for (_household, batch), frame in frames.items():
            cadence_rows += cadence_to_rows(
                batch, detect_column_cadence(frame, grid_minutes=GRID_MINUTES),
            )
        write_column_cadence(cadence_rows, mapped)

        agg = tmp_path / "agg"
        previous = etdtransform.options.mapped_folder_path
        etdtransform.options.mapped_folder_path = mapped
        try:
            prepare_diff_averages_sharded(
                mapped_folder_path=mapped,
                aggregate_folder_path=agg,
                cumulative_columns=["ElektriciteitNetgebruik"],
                min_eligible=1,
            )
        finally:
            etdtransform.options.mapped_folder_path = previous
        return read_diffs(agg / "sharded")

    def test_each_project_averages_its_own_household_batches(self, tmp_path):
        averages = self._averages(tmp_path)
        column = "ElektriciteitNetgebruikDiff_avg"

        for project, expected in (
            (1, self.FIRST_VALUE), (2, self.SECOND_VALUE),
        ):
            own = averages.loc[averages[ids.PROJECT] == project, column]
            assert len(own), f"project {project} published no average"
            assert own.notna().all(), (
                f"project {project} withheld its average; every household "
                f"batch covers its window, so a gate removed it"
            )
            assert own.astype(float).round(6).eq(expected).all(), (
                f"project {project} averaged {sorted(set(own.astype(float)))} "
                f"rather than its own {expected} -- a value between the two "
                f"projects' means the shared household's two household batches "
                f"were pooled"
            )

    def test_a_timestamp_twice_in_one_household_batch_still_raises(self, tmp_path):
        """The engine-side guard, through the stage: the half that remains."""
        from conftest import household_batch_frame, write_shard_tree
        from etdmap.testing import PAIRS_EMPTY

        batch = one_to_one_test_household_batch_of(self.SHARED_HOUSEHOLD)
        once = household_batch_frame(
            n_rows=6, start="2024-01-01", freq=f"{GRID_MINUTES}min",
            project=None, pairs=PAIRS_EMPTY,
            ElektriciteitNetgebruikDiff=self.FIRST_VALUE,
        )
        mapped = tmp_path / "mapped"
        _write_index(mapped, [batch], Meenemen=True)
        write_shard_tree(
            mapped / "sharded",
            {(self.SHARED_HOUSEHOLD, batch): pd.concat(
                [once, once], ignore_index=True
            )},
        )
        with pytest.raises(ValueError, match=ids.HOUSEHOLD_BATCH):
            prepare_diff_averages_sharded(
                mapped_folder_path=mapped,
                aggregate_folder_path=tmp_path / "agg",
                cumulative_columns=["ElektriciteitNetgebruik"],
            )

    def test_both_projects_are_present(self, tmp_path):
        """The run completes with one household in two projects."""
        averages = self._averages(tmp_path)
        assert set(int(p) for p in averages[ids.PROJECT].dropna().unique()) == {1, 2}
