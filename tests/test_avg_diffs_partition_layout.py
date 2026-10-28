"""The averages artifact: one partition per project and cadence, one reader.

An average is energy per interval, so the interval is part of what the number
means. Putting it in the partition path as well as in a column means a reader
cannot take a quarter-hourly average for a five-minute one by dropping a
column, and it is what lets a chunk read only its own project.

There is one reader for both layouts, :func:`read_diffs`. That matters more than
it looks: a partitioned read cannot go through ``pd.read_parquet`` with the
nullable backend at all -- the partition columns arrive dictionary-encoded and
it raises -- so a second reader would either be wrong or would quietly return
numpy dtypes into a pipeline that depends on ``pd.NA``.
"""

import os

import pandas as pd
import pytest
from etdmap import ids
from etdmap.testing import datasets_frame, household_batch_of

from etdtransform.impute import AVG_DIFFS_ARTIFACT, read_diffs, write_diffs

pytestmark = pytest.mark.integration

COLUMN = "WarmteproductieRuimteverwarmingDiff"


def _averages():
    """Two projects on their own grids, as the averaging produces them.

    Built here because the two cadences and the project column are the
    partitioning this file is about, and no shared builder states an averages
    frame.
    """
    rows = []
    for project, grid in ((1, 5), (2, 15)):
        stamps = pd.date_range("2026-01-01", periods=4, freq=f"{grid}min")
        rows.append(pd.DataFrame({
            ids.PROJECT: pd.array([project] * 4, dtype="Int64"),
            "ReadingDate": stamps,
            f"{COLUMN}_avg": pd.array(
                [0.01, 0.02, pd.NA, 0.04], dtype="Float64"
            ),
            "cadence_minutes": pd.array([grid] * 4, dtype="Int64"),
        }))
    return pd.concat(rows, ignore_index=True)


def _partitions(root):
    return sorted(
        os.path.relpath(base, root).replace("\\", "/")
        for base, _dirs, files in os.walk(root)
        if files
    )


class TestTheLayout:
    def test_one_partition_per_project_and_cadence(self, tmp_path):
        write_diffs(_averages(), aggregate_folder_path=tmp_path)
        assert _partitions(tmp_path / AVG_DIFFS_ARTIFACT) == [
            "ProjectIdBSV=1/cadence_minutes=5",
            "ProjectIdBSV=2/cadence_minutes=15",
        ]

    def test_it_is_the_suffixless_spelling_of_the_canonical_name(self, tmp_path):
        """``promote_aggregate_outputs`` recognises a directory artifact by the
        canonical name without ``.parquet``, so the directory has to be named
        that way or promotion treats it as structure and leaves it behind."""
        target = write_diffs(_averages(), aggregate_folder_path=tmp_path)
        assert os.path.isdir(target)
        assert os.path.basename(target) == "avg_diffs"

    def test_a_rerun_replaces_the_partitions(self, tmp_path):
        """The writer owns its output directory. A partition a later run no
        longer produces would otherwise be read back beside that run's."""
        write_diffs(_averages(), aggregate_folder_path=tmp_path)
        smaller = _averages()
        smaller = smaller[smaller[ids.PROJECT] == 1]
        write_diffs(smaller, aggregate_folder_path=tmp_path)

        assert _partitions(tmp_path / AVG_DIFFS_ARTIFACT) == [
            "ProjectIdBSV=1/cadence_minutes=5",
        ]
        assert len(read_diffs(tmp_path)) == len(smaller)

    def test_averages_with_no_cadence_are_refused(self, tmp_path):
        bare = _averages().drop(columns=["cadence_minutes"])
        with pytest.raises(ValueError, match="cadence_minutes"):
            write_diffs(bare, aggregate_folder_path=tmp_path)


class TestTheRoundTrip:
    def test_what_was_written_comes_back(self, tmp_path):
        written = _averages()
        write_diffs(written, aggregate_folder_path=tmp_path)

        back = read_diffs(tmp_path, with_cadence=True)
        keys = [ids.PROJECT, "ReadingDate"]
        assert sorted(back.columns) == sorted(written.columns)
        pd.testing.assert_frame_equal(
            back.sort_values(keys).reset_index(drop=True),
            written.sort_values(keys).reset_index(drop=True),
            check_like=True,
        )

    def test_the_dtypes_are_nullable(self, tmp_path):
        """ADR-005. The partition columns are the ones at risk: they are
        reconstructed from the path, not read from a file."""
        write_diffs(_averages(), aggregate_folder_path=tmp_path)
        back = read_diffs(tmp_path, with_cadence=True)
        assert str(back[ids.PROJECT].dtype) == "Int64"
        assert str(back["cadence_minutes"].dtype) == "Int64"
        assert str(back[f"{COLUMN}_avg"].dtype) == "Float64"

    def test_a_missing_average_stays_missing(self, tmp_path):
        """Not zero. An unaveraged slot has no shape to fill from, and zero
        would be a shape."""
        write_diffs(_averages(), aggregate_folder_path=tmp_path)
        back = read_diffs(tmp_path)
        assert int(back[f"{COLUMN}_avg"].isna().sum()) == 2


class TestWhatTheFillReceives:
    """The fill merges these onto a chunk on (project, ReadingDate), and
    anything else in the frame lands in the written shards."""

    def test_the_cadence_is_left_behind_by_default(self, tmp_path):
        write_diffs(_averages(), aggregate_folder_path=tmp_path)
        assert "cadence_minutes" not in read_diffs(tmp_path).columns

    def test_it_can_be_asked_for(self, tmp_path):
        write_diffs(_averages(), aggregate_folder_path=tmp_path)
        assert "cadence_minutes" in read_diffs(
            tmp_path, with_cadence=True
        ).columns

    def test_it_merges_onto_a_chunk_on_project_and_date(self, tmp_path):
        write_diffs(_averages(), aggregate_folder_path=tmp_path)
        # One dataset of project 2, on that project's quarter-hourly grid and
        # tz-naive, which is what the merge keys have to meet.
        chunk = datasets_frame(
            {household_batch_of(7): {}}, n_rows=2, project=2,
            start="2026-01-01", freq="15min", tz=None,
        )
        merged = chunk.merge(
            read_diffs(tmp_path),
            on=[ids.PROJECT, "ReadingDate"], how="left",
        )
        assert [float(v) for v in merged[f"{COLUMN}_avg"]] == [0.01, 0.02]


class TestTheLegacySingleFile:
    """A run predating the partitioning left one file. The same reader has to
    read it, or a re-run is required before anything can be compared against
    it."""

    def test_the_file_layout_still_reads(self, tmp_path):
        written = _averages().drop(columns=["cadence_minutes"])
        written.to_parquet(
            tmp_path / f"{AVG_DIFFS_ARTIFACT}.parquet", engine="pyarrow",
            index=False,
        )
        back = read_diffs(tmp_path)
        assert len(back) == len(written)
        assert str(back[ids.PROJECT].dtype) == "Int64"

    def test_the_directory_wins_when_both_are_present(self, tmp_path):
        """Promotion supersedes the stale form, but a staging area can hold
        both mid-migration and the newer layout is the one the run wrote."""
        _averages().head(1).drop(columns=["cadence_minutes"]).to_parquet(
            tmp_path / f"{AVG_DIFFS_ARTIFACT}.parquet", engine="pyarrow",
            index=False,
        )
        write_diffs(_averages(), aggregate_folder_path=tmp_path)
        assert len(read_diffs(tmp_path)) == len(_averages())
