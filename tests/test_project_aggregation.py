"""
Stage 6: project aggregation.

Two subjects. **What the stage computes** -- the project value at a timestamp
is the mean over that project's households and no other project's. **What it
refuses** -- a source with duplicate (ids.HOUSEHOLD, ReadingDate) rows.

Project outputs are single parquet files by design, whatever shape the
household artifacts they read had; that is this writer's own behaviour, not a
mode. Cross-FORMAT reading is `etdmap.storage`'s subject and is tested there.
"""

import shutil

import pandas as pd
import pytest

from etdmap import ids
from etdtransform.aggregate import (
    aggregate_project_data_duckdb,
    resample_hh_data_duckdb,
)

pytestmark = [pytest.mark.integration, pytest.mark.pipeline]





_AVG_COL = "ElektriciteitNetgebruikHoogDiff"
_PROJECT_CONFIG = {
    _AVG_COL: {"resample_method": "sum", "aggregate_method": "avg"},
}

#: (project, household) -> value. Two projects, two households each, with
#: per-project means (3.0 and 20.0) that are far apart and nowhere near the
#: mean over all four (11.5), so aggregating across projects is visible
#: rather than plausible.
_MEMBERSHIP = {(1, 1): 2.0, (1, 2): 4.0, (2, 3): 10.0, (2, 4): 30.0}
_EXPECTED_MEAN = {1: 3.0, 2: 20.0}
_SLOTS = 3


class TestProjectAggregateIsTheMeanOverItsOwnHouseholds:
    """The project value at a timestamp is the mean over that project's
    households, and no other project's.

    The monolith-versus-shards comparison protected this indirectly, through a
    single project. Two projects state it directly, and catch the case the
    comparison could not: both layouts aggregating across projects together.
    """

    @pytest.fixture
    def project_rows(self, tmp_path):
        from unittest.mock import patch

        from conftest import dataset_frame, write_shard_tree

        agg = tmp_path / "agg"
        write_shard_tree(agg / "household_15min", {
            hid: dataset_frame(n_rows=_SLOTS, project=pid, freq="15min",
                                 **{_AVG_COL: value})
            for (pid, hid), value in _MEMBERSHIP.items()
        })

        with patch("etdmap.data_model.get_aggregation_config",
                   return_value=_PROJECT_CONFIG):
            aggregate_project_data_duckdb(
                intervals=("15min",), aggregate_folder_path=str(agg),
            )
        return pd.read_parquet(agg / "project_15min.parquet",
                               dtype_backend="numpy_nullable")

    def test_each_project_takes_the_mean_of_its_own_households(self, project_rows):
        for pid, expected in _EXPECTED_MEAN.items():
            rows = project_rows[project_rows[ids.PROJECT] == pid]
            assert rows[_AVG_COL].astype(float).tolist() == [expected] * _SLOTS

    def test_both_projects_are_present_and_separate(self, project_rows):
        assert sorted(project_rows[ids.PROJECT].unique().tolist()) == [1, 2]
        assert len(project_rows) == _SLOTS * len(_EXPECTED_MEAN)

    def test_the_project_output_is_a_single_file(self, tmp_path, project_rows):
        # Project outputs stay monoliths by design, whatever shape the
        # household artifacts they were read from had.
        assert (tmp_path / "agg" / "project_15min.parquet").is_file()


class TestSourceGuards:
    """What the stage refuses."""

    def test_duplicate_reading_dates_in_sharded_source_raise(self, tmp_path,
                                                            fresh_shards):
        # Copies a household batch directory in, so it needs a tree of its own.
        from conftest import shard_dir

        shard_src = fresh_shards("household_calculated")
        out_dir = tmp_path / "resampled"
        out_dir.mkdir()
        resample_hh_data_duckdb(
            str(shard_src), str(out_dir), intervals=("60min",),
        )
        # Duplicate one household's resampled shard under a second household batch id.
        hh_dir = next((out_dir / "household_60min").glob("HuisIdBSV=*"))
        household_id = int(hh_dir.name.split("=")[1])
        src_batch = next(hh_dir.glob("HuisBatchIdBSV=*"))
        shutil.copytree(src_batch, shard_dir(out_dir / "household_60min",
                                             household_id, 999))

        with pytest.raises(ValueError):
            aggregate_project_data_duckdb(
                intervals=("60min",), aggregate_folder_path=str(out_dir)
            )
