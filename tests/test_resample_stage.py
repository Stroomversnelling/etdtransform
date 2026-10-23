"""
Stage 5: resampling household intervals.

Two subjects. **What the stage computes** -- a resampled value sums within one
household and one interval, so neither how rows are grouped into files nor
which other households are present can change a household's answer. **What the
writer refuses** -- a source with duplicate (HuisIdBSV, ReadingDate) rows, or
one with no HuisBatchIdBSV to partition on.

Cross-FORMAT reading (resolving a monolith or a hive tree) is
`etdmap.storage`'s subject and is tested there, in
test_storage_source_layout.py and test_storage_read_source_frame.py.
"""

import shutil

import pandas as pd
import pytest

from etdtransform.aggregate import resample_hh_data_duckdb

pytestmark = [pytest.mark.integration, pytest.mark.pipeline]

INTERVALS = ("5min", "60min")




#: One column, summed on resample -- enough to state the property, and
#: isolated from data-model syncs the way the other resample tests are.
_SUM_COL = "ElektriciteitNetgebruikHoogDiff"
_RESAMPLE_CONFIG = {
    _SUM_COL: {"resample_method": "sum", "aggregate_method": "avg"},
}

#: Distinct orders of magnitude per household, so any value that crossed
#: between them would be visibly wrong rather than plausibly close.
_PER_HOUSEHOLD_VALUE = {1: 1.0, 2: 10.0, 3: 100.0}
_SLOTS = 6          # six 5-minute rows
_PER_WINDOW = 3     # three of them per 15-minute window


class TestResamplingIsPerHouseholdAndInterval:
    """Resampling sums within one household and one interval, so neither how
    rows are grouped into files nor which other households are present can
    change a household's answer.

    This is what the monolith-versus-shards comparison was protecting
    indirectly; asserted directly it also catches both sides being wrong
    together, which a comparison cannot.
    """

    @pytest.fixture
    def resampled(self, tmp_path):
        from unittest.mock import patch

        from conftest import household_frame, write_shard_tree
        from etdmap.storage import read_source_frame

        src = write_shard_tree(tmp_path / "calculated_shards", {
            hid: household_frame(n_rows=_SLOTS, **{_SUM_COL: value})
            for hid, value in _PER_HOUSEHOLD_VALUE.items()
        })
        out = tmp_path / "out"
        out.mkdir()
        with patch("etdmap.data_model.get_aggregation_config",
                   return_value=_RESAMPLE_CONFIG):
            resample_hh_data_duckdb(str(src), str(out), intervals=("15min",))
        return read_source_frame(str(out / "household_15min"))

    def test_each_household_sums_only_its_own_rows(self, resampled):
        # Three 5-minute rows per 15-minute window, so the window total is
        # three times that household's value and nothing else.
        for hid, value in _PER_HOUSEHOLD_VALUE.items():
            rows = resampled[resampled["HuisIdBSV"] == hid]
            assert rows[_SUM_COL].astype(float).tolist() == [value * _PER_WINDOW] * 2

    def test_the_windows_are_not_summed_together(self, resampled):
        # Two windows per household, each its own total -- not one row of six
        # readings, and not a running total across windows.
        for hid in _PER_HOUSEHOLD_VALUE:
            assert len(resampled[resampled["HuisIdBSV"] == hid]) == 2

    def test_every_household_survives_the_resample(self, resampled):
        assert sorted(resampled["HuisIdBSV"].unique().tolist()) == sorted(
            _PER_HOUSEHOLD_VALUE
        )


class TestSourceGuards:
    """What the writer refuses, and what it does on a re-run."""

    def test_rerun_overwrites_existing_shard_output(self, tmp_path,
                                                    calculated_shards):
        """Regression: the writer clears its existing output directory on a
        re-run. This cleanup only executes when output already exists -- a
        path that tests with freshly created directories never reach."""
        from etdmap.storage import read_source_frame

        shard_src = calculated_shards
        out = tmp_path / "out"
        out.mkdir()
        resample_hh_data_duckdb(str(shard_src), str(out), intervals=("60min",))
        first = read_source_frame(str(out / "household_60min"))
        resample_hh_data_duckdb(str(shard_src), str(out), intervals=("60min",))  # must not raise
        second = read_source_frame(str(out / "household_60min"))
        assert len(first) == len(second)

    def test_duplicate_reading_dates_in_sharded_source_raise(self, tmp_path,
                                                            fresh_shards):
        # Copies a batch directory in, so it needs a tree of its own.
        from conftest import shard_dir

        shard_src = fresh_shards("household_calculated")
        src1 = next(shard_dir(shard_src, 1).parent.glob("HuisBatchIdBSV=*"))
        shutil.copytree(src1, shard_dir(shard_src, 1, 999))

        with pytest.raises(ValueError):
            resample_hh_data_duckdb(
                str(shard_src), str(tmp_path / "out"), intervals=("60min",),
            )

    def test_a_source_without_a_batch_column_is_refused(self, tmp_path):
        # As in test_calculated_stage: a source with no batch id to file the rows
        # under is a caller error, stated with a frame that has none.
        from conftest import household_frame

        source = tmp_path / "no_batch_id.parquet"
        household_frame(**{_SUM_COL: 1.0}).to_parquet(source, index=False)
        with pytest.raises(ValueError, match="HuisBatchIdBSV"):
            resample_hh_data_duckdb(
                str(source), str(tmp_path / "out"), intervals=("60min",),
            )
