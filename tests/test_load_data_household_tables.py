"""
Contract tests for the household loader API.

`get_household_tables()` is used directly by downstream analysis (notebooks
and scripts outside this repo) but had no test. It was refactored to delegate
to the single-table loader `get_household_table()`; these tests pin the
contract so that refactor -- and any future one -- is provably
behaviour-preserving:

- the returned dict keys are exactly what the flags ask for,
- each table is joined with the index,
- the single-table loader and the multi-table loader return the SAME data for
  the same name.

Deliberately built on a tiny hand-made aggregate folder rather than the full
pipeline fixture, so the loader contract is tested independently of pipeline
behaviour and runs fast.
"""

import etdmap
import pandas as pd
import pytest
from etdmap import ids
from etdmap.testing import datasets_frame, household_batch_of, index_frame

import etdtransform
from etdtransform.load_data import get_household_table, get_household_tables

pytestmark = pytest.mark.integration

_HOUSEHOLDS = [1, 2]
_ROWS_PER_HH = 4

#: The dataset each household in this folder is read through. HuisBatchIdBSV
#: is what the index is joined on, so every artifact here carries it.
_DATASETS = {
    household_id: household_batch_of(household_id)
    for household_id in _HOUSEHOLDS
}


@pytest.fixture
def aggregate_folder(tmp_path):
    """A minimal aggregate folder: one artifact per name the loader can ask for."""
    rows = datasets_frame(
        {
            dataset: {
                "value": [
                    float(household_id * 10 + i) for i in range(_ROWS_PER_HH)
                ]
            }
            for household_id, dataset in _DATASETS.items()
        },
        freq="1h",
    )
    for name in ("household_60min", "household_default", "household_calculated"):
        rows.to_parquet(tmp_path / f"{name}.parquet", engine="pyarrow")
    index_frame(_DATASETS.values(), Meenemen=True).to_parquet(
        tmp_path / "index.parquet", engine="pyarrow"
    )

    prev = (
        etdtransform.options.aggregate_folder_path,
        etdtransform.options.mapped_folder_path,
        etdmap.options.mapped_folder_path,
    )
    etdtransform.options.aggregate_folder_path = str(tmp_path)
    etdtransform.options.mapped_folder_path = str(tmp_path)
    etdmap.options.mapped_folder_path = str(tmp_path)
    yield tmp_path
    (
        etdtransform.options.aggregate_folder_path,
        etdtransform.options.mapped_folder_path,
        etdmap.options.mapped_folder_path,
    ) = prev


class TestSingleTableLoader:
    def test_returns_rows_joined_with_index(self, aggregate_folder):
        df = get_household_table("60min", include_weather=False).execute()
        assert len(df) == len(_HOUSEHOLDS) * _ROWS_PER_HH
        # the index join contributes Meenemen
        assert "Meenemen" in df.columns
        assert set(df[ids.HOUSEHOLD]) == set(_HOUSEHOLDS)

    def test_weather_can_be_skipped(self, aggregate_folder):
        """The default matches get_household_tables (weather included); a
        caller that does not need it says so and avoids the join."""
        import inspect
        assert (
            inspect.signature(get_household_table)
            .parameters["include_weather"].default is True
        ), "must default the same way as get_household_tables"
        df = get_household_table("60min", include_weather=False).execute()
        assert not any(c.lower().startswith("temperatuur") for c in df.columns)


class TestMultiTableLoaderContract:
    """The API downstream analysis actually calls -- keys must not drift."""

    def test_intervals_only_by_default_flags(self, aggregate_folder, monkeypatch):
        monkeypatch.setattr("etdtransform.load_data.intervals", ["60min"])
        tables = get_household_tables(
            include_weather=False,
            include_preimputed_data=False,
            include_calculated_data=False,
        )
        assert set(tables) == {"60min"}

    def test_flags_add_default_and_calculated(self, aggregate_folder, monkeypatch):
        monkeypatch.setattr("etdtransform.load_data.intervals", ["60min"])
        tables = get_household_tables(
            include_weather=False,
            include_preimputed_data=True,
            include_calculated_data=True,
        )
        assert set(tables) == {"60min", "default", "calculated"}


class TestLoadersAgree:
    """The refactor's core claim: one table or many, the data is identical."""

    @pytest.mark.parametrize("name", ["60min", "default", "calculated"])
    def test_single_matches_multi(self, aggregate_folder, monkeypatch, name):
        monkeypatch.setattr("etdtransform.load_data.intervals", ["60min"])
        multi = get_household_tables(
            include_weather=False,
            include_preimputed_data=True,
            include_calculated_data=True,
        )[name].execute()
        single = get_household_table(name, include_weather=False).execute()

        sort_cols = [ids.HOUSEHOLD, "ReadingDate"]
        pd.testing.assert_frame_equal(
            multi.sort_values(sort_cols).reset_index(drop=True),
            single.sort_values(sort_cols).reset_index(drop=True),
        )


class TestDatasetAndProjectFilters:
    """A filter prunes shard paths so the reader skips the other directories,
    then selects rows on the dataset; results must be identical either way."""

    def test_dataset_ids_restrict_rows(self, aggregate_folder):
        one = _DATASETS[1]
        df = get_household_table(
            "60min", include_weather=False, household_batch_ids=[one],
        ).execute()
        assert set(df[ids.HOUSEHOLD_BATCH]) == {one}
        assert len(df) == _ROWS_PER_HH

    def test_unfiltered_matches_filtered_union(self, aggregate_folder):
        full = get_household_table("60min", include_weather=False).execute()
        parts = pd.concat([
            get_household_table(
                "60min", include_weather=False, household_batch_ids=[dataset],
            ).execute()
            for dataset in _DATASETS.values()
        ])
        sort_cols = [ids.HOUSEHOLD_BATCH, ids.READING_DATE]
        pd.testing.assert_frame_equal(
            full.sort_values(sort_cols).reset_index(drop=True),
            parts.sort_values(sort_cols).reset_index(drop=True),
        )

    def test_projects_resolve_to_their_datasets(self, aggregate_folder):
        """Projects are a column, not a partition, so they are resolved to
        dataset ids through the index before reading."""
        df = get_household_table("60min", include_weather=False, projects=[1]).execute()
        assert set(df[ids.HOUSEHOLD_BATCH]) == set(_DATASETS.values())

    def test_project_and_dataset_filters_intersect(self, aggregate_folder):
        one = _DATASETS[2]
        df = get_household_table(
            "60min", include_weather=False, projects=[1],
            household_batch_ids=[one],
        ).execute()
        assert set(df[ids.HOUSEHOLD_BATCH]) == {one}

    def test_unknown_project_yields_no_datasets(self, aggregate_folder):
        from etdtransform.load_data import household_batches_in_projects
        assert household_batches_in_projects([99999]) == []
