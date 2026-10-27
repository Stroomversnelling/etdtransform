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

import pandas as pd
import pytest

import etdmap
import etdtransform
from etdtransform.load_data import get_household_table, get_household_tables

pytestmark = pytest.mark.integration

_HOUSEHOLDS = [1, 2]
_ROWS_PER_HH = 4


def _household_frame() -> pd.DataFrame:
    rows = []
    for household_id in _HOUSEHOLDS:
        for i in range(_ROWS_PER_HH):
            rows.append({
                "HuisIdBSV": household_id,
                "HuisBatchIdBSV": household_id + 1000,
                "ProjectIdBSV": 1,
                "ReadingDate": pd.Timestamp("2024-01-01", tz="UTC")
                + pd.Timedelta(hours=i),
                "value": float(household_id * 10 + i),
            })
    return pd.DataFrame(rows).astype({
        "HuisIdBSV": "int64", "ProjectIdBSV": "int64", "value": "float64",
        "HuisBatchIdBSV": "int64", "ProjectIdBSV": "int64", "value": "float64",
    })


def _index_frame() -> pd.DataFrame:
    """A well-formed index: read_index requires the full metadata column
    set."""
    n = len(_HOUSEHOLDS)
    return pd.DataFrame({
        "HuisIdLeverancier": pd.array([f"L{h}" for h in _HOUSEHOLDS], dtype="string"),
        "HuisIdBSV": pd.array(_HOUSEHOLDS, dtype="Int64"),
        "HuisBatchIdBSV": pd.array([value + 1000 for value in _HOUSEHOLDS], dtype="Int64"),
        "Meenemen": pd.array([True] * n, dtype="boolean"),
        "ProjectIdLeverancier": pd.array(["P1"] * n, dtype="string"),
        "ProjectIdBSV": pd.array([1] * n, dtype="Int64"),
        "Notities": pd.array([""] * n, dtype="string"),
        "Dataleverancier": pd.array(["etdmap"] * n, dtype="string"),
    })


@pytest.fixture
def aggregate_folder(tmp_path):
    """A minimal aggregate folder: one artifact per name the loader can ask for."""
    for name in ("household_60min", "household_default", "household_calculated"):
        _household_frame().to_parquet(tmp_path / f"{name}.parquet", engine="pyarrow")
    _index_frame().to_parquet(tmp_path / "index.parquet", engine="pyarrow")

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
        assert set(df["HuisIdBSV"]) == set(_HOUSEHOLDS)

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

        sort_cols = ["HuisIdBSV", "ReadingDate"]
        pd.testing.assert_frame_equal(
            multi.sort_values(sort_cols).reset_index(drop=True),
            single.sort_values(sort_cols).reset_index(drop=True),
        )


class TestHouseholdAndProjectFilters:
    """Filters are resolved to shard paths so the reader skips other
    households entirely; results must be identical either way."""

    def test_household_ids_restrict_rows(self, aggregate_folder):
        df = get_household_table("60min", include_weather=False, household_ids=[1]).execute()
        assert set(df["HuisIdBSV"]) == {1}
        assert len(df) == _ROWS_PER_HH

    def test_unfiltered_matches_filtered_union(self, aggregate_folder):
        full = get_household_table("60min", include_weather=False).execute()
        parts = pd.concat([
            get_household_table("60min", include_weather=False, household_ids=[h]).execute()
            for h in _HOUSEHOLDS
        ])
        sort_cols = ["HuisIdBSV", "ReadingDate"]
        pd.testing.assert_frame_equal(
            full.sort_values(sort_cols).reset_index(drop=True),
            parts.sort_values(sort_cols).reset_index(drop=True),
        )

    def test_projects_resolve_to_their_households(self, aggregate_folder):
        """Projects are a column, not a partition, so they are resolved to
        household ids through the index before reading."""
        df = get_household_table("60min", include_weather=False, projects=[1]).execute()
        assert set(df["HuisIdBSV"]) == set(_HOUSEHOLDS)

    def test_project_and_household_filters_intersect(self, aggregate_folder):
        df = get_household_table("60min", include_weather=False, projects=[1], household_ids=[2]).execute()
        assert set(df["HuisIdBSV"]) == {2}

    def test_unknown_project_yields_no_households(self, aggregate_folder):
        from etdtransform.load_data import households_in_projects
        assert households_in_projects([99999]) == []
