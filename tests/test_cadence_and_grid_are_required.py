"""The cadence sidecar and the declared grid are read, never assumed.

Both set the spacing a threshold is judged at. Assuming five minutes for a
household that reports hourly judges its hour of consumption against a
five-minute bound, so the reading is nulled as out of bounds and its volume
leaves the dataset silently. The absence of either is therefore an error that
names the fix, not a default.
"""

import ibis
import pandas as pd
import pytest

from etdmap.mapping_helpers import CADENCE_FILENAME
from etdtransform.bounds import dataset_grid_minutes, load_column_cadence
from etdtransform.impute import calculate_average_diff_ibis

pytestmark = pytest.mark.unit


def _write_cadence(folder):
    pd.DataFrame({
        "HuisIdBSV": pd.array([1], dtype="Int64"),
        "column": pd.array(["ElektriciteitNetgebruik"], dtype="string"),
        "cadence_minutes": pd.array([15], dtype="Int64"),
    }).to_parquet(folder / CADENCE_FILENAME, engine="pyarrow")


def _write_index(folder, frequency="5min"):
    pd.DataFrame({
        "HuisIdBSV": pd.array([1], dtype="Int64"),
        "Gegevensfrequentie": pd.array([frequency], dtype="string"),
    }).to_parquet(folder / "index.parquet", engine="pyarrow")


class TestTheCadenceSidecarIsRequired:
    def test_a_missing_sidecar_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=CADENCE_FILENAME):
            load_column_cadence(str(tmp_path))

    def test_the_message_says_what_to_do(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Re-map"):
            load_column_cadence(str(tmp_path))

    def test_no_configured_folder_raises(self, monkeypatch):
        import etdtransform

        monkeypatch.setattr(
            etdtransform.options, "mapped_folder_path", None, raising=False
        )
        with pytest.raises(FileNotFoundError, match="mapped_folder_path"):
            load_column_cadence()

    def test_a_present_sidecar_is_returned(self, tmp_path):
        _write_cadence(tmp_path)
        cadence = load_column_cadence(str(tmp_path))
        assert not cadence.empty
        assert int(cadence["cadence_minutes"].iloc[0]) == 15


class TestTheDeclaredGridIsRequired:
    def test_a_missing_index_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="index.parquet"):
            dataset_grid_minutes(str(tmp_path))

    def test_no_configured_folder_raises(self, monkeypatch):
        import etdtransform

        monkeypatch.setattr(
            etdtransform.options, "mapped_folder_path", None, raising=False
        )
        with pytest.raises(FileNotFoundError, match="mapped_folder_path"):
            dataset_grid_minutes()

    def test_an_index_without_the_field_raises(self, tmp_path):
        pd.DataFrame({
            "HuisIdBSV": pd.array([1], dtype="Int64"),
        }).to_parquet(tmp_path / "index.parquet", engine="pyarrow")
        with pytest.raises(ValueError, match="Gegevensfrequentie"):
            dataset_grid_minutes(str(tmp_path))

    def test_an_unreadable_frequency_raises(self, tmp_path):
        _write_index(tmp_path, frequency="whenever")
        with pytest.raises(ValueError, match="Gegevensfrequentie"):
            dataset_grid_minutes(str(tmp_path))

    def test_a_declared_grid_is_returned(self, tmp_path):
        _write_index(tmp_path, frequency="15min")
        assert dataset_grid_minutes(str(tmp_path)) == 15.0

    def test_the_finest_declared_grid_wins(self, tmp_path):
        """An aggregated dataset spans batches; its rows sit on the finest of
        their grids."""
        pd.DataFrame({
            "HuisIdBSV": pd.array([1, 2], dtype="Int64"),
            "Gegevensfrequentie": pd.array(["15min", "5min"], dtype="string"),
        }).to_parquet(tmp_path / "index.parquet", engine="pyarrow")
        assert dataset_grid_minutes(str(tmp_path)) == 5.0


class TestTheGridIsNotDefaultedDownstream:
    def test_the_spreading_takes_the_grid_as_a_required_argument(self):
        """The spreading divides a window's volume by the slots it covers, so
        the grid decides how many slots that is. A default here would spread an
        hourly reading over whatever spacing the default names."""
        import inspect

        from etdtransform.impute import _resample_average_ibis

        parameter = inspect.signature(_resample_average_ibis).parameters[
            "grid_minutes"
        ]
        assert parameter.default is inspect.Parameter.empty

    def test_the_averaging_resolves_the_grid_per_project(self):
        """``calculate_average_diff_ibis`` accepts ``grid_minutes=None`` and
        resolves the grid per project from the cadence table. Accepting None is
        not the same as defaulting to a number: a project that resolves none
        raises, naming itself, rather than being averaged at some other
        project's spacing."""
        frame = pd.DataFrame({
            "ProjectIdBSV": pd.array([4, 4], dtype="Int64"),
            "HuisIdBSV": pd.array([1, 1], dtype="Int64"),
            "ReadingDate": pd.to_datetime(
                ["2026-01-01 00:00", "2026-01-01 00:05"]
            ),
            "ElektriciteitNetgebruikDiff": pd.array(
                [1.0, 1.0], dtype="Float64"
            ),
        })
        # A cadence table with a row for the column but none for ReadingDate,
        # which is where the household's grid is recorded.
        cadence = pd.DataFrame({
            "HuisIdBSV": pd.array([1], dtype="Int64"),
            "column": pd.array(["ElektriciteitNetgebruik"], dtype="string"),
            "cadence_minutes": pd.array([5], dtype="Int64"),
        })

        with pytest.raises(ValueError, match=r"Project\(s\) \[4\] resolve no grid"):
            calculate_average_diff_ibis(
                ibis.memtable(frame),
                "ProjectIdBSV",
                ["ElektriciteitNetgebruikDiff"],
                cadence=cadence,
            )