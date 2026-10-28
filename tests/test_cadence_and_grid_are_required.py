"""The cadence sidecar and the declared grid are read, never assumed.

Both set the spacing a threshold is judged at. Assuming five minutes for a
household that reports hourly judges its hour of consumption against a
five-minute bound, so the reading is nulled as out of bounds and its volume
leaves the household batch silently. The absence of either is therefore an error that
names the fix, not a default.
"""

import ibis
import pytest
from etdmap import ids
from etdmap.mapping_helpers import CADENCE_FILENAME
from etdmap.testing import (
    PAIRS_EMPTY,
    cadence_sidecar,
    datasets_frame,
    household_batch_of,
    index_frame,
    sidecar_frame,
)

from etdtransform.bounds import dataset_grid_minutes, load_column_cadence
from etdtransform.impute import calculate_average_diff_ibis

pytestmark = pytest.mark.unit

#: The household the artifacts below describe. Its household batch -- the key of the
#: index, of the cadence sidecar and of every lookup built from them -- is
#: ``household_batch_of(HOUSEHOLD)``.
HOUSEHOLD = 1

CUM = "ElektriciteitNetgebruik"
DIFF = f"{CUM}Diff"

#: The cadence the sidecar below declares, and the grid its ReadingDate row
#: carries with it.
CADENCE = 15


def _write_cadence(folder):
    cadence_sidecar(
        [CUM],
        minutes=CADENCE,
        household_batch_id=household_batch_of(HOUSEHOLD),
    ).to_parquet(folder / CADENCE_FILENAME, engine="pyarrow")


def _write_index(folder, frequency="5min"):
    index_frame(
        [household_batch_of(HOUSEHOLD)], Gegevensfrequentie=frequency
    ).to_parquet(folder / "index.parquet", engine="pyarrow")


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
        assert int(cadence["cadence_minutes"].iloc[0]) == CADENCE


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
        # The absent column is the subject, so it is dropped from the index
        # shape rather than the shape being built without it.
        index_frame([household_batch_of(HOUSEHOLD)]).drop(
            columns=["Gegevensfrequentie"]
        ).to_parquet(tmp_path / "index.parquet", engine="pyarrow")
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
        """An aggregated household batch spans household batches; its rows sit on the finest of
        their grids."""
        index_frame(
            [household_batch_of(1), household_batch_of(2)],
            Gegevensfrequentie=["15min", "5min"],
        ).to_parquet(tmp_path / "index.parquet", engine="pyarrow")
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
        frame = datasets_frame(
            {household_batch_of(HOUSEHOLD): {DIFF: [1.0, 1.0]}},
            project=4,
            start="2026-01-01",
            tz=None,
            # The averaging takes a household batch's volume from the cumulative column
            # wherever that column holds a value; here the interval column is
            # the source.
            pairs=PAIRS_EMPTY,
        )
        # A cadence table with a row for the column but none for ReadingDate,
        # which is where the household's grid is recorded.
        cadence = sidecar_frame(
            "cadence",
            [{"column": CUM, "cadence_minutes": 5}],
            household_batch_id=household_batch_of(HOUSEHOLD),
        )

        with pytest.raises(ValueError, match=r"Project\(s\) \[4\] resolve no grid"):
            calculate_average_diff_ibis(
                ibis.memtable(frame),
                ids.PROJECT,
                [DIFF],
                cadence=cadence,
            )
