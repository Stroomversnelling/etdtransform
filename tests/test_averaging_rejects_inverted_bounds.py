"""An inverted bound stops the averaging run.

A column's floor and its ceiling are scaled to a reporting cadence by different
rules: the floor in proportion to the interval, the peak by the shape the
threshold declares. A pair that holds at the declared interval can therefore
come out inverted at a coarser one, and a floor above the ceiling admits no
value at all -- every reading of that column would be discarded as out of
bounds and the average would be empty.

That is a contradiction between metadata rows, not a condition of the data, so
both averaging paths stop instead of averaging nothing.
"""

import logging

import ibis
import pandas as pd
import pytest

from etdmap.data_model import thresholds_dict
from etdtransform.impute import calculate_average_diff_ibis
from conftest import cadence_table, project_averages

pytestmark = pytest.mark.unit

CUM = "WatergebruikWarmTapwater"
DIFF = f"{CUM}Diff"

#: Declared bounds that hold at the five-minute interval they are declared
#: against -- 10 below 20 -- and invert at sixty minutes, where the floor
#: reaches 120 and the peak only 69.28.
INVERTING = {
    "Min": 10.0,
    "Max": 20.0,
    "ThresholdType": "5-minute",
    "UpperThresholdSustained": pd.NA,
    "UpperThresholdExponent": pd.NA,
}

INVERTED_AT_MINUTES = 60


@pytest.fixture(autouse=True)
def _quiet_logs():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _frame(households=(1, 2), n=8):
    """Two households of ordinary readings on the grid, one slot unmeasured."""
    dates = list(pd.date_range("2026-01-01", periods=n, freq="5min"))
    rows = len(households) * n
    values = [3.0] * rows
    values[1] = pd.NA

    ids = [household for household in households for _ in dates]
    return pd.DataFrame({
        "HuisIdBSV": pd.array(ids, dtype="Int64"),
        "HuisBatchIdBSV": pd.array(ids, dtype="Int64"),
        "ProjectIdBSV": pd.array([1] * rows, dtype="Int64"),
        "ReadingDate": dates * len(households),
        DIFF: pd.array(values, dtype="Float64"),
    })


def _average_pandas(df, cadence_minutes):
    return project_averages(
        df,
        "ProjectIdBSV",
        [DIFF],
        cadence=cadence_table(df, [CUM], minutes=cadence_minutes),
        grid_minutes=5,
    )


def _average_ibis(df, cadence_minutes):
    return calculate_average_diff_ibis(
        ibis.memtable(df),
        "ProjectIdBSV",
        [DIFF],
        cadence=cadence_table(df, [CUM], minutes=cadence_minutes),
        grid_minutes=5,
    )


PATHS = {"pandas": _average_pandas, "ibis": _average_ibis}


@pytest.fixture
def inverting_threshold(monkeypatch):
    """Declare bounds for DIFF that invert once scaled to a coarse cadence."""
    monkeypatch.setitem(thresholds_dict, DIFF, dict(INVERTING))


class TestAnInvertedBoundStopsBothPaths:
    @pytest.mark.parametrize("path", sorted(PATHS))
    def test_it_raises(self, path, inverting_threshold):
        with pytest.raises(ValueError) as raised:
            PATHS[path](_frame(), INVERTED_AT_MINUTES)

        message = str(raised.value)
        assert DIFF in message
        assert "120" in message
        assert "69.28" in message
        assert str(INVERTED_AT_MINUTES) in message
        assert "inconsistency" in message

    @pytest.mark.parametrize("path", sorted(PATHS))
    def test_the_message_names_both_scaled_bounds_in_order(
        self, path, inverting_threshold
    ):
        with pytest.raises(ValueError) as raised:
            PATHS[path](_frame(), INVERTED_AT_MINUTES)

        message = str(raised.value)
        assert message.index("120") < message.index("69.28")


class TestNormalBoundsAreAveraged:
    @pytest.mark.parametrize("path", sorted(PATHS))
    @pytest.mark.parametrize("cadence_minutes", [5, 15, 60])
    def test_the_declared_thresholds_do_not_raise(self, path, cadence_minutes):
        """The committed thresholds hold at every cadence the data reports at."""
        result = PATHS[path](_frame(), cadence_minutes)
        assert DIFF in result

    @pytest.mark.parametrize("path", sorted(PATHS))
    def test_the_same_pair_is_accepted_at_its_own_interval(
        self, path, inverting_threshold
    ):
        """Only the scaling inverts this pair: at five minutes it is 10 below 20."""
        result = PATHS[path](_frame(), 5)
        assert DIFF in result
