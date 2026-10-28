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
from conftest import cadence_table, project_averages
from etdmap import ids
from etdmap.data_model import cumulative_column
from etdmap.testing import (
    PAIRS_EMPTY,
    datasets_frame,
    declare_synthetic_model,
    household_batch_of,
    synthetic_model,
)

from etdtransform.impute import calculate_average_diff_ibis

pytestmark = pytest.mark.unit

CUM = "WatergebruikWarmTapwater"
DIFF = f"{CUM}Diff"

#: Declared bounds that hold at the five-minute interval they are declared
#: against -- 10 below 20 -- and invert at sixty minutes, where the floor
#: reaches 120 and the peak only 69.28.
INVERTING_MIN = 10.0
INVERTING_MAX = 20.0

INVERTED_AT_MINUTES = 60


@pytest.fixture(autouse=True)
def _quiet_logs():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


@pytest.fixture
def inverting_model():
    """A model whose interval columns carry bounds that invert once scaled to a
    coarse cadence.

    The bounds are written against five minutes and declare neither a sustained
    level nor an exponent, so at sixty minutes the floor reaches twelve times
    its declared value while the peak grows by the square root of twelve.
    """
    with declare_synthetic_model(
        synthetic_model(maximum=INVERTING_MAX, minimum=INVERTING_MIN)
    ) as declared:
        yield declared


def _frame(column, households=(1, 2), n=8):
    """Two households of ordinary readings on the grid, one slot unmeasured.

    ``PAIRS_EMPTY``: *column*'s cumulative counterpart is present -- a frame
    holding half a pair is not a household artifact -- and empty, so the
    averaging takes each household batch's volume from the values stated here.
    """
    values = [3.0] * (len(households) * n)
    values[1] = pd.NA
    return datasets_frame(
        {
            household_batch_of(household): {
                column: values[position * n:(position + 1) * n]
            }
            for position, household in enumerate(households)
        },
        start="2026-01-01",
        freq="5min",
        tz=None,
        pairs=PAIRS_EMPTY,
    )


def _average_pandas(df, column, cadence_minutes):
    return project_averages(
        df,
        ids.PROJECT,
        [column],
        cadence=cadence_table(
            df, [cumulative_column(column)], minutes=cadence_minutes
        ),
        grid_minutes=5,
    )


def _average_ibis(df, column, cadence_minutes):
    return calculate_average_diff_ibis(
        ibis.memtable(df),
        ids.PROJECT,
        [column],
        cadence=cadence_table(
            df, [cumulative_column(column)], minutes=cadence_minutes
        ),
        grid_minutes=5,
    )


PATHS = {"pandas": _average_pandas, "ibis": _average_ibis}


class TestAnInvertedBoundStopsBothPaths:
    @pytest.mark.parametrize("path", sorted(PATHS))
    def test_it_raises(self, path, inverting_model):
        column = inverting_model.device_complement
        with pytest.raises(ValueError) as raised:
            PATHS[path](_frame(column), column, INVERTED_AT_MINUTES)

        message = str(raised.value)
        assert column in message
        assert "120" in message
        assert "69.28" in message
        assert str(INVERTED_AT_MINUTES) in message
        assert "inconsistency" in message

    @pytest.mark.parametrize("path", sorted(PATHS))
    def test_the_message_names_both_scaled_bounds_in_order(
        self, path, inverting_model
    ):
        column = inverting_model.device_complement
        with pytest.raises(ValueError) as raised:
            PATHS[path](_frame(column), column, INVERTED_AT_MINUTES)

        message = str(raised.value)
        assert message.index("120") < message.index("69.28")


class TestNormalBoundsAreAveraged:
    @pytest.mark.parametrize("path", sorted(PATHS))
    @pytest.mark.parametrize("cadence_minutes", [5, 15, 60])
    def test_the_declared_thresholds_do_not_raise(self, path, cadence_minutes):
        """The committed thresholds hold at every cadence the data reports at.

        The one test here whose subject is the shipped metadata, so it names a
        real variable rather than declaring a model of its own.
        """
        result = PATHS[path](_frame(DIFF), DIFF, cadence_minutes)
        assert DIFF in result

    @pytest.mark.parametrize("path", sorted(PATHS))
    def test_the_same_pair_is_accepted_at_its_own_interval(
        self, path, inverting_model
    ):
        """Only the scaling inverts this pair: at five minutes it is 10 below 20."""
        column = inverting_model.device_complement
        result = PATHS[path](_frame(column), column, 5)
        assert column in result
