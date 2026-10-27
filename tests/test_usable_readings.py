"""Which readings of a column count as measurements of it.

The averaging and the fill both ask this before they use a value: a reading the
mapping stage wrote, blanked or disqualified is not evidence of what the meter
reported, so it must not shape a project average or anchor a gap.

The predicate reads the mapping findings column beside the data column. A column
with no findings column has had nothing recorded against it, so every value in
it counts.
"""

import pandas as pd
import pytest
from etdmap.flags import MappingFlag, flag_column

from etdtransform.bounds import usable_readings

pytestmark = pytest.mark.unit

CUM = "ElektriciteitNetgebruik"


def _frame(values, flags=None):
    n = len(values)
    frame = pd.DataFrame({
        "HuisIdBSV": pd.array([1] * n, dtype="Int64"),
        "HuisBatchIdBSV": pd.array([value + 1000 for value in [1] * n], dtype="Int64"),
        "ReadingDate": pd.date_range("2026-01-01", periods=n, freq="5min"),
        CUM: pd.array(values, dtype="Float64"),
    })
    if flags is not None:
        frame[flag_column(CUM)] = pd.array(flags, dtype="Int64")
    return frame


class TestWhatDisqualifiesAReading:
    def test_every_disqualifying_finding_is_rejected(self):
        for flag in (
            MappingFlag.FILLED_DOWN, MappingFlag.FILLED_ZERO,
            MappingFlag.FROZEN_SPAN, MappingFlag.CATCHUP,
            MappingFlag.CUMULATIVE_METER_RESET, MappingFlag.OUT_OF_BOUNDS_HIGH,
        ):
            frame = _frame([100.0, 101.0], flags=[0, flag.value])
            assert usable_readings(frame, CUM).tolist() == [True, False], flag.name

    def test_a_column_without_a_findings_column_is_all_usable(self):
        """Nothing was recorded against these values, which is different from
        something having been recorded and found acceptable."""
        frame = _frame([100.0, 101.0])
        assert flag_column(CUM) not in frame.columns
        assert usable_readings(frame, CUM).all()

    def test_an_absent_value_is_not_a_usable_reading(self):
        frame = _frame([100.0, pd.NA, 102.0])
        assert usable_readings(frame, CUM).tolist() == [True, False, True]

    def test_a_delivery_signal_does_not_disqualify_a_reading(self):
        """RAW_DELIVERED_VALUE records that the supplier sent this value, which
        is the opposite of a reason to exclude it."""
        frame = _frame(
            [100.0, 101.0],
            flags=[MappingFlag.RAW_DELIVERED_VALUE.value] * 2,
        )
        assert usable_readings(frame, CUM).all()
