"""Resampling a household's readings onto the grid.

Two consecutive usable readings bound a window. Its volume is spread evenly
over the grid slots it covers, ending at the later reading, because an interval
value at T covers (T_prev, T].

A household already reporting at the grid cadence has one-slot windows, so
resampling returns its interval column unchanged. That equivalence is what makes
the rule safe to apply to every household rather than only to coarse ones.
"""

import pandas as pd
import pytest

from etdmap.flags import MappingFlag, flag_column
from etdtransform.bounds import resample_to_grid, usable_readings

pytestmark = pytest.mark.unit

CUM = "ElektriciteitNetgebruik"
IV = "ElektriciteitNetgebruikDiff"


def _cadence(minutes, household=1, column=CUM):
    return pd.DataFrame({
        "HuisIdBSV": pd.array([household], dtype="Int64"),
        "column": pd.array([column], dtype="string"),
        "cadence_minutes": pd.array([minutes], dtype="Int64"),
    })


def _frame(cumulative, household=1, flags=None, interval=None):
    n = len(cumulative)
    frame = pd.DataFrame({
        "HuisIdBSV": pd.array([household] * n, dtype="Int64"),
        "ReadingDate": pd.date_range("2026-01-01", periods=n, freq="5min"),
        CUM: pd.array(cumulative, dtype="Float64"),
    })
    frame[IV] = frame[CUM].diff().astype("Float64")
    if interval is not None:
        frame[IV] = pd.array(interval, dtype="Float64")
    if flags is not None:
        frame[flag_column(CUM)] = pd.array(flags, dtype="Int64")
    return frame


def _out(frame, cadence):
    return resample_to_grid(frame, IV, cadence, grid_minutes=5).astype("Float64")


class TestGridCadenceIsIdentity:
    def test_a_grid_household_resamples_to_its_interval_column(self):
        frame = _frame([100.0, 101.0, 103.0, 106.0, 110.0])
        out = _out(frame, _cadence(5))
        # Row 0 has no predecessor, so no window ends there.
        pd.testing.assert_series_equal(
            out.iloc[1:], frame[IV].iloc[1:], check_names=False
        )

    def test_the_first_reading_contributes_nothing(self):
        out = _out(_frame([100.0, 101.0, 103.0]), _cadence(5))
        assert pd.isna(out.iloc[0])


class TestCoarseColumn:
    """The case M exists for: the interval column is empty because .diff()
    crosses pd.NA, and the volume has to come from the cumulative."""

    @staticmethod
    def _coarse():
        # Readings every 15 minutes on a 5-minute grid.
        cumulative = [100.0, pd.NA, pd.NA, 106.0, pd.NA, pd.NA, 115.0, pd.NA, pd.NA]
        return _frame(cumulative)

    def test_the_interval_column_really_is_empty(self):
        frame = self._coarse()
        assert int(frame[IV].notna().sum()) == 0

    def test_each_window_is_spread_backward_over_its_slots(self):
        out = _out(self._coarse(), _cadence(15))
        # 6.0 over (00:00, 00:15] lands on 00:05, 00:10, 00:15 -- not 00:00.
        assert [None if pd.isna(v) else round(float(v), 6) for v in out] == [
            None, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, None, None
        ]

    def test_volume_is_conserved(self):
        out = _out(self._coarse(), _cadence(15))
        assert float(out.sum()) == pytest.approx(15.0)


class TestOutageWindows:
    def test_a_window_longer_than_one_interval_contributes_nothing(self):
        # Readings at rows 0 and 6: six slots, but the column reports every 15
        # minutes (three slots), so this spans an outage.
        cumulative = [100.0] + [pd.NA] * 5 + [130.0] + [pd.NA] * 2
        out = _out(_frame(cumulative), _cadence(15))
        assert out.isna().all()

    def test_good_windows_survive_beside_an_outage(self):
        cumulative = [100.0, pd.NA, pd.NA, 106.0] + [pd.NA] * 5 + [200.0]
        out = _out(_frame(cumulative), _cadence(15))
        assert [None if pd.isna(v) else float(v) for v in out] == [
            None, 2.0, 2.0, 2.0, None, None, None, None, None, None
        ]


class TestFlagsExcludeReadings:
    def test_a_flagged_reading_leaves_a_window_that_spans_an_outage(self):
        """Dropping a reading widens the window across it. For a column
        reporting every 5 minutes that window is 2 slots -- longer than one
        reporting interval -- so it is an outage and contributes nothing."""
        flags = [0, 0, MappingFlag.FROZEN_SPAN.value, 0, 0]
        frame = _frame([100.0, 101.0, 102.0, 103.0, 104.0], flags=flags)
        out = _out(frame, _cadence(5))
        assert [None if pd.isna(v) else float(v) for v in out] == [
            None, 1.0, None, None, 1.0
        ]

    def test_usable_readings_rejects_every_disqualifying_flag(self):
        for flag in (
            MappingFlag.FILLED_DOWN, MappingFlag.FILLED_ZERO,
            MappingFlag.FROZEN_SPAN, MappingFlag.CATCHUP,
            MappingFlag.CUMULATIVE_METER_RESET, MappingFlag.OUT_OF_BOUNDS_HIGH,
        ):
            frame = _frame([100.0, 101.0], flags=[0, flag.value])
            assert usable_readings(frame, CUM).tolist() == [True, False], flag.name

    def test_a_column_without_a_findings_column_is_all_usable(self):
        frame = _frame([100.0, 101.0])
        assert flag_column(CUM) not in frame.columns
        assert usable_readings(frame, CUM).all()


class TestSourceSelection:
    def test_a_delivered_interval_column_is_used_when_the_cumulative_is_empty(self):
        frame = _frame([pd.NA] * 6, interval=[pd.NA, pd.NA, 6.0, pd.NA, pd.NA, 9.0])
        out = _out(frame, _cadence(15, column=IV))
        # A delivered interval value already is its window's volume, and the
        # window is its cadence however long ago the last reading arrived. So it
        # needs no predecessor and the first reading counts.
        assert [None if pd.isna(v) else float(v) for v in out] == [
            2.0, 2.0, 2.0, 3.0, 3.0, 3.0
        ]
        assert float(out.sum()) == pytest.approx(15.0)

    def test_the_cumulative_wins_when_both_are_present(self):
        frame = _frame([100.0, 101.0, 103.0])
        frame[IV] = pd.array([99.0, 99.0, 99.0], dtype="Float64")
        out = _out(frame, _cadence(5))
        assert [None if pd.isna(v) else float(v) for v in out] == [None, 1.0, 2.0]


class TestMultipleHouseholds:
    def test_households_do_not_leak_into_each_other(self):
        a = _frame([100.0, 101.0, 103.0], household=1)
        b = _frame([500.0, 505.0, 515.0], household=2)
        frame = pd.concat([a, b], ignore_index=True)
        cadence = pd.concat([_cadence(5, 1), _cadence(5, 2)], ignore_index=True)
        out = _out(frame, cadence)
        assert [None if pd.isna(v) else float(v) for v in out] == [
            None, 1.0, 2.0, None, 5.0, 10.0
        ]
