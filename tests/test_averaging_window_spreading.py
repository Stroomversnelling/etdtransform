"""How the averaging spreads a window, slot by slot.

Each window's volume is divided by the slots it covers and placed on them,
ending at the reading that closes it. The implementation expands each window
against a small table of offsets, because DuckDB does not expose IGNORE NULLS
through ibis 9.5.

The frames here are a handful of rows with the expected values spelled out, so
``spread_average`` calls the spreading with a one-slot window -- the presence
gates would otherwise exclude a frame this short wholesale. Those gates are
tested in test_minimum_household_presence.py, and the same spreading over a
realistic span in test_averaging_coarse_delivery.py.
"""

import ibis
import pandas as pd
import pytest

from conftest import spread_average

pytestmark = pytest.mark.unit

CUM = "ElektriciteitNetgebruik"
IV = "ElektriciteitNetgebruikDiff"
GRID = 5


def _batch(household):
    """The dataset id a frame for *household* carries.

    A household can be delivered as several datasets, so the two ids are
    separate numbers: the cadence, the readings and the sidecar rows belong to
    the dataset, and the household is what it is counted under.
    """
    return household + 1000


def _average(frame, cadence, drop_na=True):
    """The spread, per slot, as a list with None for an empty slot."""
    avg, _, _ = spread_average(frame, cadence, IV, GRID, drop_na=drop_na)
    avg = avg.sort_values("ReadingDate")
    return [
        None if pd.isna(v) else round(float(v), 10)
        for v in avg[f"{IV}_avg"]
    ]


def _frame(per_household):
    """per_household: {HuisIdBSV: [cumulative readings, pd.NA where absent]}

    One dataset per household, carrying both ids: the readings are the
    dataset's, and the household is the one it belongs to.
    """
    parts = []
    for household, readings in per_household.items():
        frame = pd.DataFrame({
            "HuisIdBSV": pd.array([household] * len(readings), dtype="Int64"),
            "HuisBatchIdBSV": pd.array(
                [_batch(household)] * len(readings), dtype="Int64"
            ),
            "ProjectIdBSV": pd.array([1] * len(readings), dtype="Int64"),
            "ReadingDate": pd.date_range(
                "2026-01-01", periods=len(readings), freq=f"{GRID}min"
            ),
            CUM: pd.array(readings, dtype="Float64"),
        })
        frame[IV] = frame[CUM].diff().astype("Float64")
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def _cadence(minutes_by_household):
    """A cadence row per household's dataset, declaring what its readings cover.

    Cadence is a property of the dataset, so the rows are keyed on
    ``HuisBatchIdBSV``; the household each belongs to rides along.
    """
    return pd.DataFrame({
        "HuisIdBSV": pd.array(list(minutes_by_household), dtype="Int64"),
        "HuisBatchIdBSV": pd.array(
            [_batch(household) for household in minutes_by_household],
            dtype="Int64",
        ),
        "column": pd.array([CUM] * len(minutes_by_household), dtype="string"),
        "cadence_minutes": pd.array(
            list(minutes_by_household.values()), dtype="Int64"
        ),
    })


class TestHowAWindowIsSpread:
    def test_a_grid_cadence_household_keeps_its_own_intervals(self):
        """One-slot windows, so the spread is the interval column itself. Slot 0
        has no predecessor to difference against and reports nothing."""
        frame = _frame({1: [100.0, 101.0, 103.0, 106.0, 110.0]})
        assert _average(frame, _cadence({1: 5})) == pytest.approx(
            [1.0, 2.0, 3.0, 4.0]
        )

    def test_a_coarse_household_divides_each_rise_by_its_slots(self):
        """Rises of 6 and 9 over three slots each, placed on the slots they
        cover and ending at the reading that closes them."""
        frame = _frame({
            1: [100.0, pd.NA, pd.NA, 106.0, pd.NA, pd.NA, 115.0, pd.NA, pd.NA]
        })
        assert _average(frame, _cadence({1: 15})) == pytest.approx(
            [2.0, 2.0, 2.0, 3.0, 3.0, 3.0]
        )

    def test_a_mixed_cadence_project_averages_what_each_contributes(self):
        """The case production data does not contain and the method exists for.
        Slot 00:15 averages household 1's 3.0 with household 2's 12/3 = 4.0."""
        frame = _frame({
            1: [100.0, 101.0, 103.0, 106.0, 107.0, 108.0, 110.0, 111.0, 112.0],
            2: [500.0, pd.NA, pd.NA, 512.0, pd.NA, pd.NA, 530.0, pd.NA, pd.NA],
        })
        avg, _, _ = spread_average(frame, _cadence({1: 5, 2: 15}), IV, GRID)
        row = avg[avg["ReadingDate"] == pd.Timestamp("2026-01-01 00:15")]
        assert float(row[f"{IV}_avg"].iloc[0]) == pytest.approx(3.5)

    def test_an_outage_window_contributes_nothing(self):
        """A window longer than the column's reporting interval spans an outage.
        Distributing measured consumption across it is imputation's work."""
        frame = _frame({1: [100.0] + [pd.NA] * 5 + [130.0] + [pd.NA] * 2})
        assert _average(frame, _cadence({1: 15})) == []

    def test_three_cadences_in_one_project(self):
        """Per slot: household 1 at the grid, household 2 at 12/3 = 4.0 then
        18/3 = 6.0, household 3 at 30/6 = 5.0 across its single window. The last
        two slots have household 1 alone."""
        frame = _frame({
            1: [100.0, 101.0, 103.0, 106.0, 107.0, 108.0, 110.0, 111.0, 112.0],
            2: [500.0, pd.NA, pd.NA, 512.0, pd.NA, pd.NA, 530.0, pd.NA, pd.NA],
            3: [900.0, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 930.0, pd.NA, pd.NA],
        })
        assert _average(frame, _cadence({1: 5, 2: 15, 3: 30})) == pytest.approx(
            [10 / 3, 11 / 3, 4.0, 4.0, 4.0, 13 / 3, 1.0, 1.0]
        )


class TestCoverageAndDuty:
    def test_a_column_that_delivers_nothing_has_zero_coverage(self):
        frame = _frame({1: [pd.NA] * 6})
        _, coverage, duty = spread_average(frame, _cadence({1: 5}), IV, GRID)
        assert coverage == 0.0
        assert duty == 0.0

    def test_a_coarse_column_now_has_coverage(self):
        """What M is for: this column had no average at all before."""
        frame = _frame({
            1: [100.0, pd.NA, pd.NA, 106.0, pd.NA, pd.NA, 115.0, pd.NA, pd.NA]
        })
        _, coverage, _ = spread_average(frame, _cadence({1: 15}), IV, GRID)
        assert coverage > 0.0

    def test_duty_is_zero_for_a_column_that_never_advances(self):
        frame = _frame({1: [100.0, 100.0, 100.0, 100.0]})
        _, coverage, duty = spread_average(frame, _cadence({1: 5}), IV, GRID)
        assert coverage > 0.0
        assert duty == 0.0


class TestWhichFormSuppliesTheVolume:
    def test_a_delivered_interval_is_its_own_window_volume(self):
        """No register at all, so the interval column supplies the volume and is
        divided by the slots it covers rather than differenced first. It needs
        no predecessor, so its first reading counts."""
        frame = _frame({1: [pd.NA] * 6})
        frame[IV] = pd.array([pd.NA, pd.NA, 6.0, pd.NA, pd.NA, 9.0], dtype="Float64")
        cadence = _cadence({1: 15})
        cadence["column"] = pd.array([IV], dtype="string")
        assert _average(frame, cadence) == pytest.approx(
            [2.0, 2.0, 2.0, 3.0, 3.0, 3.0]
        )

    def test_one_household_on_each_source(self):
        frame = _frame({
            1: [100.0, pd.NA, pd.NA, 106.0, pd.NA, pd.NA, 115.0, pd.NA, pd.NA],
            2: [pd.NA] * 9,
        })
        frame.loc[frame["HuisIdBSV"] == 2, IV] = pd.array(
            [pd.NA, pd.NA, 3.0, pd.NA, pd.NA, 6.0, pd.NA, pd.NA, 9.0],
            dtype="Float64",
        )
        cadence = pd.concat([
            _cadence({1: 15}),
            pd.DataFrame({
                "HuisIdBSV": pd.array([2], dtype="Int64"),
                "HuisBatchIdBSV": pd.array([_batch(2)], dtype="Int64"),
                "column": pd.array([IV], dtype="string"),
                "cadence_minutes": pd.array([15], dtype="Int64"),
            }),
        ], ignore_index=True)
        # Both households report coarsely, by different forms, in one project.
        # Household 1's register gives windows of 6/3 = 2.0 on slots 1-3 and
        # 9/3 = 3.0 on slots 4-6, with slots 0, 7 and 8 bounded by no second
        # reading. Household 2's delivered intervals give 3/3 = 1.0 on slots
        # 0-2, 6/3 = 2.0 on 3-5 and 9/3 = 3.0 on 6-8, needing no predecessor.
        # Each slot averages whoever covered it.
        assert _average(frame, cadence) == pytest.approx(
            [1.0, 1.5, 1.5, 2.0, 2.5, 2.5, 3.0, 3.0, 3.0]
        )


class TestUncoveredSlotsAreRowsNotAbsences:
    """A slot no window covers must come back as a row carrying pd.NA, not be
    dropped.

    Grouping over the expanded windows alone loses those slots, and a missing
    row is not the same as a row with no value: the merge downstream is on
    (project, ReadingDate), so an absent row silently changes what imputation
    sees. This is what the real fixtures caught and the synthetic ones did not.
    """

    @staticmethod
    def _with_an_uncovered_stretch():
        # A reading at row 0 and another at row 7, with the column reporting
        # every 15 minutes. The window between them is too long, so rows 1-7 are
        # covered by nothing at all.
        return _frame({1: [100.0, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 130.0, pd.NA]})

    def test_a_row_comes_back_for_every_slot(self):
        frame = self._with_an_uncovered_stretch()
        avg, _, _ = spread_average(
            frame, _cadence({1: 15}), IV, GRID, drop_na=False
        )
        assert len(avg) == frame["ReadingDate"].nunique()

    def test_the_uncovered_slots_carry_na(self):
        frame = self._with_an_uncovered_stretch()
        avg, _, _ = spread_average(
            frame, _cadence({1: 15}), IV, GRID, drop_na=False
        )
        assert avg[f"{IV}_avg"].isna().all()
