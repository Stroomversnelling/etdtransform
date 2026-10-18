"""Plan group K: every household contributes to its project average.

The relative filter that used to run here excluded a household whose maximum
reached twice the 95th percentile of the project's household maxima. Three
things were wrong with it, and they are what these tests pin:

- **All-or-nothing per household.** One bad reading cost every good row that
  household had. The per-value threshold discard removes the bad value and keeps
  the rest.
- **Far looser than the declared bound.** Measured at 86x on
  `WatergebruikWarmTapwaterDiff` (8,579 against a Max of 100) and 17.6x on
  `ElektriciteitsgebruikBoosterDiff`. It excluded 1 household of the 42 and 20
  that were genuinely out of bounds.
- **Blind to a systemic error.** If every household reports the same impossible
  value, every maximum equals it, the 95th percentile equals it, and
  `max < 2 x max` includes everyone.

Fixture note: `ElektriciteitNetgebruikDiff` declares a maximum of 1.5. Any test
value above that is discarded per value before averaging, which is correct but
turns a test of the household filter into a test of the threshold. Keep ordinary
values well under it and make the out-of-bounds ones obviously extreme.
"""

import pandas as pd
import pytest

from conftest import cadence_table, project_averages

pytestmark = pytest.mark.unit

CUM = "ElektriciteitNetgebruik"
DIFF = f"{CUM}Diff"
N = 12


def _household(household, values, project=1):
    return pd.DataFrame({
        "HuisIdBSV": pd.array([household] * len(values), dtype="Int64"),
        "ProjectIdBSV": pd.array([project] * len(values), dtype="Int64"),
        "ReadingDate": pd.date_range("2026-01-01", periods=len(values), freq="5min"),
        DIFF: pd.array(values, dtype="Float64"),
    })


def _average(frame):
    result = project_averages(
        frame, "ProjectIdBSV", [DIFF],
        cadence=cadence_table(frame, [CUM]), grid_minutes=5,
    )
    return result[DIFF]["avg_diff"]


class TestOneBadReadingDoesNotCostAHousehold:
    """The case the old filter got most wrong."""

    @staticmethod
    def _frames():
        ordinary = [_household(h, [0.5] * N, project=1) for h in (1, 2, 3, 4)]
        # Household 5 is ordinary except for one enormous reading.
        spiking = [0.5] * N
        spiking[6] = 900.0
        return pd.concat(ordinary + [_household(5, spiking)], ignore_index=True)

    def test_the_spiking_household_still_contributes_its_good_rows(self):
        avg = _average(self._frames())
        # At a slot where household 5 reported normally, all five contribute and
        # the average is their common value.
        at_slot_0 = avg.loc[avg["ReadingDate"] == pd.Timestamp("2026-01-01 00:00")]
        assert float(at_slot_0[f"{DIFF}_avg"].iloc[0]) == pytest.approx(0.5)

    def test_the_bad_value_itself_is_discarded(self):
        """Per value, not per household: the spike goes, the row's slot keeps
        the other four households."""
        avg = _average(self._frames())
        at_spike = avg.loc[avg["ReadingDate"] == pd.Timestamp("2026-01-01 00:30")]
        assert float(at_spike[f"{DIFF}_avg"].iloc[0]) == pytest.approx(0.5)


class TestASystemicErrorIsNotHiddenByRelativeComparison:
    def test_every_household_reporting_the_same_impossible_value(self):
        """The old filter could never catch this: every maximum equals the same
        impossible number, so nothing is relatively extreme. The declared bound
        does not care what the neighbours did."""
        frame = pd.concat(
            [_household(h, [900.0] * N) for h in (1, 2, 3, 4, 5)],
            ignore_index=True,
        )
        avg = _average(frame)
        values = avg[f"{DIFF}_avg"]
        assert values.isna().all(), (
            "an impossible value reported by every household must still be "
            "discarded"
        )


class TestAllHouseholdsAreCounted:
    def test_no_household_is_excluded_from_the_average(self):
        """Values stay under the 1.5 maximum on purpose.

        Anything above it is discarded per value, which is correct but would
        make this test measure the threshold rather than the household filter.
        """
        frame = pd.concat(
            [_household(h, [h / 10] * N) for h in (1, 2, 3, 4, 5)],
            ignore_index=True,
        )
        avg = _average(frame)
        # 0.1 + 0.2 + 0.3 + 0.4 + 0.5 over five households.
        at_slot_0 = avg.loc[avg["ReadingDate"] == pd.Timestamp("2026-01-01 00:00")]
        assert float(at_slot_0[f"{DIFF}_avg"].iloc[0]) == pytest.approx(0.3)
