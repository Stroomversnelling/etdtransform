"""Every household contributes to its project average.

A household is never excluded wholesale, and these tests pin the three
properties that requires:

- **Exclusion is per value, not per household.** One bad reading must cost only
  itself; the per-value threshold discard removes it and keeps the rest of that
  household's rows.
- **The bound that decides is the declared one.** A relative rule measured
  against the other households in the project is far looser: on
  `WatergebruikWarmTapwaterDiff` twice the 95th percentile of the maxima sits at
  8,579 against a declared Max of 100, so it admits 41 of the 42 households
  that are genuinely out of bounds.
- **A systemic error must still be caught.** If every household reports the
  same impossible value then every maximum equals it, so any rule comparing a
  household against its peers admits all of them.

Fixture note: `ElektriciteitNetgebruikDiff` declares a maximum of 1.5. Any test
value above that is discarded per value before averaging, which is correct but
turns a test of household eligibility into a test of the threshold. Keep
ordinary values well under it and make the out-of-bounds ones obviously
extreme.
"""

import pandas as pd
import pytest
from conftest import cadence_table, project_averages
from etdmap import ids
from etdmap.testing import PAIRS_EMPTY, datasets_frame, household_batch_of

pytestmark = pytest.mark.unit

CUM = "ElektriciteitNetgebruik"
DIFF = f"{CUM}Diff"
N = 12


def _household(household, values, project=1):
    """One household batch's rows, holding *values*.

    ``PAIRS_EMPTY``: the cumulative counterpart is present -- a frame holding
    half a pair is not a household artifact -- and empty, so the averaging
    takes the household batch's volume from the values stated here.
    """
    return datasets_frame(
        {household_batch_of(household): {DIFF: values}},
        start="2026-01-01",
        freq="5min",
        tz=None,
        project=project,
        pairs=PAIRS_EMPTY,
    )


def _average(frame):
    result = project_averages(
        frame, ids.PROJECT, [DIFF],
        cadence=cadence_table(frame, [CUM]), grid_minutes=5,
        # These frames hold a few households so each one's contribution is
        # readable; the household floor is about project size and is tested at
        # its production value in test_minimum_household_presence.
        min_eligible=1,
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
