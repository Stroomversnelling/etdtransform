"""Rule bounds at each household's own cadence, in one frame.

``_row_bounds_per_cadence`` exists because a constraint package fixes one
cadence while households report at several, so the frame is resolved in groups
and each group narrowed with its own package.

The number that matters is the ratio. A reading covering twelve intervals is not
twelve times one interval's plausible maximum -- consumption averages out over
an hour -- so the bound grows as the square root of the ratio: 1.732x at fifteen
minutes and 3.464x at sixty, which is what
``docs/checks/imputation-coarse-cadence-findings-2026-10-17.md`` recorded.

"""

import pandas as pd
import pytest
from etdmap import ids
from etdmap.data_model import resolve_bounds, thresholds_dict
from etdmap.testing import cadence_sidecar, datasets_frame, household_batch_of
from test_constraint_package import _constraints

import etdtransform.vectorized_impute as vi

pytestmark = pytest.mark.unit

CUM = "ElektriciteitNetgebruik"
DIFF = f"{CUM}Diff"
N = 4


def _rows(household, minutes, n=N):
    """One household batch's rows, spaced at its own reporting interval."""
    return datasets_frame(
        {household_batch_of(household): {
            DIFF: [0.1] * n,
            # The rule's input, never known, so nothing narrows and the bound
            # is the scaled threshold alone -- which is what this file is
            # about.
            "K": [pd.NA] * n,
        }},
        start="2026-01-01",
        freq=f"{minutes}min",
        tz=None,
        # The bound is read from the project, the two ids and the rule's own
        # variables, so no cumulative counterpart is completed: a usable
        # cumulative column here would be a reading the rule never looks at.
        pairs=False,
    )


def _cadence(by_household):
    """A cadence table recording a different interval per household batch.

    Under the cumulative name, which is where cadence is recorded -- the
    interval column resolves through it. Every household batch sits on the five-minute
    grid and reports the column at its own interval, so the column is as coarse
    as or coarser than the grid and the ratio the bound scales by is the
    column's own.
    """
    return pd.concat(
        [
            cadence_sidecar(
                [CUM],
                minutes=minutes,
                grid_minutes=5,
                household_batch_id=household_batch_of(household),
            )
            for household, minutes in by_household.items()
        ],
        ignore_index=True,
    )


def _bounds(frame, by_household):
    return vi._row_bounds_per_cadence(
        frame,
        DIFF,
        _constraints((DIFF, "K", {"K": 1})),
        _cadence(by_household),
        {},
        physical_model_by_project={1: "Universeel"},
    )


def _declared_at(minutes):
    return resolve_bounds(thresholds_dict[DIFF], minutes, DIFF)[1]


class TestEachHouseholdTakesItsOwnCadence:
    def test_two_cadences_in_one_frame_give_two_bounds(self):
        frame = pd.concat([_rows(1, 5), _rows(2, 60)], ignore_index=True)
        _lo, hi, _imputed = _bounds(frame, {1: 5, 2: 60})

        fine = hi[frame[ids.HOUSEHOLD] == 1]
        coarse = hi[frame[ids.HOUSEHOLD] == 2]
        assert fine.nunique() == 1 and coarse.nunique() == 1
        assert float(coarse.iloc[0]) > float(fine.iloc[0])

    def test_the_hourly_bound_is_the_root_of_twelve_times_the_five_minute_one(
        self,
    ):
        frame = pd.concat([_rows(1, 5), _rows(2, 60)], ignore_index=True)
        _lo, hi, _imputed = _bounds(frame, {1: 5, 2: 60})

        fine = float(hi[frame[ids.HOUSEHOLD] == 1].iloc[0])
        coarse = float(hi[frame[ids.HOUSEHOLD] == 2].iloc[0])
        assert coarse / fine == pytest.approx(12**0.5, rel=1e-9)
        assert coarse / fine == pytest.approx(3.464, abs=1e-3)

    @pytest.mark.parametrize("minutes", [5, 15, 60, 1440])
    def test_each_bound_is_the_threshold_resolved_at_that_cadence(
        self, minutes
    ):
        """Against the data model rather than a number written here, so the
        test does not pin a threshold value that the metadata owns."""
        frame = _rows(1, minutes)
        _lo, hi, _imputed = _bounds(frame, {1: minutes})
        assert float(hi.iloc[0]) == pytest.approx(_declared_at(minutes))

    def test_three_cadences_are_each_resolved_separately(self):
        frame = pd.concat(
            [_rows(1, 5), _rows(2, 15), _rows(3, 60)], ignore_index=True
        )
        _lo, hi, _imputed = _bounds(frame, {1: 5, 2: 15, 3: 60})

        for household, minutes in ((1, 5), (2, 15), (3, 60)):
            own = hi[frame[ids.HOUSEHOLD] == household]
            assert float(own.iloc[0]) == pytest.approx(
                _declared_at(minutes)
            ), household

    def test_a_households_bound_does_not_move_when_a_neighbour_joins(self):
        """The grouping has to be per household, not per frame: a frame-wide
        cadence would make one household's bound depend on who else is in the
        chunk, which is how a coarse household's readings get nulled."""
        alone = _rows(1, 5)
        _lo, hi_alone, _i = _bounds(alone, {1: 5})

        together = pd.concat([_rows(1, 5), _rows(2, 60)], ignore_index=True)
        _lo, hi_together, _i = _bounds(together, {1: 5, 2: 60})

        assert float(hi_together[together[ids.HOUSEHOLD] == 1].iloc[0]) == (
            pytest.approx(float(hi_alone.iloc[0]))
        )

    def test_the_results_are_aligned_to_the_frames_index(self):
        frame = pd.concat([_rows(1, 5), _rows(2, 60)], ignore_index=True)
        lo, hi, imputed = _bounds(frame, {1: 5, 2: 60})
        for result in (lo, hi, imputed):
            assert list(result.index) == list(frame.index)


class TestAHouseholdTheTableDoesNotMention:
    def test_it_takes_the_declared_bound_unscaled(self):
        """The same resolution the column thresholds apply: unknown cadence
        means the bound is used as written, which is a decision the cadence
        table is meant to prevent and is logged rather than hidden."""
        frame = pd.concat([_rows(1, 5), _rows(2, 60)], ignore_index=True)
        _lo, hi, _imputed = _bounds(frame, {1: 5})

        unmentioned = hi[frame[ids.HOUSEHOLD] == 2]
        assert float(unmentioned.iloc[0]) == pytest.approx(
            float(thresholds_dict[DIFF]["Max"])
        )
