"""Anchored fills clip to the row bound and the volume moves within the gap.

The jump is a measurement, so what a clip removes from one cell belongs in the
gap's others: placed in one pass, in proportion to remaining headroom, with
zero-profile cells held in reserve for any remainder. An unanchored fill has no
measured total, so its clip gives nothing back.
"""

import pandas as pd
import pytest
from test_constraint_package import _constraints
from test_row_bounds_in_the_fill import DIFF, FOLLOWS_K, _frame, _impute

from etdtransform.vectorized_impute import ImputeType

pytestmark = pytest.mark.unit

ITYPE = f"{DIFF}_impute_type"


def _flags(out, row):
    value = out.loc[row, ITYPE]
    return 0 if pd.isna(value) else int(value)


class TestTheCeilingDirection:
    def test_clipped_volume_moves_in_proportion_to_headroom(self):
        # K pins row 3 to [1, 1]. Its average clips 1.5 -> 1.0, the scaled
        # fill lands [0.6, 1.2, 1.2], and the clip to 1.0 leaves 0.2 to place
        # against headrooms [0.9, 0.3] -- so 0.15 and 0.05.
        out = _impute(
            _frame(
                [100.0, pd.NA, pd.NA, 103.0],
                [pd.NA, 0.5, 1.0, 1.5],
                k=[pd.NA, pd.NA, pd.NA, 1.0],
            ),
            constraints=_constraints(*FOLLOWS_K),
        )
        assert out.loc[1:3, DIFF].astype(float).tolist() == pytest.approx(
            [0.75, 1.25, 1.0]
        )
        # The gap total is the measured jump, exactly.
        assert float(out.loc[1:3, DIFF].sum()) == pytest.approx(3.0)
        assert _flags(out, 3) & ImputeType.CLIPPED.value
        assert _flags(out, 1) & ImputeType.CLIPPED_REDISTRIBUTION.value
        assert _flags(out, 2) & ImputeType.CLIPPED_REDISTRIBUTION.value
        assert not _flags(out, 3) & ImputeType.CLIPPED_REDISTRIBUTION.value

    def test_a_uniform_fill_is_allocated_too(self):
        # No shape anywhere: the jump of 3 spreads evenly, row 3's cap of 0.4
        # frees 0.6, and the two open cells split it by their equal headroom.
        out = _impute(
            _frame(
                [100.0, pd.NA, pd.NA, 103.0],
                [pd.NA] * 4,
                k=[pd.NA, pd.NA, pd.NA, 0.4],
            ),
            constraints=_constraints(*FOLLOWS_K),
        )
        assert out.loc[1:3, DIFF].astype(float).tolist() == pytest.approx(
            [1.3, 1.3, 0.4]
        )
        assert _flags(out, 3) & ImputeType.CLIPPED.value
        for row in (1, 2):
            assert _flags(out, row) & ImputeType.UNIFORM_FILL.value
            assert _flags(out, row) & ImputeType.CLIPPED_REDISTRIBUTION.value
            assert not _flags(out, row) & ImputeType.CLIPPED_AVERAGE.value


class TestTheFloorDirection:
    def test_a_floor_clip_takes_volume_back_from_the_others(self):
        # K raises row 1's floor to 1.0. Its average clips 0.4 -> 1.0, the
        # scaled fill lands [0.8, 0.4, 1.2], the floor pushes 0.8 up to 1.0,
        # and the 0.2 comes back out of the others against headrooms
        # [0.4, 1.2] -- so 0.05 and 0.15.
        out = _impute(
            _frame(
                [100.0, pd.NA, pd.NA, 102.4],
                [pd.NA, 0.4, 0.5, 1.5],
                k=[pd.NA, 1.0, pd.NA, pd.NA],
            ),
            constraints=_constraints(*FOLLOWS_K),
        )
        assert out.loc[1:3, DIFF].astype(float).tolist() == pytest.approx(
            [1.0, 0.35, 1.05]
        )
        assert float(out.loc[1:3, DIFF].sum()) == pytest.approx(2.4)
        assert _flags(out, 1) & ImputeType.CLIPPED.value
        assert _flags(out, 2) & ImputeType.CLIPPED_REDISTRIBUTION.value
        assert _flags(out, 3) & ImputeType.CLIPPED_REDISTRIBUTION.value


class TestTheReserve:
    def test_zero_profile_cells_take_the_remainder_uniformly(self):
        # Rows 1 and 2 cap at 1.2 and there is no open shaped cell left, so
        # the 0.6 goes to the zero-profile cell, within its own bound.
        out = _impute(
            _frame(
                [100.0, pd.NA, pd.NA, 103.0],
                [pd.NA, 1.0, 1.0, 0.0],
                k=[pd.NA, 1.2, 1.2, pd.NA],
            ),
            constraints=_constraints(*FOLLOWS_K),
        )
        assert out.loc[1:3, DIFF].astype(float).tolist() == pytest.approx(
            [1.2, 1.2, 0.6]
        )
        assert float(out.loc[1:3, DIFF].sum()) == pytest.approx(3.0)
        assert _flags(out, 3) & ImputeType.UNIFORM_RESIDUAL.value
        assert not _flags(out, 3) & ImputeType.CLIPPED.value
        for row in (1, 2):
            assert _flags(out, row) & ImputeType.CLIPPED.value

    def test_a_gap_that_cannot_meet_its_jump_stays_at_the_caps(self):
        # Every cell is at a bound: rows 1-2 cut to 1.2, row 3 raised to its
        # floor of 0.3. The missing 0.3 has nowhere to go and the values stay
        # where the bounds put them.
        out = _impute(
            _frame(
                [100.0, pd.NA, pd.NA, 103.0],
                [pd.NA, 1.0, 1.0, 0.0],
                k=[pd.NA, 1.2, 1.2, 0.3],
            ),
            constraints=_constraints(*FOLLOWS_K),
        )
        assert out.loc[1:3, DIFF].astype(float).tolist() == pytest.approx(
            [1.2, 1.2, 0.3]
        )
        for row in (1, 2, 3):
            assert _flags(out, row) & ImputeType.CLIPPED.value


class TestUnanchoredFillsGiveNothingBack:
    def test_the_clip_is_local_to_the_cell(self):
        cumulative = [10.0, 10.5, 11.0, 11.5, 12.0, 12.5] + [pd.NA] * 4
        diffs = [0.5] * 6 + [pd.NA] * 4
        k = [pd.NA] * 7 + [0.3] + [pd.NA] * 2
        out = _impute(
            _frame(cumulative, [1.0] * 10, diffs=diffs, k=k),
            constraints=_constraints(*FOLLOWS_K),
        )
        assert float(out.loc[7, DIFF]) == pytest.approx(0.3)
        assert _flags(out, 7) & ImputeType.CLIPPED.value
        # The neighbours keep the household-scaled 0.5: no give-back without
        # a measured total.
        for row in (6, 8, 9):
            assert float(out.loc[row, DIFF]) == pytest.approx(0.5)
            assert not _flags(out, row) & ImputeType.CLIPPED_REDISTRIBUTION.value
