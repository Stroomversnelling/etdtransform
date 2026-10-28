"""Which fill a gap gets is decided by whether the average is there.

Three cases, decided on presence:

  average present for every cell   -> proportional
  present for some cells           -> proportional, the rest taking the gap mean
  absent for every cell            -> uniform

Two constraints on the fixtures. The gap group runs from the first missing cell
up to and including the row carrying the closing cumulative reading -- that
row's interval value is consumption over the step into it, so it belongs to the
gap. And filled values must stay under the column's declared maximum (1.5 for
``ElektriciteitNetgebruikDiff``), because the threshold pass runs after
imputation and nulls anything above it.
"""

import pandas as pd
import pytest
from conftest import cadence_table
from etdmap.testing import datasets_frame, household_batch_of

from etdtransform.vectorized_impute import (
    ImputeType,
    process_gap_and_cumulative_groups,
    process_imputation_vectorized,
)

pytestmark = pytest.mark.unit

DIFF = "ElektriciteitNetgebruikDiff"
CUM = "ElektriciteitNetgebruik"
AVG = f"{DIFF}_avg"
IMPUTED = f"{DIFF}_is_imputed"
ITYPE = f"{DIFF}_impute_type"

#: The one household every frame below describes.
HOUSEHOLD = 1

# Readings at rows 0 and 4; rows 1-4 are the gap. A jump of 4.0 over four cells
# averages 1.0, comfortably under the 1.5 maximum.
CUMULATIVE = [100.0, pd.NA, pd.NA, pd.NA, 104.0]
GAP = slice(1, 4)
JUMP = 4.0


def _frame(averages, end=CUMULATIVE[-1]):
    cumulative = CUMULATIVE[:-1] + [end]
    n = len(cumulative)
    return datasets_frame(
        {household_batch_of(HOUSEHOLD): {
            CUM: cumulative,
            DIFF: [pd.NA] * n,
            AVG: list(averages),
        }},
        start="2026-01-01",
        freq="5min",
        tz=None,
        # Both halves of the pair are stated here: the anchored cumulative
        # column, and the empty interval column the fill writes into.
        pairs=False,
    )


def _impute(averages, end=CUMULATIVE[-1]):
    frame = _frame(averages, end)
    prepared = process_gap_and_cumulative_groups(frame, diff_col=DIFF, cum_col=CUM)
    return process_imputation_vectorized(
        prepared, diff_col=DIFF, cum_col=CUM, avg_col=AVG,
        impute_type_col=ITYPE, is_imputed_col=IMPUTED,
        cadence=cadence_table(frame, [CUM]),
    )


def _filled(out):
    return out.loc[GAP, DIFF]


def _types(out):
    """Impute types in the gap, with the threshold bit masked off.

    The threshold pass ORs its own bit onto whatever the fill recorded, so
    asserting on the raw value would test two things at once.
    """
    return {
        int(v) & ~ImputeType.THRESHOLD_ADJUSTED.value
        for v in out.loc[GAP, ITYPE].dropna()
    }


class TestTheThreeRoutes:
    def test_a_complete_average_gives_a_proportional_fill(self):
        out = _impute([pd.NA, 0.5, 1.0, 0.5, 1.0])
        filled = _filled(out).astype(float).tolist()
        assert float(sum(filled)) == pytest.approx(JUMP)
        # Shape 0.5 : 1.0 : 0.5 : 1.0 scaled to a jump of 4.0.
        assert filled == pytest.approx([0.667, 1.333, 0.667, 1.333], abs=1e-3)
        assert _types(out) == {ImputeType.SCALED_FILL.value}

    def test_an_absent_average_gives_a_uniform_fill(self):
        out = _impute([pd.NA] * 5)
        filled = _filled(out).astype(float).tolist()
        assert filled == pytest.approx([1.0, 1.0, 1.0, 1.0])
        assert float(sum(filled)) == pytest.approx(JUMP)
        assert _types(out) == {ImputeType.UNIFORM_FILL.value}

    def test_a_partly_present_average_fills_every_cell(self):
        """The cells with no average are filled too, not left empty."""
        out = _impute([pd.NA, 1.0, pd.NA, 1.0, pd.NA])
        filled = _filled(out)
        assert filled.notna().all(), "a cell of the gap was left unfilled"
        assert float(filled.sum()) == pytest.approx(JUMP)
        assert filled.astype(float).tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])

    def test_a_partly_present_average_keeps_the_shape_it_has(self):
        out = _impute([pd.NA, 1.5, pd.NA, 0.5, pd.NA])
        filled = _filled(out)
        assert filled.notna().all()
        assert float(filled.sum()) == pytest.approx(JUMP)
        # Known cells stay in 3:1; the unknown ones sit at their mean of 1.0.
        assert filled.astype(float).tolist() == pytest.approx([1.5, 1.0, 0.5, 1.0])


class TestZeroIsNotAbsent:
    """A shape that is genuinely zero and one that is missing are different
    facts. A sum over the gap cannot tell them apart."""

    def test_an_all_zero_average_gives_a_uniform_fill(self):
        out = _impute([pd.NA, 0.0, 0.0, 0.0, 0.0])
        filled = _filled(out).astype(float).tolist()
        assert filled == pytest.approx([1.0, 1.0, 1.0, 1.0])
        assert _types(out) == {ImputeType.UNIFORM_FILL.value}

    def test_a_partly_zero_average_keeps_its_zeros_and_conserves_volume(self):
        """A smaller jump here on purpose.

        With a jump of 4.0 the two non-zero cells take 2.0 each, which is over
        the 1.5 maximum -- and a flat run followed by an over-bound value is
        exactly the frozen-span-and-catch-up shape, which `_catchup_positions`
        deliberately spares. The test would then pass because of catch-up
        handling rather than because of the routing it claims to test.
        """
        out = _impute([pd.NA, 0.0, 1.0, 0.0, 1.0], end=102.0)
        filled = _filled(out)
        assert float(filled.sum()) == pytest.approx(2.0)
        assert filled.notna().all()
        # A zero here means no consumption, so those cells stay empty.
        assert filled.astype(float).tolist() == pytest.approx([0.0, 1.0, 0.0, 1.0])


class TestVolumeIsAlwaysConserved:
    @pytest.mark.parametrize("averages", [
        [pd.NA, 0.5, 1.0, 0.5, 1.0],
        [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
        [pd.NA, 1.0, pd.NA, 1.0, pd.NA],
        [pd.NA, 0.0, 0.0, 0.0, 0.0],
        [pd.NA, 1.0, pd.NA, pd.NA, pd.NA],
    ])
    def test_the_gap_jump_is_conserved_whatever_the_average(self, averages):
        out = _impute(averages)
        filled = _filled(out)
        assert filled.notna().all(), "a cell of the gap was left unfilled"
        assert float(filled.sum()) == pytest.approx(JUMP)
