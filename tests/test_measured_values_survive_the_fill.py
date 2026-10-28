"""The fill writes into gaps and nowhere else.

Imputation's core safety property: a cell that already holds a reading comes
out of the fill holding the same reading. If it ever did not, every downstream
number would move -- the cumulative rebuild, the resampled intervals, the
project averages -- and the output would still look well formed.

Two existing tests sit either side of this one without covering it.
`test_flags_are_only_added.py::test_an_untouched_row_keeps_only_what_it_had`
asserts on the flag bits, not the value. `test_group5_fill_rules.py::
TestNoDeliveredNoFill::test_an_all_na_household_is_untouched_with_no_flags`
covers a household with nothing to preserve. The uncovered case is the normal
one: a household holding both readings and gaps.

Scoped to the INTERVAL column, which is what the fill writes into. The
cumulative column is rebuilt from it afterwards by
`reconstruct_cumulative_columns`, so its values are expected to move.

In-bounds values only. A reading that violates its threshold IS rewritten, on
purpose, and marked THRESHOLD_ADJUSTED -- that behaviour is asserted in
test_impute_type_flags.py and is not what this file is about.
"""

import pandas as pd
import pytest
from conftest import cadence_table
from etdmap import ids
from etdmap.testing import household_batch_of, household_batches_frame

from etdtransform.vectorized_impute import impute_and_normalize

pytestmark = pytest.mark.unit

CUM = "ElektriciteitNetgebruik"
DIFF = "ElektriciteitNetgebruikDiff"

#: The one household this frame describes.
HOUSEHOLD = 1

#: Ten five-minute slots for one household. An interval is measured where the
#: register was read on consecutive slots, so the two run together:
#:
#:   slot    0      1      2      3    4    5      6      7    8    9
#:   CUM   100.0  100.4  101.1   NA   NA  103.0  103.9   NA   NA  106.0
#:   DIFF    0.0    0.4    0.7    NA   NA    NA     0.9   NA   NA    NA
#:
#: That leaves two gaps, each bounded by a read register so the fill has a jump
#: to distribute, and three measured intervals sitting between them. The
#: measured values are distinct, so a preserved cell is identified by its own
#: value rather than by its position. Slot 0 is excluded: its 0.0 is the series
#: seed, not a reading.
MEASURED = {1: 0.4, 2: 0.7, 6: 0.9}
_CUM = [100.0, 100.4, 101.1, pd.NA, pd.NA, 103.0, 103.9, pd.NA, pd.NA, 106.0]
_DIFF = [0.0, 0.4, 0.7, pd.NA, pd.NA, pd.NA, 0.9, pd.NA, pd.NA, pd.NA]
_N = len(_CUM)


def _frame():
    return household_batches_frame(
        {household_batch_of(HOUSEHOLD): {
            CUM: _CUM,
            DIFF: _DIFF,
            # The project average the scaled fill reads. Well inside the
            # column's declared bound, so nothing here is a threshold case.
            f"{DIFF}_avg": [0.6] * _N,
        }},
        start="2026-01-01",
        freq="5min",
        tz=None,
        # Both halves of the pair are stated here: the read cumulative column,
        # and the interval column the fill writes into.
        pairs=False,
    )


@pytest.fixture
def filled():
    frame = _frame()
    out, _, _ = impute_and_normalize(
        frame, [CUM], ids.PROJECT,
        cadence=cadence_table(frame, [CUM]),
    )
    return out


def test_every_measured_cell_keeps_its_own_value(filled):
    for position, value in MEASURED.items():
        got = filled.loc[position, DIFF]
        assert not pd.isna(got), f"slot {position} held {value} and came back empty"
        assert float(got) == pytest.approx(value), (
            f"slot {position}: {value} became {float(got)}"
        )


def test_no_measured_cell_is_marked_imputed(filled):
    """The companion half. A preserved value that the run also claims to have
    imputed would make the provenance wrong even though the number is right,
    and a reader uses that mark to decide what the cell is worth."""
    marks = filled[f"{DIFF}_is_imputed"]
    for position in MEASURED:
        assert not bool(marks.loc[position]), f"slot {position} marked imputed"


def test_the_gaps_really_were_filled(filled):
    """Without this the file could pass on a run that did nothing at all.

    Four slots were empty and bounded by a read register at both ends, so the
    fill has work to do; if it did none, preserving the other four is no
    evidence.
    """
    gaps = [i for i in range(_N) if i not in MEASURED]
    assert filled.loc[gaps, DIFF].notna().any(), "the fill wrote nothing"
    assert filled.loc[gaps, f"{DIFF}_is_imputed"].any()
