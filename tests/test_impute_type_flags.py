"""Every ImputeType member, one targeted case each.

These are unit tests on the fill branches, not fixture runs. Each builds the
smallest frame that forces one branch and asserts on the impute type it writes.

Two flags per filled cell. The members up to THRESHOLD_ADJUSTED name the
condition that led to the fill; the members after it name the action taken.
`_types` asserts on the whole value, so a condition paired with the wrong
action fails.

The threshold pass runs after imputation and ORs THRESHOLD_ADJUSTED onto
whatever the fill recorded, so every fixture keeps its filled values under the
column's declared maximum (1.5 for ElektriciteitNetgebruikDiff) except the one
test that is about the threshold pass itself.
"""

import pandas as pd
import pytest

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

# The signal / action / modifier split is defined once, in
# test_impute_flag_taxonomy.py. This file asserts that each member is actually
# produced by the code, so it imports the classification rather than repeating
# it -- two lists would drift.
from test_impute_flag_taxonomy import ACTIONS, SIGNALS  # noqa: E402
from conftest import cadence_table

CONDITIONS = SIGNALS

# Recorded by _expect as each test asserts; read by the contract test below.
SEEN: set[ImputeType] = set()


def _frame(cumulative, averages, diffs=None, household=1):
    n = len(cumulative)
    return pd.DataFrame({
        "HuisIdBSV": pd.array([household] * n, dtype="Int64"),
        "ProjectIdBSV": pd.array([1] * n, dtype="Int64"),
        "ReadingDate": pd.date_range("2026-01-01", periods=n, freq="5min"),
        CUM: pd.array(cumulative, dtype="Float64"),
        DIFF: pd.array([pd.NA] * n if diffs is None else diffs, dtype="Float64"),
        AVG: pd.array(averages, dtype="Float64"),
    })


def _impute(frame):
    prepared = process_gap_and_cumulative_groups(frame, diff_col=DIFF, cum_col=CUM)
    return process_imputation_vectorized(
        prepared, diff_col=DIFF, cum_col=CUM, avg_col=AVG,
        impute_type_col=ITYPE, is_imputed_col=IMPUTED,
        cadence=cadence_table(frame, [CUM]),
    )


def _types(out, rows):
    return {int(v) for v in out.loc[rows, ITYPE].dropna()}


def _expect(out, rows, *members):
    """Assert the exact impute type on `rows` and record the members covered."""
    value = 0
    for member in members:
        value |= member.value
        SEEN.add(member)
    assert _types(out, rows) == {value}, (
        f"expected {' | '.join(m.name for m in members)} ({value}), "
        f"got {_types(out, rows)}"
    )


class TestGapJumpConditions:
    """A gap bounded by two cumulative readings. The jump between them is the
    measured quantity the fill has to distribute."""

    def test_a_negative_jump_fills_zeros(self):
        # The meter reads lower than it did: a reset, not consumption.
        out = _impute(_frame([100.0, pd.NA, pd.NA, 90.0], [1.0] * 4))
        _expect(out, slice(1, 3), ImputeType.NEGATIVE_GAP_JUMP, ImputeType.ZERO_FILL)

    def test_a_jump_of_zero_fills_zeros(self):
        out = _impute(_frame([100.0, pd.NA, pd.NA, 100.0], [1.0] * 4))
        _expect(out, slice(1, 3), ImputeType.NEAR_ZERO_GAP_JUMP, ImputeType.ZERO_FILL)

    def test_no_shape_spreads_the_jump_evenly(self):
        out = _impute(_frame([100.0, pd.NA, pd.NA, 102.0], [pd.NA] * 4))
        _expect(out, slice(1, 3), ImputeType.UNIFORM_FILL)
        # Three cells carry the gap: the two missing ones and the row holding
        # the closing reading, whose interval covers the step into it.
        assert out.loc[1:3, DIFF].astype(float).tolist() == pytest.approx([2 / 3] * 3)

    def test_a_shape_distributes_the_jump_by_it(self):
        out = _impute(_frame([100.0, pd.NA, pd.NA, 103.0], [pd.NA, 0.5, 1.0, 1.5]))
        _expect(out, slice(1, 3), ImputeType.SCALED_FILL)
        # Shape 0.5 : 1.0 : 1.5 carrying a jump of 3.0.
        assert out.loc[1:3, DIFF].astype(float).tolist() == pytest.approx([0.5, 1.0, 1.5])


class TestOnlyOneBoundReading:
    """One end of the gap has a cumulative reading and the other does not, so
    there is no jump to distribute."""

    def test_a_zero_closing_reading_no_longer_fills_zeros(self):
        """Filling zeros here would say the meter stood at zero for the whole
        gap. With no opening reading the cumulative is pd.NA, not zero, so
        nothing supports that and nothing is written."""
        out = _impute(_frame([pd.NA, pd.NA, 0.0, 1.0], [1.0] * 4))
        assert out.loc[0:2, DIFF].isna().all()
        _expect(out, slice(0, 2), ImputeType.ZERO_END_VALUE,
                ImputeType.ONLY_CLOSING_READING, ImputeType.NOT_FILLED)

    def test_a_positive_closing_reading_with_no_evidence_fills_nothing(self):
        """Too few comparable rows for a household factor, so the same rule as
        the opening-reading case applies: nothing is written."""
        out = _impute(_frame([pd.NA, pd.NA, 50.0, 51.0], [1.0, 1.0, 1.0, 1.0]))
        assert out.loc[0:2, DIFF].isna().all()
        _expect(
            out, slice(0, 2),
            ImputeType.POSITIVE_END_VALUE, ImputeType.ONLY_CLOSING_READING,
            ImputeType.NOT_FILLED,
        )

    def test_a_closing_reading_scales_by_the_household(self):
        """With the factor resting on enough of the series, the gap is filled
        at the household's own level -- the mirror of the opening-reading
        case."""
        # The closing reading must land on a row whose interval is still
        # missing, or it does not close the gap: the group ends where a
        # cumulative value meets an absent diff.
        cumulative = [pd.NA, pd.NA, pd.NA, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0]
        diffs = [pd.NA] * 4 + [0.5] * 6
        out = _impute(_frame(cumulative, [1.0] * 10, diffs=diffs))
        filled = out.loc[0:3, DIFF]
        assert filled.notna().all()
        assert filled.astype(float).tolist() == pytest.approx([0.5] * 4)
        _expect(
            out, slice(0, 3),
            ImputeType.POSITIVE_END_VALUE, ImputeType.ONLY_CLOSING_READING,
            ImputeType.HOUSEHOLD_SCALED_AVERAGE,
        )

    def test_an_opening_reading_and_no_close_scales_by_the_household(self):
        """Enough comparable rows for a real factor, so the scaled flag holds.

        The household's own values run at half the average over the comparable
        rows, so the factor is 0.5 and the fill must be half the average.
        """
        cumulative = [10.0, 10.5, 11.0, 11.5, 12.0, 12.5] + [pd.NA] * 4
        diffs = [0.5] * 6 + [pd.NA] * 4
        out = _impute(_frame(cumulative, [1.0] * 10, diffs=diffs))
        _expect(
            out, slice(6, 9),
            ImputeType.NO_END_VALUE, ImputeType.ONLY_OPENING_READING,
            ImputeType.HOUSEHOLD_SCALED_AVERAGE,
        )
        # Six comparable rows against an average of 1.0 give a factor of 0.5,
        # and six of ten clears the more-than-half bar the factor requires.
        assert out.loc[6:9, DIFF].astype(float).tolist() == pytest.approx([0.5] * 4)

    def test_an_opening_reading_with_no_evidence_fills_nothing(self):
        """Too few comparable rows, so the factor would fall back to 1.0.

        Nothing measured bounds this gap, so no value is written and the
        condition is recorded alongside NOT_FILLED.
        """
        cumulative = [10.0] + [pd.NA] * 5
        out = _impute(_frame(cumulative, [1.0] * 6))
        _expect(
            out, slice(1, 5),
            ImputeType.NO_END_VALUE, ImputeType.ONLY_OPENING_READING,
            ImputeType.NOT_FILLED,
        )
        assert out.loc[1:5, DIFF].isna().all(), (
            "a gap with no closing reading and no household evidence was filled"
        )


class TestNoCumulativeAtAll:
    """Neither end of the gap has a cumulative reading."""

    def test_a_household_delivering_intervals_is_scaled_to_its_own_level(self):
        diffs = [pd.NA, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, pd.NA]
        out = _impute(_frame([pd.NA] * 8, [1.0] * 8, diffs=diffs))
        _expect(
            out, slice(7, 7),
            ImputeType.NO_CUMULATIVE_VALUE, ImputeType.HOUSEHOLD_SCALED_AVERAGE,
        )
        assert float(out.loc[7, DIFF]) == pytest.approx(0.5)

    def test_intervals_with_too_little_evidence_fill_nothing(self):
        """Interval delivery, but not enough of the series to scale with.

        Three comparable rows in eight is below the half the factor needs, so
        it falls back to 1.0 and nothing is written.
        """
        diffs = [pd.NA, 0.5, 0.5, 0.5, pd.NA, pd.NA, pd.NA, pd.NA]
        out = _impute(_frame([pd.NA] * 8, [1.0] * 8, diffs=diffs))
        gap = out.loc[4:7, DIFF]
        assert gap.isna().all(), "a gap with no cumulative bound was filled"
        _expect(
            out, slice(4, 7),
            ImputeType.NO_CUMULATIVE_VALUE, ImputeType.NOT_FILLED,
        )

    def test_a_household_with_only_a_seed_zero_is_not_interval_delivery(self):
        """add_diff_columns seeds row 0 of every household with 0. That
        structural zero must not count as delivered interval data.
        """
        out = _impute(_frame([pd.NA] * 6, [1.0] * 6, diffs=[0.0] + [pd.NA] * 5))
        filled = out.loc[1:5, DIFF]
        assert filled.isna().all(), (
            "a household whose only interval value is the row-0 seed zero was "
            "treated as delivering interval data and got filled"
        )

    def test_nothing_at_all_fills_nothing(self):
        out = _impute(_frame([pd.NA] * 6, [1.0] * 6))
        assert out.loc[1:5, DIFF].isna().all()


class TestActionsWithoutAConditionOfTheirOwn:
    def test_a_cell_no_branch_wrote_to_carries_not_filled(self):
        """No average anywhere and no closing reading: nothing can be computed.

        Row 0 is excluded from NOT_FILLED by design -- it bounds no gap -- so
        the flag is asserted from row 1 on.
        """
        out = _impute(_frame([pd.NA, pd.NA, 5.0, pd.NA, pd.NA, pd.NA], [pd.NA] * 6))
        unfilled = out.index[out[DIFF].isna() & out[ITYPE].notna()]
        assert len(unfilled) > 0, "expected at least one unfilled gap cell"
        for value in out.loc[unfilled, ITYPE]:
            assert int(value) & ImputeType.NOT_FILLED.value
        SEEN.add(ImputeType.NOT_FILLED)

    def test_the_threshold_pass_records_the_condition_and_its_action(self):
        """Out of bounds is the condition; substituting the average is the
        action. A cell keeps the fill it was given -- the history is both."""
        out = _impute(_frame([100.0, pd.NA, pd.NA, 112.0], [1.0] * 4))
        types = [int(v) for v in out.loc[1:3, ITYPE].dropna()]
        violated = [t for t in types if t & ImputeType.THRESHOLD_VIOLATED.value]
        assert violated, "an over-bound fill was not marked THRESHOLD_VIOLATED"
        for value in violated:
            assert value & ImputeType.FILLED.value, (
                "the fill that produced the value is no longer in the history"
            )
            assert not (value & ImputeType.NOT_FILLED.value)
            # The substitute is the project average as it stands: the threshold
            # pass has no household factor to scale it by. It is now the only
            # producer of this flag.
            assert value & ImputeType.UNSCALED_AVERAGE.value
        SEEN.add(ImputeType.THRESHOLD_VIOLATED)
        SEEN.add(ImputeType.UNSCALED_AVERAGE)

    def test_the_threshold_pass_records_a_blanking_separately(self):
        """With no average to substitute, the pass blanks the value. That is a
        different action from substituting one, and from imputation's own
        NULLED."""
        out = _impute(_frame([100.0, pd.NA, pd.NA, 112.0], [pd.NA] * 4))
        blanked = out.loc[1:3, DIFF].isna()
        types = [int(v) for v in out.loc[1:3, ITYPE].dropna()]
        if blanked.any():
            assert any(
                t & ImputeType.NULLED_POST_IMPUTATION.value for t in types
            ), "a blanked cell was not marked NULLED_POST_IMPUTATION"
        SEEN.add(ImputeType.NULLED_POST_IMPUTATION)
        SEEN.add(ImputeType.NULLED)

    def test_a_fill_over_the_maximum_is_marked_adjusted(self):
        """1.5 is the declared maximum for this column; a jump of 12.0 over
        three cells lands at 4.0 each and has to be cut."""
        out = _impute(_frame([100.0, pd.NA, pd.NA, 112.0], [pd.NA] * 4))
        adjusted = [
            int(v) for v in out.loc[1:3, ITYPE].dropna()
            if int(v) & ImputeType.THRESHOLD_ADJUSTED.value
        ]
        assert adjusted, "values over the maximum were not marked THRESHOLD_ADJUSTED"
        SEEN.add(ImputeType.THRESHOLD_ADJUSTED)


class TestEveryMemberIsCovered:
    """Contract guard: a flag added to ImputeType without a test here fails."""

    def test_the_two_groups_account_for_every_member(self):
        members = {m for m in ImputeType if m.name and bin(m.value).count("1") == 1}
        assert members == SIGNALS | ACTIONS, (
            "ImputeType gained or lost a member; classify it in "
            "test_impute_flag_taxonomy.py and give it a test here"
        )

    def test_every_member_was_asserted_on(self):
        missing = sorted(m.name for m in (SIGNALS | ACTIONS) - SEEN)
        assert not missing, f"ImputeType members with no test: {missing}"
