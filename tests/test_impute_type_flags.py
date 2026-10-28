"""Every ImputeType member, one targeted case each.

These are unit tests on the fill branches, not fixture runs. Each builds the
smallest frame that forces one branch and asserts on the impute type it writes.

Two flags per filled cell: a signal naming the condition that led to the fill
and an action naming what was done -- the split is the SIGNALS and ACTIONS
composites on the class. `_types` asserts on the whole value, so a condition
paired with the wrong action fails.

The threshold pass runs after imputation and ORs THRESHOLD_ADJUSTED onto
whatever the fill recorded, so every fixture keeps its filled values under the
column's declared maximum (1.5 for ElektriciteitNetgebruikDiff) except the one
test that is about the threshold pass itself.
"""

import ast
from pathlib import Path

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

# The signal / action split lives on the class itself, as the SIGNALS and
# ACTIONS composites. This file asserts that each member is actually produced
# by the code, so it derives the member set rather than keeping a copy -- a
# second list would drift.
from conftest import cadence_table
from etdmap.testing import household_batch_of, household_batches_frame

ALL_FLAGS = {m for m in ImputeType if m.name and bin(m.value).count("1") == 1}
CONDITIONS = {m for m in ALL_FLAGS if m & ImputeType.SIGNALS}


def _frame(cumulative, averages, diffs=None, household=1):
    """One household batch of *household*, tz-naive, at the 5-minute grid.

    Both halves of the cumulative/interval pair are stated, so the builder
    completes neither: the cumulative readings bound the gaps and the interval
    values are the shape the fill distributes by.
    """
    return household_batches_frame(
        {household_batch_of(household): {
            CUM: list(cumulative),
            DIFF: pd.NA if diffs is None else list(diffs),
            AVG: list(averages),
        }},
        start="2026-01-01", freq="5min", tz=None,
    )


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
    """Assert the exact impute type on `rows`."""
    value = 0
    for member in members:
        value |= member.value
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
                ImputeType.ONLY_CLOSING_READING, ImputeType.GAP_NOT_FILLED)

    def test_a_positive_closing_reading_with_no_evidence_fills_nothing(self):
        """Too few comparable rows for a household factor, so the same rule as
        the opening-reading case applies: nothing is written."""
        out = _impute(_frame([pd.NA, pd.NA, 50.0, 51.0], [1.0, 1.0, 1.0, 1.0]))
        assert out.loc[0:2, DIFF].isna().all()
        _expect(
            out, slice(0, 2),
            ImputeType.POSITIVE_END_VALUE, ImputeType.ONLY_CLOSING_READING,
            ImputeType.GAP_NOT_FILLED,
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
        condition is recorded alongside GAP_NOT_FILLED.
        """
        cumulative = [10.0] + [pd.NA] * 5
        out = _impute(_frame(cumulative, [1.0] * 6))
        _expect(
            out, slice(1, 5),
            ImputeType.NO_END_VALUE, ImputeType.ONLY_OPENING_READING,
            ImputeType.GAP_NOT_FILLED,
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
            ImputeType.NO_CUMULATIVE_VALUE, ImputeType.GAP_NOT_FILLED,
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

        Row 0 is excluded from GAP_NOT_FILLED by design -- it bounds no gap -- so
        the flag is asserted from row 1 on.
        """
        out = _impute(_frame([pd.NA, pd.NA, 5.0, pd.NA, pd.NA, pd.NA], [pd.NA] * 6))
        unfilled = out.index[out[DIFF].isna() & out[ITYPE].notna()]
        assert len(unfilled) > 0, "expected at least one unfilled gap cell"
        for value in out.loc[unfilled, ITYPE]:
            assert int(value) & ImputeType.GAP_NOT_FILLED.value

    def test_the_threshold_pass_records_the_condition_and_its_action(self):
        """Out of bounds is the condition; substituting the average is the
        action. A cell keeps the fill it was given -- the history is both."""
        out = _impute(_frame([100.0, pd.NA, pd.NA, 112.0], [1.0] * 4))
        types = [int(v) for v in out.loc[1:3, ITYPE].dropna()]
        violated = [t for t in types if t & ImputeType.THRESHOLD_VIOLATED.value]
        assert violated, "an over-bound fill was not marked THRESHOLD_VIOLATED"
        for value in violated:
            assert value & ImputeType.GAP_FILLED.value, (
                "the fill that produced the value is no longer in the history"
            )
            assert not (value & ImputeType.GAP_NOT_FILLED.value)
            # The substitute is the project average as it stands: the threshold
            # pass has no household factor to scale it by. It is now the only
            # producer of this flag.
            assert value & ImputeType.UNSCALED_AVERAGE.value

    def test_the_threshold_pass_records_a_blanking_separately(self):
        """With no average to substitute, the pass blanks the value. That is a
        different action from substituting one, and from imputation's own
        GAP_NULLED."""
        out = _impute(_frame([100.0, pd.NA, pd.NA, 112.0], [pd.NA] * 4))
        blanked = out.loc[1:3, DIFF].isna()
        types = [int(v) for v in out.loc[1:3, ITYPE].dropna()]
        if blanked.any():
            assert any(
                t & ImputeType.NULLED_POST_IMPUTATION.value for t in types
            ), "a blanked cell was not marked NULLED_POST_IMPUTATION"

    def test_a_fill_over_the_maximum_is_marked_adjusted(self):
        """1.5 is the declared maximum for this column; a jump of 12.0 over
        three cells lands at 4.0 each and has to be cut."""
        out = _impute(_frame([100.0, pd.NA, pd.NA, 112.0], [pd.NA] * 4))
        adjusted = [
            int(v) for v in out.loc[1:3, ITYPE].dropna()
            if int(v) & ImputeType.THRESHOLD_ADJUSTED.value
        ]
        assert adjusted, "values over the maximum were not marked THRESHOLD_ADJUSTED"


# Files whose subject is the vocabulary itself: they name members to check
# the class, not the behaviour, so a mention there is not a test of the flag.
STRUCTURAL_TEST_FILES = {"test_impute_flag_taxonomy.py"}


def _members_mentioned_in_tests(class_name):
    """Member names referenced as `<class_name>.<MEMBER>` in any test file.

    A flag a producer test asserts on is necessarily named there, so a member
    no test file even mentions has no test. This is an inventory check only --
    that the assertion behind a mention is a real one is each producer test's
    own job.
    """
    mentioned = set()
    for path in sorted(Path(__file__).parent.glob("test_*.py")):
        if path.name in STRUCTURAL_TEST_FILES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == class_name
            ):
                mentioned.add(node.attr)
    return mentioned


# Vocabulary declared ahead of its producer: the meaning is fixed in
# DESCRIPTIONS, but nothing writes the flag yet. A member leaves this set the
# moment a producer test names it -- the honesty test below fails and says so.
# GAP_NULLED: the value-blanking action of the imputation's exclusive triple
# (GAP_FILLED, GAP_NOT_FILLED, GAP_NULLED); every live blanking today happens
# in mapping or in the threshold pass, which record themselves differently.
# The rest have no writer anywhere in the pipeline: each one's DESCRIPTIONS
# entry opens with RESERVED and says what sets it, so the reason for the
# exemption is recorded once, beside the meaning.
UNPRODUCED = {
    "GAP_NULLED",
    "IN_GAP",
    "CALCULATED_FROM_HARD_KNOWNS",
    "CALCULATED_FROM_SOFT_KNOWNS",
    "CALCULATED_RULE_BOUND_FROM_HARD_KNOWNS",
    "NO_AVERAGE_SHAPE",
    "SPARSE_AVERAGE",
}


class TestEveryMemberIsCovered:
    """Contract guard: a flag added to ImputeType with no test anywhere fails.

    The producer tests live in more than one file -- fills here, threshold and
    bound behaviour in their own files -- so coverage is collected by scanning
    every test file, not by recording within this one. Declared-ahead
    vocabulary sits in UNPRODUCED instead of staying red: both tests here are
    green in the steady state, and each direction of drift fails exactly one
    of them.
    """

    def test_every_member_with_a_producer_is_tested(self):
        mentioned = _members_mentioned_in_tests("ImputeType")
        missing = sorted(
            m.name for m in ALL_FLAGS
            if m.name not in mentioned and m.name not in UNPRODUCED
        )
        assert not missing, f"ImputeType members with no test: {missing}"

    def test_the_unproduced_list_is_honest(self):
        """A member both declared unproduced and tested is one whose producer
        has arrived: remove it from UNPRODUCED rather than let the exception
        outlive its reason."""
        mentioned = _members_mentioned_in_tests("ImputeType")
        arrived = sorted(UNPRODUCED & mentioned)
        assert not arrived, (
            f"now produced and tested -- remove from UNPRODUCED: {arrived}"
        )

    def test_the_unproduced_list_names_real_members(self):
        unknown = sorted(UNPRODUCED - {m.name for m in ALL_FLAGS})
        assert not unknown, (
            f"UNPRODUCED names members that do not exist: {unknown}"
        )

    def test_declared_but_unused_members_are_reported(self):
        """Always passes; reminds the runner, in the run's warnings summary,
        which members are declared but expected not to be in use. A quiet
        exception is one nobody re-examines."""
        if UNPRODUCED:
            import warnings

            warnings.warn(
                "ImputeType members declared but expected not to be in use "
                f"(no producer, so no tests): {sorted(UNPRODUCED)}",
                UserWarning,
                stacklevel=1,
            )
