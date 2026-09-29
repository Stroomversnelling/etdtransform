"""ImputeType members divide into signals and actions.

A signal names the condition a cell was in. An action names what imputation did
about it.

Imputation does exactly one thing to any cell: it writes a value, writes
nothing, or blanks it. So exactly one of FILLED, NOT_FILLED, NULLED applies.
The threshold pass runs afterwards and records its own action separately, which
is why a cell can read FILLED and NULLED_POST_IMPUTATION at once -- that is the
history, not a contradiction.
"""

import pytest

from etdtransform.vectorized_impute import ImputeType

SIGNALS = {
    ImputeType.NEGATIVE_GAP_JUMP,
    ImputeType.NEAR_ZERO_GAP_JUMP,
    ImputeType.UNIFORM_FILL,
    ImputeType.SCALED_FILL,
    ImputeType.ZERO_END_VALUE,
    ImputeType.POSITIVE_END_VALUE,
    ImputeType.NO_END_VALUE,
    ImputeType.NO_CUMULATIVE_VALUE,
    # An alias of THRESHOLD_VIOLATED in practice, and deprecated for that
    # reason; still a condition rather than something imputation did.
    ImputeType.THRESHOLD_ADJUSTED,
    ImputeType.ONLY_CLOSING_READING,
    ImputeType.ONLY_OPENING_READING,
    ImputeType.THRESHOLD_VIOLATED,
}
ACTIONS = {
    ImputeType.ZERO_FILL,
    ImputeType.UNSCALED_AVERAGE,
    ImputeType.HOUSEHOLD_SCALED_AVERAGE,
    ImputeType.NOT_FILLED,
    ImputeType.NULLED,
    ImputeType.NULLED_POST_IMPUTATION,
}


def _single_bits():
    return {m for m in ImputeType if m.name and bin(m.value).count("1") == 1}


class TestTheTaxonomyIsComplete:
    def test_every_member_is_classified(self):
        missing = sorted(m.name for m in _single_bits() - SIGNALS - ACTIONS)
        assert not missing, (
            f"ImputeType members classified as neither signal nor action: "
            f"{missing}. Add each to a group in this file."
        )

    def test_no_member_is_in_two_groups(self):
        both = sorted(m.name for m in SIGNALS & ACTIONS)
        assert not both, f"members in both signal and action: {both}"


class TestFilledIsTheComplementOfNotFilled:
    """FILLED collects the actions that wrote a value, so a reader can select
    cells carrying an imputed number without naming every fill method."""

    def test_filled_is_the_value_writing_actions(self):
        assert ImputeType.FILLED == (
            ImputeType.ZERO_FILL
            | ImputeType.UNSCALED_AVERAGE
            | ImputeType.HOUSEHOLD_SCALED_AVERAGE
        )

    def test_filled_excludes_not_filled(self):
        assert not (ImputeType.FILLED & ImputeType.NOT_FILLED)

    def test_filled_excludes_the_blanking_actions(self):
        assert not (ImputeType.FILLED & ImputeType.NULLED)
        assert not (ImputeType.FILLED & ImputeType.NULLED_POST_IMPUTATION)

    @pytest.mark.parametrize("method", [
        ImputeType.ZERO_FILL,
        ImputeType.UNSCALED_AVERAGE,
        ImputeType.HOUSEHOLD_SCALED_AVERAGE,
    ])
    def test_each_fill_method_reaches_filled(self, method):
        assert method & ImputeType.FILLED


class TestImputedValue:
    """Every cell imputation changed, whichever way it changed it."""

    def test_it_excludes_not_filled(self):
        assert not (ImputeType.IMPUTED_VALUE & ImputeType.NOT_FILLED)

    def test_it_covers_the_fill_methods_and_the_blanking(self):
        assert ImputeType.FILLED & ImputeType.IMPUTED_VALUE
        assert ImputeType.NULLED & ImputeType.IMPUTED_VALUE
        assert ImputeType.NULLED_POST_IMPUTATION & ImputeType.IMPUTED_VALUE


class TestTheThresholdPassIsSeparate:
    """It runs after imputation, so its action does not belong to the
    exclusive triple and does not clear what imputation recorded."""

    def test_threshold_violated_is_a_signal_not_an_action(self):
        assert ImputeType.THRESHOLD_VIOLATED in SIGNALS
        assert ImputeType.THRESHOLD_VIOLATED not in ACTIONS

    def test_a_filled_cell_the_threshold_nulled_keeps_both(self):
        recorded = (
            ImputeType.SCALED_FILL
            | ImputeType.HOUSEHOLD_SCALED_AVERAGE
            | ImputeType.THRESHOLD_VIOLATED
            | ImputeType.NULLED_POST_IMPUTATION
        )
        assert recorded & ImputeType.FILLED, "the fill is still in the history"
        assert recorded & ImputeType.NULLED_POST_IMPUTATION
        assert not (recorded & ImputeType.NOT_FILLED)


class TestValuesAreStable:
    """Bits are written into every imputed dataset, so a member's value is part
    of what the data means and cannot move once written.

    The values themselves are not restated here. They are configuration, recorded
    in etdworkflow's generated flags.json and beside every data folder, where a
    change is reviewed as a diff. Transcribing them into this file would mean
    editing it whenever the configuration legitimately changed, which checks
    nothing. What is checked here is the property that makes the recording
    trustworthy: appending is the only way a member arrives."""

    def test_single_bits_are_distinct(self):
        # Canonical members only. __members__ also yields aliases -- a composite
        # naming one bit, such as MODIFIERS -- which would read as a duplicate.
        singles = [m.value for m in _single_bits()]
        assert len(singles) == len(set(singles)), "two members share a bit"

    def test_declaration_order_is_ascending(self):
        """auto() ties a value to its source position, so a member out of order
        means the source was reordered and every later value moved with it --
        which silently reinterprets every dataset written before the change."""
        singles = [m.value for m in ImputeType if m.name and m.value]
        assert singles == sorted(singles), (
            "members are not in ascending value order, so one was inserted "
            "rather than appended"
        )

    def test_new_members_were_appended(self):
        for name in ("NO_CUMULATIVE_VALUE", "ONLY_CLOSING_READING",
                     "ONLY_OPENING_READING",
                     "THRESHOLD_VIOLATED", "NULLED", "NULLED_POST_IMPUTATION"):
            assert ImputeType[name].value > ImputeType.NOT_FILLED.value, (
                f"{name} was inserted rather than appended, which renumbers "
                f"every member after it"
            )


class TestTheCategoriesAreDeclaredNotDescribed:
    """SIGNALS and ACTIONS are composites on the class, so a report can label any
    flag without keeping its own copy of the split. A copy is what drifts."""

    def test_every_single_bit_is_in_exactly_one_category(self):
        for member in _single_bits():
            in_signals = bool(member & ImputeType.SIGNALS)
            in_actions = bool(member & ImputeType.ACTIONS)
            assert in_signals != in_actions, (
                f"{member.name} is in "
                f"{'both' if in_signals else 'neither'} category"
            )

    def test_the_declared_categories_match_this_file(self):
        for group, expected, label in (
            (ImputeType.SIGNALS, SIGNALS, "SIGNALS"),
            (ImputeType.ACTIONS, ACTIONS, "ACTIONS"),
        ):
            assert {m.name for m in _single_bits() if m & group} == {
                m.name for m in expected
            }, f"{label} on the class disagrees with the list in this file"

    def test_the_categories_do_not_overlap(self):
        assert not (ImputeType.SIGNALS & ImputeType.ACTIONS)

    def test_the_action_composites_hold_only_actions(self):
        """FILLED and IMPUTED_VALUE select what was written, so a signal in
        either would select cells nothing was written to."""
        assert not (ImputeType.FILLED & ImputeType.SIGNALS)
        assert not (ImputeType.IMPUTED_VALUE & ImputeType.SIGNALS)
