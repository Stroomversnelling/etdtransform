"""ImputeType members divide into signals and actions.

A signal names the condition a cell was in. An action names what imputation did
about it.

Imputation does exactly one thing to any cell: it writes a value, writes
nothing, or blanks it. So exactly one of GAP_FILLED, GAP_NOT_FILLED, GAP_NULLED applies.
The threshold pass runs afterwards and records its own action separately, which
is why a cell can read GAP_FILLED and NULLED_POST_IMPUTATION at once -- that is the
history, not a contradiction.
"""

import pytest

from etdtransform.vectorized_impute import ImputeType


def _single_bits():
    return {m for m in ImputeType if m.name and bin(m.value).count("1") == 1}


class TestTheTaxonomyIsComplete:
    def test_flag_kind_answers_for_every_member(self):
        """Every single-bit member is in exactly one of the SIGNALS and ACTIONS
        composites, which is what lets flag_kind answer without raising. The
        composites on the class are the only statement of the split; whether
        each individual flag sits in the right one is proven by the behavioural
        test its producer must have, not by a second list here."""
        from etdmap.flags import flag_kind

        for member in _single_bits():
            assert flag_kind(member) in ("signal", "action"), member.name


class TestFilledIsTheComplementOfNotFilled:
    """GAP_FILLED collects the actions that wrote a value, so a reader can select
    cells carrying an imputed number without naming every fill method."""

    def test_filled_is_the_value_writing_actions(self):
        assert ImputeType.GAP_FILLED == (
            ImputeType.UNIFORM_FILL
            | ImputeType.SCALED_FILL
            | ImputeType.ZERO_FILL
            | ImputeType.UNSCALED_AVERAGE
            | ImputeType.HOUSEHOLD_SCALED_AVERAGE
        )

    def test_filled_excludes_not_filled(self):
        assert not (ImputeType.GAP_FILLED & ImputeType.GAP_NOT_FILLED)

    def test_filled_excludes_the_blanking_actions(self):
        assert not (ImputeType.GAP_FILLED & ImputeType.GAP_NULLED)
        assert not (ImputeType.GAP_FILLED & ImputeType.NULLED_POST_IMPUTATION)

    @pytest.mark.parametrize("method", [
        ImputeType.UNIFORM_FILL,
        ImputeType.SCALED_FILL,
        ImputeType.ZERO_FILL,
        ImputeType.UNSCALED_AVERAGE,
        ImputeType.HOUSEHOLD_SCALED_AVERAGE,
    ])
    def test_each_fill_method_reaches_filled(self, method):
        assert method & ImputeType.GAP_FILLED


class TestImputedValue:
    """Every cell imputation changed, whichever way it changed it."""

    def test_it_excludes_not_filled(self):
        assert not (ImputeType.IMPUTED_VALUE & ImputeType.GAP_NOT_FILLED)

    def test_it_covers_the_fill_methods_and_the_blanking(self):
        assert ImputeType.GAP_FILLED & ImputeType.IMPUTED_VALUE
        assert ImputeType.GAP_NULLED & ImputeType.IMPUTED_VALUE
        assert ImputeType.NULLED_POST_IMPUTATION & ImputeType.IMPUTED_VALUE


class TestTheThresholdPassIsSeparate:
    """It runs after imputation, so its action does not belong to the
    exclusive triple and does not clear what imputation recorded."""

    def test_threshold_violated_is_a_signal_not_an_action(self):
        from etdmap.flags import flag_kind

        assert flag_kind(ImputeType.THRESHOLD_VIOLATED) == "signal"

    def test_a_filled_cell_the_threshold_nulled_keeps_both(self):
        recorded = (
            ImputeType.SCALED_FILL
            | ImputeType.HOUSEHOLD_SCALED_AVERAGE
            | ImputeType.THRESHOLD_VIOLATED
            | ImputeType.NULLED_POST_IMPUTATION
        )
        assert recorded & ImputeType.GAP_FILLED, "the fill is still in the history"
        assert recorded & ImputeType.NULLED_POST_IMPUTATION
        assert not (recorded & ImputeType.GAP_NOT_FILLED)


class TestValuesAreStable:
    """Bits are written into every imputed household batch, so a member's value is part
    of what the data means and cannot move once written.

    The values themselves are not restated here. They are configuration, recorded
    in the ``flags.json`` written beside every data folder, where a change is
    reviewed as a diff. Transcribing them into this file would mean
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
        which silently reinterprets every household batch written before the change."""
        singles = [m.value for m in ImputeType if m.name and m.value]
        assert singles == sorted(singles), (
            "members are not in ascending value order, so one was inserted "
            "rather than appended"
        )

    def test_new_members_were_appended(self):
        for name in ("NO_CUMULATIVE_VALUE", "ONLY_CLOSING_READING",
                     "ONLY_OPENING_READING",
                     "THRESHOLD_VIOLATED", "GAP_NULLED", "NULLED_POST_IMPUTATION"):
            assert ImputeType[name].value > ImputeType.GAP_NOT_FILLED.value, (
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

    def test_the_categories_do_not_overlap(self):
        assert not (ImputeType.SIGNALS & ImputeType.ACTIONS)

    def test_the_action_composites_hold_only_actions(self):
        """GAP_FILLED and IMPUTED_VALUE select what was written, so a signal in
        either would select cells nothing was written to."""
        assert not (ImputeType.GAP_FILLED & ImputeType.SIGNALS)
        assert not (ImputeType.IMPUTED_VALUE & ImputeType.SIGNALS)

class TestEveryMemberHasAMeaning:
    """DESCRIPTIONS beside the class is the one place a flag's meaning lives.
    A member without one ships a bit nobody can read; a stale key describes a
    flag that does not exist."""

    def test_every_single_bit_is_described(self):
        from etdtransform.flags import DESCRIPTIONS

        missing = sorted(
            m.name for m in _single_bits() if m.name not in DESCRIPTIONS
        )
        assert not missing, f"members with no description: {missing}"

    def test_no_description_names_a_missing_member(self):
        from etdtransform.flags import DESCRIPTIONS

        stale = sorted(set(DESCRIPTIONS) - {m.name for m in _single_bits()})
        assert not stale, f"descriptions for members that do not exist: {stale}"

    def test_the_mapping_class_is_held_to_the_same_bar(self):
        from etdmap.flags import DESCRIPTIONS as MAPPING
        from etdmap.flags import MappingFlag

        members = {
            m.name for m in MappingFlag
            if m.name and bin(m.value).count("1") == 1
        }
        assert members == set(MAPPING)
