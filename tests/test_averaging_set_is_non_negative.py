"""The averaging set cannot contain a negative value, and why that is not luck.

Three things make a negative unreachable by the mean:

1. `assert_averageable` refuses at the entry point any column that is not the
   interval form of a variable declared `Monotoon`. Accumulation is what makes
   an interval a volume the average may sum, scale and redistribute.
2. `assert_no_negative_interval` raises on the way in if one arrives anyway. It
   is a data-model error, not a value to filter out: mapping is what blanks a
   decreasing reading, and a negative here means it did not.
3. Where a meter does go down, mapping classifies the cause into one of the
   `CUMULATIVE_UNUSABLE` findings and blanks the readings. Exhaustiveness of
   that classification is tested in etdmap
   (`test_cases_are_mutually_exclusive_and_cover_every_negative_diff`).

Note what is NOT relied on. `Cumulatief=ja` says a variable is a running total,
which is a statement about meaning, not about sign; `Monotoon` is the separate
declaration that its values do not decrease.

These tests pin (1) and (2). `Netuitwisseling` and the two
`ElektriciteitsgebruikTotaal*` columns are the obvious candidates for being
added to the averaging set by mistake -- they are interval columns that can go
negative by design.
"""

import pandas as pd
import pytest

from etdmap.data_model import (
    cumulative_columns,
    interval_column,
    monotonic_interval_columns,
    thresholds_dict,
)
from etdtransform.impute import (
    assert_averageable,
    assert_no_negative_interval,
    get_diff_columns,
)

pytestmark = pytest.mark.unit

AVERAGED = get_diff_columns(cumulative_columns)


def test_the_averaging_set_is_not_empty():
    """A vacuous set would make every other test here pass for nothing.

    The floor is 18 since 2026-10-01: the four Hoog/Laag tariff splits left
    the averaging set (get_diff_columns drops them -- the combined root is
    averaged instead; the discrete alternating splits are not physically
    averageable)."""
    assert len(AVERAGED) >= 18, f"only {len(AVERAGED)} columns averaged"


def test_tariff_splits_are_not_averaged():
    """The splits' departure is deliberate, so pin it here beside the floor."""
    from etdmap.data_model import tariff_partner_columns

    leaked = [c for c in AVERAGED if c in tariff_partner_columns]
    assert leaked == [], f"tariff splits back in the averaging set: {leaked}"


@pytest.mark.parametrize("column", AVERAGED)
def test_every_averaged_column_accumulates(column):
    assert column in monotonic_interval_columns, (
        f"{column} is averaged but its cumulative variable is not declared "
        f"Monotoon, so its values may decrease and the average would treat a "
        f"decrease as a volume"
    )


@pytest.mark.parametrize("column", AVERAGED)
def test_every_averaged_column_is_one_half_of_a_declared_pair(column):
    """Provenance, not sign: the column must be the declared complement of a
    cumulative variable rather than merely a name ending in Diff."""
    sources = [c for c in cumulative_columns if interval_column(c) == column]
    assert len(sources) == 1, (
        f"{column} is averaged but is the interval form of {len(sources)} "
        f"variables: {sources}"
    )


class TestTheEntryPointRefusesWhatCannotBeAveraged:
    def test_the_real_set_passes(self):
        assert_averageable(AVERAGED)

    def test_a_column_that_may_decrease_is_refused(self):
        with pytest.raises(ValueError, match="Monotoon"):
            assert_averageable(["Netuitwisseling"])

    def test_a_column_with_no_declared_complement_is_refused(self):
        """An unresolvable link is not a declaration that the variable
        accumulates, so it is refused rather than assumed."""
        with pytest.raises(ValueError, match="Monotoon"):
            assert_averageable(["NotAColumnInTheDataModel"])

    def test_the_message_names_the_offending_column(self):
        with pytest.raises(ValueError, match="Netuitwisseling"):
            assert_averageable([*AVERAGED, "Netuitwisseling"])


class TestANegativeIntervalStops():
    """Filtering a negative out would hide that mapping did not do its job."""

    def test_a_negative_raises(self):
        with pytest.raises(ValueError, match="negative interval"):
            assert_no_negative_interval("GasgebruikDiff", -0.25)

    def test_the_message_carries_the_value(self):
        with pytest.raises(ValueError, match="-0.25"):
            assert_no_negative_interval("GasgebruikDiff", -0.25)

    @pytest.mark.parametrize("smallest", [0.0, 0.25, None, pd.NA])
    def test_anything_else_passes(self, smallest):
        assert_no_negative_interval("GasgebruikDiff", smallest)


def test_a_negative_capable_column_is_not_in_the_averaging_set():
    """Named explicitly, because these are the ones that would break the
    assumption if the averaging set ever grew."""
    for column in (
        "Netuitwisseling",
        "ElektriciteitsgebruikTotaalHuishoudelijk",
        "ElektriciteitsgebruikTotaalGebouwgebonden",
    ):
        lower = thresholds_dict.get(column, {}).get("Min")
        if lower is None or pd.isna(lower) or float(lower) >= 0:
            continue  # its declared floor changed; nothing to guard here
        assert column not in AVERAGED, (
            f"{column} declares Min={lower} and is now averaged. Decide what "
            f"should happen to its negative values before including it."
        )
