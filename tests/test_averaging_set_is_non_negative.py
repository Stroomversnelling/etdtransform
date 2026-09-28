"""The averaging set cannot contain a negative value, and why that is not luck.

`calculate_average_diff` used to scan every column on every run for a negative
and raise if it found one. That scan is gone, because three things upstream make
a negative unreachable:

1. Every averaged column declares `Min = 0.0`, and the threshold filter discards
   what falls below `Min` before the mean is taken. **The discard is the
   guarantee, not the declaration** -- a column declaring a zero floor can still
   be delivered with a negative value; what makes the average safe is that the
   value is removed first.
2. `assert_averageable` refuses at the entry point any column whose declared
   `Min` is negative or absent, so the filter can always floor what it sees.
3. Where a meter does go down, mapping classifies the cause into one of the
   `CUMULATIVE_UNUSABLE` findings and blanks the readings. Exhaustiveness of that
   classification is tested in etdmap
   (`test_cases_are_mutually_exclusive_and_cover_every_negative_diff`).

Note what is NOT relied on. `Cumulatief=ja` says a variable is a running total,
which is a statement about meaning, not a promise that its values never
decrease -- meter resets, replacements and rollovers all break that, which is why
mapping has four findings for them. Monotonicity is assumed nowhere.

These tests pin (1) and (2). If someone adds a negative-capable column to the
averaging set -- `Netuitwisseling` and the two `ElektriciteitsgebruikTotaal*`
columns are `5-minute` interval columns with `Min = -2.01`, and are the obvious
candidates -- these fail, and the decision to drop the runtime check has to be
revisited rather than silently outlived.
"""

import pandas as pd
import pytest

from etdmap.data_model import (
    cumulative_columns,
    interval_column,
    thresholds_dict,
)
from etdtransform.impute import get_diff_columns

AVERAGED = get_diff_columns(cumulative_columns)


def test_the_averaging_set_is_not_empty():
    """A vacuous set would make every other test here pass for nothing."""
    assert len(AVERAGED) >= 20, f"only {len(AVERAGED)} columns averaged"


@pytest.mark.parametrize("column", AVERAGED)
def test_every_averaged_column_declares_a_non_negative_floor(column):
    declared = thresholds_dict.get(column, {})
    lower = declared.get("Min")
    assert lower is not None and pd.notna(lower), (
        f"{column} is averaged but declares no Min, so a negative would not be "
        f"discarded"
    )
    assert float(lower) >= 0.0, (
        f"{column} declares Min={lower}. A negative value can now reach the "
        f"project average, and the runtime check that used to catch it was "
        f"removed on the grounds that this could not happen."
    )


@pytest.mark.parametrize("column", AVERAGED)
def test_every_averaged_column_is_one_half_of_a_declared_pair(column):
    """Provenance, not sign: the column must be the declared complement of a
    cumulative variable rather than merely a name ending in Diff. This says
    nothing about whether its values can be negative -- that is the Min check."""
    sources = [c for c in cumulative_columns if interval_column(c) == column]
    assert len(sources) == 1, (
        f"{column} is averaged but is the interval form of {len(sources)} "
        f"variables: {sources}"
    )


def test_the_entry_point_refuses_a_negative_capable_column():
    from etdtransform.impute import assert_averageable

    assert_averageable(AVERAGED)  # the real set passes
    with pytest.raises(ValueError, match="Min="):
        assert_averageable(["Netuitwisseling"])


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
            f"{column} declares Min={lower} and is now averaged. The averaging "
            f"no longer checks for negatives; decide what should happen before "
            f"including it."
        )
