"""The fill reads its average through the row's narrowed bound.

With a constraint table handed to `impute_and_normalize`, each row's bound for
the column comes from the rules and the row's own known values. The average is
clipped into that bound before any route reads it (`CLIPPED_AVERAGE` where a
clipped average was used), and a delivered value outside the narrowed bound --
while inside the bound its own column declares -- is signalled with
`DELIVERED_OUTSIDE_RULE_BOUND` and never touched.
"""

import pandas as pd
import pytest
from conftest import cadence_table
from test_constraint_package import _constraints

from etdtransform.vectorized_impute import ImputeType, impute_and_normalize

pytestmark = pytest.mark.unit

CUM = "ElektriciteitNetgebruik"
DIFF = "ElektriciteitNetgebruikDiff"
AVG = f"{DIFF}_avg"
ITYPE = f"{DIFF}_impute_type"

# The row bound follows K exactly: DIFF = K, with K unbounded in the
# thresholds, so every narrowing in these tests is the row's own.
FOLLOWS_K = (((DIFF), "K", {"K": 1}),)


def _frame(cumulative, averages, diffs=None, k=None):
    n = len(cumulative)
    data = {
        "HuisIdBSV": pd.array([1] * n, dtype="Int64"),
        "ProjectIdBSV": pd.array([1] * n, dtype="Int64"),
        "ReadingDate": pd.date_range("2026-01-01", periods=n, freq="5min"),
        CUM: pd.array(cumulative, dtype="Float64"),
        DIFF: pd.array([pd.NA] * n if diffs is None else diffs, dtype="Float64"),
        AVG: pd.array(averages, dtype="Float64"),
    }
    if k is not None:
        data["K"] = pd.array(k, dtype="Float64")
    return pd.DataFrame(data)


def _impute(frame, constraints=None):
    out, _, _ = impute_and_normalize(
        frame,
        [CUM],
        "ProjectIdBSV",
        cadence=cadence_table(frame, [CUM]),
        constraints=constraints,
    )
    return out


def _types(out, rows):
    return [
        pd.NA if pd.isna(v) else int(v) for v in out.loc[rows, ITYPE]
    ]


class TestTheAverageIsClippedIntoTheRowBound:
    def test_a_clipped_average_that_was_used_is_signalled_per_cell(self):
        # Jump of 3 over three cells. K pins each row's bound to [1, 1], so the
        # shape 0.5 : 1.0 : 1.5 becomes 1 : 1 : 1 and the fill is even.
        out = _impute(
            _frame(
                [100.0, pd.NA, pd.NA, 103.0],
                [pd.NA, 0.5, 1.0, 1.5],
                k=[pd.NA, 1.0, 1.0, 1.0],
            ),
            constraints=_constraints(*FOLLOWS_K),
        )
        assert out.loc[1:3, DIFF].astype(float).tolist() == pytest.approx([1.0] * 3)
        clipped = [
            t & ImputeType.CLIPPED_AVERAGE.value for t in _types(out, slice(1, 3))
        ]
        # The middle cell's average already sat at 1.0, so only its neighbours
        # were clipped: the signal is per cell, not per gap.
        assert [bool(c) for c in clipped] == [True, False, True]

    def test_a_clipped_average_no_route_read_carries_no_signal(self):
        # A jump of zero fills zeros without reading the average, however
        # tightly the rules would have clipped it.
        out = _impute(
            _frame(
                [100.0, pd.NA, pd.NA, 100.0],
                [pd.NA, 0.5, 0.5, pd.NA],
                k=[pd.NA, 0.2, 0.2, pd.NA],
            ),
            constraints=_constraints(*FOLLOWS_K),
        )
        assert out.loc[1:2, DIFF].astype(float).tolist() == [0.0, 0.0]
        for value in _types(out, slice(1, 2)):
            assert not value & ImputeType.CLIPPED_AVERAGE.value

    def test_the_average_column_itself_is_not_rewritten(self):
        frame = _frame(
            [100.0, pd.NA, pd.NA, 103.0],
            [pd.NA, 0.5, 1.0, 1.5],
            k=[pd.NA, 1.0, 1.0, 1.0],
        )
        before = frame[AVG].copy()
        out = _impute(frame, constraints=_constraints(*FOLLOWS_K))
        pd.testing.assert_series_equal(out[AVG], before, check_names=False)
        assert f"{AVG}_bounded" not in out.columns


class TestADeliveredValueOutsideTheRuleBound:
    def test_it_is_signalled_and_left_alone(self):
        # 1.2 is fine for the column (declared max 1.5) but K narrows its row
        # to [0.5, 0.5]. One other value is missing so the fill has work to do.
        out = _impute(
            _frame(
                [100.0, 100.5, pd.NA, 102.0],
                [pd.NA, 1.0, 1.0, 1.0],
                diffs=[pd.NA, 1.2, pd.NA, pd.NA],
                k=[pd.NA, 0.5, pd.NA, pd.NA],
            ),
            constraints=_constraints(*FOLLOWS_K),
        )
        assert float(out.loc[1, DIFF]) == pytest.approx(1.2), (
            "the signal must not change the delivered value"
        )
        assert _types(out, slice(1, 1))[0] & ImputeType.DELIVERED_OUTSIDE_RULE_BOUND.value
        assert not out.loc[1, f"{DIFF}_is_imputed"]

    def test_a_value_outside_its_own_declared_bound_is_not_this_signal(self):
        # 2.0 breaks the declared 1.5, which is the threshold pass's case, not
        # this signal's. Row 0 delivers a real value: an opening NA would make
        # the 2.0 read as a frozen span's catch-up, which the pass keeps.
        out = _impute(
            _frame(
                [100.0, 102.0, pd.NA, 103.0],
                [pd.NA, 1.0, 1.0, 1.0],
                diffs=[0.5, 2.0, pd.NA, pd.NA],
                k=[pd.NA, 0.5, pd.NA, pd.NA],
            ),
            constraints=_constraints(*FOLLOWS_K),
        )
        value = _types(out, slice(1, 1))[0]
        assert not value & ImputeType.DELIVERED_OUTSIDE_RULE_BOUND.value
        assert value & ImputeType.THRESHOLD_VIOLATED.value

    def test_a_delivered_value_inside_the_rule_bound_is_silent(self):
        out = _impute(
            _frame(
                [100.0, 100.5, pd.NA, 102.0],
                [pd.NA, 1.0, 1.0, 1.0],
                diffs=[pd.NA, 0.5, pd.NA, pd.NA],
                k=[pd.NA, 0.5, pd.NA, pd.NA],
            ),
            constraints=_constraints(*FOLLOWS_K),
        )
        value = _types(out, slice(1, 1))[0]
        if pd.notna(value):
            assert not value & ImputeType.DELIVERED_OUTSIDE_RULE_BOUND.value


class TestTheOffSwitch:
    def test_no_constraints_means_no_phase_3_flags(self):
        out = _impute(
            _frame([100.0, pd.NA, pd.NA, 103.0], [pd.NA, 0.5, 1.0, 1.5]),
        )
        for value in _types(out, slice(1, 3)):
            assert not value & ImputeType.CLIPPED_AVERAGE.value
            assert not value & ImputeType.DELIVERED_OUTSIDE_RULE_BOUND.value

    def test_an_empty_table_leaves_the_fill_at_the_declared_bounds(self):
        with_table = _impute(
            _frame([100.0, pd.NA, pd.NA, 103.0], [pd.NA, 0.5, 1.0, 1.5]),
            constraints=_constraints(),
        )
        without = _impute(
            _frame([100.0, pd.NA, pd.NA, 103.0], [pd.NA, 0.5, 1.0, 1.5]),
        )
        pd.testing.assert_series_equal(with_table[DIFF], without[DIFF])
