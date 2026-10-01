"""`clip_to_bounds` pushes a value into a range and says what it did.

Every route that writes an imputed value passes through it, so the masks it
returns are what the caller turns into flags: *moved* becomes the clip action,
*empty* becomes the infeasible-bound signal and leaves the cell unwritten.
"""

import pandas as pd
import pytest

from etdtransform.bounds import clip_to_bounds


def _s(*values):
    return pd.array(values, dtype="Float64")


def _series(*values):
    return pd.Series(_s(*values))


class TestScalarBounds:
    def test_a_value_above_the_cap_comes_down_to_it(self):
        out, moved, empty = clip_to_bounds(_series(5.0), 0.0, 2.0)
        assert out.tolist() == [2.0]
        assert moved.tolist() == [True]
        assert empty.tolist() == [False]

    def test_a_value_below_the_floor_comes_up_to_it(self):
        out, moved, _ = clip_to_bounds(_series(-1.0), 0.0, 2.0)
        assert out.tolist() == [0.0]
        assert moved.tolist() == [True]

    def test_a_value_inside_the_range_is_untouched_and_unmarked(self):
        out, moved, _ = clip_to_bounds(_series(1.0), 0.0, 2.0)
        assert out.tolist() == [1.0]
        assert moved.tolist() == [False]

    def test_a_value_exactly_on_a_bound_is_not_moved(self):
        out, moved, _ = clip_to_bounds(_series(0.0, 2.0), 0.0, 2.0)
        assert out.tolist() == [0.0, 2.0]
        assert moved.tolist() == [False, False]


class TestAnAbsentBoundIsNoBound:
    def test_na_cap_leaves_the_high_side_unbounded(self):
        out, moved, _ = clip_to_bounds(_series(1e9), 0.0, pd.NA)
        assert out.tolist() == [1e9]
        assert moved.tolist() == [False]

    def test_na_floor_leaves_the_low_side_unbounded(self):
        out, moved, _ = clip_to_bounds(_series(-1e9), pd.NA, 2.0)
        assert out.tolist() == [-1e9]
        assert moved.tolist() == [False]

    def test_both_absent_changes_nothing(self):
        out, moved, empty = clip_to_bounds(_series(-5.0, 5.0), pd.NA, pd.NA)
        assert out.tolist() == [-5.0, 5.0]
        assert moved.tolist() == [False, False]
        assert empty.tolist() == [False, False]

    def test_none_is_treated_as_absent(self):
        out, moved, _ = clip_to_bounds(_series(1e9), None, None)
        assert out.tolist() == [1e9]
        assert moved.tolist() == [False]


class TestPerRowBounds:
    def test_each_row_takes_its_own_pair(self):
        out, moved, _ = clip_to_bounds(
            _series(5.0, 5.0, 5.0),
            _series(0.0, 0.0, 0.0),
            _series(2.0, 10.0, 4.0),
        )
        assert out.tolist() == [2.0, 5.0, 4.0]
        assert moved.tolist() == [True, False, True]

    def test_na_within_a_series_bound_leaves_that_row_unbounded(self):
        out, moved, _ = clip_to_bounds(
            _series(5.0, 5.0),
            _series(0.0, 0.0),
            _series(2.0, pd.NA),
        )
        assert out.tolist() == [2.0, 5.0]
        assert moved.tolist() == [True, False]

    def test_a_series_bound_may_be_mixed_with_a_scalar(self):
        out, moved, _ = clip_to_bounds(
            _series(5.0, 5.0), 0.0, _series(2.0, 10.0)
        )
        assert out.tolist() == [2.0, 5.0]
        assert moved.tolist() == [True, False]


class TestAnEmptyRangeAllowsNothing:
    def test_lo_above_hi_writes_no_value(self):
        out, moved, empty = clip_to_bounds(_series(1.0), 5.0, 2.0)
        assert out.isna().tolist() == [True]
        assert empty.tolist() == [True]

    def test_an_empty_range_is_not_reported_as_a_clip(self):
        """The two are different findings: one is an action on a value, the
        other says no value was permitted. A caller flags them separately."""
        _, moved, empty = clip_to_bounds(_series(1.0), 5.0, 2.0)
        assert moved.tolist() == [False]
        assert empty.tolist() == [True]

    def test_only_the_rows_with_an_empty_range_are_emptied(self):
        out, _, empty = clip_to_bounds(
            _series(1.0, 1.0),
            _series(5.0, 0.0),
            _series(2.0, 2.0),
        )
        assert out.isna().tolist() == [True, False]
        assert out.tolist()[1] == 1.0
        assert empty.tolist() == [True, False]

    def test_equal_bounds_are_not_empty(self):
        out, moved, empty = clip_to_bounds(_series(1.0), 2.0, 2.0)
        assert out.tolist() == [2.0]
        assert moved.tolist() == [True]
        assert empty.tolist() == [False]


class TestMissingValues:
    def test_na_in_stays_na_out_and_is_not_marked(self):
        out, moved, empty = clip_to_bounds(pd.Series(_s(pd.NA)), 0.0, 2.0)
        assert out.isna().tolist() == [True]
        assert moved.tolist() == [False]
        assert empty.tolist() == [False]

    def test_an_na_value_in_an_empty_range_is_still_reported_empty(self):
        _, moved, empty = clip_to_bounds(pd.Series(_s(pd.NA)), 5.0, 2.0)
        assert moved.tolist() == [False]
        assert empty.tolist() == [True]


class TestTheContract:
    def test_the_result_keeps_the_nullable_dtype(self):
        out, _, _ = clip_to_bounds(_series(5.0), 0.0, 2.0)
        assert out.dtype == "Float64"

    def test_the_masks_are_plain_booleans_with_no_na(self):
        _, moved, empty = clip_to_bounds(
            _series(5.0, pd.NA), _series(0.0, pd.NA), _series(2.0, pd.NA)
        )
        for mask in (moved, empty):
            assert mask.dtype == bool
            assert not mask.isna().any()

    def test_the_input_is_not_modified(self):
        values = _series(5.0)
        clip_to_bounds(values, 0.0, 2.0)
        assert values.tolist() == [5.0]

    def test_the_index_is_preserved(self):
        values = pd.Series(_s(5.0, 5.0), index=[10, 20])
        out, moved, empty = clip_to_bounds(values, 0.0, 2.0)
        for result in (out, moved, empty):
            assert result.index.tolist() == [10, 20]

    def test_clipping_is_idempotent(self):
        once, _, _ = clip_to_bounds(_series(5.0, -1.0, 1.0), 0.0, 2.0)
        twice, moved, _ = clip_to_bounds(once, 0.0, 2.0)
        assert twice.tolist() == once.tolist()
        assert moved.tolist() == [False, False, False]

    def test_an_empty_frame_returns_empty_results(self):
        out, moved, empty = clip_to_bounds(pd.Series(_s()), 0.0, 2.0)
        assert out.empty and moved.empty and empty.empty

    @pytest.mark.parametrize("value", [0.0, 1.0, 2.0])
    def test_nothing_ends_outside_the_range(self, value):
        out, _, _ = clip_to_bounds(_series(value - 10, value, value + 10), 0.0, 2.0)
        assert ((out >= 0.0) & (out <= 2.0)).all()
