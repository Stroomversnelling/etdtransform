"""`narrow_bounds_by_row` narrows a column's bound per row, from the row's knowns.

The package fixes the cadence: every declared bound in it is already scaled to
the grid, and the knowns substituted into a rule are values on that same grid.
Nothing here defaults to five minutes; every case builds its package at an
explicit cadence and several cases hold the same data against more than one.
"""

import pandas as pd
import pytest
from etdmap.flags import MappingFlag
from test_constraint_package import _constraints, _flat, _interval, _thresholds

from etdtransform.constraints import build_constraint_package, narrow_bounds_by_row

pytestmark = pytest.mark.unit


def _package(table, cadence, **thresholds):
    return build_constraint_package(
        table, cadence_minutes=cadence, thresholds=_thresholds(**thresholds)
    )


def _frame(**columns):
    n = max(len(v) for v in columns.values())
    data = {}
    for name, values in columns.items():
        if name.endswith("_flags"):
            data[name] = pd.array(values, dtype="Int64")
        else:
            data[name] = pd.array(values, dtype="Float64")
    return pd.DataFrame(data, index=range(n))


DIFFERENCE = (("T", "A - B", {"A": 1, "B": -1}),)


class TestNothingKnownMeansThePresolvedBound:
    @pytest.mark.parametrize("cadence", [5, 15, 60, 1440])
    def test_every_row_carries_the_package_bound(self, cadence):
        package = _package(
            _constraints(*DIFFERENCE), cadence,
            T=_interval(-10.0, 10.0), A=_interval(0.0, 2.0), B=_interval(0.0, 1.0),
        )
        frame = _frame(T=[pd.NA, pd.NA], A=[pd.NA, pd.NA], B=[pd.NA, pd.NA])
        lo, hi, from_imputed = narrow_bounds_by_row(frame, "T", package)
        expected_lo, expected_hi = package.bounds["T"]
        assert lo.astype(float).tolist() == [expected_lo] * 2
        assert hi.astype(float).tolist() == [expected_hi] * 2
        assert not from_imputed.any()

    @pytest.mark.parametrize("cadence", [15, 60, 1440])
    def test_the_presolved_bound_is_the_grid_cadence_s_own(self, cadence):
        """T = A - B with everything unknown pins T to what A and B may hold at
        THIS grid, through etdmap's own upper-bound shape -- not a rescaling of
        the five-minute figures."""
        from etdmap.data_model import scale_upper_bound

        package = _package(
            _constraints(*DIFFERENCE), cadence,
            T=_interval(-10.0, 10.0), A=_interval(0.0, 2.0),
            B=_interval(0.0, 1.0),
        )
        frame = _frame(T=[pd.NA], A=[pd.NA], B=[pd.NA])
        lo, hi, _ = narrow_bounds_by_row(frame, "T", package)
        n = cadence / 5
        assert float(lo.iloc[0]) == pytest.approx(-scale_upper_bound(1.0, n))
        assert float(hi.iloc[0]) == pytest.approx(scale_upper_bound(2.0, n))


class TestKnownsNarrowTheRow:
    def test_a_known_variable_substitutes_its_value(self):
        # T = A - B with A known at 1.0 and B in [0, 1]: T sits in [0, 1].
        package = _package(
            _constraints(*DIFFERENCE), 5,
            T=_interval(-10.0, 10.0), A=_interval(0.0, 2.0), B=_interval(0.0, 1.0),
        )
        frame = _frame(T=[pd.NA], A=[1.0], B=[pd.NA])
        lo, hi, _ = narrow_bounds_by_row(frame, "T", package)
        assert float(lo.iloc[0]) == 0.0
        assert float(hi.iloc[0]) == 1.0

    def test_rows_differ_when_their_knowns_differ(self):
        package = _package(
            _constraints(*DIFFERENCE), 5,
            T=_interval(-10.0, 10.0), A=_interval(0.0, 2.0), B=_interval(0.0, 1.0),
        )
        frame = _frame(T=[pd.NA, pd.NA], A=[1.0, pd.NA], B=[pd.NA, 0.25])
        lo, hi, _ = narrow_bounds_by_row(frame, "T", package)
        assert (float(lo.iloc[0]), float(hi.iloc[0])) == (0.0, 1.0)
        # Row 1: A unknown in [0, 2], B known at 0.25: T in [-0.25, 1.75].
        assert (float(lo.iloc[1]), float(hi.iloc[1])) == (-0.25, 1.75)

    def test_the_signs_pick_the_endpoints(self):
        # B enters negatively, so the low end of T takes B's HIGH end.
        package = _package(
            _constraints(("T", "A - B", {"A": 1, "B": -1})), 5,
            T=_interval(-10.0, 10.0), A=_interval(0.0, 2.0), B=_interval(0.0, 3.0),
        )
        frame = _frame(T=[pd.NA], A=[pd.NA], B=[pd.NA])
        lo, hi, _ = narrow_bounds_by_row(frame, "T", package)
        assert float(lo.iloc[0]) == -3.0
        assert float(hi.iloc[0]) == 2.0

    def test_the_row_bound_never_leaves_the_package_bound(self):
        # The rule allows more than the target's own declared range permits.
        package = _package(
            _constraints(*DIFFERENCE), 5,
            T=_interval(0.0, 0.5), A=_interval(0.0, 2.0), B=_interval(0.0, 1.0),
        )
        frame = _frame(T=[pd.NA], A=[2.0], B=[0.0])
        lo, hi, _ = narrow_bounds_by_row(frame, "T", package)
        assert float(hi.iloc[0]) == 0.5

    def test_narrowing_happens_at_the_cadence_not_at_five_minutes(self):
        """A known on a 60-minute grid meets endpoints scaled to 60 minutes:
        B's high end there follows the declared upper-bound shape, not the
        declared five-minute figure and not a linear rescale of it."""
        from etdmap.data_model import scale_upper_bound

        package = _package(
            _constraints(*DIFFERENCE), 60,
            T=_interval(-10.0, 10.0), A=_interval(0.0, 2.0), B=_interval(0.0, 1.0),
        )
        frame = _frame(T=[pd.NA], A=[6.0], B=[pd.NA])
        lo, hi, _ = narrow_bounds_by_row(frame, "T", package)
        b_hi_60 = scale_upper_bound(1.0, 60 / 5)
        assert b_hi_60 not in (1.0, 12.0)
        assert float(lo.iloc[0]) == pytest.approx(6.0 - b_hi_60)
        assert float(hi.iloc[0]) == pytest.approx(6.0)  # 6 - B_lo(60m) = 6 - 0


class TestWhatCountsAsKnown:
    def test_a_flagged_value_is_not_a_known(self):
        package = _package(
            _constraints(*DIFFERENCE), 5,
            T=_interval(-10.0, 10.0), A=_interval(0.0, 2.0), B=_interval(0.0, 1.0),
        )
        frame = _frame(
            T=[pd.NA, pd.NA], A=[1.0, 1.0], B=[pd.NA, pd.NA],
            A_flags=[MappingFlag.FROZEN_SPAN.value, MappingFlag.NONE.value],
        )
        lo, hi, _ = narrow_bounds_by_row(frame, "T", package)
        # Row 0's A is condemned, so its row keeps the presolved bound.
        assert float(lo.iloc[0]) == package.bounds["T"][0]
        assert float(lo.iloc[1]) == 0.0

    def test_a_variable_the_frame_does_not_carry_speaks_through_its_range(self):
        """A constraint naming a column this stage cannot impute still applies:
        the absent variable contributes its declared interval."""
        package = _package(
            _constraints(*DIFFERENCE), 5,
            T=_interval(-10.0, 10.0), A=_interval(0.0, 2.0), B=_interval(0.0, 1.0),
        )
        frame = _frame(T=[pd.NA], A=[1.5])  # no B column at all
        lo, hi, _ = narrow_bounds_by_row(frame, "T", package)
        assert float(lo.iloc[0]) == 0.5
        assert float(hi.iloc[0]) == 1.5

    def test_a_delivered_value_of_an_unimputable_variable_is_respected(self):
        # A flat (algemeen) bound does not scale; its delivered value narrows
        # exactly the same way at any cadence.
        for cadence in (5, 60):
            package = _package(
                _constraints(("T", "A * P / 100", {"A": 1, "P": 1})), cadence,
                T=_interval(0.0, 100.0), A=_interval(0.0, 2.0), P=_flat(0.0, 100.0),
            )
            frame = _frame(T=[pd.NA], A=[pd.NA], P=[50.0])
            lo, hi, _ = narrow_bounds_by_row(frame, "T", package)
            assert float(lo.iloc[0]) == 0.0
            assert float(hi.iloc[0]) == pytest.approx(
                package.bounds["A"][1] * 0.5
            )


class TestUnboundedSides:
    def test_an_unbounded_side_comes_back_as_na(self):
        package = _package(_constraints(), 5, T=_interval(0.0, pd.NA))
        frame = _frame(T=[pd.NA])
        lo, hi, _ = narrow_bounds_by_row(frame, "T", package)
        assert float(lo.iloc[0]) == 0.0
        assert hi.isna().all()

    def test_a_zero_known_times_an_unbounded_unknown_narrows_nothing(self):
        """0 * inf is nan, and an undecidable end must not narrow."""
        package = _package(
            _constraints(("T", "A * B", {"A": 1, "B": 1})), 5,
            T=_interval(0.0, pd.NA), A=_interval(0.0, 2.0),
            B=_interval(0.0, pd.NA),
        )
        frame = _frame(T=[pd.NA], A=[0.0], B=[pd.NA])
        lo, hi, _ = narrow_bounds_by_row(frame, "T", package)
        assert float(lo.iloc[0]) == 0.0
        assert hi.isna().all()


class TestBoundsFedByTheFill:
    def test_an_imputed_known_that_tightens_is_marked(self):
        package = _package(
            _constraints(*DIFFERENCE), 5,
            T=_interval(-10.0, 10.0), A=_interval(0.0, 2.0), B=_interval(0.0, 1.0),
        )
        frame = _frame(T=[pd.NA, pd.NA], A=[1.0, 1.0], B=[pd.NA, pd.NA])
        imputed = {"A": pd.Series([True, False], dtype="boolean")}
        _, _, from_imputed = narrow_bounds_by_row(
            frame, "T", package, imputed=imputed
        )
        assert from_imputed.tolist() == [True, False]

    def test_an_imputed_known_that_tightens_nothing_is_not_marked(self):
        # The rule's implication is wider than T's own bound everywhere.
        package = _package(
            _constraints(*DIFFERENCE), 5,
            T=_interval(0.0, 0.1), A=_interval(0.0, 2.0), B=_interval(0.0, 1.0),
        )
        frame = _frame(T=[pd.NA], A=[1.0], B=[pd.NA])
        imputed = {"A": pd.Series([True], dtype="boolean")}
        _, _, from_imputed = narrow_bounds_by_row(
            frame, "T", package, imputed=imputed
        )
        assert not from_imputed.any()
