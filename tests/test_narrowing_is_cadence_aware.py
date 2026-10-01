"""Narrowing happens at the cadence, not at the declared interval then rescaled.

The rules add bounds together: `T <= A_max + B_max`. Rescaling that sum to a
coarser reporting interval gives the same answer as narrowing at that interval
only while every column's upper bound scales multiplicatively in its declared
Max -- because only then does `scale(A + B) == scale(A) + scale(B)`.

That holds by accident of today's metadata, where no column declares
`UpperThresholdSustained` or `UpperThresholdExponent`, so every bound takes one
shared default shape with a zero sustained level and the scaling is a plain
multiply. It is not a property of the model. Those fields exist to be used, and
the moment a column declares a sustained level its shape becomes affine with an
offset and the identity breaks.

So these tests declare the shapes rather than relying on the data, and they run
at 15 and 60 minutes as well as 5. There is no five-minute default anywhere in
the model, and a test that only ever exercises five minutes cannot see the
difference between the two orders of operations.
"""

import pandas as pd
import pytest

from etdmap.data_model import scale_bounds
from etdtransform.constraints import (
    CONSTRAINT_COLUMNS,
    build_constraint_package,
)

pytestmark = pytest.mark.unit

CADENCES = (5, 15, 60, 1440)


def _sum_constraint():
    """`T = A + B`, so T is bounded by the sum of A's and B's bounds."""
    records = [
        {
            "lhs": "T", "rhs_text": "A + B", "variable": variable, "sign": 1,
            "derivative_text": "1", "rhs_var_count": 2,
            "physical_models": ["Universeel"],
        }
        for variable in ("A", "B")
    ]
    return pd.DataFrame(records, columns=list(CONSTRAINT_COLUMNS)).astype(
        {"sign": "Int64", "rhs_var_count": "Int64"}
    )


def _thresholds(sustained=pd.NA, exponent=pd.NA):
    """A, B capped at 1.0 each; T declared far looser so the rules decide it."""
    def entry(maximum):
        return {
            "Min": 0.0,
            "Max": maximum,
            "ThresholdType": "5-minute",
            "UpperThresholdSustained": sustained,
            "UpperThresholdExponent": exponent,
        }

    return {"T": entry(999.0), "A": entry(1.0), "B": entry(1.0)}


def _narrow_at(cadence, thresholds):
    """What the implementation does: narrow with every input already scaled."""
    package = build_constraint_package(
        _sum_constraint(), cadence, thresholds=thresholds
    )
    return float(package.bounds["T"][1])


def _narrow_then_scale(cadence, thresholds):
    """The shortcut: narrow at the declared interval, then scale the result."""
    declared_narrowed = build_constraint_package(
        _sum_constraint(), None, thresholds=thresholds
    ).bounds["T"][1]
    entry = {**thresholds["T"], "Max": declared_narrowed}
    return float(scale_bounds(entry, cadence)[1])


class TestWithTodaysUnshapedThresholds:
    """No column declares a shape, so the scaling is a plain multiply and the
    two orders agree. This is the case the real data exercises, and on its own
    it cannot tell the two apart."""

    @pytest.mark.parametrize("cadence", CADENCES)
    def test_the_two_orders_agree(self, cadence):
        thresholds = _thresholds()
        assert _narrow_at(cadence, thresholds) == pytest.approx(
            _narrow_then_scale(cadence, thresholds)
        )

    def test_the_declared_interval_gives_the_sum_of_the_parts(self):
        assert _narrow_at(5, _thresholds()) == pytest.approx(2.0)


class TestWithASustainedLevelDeclared:
    """A non-zero sustained level makes the shape affine with an offset, so the
    sum no longer survives rescaling. This is the case that separates a correct
    implementation from the shortcut, and no amount of five-minute data shows
    it."""

    THRESHOLDS = _thresholds(sustained=0.2, exponent=0.5)

    @pytest.mark.parametrize("cadence", (15, 60, 1440))
    def test_the_two_orders_disagree(self, cadence):
        at_cadence = _narrow_at(cadence, self.THRESHOLDS)
        then_scaled = _narrow_then_scale(cadence, self.THRESHOLDS)
        assert at_cadence != pytest.approx(then_scaled), (
            f"at {cadence} minutes the shortcut happens to agree, so this "
            f"fixture no longer separates the two orders"
        )

    @pytest.mark.parametrize("cadence", (15, 60, 1440))
    def test_narrowing_at_the_cadence_is_the_sum_of_the_scaled_parts(self, cadence):
        """The correct answer by construction: T cannot exceed what A and B can
        each hold over an interval of this length."""
        scaled_a = float(scale_bounds(self.THRESHOLDS["A"], cadence)[1])
        scaled_b = float(scale_bounds(self.THRESHOLDS["B"], cadence)[1])
        assert _narrow_at(cadence, self.THRESHOLDS) == pytest.approx(
            scaled_a + scaled_b
        )

    @pytest.mark.parametrize("cadence", (15, 60, 1440))
    def test_the_shortcut_is_the_one_that_is_wrong(self, cadence):
        """Named explicitly so the failure says which side to trust."""
        scaled_sum = float(
            scale_bounds(self.THRESHOLDS["A"], cadence)[1]
        ) + float(scale_bounds(self.THRESHOLDS["B"], cadence)[1])
        assert _narrow_then_scale(cadence, self.THRESHOLDS) != pytest.approx(
            scaled_sum
        )

    def test_at_the_declared_interval_they_still_agree(self):
        """Scaling by one interval is the identity, so five minutes cannot
        distinguish them -- which is the whole point."""
        assert _narrow_at(5, self.THRESHOLDS) == pytest.approx(
            _narrow_then_scale(5, self.THRESHOLDS)
        )


class TestTheBoundStillTightensAtEveryCadence:
    @pytest.mark.parametrize("cadence", CADENCES)
    @pytest.mark.parametrize("shape", [(pd.NA, pd.NA), (0.2, 0.5)])
    def test_the_rules_beat_the_loose_declared_bound(self, cadence, shape):
        thresholds = _thresholds(*shape)
        declared = float(scale_bounds(thresholds["T"], cadence)[1])
        assert _narrow_at(cadence, thresholds) < declared

    @pytest.mark.parametrize("cadence", CADENCES)
    def test_a_tighter_declared_bound_wins(self, cadence):
        """Narrowing intersects; it never loosens what the column declares."""
        thresholds = _thresholds()
        thresholds["T"] = {**thresholds["T"], "Max": 0.1}
        assert _narrow_at(cadence, thresholds) == pytest.approx(
            float(scale_bounds(thresholds["T"], cadence)[1])
        )
