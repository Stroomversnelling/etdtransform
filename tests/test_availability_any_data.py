"""Availability is "holds at least one value". There is no fraction test.

The 0.95 fraction test is REMOVED, not configurable. It existed because a
missing cell used to be read as 0, so a column had to be nearly complete
before a formula over it could be trusted. With pd.NA propagating, a
partly-missing input yields a partly-missing output, which is the honest
answer.

It decided three things with one number: which formulas were feasible, which
columns became derivation TARGETS, and how households were grouped into plans.
The second meant a delivered column at 94% was overwritten by a formula,
losing the 94% that was measured.
"""

import pandas as pd

from etdtransform.catalog.query import DatasetAdapter

CATALOG = pd.DataFrame([
    {"lhs": "T", "rhs_text": "A + B", "rhs_vars": ["A", "B"], "rhs_var_count": 2},
])


def _frame(fractions, n=100):
    """A frame whose columns hold the given non-null fraction."""
    data = {}
    for column, fraction in fractions.items():
        present = int(round(fraction * n))
        data[column] = pd.array(
            [1.0] * present + [None] * (n - present), dtype="Float64"
        )
    return pd.DataFrame(data)


class TestAvailableColumns:
    def test_a_sparse_column_is_available(self):
        # One value in a hundred is enough: the formula will yield a value on
        # that row and pd.NA on the rest.
        adapter = DatasetAdapter(CATALOG)
        assert adapter.available_columns(_frame({"A": 0.01, "B": 1.0})) == {"A", "B"}

    def test_a_column_just_under_the_old_threshold_is_available(self):
        # 94% was the case that made a delivered column a derivation target.
        assert DatasetAdapter(CATALOG).available_columns(
            _frame({"A": 0.94, "B": 1.0})
        ) == {"A", "B"}

    def test_an_empty_column_is_not_available(self):
        # Nothing in it cannot contribute to any formula. This is the one
        # exclusion that survives, and it is not a fraction question.
        assert DatasetAdapter(CATALOG).available_columns(
            _frame({"A": 0.0, "B": 1.0})
        ) == {"B"}

    def test_the_adapter_takes_no_threshold(self):
        # The parameter is gone rather than defaulted, so no caller can
        # reintroduce the fraction test by passing one.
        import inspect

        parameters = inspect.signature(DatasetAdapter.__init__).parameters
        assert set(parameters) == {"self", "catalog_df"}


class TestFeasibilityConsequence:
    def test_a_sparse_input_makes_the_target_derivable(self):
        report = DatasetAdapter(CATALOG).feasibility_report(
            _frame({"A": 0.01, "B": 1.0}), {"T"}
        )
        assert "T" in report["derivable"]
        assert report["not_derivable"] == []

    def test_an_empty_input_still_blocks_the_target(self):
        report = DatasetAdapter(CATALOG).feasibility_report(
            _frame({"A": 0.0, "B": 1.0}), {"T"}
        )
        assert report["not_derivable"] == ["T"]


class TestTargetSelection:
    """The second consequence of the old test: a column that IS available is
    not a target, so no formula overwrites it."""

    def test_a_partly_complete_delivered_column_is_not_a_target(self):
        frame = _frame({"A": 0.94, "B": 1.0, "T": 0.94})
        available = DatasetAdapter(CATALOG).available_columns(frame)
        assert "T" in available
        # The expression the callers use to pick targets.
        assert ({"T"} - available) == set()

    def test_a_wholly_absent_column_is_still_a_target(self):
        frame = _frame({"A": 1.0, "B": 1.0, "T": 0.0})
        available = DatasetAdapter(CATALOG).available_columns(frame)
        assert ({"T"} - available) == {"T"}
