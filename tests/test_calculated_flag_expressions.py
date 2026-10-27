"""The flags the calculated stage writes, through the expressions it runs.

``calculated_flag_expressions`` is the ibis path. The calculated stage never
materialises the frame, so the marks are built as expressions and handed to
``mutate`` (``calculated_stage.py:474-479``). These assertions go through that
function, so a change to what production runs cannot pass here.

``CALCULATED_COLUMN`` marks a cell as produced by a catalog formula rather than
measured -- which cannot be read from a sidecar, because the derivation is
decided per household and the question is about a cell.
``CALCULATED_THRESHOLD_VIOLATED`` marks it as outside the column's declared
bound, a check nothing else in the pipeline performs: a derived column's
``validate_`` column is written before this stage runs, so it exists and is
empty. ``CALCULATED_UNDEFINED`` separates "the formula does not apply to this
row" from "an input was missing", which both leave ``pd.NA`` in the cell.

``CALCULATED_OUTSIDE_RULE_BOUND`` is deliberately absent: that bound is a
per-row intersection over substituted values, which this path does not compute.

Fixtures are small synthetic frames; never real data.
"""

import ibis
import pandas as pd
import pytest

from etdtransform.calculated_flags import (
    calculated_flag_expressions,
    grid_scaled_bounds,
    impute_type_column,
)
from etdtransform.flags import ImputeType

pytestmark = pytest.mark.unit

#: A real data-model column, not an invented one: the bound comes from the
#: thresholds table, so a made-up name would simply carry no bound and every
#: threshold assertion below would pass without testing anything.
COLUMN = "ElektriciteitsgebruikTotaalBruto"
GRID_MINUTES = 5
FLAG = impute_type_column(COLUMN)


def _frame(values, households=None, flags=None):
    n = len(values)
    df = pd.DataFrame({
        "HuisIdBSV": pd.array(households or [1] * n, dtype="Int64"),
        "HuisBatchIdBSV": pd.array(households or [1] * n, dtype="Int64"),
        "ReadingDate": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
        COLUMN: pd.array(values, dtype="Float64"),
    })
    if flags is not None:
        df[FLAG] = pd.array(flags, dtype="Int64")
    return df


def _flags(df, derived=None, undefined=None):
    """Run the production expressions over *df* and return the flag column.

    *undefined* is a callable taking the table, so a test can build a predicate
    against the same table the expressions are built from -- which is how
    ``calculated_stage`` supplies it from ``undefined_where``.
    """
    table = ibis.memtable(df)
    expressions = calculated_flag_expressions(
        table,
        derived if derived is not None else [COLUMN],
        GRID_MINUTES,
        undefined=undefined(table) if undefined else None,
    )
    out = table.mutate(**expressions).execute()
    return out[FLAG].fillna(0).astype("Int64")


def test_the_fixture_column_declares_the_bound_these_tests_rely_on():
    """Asserted rather than assumed, because every threshold case rests on it:
    a minimum of zero that -0.5 breaks and 0.0 does not, and no upper bound an
    in-range fixture value reaches."""
    lower, upper = grid_scaled_bounds(COLUMN, GRID_MINUTES)
    assert pd.notna(lower), f"{COLUMN} declares no minimum"
    assert float(lower) == 0.0
    if pd.notna(upper):
        assert float(upper) > 3.0


class TestTheCalculatedMark:
    def test_a_cell_with_a_value_is_marked_calculated(self):
        flags = _flags(_frame([1.0, 2.0, 3.0]))
        assert ((flags & ImputeType.CALCULATED_COLUMN.value) != 0).all()

    def test_a_missing_cell_is_not_marked(self):
        # Nothing was calculated there, and an absent value cannot be out of
        # range either.
        flags = _flags(_frame([1.0, pd.NA, 3.0]))
        marked = (flags & ImputeType.CALCULATED_COLUMN.value) != 0
        assert list(marked) == [True, False, True]

    def test_existing_flags_are_kept(self):
        # The fill's description of a cell must survive this stage: the
        # expression ORs onto the column rather than replacing it.
        flags = _flags(_frame([1.0, 2.0], flags=[ImputeType.SCALED_FILL.value, 0]))
        assert bool(flags.iloc[0] & ImputeType.SCALED_FILL.value)
        assert bool(flags.iloc[0] & ImputeType.CALCULATED_COLUMN.value)

    def test_a_null_in_an_existing_flag_column_does_not_swallow_the_mark(self):
        # fill_null(0) before the OR: a null flag means nothing recorded yet,
        # not "discard what this stage found".
        flags = _flags(_frame([1.0, 2.0], flags=[pd.NA, 0]))
        assert bool(flags.iloc[0] & ImputeType.CALCULATED_COLUMN.value)

    def test_a_column_not_in_the_table_is_skipped(self):
        table = ibis.memtable(_frame([1.0]))
        expressions = calculated_flag_expressions(
            table, [COLUMN, "NotPresent"], GRID_MINUTES,
        )
        assert FLAG in expressions
        assert impute_type_column("NotPresent") not in expressions


class TestThresholdViolation:
    def test_a_value_below_the_minimum_is_flagged(self):
        flags = _flags(_frame([1.0, -0.5, 0.0]))
        violated = (flags & ImputeType.CALCULATED_THRESHOLD_VIOLATED.value) != 0
        # Zero is not below a minimum of zero.
        assert list(violated) == [False, True, False]

    def test_a_violating_cell_is_also_marked_calculated(self):
        # The two are independent findings about one cell: it was derived, and
        # what was derived is out of range.
        flags = _flags(_frame([-0.5]))
        assert bool(flags.iloc[0] & ImputeType.CALCULATED_COLUMN.value)
        assert bool(flags.iloc[0] & ImputeType.CALCULATED_THRESHOLD_VIOLATED.value)

    def test_a_missing_cell_is_never_a_violation(self):
        # An absent value cannot be out of range.
        flags = _flags(_frame([pd.NA, pd.NA]))
        violated = flags & ImputeType.CALCULATED_THRESHOLD_VIOLATED.value
        assert int(violated.sum()) == 0

    def test_a_missing_cell_has_nothing_done_to_it(self):
        # Stated as the ACTION class rather than as "no bits at all": a later
        # SIGNAL about the cell is a different kind of claim and must not read
        # as a regression here.
        flags = _flags(_frame([pd.NA, pd.NA]))
        assert int((flags & ImputeType.ACTIONS.value).sum()) == 0


class TestUndefinedCalculation:
    """A formula that cannot produce a value even with every input present is a
    different finding from a missing input. Both leave pd.NA in the cell, so
    without a flag the two are indistinguishable -- and only one of them means
    the formula does not apply to that row."""

    def test_the_condition_marks_the_rows_it_selects(self):
        flags = _flags(
            _frame([1.0, pd.NA, pd.NA], households=[1, 2, 3]),
            undefined=lambda t: {COLUMN: t["HuisIdBSV"] == 2},
        )
        undefined = (flags & ImputeType.CALCULATED_UNDEFINED.value) != 0
        assert list(undefined) == [False, True, False]

    def test_no_condition_leaves_the_mark_off(self):
        flags = _flags(_frame([1.0, pd.NA]))
        assert int((flags & ImputeType.CALCULATED_UNDEFINED.value).sum()) == 0

    def test_an_undefined_cell_is_not_marked_calculated(self):
        # Nothing was calculated there: the cell holds no value.
        flags = _flags(
            _frame([pd.NA]), undefined=lambda t: {COLUMN: t["HuisIdBSV"] == 1},
        )
        assert int((flags & ImputeType.CALCULATED_COLUMN.value).sum()) == 0
        assert bool(flags.iloc[0] & ImputeType.CALCULATED_UNDEFINED.value)

    def test_a_condition_on_another_column_is_ignored(self):
        # The mask is keyed per column; one column's undefined rows say nothing
        # about another's.
        flags = _flags(
            _frame([pd.NA]), undefined=lambda t: {"SomeOtherColumn": t["HuisIdBSV"] == 1},
        )
        assert int((flags & ImputeType.CALCULATED_UNDEFINED.value).sum()) == 0


class TestTheReturnContract:
    def test_it_returns_one_expression_per_derived_column_present(self):
        table = ibis.memtable(_frame([1.0]))
        expressions = calculated_flag_expressions(table, [COLUMN], GRID_MINUTES)
        assert list(expressions) == [FLAG]

    def test_the_rule_bound_flag_is_not_written_here(self):
        # The rule bound is a per-row intersection over substituted values,
        # which this path does not compute. A value far outside any rule bound
        # still gets only the threshold finding.
        flags = _flags(_frame([-0.5]))
        assert not bool(
            flags.iloc[0] & ImputeType.CALCULATED_OUTSIDE_RULE_BOUND.value
        )
