"""A catalog expression is compiled once, and the zero-divisor rule is one rule.

Compiling with ``sympy.lambdify`` generates and executes Python source, so it is
the expensive part of applying a formula and it does not depend on the table or
frame the formula is applied to. These tests pin that it happens once per
expression, and that all three evaluators agree on what counts as a ratio.

The split that matters: a **constant** divisor is not a ratio. Nulling a nonzero
literal can never fire, so such an expression compiles as a whole and takes no
guard. Only a denominator carrying a column can be zero at runtime.
"""

import ibis
import pandas as pd
import pytest
from sympy.parsing.sympy_parser import parse_expr

from etdtransform.calculated_columns import (
    evaluate_over_frame,
    sympy_to_ibis,
    undefined_where,
)
from etdtransform.catalog.compiled import (
    CompiledExpression,
    as_compiled,
    compile_expression,
    compile_plan,
)

pytestmark = pytest.mark.unit

PLAIN = "a + b"
CONSTANT_DIVISOR = "a/100 + b"
TRUE_RATIO = "a/b"
SUM_OF_RATIOS = "a/c + b/c"


def _frame():
    return pd.DataFrame({
        "a": pd.array([10.0, 20.0, 30.0, pd.NA], dtype="Float64"),
        "b": pd.array([2.0, 0.0, 5.0, 4.0], dtype="Float64"),
        "c": pd.array([4.0, 0.0, 2.0, 1.0], dtype="Float64"),
    })


def _table(frame=None):
    return ibis.memtable(
        (frame if frame is not None else _frame()).astype(float)
    )


class TestWhatCountsAsARatio:
    """``is_ratio`` is the one decision the guard turns on."""

    def test_a_plain_linear_rule_is_not_a_ratio(self):
        compiled = compile_expression(parse_expr(PLAIN))
        assert compiled.is_ratio is False
        assert compiled.symbols == ("a", "b")
        assert compiled.denominator_symbols == ()

    def test_a_constant_divisor_is_not_a_ratio(self):
        """The bottom is a literal, so it can never be zero at runtime."""
        compiled = compile_expression(parse_expr(CONSTANT_DIVISOR))
        assert compiled.is_ratio is False
        assert compiled.denominator_symbols == ()

    def test_a_column_on_the_bottom_is_a_ratio(self):
        compiled = compile_expression(parse_expr(TRUE_RATIO))
        assert compiled.is_ratio is True
        assert compiled.denominator_symbols == ("b",)

    def test_a_sum_of_ratios_folds_onto_one_denominator(self):
        compiled = compile_expression(parse_expr(SUM_OF_RATIOS))
        assert compiled.is_ratio is True
        assert compiled.denominator_symbols == ("c",)

    def test_a_constant_expression_has_no_symbols(self):
        compiled = compile_expression(parse_expr("3"))
        assert compiled.symbols == ()
        assert compiled.is_ratio is False


class TestTheCompileHappensOnce:
    def test_the_same_expression_resolves_to_the_same_object(self):
        first = compile_expression(parse_expr(PLAIN))
        second = compile_expression(parse_expr(PLAIN))
        assert first is second

    def test_an_equal_expression_written_differently_still_hits(self):
        """SymPy normalises, so the cache key is the expression not the text."""
        assert compile_expression(parse_expr("a + b")) is compile_expression(
            parse_expr("b + a")
        )

    def test_different_expressions_do_not_collide(self):
        assert compile_expression(parse_expr(PLAIN)) is not compile_expression(
            parse_expr(TRUE_RATIO)
        )

    def test_as_compiled_passes_a_compiled_object_through(self):
        compiled = compile_expression(parse_expr(PLAIN))
        assert as_compiled(compiled) is compiled

    def test_as_compiled_compiles_a_bare_expression(self):
        result = as_compiled(parse_expr(PLAIN))
        assert isinstance(result, CompiledExpression)

    def test_compile_plan_keeps_the_plan_shape(self):
        plan = [("target", parse_expr(PLAIN)), ("other", parse_expr(TRUE_RATIO))]
        compiled_plan = compile_plan(plan)

        assert [name for name, _ in compiled_plan] == ["target", "other"]
        assert all(
            isinstance(expr, CompiledExpression) for _, expr in compiled_plan
        )

    def test_compile_plan_is_idempotent(self):
        """Compiling a plan twice must not rebuild the callables."""
        plan = [("target", parse_expr(PLAIN))]
        once = compile_plan(plan)
        twice = compile_plan(once)
        assert once[0][1] is twice[0][1]


class TestTheEvaluatorsAgree:
    """All three read the same ``is_ratio``, so none can take a different
    branch for the same expression."""

    @pytest.mark.parametrize(
        "text", [PLAIN, CONSTANT_DIVISOR], ids=["plain", "constant divisor"]
    )
    def test_undefined_where_is_none_when_not_a_ratio(self, text):
        assert undefined_where(parse_expr(text), _table()) is None

    @pytest.mark.parametrize(
        "text", [TRUE_RATIO, SUM_OF_RATIOS], ids=["ratio", "sum of ratios"]
    )
    def test_undefined_where_marks_the_zero_divisor_rows(self, text):
        table = _table()
        mask = undefined_where(parse_expr(text), table)
        result = table.mutate(flagged=mask).execute()

        assert list(result["flagged"]) == [False, True, False, False]

    def test_the_ibis_path_nulls_a_zero_divisor(self):
        table = _table()
        built = table.mutate(out=sympy_to_ibis(parse_expr(TRUE_RATIO), table))
        out = built.execute()["out"]

        assert out[0] == pytest.approx(5.0)
        assert pd.isna(out[1])

    def test_the_pandas_path_nulls_a_zero_divisor(self):
        out = evaluate_over_frame(parse_expr(TRUE_RATIO), _frame())

        assert out[0] == pytest.approx(5.0)
        assert out.isna()[1]

    def test_a_constant_divisor_is_not_nulled(self):
        """A literal divisor must survive, not be guarded into pd.NA."""
        out = evaluate_over_frame(parse_expr("a/100"), _frame())

        assert out[0] == pytest.approx(0.1)
        assert out[1] == pytest.approx(0.2)

    def test_the_two_paths_agree_on_every_shape(self):
        for text in (PLAIN, CONSTANT_DIVISOR, TRUE_RATIO, SUM_OF_RATIOS):
            expr = parse_expr(text)
            table = _table()
            from_ibis = (
                table.mutate(out=sympy_to_ibis(expr, table))
                .execute()["out"]
                .astype("Float64")
            )
            from_pandas = evaluate_over_frame(expr, _frame())

            pd.testing.assert_series_equal(
                from_ibis, from_pandas, check_names=False, check_dtype=False
            )

    def test_a_missing_input_is_unknown_not_zero(self):
        """Row 3 has no `a`, so a sum reading it is unknown there."""
        out = evaluate_over_frame(parse_expr(PLAIN), _frame())

        assert out.isna()[3]

    def test_a_compiled_object_is_accepted_everywhere(self):
        compiled = compile_expression(parse_expr(TRUE_RATIO))
        table = _table()

        assert undefined_where(compiled, table) is not None
        assert evaluate_over_frame(compiled, _frame()).notna().any()
        table.mutate(out=sympy_to_ibis(compiled, table)).execute()

    def test_evaluate_over_frame_names_the_columns_it_cannot_find(self):
        frame = _frame().drop(columns=["b"])
        with pytest.raises(KeyError, match="b"):
            evaluate_over_frame(parse_expr(PLAIN), frame)


class TestTheCompiledFormRefusesTheWrongAccessor:
    """A caller that asks a whole expression for its denominator has taken the
    wrong branch, and a silent answer would hide it."""

    def test_whole_raises_on_a_ratio(self):
        compiled = compile_expression(parse_expr(TRUE_RATIO))
        with pytest.raises(ValueError, match="ratio"):
            compiled.whole(lambda name: _frame()[name])

    def test_numerator_raises_when_not_a_ratio(self):
        compiled = compile_expression(parse_expr(PLAIN))
        with pytest.raises(ValueError, match="not a ratio"):
            compiled.numerator(lambda name: _frame()[name])

    def test_denominator_raises_when_not_a_ratio(self):
        compiled = compile_expression(parse_expr(CONSTANT_DIVISOR))
        with pytest.raises(ValueError, match="not a ratio"):
            compiled.denominator(lambda name: _frame()[name])


class TestTheRealCatalogCompiles:
    """Every shipped rule has to compile, or the stage fails at planning time
    on data rather than here."""

    def test_every_catalog_rule_compiles(self):
        from etdmap.catalog import load_catalog

        catalog = load_catalog()
        kinds = {"whole": 0, "ratio": 0}
        for text in catalog["rhs_text"].dropna().unique():
            compiled = compile_expression(parse_expr(text))
            kinds["ratio" if compiled.is_ratio else "whole"] += 1

        assert kinds["whole"] > 0
        assert kinds["ratio"] > 0
        assert sum(kinds.values()) == len(
            catalog["rhs_text"].dropna().unique()
        )

    def test_a_constant_divisor_rule_takes_no_guard(self):
        """Measured on the shipped catalog: these exist, and they used to take
        the ratio branch in the ibis path only."""
        from etdmap.catalog import load_catalog

        catalog = load_catalog()
        constant_divisor = [
            text
            for text in catalog["rhs_text"].dropna().unique()
            if (lambda d: d != 1 and not d.free_symbols)(
                parse_expr(text).as_numer_denom()[1]
            )
        ]

        assert constant_divisor, "expected constant-divisor rules in the catalog"
        for text in constant_divisor:
            assert compile_expression(parse_expr(text)).is_ratio is False
