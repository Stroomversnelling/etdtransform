import ibis
import pandas as pd
import pytest
import sympy as sp
from etdmap import ids
from etdmap.testing import datasets_frame, household_batch_of

from etdtransform.aggregate import add_calculated_columns_to_hh_data_ibis
from etdtransform.calculated_columns import sympy_to_ibis, undefined_where

pytestmark = pytest.mark.integration


def _make_catalog(*rules):
    """Build a minimal catalog DataFrame from (lhs, rhs_text, rhs_vars) tuples."""
    rows = [
        {"lhs": lhs, "rhs_text": rhs_text, "rhs_vars": rhs_vars, "rhs_var_count": len(rhs_vars)}
        for lhs, rhs_text, rhs_vars in rules
    ]
    return pd.DataFrame(rows)

A, B, C = sp.symbols("A B C")


@pytest.fixture
def tbl():
    """Three rows of A, B and C, stated here because the per-cell null
    placement is what every assertion over this table reads."""
    df = pd.DataFrame(
        {
            "A": pd.array([1.0, 2.0, None], dtype="Float64"),
            "B": pd.array([3.0, None, 5.0], dtype="Float64"),
            "C": pd.array([2.0, 2.0, 2.0], dtype="Float64"),
        }
    )
    return ibis.memtable(df)


def _exec(tbl, expr):
    """The derived column as a list, with None where the result is null.

    None rather than nan, so a propagated null is asserted as a distinct
    outcome and cannot be confused with a computed zero -- which is the whole
    point of the contract these tests pin.
    """
    col = sympy_to_ibis(expr, tbl)
    series = tbl.mutate(_result=col).execute()["_result"]
    return [None if pd.isna(v) else float(v) for v in series]


def test_addition(tbl):
    # A + B, nulls propagating: [1+3, 2+null, null+5] = [4, null, null]
    assert _exec(tbl, A + B) == [4.0, None, None]


def test_subtraction(tbl):
    assert _exec(tbl, A - B) == [-2.0, None, None]


def test_multiplication(tbl):
    # Null times a number is null, NOT zero. Treating the null as zero is what
    # makes a product look like a measured nothing.
    assert _exec(tbl, A * B) == [3.0, None, None]


def test_division(tbl):
    # C is never null, so only A's null propagates.
    assert _exec(tbl, A / C) == [0.5, 1.0, None]


def test_grouped_expr_with_parentheses(tbl):
    # Grouping is preserved, and a null anywhere in the tree reaches the result.
    assert _exec(tbl, (A + B) * C) == [8.0, None, None]


def test_constant_factor(tbl):
    assert _exec(tbl, sp.Rational(1, 2) * A) == [0.5, 1.0, None]


def test_null_input_propagates_and_is_not_zero(tbl):
    # The contract this file exists to pin. A null input yields a null result:
    # a derived value whose input is unknown is unknown. Substituting zero
    # states that the quantity was measured as nothing, which produced
    # negative gross consumption wherever solar production was missing while
    # the export register advanced.
    result = _exec(tbl, A * C)
    assert result[2] is None
    assert result[2] != 0.0


def test_constant_expr_no_free_symbols():
    # Expression with no free symbols returns an ibis literal
    expr = sp.Integer(42)
    df = pd.DataFrame({"X": pd.array([1.0, 2.0], dtype="Float64")})
    t = ibis.memtable(df)
    col = sympy_to_ibis(expr, t)
    result = t.mutate(_result=col).execute()["_result"].astype(float).tolist()
    assert result == pytest.approx([42.0, 42.0])


def test_unsupported_op_raises(tbl):
    # sp.sin is not overloaded on ibis column expressions; must raise clearly
    expr = sp.sin(A)
    with pytest.raises(Exception):
        _exec(tbl, expr)


# ---------------------------------------------------------------------------
# `sympy_to_ibis` against SymPy itself.
#
# Not a comparison of two pipeline implementations: the reference here is
# `sympy.lambdify` over nullable Series, which is the SPECIFICATION of what an
# expression means. What is asserted is that translating the same expression
# into ibis preserves it -- the numbers and, just as much, which cells come
# back unknown.
# ---------------------------------------------------------------------------

def _pandas_result(df, expr):
    """Apply expr via lambdify on the nullable Series, as the pandas path does.

    No fill and no conversion to a numpy float array: both would turn pd.NA
    into a value and hide the thing being compared. Nullable Series arithmetic
    propagates pd.NA on its own, which is why the pipeline uses it.
    """
    symbols = sorted(expr.free_symbols, key=str)
    f = sp.lambdify([str(s) for s in symbols], expr)
    col_data = {str(s): df[str(s)] for s in symbols}
    series = f(**col_data)
    return [None if pd.isna(v) else float(v) for v in series]


@pytest.mark.parametrize(
    "label, expr",
    [
        ("A + B", A + B),
        ("A - B", A - B),
        ("A * B", A * B),
        ("A / C", A / C),
        ("(A + B) * C", (A + B) * C),
        ("A - B + C", A - B + C),
        ("sp.Rational(1,2)*A + B", sp.Rational(1, 2) * A + B),
        ("A * B + C", A * B + C),
        ("(A - C) * B", (A - C) * B),
    ],
)
def test_pandas_ibis_equivalence(tbl, label, expr):
    """ibis result must match pandas lambdify result for the same expression.

    Including WHERE the nulls land: the two engines agreeing on the numbers
    while disagreeing on which cells are unknown would be the same defect in a
    quieter form.
    """
    df = tbl.execute()
    for column in ("A", "B", "C"):
        df[column] = df[column].astype("Float64")

    ibis_result = _exec(tbl, expr)
    pandas_result = _pandas_result(df, expr)

    assert len(ibis_result) == len(pandas_result)
    for position, (got, want) in enumerate(zip(ibis_result, pandas_result)):
        if want is None or got is None:
            assert got is want, (
                f"null placement differs for '{label}' at row {position}: "
                f"pandas={pandas_result}, ibis={ibis_result}"
            )
        else:
            assert got == pytest.approx(want, rel=1e-9), (
                f"pandas vs ibis mismatch for '{label}' at row {position}: "
                f"pandas={pandas_result}, ibis={ibis_result}"
            )


# ---------------------------------------------------------------------------
# Integration: the calculated stage over a multi-schema fixture
#
# Three households with intentionally different available columns:
#   HH 1: A and B present, C absent  -> catalog derives Target = A + B
#   HH 2: A and C present, B absent  -> catalog derives Target = A + C (fallback rule)
#   HH 3: A, B, and C all present    -> catalog derives Target = A + B (preferred rule)
#
# This produces three distinct schema groups, exercising the full grouping
# and planning logic in add_calculated_columns_to_hh_data_ibis.
# ---------------------------------------------------------------------------

N = 10  # rows per household

#: What each household holds for A, B and C. An all-pd.NA column holds no
#: value, so it is absent from that household's schema.
_MULTI_SCHEMA_VALUES = {
    1: {"A": 1.0, "B": 2.0, "C": pd.NA},   # HH 1: A and B
    2: {"A": 3.0, "B": pd.NA, "C": 4.0},   # HH 2: A and C
    3: {"A": 5.0, "B": 6.0, "C": 7.0},     # HH 3: A, B, and C
}


def _make_multi_schema_df():
    """One dataset per household, each over A, B and C.

    No cumulative counterpart is completed: A, B and C are not data model
    variables, so none of them has one.
    """
    return datasets_frame(
        {
            household_batch_of(household_id): {
                column: [value] * N for column, value in values.items()
            }
            for household_id, values in _MULTI_SCHEMA_VALUES.items()
        },
        pairs=False,
    )


def _run_ibis(tmp_path, catalog, targets, df=None):
    """Run the production calculated stage over the multi-schema fixture.

    The source is a single parquet, the layout that takes both ids from the
    file, which is where the fixture carries them -- a shard tree takes them
    from its path instead. The OUTPUT is a hive tree, which is the only shape
    the stage writes.
    """
    from etdmap.storage import read_source_frame

    if df is None:
        df = _make_multi_schema_df()
    src = str(tmp_path / "household_imputed.parquet")
    out = str(tmp_path / "household_calculated")
    df.to_parquet(src, engine="pyarrow")
    add_calculated_columns_to_hh_data_ibis(
        source_path=src,
        output_path=out,
        catalog_df=catalog,
        target_columns=targets,
    )
    return read_source_frame(out)


def _multi_schema_catalog():
    """
    Two derivation paths for 'Target':
      preferred:  Target = A + B  (used when both A and B available)
      fallback:   Target = A + C  (used when A and C available but B absent)
    """
    return _make_catalog(
        ("Target", "A + B", ["A", "B"]),
        ("Target", "A + C", ["A", "C"]),
    )


class TestSchemaGroupingPicksTheRulePerHousehold:
    def test_three_schema_groups_produced(self, tmp_path, caplog):
        """The ibis path must log exactly 3 schema groups for this fixture."""
        import logging
        with caplog.at_level(logging.INFO):
            _run_ibis(tmp_path, _multi_schema_catalog(), ["Target"])
        assert "3 unique schema groups" in caplog.text

    def test_different_rules_applied_per_schema(self, tmp_path):
        """
        HH 1 derives Target via A+B; HH 2 via A+C.
        Verify the numeric values match the expected rule, not a cross-household mix.
        """
        df_out = _run_ibis(tmp_path, _multi_schema_catalog(), ["Target"])

        # HH 1: Target = A + B = 1 + 2 = 3
        hh1 = df_out[df_out[ids.HOUSEHOLD] == 1]["Target"].astype(float).tolist()
        assert hh1 == pytest.approx([3.0] * N)

        # HH 2: Target = A + C = 3 + 4 = 7  (B was absent, so fallback rule used)
        hh2 = df_out[df_out[ids.HOUSEHOLD] == 2]["Target"].astype(float).tolist()
        assert hh2 == pytest.approx([7.0] * N)

        # HH 3: Target = A + B = 5 + 6 = 11  (preferred rule, C ignored)
        hh3 = df_out[df_out[ids.HOUSEHOLD] == 3]["Target"].astype(float).tolist()
        assert hh3 == pytest.approx([11.0] * N)


# ---------------------------------------------------------------------------
# Ratios: a zero denominator is undefined, not a value.
# ---------------------------------------------------------------------------

def _ratio_table():
    """A numerator and a denominator over four rows, stated here because the
    zero and the two nulls in them are what the ratio tests read."""
    return ibis.memtable(pd.DataFrame({
        "t": pd.array([1.0, 1.0, 1.0, None], dtype="Float64"),
        "z": pd.array([2.0, 0.0, None, 5.0], dtype="Float64"),
    }))


RATIO = sp.sympify("(-100*t + 100*z)/z")


def test_zero_denominator_yields_null_not_inf_or_nan():
    # ZelfgebruikPercentage divides by solar production, which is genuinely
    # zero every night. Before the guard, 21,652,383 of its 39,840,815 estate
    # cells held NaN -- a value that passes notnull() and poisons every later
    # sum and mean.
    table = _ratio_table()
    built = table.mutate(r=sympy_to_ibis(RATIO, table))
    counts = built.aggregate(
        null=built.r.isnull().sum(),
        nan=(built.r.notnull() & built.r.isnan()).sum(),
    ).execute().iloc[0]
    assert int(counts["nan"]) == 0
    assert int(counts["null"]) == 3  # z==0, z null, t null


def test_a_defined_ratio_still_computes():
    table = _ratio_table()
    built = table.mutate(r=sympy_to_ibis(RATIO, table)).execute()
    assert float(built["r"].iloc[0]) == pytest.approx(50.0)


def _ratio_frame():
    """The same four rows as :func:`_ratio_table`, as the frame the pandas
    evaluator takes, so the two evaluators are compared on one input."""
    return pd.DataFrame({
        "t": pd.array([1.0, 1.0, 1.0, None], dtype="Float64"),
        "z": pd.array([2.0, 0.0, None, 5.0], dtype="Float64"),
    })


class TestTheTwoEvaluatorsAgreeOnAZeroDenominator:
    """``evaluate_over_frame`` is the pandas counterpart of ``sympy_to_ibis``
    and must answer a zero denominator the same way.

    The two run in different passes over the same catalog expressions, so a
    disagreement means one pass calls a cell undefined and the other hands on
    an infinity. An infinity passes ``notnull()``, fails the row's rule bound,
    and is discarded -- so the cell ends up empty either way, but only the ibis
    pass records ``CALCULATED_UNDEFINED`` for it.
    """

    def test_a_zero_denominator_is_not_a_value(self):
        from etdtransform.calculated_columns import evaluate_over_frame

        computed = evaluate_over_frame(RATIO, _ratio_frame())
        import numpy as np

        assert not np.isinf(
            computed.astype("float64").to_numpy(na_value=0.0)
        ).any(), "an infinity passes notnull() and poisons every later sum"
        # z==0, z null, t null -- the same three the ibis path nulls.
        assert int(computed.isna().sum()) == 3

    def test_a_defined_ratio_still_computes(self):
        from etdtransform.calculated_columns import evaluate_over_frame

        computed = evaluate_over_frame(RATIO, _ratio_frame())
        assert float(computed.iloc[0]) == pytest.approx(50.0)

    def test_it_matches_the_ibis_path_cell_for_cell(self):
        from etdtransform.calculated_columns import evaluate_over_frame

        frame = _ratio_frame()
        table = ibis.memtable(frame)
        from_ibis = table.mutate(r=sympy_to_ibis(RATIO, table)).execute()["r"]
        from_pandas = evaluate_over_frame(RATIO, frame)
        assert list(from_pandas.isna()) == list(from_ibis.isna())
        for a, b in zip(from_pandas.dropna(), from_ibis.dropna()):
            assert float(a) == pytest.approx(float(b))


def test_undefined_where_marks_only_the_zero_denominator():
    table = _ratio_table()
    built = table.mutate(u=undefined_where(RATIO, table)).execute()
    # Row 1 is z == 0. Rows 2 and 3 are missing inputs, which is a different
    # finding: the data is absent rather than the formula inapplicable.
    assert list(built["u"]) == [False, True, False, False]


def test_undefined_where_is_none_for_a_non_ratio():
    # So a caller can skip the work instead of evaluating a constant false
    # over every row.
    assert undefined_where(A - B + C, _ratio_table()) is None


def test_a_constant_denominator_is_never_undefined():
    assert undefined_where(sp.sympify("t/2"), _ratio_table()) is None


class TestNothingSubstitutesZeroForAnAbsentColumn:
    """The stage does not fill a column with zero before planning.

    A removed shim took a named list of columns and zero-filled them ahead of
    the availability scan. Two consequences, both of which these tests would
    catch if it came back. The zero-filled column then scored fully available,
    so it could never be chosen for derivation -- the scan reads
    ``notnull().mean()`` and a filled column reads 1.0. And the list was
    fleet-wide while the fact it encoded is per household, so a household that
    merely had a data gap in that column got a fabricated zero.

    Zero for a genuinely absent device is substituted inside the expression
    per household by ``specialize_catalog_for_absent``, from that household's
    own declaration. An input that is simply unknown stays unknown.
    """

    _run = staticmethod(_run_ibis)

    def test_an_all_null_input_leaves_the_target_empty(self, tmp_path):
        # One rule only, so no fallback can mask the behaviour: household 2's
        # B is all-NA, and A + B is the only way to Target. If anything
        # substituted zero for B, household 2's Target would read its A (3.0)
        # instead of staying empty -- a measured quantity of nothing, invented.
        result = self._run(
            tmp_path, _make_catalog(("Target", "A + B", ["A", "B"])), ["Target"]
        )
        household_2 = result[result[ids.HOUSEHOLD] == 2]
        assert household_2["Target"].isna().all()
        assert not (household_2["Target"] == 3.0).any()

    def test_the_households_that_can_be_derived_still_are(self, tmp_path):
        # The companion direction: refusing to invent a zero must not cost the
        # households whose inputs are present. 1 + 2 and 5 + 6.
        result = self._run(
            tmp_path, _make_catalog(("Target", "A + B", ["A", "B"])), ["Target"]
        )
        for household_id, expected in ((1, 3.0), (3, 11.0)):
            values = result[result[ids.HOUSEHOLD] == household_id]["Target"]
            assert values.notna().all()
            assert float(values.iloc[0]) == pytest.approx(expected)

    def test_the_absent_input_does_not_read_as_available(self, tmp_path):
        # Why the shim could never be chosen for derivation: with a fallback
        # present, household 2 must take A + C. A zero-filled B would have
        # scored 1.0 on the availability scan and won the preferred rule,
        # yielding 3.0 from a column that holds nothing.
        result = self._run(tmp_path, _multi_schema_catalog(), ["Target"])
        household_2 = result[result[ids.HOUSEHOLD] == 2]
        assert household_2["Target"].notna().all()
        assert float(household_2["Target"].iloc[0]) == pytest.approx(7.0)
