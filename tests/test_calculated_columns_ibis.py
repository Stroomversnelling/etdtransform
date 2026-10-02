import ibis
import pandas as pd
import pytest
import sympy as sp

from etdtransform.calculated_columns import sympy_to_ibis, undefined_where
from etdtransform.aggregate import add_calculated_columns_to_hh_data_ibis
from etdtransform.calculated_columns import add_calculated_columns_adaptive

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
# Pandas vs ibis equivalence -- cross-check that both paths produce identical
# numeric results for the same SymPy expressions.
# The pandas path is the reference (already validated by test_calculated_columns_adaptive).
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
# Integration: ibis pipeline vs pandas adaptive -- multi-schema fixture
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


def _make_multi_schema_df():
    """
    Build a DataFrame with 3 households having different available columns.
    B is all-NA for HH 2 (so completeness < threshold -> absent from its schema).
    C is all-NA for HH 1 (absent from its schema).
    """
    rows = []
    for household_id, a_val, b_val, c_val in [
        (1, 1.0, 2.0, None),   # HH 1: A and B
        (2, 3.0, None, 4.0),   # HH 2: A and C
        (3, 5.0, 6.0, 7.0),   # HH 3: A, B, and C
    ]:
        for _ in range(N):
            rows.append({
                "HuisIdBSV": pd.NA if household_id is None else household_id,
                "ProjectIdBSV": 1,
                "A": a_val,
                "B": b_val,
                "C": c_val,
            })
    df = pd.DataFrame(rows)
    df["HuisIdBSV"] = df["HuisIdBSV"].astype("Int64")
    df["ProjectIdBSV"] = df["ProjectIdBSV"].astype("Int64")
    for col in ["A", "B", "C"]:
        df[col] = df[col].astype("Float64")
    return df


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


class TestIbisVsPandasMultiSchema:
    def test_same_derived_values_per_household(self, tmp_path):
        """Ibis and pandas adaptive must produce identical Target values per household."""
        df = _make_multi_schema_df()
        catalog = _multi_schema_catalog()

        # Pandas reference path
        df_pandas = add_calculated_columns_adaptive(
            df.copy(), catalog_df=catalog, target_columns={"Target"}
        )

        # Ibis path via parquet
        src = str(tmp_path / "imputed.parquet")
        out = str(tmp_path / "calculated.parquet")
        df.to_parquet(src, engine="pyarrow")
        add_calculated_columns_to_hh_data_ibis(
            source_path=src,
            output_path=out,
            catalog_df=catalog,
            target_columns=["Target"],
        )
        df_ibis = pd.read_parquet(out, dtype_backend="numpy_nullable")

        for household_id in [1, 2, 3]:
            pandas_vals = (
                df_pandas[df_pandas["HuisIdBSV"] == household_id]["Target"]
                .astype(float).tolist()
            )
            ibis_vals = (
                df_ibis[df_ibis["HuisIdBSV"] == household_id]["Target"]
                .astype(float).tolist()
            )
            assert ibis_vals == pytest.approx(pandas_vals, rel=1e-9), (
                f"HH {household_id}: pandas={pandas_vals} ibis={ibis_vals}"
            )

    def test_three_schema_groups_produced(self, tmp_path, caplog):
        """The ibis path must log exactly 3 schema groups for this fixture."""
        import logging
        df = _make_multi_schema_df()
        catalog = _multi_schema_catalog()
        src = str(tmp_path / "imputed.parquet")
        out = str(tmp_path / "calculated.parquet")
        df.to_parquet(src, engine="pyarrow")
        with caplog.at_level(logging.INFO):
            add_calculated_columns_to_hh_data_ibis(
                source_path=src,
                output_path=out,
                catalog_df=catalog,
                target_columns=["Target"],
            )
        assert "3 unique schema groups" in caplog.text

    def test_different_rules_applied_per_schema(self, tmp_path):
        """
        HH 1 derives Target via A+B; HH 2 via A+C.
        Verify the numeric values match the expected rule, not a cross-household mix.
        """
        df = _make_multi_schema_df()
        catalog = _multi_schema_catalog()
        src = str(tmp_path / "imputed.parquet")
        out = str(tmp_path / "calculated.parquet")
        df.to_parquet(src, engine="pyarrow")
        add_calculated_columns_to_hh_data_ibis(
            source_path=src,
            output_path=out,
            catalog_df=catalog,
            target_columns=["Target"],
        )
        df_out = pd.read_parquet(out, dtype_backend="numpy_nullable")

        # HH 1: Target = A + B = 1 + 2 = 3
        hh1 = df_out[df_out["HuisIdBSV"] == 1]["Target"].astype(float).tolist()
        assert hh1 == pytest.approx([3.0] * N)

        # HH 2: Target = A + C = 3 + 4 = 7  (B was absent, so fallback rule used)
        hh2 = df_out[df_out["HuisIdBSV"] == 2]["Target"].astype(float).tolist()
        assert hh2 == pytest.approx([7.0] * N)

        # HH 3: Target = A + B = 5 + 6 = 11  (preferred rule, C ignored)
        hh3 = df_out[df_out["HuisIdBSV"] == 3]["Target"].astype(float).tolist()
        assert hh3 == pytest.approx([11.0] * N)


# ---------------------------------------------------------------------------
# Ratios: a zero denominator is undefined, not a value.
# ---------------------------------------------------------------------------

def _ratio_table():
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
