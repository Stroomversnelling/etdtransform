import ibis
import pandas as pd
import pytest
import sympy as sp

from etdtransform.calculated_columns import sympy_to_ibis
from etdtransform.aggregate import add_calculated_columns_to_hh_data_ibis
from etdtransform.calculated_columns import add_calculated_columns_adaptive


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
    col = sympy_to_ibis(expr, tbl)
    return tbl.mutate(_result=col).execute()["_result"].astype(float).tolist()


def test_addition(tbl):
    # A + B with fill_null(0): [1+3, 2+0, 0+5] = [4, 2, 5]
    result = _exec(tbl, A + B)
    assert result == pytest.approx([4.0, 2.0, 5.0])


def test_subtraction(tbl):
    # A - B with fill_null(0): [1-3, 2-0, 0-5] = [-2, 2, -5]
    result = _exec(tbl, A - B)
    assert result == pytest.approx([-2.0, 2.0, -5.0])


def test_multiplication(tbl):
    # A * B with fill_null(0): [1*3, 2*0, 0*5] = [3, 0, 0]
    result = _exec(tbl, A * B)
    assert result == pytest.approx([3.0, 0.0, 0.0])


def test_division(tbl):
    # A / C: [1/2, 2/2, 0/2] = [0.5, 1.0, 0.0]
    result = _exec(tbl, A / C)
    assert result == pytest.approx([0.5, 1.0, 0.0])


def test_grouped_expr_with_parentheses(tbl):
    # (A + B) * C -- verifies SymPy tree structure preserves grouping
    # A: [1,2,0], B: [3,0,5], C: [2,2,2]
    # (1+3)*2=8, (2+0)*2=4, (0+5)*2=10
    result = _exec(tbl, (A + B) * C)
    assert result == pytest.approx([8.0, 4.0, 10.0])


def test_constant_factor(tbl):
    # sp.Rational(1, 2) * A: 0.5 * [1, 2, 0] = [0.5, 1.0, 0.0]
    result = _exec(tbl, sp.Rational(1, 2) * A)
    assert result == pytest.approx([0.5, 1.0, 0.0])


def test_null_input_fills_to_zero(tbl):
    # When A is null, fill_null(0) makes it 0 -- result is 0, not null
    # Row 2 (index 2): A=None -> 0, C=2 -> 0*2=0
    result = _exec(tbl, A * C)
    assert result[2] == pytest.approx(0.0)


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
    """Apply expr via lambdify + fillna(0), matching the adaptive pandas path."""
    symbols = sorted(expr.free_symbols, key=str)
    f = sp.lambdify([str(s) for s in symbols], expr)
    col_data = {str(s): df[str(s)].fillna(0).to_numpy(dtype=float) for s in symbols}
    return list(f(**col_data))


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
    """ibis result must match pandas lambdify result for the same expression."""
    df = tbl.execute()

    ibis_result = _exec(tbl, expr)
    pandas_result = _pandas_result(df, expr)

    assert ibis_result == pytest.approx(pandas_result, rel=1e-9), (
        f"pandas vs ibis mismatch for '{label}': "
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
    for huis_id, a_val, b_val, c_val in [
        (1, 1.0, 2.0, None),   # HH 1: A and B
        (2, 3.0, None, 4.0),   # HH 2: A and C
        (3, 5.0, 6.0, 7.0),   # HH 3: A, B, and C
    ]:
        for _ in range(N):
            rows.append({
                "HuisIdBSV": pd.NA if huis_id is None else huis_id,
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

        for huis_id in [1, 2, 3]:
            pandas_vals = (
                df_pandas[df_pandas["HuisIdBSV"] == huis_id]["Target"]
                .astype(float).tolist()
            )
            ibis_vals = (
                df_ibis[df_ibis["HuisIdBSV"] == huis_id]["Target"]
                .astype(float).tolist()
            )
            assert ibis_vals == pytest.approx(pandas_vals, rel=1e-9), (
                f"HH {huis_id}: pandas={pandas_vals} ibis={ibis_vals}"
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
