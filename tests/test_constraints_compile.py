"""`compile_constraints` turns catalog rows into the constraint table.

The catalog already holds one row per (target, derivation) pair, so nothing is
solved here. What is added is the sign of each variable's partial derivative,
which is what lets the processor pick the endpoint that widens a bound.
"""

import pandas as pd
import pytest

from etdtransform.constraints import CONSTRAINT_COLUMNS, compile_constraints


def _catalog(*rows):
    return pd.DataFrame(
        [
            {
                "lhs": lhs,
                "rhs_text": rhs,
                "rhs_vars": sorted(variables),
                "rhs_var_count": len(variables),
                "physical_models": ["Universeel"],
            }
            for lhs, rhs, variables in rows
        ]
    )


def _interval(**pairs):
    return {
        name: {"Min": lo, "Max": hi, "ThresholdType": "5-minute"}
        for name, (lo, hi) in pairs.items()
    }


SUM = ("Total", "PartA + PartB", ["PartA", "PartB"])
DIFFERENCE = ("Net", "Gross - Export", ["Gross", "Export"])


class TestOnlyBoundedTargetsSurvive:
    def test_a_target_with_a_declared_bound_is_kept(self):
        out = compile_constraints(
            _catalog(SUM), thresholds=_interval(Total=(0.0, 1.0))
        )
        assert set(out["lhs"]) == {"Total"}

    def test_a_target_with_no_bound_at_all_is_dropped(self):
        out = compile_constraints(_catalog(SUM), thresholds=_interval())
        assert out.empty

    def test_one_declared_side_is_enough(self):
        out = compile_constraints(
            _catalog(SUM), thresholds=_interval(Total=(0.0, pd.NA))
        )
        assert set(out["lhs"]) == {"Total"}

    def test_a_bound_of_na_on_both_sides_is_no_bound(self):
        out = compile_constraints(
            _catalog(SUM), thresholds=_interval(Total=(pd.NA, pd.NA))
        )
        assert out.empty


class TestOneRowPerVariable:
    def test_each_variable_of_each_derivation_gets_a_row(self):
        out = compile_constraints(
            _catalog(SUM, DIFFERENCE),
            thresholds=_interval(Total=(0.0, 1.0), Net=(0.0, 1.0)),
        )
        assert len(out) == 4
        assert set(zip(out["lhs"], out["variable"])) == {
            ("Total", "PartA"),
            ("Total", "PartB"),
            ("Net", "Gross"),
            ("Net", "Export"),
        }

    def test_the_solved_expression_is_carried_through_unchanged(self):
        out = compile_constraints(
            _catalog(SUM), thresholds=_interval(Total=(0.0, 1.0))
        )
        assert set(out["rhs_text"]) == {"PartA + PartB"}

    def test_two_derivations_of_one_target_both_survive(self):
        out = compile_constraints(
            _catalog(SUM, ("Total", "PartC * 2", ["PartC"])),
            thresholds=_interval(Total=(0.0, 1.0)),
        )
        assert set(out["rhs_text"]) == {"PartA + PartB", "PartC * 2"}


class TestTheSign:
    def test_a_variable_added_has_a_positive_sign(self):
        out = compile_constraints(
            _catalog(SUM), thresholds=_interval(Total=(0.0, 1.0))
        )
        assert set(out["sign"]) == {1}

    def test_a_variable_subtracted_has_a_negative_sign(self):
        out = compile_constraints(
            _catalog(DIFFERENCE), thresholds=_interval(Net=(0.0, 1.0))
        )
        signs = dict(zip(out["variable"], out["sign"]))
        assert signs == {"Gross": 1, "Export": -1}

    def test_a_variable_that_cancels_has_a_zero_sign(self):
        out = compile_constraints(
            _catalog(("T", "A + B - B", ["A", "B"])),
            thresholds=_interval(T=(0.0, 1.0)),
        )
        signs = dict(zip(out["variable"], out["sign"]))
        assert signs["B"] == 0
        assert signs["A"] == 1

    def test_a_scaled_variable_keeps_the_sign_of_its_coefficient(self):
        out = compile_constraints(
            _catalog(("T", "3*A - 0.5*B", ["A", "B"])),
            thresholds=_interval(T=(0.0, 1.0)),
        )
        assert dict(zip(out["variable"], out["sign"])) == {"A": 1, "B": -1}


class TestNonlinearRowsAreDropped:
    def test_a_derivation_dividing_by_a_variable_is_dropped(self):
        out = compile_constraints(
            _catalog(SUM, ("Total", "PartA / PartB", ["PartA", "PartB"])),
            thresholds=_interval(Total=(0.0, 1.0)),
        )
        assert set(out["rhs_text"]) == {"PartA + PartB"}

    def test_dropping_every_derivation_leaves_an_empty_table(self):
        out = compile_constraints(
            _catalog(("T", "A / B", ["A", "B"])),
            thresholds=_interval(T=(0.0, 1.0)),
        )
        assert out.empty
        assert list(out.columns) == list(CONSTRAINT_COLUMNS)


class TestAProductOfTwoUnknowns:
    """`classify_linear_nonlinear` keeps these rows, so the declared bounds of the
    other factor are what decide whether a sign exists."""

    PRODUCT = ("T", "A * B", ["A", "B"])

    def test_a_non_negative_other_factor_gives_a_positive_sign(self):
        out = compile_constraints(
            _catalog(self.PRODUCT),
            thresholds=_interval(T=(0.0, 1.0), A=(0.0, 5.0), B=(0.0, 5.0)),
        )
        assert dict(zip(out["variable"], out["sign"])) == {"A": 1, "B": 1}

    def test_a_non_positive_other_factor_flips_the_sign(self):
        out = compile_constraints(
            _catalog(self.PRODUCT),
            thresholds=_interval(T=(0.0, 1.0), A=(0.0, 5.0), B=(-5.0, 0.0)),
        )
        signs = dict(zip(out["variable"], out["sign"]))
        assert signs["A"] == -1
        assert signs["B"] == 1

    def test_an_other_factor_straddling_zero_leaves_no_sign(self):
        out = compile_constraints(
            _catalog(self.PRODUCT),
            thresholds=_interval(T=(0.0, 1.0), A=(0.0, 5.0), B=(-5.0, 5.0)),
        )
        signs = dict(zip(out["variable"], out["sign"]))
        assert pd.isna(signs["A"]), "A's coefficient is B, which straddles zero"
        assert signs["B"] == 1, "B's coefficient is A, which does not"

    def test_an_undeclared_other_factor_leaves_no_sign(self):
        out = compile_constraints(
            _catalog(self.PRODUCT), thresholds=_interval(T=(0.0, 1.0))
        )
        assert out["sign"].isna().tolist() == [True, True]

    def test_the_coefficient_it_depends_on_is_recorded(self):
        out = compile_constraints(
            _catalog(self.PRODUCT), thresholds=_interval(T=(0.0, 1.0))
        )
        assert dict(zip(out["variable"], out["derivative_text"])) == {"A": "B", "B": "A"}

    def test_a_variable_added_alongside_a_product_keeps_its_sign(self):
        out = compile_constraints(
            _catalog(("T", "C + A*B", ["A", "B", "C"])),
            thresholds=_interval(T=(0.0, 1.0)),
        )
        signs = dict(zip(out["variable"], out["sign"]))
        assert signs["C"] == 1
        assert pd.isna(signs["A"]) and pd.isna(signs["B"])

    def test_a_constant_coefficient_is_recorded_as_its_value(self):
        out = compile_constraints(
            _catalog(("T", "3*A - 0.5*B", ["A", "B"])),
            thresholds=_interval(T=(0.0, 1.0)),
        )
        assert dict(zip(out["variable"], out["derivative_text"])) == {
            "A": "3",
            "B": "-0.500000000000000",
        }


class TestASignedVariableIsNoObstacle:
    """The sign is the derivative's, not the variable's, so a value free to go
    negative still gets an exact sign wherever its coefficient is constant."""

    NET = ("Netuitwisseling", "Netgebruik - Teruglevering", ["Netgebruik", "Teruglevering"])

    def test_a_target_free_to_go_negative_still_signs_its_terms(self):
        out = compile_constraints(
            _catalog(self.NET),
            thresholds=_interval(
                Netuitwisseling=(-2.01, 2.01),
                Netgebruik=(0.0, 2.01),
                Teruglevering=(0.0, 2.01),
            ),
        )
        assert dict(zip(out["variable"], out["sign"])) == {
            "Netgebruik": 1,
            "Teruglevering": -1,
        }

    def test_a_term_free_to_go_negative_still_gets_a_sign(self):
        out = compile_constraints(
            _catalog(("T", "Swing + Fixed", ["Swing", "Fixed"])),
            thresholds=_interval(
                T=(0.0, 5.0), Swing=(-2.01, 2.01), Fixed=(0.0, 5.0)
            ),
        )
        assert dict(zip(out["variable"], out["sign"])) == {"Swing": 1, "Fixed": 1}


class TestAMomentaanColumnStopsTheCompile:
    def test_a_momentaan_variable_raises(self):
        thresholds = _interval(Total=(0.0, 1.0))
        thresholds["PartA"] = {
            "Min": 0.0, "Max": 1.0, "ThresholdType": "momentaan",
        }
        with pytest.raises(ValueError) as raised:
            compile_constraints(_catalog(SUM), thresholds=thresholds)
        message = str(raised.value)
        assert "PartA" in message
        assert "momentaan" in message

    def test_a_momentaan_target_raises(self):
        thresholds = {
            "Total": {"Min": 0.0, "Max": 1.0, "ThresholdType": "momentaan"},
            "PartA": {"Min": 0.0, "Max": 1.0, "ThresholdType": "5-minute"},
            "PartB": {"Min": 0.0, "Max": 1.0, "ThresholdType": "5-minute"},
        }
        with pytest.raises(ValueError) as raised:
            compile_constraints(_catalog(SUM), thresholds=thresholds)
        assert "Total" in str(raised.value)

    def test_a_momentaan_column_in_a_dropped_row_does_not_raise(self):
        thresholds = _interval(T=(0.0, 1.0))
        thresholds["B"] = {"Min": 0.0, "Max": 1.0, "ThresholdType": "momentaan"}
        out = compile_constraints(
            _catalog(("T", "A / B", ["A", "B"])), thresholds=thresholds
        )
        assert out.empty


class TestTheContract:
    def test_the_column_set_is_fixed(self):
        out = compile_constraints(
            _catalog(SUM), thresholds=_interval(Total=(0.0, 1.0))
        )
        assert list(out.columns) == list(CONSTRAINT_COLUMNS)

    def test_the_dtypes_are_nullable(self):
        out = compile_constraints(
            _catalog(SUM), thresholds=_interval(Total=(0.0, 1.0))
        )
        assert out["sign"].dtype == "Int64"
        assert out["rhs_var_count"].dtype == "Int64"
        for column in ("lhs", "rhs_text", "variable", "derivative_text"):
            assert out[column].dtype == "string"

    def test_it_survives_a_parquet_round_trip(self, tmp_path):
        out = compile_constraints(
            _catalog(SUM, ("T", "A * B", ["A", "B"])),
            thresholds=_interval(Total=(0.0, 1.0), T=(0.0, 1.0)),
        )
        path = tmp_path / "constraints.parquet"
        out.to_parquet(path, index=False)
        back = pd.read_parquet(path, dtype_backend="numpy_nullable")
        pd.testing.assert_frame_equal(back, out)

    def test_an_empty_catalog_gives_an_empty_table_with_the_columns(self):
        out = compile_constraints(_catalog(), thresholds=_interval())
        assert out.empty
        assert list(out.columns) == list(CONSTRAINT_COLUMNS)

    def test_the_catalog_is_not_modified(self):
        catalog = _catalog(SUM)
        before = catalog.copy(deep=True)
        compile_constraints(catalog, thresholds=_interval(Total=(0.0, 1.0)))
        pd.testing.assert_frame_equal(catalog, before)

    def test_compiling_twice_gives_the_same_table(self):
        args = (_catalog(SUM, DIFFERENCE),)
        kwargs = {"thresholds": _interval(Total=(0.0, 1.0), Net=(0.0, 1.0))}
        first = compile_constraints(*args, **kwargs)
        second = compile_constraints(*args, **kwargs)
        pd.testing.assert_frame_equal(first, second)


class TestAgainstTheCommittedCatalog:
    """The real artifact, so a change to the rules or the thresholds shows up."""

    @pytest.fixture(scope="class")
    def compiled(self):
        from pathlib import Path

        import etdmap

        path = Path(etdmap.__file__).parent / "data" / "catalog.parquet"
        catalog = pd.read_parquet(path, dtype_backend="numpy_nullable")
        return catalog, compile_constraints(catalog)

    def test_it_produces_constraints(self, compiled):
        _, out = compiled
        assert not out.empty

    def test_every_target_carries_a_declared_bound(self, compiled):
        from etdmap.data_model import thresholds_dict

        _, out = compiled
        for target in out["lhs"].unique():
            declared = thresholds_dict[target]
            assert pd.notna(declared["Min"]) or pd.notna(declared["Max"])

    def test_every_sign_is_minus_one_zero_one_or_absent(self, compiled):
        _, out = compiled
        assert set(out["sign"].dropna().unique()) <= {-1, 0, 1}

    def test_only_four_rows_end_up_without_a_sign(self, compiled):
        """A change here means the rules or the thresholds changed, not that the
        compiler broke."""
        _, out = compiled
        assert int(out["sign"].isna().sum()) == 4

    def test_the_sign_that_is_missing_is_missing_for_a_physical_reason(self, compiled):
        """`Netuitwisseling` is declared free to go either way, so the coefficient
        of the self-use percentage straddles zero and its direction is genuinely
        undecided. Nothing about the compiler can recover it."""
        _, out = compiled
        unsigned = out[out["sign"].isna()]
        assert set(unsigned["lhs"]) == {"Zelfgebruik"}
        assert set(unsigned["variable"]) == {"ZelfgebruikPercentage"}
        assert unsigned["derivative_text"].str.contains("Netuitwisseling").all()

    def test_a_signed_variable_still_gets_a_sign_where_its_term_is_plain(self, compiled):
        _, out = compiled
        net = out[out["variable"] == "Netuitwisseling"]
        assert not net.empty
        assert net["sign"].notna().all()

    def test_every_target_keeps_a_derivation_signed_all_the_way_through(self, compiled):
        _, out = compiled
        fully_signed = out.groupby(["lhs", "rhs_text"])["sign"].apply(
            lambda signs: signs.notna().all()
        )
        assert fully_signed.groupby("lhs").any().all()

    def test_it_keeps_fewer_rows_than_the_catalog_holds(self, compiled):
        catalog, out = compiled
        assert out["rhs_text"].nunique() < len(catalog)

    def test_the_building_bound_chain_is_present(self, compiled):
        _, out = compiled
        wtw = out[out["lhs"] == "ElektriciteitsgebruikWTWDiff"]
        assert not wtw.empty
