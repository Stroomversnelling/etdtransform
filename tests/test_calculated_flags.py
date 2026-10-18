"""The flags the calculated stage writes.

``CALCULATED_COLUMN`` marks a cell as produced by a catalog formula rather than
measured -- which cannot be read from a sidecar, because the derivation is
decided per household and the question is about a cell.
``CALCULATED_THRESHOLD_VIOLATED`` marks it as outside the column's declared
bound, a check nothing in the pipeline performed: a derived column's
``validate_`` column is written before this stage runs, so it exists and is
empty.

Fixtures are small synthetic frames; never real data.
"""

import pandas as pd
import pytest

from etdtransform.calculated_flags import (
    impute_type_column,
    grid_scaled_bounds,
    scan_calculated_columns,
    threshold_violation_summary,
)
from etdtransform.flags import ImputeType

#: A column with a declared minimum of 0 and no maximum, which is the shape
#: every derived energy column takes in the thresholds table.
COLUMN = "ElektriciteitsgebruikTotaalBruto"


def _frame(values, households=None):
    n = len(values)
    return pd.DataFrame({
        "HuisIdBSV": pd.array(households or [1] * n, dtype="Int64"),
        "ReadingDate": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
        COLUMN: pd.array(values, dtype="Float64"),
    })


def _flags(df):
    return df[impute_type_column(COLUMN)].fillna(0).astype("Int64")


class TestCalculatedMark:
    def test_a_cell_with_a_value_is_marked_calculated(self):
        df = scan_calculated_columns(
            _frame([1.0, 2.0, 3.0]), [COLUMN], grid_minutes=5
        )
        assert ((_flags(df) & ImputeType.CALCULATED_COLUMN.value) != 0).all()

    def test_a_missing_cell_is_not_marked(self):
        # Nothing was calculated there, and an absent value cannot be out of
        # range either.
        df = scan_calculated_columns(
            _frame([1.0, pd.NA, 3.0]), [COLUMN], grid_minutes=5
        )
        marked = (_flags(df) & ImputeType.CALCULATED_COLUMN.value) != 0
        assert list(marked) == [True, False, True]

    def test_existing_flags_are_kept(self):
        # The fill's description of a cell must survive this stage.
        df = _frame([1.0, 2.0])
        df[impute_type_column(COLUMN)] = pd.array(
            [ImputeType.SCALED_FILL.value, 0], dtype="Int64"
        )
        df = scan_calculated_columns(df, [COLUMN], grid_minutes=5)
        assert bool(_flags(df).iloc[0] & ImputeType.SCALED_FILL.value)
        assert bool(_flags(df).iloc[0] & ImputeType.CALCULATED_COLUMN.value)

    def test_a_column_not_in_the_frame_is_skipped(self):
        df = scan_calculated_columns(
            _frame([1.0]), [COLUMN, "NotPresent"], grid_minutes=5
        )
        assert "NotPresent_impute_type" not in df.columns


class TestThresholdViolation:
    def test_a_value_below_the_minimum_is_flagged(self):
        df = scan_calculated_columns(
            _frame([1.0, -0.5, 0.0]), [COLUMN], grid_minutes=5
        )
        violated = (_flags(df) & ImputeType.CALCULATED_THRESHOLD_VIOLATED.value) != 0
        # Zero is not below a minimum of zero.
        assert list(violated) == [False, True, False]

    def test_a_violating_cell_is_also_marked_calculated(self):
        df = scan_calculated_columns(_frame([-0.5]), [COLUMN], grid_minutes=5)
        assert bool(_flags(df).iloc[0] & ImputeType.CALCULATED_COLUMN.value)
        assert bool(_flags(df).iloc[0] & ImputeType.CALCULATED_THRESHOLD_VIOLATED.value)

    def test_a_missing_cell_is_never_a_violation(self):
        df = scan_calculated_columns(
            _frame([pd.NA, pd.NA]), [COLUMN], grid_minutes=5
        )
        assert int((_flags(df) != 0).sum()) == 0

    def test_summary_reports_the_household_and_the_worst_value(self):
        df = scan_calculated_columns(
            _frame([-0.5, 1.0, -2.0], households=[1, 1, 2]),
            [COLUMN], grid_minutes=5,
        )
        summary = threshold_violation_summary(df, [COLUMN])
        assert set(summary["HuisIdBSV"]) == {1, 2}
        worst = summary.set_index("HuisIdBSV")["worst"]
        assert float(worst.loc[1]) == pytest.approx(-0.5)
        assert float(worst.loc[2]) == pytest.approx(-2.0)

    def test_summary_is_empty_when_nothing_violates(self):
        df = scan_calculated_columns(_frame([1.0, 2.0]), [COLUMN], grid_minutes=5)
        assert threshold_violation_summary(df, [COLUMN]).empty

    def test_summary_reads_the_flags_rather_than_recomputing(self):
        # Clearing the flag must empty the summary: if the summary recomputed
        # the bounds it could disagree with what the data says.
        df = scan_calculated_columns(_frame([-0.5]), [COLUMN], grid_minutes=5)
        df[impute_type_column(COLUMN)] = pd.array([0], dtype="Int64")
        assert threshold_violation_summary(df, [COLUMN]).empty


class TestRuleBoundViolation:
    """The second bound source. The threshold says a value is impossible for
    the variable; the rule bound says it contradicts the rest of its own row.
    They are separate flags because they are separate claims."""

    @staticmethod
    def _package(lo, hi):
        """A package whose only statement is a declared interval for COLUMN,
        so the per-row narrowing starts and ends at that interval and the test
        measures the flag rather than the solver."""
        import pandas as pd_

        from etdtransform.constraints import build_constraint_package

        empty = pd_.DataFrame({
            "lhs": pd_.array([], dtype="string"),
            "rhs_text": pd_.array([], dtype="string"),
            "variable": pd_.array([], dtype="string"),
            "sign": pd_.array([], dtype="Int64"),
            "derivative_text": pd_.array([], dtype="string"),
            "rhs_var_count": pd_.array([], dtype="Int64"),
        })
        return build_constraint_package(
            empty, cadence_minutes=5,
            thresholds={COLUMN: {
                "Min": lo, "Max": hi, "ThresholdType": "algemeen",
                "UpperThresholdSustained": pd_.NA,
                "UpperThresholdExponent": pd_.NA,
            }},
        )

    def test_without_a_package_the_flag_is_not_written(self):
        # A bound nobody computed is not the same as a bound nothing violates,
        # so the absence of a package must leave the flag off rather than
        # reading as "no row contradicts its own row".
        df = scan_calculated_columns(_frame([-5.0, 1.0]), [COLUMN], grid_minutes=5)
        written = _flags(df) & ImputeType.CALCULATED_OUTSIDE_RULE_BOUND.value
        assert int((written != 0).sum()) == 0

    def test_a_value_outside_the_rule_bound_is_flagged(self):
        df = scan_calculated_columns(
            _frame([0.5, 5.0]), [COLUMN], grid_minutes=5,
            package=self._package(0.0, 1.0),
        )
        broke = (_flags(df) & ImputeType.CALCULATED_OUTSIDE_RULE_BOUND.value) != 0
        assert list(broke) == [False, True]

    def test_the_two_violation_flags_are_independent(self):
        # 5.0 is inside the column threshold (min 0, no max) and outside this
        # rule bound, so exactly one of the two must fire.
        df = scan_calculated_columns(
            _frame([5.0]), [COLUMN], grid_minutes=5,
            package=self._package(0.0, 1.0),
        )
        flags = _flags(df).iloc[0]
        assert bool(flags & ImputeType.CALCULATED_OUTSIDE_RULE_BOUND.value)
        assert not bool(flags & ImputeType.CALCULATED_THRESHOLD_VIOLATED.value)

    def test_a_missing_cell_is_never_a_rule_violation(self):
        df = scan_calculated_columns(
            _frame([pd.NA]), [COLUMN], grid_minutes=5,
            package=self._package(0.0, 1.0),
        )
        assert int((_flags(df) != 0).sum()) == 0


class TestHardVersusSoftKnowns:
    """A rule violation standing on measurements alone is a different finding
    from one whose bound rests on the fill's own output. The narrowing already
    computes which, so the distinction costs nothing."""

    def test_a_bound_on_measurements_alone_is_not_marked_soft(self):
        df = scan_calculated_columns(
            _frame([5.0]), [COLUMN], grid_minutes=5,
            package=TestRuleBoundViolation._package(0.0, 1.0),
        )
        flags = _flags(df).iloc[0]
        assert bool(flags & ImputeType.CALCULATED_OUTSIDE_RULE_BOUND.value)
        assert not bool(
            flags & ImputeType.CALCULATED_RULE_BOUND_FROM_SOFT_KNOWNS.value
        ), "nothing in this frame was filled, so no bound can rest on a fill"

    def test_imputed_masks_are_read_from_the_is_imputed_columns(self):
        from etdtransform.calculated_flags import imputed_masks

        df = _frame([1.0, 2.0])
        df["ZonopwekBruto_is_imputed"] = pd.array([True, False], dtype="boolean")
        df["GasgebruikDiff_is_imputed"] = pd.array([False, pd.NA], dtype="boolean")
        masks = imputed_masks(df)
        assert set(masks) == {"ZonopwekBruto", "GasgebruikDiff"}
        assert list(masks["ZonopwekBruto"]) == [True, False]
        # pd.NA is not evidence that the fill wrote the cell.
        assert list(masks["GasgebruikDiff"]) == [False, False]

    def test_the_summary_separates_the_two(self):
        df = scan_calculated_columns(
            _frame([5.0, 0.5]), [COLUMN], grid_minutes=5,
            package=TestRuleBoundViolation._package(0.0, 1.0),
        )
        summary = threshold_violation_summary(df, [COLUMN])
        rule_rows = summary[summary["source"].str.startswith("rule bound")]
        assert len(rule_rows) == 1
        assert "measurements alone" in rule_rows.iloc[0]["source"]


class TestGridCadence:
    """A calculated column is judged at the grid, never at the threshold's
    declared unit and never at a delivered column's own cadence: after
    imputation every variable the derivation reads sits on the grid."""

    def test_a_coarser_grid_widens_a_scaling_bound(self):
        from etdtransform.calculated_flags import grid_scaled_bounds

        # Netuitwisseling is declared +/-2.01 per five minutes, so an hourly
        # grid holds twelve of those.
        _, at_five = grid_scaled_bounds("Netuitwisseling", 5)
        _, at_sixty = grid_scaled_bounds("Netuitwisseling", 60)
        assert float(at_sixty) > float(at_five)

    def test_a_non_scaling_bound_is_unchanged_by_the_grid(self):
        from etdtransform.calculated_flags import grid_scaled_bounds

        # ZelfgebruikPercentage is a percentage, declared "algemeen": 100 is
        # 100 whatever interval the row covers.
        assert (grid_scaled_bounds("ZelfgebruikPercentage", 5)
                == grid_scaled_bounds("ZelfgebruikPercentage", 60))


class TestUndefinedCalculation:
    """A formula that cannot produce a value even with every input present is
    a different finding from a missing input. Both leave pd.NA in the cell, so
    without a flag the two are indistinguishable -- and only one of them means
    the formula does not apply to that row."""

    def test_the_flag_fires_where_the_formula_is_undefined(self):
        condition = pd.Series([False, True, False])
        df = scan_calculated_columns(
            _frame([1.0, pd.NA, pd.NA]), [COLUMN], grid_minutes=5,
        )
        # The pandas scan takes the mask the same way the ibis path does.
        flags = _flags(df)
        df[impute_type_column(COLUMN)] = flags | (
            condition.astype("Int64") * ImputeType.CALCULATED_UNDEFINED.value
        )
        undefined = (_flags(df) & ImputeType.CALCULATED_UNDEFINED.value) != 0
        assert list(undefined) == [False, True, False]

    def test_an_undefined_cell_is_not_marked_calculated(self):
        # Nothing was calculated there: the cell holds no value.
        df = scan_calculated_columns(_frame([pd.NA]), [COLUMN], grid_minutes=5)
        assert int((_flags(df) & ImputeType.CALCULATED_COLUMN.value).sum()) == 0

    def test_undefined_is_a_signal_not_an_action(self):
        assert bool(ImputeType.CALCULATED_UNDEFINED & ImputeType.SIGNALS)
        assert not bool(ImputeType.CALCULATED_UNDEFINED & ImputeType.ACTIONS)
        # And not an imputed value -- nothing was filled.
        assert not bool(ImputeType.CALCULATED_UNDEFINED & ImputeType.IMPUTED_VALUE)


class TestDeriveFromRuleBound:
    """Cells the rules narrow to a single value. A derivation, not a guess:
    the other values in the row determine the column exactly, and no catalog
    formula reached the cell."""

    @staticmethod
    def _package(lo, hi):
        import pandas as pd_

        from etdtransform.constraints import build_constraint_package

        empty = pd_.DataFrame({
            "lhs": pd_.array([], dtype="string"),
            "rhs_text": pd_.array([], dtype="string"),
            "variable": pd_.array([], dtype="string"),
            "sign": pd_.array([], dtype="Int64"),
            "derivative_text": pd_.array([], dtype="string"),
            "rhs_var_count": pd_.array([], dtype="Int64"),
        })
        return build_constraint_package(
            empty, cadence_minutes=5,
            thresholds={COLUMN: {
                "Min": lo, "Max": hi, "ThresholdType": "algemeen",
                "UpperThresholdSustained": pd_.NA,
                "UpperThresholdExponent": pd_.NA,
            }},
        )

    def test_a_point_bound_fills_a_missing_cell(self):
        from etdtransform.calculated_flags import derive_from_rule_bound

        # Min == Max, so the bound admits exactly one value.
        df, filled = derive_from_rule_bound(
            _frame([pd.NA, pd.NA]), [COLUMN], self._package(0.4, 0.4)
        )
        assert filled == {COLUMN: 2}
        assert [float(v) for v in df[COLUMN]] == [pytest.approx(0.4)] * 2
        assert ((_flags(df) & ImputeType.CALCULATED_FROM_RULE_BOUND.value) != 0).all()

    def test_a_wide_bound_fills_nothing(self):
        from etdtransform.calculated_flags import derive_from_rule_bound

        df, filled = derive_from_rule_bound(
            _frame([pd.NA]), [COLUMN], self._package(0.0, 1.0)
        )
        assert filled == {}
        assert pd.isna(df[COLUMN].iloc[0])
        # No flag column is created when nothing is pinned: an empty
        # impute-type column would read as "examined and found nothing".
        assert impute_type_column(COLUMN) not in df.columns

    def test_a_delivered_value_is_never_overwritten(self):
        from etdtransform.calculated_flags import derive_from_rule_bound

        df, filled = derive_from_rule_bound(
            _frame([0.9, pd.NA]), [COLUMN], self._package(0.4, 0.4)
        )
        # Only the missing cell moved, even though the bound pins both.
        assert filled == {COLUMN: 1}
        assert float(df[COLUMN].iloc[0]) == pytest.approx(0.9)
        assert float(df[COLUMN].iloc[1]) == pytest.approx(0.4)
        flags = _flags(df)
        assert not bool(flags.iloc[0] & ImputeType.CALCULATED_FROM_RULE_BOUND.value)
        assert bool(flags.iloc[1] & ImputeType.CALCULATED_FROM_RULE_BOUND.value)

    def test_the_flag_is_an_action_distinct_from_CALCULATED(self):
        # A reader must be able to tell a constraint-pinned value from one
        # produced by evaluating a formula.
        assert bool(ImputeType.CALCULATED_FROM_RULE_BOUND & ImputeType.ACTIONS)
        assert not bool(ImputeType.CALCULATED_FROM_RULE_BOUND & ImputeType.SIGNALS)
        assert not bool(
            ImputeType.CALCULATED_FROM_RULE_BOUND & ImputeType.IMPUTED_VALUE
        )
        assert (ImputeType.CALCULATED_FROM_RULE_BOUND.value
                != ImputeType.CALCULATED_COLUMN.value)

    def test_existing_flags_survive(self):
        from etdtransform.calculated_flags import derive_from_rule_bound

        df = _frame([pd.NA])
        df[impute_type_column(COLUMN)] = pd.array(
            [ImputeType.GAP_NOT_FILLED.value], dtype="Int64"
        )
        df, _ = derive_from_rule_bound(df, [COLUMN], self._package(0.4, 0.4))
        flags = _flags(df)
        assert bool(flags.iloc[0] & ImputeType.GAP_NOT_FILLED.value)
        assert bool(flags.iloc[0] & ImputeType.CALCULATED_FROM_RULE_BOUND.value)


class TestCalculatedValueComposite:
    """One test for "did the calculation produce this value", whichever route
    produced it. The counterpart of IMPUTED_VALUE."""

    def test_it_covers_both_routes(self):
        assert bool(ImputeType.CALCULATED_VALUE & ImputeType.CALCULATED_COLUMN)
        assert bool(
            ImputeType.CALCULATED_VALUE & ImputeType.CALCULATED_FROM_RULE_BOUND
        )

    def test_it_is_actions_only(self):
        # A composite that admitted a signal would make "was this calculated"
        # true for a cell that was merely judged.
        assert (ImputeType.CALCULATED_VALUE & ImputeType.ACTIONS) \
            == ImputeType.CALCULATED_VALUE
        for signal in (
            ImputeType.CALCULATED_THRESHOLD_VIOLATED,
            ImputeType.CALCULATED_OUTSIDE_RULE_BOUND,
            ImputeType.CALCULATED_RULE_BOUND_FROM_SOFT_KNOWNS,
            ImputeType.CALCULATED_UNDEFINED,
        ):
            assert not bool(ImputeType.CALCULATED_VALUE & signal)

    def test_it_does_not_overlap_imputed_value(self):
        # Imputed and calculated are different provenance; a cell must not
        # read as both through the composites.
        assert not bool(ImputeType.CALCULATED_VALUE & ImputeType.IMPUTED_VALUE)

    def test_a_rule_bound_fill_is_caught_by_it(self):
        from etdtransform.calculated_flags import derive_from_rule_bound

        df, _ = derive_from_rule_bound(
            _frame([pd.NA]), [COLUMN],
            TestDeriveFromRuleBound._package(0.4, 0.4),
        )
        assert bool(_flags(df).iloc[0] & ImputeType.CALCULATED_VALUE.value)

    def test_a_formula_derived_cell_is_caught_by_it(self):
        df = scan_calculated_columns(_frame([0.5]), [COLUMN], grid_minutes=5)
        assert bool(_flags(df).iloc[0] & ImputeType.CALCULATED_VALUE.value)

    def test_a_missing_cell_is_not(self):
        df = scan_calculated_columns(_frame([pd.NA]), [COLUMN], grid_minutes=5)
        assert not bool(_flags(df).iloc[0] & ImputeType.CALCULATED_VALUE.value)


class TestFillFromFormula:
    """Gaps in a column that a catalog formula reaches from the other values in
    the same row. The derivation pass cannot do this: a delivered column was
    never a derivation target, so its gaps were never a candidate for one."""

    LEFT = "ElektriciteitNetgebruikDiff"
    RIGHT = "ElektriciteitTerugleveringDiff"
    THIRD = "ZonopwekBruto"
    FOURTH = "Netuitwisseling"

    @staticmethod
    def _catalog(rows):
        return pd.DataFrame([
            {
                "lhs": lhs,
                "rhs_text": text,
                "rhs_vars": inputs,
                "rhs_var_count": len(inputs),
            }
            for lhs, text, inputs in rows
        ])

    @classmethod
    def _sum_catalog(cls):
        return cls._catalog([
            (COLUMN, f"{cls.LEFT} + {cls.RIGHT}", [cls.LEFT, cls.RIGHT]),
        ])

    @classmethod
    def _frame_with_inputs(cls, target, left, right, third=None, fourth=None):
        df = _frame(target)
        df[cls.LEFT] = pd.array(left, dtype="Float64")
        df[cls.RIGHT] = pd.array(right, dtype="Float64")
        if third is not None:
            df[cls.THIRD] = pd.array(third, dtype="Float64")
        if fourth is not None:
            df[cls.FOURTH] = pd.array(fourth, dtype="Float64")
        return df

    def test_a_gap_is_filled_where_both_inputs_are_present(self):
        from etdtransform.calculated_flags import fill_from_formula

        df, filled, plan = fill_from_formula(
            self._frame_with_inputs([pd.NA], [0.1], [0.2]),
            [COLUMN], self._sum_catalog(),
            TestDeriveFromRuleBound._package(0.0, 1.6),
        )
        assert filled == {COLUMN: 1}
        assert float(df[COLUMN].iloc[0]) == pytest.approx(0.3)
        assert bool(_flags(df).iloc[0] & ImputeType.CALCULATED_COLUMN.value)
        assert int(plan["cells_filled"].iloc[0]) == 1

    def test_a_row_missing_an_input_is_left_alone(self):
        from etdtransform.calculated_flags import fill_from_formula

        # The not-null map for this row does not contain RIGHT, so the only
        # formula is not a candidate. No reasoning about derivation order.
        df, filled, plan = fill_from_formula(
            self._frame_with_inputs([pd.NA], [0.1], [pd.NA]),
            [COLUMN], self._sum_catalog(),
            TestDeriveFromRuleBound._package(0.0, 1.6),
        )
        assert filled == {}
        assert pd.isna(df[COLUMN].iloc[0])
        assert plan.empty

    def test_a_delivered_value_is_never_overwritten(self):
        from etdtransform.calculated_flags import fill_from_formula

        df, filled, _ = fill_from_formula(
            self._frame_with_inputs([0.9, pd.NA], [0.1, 0.1], [0.2, 0.2]),
            [COLUMN], self._sum_catalog(),
            TestDeriveFromRuleBound._package(0.0, 1.6),
        )
        assert filled == {COLUMN: 1}
        assert float(df[COLUMN].iloc[0]) == pytest.approx(0.9)
        assert float(df[COLUMN].iloc[1]) == pytest.approx(0.3)
        flags = _flags(df)
        assert not bool(flags.iloc[0] & ImputeType.CALCULATED_COLUMN.value)
        assert bool(flags.iloc[1] & ImputeType.CALCULATED_COLUMN.value)

    def test_a_value_outside_the_rows_rule_bound_is_rejected(self):
        from etdtransform.calculated_flags import fill_from_formula

        # The fill is an invented value, so it has to be plausible. 99 kWh in
        # five minutes is not, and the bound is what says so.
        df, filled, plan = fill_from_formula(
            self._frame_with_inputs([pd.NA], [50.0], [49.0]),
            [COLUMN], self._sum_catalog(),
            TestDeriveFromRuleBound._package(0.0, 1.6),
        )
        assert filled == {}
        assert pd.isna(df[COLUMN].iloc[0])
        assert int(plan["cells_rejected"].iloc[0]) == 1

    def test_the_formula_with_the_fewest_inputs_is_chosen(self):
        from etdtransform.calculated_flags import fill_from_formula

        catalog = self._catalog([
            (COLUMN, f"{self.LEFT} + {self.RIGHT}", [self.LEFT, self.RIGHT]),
            (COLUMN, f"{self.LEFT} + {self.RIGHT} + {self.THIRD}",
             [self.LEFT, self.RIGHT, self.THIRD]),
        ])
        df, filled, plan = fill_from_formula(
            self._frame_with_inputs([pd.NA], [0.1], [0.2], third=[0.4]),
            [COLUMN], catalog, TestDeriveFromRuleBound._package(0.0, 1.6),
        )
        # Both are candidates; only one is evaluated, and it is the short one.
        assert len(plan) == 1
        assert plan["rhs_vars"].iloc[0] == ",".join(sorted([self.LEFT, self.RIGHT]))
        assert float(df[COLUMN].iloc[0]) == pytest.approx(0.3)

    def test_a_formula_on_measurements_beats_one_using_a_filled_cell(self):
        from etdtransform.calculated_flags import fill_from_formula

        # Both are two-input formulas, so length cannot decide, and the one
        # using the filled cell sorts FIRST by text -- it would win on the
        # tie-break. Only the provenance criterion can explain the choice.
        catalog = self._catalog([
            (COLUMN, f"{self.LEFT} + {self.RIGHT}", [self.LEFT, self.RIGHT]),
            (COLUMN, f"{self.FOURTH} + {self.THIRD}", [self.FOURTH, self.THIRD]),
        ])
        df = self._frame_with_inputs(
            [pd.NA], [0.1], [0.2], third=[0.25], fourth=[0.25]
        )
        df[f"{self.LEFT}_is_imputed"] = pd.array([True], dtype="boolean")
        df, _, plan = fill_from_formula(
            df, [COLUMN], catalog, TestDeriveFromRuleBound._package(0.0, 1.6)
        )
        assert int(plan["inputs_from_fill"].iloc[0]) == 0
        assert float(df[COLUMN].iloc[0]) == pytest.approx(0.5)

    def test_rows_with_different_maps_are_handled_separately(self):
        from etdtransform.calculated_flags import fill_from_formula

        catalog = self._catalog([
            (COLUMN, f"{self.LEFT} + {self.RIGHT}", [self.LEFT, self.RIGHT]),
            (COLUMN, f"2*{self.THIRD}", [self.THIRD]),
        ])
        # Row 0 has both inputs of the first formula; row 1 has only THIRD.
        df, filled, plan = fill_from_formula(
            self._frame_with_inputs(
                [pd.NA, pd.NA], [0.1, pd.NA], [0.2, pd.NA], third=[pd.NA, 0.25]
            ),
            [COLUMN], catalog, TestDeriveFromRuleBound._package(0.0, 1.6),
        )
        assert filled == {COLUMN: 2}
        assert float(df[COLUMN].iloc[0]) == pytest.approx(0.3)
        assert float(df[COLUMN].iloc[1]) == pytest.approx(0.5)
        assert len(plan) == 2
        assert set(plan["mask"]) == {
            "|".join(sorted([self.LEFT, self.RIGHT])), self.THIRD,
        }

    def test_a_tie_on_length_is_broken_by_the_expression_text(self):
        from etdtransform.calculated_flags import choose_formula

        catalog = self._catalog([
            (COLUMN, f"{self.RIGHT} * 2", [self.RIGHT]),
            (COLUMN, f"{self.LEFT} * 2", [self.LEFT]),
        ])
        rows = list(catalog.itertuples())
        available = {self.LEFT, self.RIGHT}
        first = choose_formula(rows, available)
        # Same answer whichever order the catalog rows arrive in: a column's
        # provenance must not vary between runs.
        assert choose_formula(list(reversed(rows)), available).rhs_text \
            == first.rhs_text

    def test_a_formula_that_needs_its_own_target_is_not_a_candidate(self):
        from etdtransform.calculated_flags import choose_formula

        catalog = self._catalog([(COLUMN, f"{COLUMN} * 2", [COLUMN])])
        assert choose_formula(
            list(catalog.itertuples()), {COLUMN, self.LEFT}
        ) is None

    def test_a_frame_with_no_gaps_returns_an_empty_plan(self):
        from etdtransform.calculated_flags import fill_from_formula

        df, filled, plan = fill_from_formula(
            self._frame_with_inputs([0.5], [0.1], [0.2]),
            [COLUMN], self._sum_catalog(),
            TestDeriveFromRuleBound._package(0.0, 1.6),
        )
        assert filled == {}
        assert plan.empty
        assert impute_type_column(COLUMN) not in df.columns


class TestFillCalculatedGaps:
    """Both routes as the calculated stage runs them."""

    @staticmethod
    def _catalog():
        return TestFillFromFormula._catalog([(
            COLUMN,
            f"{TestFillFromFormula.LEFT} + {TestFillFromFormula.RIGHT}",
            [TestFillFromFormula.LEFT, TestFillFromFormula.RIGHT],
        )])

    def test_the_two_routes_are_distinguishable_per_cell(self):
        from etdtransform.calculated_flags import fill_calculated_gaps

        # Row 0 has the formula's inputs; row 1 does not, so only the point
        # bound can reach it.
        df = TestFillFromFormula._frame_with_inputs(
            [pd.NA, pd.NA], [0.4, pd.NA], [0.0, pd.NA]
        )
        df, report, plan = fill_calculated_gaps(
            df, [COLUMN], self._catalog(),
            TestDeriveFromRuleBound._package(0.4, 0.4),
        )
        assert report["from_formula"] == {COLUMN: 1}
        assert report["from_rule_bound"] == {COLUMN: 1}
        assert report["cells"] == 2
        flags = _flags(df)
        assert bool(flags.iloc[0] & ImputeType.CALCULATED_COLUMN.value)
        assert not bool(flags.iloc[0] & ImputeType.CALCULATED_FROM_RULE_BOUND.value)
        assert bool(flags.iloc[1] & ImputeType.CALCULATED_FROM_RULE_BOUND.value)
        assert not bool(flags.iloc[1] & ImputeType.CALCULATED_COLUMN.value)

    def test_every_filled_cell_answers_to_the_composite(self):
        from etdtransform.calculated_flags import fill_calculated_gaps

        df = TestFillFromFormula._frame_with_inputs(
            [pd.NA, pd.NA], [0.4, pd.NA], [0.0, pd.NA]
        )
        df, _, _ = fill_calculated_gaps(
            df, [COLUMN], self._catalog(),
            TestDeriveFromRuleBound._package(0.4, 0.4),
        )
        assert df[COLUMN].notna().all()
        assert ((_flags(df) & ImputeType.CALCULATED_VALUE.value) != 0).all()

    def test_it_runs_over_an_ibis_table_and_patches_it(self):
        import ibis

        from etdtransform.calculated_flags import (
            BITS_SUFFIX, FILL_SUFFIX, fill_gaps_in_table,
        )
        from etdtransform.constraints import RuleSet

        ibis.set_backend("duckdb")
        # Two rows: the first has the formula's inputs, the second does not and
        # is reached only by the point bound.
        frame = TestFillFromFormula._frame_with_inputs(
            [pd.NA, pd.NA], [0.4, pd.NA], [0.0, pd.NA]
        )
        rule_sets = {"": RuleSet(
            apparaat_set="",
            households=[1],
            formulas=self._catalog(),
            package=TestDeriveFromRuleBound._package(0.4, 0.4),
        )}
        table, report, plan = fill_gaps_in_table(
            ibis.memtable(frame), [1], rule_sets
        )
        out = table.execute().sort_values("ReadingDate").reset_index(drop=True)

        assert report["cells"] == 2
        # The values reach the table, not just the pandas slice.
        assert [float(v) for v in out[COLUMN]] == [pytest.approx(0.4)] * 2
        flags = out[impute_type_column(COLUMN)].fillna(0).astype("Int64")
        assert ((flags & ImputeType.CALCULATED_VALUE.value) != 0).all()
        # The patch columns are scaffolding and must not reach a schema.
        assert not [
            c for c in out.columns
            if c.endswith((FILL_SUFFIX, BITS_SUFFIX, "_right"))
        ]

    def test_a_cell_the_fill_did_not_touch_keeps_its_value(self):
        import ibis

        from etdtransform.calculated_flags import fill_gaps_in_table
        from etdtransform.constraints import RuleSet

        ibis.set_backend("duckdb")
        # The bound pins 0.4, and the delivered 0.9 contradicts it. Coalesce
        # must still leave the delivered value alone -- the fill writes gaps,
        # it does not correct measurements.
        frame = TestFillFromFormula._frame_with_inputs(
            [0.9, pd.NA], [pd.NA, pd.NA], [pd.NA, pd.NA]
        )
        rule_sets = {"": RuleSet(
            apparaat_set="",
            households=[1],
            formulas=self._catalog(),
            package=TestDeriveFromRuleBound._package(0.4, 0.4),
        )}
        table, report, _ = fill_gaps_in_table(
            ibis.memtable(frame), [1], rule_sets
        )
        out = table.execute().sort_values("ReadingDate").reset_index(drop=True)
        assert report["cells"] == 1
        assert float(out[COLUMN].iloc[0]) == pytest.approx(0.9)
        assert float(out[COLUMN].iloc[1]) == pytest.approx(0.4)

    def test_a_household_outside_the_rule_set_is_not_touched(self):
        import ibis

        from etdtransform.calculated_flags import fill_gaps_in_table
        from etdtransform.constraints import RuleSet

        ibis.set_backend("duckdb")
        frame = TestFillFromFormula._frame_with_inputs(
            [pd.NA], [pd.NA], [pd.NA]
        )
        # The rule set covers household 2; the table holds household 1, which
        # therefore has no formulas and no bounds of its own.
        rule_sets = {"": RuleSet(
            apparaat_set="",
            households=[2],
            formulas=self._catalog(),
            package=TestDeriveFromRuleBound._package(0.4, 0.4),
        )}
        table, report, plan = fill_gaps_in_table(
            ibis.memtable(frame), [1], rule_sets
        )
        assert report["cells"] == 0
        assert plan.empty
        assert pd.isna(table.execute()[COLUMN].iloc[0])

    def test_the_formula_route_runs_first(self):
        from etdtransform.calculated_flags import fill_calculated_gaps

        # The point bound would pin this cell to 0.4 as well, and the formula
        # sums to 0.4, so the value cannot tell which route ran. The flag can,
        # and it has to say the formula: that route runs first and the second
        # only touches cells still pd.NA.
        df = TestFillFromFormula._frame_with_inputs([pd.NA], [0.4], [0.0])
        df, report, _ = fill_calculated_gaps(
            df, [COLUMN], self._catalog(),
            TestDeriveFromRuleBound._package(0.4, 0.4),
        )
        assert report["from_rule_bound"] == {}
        assert bool(_flags(df).iloc[0] & ImputeType.CALCULATED_COLUMN.value)
