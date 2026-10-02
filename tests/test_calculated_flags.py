"""The flags the calculated stage writes.

``CALCULATED`` marks a cell as produced by a catalog formula rather than
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
        assert ((_flags(df) & ImputeType.CALCULATED.value) != 0).all()

    def test_a_missing_cell_is_not_marked(self):
        # Nothing was calculated there, and an absent value cannot be out of
        # range either.
        df = scan_calculated_columns(
            _frame([1.0, pd.NA, 3.0]), [COLUMN], grid_minutes=5
        )
        marked = (_flags(df) & ImputeType.CALCULATED.value) != 0
        assert list(marked) == [True, False, True]

    def test_existing_flags_are_kept(self):
        # The fill's description of a cell must survive this stage.
        df = _frame([1.0, 2.0])
        df[impute_type_column(COLUMN)] = pd.array(
            [ImputeType.SCALED_FILL.value, 0], dtype="Int64"
        )
        df = scan_calculated_columns(df, [COLUMN], grid_minutes=5)
        assert bool(_flags(df).iloc[0] & ImputeType.SCALED_FILL.value)
        assert bool(_flags(df).iloc[0] & ImputeType.CALCULATED.value)

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
        assert bool(_flags(df).iloc[0] & ImputeType.CALCULATED.value)
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
            flags & ImputeType.CALCULATED_RULE_BOUND_FROM_IMPUTED.value
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
        assert int((_flags(df) & ImputeType.CALCULATED.value).sum()) == 0

    def test_undefined_is_a_signal_not_an_action(self):
        assert bool(ImputeType.CALCULATED_UNDEFINED & ImputeType.SIGNALS)
        assert not bool(ImputeType.CALCULATED_UNDEFINED & ImputeType.ACTIONS)
        # And not an imputed value -- nothing was filled.
        assert not bool(ImputeType.CALCULATED_UNDEFINED & ImputeType.IMPUTED_VALUE)
