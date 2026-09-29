"""The gap statistics `impute_and_normalize` returns alongside the filled data.

One row per (project, household, column), reporting what the fill did and
whether it conserved the volume the cumulative variable accounts for.
`deviation` is the quantity that matters: the filled interval total minus the
cumulative variable's own range over the same rows.
"""

import pandas as pd
import pytest

from etdtransform.vectorized_impute import ImputeType, impute_and_normalize
from conftest import cadence_table

pytestmark = pytest.mark.unit

CUM = "ElektriciteitNetgebruik"
DIFF = "ElektriciteitNetgebruikDiff"


def _frame(cumulative, diffs, averages):
    n = len(cumulative)
    return pd.DataFrame({
        "HuisIdBSV": pd.array([1] * n, dtype="Int64"),
        "ProjectIdBSV": pd.array([1] * n, dtype="Int64"),
        "ReadingDate": pd.date_range("2026-01-01", periods=n, freq="5min"),
        CUM: pd.array(cumulative, dtype="Float64"),
        DIFF: pd.array(diffs, dtype="Float64"),
        f"{DIFF}_avg": pd.array(averages, dtype="Float64"),
    })


def _impute(frame):
    filled, gap_stats, _ = impute_and_normalize(
        frame, [CUM], "ProjectIdBSV", pd.DataFrame(),
        cadence=cadence_table(frame, [CUM]),
    )
    return filled, gap_stats


@pytest.fixture()
def conserved():
    """A gap bounded by two readings, filled proportionally."""
    return _impute(_frame(
        cumulative=[100.0, pd.NA, pd.NA, pd.NA, 104.0, 105.0],
        diffs=[0.0, pd.NA, pd.NA, pd.NA, pd.NA, 1.0],
        averages=[1.0] * 6,
    ))


class TestTheStatsDescribeTheFill:
    def test_one_row_per_household_and_column(self, conserved):
        _, stats = conserved
        assert len(stats) == 1
        row = stats.iloc[0]
        assert row["column"] == DIFF
        assert int(row["HuisIdBSV"]) == 1
        assert int(row["ProjectIdBSV"]) == 1

    def test_diff_total_is_the_filled_column(self, conserved):
        filled, stats = conserved
        assert float(stats.iloc[0]["diff_col_total"]) == pytest.approx(
            float(filled[DIFF].sum())
        )

    def test_cum_range_is_the_cumulative_min_to_max(self, conserved):
        filled, stats = conserved
        expected = float(filled[CUM].max()) - float(filled[CUM].min())
        assert float(stats.iloc[0]["cum_col_min_max_diff"]) == pytest.approx(expected)

    def test_imputed_counts_the_cells_carrying_a_type(self, conserved):
        filled, stats = conserved
        assert int(stats.iloc[0]["imputed"]) == int(
            filled[f"{DIFF}_impute_type"].notna().sum()
        )

    def test_bitwise_methods_is_the_or_of_methods(self, conserved):
        _, stats = conserved
        row = stats.iloc[0]
        expected = 0
        for method in row["methods"]:
            expected |= int(method)
        assert int(row["bitwise_methods"]) == expected

    def test_the_method_recorded_is_the_one_the_fill_used(self, conserved):
        filled, stats = conserved
        assert int(stats.iloc[0]["bitwise_methods"]) & ImputeType.SCALED_FILL.value


class TestDeviationReportsConservation:
    """`deviation` is the reason these statistics exist: it says whether the
    filled intervals still add up to what the cumulative variable recorded."""

    def test_an_anchored_fill_deviates_by_nothing(self, conserved):
        _, stats = conserved
        assert float(stats.iloc[0]["deviation"]) == pytest.approx(0.0)

    def test_a_meter_reset_is_reported_as_a_shortfall(self):
        """The cumulative drops, so the gap fills with zeros and the interval
        total cannot match the cumulative's min-to-max range."""
        _, stats = _impute(_frame(
            cumulative=[100.0, pd.NA, pd.NA, 90.0],
            diffs=[0.0, pd.NA, pd.NA, pd.NA],
            averages=[1.0] * 4,
        ))
        row = stats.iloc[0]
        assert int(row["bitwise_methods"]) & ImputeType.NEGATIVE_GAP_JUMP.value
        assert float(row["diff_col_total"]) == pytest.approx(0.0)
        assert float(row["deviation"]) < 0
