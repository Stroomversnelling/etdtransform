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
        frame, [CUM], "ProjectIdBSV",
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

    def test_imputed_counts_the_cells_a_value_was_written_into(self, conserved):
        """Four interval cells arrived empty and all four were filled, so all
        four count. What must not be counted is a cell left empty, which still
        carries NOT_FILLED -- counting types is what made `imputed` equal
        `missing` on every row of both household sidecars."""
        filled, stats = conserved
        assert int(stats.iloc[0]["imputed"]) == 4
        assert filled[DIFF].notna().all(), "the fixture fills its whole gap"

    def test_imputed_and_imputed_na_partition_missing(self, conserved):
        _, stats = conserved
        row = stats.iloc[0]
        assert int(row["imputed"]) + int(row["imputed_na"]) == int(row["missing"])

    def test_a_fully_filled_gap_leaves_nothing_empty(self, conserved):
        _, stats = conserved
        assert int(stats.iloc[0]["imputed_na"]) == 0

    def test_percentage_imputed_is_not_a_hundred_by_construction(self, conserved):
        """It is derived from `imputed`, so the old count made it meaningless."""
        _, stats = conserved
        row = stats.iloc[0]
        assert int(row["imputed"]) < len(conserved[0])

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


class TestAnUnfillableGapIsCounted:
    """The defect these counts had: `GasgebruikDiff` held 38.8M empty cells
    while its sidecar row said every missing cell was filled."""

    @pytest.fixture()
    def no_average(self):
        """A gap with no closing reading and no project average, so nothing can
        establish a level and the cells stay empty."""
        return _impute(_frame(
            cumulative=[100.0, pd.NA, pd.NA, pd.NA],
            diffs=[0.0, pd.NA, pd.NA, pd.NA],
            averages=[pd.NA] * 4,
        ))

    def test_the_empty_cells_are_reported_as_empty(self, no_average):
        filled, stats = no_average
        still_empty = int(filled[DIFF].isna().sum())
        assert still_empty > 0, "the fixture must leave cells empty"
        assert int(stats.iloc[0]["imputed_na"]) == still_empty

    def test_imputed_does_not_claim_them(self, no_average):
        _, stats = no_average
        row = stats.iloc[0]
        assert int(row["imputed"]) < int(row["missing"])

    def test_they_still_partition_missing(self, no_average):
        _, stats = no_average
        row = stats.iloc[0]
        assert int(row["imputed"]) + int(row["imputed_na"]) == int(row["missing"])


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
