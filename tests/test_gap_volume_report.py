"""The fill reports, per gap, what it wrote, what the bounds moved, and what
could not be placed.

Collected inside the fill before the threshold pass, carried out through the
`volume_reports` list, and keyed on household, column and gap. Data-volume
sums take min_count=1, so a gap holding no values reports pd.NA, never zero.
"""

import pandas as pd
import pytest
from conftest import cadence_table
from test_constraint_package import _constraints
from test_row_bounds_in_the_fill import CUM, DIFF, FOLLOWS_K, _frame

from etdtransform.vectorized_impute import ImputeType, impute_and_normalize

pytestmark = pytest.mark.unit


def _impute_with_report(frame, constraints=None):
    reports: list = []
    out, _, _ = impute_and_normalize(
        frame,
        [CUM],
        "ProjectIdBSV",
        cadence=cadence_table(frame, [CUM]),
        constraints=constraints,
        volume_reports=reports,
    )
    assert len(reports) == 1
    return out, reports[0]


def _the_gap(report):
    """The gap under test: a household's row-0 seed makes its own one-cell
    pseudo-gap, so take the longest gap rather than the first row."""
    return report.sort_values("gap_length", ascending=False).iloc[0]


class TestAnAnchoredGap:
    def test_totals_and_moved_volume(self):
        # The ceiling scenario from the allocation tests: routes write
        # [0.6, 1.2, 1.2], the cap cuts 0.2 and the give-back places it, so
        # both totals equal the jump and the gap deviates by nothing.
        _, report = _impute_with_report(
            _frame(
                [100.0, pd.NA, pd.NA, 103.0],
                [pd.NA, 0.5, 1.0, 1.5],
                k=[pd.NA, pd.NA, pd.NA, 1.0],
            ),
            constraints=_constraints(*FOLLOWS_K),
        )
        row = _the_gap(report)
        assert row["column"] == DIFF
        assert bool(row["anchored"])
        assert float(row["gap_jump"]) == pytest.approx(3.0)
        assert float(row["total_routed"]) == pytest.approx(3.0)
        assert float(row["total_final"]) == pytest.approx(3.0)
        assert float(row["deviation"]) == pytest.approx(0.0)
        assert int(row["cells_clipped"]) == 1
        assert int(row["cells_absorbed"]) == 2
        assert float(row["redistributed_volume"]) == pytest.approx(0.2)
        assert float(row["unplaced_volume"]) == pytest.approx(0.0)
        assert row["rule_expressions"] == "K"
        assert int(row["impute_type_bits"]) & ImputeType.CLIPPED.value

    def test_an_unmeetable_jump_reports_its_deficit(self):
        # Every cell ends at a bound and 0.3 has nowhere to go.
        _, report = _impute_with_report(
            _frame(
                [100.0, pd.NA, pd.NA, 103.0],
                [pd.NA, 1.0, 1.0, 0.0],
                k=[pd.NA, 1.2, 1.2, 0.3],
            ),
            constraints=_constraints(*FOLLOWS_K),
        )
        row = _the_gap(report)
        assert float(row["total_final"]) == pytest.approx(2.7)
        assert float(row["deviation"]) == pytest.approx(-0.3)
        assert float(row["deviation_fraction"]) == pytest.approx(-0.1)
        assert float(row["unplaced_volume"]) == pytest.approx(0.3)
        assert int(row["impute_type_bits"]) & ImputeType.INFEASIBLE_GAP.value

    def test_the_reserve_volume_is_reported(self):
        _, report = _impute_with_report(
            _frame(
                [100.0, pd.NA, pd.NA, 103.0],
                [pd.NA, 1.0, 1.0, 0.0],
                k=[pd.NA, 1.2, 1.2, pd.NA],
            ),
            constraints=_constraints(*FOLLOWS_K),
        )
        row = _the_gap(report)
        assert float(row["uniform_residual_volume"]) == pytest.approx(0.6)
        assert int(row["cells_residual"]) == 1
        assert float(row["total_final"]) == pytest.approx(3.0)


class TestAnUnanchoredGap:
    def test_what_clipping_removed_is_reported_with_no_jump(self):
        cumulative = [10.0, 10.5, 11.0, 11.5, 12.0, 12.5] + [pd.NA] * 4
        diffs = [0.5] * 6 + [pd.NA] * 4
        k = [pd.NA] * 7 + [0.3] + [pd.NA] * 2
        _, report = _impute_with_report(
            _frame(cumulative, [1.0] * 10, diffs=diffs, k=k),
            constraints=_constraints(*FOLLOWS_K),
        )
        row = _the_gap(report)
        assert not bool(row["anchored"])
        assert pd.isna(row["gap_jump"])
        assert pd.isna(row["deviation"])
        # K pins the row to [0.3, 0.3]; its bounded average is 0.3, the
        # household factor halves it to 0.15, and the floor raises it back to
        # 0.3 -- so the clip put 0.15 IN, negative under the sign convention.
        assert float(row["unanchored_removed_volume"]) == pytest.approx(-0.15)
        assert float(row["redistributed_volume"]) == pytest.approx(0.0), (
            "an unanchored gap gives nothing back"
        )


class TestWithoutConstraints:
    def test_the_report_still_carries_the_fill_totals(self):
        _, report = _impute_with_report(
            _frame([100.0, pd.NA, pd.NA, 103.0], [pd.NA, 0.5, 1.0, 1.5]),
        )
        row = _the_gap(report)
        assert float(row["total_routed"]) == pytest.approx(3.0)
        assert float(row["total_final"]) == pytest.approx(3.0)
        assert int(row["cells_clipped"]) == 0
        assert row["rule_expressions"] == ""


class TestTheWriter:
    def test_the_artifact_and_both_rollups_are_written(self, tmp_path):
        from etdtransform.aggregate import _write_volume_reports

        _, report = _impute_with_report(
            _frame(
                [100.0, pd.NA, pd.NA, 103.0],
                [pd.NA, 0.5, 1.0, 1.5],
                k=[pd.NA, pd.NA, pd.NA, 1.0],
            ),
            constraints=_constraints(*FOLLOWS_K),
        )
        _write_volume_reports([report], tmp_path)

        gaps = pd.read_parquet(
            tmp_path / "impute_volume_gap.parquet",
            dtype_backend="numpy_nullable",
        )
        assert len(gaps) == len(report)

        household = pd.read_parquet(
            tmp_path / "impute_volume_household.parquet",
            dtype_backend="numpy_nullable",
        )
        row = household[household["column"] == DIFF].iloc[0]
        assert int(row["n_gaps"]) == len(report)
        # The seed pseudo-gap's NA totals must not zero the sum: min_count=1
        # keeps the real gap's figure.
        assert float(row["total_final"]) == pytest.approx(3.0)
        assert int(row["n_gaps_infeasible"]) == 0

        project = pd.read_parquet(
            tmp_path / "impute_volume_project.parquet",
            dtype_backend="numpy_nullable",
        )
        assert list(project.columns[:2]) == ["ProjectIdBSV", "column"]
        assert float(
            project[project["column"] == DIFF].iloc[0]["redistributed_volume"]
        ) == pytest.approx(0.2)

    def test_nothing_to_report_writes_nothing(self, tmp_path):
        from etdtransform.aggregate import _write_volume_reports

        _write_volume_reports([], tmp_path)
        _write_volume_reports(None, tmp_path)
        assert not list(tmp_path.glob("impute_volume_*.parquet"))


class TestTheGapBitsAreComplete:
    def test_clipped_average_reaches_the_gap_bits(self):
        """The signals are written before the report collects, so a gap whose
        shape the bounded average changed is identifiable from the report
        alone -- the regression that motivated this: gap bits missing
        CLIPPED_AVERAGE made flagged gaps undercountable."""
        _, report = _impute_with_report(
            _frame(
                [100.0, pd.NA, pd.NA, 103.0],
                [pd.NA, 0.5, 1.0, 1.5],
                k=[pd.NA, pd.NA, pd.NA, 1.0],
            ),
            constraints=_constraints(*FOLLOWS_K),
        )
        row = _the_gap(report)
        assert int(row["impute_type_bits"]) & ImputeType.CLIPPED_AVERAGE.value
