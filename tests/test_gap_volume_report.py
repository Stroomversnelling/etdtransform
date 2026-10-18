"""The fill accumulates, per gap, the TRANSIENT state the persisted outputs
cannot reproduce: pre-threshold totals and bits, the routed and moved
volumes, and the span keys the post-processor joins on.

Collected inside the fill before the threshold pass, carried out through the
`volume_reports` list, and keyed on household, column and gap span. The
derivable half of the old full artifact (cell counts, gap_length, deviation,
rule_expressions) is assembled post-hoc by etdworkflow/postprocess_reports.py
and tested there; this file covers what the RUN must get right. Data-volume
sums take min_count=1, so a gap holding no values reports pd.NA, never zero.
"""

import numpy as np
import pandas as pd
import pytest
from conftest import cadence_table
from test_constraint_package import _constraints
from test_row_bounds_in_the_fill import CUM, DIFF, FOLLOWS_K, _frame

from etdtransform.vectorized_impute import (
    ImputeType,
    _bits_or_by_group,
    impute_and_normalize,
)

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
    pseudo-gap, so take the longest span rather than the first row."""
    span = report["gap_end"] - report["gap_start"]
    return report.loc[span.idxmax()]


class TestAnAnchoredGap:
    def test_totals_and_moved_volume(self):
        # The ceiling scenario from the allocation tests: routes write
        # [0.6, 1.2, 1.2], the cap cuts 0.2 and the give-back places it, so
        # both totals equal the jump.
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
        assert float(row["redistributed_volume"]) == pytest.approx(0.2)
        assert float(row["unplaced_volume"]) == pytest.approx(0.0)
        assert int(row["impute_type_bits"]) & ImputeType.CLIPPED.value

    def test_the_span_keys_cover_the_gap(self):
        frame = _frame(
            [100.0, pd.NA, pd.NA, 103.0],
            [pd.NA, 0.5, 1.0, 1.5],
            k=[pd.NA, pd.NA, pd.NA, 1.0],
        )
        dates = frame["ReadingDate"].tolist()
        _, report = _impute_with_report(
            frame, constraints=_constraints(*FOLLOWS_K)
        )
        row = _the_gap(report)
        # The gap is rows 1..3 of the household; the span keys are its first
        # and last ReadingDate so the post-processor can join on them without
        # trusting re-derived gap numbering.
        assert row["gap_start"] == dates[1]
        assert row["gap_end"] == dates[3]

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
        assert float(row["total_final"]) == pytest.approx(3.0)
        assert int(row["impute_type_bits"]) & ImputeType.UNIFORM_RESIDUAL.value


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
        assert not int(row["impute_type_bits"]) & ImputeType.CLIPPED.value


class TestTheWriter:
    def test_the_accumulator_is_written_and_nothing_else(self, tmp_path):
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
            tmp_path / "impute_volume_gap_accumulator.parquet",
            dtype_backend="numpy_nullable",
        )
        assert len(gaps) == len(report)
        for column in ("gap_start", "gap_end", "total_final", "total_routed",
                       "impute_type_bits", "redistributed_volume"):
            assert column in gaps.columns

        # The full artifact and its rollups are post-processing outputs; the
        # run writes none of them.
        written = {p.name for p in tmp_path.glob("impute_volume_*.parquet")}
        assert written == {"impute_volume_gap_accumulator.parquet"}

    def test_nothing_to_report_writes_nothing(self, tmp_path):
        from etdtransform.aggregate import _write_volume_reports

        _write_volume_reports([], tmp_path)
        _write_volume_reports(None, tmp_path)
        assert not list(tmp_path.glob("impute_volume_*.parquet"))


def _bits_or_by_group_reference(values, group):
    """Bitwise OR per group via a python reduce per group."""
    return (
        values.groupby(group)
        .agg(lambda v: int(np.bitwise_or.reduce(v.to_numpy(dtype="int64"))))
        .astype("Int64")
    )


class TestTheGapBitsReduction:
    """`_bits_or_by_group` must agree with a python reduce per group on every
    shape the fill can hand it."""

    @staticmethod
    def _assert_agrees(bit_values, group_values):
        group = pd.Series(group_values, name="cumulative_value_group")
        values = pd.Series(bit_values, dtype="Int64").fillna(
            ImputeType.NONE.value
        )
        pd.testing.assert_series_equal(
            _bits_or_by_group(values, group),
            _bits_or_by_group_reference(values, group),
        )

    def test_gaps_whose_bits_differ(self):
        self._assert_agrees(
            [
                ImputeType.SCALED_FILL.value,
                ImputeType.CLIPPED.value,
                ImputeType.UNIFORM_FILL.value,
                ImputeType.INFEASIBLE_GAP.value,
                ImputeType.GAP_NOT_FILLED.value,
            ],
            [1, 1, 2, 2, 3],
        )

    def test_bits_accumulate_within_one_gap(self):
        self._assert_agrees(
            [
                ImputeType.SCALED_FILL.value,
                ImputeType.CLIPPED.value,
                ImputeType.CLIPPED_AVERAGE.value,
            ],
            [7, 7, 7],
        )

    def test_an_all_zero_gap(self):
        self._assert_agrees(
            [ImputeType.NONE.value, ImputeType.NONE.value,
             ImputeType.SCALED_FILL.value],
            [1, 1, 2],
        )

    def test_a_single_row_gap(self):
        self._assert_agrees([ImputeType.CLIPPED.value], [4])

    def test_single_row_gaps_only(self):
        self._assert_agrees(
            [ImputeType.SCALED_FILL.value, ImputeType.CLIPPED.value,
             ImputeType.GAP_NOT_FILLED.value],
            [1, 2, 3],
        )

    def test_na_bits_count_as_none(self):
        self._assert_agrees(
            [pd.NA, ImputeType.SCALED_FILL.value, pd.NA],
            [1, 1, 2],
        )

    def test_every_bit_set_somewhere_in_the_gap(self):
        self._assert_agrees(
            [member.value for member in ImputeType if member.value],
            [1] * len([member for member in ImputeType if member.value]),
        )

    def test_non_contiguous_groups_still_fold(self):
        """Interleaved group ids cannot arise from the fill's cumsum grouping;
        the fold keeps the reduction correct if they ever do."""
        self._assert_agrees(
            [
                ImputeType.SCALED_FILL.value,
                ImputeType.CLIPPED.value,
                ImputeType.UNIFORM_FILL.value,
            ],
            [1, 2, 1],
        )


class TestTheGapBitsAreComplete:
    def test_clipped_average_reaches_the_gap_bits(self):
        """The signals are written before the accumulator collects, so a gap
        whose shape the bounded average changed is identifiable from the
        accumulator alone -- the regression that motivated this: gap bits
        missing CLIPPED_AVERAGE made flagged gaps undercountable. The bits
        stay in the accumulator precisely because they are a PRE-THRESHOLD
        snapshot no persisted flag column reproduces."""
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
