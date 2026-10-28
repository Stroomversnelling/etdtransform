"""Group 5's data-driven fill rules: no metadata reaches the fill.

Three behaviours, one file:

- The no-delivered-no-fill rule: a (household, column) with zero delivered
  readings is never filled -- not treated as a gap, no flags, no volume
  rows, cells stay NA. Delivered is the usable-readings predicate, so a
  column of pipeline-invented zeros (NOT_A_MEASUREMENT bits) does not
  count. This closes the broadcast bug class where an undelivered column
  was invented wholesale from peer averages.
- JITTER_CANDIDATE: an empty bound whose contradiction is at metering
  resolution is marked for assessment against the ANNOUNCED scale
  (DEFAULT_JITTER_ASSESSMENT_SCALE_KWH -- a named constant pending a
  metadata metering-resolution field, which the data model does not yet
  carry). Signal only; what fills or stays unfilled is unchanged. The
  per-gap overshoot maximum lands in the accumulator so the assessment is
  never locked to the scale.
- The whole-household no-known short-circuit: a household none of whose
  rule inputs is ever known takes the presolved bound without per-row
  evaluation, gated below the measured ~50% evaluated-share crossover
  (docs/performance/fill-eval-synthetic-2026-10-01.md). Mathematically
  identical to the full evaluation; pinned by comparing both paths.
"""

import logging

import pandas as pd
import pytest
from conftest import cadence_table
from etdmap import ids
from etdmap.flags import MappingFlag
from etdmap.testing import household_batch_of, household_batches_frame
from test_constraint_package import _constraints

import etdtransform.vectorized_impute as vi
from etdtransform.vectorized_impute import (
    DEFAULT_JITTER_ASSESSMENT_SCALE_KWH,
    ImputeType,
    impute_and_normalize,
)

pytestmark = pytest.mark.unit

CUM = "ElektriciteitNetgebruik"
DIFF = "ElektriciteitNetgebruikDiff"


def _household(household_id, n, **columns):
    """One household batch of *household_id* over *n* rows, tz-naive at the 5-minute
    grid. A `_flags` column holds bits, so it is Int64 rather than Float64.

    Every caller states the cumulative column alongside its interval one, so
    the builder completes neither half: delivery is read off the cumulative
    readings the test puts there.
    """
    return household_batches_frame(
        {household_batch_of(household_id): {
            name: list(values) for name, values in columns.items()
        }},
        start="2026-01-01", freq="5min", tz=None, n_rows=n,
        dtypes={
            name: "Int64" for name in columns if name.endswith("_flags")
        },
    )


def _frame(*households):
    return pd.concat(households, ignore_index=True)


def _impute(frame, constraints=None, volume_reports=None):
    out, _, _ = impute_and_normalize(
        frame,
        [CUM],
        ids.PROJECT,
        cadence=cadence_table(frame, [CUM]),
        constraints=constraints,
        volume_reports=volume_reports,
    )
    return out


def _flags(out, row, column=DIFF):
    value = out.loc[row, f"{column}_impute_type"]
    return 0 if pd.isna(value) else int(value)


def _delivering_household(household_id=1, n=4):
    return _household(
        household_id, n,
        **{
            CUM: [100.0] + [pd.NA] * (n - 2) + [103.0],
            DIFF: [pd.NA] * n,
            f"{DIFF}_avg": [pd.NA] + [1.0] * (n - 1),
        },
    )


class TestNoDeliveredNoFill:
    def _all_na_household(self, household_id=2, n=4):
        return _household(
            household_id, n,
            **{
                CUM: [pd.NA] * n,
                DIFF: [pd.NA] * n,
                f"{DIFF}_avg": [pd.NA] + [1.0] * (n - 1),
            },
        )

    def test_an_all_na_household_is_untouched_with_no_flags(self):
        volume_reports = []
        out = _impute(
            _frame(_delivering_household(1), self._all_na_household(2)),
            volume_reports=volume_reports,
        )
        empty_rows = out.index[out[ids.HOUSEHOLD] == 2]
        assert out.loc[empty_rows, DIFF].isna().all()
        assert not out.loc[empty_rows, f"{DIFF}_is_imputed"].any()
        assert out.loc[empty_rows, f"{DIFF}_impute_type"].isna().all()

    def test_the_skip_is_per_household_not_per_column(self):
        # The neighbour WITH data fills in the same chunk, same column.
        out = _impute(
            _frame(_delivering_household(1), self._all_na_household(2)),
        )
        filled_rows = out.index[(out[ids.HOUSEHOLD] == 1)][1:]
        assert out.loc[filled_rows, DIFF].notna().all()
        assert out.loc[filled_rows, f"{DIFF}_is_imputed"].all()

    def test_no_volume_rows_are_collected_for_the_skipped_household_batch(self):
        volume_reports = []
        _impute(
            _frame(_delivering_household(1), self._all_na_household(2)),
            volume_reports=volume_reports,
        )
        gaps = volume_reports[-1]
        assert not gaps.empty
        # A gap belongs to the series it was found in, so the accumulator
        # names the household batch.
        assert set(gaps[ids.HOUSEHOLD_BATCH].astype(int)) == {
            household_batch_of(1)
        }

    def test_invented_zeros_do_not_count_as_delivered(self):
        # The zero-prefill era: a cumulative of pipeline-written zeros,
        # condemned by its own findings column. Previously the zero jump
        # anchored gaps and the column ZERO_FILLed; now nothing counts as
        # delivered, so nothing is a gap.
        n = 4
        invented = _household(
            2, n,
            **{
                CUM: [0.0] * n,
                f"{CUM}_flags": [MappingFlag.FILLED_ZERO.value] * n,
                DIFF: [pd.NA] * n,
                f"{DIFF}_avg": [pd.NA] + [1.0] * (n - 1),
            },
        )
        out = _impute(_frame(_delivering_household(1), invented))
        empty_rows = out.index[out[ids.HOUSEHOLD] == 2]
        assert out.loc[empty_rows, DIFF].isna().all()
        assert out.loc[empty_rows, f"{DIFF}_impute_type"].isna().all()

    def test_one_delivered_reading_keeps_the_household_fillable(self):
        # A single usable cumulative pair: fills proceed as today.
        one_reading = _household(
            2, 4,
            **{
                CUM: [200.0, pd.NA, pd.NA, 202.0],
                DIFF: [pd.NA] * 4,
                f"{DIFF}_avg": [pd.NA, 1.0, 1.0, 1.0],
            },
        )
        out = _impute(_frame(_delivering_household(1), one_reading))
        rows = out.index[out[ids.HOUSEHOLD] == 2][1:]
        assert out.loc[rows, DIFF].notna().all()


class TestJitterCandidate:
    # K pins the row high while L pins it low; the overshoot is their gap.
    def _run(self, k_value, l_value):
        frame = _household(
            1, 4,
            **{
                CUM: [100.0, pd.NA, pd.NA, 103.0],
                DIFF: [pd.NA] * 4,
                f"{DIFF}_avg": [pd.NA, 1.0, 1.0, 1.0],
                "K": [pd.NA, pd.NA, k_value, pd.NA],
                "L": [pd.NA, pd.NA, l_value, pd.NA],
            },
        )
        volume_reports = []
        out = _impute(
            frame,
            constraints=_constraints(
                (DIFF, "K", {"K": 1}),
                (DIFF, "L", {"L": 1}),
            ),
            volume_reports=volume_reports,
        )
        return out, volume_reports[-1]

    def test_a_metering_scale_contradiction_is_a_candidate(self):
        out, _ = self._run(1.0, 0.99)
        value = _flags(out, 2)
        assert value & ImputeType.JITTER_CANDIDATE.value
        assert value & ImputeType.INFEASIBLE_BOUNDS_HARD.value

    def test_the_signal_changes_no_fill_behaviour(self):
        out, _ = self._run(1.0, 0.99)
        assert pd.isna(out.loc[2, DIFF])
        assert _flags(out, 2) & ImputeType.GAP_NOT_FILLED.value
        for row in (1, 3):
            assert pd.notna(out.loc[row, DIFF])

    def test_a_large_contradiction_is_not_a_candidate(self):
        out, _ = self._run(1.0, 0.5)
        value = _flags(out, 2)
        assert value & ImputeType.INFEASIBLE_BOUNDS_HARD.value
        assert not value & ImputeType.JITTER_CANDIDATE.value

    def test_the_accumulator_carries_the_overshoot_magnitude(self):
        _, gaps = self._run(1.0, 0.99)
        assert "bound_overshoot_max" in gaps.columns
        overshoot = gaps["bound_overshoot_max"].dropna()
        assert len(overshoot) == 1
        assert float(overshoot.iloc[0]) == pytest.approx(0.01)

    def test_jitter_candidate_is_a_signal_never_an_action(self):
        assert ImputeType.JITTER_CANDIDATE & ImputeType.SIGNALS
        assert not ImputeType.JITTER_CANDIDATE & ImputeType.ACTIONS
        assert not ImputeType.JITTER_CANDIDATE & ImputeType.IMPUTED_VALUE

    def test_the_assessment_scale_is_announced_not_silent(self):
        # The named constant is the scale pending a metadata field; a
        # per-column override takes precedence over the default.
        assert DEFAULT_JITTER_ASSESSMENT_SCALE_KWH == 0.02
        assert vi._jitter_assessment_scale(DIFF) == 0.02
        try:
            vi.JITTER_ASSESSMENT_SCALE_KWH[DIFF] = 0.005
            assert vi._jitter_assessment_scale(DIFF) == 0.005
        finally:
            vi.JITTER_ASSESSMENT_SCALE_KWH.clear()


class TestNoKnownShortCircuit:
    def _frame_with_unknown_majority(self):
        # Household 1 (4 rows) knows K once; household 2 (12 rows) never
        # does: evaluated share 0.25, under the gate, so household 2 takes
        # the presolved bound without per-row evaluation.
        known = _household(
            1, 4,
            **{
                CUM: [100.0, pd.NA, pd.NA, 103.0],
                DIFF: [pd.NA] * 4,
                f"{DIFF}_avg": [pd.NA, 1.0, 1.0, 1.0],
                "K": [pd.NA, pd.NA, 2.0, pd.NA],
            },
        )
        unknown = _household(
            2, 12,
            **{
                CUM: [200.0] + [pd.NA] * 10 + [211.0],
                DIFF: [pd.NA] * 12,
                f"{DIFF}_avg": [pd.NA] + [1.0] * 11,
                "K": [pd.NA] * 12,
            },
        )
        return _frame(known, unknown)

    def _bounds(self, frame, monkeypatch, gate):
        monkeypatch.setattr(vi, "MASKED_EVAL_MAX_EVALUATED_SHARE", gate)
        return vi._row_bounds_per_cadence(
            frame,
            DIFF,
            _constraints((DIFF, "K", {"K": 1})),
            cadence_table(frame, [CUM]),
            {},
            physical_model_by_project={1: "Universeel"},
        )

    def test_short_circuited_bounds_equal_the_full_evaluation(
        self, monkeypatch
    ):
        frame = self._frame_with_unknown_majority()
        lo_split, hi_split, imp_split = self._bounds(frame, monkeypatch, 0.5)
        # Gate at 0.0: the share is never below it, so every row evaluates.
        lo_full, hi_full, imp_full = self._bounds(frame, monkeypatch, 0.0)
        pd.testing.assert_series_equal(lo_split, lo_full)
        pd.testing.assert_series_equal(hi_split, hi_full)
        pd.testing.assert_series_equal(imp_split, imp_full)

    def test_the_short_circuit_engages_below_the_gate(
        self, monkeypatch, caplog
    ):
        frame = self._frame_with_unknown_majority()
        with caplog.at_level(logging.INFO):
            self._bounds(frame, monkeypatch, 0.5)
        assert any(
            "presolved bound without per-row evaluation" in record.message
            for record in caplog.records
        )

    def test_the_whole_path_runs_above_the_gate(self, monkeypatch, caplog):
        frame = self._frame_with_unknown_majority()
        with caplog.at_level(logging.INFO):
            self._bounds(frame, monkeypatch, 0.0)
        assert not any(
            "presolved bound without per-row evaluation" in record.message
            for record in caplog.records
        )

    def test_fill_outputs_are_identical_under_both_paths(self, monkeypatch):
        constraints = _constraints((DIFF, "K", {"K": 1}))
        monkeypatch.setattr(vi, "MASKED_EVAL_MAX_EVALUATED_SHARE", 0.5)
        split = _impute(
            self._frame_with_unknown_majority(), constraints=constraints
        )
        monkeypatch.setattr(vi, "MASKED_EVAL_MAX_EVALUATED_SHARE", 0.0)
        full = _impute(
            self._frame_with_unknown_majority(), constraints=constraints
        )
        pd.testing.assert_series_equal(split[DIFF], full[DIFF])
        pd.testing.assert_series_equal(
            split[f"{DIFF}_impute_type"], full[f"{DIFF}_impute_type"]
        )
