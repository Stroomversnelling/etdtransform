"""An empty row bound permits no value; an unmeetable gap total is a finding.

Two rules that disagree about a row leave it no permitted value: nothing is
written there, the cell reads GAP_NOT_FILLED, and the cause is split by what fed
the bound -- measurements alone (INFEASIBLE_BOUNDS_HARD) or a cell this run
filled (INFEASIBLE_BOUNDS_SOFT). A gap holding such a cell, or one whose
allocation could not place all its volume, carries INFEASIBLE_GAP on every
cell.
"""

import pandas as pd
import pytest
from conftest import cadence_table
from test_constraint_package import _constraints

from etdtransform.vectorized_impute import ImputeType, impute_and_normalize

pytestmark = pytest.mark.unit

CUM = "ElektriciteitNetgebruik"
DIFF = "ElektriciteitNetgebruikDiff"
CUM_B = "ElektriciteitTeruglevering"
DIFF_B = "ElektriciteitTerugleveringDiff"

# K and L are unbounded in the thresholds; known together they can disagree.
CONFLICT = (
    (DIFF, "K", {"K": 1}),
    (DIFF, "L", {"L": 1}),
)


def _frame(columns):
    n = max(len(v) for v in columns.values())
    data = {
        "HuisIdBSV": pd.array([1] * n, dtype="Int64"),
        "HuisBatchIdBSV": pd.array([value + 1000 for value in [1] * n], dtype="Int64"),
        "ProjectIdBSV": pd.array([1] * n, dtype="Int64"),
        "ReadingDate": pd.date_range("2026-01-01", periods=n, freq="5min"),
    }
    for name, values in columns.items():
        data[name] = pd.array(values, dtype="Float64")
    return pd.DataFrame(data)


def _impute(frame, cums, constraints):
    out, _, _ = impute_and_normalize(
        frame,
        cums,
        "ProjectIdBSV",
        cadence=cadence_table(frame, cums),
        constraints=constraints,
    )
    return out


def _flags(out, row, column=DIFF):
    value = out.loc[row, f"{column}_impute_type"]
    return 0 if pd.isna(value) else int(value)


class TestACellWithNoPermittedValue:
    def _run(self, averages):
        # K pins row 2 to [1, 1] while L pins it to [0.5, 0.5]: empty.
        return _impute(
            _frame({
                CUM: [100.0, pd.NA, pd.NA, 103.0],
                DIFF: [pd.NA] * 4,
                f"{DIFF}_avg": averages,
                "K": [pd.NA, pd.NA, 1.0, pd.NA],
                "L": [pd.NA, pd.NA, 0.5, pd.NA],
            }),
            [CUM],
            _constraints(*CONFLICT),
        )

    def test_nothing_is_written_and_the_cell_reads_not_filled(self):
        out = self._run([pd.NA, 1.0, 1.0, 1.0])
        assert pd.isna(out.loc[2, DIFF])
        assert not out.loc[2, f"{DIFF}_is_imputed"]
        value = _flags(out, 2)
        assert value & ImputeType.GAP_NOT_FILLED.value
        assert not value & ImputeType.GAP_FILLED.value

    def test_measurements_alone_read_hard_not_soft(self):
        out = self._run([pd.NA, 1.0, 1.0, 1.0])
        value = _flags(out, 2)
        assert value & ImputeType.INFEASIBLE_BOUNDS_HARD.value
        assert not value & ImputeType.INFEASIBLE_BOUNDS_SOFT.value

    def test_the_gap_carries_infeasible_gap_on_every_cell(self):
        out = self._run([pd.NA, 1.0, 1.0, 1.0])
        for row in (1, 2, 3):
            assert _flags(out, row) & ImputeType.INFEASIBLE_GAP.value, row
        assert not _flags(out, 0) & ImputeType.INFEASIBLE_GAP.value

    def test_the_other_cells_still_fill(self):
        out = self._run([pd.NA, 1.0, 1.0, 1.0])
        for row in (1, 3):
            assert pd.notna(out.loc[row, DIFF])
            assert _flags(out, row) & ImputeType.GAP_FILLED.value

    def test_a_uniform_gap_skips_the_cell_too(self):
        # No shape anywhere: the spread must still not write into the cell.
        out = self._run([pd.NA] * 4)
        assert pd.isna(out.loc[2, DIFF])
        assert _flags(out, 2) & ImputeType.GAP_NOT_FILLED.value
        for row in (1, 3):
            assert float(out.loc[row, DIFF]) == pytest.approx(1.0)
            assert _flags(out, row) & ImputeType.UNIFORM_FILL.value
            assert _flags(out, row) & ImputeType.INFEASIBLE_GAP.value

    def test_a_zero_fill_gap_skips_the_cell_but_is_not_infeasible(self):
        # A negative jump is a meter reset, not an unmeetable positive total.
        out = _impute(
            _frame({
                CUM: [100.0, pd.NA, pd.NA, 90.0],
                DIFF: [pd.NA] * 4,
                f"{DIFF}_avg": [pd.NA, 1.0, 1.0, 1.0],
                "K": [pd.NA, pd.NA, 1.0, pd.NA],
                "L": [pd.NA, pd.NA, 0.5, pd.NA],
            }),
            [CUM],
            _constraints(*CONFLICT),
        )
        assert pd.isna(out.loc[2, DIFF])
        assert _flags(out, 2) & ImputeType.INFEASIBLE_BOUNDS_HARD.value
        for row in (1, 3):
            assert float(out.loc[row, DIFF]) == 0.0
        for row in (0, 1, 2, 3):
            assert not _flags(out, row) & ImputeType.INFEASIBLE_GAP.value, row


class TestABoundFedByTheFill:
    def test_an_imputed_known_makes_the_contradiction_soft(self):
        # Column A fills first; B's bound then rests on A's imputed cell
        # through one rule while a delivered L contradicts it through the
        # other. The contradiction is one this run made.
        out = _impute(
            _frame({
                CUM: [100.0, pd.NA, pd.NA, 103.0],
                DIFF: [pd.NA] * 4,
                f"{DIFF}_avg": [pd.NA, 1.0, 1.0, 1.0],
                CUM_B: [50.0, 50.5, pd.NA, 51.5],
                DIFF_B: [pd.NA, 0.5, pd.NA, pd.NA],
                f"{DIFF_B}_avg": [pd.NA, 1.0, 1.0, 1.0],
                "L": [pd.NA, pd.NA, 0.5, pd.NA],
            }),
            [CUM, CUM_B],
            _constraints(
                (DIFF_B, DIFF, {DIFF: 1}),
                (DIFF_B, "L", {"L": 1}),
            ),
        )
        # A's row 2 was filled by this run, and B's bound there rests on it.
        assert out.loc[2, f"{DIFF}_is_imputed"]
        value = _flags(out, 2, DIFF_B)
        assert value & ImputeType.INFEASIBLE_BOUNDS_SOFT.value
        assert not value & ImputeType.INFEASIBLE_BOUNDS_HARD.value
        assert pd.isna(out.loc[2, DIFF_B])
        assert value & ImputeType.GAP_NOT_FILLED.value

    def test_a_bound_resting_on_a_fill_is_recorded_where_it_acts(self):
        # B's bound follows A per row, and A's gap cells were filled by this
        # run -- so what B writes under that bound inherits the fill, and
        # says so. A's own cells carry no such mark: its bounds rest on the
        # declared ranges alone.
        out = _impute(
            _frame({
                CUM: [100.0, 100.5, pd.NA, 102.0],
                DIFF: [pd.NA, 0.5, pd.NA, pd.NA],
                f"{DIFF}_avg": [pd.NA, 1.0, 1.0, 1.0],
                CUM_B: [50.0, 50.5, pd.NA, 51.5],
                DIFF_B: [pd.NA, 0.5, pd.NA, pd.NA],
                f"{DIFF_B}_avg": [pd.NA, 1.0, 0.5, 1.0],
            }),
            [CUM, CUM_B],
            _constraints((DIFF_B, DIFF, {DIFF: 1})),
        )
        for row in (2, 3):
            assert out.loc[row, f"{DIFF}_is_imputed"]
            assert _flags(out, row, DIFF_B) & ImputeType.BOUND_FROM_IMPUTED.value
            assert not _flags(out, row, DIFF) & ImputeType.BOUND_FROM_IMPUTED.value
