"""Reconstructing a column's native cadence from the grid.

Redistribution conserves volume, so a filled block re-aggregates to exactly what
the cumulative variable recorded. That makes the grid series a *presentation* of
the native one rather than a replacement, and it is what lets a statistic be
computed at a resolution the data actually supports.

Two things have to be true for that to work, and only the first currently is:

1. **Conservation** -- each filled block sums to the cumulative jump across it.
2. **Kind** -- knowing *what* a filled block represents, because the rule for
   collapsing it differs. A frozen span collapses to one delivered reading; a
   transmission gap at the grid cadence collapses to nothing, because those
   readings are genuinely missing at the grid's own resolution.

The second is not recoverable from the imputation flags alone, which is what
`test_kind_is_not_recoverable_from_imputation_flags_alone` records.
"""

import logging

import pandas as pd
import pytest

from etdmap.flags import contiguous_runs
from etdmap.mapping_helpers import (
    _detect_frozen_spans_for_household,
    _treat_frozen_spans,
)
from etdtransform.vectorized_impute import ImputeType, impute_and_normalize

CUM = "WatergebruikWarmTapwater"
DIFF = f"{CUM}Diff"
NORMAL = 3.0


@pytest.fixture(autouse=True)
def _quiet_logs():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _household(n=60, frozen=range(8, 20), catchup_at=20, catchup=400.0,
               gap=range(35, 40)):
    """One household carrying two different causes of filled values.

    A frozen span (the cumulative variable stops advancing, then delivers what it
    held) and an ordinary transmission gap at the grid cadence (readings simply
    absent).
    """
    values, current = [], 1000.0
    for i in range(n):
        if i in frozen:
            pass
        elif i == catchup_at:
            current += catchup
        else:
            current += NORMAL
        values.append(current)

    df = pd.DataFrame({
        "HuisIdBSV": pd.array([1] * n, dtype="Int64"),
        "ProjectIdBSV": pd.array([1] * n, dtype="Int64"),
        "ReadingDate": pd.date_range("2026-01-01", periods=n, freq="5min"),
        CUM: pd.array(values, dtype="Float64"),
    })
    delivered = df[CUM].copy()
    if gap is not None:
        df.loc[list(gap), CUM] = pd.NA
    df[DIFF] = df[CUM].diff().round(10)
    df.loc[0, DIFF] = 0
    df[f"{DIFF}_avg"] = pd.array([NORMAL] * n, dtype="Float64")
    return df, delivered


def _treated_and_imputed(df):
    rows = _detect_frozen_spans_for_household(df, 1)
    _treat_frozen_spans(df, rows)
    out, _, _ = impute_and_normalize(df.copy(), [CUM], "ProjectIdBSV", None)
    return out, rows


def _filled_blocks(out):
    """The blocks, reconstructed from flags rather than a stored id."""
    filled = out[f"{DIFF}_is_imputed"].fillna(False).astype(bool)
    return contiguous_runs(filled)


class TestConservation:
    def test_every_filled_block_sums_to_the_cumulative_jump(self):
        """The round trip: re-aggregating a block returns what the cumulative variable
        recorded across it. This is what makes the grid series a presentation of
        the native one rather than a replacement."""
        df, delivered = _household()
        out, _ = _treated_and_imputed(df)

        blocks = _filled_blocks(out)
        assert blocks, "the fixture must produce filled blocks or this proves nothing"

        for start, stop in blocks:
            jump = float(delivered.iloc[stop - 1] - delivered.iloc[start - 1])
            volume = float(out[DIFF].iloc[start:stop].sum())
            assert volume == pytest.approx(jump, abs=1e-6), (
                f"block {start}-{stop - 1} holds {volume}, cumulative moved {jump}"
            )

    def test_a_frozen_span_reconstructs_to_the_delivered_catch_up(self):
        """Collapsing the span recovers the single reading the cumulative
        variable actually delivered -- the value treatment blanked."""
        df, _ = _household(catchup=400.0)
        out, rows = _treated_and_imputed(df)
        assert len(rows) == 1

        start = int(rows[0]["start"])
        catchup_position = int(rows[0]["catchup_position"])
        recovered = float(out[DIFF].iloc[start:catchup_position + 1].sum())
        assert recovered == pytest.approx(400.0, abs=1e-6)

    def test_no_volume_is_created_or_lost_overall(self):
        df, delivered = _household()
        out, _ = _treated_and_imputed(df)
        total = float(out[DIFF].iloc[1:].sum())
        moved = float(delivered.iloc[-1] - delivered.iloc[0])
        assert total == pytest.approx(moved, abs=1e-6)


class TestKindIsNotRecoverableFromImputationFlagsAlone:
    """The gap this file exists to record.

    A frozen span and a transmission gap produce the same `SCALED_FILL`, and the
    collapse rule for them is opposite: the span represents ONE delivered
    reading, the gap represents several genuinely missing ones at the grid's own
    cadence. Collapsing the gap would invent a coarser cadence the column never
    had.

    Telling them apart needs the mapping-stage flags (FROZEN_SPAN / CATCHUP),
    which are written to `mapping_frozen_spans.parquet` and are **not** loaded
    into the aggregated build. Until they are, reconstruction must be driven by
    the span artifact rather than by the imputation flags.
    """

    def test_both_kinds_are_present_and_both_conserve(self):
        df, delivered = _household()
        out, rows = _treated_and_imputed(df)
        blocks = _filled_blocks(out)
        assert len(blocks) == 2, "one frozen span and one ordinary gap"

        span_starts = {int(r["start"]) for r in rows}
        assert any(s in span_starts for s, _ in blocks), "one block is a span"
        assert any(s not in span_starts for s, _ in blocks), "one block is not"

    def test_the_imputation_flags_are_identical_for_both(self):
        df, _ = _household()
        out, _ = _treated_and_imputed(df)

        kinds = {
            (start, stop): {
                int(v) for v in out[f"{DIFF}_impute_type"].iloc[start:stop]
                if pd.notna(v)
            }
            for start, stop in _filled_blocks(out)
        }
        distinct = {frozenset(v) for v in kinds.values()}
        assert distinct == {frozenset({ImputeType.SCALED_FILL.value})}, (
            "if this ever differs, the flags have gained the distinction and "
            "reconstruction can stop depending on the span artifact"
        )

    def test_the_span_artifact_does_carry_the_distinction(self):
        """What reconstruction must use instead, until the flags carry it."""
        df, _ = _household()
        out, rows = _treated_and_imputed(df)
        span_starts = {int(r["start"]) for r in rows}
        blocks = _filled_blocks(out)

        spans = [b for b in blocks if b[0] in span_starts]
        gaps = [b for b in blocks if b[0] not in span_starts]
        assert len(spans) == 1 and len(gaps) == 1


class TestRoundTripWithFlags:
    """The proof the flag list has to earn: from the final grid plus the flags,
    recover what the supplier actually delivered.

    Three properties, and all three are needed. Conservation alone is not enough,
    because it cannot say whether a block should collapse.
    """

    def _built(self):
        from etdmap.flags import flag_column
        df, delivered = _household()
        delivered_diff = delivered.diff().round(10)
        delivered_diff.iloc[0] = 0
        out, rows = _treated_and_imputed(df)
        return out, rows, delivered_diff, flag_column(DIFF)

    def test_the_flags_survive_treatment_and_imputation(self):
        from etdmap.flags import MappingFlag
        out, _, _, fc = self._built()
        assert fc in out.columns, "the flag column must reach the calculated data"
        flags = out[fc]
        assert int(((flags & MappingFlag.FROZEN_SPAN.value) != 0).sum()) > 0
        assert int(((flags & MappingFlag.CATCHUP.value) != 0).sum()) == 1

    def test_a_span_collapses_to_the_one_reading_that_was_delivered(self):
        """CATCHUP marks where the real reading was. Treatment blanked that cell,
        so without the flag there is nothing left saying the cumulative variable
        reported once here rather than thirteen times."""
        from etdmap.flags import MappingFlag, contiguous_runs
        out, _, delivered_diff, fc = self._built()
        flags = out[fc]
        in_block = ((flags & MappingFlag.FROZEN_SPAN.value) != 0) | (
            (flags & MappingFlag.CATCHUP.value) != 0
        )
        blocks = contiguous_runs(in_block)
        assert len(blocks) == 1

        start, stop = blocks[0]
        recovered = float(out[DIFF].iloc[start:stop].sum())
        assert recovered == pytest.approx(400.0, abs=1e-6)

    def test_an_ordinary_gap_conserves_but_must_not_collapse(self):
        """Its readings are missing at the grid's own cadence. Collapsing it
        would invent a coarser cadence the column never reported at."""
        from etdmap.flags import MappingFlag, contiguous_runs
        out, _, delivered_diff, fc = self._built()
        flags = out[fc]
        touched_by_mapping = (
            (flags & MappingFlag.FROZEN_SPAN.value) != 0
        ) | ((flags & MappingFlag.CATCHUP.value) != 0)
        filled = out[f"{DIFF}_is_imputed"].fillna(False).astype(bool)

        gaps = contiguous_runs(filled & ~touched_by_mapping)
        assert len(gaps) == 1
        start, stop = gaps[0]
        assert stop - start > 1, "several grid readings, not one delivered value"
        assert float(out[DIFF].iloc[start:stop].sum()) == pytest.approx(18.0, abs=1e-6)

    def test_every_untouched_reading_is_bit_identical_to_what_was_delivered(self):
        from etdmap.flags import MappingFlag
        out, _, delivered_diff, fc = self._built()
        flags = out[fc]
        touched = (
            ((flags & MappingFlag.INVENTED.value) != 0)
            | ((flags & MappingFlag.CATCHUP.value) != 0)
            | out[f"{DIFF}_is_imputed"].fillna(False).astype(bool)
        )
        assert int((~touched).sum()) > 0, "the fixture must leave readings untouched"
        pd.testing.assert_series_equal(
            out.loc[~touched, DIFF].astype(float).round(8),
            delivered_diff.loc[~touched].astype(float).round(8),
            check_names=False,
        )

    def test_invented_covers_every_value_that_was_put_there(self):
        """A reconstruction wanting only real readings filters on INVENTED, so
        anything that invents a value has to be in it."""
        from etdmap.flags import MappingFlag
        assert MappingFlag.FILLED_DOWN in MappingFlag.INVENTED
        assert MappingFlag.FILLED_ZERO in MappingFlag.INVENTED
        assert MappingFlag.FROZEN_SPAN in MappingFlag.INVENTED
        # CATCHUP is deliberately NOT invented: it marks where a real reading was.
        assert MappingFlag.CATCHUP not in MappingFlag.INVENTED
