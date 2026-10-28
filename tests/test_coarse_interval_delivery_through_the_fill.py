"""A coarsely delivered interval column, end to end through the fill.

The shape is a column delivered hourly on a five-minute grid with its register
absent and one tick dropped -- the delivery shape the spread exists for.

Built synthetically and small. Nothing is read from disk and no fixture is
committed: the shape is three facts -- a register that never arrived, readings
at one tick per hour, one of those ticks missing -- and a frame stating them
directly is cheaper and clearer than a generated dataset.

What the fill does with it, and the distinction worth holding on to: the spread
**redistributes** what was delivered, while the fill **adds** an estimate where
nothing was. So the column's total is NOT conserved end to end, and a test
asserting that it is would be wrong. The two properties that do hold are a
partition -- every cell is either spread or imputed -- and per-window
conservation on every window a delivered reading actually covers.
"""

import pandas as pd
import pytest
from etdmap import ids
from etdmap.mapping_helpers import cadence_to_rows, detect_column_cadence
from etdmap.testing import (
    PAIRS_EMPTY,
    datasets_frame,
    declare_synthetic_model,
    household_batch_of,
    sidecar_frame,
    synthetic_model,
)

from etdtransform.vectorized_impute import ImputeType, impute_and_normalize

pytestmark = pytest.mark.unit

#: The model these tests run against, declared here rather than read from the
#: metadata. What is under test is how a coarse delivery travels through the
#: fill, which needs a register, an interval complement and one bound -- not
#: any particular variable. The declared Max is the per-slot bound the cases
#: below are written around.
MODEL = synthetic_model(maximum=1.4375)
#: The register of the variable, and the interval column the supplier delivers.
CUM = MODEL.device
DIFF = MODEL.device_complement

GRID = 5
#: Grid slots one hourly reading accounts for.
K = 12
#: One day. Enough for a dropped tick to sit well inside it and for the trailing
#: slots after the last tick to be visible, which is where the fill's other work
#: shows up.
N = 288
#: Each delivered reading, in the column's own units. Below the per-slot bound
#: once spread (1.5), and above it unspread, so the cadence amendment is load
#: bearing rather than incidental.
READING = 0.6
#: The household every frame below belongs to.
HOUSEHOLD = 1


@pytest.fixture(autouse=True)
def _declared_model():
    with declare_synthetic_model(MODEL):
        yield


def _hourly_frame(reading):
    """One household batch on the grid, carrying *reading* on every hourly tick.

    Returns the frame and the row positions those ticks landed on.
    """
    # PAIRS_EMPTY keeps the register present and empty, which is the delivery
    # shape under test: with a usable register the fill would read its volume
    # from there instead of from the interval column.
    frame = datasets_frame(
        {household_batch_of(HOUSEHOLD): {DIFF: [pd.NA] * N}},
        start="2026-01-01",
        freq="5min",
        tz=None,
        pairs=PAIRS_EMPTY,
        # The project average is an input to the fill, not something this test
        # should derive: one slot's worth of a delivered reading.
        **{f"{DIFF}_avg": reading / K},
    )
    ticks = [
        position for position, stamp in enumerate(frame["ReadingDate"])
        if stamp.minute == 0
    ]
    frame.loc[ticks, DIFF] = reading
    return frame, ticks


def _delivered(drop_tick=5):
    """One household batch: register absent, interval delivered hourly, one tick gone.

    *drop_tick* indexes the hourly ticks, not the rows.
    """
    frame, ticks = _hourly_frame(READING)
    dropped = ticks[drop_tick]
    frame.loc[dropped, DIFF] = pd.NA
    return frame, ticks, dropped


def _cadence(frame):
    """Measured, not hand-written, so the delivery form resolves as it does in
    production: 60 minutes and a reading count on the interval column, and zero
    readings on the register that never arrived.

    Keyed on the household batch *frame* carries, as the sidecar is: the fill joins the
    table on `ids.HOUSEHOLD_BATCH`, so a table keyed on anything else reaches no row
    and the delivered form goes unrecorded.
    """
    batches = sorted(
        int(b) for b in frame[ids.HOUSEHOLD_BATCH].dropna().unique()
    )
    assert len(batches) == 1, "one dataset per frame here"
    return sidecar_frame(
        "cadence",
        cadence_to_rows(
            batches[0], detect_column_cadence(frame, grid_minutes=GRID)
        ),
    )


def _filled(drop_tick=5):
    frame, ticks, dropped = _delivered(drop_tick)
    delivered_total = float(frame[DIFF].sum())
    out, _, _ = impute_and_normalize(
        frame.copy(), [CUM], ids.PROJECT, cadence=_cadence(frame),
    )
    return out, ticks, dropped, delivered_total


class TestTheDeliveryShapeIsWhatTheFixtureHas:
    """The preconditions. If these drift, the rest of the file is testing
    something other than a coarse interval delivery."""

    def test_the_register_never_arrived(self):
        frame, _, _ = _delivered()
        assert frame[CUM].isna().all()

    def test_the_interval_reports_once_an_hour_with_one_tick_missing(self):
        frame, ticks, _ = _delivered()
        assert len(ticks) == N // K
        assert int(frame[DIFF].notna().sum()) == len(ticks) - 1

    def test_the_cadence_table_states_both_forms(self):
        frame, _, _ = _delivered()
        table = _cadence(frame).set_index("column")
        assert int(table.loc[DIFF, "cadence_minutes"]) == 60
        assert int(table.loc[DIFF, "n_readings"]) == N // K - 1
        assert int(table.loc[CUM, "n_readings"]) == 0


class TestEveryCellIsAccountedFor:
    """The structural property. A cell either holds a share of something
    delivered, or holds an estimate because nothing was -- never both, and never
    neither."""

    def test_spread_and_imputed_partition_the_column(self):
        out, _, _, _ = _filled()
        spread = (
            out[f"{DIFF}_impute_type"].fillna(0).astype("int64")
            & ImputeType.RESAMPLED_TO_GRID.value
        ) != 0
        imputed = out[f"{DIFF}_is_imputed"].fillna(False).astype(bool)

        assert not (spread & imputed).any(), "a cell cannot be both"
        assert (spread | imputed).all(), "a cell cannot be neither"

    def test_no_cell_is_left_empty(self):
        out, _, _, _ = _filled()
        assert int(out[DIFF].isna().sum()) == 0


class TestWhatTheSpreadConserves:
    """Per window, not overall. A delivered reading's share is placed on the
    slots it accounts for and nowhere else."""

    def test_a_covered_window_sums_to_its_delivered_reading(self):
        out, ticks, dropped, _ = _filled()
        # A tick with a full window behind it, well away from both edges and
        # from the dropped one.
        tick = next(
            t for t in ticks
            if t >= K and t != dropped and abs(t - dropped) > K
        )
        window = out[DIFF].iloc[tick - K + 1:tick + 1]
        assert float(window.sum()) == pytest.approx(READING, abs=1e-9)

    def test_each_slot_of_that_window_holds_one_slots_worth(self):
        out, ticks, dropped, _ = _filled()
        tick = next(
            t for t in ticks
            if t >= K and t != dropped and abs(t - dropped) > K
        )
        window = out[DIFF].iloc[tick - K + 1:tick + 1]
        for value in window:
            assert float(value) == pytest.approx(READING / K, abs=1e-9)

    def test_the_total_is_not_conserved_and_that_is_correct(self):
        """The assertion it would be tempting to write, and why it is wrong: the
        fill adds an estimate for the dropped tick and for the slots after the
        last one, which were never delivered. Volume rising is the fill doing
        its job, so what is checked is that every added unit sits in a cell
        marked imputed."""
        out, _, _, delivered_total = _filled()
        imputed = out[f"{DIFF}_is_imputed"].fillna(False).astype(bool)

        assert float(out[DIFF].sum()) > delivered_total
        assert float(out.loc[imputed, DIFF].sum()) > 0
        assert float(out.loc[~imputed, DIFF].sum()) <= delivered_total + 1e-9


class TestWhatTheFillHadToEstimate:
    """Exactly the slots no delivered reading covers."""

    def test_the_dropped_ticks_window_is_filled(self):
        out, _, dropped, _ = _filled()
        imputed = out[f"{DIFF}_is_imputed"].fillna(False).astype(bool)
        assert imputed.iloc[dropped - K + 1:dropped + 1].all(), (
            "the hour whose reading never arrived"
        )

    def test_the_slots_after_the_last_tick_are_filled(self):
        """A reading accounts for the slots up to and including its own, so the
        slots after the final tick have no reading behind them."""
        out, ticks, _, _ = _filled()
        imputed = out[f"{DIFF}_is_imputed"].fillna(False).astype(bool)
        assert imputed.iloc[ticks[-1] + 1:].all()

    def test_nothing_else_is_estimated(self):
        out, ticks, dropped, _ = _filled()
        imputed = out[f"{DIFF}_is_imputed"].fillna(False).astype(bool)
        expected = set(range(dropped - K + 1, dropped + 1)) | set(
            range(ticks[-1] + 1, N)
        )
        assert set(out.index[imputed]) == expected


class TestTheColumnIsJudgedPerSlot:
    """The cadence amendment has to reach the threshold pass, or a spread column
    is bounded as though each cell still held a whole hour."""

    def test_no_cell_exceeds_the_per_slot_bound(self):
        out, _, _, _ = _filled()
        # thresholds.csv gives this column 1.5 per five minutes. Unspread, a
        # 0.6 hourly reading sits under it too -- so this is the weaker half of
        # the check and the next test is the sharp one.
        assert bool((out[DIFF].dropna() <= 1.5).all())

    def test_a_reading_that_only_fits_once_spread_survives(self):
        """An hourly reading above the per-slot bound but below the hourly one.
        Spreading puts each slot under the bound; not spreading would have the
        threshold pass remove the lot."""
        over_a_slot = 1.5 * 2  # above the per-slot bound, below the hourly one
        frame, _ = _hourly_frame(over_a_slot)
        out, _, _ = impute_and_normalize(
            frame.copy(), [CUM], ids.PROJECT, cadence=_cadence(frame),
        )
        assert float(out[DIFF].max()) == pytest.approx(
            over_a_slot / K, abs=1e-9
        )
        assert bool((out[DIFF].dropna() <= 1.5).all())
