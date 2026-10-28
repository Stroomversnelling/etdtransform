"""avg_window_presence.parquet records who the presence gates judged.

The gates blank values without recording which household fell short in which
window; this artifact is that record, one row per project, household, column
and presence window, with the raw counts so every judgement can be re-derived.
"""

import ibis
import pandas as pd
from etdmap import ids
from etdmap.testing import (
    PAIRS_EMPTY,
    cadence_sidecar,
    household_batch_of,
    household_batches_frame,
)

from etdtransform.impute import calculate_average_diff_ibis, write_window_presence
from etdtransform.presence import window_index, window_presence_frame

COL = "ElektriciteitNetgebruikDiff"
CUM = "ElektriciteitNetgebruik"

#: What window_presence_frame derives from counts alone. A caller that does not
#: spread supplies only these.
ARTIFACT_COLUMNS = [
    "column", ids.PROJECT, ids.HOUSEHOLD_BATCH, "window_start",
    "window_minutes",
    "n_slots", "n_delivered", "eligible", "n_eligible", "n_contributing",
    "slot_minimum", "averageable",
]

#: Measurements only the spreading can make, carried through rather than
#: derived: how many windows it skipped as meter outages, and which of a
#: household's two possible sources supplied its volume.
SPREAD_MEASUREMENTS = ["n_outage_windows", "n_outage_slots", "source"]


def _frame(values_by_household_batch, project=1, freq_minutes=15):
    # The cumulative counterpart must carry no readings, so the averaging takes
    # each household batch's volume from the interval values stated here: it prefers the
    # cumulative column wherever that holds any value.
    return household_batches_frame(
        {
            household_batch: {COL: list(values)}
            for household_batch, values in values_by_household_batch.items()
        },
        start="2026-01-05",
        freq=f"{freq_minutes}min",
        project=project,
        tz=None,
        pairs=PAIRS_EMPTY,
    )


def _cadence_for(frame, minutes=15):
    """A cadence table for *frame*'s household batches, recorded under the cumulative
    name -- which is where cadence is recorded, and where the interval column
    resolves it from -- plus the ReadingDate row carrying the grid."""
    return pd.concat(
        [
            cadence_sidecar([CUM], minutes=minutes, household_batch_id=household_batch)
            for household_batch in sorted(frame[ids.HOUSEHOLD_BATCH].unique())
        ],
        ignore_index=True,
    )


def _counts(timestamps, window_minutes, n_slots, n_delivered):
    # The tallies are the only input the judgements are derived from, so each
    # test states the coverage it means to have judged rather than leaving it to
    # be counted off readings. One household batch per stated coverage, because a
    # judgement is per (project, household batch, window).
    frame = pd.concat(
        [
            household_batches_frame(
                {household_batch_of(position): {
                    "n_slots": [slots], "n_delivered": [delivered],
                }},
                start=stamp,
                tz=None,
                dtypes={"n_slots": "Int64", "n_delivered": "Int64"},
            )
            for position, (stamp, slots, delivered) in enumerate(
                zip(timestamps, n_slots, n_delivered), start=1
            )
        ],
        ignore_index=True,
    )
    frame["_window"] = window_index(frame[ids.READING_DATE], window_minutes)
    return frame


class TestTheAveragingReturnsTheFrame:
    """Through ``calculate_average_diff_ibis``, the engine that writes this
    artifact in the pipeline."""

    def _presence(self, frame, **kwargs):
        kwargs.setdefault("min_eligible", 1)
        result = calculate_average_diff_ibis(
            ibis.memtable(frame), ids.PROJECT, [COL],
            cadence=_cadence_for(frame), grid_minutes=15, **kwargs
        )
        return result[COL]["window_presence"].sort_values(
            [ids.HOUSEHOLD_BATCH, "window_start"]
        ).reset_index(drop=True)

    def test_counts_and_judgements_match_the_gates(self):
        # A 15-minute day is 96 slots. Household 1 covers it; household 2
        # delivers 10 slots and is ineligible.
        frame = _frame({
            household_batch_of(1): [1.0] * 96,
            household_batch_of(2): [1.0] * 10 + [pd.NA] * 86,
        })
        presence = self._presence(frame, min_eligible=10)

        # The engine spreads, so its artifact carries the measurements the
        # derived judgements cannot supply as well as the judgements.
        assert list(presence.columns) == ARTIFACT_COLUMNS + SPREAD_MEASUREMENTS
        assert len(presence) == 2
        assert presence["n_slots"].tolist() == [96, 96]
        assert presence["n_delivered"].tolist() == [96, 10]
        assert presence["eligible"].tolist() == [True, False]
        # One household covered the day, so the window has one eligible
        # household, it contributed, and a slot needs exactly it.
        assert presence["n_eligible"].tolist() == [1, 1]
        assert presence["n_contributing"].tolist() == [1, 1]
        assert presence["slot_minimum"].tolist() == [1, 1]
        # ...and the window is still not averageable, which is the fact the
        # counts alone do not carry: the fraction is satisfied by that one
        # household while the project floor is not.
        assert presence["averageable"].tolist() == [False, False]
        assert (presence["column"] == COL).all()
        assert presence["window_start"].tolist() == (
            [pd.Timestamp("2026-01-05")] * 2
        )
        # Every row says which source fed the household: it is decided for all
        # of them, so a null would mean the record was lost rather than that
        # there was nothing to record.
        assert set(presence["source"]) == {"interval"}
        assert int(presence["n_outage_windows"].sum()) == 0

    def test_a_household_spanning_two_windows_gets_a_row_per_window(self):
        frame = _frame({household_batch_of(1): [1.0] * 192})
        presence = self._presence(frame)
        assert presence["window_start"].tolist() == [
            pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06"),
        ]
        assert presence["n_slots"].tolist() == [96, 96]


class TestTheJudgementsAreDerivedFromTheCounts:
    def test_the_window_length_places_the_window_start(self):
        """The window grid is the column's own, so the same timestamps land
        in different windows under different lengths and the artifact must
        say which."""
        stamps = [pd.Timestamp("2026-01-06 12:00")] * 2

        daily = window_presence_frame(
            _counts(stamps, 1440, [96, 96], [96, 96]),
            COL, 1440, ids.PROJECT,
        )
        assert daily["window_start"].tolist() == (
            [pd.Timestamp("2026-01-06")] * 2
        )
        assert daily["window_minutes"].tolist() == [1440.0, 1440.0]

        two_daily = window_presence_frame(
            _counts(stamps, 2880, [192, 192], [192, 192]),
            COL, 2880, ids.PROJECT,
        )
        # The 2880-minute grid counts from the epoch, so its boundary is
        # 1970-01-01 plus a whole number of two-day steps.
        assert two_daily["window_start"].tolist() == (
            [pd.Timestamp("2026-01-05")] * 2
        )
        assert two_daily["window_minutes"].tolist() == [2880.0, 2880.0]

    def test_contributing_parts_from_eligible_when_coverage_is_not_demanded(self):
        """At the default minimum an eligible household has delivered most of
        the window, so the two tallies agree; the schema is not allowed to
        assume that."""
        stamps = [pd.Timestamp("2026-01-05")] * 2
        out = window_presence_frame(
            _counts(stamps, 1440, [96, 96], [96, 0]),
            COL, 1440, ids.PROJECT, min_coverage=0.0,
        )
        assert out["eligible"].tolist() == [True, True]
        assert out["n_eligible"].tolist() == [2, 2]
        assert out["n_contributing"].tolist() == [1, 1]

    def test_the_dtypes_are_nullable(self):
        out = window_presence_frame(
            _counts([pd.Timestamp("2026-01-05")], 1440, [96], [96]),
            COL, 1440, ids.PROJECT,
        )
        assert str(out["n_slots"].dtype) == "Int64"
        assert str(out["n_delivered"].dtype) == "Int64"
        assert str(out["n_eligible"].dtype) == "Int64"
        assert str(out["n_contributing"].dtype) == "Int64"
        assert str(out["slot_minimum"].dtype) == "Int64"
        assert str(out["eligible"].dtype) == "boolean"
        assert str(out["window_minutes"].dtype) == "Float64"


class TestTheWriter:
    def _presence(self):
        return window_presence_frame(
            _counts([pd.Timestamp("2026-01-05")] * 2, 1440, [96, 96], [96, 10]),
            COL, 1440, ids.PROJECT,
        )

    def test_it_writes_the_frames_it_was_given(self, tmp_path):
        avg_diff_dict = {
            COL: {"window_presence": self._presence()},
            "GasgebruikDiff": {"window_presence": None},
        }
        out = write_window_presence(avg_diff_dict, aggregate_folder_path=tmp_path)
        stored = pd.read_parquet(
            tmp_path / "avg_window_presence.parquet",
            dtype_backend="numpy_nullable",
        )
        assert list(stored.columns) == ARTIFACT_COLUMNS
        assert len(stored) == len(out) == 2
        assert stored["column"].unique().tolist() == [COL]

    def test_a_dict_without_presence_writes_nothing(self, tmp_path):
        out = write_window_presence(
            {COL: {"avg_diff": pd.DataFrame()}}, aggregate_folder_path=tmp_path
        )
        assert out.empty
        assert not (tmp_path / "avg_window_presence.parquet").exists()
