"""avg_window_presence.parquet records who the presence gates judged.

The gates blank values without recording which household fell short in which
window; this artifact is that record, one row per project, household, column
and presence window, with the raw counts so every judgement can be re-derived.
"""

import pandas as pd
import pytest

from etdtransform.impute import write_window_presence
from etdtransform.presence import (
    apply_presence_filter,
    window_index,
    window_presence_frame,
)

COL = "ElektriciteitNetgebruikDiff"

#: What window_presence_frame derives from counts alone, which is what the
#: pandas presence filter can supply.
ARTIFACT_COLUMNS = [
    "column", "ProjectIdBSV", "HuisIdBSV", "window_start", "window_minutes",
    "n_slots", "n_delivered", "eligible", "n_eligible", "n_contributing",
    "slot_minimum", "averageable",
]

#: Measurements only the spreading can make, carried through rather than
#: derived: how many windows it skipped as meter outages, and which of a
#: household's two possible sources supplied its volume.
SPREAD_MEASUREMENTS = ["n_outage_windows", "n_outage_slots", "source"]


def _frame(values_by_household, project=1, freq_minutes=15):
    frames = []
    for hid, vals in values_by_household.items():
        n = len(vals)
        frames.append(pd.DataFrame({
            "HuisIdBSV": pd.array([hid] * n, dtype="Int64"),
            "ProjectIdBSV": pd.array([project] * n, dtype="Int64"),
            "ReadingDate": pd.date_range(
                "2026-01-05", periods=n, freq=f"{freq_minutes}min"
            ),
            COL: pd.array(vals, dtype="Float64"),
        }))
    return pd.concat(frames, ignore_index=True)


def _counts(timestamps, window_minutes, n_slots, n_delivered):
    return pd.DataFrame({
        "ProjectIdBSV": pd.array([1] * len(timestamps), dtype="Int64"),
        "HuisIdBSV": pd.array(range(1, len(timestamps) + 1), dtype="Int64"),
        "_window": window_index(pd.Series(timestamps), window_minutes),
        "n_slots": pd.array(n_slots, dtype="Int64"),
        "n_delivered": pd.array(n_delivered, dtype="Int64"),
    })


class TestTheFilterReturnsTheFrame:
    def test_counts_and_judgements_match_the_gates(self):
        # A 15-minute day is 96 slots. Household 1 covers it; household 2
        # delivers 10 slots and is ineligible.
        frame = _frame({
            1: [1.0] * 96,
            2: [1.0] * 10 + [pd.NA] * 86,
        })
        _, _, presence = apply_presence_filter(
            frame, COL, "ProjectIdBSV", window_minutes=1440
        )

        assert list(presence.columns) == ARTIFACT_COLUMNS
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

    def test_a_household_spanning_two_windows_gets_a_row_per_window(self):
        frame = _frame({1: [1.0] * 192})
        _, _, presence = apply_presence_filter(
            frame, COL, "ProjectIdBSV", window_minutes=1440
        )
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
            COL, 1440, "ProjectIdBSV",
        )
        assert daily["window_start"].tolist() == (
            [pd.Timestamp("2026-01-06")] * 2
        )
        assert daily["window_minutes"].tolist() == [1440.0, 1440.0]

        two_daily = window_presence_frame(
            _counts(stamps, 2880, [192, 192], [192, 192]),
            COL, 2880, "ProjectIdBSV",
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
            COL, 1440, "ProjectIdBSV", min_coverage=0.0,
        )
        assert out["eligible"].tolist() == [True, True]
        assert out["n_eligible"].tolist() == [2, 2]
        assert out["n_contributing"].tolist() == [1, 1]

    def test_the_dtypes_are_nullable(self):
        out = window_presence_frame(
            _counts([pd.Timestamp("2026-01-05")], 1440, [96], [96]),
            COL, 1440, "ProjectIdBSV",
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
            COL, 1440, "ProjectIdBSV",
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


@pytest.mark.integration
@pytest.mark.pipeline
class TestBothAveragingPathsWriteIt:
    def test_the_pandas_and_ibis_runs_both_produced_the_artifact(
        self, pandas_pipeline, ibis_pipeline
    ):
        frames = {}
        for name, out_dir in (("pandas", pandas_pipeline), ("ibis", ibis_pipeline)):
            path = out_dir / "avg_window_presence.parquet"
            assert path.exists(), f"the {name} averaging wrote no window presence"
            frames[name] = pd.read_parquet(path, dtype_backend="numpy_nullable")
        for name, stored in frames.items():
            # The averaging spreads, so its artifact carries the measurements
            # the pandas filter cannot make as well as the derived judgements.
            assert list(stored.columns) == (
                ARTIFACT_COLUMNS + SPREAD_MEASUREMENTS
            ), name
            assert len(stored) > 0, name
            # Every row says which source fed the household: it is decided for
            # all of them, so a null would mean the record was lost, not that
            # there was nothing to record.
            assert int(stored["source"].notna().sum()) == len(stored), name
            assert set(stored["source"]) <= {"cumulative", "interval"}, name
