"""The averaging records the judgements it made, not only their outcome.

Four numbers were computed to decide something and then dropped: how many
households a slot's average rests on and how many were eligible that window;
how many readings were thrown out as over the column bound or as written by the
pipeline; how many windows were skipped as meter outages; and which of a
household's two possible sources supplied its volume.

Each one is the difference between an average a reader can account for and a
number that is simply thin. A window thin because a meter was out and one thin
because nobody delivered are different facts about a project, and the surviving
average is identical in both cases.

Nothing here moves a value. Frames are built in the test and hold a few
households, which is below the production household floor, so they pass
``min_eligible=1`` -- the floor has its own tests in
``test_minimum_household_presence.py``.
"""

import pandas as pd
import pytest
from conftest import cadence_table, project_averages

from etdmap.flags import MappingFlag, flag_column
from etdmap.testing import declare_synthetic_model, synthetic_model
from etdtransform.impute import read_diffs, write_diffs

pytestmark = pytest.mark.unit

#: The model these tests run against, declared here rather than read from the
#: metadata. What is under test is what the averaging records, which needs a
#: cumulative variable, its interval complement and one bound -- not any
#: particular variable. Naming a real one pinned this file to whatever Max the
#: Grist-derived thresholds carried, and a metadata edit then discarded every
#: reading and emptied the averages.
MODEL = synthetic_model(maximum=0.0576)
CUM = MODEL.device
DIFF = MODEL.device_complement
GRID = 5
#: Under the bound this file declares, so a value placed above it in a test is
#: the only thing the discard can be about.
READING = 0.01
OVER_THE_BOUND = 10.0
N = 4


@pytest.fixture(autouse=True)
def _declared_model(monkeypatch):
    declare_synthetic_model(monkeypatch, MODEL)


def _frame(per_household, project_of=None, register=None):
    """``{household: [interval values]}`` on one five-minute grid.

    *project_of* maps a household to its project, defaulting to project 1.
    *register* maps a household to its cumulative readings, which is what makes
    the averaging take its volume from the register rather than the interval.
    """
    parts = []
    for household, values in per_household.items():
        stamps = pd.date_range("2026-01-01", periods=len(values), freq=f"{GRID}min")
        part = pd.DataFrame({
            "HuisIdBSV": pd.array([household] * len(values), dtype="Int64"),
            "ProjectIdBSV": pd.array(
                [(project_of or {}).get(household, 1)] * len(values),
                dtype="Int64",
            ),
            "ReadingDate": stamps,
            DIFF: pd.array(values, dtype="Float64"),
        })
        part[CUM] = pd.array(
            (register or {}).get(household, [pd.NA] * len(values)),
            dtype="Float64",
        )
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _averaged(frame, **kwargs):
    kwargs.setdefault("min_eligible", 1)
    return project_averages(
        frame, "ProjectIdBSV", [DIFF],
        cadence=cadence_table(frame, [CUM], minutes=GRID),
        grid_minutes=GRID, **kwargs,
    )[DIFF]


class TestWhatASlotsAverageRestsOn:
    """A4. The contributor count is per slot and depends on where each
    household's own readings fell, so the per-window presence artifact cannot
    reconstruct it."""

    def test_the_counts_come_out_with_the_average(self):
        result = _averaged(_frame({1: [READING] * N, 2: [READING] * N}))
        avg = result["avg_diff"]

        assert f"{DIFF}_contributors" in avg.columns
        assert f"{DIFF}_n_eligible" in avg.columns
        assert [int(v) for v in avg[f"{DIFF}_contributors"]] == [2] * N
        assert [int(v) for v in avg[f"{DIFF}_n_eligible"]] == [2] * N

    def test_a_slot_one_household_missed_says_so(self):
        """Forty slots, not four: a household missing one of four covers 0.75
        of its window and is dropped from the whole day by the coverage gate,
        which would make this a test of that gate. At forty it covers 0.975,
        stays eligible, and only the slot it missed is thin -- which is the
        distinction the two counts exist to carry."""
        wide = 40
        values = [READING] * wide
        thin = list(values)
        thin[2] = pd.NA
        avg = _averaged(_frame({1: values, 2: thin}))["avg_diff"]

        contributors = [int(v) for v in avg[f"{DIFF}_contributors"]]
        assert contributors[2] == 1, "the slot the second household missed"
        assert contributors[:2] + contributors[3:] == [2] * (wide - 1)
        assert [int(v) for v in avg[f"{DIFF}_n_eligible"]] == [2] * wide

    def test_the_counts_are_nullable_integers(self):
        avg = _averaged(_frame({1: [READING] * N}))["avg_diff"]
        for name in (f"{DIFF}_contributors", f"{DIFF}_n_eligible"):
            assert str(avg[name].dtype) == "Int64"

    def test_none_of_them_reads_as_all_missing(self):
        """The failure this guards is a column declared and never populated,
        which reads as a persisted figure and carries nothing."""
        avg = _averaged(_frame({1: [READING] * N, 2: [READING] * N}))["avg_diff"]
        for name in (f"{DIFF}_contributors", f"{DIFF}_n_eligible"):
            assert int(avg[name].notna().sum()) == len(avg), name


class TestTheFillIsNotGivenThem:
    """They are persisted for a reader, not for the merge: the fill joins the
    averages onto a chunk and every other column in the frame would ride into
    the written shards."""

    def test_the_reader_hides_them_by_default(self, tmp_path):
        avg = _averaged(_frame({1: [READING] * N}))["avg_diff"].copy()
        avg["cadence_minutes"] = pd.array([GRID] * len(avg), dtype="Int64")
        write_diffs(avg, aggregate_folder_path=tmp_path)

        plain = read_diffs(tmp_path)
        # A set, not a list: a partitioned read returns the partition columns
        # after the stored ones, and the column order of this frame is not
        # something anything depends on.
        assert set(plain.columns) == {
            "ProjectIdBSV", "ReadingDate", f"{DIFF}_avg",
        }

    def test_they_can_be_asked_for(self, tmp_path):
        avg = _averaged(_frame({1: [READING] * N}))["avg_diff"].copy()
        avg["cadence_minutes"] = pd.array([GRID] * len(avg), dtype="Int64")
        write_diffs(avg, aggregate_folder_path=tmp_path)

        full = read_diffs(tmp_path, with_counts=True)
        assert f"{DIFF}_contributors" in full.columns
        assert f"{DIFF}_n_eligible" in full.columns


class TestWhatTheAveragingThrewOut:
    """A5. Per project, beside the coverage it produced: a dataset-wide total
    cannot say which project lost readings."""

    def test_a_reading_over_the_bound_is_counted_against_its_project(self):
        values = [READING] * N
        spiking = list(values)
        spiking[1] = OVER_THE_BOUND
        frame = _frame(
            {1: values, 2: spiking}, project_of={1: 1, 2: 2},
        )
        duty = _averaged(frame)["coverage_duty"].set_index("ProjectIdBSV")

        assert int(duty.loc[1, "n_discarded"]) == 0
        assert int(duty.loc[2, "n_discarded"]) == 1

    def test_a_value_the_pipeline_wrote_is_counted_too(self):
        frame = _frame({1: [READING] * N}, project_of={1: 3})
        findings = flag_column(DIFF)
        frame[findings] = pd.array([0] * N, dtype="Int64")
        frame.loc[2, findings] = int(MappingFlag.FILLED_DOWN)

        duty = _averaged(frame)["coverage_duty"].set_index("ProjectIdBSV")
        assert int(duty.loc[3, "n_flagged"]) == 1

    def test_the_counts_add_up_to_the_totals_the_caller_reports(self):
        values = [READING] * N
        spiking = list(values)
        spiking[1] = OVER_THE_BOUND
        frame = _frame(
            {1: spiking, 2: spiking}, project_of={1: 1, 2: 2},
        )
        result = _averaged(frame)
        assert int(result["coverage_duty"]["n_discarded"].sum()) == (
            result["n_discarded"]
        )
        assert result["n_discarded"] == 2

    def test_nothing_removed_reads_as_zero_not_missing(self):
        """At this grain a zero is measured: the averaging looked at every row
        of every project, so "no readings removed" is a finding."""
        duty = _averaged(_frame({1: [READING] * N}))["coverage_duty"]
        for name in ("n_discarded", "n_flagged"):
            assert int(duty[name].notna().sum()) == len(duty), name
            assert int(duty[name].sum()) == 0


class TestWindowsSkippedAsOutages:
    """A6. A cumulative window longer than one reporting interval spans a meter
    outage and is not spread. The filter is right; its silence was not."""

    def test_an_outage_window_is_recorded(self):
        # The register reports at rows 0, 1 and 5. The window closing at row 5
        # is four rows long against a one-slot reporting interval, so it is an
        # outage and is not spread.
        register = [0.0, 1.0, pd.NA, pd.NA, pd.NA, 5.0]
        frame = _frame(
            {1: [READING] * 6}, register={1: register},
        )
        presence = _averaged(frame)["window_presence"]

        assert "n_outage_windows" in presence.columns
        assert int(presence["n_outage_windows"].sum()) == 1
        assert int(presence["n_outage_slots"].sum()) == 4

    def test_a_household_with_no_outage_reads_zero(self):
        register = [0.0, 1.0, 2.0, 3.0]
        frame = _frame({1: [READING] * N}, register={1: register})
        presence = _averaged(frame)["window_presence"]

        assert int(presence["n_outage_windows"].sum()) == 0
        assert int(presence["n_outage_windows"].notna().sum()) == len(presence)

    def test_an_interval_delivered_household_cannot_have_one(self):
        """Its window is its own cadence however long ago the last reading
        arrived, so there is no span for an outage to sit in."""
        frame = _frame({1: [READING, pd.NA, pd.NA, READING]})
        presence = _averaged(frame)["window_presence"]
        assert int(presence["n_outage_windows"].sum()) == 0


class TestWhichSourceFedEachHousehold:
    """A7. Decided per household per column, used, and until now forgotten."""

    def test_a_register_household_reads_cumulative(self):
        frame = _frame(
            {1: [READING] * N}, register={1: [0.0, 1.0, 2.0, 3.0]},
        )
        presence = _averaged(frame)["window_presence"].set_index("HuisIdBSV")
        assert presence.loc[1, "source"] == "cumulative"

    def test_an_interval_only_household_reads_interval(self):
        presence = _averaged(_frame({1: [READING] * N}))["window_presence"]
        assert set(presence["source"]) == {"interval"}

    def test_two_households_on_different_sources_are_told_apart(self):
        frame = _frame(
            {1: [READING] * N, 2: [READING] * N},
            register={1: [0.0, 1.0, 2.0, 3.0]},
        )
        presence = _averaged(frame)["window_presence"].set_index("HuisIdBSV")
        assert presence.loc[1, "source"] == "cumulative"
        assert presence.loc[2, "source"] == "interval"

    def test_it_is_never_left_unstated(self):
        frame = _frame(
            {1: [READING] * N, 2: [READING] * N},
            register={1: [0.0, 1.0, 2.0, 3.0]},
        )
        presence = _averaged(frame)["window_presence"]
        assert int(presence["source"].notna().sum()) == len(presence)
