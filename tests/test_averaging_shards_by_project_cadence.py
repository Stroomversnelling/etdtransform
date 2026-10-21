"""Each project's average on the grid that project reports at.

The averages are sharded on (project, cadence), so within a pass there is one
grid and the spreading denominator and the step-back are that grid.

One grid shared by every project spreads a project reporting less often over
slots it has no rows for: the per-slot share is divided by the shared slot count
instead of the project's, and the placed slots falling between the project's
rows are dropped by the final join. A quarter-hourly project on a five-minute
grid publishes a third of its volume, at a third of the right level.
``grid_minutes`` forces that case on request -- ``TestWhatOneSharedGridDoes``.

Frames here are built in the test. Two projects, their declared grids and a
reading an hour is the whole of the case, and the cadence table is **measured**
by ``detect_column_cadence`` so the delivery form resolves the way it resolves
in production.
"""

import ibis
import pandas as pd
import pytest
from conftest import spread_average

from etdmap.mapping_helpers import detect_column_cadence
from etdtransform.bounds import grid_minutes_by_project
from etdtransform.flags import ImputeType
from etdtransform.impute import calculate_average_diff_ibis
from etdtransform.vectorized_impute import impute_and_normalize

pytestmark = pytest.mark.unit

CUM = "WarmteproductieRuimteverwarming"
DIFF = f"{CUM}Diff"

#: Project -> the grid its rows sit on. Fifteen is not what the other project
#: reports at, which is the whole point: neither may be spread over the other's
#: slots.
GRIDS = {1: 5, 2: 15}
#: The column reports once an hour whatever grid its project is on.
DELIVERED_EVERY = 60
#: Below the column's own bound scaled to an hour (0.0576 per five minutes
#: becomes 0.1995 at sixty), so what this file measures is the spreading rather
#: than the pre-averaging discard. Both projects deliver hourly, so both are
#: judged against the same bound.
READING = 0.12
#: Per project. Ten is the eligible-household count a project average needs, so
#: a thin fixture would be testing the presence gate instead of the spreading.
HOUSEHOLDS = 10
HOURS = 24


def _household(household, project, grid):
    """One household: register absent, the interval delivered once an hour."""
    periods = HOURS * 60 // grid
    stamps = pd.date_range("2026-01-01", periods=periods, freq=f"{grid}min")
    values = pd.Series(pd.NA, index=range(periods), dtype="Float64")
    for position, stamp in enumerate(stamps):
        if stamp.minute == 0:
            values.iloc[position] = READING
    return pd.DataFrame({
        "HuisIdBSV": pd.array([household] * periods, dtype="Int64"),
        "ProjectIdBSV": pd.array([project] * periods, dtype="Int64"),
        "ReadingDate": stamps,
        CUM: pd.array([pd.NA] * periods, dtype="Float64"),
        DIFF: values.values,
    })


def _estate():
    """The two projects, and the cadence table measured off their own rows."""
    frames = []
    cadences = []
    household = 0
    for project, grid in GRIDS.items():
        for _ in range(HOUSEHOLDS):
            household += 1
            frame = _household(household, project, grid)
            frames.append(frame)
            table = detect_column_cadence(frame, grid_minutes=grid)
            table.insert(
                0, "HuisIdBSV",
                pd.array([household] * len(table), dtype="Int64"),
            )
            cadences.append(table)
    return (
        pd.concat(frames, ignore_index=True),
        pd.concat(cadences, ignore_index=True),
    )


def _averages(frame, cadence, **kwargs):
    result = calculate_average_diff_ibis(
        ibis.memtable(frame), "ProjectIdBSV", [DIFF], cadence=cadence, **kwargs
    )
    return result[DIFF]


def _covering_window(avg, project, grid, ends_at):
    """The slots the hourly reading at *ends_at* accounts for, that project's.

    A reading covers the slots back to and including its own, so the window is
    one hour of the project's own slots -- twelve of them at five minutes, four
    at fifteen.
    """
    rows = avg[avg["ProjectIdBSV"] == project].sort_values("ReadingDate")
    start = ends_at - pd.Timedelta(minutes=DELIVERED_EVERY - grid)
    inside = rows[
        (rows["ReadingDate"] >= start) & (rows["ReadingDate"] <= ends_at)
    ]
    return inside[f"{DIFF}_avg"]


class TestTheFixtureHoldsTwoGrids:
    """If the grids do not resolve per project, nothing below is testing the
    per-project spreading."""

    def test_the_grids_resolve_from_the_measured_cadence(self):
        frame, cadence = _estate()
        membership = {
            int(h): int(p) for h, p in zip(
                frame["HuisIdBSV"], frame["ProjectIdBSV"]
            )
        }
        assert grid_minutes_by_project(cadence, membership) == {
            1: 5.0, 2: 15.0,
        }

    def test_each_project_reports_the_same_volume_an_hour(self):
        frame, _ = _estate()
        per_project = frame.groupby("ProjectIdBSV")[DIFF].sum()
        assert float(per_project.loc[1]) == pytest.approx(
            float(per_project.loc[2])
        )


class TestTheCoarseProjectKeepsItsVolume:
    """A quarter-hourly project's hour sums to its delivered reading, spread
    over the four slots the project reports at."""

    def test_the_quarter_hourly_project_sums_to_its_reading(self):
        frame, cadence = _estate()
        avg = _averages(frame, cadence)["avg_diff"]
        window = _covering_window(
            avg, 2, GRIDS[2], pd.Timestamp("2026-01-01 12:00")
        )
        assert int(window.notna().sum()) == DELIVERED_EVERY // GRIDS[2]
        assert float(window.sum()) == pytest.approx(READING, abs=1e-9)

    def test_the_five_minute_project_sums_to_its_reading_too(self):
        """The project whose grid is the finer of the two, as the control."""
        frame, cadence = _estate()
        avg = _averages(frame, cadence)["avg_diff"]
        window = _covering_window(
            avg, 1, GRIDS[1], pd.Timestamp("2026-01-01 12:00")
        )
        assert int(window.notna().sum()) == DELIVERED_EVERY // GRIDS[1]
        assert float(window.sum()) == pytest.approx(READING, abs=1e-9)

    @pytest.mark.parametrize("project", sorted(GRIDS))
    def test_a_slot_holds_one_project_cadence_interval(self, project):
        frame, cadence = _estate()
        avg = _averages(frame, cadence)["avg_diff"]
        window = _covering_window(
            avg, project, GRIDS[project], pd.Timestamp("2026-01-01 12:00")
        )
        expected = READING / (DELIVERED_EVERY / GRIDS[project])
        for value in window:
            assert float(value) == pytest.approx(expected, abs=1e-9)


class TestWhatOneSharedGridDoes:
    """``grid_minutes`` forces one grid on every project, which is the case the
    sharding exists to avoid. These are the numbers it produces.
    """

    def test_the_coarse_project_loses_two_thirds_of_its_volume(self):
        frame, cadence = _estate()
        avg = _averages(frame, cadence, grid_minutes=GRIDS[1])["avg_diff"]
        window = _covering_window(
            avg, 2, GRIDS[2], pd.Timestamp("2026-01-01 12:00")
        )
        # Four of the twelve placed slots fall on rows the project has; each
        # holds a twelfth of the reading rather than a quarter.
        assert float(window.sum()) == pytest.approx(READING / 3, abs=1e-9)
        for value in window:
            assert float(value) == pytest.approx(READING / 12, abs=1e-9)

    def test_the_project_on_the_shared_grid_is_unaffected(self):
        """The project whose own grid is the shared one reaches the same
        numbers either way, which is why the shared grid is not self-evidently
        wrong from one project's output."""
        frame, cadence = _estate()
        forced = _averages(frame, cadence, grid_minutes=GRIDS[1])["avg_diff"]
        sharded = _averages(frame, cadence)["avg_diff"]
        for avg in (forced, sharded):
            window = _covering_window(
                avg, 1, GRIDS[1], pd.Timestamp("2026-01-01 12:00")
            )
            assert float(window.sum()) == pytest.approx(READING, abs=1e-9)


class TestWhereEachProjectsAverageLands:
    def test_the_coarse_project_covers_its_own_slots_not_a_quarter_of_them(self):
        """Placement, as distinct from level: every slot the project reports at
        and has a reading behind carries an average. Spread at five minutes,
        only the slots that happened to coincide did."""
        frame, cadence = _estate()
        avg = _averages(frame, cadence)["avg_diff"]
        rows = avg[avg["ProjectIdBSV"] == 2]
        covered = int(rows[f"{DIFF}_avg"].notna().sum())

        slots_per_reading = DELIVERED_EVERY // GRIDS[2]
        # Every slot a reading covers, less the first reading's slots that fall
        # before the frame begins and the slots after the last reading, which
        # have no reading behind them.
        assert covered == 1 + (HOURS - 1) * slots_per_reading

    def test_neither_project_is_given_the_others_slots(self):
        frame, cadence = _estate()
        avg = _averages(frame, cadence)["avg_diff"]
        for project, grid in GRIDS.items():
            rows = avg[avg["ProjectIdBSV"] == project]
            offsets = set(
                int(m) % grid for m in rows["ReadingDate"].dt.minute
            )
            assert offsets == {0}, (
                f"project {project} has rows off its own {grid}-minute grid"
            )


class TestTheCadenceRidesWithTheAverage:
    """What the number means. An average is energy per interval, and without the
    interval beside it a quarter-hourly average reads as a five-minute one."""

    def test_each_project_carries_the_interval_it_was_spread_at(self):
        frame, cadence = _estate()
        avg = _averages(frame, cadence)["avg_diff"]
        recorded = (
            avg.groupby("ProjectIdBSV")["cadence_minutes"].unique().to_dict()
        )
        for project, grid in GRIDS.items():
            assert [int(m) for m in recorded[project]] == [grid]

    def test_the_counts_cover_both_passes(self):
        """A per-column figure has to add up over the passes, not report the
        last one."""
        frame, cadence = _estate()
        result = _averages(frame, cadence)
        assert result["coverage"] > 0.0
        assert set(result["coverage_duty"]["ProjectIdBSV"]) == set(GRIDS)
        assert set(result["window_presence"]["ProjectIdBSV"]) == set(GRIDS)


class TestAGridThatDividesNothing:
    """Seven minutes, against a five elsewhere. Once each project carries its
    own grid, nothing has to divide anything -- which is the property that makes
    adding a project at a new cadence safe rather than arithmetic to check."""

    def test_both_projects_spread_over_their_own_slots(self):
        value = 0.02
        rows = []
        # Project 1 on a five-minute grid, reporting every ten.
        stamps = pd.date_range("2026-01-01", periods=4, freq="5min")
        rows.append(pd.DataFrame({
            "HuisIdBSV": pd.array([1] * 4, dtype="Int64"),
            "ProjectIdBSV": pd.array([1] * 4, dtype="Int64"),
            "ReadingDate": stamps,
            DIFF: pd.array([pd.NA, pd.NA, value, pd.NA], dtype="Float64"),
        }))
        # Project 3 on a seven-minute grid, reporting every twenty-eight.
        stamps = pd.date_range("2026-01-01", periods=5, freq="7min")
        rows.append(pd.DataFrame({
            "HuisIdBSV": pd.array([2] * 5, dtype="Int64"),
            "ProjectIdBSV": pd.array([3] * 5, dtype="Int64"),
            "ReadingDate": stamps,
            DIFF: pd.array(
                [pd.NA, pd.NA, pd.NA, pd.NA, value], dtype="Float64"
            ),
        }))
        frame = pd.concat(rows, ignore_index=True)
        cadence = pd.DataFrame({
            "HuisIdBSV": pd.array([1, 2], dtype="Int64"),
            "cadence_minutes": pd.array([10, 28], dtype="Int64"),
        })

        avg, _coverage, _duty = spread_average(
            frame, cadence, DIFF, grid_minutes={1: 5, 3: 7},
        )

        five = avg[avg["ProjectIdBSV"] == 1][f"{DIFF}_avg"]
        seven = avg[avg["ProjectIdBSV"] == 3][f"{DIFF}_avg"]

        assert len(five) == 2, "ten minutes is two five-minute slots"
        assert [float(v) for v in five] == pytest.approx([value / 2] * 2)

        assert len(seven) == 4, "twenty-eight minutes is four seven-minute slots"
        assert [float(v) for v in seven] == pytest.approx([value / 4] * 4)
        assert float(seven.sum()) == pytest.approx(value, abs=1e-9)


class TestACoarseProjectsGapIsFilledWithItsOwnInterval:
    """End to end: the average this file is about, used by the fill.

    The average is derived rather than handed in, so the number under test is
    the one the averaging produces.

    A shared grid reaches the published average but not a filled cell on this
    route: ``HOUSEHOLD_SCALED_AVERAGE`` scales the average's *shape* to the
    household, and a uniformly wrong average cancels in the scale factor. What
    it does reach is coverage -- where a reading covers several of the project's
    slots, a shared grid places them at the wrong spacing and those slots get
    no average at all, which no scaling recovers. That is
    ``TestTheCoarseProjectKeepsItsVolume``.
    """

    #: Delivered at the project's own grid, so nothing about the *column*
    #: cadence is in play -- only the project's. Under the bound at fifteen
    #: minutes (0.0576 scaled by sqrt(3) is about 0.0998).
    AT_GRID = 0.05
    GRID = 15
    SLOTS = 40
    GAP = slice(20, 24)

    def _frame(self, households=10):
        parts = []
        for household in range(1, households + 1):
            stamps = pd.date_range(
                "2026-01-01", periods=self.SLOTS, freq=f"{self.GRID}min"
            )
            parts.append(pd.DataFrame({
                "HuisIdBSV": pd.array(
                    [household] * self.SLOTS, dtype="Int64"
                ),
                "ProjectIdBSV": pd.array([2] * self.SLOTS, dtype="Int64"),
                "ReadingDate": stamps,
                CUM: pd.array([pd.NA] * self.SLOTS, dtype="Float64"),
                DIFF: pd.array(
                    [self.AT_GRID] * self.SLOTS, dtype="Float64"
                ),
            }))
        return pd.concat(parts, ignore_index=True)

    def _cadence(self, frame):
        tables = []
        for household in sorted(int(h) for h in frame["HuisIdBSV"].unique()):
            own = frame[frame["HuisIdBSV"] == household]
            table = detect_column_cadence(own, grid_minutes=self.GRID)
            table.insert(
                0, "HuisIdBSV",
                pd.array([household] * len(table), dtype="Int64"),
            )
            tables.append(table)
        return pd.concat(tables, ignore_index=True)

    def _averaged_then_filled(self, **kwargs):
        frame = self._frame()
        cadence = self._cadence(frame)
        avg = _averages(frame, cadence, **kwargs)["avg_diff"]

        # One household loses a window; the rest of the project still delivers,
        # which is what gives the gap a shape to be filled from.
        gapped = frame.copy()
        first = gapped[gapped["HuisIdBSV"] == 1].index
        gapped.loc[first[self.GAP], DIFF] = pd.NA

        gapped = gapped.merge(
            avg[["ProjectIdBSV", "ReadingDate", f"{DIFF}_avg"]],
            on=["ProjectIdBSV", "ReadingDate"], how="left",
        )
        out, _stats, _reports = impute_and_normalize(
            gapped, [CUM], "ProjectIdBSV", cadence=cadence,
        )
        return out[out["HuisIdBSV"] == 1].reset_index(drop=True)

    def test_the_average_reaching_the_fill_is_a_quarter_hour(self):
        frame = self._frame()
        avg = _averages(frame, self._cadence(frame))["avg_diff"]
        delivered = avg[avg[f"{DIFF}_avg"].notna()][f"{DIFF}_avg"]
        assert len(delivered) > 0
        for value in delivered:
            assert float(value) == pytest.approx(self.AT_GRID, abs=1e-9)

    def test_the_filled_cells_hold_a_quarter_hour_each(self):
        filled = self._averaged_then_filled()
        window = filled[DIFF].iloc[self.GAP]
        assert int(window.notna().sum()) == self.GAP.stop - self.GAP.start
        for value in window:
            assert float(value) == pytest.approx(self.AT_GRID, abs=1e-9)

    def test_the_cells_are_marked_as_filled(self):
        filled = self._averaged_then_filled()
        imputed = filled[f"{DIFF}_is_imputed"].fillna(False).astype(bool)
        assert bool(imputed.iloc[self.GAP].all())

    def test_one_shared_grid_publishes_a_third_of_the_interval(self):
        """What the dataset-wide grid did to the artifact itself. A consumer
        reading an average of a quarter-hourly project got a third of the
        energy that interval carries."""
        frame = self._frame()
        avg = _averages(
            frame, self._cadence(frame), grid_minutes=5
        )["avg_diff"]
        published = avg[avg[f"{DIFF}_avg"].notna()][f"{DIFF}_avg"]
        assert len(published) > 0
        for value in published:
            assert float(value) == pytest.approx(self.AT_GRID / 3, abs=1e-9)

    def test_but_a_household_scaled_fill_cancels_it(self):
        """``HOUSEHOLD_SCALED_AVERAGE``: the average supplies a shape and the
        household's own level scales it, so a uniformly wrong average produces
        an exactly compensating factor and the filled cell comes out right from
        a published number that is wrong. What it cannot compensate for is a
        slot with no average at all.
        """
        filled = self._averaged_then_filled(grid_minutes=5)
        window = filled[DIFF].iloc[self.GAP]
        for value in window:
            assert float(value) == pytest.approx(self.AT_GRID, abs=1e-9)

        scaled = (
            filled[f"{DIFF}_impute_type"].iloc[self.GAP].fillna(0).astype("int64")
            & ImputeType.HOUSEHOLD_SCALED_AVERAGE.value
        ) != 0
        assert bool(scaled.all()), "the route the cancellation depends on"


class TestAMixedCadenceProjectIsRefused:
    """Averaging a project whose households sit on different grids is refused:
    the machinery to do it correctly has not been built, and a number nobody has
    checked is worse than a refusal."""

    def test_the_averaging_raises_naming_the_project_and_the_cadences(self):
        frames = []
        cadences = []
        # One project, two grids.
        for household, grid in ((1, 5), (2, 15)):
            frame = _household(household, 7, grid)
            frames.append(frame)
            table = detect_column_cadence(frame, grid_minutes=grid)
            table.insert(
                0, "HuisIdBSV",
                pd.array([household] * len(table), dtype="Int64"),
            )
            cadences.append(table)

        with pytest.raises(ValueError) as raised:
            _averages(
                pd.concat(frames, ignore_index=True),
                pd.concat(cadences, ignore_index=True),
            )

        message = str(raised.value)
        assert "Project 7 declares 2 household cadences (5, 15 minutes)" in message
        assert "cannot be calculated" in message
