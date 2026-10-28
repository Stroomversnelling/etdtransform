"""One cadence per project, or a refusal that says which project.

A project average is a shape placed on the slots a project reports at, so it
needs one interval per project. An estate-wide figure cannot supply it: it is
the finest grid any household batch declares, so a project reporting quarter-hourly would
be spread across five-minute steps because some other project reports that
often.

``grid_minutes_by_project`` resolves that interval per project and raises where
a project's own households disagree, which is the one case where no answer
exists -- picking a grid would either invent slots the project never reported at
or discard volume.

Frames here are a handful of timestamps each. The cadence table is **measured**
by ``detect_column_cadence`` rather than hand-written, so the grid resolves the
way it resolves in production; nothing is read from disk and no fixture is
committed.
"""

import pandas as pd
import pytest

from etdmap.mapping_helpers import detect_column_cadence
from etdmap import ids
from etdtransform.bounds import (
    grid_minutes_by_household_batch,
    grid_minutes_by_project,
)

pytestmark = pytest.mark.unit

COLUMN = "WarmteproductieRuimteverwarmingDiff"


def _household_rows(household, minutes, periods=12):
    """One household's cadence rows, at *minutes* between readings."""
    stamps = pd.date_range("2026-01-01", periods=periods, freq=f"{minutes}min")
    frame = pd.DataFrame({
        "ReadingDate": stamps,
        COLUMN: pd.array([1.0] * periods, dtype="Float64"),
    })
    table = detect_column_cadence(frame, grid_minutes=minutes)
    table.insert(
        0, ids.HOUSEHOLD_BATCH, pd.array([household] * len(table), dtype="Int64")
    )
    return table


def _cadence(*households):
    """*households* is (household id, minutes) pairs."""
    return pd.concat(
        [_household_rows(h, m) for h, m in households], ignore_index=True
    )


class TestTheFixtureMeasuresWhatItClaims:
    """If the grid does not resolve per household, the rest of this file is
    testing the resolver against a table that never had the information."""

    def test_the_grid_is_read_per_household(self):
        cadence = _cadence((1, 5), (2, 15), (3, 60))
        assert grid_minutes_by_household_batch(cadence) == {
            1: 5.0, 2: 15.0, 3: 60.0,
        }


class TestEachProjectResolvesItsOwnGrid:
    def test_two_projects_at_different_cadences(self):
        cadence = _cadence((1, 5), (2, 5), (3, 15), (4, 15))
        resolved = grid_minutes_by_project(
            cadence, {1: 10, 2: 10, 3: 20, 4: 20}
        )
        assert resolved == {10: 5.0, 20: 15.0}

    def test_a_cadence_that_divides_nothing_resolves_too(self):
        """Seven minutes against a five elsewhere. Nothing has to divide
        anything once each project carries its own interval, so a cadence that
        is not a multiple of any other is not a special case."""
        cadence = _cadence((1, 7), (2, 5))
        assert grid_minutes_by_project(cadence, {1: 10, 2: 20}) == {
            10: 7.0, 20: 5.0,
        }

    def test_one_project_at_one_cadence(self):
        cadence = _cadence((1, 15), (2, 15), (3, 15))
        assert grid_minutes_by_project(cadence, {1: 4, 2: 4, 3: 4}) == {
            4: 15.0,
        }


class TestAProjectWithoutOneCadenceIsRefused:
    """The A3 case. A mixed-cadence project is refused because the machinery to
    average one correctly has not been built: there is no checked rule for which
    grid the project's average sits on."""

    def test_it_raises_naming_the_project_and_the_cadences(self):
        cadence = _cadence((1, 5), (2, 15))
        with pytest.raises(ValueError) as raised:
            grid_minutes_by_project(cadence, {1: 7, 2: 7})

        assert (
            "Project 7 declares 2 household cadences (5, 15 minutes). "
            "A project average requires one cadence per project, so it cannot "
            "be calculated."
        ) in str(raised.value)

    def test_it_names_the_households_on_each_cadence(self):
        """Which households disagree is the question the reader has next, and
        the resolver is the only place that still knows."""
        cadence = _cadence((11, 5), (12, 5), (13, 60))
        with pytest.raises(ValueError) as raised:
            grid_minutes_by_project(cadence, {11: 3, 12: 3, 13: 3})

        message = str(raised.value)
        assert "5 min: [11, 12]" in message
        assert "60 min: [13]" in message

    def test_three_cadences_are_all_listed_in_order(self):
        cadence = _cadence((1, 60), (2, 5), (3, 15))
        with pytest.raises(ValueError) as raised:
            grid_minutes_by_project(cadence, {1: 9, 2: 9, 3: 9})

        assert "declares 3 household cadences (5, 15, 60 minutes)" in str(
            raised.value
        )

    def test_a_disagreement_in_one_project_refuses_the_whole_call(self):
        """Returning the projects that do resolve and dropping the one that does
        not would publish averages for some projects and silently none for
        another, which reads as a project with no data."""
        cadence = _cadence((1, 5), (2, 5), (3, 15))
        with pytest.raises(ValueError, match="Project 2"):
            grid_minutes_by_project(cadence, {1: 1, 2: 2, 3: 2})


class TestWhatTheResolverCannotSee:
    def test_no_cadence_table_resolves_nothing(self):
        assert grid_minutes_by_project(None, {1: 1}) == {}
        assert grid_minutes_by_project(pd.DataFrame(), {1: 1}) == {}

    def test_a_household_outside_the_mapping_is_not_attributed(self):
        """Its grid is real but belongs to no project the caller named, so
        attributing it would put a cadence on a project by guesswork."""
        cadence = _cadence((1, 5), (2, 15))
        assert grid_minutes_by_project(cadence, {1: 10}) == {10: 5.0}

    def test_a_project_whose_households_resolve_no_grid_is_omitted(self):
        """Omitted rather than defaulted: a project with no readable grid has no
        interval to spread onto, and a fallback would be invented."""
        cadence = _cadence((1, 5), (2, 15))
        cadence.loc[cadence[ids.HOUSEHOLD_BATCH] == 2, "cadence_minutes"] = pd.NA

        resolved = grid_minutes_by_project(cadence, {1: 10, 2: 20})
        assert resolved == {10: 5.0}
        assert 20 not in resolved
