"""A findings column accumulates, so a second writer must OR rather than assign.

The imputation writes its impute-type column two ways, and both are correct:

- The fill routes **assign**. Their masks partition the gap cells -- on the sign
  and size of the gap jump, then on which end reading exists -- and
  ``setup_impute_columns`` initialises the column to all-NA immediately before,
  so each route is the first and only writer of its rows. Assigning a scalar
  there is cheaper than a read-modify-write on tens of millions of rows.
- ``apply_thresholds`` **OR-s**, because it runs after the fill and its rows
  already carry a route action.

So the rule is not "always OR": it is "OR when you are not the first writer".
What keeps the assignments safe is the exclusivity of the routes, and what keeps
the later pass safe is that it accumulates. These tests hold both ends, so a new
producer cannot quietly break either.
"""

import ast
from pathlib import Path

import pandas as pd
import pytest

import etdtransform.vectorized_impute as vectorized_impute
from conftest import cadence_table
from etdtransform.vectorized_impute import (
    ImputeType,
    apply_thresholds,
    impute_and_normalize,
)

pytestmark = pytest.mark.unit

# Assigning instead of OR-ing is the exception, so it is declared at the site.
# Enforced below, because a convention nobody checks is a convention nobody keeps.
EXCEPTION_MARKER = "first write"

CUM = "ElektriciteitNetgebruik"
DIFF = "ElektriciteitNetgebruikDiff"

# The actions the fill routes assign. Exactly one may land on a cell, which is
# what makes assigning instead of OR-ing safe. UNSCALED_AVERAGE is deliberately
# absent: the threshold pass adds it on top of a route, via the OR path.
FILL_ROUTES = (
    ImputeType.ZERO_FILL,
    ImputeType.UNIFORM_FILL,
    ImputeType.SCALED_FILL,
    ImputeType.HOUSEHOLD_SCALED_AVERAGE,
    ImputeType.GAP_NOT_FILLED,
)


def _frame(cumulative, diffs, averages):
    n = len(cumulative)
    return pd.DataFrame({
        "HuisIdBSV": pd.array([1] * n, dtype="Int64"),
        "HuisBatchIdBSV": pd.array([1] * n, dtype="Int64"),
        "ProjectIdBSV": pd.array([1] * n, dtype="Int64"),
        "ReadingDate": pd.date_range("2026-01-01", periods=n, freq="5min"),
        CUM: pd.array(cumulative, dtype="Float64"),
        DIFF: pd.array(diffs, dtype="Float64"),
        f"{DIFF}_avg": pd.array(averages, dtype="Float64"),
    })


def _routes_per_cell(filled):
    types = filled[f"{DIFF}_impute_type"].fillna(0).astype("Int64")
    counts = pd.Series(0, index=filled.index, dtype="Int64")
    for route in FILL_ROUTES:
        counts = counts + ((types & route.value) != 0).astype("Int64")
    return counts


def _flag_writes():
    """Every write to the impute-type column, classified.

    Yields (line number, kind) where kind is "or" when the write reads the
    column before writing it, "init" when it creates the column as a fresh
    all-NA Series, and "assign" when it replaces the integer outright.
    """
    source = Path(vectorized_impute.__file__).read_text(encoding="utf-8")
    lines = source.splitlines()

    def mentions(node, name):
        return any(
            isinstance(child, ast.Name) and child.id == name
            for child in ast.walk(node)
        )

    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not (
                isinstance(target, ast.Subscript)
                and mentions(target.slice, "impute_type_col")
            ):
                continue
            if mentions(node.value, "impute_type_col"):
                kind = "or"
            elif isinstance(node.value, ast.Call) and "Series" in ast.dump(
                node.value.func
            ):
                kind = "init"
            else:
                kind = "assign"
            yield node.lineno, kind, lines


class TestEveryAssignmentDeclaresItself:
    """A write that replaces the findings integer is only safe when it is the
    first one to touch those rows. That is not visible from the line itself, so
    it is stated there and checked here."""

    @pytest.fixture(scope="class")
    def writes(self):
        found = list(_flag_writes())
        assert found, "found no writes to parse -- the detection has gone stale"
        return found

    def test_the_file_still_holds_both_kinds(self, writes):
        """If one kind disappears the rule has been flattened, and this file's
        premise no longer matches the code."""
        kinds = {kind for _, kind, _ in writes}
        assert {"or", "assign"} <= kinds, kinds

    def test_every_assignment_carries_the_marker(self, writes):
        unmarked = []
        for lineno, kind, lines in writes:
            if kind != "assign":
                continue
            above = lines[max(0, lineno - 3) : lineno - 1]
            if not any(EXCEPTION_MARKER in line.lower() for line in above):
                unmarked.append(lineno)
        assert not unmarked, (
            f"lines {unmarked} replace the findings integer without saying why. "
            f"Either OR the bit in, or state directly above the write that it "
            f"is the first write and the routes are exclusive."
        )

    def test_an_or_write_does_not_claim_the_exception(self, writes):
        """The marker means "I checked that nothing is here yet". On a write that
        OR-s it is noise, and would mislead the next reader."""
        misleading = [
            lineno
            for lineno, kind, lines in writes
            if kind == "or"
            and any(
                EXCEPTION_MARKER in line.lower()
                for line in lines[max(0, lineno - 3) : lineno - 1]
            )
        ]
        assert not misleading, misleading


class TestTheFillRoutesAreExclusive:
    """Each route assigns, so two routes on one cell would mean one silently
    overwrote the other."""

    @pytest.mark.parametrize(
        ("label", "cumulative", "diffs", "averages"),
        [
            ("anchored, scalable", [100.0, pd.NA, pd.NA, 104.0], [0.0, pd.NA, pd.NA, pd.NA], [1.0] * 4),
            ("anchored, no shape", [100.0, pd.NA, pd.NA, 104.0], [0.0, pd.NA, pd.NA, pd.NA], [0.0] * 4),
            ("meter fell", [100.0, pd.NA, pd.NA, 90.0], [0.0, pd.NA, pd.NA, pd.NA], [1.0] * 4),
            ("total flat", [100.0, pd.NA, pd.NA, 100.0], [0.0, pd.NA, pd.NA, pd.NA], [1.0] * 4),
            ("no closing reading", [100.0, pd.NA, pd.NA, pd.NA], [0.0, pd.NA, pd.NA, pd.NA], [1.0] * 4),
            ("no closing, no average", [100.0, pd.NA, pd.NA, pd.NA], [0.0, pd.NA, pd.NA, pd.NA], [pd.NA] * 4),
        ],
    )
    def test_no_cell_carries_two_routes(self, label, cumulative, diffs, averages):
        frame = _frame(cumulative, diffs, averages)
        filled, _, _ = impute_and_normalize(
            frame, [CUM], "ProjectIdBSV", cadence=cadence_table(frame, [CUM])
        )
        counts = _routes_per_cell(filled)
        assert (counts <= 1).all(), (
            f"{label}: a cell carries {int(counts.max())} fill routes, so one "
            f"route's assignment overwrote another's"
        )

    def test_every_gap_cell_gets_a_route(self):
        """The other half of exclusivity: exactly one, not at most one."""
        frame = _frame(
            [100.0, pd.NA, pd.NA, 104.0], [0.0, pd.NA, pd.NA, pd.NA], [1.0] * 4
        )
        filled, _, _ = impute_and_normalize(
            frame, [CUM], "ProjectIdBSV", cadence=cadence_table(frame, [CUM])
        )
        in_gap = filled[CUM].isna() | filled.index.isin([1, 2, 3])
        assert (_routes_per_cell(filled)[in_gap] == 1).all()


class TestTheThresholdPassAccumulates:
    """It runs after the fill, so it must add to what the route recorded."""

    def _with_flag(self, preset):
        df = pd.DataFrame({
            "HuisIdBSV": pd.array([1, 1], dtype="Int64"),
            "HuisBatchIdBSV": pd.array([1, 1], dtype="Int64"),
            DIFF: pd.array([0.5, 99.0], dtype="Float64"),
            f"{DIFF}_avg": pd.array([0.5, 0.5], dtype="Float64"),
            f"{DIFF}_impute_type": pd.array([preset, preset], dtype="Int64"),
            f"{DIFF}_is_imputed": pd.array([True, True], dtype="boolean"),
        })
        return apply_thresholds(
            df, 0.0, 1.0, DIFF, f"{DIFF}_avg",
            f"{DIFF}_impute_type", f"{DIFF}_is_imputed",
        )

    def test_the_route_already_recorded_survives(self):
        out = self._with_flag(ImputeType.SCALED_FILL.value)
        types = out[f"{DIFF}_impute_type"].astype("Int64")
        assert bool(types.iloc[1] & ImputeType.SCALED_FILL.value), (
            "the threshold pass replaced the fill route instead of adding to it"
        )

    def test_its_own_flags_are_added_alongside(self):
        out = self._with_flag(ImputeType.SCALED_FILL.value)
        types = out[f"{DIFF}_impute_type"].astype("Int64")
        for flag in (
            ImputeType.THRESHOLD_VIOLATED,
            ImputeType.THRESHOLD_ADJUSTED,
            ImputeType.UNSCALED_AVERAGE,
        ):
            assert bool(types.iloc[1] & flag.value), flag.name

    def test_an_untouched_row_keeps_only_what_it_had(self):
        out = self._with_flag(ImputeType.SCALED_FILL.value)
        types = out[f"{DIFF}_impute_type"].astype("Int64")
        assert int(types.iloc[0]) == ImputeType.SCALED_FILL.value
