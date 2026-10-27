"""A chunk is a memory window, and the spread must not be able to see it.

Chunking batches by **household**: ``_chunk_ids_by_project`` returns household
ids and ``impute_mapped_households_sharded`` reads batches of ``chunk_size``
households. So a household's whole series is always in one chunk, and a spread
window -- which lives inside one household's series -- cannot straddle a chunk
boundary.

Two halves, both asserted rather than argued: the chunker never splits a
household, and a household's spread is identical alone and among peers.
"""

import pandas as pd
import pytest

from etdmap.mapping_helpers import detect_column_cadence
from etdtransform.aggregate import _chunk_ids_by_project
from etdtransform.vectorized_impute import ImputeType, impute_and_normalize

pytestmark = pytest.mark.unit

CUM = "WarmteproductieRuimteverwarming"
DIFF = f"{CUM}Diff"
GRID = 5
#: Grid slots one hourly reading accounts for.
K = 12
N = 288
READING = 0.6


class TestTheChunkerNeverSplitsAHousehold:
    """The structural half. If a chunk could hold part of a household, every
    behavioural claim below would be about the wrong thing."""

    def test_every_household_lands_in_exactly_one_chunk(self):
        ids = list(range(1, 26))
        project_of = {h: 1 if h <= 12 else 2 for h in ids}
        chunks = _chunk_ids_by_project(ids, project_of, chunk_size=5)

        seen = [h for chunk in chunks for h in chunk]
        assert sorted(seen) == ids
        assert len(seen) == len(set(seen)), "a household in two chunks"

    def test_a_chunk_is_household_ids_not_rows(self):
        """The unit the cap applies to. A row-based cap is what would let a
        series be divided."""
        ids = list(range(1, 8))
        chunks = _chunk_ids_by_project(ids, {h: 1 for h in ids}, chunk_size=3)
        assert [len(c) for c in chunks] == [3, 3, 1]
        for chunk in chunks:
            assert all(isinstance(h, int) for h in chunk)

    @pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 100])
    def test_the_partition_holds_at_every_size(self, chunk_size):
        ids = list(range(1, 13))
        project_of = {h: (h % 3) for h in ids}
        chunks = _chunk_ids_by_project(ids, project_of, chunk_size)
        seen = [h for chunk in chunks for h in chunk]
        assert sorted(seen) == ids
        assert len(seen) == len(set(seen))


def _household(household):
    """Register absent, the interval delivered hourly on a five-minute grid."""
    stamps = pd.date_range("2026-01-01", periods=N, freq=f"{GRID}min")
    values = pd.Series(pd.NA, index=range(N), dtype="Float64")
    for position, stamp in enumerate(stamps):
        if stamp.minute == 0:
            values.iloc[position] = READING
    frame = pd.DataFrame({
        "HuisIdBSV": pd.array([household] * N, dtype="Int64"),
        "HuisBatchIdBSV": pd.array([household] * N, dtype="Int64"),
        "ProjectIdBSV": pd.array([1] * N, dtype="Int64"),
        "ReadingDate": stamps,
        CUM: pd.array([pd.NA] * N, dtype="Float64"),
        DIFF: values.values,
    })
    frame[f"{DIFF}_avg"] = pd.array([READING / K] * N, dtype="Float64")
    return frame


def _cadence(frame):
    tables = []
    for household in sorted(int(h) for h in frame["HuisIdBSV"].unique()):
        own = frame[frame["HuisIdBSV"] == household]
        table = detect_column_cadence(own, grid_minutes=GRID)
        table.insert(
            0, "HuisBatchIdBSV",
            pd.array([household] * len(table), dtype="Int64"),
        )
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def _filled(frame, household):
    out, _stats, _reports = impute_and_normalize(
        frame.copy(), [CUM], "ProjectIdBSV", cadence=_cadence(frame),
    )
    return (
        out[out["HuisIdBSV"] == household]
        .sort_values("ReadingDate")
        .reset_index(drop=True)
    )


class TestAHouseholdsSpreadDoesNotDependOnItsCompany:
    """The behavioural half, which is what the chunk cap could only break by
    dividing a series."""

    def test_the_spread_values_are_identical_alone_and_among_peers(self):
        alone = _filled(_household(1), 1)
        together = _filled(
            pd.concat(
                [_household(1), _household(2), _household(3)],
                ignore_index=True,
            ),
            1,
        )
        pd.testing.assert_series_equal(alone[DIFF], together[DIFF])

    def test_the_findings_are_identical_too(self):
        """Values matching while the record of how they got there differs would
        be the same defect one step later."""
        alone = _filled(_household(1), 1)
        together = _filled(
            pd.concat([_household(1), _household(2)], ignore_index=True), 1
        )
        pd.testing.assert_series_equal(
            alone[f"{DIFF}_impute_type"], together[f"{DIFF}_impute_type"]
        )

    def test_the_spread_is_recorded_in_both(self):
        """``RESAMPLED_TO_GRID`` is what marks a cell as holding a share of a
        coarser reading, and it appeared in one test file before this one."""
        for frame, household in (
            (_household(1), 1),
            (pd.concat([_household(1), _household(2)], ignore_index=True), 1),
        ):
            out = _filled(frame, household)
            spread = (
                out[f"{DIFF}_impute_type"].fillna(0).astype("int64")
                & ImputeType.RESAMPLED_TO_GRID.value
            ) != 0
            assert int(spread.sum()) > 0

    def test_a_window_still_conserves_its_reading_among_peers(self):
        """The arithmetic the spread exists for, checked in the frame that holds
        several households, since that is the frame a chunk actually is."""
        together = _filled(
            pd.concat(
                [_household(1), _household(2), _household(3)],
                ignore_index=True,
            ),
            2,
        )
        # A window well inside the frame, so neither edge is in play.
        window = together[DIFF].iloc[144 - K + 1:144 + 1]
        assert float(window.sum()) == pytest.approx(READING, abs=1e-9)
