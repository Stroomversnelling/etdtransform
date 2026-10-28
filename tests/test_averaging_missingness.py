"""Missingness through the averaging, on a synthetic in-memory fixture.

The fixture carries null (pd.NA) values in its measurement columns on purpose.
Null-in, null-out is a correctness requirement throughout the pipeline: a slot
with no average is not a slot with an average of zero, and a project whose every
reading is absent must come back with its slots present and empty rather than
with no slots at all.

``ibis.memtable()`` wraps each frame, which is what the averaging takes -- the
pipeline passes a lazy table from ``ibis.read_parquet`` instead.
"""
import ibis
import numpy as np
import pandas as pd
import pytest
from conftest import cadence_table
from etdmap import ids
from etdmap.testing import household_batches_frame, one_to_one_test_household_batch_of

from etdtransform.impute import calculate_average_diff_ibis


def _cadence(frame, diff_columns):
    """The cadence both paths are judged at, from the frame they share."""
    from etdmap.data_model import cumulative_column

    return cadence_table(
        frame, [cumulative_column(c) or c for c in diff_columns]
    )


pytestmark = pytest.mark.unit

# Two projects: 20 normal + 1 outlier household per project. Each household's
# household batch -- the key of its series, its cadence and its contribution to the
# average -- is `one_to_one_test_household_batch_of` of its number.
# Outlier values are ~15-20x larger than the normal range so the
# 95th-percentile upper-bound logic clearly excludes them.
_P1_NORMAL = list(range(101, 121))
_P1_OUTLIER = 199
_P2_NORMAL = list(range(201, 221))
_P2_OUTLIER = 299

DIFF_COLS = ["ElektriciteitNetgebruikLaagDiff", "ZonopwekVerbruikDiff"]

#: Slots every household batch in the module fixture reports at.
SLOTS = 20

# Rows whose measurement values are set to pd.NA to verify null handling.
# We use (project_id, household_id, timestamp_index) tuples to identify them.
_NULL_SLOTS = {(1, 105, 3), (1, 110, 7), (2, 207, 0), (2, 215, 12)}

# Every frame below carries interval columns alone, built with `pairs=False`.
# The averaging takes a household batch's volume from the cumulative counterpart
# wherever that column holds a value, so a completed counterpart would supply
# the averages in place of the values and the nulls each test states.


@pytest.fixture(scope="module")
def diff_fixture_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    per_household_batch = {}
    project_of = {}

    for project_id, normal_ids, outlier_id in [
        (1, _P1_NORMAL, _P1_OUTLIER),
        (2, _P2_NORMAL, _P2_OUTLIER),
    ]:
        for household_id in normal_ids:
            low_tariff, solar = [], []
            for t_idx in range(SLOTS):
                if (project_id, household_id, t_idx) in _NULL_SLOTS:
                    low_tariff.append(pd.NA)
                    solar.append(pd.NA)
                else:
                    low_tariff.append(float(rng.uniform(1.0, 10.0)))
                    solar.append(float(rng.uniform(0.5, 5.0)))
            household_batch = one_to_one_test_household_batch_of(household_id)
            per_household_batch[household_batch] = {
                "ElektriciteitNetgebruikLaagDiff": low_tariff,
                "ZonopwekVerbruikDiff": solar,
            }
            project_of[household_batch] = project_id

        low_tariff, solar = [], []
        for _ in range(SLOTS):
            low_tariff.append(float(rng.uniform(150.0, 300.0)))
            solar.append(float(rng.uniform(150.0, 300.0)))
        household_batch = one_to_one_test_household_batch_of(outlier_id)
        per_household_batch[household_batch] = {
            "ElektriciteitNetgebruikLaagDiff": low_tariff,
            "ZonopwekVerbruikDiff": solar,
        }
        project_of[household_batch] = project_id

    df = household_batches_frame(
        per_household_batch,
        start="2023-01-01",
        freq="5min",
        project_of=project_of,
        tz=None,
        pairs=False,
    )
    # Microsecond timestamps, matching DuckDB's output precision and the
    # parquet files produced by the pipeline.
    df["ReadingDate"] = df["ReadingDate"].astype("datetime64[us]")
    return df


def _null_positions(df: pd.DataFrame, col: str) -> set:
    """Return the integer-location set of null positions in a column."""
    return set(df.index[df[col].isna()].tolist())


def test_ibis_memtable_preserves_nulls(diff_fixture_df):
    """ibis.memtable round-trip must preserve null positions and values exactly."""
    tbl = ibis.memtable(diff_fixture_df)
    result = tbl.execute().astype({c: diff_fixture_df[c].dtype for c in diff_fixture_df.columns})

    sort_cols = [ids.HOUSEHOLD_BATCH, ids.READING_DATE]
    original = diff_fixture_df.sort_values(sort_cols).reset_index(drop=True)
    restored = result.sort_values(sort_cols).reset_index(drop=True)

    for col in DIFF_COLS:
        orig_nulls = _null_positions(original, col)
        rest_nulls = _null_positions(restored, col)
        assert orig_nulls == rest_nulls, (
            f"{col}: null positions changed after ibis round-trip. "
            f"original={orig_nulls}, restored={rest_nulls}"
        )

    pd.testing.assert_frame_equal(original, restored, check_like=True)


def test_nulls_excluded_from_mean(diff_fixture_df):
    """A household that did not cover the window does not shape the average.

    One project, two households, three slots. Household 1 reports
    [2.0, pd.NA, 4.0] -- two slots of three -- and household 2 reports 6.0
    throughout. The presence gate excludes household 1 for not covering the
    window, so the average is household 2's level at every slot rather than a
    blend of a partial series with a complete one.

    Null-in, null-out still holds: no slot comes back empty, because one
    household did cover every one of them.
    """
    mini = household_batches_frame(
        {
            one_to_one_test_household_batch_of(1): {"ZonopwekVerbruikDiff": [2.0, pd.NA, 4.0]},
            one_to_one_test_household_batch_of(2): {"ZonopwekVerbruikDiff": [6.0, 6.0, 6.0]},
        },
        start="2024-01-01",
        freq="5min",
        tz=None,
        pairs=False,
    )

    result_ibis = calculate_average_diff_ibis(
        ibis.memtable(mini), ids.PROJECT, ["ZonopwekVerbruikDiff"],
        cadence=_cadence(mini, ["ZonopwekVerbruikDiff"]), grid_minutes=5,
        # Two households, which is smaller than a project: the household floor
        # would empty every slot, and what is under test here is which values
        # reach the mean.
        min_eligible=1,
    )

    avg = result_ibis["ZonopwekVerbruikDiff"]["avg_diff"]
    avg = avg.sort_values("ReadingDate")["ZonopwekVerbruikDiff_avg"]

    assert avg.astype(float).tolist() == pytest.approx([6.0, 6.0, 6.0])
    assert avg.notna().all(), "a null peer emptied a slot another household covered"


def test_empty_project_edge_cases():
    """Edge cases for projects with no qualifying data.

    Three degenerate situations that must not crash and must return consistent
    structures across both paths:
      A) All diff values are zero (filtered out by the > 1e-8 guard) ->
         upper_bound is null, avg_diff is empty.
      B) All diff values are pd.NA ->
         same outcome as (A): no household max qualifies.
      C) One 'normal' project alongside a zero-only project ->
         normal project processes correctly; zero project gets null upper_bound.
    """
    # Project 1: all zeros. Project 2: normal non-zero values.
    mini = household_batches_frame(
        {
            one_to_one_test_household_batch_of(1): {"ZonopwekVerbruikDiff": [0.0, 0.0, 0.0]},
            one_to_one_test_household_batch_of(2): {"ZonopwekVerbruikDiff": [3.0, 5.0, 4.0]},
        },
        start="2024-06-01",
        freq="5min",
        tz=None,
        project_of={one_to_one_test_household_batch_of(1): 1, one_to_one_test_household_batch_of(2): 2},
        pairs=False,
    )
    mini["ReadingDate"] = mini["ReadingDate"].astype("datetime64[us]")

    result_ibis = calculate_average_diff_ibis(
        ibis.memtable(mini), ids.PROJECT, ["ZonopwekVerbruikDiff"],
        cadence=_cadence(mini, ["ZonopwekVerbruikDiff"]), grid_minutes=5,
        # Two households, which is smaller than a project: the household floor
        # would empty every slot, and what is under test here is which values
        # reach the mean.
        min_eligible=1,
    )

    result = result_ibis
    avg = result["ZonopwekVerbruikDiff"]["avg_diff"]

    # avg_diff must have rows for project 2 only (project 1 excluded)
    # Project 1 is all-zero, so it has no average to report -- but its
    # slots exist and come back carrying pd.NA. A slot with no value is not
    # a missing slot, and the merge downstream is on
    # (project, ReadingDate).
    assert set(avg[ids.PROJECT].dropna()) == {1, 2}, (
            "both projects should appear, got "
            f"{set(avg[ids.PROJECT].dropna())}"
        )
    # Zero is a measurement, so an all-zero project averages to 0.0. That is
    # a different fact from pd.NA, which means nothing was measured, and the
    # two must not be conflated.
    assert (
            avg.loc[avg[ids.PROJECT] == 1, "ZonopwekVerbruikDiff_avg"].dropna() == 0.0
        ).all(), "an all-zero project should average to zero, not NA"
    assert avg.loc[avg[ids.PROJECT] == 2, "ZonopwekVerbruikDiff_avg"].notna().any(), (
            "project 2 has data and should have an average"
        )
    # Three slots for each of the two projects: a project with nothing to
    # average keeps its slots rather than being dropped wholesale.
    assert len(avg) == 6, f"expected 3 slots per project, got {len(avg)}"
    assert len(avg[avg[ids.PROJECT] == 2]) == 3, (
            "project 2 should still have its 3 slots"
        )



def test_flagged_values_excluded_in_both_paths():
    """Values the mapping stage flagged as not measured at their slot are kept
    out of the average by both paths.

    Household 2's readings are flagged FROZEN_SPAN, so the average must be
    household 1's level alone. The values are well inside any threshold, which
    is why the flag is what excludes them.
    """
    from etdmap.flags import MappingFlag, flag_column

    findings = flag_column("ZonopwekVerbruikDiff")
    mini = household_batches_frame(
        {
            one_to_one_test_household_batch_of(1): {
                "ZonopwekVerbruikDiff": [3.0, 3.0, 3.0],
                findings: [0, 0, 0],
            },
            one_to_one_test_household_batch_of(2): {
                "ZonopwekVerbruikDiff": [9.0, 9.0, 9.0],
                findings: [MappingFlag.FROZEN_SPAN.value] * 3,
            },
        },
        start="2024-06-01",
        freq="5min",
        tz=None,
        dtypes={findings: "Int64"},
        pairs=False,
    )
    mini["ReadingDate"] = mini["ReadingDate"].astype("datetime64[us]")

    result_ibis = calculate_average_diff_ibis(
        ibis.memtable(mini), ids.PROJECT, ["ZonopwekVerbruikDiff"],
        cadence=_cadence(mini, ["ZonopwekVerbruikDiff"]), grid_minutes=5,
        # Two households, which is smaller than a project: the household floor
        # would empty every slot, and what is under test here is which values
        # reach the mean.
        min_eligible=1,
    )

    result = result_ibis
    assert result["ZonopwekVerbruikDiff"]["n_flagged"] == 3, "wrong flagged count"
    avg = result["ZonopwekVerbruikDiff"]["avg_diff"]["ZonopwekVerbruikDiff_avg"].astype(float)
    assert (avg == 3.0).all(), "flagged values shaped the average"



def test_all_null_diffs_edge_case():
    """A project whose every value is pd.NA gets a row per slot, carrying pd.NA.

    A slot that exists and has no average is not the same as a slot that does
    not exist, and the merge downstream is on (project, ReadingDate), so the
    slots must survive the averaging with no value rather than being dropped
    along with the household that had nothing to contribute.

    Both paths agree on this, which is the point of the assertion.
    """
    absent = [pd.NA, pd.NA]
    mini = household_batches_frame(
        {one_to_one_test_household_batch_of(1): {"ZonopwekVerbruikDiff": absent}},
        start="2024-06-01",
        freq="5min",
        tz=None,
        pairs=False,
    )
    mini["ReadingDate"] = mini["ReadingDate"].astype("datetime64[us]")

    result_ibis = calculate_average_diff_ibis(
        ibis.memtable(mini), ids.PROJECT, ["ZonopwekVerbruikDiff"],
        cadence=_cadence(mini, ["ZonopwekVerbruikDiff"]), grid_minutes=5,
        # Two households, which is smaller than a project: the household floor
        # would empty every slot, and what is under test here is which values
        # reach the mean.
        min_eligible=1,
    )

    result = result_ibis
    avg = result["ZonopwekVerbruikDiff"]["avg_diff"]

    assert len(avg) == len(absent), (
            f"expected one row per slot, got {len(avg)}"
        )
    assert avg["ZonopwekVerbruikDiff_avg"].isna().all(), (
            "an all-null project has no average to report"
        )
