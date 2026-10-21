"""Missingness through the averaging, on a synthetic in-memory fixture.

The fixture carries null (pd.NA) values in its measurement columns on purpose.
Null-in, null-out is a correctness requirement throughout the pipeline: a slot
with no average is not a slot with an average of zero, and a project whose every
reading is absent must come back with its slots present and empty rather than
with no slots at all.

``ibis.memtable()`` wraps each frame, which is what the averaging takes -- the
pipeline passes a lazy table from ``ibis.read_parquet`` instead.
"""
import numpy as np
import pandas as pd
import pytest

import ibis

from etdtransform.impute import calculate_average_diff_ibis
from conftest import cadence_table


def _cadence(frame, diff_columns):
    """The cadence both paths are judged at, from the frame they share."""
    from etdmap.data_model import cumulative_column

    return cadence_table(
        frame, [cumulative_column(c) or c for c in diff_columns]
    )


pytestmark = pytest.mark.unit

# Two projects: 20 normal + 1 outlier per project.
# Outlier values are ~15-20x larger than the normal range so the
# 95th-percentile upper-bound logic clearly excludes them.
_P1_NORMAL = list(range(101, 121))
_P1_OUTLIER = 199
_P2_NORMAL = list(range(201, 221))
_P2_OUTLIER = 299

DIFF_COLS = ["ElektriciteitNetgebruikLaagDiff", "ZonopwekVerbruikDiff"]

# Rows whose measurement values are set to pd.NA to verify null handling.
# We use (project_id, household_id, timestamp_index) tuples to identify them.
_NULL_SLOTS = {(1, 105, 3), (1, 110, 7), (2, 207, 0), (2, 215, 12)}


@pytest.fixture(scope="module")
def diff_fixture_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=20, freq="5min").as_unit("us")
    rows = []

    for project_id, normal_ids, outlier_id in [
        (1, _P1_NORMAL, _P1_OUTLIER),
        (2, _P2_NORMAL, _P2_OUTLIER),
    ]:
        for household_id in normal_ids:
            for t_idx, dt in enumerate(dates):
                null_this = (project_id, household_id, t_idx) in _NULL_SLOTS
                rows.append({
                    "ProjectIdBSV": project_id,
                    "HuisIdBSV": household_id,
                    "ReadingDate": dt,
                    "ElektriciteitNetgebruikLaagDiff": pd.NA if null_this else float(rng.uniform(1.0, 10.0)),
                    "ZonopwekVerbruikDiff": pd.NA if null_this else float(rng.uniform(0.5, 5.0)),
                })
        for dt in dates:
            rows.append({
                "ProjectIdBSV": project_id,
                "HuisIdBSV": outlier_id,
                "ReadingDate": dt,
                "ElektriciteitNetgebruikLaagDiff": float(rng.uniform(150.0, 300.0)),
                "ZonopwekVerbruikDiff": float(rng.uniform(150.0, 300.0)),
            })

    df = pd.DataFrame(rows).astype({
        "ProjectIdBSV": "Int64",
        "HuisIdBSV": "Int64",
        "ElektriciteitNetgebruikLaagDiff": "Float64",
        "ZonopwekVerbruikDiff": "Float64",
    })
    # Row-by-row construction reverts timestamps to ns; pin to us to match
    # DuckDB's output precision and the parquet files produced by the pipeline.
    df["ReadingDate"] = df["ReadingDate"].astype("datetime64[us]")
    return df


def _null_positions(df: pd.DataFrame, col: str) -> set:
    """Return the integer-location set of null positions in a column."""
    return set(df.index[df[col].isna()].tolist())


def test_ibis_memtable_preserves_nulls(diff_fixture_df):
    """ibis.memtable round-trip must preserve null positions and values exactly."""
    tbl = ibis.memtable(diff_fixture_df)
    result = tbl.execute().astype({c: diff_fixture_df[c].dtype for c in diff_fixture_df.columns})

    sort_cols = ["HuisIdBSV", "ReadingDate"]
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
    dates = pd.date_range("2024-01-01", periods=3, freq="5min")
    mini = pd.DataFrame({
        "ProjectIdBSV": pd.array([1, 1, 1, 1, 1, 1], dtype="Int64"),
        "HuisIdBSV": pd.array([1, 1, 1, 2, 2, 2], dtype="Int64"),
        "ReadingDate": list(dates) * 2,
        "ZonopwekVerbruikDiff": pd.array([2.0, pd.NA, 4.0, 6.0, 6.0, 6.0], dtype="Float64"),
    })

    result_ibis = calculate_average_diff_ibis(
        ibis.memtable(mini), "ProjectIdBSV", ["ZonopwekVerbruikDiff"],
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
    dates = pd.date_range("2024-06-01", periods=3, freq="5min")

    # Project 1: all zeros. Project 2: normal non-zero values.
    mini = pd.DataFrame({
        "ProjectIdBSV": pd.array([1, 1, 1, 2, 2, 2], dtype="Int64"),
        "HuisIdBSV":    pd.array([1, 1, 1, 2, 2, 2], dtype="Int64"),
        "ReadingDate":  list(dates) * 2,
        "ZonopwekVerbruikDiff":     pd.array([0.0, 0.0, 0.0, 3.0, 5.0, 4.0], dtype="Float64"),
    })
    mini["ReadingDate"] = mini["ReadingDate"].astype("datetime64[us]")

    result_ibis = calculate_average_diff_ibis(
        ibis.memtable(mini), "ProjectIdBSV", ["ZonopwekVerbruikDiff"],
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
    assert set(avg["ProjectIdBSV"].dropna()) == {1, 2}, (
            f"{label}: both projects should appear, got "
            f"{set(avg['ProjectIdBSV'].dropna())}"
        )
    # Zero is a measurement, so an all-zero project averages to 0.0. That is
    # a different fact from pd.NA, which means nothing was measured, and the
    # two must not be conflated.
    assert (
            avg.loc[avg["ProjectIdBSV"] == 1, "ZonopwekVerbruikDiff_avg"].dropna() == 0.0
        ).all(), f"{label}: an all-zero project should average to zero, not NA"
    assert avg.loc[avg["ProjectIdBSV"] == 2, "ZonopwekVerbruikDiff_avg"].notna().any(), (
            f"{label}: project 2 has data and should have an average"
        )
    # Three slots for each of the two projects, now that project 1 is no
    # longer dropped wholesale.
    assert len(avg) == 6, f"{label}: expected 3 slots per project, got {len(avg)}"
    assert len(avg[avg["ProjectIdBSV"] == 2]) == 3, (
            f"{label}: project 2 should still have its 3 slots"
        )



def test_flagged_values_excluded_in_both_paths():
    """Values the mapping stage flagged as not measured at their slot are kept
    out of the average by both paths.

    Household 2's readings are flagged FROZEN_SPAN, so the average must be
    household 1's level alone. The values are well inside any threshold, which
    is why the flag is what excludes them.
    """
    from etdmap.flags import MappingFlag, flag_column

    dates = pd.date_range("2024-06-01", periods=3, freq="5min")
    mini = pd.DataFrame({
        "ProjectIdBSV": pd.array([1] * 6, dtype="Int64"),
        "HuisIdBSV": pd.array([1, 1, 1, 2, 2, 2], dtype="Int64"),
        "ReadingDate": list(dates) * 2,
        "ZonopwekVerbruikDiff": pd.array([3.0, 3.0, 3.0, 9.0, 9.0, 9.0], dtype="Float64"),
        flag_column("ZonopwekVerbruikDiff"): pd.array(
            [0, 0, 0] + [MappingFlag.FROZEN_SPAN.value] * 3, dtype="Int64"
        ),
    })
    mini["ReadingDate"] = mini["ReadingDate"].astype("datetime64[us]")

    result_ibis = calculate_average_diff_ibis(
        ibis.memtable(mini), "ProjectIdBSV", ["ZonopwekVerbruikDiff"],
        cadence=_cadence(mini, ["ZonopwekVerbruikDiff"]), grid_minutes=5,
        # Two households, which is smaller than a project: the household floor
        # would empty every slot, and what is under test here is which values
        # reach the mean.
        min_eligible=1,
    )

    result = result_ibis
    assert result["ZonopwekVerbruikDiff"]["n_flagged"] == 3, f"{label}: wrong flagged count"
    avg = result["ZonopwekVerbruikDiff"]["avg_diff"]["ZonopwekVerbruikDiff_avg"].astype(float)
    assert (avg == 3.0).all(), f"{label}: flagged values shaped the average"



def test_all_null_diffs_edge_case():
    """A project whose every value is pd.NA gets a row per slot, carrying pd.NA.

    It used to get no rows at all, but only as a side effect of the relative
    household filter: a household whose maximum was NA failed
    `max < 2 x p95(maxima)` and was dropped, taking its slots with it. Plan group
    K removed that filter, so the slots survive with no value -- which is what
    they are. A slot that exists and has no average is not the same as a slot
    that does not exist, and the merge downstream is on (project, ReadingDate).

    Both paths agree on this, which is the point of the assertion.
    """
    dates = pd.date_range("2024-06-01", periods=2, freq="5min")
    mini = pd.DataFrame({
        "ProjectIdBSV": pd.array([1, 1], dtype="Int64"),
        "HuisIdBSV":    pd.array([1, 1], dtype="Int64"),
        "ReadingDate":  list(dates),
        "ZonopwekVerbruikDiff":     pd.array([pd.NA, pd.NA], dtype="Float64"),
    })
    mini["ReadingDate"] = mini["ReadingDate"].astype("datetime64[us]")

    result_ibis = calculate_average_diff_ibis(
        ibis.memtable(mini), "ProjectIdBSV", ["ZonopwekVerbruikDiff"],
        cadence=_cadence(mini, ["ZonopwekVerbruikDiff"]), grid_minutes=5,
        # Two households, which is smaller than a project: the household floor
        # would empty every slot, and what is under test here is which values
        # reach the mean.
        min_eligible=1,
    )

    result = result_ibis
    avg = result["ZonopwekVerbruikDiff"]["avg_diff"]

    assert len(avg) == len(dates), (
            f"{label}: expected one row per slot, got {len(avg)}"
        )
    assert avg["ZonopwekVerbruikDiff_avg"].isna().all(), (
            f"{label}: an all-null project has no average to report"
        )
