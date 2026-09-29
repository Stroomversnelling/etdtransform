"""The pandas and ibis averaging paths must agree.

The ibis path is the one production runs. The pandas path is what the tests
reason about, so an equivalence test is the only thing that makes reasoning
about one tell you anything about the other.

They reach the same answer by different routes: pandas carries each window's
value backwards over the slots it covers, ibis expands each window against a
small table of offsets, because DuckDB does not expose IGNORE NULLS through
ibis 9.5.
"""

import ibis
import pandas as pd
import pytest

from etdtransform.bounds import resample_to_grid
from etdtransform.impute import _resample_average_ibis

pytestmark = pytest.mark.unit

CUM = "ElektriciteitNetgebruik"
IV = "ElektriciteitNetgebruikDiff"
GRID = 5


def _frame(per_household):
    """per_household: {HuisIdBSV: [cumulative readings, pd.NA where absent]}"""
    parts = []
    for household, readings in per_household.items():
        frame = pd.DataFrame({
            "HuisIdBSV": pd.array([household] * len(readings), dtype="Int64"),
            "ProjectIdBSV": pd.array([1] * len(readings), dtype="Int64"),
            "ReadingDate": pd.date_range(
                "2026-01-01", periods=len(readings), freq=f"{GRID}min"
            ),
            CUM: pd.array(readings, dtype="Float64"),
        })
        frame[IV] = frame[CUM].diff().astype("Float64")
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def _cadence(minutes_by_household):
    return pd.DataFrame({
        "HuisIdBSV": pd.array(list(minutes_by_household), dtype="Int64"),
        "column": pd.array([CUM] * len(minutes_by_household), dtype="string"),
        "cadence_minutes": pd.array(
            list(minutes_by_household.values()), dtype="Int64"
        ),
    })


def _pandas_average(frame, cadence, drop_na=True):
    resampled = frame.assign(**{IV: resample_to_grid(frame, IV, cadence, GRID)})
    out = (
        resampled.groupby(["ProjectIdBSV", "ReadingDate"])[IV].mean().reset_index()
    )
    out.columns = ["ProjectIdBSV", "ReadingDate", f"{IV}_avg"]
    if drop_na:
        out = out[out[f"{IV}_avg"].notna()]
    return out.reset_index(drop=True)


def _ibis_average(frame, cadence, drop_na=True):
    """Prepare exactly as calculate_average_diff_ibis does.

    The cumulative supplies the volume for every household that has usable
    readings of it; the interval column supplies it for the rest. Handing the
    cumulative in unconditionally would agree with pandas for the wrong reason.
    """
    minutes = dict(zip(cadence["HuisIdBSV"], cadence["cadence_minutes"]))
    on_cumulative = frame["HuisIdBSV"].isin(
        frame.loc[frame[CUM].notna(), "HuisIdBSV"].unique()
    )
    prepared = ibis.memtable(
        frame.assign(
            _value=frame[CUM].where(on_cumulative, frame[IV]).astype("Float64"),
            _minutes=frame["HuisIdBSV"].map(minutes).astype("Float64"),
            _on_cumulative=on_cumulative.astype("boolean"),
        )
    )
    # Both are properties of the frame, counted once by the caller rather than
    # once per column -- the test prepares them the same way.
    total_slots = int(
        frame[["ProjectIdBSV", "ReadingDate"]].drop_duplicates().shape[0]
    )
    coarsest = max(cadence["cadence_minutes"].dropna())
    max_slots = max(1, int(round(float(coarsest) / GRID)))
    # One slot per window neutralises the presence gates: a household covers a
    # window it delivered for and covers nothing else, and a slot's minimum
    # never exceeds its own contributors. What is compared here is the
    # resampling, which the two paths reach by different routes. Agreement with
    # the gates applied is tested in test_minimum_household_presence.py.
    avg, coverage, duty = _resample_average_ibis(
        prepared, IV, "ProjectIdBSV", GRID,
        max_slots=max_slots, total_slots=total_slots,
        window_minutes=GRID,
    )
    if drop_na:
        avg = avg[avg[f"{IV}_avg"].notna()]
    return avg.reset_index(drop=True), coverage, duty


def _assert_same(frame, cadence):
    expected = _pandas_average(frame, cadence)
    actual, _, _ = _ibis_average(frame, cadence)
    assert len(actual) == len(expected), (
        f"row counts differ: ibis {len(actual)}, pandas {len(expected)}"
    )
    merged = expected.merge(
        actual, on=["ProjectIdBSV", "ReadingDate"], suffixes=("_pandas", "_ibis")
    )
    assert len(merged) == len(expected), "slots differ between the two paths"
    for _, row in merged.iterrows():
        assert float(row[f"{IV}_avg_ibis"]) == pytest.approx(
            float(row[f"{IV}_avg_pandas"])
        ), f"differ at {row['ReadingDate']}"
    return expected


class TestPathsAgree:
    def test_one_grid_cadence_household(self):
        frame = _frame({1: [100.0, 101.0, 103.0, 106.0, 110.0]})
        _assert_same(frame, _cadence({1: 5}))

    def test_one_coarse_household(self):
        frame = _frame({
            1: [100.0, pd.NA, pd.NA, 106.0, pd.NA, pd.NA, 115.0, pd.NA, pd.NA]
        })
        _assert_same(frame, _cadence({1: 15}))

    def test_a_mixed_cadence_project(self):
        """The case production data does not contain and the method exists for."""
        frame = _frame({
            1: [100.0, 101.0, 103.0, 106.0, 107.0, 108.0, 110.0, 111.0, 112.0],
            2: [500.0, pd.NA, pd.NA, 512.0, pd.NA, pd.NA, 530.0, pd.NA, pd.NA],
        })
        result = _assert_same(frame, _cadence({1: 5, 2: 15}))
        # Slot 00:15 averages household 1's 3.0 with household 2's 12/3 = 4.0.
        row = result[result["ReadingDate"] == pd.Timestamp("2026-01-01 00:15")]
        assert float(row[f"{IV}_avg"].iloc[0]) == pytest.approx(3.5)

    def test_an_outage_window_is_dropped_by_both(self):
        frame = _frame({1: [100.0] + [pd.NA] * 5 + [130.0] + [pd.NA] * 2})
        expected = _pandas_average(frame, _cadence({1: 15}))
        actual, _, _ = _ibis_average(frame, _cadence({1: 15}))
        assert expected.empty and actual.empty

    def test_households_at_three_different_cadences(self):
        frame = _frame({
            1: [100.0, 101.0, 103.0, 106.0, 107.0, 108.0, 110.0, 111.0, 112.0],
            2: [500.0, pd.NA, pd.NA, 512.0, pd.NA, pd.NA, 530.0, pd.NA, pd.NA],
            3: [900.0, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 930.0, pd.NA, pd.NA],
        })
        _assert_same(frame, _cadence({1: 5, 2: 15, 3: 30}))


class TestCoverageAndDuty:
    def test_a_column_that_delivers_nothing_has_zero_coverage(self):
        frame = _frame({1: [pd.NA] * 6})
        _, coverage, duty = _ibis_average(frame, _cadence({1: 5}))
        assert coverage == 0.0
        assert duty == 0.0

    def test_a_coarse_column_now_has_coverage(self):
        """What M is for: this column had no average at all before."""
        frame = _frame({
            1: [100.0, pd.NA, pd.NA, 106.0, pd.NA, pd.NA, 115.0, pd.NA, pd.NA]
        })
        _, coverage, _ = _ibis_average(frame, _cadence({1: 15}))
        assert coverage > 0.0

    def test_duty_is_zero_for_a_column_that_never_advances(self):
        frame = _frame({1: [100.0, 100.0, 100.0, 100.0]})
        _, coverage, duty = _ibis_average(frame, _cadence({1: 5}))
        assert coverage > 0.0
        assert duty == 0.0


class TestSourceSelectionAgrees:
    def test_a_delivered_interval_column_agrees_across_paths(self):
        """No cumulative at all, so both paths take the interval column and
        treat its value as the window volume rather than differencing it."""
        frame = _frame({1: [pd.NA] * 6})
        frame[IV] = pd.array([pd.NA, pd.NA, 6.0, pd.NA, pd.NA, 9.0], dtype="Float64")
        cadence = _cadence({1: 15})
        cadence["column"] = pd.array([IV], dtype="string")
        _assert_same(frame, cadence)

    def test_one_household_on_each_source(self):
        frame = _frame({
            1: [100.0, pd.NA, pd.NA, 106.0, pd.NA, pd.NA, 115.0, pd.NA, pd.NA],
            2: [pd.NA] * 9,
        })
        frame.loc[frame["HuisIdBSV"] == 2, IV] = pd.array(
            [pd.NA, pd.NA, 3.0, pd.NA, pd.NA, 6.0, pd.NA, pd.NA, 9.0],
            dtype="Float64",
        )
        cadence = pd.concat([
            _cadence({1: 15}),
            pd.DataFrame({
                "HuisIdBSV": pd.array([2], dtype="Int64"),
                "column": pd.array([IV], dtype="string"),
                "cadence_minutes": pd.array([15], dtype="Int64"),
            }),
        ], ignore_index=True)
        _assert_same(frame, cadence)


class TestUncoveredSlotsAreRowsNotAbsences:
    """A slot no window covers must come back as a row carrying pd.NA, not be
    dropped.

    Grouping over the expanded windows alone loses those slots, and a missing
    row is not the same as a row with no value: the merge downstream is on
    (project, ReadingDate), so an absent row silently changes what imputation
    sees. This is what the real fixtures caught and the synthetic ones did not.
    """

    @staticmethod
    def _with_an_uncovered_stretch():
        # A reading at row 0 and another at row 7, with the column reporting
        # every 15 minutes. The window between them is too long, so rows 1-7 are
        # covered by nothing at all.
        return _frame({1: [100.0, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 130.0, pd.NA]})

    def test_both_paths_return_a_row_for_every_slot(self):
        frame = self._with_an_uncovered_stretch()
        cadence = _cadence({1: 15})
        expected = _pandas_average(frame, cadence, drop_na=False)
        actual, _, _ = _ibis_average(frame, cadence, drop_na=False)
        assert len(actual) == len(expected) == frame["ReadingDate"].nunique()

    def test_the_uncovered_slots_carry_na_in_both(self):
        frame = self._with_an_uncovered_stretch()
        cadence = _cadence({1: 15})
        expected = _pandas_average(frame, cadence, drop_na=False)
        actual, _, _ = _ibis_average(frame, cadence, drop_na=False)
        assert expected[f"{IV}_avg"].isna().all()
        assert actual[f"{IV}_avg"].isna().all()
