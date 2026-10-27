"""`avg_coverage_duty.parquet`: how much of each project average exists and moves.

The fill orders its columns by coverage, then duty. That ordering has to be the
same on a re-run as it was on the run that produced the averages, so it is read
from an artifact written beside `avg_diffs.parquet` rather than recomputed from
whatever happens to be in memory.

Two figures, because they answer different questions. *coverage* is the share of
a project's slots the average has a value for at all. *duty* is, of those, the
share carrying consumption rather than a measured zero -- a column can be fully
covered and almost entirely idle, and ordering by coverage alone would put such
a column ahead of one that actually moves.
"""

import pandas as pd
import pytest

from conftest import cadence_table, project_averages
from etdtransform.impute import (
    _coverage_duty_by_project,
    write_coverage_duty,
)

pytestmark = pytest.mark.unit

CUM = "ElektriciteitNetgebruik"
DIFF = "ElektriciteitNetgebruikDiff"
AVG = f"{DIFF}_avg"


def _avg_frame(projects, values):
    return pd.DataFrame({
        "ProjectIdBSV": pd.array(projects, dtype="Int64"),
        "ReadingDate": pd.date_range("2026-01-01", periods=len(projects), freq="5min"),
        AVG: pd.array(values, dtype="Float64"),
    })


class TestCoverageIsTheShareOfSlotsWithAValue:
    def test_every_slot_covered(self):
        out = _coverage_duty_by_project(
            _avg_frame([1, 1, 1], [0.1, 0.2, 0.3]), DIFF, "ProjectIdBSV"
        )
        assert float(out["coverage"].iloc[0]) == 1.0

    def test_half_the_slots_covered(self):
        out = _coverage_duty_by_project(
            _avg_frame([1, 1, 1, 1], [0.1, pd.NA, 0.3, pd.NA]), DIFF, "ProjectIdBSV"
        )
        assert float(out["coverage"].iloc[0]) == 0.5

    def test_nothing_covered(self):
        out = _coverage_duty_by_project(
            _avg_frame([1, 1], [pd.NA, pd.NA]), DIFF, "ProjectIdBSV"
        )
        assert float(out["coverage"].iloc[0]) == 0.0


class TestDutyIsMeasuredAmongTheCoveredSlotsOnly:
    def test_all_covered_slots_carry_consumption(self):
        out = _coverage_duty_by_project(
            _avg_frame([1, 1], [0.1, 0.2]), DIFF, "ProjectIdBSV"
        )
        assert float(out["duty"].iloc[0]) == 1.0

    def test_a_measured_zero_is_covered_but_idle(self):
        """Zero is a measurement, so it counts for coverage and not for duty.
        Conflating the two would read an idle device as an absent one."""
        out = _coverage_duty_by_project(
            _avg_frame([1, 1, 1, 1], [0.0, 0.0, 0.0, 0.4]), DIFF, "ProjectIdBSV"
        )
        row = out.iloc[0]
        assert float(row["coverage"]) == 1.0
        assert float(row["duty"]) == 0.25

    def test_an_absent_slot_is_excluded_from_the_denominator(self):
        out = _coverage_duty_by_project(
            _avg_frame([1, 1, 1, 1], [0.4, pd.NA, pd.NA, pd.NA]), DIFF, "ProjectIdBSV"
        )
        row = out.iloc[0]
        assert float(row["coverage"]) == 0.25
        assert float(row["duty"]) == 1.0, "one covered slot, and it consumes"

    def test_no_covered_slots_gives_zero_not_a_division(self):
        out = _coverage_duty_by_project(
            _avg_frame([1, 1], [pd.NA, pd.NA]), DIFF, "ProjectIdBSV"
        )
        assert float(out["duty"].iloc[0]) == 0.0


class TestOneRowPerProjectAndColumn:
    def test_each_project_is_counted_separately(self):
        out = _coverage_duty_by_project(
            _avg_frame([1, 1, 2, 2], [0.1, 0.2, 0.3, pd.NA]), DIFF, "ProjectIdBSV"
        ).set_index("ProjectIdBSV")
        assert float(out.loc[1, "coverage"]) == 1.0
        assert float(out.loc[2, "coverage"]) == 0.5

    def test_the_column_is_named_on_every_row(self):
        out = _coverage_duty_by_project(
            _avg_frame([1, 2], [0.1, 0.2]), DIFF, "ProjectIdBSV"
        )
        assert set(out["column"]) == {DIFF}

    def test_the_counts_behind_the_shares_are_kept(self):
        """A share alone cannot say whether it rests on three slots or a million."""
        out = _coverage_duty_by_project(
            _avg_frame([1, 1, 1, 1], [0.0, 0.4, pd.NA, pd.NA]), DIFF, "ProjectIdBSV"
        ).iloc[0]
        assert int(out["slots"]) == 4
        assert int(out["present"]) == 2
        assert int(out["consuming"]) == 1


class TestTheArtifact:
    def _dict(self, *columns):
        return {
            col: {
                "coverage_duty": _coverage_duty_by_project(
                    _avg_frame([1, 2], [0.1, 0.2]).rename(
                        columns={AVG: f"{col}_avg"}
                    ),
                    col,
                    "ProjectIdBSV",
                )
            }
            for col in columns
        }

    def test_it_holds_every_column(self, tmp_path):
        out = write_coverage_duty(
            self._dict(DIFF, "GasgebruikDiff"), aggregate_folder_path=tmp_path
        )
        assert set(out["column"]) == {DIFF, "GasgebruikDiff"}

    def test_it_lands_beside_the_averages(self, tmp_path):
        write_coverage_duty(self._dict(DIFF), aggregate_folder_path=tmp_path)
        assert (tmp_path / "avg_coverage_duty.parquet").is_file()

    def test_it_survives_a_round_trip(self, tmp_path):
        written = write_coverage_duty(
            self._dict(DIFF), aggregate_folder_path=tmp_path
        )
        back = pd.read_parquet(
            tmp_path / "avg_coverage_duty.parquet", dtype_backend="numpy_nullable"
        )
        assert list(back.columns) == list(written.columns)
        assert len(back) == len(written)

    def test_nothing_to_write_writes_nothing(self, tmp_path):
        assert write_coverage_duty({}, aggregate_folder_path=tmp_path).empty
        assert not (tmp_path / "avg_coverage_duty.parquet").exists()


class TestItAgreesWithTheScalarTheAveragingAlreadyReported:
    """The per-project table and the existing whole-estate scalar are the same
    quantity at different grain, so a single project must give both."""

    def test_one_project_matches_the_scalar(self):
        frame = pd.DataFrame({
            "HuisIdBSV": pd.array([1] * 4, dtype="Int64"),
            "HuisBatchIdBSV": pd.array([value + 1000 for value in [1] * 4], dtype="Int64"),
            "ProjectIdBSV": pd.array([1] * 4, dtype="Int64"),
            "ReadingDate": pd.date_range("2026-01-01", periods=4, freq="5min"),
            DIFF: pd.array([0.0, 0.4, 0.2, 0.1], dtype="Float64"),
        })
        result = project_averages(
            frame, "ProjectIdBSV", [DIFF],
            cadence=cadence_table(frame, [CUM]), grid_minutes=5,
        )[DIFF]
        row = result["coverage_duty"].iloc[0]
        assert float(row["coverage"]) == pytest.approx(result["coverage"])
        assert float(row["duty"]) == pytest.approx(result["duty"])
