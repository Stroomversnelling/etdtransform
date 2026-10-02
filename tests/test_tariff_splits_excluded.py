"""Tariff splits (Hoog/Laag) leave averaging and imputation.

The combined root is delivered at mapping (combine_tariff_registers), so the
continuous root is what gets averaged and filled; the discrete alternating
splits are not physically averageable. They stay in the mapped data
untouched: no averages, no fills, no impute flag columns, and their
delivered cumulative registers are never rebuilt by reconstruction.
Membership comes from the data model's tariff pair structures, applied at
the column-list choke points (get_diff_columns for the averaging twins;
impute_and_normalize for every fill shell).
"""

import pandas as pd
import pytest
from conftest import cadence_table

from etdtransform.impute import (
    calculate_average_diff,
    concatenate_avg_diff_columns,
    get_diff_columns,
)
from etdtransform.vectorized_impute import (
    impute_and_normalize,
    without_tariff_splits,
)

pytestmark = pytest.mark.unit

ROOT = "ElektriciteitNetgebruik"
ROOT_DIFF = "ElektriciteitNetgebruikDiff"
HOOG = "ElektriciteitNetgebruikHoog"
LAAG = "ElektriciteitNetgebruikLaag"
HOOG_DIFF = "ElektriciteitNetgebruikHoogDiff"
LAAG_DIFF = "ElektriciteitNetgebruikLaagDiff"
WTW = "ElektriciteitsgebruikWTW"
WTW_DIFF = "ElektriciteitsgebruikWTWDiff"


class TestWithoutTariffSplits:
    def test_drops_cumulative_and_diff_split_forms(self):
        columns = [ROOT, HOOG, LAAG, ROOT_DIFF, HOOG_DIFF, LAAG_DIFF, WTW]
        assert without_tariff_splits(columns) == [ROOT, ROOT_DIFF, WTW]

    def test_non_tariff_columns_pass_untouched(self):
        columns = [WTW, WTW_DIFF, "Gasgebruik"]
        assert without_tariff_splits(columns) == columns


class TestAveragingChokePoint:
    def test_get_diff_columns_excludes_splits_keeps_root(self):
        diff = get_diff_columns([ROOT, HOOG, LAAG, WTW])
        assert ROOT_DIFF in diff
        assert WTW_DIFF in diff
        assert HOOG_DIFF not in diff
        assert LAAG_DIFF not in diff


class TestAveragingMicroMixedSupplier:
    """Action 10d: a mixed-supplier frame -- project 1 (root delivered
    directly) and project 2 (root combined from splits at mapping; the frame
    carries the combined root plus the delivered splits). avg_diffs carries
    NO split columns, the root IS averaged in both projects, and a
    non-tariff column is unaffected."""

    def _frame(self):
        n = 12
        dates = pd.date_range("2026-01-01", periods=n, freq="5min")
        rows = []
        for project, household, root_value in (
            (1, 1, 1.0), (1, 2, 3.0), (2, 3, 2.0), (2, 4, 4.0),
        ):
            rows.append(pd.DataFrame({
                "HuisIdBSV": pd.array([household] * n, dtype="Int64"),
                "ProjectIdBSV": pd.array([project] * n, dtype="Int64"),
                "ReadingDate": dates,
                ROOT_DIFF: pd.array([root_value] * n, dtype="Float64"),
                # project 2 also carries its delivered splits (kept as mapped)
                HOOG_DIFF: pd.array(
                    [root_value / 2 if project == 2 else pd.NA] * n,
                    dtype="Float64",
                ),
                LAAG_DIFF: pd.array(
                    [root_value / 2 if project == 2 else pd.NA] * n,
                    dtype="Float64",
                ),
                WTW_DIFF: pd.array([0.5] * n, dtype="Float64"),
            }))
        return pd.concat(rows, ignore_index=True)

    def test_avg_diffs_has_root_not_splits(self):
        df = self._frame()
        diff_columns = get_diff_columns([ROOT, HOOG, LAAG, WTW])
        avg_dict = calculate_average_diff(
            df, "ProjectIdBSV", diff_columns,
            bounds={},
            cadence=cadence_table(df, [ROOT, WTW]),
            grid_minutes=5,
        )
        assert set(avg_dict) == {ROOT_DIFF, WTW_DIFF}

        diffs = concatenate_avg_diff_columns(avg_dict, "ProjectIdBSV")
        avg_cols = [c for c in diffs.columns if c.endswith("_avg")]
        assert f"{ROOT_DIFF}_avg" in avg_cols
        assert f"{WTW_DIFF}_avg" in avg_cols
        assert f"{HOOG_DIFF}_avg" not in avg_cols
        assert f"{LAAG_DIFF}_avg" not in avg_cols

        # The root average is the project mean of the delivered roots.
        by_project = diffs.groupby("ProjectIdBSV")[f"{ROOT_DIFF}_avg"].mean()
        assert float(by_project.loc[1]) == pytest.approx(2.0)
        assert float(by_project.loc[2]) == pytest.approx(3.0)
        # The non-tariff column is unaffected by the exclusion.
        assert float(diffs[f"{WTW_DIFF}_avg"].mean()) == pytest.approx(0.5)


class TestFillSkipsSplits:
    """The fill never processes a split: no impute flag columns appear for
    it, and its delivered values (including gaps) stay byte-identical."""

    def _frame(self):
        n = 8
        split_diff = [0.5, 0.5, pd.NA, pd.NA, 0.5, 0.5, 0.5, 0.5]
        split_cum = [0.5, 1.0, pd.NA, pd.NA, 2.0, 2.5, 3.0, 3.5]
        return pd.DataFrame({
            "HuisIdBSV": pd.array([1] * n, dtype="Int64"),
            "ProjectIdBSV": pd.array([1] * n, dtype="Int64"),
            "ReadingDate": pd.date_range("2026-01-01", periods=n, freq="5min"),
            ROOT: pd.array([1.0, 2.0, pd.NA, pd.NA, 5.0, 6.0, 7.0, 8.0], dtype="Float64"),
            ROOT_DIFF: pd.array([1.0, 1.0, pd.NA, pd.NA, pd.NA, 1.0, 1.0, 1.0], dtype="Float64"),
            f"{ROOT_DIFF}_avg": pd.array([1.0] * n, dtype="Float64"),
            HOOG: pd.array(split_cum, dtype="Float64"),
            HOOG_DIFF: pd.array(split_diff, dtype="Float64"),
            LAAG: pd.array(split_cum, dtype="Float64"),
            LAAG_DIFF: pd.array(split_diff, dtype="Float64"),
        })

    def test_split_gets_no_flags_and_stays_untouched(self):
        frame = self._frame()
        before_hoog = frame[HOOG_DIFF].copy()
        out, _, _ = impute_and_normalize(
            frame,
            [ROOT, HOOG, LAAG],
            "ProjectIdBSV",
            cadence=cadence_table(frame, [ROOT, HOOG, LAAG]),
        )
        # The root was processed: its flag column exists and the gap filled.
        assert f"{ROOT_DIFF}_impute_type" in out.columns
        assert out[ROOT_DIFF].notna().all()
        # The splits were not: no flag columns, values byte-identical,
        # the gap still a gap.
        assert f"{HOOG_DIFF}_impute_type" not in out.columns
        assert f"{LAAG_DIFF}_impute_type" not in out.columns
        pd.testing.assert_series_equal(
            out[HOOG_DIFF], before_hoog, check_names=False
        )
        assert out[HOOG_DIFF].isna().sum() == 2

    def test_split_only_list_raises_loudly(self):
        """A fill list containing only splits filters down to nothing, and
        the existing nothing-was-imputed guard raises rather than silently
        producing an empty run -- a misconfigured cum_cols must announce
        itself."""
        frame = self._frame()
        with pytest.raises(RuntimeError, match="No columns were imputed"):
            impute_and_normalize(
                frame,
                [HOOG, LAAG],
                "ProjectIdBSV",
                cadence=cadence_table(frame, [HOOG, LAAG]),
            )
