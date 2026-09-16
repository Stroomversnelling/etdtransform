"""Proposal D: a project average that keeps households at their own cadence.

The two populations in a mixed-cadence project know different things. Households
reporting at the grid cadence are the only ones who can see *when* consumption
happens. Households reporting hourly still know *how much*. Averaging them
together mixes quantities covering different spans of time, and the result is
wrong in both level and shape.

D takes the shape from the grid-cadence households and the level from everyone,
then rescales the shape to match. Nothing is discarded and no project is split.
"""

import logging
import math

import pandas as pd
import pytest

from etdtransform.bounds import shape_and_level_average

CUM = "WatergebruikWarmTapwater"
DIFF = f"{CUM}Diff"
N = 48  # four hours on a five-minute grid


@pytest.fixture(autouse=True)
def _quiet_logs():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _true_shape():
    """A sinusoid: mean 2.0, swinging between 0.2 and 3.8."""
    return [2.0 + 1.8 * math.sin(i / N * 2 * math.pi) for i in range(N)]


def _project(n_grid=6, n_hourly=2):
    """One project: some households on the grid, some reporting hourly.

    Every household consumes exactly the same thing. Only how often they report
    it differs, so any correct average must return the true shape.
    """
    shape = _true_shape()
    dates = pd.date_range("2026-01-01", periods=N, freq="5min")
    rows = []
    for hh in range(1, n_grid + 1):
        rows += [
            {"HuisIdBSV": hh, "ProjectIdBSV": 1, "ReadingDate": d, DIFF: shape[i]}
            for i, d in enumerate(dates)
        ]
    for hh in range(n_grid + 1, n_grid + n_hourly + 1):
        rows += [
            {
                "HuisIdBSV": hh, "ProjectIdBSV": 1, "ReadingDate": d,
                # one reading per twelve slots, holding the hour's total
                DIFF: sum(shape[i - 11:i + 1]) if i % 12 == 11 else pd.NA,
            }
            for i, d in enumerate(dates)
        ]
    df = pd.DataFrame(rows)
    df[DIFF] = df[DIFF].astype("Float64")

    cadence = pd.DataFrame({
        "HuisIdBSV": pd.array(range(1, n_grid + n_hourly + 1), dtype="Int64"),
        "column": pd.array([CUM] * (n_grid + n_hourly), dtype="string"),
        "cadence_minutes": pd.array(
            [5] * n_grid + [60] * n_hourly, dtype="Int64"
        ),
    })
    return df, cadence, pd.Series(shape)


def _rmse(values, truth):
    got = pd.Series(values).astype(float).reset_index(drop=True)
    return float(((got - truth) ** 2).mean() ** 0.5)


class TestShapeAndLevelAverage:
    def test_it_recovers_the_true_shape_exactly(self):
        df, cadence, truth = _project()
        avg, _ = shape_and_level_average(df, DIFF, "ProjectIdBSV", cadence)
        assert _rmse(avg[f"{DIFF}_avg"].values, truth) == pytest.approx(0.0, abs=1e-9)

    def test_the_naive_mean_does_not(self):
        """Mixing cadences puts twelve slots' worth into one slot. The error is
        comparable to the signal itself, so this is not a refinement."""
        df, cadence, truth = _project()
        naive = df.groupby("ReadingDate")[DIFF].mean().reset_index()
        assert _rmse(naive[DIFF].values, truth) > 1.0

    def test_the_level_uses_every_household(self):
        df, cadence, truth = _project()
        avg, diagnostics = shape_and_level_average(df, DIFF, "ProjectIdBSV", cadence)
        assert diagnostics["shape_households"] == 6
        assert diagnostics["level_households"] == 8
        assert float(avg[f"{DIFF}_avg"].mean()) == pytest.approx(2.0, abs=1e-9)

    def test_a_uniform_cadence_project_is_unaffected(self):
        """Where every household reports at the grid cadence, D must reduce to
        the plain mean -- otherwise it would change results that are already
        right."""
        df, cadence, truth = _project(n_grid=6, n_hourly=0)
        avg, _ = shape_and_level_average(df, DIFF, "ProjectIdBSV", cadence)
        naive = df.groupby("ReadingDate")[DIFF].mean().reset_index()
        pd.testing.assert_series_equal(
            avg[f"{DIFF}_avg"].astype(float).round(9),
            naive[DIFF].astype(float).round(9),
            check_names=False,
        )

    def test_an_unbalanced_panel_at_uniform_cadence_is_also_unaffected(self):
        """The case that caught a real flaw. One household with a gap contributes
        to some slots and not others, which shifts the mean of the per-slot means
        without any cadence being involved.

        Scaling the shape to match an absolute level rescaled for that imbalance
        too -- 6% on this fixture. The scale is a ratio of two levels computed
        the same way, so with no coarse households it is exactly 1.
        """
        df, cadence, truth = _project(n_grid=6, n_hourly=0)
        gap = (df.HuisIdBSV == 3) & df.ReadingDate.isin(
            pd.date_range("2026-01-01", periods=N, freq="5min")[10:25]
        )
        df.loc[gap, DIFF] = pd.NA

        avg, _ = shape_and_level_average(df, DIFF, "ProjectIdBSV", cadence)
        naive = df.groupby("ReadingDate")[DIFF].mean().reset_index()
        pd.testing.assert_series_equal(
            avg[f"{DIFF}_avg"].astype(float).round(9),
            naive[DIFF].astype(float).round(9),
            check_names=False,
        )

    def test_it_falls_back_when_no_household_reports_at_the_grid_cadence(self):
        """With no grid-cadence household there is no shape to take, so it
        degrades to a rate-normalised mean rather than failing."""
        df, cadence, truth = _project(n_grid=0, n_hourly=3)
        avg, diagnostics = shape_and_level_average(df, DIFF, "ProjectIdBSV", cadence)
        assert diagnostics["fallback"] is True
        assert len(avg) > 0

    def test_a_project_without_grid_households_still_gets_an_average(self):
        """The fallback is per project, not per run. A project with no
        grid-cadence household has no shape to scale; taking the fallback only
        when *no* project has one would drop it from the result entirely, and
        its households would have nothing to impute from.
        """
        grid, cadence_grid, _ = _project(n_grid=4, n_hourly=0)
        coarse, cadence_coarse, _ = _project(n_grid=0, n_hourly=2)
        coarse["ProjectIdBSV"] = 2
        coarse["HuisIdBSV"] += 100
        cadence_coarse["HuisIdBSV"] += 100

        df = pd.concat([grid, coarse], ignore_index=True)
        cadence = pd.concat([cadence_grid, cadence_coarse], ignore_index=True)

        avg, diagnostics = shape_and_level_average(df, DIFF, "ProjectIdBSV", cadence)

        assert diagnostics["fallback"] is True
        assert set(avg["ProjectIdBSV"]) == {1, 2}
        coarse_avg = avg.loc[avg.ProjectIdBSV == 2, f"{DIFF}_avg"].astype(float)
        assert coarse_avg.notna().any()
        # An hourly reading spread over the grid: the mean of the rates it
        # implies is the level the household actually consumed.
        assert float(coarse_avg.mean()) == pytest.approx(2.0, abs=1e-9)


class TestBothPathsAgreeUnderMixedCadence:
    """The two averaging paths are documented as substitutable, and mixed
    cadence is the one case where an arithmetic mean would differ from D.
    """

    @staticmethod
    def _both(df, cadence):
        import ibis

        from etdtransform.impute import (
            calculate_average_diff,
            calculate_average_diff_ibis,
        )

        frame = df.copy()
        frame["ProjectIdBSV"] = frame["ProjectIdBSV"].astype("Int64")
        frame["HuisIdBSV"] = frame["HuisIdBSV"].astype("Int64")
        frame["ReadingDate"] = frame["ReadingDate"].astype("datetime64[us]")

        kwargs = dict(bounds={}, cadence=cadence)
        from_pandas = calculate_average_diff(
            frame, "ProjectIdBSV", [DIFF], **kwargs
        )[DIFF]
        from_ibis = calculate_average_diff_ibis(
            ibis.memtable(frame), "ProjectIdBSV", [DIFF], **kwargs
        )[DIFF]

        order = ["ProjectIdBSV", "ReadingDate"]
        return (
            from_pandas["avg_diff"].sort_values(order).reset_index(drop=True),
            from_ibis["avg_diff"].sort_values(order).reset_index(drop=True),
            from_pandas,
            from_ibis,
        )

    def test_the_ibis_path_recovers_the_true_shape_too(self):
        df, cadence, truth = _project()
        _, ibis_avg, _, _ = self._both(df, cadence)
        assert _rmse(ibis_avg[f"{DIFF}_avg"].values, truth) == pytest.approx(
            0.0, abs=1e-9
        )

    def test_the_two_paths_produce_the_same_average(self):
        df, cadence, truth = _project()
        pandas_avg, ibis_avg, pandas_result, ibis_result = self._both(df, cadence)
        pd.testing.assert_frame_equal(
            pandas_avg, ibis_avg, check_like=True, rtol=1e-9
        )
        assert (
            ibis_result["shape_level"]["shape_households"]
            == pandas_result["shape_level"]["shape_households"]
        )

    def test_they_agree_at_uniform_cadence_as_well(self):
        df, cadence, truth = _project(n_grid=6, n_hourly=0)
        pandas_avg, ibis_avg, _, _ = self._both(df, cadence)
        pd.testing.assert_frame_equal(
            pandas_avg, ibis_avg, check_like=True, rtol=1e-9
        )

    def test_they_agree_when_there_is_no_shape_to_take(self):
        df, cadence, truth = _project(n_grid=0, n_hourly=3)
        pandas_avg, ibis_avg, pandas_result, ibis_result = self._both(df, cadence)
        assert ibis_result["shape_level"]["fallback"] is True
        assert pandas_result["shape_level"]["fallback"] is True
        pd.testing.assert_frame_equal(
            pandas_avg, ibis_avg, check_like=True, rtol=1e-9
        )

    def test_they_agree_when_one_project_falls_back_and_another_does_not(self):
        grid, cadence_grid, _ = _project(n_grid=4, n_hourly=0)
        coarse, cadence_coarse, _ = _project(n_grid=0, n_hourly=2)
        coarse["ProjectIdBSV"] = 2
        coarse["HuisIdBSV"] += 100
        cadence_coarse["HuisIdBSV"] += 100

        df = pd.concat([grid, coarse], ignore_index=True)
        cadence = pd.concat([cadence_grid, cadence_coarse], ignore_index=True)

        pandas_avg, ibis_avg, _, _ = self._both(df, cadence)
        assert set(pandas_avg["ProjectIdBSV"]) == {1, 2}
        pd.testing.assert_frame_equal(
            pandas_avg, ibis_avg, check_like=True, rtol=1e-9
        )
