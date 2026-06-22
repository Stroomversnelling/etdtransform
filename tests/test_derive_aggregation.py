"""
Tests for the 'derive' resample/aggregate method.

A 'derive' column is an intensive quantity (a ratio, e.g. ZelfgebruikPercentage)
that must NOT be summed or averaged as a stored value. Instead it is recomputed
from its catalog rule AFTER its components have been aggregated to the target
resolution -- a production-weighted result, not the (wrong) unweighted mean of
the per-interval / per-household ratios.

These tests are self-contained: they patch get_aggregation_config and
load_catalog with synthetic objects, so they do not depend on the installed
catalog carrying the ZelfgebruikPercentage rule.
"""

import os

import pandas as pd
import pytest
from unittest.mock import patch

import etdtransform
from etdtransform.aggregate import (
    _catalog_hash,
    _derive_columns_in_parquet,
    _write_calc_provenance,
    resample_hh_data_duckdb,
    aggregate_project_data_duckdb,
)

REL_TOL = 1e-9

# Rule: ZelfgebruikPercentage = 100 * Zelfgebruik / ZonopwekBruto (percent form).
_ZP = "ZelfgebruikPercentage"


def _synth_catalog():
    """Minimal catalog with just the ZP ratio rule (adapter-compatible)."""
    return pd.DataFrame(
        [
            {
                "lhs": _ZP,
                "rhs_text": "100*Zelfgebruik/ZonopwekBruto",
                "rhs_vars": ["Zelfgebruik", "ZonopwekBruto"],
                "rhs_var_count": 2,
            }
        ]
    )


def _derive_config():
    """Components aggregate normally; the ratio is 'derive' on both axes."""
    return {
        "Zelfgebruik": {"resample_method": "sum", "aggregate_method": "avg"},
        "ZonopwekBruto": {"resample_method": "sum", "aggregate_method": "avg"},
        _ZP: {"resample_method": "derive", "aggregate_method": "derive"},
    }


def _read(path):
    return pd.read_parquet(path, dtype_backend="numpy_nullable")


# ---------------------------------------------------------------------------
# Unit: the helper itself
# ---------------------------------------------------------------------------

class TestDeriveHelper:
    def test_recompute_from_components_and_mask_zero_denominator(self, tmp_path):
        path = str(tmp_path / "agg.parquet")
        pd.DataFrame(
            {
                "ProjectIdBSV": [1, 1, 1],
                "Zelfgebruik": pd.array([3.0, 0.0, 6.0], dtype="Float64"),
                "ZonopwekBruto": pd.array([12.0, 0.0, 8.0], dtype="Float64"),
            }
        ).to_parquet(path)

        _derive_columns_in_parquet(path, [_ZP], catalog_df=_synth_catalog())

        out = _read(path)
        zp = out[_ZP]
        assert float(zp[0]) == pytest.approx(25.0, rel=REL_TOL)   # 100*3/12
        assert float(zp[2]) == pytest.approx(75.0, rel=REL_TOL)   # 100*6/8
        assert pd.isna(zp[1])                                     # 0/0 -> NA

    def test_overwrites_stale_value(self, tmp_path):
        path = str(tmp_path / "agg.parquet")
        pd.DataFrame(
            {
                "ProjectIdBSV": [1],
                "Zelfgebruik": pd.array([3.0], dtype="Float64"),
                "ZonopwekBruto": pd.array([12.0], dtype="Float64"),
                _ZP: pd.array([-999.0], dtype="Float64"),  # stale, must be replaced
            }
        ).to_parquet(path)

        _derive_columns_in_parquet(path, [_ZP], catalog_df=_synth_catalog())

        assert float(_read(path)[_ZP][0]) == pytest.approx(25.0, rel=REL_TOL)

    def test_missing_component_skips_and_logs(self, tmp_path):
        path = str(tmp_path / "agg.parquet")
        pd.DataFrame(
            {"ProjectIdBSV": [1], "Zelfgebruik": pd.array([3.0], dtype="Float64")}
        ).to_parquet(path)

        # No ZonopwekBruto -> not derivable; must not raise, must not add the column.
        _derive_columns_in_parquet(path, [_ZP], catalog_df=_synth_catalog())

        assert _ZP not in _read(path).columns


# ---------------------------------------------------------------------------
# Integration: household resample
# ---------------------------------------------------------------------------

def _write_calculated(path):
    """6 rows on a 5min grid for one household = two 15min buckets of 3."""
    ts = pd.date_range("2024-01-01 00:00", periods=6, freq="5min")
    zelf = pd.array([1.0, 2.0, 0.0, 5.0, 0.0, 1.0], dtype="Float64")
    zon = pd.array([4.0, 6.0, 0.0, 5.0, 0.0, 5.0], dtype="Float64")
    # Correct per-5min ratio at the calculated stage (night row -> NA).
    zp = pd.array([25.0, pd.NA, pd.NA, 100.0, pd.NA, 20.0], dtype="Float64")
    pd.DataFrame(
        {
            "HuisIdBSV": [1] * 6,
            "ProjectIdBSV": [1] * 6,
            "ReadingDate": ts,
            "Zelfgebruik": zelf,
            "ZonopwekBruto": zon,
            _ZP: zp,
        }
    ).to_parquet(path)


class TestResampleDerive:
    def test_5min_carries_15min_rederives(self, tmp_path):
        src = str(tmp_path / "household_calculated.parquet")
        out = str(tmp_path / "out")
        os.makedirs(out, exist_ok=True)
        _write_calculated(src)

        with patch("etdmap.data_model.get_aggregation_config", return_value=_derive_config()), \
             patch("etdmap.catalog.load_catalog", return_value=_synth_catalog()):
            resample_hh_data_duckdb(source_path=src, output_dir=out,
                                    intervals=("5min", "15min"))

        # 5min: passthrough carries the calculated-stage ratio unchanged.
        df5 = _read(os.path.join(out, "household_5min.parquet")).sort_values("ReadingDate")
        assert float(df5[_ZP].iloc[0]) == pytest.approx(25.0, rel=REL_TOL)
        assert float(df5[_ZP].iloc[3]) == pytest.approx(100.0, rel=REL_TOL)

        # 15min: recomputed from summed components, NOT the mean of 5min ratios.
        df15 = _read(os.path.join(out, "household_15min.parquet")).sort_values("ReadingDate")
        # bucket 1: sum Zelf=3, Zon=10 -> 30 ; bucket 2: sum Zelf=6, Zon=10 -> 60
        assert float(df15[_ZP].iloc[0]) == pytest.approx(30.0, rel=REL_TOL)
        assert float(df15[_ZP].iloc[1]) == pytest.approx(60.0, rel=REL_TOL)
        # mean of the per-5min ratios for bucket 1 would be (25 + 100)/... -> not 30
        assert float(df15[_ZP].iloc[0]) != pytest.approx(62.5, rel=REL_TOL)

        # provenance sidecar: which equation produced the ratio, per scope.
        prov = _read(os.path.join(out, "derive_provenance_household.parquet"))
        assert (prov["variable"] == _ZP).all()
        assert "household_15min" in set(prov["scope"])
        rhs = prov.loc[prov["scope"] == "household_15min", "rhs_text"].iloc[0]
        assert "Zelfgebruik" in rhs and "ZonopwekBruto" in rhs
        assert prov["catalog_hash"].notna().all()


# ---------------------------------------------------------------------------
# Integration: project (cross-household) aggregation
# ---------------------------------------------------------------------------

class TestProjectDerive:
    def test_fleet_ratio_is_weighted_not_mean_of_ratios(self, tmp_path):
        # Two households, SAME timestamp, different denominators so the
        # weighted ratio differs from the mean of per-household ratios.
        ts = pd.Timestamp("2024-01-01 00:00")
        pd.DataFrame(
            {
                "HuisIdBSV": [1, 2],
                "ProjectIdBSV": [1, 1],
                "ReadingDate": [ts, ts],
                "Zelfgebruik": pd.array([1.0, 1.0], dtype="Float64"),
                "ZonopwekBruto": pd.array([2.0, 10.0], dtype="Float64"),
                _ZP: pd.array([50.0, 10.0], dtype="Float64"),  # per-HH ratios
            }
        ).to_parquet(str(tmp_path / "household_5min.parquet"))

        old = etdtransform.options.aggregate_folder_path
        etdtransform.options.aggregate_folder_path = tmp_path
        try:
            with patch("etdmap.data_model.get_aggregation_config", return_value=_derive_config()), \
                 patch("etdmap.catalog.load_catalog", return_value=_synth_catalog()):
                aggregate_project_data_duckdb(intervals=("5min",))
        finally:
            etdtransform.options.aggregate_folder_path = old

        prj = _read(str(tmp_path / "project_5min.parquet"))
        # components avg across HH: Zelf=1, Zon=6 -> 100*1/6 = 16.667
        # (== sum/sum; the /N cancels). NOT the mean of per-HH ratios (= 30).
        assert float(prj[_ZP].iloc[0]) == pytest.approx(100.0 / 6.0, rel=1e-6)
        assert float(prj[_ZP].iloc[0]) != pytest.approx(30.0, rel=1e-3)

        prov = _read(str(tmp_path / "derive_provenance_project.parquet"))
        assert "project_5min" in set(prov["scope"])
        assert (prov["variable"] == _ZP).all()


# ---------------------------------------------------------------------------
# Provenance helpers (calculated-stage)
# ---------------------------------------------------------------------------

class TestProvenanceHelpers:
    def test_catalog_hash_is_order_independent_and_content_sensitive(self):
        c = pd.DataFrame({"lhs": ["A", "B"], "rhs_text": ["X+Y", "Z"]})
        c_reordered = c.iloc[::-1].reset_index(drop=True)
        assert _catalog_hash(c) == _catalog_hash(c_reordered)  # sorted internally
        c_changed = pd.DataFrame({"lhs": ["A", "B"], "rhs_text": ["X+Y", "W"]})
        assert _catalog_hash(c) != _catalog_hash(c_changed)

    def test_write_calc_provenance_roundtrip(self, tmp_path):
        out = str(tmp_path / "household_calculated.parquet")  # only dirname is used
        plan_rows = [
            {"schema_group_id": 0, "variable": _ZP,
             "rhs_text": "100*Zelfgebruik/ZonopwekBruto",
             "rhs_vars": "Zelfgebruik,ZonopwekBruto"},
        ]
        hh_rows = [
            {"HuisIdBSV": 1, "schema_group_id": 0},
            {"HuisIdBSV": 2, "schema_group_id": 0},
        ]
        _write_calc_provenance(out, plan_rows, hh_rows, "cafef00d")

        plan = _read(str(tmp_path / "derivation_plan.parquet"))
        hh = _read(str(tmp_path / "household_derivation.parquet"))
        assert list(plan["variable"]) == [_ZP]
        assert (plan["catalog_hash"] == "cafef00d").all()
        assert set(int(x) for x in hh["HuisIdBSV"]) == {1, 2}
        assert (hh["schema_group_id"] == 0).all()
