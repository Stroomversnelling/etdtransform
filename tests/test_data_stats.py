"""Schema parity tests for the wide-parquet stats producer
(parent ADR-018).

Asserts that ``etdtransform.data_stats.get_data_stats`` (the
ibis-on-DuckDB wide-parquet stats path) emits the same column set and
schema-compliant values that the per-Series kernel
``etdmap.compute_numeric_column_stats`` produces. Drift between the
two paths breaks the project's ADR-018 contract and any downstream
report that consumes both.

Lives in etdtransform tests because etdtransform owns the function
under test (and the file imports etdmap and ibis, both of which are
already test deps for the rest of the suite).
"""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from etdmap.mapping_helpers import (
    NUMERIC_STATS_SCHEMA,
    _STATS_DTYPES,
    compute_numeric_column_stats,
)

from etdtransform.data_stats import get_data_stats


def _build_fixture_parquet(tmp_dir: str) -> str:
    """
    Two households + one all-NA household, one full year of monthly
    rows, one numeric col (`kWh`) and one boolean col (`validate_ok`).

    Layout choices:
      * HH 1 has data for every month in the year.
      * HH 2 has data only for the cold months (Oct-Apr / months 10-4).
      * HH 3 has no kWh data at all (tests count=0 cross-join padding)
        but does have boolean data.
    """
    rows = []
    cold_months = {10, 11, 12, 1, 2, 3, 4}
    for hh, mode in [(1, "full"), (2, "cold_only"), (3, "no_kwh")]:
        for month in range(1, 13):
            ts = pd.Timestamp(f"2024-{month:02d}-15", tz="UTC")
            if mode == "full":
                kwh = float(month) * 10.0
                ok = True
            elif mode == "cold_only":
                kwh = float(month) * 10.0 if month in cold_months else None
                ok = month in cold_months
            else:  # no_kwh
                kwh = None
                ok = False
            rows.append({
                "HuisIdBSV": hh,
                "ReadingDate": ts,
                "kWh": kwh,
                "validate_ok": ok,
            })
    df = pd.DataFrame(rows)
    path = os.path.join(tmp_dir, "fixture.parquet")
    pq.write_table(pa.Table.from_pandas(df), path)
    return path


@pytest.fixture(scope="module")
def fixture_parquet():
    """Build the fixture once per test module."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _build_fixture_parquet(tmp)
        yield path


class TestSchemaContract:
    """get_data_stats output column set + dtypes match _STATS_DTYPES."""

    def test_output_columns_are_superset_of_stats_dtypes(self, fixture_parquet):
        """Every key in etdmap._STATS_DTYPES is present in the output,
        plus the wide-stage extras (Identifier, HuisIdBSV, source_file,
        quantile_mode)."""
        out = get_data_stats(fixture_parquet, seasonal=False)
        missing = set(_STATS_DTYPES) - set(out.columns)
        assert not missing, (
            f"get_data_stats missing schema columns: {sorted(missing)}"
        )
        for extra in ("Identifier", "HuisIdBSV", "source_file", "quantile_mode"):
            assert extra in out.columns, (
                f"wide-stage extra column '{extra}' missing from output"
            )

    def test_dtypes_match_stats_dtypes_contract(self, fixture_parquet):
        """Every column listed in _STATS_DTYPES has the contract dtype.
        Drift here breaks CSV round-trip for downstream consumers."""
        out = get_data_stats(fixture_parquet, seasonal=False)
        for col, expected in _STATS_DTYPES.items():
            if col not in out.columns:
                continue
            actual = str(out[col].dtype)
            assert actual == expected, (
                f"dtype drift on '{col}': expected {expected}, got {actual}"
            )

    def test_numeric_stats_schema_columns_present(self, fixture_parquet):
        """NUMERIC_STATS_SCHEMA columns (the per-Series kernel's keys)
        all appear in the wide-stage output for parity with the
        mapped-mode path."""
        out = get_data_stats(fixture_parquet, seasonal=False)
        for col in NUMERIC_STATS_SCHEMA:
            assert col in out.columns, f"NUMERIC_STATS_SCHEMA col '{col}' missing"


class TestValueParityExactPrecision:
    """precision='exact' values match the per-Series kernel
    compute_numeric_column_stats. This is the byte-equal contract
    that cross-stage drift reports rely on."""

    def test_exact_quantile_matches_kernel_for_single_hh(self, fixture_parquet):
        """For HH=1, column=kWh, annual: get_data_stats(exact) row
        values match compute_numeric_column_stats on the same series."""
        df = pd.read_parquet(fixture_parquet, columns=["HuisIdBSV", "kWh"])
        sub = df[df["HuisIdBSV"] == 1]["kWh"].dropna()
        sub_nullable = pd.array(sub.tolist(), dtype="Float64")
        kernel = compute_numeric_column_stats(pd.Series(sub_nullable))

        out = get_data_stats(fixture_parquet, seasonal=False, precision="exact")
        row = out[
            (out["HuisIdBSV"] == 1)
            & (out["column"] == "kWh")
            & (out["season"] == "annual")
        ].iloc[0]

        for key in NUMERIC_STATS_SCHEMA:
            kv = kernel[key]
            tv = row[key]
            if pd.isna(kv) and pd.isna(tv):
                continue
            assert float(tv) == pytest.approx(float(kv), rel=1e-9, abs=1e-9), (
                f"get_data_stats(exact) '{key}' diverged from kernel: "
                f"got {tv}, kernel {kv}"
            )

    def test_quantile_mode_column_records_precision(self, fixture_parquet):
        """The quantile_mode column tags every row with the precision
        the producer used so downstream consumers can detect approx
        values."""
        out_exact = get_data_stats(fixture_parquet, seasonal=False, precision="exact")
        out_approx = get_data_stats(fixture_parquet, seasonal=False, precision="approx")
        assert (out_exact["quantile_mode"] == "exact").all()
        assert (out_approx["quantile_mode"] == "approx").all()


class TestCrossJoinPadding:
    """The cross-product (HH x col x season) is reinstated after the
    ibis aggregation -- the mapped-mode kernel emits a count=0 row
    for an all-NA column, and the wide-stage path must too."""

    def test_zero_count_row_for_hh_with_no_column_data(self, fixture_parquet):
        """HH=3 has no kWh data. Output must contain a row for
        (HH=3, kWh, annual) with count=0, missing>0, and NA stat
        values -- not a missing row."""
        out = get_data_stats(fixture_parquet, seasonal=False)
        sub = out[
            (out["HuisIdBSV"] == 3)
            & (out["column"] == "kWh")
            & (out["season"] == "annual")
        ]
        assert len(sub) == 1, "expected exactly one row for HH=3, kWh, annual"
        row = sub.iloc[0]
        assert int(row["count"]) == 0
        assert int(row["missing"]) > 0
        for stat in ("mean", "std", "min", "max", "median"):
            assert pd.isna(row[stat]), f"{stat} should be NA when count=0"

    def test_seasonal_padding_covers_all_seasons(self, fixture_parquet):
        """With seasonal=True every (HH, col) combo gets a row per
        season slice -- annual, cold, warm -- even when the HH has no
        data for that slice."""
        out = get_data_stats(fixture_parquet, seasonal=True)
        # HH=2 has data only in cold months; warm slice should still
        # emit a row but with count=0.
        warm = out[
            (out["HuisIdBSV"] == 2)
            & (out["column"] == "kWh")
            & (out["season"] == "warm")
        ]
        assert len(warm) == 1
        assert int(warm.iloc[0]["count"]) == 0


class TestBooleanBranch:
    """Boolean cols route through the cheap-reductions-only path:
    count / mean / min / max populate; std / median / quantiles / iqr
    stay NA. This mirrors the mapped-mode kernel's boolean branch."""

    def test_boolean_emits_only_cheap_stats(self, fixture_parquet):
        """For HH=1, column=validate_ok, annual: count/mean/min/max
        are populated, quantile stats are NA."""
        out = get_data_stats(fixture_parquet, seasonal=False)
        sub = out[
            (out["HuisIdBSV"] == 1)
            & (out["column"] == "validate_ok")
            & (out["season"] == "annual")
        ].iloc[0]
        assert int(sub["count"]) == 12
        # All True for HH=1 -> mean / min / max all 1.0
        assert float(sub["mean"]) == pytest.approx(1.0)
        assert float(sub["min"]) == pytest.approx(1.0)
        assert float(sub["max"]) == pytest.approx(1.0)
        # Quantile stats are NA on boolean columns (kernel parity).
        for stat in ("std", "median", "p01", "p25", "p75", "p99", "iqr"):
            assert pd.isna(sub[stat]), f"{stat} should be NA on a boolean col"

    def test_boolean_type_string_is_recorded(self, fixture_parquet):
        """The 'type' column reflects the parquet's stored dtype
        ('bool' for booleans, not the float64 used for the lazy
        aggregation)."""
        out = get_data_stats(fixture_parquet, seasonal=False)
        sub = out[out["column"] == "validate_ok"].iloc[0]
        assert "bool" in str(sub["type"]).lower(), (
            f"expected boolean type in 'type' col, got {sub['type']!r}"
        )
