"""
Tests for add_calculated_columns_adaptive() and DatasetAdapter.available_columns().

All fixtures use pandas nullable types (Float64 / Int64 / pd.NA) throughout,
matching the project convention that missingness is represented by pd.NA, not NaN.

Inline catalog DataFrames are used for unit tests -- no real parquet files needed.
Tests that need the real catalog use the `real_catalog` fixture, which fails
when catalog.parquet is absent. The catalog ships inside etdmap.
"""

import logging

import pandas as pd
import pytest

from etdtransform.catalog.query import DatasetAdapter
from etdtransform.calculated_columns import (
    add_calculated_columns_adaptive,
    add_calculated_columns_imputed_data,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_catalog(*rules):
    """Build a minimal catalog DataFrame from (lhs, rhs_text, rhs_vars) tuples."""
    rows = []
    for lhs, rhs_text, rhs_vars in rules:
        rows.append({
            "lhs": lhs,
            "rhs_text": rhs_text,
            "rhs_vars": rhs_vars,
            "rhs_var_count": len(rhs_vars),
        })
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=["lhs", "rhs_text", "rhs_vars", "rhs_var_count"])


def _full_col(n=100, value=1.0, dtype="Float64"):
    """Series that is 100% non-null, using pandas nullable type."""
    return pd.array([value] * n, dtype=dtype)


def _sparse_col(n=100, frac=0.5, value=1.0):
    """Series with `frac` non-null fraction using pd.NA (not NaN)."""
    data = [value] * int(n * frac) + [pd.NA] * (n - int(n * frac))
    return pd.array(data, dtype="Float64")


def _make_df(cols: dict, n=100, household_id=1) -> pd.DataFrame:
    """
    Build a DataFrame with HuisIdBSV and the given columns.
    Values in cols must already be array-like (use _full_col / _sparse_col).
    """
    data = {"HuisIdBSV": pd.array([household_id] * n, dtype="Int64")}
    data.update(cols)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# DatasetAdapter.available_columns
# ---------------------------------------------------------------------------

class TestAvailableColumns:
    def test_above_threshold_included(self):
        df = _make_df({"A": _full_col(), "B": _sparse_col(frac=0.96)})
        adapter = DatasetAdapter(_make_catalog(), completeness_threshold=0.95)
        assert "A" in adapter.available_columns(df)
        assert "B" in adapter.available_columns(df)

    def test_below_threshold_excluded(self):
        df = _make_df({"A": _sparse_col(frac=0.80)})
        adapter = DatasetAdapter(_make_catalog(), completeness_threshold=0.95)
        assert "A" not in adapter.available_columns(df)

    def test_exactly_at_threshold_included(self):
        n = 100
        data = [1.0] * 95 + [pd.NA] * 5
        df = _make_df({"A": pd.array(data, dtype="Float64")}, n=n)
        adapter = DatasetAdapter(_make_catalog(), completeness_threshold=0.95)
        assert "A" in adapter.available_columns(df)

    def test_pdna_counted_as_missing(self):
        """pd.NA values must be counted as missing (not as 0 or some other value)."""
        n = 100
        # 60 present, 40 pd.NA -> fraction 0.60 < 0.95
        data = [1.0] * 60 + [pd.NA] * 40
        df = _make_df({"A": pd.array(data, dtype="Float64")}, n=n)
        adapter = DatasetAdapter(_make_catalog(), completeness_threshold=0.95)
        assert "A" not in adapter.available_columns(df)


# ---------------------------------------------------------------------------
# add_calculated_columns_adaptive -- derivation correctness
# ---------------------------------------------------------------------------

class TestAdaptiveDerivation:
    def _minimal_setup(self):
        """Catalog with one rule: C = A + B. df has A and B."""
        catalog = _make_catalog(("C", "A + B", ["A", "B"]))
        df = _make_df({"A": _full_col(value=2.0), "B": _full_col(value=3.0)})
        return df, catalog

    def test_derives_missing_column(self):
        df, catalog = self._minimal_setup()
        result = add_calculated_columns_adaptive(df, catalog, target_columns={"C"})
        assert "C" in result.columns
        assert (result["C"] == 5.0).all()

    def test_available_column_not_overwritten(self):
        catalog = _make_catalog(("C", "A + B", ["A", "B"]))
        df = _make_df({
            "A": _full_col(value=2.0),
            "B": _full_col(value=3.0),
            "C": _full_col(value=99.0),
        })
        result = add_calculated_columns_adaptive(df, catalog, target_columns={"C"})
        assert (result["C"] == 99.0).all(), "Available column must not be overwritten"

    def test_cascade_derivation(self):
        """D = B + C where C = A + B (two-step cascade)."""
        catalog = _make_catalog(
            ("C", "A + B", ["A", "B"]),
            ("D", "B + C", ["B", "C"]),
        )
        df = _make_df({"A": _full_col(value=1.0), "B": _full_col(value=2.0)})
        result = add_calculated_columns_adaptive(df, catalog, target_columns={"C", "D"})
        assert "C" in result.columns
        assert "D" in result.columns
        assert (result["C"] == 3.0).all()  # 1 + 2
        assert (result["D"] == 5.0).all()  # 2 + 3

    def test_pdna_rhs_input_propagates_by_default(self):
        """pd.NA in an input column yields pd.NA in the derived column.

        A derived value whose input is unknown is unknown. Treating the
        missing input as a measured zero is what produced negative gross
        consumption for households whose solar reading was absent while their
        export register advanced.
        """
        n = 100
        # A is 99% non-null (above threshold) so it counts as available
        a_vals = [1.0] * (n - 1) + [pd.NA]
        catalog = _make_catalog(("C", "A + B", ["A", "B"]))
        df = _make_df({
            "A": pd.array(a_vals, dtype="Float64"),
            "B": pd.array([1.0] * n, dtype="Float64"),
        })
        result = add_calculated_columns_adaptive(df, catalog, target_columns={"C"})
        assert "C" in result.columns
        assert pd.isna(result["C"].iloc[-1]), (
            "pd.NA in A must reach C rather than being read as a zero"
        )
        # The rows with a value are unaffected.
        assert float(result["C"].iloc[0]) == pytest.approx(2.0)

    def test_fillna_true_still_substitutes_zero(self):
        """The old behaviour remains reachable by asking for it explicitly."""
        n = 100
        a_vals = [1.0] * (n - 1) + [pd.NA]
        catalog = _make_catalog(("C", "A + B", ["A", "B"]))
        df = _make_df({
            "A": pd.array(a_vals, dtype="Float64"),
            "B": pd.array([1.0] * n, dtype="Float64"),
        })
        result = add_calculated_columns_adaptive(
            df, catalog, target_columns={"C"}, fillna=True
        )
        assert float(result["C"].iloc[-1]) == pytest.approx(1.0)

    def test_multi_household_plan_cached(self):
        """Two households with identical available columns share one execution plan."""
        catalog = _make_catalog(("C", "A + B", ["A", "B"]))
        df = pd.concat([
            _make_df({"A": _full_col(value=1.0), "B": _full_col(value=2.0)}, household_id=1),
            _make_df({"A": _full_col(value=3.0), "B": _full_col(value=4.0)}, household_id=2),
        ], ignore_index=True)
        result = add_calculated_columns_adaptive(df, catalog, target_columns={"C"})
        assert "C" in result.columns
        hh1 = result[result["HuisIdBSV"] == 1]["C"]
        hh2 = result[result["HuisIdBSV"] == 2]["C"]
        assert (hh1 == 3.0).all()
        assert (hh2 == 7.0).all()


# ---------------------------------------------------------------------------
# PrestatiedataBerekend columns appear in target set
# ---------------------------------------------------------------------------

class TestPrestatiedataBerekendInTargets:
    def test_berekend_columns_are_attempted(self):
        from etdmap.data_model import all_performance_data_columns
        from etdmap.catalog import load_catalog

        catalog_df = load_catalog()

        from etdmap.data_model import model_column_order
        catalog_lhs = set(catalog_df["lhs"].dropna().unique())
        in_all = catalog_lhs & set(all_performance_data_columns)

        assert in_all == catalog_lhs, (
            f"Catalog targets not in all_performance_data_columns: {catalog_lhs - in_all}"
        )


# ---------------------------------------------------------------------------
# Completeness reporting
# ---------------------------------------------------------------------------

class TestCompletenessReporting:
    def test_not_derivable_logs_error(self, caplog):
        """A column that cannot be derived due to missing inputs logs an error."""
        catalog = _make_catalog(("C", "A + B", ["A", "B"]))
        # df has A but not B -- C cannot be derived
        df = _make_df({"A": _full_col(value=1.0)})
        with caplog.at_level(logging.ERROR, logger="root"):
            add_calculated_columns_adaptive(df, catalog, target_columns={"C"})
        assert any("could not derive" in r.message.lower() for r in caplog.records), (
            "Expected logging.error for not_derivable columns"
        )

    def test_missing_required_logs_error(self, caplog):
        """If a Vereist=ja column is absent after derivation, an error is logged."""
        from etdmap.data_model import required_performance_data_columns
        # An empty required-column set is a data-model defect, not a reason to
        # stand this test down: with nothing required, the error path under
        # test cannot fire and a skip would hide both problems at once.
        assert required_performance_data_columns, (
            "etdmodel.csv defines no Vereist=ja performance data columns. "
            "Fix the data model (sync from Grist) -- this test cannot verify "
            "the missing-required-column error path without them."
        )

        catalog = _make_catalog()
        df = _make_df({"SomeIrrelevantCol": _full_col()})
        with caplog.at_level(logging.ERROR, logger="root"):
            add_calculated_columns_adaptive(df, catalog, target_columns=set())
        assert any("Required columns" in r.message for r in caplog.records), (
            "Expected logging.error for missing required columns"
        )

    def test_no_error_when_all_targets_available(self, caplog):
        catalog = _make_catalog(("C", "A + B", ["A", "B"]))
        df = _make_df({"A": _full_col(), "B": _full_col(), "C": _full_col()})
        with caplog.at_level(logging.ERROR, logger="root"):
            add_calculated_columns_adaptive(df, catalog, target_columns={"C"})
        derive_errors = [
            r for r in caplog.records
            if r.levelno >= logging.ERROR and "Could not derive" in r.message
        ]
        assert not derive_errors


# ---------------------------------------------------------------------------
# Execution plan completeness -- no model columns silently skipped
# ---------------------------------------------------------------------------

class TestNoSilentlySkippedColumns:
    def test_berekend_columns_targeted_and_derived_from_available_seed(self):
        from etdmap.catalog import load_catalog

        catalog_df = load_catalog()

        seed_col = "ElektriciteitsgebruikTotaalNetto"
        expected_derived = "ElektriciteitNetgebruikDiff"

        df = _make_df({seed_col: _full_col()})
        initial_cols = set(df.columns)

        result = add_calculated_columns_adaptive(df, catalog_df)

        newly_derived = set(result.columns) - initial_cols
        assert expected_derived in newly_derived, (
            f"Expected '{expected_derived}' to be derived from '{seed_col}', "
            f"but derived set was: {sorted(newly_derived)}"
        )


# ---------------------------------------------------------------------------
# Standard path -- adaptive matches old (non-adaptive) approach
# ---------------------------------------------------------------------------

# Raw Diff columns that feed the old add_calculated_columns_imputed_data approach.
# ZonopwekBruto is used directly here (post-shim name) to avoid the rename shim
# from interfering with the column-naming comparison.
_RAW_DIFF_COLS = [
    "ElektriciteitTerugleveringLaagDiff",
    "ElektriciteitTerugleveringHoogDiff",
    "ElektriciteitNetgebruikLaagDiff",
    "ElektriciteitNetgebruikHoogDiff",
    "ElektriciteitsgebruikWarmtepompDiff",
    "ElektriciteitsgebruikBoosterDiff",
    "ElektriciteitsgebruikBoilervatDiff",
    "ElektriciteitsgebruikWTWDiff",
    "ElektriciteitsgebruikRadiatorDiff",
    "ZonopwekBruto",
]

# The combined tariff roots the mapping stage delivers (combine_tariff_registers):
# root = Hoog + Laag wherever both splits are delivered. The legacy calculated
# path reads the roots for the Netto composites since 2026-10-01.
_ROOT_DIFF_COLS = [
    "ElektriciteitTerugleveringDiff",
    "ElektriciteitNetgebruikDiff",
]

_DERIVED_COLS = [
    "TerugleveringTotaalNetto",
    "ElektriciteitsgebruikTotaalNetto",
    "Netuitwisseling",
    "ElektriciteitsgebruikTotaalWarmtepomp",
    "ElektriciteitsgebruikTotaalGebouwgebonden",
    "ElektriciteitsgebruikTotaalHuishoudelijk",
    "Zelfgebruik",
    "ElektriciteitsgebruikTotaalBruto",
]


def _standard_raw_df(n=50, household_id=1) -> pd.DataFrame:
    """A DataFrame with all raw Diff cols at 1.0 (Float64) for one household,
    plus the combined tariff roots at 2.0 -- root = Hoog + Laag, exactly what
    the mapping combination delivers, so the legacy (root-based) and adaptive
    paths agree numerically."""
    cols = {c: _full_col(n=n, value=1.0) for c in _RAW_DIFF_COLS}
    cols.update({c: _full_col(n=n, value=2.0) for c in _ROOT_DIFF_COLS})
    return _make_df(cols, n=n, household_id=household_id)


@pytest.fixture(scope="module")
def real_catalog():
    """
    The real derivation catalog. Missing it is a hard failure.

    catalog.parquet is committed to etdmap and ships as package data
    (etdmap/data/catalog.parquet). If load_catalog() cannot find it, etdmap is
    broken or mis-installed.
    """
    from etdmap.catalog import load_catalog
    try:
        return load_catalog()
    except FileNotFoundError as exc:
        raise RuntimeError(
            "etdmap is installed without its bundled catalog.parquet. "
            "Reinstall it (pip install ../etdmap) -- do not skip this test. "
            "Original error: %s" % exc
        ) from exc


class TestStandardPathMatchesOldApproach:
    def test_derived_columns_present(self, real_catalog):
        df = _standard_raw_df()
        result = add_calculated_columns_adaptive(df, real_catalog)
        for col in _DERIVED_COLS:
            assert col in result.columns, f"Expected derived column missing: {col}"

    def test_derived_values_match_old_approach(self, real_catalog):
        """Adaptive on full raw data must produce the same numeric values as the old approach."""
        df_adaptive = _standard_raw_df()
        result_adaptive = add_calculated_columns_adaptive(df_adaptive, real_catalog)

        # Old approach requires Zon-opwekTotaalDiff (it renames internally to ZonopwekBruto).
        # Build an equivalent df using the old column name.
        df_old = _standard_raw_df().rename(columns={"ZonopwekBruto": "Zon-opwekTotaalDiff"})
        # Drop HuisIdBSV -- old approach doesn't need it
        df_old = df_old.drop(columns=["HuisIdBSV"])
        result_old = add_calculated_columns_imputed_data(df_old, fillna=True)

        for col in _DERIVED_COLS:
            assert col in result_adaptive.columns, f"Adaptive missing: {col}"
            assert col in result_old.columns, f"Old approach missing: {col}"
            pd.testing.assert_series_equal(
                result_adaptive[col].reset_index(drop=True).astype(float),
                result_old[col].reset_index(drop=True).astype(float),
                check_names=False,
                rtol=1e-9,
                obj=f"Column {col}",
            )


# ---------------------------------------------------------------------------
# Occlusion tests -- back-calculation via catalog rules
# ---------------------------------------------------------------------------

class TestOcclusionBackCalculation:
    """
    Back-calculation tests: simulate a provider who supplies an aggregated quantity
    directly instead of its individual components.

    Unit tests use an inline catalog with only the specific rule under test, making
    the derivation path deterministic.  Integration tests use the real catalog and
    verify only the column is derived (not the exact rule path used).

    Known back-calculation identity (the vehicle for the integration tests):
        ElektriciteitsgebruikBoilervatDiff
            = ElektriciteitsgebruikTotaalWarmtepomp
              - ElektriciteitsgebruikWarmtepompDiff
              - ElektriciteitsgebruikBoosterDiff

    The integration tests used a TARIFF SPLIT
    (ElektriciteitTerugleveringLaagDiff = TerugleveringTotaalNetto - HoogDiff)
    until 2026-10-01. The tariff-sum rules are bookkeeping identities, not
    physical relations, so the not_physical exclusion removed them from the
    catalog and a split is no longer a catalog target at all -- the mapping
    stage combines the registers instead. The behaviour under test is
    unchanged (occlude a component, recover it from its total, let the
    recovery feed downstream derivations); only the vehicle moved to a
    heat-pump component, which the catalog genuinely derives. The unit tests
    below keep their inline catalogs and are unaffected by what the real
    catalog holds.
    """

    # -- Unit: inline catalog, deterministic rule path --

    def test_single_column_occlusion_inline_catalog(self):
        """
        Using an inline catalog with only the back-calculation rule:
            LaagDiff = Totaal - HoogDiff

        With Totaal=5.0 and HoogDiff=3.0, LaagDiff must be derived as 2.0.
        The occluded column is all pd.NA so the adapter does not treat it as available.
        """
        n = 50
        catalog = _make_catalog(
            (
                "ElektriciteitTerugleveringLaagDiff",
                "TerugleveringTotaalNetto - ElektriciteitTerugleveringHoogDiff",
                ["TerugleveringTotaalNetto", "ElektriciteitTerugleveringHoogDiff"],
            )
        )
        df = _make_df(
            {
                "ElektriciteitTerugleveringHoogDiff": _full_col(n=n, value=3.0),
                "TerugleveringTotaalNetto": _full_col(n=n, value=5.0),
                "ElektriciteitTerugleveringLaagDiff": pd.array([pd.NA] * n, dtype="Float64"),
            },
            n=n,
        )
        result = add_calculated_columns_adaptive(
            df, catalog,
            target_columns={"ElektriciteitTerugleveringLaagDiff"},
        )
        assert "ElektriciteitTerugleveringLaagDiff" in result.columns
        derived = result["ElektriciteitTerugleveringLaagDiff"].dropna().astype(float)
        assert derived.tolist() == pytest.approx([2.0] * n), (
            f"Expected back-calculated value 2.0 for all rows, got: {derived.unique().tolist()}"
        )

    def test_two_column_occlusion_cascade_inline_catalog(self):
        """
        Two-step occlusion: LaagDiff and TotaalNetto are both absent.
        Provide only HoogDiff=3.0 and a higher-level aggregate HigherTotal=7.0.
        Catalog rules:
            TotaalNetto = HigherTotal - Correction    (Correction=2.0 in df)
            LaagDiff    = TotaalNetto - HoogDiff
        Both must be derived in topological order: TotaalNetto first, then LaagDiff.
        """
        n = 30
        catalog = _make_catalog(
            (
                "TerugleveringTotaalNetto",
                "HigherTotal - Correction",
                ["HigherTotal", "Correction"],
            ),
            (
                "ElektriciteitTerugleveringLaagDiff",
                "TerugleveringTotaalNetto - ElektriciteitTerugleveringHoogDiff",
                ["TerugleveringTotaalNetto", "ElektriciteitTerugleveringHoogDiff"],
            ),
        )
        df = _make_df(
            {
                "HigherTotal": _full_col(n=n, value=7.0),
                "Correction": _full_col(n=n, value=2.0),  # TotaalNetto = 5.0
                "ElektriciteitTerugleveringHoogDiff": _full_col(n=n, value=3.0),
                "TerugleveringTotaalNetto": pd.array([pd.NA] * n, dtype="Float64"),
                "ElektriciteitTerugleveringLaagDiff": pd.array([pd.NA] * n, dtype="Float64"),
            },
            n=n,
        )
        result = add_calculated_columns_adaptive(
            df, catalog,
            target_columns={"TerugleveringTotaalNetto", "ElektriciteitTerugleveringLaagDiff"},
        )
        totaal = result["TerugleveringTotaalNetto"].dropna().astype(float)
        laag = result["ElektriciteitTerugleveringLaagDiff"].dropna().astype(float)
        assert totaal.tolist() == pytest.approx([5.0] * n), "TotaalNetto should be 5.0"
        assert laag.tolist() == pytest.approx([2.0] * n), "LaagDiff should be 2.0"

    # -- Integration: real catalog, column presence and value correctness --

    def test_occlusion_column_derived_with_real_catalog(self, real_catalog):
        """
        With the real catalog, occlude ElektriciteitsgebruikBoilervatDiff (all
        pd.NA) and provide ElektriciteitsgebruikTotaalWarmtepomp as direct
        input. The catalog must find a rule that derives the component from
        the available columns.

        Vehicle changed from a tariff split on 2026-10-01 -- see the class
        docstring; the assertion intent is unchanged.
        """
        n = 50
        df = _make_df(
            {
                "ElektriciteitsgebruikTotaalWarmtepomp": _full_col(n=n, value=5.0),
                "ElektriciteitsgebruikBoilervatDiff": pd.array([pd.NA] * n, dtype="Float64"),
                **{c: _full_col(n=n, value=1.0) for c in _RAW_DIFF_COLS
                   if c != "ElektriciteitsgebruikBoilervatDiff"},
                **{c: _full_col(n=n, value=2.0) for c in _ROOT_DIFF_COLS},
            },
            n=n,
        )
        result = add_calculated_columns_adaptive(df, real_catalog)

        assert "ElektriciteitsgebruikBoilervatDiff" in result.columns, (
            "Real catalog must contain a rule to derive "
            "ElektriciteitsgebruikBoilervatDiff"
        )
        assert result["ElektriciteitsgebruikBoilervatDiff"].notna().any(), (
            "Derived ElektriciteitsgebruikBoilervatDiff must have non-null values"
        )

    def test_occluded_column_enables_downstream_derivation(self, real_catalog):
        """After back-calculating the occluded component, downstream summary
        columns must be present.

        Vehicle changed from a tariff split on 2026-10-01 -- see the class
        docstring. The roots are supplied directly, as the mapping
        combination now delivers them.
        """
        n = 50
        df = _make_df(
            {
                "ElektriciteitsgebruikTotaalWarmtepomp": _full_col(n=n, value=5.0),
                "ElektriciteitsgebruikBoilervatDiff": pd.array([pd.NA] * n, dtype="Float64"),
                **{c: _full_col(n=n, value=1.0) for c in _RAW_DIFF_COLS
                   if c != "ElektriciteitsgebruikBoilervatDiff"},
                **{c: _full_col(n=n, value=2.0) for c in _ROOT_DIFF_COLS},
            },
            n=n,
        )
        result = add_calculated_columns_adaptive(df, real_catalog)
        for col in ("Zelfgebruik", "ElektriciteitsgebruikTotaalBruto"):
            assert col in result.columns, (
                f"Downstream column {col!r} missing after back-calculation chain"
            )

    def test_full_loop_adaptive_recovers_occluded_component(self, real_catalog):
        """
        Full-loop test:
        1. Build df with all raw diff cols at value 1.0.
        2. Run old approach -> get all derived cols (incl.
           ElektriciteitsgebruikTotaalWarmtepomp = 3.0, the sum of its three
           components at 1.0 each).
        3. Occlude ElektriciteitsgebruikBoilervatDiff (set to all pd.NA).
        4. Run adaptive -- it now has TotaalWarmtepomp=3.0, WarmtepompDiff=1.0
           and BoosterDiff=1.0 available.
        5. Verify derived BoilervatDiff == 1.0 (the original value:
           3.0 - 1.0 - 1.0 = 1.0).

        Vehicle changed from a tariff split on 2026-10-01 -- see the class
        docstring; the recovered value is still the original 1.0, so the
        assertion is unchanged.
        """
        n = 50
        # Step 1+2: build full df with old approach
        df_old = _standard_raw_df(n=n).rename(
            columns={"ZonopwekBruto": "Zon-opwekTotaalDiff"}
        ).drop(columns=["HuisIdBSV"])
        df_with_derived = add_calculated_columns_imputed_data(df_old.copy(), fillna=True)
        if "Zon-opwekTotaalDiff" in df_with_derived.columns:
            df_with_derived = df_with_derived.rename(
                columns={"Zon-opwekTotaalDiff": "ZonopwekBruto"}
            )
        df_with_derived.insert(0, "HuisIdBSV", pd.array([1] * n, dtype="Int64"))

        # Step 3: occlude the raw component
        df_occluded = df_with_derived.copy()
        df_occluded["ElektriciteitsgebruikBoilervatDiff"] = pd.array(
            [pd.NA] * n, dtype="Float64"
        )

        # Step 4: run adaptive
        result = add_calculated_columns_adaptive(df_occluded, real_catalog)

        # Step 5: verify recovery -- TotaalWarmtepomp=3.0 minus the two
        # delivered components at 1.0 each -> BoilervatDiff=1.0
        assert "ElektriciteitsgebruikBoilervatDiff" in result.columns
        recovered = result["ElektriciteitsgebruikBoilervatDiff"].dropna().astype(float)
        assert recovered.tolist() == pytest.approx([1.0] * n, rel=1e-6), (
            f"Full-loop recovery: expected 1.0 everywhere, got {recovered.unique().tolist()}"
        )
