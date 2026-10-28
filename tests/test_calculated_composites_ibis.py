"""The Netto composites, through the stage production runs.

The ibis calculated stage derives nothing by hand: it evaluates whatever the
real catalog declares, choosing the rule with the fewest inputs. So the value
of a composite is the product of two things -- the catalog's formula and
``sympy_to_ibis``'s evaluation -- and only an end-to-end assertion against the
REAL catalog covers both. ``test_calculated_columns_ibis.py`` already pins
``sympy_to_ibis`` on synthetic formulas; this file pins the composites the
pipeline actually publishes.

The expectations are arithmetic on the fixture, computed here rather than taken
from a second implementation. ``test_the_catalog_still_declares_these_rules``
asserts the formulas the arithmetic assumes, so a catalog change surfaces as a
failure naming the rule rather than as a wrong number.

Fixtures are small synthetic frames; never real data.
"""

import pandas as pd
import pytest

from etdtransform.aggregate import add_calculated_columns_to_hh_data_ibis

pytestmark = pytest.mark.integration

N = 4

#: The combined tariff roots, delivered at mapping by combine_tariff_registers.
TERUGLEVERING_ROOT = "ElektriciteitTerugleveringDiff"
NETGEBRUIK_ROOT = "ElektriciteitNetgebruikDiff"

COMPOSITES = [
    "TerugleveringTotaalNetto",
    "ElektriciteitsgebruikTotaalNetto",
    "Netuitwisseling",
]


def _frame(teruglevering, netgebruik):
    """A frame the stage will accept: both halves of every cumulative/interval
    pair present.

    The stage asserts the pairing (``assert_cumulative_interval_pairs_complete``),
    because a set holding one half means the mapping conversion did not happen
    and every later stage would read the missing half as the variable being
    absent. ``dataset_frame`` derives the cumulative counterparts from the
    model, so the fixture cannot drift from the pairing the guard enforces.
    """
    from conftest import dataset_frame

    return dataset_frame(
        n_rows=N,
        **{TERUGLEVERING_ROOT: teruglevering, NETGEBRUIK_ROOT: netgebruik},
    )


def _derive(df, tmp_path):
    """Run the production calculated stage over *df* and return the output.

    Source and output are hive trees, which is the only shape the stage
    writes.
    """
    from conftest import write_shard_tree
    from etdmap.storage import read_source_frame

    src = write_shard_tree(tmp_path / "household_imputed", {1: df})
    out = str(tmp_path / "household_calculated")
    add_calculated_columns_to_hh_data_ibis(
        source_path=str(src), output_path=out, target_columns=COMPOSITES,
    )
    return read_source_frame(out)


def _values(out, column):
    """The column as a list, None where null -- so a propagated null reads as a
    distinct outcome and cannot be mistaken for a computed zero."""
    return [None if pd.isna(v) else float(v) for v in out[column]]


def test_the_catalog_still_declares_these_rules():
    """The arithmetic below assumes the cheapest rule for each composite is the
    root (or the difference of roots). Asserted against the catalog, so a
    change to a formula fails here naming the rule, rather than silently
    changing what the value assertions mean."""
    from etdmap.catalog import load_catalog

    catalog = load_catalog()

    def cheapest(lhs):
        rules = catalog[catalog["lhs"] == lhs].sort_values("rhs_var_count")
        assert not rules.empty, f"the catalog declares no rule for {lhs}"
        return rules.iloc[0]

    assert cheapest("TerugleveringTotaalNetto")["rhs_text"] == TERUGLEVERING_ROOT
    assert cheapest("ElektriciteitsgebruikTotaalNetto")["rhs_text"] == NETGEBRUIK_ROOT

    # Netuitwisseling is a two-input rule either way round; what matters is
    # that it is the difference of the two roots and nothing else.
    netuitwisseling = catalog[
        (catalog["lhs"] == "Netuitwisseling") & (catalog["rhs_var_count"] == 2)
    ]
    assert any(
        set(r["rhs_vars"]) == {TERUGLEVERING_ROOT, NETGEBRUIK_ROOT}
        for _, r in netuitwisseling.iterrows()
    ), "no two-root rule for Netuitwisseling"


class TestCompositesFromTheRoots:
    def test_each_composite_takes_its_value_from_the_roots(self, tmp_path):
        # teruglevering 2.0, netgebruik 3.0 -> exchange is 3.0 - 2.0 = 1.0
        out = _derive(_frame([2.0] * N, [3.0] * N), tmp_path)
        assert _values(out, "TerugleveringTotaalNetto") == [2.0] * N
        assert _values(out, "ElektriciteitsgebruikTotaalNetto") == [3.0] * N
        assert _values(out, "Netuitwisseling") == [1.0] * N

    def test_the_exchange_is_signed(self, tmp_path):
        # More exported than imported leaves a negative exchange, which is a
        # real reading and not an error: 1.0 - 4.0 = -3.0.
        out = _derive(_frame([4.0] * N, [1.0] * N), tmp_path)
        assert _values(out, "Netuitwisseling") == [-3.0] * N


class TestAMissingRootPropagates:
    """A missing root yields NA, never a silent zero. Treating the absent root
    as zero would publish a composite that reads as a measured value, which is
    the defect the root-based derivation exists to avoid."""

    def test_a_missing_root_cell_leaves_its_own_composite_null(self, tmp_path):
        out = _derive(_frame([2.0, pd.NA, 2.0, 2.0], [3.0] * N), tmp_path)
        assert _values(out, "TerugleveringTotaalNetto") == [2.0, None, 2.0, 2.0]

    def test_a_missing_root_cell_leaves_the_exchange_null(self, tmp_path):
        # The exchange needs both roots, so a gap in either one is a gap in it.
        out = _derive(_frame([2.0, pd.NA, 2.0, 2.0], [3.0] * N), tmp_path)
        assert _values(out, "Netuitwisseling") == [1.0, None, 1.0, 1.0]

    def test_the_other_composite_is_unaffected(self, tmp_path):
        # Strict propagation is per cell and per formula: the netgebruik root
        # is present throughout, so its own total is complete.
        out = _derive(_frame([2.0, pd.NA, 2.0, 2.0], [3.0] * N), tmp_path)
        assert _values(out, "ElektriciteitsgebruikTotaalNetto") == [3.0] * N

    def test_a_gap_in_each_root_gaps_only_its_own_rows(self, tmp_path):
        out = _derive(
            _frame([2.0, pd.NA, 2.0, 2.0], [3.0, 3.0, pd.NA, 3.0]), tmp_path
        )
        assert _values(out, "TerugleveringTotaalNetto") == [2.0, None, 2.0, 2.0]
        assert _values(out, "ElektriciteitsgebruikTotaalNetto") == [3.0, 3.0, None, 3.0]
        assert _values(out, "Netuitwisseling") == [1.0, None, None, 1.0]
