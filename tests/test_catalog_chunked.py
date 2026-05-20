"""Correctness tests for etdtransform.catalog.chunked.

The chunked builder produces the same catalog (modulo cosmetic sympy
canonical-form differences in rhs_text) as the serial reference. Equality
is checked on the semantic key set:
   { (lhs, frozenset(rhs_vars), tuple(sorted(physical_models))) }.

These tests assert:
  1. parallel == serial for a small linear-only rule set
  2. parallel == serial for a rule set including one non-linear rule
  3. cross-model dedup: rules tagged with multiple models produce
     one row per (lhs, rhs_vars), with the union of model tags
  4. per-rule cache invalidation: changing one non-linear rule only
     rebuilds that rule's chunk
"""
from __future__ import annotations

import pandas as pd
import pytest

from etdtransform.catalog import (
    build_chunked,
    build_serial_reference,
    canonical_linear_hash,
    canonical_nl_chunk_hash,
    classify_linear_nonlinear,
    compose_and_prune,
    discover_models,
    plan_chunked_build,
    verify_against_serial,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def linear_rules_one_model():
    return [
        {
            "lhs": "Zelfgebruik",
            "rhs": "ZonopwekBruto - Terug",
            "rhs_vars": ["ZonopwekBruto", "Terug"],
            "physical_models": ["Universeel"],
        },
        {
            "lhs": "ZonopwekBruto",
            "rhs": "Zelfgebruik + Terug",
            "rhs_vars": ["Zelfgebruik", "Terug"],
            "physical_models": ["Universeel"],
        },
    ]


@pytest.fixture
def rules_with_nonlinear():
    return [
        {
            "lhs": "Zelfgebruik",
            "rhs": "ZonopwekBruto - Terug",
            "rhs_vars": ["ZonopwekBruto", "Terug"],
            "physical_models": ["Universeel"],
        },
        {
            "lhs": "ZelfgebruikPercentage",
            "rhs": "Zelfgebruik / ZonopwekBruto",
            "rhs_vars": ["Zelfgebruik", "ZonopwekBruto"],
            "physical_models": ["Universeel"],
        },
    ]


@pytest.fixture
def rules_two_models():
    """Same shape as rules_with_nonlinear but each rule applies to two
    physical models. After compose+dedup, every derivation should have
    physical_models == sorted({Universeel, Hybride}) where the underlying
    algebra agrees.
    """
    return [
        {
            "lhs": "Zelfgebruik",
            "rhs": "ZonopwekBruto - Terug",
            "rhs_vars": ["ZonopwekBruto", "Terug"],
            "physical_models": ["Universeel", "Hybride"],
        },
        {
            "lhs": "ZelfgebruikPercentage",
            "rhs": "Zelfgebruik / ZonopwekBruto",
            "rhs_vars": ["Zelfgebruik", "ZonopwekBruto"],
            "physical_models": ["Universeel", "Hybride"],
        },
    ]


# ---------------------------------------------------------------------------
# Equality: parallel chunked build == serial reference
# ---------------------------------------------------------------------------

def test_parallel_equals_serial_linear_only(linear_rules_one_model, tmp_path):
    report = verify_against_serial(linear_rules_one_model, cache_dir=tmp_path / "cache")
    assert report["equal_keysets"], (
        f"keysets differ: only_serial={report['only_serial']}, "
        f"only_parallel={report['only_parallel']}"
    )
    assert report["serial_rows"] == report["parallel_rows"]


def test_parallel_equals_serial_with_nonlinear(rules_with_nonlinear, tmp_path):
    report = verify_against_serial(rules_with_nonlinear, cache_dir=tmp_path / "cache")
    assert report["equal_keysets"], (
        f"keysets differ: only_serial={report['only_serial']}, "
        f"only_parallel={report['only_parallel']}"
    )


def test_parallel_equals_serial_two_models(rules_two_models, tmp_path):
    report = verify_against_serial(rules_two_models, cache_dir=tmp_path / "cache")
    assert report["equal_keysets"], (
        f"keysets differ: only_serial={report['only_serial']}, "
        f"only_parallel={report['only_parallel']}"
    )


# ---------------------------------------------------------------------------
# Cross-model dedup
# ---------------------------------------------------------------------------

def test_cross_model_dedup_unions_models(rules_two_models, tmp_path):
    """Each derivation in rules_two_models is valid in both Universeel and
    Hybride. After compose+dedup the catalog should have one row per
    (lhs, rhs_vars) with physical_models == ['Hybride', 'Universeel']
    (sorted)."""
    df = build_chunked(rules_two_models, cache_dir=tmp_path / "cache", log=lambda _: None)
    assert len(df) > 0
    for _, row in df.iterrows():
        assert sorted(row["physical_models"]) == ["Hybride", "Universeel"], (
            f"row {row['lhs']} = {row['rhs_text']} has "
            f"physical_models={row['physical_models']}, expected both models"
        )


def test_classification_linear_vs_nonlinear(rules_with_nonlinear):
    linear, nonlinear = classify_linear_nonlinear(rules_with_nonlinear)
    assert len(linear) == 1
    assert linear[0]["lhs"] == "Zelfgebruik"
    assert len(nonlinear) == 1
    assert nonlinear[0]["lhs"] == "ZelfgebruikPercentage"


def test_discover_models_unions_across_rules(rules_two_models):
    assert discover_models(rules_two_models) == ["Hybride", "Universeel"]


# ---------------------------------------------------------------------------
# Cache: hit on no-change, miss on change
# ---------------------------------------------------------------------------

def test_cache_hit_skips_rebuild(rules_with_nonlinear, tmp_path):
    """Second build with same rules + same cache dir should reuse every chunk."""
    cache_dir = tmp_path / "cache"

    logs1 = []
    build_chunked(rules_with_nonlinear, cache_dir=cache_dir, log=logs1.append)

    logs2 = []
    build_chunked(rules_with_nonlinear, cache_dir=cache_dir, log=logs2.append)

    # Second run: every linear and nl chunk should be cache=hit
    miss_lines = [line for line in logs2 if "cache=miss" in line]
    hit_lines = [line for line in logs2 if "cache=hit" in line]
    assert len(miss_lines) == 0, f"expected all hits on warm cache, got misses: {miss_lines}"
    assert len(hit_lines) >= 1


def test_per_rule_cache_invalidation(rules_with_nonlinear, tmp_path):
    """Changing only the non-linear rule should invalidate only its chunk;
    the linear chunk's cache hash is unchanged.
    """
    cache_dir = tmp_path / "cache"

    # Warm cache
    build_chunked(rules_with_nonlinear, cache_dir=cache_dir, log=lambda _: None)

    # Mutate the non-linear rule's RHS: rename ratio direction so canonical
    # hash changes but everything else stays the same.
    mutated = [dict(r) for r in rules_with_nonlinear]
    mutated[1]["rhs"] = "ZonopwekBruto / Zelfgebruik"  # was Zelfgebruik / ZonopwekBruto

    logs = []
    build_chunked(mutated, cache_dir=cache_dir, log=logs.append)

    linear_lines = [line for line in logs if "[build_chunked] linear cache=" in line]
    nl_lines = [line for line in logs if "[build_chunked] nl-chunk cache=" in line]

    # Linear chunk hash should still hit (only the non-linear rule changed)
    assert any("cache=hit" in line for line in linear_lines), (
        f"expected linear cache=hit after only-NL change; got {linear_lines}"
    )
    # The non-linear chunk should be a miss
    assert any("cache=miss" in line for line in nl_lines), (
        f"expected nl-chunk cache=miss after rule change; got {nl_lines}"
    )


def test_canonical_hash_stable_under_cosmetic_changes(linear_rules_one_model):
    """The canonical-content hash must be invariant under whitespace and
    operator-order differences in the RHS string (sympy canonicalises).
    """
    h1 = canonical_linear_hash(linear_rules_one_model, "Universeel")

    cosmetic = [dict(r) for r in linear_rules_one_model]
    cosmetic[0]["rhs"] = "ZonopwekBruto  -  Terug"  # extra spaces
    h2 = canonical_linear_hash(cosmetic, "Universeel")
    assert h1 == h2, "cosmetic whitespace must not change hash"

    rearranged = [dict(r) for r in linear_rules_one_model]
    rearranged[0]["rhs"] = "-Terug + ZonopwekBruto"  # equivalent reordering
    h3 = canonical_linear_hash(rearranged, "Universeel")
    assert h1 == h3, "equivalent algebraic reordering must not change hash"


def test_canonical_hash_changes_on_real_change(linear_rules_one_model):
    h1 = canonical_linear_hash(linear_rules_one_model, "Universeel")
    real_change = [dict(r) for r in linear_rules_one_model]
    real_change[0]["rhs"] = "ZonopwekBruto - 2*Terug"  # different math
    h2 = canonical_linear_hash(real_change, "Universeel")
    assert h1 != h2, "real semantic change must change hash"


# ---------------------------------------------------------------------------
# plan_chunked_build
# ---------------------------------------------------------------------------

def test_plan_chunked_build_empty_cache_reports_all_misses(rules_with_nonlinear, tmp_path):
    """On an empty cache, every chunk should appear in would_rebuild."""
    plan = plan_chunked_build(rules_with_nonlinear, cache_dir=tmp_path / "cache")
    assert plan["would_hit"] == []
    assert len(plan["would_rebuild"]) == plan["total_chunks"]
    # Expect 1 linear chunk + 1 NL chunk per model
    assert len(plan["would_rebuild"]) == len(plan["models"]) * 2


def test_plan_chunked_build_warm_cache_reports_all_hits(rules_with_nonlinear, tmp_path):
    """After build_chunked populates the cache, plan should report all hits."""
    from etdtransform.catalog import build_chunked

    cache_dir = tmp_path / "cache"
    build_chunked(rules_with_nonlinear, cache_dir=cache_dir, log=lambda _: None)
    plan = plan_chunked_build(rules_with_nonlinear, cache_dir=cache_dir)
    assert plan["would_rebuild"] == []
    assert len(plan["would_hit"]) == plan["total_chunks"]


def test_plan_chunked_build_detects_only_nl_rule_change(rules_with_nonlinear, tmp_path):
    """Mutating only the non-linear rule should leave linear chunks cached
    and mark only the NL chunks as would_rebuild.
    """
    from etdtransform.catalog import build_chunked

    cache_dir = tmp_path / "cache"
    build_chunked(rules_with_nonlinear, cache_dir=cache_dir, log=lambda _: None)

    mutated = [dict(r) for r in rules_with_nonlinear]
    mutated[1]["rhs"] = "ZonopwekBruto / Zelfgebruik"  # was Zelfgebruik / ZonopwekBruto

    plan = plan_chunked_build(mutated, cache_dir=cache_dir)
    kinds_rebuilt = {c["kind"] for c in plan["would_rebuild"]}
    kinds_hit = {c["kind"] for c in plan["would_hit"]}
    # Linear chunks should still hit cache
    assert "linear" in kinds_hit
    assert "linear" not in kinds_rebuilt
    # NL chunk(s) should be marked for rebuild
    assert any(k.startswith("nl-") for k in kinds_rebuilt)


# ---------------------------------------------------------------------------
# Compose+prune unit tests
# ---------------------------------------------------------------------------

def test_compose_unions_models_for_identical_derivations():
    rows = [
        {"lhs": "A", "rhs_text": "B+C", "rhs_vars": ["B", "C"], "rhs_var_count": 2, "physical_models": ["Universeel"]},
        {"lhs": "A", "rhs_text": "B+C", "rhs_vars": ["B", "C"], "rhs_var_count": 2, "physical_models": ["Hybride"]},
    ]
    out = compose_and_prune(rows)
    assert len(out) == 1
    assert out[0]["physical_models"] == ["Hybride", "Universeel"]


def test_compose_pareto_drops_supersets_per_model():
    rows = [
        # Tight derivation: A from {B}, available in Universeel
        {"lhs": "A", "rhs_text": "B", "rhs_vars": ["B"], "rhs_var_count": 1, "physical_models": ["Universeel"]},
        # Looser derivation: A from {B, C} -- dominated in Universeel by the above
        {"lhs": "A", "rhs_text": "B+0*C", "rhs_vars": ["B", "C"], "rhs_var_count": 2, "physical_models": ["Universeel"]},
        # Same loose derivation but in a different model where the tight one does not exist -- must survive
        {"lhs": "A", "rhs_text": "B+0*C", "rhs_vars": ["B", "C"], "rhs_var_count": 2, "physical_models": ["Hybride"]},
    ]
    out = compose_and_prune(rows)
    by_key = {(r["lhs"], frozenset(r["rhs_vars"])): r for r in out}
    # Tight one survives in Universeel only
    assert ("A", frozenset(["B"])) in by_key
    assert by_key[("A", frozenset(["B"]))]["physical_models"] == ["Universeel"]
    # Loose one is dominated in Universeel but survives in Hybride
    assert ("A", frozenset(["B", "C"])) in by_key
    assert by_key[("A", frozenset(["B", "C"]))]["physical_models"] == ["Hybride"]
