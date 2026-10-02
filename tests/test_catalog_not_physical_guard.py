"""The catalog build refuses not_physical rules. The sync excludes them at
compilation time (they are bookkeeping identities, e.g. tariff sums); this
guard is etdtransform's side of the contract -- if a caller ever passes one
through, the build raises instead of letting it reach derivations and,
downstream, constraints.
"""

import pytest

from etdtransform.catalog.chunked import build_chunked


def _rule(lhs, rhs, rhs_vars, models, not_physical=False):
    return {
        "lhs": lhs,
        "rhs": rhs,
        "rhs_vars": rhs_vars,
        "physical_models": models,
        "not_physical": not_physical,
    }


class TestNotPhysicalGuard:
    def test_not_physical_rule_raises_before_any_work(self, tmp_path):
        rules = [
            _rule("Root", "Hoog + Laag", ["Hoog", "Laag"], ["M"], not_physical=True),
        ]
        with pytest.raises(ValueError, match="not_physical rule\\(s\\) reached"):
            build_chunked(rules, cache_dir=tmp_path)
        assert not any(tmp_path.iterdir()), "guard must fire before cache writes"

    def test_offending_rule_is_named(self, tmp_path):
        rules = [
            _rule("TotaalX", "A + B", ["A", "B"], ["M"]),
            _rule("Root", "Hoog + Laag", ["Hoog", "Laag"], ["M"], not_physical=True),
        ]
        with pytest.raises(ValueError, match="Root"):
            build_chunked(rules, cache_dir=tmp_path)

    def test_unflagged_rules_build(self, tmp_path):
        rules = [_rule("TotaalX", "A + B", ["A", "B"], ["M"])]
        catalog = build_chunked(rules, cache_dir=tmp_path, log=lambda s: None)
        assert not catalog.empty
        assert "TotaalX" in set(catalog["lhs"])

    def test_rules_without_the_key_pass_the_guard(self, tmp_path):
        # Older callers build rule dicts without a not_physical key; absence
        # means "physical" and must not raise.
        rules = [{
            "lhs": "TotaalX", "rhs": "A + B",
            "rhs_vars": ["A", "B"], "physical_models": ["M"],
        }]
        catalog = build_chunked(rules, cache_dir=tmp_path, log=lambda s: None)
        assert not catalog.empty
