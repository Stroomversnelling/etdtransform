"""`build_constraint_package` turns the compiler's table into bounds at a cadence.

The compiler is cadence-blind and ships expression strings. The processor runs in
the imputation stage, where the grid cadence is known, and is what turns those
expressions into numbers: it scales every declared bound to the cadence, narrows
each one by the rules, groups the columns that constrain each other, and
lambdifies the expressions for per-row use.

An empty table is the off switch and must produce a package that narrows nothing.
"""

import pandas as pd
import pytest

from etdtransform.constraints import (
    CONSTRAINT_COLUMNS,
    build_constraint_package,
    compile_constraints,
)

INF = float("inf")


def _constraints(*rows):
    """A compiler-shaped table. Each row is (lhs, rhs_text, {var: sign})."""
    records = []
    for lhs, rhs_text, signs in rows:
        for variable, sign in signs.items():
            records.append({
                "lhs": lhs,
                "rhs_text": rhs_text,
                "variable": variable,
                "sign": sign,
                "derivative_text": str(sign),
                "rhs_var_count": len(signs),
                "physical_models": ["Universeel"],
            })
    table = pd.DataFrame(records, columns=list(CONSTRAINT_COLUMNS))
    return table.astype({"sign": "Int64", "rhs_var_count": "Int64"})


def _thresholds(**pairs):
    return {
        name: {
            "Min": lo, "Max": hi, "ThresholdType": kind,
            "UpperThresholdSustained": pd.NA, "UpperThresholdExponent": pd.NA,
        }
        for name, (lo, hi, kind) in pairs.items()
    }


def _interval(lo, hi):
    return (lo, hi, "5-minute")


def _flat(lo, hi):
    """A bound that does not scale with cadence."""
    return (lo, hi, "algemeen")


class TestTheOffSwitch:
    def test_an_empty_table_narrows_nothing(self):
        package = build_constraint_package(
            _constraints(), cadence_minutes=5,
            thresholds=_thresholds(A=_interval(0.0, 1.0)),
        )
        assert package.rules_by_target == {}
        assert package.clusters == ()

    def test_an_empty_table_still_reports_declared_bounds(self):
        """The fill needs a bound for every column whether rules exist or not."""
        package = build_constraint_package(
            _constraints(), cadence_minutes=5,
            thresholds=_thresholds(A=_interval(0.0, 1.0)),
        )
        assert package.bounds["A"] == (0.0, 1.0)


class TestBoundsComeFromTheCadence:
    TABLE = (("T", "A + B", {"A": 1, "B": 1}),)

    def test_a_five_minute_bound_scales_with_the_grid(self):
        at_five = build_constraint_package(
            _constraints(*self.TABLE), cadence_minutes=5,
            thresholds=_thresholds(T=_interval(0.0, 10.0), A=_interval(0.0, 1.0), B=_interval(0.0, 1.0)),
        )
        at_sixty = build_constraint_package(
            _constraints(*self.TABLE), cadence_minutes=60,
            thresholds=_thresholds(T=_interval(0.0, 10.0), A=_interval(0.0, 1.0), B=_interval(0.0, 1.0)),
        )
        assert at_sixty.bounds["A"][1] > at_five.bounds["A"][1]

    def test_a_flat_bound_does_not_scale(self):
        package = build_constraint_package(
            _constraints(), cadence_minutes=60,
            thresholds=_thresholds(P=_flat(0.0, 100.0)),
        )
        assert package.bounds["P"] == (0.0, 100.0)

    def test_an_undeclared_side_is_infinite_not_missing(self):
        """Interval arithmetic needs a number. pd.NA means no bound, which is
        an infinity, not an unknown."""
        package = build_constraint_package(
            _constraints(), cadence_minutes=5,
            thresholds=_thresholds(A=_interval(0.0, pd.NA)),
        )
        assert package.bounds["A"] == (0.0, INF)

    def test_the_cadence_is_recorded(self):
        package = build_constraint_package(
            _constraints(), cadence_minutes=15, thresholds=_thresholds(),
        )
        assert package.cadence_minutes == 15


class TestTheRulesNarrowTheBounds:
    def test_a_sum_of_capped_parts_caps_the_total(self):
        package = build_constraint_package(
            _constraints(("T", "A + B", {"A": 1, "B": 1})),
            cadence_minutes=5,
            thresholds=_thresholds(
                T=_interval(0.0, 10.0), A=_interval(0.0, 1.0), B=_interval(0.0, 2.0),
            ),
        )
        assert package.bounds["T"] == (0.0, 3.0), "10 declared, 3 implied"

    def test_a_target_with_no_declared_max_can_gain_one(self):
        package = build_constraint_package(
            _constraints(("T", "A + B", {"A": 1, "B": 1})),
            cadence_minutes=5,
            thresholds=_thresholds(
                T=_interval(0.0, pd.NA), A=_interval(0.0, 1.0), B=_interval(0.0, 2.0),
            ),
        )
        assert package.bounds["T"] == (0.0, 3.0)

    def test_a_subtraction_uses_the_opposite_endpoint(self):
        package = build_constraint_package(
            _constraints(("T", "A - B", {"A": 1, "B": -1})),
            cadence_minutes=5,
            thresholds=_thresholds(
                T=_interval(-99.0, 99.0), A=_interval(0.0, 1.0), B=_interval(0.0, 2.0),
            ),
        )
        assert package.bounds["T"] == (-2.0, 1.0)

    def test_the_declared_bound_wins_when_it_is_tighter(self):
        package = build_constraint_package(
            _constraints(("T", "A + B", {"A": 1, "B": 1})),
            cadence_minutes=5,
            thresholds=_thresholds(
                T=_interval(0.0, 1.0), A=_interval(0.0, 10.0), B=_interval(0.0, 10.0),
            ),
        )
        assert package.bounds["T"] == (0.0, 1.0)

    def test_narrowing_propagates_through_a_chain(self):
        """T is capped by A and B; U is capped by T, so U's cap depends on a
        bound that narrowing produced. One pass over the rules cannot see it."""
        package = build_constraint_package(
            _constraints(
                ("T", "A + B", {"A": 1, "B": 1}),
                ("U", "T + C", {"T": 1, "C": 1}),
            ),
            cadence_minutes=5,
            thresholds=_thresholds(
                T=_interval(0.0, 100.0), U=_interval(0.0, 100.0),
                A=_interval(0.0, 1.0), B=_interval(0.0, 1.0), C=_interval(0.0, 1.0),
            ),
        )
        assert package.bounds["T"] == (0.0, 2.0)
        assert package.bounds["U"] == (0.0, 3.0), "needs T's narrowed cap, not its declared one"

    def test_narrowing_is_idempotent(self):
        table = _constraints(("T", "A + B", {"A": 1, "B": 1}))
        thresholds = _thresholds(
            T=_interval(0.0, 10.0), A=_interval(0.0, 1.0), B=_interval(0.0, 2.0),
        )
        first = build_constraint_package(table, 5, thresholds=thresholds)
        second = build_constraint_package(table, 5, thresholds=thresholds)
        assert first.bounds == second.bounds

    def test_an_unbounded_part_leaves_the_total_unbounded(self):
        package = build_constraint_package(
            _constraints(("T", "A + B", {"A": 1, "B": 1})),
            cadence_minutes=5,
            thresholds=_thresholds(
                T=_interval(0.0, pd.NA), A=_interval(0.0, pd.NA), B=_interval(0.0, 2.0),
            ),
        )
        assert package.bounds["T"] == (0.0, INF)


class TestAnAbsentSignIsSkipped:
    def test_a_derivation_with_an_absent_sign_does_not_narrow(self):
        """An absent sign means the direction is unknown, which is not zero. Using
        it would invent a bound."""
        package = build_constraint_package(
            _constraints(("T", "A*B", {"A": pd.NA, "B": pd.NA})),
            cadence_minutes=5,
            thresholds=_thresholds(
                T=_interval(0.0, 10.0), A=_interval(0.0, 1.0), B=_interval(0.0, 2.0),
            ),
        )
        assert package.bounds["T"] == (0.0, 10.0), "declared bound untouched"

    def test_a_usable_derivation_still_narrows_alongside_an_unusable_one(self):
        package = build_constraint_package(
            _constraints(
                ("T", "A*B", {"A": pd.NA, "B": pd.NA}),
                ("T", "A + B", {"A": 1, "B": 1}),
            ),
            cadence_minutes=5,
            thresholds=_thresholds(
                T=_interval(0.0, 10.0), A=_interval(0.0, 1.0), B=_interval(0.0, 2.0),
            ),
        )
        assert package.bounds["T"] == (0.0, 3.0)


class TestAContradictionIsAConfigurationError:
    def test_an_empty_intersection_raises(self):
        with pytest.raises(ValueError) as raised:
            build_constraint_package(
                _constraints(("T", "A + B", {"A": 1, "B": 1})),
                cadence_minutes=5,
                thresholds=_thresholds(
                    T=_interval(50.0, 99.0), A=_interval(0.0, 1.0), B=_interval(0.0, 2.0),
                ),
            )
        assert "T" in str(raised.value)

    def test_the_message_gives_both_ranges_and_the_rule(self):
        with pytest.raises(ValueError) as raised:
            build_constraint_package(
                _constraints(("T", "A + B", {"A": 1, "B": 1})),
                cadence_minutes=5,
                thresholds=_thresholds(
                    T=_interval(50.0, 99.0), A=_interval(0.0, 1.0), B=_interval(0.0, 2.0),
                ),
            )
        message = str(raised.value)
        assert "A + B" in message
        assert "50" in message and "3" in message

    def test_the_message_names_the_cadence(self):
        """The same metadata can be consistent at one cadence and not another."""
        with pytest.raises(ValueError) as raised:
            build_constraint_package(
                _constraints(("T", "A + B", {"A": 1, "B": 1})),
                cadence_minutes=60,
                thresholds=_thresholds(
                    T=_flat(50.0, 99.0), A=_interval(0.0, 1.0), B=_interval(0.0, 2.0),
                ),
            )
        assert "60" in str(raised.value)


class TestClusters:
    def test_a_constraint_links_its_target_and_its_variables(self):
        package = build_constraint_package(
            _constraints(("T", "A + B", {"A": 1, "B": 1})),
            cadence_minutes=5,
            thresholds=_thresholds(
                T=_interval(0.0, 10.0), A=_interval(0.0, 1.0), B=_interval(0.0, 2.0),
            ),
        )
        assert package.clusters == (frozenset({"T", "A", "B"}),)

    def test_unrelated_constraints_stay_in_separate_clusters(self):
        package = build_constraint_package(
            _constraints(
                ("T", "A + B", {"A": 1, "B": 1}),
                ("U", "C + D", {"C": 1, "D": 1}),
            ),
            cadence_minutes=5,
            thresholds=_thresholds(**{
                n: _interval(0.0, 10.0) for n in ("T", "U", "A", "B", "C", "D")
            }),
        )
        assert len(package.clusters) == 2
        assert frozenset({"T", "A", "B"}) in package.clusters
        assert frozenset({"U", "C", "D"}) in package.clusters

    def test_a_shared_column_merges_two_clusters(self):
        package = build_constraint_package(
            _constraints(
                ("T", "A + B", {"A": 1, "B": 1}),
                ("U", "B + C", {"B": 1, "C": 1}),
            ),
            cadence_minutes=5,
            thresholds=_thresholds(**{
                n: _interval(0.0, 10.0) for n in ("T", "U", "A", "B", "C")
            }),
        )
        assert package.clusters == (frozenset({"T", "U", "A", "B", "C"}),)

    def test_a_column_in_no_constraint_is_in_no_cluster(self):
        package = build_constraint_package(
            _constraints(("T", "A + B", {"A": 1, "B": 1})),
            cadence_minutes=5,
            thresholds=_thresholds(**{
                n: _interval(0.0, 10.0) for n in ("T", "A", "B", "Lonely")
            }),
        )
        assert not any("Lonely" in cluster for cluster in package.clusters)


class TestTheLambdifiedRules:
    def test_a_rule_evaluates_its_expression(self):
        package = build_constraint_package(
            _constraints(("T", "A + B", {"A": 1, "B": 1})),
            cadence_minutes=5,
            thresholds=_thresholds(
                T=_interval(0.0, 10.0), A=_interval(0.0, 1.0), B=_interval(0.0, 2.0),
            ),
        )
        rule = package.rules_by_target["T"][0]
        assert rule.evaluate(**{"A": 1.0, "B": 2.0}) == 3.0

    def test_a_rule_evaluates_over_arrays(self):
        """The fill works on columns, so a scalar-only callable would be useless."""
        package = build_constraint_package(
            _constraints(("T", "A - B", {"A": 1, "B": -1})),
            cadence_minutes=5,
            thresholds=_thresholds(
                T=_interval(-9.0, 9.0), A=_interval(0.0, 5.0), B=_interval(0.0, 5.0),
            ),
        )
        rule = package.rules_by_target["T"][0]
        out = rule.evaluate(
            A=pd.Series([3.0, 4.0]).to_numpy(), B=pd.Series([1.0, 1.0]).to_numpy(),
        )
        assert list(out) == [2.0, 3.0]

    def test_a_rule_carries_its_signs_and_variables(self):
        package = build_constraint_package(
            _constraints(("T", "A - B", {"A": 1, "B": -1})),
            cadence_minutes=5,
            thresholds=_thresholds(
                T=_interval(-9.0, 9.0), A=_interval(0.0, 5.0), B=_interval(0.0, 5.0),
            ),
        )
        rule = package.rules_by_target["T"][0]
        assert set(rule.variables) == {"A", "B"}
        assert rule.signs == {"A": 1, "B": -1}

    def test_every_derivation_of_a_target_is_kept(self):
        package = build_constraint_package(
            _constraints(
                ("T", "A + B", {"A": 1, "B": 1}),
                ("T", "C + D", {"C": 1, "D": 1}),
            ),
            cadence_minutes=5,
            thresholds=_thresholds(**{
                n: _interval(0.0, 10.0) for n in ("T", "A", "B", "C", "D")
            }),
        )
        assert len(package.rules_by_target["T"]) == 2


class TestAgainstTheCommittedCatalog:
    @pytest.fixture(scope="class")
    def package(self):
        from pathlib import Path

        import etdmap

        path = Path(etdmap.__file__).parent / "data" / "catalog.parquet"
        catalog = pd.read_parquet(path, dtype_backend="numpy_nullable")
        return build_constraint_package(compile_constraints(catalog), cadence_minutes=5)

    def test_it_builds_at_the_five_minute_grid(self, package):
        assert package.rules_by_target
        assert package.clusters

    def test_the_real_metadata_holds_no_contradiction(self, package):
        """Building at all means every intersection was non-empty, which is what
        check_implied_bounds.py reported."""
        for column, (lo, hi) in package.bounds.items():
            assert lo <= hi, column

    def _narrowed(self, package):
        from etdmap.data_model import scale_bounds, thresholds_dict

        narrowed = {}
        for column, (lo, hi) in package.bounds.items():
            declared = thresholds_dict.get(column)
            if not declared:
                continue
            d_lo, d_hi = scale_bounds(declared, package.cadence_minutes)
            d_lo = -INF if pd.isna(d_lo) else float(d_lo)
            d_hi = INF if pd.isna(d_hi) else float(d_hi)
            if lo > d_lo + 1e-12 or hi < d_hi - 1e-12:
                narrowed[column] = (lo, hi)
        return narrowed

    # Cross-checked against etdworkflow/check_implied_bounds.py, which reaches
    # these by a different route: sympy.solve over Rule.csv with coefficient
    # interval arithmetic, rather than sign-directed endpoints over the catalog.
    # Both produce exactly this table. A change here means the rules or the
    # thresholds moved; re-run that script and update this if so.
    #
    # Updated 2026-10-01 for the not_physical exclusion: the two tariff SPLIT
    # columns (ElektriciteitTerugleveringHoogDiff / ...LaagDiff), previously
    # narrowed to (0.0, 0.96), are gone from this table. The tariff-sum rules
    # that linked a split to the narrowed root are bookkeeping identities, not
    # physical relations, so they no longer reach the catalog -- nothing
    # narrows a split and each keeps its own declared bound. The eight
    # surviving entries are unchanged, which is the point: excluding those
    # rules moved nothing else.
    NARROWED_AT_FIVE_MINUTES = {
        "ElektriciteitTerugleveringDiff": (0.0, 0.96),
        "ElektriciteitsgebruikTotaalBruto": (0.0, 2.46),
        "ElektriciteitsgebruikTotaalGebouwgebonden": (0.0, 2.01),
        "ElektriciteitsgebruikTotaalNetto": (0.0, 1.5),
        "ElektriciteitsgebruikTotaalWarmtepomp": (0.0, 1.6),
        "Netuitwisseling": (-0.96, 1.5),
        "TerugleveringTotaalNetto": (0.0, 0.96),
        "Zelfgebruik": (0.0, 0.96),
    }

    def test_it_narrows_exactly_what_the_oracle_narrows(self, package):
        assert self._narrowed(package) == pytest.approx(
            self.NARROWED_AT_FIVE_MINUTES
        )

    def test_five_columns_gain_a_max_where_none_was_declared(self, package):
        from etdmap.data_model import thresholds_dict

        gained = [
            column
            for column in self._narrowed(package)
            if pd.isna(thresholds_dict[column]["Max"])
        ]
        assert len(gained) == 5

    def test_teruglevering_is_capped_by_what_the_panels_can_make(self, package):
        """Not a magic number: 0.96 is ZonopwekBruto's declared cap, and you
        cannot export more electricity than was generated."""
        declared_generation = package.bounds["ZonopwekBruto"][1]
        assert package.bounds["ElektriciteitTerugleveringDiff"][1] == declared_generation

    def test_every_electricity_column_lands_in_one_cluster(self, package):
        """The electricity rules are interlinked, so they should not fragment."""
        biggest = max(package.clusters, key=len)
        assert len(biggest) >= 15

    def test_a_coarser_grid_gives_wider_bounds(self, package):
        from pathlib import Path

        import etdmap

        path = Path(etdmap.__file__).parent / "data" / "catalog.parquet"
        catalog = pd.read_parquet(path, dtype_backend="numpy_nullable")
        hourly = build_constraint_package(
            compile_constraints(catalog), cadence_minutes=60
        )
        column = "ElektriciteitNetgebruikDiff"
        assert hourly.bounds[column][1] > package.bounds[column][1]


class TestApparaatSetSelection:
    """A set-keyed table holds one specialized table per declared
    absent-device set: the same rules with that set's device terms
    substituted to zero. Selecting ONE is mandatory -- unioning them applies
    the tightest specialization to households that have all their devices,
    which would clip legitimate values.
    """

    def _set_keyed(self):
        """Two tables: a base rule and a tighter specialized variant."""
        base = _constraints(
            ("Total", "PartA + PartB", {"PartA": 1, "PartB": 1}),
        )
        base["apparaat_set"] = ""
        specialized = _constraints(
            ("Total", "PartA", {"PartA": 1}),
        )
        specialized["apparaat_set"] = "DeviceB"
        return pd.concat([base, specialized], ignore_index=True)

    def _rule_count(self, package):
        return sum(len(rules) for rules in package.rules_by_target.values())

    def test_default_picks_the_none_absent_base_table(self):
        package = build_constraint_package(self._set_keyed(), cadence_minutes=5)
        rules = package.rules_by_target["Total"]
        assert {v for rule in rules for v in rule.variables} == {"PartA", "PartB"}

    def test_an_explicit_set_picks_that_table(self):
        package = build_constraint_package(
            self._set_keyed(), cadence_minutes=5, apparaat_set="DeviceB"
        )
        rules = package.rules_by_target["Total"]
        assert {v for rule in rules for v in rule.variables} == {"PartA"}

    def test_sets_are_never_mixed(self):
        table = self._set_keyed()
        base = build_constraint_package(table, cadence_minutes=5)
        specialized = build_constraint_package(
            table, cadence_minutes=5, apparaat_set="DeviceB"
        )
        # Each package carries one table's rules, never the concatenation.
        assert self._rule_count(base) == 1
        assert self._rule_count(specialized) == 1

    def test_an_unknown_set_raises_rather_than_narrowing_nothing(self):
        with pytest.raises(ValueError, match="No compiled constraint table"):
            build_constraint_package(
                self._set_keyed(), cadence_minutes=5, apparaat_set="Absent"
            )

    def test_a_frame_already_scoped_to_one_set_passes_through(self):
        """The fill pre-filters by set before building; that frame carries a
        single non-empty value and must not be filtered against the default."""
        table = self._set_keyed()
        scoped = table[table["apparaat_set"] == "DeviceB"]
        package = build_constraint_package(scoped, cadence_minutes=5)
        rules = package.rules_by_target["Total"]
        assert {v for rule in rules for v in rule.variables} == {"PartA"}

    def test_a_table_without_the_column_is_unaffected(self):
        package = build_constraint_package(
            _constraints(("Total", "PartA + PartB", {"PartA": 1, "PartB": 1})),
            cadence_minutes=5,
        )
        assert self._rule_count(package) == 1
