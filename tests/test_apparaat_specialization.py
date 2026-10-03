"""Sync-time constraint specialization per apparaat-set, and its run-time
selection.

The sync substitutes 0 for each declared-absent device set's variables in
the catalog (``specialize_catalog_for_absent``), compiles one constraint
table per set, and writes them under an ``apparaat_set`` key. The fill only
SELECTS: each household narrows with the table its declarations pick --
no sympy at run time. A pre-specialization table (no key column) is the
empty-set table for every household, warned, so nothing breaks before the
next sync apply.
"""

import pandas as pd
import pytest
from conftest import cadence_table
from pandas.testing import assert_frame_equal
from test_constraint_package import _constraints

from etdtransform.constraints import (
    apparaat_set_by_household,
    apparaat_set_key,
    specialize_catalog_for_absent,
)
from etdtransform.vectorized_impute import impute_and_normalize

pytestmark = pytest.mark.unit

CUM = "ElektriciteitNetgebruik"
DIFF = "ElektriciteitNetgebruikDiff"
AVG = f"{DIFF}_avg"


def _catalog(rows):
    return pd.DataFrame(
        rows,
        columns=["lhs", "rhs_text", "rhs_vars", "rhs_var_count", "physical_models"],
    )


def _row(lhs, rhs_text, rhs_vars):
    return {
        "lhs": lhs,
        "rhs_text": rhs_text,
        "rhs_vars": rhs_vars,
        "rhs_var_count": len(rhs_vars),
        "physical_models": ["Universeel"],
    }


class TestApparaatSetKey:
    def test_sorted_pipe_joined_and_empty(self):
        assert apparaat_set_key({"Zb", "Aa"}) == "Aa|Zb"
        assert apparaat_set_key(set()) == ""


class TestSpecializeCatalogForAbsent:
    def test_zeroed_summand_simplifies_and_leaves_rhs_vars(self):
        out = specialize_catalog_for_absent(
            _catalog([_row("TotaalX", "A + B", ["A", "B"])]), {"B"}
        )
        assert out.loc[0, "rhs_text"] == "A"
        assert list(out.loc[0, "rhs_vars"]) == ["A"]
        assert int(out.loc[0, "rhs_var_count"]) == 1

    def test_rule_reduced_to_a_constant_disappears(self):
        # Absence may only widen: a target pinned to a constant purely by
        # declarations would let metadata specify a value, so the row drops
        # and the declared bounds stay the only bound on that target.
        out = specialize_catalog_for_absent(
            _catalog([_row("TotaalX", "A + B", ["A", "B"])]), {"A", "B"}
        )
        assert out.empty

    def test_absent_as_target_drops(self):
        out = specialize_catalog_for_absent(
            _catalog([_row("B", "TotaalX - A", ["TotaalX", "A"])]), {"B"}
        )
        assert out.empty

    def test_untouched_rule_is_byte_identical(self):
        catalog = _catalog(
            [
                _row("TotaalX", "A + B", ["A", "B"]),
                _row("C", "D*2", ["D"]),
            ]
        )
        out = specialize_catalog_for_absent(catalog, {"B"})
        untouched = out[out["lhs"] == "C"].reset_index(drop=True)
        expected = catalog[catalog["lhs"] == "C"].reset_index(drop=True)
        assert_frame_equal(
            untouched.drop(columns=["rhs_var_count"]),
            expected.drop(columns=["rhs_var_count"]),
        )
        assert untouched.loc[0, "rhs_text"] == "D*2"

    def test_empty_set_is_identity(self):
        catalog = _catalog([_row("TotaalX", "A + B", ["A", "B"])])
        assert_frame_equal(
            specialize_catalog_for_absent(catalog, set()).drop(
                columns=["rhs_var_count"]
            ),
            catalog.drop(columns=["rhs_var_count"]),
        )

    def test_declared_cumulative_name_also_zeroes_its_interval_form(self):
        # Declarations carry cumulative device names; catalog expressions
        # name the Diff forms. The data-model linkage covers both.
        out = specialize_catalog_for_absent(
            _catalog(
                [
                    _row(
                        "ElektriciteitsgebruikTotaalWarmtepomp",
                        "ElektriciteitsgebruikWarmtepompDiff + ElektriciteitsgebruikRadiatorDiff",
                        [
                            "ElektriciteitsgebruikWarmtepompDiff",
                            "ElektriciteitsgebruikRadiatorDiff",
                        ],
                    )
                ]
            ),
            {"ElektriciteitsgebruikRadiator"},
        )
        assert out.loc[0, "rhs_text"] == "ElektriciteitsgebruikWarmtepompDiff"
        assert list(out.loc[0, "rhs_vars"]) == [
            "ElektriciteitsgebruikWarmtepompDiff"
        ]


class TestApparaatSetByHousehold:
    def test_resolves_each_household_to_its_canonical_key(self, tmp_path):
        path = tmp_path / "metadata.csv"
        pd.DataFrame(
            {
                "HuisIdBSV": [1, 2],
                "ApparaatOntbrekend": [
                    "ElektriciteitsgebruikWTW, ElektriciteitsgebruikRadiator",
                    "",
                ],
            }
        ).to_csv(path, index=False)
        resolved = apparaat_set_by_household(path)
        assert resolved == {
            1: "ElektriciteitsgebruikRadiator|ElektriciteitsgebruikWTW",
            2: "",
        }

    def test_metadata_without_the_column_raises(self, tmp_path):
        path = tmp_path / "stale.csv"
        pd.DataFrame({"HuisIdBSV": [1]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="ApparaatOntbrekend"):
            apparaat_set_by_household(path)


class TestRuleSetsByHousehold:
    """What a household may derive, resolved from its declarations. One
    composition in the library, so no caller repeats the two steps."""

    @staticmethod
    def _metadata(tmp_path, declarations):
        path = tmp_path / "metadata.csv"
        pd.DataFrame({
            "HuisIdBSV": list(declarations),
            "ApparaatOntbrekend": list(declarations.values()),
        }).to_csv(path, index=False)
        return path

    def _catalog_with_a_device_term(self):
        return _catalog([
            _row(
                "ElektriciteitsgebruikTotaalBruto",
                f"{DIFF} + ElektriciteitsgebruikRadiatorDiff",
                [DIFF, "ElektriciteitsgebruikRadiatorDiff"],
            ),
        ])

    def test_households_are_grouped_by_their_declaration(self, tmp_path):
        from etdtransform.constraints import rule_sets_by_household

        path = self._metadata(tmp_path, {
            1: "ElektriciteitsgebruikRadiator", 2: "ElektriciteitsgebruikRadiator", 3: "",
        })
        resolved = rule_sets_by_household(
            [1, 2, 3], self._catalog_with_a_device_term(), 5,
            metadata_path=path,
        )
        assert set(resolved) == {"ElektriciteitsgebruikRadiator", ""}
        assert resolved["ElektriciteitsgebruikRadiator"].households == [1, 2]
        assert resolved[""].households == [3]

    def test_an_absent_device_leaves_the_formula_set(self, tmp_path):
        from etdtransform.constraints import rule_sets_by_household

        path = self._metadata(tmp_path, {
            1: "ElektriciteitsgebruikRadiator", 2: "",
        })
        resolved = rule_sets_by_household(
            [1, 2], self._catalog_with_a_device_term(), 5, metadata_path=path,
        )
        # Household 1 has no radiator meter, so the term is substituted to zero and
        # its formula no longer names it -- it is not a candidate that a null
        # check rejects later, it is simply not in the set.
        specialized = resolved["ElektriciteitsgebruikRadiator"].formulas
        assert list(specialized["rhs_vars"]) == [[DIFF]]
        assert "ElektriciteitsgebruikRadiatorDiff" not in str(
            specialized["rhs_text"].iloc[0]
        )
        assert list(resolved[""].formulas["rhs_vars"]) == [
            [DIFF, "ElektriciteitsgebruikRadiatorDiff"]
        ]

    def test_a_household_the_metadata_does_not_mention_raises(self, tmp_path):
        from etdtransform.constraints import rule_sets_by_household

        path = self._metadata(tmp_path, {1: ""})
        # Defaulting to the full model would narrow HARDER on a household
        # whose devices are unknown, and absence may only widen.
        with pytest.raises(ValueError, match="not in the BSV metadata"):
            rule_sets_by_household(
                [1, 99], self._catalog_with_a_device_term(), 5,
                metadata_path=path,
            )

    def test_each_set_gets_a_package_at_the_grid_cadence(self, tmp_path):
        from etdtransform.constraints import rule_sets_by_household

        path = self._metadata(tmp_path, {1: ""})
        resolved = rule_sets_by_household(
            [1], self._catalog_with_a_device_term(), 15, metadata_path=path,
        )
        assert resolved[""].package is not None
        assert resolved[""].apparaat_set == ""


def _set_keyed(tables):
    """Concatenate (set_key, constraints_frame) pairs under the key column."""
    frames = []
    for set_key, table in tables:
        keyed = table.copy()
        keyed["apparaat_set"] = pd.array([set_key] * len(keyed), dtype="string")
        frames.append(keyed)
    return pd.concat(frames, ignore_index=True)


def _two_household_frame():
    """Two anchored gaps: household 1 jumps 3, household 2 jumps 1.5.

    K and K2 both sit inside the column's DECLARED 5-minute bound
    ([0, 1.5] for ElektriciteitNetgebruikDiff) -- a pin outside it would
    intersect to an empty bound and withhold the fill, which is a different
    behaviour than the selection this file tests.
    """

    def rows(household, start, end):
        return pd.DataFrame(
            {
                "HuisIdBSV": pd.array([household] * 4, dtype="Int64"),
                "ProjectIdBSV": pd.array([1] * 4, dtype="Int64"),
                "ReadingDate": pd.date_range(
                    "2026-01-01", periods=4, freq="5min"
                ),
                CUM: pd.array([start, pd.NA, pd.NA, end], dtype="Float64"),
                DIFF: pd.array([pd.NA] * 4, dtype="Float64"),
                AVG: pd.array([pd.NA, 0.5, 1.0, 1.5], dtype="Float64"),
                "K": pd.array([pd.NA, 1.0, 1.0, 1.0], dtype="Float64"),
                "K2": pd.array([pd.NA, 0.5, 0.5, 0.5], dtype="Float64"),
            }
        )

    return pd.concat(
        [rows(1, 100.0, 103.0), rows(2, 100.0, 101.5)], ignore_index=True
    )


RADIATOR_SET = "ElektriciteitsgebruikRadiator"


def _impute(frame, constraints, apparaat_sets=None):
    out, _, _ = impute_and_normalize(
        frame,
        [CUM],
        "ProjectIdBSV",
        cadence=cadence_table(frame, [CUM]),
        constraints=constraints,
        apparaat_sets=apparaat_sets,
    )
    return out


class TestRunTimeSelection:
    def test_each_household_narrows_with_its_own_sets_table(self):
        constraints = _set_keyed(
            [
                ("", _constraints((DIFF, "K", {"K": 1}))),
                (RADIATOR_SET, _constraints((DIFF, "K2", {"K2": 1}))),
            ]
        )
        out = _impute(
            _two_household_frame(),
            constraints,
            apparaat_sets={1: "", 2: RADIATOR_SET},
        )
        one = out[out["HuisIdBSV"] == 1]
        two = out[out["HuisIdBSV"] == 2]
        assert one[DIFF].iloc[1:].astype(float).tolist() == pytest.approx([1.0] * 3)
        assert two[DIFF].iloc[1:].astype(float).tolist() == pytest.approx([0.5] * 3)

    def test_a_set_without_a_compiled_table_raises_as_unsynced(self):
        constraints = _set_keyed([("", _constraints((DIFF, "K", {"K": 1})))])
        with pytest.raises(ValueError, match="data model sync"):
            _impute(
                _two_household_frame(),
                constraints,
                apparaat_sets={1: "", 2: RADIATOR_SET},
            )

    def test_a_household_without_a_set_raises_as_metadata_gap(self):
        constraints = _set_keyed([("", _constraints((DIFF, "K", {"K": 1})))])
        with pytest.raises(ValueError, match="apparaat-set registered"):
            _impute(
                _two_household_frame(),
                constraints,
                apparaat_sets={1: ""},
            )

    def test_pre_specialization_table_is_the_empty_set_with_a_warning(
        self, caplog
    ):
        legacy = _constraints((DIFF, "K", {"K": 1}))
        with caplog.at_level("WARNING"):
            out = _impute(_two_household_frame(), legacy)
        assert any("apparaat_set" in record.message for record in caplog.records)
        # Both households narrow with the one table, exactly as before the
        # split: household 2's bound follows K too, so its jump of 6 cannot
        # be placed inside [1, 1] rows and the values pin at 1 each.
        one = out[out["HuisIdBSV"] == 1]
        two = out[out["HuisIdBSV"] == 2]
        assert one[DIFF].iloc[1:].astype(float).tolist() == pytest.approx([1.0] * 3)
        assert two[DIFF].iloc[1:].astype(float).tolist() == pytest.approx([1.0] * 3)
