"""A rule narrows only the households whose physical model it applies to.

The compiled table carries each rule's physical_models; the synced project
registry carries each project's model; the fill joins the two per row. A
project with no registered model raises -- the model decides the rules, so it
cannot default.
"""

import pandas as pd
import pytest
from conftest import cadence_table
from etdmap import ids
from etdmap.testing import household_batches_frame, one_to_one_test_household_batch_of

from etdtransform.constraints import CONSTRAINT_COLUMNS, build_constraint_package
from etdtransform.vectorized_impute import ImputeType, impute_and_normalize

pytestmark = pytest.mark.unit

CUM = "ElektriciteitNetgebruik"
DIFF = "ElektriciteitNetgebruikDiff"
ITYPE = f"{DIFF}_impute_type"

#: One household per project, as ``household -> ids.PROJECT``. Each
#: household's household batch is ``one_to_one_test_household_batch_of`` of its number.
PROJECT_OF_HOUSEHOLD = {1: 1, 2: 2}

#: The readings every household carries. Identical in both projects, so the
#: physical model is the only thing that differs between them.
_READINGS = {
    CUM: [100.0, 100.5, pd.NA, 102.0],
    DIFF: [pd.NA, 1.2, pd.NA, pd.NA],
    f"{DIFF}_avg": [pd.NA, 1.0, 1.0, 1.0],
    "K": [pd.NA, 0.5, pd.NA, pd.NA],
}


def _constraints_for_models(*rows):
    """A compiler-shaped table; each row is (lhs, rhs, {var: sign}, models).

    Local because the rule rows and their physical_models lists are what this
    file is about, and no shared builder owns a compiled constraint table.
    """
    records = []
    for lhs, rhs_text, signs, models in rows:
        for variable, sign in signs.items():
            records.append({
                "lhs": lhs,
                "rhs_text": rhs_text,
                "variable": variable,
                "sign": sign,
                "derivative_text": str(sign),
                "rhs_var_count": len(signs),
                "physical_models": list(models),
            })
    table = pd.DataFrame(records, columns=list(CONSTRAINT_COLUMNS))
    return table.astype({"sign": "Int64", "rhs_var_count": "Int64"})


def _two_project_frame():
    """One household per project, identical data: only the model differs."""
    return household_batches_frame(
        {one_to_one_test_household_batch_of(household): _READINGS
         for household in PROJECT_OF_HOUSEHOLD},
        project_of={one_to_one_test_household_batch_of(household): project
                    for household, project in PROJECT_OF_HOUSEHOLD.items()},
        start="2026-01-01",
        freq="5min",
        tz=None,
        # Both halves of the pair are stated here, so there is nothing to
        # complete.
        pairs=False,
    )


class TestThePackageSelectsByModel:
    TABLE = _constraints_for_models(
        ((DIFF), "K", {"K": 1}, ["All-Electric"]),
        ((DIFF), "L", {"L": 1}, ["All-Electric", "Hybride", "Universeel"]),
    )

    def test_a_model_keeps_only_its_rules(self):
        package = build_constraint_package(
            self.TABLE, cadence_minutes=5, physical_model="Hybride"
        )
        expressions = {
            rule.expression for rules in package.rules_by_target.values()
            for rule in rules
        }
        assert expressions == {"L"}

    def test_the_named_model_keeps_both(self):
        package = build_constraint_package(
            self.TABLE, cadence_minutes=5, physical_model="All-Electric"
        )
        expressions = {
            rule.expression for rules in package.rules_by_target.values()
            for rule in rules
        }
        assert expressions == {"K", "L"}

    def test_no_model_keeps_every_rule(self):
        """Only right for a caller that has already scoped its rows, which is
        what the unit tests around the narrowing do."""
        package = build_constraint_package(
            self.TABLE, cadence_minutes=5, physical_model=None
        )
        expressions = {
            rule.expression for rules in package.rules_by_target.values()
            for rule in rules
        }
        assert expressions == {"K", "L"}


class TestTheFillJoinsModelPerRow:
    def test_a_restricted_rule_narrows_only_its_model_s_households(self):
        # Identical data in both projects; the rule follows K and applies to
        # All-Electric only. The delivered 1.2 sits outside K's [0.5, 0.5]
        # and inside the declared bound, so the signal fires exactly where
        # the rule applies.
        frame = _two_project_frame()
        out, _, _ = impute_and_normalize(
            frame,
            [CUM],
            ids.PROJECT,
            cadence=cadence_table(frame, [CUM]),
            constraints=_constraints_for_models(
                ((DIFF), "K", {"K": 1}, ["All-Electric"]),
            ),
            physical_models={1: "All-Electric", 2: "Hybride"},
        )
        by_household_batch = {
            household: out[
                out[ids.HOUSEHOLD_BATCH] == one_to_one_test_household_batch_of(household)
            ].reset_index(drop=True)
            for household in (1, 2)
        }
        flagged = int(by_household_batch[1].loc[1, ITYPE])
        assert flagged & ImputeType.DELIVERED_OUTSIDE_RULE_BOUND.value
        other = by_household_batch[2].loc[1, ITYPE]
        assert pd.isna(other) or not (
            int(other) & ImputeType.DELIVERED_OUTSIDE_RULE_BOUND.value
        )

    def test_an_unregistered_project_raises(self):
        frame = _two_project_frame()
        with pytest.raises(ValueError, match="ProjectIdBSV \\[2\\]"):
            impute_and_normalize(
                frame,
                [CUM],
                ids.PROJECT,
                cadence=cadence_table(frame, [CUM]),
                constraints=_constraints_for_models(
                    ((DIFF), "K", {"K": 1}, ["All-Electric"]),
                ),
                physical_models={1: "All-Electric"},
            )
