"""The fill order comes from the coverage and duty artifact.

Coverage descending, then duty descending, then alphanumerically by name --
so the columns whose averages exist most and move most are filled first, and
a later column's bounds rest on the best-known fills. A re-run must order
exactly as the run that produced the averages did, which is why the order is
read from `avg_coverage_duty.parquet` rather than recomputed.
"""

import pandas as pd
import pytest

from etdtransform.vectorized_impute import order_columns_for_fill
from etdmap import ids

pytestmark = pytest.mark.unit

# Real model pairs: the ordering resolves each cumulative to its interval
# column through the data model, exactly as the fill loop does.
NG = "ElektriciteitNetgebruik"
TL = "ElektriciteitTeruglevering"
WTW = "ElektriciteitsgebruikWTW"


def _table(*rows):
    """(diff column, project, coverage, duty) rows shaped like the artifact."""
    return pd.DataFrame({
        "column": pd.array([r[0] for r in rows], dtype="string"),
        ids.PROJECT: pd.array([r[1] for r in rows], dtype="Int64"),
        "coverage": pd.array([r[2] for r in rows], dtype="Float64"),
        "duty": pd.array([r[3] for r in rows], dtype="Float64"),
    })


class TestTheOrdering:
    def test_coverage_descends_first(self):
        table = _table(
            (f"{NG}Diff", 1, 0.5, 1.0),
            (f"{TL}Diff", 1, 0.9, 0.1),
        )
        assert order_columns_for_fill([NG, TL], table) == [TL, NG]

    def test_duty_breaks_a_coverage_tie(self):
        table = _table(
            (f"{NG}Diff", 1, 0.9, 0.2),
            (f"{TL}Diff", 1, 0.9, 0.8),
        )
        assert order_columns_for_fill([NG, TL], table) == [TL, NG]

    def test_the_name_breaks_a_full_tie(self):
        table = _table(
            (f"{TL}Diff", 1, 0.9, 0.5),
            (f"{NG}Diff", 1, 0.9, 0.5),
        )
        assert order_columns_for_fill([TL, NG], table) == [NG, TL]

    def test_the_median_across_projects_decides(self):
        # NG: median coverage 0.8 over three projects; TL: 0.6. One stray
        # project must not flip the order the majority sets.
        table = _table(
            (f"{NG}Diff", 1, 0.8, 0.5),
            (f"{NG}Diff", 2, 0.8, 0.5),
            (f"{NG}Diff", 3, 0.1, 0.5),
            (f"{TL}Diff", 1, 0.6, 0.5),
            (f"{TL}Diff", 2, 0.6, 0.5),
            (f"{TL}Diff", 3, 0.99, 0.5),
        )
        assert order_columns_for_fill([TL, NG], table) == [NG, TL]

    def test_a_column_the_table_does_not_carry_goes_last(self):
        table = _table((f"{WTW}Diff", 1, 0.1, 0.1))
        assert order_columns_for_fill([NG, TL, WTW], table) == [WTW, NG, TL]
