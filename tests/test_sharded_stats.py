"""
Wide-stage stats (etdtransform.data_stats.get_data_stats).

Two subjects. **What it computes** -- a household's stats come from that
household's rows and no others. **How it degrades** -- a column whose variance
overflows float64 reports NA with a warning rather than killing the report.

Cross-FORMAT reading is `etdmap.storage`'s subject and is tested there.
"""


import pandas as pd

import etdtransform
from etdtransform.data_stats import get_data_stats
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.pipeline]





def test_variance_overflow_reports_na_not_crash(tmp_path, caplog):
    """A column whose values are large enough that variance overflows float64
    (found on ZelfgebruikPercentage: a ratio with near-zero denominators) must
    not kill the whole stats report: that column's stats become NA with a
    factual warning, counts are kept, and every other column reports normally
    (ADR-022: NA = 'we do not know', stated visibly)."""
    from conftest import household_frame, write_shard_tree

    # A hive tree, because that is the shape a wide stage artifact has when
    # the stats are collected over it in production.
    root = write_shard_tree(tmp_path / "household_calculated", {
        1: household_frame(
            n_rows=10,
            Normal=[float(i) for i in range(10)],
            HugeRatio=[1e200, 2e200] * 5,
        )
    })

    stats = get_data_stats(str(root), precision="exact")

    assert {"Normal", "HugeRatio"}.issubset(set(stats["column"]))
    normal = stats[stats["column"] == "Normal"].iloc[0]
    assert float(normal["std"]) > 0
    huge = stats[stats["column"] == "HugeRatio"].iloc[0]
    assert int(huge["count"]) == 10  # counts survive
    assert pd.isna(huge["std"]) and pd.isna(huge["mean"])
    warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any("HugeRatio" in w for w in warnings)


#: Per household, a block of values whose mean, min and max share no value
#: with the other household's -- so stats pooled across households would be
#: wrong rather than close. Pooled, both means would read 7.0.
_PER_HOUSEHOLD = {1: [0.0, 1.0, 2.0, 3.0, 4.0], 2: [10.0, 11.0, 12.0, 13.0, 14.0]}
_EXPECTED = {1: {"mean": 2.0, "min": 0.0, "max": 4.0, "count": 5},
             2: {"mean": 12.0, "min": 10.0, "max": 14.0, "count": 5}}


@pytest.fixture
def per_household_stats(tmp_path):
    """Stats over a sharded source holding two households."""
    from conftest import household_frame, write_shard_tree

    root = write_shard_tree(tmp_path / "stage", {
        hid: household_frame(n_rows=len(values), Measured=values)
        for hid, values in _PER_HOUSEHOLD.items()
    })
    return get_data_stats(str(root), precision="exact")


def test_stats_are_computed_over_each_households_own_rows(per_household_stats):
    """A household's stats come from its rows and no others.

    The monolith-versus-shards comparison protected this indirectly. Stated
    directly it also catches the case a comparison cannot: both layouts
    pooling households together.
    """
    measured = per_household_stats[per_household_stats["column"] == "Measured"]
    assert not measured.empty, "no stats row for the Measured column"
    for hid, expected in _EXPECTED.items():
        rows = measured[measured["HuisIdBSV"] == hid]
        assert len(rows) == 1, f"household {hid}: {len(rows)} stats rows"
        row = rows.iloc[0]
        assert int(row["count"]) == expected["count"]
        assert float(row["mean"]) == pytest.approx(expected["mean"])
        assert float(row["min"]) == pytest.approx(expected["min"])
        assert float(row["max"]) == pytest.approx(expected["max"])


def test_both_households_get_their_own_row(per_household_stats):
    measured = per_household_stats[per_household_stats["column"] == "Measured"]
    assert sorted(measured["HuisIdBSV"].unique().tolist()) == sorted(_PER_HOUSEHOLD)


