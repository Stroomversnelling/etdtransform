"""
Wide-stage stats (etdtransform.data_stats.get_data_stats).

Two subjects. **What it computes** -- a household batch dataset's stats come
from that dataset's rows and no others. **How it degrades** -- a column whose
variance overflows float64 reports NA with a warning rather than killing the
report.

Cross-FORMAT reading is `etdmap.storage`'s subject and is tested there.
"""


import pandas as pd

import etdtransform
from etdtransform.data_stats import get_data_stats
import pytest
from etdmap import ids

pytestmark = [pytest.mark.integration, pytest.mark.pipeline]





def test_variance_overflow_reports_na_not_crash(tmp_path, caplog):
    """A column whose values are large enough that variance overflows float64
    (found on ZelfgebruikPercentage: a ratio with near-zero denominators) must
    not kill the whole stats report: that column's stats become NA with a
    factual warning, counts are kept, and every other column reports normally
    (ADR-022: NA = 'we do not know', stated visibly)."""
    from conftest import dataset_frame, write_shard_tree

    # A hive tree, because that is the shape a wide stage artifact has when
    # the stats are collected over it in production.
    root = write_shard_tree(tmp_path / "household_calculated", {
        1: dataset_frame(
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


#: Two household batch datasets of ONE household -- the only shape in which
#: keying on the dataset differs from keying on the dwelling. Each block's
#: mean, min and max share no value with the other's, so stats pooled across
#: the two would be wrong rather than close: pooled, both means read 7.0.
_HOUSEHOLD = 1
_PER_DATASET = {101: [0.0, 1.0, 2.0, 3.0, 4.0], 102: [10.0, 11.0, 12.0, 13.0, 14.0]}
_EXPECTED = {101: {"mean": 2.0, "min": 0.0, "max": 4.0, "count": 5},
             102: {"mean": 12.0, "min": 10.0, "max": 14.0, "count": 5}}


@pytest.fixture
def per_dataset_stats(tmp_path):
    """Stats over a sharded source holding one household's two datasets."""
    from conftest import dataset_frame, write_shard_tree

    root = write_shard_tree(tmp_path / "stage", {
        (_HOUSEHOLD, batch_id): dataset_frame(n_rows=len(values), Measured=values)
        for batch_id, values in _PER_DATASET.items()
    })
    return get_data_stats(str(root), precision="exact")


def test_stats_are_computed_over_each_datasets_own_rows(per_dataset_stats):
    """A dataset's stats come from its rows and no others.

    Both datasets here belong to one household, so a producer keyed on the
    dwelling returns ONE row per column holding the pooled figures. The
    monolith-versus-shards comparison could not catch that, because both
    layouts would pool identically.
    """
    measured = per_dataset_stats[per_dataset_stats["column"] == "Measured"]
    assert not measured.empty, "no stats row for the Measured column"
    for batch_id, expected in _EXPECTED.items():
        rows = measured[measured[ids.HOUSEHOLD_BATCH] == batch_id]
        assert len(rows) == 1, f"dataset {batch_id}: {len(rows)} stats rows"
        row = rows.iloc[0]
        assert int(row["count"]) == expected["count"]
        assert float(row["mean"]) == pytest.approx(expected["mean"])
        assert float(row["min"]) == pytest.approx(expected["min"])
        assert float(row["max"]) == pytest.approx(expected["max"])


def test_a_dataset_holding_no_value_still_gets_a_row(tmp_path):
    """A dataset present in the artifact but with nothing in a column gets a
    row for it anyway, reporting count 0.

    This is what the (dataset x column) cross product is reinstated for, and
    it is the only shape in which the per-dataset row count can be wrong:
    with every dataset holding values, any grouping at all returns one row
    each, so a fixture without an empty dataset asserts nothing.

    Absent rather than zero: a dataset with no reading for a variable has not
    measured nothing, it has measured nothing at all, and the report says so by
    carrying the row with a count of 0 instead of omitting it.
    """
    from conftest import dataset_frame, write_shard_tree

    root = write_shard_tree(tmp_path / "stage", {
        1: dataset_frame(n_rows=5, Measured=[0.0, 1.0, 2.0, 3.0, 4.0]),
        2: dataset_frame(n_rows=5, Measured=[pd.NA] * 5),
    })
    stats = get_data_stats(str(root), precision="exact")

    measured = stats[stats["column"] == "Measured"]
    assert sorted(measured[ids.HOUSEHOLD_BATCH].unique().tolist()) == [1, 2]
    empty = measured[measured[ids.HOUSEHOLD_BATCH] == 2]
    assert len(empty) == 1
    assert int(empty.iloc[0]["count"]) == 0


