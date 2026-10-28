"""
Stage 4: the calculated columns.

Two subjects. **What the stage computes** -- a derived column comes from its
own row's inputs, so how rows are grouped into files cannot change it. **What
the writer refuses** -- a source with duplicate (ids.HOUSEHOLD, ReadingDate) rows
(overlapping household batches), or one with no ids.HOUSEHOLD_BATCH to partition on.

Cross-FORMAT reading (resolving a monolith or a hive tree) is
`etdmap.storage`'s subject and is tested there.
"""

import shutil

import pytest
from etdmap import ids

from etdtransform.aggregate import add_calculated_columns_to_hh_data_ibis

pytestmark = [pytest.mark.integration, pytest.mark.pipeline]


class TestSourceGuards:
    """What the writer refuses."""

    def test_duplicate_reading_dates_in_sharded_source_raise(self, tmp_path,
                                                            fresh_shards):
        # Copies a household batch directory in, so it needs a tree of its own.
        from conftest import shard_dir

        shard_src = fresh_shards("household_imputed")
        # Duplicate household 1 under a second household batch: overlapping periods.
        src1 = next(shard_dir(shard_src, 1).parent.glob("HuisBatchIdBSV=*"))
        shutil.copytree(src1, shard_dir(shard_src, 1, 999))

        with pytest.raises(ValueError):
            add_calculated_columns_to_hh_data_ibis(
                source_path=str(shard_src),
                output_path=str(tmp_path / "out"),
            )

    def test_a_source_without_a_batch_column_is_refused(self, tmp_path):
        # The output is one shard per (household, household batch), so a source with no
        # household batch id to file the rows under is a caller error and must fail
        # before anything is written. Stated with a frame that simply has no
        # ids.HOUSEHOLD_BATCH, rather than borrowing an artifact that happens to
        # lack one.
        from conftest import household_batch_frame

        source = tmp_path / "no_batch_id.parquet"
        household_batch_frame(ElektriciteitNetgebruikDiff=1.0).to_parquet(
            source, index=False
        )
        with pytest.raises(ValueError, match=ids.HOUSEHOLD_BATCH):
            add_calculated_columns_to_hh_data_ibis(
                source_path=str(source),
                output_path=str(tmp_path / "out"),
            )


#: Per household: (teruglevering, netgebruik). Distinct per household and
#: chosen so that a derivation reaching across households cannot land on the
#: right answer -- the exchanges are 1.0, -4.0 and 7.0, and no two households
#: share an input value.
_PER_HOUSEHOLD = {1: (2.0, 3.0), 2: (5.0, 1.0), 3: (0.0, 7.0)}

_TERUG = "ElektriciteitTerugleveringDiff"
_NETG = "ElektriciteitNetgebruikDiff"
_ROWS = 4


class TestDerivationIsPerRow:
    """A derived column is computed from its own row's inputs, so how rows are
    grouped into files cannot change it.

    This is the property the monolith-versus-shards comparison was protecting
    indirectly. Asserted directly it is stronger: two layouts can agree while
    both are wrong, which is how a 4.5 stood in for a 6.0 elsewhere in this
    suite.
    """

    TARGETS = ["TerugleveringTotaalNetto", "ElektriciteitsgebruikTotaalNetto",
               "Netuitwisseling"]

    @pytest.fixture
    def derived(self, tmp_path):
        from conftest import household_batch_frame, write_shard_tree

        src = write_shard_tree(tmp_path / "imputed_shards", {
            hid: household_batch_frame(n_rows=_ROWS, **{_TERUG: terug, _NETG: netg})
            for hid, (terug, netg) in _PER_HOUSEHOLD.items()
        })
        out = tmp_path / "calculated_shards"
        add_calculated_columns_to_hh_data_ibis(
            source_path=str(src), output_path=str(out),
            target_columns=self.TARGETS,
        )
        from etdmap.storage import read_source_frame

        return read_source_frame(str(out))

    def test_every_household_keeps_its_own_values(self, derived):
        for hid, (terug, netg) in _PER_HOUSEHOLD.items():
            rows = derived[derived[ids.HOUSEHOLD] == hid]
            assert len(rows) == _ROWS, f"household {hid}: {len(rows)} rows"
            assert rows["TerugleveringTotaalNetto"].astype(float).tolist() == [terug] * _ROWS
            assert rows["ElektriciteitsgebruikTotaalNetto"].astype(float).tolist() == [netg] * _ROWS

    def test_the_exchange_is_each_households_own_difference(self, derived):
        # netgebruik - teruglevering, per household: 1.0, -4.0, 7.0. A
        # derivation mixing households could not produce all three.
        for hid, (terug, netg) in _PER_HOUSEHOLD.items():
            rows = derived[derived[ids.HOUSEHOLD] == hid]
            assert rows["Netuitwisseling"].astype(float).tolist() == [netg - terug] * _ROWS

    def test_no_household_is_dropped_or_duplicated(self, derived):
        assert sorted(derived[ids.HOUSEHOLD].unique().tolist()) == sorted(_PER_HOUSEHOLD)
        assert len(derived) == _ROWS * len(_PER_HOUSEHOLD)
