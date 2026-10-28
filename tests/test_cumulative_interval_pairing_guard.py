"""Both halves of every cumulative/interval pair must be present after mapping.

The mapping derives whichever form the supplier did not deliver, so a set
holding one half means that conversion did not happen -- and every later stage
reads the missing half as the variable being absent rather than as a skipped
step, which is a silent wrong answer rather than a loud one.

The pairing comes from `ComplementVariabele` through
`etdmap.data_model.complement_column`, never from the `Diff` suffix: the real
model has a pair that does not follow it.
"""

import pytest

from etdmap.data_model import complement_column
from etdtransform.impute import assert_cumulative_interval_pairs_complete
from etdmap import ids

#: A real pair from the committed data model whose names do NOT follow the
#: Diff convention -- the case a suffix-based guard gets wrong.
CUMULATIVE = "ZonopwekTotaal"
INTERVAL = "ZonopwekBruto"

#: A real pair that does follow it.
PLAIN_CUMULATIVE = "Gasgebruik"
PLAIN_INTERVAL = "GasgebruikDiff"


def test_the_fixture_pairs_are_what_the_model_declares():
    # If the data model changes these names the test must fail loudly rather
    # than silently stop testing the thing it names.
    assert complement_column(CUMULATIVE) == INTERVAL
    assert complement_column(INTERVAL) == CUMULATIVE
    assert complement_column(PLAIN_CUMULATIVE) == PLAIN_INTERVAL


class TestCompletePairsPass:
    def test_both_halves_present(self):
        assert_cumulative_interval_pairs_complete([CUMULATIVE, INTERVAL])

    def test_the_non_suffix_pair_is_not_reported_broken(self):
        # A guard built on "<column>Diff" would raise here.
        assert_cumulative_interval_pairs_complete(
            [CUMULATIVE, INTERVAL, PLAIN_CUMULATIVE, PLAIN_INTERVAL]
        )

    def test_columns_with_no_declared_complement_are_ignored(self):
        assert_cumulative_interval_pairs_complete(
            [ids.HOUSEHOLD, "ReadingDate", ids.PROJECT]
        )

    def test_an_empty_set_passes(self):
        assert_cumulative_interval_pairs_complete([])


class TestHalfPairsRaise:
    def test_cumulative_without_its_interval(self):
        with pytest.raises(ValueError) as caught:
            assert_cumulative_interval_pairs_complete([CUMULATIVE])
        message = str(caught.value)
        assert CUMULATIVE in message and INTERVAL in message

    def test_interval_without_its_cumulative(self):
        # Both directions, so the guard does not depend on which half is held.
        with pytest.raises(ValueError) as caught:
            assert_cumulative_interval_pairs_complete([INTERVAL])
        message = str(caught.value)
        assert INTERVAL in message and CUMULATIVE in message

    def test_every_violation_is_named_in_one_raise(self):
        # One raise listing all of them, not the first one found: a caller
        # fixing these wants the whole list.
        with pytest.raises(ValueError) as caught:
            assert_cumulative_interval_pairs_complete(
                [CUMULATIVE, PLAIN_CUMULATIVE]
            )
        message = str(caught.value)
        assert INTERVAL in message
        assert PLAIN_INTERVAL in message

    def test_the_message_points_at_the_mapping_stage(self):
        with pytest.raises(ValueError, match="mapping"):
            assert_cumulative_interval_pairs_complete([CUMULATIVE])


class TestTheCalculatedStageCallsIt:
    """Wired at the stage entry, so a half pair stops the run rather than
    quietly changing the derivation plan. One stage call on a tiny parquet --
    no pipeline."""

    @staticmethod
    def _source(tmp_path, columns):
        """A sharded imputed artifact holding only *columns*.

        The frame is spelled out rather than built by `dataset_frame`,
        because that helper completes every pair from the model and a HALF
        pair is exactly what this fixture has to hold. The LAYOUT is the real
        one, so the guard is shown firing on the shape the stage actually
        reads.
        """
        import pandas as pd_
        from conftest import write_shard_tree

        n = 4
        frame = pd_.DataFrame({
            ids.PROJECT: pd_.array([1] * n, dtype="Int64"),
            "ReadingDate": pd_.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
            **{c: pd_.array([1.0] * n, dtype="Float64") for c in columns},
        })
        return write_shard_tree(tmp_path / "household_imputed", {1: frame})

    def test_a_half_pair_in_the_source_raises(self, tmp_path):
        from etdtransform.aggregate import add_calculated_columns_to_hh_data_ibis

        source = self._source(tmp_path, [CUMULATIVE])
        with pytest.raises(ValueError) as caught:
            add_calculated_columns_to_hh_data_ibis(
                source_path=str(source),
                output_path=str(tmp_path / "out.parquet"),
            )
        message = str(caught.value)
        assert CUMULATIVE in message and INTERVAL in message

    def test_it_raises_before_anything_is_written(self, tmp_path):
        from etdtransform.aggregate import add_calculated_columns_to_hh_data_ibis

        source = self._source(tmp_path, [CUMULATIVE])
        output = tmp_path / "out.parquet"
        with pytest.raises(ValueError):
            add_calculated_columns_to_hh_data_ibis(
                source_path=str(source), output_path=str(output)
            )
        assert not output.exists(), "the guard must stop the run before the write"
