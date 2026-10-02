"""The three Netto composites in the legacy calculated path.

Characterization (captured against the pre-2026-10-01 code, documented in
TestOldBehaviourCharacterization's docstrings) and the replacement contract:
the composites derive from the combined tariff ROOTS -- delivered at mapping
by combine_tariff_registers -- with strict NA propagation. A missing root
yields NA, never a silent zero.
"""

import pandas as pd
import pytest

from etdtransform.calculated_columns import add_calculated_columns_imputed_data

N = 6


def _col(value):
    return pd.array([value] * N, dtype="Float64")


def _na_col():
    return pd.array([pd.NA] * N, dtype="Float64")


def _frame(**overrides) -> pd.DataFrame:
    """A minimal frame carrying every column the legacy path touches.

    Mirrors the mapped reality: rearrange_model_columns(add_columns=True)
    guarantees every model column exists, as all-NA where undelivered.
    """
    base = {
        # tariff roots (combined at mapping)
        "ElektriciteitTerugleveringDiff": _col(2.0),
        "ElektriciteitNetgebruikDiff": _col(3.0),
        # tariff splits (delivered registers, kept as mapped)
        "ElektriciteitTerugleveringLaagDiff": _col(1.5),
        "ElektriciteitTerugleveringHoogDiff": _col(0.5),
        "ElektriciteitNetgebruikLaagDiff": _col(2.0),
        "ElektriciteitNetgebruikHoogDiff": _col(1.0),
        # device columns the later composites read
        "ElektriciteitsgebruikWarmtepompDiff": _col(0.2),
        "ElektriciteitsgebruikBoosterDiff": _col(0.1),
        "ElektriciteitsgebruikBoilervatDiff": _col(0.1),
        "ElektriciteitsgebruikWTWDiff": _col(0.05),
        "ElektriciteitsgebruikRadiatorDiff": _col(0.0),
        "Zon-opwekTotaalDiff": _col(1.0),
    }
    base.update(overrides)
    return pd.DataFrame(base)


class TestNettoCompositesFromRoots:
    def test_values_equal_old_behaviour_where_both_splits_delivered(self):
        """Where both splits were delivered the root equals hoog+laag by the
        mapping combination, so the root-based composite equals the old
        splits-with-fillna(0) sum exactly."""
        df = add_calculated_columns_imputed_data(_frame(), fillna=True)
        assert df["TerugleveringTotaalNetto"].astype(float).tolist() == [2.0] * N
        assert df["ElektriciteitsgebruikTotaalNetto"].astype(float).tolist() == [3.0] * N
        assert df["Netuitwisseling"].astype(float).tolist() == [1.0] * N

    def test_missing_root_yields_na_not_zero(self):
        """BEHAVIOUR CHANGE, intended (ADR-022 strict NA): under the old
        code a household with no tariff data at all got composites of
        literal 0.0 (fillna(0) on both splits). Now a missing root
        propagates NA -- no data is never zero consumption."""
        df = add_calculated_columns_imputed_data(
            _frame(
                ElektriciteitTerugleveringDiff=_na_col(),
                ElektriciteitNetgebruikDiff=_na_col(),
                ElektriciteitTerugleveringLaagDiff=_na_col(),
                ElektriciteitTerugleveringHoogDiff=_na_col(),
                ElektriciteitNetgebruikLaagDiff=_na_col(),
                ElektriciteitNetgebruikHoogDiff=_na_col(),
            ),
            fillna=True,
        )
        assert df["TerugleveringTotaalNetto"].isna().all()
        assert df["ElektriciteitsgebruikTotaalNetto"].isna().all()
        assert df["Netuitwisseling"].isna().all()

    def test_split_less_supplier_uses_delivered_roots(self):
        """The pre-existing bug, fixed: a supplier delivering ONLY roots
        (splits all-NA) used to get composites of 0.0 -- the delivered
        roots were ignored. Now the roots carry the composites."""
        df = add_calculated_columns_imputed_data(
            _frame(
                ElektriciteitTerugleveringLaagDiff=_na_col(),
                ElektriciteitTerugleveringHoogDiff=_na_col(),
                ElektriciteitNetgebruikLaagDiff=_na_col(),
                ElektriciteitNetgebruikHoogDiff=_na_col(),
            ),
            fillna=True,
        )
        assert df["TerugleveringTotaalNetto"].astype(float).tolist() == [2.0] * N
        assert df["ElektriciteitsgebruikTotaalNetto"].astype(float).tolist() == [3.0] * N
        assert df["Netuitwisseling"].astype(float).tolist() == [1.0] * N

    def test_mixed_frame_rows_follow_their_own_root(self):
        """Rows with a root get values; rows without get NA -- per row,
        never per column."""
        terug = pd.array([2.0, pd.NA, 2.0, pd.NA, 2.0, pd.NA], dtype="Float64")
        net = pd.array([3.0, pd.NA, 3.0, pd.NA, 3.0, pd.NA], dtype="Float64")
        df = add_calculated_columns_imputed_data(
            _frame(
                ElektriciteitTerugleveringDiff=terug,
                ElektriciteitNetgebruikDiff=net,
            ),
            fillna=True,
        )
        expected_na = [False, True, False, True, False, True]
        assert df["TerugleveringTotaalNetto"].isna().tolist() == expected_na
        assert df["Netuitwisseling"].isna().tolist() == expected_na

    def test_na_branch_matches_fillna_branch_for_the_three_composites(self):
        """The three Netto composites are strict-NA in BOTH branches now;
        the fillna flag keeps governing only the other composites."""
        a = add_calculated_columns_imputed_data(_frame(), fillna=True)
        b = add_calculated_columns_imputed_data(_frame(), fillna=False)
        for col in (
            "TerugleveringTotaalNetto",
            "ElektriciteitsgebruikTotaalNetto",
            "Netuitwisseling",
        ):
            pd.testing.assert_series_equal(a[col], b[col], check_names=False)


class TestOldBehaviourCharacterization:
    """What the pre-change code actually did, captured 2026-10-01 before the
    rewrite (assertions here document the OLD behaviour and are inverted by
    the fix; the test asserts the NEW state and the docstring preserves the
    record):

    - fillna=True, splits all-NA, roots delivered: TerugleveringTotaalNetto,
      ElektriciteitsgebruikTotaalNetto and Netuitwisseling all came out 0.0
      on every row -- fillna(0) on both splits fabricated zero composites
      and the delivered roots were never read.
    - fillna=False carried a second latent bug: the continuation line
      `+ df[...]` stood alone as a no-op statement, so
      TerugleveringTotaalNetto silently equalled the Laag split only.
    """

    def test_fabricated_zeros_are_gone(self):
        df = add_calculated_columns_imputed_data(
            _frame(
                ElektriciteitTerugleveringLaagDiff=_na_col(),
                ElektriciteitTerugleveringHoogDiff=_na_col(),
                ElektriciteitNetgebruikLaagDiff=_na_col(),
                ElektriciteitNetgebruikHoogDiff=_na_col(),
            ),
            fillna=True,
        )
        assert not (df["TerugleveringTotaalNetto"] == 0.0).all()
        assert not (df["ElektriciteitsgebruikTotaalNetto"] == 0.0).all()
