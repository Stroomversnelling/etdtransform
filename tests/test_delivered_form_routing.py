"""Which logic the fill resolves a column by, and what each one may read.

Two logics share one pass. The cumulative one reads a delivered register
together with the intervals differenced from it, and measures a jump across a
gap to distribute. The interval one has no register, so it has no gap and no
jump: it fills each missing cell from the project average brought to the
household's level.

Routing is from the cadence sidecar, not from the values, because a register
whose every reading was later condemned still *was* delivered -- and the two
logics are told apart by what they are allowed to read.

Synthetic frames only -- no pipeline, no fixtures.
"""

import logging

import pandas as pd
import pytest

from etdtransform.vectorized_impute import (
    ImputeType,
    delivery_paths,
    impute_and_normalize,
)

pytestmark = pytest.mark.unit

GRID = 5
CUM = "ElektriciteitNetgebruik"
DIFF = f"{CUM}Diff"
OTHER_CUM = "Gasgebruik"
OTHER_DIFF = f"{OTHER_CUM}Diff"
N = 24

#: Flags that only mean something given a register to anchor on. None may land
#: on a cell the interval logic resolved.
NEEDS_A_REGISTER = (
    ImputeType.NEGATIVE_GAP_JUMP,
    ImputeType.NEAR_ZERO_GAP_JUMP,
    ImputeType.ZERO_END_VALUE,
    ImputeType.POSITIVE_END_VALUE,
    ImputeType.NO_END_VALUE,
    ImputeType.ONLY_OPENING_READING,
    ImputeType.ONLY_CLOSING_READING,
    ImputeType.NO_CUMULATIVE_VALUE,
    ImputeType.UNIFORM_FILL,
    ImputeType.SCALED_FILL,
    ImputeType.ZERO_FILL,
    ImputeType.DELIVERED_CUMULATIVE_IMPUTATION,
)


@pytest.fixture(autouse=True)
def _quiet_logs():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _cadence(rows):
    return pd.DataFrame({
        "HuisIdBSV": pd.array([r[0] for r in rows], dtype="Int64"),
        "column": pd.array([r[1] for r in rows], dtype="string"),
        "cadence_minutes": pd.array([r[2] for r in rows], dtype="Int64"),
        "n_readings": pd.array([r[3] for r in rows], dtype="Int64"),
    })


def _form_rows(household, cum_col, diff_col, register, intervals):
    """Sidecar rows stating which forms arrived for one (household, variable)."""
    return [
        (household, "ReadingDate", GRID, N),
        (household, cum_col, GRID, N if register else 0),
        (household, diff_col, GRID, N if intervals else 0),
    ]


def _register_frame(household=1, n=N, rise=0.5, gap=(8, 11)):
    """A delivered register with a run of missing intervals inside it."""
    total, values = 100.0, []
    for _ in range(n):
        total += rise
        values.append(total)
    df = pd.DataFrame({
        "HuisIdBSV": pd.array([household] * n, dtype="Int64"),
        "ProjectIdBSV": pd.array([1] * n, dtype="Int64"),
        "ReadingDate": pd.date_range("2026-01-01", periods=n, freq="5min"),
        CUM: pd.array(values, dtype="Float64"),
    })
    df[DIFF] = df[CUM].diff().round(10)
    df.loc[0, DIFF] = 0
    df.loc[gap[0]:gap[1], DIFF] = pd.NA
    df[f"{DIFF}_avg"] = pd.array([rise] * n, dtype="Float64")
    return df


def _interval_frame(household=1, n=N, value=0.5, missing=(8, 11)):
    """Delivered intervals at the grid, no register, with cells missing."""
    df = pd.DataFrame({
        "HuisIdBSV": pd.array([household] * n, dtype="Int64"),
        "ProjectIdBSV": pd.array([1] * n, dtype="Int64"),
        "ReadingDate": pd.date_range("2026-01-01", periods=n, freq="5min"),
        CUM: pd.array([pd.NA] * n, dtype="Float64"),
        DIFF: pd.array([value] * n, dtype="Float64"),
    })
    df.loc[missing[0]:missing[1], DIFF] = pd.NA
    df[f"{DIFF}_avg"] = pd.array([value] * n, dtype="Float64")
    return df


def _impute(df, cadence, columns=(CUM,)):
    out, _, _ = impute_and_normalize(
        df.copy(), list(columns), "ProjectIdBSV", cadence=cadence
    )
    return out


def _impute_with_constraints(df, rules, cadence=None):
    """The fill with a constraint table, so each row's bound comes from the
    rules and that row's own known values."""
    from test_constraint_package import _constraints

    out, _, _ = impute_and_normalize(
        df.copy(),
        [CUM],
        "ProjectIdBSV",
        cadence=cadence or _cadence(_form_rows(1, CUM, DIFF, False, True)),
        constraints=_constraints(*rules),
    )
    return out


def _types(out, diff_col=DIFF):
    return out[f"{diff_col}_impute_type"].fillna(0).astype("Int64")


def _has(out, flag, diff_col=DIFF):
    return ((_types(out, diff_col) & flag.value) != 0).tolist()


class TestTheFourDeliveryCombinations:
    def test_a_delivered_register_takes_the_cumulative_logic(self):
        df = _register_frame()
        paths = delivery_paths(
            df, CUM, DIFF, _cadence(_form_rows(1, CUM, DIFF, True, False))
        )
        assert bool(paths.cumulative.all())

    def test_both_forms_delivered_takes_the_cumulative_logic(self):
        """The register is authoritative, and the cumulative logic reads it
        together with the intervals differenced from it."""
        df = _register_frame()
        paths = delivery_paths(
            df, CUM, DIFF, _cadence(_form_rows(1, CUM, DIFF, True, True))
        )
        assert bool(paths.cumulative.all())

    def test_intervals_alone_take_the_interval_logic(self):
        df = _interval_frame()
        paths = delivery_paths(
            df, CUM, DIFF, _cadence(_form_rows(1, CUM, DIFF, False, True))
        )
        assert bool(paths.interval.all())

    def test_neither_form_takes_no_logic(self):
        df = _interval_frame()
        paths = delivery_paths(
            df, CUM, DIFF, _cadence(_form_rows(1, CUM, DIFF, False, False))
        )
        assert bool(paths.nothing.all())


class TestTheRoutingIsPerHousehold:
    def test_a_mixed_chunk_routes_each_household_separately(self):
        """One column, one pass, two households of opposite form. A per-call
        switch would send both the same way."""
        df = pd.concat(
            [_register_frame(household=1), _interval_frame(household=2)],
            ignore_index=True,
        )
        cadence = _cadence(
            _form_rows(1, CUM, DIFF, True, False)
            + _form_rows(2, CUM, DIFF, False, True)
        )
        paths = delivery_paths(df, CUM, DIFF, cadence)
        first = df["HuisIdBSV"] == 1
        assert bool(paths.cumulative[first].all())
        assert bool(paths.interval[~first].all())

    def test_a_second_column_with_the_forms_swapped_routes_the_other_way(self):
        """The same two households, opposite forms on a different variable. A
        switch cached per household or per call cannot pass both."""
        df = pd.concat(
            [_register_frame(household=1), _interval_frame(household=2)],
            ignore_index=True,
        )
        cadence = _cadence(
            _form_rows(1, CUM, DIFF, True, False)
            + _form_rows(2, CUM, DIFF, False, True)
            + [
                (1, OTHER_CUM, GRID, 0),
                (1, OTHER_DIFF, GRID, N),
                (2, OTHER_CUM, GRID, N),
                (2, OTHER_DIFF, GRID, 0),
            ]
        )
        first = df["HuisIdBSV"] == 1
        swapped = delivery_paths(df, OTHER_CUM, OTHER_DIFF, cadence)
        assert bool(swapped.interval[first].all())
        assert bool(swapped.cumulative[~first].all())


class TestTheGuards:
    def test_an_na_household_id_is_refused(self):
        """isin() reads pd.NA as no-match, so such a row would route silently
        into nothing-delivered and still satisfy the partition check."""
        df = _register_frame()
        df.loc[3, "HuisIdBSV"] = pd.NA
        with pytest.raises(ValueError, match="carry no HuisIdBSV"):
            delivery_paths(
                df, CUM, DIFF, _cadence(_form_rows(1, CUM, DIFF, True, False))
            )

    def test_a_household_absent_from_the_sidecar_is_refused(self):
        """Absent metadata is not a delivery fact: every row would classify as
        nothing-delivered and go unfilled."""
        df = _register_frame(household=7)
        with pytest.raises(ValueError, match="no row in the column-cadence"):
            delivery_paths(
                df, CUM, DIFF, _cadence(_form_rows(1, CUM, DIFF, True, False))
            )

    def test_no_cadence_table_is_refused(self):
        with pytest.raises(ValueError, match="no cadence table"):
            delivery_paths(_register_frame(), CUM, DIFF, None)

    def test_a_frame_without_household_ids_is_refused(self):
        df = _register_frame().drop(columns=["HuisIdBSV"])
        with pytest.raises(ValueError, match="no HuisIdBSV"):
            delivery_paths(
                df, CUM, DIFF, _cadence(_form_rows(1, CUM, DIFF, True, False))
            )

    def test_the_three_paths_partition_every_row(self):
        df = pd.concat(
            [
                _register_frame(household=1),
                _interval_frame(household=2),
                _interval_frame(household=3),
            ],
            ignore_index=True,
        )
        cadence = _cadence(
            _form_rows(1, CUM, DIFF, True, True)
            + _form_rows(2, CUM, DIFF, False, True)
            + _form_rows(3, CUM, DIFF, False, False)
        )
        paths = delivery_paths(df, CUM, DIFF, cadence)
        claims = (
            paths.cumulative.astype(int)
            + paths.interval.astype(int)
            + paths.nothing.astype(int)
        )
        assert set(claims.unique()) == {1}

    def test_no_path_mask_holds_a_missing_value(self):
        df = _register_frame()
        paths = delivery_paths(
            df, CUM, DIFF, _cadence(_form_rows(1, CUM, DIFF, True, False))
        )
        for mask in paths:
            assert not bool(mask.isna().any())

    def test_a_table_with_no_delivery_counts_keeps_the_cumulative_logic(self):
        """An unrecorded form must not stop the fill. The cumulative logic is
        what ran before this routing existed, so that is the fallback."""
        df = _register_frame()
        cadence = pd.DataFrame({
            "HuisIdBSV": pd.array([1], dtype="Int64"),
            "column": pd.array([CUM], dtype="string"),
            "cadence_minutes": pd.array([GRID], dtype="Int64"),
        })
        paths = delivery_paths(df, CUM, DIFF, cadence)
        assert bool(paths.cumulative.all())


class TestTheIntervalBranch:
    def test_a_missing_cell_is_filled_from_the_scaled_average(self):
        df = _interval_frame(value=0.5)
        out = _impute(df, _cadence(_form_rows(1, CUM, DIFF, False, True)))
        # The household runs at the average, so the factor is 1.0 and the fill
        # is the average itself.
        assert out.loc[8:11, DIFF].notna().all()
        assert out.loc[8:11, DIFF].astype(float).tolist() == pytest.approx(
            [0.5] * 4
        )

    def test_the_fill_is_recorded_as_a_household_scaled_average(self):
        out = _impute(
            _interval_frame(), _cadence(_form_rows(1, CUM, DIFF, False, True))
        )
        assert all(_has(out, ImputeType.HOUSEHOLD_SCALED_AVERAGE)[8:12])

    def test_the_branch_is_recorded(self):
        out = _impute(
            _interval_frame(), _cadence(_form_rows(1, CUM, DIFF, False, True))
        )
        assert all(_has(out, ImputeType.DELIVERED_INTERVAL_IMPUTATION)[8:12])

    def test_no_flag_that_needs_a_register_is_set(self):
        out = _impute(
            _interval_frame(), _cadence(_form_rows(1, CUM, DIFF, False, True))
        )
        for flag in NEEDS_A_REGISTER:
            assert not any(_has(out, flag)), flag.name

    def test_the_rows_are_not_treated_as_a_gap(self):
        """A gap is a run bounded by register readings. These rows have none, so
        the gap bookkeeping is blanked and every reader of it skips them -- here,
        the end-of-fill sweep, which would otherwise claim a structure they do
        not have."""
        out = _impute(
            _interval_frame(), _cadence(_form_rows(1, CUM, DIFF, False, True))
        )
        assert not any(_has(out, ImputeType.GAP_NOT_FILLED))

    def test_the_gap_counts_report_nothing_for_it(self):
        """The same blanking, seen from the statistics: they are derived from the
        gap structures, so an interval-only household reports no gaps rather
        than gaps of zero length. What it filled is in the branch signal and the
        fill action instead."""
        _, gap_stats, _ = impute_and_normalize(
            _interval_frame().copy(), [CUM], "ProjectIdBSV",
            cadence=_cadence(_form_rows(1, CUM, DIFF, False, True)),
        )
        row = gap_stats.iloc[0]
        assert int(row["missing"]) == 0
        assert int(row["imputed"]) == 0
        assert int(row["bitwise_methods"]) == (
            ImputeType.DELIVERED_INTERVAL_IMPUTATION.value
            | ImputeType.HOUSEHOLD_SCALED_AVERAGE.value
        )

    def test_a_cell_it_cannot_scale_is_left_empty_and_not_called_a_gap(self):
        """Too few comparable rows for a household factor: filling with the raw
        project average would assert a level the household has not shown."""
        df = _interval_frame(n=12, value=0.5, missing=(1, 10))
        out = _impute(df, _cadence(_form_rows(1, CUM, DIFF, False, True)))
        assert out.loc[1:10, DIFF].isna().all()
        assert not any(_has(out, ImputeType.GAP_NOT_FILLED))
        # Still recorded as this branch's, with no fill action: which logic
        # looked is a separate fact from what it managed to do.
        assert all(_has(out, ImputeType.DELIVERED_INTERVAL_IMPUTATION)[1:11])


class TestTheGapReadersSkipTheIntervalPath:
    """Three places read the gap bookkeeping. Blanking it on the interval path
    is what keeps all three from reporting a structure those rows lack."""

    def test_no_error_is_logged_about_cells_still_missing_in_a_gap(self, caplog):
        """The reader counts `diff.isna() & cumulative_value_group.notna()`. An
        interval-only household with cells it could not scale would otherwise
        look like a fill that failed."""
        df = _interval_frame(n=12, value=0.5, missing=(1, 10))
        with caplog.at_level(logging.ERROR):
            logging.disable(logging.NOTSET)
            _impute(df, _cadence(_form_rows(1, CUM, DIFF, False, True)))
        assert "still missing in" not in caplog.text

    def test_the_end_of_fill_sweep_does_not_claim_them(self):
        df = _interval_frame(n=12, value=0.5, missing=(1, 10))
        out = _impute(df, _cadence(_form_rows(1, CUM, DIFF, False, True)))
        assert not any(_has(out, ImputeType.GAP_NOT_FILLED))


class TestTheOnlyReadingOnRowZero:
    """add_diff_columns seeds row 0 of each household with a zero, so the
    usable-readings test excludes it. A household whose one delivered interval
    landed there has a delivery the sidecar records and no reading the fill can
    scale from -- the two facts the routing and the usable-readings mask keep
    apart."""

    def _frame(self, n=12):
        df = _interval_frame(n=n, value=0.5, missing=(0, n - 1))
        df.loc[0, DIFF] = 0.5
        return df

    def test_it_is_routed_to_the_interval_logic(self):
        paths = delivery_paths(
            self._frame(), CUM, DIFF,
            _cadence(_form_rows(1, CUM, DIFF, False, True)),
        )
        assert bool(paths.interval.all())

    def test_but_nothing_is_filled_from_it(self):
        out = _impute(
            self._frame(), _cadence(_form_rows(1, CUM, DIFF, False, True))
        )
        assert out.loc[1:, DIFF].isna().all()


class TestTheCumulativeBranchIsUnchangedApartFromItsSignal:
    def test_the_gap_is_still_filled_the_same_way(self):
        cadence = _cadence(_form_rows(1, CUM, DIFF, True, False))
        out = _impute(_register_frame(), cadence)
        assert out.loc[8:11, DIFF].notna().all()
        assert out.loc[8:11, DIFF].astype(float).tolist() == pytest.approx(
            [0.5] * 4
        )

    def test_the_branch_is_recorded(self):
        cadence = _cadence(_form_rows(1, CUM, DIFF, True, False))
        out = _impute(_register_frame(), cadence)
        assert all(_has(out, ImputeType.DELIVERED_CUMULATIVE_IMPUTATION)[8:12])

    def test_the_interval_branch_never_claims_its_cells(self):
        cadence = _cadence(_form_rows(1, CUM, DIFF, True, False))
        out = _impute(_register_frame(), cadence)
        assert not any(_has(out, ImputeType.DELIVERED_INTERVAL_IMPUTATION))


class TestTheIntervalBranchReadsTheRuleBounds:
    """The average the fill reads is clipped into each row's narrowed bound
    before any route sees it, and the narrowing comes from the constraint
    system. The interval branch multiplies that average by the household
    factor, so it has to be reading the clipped copy -- and it has to leave a
    row alone where the rules permit no value at all."""

    #: The row bound follows K exactly, with K unbounded in the thresholds, so
    #: every narrowing here is the row's own.
    FOLLOWS_K = ((DIFF, "K", {"K": 1}),)

    def _frame(self, k, value=0.5, missing=(8, 11), n=N):
        df = _interval_frame(n=n, value=value, missing=missing)
        df["K"] = pd.array(k, dtype="Float64")
        return df

    def test_the_fill_uses_the_average_clipped_into_the_row_bound(self):
        """The average says 0.5; the rules narrow these rows to 0.2. The branch
        must write 0.2, not 0.5."""
        k = [pd.NA] * N
        for row in range(8, 12):
            k[row] = 0.2
        out = _impute_with_constraints(self._frame(k), self.FOLLOWS_K)
        assert out.loc[8:11, DIFF].astype(float).tolist() == pytest.approx(
            [0.2] * 4
        )

    def test_a_clipped_average_is_signalled(self):
        k = [pd.NA] * N
        for row in range(8, 12):
            k[row] = 0.2
        out = _impute_with_constraints(self._frame(k), self.FOLLOWS_K)
        assert all(_has(out, ImputeType.CLIPPED_AVERAGE)[8:12])

    def test_a_row_the_rules_permit_no_value_for_is_left_empty(self):
        """An empty bound admits nothing, so the branch writes nothing -- the
        same rule the cumulative routes follow."""
        k = [pd.NA] * N
        for row in range(8, 12):
            k[row] = -1.0  # below the column's own Min of 0, so no value fits
        out = _impute_with_constraints(self._frame(k), self.FOLLOWS_K)
        assert out.loc[8:11, DIFF].isna().all()

    def test_the_rule_bound_is_built_at_the_cadence_the_values_are_on(self):
        """The constraint packages are grouped by cadence. These values sit at
        the grid, so the package that narrows them must be the grid one -- a
        bound built for an hourly reading would be k times too loose."""
        k = [pd.NA] * N
        for row in range(8, 12):
            k[row] = 0.2
        out = _impute_with_constraints(self._frame(k), self.FOLLOWS_K)
        # 0.2 is what one grid slot may hold under the rule; an hourly package
        # would have admitted the unclipped 0.5.
        assert out.loc[8:11, DIFF].astype(float).tolist() == pytest.approx(
            [0.2] * 4
        )


class TestExactlyOneBranchSignal:
    def test_no_routed_cell_carries_both(self):
        df = pd.concat(
            [_register_frame(household=1), _interval_frame(household=2)],
            ignore_index=True,
        )
        cadence = _cadence(
            _form_rows(1, CUM, DIFF, True, False)
            + _form_rows(2, CUM, DIFF, False, True)
        )
        out = _impute(df, cadence)
        both = [
            cumulative and interval
            for cumulative, interval in zip(
                _has(out, ImputeType.DELIVERED_CUMULATIVE_IMPUTATION),
                _has(out, ImputeType.DELIVERED_INTERVAL_IMPUTATION),
            )
        ]
        assert not any(both)

    def test_a_household_the_supplier_sent_nothing_for_carries_neither(self):
        """Nothing-delivered is not a logic: no route reads it, nothing is
        written, and the cells stay empty."""
        df = _interval_frame()
        df[DIFF] = pd.array([pd.NA] * N, dtype="Float64")
        out = _impute(df, _cadence(_form_rows(1, CUM, DIFF, False, False)))
        assert not any(_has(out, ImputeType.DELIVERED_CUMULATIVE_IMPUTATION))
        assert not any(_has(out, ImputeType.DELIVERED_INTERVAL_IMPUTATION))
        assert out[DIFF].isna().all()
