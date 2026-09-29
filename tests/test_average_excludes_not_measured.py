"""What may shape a project average.

An average is a statement about what households consumed at a time of day, so
only values measured at that time may shape it.

**A threshold cannot enforce this, and the reason is not that the values are
small.** Once a frozen span has been treated, its volume is spread across the
span and every resulting value lands comfortably *inside* the bounds -- 600 L
spread over 21 intervals is 28.57 each against a bound of 100. They look like
ordinary readings. The cumulative variable never reported any of them.

So the exclusion is by flag, and the flag records how the value came to be there
rather than how large it is:

**Invented** -- filled down, filled to zero, or redistributed across a frozen
span. Nothing was measured at this slot.

**Mistimed** -- the catch-up reading itself. It was measured, but for a window
rather than for this slot, and treatment has since blanked and refilled it.
"""

import logging

import pandas as pd
import pytest

from etdmap.flags import MappingFlag, flag_column
from etdmap.mapping_helpers import (
    _detect_frozen_spans_for_household,
    _treat_frozen_spans,
)
from etdtransform.impute import calculate_average_diff
from etdtransform.vectorized_impute import impute_and_normalize
from conftest import cadence_table

pytestmark = pytest.mark.unit

CUM = "WatergebruikWarmTapwater"
DIFF = f"{CUM}Diff"
FLAGS = flag_column(DIFF)
NORMAL = 3.0
GRID_BOUND = 100.0


@pytest.fixture(autouse=True)
def _quiet_logs():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


#: A catch-up just over the detection bound. Spread across the freeze this gives
#: a per-interval value close to normal consumption, which is the case that
#: matters: the household survives the relative outlier filter, and the values
#: survive the threshold, so the flag is the only thing that can exclude them.
#: A huge catch-up would be caught by both of those anyway.
MODEST_CATCHUP = 110.0


def _household_with_a_treated_span(hh, catchup=MODEST_CATCHUP, n=40, freeze=(10, 29)):
    """One household whose cumulative variable froze and then caught up, put
    through the real path: detect at the column bound, treat, impute.

    Returns the frame as it reaches averaging -- the span redistributed, the
    flags recording it.
    """
    first, last = freeze
    values, current = [], 1000.0
    for i in range(n):
        if first <= i <= last:
            pass
        elif i == last + 1:
            current += catchup
        else:
            current += NORMAL
        values.append(current)

    df = pd.DataFrame({
        "HuisIdBSV": pd.array([hh] * n, dtype="Int64"),
        "ProjectIdBSV": pd.array([1] * n, dtype="Int64"),
        "ReadingDate": pd.date_range("2026-01-01", periods=n, freq="5min"),
        CUM: pd.array(values, dtype="Float64"),
    })
    df[DIFF] = df[CUM].diff().round(10)
    df.loc[0, DIFF] = 0
    df[f"{DIFF}_avg"] = pd.array([NORMAL] * n, dtype="Float64")

    rows = _detect_frozen_spans_for_household(df, hh)
    assert rows, "the fixture must produce a detected span"
    _treat_frozen_spans(df, rows)
    out, _, _ = impute_and_normalize(
        df.copy(), [CUM], "ProjectIdBSV", None, cadence=cadence_table(df, [CUM])
    )
    return out


def _clean_household(hh, n=40):
    """A household with nothing wrong with it.

    The first interval is 0, as it is for any diff series: there is no previous
    reading to difference against. The span household's fixture does the same, so
    the two are comparable at every slot.
    """
    df = pd.DataFrame({
        "HuisIdBSV": pd.array([hh] * n, dtype="Int64"),
        "ProjectIdBSV": pd.array([1] * n, dtype="Int64"),
        "ReadingDate": pd.date_range("2026-01-01", periods=n, freq="5min"),
        DIFF: pd.array([0.0] + [NORMAL] * (n - 1), dtype="Float64"),
        FLAGS: pd.array([0] * n, dtype="Int64"),
    })
    return df


def _project(n_clean=19, with_span=1):
    """A project of clean households plus some that had a span treated."""
    frames = [_clean_household(hh) for hh in range(1, n_clean + 1)]
    frames += [
        _household_with_a_treated_span(hh)
        for hh in range(n_clean + 1, n_clean + with_span + 1)
    ]
    keep = ["HuisIdBSV", "ProjectIdBSV", "ReadingDate", DIFF, FLAGS]
    return pd.concat([f[keep] for f in frames], ignore_index=True)


def _average(df):
    result = calculate_average_diff(
        df, "ProjectIdBSV", [DIFF],
        cadence=cadence_table(df, [CUM]), grid_minutes=5,
    )[DIFF]
    avg = result["avg_diff"].sort_values("ReadingDate")[f"{DIFF}_avg"].astype(float)
    # Slot 0 is zero for every household -- a diff series has no previous
    # reading to difference against -- so it says nothing about the exclusion.
    return avg.iloc[1:].reset_index(drop=True), result


class TestRedistributedValuesLookOrdinary:
    def test_they_land_inside_the_bounds(self):
        """The reason a threshold cannot exclude them: redistribution puts the
        volume inside the bounds. 110 L over 21 intervals is ~5.2 each, against a
        bound of 100 -- and close enough to the normal 3.0 that the relative
        outlier filter keeps the household too."""
        out = _household_with_a_treated_span(1)
        touched = (
            out[FLAGS]
            & (MappingFlag.INVENTED.value | MappingFlag.CATCHUP.value)
        ) != 0
        values = out.loc[touched, DIFF].astype(float)
        assert int(touched.sum()) == 21
        assert float(values.max()) < GRID_BOUND

    def test_but_every_one_of_them_is_flagged(self):
        out = _household_with_a_treated_span(1)
        touched = (
            out[FLAGS]
            & (MappingFlag.INVENTED.value | MappingFlag.CATCHUP.value)
        ) != 0
        assert int(touched.sum()) == 21


class TestTheAverageExcludesThem:
    def test_the_threshold_excludes_nothing_here(self):
        _, result = _average(_project())
        assert result["n_discarded"] == 0, "the redistributed values are in bounds"

    def test_the_flag_excludes_them(self):
        _, result = _average(_project())
        assert result["n_flagged"] == 21

    def test_the_average_is_the_clean_households_level(self):
        avg, _ = _average(_project())
        assert float(avg.mean()) == pytest.approx(NORMAL, abs=1e-9)

    def test_without_the_flags_the_redistributed_values_shape_the_average(self):
        """What the exclusion buys. The spread values are wrong at every slot --
        too high across the freeze, and the reading that held the volume is gone."""
        df = _project()
        _, with_flags = _average(df)
        avg_without, without = _average(df.drop(columns=[FLAGS]))
        assert without["n_flagged"] == 0
        assert float(avg_without.mean()) != pytest.approx(NORMAL, abs=1e-6)


class TestNothingChangesWhereThereIsNothingToExclude:
    def test_a_project_with_no_spans_is_unaffected(self):
        df = _project(n_clean=20, with_span=0)
        avg, result = _average(df)
        assert result["n_flagged"] == 0
        assert float(avg.mean()) == pytest.approx(NORMAL, abs=1e-9)

    def test_data_without_a_flags_column_still_works(self):
        """Mapped data produced before flags existed has no findings column."""
        df = _project(n_clean=20, with_span=0).drop(columns=[FLAGS])
        avg, result = _average(df)
        assert result["n_flagged"] == 0
        assert float(avg.mean()) == pytest.approx(NORMAL, abs=1e-9)

    @pytest.mark.parametrize(
        "flag",
        [MappingFlag.FILLED_DOWN, MappingFlag.FILLED_ZERO, MappingFlag.FROZEN_SPAN],
    )
    def test_every_invented_kind_is_kept_out(self, flag):
        df = _project(n_clean=20, with_span=0)
        target = df.HuisIdBSV.isin([1, 2])
        df.loc[target, DIFF] = 99.0
        df.loc[target, FLAGS] = flag.value

        avg, result = _average(df)
        assert result["n_flagged"] == int(target.sum())
        assert float(avg.mean()) == pytest.approx(NORMAL, abs=1e-9)
