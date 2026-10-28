"""The upspread step of the imputation pass.

A supplier that delivers an interval column coarsely sends one reading per k
grid slots: the consumption of a whole hour sitting in a single five-minute
cell, with the cells around it empty. Nothing downstream can read that as a
grid-cadence column -- the empty cells report "unknown" for periods whose
energy is in hand, and a bound written per slot judges a reading holding k of
them.

So the reading is placed on the k slots it covers, before the fill looks for
what is missing. The averaging does its own spreading, inside the aggregation
it runs as SQL; this is the fill's, on the frame it is about to read.

Only a variable delivered as intervals only is spread. Where a register exists
it is authoritative and the cumulative logic reads it, at its own cadence.
"""

import pandas as pd

from etdmap import ids
from etdtransform.bounds import (
    cadence_minutes_by_household_batch,
    delivered_by_household_batch,
    grid_minutes_by_household_batch,
)


def spread_steps_by_household(cum_col, diff_col, cadence) -> dict:
    """
    How many grid slots one reading covers, for each household dataset whose
    *diff_col* needs spreading.

    A dataset qualifies when the variable was delivered as **intervals only**
    -- the interval column carried readings and the register did not -- and the
    readings cover a whole number of grid slots, more than one. The register
    path is the cumulative logic's business and is never spread here.

    A dataset is left out, rather than raised on, when its cadence does not
    divide the grid. A declared cadence like that has already been refused at
    mapping, so what reaches this point is a measured one -- a column too
    irregular to have a reading lattice -- and failing a run on detection noise
    would be wrong.

    Parameters
    ----------
    cum_col : str
        The register column of the variable.
    diff_col : str
        Its interval column, the one that would be spread.
    cadence : pd.DataFrame or None
        The cadence table from :func:`load_column_cadence`.

    Returns
    -------
    dict
        ``ids.HOUSEHOLD_BATCH`` -> slots per reading, for the qualifying
        datasets only.
    """
    interval_delivered = delivered_by_household_batch(diff_col, cadence)
    if not any(interval_delivered.values()):
        return {}

    register_delivered = delivered_by_household_batch(cum_col, cadence)
    minutes = cadence_minutes_by_household_batch(diff_col, cadence)
    grids = grid_minutes_by_household_batch(cadence)

    steps = {}
    for household, delivered in interval_delivered.items():
        if not delivered or register_delivered.get(household, False):
            continue
        reading_minutes = minutes.get(household)
        grid = grids.get(household)
        if reading_minutes is None or grid is None:
            continue
        step = float(reading_minutes) / grid
        if step < 2 or step != int(step):
            continue
        steps[household] = int(step)

    return steps


def cadence_with_column_at_grid(
    cadence, column, household_batches
) -> pd.DataFrame:
    """
    A copy of *cadence* recording *column* at the grid, for *household_batches*.

    Once a column has been spread it reports every grid row, so that is the
    interval anything judging its values has to use -- the column bound, the
    fill bound, the threshold pass and the rule bounds all scale from this
    table. Only the rows named are touched, so every other cadence in it is the
    one it already held.

    Returns a copy: the table belongs to the caller of the fill, and what
    happened to a column inside one pass is not a fact about the mapped data.

    Parameters
    ----------
    cadence : pd.DataFrame
        The cadence table from :func:`load_column_cadence`.
    column : str
        The column that was spread.
    household_batches : iterable of int
        The household datasets whose rows for *column* to rewrite.

    Returns
    -------
    pd.DataFrame
        The amended copy.
    """
    household_batches = [int(b) for b in household_batches]
    if cadence is None or cadence.empty or not household_batches:
        return cadence

    grids = grid_minutes_by_household_batch(cadence)
    amended = cadence.copy()
    rows = (
        (amended["column"] == column)
        & amended[ids.HOUSEHOLD_BATCH].isin(household_batches)
    )
    amended.loc[rows, "cadence_minutes"] = (
        amended.loc[rows, ids.HOUSEHOLD_BATCH].map(grids).astype("Int64")
    )
    return amended


def spread_delivered_interval_to_grid(
    df, col, k_by_household_batch, household_batch_col=ids.HOUSEHOLD_BATCH
):
    """
    Spread each delivered interval reading over the *k* grid slots it covers.

    An upspread: a reading is the consumption of the *k* slots ending at its own
    timestamp, so each of those slots takes ``value / k``. A slot whose own
    reading was never delivered stays ``pd.NA`` -- the fill is limited to
    ``k - 1`` rows, so it cannot reach back across a tick that carried no value,
    and an hour the meter never reported stays unknown rather than borrowing
    from the hour after it.

    The averaging reaches the same values by a different route -- it expands
    each window against a table of offsets in SQL -- and the two must agree slot
    for slot, because the interval branch of the fill multiplies a household's
    own values by the average. ``test_spread_delivered_interval`` asserts that.

    Parameters
    ----------
    df : pd.DataFrame
        Rows sorted by household dataset then ReadingDate, padded to the grid,
        so a row's position is its slot.
    col : str
        The delivered interval column to spread.
    k_by_household_batch : dict
        ``ids.HOUSEHOLD_BATCH`` -> how many grid slots one of that dataset's
        readings covers. Datasets absent from it are left untouched.
    household_batch_col : str
        Name of the dataset identifier column.

    Returns
    -------
    pd.Series
        Float64 on ``df.index``, holding the spread value at every slot a
        reading covers and ``pd.NA`` everywhere else, including every row of a
        dataset not in *k_by_household_batch*.
    """
    out = pd.Series(pd.NA, index=df.index, dtype="Float64")
    if not k_by_household_batch:
        return out

    id_column = df[household_batch_col]
    # One native groupby per distinct k. k is a per-dataset fact and bfill
    # takes no per-row limit, so the loop is over the distinct values rather
    # than over the datasets.
    for k in sorted(set(k_by_household_batch.values())):
        batch_ids = [h for h, k_h in k_by_household_batch.items() if k_h == k]
        rows = id_column.isin(batch_ids)
        if not rows.any():
            continue
        block = df.loc[rows, [household_batch_col, col]]
        out.loc[rows] = (
            block.groupby(household_batch_col, sort=False)[col].bfill(limit=k - 1) / k
        )

    return out
