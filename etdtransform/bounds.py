"""Threshold selection for interval columns.

A variable's threshold is declared against a unit of time. A ``5-minute``
threshold states what one five-minute interval may hold. Two intervals are
relevant when a column does not report at the grid cadence:

Column threshold
    The declared threshold scaled to how often the column reports, per
    household. A cumulative variable reporting hourly delivers an hour of consumption in a
    single reading, so its plausible maximum is twelve times the declared
    five-minute value.

Grid threshold
    The declared threshold itself. It applies to a value occupying one slot of
    the grid and carrying one interval of consumption.

Which applies depends on what the value represents:

===========================  ==============  ==================================
Task                         Threshold       Rationale
===========================  ==============  ==================================
frozen-span jump detection   column          a catch-up is judged against what
                                             the cumulative variable delivers in one of
                                             its own reporting intervals
physical outlier detection   column          a delivered reading is implausible
                                             only relative to its own interval
filling                      grid            a filled value occupies one grid
                                             slot
project averages             column          a delivered reading contributes at
                                             its own cadence
===========================  ==============  ==================================

Applying the grid threshold to a delivered hourly reading rejects a plausible
value. Applying the column threshold to a filled value permits twelve intervals
of consumption in a single grid slot.

Only ``5-minute`` thresholds scale. A ``momentaan`` threshold applies to an
instantaneous reading, which does not accumulate over a longer interval, and a
``cumulatief`` threshold applies to a yearly total.
"""

import logging
import os

import pandas as pd
from etdmap.data_model import (
    cadence_lookup_names,
    cumulative_column,
    scale_bounds,
    thresholds_dict,
)
from etdmap.flags import MappingFlag, flag_column
from etdmap.index_helpers import DEFAULT_GRID_MINUTES, frequency_to_minutes


def load_column_cadence(mapped_folder_path=None) -> pd.DataFrame:
    """
    Read the per-column cadence sidecar written during mapping.

    Cadence is determined at the mapping stage, from the timestamps as
    delivered and before any fill, and written to a parquet beside the mapped
    data. It is recorded per (household, column) and is not carried in the
    aggregated table.

    Returns
    -------
    pd.DataFrame
        The cadence table, or an empty DataFrame when the sidecar is absent.
        An empty table leaves every threshold unscaled.
    """
    import etdtransform
    from etdmap.mapping_helpers import CADENCE_FILENAME

    folder = mapped_folder_path or getattr(
        etdtransform.options, "mapped_folder_path", None
    )
    if not folder:
        return pd.DataFrame()

    path = os.path.join(folder, CADENCE_FILENAME)
    if not os.path.exists(path):
        logging.info(
            f"No {CADENCE_FILENAME} in {folder}; bounds will not be scaled to "
            f"cadence."
        )
        return pd.DataFrame()

    return pd.read_parquet(path, dtype_backend="numpy_nullable")


def dataset_grid_minutes(mapped_folder_path=None, default=DEFAULT_GRID_MINUTES):
    """
    The grid the mapped data sits on, in minutes.

    Each household batch declares its own grid as ``Gegevensfrequentie``, and
    the index carries it. An aggregated dataset spans batches, so its grid is
    the finest of theirs: that is the spacing its rows are on, and the spacing a
    shape can be resolved at. Batches on a coarser grid contribute level but not
    shape, which is what the averaging already does with a column's cadence.

    Falls back to *default* when the index carries no declared frequency, which
    is the state of any dataset mapped before the field existed.
    """
    import etdtransform

    folder = mapped_folder_path or getattr(
        etdtransform.options, "mapped_folder_path", None
    )
    if not folder:
        return float(default)

    path = os.path.join(str(folder), "index.parquet")
    if not os.path.exists(path):
        return float(default)

    index = pd.read_parquet(path, dtype_backend="numpy_nullable")
    if "Gegevensfrequentie" not in index.columns:
        return float(default)

    declared = index["Gegevensfrequentie"].dropna().unique()
    minutes = sorted(
        m for m in (frequency_to_minutes(value) for value in declared)
        if pd.notna(m)
    )
    if not minutes:
        return float(default)
    if len(minutes) > 1:
        logging.info(
            f"Household batches declare {len(minutes)} different grids "
            f"({', '.join(str(m) for m in minutes)} minutes). Using the finest "
            f"as the dataset grid; coarser batches contribute level but not "
            f"shape."
        )
    return float(minutes[0])


def cadence_minutes_by_household(diff_col, cadence) -> dict:
    """
    The interval each household's reading of *diff_col* covers, in minutes.

    Households absent from the table, or recorded with no cadence, are omitted;
    callers decide what an unknown cadence means for them.
    """
    if cadence is None or cadence.empty:
        return {}

    rows = cadence[cadence["column"].isin(cadence_lookup_names(diff_col))]
    return {
        int(r.HuisIdBSV): float(r.cadence_minutes)
        for r in rows.itertuples()
        if pd.notna(r.cadence_minutes)
    }


def grid_bound(diff_col, bound_key="Max"):
    """
    Return the declared threshold for *diff_col*.

    This is the grid threshold: what one slot of the grid may hold. It is a
    scalar, since the grid is the same for every household.
    """
    return thresholds_dict.get(diff_col, {}).get(bound_key, pd.NA)


def column_bound_by_household(diff_col, cadence, bound_key="Max") -> dict:
    """
    The threshold for *diff_col* scaled to each household's cadence.

    Households absent from the cadence table are omitted; they take the declared
    threshold, which :func:`grid_bound` returns.
    """
    declared = thresholds_dict.get(diff_col, {})
    index = 0 if bound_key == "Min" else 1
    if pd.isna(scale_bounds(declared)[index]):
        return {}

    return {
        household: scale_bounds(declared, cadence_minutes)[index]
        for household, cadence_minutes in cadence_minutes_by_household(
            diff_col, cadence
        ).items()
    }


def column_bound(df, diff_col, cadence, bound_key="Max"):
    """
    Return the threshold for *diff_col* scaled to each household's cadence.

    An interval column inherits the cadence of the cumulative variable it is derived from,
    since it is computed during the pipeline and does not appear in the cadence
    table under its own name.

    Parameters
    ----------
    df : pd.DataFrame
        Rows to return a threshold for. Must contain ``HuisIdBSV``.
    diff_col : str
        Interval column name.
    cadence : pd.DataFrame
        Cadence table from :func:`load_column_cadence`.
    bound_key : str
        Which threshold to scale, ``"Max"`` or ``"Min"``.

    Returns
    -------
    pd.Series or scalar
        One threshold per row of *df*. Returns the declared scalar threshold
        when no cadence is available, so the result can be passed to
        ``bound_flags`` either way. Households absent from the cadence table
        take the declared threshold.
    """
    declared = thresholds_dict.get(diff_col, {})
    bound = declared.get(bound_key, pd.NA)
    if pd.isna(bound) or cadence is None or cadence.empty:
        return bound

    per_household = column_bound_by_household(diff_col, cadence, bound_key=bound_key)
    if not per_household or "HuisIdBSV" not in df.columns:
        return bound

    scaled = df["HuisIdBSV"].map(per_household).astype("Float64").fillna(bound)
    n_scaled = int((scaled != bound).sum())
    if n_scaled:
        logging.info(
            f"{diff_col}: {n_scaled} row(s) judged at their column's own "
            f"cadence; declared {bound}, scaled up to {float(scaled.max())}."
        )
    return scaled


def fill_bound(df, diff_col, cadence, is_imputed_col=None, bound_key="Max"):
    """
    Return the threshold appropriate to each row's provenance.

    Filled rows take the grid threshold, since a filled value occupies one grid
    slot. Delivered rows take the column threshold, since a delivered reading
    covers the column's own reporting interval. Where a column reports at the
    grid cadence the two thresholds are equal.

    Parameters
    ----------
    df : pd.DataFrame
        Rows to return a threshold for.
    diff_col : str
        Interval column name.
    cadence : pd.DataFrame
        Cadence table from :func:`load_column_cadence`.
    is_imputed_col : str, optional
        Boolean column marking filled rows. Without it every row is treated as
        delivered.
    bound_key : str
        Which threshold to use, ``"Max"`` or ``"Min"``.

    Returns
    -------
    pd.Series or scalar
        One threshold per row of *df*.
    """
    scaled = column_bound(df, diff_col, cadence, bound_key=bound_key)
    declared = grid_bound(diff_col, bound_key=bound_key)

    if not isinstance(scaled, pd.Series):
        return scaled
    if is_imputed_col and is_imputed_col in df.columns:
        return scaled.mask(df[is_imputed_col].fillna(False).astype(bool), declared)
    return scaled

def usable_readings(df, column):
    """
    Rows where *column* holds a measurement.

    A value is a measurement when it is present and carries no finding that
    disqualifies it: written rather than read, read for a different interval,
    condemned by a cumulative fault, or outside what the variable may hold.
    ``MappingFlag.NOT_A_MEASUREMENT`` is that one test.

    A column with no findings column has nothing recorded against it, so every
    present value counts.
    """
    if column is None or column not in df.columns:
        return pd.Series(False, index=df.index)

    usable = df[column].notna()
    findings = flag_column(column)
    if findings in df.columns:
        clean = (
            df[findings].fillna(0) & MappingFlag.NOT_A_MEASUREMENT.value
        ) == 0
        usable = usable & clean
    return usable


def resample_to_grid(df, diff_col, cadence, grid_minutes=DEFAULT_GRID_MINUTES):
    """
    Spread each household's readings of *diff_col* evenly over the grid.

    Two consecutive usable readings bound a window. Its volume is the
    consumption delivered over it, and its length in grid slots is the number of
    rows between them -- ``ensure_intervals`` gives one row per slot, so the row
    count is the slot count. The volume is divided by that length and written to
    each of the window's slots, ending at the later reading: an interval value at
    ``T`` covers ``(T_prev, T]``, so a fifteen-minute reading at 00:15 fills
    00:05, 00:10 and 00:15.

    A household reporting at the grid cadence has one-slot windows, so its
    output is its interval column unchanged.

    The cumulative column supplies the volume wherever it holds usable readings,
    since it is the authoritative form; the interval column supplies it only
    where the cumulative does not. A window longer than one reporting interval
    is a meter outage rather than a reading interval and contributes nothing --
    distributing measured consumption across a gap is imputation's work, not the
    average's.

    Parameters
    ----------
    df : pd.DataFrame
        Household readings, sorted by household then ReadingDate. Must contain
        ``HuisIdBSV`` and *diff_col*.
    diff_col : str
        Interval column to resample.
    cadence : pd.DataFrame
        Cadence table from :func:`load_column_cadence`.
    grid_minutes : float
        Grid interval in minutes.

    Returns
    -------
    pd.Series
        Float64, aligned to *df*, holding the resampled value at every slot a
        window covers and pd.NA elsewhere.
    """
    households = df["HuisIdBSV"]
    position = df.groupby(households).cumcount()

    # The cumulative variable is authoritative, so it is the source wherever it
    # has anything to say. Deciding per household rather than per row keeps a
    # single column behind every window.
    cumulative = cumulative_column(diff_col)
    from_cumulative = usable_readings(df, cumulative)
    cumulative_households = set(households[from_cumulative].unique())
    on_cumulative = households.isin(cumulative_households)

    usable = from_cumulative.where(on_cumulative, usable_readings(df, diff_col))
    if not usable.any():
        return pd.Series(pd.NA, index=df.index, dtype="Float64")

    if cumulative and cumulative in df.columns:
        source = df[cumulative].where(on_cumulative, df[diff_col])
    else:
        source = df[diff_col]

    readings = pd.DataFrame({
        "household": households[usable],
        "position": position[usable],
        "value": source[usable].astype("Float64"),
        "cumulative": on_cumulative[usable],
    })
    by_household = readings.groupby("household", sort=False)

    minutes = cadence_minutes_by_household(diff_col, cadence)
    slots = (
        readings["household"].map(minutes).astype("Float64") / float(grid_minutes)
    ).fillna(1.0)

    # A cumulative reading states a running total, so a window's volume is the
    # rise since the previous reading and its length is the distance back to it.
    # A delivered interval value already is the volume for its own window, and
    # that window is its cadence however long ago the last reading arrived -- so
    # it needs no predecessor, and the first reading of such a column counts.
    rise = by_household["value"].diff()
    gap = by_household["position"].diff().astype("Float64")

    on_cum = readings["cumulative"]
    volume = rise.where(on_cum, readings["value"])
    length = gap.where(on_cum, slots)

    # A cumulative window longer than one reporting interval spans a meter
    # outage. An interval-sourced window is its cadence by construction and
    # cannot span one.
    usable_window = (length.notna() & (length > 0)).fillna(False)
    usable_window &= (~on_cum | (gap <= slots)).fillna(False)

    per_slot = (volume / length).where(usable_window)

    # Each window is written onto the slots it covers, counting back from the
    # reading that ends it. Expanding rather than carrying backwards keeps
    # windows that do not touch -- an interval-sourced column's windows are its
    # cadence and may leave slots uncovered between them.
    placed = pd.Series(pd.NA, index=df.index, dtype="Float64")
    keep = per_slot.notna()
    if not keep.any():
        return placed

    spans = length[keep].astype("Int64").astype(int)
    row = pd.Series(range(len(df)), index=df.index)
    ends = row[readings.index][keep]

    covered = ends.repeat(spans).to_numpy() - _offsets_within(spans)
    values = per_slot[keep].repeat(spans).to_numpy()

    inside = covered >= 0
    placed.iloc[covered[inside]] = values[inside]
    return placed


def _offsets_within(spans):
    """0, 1, ... n-1 for each span in *spans*, concatenated.

    Index arithmetic rather than a data operation, so numpy is the right tool:
    the values it produces are row offsets, never measurements.
    """
    import numpy as np

    lengths = spans.to_numpy()
    total = int(lengths.sum())
    starts = np.repeat(np.cumsum(lengths) - lengths, lengths)
    return np.arange(total) - starts

