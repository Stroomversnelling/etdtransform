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
    tariff_root_to_splits,
    thresholds_dict,
)
from etdmap.flags import MappingFlag, _as_limit, flag_column
from etdmap.index_helpers import frequency_to_minutes


def load_column_cadence(mapped_folder_path=None) -> pd.DataFrame:
    """
    Read the per-column cadence sidecar written during mapping.

    Cadence is determined at the mapping stage, from the timestamps as
    delivered and before any fill, and written to a parquet beside the mapped
    data. It is recorded per (household, column) and is not carried in the
    aggregated table.

    An absent sidecar is an error rather than a default. Without it every
    threshold is judged at the grid cadence, which silently tightens the bound
    for every household reporting more coarsely and nulls their readings as
    out of bounds. Re-map the dataset to produce it.

    Returns
    -------
    pd.DataFrame
        The cadence table.

    Raises
    ------
    FileNotFoundError
        When no mapped folder is configured, or the sidecar is not in it.
    """
    import etdtransform
    from etdmap.mapping_helpers import CADENCE_FILENAME

    folder = mapped_folder_path or getattr(
        etdtransform.options, "mapped_folder_path", None
    )
    if not folder:
        raise FileNotFoundError(
            f"No mapped folder is configured, so {CADENCE_FILENAME} cannot be "
            f"read. Set etdtransform.options.mapped_folder_path or pass "
            f"mapped_folder_path."
        )

    path = os.path.join(folder, CADENCE_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No {CADENCE_FILENAME} in {folder}. Every column's threshold is "
            f"scaled to the cadence it reports at, so without it a coarsely "
            f"reporting household is judged against the grid bound and its "
            f"readings are nulled. Re-map the dataset."
        )

    return pd.read_parquet(path, dtype_backend="numpy_nullable")


def dataset_grid_minutes(mapped_folder_path=None):
    """
    The grid the mapped data sits on, in minutes.

    Each household batch declares its own grid as ``Gegevensfrequentie``, and
    the index carries it. An aggregated dataset spans batches, so its grid is
    the finest of theirs: that is the spacing its rows are on, and the spacing a
    shape can be resolved at. Batches on a coarser grid contribute level but not
    shape, which is what the averaging already does with a column's cadence.

    An undeclared grid is an error rather than a default. The grid sets the
    spacing every threshold is judged at, so assuming one silently rescales
    every bound in the dataset.

    Raises
    ------
    FileNotFoundError
        When no mapped folder is configured, or the index is not in it.
    ValueError
        When the index declares no usable frequency.
    """
    import etdtransform

    folder = mapped_folder_path or getattr(
        etdtransform.options, "mapped_folder_path", None
    )
    if not folder:
        raise FileNotFoundError(
            "No mapped folder is configured, so the dataset grid cannot be "
            "read. Set etdtransform.options.mapped_folder_path or pass "
            "mapped_folder_path."
        )

    path = os.path.join(str(folder), "index.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No index.parquet in {folder}, so the dataset grid is unknown. "
            f"Every threshold is judged at the grid spacing. Re-map the dataset."
        )

    index = pd.read_parquet(path, dtype_backend="numpy_nullable")
    if "Gegevensfrequentie" not in index.columns:
        raise ValueError(
            f"index.parquet in {folder} carries no Gegevensfrequentie, so the "
            f"dataset grid is unknown. Check the data model. Re-map the dataset."
        )

    declared = index["Gegevensfrequentie"].dropna().unique()
    minutes = sorted(
        m for m in (frequency_to_minutes(value) for value in declared)
        if pd.notna(m)
    )
    if not minutes:
        raise ValueError(
            f"index.parquet in {folder} declares no readable Gegevensfrequentie "
            f"({list(declared)}), so the dataset grid is unknown. "
            f"Check the data model."
        )
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

    *diff_col*'s own row wins. It is looked up under the variable it is paired
    with as well, because cadence is measured on the columns as delivered and a
    column derived later is not in the table under its own name -- but that is a
    fallback for a column the table has nothing for, and must not override one
    it does. A variable delivered as intervals only has a row for its absent
    register too, carrying the grid, and letting that through would report the
    delivered interval as reporting every slot.

    Parameters
    ----------
    diff_col : str
        The column being asked about.
    cadence : pd.DataFrame or None
        The cadence table from :func:`load_column_cadence`: one row per
        (HuisIdBSV, column), with ``cadence_minutes`` holding the interval that
        column's readings cover. None or empty resolves nothing.

    Returns
    -------
    dict
        HuisIdBSV -> minutes.
    """
    if cadence is None or cadence.empty:
        return {}

    def _minutes(names):
        rows = cadence[cadence["column"].isin(names)]
        return {
            int(r.HuisIdBSV): float(r.cadence_minutes)
            for r in rows.itertuples()
            if pd.notna(r.cadence_minutes)
        }

    paired = [n for n in cadence_lookup_names(diff_col) if n != diff_col]
    return {**_minutes(paired), **_minutes([diff_col])}


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

    Parameters
    ----------
    diff_col : str
        The column being bounded.
    cadence : pd.DataFrame or None
        The cadence table from :func:`load_column_cadence`. Each household's
        bound is the declared threshold scaled to the interval its readings of
        *diff_col* cover.
    bound_key : str
        ``"Max"`` or ``"Min"``.

    Returns
    -------
    dict
        HuisIdBSV -> bound.
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

    A household with no recorded cadence for this column takes the declared
    bound, which is the bound for one interval of the threshold's own unit. The
    cadence table records a column only if it carried readings when cadence was
    measured, and some columns acquire values afterwards -- a zero written for an
    absent device, a tariff column synthesised from a total -- so an absent row
    is normal rather than an error. The missing rows are logged, because for a
    column that does report coarsely it would mean judging it too tightly.

    Returns
    -------
    pd.Series or scalar
        One threshold per row of *df*. A scalar when the threshold is not
        declared (unbounded) so the result can be passed to ``bound_flags``
        either way.

    Raises
    ------
    ValueError
        When there is no cadence table at all, since then no column could be
        scaled and every household would silently take the grid bound.
    """
    declared = thresholds_dict.get(diff_col, {})
    bound = declared.get(bound_key, pd.NA)
    if pd.isna(bound):
        return bound

    if cadence is None or cadence.empty:
        raise ValueError(
            f"{diff_col}: no cadence table, so every household would be judged "
            f"at the grid interval. Re-map the dataset."
        )

    per_household = column_bound_by_household(diff_col, cadence, bound_key=bound_key)
    if "HuisIdBSV" not in df.columns:
        return bound

    unrecorded = sorted(
        set(df["HuisIdBSV"].dropna().unique()) - set(per_household)
    )
    if unrecorded:
        logging.info(
            f"{diff_col}: {len(unrecorded)} household(s) have no recorded "
            f"cadence and take the declared bound {bound} "
            f"({unrecorded[:10]})."
        )

    scaled = df["HuisIdBSV"].map(per_household).astype("Float64").fillna(bound)
    n_scaled = int((scaled != bound).sum())
    if n_scaled:
        logging.info(
            f"{diff_col}: {n_scaled} row(s) judged at their column's own "
            f"cadence; declared {bound}, scaled up to {float(scaled.max())}."
        )
    return scaled


def clip_to_bounds(values, lo, hi):
    """
    Push *values* into ``[lo, hi]`` and report what that required.

    Every route that writes an imputed value goes through here, so one
    definition of "in range" serves them all rather than each carrying its own.

    Either bound is a scalar for the whole column or one value per row: a
    cadence-scaled bound varies by household, and a rule-derived bound varies by
    row, being a formula in that row's other cells. An absent bound is no bound,
    per row as well as per column.

    Parameters
    ----------
    values : pd.Series
        Nullable ``Float64``. A missing value stays missing and is not marked.
    lo, hi : float, pd.Series, pd.NA or None
        The range. ``pd.NA`` on a side leaves that side unbounded.

    Returns
    -------
    tuple of (pd.Series, pd.Series, pd.Series)
        The clipped values, then two boolean masks: *moved* where a value was
        brought to a bound, and *empty* where the range allows nothing because
        *lo* came out above *hi*. The masks carry no NA, so a caller can map
        them straight onto its own flags. They never overlap: an empty range
        writes nothing, which is a different finding from moving a value.

    Notes
    -----
    Flags are deliberately not set here. Mapping records a ``MappingFlag`` and
    imputation an ``ImputeType`` for the same events, so the categories stay
    with the caller and this function stays testable without either.
    """
    inf = float("inf")
    lower = _as_limit(lo, -inf, values.index)
    upper = _as_limit(hi, inf, values.index)

    # Series.clip would propagate a NA threshold into the result; an absent
    # bound means no bound.
    empty = _mask(lower > upper, values.index)
    below = _mask(values < lower, values.index) & ~empty
    above = _mask(values > upper, values.index) & ~empty

    out = values.mask(below, lower).mask(above, upper).mask(empty, pd.NA)
    return out.astype("Float64"), below | above, empty


def _mask(condition, index):
    """A comparison as a plain boolean mask. NA is not a finding, so it is False."""
    if isinstance(condition, pd.Series):
        return condition.reindex(index).fillna(False).astype(bool)
    return pd.Series(bool(condition) if condition is not pd.NA else False, index=index)


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


def delivered_by_household(column, cadence) -> dict:
    """
    Whether *column* carried delivered readings, per household.

    Read from ``n_readings``, counted on the data as delivered before any fill.
    A household with no row for *column*, or a row recording no readings, did
    not have that form delivered.

    The cumulative/interval pairing is **not** followed. A variable delivered in
    both forms has a row for each, and one delivered as intervals only still has
    a row for its absent register, so resolving a column through its partner --
    as :func:`cadence_minutes_by_household` does, for the stages asking about a
    column derived later -- would report a delivery that never happened.

    A tariff root resolves through its splits as well. Where a supplier sends
    peak and off-peak registers, ``combine_tariff_registers`` adds the root they
    sum to, and the root is the column the averaging and the fill work on
    (:func:`etdtransform.vectorized_impute.without_tariff_splits`) -- but
    cadence is measured before the combination, so the root has no row of its
    own and the quantity is recorded only under the names it arrived as.
    Delivery is therefore an OR over the names it may have arrived under, not a
    precedence between them: cadence needs one value and so has a winner, while
    a reading under any of these names is a reading of the same quantity. Pair
    membership comes from the data model's own structures, never from the name.

    Metadata, not data: a delivered register whose every reading is later
    condemned still *was* delivered, and reading the values would say otherwise.

    Parameters
    ----------
    column : str
        The column name as delivered, e.g. a register or an interval column.
    cadence : pd.DataFrame or None
        The cadence table from :func:`load_column_cadence`.

    Returns
    -------
    dict
        HuisIdBSV -> bool, for every household the table has a row for under any
        of the names read.
    """
    if cadence is None or cadence.empty:
        return {}

    # A table without the field cannot say what was delivered, which is the same
    # state as a row that records no readings: not delivered. Reporting a
    # delivery on no evidence would route a column down a path its data does not
    # support, so the unknown resolves the safe way.
    if "n_readings" not in cadence.columns:
        return {}

    names = [column, *tariff_root_to_splits.get(column, ())]
    rows = cadence[cadence["column"].isin(names)]
    delivered = {}
    for r in rows.itertuples():
        household = int(r.HuisIdBSV)
        delivered[household] = delivered.get(household, False) or bool(
            pd.notna(r.n_readings) and r.n_readings > 0
        )
    return delivered


def grid_minutes_by_household(cadence, date_column="ReadingDate") -> dict:
    """
    The interval each household's rows sit on, in minutes.

    ``detect_column_cadence`` measures the timestamp column alongside the data
    columns, so its row carries the grid the household's batch declares -- the
    spacing ``ensure_intervals`` padded that household to. Read from the table
    rather than from a dataset-wide figure, because an aggregated dataset spans
    batches and a household on a coarser grid would be judged against the finest
    of them.

    Parameters
    ----------
    cadence : pd.DataFrame or None
        The cadence table from :func:`load_column_cadence`.
    date_column : str
        Name of the timestamp column, whose row carries the grid.

    Returns
    -------
    dict
        HuisIdBSV -> minutes, omitting any household with no usable row.
    """
    if cadence is None or cadence.empty:
        return {}

    rows = cadence[cadence["column"] == date_column]
    return {
        int(r.HuisIdBSV): float(r.cadence_minutes)
        for r in rows.itertuples()
        if pd.notna(r.cadence_minutes) and float(r.cadence_minutes) > 0
    }


