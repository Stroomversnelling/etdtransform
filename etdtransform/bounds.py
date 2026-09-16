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
from etdmap.data_model import cumulative_column, thresholds_dict
from etdmap.flags import scale_bound_to_cadence
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


def _cadence_names(diff_col):
    """The names to look *diff_col* up under in the cadence table.

    Cadence is measured on the columns as delivered. An interval column derived
    during the pipeline is not there under its own name, so its cumulative
    variable is used instead. A delivered interval column is found under its own
    name and has no cumulative counterpart to fall back to.
    """
    cumulative = cumulative_column(diff_col)
    return [diff_col] if cumulative is None else [diff_col, cumulative]


def cadence_minutes_by_household(diff_col, cadence) -> dict:
    """
    The interval each household's reading of *diff_col* covers, in minutes.

    Households absent from the table, or recorded with no cadence, are omitted;
    callers decide what an unknown cadence means for them.
    """
    if cadence is None or cadence.empty:
        return {}

    rows = cadence[cadence["column"].isin(_cadence_names(diff_col))]
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
    bound = declared.get(bound_key, pd.NA)
    if pd.isna(bound):
        return {}

    return {
        household: scale_bound_to_cadence(
            bound, declared.get("ThresholdType"), cadence_minutes
        )
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


def shape_and_level_average(
    df,
    diff_col,
    project_id_column,
    cadence,
    grid_minutes=5,
):
    """
    Compute a project average across households reporting at different cadences.

    The average is derived in two parts:

    Shape
        The distribution of consumption across the day, taken from households
        reporting at the grid cadence. A household reporting hourly records
        twelve grid intervals in one reading and cannot distinguish between
        them, so it carries no information about shape.

    Level
        Mean consumption per grid interval, taken from all households. Each
        reading is divided by the number of grid intervals it covers. For a
        cumulative variable this conversion is exact, since the reading is the
        total over its interval.

    The shape is then rescaled by the ratio of the level across all households
    to the level across the grid-cadence households alone. Where every household
    reports at the grid cadence the two levels are equal and the result is the
    arithmetic mean.

    A project with no grid-cadence household has no shape to scale. Its rows
    come from the mean of the per-grid-interval rates instead, per project, so
    that it still receives an average while the other projects keep their shape.

    Parameters
    ----------
    df : pd.DataFrame
        Household readings. Must contain ``HuisIdBSV``, ``ReadingDate``,
        *project_id_column* and *diff_col*.
    diff_col : str
        Interval column to average.
    project_id_column : str
        Column identifying the project.
    cadence : pd.DataFrame
        Cadence table from :func:`load_column_cadence`.
    grid_minutes : int
        Grid interval in minutes.

    Returns
    -------
    tuple of (pd.DataFrame, dict)
        The average, with columns *project_id_column*, ``ReadingDate`` and
        ``<diff_col>_avg``; and a dict reporting how many households
        contributed to each part. ``fallback`` is True when at least one
        project had no grid-cadence household.
    """
    minutes = _cadence_minutes_per_row(df, diff_col, cadence, grid_minutes)

    at_grid = minutes == grid_minutes
    intervals = minutes / grid_minutes

    # Level: every household, converted to a per-grid-interval rate.
    rate = df[diff_col] / intervals
    level_all = rate.groupby(df[project_id_column]).mean()

    # Shape: grid-cadence households only.
    shape = (
        df[at_grid]
        .groupby([project_id_column, "ReadingDate"])[diff_col]
        .mean()
        .rename("shape")
        .reset_index()
    )
    level_grid = (
        rate[at_grid].groupby(df.loc[at_grid, project_id_column]).mean()
    )

    # A project with no grid-cadence household has no shape to scale, and must
    # not simply drop out of the result. Its average is the mean of the
    # per-grid-interval rates instead.
    without_shape = set(df[project_id_column].dropna().unique()) - set(
        shape[project_id_column]
    )
    fallback = None
    if without_shape:
        rows = df[df[project_id_column].isin(without_shape)]
        fallback = (
            rows.assign(_rate=rate[rows.index])
            .groupby([project_id_column, "ReadingDate"])["_rate"]
            .mean()
            .rename(f"{diff_col}_avg")
            .reset_index()
        )

    avg = scale_shape_to_level(
        shape, level_all, level_grid, project_id_column, diff_col, fallback=fallback
    )
    return avg, {
        "shape_households": int(df.loc[at_grid, "HuisIdBSV"].nunique()),
        "level_households": int(df["HuisIdBSV"].nunique()),
        "fallback": bool(without_shape),
    }


def scale_shape_to_level(
    shape, level_all, level_grid, project_id_column, diff_col, fallback=None
):
    """
    Rescale a grid-cadence shape to the level measured across all households.

    Both levels are mean consumption per grid interval, computed the same way,
    so they are equal when every household reports at the grid cadence and the
    scale is then exactly 1.

    Comparing the shape's own mean against the level instead would also rescale
    for panel imbalance: a household with a gap contributes to some slots and
    not others, which shifts the mean of the per-slot means independently of
    cadence.

    Parameters
    ----------
    shape : pd.DataFrame
        One row per (project, ReadingDate), with the mean over grid-cadence
        households in a column named ``shape``.
    level_all : pd.Series
        Mean per-grid-interval consumption per project, over all households.
    level_grid : pd.Series
        The same over grid-cadence households only. Indexed by project, as
        *level_all* is.
    project_id_column : str
        Column identifying the project.
    diff_col : str
        Interval column being averaged.
    fallback : pd.DataFrame, optional
        Averages for the projects that have no shape, already in the result's
        columns. Appended as they are.

    Returns
    -------
    pd.DataFrame
        Columns *project_id_column*, ``ReadingDate`` and ``<diff_col>_avg``.
    """
    scale_by_project = (level_all / level_grid).where(level_grid > 0, 1.0)
    scale = shape[project_id_column].map(scale_by_project).fillna(1.0)

    out = shape.copy()
    out[f"{diff_col}_avg"] = out["shape"] * scale
    out = out[[project_id_column, "ReadingDate", f"{diff_col}_avg"]]

    if fallback is None or fallback.empty:
        return out
    if out.empty:
        return fallback.reset_index(drop=True)
    return pd.concat([out, fallback], ignore_index=True).sort_values(
        [project_id_column, "ReadingDate"]
    ).reset_index(drop=True)


def _cadence_minutes_per_row(df, diff_col, cadence, grid_minutes):
    """Return the interval each row's reading covers, in minutes, per household.

    Households absent from the cadence table are assumed to report at the grid
    cadence.
    """
    if cadence is None or cadence.empty or "HuisIdBSV" not in df.columns:
        return pd.Series(float(grid_minutes), index=df.index)

    per_household = cadence_minutes_by_household(diff_col, cadence)
    return (
        df["HuisIdBSV"].map(per_household).astype("Float64").fillna(float(grid_minutes))
    )
