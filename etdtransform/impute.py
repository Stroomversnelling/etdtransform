import logging
import os
from math import floor, isclose, log10

import numpy as np
import pandas as pd
from etdmap.data_model import thresholds_dict

from etdmap.flags import MappingFlag, flag_column

from etdtransform.bounds import (
    cadence_minutes_by_household,
    column_bound,
    column_bound_by_household,
    dataset_grid_minutes,
    load_column_cadence,
    scale_shape_to_level,
    shape_and_level_average,
)

import etdtransform
from etdtransform.vectorized_impute import impute_and_normalize

#: Findings that disqualify a value from shaping a project average: it was
#: written by the pipeline rather than measured, or it was measured over a
#: window rather than at the interval it sits on.
_NOT_MEASURED_HERE = MappingFlag.INVENTED.value | MappingFlag.CATCHUP.value


def calculate_average_diff(
    df: pd.DataFrame,
    project_id_column: str,
    diff_columns: list[str],
    bounds: dict | None = None,
    cadence: pd.DataFrame | None = None,
    grid_minutes: float | None = None,
) -> pd.DataFrame:
    """
    Calculate average differences for specified columns grouped by project and reading date.

    This function computes the average differences for the specified columns,
    excluding outliers based on a 95th percentile threshold. It's used to prepare
    data for imputation of missing values.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data.
    project_id_column : str
        The name of the column containing project IDs.
    diff_columns : list[str]
        A list of column names for which to calculate average differences.
    bounds : dict, optional
        Column -> upper threshold. A value above its threshold is discarded
        before averaging. Defaults to the Max declared for the variable in the
        thresholds table; pass ``{}`` to disable, or a mapping to override.
    cadence : pd.DataFrame, optional
        The per-column cadence table. Households reporting coarser than the grid
        contribute to the level but not to the shape. Defaults to the sidecar
        written at mapping; an empty frame makes this the arithmetic mean.
    grid_minutes : float, optional
        The grid the data sits on. Resolved from the household batches'
        declared Gegevensfrequentie when not given.

    Returns
    -------
    dict
        A dictionary where keys are column names and values are dictionaries containing:
        - 'avg_diff': DataFrame with average differences
        - 'upper_bounds': DataFrame with upper bounds for outlier exclusion
        - 'household_max_with_bounds': DataFrame with household maximum values and bounds
        - 'n_discarded': how many values were discarded for exceeding the
          column threshold
        - 'n_flagged': how many values were excluded as invented or mistimed
        - 'shape_level': how many households contributed to the shape and to the
          level, from :func:`etdtransform.bounds.shape_and_level_average`

    Notes
    -----
    This function uses a 95th percentile threshold to exclude outliers when calculating
    averages. The threshold is doubled to create an upper bound for inclusion in the
    average calculation.

    Warnings
    --------
    - Negative difference values will raise a ValueError.
    - Missing values in the resulting average columns will be logged as errors.

    """
    logging.info("Calculating Diff column averages.")

    if bounds is None:
        bounds = {
            col: thresholds_dict[col]["Max"]
            for col in diff_columns
            if col in thresholds_dict
        }
    if cadence is None:
        cadence = load_column_cadence()
    if grid_minutes is None:
        grid_minutes = dataset_grid_minutes()

    def safe_quantile(group, col_name):
        filtered_group = group[group[col_name] > 1e-8]
        if filtered_group.empty:
            return pd.Series({col_name: pd.NA}, dtype="Float64")
        else:
            return pd.Series({col_name: filtered_group[col_name].quantile(0.95)})

    # per Diff column max per house
    logging.info("Calculating max values per household.")
    household_max = (
        df.groupby([project_id_column, "HuisIdBSV"])[diff_columns].max().reset_index()
    )
    household_max.columns = [project_id_column, "HuisIdBSV"] + [
        f"{col}_household_max" for col in diff_columns
    ]

    for col in diff_columns:
        household_max[f"{col}_household_max"] = household_max[f"{col}_household_max"].astype(
            "Float64",
        )

    avg_diff_dict = {}

    for col in diff_columns:
        logging.info(f"Handling column: {col}")

        logging.info(f"Calculating the 95th percentile (upper bound) for {col}.")
        upper_bounds = (
            household_max.groupby(project_id_column)
            .apply(safe_quantile, f"{col}_household_max")
            .reset_index()
        )
        upper_bounds.columns = [project_id_column, f"{col}_upper_bound"]
        upper_bounds[f"{col}_upper_bound"] = upper_bounds[
            f"{col}_upper_bound"
        ].multiply(2)

        logging.info(f"Identifying households to include for {col}.")
        household_max_with_bounds = household_max[
            [project_id_column, "HuisIdBSV", f"{col}_household_max"]
        ].merge(upper_bounds, on=project_id_column, how="left")
        include_mask = (
            household_max_with_bounds[f"{col}_household_max"]
            < household_max_with_bounds[f"{col}_upper_bound"]
        )
        households_to_include = household_max_with_bounds.loc[include_mask, "HuisIdBSV"]

        logging.info(f"Filtering the dataframe for {col}.")
        wanted = ["HuisIdBSV", project_id_column, "ReadingDate", col]
        if flag_column(col) in df.columns:
            wanted.append(flag_column(col))
        df_filtered = df[wanted][df["HuisIdBSV"].isin(households_to_include)]

        logging.info(f"Checking for negative Diff values in {col}.")
        if (df_filtered[col] < 0).any():
            raise ValueError("Negative Diff values found")
        # df_filtered[df_filtered[col]<0][col] = pd.NA

        df_filtered = df_filtered.copy()

        # An average states what households consumed at a time of day, so only
        # values measured at that time may shape it. Two kinds are excluded:
        #
        #   invented  -- filled down, filled to zero, or redistributed across a
        #                frozen span. Nothing was measured at this slot.
        #   mistimed  -- a catch-up. It was measured, but over a window rather
        #                than at this slot.
        #
        # Neither is excluded by a threshold. Redistribution places its values
        # inside the bounds, and a catch-up is plausible at the column's own
        # cadence, so the flags are what identifies them.
        n_flagged = 0
        findings = flag_column(col)
        if findings in df_filtered.columns:
            not_measured_here = (
                df_filtered[findings].fillna(0) & _NOT_MEASURED_HERE
            ) != 0
            n_flagged = int(not_measured_here.sum())
            if n_flagged:
                df_filtered.loc[not_measured_here, col] = pd.NA
                logging.info(
                    f"Excluded {n_flagged} value(s) from the {col} average: "
                    f"invented or recorded for a different window."
                )

        # A value above the variable's threshold is implausible rather than
        # merely large, so it is discarded instead of clipped: clipping would
        # leave a fabricated number in the pool, while discarding leaves the
        # average to the households that reported plausible values.
        #
        # The threshold used is the column threshold, scaled to how often each
        # household reports, so a delivered hourly reading is judged against
        # what an hour may hold rather than what five minutes may hold.
        n_discarded = 0
        bound = (bounds or {}).get(col, pd.NA)
        if bound is None or (not isinstance(bound, pd.Series) and pd.isna(bound)):
            bound = pd.NA
        else:
            bound = column_bound(df_filtered, col, cadence)
        if not (isinstance(bound, pd.Series) or pd.isna(bound)):
            bound = pd.Series(bound, index=df_filtered.index)
        if isinstance(bound, pd.Series):
            out_of_bounds = (df_filtered[col] > bound).fillna(False)
            n_discarded = int(out_of_bounds.sum())
            if n_discarded:
                df_filtered.loc[out_of_bounds, col] = pd.NA
                logging.info(
                    f"Discarded {n_discarded} value(s) above the column bound "
                    f"for {col} before averaging."
                )

        logging.info(f"Calculating the average differences for {col}.")
        # A household reporting coarser than the grid records how much was
        # consumed but not when within its reporting interval, so averaging its
        # readings in directly would place several grid intervals of consumption
        # in one slot. shape_and_level_average takes the distribution across the
        # day from the households reporting at the grid cadence and the level
        # from all of them. Where every household reports at the grid cadence
        # the result is the arithmetic mean.
        avg_diff, shape_level = shape_and_level_average(
            df_filtered, col, project_id_column, cadence,
            grid_minutes=grid_minutes,
        )
        avg_diff.columns = [project_id_column, "ReadingDate", f"{col}_avg"]
        impute_na = avg_diff[col + "_avg"].isna().sum()
        if impute_na > 0:
            logging.error(
                f"Average column `{col}_avg` has {impute_na} missing impute values.",
            )

        avg_diff_dict[col] = {
            "avg_diff": avg_diff,
            "upper_bounds": upper_bounds,
            "household_max_with_bounds": household_max_with_bounds,
            "n_discarded": n_discarded,
            "n_flagged": n_flagged,
            "shape_level": shape_level,
        }

    return avg_diff_dict


def calculate_average_diff_ibis(
    tbl,
    project_id_column: str,
    diff_columns: list[str],
    bounds: dict | None = None,
    cadence: pd.DataFrame | None = None,
    grid_minutes: float | None = None,
) -> dict:
    """
    Ibis/DuckDB variant of calculate_average_diff.

    Accepts an ibis.Table and performs all aggregations as SQL so the full
    dataset never materialises in Python memory.  Only the small result tables
    (one row per project, or one row per project x date) are executed, and the
    scaling step they feed is the one the pandas path uses.

    Returns the same dict structure as calculate_average_diff so that callers
    can substitute either function without downstream changes.

    Parameters
    ----------
    tbl : ibis.Table
        Household data table (e.g. from get_household_tables()["default"]).
    project_id_column : str
        Name of the project ID column.
    diff_columns : list[str]
        Diff column names to process.
    bounds : dict, optional
        Column -> upper threshold, as in calculate_average_diff.
    cadence : pd.DataFrame, optional
        The per-column cadence table, as in calculate_average_diff.
    grid_minutes : float, optional
        The grid the data sits on, as in calculate_average_diff.
    """
    import ibis
    import ibis.selectors as s

    logging.info("Calculating Diff column averages (Ibis).")

    if bounds is None:
        bounds = {
            col: thresholds_dict[col]["Max"]
            for col in diff_columns
            if col in thresholds_dict
        }
    if cadence is None:
        cadence = load_column_cadence()
    if grid_minutes is None:
        grid_minutes = dataset_grid_minutes()

    hh_max_agg = {f"{col}_household_max": tbl[col].cast("Float64").max() for col in diff_columns}
    household_max = tbl.group_by([project_id_column, "HuisIdBSV"]).aggregate(**hh_max_agg)

    avg_diff_dict = {}

    for col in diff_columns:
        logging.info(f"Handling column (Ibis): {col}")
        max_col = f"{col}_household_max"
        n_discarded = 0
        n_flagged = 0
        shape_level = {"shape_households": 0, "level_households": 0, "fallback": False}

        # 95th percentile of per-household max (near-zero excluded), doubled -> upper bound.
        # All projects appear in ub_tbl; projects with no qualifying households get NULL so
        # that the return dict matches the pandas version's safe_quantile(-> pd.NA) behaviour.
        filtered_max = household_max.filter(household_max[max_col] > 1e-8)
        ub_inner = filtered_max.group_by(project_id_column).aggregate(
            **{f"{col}_upper_bound": filtered_max[max_col].quantile(0.95) * 2}
        )
        all_projects = household_max.select(project_id_column).distinct()
        ub_tbl = all_projects.left_join(ub_inner, project_id_column).select(
            ~s.endswith("_right")
        )

        # Join upper bounds back to get include/exclude status per household
        joined = (
            household_max
            .left_join(ub_tbl, project_id_column)
            .select(~s.endswith("_right"))
        )
        included_ids = (
            joined
            .filter(joined[max_col] < joined[f"{col}_upper_bound"])
            .select("HuisIdBSV")
            .execute()["HuisIdBSV"]
            .tolist()
        )

        if not included_ids:
            # No households qualify (all-zero or all-null diffs, or all above upper bound).
            # Return an empty avg_diff with the correct schema rather than calling isin([])
            # which is undefined behaviour in some SQL backends.
            logging.warning(
                f"No households qualified for column `{col}`; avg_diff will be empty."
            )
            avg_diff = pd.DataFrame(
                {
                    project_id_column: pd.array([], dtype="Int64"),
                    "ReadingDate": pd.array([], dtype="datetime64[us]"),
                    f"{col}_avg": pd.array([], dtype="Float64"),
                }
            )
        else:
            # Check for negative values in filtered data (mirrors the pandas version's guard)
            has_negative = (
                tbl
                .filter(tbl["HuisIdBSV"].isin(included_ids))
                .filter(tbl[col] < 0)
                .count()
                .execute()
            )
            if has_negative > 0:
                raise ValueError("Negative Diff values found")

            included = tbl.filter(tbl["HuisIdBSV"].isin(included_ids))
            value = included[col].cast("float64")

            # The interval each household's reading covers. Households absent
            # from the cadence table report at the grid cadence.
            minutes = _per_household(
                included["HuisIdBSV"],
                cadence_minutes_by_household(col, cadence),
                grid_minutes,
            )

            # Exclude values that were not measured at this slot: written by the
            # pipeline, or recorded for a window rather than for this interval.
            # See calculate_average_diff for why a threshold cannot identify
            # them. Mapped data produced before flags existed has no findings
            # column, in which case nothing is excluded here.
            findings = flag_column(col)
            if findings in included.columns:
                not_measured_here = (
                    included[findings].fill_null(0) & _NOT_MEASURED_HERE
                ) != 0
                n_flagged = int(included.filter(not_measured_here).count().execute())
                if n_flagged:
                    logging.info(
                        f"Excluded {n_flagged} value(s) from the {col} average: "
                        f"invented or recorded for a different window."
                    )
                    value = not_measured_here.ifelse(
                        ibis.null().cast("float64"), value
                    )

            # A value above the variable's threshold is implausible rather than
            # merely large, so it is discarded instead of clipped. The threshold
            # is scaled to each household's cadence, as in the pandas path.
            bound = (bounds or {}).get(col, pd.NA)
            if pd.notna(bound):
                scaled_bound = _per_household(
                    included["HuisIdBSV"],
                    column_bound_by_household(col, cadence),
                    bound,
                )
                over = value > scaled_bound
                n_discarded = int(included.filter(over).count().execute())
                if n_discarded:
                    logging.info(
                        f"Discarded {n_discarded} value(s) above the column "
                        f"bound for {col} before averaging."
                    )
                # NULL the value rather than filtering the row out. Filtering
                # would drop a (project, timestamp) group entirely when every
                # value in it is discarded, while the pandas path keeps the
                # group with an NA average.
                value = over.ifelse(ibis.null().cast("float64"), value)

            prepared = included.mutate(_value=value, _minutes=minutes)
            prepared = prepared.mutate(
                _rate=prepared["_value"] / (prepared["_minutes"] / grid_minutes),
                _at_grid=prepared["_minutes"] == float(grid_minutes),
            )
            avg_diff, shape_level = _shape_and_level_average_ibis(
                prepared, col, project_id_column
            )

        impute_na = avg_diff[f"{col}_avg"].isna().sum()
        if impute_na > 0:
            logging.error(
                f"Average column `{col}_avg` has {impute_na} missing impute values.",
            )

        # household_max is a measurement column, so it is promoted to Float64 to
        # match the explicit cast the pandas path applies to the same column.
        avg_diff_dict[col] = {
            "avg_diff": avg_diff,
            "upper_bounds": _to_nullable(ub_tbl.execute()),
            "household_max_with_bounds": _to_nullable(
                joined.execute(), float_cols=[max_col]
            ),
            "n_discarded": n_discarded,
            "n_flagged": n_flagged,
            "shape_level": shape_level,
        }

    return avg_diff_dict


def _to_nullable(df: pd.DataFrame, float_cols: list[str] | None = None) -> pd.DataFrame:
    """Cast numpy int dtypes to Int64 and nominated float cols to Float64.

    int32/int64 cannot hold pd.NA so they are always promoted to Int64.
    float64 supports NaN but is promoted to Float64 only for explicitly
    nominated measurement columns, keeping non-measurement floats (e.g.
    upper_bound, which safe_quantile returns as plain float) unchanged so
    that dtypes match the pandas path exactly.
    """
    for c in df.select_dtypes(include=["int32", "int64"]).columns:
        df[c] = df[c].astype("Int64")
    for c in float_cols or []:
        if c in df.columns:
            df[c] = df[c].astype("Float64")
    return df


def _per_household(household_column, by_household, default):
    """
    A per-household number as an ibis expression, defaulting where unknown.

    An empty mapping means nothing is known per household, which is the state
    when no cadence sidecar exists, so every row takes *default*.
    """
    import ibis

    known = {
        int(household): float(value)
        for household, value in by_household.items()
        if pd.notna(value)
    }
    if not known:
        return ibis.literal(float(default)).cast("float64")
    return household_column.substitute(known, else_=float(default)).cast("float64")


def _shape_and_level_average_ibis(prepared, diff_col, project_id_column):
    """
    The project average of *diff_col*, across households reporting at different
    cadences, computed in SQL.

    The counterpart of :func:`etdtransform.bounds.shape_and_level_average`: the
    distribution across the day comes from the households reporting at the grid
    cadence, the level from all of them, and the two are combined by the same
    scaling function the pandas path uses.

    Only the aggregates are executed -- one row per (project, ReadingDate) for
    the shape and one row per project for the levels.

    Parameters
    ----------
    prepared : ibis.Table
        Carrying ``_value`` (the value after exclusions), ``_rate`` (that value
        per grid interval) and ``_at_grid``.
    diff_col : str
        Interval column being averaged.
    project_id_column : str
        Column identifying the project.

    Returns
    -------
    tuple of (pd.DataFrame, dict)
        As :func:`etdtransform.bounds.shape_and_level_average` returns.
    """
    import ibis

    levels = _to_nullable(
        prepared.group_by(project_id_column)
        .aggregate(
            level_all=prepared["_rate"].mean(),
            level_grid=prepared["_at_grid"]
            .ifelse(prepared["_rate"], ibis.null().cast("float64"))
            .mean(),
            shape_households=prepared["_at_grid"]
            .ifelse(prepared["HuisIdBSV"], ibis.null())
            .nunique(),
            level_households=prepared["HuisIdBSV"].nunique(),
        )
        .execute(),
        float_cols=["level_all", "level_grid"],
    )

    at_grid = prepared.filter(prepared["_at_grid"])
    shape = _to_nullable(
        at_grid.group_by([project_id_column, "ReadingDate"])
        .aggregate(shape=at_grid["_value"].mean())
        .execute(),
        float_cols=["shape"],
    )

    # A project with no grid-cadence household has no shape to scale, and must
    # not simply drop out of the result. Its average is the mean of the
    # per-grid-interval rates instead.
    without_shape = set(levels.loc[levels["shape_households"] == 0, project_id_column])
    fallback = None
    if without_shape:
        rows = prepared.filter(
            prepared[project_id_column].isin(sorted(without_shape))
        )
        fallback = _to_nullable(
            rows.group_by([project_id_column, "ReadingDate"])
            .aggregate(**{f"{diff_col}_avg": rows["_rate"].mean()})
            .execute(),
            float_cols=[f"{diff_col}_avg"],
        )

    indexed = levels.set_index(project_id_column)
    avg = scale_shape_to_level(
        shape,
        indexed["level_all"],
        indexed["level_grid"],
        project_id_column,
        diff_col,
        fallback=fallback,
    )
    return avg, {
        "shape_households": int(levels["shape_households"].sum()),
        "level_households": int(levels["level_households"].sum()),
        "fallback": bool(without_shape),
    }


def concatenate_household_max_with_bounds(avg_diff_dict, project_id_column):
    """
    Concatenate household maximum values and bounds for all columns.

    Each column may cover a different subset of (project, household) pairs
    (e.g. one supplier vs another). An outer merge on the key columns is used
    so that households missing a column receive pd.NA rather than being
    silently assigned another column's values via positional alignment.
    """
    result_df = None
    for col, data in avg_diff_dict.items():
        df_col = data["household_max_with_bounds"][
            [project_id_column, "HuisIdBSV", f"{col}_household_max", f"{col}_upper_bound"]
        ]
        if result_df is None:
            result_df = df_col
        else:
            result_df = result_df.merge(
                df_col, on=[project_id_column, "HuisIdBSV"], how="outer"
            )
    return result_df


def concatenate_avg_diff_columns(avg_diff_dict, project_id_column):
    """
    Concatenate average difference columns for all variables.

    Each column may cover a different subset of (project, date) pairs
    (e.g. one supplier vs another). An outer merge on the key columns is used
    so that projects missing a column receive pd.NA rather than being
    silently assigned another column's values via positional alignment.
    """
    result_df = None
    for col, data in avg_diff_dict.items():
        df_col = data["avg_diff"]
        if result_df is None:
            result_df = df_col
        else:
            result_df = result_df.merge(
                df_col, on=[project_id_column, "ReadingDate"], how="outer"
            )
    return result_df


def equal_sig_fig(a, b, sig_figs):
    """
    Compare two numbers for equality up to a specified number of significant figures.

    This function rounds both numbers to the specified number of significant figures
    and then compares them for equality using a relative tolerance that scales with
    the magnitude of the numbers.

    Parameters
    ----------
    a : float
        The first number to compare.
    b : float
        The second number to compare.
    sig_figs : int
        The number of significant figures to consider for comparison.

    Returns
    -------
    bool
        True if the numbers are equal up to the specified number of significant figures,
        False otherwise.

    Notes
    -----
    This function uses the `isclose` function from the `math` module to compare the
    rounded numbers with a relative tolerance based on the number of significant figures.

    """
    # Define a helper function to scale the number to significant figures
    def round_to_sig_figs(x, sig_figs):
        if x == 0:
            return 0
        return round(x, sig_figs - int(floor(log10(abs(x)))) - 1)

    # Round both numbers to the specified significant figures
    a_rounded = round_to_sig_figs(a, sig_figs)
    b_rounded = round_to_sig_figs(b, sig_figs)

    # Apply a relative tolerance that scales with the magnitude of the numbers
    tolerance = 10 ** (-sig_figs)

    # Use isclose to compare with relative tolerance
    return isclose(a_rounded, b_rounded, rel_tol=tolerance)




def validate_household_column(household_df, cum_col, household_id):
    """
    Validate a household column for data quality and completeness.

    This function checks a specific column in a household DataFrame for missing values,
    zero values, and lack of change. It logs warnings and information about the data quality.

    Parameters
    ----------
    household_df : pd.DataFrame
        The DataFrame containing household data.
    cum_col : str
        The name of the cumulative column to validate.
    household_id : str
        The unique identifier for the household.

    Returns
    -------
    bool
        True if the column passes all checks, False otherwise.

    Notes
    -----
    This function is currently unused in the main processing pipeline.

    Warnings
    --------
    - Logs a warning if more than 40% of values in the column are missing.
    - Logs information about the number of missing values, zero values, and lack of change.

    """
    n_na = household_df[cum_col].isna().sum()
    len_df = len(household_df.index)

    if n_na == len_df:
        logging.info(
            f"HuisIdBSV {household_id} has all {n_na} missing values in {cum_col} of {len_df} records. Skipping column.",
        )
        return False
    elif n_na / len_df > 0.4:
        percent_na = 100 * n_na / len_df
        logging.error(
            f"HuisIdBSV {household_id} has {percent_na:.2f}% missing values in {cum_col}. Consider removing.",
        )
    else:
        logging.info(
            f"HuisIdBSV {household_id} has {n_na} missing values in {cum_col} of {len_df} records.",
        )

    if round(household_df[cum_col].sum(), 10) == 0:
        logging.info(
            f"HuisIdBSV {household_id} has no non-zero values in {cum_col}. Skipping column.",
        )
        return False
    if round(household_df[cum_col].max() - household_df[cum_col].min(), 10) == 0:
        logging.info(
            f"HuisIdBSV {household_id} has no change in {cum_col}. Skipping column.",
        )
        return False
    if round(household_df[f"{cum_col}Diff"].sum(), 10) == 0:
        logging.warning(
            f"HuisIdBSV {household_id} has no non-zero values in {cum_col}Diff before imputation.",
        )

    return True


# currently unused - apply as a sense check to ensure not too many values are missing
def get_reading_date_imputation_stats(df, project_id_column, cumulative_columns):
    """
    Calculate imputation statistics for each reading date and cumulative column.

    This function computes various statistics related to imputation for each reading date
    and cumulative column, including the number of imputed values, missing values, and
    original values.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    project_id_column : str
        The name of the column containing project IDs.
    cumulative_columns : list
        A list of cumulative column names to analyze.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing imputation statistics for each reading date and column.

    Notes
    -----
    This function is currently unused but can be applied as a sense check to ensure
    not too many values are missing.

    The resulting DataFrame includes the following columns:
    - project_id_column
    - ReadingDate
    - column
    - imputed
    - na
    - total_records
    - original
    - percent_imputed
    - percent_na
    - percent_original

    """
    grouped = df.groupby([project_id_column, "ReadingDate"])
    total_stats = grouped.size().rename("total_records")

    df_list = []
    for col in cumulative_columns:
        logging.info(f"Calculating imputation statistics by ReadingDate for {col}")

        diff_col = f"{col}Diff"
        is_imputed_col = f"{diff_col}_is_imputed"

        imputed_stats = grouped[is_imputed_col].sum().rename("imputed")
        na_stats = grouped[diff_col].apply(lambda x: x.isna().sum()).rename("na")

        stats_df = pd.concat(
            [imputed_stats, na_stats, total_stats],
            axis=1,
            ignore_index=False,
        )
        # stats_df = pd.concat([imputed_stats, na_stats, total_stats], axis=1).reset_index()

        stats_df["original"] = (
            stats_df["total_records"] - stats_df["imputed"] - stats_df["na"]
        )
        stats_df["percent_imputed"] = (
            stats_df["imputed"] / stats_df["total_records"]
        ) * 100
        stats_df["percent_na"] = (stats_df["na"] / stats_df["total_records"]) * 100
        stats_df["percent_original"] = (
            stats_df["original"] / stats_df["total_records"]
        ) * 100
        stats_df["column"] = col

        # Append the DataFrame to the list
        df_list.append(stats_df)

    # Concatenate all the DataFrames in the list
    logging.info(f"Concatenating the reading date statistics")
    imputation_reading_date_stats_df = pd.concat(
        df_list,
        ignore_index=False,
    ).reset_index()

    return imputation_reading_date_stats_df


def sort_for_impute(df: pd.DataFrame, project_id_column: str):
    """
    Sort the DataFrame to prepare for imputation.

    This function sorts the input DataFrame by project ID, household ID, and reading date.
    Sorting is necessary to ensure correct imputation of missing values.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to be sorted.
    project_id_column : str
        The name of the column containing project IDs.

    Returns
    -------
    pd.DataFrame
        The sorted DataFrame.

    Notes
    -----
    The sorting order is: project ID, household ID (HuisIdBSV), and reading date (ReadingDate).
    This order is crucial for the imputation process to work correctly.

    """
    logging.info("Sorting to prepare for imputation.")
    return df.sort_values(by=[project_id_column, "HuisIdBSV", "ReadingDate"])


def get_diff_columns(cumulative_columns: list):
    """
    Generate difference column names from cumulative column names.

    This function takes a list of cumulative column names and returns a list of
    corresponding difference column names by appending 'Diff' to each name.

    Parameters
    ----------
    cumulative_columns : list
        A list of cumulative column names.

    Returns
    -------
    list
        A list of difference column names.

    Notes
    -----
    This function is used to create names for columns that will store the differences
    between consecutive cumulative values.

    """
    return [col + "Diff" for col in cumulative_columns]

def prepare_diff_averages_for_impute(
    df: pd.DataFrame,
    project_id_column: str,
    cumulative_columns: list,
    sorted=False,
):
    """
    Prepare difference columns for imputation.

    This function calculates average differences, combines them, and prepares
    household maximum and bound information for imputation.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data.
    project_id_column : str
        The name of the column containing project IDs.
    cumulative_columns : list
        A list of cumulative column names.
    sorted : bool, optional
        Whether the DataFrame is already sorted. Default is False.

    Returns
    -------
    tuple
        A tuple containing:
        - diff_columns: list of difference column names
        - diffs: DataFrame with average differences
        - max_bound: DataFrame with household maximum and bound information

    Notes
    -----
    This function performs the following steps:
    1. Sorts the DataFrame if not already sorted.
    2. Calculates average differences for each cumulative column.
    3. Combines average differences and household maximum/bound information.
    4. Saves the results to parquet files for later use.

    The resulting files are saved in the directory specified by
    etdtransform.options.aggregate_folder_path.

    """
    if not sorted:
        df = sort_for_impute(df, project_id_column)

    diff_columns = get_diff_columns(cumulative_columns)

    logging.info("Starting to prepare diff averages.")
    avg_diff_dict = calculate_average_diff(df, project_id_column, diff_columns)
    logging.info("Combining average diff columns.")
    diffs = concatenate_avg_diff_columns(avg_diff_dict, project_id_column)
    logging.info("Combining household diff maximum and bounds used for diff columns.")
    max_bound = concatenate_household_max_with_bounds(avg_diff_dict, project_id_column)

    logging.info("Saving average diff columns in avg_diffs.parquet")
    diffs.to_parquet(
        os.path.join(etdtransform.options.aggregate_folder_path, "avg_diffs.parquet"),
        engine="pyarrow",
    )
    logging.info(
        "Saving household diff max and bounds used in household_diff_max_bounds.parquet",
    )
    max_bound.to_parquet(
        os.path.join(etdtransform.options.aggregate_folder_path, "household_diff_max_bounds.parquet"),
        engine="pyarrow",
    )

    return diff_columns, diffs, max_bound


def prepare_diff_averages_for_impute_ibis(
    tbl,
    project_id_column: str,
    cumulative_columns: list,
):
    """
    Ibis variant of prepare_diff_averages_for_impute.

    Accepts an ibis.Table (e.g. from ibis.read_parquet) instead of a pandas
    DataFrame.  All heavy aggregations run as SQL inside DuckDB; only the small
    result tables (one row per project or per project x date) are materialised and
    saved as parquet artefacts.  The input table is never executed in full.

    Parameters
    ----------
    tbl : ibis.Table
        Lazy household table containing the Diff columns for each cumulative column.
    project_id_column : str
        Name of the project ID column.
    cumulative_columns : list
        List of cumulative column names (without "Diff" suffix).

    Returns
    -------
    tuple
        (diff_columns, diffs, max_bound) -- same structure as
        prepare_diff_averages_for_impute.
    """
    diff_columns = get_diff_columns(cumulative_columns)

    logging.info("Starting to prepare diff averages (Ibis).")
    avg_diff_dict = calculate_average_diff_ibis(tbl, project_id_column, diff_columns)

    logging.info("Combining average diff columns.")
    diffs = concatenate_avg_diff_columns(avg_diff_dict, project_id_column)

    logging.info("Combining household diff maximum and bounds used for diff columns.")
    max_bound = concatenate_household_max_with_bounds(avg_diff_dict, project_id_column)

    logging.info("Saving average diff columns in avg_diffs.parquet")
    diffs.to_parquet(
        os.path.join(etdtransform.options.aggregate_folder_path, "avg_diffs.parquet"),
        engine="pyarrow",
    )
    logging.info(
        "Saving household diff max and bounds used in household_diff_max_bounds.parquet",
    )
    max_bound.to_parquet(
        os.path.join(etdtransform.options.aggregate_folder_path, "household_diff_max_bounds.parquet"),
        engine="pyarrow",
    )

    return diff_columns, diffs, max_bound


def read_diffs():
    """
    Read average differences from a parquet file.

    This function reads the average differences data from a parquet file
    located in the aggregate folder specified in the etdtransform options.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the average differences data.

    Notes
    -----
    The function assumes that the 'avg_diffs.parquet' file exists in the
    aggregate folder path specified in etdtransform.options.aggregate_folder_path.

    This function is typically used to load pre-calculated average differences
    for use in imputation processes.

    """
    return pd.read_parquet(
        os.path.join(etdtransform.options.aggregate_folder_path, "avg_diffs.parquet"),
        dtype_backend="numpy_nullable",
    )


def process_and_impute(
    df: pd.DataFrame,
    project_id_column: str,
    cumulative_columns: list,
    sorted=False,
    diffs_calculated=False,
    optimized=False,
):
    """
    Process and impute missing values in the dataset.

    This function performs data processing and imputation on the input DataFrame.
    It can either calculate differences or load pre-calculated differences,
    and then applies imputation methods to fill missing values.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data to be processed and imputed.
    project_id_column : str
        The name of the column containing project IDs.
    cumulative_columns : list
        A list of cumulative column names to be processed.
    sorted : bool, optional
        Whether the DataFrame is already sorted. Default is False.
    diffs_calculated : bool, optional
        Whether differences have already been calculated. Default is False.
    optimized : bool, optional
        Whether to use optimized imputation methods. Default is False.

    Returns
    -------
    tuple
        A tuple containing:
        - df: The processed and imputed DataFrame
        - imputation_summary_house: Summary of imputation statistics per house
        - imputation_summary_project: Summary of imputation statistics per project
        - imputation_reading_date_stats_df: Statistics of imputation by reading date

    Notes
    -----
    This function performs the following steps:
    1. Sorts the DataFrame if not already sorted.
    2. Loads or calculates differences.
    3. Merges average differences into the household DataFrame.
    4. Applies imputation methods (either optimized or standard).
    5. Calculates and saves imputation statistics.
    6. Provides warnings for high imputation percentages.

    The function saves various statistics and summary files in the
    aggregate folder specified in etdtransform.options.aggregate_folder_path.

    Warnings
    --------
    - Logs warnings if any house or project has more than 40% imputed values.
    - Logs warnings if any reading date has more than 40% imputed values.

    """
    if not sorted:
        df = sort_for_impute(df, project_id_column)

    if diffs_calculated:
        logging.info("Loading average diffs from file...")
        diffs = read_diffs()
        max_bound = pd.read_parquet(
            os.path.join(etdtransform.options.aggregate_folder_path, "household_diff_max_bounds.parquet"),
            dtype_backend="numpy_nullable",
        )
    else:
        diff_columns, diffs, max_bound = prepare_diff_averages_for_impute(
            df=df,
            project_id_column=project_id_column,
            cumulative_columns=cumulative_columns,
            sorted=True,
        )

    logging.info(
        "Merging the average differences into the household dataframe for imputation.",
    )
    df = df.merge(diffs, on=[project_id_column, "ReadingDate"], how="left")

    logging.info("Merge completed.")

    if optimized:
        optimized_label = "_optimized"
        df, imputation_gap_stats_df, imputation_reading_date_stats_df = (
            impute_and_normalize_optimized(
                df,
                cumulative_columns,
                project_id_column,
                max_bound,
            )
        )
    else:
        optimized_label = ""
        df, imputation_gap_stats_df, imputation_reading_date_stats_df = (
            impute_and_normalize(df, cumulative_columns, project_id_column, max_bound)
        )

    logging.info("Saving imputation gap statistics...")
    imputation_gap_stats_df.to_parquet(
        os.path.join(
            etdtransform.options.aggregate_folder_path,
            f"impute_gap_stats{optimized_label}.parquet",
        ),
        engine="pyarrow",
    )

    logging.info("Summarizing imputation_gap_stats_df per house and column")
    imputation_summary_house = imputation_gap_stats_df[
        [
            project_id_column,
            "HuisIdBSV",
            "column",
            "diff_col_total",
            "cum_col_min_max_diff",
            "missing",
            "imputed",
            "imputed_na",
            "methods",
            "bitwise_methods",
        ]
    ].reset_index()

    logging.info("Calculating the total records for each house")
    total_records_house = (
        df.groupby("HuisIdBSV").size().reset_index(name="total_records")
    )

    logging.info("Merging total records with house imputation summary")
    imputation_summary_house = imputation_summary_house.merge(
        total_records_house,
        on=["HuisIdBSV"],
    )
    imputation_summary_house["percentage_imputed"] = (
        imputation_summary_house["imputed"] / imputation_summary_house["total_records"]
    ) * 100

    logging.info("Summarizing imputation_gap_stats_df per project and column")
    imputation_summary_project = (
        imputation_gap_stats_df.groupby([project_id_column, "column"])
        .agg(
            {
                "bitwise_methods": lambda x: np.bitwise_or.reduce(x),
                "methods": lambda x: list(set().union(*x)),
                "missing": "sum",
                "imputed": "sum",
                "imputed_na": "sum",
            },
        )
        .reset_index()
    )

    logging.info(
        "Calculate the total records for each project and column from the original dataframe",
    )
    total_records_project = (
        df.groupby(project_id_column).size().reset_index(name="total_records")
    )

    logging.info("Merge total records with project imputation summary")
    imputation_summary_project = imputation_summary_project.merge(
        total_records_project,
        on=[project_id_column],
    )
    imputation_summary_project["percentage_imputed"] = (
        imputation_summary_project["imputed"]
        / imputation_summary_project["total_records"]
    ) * 100

    logging.info("Provide warnings if any house has > 40% imputed")
    over_40_percent_imputed_house = imputation_summary_house[
        imputation_summary_house["percentage_imputed"] > 40
    ]
    for _, row in over_40_percent_imputed_house.iterrows():
        logging.warning(
            f"House {row['HuisIdBSV']}, Column {row['column']} has {row['percentage_imputed']:.2f}% imputed values.",
        )

    logging.info("Provide warnings if any project has > 40% imputed")
    over_40_percent_imputed_project = imputation_summary_project[
        imputation_summary_project["percentage_imputed"] > 40
    ]
    for _, row in over_40_percent_imputed_project.iterrows():
        logging.warning(
            f"Project {row[project_id_column]}, Column {row['column']} has {row['percentage_imputed']:.2f}% imputed values.",
        )

    if imputation_reading_date_stats_df:
        logging.info("Provide warnings if any ReadingDates are over 40% imputed values")
        over_40_percent_imputed_dates = imputation_reading_date_stats_df[
            imputation_reading_date_stats_df["percent_imputed"] > 40
        ]
        for _, row in over_40_percent_imputed_dates.iterrows():
            logging.warning(
                f"ReadingDate {row['ReadingDate']}, Project {row[project_id_column]}, Column {row['column']} has {row['percent_imputed']:.2f}% imputed values.",
            )
    else:
        logging.warning("Not calculating reading date stats")

    return (
        df,
        imputation_summary_house,
        imputation_summary_project,
        imputation_reading_date_stats_df,
    )

def assert_imputation_input_batch_safe(df: pd.DataFrame) -> None:
    """
    Reject imputation input that cannot be imputed correctly.

    Expects one row per (HuisIdBSV, ReadingDate), and at most one
    HuisBatchIdBSV per HuisIdBSV. Call before averaging or imputing.

    Raises
    ------
    HouseBatchOverlapError
        A household has rows in more than one batch. Imputing the
        concatenated series would fill the between-batch gap, which is real
        non-delivery rather than missing data.
    ValueError
        Duplicate (HuisIdBSV, ReadingDate) rows. These double-weight the
        project averages and break the diff and sort order.
    """
    from etdmap.index_helpers import HouseBatchOverlapError

    if "HuisBatchIdBSV" in df.columns:
        counts = df.groupby("HuisIdBSV")["HuisBatchIdBSV"].nunique()
        multi = sorted(int(h) for h, n in counts.items() if n > 1)
        if multi:
            raise HouseBatchOverlapError(
                f"[imputation] Household(s) {multi} span more than one batch. "
                f"Imputing a concatenated multi-batch series would fill the "
                f"between-batch gap (real non-delivery). Batch-aware imputation "
                f"is not implemented yet; impute per batch or exclude."
            )
    dup_mask = df.duplicated(subset=["HuisIdBSV", "ReadingDate"])
    if bool(dup_mask.any()):
        bad = sorted(int(h) for h in df.loc[dup_mask, "HuisIdBSV"].dropna().unique())
        raise ValueError(
            f"[imputation] Duplicate (HuisIdBSV, ReadingDate) rows for household(s) "
            f"{bad} -- overlapping batch periods double-weight project averages and "
            f"break the diff math. Resolve the batch overlap first."
        )


def _assert_batch_safe_ibis(tbl) -> None:
    """Engine-side variant of assert_imputation_input_batch_safe for lazy tables
    (cheap aggregates; the table is never materialised in full)."""
    from etdmap.index_helpers import HouseBatchOverlapError

    if "HuisBatchIdBSV" in tbl.columns:
        per_hh = (
            tbl.group_by("HuisIdBSV")
            .aggregate(n_batches=tbl["HuisBatchIdBSV"].nunique())
        )
        multi = per_hh.filter(per_hh.n_batches > 1)["HuisIdBSV"].execute()
        if len(multi):
            raise HouseBatchOverlapError(
                f"[prepare_diff_averages] Household(s) {sorted(int(h) for h in multi)} span "
                f"more than one batch; project averages would mix batches. "
                f"Batch-aware averaging is not implemented yet."
            )
    n_rows = int(tbl.count().execute())
    n_keys = int(tbl[["HuisIdBSV", "ReadingDate"]].distinct().count().execute())
    if n_rows != n_keys:
        raise ValueError(
            f"[prepare_diff_averages] {n_rows - n_keys} duplicate (HuisIdBSV, ReadingDate) "
            f"row(s) -- overlapping batch periods would double-weight the project "
            f"averages. Resolve the batch overlap first."
        )


def prepare_diff_averages_sharded(
    mapped_folder_path=None,
    aggregate_folder_path=None,
    cumulative_columns: list = None,
    household_ids=None,
):
    """
    Sharded version of the avg-diffs stage: compute the imputation estimation
    inputs (avg_diffs.parquet + household_diff_max_bounds.parquet) directly
    from the SHARDED mapped data.

    I/O shell only -- the computation is the SAME prepare_diff_averages_for_impute_ibis
    the legacy path uses, fed a lazy table built from mapped_household_table
    (DuckDB hive-glob, per the measured perf decision) with `ProjectIdBSV` joined
    from the index and `Meenemen` inclusion applied at read time.

    Outputs are written under ``<aggregate_folder_path>/sharded/`` (the sharded
    pipeline's artifact area), keeping legacy artifacts untouched.

    Batch safety: refuses multi-batch households and overlapping batch periods
    (see assert_imputation_input_batch_safe) -- the estimation inputs feed
    everything downstream.
    """
    import ibis as _ibis

    from etdmap.index_helpers import read_index
    from etdtransform.load_data import included_household_ids, mapped_household_table

    if mapped_folder_path is None:
        mapped_folder_path = etdtransform.options.mapped_folder_path
    if aggregate_folder_path is None:
        raise ValueError(
            "prepare_diff_averages_sharded: aggregate_folder_path is required and has "
            "no default (the config value points at the promoted production "
            "area)."
        )

    if household_ids is not None:
        ids = sorted(int(x) for x in household_ids)
    else:
        ids = included_household_ids(mapped_folder_path)

    index_df, _ = read_index(mapped_folder_path)
    if "HuisBatchIdBSV" not in index_df.columns:
        raise ValueError(
            "index.parquet has no HuisBatchIdBSV column; "
            "re-run the index save to add the household-batch columns."
        )
    # Keyed on the household batch, NOT the household: a household's project assignment is
    # constant WITHIN a batch but may differ BETWEEN batches (project
    # definitions change over time; the household does not). So
    # HuisIdBSV -> ProjectIdBSV is not a function over time, while
    # HuisBatchIdBSV -> ProjectIdBSV is. HuisBatchIdBSV is also the primary key
    # of index.parquet (one row per household batch), so keying on HuisIdBSV
    # would additionally keep whichever batch row happened to come last.

    household_batch_projects = index_df[["HuisBatchIdBSV", "ProjectIdBSV"]].dropna()
    project_map = _ibis.memtable(
        pd.DataFrame({
            "HuisBatchIdBSV": household_batch_projects["HuisBatchIdBSV"].astype("int64"),
            "ProjectIdBSV": household_batch_projects["ProjectIdBSV"].astype("int64"),
        })
    )

    tbl = mapped_household_table(mapped_folder_path)
    tbl = tbl.filter(tbl["HuisIdBSV"].isin(ids))
    tbl = tbl.join(project_map, tbl["HuisBatchIdBSV"] == project_map["HuisBatchIdBSV"])

    _assert_batch_safe_ibis(tbl)

    out_dir = os.path.join(str(aggregate_folder_path), "sharded")
    os.makedirs(out_dir, exist_ok=True)

    # prepare_diff_averages_for_impute_ibis writes via options.aggregate_folder_path;
    # scope it to the sharded artifact area for this call.
    previous = etdtransform.options.aggregate_folder_path
    etdtransform.options.aggregate_folder_path = out_dir
    try:
        return prepare_diff_averages_for_impute_ibis(
            tbl,
            project_id_column="ProjectIdBSV",
            cumulative_columns=cumulative_columns,
        )
    finally:
        etdtransform.options.aggregate_folder_path = previous
