import logging
import os
from math import floor, isclose, log10

import numpy as np
import pandas as pd

import etdtransform
from etdtransform.vectorized_impute import impute_and_normalize


def calculate_average_diff(
    df: pd.DataFrame,
    project_id_column: str,
    diff_columns: list[str],
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

    Returns
    -------
    dict
        A dictionary where keys are column names and values are dictionaries containing:
        - 'avg_diff': DataFrame with average differences
        - 'upper_bounds': DataFrame with upper bounds for outlier exclusion
        - 'household_max_with_bounds': DataFrame with household maximum values and bounds

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
        f"{col}_huis_max" for col in diff_columns
    ]

    for col in diff_columns:
        household_max[f"{col}_huis_max"] = household_max[f"{col}_huis_max"].astype(
            "Float64",
        )

    avg_diff_dict = {}

    for col in diff_columns:
        logging.info(f"Handling column: {col}")

        logging.info(f"Calculating the 95th percentile (upper bound) for {col}.")
        upper_bounds = (
            household_max.groupby(project_id_column)
            .apply(safe_quantile, f"{col}_huis_max")
            .reset_index()
        )
        upper_bounds.columns = [project_id_column, f"{col}_upper_bound"]
        upper_bounds[f"{col}_upper_bound"] = upper_bounds[
            f"{col}_upper_bound"
        ].multiply(2)

        logging.info(f"Identifying households to include for {col}.")
        household_max_with_bounds = household_max[
            [project_id_column, "HuisIdBSV", f"{col}_huis_max"]
        ].merge(upper_bounds, on=project_id_column, how="left")
        include_mask = (
            household_max_with_bounds[f"{col}_huis_max"]
            < household_max_with_bounds[f"{col}_upper_bound"]
        )
        households_to_include = household_max_with_bounds.loc[include_mask, "HuisIdBSV"]

        logging.info(f"Filtering the dataframe for {col}.")
        df_filtered = df[["HuisIdBSV", project_id_column, "ReadingDate", col]][
            df["HuisIdBSV"].isin(households_to_include)
        ][[project_id_column, "ReadingDate", col]]

        logging.info(f"Checking for negative Diff values in {col}.")
        if (df_filtered[col] < 0).any():
            raise ValueError("Negative Diff values found")
        # df_filtered[df_filtered[col]<0][col] = pd.NA

        logging.info(f"Calculating the average differences for {col}.")
        avg_diff = (
            df_filtered.groupby([project_id_column, "ReadingDate"])[col]
            .mean()
            .reset_index()
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
        }

    return avg_diff_dict


def calculate_average_diff_ibis(
    tbl,
    project_id_column: str,
    diff_columns: list[str],
) -> dict:
    """
    Ibis/DuckDB variant of calculate_average_diff.

    Accepts an ibis.Table and performs all aggregations as SQL so the full
    dataset never materialises in Python memory.  Only the small result tables
    (one row per project × date) are returned as pandas DataFrames.

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
    """
    import ibis
    import ibis.selectors as s

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

    logging.info("Calculating Diff column averages (Ibis).")

    hh_max_agg = {f"{col}_huis_max": tbl[col].cast("Float64").max() for col in diff_columns}
    household_max = tbl.group_by([project_id_column, "HuisIdBSV"]).aggregate(**hh_max_agg)

    avg_diff_dict = {}

    for col in diff_columns:
        logging.info(f"Handling column (Ibis): {col}")
        max_col = f"{col}_huis_max"

        # 95th percentile of per-household max (near-zero excluded), doubled → upper bound.
        # All projects appear in ub_tbl; projects with no qualifying households get NULL so
        # that the return dict matches the pandas version's safe_quantile(→ pd.NA) behaviour.
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

            avg_diff = _to_nullable(
                tbl
                .filter(tbl["HuisIdBSV"].isin(included_ids))
                .group_by([project_id_column, "ReadingDate"])
                .aggregate(**{f"{col}_avg": tbl[col].cast("Float64").mean()})
                .execute(),
                float_cols=[f"{col}_avg"],
            )

        impute_na = avg_diff[f"{col}_avg"].isna().sum()
        if impute_na > 0:
            logging.error(
                f"Average column `{col}_avg` has {impute_na} missing impute values.",
            )

        # huis_max is a measurement column — promote to Float64 to match the explicit
        # .astype("Float64") in the pandas path (lines 72-74 of calculate_average_diff).
        avg_diff_dict[col] = {
            "avg_diff": avg_diff,
            "upper_bounds": _to_nullable(ub_tbl.execute()),
            "household_max_with_bounds": _to_nullable(
                joined.execute(), float_cols=[max_col]
            ),
        }

    return avg_diff_dict


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
            [project_id_column, "HuisIdBSV", f"{col}_huis_max", f"{col}_upper_bound"]
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




def validate_household_column(household_df, cum_col, huis_code):
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
    huis_code : str
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
            f"HuisIdBSV {huis_code} has all {n_na} missing values in {cum_col} of {len_df} records. Skipping column.",
        )
        return False
    elif n_na / len_df > 0.4:
        percent_na = 100 * n_na / len_df
        logging.error(
            f"HuisIdBSV {huis_code} has {percent_na:.2f}% missing values in {cum_col}. Consider removing.",
        )
    else:
        logging.info(
            f"HuisIdBSV {huis_code} has {n_na} missing values in {cum_col} of {len_df} records.",
        )

    if round(household_df[cum_col].sum(), 10) == 0:
        logging.info(
            f"HuisIdBSV {huis_code} has no non-zero values in {cum_col}. Skipping column.",
        )
        return False
    if round(household_df[cum_col].max() - household_df[cum_col].min(), 10) == 0:
        logging.info(
            f"HuisIdBSV {huis_code} has no change in {cum_col}. Skipping column.",
        )
        return False
    if round(household_df[f"{cum_col}Diff"].sum(), 10) == 0:
        logging.warning(
            f"HuisIdBSV {huis_code} has no non-zero values in {cum_col}Diff before imputation.",
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

def prepare_diffs_for_impute(
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

    logging.info("Starting to prepare diffs.")
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


def prepare_diffs_for_impute_ibis(
    tbl,
    project_id_column: str,
    cumulative_columns: list,
):
    """
    Ibis variant of prepare_diffs_for_impute.

    Accepts an ibis.Table (e.g. from ibis.read_parquet) instead of a pandas
    DataFrame.  All heavy aggregations run as SQL inside DuckDB; only the small
    result tables (one row per project or per project×date) are materialised and
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
        (diff_columns, diffs, max_bound) — same structure as prepare_diffs_for_impute.
    """
    diff_columns = get_diff_columns(cumulative_columns)

    logging.info("Starting to prepare diffs (Ibis).")
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
        diff_columns, diffs, max_bound = prepare_diffs_for_impute(
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
    Computational batch-safety guard for the averaging/imputation chain.

    Independent of the registry coexistence guard (which deliberately relaxes
    once index.parquet is retired): even in the post-switch era, this chain
    must refuse input it cannot impute correctly --

    - a household spanning MORE THAN ONE batch: concatenating batches would let
      the large-gap project-average method impute across the BETWEEN-batch gap,
      which is real non-delivery (the cross-batch continuity invariant).
      Batch-aware imputation is native-resolution-phase work; until it exists,
      stop loudly (HuisBatchOverlapError).
    - duplicate (HuisIdBSV, ReadingDate) rows: overlapping batch periods
      double-weight project averages and break the sort/diff math (ValueError).
    """
    from etdmap.index_helpers import HuisBatchOverlapError

    if "HuisBatchIdBSV" in df.columns:
        counts = df.groupby("HuisIdBSV")["HuisBatchIdBSV"].nunique()
        multi = sorted(int(h) for h, n in counts.items() if n > 1)
        if multi:
            raise HuisBatchOverlapError(
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
    from etdmap.index_helpers import HuisBatchOverlapError

    if "HuisBatchIdBSV" in tbl.columns:
        per_hh = (
            tbl.group_by("HuisIdBSV")
            .aggregate(n_batches=tbl["HuisBatchIdBSV"].nunique())
        )
        multi = per_hh.filter(per_hh.n_batches > 1)["HuisIdBSV"].execute()
        if len(multi):
            raise HuisBatchOverlapError(
                f"[prepare_diffs] Household(s) {sorted(int(h) for h in multi)} span "
                f"more than one batch; project averages would mix batches. "
                f"Batch-aware averaging is not implemented yet."
            )
    n_rows = int(tbl.count().execute())
    n_keys = int(tbl[["HuisIdBSV", "ReadingDate"]].distinct().count().execute())
    if n_rows != n_keys:
        raise ValueError(
            f"[prepare_diffs] {n_rows - n_keys} duplicate (HuisIdBSV, ReadingDate) "
            f"row(s) -- overlapping batch periods would double-weight the project "
            f"averages. Resolve the batch overlap first."
        )


def prepare_diffs_sharded(
    mapped_folder_path=None,
    aggregate_folder_path=None,
    cumulative_columns: list = None,
    huis_ids=None,
):
    """
    Sharded version of the avg-diffs stage: compute the imputation estimation
    inputs (avg_diffs.parquet + household_diff_max_bounds.parquet) directly
    from the SHARDED mapped data.

    I/O shell only -- the computation is the SAME prepare_diffs_for_impute_ibis
    the legacy path uses, fed a lazy table built from mapped_household_table
    (DuckDB hive-glob, per the measured perf decision) with ProjectIdBSV joined
    from the batch registry and Meenemen inclusion applied at read time.

    Outputs are written under ``<aggregate_folder_path>/sharded/`` (the sharded
    pipeline's artifact area), keeping legacy artifacts untouched.

    Batch safety: refuses multi-batch households and overlapping batch periods
    (see assert_imputation_input_batch_safe) -- the estimation inputs feed
    everything downstream.
    """
    import ibis as _ibis

    from etdmap.index_helpers import read_batch_index
    from etdtransform.load_data import included_household_ids, mapped_household_table

    if mapped_folder_path is None:
        mapped_folder_path = etdtransform.options.mapped_folder_path
    if aggregate_folder_path is None:
        raise ValueError(
            "prepare_diffs_sharded: aggregate_folder_path is required and has "
            "no default (the config value points at the promoted production "
            "area)."
        )

    if huis_ids is not None:
        ids = sorted(int(x) for x in huis_ids)
    else:
        ids = included_household_ids(mapped_folder_path)

    batch_index_df, _ = read_batch_index(mapped_folder_path)
    project_map = _ibis.memtable(
        pd.DataFrame({
            "HuisBatchIdBSV": batch_index_df["HuisBatchIdBSV"].astype("int64"),
            "ProjectIdBSV": batch_index_df["ProjectIdBSV"].astype("int64"),
        })
    )

    tbl = mapped_household_table(mapped_folder_path)
    tbl = tbl.filter(tbl["HuisIdBSV"].isin(ids))
    tbl = tbl.join(project_map, tbl["HuisBatchIdBSV"] == project_map["HuisBatchIdBSV"])

    _assert_batch_safe_ibis(tbl)

    out_dir = os.path.join(str(aggregate_folder_path), "sharded")
    os.makedirs(out_dir, exist_ok=True)

    # prepare_diffs_for_impute_ibis writes via options.aggregate_folder_path;
    # scope it to the sharded artifact area for this call.
    previous = etdtransform.options.aggregate_folder_path
    etdtransform.options.aggregate_folder_path = out_dir
    try:
        return prepare_diffs_for_impute_ibis(
            tbl,
            project_id_column="ProjectIdBSV",
            cumulative_columns=cumulative_columns,
        )
    finally:
        etdtransform.options.aggregate_folder_path = previous
