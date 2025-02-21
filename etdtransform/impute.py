import logging
import os
from math import floor, isclose, log10

import numpy as np
import pandas as pd

import etdtransform
from etdtransform.vectorized_impute import impute_and_normalize_vectorized


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


def concatenate_household_max_with_bounds(avg_diff_dict, project_id_column):
    """
    Concatenate household maximum values and bounds for all columns.

    This function combines the household maximum values and upper bounds
    for all columns in the avg_diff_dict into a single DataFrame.

    Parameters
    ----------
    avg_diff_dict : dict
        A dictionary containing average difference data for each column.
    project_id_column : str
        The name of the column containing project IDs.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing concatenated household maximum values and bounds
        for all columns.

    Notes
    -----
    This function assumes that the 'household_max_with_bounds' key exists in each
    dictionary within avg_diff_dict and contains the columns 'ProjectIdBSV' (or other specified project id column),
    'HuisIdBSV', '{col}_huis_max', and '{col}_upper_bound'.

    """
    first_key = next(iter(avg_diff_dict))
    key_columns = avg_diff_dict[first_key]["household_max_with_bounds"][
        [project_id_column, "HuisIdBSV"]
    ]
    columns = [key_columns]
    for col, data in avg_diff_dict.items():
        max_bound = data["household_max_with_bounds"][
            [col + "_huis_max", col + "_upper_bound"]
        ]
        columns.append(max_bound)
    result_df = pd.concat(columns, axis=1)
    return result_df


def concatenate_avg_diff_columns(avg_diff_dict, project_id_column):
    """
    Concatenate average difference columns for all variables.

    This function combines the average difference columns for all variables
    in the avg_diff_dict into a single DataFrame.

    Parameters
    ----------
    avg_diff_dict : dict
        A dictionary containing average difference data for each column.
    project_id_column : str
        The name of the column containing project IDs.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing concatenated average difference columns
        for all variables.

    Notes
    -----
    This function assumes that the 'avg_diff' key exists in each dictionary
    within avg_diff_dict and contains the columns 'ProjectIdBSV' or specified project_id_column, 'ReadingDate',
    and '{col}_avg'.

    """
    first_key = next(iter(avg_diff_dict))
    key_columns = avg_diff_dict[first_key]["avg_diff"][
        [project_id_column, "ReadingDate"]
    ]
    columns = [key_columns]
    for col, data in avg_diff_dict.items():
        avg_col = data["avg_diff"][col + "_avg"]
        columns.append(avg_col)
    result_df = pd.concat(columns, axis=1)
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
    return pd.read_parquet(os.path.join(etdtransform.options.aggregate_folder_path, "avg_diffs.parquet"))


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

impute_and_normalize_optimized = impute_and_normalize_vectorized
impute_and_normalize = impute_and_normalize_vectorized