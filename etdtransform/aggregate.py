import logging
import os
import re
from typing import Optional

import ibis
import numpy as np
import pandas as pd
from etdmap.data_model import cumulative_columns
from etdmap.index_helpers import read_index, update_meenemen

import etdtransform
from etdtransform.calculated_columns import add_calculated_columns_imputed_data
from etdtransform.impute import process_and_impute

"""
Aggregating the data for a given time interval
Example intervals:
1 hour: '1h'
15 min: '15min'
5 min: '5min'
"""


def read_hh_data(interval="default", metadata_columns=None):
    """
    Read household data from a parquet file and optionally add index columns to.

    Parameters
    ----------
    interval : str, optional
        The time interval of the data to read, by default "default"
    metadata_columns : list, optional
        Additional columns to include from the index, by default None

    Returns
    -------
    pd.DataFrame
        The household data with optional index columns added

    Notes
    -----
    This function reads parquet files from a predefined folder path.
    """
    if not metadata_columns:
        metadata_columns = []
    df = pd.read_parquet(
        os.path.join(etdtransform.options.aggregate_folder_path, f"household_{interval}.parquet"),
    )
    return add_index_columns(df, columns=metadata_columns)


def add_index_columns(df: pd.DataFrame, columns: Optional[list] = None) -> pd.DataFrame:
    """
    Add index columns to the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame
    columns : list, optional
        Additional columns to include from the index, by default None

    Returns
    -------
    pd.DataFrame
        The DataFrame with added index columns

    Notes
    -----
    This function merges the input DataFrame with an index DataFrame based on 'HuisIdBSV' and 'ProjectIdBSV'.
    """
    if columns:
        index_df, index_path = read_index()
        columns_to_select = ["HuisIdBSV", "ProjectIdBSV", *columns]
        columns_to_select = list(set(columns_to_select))
        index_df = index_df[columns_to_select]
        df = df.merge(index_df, on=["HuisIdBSV", "ProjectIdBSV"], how="left")
        return df
    else:
        return df


def aggregate_hh_data_5min():
    """
    Aggregate household data into 5-minute intervals.

    Notes
    -----
    This function reads individual household parquet files, concatenates them,
    and saves the result as a single parquet file.
    """
    logging.info("Starting to aggregate household data.")

    index_df = update_meenemen()

    data_frames = []

    index_df = index_df[index_df["Meenemen"]]

    for _, row in index_df.iterrows():
        huis_id_bsv = row["HuisIdBSV"]
        project_code = row["ProjectIdBSV"]
        file_name = f"household_{huis_id_bsv}_table.parquet"

        file_path = os.path.join(etdtransform.options.mapped_folder_path, file_name)
        household_df = pd.read_parquet(file_path)

        household_df["ProjectIdBSV"] = project_code
        household_df["HuisIdBSV"] = huis_id_bsv

        data_frames.append(household_df)
        logging.info(f"Added {file_name}")

    logging.info("Concatenate all HH dataframes.")
    df = pd.concat(data_frames, ignore_index=True)
    logging.info("Saving HH data to parquet file.")
    df.to_parquet(
        os.path.join(etdtransform.options.aggregate_folder_path, "household_default.parquet"),
        engine="pyarrow",
    )


def impute_hh_data_5min(
    df,
    cum_cols=cumulative_columns,
    sorted=False,
    diffs_calculated=False,
    optimized=False,
):
    """
    Impute missing values in household data and save results.

    Parameters
    ----------
    df : pd.DataFrame, optional
        The input DataFrame, if None it will be read from a file
    cum_cols : list, optional
        List of cumulative columns to process, by default cumulative_columns
    sorted : bool, optional
        Whether the data is already sorted, by default False
    diffs_calculated : bool, optional
        Whether differences are already calculated, by default False
    optimized : bool, optional
        Whether to use optimized processing, by default False

    Returns
    -------
    pd.DataFrame
        The imputed household data

    Notes
    -----
    This function performs imputation, calculates differences, and saves various summary statistics.
    """
    logging.info("Loading HH data from parquet file.")

    if df is None:
        df = read_hh_data(interval="default", metadata_columns=["ProjectIdBSV"])

    # Call the imputation function
    logging.info("Starting the imputation.")

    # df = apply_rolling_iqr_imputation(
    #     df=df,
    #     time_col="ReadingDate",
    #     variable_names=cum_cols,
    #     group_vars=["HuisIdBSV"],
    #     iqr_factor=1.5,
    #     window_weeks=4,
    #     min_valid_ratio=.4
    #     )

    (
        df,
        imputation_summary_house,
        imputation_summary_project,
        imputation_reading_date_stats_df,
    ) = process_and_impute(
        df=df,
        project_id_column="ProjectIdBSV",
        cumulative_columns=cum_cols,
        sorted=sorted,
        diffs_calculated=diffs_calculated,
        optimized=optimized,
    )

    diff_columns = [col + "Diff" for col in cumulative_columns]

    logging.info("Averaging all diffs by project and reading date.")

    aggregated_diff = (
        df.groupby(["ProjectIdBSV", "ReadingDate"])[diff_columns].mean().reset_index()
    )

    logging.info("Saving results")
    # Save the results

    modified_household_dfs = []

    for _huis_code, household_df in df.groupby("HuisIdBSV"):
        for col in cumulative_columns:
            household_df[col + "Original"] = household_df[col]  # rename
            household_df[col] = household_df[col + "Diff"].cumsum()
            household_df[col + "Check"] = (
                household_df[col] - household_df[col + "Original"]
            ).diff()

        modified_household_dfs.append(household_df)

    df = pd.concat(modified_household_dfs, ignore_index=True)

    logging.info("Re-arranging columns.")
    # df = rearrange_model_columns(household_df=df)

    # df.drop(columns=diff_columns)

    if optimized:
        optimized_label = "_optimized"
    else:
        optimized_label = ""

    logging.info("Saving files.")
    df.to_parquet(
        os.path.join(
            etdtransform.options.aggregate_folder_path,
            f"household_imputed{optimized_label}.parquet",
        ),
        engine="pyarrow",
    )

    aggregated_diff.to_parquet(
        os.path.join(
            etdtransform.options.aggregate_folder_path,
            f"household_aggregated_diff{optimized_label}.parquet",
        ),
        engine="pyarrow",
    )
    imputation_summary_house.to_parquet(
        os.path.join(
            etdtransform.options.aggregate_folder_path,
            f"impute_summary_household{optimized_label}.parquet",
        ),
        engine="pyarrow",
    )
    imputation_summary_project.to_parquet(
        os.path.join(
            etdtransform.options.aggregate_folder_path,
            f"impute_summary_project{optimized_label}.parquet",
        ),
        engine="pyarrow",
    )

    if imputation_reading_date_stats_df:
        imputation_reading_date_stats_df.to_parquet(
            os.path.join(
                etdtransform.options.aggregate_folder_path,
                f"impute_summary_reading_date{optimized_label}.parquet",
            ),
            engine="pyarrow",
        )

    logging.info("Done")

    return df


def add_calculated_columns_to_hh_data(df):
    """
    Add calculated columns to household data and save the result.

    Parameters
    ----------
    df : pd.DataFrame, optional
        The input DataFrame, if None it will be read from a file

    Returns
    -------
    pd.DataFrame
        The DataFrame with added calculated columns

    Notes
    -----
    This function adds calculated columns to the household data and saves the result as a parquet file.
    """
    logging.info("Loading imputed data from parquet file.")
    if df is None:
        df = read_hh_data(interval="imputed")

    logging.info("Calculating: ")
    df = add_calculated_columns_imputed_data(df)

    logging.info("Saving calculated columns to file: household_calculated.parquet")
    df.to_parquet(
        os.path.join(etdtransform.options.aggregate_folder_path, "household_calculated.parquet"),
        engine="pyarrow",
    )

    return df


def read_aggregate(name, interval):
    """
    Read an aggregate parquet file.

    Parameters
    ----------
    name : str
        The name of the aggregate
    interval : str
        The time interval of the aggregate

    Returns
    -------
    pd.DataFrame
        The aggregate data

    Notes
    -----
    This function reads a parquet file based on the provided name and interval.
    """
    safe_name = re.sub(r"\W+", "_", name.lower())
    return pd.read_parquet(
        os.path.join(etdtransform.options.aggregate_folder_path, f"{safe_name}_{interval}.parquet"),
    )


def get_aggregate_table(name, interval):
    """
    Get an aggregate table as an ibis table.

    Parameters
    ----------
    name : str
        The name of the aggregate
    interval : str
        The time interval of the aggregate

    Returns
    -------
    ibis.Table
        The aggregate data as an ibis table

    Notes
    -----
    This function reads a parquet file and returns it as an ibis table.
    """
    safe_name = re.sub(r"\W+", "_", name.lower())
    parquet_path = os.path.join(
        etdtransform.options.aggregate_folder_path,
        f"{safe_name}_{interval}.parquet",
    )
    return ibis.read_parquet(parquet_path)


def resample_hh_data(df=None, intervals=("60min", "15min", "5min")):
    """
    Resample household data to different time intervals.

    Parameters
    ----------
    df : pd.DataFrame, optional
        The input DataFrame, if None it will be read from a file
    intervals : tuple, optional
        The time intervals to resample to, by default ("60min", "15min", "5min")

    Notes
    -----
    This function resamples household data to specified time intervals and saves the results.
    """
    group_column = ["ProjectIdBSV", "HuisIdBSV"]
    if df is None:
        logging.info("Loading data with calculated columns to resample hh data")
        df = read_hh_data(interval="calculated")
    else:
        logging.warning(
            "If passing a dataframe to resample_hh_data() be sure to use a copy as it may be modified in place.",
        )

    for interval in intervals:
        logging.info(f"-- Starting household resampling with {interval} intervals --")

        if interval == "5min":
            logging.info(
                "-- 5min interval - applying shortcut without transformation --",
            )
            columns_to_copy = [
                "ReadingDate",
                *group_column,
                *list(aggregation_variables.keys()),
            ]

            for _var, config in aggregation_variables.items():
                validator_column = config.get("validator_column")
                if validator_column:
                    columns_to_copy.append(validator_column)

            df = df[columns_to_copy]

            logging.info(
                f"{interval}min interval - removing variables that do not pass filters"
            )
            for var, config in aggregation_variables.items():
                validator_column = config.get("validator_column")
                if validator_column:
                    df.loc[df[validator_column] is False, var] = pd.NA

            logging.info(
                f"-- {interval}-min interval - saving file household_5min.parquet --"
            )
            df.to_parquet(
                os.path.join(etdtransform.options.aggregate_folder_path, "household_5min.parquet"),
                engine="pyarrow",
            )
        else:
            resample_and_save(df, group_column, interval=interval, alt_name="household")


def aggregate_project_data(intervals=("5min", "15min", "60min")):
    """
    Aggregate project data for different time intervals.

    Parameters
    ----------
    intervals : tuple, optional
        The time intervals to aggregate, by default ("5min", "15min", "60min")

    Notes
    -----
    This function aggregates project data for specified time intervals and saves the results.
    """
    group_column = ["ProjectIdBSV"]
    for interval in intervals:
        logging.info(
            f"-- Starting {group_column} aggregation with {interval} intervals --",
        )
        df = read_hh_data(interval=interval)
        aggregate_and_save(df, group_column, interval=interval, alt_name="project")


# def aggregate_weerstation_data(index_df):
#     group_column = ['Weerstation']
#     intervals = ['5min', '15min', '60min']
#     for interval in intervals:
#         logging.info(f'-- Starting {group_column} aggregation with {interval} intervals --')
#         df = read_hh_data(interval = interval, metadata_columns = ['Weerstation'])
#         aggregate_and_save(df, group_column, interval=interval)


def aggregate_and_save(
    df,
    group_column=("ProjectIdBSV"),
    interval="5min",
    alt_name=None,
):
    """
    Aggregate data and save the result.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame
    group_column : tuple, optional
        The column(s) to group by, by default ("ProjectIdBSV")
    interval : str, optional
        The time interval for aggregation, by default "5min"
    alt_name : str, optional
        An alternative name for the output file, by default None

    Notes
    -----
    This function aggregates data, merges with size information, and saves the result as a parquet file.
    """
    df_grouped = df.groupby(["ReadingDate", *list(group_column)])
    df_size = df_grouped.size().reset_index(name="n")
    if alt_name is None:
        alt_name = group_column
    df = aggregate_by_columns(df, group_column=group_column, size=df_size)
    df = df.merge(df_size, on=["ReadingDate", *list(group_column)], how="left")
    safe_name = re.sub(r"\W+", "_", alt_name.lower())
    df.to_parquet(
        os.path.join(etdtransform.options.aggregate_folder_path, f"{safe_name}_{interval}.parquet"),
        engine="pyarrow",
    )


def aggregate_by_columns(df, group_column, size):
    """
    Aggregate data by columns.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame
    group_column : list
        The column(s) to group by
    size : pd.DataFrame
        DataFrame containing size information

    Returns
    -------
    pd.DataFrame
        The aggregated DataFrame

    Notes
    -----
    This function aggregates data for each variable defined in aggregation_variables.
    """
    first = True
    combined_results = None
    for var, config in aggregation_variables.items():
        logging.info(f"In loop for to aggregate by column {var}")

        method = config["aggregate_method"]

        if (
            method == "diff_cumsum"
            and not first
            and var + "Diff" in combined_results.columns
        ):
            result = aggregate_diff_cumsum(
                df,
                var,
                group_column,
                size,
                combined_results=combined_results,
            )
        else:
            result = aggregate_variable(df, var, config, group_column, size)

        if first:
            combined_results = result
            first = False
        else:
            combined_results = combined_results.merge(
                result,
                on=["ReadingDate", *group_column],
                how="outer",
            )

    logging.info(f"Combining aggregated dataset grouped by: {group_column}")
    return combined_results.reset_index()


def aggregate_variable(df_grouped, var, config, group_column, size):
    """
    Aggregate a single variable.

    Parameters
    ----------
    df_grouped : pd.DataFrame
        The grouped DataFrame
    var : str
        The variable to aggregate
    config : dict
        Configuration for the aggregation
    group_column : list
        The column(s) to group by
    size : pd.DataFrame
        DataFrame containing size information

    Returns
    -------
    pd.DataFrame
        The aggregated DataFrame for the variable

    Notes
    -----
    This function aggregates a single variable based on the specified method in the config.
    """
    logging.info(f"{group_column} : column {var}")
    method = config["aggregate_method"]

    # not including validator columns as they are not aggregated in the household data atm
    # validator_column = config.get('validator_column')

    columns_to_select = ["ReadingDate", *group_column, var]

    if method == "diff_cumsum":
        columns_to_select = [*columns_to_select, var + "Diff"]

    # if validator_column:
    #     columns_to_copy.append(validator_column)

    df_copy = df_grouped[columns_to_select]

    # if validator_column:
    #     df_copy.loc[df_copy[validator_column] != True, var] = pd.NA

    if method == "sum":
        return aggregate_sum(df_copy, var, ["ReadingDate", *group_column], size)
    elif method == "max":
        return aggregate_max(df_copy, var, ["ReadingDate", *group_column], size)
    elif method == "avg":
        return aggregate_avg(df_copy, var, ["ReadingDate", *group_column], size)
    elif method == "diff_cumsum":
        # ReadingDate left out here to allow cumsum to proceed per project with pre-sorted rows
        return aggregate_diff_cumsum(df_copy, var, group_column, size)


# would be smarter to do these variables with method diff_sum only after calculating the average Diff columns
def aggregate_diff_cumsum(df, column, group_column, size, combined_results=None):
    """
    Aggregate cumulative sum of differences.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame
    column : str
        The column to aggregate
    group_column : list
        The column(s) to group by
    size : pd.DataFrame
        DataFrame containing size information
    combined_results : pd.DataFrame, optional
        Previously combined results, by default None

    Returns
    -------
    pd.DataFrame
        The aggregated DataFrame

    Notes
    -----
    This function calculates the cumulative sum of differences for the specified column.
    """
    diff_column = column + "Diff"
    logging.info(
        f"Aggregate cumsum of diff column: {group_column} / {column} / {diff_column}",
    )
    if combined_results is None:
        logging.info("Calculating Diff as not included.")
        aggregated = aggregate_avg(
            df,
            diff_column,
            ["ReadingDate", *group_column],
            size,
        )
    else:
        logging.info("Diff precalculated. No need to recalculate. Making a copy.")
        aggregated = combined_results[
            ["ReadingDate", *group_column, diff_column]
        ].copy()
    logging.info(
        f"Transform average diff to calculate cumsum: {group_column} / {column} / {column}Diff",
    )
    aggregated[column] = aggregated.groupby(group_column)[diff_column].transform(
        pd.Series.cumsum,
    )
    logging.info("Add missing values")
    aggregated[aggregated[diff_column].isna()][column] = pd.NA
    logging.info("Drop column")
    aggregated = aggregated.drop(columns=[diff_column])
    logging.info("Finished")
    return aggregated


def aggregate_sum(df, column, group_column, size):
    """
    Aggregate sum of a column.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame
    column : str
        The column to aggregate
    group_column : list
        The column(s) to group by
    size : pd.DataFrame
        DataFrame containing size information

    Returns
    -------
    pd.DataFrame
        The aggregated DataFrame

    Notes
    -----
    This function calculates the sum of the specified column, requiring at least 60% of values to be present.
    """
    logging.info(f"aggregate sum: {group_column} / {column}")
    grouped = df.groupby(group_column)
    aggregated = grouped[column].agg(sum, min_count=size["n"] * 0.6).reset_index()
    return aggregated


def aggregate_max(df, column, group_column, size):
    """
    Aggregate maximum of a column.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame
    column : str
        The column to aggregate
    group_column : list
        The column(s) to group by
    size : pd.DataFrame
        DataFrame containing size information

    Returns
    -------
    pd.DataFrame
        The aggregated DataFrame

    Notes
    -----
    This function calculates the maximum of the specified column, requiring at least 60% of values to be present.
    """
    logging.info(f"aggregate sum: {group_column} / {column}")
    grouped = df.groupby(group_column)
    aggregated = grouped[column].agg(max, min_count=size["n"] * 0.6).reset_index()
    return aggregated


def aggregate_avg(df, column, group_column, size):
    """
    Aggregate average of a column.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame
    column : str
        The column to aggregate
    group_column : list
        The column(s) to group by
    size : pd.DataFrame
        DataFrame containing size information

    Returns
    -------
    pd.DataFrame
        The aggregated DataFrame

    Notes
    -----
    This function calculates the average of the specified column, requiring at least 60% of values to be present.
    """
    logging.info(f"aggregate avg: {group_column} / {column}")

    # Group by the specified column
    grouped = df.groupby(group_column)

    # Aggregate with sum and count
    aggregated = grouped.agg(
        sum_agg=(column, "sum"),
        count_agg=(column, "count"),
    ).reset_index()

    aggregated[column] = np.where(
        aggregated["count_agg"] >= size["n"] * 0.6,
        aggregated["sum_agg"] / aggregated["count_agg"],
        pd.NA,
    )
    aggregated = aggregated.drop(columns=["sum_agg", "count_agg"])

    return aggregated


def resample_and_save(
    df,
    group_column=("ProjectIdBSV", "HuisIdBSV"),
    interval="5min",
    alt_name=None,
):
    """
    Resample data and save the result.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame
    group_column : tuple, optional
        The column(s) to group by, by default ("ProjectIdBSV", "HuisIdBSV")
    interval : str, optional
        The time interval for resampling, by default "5min"
    alt_name : str, optional
        An alternative name for the output file, by default None

    Notes
    -----
    This function resamples data and saves the result as a parquet file.
    """
    if alt_name is None:
        alt_name = "_".join(group_column)
    df = df.set_index("ReadingDate")
    df = resample_by_columns(df, group_column=group_column, interval=interval)
    df.reset_index(inplace=True)
    safe_name = re.sub(r"\W+", "_", alt_name.lower())
    df.to_parquet(
        os.path.join(etdtransform.options.aggregate_folder_path, f"{safe_name}_{interval}.parquet"),
        engine="pyarrow",
    )


def resample_by_columns(
    df,
    group_column=None,
    interval="15min",
):
    """
    Resample data by columns.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame
    group_column : list, optional
        The column(s) to group by, by default None
    interval : str, optional
        The time interval for resampling, by default "15min"

    Returns
    -------
    pd.DataFrame
        The resampled DataFrame

    Notes
    -----
    This function resamples data for each variable defined in aggregation_variables.
    """
    # resampled_dfs = []
    if group_column is None:
        group_column = ["ProjectIdBSV", "HuisIdBSV"]

    if interval == "5min":
        min_count = 1
    elif interval == "15min":
        min_count = 3
    elif interval == "60min":
        min_count = 12
    elif interval == "6h":
        min_count = 72
    elif interval == "24h":
        min_count = 288
    else:
        raise Exception(f'Unknown interval "{interval}"')

    # Generate the initial dataset with only group_column and ReadingDate
    df_copy = df[group_column].copy()

    combined_results = (
        df_copy.groupby(group_column)
        .resample(interval)
        .size()
        .reset_index()
        .drop(columns=0)
    )

    for var, config in aggregation_variables.items():
        logging.info(f"in loop for {var}")
        result = resample_variable(df, var, config, interval, group_column, min_count)
        combined_results = combined_results.merge(
            result,
            on=["ReadingDate", *group_column],
            how="outer",
        )

    logging.info(f"Combining dataset: {interval} / {group_column}")
    combined_results.reset_index(inplace=True)

    return combined_results


def resample_variable(df, var, config, interval, group_column, min_count):
    """
    Resample a single variable.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame
    var : str
        The variable to resample
    config : dict
        Configuration for the resampling
    interval : str
        The time interval for resampling
    group_column : list
        The column(s) to group by
    min_count : int
        The minimum count required for resampling

    Returns
    -------
    pd.DataFrame
        The resampled DataFrame for the variable

    Notes
    -----
    This function resamples a single variable based on the specified method in the config.
    """
    logging.info(f"{group_column} / {interval}: column {var}")
    method = config["resample_method"]
    validator_column = config.get("validator_column")

    columns_to_copy = [*group_column, var]
    if validator_column:
        columns_to_copy.append(validator_column)
    df_copy = df[columns_to_copy].copy()

    # Filter by validator column if specified
    if validator_column:
        df_copy.loc[df_copy[validator_column] is False, var] = pd.NA

    if method == "sum":
        return resample_sum(df_copy, var, interval, group_column, min_count)
    elif method == "max":
        return resample_max(df_copy, var, interval, group_column, min_count)
    elif method == "avg":
        return resample_avg(df_copy, var, interval, group_column, min_count)


def resample_max(df, column, interval, group_column, min_count):
    """
    Resample maximum of a column.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame
    column : str
        The column to resample
    interval : str
        The time interval for resampling
    group_column : list
        The column(s) to group by
    min_count : int
        The minimum count required for resampling

    Returns
    -------
    pd.DataFrame
        The resampled DataFrame

    Notes
    -----
    This function resamples the maximum of the specified column.
    """
    logging.info(f"resample max: {group_column} / {interval}: {column}")
    resampled = (
        df.groupby(group_column)[column]
        .resample(interval)
        .max(min_count=min_count)
        .reset_index()
    )
    return resampled


def resample_sum(df, column, interval, group_column, min_count):
    """
    Resample sum of a column.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame
    column : str
        The column to resample
    interval : str
        The time interval for resampling
    group_column : list
        The column(s) to group by
    min_count : int
        The minimum count required for resampling

    Returns
    -------
    pd.DataFrame
        The resampled DataFrame

    Notes
    -----
    This function resamples the sum of the specified column.
    """
    logging.info(f"resample sum: {group_column} / {interval}: {column}")
    resampled = (
        df.groupby(group_column)[column]
        .resample(interval)
        .sum(min_count=min_count)
        .reset_index()
    )
    # resampled = df.groupby(group_column)[column].resample(interval).apply(
    #     lambda x: pd.NA if x.isnull().any() else x.sum()
    # ).reset_index()
    # resampled = resampled.groupby('ReadingDate')[column].apply(
    #     lambda x: pd.NA if x.isnull().any() else x.sum()
    # ).reset_index()
    return resampled


def resample_avg(df, column, interval, group_column, min_count):
    """
    Resample average of a column.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    column : str
        The column to resample.
    interval : str
        The time interval for resampling.
    group_column : list
        The column(s) to group by.
    min_count : int
        The minimum count required for resampling.

    Returns
    -------
    pd.DataFrame
        The resampled DataFrame.

    Notes
    -----
    This function resamples the average of the specified column, requiring at least `min_count` values to be present.
    """
    logging.info(f"resample avg: {group_column} / {interval}: {column}")
    resampled = (
        df.groupby(group_column)
        .resample(interval)[column]
        .agg(["sum", "count"])
        .reset_index()
    )
    resampled[column] = np.where(
        resampled["count"] >= min_count,
        resampled["sum"] / resampled["count"],
        pd.NA,
    )
    resampled = resampled.drop(columns=["sum", "count"])
    # resampled = df.groupby(group_column)[column].resample(interval).apply(
    #     lambda x: pd.NA if x.isnull().any() else x.mean()
    # ).reset_index()
    # resampled = resampled.groupby('ReadingDate')[column].apply(
    #     lambda x: pd.NA if x.isnull().any() else x.mean()
    # ).reset_index()
    return resampled


# List of variables with their corresponding aggregation methods - lines marked with ## need a check of the methods - consider for some using 'last_value' for instantaneous variables

aggregation_variables = {
    "ElektriciteitNetgebruikHoogDiff": {
        "resample_method": "sum",
        "aggregate_method": "avg",
    },
    #'ElektriciteitNetgebruikHoog': {'resample_method': 'max', 'aggregate_method': 'diff_cumsum'},
    "ElektriciteitNetgebruikLaagDiff": {
        "resample_method": "sum",
        "aggregate_method": "avg",
    },
    #'ElektriciteitNetgebruikLaag': {'resample_method': 'max', 'aggregate_method': 'diff_cumsum'},
    "ElektriciteitTerugleveringHoogDiff": {
        "resample_method": "sum",
        "aggregate_method": "avg",
    },
    #'ElektriciteitTerugleveringHoog': {'resample_method': 'max', 'aggregate_method': 'diff_cumsum'},
    "ElektriciteitTerugleveringLaagDiff": {
        "resample_method": "sum",
        "aggregate_method": "avg",
    },
    #'ElektriciteitTerugleveringLaag': {'resample_method': 'max', 'aggregate_method': 'diff_cumsum'},
    ## 'ElektriciteitVermogen': {'resample_method': 'avg', 'aggregate_method': 'avg', 'validator_column': 'validate_elektriciteit_vermogen'},
    ## 'Gasgebruik': {'resample_method': 'max', 'aggregate_method': 'diff_cumsum'},
    "ElektriciteitsgebruikWTWDiff": {
        "resample_method": "sum",
        "aggregate_method": "avg",
    },
    #'ElektriciteitsgebruikWTW': {'resample_method': 'max', 'aggregate_method': 'diff_cumsum'},
    "ElektriciteitsgebruikWarmtepompDiff": {
        "resample_method": "sum",
        "aggregate_method": "avg",
    },
    #'ElektriciteitsgebruikWarmtepomp': {'resample_method': 'max', 'aggregate_method': 'diff_cumsum'},
    "ElektriciteitsgebruikBoosterDiff": {
        "resample_method": "sum",
        "aggregate_method": "avg",
    },
    #'ElektriciteitsgebruikBooster': {'resample_method': 'max', 'aggregate_method': 'diff_cumsum'},
    "ElektriciteitsgebruikBoilervatDiff": {
        "resample_method": "sum",
        "aggregate_method": "avg",
    },
    #'ElektriciteitsgebruikBoilervat': {'resample_method': 'max', 'aggregate_method': 'diff_cumsum'},
    "ElektriciteitsgebruikRadiatorDiff": {
        "resample_method": "sum",
        "aggregate_method": "avg",
    },
    #'ElektriciteitsgebruikRadiator': {'resample_method': 'max', 'aggregate_method': 'diff_cumsum'},
    ## 'TemperatuurWarmTapwater': {'resample_method': 'avg', 'aggregate_method': 'avg', 'validator_column': 'validate_temperatuur_warm_tapwater'},
    ## 'TemperatuurWoonkamer': {'resample_method': 'avg', 'aggregate_method': 'avg', 'validator_column': 'validate_temperatuur_woonkamer'},
    ## 'TemperatuurSetpointWoonkamer': {'resample_method': 'avg', 'aggregate_method': 'avg', 'validator_column': 'validate_temperatuur_setpoint_woonkamer'},
    ## 'WarmteproductieWarmtepomp': {'resample_method': 'max', 'aggregate_method': 'avg'},
    ## 'WatergebruikWarmTapwater': {'resample_method': 'max', 'aggregate_method': 'avg'},
    ## 'Zon-opwekMomentaan': {'resample_method': 'avg', 'aggregate_method': 'avg', 'validator_column': 'validate_zon_opwek_momentaan'},
    "ZonopwekBruto": {
        "resample_method": "sum", 
        "aggregate_method": "avg"
        },
    #'Zon-opwekTotaal': {'resample_method': 'max', 'aggregate_method': 'diff_cumsum'},
    ## 'CO2': {'resample_method': 'avg', 'aggregate_method': 'avg', 'validator_column': 'validate_co2'},
    ## 'Luchtvochtigheid': {'resample_method': 'avg', 'aggregate_method': 'avg', 'validator_column': 'validate_luchtvochtigheid'},
    ## 'Ventilatiedebiet': {'resample_method': 'avg', 'aggregate_method': 'avg', 'validator_column': 'validate_ventilatiedebiet'},
    "TerugleveringTotaalNetto": {
        "resample_method": "sum", 
        "aggregate_method": "avg"
        },
    "ElektriciteitsgebruikTotaalNetto": {
        "resample_method": "sum",
        "aggregate_method": "avg",
    },
    "Netuitwisseling": {"resample_method": "sum", "aggregate_method": "avg"},
    "ElektriciteitsgebruikTotaalWarmtepomp": {
        "resample_method": "sum",
        "aggregate_method": "avg",
    },
    "ElektriciteitsgebruikTotaalGebouwgebonden": {
        "resample_method": "sum",
        "aggregate_method": "avg",
    },
    "ElektriciteitsgebruikTotaalHuishoudelijk": {
        "resample_method": "sum",
        "aggregate_method": "avg",
    },
    "Zelfgebruik": {"resample_method": "sum", "aggregate_method": "avg"},
    "ElektriciteitsgebruikTotaalBruto": {
        "resample_method": "sum",
        "aggregate_method": "avg",
    },
}
