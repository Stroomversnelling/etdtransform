import logging
import math
from datetime import datetime, timedelta

import ibis
import pandas as pd


def add_calculated_columns_imputed_data(df, fillna = True):
    """
    Add calculated columns to the input DataFrame based on existing data.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing energy usage and production data.
    fillna : bool, optional
        Whether to fill missing values with 0 before performing calculations.
        Default is True.

    Returns
    -------
    pd.DataFrame
        The modified DataFrame with additional calculated columns.

    Notes
    -----
    - This function assumes that the input DataFrame contains the necessary
      columns such as 'ElektriciteitTerugleveringLaagDiff',
      'ElektriciteitTerugleveringHoogDiff', 'ElektriciteitNetgebruikLaagDiff',
      'ElektriciteitNetgebruikHoogDiff', 'ElektriciteitsgebruikWarmtepompDiff',
      'ElektriciteitsgebruikBoosterDiff', 'ElektriciteitsgebruikBoilervatDiff',
      'ElektriciteitsgebruikWTWDiff', 'ElektriciteitsgebruikRadiatorDiff',
      and 'Zon-opwekTotaalDiff'.
    - The function assumes that missing values can be treated a 0s, typically after
      data cleaning and imputation.
    - The function fills missing values in each column with 0 before performing
      calculations to ensure that the operations do not fail due to missing data.
    """

    if fillna:

        logging.info("Calculating TerugleveringTotaalNetto")
        df["TerugleveringTotaalNetto"] = df["ElektriciteitTerugleveringLaagDiff"].fillna(
            0,
        ) + df["ElektriciteitTerugleveringHoogDiff"].fillna(0)
        logging.info("Calculating ElektriciteitsgebruikTotaalNetto")

        df["ElektriciteitsgebruikTotaalNetto"] = df[
            "ElektriciteitNetgebruikLaagDiff"
        ].fillna(0) + df["ElektriciteitNetgebruikHoogDiff"].fillna(0)

        logging.info("Calculating Netuitwisseling")
        df["Netuitwisseling"] = df["ElektriciteitsgebruikTotaalNetto"].fillna(0) - df[
            "TerugleveringTotaalNetto"
        ].fillna(0)

        logging.info("Calculating ElektriciteitsgebruikTotaalWarmtepomp")
        df["ElektriciteitsgebruikTotaalWarmtepomp"] = df[
            "ElektriciteitsgebruikWarmtepompDiff"
        ].fillna(0) + df["ElektriciteitsgebruikBoosterDiff"].fillna(0)

        logging.info("Calculating ElektriciteitsgebruikTotaalGebouwgebonden")
        df["ElektriciteitsgebruikTotaalGebouwgebonden"] = (
            df["ElektriciteitsgebruikTotaalWarmtepomp"].fillna(0)
            + df["ElektriciteitsgebruikBoilervatDiff"].fillna(0)
            + df["ElektriciteitsgebruikWTWDiff"].fillna(0)
            + df["ElektriciteitsgebruikRadiatorDiff"].fillna(0)
        )

        logging.info("Renaming Zon-opwekTotaalDiff to ZonopwekBruto")
        df.rename(columns={"Zon-opwekTotaalDiff": "ZonopwekBruto"}, inplace=True)

        logging.info("Calculating ElektriciteitsgebruikTotaalHuishoudelijk")
        df["ElektriciteitsgebruikTotaalHuishoudelijk"] = (
            df["Netuitwisseling"].fillna(0)
            + df["ZonopwekBruto"].fillna(0)
            - df["ElektriciteitsgebruikTotaalGebouwgebonden"].fillna(0)
        )

        logging.info("Calculating Zelfgebruik")
        df["Zelfgebruik"] = df["ZonopwekBruto"].fillna(0) - df[
            "TerugleveringTotaalNetto"
        ].fillna(0)

        logging.info("Calculating ElektriciteitsgebruikTotaalBruto")
        df["ElektriciteitsgebruikTotaalBruto"] = df[
            "ElektriciteitsgebruikTotaalNetto"
        ].fillna(0) + df["Zelfgebruik"].fillna(0)
    else:
        logging.info("Calculating TerugleveringTotaalNetto")
        df["TerugleveringTotaalNetto"] = df["ElektriciteitTerugleveringLaagDiff"]
        + df["ElektriciteitTerugleveringHoogDiff"]
        logging.info("Calculating ElektriciteitsgebruikTotaalNetto")

        df["ElektriciteitsgebruikTotaalNetto"] = df[
            "ElektriciteitNetgebruikLaagDiff"
        ] + df["ElektriciteitNetgebruikHoogDiff"]

        logging.info("Calculating Netuitwisseling")
        df["Netuitwisseling"] = df["ElektriciteitsgebruikTotaalNetto"] - df[
            "TerugleveringTotaalNetto"
        ]

        logging.info("Calculating ElektriciteitsgebruikTotaalWarmtepomp")
        df["ElektriciteitsgebruikTotaalWarmtepomp"] = df[
            "ElektriciteitsgebruikWarmtepompDiff"
        ] + df["ElektriciteitsgebruikBoosterDiff"]

        logging.info("Calculating ElektriciteitsgebruikTotaalGebouwgebonden")
        df["ElektriciteitsgebruikTotaalGebouwgebonden"] = (
            df["ElektriciteitsgebruikTotaalWarmtepomp"]
            + df["ElektriciteitsgebruikBoilervatDiff"]
            + df["ElektriciteitsgebruikWTWDiff"]
            + df["ElektriciteitsgebruikRadiatorDiff"]
        )

        logging.info("Renaming Zon-opwekTotaalDiff to ZonopwekBruto")
        df.rename(columns={"Zon-opwekTotaalDiff": "ZonopwekBruto"}, inplace=True)

        logging.info("Calculating ElektriciteitsgebruikTotaalHuishoudelijk")
        df["ElektriciteitsgebruikTotaalHuishoudelijk"] = (
            df["Netuitwisseling"]
            + df["ZonopwekBruto"]
            - df["ElektriciteitsgebruikTotaalGebouwgebonden"]
        )

        logging.info("Calculating Zelfgebruik")
        df["Zelfgebruik"] = df["ZonopwekBruto"] - df[
            "TerugleveringTotaalNetto"
        ]

        logging.info("Calculating ElektriciteitsgebruikTotaalBruto")
        df["ElektriciteitsgebruikTotaalBruto"] = df[
            "ElektriciteitsgebruikTotaalNetto"
        ] + df["Zelfgebruik"]


    return df


# This calculation is done on a df with resampled data - it results in net_impact
# Average of 'ElektriciteitsgebruikTotaalNetto' and
# Average of 'ElektriciteitsgebruikTotaalWarmtepomp' as
# a proxy for the the coldest two weeks
# Average 'ZonopwekBruto' for the sunniest one week (days = 7)
# This is a rolling average so does not handle the edges of the year perfectly
def add_rolling_avg(
    group,
    var="ElektriciteitsgebruikTotaalNetto",
    days=14,
    avg_var="RollingAverage",
):
    """
    Add a rolling average column to each group in the DataFrame.

    Parameters
    ----------
    group : pd.DataFrame
        The DataFrame group on which to perform the operation.
        This should be a subset of a larger DataFrame that has been grouped by some key,
        for example, `df.groupby('some_column').apply(add_rolling_avg)`.
    var : str, optional
        The name of the column in 'group' for which the rolling average will be calculated.
        Default is 'ElektriciteitsgebruikTotaalNetto'.
    days : int, optional
        The number of days over which to calculate the rolling average.
        Default is 14 days.
    avg_var : str, optional
        The name of the new column in 'group' that will store the calculated rolling average values.
        Default is 'RollingAverage'.
    Returns
    -------
    pd.DataFrame
        A DataFrame with an additional column containing the rolling averages.

    Notes
    -----
    - This function assumes that the 'ReadingDate' column exists in the input DataFrame and
      is sorted in ascending order. The 'ReadingDate' column should be of datetime type.
    - The function calculates a rolling average using a window size determined by the number of days
      specified and the frequency of the data points in the group. It uses a forward-looking window,
      meaning that for each date, it computes the average of the next `days` worth of data points.
    - To handle cases where there are missing dates or irregular sampling intervals, the function
      first calculates the time difference between consecutive readings to determine how many timesteps
      correspond to the specified number of days. It then uses this calculated window size for the
      rolling average computation.
    - The `min_periods` parameter in the rolling method is set to half of the window size,
      ensuring that partial windows at the beginning and end of the group are still computed if they have
      sufficient data points.

    """
    group = group.sort_values("ReadingDate")

    timedelta = (
        group["ReadingDate"].iloc[1] - group["ReadingDate"].iloc[0]
    ).total_seconds()
    needed_timesteps = int(pd.Timedelta(days=days).total_seconds() / timedelta)

    group[avg_var] = (
        group[var]
        .rolling(window=needed_timesteps, min_periods=int(needed_timesteps / 2))
        .mean()
    )
    return group


# Define the get_highest_avg_period function
def get_highest_avg_period(group, avg_var="RollingAverage", days=14):
    """
    Retrieve the start time, end time, and highest rolling average for each group in the DataFrame.

    Parameters
    ----------
    group : pd.DataFrame
        The DataFrame group on which to perform the operation.
        This should be a subset of a larger DataFrame that has been grouped by some key,
        for example, `df.groupby('some_column').apply(get_highest_avg_period)`.
    avg_var : str, optional
        The name of the column in 'group' containing the rolling averages.
        Default is 'RollingAverage'.
    days : int, optional
        The number of days over which the rolling average was calculated.
        Default is 14 days.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns for each group variable (if applicable), start time,
        end time, and highest rolling average.

    Notes
    -----
    - This function assumes that the 'ReadingDate' column exists in the input DataFrame and
      is sorted in ascending order. The 'ReadingDate' column should be of datetime type.
    - The function identifies rows with the highest value in the `avg_var` column and calculates
      the corresponding start and end times based on the number of days specified.
    - To handle cases where there are missing dates or irregular sampling intervals, the function
      first calculates the time difference between consecutive readings to determine how many timesteps
      correspond to the specified number of days. It then uses this calculated window size for
      determining the start and end times.
    - If the calculated start index is out of bounds (greater than or equal to the length of the group),
      it sets the start index to the last valid index in the group.

    """
    results = []

    highest_rolling_avg = group[avg_var].max()
    highest_rolling_avg_rows = group[group[avg_var] == highest_rolling_avg]

    timedelta = (
        group["ReadingDate"].iloc[1] - group["ReadingDate"].iloc[0]
    ).total_seconds()
    needed_timesteps = int(pd.Timedelta(days=days).total_seconds() / timedelta)

    for idx in highest_rolling_avg_rows.index:
        # start_time = group.loc[idx, 'ReadingDate']
        # end_time_index = group.index.get_loc(idx) + needed_timesteps - 1

        # if end_time_index >= len(group):
        #    end_time_index = len(group) - 1

        # end_time = group.iloc[end_time_index]['ReadingDate']

        end_time = group.loc[idx, "ReadingDate"]
        start_time_index = group.index.get_loc(idx) - needed_timesteps + 1

        if start_time_index >= len(group):
            start_time_index = len(group) - 1

        start_time = group.iloc[start_time_index]["ReadingDate"]
        # MJW: changed start time to end time and worked back from there. The rolling average was looking backwards, so the plot also needs to do that.

        result = {
            "StartTime": start_time,
            "EndTime": end_time,
            "HighestRollingAverage": highest_rolling_avg,
        }

        # Extract the group variable name and value
        if group.index.name is None:
            result["Group"] = group.name
        else:
            group_var_name = group.index.name
            result[group_var_name] = group.name

        results.append(result)

    return pd.DataFrame(results)


def gelijktijdigheid(df, df_5min, rolling_average="RollingAverage", group_var=None):
    """
    Calculate the ratio of the highest rolling average of the given rolling average column between daily and 5-minute interval data.

    Parameters
    ----------
    df : pd.DataFrame
        The input daily DataFrame containing the data.
    df_5min : pd.DataFrame
        The input 5-minute interval DataFrame containing the data.
    rolling_average : str, optional
        The column name of the rolling average. Default is "RollingAverage".
    group_var : str, optional
        The column name to group by. Default is None.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the group variable (if applicable) and the ratio of the highest rolling average value.
    """
    def highest_avg(group):
        return group[rolling_average].max()

    if group_var:
        daily_max = df.groupby(group_var).apply(highest_avg).rename("HighestDailyAvg")
        min_max = df_5min.groupby(group_var).apply(highest_avg).rename("Highest5MinAvg")
        result = pd.concat([daily_max, min_max], axis=1)
        result["Ratio"] = result["HighestDailyAvg"] / result["Highest5MinAvg"]
        result.reset_index(inplace=True)
    else:
        highest_daily_avg = highest_avg(df)
        highest_5min_avg = highest_avg(df_5min)
        result = pd.DataFrame(
            {
                "HighestDailyAvg": [highest_daily_avg],
                "Highest5MinAvg": [highest_5min_avg],
                "Ratio": [highest_daily_avg / highest_5min_avg],
            },
        )

    return result


def get_lowest_avg_period(group, avg_var="RollingAvg_Temperatuur", days=14):
    """
    Retrieve the start time, end time, and lowest rolling average for each group in the DataFrame.

    Parameters
    ----------
    group : pd.DataFrame
        The DataFrame group on which to perform the operation.
        This should be a subset of a larger DataFrame that has been grouped by some key,
        for example, `df.groupby('some_column').apply(get_lowest_avg_period)`.
    avg_var : str, optional
        The name of the column in 'group' containing the rolling averages.
        Default is 'RollingAvg_Temperatuur'.
    days : int, optional
        The number of days over which the rolling average was calculated.
        Default is 14 days.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns for each group variable (if applicable), start time,
        end time, and lowest rolling average.

    Notes
    -----
    - This function assumes that the 'ReadingDate' column exists in the input DataFrame and
      is sorted in ascending order. The 'ReadingDate' column should be of datetime type.
    - The function identifies rows with the lowest value in the `avg_var` column and calculates
      the corresponding start and end times based on the number of days specified.
    - To handle cases where there are missing dates or irregular sampling intervals, the function
      first calculates the time difference between consecutive readings to determine how many timesteps
      correspond to the specified number of days. It then uses this calculated window size for
      determining the start and end times.
    - If the calculated start index is out of bounds (greater than or equal to the length of the group),
      it sets the start index to the last valid index in the group.

    """
    results = []

    lowest_rolling_avg = group[avg_var].min()
    lowest_rolling_avg_rows = group[group[avg_var] == lowest_rolling_avg]

    timedelta = (
        group["ReadingDate"].iloc[1] - group["ReadingDate"].iloc[0]
    ).total_seconds()
    needed_timesteps = int(pd.Timedelta(days=days).total_seconds() / timedelta)

    for idx in lowest_rolling_avg_rows.index:
        # start_time = group.loc[idx, 'ReadingDate']
        # end_time_index = group.index.get_loc(idx) + needed_timesteps - 1

        # if end_time_index >= len(group):
        #    end_time_index = len(group) - 1

        # end_time = group.iloc[end_time_index]['ReadingDate']

        end_time = group.loc[idx, "ReadingDate"]
        start_time_index = group.index.get_loc(idx) - needed_timesteps + 1

        if start_time_index >= len(group):
            start_time_index = len(group) - 1

        start_time = group.iloc[start_time_index]["ReadingDate"]
        # MJW: changed start time to end time and worked back from there. The rolling average was looking backwards, so the plot also needs to do that.

        result = {
            "StartTime": start_time,
            "EndTime": end_time,
            "LowestRollingAverage": lowest_rolling_avg,
        }

        # Extract the group variable name and value
        if group.index.name is None:
            result["Group"] = group.name
        else:
            group_var_name = group.index.name
            result[group_var_name] = group.name

        results.append(result)

    return pd.DataFrame(results)


# Find the start and end of the coldest two-week period for each KNMI station
def mark_coldest_two_weeks(group, avg_var="TemperatuurRA", days=14):
    """
    Marks the coldest two-week period for each group in the DataFrame.

    Parameters
    ----------
    group : pd.DataFrame
        The DataFrame group.
    avg_var : str
        The variable containing the rolling averages.
    days : int
        The number of days over which the rolling average was calculated.

    Returns
    -------
    pd.Series
        A boolean Series indicating whether each row is within the coldest two-week period.
    """
    original_index = group.index

    if ("YYYYMMDD" in group.columns) and ("HH" in group.columns):
        group = group.sort_values(by=["YYYYMMDD", "HH"])
    elif "ReadingDate" in group.columns:
        group = group.sort_values(by=["ReadingDate"])
    else:
        raise ValueError(
            "Required columns missing in DataFrame."
            "Required at least 'YYYYMMDD' and 'HH' or 'ReadingDate'"
            )
    coldest_period = pd.Series(False, index=group.index)
    lowest_rolling_avg = group[avg_var].min()
    lowest_rolling_avg_rows = group[group[avg_var] == lowest_rolling_avg]

    # needed_timesteps = 24 * days
    timedelta = (
        group["ReadingDate"].iloc[1] - group["ReadingDate"].iloc[0]
    ).total_seconds()
    needed_timesteps = int(pd.Timedelta(days=days).total_seconds() / timedelta)

    # Rolling average looks backwards, so the plot also needs to do that.
    for idx in lowest_rolling_avg_rows.index:
        end_idx = group.index.get_loc(idx)
        start_idx = end_idx - needed_timesteps

        if start_idx > len(group):
            start_idx = len(group)

        coldest_period.iloc[start_idx:end_idx] = True

    coldest_period = coldest_period.reindex(original_index)

    return coldest_period

# Find the start and end of the one week period with the highest peak for each project
def mark_highest_peak(group, var="ElektriciteitsgebruikTotaalNetto", days=6):
    """
    Marks the one-week period for each group in the DataFrame around the highest peak.

    Parameters
    ----------
    group : pd.DataFrame
        The DataFrame group.
    var : str
        The variable containing the peak energy use.
    days : int
        The number of days to include.

    Returns
    -------
    pd.Series
        A boolean Series indicating whether each row is within the one-week period around the highest peak.
    """
    original_index = group.index

    group = group.sort_values(by=["ReadingDate"])

    peak_period = pd.Series(False, index=group.index)
    highest_peak = group[var].max()
    highest_peak_rows = group[group[var] == highest_peak]

    timedelta = (
        group["ReadingDate"].iloc[1] - group["ReadingDate"].iloc[0]
    ).total_seconds()
    needed_timesteps = int(pd.Timedelta(days=days).total_seconds() / timedelta)

    for idx in highest_peak_rows.index:
        start_idx = group.index.get_loc(idx) - math.ceil(needed_timesteps / 2)
        end_idx = start_idx + needed_timesteps

        if end_idx > len(group):
            end_idx = len(group)

        peak_period.iloc[start_idx:end_idx] = True

    peak_period = peak_period.reindex(original_index)

    return peak_period


def switch_multiplier(interval_choice):
    """
    Returns the multiplier for the switches in the calculation of the calculated columns.

    Parameters
    ----------
    interval_choice : str
        The interval over which the data is aggregated.

    Returns
    -------
    int
        The multiplier to use in unit conversions.
    """

    if interval_choice == "5min":
        return 12
    elif interval_choice == "15min":
        return 4
    elif interval_choice == "60min":
        return 1
    elif interval_choice == "6h":
        return 1 / 6
    elif interval_choice == "24h":
        return 1 / 24
    else:
        raise Exception("Unknown interval")

intervals = ["5min", "15min", "60min", "24h"]


def add_normalized_datetime(
    x,
    reference_date=pd.Timestamp("2023-01-02"),
    datetime_column="ReadingDate",
):
    """
    Adds a normalized datetime column to the DataFrame or Ibis Table.
    Used to do analyses that depend on the time of day rather than date.

    Parameters
    ----------
    x : pd.DataFrame or ibis.expr.types.TableExpr
        The DataFrame or Table.
    reference_date : datetime.datetime, optional
        The date used as reference for the normalization. Default is '2023-01-02'.
    datetime_column : str, optional
        The name of the column containing the datetime. Default is 'ReadingDate'.
    Returns
    -------
    pd.DataFrame or ibis.expr.types.TableExpr
        The DataFrame or Table with a new column 'normalized_datetime'.
    """
    if isinstance(x, pd.DataFrame):
        x.loc[:, "time_of_day"] = x[datetime_column].dt.time.copy()
        x.loc[:,"day_of_week"] = x[datetime_column].dt.dayofweek.copy()  # Monday=0, Sunday=6

        x.loc[:,"normalized_datetime"] = x["time_of_day"].apply(
            lambda t: datetime.combine(reference_date + pd.Timedelta(days=0), t),
        ).copy() + pd.to_timedelta(x["day_of_week"], unit="D")

        return x
    elif isinstance(x, ibis.expr.types.Expr):
        # Ibis logic
        reference_date_literal = ibis.literal(reference_date)

        # Extract time of day and day of week
        time_of_day = x[datetime_column].time()
        day_of_week = x[datetime_column].day_of_week.index()

        combined_date = reference_date_literal + (
            day_of_week.cast("int32") * ibis.interval(days=1)
        )

        time_of_day_to_add = x[datetime_column] - x[datetime_column].date().cast(
            "timestamp",
        )
        normalized_datetime = combined_date + time_of_day_to_add

        return x.mutate(
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            normalized_datetime=normalized_datetime,
        )
    else:
        raise TypeError("Input must be either a pandas DataFrame or an Ibis TableExpr")