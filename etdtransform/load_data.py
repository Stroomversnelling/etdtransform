import logging
import os
from typing import List, Optional

import ibis
import ibis.selectors as s
import pandas as pd
from ibis import _

import etdtransform
from etdtransform.aggregate import (
    get_aggregate_table,
    read_aggregate,
)
from etdtransform.calculated_columns import intervals, mark_coldest_two_weeks
from etdtransform.knmi import (
    get_project_weather_station_data,
    get_weather_data,
    weather_columns,
)


def get_household_tables(include_weather: bool = True) -> dict[str, ibis.Expr]:
    """
    Reads household data tables for different intervals and joins them with an index table.
    Optionally integrates weather data.

    Parameters
    ----------
    include_weather : bool, optional
        If True, includes weather data in the returned tables (default is True).

    Returns
    -------
    dict[str, ibis.Expr]
        A dictionary where keys are interval names (e.g., 'hourly', 'daily') and values
        are the corresponding Ibis tables.
    """
    household_tbls = {}

    if include_weather:
        household_tbls["weather"] = get_weather_data_table()
        weather_station_table = get_weather_station_table()

    for interval in intervals:
        household_parquet = os.path.join(
            etdtransform.options.aggregate_folder_path, f"household_{interval}.parquet"
        )
        household_table = ibis.read_parquet(household_parquet)

        # conversions for files that are in the 'old' (pre-package) format.
        if "HuisIdBSV" not in household_table.columns:
            household_table = household_table.rename(HuisIdBSV="HuisCode")

        hh_joined = join_index_table(household_table)

        if include_weather:
            hh_joined = join_weather_data(
                hh_joined,
                weather_station_table=weather_station_table,
                weather_table=household_tbls["weather"],
            )

        household_tbls[interval] = hh_joined

    return household_tbls

def join_index_table(
    tbl: ibis.Expr,
    index_table: Optional[ibis.Expr] = None,
    index_join_columns: List[str] = ["HuisIdBSV", "ProjectIdBSV"],
) -> ibis.Expr:
    """
    Joins a given table with an index table on specified columns.

    Parameters
    ----------
    tbl : ibis.Expr
        The table to join.
    index_table : Optional[ibis.Expr], optional
        The index table. If None, reads from default parquet file (default is None).
    index_join_columns : List[str], optional
        Columns to use for the join (default is ["HuisCode", "ProjectIdBSV"]).

    Returns
    -------
    ibis.Expr
        The table joined with the index table.
    """
    if index_table is None:
        index_table = ibis.read_parquet(
            os.path.join(etdtransform.options.mapped_folder_path, "index.parquet")
        )

    # conversion for old (pre-package version) files:
    if "HuisIdBSV" not in index_table.columns:
        index_table = index_table.rename(HuisIdBSV="HuisCode")

    return tbl.left_join(index_table, index_join_columns)

def get_weather_data_table() -> ibis.Expr:
    """
    Processes and transforms weather data into an Ibis table with additional calculated columns.

    The transformations include:
    - Rolling 14-day averages of temperature and perceived temperature.
    - Identifying the coldest two weeks based on rolling averages.
    - Adding ISO week, day of week, and weekly summary calculations.

    Returns
    -------
    ibis.Expr
        An Ibis table containing transformed weather data with additional calculated columns.
    
    Notes
    -----
    The weather data is grouped by station ('STN') and aggregated weekly.
    """
    weather_data_df = get_weather_data()
    weather_data_df = weather_data_df.sort_values(["STN", "YYYYMMDD", "HH"])
    weather_data_df["TemperatuurRA"] = weather_data_df.groupby("STN")[
        "Temperatuur"
    ].transform(lambda x: x.rolling(window=14 * 24, min_periods=7 * 24).mean())
    weather_data_df["GevoelstemperatuurRA"] = weather_data_df.groupby("STN")[
        "Gevoelstemperatuur"
    ].transform(lambda x: x.rolling(window=14 * 24, min_periods=7 * 24).mean())
    weather_data_df["Koudste2WkTemperatuur"] = (
        weather_data_df.groupby("STN")
        .apply(mark_coldest_two_weeks, avg_var="TemperatuurRA", include_groups=False)
        .reset_index(level=0, drop=True)
    )
    weather_data_df["Koudste2WkGevoelstemperatuur"] = (
        weather_data_df.groupby("STN")
        .apply(
            mark_coldest_two_weeks, avg_var="GevoelstemperatuurRA", include_groups=False
        )
        .reset_index(level=0, drop=True)
    )

    # Read weather data directly into an Ibis table
    weather_data = ibis.memtable(weather_data_df)

    # Add columns for ISO week number and day of the week
    # Get week of the year from column `YYYYMMDD`
    weather_data = weather_data.mutate(
        temp_date_string=(
            weather_data.YYYYMMDD.cast("string").substr(0, 4)
            + "-"
            + weather_data.YYYYMMDD.cast("string").substr(4, 2)
            + "-"
            + weather_data.YYYYMMDD.cast("string").substr(6, 2)
        )
    )

    weather_data = weather_data.mutate(
        date_column=weather_data.temp_date_string.cast("date")
    )

    weather_data = weather_data.mutate(
        datetime_column=(
            weather_data.temp_date_string
            + " "
            + weather_data.HH.cast("string").lpad(2, "0")
            + ":00:00"
        ).cast("timestamp")
    )

    weather_data = weather_data.mutate(
        year=weather_data.date_column.year(),
        week_of_year=weather_data.date_column.week_of_year(),
        day_of_week=weather_data.date_column.day_of_week.index(),  # 0=Monday, 6=Sunday
    )

    # Define a weekly grouping window by station and week
    iso_weekly_grouping = ["STN", "year", "week_of_year"]

    # Calculate weekly averages and count of days in each week
    iso_weekly_window = ibis.window(group_by=iso_weekly_grouping)

    weather_data = weather_data.mutate(
        TemperatuurISOWk=weather_data["Temperatuur"].mean().over(iso_weekly_window),
        GevoelstemperatuurISOWk=weather_data["Gevoelstemperatuur"]
        .mean()
        .over(iso_weekly_window),
        days_in_week=(
            weather_data["date_column"].count().over(iso_weekly_window) / 24.0
        ).cast("int"),
    )

    weekly_summary = weather_data[
        "STN",
        "year",
        "week_of_year",
        "TemperatuurISOWk",
        "GevoelstemperatuurISOWk",
        "days_in_week",
    ].distinct()

    # Filter for complete weeks with 7 days, if necessary
    # weekly_summary = weekly_summary.filter(weekly_summary['days_in_week'] == 7)

    temp_window = ibis.window(group_by=["STN", "year"], order_by="TemperatuurISOWk")
    feeling_window = ibis.window(
        group_by=["STN", "year"], order_by="GevoelstemperatuurISOWk"
    )

    # Rank weeks within each station by average temperature and perceived temperature
    weekly_summary_rank = weekly_summary.mutate(
        rank_temperatuur=weekly_summary["TemperatuurISOWk"].rank().over(temp_window),
        rank_gevoelstemperatuur=weekly_summary["GevoelstemperatuurISOWk"]
        .rank()
        .over(feeling_window),
    )

    weekly_summary_rank = weekly_summary_rank.mutate(
        rank_row_temperatuur=ibis.row_number().over(temp_window)
    ).mutate(
        Koudste2ISOWkTemperatuur=(_.rank_row_temperatuur < 2)
        & ~_.rank_row_temperatuur.isnull()
    )  # .drop('rank_row_temperatuur')

    weekly_summary_rank = weekly_summary_rank.mutate(
        rank_row_gevoelstemperatuur=ibis.row_number().over(feeling_window)
    ).mutate(
        Koudste2ISOWkGevoelstemperatuur=(_.rank_row_gevoelstemperatuur < 2)
        & ~_.rank_row_gevoelstemperatuur.isnull()
    )

    weekly_summary_rank = weekly_summary_rank.select(
        [
            "STN",
            "year",
            "week_of_year",
            "Koudste2ISOWkTemperatuur",
            "Koudste2ISOWkGevoelstemperatuur",
        ]
    )

    weather_data = weather_data.left_join(
        weekly_summary_rank, ["STN", "year", "week_of_year"]
    ).select(~s.contains("_right"))

    # Add the transformed weather data as an Ibis table
    return weather_data

def get_weather_station_table() -> ibis.Expr:
    """
    Retrieves weather station data as an Ibis table.

    Returns
    -------
    ibis.Expr
        An Ibis table containing weather station data.
    """
    project_weather_station_df = get_project_weather_station_data()
    return ibis.memtable(project_weather_station_df)

def join_weather_data(tbl: ibis.Expr, weather_station_table: Optional[ibis.Expr] = None, weather_table: Optional[ibis.Expr] = None) -> ibis.Expr:
    """
    Joins weather data with a given table.

    Parameters
    ----------
    tbl : ibis.Expr
        The table to join with weather data.
    weather_station_table : Optional[ibis.Expr], optional
        The weather station mapping table (default is None, meaning it will be retrieved).
    weather_table : Optional[ibis.Expr], optional
        The table containing weather data (default is None, meaning it will be retrieved).

    Returns
    -------
    ibis.Expr
        The input table with joined weather data.
    """
    if weather_table is None:
        weather_table = get_weather_data_table()
    if weather_station_table is None:
        weather_station_table = get_weather_station_table()

    # Join project table with project weather station data

    if 'Weerstation' in tbl.columns:
        tbl = tbl.drop('Weerstation')
    if 'STN' in tbl.columns:
        tbl = tbl.drop('STN')

    tbl = tbl.left_join(
        weather_station_table, tbl.ProjectIdBSV == weather_station_table.ProjectIdBSV
    ).select(
        [
            tbl,
            "Weerstation",
            "STN",
        ]  # Select all columns from tbl plus Weerstation and STN
    )

    # Extract hour (HH) and date (YYYYMMDD) for the join
    if 'YYYYMMDD' not in tbl.columns:
        tbl = tbl.mutate(
            HH=tbl.ReadingDate.hour() + 1,
            YYYYMMDD=tbl.ReadingDate.strftime("%Y%m%d").cast("int"),
        )

    # Join with weather data
    tbl = tbl.left_join(
        weather_table,
        [
            tbl.STN == weather_table.STN,
            tbl.YYYYMMDD == weather_table.YYYYMMDD,
            tbl.HH == weather_table.HH,
        ],
    )

    return tbl


def get_project_tables(include_weather=True) -> dict[str, ibis.Expr]:
    """
    Retrieves aggregate project data tables from Parquet files, integrates weather data,
    and returns a dictionary of Ibis tables for each interval.

    Returns:
        dict[str, ibis.Expr]: A dictionary where keys are intervals (and additional metadata like
        'project_weather') and values are the corresponding Ibis tables with integrated weather data.
    """
    tables = {}

    if include_weather:
        tables["weather"] = get_weather_data_table()
        weather_station_table = get_weather_station_table()
        tables["project_weather"] = weather_station_table

    # Load interval tables with weather data
    for interval in intervals:
        # Read Parquet file for each interval as Ibis table
        project_table = get_aggregate_table(name="project", interval=interval)
        project_table = project_table.filter(project_table.ProjectIdBSV != 6.0)

        if include_weather:
            project_table = join_weather_data(
                project_table,
                weather_station_table=weather_station_table,
                weather_table=tables["weather"],
            )

        tables[interval] = project_table

    return tables


def get_dfs():
    """
    Reads the aggregate data from the parquet files and adds weather data for analysis. It returns a dictionary of DataFrames for each interval.
    """
    dfs = {}

    weather_data_df = get_weather_data()
    weather_data_df = weather_data_df.sort_values(["STN", "YYYYMMDD", "HH"])
    weather_data_df["TemperatuurRA"] = weather_data_df.groupby("STN")[
        "Temperatuur"
    ].transform(lambda x: x.rolling(window=14 * 24, min_periods=7 * 24).mean())
    weather_data_df["GevoelstemperatuurRA"] = weather_data_df.groupby("STN")[
        "Gevoelstemperatuur"
    ].transform(lambda x: x.rolling(window=14 * 24, min_periods=7 * 24).mean())
    weather_data_df["Koudste2WkTemperatuur"] = (
        weather_data_df.groupby("STN")
        .apply(mark_coldest_two_weeks, avg_var="TemperatuurRA", include_groups=False)
        .reset_index(level=0, drop=True)
    )
    weather_data_df["Koudste2WkGevoelstemperatuur"] = (
        weather_data_df.groupby("STN")
        .apply(
            mark_coldest_two_weeks, avg_var="GevoelstemperatuurRA", include_groups=False
        )
        .reset_index(level=0, drop=True)
    )

    project_weather_station_df = get_project_weather_station_data()

    for interval in intervals:
        dfs[interval] = read_aggregate(name="project", interval=interval)
        dfs[interval] = dfs[interval][dfs[interval]["ProjectIdBSV"] != 6.0]

        dfs[interval] = pd.merge(
            dfs[interval],
            project_weather_station_df[["ProjectIdBSV", "Weerstation", "STN"]],
            on="ProjectIdBSV",
            how="left",
        )

        if "YYYYMMDD" not in dfs.columns:
            dfs[interval]["HH"] = (
                dfs[interval]["ReadingDate"].dt.strftime("%H").astype(int) + 1
            )
            dfs[interval]["YYYYMMDD"] = (
                dfs[interval]["ReadingDate"].dt.strftime("%Y%m%d").astype(int)
            )

        dfs[interval] = pd.merge(
            dfs[interval],
            weather_data_df[weather_columns],
            left_on=["STN", "YYYYMMDD", "HH"],
            right_on=["STN", "YYYYMMDD", "HH"],
            how="left",
        )

        # Check for missing temperatures after the merge
        missing_temps = dfs[interval][dfs[interval]["Temperatuur"].isnull()]
        if not missing_temps.empty:
            total_records = dfs[interval].shape[0]
            missing_count = missing_temps.shape[0]
            percentage_missing = (missing_count / total_records) * 100

            print(
                f"Missing {missing_count} temperature out of {total_records} records ({percentage_missing:.2f}%)."
            )
            logging.info(
                f"Missing {missing_count} temperature out of {total_records} records ({percentage_missing:.2f}%)."
            )
            print(
                f"Affected ProjectIdBSV(s): {missing_temps['ProjectIdBSV'].unique().tolist()}"
            )
            logging.info(
                f"Affected ProjectIdBSV(s): {missing_temps['ProjectIdBSV'].unique().tolist()}"
            )
            print(
                f"Missing temperature dates: {missing_temps['ReadingDate'].dt.date.unique().tolist()}"
            )
            logging.info(
                f"Missing temperature dates: {missing_temps['ReadingDate'].dt.date.unique().tolist()}"
            )
            print(
                f"Missing temperature data details:\n{missing_temps[['ProjectIdBSV', 'ReadingDate', 'Weerstation', 'STN', 'YYYYMMDD', 'HH']].head()}"
            )  # Print a few details of missing records
            logging.info(
                f"Missing temperature data details:\n{missing_temps[['ProjectIdBSV', 'ReadingDate', 'Weerstation', 'STN', 'YYYYMMDD', 'HH']].head()}"
            )  # Print a few details of missing records

        missing_apparent_temps = dfs[interval][
            dfs[interval]["Gevoelstemperatuur"].isnull()
        ]
        if not missing_apparent_temps.empty:
            total_records = dfs[interval].shape[0]
            missing_count = missing_apparent_temps.shape[0]
            percentage_missing = (missing_count / total_records) * 100

            print(
                f"Missing {missing_count} apparent temperature out of {total_records} records ({percentage_missing:.2f}%)."
            )
            logging.info(
                f"Missing {missing_count} apparent temperature out of {total_records} records ({percentage_missing:.2f}%)."
            )
            print(
                f"Affected ProjectIdBSV(s): {missing_apparent_temps['ProjectIdBSV'].unique().tolist()}"
            )
            logging.info(
                f"Affected ProjectIdBSV(s): {missing_apparent_temps['ProjectIdBSV'].unique().tolist()}"
            )
            print(
                f"Missing apparent temperature dates: {missing_apparent_temps['ReadingDate'].dt.date.unique().tolist()}"
            )
            logging.info(
                f"Missing apparent temperature dates: {missing_apparent_temps['ReadingDate'].dt.date.unique().tolist()}"
            )
            print(
                f"Missing apparent temperature data details:\n{missing_apparent_temps[['ProjectIdBSV', 'ReadingDate', 'Weerstation', 'STN', 'YYYYMMDD', 'HH']].head()}"
            )  # Print a few details of missing records
            logging.info(
                f"Missing apparent temperature data details:\n{missing_apparent_temps[['ProjectIdBSV', 'ReadingDate', 'Weerstation', 'STN', 'YYYYMMDD', 'HH']].head()}"
            )  # Print a few details of missing records

    return dfs
