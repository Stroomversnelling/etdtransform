import logging
import os
from typing import List, Optional

import ibis
import ibis.selectors as s
import numpy as np
import pandas as pd
from calculated_columns import intervals, mark_coldest_two_weeks
from etdmap import (
    aggregate_folder_path,
    get_aggregate_table,
    mapped_folder_path,
    read_aggregate,
)
from ibis import _

from etdtransform.knmi import (
    get_project_weather_station_data,
    get_weather_data,
    weather_columns,
)


def get_household_tables(include_weather=True) -> dict[str, ibis.Expr]:
    """
    Reads household data tables for different intervals and joins them with an index table as a dictionary of Ibis tables.

    Returns:
        dict[str, ibis.Expr]: A dictionary where keys are intervals and values
        are the joined Ibis tables for each interval.
    """
    household_tbls = {}

    if include_weather:
        household_tbls["weather"] = get_weather_data_table()
        weather_station_table = get_weather_station_table()

    for interval in intervals:
        household_parquet = os.path.join(
            aggregate_folder_path, f"household_{interval}.parquet"
        )
        household_table = ibis.read_parquet(household_parquet)

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
    index_join_columns: List[str] = ["HuisCode", "ProjectIdBSV"],
) -> ibis.Expr:
    if index_table is None:
        index_table = ibis.read_parquet(
            os.path.join(mapped_folder_path, "index.parquet")
        )

    return tbl.left_join(index_table, index_join_columns)


def get_weather_data_table() -> ibis.Expr:
    """
    Processes and transforms weather data into an Ibis table with additional calculated columns.

    Returns:
        ibis.Expr: An Ibis table containing transformed weather data with rolling averages,
        weekly summaries, and flags for the coldest two weeks based on temperature and perceived temperature.
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
    Weather statation data as an Ibis table.

    Returns:
        ibis.Expr: An Ibis table containing weather station data.
    """
    project_weather_station_df = get_project_weather_station_data()
    return ibis.memtable(project_weather_station_df)


def join_weather_data(tbl, weather_station_table=None, weather_table=None) -> ibis.Expr:
    if weather_table is None:
        weather_table = get_weather_data_table()
    if weather_station_table is None:
        weather_station_table = get_weather_station_table()

    # Join project table with project weather station data
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
