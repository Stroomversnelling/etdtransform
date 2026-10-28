import logging
import os
from typing import List, Optional, Tuple

import ibis
import ibis.selectors as s
import pandas as pd
from ibis import _

from etdmap import ids
from etdmap.data_model import load_etdmodel

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


def households_of_household_batches(household_batch_ids) -> list:
    """The households the given household batches belong to, from the index.

    The shard path carries a household level, so pruning directories needs the
    households a request's household batches sit under. Selecting rows is a separate,
    exact step on ``ids.HOUSEHOLD_BATCH``.
    """
    from etdmap.index_helpers import read_index

    wanted = {int(b) for b in household_batch_ids}
    index_df, _ = read_index(etdtransform.options.mapped_folder_path)
    match = index_df.loc[
        index_df[ids.HOUSEHOLD_BATCH].isin(wanted), ids.HOUSEHOLD
    ]
    return sorted(int(h) for h in match.dropna().unique())


def _household_table(name: str, household_batch_ids=None) -> ibis.Expr:
    """Read the household artifact called ``name`` and join it with the index.

    ``name`` is the artifact name without the ``household_`` prefix: a
    resampled interval or a pipeline stage. Internal helper so the read+join
    exists once; callers use get_household_table / get_household_tables.
    """
    from etdmap.storage import resolve_source_path

    source = resolve_source_path(
        etdtransform.options.aggregate_folder_path, f"household_{name}"
    )
    # Directory pruning takes the households the household batches sit under, because
    # that is the level the shard path has. Row selection is exact and keys on
    # the household batch, so a household's other household batches do not ride along.
    prune_to = (
        None if household_batch_ids is None
        else households_of_household_batches(household_batch_ids)
    )
    tbl = source_table(source, household_ids=prune_to)
    if household_batch_ids is not None:
        # Applied for BOTH layouts so the result does not depend on how the
        # data happens to be stored: a single-file source has no directories
        # to select from.
        tbl = tbl.filter(
            tbl[ids.HOUSEHOLD_BATCH].isin(
                sorted({int(b) for b in household_batch_ids})
            )
        )
    return join_index_table(tbl)


def household_batches_in_projects(projects) -> list:
    """
    The household batches belonging to the given projects.

    Projects are a column inside the data, not a storage partition, so a
    project filter cannot select files on its own. Resolving it to household batches
    through the index lets the reader open only the shards concerned.

    Resolved to household batches rather than households because a household's household batches
    may sit in different projects: resolving to the household would pull in the
    ones the request did not ask for.
    """
    from etdmap.index_helpers import read_index

    wanted = {int(p) for p in projects}
    index_df, _ = read_index(etdtransform.options.mapped_folder_path)
    match = index_df.loc[
        index_df[ids.PROJECT].isin(wanted), ids.HOUSEHOLD_BATCH
    ]
    return sorted(int(b) for b in match.dropna().unique())


def get_household_table(
    interval: str,
    include_weather: bool = True,
    household_batch_ids=None,
    projects=None,
) -> ibis.Expr:
    """
    Read the household data for one interval, joined with the index.

    Callers name the data they want, never a file path.

    Restricting to household batches or projects is worth doing here rather than
    filtering the returned table: a household batch's rows sit in their own shard
    directory, so naming them up front lets the reader skip the others
    entirely instead of opening every one to find out.

    Parameters
    ----------
    interval : str
        The resampled interval to read, e.g. ``"60min"`` or ``"24h"``. The
        set of intervals the pipeline produces is declared in the metadata,
        not fixed by this function.
    include_weather : bool, optional
        Join weather data as well (default True, matching
        :func:`get_household_tables`). When loading several tables at once,
        prefer that function: it reads the weather tables once and reuses
        them instead of per table.
    household_batch_ids : iterable of int, optional
        Restrict to these ``ids.HOUSEHOLD_BATCH`` values -- the household batches to
        read. None reads every household batch.
    projects : iterable of int, optional
        Restrict to these ``ids.PROJECT`` values, resolved to their household batches
        through the index. Combines with ``household_batch_ids`` as an
        intersection.

    Returns
    -------
    ibis.Expr
        Lazy table; the caller adds further filters and materialises when
        needed.
    """
    if projects is not None:
        in_projects = set(household_batches_in_projects(projects))
        household_batch_ids = (
            sorted(in_projects) if household_batch_ids is None
            else sorted(in_projects & {int(b) for b in household_batch_ids})
        )
    tbl = _household_table(interval, household_batch_ids=household_batch_ids)
    if include_weather:
        tbl = join_weather_data(tbl)
    return tbl


def get_household_tables(include_weather: bool = True, include_preimputed_data: bool = True, include_calculated_data: bool = True) -> dict[str, ibis.Expr]:
    """
    Reads household data tables for different intervals and joins them with an index table.
    Optionally integrates weather data.

    Parameters
    ----------
    include_weather : bool, optional
        If True, includes weather data in the returned tables (default is True).
    include_preimputed_data : bool, optional
        If True, includes the the table of household data before imputation (default is False).
    include_calculated_data : bool, optional
        If True, includes the the table of household data including all imputation and calulated columns before aggregation (default is False).

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

    if include_preimputed_data:
        # weather is joined once for all tables below, not per table
        household_tbls["default"] = get_household_table(
            "default", include_weather=False
        )

    if include_calculated_data:
        household_tbls["calculated"] = get_household_table(
            "calculated", include_weather=False
        )

    for interval in intervals:
        household_tbls[interval] = get_household_table(
            interval, include_weather=False
        )

    if include_weather:
        for key in household_tbls.keys():
            if key != "weather":
                household_tbls[key] = join_weather_data(
                    household_tbls[key],
                    weather_station_table=weather_station_table,
                    weather_table=household_tbls["weather"],
                )

    return household_tbls


def join_index_table(
    tbl: ibis.Expr,
    index_table: Optional[ibis.Expr] = None,
    index_join_columns: Tuple[str] = (ids.HOUSEHOLD_BATCH, ids.PROJECT),
) -> ibis.Expr:
    """
    Joins a given table with an index table on specified columns.

    index.parquet holds one row per household batch, so
    ``ids.HOUSEHOLD_BATCH`` is its key and the household id is not unique in
    it. ``ids.PROJECT`` stays in the key because both sides carry it; the
    index's household id arrives with a ``_right`` suffix and the data table's
    own household id remains authoritative.

    Parameters
    ----------
    tbl : ibis.Expr
        The table to join.
    index_table : Optional[ibis.Expr], optional
        The index table. If None, reads from default parquet file (default is None).
    index_join_columns : List[str], optional
        Columns to use for the join (default is [ids.HOUSEHOLD_BATCH, ids.PROJECT]).

    Returns
    -------
    ibis.Expr
        The table joined with the index table.
    """
    if index_table is None:
        index_table = ibis.read_parquet(
            os.path.join(etdtransform.options.mapped_folder_path, "index.parquet")
        )

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

    # Add columns according to our data model
    etdmodel =  load_etdmodel()
    # etdmodel["KNMI"] holds the KNMI column names
    # etdmodel["Variabele"] holds the data model's variable names

    filtered = etdmodel.dropna(subset=["KNMI", "Variabele"]).drop_duplicates(subset=["KNMI", "Variabele"])
    var_mapping = dict(zip(filtered["KNMI"], filtered["Variabele"]))

    for old_var, new_var in var_mapping.items():
        weather_data = weather_data.mutate(weather_data[old_var].name(new_var))

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

    # Extract hour (HH) and date (YYYYMMDD) for the join.
    # According to KNMI, "HH = tijd (HH=uur, UT.12 UT=13 MET, 14 MEZT. 
    # Uurvak 05 loopt van 04.00 UT tot 5.00 UT / time (HH uur/hour, UT. 12 UT=13 MET, 
    # 14 MEZT. Hourly division 05 runs from 04.00 UT to 5.00 UT"
    # KNMI hourly data labels each row with an "uurvak" HH in 1..24, where
    # uurvak h covers [(h-1):00, h:00) UT and the instantaneous values (T, U,
    # ...) are the observation at the end of that vak, i.e. at h:00 UT.
    # ReadingDate is naive UTC, so a reading at clock-hour h:-- must join to
    # uurvak h. Deriving both YYYYMMDD and HH from (ReadingDate -
    # 1 hour) yields exactly that (ref.hour()+1 == h for h in 1..23) and also
    # handles the midnight boundary correctly: a 00:00 reading maps to uurvak
    # 24 of the previous day instead of the non-existent HH=0.
    if 'YYYYMMDD' not in tbl.columns:
        ref = tbl.ReadingDate - ibis.interval(hours=1)
        tbl = tbl.mutate(
            HH=ref.hour() + 1,
            YYYYMMDD=ref.strftime("%Y%m%d").cast("int"),
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
        dfs[interval] = dfs[interval][dfs[interval][ids.PROJECT] != 6.0]

        dfs[interval] = pd.merge(
            dfs[interval],
            project_weather_station_df[[ids.PROJECT, "Weerstation", "STN"]],
            on=ids.PROJECT,
            how="left",
        )

        if "YYYYMMDD" not in dfs.columns:
            # See join_weather_data: KNMI uurvak HH (1..24) for a reading at
            # clock-hour h:00 UT is uurvak h (obs at end of vak), so derive
            # both HH and YYYYMMDD from (ReadingDate - 1 hour). That maps 00:00
            # readings to uurvak 24 of the previous day rather than to the
            # non-existent HH=0.
            ref = dfs[interval]["ReadingDate"] - pd.Timedelta(hours=1)
            dfs[interval]["HH"] = ref.dt.hour + 1
            dfs[interval]["YYYYMMDD"] = ref.dt.strftime("%Y%m%d").astype(int)

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
                f"Affected {ids.PROJECT}(s): {missing_temps[ids.PROJECT].unique().tolist()}"
            )
            logging.info(
                f"Affected {ids.PROJECT}(s): {missing_temps[ids.PROJECT].unique().tolist()}"
            )
            print(
                f"Missing temperature dates: {missing_temps['ReadingDate'].dt.date.unique().tolist()}"
            )
            logging.info(
                f"Missing temperature dates: {missing_temps['ReadingDate'].dt.date.unique().tolist()}"
            )
            print(
                f"Missing temperature data details:\n{missing_temps[[ids.PROJECT, 'ReadingDate', 'Weerstation', 'STN', 'YYYYMMDD', 'HH']].head()}"
            )  # Print a few details of missing records
            logging.info(
                f"Missing temperature data details:\n{missing_temps[[ids.PROJECT, 'ReadingDate', 'Weerstation', 'STN', 'YYYYMMDD', 'HH']].head()}"
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
                f"Affected {ids.PROJECT}(s): {missing_apparent_temps[ids.PROJECT].unique().tolist()}"
            )
            logging.info(
                f"Affected {ids.PROJECT}(s): {missing_apparent_temps[ids.PROJECT].unique().tolist()}"
            )
            print(
                f"Missing apparent temperature dates: {missing_apparent_temps['ReadingDate'].dt.date.unique().tolist()}"
            )
            logging.info(
                f"Missing apparent temperature dates: {missing_apparent_temps['ReadingDate'].dt.date.unique().tolist()}"
            )
            print(
                f"Missing apparent temperature data details:\n{missing_apparent_temps[[ids.PROJECT, 'ReadingDate', 'Weerstation', 'STN', 'YYYYMMDD', 'HH']].head()}"
            )  # Print a few details of missing records
            logging.info(
                f"Missing apparent temperature data details:\n{missing_apparent_temps[[ids.PROJECT, 'ReadingDate', 'Weerstation', 'STN', 'YYYYMMDD', 'HH']].head()}"
            )  # Print a few details of missing records

    return dfs


# ---------------------------------------------------------------------------
# Reading mapped household data.
#
# Mapped data is stored either as one parquet file per household (the legacy
# "flat" layout) or as a partitioned folder tree (the "sharded" layout, with
# folder names like HuisIdBSV=1/HuisBatchIdBSV=1). All reading functions
# detect the layout themselves -- callers never pass a format.
#
# Two read patterns, chosen by measurement:
#   - a specific set of households: read each household's file directly with
#     pandas (filtering a whole-folder scan down to a few households is
#     slower);
#   - the whole dataset: one lazy DuckDB/ibis query over all files at once.
#
# The functions that need no query engine live in etdmap.storage (etdmap is
# the base loading library that every repo already uses). They are
# re-exported here so existing callers keep working; this module adds only
# the DuckDB/ibis-based table constructors.
# ---------------------------------------------------------------------------

from etdmap.storage import (  # noqa: F401  (re-exports)
    detect_mapped_format,
    mapped_household_files,
    read_mapped_households,
    read_source_frame,
    resolve_source_path,
    shard_glob,
    source_is_sharded,
)


def source_table(path, household_ids=None) -> "ibis.Expr":
    """
    Lazy ibis/DuckDB table over a stage source: a single parquet file or a hive
    shard directory (hive partition ids become columns). The one engine-bound
    wrapper on top of etdmap.storage's layout detection.

    ``household_ids`` restricts a sharded source to those households' shard
    directories, so the reader never opens the rest.

    A directory whose partitions live under ``sharded/`` resolves to them, so
    the stage folder and that subdirectory read the same and the flat files
    beside them are never swept in.

    Raises ``ValueError`` for a flat directory. It carries the household id only
    in its file names, which this reader cannot see, so the lazy table would
    have no ``ids.HOUSEHOLD`` column and nothing would say so.
    ``read_source_frame`` reads that layout and injects the ids, which is the
    same choice ``mapped_household_table`` already makes.
    """
    from etdmap.storage import source_layout, source_read_targets

    p = str(path)
    layout = source_layout(p)
    if layout == "flat":
        raise ValueError(
            f"{p} is a flat directory of per-household files. A lazy table over "
            f"it would carry no {ids.HOUSEHOLD}. Use "
            f"etdmap.storage.read_source_frame, or point at the sharded "
            f"artifact."
        )
    if layout == "sharded":
        targets = source_read_targets(p, household_ids=household_ids)
        targets = (
            targets.replace("\\", "/") if isinstance(targets, str)
            else [t.replace("\\", "/") for t in targets]
        )
        return ibis.read_parquet(
            targets,
            hive_partitioning=True,
            union_by_name=True,
        )
    return ibis.read_parquet(p)


def mapped_household_table(mapped_folder_path=None) -> ibis.Expr:
    """
    Whole-household-batch lazy table over the SHARDED mapped data: one Ibis expression
    over all shards (hive partition columns ids.HOUSEHOLD/ids.HOUSEHOLD_BATCH included;
    per-supplier schema differences handled by union_by_name). Filters on
    ids.HOUSEHOLD/ids.HOUSEHOLD_BATCH prune partitions engine-side.

    Sharded-forward API: for a flat-only folder the legacy whole-table view is
    the stage-2 monolith (household_default.parquet via the existing loaders),
    so this raises rather than reimplementing that path.
    """
    if mapped_folder_path is None:
        mapped_folder_path = etdtransform.options.mapped_folder_path
    root = str(mapped_folder_path)
    if detect_mapped_format(root) != "sharded":
        raise ValueError(
            f"mapped_household_table requires the sharded layout; {root} is flat. "
            f"The legacy whole-table view of flat mapped data is the stage-2 "
            f"monolith (household_default.parquet via get_household_tables/"
            f"read_hh_data)."
        )
    pattern = os.path.join(root, "sharded", "**", "*.parquet").replace("\\", "/")
    return ibis.read_parquet(pattern, hive_partitioning=True, union_by_name=True)


def included_household_ids(mapped_folder_path=None) -> list:
    """
    The households whose household batch has `Meenemen` == True, from the index.

    A household-level grouping for reading, not a unit of analysis: callers
    use it to choose which shard directories to open.

    `Meenemen` is defined per household batch (one index row per
    ``ids.HOUSEHOLD_BATCH``). Collapsing the decision to household ids is valid
    only while no household holds more than one household batch; this function
    raises HouseBatchOverlapError otherwise.

    Inclusion is resolved at read time from the index instead of being
    baked into a materialised default file.
    """
    from etdmap.index_helpers import HouseBatchOverlapError, read_index

    if mapped_folder_path is None:
        mapped_folder_path = etdtransform.options.mapped_folder_path
    index_path = os.path.join(str(mapped_folder_path), "index.parquet")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"No index.parquet at {index_path}; cannot resolve the "
            f"included households. Map the data first (the index is written "
            f"by save_index_to_parquet)."
        )
    index_df, _ = read_index(mapped_folder_path)
    included = index_df.loc[
        index_df["Meenemen"] == True, ids.HOUSEHOLD  # noqa: E712
    ].dropna()
    dup = included.duplicated()
    if bool(dup.any()):
        multi = sorted(int(h) for h in included[dup].unique())
        raise HouseBatchOverlapError(
            f"{ids.HOUSEHOLD} {multi} hold more than one included household "
            f"batch in the index; collapsing `Meenemen` to household ids is "
            f"not valid. A household holding more than one household batch is "
            f"valid in the data model but not yet supported."
        )
    return sorted(int(h) for h in included.unique())
