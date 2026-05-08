import logging
import os
import re
from typing import Optional

import ibis
import numpy as np
import pandas as pd
from etdmap.data_model import cumulative_columns, get_aggregation_config
from etdmap.index_helpers import read_index

import etdtransform
from etdtransform.calculated_columns import add_calculated_columns_imputed_data, add_calculated_columns_adaptive
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
        dtype_backend="numpy_nullable",
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


def aggregate_hh_data_5min(sample_ratio=1.0):
    """
    Aggregate household data into 5-minute intervals.

    Parameters
    ----------
    sample_ratio : float
        The percentage of households to include (0.0 to 1.0).
        Sampling is stratified by ProjectIdBSV to ensure equal distribution.

    Notes
    -----
    This function reads individual household parquet files, concatenates them,
    and saves the result as a single parquet file.
    """
    # Validate percentage
    if not (0.0 <= sample_ratio <= 1.0):
        raise ValueError("sample_ratio must be between 0.0 and 1.0")
    else:
        logging.info(f"Starting to aggregate household data (sample: {sample_ratio*100}%).")


    index_df, _ = read_index()


    # 1. Filter: Only keep Meenemen == 1
    df = index_df[index_df["Meenemen"]]

    if df.empty:
        logging.warning("No households found with Meenemen == 1.")
        return

    # 2. Sample: Stratified random sampling by ProjectIdBSV
    # frac handles the percentage. It automatically takes what's available in small groups.
    if sample_ratio != 1.0:
        sampled_df = df.groupby("ProjectIdBSV", group_keys=False).sample(frac=sample_ratio)
    else:
        sampled_df = df

    logging.info(f"Selected {len(sampled_df)} households from {len(df)} total.")

    # 3. Extract IDs to iterate
    # We only need these two columns for the loop
    ids_to_process = sampled_df[["HuisIdBSV", "ProjectIdBSV"]]

    data_frames = []

    for _, row in ids_to_process.iterrows():
        huis_id_bsv = row["HuisIdBSV"]
        project_code = row["ProjectIdBSV"]
        file_name = f"household_{huis_id_bsv}_table.parquet"
        file_path = os.path.join(etdtransform.options.mapped_folder_path, file_name)

        if not os.path.exists(file_path):
            continue

        household_df = pd.read_parquet(file_path)
        household_df["ProjectIdBSV"] = project_code
        household_df["HuisIdBSV"] = huis_id_bsv
        data_frames.append(household_df)

        logging.info(f"Added {file_name}")

    if not data_frames:
        raise ValueError("No data frames to aggregate.")

    logging.info("Concatenate all HH dataframes.")
    df_result = pd.concat(data_frames, ignore_index=True)

    logging.info("Saving HH data to parquet file.")
    df_result.to_parquet(
        os.path.join(etdtransform.options.aggregate_folder_path, "household_default.parquet"),
        engine="pyarrow",
    )


def aggregate_hh_data_5min_ibis(
    sample_ratio: float = 1.0,
    columns: Optional[list] = None,
    batch_size: int = 200,
):
    """
    Ibis/DuckDB variant of aggregate_hh_data_5min.

    Household files are processed in batches of batch_size to keep each Ibis
    expression small and safe for sqlglot's code generator.  Each batch is
    written to a private temp parquet file, then all batch parquets are
    combined with ibis.read_parquet(list) — a native DuckDB multi-file scan
    that requires no Ibis expression tree regardless of the number of batches.
    This approach scales to any number of files.

    A glob pattern is not used because some older household files carry
    TIMESTAMP WITH TIME ZONE parquet metadata while newer ones use TIMESTAMP_NS.
    Once all mapped files share a consistent ReadingDate type the batching can
    be replaced by a single ibis.read_parquet(glob) call.

    Parameters
    ----------
    sample_ratio : float
        Stratified sampling fraction by ProjectIdBSV (0.0-1.0).
    columns : list, optional
        Cumulative measurement columns to include.  When set, only
        HuisIdBSV, ProjectIdBSV, ReadingDate, and these columns are written.
    batch_size : int
        Household files per batch (default 200).
    """
    import shutil
    import tempfile

    if not (0.0 <= sample_ratio <= 1.0):
        raise ValueError("sample_ratio must be between 0.0 and 1.0")

    logging.info(f"Starting Ibis aggregation of household data (sample: {sample_ratio * 100}%).")

    index_df, _ = read_index()
    ids_df = index_df[index_df["Meenemen"]]

    if ids_df.empty:
        logging.warning("No households found with Meenemen == 1.")
        return

    if sample_ratio != 1.0:
        ids_df = ids_df.groupby("ProjectIdBSV", group_keys=False).sample(frac=sample_ratio)

    logging.info(f"Selected {len(ids_df)} households from {len(index_df[index_df['Meenemen']])} total.")

    id_cols = ["HuisIdBSV", "ProjectIdBSV", "ReadingDate"]
    rows = list(ids_df.itertuples(index=False))
    out_path = os.path.join(etdtransform.options.aggregate_folder_path, "household_default.parquet")

    tmp_dir = tempfile.mkdtemp(prefix="etd_agg_")
    try:
        batch_parquets = []

        for batch_start in range(0, len(rows), batch_size):
            batch = rows[batch_start : batch_start + batch_size]
            batch_tables = []

            for row in batch:
                file_path = os.path.join(
                    etdtransform.options.mapped_folder_path,
                    f"household_{row.HuisIdBSV}_table.parquet",
                )
                if not os.path.exists(file_path):
                    continue

                tbl_hh = ibis.read_parquet(file_path).mutate(
                    HuisIdBSV=ibis.literal(int(row.HuisIdBSV), type="int64"),
                    ProjectIdBSV=ibis.literal(int(row.ProjectIdBSV), type="int64"),
                )
                if columns is not None:
                    keep = id_cols + [c for c in columns if c not in id_cols]
                    # Null-fill columns absent in this file so all tables share
                    # the same schema and ibis union does not raise a type error.
                    select_exprs = []
                    for c in keep:
                        if c in tbl_hh.columns:
                            select_exprs.append(tbl_hh[c])
                        else:
                            select_exprs.append(ibis.null().cast("float64").name(c))
                    tbl_hh = tbl_hh.select(select_exprs)

                batch_tables.append(tbl_hh)

            if not batch_tables:
                continue

            batch_result = batch_tables[0]
            for t in batch_tables[1:]:
                batch_result = batch_result.union(t)

            batch_path = os.path.join(tmp_dir, f"batch_{batch_start:06d}.parquet")
            batch_result.to_parquet(batch_path)
            batch_parquets.append(batch_path)
            logging.info(
                f"Wrote batch {batch_start // batch_size + 1}"
                f" ({len(batch_tables)} households) to temp parquet"
            )

        if not batch_parquets:
            raise ValueError("No household files found to aggregate.")

        logging.info(f"Combining {len(batch_parquets)} batch parquets -> {out_path}")
        final_tbl = ibis.read_parquet(batch_parquets)
        # DuckDB downcasts small-valued columns to int32 in parquet; enforce Int64 per ADR-005
        final_tbl = final_tbl.mutate(
            HuisIdBSV=final_tbl["HuisIdBSV"].cast("int64"),
            ProjectIdBSV=final_tbl["ProjectIdBSV"].cast("int64"),
        )
        final_tbl.to_parquet(out_path)
        logging.info(f"Saved aggregated household data to {out_path}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def aggregate_hh_data_duckdb(
    sample_ratio: float = 1.0,
    columns: Optional[list] = None,
) -> None:
    """
    Aggregate mapped household parquets into household_default.parquet using DuckDB.

    Uses read_parquet with union_by_name=True so supplier-specific extra columns
    are handled natively — no schema alignment, no pandas accumulation in memory.
    HuisIdBSV and ProjectIdBSV are injected via a JOIN on a filename->ID mapping
    table registered from the index (DuckDB filename=True feature).

    Parameters
    ----------
    sample_ratio : float
        Stratified sampling fraction by ProjectIdBSV (0.0-1.0).
    columns : list, optional
        Data columns to include in addition to HuisIdBSV, ProjectIdBSV, ReadingDate.
        When None, all columns are included.  Passing cumulative_columns is the
        typical use case and reduces output size significantly.
    """
    import duckdb

    if not (0.0 <= sample_ratio <= 1.0):
        raise ValueError("sample_ratio must be between 0.0 and 1.0")

    index_df, _ = read_index()
    ids_df = index_df[index_df["Meenemen"]]
    if ids_df.empty:
        logging.warning("No households found with Meenemen == 1.")
        return

    if sample_ratio != 1.0:
        ids_df = ids_df.groupby("ProjectIdBSV", group_keys=False).sample(frac=sample_ratio)

    logging.info(f"DuckDB aggregation: {len(ids_df)} households (sample {sample_ratio*100:.0f}%).")

    folder = etdtransform.options.mapped_folder_path
    mapping_rows = []
    file_paths = []
    for row in ids_df.itertuples(index=False):
        path = os.path.join(folder, f"household_{row.HuisIdBSV}_table.parquet")
        if not os.path.exists(path):
            continue
        # Use forward slashes — DuckDB on Windows handles both but forward is safer in SQL
        fwd = path.replace("\\", "/")
        mapping_rows.append((fwd, int(row.HuisIdBSV), int(row.ProjectIdBSV)))
        file_paths.append(fwd)

    if not file_paths:
        raise ValueError("No household parquet files found.")

    out_path = os.path.join(
        etdtransform.options.aggregate_folder_path, "household_default.parquet"
    ).replace("\\", "/")

    con = duckdb.connect()
    try:
        con.execute(
            "CREATE TEMP TABLE id_map (file_path VARCHAR, HuisIdBSV BIGINT, ProjectIdBSV BIGINT)"
        )
        con.executemany("INSERT INTO id_map VALUES (?, ?, ?)", mapping_rows)

        paths_sql = ", ".join(f"'{p}'" for p in file_paths)

        if columns is not None:
            # Discover the unified schema across all files.
            available = {
                r[0]
                for r in con.execute(
                    f"DESCRIBE SELECT * FROM read_parquet([{paths_sql}], union_by_name=True)"
                ).fetchall()
            }
            # Strict projection: only the requested cumulative columns.
            # Non-cumulative columns (temperatures, setpoints) are late-joined from
            # household_default.parquet at analysis time (Option B, plan ADR).
            data_cols = [c for c in columns if c in available]
            col_sql = "m.HuisIdBSV, m.ProjectIdBSV, d.ReadingDate" + (
                ", " + ", ".join(f'd."{c}"' for c in data_cols) if data_cols else ""
            )
        else:
            col_sql = "m.HuisIdBSV, m.ProjectIdBSV, d.* EXCLUDE (filename)"

        con.execute(f"""
            COPY (
                SELECT {col_sql}
                FROM read_parquet([{paths_sql}], union_by_name=True, filename=True) d
                JOIN id_map m ON d.filename = m.file_path
            ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION SNAPPY)
        """)
        logging.info(f"Saved aggregated household data to {out_path}")
    finally:
        con.close()


def reconstruct_cumulative_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Rebuild cumulative columns from their imputed Diff columns in-place.

    For each col in cols:
      - col + "Original"  = original cumulative value (saved for audit)
      - col               = cumsum of col + "Diff" per household (rebuilt)
      - col + "Check"     = per-household diff of (rebuilt - original)

    Data must be sorted by HuisIdBSV then ReadingDate before calling this.
    Mutates df in-place and returns it.
    """
    for col in cols:
        df[col + "Original"] = df[col]
        df[col] = df.groupby("HuisIdBSV", sort=False)[col + "Diff"].cumsum()
        df[col + "Check"] = (
            (df[col] - df[col + "Original"]).groupby(df["HuisIdBSV"]).diff()
        )
    return df


def reconstruct_cumulative_columns_ibis(tbl: "ibis.Table", cols: list) -> "ibis.Table":
    """
    Ibis/DuckDB variant of reconstruct_cumulative_columns.

    Builds lazy window expressions — nothing is executed until the caller
    calls .to_parquet() or .execute().  All three derived columns per col
    (Original, rebuilt cumsum, Check) are added as Ibis mutations.

    Typical pipeline use:
        tbl = ibis.read_parquet(imputed_path)
        tbl = reconstruct_cumulative_columns_ibis(tbl, cum_cols)
        tbl.to_parquet(imputed_path)   # overwrites with reconstructed data
    """
    win_cum = ibis.window(
        group_by="HuisIdBSV", order_by="ReadingDate", preceding=None, following=0
    )
    win_lag = ibis.window(group_by="HuisIdBSV", order_by="ReadingDate")

    # Pass 1: all orig_col saves + cumsum rewrites in one mutate().
    # Batching avoids building a 2*len(cols)-deep nested subquery chain —
    # each loop iteration would otherwise wrap tbl in another subquery level,
    # causing DuckDB query-planning time to grow quadratically with col count.
    # All cols are independent of each other so a single mutate is correct.
    first_pass = {}
    for col in cols:
        first_pass[col + "Original"] = tbl[col]
        first_pass[col] = tbl[col + "Diff"].sum().over(win_cum)
    tbl = tbl.mutate(**first_pass)

    # Pass 2: all check_col computations in one mutate().
    # Each check_col references the updated col and orig_col from pass 1 but
    # is independent of every other check_col, so batching is safe.
    second_pass = {}
    for col in cols:
        rmo = tbl[col] - tbl[col + "Original"]
        second_pass[col + "Check"] = rmo - rmo.lag(1).over(win_lag)
    tbl = tbl.mutate(**second_pass)

    return tbl


def impute_hh_data_5min(
    df,
    cum_cols=cumulative_columns,
    sorted=False,
    diffs_calculated=False,
    optimized=False,
    reconstruct_in_pandas: bool = True,
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
    reconstruct_in_pandas : bool, optional
        Whether to reconstruct cumulative columns (*Original, cumsum, *Check) in
        pandas before writing to parquet.  Set to False when the caller will do
        the reconstruction via reconstruct_cumulative_columns_ibis() on the saved
        parquet — avoids materialising the enlarged DataFrame in pandas RAM.
        Default True preserves the original behaviour.

    Returns
    -------
    pd.DataFrame
        The imputed household data (without reconstruction when reconstruct_in_pandas=False)

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

    diff_columns = [col + "Diff" for col in cum_cols if col + "Diff" in df.columns]

    logging.info("Averaging all diffs by project and reading date.")

    aggregated_diff = (
        df.groupby(["ProjectIdBSV", "ReadingDate"])[diff_columns].mean().reset_index()
    )

    if reconstruct_in_pandas:
        logging.info("Reconstructing cumulative columns from diffs.")
        df = reconstruct_cumulative_columns(df, cumulative_columns)

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


def impute_hh_data_5min_chunked(
    source_path,
    chunk_size: int = 50,
    cum_cols=cumulative_columns,
    huis_ids=None,
):
    """
    Chunked variant of impute_hh_data_5min.

    Reads source_path in batches of chunk_size households, imputes each batch
    with the column-vectorised engine, and streams results to
    household_imputed.parquet via PyArrow ParquetWriter.  RAM footprint is
    bounded by chunk_size rather than the full dataset.

    Requires avg_diffs.parquet and household_diff_max_bounds.parquet to already
    exist -- run prepare_diffs_for_impute / prepare_diffs_for_impute_ibis first.

    Parameters
    ----------
    source_path : str | Path
        Path to the source household parquet (always household_default.parquet).
    chunk_size : int
        Number of households per batch.  Default 50.
    cum_cols : list
        Cumulative columns to impute.
    huis_ids : list | None
        If provided, restrict processing to this subset of HuisIdBSV values.
        Used by sample/filter steps to avoid writing a temp parquet.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import duckdb
    from etdtransform.impute import sort_for_impute, read_diffs
    from etdtransform.vectorized_impute import impute_and_normalize, methods_to_bitwise

    source_path = str(source_path)
    # Columns needed by the imputation engine: IDs + cumulative + their Diff columns.
    # All OTHER columns from the source are preserved unchanged and passed through to
    # household_imputed.parquet so downstream steps (e.g. add_calculated_columns) see
    # the same full column set as the pandas full-load path.
    _schema_cols = set(pq.read_schema(source_path).names)
    diff_cols = [f"{col}Diff" for col in cum_cols]
    impute_cols = set(
        c for c in ["HuisIdBSV", "ProjectIdBSV", "ReadingDate"] + list(cum_cols) + diff_cols
        if c in _schema_cols
    )

    # Step 1: resolve the household ID set — column-only read, tiny
    if huis_ids is not None:
        all_ids = sorted(int(x) for x in huis_ids)
    else:
        all_ids = sorted(
            int(x) for x in
            pd.read_parquet(source_path, columns=["HuisIdBSV"], dtype_backend="numpy_nullable")
            ["HuisIdBSV"].dropna().unique()
        )
    n_hh = len(all_ids)
    chunks = [all_ids[i:i + chunk_size] for i in range(0, n_hh, chunk_size)]
    logging.info(f"[chunked impute] {n_hh} households -> {len(chunks)} chunks of {chunk_size}")

    # Step 2: load pre-computed diffs (small files, always required)
    logging.info("[chunked impute] Loading pre-computed diff averages...")
    diffs = read_diffs()
    max_bound = pd.read_parquet(
        os.path.join(etdtransform.options.aggregate_folder_path, "household_diff_max_bounds.parquet"),
        dtype_backend="numpy_nullable",
    )

    out_path = os.path.join(etdtransform.options.aggregate_folder_path, "household_imputed.parquet")
    writer: "pq.ParquetWriter | None" = None
    all_gap_stats: list = []
    total_records: dict = {}

    # Step 3: impute each chunk
    for i, chunk_ids in enumerate(chunks):
        logging.info(f"[chunked impute] Chunk {i + 1}/{len(chunks)}: {len(chunk_ids)} households")

        # Read ALL source columns — pass-through columns are preserved unchanged so
        # household_imputed.parquet has the same schema as household_default.parquet.
        chunk_df = pd.read_parquet(
            source_path,
            filters=[("HuisIdBSV", "in", chunk_ids)],
            dtype_backend="numpy_nullable",
        )

        chunk_df = sort_for_impute(chunk_df, "ProjectIdBSV")
        chunk_df = chunk_df.merge(diffs, on=["ProjectIdBSV", "ReadingDate"], how="left")

        chunk_df, gap_stats_chunk, _ = impute_and_normalize(
            chunk_df, list(cum_cols), "ProjectIdBSV", max_bound
        )

        # Rebuild cumulative columns from imputed Diff columns per household.
        # chunk_df is already sorted by sort_for_impute, so cumsum is correct.
        cols_to_reconstruct = [c for c in cum_cols if f"{c}Diff" in chunk_df.columns]
        chunk_df = reconstruct_cumulative_columns(chunk_df, cols_to_reconstruct)

        all_gap_stats.append(gap_stats_chunk)
        for hid, cnt in chunk_df.groupby("HuisIdBSV").size().items():
            total_records[int(hid)] = int(cnt)

        chunk_table = pa.Table.from_pandas(chunk_df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, chunk_table.schema, compression="snappy")
        writer.write_table(chunk_table)
        del chunk_df

    if writer:
        writer.close()

    # Step 4: combine gap stats and save summaries
    logging.info("[chunked impute] Combining gap stats and saving summaries...")
    non_empty = [g for g in all_gap_stats if not g.empty]
    if not non_empty:
        raise RuntimeError(
            "[chunked impute] No gap stats produced across any chunk. "
            "This means no Diff columns were present after merging diffs. "
            "Ensure prepare_diffs_for_impute has been run before imputation."
        )
    imputation_gap_stats_df = pd.concat(non_empty, ignore_index=True)
    imputation_gap_stats_df["bitwise_methods"] = methods_to_bitwise(
        imputation_gap_stats_df["methods"]
    )

    imputation_gap_stats_df.to_parquet(
        os.path.join(etdtransform.options.aggregate_folder_path, "impute_gap_stats.parquet"),
        engine="pyarrow",
    )

    total_records_df = pd.DataFrame(
        [{"HuisIdBSV": k, "total_records": v} for k, v in total_records.items()]
    )
    summary_house = (
        imputation_gap_stats_df[[
            "ProjectIdBSV", "HuisIdBSV", "column", "diff_col_total",
            "cum_col_min_max_diff", "missing", "imputed", "imputed_na",
            "methods", "bitwise_methods",
        ]]
        .merge(total_records_df, on="HuisIdBSV")
    )
    summary_house["percentage_imputed"] = (
        summary_house["imputed"] / summary_house["total_records"] * 100
    )
    summary_house.to_parquet(
        os.path.join(etdtransform.options.aggregate_folder_path, "impute_summary_household.parquet"),
        engine="pyarrow",
    )

    total_project = (
        imputation_gap_stats_df[["ProjectIdBSV", "HuisIdBSV"]]
        .drop_duplicates()
        .merge(total_records_df, on="HuisIdBSV")
        .groupby("ProjectIdBSV")["total_records"].sum()
        .reset_index()
    )
    summary_project = (
        imputation_gap_stats_df.groupby(["ProjectIdBSV", "column"])
        .agg(
            bitwise_methods=("bitwise_methods", lambda x: np.bitwise_or.reduce(x.values)),
            methods=("methods", lambda x: list(set().union(*x))),
            missing=("missing", "sum"),
            imputed=("imputed", "sum"),
            imputed_na=("imputed_na", "sum"),
        )
        .reset_index()
        .merge(total_project, on="ProjectIdBSV")
    )
    summary_project["percentage_imputed"] = (
        summary_project["imputed"] / summary_project["total_records"] * 100
    )
    summary_project.to_parquet(
        os.path.join(etdtransform.options.aggregate_folder_path, "impute_summary_project.parquet"),
        engine="pyarrow",
    )

    # Step 5: aggregated diff via DuckDB on the finished parquet — no pandas RAM
    diff_cols_present = [
        c for c in [f"{col}Diff" for col in cum_cols]
        if c in pq.read_schema(out_path).names
    ]
    if diff_cols_present:
        agg_sql = ", ".join(f'AVG("{c}") AS "{c}"' for c in diff_cols_present)
        aggregated_diff = duckdb.query(
            f'SELECT ProjectIdBSV, ReadingDate, {agg_sql} '
            f"FROM read_parquet('{out_path}') "
            f"GROUP BY ProjectIdBSV, ReadingDate"
        ).df()
        for c in diff_cols_present:
            aggregated_diff[c] = aggregated_diff[c].astype("Float64")
        aggregated_diff.to_parquet(
            os.path.join(etdtransform.options.aggregate_folder_path, "household_aggregated_diff.parquet"),
            engine="pyarrow",
        )

    over_40 = summary_house[summary_house["percentage_imputed"] > 40]
    for _, row in over_40.iterrows():
        logging.warning(
            f"House {int(row['HuisIdBSV'])}, Column {row['column']} has "
            f"{row['percentage_imputed']:.2f}% imputed values."
        )

    logging.info("[chunked impute] Done.")


def add_calculated_columns_to_hh_data_ibis(
    source_path: str,
    output_path: str,
    completeness_threshold: float = 0.95,
    catalog_df=None,
    target_columns=None,
    fillna_vars: list = None,
) -> None:
    """
    Ibis/DuckDB variant: derives calculated columns without a full pandas load.

    Reads source_path directly from disk, computes per-household column
    availability in one DuckDB GROUP BY pass, groups households by schema
    (frozenset of available catalog input columns), derives calculated columns
    per schema group via ibis mutate, and writes combined output to output_path.

    Phases:
      1. Availability analysis -- one DuckDB GROUP BY pass over the parquet.
      2. Planning -- DatasetAdapter.execution_plan_for_available per schema group.
      3. Mutate per schema group -- ibis filter + sequential mutate per derived col.
      4. Combine -- DuckDB read_parquet with union_by_name=True.

    Parameters
    ----------
    source_path : str
        Path to household_imputed.parquet (input).
    output_path : str
        Path for household_calculated.parquet (output).
    completeness_threshold : float
        Minimum non-null fraction for a column to be considered available.
    fillna_vars : list, optional
        Column names to fill with 0 before availability analysis and derivation.
        Use for device columns that were not installed (all-null in the source)
        but must be treated as 0 for downstream calculations to be correct.
        Columns not present in the source parquet are logged and skipped.
    """
    import duckdb
    import shutil
    import tempfile
    from etdmap.catalog import load_catalog
    from etdmap.data_model import all_performance_data_columns, required_performance_data_columns
    from etdtransform.catalog.query import DatasetAdapter
    from etdtransform.calculated_columns import sympy_to_ibis

    source_path = str(source_path)
    output_path = str(output_path)

    logging.info("[calc ibis] Loading catalog...")
    if catalog_df is None:
        catalog_df = load_catalog()
    adapter = DatasetAdapter(catalog_df, completeness_threshold)

    if target_columns is not None:
        all_perf_cols = set(target_columns)
        req_perf_cols = set()
    else:
        all_perf_cols = set(all_performance_data_columns)
        req_perf_cols = set(required_performance_data_columns)
    derivable_in_catalog = set(catalog_df["lhs"].unique())

    # Phase 1: schema check + rename shim + availability analysis
    tbl = ibis.read_parquet(source_path)
    parquet_col_set = set(tbl.columns)

    # Zon-opwekTotaalDiff -> ZonopwekBruto compatibility shim
    # (hyphenated name is unparseable as a SymPy symbol; rename before catalog use)
    has_zon_rename = (
        "Zon-opwekTotaalDiff" in parquet_col_set
        and "ZonopwekBruto" not in parquet_col_set
    )
    if has_zon_rename:
        tbl = tbl.rename({"ZonopwekBruto": "Zon-opwekTotaalDiff"})
        logging.info(
            "[calc ibis] Renamed Zon-opwekTotaalDiff -> ZonopwekBruto "
            "(compatibility shim)"
        )

    effective_cols = set(tbl.columns)

    # fillna_vars shim: fill specified absent-device columns with 0 before availability
    # analysis so they are treated as available (not missing) by the adaptive planner.
    _fillna_vars = list(fillna_vars) if fillna_vars else []
    _present_fillna = [c for c in _fillna_vars if c in effective_cols]
    _missing_fillna = [c for c in _fillna_vars if c not in effective_cols]
    if _present_fillna:
        tbl = tbl.mutate(**{c: tbl[c].fillna(0) for c in _present_fillna})
        logging.info(f"[calc ibis] fillna(0) applied to existing columns: {_present_fillna}")
    if _missing_fillna:
        logging.warning(
            f"[calc ibis] fillna_vars columns not in source parquet -- "
            f"creating as all-zero: {_missing_fillna}"
        )
        tbl = tbl.mutate(**{c: ibis.literal(0.0) for c in _missing_fillna})
        effective_cols = effective_cols | set(_missing_fillna)

    # Candidate input columns: catalog rhs_vars + performance data cols, present in parquet
    all_rhs_vars = {v for vars_list in catalog_df["rhs_vars"] for v in vars_list}
    analysis_cols = sorted(effective_cols & (all_rhs_vars | all_perf_cols))

    logging.info(
        f"[calc ibis] Computing per-household availability for "
        f"{len(analysis_cols)} candidate columns..."
    )

    agg_exprs = [tbl[col].notnull().mean().name(col) for col in analysis_cols]
    availability_df = tbl.group_by("HuisIdBSV").agg(agg_exprs).execute()

    # Build schema groups: frozenset(available_cols) -> list[int HuisIdBSV]
    schema_groups: dict = {}
    for _, row in availability_df.iterrows():
        huis_id = int(row["HuisIdBSV"])
        avail = frozenset(
            col for col in analysis_cols
            if not pd.isna(row.get(col, float("nan"))) and row[col] >= completeness_threshold
        )
        schema_groups.setdefault(avail, []).append(huis_id)

    logging.info(
        f"[calc ibis] {len(schema_groups)} unique schema groups "
        f"across {len(availability_df)} households"
    )

    # Phase 2 + 3: plan and mutate per schema group
    tmp_dir = tempfile.mkdtemp(prefix="etd_calc_")
    temp_paths = []

    try:
        for group_idx, (available_cols, group_ids) in enumerate(schema_groups.items()):
            targets = (all_perf_cols - available_cols) & derivable_in_catalog

            if not targets:
                plan: list = []
                not_derivable: list = []
            else:
                plan, not_derivable = adapter.execution_plan_for_available(
                    available_cols, targets
                )

            if not_derivable:
                # Find which RHS input columns failed the completeness threshold
                # (present in the parquet but below threshold -- not absent entirely).
                needed_rhs: set = set()
                for tgt in not_derivable:
                    for _, rule_row in catalog_df[catalog_df["lhs"] == tgt].iterrows():
                        needed_rhs.update(rule_row["rhs_vars"])
                blocked_inputs = sorted((needed_rhs - available_cols) & set(analysis_cols))
                if blocked_inputs:
                    group_avail = availability_df[availability_df["HuisIdBSV"].isin(group_ids)]
                    avail_by_col = {
                        col: float(group_avail[col].mean())
                        for col in blocked_inputs
                        if col in group_avail.columns
                    }
                    blocked_str = ", ".join(
                        f"{c}={v:.1%}"
                        for c, v in sorted(avail_by_col.items(), key=lambda x: x[1])
                    )
                    logging.warning(
                        f"[calc ibis] Schema group {group_idx}: derivation blocked -- "
                        f"completeness below {completeness_threshold:.0%} threshold: "
                        f"{blocked_str}. "
                        f"Lower completeness_threshold to derive columns from partial data."
                    )

                not_req = set(not_derivable) & req_perf_cols
                hh_sample = sorted(group_ids)[:10]
                hh_note = (
                    f"{len(group_ids)} HH: {hh_sample}"
                    if len(group_ids) <= 10
                    else f"{len(group_ids)} HH (first 10): {hh_sample}"
                )
                logging.error(
                    f"[calc ibis] Schema group {group_idx} ({hh_note}): "
                    f"could not derive {sorted(not_derivable)}"
                    + (f" (required: {sorted(not_req)})" if not_req else "")
                )

            group_tbl = ibis.read_parquet(source_path)
            group_tbl = group_tbl.filter(group_tbl["HuisIdBSV"].isin(group_ids))
            if has_zon_rename:
                group_tbl = group_tbl.rename({"ZonopwekBruto": "Zon-opwekTotaalDiff"})
            if _present_fillna:
                group_tbl = group_tbl.mutate(
                    **{c: group_tbl[c].fill_null(0.0) for c in _present_fillna}
                )
            if _missing_fillna:
                group_tbl = group_tbl.mutate(
                    **{c: ibis.literal(0.0) for c in _missing_fillna}
                )

            for col_name, rhs_expr in plan:
                group_tbl = group_tbl.mutate(**{col_name: sympy_to_ibis(rhs_expr, group_tbl)})

            temp_path = os.path.join(tmp_dir, f"calc_group_{group_idx:04d}.parquet")
            group_tbl.to_parquet(temp_path)
            temp_paths.append(temp_path)
            logging.info(
                f"[calc ibis] Group {group_idx + 1}/{len(schema_groups)}: "
                f"{len(group_ids)} HH, {len(plan)} columns derived"
            )

        if not temp_paths:
            raise RuntimeError("[calc ibis] No schema groups processed -- no output written.")

        # Phase 4: combine with Polars streaming (diagonal_relaxed fills missing cols with null)
        import polars as pl
        import tempfile
        import pyarrow as pa
        import pyarrow.parquet as pq
        logging.info(
            f"[calc ibis] Combining {len(temp_paths)} group parquets -> {output_path}"
        )
        frames = [pl.scan_parquet(p) for p in temp_paths]
        pl.concat(frames, how="diagonal_relaxed").sink_parquet(output_path, compression="snappy")

        # Polars writes string columns as large_string (large_utf8). Pandas
        # dtype_backend="numpy_nullable" maps utf8 -> string[python] but maps
        # large_utf8 -> object, violating ADR-005. Cast large_string -> string
        # via a PyArrow streaming passthrough so the output is dtype-correct.
        pf = pq.ParquetFile(output_path)
        arrow_schema = pf.schema_arrow
        if any(pa.types.is_large_string(arrow_schema.field(i).type) for i in range(len(arrow_schema))):
            new_schema = pa.schema([
                f.with_type(pa.string()) if pa.types.is_large_string(f.type) else f
                for f in arrow_schema
            ])
            tmp_fd, tmp_path = tempfile.mkstemp(
                suffix=".parquet", dir=os.path.dirname(output_path)
            )
            os.close(tmp_fd)
            try:
                with pq.ParquetWriter(tmp_path, new_schema, compression="snappy") as writer:
                    for batch in pf.iter_batches():
                        writer.write_batch(batch.cast(new_schema))
                pf = None  # release file handle before replace (Windows requires this)
                os.replace(tmp_path, output_path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        else:
            pf = None

        logging.info("[calc ibis] Saved household_calculated.parquet")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def add_calculated_columns_to_hh_data(df, adaptive = False):
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
    if df is None:
        logging.info("Loading imputed data from parquet file.")
        df = read_hh_data(interval="imputed")

    logging.info("Calculating: ")
    if adaptive:
        df = add_calculated_columns_adaptive(df)
    else:
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
        The input DataFrame.  When None, dispatches to resample_hh_data_duckdb()
        which reads household_calculated.parquet directly without loading it into
        memory.  Pass a DataFrame only for testing or one-off use cases.
    intervals : tuple, optional
        The time intervals to resample to, by default ("60min", "15min", "5min")

    Notes
    -----
    This function resamples household data to specified time intervals and saves the results.
    """
    if df is None:
        logging.info("resample_hh_data: dispatching to DuckDB path")
        source_path = os.path.join(
            etdtransform.options.aggregate_folder_path, "household_calculated.parquet"
        )
        resample_hh_data_duckdb(
            source_path=source_path,
            output_dir=etdtransform.options.aggregate_folder_path,
            intervals=intervals,
        )
        return

    logging.warning(
        "If passing a dataframe to resample_hh_data() be sure to use a copy as it may be modified in place.",
    )
    group_column = ["ProjectIdBSV", "HuisIdBSV"]

    for interval in intervals:
        logging.info(f"-- Starting household resampling with {interval} intervals --")

        if interval == "5min":
            logging.info(
                "-- 5min interval - applying shortcut without transformation --",
            )
            active_vars = active_aggregation_variables(df)
            columns_to_copy = [
                "ReadingDate",
                *group_column,
                *list(active_vars.keys()),
            ]

            for _var, config in active_vars.items():
                validator_column = config.get("validator_column")
                if validator_column:
                    columns_to_copy.append(validator_column)

            df = df[columns_to_copy]

            logging.info(
                f"{interval}min interval - removing variables that do not pass filters"
            )
            for var, config in active_vars.items():
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
    Dispatches to aggregate_project_data_duckdb(), which reads each
    household_{interval}.parquet directly without loading it into memory.
    """
    logging.info("aggregate_project_data: dispatching to DuckDB path")
    aggregate_project_data_duckdb(intervals=intervals)


def _sql_path(path) -> str:
    """Return path as a single-quoted SQL string literal with forward slashes."""
    return "'" + str(path).replace("\\", "/").replace("'", "''") + "'"


def _resample_sql_expr(col: str, method: str, min_count: int) -> str:
    """Return the DuckDB SQL expression for resampling a single column.

    Column names are double-quoted so hyphens (e.g. 'Zon-opwekTotaal') are
    not parsed as arithmetic by DuckDB. All aggregate operands are cast to
    DOUBLE so boolean columns (e.g. VentilatieFiltermelding) produce numeric
    output (True -> 1.0, False -> 0.0) rather than raising a type error.
    """
    qcol = f'"{col}"'
    num = f'CAST("{col}" AS DOUBLE)'
    if method == "sum":
        return f"CASE WHEN COUNT({qcol}) >= {min_count} THEN SUM({num}) ELSE NULL END AS {qcol}"
    if method == "max":
        return f"CASE WHEN COUNT({qcol}) >= {min_count} THEN MAX({num}) ELSE NULL END AS {qcol}"
    if method == "avg":
        return (
            f"CASE WHEN COUNT({qcol}) >= {min_count}"
            f" THEN SUM({num}) / COUNT({qcol}) ELSE NULL END AS {qcol}"
        )
    raise ValueError(f"Unknown resample_method '{method}' for column '{col}'")


def _parse_interval(interval: str):
    """Return (bucket_minutes, duckdb_interval_expr) for an interval string.

    Accepts Nmin (e.g. "5min", "15min", "60min") or Nh (e.g. "6h", "24h").
    """
    import re
    m = re.fullmatch(r"(\d+)min", interval)
    if m:
        mins = int(m.group(1))
        return mins, f"INTERVAL '{mins} minutes'"
    m = re.fullmatch(r"(\d+)h", interval)
    if m:
        hours = int(m.group(1))
        return hours * 60, f"INTERVAL '{hours} hours'"
    raise ValueError(f"Unsupported interval '{interval}'. Expected Nmin or Nh.")


def resample_hh_data_duckdb(
    source_path: str,
    output_dir: str,
    intervals: tuple = ("60min", "15min", "5min"),
) -> None:
    """
    Resample household_calculated.parquet to multiple time intervals using DuckDB.

    Reads resampling config from etdmap.data_model.get_aggregation_config() (ADR-004).
    Only processes columns present in the source parquet (guards against columns not
    yet computed for this run).

    For 5min: column selection only -- no aggregation.
    For 15min / 60min: TIME_BUCKET groupby for Diff/total columns, then window
    cumsum over HuisIdBSV + ProjectIdBSV to reconstruct cumulative counterparts
    for every Diff column in the config.

    Type validation: every aggregation column must be float, int, or boolean
    (boolean is cast to 0.0/1.0 in SUM/AVG SQL). Any other type triggers a
    ValueError listing all offending columns -- per ADR-003, data-correctness
    errors fail loudly. Errors are also logged individually so the operator
    sees the full picture in one run.

    Parameters
    ----------
    source_path : str
        Path to household_calculated.parquet.
    output_dir : str
        Directory where household_{interval}.parquet files are written.
    intervals : tuple
        Intervals to produce. Supported formats: Nmin (e.g. "5min", "15min", "60min")
        or Nh (e.g. "6h", "24h"). The "5min" interval is a column-selection passthrough
        with no aggregation; all others use TIME_BUCKET grouping.

    Raises
    ------
    ValueError
        If any column in the active aggregation config has an unsupported type
        (anything other than float, int, or boolean).
    """
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq
    from etdmap.data_model import get_aggregation_config

    config = get_aggregation_config()

    # Determine which config columns actually exist in the source file.
    _source_schema = pq.read_schema(source_path)
    source_schema_names = set(_source_schema.names)
    _schema_types = {field.name: field.type for field in _source_schema}
    active_config = {
        col: cfg for col, cfg in config.items() if col in source_schema_names
    }

    # Validate that every aggregation column has a supported type. The SUM/AVG
    # SQL casts each column to DOUBLE; that cast is meaningful only for
    # floating, integer, and boolean (boolean -> 0.0/1.0). Any other type
    # (string, timestamp, decimal, list, struct, ...) either errors at SQL
    # execution or produces nonsense, so we refuse to proceed and let the
    # operator either fix the source data or set AggregatieMeenemen=False in
    # etdmodel.csv. ADR-003: hard fail on data-correctness errors.
    type_errors: list[tuple[str, str]] = []
    for col in active_config:
        col_type = _schema_types.get(col)
        if col_type is None:
            continue
        if pa.types.is_floating(col_type) or pa.types.is_integer(col_type):
            continue
        if pa.types.is_boolean(col_type):
            continue  # intentional: boolean -> 0.0/1.0 in SUM/AVG SQL
        logging.error(
            f"resample_hh_data_duckdb: column '{col}' has unsupported type "
            f"'{col_type}' for aggregation."
        )
        type_errors.append((col, str(col_type)))
    if type_errors:
        raise ValueError(
            f"resample_hh_data_duckdb: {len(type_errors)} column(s) have "
            f"unsupported types for aggregation. Supported types are float, "
            f"int, and boolean. Either fix the source data types, or set "
            f"AggregatieMeenemen=False in etdmodel.csv. Offending columns: "
            + ", ".join(f"{c} ({t})" for c, t in type_errors)
        )

    # Diff columns whose cumulative counterpart should be rebuilt.
    diff_cols = [col for col in active_config if col.endswith("Diff")]
    cumul_cols = [col[:-4] for col in diff_cols]  # strip "Diff" suffix

    id_cols = ["HuisIdBSV", "ProjectIdBSV", "ReadingDate"]

    for interval in intervals:
        out_path = os.path.join(output_dir, f"household_{interval}.parquet")
        logging.info(f"resample_hh_data_duckdb: writing {interval} -> {out_path}")

        if interval == "5min":
            # Column selection only -- include config columns + existing cumulative counterparts.
            select_cols = id_cols + list(active_config.keys())
            for c in cumul_cols:
                if c in source_schema_names and c not in select_cols:
                    select_cols.append(c)
            # Double-quote identifiers so hyphenated names (e.g. "Zon-opwekTotaal") are
            # not parsed as arithmetic expressions by DuckDB.
            quoted = ", ".join(f'"{c}"' for c in select_cols)
            sql = (
                f"COPY (SELECT {quoted} FROM read_parquet({_sql_path(source_path)}))"
                f" TO {_sql_path(out_path)} (FORMAT PARQUET, COMPRESSION SNAPPY)"
            )
            with duckdb.connect() as con:
                con.execute(sql)

        else:
            bucket_minutes, bucket_expr = _parse_interval(interval)
            min_count = bucket_minutes // 5

            resample_exprs = [
                _resample_sql_expr(col, cfg["resample_method"], min_count)
                for col, cfg in active_config.items()
            ]
            resample_clause = ",\n            ".join(resample_exprs)

            cumsum_exprs = [
                f'SUM("{diff_col}") OVER ('
                f"PARTITION BY HuisIdBSV, ProjectIdBSV "
                f"ORDER BY ReadingDate "
                f"ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
                f') AS "{base_col}"'
                for diff_col, base_col in zip(diff_cols, cumul_cols)
            ]
            cumsum_clause = (",\n        " + ",\n        ".join(cumsum_exprs)) if cumsum_exprs else ""

            sql = f"""
            COPY (
                WITH resampled AS (
                    SELECT
                        HuisIdBSV, ProjectIdBSV,
                        TIME_BUCKET({bucket_expr}, ReadingDate) AS ReadingDate,
                        {resample_clause}
                    FROM read_parquet({_sql_path(source_path)})
                    GROUP BY HuisIdBSV, ProjectIdBSV,
                             TIME_BUCKET({bucket_expr}, ReadingDate)
                )
                SELECT *{cumsum_clause}
                FROM resampled
                ORDER BY HuisIdBSV, ProjectIdBSV, ReadingDate
            ) TO {_sql_path(out_path)} (FORMAT PARQUET, COMPRESSION SNAPPY)
            """
            with duckdb.connect() as con:
                con.execute(sql)

        logging.info(f"resample_hh_data_duckdb: done {interval}")


def aggregate_project_data_duckdb(
    intervals: tuple = ("5min", "15min", "60min"),
) -> None:
    """
    Aggregate resampled household parquet files to project level using DuckDB.

    Reads aggregation config from etdmap.data_model.get_aggregation_config() (ADR-004).
    One DuckDB pass per interval. Produces project_{interval}.parquet in
    etdtransform.options.aggregate_folder_path.

    Cumulative counterparts of Diff columns are rebuilt via window cumsum
    (PARTITION BY ProjectIdBSV ORDER BY ReadingDate) after the GROUP BY step,
    matching the convention in resample_hh_data_duckdb.

    Type validation: every aggregation column must be float, int, or boolean
    (same rule as resample_hh_data_duckdb). Any other type triggers a
    ValueError listing all offending columns for the current interval -- per
    ADR-003, data-correctness errors fail loudly.

    Parameters
    ----------
    intervals : tuple
        Intervals to aggregate; each element must match a household_{interval}.parquet
        file produced by a prior resample step.

    Raises
    ------
    ValueError
        If any column in the active aggregation config (per interval) has an
        unsupported type (anything other than float, int, or boolean).
    """
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq
    from etdmap.data_model import get_aggregation_config

    config = get_aggregation_config()
    output_dir = etdtransform.options.aggregate_folder_path

    # Diff columns whose cumulative counterpart should be rebuilt.
    diff_cols_all = [col for col in config if col.endswith("Diff")]
    cumul_cols_all = [col[:-4] for col in diff_cols_all]

    for interval in intervals:
        source_path = os.path.join(output_dir, f"household_{interval}.parquet")
        out_path = os.path.join(output_dir, f"project_{interval}.parquet")
        logging.info(f"aggregate_project_data_duckdb: writing {interval} -> {out_path}")

        _source_schema = pq.read_schema(source_path)
        source_schema_names = set(_source_schema.names)
        _schema_types = {field.name: field.type for field in _source_schema}
        active_config = {col: cfg for col, cfg in config.items() if col in source_schema_names}

        # Validate types -- same rule as resample_hh_data_duckdb. Float, int,
        # and boolean are aggregable; anything else is a data-correctness
        # error and is refused per ADR-003.
        type_errors: list[tuple[str, str]] = []
        for col in active_config:
            col_type = _schema_types.get(col)
            if col_type is None:
                continue
            if pa.types.is_floating(col_type) or pa.types.is_integer(col_type):
                continue
            if pa.types.is_boolean(col_type):
                continue  # intentional: boolean -> 0.0/1.0 in SUM/AVG SQL
            logging.error(
                f"aggregate_project_data_duckdb [{interval}]: column '{col}' has "
                f"unsupported type '{col_type}' for aggregation."
            )
            type_errors.append((col, str(col_type)))
        if type_errors:
            raise ValueError(
                f"aggregate_project_data_duckdb [{interval}]: {len(type_errors)} "
                f"column(s) have unsupported types for aggregation. Supported "
                f"types are float, int, and boolean. Either fix the source data "
                f"types, or set AggregatieMeenemen=False in etdmodel.csv. "
                f"Offending columns: "
                + ", ".join(f"{c} ({t})" for c, t in type_errors)
            )

        diff_cols = [col for col in active_config if col.endswith("Diff")]
        cumul_cols = [col[:-4] for col in diff_cols]

        agg_exprs = []
        for col, cfg in active_config.items():
            method = cfg["aggregate_method"]
            # CAST to DOUBLE handles boolean columns (True/False -> 1.0/0.0).
            if method == "avg":
                agg_exprs.append(f'AVG(CAST("{col}" AS DOUBLE)) AS "{col}"')
            elif method == "sum":
                agg_exprs.append(f'SUM(CAST("{col}" AS DOUBLE)) AS "{col}"')
            else:
                raise ValueError(f"Unknown aggregate_method '{method}' for column '{col}'")

        agg_clause = ",\n            ".join(agg_exprs)

        cumsum_exprs = [
            f'SUM("{diff_col}") OVER ('
            f"PARTITION BY ProjectIdBSV "
            f"ORDER BY ReadingDate "
            f"ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
            f') AS "{base_col}"'
            for diff_col, base_col in zip(diff_cols, cumul_cols)
        ]
        cumsum_clause = (",\n        " + ",\n        ".join(cumsum_exprs)) if cumsum_exprs else ""

        sql = f"""
        COPY (
            WITH aggregated AS (
                SELECT
                    ProjectIdBSV,
                    ReadingDate,
                    {agg_clause}
                FROM read_parquet({_sql_path(source_path)})
                GROUP BY ProjectIdBSV, ReadingDate
            )
            SELECT *{cumsum_clause}
            FROM aggregated
            ORDER BY ProjectIdBSV, ReadingDate
        ) TO {_sql_path(out_path)} (FORMAT PARQUET, COMPRESSION SNAPPY)
        """
        with duckdb.connect() as con:
            con.execute(sql)

        logging.info(f"aggregate_project_data_duckdb: done {interval}")


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
    for var, config in active_aggregation_variables(df).items():
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

    for var, config in active_aggregation_variables(df).items():
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


# Active aggregation map: variable -> {resample_method, aggregate_method}.
# Sourced from the etdmap data model (etdmodel.csv -> AggregatieMeenemen=True
# rows with their ResamplingMethode / AggregatieMethode columns). Update the
# Grist data model and re-sync etdmodel.csv to change what gets aggregated;
# do not maintain a separate hardcoded list here.
#
# Compatibility shim: keep both naming conventions for the Zon production
# variable in the dict so callers can look up by either name. The data model
# typically registers both `Zon-opwekTotaalDiff` (Grist-original, hyphenated)
# and `ZonopwekBruto` (runtime form used by SymPy / catalog code -- see
# has_zon_rename in this file and the same shim in calculated_columns.py).
# This idempotent additive shim ensures the alias still exists if either
# entry is later removed from etdmodel.csv. Extend with additional aliases
# in the same shape if other hyphenated names need similar treatment.
aggregation_variables = get_aggregation_config()
if (
    "Zon-opwekTotaalDiff" in aggregation_variables
    and "ZonopwekBruto" not in aggregation_variables
):
    aggregation_variables["ZonopwekBruto"] = aggregation_variables[
        "Zon-opwekTotaalDiff"
    ]
elif (
    "ZonopwekBruto" in aggregation_variables
    and "Zon-opwekTotaalDiff" not in aggregation_variables
):
    aggregation_variables["Zon-opwekTotaalDiff"] = aggregation_variables[
        "ZonopwekBruto"
    ]


def active_aggregation_variables(df: pd.DataFrame) -> dict:
    """
    Return the subset of `aggregation_variables` whose key is a column in df.

    Use this at the entry of any function that iterates `aggregation_variables`
    and accesses df[var]. Variables registered as `AggregatieMeenemen=True` in
    the data model that are not (yet) generated by the upstream pipeline -- or
    aliases like `Zon-opwekTotaalDiff` that exist in the model alongside their
    runtime form `ZonopwekBruto` -- are skipped here so the run does not
    crash on a missing column. Skipped variables are logged once per call.
    """
    active = {}
    skipped = []
    df_cols = set(df.columns)
    for var, config in aggregation_variables.items():
        if var in df_cols:
            active[var] = config
        else:
            skipped.append(var)
    if skipped:
        logging.warning(
            f"active_aggregation_variables: skipping {len(skipped)} variable(s) "
            f"not present in DataFrame: {sorted(skipped)}"
        )
    return active
