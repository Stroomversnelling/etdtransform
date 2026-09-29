import logging
import os
from enum import IntFlag, auto

import numpy as np
import pandas as pd
from etdmap.flags import (
    MappingFlag,
    bound_flags,
    frozen_spans,
)
from etdmap.data_model import interval_column
from etdmap.record_validators import thresholds_dict

from etdtransform.bounds import (
    column_bound,
    fill_bound,
    load_column_cadence,
)

__all__ = [
    'ImputeType',
    'apply_thresholds',
    'drop_temp_cols',
    'impute_and_normalize',
    'methods_to_bitwise',
    'process_gap_and_cumulative_groups',
    'process_imputation_vectorized'
]

def methods_to_bitwise(methods_column):
    """
    Convert methods to bitwise representation.

    This function takes a column of methods and converts each method to a bitwise
    representation. Each method is represented by a bit in the resulting integer.

    Parameters
    ----------
    methods_column : array-like
        A column containing lists of method numbers.

    Returns
    -------
    numpy.ndarray
        An array of integers where each integer represents the bitwise
        representation of the methods for that row.

    Notes
    -----
    The values in ``methods_column`` are ``ImputeType`` members, which is an
    ``IntFlag`` built with ``auto()``. Combining them is therefore a
    plain OR of the values.

    """

    bitwise_values = np.zeros(len(methods_column), dtype=np.int64)

    for i, methods in enumerate(methods_column):
        bitwise_value = 0
        for method in methods:
            if method > 0:
                bitwise_value |= int(method)
        bitwise_values[i] = bitwise_value

    return bitwise_values


def apply_thresholds(
    df,
    lower_bound,
    upper_bound,
    diff_col,
    avg_col,
    impute_type_col,
    is_imputed_col,
    detection_bound=None,
):
    """
    Apply thresholds to difference column and update imputation flags.

    This function applies lower and upper bounds to a difference column in the
    DataFrame. Values outside these bounds are replaced with average values,
    and corresponding imputation flags are updated.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data.
    lower_bound : float
        The lower threshold for the difference column.
    upper_bound : float
        The upper threshold for the difference column.
    diff_col : str
        The name of the difference column to apply thresholds to.
    avg_col : str
        The name of the column containing average values to use for imputation.
    impute_type_col : str
        The name of the column indicating the imputation type.
    is_imputed_col : str
        The name of the column indicating whether a value is imputed.
    detection_bound : float, pandas.Series or None, optional
        The threshold used to decide whether a value is implausible, when it
        differs from *upper_bound*. A filled value is judged against the grid
        threshold while a delivered value is judged against the column
        threshold, so the two differ wherever a column does not report at the
        grid cadence. Defaults to *upper_bound*.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with thresholds applied and imputation flags updated.

    Notes
    -----
    This function modifies the input DataFrame in-place and also returns it.
    Values outside the thresholds are replaced with average values and marked
    as imputed.
    """
    # Out-of-bounds rows, judged against the column threshold.
    mask = (
        bound_flags(
            df[diff_col],
            lower_bound,
            upper_bound if detection_bound is None else detection_bound,
        )
        != MappingFlag.NONE.value
    )

    # Catch-up readings are excluded from the mask: their volume is already
    # accounted for by the cumulative variable.
    catchup = _catchup_positions(
        df, diff_col, upper_bound if detection_bound is None else detection_bound
    )
    if catchup.any():
        logging.info(
            f"Keeping {int(catchup.sum())} catch-up value(s) in {diff_col}: "
            f"above the bound, but the cumulative variable accounts for them."
        )
        mask = mask & ~catchup

    # The average is the replacement, so where it is absent the pass blanks the
    # value instead of substituting one. The two outcomes carry different
    # action flags; THRESHOLD_VIOLATED records the condition either way.
    replaced = mask & df[avg_col].notna()
    blanked = mask & df[avg_col].isna()

    df.loc[mask, diff_col] = df.loc[mask, avg_col]
    df.loc[mask, is_imputed_col] = True
    df.loc[replaced, impute_type_col] = (
        df[impute_type_col].fillna(ImputeType.NONE.value)
        | ImputeType.THRESHOLD_ADJUSTED.value
        | ImputeType.THRESHOLD_VIOLATED.value
        | ImputeType.UNSCALED_AVERAGE.value
    )
    df.loc[blanked, impute_type_col] = (
        df[impute_type_col].fillna(ImputeType.NONE.value)
        | ImputeType.THRESHOLD_ADJUSTED.value
        | ImputeType.THRESHOLD_VIOLATED.value
        | ImputeType.NULLED_POST_IMPUTATION.value
    )

    return df


def _catchup_positions(df, diff_col, upper_bound) -> pd.Series:
    """
    Rows where an over-bound value is the catch-up of a frozen span.

    Calls the mapping-stage detector, ``etdmap.flags.frozen_spans``. A value
    counts as a catch-up only when a flat run of non-advancing readings sits
    directly before it.

    Runs per household, since a span is a property of one household's series.
    *upper_bound* is a scalar or one value per row; the bound is the same for
    every row of a household, so the first of them is the household's bound.
    """
    found = pd.Series(False, index=df.index)
    if df.empty:
        return found

    per_row = isinstance(upper_bound, pd.Series)
    if not per_row and pd.isna(upper_bound):
        return found

    group_column = "HuisIdBSV" if "HuisIdBSV" in df.columns else None
    groups = (
        df.groupby(group_column, sort=False) if group_column else [(None, df)]
    )
    for _, group in groups:
        bound = upper_bound.loc[group.index].iloc[0] if per_row else upper_bound
        if pd.isna(bound):
            continue
        spans = frozen_spans(
            group[diff_col].reset_index(drop=True), upper_bound=bound
        )
        if not spans.empty:
            found.loc[group.index[spans["catchup_position"].astype(int)]] = True

    return found


def impute_and_normalize(
    df: pd.DataFrame,
    cumulative_columns: list,
    project_id_column: str,
    max_bound: pd.DataFrame,
    cadence: pd.DataFrame | None = None,
):
    """
    Perform vectorized imputation and normalization on cumulative columns

    This function applies imputation techniques to fill missing values in
    cumulative columns and normalizes the data. It uses vectorized operations
    for improved performance.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data to be imputed and normalized.
    cumulative_columns : list
        A list of column names representing cumulative variables to be processed.
    project_id_column : str
        The name of the column containing project identifiers.
    max_bound : pandas.DataFrame
        A DataFrame containing maximum bounds for each variable.
    cadence : pandas.DataFrame, optional
        The per-column cadence table. Every threshold is scaled to the cadence
        the column reports at, so it is read from the mapped folder when not
        given, and an absent sidecar is an error rather than a grid assumption.

    Returns
    -------
    tuple
        A tuple containing three elements:

        - df : pandas.DataFrame
            The imputed and normalized DataFrame.

        - imputation_gap_stats_df : pandas.DataFrame
            Statistics about the imputation process for each gap.

        - imputation_reading_date_stats_df : None or pandas.DataFrame
            Statistics about imputation by reading date (if calculated).

    Notes
    -----
    This function applies various imputation methods based on the nature of
    the missing data and the available information. It handles different
    scenarios such as gaps in data, zero jumps, and negative jumps.

    The function also calculates and returns statistics about the imputation
    process, which can be useful for quality assessment.

    Warnings
    --------
    - The function may modify the input DataFrame in-place.
    - Imputation methods may introduce bias or affect the variance of the data.
    - Large amounts of imputed data may significantly affect analysis results.
    """

    logging.info("Starting to impute cumulative column diffs (vectorized).")

    # Read once per call: the bound for every column is scaled from this.
    _cadence = load_column_cadence() if cadence is None else cadence

    def calculate_imputation_gap_stats(group, cum_col, diff_col, impute_type_col):
        diff_column_total = group[diff_col].sum()
        cum_column_total_difference = group[cum_col].max() - group[cum_col].min()
        difference_in_calculation = diff_column_total - cum_column_total_difference
        missing_count = (~group["gap_length"].isna()).sum()
        methods = list(group[impute_type_col].unique().dropna())
        imputed_count = (group[impute_type_col].notna()).sum()
        imputed_na_count = group["cumulative_value_group"].notna().sum() - imputed_count

        return pd.Series(
            {
                "column": diff_col,
                "diff_col_total": diff_column_total,
                "cum_col_min_max_diff": cum_column_total_difference,
                "deviation": difference_in_calculation,
                "missing": missing_count,
                "methods": methods,
                "imputed": imputed_count,
                "imputed_na": imputed_na_count,
            },
        )

    imputation_gap_stats = []

    for cum_col in cumulative_columns:
        logging.info(f"Starting {cum_col} vectorized imputation")

        temp_cols = ["gap_length", "cumulative_value_group"]

        diff_col = interval_column(cum_col)
        old_diff_col = f"{cum_col}OldDiff"
        is_imputed_col = f"{diff_col}_is_imputed"
        impute_type_col = f"{diff_col}_impute_type"
        avg_col = f"{diff_col}_avg"

        if diff_col not in df.columns:
            logging.info(f"[impute] {diff_col} not in chunk columns, skipping.")
            continue

        df[old_diff_col] = df[diff_col]

        drop_temp_cols(df, logLeftoverError=True)

        if df[diff_col].isna().sum() == 0:
            logging.info(
                f"No values to impute in {diff_col}. Only checking thresholds.",
            )
            df[is_imputed_col] = False
            df[impute_type_col] = pd.Series(pd.NA, dtype="Int64", index=df.index)
            df["gap_length"] = pd.Series(pd.NA, dtype="Int64", index=df.index)
            df["cumulative_value_group"] = pd.Series(
                pd.NA,
                dtype="Int64",
                index=df.index,
            )
            logging.info("Apply thresholds to remove physically impossible outliers")
            df = apply_thresholds(
                df,
                lower_bound=thresholds_dict[diff_col]["Min"],
                upper_bound=fill_bound(
                    df, diff_col, _cadence, is_imputed_col=is_imputed_col
                ),
                detection_bound=column_bound(df, diff_col, _cadence),
                diff_col=diff_col,
                avg_col=avg_col,
                impute_type_col=impute_type_col,
                is_imputed_col=is_imputed_col,
            )

        else:
            logging.info(f"Defining {diff_col} gap groups")
            df = process_gap_and_cumulative_groups(
                df,
                diff_col=diff_col,
                cum_col=cum_col,
            )

            logging.info(f"Imputing {diff_col}")
            df = process_imputation_vectorized(
                df,
                diff_col=diff_col,
                cum_col=cum_col,
                avg_col=avg_col,
                impute_type_col=impute_type_col,
                is_imputed_col=is_imputed_col,
                cadence=_cadence,
            )

            # Every cell inside a gap should carry a value once imputation
            # has run.
            inside_a_gap = df["cumulative_value_group"].notna()
            remaining_na = int((df[diff_col].isna() & inside_a_gap).sum())

            if remaining_na > 0:
                without_average = int(
                    (df[diff_col].isna() & inside_a_gap & df[avg_col].isna()).sum()
                )
                logging.error(
                    f"{remaining_na} cell(s) inside a gap are still missing in "
                    f"column {diff_col} after imputation, {without_average} of "
                    f"them where no project average was available. Check masks "
                    f"and imputation logic.",
                )

        logging.info(f"Calculating {diff_col} imputation gap stats")
        _grp = df.groupby([project_id_column, "HuisIdBSV"])
        # One C-level pass for all scalar stats; apply only for the methods list.
        _stats = _grp.agg(
            diff_col_total=(diff_col, "sum"),
            _cum_max=(cum_col, "max"),
            _cum_min=(cum_col, "min"),
            missing=("gap_length", "count"),
            imputed=(impute_type_col, "count"),
            _cvg_count=("cumulative_value_group", "count"),
        ).reset_index()
        _stats["cum_col_min_max_diff"] = _stats["_cum_max"] - _stats["_cum_min"]
        _stats["deviation"] = _stats["diff_col_total"] - _stats["cum_col_min_max_diff"]
        _stats["imputed_na"] = _stats["_cvg_count"] - _stats["imputed"]
        _stats["column"] = diff_col
        _stats["methods"] = (
            _grp[impute_type_col]
            .apply(lambda x: sorted(set(v for v in x if pd.notna(v))))
            .values
        )
        _stats.drop(columns=["_cum_max", "_cum_min", "_cvg_count"], inplace=True)
        imputation_gap_stats.append(_stats)

        drop_temp_cols(df, temp_cols=["gap_length", "cumulative_value_group"])

        drop_temp_cols(df, logLeftoverError=True)

    if not imputation_gap_stats:
        raise RuntimeError(
            "No columns were imputed -- all Diff columns were absent from this chunk. "
            "Check that prepare_diff_averages_for_impute has run and that cum_cols matches the source data."
        )
    imputation_gap_stats_df = pd.concat(imputation_gap_stats, ignore_index=True)
    imputation_gap_stats_df["bitwise_methods"] = methods_to_bitwise(
        imputation_gap_stats_df["methods"],
    )

    imputation_reading_date_stats_df = None

    return df, imputation_gap_stats_df, imputation_reading_date_stats_df


def drop_temp_cols(
    df,
    temp_cols=None,
    logLeftoverError=False,
):
    """
    Drop temporary columns from the DataFrame.

    This function removes specified temporary columns from the DataFrame.
    If no columns are specified, it drops a predefined set of temporary columns.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame from which to drop columns.
    temp_cols : list, optional
        A list of column names to drop. If None, a default set of temporary
        columns will be used.
    logLeftoverError : bool, optional
        If True, log an error message for any leftover columns to be removed.

    Notes
    -----
    This function modifies the DataFrame in-place.

    The default set of temporary columns includes various intermediate
    calculation columns used in the imputation process.

    Warnings
    --------
    - If logLeftoverError is True and there are columns to be dropped, an error
      message will be logged, which might indicate unintended remnants in the
      data processing pipeline.
    """

    if temp_cols is None:
        temp_cols = [
            "gap_start",
            "gap_group",
            "gap_jump_is_na_mask",
            "gap_jump",
            "house_impute_factor",
            "avg_na",
            "impute_jump",
            "impute_na",
            "impute_values",
            "impute_na_ratio",
            "cum_value_encountered",
            "prev_cum_value",
            "end_cum_value",
            "impute_type_old",
            "gap_length",
            "cumulative_value_group",
            "no_diff_mask"
            ]


    cols_to_drop = [col for col in temp_cols if col in df.columns]

    if logLeftoverError and len(cols_to_drop) > 0:
        logging.error(
            f"There are some leftover columns to remove from the code: {cols_to_drop}",
        )

    df.drop(columns=cols_to_drop, inplace=True)


def process_gap_and_cumulative_groups(df, diff_col, cum_col):
    """
    Process gap and cumulative value groups in the DataFrame.

    This function identifies gaps in the data, creates gap groups, and
    establishes cumulative value groups based on the presence of NA values
    and transitions between households.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to process.
    diff_col : str
        The name of the difference column to analyze for gaps.
    cum_col : str
        The name of the cumulative column to use for value grouping.

    Returns
    -------
    pandas.DataFrame
        The processed DataFrame with added columns for gap and cumulative
        value grouping.

    Notes
    -----
    This function adds several temporary columns to the DataFrame:
    - 'gap_start': Identifies the start of a diff column gap or transition between households.
    - 'gap_group': Groups consecutive NA values in diff columns.
    - 'cum_value_encountered': Marks where a non-NA value is encountered in cumulative column.
    - 'cumulative_value_group': Groups gaps based on cumulative values.
    - 'gap_length': The length of each gap group.

    These columns are crucial for the subsequent imputation process. It returns only for further imputation:
    - 'cumulative_value_group'
    - 'gap_length'

    Warnings
    --------
    - This function modifies the input DataFrame in-place.
    - The added columns should be handled carefully in subsequent processing steps.
    """
    temp_cols = ["gap_start", "gap_group", "cum_value_encountered"]

    # Step 1: Identify NA values in diff_col
    is_na_mask = df[diff_col].isna()

    # Step 2: Identify gap start only for NA values and transitions between households
    df["gap_start"] = (is_na_mask & (~is_na_mask.shift(1, fill_value=False))) | (
        (df["HuisIdBSV"] != df["HuisIdBSV"].shift(1)) & is_na_mask
    )

    # Step 3: Create gap groups by cumulative sum, but now ensure it doesn't increment unnecessarily at house boundaries
    df["gap_group"] = df["gap_start"].cumsum()

    # Step 4: Ensure non-NA values are not included in any gap group
    df["gap_group"] = df["gap_group"].mask(~is_na_mask, pd.NA)

    # Step 5: Mark the end of a group where a non-NA value is encountered in cum_col
    df["cum_value_encountered"] = df[cum_col].notna() & is_na_mask

    # Step 6: Adjust cumulative group logic to handle gaps without cumulative values
    df["cumulative_value_group"] = (
        df["cum_value_encountered"].shift(1, fill_value=False) | df["gap_start"]
    ).cumsum()

    # Step 7: Ensure that the group continues when there is no cumulative value in the gap
    df["cumulative_value_group"] = df["cumulative_value_group"].ffill()

    # Step 8: Ensure non-NA values in diff_col don't belong to a group

    # Combine all conditions
    exclude_mask = ~is_na_mask

    df["cumulative_value_group"] = df["cumulative_value_group"].mask(
        exclude_mask,
        pd.NA,
    )

    # Step 7: Add the count of records in each group consistently for all rows in the group
    df["gap_length"] = df.groupby("cumulative_value_group").transform("size")

    drop_temp_cols(df, temp_cols=temp_cols)

    return df


def process_imputation_vectorized(
    df,
    diff_col,
    cum_col,
    avg_col,
    impute_type_col,
    is_imputed_col,
    cadence=None,
):
    """
    Perform vectorized imputation on the DataFrame.

    This function applies various imputation methods to fill missing values
    in the difference column based on cumulative values and average differences.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to impute.
    diff_col : str
        The name of the difference column to impute.
    cum_col : str
        The name of the cumulative column used for imputation.
    avg_col : str
        The name of the column containing average differences.
    impute_type_col : str
        The name of the column to store imputation type.
    is_imputed_col : str
        The name of the column to indicate whether a value is imputed.
    cadence : pandas.DataFrame, optional
        The per-column cadence table from
        :func:`etdtransform.bounds.load_column_cadence`, used to judge a
        delivered value against the interval it covers. Without it every value
        is judged against the grid threshold.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with imputed values and additional columns indicating
        imputation types and statistics.

    Notes
    -----
    This function applies several imputation methods for diff columns:
    - Filling with zeros for flat or near-zero gaps
    - Linear filling for positive gaps with near-zero impute jumps
    - Scaled impute value filling for positive gaps with positive impute jumps
    - Handling cases with no gap jump (e.g., at the start or end of the dataset)

    The function also applies thresholds to remove physically impossible outliers.

    Warnings
    --------
    - This function modifies the input DataFrame in-place.
    - The imputation process may introduce bias, especially in cases with large gaps
      or when a significant portion of the data is imputed.
    - The function assumes that the input data has been properly prepared and sorted.
    """
    def setup_prev_value_columns(df, cum_col):
        # Shift by one to get the immediate previous time step value before the group and make sure no other time step is there
        df["prev_cum_value"] = df[cum_col].shift(1)

        first_in_group_mask = (
            df["cumulative_value_group"] != df["cumulative_value_group"].shift(1)
        ) & ~(df["cumulative_value_group"].isna())

        first_in_house = df["HuisIdBSV"] != df["HuisIdBSV"].shift(1)

        # remove all values that are not the first in the group or that are from another household
        df.loc[
            ~first_in_group_mask | df["cumulative_value_group"].isna() | first_in_house,
            "prev_cum_value",
        ] = pd.NA

        # Remove negative previous values, as they are not valid for imputation
        df.loc[df["prev_cum_value"] < 0, "prev_cum_value"] = pd.NA

        # fill for each group
        df["prev_cum_value"] = df.groupby("cumulative_value_group")[
            "prev_cum_value"
        ].ffill()

        return df

    def ensure_Float64(df, columns=["CumCol", "CumColDiff", "diff_avg"]):
        for col in columns:
            if df[col].dtype != "Float64":
                logging.warning(
                    f'{col} dtype is "{df[col].dtype}". Attempting to convert.',
                )
                try:
                    # Attempt to convert the column to Float64
                    df[col] = df[col].astype("Float64")
                    logging.info(f"{col} successfully converted to Float64.")
                except Exception as e:
                    logging.error(f"Failed to convert {col} to Float64: {e}")
                    raise e
        return df

    def setup_gap_stats(df):
        df["gap_jump"] = df["end_cum_value"] - df["prev_cum_value"]
        df["gap_jump_is_na_mask"] = (
            df["gap_jump"].isna() & df["cumulative_value_group"].notna()
        )

        # df['no_diff_mask'] = df['cumulative_value_group'].notna() & (df['prev_cum_value'].isna() | df['end_cum_value'].isna() | (df['gap_jump'] <= 0))

        return df

    def setup_impute_columns(df, avg_col, impute_type_col, is_imputed_col):
        # add impute jump and tracking of missing values
        df["avg_na"] = df[avg_col].isna()
        df["impute_na"] = df.groupby("cumulative_value_group")["avg_na"].transform(
            "sum",
        )
        df["impute_values"] = df[avg_col]  # pd.NA where avg unavailable; do not substitute 0

        if (df["impute_values"] < 0).any():
            raise Exception("Negative impute jump - not allowed - check averages")

        df["impute_na_ratio"] = df["impute_na"] / df["gap_length"]
        df["impute_jump"] = df.groupby("cumulative_value_group")[
            "impute_values"
        ].transform("sum")

        # How many cells of the gap the average has a value for. The fill
        # routes on this count, not on the summed average.
        df["impute_present"] = df["gap_length"] - df["impute_na"]

        # Cells of a partly-covered gap that have no average take the gap's own
        # mean. Both terms are reductions already computed above.
        partial = (df["impute_na"] > 0) & (df["impute_present"] > 0)
        gap_mean = df["impute_jump"] / df["impute_present"]
        df["impute_values"] = df["impute_values"].where(
            ~(partial & df["avg_na"]), gap_mean
        )
        # Recomputed after the substitution so it matches impute_values.
        df["impute_jump"] = df["impute_jump"].where(
            ~partial, df["impute_jump"] * df["gap_length"] / df["impute_present"]
        )

        # calculate cumulative total for household for scaling the averages (optional)
        df[is_imputed_col] = False
        df[impute_type_col] = pd.Series(pd.NA, dtype="Int64", index=df.index)

        return df

    def setup_proportional_adjustment_to_impute(df, diff_col, avg_col):
        # Calculate the proportional adjustment to impute averages
        df["comparable_to_impute_mask"] = (
            ~df["avg_na"] & df[diff_col].notna() & (df[diff_col] >= 0)
        )

        df["diff_avg_sum"] = (
            df[avg_col]
            .where(df["comparable_to_impute_mask"])
            .groupby(df["HuisIdBSV"])
            .transform("sum")
        )
        df["cum_diff_sum"] = (
            df[diff_col]
            .where(df["comparable_to_impute_mask"])
            .groupby(df["HuisIdBSV"])
            .transform("sum")
        )

        comparable_counts = df.groupby("HuisIdBSV")[
            "comparable_to_impute_mask"
        ].transform("sum")
        total_counts = df.groupby("HuisIdBSV")[diff_col].transform("size")
        not_enough_comparable = comparable_counts <= (total_counts / 2)

        # The household's level relative to the project average: 1.0 matches the
        # average, 2.0 is twice it. Multiplies the average to bring it to this
        # household's level.
        df["house_impute_factor"] = (df["cum_diff_sum"] / df["diff_avg_sum"]).replace(
            [float("inf"), -float("inf")],
            pd.NA,
        )
        # True where the factor below is the neutral 1.0 rather than a value
        # computed from the household's own rows. The fill branches read it to
        # choose between the scaled and unscaled impute types.
        df["house_factor_is_fallback"] = (
            df["house_impute_factor"].isna() | not_enough_comparable
        )
        df["house_impute_factor"] = (
            df["house_impute_factor"].fillna(1.0).mask(not_enough_comparable, 1.0)
        )

        return df

    temp_cols = [
        "gap_jump_is_na_mask",
        "gap_jump",
        "house_impute_factor",
        "house_factor_is_fallback",
        "avg_na",
        "impute_jump",
        "impute_na",
        "impute_values",
        "impute_na_ratio",
        "cum_value_encountered",
        "prev_cum_value",
        "end_cum_value",
        "impute_type_old",
    ]

    logging.info(f"Ensuring {[cum_col, diff_col, avg_col]} are Float64")

    df = ensure_Float64(df, columns=[cum_col, diff_col, avg_col])

    logging.info("Setting up previous value column")
    df = setup_prev_value_columns(df, cum_col)

    df["end_cum_value"] = df.groupby("cumulative_value_group")[cum_col].transform(
        "last",
    )
    df.loc[df["end_cum_value"] < 0, "end_cum_value"] = pd.NA

    logging.info("Setting up gap calculations")
    df = setup_gap_stats(df)

    logging.info("Setting up impute columns")
    df = setup_impute_columns(
        df,
        avg_col=avg_col,
        impute_type_col=impute_type_col,
        is_imputed_col=is_imputed_col,
    )

    logging.info("Setting up proportional adjustment to mpute")
    df = setup_proportional_adjustment_to_impute(df, diff_col=diff_col, avg_col=avg_col)

    def impute_with_gap_jump(df, diff_col, is_imputed_col, impute_type_col):
        has_gap_jump_mask = (
            ~df["gap_jump_is_na_mask"] & df["cumulative_value_group"].notna()
        )

        ## if has gap jump
        # negative gap jump
        flat_gap_jump_mask = has_gap_jump_mask & (df["gap_jump"] < 0)
        df.loc[flat_gap_jump_mask, is_imputed_col] = True
        df.loc[flat_gap_jump_mask, diff_col] = 0
        df.loc[flat_gap_jump_mask, impute_type_col] = (
            ImputeType.NEGATIVE_GAP_JUMP.value | ImputeType.ZERO_FILL.value
        )

        # gap jump near zero - fill with zeros
        flat_gap_jump_mask = (
            has_gap_jump_mask & (df["gap_jump"] >= 0) & (df["gap_jump"] < 1e-8)
        )
        df.loc[flat_gap_jump_mask, is_imputed_col] = True
        df.loc[flat_gap_jump_mask, diff_col] = 0
        df.loc[flat_gap_jump_mask, impute_type_col] = (
            ImputeType.NEAR_ZERO_GAP_JUMP.value | ImputeType.ZERO_FILL.value
        )

        # No average for any cell of the gap, or an average that is present but
        # everywhere zero: no shape to distribute by, so the jump is spread
        # evenly.
        no_shape = (df["impute_present"] <= 0) | (df["impute_jump"] < 1e-8)
        positive_gap_linear_mask = (
            has_gap_jump_mask & (df["gap_jump"] >= 1e-8) & no_shape
        )
        df.loc[positive_gap_linear_mask, is_imputed_col] = True
        df.loc[positive_gap_linear_mask, diff_col] = round(
            df["gap_jump"] / df["gap_length"],
            10,
        )
        df.loc[positive_gap_linear_mask, impute_type_col] = ImputeType.UNIFORM_FILL.value

        # A shape exists for at least one cell of the gap, so the jump is
        # distributed by it. Cells without an average carry the gap's own mean
        # by this point.
        positive_gap_scaled_mask = (
            has_gap_jump_mask & (df["gap_jump"] >= 1e-8) & ~no_shape
        )
        _can_scaled = positive_gap_scaled_mask & df["impute_values"].notna()
        df.loc[_can_scaled, is_imputed_col] = True
        df.loc[_can_scaled, diff_col] = round(
            df.loc[_can_scaled, "impute_values"]
            * (
                df.loc[_can_scaled, "gap_jump"]
                / df.loc[_can_scaled, "impute_jump"]
            ),
            10,
        )
        df.loc[_can_scaled, impute_type_col] = ImputeType.SCALED_FILL.value

        return df

    logging.info("Impute with gap jump")
    df = impute_with_gap_jump(
        df,
        diff_col=diff_col,
        is_imputed_col=is_imputed_col,
        impute_type_col=impute_type_col,
    )

    def impute_wo_gap_jump(df, diff_col, impute_type_col, is_imputed_col):
        wo_gap_jump_mask = (
            df["gap_jump_is_na_mask"] & df["cumulative_value_group"].notna()
        )

        ###
        ## else no gap jump:
        # no starting value for gap and no end value - throw an exception
        nogpjump_no_start_no_end_value_mask = (
            wo_gap_jump_mask & df["end_cum_value"].isna() & df["prev_cum_value"].isna()
        )
        if nogpjump_no_start_no_end_value_mask.any():
            # add_diff_columns seeds row 0 of each household with 0, so that row
            # is excluded when testing whether the household delivers intervals.
            _first_in_house = df["HuisIdBSV"] != df["HuisIdBSV"].shift(1)
            house_no_diff_values = (
                (df[diff_col].notna() & ~_first_in_house)
                .groupby(df["HuisIdBSV"])
                .transform("sum")
                == 0
            )

            # Neither end of the gap has a cumulative value. The household
            # delivers this variable as intervals only when the interval column
            # holds values and the cumulative does not.
            _interval_only = (
                nogpjump_no_start_no_end_value_mask
                & ~house_no_diff_values
                & df["impute_values"].notna()
            )
            # No cumulative reading bounds the gap, so house_impute_factor is
            # the only scale available. Fill where it was computed from the
            # household's own rows; leave the cells empty where it fell back.
            _fallback = df["house_factor_is_fallback"]
            _fill_io = _interval_only & ~_fallback
            _skip_io = _interval_only & _fallback

            if _fill_io.any():
                logging.info(
                    f"{diff_col}: filling {int(_fill_io.sum())} interval(s) from "
                    f"the project average scaled to the household -- the variable "
                    f"is delivered as intervals only for this household."
                )
                df.loc[_fill_io, is_imputed_col] = True
                df.loc[_fill_io, diff_col] = (
                    df.loc[_fill_io, "impute_values"]
                    * df.loc[_fill_io, "house_impute_factor"]
                )
                df.loc[_fill_io, impute_type_col] = (
                    ImputeType.NO_CUMULATIVE_VALUE.value
                    | ImputeType.HOUSEHOLD_SCALED_AVERAGE.value
                )

            if _skip_io.any():
                logging.warning(
                    f"{diff_col}: leaving {int(_skip_io.sum())} interval(s) "
                    f"unfilled -- no cumulative reading bounds them and the "
                    f"household factor has too few comparable rows to scale with."
                )
                df.loc[_skip_io, impute_type_col] = (
                    ImputeType.NO_CUMULATIVE_VALUE.value
                    | ImputeType.NOT_FILLED.value
                )

            _truly_empty = nogpjump_no_start_no_end_value_mask & house_no_diff_values
            if _truly_empty.any():
                logging.error(
                    f"{diff_col}: no cumulative and no interval values for "
                    f"{int(df.loc[_truly_empty, 'HuisIdBSV'].nunique())} household(s). "
                    f"Nothing to impute from."
                )

        ### has end value but no start value for gap
        nogpjump_has_end_value_mask = (
            wo_gap_jump_mask & ~df["end_cum_value"].isna() & df["prev_cum_value"].isna()
        )

        #### has end value is 0 - fill with 0 - type 6
        nogpjump_has_end_value_zero_mask = nogpjump_has_end_value_mask & (
            df["end_cum_value"] < 1e-8
        )
        # DISABLED, deliberately reversible. Filling zeros here says the meter
        # stood at zero for the whole gap, which follows only if it had never
        # advanced. With no opening reading there is nothing to support that:
        # the cumulative is pd.NA, not zero. Left in place so the previous
        # behaviour can be restored by uncommenting.
        #
        # df.loc[nogpjump_has_end_value_zero_mask, is_imputed_col] = True
        # df.loc[nogpjump_has_end_value_zero_mask, diff_col] = 0
        # df.loc[nogpjump_has_end_value_zero_mask, impute_type_col] = (
        #     ImputeType.ZERO_END_VALUE.value
        #     | ImputeType.ONLY_CLOSING_READING.value
        #     | ImputeType.ZERO_FILL.value
        # )
        if nogpjump_has_end_value_zero_mask.any():
            df.loc[nogpjump_has_end_value_zero_mask, impute_type_col] = (
                ImputeType.ZERO_END_VALUE.value
                | ImputeType.ONLY_CLOSING_READING.value
                | ImputeType.NOT_FILLED.value
            )

        #### end value > 0 - fill with impute values - type 7 (only when avg available)
        nogpjump_has_end_value_positive_mask = nogpjump_has_end_value_mask & (
            df["end_cum_value"] > 1e-8
        )
        # Only a closing reading bounds this gap, which is the mirror of the
        # opening-reading case below and takes the same treatment: scale the
        # average to the household where the factor rests on enough of the
        # series, and write nothing where it does not.
        _can_fill_ev = nogpjump_has_end_value_positive_mask & df["impute_values"].notna()
        _ev_fallback = df["house_factor_is_fallback"]
        _fill_ev = _can_fill_ev & ~_ev_fallback
        _skip_ev = _can_fill_ev & _ev_fallback

        df.loc[_fill_ev, is_imputed_col] = True
        df.loc[_fill_ev, diff_col] = (
            df.loc[_fill_ev, "impute_values"]
            * df.loc[_fill_ev, "house_impute_factor"]
        )
        df.loc[_fill_ev, impute_type_col] = (
            ImputeType.POSITIVE_END_VALUE.value
            | ImputeType.ONLY_CLOSING_READING.value
            | ImputeType.HOUSEHOLD_SCALED_AVERAGE.value
        )

        if _skip_ev.any():
            logging.warning(
                f"{diff_col}: leaving {int(_skip_ev.sum())} interval(s) unfilled "
                f"-- the gap has no opening reading and the household factor has "
                f"too few comparable rows to scale with."
            )
            df.loc[_skip_ev, impute_type_col] = (
                ImputeType.POSITIVE_END_VALUE.value
                | ImputeType.ONLY_CLOSING_READING.value
                | ImputeType.NOT_FILLED.value
            )

        #### end value < 0 - raise exception
        if (df["end_cum_value"] < 0).any():
            raise Exception(
                "Negative next value at end of gap - that is not supposed to happen!",
            )

        ### start value but no end value for gap - fill with impute values (only when avg available)
        nogpjump_has_start_value_mask = (
            wo_gap_jump_mask & df["end_cum_value"].isna() & ~df["prev_cum_value"].isna()
        )
        # The gap opens on a reading and never closes, so again only
        # house_impute_factor scales it. Same rule as the no-cumulative case.
        _can_fill_sv = nogpjump_has_start_value_mask & df["impute_values"].notna()
        _sv_fallback = df["house_factor_is_fallback"]
        _fill_sv = _can_fill_sv & ~_sv_fallback
        _skip_sv = _can_fill_sv & _sv_fallback

        df.loc[_fill_sv, is_imputed_col] = True
        df.loc[_fill_sv, diff_col] = (
            df.loc[_fill_sv, "impute_values"]
            * df.loc[_fill_sv, "house_impute_factor"]
        )
        df.loc[_fill_sv, impute_type_col] = (
            ImputeType.NO_END_VALUE.value
            | ImputeType.ONLY_OPENING_READING.value
            | ImputeType.HOUSEHOLD_SCALED_AVERAGE.value
        )

        if _skip_sv.any():
            logging.warning(
                f"{diff_col}: leaving {int(_skip_sv.sum())} interval(s) unfilled "
                f"-- the gap has no closing reading and the household factor has "
                f"too few comparable rows to scale with."
            )
            df.loc[_skip_sv, impute_type_col] = (
                ImputeType.NO_END_VALUE.value
                | ImputeType.ONLY_OPENING_READING.value
                | ImputeType.NOT_FILLED.value
            )

        return df

    logging.info("Impute without gap jump")
    df = impute_wo_gap_jump(
        df,
        diff_col=diff_col,
        impute_type_col=impute_type_col,
        is_imputed_col=is_imputed_col,
    )

    # Gap cells no branch above wrote to. Row 0 of a household bounds no gap,
    # so it is excluded.
    _first_in_house = df["HuisIdBSV"] != df["HuisIdBSV"].shift(1)
    _unfilled = (
        df[diff_col].isna()
        & df["cumulative_value_group"].notna()
        & ~_first_in_house
    )
    if _unfilled.any():
        df.loc[_unfilled, impute_type_col] = (
            df[impute_type_col].fillna(ImputeType.NONE.value)
            | ImputeType.NOT_FILLED.value
        )
        logging.warning(
            f"{diff_col}: {int(_unfilled.sum())} cell(s) inside a gap were not "
            f"filled and carry NOT_FILLED."
        )

    logging.info("Apply thresholds to remove physically impossible outliers")
    df = apply_thresholds(
        df,
        lower_bound=thresholds_dict[diff_col]["Min"],
        upper_bound=fill_bound(
            df, diff_col, cadence, is_imputed_col=is_imputed_col
        ),
        detection_bound=column_bound(df, diff_col, cadence),
        diff_col=diff_col,
        avg_col=avg_col,
        impute_type_col=impute_type_col,
        is_imputed_col=is_imputed_col,
    )

    # Households without a cumulative sum above 0 or all cum_col values are NA
    house_no_cum_sum = (
        df.groupby("HuisIdBSV")[cum_col].transform("sum").fillna(0) <= 0
    ) | (df.groupby("HuisIdBSV")[cum_col].transform("count") == 0)

    # Households where max - min cumulative sum is not > 0
    houses_cum_min_max = (
        df[cum_col].groupby(df["HuisIdBSV"]).transform("max")
        - df[cum_col].groupby(df["HuisIdBSV"]).transform("min")
    ).fillna(0) <= 0

    drop_temp_cols(df, temp_cols=temp_cols)

    return df


class ImputeType(IntFlag):
    """
    Enumeration of imputation types used in the vectorized imputation process.

    This class defines the different types of imputation methods applied
    during the vectorized imputation process for handling missing or
    problematic data in time series.

    Attributes
    ----------
    NONE : int
        Represents no imputation.
    NEGATIVE_GAP_JUMP : int
        Represents a negative gap jump. Fills with zeros (potentially a meter reset)
    NEAR_ZERO_GAP_JUMP : int
        Represents a gap jump near zero. Fills with zeros (no change).
    UNIFORM_FILL : int
        Represents a linear fill for positive gaps with near-zero impute jumps based on average.
    SCALED_FILL : int
        Represents a scaled fill for positive gaps with positive impute jumps based on average.
    ZERO_END_VALUE : int
        Represents imputation when end value is zero and there is no start value. Fills with zeros.
    POSITIVE_END_VALUE : int
        Represents imputation when end value is positive but there is no start value. Fills with averages.
    NO_END_VALUE : int
        Represents imputation when there is no end value. Fills with averages.
    THRESHOLD_ADJUSTED : int
        Represents values adjusted due to threshold violations. This happens after imputation and could be triggered by imputed values.
    ZERO_FILL : int
        A zero was written. Set by the three conditions that fill with zeros.
    UNSCALED_AVERAGE : int
        The project average was written as it stands, with no household factor
        applied. Also set where a household factor was intended but fell back to
        the neutral 1.0 for want of comparable rows.
    HOUSEHOLD_SCALED_AVERAGE : int
        The project average was written multiplied by a household factor that
        was computed from that household's own comparable rows.
    NOT_FILLED : int
        A cell inside a gap that no branch wrote to, left as pd.NA.
    NO_CUMULATIVE_VALUE : int
        Represents imputation of a variable delivered as intervals only, where
        neither end of the gap has a cumulative value to scale against. Fills
        with averages.

    Notes
    -----
    The THRESHOLD_ADJUSTED flag can be combined with other imputation types.

    The members above THRESHOLD_ADJUSTED name the condition that led to a fill;
    those below it name the action taken. Every cell that was written carries
    exactly one action flag, so a reader can select on what happened to a value
    without enumerating the conditions that can lead there.

    .. :no-index:
    """

    NONE = 0

    # --- signals: the condition the cell was in ---

    NEGATIVE_GAP_JUMP = auto()
    NEAR_ZERO_GAP_JUMP = auto()
    UNIFORM_FILL = auto()
    SCALED_FILL = auto()
    ZERO_END_VALUE = auto()
    POSITIVE_END_VALUE = auto()
    NO_END_VALUE = auto()

    # DEPRECATED. An alias of THRESHOLD_VIOLATED in practice: the threshold pass
    # sets both on the same rows, so this adds nothing THRESHOLD_VIOLATED and the
    # action flags do not already say. It predates the split of condition from
    # action and is kept because its bit is written into every dataset. Read
    # THRESHOLD_VIOLATED for the condition and the action flags for the outcome.
    THRESHOLD_ADJUSTED = auto()

    # --- actions: what imputation did about it ---

    ZERO_FILL = auto()
    UNSCALED_AVERAGE = auto()
    HOUSEHOLD_SCALED_AVERAGE = auto()
    NOT_FILLED = auto()

    # --- appended: signals ---

    # No opening cumulative reading bounds the gap; the closing one does.
    # Replaces the pair ZERO_END_VALUE / POSITIVE_END_VALUE, which split the
    # same condition by the closing value rather than by what is known.
    ONLY_CLOSING_READING = auto()

    # An opening cumulative reading bounds the gap; no closing one does.
    ONLY_OPENING_READING = auto()

    # The value lies outside the bound the threshold pass judges it against.
    # The condition only; what the pass then does is an action of its own.
    # Supersedes THRESHOLD_ADJUSTED, which is set on the same rows.
    THRESHOLD_VIOLATED = auto()

    # --- appended: actions ---

    # Imputation blanked the cell.
    NULLED = auto()

    # The threshold pass blanked the cell after imputation had written it.
    # Separate from NULLED because it is a different pass acting on a value
    # imputation had already recorded a decision about.
    NULLED_POST_IMPUTATION = auto()

    # --- appended: signals ---

    # Neither end of the gap has a cumulative reading, which is the state of a
    # variable delivered as intervals only.
    NO_CUMULATIVE_VALUE = auto()

    # --- composites ---

    # The two categories, declared rather than left to a comment. A signal names
    # the condition a cell was in; an action names what imputation did about it.
    # Every single-bit member belongs to exactly one, which a test pins, so a
    # report can label any flag without keeping its own list.
    SIGNALS = (
        NEGATIVE_GAP_JUMP
        | NEAR_ZERO_GAP_JUMP
        | UNIFORM_FILL
        | SCALED_FILL
        | ZERO_END_VALUE
        | POSITIVE_END_VALUE
        | NO_END_VALUE
        | THRESHOLD_ADJUSTED
        | ONLY_CLOSING_READING
        | ONLY_OPENING_READING
        | THRESHOLD_VIOLATED
        | NO_CUMULATIVE_VALUE
    )
    ACTIONS = (
        ZERO_FILL
        | UNSCALED_AVERAGE
        | HOUSEHOLD_SCALED_AVERAGE
        | NOT_FILLED
        | NULLED
        | NULLED_POST_IMPUTATION
    )

    # Imputation wrote a value. Selects an imputed number without naming every
    # fill method.
    FILLED = ZERO_FILL | UNSCALED_AVERAGE | HOUSEHOLD_SCALED_AVERAGE

    # Every cell imputation changed, whichever way. NOT_FILLED is excluded:
    # nothing was written there.
    IMPUTED_VALUE = FILLED | NULLED | NULLED_POST_IMPUTATION
