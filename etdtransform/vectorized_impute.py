import logging
from enum import IntFlag, auto

import numpy as np
import pandas as pd
from etdmap.record_validators import thresholds_dict


def methods_to_bitwise_vectorized(methods_column):
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
    The function assumes that method numbers start from 1 and correspond to
    bit positions (method 1 = bit 0, method 2 = bit 1, etc.).

    This vectorized version is optimized for performance with NumPy.
    """
    bitwise_values = np.zeros(len(methods_column), dtype=np.int64)

    for i, methods in enumerate(methods_column):
        bitwise_value = 0
        for method in methods:
            if method > 0:
                bitwise_value |= 1 << (
                    method - 1
                )  # Same logic as before, but faster with NumPy
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
    mask = ((df[diff_col] < lower_bound) | (df[diff_col] > upper_bound)) & df[
        diff_col
    ].notna()
    df.loc[mask, diff_col] = df.loc[mask, avg_col]
    df.loc[mask, is_imputed_col] = True
    df.loc[mask, impute_type_col] = df[impute_type_col].fillna(ImputeType.NONE) | ImputeType.THRESHOLD_ADJUSTED

    return df


def impute_and_normalize_vectorized(
    df: pd.DataFrame,
    cumulative_columns: list,
    project_id_column: str,
    max_bound: pd.DataFrame,
):
    """
    Perform vectorized imputation and normalization on cumulative columns.

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

        diff_col = f"{cum_col}Diff"
        old_diff_col = f"{cum_col}OldDiff"
        is_imputed_col = f"{diff_col}_is_imputed"
        impute_type_col = f"{diff_col}_impute_type"
        avg_col = f"{diff_col}_avg"

        df[old_diff_col] = df[diff_col]

        drop_temp_cols(df, logLeftoverError=True)

        if df[diff_col].isna().sum() == 0:
            logging.info(
                f"No values to impute in {diff_col}. Only checking thresholds.",
            )
            df[is_imputed_col] = False
            df[impute_type_col] = pd.Series(pd.NA, dtype="Int8", index=df.index)
            df["gap_length"] = pd.Series(pd.NA, dtype="Int8", index=df.index)
            df["cumulative_value_group"] = pd.Series(
                pd.NA,
                dtype="Int8",
                index=df.index,
            )
            logging.info("Apply thresholds to remove physically impossible outliers")
            df = apply_thresholds(
                df,
                lower_bound=thresholds_dict[diff_col]["Min"],
                upper_bound=thresholds_dict[diff_col]["Max"],
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
            )

            remaining_na = df[diff_col].isna().sum()

            if remaining_na > 0:
                logging.error(
                    f"{remaining_na} missing values still exist in column {diff_col}. Check masks and imputation logic.",
                )

        logging.info(f"Calculating {diff_col} imputation gap stats")
        imputation_gap_stats.append(
            df.groupby([project_id_column, "HuisIdBSV"])
            .apply(calculate_imputation_gap_stats, cum_col, diff_col, impute_type_col)
            .reset_index(),
        )

        drop_temp_cols(df, temp_cols=["gap_length", "cumulative_value_group"])

        drop_temp_cols(df, logLeftoverError=True)

    imputation_gap_stats_df = pd.concat(imputation_gap_stats, ignore_index=True)
    imputation_gap_stats_df["bitwise_methods"] = methods_to_bitwise_vectorized(
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
        df["impute_values"] = df[avg_col].fillna(0)

        if (df["impute_values"] < 0).any():
            raise Exception("Negative impute jump - not allowed - check averages")

        df["impute_na_ratio"] = df["impute_na"] / df["gap_length"]
        df["impute_jump"] = df.groupby("cumulative_value_group")[
            "impute_values"
        ].transform("sum")

        # calculate cumulative total for household for scaling the averages (optional)
        df[is_imputed_col] = False
        df[impute_type_col] = pd.Series(pd.NA, dtype="Int8", index=df.index)

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

        df["house_impute_factor"] = (df["diff_avg_sum"] / df["cum_diff_sum"]).replace(
            [float("inf"), -float("inf")],
            pd.NA,
        )
        df["house_impute_factor"] = (
            df["house_impute_factor"].fillna(1.0).mask(not_enough_comparable, 1.0)
        )

        return df

    temp_cols = [
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
        df.loc[flat_gap_jump_mask, impute_type_col] = ImputeType.NEGATIVE_GAP_JUMP

        # gap jump near zero - fill with zeros
        flat_gap_jump_mask = (
            has_gap_jump_mask & (df["gap_jump"] >= 0) & (df["gap_jump"] < 1e-8)
        )
        df.loc[flat_gap_jump_mask, is_imputed_col] = True
        df.loc[flat_gap_jump_mask, diff_col] = 0
        df.loc[flat_gap_jump_mask, impute_type_col] = ImputeType.NEAR_ZERO_GAP_JUMP

        # positive gap jump and impute jump near zero - linear fill
        # round(gap_jump / gap_length,10)
        positive_gap_linear_mask = (
            has_gap_jump_mask & (df["gap_jump"] >= 1e-8) & (df["impute_jump"] < 1e-8)
        )
        df.loc[positive_gap_linear_mask, is_imputed_col] = True
        df.loc[positive_gap_linear_mask, diff_col] = round(
            df["gap_jump"] / df["gap_length"],
            10,
        )
        df.loc[positive_gap_linear_mask, impute_type_col] = ImputeType.LINEAR_FILL

        # positive gap jump and positive impute jump and - scaled impute value fill
        # (optional for future: add logic to look at impute_na_ratio)
        positive_gap_scaled_mask = (
            has_gap_jump_mask & (df["gap_jump"] >= 1e-8) & (df["impute_jump"] >= 1e-8)
        )
        df.loc[positive_gap_scaled_mask, is_imputed_col] = True
        df.loc[positive_gap_scaled_mask, diff_col] = round(
            df.loc[positive_gap_scaled_mask, "impute_values"]
            * (
                df.loc[positive_gap_scaled_mask, "gap_jump"]
                / df.loc[positive_gap_scaled_mask, "impute_jump"]
            ),
            10,
        )
        df.loc[positive_gap_scaled_mask, impute_type_col] = ImputeType.SCALED_FILL

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
            logging.error(f"No next value or last value for a gap. Whole column empty?")
            house_no_diff_values = (
                df.groupby("HuisIdBSV")[diff_col].transform("count") == 0
            )

            # raise Exception(f'No next value or last value for a gap.')

        ### has end value but no start value for gap
        nogpjump_has_end_value_mask = (
            wo_gap_jump_mask & ~df["end_cum_value"].isna() & df["prev_cum_value"].isna()
        )

        #### has end value is 0 - fill with 0 - type 6
        nogpjump_has_end_value_zero_mask = nogpjump_has_end_value_mask & (
            df["end_cum_value"] < 1e-8
        )
        df.loc[nogpjump_has_end_value_zero_mask, is_imputed_col] = True
        df.loc[nogpjump_has_end_value_zero_mask, diff_col] = 0
        df.loc[nogpjump_has_end_value_zero_mask, impute_type_col] = ImputeType.ZERO_END_VALUE

        #### end value > 0 - fill with impute values - type 7
        nogpjump_has_end_value_positive_mask = nogpjump_has_end_value_mask & (
            df["end_cum_value"] > 1e-8
        )
        df.loc[nogpjump_has_end_value_positive_mask, is_imputed_col] = True
        df.loc[nogpjump_has_end_value_positive_mask, diff_col] = df.loc[
            nogpjump_has_end_value_positive_mask,
            "impute_values",
        ]
        df.loc[nogpjump_has_end_value_positive_mask, impute_type_col] = ImputeType.POSITIVE_END_VALUE

        #### end value < 0 - raise exception
        if (df["end_cum_value"] < 0).any():
            raise Exception(
                "Negative next value at end of gap - that is not supposed to happen!",
            )

        ### start value but no end value for gap - fill with impute values
        nogpjump_has_start_value_mask = (
            wo_gap_jump_mask & df["end_cum_value"].isna() & ~df["prev_cum_value"].isna()
        )
        df.loc[nogpjump_has_start_value_mask, is_imputed_col] = True
        df.loc[nogpjump_has_start_value_mask, diff_col] = (
            df.loc[nogpjump_has_start_value_mask, "impute_values"]
            * df.loc[nogpjump_has_start_value_mask, "house_impute_factor"]
        )
        df.loc[nogpjump_has_start_value_mask, impute_type_col] = ImputeType.NO_END_VALUE

        return df

    logging.info("Impute without gap jump")
    df = impute_wo_gap_jump(
        df,
        diff_col=diff_col,
        impute_type_col=impute_type_col,
        is_imputed_col=is_imputed_col,
    )

    logging.info("Apply thresholds to remove physically impossible outliers")
    df = apply_thresholds(
        df,
        lower_bound=thresholds_dict[diff_col]["Min"],
        upper_bound=thresholds_dict[diff_col]["Max"],
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

from enum import IntFlag, auto

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
    LINEAR_FILL : int
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

    Notes
    -----
    The enumeration values are automatically assigned using auto().
    The THRESHOLD_ADJUSTED flag can be combined with other imputation types.
    """

    NONE = 0
    NEGATIVE_GAP_JUMP = auto()
    NEAR_ZERO_GAP_JUMP = auto()
    LINEAR_FILL = auto()
    SCALED_FILL = auto()
    ZERO_END_VALUE = auto()
    POSITIVE_END_VALUE = auto()
    NO_END_VALUE = auto()
    THRESHOLD_ADJUSTED = auto()
