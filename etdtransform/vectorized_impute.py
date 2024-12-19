import logging

import numpy as np
import pandas as pd
from etdmap.record_validators import thresholds_dict


def methods_to_bitwise_vectorized(methods_column):
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
    mask = ((df[diff_col] < lower_bound) | (df[diff_col] > upper_bound)) & df[
        diff_col
    ].notna()
    df.loc[mask, diff_col] = df.loc[mask, avg_col]
    df.loc[mask, is_imputed_col] = True
    df.loc[mask, impute_type_col] = 13 + df[impute_type_col].fillna(0)

    return df


def impute_and_normalize_vectorized(
    df: pd.DataFrame,
    cumulative_columns: list,
    project_id_column: str,
    max_bound: pd.DataFrame,
):
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
            df.groupby([project_id_column, "HuisCode"])
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


# Function to create groups and cumulative value groups
def process_gap_and_cumulative_groups(df, diff_col, cum_col):
    temp_cols = ["gap_start", "gap_group", "cum_value_encountered"]

    # Step 1: Identify NA values in diff_col
    is_na_mask = df[diff_col].isna()

    # Step 2: Identify gap start only for NA values and transitions between households
    df["gap_start"] = (is_na_mask & (~is_na_mask.shift(1, fill_value=False))) | (
        (df["HuisCode"] != df["HuisCode"].shift(1)) & is_na_mask
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


# Function to handle imputation with fixed values for negative or invalid differences
def process_imputation_vectorized(
    df,
    diff_col,
    cum_col,
    avg_col,
    impute_type_col,
    is_imputed_col,
):
    def setup_prev_value_columns(df, cum_col):
        # Shift by one to get the immediate previous time step value before the group and make sure no other time step is there
        df["prev_cum_value"] = df[cum_col].shift(1)

        first_in_group_mask = (
            df["cumulative_value_group"] != df["cumulative_value_group"].shift(1)
        ) & ~(df["cumulative_value_group"].isna())

        first_in_house = df["HuisCode"] != df["HuisCode"].shift(1)

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
            .groupby(df["HuisCode"])
            .transform("sum")
        )
        df["cum_diff_sum"] = (
            df[diff_col]
            .where(df["comparable_to_impute_mask"])
            .groupby(df["HuisCode"])
            .transform("sum")
        )

        comparable_counts = df.groupby("HuisCode")[
            "comparable_to_impute_mask"
        ].transform("sum")
        total_counts = df.groupby("HuisCode")[diff_col].transform("size")
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
        df.loc[flat_gap_jump_mask, impute_type_col] = 2

        # gap jump near zero - fill with zeros
        flat_gap_jump_mask = (
            has_gap_jump_mask & (df["gap_jump"] >= 0) & (df["gap_jump"] < 1e-8)
        )
        df.loc[flat_gap_jump_mask, is_imputed_col] = True
        df.loc[flat_gap_jump_mask, diff_col] = 0
        df.loc[flat_gap_jump_mask, impute_type_col] = 3

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
        df.loc[positive_gap_linear_mask, impute_type_col] = 4

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
        df.loc[positive_gap_scaled_mask, impute_type_col] = 5

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
                df.groupby("HuisCode")[diff_col].transform("count") == 0
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
        df.loc[nogpjump_has_end_value_zero_mask, impute_type_col] = 6

        #### end value > 0 - fill with impute values - type 7
        nogpjump_has_end_value_positive_mask = nogpjump_has_end_value_mask & (
            df["end_cum_value"] > 1e-8
        )
        df.loc[nogpjump_has_end_value_positive_mask, is_imputed_col] = True
        df.loc[nogpjump_has_end_value_positive_mask, diff_col] = df.loc[
            nogpjump_has_end_value_positive_mask,
            "impute_values",
        ]
        df.loc[nogpjump_has_end_value_positive_mask, impute_type_col] = 7

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
        df.loc[nogpjump_has_start_value_mask, impute_type_col] = 8

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
        df.groupby("HuisCode")[cum_col].transform("sum").fillna(0) <= 0
    ) | (df.groupby("HuisCode")[cum_col].transform("count") == 0)

    # Households where max - min cumulative sum is not > 0
    houses_cum_min_max = (
        df[cum_col].groupby(df["HuisCode"]).transform("max")
        - df[cum_col].groupby(df["HuisCode"]).transform("min")
    ).fillna(0) <= 0

    drop_temp_cols(df, temp_cols=temp_cols)

    return df
