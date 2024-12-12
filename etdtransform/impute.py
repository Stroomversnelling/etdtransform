import logging
import os
from enum import IntEnum
from math import floor, isclose, log10

import numpy as np
import pandas as pd
from etl.validators.record_validators import thresholds_dict
from etl.vectorized_impute import impute_and_normalize_vectorized

impute_and_normalize_optimized = impute_and_normalize_vectorized


def calculate_average_diff(
    df: pd.DataFrame,
    project_id_column,
    diff_columns: list,
) -> pd.DataFrame:
    logging.info(f"Calculating Diff column averages.")

    def safe_quantile(group, col_name):
        filtered_group = group[group[col_name] > 1e-8]
        if filtered_group.empty:
            return pd.Series({col_name: pd.NA}, dtype="Float64")
        else:
            return pd.Series({col_name: filtered_group[col_name].quantile(0.95)})

    # per Diff column max per house
    logging.info("Calculating max values per household.")
    household_max = (
        df.groupby([project_id_column, "HuisCode"])[diff_columns].max().reset_index()
    )
    household_max.columns = [project_id_column, "HuisCode"] + [
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
            [project_id_column, "HuisCode", f"{col}_huis_max"]
        ].merge(upper_bounds, on=project_id_column, how="left")
        include_mask = (
            household_max_with_bounds[f"{col}_huis_max"]
            < household_max_with_bounds[f"{col}_upper_bound"]
        )
        households_to_include = household_max_with_bounds.loc[include_mask, "HuisCode"]

        logging.info(f"Filtering the dataframe for {col}.")
        df_filtered = df[["HuisCode", project_id_column, "ReadingDate", col]][
            df["HuisCode"].isin(households_to_include)
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
    first_key = next(iter(avg_diff_dict))
    key_columns = avg_diff_dict[first_key]["household_max_with_bounds"][
        [project_id_column, "HuisCode"]
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


class ImputeType(IntEnum):
    FIRST_GAP = 1
    RATIO_ADJUSTED = 2
    EQUAL_DISTRIBUTION = 3
    ZERO_JUMP = 0
    NEGATIVE_JUMP = -1
    AVERAGE_FILL = 6
    RATIO_IMPUTE = 7
    ZERO_IMPUTE = 8


def validate_household_column(household_df, cum_col, huis_code):
    n_na = household_df[cum_col].isna().sum()
    len_df = len(household_df.index)

    if n_na == len_df:
        logging.info(
            f"HuisCode {huis_code} has all {n_na} missing values in {cum_col} of {len_df} records. Skipping column.",
        )
        return False
    elif n_na / len_df > 0.4:
        percent_na = 100 * n_na / len_df
        logging.error(
            f"HuisCode {huis_code} has {percent_na:.2f}% missing values in {cum_col}. Consider removing.",
        )
    else:
        logging.info(
            f"HuisCode {huis_code} has {n_na} missing values in {cum_col} of {len_df} records.",
        )

    if round(household_df[cum_col].sum(), 10) == 0:
        logging.info(
            f"HuisCode {huis_code} has no non-zero values in {cum_col}. Skipping column.",
        )
        return False
    if round(household_df[cum_col].max() - household_df[cum_col].min(), 10) == 0:
        logging.info(
            f"HuisCode {huis_code} has no change in {cum_col}. Skipping column.",
        )
        return False
    if round(household_df[f"{cum_col}Diff"].sum(), 10) == 0:
        logging.warning(
            f"HuisCode {huis_code} has no non-zero values in {cum_col}Diff before imputation.",
        )

    return True


def identify_gaps(household_df, diff_col):
    gap_col = f"{diff_col}_gap"
    household_df[gap_col] = household_df[diff_col].isna().astype(int)

    gap_starts = household_df[
        household_df[gap_col] & ~household_df[gap_col].shift(1, fill_value=False)
    ].index
    gap_ends = household_df[
        household_df[gap_col] & ~household_df[gap_col].shift(-1, fill_value=False)
    ].index

    return list(zip(gap_starts, gap_ends))


def enforce_thresholds_upper_bounds(household_df, diff_col, max_allowed):
    print()


def imputation_column_info_checks(household_df, cum_col, huis_code):
    na_values = household_df[cum_col].isna()
    n_na = na_values.sum()
    len_df = len(household_df.index)

    if n_na == len_df:
        logging.info(
            f"HuisCode {huis_code} has all {n_na} missing values in {cum_col} of {len_df} records. Skipping column.",
        )
        return False
    elif n_na / len_df > 0.4:
        percent_na = 100 * n_na / len_df
        logging.error(
            f"HuisCode {huis_code} has {percent_na}% missing values in {cum_col}. Consider removing.",
        )
    else:
        logging.info(
            f"HuisCode {huis_code} has {n_na} missing values in {cum_col} of {len_df} records.",
        )

    if round(household_df[cum_col].sum(), 10) == 0:
        logging.info(
            f"HuisCode {huis_code} has no non-zero values in {cum_col}. Skipping column.",
        )
        return False

    return True


def check_cumulative_difference(household_df, cum_col, huis_code):
    diff_col = cum_col + "Diff"
    cum_column_total_difference = round(
        household_df[cum_col].max() - household_df[cum_col].min(),
        10,
    )
    if isclose(cum_column_total_difference, 0):
        logging.info(
            f"HuisCode {huis_code} has no change in {cum_col}. Skipping column.",
        )
        return cum_column_total_difference, False
    if round(household_df[diff_col].sum(), 10) == 0:
        logging.warning(
            f"HuisCode {huis_code} has no non-zero values in {diff_col} before imputation.",
        )

    return cum_column_total_difference, True


def impute_and_normalize_old(
    df: pd.DataFrame,
    cumulative_columns: list,
    project_id_column: str,
    max_bound: pd.DataFrame,
):
    logging.info("Starting to impute cumulative column diffs.")
    imputation_gap_stats = []

    logging.info("Creating imputed boolean flag column for each variable.")
    for col in cumulative_columns:
        df[col + "Diff_is_imputed"] = False
        df[col + "Diff_impute_type"] = pd.Series(pd.NA, dtype="Int8", index=df.index)

    col = None

    logging.info("Splitting.")

    gap_info = []
    previous_huis_code = -1

    modified_household_dfs = []

    for huis_code, household_df in df.groupby("HuisCode"):
        logging.info(f"Finding gaps in {huis_code}")

        for cum_col in cumulative_columns:
            logging.info(f"Finding gaps in {huis_code}/{cum_col}")
            diff_col = f"{cum_col}Diff"
            is_imputed_col = f"{diff_col}_is_imputed"
            impute_type_col = f"{diff_col}_impute_type"
            avg_col = f"{diff_col}_avg"
            diff_gap_col = diff_col + "_gap"

            total_gap_count = 0

            if cum_col in household_df.columns:
                na_values = household_df[cum_col].isna()
                n_na = na_values.sum()
                len_df = len(household_df.index)

                if n_na == len_df:
                    logging.info(
                        f"HuisCode {huis_code} has all {n_na} missing values in {cum_col} of {len_df} records. Skipping column.",
                    )
                    continue
                elif n_na / len_df > 0.4:
                    percent_na = 100 * n_na / len_df
                    logging.error(
                        f"HuisCode {huis_code} has {percent_na}% missing values in {cum_col}. Consider removing.",
                    )
                else:
                    logging.info(
                        f"HuisCode {huis_code} has {n_na} missing values in {cum_col} of {len_df} records.",
                    )

                if round(household_df[cum_col].sum(), 10) == 0:
                    logging.info(
                        f"HuisCode {huis_code} has no non-zero values in {cum_col}. Skipping column.",
                    )
                    continue

                cum_column_total_difference = round(
                    household_df[cum_col].max() - household_df[cum_col].min(),
                    10,
                )

                if isclose(cum_column_total_difference, 0):
                    logging.info(
                        f"HuisCode {huis_code} has no change in {cum_col}. Skipping column.",
                    )
                    continue
                if round(household_df[diff_col].sum(), 10) == 0:
                    logging.warning(
                        f"HuisCode {huis_code} has no non-zero values in {diff_col} before imputation.",
                    )

                household_df[impute_type_col] = pd.Series(
                    pd.NA,
                    dtype="Int8",
                    index=household_df.index,
                )

                # Gap identification
                household_df[diff_gap_col] = household_df[diff_col].isna().astype(int)

                diff_gap_starts = household_df[
                    household_df[diff_gap_col]
                    & ~household_df[diff_gap_col].shift(1, fill_value=False)
                ].index
                diff_gap_ends = household_df[
                    household_df[diff_gap_col]
                    & ~household_df[diff_gap_col].shift(-1, fill_value=False)
                ].index

                n_gaps = len(diff_gap_starts)
                logging.info(
                    f"There are {n_gaps} diff gaps in HuisCode {huis_code} for {cum_col}",
                )

                gap_avg_missing_warning = False

                for start_diff_gap, end_diff_gap in zip(diff_gap_starts, diff_gap_ends):
                    cum_gap_values = list(
                        household_df.loc[start_diff_gap:end_diff_gap, cum_col]
                        .dropna()
                        .index,
                    )
                    if pd.isna(household_df.loc[end_diff_gap, cum_col]):
                        cum_gap_values.append(end_diff_gap)

                    start = start_diff_gap

                    for end in cum_gap_values:
                        # logging.info(f'start {start} to end {end}; huis: {huis_code}')

                        if start - 1 >= household_df.index[0]:
                            prev_cum_value = household_df.loc[start - 1, cum_col]
                            if pd.isna(prev_cum_value):
                                prev_cum_value = None
                            first_gap = False
                        else:
                            first_gap = True
                            prev_cum_value = None

                        next_cum_value = household_df.loc[end, cum_col]
                        if pd.isna(next_cum_value):
                            next_cum_value = None

                        gap_length = (end + 1) - start

                        gap_jump = (
                            round(next_cum_value - prev_cum_value, 10)
                            if prev_cum_value is not None and next_cum_value is not None
                            else None
                        )

                        gap_info.append(
                            {
                                "HuisCode": huis_code,
                                "column": cum_col,
                                "gap_jump": gap_jump,
                                "gap_length": gap_length,
                                "first_gap": first_gap,
                                "start": start,
                                "end": end,
                            },
                        )

                        total_gap_count = total_gap_count + gap_length

                        if gap_length > 0:
                            # if huis_code != previous_huis_code:
                            # logging.info(f'Calculating gaps in {huis_code}/{diff_col} (previous code = {previous_huis_code})')
                            # previous_huis_code = huis_code

                            impute_values = household_df.loc[start:end, avg_col]
                            impute_na = impute_values.isna().sum()

                            if (
                                impute_na > 0
                            ):  # it would be better to not have any na values
                                if gap_avg_missing_warning == False:
                                    logging.warning(
                                        f"Gap records are missing {impute_na} impute values based on average diff values. Using 0s or linear interpolation. Check results.",
                                    )
                                    gap_avg_missing_warning = True
                                # We will need to check if this does not result in artifical peaks when there is only one or few timesteps with a non-0 value.
                                impute_values = impute_values.fillna(0)

                            impute_jump = round(impute_values.sum(), 10)
                            imputed_count = 0

                            if impute_jump < 0:
                                raise Exception("Negative impute jump")

                            if (
                                first_gap == True
                            ):  # the first diff value is always NA since there is nothing to difference with and thus always has to be filled with the average
                                household_df.loc[start:end, diff_col] = (
                                    impute_values  # may need to check if this doesn't always work
                                )
                                household_df.loc[start:end, is_imputed_col] = True
                                household_df.loc[start:end, impute_type_col] = 1
                            else:
                                if gap_jump is not None:
                                    if (
                                        round(gap_jump, 10) < 0
                                    ):  # we could consider adding the average instead of zeros here, especially if the gap is long (for example > 1hr)
                                        logging.error(
                                            f"Negative gap jump of {gap_jump} with HuisCode {huis_code} in column '{cum_col}'. Impute type = -1 (adding 0s). Consider removing.",
                                        )
                                        household_df.loc[start:end, diff_col] = 0
                                        household_df.loc[start:end, is_imputed_col] = (
                                            True
                                        )
                                        household_df.loc[start:end, impute_type_col] = 2
                                    elif round(gap_jump, 10) == 0:
                                        household_df.loc[start:end, diff_col] = 0
                                        household_df.loc[start:end, is_imputed_col] = (
                                            True
                                        )
                                        household_df.loc[start:end, impute_type_col] = 3
                                    elif round(gap_jump, 10) > 0:
                                        if (
                                            round(impute_jump, 10) == 0
                                        ):  # this could change the shape of the curve a lot if it turns 0,100,0,100 into 50,50,50,50
                                            household_df.loc[start:end, diff_col] = (
                                                round(gap_jump / gap_length, 10)
                                            )
                                            household_df.loc[
                                                start:end,
                                                is_imputed_col,
                                            ] = True
                                            household_df.loc[
                                                start:end,
                                                impute_type_col,
                                            ] = 4
                                        else:
                                            ratio = round(gap_jump / impute_jump, 10)
                                            if ratio < 0:
                                                raise Exception("Negative ratio!")
                                            household_df.loc[start:end, diff_col] = (
                                                impute_values * round(ratio, 10)
                                            )
                                            household_df.loc[
                                                start:end,
                                                is_imputed_col,
                                            ] = True
                                            household_df.loc[
                                                start:end,
                                                impute_type_col,
                                            ] = 5
                                    else:
                                        raise Exception(f"Unknown condition")
                                else:
                                    if next_cum_value is not None:
                                        if round(next_cum_value, 10) == 0:
                                            household_df.loc[start:end, diff_col] = 0
                                            household_df.loc[
                                                start:end,
                                                is_imputed_col,
                                            ] = True
                                            household_df.loc[
                                                start:end,
                                                impute_type_col,
                                            ] = 6
                                        else:
                                            if (
                                                round(next_cum_value, 10) > 0
                                            ):  # fill with averages
                                                household_df.loc[
                                                    start:end,
                                                    diff_col,
                                                ] = impute_values
                                                household_df.loc[
                                                    start:end,
                                                    is_imputed_col,
                                                ] = True
                                                household_df.loc[
                                                    start:end,
                                                    impute_type_col,
                                                ] = 7
                                            elif round(next_cum_value, 10) < 0:
                                                raise Exception("Negative next value!")
                                            else:
                                                raise Exception(
                                                    f"Unknown condition for gap next value = {next_cum_value} but not first gap and no gap_value.",
                                                )
                                    elif prev_cum_value is not None:
                                        if (
                                            household_df.loc[0:end, diff_col].sum() > 0
                                        ):  # there are positive diff values prior this gap
                                            # Get a ratio of the expected cumulative increase against the observed increase for records without missing values if reasonable
                                            sum_records = len(
                                                household_df[
                                                    household_df[avg_col].notna()
                                                    & household_df[diff_col].notna()
                                                ].index,
                                            )
                                            if sum_records > len_df / 2:
                                                avg_col_sum = household_df[
                                                    household_df[avg_col].notna()
                                                    & household_df[diff_col].notna()
                                                ][avg_col].sum()
                                                dif_col_sum = household_df[
                                                    household_df[avg_col].notna()
                                                    & household_df[diff_col].notna()
                                                ][diff_col].sum()
                                                if avg_col_sum > 0:
                                                    if (
                                                        dif_col_sum > 0
                                                    ):  # can calculate the ratio
                                                        avg_to_diff_ratio = (
                                                            dif_col_sum / avg_col_sum
                                                        )
                                                        household_df.loc[
                                                            start:end,
                                                            diff_col,
                                                        ] = impute_values * round(
                                                            avg_to_diff_ratio,
                                                            10,
                                                        )
                                                        household_df.loc[
                                                            start:end,
                                                            is_imputed_col,
                                                        ] = True
                                                        household_df.loc[
                                                            start:end,
                                                            impute_type_col,
                                                        ] = 8
                                                    else:
                                                        household_df.loc[
                                                            start:end,
                                                            diff_col,
                                                        ] = 0
                                                        household_df.loc[
                                                            start:end,
                                                            is_imputed_col,
                                                        ] = True
                                                        household_df.loc[
                                                            start:end,
                                                            impute_type_col,
                                                        ] = 9
                                                else:
                                                    household_df.loc[
                                                        start:end,
                                                        diff_col,
                                                    ] = 0
                                                    household_df.loc[
                                                        start:end,
                                                        is_imputed_col,
                                                    ] = True
                                                    household_df.loc[
                                                        start:end,
                                                        impute_type_col,
                                                    ] = 10
                                            else:
                                                household_df.loc[
                                                    start:end,
                                                    diff_col,
                                                ] = impute_values
                                                household_df.loc[
                                                    start:end,
                                                    is_imputed_col,
                                                ] = True
                                                household_df.loc[
                                                    start:end,
                                                    impute_type_col,
                                                ] = 11
                                        else:
                                            household_df.loc[start:end, diff_col] = 0
                                            household_df.loc[
                                                start:end,
                                                is_imputed_col,
                                            ] = True
                                            household_df.loc[
                                                start:end,
                                                impute_type_col,
                                            ] = 12
                                    else:
                                        raise Exception(
                                            f"No next value or last value for gap with HuisCode {huis_code} in column {cum_col}. Probably should have skipped column and initial checks may need to be adjusted.",
                                        )

                        if (
                            gap_jump is not None
                        ):  # check if the sum  of imputed values and the gap jump are the same after the imputation.
                            sum_diff = household_df.loc[start:end, diff_col].sum()
                            if gap_jump > 0 and not equal_sig_fig(
                                a=sum_diff,
                                b=gap_jump,
                                sig_figs=5,
                            ):
                                raise Exception(
                                    f"Gap jump ({gap_jump}) and sum of imputed values ({sum_diff}) are not equal!",
                                )

                        start = end + 1

                # remove values based on physical thresholds and replace with average values
                mask = (
                    (household_df[diff_col] < thresholds_dict[diff_col]["Min"])
                    | (household_df[diff_col] > thresholds_dict[diff_col]["Max"])
                ) & household_df[diff_col].notna()
                if any(mask):
                    replacing_out_of_thresholds = mask.sum()
                    logging.error(
                        f"Replacing n={replacing_out_of_thresholds} values that exceeded physical thresholds for {diff_col} for HuisCode {huis_code}.",
                    )

                    household_df.loc[mask, diff_col] = household_df.loc[mask, avg_col]
                    household_df.loc[mask, is_imputed_col] = True
                    household_df.loc[mask, impute_type_col] = 13 + df[
                        impute_type_col
                    ].fillna(0)

                max_allowed = max_bound.loc[
                    max_bound["HuisCode"] == huis_code,
                    f"{diff_col}_upper_bound",
                ].values[0]
                if max_allowed is None:
                    logging.error(
                        f"No upper bound for {diff_col} in HuisCode {huis_code}.",
                    )
                else:
                    # household_df.loc[household_df[diff_col] > max_allowed, is_imputed_col] = True
                    # household_df.loc[household_df[diff_col] > max_allowed, impute_type_col] = 14
                    # household_df.loc[household_df[diff_col] > max_allowed, diff_col] = household_df.loc[household_df[diff_col] > max_allowed, avg_col]

                    # replacing_upper_bound = household_df.loc[household_df[impute_type_col] == 14,is_imputed_col].sum()

                    # if replacing_upper_bound > 0:
                    #    logging.error(f'Replacing {replacing_upper_bound} values that exceeded the {max_allowed} upper bound for {diff_col} for HuisCode {huis_code}.')

                    beyond_upper_bound = sum(household_df[diff_col] > max_allowed)
                    if beyond_upper_bound > 0:
                        logging.error(
                            f"There are {beyond_upper_bound} values that exceeded {max_allowed} upper bound used in the averages for {diff_col} for HuisCode {huis_code}. Not removed.",
                        )

                # Sanity check
                diff_column_total = household_df[diff_col].sum()
                difference_in_calculation = (
                    diff_column_total - cum_column_total_difference
                )

                if not isclose(cum_column_total_difference, diff_column_total):
                    if cum_column_total_difference > 0:
                        percent_diff = abs(
                            round(
                                difference_in_calculation
                                * 100
                                / cum_column_total_difference,
                                4,
                            ),
                        )
                        if percent_diff > 100 / (365 * 24 * 12):
                            logging.error(
                                f"Potential imputation error: there is a difference of {difference_in_calculation} ({percent_diff}%) between `{diff_col}` sum total ({diff_column_total}) and the cumulative minimum and maximum ({cum_column_total_difference})!",
                            )
                        else:
                            logging.info(
                                f"Minor deviation in imputation total: there is a minimal difference {difference_in_calculation} ({percent_diff}%) between `{diff_col}` sum total ({diff_column_total}) and the cumulative minimum and maximum ({cum_column_total_difference}).",
                            )
                    else:
                        logging.error(
                            f"Potential imputation error: there is a difference of {difference_in_calculation} between `{diff_col}` sum total ({diff_column_total}) and the cumulative minimum and maximum ({cum_column_total_difference})!",
                        )
                else:
                    logging.info(
                        f"Success! The totals match for huis_code {huis_code}. Imputation is successful.",
                    )

                missing_count = total_gap_count
                imputed_count = household_df[is_imputed_col].sum()
                imputed_na_count = missing_count - imputed_count

                methods = list(household_df[impute_type_col].dropna().unique())

                imputation_gap_stats.append(
                    {
                        project_id_column: household_df.loc[
                            household_df.index[0],
                            project_id_column,
                        ],
                        "HuisCode": huis_code,
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

        modified_household_dfs.append(household_df)

    df = pd.concat(modified_household_dfs, ignore_index=True)

    # imputation_reading_date_stats_df = get_reading_date_imputation_stats(df, project_id_column, cumulative_columns)
    imputation_reading_date_stats_df = None

    # Drop columns with Diff_avg, _gap_id, and _is_imputed suffixes
    drop_columns = [f"{col}Diff_avg" for col in cumulative_columns] + [
        f"{col}Diff_is_imputed" for col in cumulative_columns
    ]
    # df = df.drop(columns=drop_columns)

    logging.info(f"Concatenating the other imputation statistics")
    imputation_gap_stats_df = pd.DataFrame(imputation_gap_stats)
    imputation_gap_stats_df["bitwise_methods"] = methods_to_bitwise_vectorized(
        imputation_gap_stats_df["methods"],
    )

    return df, imputation_gap_stats_df, imputation_reading_date_stats_df


def methods_to_bitwise(methods):
    bitwise_value = 0
    for method in methods:
        bitwise_value |= 1 << (
            method - 1
        )  # Subtract 1 because method 1 is stored in the least significant bit
    return bitwise_value


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


def get_reading_date_imputation_stats(df, project_id_column, cumulative_columns):
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
    logging.info("Sorting to prepare for imputation.")
    return df.sort_values(by=[project_id_column, "HuisCode", "ReadingDate"])


def get_diff_columns(cumulative_columns: list):
    return [col + "Diff" for col in cumulative_columns]


aggregate_folder_path = os.getenv("AGGREGATE_FOLDER_PATH")


def prepare_diffs_for_impute(
    df: pd.DataFrame,
    project_id_column: str,
    cumulative_columns: list,
    sorted=False,
):
    if sorted != True:
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
        os.path.join(aggregate_folder_path, "avg_diffs.parquet"),
        engine="pyarrow",
    )
    logging.info(
        "Saving household diff max and bounds used in household_diff_max_bounds.parquet",
    )
    max_bound.to_parquet(
        os.path.join(aggregate_folder_path, "household_diff_max_bounds.parquet"),
        engine="pyarrow",
    )

    return diff_columns, diffs, max_bound


def read_diffs():
    return pd.read_parquet(os.path.join(aggregate_folder_path, "avg_diffs.parquet"))


def process_and_impute(
    df: pd.DataFrame,
    project_id_column: str,
    cumulative_columns: list,
    sorted=False,
    diffs_calculated=False,
    optimized=False,
):
    if sorted != True:
        df = sort_for_impute(df, project_id_column)

    if diffs_calculated:
        logging.info("Loading average diffs from file...")
        diffs = read_diffs()
        max_bound = pd.read_parquet(
            os.path.join(aggregate_folder_path, "household_diff_max_bounds.parquet"),
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

    if optimized == True:
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
            aggregate_folder_path,
            f"impute_gap_stats{optimized_label}.parquet",
        ),
        engine="pyarrow",
    )

    logging.info("Summarizing imputation_gap_stats_df per house and column")
    imputation_summary_house = imputation_gap_stats_df[
        [
            project_id_column,
            "HuisCode",
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
        df.groupby("HuisCode").size().reset_index(name="total_records")
    )

    logging.info("Merging total records with house imputation summary")
    imputation_summary_house = imputation_summary_house.merge(
        total_records_house,
        on=["HuisCode"],
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
            f"House {row['HuisCode']}, Column {row['column']} has {row['percentage_imputed']:.2f}% imputed values.",
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


impute_and_normalize = impute_and_normalize_vectorized
