import os
from pathlib import Path

import conftest
import etdmap
import etdmap.index_helpers
import pandas as pd
import pyarrow.parquet as pq
import pytest
import etdmap.data
from test_helpers import get_metadata_parquet_file

import etdtransform
from etdtransform.aggregate import (
    add_calculated_columns_to_hh_data,
    aggregate_hh_data_5min,
    aggregate_project_data,
    impute_hh_data_5min,
    read_hh_data,
    resample_hh_data,
)
from etdtransform.impute import prepare_diffs_for_impute


def test_total_workflow_imputations():

    # update with the manual steps from bsv metadata
    index_df = etdmap.index_helpers.update_meenemen()

    # test if indeed ProjectBSVId and column Meenemen are present
    assert 'Meenemen' in index_df.columns, 'column "Meenmen" not in index_df'
    # Since this is a manual check, ensure that all values have been given either
    # true or false and not none
    assert index_df['Meenemen'].notna().all(), 'column Meenemen contains None values' 
    assert index_df['Meenemen'].map(lambda x: isinstance(x, bool)).all(), 'column Meenemen has non-boolean values'
    assert index_df['ProjectIdBSV'].notna().all(), 'column ProjectIdBSV has None values'
    assert index_df['ProjectIdBSV'].dtype in [int, 'int64', 'Int64', 'int32'], "Column 'ProjectIdBSV' is not an integer dtype"

    # update_index
    # 10 columns
    cum_cols_list = etdmap.data_model.cumulative_columns[0:10]
    # cum_cols_list = [
    #         "ElektriciteitsgebruikBooster",
    #         "ElektriciteitsgebruikBoilervat",
    #         "ElektriciteitsgebruikWTW",
    #         "ElektriciteitsgebruikRadiator",
    #         "Zon-opwekTotaal",
    #         "ElektriciteitsgebruikWarmtepomp",
    #         "ElektriciteitTerugleveringLaag",
    #         "ElektriciteitTerugleveringHoog",
    #         "ElektriciteitNetgebruikLaag",
    #         "ElektriciteitNetgebruikHoog",
    #     ]
    # Aggregate mapped date (not in 'all' workflow)
    aggregate_hh_data_5min()
    # Check if columns were added and if length of file is correct
    file_path = os.path.join(etdtransform.options.aggregate_folder_path, "household_default.parquet")
    default_df = pd.read_parquet(file_path)
    assert "ProjectIdBSV" in default_df.columns, 'No column ProjectIdBSV in aggregated household_df'
    assert "HuisIdBSV" in default_df.columns, 'No column HuisIdBSV in aggregated household_df'

    # The default file is the aggregated file for houseshold_dfs 
    # it should therefore be the legth of the household dfs
    nmbr_huisids = len(index_df.loc[:, "HuisIdBSV"].unique())
    huis_id_bsv = index_df.loc[0, "HuisIdBSV"]
    file_name = f"household_{huis_id_bsv}_table.parquet"
    file_path_hh = os.path.join(etdtransform.options.mapped_folder_path, file_name)
    household_df = pd.read_parquet(file_path_hh)
    len_per_hh = len(household_df)
    assert len(default_df) == len_per_hh * nmbr_huisids, 'aggregated hh file (default) does not have the right length'  

    # # "load default data", It's possible to also add columns
    df = read_hh_data(interval="default", metadata_columns=['Dataleverancier'])
    assert all([col in df.columns for col in ["HuisIdBSV", "ProjectIdBSV", "Dataleverancier"]])

    # # "prepare and save diff averages",
    prepare_diffs_for_impute(
        df,
        project_id_column="ProjectIdBSV",
        cumulative_columns=cum_cols_list,
        sorted=False,
    )
    diffs_calculated = True
    # should create new files
    path_avg_diffs = os.path.join(etdtransform.options.aggregate_folder_path, "avg_diffs.parquet")
    path_max_bound = os.path.join(etdtransform.options.aggregate_folder_path, "household_diff_max_bounds.parquet")    
    assert os.path.isfile(path_avg_diffs)
    assert os.path.isfile(path_max_bound)

    # "impute"
    df_imputed = impute_hh_data_5min(
            df,
            cum_cols=cum_cols_list,
            sorted=True,
            diffs_calculated=diffs_calculated,
        )
    # should create the following files
    hh_agg_diff_path = os.path.join(
            etdtransform.options.aggregate_folder_path,
            "household_aggregated_diff.parquet",
        )
    imputation_summary_house_path = os.path.join(
            etdtransform.options.aggregate_folder_path,
            "impute_summary_household.parquet",
        )
    imputation_summary_project_path = os.path.join(
            etdtransform.options.aggregate_folder_path,
            "impute_summary_project.parquet",
        )
    assert os.path.isfile(hh_agg_diff_path)
    assert os.path.isfile(imputation_summary_house_path)
    assert os.path.isfile(imputation_summary_project_path)

    # "add calculated columns"
    add_calculated_columns_to_hh_data(df_imputed)
    # should create file: 
    hh_calculated_path = os.path.join(etdtransform.options.aggregate_folder_path, "household_calculated.parquet")
    assert os.path.isfile(hh_calculated_path)
    # the household_calculated file should contain the following cols:
    calc_cols = [
        "TerugleveringTotaalNetto",
        "ElektriciteitsgebruikTotaalNetto",
        "ElektriciteitsgebruikTotaalWarmtepomp",
        "ElektriciteitsgebruikTotaalGebouwgebonden",
        "ElektriciteitsgebruikTotaalHuishoudelijk",
        "Zelfgebruik",
        "ElektriciteitsgebruikTotaalBruto"
    ]
    df_hh_calc = pd.read_parquet(hh_calculated_path)
    assert all([col in df_hh_calc.columns for col in calc_cols])

    #"resample_hh_5min"
    resample_hh_data(intervals=["5min"])
    # Should create file
    hh_5min_path = os.path.join(etdtransform.options.aggregate_folder_path, "household_5min.parquet")
    assert os.path.isfile(hh_5min_path)

    # Note all following imputations and aggregations
    # will be run here, and tested in test_files_equal_expected 

    # "aggregate_project_5min"
    aggregate_project_data(intervals=["5min"])
    # "resample_hh_15_60min"
    resample_hh_data(intervals=["60min", "15min"])
    # "aggregate_project_15_60min"
    aggregate_project_data(intervals=["60min", "15min"])
    # "resample_hh_24h"
    resample_hh_data(intervals=["24h"])
    # "aggregate_project_24h"
    aggregate_project_data(intervals=["24h"])
    # "resample_hh_6h"
    resample_hh_data(intervals=["6h"])
    # "aggregate_project_6h"
    aggregate_project_data(intervals=["6h"])


def _check_metadatafiles_are_equal(load_metadata, stored_path, generated_path):

    expected_metadata = load_metadata(stored_path)

    parquet_file = pq.ParquetFile(generated_path)
    actual_metadata = get_metadata_parquet_file(parquet_file)
    # The meta data contains:
    # the number of rows & cols,
    # for each column the min, max values and null count
    return actual_metadata == expected_metadata


def _check_samples_are_equal(expected_path, generated_path):
    """
    Checks if expected vs. generated samples of .parquet files are equal.
    """
    df_expected = pd.read_parquet(expected_path)

    df_generated_full = pd.read_parquet(generated_path)
    sample_size = min(100, len(df_generated_full))
    df_generated_sample = df_generated_full.sample(n=sample_size, random_state=42)
    return df_expected.equals(df_generated_sample)


def test_files_equal_expected(load_metadata):
    """
    Checks for each file generated by the workflow if
    its sample and its metadata match the expected files.
    """
    for name in conftest.file_names:
        name=name.split('.parquet')[0]
        expected_path = Path(f"tests/data/metadata_{name}.json")
        generated_path = os.path.join(etdtransform.options.aggregate_folder_path, f"{name}.parquet")
        assert _check_metadatafiles_are_equal(
            load_metadata,
            expected_path,
            generated_path
            ), f"expected vs. generaged metadata files do not match for metadata_{name}.json"

        # check sample of file
        expected_path = Path(f"tests/data/sample_{name}.parquet")
        generated_path = os.path.join(etdtransform.options.aggregate_folder_path, f"{name}.parquet")
        assert _check_samples_are_equal(
            expected_path,
            generated_path
            ), f"expected vs. generaged files do not match for sample_{name}.parquet"

if __name__ == "__main__":
    # Run pytest for debugging the testing
    pytest.main(["-v"])
