import os
from pathlib import Path

import etdmap
import etdmap.index_helpers
import pandas as pd
import pytest
import yaml
from etdmap.index_helpers import read_index, update_meenemen

import etdtransform
from etdtransform.aggregate import (
    add_calculated_columns_to_hh_data,
    aggregate_hh_data_5min,
    aggregate_project_data,
    impute_hh_data_5min,
    read_hh_data,
    resample_hh_data,
)
from etdtransform.impute import prepare_diffs_for_impute, sort_for_impute


def test_total_workflow_imputations():

    # set paths
    def load_config(config_path):
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)

    test_config_path = Path("config_test.yaml")
    if os.path.isfile(test_config_path):
        config = load_config(test_config_path)
    else:
         raise FileNotFoundError("no file named 'config_test.yaml'")

    etdmap.options.mapped_folder_path = Path(config['etdmap_configuration']['mapped_folder_path'])
    etdmap.options.aggregate_folder_path = Path(config['etdmap_configuration']['aggregate_folder_path'])
    etdmap.options.bsv_metadata_file = Path(config['etdmap_configuration']['bsv_metadata_file'])
    etdtransform.options.mapped_folder_path = Path(config['etdtransform_configuration']['mapped_folder_path'])
    etdtransform.options.aggregate_folder_path = etdmap.options.aggregate_folder_path

    # update with the manual steps from bsv metadata
    index_df = etdmap.index_helpers.update_meenemen()

    # test if indeed ProjectBSVId and column Meenemen are present
    assert 'Meenemen' in index_df.columns, 'column "Meenmen" not in index_df'
    # Since this is a manual check, ensure that all values have been given either
    # true or false and not none
    assert index_df['Meenemen'].notna().all(), 'column Meenemen contains None values' 
    assert index_df['Meenemen'].map(lambda x: isinstance(x, bool)).all(), 'column Meenemen has non-boolean values'
    assert index_df['ProjectIdBSV'].notna().all(), 'column PorjectIdBSV has None values'
    assert index_df['ProjectIdBSV'].dtype in [int, 'int64', 'Int64', 'int32'], "Column 'ProjectIdBSV' is not an integer dtype"

    # update_index
    # 10 columns
    cum_cols_list = [
            "ElektriciteitsgebruikBooster",
            "ElektriciteitsgebruikBoilervat",
            "ElektriciteitsgebruikWTW",
            "ElektriciteitsgebruikRadiator",
            "Zon-opwekTotaal",
            "ElektriciteitsgebruikWarmtepomp",
            "ElektriciteitTerugleveringLaag",
            "ElektriciteitTerugleveringHoog",
            "ElektriciteitNetgebruikLaag",
            "ElektriciteitNetgebruikHoog",
        ]
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

    # "add calculated columns"
    add_calculated_columns_to_hh_data(df_imputed)

    #"resample_hh_5min"
    resample_hh_data(intervals=["5min"])

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

if __name__ == "__main__":
    # Run pytest for debugging the testing
    pytest.main(["-v"])