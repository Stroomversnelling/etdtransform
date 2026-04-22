"""
End-to-end imputation workflow tests.

NOTE: These are integration tests that use the local test fixture dataset
configured in config_test.yaml (see config_test_template.yaml for the
reference template). The fixture is a small anonymised dataset -- NOT
production data. Tests must pass on that fixture; any failure is a real bug.
"""
import logging
import os
from pathlib import Path

import conftest
import etdmap
import etdmap.data
import etdmap.index_helpers
import pandas as pd
import pyarrow.parquet as pq
import pytest
from test_helpers import generate_metadata_parquet_file

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


@pytest.fixture(scope="session")
def index_df():
    # update with the manual steps from bsv metadata
    return etdmap.index_helpers.update_meenemen()

@pytest.fixture(scope="session")
def cum_cols_list():
    # 10 columns
    return etdmap.data_model.cumulative_columns[0:10]

def test_meenemen(index_df):
    # test if indeed ProjectBSVId and column Meenemen are present
    assert 'Meenemen' in index_df.columns, 'column "Meenmen" not in index_df'
    # Since this is a manual check, ensure that all values have been given either
    # true or false and not none
    assert index_df['Meenemen'].notna().all(), 'column Meenemen contains None values' 
    assert index_df['Meenemen'].map(lambda x: isinstance(x, bool)).all(), 'column Meenemen has non-boolean values'

def test_project_id(index_df):
    assert index_df['ProjectIdBSV'].notna().all(), 'column ProjectIdBSV has None values'
    assert index_df['ProjectIdBSV'].dtype in [int, 'int64', 'Int64', 'int32'], "Column 'ProjectIdBSV' is not an integer dtype"


def test_total_workflow_imputations(index_df, cum_cols_list):
    logging.info("Aggregating 5 minute household data.")

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

    logging.info("Loading default data with additional columns.")
    # # "load default data", It's possible to also add columns
    df = read_hh_data(interval="default", metadata_columns=['Dataleverancier'])
    assert all([col in df.columns for col in ["HuisIdBSV", "ProjectIdBSV", "Dataleverancier"]])

    logging.info("Preparing diffs.")
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

    logging.info("Imputing data.")
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

    logging.info("Adding calculated columns.")
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

    logging.info("Resampling HH data to 5 minutes.")
    #"resample_hh_5min"
    resample_hh_data(intervals=["5min"])
    # Should create file
    hh_5min_path = os.path.join(etdtransform.options.aggregate_folder_path, "household_5min.parquet")
    assert os.path.isfile(hh_5min_path)

    # Note all following imputations and aggregations
    # will be run here, and tested in test_files_equal_expected 

    # "aggregate_project_5min"
    logging.info("Aggregating project data to 5 minutes.")
    aggregate_project_data(intervals=["5min"])
    # "resample_hh_15_60min"
    logging.info("Resample HH data to 60 and 15 minutes.")
    resample_hh_data(intervals=["60min", "15min"])
    # "aggregate_project_15_60min"
    logging.info("Aggregating project data to 60 and 15 minutes.")
    aggregate_project_data(intervals=["60min", "15min"])
    # "resample_hh_24h"
    logging.info("Resample HH data to 24 hours.")
    resample_hh_data(intervals=["24h"])
    # "aggregate_project_24h"
    logging.info("Aggregating project data to 24 hours.")
    aggregate_project_data(intervals=["24h"])
    # "resample_hh_6h"
    logging.info("Resample HH data to 6 hours.")
    resample_hh_data(intervals=["6h"])
    # "aggregate_project_6h"
    logging.info("Aggregating project data to 6 hours.")
    aggregate_project_data(intervals=["6h"])


def _check_metadatafiles_are_equal(load_metadata, stored_path, generated_path):

    expected_metadata = load_metadata(stored_path)

    parquet_file = pq.ParquetFile(generated_path)
    actual_metadata = generate_metadata_parquet_file(parquet_file)
    # The meta data contains:
    # the number of rows & cols,
    # for each column the min, max values and null count

    results = _diff_json(expected_metadata, actual_metadata)

    if len(results) > 0:
        logging.info(f"Found {len(results)} differences in stats of variables (test fixture, generated stats)")

    return results, expected_metadata, actual_metadata


def _check_samples_are_equal(expected_path, generated_path, rtol=1e-10):
    """
    Checks if expected vs. generated samples of .parquet files are equal.

    Uses relative tolerance for float columns to absorb floating-point
    non-determinism from DuckDB parallel aggregation (project-level sums).
    Structural differences (column set, row count, dtypes) are still exact.
    """
    df_expected = pd.read_parquet(expected_path)

    df_generated_full = pd.read_parquet(generated_path)
    sample_size = min(100, len(df_generated_full))
    df_generated_sample = df_generated_full.sample(n=sample_size, random_state=42)
    try:
        pd.testing.assert_frame_equal(
            df_expected.reset_index(drop=True),
            df_generated_sample.reset_index(drop=True),
            check_exact=False,
            rtol=rtol,
        )
        return True
    except AssertionError:
        return False


def _diff_json(a, b, path="", float_rel_tol=1e-10):
    """Compare two JSON-deserialized objects recursively.

    Numeric string values (parquet min/max statistics) are compared with
    float_rel_tol relative tolerance to absorb floating-point non-determinism
    from groupby aggregations. Structural differences (missing keys, type
    mismatches, null_count changes) are always exact.
    """
    results = []

    def _record(diff):
        logging.info(diff)
        results.append(diff)

    def _is_numeric(v):
        try:
            float(v)
            return True
        except (TypeError, ValueError):
            return False

    def _recurse(a, b, path):
        if type(a) != type(b):
            _record(f"{path}: type mismatch {type(a).__name__} != {type(b).__name__}")
        elif isinstance(a, dict):
            keys = set(a.keys()).union(b.keys())
            for k in keys:
                if k not in a:
                    _record(f"{path}.{k}: missing in first")
                elif k not in b:
                    _record(f"{path}.{k}: missing in second")
                else:
                    _recurse(a[k], b[k], f"{path}.{k}")
        elif isinstance(a, list):
            for i in range(min(len(a), len(b))):
                _recurse(a[i], b[i], f"{path}[{i}]")
            if len(a) != len(b):
                _record(f"{path}: list length differs {len(a)} != {len(b)}")
        else:
            if a != b:
                # Allow tiny float rounding in parquet statistics (min/max stored as strings)
                if isinstance(a, str) and isinstance(b, str) and _is_numeric(a) and _is_numeric(b):
                    import math
                    if not math.isclose(float(a), float(b), rel_tol=float_rel_tol):
                        _record(f"{path}: {a} != {b}")
                else:
                    _record(f"{path}: {a} != {b}")

    _recurse(a, b, path or "$")
    return results

def test_files_equal_expected(load_metadata):
    """
    Checks for each file generated by the workflow if
    its sample and its metadata match the expected files.
    """
    for name in conftest.file_names:
        name=name.split('.parquet')[0]
        expected_path = Path(f"tests/data/metadata_{name}.json")
        generated_path = os.path.join(etdtransform.options.aggregate_folder_path, f"{name}.parquet")
        results, expected_json, generated_json = _check_metadatafiles_are_equal(
            load_metadata,
            expected_path,
            generated_path
            )

        assert len(results) == 0, f"expected vs. generaged metadata files do not match for metadata_{name}.json; see log file for differences"

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
