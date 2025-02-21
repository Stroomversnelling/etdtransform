import json
import os
from pathlib import Path

import etdmap
import pytest
import yaml

import etdtransform


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
# etdmap.options.aggregate_folder_path = Path(config['etdmap_configuration']['aggregate_folder_path'])
etdmap.options.bsv_metadata_file = Path(config['etdmap_configuration']['bsv_metadata_file'])
etdtransform.options.mapped_folder_path = Path(config['etdtransform_configuration']['mapped_folder_path'])
etdtransform.options.aggregate_folder_path = Path(config['etdtransform_configuration']['aggregate_folder_path'])

file_names = [
    "household_24h.parquet",
    "project_24h.parquet",
    "household_6h.parquet",
    "project_6h.parquet",
    "avg_diffs.parquet",
    "household_5min.parquet",
    "household_aggregated_diff.parquet",
    "household_calculated.parquet",
    "household_default.parquet",
    "household_diff_max_bounds.parquet",
    "household_imputed.parquet",
    "impute_gap_stats.parquet",
    "impute_summary_household.parquet",
    "impute_summary_project.parquet",
    "project_5min.parquet",
    "household_60min.parquet",
    "household_15min.parquet",
    "project_60min.parquet",
    "project_15min.parquet"
]

@pytest.fixture
def load_metadata():
    def _load_metadata(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    # return inner function as ficture
    return _load_metadata
