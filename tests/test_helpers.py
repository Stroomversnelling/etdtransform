import json
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from conftest import config, file_names

import etdtransform

etdtransform.options.aggregate_folder_path = Path(config['etdtransform_configuration']['aggregate_folder_path'])

def get_metadata_parquet_file(parquet_file):
    """Get metadata from a parquet file object"""
    metadata_dict = {
        "num_rows": parquet_file.metadata.num_rows,
        "num_columns": parquet_file.metadata.num_columns,
        "column_details": {}
    }
    # Extract schema information
    schema = parquet_file.schema

    # Extract metadata for each column
    for i in range(len(schema)):  # Use len(schema) instead of schema.num_fields
        column_name = schema.names[i]  # Correct way to get column names

        # Extract statistics from the first row group (if available)
        column_stats = {"min": None, "max": None, "null_count": None}
        if parquet_file.metadata.num_row_groups > 0:
            column_meta = parquet_file.metadata.row_group(0).column(i)
            stats = column_meta.statistics

            if stats:
                column_stats["min"] = str(stats.min) if stats.has_min_max else None
                column_stats["max"] = str(stats.max) if stats.has_min_max else None
                column_stats["null_count"] = stats.null_count if stats.has_null_count else None

        # Store in metadata dictionary
        metadata_dict["column_details"][column_name] = {
            "statistics": column_stats
        }

    return metadata_dict

def create_metadata_testfile_from_valid_run(path_to_file, save_name='metadata.json'):
    """Create metadata files of valid files.

    Done only to generate data which will be used to check
    if the outcome is as expected. Stores the metadatafiles
    in the data dir under the tests folder.

    Args:
        filename (str): filename of correctly generated file.
    """
    parquet_file = pq.ParquetFile(path_to_file)
    metadata_dict = get_metadata_parquet_file(parquet_file)

    # Save metadata as JSON
    with open(f"tests/data/{save_name}.json", "w") as f:
        json.dump(metadata_dict, f, indent=4)


def store_sample_from_valid_run(path_to_file, save_name):
    """
    Stores a reproducible sample of a valid outcome to be tested against.

    Args:
        path_to_file (str): Path to parquet file with data
    """
    df = pd.read_parquet(path_to_file)
    sample_size = min(100, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)

    store_folder = r"tests/data"
    sample_df.to_parquet(os.path.join(os.getcwd(), store_folder, f'{save_name}.parquet'), engine="pyarrow")


def store_sample_and_metadata_for_all_transformed_files(path_to_folder):

    for file in file_names:
        path_to_file = os.path.join(path_to_folder, file)
        name = file.split('.parquet')[0]
        create_metadata_testfile_from_valid_run(
            path_to_file,
            save_name=f'metadata_{name}'
            )
        store_sample_from_valid_run(
            path_to_file,
            save_name=f'sample_{name}')

if __name__ == "__main__":
    # Create metadata & sample files for each created file
    # by the different imputation steps in etdtransform
    # used for testing. Needs te be run only when the behaviour
    # of etdtransform has changed and the output is checked carefully.

    store_sample_and_metadata_for_all_transformed_files(etdtransform.options.aggregate_folder_path)
