import json
import os

import pandas as pd
import pyarrow.parquet as pq


def create_metadata_testfiles_from_valid_run(path_to_file, save_name='metadata.json'):
    """Create metadata files of valid files.

    Done only to generate data which will be used to check
    if the outcome is as expected. Stores the metadatafiles 
    in the data dir under the tests folder.

    Args:
        filename (str): filename of correctly generated file.
    """
    parquet_file = pq.ParquetFile(path_to_file)

    # Extract schema and metadata
    metadata = {
        "schema": str(parquet_file.schema),  # Schema details
        "num_rows": parquet_file.metadata.num_rows,  # Total row count
        "num_row_groups": parquet_file.num_row_groups,  # Number of row groups
        "num_columns": parquet_file.metadata.num_columns,  # Total columns
        "file_size": parquet_file.metadata.serialized_size,  # File size in bytes
    }

    # Save metadata as JSON
    with open(f"tests/data/{save_name}", "w") as f:
        json.dump(metadata, f, indent=4)


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
    filenames = [
        "impute_summary_household.parquet",
        "impute_summary_project.parquet",
        "avg_diffs.parquet",
        "household_aggregated_diff.parquet",
        "household_calculated.parquet",
        "household_default.parquet",
        "household_diff_max_bounds.parquet",
        "household_imputed.parquet",
        "impute_gap_stats.parquet",
    ]

    for file in filenames:
        path_to_file = os.path.join(path_to_folder, file)
        create_metadata_testfiles_from_valid_run(
            path_to_file,
            save_name=f'metadata_{file.split('.parquet')[0]}.json'
            )
        store_sample_from_valid_run(
            path_to_file, 
            save_name=f'sample_{file.split('.parquet')[0]}.parquet')

if __name__ == "__main__":
    # Create metadata & sample files for each created file
    # by the different imputation steps in etdtransform
    # used for testing. Needs te be run only when the behaviour 
    # of etdtransform has changed and the output is checked carefully.
    store_sample_and_metadata_for_all_transformed_files(r'H:\Mijn Drive\Pion\Opdrachten\StroomVersnelling\temp_data\aggregate_fixtures')