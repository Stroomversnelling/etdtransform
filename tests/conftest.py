"""
Shared test configuration for etdtransform.

Test fixture dataset: paths are loaded from config_test.yaml (see
config_test_template.yaml for the committed reference template). This is a
small anonymised dataset -- NOT production data. All tests in this suite must
pass against that fixture; there are no "pre-existing" failures (see ADR-007).
"""
import json
import logging
import os
from pathlib import Path

import etdmap
import pytest
import yaml

import etdtransform


def pytest_configure(config):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler("test.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

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


@pytest.fixture(scope="session", autouse=True)
def _require_etdmap_mapped_fixtures():
    """
    Hard-fail at session start if testdata/mapped/ is not populated.

    etdtransform pipeline tests read household_*.parquet and index.parquet
    from `etdtransform.options.mapped_folder_path` as their input. Those
    files are produced by etdmap's `mapped_fixtures` session fixture
    (etdmap/tests/conftest.py), which runs when an etdmap test that
    requests it is collected -- e.g. test_index_helpers.py tests.

    Without that population, etdtransform tests fail in non-obvious ways:
    `update_meenemen()` errors with "Household mismatch", or pipeline
    functions silently process empty input. We fail loudly here with a
    clear instruction instead.

    Fixture invalidation across commits is currently parked -- if a code
    change requires regenerating the fixtures, manually re-run the etdmap
    suite. (See ADR-007 for fixture regeneration discipline.)
    """
    import pyarrow.parquet as pq
    mapped_path = Path(config['etdtransform_configuration']['mapped_folder_path'])
    index_parquet = mapped_path / "index.parquet"
    instructions = (
        "etdtransform tests require testdata/mapped/ to be populated by etdmap's "
        "test fixtures (the `mapped_fixtures` session fixture in "
        "etdmap/tests/conftest.py). Run etdmap's test suite first:\n"
        "    cd ../etdmap && .venv/Scripts/python -m pytest tests/ -v\n"
        "Then re-run etdtransform tests."
    )
    if not index_parquet.exists():
        pytest.exit(
            f"FIXTURE PRECONDITION FAILED: {index_parquet} does not exist.\n\n"
            + instructions,
            returncode=2,
        )
    try:
        nrows = pq.read_metadata(index_parquet).num_rows
    except Exception as exc:
        pytest.exit(
            f"FIXTURE PRECONDITION FAILED: cannot read {index_parquet} ({exc}).\n\n"
            + instructions,
            returncode=2,
        )
    if nrows == 0:
        pytest.exit(
            f"FIXTURE PRECONDITION FAILED: {index_parquet} exists but has 0 rows. "
            f"This typically means etdmap's session-start cleanup ran but no test "
            f"requested mapped_fixtures.\n\n"
            + instructions,
            returncode=2,
        )
