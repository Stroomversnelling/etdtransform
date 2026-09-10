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
import etdmap.data_model
import etdmap.index_helpers
import ibis
import pandas as pd
import pytest
import yaml

import etdtransform
from etdtransform.aggregate import (
    add_calculated_columns_to_hh_data,
    add_calculated_columns_to_hh_data_ibis,
    aggregate_hh_data_5min,
    aggregate_hh_data_5min_ibis,
    aggregate_hh_data_duckdb,
    impute_hh_data_5min,
    impute_hh_data_5min_chunked,
    read_hh_data,
)
from etdtransform.impute import prepare_diff_averages_for_impute, prepare_diff_averages_for_impute_ibis


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
    `update_include()` errors with "Household mismatch", or pipeline
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


# ---------------------------------------------------------------------------
# Shared pipeline fixtures.
#
# These run the full aggregate -> diffs -> impute -> calc pipeline ONCE per
# session into an isolated tmpdir, with three implementations: pandas, ibis,
# and duckdb (aggregate via duckdb, rest via pandas). Tests across the whole
# etdtransform suite share these outputs to compare implementations and
# verify downstream stages (resample, project aggregation, etc.).
#
# Previously each test file (test_pipeline_equivalence.py, test_resample_duckdb.py)
# defined its own copy of the ibis pipeline as a session fixture, causing the
# same pipeline to run twice -- ~128s of duplicated work per suite run. By
# lifting these to conftest, both files share the same fixture instance.
# ---------------------------------------------------------------------------

def _run_pandas_pipeline(out_dir: Path, cum_cols: list) -> None:
    old = etdtransform.options.aggregate_folder_path
    etdtransform.options.aggregate_folder_path = out_dir
    try:
        aggregate_hh_data_5min()
        df = read_hh_data(interval="default")
        prepare_diff_averages_for_impute(
            df, project_id_column="ProjectIdBSV", cumulative_columns=cum_cols, sorted=False
        )
        # sorted=False: prepare_diff_averages_for_impute sorts a local copy internally and does
        # not return it, so df is still unsorted here. Always sort before imputing.
        df_imputed = impute_hh_data_5min(
            df, cum_cols=cum_cols, sorted=False, diffs_calculated=True
        )
        # adaptive=True: matches the ibis path which uses the adaptive catalog-driven approach
        add_calculated_columns_to_hh_data(df_imputed, adaptive=True)
    finally:
        etdtransform.options.aggregate_folder_path = old


def _run_ibis_pipeline(out_dir: Path, cum_cols: list) -> None:
    old = etdtransform.options.aggregate_folder_path
    etdtransform.options.aggregate_folder_path = out_dir
    try:
        aggregate_hh_data_5min_ibis()
        tbl = ibis.read_parquet(str(out_dir / "household_default.parquet"))
        prepare_diff_averages_for_impute_ibis(
            tbl, project_id_column="ProjectIdBSV", cumulative_columns=cum_cols
        )
        impute_hh_data_5min_chunked(
            source_path=out_dir / "household_default.parquet",
            cum_cols=cum_cols,
        )
        add_calculated_columns_to_hh_data_ibis(
            source_path=str(out_dir / "household_imputed.parquet"),
            output_path=str(out_dir / "household_calculated.parquet"),
        )
    finally:
        etdtransform.options.aggregate_folder_path = old


def _run_duckdb_pipeline(out_dir: Path, cum_cols: list) -> None:
    """DuckDB aggregation + pandas diffs + pandas impute + pandas adaptive calc columns."""
    old = etdtransform.options.aggregate_folder_path
    etdtransform.options.aggregate_folder_path = out_dir
    try:
        aggregate_hh_data_duckdb()
        df = read_hh_data(interval="default")
        prepare_diff_averages_for_impute(
            df, project_id_column="ProjectIdBSV", cumulative_columns=cum_cols, sorted=False
        )
        df_imputed = impute_hh_data_5min(
            df, cum_cols=cum_cols, sorted=False, diffs_calculated=True
        )
        add_calculated_columns_to_hh_data(df_imputed, adaptive=True)
    finally:
        etdtransform.options.aggregate_folder_path = old


@pytest.fixture(scope="session")
def _include_updated():
    """Populate Meenemen from bsv_metadata_file before any pipeline runs.

    aggregate_hh_data_5min (and ibis/duckdb variants) filter on Meenemen == 1.
    Without this call the index has no Meenemen == True rows and every pipeline
    exits early with no output.

    update_include saves through save_index_to_parquet, which writes the
    index.parquet (including the household-batch
    columns). That requires the fixture HuisBatch sync CSV that etdmap's
    conftest writes next to the fixture BSV metadata -- checked here so a
    missing prerequisite fails with a pointer instead of a skipped index.
    """
    _sync_csv = Path(str(etdmap.options.bsv_metadata_file)).parent / "household_batch_sync.csv"
    etdmap.options.household_batch_csv_path = _sync_csv
    if not _sync_csv.exists():
        raise FileNotFoundError(
            f"Fixture HuisBatch sync CSV not found at {_sync_csv}. Run "
            f"etdmap's test suite first (its conftest writes it)."
        )
    etdmap.index_helpers.update_include()


@pytest.fixture(scope="session")
def _cum_cols():
    return etdmap.data_model.cumulative_columns[:10]


@pytest.fixture(scope="session")
def pandas_pipeline(tmp_path_factory, _cum_cols, _include_updated):
    out = tmp_path_factory.mktemp("pandas_pipeline")
    _run_pandas_pipeline(out, _cum_cols)
    return out


@pytest.fixture(scope="session")
def ibis_pipeline(tmp_path_factory, _cum_cols, _include_updated):
    out = tmp_path_factory.mktemp("ibis_pipeline")
    _run_ibis_pipeline(out, _cum_cols)
    return out


@pytest.fixture(scope="session")
def duckdb_pipeline(tmp_path_factory, _cum_cols, _include_updated):
    out = tmp_path_factory.mktemp("duckdb_pipeline")
    _run_duckdb_pipeline(out, _cum_cols)
    return out


# ---------------------------------------------------------------------------
# No skipped tests (ADR-007). A skip hides a missing prerequisite or an
# unmet condition as "not run", which reads as green when it is not. This
# hook turns ANY skip -- pytest.skip(), skip/skipif markers, importorskip --
# into a hard failure at runtime.
#
# THE RULE: if a test is there, you may not write skipping in code. Make the
# prerequisite available (build the fixture/artifact it needs), or let the
# test fail so the gap is visible. Do not add pytest.skip.
# ---------------------------------------------------------------------------
import pytest as _pytest

# Only these classes of skip may exist. A skip is allowed when its reason
# begins with "<TOKEN>: " and TOKEN is a key below.
#
# The allowlist is EMPTY on purpose. No skip in this code base has yet been
# shown to be legitimate -- every one examined turned out to be a missing
# prerequisite that should fail loudly instead. Empty means every skip is a
# hard failure until someone argues a class onto the list IN REVIEW. Adding a
# key is a deliberate, visible decision; deciding in the moment is not.
ALLOWED_SKIP_CLASSES = {
    # "PLATFORM": "behaviour genuinely differs by OS and is covered elsewhere",
    # "OPTIONAL_DEP": "an optional third-party package is not installed",
}


def _skip_reason(report):
    """
    The reason text of a skip.

    report.longrepr for a skip is the tuple (path, lineno, "Skipped: <reason>")
    for both a skipif marker and an in-body pytest.skip(). Verified against
    pytest directly rather than assumed.
    """
    longrepr = report.longrepr
    if isinstance(longrepr, tuple) and len(longrepr) >= 3:
        text = str(longrepr[2])
    else:
        text = str(longrepr)
    prefix = "Skipped: "
    return text[len(prefix):] if text.startswith(prefix) else text


@_pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if not report.skipped:
        return
    reason = _skip_reason(report)
    token = reason.split(":", 1)[0].strip() if ":" in reason else ""
    if token in ALLOWED_SKIP_CLASSES:
        return
    allowed = sorted(ALLOWED_SKIP_CLASSES) or ["(none -- the allowlist is empty)"]
    report.outcome = "failed"
    report.longrepr = (
        "SKIP NOT ALLOWED (ADR-007).\n"
        "  reason given: %r\n"
        "  skip class  : %r\n"
        "  allowed     : %s\n"
        "If a test is there, it must run. Provide the prerequisite it needs, or\n"
        "let it fail so the gap is visible. If a skip is genuinely warranted,\n"
        "add its class to ALLOWED_SKIP_CLASSES in review -- never silently."
        % (reason, token, allowed)
    )
