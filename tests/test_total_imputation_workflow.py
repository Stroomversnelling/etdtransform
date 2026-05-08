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
from etdmap.data_model import test_aggregation_columns
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

def _filter_metadata_to_subset(metadata: dict, subset: set) -> dict:
    """Return a metadata dict whose column_details is filtered to keys in subset.

    num_rows is preserved (it's a structural property of the dataset).
    num_columns is recomputed to reflect the filtered column count so it stays
    self-consistent within the filtered metadata.
    """
    details = metadata.get("column_details", {}) or {}
    filtered_details = {k: v for k, v in details.items() if k in subset}
    return {
        "num_rows": metadata.get("num_rows"),
        "num_columns": len(filtered_details),
        "column_details": filtered_details,
    }


# ---------------------------------------------------------------------------
# Three independent fixture comparison tests, all scoped to
# etdmap.data_model.test_aggregation_columns. Filtering is applied to BOTH
# fixture and generated output BEFORE any check, so data-model expansion
# outside the subset is invisible to the test (we test the code, not the
# data model). Each test runs independently in pytest -- a failure in one
# does not short-circuit the others.
#
#   - schema test : column set diff in subset (added/removed) -> fail
#   - values test : row values diff in subset on sample -> fail
#   - stats test  : per-column min/max/null_count diff in subset -> fail
# ---------------------------------------------------------------------------

def test_files_schema_equal_expected():
    """
    Schema check: for each file in conftest.file_names, the column SET
    inside test_aggregation_columns must be identical between the stored
    fixture parquet and the freshly-generated parquet. Files with no overlap
    in the subset trivially pass.

    Failures separate additions (in generated, missing from fixture --
    fixture stale, regenerate) from removals (in fixture, missing from
    generated -- pipeline regression). Both kinds fail the test.
    """
    subset = set(test_aggregation_columns)
    failures: list[tuple[str, list[str], list[str]]] = []
    for name in conftest.file_names:
        name = name.split('.parquet')[0]
        fixture_path = Path(f"tests/data/sample_{name}.parquet")
        generated_path = os.path.join(etdtransform.options.aggregate_folder_path, f"{name}.parquet")
        if not fixture_path.exists() or not os.path.exists(generated_path):
            logging.warning(f"schema check skipped for {name}: missing fixture or generated file")
            continue
        # Schema-only reads -- parquet metadata, no data scan.
        fix_cols = set(pq.read_schema(fixture_path).names) & subset
        gen_cols = set(pq.read_schema(generated_path).names) & subset
        added = sorted(gen_cols - fix_cols)
        removed = sorted(fix_cols - gen_cols)
        if added or removed:
            failures.append((name, added, removed))
    if failures:
        msgs = []
        for name, added, removed in failures:
            parts = []
            if added:
                parts.append(f"in generated but not in fixture (regenerate fixtures): {added}")
            if removed:
                parts.append(f"in fixture but not in generated (pipeline regression): {removed}")
            msgs.append(f"  {name}: " + "; ".join(parts))
        raise AssertionError(
            f"Schema deviation in test_aggregation_columns subset for "
            f"{len(failures)} file(s):\n" + "\n".join(msgs)
        )


def test_files_values_equal_expected():
    """
    Value check: compare row-level values on the test_aggregation_columns
    subset between the stored sample parquet (random_state=42, n=100) and
    a re-sampled slice of the freshly-generated full parquet, using
    relative tolerance to absorb floating-point non-determinism from
    groupby aggregation.

    Files with no overlap in the subset trivially pass. Schema deviations
    are not this test's concern -- they are reported by
    test_files_schema_equal_expected.
    """
    subset = set(test_aggregation_columns)
    failures: list[tuple[str, list[str]]] = []
    for name in conftest.file_names:
        name = name.split('.parquet')[0]
        fixture_path = Path(f"tests/data/sample_{name}.parquet")
        generated_path = os.path.join(etdtransform.options.aggregate_folder_path, f"{name}.parquet")
        if not fixture_path.exists() or not os.path.exists(generated_path):
            logging.warning(f"value check skipped for {name}: missing fixture or generated file")
            continue
        # Determine subset columns present in each file from parquet metadata,
        # then load only those columns -- avoids scanning the ~80% of the
        # parquet that is outside the test subset.
        fix_subset_cols = list(set(pq.read_schema(fixture_path).names) & subset)
        gen_subset_cols = list(set(pq.read_schema(generated_path).names) & subset)
        shared = sorted(set(fix_subset_cols) & set(gen_subset_cols))
        if not shared:
            continue  # subset doesn't apply to this file
        fix_sub = pd.read_parquet(fixture_path, columns=shared)
        gen_full_sub = pd.read_parquet(generated_path, columns=shared)
        sample_size = min(100, len(gen_full_sub))
        gen_sub = gen_full_sub.sample(n=sample_size, random_state=42)
        differing: list[str] = []
        for col in shared:
            fa = fix_sub[col].reset_index(drop=True)
            ga = gen_sub[col].reset_index(drop=True)
            try:
                pd.testing.assert_series_equal(
                    fa, ga, check_exact=False, rtol=1e-10, check_names=False
                )
            except AssertionError:
                differing.append(col)
        if differing:
            failures.append((name, differing))
    if failures:
        msgs = [f"  {name}: differing columns: {cols}" for name, cols in failures]
        raise AssertionError(
            f"Value drift in test_aggregation_columns subset for "
            f"{len(failures)} file(s):\n" + "\n".join(msgs)
        )


def test_files_metadata_stats_equal_expected(load_metadata):
    """
    Metadata stats check: per-column statistics (min, max, null_count) in
    the stored fixture JSON must match the stats derived from the freshly
    generated FULL parquet, both filtered to the test_aggregation_columns
    subset. Compares only columns that appear in both after filtering --
    schema deviations are reported by test_files_schema_equal_expected.

    Files with no overlap in the subset trivially pass.
    """
    subset = set(test_aggregation_columns)
    failures: list[tuple[str, list[str]]] = []
    for name in conftest.file_names:
        name = name.split('.parquet')[0]
        expected_path = Path(f"tests/data/metadata_{name}.json")
        generated_path = os.path.join(etdtransform.options.aggregate_folder_path, f"{name}.parquet")
        if not expected_path.exists() or not os.path.exists(generated_path):
            logging.warning(f"stats check skipped for {name}: missing fixture or generated file")
            continue
        expected = load_metadata(expected_path)
        actual = generate_metadata_parquet_file(pq.ParquetFile(generated_path))
        expected_filt = _filter_metadata_to_subset(expected, subset)
        actual_filt = _filter_metadata_to_subset(actual, subset)
        exp_details = expected_filt.get("column_details", {}) or {}
        act_details = actual_filt.get("column_details", {}) or {}
        shared = sorted(set(exp_details.keys()) & set(act_details.keys()))
        diffs: list[str] = []
        for col in shared:
            col_diffs = _diff_json(exp_details[col], act_details[col], path=col)
            diffs.extend(col_diffs)
        if diffs:
            failures.append((name, diffs))
    if failures:
        msgs = []
        for name, diffs in failures:
            msgs.append(f"  {name}: {len(diffs)} stat diff(s)")
            for d in diffs[:5]:
                msgs.append(f"    {d}")
            if len(diffs) > 5:
                msgs.append(f"    ... ({len(diffs) - 5} more)")
        raise AssertionError(
            f"Metadata stat differences in test_aggregation_columns subset "
            f"for {len(failures)} file(s):\n" + "\n".join(msgs)
        )


if __name__ == "__main__":
    # Run pytest for debugging the testing
    pytest.main(["-v"])
