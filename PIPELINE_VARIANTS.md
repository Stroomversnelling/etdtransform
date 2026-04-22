# Pipeline function variants

Each pipeline step has two or three implementations. The pandas path is the
original reference. Ibis/DuckDB paths exist to reduce RAM footprint. Until the
ibis paths are fully validated and the old code deleted, both must be kept in
sync and cross-checked by `tests/test_pipeline_equivalence.py`.

## Step 1 -- Household aggregation

Reads individual mapped household parquets, combines into one aggregate file.

| Variant | Function | Output |
|---------|----------|--------|
| pandas | `aggregate_hh_data_5min()` | `household_default.parquet` |
| ibis (batched) | `aggregate_hh_data_5min_ibis()` | `household_default.parquet` |
| DuckDB (union) | `aggregate_hh_data_duckdb()` | `household_default.parquet` |

Notes:
- The ibis variant batches household files to keep expression trees small.
- The DuckDB variant uses `read_parquet(union_by_name=True)` and a JOIN on a
  filename-to-ID mapping table; no intermediate pandas accumulation.
- All three accept `sample_ratio` for stratified sampling by `ProjectIdBSV`.
- Cleanup candidate: keep DuckDB variant (fastest, no ibis expression tree
  size limit), delete pandas and ibis variants once validated.

## Step 2 -- Diff averages

Computes per-project average `{col}Diff` values used as imputation priors.

| Variant | Function | Input | Output |
|---------|----------|-------|--------|
| pandas | `prepare_diffs_for_impute(df, ...)` | pandas DataFrame | `avg_diffs.parquet`, `household_diff_max_bounds.parquet` |
| ibis | `prepare_diffs_for_impute_ibis(tbl, ...)` | ibis Table | same files |

Notes:
- Both call `calculate_average_diff` / `calculate_average_diff_ibis`
  respectively for the heavy aggregation, then share
  `concatenate_avg_diff_columns` and `concatenate_household_max_with_bounds`
  for the combine step (these are pandas, but operate on small result frames).
- Cleanup candidate: delete pandas variant once ibis path is validated.

### Sub-functions used in Step 2

| Function | Location | Called by | Purpose |
|----------|----------|-----------|---------|
| `calculate_average_diff` | `impute.py` | pandas variant | Per-project groupby mean of each `{col}Diff` |
| `calculate_average_diff_ibis` | `impute.py` | ibis variant | Same via ibis aggregation (DuckDB) |
| `concatenate_avg_diff_columns` | `impute.py` | both variants | Wide-pivots the per-column avg results into one frame |
| `concatenate_household_max_with_bounds` | `impute.py` | both variants | Combines per-household max bounds across columns |

## Step 3 -- Imputation

Fills missing values in cumulative columns using per-project diff averages.

| Variant | Function | Input | Output |
|---------|----------|-------|--------|
| pandas (full load) | `impute_hh_data_5min(df, ...)` | pandas DataFrame in memory | `household_imputed.parquet` |
| chunked pandas | `impute_hh_data_5min_chunked(source_path, ...)` | path to parquet | `household_imputed.parquet` |

Notes:
- Both variants use the same vectorised imputation engine
  (`vectorized_impute.impute_and_normalize`).
- The chunked variant reads `source_path` in batches of `chunk_size`
  households and streams results via `pyarrow.ParquetWriter`; RAM is
  bounded by chunk size rather than the full dataset.
- Both require `avg_diffs.parquet` and `household_diff_max_bounds.parquet`
  to already exist in `aggregate_folder_path`.
- Both variants call `reconstruct_cumulative_columns` (pandas) after
  imputation to rebuild the cumulative column values from their imputed
  Diff columns via per-household `cumsum`. The chunked variant does this
  per chunk (safe because chunks contain whole households).
- Cleanup candidate: delete full-load variant once chunked path is validated.

### Sub-functions used in Step 3

| Function | Location | Called by | Purpose |
|----------|----------|-----------|---------|
| `impute_and_normalize` | `vectorized_impute.py` | both imputation variants | Core vectorised imputer: gap detection, fill strategies, threshold checks |
| `impute_and_normalize_optimized` | `vectorized_impute.py` | `process_and_impute(optimized=True)` | Optimized variant (experimental, not in main path) |
| `reconstruct_cumulative_columns` | `aggregate.py` | full-load and chunked variants | Pandas: saves Original col, rebuilds col via cumsum of Diff, adds Check col |
| `reconstruct_cumulative_columns_ibis` | `aggregate.py` | (defined, not in main path) | Ibis/DuckDB lazy variant; has known TODO issues (36-level subquery depth, hangs on degenerate data) |
| `process_and_impute` | `impute.py` | `impute_hh_data_5min` | Orchestrates sort, diff load, merge, impute, stats save for full-load path |
| `sort_for_impute` | `impute.py` | both variants | Sorts by `ProjectIdBSV`, `HuisIdBSV`, `ReadingDate` |
| `read_diffs` | `impute.py` | both variants | Loads `avg_diffs.parquet` with `dtype_backend="numpy_nullable"` |

## Step 4 -- Calculated columns

Derives columns defined in `catalog.parquet` from imputed source columns.

| Variant | Function | Input | Output |
|---------|----------|-------|--------|
| pandas (full load) | `add_calculated_columns_to_hh_data(df, adaptive=True)` | pandas DataFrame in memory | `household_calculated.parquet` |
| ibis/DuckDB | `add_calculated_columns_to_hh_data_ibis(source_path, output_path)` | path to parquet | `household_calculated.parquet` |

Notes:
- The ibis variant runs a GROUP BY availability analysis in one DuckDB pass,
  groups households by schema (`frozenset` of available columns), derives
  each schema group with `ibis.mutate`, and combines via DuckDB
  `read_parquet(union_by_name=True)`.
- SymPy expressions are translated to ibis column expressions via
  `sympy_to_ibis` (uses `sp.lambdify`; ibis columns overload Python
  arithmetic operators).
- Cleanup candidate: delete pandas variant once ibis path is validated.

## Cross-check tests

`tests/test_pipeline_equivalence.py` runs all three pipelines on the test
fixture (10 households, 2 projects) and asserts frame equality using
`pd.testing.assert_frame_equal` with `rtol=1e-6`. Missingness is a
first-class value: `pd.NA` in one path and `0.0` in the other is a failure.

## Benchmarking

See `TODO.md` for the planned `bench_operations.py` script:
`--operation <name> --approaches pandas,ibis,duckdb --input-parquet <path> --sample-fraction 0.15`
