# One implementation per pipeline step

Each step of this pipeline has **one** implementation. It did not always: most
steps were built two or three times over -- pandas, ibis, DuckDB -- because each
was a *candidate*, written to be profiled against the others and chosen on
measured performance and scaling. Proving the candidates produced identical
output was the precondition for that comparison, which is what the equivalence
tests between them were for.

Those bake-offs are finished. This file records which implementation won each
step, so a reader does not have to infer it, and so a future change knows where
the one place to make it is.

The losing candidates were not deleted blind: each is archived with the
measurement or decision record that retired it, outside this repo in the
internal workflow repository's `archive/pipeline_variants/`. Their briefs are in
`docs/performance/RESULTS.md` there.

## The canonical chain

Stage numbers follow the pipeline documentation. There is no stage 2 -- the step
that assembled the mapped data into one wide artifact was removed, because the
assembling now happens while the data is read.

| Stage | Step | Function |
|---|---|---|
| 3 | Diff averages | `impute.prepare_diff_averages_sharded` -> `impute.calculate_average_diff_ibis` |
| 3 | Imputation | `aggregate.impute_mapped_households_sharded` -> `aggregate._impute_chunk_core` -> `vectorized_impute.impute_and_normalize` |
| 4 | Calculated columns | `calculated_stage.add_calculated_columns_to_hh_data_ibis` |
| 5 | Resampling | `aggregate.resample_hh_data_duckdb` with `aggregate._resample_sql_expr` |
| 6 | Project aggregation | `aggregate.aggregate_project_data_duckdb` |

The engine choices, in short: DuckDB wherever the work is a bulk scan or a
group-by over the whole population; ibis where expressions have to be built per
household from a catalog; pandas only inside the vectorised imputation core,
where the work is per household and already bounded.

Helpers the chain shares, which are not variants of each other:

| Function | Location | Purpose |
|---|---|---|
| `impute_and_normalize` | `vectorized_impute.py` | the vectorised imputer: gap detection, fill routing, threshold checks |
| `reconstruct_cumulative_columns` | `aggregate.py` | saves the original, rebuilds the cumulative column as a cumsum of its interval column, adds a Check column |
| `sort_for_impute` | `impute.py` | sorts by `ProjectIdBSV`, `HuisIdBSV`, `ReadingDate` |
| `read_diffs` | `impute.py` | loads the `avg_diffs` partitions with nullable dtypes |
| `sympy_to_ibis` | `calculated_columns.py` | translates a catalog formula into an ibis column expression |

`reconstruct_cumulative_columns_ibis` exists beside the pandas one and is
currently parked rather than canonical.

## Storage layouts and format detection

Reading and writing are not symmetric, and deliberately so.

- **Reading** is format-detected everywhere. A sharded folder source is read
  with per-supplier column differences unified by name; a single parquet file is
  read as a file. Callers never pass a format, which is what keeps an older
  single-file baseline readable. This applies to the whole chain, including the
  wide-stage statistics.
- **Writing** is not a choice: each stage writes the one shape its artifact has.
  The household stages and the resampled household intervals are hive shard
  trees, one shard per (household, batch). The project aggregates are single
  files, because project-level data is small, and that writer has no
  partitioning option at all.
- **Guards** run where computation starts. Duplicate (household, timestamp)
  rows, or a household with data in more than one batch, raise rather than being
  silently combined. A household source with no `HuisBatchIdBSV` is refused,
  because that is the key the output is filed under.

Reading both layouts is tested in `etdmap`, which owns the read layer
(`test_storage_source_layout.py`, `test_storage_read_source_frame.py`). The
`tests/test_sharded_*.py` files here assert this package's own behaviour: the
values each stage computes, and what its writer refuses.

## Cross-check tests

There are none, by design.

Three parallel pipelines and a frame-equality gate between them existed while
the engines were being chosen. With one implementation per step there is nothing
to compare against -- so each stage's tests assert values against expectations
computed by hand from the fixture instead. That is stronger than the comparison
it replaces, because two implementations can agree while both are wrong: a
retired equivalence test in this suite asserted a project mean of 4.5 where the
answer was 6.0, and passed for years because both paths shared the error.
