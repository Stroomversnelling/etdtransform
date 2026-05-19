# About etdtransform

The `etdtransform` package provides the required helpers to work with the `Energietransitie Dataset` (ETD). The ETD is a model defining important variables for energy in the built environment, which are used to inform policy and planning decisions in the Netherlands. For an overview of the ETD and all documentation, see <a href="https://energietransitiedataset.nl/">https://energietransitiedataset.nl/</a>.

It depends on `etdmap` for the ETD data model definitions and for some data loading functions. It is expected that any datasets used have already been mapped and undergone basic quality control. See `etdmap` for more information.

# License

The package may only be used for open processing. You agree to publicly share the intended application, obtained insights, and applied calculation methods under the same license.

# Citation

If you use this package or any code in this package or refer to output, please use the following citations for attribution:

_Witkamp, Dickinson, Izeboud (2024). etdtransform: A Python package for improving the data quality of data in the Energietransitie Dataset (ETD) model._

# Installation

If you want to use this package on its own, you can install it with pip:

```bash
git clone https://github.com/Stroomversnelling/etdtransform.git
cd etdtransform
pip install .
```

# Developing and contributing

If you would like to contribute to the package code, we would create an environment and install it in editable mode:

```bash	
git clone https://github.com/Stroomversnelling/etdtransform.git
cd etdtransform
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
pip install -e .
```

## Test fixtures

The test suite runs the aggregation/imputation pipeline on the anonymised
synthetic test dataset (configured via `config_test.yaml`, see
`config_test_template.yaml` for the committed reference) and compares the
freshly-generated output against committed fixture files in `tests/data/`:

- `metadata_*.json` -- per-column statistics summary of each pipeline output.
- `sample_*.parquet` -- a deterministic 100-row sample of each pipeline output
  (`df.sample(n=100, random_state=42)`).

Both are derived from anonymised synthetic data per ADR-007 (see
`../DECISIONS.md`); they are not production data. The synthetic input is
sourced from `etdmap`'s test fixtures, which themselves are fully
deterministic (PCG64 seed=42).

`*.parquet` and `*.csv` are gitignored by default; the fixture files are
re-included via explicit `!tests/data/sample_*.parquet` and
`!tests/data/metadata_*.json` exceptions in `.gitignore`.

To regenerate the fixtures after an intentional pipeline change, ensure
`testdata/aggregate/` is freshly populated by running the etdtransform
workflow, then from the repo root:

```bash
python tests/test_helpers.py
```

This overwrites `tests/data/metadata_*.json` and `tests/data/sample_*.parquet`.
Per ADR-007, fixture regeneration must be deliberate -- review the diff with
`git diff tests/data/` before committing, and confirm the change is the
expected consequence of a known pipeline update rather than a silent drift.

### Comparison strategy: subset-scoped, three independent checks

Fixture comparisons in `tests/test_total_imputation_workflow.py` run as
three independent pytest tests, each scoped to the test subset
`etdmap.data_model.test_aggregation_columns`:

- `test_files_schema_equal_expected` -- fails if the column SET inside the
  subset differs between fixture and generated (reports added vs removed
  separately).
- `test_files_values_equal_expected` -- fails if any value in the subset
  drifts between the stored 100-row sample and a re-sampled slice of the
  generated full parquet (relative tolerance `1e-10`).
- `test_files_metadata_stats_equal_expected` -- fails if per-column
  statistics (min, max, null_count) drift in the subset.

Filtering is applied to BOTH fixture and generated BEFORE any check, so
data-model expansion outside the subset never breaks tests. Adding a
variable to the comparison surface is a deliberate code change: extend
`etdmap.data_model.test_aggregation_columns` and add plausible synthetic
ranges for the new variable in `etdmap/tests/conftest.py::raw_data_fixture`.

The three tests run independently. A failure in one does not short-circuit
the others, so a regeneration-only situation (schema drift, no value drift)
produces a clear "regenerate fixtures" signal distinct from a real
regression.

# Configuration

To use most functions in this package, one needs to configure options so that the location of the mapped files created with `etdmap` and the location of aggregated data created with this package is defined up front.

```python
import etdtransform

etdtransform.options.mapped_folder = 'mapped_folder_path' # path to folder where mapped files are stored
etdtransform.options.aggregate_folder = 'aggregate_folder_path' # path to folder where aggregated files are stored
etdtransform.options.weather_folder = 'KNMI_weather_data_folder_path' # path to the KNMI weather data folder
etdtransform.options.weather_file = 'path_to_KNMI_stations_file' # path to the KNMI weather stations data file
```

# Loading mapped data as Ibis tables

Typically, we will first load data from the mapped parquet files stored in the configured folders. We prefer to load them as Ibis tables and - after selecting the appropriate columns and filtering the desired rows, and merging with other required data - transforming them to an in-memory format, such as a Pandas dataframe. This ensures that the data is loaded quickly and efficiently despite the large number of columns and records in the dataset. We will provide a few examples below.

There are two main types of aggregated ETD datasets:

- data for individually connected units in the built environment, such as a household or, perhaps in the future, a charging point, and
- project level data, an aggregated collection of network connected units based on `ProjectIdBSV` and often representing a limited geographic scope, e.g. a neighborhood.

Building units are usually a single household and may represent an apartment or other type of home. The data from building units is described in the `etdmap` package. 

To load household data, use the following:

```python
from etdtransform.load_data import load_household_tables
import ibis
from ibis import _
import ibis.selectors as s

etdtransform.options.aggregate_folder = 'aggregate_folder_path' # path to folder where aggregated files are stored

desired_interval = '5min' # 5min, 15min, 60min, 24h
tbls = load_household_tables()

# load only data for a specific interval, specific columns
household_5min = tbls['5min'].select(
    'ReadingDate',
    'ProjectIdBSV',
    'HuisIdBSV',
    'ElektriciteitNetgebruikHoog',
    'ElektriciteitNetgebruikLaag'
)

# Use selectors instead to select using the variable suffixes for 'ProjectIdBSV', 'HuisIdBSV' and prefixes for 'ElektriciteitNetgebruikHoog', 'ElektriciteitNetgebruikLaag'
household_5min = tbls['5min'].select(
    'ReadingDate',
    s.endswith('BSV'),
    s.startswith('ElektriciteitNetgebruik')
)

# filter out only households from project 1.0 with a surface of 100 m2 or more
household_5min = household_5min.filter(
    _.ProjectIdBSV = 1.0,
    _.Oppervlakte >= 100,
)

# convert to pandas for further processing or graphing
household_df = household_5min.to_pandas()

```

Projects are a collection of connected units (mostly: homes) with similar characteristics. Often these were redeveloped at the same time using similar insulation and installation choices. It can be a proxy for a neighborhood. Each project has its own metadata.

```python

from etdtransform.load_data import load_project_tables
import ibis
from ibis import _
import ibis.selectors as s

etdtransform.options.aggregate_folder = 'aggregate_folder_path' # path to folder where aggregated files are stored

desired_interval = '5min' # 5min, 15min, 60min, 24h
project_tbls = load_project_tables()

# load only data for a specific interval, specific columns
project_24h = project_tbls['24h'].select('ReadingDate','ProjectIdBSV', 'ElektriciteitsgebruikTotaalNetto')

# exclude project 6.0
project_24h = household_5min.filter(
    _.ProjectIdBSV != 6.0
)

# convert to pandas for further processing or graphing
project_df = project_24h.to_pandas()

```

To load data from a set before the resampling and aggregation, it is possible to use `get_hh_table()` and pass the function the following non-standard intervals:

- default: load household data before any imputation has been done
- imputed: load household data that has been imputed, including all `is_imputed` columns for indentifying values as imputed
- calculated: same as 'imputed' but with the addition of calculated columns

For example, to get columns that are used to calculate Electricity Net Usage `ElektriciteitsgebruikTotaalNetto` and the columns that identify if any of these values are imputed, we can do the following.

```python
import ibis
from ibis import _
import ibis.selectors as s
from etdtransform.load_data import get_hh_table
etdtransform.options.aggregate_folder = 'aggregate_folder_path' # path to folder where aggregated files are stored

df_calculated = get_hh_table('5min', 'calculated').select(
    'ReadingDate',
    'ProjectIdBSV',
    'HuisIdBSV',
    'ElektriciteitNetgebruikHoog',
    'ElektriciteitNetgebruikLaag',
    'ElektriciteitNetgebruikHoog_is_imputed', # True or False
    'ElektriciteitNetgebruikLaag_is_imputed', # True or False
    'ElektriciteitsgebruikTotaalNetto' # calculated from Hoog and Laag variables
).to_pandas()
```

# Transformations to prepare datasets for loading (advanced)

After data from different data sources are mapped to the ETD data model using the `etdmap` package, data files are placed in the mapped folder. The mapped data folder contains an index file and a file per connected building unit. In order to prepare these files for loading in analytical workflows and notebooks,  we first combine all data into a single dataset and then impute missing values where possible. Finally, we aggregate and resample data into the final datasets.

_While most users will not need to apply these transformations as the datasets will already have been aggregated, some of the following operations require large amounts of RAM and can surpass 100GB of RAM to run efficiently. Please consider whether you have enough RAM and time available if you are processing the data. In the future, we could reduce the RAM requirements if needed._

## Combining individual building unit data

This step will create a single dataset and remove households marked in the household metadata to not be used (‘Meenemen’). Households with poor data, including missing variables or large amounts of missing data or other data anomolies that cannot be automatically cleaned are marked 0 so they are not included. All other households are marked 1 to include in our transformations.

For nearly 300 househoulds, it requires over 25GB of RAM to run efficiently. This function will save `household_default.parquet`.

```python
from etdtransform import aggregate_hh_data_5min

etdtransform.options.mapped_folder = 'mapped_folder_path' # path to folder where mapping files are stored
etdtransform.options.aggregate_folder = 'aggregate_folder_path' # path to folder where aggregated files are stored

aggregate_hh_data_5min() # this will generate the `household_default.parquet` file in the aggregate data folder
```

## Imputation to fill gaps

Ideally, each data provider has provided 5 minute resolution for all the devices. In some cases, not all variables are available at this level of granularity and in others interval may be 15 minutes or more. In cases where data is given at 5 minute intervals, we take columns that have longer intervals and impute missing values where possible.

Gaps in data are filled in different ways based on the patterns we found in the raw data. There are a few different imputation techniques that are used to impute the **expected change in cumulative variables**:
- Filling in 0s where there is a gap in data but no subsequent change in later cumulative values
- Filling in 0s where the cumulative value has decreased but should not normally. For example, when a cumulative energy meter counter jumps down to 0 and then increases again to the value it had before the jump to 0. The assumption is that there has been no real change and there was a data registration anomaly.
- Filling in based on the _project average change_ where there is no data in a household but data is available from other households in the same project. This average is scaled so that the total change over the missing time period matches the next available cumulative value. The scaling factor is calculated with the ratio between actual change in the household and the project average change over the same period.
- Filling in linearly where there is insufficient data from other households to calculate an average change over the period. This is only applied on cumulative variables and when the missing data covers a small time period.

Some of these imputation methods will reduce the variance of the dataset. Taking this into account, it is important to exclude datasets with too much missing data and consider this during analysis and interpretation as several measures such as the IQR or standard deviation may be sensitive to the imputation. By carefully selecting datasets and ensure project data is never missing from many households at any one point in time, these effects are very small.

When the project average change is used, the scaling factor is calculated with the ratio between:

- the difference between cumulative values in the household before and after the gap in data, e.g. if the last cumulative value was 222 and the next was 322, then the gap jump is 100, and
- the sum of project change over the same period, e.g the average change in this variable over all households in the project, for this example taken to be 50.

The calculated ratio would be 100/50 = 2 so each average change is multiplied by 2.0 and these scaled values are used to fill in the gap.

Before the average is calculated, households with outliers are removed. The upper bound is double the 95th percentile value per project per registration at a specific time. If the household maximum for the variable is within this bound, it is included in the average difference calculation. By definition, this includes 100% to 95% of households in each registered moment per project so the resulting average should be representative and useful for imputation.

### 1. Loading mapped data

For this step, the data is read from the `household_default.parquet` in the path provided.

```python
import etdtransform
from etdtransform.aggregate import read_hh_data
from etdtransform.impute import sort_for_impute, prepare_diffs_for_impute, impute_hh_data_5min

etdtransform.options.aggregate_folder = 'aggregate_folder_path' # path to folder where aggregated files are stored

df = read_hh_data(interval = 'default')

```

### 2. Calculating diff columns and average change per time step

We prepare the imputation by first calculating the differences between the consecutive values in cumulative columns to get the change every 5 minutes. These `diff` columns will be added to the dataframe.

The average `diff` per 5 minutes in a project is used to impute values.  The average difference between timesteps for all cumulative variables is saved in a series of files by `prepare_diffs_for_impute()`.

```python
# pre-sort rows to speed up the process
df = sort_for_impute(df, project_id_column='ProjectIdBSV')

# Optional: limit the operation to only a few columns of interest - otherwise leave the `cumulative_columns` parameter out
cum_cols_list = ['Zon-opwekTotaal', 'ElektriciteitsgebruikWarmtepomp']

prepare_diffs_for_impute(df, project_id_column='ProjectIdBSV', cumulative_columns=cum_cols_list, sorted = True)
```

Additional files saved to the aggregate folder are:

- `household_diff_max_bounds.parquet `: maximum differences per variable and the boundary values used to exclude outliers
- `avg_diffs.parquet`: the average difference per variable per project used for imputation

### Imputation

Finally, `impute_hh_data_5min()` will save the imputed dataset in the aggregate folder.

```python
impute_hh_data_5min(df, cum_cols=cum_cols_list, sorted=True, diffs_calculated=True)
```

Files are:

- `household_imputed.parquet`: the resulting imputed dataset with helper columns to identify imputation and methods of imputation
- `impute_summary_project.parquet`: a summary of imputation per project
- `impute_summary_household.parquet`: a summary of imputation per household

It is possible to use arguments to pass pre-processed or sampled households to this function and to avoid sorting the column or recalculating differences if they have already been calculated previously. This can be useful as the dataset generated is very large.

__As of Fall 2024, most of the imputation is now much faster with the use of a series of simple vectorized operations instead of applying functions over the data. The trade off is that the imputation code is a little less readable.__

## Add calculated columns

```python

from etdtransform.aggregate import read_hh_data
from etdtransform.calculated_columns import add_calculated_columns_to_hh_data
etdtransform.options.aggregate_folder = 'aggregate_folder_path' # path to folder where aggregated files are stored

df_imputed = read_hh_data(interval="imputed")
df_calculated = add_calculated_columns_to_hh_data(df_imputed)

```

The ibis/DuckDB variant (`add_calculated_columns_to_hh_data_ibis`) groups households by
shared available-column schema and writes per-group temp parquets, then merges them using
Polars streaming (`diagonal_relaxed`). This is 6x faster than DuckDB `union_by_name` at
scale and does not materialise the full dataset in RAM.
See `../etdworkflow/docs/performance/parquet-merge-strategy-2026-04-22.md` for the full benchmark record and implementation notes.

## Resampling to different time intervals

`resample_hh_data_duckdb` reads `household_calculated.parquet` and writes one
`household_{interval}.parquet` per requested interval. Resampling config (which
columns to include, which aggregation method to apply) is read from
`etdmap.data_model.get_aggregation_config()` so it stays in sync with the data model.

- **5min**: column selection only — no aggregation, rows are unchanged.
- **15min / 60min / 6h / 24h**: TIME_BUCKET groupby per (household, project).
  Diff columns use SUM, other columns use their configured method (avg, max, ...).
  A NULL result is returned for any bucket where fewer than `bucket_minutes / 5`
  non-null source readings exist (preserves missingness, ADR-005).
  Cumulative counterparts of Diff columns (e.g. `ElektriciteitNetgebruikHoog` from
  `ElektriciteitNetgebruikHoogDiff`) are automatically rebuilt via window cumsum.

```python
import etdtransform
from etdtransform.aggregate import resample_hh_data_duckdb

etdtransform.options.aggregate_folder_path = 'aggregate_folder_path'

# Writes household_5min.parquet, household_15min.parquet, household_60min.parquet
resample_hh_data_duckdb(
    source_path='aggregate_folder_path/household_calculated.parquet',
    output_dir='aggregate_folder_path',
    intervals=("5min", "15min", "60min"),
)

# Also supports 6h and 24h
resample_hh_data_duckdb(
    source_path='aggregate_folder_path/household_calculated.parquet',
    output_dir='aggregate_folder_path',
    intervals=("6h", "24h"),
)
```

## Aggregation of household data to project level data

`aggregate_project_data_duckdb` reads each `household_{interval}.parquet` produced by
the resample step and aggregates to project level using the method defined in the data
model per column (avg or sum across households). Diff columns produce cumulative
counterparts at project level via the same window cumsum convention.

```python
import etdtransform
from etdtransform.aggregate import aggregate_project_data_duckdb

etdtransform.options.aggregate_folder_path = 'aggregate_folder_path'

# Writes project_5min.parquet, project_15min.parquet, project_60min.parquet
aggregate_project_data_duckdb(intervals=("5min", "15min", "60min"))

# Also supports 6h and 24h
aggregate_project_data_duckdb(intervals=("6h", "24h"))
```

Both functions use a single DuckDB pass per interval (no full pandas load) and read
column configuration from `etdmap.data_model.get_aggregation_config()` at runtime.

# Loading complete datasets at once

It is not recommended to load the complete datasets in one go. It will take a lot of time to load and require a significant amount of RAM to store all the data in memory at once. Rather use `get_hh_table()` as described above.`

Some of the older workflows still do this:

```python
from etdtransform.aggregate import read_hh_data

etdtransform.options.aggregate_folder = 'aggregate_folder_path' # path to folder where aggregated files are stored

# Load household data before any imputation has been done
df_combined  = read_hh_data(interval="default")

# Load household data that has been imputed, including all `is_imputed` columns for indentifying values as imputed
df_imputed = read_hh_data(interval="imputed")

# Load household data that has been imputed with additional calculated columns
df_calculated = read_hh_data(interval="calculated")

# Load household data that has been
df_5min = read_hh_data(interval="5min")

```

# Weather data

Weather data is downloaded from KNMI and combined with project metadata to identify weather data for the closest weather station for each project. Weather data is merged with household and project datasets based on the available 1 hour resolution data. This occurs during the loading step and can be skipped to speed up data processing during analysis should the weather data not be required.











