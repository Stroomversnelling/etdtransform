# etdtransform
__"Energietransitie Dataset" transformation and loading package__

This package provides the required helpers to work with the `Energietransitie dataset` (ETD). The ETD is a model defining important variables for energy in the built environment, which are used to inform policy and planning decisions in the Netherlands. 

## Installation

## Configuration

To use most functions in this package, one needs to configure options so that the location of the mapped files created with `etdmap` and the location of aggregated data created with this package is defined up front.

```python	
import etdtransform

etdtransform.options.mapping_folder = 'mapping_folder_path' # path to folder where mapping files are stored
etdtransform.options.aggregate_folder = 'aggregate_folder_path' # path to folder where aggregated files are stored
etdtransform.options.weather_folder = 'KNMI_weather_data_folder_path' # path to the KNMI weather data folder
etdtransform.options.weather_file = 'path_to_KNMI_stations_file' # path to the KNMI weather stations data file
```

## Loading data as Ibis tables

Typically, we will first load data from parquet files stored in folders as Ibis tables and after the selecting the right columns and filtering the desired rows, and pontentially merging with other data, we will transform it to a in-memory format, such as a Pandas dataframe. This ensures that the data is loaded quickly and efficiently despite the large number of columns and records in the dataset. We will provide a few examples below.

There are two main types of ETD datasets: 

- data for connected unit in the built environment, such as a household or a charging point, and 
- project level data, an aggregated collection of network connected units. 

Building units are usually a single household and may represent an apartment or other home. The data from building units is described in the `etlmap` packge. At the moment, all bulding units are households. 

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

Projects are a collection of connected units. Often these were redeveloped at the same time or are geographically clustered. It can be a proxy for a neighborhood. Each project has also its own metadata.

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

## Transformations to prepare datasets for loading

After data from different data sources are mapped to the ETD data model using the `etdmap` package, data files are placed in the mapping folder. The mapping folder contains an index file and a file per connected building unit. In order to prepare these files for loading in analytical workflows and notebooks,  we first combine all data into a single dataset and then impute missing values where possible. Finally, we aggregate and resample data into the final datasets. 

While most users will not apply these transformations, at the moment, some of the following operations require large amounts of RAM and can surpass 100GB of RAM. Please ensure you have enough RAM available if you are processing the data.

### Combining individual building unit data

This step will create a single dataset and remove households marked in the household metadata to not be used (‘Meenemen’). Households with poor data, including missing variables or large amounts of missing data or other data anomolies that cannot be automatically cleaned are marked 0 so they are not included. All other households are marked 1 to include in our transformations.

For nearly 300 househoulds, it requires over 25GB of RAM. This function will save `household_default.parquet`.

```python
from etdtransform import aggregate_hh_data_5min

etdtransform.options.mapping_folder = 'mapping_folder_path' # path to folder where mapping files are stored
etdtransform.options.aggregate_folder = 'aggregate_folder_path' # path to folder where aggregated files are stored

aggregate_hh_data_5min() # this will generate the `household_default.parquet` file in the aggregate data folder
```

### Imputation to fill gaps

Ideally, each data provider has provided 5 minute resolution for all the devices. In some cases, not all variables are available at this level of granularity and in others interval may be 15 minutes or more. In cases where data is given at 5 minute intervals, we take columns that have longer intervals and impute missing values where possible.

Gaps in data are filled in different ways based on the patterns we found in the raw data. There are a few different imputation types of imputation that are used:
- Filling in 0s where there is a gap in data but no subsequent change in later cumulative values 
- Filling in 0s where the cumulative value have decreased but should not. For example, when an energy meter counter is reset to 0 and then increases again. The assumption is that there has been no real change.
- Filling in based on the project average change where there is no data in a household but data is available from other households in the same project. This average is scaled so that the change over the missing time period matches the next available cumulative value. The scaling factor is calculated with the ratio between actual change in the household and the project average change over the same period.
- Filling in linearly where there is insufficient data from other households to calcualte an average change over the period.

When project average change is used, the scaling factor is calculated with the ratio between:

- the difference between cumulative values in the household before and after the gap in data, e.g. if the last cumulative value was 222 and the next was 322, then the gap jump is 100, and 
- the sum of project change over the same period, e.g the average change in this variable over all households in the project is 50.

The calculated ratio would be 100/50 = 2 so each average change is multiplied by 2.0 and this values are used to fill in the gap.

Outliers are defined by first making a dataset per project of the maximum value per variable per household.  Then double the 95 percentile value per project is used as an upper bound for the household maximum. If the household maximum for the variable is within the bound, it is included in the averages difference calculation. By definition, this will exclude 0 to 5 percent of households per project so the resulting average should be representative and useful for imputation. 

#### Preparing for imputation of household data

For this step, the data is read from the `household_default.parquet` in the path provided.

```python
import etdtransform
from etdtransform.aggregate import read_hh_data
from etdtransform.impute import sort_for_impute, prepare_diffs_for_impute, impute_hh_data_5min

etdtransform.options.aggregate_folder = 'aggregate_folder_path' # path to folder where aggregated files are stored

df = read_hh_data(interval = 'default')
```

#### Calculating diff columns and average change per time step

We prepare the imputation by first calculating the differences between the consecutive values in cumulative columns to get the change every 5 minutes. These `diff` columns will be added to the dataframe. 

The average `diff` per 5 minutes in a project is used to impute values.  The average difference between timesteps for all cumulative variables is saved in a series of files by `prepare_diffs_for_impute()`. 

Before the average is calculated, households with outliers are removed. The upper bound is double the 95th percentile value per project. If the household maximum for the variable is within this bound, it is included in the average difference calculation. By definition, this excludes 0 to 5 percent of households per project so the resulting average should be representative and useful for imputation. 

Additional files saved to the aggregate folder are:

- `household_diff_max_bounds.parquet `: maximum differences per variable and the boundary values used to exclude outliers
- `avg_diffs.parquet`: the average difference per variable per project used for imputation

```python
# pre-sort rows to speed up the process
df = sort_for_impute(df, project_id_column='ProjectIdBSV')

# Optional: limit the operation to only a few columns of interest - otherwise leave the `cumulative_columns` parameter out
cum_cols_list = ['Zon-opwekTotaal', 'ElektriciteitsgebruikWarmtepomp']

prepare_diffs_for_impute(df, project_id_column='ProjectIdBSV', cumulative_columns=cum_cols_list, sorted = True)
```

#### Imputation

Finally, `impute_hh_data_5min()` will save the imputed dataset in the aggregate folder:

- `household_imputed.parquet`: the resulting imputed dataset with helper columns to identify imputation and methods of imputation

And a series of summary files:

- `impute_summary_project.parquet`: a summary of imputation per project
- `impute_summary_household.parquet`: a summary of imputation per household

It is possible to use arguments to pass pre-processed or sampled households to this function and to avoid sorting the column or recalculating differences if they have already been calculated previously. This can be useful as the dataset generated is very large with all the helper columns.

```python
impute_hh_data_5min(df, cum_cols=cum_cols_list, sorted=True, diffs_calculated=True)
```

__As of Fall 2024, most of the imputation is now much faster with the use of a series of simple vectorized operations instead of applying functions over the data. The trade off is that the imputation code is a little less readable.__

### Add calculated columns

```python

from etdtransform.aggregate import read_hh_data
from etdtransform.calculated_columns import add_calculated_columns_to_hh_data
etdtransform.options.aggregate_folder = 'aggregate_folder_path' # path to folder where aggregated files are stored

df_imputed = read_hh_data(interval="imputed")
df_calculated = add_calculated_columns_to_hh_data(df_imputed)

```

### Resampling to different time intervals

```python

from etdtransform.aggregate import read_hh_data
from etdtransform.calculated_columns import add_calculated_columns_to_hh_data
etdtransform.options.aggregate_folder = 'aggregate_folder_path' # path to folder where aggregated files are stored

# all files are saved to household_[interval].parquet in the aggregate folder, e.g. household_5min.parquet
resample_hh_data(intervals=["5min"]) # loads the household_calculated.parquet file and retains 5 minute intervals and drops unnecessary columns.
resample_hh_data(intervals=["60min", "15min"]) # loads the household_calculated.parquet file and resamples to 60min and 15min intervals in two separate datasets and drops unnecessary columns.

```

### Aggregation of household data to project level data


```python

from etdtransform.aggregate import read_hh_data
from etdtransform.calculated_columns import add_calculated_columns_to_hh_data
etdtransform.options.aggregate_folder = 'aggregate_folder_path' # path to folder where aggregated files are stored

# all files are saved to household_[interval].parquet in the aggregate folder, e.g. household_5min.parquet
resample_hh_data(intervals=["5min"]) # loads the household_calculated.parquet file and retains 5 minute intervals and drops unnecessary columns.
resample_hh_data(intervals=["60min", "15min"]) # loads the household_calculated.parquet file and resamples to 60min and 15min intervals in two separate datasets and drops unnecessary columns.

# all files are saved to project_[interval].parquet in the aggregate folder, e.g. project_5min.parquet
aggregate_project_data(intervals=["5min"]) # aggregates all households and calculated project averages for all 5 minute intervals
aggregate_project_data(intervals=["60min", "15min"]) # aggregates all households and calculated project averages for resampled 60min and 15min intervals

```

## Loading complete datasets at once

It is not recommended to load the complete datasets in one go. It will take a lot of time to load and require a significant amount of RAM to store all the data in memory at once. Rather use `get_hh_table()` as described above.`

Some of the older workflows still do this.

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

## Weather data

Weather data is downloaded from KNMI and combined with project metadata to identify the closest weather station for each project. Weather data is merged with household and project datasets based on the available 1 hour resolution data. This occurs during the loading step and can be skipped to speed up data processing during analysis should the weather data not be required.











