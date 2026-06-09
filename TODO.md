# etdtransform -- TODO

## FysischModel: per-household physical system model for model-specific derivation rules

### Background

Some optional heat-pump sub-devices (Booster, WTW, Radiator, Boilervat) are only present
for certain suppliers or certain households within a project. Without knowing whether a
device is "absent" or just "not yet reported", `add_calculated_columns_adaptive` cannot
decide whether to derive TotaalWarmtepomp from sub-device columns or skip them.

The current workaround is to add absent-device columns as all-zeros in the mapper script
before `run_standard_pipeline` (see the supplier mappers in etdworkflow). This is fragile: it
must be repeated per mapper and is invisible to `add_calculated_columns_adaptive`.

### The proper fix

Add a `FysischModel` column to the data model (etdmap). This column assigns a named
physical system configuration to each household (e.g. `"warmtepomp_met_booster"`,
`"warmtepomp_zonder_booster"`). `add_calculated_columns_adaptive` (or a wrapper in
etdtransform) would then:

1. Look up the FysischModel for each household from the metadata / index.
2. Select a model-specific catalog subset — different derivation rules apply for each model.
3. Skip derivation attempts for columns that are structurally absent in that model.

### Design notes

- `FysischModel` is a per-household attribute set by whoever created the metadata (Grist
  or the mapper). It is NOT derived from the data; it is declared knowledge about the
  physical installation.
- The catalog (EquationRegistry) should support a `physical_model` tag per equation so
  `DatasetAdapter` can filter by model when building an execution plan.
- Until this is implemented, each mapper script is responsible for zeroing absent-device
  columns before `run_standard_pipeline`.

### Files to change

- `etdmap/data_model.py`: add `FysischModel` to the household-level metadata columns
- `etdtransform/catalog/registry.py` (EquationRegistry): add optional `physical_model`
  tag to equation entries
- `etdtransform/calculated_columns.py` (`add_calculated_columns_adaptive`): accept
  optional `physical_model` Series keyed by HuisIdBSV; pass to DatasetAdapter
- `etdtransform/catalog/query.py` (DatasetAdapter): filter catalog by physical_model
  when building feasibility report and execution plan
- Tests: fixture with two households in different models, assert model-specific columns
  are only derived for the correct model

### Related

- Short-term workaround: a supplier mapper adds absent columns as Float64 all-NA + zeros
- Affected sensors: `ElektriciteitsgebruikBooster`, `ElektriciteitsgebruikWTW`,
  `ElektriciteitsgebruikRadiator`, `ElektriciteitsgebruikBoilervat`
- See also: imputation broadcast bug in "Imputation pipeline" section below

---

## Imputation pipeline: avg broadcast reaches excluded households (structural fix required)

### Background

`prepare_diffs_for_impute` / `prepare_diffs_for_impute_ibis` compute a per-sensor average
as `groupby([ProjectIdBSV, ReadingDate]).mean()` across households that pass an outlier
filter (`included_ids`). This project-level average is then broadcast to ALL households
in the project via a join on `(ProjectIdBSV, ReadingDate)` -- including households that
were excluded from the avg computation because they had no non-`pd.NA` readings for that
sensor.

The result: a household with an absent sensor receives a non-`pd.NA` avg value, which
triggers imputation and fills genuinely-absent slots with a non-zero energy value. This
inflates annual sums and introduces spurious energy where none existed.

The short-term workaround (per-mapper pre-processing) is to fill absent-sensor columns
with `0.0` before the data enters the pipeline, so those households participate in the
avg with zero weight rather than being excluded. See the supplier mappers which call
`fill_down_infrequent_devices` for the known optional-device columns.

### The structural fix

After the `included_ids` set is computed per-sensor, persist it and use it to mask the
`_avg` column after the project-level join: households NOT in `included_ids` for a sensor
should receive `pd.NA` in that sensor's `_avg` column, not the project avg.

The relevant line in both paths is `calculate_average_diff` / `calculate_average_diff_ibis`
around the `included_ids` filter. The fix needs to carry `included_ids` forward to the
join step in `concatenate_avg_diff_columns` (or apply the mask there).

### Design consideration: this is a per-dataset policy choice

The correct behaviour is NOT always to mask absent-sensor households. There are two cases:

| Case | Sensor absent means | Correct avg behaviour |
|------|--------------------|-----------------------|
| Optional device not installed | True absence -- device never present | `pd.NA` avg for that household |
| Sensor gap / device temporarily off | Data missing, device exists | Project avg is the right fill |

The imputer cannot distinguish these cases from the avg table alone. Options:

1. **Configurable flag per column** (e.g. `infrequent_device_cols: list[str]` in the
   pipeline config): columns listed here get masked to `pd.NA` for absent households;
   all others continue to use the project avg. Callers (mapper scripts) declare which
   columns represent optional devices.

2. **Minimum participation threshold**: only broadcast the avg to a household if that
   household had at least N non-`pd.NA` readings for that sensor (N configurable, e.g.
   N=1 means "device was present at least once"). Households below the threshold get
   `pd.NA`. This is simpler but does not handle the "device installed but never switched
   on in this year" case.

3. **Keep the pre-processing workaround** (`fill_down_infrequent_devices` in mapper
   scripts) as the per-dataset decision point, and fix the pipeline to be neutral
   (do not impute households that have `pd.NA` avg). This keeps policy in the mapper
   and mechanism in the imputer.

Option 3 is the most conservative and consistent with ADR-003 (hard failures on missing
required inputs -- here: no avg means no imputation). It requires that the imputer
skips any row where `_avg = pd.NA` rather than treating it as a signal to fill.

### Files to change

- `etdtransform/impute.py`: `calculate_average_diff` (pandas path) and
  `calculate_average_diff_ibis` (ibis path) -- carry `included_ids` forward
- `etdtransform/impute.py`: `concatenate_avg_diff_columns` -- apply per-sensor
  household mask after building the result DataFrame
- Tests in `tests/test_total_imputation_workflow.py` -- add a test case where one
  household has all-`pd.NA` for a sensor and verify it does not receive an imputed
  non-zero value after the fix

### Related

- Discovered via regression comparison 2026-04-21 (see etdworkflow comparison_log.md)
- Short-term workaround: a supplier mapper calls `fill_down_infrequent_devices`
  for optional-device columns before `run_standard_pipeline`
- Affected sensors in comparison run: `ElektriciteitsgebruikBoilervat`,
  `ElektriciteitsgebruikBooster`, `WatergebruikRuimteverwarming`, `WatergebruikWarmTapwater`
