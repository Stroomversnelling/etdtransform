# etdtransform -- TODO

## Implement FysischModel (per-household physical system model)

Optional heat-pump sub-devices (Booster, WTW, Radiator, Boilervat) are present in
some households and absent in others, and there is currently no way to mark a
device as present or absent for a household.

Workaround (in the supplier mapping scripts): fill an absent device's column with
0.0, for two reasons: (1) the data-model calculations and rules that include
these devices keep working even though the device is absent; and (2) if the
column were left as pd.NA, the imputer would broadcast the project average for
that device into the household's absent intervals -- a spurious side effect that
the 0.0 fill avoids. The broadcast would over estimate the total energy in the
system by effectively adding the device that should be absent.

Action: implement FysischModel so each household is assigned its physical system
model and derivation includes only the devices that model has -- replacing the
0.0 workaround. The data model already carries this (it marks which variables and
which rules belong to which model), so no etdmap change is needed now; if one
turns out to be required, raise it in etdmap's TODO.

Dependency: the household supplier metadata must provide the physical-model value
per household (to be added in time) -- that is where FysischModel comes from.
