"""Turn catalog derivations into the constraint table the processor reads.

The catalog already holds one row per (target, derivation) pair, solved. This
module selects the rows that can narrow a bound and adds the one thing the
catalog does not carry: the sign of each variable's partial derivative.

A variable whose own values are signed is no obstacle. The sign recorded is the
sign of the derivative, a property of the expression: in
``Netuitwisseling = ElektriciteitNetgebruikDiff - ElektriciteitTerugleveringDiff``
the two derivatives are +1 and -1 however negative the values get.

Where a derivative is not constant -- a product of two unknowns -- its sign
comes from the declared bounds of the variables it still names. That stays here
rather than in the processor because scaling a bound to a coarser cadence never
moves it across zero, so the answer does not depend on cadence.

Expression strings are stored, never callables. The processor lambdifies them
once it knows the cadence.

The processor is the other half, further down this module: it runs in the
imputation stage, is handed the grid cadence, and turns these expressions into
numbers. It writes nothing and caches nothing.
"""

import logging
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from etdtransform.catalog import classify_linear_nonlinear
from etdmap import ids

CONSTRAINT_COLUMNS = (
    "lhs",
    "rhs_text",
    "variable",
    "sign",
    "derivative_text",
    "rhs_var_count",
    "physical_models",
)

MOMENTAAN = "momentaan"


def compile_constraints(
    catalog: pd.DataFrame,
    thresholds: dict | None = None,
) -> pd.DataFrame:
    """One row per (target, derivation, variable) with the derivative's sign.

    ``catalog`` is a loaded ``catalog.parquet``, as built by the catalog
    package. ``thresholds`` defaults to etdmap's declared bounds.

    Raises ``ValueError`` if a kept derivation names a momentaan column.
    """
    if thresholds is None:
        from etdmap.data_model import thresholds_dict

        thresholds = thresholds_dict

    rules = [
        {"lhs": row.lhs, "rhs": row.rhs_text, "_row": row.Index}
        for row in catalog.itertuples()
        if _is_bounded(thresholds.get(row.lhs))
    ]
    linear, nonlinear = classify_linear_nonlinear(rules) if rules else ([], [])

    _reject_momentaan(linear, thresholds)

    symbols = _signed_symbols(thresholds)
    records = []
    for rule in linear:
        row = catalog.loc[rule["_row"]]
        for variable, sign, derivative in _derivatives(
            rule["rhs"], row["rhs_vars"], symbols
        ):
            records.append(
                {
                    "lhs": rule["lhs"],
                    "rhs_text": rule["rhs"],
                    "variable": variable,
                    "sign": sign,
                    "derivative_text": derivative,
                    "rhs_var_count": row["rhs_var_count"],
                    "physical_models": row["physical_models"],
                }
            )

    table = _as_table(records)
    logging.info(
        "Constraints: %d of %d catalog rows bounded, %d linear, %d dropped as "
        "nonlinear, %d (target, derivation, variable) rows emitted, %d of those "
        "with no fixed sign",
        len(rules),
        len(catalog),
        len(linear),
        len(nonlinear),
        len(table),
        int(table["sign"].isna().sum()),
    )
    return table


def apparaat_set_key(variables) -> str:
    """Canonical key for an absent-device set.

    Sorted variable names joined with ``|``; the empty string is the empty
    set. One definition, shared by the sync (which writes the set-keyed
    constraint table) and the run-time selection (which reads it), so the
    two can never serialize the same set differently.
    """
    return "|".join(sorted(str(v) for v in variables))


def specialize_catalog_for_absent(catalog: pd.DataFrame, absent_vars) -> pd.DataFrame:
    """The catalog with a declared set's absent devices substituted at 0.

    This is the derived-model machinery of the missing-meter design: absence is
    resolved ONCE at compile time, zero IN the expression, never per row. For
    each absent device variable the
    substitution covers the variable itself and its data-model-linked pair
    (the interval form of a cumulative and vice versa), because catalog
    expressions name the Diff forms while declarations carry the cumulative
    device names.

    Per catalog row:

    - a row whose ``lhs`` is itself absent is DROPPED -- a nonexistent meter
      is never a derivation target;
    - a row naming no absent variable is returned untouched (no sympy, so it
      stays byte-identical to the input row);
    - otherwise the absent symbols are substituted at 0 and the expression
      simplified; ``rhs_vars`` keeps the surviving names in their original
      order and ``rhs_var_count`` follows;
    - a row whose expression reduces to a constant is DROPPED: pinning a
      target from metadata alone would let absence specify a value, and
      absence may only widen (the doctrine); the declared bounds remain the
      only thing entitled to bound such a target.

    The constraint compiler applies this at sync time, which is why
    constraints.parquet carries one table per apparaat-set. The catalog
    carries no such column, so a run-time caller that needs a household's
    formulas calls this -- see :func:`rule_sets_by_household`.
    """
    import sympy as sp
    from sympy.parsing.sympy_parser import parse_expr

    try:
        from etdmap.data_model import cumulative_column, interval_column
    except Exception:  # pragma: no cover - etdmap always present in practice
        def interval_column(name):
            return None

        def cumulative_column(name):
            return None

    absent = set()
    for name in absent_vars:
        absent.add(str(name))
        for linked in (interval_column(str(name)), cumulative_column(str(name))):
            if linked:
                absent.add(linked)

    if not absent or catalog.empty:
        return catalog.copy()

    zeros = {sp.Symbol(name): 0 for name in absent}
    records = []
    for idx in catalog.index:
        row = catalog.loc[idx]
        if str(row["lhs"]) in absent:
            continue
        rhs_vars = [str(v) for v in row["rhs_vars"]]
        if not (set(rhs_vars) & absent):
            records.append(row.to_dict())
            continue
        expression = parse_expr(str(row["rhs_text"])).subs(zeros)
        expression = sp.simplify(expression)
        if not expression.free_symbols:
            continue
        surviving = {symbol.name for symbol in expression.free_symbols}
        updated = row.to_dict()
        updated["rhs_text"] = str(expression)
        updated["rhs_vars"] = [v for v in rhs_vars if v in surviving]
        if "rhs_var_count" in updated:
            updated["rhs_var_count"] = len(updated["rhs_vars"])
        records.append(updated)

    out = pd.DataFrame(records, columns=list(catalog.columns))
    if "rhs_var_count" in out.columns:
        out["rhs_var_count"] = out["rhs_var_count"].astype("Int64")
    return out.reset_index(drop=True)


@dataclass
class RuleSet:
    """The formulas and the bounds that apply to one apparaat-set."""

    apparaat_set: str
    households: list
    formulas: pd.DataFrame
    package: "ConstraintPackage"


def rule_sets_by_household_batch(household_batch_ids, catalog, cadence_minutes,
                                 metadata_path=None) -> dict:
    """What each of *household_batch_ids* may derive, by its apparaat-set.

    A row may only be derived with a formula whose variables are actually
    present. Where no heat pump is declared there is no heat-pump term to
    solve for and no heat-pump reading to solve with, so a formula naming one
    must not be a candidate -- not rejected later by a null check, absent from
    the formula set.

    ``ApparaatOntbrekend`` declares that per household batch, which is why
    this resolves per household batch: a later household batch may hold the
    same household renovated, so its devices -- and therefore the formulas it
    may use -- differ from the earlier one's. Everything sharing a declaration
    shares everything that follows from it, so the work is done once per
    distinct set rather than once per member: the catalog is specialized and
    the constraint package is built per set, and every member gets the same
    two objects.

    Returns ``{apparaat_set_key: RuleSet}``. A household batch the metadata
    does not mention raises rather than falling back to the full model: the
    full model has MORE rules and so narrows HARDER, and absence may only
    widen.

    Parameters
    ----------
    household_batch_ids : iterable of int
    catalog : pd.DataFrame
        The derivation catalog, unspecialized.
    cadence_minutes : int
        The grid, which is what a calculated cell covers.
    metadata_path : str or None
        Passed through to :func:`apparaat_set_by_household_batch`.
    """
    declared = apparaat_set_by_household_batch(metadata_path)
    grouped: dict = {}
    unknown = []
    for batch_id in household_batch_ids:
        batch_id = int(batch_id)
        if batch_id not in declared:
            unknown.append(batch_id)
            continue
        grouped.setdefault(declared[batch_id], []).append(batch_id)

    if unknown:
        raise ValueError(
            f"{len(unknown)} household batch(es) are not in the BSV "
            f"metadata, so their ApparaatOntbrekend declarations are unknown "
            f"and the formulas they may use cannot be resolved (first 10: "
            f"{sorted(unknown)[:10]}). Rebuild the metadata rather than "
            f"deriving them against the full model."
        )

    constraints = packaged_constraints()
    return {
        key: RuleSet(
            apparaat_set=key,
            households=sorted(members),
            formulas=specialize_catalog_for_absent(
                catalog, key.split("|") if key else []
            ),
            package=build_constraint_package(
                constraints,
                cadence_minutes=cadence_minutes,
                apparaat_set=key,
            ),
        )
        for key, members in grouped.items()
    }


_APPARAAT_SETS_CACHE: dict = {}


def apparaat_set_by_household_batch(metadata_path=None) -> dict:
    """``ids.HOUSEHOLD_BATCH -> canonical apparaat-set key`` from the metadata.

    ``ApparaatOntbrekend`` is declared per household batch: the Grist table
    behind the BSV metadata holds one row per ``ids.HOUSEHOLD_BATCH``, because
    a later household batch may hold the same household renovated and so
    carrying different devices. The declarations flow through
    ``build_bsv_metadata`` into the BSV metadata CSV; this reads them once per
    path and keys the result on the household batch, which is the grain they
    are declared at.

    Mirrors how ``physical_model_by_project`` resolves the model: the set
    decides which compiled rules apply, so a metadata file missing either the
    declaration column or the household batch column raises rather than
    defaulting -- the full model has MORE rules and so narrows HARDER, and an
    absent declaration may only widen.
    """
    if metadata_path is None:
        import etdmap

        metadata_path = getattr(etdmap.options, "bsv_metadata_file", None)
        if not metadata_path:
            raise ValueError(
                "bsv_metadata_file is not configured; the constraint "
                "selection needs it to resolve each household's "
                "ApparaatOntbrekend set."
            )

    cache_key = str(metadata_path)
    if cache_key in _APPARAAT_SETS_CACHE:
        return _APPARAAT_SETS_CACHE[cache_key]

    from etdmap.data_model import parse_device_declaration

    metadata = pd.read_csv(metadata_path, dtype_backend="numpy_nullable")
    if "ApparaatOntbrekend" not in metadata.columns:
        raise ValueError(
            f"{metadata_path} has no ApparaatOntbrekend column -- the BSV "
            f"metadata predates the device declarations. Run "
            f"sync_data_model.py (or build_bsv_metadata.py) to rebuild it."
        )
    if ids.HOUSEHOLD_BATCH not in metadata.columns:
        raise ValueError(
            f"{metadata_path} has no {ids.HOUSEHOLD_BATCH} column -- the BSV "
            f"metadata predates keying the declarations on the household "
            f"batch. ApparaatOntbrekend is declared per household batch, and "
            f"resolving it per household would give a renovated household "
            f"batch its earlier devices. Run sync_data_model.py (or "
            f"build_bsv_metadata.py) to rebuild it."
        )

    resolved: dict = {}
    for _, row in metadata.iterrows():
        batch_id = row.get(ids.HOUSEHOLD_BATCH)
        if pd.isna(batch_id):
            continue
        resolved[int(batch_id)] = apparaat_set_key(
            parse_device_declaration(row.get("ApparaatOntbrekend"))
        )
    _APPARAAT_SETS_CACHE[cache_key] = resolved
    return resolved


def _is_bounded(declared: dict | None) -> bool:
    if declared is None:
        return False
    return pd.notna(declared.get("Min")) or pd.notna(declared.get("Max"))


def _reject_momentaan(rules: list[dict], thresholds: dict) -> None:
    offenders = {}
    for rule in rules:
        for name in _symbol_names(rule["rhs"]) | {rule["lhs"]}:
            declared = thresholds.get(name)
            if declared and declared.get("ThresholdType") == MOMENTAAN:
                offenders.setdefault(name, set()).add(rule["lhs"])
    if offenders:
        detail = "; ".join(
            f"{name} (reached from {', '.join(sorted(targets))})"
            for name, targets in sorted(offenders.items())
        )
        raise ValueError(
            "A momentaan bound cannot constrain an interval value. "
            f"Remove these derivations or redeclare the bound: {detail}"
        )


def _symbol_names(rhs_text: str) -> set[str]:
    from sympy.parsing.sympy_parser import parse_expr

    return {symbol.name for symbol in parse_expr(rhs_text).free_symbols}


def _signed_symbols(thresholds: dict) -> dict:
    """A sympy symbol per declared variable, carrying what its bounds imply.

    Sympy then answers the sign question itself, including for the products
    ``classify_linear_nonlinear`` admits.
    """
    import sympy as sp

    symbols = {}
    for name, declared in thresholds.items():
        lo = declared.get("Min")
        hi = declared.get("Max")
        if pd.notna(lo) and lo > 0:
            assumption = {"positive": True}
        elif pd.notna(lo) and lo >= 0:
            assumption = {"nonnegative": True}
        elif pd.notna(hi) and hi < 0:
            assumption = {"negative": True}
        elif pd.notna(hi) and hi <= 0:
            assumption = {"nonpositive": True}
        else:
            assumption = {}
        symbols[name] = sp.Symbol(name, **assumption)
    return symbols


def _derivatives(
    rhs_text: str,
    variables,
    symbols: dict,
) -> list[tuple[str, int | None, str]]:
    """Each variable's coefficient, and the sign of that coefficient.

    +1 means the target does not decrease as the variable rises, -1 that it does
    not rise, 0 that the variable cancels out. The sign is absent only when the
    declared bounds leave the coefficient straddling zero, and the variable then
    cannot narrow a bound through this derivation.
    """
    import sympy as sp
    from sympy.parsing.sympy_parser import parse_expr

    expression = parse_expr(rhs_text, local_dict=symbols)
    derivatives = []
    for name in variables:
        symbol = symbols.get(name) or sp.Symbol(name)
        derivative = sp.diff(expression, symbol)
        derivatives.append((name, _sign_of(derivative), str(derivative)))
    return derivatives


def _sign_of(derivative) -> int | None:
    if derivative.is_zero:
        return 0
    if derivative.is_nonnegative:
        return 1
    if derivative.is_nonpositive:
        return -1
    return None


def _as_table(records: list[dict]) -> pd.DataFrame:
    table = pd.DataFrame(records, columns=list(CONSTRAINT_COLUMNS))
    return table.astype(
        {
            "lhs": "string",
            "rhs_text": "string",
            "variable": "string",
            "sign": "Int64",
            "derivative_text": "string",
            "rhs_var_count": "Int64",
        }
    )


# ---------------------------------------------------------------------------
# The processor. Runs in the imputation stage, where the grid cadence is known.
# ---------------------------------------------------------------------------

INF = float("inf")
MAX_PRESOLVE_ROUNDS = 20


@dataclass(frozen=True)
class Rule:
    """One derivation, lambdified at a cadence.

    ``signs`` maps each variable to the direction it moves the target in: +1
    means the target does not fall as the variable rises, -1 that it does not
    rise, 0 that the variable cancels out. A derivation with an unresolved sign
    never reaches here -- it is dropped, because an unknown direction cannot
    pick an endpoint.
    """

    target: str
    expression: str
    variables: tuple
    signs: dict
    evaluate: Callable


@dataclass(frozen=True)
class ConstraintPackage:
    """What the fill is handed. In memory, per run, never written or cached."""

    cadence_minutes: int
    bounds: dict
    clusters: tuple
    rules_by_target: dict
    constrained_columns: tuple


def build_constraint_package(
    constraints: pd.DataFrame,
    cadence_minutes: int,
    thresholds: dict | None = None,
    physical_model: str | None = None,
    apparaat_set: str = "",
) -> ConstraintPackage:
    """Bounds, clusters and callables for one grid cadence.

    ``constraints`` is the compiler's table. An empty one is the off switch: the
    result carries the declared bounds and narrows nothing.

    ``physical_model`` selects the rule subset the model permits, by the same
    membership the catalog's own resolution applies: a row survives when the
    model appears in its ``physical_models`` list. ``None`` keeps every rule,
    which is only right for a caller that has already scoped its rows.

    ``apparaat_set`` selects ONE specialized table when the frame carries the
    ``apparaat_set`` column, defaulting to ``""`` -- the none-absent base
    table. Sets are never unioned: each specialized table holds the rules with
    that set's device terms substituted to zero, so mixing tables applies the
    tightest specialized narrowing to households that have all their devices
    (measured on the 2026-10-01 artifact: ElektriciteitsgebruikTotaalWarmtepomp
    0..1.6 collapsing to 0..0.96). A frame already scoped to a single set is
    accepted as it is, so the fill's own pre-filter keeps working.

    Raises ``ValueError`` when a rule and the thresholds contradict each other,
    which is a configuration error rather than a data condition.
    """
    if thresholds is None:
        from etdmap.data_model import thresholds_dict

        thresholds = thresholds_dict

    if "apparaat_set" in constraints.columns and not constraints.empty:
        present = set(constraints["apparaat_set"].astype(str))
        if len(present) > 1:
            constraints = constraints[
                constraints["apparaat_set"].astype(str) == str(apparaat_set)
            ]
            if constraints.empty:
                raise ValueError(
                    f"No compiled constraint table for apparaat-set "
                    f"{str(apparaat_set)!r}; the artifact holds "
                    f"{sorted(present)!r}. Re-run the data model sync, which "
                    f"writes one table per declared set."
                )

    if physical_model is not None and not constraints.empty:
        constraints = constraints[
            constraints["physical_models"].apply(
                lambda models: physical_model in (
                    list(models) if models is not None else []
                )
            )
        ]

    bounds = _bounds_at_cadence(thresholds, cadence_minutes)
    rules = _lambdify_rules(constraints)
    _narrow_to_fixpoint(rules, bounds, cadence_minutes)

    by_target: dict = {}
    for rule in rules:
        by_target.setdefault(rule.target, []).append(rule)

    return ConstraintPackage(
        cadence_minutes=cadence_minutes,
        bounds=bounds,
        clusters=_clusters(rules),
        rules_by_target={t: tuple(rs) for t, rs in by_target.items()},
        constrained_columns=tuple(
            sorted({c for r in rules for c in (r.target, *r.variables)})
        ),
    )


def _bounds_at_cadence(thresholds: dict, cadence_minutes: int) -> dict:
    """Every declared bound, scaled to the grid.

    pd.NA means no bound, so it becomes an infinity. Interval arithmetic needs a
    number, and an absent bound is not an unknown one.
    """
    from etdmap.data_model import scale_bounds

    bounds = {}
    for name, declared in thresholds.items():
        lo, hi = scale_bounds(declared, cadence_minutes)
        bounds[name] = (
            -INF if pd.isna(lo) else float(lo),
            INF if pd.isna(hi) else float(hi),
        )
    return bounds


def _lambdify_rules(constraints: pd.DataFrame) -> list:
    """One Rule per (target, derivation), dropping any with an unresolved sign."""
    if constraints.empty:
        return []

    import sympy as sp
    from sympy.parsing.sympy_parser import parse_expr

    rules = []
    for (target, expression), group in constraints.groupby(
        ["lhs", "rhs_text"], sort=True
    ):
        signs = dict(zip(group["variable"], group["sign"]))
        if any(pd.isna(sign) for sign in signs.values()):
            continue

        variables = tuple(sorted(signs))
        symbols = [sp.Symbol(name) for name in variables]
        rules.append(
            Rule(
                target=str(target),
                expression=str(expression),
                variables=variables,
                signs={name: int(sign) for name, sign in signs.items()},
                evaluate=sp.lambdify(symbols, parse_expr(str(expression)), "numpy"),
            )
        )
    return rules


def _implied_interval(rule: Rule, bounds: dict) -> tuple:
    """The interval the rule's expression can take, given its variables' bounds.

    Each variable goes to the endpoint its sign points at: for the low end of
    the result, the endpoint that makes the target smallest. That is what a sign
    is for, and it holds for a product of two non-negative unknowns as well as
    for a sum, where reading coefficients off the expression would not.
    """
    return (
        _endpoint_value(rule, bounds, low=True),
        _endpoint_value(rule, bounds, low=False),
    )


def _endpoint_value(rule: Rule, bounds: dict, low: bool) -> float:
    point = {}
    for name in rule.variables:
        sign = rule.signs[name]
        if sign == 0:
            # It cancels out of the expression, so any finite value serves.
            point[name] = 0.0
            continue
        lo, hi = bounds.get(name, (-INF, INF))
        value = lo if (sign > 0) == low else hi
        if value in (-INF, INF):
            # Driving a variable the target follows to an infinity drives the
            # target to the matching one.
            return -INF if low else INF
        point[name] = value

    return float(rule.evaluate(**point))


def _narrow_to_fixpoint(rules: list, bounds: dict, cadence_minutes: int) -> None:
    """Intersect each target's bound with what its rules imply, until nothing
    moves. A bound tightened by one rule tightens others through further rules,
    so one pass is not enough.
    """
    for _ in range(MAX_PRESOLVE_ROUNDS):
        changed = False
        for rule in rules:
            implied = _implied_interval(rule, bounds)
            declared = bounds.get(rule.target, (-INF, INF))
            new_lo = max(declared[0], implied[0])
            new_hi = min(declared[1], implied[1])
            if new_lo > new_hi:
                where = (
                    "as declared" if cadence_minutes is None
                    else f"at {cadence_minutes} minutes"
                )
                raise ValueError(
                    f"The rules and the thresholds contradict each other on "
                    f"'{rule.target}' {where}: declared "
                    f"[{declared[0]:g}, {declared[1]:g}] but "
                    f"'{rule.target} = {rule.expression}' implies "
                    f"[{implied[0]:g}, {implied[1]:g}], which do not overlap. "
                    f"Fix the thresholds or the rule in Grist."
                )
            if new_lo > declared[0] + 1e-12 or new_hi < declared[1] - 1e-12:
                bounds[rule.target] = (new_lo, new_hi)
                changed = True
        if not changed:
            return
    logging.warning(
        "Constraint presolve did not settle in %d rounds at %d minutes; using "
        "the bounds reached so far.",
        MAX_PRESOLVE_ROUNDS,
        cadence_minutes,
    )


def _clusters(rules: list) -> tuple:
    """Columns grouped so two share a group whenever some rule names both. The
    fill resolves a group together."""
    parent: dict = {}

    def find(name):
        parent.setdefault(name, name)
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left, right):
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_left] = root_right

    for rule in rules:
        for name in rule.variables:
            union(rule.target, name)

    groups: dict = {}
    for name in parent:
        groups.setdefault(find(name), set()).add(name)
    return tuple(
        sorted(
            (frozenset(members) for members in groups.values()),
            key=lambda members: (-len(members), sorted(members)[0]),
        )
    )


def narrow_bounds_by_row(frame, target, package, imputed=None):
    """The bound each row of *frame* narrows *target* to.

    Starts from the package's presolved bound for *target* and intersects, per
    row, what every rule solving for *target* implies once the row's known
    values are substituted. A variable is known where ``usable_readings`` says
    its cell is a measurement; anywhere else it contributes the endpoint of its
    own presolved interval that its sign points at, exactly as the presolve
    does for whole columns. A variable that is not a column of *frame* is never
    known, which is how a constraint naming something this stage cannot impute
    still applies: a delivered value is respected as it is, and an absent one
    speaks through its declared range.

    *imputed* maps column name to a boolean Series marking cells this run
    wrote. A row whose bound was tightened by a rule that substituted such a
    cell is marked in the third result, so a caller can tell a bound fed by
    measurements alone from one fed by the fill's own output.

    Returns ``(lo, hi, from_imputed)``: two Float64 Series in which pd.NA
    means unbounded on that side, and a boolean Series.
    """
    # The rules are numpy-lambdified, so this function works in numpy arrays
    # between the explicit conversions at its edges (the ADR-005 exception).
    import numpy as np

    from etdtransform.bounds import usable_readings

    n = len(frame)
    start_lo, start_hi = package.bounds.get(target, (-INF, INF))
    lo = np.full(n, start_lo, dtype=float)
    hi = np.full(n, start_hi, dtype=float)
    from_imputed = np.zeros(n, dtype=bool)

    # A target's rules share variables, so the per-variable work -- the known
    # mask, the value array, and the sign-substituted inputs -- is done once
    # per call and reused across rules. Without this the narrowing rebuilds
    # the same multi-million-row arrays once per rule, which is what made it
    # the dominant cost of a chunk.
    _variables: dict = {}
    _substituted: dict = {}

    def _variable(name):
        if name not in _variables:
            known = usable_readings(frame, name).to_numpy(dtype=bool)
            if name in frame.columns:
                # na_value backs only unknown cells, which np.where discards.
                values = frame[name].to_numpy(dtype=float, na_value=0.0)
            else:
                values = np.zeros(n, dtype=float)
            if imputed is not None and name in imputed:
                imputed_known = known & imputed[name].to_numpy(dtype=bool)
            else:
                imputed_known = None
            _variables[name] = (known, values, imputed_known)
        return _variables[name]

    def _inputs(name, sign):
        key = (name, sign)
        if key not in _substituted:
            known, values, _ = _variable(name)
            if sign == 0:
                # It cancels out of the expression, so any finite value serves.
                end_low = end_high = 0.0
            else:
                bound_lo, bound_hi = package.bounds.get(name, (-INF, INF))
                end_low = bound_lo if sign > 0 else bound_hi
                end_high = bound_hi if sign > 0 else bound_lo
            _substituted[key] = (
                np.where(known, values, end_low),
                np.where(known, values, end_high),
            )
        return _substituted[key]

    for rule in package.rules_by_target.get(target, ()):
        at_low: dict = {}
        at_high: dict = {}
        rule_used_imputed = np.zeros(n, dtype=bool)
        for name in rule.variables:
            at_low[name], at_high[name] = _inputs(name, rule.signs[name])
            imputed_known = _variable(name)[2]
            if imputed_known is not None:
                rule_used_imputed |= imputed_known

        # Infinite endpoints can meet as 0*inf or inf-inf, which is nan. An
        # undecidable end narrows nothing, so the nan is expected, not invalid.
        with np.errstate(invalid="ignore"):
            rhs_low = np.asarray(rule.evaluate(**at_low), dtype=float)
            rhs_high = np.asarray(rule.evaluate(**at_high), dtype=float)
        rhs_low = np.where(np.isnan(rhs_low), -INF, rhs_low)
        rhs_high = np.where(np.isnan(rhs_high), INF, rhs_high)

        tightened = (rhs_low > lo) | (rhs_high < hi)
        from_imputed |= rule_used_imputed & tightened
        lo = np.maximum(lo, rhs_low)
        hi = np.minimum(hi, rhs_high)

    # Back to nullable at the edge: an infinity is the absence of a bound.
    lo_out = pd.Series(
        pd.array(np.where(np.isinf(lo), np.nan, lo), dtype="Float64"),
        index=frame.index,
    )
    hi_out = pd.Series(
        pd.array(np.where(np.isinf(hi), np.nan, hi), dtype="Float64"),
        index=frame.index,
    )
    imputed_out = pd.Series(from_imputed, index=frame.index, dtype="boolean")
    return lo_out, hi_out, imputed_out


def load_constraints(path) -> pd.DataFrame:
    """Read the compiler's artifact. Kept beside the catalog it derives from."""
    return pd.read_parquet(path, dtype_backend="numpy_nullable")


def packaged_constraints() -> pd.DataFrame:
    """The synced constraint table bundled with etdmap.

    The Grist sync writes constraints.parquet beside the catalog it derives
    from; this is what a pipeline run narrows with unless a caller explicitly
    runs the comparison mode without constraints. An absent file raises: the
    rules are part of the model, so a tree without them is unsynced, never a
    tree that silently fills unconstrained.
    """
    from importlib.resources import files

    artifact = files("etdmap.data").joinpath("constraints.parquet")
    if not artifact.is_file():
        raise FileNotFoundError(
            "etdmap's constraints.parquet is missing. Run the data model "
            "sync, which compiles it from the catalog, or pass "
            "use_constraints=False explicitly for a comparison run."
        )
    return load_constraints(artifact)
