"""
CatalogBuilder — BFS expansion + Pareto pruning of all derivable equations.

Ported from analysis/interactive21 - sympy experiments.py, with additions:
  - build_catalog(): top-level entry point (registry → flat catalog DataFrame)
  - catalog_to_parquet() / catalog_from_parquet(): parquet serialization with MD5 cache invalidation
  - physical_model propagation: each catalog entry carries the intersection of the
    physical models of all equations that contributed to its derivation.

Scalability note:
  The BFS expansion is adequate for ~20–50 equations. For equation sets beyond ~50,
  consider replacing the expansion phase with SymPy Matrix.rref() over the coefficient
  matrix, which finds the full linear row space in O(N²V) without exponential blowup.
  BFS remains the only option for non-linear equations.
"""

import heapq
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sympy as sp

from .registry import EquationRegistry


# ---------------------------------------------------------------------------
# Step 1: direct derivations from a single equation (fast linear path)
# ---------------------------------------------------------------------------

def _direct_derivations_linear_fast(eq_pairs, symbols):
    """
    For each equation, isolate each symbol it contains using linear coefficient extraction.

    For each (equation, models) pair:
      - Form residual r = lhs - rhs
      - For each target in r.free_symbols ∩ symbols: isolate using r = a*target + rest
        → target = -rest / a
      - Store at most one (RHS, models) per unique RHS-variable-set (support) per target.
        If the same (target, varset) is reachable from two paths with different models,
        the models are unioned (the formula is valid for either model context).

    Parameters
    ----------
    eq_pairs : list of (sp.Eq, frozenset[str])
        Each element is an equation paired with the physical models it belongs to.
    symbols : list[sp.Symbol]

    Returns
    -------
    dict[target_symbol] → dict[rhs_varset_tuple(sorted symbols)] → (Eq(target, rhs), frozenset[str])
    """
    S = set(symbols)
    out = {s: {} for s in symbols}
    name_key = lambda x: x.name

    for eq, models in eq_pairs:
        r = eq.lhs - eq.rhs
        involved = r.free_symbols & S
        if not involved:
            continue

        poly = None
        poly_ok = False
        try:
            poly = sp.Poly(r, *symbols, domain="QQ")
            poly_ok = True
        except Exception:
            poly_ok = False

        for target in involved:
            if poly_ok:
                a = poly.coeff_monomial(target)
                if a == 0:
                    continue
                rest = r - a * target
            else:
                a = sp.diff(r, target)
                if a == 0:
                    continue
                rest = r - a * target

            rhs = sp.together(-rest / a)
            rhs = sp.cancel(rhs)

            fs = rhs.free_symbols & S
            if target in fs:
                raise ValueError(
                    f"Direct derivation contains target on RHS: {target} = {rhs}"
                )

            key = tuple(sorted(fs, key=name_key))
            existing = out[target].get(key)
            if existing is None:
                out[target][key] = (sp.Eq(target, rhs), models)
            else:
                # Same formula derivable from two model contexts — union models
                existing_eq, existing_models = existing
                out[target][key] = (existing_eq, existing_models | models)

    return {t: d for t, d in out.items() if d}


# ---------------------------------------------------------------------------
# Step 2: BFS expansion over residual relations
# ---------------------------------------------------------------------------

def _build_raw_catalog(eqs, eq_models, symbols, *, max_depth=6, max_pool=200_000):
    """
    Build a catalog of all derivable expressions using BFS substitution.

    Three steps:
      1. Direct derivations from original equations (with their physical models).
      2. Global BFS expansion: substitute known RHS derivations into residuals
         to discover new relations, tracked by variable-set novelty.
         New residuals inherit the intersection of contributing models.
         If the intersection is empty, the relation is discarded.
      3. Direct derivations again from the expanded relation pool.

    Parameters
    ----------
    eqs : list[sp.Eq]
    eq_models : list[frozenset[str]]
        Physical models per equation, parallel to eqs.
    symbols : list[sp.Symbol]

    Returns
    -------
    catalog : dict[symbol] → dict[rhs_varset_tuple] → (Eq(symbol, rhs), frozenset[str])
    pool    : dict[varset_tuple] → residual_expr
    """
    S = set(symbols)

    # Step 1
    eq_pairs = list(zip(eqs, eq_models))
    direct1 = _direct_derivations_linear_fast(eq_pairs, symbols)

    # Substitution rules: sym → list of (rhs_expr, rhs_varset_set, models)
    rules = {}
    for sym, per_varset in direct1.items():
        entries = []
        for eq, models in per_varset.values():
            rhs = eq.rhs
            entries.append((rhs, set(rhs.free_symbols & S), models))
        rules[sym] = entries

    # Step 2: BFS
    def canon(expr):
        return sp.simplify(expr)

    seen_sets = set()        # set[frozenset[Symbol]]
    pool = {}                # dict[tuple(sorted symbols)] → residual_expr
    pool_models = {}         # dict[tuple(sorted symbols)] → frozenset[str]
    pq_heap = []
    seq = 0

    def push_relation(r, V_set, depth, models):
        nonlocal seq
        V_frozen = frozenset(V_set)
        V_tuple = tuple(sorted(V_set, key=lambda x: x.name))
        if V_frozen in seen_sets:
            # Same varset reached again — union models (formula valid for more models)
            pool_models[V_tuple] = pool_models[V_tuple] | models
            return
        seen_sets.add(V_frozen)
        pool[V_tuple] = r
        pool_models[V_tuple] = models
        heapq.heappush(pq_heap, ((len(V_set), depth), depth, seq, r, V_set, models))
        seq += 1

    for eq, models in zip(eqs, eq_models):
        r0 = canon(eq.lhs - eq.rhs)
        V0 = set(r0.free_symbols & S)
        push_relation(r0, V0, depth=0, models=models)

    while pq_heap and len(pool) < max_pool:
        (_, depth, _, r, V_set, models) = heapq.heappop(pq_heap)

        if depth >= max_depth:
            continue

        for s in V_set:
            if s not in rules:
                continue
            for s_rhs, Vs_set, s_models in rules[s]:
                new_models = models & s_models
                if not new_models:
                    continue  # no physical model applies to this combination

                V2_pred = (V_set - {s}) | Vs_set
                if V2_pred == V_set:
                    continue
                if frozenset(V2_pred) in seen_sets:
                    continue

                r_tmp = r.subs(s, s_rhs)
                r2 = canon(r_tmp)
                if r2 == 0:
                    continue

                V2_actual = set(r2.free_symbols & S)
                push_relation(r2, V2_actual, depth + 1, new_models)

    # Step 3: direct derivations from expanded pool
    expanded_eq_pairs = [
        (sp.Eq(r, 0), pool_models[V_tuple])
        for V_tuple, r in pool.items()
    ]
    catalog = _direct_derivations_linear_fast(expanded_eq_pairs, symbols)

    return catalog, pool


# ---------------------------------------------------------------------------
# Pareto pruning: keep only minimal-support derivations per target
# ---------------------------------------------------------------------------

def _pareto_prune(catalog):
    """
    For each target, drop any derivation whose rhs_varset is a strict superset
    of another rhs_varset for the same target.

    Returns pruned catalog (same structure).
    """
    pruned = {}

    for lhs, per_varset in catalog.items():
        items = list(per_varset.items())
        sets = [(k, frozenset(k), len(k)) for (k, _) in items]
        sets.sort(key=lambda t: t[2])

        kept_keys = []
        kept_sets = []

        for key_tuple, key_set, _sz in sets:
            dominated = any(s_small < key_set for s_small in kept_sets)
            if not dominated:
                kept_keys.append(key_tuple)
                kept_sets.append(key_set)

        new_map = {k: per_varset[k] for k in kept_keys}
        if new_map:
            pruned[lhs] = new_map

    return pruned


# ---------------------------------------------------------------------------
# Flatten to DataFrame rows
# ---------------------------------------------------------------------------

def _flatten_catalog(catalog):
    """
    Flatten catalog dict into a list of row dicts suitable for a DataFrame.

    Returns
    -------
    list[dict] with keys: lhs, rhs_text, rhs_vars (list[str]), rhs_var_count,
                          physical_models (list[str])
    """
    rows = []
    for lhs_sym, per_varset in catalog.items():
        lhs_name = lhs_sym.name if hasattr(lhs_sym, "name") else str(lhs_sym)
        for rhs_varset_tuple, (eq, models) in per_varset.items():
            rhs_vars = [s.name for s in rhs_varset_tuple]
            rows.append({
                "lhs": lhs_name,
                "rhs_text": str(eq.rhs),
                "rhs_vars": rhs_vars,
                "rhs_var_count": len(rhs_vars),
                "physical_models": sorted(models),
            })
    return rows


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_catalog(registry: EquationRegistry, max_depth=6, max_pool=200_000) -> pd.DataFrame:
    """
    Build the full derivation catalog from an EquationRegistry.

    Runs BFS expansion + Pareto pruning on the registry's equations and returns
    a flat DataFrame with one row per (target variable, derivation) pair.

    Parameters
    ----------
    registry : EquationRegistry
    max_depth : int
        Maximum substitution depth in BFS expansion.
    max_pool : int
        Maximum number of unique residual relations to explore.

    Returns
    -------
    pd.DataFrame
        Columns: lhs (str), rhs_text (str), rhs_vars (list[str]),
                 rhs_var_count (int), physical_models (list[str])
    """
    symbols_list = list(registry.symbols().values())
    eq_models = registry.equation_models
    t0 = time.perf_counter()
    raw_catalog, pool = _build_raw_catalog(
        registry.equations, eq_models, symbols_list,
        max_depth=max_depth, max_pool=max_pool,
    )
    pruned = _pareto_prune(raw_catalog)
    rows = _flatten_catalog(pruned)
    elapsed = time.perf_counter() - t0
    print(
        f"[build_catalog] {len(registry.equations)} equations, "
        f"{len(pool)} pool relations, "
        f"{len(rows)} catalog entries after pruning ({elapsed:.2f}s)"
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Parquet serialization
# ---------------------------------------------------------------------------

def catalog_to_parquet(catalog_df: pd.DataFrame, path, rule_hash: str) -> None:
    """
    Write catalog DataFrame to parquet, embedding rule_hash in file metadata
    for cache invalidation.

    The rhs_vars and physical_models columns (list[str]) are stored natively
    as parquet list columns.

    Parameters
    ----------
    catalog_df : pd.DataFrame
        Output of build_catalog().
    path : str or Path
        Destination file path.
    rule_hash : str
        MD5 hex digest of the Rule CSV bytes. Stored in parquet metadata.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(catalog_df, preserve_index=False)
    # Merge our metadata into the existing schema metadata
    existing_meta = table.schema.metadata or {}
    new_meta = {**existing_meta, b"rule_hash": rule_hash.encode()}
    table = table.replace_schema_metadata(new_meta)
    pq.write_table(table, path)


def catalog_from_parquet(path, rule_hash: str):
    """
    Load catalog DataFrame from parquet, returning None if the stored hash
    does not match rule_hash (cache invalidation).

    Parameters
    ----------
    path : str or Path
    rule_hash : str
        MD5 hex digest of the current Rule CSV bytes.

    Returns
    -------
    pd.DataFrame or None
        None means the cache is stale and the catalog must be rebuilt.
    """
    path = Path(path)
    if not path.exists():
        return None
    table = pq.read_table(path)
    stored_hash = (table.schema.metadata or {}).get(b"rule_hash", b"").decode()
    if stored_hash != rule_hash:
        return None
    return table.to_pandas()
