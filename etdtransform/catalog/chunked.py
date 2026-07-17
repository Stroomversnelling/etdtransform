"""etdtransform.catalog.chunked -- chunked, cacheable, parallel
equation-catalog builder, partitioned per physical model.

Public entry points: build_chunked, build_serial_reference,
verify_against_serial. The design used here balances performance
and catalog converage by minimizing the size of a system 
of non-linear equations to solve through partitioning and expansion
substituting linear equations into the non-linear derivations.

Partitioning
------------
A physical model (Universeel, Hybride, All-Electric, ...) is a coherent
system: rules tagged with that model describe how its quantities relate.
Each model is built independently:

  for model M:
    linear chunk     -- BFS+Pareto over linear rules that include M in
                        their physical_models list.
    non-linear (NL) chunk per NL rule R in M
                     -- sp.solve direct isolations + expansion pass
                        substituting linear-catalog derivations into R
                        and re-solving the result.

Each row produced in model M is tagged physical_models=[M]. The final
catalog is the union of all per-model rows. The DatasetAdapter query
filters by physical_model at calculation time and lands on the rows
that apply.

Caching
-------
Every chunk is hashed and stored in cache_dir/. Hashes are computed on
*canonical* rule content (sympy-simplified RHS, sorted models, sorted
rules) so cosmetic differences in the exported rule text do not invalidate
the cache. The on-sync rebuild path is: hash, check cache, build only
the misses.

Parallelism
-----------
Two phases on the same ProcessPoolExecutor:
  Phase 1: linear cores across all models (independent).
  Phase 2: nl-chunks across all (model, NL-rule) pairs (each one needs
           its model's linear catalog, which is ready after Phase 1).

This module is pure etdtransform -- it only depends on sympy, pandas,
and other etdtransform.catalog primitives (EquationRegistry,
build_catalog). It does NOT import etdmap, so the sync tooling can call
build_chunked() during catalog rebuild even while etdmap is mid-flight broken.

Future improvement: single-build with model propagation
-------------------------------------------------------
The current implementation partitions by physical model FIRST, then
builds independently per partition. Rules tagged with multiple models
(e.g. one rule valid in Universeel + Hybride + All-Electric) are
re-derived once per model partition. The compose step then dedups
identical derivations across partitions and unions their model sets,
so the final catalog is not bloated -- but the *work* is duplicated
during the build.

A smarter implementation would carry physical_models as a tracked
attribute through a single global BFS+Pareto pass (the existing
etdtransform builder already does this for the linear case via
"intersection of contributing equations' models"). Each derivation
would be computed once and tagged with the union of model contexts in
which it is valid. Non-linear expansion would intersect rule.models
with linear-catalog-row.models per substitution and skip empty
intersections.

This would save build time when many rules apply to many models, but
it requires reaching into etdtransform's BFS and reworking the
non-linear pass to thread model sets through every operation.
Deferred until model count or rule count makes the per-model
duplication actually costly.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Callable

import pandas as pd


# Default model when a rule has no physical_models field. The data
# model's rule table requires every rule to declare its models, so
# this is a safety fallback for malformed input.
_DEFAULT_MODEL = "Universeel"


# ---------------------------------------------------------------------------
# Canonical hashing
# ---------------------------------------------------------------------------

def _canonical_rule_payload(rule: dict) -> dict:
    """Hash-stable representation of one rule (used in chunk hashes)."""
    import sympy as sp
    from sympy.parsing.sympy_parser import parse_expr

    rhs_canon = str(sp.simplify(parse_expr(rule["rhs"])))
    physical = sorted(rule.get("physical_models", []) or [])
    return {
        "lhs": str(rule["lhs"]).strip(),
        "rhs": rhs_canon,
        "physical_models": physical,
    }


def _stable_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def canonical_linear_hash(linear_rules: list[dict], model: str) -> str:
    """Hash the canonical content of (linear rules in `model`, model name)."""
    payload = {
        "model": model,
        "rules": sorted(
            (_canonical_rule_payload(r) for r in linear_rules),
            key=lambda r: (r["lhs"], r["rhs"]),
        ),
    }
    return hashlib.md5(_stable_json(payload).encode()).hexdigest()


def canonical_nl_chunk_hash(nl_rule: dict, linear_hash: str, model: str) -> str:
    """Hash a non-linear chunk against (rule, linear catalog hash, model)."""
    payload = {
        "model": model,
        "rule": _canonical_rule_payload(nl_rule),
        "linear_hash": linear_hash,
    }
    return hashlib.md5(_stable_json(payload).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Classification + model discovery
# ---------------------------------------------------------------------------

def _models_of(rule: dict) -> list[str]:
    ms = rule.get("physical_models") or []
    if not ms:
        return [_DEFAULT_MODEL]
    return list(ms)


def discover_models(rules: list[dict]) -> list[str]:
    """Sorted list of every physical model named by any rule."""
    s = set()
    for r in rules:
        s.update(_models_of(r))
    return sorted(s)


def rules_for_model(rules: list[dict], model: str) -> list[dict]:
    return [r for r in rules if model in _models_of(r)]


def classify_linear_nonlinear(rules: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split rules into (linear, nonlinear) for the *current set's* symbol space."""
    import sympy as sp
    from sympy.parsing.sympy_parser import parse_expr

    all_var_names = set()
    for r in rules:
        rhs = parse_expr(r["rhs"])
        all_var_names.add(r["lhs"])
        all_var_names.update(s.name for s in rhs.free_symbols)
    all_symbols = [sp.Symbol(n) for n in sorted(all_var_names)]

    linear, nonlinear = [], []
    for rule in rules:
        if _is_linear(rule, all_symbols):
            linear.append(rule)
        else:
            nonlinear.append(rule)
    return linear, nonlinear


def _is_linear(rule: dict, all_symbols) -> bool:
    import sympy as sp
    from sympy.parsing.sympy_parser import parse_expr

    rhs_expr = parse_expr(rule["rhs"])
    lhs_sym = sp.Symbol(rule["lhs"])
    eq = sp.Eq(lhs_sym, rhs_expr)
    r = eq.lhs - eq.rhs
    S = set(all_symbols)
    involved = r.free_symbols & S
    try:
        poly = sp.Poly(r, *all_symbols, domain="QQ")
        poly_ok = True
    except Exception:
        poly_ok = False
        poly = None
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
        rhs = sp.cancel(sp.together(-rest / a))
        if target in (rhs.free_symbols & S):
            return False
    return True


# ---------------------------------------------------------------------------
# Chunk builders (operate on a single model's rules)
# ---------------------------------------------------------------------------

def _all_symbols(rules: list[dict]):
    import sympy as sp
    from sympy.parsing.sympy_parser import parse_expr

    names = set()
    for r in rules:
        rhs = parse_expr(r["rhs"])
        names.add(r["lhs"])
        names.update(s.name for s in rhs.free_symbols)
    return [sp.Symbol(n) for n in sorted(names)]


def build_linear_catalog(linear_rules: list[dict], model: str) -> pd.DataFrame:
    """Build the linear catalog for one model using etdtransform's existing
    builder. Tags every row with physical_models=[model].
    """
    from etdtransform.catalog import EquationRegistry, build_catalog

    if not linear_rules:
        return pd.DataFrame(columns=["lhs", "rhs_text", "rhs_vars", "rhs_var_count", "physical_models"])
    reg = EquationRegistry.from_rules_dicts(linear_rules)
    df = build_catalog(reg)
    if df.empty:
        return df
    df = df.copy()
    df["physical_models"] = [[model]] * len(df)
    return df


def _solve_directs(rule: dict, all_symbols, model: str) -> list[dict]:
    import sympy as sp
    from sympy.parsing.sympy_parser import parse_expr

    rhs_expr = parse_expr(rule["rhs"])
    lhs_sym = sp.Symbol(rule["lhs"])
    eq = sp.Eq(lhs_sym, rhs_expr)
    S = set(all_symbols)
    involved = (lhs_sym.free_symbols | rhs_expr.free_symbols) & S
    rows = []
    for target in involved:
        try:
            sols = sp.solve(eq, target, dict=False)
        except Exception:
            continue
        if not sols:
            continue
        if not isinstance(sols, (list, tuple)):
            sols = [sols]
        for rhs in sols:
            try:
                rhs_simp = sp.cancel(sp.together(rhs))
            except Exception:
                rhs_simp = rhs
            if target in (rhs_simp.free_symbols & S):
                continue
            fs = sorted((rhs_simp.free_symbols & S), key=lambda s: s.name)
            rows.append({
                "lhs": target.name,
                "rhs_text": str(rhs_simp),
                "rhs_vars": [s.name for s in fs],
                "rhs_var_count": len(fs),
                "physical_models": [model],
            })
    return rows


def _expand_against_catalog(rule: dict, linear_catalog: pd.DataFrame, all_symbols, model: str) -> list[dict]:
    import sympy as sp
    from sympy.parsing.sympy_parser import parse_expr

    rhs_expr = parse_expr(rule["rhs"])
    lhs_sym = sp.Symbol(rule["lhs"])
    rule_eq = sp.Eq(lhs_sym, rhs_expr)
    S = set(all_symbols)
    rule_vars = (lhs_sym.free_symbols | rhs_expr.free_symbols) & S
    rows = []
    seen = set()

    for x_sym in rule_vars:
        cat_rows = linear_catalog[linear_catalog["lhs"] == x_sym.name]
        for _, cat_row in cat_rows.iterrows():
            try:
                sub_expr = parse_expr(cat_row["rhs_text"])
            except Exception:
                continue
            substituted_eq = sp.Eq(
                rule_eq.lhs.subs(x_sym, sub_expr),
                rule_eq.rhs.subs(x_sym, sub_expr),
            )
            remaining = (
                substituted_eq.lhs.free_symbols | substituted_eq.rhs.free_symbols
            ) & S
            for target in remaining:
                try:
                    sols = sp.solve(substituted_eq, target, dict=False)
                except Exception:
                    continue
                if not sols:
                    continue
                if not isinstance(sols, (list, tuple)):
                    sols = [sols]
                for rhs in sols:
                    try:
                        rhs_simp = sp.cancel(sp.together(rhs))
                    except Exception:
                        rhs_simp = rhs
                    if target in (rhs_simp.free_symbols & S):
                        continue
                    key = (target.name, str(rhs_simp))
                    if key in seen:
                        continue
                    seen.add(key)
                    fs = sorted((rhs_simp.free_symbols & S), key=lambda s: s.name)
                    rows.append({
                        "lhs": target.name,
                        "rhs_text": str(rhs_simp),
                        "rhs_vars": [s.name for s in fs],
                        "rhs_var_count": len(fs),
                        "physical_models": [model],
                    })
    return rows


def build_nl_chunk(rule: dict, linear_catalog: pd.DataFrame, all_symbols, model: str) -> pd.DataFrame:
    """Build one non-linear chunk for one model: directs + expansion."""
    rows = _solve_directs(rule, all_symbols, model)
    rows.extend(_expand_against_catalog(rule, linear_catalog, all_symbols, model))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Compose + prune (cross-model dedup, per-model Pareto)
# ---------------------------------------------------------------------------

def compose_and_prune(rows: list[dict]) -> list[dict]:
    """Union per-model chunk rows into a single deduplicated catalog.

    Two passes:

    1. Cross-model dedup. Group by (lhs, frozenset(rhs_vars)) -- the
       semantic key the query consumer uses for feasibility. Multiple
       chunks producing the "same derivation" (different physical model
       partitions arriving at functionally equivalent rows) collapse
       into one entry, with physical_models = union of all source rows.
       rhs_text is deterministic: keep the alphabetically-first form
       so the output is reproducible.

    2. Per-model Pareto prune. For each physical model M, take the
       subset of derivations valid in M (M appears in physical_models),
       drop any whose rhs_vars is a strict superset of another's for
       the same lhs in that model. A derivation that survives in M
       but is dominated in N is tagged with {M} only.

    A row whose dedup group survives in no model after pruning is
    dropped entirely.
    """
    # Pass 1: cross-model dedup by (lhs, frozenset(rhs_vars))
    by_key: dict[tuple, dict] = {}
    for r in rows:
        key = (r["lhs"], frozenset(r["rhs_vars"]))
        if key in by_key:
            existing = by_key[key]
            existing_models = set(existing["physical_models"] or [])
            existing_models.update(r["physical_models"] or [])
            existing["physical_models"] = sorted(existing_models)
            # Keep alphabetically-first rhs_text for determinism
            if r["rhs_text"] < existing["rhs_text"]:
                existing["rhs_text"] = r["rhs_text"]
        else:
            by_key[key] = {
                "lhs": r["lhs"],
                "rhs_text": r["rhs_text"],
                "rhs_vars": list(key[1]),
                "rhs_var_count": len(key[1]),
                "physical_models": sorted(r["physical_models"] or []),
            }

    # Pass 2: per-model Pareto. For each model, drop dominated derivations.
    # Track which models each derivation survives in; emit with that surviving set.
    deduped = list(by_key.values())
    survivors: dict[tuple, set[str]] = {
        (r["lhs"], frozenset(r["rhs_vars"])): set() for r in deduped
    }
    all_models = sorted({m for r in deduped for m in r["physical_models"]})

    for m in all_models:
        in_model = [r for r in deduped if m in r["physical_models"]]
        by_lhs: dict[str, list[tuple[dict, frozenset]]] = {}
        for r in in_model:
            by_lhs.setdefault(r["lhs"], []).append((r, frozenset(r["rhs_vars"])))
        for lhs, items in by_lhs.items():
            items.sort(key=lambda x: len(x[1]))
            kept_sets: list[frozenset] = []
            for (r, s) in items:
                if any(small < s for small in kept_sets):
                    continue
                survivors[(r["lhs"], s)].add(m)
                kept_sets.append(s)

    # Emit each derivation tagged with the model set in which it survives
    final: list[dict] = []
    for key, models_kept in survivors.items():
        if not models_kept:
            continue
        r = by_key[key]
        final.append({
            "lhs": r["lhs"],
            "rhs_text": r["rhs_text"],
            "rhs_vars": r["rhs_vars"],
            "rhs_var_count": r["rhs_var_count"],
            "physical_models": sorted(models_kept),
        })
    return final


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def _chunk_path(cache_dir: Path, kind: str, model: str, chunk_hash: str) -> Path:
    safe_model = model.replace("/", "_").replace("\\", "_")
    safe_kind = kind.replace("/", "_").replace("\\", "_")
    return cache_dir / f"{safe_kind}-{safe_model}-{chunk_hash}.parquet"


def _load_chunk(cache_dir: Path, kind: str, model: str, chunk_hash: str) -> pd.DataFrame | None:
    path = _chunk_path(cache_dir, kind, model, chunk_hash)
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path, dtype_backend="numpy_nullable")
    except Exception:
        return None


def _save_chunk(cache_dir: Path, kind: str, model: str, chunk_hash: str, df: pd.DataFrame) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _chunk_path(cache_dir, kind, model, chunk_hash)
    df.to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------
# ProcessPool worker entry points (must be picklable)
# ---------------------------------------------------------------------------

def _worker_build_linear(rule_payloads_json: str, model: str) -> list[dict]:
    """Worker: build linear chunk for one model. Returns list of row dicts."""
    rules = json.loads(rule_payloads_json)
    df = build_linear_catalog(rules, model)
    return df.to_dict(orient="records")


def _worker_build_nl_chunk(
    rule_json: str,
    linear_records: list[dict],
    all_sym_names: list[str],
    model: str,
) -> list[dict]:
    """Worker: build one (model, NL-rule) chunk."""
    import sympy as sp

    rule = json.loads(rule_json)
    cat_df = pd.DataFrame(linear_records) if linear_records else pd.DataFrame(
        columns=["lhs", "rhs_text", "rhs_vars", "rhs_var_count", "physical_models"]
    )
    syms = [sp.Symbol(n) for n in all_sym_names]
    df = build_nl_chunk(rule, cat_df, syms, model)
    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Cheap plan: which chunks would rebuild on a build_chunked call?
# ---------------------------------------------------------------------------

def plan_chunked_build(rules: list[dict], cache_dir: Path | str) -> dict:
    """Inspect the chunk cache against the canonical hashes of `rules` and
    return what build_chunked would rebuild, without running any SymPy.

    Fast: classifies rules, computes per-model linear hashes and per-NL
    chunk hashes, checks the cache directory for the corresponding
    parquet files. No process pool, no expansion work, no row counts.

    Used by the sync tooling's default dry-run path so a no-change
    sync can report "0 chunks would rebuild" in under a second without
    paying the cold-build cost. Apply (and dry-run + simulate flag)
    still call build_chunked, which produces identical numbers because
    cache files are written there.

    Returns
    -------
    dict with keys:
      models          : sorted list of physical model names
      cache_dir       : absolute Path of cache directory
      would_rebuild   : list of dicts, one per chunk that would rebuild
                        {kind, model, rule (or None), hash}
      would_hit       : list of dicts in the same shape for cached chunks
      total_chunks    : len(would_rebuild) + len(would_hit)
    """
    cache_dir = Path(cache_dir)
    models = discover_models(rules)
    would_rebuild: list[dict] = []
    would_hit: list[dict] = []

    for m in models:
        m_rules = rules_for_model(rules, m)
        linear, nonlinear = classify_linear_nonlinear(m_rules)

        # Linear chunk
        lh = canonical_linear_hash(linear, m)
        entry = {"kind": "linear", "model": m, "rule": None, "hash": lh}
        if _chunk_path(cache_dir, "linear", m, lh).exists():
            would_hit.append(entry)
        else:
            would_rebuild.append(entry)

        # Non-linear chunks
        for r in nonlinear:
            nh = canonical_nl_chunk_hash(r, lh, m)
            entry = {
                "kind": f"nl-{r['lhs']}",
                "model": m,
                "rule": r["lhs"],
                "hash": nh,
            }
            if _chunk_path(cache_dir, f"nl-{r['lhs']}", m, nh).exists():
                would_hit.append(entry)
            else:
                would_rebuild.append(entry)

    return {
        "models": models,
        "cache_dir": cache_dir,
        "would_rebuild": would_rebuild,
        "would_hit": would_hit,
        "total_chunks": len(would_rebuild) + len(would_hit),
    }


# ---------------------------------------------------------------------------
# Top-level chunked + parallel build
# ---------------------------------------------------------------------------

_CATALOG_COLUMNS = ["lhs", "rhs_text", "rhs_vars", "rhs_var_count", "physical_models"]


def _canonical_catalog(rows: list[dict]) -> pd.DataFrame:
    """Build the catalog frame in a canonical, build-order-independent row order.

    The set of equations the build produces is deterministic, but the order
    rows are emitted in is not: the build walks set/dict collections of SymPy
    symbols whose iteration order shifts with per-process hash randomization.
    Sorting on a total-order key here makes the written artifact byte-stable
    across runs and machines, so a row-by-row diff of catalog.parquet reflects
    real changes to the equation set instead of incidental row shuffling.
    """
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=_CATALOG_COLUMNS)
    # rhs_vars / physical_models are variable/model *sets*; their stored element
    # order is hash-dependent and carries no meaning, so sort each cell to a
    # canonical order. Without this the row order is stable but the list cells
    # still serialise differently across builds.
    df["rhs_vars"] = df["rhs_vars"].map(lambda v: sorted(str(x) for x in v))
    df["physical_models"] = df["physical_models"].map(
        lambda v: sorted(str(x) for x in v)
    )
    keys = pd.DataFrame(
        {
            "lhs": df["lhs"].astype(str),
            "rhs_text": df["rhs_text"].astype(str),
            "rhs_var_count": df["rhs_var_count"],
            "rhs_vars_key": df["rhs_vars"].map(lambda v: ",".join(v)),
            "models_key": df["physical_models"].map(lambda v: ",".join(v)),
        }
    )
    order = keys.sort_values(
        ["lhs", "rhs_var_count", "rhs_text", "rhs_vars_key", "models_key"]
    ).index
    return df.loc[order].reset_index(drop=True)


def build_chunked(
    rules: list[dict],
    cache_dir: Path | str,
    n_workers: int | None = None,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Build the full catalog: per-model linear cores + per-(model, NL rule)
    chunks. Parallelized across all chunk tasks. Returns the composed
    Pareto-pruned DataFrame.

    Logs are structured: `[build_chunked] <key>=<value> ...`.
    """
    cache_dir = Path(cache_dir)
    t_start = time.perf_counter()

    models = discover_models(rules)
    log(f"[build_chunked] models n={len(models)} list={models}")

    # Per-model planning -----------------------------------------------------
    # plan[m] = {
    #   "linear_rules": [...], "linear_hash": "...",
    #   "linear_df": None | df,             # filled when ready
    #   "nl_tasks": [(rule, hash), ...],    # to dispatch in phase 2
    #   "nl_dfs": [None|df, ...],           # filled when ready
    # }
    plan: dict[str, dict] = {}
    for m in models:
        m_rules = rules_for_model(rules, m)
        linear, nonlinear = classify_linear_nonlinear(m_rules)
        linear_hash = canonical_linear_hash(linear, m)
        plan[m] = {
            "linear_rules": linear,
            "nonlinear_rules": nonlinear,
            "linear_hash": linear_hash,
            "linear_df": None,
            "nl_tasks": [],
            "nl_dfs": [None] * len(nonlinear),
        }
        log(
            f"[build_chunked] plan model={m} "
            f"n_linear={len(linear)} n_nonlinear={len(nonlinear)} "
            f"linear_hash={linear_hash[:10]}"
        )

    # Phase 1: linear cores --------------------------------------------------
    # First: check cache for every model's linear chunk. Anything missing
    # gets dispatched to the worker pool. Linear cores across models are
    # independent and parallelisable.
    linear_misses = []
    for m in models:
        h = plan[m]["linear_hash"]
        cached = _load_chunk(cache_dir, "linear", m, h)
        if cached is not None:
            plan[m]["linear_df"] = cached
            log(
                f"[build_chunked] linear cache=hit model={m} "
                f"hash={h[:10]} rows={len(cached)}"
            )
        else:
            linear_misses.append(m)

    workers = n_workers if n_workers is not None else max(1, os.cpu_count() or 1)
    if linear_misses:
        t0 = time.perf_counter()
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {}
            for m in linear_misses:
                futs[ex.submit(
                    _worker_build_linear,
                    json.dumps(plan[m]["linear_rules"]),
                    m,
                )] = m
            for fut in cf.as_completed(futs):
                m = futs[fut]
                rows = fut.result()
                df = pd.DataFrame(rows)
                plan[m]["linear_df"] = df
                _save_chunk(cache_dir, "linear", m, plan[m]["linear_hash"], df)
                log(
                    f"[build_chunked] linear cache=miss model={m} "
                    f"hash={plan[m]['linear_hash'][:10]} rows={len(df)} (built)"
                )
        log(
            f"[build_chunked] phase1_linear built n={len(linear_misses)} "
            f"workers={workers} wall_s={time.perf_counter() - t0:.2f}"
        )
    else:
        log("[build_chunked] phase1_linear all_cached")

    # Phase 2: nl-chunks -----------------------------------------------------
    # For each model with non-linear rules: hash each NL chunk against the
    # model's linear catalog, check cache, dispatch misses. All NL tasks
    # across all models go into a single ProcessPool.
    all_sym_names_global = [s.name for s in _all_symbols(rules)]
    nl_misses: list[tuple[str, int, dict, str]] = []  # (model, idx, rule, hash)
    for m in models:
        linear_hash = plan[m]["linear_hash"]
        for i, rule in enumerate(plan[m]["nonlinear_rules"]):
            h = canonical_nl_chunk_hash(rule, linear_hash, m)
            plan[m]["nl_tasks"].append((rule, h))
            cached = _load_chunk(cache_dir, f"nl-{rule['lhs']}", m, h)
            if cached is not None:
                plan[m]["nl_dfs"][i] = cached
                log(
                    f"[build_chunked] nl-chunk cache=hit model={m} "
                    f"lhs={rule['lhs']} hash={h[:10]} rows={len(cached)}"
                )
            else:
                nl_misses.append((m, i, rule, h))

    if nl_misses:
        t0 = time.perf_counter()
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {}
            for (m, i, rule, h) in nl_misses:
                linear_records = plan[m]["linear_df"].to_dict(orient="records") \
                    if plan[m]["linear_df"] is not None else []
                futs[ex.submit(
                    _worker_build_nl_chunk,
                    json.dumps(rule),
                    linear_records,
                    all_sym_names_global,
                    m,
                )] = (m, i, rule, h)
            for fut in cf.as_completed(futs):
                m, i, rule, h = futs[fut]
                rows = fut.result()
                df = pd.DataFrame(rows)
                plan[m]["nl_dfs"][i] = df
                _save_chunk(cache_dir, f"nl-{rule['lhs']}", m, h, df)
                log(
                    f"[build_chunked] nl-chunk cache=miss model={m} "
                    f"lhs={rule['lhs']} hash={h[:10]} rows={len(df)} (built)"
                )
        log(
            f"[build_chunked] phase2_nl built n={len(nl_misses)} "
            f"workers={workers} wall_s={time.perf_counter() - t0:.2f}"
        )
    else:
        log("[build_chunked] phase2_nl all_cached")

    # Compose + prune -------------------------------------------------------
    t0 = time.perf_counter()
    all_rows = []
    for m in models:
        if plan[m]["linear_df"] is not None and not plan[m]["linear_df"].empty:
            all_rows.extend(plan[m]["linear_df"].to_dict(orient="records"))
        for df in plan[m]["nl_dfs"]:
            if df is not None and not df.empty:
                all_rows.extend(df.to_dict(orient="records"))
    pruned_rows = compose_and_prune(all_rows)
    log(
        f"[build_chunked] compose+prune union={len(all_rows)} "
        f"pareto={len(pruned_rows)} wall_s={time.perf_counter() - t0:.2f}"
    )

    log(
        f"[build_chunked] DONE total_wall_s={time.perf_counter() - t_start:.2f} "
        f"rows={len(pruned_rows)}"
    )
    return _canonical_catalog(pruned_rows)


# ---------------------------------------------------------------------------
# Serial reference (for verification)
# ---------------------------------------------------------------------------

def build_serial_reference(rules: list[dict]) -> pd.DataFrame:
    """Build the same catalog one model at a time, in-process, no caching.

    Used by verify_against_parallel to confirm the parallel build produces
    the same (lhs, rhs_vars, physical_models) key-set as the serial path.
    """
    all_rows = []
    for m in discover_models(rules):
        m_rules = rules_for_model(rules, m)
        linear, nonlinear = classify_linear_nonlinear(m_rules)
        linear_df = build_linear_catalog(linear, m)
        syms = _all_symbols(rules)
        if not linear_df.empty:
            all_rows.extend(linear_df.to_dict(orient="records"))
        for rule in nonlinear:
            chunk = build_nl_chunk(rule, linear_df, syms, m)
            if not chunk.empty:
                all_rows.extend(chunk.to_dict(orient="records"))
    return _canonical_catalog(compose_and_prune(all_rows))


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _row_keyset(df: pd.DataFrame) -> set[tuple]:
    """Reduce a catalog to {(lhs, frozenset(rhs_vars), tuple(sorted(physical_models)))}.

    rhs_text can vary across sympy canonical forms; the consumer only uses
    (lhs, rhs_vars, physical_models), so that's the equality we check.
    """
    out = set()
    for _, row in df.iterrows():
        pm = tuple(sorted(row.get("physical_models") or []))
        out.add((row["lhs"], frozenset(row["rhs_vars"]), pm))
    return out


def verify_against_serial(rules: list[dict], cache_dir: Path | str | None = None) -> dict:
    """Build via serial path and chunked-parallel path, compare keysets."""
    t0 = time.perf_counter()
    serial_df = build_serial_reference(rules)
    t_serial = time.perf_counter() - t0

    if cache_dir is None:
        cache_dir = Path(tempfile.mkdtemp(prefix="catalog_verify_"))
    t0 = time.perf_counter()
    parallel_df = build_chunked(rules, cache_dir=cache_dir, log=lambda _: None)
    t_parallel = time.perf_counter() - t0

    s_keys = _row_keyset(serial_df)
    p_keys = _row_keyset(parallel_df)
    return {
        "serial_rows": len(serial_df),
        "parallel_rows": len(parallel_df),
        "serial_keys": len(s_keys),
        "parallel_keys": len(p_keys),
        "only_serial": sorted(s_keys - p_keys),
        "only_parallel": sorted(p_keys - s_keys),
        "equal_keysets": s_keys == p_keys,
        "wall_serial_s": t_serial,
        "wall_parallel_s": t_parallel,
    }
