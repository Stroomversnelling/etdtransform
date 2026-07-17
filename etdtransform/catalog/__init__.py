"""
etdtransform.catalog — Adaptive equation catalog for variable derivation.

Provides tools to:
  1. Build a catalog of all possible derivations for each variable from any valid
     subset of other variables (using equations defined in the data model's
rule table).
  2. Serialize/deserialize the catalog to/from Parquet for fast loading.
  3. At query time, assess which columns in a specific dataset are 'effectively raw'
     (have sufficient real data) and select the best derivation formulas.

Public API:
  EquationRegistry      — holds SymPy equations parsed from Rule CSV dicts
  build_catalog         — BFS expansion + Pareto pruning → flat catalog DataFrame
                          (linear-only; raises on non-linear rules)
  build_chunked         — chunked, cacheable, parallel build partitioned by
                          physical model; handles linear + non-linear rules
                          via sp.solve directs + expansion. Preferred entrypoint
                          for the project's sync tooling since it caches per-chunk and
                          avoids redundant work on no-change syncs.
  plan_chunked_build    — cheap dry-run: hash + cache-check only, no SymPy.
                          Returns the would_rebuild / would_hit list so
                          sync's default dry-run can report what apply would
                          do in <1s.
  build_serial_reference — same algorithm as build_chunked but in-process and
                          no caching, used for correctness verification.
  verify_against_serial — run both build_chunked and build_serial_reference,
                          confirm semantic key-set equality.
  catalog_to_parquet    — serialize catalog DataFrame to parquet with hash metadata
  catalog_from_parquet  — load catalog DataFrame from parquet, verify hash
  DatasetAdapter        — per-dataset query: feasibility report + execution plan
"""

from .builder import build_catalog, catalog_from_parquet, catalog_to_parquet
from .chunked import (
    build_chunked,
    build_serial_reference,
    canonical_linear_hash,
    canonical_nl_chunk_hash,
    classify_linear_nonlinear,
    compose_and_prune,
    discover_models,
    plan_chunked_build,
    verify_against_serial,
)
from .query import DatasetAdapter
from .registry import EquationRegistry
from .validate import validate_rule_expressions

__all__ = [
    "EquationRegistry",
    "build_catalog",
    "build_chunked",
    "build_serial_reference",
    "canonical_linear_hash",
    "canonical_nl_chunk_hash",
    "catalog_to_parquet",
    "catalog_from_parquet",
    "classify_linear_nonlinear",
    "compose_and_prune",
    "DatasetAdapter",
    "discover_models",
    "plan_chunked_build",
    "validate_rule_expressions",
    "verify_against_serial",
]
