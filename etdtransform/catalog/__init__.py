"""
etdtransform.catalog — Adaptive equation catalog for variable derivation.

Provides tools to:
  1. Build a catalog of all possible derivations for each variable from any valid
     subset of other variables (using equations defined in the Grist Rule table).
  2. Serialize/deserialize the catalog to/from Parquet for fast loading.
  3. At query time, assess which columns in a specific dataset are 'effectively raw'
     (have sufficient real data) and select the best derivation formulas.

Public API:
  EquationRegistry      — holds SymPy equations parsed from Rule CSV dicts
  build_catalog         — BFS expansion + Pareto pruning → flat catalog DataFrame
  catalog_to_parquet    — serialize catalog DataFrame to parquet with hash metadata
  catalog_from_parquet  — load catalog DataFrame from parquet, verify hash
  DatasetAdapter        — per-dataset query: feasibility report + execution plan
"""

from .builder import build_catalog, catalog_from_parquet, catalog_to_parquet
from .query import DatasetAdapter
from .registry import EquationRegistry
from .validate import validate_rule_expressions

__all__ = [
    "EquationRegistry",
    "build_catalog",
    "catalog_to_parquet",
    "catalog_from_parquet",
    "DatasetAdapter",
    "validate_rule_expressions",
]
