"""
DatasetAdapter — per-dataset query against the pre-built catalog.

Given a DataFrame (the actual mapped dataset from a provider), the adapter:
  1. Assesses which columns have sufficient real data ('effectively raw').
  2. Finds feasible catalog entries for each required target variable.
  3. Selects the best derivation per target (sorted by preference).
  4. Returns an execution plan: [(target_col, rhs_sympy_expr), ...] in
     topological order so each column is computed after its dependencies.

'Effectively raw' means: the column exists in the DataFrame AND its non-null
fraction is >= completeness_threshold (default 0.95). This is assessed per
dataset — a column that is 'effectively raw' for Provider B (who supplies it
directly) may be 'derived' for Provider A (who only supplies its components).
"""

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

import pandas as pd


class DatasetAdapter:
    """
    Query interface over a pre-loaded catalog DataFrame.

    Load the catalog once per process (from parquet via catalog_from_parquet),
    then call feasibility_report() or execution_plan() for each dataset.

    Parameters
    ----------
    catalog_df : pd.DataFrame
        Output of build_catalog() or loaded from catalog_from_parquet().
        Must have columns: lhs (str), rhs_text (str), rhs_vars (list[str]),
        rhs_var_count (int).
    completeness_threshold : float
        Minimum non-null fraction for a column to be considered 'effectively raw'.
        Default 0.95 (configurable per use case).
    """

    def __init__(self, catalog_df: pd.DataFrame, completeness_threshold: float = 0.95):
        self._catalog = catalog_df
        self.completeness_threshold = completeness_threshold

    # ------------------------------------------------------------------
    # Column assessment
    # ------------------------------------------------------------------

    def assess_columns(self, df: pd.DataFrame) -> dict:
        """
        Return {col_name: non_null_fraction} for each column in df.
        """
        return {col: float(df[col].notna().mean()) for col in df.columns}

    def effective_raw(self, df: pd.DataFrame) -> set:
        """
        Return the set of column names considered 'effectively raw' for this dataset:
        - column exists in df AND
        - non-null fraction >= completeness_threshold
        """
        return {
            col for col, frac in self.assess_columns(df).items()
            if frac >= self.completeness_threshold
        }

    # ------------------------------------------------------------------
    # Feasibility check (iterative until fixed point)
    # ------------------------------------------------------------------

    def _filter_by_physical_model(self, rows: pd.DataFrame, physical_model) -> pd.DataFrame:
        """
        If physical_model is given and the catalog has a physical_models column,
        return only rows where physical_model appears in that row's physical_models list.
        Otherwise return rows unchanged.
        """
        if physical_model is None:
            return rows
        if "physical_models" not in rows.columns:
            return rows
        return rows[rows["physical_models"].apply(
            lambda ms: physical_model in (ms if isinstance(ms, list) else [])
        )]

    def _feasible_entries_for(self, target: str, available: set, physical_model=None) -> list:
        """
        Return catalog rows where lhs == target AND set(rhs_vars) ⊆ available,
        optionally filtered to those whose physical_models includes physical_model.
        The 'n_non_raw' count uses the initial effective_raw set (not the
        growing available set) so preference reflects actual provider data.
        """
        rows = self._catalog[self._catalog["lhs"] == target]
        rows = self._filter_by_physical_model(rows, physical_model)
        feasible = []
        for _, row in rows.iterrows():
            rhs_vars = set(row["rhs_vars"])
            if rhs_vars <= available:
                feasible.append(row)
        return feasible

    def feasibility_report(self, df: pd.DataFrame, required_targets: set, physical_model=None) -> dict:
        """
        Determine which required target columns can be derived from this dataset.

        Uses an iterative fixed-point approach: a newly derivable column is added
        to the available set and may unlock further derivations.

        Parameters
        ----------
        df : pd.DataFrame
            The actual mapped dataset.
        required_targets : set[str]
            Column names that need to be derived.
        physical_model : str or None
            When given, only catalog entries whose physical_models list includes
            this value are considered. When None, all entries are considered.

        Returns
        -------
        dict with keys:
          effectively_raw : set[str]
          column_completeness : dict[str, float]
          derivable : dict[str, list[dict]]  — sorted catalog entries per target
          not_derivable : list[str]
        """
        eff_raw = self.effective_raw(df)
        completeness = self.assess_columns(df)
        available = set(eff_raw)

        remaining = {t for t in required_targets if t not in available}
        derivable = {}

        changed = True
        while changed and remaining:
            changed = False
            still_remaining = set()
            for target in remaining:
                entries = self._feasible_entries_for(target, available, physical_model)
                if entries:
                    # Sort: fewest non-raw-inputs first, then fewest total vars
                    entries.sort(key=lambda r: (
                        sum(1 for v in r["rhs_vars"] if v not in eff_raw),
                        r["rhs_var_count"],
                    ))
                    derivable[target] = [r.to_dict() for r in entries]
                    available.add(target)
                    changed = True
                else:
                    still_remaining.add(target)
            remaining = still_remaining

        return {
            "effectively_raw": eff_raw,
            "column_completeness": completeness,
            "derivable": derivable,
            "not_derivable": sorted(remaining),
        }

    # ------------------------------------------------------------------
    # Execution plan
    # ------------------------------------------------------------------

    def execution_plan(self, df: pd.DataFrame, required_targets: set, physical_model=None) -> list:
        """
        Return [(target_col_name, rhs_sympy_expr), ...] in topological order.

        Targets already effectively raw are excluded (no derivation needed).
        Parses rhs_text via parse_expr only for the selected entries.

        Parameters
        ----------
        df : pd.DataFrame
        required_targets : set[str]
        physical_model : str or None
            When given, only catalog entries whose physical_models list includes
            this value are used. When None, all entries are considered.

        Returns
        -------
        list[tuple[str, sp.Expr]]

        Raises
        ------
        ValueError
            If any required target is not derivable from the available columns,
            or if a dependency cycle is detected.
        """
        report = self.feasibility_report(df, required_targets, physical_model)

        if report["not_derivable"]:
            raise ValueError(
                f"Cannot derive required column(s): {report['not_derivable']}. "
                f"Effectively raw columns: {sorted(report['effectively_raw'])}"
            )

        eff_raw = report["effectively_raw"]

        # Select the best (first) catalog entry for each derivable target
        # and parse its rhs_text into a SymPy expression
        selected = {}  # target_name → (rhs_expr, rhs_vars_set)
        for target, entries in report["derivable"].items():
            best = entries[0]
            rhs_expr = parse_expr(best["rhs_text"])
            selected[target] = (rhs_expr, set(best["rhs_vars"]))

        # Topological sort: order derivations so each target is computed
        # after all its non-raw dependencies
        order = _toposort(selected, eff_raw)

        return [(name, selected[name][0]) for name in order]


# ---------------------------------------------------------------------------
# Topological sort helpers
# ---------------------------------------------------------------------------

def _toposort(selected: dict, eff_raw: set) -> list:
    """
    Topologically sort the selected derivations.

    selected : dict[target_name] → (rhs_expr, rhs_vars_set)
    eff_raw  : set of column names with real data (no derivation needed)

    Returns ordered list of target names. Raises ValueError on cycle.
    """
    # Build adjacency: dependency_name → {targets that need it}
    # A target depends on any rhs_var that is itself a derived target (not eff_raw)
    derived_names = set(selected.keys())

    deps = {name: set() for name in derived_names}
    for name, (_, rhs_vars) in selected.items():
        for v in rhs_vars:
            if v in derived_names:
                deps[name].add(v)  # name depends on v

    # Kahn's algorithm
    in_degree = {n: len(d) for n, d in deps.items()}
    queue = [n for n, d in in_degree.items() if d == 0]
    order = []

    while queue:
        queue.sort()  # deterministic ordering among zero-degree nodes
        node = queue.pop(0)
        order.append(node)
        # Find all nodes that depend on `node`
        for other in derived_names:
            if node in deps[other]:
                in_degree[other] -= 1
                if in_degree[other] == 0:
                    queue.append(other)

    if len(order) != len(derived_names):
        cycle_nodes = derived_names - set(order)
        raise ValueError(
            f"Dependency cycle detected among derived columns: {cycle_nodes}"
        )

    return order
