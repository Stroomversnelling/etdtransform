"""
EquationRegistry — parses Rule CSV dicts into SymPy equations.

Rule CSV dicts have the form:
  {"lhs": str, "rhs": str, "rhs_vars": list[str], "physical_models": list[str]}
where `lhs` is the variable being defined and `rhs` is a SymPy-parseable expression
string using Python arithmetic operators and variable names as identifiers.

Example:
  {"lhs": "TerugleveringTotaalNetto",
   "rhs": "ElektriciteitTerugleveringLaagDiff + ElektriciteitTerugleveringHoogDiff",
   "rhs_vars": ["ElektriciteitTerugleveringLaagDiff", "ElektriciteitTerugleveringHoogDiff"],
   "physical_models": ["Universeel", "All-Electric", "Hybride"]}
"""

from dataclasses import dataclass, field

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr


@dataclass
class EquationRegistry:
    equations: list  # list[sp.Eq]
    all_variable_names: list  # list[str] — union of all LHS + RHS variable names
    equation_models: list  # list[frozenset[str]] — physical models per equation (parallel to equations)

    @classmethod
    def from_rules_dicts(cls, rules: list) -> "EquationRegistry":
        """
        Parse rule dicts into SymPy Eq objects.

        Parameters
        ----------
        rules : list[dict]
            Each dict must have "lhs" (str) and "rhs" (str).
            Optional: "physical_models" (list[str]) — physical models this rule applies to.
            SymPy creates Symbol objects automatically for any identifier in the
            expression strings — no pre-declared symbol list needed.

        Returns
        -------
        EquationRegistry
        """
        equations = []
        equation_models = []
        all_var_names = set()

        for rule in rules:
            lhs_name = rule["lhs"]
            rhs_str = rule["rhs"]
            models = frozenset(rule.get("physical_models", []))

            # parse_expr auto-creates SymPy Symbol objects for any identifier.
            # We do not pass local_dict here — SymPy's symbol cache ensures that
            # symbols with the same name are identical objects across calls.
            rhs_expr = parse_expr(rhs_str)
            lhs_sym = sp.Symbol(lhs_name)

            equations.append(sp.Eq(lhs_sym, rhs_expr))
            equation_models.append(models)
            all_var_names.add(lhs_name)
            all_var_names.update(s.name for s in rhs_expr.free_symbols)

        return cls(
            equations=equations,
            all_variable_names=sorted(all_var_names),
            equation_models=equation_models,
        )

    def symbols(self) -> dict:
        """Return {name: sp.Symbol} for all variables in the registry."""
        return {name: sp.Symbol(name) for name in self.all_variable_names}
