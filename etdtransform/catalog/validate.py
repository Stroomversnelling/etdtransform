"""
validate_rule_expressions — SymPy-based validation of Rule table entries.

Kept in etdtransform to contain the SymPy dependency. Called by sync_data_model.py
(etdworkflow) via a thin import, so the rest of etdworkflow stays SymPy-free.
"""

from sympy.parsing.sympy_parser import parse_expr


def validate_rule_expressions(rules: list) -> list:
    """
    Validate each rule's expression and cross-check rhs_vars against parsed symbols.

    For each rule:
      1. Parse the ``rhs`` string with SymPy — report if unparseable.
      2. Compare the free symbols extracted from the parsed expression to the
         ``rhs_vars`` list — report any mismatch.

    Parameters
    ----------
    rules : list[dict]
        Each dict must have:
          "lhs"      str        — target variable name
          "rhs"      str        — SymPy-parseable expression string
          "rhs_vars" list[str]  — expected input variable names

    Returns
    -------
    list[str]
        Issue strings (empty list means everything is valid).
    """
    issues = []

    for rule in rules:
        lhs = str(rule.get("lhs", "?")).strip()
        rhs_str = str(rule.get("rhs", "")).strip()
        rhs_vars = set(rule.get("rhs_vars", []))

        if not rhs_str:
            issues.append(f"Rule '{lhs}': expression is empty")
            continue

        try:
            expr = parse_expr(rhs_str)
        except Exception as exc:
            issues.append(
                f"Rule '{lhs}': expression cannot be parsed as a SymPy expression — {exc}"
            )
            continue

        extracted = {s.name for s in expr.free_symbols}

        if extracted != rhs_vars:
            parts = []
            extra = extracted - rhs_vars
            missing = rhs_vars - extracted
            if extra:
                parts.append(f"in expression but not in rhs_vars: {sorted(extra)}")
            if missing:
                parts.append(f"in rhs_vars but not in expression: {sorted(missing)}")
            issues.append(
                f"Rule '{lhs}': rhs_vars mismatch — {'; '.join(parts)}"
            )

    return issues
