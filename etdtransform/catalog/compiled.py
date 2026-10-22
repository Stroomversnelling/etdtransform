"""
Compiled catalog expressions -- the callable form of a catalog formula.

A catalog rule is stored as text, parsed to a SymPy expression, and then turned
into a Python callable by ``sympy.lambdify`` before it can be applied to an ibis
table or a pandas frame. Compiling is a property of the expression alone, not of
the table, frame or household it is applied to, so it is done here once and the
result reused.

``lambdify`` generates Python source and executes it, which costs roughly a
millisecond per expression. Applying the result costs nothing by comparison.

The zero-divisor rule lives here, stated once, because all three evaluators need
the same answer:

A ratio whose denominator is zero is undefined, and pd.NA is how this pipeline
says undefined. Without the guard the engine returns inf or NaN -- a VALUE that
passes notnull() and poisons every later sum and mean. ``as_numer_denom`` folds a
sum of ratios onto one denominator, so guarding that one product covers each of
them.

A **constant** divisor is not a ratio for this purpose: nulling a nonzero literal
can never fire, so such an expression compiles as a whole and takes no guard.
"""

import functools

import sympy as sp


class CompiledExpression:
    """A catalog expression and the callables it was compiled into.

    Attributes
    ----------
    expr : sympy.Expr
        The expression this was built from, kept for messages and provenance.
    symbols : tuple of str
        Input column names, sorted. These are the columns a caller must supply.
    is_ratio : bool
        True only when the denominator carries free symbols, so a zero divisor
        is reachable and the guard applies.
    denominator_symbols : tuple of str
        Input column names of the denominator, empty when not ``is_ratio``.
    """

    __slots__ = (
        "_denominator",
        "_numerator",
        "_numerator_symbols",
        "_whole",
        "denominator_symbols",
        "expr",
        "is_ratio",
        "symbols",
    )

    def __init__(self, expr):
        self.expr = expr
        self.symbols = tuple(sorted((str(s) for s in expr.free_symbols)))

        numerator, denominator = expr.as_numer_denom()
        self.is_ratio = bool(
            denominator != 1 and denominator.free_symbols
        )

        if self.is_ratio:
            self._whole = None
            self._numerator_symbols = tuple(
                sorted(str(s) for s in numerator.free_symbols)
            )
            self.denominator_symbols = tuple(
                sorted(str(s) for s in denominator.free_symbols)
            )
            self._numerator = _compile(numerator, self._numerator_symbols)
            self._denominator = _compile(denominator, self.denominator_symbols)
        else:
            self._whole = _compile(expr, self.symbols)
            self._numerator = None
            self._numerator_symbols = ()
            self.denominator_symbols = ()
            self._denominator = None

    def __repr__(self):
        kind = "ratio" if self.is_ratio else "whole"
        return f"CompiledExpression({self.expr}, {kind})"

    def whole(self, resolve):
        """The expression applied to the columns *resolve* returns by name.

        Only valid when ``is_ratio`` is False.
        """
        if self._whole is None:
            raise ValueError(
                f"{self.expr} is a ratio; apply numerator and denominator "
                f"separately so the zero divisor can be guarded."
            )
        return self._whole(*[resolve(name) for name in self.symbols])

    def numerator(self, resolve):
        """The numerator applied to the columns *resolve* returns by name."""
        if self._numerator is None:
            raise ValueError(f"{self.expr} is not a ratio; use whole().")
        return self._numerator(
            *[resolve(name) for name in self._numerator_symbols]
        )

    def denominator(self, resolve):
        """The denominator applied to the columns *resolve* returns by name."""
        if self._denominator is None:
            raise ValueError(f"{self.expr} is not a ratio; use whole().")
        return self._denominator(
            *[resolve(name) for name in self.denominator_symbols]
        )


def _compile(expr, symbol_names):
    """A callable for *expr* taking *symbol_names* as positional arguments.

    A constant expression is compiled too, so every caller applies a callable
    rather than branching on whether one exists.
    """
    return sp.lambdify(list(symbol_names), expr)


@functools.lru_cache(maxsize=None)
def compile_expression(expr) -> CompiledExpression:
    """The compiled form of *expr*, built once per expression per process.

    SymPy expressions are immutable and hashable, so the same formula -- whether
    it arrives from another schema group, another household or another stage --
    resolves to the same object and is compiled once.

    Parameters
    ----------
    expr : sympy.Expr

    Returns
    -------
    CompiledExpression
    """
    return CompiledExpression(expr)


def as_compiled(expr_or_compiled) -> CompiledExpression:
    """*expr_or_compiled* as a :class:`CompiledExpression`.

    Accepts either form so a caller holding a raw SymPy expression does not
    have to compile it itself.
    """
    if isinstance(expr_or_compiled, CompiledExpression):
        return expr_or_compiled
    return compile_expression(expr_or_compiled)


def compile_plan(plan) -> list:
    """An execution plan with its expressions compiled.

    *plan* is the ``[(target_column, sympy_expr), ...]`` list
    :meth:`DatasetAdapter.execution_plan_for_available` returns. The result has
    the same shape with each expression replaced by its compiled form, so the
    compile happens once for the plan rather than once per table it is applied
    to.
    """
    return [(column, as_compiled(expr)) for column, expr in plan]
