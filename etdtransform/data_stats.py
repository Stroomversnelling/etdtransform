"""
Per-(HuisIdBSV, column, season) stats producer for the wide pipeline
parquets emitted by etdtransform (``household_default.parquet``,
``household_imputed.parquet``, ``household_calculated.parquet``).

This module is the symmetric counterpart to
``etdmap.mapping_helpers.get_data_stats``. That function handles the
raw and mapped stages -- many small per-household parquet files, fanned
out with ProcessPoolExecutor and pandas vectorised reductions. This
function handles transform's own output: a single wide parquet per
stage, processed lazily through ibis / DuckDB so the parquet stays on
disk and only the per-(HH, col) aggregate result lands in pandas.

The output schema is identical to ``get_data_stats`` (enforced via the
shared ``etdmap._STATS_DTYPES`` contract -- parent ADR-018) so a single
downstream report consumes both producers without branching on stage.

Engine choice: ibis-on-DuckDB, not polars. A memory-aware benchmark
showed polars peaks at ~50 GB RSS on this workload (unusable on
laptop-class hardware) while ibis stays at 2 GB exact or 0.4 GB with
``approx_quantile``. A wall-time-only profile is insufficient here:
it mis-ranks polars as fastest while hiding its memory cost.

Quantile precision: defaults to DuckDB's ``approx_quantile`` (t-digest)
because it is 2x faster than exact AND 5x less memory AND fits the
laptop budget. Pass ``precision="exact"`` for byte-equal comparison
work (e.g. parked cross-stage drift reports). The chosen mode is
recorded in the output ``quantile_mode`` column so downstream consumers
can branch on it.
"""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

from etdmap.data_stats import (
    DEFAULT_SEASON_MONTHS,
    _STATS_DTYPES,  # noqa: F401  -- imported for documentation; _cast_stats_dtypes applies it
    _cast_stats_dtypes,
)

logger = logging.getLogger(__name__)


def get_data_stats(
        parquet_path,
        *,
        seasonal: bool = False,
        seasons: dict | None = None,
        value_cols: list | None = None,
        excluded_cols: set | None = None,
        precision: Literal["approx", "exact"] = "approx",
) -> pd.DataFrame:
    """
    Per-(HuisIdBSV, column, season) summary stats from a wide pipeline
    parquet (``household_default`` / ``household_imputed`` /
    ``household_calculated``).

    The wide household parquet is keyed by (HuisIdBSV, ReadingDate) and
    contains every household's rows in a single file. This function
    emits the same per-entity long-format stats that
    ``etdmap.mapping_helpers.get_data_stats`` produces in mapped mode,
    so a single downstream report consumes both producers without
    branching on stage. Schema contract is ``etdmap._STATS_DTYPES``
    (parent ADR-018); ``_cast_stats_dtypes`` is applied before return.

    Implementation: one ibis lazy aggregation per (season slice x
    column) combination. Predicate pushdown filters at the parquet scan
    layer so each season is a single DuckDB pass with no full
    materialisation in pandas. To preserve parity with the per-HH
    mapped path -- which emits a row even for all-NA columns -- the
    (HH x col) cross product is reinstated after the lazy aggregation,
    with ``count=0`` and the per-(HH, season) row count fed into
    ``missing`` / ``errors``.

    Parameters
    ----------
    parquet_path : str | os.PathLike
        Path to the wide household parquet.
    seasonal : bool, optional
        If True, emit one row per (HuisIdBSV, column, season) where
        the season set is 'annual' plus each entry in ``seasons``.
        Default False (annual only).
    seasons : dict[str, set[int]] | None, optional
        Season-name -> set-of-month-numbers (1..12). None falls back to
        ``etdmap.mapping_helpers.DEFAULT_SEASON_MONTHS`` (the
        Netherlands cold/warm verwarmingsperiode partition). The name
        'annual' is reserved.
    value_cols : list[str] | None, optional
        Columns to stat. None means "auto-discover all numeric /
        boolean columns in the parquet except the identifier / time
        columns".
    excluded_cols : set[str] | None, optional
        Column names to exclude from auto-discovery. None falls back
        to ``{"HuisIdBSV", "ProjectIdBSV", "ReadingDate"}``.
    precision : {'approx', 'exact'}, default 'approx'
        Quantile-compute precision:
          * ``'approx'`` -- DuckDB t-digest via ``approx_quantile``.
            Median rel-err ~0.03 % on well-distributed columns; up to
            ~1-2 % on most, with isolated tail-quantile outliers up
            to ~26 % on heavy-tailed cumulative meters. ~2x faster
            and ~5x less peak RSS than exact.
            Adequate for outlier-detection reports.
          * ``'exact'`` -- pandas-compatible linear-interpolated
            quantile. Required for byte-equal cross-stage comparisons
            (e.g. comparing mapped-stage stats to default-stage stats).

    Returns
    -------
    pd.DataFrame
        Long-format stats with one row per (HuisIdBSV, column, season).
        Columns match the ``etdmap._STATS_DTYPES`` contract plus a
        ``quantile_mode`` column recording the precision used so
        downstream consumers can detect approximate values.
    """
    import ibis  # heavy dep; lazy-imported so module load stays cheap

    parquet_path = str(parquet_path)
    tbl = ibis.read_parquet(parquet_path)
    schema = tbl.schema()

    if excluded_cols is None:
        excluded_cols = {"HuisIdBSV", "ProjectIdBSV", "ReadingDate"}

    if value_cols is None:
        # Auto-discover numeric / boolean columns. ibis is_numeric() does
        # NOT include booleans, so we test both. Datetime and string
        # columns are skipped here -- the kernel's datetime min/max and
        # object top5 branches are intentionally mapped/raw only.
        value_cols = []
        for name in schema.names:
            if name in excluded_cols:
                continue
            dt = schema[name]
            if dt.is_numeric() or dt.is_boolean():
                value_cols.append(name)

    if not value_cols:
        return pd.DataFrame()

    # Split numeric and boolean value cols. The mapped-mode kernel
    # routes booleans through a min/max/mean-only path (no std / median
    # / quantiles / iqr); we do the same here for parity. Booleans
    # also fuse cleanly in DuckDB (cheap reductions, one scan for many
    # cols), while quantile-heavy numerics get one ibis execute per
    # column (DuckDB does not fuse quantile() across columns -- see
    # `compute-stats-strategies-2026-05-08.md` §3).
    numeric_cols = [
        c for c in value_cols
        if schema[c].is_numeric() and not schema[c].is_boolean()
    ]
    bool_cols = [c for c in value_cols if schema[c].is_boolean()]

    # Cast booleans to float64 at the lazy layer so min / max / mean
    # reductions return floats; ibis quantile() on a boolean returns
    # boolean (which then breaks numpy subtraction downstream), and
    # bool mean lands the fail-rate semantic we want anyway.
    if bool_cols:
        tbl = tbl.mutate(**{c: tbl[c].cast("float64") for c in bool_cols})

    season_map = DEFAULT_SEASON_MONTHS if seasons is None else seasons
    if seasonal:
        # 'annual' first -- the full-year slice, no month filter,
        # always emitted. The reserved name in season_map is skipped.
        slice_specs = [("annual", None)]
        for name, month_set in season_map.items():
            if name == "annual":
                continue
            slice_specs.append((name, set(month_set)))
    else:
        slice_specs = [("annual", None)]

    # Quantile op selector. approx is the laptop-friendly default;
    # exact is opt-in for byte-equal cross-stage diffs.
    def quantile_op(expr, q: float):
        return (
            expr.quantile(q) if precision == "exact" else expr.approx_quantile(q)
        )

    long_pieces = []
    group_n_pieces = []
    huis_universe: set = set()

    for season_name, month_set in slice_specs:
        if month_set is None:
            slc = tbl
        else:
            slc = tbl.filter(tbl.ReadingDate.month().isin(list(month_set)))

        # Per (HuisIdBSV) row count for this season slice. Drives the
        # missing / errors columns: missing = group_n - count(non-null).
        # One small ibis execute per season.
        n_df = (
            slc.group_by("HuisIdBSV")
               .aggregate(_group_n=slc.HuisIdBSV.count())
               .execute()
        )
        n_df["season"] = season_name
        group_n_pieces.append(n_df)
        huis_universe.update(n_df["HuisIdBSV"].tolist())

        # Numeric cols: one ibis execute per column. DuckDB does NOT
        # fuse quantile() expressions across cols, so per-col is the
        # correct shape (one parquet scan per col, ~4 quantile passes
        # inside each). The approx_quantile path is faster AND uses
        # less RAM than exact because t-digest state is bounded; pick
        # via `precision` arg.
        for col in numeric_cols:
            e = slc[col]
            agg_exprs = {
                "count": e.count(),
                "mean": e.mean(),
                "std": e.std(),
                "min": e.min(),
                "max": e.max(),
                "median": quantile_op(e, 0.5),
                "p01": quantile_op(e, 0.01),
                "p25": quantile_op(e, 0.25),
                "p75": quantile_op(e, 0.75),
                "p99": quantile_op(e, 0.99),
            }
            sub = slc.group_by("HuisIdBSV").aggregate(**agg_exprs).execute()
            sub = sub[sub["count"].fillna(0) > 0].copy()
            if sub.empty:
                continue
            sub["iqr"] = sub["p75"] - sub["p25"]
            sub.insert(0, "variable", col)
            sub["season"] = season_name
            long_pieces.append(sub)

        # Boolean cols: one fused ibis execute for all of them.
        # No quantiles -- count / mean / min / max exhaust the kernel's
        # bool semantics. DuckDB fuses these cheap reductions across
        # columns in a single scan; ~110 validators collapse from
        # ~110 executes to one.
        if bool_cols:
            agg_exprs_bool: dict = {}
            for col in bool_cols:
                e = slc[col]
                agg_exprs_bool[f"{col}__count"] = e.count()
                agg_exprs_bool[f"{col}__mean"] = e.mean()
                agg_exprs_bool[f"{col}__min"] = e.min()
                agg_exprs_bool[f"{col}__max"] = e.max()
            wide = slc.group_by("HuisIdBSV").aggregate(**agg_exprs_bool).execute()

            bool_pieces = []
            for col in bool_cols:
                sub = wide[["HuisIdBSV"]].copy()
                sub["variable"] = col
                sub["count"] = wide[f"{col}__count"]
                sub["mean"] = wide[f"{col}__mean"]
                sub["min"] = wide[f"{col}__min"]
                sub["max"] = wide[f"{col}__max"]
                sub = sub[sub["count"].fillna(0) > 0]
                bool_pieces.append(sub)
            if bool_pieces:
                bool_long = pd.concat(bool_pieces, ignore_index=True)
                bool_long["season"] = season_name
                long_pieces.append(bool_long)

    group_n = (
        pd.concat(group_n_pieces, ignore_index=True)
        if group_n_pieces
        else pd.DataFrame(columns=["HuisIdBSV", "_group_n", "season"])
    )

    # Reinstate (HH x col x season) combos that the lazy aggregation
    # dropped because count was zero. For wide parquets that means an
    # all-NA column for a household -- which the mapped producer would
    # emit as a count=0 row. Left-join keeps the (HH, col, season)
    # universe; missing stats stay NaN.
    seasons_emitted = [s for s, _ in slice_specs]
    grid = pd.MultiIndex.from_product(
        [sorted(huis_universe), value_cols, seasons_emitted],
        names=["HuisIdBSV", "variable", "season"],
    ).to_frame(index=False)

    if long_pieces:
        long = pd.concat(long_pieces, ignore_index=True)
    else:
        long = pd.DataFrame(
            columns=["HuisIdBSV", "variable", "season",
                     "count", "mean", "std", "min", "max", "median",
                     "p01", "p25", "p75", "p99", "iqr"]
        )

    long = grid.merge(
        long,
        on=["HuisIdBSV", "variable", "season"],
        how="left",
    )
    long["count"] = long["count"].fillna(0).astype("Int64")

    long = long.merge(group_n, on=["HuisIdBSV", "season"], how="left")
    long["_group_n"] = long["_group_n"].fillna(0).astype("Int64")
    long["missing"] = long["_group_n"] - long["count"]
    long["errors"] = long["missing"]
    long = long.drop(columns=["_group_n"])

    long = long.rename(columns={"variable": "column"})
    # HuisIdBSV is the natural key in the wide parquet; cast to the
    # project's nullable Int64 (ADR-005). Identifier mirrors the
    # mapped-mode contract -- the mode-agnostic handle as a string.
    long["HuisIdBSV"] = long["HuisIdBSV"].astype("Int64")
    long["Identifier"] = long["HuisIdBSV"].astype("string")

    # Per-column dtype from the parquet schema (Float64, Boolean, etc.).
    type_map = {name: str(schema[name]) for name in value_cols}
    long["type"] = long["column"].map(type_map).astype("string")

    # Datetime / object stats stay NA: wide-parquet stages carry
    # numeric value columns only; the kernel's datetime min/max and
    # object top5 branches don't apply here.
    long["min_datetime"] = pd.NaT
    long["max_datetime"] = pd.NaT
    long["top5"] = pd.NA

    # source_file is always present in get_data_stats output to keep
    # the column set shape-stable across modes. Populated in raw mode,
    # all-NA otherwise. Wide-parquet stages are mapped-style
    # (HuisIdBSV-keyed), so source_file is all-NA here.
    long["source_file"] = pd.Series(
        [pd.NA] * len(long), index=long.index, dtype="string"
    )

    # Record which quantile mode produced these numbers so downstream
    # consumers (notably cross-stage drift reports) can detect the
    # approximate path and either accept the tolerance or require
    # exact recomputation.
    long["quantile_mode"] = pd.Series(
        [precision] * len(long), index=long.index, dtype="string"
    )

    # Order columns to match get_data_stats output. _cast_stats_dtypes
    # then enforces the shared _STATS_DTYPES contract.
    ordered = [
        "Identifier", "column", "type",
        "count", "missing", "errors",
        "min", "max", "mean", "std", "median", "iqr",
        "p01", "p25", "p75", "p99",
        "min_datetime", "max_datetime", "top5",
        "season", "HuisIdBSV", "source_file",
        "quantile_mode",
    ]
    long = long[[c for c in ordered if c in long.columns]]
    long = _cast_stats_dtypes(long)
    return long
