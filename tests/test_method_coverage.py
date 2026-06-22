"""
Contract guard: every resample/aggregate method the real data model declares
must be implemented by the pipeline.

This is the deterministic, every-run version of how the unimplemented `derive`
resample method was originally caught only by luck (a crash deep in the pandas
resample path). The aggregation config comes from the real etdmodel.csv via
`etdmap.data_model.get_aggregation_config()`; if it ever declares a method that
the pipeline does not handle, this fails immediately and names it -- instead of
surfacing as a `merge(None)`/`Unknown method` crash during a run.

Implemented sets are tied to the dispatch code in
etdtransform/etdtransform/aggregate.py:
  - resample: pandas `resample_variable` (sum/max/avg, derive skipped + derived
    as a post-step) AND DuckDB `_resample_sql_expr` (sum/max/avg) + the derive
    post-step. Intersection = {sum, max, avg, derive}.
  - aggregate: the live DuckDB path `aggregate_project_data_duckdb`
    (avg/sum, derive skipped + derived post-aggregation) = {avg, sum, derive}.
Add a method here only after it is implemented in the path(s) above.
"""
from etdmap.data_model import get_aggregation_config

RESAMPLE_METHODS_IMPLEMENTED = {"sum", "max", "avg", "derive"}
AGGREGATE_METHODS_IMPLEMENTED = {"avg", "sum", "derive"}


def test_resample_methods_are_implemented():
    cfg = get_aggregation_config()
    used = {v["resample_method"] for v in cfg.values()}
    unimplemented = used - RESAMPLE_METHODS_IMPLEMENTED
    assert not unimplemented, (
        f"etdmodel.csv declares resample_method(s) {sorted(unimplemented)} that are not "
        f"implemented in both the pandas (resample_variable) and DuckDB "
        f"(_resample_sql_expr + derive post-step) paths of etdtransform/aggregate.py. "
        f"Implement them in both paths (and add here), or fix the data model."
    )


def test_aggregate_methods_are_implemented():
    cfg = get_aggregation_config()
    used = {v["aggregate_method"] for v in cfg.values()}
    unimplemented = used - AGGREGATE_METHODS_IMPLEMENTED
    assert not unimplemented, (
        f"etdmodel.csv declares aggregate_method(s) {sorted(unimplemented)} that are not "
        f"implemented in the live DuckDB aggregate path (aggregate_project_data_duckdb) "
        f"of etdtransform/aggregate.py. Implement them (and add here), or fix the data model."
    )
