"""
Contract guard: every resample/aggregate method the real data model declares
must be implemented by the pipeline.

This is the deterministic, every-run version of how the unimplemented `derive`
resample method was originally caught only by luck (a crash deep in a resample
path). The aggregation config comes from the real etdmodel.csv via
`etdmap.data_model.get_aggregation_config()`; if it ever declares a method that
the pipeline does not handle, this fails immediately and names it -- instead of
surfacing as a `merge(None)`/`Unknown method` crash during a run.

Implemented sets are tied to the dispatch code in
etdtransform/etdtransform/aggregate.py:
  - resample: `_resample_sql_expr` (sum/max/avg) plus the `derive` post-step,
    which recomputes a ratio from its components after they are resampled
    = {sum, max, avg, derive}.
  - aggregate: `aggregate_project_data_duckdb` (avg/sum) plus the same derive
    post-step, run after aggregation = {avg, sum, derive}.
Add a method here only after it is implemented in the path(s) above.

This file names those functions but calls neither: what it guards is that the
declared method set and the implemented method set agree, which is a property
of the two sets.
"""
from etdmap.data_model import get_aggregation_config
import pytest

pytestmark = pytest.mark.unit

RESAMPLE_METHODS_IMPLEMENTED = {"sum", "max", "avg", "derive"}
AGGREGATE_METHODS_IMPLEMENTED = {"avg", "sum", "derive"}


def test_resample_methods_are_implemented():
    cfg = get_aggregation_config()
    used = {v["resample_method"] for v in cfg.values()}
    unimplemented = used - RESAMPLE_METHODS_IMPLEMENTED
    assert not unimplemented, (
        f"etdmodel.csv declares resample_method(s) {sorted(unimplemented)} that are not "
        f"implemented in the resample path (_resample_sql_expr + the derive "
        f"post-step) of etdtransform/aggregate.py. "
        f"Implement them there (and add here), or fix the data model."
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
