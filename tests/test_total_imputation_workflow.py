"""The canonical chain, composed end to end.

Every other file in this suite tests one stage. This one tests that they
COMPOSE: each stage accepts the artifact the previous stage wrote, each
artifact a full run is supposed to produce appears, and nothing is lost
between stages -- no household dropped, no project dropped, no derived column
silently absent.

What it deliberately does NOT do is compare against a recorded copy of a
previous run. Three such comparisons used to live here, against
`tests/data/sample_*.parquet` and `tests/data/metadata_*.json`. A pin asserts
"identical to whenever someone last looked", so every deliberate change to the
pipeline reddens it and the pin cannot say which changes were deliberate --
these two had been red since the etdmap flag taxonomy gained
INTERVAL_FROM_DELIVERED_CUMULATIVE, reporting `min 0 != 2048` for a change that
was correct. Per-stage VALUES are asserted in the per-stage files, against
hand-computed expectations on frames those tests build themselves.

The chain here is the one the runner's default path runs: mapped shards ->
diff averages -> impute -> calculated -> resample -> project aggregate. There
is no assembly step and no monolith.
"""
import logging
import os

import pandas as pd
import pytest

import etdmap
import etdmap.data
import etdmap.data_model
import etdmap.index_helpers

import etdtransform
from etdmap.storage import read_source_frame, resolve_source_path
from etdtransform.impute import AVG_DIFFS_ARTIFACT, read_diffs

pytestmark = [pytest.mark.integration, pytest.mark.pipeline]

from conftest import INTERVALS

#: Composites the calculated stage derives for this fixture. Named rather than
#: read from the catalog: the point is that the chain publishes them, and a
#: catalog read here would pass whatever the catalog happened to say.
COMPOSITES = [
    "TerugleveringTotaalNetto",
    "ElektriciteitsgebruikTotaalNetto",
    "ElektriciteitsgebruikTotaalWarmtepomp",
    "ElektriciteitsgebruikTotaalGebouwgebonden",
    "ElektriciteitsgebruikTotaalHuishoudelijk",
    "Zelfgebruik",
    "ElektriciteitsgebruikTotaalBruto",
]

#: Written by the imputation step itself.
IN_RUN_SUMMARIES = (
    "impute_summary_household.parquet",
    "impute_volume_gap_accumulator.parquet",
)

#: What the averaging recorded about its own decisions, written beside the
#: averages into the sharded area.
AVERAGING_REPORTS = (
    "avg_coverage_duty",
    "avg_window_presence",
)

#: Derived later by etdworkflow's post-processing step from the artifacts
#: above. A pipeline run must not produce them -- if it does, two producers
#: exist for one artifact and which one wrote production becomes unknowable.
POSTPROCESSING_ONLY = (
    "household_aggregated_diff.parquet",
    "impute_summary_project.parquet",
    "impute_volume_gap.parquet",
    "impute_volume_household.parquet",
    "impute_volume_project.parquet",
)


@pytest.fixture(scope="session")
def index_df(_include_updated):
    """The fixture dataset's index, with Meenemen populated.

    Through `_include_updated` rather than calling `update_include` against
    whatever the options name, so the index read is the one this session
    generated.
    """
    return etdmap.index_helpers.read_index()[0]


def test_include(index_df):
    # test if indeed ProjectBSVId and column Meenemen are present
    assert 'Meenemen' in index_df.columns, 'column "Meenmen" not in index_df'
    # Since this is a manual check, ensure that all values have been given either
    # true or false and not none
    assert index_df['Meenemen'].notna().all(), 'column Meenemen contains None values'
    assert index_df['Meenemen'].map(lambda x: isinstance(x, bool)).all(), 'column Meenemen has non-boolean values'


def test_project_id(index_df):
    assert index_df['ProjectIdBSV'].notna().all(), 'column ProjectIdBSV has None values'
    assert index_df['ProjectIdBSV'].dtype in [int, 'int64', 'Int64', 'int32'], "Column 'ProjectIdBSV' is not an integer dtype"


@pytest.fixture(scope="session")
def chain_frames(full_chain):
    """Each household artifact in the chain, read once."""
    frames = {
        "imputed": read_source_frame(str(full_chain["imputed"])),
        "calculated": read_source_frame(str(full_chain["calculated"])),
    }
    for interval in INTERVALS:
        frames[interval] = read_source_frame(
            str(full_chain["resampled"] / f"household_{interval}")
        )
    return frames


# ---------------------------------------------------------------------------
# Every artifact the run is supposed to produce
# ---------------------------------------------------------------------------

def test_the_diff_averages_resolve_and_read(full_chain):
    """Imputation's one input artifact. `avg_diffs` is a directory of
    (project, cadence) partitions, so what is asserted is that it resolves and
    reads -- not that a particular layout is on disk."""
    path = resolve_source_path(full_chain["agg_root"], AVG_DIFFS_ARTIFACT)
    assert os.path.exists(path), f"{AVG_DIFFS_ARTIFACT} did not resolve"
    assert not read_diffs().empty


def test_every_stage_wrote_its_household_artifact(full_chain):
    assert full_chain["imputed"].is_dir()
    assert full_chain["calculated"].is_dir()
    for interval in INTERVALS:
        target = full_chain["resampled"] / f"household_{interval}"
        assert target.is_dir(), f"household_{interval} is not a shard directory"


def test_every_interval_got_a_project_aggregate(full_chain):
    # Project aggregates are single files by that writer's own design,
    # whatever shape the household artifacts they read had.
    for interval in INTERVALS:
        target = full_chain["resampled"] / f"project_{interval}.parquet"
        assert target.is_file(), f"project_{interval}.parquet is not a file"


def test_the_run_writes_its_own_summaries(full_chain):
    for name in IN_RUN_SUMMARIES:
        assert (full_chain["sharded_area"] / name).is_file(), f"{name} missing"


def test_the_run_writes_the_averaging_reports(full_chain):
    """What the averaging decided, beside the averages themselves.

    Their writers are covered per stage; what is asserted here is that a full
    run emits them, which a stage test cannot see. A run without them produces
    averages whose eligibility and fill order are no longer recoverable.
    """
    for name in AVERAGING_REPORTS:
        target = resolve_source_path(full_chain["sharded_area"], name)
        assert os.path.exists(target), f"{name} did not resolve from the run"


def test_the_in_run_summary_is_the_slim_one(full_chain):
    """The counts half of the household summary belongs to post-processing.
    Both halves in one file would mean two producers for one artifact."""
    slim = pd.read_parquet(
        full_chain["sharded_area"] / "impute_summary_household.parquet",
        dtype_backend="numpy_nullable",
    )
    assert "diff_col_total" in slim.columns
    assert "missing" not in slim.columns


def test_the_run_does_not_write_the_post_processing_artifacts(full_chain):
    for name in POSTPROCESSING_ONLY:
        for folder in (
            full_chain["agg_root"],
            full_chain["sharded_area"],
            full_chain["resampled"],
        ):
            assert not (folder / name).exists(), (
                f"{name} must come from the post-processing step, not the run"
            )


# ---------------------------------------------------------------------------
# Nothing is lost between stages
# ---------------------------------------------------------------------------

def test_no_household_is_dropped_along_the_chain(chain_frames, index_df):
    """A stage that silently dropped a household would still produce a
    well-formed artifact, so the household set is compared at every step
    rather than only at the end."""
    expected = set(
        index_df.loc[index_df["Meenemen"] == True, "HuisIdBSV"]  # noqa: E712
        .dropna().astype(int).tolist()
    )
    assert expected, "the fixture index includes no households"
    for stage, frame in chain_frames.items():
        got = set(frame["HuisIdBSV"].dropna().astype(int).tolist())
        assert got == expected, (
            f"{stage}: households {sorted(expected - got)} lost, "
            f"{sorted(got - expected)} appeared"
        )


def test_no_project_is_dropped_into_the_project_aggregate(chain_frames, full_chain):
    for interval in INTERVALS:
        households = chain_frames[interval]
        projects = pd.read_parquet(
            full_chain["resampled"] / f"project_{interval}.parquet",
            dtype_backend="numpy_nullable",
        )
        expected = set(households["ProjectIdBSV"].dropna().astype(int).tolist())
        got = set(projects["ProjectIdBSV"].dropna().astype(int).tolist())
        assert got == expected, f"project_{interval}: {expected} -> {got}"


def test_the_calculated_stage_publishes_the_composites(chain_frames):
    missing = [c for c in COMPOSITES if c not in chain_frames["calculated"].columns]
    assert not missing, f"the calculated artifact is missing {missing}"


def test_the_imputation_marks_travel_into_the_calculated_artifact(chain_frames):
    """Imputation writes three columns per imputed variable -- the flags, the
    boolean, and the type. They are how a reader tells a measured cell from a
    filled one, so a stage that dropped them would publish data whose
    provenance is gone while every value still looked right.

    Which variables carry them comes from the imputed artifact rather than
    from a list here, so the check cannot drift from the data model.
    """
    imputed = chain_frames["imputed"]
    calculated = chain_frames["calculated"]
    marks = [
        c for c in imputed.columns
        if c.endswith(("_flags", "_is_imputed", "_impute_type"))
    ]
    assert marks, "imputation wrote no flag, is_imputed or impute_type column"
    missing = [c for c in marks if c not in calculated.columns]
    assert not missing, f"the calculated stage dropped {len(missing)}: {missing[:5]}"


def test_the_curated_variables_survive_into_the_calculated_artifact(chain_frames):
    """The variables the suite judges behaviour on, still there at the end.

    Imputation and the calculated stage share a grain, so a curated variable
    present in the first and absent from the second was dropped. Taken from
    `test_aggregation_columns` rather than from a recorded copy of a previous
    run, so it says a column went missing without also reddening whenever a
    value legitimately moves.
    """
    from etdmap.data_model import test_aggregation_columns

    curated = set(test_aggregation_columns)
    present = curated & set(chain_frames["imputed"].columns)
    assert present, "the imputed artifact carries none of the curated variables"

    missing = sorted(present - set(chain_frames["calculated"].columns))
    assert not missing, f"the calculated stage dropped curated {missing}"


def test_the_curated_variables_reach_every_resampled_artifact(chain_frames):
    """A resample or aggregation that emitted none of them would still produce
    a well-formed artifact, so each one is checked for carrying some.

    Not the whole set: a coarser interval and a project aggregate legitimately
    carry a different column set from the grid, and asserting equality there
    would pin the shape of the resampling rather than test it.
    """
    from etdmap.data_model import test_aggregation_columns

    curated = set(test_aggregation_columns)
    for interval in INTERVALS:
        frame = chain_frames[interval]
        assert curated & set(frame.columns), (
            f"household_{interval} carries none of the curated variables"
        )


# ---------------------------------------------------------------------------
# The grain each stage produces
# ---------------------------------------------------------------------------

def test_the_passthrough_interval_keeps_every_row(chain_frames):
    # 5min is column selection only, so it neither groups nor drops.
    assert len(chain_frames["5min"]) == len(chain_frames["calculated"])


@pytest.mark.parametrize("interval, minutes", [("60min", 60), ("24h", 24 * 60)])
def test_a_grouping_interval_buckets_each_household_separately(
    chain_frames, interval, minutes
):
    """One row per (household, bucket). Computed from the 5min timestamps by
    flooring -- independent of the DuckDB TIME_BUCKET the stage uses, so the
    two agreeing is evidence rather than a tautology.
    """
    five = chain_frames["5min"]
    expected = (
        five.assign(_bucket=five["ReadingDate"].dt.floor(f"{minutes}min"))
        .groupby(["HuisIdBSV", "_bucket"])
        .ngroups
    )
    assert len(chain_frames[interval]) == expected


def test_a_coarser_interval_has_fewer_rows_than_the_grid(chain_frames):
    # The ordering across the whole set, as a guard against an interval being
    # silently produced at the wrong grain.
    counts = [len(chain_frames[i]) for i in INTERVALS]
    assert counts == sorted(counts, reverse=True), (
        f"row counts are not decreasing across {INTERVALS}: {counts}"
    )
    assert counts[-1] > 0, "the coarsest interval produced no rows"
