"""The canonical chain's output against the committed reference pins.

Three comparisons, all scoped to `test_helpers.compared_columns()` -- the
curated variables plus the flag columns derived from them -- so growth in the
data model outside that subset is invisible here. Each runs independently, so
one failing does not hide the others.

  columns     -- the column SET within the subset
  values      -- row values on the stored 100-row sample
  statistics  -- per-column min / max / null_count over the FULL artifact

This is a shim. The primary assurance is the post-pipeline comparison scripts
and the behavioural tests; what this adds is noticing a value that moved
without anyone predicting it, which no asserted property can do. It stays until
behavioural coverage is ready to carry that on its own.

Regenerating, after reviewing why the output moved:

    python -m pytest tests/test_reference_pins.py --full-suite --regenerate-pins

That flag selects the regeneration and nothing else, so a normal run can never
rewrite the thing it compares against. An artifact with no pin is reported, not
created: adding one enables a comparison nobody decided to make. Pass
ETD_CREATE_PINS=1 alongside the flag to create.
"""

import os

import pandas as pd
import pytest
from conftest import INTERVALS
from test_helpers import (
    PINNED_ARTIFACTS,
    artifact_metadata,
    compared_columns,
    read_pinned_metadata,
    read_pinned_sample,
    row_keys,
    stale_pins,
    statistic_differences,
    write_pins,
)

from etdmap.storage import read_source_frame, resolve_source_path
from etdtransform.impute import read_diffs

pytestmark = [pytest.mark.integration, pytest.mark.pipeline]

#: Which of the chain's output roots each artifact lands in. Stage 3 writes
#: everything into the `sharded` area: the shell repoints
#: `options.aggregate_folder_path` at it for the duration of the call
#: (`impute.py:1527-1530`), so the averaging reports follow the shards there.
#: Stages 5 and 6 write to the folder they were given.
_ROOT = {
    "avg_diffs": "sharded_area",
    "avg_coverage_duty": "sharded_area",
    "avg_window_presence": "sharded_area",
    "household_imputed": "sharded_area",
    "impute_summary_household": "sharded_area",
    "impute_volume_gap_accumulator": "sharded_area",
    "household_calculated": "calculated_root",
    **{f"household_{i}": "resampled" for i in INTERVALS},
    **{f"project_{i}": "resampled" for i in INTERVALS},
}


@pytest.fixture(scope="session")
def produced_frames(full_chain):
    """Every pinned artifact the chain produced, read once.

    Read through `resolve_source_path` and `read_source_frame`, so a hive
    shard tree becomes a frame instead of being skipped -- which is what
    quietly hollowed out the previous pins.
    """
    roots = {
        "agg_root": full_chain["agg_root"],
        "sharded_area": full_chain["sharded_area"],
        # The calculated artifact is its own directory; its parent is the
        # folder an artifact name resolves against.
        "calculated_root": full_chain["calculated"].parent,
        "resampled": full_chain["resampled"],
    }
    frames = {}
    for artifact in PINNED_ARTIFACTS:
        if artifact == "avg_diffs":
            # Partitioned on (project, cadence), not on HuisIdBSV, so the
            # generic shard reader does not recognise the layout. `read_diffs`
            # is declared the one reader for this artifact and handles both
            # the partitioned directory and an older single file.
            frames[artifact] = read_diffs(
                aggregate_folder_path=full_chain["sharded_area"]
            )
            continue
        path = resolve_source_path(roots[_ROOT[artifact]], artifact)
        if not os.path.exists(path):
            continue
        frames[artifact] = read_source_frame(str(path))
    return frames


def _shared(artifact, produced_frames, pinned_names):
    """The compared-subset columns present on both sides."""
    subset = compared_columns()
    produced = set(produced_frames[artifact].columns) & subset
    return sorted(produced & (set(pinned_names) & subset))


def test_the_chain_produced_every_pinned_artifact(produced_frames):
    """A pin for an artifact the run did not produce would otherwise be
    skipped silently, which is how the previous pins stopped covering six of
    them without anyone noticing."""
    missing = [a for a in PINNED_ARTIFACTS if a not in produced_frames]
    assert not missing, f"pinned but not produced: {missing}"


def test_no_pin_names_a_retired_artifact():
    stale = stale_pins()
    assert not stale, f"pin files for artifacts no longer produced: {stale}"


def test_pinned_artifacts_have_the_same_columns(produced_frames):
    """Within the compared subset, the column set must be identical.

    Additions mean the pin is stale and should be regenerated; removals mean a
    stage stopped writing a column, which is a regression. Both fail, and the
    message says which it is.
    """
    failures = []
    for artifact in sorted(produced_frames):
        pinned = read_pinned_metadata(artifact)
        if pinned is None:
            continue
        subset = compared_columns()
        have = set(produced_frames[artifact].columns) & subset
        want = set(pinned["column_details"]) & subset
        added, removed = sorted(have - want), sorted(want - have)
        if added or removed:
            parts = []
            if added:
                parts.append(f"produced but not pinned (regenerate): {added}")
            if removed:
                parts.append(f"pinned but not produced (regression): {removed}")
            failures.append(f"  {artifact}: " + "; ".join(parts))
    assert not failures, (
        "column set differs within the compared subset:\n" + "\n".join(failures)
    )


def test_pinned_sample_values_match(produced_frames):
    """The pinned rows, found in the produced artifact BY KEY.

    Not by position: DuckDB writes partitions in whatever order its threads
    finish, so an artifact's row order moves between runs while its contents do
    not. Re-sampling positionally would compare different rows and report every
    column as changed, which is what it did before this was keyed.
    """
    failures = []
    for artifact in sorted(produced_frames):
        pinned = read_pinned_sample(artifact)
        if pinned is None:
            continue
        produced = produced_frames[artifact]
        keys = row_keys(pinned, produced)
        shared = _shared(artifact, produced_frames, pinned.columns)
        if not shared:
            continue
        assert keys, f"{artifact}: no identifying column to match rows on"
        left = pinned[keys + shared].sort_values(keys).reset_index(drop=True)
        found = produced[keys + shared].merge(pinned[keys], on=keys, how="inner")
        right = found.sort_values(keys).reset_index(drop=True)
        if len(left) != len(right):
            failures.append(
                f"  {artifact}: {len(left)} pinned row(s) matched "
                f"{len(right)} in the produced artifact"
            )
            continue
        differing = []
        for column in shared:
            try:
                pd.testing.assert_series_equal(
                    left[column], right[column], check_exact=False,
                    rtol=1e-10, check_names=False, check_dtype=False,
                )
            except AssertionError:
                differing.append(column)
        if differing:
            failures.append(f"  {artifact}: {differing}")
    assert not failures, "sampled values differ:\n" + "\n".join(failures)


def test_pinned_statistics_match(produced_frames):
    """Per-column min / max / null_count over the full artifact.

    `null_count` exactly; `min` and `max` within a relative tolerance, because
    the project aggregation sums in parallel and float addition is not
    associative, so the last digits of a mean move between runs.
    """
    failures = []
    for artifact in sorted(produced_frames):
        pinned = read_pinned_metadata(artifact)
        if pinned is None:
            continue
        produced = artifact_metadata(produced_frames[artifact])
        shared = (
            set(pinned["column_details"]) & set(produced["column_details"])
            & compared_columns()
        )
        for column, key, want, have in statistic_differences(
            pinned["column_details"], produced["column_details"], shared
        ):
            failures.append(f"  {artifact}.{column}.{key}: {want} != {have}")
    assert not failures, (
        f"{len(failures)} statistic(s) differ:\n" + "\n".join(failures[:40])
    )


@pytest.mark.regenerate
def test_regenerate_the_pins(produced_frames):
    """Writes the reference files. Deselected unless --regenerate-pins."""
    written = write_pins(
        produced_frames, create=bool(os.environ.get("ETD_CREATE_PINS"))
    )
    assert written, "nothing was written"
    for artifact in written:
        assert read_pinned_metadata(artifact) is not None
        assert read_pinned_sample(artifact) is not None
