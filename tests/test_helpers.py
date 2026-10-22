"""The reference pins: writing them, and what they cover.

A pin is a saved copy of an artifact the chain produced, committed under
`tests/data/` and compared against on every run. It is a shim, not the primary
assurance -- the post-pipeline comparison scripts and the behavioural tests are
that -- but it is the only thing that notices a value moving without anyone
having predicted it, so it stays until behavioural coverage is ready.

Two files per artifact:
  sample_<artifact>.parquet  -- 100 rows, random_state=42
  metadata_<artifact>.json   -- num_rows, num_columns, and per column
                                min / max / null_count

Statistics are computed from the FRAME, not from parquet row-group metadata.
Most artifacts are hive shard trees with no single footer, which is what
silently hollowed out the previous version of this file: it resolved each
artifact and then skipped anything that was a directory, so as stages moved to
sharded output the pins quietly stopped covering them.

Despite the name, pytest collects no tests here. Regeneration runs through
test_reference_pins.py, which has the chain fixture.
"""

import json
from pathlib import Path

import pandas as pd

#: The artifacts the chain produces, named without a `.parquet` suffix because
#: most of them are directories (ADR-001: the suffix asserts a shape the name
#: does not carry). `resolve_source_path` turns a name into whichever it is.
PINNED_ARTIFACTS = (
    "avg_diffs",
    "avg_coverage_duty",
    "avg_window_presence",
    "household_imputed",
    "impute_summary_household",
    "impute_volume_gap_accumulator",
    "household_calculated",
    "household_5min",
    "household_15min",
    "household_60min",
    "household_6h",
    "household_24h",
    "project_5min",
    "project_15min",
    "project_60min",
    "project_6h",
    "project_24h",
)

#: Columns the pipeline derives from a variable's name. The flag columns hold
#: what a reader decodes the data with, so a change in them is a change in what
#: the dataset means. Derived by rule so they never become a second list.
_DERIVED_SUFFIXES = ("_flags", "_is_imputed", "_impute_type")

DATA_FOLDER = Path(__file__).parent / "data"


def compared_columns() -> set:
    """The curated variables plus the columns derived from them.

    Scoping the comparison keeps data-model growth outside the subset
    invisible: the pin tests the code, not the breadth of the model.
    """
    from etdmap.data_model import test_aggregation_columns

    return set(test_aggregation_columns) | {
        f"{column}{suffix}"
        for column in test_aggregation_columns
        for suffix in _DERIVED_SUFFIXES
    }


def artifact_metadata(frame: pd.DataFrame) -> dict:
    """Shape and per-column statistics, in the same JSON shape as etdmap's
    pins so the two layers stay comparable by eye.

    Values are stringified, as they are there. `min` and `max` on a column
    holding nothing are None rather than a sentinel, which is what keeps
    "nothing recorded" distinct from "recorded as zero".
    """
    details = {}
    for name in frame.columns:
        column = frame[name]
        present = column.dropna()
        if len(present):
            minimum, maximum = str(present.min()), str(present.max())
        else:
            minimum = maximum = None
        details[name] = {
            "statistics": {
                "min": minimum,
                "max": maximum,
                "null_count": int(column.isna().sum()),
            }
        }
    return {
        "num_rows": int(len(frame)),
        "num_columns": int(len(frame.columns)),
        "column_details": details,
    }


def write_pins(frames: dict, data_folder=None, create: bool = False) -> list:
    """Write the pin pair for each artifact in *frames*.

    Only refreshes a pin that already exists unless *create*: writing one for
    an artifact that has none silently enables a comparison nobody decided to
    make. Returns the artifact names written.
    """
    folder = Path(data_folder or DATA_FOLDER)
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for artifact, frame in sorted(frames.items()):
        sample_path = folder / f"sample_{artifact}.parquet"
        if not create and not sample_path.is_file():
            print(f"skipping {artifact}: no pin yet, and adding one is a "
                  f"decision rather than a regeneration")
            continue
        frame.sample(n=min(100, len(frame)), random_state=42).to_parquet(
            sample_path, engine="pyarrow"
        )
        (folder / f"metadata_{artifact}.json").write_text(
            json.dumps(artifact_metadata(frame), indent=4), encoding="utf-8"
        )
        written.append(artifact)
    return written


def read_pinned_metadata(artifact, data_folder=None):
    path = Path(data_folder or DATA_FOLDER) / f"metadata_{artifact}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_pinned_sample(artifact, data_folder=None):
    """The stored sample, with the row index kept a plain integer.

    The nullable backend applies to a materialised row index too, which comes
    back as Int64. An index is a label and never holds pd.NA, and the sample is
    the only parquet here written from a non-range index (a `.sample()`
    result), so without the cast the two sides differ on index dtype alone.
    """
    path = Path(data_folder or DATA_FOLDER) / f"sample_{artifact}.parquet"
    if not path.is_file():
        return None
    frame = pd.read_parquet(path, dtype_backend="numpy_nullable")
    if str(frame.index.dtype) == "Int64":
        frame.index = frame.index.astype("int64")
    return frame


#: Columns that identify a row, in the order they are tried. An artifact's row
#: ORDER is incidental -- DuckDB writes partitions in whatever order its
#: threads finish -- so the pinned rows are matched by key rather than by
#: position. Matching positionally silently compares different rows.
_KEY_CANDIDATES = (
    "HuisIdBSV", "ProjectIdBSV", "ReadingDate", "column", "cadence_minutes",
)

#: Relative tolerance on a pinned numeric statistic. Project aggregation sums
#: over households in parallel and floating-point addition is not associative,
#: so the last few digits of a mean move between runs. Measured: the drift is
#: at the 1e-15 level, so 1e-10 absorbs it while still catching anything real.
STAT_REL_TOL = 1e-10


def row_keys(pinned, produced) -> list:
    """The identifying columns present on both sides."""
    return [
        c for c in _KEY_CANDIDATES
        if c in pinned.columns and c in produced.columns
    ]


def _numbers_differ(left, right, rel_tol=STAT_REL_TOL) -> bool:
    """Whether two stringified statistics differ beyond float noise.

    None, and anything that is not a number, compares exactly: a min that
    became absent is a real change, not a rounding one.
    """
    if left == right:
        return False
    if left is None or right is None:
        return True
    try:
        a, b = float(left), float(right)
    except (TypeError, ValueError):
        return True
    if a == b:
        return False
    scale = max(abs(a), abs(b))
    return abs(a - b) > rel_tol * scale


def statistic_differences(pinned_stats, produced_stats, columns) -> list:
    """Differing (column, key, pinned, produced) over *columns*.

    `null_count` is compared exactly -- a cell becoming empty is never
    rounding -- and `min` / `max` within STAT_REL_TOL.
    """
    out = []
    for column in sorted(columns):
        want = pinned_stats[column]["statistics"]
        have = produced_stats[column]["statistics"]
        if str(want["null_count"]) != str(have["null_count"]):
            out.append((column, "null_count", want["null_count"],
                        have["null_count"]))
        for key in ("min", "max"):
            if _numbers_differ(want[key], have[key]):
                out.append((column, key, want[key], have[key]))
    return out


def stale_pins(data_folder=None) -> list:
    """Pin files naming an artifact the chain no longer produces."""
    folder = Path(data_folder or DATA_FOLDER)
    if not folder.is_dir():
        return []
    stale = []
    for path in sorted(folder.iterdir()):
        for prefix, suffix in (("sample_", ".parquet"), ("metadata_", ".json")):
            if path.name.startswith(prefix) and path.name.endswith(suffix):
                artifact = path.name[len(prefix):-len(suffix)]
                if artifact not in PINNED_ARTIFACTS:
                    stale.append(path.name)
    return stale
