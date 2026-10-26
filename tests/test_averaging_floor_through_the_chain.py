"""The averaging floor, at its production value, through the real stage.

``test_minimum_household_presence.py`` covers the gate as a function, on frames
it builds. This covers the step the fixture used to prevent: that the stage
reading a mapped dataset applies the same floor, so a project with enough
eligible households gets an average and one below it does not.

The shape is the behaviour here, so this file builds its own dataset rather than
sharing the integration one: two projects at the floor, with one household of
the first excluded through ``Meenemen``, which is how production drops a
household.
"""

import shutil

import pandas as pd
import pytest
from conftest import shard_dir

import etdmap
import etdmap.index_helpers
from etdmap.testing import generate_mapped_dataset
from etdtransform.impute import AVG_DIFFS_ARTIFACT, prepare_diff_averages_sharded
from etdtransform.presence import MIN_ELIGIBLE_HOUSEHOLDS

pytestmark = [pytest.mark.integration, pytest.mark.pipeline]

#: Both projects start at the floor, so the one that keeps every household is
#: averageable and the one that loses a household is one short of it.
AT_THE_FLOOR = MIN_ELIGIBLE_HOUSEHOLDS
SHORT_PROJECT = 1
FULL_PROJECT = 2


@pytest.fixture(scope="module")
def averaged(tmp_path_factory):
    """`avg_diffs`, from a dataset where one project is one household short.

    Module-scoped: the chain is run once and both verdicts are read from it,
    rather than generating a dataset per assertion.
    """
    config_dir = tmp_path_factory.mktemp("floor")
    mapped = generate_mapped_dataset(
        config_dir / "mapped",
        bsv_metadata_file=_metadata_path(),
        households_per_project=AT_THE_FLOOR,
        projects=2,
        raw_folder=config_dir / "raw",
    )

    # Exclude one household of the short project, the way production does.
    index_path = mapped / "index.parquet"
    index_df = pd.read_parquet(index_path, dtype_backend="numpy_nullable")
    index_df["Meenemen"] = pd.array([True] * len(index_df), dtype="boolean")
    dropped = int(
        index_df.loc[index_df["ProjectIdBSV"] == SHORT_PROJECT, "HuisIdBSV"].iloc[0]
    )
    index_df.loc[index_df["HuisIdBSV"] == dropped, "Meenemen"] = False
    index_df.to_parquet(index_path, engine="pyarrow")

    sharded = config_dir / "sharded_mapped"
    for source in sorted(mapped.glob("household_*_table.parquet")):
        household_id = int(source.stem.split("_")[1])
        target = shard_dir(sharded / "sharded", household_id)
        target.mkdir(parents=True)
        shutil.copyfile(source, target / "part.parquet")
    shutil.copyfile(index_path, sharded / "index.parquet")

    aggregate = config_dir / "aggregate"
    previous = etdmap.options.mapped_folder_path
    etdmap.options.mapped_folder_path = sharded
    try:
        prepare_diff_averages_sharded(
            mapped_folder_path=sharded,
            aggregate_folder_path=aggregate,
            cumulative_columns=["Gasgebruik"],
        )
    finally:
        etdmap.options.mapped_folder_path = previous

    from etdtransform.impute import read_diffs

    return read_diffs(aggregate_folder_path=aggregate), dropped


def _metadata_path():
    """The BSV metadata the generator starts from, for the Meenemen stamp."""
    from conftest import config

    return config["etdmap_configuration"]["bsv_metadata_file"]


def _average_column(frame):
    averages = [c for c in frame.columns if c.endswith("_avg")]
    assert averages, f"no average column in {sorted(frame.columns)}"
    return averages[0]


class TestTheFloorDecidesWhetherAnAverageIsPublished:
    def test_the_project_at_the_floor_is_averaged(self, averaged):
        frame, _ = averaged
        column = _average_column(frame)
        rows = frame[frame["ProjectIdBSV"] == FULL_PROJECT]

        assert not rows.empty, f"project {FULL_PROJECT} has no rows at all"
        assert rows[column].notna().any(), (
            f"project {FULL_PROJECT} has {AT_THE_FLOOR} eligible households, "
            f"which meets the floor of {MIN_ELIGIBLE_HOUSEHOLDS}, so its "
            f"average must be published"
        )

    def test_the_project_below_the_floor_is_not(self, averaged):
        frame, dropped = averaged
        column = _average_column(frame)
        rows = frame[frame["ProjectIdBSV"] == SHORT_PROJECT]

        assert rows[column].isna().all(), (
            f"project {SHORT_PROJECT} lost household {dropped}, leaving "
            f"{AT_THE_FLOOR - 1} eligible against a floor of "
            f"{MIN_ELIGIBLE_HOUSEHOLDS}, so it must publish no average"
        )

    def test_the_two_projects_disagree(self, averaged):
        """Guards the pair against both verdicts collapsing to one value, which
        a bug in either direction would otherwise satisfy one assertion of."""
        frame, _ = averaged
        column = _average_column(frame)
        published = {
            int(project): frame.loc[
                frame["ProjectIdBSV"] == project, column
            ].notna().any()
            for project in (SHORT_PROJECT, FULL_PROJECT)
        }

        assert published == {SHORT_PROJECT: False, FULL_PROJECT: True}
