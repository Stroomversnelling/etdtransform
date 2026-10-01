"""A memory chunk never holds households from two projects.

The fill reads a project average and reports per project, so a chunk holding
halves of two projects holds two unrelated populations. The cap still bounds
memory within each project.
"""

import pytest

from etdtransform.aggregate import _chunk_ids_by_project

pytestmark = pytest.mark.unit

PROJECTS = {1: 10, 2: 10, 3: 10, 4: 20, 5: 20, 6: 20}


class TestChunksFollowProjects:
    def test_no_chunk_straddles_a_project(self):
        chunks = _chunk_ids_by_project([1, 4, 2, 5, 3, 6], PROJECTS, chunk_size=4)
        for chunk in chunks:
            assert len({PROJECTS[h] for h in chunk}) == 1, (
                f"chunk {chunk} mixes projects"
            )

    def test_the_cap_still_holds_inside_a_project(self):
        chunks = _chunk_ids_by_project([1, 2, 3], {1: 10, 2: 10, 3: 10}, chunk_size=2)
        assert [len(c) for c in chunks] == [2, 1]

    def test_every_household_lands_exactly_once(self):
        all_ids = [6, 3, 1, 5, 2, 4]
        chunks = _chunk_ids_by_project(all_ids, PROJECTS, chunk_size=2)
        seen = [h for chunk in chunks for h in chunk]
        assert sorted(seen) == sorted(all_ids)

    def test_an_unknown_project_is_kept_in_its_own_group_last(self):
        chunks = _chunk_ids_by_project([1, 2, 99], {1: 10, 2: 10}, chunk_size=4)
        assert chunks == [[1, 2], [99]]

    def test_input_order_does_not_change_the_chunks(self):
        forward = _chunk_ids_by_project([1, 2, 3, 4, 5, 6], PROJECTS, chunk_size=2)
        shuffled = _chunk_ids_by_project([5, 1, 6, 2, 4, 3], PROJECTS, chunk_size=2)
        assert forward == shuffled

    def test_no_map_at_all_degrades_to_plain_windows(self):
        # The legacy shell passes an empty map when the source has no project
        # column; chunking then only bounds memory.
        chunks = _chunk_ids_by_project([1, 2, 3], {}, chunk_size=2)
        assert chunks == [[1, 2], [3]]
