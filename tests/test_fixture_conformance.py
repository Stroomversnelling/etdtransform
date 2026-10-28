"""Every test in this tree builds its data from the shared builders.

A test that assembles a schema or a dtype by hand encodes a model of the data
rather than exercising the one the library has, and the failure modes are
quiet: an omitted sidecar column reads as "nothing recorded" instead of
raising, a numpy dtype cannot hold pd.NA, and a duplicate dict key drops a
column without a word.

The checks live in ``etdmap.conformance`` so this file, its counterpart in
etdmap and ``etdworkflow/loc_metric.py`` cannot drift apart.

What to do when one of these fails is in etdmap's ``docs/WRITING_TESTS.md``.
"""

import pathlib

import pytest

from etdmap.conformance import CHECKS, scan

pytestmark = pytest.mark.unit

TESTS_DIR = pathlib.Path(__file__).parent
GUIDE = "etdmap/docs/WRITING_TESTS.md"

#: Files exempt from a check, and why. A test whose subject IS the malformed
#: shape a guard must reject belongs here, named, with the reason -- not
#: silently tolerated. Keyed on the check name, then the file name.
ALLOWED: dict[str, dict[str, str]] = {
    "id_columns_in_construction": {},
    "sidecar_schema_restated": {},
    "numpy_dtypes_in_construction": {},
    "duplicate_dict_keys": {},
}

#: The checks this tree is held to.
ENFORCED = tuple(ALLOWED)


def _sources():
    for path in sorted(TESTS_DIR.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path.name, path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def findings():
    return scan(_sources(), checks=ENFORCED)


@pytest.mark.parametrize("check", ENFORCED)
def test_the_tree_conforms(check, findings):
    """No test in this tree hand-builds what a shared builder owns."""
    exempt = ALLOWED[check]
    offending = [f for f in findings[check] if f.path not in exempt]

    assert not offending, (
        f"{check}: {len(offending)} site(s) build data the shared builders "
        f"own.\n"
        + "\n".join(f"  {f}" for f in offending)
        + f"\n\nSee {GUIDE}. If a site's subject IS this shape, add its file "
        f"to ALLOWED['{check}'] with the reason."
    )


def test_every_check_is_enforced():
    """A check added to the library is enforced here, not silently ignored."""
    assert set(CHECKS) == set(ENFORCED), (
        "etdmap.conformance.CHECKS has changed. Add the new check to ENFORCED "
        "with an empty ALLOWED entry, so it is held from the start rather than "
        "acquiring a backlog first."
    )


def test_an_exemption_names_a_file_that_exists():
    """An exemption outlives the file it was written for otherwise."""
    names = {name for name, _ in _sources()}
    stale = {
        (check, path)
        for check, exempt in ALLOWED.items()
        for path in exempt
        if path not in names
    }
    assert not stale, f"ALLOWED names file(s) that no longer exist: {stale}"
