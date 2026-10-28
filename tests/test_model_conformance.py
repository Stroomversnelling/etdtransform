"""The package follows the data model: what it keys on, and what it calls it.

``etdmap/data/model.yaml`` states the model. These checks hold this repo to it
on two counts: no key names the household where the household batch is the
unit, and each name says which thing it means -- in docstrings, in comments
and in messages -- with a column literal belonging in a path on disk rather
than in a sentence.

Separate from ``test_fixture_conformance.py`` because the subject is
different. That file holds the TESTS to how they build data; this one holds the
whole package -- library and tests -- to the model. A wrong name in a docstring
misleads the next reader as effectively as a wrong key misleads the machine,
and a wrong name in a raise misleads whoever is reading the failure.

The failure message says what to write. Note that a CITATION has to sit on one
line to be recognised as one: the checks read line by line, so a quoted phrase
broken over two lines reads as a use rather than a quotation -- which is how
this very docstring first failed its own gate.
"""

import pathlib

import pytest

from etdmap.conformance import KEY_CHECKS, PROSE_CHECKS, scan

REPO = pathlib.Path(__file__).parent.parent
TREES = (REPO / "etdtransform", REPO / "tests")
MODEL = "etdmap/etdmap/data/model.yaml"

#: Files exempt from a check, and why. Every naming count was driven to zero
#: before this gate landed, so the prose entries are empty and meant to stay
#: that way. The keying entries are the two places a household IS the right
#: key, each for a reason the model itself gives.
ALLOWED: dict[str, dict[str, str]] = {
    "household_keyed_grouping": {
        "aggregate.py": (
            "the read chunker groups shard DIRECTORIES, whose path level is "
            "the household, to pick one project per chunk; a chunk is a "
            "grouping for reading and not a unit of analysis, which the code "
            "says where it does it"
        ),
    },
}
for _name in list(PROSE_CHECKS) + list(KEY_CHECKS):
    ALLOWED.setdefault(_name, {})

ENFORCED = tuple(sorted(set(PROSE_CHECKS) | set(KEY_CHECKS)))


def _sources():
    for tree in TREES:
        for path in sorted(tree.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            yield path.name, path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def findings():
    return scan(_sources(), checks=ENFORCED)


@pytest.mark.parametrize("check", ENFORCED)
def test_the_package_follows_the_model(check, findings):
    exempt = ALLOWED[check]
    offending = [f for f in findings[check] if f.path not in exempt]

    assert not offending, (
        f"{check}: {len(offending)} site(s) depart from the model.\n"
        + "\n".join(f"  {f}" for f in offending[:40])
        + f"\n\nThe model is {MODEL}: `prose` for the words, `keys` and "
        f"`entities` for what to key on. If a site's sense is one the model "
        f"allows, add that sense to the model rather than an exemption here "
        f"-- the allowances are derived from it."
    )


def test_every_model_check_is_enforced():
    """A check added to the library is enforced here, not left measuring."""
    assert set(PROSE_CHECKS) | set(KEY_CHECKS) == set(ENFORCED), (
        "etdmap.conformance gained a prose or key check. Add it to ENFORCED "
        "with an empty ALLOWED entry, so it is held from the start rather "
        "than acquiring a backlog first."
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
