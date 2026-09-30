"""`docs/results/2026-09-30-verified.md` is the generator's output; every source exists.

The file is generated from ledgers read by pinned commit (`scripts/verified_results.py`).
A hand edit to it, a changed ledger at a pinned commit, or a renamed key must redden
this file -- not pass silently, and not produce a row with a blank in it.

⚠️ The E0d and B5 sources are read from `run/e0d` and `run/b5-convergence` by SHA.
A clone without those commits cannot build the table: the generator exits 3 and
these tests FAIL (never skip). CI checks out with `fetch-depth: 0`, which fetches
them while the branches exist or once they are merged.

A pin failure stays a hard fail, not a skip: battery shards are worktrees of this
repo and share its objects, so a missing pin there is a real defect, not the checkout.
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

import verified_results as vr  # noqa: E402

from rsr.exit_codes import Exit  # noqa: E402


@pytest.fixture(scope="module")
def generated() -> str:
    return vr.render()


def test_the_committed_file_is_the_generators_output(generated):
    """An empty diff. The committed Markdown is regenerated and compared byte for byte."""
    committed = (_REPO / vr.OUT).read_text()
    diff = "".join(
        difflib.unified_diff(
            committed.splitlines(keepends=True),
            generated.splitlines(keepends=True),
            fromfile=f"committed {vr.OUT}",
            tofile="scripts/verified_results.py",
        )
    )
    assert diff == "", "regenerate with --write, or find what moved:\n" + diff[:4000]


def test_two_runs_are_byte_identical(generated):
    """No timestamp, no dict-order or float-repr drift: a second render is identical."""
    vr._CACHE.clear()
    assert vr.render() == generated


def test_every_rows_source_key_exists():
    """Each ref of each row resolves at its pinned commit to a non-null value, and each
    ledger ref is a row key of that ledger. `resolve` raises `Missing` otherwise."""
    rows = vr.build_rows()
    assert rows, "the table has no rows"
    n = 0
    for row in rows:
        assert row.refs, row.claim
        assert row.values.strip(), row.claim
        for ref in row.refs:
            assert vr.resolve(ref) is not None, (row.claim, ref)
            if ref.kind == "row":
                src = vr.SOURCES[ref.source]
                keys = {r["key"] for r in vr._json(src.sha, ref.file)["rows"]}
                assert ref.key in keys, (row.claim, ref.key)
            n += 1
    assert n >= len(rows)


def test_every_source_has_a_valid_verification_record():
    """The verifier column is computed, not typed: a record that fails
    `orchestrator.verify.validate_record` must not print CONFIRMED."""
    for name in vr.SOURCES:
        v = vr.verification(name)
        assert v.problems == [], (name, v.problems)
        assert v.verdict.startswith("CONFIRMED"), (name, v.verdict)


def test_every_scope_quote_is_in_its_file():
    for source, path, quote in vr.SCOPE:
        assert vr.scope_line(source, path, quote) == quote


def test_a_missing_source_exits_3(monkeypatch):
    """A source that cannot be read is DID NOT RUN, never a table with a hole in it."""
    bad = vr.Source(
        "E0d", "e0d", "run/e0d", "0" * 40, "run/e0d", "experiments/e0d/RESULTS.md"
    )
    monkeypatch.setitem(vr.SOURCES, "E0d", bad)
    monkeypatch.setattr(vr, "_CACHE", {})
    assert vr.main(["--check"]) == Exit.DID_NOT_RUN


def test_a_hand_edited_file_exits_1(monkeypatch, tmp_path, generated):
    edited = tmp_path / "verified.md"
    edited.write_text(generated.replace("AGREE", "AGREED", 1))
    monkeypatch.setattr(vr, "OUT", str(edited))
    assert vr.main(["--check"]) == Exit.FAIL
    edited.write_text(generated)
    assert vr.main(["--check"]) == Exit.OK
