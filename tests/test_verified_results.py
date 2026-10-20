"""`docs/results/2026-09-30-verified.md` is the generator's output; every source exists.

The file is generated from ledgers read by pinned commit (`scripts/verified_results.py`).
A hand edit to it, a changed ledger at a pinned commit, or a renamed key must redden
this file -- not pass silently, and not produce a row with a blank in it.

⚠️ The E0d and B5 sources are read from unmerged run branches by SHA (0e51139,
deaeaa6). Those pins are protected by the annotated tags `verified/e0d-0e51139` and
`verified/b5-deaeaa6` on origin, so they stay reachable if a branch is deleted,
squashed or rebased; `test_every_unmerged_pin_is_protected_by_a_tag` checks it. CI
checks out with `fetch-depth: 0`, which fetches the tags.

A pin failure stays a hard fail, not a skip: battery shards are worktrees of this
repo and share its objects, so a missing pin there is a real defect, not the checkout.
"""

from __future__ import annotations

import difflib
import json
import subprocess
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


def test_every_unmerged_pin_is_protected_by_a_tag():
    """A pin not on night/2026-09-30 must be the commit its annotated tag names."""
    for src in vr.SOURCES.values():
        if src.sha == vr.NIGHT[1]:
            continue
        assert src.tag, f"{src.name} is pinned off night with no protecting tag"
        r = subprocess.run(
            [vr._git_exe(), "-C", str(_REPO), "rev-parse", f"{src.tag}^{{commit}}"],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, (src.tag, r.stderr)
        assert r.stdout.strip() == src.sha, (src.tag, r.stdout)
        kind = subprocess.run(
            [vr._git_exe(), "-C", str(_REPO), "cat-file", "-t", src.tag],
            capture_output=True,
            text=True,
        )
        assert kind.stdout.strip() == "tag", f"{src.tag} is not an annotated tag"


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


# --------------------------------------------------------------------------- #
# refusals: a doctored source must be DID NOT RUN (3), never a printed placeholder
# (review M-1 of 7af0f40). Each doctors one file in the generator's read cache.
# --------------------------------------------------------------------------- #

NIGHT = vr.NIGHT[1]
B1_LEDGER = "runs/b1-beta-inertness/ledger.json"
E0H_3000 = "runs/b2-psi-probe/phaseB/e0h-ckpt3000.json"
B0_RESULTS = "experiments/b0-ceilings/RESULTS.md"
B2_RESULTS = "experiments/b2-psi-probe/RESULTS.md"
B2_LINE = "| 1 | ψ̂-U |"


def _doctor(monkeypatch, sha: str, path: str, edit) -> None:
    monkeypatch.setattr(vr, "_CACHE", {})
    vr._CACHE[(sha, path)] = edit(vr.git_show(sha, path))


def _json_edit(fn):
    def edit(text: str) -> str:
        doc = json.loads(text)
        fn(doc)
        return json.dumps(doc)

    return edit


def _row(doc: dict, key: str) -> dict:
    (row,) = [r for r in doc["rows"] if r["key"] == key]
    return row


def _in_b2_section(fn):
    """Apply `fn` to the §2.1 gating section only; the same line recurs elsewhere."""

    def edit(text: str) -> str:
        head, sep, tail = text.partition(vr.B2_DELTA_SECTION)
        section, nxt, rest = tail.partition("\n#")
        return head + sep + fn(section) + nxt + rest

    return edit


def _dup_line(section: str) -> str:
    lines = section.split("\n")
    (i,) = [j for j, ln in enumerate(lines) if B2_LINE in ln]
    return "\n".join([*lines[: i + 1], lines[i], *lines[i + 1 :]])


def _drop_line(section: str) -> str:
    return "\n".join(ln for ln in section.split("\n") if B2_LINE not in ln)


def _no_value(doc: dict) -> None:
    del _row(doc, "N_ok")["value"]


def _null_value(doc: dict) -> None:
    _row(doc, "N_ok")["value"] = None


def _dup_key(doc: dict) -> None:
    doc["rows"].append(dict(_row(doc, "N_ok")))


def _drop_key(doc: dict) -> None:
    doc["rows"] = [r for r in doc["rows"] if r["key"] != "N_ok"]


def _null_top(doc: dict) -> None:
    doc["verdict"]["outcome"] = None


def _null_json(doc: dict) -> None:
    doc["rc"] = None


#: Case ids carry no spaces: the battery cuts a failing node id at its first space.
DOCTORED = {
    "missing-ledger-key": (B1_LEDGER, _json_edit(_drop_key)),
    "duplicated-ledger-key": (B1_LEDGER, _json_edit(_dup_key)),
    "null-row-value": (B1_LEDGER, _json_edit(_null_value)),
    "row-without-a-value-field": (B1_LEDGER, _json_edit(_no_value)),
    "null-top-level-value": (B1_LEDGER, _json_edit(_null_top)),
    "null-value-in-a-committed-json-file": (E0H_3000, _json_edit(_null_json)),
    "scope-quote-absent-from-its-file": (
        B0_RESULTS,
        lambda text: text.replace("**Descriptive only. No gate.**", ""),
    ),
    "B2-table-line-matched-0-times": (B2_RESULTS, _in_b2_section(_drop_line)),
    "B2-table-line-matched-2-times": (B2_RESULTS, _in_b2_section(_dup_line)),
}


@pytest.mark.parametrize("case", sorted(DOCTORED))
def test_a_doctored_source_exits_3(monkeypatch, capsys, case):
    path, edit = DOCTORED[case]
    _doctor(monkeypatch, NIGHT, path, edit)
    assert vr.main(["--check"]) == Exit.DID_NOT_RUN, capsys.readouterr()
    assert "DID NOT RUN" in capsys.readouterr().err
