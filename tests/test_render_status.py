"""`scripts/render_status.py`: the research map's data is checked against the ledgers.

The page renders `docs/status/status.json`. Its numbers are only allowed to be ledger
values, equal to the bit, and its evidence paths must exist -- otherwise the map is
one more layer of prose between the ledgers and the reader, which is where every
retracted number on this project was born (RESEARCH-CONTEXT §11).
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

import render_status as rst  # noqa: E402

COMMITTED = json.loads(rst.STATUS.read_text())


def _write(tmp_path: Path, doc: dict) -> Path:
    p = tmp_path / "status.json"
    p.write_text(json.dumps(doc))
    return p


def _first_number(doc: dict) -> dict:
    return next(n for it in doc["items"] for n in it["numbers"])


def test_render_status_the_committed_status_checks():
    assert rst.check(COMMITTED) == []
    assert rst.main(["--check"]) == 0


def test_render_status_carries_the_decisive_and_headroom_numbers():
    """The list cannot be emptied to make --check pass vacuously."""
    keys = {(n["ledger"], n["key"]) for it in COMMITTED["items"] for n in it["numbers"]}
    for k in ("armA.ratio", "armB.ratio", "armC.ratio"):
        assert ("runs/decisive-shuffle/ledger.json", k) in keys
    assert (
        "runs/efeas-synthetic-s003/ledger.json",
        "headroom_oracle_minus_fifo",
    ) in keys


def test_render_status_refuses_a_number_that_disagrees_with_its_ledger(tmp_path):
    doc = copy.deepcopy(COMMITTED)
    n = _first_number(doc)
    n["value"] = n["value"] + 1e-9
    problems = rst.check(doc)
    assert len(problems) == 1 and n["key"] in problems[0]
    assert rst.main(["--check", "--status", str(_write(tmp_path, doc))]) == 1


def test_render_status_flags_a_key_its_ledger_does_not_have():
    doc = copy.deepcopy(COMMITTED)
    _first_number(doc)["key"] = "no.such.key"
    (problem,) = rst.check(doc)
    assert "no.such.key" in problem


def test_render_status_flags_a_missing_evidence_path():
    doc = copy.deepcopy(COMMITTED)
    doc["items"][0]["evidence"].append({"path": "experiments/no-such/RESULTS.md"})
    (problem,) = rst.check(doc)
    assert "experiments/no-such/RESULTS.md" in problem


def test_render_status_flags_an_unknown_status_and_a_section_15_reference():
    doc = copy.deepcopy(COMMITTED)
    doc["items"][0]["status"] = "green"
    doc["items"][1]["what_it_tests"] = "see §15"
    problems = rst.check(doc)
    assert any("'green'" in p for p in problems)
    assert any("forbidden" in p for p in problems)


def test_render_status_missing_status_did_not_run(tmp_path):
    assert rst.main(["--check", "--status", str(tmp_path / "none.json")]) == 3


def test_render_status_emit_js_writes_the_window_global(tmp_path):
    out = tmp_path / "status.js"
    assert rst.main(["--emit-js", str(out)]) == 0
    text = out.read_text()
    assert text.startswith("window.RSR_STATUS = {") and text.endswith("};\n")
    assert json.loads(text[len("window.RSR_STATUS = ") : -2]) == COMMITTED


def test_render_status_refuses_a_number_mismatch_before_emitting_js(tmp_path):
    doc = copy.deepcopy(COMMITTED)
    _first_number(doc)["value"] = -1.0
    out = tmp_path / "status.js"
    rc = rst.main(["--status", str(_write(tmp_path, doc)), "--emit-js", str(out)])
    assert rc == 1
    assert not out.exists()
