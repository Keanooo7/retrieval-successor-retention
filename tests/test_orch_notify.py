"""`orchestrator.notify` -- one pinned `rsr-digest` issue, deduplicated by content.

R-2026-09-22-owner-out-of-loop: the owner hears from the loop only there. The stub
`gh` (RSR_GH_BIN) is the only binary ever invoked.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "tests"))

from _orch_loop_helpers import Orch  # noqa: E402


@pytest.fixture
def orch(tmp_path, monkeypatch):
    return Orch(tmp_path, monkeypatch)


def test_the_first_notice_creates_and_pins_the_issue_then_comments(orch, tmp_path):
    body = tmp_path / "body.md"
    body.write_text("one\n")
    assert orch.run("notify", "first", str(body)).returncode == 0
    body.write_text("two\n")
    assert orch.run("notify", "second", str(body)).returncode == 0
    verbs = [c[:2] for c in orch.calls("gh")]
    assert ["issue", "create"] in verbs
    assert ["issue", "pin"] in verbs
    assert [c[:2] for c in orch.posts()] == [["issue", "create"], ["issue", "comment"]]
    create = next(c for c in orch.calls("gh") if c[:2] == ["issue", "create"])
    assert create[create.index("--label") + 1] == "rsr-digest"


def test_the_same_content_is_posted_once(orch, tmp_path):
    body = tmp_path / "body.md"
    body.write_text("same\n")
    rcs = [orch.run("notify", "t", str(body)).returncode for _ in range(3)]
    assert rcs == [0, 0, 0]
    assert len(orch.posts()) == 1


def test_a_gh_refusal_is_a_failure_and_is_not_recorded_as_sent(orch, tmp_path):
    body = tmp_path / "body.md"
    body.write_text("x\n")
    assert orch.run("notify", "t", str(body), STUB_GH_RC="1").returncode == 1
    assert orch.run("notify", "t", str(body)).returncode == 0
    # the refused attempt stopped at `issue list`; the retry was not deduped away
    assert len(orch.posts()) == 1


def test_a_missing_gh_did_not_run(orch, tmp_path):
    body = tmp_path / "body.md"
    body.write_text("x\n")
    proc = orch.run("notify", "t", str(body), RSR_GH_BIN=str(tmp_path / "no-such-gh"))
    assert proc.returncode == 3
