"""`orchestrator.dispatch` and `orchestrator.submit` -- launching work, detached.

The researcher's prompt is a verb and a path (rsr-manager.md: never a paraphrase
of the brief); a brief with lint findings is not dispatched; long compute is
submitted only from a clean tree at a pinned sha.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "tests"))

from orchestrator import loopcore as lc  # noqa: E402
from orchestrator import night  # noqa: E402

from _orch_loop_helpers import NIGHT, Orch  # noqa: E402
from rsr.exit_codes import Exit  # noqa: E402


@pytest.fixture
def orch(tmp_path, monkeypatch):
    o = Orch(tmp_path, monkeypatch)
    code, msg = night.open_night(o.root, NIGHT, lc.Window())
    assert code == Exit.OK, msg
    o.set_queue([o.item("a")])
    return o


def test_a_brief_with_lint_findings_is_not_dispatched(orch):
    proc = orch.run("dispatch", "a", STUB_LINT_RC="1")
    assert proc.returncode == 3
    assert "lint_brief" in proc.stderr
    assert orch.calls("claude") == []
    assert not (orch.root / ".worktrees" / "a").exists()
    assert orch.calls("lint_brief") == [
        ["docs/lab-notes/dispatch-a.md", "--base", orch.base]
    ]


def test_an_item_not_in_the_ready_set_is_refused(orch):
    assert orch.run("dispatch", "zzz").returncode == 3
    assert orch.calls("claude") == []


def test_dispatch_launches_the_researcher_in_its_own_worktree_at_the_base(orch):
    proc = orch.run("dispatch", "a")
    assert proc.returncode == 0, proc.stderr
    res = orch.wait_result(1)
    assert res["rc"] == 0
    wt = orch.root / ".worktrees" / "a"
    assert orch.git("rev-parse", "HEAD", cwd=wt) == orch.base
    assert orch.git("rev-parse", "--abbrev-ref", "HEAD", cwd=wt) == "run/a"
    (call,) = orch.calls("claude")
    assert call["argv"] == [
        "-p", "Execute docs/lab-notes/dispatch-a.md",
        "--agent", "rsr-researcher",
        "--settings", ".claude/settings.researcher.json",
        "--permission-mode", "dontAsk",
        "--output-format", "json",
        "--max-budget-usd", "5",
    ]  # fmt: skip
    assert Path(call["cwd"]).resolve() == wt.resolve()
    assert orch.queue()["a"]["status"] == "running"
    spend = lc.read_jsonl(lc.spend_path(orch.root))
    assert [(s["kind"], s["cost_usd"]) for s in spend] == [("researcher", 0.25)]


def _worktree(orch: Orch) -> Path:
    wt = orch.root / ".worktrees" / "a"
    lc.git(orch.root, "worktree", "add", "-b", "run/a", str(wt), orch.base, check=True)
    return wt


def _submit(orch: Orch, wt: Path, **env):
    import os
    import subprocess

    return subprocess.run(
        [
            sys.executable, "-m", "orchestrator.submit",
            "--item", "a", "--lane", "cpu", "--slots", "1", "--job-id", "a-job",
            "--", sys.executable, "-c", "print('trained')",
        ],
        cwd=wt, capture_output=True, text=True, env={**os.environ, **env}, check=False,
    )  # fmt: skip


def test_submit_refuses_a_dirty_tree(orch):
    wt = _worktree(orch)
    (wt / "uncommitted.py").write_text("x = 1\n")
    proc = _submit(orch, wt)
    assert proc.returncode == 3
    assert "not clean" in proc.stderr
    assert orch.calls("slot") == []


def test_submit_refuses_when_uv_sync_frozen_fails(orch):
    wt = _worktree(orch)
    proc = _submit(orch, wt, STUB_UV_RC="1")
    assert proc.returncode == 3
    assert orch.calls("slot") == []


def test_submit_launches_the_slot_detached_at_the_pinned_sha(orch):
    wt = _worktree(orch)
    head = orch.commit(wt, {"runs/a/manifest.json": "{}\n"}, "manifest")
    proc = _submit(orch, wt)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "a-job"
    rec = orch.wait_job("a-job")
    assert rec["status"] == "done"
    assert rec["rc"] == 0
    assert rec["pinned_sha"] == head
    (sub,) = lc.read_jsonl(orch.root / ".orchestrator" / "state" / "submitted.jsonl")
    assert sub["pinned_sha"] == head
