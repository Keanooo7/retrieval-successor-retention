"""`orchestrator.tick` (and its launchd wrapper `tick.zsh`) -- one pass of the loop.

Every test runs against a throwaway repo with stubbed `claude` and `gh`
(tests/_orch_loop_helpers.py); the number of stub invocations IS the assertion that
nothing launched. R-2026-09-22-night-branch is checked end to end: a whole night --
open, dispatch, return, verify, merge, idle -- leaves `refs/heads/main` (local and
origin) byte-identical.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "tests"))

from orchestrator import loopcore as lc  # noqa: E402

from _orch_loop_helpers import NIGHT, Orch  # noqa: E402

TICK_ZSH = _REPO / "scripts" / "orchestrator" / "tick.zsh"


@pytest.fixture
def orch(tmp_path, monkeypatch):
    return Orch(tmp_path, monkeypatch)


def test_halt_is_respected_before_anything_else(orch):
    lc.halt(orch.root, "owner: stopped by hand")
    orch.set_queue([orch.item("a")])
    proc = orch.run("tick")
    assert proc.returncode == 0, proc.stderr
    assert "HALT present" in proc.stdout
    assert orch.calls("claude") == []
    assert orch.calls("gh") == []
    assert lc.read_night(orch.root) is None, "a halted tick opens no night"


def test_caps_unset_refuse_3_and_notify_once(orch):
    orch.write_env("UNSET", "UNSET")
    orch.set_queue([orch.item("a")])
    rcs = [orch.run("tick").returncode for _ in range(2)]
    assert rcs == [3, 3]
    assert orch.calls("claude") == []
    assert len(orch.posts()) == 1
    assert lc.read_night(orch.root) is None


def test_the_idle_path_launches_nothing_and_notifies_once(orch):
    rcs = [orch.run("tick").returncode for _ in range(2)]
    assert rcs == [0, 0]
    assert orch.calls("claude") == [], "idle must not start a claude session"
    assert len(orch.posts()) == 1, [c[:2] for c in orch.calls("gh")]
    assert lc.read_night(orch.root)["date"] == NIGHT


def test_the_night_spend_cap_halts_before_any_launch(orch):
    orch.write_env("5", "1.0")
    orch.set_queue([orch.item("a")])
    lc.append_jsonl(lc.spend_path(orch.root), {"night": NIGHT, "cost_usd": 1.5})
    proc = orch.run("tick")
    assert proc.returncode == 0, proc.stderr
    assert "RSR_NIGHT_CAP" in (lc.halted(orch.root) or "")
    assert orch.calls("claude") == []


def test_a_manager_cycle_cost_is_recorded_and_reaching_the_cap_halts(orch):
    orch.write_env("5", "1.0")
    p = lc.load_pipeline(orch.root)
    p["awaiting"] = ["x"]  # something for the manager to verify
    lc.save_pipeline(orch.root, p)
    proc = orch.run("tick", STUB_CLAUDE_COST="1.5")
    assert proc.returncode == 0, proc.stderr
    assert len(orch.calls("claude")) == 1
    spend = lc.read_jsonl(lc.spend_path(orch.root))
    assert [(s["kind"], s["cost_usd"]) for s in spend] == [("manager", 1.5)]
    assert "RSR_NIGHT_CAP" in (lc.halted(orch.root) or "")


def test_a_manager_cycle_is_not_repeated_when_nothing_changed(orch):
    """A manager cycle ending is not news: without a researcher or job changing
    state, the next tick inside RSR_MANAGER_INTERVAL_MIN starts no paid cycle."""
    p = lc.load_pipeline(orch.root)
    p["awaiting"] = ["x"]
    lc.save_pipeline(orch.root, p)
    assert [orch.run("tick").returncode for _ in range(2)] == [0, 0]
    assert len(orch.calls("claude")) == 1


def test_dry_run_launches_nothing(orch):
    assert orch.run("night", "open").returncode == 0
    orch.set_queue([orch.item("a")])
    p = lc.load_pipeline(orch.root)
    p["awaiting"] = ["x"]
    lc.save_pipeline(orch.root, p)
    proc = orch.run("tick", DRY_RUN="1")
    assert proc.returncode == 0, proc.stderr
    assert "would launch researcher for a" in proc.stdout
    assert "would run a manager cycle" in proc.stdout
    assert orch.calls("claude") == []
    assert orch.calls("gh") == []
    assert not (orch.root / ".worktrees" / "a").exists()
    assert orch.queue()["a"]["status"] == "ready"
    assert list((orch.root / ".orchestrator" / "cycles").glob("*.json")) == []


def test_dry_run_opens_no_night(orch):
    proc = orch.run("tick", DRY_RUN="1")
    assert proc.returncode == 0, proc.stderr
    assert lc.read_night(orch.root) is None
    assert lc.git(orch.root, "rev-parse", "--verify", f"night/{NIGHT}").returncode != 0


def test_a_whole_night_never_moves_main(orch):
    main_before = orch.main_refs()
    orch.set_queue([orch.item("a")])

    # tick 1: open the night, dispatch `a`, run one manager cycle
    assert orch.run("tick").returncode == 0
    res = orch.wait_result(1)
    assert res["rc"] == 0
    wt = orch.root / ".worktrees" / "a"
    assert orch.git("rev-parse", "--abbrev-ref", "HEAD", cwd=wt) == "run/a"
    run_head = orch.commit(wt, {"runs/a/ledger.json": '{"loss": 1.0}\n'}, "run a")

    # tick 2: the researcher returned -> verifying; the manager verifies it
    assert orch.run("tick").returncode == 0
    assert orch.queue()["a"]["status"] == "verifying"
    mgr = orch.root / ".worktrees" / "_mgr"
    orch.commit(mgr, {"runs/a/verification.json": json.dumps({"status": "ok"})}, "v")

    # tick 3: merged into the night branch; then idle
    assert orch.run("tick").returncode == 0
    assert orch.queue()["a"]["status"] == "merged"
    night_head = orch.git("rev-parse", f"night/{NIGHT}")
    assert orch.git("merge-base", "--is-ancestor", run_head, night_head) == ""
    assert orch.main_refs() == main_before
    kinds = [c["argv"][c["argv"].index("--agent") + 1] for c in orch.calls("claude")]
    # the researcher is detached, so its call may log after the manager's
    assert sorted(kinds) == ["rsr-manager", "rsr-manager", "rsr-researcher"]


# --------------------------------------------------------------------------- #
# tick.zsh -- the launchd entry point
# --------------------------------------------------------------------------- #

needs_zsh = pytest.mark.skipif(
    shutil.which("zsh") is None, reason="zsh not installed (tick.zsh is macOS-only)"
)


def _zsh(**env) -> subprocess.CompletedProcess:
    import os

    return subprocess.run(
        ["zsh", str(TICK_ZSH)],
        capture_output=True,
        text=True,
        env={**os.environ, "RSR_PYTHON": sys.executable, **env},
        check=False,
    )


@needs_zsh
def test_tick_zsh_respects_halt(orch):
    lc.halt(orch.root, "owner: stopped")
    proc = _zsh()
    assert proc.returncode == 0, proc.stderr
    assert "HALT present" in proc.stdout
    assert orch.calls("claude") == []


@needs_zsh
def test_tick_zsh_reports_the_pass_status_unborrowed(orch):
    orch.write_env("UNSET", "UNSET")
    proc = _zsh()
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "tick exited 3" in proc.stdout
