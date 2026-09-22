"""`orchestrator.reconcile` -- recorded state against the processes that exist.

Covers the stop rule *"Two failures with the same cause"*
(docs/lab-notes/dispatch-2026-09-21-overnight.md) as a mechanism rather than a
manager's judgement, and a dead job pid becoming `crashed`. Each has a mutation in
scripts/mutation_battery.py (`# --- orchestrator: reconcile ---`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "tests"))

from orchestrator import loopcore as lc  # noqa: E402
from orchestrator import reconcile  # noqa: E402

from _orch_loop_helpers import Orch, dead_pid  # noqa: E402
from rsr.exit_codes import Exit  # noqa: E402


@pytest.fixture
def orch(tmp_path, monkeypatch):
    return Orch(tmp_path, monkeypatch)


def _session(orch: Orch, n: int, item: str, rc: int | None, kind="researcher"):
    lc.write_json(
        lc.cycle_path(orch.root, n),
        {"n": n, "kind": kind, "item": item, "job": None, "pid": dead_pid()},
    )
    if rc is not None:
        from orchestrator.session import cause_of

        lc.write_json(
            lc.result_path(orch.root, n),
            {"n": n, "rc": rc, "cause": cause_of(rc, {}), "launched": False},
        )


def test_a_running_job_whose_pid_is_dead_becomes_crashed_and_its_item_collecting(orch):
    orch.set_queue([orch.item("a", "running")])
    job = orch.root / ".orchestrator" / "jobs" / "a-1.json"
    lc.write_json(
        job, {"job_id": "a-1", "item_id": "a", "status": "running", "pid": dead_pid()}
    )

    assert reconcile.reconcile(orch.root) == Exit.OK

    assert lc.load_json(job, {})["status"] == "crashed"
    assert orch.queue()["a"]["status"] == "collecting"
    assert lc.load_pipeline(orch.root)["collecting"] == {"a-1": "a"}
    # idempotent: a second pass does not re-issue the transition
    n_calls = len(orch.calls("workqueue"))
    assert reconcile.reconcile(orch.root) == Exit.OK
    assert len(orch.calls("workqueue")) == n_calls


def test_a_running_job_whose_pid_is_alive_is_left_running(orch):
    job = orch.root / ".orchestrator" / "jobs" / "a-2.json"
    lc.write_json(job, {"job_id": "a-2", "item_id": "a", "status": "running", "pid": 1})
    reconcile.reconcile(orch.root)
    assert lc.load_json(job, {})["status"] == "running"


def test_a_nonzero_session_sends_the_item_back_to_ready_with_attempt_plus_one(orch):
    orch.set_queue([orch.item("a", "running")])
    wt = orch.root / ".worktrees" / "a"
    wt.mkdir(parents=True)
    (wt / "evidence.txt").write_text("a failed attempt's tree is evidence\n")
    _session(orch, 1, "a", 143)  # SIGTERM, as a shell reports it

    reconcile.reconcile(orch.root)

    assert orch.queue()["a"]["status"] == "ready"
    assert orch.queue()["a"]["attempt"] == 2
    assert (wt / "evidence.txt").exists(), "reconcile must leave worktrees untouched"


def test_two_failures_with_the_same_cause_park_the_item(orch):
    orch.set_queue([orch.item("a", "running")])
    _session(orch, 1, "a", 143)
    reconcile.reconcile(orch.root)
    assert orch.queue()["a"]["status"] == "ready"

    _session(orch, 2, "a", 143)
    reconcile.reconcile(orch.root)

    assert orch.queue()["a"]["status"] == "parked"
    assert "same cause" in lc.load_pipeline(orch.root)["parked"]["a"]


def test_two_failures_with_different_causes_do_not_park(orch):
    orch.set_queue([orch.item("a", "running")])
    _session(orch, 1, "a", 1)
    reconcile.reconcile(orch.root)
    _session(orch, 2, "a", 143)
    reconcile.reconcile(orch.root)
    assert orch.queue()["a"]["status"] == "ready"
    assert orch.queue()["a"]["attempt"] == 3


def test_a_session_that_died_without_a_result_is_a_failure_not_a_pass(orch):
    orch.set_queue([orch.item("a", "running")])
    _session(orch, 1, "a", None)
    reconcile.reconcile(orch.root)
    assert orch.queue()["a"]["status"] == "ready"
    assert lc.load_pipeline(orch.root)["awaiting"] == []


def test_a_clean_session_moves_the_item_to_verifying(orch):
    orch.set_queue([orch.item("a", "running")])
    _session(orch, 1, "a", 0)
    reconcile.reconcile(orch.root)
    assert orch.queue()["a"]["status"] == "verifying"
    assert lc.load_pipeline(orch.root)["awaiting"] == ["a"]


def test_two_manager_failures_with_the_same_cause_halt_the_loop(orch):
    _session(orch, 1, None, 1, kind="manager")
    reconcile.reconcile(orch.root)
    assert lc.halted(orch.root) is None
    _session(orch, 2, None, 1, kind="manager")
    reconcile.reconcile(orch.root)
    assert "manager" in (lc.halted(orch.root) or "")


def test_the_routing_table_carries_inert_as_5():
    """R-2026-09-22-inert-exit-5. `Exit` may not have INERT yet; IF it does, its
    value must be 5 -- asserted through getattr so the test runs either way."""
    assert getattr(Exit, "INERT", 5) == 5
    assert lc.INERT == 5
    assert "INERT" in lc.ROUTING[5]
    assert "retention experiments on that checkpoint refused" in lc.ROUTING[5]
    assert lc.norm_rc(-15) == 143
