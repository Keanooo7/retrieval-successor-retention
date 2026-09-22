"""`orchestrator.merge` -- the verification gate and the frozen-diff guard.

R-2026-09-22-night-branch: a run branch merges into `night/<date>` only, and only
with `runs/<id>/verification.json` `ok` and a clean frozen diff. A guard trip is
rsr-manager.md's "Frozen things" rejection -- it stops the night (HALT).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "tests"))

from orchestrator import loopcore as lc  # noqa: E402
from orchestrator import night  # noqa: E402

from _orch_loop_helpers import CONSTANTS_PY, NIGHT, Orch  # noqa: E402
from rsr.exit_codes import Exit  # noqa: E402


@pytest.fixture
def orch(tmp_path, monkeypatch):
    o = Orch(tmp_path, monkeypatch)
    code, msg = night.open_night(o.root, NIGHT, lc.Window())
    assert code == Exit.OK, msg
    o.mgr = night.mgr_worktree(o.root)
    return o


def _run_branch(orch: Orch, files: dict[str, str], item: str = "a") -> str:
    wt = orch.root / ".worktrees" / item
    lc.git(
        orch.root, "worktree", "add", "-b", f"run/{item}", str(wt), orch.base, check=True
    )
    return orch.commit(wt, files, f"run {item}")


def _verify(orch: Orch, status: str = "ok", item: str = "a") -> None:
    orch.commit(
        orch.mgr,
        {f"runs/{item}/verification.json": json.dumps({"status": status})},
        f"verify {item}",
    )


def _night_head(orch: Orch) -> str:
    return orch.git("rev-parse", f"refs/heads/night/{NIGHT}")


def test_merge_is_refused_without_a_verification_record(orch):
    _run_branch(orch, {"runs/a/ledger.json": "{}\n"})
    before = _night_head(orch)
    proc = orch.run("merge", "a")
    assert proc.returncode == 3, proc.stderr
    assert "no verification record" in proc.stderr
    assert _night_head(orch) == before


def test_a_failed_verification_is_refused_as_a_failure(orch):
    _run_branch(orch, {"runs/a/ledger.json": "{}\n"})
    _verify(orch, "failed")
    before = _night_head(orch)
    assert orch.run("merge", "a").returncode == 1
    assert _night_head(orch) == before


def test_a_verified_clean_branch_merges_into_the_night_branch_only(orch):
    main_before = orch.main_refs()
    head = _run_branch(orch, {"runs/a/ledger.json": "{}\n", "src/new.py": "x = 1\n"})
    _verify(orch)
    proc = orch.run("merge", "a")
    assert proc.returncode == 0, proc.stderr
    parents = orch.git("rev-list", "--parents", "-n", "1", _night_head(orch)).split()
    assert len(parents) == 3, "a merge commit, not a fast-forward"
    assert head in parents
    assert orch.main_refs() == main_before
    assert lc.halted(orch.root) is None


def test_the_guard_trips_on_a_preregistration_edit(orch):
    _run_branch(orch, {"preregistration/e3.md": "threshold: 0.50\n"})
    _verify(orch)
    before = _night_head(orch)
    proc = orch.run("merge", "a")
    assert proc.returncode == 1
    assert "preregistration/e3.md" in (lc.halted(orch.root) or "")
    assert _night_head(orch) == before
    assert len(orch.posts()) == 1, "the owner is notified of a guard trip"


def test_the_guard_trips_on_a_frozen_constant_edit(orch):
    edited = CONSTANTS_PY.replace("value=1.0", "value=2.0")
    assert edited != CONSTANTS_PY
    _run_branch(orch, {"src/rsr/constants.py": edited})
    _verify(orch)
    proc = orch.run("merge", "a")
    assert proc.returncode == 1
    assert "FROZEN 'b_max' changed" in (lc.halted(orch.root) or "")


def test_the_guard_does_not_trip_on_a_measured_constant_edit(orch):
    """Specificity: the guard protects FROZEN entries, not the whole file."""
    edited = CONSTANTS_PY.replace('source_experiment="E0e"', 'source_experiment="E1"')
    assert edited != CONSTANTS_PY
    _run_branch(orch, {"src/rsr/constants.py": edited})
    _verify(orch)
    proc = orch.run("merge", "a")
    assert proc.returncode == 0, proc.stderr
    assert lc.halted(orch.root) is None


def test_the_guard_trips_on_an_existing_run_record_but_not_a_new_run_dir(orch):
    _run_branch(orch, {"runs/old/ledger.json": '{"v": 2}\n', "runs/a/x.json": "{}\n"})
    _verify(orch)
    assert orch.run("merge", "a").returncode == 1
    halt = lc.halted(orch.root) or ""
    assert "runs/old/ledger.json" in halt
    assert "runs/a/x.json" not in halt


@pytest.mark.parametrize(
    "path", ["runs/canary/baseline.json", "docs/owner/ruling.md", "docs/spec/spec.md"]
)
def test_the_guard_trips_on_every_other_frozen_path(orch, path):
    _run_branch(orch, {path: "edited\n"})
    _verify(orch)
    assert orch.run("merge", "a").returncode == 1
    assert path in (lc.halted(orch.root) or "")
