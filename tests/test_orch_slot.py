"""`scripts/orchestrator/slot.py run`: acquire, run, record, release.

Guards, each with a mutation in `scripts/mutation_battery.py` (block
``# --- orchestrator: slot ---``):

* the child's rc passes through **verbatim** (0, 1, 2, 3, 5, 137) -- mutation:
  collapse it to a bool;
* the thread env is forced to exactly ``slots`` -- mutation: never set it;
* ``--pinned-sha`` refuses a wrong HEAD or a dirty tree outside runs/;
* SIGTERM is forwarded and recorded as ``crashed``.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "tests"))

from orchestrator import lanes  # noqa: E402

from test_orch_lanes import make_root  # noqa: E402

PY = sys.executable


def slot(
    root: Path,
    *args: str,
    cmd: list[str],
    cwd: Path | None = None,
    env_extra: dict | None = None,
    popen: bool = False,
):
    env = {
        **os.environ,
        "RSR_ORCH_ROOT": str(root),
        "PYTHONPATH": str(_REPO / "scripts"),
        **(env_extra or {}),
    }
    argv = [PY, "-m", "orchestrator.slot", "run", *args, "--", *cmd]
    if popen:
        return subprocess.Popen(
            argv,
            env=env,
            cwd=cwd or root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    return subprocess.run(argv, env=env, cwd=cwd or root, capture_output=True, text=True)


def record(root: Path, job_id: str) -> dict:
    return json.loads((root / ".orchestrator" / "jobs" / f"{job_id}.json").read_text())


def _wait_for(pred, timeout=20.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        v = pred()
        if v:
            return v
        time.sleep(0.05)
    raise AssertionError("timed out")


# -------------------------------------------------------------------- rc verbatim


@pytest.mark.parametrize("rc", [0, 1, 2, 3, 5, 137])
def test_the_childs_rc_passes_through_verbatim(tmp_path, rc):
    root = make_root(tmp_path)
    r = slot(
        root, "--lane", "cpu-det", "--job-id", f"rc{rc}", cmd=["sh", "-c", f"exit {rc}"]
    )
    assert r.returncode == rc, r.stderr
    rec = record(root, f"rc{rc}")
    assert rec["rc"] == rc
    assert rec["status"] == "done"


def test_a_child_killed_by_a_signal_exits_128_plus_n(tmp_path):
    root = make_root(tmp_path)
    r = slot(root, "--lane", "cpu-det", "--job-id", "k9", cmd=["sh", "-c", "kill -9 $$"])
    assert r.returncode == 137
    rec = record(root, "k9")
    assert (rec["rc"], rec["signal"], rec["status"]) == (137, 9, "crashed")


def test_the_job_record_has_every_contract_field(tmp_path):
    root = make_root(tmp_path)
    r = slot(root, "--lane", "cpu-det", "--job-id", "J", "--item", "I1", cmd=["true"])
    assert r.returncode == 0, r.stderr
    rec = record(root, "J")
    for k in (
        "job_id",
        "item_id",
        "lane",
        "slots",
        "cmd",
        "cwd",
        "pid",
        "pinned_sha",
        "start",
        "end",
        "rc",
        "status",
    ):
        assert k in rec, k
    assert rec["item_id"] == "I1" and rec["cmd"] == ["true"]


def test_an_existing_job_record_is_not_overwritten(tmp_path):
    root = make_root(tmp_path)
    assert slot(root, "--lane", "cpu-det", "--job-id", "J", cmd=["true"]).returncode == 0
    r = slot(root, "--lane", "cpu-det", "--job-id", "J", cmd=["false"])
    assert r.returncode == 3
    assert record(root, "J")["rc"] == 0


# --------------------------------------------------------------------- thread env


def test_cpu_det_forces_exactly_slots_threads(tmp_path):
    root = make_root(tmp_path, cpu_det_slots=3)
    out = tmp_path / "env.json"
    code = (
        f"import json,os; json.dump({{k: os.environ.get(k) for k in "
        f"{list(lanes.THREAD_ENV_VARS)!r}}}, open({str(out)!r}, 'w'))"
    )
    r = slot(
        root,
        "--lane",
        "cpu-det",
        "--slots",
        "2",
        cmd=[PY, "-c", code],
        env_extra=dict.fromkeys(lanes.THREAD_ENV_VARS, "16"),
    )
    assert r.returncode == 0, r.stderr
    assert json.loads(out.read_text()) == dict.fromkeys(lanes.THREAD_ENV_VARS, "2")


# ---------------------------------------------------------------------- refusals


def test_missing_lanes_file_refuses_3_and_does_not_run(tmp_path):
    marker = tmp_path / "ran"
    r = slot(tmp_path, "--lane", "cpu-det", cmd=["touch", str(marker)])
    assert r.returncode == 3
    assert "C0" in r.stderr
    assert not marker.exists()
    assert not (tmp_path / ".orchestrator" / "jobs").exists() or not list(
        (tmp_path / ".orchestrator" / "jobs").iterdir()
    )


def test_a_full_lane_refuses_3_after_the_wait(tmp_path):
    root = make_root(tmp_path, cpu_det_slots=1)
    cfg = lanes.load_config(root)
    lease = lanes.acquire(root, cfg, "cpu-det", 1)
    try:
        r = slot(root, "--lane", "cpu-det", "--wait", "0.3", cmd=["true"])
        assert r.returncode == 3
        assert "not admitted" in r.stderr
    finally:
        lease.release()


def test_mps_without_a_declared_peak_refuses_3(tmp_path):
    root = make_root(tmp_path)
    r = slot(root, "--lane", "mps", cmd=["true"])
    assert r.returncode == 3
    assert "peak-gb" in r.stderr


def test_a_missing_executable_refuses_3(tmp_path):
    root = make_root(tmp_path)
    r = slot(root, "--lane", "cpu-det", cmd=[str(tmp_path / "no-such-binary")])
    assert r.returncode == 3


# --------------------------------------------------------------------- pinned sha


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        [lanes._git(), "-C", str(cwd), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    _git(r, "config", "user.email", "t@t")
    _git(r, "config", "user.name", "t")
    (r / "a.py").write_text("x = 1\n")
    (r / "runs").mkdir()
    (r / "runs" / ".keep").write_text("")
    _git(r, "add", "-A")
    _git(r, "commit", "-qm", "init")
    return r


def test_pinned_sha_runs_on_a_clean_matching_head(tmp_path, repo):
    root = make_root(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    r = slot(
        root, "--lane", "cpu-det", "--pinned-sha", sha, "--cwd", str(repo), cmd=["true"]
    )
    assert r.returncode == 0, r.stderr


def test_pinned_sha_refuses_a_different_head(tmp_path, repo):
    root = make_root(tmp_path)
    old = _git(repo, "rev-parse", "HEAD")
    (repo / "a.py").write_text("x = 2\n")
    _git(repo, "commit", "-qam", "two")
    marker = tmp_path / "ran"
    r = slot(
        root,
        "--lane",
        "cpu-det",
        "--pinned-sha",
        old,
        "--cwd",
        str(repo),
        cmd=["touch", str(marker)],
    )
    assert r.returncode == 3
    assert not marker.exists()


def test_pinned_sha_refuses_a_dirty_tree(tmp_path, repo):
    root = make_root(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    (repo / "a.py").write_text("x = 3\n")
    r = slot(
        root, "--lane", "cpu-det", "--pinned-sha", sha, "--cwd", str(repo), cmd=["true"]
    )
    assert r.returncode == 3
    assert "changes outside runs/" in r.stderr


def test_pinned_sha_ignores_changes_under_runs(tmp_path, repo):
    root = make_root(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    (repo / "runs" / "out.json").write_text("{}")
    (repo / "runs" / ".keep").write_text("changed")
    r = slot(
        root, "--lane", "cpu-det", "--pinned-sha", sha, "--cwd", str(repo), cmd=["true"]
    )
    assert r.returncode == 0, r.stderr


# ------------------------------------------------------------------------ signals


def test_sigterm_is_forwarded_and_recorded(tmp_path):
    root = make_root(tmp_path)
    p = slot(
        root,
        "--lane",
        "cpu-det",
        "--job-id",
        "T",
        cmd=[PY, "-c", "import time; time.sleep(60)"],
        popen=True,
    )
    jobs = root / ".orchestrator" / "jobs" / "T.json"
    _wait_for(jobs.exists)
    p.send_signal(signal.SIGTERM)
    rc = p.wait(timeout=20)
    assert rc == 128 + signal.SIGTERM
    rec = record(root, "T")
    assert (rec["status"], rec["signal"], rec["rc"]) == ("crashed", 15, 143)


def test_slots_stay_held_while_an_orphaned_child_lives(tmp_path):
    """SIGKILL the wrapper: the child keeps the lock fds, so its cores stay
    reserved; the record is left `running` for the reconciler; the slot frees
    when the child dies."""
    root = make_root(tmp_path, cpu_det_slots=1)
    cfg = lanes.load_config(root)
    p = slot(
        root,
        "--lane",
        "cpu-det",
        "--job-id",
        "O",
        cmd=[PY, "-c", "import time; time.sleep(60)"],
        popen=True,
    )
    jobs = root / ".orchestrator" / "jobs" / "O.json"
    _wait_for(jobs.exists)
    child = record(root, "O")["pid"]
    try:
        p.send_signal(signal.SIGKILL)
        p.wait()
        assert lanes.status(root, cfg)["lanes"]["cpu-det"]["free"] == 0
        assert record(root, "O")["status"] == "running"
    finally:
        os.kill(child, signal.SIGKILL)
    _wait_for(lambda: lanes.status(root, cfg)["lanes"]["cpu-det"]["free"] == 1)
