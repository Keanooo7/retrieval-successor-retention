"""I1 (P1.1, 2026-09-29): the battery never mutates the tree it runs from.

Two failures this file exists to catch, both observed on this project:

* **A stopped battery left a mutated source file in the live checkout** -- twice
  (``loop.py``, ``lanes.py``; memory ``battery-operations``). Every suite now runs
  in a shard worktree; the kill tests below stop the REAL battery with SIGTERM,
  SIGHUP and SIGKILL, and kill its pytest child, and then require the invoking
  tree's ``git status --porcelain`` to be empty and no shard registration to be
  left (after SIGKILL: after the documented ``--prune-shards``).
* **A suite ran on another checkout's venv** (2026-09-27, DIGEST cycle 16b) and
  its green census certified a tree it never imported. The shard's pytest reports
  where ``rsr`` resolved, in-process and in a child; anywhere but the shard is
  DID NOT RUN, never PROVEN or LEAKS.

The end-to-end tests run the real ``mutation_battery.main`` on a stub repository
(``tests/_battery_stub.py``) in a subprocess, so each takes about a second.
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

import battery_isolation as iso  # noqa: E402
import mutation_battery as mb  # noqa: E402

from _battery_stub import build_stub, clean_env  # noqa: E402

STUB = _REPO / "tests" / "_battery_stub.py"


# --------------------------------------------------------------------- helpers


def _porcelain(root: Path) -> str:
    return iso.git(root, "status", "--porcelain").stdout


def _worktrees(root: Path) -> set[Path]:
    return iso._registered(root)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_gone(pid: int, timeout: float = 30.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def stub(tmp_path):
    return build_stub(tmp_path / "stub").resolve()


def _drive(stub: Path, pool: Path, names, *args, mode="good", **popen):
    argv = [
        sys.executable,
        str(STUB),
        "drive",
        str(stub),
        mode,
        ",".join(names),
        "--shard-dir",
        str(pool),
        *args,
    ]
    return argv, popen


def _run(stub, pool, names, *args, mode="good"):
    argv, _ = _drive(stub, pool, names, *args, mode=mode)
    return subprocess.run(
        argv, capture_output=True, text=True, env=clean_env(), timeout=300
    )


def _start(stub, pool, names, marker, *args):
    argv, _ = _drive(stub, pool, names, *args)
    return subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=clean_env(STUB_MARKER=str(marker)),
        start_new_session=True,
    )


def _wait_marker(marker: Path, proc, timeout: float = 120.0) -> int:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if marker.exists():
            return int(marker.read_text())
        if proc.poll() is not None:
            out, err = proc.communicate()
            raise AssertionError(f"battery exited {proc.returncode} early:\n{out}\n{err}")
        time.sleep(0.05)
    proc.kill()
    raise AssertionError("the slow mutation's pytest never started")


# ------------------------------------------------------------- the probe check


_IN = "/shard/src/rsr/__init__.py"
_OUT = "/elsewhere/src/rsr/__init__.py"


@pytest.mark.parametrize(
    ("probe", "bad"),
    [
        ({"in_process": _IN, "child": _IN, "path_python": _IN}, None),
        ({"in_process": _IN, "child": _IN, "path_python": None}, None),
        ({"in_process": _OUT, "child": _IN, "path_python": _IN}, "in_process"),
        ({"in_process": _IN, "child": _OUT, "path_python": _IN}, "child"),
        ({"in_process": _IN, "child": _IN, "path_python": _OUT}, "path_python"),
        ({"in_process": _IN, "child": None, "path_python": _IN}, "child"),
        (None, "no probe"),
    ],
    ids=[
        "all-in",
        "no-path-python",
        "in-process-out",
        "child-out",
        "path-out",
        "child-missing",
        "missing",
    ],
)
def test_the_path_assertion_requires_rsr_under_the_shard(probe, bad):
    got = iso.probe_problem(probe, Path("/shard"))
    if bad is None:
        assert got is None
    else:
        assert got is not None and bad in got


def test_a_prefix_sibling_is_not_under_the_shard():
    assert not iso.is_under("/shard-1/src/rsr/__init__.py", Path("/shard"))


def test_the_suite_env_is_the_shards_own(tmp_path):
    shard = iso.Shard(tmp_path / "s", "0" * 40, 0, tmp_path / "p.json")
    base = {
        "PATH": f"/host/.venv/bin{os.pathsep}/usr/bin",
        "VIRTUAL_ENV": "/host/.venv",
        "RSR_ORCH_ROOT": "/host",
    }
    env = iso.shard_env(shard, base)
    assert env["RSR_ORCH_ROOT"] == str(shard.root)
    assert env["VIRTUAL_ENV"] == str(shard.root / ".venv")
    assert env[iso.PROBE_ENV] == str(shard.probe)
    path = env["PATH"].split(os.pathsep)
    assert path[0] == str(shard.root / ".venv" / "bin")
    assert "/host/.venv/bin" not in path


def test_a_shard_suite_writes_no_bytecode(tmp_path):
    """A same-size mutation restored within the same second passes the .pyc
    mtime+size check; the next suite then runs the previous mutation's code."""
    shard = iso.Shard(tmp_path / "s", "0" * 40, 0, tmp_path / "p.json")
    assert iso.shard_env(shard, {})["PYTHONDONTWRITEBYTECODE"] == "1"


def test_reset_purges_bytecode_outside_the_venv(tmp_path):
    for d in ("src/rsr/__pycache__", ".venv/lib/__pycache__", "tests/__pycache__"):
        (tmp_path / d).mkdir(parents=True)
        (tmp_path / d / "m.cpython-312.pyc").write_bytes(b"")
    iso.purge_bytecode(tmp_path)
    assert not (tmp_path / "src/rsr/__pycache__").exists()
    assert not (tmp_path / "tests/__pycache__").exists()
    assert (tmp_path / ".venv/lib/__pycache__/m.cpython-312.pyc").exists()


def test_a_killed_pytest_is_did_not_run_not_a_failure(tmp_path, monkeypatch):
    shard = iso.Shard(tmp_path, "0" * 40, 0, tmp_path / "p.json")

    def killed(argv, **kw):
        return subprocess.CompletedProcess(argv, -9, "", "")

    monkeypatch.setattr(mb.subprocess, "run", killed)
    with pytest.raises(mb.SuiteDidNotRun, match="signal 9"):
        mb.run_suite(shard)


# ------------------------------------------------------------- pinned, clean


def test_a_dirty_invoking_tree_is_refused(stub):
    (stub / "src" / "rsr" / "calc.py").write_text("VALUE = 9\n")
    with pytest.raises(iso.Unisolated, match="uncommitted"):
        iso.pinned_sha(stub)


def test_an_untracked_file_in_the_invoking_tree_is_refused(stub):
    (stub / "new.py").write_text("")
    with pytest.raises(iso.Unisolated, match="uncommitted"):
        iso.pinned_sha(stub)


def test_a_pool_inside_the_invoking_tree_is_refused(stub):
    sha = iso.pinned_sha(stub)
    with (
        pytest.raises(iso.Unisolated, match="inside the invoking tree"),
        iso.open_pool(stub, sha, pool=stub / "pool"),
    ):
        pass


def test_reset_restores_the_pinned_tree(stub, tmp_path, monkeypatch):
    from _battery_stub import stub_sync

    monkeypatch.setattr(iso, "sync_venv", stub_sync("good"))
    sha = iso.pinned_sha(stub)
    with iso.open_pool(stub, sha, pool=tmp_path / "pool") as (shard,):
        (shard.root / "src" / "rsr" / "calc.py").write_text("VALUE = 9\n")
        (shard.root / "src" / "rsr" / "stray.py").write_text("")
        iso.reset_shard(shard)
        assert _porcelain(shard.root) == ""
        assert (shard.root / ".venv" / "bin" / "python").exists()
    assert _worktrees(stub) == {stub}


# ------------------------------------------------------------- end to end


def test_verdicts_come_from_the_shard_and_the_live_tree_is_untouched(stub, tmp_path):
    out = tmp_path / "rows.json"
    proc = _run(
        stub, tmp_path / "pool", ["proven", "leaks", "adds-nothing"], "--json", str(out)
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    rows = {r["mutation"]: r for r in json.loads(out.read_text())}
    assert {k: r["verdict"] for k, r in rows.items()} == {
        "proven": "PROVEN",
        "leaks": "LEAKS",
        "adds-nothing": "ADDS NOTHING",
    }
    assert _porcelain(stub) == ""
    assert _worktrees(stub) == {stub}


def test_check_still_fails_on_an_unproven_gate(stub, tmp_path):
    proc = _run(stub, tmp_path / "pool", ["proven", "adds-nothing"], "--check")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "1/2 gates proven" in proc.stdout


def test_a_suite_on_another_checkouts_venv_did_not_run(stub, tmp_path):
    """The 09-27 split-brain: the shard's pytest imports the HOST's rsr."""
    proc = _run(stub, tmp_path / "pool", ["proven"], "--check", mode="foreign-pytest")
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "is not under" in proc.stderr
    assert "PROVEN" not in proc.stdout
    assert _porcelain(stub) == ""
    assert _worktrees(stub) == {stub}


def test_a_mutation_whose_suite_reports_no_probe_did_not_run(stub, tmp_path):
    out = tmp_path / "rows.json"
    proc = _run(stub, tmp_path / "pool", ["proven", "no-probe"], "--json", str(out))
    assert proc.returncode == 3, proc.stdout + proc.stderr
    rows = {r["mutation"]: r for r in json.loads(out.read_text())}
    assert rows["proven"]["verdict"] == "PROVEN"
    assert rows["no-probe"]["verdict"] == "DID_NOT_RUN"
    assert "no probe" in rows["no-probe"]["did_not_run"]


# ------------------------------------------------------------- kill tests


@pytest.mark.parametrize(
    "sig", [signal.SIGTERM, signal.SIGHUP], ids=["SIGTERM", "SIGHUP"]
)
def test_a_stopped_battery_leaves_the_live_tree_and_git_clean(stub, tmp_path, sig):
    marker = tmp_path / "pytest.pid"
    proc = _start(stub, tmp_path / "pool", ["proven", "slow"], marker)
    pytest_pid = _wait_marker(marker, proc)
    # mid-mutation: the mutated file is in the shard, not here
    assert _porcelain(stub) == ""
    os.kill(proc.pid, sig)
    out, err = proc.communicate(timeout=120)
    assert proc.returncode == 128 + sig, out + err
    assert f"INTERRUPTED by {sig.name}" in err
    assert _wait_gone(pytest_pid), "the battery's pytest child outlived it"
    assert _porcelain(stub) == ""
    assert _worktrees(stub) == {stub}


def test_a_sigkilled_battery_leaves_only_shards_and_prune_removes_them(stub, tmp_path):
    marker, pool = tmp_path / "pytest.pid", tmp_path / "pool"
    proc = _start(stub, pool, ["slow"], marker)
    pytest_pid = _wait_marker(marker, proc)
    os.kill(proc.pid, signal.SIGKILL)
    proc.communicate(timeout=60)
    assert proc.returncode == -signal.SIGKILL
    # SIGKILL cannot be handled: its orphaned pytest is still asleep in the shard
    os.kill(pytest_pid, signal.SIGKILL)
    assert _wait_gone(pytest_pid)
    assert _porcelain(stub) == ""
    left = _worktrees(stub) - {stub}
    assert left and all(iso.is_under(w, pool) for w in left), left
    prune = _run(stub, pool, [], "--prune-shards")
    assert prune.returncode == 0, prune.stdout + prune.stderr
    assert _worktrees(stub) == {stub}
    assert _porcelain(stub) == ""


def test_a_killed_pytest_child_is_did_not_run_and_the_tree_stays_clean(stub, tmp_path):
    marker, out = tmp_path / "pytest.pid", tmp_path / "rows.json"
    proc = _start(stub, tmp_path / "pool", ["slow", "proven"], marker, "--json", str(out))
    pytest_pid = _wait_marker(marker, proc)
    os.kill(pytest_pid, signal.SIGKILL)
    stdout, stderr = proc.communicate(timeout=120)
    assert proc.returncode == 3, stdout + stderr
    rows = {r["mutation"]: r for r in json.loads(out.read_text())}
    assert rows["slow"]["verdict"] == "DID_NOT_RUN"
    assert "killed by signal 9" in rows["slow"]["did_not_run"]
    # the battery went on, in a reset shard, and the next verdict is real
    assert rows["proven"]["verdict"] == "PROVEN"
    assert _porcelain(stub) == ""
    assert _worktrees(stub) == {stub}
