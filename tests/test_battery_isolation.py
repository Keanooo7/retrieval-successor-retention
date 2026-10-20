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


def _wait_marker(marker: Path, proc, timeout: float = 120.0) -> tuple[int, int]:
    """``(pytest pid, grandchild pid)`` once the slow mutation's test is running."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if marker.exists():
            a, b = marker.read_text().split()
            return int(a), int(b)
        if proc.poll() is not None:
            out, err = proc.communicate()
            raise AssertionError(f"battery exited {proc.returncode} early:\n{out}\n{err}")
        time.sleep(0.05)
    proc.kill()
    raise AssertionError("the slow mutation's pytest never started")


# ------------------------------------------------------------- the probe check


_IN = "/shard/src/rsr/__init__.py"
_OUT = "/elsewhere/src/rsr/__init__.py"
_OK = {"in_process": _IN, "child": _IN, "path_python": _IN, "exitstatus": 0}


@pytest.mark.parametrize(
    ("probe", "bad"),
    [
        ({**_OK, "path_python": _IN}, None),
        ({**_OK, "path_python": None}, None),
        ({**_OK, "exitstatus": 1}, None),
        ({**_OK, "in_process": _OUT}, "in_process"),
        ({**_OK, "child": _OUT}, "child"),
        ({**_OK, "path_python": _OUT}, "path_python"),
        ({**_OK, "child": None}, "child"),
        ({**_OK, "exitstatus": 2}, "exit status was 2"),
        ({**_OK, "exitstatus": 3}, "exit status was 3"),
        ({**_OK, "exitstatus": None}, "exit status was None"),
        (None, "no probe"),
    ],
    ids=[
        "all-in",
        "no-path-python",
        "some-failed",
        "in-process-out",
        "child-out",
        "path-out",
        "child-missing",
        "exit-interrupted",
        "exit-internal",
        "exit-missing",
        "missing",
    ],
)
def test_the_path_assertion_requires_rsr_under_the_shard(probe, bad):
    got = iso.probe_problem(probe, Path("/shard"))
    if bad is None:
        assert got is None
    else:
        assert got is not None and bad in got


@pytest.mark.parametrize(
    "rc", [2, 3, 4, 5], ids=["interrupted", "internal", "usage", "no-tests"]
)
def test_a_suite_that_did_not_complete_did_not_run(rc):
    """Review MAJOR-1: pytest's process status alone, with a clean probe."""
    got = iso.suite_problem(rc, _OK, Path("/shard"))
    assert got is not None and f"exited {rc}" in got


def test_a_completed_suite_is_judged_by_its_probe():
    assert iso.suite_problem(1, _OK, Path("/shard")) is None
    assert "no probe" in iso.suite_problem(0, None, Path("/shard"))


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
    assert env["RSR_ORCH_ROOT"] == str(shard.root)
    assert env["VIRTUAL_ENV"] == str(shard.root / ".venv")
    assert env[iso.PROBE_ENV] == str(shard.probe)
    path = env["PATH"].split(os.pathsep)
    assert path[0] == str(shard.root / ".venv" / "bin")
    assert "/host/.venv/bin" not in path


def test_the_suite_never_inherits_pythonpath_or_a_pycache_prefix(tmp_path):
    """Review MINOR-1: another checkout's scripts/ via PYTHONPATH passes the rsr
    probe; a pycache prefix puts bytecode where the reset cannot reach it."""
    shard = iso.Shard(tmp_path / "s", "0" * 40, 0, tmp_path / "p.json")
    env = iso.shard_env(
        shard, {"PYTHONPATH": "/host/scripts", "PYTHONPYCACHEPREFIX": "/tmp/pyc"}
    )
    assert "PYTHONPATH" not in env and "PYTHONPYCACHEPREFIX" not in env


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

    def killed(argv, cwd, env):
        return subprocess.CompletedProcess(argv, -9, "", "")

    monkeypatch.setattr(mb, "_spawn", killed)
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


def test_a_kept_shard_is_reused_at_the_new_pinned_sha(stub, tmp_path, monkeypatch):
    from _battery_stub import CALC, stub_sync

    monkeypatch.setattr(iso, "sync_venv", stub_sync("good"))
    pool = tmp_path / "pool"
    with iso.open_pool(stub, iso.pinned_sha(stub), pool=pool, keep=True) as (s1,):
        (s1.root / "runs").mkdir()
        (s1.root / "runs" / "state.json").write_text("{}")  # ignored: runs/
        (s1.root / "src" / "rsr" / "__pycache__").mkdir()
        (s1.root / "src" / "rsr" / "calc.py").write_text("VALUE = 9\n")
    assert _worktrees(stub) == {stub, s1.root}
    (stub / "src" / "rsr" / "calc.py").write_text(CALC + "# v2\n")
    iso.git(stub, "add", "-A")
    iso.git(stub, "-c", "user.email=s@r", "-c", "user.name=s", "commit", "-qm", "v2")
    sha2 = iso.pinned_sha(stub)
    with iso.open_pool(stub, sha2, pool=pool) as (s2,):
        assert s2.root == s1.root
        assert iso.git(s2.root, "rev-parse", "HEAD").stdout.strip() == sha2
        assert _porcelain(s2.root) == ""
        assert not (s2.root / "src" / "rsr" / "__pycache__").exists()
        assert not (s2.root / "runs").exists()
    assert _worktrees(stub) == {stub}


def test_a_directory_in_the_pool_that_is_not_our_worktree_is_refused(stub, tmp_path):
    pool = tmp_path / "pool"
    (pool / "shard-0").mkdir(parents=True)
    (pool / "shard-0" / "keep.txt").write_text("someone else's")
    with (
        pytest.raises(iso.Unisolated, match="is not a worktree of"),
        iso.open_pool(stub, iso.pinned_sha(stub), pool=pool),
    ):
        pass
    assert (pool / "shard-0" / "keep.txt").read_text() == "someone else's"


def test_two_batteries_never_share_a_pool(stub, tmp_path, monkeypatch):
    from _battery_stub import stub_sync

    monkeypatch.setattr(iso, "sync_venv", stub_sync("good"))
    pool, sha = tmp_path / "pool", iso.pinned_sha(stub)
    with (
        iso.open_pool(stub, sha, pool=pool),
        pytest.raises(iso.Unisolated, match="another battery holds"),
        iso.open_pool(stub, sha, pool=pool),
    ):
        pass
    assert _worktrees(stub) == {stub}


def test_an_ignored_file_left_in_a_shard_is_not_clean(stub, tmp_path, monkeypatch):
    """Review MAJOR-2: `git status --porcelain` cannot see an ignored runs/ file."""
    from _battery_stub import stub_sync

    monkeypatch.setattr(iso, "sync_venv", stub_sync("good"))
    with iso.open_pool(stub, iso.pinned_sha(stub), pool=tmp_path / "pool") as (sh,):
        (sh.root / "runs").mkdir()
        (sh.root / "runs" / "state.json").write_text("{}")
        with pytest.raises(iso.Unisolated, match="runs/"):
            iso.verify_clean(sh)
        iso.reset_shard(sh)
        assert not (sh.root / "runs").exists()
        assert (sh.root / ".venv" / "bin" / "python").exists()


def test_prune_refuses_while_a_battery_holds_the_pool(stub, tmp_path, monkeypatch):
    """Review MAJOR-3: the documented cleanup must never remove a live shard."""
    from _battery_stub import stub_sync

    monkeypatch.setattr(iso, "sync_venv", stub_sync("good"))
    pool = tmp_path / "pool"
    with iso.open_pool(stub, iso.pinned_sha(stub), pool=pool) as (sh,):
        prune = _run(stub, pool, [], "--prune-shards")
        assert prune.returncode == 3, prune.stdout + prune.stderr
        assert "another battery holds the shard pool" in prune.stderr
        assert sh.root in _worktrees(stub) and sh.root.exists()
    assert _worktrees(stub) == {stub}


def test_battery_subset_passes_the_shard_dir_through():
    import battery_subset

    assert battery_subset.parse(["R", "o.json", "--shard-dir", "/p", "a", "b"]) == (
        "R",
        "o.json",
        ["a", "b"],
        ["--shard-dir", "/p"],
    )
    assert battery_subset.parse(["R", "o.json", "a"]) == ("R", "o.json", ["a"], [])


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
    """The 09-27 split-brain: the shard's pytest runs on the HOST's interpreter.

    The suite itself is green (the stub conftest puts the shard's src first), so
    without the path assertion this scores PROVEN; only the child clause sees it.
    """
    proc = _run(stub, tmp_path / "pool", ["proven"], "--check", mode="foreign-pytest")
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "(child)" in proc.stderr and "is not under" in proc.stderr
    assert "not green" not in proc.stderr
    assert "PROVEN" not in proc.stdout
    assert _porcelain(stub) == ""
    assert _worktrees(stub) == {stub}


def test_a_shard_venv_that_imports_another_checkout_is_refused_before_any_suite(
    stub, tmp_path
):
    proc = _run(stub, tmp_path / "pool", ["proven"], "--check", mode="foreign-src")
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "its python resolves rsr to" in proc.stderr
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


def test_a_baseline_that_did_not_run_exits_3(stub, tmp_path):
    """Review MINOR-8: a baseline with no probe is DID NOT RUN, not a failure."""
    proc = _run(stub, tmp_path / "pool", ["proven"], "--check", mode="no-probe")
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "the baseline suite did not run" in proc.stderr
    assert _worktrees(stub) == {stub}


def test_markdown_is_not_written_while_a_row_did_not_run(stub, tmp_path):
    """Review MINOR-4: the committed table records verdicts; DID_NOT_RUN is none."""
    md = tmp_path / "battery.md"
    md.write_text("the previous record\n")
    proc = _run(stub, tmp_path / "pool", ["proven", "no-probe"], "--markdown", str(md))
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "NOT WRITING" in proc.stderr
    assert md.read_text() == "the previous record\n"


def test_markdown_is_written_when_every_row_ran(stub, tmp_path):
    md = tmp_path / "battery.md"
    proc = _run(stub, tmp_path / "pool", ["proven"], "--markdown", str(md))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "**PROVEN**" in md.read_text()


def test_an_ignored_file_one_mutation_writes_cannot_change_the_next_verdict(
    stub, tmp_path
):
    """Review MAJOR-2, end to end: `adds-nothing` (OTHER = 5) makes the stub's
    test_no_poison write the ignored runs/poison; left in the shard, it fails that
    test under `proven` and scores it LEAKS."""
    out = tmp_path / "rows.json"
    proc = _run(stub, tmp_path / "pool", ["adds-nothing", "proven"], "--json", str(out))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    rows = {r["mutation"]: r for r in json.loads(out.read_text())}
    assert rows["adds-nothing"]["verdict"] == "ADDS NOTHING"
    assert rows["proven"]["verdict"] == "PROVEN", rows["proven"]["off_gate"]


def test_bytecode_from_one_mutation_cannot_run_under_the_next(stub, tmp_path):
    """Review MINOR-5, end to end. Both stub mutations keep calc.py's size and the
    stub pins its mtime, so any .pyc that survived from one suite to the next
    would be trusted: `adds-nothing` would then run `proven`'s VALUE = 3 (or the
    baseline's bytecode would hide `proven`) -- the stale-bytecode PROVEN of
    2026-09-29."""
    out = tmp_path / "rows.json"
    proc = _run(stub, tmp_path / "pool", ["proven", "adds-nothing"], "--json", str(out))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    rows = {r["mutation"]: r["verdict"] for r in json.loads(out.read_text())}
    assert rows == {"proven": "PROVEN", "adds-nothing": "ADDS NOTHING"}


# ------------------------------------------------------------- kill tests


@pytest.mark.parametrize(
    "sig",
    [signal.SIGTERM, signal.SIGHUP, signal.SIGINT],
    ids=["SIGTERM", "SIGHUP", "SIGINT"],
)
def test_a_stopped_battery_leaves_the_live_tree_and_git_clean(stub, tmp_path, sig):
    marker = tmp_path / "pytest.pid"
    proc = _start(stub, tmp_path / "pool", ["proven", "slow"], marker)
    pytest_pid, _grandchild = _wait_marker(marker, proc)
    # mid-mutation: the mutated file is in the shard, not here
    assert _porcelain(stub) == ""
    os.kill(proc.pid, sig)
    out, err = proc.communicate(timeout=120)
    assert proc.returncode == 128 + sig, out + err
    assert f"INTERRUPTED by {sig.name}" in err
    assert _wait_gone(pytest_pid), "the battery's pytest child outlived it"
    assert _porcelain(stub) == ""
    assert _worktrees(stub) == {stub}


def test_a_stopped_battery_kills_its_suites_grandchildren(stub, tmp_path):
    """Review MINOR-2: pytest's whole process group goes, not only pytest."""
    marker = tmp_path / "pytest.pid"
    proc = _start(stub, tmp_path / "pool", ["slow"], marker)
    pytest_pid, grandchild = _wait_marker(marker, proc)
    os.kill(proc.pid, signal.SIGTERM)
    proc.communicate(timeout=120)
    assert _wait_gone(pytest_pid)
    assert _wait_gone(grandchild), "the suite's grandchild outlived the battery"


def test_a_sigkilled_battery_leaves_only_shards_and_prune_removes_them(stub, tmp_path):
    marker, pool = tmp_path / "pytest.pid", tmp_path / "pool"
    proc = _start(stub, pool, ["slow"], marker)
    pytest_pid, grandchild = _wait_marker(marker, proc)
    os.kill(proc.pid, signal.SIGKILL)
    proc.communicate(timeout=60)
    assert proc.returncode == -signal.SIGKILL
    # SIGKILL cannot be handled: its orphaned suite is still asleep in the shard.
    # pytest leads its own process group, so one killpg takes the grandchild too.
    os.killpg(pytest_pid, signal.SIGKILL)
    assert _wait_gone(pytest_pid) and _wait_gone(grandchild)
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
    pytest_pid, grandchild = _wait_marker(marker, proc)
    os.kill(pytest_pid, signal.SIGKILL)
    stdout, stderr = proc.communicate(timeout=120)
    assert proc.returncode == 3, stdout + stderr
    rows = {r["mutation"]: r for r in json.loads(out.read_text())}
    assert rows["slow"]["verdict"] == "DID_NOT_RUN"
    assert "killed by signal 9" in rows["slow"]["did_not_run"]
    # the dead pytest's grandchild is killed with its group, before the next suite
    assert _wait_gone(grandchild)
    # the battery went on, in a reset shard, and the next verdict is real
    assert rows["proven"]["verdict"] == "PROVEN"
    assert _porcelain(stub) == ""
    assert _worktrees(stub) == {stub}


def test_a_sigint_to_the_pytest_child_alone_is_did_not_run(stub, tmp_path):
    """Review MAJOR-1: pytest catches SIGINT, runs sessionfinish (so the probe is
    written and its paths are good) and exits 2. That was scored ADDS NOTHING
    with the battery exiting 0; an interrupt after the gate failed would have
    scored PROVEN with every later test unrun."""
    marker, out = tmp_path / "pytest.pid", tmp_path / "rows.json"
    proc = _start(stub, tmp_path / "pool", ["slow", "proven"], marker, "--json", str(out))
    pytest_pid, grandchild = _wait_marker(marker, proc)
    os.kill(pytest_pid, signal.SIGINT)
    stdout, stderr = proc.communicate(timeout=120)
    assert proc.returncode == 3, stdout + stderr
    rows = {r["mutation"]: r for r in json.loads(out.read_text())}
    # either layer may name it (process status 2, or the probe's exitstatus 2)
    assert rows["slow"]["verdict"] == "DID_NOT_RUN", rows["slow"]
    assert rows["proven"]["verdict"] == "PROVEN"
    assert _wait_gone(grandchild)
    assert _porcelain(stub) == ""
    assert _worktrees(stub) == {stub}
