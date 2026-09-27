"""The stub `slot`'s job record is never visible half-written (I5b, PLAN-v4 I5).

`tests/_orch_loop_helpers.py`'s stub `slot` wrote its job record with
`path.write_text(...)`, which opens the file with `"w"` -- truncating it -- and only
then writes. `Orch.wait_job` meanwhile polls `json.loads(path.read_text())` every
50 ms. A poll that lands between the truncate and the write reads `""` and raises
`JSONDecodeError`. The real `scripts/orchestrator/slot.py::_write_record` already
writes a temporary file and `os.replace`s it; the stub did not.

This test holds that window open instead of hoping to land in it: the stub runs
under a wrapper that pauses every write-open inside `.orchestrator/jobs/` until the
test says go, and at each pause the test does `wait_job`'s read. Deterministic, no
timing assumption beyond a 30 s safety limit.

📌 This demonstrates a race in the stub. It does **not** establish that the race
caused the 2026-09-26 red on
`test_submit_launches_the_slot_detached_at_the_pinned_sha`: that run left only a
node id (docs/lab-notes/i5-flake.md).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from _orch_loop_helpers import _STUB_HEAD, STUBS

# Runs a script with every write-open under a `jobs/` directory paused, after the
# open (so "w" has already truncated) and before the first byte is written.
PAUSER = """\
import io, os, pathlib, runpy, sys, time

barrier = pathlib.Path(os.environ["PAUSE_BARRIER"])
_real_open = io.open


def _open(file, mode="r", *args, **kwargs):
    f = _real_open(file, mode, *args, **kwargs)
    if isinstance(file, (str, os.PathLike)) and "w" in mode:
        p = pathlib.Path(file)
        if p.parent.name == "jobs":
            n = len(list(barrier.glob("paused-*")))
            (barrier / f"paused-{n}").write_text(p.name)
            deadline = time.monotonic() + 30
            while not (barrier / f"go-{n}").exists() and time.monotonic() < deadline:
                time.sleep(0.005)
    return f


io.open = _open
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
"""


def _await(path: Path, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() > deadline:
            raise AssertionError(f"the writer never reached {path.name}")
        time.sleep(0.005)


def _wait_job_read(path: Path) -> dict | None:
    """One poll of `Orch.wait_job`, verbatim: exists, then `json.loads(read_text())`."""
    if path.exists():
        return json.loads(path.read_text())
    return None


def test_the_stub_slot_never_exposes_a_half_written_job_record(tmp_path):
    barrier = tmp_path / "barrier"
    barrier.mkdir()
    stub = tmp_path / "slot"
    stub.write_text(_STUB_HEAD + STUBS["slot"])
    pauser = tmp_path / "pause_writes.py"
    pauser.write_text(PAUSER)
    root = tmp_path / "repo"
    job = root / ".orchestrator" / "jobs" / "j1.json"
    env = {
        **os.environ,
        "RSR_ORCH_ROOT": str(root),
        "STUB_LOG_DIR": str(tmp_path),
        "PAUSE_BARRIER": str(barrier),
    }
    argv = [
        sys.executable,
        str(pauser),
        str(stub),
        "run",
        "--lane",
        "cpu",
        "--slots",
        "1",
        "--item",
        "a",
        "--job-id",
        "j1",
        "--pinned-sha",
        "abc",
        "--",
        sys.executable,
        "-c",
        "pass",
    ]
    proc = subprocess.Popen(argv, env=env)
    torn: list[str] = []
    try:
        # the stub writes the record twice: "running", then "done"
        for n in range(2):
            _await(barrier / f"paused-{n}")
            try:
                _wait_job_read(job)
            except json.JSONDecodeError as e:
                torn.append(f"pause {n}: {job.read_text()!r} -> {e}")
            (barrier / f"go-{n}").touch()
    finally:
        for n in range(4):
            (barrier / f"go-{n}").touch()
        rc = proc.wait(timeout=60)
    assert torn == [], f"wait_job's read saw a half-written job record: {torn}"
    assert rc == 0
    assert _wait_job_read(job)["status"] == "done"
    assert sorted(p.name for p in job.parent.iterdir()) == ["j1.json"]
