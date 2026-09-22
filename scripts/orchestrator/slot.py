"""Run one command inside a lane: acquire, run, record, release.

::

    PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run \\
        --lane cpu-det --slots 1 [--wait SECONDS] [--job-id ID] [--item ID] \\
        [--pinned-sha SHA] [--peak-gb G] [--cwd DIR] -- cmd ...

Exit status
===========

* **The command ran:** its return code, **verbatim** -- 0, 1, 2, 3, 5, 137, any
  of them. A child killed by signal ``N`` exits ``128 + N`` (shell convention),
  so SIGKILL is 137 whether the child said it or the kernel did. Collapsing this
  to a bool or into the 0-4 protocol would destroy the one fact the scheduler
  exists to carry. This is the only path in the orchestrator that does not go
  through ``rsr.exit_codes.status()``, because that function (correctly) refuses
  codes outside 0-4 -- and a child's code is not ours to reinterpret.
* **The command did not run** (``ops/lanes.json`` absent -- C0 has not run; no
  admission within ``--wait``; pinned sha mismatch or dirty tree; the executable
  does not exist): **3** via ``rsr.exit_codes.refuse``, with ``DID NOT RUN`` on
  stderr and **no job record written**.

⚠️ A child that itself exits 3 is indistinguishable by exit status alone from a
refusal. The job record disambiguates: a refusal writes none.

Job record -- ``.orchestrator/jobs/<job_id>.json``
==================================================

``{"job_id", "item_id", "lane", "slots", "cmd", "cwd", "pid", "pinned_sha",
"start", "end", "rc", "status"}`` plus ``slot_pid``, ``threads``, ``held``
(the lock slots) and ``signal``.

* ``pid`` is the **child's** pid, not this wrapper's. The child inherits the lock
  descriptors, so slots stay held exactly as long as ``pid`` lives; a reconciler
  that finds ``status == "running"`` and ``pid`` dead is looking at a job whose
  slots are already free.
* ``status``: ``running`` while the child runs; ``done`` when it exited on its own
  (any rc); ``crashed`` when it was terminated by a signal, or when this wrapper
  received SIGTERM/SIGINT/SIGHUP and forwarded it. A crash of the wrapper itself
  leaves ``running`` -- detected by the reconciler, not here.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import NoReturn

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from orchestrator import lanes  # noqa: E402
from rsr.exit_codes import ArgumentParser, Exit, refuse, run_main  # noqa: E402

__all__ = ["check_pinned", "exit_verbatim", "job_threads", "main", "run_job"]

_FORWARDED = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)


def job_threads(lane: str, k: int, cfg: lanes.LanesConfig) -> int | None:
    """Thread count forced on the job, or None for lanes that do no CPU math."""
    if lane == "cpu-det":
        return k
    if lane == "battery":
        return max(cfg.battery_cpu_slots, 1)
    if lane == "mps":
        return max(cfg.mps_reserves_cpu_slots, 1)
    return None


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [lanes._git(), "-C", str(cwd), *args], capture_output=True, text=True
    )


def check_pinned(cwd: Path, sha: str) -> None:
    """Refuse unless ``cwd``'s HEAD is ``sha`` and the tree outside runs/ is clean."""
    want = _git(cwd, "rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}")
    head = _git(cwd, "rev-parse", "HEAD")
    if want.returncode != 0 or head.returncode != 0:
        raise lanes.Refused(f"--pinned-sha {sha}: cannot resolve it in {cwd}")
    if want.stdout.strip() != head.stdout.strip():
        raise lanes.Refused(
            f"--pinned-sha {sha}: HEAD of {cwd} is {head.stdout.strip()}, "
            f"not {want.stdout.strip()}"
        )
    st = _git(cwd, "status", "--porcelain", "--", ":(top)", ":(top,exclude)runs")
    if st.returncode != 0:
        raise lanes.Refused(f"git status failed in {cwd}: {st.stderr.strip()}")
    if st.stdout.strip():
        raise lanes.Refused(
            f"--pinned-sha {sha}: {cwd} has changes outside runs/, so the code "
            f"that would run is not the pinned commit:\n{st.stdout}"
        )


def _write_record(path: Path, record: dict) -> None:
    tmp = path.with_suffix(f".json.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(tmp, path)


def _parse(argv: list[str]):
    if "--" in argv:
        i = argv.index("--")
        opts, cmd = argv[:i], argv[i + 1 :]
    else:
        opts, cmd = argv, []
    ap = ArgumentParser(prog="slot")
    sub = ap.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run")
    run.add_argument("--lane", required=True, choices=lanes.LANES)
    run.add_argument("--slots", type=int, default=1)
    run.add_argument("--wait", type=float, default=0.0)
    run.add_argument("--job-id", default=None)
    run.add_argument("--item", default=None)
    run.add_argument("--pinned-sha", default=None)
    run.add_argument("--peak-gb", type=float, default=None)
    run.add_argument("--cwd", type=Path, default=None)
    args = ap.parse_args(opts)
    if not cmd:
        ap.error("no command given after `--`")
    return args, cmd


def run_job(args, cmd: list[str]) -> int:
    """Acquire, run, record, release. Returns the child's rc (signal N -> 128+N).
    Every refusal exits 3 from here via `refuse`, before the child exists."""
    cwd = (args.cwd or Path.cwd()).resolve()
    try:
        root = lanes.orch_root()
        cfg = lanes.load_config(root)
        if args.lane == "mps" and args.peak_gb is None:
            raise lanes.Refused("the mps lane requires a declared --peak-gb")
        if args.pinned_sha:
            check_pinned(cwd, args.pinned_sha)
        job_id = args.job_id or (
            time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + f"-{args.lane}-{os.getpid()}"
        )
        if "/" in job_id or job_id.startswith("."):
            raise lanes.Refused(f"job id {job_id!r} is not a plain file name")
        jobs = root / ".orchestrator" / "jobs"
        jobs.mkdir(parents=True, exist_ok=True)
        record_path = jobs / f"{job_id}.json"
        if record_path.exists():
            raise lanes.Refused(f"job record {record_path} already exists")
        lease = lanes.acquire(
            root,
            cfg,
            args.lane,
            args.slots,
            wait=args.wait,
            info={"job_id": job_id, "item_id": args.item},
            peak_gb=args.peak_gb,
        )
    except lanes.Refused as e:
        refuse(Exit.DID_NOT_RUN, str(e))

    threads = job_threads(args.lane, args.slots, cfg)
    env = dict(os.environ)
    if threads is not None:
        env.update(lanes.thread_env(threads))

    received: list[int] = []
    child: list[subprocess.Popen] = []

    def forward(signum, _frame):
        received.append(signum)
        if child and child[0].poll() is None:
            child[0].send_signal(signum)

    previous = {s: signal.signal(s, forward) for s in _FORWARDED}
    try:
        try:
            proc = subprocess.Popen(cmd, cwd=cwd, env=env, pass_fds=lease.fds())
        except OSError as e:
            lease.release()
            refuse(Exit.DID_NOT_RUN, f"cannot start {cmd[0]!r}: {e}")
        child.append(proc)
        for s in received:  # a signal that arrived before the child existed
            proc.send_signal(s)
        record = {
            "job_id": job_id,
            "item_id": args.item,
            "lane": args.lane,
            "slots": args.slots,
            "cmd": cmd,
            "cwd": str(cwd),
            "pid": proc.pid,
            "slot_pid": os.getpid(),
            "pinned_sha": args.pinned_sha,
            "threads": threads,
            "held": lease.slots(),
            "start": lanes._utc(),
            "end": None,
            "rc": None,
            "signal": None,
            "status": "running",
        }
        _write_record(record_path, record)
        raw = proc.wait()
    finally:
        for s, h in previous.items():
            signal.signal(s, h)
        lease.release()

    rc = raw if raw >= 0 else 128 - raw
    record.update(
        end=lanes._utc(),
        rc=rc,
        signal=(-raw if raw < 0 else (received[0] if received else None)),
        status="crashed" if (raw < 0 or received) else "done",
    )
    _write_record(record_path, record)
    return rc


def exit_verbatim(rc: int) -> NoReturn:
    """🔴 The child's rc, as the child gave it. Not `status()`: that refuses 5 and
    137, and a child's code is not ours to reinterpret (module docstring)."""
    raise SystemExit(rc)


def main(argv: list[str] | None = None) -> Exit:
    """Never returns: a refusal exits 3 via `refuse`, a job exits with its own rc.
    `run_main` stays the entry point so the protocol checks see one."""
    args, cmd = _parse(sys.argv[1:] if argv is None else argv)
    exit_verbatim(run_job(args, cmd))


if __name__ == "__main__":
    run_main(main)
