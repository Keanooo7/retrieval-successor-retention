"""The Mac Studio lane scheduler: counting semaphores over `fcntl.flock`.

ADR-0007: everything runs on one machine (M4 Max, 16 cores, 64 GB unified), so
"how many jobs at once" is a question about *this* machine, answered by measuring
it -- capacity experiment **C0** -- never by a number typed into a config.

Lanes
=====

==========  ================================  =====================================
lane        slots                             admission also requires
==========  ================================  =====================================
``cpu-det`` ``cpu_det_slots``                 --
``mps``     1                                 ``mps_reserves_cpu_slots`` cpu-det
                                              slots, a declared ``--peak-gb``, and
                                              ``free - peak >= reserve_gb``
``battery`` 1                                 ``battery_cpu_slots`` cpu-det slots
``agent``   ``agent_sessions``                --
==========  ================================  =====================================

Why the lanes are split this way: CPU runs are bit-exact and MPS runs are not
(~2.5e-6 floor), so a result's lane is part of its provenance; the decisive run
used 9 concurrent 1-thread CPU processes; and the mutation battery mutates source
in place, so it runs in its own worktree but still competes for the same cores.

Each slot is one file, ``.orchestrator/locks/<lane>/slot-<k>.lock``, held with
``flock(LOCK_EX)``. The kernel drops the lock when the last descriptor on it
closes, so **a holder that dies releases its slots with no cleanup step** -- there
is no stale-lock state to reconcile. ``slot.py`` passes its lock descriptors to
the child it runs, so the slots stay held for as long as *either* the wrapper or
the job is alive: a SIGKILLed wrapper does not free cores a still-running job is
using.

``ops/lanes.json`` -- the schema
================================

::

    {
      "generated_from": {
        "experiment": "C0",
        "run_id": "<C0's run id>",
        "ledger_path": "runs/capacity-c0/ledger.json",
        "ledger_git_sha": "<provenance.git_sha of that ledger>",
        "ledger_status": "ok",
        "ledger_keys": {"cpu_det_slots": "c0.cpu_det_slots", ...},
        "generated_utc": "..."
      },
      "cpu_det_slots": <int >= 1>,
      "battery_cpu_slots": <int, 0..cpu_det_slots>,
      "agent_sessions": <int >= 1>,
      "reserve_gb": <number >= 0>,
      "mps_reserves_cpu_slots": <int, 0..cpu_det_slots>
    }

🔴 **Every number must name its ledger key.** ``generated_from.ledger_keys`` must
map every field to exactly the key in `C0_LEDGER_KEYS`; a file without that map is
refused as hand-typed. The file is **written by ``lanes generate``** from C0's
ledger, never edited -- the same rule as ``docs/mutation-battery.md``: a record
that has to be retyped is a record that goes stale.

C0 has not run, so ``ops/lanes.json`` is **absent** and every lane refuses with
exit 3 naming C0. That is the correct state, not a bug to route around
(CLAUDE.md, "if the registry refuses your read, run the experiment").

CLI::

    PYTHONPATH=scripts .venv/bin/python -m orchestrator.lanes status [--json]
    PYTHONPATH=scripts .venv/bin/python -m orchestrator.lanes generate \\
        --ledger runs/capacity-c0/ledger.json
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import errno
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from rsr.exit_codes import ArgumentParser, Exit, refuse, run_main  # noqa: E402

__all__ = [
    "C0_EXPERIMENT",
    "C0_LEDGER",
    "C0_LEDGER_KEYS",
    "LANES",
    "LANES_FILE",
    "THREAD_ENV_VARS",
    "LanesConfig",
    "Lease",
    "Refused",
    "acquire",
    "free_memory_bytes",
    "generate",
    "load_config",
    "orch_root",
    "status",
    "thread_env",
]

C0_EXPERIMENT = "C0"
"""The capacity experiment every lane number comes from."""

C0_LEDGER = "runs/capacity-c0/ledger.json"

C0_LEDGER_KEYS: dict[str, str] = {
    "cpu_det_slots": "c0.cpu_det_slots",
    "battery_cpu_slots": "c0.battery_cpu_slots",
    "agent_sessions": "c0.agent_sessions",
    "reserve_gb": "c0.reserve_gb",
    "mps_reserves_cpu_slots": "c0.mps_reserves_cpu_slots",
}
"""Field of ``ops/lanes.json`` -> the ledger row key C0 must write it under.

📌 C0's pre-registration does not exist yet. These names are the contract that
pre-registration has to adopt (or change here, in the same commit); ``lanes
generate`` refuses a ledger that lacks any of them."""

_INT_FIELDS = (
    "cpu_det_slots",
    "battery_cpu_slots",
    "agent_sessions",
    "mps_reserves_cpu_slots",
)
_FLOAT_FIELDS = ("reserve_gb",)

LANES = ("cpu-det", "mps", "battery", "agent")
LANES_FILE = Path("ops") / "lanes.json"

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "RSR_TORCH_THREADS",
)
"""Forced to the slot count for CPU work. ``RSR_TORCH_THREADS`` is for scripts to
pass to ``torch.set_num_threads``; the rest cap the BLAS/OpenMP pools."""

POLL_SECONDS = 0.1
_GIB = 1024**3


class Refused(Exception):
    """A precondition refused: the caller maps this to exit 3, *did not run*."""


# --------------------------------------------------------------------------- root


def _git() -> str:
    for cand in (os.environ.get("RSR_GIT"), "/opt/homebrew/bin/git"):
        if cand and Path(cand).exists():
            return cand
    found = shutil.which("git")
    if found is None:
        raise Refused("no git binary found (set RSR_GIT)")
    return found


def orch_root(cwd: Path | None = None) -> Path:
    """``$RSR_ORCH_ROOT`` if set, else the git toplevel of ``cwd``.

    ⚠️ Slots are only shared between processes that resolve the **same** root. A
    process started inside a separate worktree resolves that worktree's toplevel;
    anything that must share the Mac Studio's cores with the main checkout (the
    battery in its own worktree) must be launched with ``RSR_ORCH_ROOT`` set, or
    from the main checkout with ``slot.py --cwd <worktree>``.
    """
    env = os.environ.get("RSR_ORCH_ROOT")
    if env:
        return Path(env).resolve()
    r = subprocess.run(
        [_git(), "rev-parse", "--show-toplevel"],
        cwd=cwd or Path.cwd(),
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise Refused(
            f"cannot resolve the orchestrator root: RSR_ORCH_ROOT is unset and git "
            f"rev-parse failed: {r.stderr.strip()}"
        )
    return Path(r.stdout.strip()).resolve()


# ------------------------------------------------------------------------- config


@dataclass(frozen=True)
class LanesConfig:
    cpu_det_slots: int
    battery_cpu_slots: int
    agent_sessions: int
    reserve_gb: float
    mps_reserves_cpu_slots: int
    generated_from: dict = field(default_factory=dict)

    def capacity(self, lane: str) -> int:
        return {
            "cpu-det": self.cpu_det_slots,
            "mps": 1,
            "battery": 1,
            "agent": self.agent_sessions,
        }[lane]

    def demand(self, lane: str, k: int) -> list[tuple[str, int]]:
        """Every (lane, count) an admission to ``lane`` with ``k`` slots takes."""
        if lane == "mps":
            return [("mps", k), ("cpu-det", self.mps_reserves_cpu_slots)]
        if lane == "battery":
            return [("battery", k), ("cpu-det", self.battery_cpu_slots)]
        return [(lane, k)]


def _absent_message(path: Path) -> str:
    return (
        f"{path} is absent. Lane capacities are MEASURED by capacity experiment "
        f"{C0_EXPERIMENT}, which has not run: run {C0_EXPERIMENT}, then "
        f"`lanes generate --ledger {C0_LEDGER}`. Do not hand-type the file."
    )


def _validate(doc: object, where: str) -> LanesConfig:
    if not isinstance(doc, dict):
        raise Refused(f"{where}: not a JSON object")
    gen = doc.get("generated_from")
    if not isinstance(gen, dict) or gen.get("ledger_keys") != C0_LEDGER_KEYS:
        raise Refused(
            f"{where}: generated_from.ledger_keys must name the {C0_EXPERIMENT} "
            f"ledger key of every field ({C0_LEDGER_KEYS}); a lanes file without "
            f"them is hand-typed. Regenerate it with `lanes generate`."
        )
    for key in ("run_id", "ledger_git_sha"):
        if not gen.get(key):
            raise Refused(f"{where}: generated_from.{key} is missing")
    vals: dict[str, int | float] = {}
    for name in _INT_FIELDS:
        v = doc.get(name)
        if isinstance(v, bool) or not isinstance(v, int):
            raise Refused(f"{where}: {name} must be an int, got {v!r}")
        vals[name] = v
    for name in _FLOAT_FIELDS:
        v = doc.get(name)
        if isinstance(v, bool) or not isinstance(v, int | float):
            raise Refused(f"{where}: {name} must be a number, got {v!r}")
        vals[name] = float(v)
    cpu = vals["cpu_det_slots"]
    if cpu < 1:
        raise Refused(f"{where}: cpu_det_slots must be >= 1, got {cpu}")
    if vals["agent_sessions"] < 1:
        raise Refused(f"{where}: agent_sessions must be >= 1")
    if vals["reserve_gb"] < 0:
        raise Refused(f"{where}: reserve_gb must be >= 0")
    for name in ("battery_cpu_slots", "mps_reserves_cpu_slots"):
        if not 0 <= vals[name] <= cpu:
            raise Refused(f"{where}: {name}={vals[name]} is outside 0..{cpu}")
    return LanesConfig(**vals, generated_from=gen)  # type: ignore[arg-type]


def load_config(root: Path) -> LanesConfig:
    path = root / LANES_FILE
    if not path.exists():
        raise Refused(_absent_message(path))
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise Refused(f"{path}: not valid JSON ({e})") from None
    return _validate(doc, str(path))


def generate(ledger_path: Path) -> dict:
    """Build the ``ops/lanes.json`` document from C0's ledger. Refuses on any gap."""
    if not ledger_path.exists():
        raise Refused(f"{ledger_path} is absent: {C0_EXPERIMENT} has not run")
    led = json.loads(ledger_path.read_text())
    if led.get("status") != "ok":
        raise Refused(
            f"{ledger_path}: status is {led.get('status')!r}, not 'ok'; a partial or "
            f"crashed {C0_EXPERIMENT} does not set capacities"
        )
    rows = {r.get("key"): r for r in led.get("rows", []) if isinstance(r, dict)}
    missing = [k for k in C0_LEDGER_KEYS.values() if k not in rows]
    if missing:
        raise Refused(f"{ledger_path}: no row for ledger key(s) {missing}")
    doc: dict = {}
    for name, key in C0_LEDGER_KEYS.items():
        doc[name] = rows[key].get("value")
    prov = led.get("provenance") or {}
    doc = {
        "generated_from": {
            "experiment": C0_EXPERIMENT,
            "run_id": led.get("run_id"),
            "ledger_path": str(ledger_path),
            "ledger_git_sha": prov.get("git_sha"),
            "ledger_status": led.get("status"),
            "ledger_keys": dict(C0_LEDGER_KEYS),
            "generated_utc": _utc(),
        },
        **doc,
    }
    _validate(doc, str(ledger_path))  # the same checks load_config will apply
    return doc


# ------------------------------------------------------------------------- memory


_PAGE_RE = re.compile(r"page size of (\d+) bytes")
_ROW_RE = re.compile(r"^Pages (free|inactive|speculative):\s+(\d+)\.?\s*$", re.M)


def _vm_stat_text() -> str:
    exe = shutil.which("vm_stat") or "/usr/bin/vm_stat"
    try:
        r = subprocess.run([exe], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise Refused(f"cannot measure free memory: vm_stat failed ({e})") from None
    if r.returncode != 0:
        raise Refused(f"cannot measure free memory: vm_stat exited {r.returncode}")
    return r.stdout


def free_memory_bytes(text: str | None = None) -> int:
    """(free + inactive + speculative) pages x page size, from ``vm_stat``."""
    text = _vm_stat_text() if text is None else text
    page = _PAGE_RE.search(text)
    rows = dict(_ROW_RE.findall(text))
    if page is None or set(rows) != {"free", "inactive", "speculative"}:
        raise Refused(f"cannot measure free memory: unparseable vm_stat:\n{text}")
    return sum(int(v) for v in rows.values()) * int(page.group(1))


# -------------------------------------------------------------------------- locks


def thread_env(n: int) -> dict[str, str]:
    return dict.fromkeys(THREAD_ENV_VARS, str(n))


def _utc() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")


def _lock_path(root: Path, lane: str, k: int) -> Path:
    return root / ".orchestrator" / "locks" / lane / f"slot-{k}.lock"


def _try_lock(path: Path) -> int | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        os.close(fd)
        if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
            return None
        raise
    return fd


@dataclass
class Lease:
    held: list[tuple[str, int, int]]  # (lane, slot index, fd)

    def fds(self) -> tuple[int, ...]:
        return tuple(fd for _lane, _k, fd in self.held)

    def slots(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for lane, k, _fd in self.held:
            out.setdefault(lane, []).append(k)
        return out

    def release(self) -> None:
        for _lane, _k, fd in self.held:
            with contextlib.suppress(OSError):
                os.close(fd)
        self.held = []


def _attempt(root: Path, demand: list[tuple[str, int]], cfg: LanesConfig, info: dict):
    got: list[tuple[str, int, int]] = []
    for lane, count in demand:
        n = 0
        for k in range(cfg.capacity(lane)):
            if n == count:
                break
            fd = _try_lock(_lock_path(root, lane, k))
            if fd is not None:
                got.append((lane, k, fd))
                n += 1
        if n < count:
            Lease(got).release()
            return None
    payload = json.dumps({**info, "acquired_utc": _utc()}).encode()
    for _lane, _k, fd in got:
        os.ftruncate(fd, 0)
        os.pwrite(fd, payload, 0)
    return Lease(got)


def acquire(
    root: Path,
    cfg: LanesConfig,
    lane: str,
    k: int,
    *,
    wait: float = 0.0,
    info: dict | None = None,
    peak_gb: float | None = None,
    free_bytes=free_memory_bytes,
) -> Lease:
    """All-or-nothing: take every slot the admission needs, or none of them.

    Partial holdings are released before each retry, so two multi-slot admissions
    cannot deadlock each other. Raises `Refused` when ``wait`` expires.
    """
    if lane not in LANES:
        raise Refused(f"unknown lane {lane!r}; lanes are {LANES}")
    if k < 1 or k > cfg.capacity(lane):
        raise Refused(f"lane {lane} has {cfg.capacity(lane)} slot(s); asked for {k}")
    demand = [(ln, c) for ln, c in cfg.demand(lane, k) if c > 0]
    for ln, c in demand:
        if c > cfg.capacity(ln):
            raise Refused(f"admission to {lane} needs {c} {ln} slots, lane has fewer")
    if lane == "mps" and (peak_gb is None or peak_gb <= 0):
        raise Refused("the mps lane requires a declared --peak-gb > 0")
    info = {"pid": os.getpid(), "lane": lane, **(info or {})}
    deadline = time.monotonic() + max(wait, 0.0)
    last = "slots busy"
    while True:
        lease = _attempt(root, demand, cfg, info)
        if lease is not None:
            if lane != "mps":
                return lease
            free_gb = free_bytes() / _GIB
            if free_gb - peak_gb >= cfg.reserve_gb:
                return lease
            lease.release()
            last = (
                f"insufficient memory: free {free_gb:.2f} GB - peak {peak_gb:.2f} GB "
                f"< reserve {cfg.reserve_gb:.2f} GB"
            )
        if time.monotonic() >= deadline:
            raise Refused(f"lane {lane} x{k}: not admitted within {wait}s ({last})")
        time.sleep(POLL_SECONDS)


def status(root: Path, cfg: LanesConfig) -> dict:
    """Probe every slot. ⚠️ A probe holds a free slot for microseconds, so an
    acquirer with ``wait=0`` racing it can be refused spuriously; use a wait."""
    lanes: dict[str, dict] = {}
    for lane in LANES:
        held = []
        for k in range(cfg.capacity(lane)):
            path = _lock_path(root, lane, k)
            fd = _try_lock(path)
            if fd is not None:
                os.close(fd)
                continue
            try:
                holder = json.loads(path.read_text() or "{}")
            except (OSError, json.JSONDecodeError):
                holder = {}
            held.append({"slot": k, "holder": holder})
        cap = cfg.capacity(lane)
        lanes[lane] = {"capacity": cap, "held": held, "free": cap - len(held)}
    return {
        "root": str(root),
        "generated_from": cfg.generated_from,
        "reserve_gb": cfg.reserve_gb,
        "lanes": lanes,
    }


# ---------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> Exit:
    ap = ArgumentParser(prog="lanes")
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("status")
    st.add_argument("--json", action="store_true")
    gen = sub.add_parser("generate")
    gen.add_argument("--ledger", type=Path, default=None)
    gen.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    try:
        root = orch_root()
        if args.cmd == "generate":
            ledger = args.ledger or root / C0_LEDGER
            doc = generate(ledger)
            out = args.out or root / LANES_FILE
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(doc, indent=2) + "\n")
            print(f"wrote {out} from {ledger} (run {doc['generated_from']['run_id']})")
            return Exit.OK
        cfg = load_config(root)
        report = status(root, cfg)
    except Refused as e:
        refuse(Exit.DID_NOT_RUN, str(e))
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for lane, s in report["lanes"].items():
            holders = ", ".join(
                f"{h['slot']}:{h['holder'].get('job_id', h['holder'].get('pid'))}"
                for h in s["held"]
            )
            print(f"{lane:8s} {s['free']}/{s['capacity']} free  {holders}")
    return Exit.OK


if __name__ == "__main__":
    run_main(main)
