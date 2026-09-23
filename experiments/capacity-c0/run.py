"""C0 -- measure this machine's lanes (ADR-0007; `scripts/orchestrator/lanes.py`).

Pre-registration: `experiments/capacity-c0/PREREG.md`, committed ahead of this file.
Brief: `docs/lab-notes/dispatch-capacity-c0.md`. Servers:
`docs/owner/rulings/R-2026-09-22-mlx-servers.md`.

Every rule below is the PREREG's ``thresholds`` block, transcribed into a **pure
function** (section "the rules"), so each is testable on a synthetic table without
running anything. The measurement (section "the measurement") only produces the
tables those functions read. Nothing here types a capacity: every one of the five
``C0_LEDGER_KEYS`` rows is the return value of its rule over measured inputs.

Order (PREREG "Falsifier first"; brief "the solo training job runs first"):

1. the owner's LLM servers are stopped when a signed ruling says they YIELD
   (identified by command line only; everything recorded), THEN the quiet check;
2. the determinism sweep on the training workload -- the solo 1-thread job first,
   whose ``ru_maxrss`` sets ``k_mem_training``; one mismatch => falsified, no
   capacity key, exit 1;
3. the core sweep on the batch-4 workload (solo first, sets ``k_mem_core``);
4. the suite timings W(n); 5. the MPS job; 6. the rules; 7. the ledger.

The servers are restarted in a ``finally``, so an exception, a deadline or a
falsified verdict still restarts them.

Usage (attended, on a quiet machine)::

    uv run python experiments/capacity-c0/run.py --dry-plan     # prints, runs nothing
    uv run python experiments/capacity-c0/run.py                # the measurement
    uv run python experiments/capacity-c0/run.py --render-results   # RESULTS.md
    PYTHONPATH=scripts .venv/bin/python -m orchestrator.lanes generate \\
        --ledger runs/capacity-c0/ledger.json

Exit codes (`rsr.exit_codes`): 0 survived and the five keys written · 1 falsified
(no capacity key written) · 3 did not run or did not complete (machine not quiet,
deadline -> ledger ``partial``, a killed or failed job, an exception).
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from orchestrator import lanes  # noqa: E402

from rsr.exit_codes import ArgumentParser, Exit, refuse, run_main, status  # noqa: E402

EXPERIMENT = "experiments/capacity-c0/run.py"
RUN_ID = "capacity-c0"
PREREG = "experiments/capacity-c0/PREREG.md"

# --------------------------------------------------------------------------- #
# 🔒 PREREG.md `thresholds`, transcribed. Changing any of these is changing the
# pre-registration (brief do_not: "Change any threshold, the sweep, or the
# workload").
# --------------------------------------------------------------------------- #

SEED = 0
GIB = 2**30
#: quiet_machine: "no process outside this run with RSS > 2 GB". Read in GiB, as
#: memory_cap says of all memory in this PREREG.
QUIET_RSS_GIB = 2.0
#: memory_cap: k_mem_w = floor((hw.memsize/2^30 - 8) / R_w).
MEM_MARGIN_GIB = 8
#: training_workload: S0-03's CONFIG, seed 0, iters 10, CPU.
TRAINING_ITERS = 10
#: core_workload: the same CONFIG with batch 4, iters 30, CPU, 1 thread.
CORE_BATCH = 4
CORE_ITERS = 30
#: determinism_sweep: 1-thread k in {1,2,4,8} up to k_mem_training, 2 reps; plus
#: 5-thread solo, then 3 concurrent if 3 <= k_mem_training.
DET_T1_KS = (1, 2, 4, 8)
DET_REPS = 2
DET_T5_THREADS = 5
DET_T5_CONCURRENT = 3
#: core_sweep_k and repetitions.
CORE_SWEEP_K = (1, 2, 4, 8, 10, 12, 14, 15, 16)
CORE_REPS = 2
#: cpu_det_slots: g_j >= 0.5 * T(1).
MARGINAL_FRACTION = 0.5
#: battery_cpu_slots: n in {1,2,4,8}; smallest n with W(n) <= 1.15 * min W.
BATTERY_THREADS = (1, 2, 4, 8)
BATTERY_TOLERANCE = 1.15
#: mps_reserves_cpu_slots: the training workload on mps, iters 50, uncapped.
MPS_ITERS = 50
#: agent_sessions: policy value 2; 1 if 2 * p95 > 0.10 * physical RAM.
AGENT_POLICY = 2
AGENT_RAM_FRACTION = 0.10
#: reserve_gb: + 4.0 GB OS margin, rounded UP to 0.5 GB.
OS_MARGIN_GB = 4.0
RESERVE_QUANTUM_GB = 0.5
#: conflicts (a): retrieval-curve needs 15 thread-slots.
RETRIEVAL_CURVE_SLOTS = 15
#: deadline_hours.
DEADLINE_HOURS = 4
#: Samplers (quiet_machine / mps / agent): 1 Hz.
SAMPLE_HZ = 1.0

#: R-2026-09-22-mlx-servers: "the owner's LLM servers ... identified by command
#: line". Matched as a whole path component or token, never as a substring of an
#: unrelated word.
SERVER_NAMES = ("mlx_lm.server", "llama-server")
_SERVER_RE = re.compile(
    r"(?:^|[\s/])(" + "|".join(map(re.escape, SERVER_NAMES)) + r")(?=\s|$)"
)

STOP_GRACE_S = 30.0
KILL_GRACE_S = 10.0
RESPAWN_WATCH_S = 5.0

#: Brief "Budget" -- PREDICTIONS for --dry-plan only, never used by a rule.
PRED = {
    "s_per_iter_training": 4.8,  # "about 4.6-4.9 s per iteration, from the decisive run"
    "s_per_sample_core": 0.45,  # PREREG core_workload "~0.45 s" per sample
    "startup_s": 10.0,  # an assumption: python + torch import + corpus
    "suite_s": 80.0,  # "Suite timings: about 4 x 80 s"
    "s_per_iter_mps": 1.0,  # an assumption: no MPS figure is in the brief
    "r_training_gib": 6.8,  # PREREG memory_cap "R_training ~ 6.8 GiB"
    "r_core_gib": 2.5,  # PREREG core_workload "~2.5 GiB peak"
    "perf_cores": 12,  # PREREG "12 performance and 4 efficiency cores"
}


class Refusal(Exception):
    """A precondition refused: exit 3, *did not run*."""


class DeadlineExceeded(Exception):
    """The 4 h deadline passed: ledger ``partial``, exit 3."""


class JobFailed(Exception):
    """A job exited non-zero, was killed, or wrote no usable result: ``partial``."""


# =========================================================================== #
# the rules -- pure functions, each PREREG `thresholds` entry
# =========================================================================== #


def maxrss_bytes(ru_maxrss: int, platform: str = sys.platform) -> int:
    """``ru_maxrss`` in bytes. macOS reports BYTES, Linux reports KiB
    (getrusage(2) on each). Read from ``os.wait4`` on the child (memory_cap)."""
    return int(ru_maxrss) if platform == "darwin" else int(ru_maxrss) * 1024


def k_mem(memsize_bytes: int, r_job_bytes: int) -> int:
    """memory_cap: ``floor((hw.memsize/2^30 - 8) / R_w)``, R_w in GiB."""
    if r_job_bytes <= 0:
        raise ValueError(f"R_w must be > 0 bytes, got {r_job_bytes}")
    return math.floor((memsize_bytes / GIB - MEM_MARGIN_GIB) / (r_job_bytes / GIB))


def sweep_levels(ks: Iterable[int], k_mem_w: int) -> list[int]:
    """memory_cap: "No sweep level may exceed its workload's k_mem"."""
    return [k for k in ks if k <= k_mem_w]


def makespan(job_times_by_rep: list[list[float]]) -> float:
    """repetitions: makespan = the longest job time at the level, and the SLOWER of
    the repetitions is used."""
    if not job_times_by_rep or any(not rep for rep in job_times_by_rep):
        raise ValueError("a level needs >= 1 repetition, each with >= 1 job time")
    per_rep = [max(rep) for rep in job_times_by_rep]
    return max(per_rep)


def cpu_det_slots(makespans: dict[int, float]) -> tuple[int, list[dict[str, Any]]]:
    """cpu_det_slots: T(0) = 0; T(k) = k / makespan(k); g_i = (T(k_i) - T(k_{i-1}))
    / (k_i - k_{i-1}); the largest k_i with g_j >= 0.5 * T(1) for EVERY sweep step
    j <= i. The first step (0 -> 1) always passes, so the minimum is 1.

    Returns ``(slots, table)``; the table is written to the ledger as is.
    """
    ks = sorted(makespans)
    if not ks or ks[0] != 1:
        raise ValueError(f"the core sweep must include k=1 (T(1)); got {ks}")
    T = {k: k / makespans[k] for k in ks}
    t1 = T[1]
    prev_k, prev_t = 0, 0.0
    best = 0
    all_pass = True
    table = []
    for k in ks:
        g = (T[k] - prev_t) / (k - prev_k)
        passes = g >= MARGINAL_FRACTION * t1
        all_pass = all_pass and passes
        if all_pass:
            best = k
        table.append(
            {
                "k": k,
                "makespan_s": makespans[k],
                "T": T[k],
                "g": g,
                "threshold": MARGINAL_FRACTION * t1,
                "step_passes": passes,
                "prefix_passes": all_pass,
            }
        )
        prev_k, prev_t = k, T[k]
    return best, table


def battery_cpu_slots_raw(wall_s: dict[int, float]) -> int:
    """battery_cpu_slots: the smallest n with W(n) <= 1.15 * min W."""
    if not wall_s:
        raise ValueError("no suite timings")
    floor = BATTERY_TOLERANCE * min(wall_s.values())
    return min(n for n, w in wall_s.items() if w <= floor)


def clamp_to(raw: int, cpu_det: int) -> int:
    """ "Clamped to cpu_det_slots"."""
    return min(raw, cpu_det)


def p95(samples: Iterable[float]) -> float:
    """95th percentile, NEAREST RANK (an observed sample: sorted[ceil(0.95 n) - 1]).
    The PREREG says "p95" without a method; nearest rank never interpolates a value
    that was not observed."""
    xs = sorted(float(x) for x in samples)
    if not xs:
        raise ValueError("p95 of no samples")
    return xs[math.ceil(0.95 * len(xs)) - 1]


def mps_reserves_cpu_slots_raw(cores_samples: Iterable[float]) -> int:
    """mps_reserves_cpu_slots: ceil(p95) of ``ps %cpu``/100, minimum 1."""
    return max(1, math.ceil(p95(cores_samples)))


def agent_sessions(p95_agent_gib: float, physical_gib: float) -> int:
    """agent_sessions: POLICY value 2; 1 if 2 * p95 > 0.10 * physical RAM."""
    if AGENT_POLICY * p95_agent_gib > AGENT_RAM_FRACTION * physical_gib:
        return 1
    return AGENT_POLICY


def reserve_gb(
    sessions: int, p95_agent_gib: float, foreign_rss_gib: float = 0.0
) -> float:
    """reserve_gb: agent_sessions * p95 agent RSS + 4.0 GB (+ foreign RSS if the
    servers stay up by ruling), rounded UP to 0.5 GB. Excludes cpu-det memory."""
    x = sessions * p95_agent_gib + OS_MARGIN_GB + foreign_rss_gib
    return math.ceil(x / RESERVE_QUANTUM_GB) * RESERVE_QUANTUM_GB


def conflicts(
    *,
    cpu_det: int,
    k_mem_training: int,
    reserve: float,
    mps_job_peak_gb: float,
    physical_gib: float,
) -> dict[str, dict[str, Any]]:
    """PREREG conflicts (a), (b), (c) -- reported, never resolved by a rule."""
    return {
        "a": {
            "cpu_det_slots": cpu_det,
            "retrieval_curve_slots": RETRIEVAL_CURVE_SLOTS,
            "cpu_det_slots_ge_retrieval_curve": cpu_det >= RETRIEVAL_CURVE_SLOTS,
            "blocks_retrieval_curve": cpu_det < RETRIEVAL_CURVE_SLOTS,
        },
        "b": {
            "k_mem_training": k_mem_training,
            "cpu_det_slots": cpu_det,
            "conflict": k_mem_training < cpu_det,
        },
        "c": {
            "reserve_gb": reserve,
            "mps_job_peak_gb": mps_job_peak_gb,
            "sum": reserve + mps_job_peak_gb,
            "physical_gib": physical_gib,
            "conflict": reserve + mps_job_peak_gb > physical_gib,
        },
    }


@dataclass(frozen=True)
class HashedJob:
    """One determinism-sweep job, as the verdict sees it."""

    threads: int
    k: int
    rep: int
    j: int
    output_hash: str


def determinism_verdict(
    solo_hash: dict[int, str], jobs: Iterable[HashedJob]
) -> tuple[str, dict[str, dict[str, int]], list[dict[str, Any]]]:
    """decision_rule: any job whose output hash differs from the solo run's at the
    same thread count => ``falsified``. Returns (outcome, per-level match counts,
    mismatches)."""
    levels: dict[str, dict[str, int]] = {}
    mismatches: list[dict[str, Any]] = []
    for job in jobs:
        ref = solo_hash[job.threads]
        match = job.output_hash == ref
        lv = levels.setdefault(
            f"t{job.threads}.k{job.k}.rep{job.rep}", {"n_jobs": 0, "n_match": 0}
        )
        lv["n_jobs"] += 1
        lv["n_match"] += int(match)
        if not match:
            mismatches.append({**dataclasses.asdict(job), "solo_hash": ref})
    return ("falsified" if mismatches else "survived"), levels, mismatches


# --------------------------------------------------------------------------- #
# job layout
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class JobSpec:
    part: str  # "det" | "core" | "mps"
    workload: str  # "training" | "core"
    threads: int | None  # None = uncapped (the MPS job)
    k: int
    rep: int
    j: int
    device: str
    iters: int

    @property
    def tag(self) -> str:
        t = "uncapped" if self.threads is None else self.threads
        return f"{self.part}/t{t}/k{self.k}/rep{self.rep}/job{self.j}"


def job_out_dir(root: Path, spec: JobSpec) -> Path:
    """job_isolation: runs/capacity-c0/sweep/<part>/t<threads>/k<k>/rep<r>/job<j>."""
    t = "uncapped" if spec.threads is None else spec.threads
    return (
        root
        / "sweep"
        / spec.part
        / f"t{t}"
        / f"k{spec.k}"
        / f"rep{spec.rep}"
        / f"job{spec.j}"
    )


def _wave(part, workload, threads, k, rep, device, iters) -> list[JobSpec]:
    return [JobSpec(part, workload, threads, k, rep, j, device, iters) for j in range(k)]


def solo_training_wave() -> list[JobSpec]:
    """The first job of the run (brief): its ru_maxrss sets k_mem_training."""
    return _wave("det", "training", 1, 1, 0, "cpu", TRAINING_ITERS)


def det_waves(k_mem_training: int) -> list[list[JobSpec]]:
    """The determinism sweep AFTER the solo wave: 1-thread levels (capped at
    k_mem_training) x 2 reps, then the 5-thread solo and, if 3 <= k_mem_training,
    the 3 concurrent 5-thread jobs."""
    waves = []
    for k in sweep_levels(DET_T1_KS, k_mem_training):
        for rep in range(DET_REPS):
            if (k, rep) == (1, 0):
                continue  # the solo wave, already run
            waves.append(_wave("det", "training", 1, k, rep, "cpu", TRAINING_ITERS))
    waves.append(_wave("det", "training", DET_T5_THREADS, 1, 0, "cpu", TRAINING_ITERS))
    if k_mem_training >= DET_T5_CONCURRENT:
        waves.append(
            _wave(
                "det",
                "training",
                DET_T5_THREADS,
                DET_T5_CONCURRENT,
                0,
                "cpu",
                TRAINING_ITERS,
            )
        )
    return waves


def solo_core_wave() -> list[JobSpec]:
    return _wave("core", "core", 1, 1, 0, "cpu", CORE_ITERS)


def core_waves(k_mem_core: int) -> list[list[JobSpec]]:
    waves = []
    for k in sweep_levels(CORE_SWEEP_K, k_mem_core):
        for rep in range(CORE_REPS):
            if (k, rep) == (1, 0):
                continue
            waves.append(_wave("core", "core", 1, k, rep, "cpu", CORE_ITERS))
    return waves


def mps_wave() -> list[JobSpec]:
    return _wave("mps", "training", None, 1, 0, "mps", MPS_ITERS)


def full_plan(k_mem_training: int, k_mem_core: int) -> list[list[JobSpec]]:
    return [
        solo_training_wave(),
        *det_waves(k_mem_training),
        solo_core_wave(),
        *core_waves(k_mem_core),
        mps_wave(),
    ]


# --------------------------------------------------------------------------- #
# processes, servers and the quiet check
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Proc:
    pid: int
    ppid: int
    rss_bytes: int
    cpu_pct: float
    command: str


def parse_ps(text: str) -> list[Proc]:
    """``ps -axo pid=,ppid=,rss=,%cpu=,command=``: rss is KiB on macOS and Linux."""
    out = []
    for line in text.splitlines():
        parts = line.split(None, 4)
        if len(parts) < 4:
            continue
        try:
            pid, ppid, rss = int(parts[0]), int(parts[1]), int(parts[2])
            cpu = float(parts[3].replace(",", "."))
        except ValueError:
            continue
        out.append(Proc(pid, ppid, rss * 1024, cpu, parts[4] if len(parts) > 4 else ""))
    return out


def list_processes() -> list[Proc]:
    r = subprocess.run(
        ["/bin/ps", "-axo", "pid=,ppid=,rss=,%cpu=,command="],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if r.returncode != 0:
        raise Refusal(f"ps exited {r.returncode}: {r.stderr.strip()}")
    return parse_ps(r.stdout)


def descendants(procs: list[Proc], root_pid: int) -> set[int]:
    """``root_pid`` and every process below it."""
    kids: dict[int, list[int]] = {}
    for p in procs:
        kids.setdefault(p.ppid, []).append(p.pid)
    out, todo = set(), [root_pid]
    while todo:
        pid = todo.pop()
        if pid in out:
            continue
        out.add(pid)
        todo.extend(kids.get(pid, ()))
    return out


def tree_rss_bytes(
    procs: list[Proc], root_pid: int, exclude: set[int] = frozenset()
) -> int:
    tree = descendants(procs, root_pid) - set(exclude)
    return sum(p.rss_bytes for p in procs if p.pid in tree)


def tree_cores(procs: list[Proc], root_pid: int) -> float:
    """``ps %cpu``/100 summed over the tree (mps_reserves_cpu_slots)."""
    tree = descendants(procs, root_pid)
    return sum(p.cpu_pct for p in procs if p.pid in tree) / 100.0


def is_server(command: str) -> bool:
    """The owner's LLM servers, by command line ONLY (the ruling)."""
    return bool(_SERVER_RE.search(command))


def match_servers(procs: list[Proc], own: set[int]) -> list[Proc]:
    return [p for p in procs if p.pid not in own and is_server(p.command)]


def quiet_offenders(
    procs: list[Proc], own: set[int], *, exempt_servers: bool = False
) -> list[Proc]:
    """quiet_machine: every process outside this run with RSS > 2 GB. Only a signed
    STAY UP ruling exempts the servers; nothing else is ever exempt."""
    limit = QUIET_RSS_GIB * GIB
    return [
        p
        for p in procs
        if p.pid not in own
        and p.rss_bytes > limit
        and not (exempt_servers and is_server(p.command))
    ]


def find_agent_pid(procs: list[Proc], start_pid: int) -> int | None:
    """agent_sessions: the nearest ancestor of this run whose executable is
    ``claude`` -- "the claude process running C0"."""
    by_pid = {p.pid: p for p in procs}
    seen: set[int] = set()
    pid = start_pid
    while pid in by_pid and pid not in seen and pid > 1:
        seen.add(pid)
        p = by_pid[pid]
        toks = p.command.split()
        if toks and os.path.basename(toks[0]) == "claude":
            return pid
        pid = p.ppid
    return None


# --------------------------------------------------------------------------- #
# the servers ruling
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ServersRuling:
    id: str | None
    mode: str  # "yield" | "stay_up" | "none"
    text: str


def read_servers_ruling(rulings_dir: Path) -> ServersRuling:
    """The latest SIGNED ``R-*-mlx-servers`` ruling (``workqueue.ruling_problem``),
    and whether its text says the servers YIELD or STAY UP. Both, or neither, is
    refused: the run cannot guess which the owner meant."""
    from orchestrator import workqueue

    found = []
    for path in sorted(rulings_dir.glob("R-*-mlx-servers.md")):
        text = path.read_text()
        problem = workqueue.ruling_problem(path.name, text)
        if problem is not None:
            raise Refusal(f"{path}: servers ruling does not sign: {problem}")
        meta, body = workqueue.split_front_matter(text)
        found.append((str(meta.get("date")), path.stem, body))
    if not found:
        return ServersRuling(None, "none", "")
    found.sort()
    if len(found) > 1 and found[-1][0] == found[-2][0]:
        raise Refusal(
            f"two servers rulings share the latest date {found[-1][0]}: "
            f"{found[-2][1]}, {found[-1][1]}"
        )
    _date, rid, body = found[-1]
    says_yield = re.search(r"\byield", body, re.IGNORECASE) is not None
    says_stay = "STAY UP" in body
    if says_yield == says_stay:
        raise Refusal(
            f"{rid}: cannot tell whether the servers YIELD or STAY UP "
            f"(yield={says_yield}, STAY UP={says_stay})"
        )
    return ServersRuling(rid, "yield" if says_yield else "stay_up", body.strip())


# --------------------------------------------------------------------------- #
# output hash (PREREG output_hash)
# --------------------------------------------------------------------------- #


def state_dict_sha256(state: dict) -> str:
    """sha256 over the state_dict in sorted key order, feeding each key's name,
    dtype, shape and ``t.detach().contiguous().cpu().numpy().tobytes()``."""
    h = hashlib.sha256()
    for name in sorted(state):
        t = state[name]
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(repr(tuple(t.shape)).encode())
        h.update(t.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def output_hash(final_loss: float, state_sha: str) -> str:
    """``repr(final loss)`` + the state_dict sha256. Never the checkpoint FILE."""
    return f"{final_loss!r}:{state_sha}"


# =========================================================================== #
# the measurement
# =========================================================================== #


def _s003():
    """S0-03's run.py, imported (training_workload: "S0-03's CONFIG (imported)")."""
    path = ROOT / "experiments" / "s0-03-rewardable-corpus" / "run.py"
    spec = importlib.util.spec_from_file_location("_s003_run", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def job_main(workload: str, device: str, iters: int, out_dir: Path) -> dict:
    """One job, in its own process. S0-03's ``single()`` -- its ``train()`` call,
    unchanged -- with ``batch`` replaced by 4 for the core workload."""
    import torch

    from rsr.train import checkpoint as ck

    threads = os.environ.get("RSR_TORCH_THREADS")
    if threads is not None:
        torch.set_num_threads(int(threads))
    mod = _s003()
    if workload == "core":
        mod.CONFIG = {**mod.CONFIG, "batch": CORE_BATCH}
    elif workload != "training":
        raise ValueError(f"unknown workload {workload!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    r = mod.single(SEED, iters, device, out_dir)
    ckpt = Path(r["checkpoint"])
    payload = ck.load(ckpt, restore_rng=False, map_location="cpu")
    digest = state_dict_sha256(payload["model"])
    final_loss = r["final"]["loss"]
    return {
        "workload": workload,
        "device": device,
        "iters": iters,
        "batch": mod.CONFIG["batch"],
        "seed": SEED,
        "steps_done": r["steps_done"],
        "config_hash": r["config_hash"],
        "final_loss": final_loss,
        "state_sha256": digest,
        "output_hash": output_hash(final_loss, digest),
        "checkpoint": str(ckpt),
        "torch_num_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
    }


def heartbeat_stats(path: Path) -> dict[str, Any]:
    """repetitions: a job's time = its train-loop time from its own heartbeat,
    first beat to last beat. Also the max ``mem_gb`` (the MPS peak, conflict c)."""
    beats = []
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("kind") == "beat":
            beats.append(rec)
    mem = [b["mem_gb"] for b in beats if b.get("mem_gb") is not None]
    return {
        "n_beats": len(beats),
        "job_time_s": (beats[-1]["elapsed_s"] - beats[0]["elapsed_s"])
        if len(beats) >= 2
        else None,
        "mem_gb_max": max(mem) if mem else None,
    }


@dataclass
class JobResult:
    spec: JobSpec
    out_dir: str
    argv: list[str]
    rc: int | None
    signal: int | None
    maxrss_bytes: int | None
    job_time_s: float | None = None
    n_beats: int = 0
    mem_gb_max: float | None = None
    output_hash: str | None = None
    result: dict = field(default_factory=dict)


def job_argv(spec: JobSpec, out_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--job",
        "--workload",
        spec.workload,
        "--device",
        spec.device,
        "--iters",
        str(spec.iters),
        "--out-dir",
        str(out_dir),
    ]


def job_env(threads: int | None) -> dict[str, str]:
    """``lanes.thread_env(n)`` for a capped job; for the uncapped MPS job every
    thread variable is REMOVED, not set."""
    env = {k: v for k, v in os.environ.items() if k not in lanes.THREAD_ENV_VARS}
    if threads is not None:
        env.update(lanes.thread_env(threads))
    return env


class Wave:
    """Spawn a wave of jobs at once, reap each with ``os.wait4`` (ru_maxrss)."""

    def __init__(
        self,
        root: Path,
        deadline_at: float,
        on_start=None,
        on_end=None,
        now=time.monotonic,
    ):
        self.root = root
        self.deadline_at = deadline_at
        self.now = now
        self.on_start = on_start or (lambda pids: None)
        self.on_end = on_end or (lambda: None)

    def __call__(self, specs: list[JobSpec]) -> list[JobResult]:
        if self.now() >= self.deadline_at:
            raise DeadlineExceeded(f"deadline reached before wave {specs[0].tag}")
        procs: list[tuple[JobSpec, Path, list[str], subprocess.Popen, Any]] = []
        results: dict[int, JobResult] = {}
        try:
            for spec in specs:
                out = job_out_dir(self.root, spec)
                if out.exists():
                    raise Refusal(f"{out} exists: job_isolation needs a fresh out_dir")
                out.mkdir(parents=True)
                argv = job_argv(spec, out)
                fh = open(out / "job.log", "w")  # noqa: SIM115 -- closed below
                p = subprocess.Popen(
                    argv, cwd=ROOT, env=job_env(spec.threads), stdout=fh, stderr=fh
                )
                procs.append((spec, out, argv, p, fh))
            self.on_start([p.pid for *_x, p, _fh in procs])
            pending = {
                p.pid: (spec, out, argv, p, fh) for spec, out, argv, p, fh in procs
            }
            while pending:
                if self.now() >= self.deadline_at:
                    raise DeadlineExceeded(f"deadline reached during wave {specs[0].tag}")
                for pid in list(pending):
                    got, status, ru = os.wait4(pid, os.WNOHANG)
                    if got == 0:
                        continue
                    spec, out, argv, p, fh = pending.pop(pid)
                    p.returncode = os.waitstatus_to_exitcode(status)  # reaped here
                    fh.close()
                    sig = os.WTERMSIG(status) if os.WIFSIGNALED(status) else None
                    rc = os.WEXITSTATUS(status) if os.WIFEXITED(status) else None
                    results[pid] = JobResult(
                        spec, str(out), argv, rc, sig, maxrss_bytes(ru.ru_maxrss)
                    )
                if pending:
                    time.sleep(0.2)
        finally:
            self.on_end()
            for *_x, p, fh in procs:
                if p.returncode is None:
                    try:
                        p.kill()  # this run's OWN child, never a foreign process
                        p.wait(timeout=30)
                    except (OSError, subprocess.TimeoutExpired):
                        pass
                fh.close()
        out_list = [results[p.pid] for *_x, p, _fh in procs]
        for r in out_list:
            if r.signal is not None or r.rc != 0:
                raise JobFailed(
                    f"{r.spec.tag}: rc={r.rc} signal={r.signal} (see {r.out_dir}/job.log)"
                )
            res = Path(r.out_dir) / "result.json"
            hb = Path(r.out_dir) / "heartbeat.jsonl"
            if not res.exists() or not hb.exists():
                raise JobFailed(f"{r.spec.tag}: no result.json or heartbeat.jsonl")
            r.result = json.loads(res.read_text())
            r.output_hash = r.result["output_hash"]
            st = heartbeat_stats(hb)
            r.job_time_s, r.n_beats, r.mem_gb_max = (
                st["job_time_s"],
                st["n_beats"],
                st["mem_gb_max"],
            )
        return out_list


def job_times(rs: list[JobResult]) -> list[float]:
    """The timed jobs' heartbeat times; a job with < 2 beats has no time."""
    missing = [r.spec.tag for r in rs if r.job_time_s is None]
    if missing:
        raise JobFailed(f"{missing}: fewer than two heartbeat beats, so no job time")
    return [r.job_time_s for r in rs]


def time_suite(n: int, deadline_at: float) -> dict[str, Any]:
    """battery_cpu_slots: W(n) = ``pytest -rs --tb=no`` wall time, threads set the
    way ``suite_threads()`` sets them (``lanes.thread_env(n)``), and the census
    redirected as the battery redirects it (never ``test-count.json``)."""
    remaining = deadline_at - time.monotonic()
    if remaining <= 0:
        raise DeadlineExceeded(f"deadline reached before suite n={n}")
    env = {**job_env(n), "RSR_TEST_COUNT": "runs/.c0-suite-census.json"}
    argv = [str(ROOT / ".venv" / "bin" / "pytest"), "-rs", "--tb=no"]
    t0 = time.monotonic()
    try:
        r = subprocess.run(
            argv, cwd=ROOT, env=env, capture_output=True, text=True, timeout=remaining
        )
    except subprocess.TimeoutExpired:
        raise DeadlineExceeded(f"deadline reached during suite n={n}") from None
    wall = time.monotonic() - t0
    census = [ln for ln in r.stdout.splitlines() if re.search(r"\d+ (passed|failed)", ln)]
    if r.returncode != 0:
        raise JobFailed(f"suite at n={n} exited {r.returncode}: {census[-1:]}")
    return {"n": n, "wall_s": wall, "rc": r.returncode, "census": census[-1:]}


class Sampler(threading.Thread):
    """1 Hz ``ps``: the agent tree's RSS for the whole run, and the MPS job's tree
    ``%cpu``/100 while it runs."""

    def __init__(self, agent_pid: int, own_pid: int, procs_fn=list_processes):
        super().__init__(daemon=True)
        self.agent_pid, self.own_pid, self.procs_fn = agent_pid, own_pid, procs_fn
        self.agent_rss: list[int] = []
        self.mps_cores: list[float] = []
        self.watch: list[int] = []
        self._halt = threading.Event()  # not `_stop`: Thread defines `_stop()`

    def sample_once(self) -> None:
        procs = self.procs_fn()
        own = descendants(procs, self.own_pid)
        self.agent_rss.append(tree_rss_bytes(procs, self.agent_pid, exclude=own))
        for pid in list(self.watch):
            self.mps_cores.append(tree_cores(procs, pid))

    def run(self) -> None:
        while not self._halt.wait(1.0 / SAMPLE_HZ):
            # a failed sample is a missing sample, never a crash
            with contextlib.suppress(Exception):
                self.sample_once()

    def stop(self) -> None:
        self._halt.set()


# --------------------------------------------------------------------------- #
# servers: stop, record, restart
# --------------------------------------------------------------------------- #


def _cwd_of(pid: int) -> str | None:
    try:
        r = subprocess.run(
            ["/usr/sbin/lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in r.stdout.splitlines():
        if line.startswith("n"):
            return line[1:]
    return None


def spawn_server(record: dict, log_dir: Path) -> int:
    """Restart one stopped server from its recorded command line and cwd, detached
    (its own session), with this process's environment."""
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = open(log_dir / f"restart-{record['pid']}.log", "a")  # noqa: SIM115
    try:
        argv = shlex.split(record["command"])
    except ValueError:
        argv = record["command"].split()
    p = subprocess.Popen(
        argv,
        cwd=record.get("cwd") or str(Path.home()),
        stdin=subprocess.DEVNULL,
        stdout=fh,
        stderr=fh,
        start_new_session=True,
    )
    fh.close()
    return p.pid


@dataclass
class System:
    """Everything that touches the machine, injectable so a test never does."""

    procs: Callable[[], list[Proc]] = list_processes
    kill: Callable[[int, int], None] = os.kill
    spawn_server: Callable[[dict, Path], int] = spawn_server
    cwd_of: Callable[[int], str | None] = _cwd_of
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], float] = time.monotonic
    own_pid: int = field(default_factory=os.getpid)
    memsize: Callable[[], int] = lambda: int(_sysctl("hw.memsize"))
    free_bytes: Callable[[], int] = lanes.free_memory_bytes


def _sysctl(name: str) -> str:
    r = subprocess.run(["/usr/sbin/sysctl", "-n", name], capture_output=True, text=True)
    if r.returncode != 0:
        raise Refusal(f"sysctl {name} exited {r.returncode}: {r.stderr.strip()}")
    return r.stdout.strip()


def stop_servers(sysm: System, records: list[dict]) -> list[dict]:
    """Stop the owner's LLM servers (SIGTERM, then SIGKILL after a grace), appending
    to ``records`` -- the caller's list, so the record survives a refusal half-way:
    PID, command line, cwd, RSS freed, how it was stopped. Refuses if a server
    survives or respawns (a supervisor would fight the run)."""
    procs = sysm.procs()
    own = descendants(procs, sysm.own_pid)
    servers = match_servers(procs, own)
    records.extend(
        {
            "pid": p.pid,
            "ppid": p.ppid,
            "command": p.command,
            "cwd": sysm.cwd_of(p.pid),
            "rss_bytes": p.rss_bytes,
            "rss_gib": p.rss_bytes / GIB,
            "signal": "SIGTERM",
            "stopped": False,
        }
        for p in servers
    )
    if not records:
        return records
    for rec in records:
        sysm.kill(rec["pid"], signal.SIGTERM)

    def alive() -> set[int]:
        now = {p.pid for p in sysm.procs()}
        return {r["pid"] for r in records} & now

    for grace, sig in ((STOP_GRACE_S, signal.SIGKILL), (KILL_GRACE_S, None)):
        t0 = sysm.now()
        while alive() and sysm.now() - t0 < grace:
            sysm.sleep(0.5)
        left = alive()
        if not left:
            break
        if sig is None:
            for rec in records:
                rec["stopped"] = rec["pid"] not in left
            raise Refusal(f"servers {sorted(left)} did not stop after SIGKILL")
        for rec in records:
            if rec["pid"] in left:
                sysm.kill(rec["pid"], sig)
                rec["signal"] = "SIGTERM then SIGKILL"
    for rec in records:
        rec["stopped"] = True
    t0 = sysm.now()
    while sysm.now() - t0 < RESPAWN_WATCH_S:
        sysm.sleep(1.0)
    procs = sysm.procs()
    again = match_servers(procs, descendants(procs, sysm.own_pid))
    if again:
        raise Refusal(
            f"servers respawned after being stopped (pids {[p.pid for p in again]}): "
            f"something supervises them; C0 cannot measure a quiet machine"
        )
    return records


def restart_servers(sysm: System, stopped: list[dict], log_dir: Path) -> list[dict]:
    """Restart every server this run stopped, unless one with the same command line
    is already running. Never touches any other process."""
    out = []
    if not stopped:
        return out
    running = {p.command for p in sysm.procs()}
    for rec in stopped:
        if not rec.get("stopped"):
            out.append(
                {"pid_was": rec["pid"], "restarted": False, "why": "never stopped"}
            )
            continue
        if rec["command"] in running:
            out.append(
                {"pid_was": rec["pid"], "restarted": False, "why": "already running"}
            )
            continue
        try:
            pid = sysm.spawn_server(rec, log_dir)
            out.append({"pid_was": rec["pid"], "restarted": True, "new_pid": pid})
        except Exception as e:  # keep restarting the others
            out.append({"pid_was": rec["pid"], "restarted": False, "why": repr(e)})
    return out


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #


def corpus_sha256() -> str:
    """sha256 of the workload corpus: S0-03's training documents at seed 0."""
    from rsr.data.synthetic import SyntheticConfig, generate

    S = _s003().CONFIG["steps_per_stream"]
    docs = generate(SyntheticConfig(sentences_per_document=S, seed=SEED))
    blob = json.dumps([dataclasses.asdict(d) for d in docs], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def machine() -> dict[str, Any]:
    names = (
        "hw.model",
        "hw.memsize",
        "hw.ncpu",
        "hw.perflevel0.physicalcpu",
        "hw.perflevel1.physicalcpu",
        "machdep.cpu.brand_string",
        "kern.osproductversion",
    )
    vals = {}
    for n in names:
        try:
            vals[n] = _sysctl(n)
        except Refusal as e:
            vals[n] = f"<{e}>"
    vals["summary"] = (
        f"sysctl hw.model {vals['hw.model']}, {vals['machdep.cpu.brand_string']}, "
        f"hw.perflevel0.physicalcpu {vals['hw.perflevel0.physicalcpu']}, "
        f"hw.perflevel1.physicalcpu {vals['hw.perflevel1.physicalcpu']}, "
        f"hw.memsize {vals['hw.memsize']}"
    )
    return vals


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #


@dataclass
class Measure:
    """The four measurement callables the run needs, injectable for tests."""

    wave: Callable[[list[JobSpec]], list[JobResult]]
    suite: Callable[[int], dict[str, Any]]
    sampler: Any  # .agent_rss, .mps_cores, .watch, .sample_once()


def _job_row(r: JobResult) -> dict[str, Any]:
    return {
        "tag": r.spec.tag,
        "out_dir": r.out_dir,
        "rc": r.rc,
        "maxrss_bytes": r.maxrss_bytes,
        "job_time_s": r.job_time_s,
        "n_beats": r.n_beats,
        "output_hash": r.output_hash,
        "torch_num_threads": r.result.get("torch_num_threads"),
        "mem_gb_max": r.mem_gb_max,
    }


def measure(led, measure_: Measure, *, memsize: int, stay_up_foreign_gib: float | None):
    """Steps 2-6 of the module docstring. Writes rows as it goes, so a partial
    ledger holds everything measured before the stop. Returns the Exit."""
    physical_gib = memsize / GIB
    led.note("c0.memsize_bytes", memsize, how="sysctl hw.memsize")
    led.note("c0.physical_gib", physical_gib, how="sysctl hw.memsize / 2^30")
    jobs_log: list[dict] = []
    led.note(
        "c0.jobs", jobs_log, how="every job: out_dir, rc, os.wait4 ru_maxrss, heartbeat"
    )

    def run(specs):
        rs = measure_.wave(specs)
        jobs_log.extend(_job_row(r) for r in rs)
        for r in rs:
            led.command(
                " ".join([".venv/bin/python", EXPERIMENT, *r.argv[2:]]),
                exit_code=r.rc,
                note=r.spec.tag,
            )
        return rs

    # -- 2. determinism sweep (falsifier first) ------------------------------ #
    solo = run(solo_training_wave())[0]
    r_training = solo.maxrss_bytes
    kmt = k_mem(memsize, r_training)
    led.note(
        "c0.r_job_gib.training",
        r_training / GIB,
        how="solo 1-thread training job, os.wait4 ru_maxrss",
    )
    led.note("c0.k_mem_training", kmt, how="floor((hw.memsize/2^30 - 8) / R_training)")
    solo_hash = {1: solo.output_hash}
    hashed = [HashedJob(1, 1, 0, 0, solo.output_hash)]
    for wave in det_waves(kmt):
        rs = run(wave)
        if wave[0].threads == DET_T5_THREADS and wave[0].k == 1:
            solo_hash[DET_T5_THREADS] = rs[0].output_hash
        hashed += [
            HashedJob(r.spec.threads, r.spec.k, r.spec.rep, r.spec.j, r.output_hash)
            for r in rs
        ]
    outcome, levels, mismatches = determinism_verdict(solo_hash, hashed)
    led.note(
        "c0.det.solo_hash",
        {f"t{t}": h for t, h in solo_hash.items()},
        how="PREREG output_hash of the solo job per thread count",
    )
    led.note(
        "c0.det.levels",
        levels,
        how="match count per (threads, k, rep) vs the solo hash at the same thread count",
    )
    led.note("c0.det.n_jobs", len(hashed), how="len(determinism-sweep jobs)")
    led.note("c0.det.n_mismatch", len(mismatches), how="len(mismatches)")
    led.note("c0.det.mismatches", mismatches, how="determinism_verdict")
    led.note(
        "c0.det.t1_equals_t5",
        solo_hash.get(1) == solo_hash.get(DET_T5_THREADS),
        how="secondary: not a rule",
    )
    led.note("c0.verdict", outcome, how="PREREG decision_rule")
    if outcome == "falsified":
        led.verdict(
            falsifier=_FALSIFIER,
            outcome="falsified",
            detail=(
                f"{len(mismatches)} of {len(hashed)} determinism-sweep jobs differ "
                f"from the solo hash; no capacity key written"
            ),
        )
        return Exit.FAIL

    # -- 3. core sweep ------------------------------------------------------- #
    csolo = run(solo_core_wave())[0]
    r_core = csolo.maxrss_bytes
    kmc = k_mem(memsize, r_core)
    led.note(
        "c0.r_job_gib.core",
        r_core / GIB,
        how="solo 1-thread core job, os.wait4 ru_maxrss",
    )
    led.note("c0.k_mem_core", kmc, how="floor((hw.memsize/2^30 - 8) / R_core)")
    times: dict[int, list[list[float]]] = {1: [job_times([csolo])]}
    for wave in core_waves(kmc):
        rs = run(wave)
        times.setdefault(wave[0].k, []).append(job_times(rs))
    spans = {k: makespan(reps) for k, reps in times.items()}
    led.note(
        "c0.core.job_times_s",
        {f"k{k}": v for k, v in times.items()},
        how="heartbeat first->last beat, per rep",
    )
    slots, table = cpu_det_slots(spans)
    led.note(
        "c0.core.table",
        table,
        how="cpu_det_slots rule, T(k) = k / makespan(k), slower repetition",
    )

    # -- 4. suite timings ---------------------------------------------------- #
    W = {}
    for n in BATTERY_THREADS:
        s = measure_.suite(n)
        W[n] = s["wall_s"]
        led.note(
            f"c0.battery.n{n}",
            s,
            how="pytest -rs --tb=no wall time under lanes.thread_env(n)",
        )
    b_raw = battery_cpu_slots_raw(W)

    # -- 5. MPS job ---------------------------------------------------------- #
    mps = run(mps_wave())[0]
    cores = list(measure_.sampler.mps_cores)
    led.note("c0.mps.cores_samples", cores, how="ps %cpu/100 of the MPS job's tree, 1 Hz")
    led.note(
        "c0.mps.maxrss_gib", mps.maxrss_bytes / GIB, how="os.wait4 ru_maxrss, secondary"
    )
    if not cores:
        raise JobFailed("the MPS job produced no %cpu samples")
    if mps.mem_gb_max is None:
        raise JobFailed("the MPS job's heartbeat has no mem_gb")
    m_raw = mps_reserves_cpu_slots_raw(cores)
    mps_peak = mps.mem_gb_max

    # -- 6. agent feasibility, reserve, the five keys ------------------------ #
    measure_.sampler.sample_once()
    agent = [b / GIB for b in measure_.sampler.agent_rss]
    agent_p95 = p95(agent)
    sessions = agent_sessions(agent_p95, physical_gib)
    foreign = stay_up_foreign_gib or 0.0
    reserve = reserve_gb(sessions, agent_p95, foreign)

    led.note(
        "c0.agent.rss_gib_p95",
        agent_p95,
        how="nearest-rank p95 of the agent tree RSS (C0's own tree excluded), 1 Hz",
    )
    led.note("c0.agent.n_samples", len(agent), how="len(agent RSS samples)")
    led.note("c0.cpu_det_slots", slots, how="cpu_det_slots rule over c0.core.table")
    led.note(
        "c0.battery_cpu_slots_raw", b_raw, how="smallest n with W(n) <= 1.15 * min W"
    )
    led.note(
        "c0.battery_cpu_slots",
        clamp_to(b_raw, slots),
        how="battery_cpu_slots_raw clamped to cpu_det_slots",
    )
    led.note(
        "c0.mps.cores_p95", p95(cores), how="nearest-rank p95 of c0.mps.cores_samples"
    )
    led.note("c0.mps_reserves_cpu_slots_raw", m_raw, how="max(1, ceil(p95))")
    led.note(
        "c0.mps_reserves_cpu_slots",
        clamp_to(m_raw, slots),
        how="raw clamped to cpu_det_slots",
    )
    led.note(
        "c0.mps_job_peak_gb",
        mps_peak,
        how="max heartbeat mem_gb (mps driver_allocated_memory / 1e9), C0's own MPS run",
    )
    led.note(
        "c0.agent_sessions",
        sessions,
        how="policy 2; 1 if 2 * p95 agent RSS > 0.10 * physical RAM",
    )
    led.note(
        "c0.reserve_gb",
        reserve,
        how="agent_sessions * p95 agent RSS + 4.0 (+ foreign if STAY UP), up to 0.5",
    )
    for name, row in conflicts(
        cpu_det=slots,
        k_mem_training=kmt,
        reserve=reserve,
        mps_job_peak_gb=mps_peak,
        physical_gib=physical_gib,
    ).items():
        led.note(
            f"c0.conflict.{name}",
            row,
            how="PREREG conflicts: reported, never resolved by a rule",
        )
    led.verdict(
        falsifier=_FALSIFIER,
        outcome="survived",
        detail=(
            f"all {len(hashed)} determinism-sweep jobs match the solo hash at "
            f"their thread count"
        ),
    )
    return Exit.OK


_FALSIFIER = (
    "A deterministic CPU training job on the Mac Studio produces bit-identical output "
    "no matter how many other such jobs run beside it, at a fixed per-job thread "
    "count (1 and 5)."
)


def execute(
    led,
    sysm: System,
    make_measure: Callable[[float], Measure],
    *,
    ruling: ServersRuling,
    log_dir: Path,
) -> Exit:
    """Servers -> quiet check -> measure -> ledger, with the servers restarted in a
    ``finally``. ``make_measure(deadline_at)`` builds the measurement."""
    led.note(
        "c0.servers_ruling",
        {"id": ruling.id, "mode": ruling.mode, "text": ruling.text},
        how="docs/owner/rulings/R-*-mlx-servers.md, latest signed",
    )
    stopped: list[dict] = []
    code = Exit.DID_NOT_RUN
    try:
        free_before = sysm.free_bytes()
        led.note(
            "c0.stopped_servers",
            stopped,
            how="R-2026-09-22-mlx-servers: PIDs, command lines, RSS freed",
        )
        if ruling.mode == "yield":
            stop_servers(sysm, stopped)
        led.note(
            "c0.stopped_servers_rss_gib",
            sum(r["rss_gib"] for r in stopped),
            how="sum of rss_gib over c0.stopped_servers",
        )
        led.note(
            "c0.free_gib.start",
            free_before / GIB,
            how="lanes.free_memory_bytes before any server stop",
        )
        led.note(
            "c0.free_gib.after_stop",
            sysm.free_bytes() / GIB,
            how="lanes.free_memory_bytes after the stop",
        )
        procs = sysm.procs()
        own = descendants(procs, sysm.own_pid)
        offenders = quiet_offenders(procs, own, exempt_servers=ruling.mode == "stay_up")
        led.note(
            "c0.quiet.offenders",
            [
                {"pid": p.pid, "rss_gib": p.rss_bytes / GIB, "command": p.command}
                for p in offenders
            ],
            how="processes outside this run with RSS > 2 GiB at start",
        )
        foreign = None
        if ruling.mode == "stay_up":
            foreign = sum(p.rss_bytes for p in match_servers(procs, own)) / GIB
            led.note(
                "c0.foreign_rss_gb",
                foreign,
                how="RSS of the servers left up by ruling, at start (GiB)",
            )
        if offenders:
            raise Refusal(
                "machine not quiet: "
                + "; ".join(
                    f"pid {p.pid} {p.rss_bytes / GIB:.2f} GiB {p.command[:80]}"
                    for p in offenders
                )
            )
        deadline_at = sysm.now() + DEADLINE_HOURS * 3600
        led.note("c0.deadline_hours", DEADLINE_HOURS, how="PREREG deadline_hours")
        measure_ = make_measure(deadline_at)
        try:
            code = measure(
                led, measure_, memsize=sysm.memsize(), stay_up_foreign_gib=foreign
            )
            led.status("ok")
        except BaseException as e:
            # PREREG decision_rule: any stop before completion is "partial".
            led.status("partial")
            led.note(
                "c0.stopped_because",
                f"{type(e).__name__}: {e}",
                how="the exception that stopped the run",
            )
            led.verdict(
                falsifier=_FALSIFIER,
                outcome="inconclusive",
                detail=f"partial: {type(e).__name__}: {e}",
            )
            raise
    except BaseException as e:
        if led.doc.get("status") is None:  # refused before the measurement began
            led.status("did_not_run")
            led.note("c0.stopped_because", f"{type(e).__name__}: {e}", how="the refusal")
            led.verdict(
                falsifier=_FALSIFIER, outcome="inconclusive", detail=f"did not run: {e}"
            )
        raise
    finally:
        restarted = restart_servers(sysm, stopped, log_dir)
        led.note("c0.restarted_servers", restarted, how="restart_servers, in a finally")
        led.write()
    return code


# =========================================================================== #
# RESULTS.md, rendered from the ledger
# =========================================================================== #


def _row(rows: dict, key: str) -> Any:
    return rows[key]["value"] if key in rows else None


def _yn(b: bool | None) -> str:
    return "—" if b is None else ("yes" if b else "no")


def render_results(doc: dict) -> str:
    """RESULTS.md from the ledger: every number sits beside its ledger key, so
    ``render_scoreboard.py --audit`` can back it (done_when)."""
    rows = {r["key"]: r for r in doc.get("rows", [])}
    prov = doc.get("provenance", {})
    mach = _row(rows, "c0.machine") or {}
    out = [
        "# C0 -- RESULTS",
        "",
        "<!-- GENERATED by `experiments/capacity-c0/run.py --render-results` from "
        "runs/capacity-c0/ledger.json. Do not edit: regenerate. -->",
        "",
        f"Command: `uv run python {EXPERIMENT}`, then "
        f"`uv run python {EXPERIMENT} --render-results`. Ledger `status`: "
        f"**{doc.get('status')}**.",
        "",
        # The run id is named so the audit reads the quoted sysctl string as this
        # ledger's text (render_scoreboard._quoted), not as typed numbers.
        f"Provenance of run `{doc.get('run_id')}`: "
        f"`provenance.git_sha` `{prov.get('git_sha')}` · "
        f"`{mach.get('summary')}` · "
        f"`c0.corpus_sha256` `{_row(rows, 'c0.corpus_sha256')}` · "
        f"`seeds_actually_run` {doc.get('seeds_actually_run')}.",
        "",
        f"## Verdict: `c0.verdict` **{_row(rows, 'c0.verdict')}**",
        "",
        f"`c0.det.n_mismatch` {_row(rows, 'c0.det.n_mismatch')} of `c0.det.n_jobs` "
        f"{_row(rows, 'c0.det.n_jobs')} determinism-sweep jobs differ from the solo "
        f"hash.",
        "",
        "## The five capacities",
        "",
        "| ledger key | value |",
        "|---|---|",
    ]
    for key in lanes.C0_LEDGER_KEYS.values():
        out.append(f"| `{key}` | {_row(rows, key) if key in rows else 'not written'} |")
    out += [
        "",
        "## Memory caps and raw values",
        "",
        "| ledger key | value |",
        "|---|---|",
    ]
    for key in (
        "c0.r_job_gib.training",
        "c0.k_mem_training",
        "c0.r_job_gib.core",
        "c0.k_mem_core",
        "c0.battery_cpu_slots_raw",
        "c0.mps_reserves_cpu_slots_raw",
        "c0.mps_job_peak_gb",
        "c0.agent.rss_gib_p95",
        "c0.stopped_servers_rss_gib",
    ):
        if key in rows:
            out.append(f"| `{key}` | {_row(rows, key)} |")
    out += ["", "## Conflicts (reported, never resolved by a rule)", ""]
    a, b, c = (_row(rows, f"c0.conflict.{x}") for x in "abc")
    if a:
        out.append(
            f"- (a) `c0.conflict.a.cpu_det_slots_ge_retrieval_curve`: "
            f"**{_yn(a['cpu_det_slots_ge_retrieval_curve'])}**; "
            f"`c0.conflict.a.cpu_det_slots` "
            f"{a['cpu_det_slots']} against `c0.conflict.a.retrieval_curve_slots` "
            f"{a['retrieval_curve_slots']}."
        )
    if b:
        out.append(
            f"- (b) `c0.conflict.b.conflict` (k_mem_training < cpu_det_slots): "
            f"**{_yn(b['conflict'])}**; "
            f"`c0.conflict.b.k_mem_training` {b['k_mem_training']} "
            f"against `c0.conflict.b.cpu_det_slots` {b['cpu_det_slots']}."
        )
    if c:
        out.append(
            f"- (c) `c0.conflict.c.conflict` (reserve_gb + mps_job_peak_gb > physical): "
            f"**{_yn(c['conflict'])}**; `c0.conflict.c.sum` {c['sum']} against "
            f"`c0.conflict.c.physical_gib` {c['physical_gib']}."
        )
    if not (a or b or c):
        out.append("None computed: the run did not reach the rules.")
    out += ["", "## BRIEF ERRORS", "", "Written by the executor after the run.", ""]
    return "\n".join(out)


# =========================================================================== #
# --dry-plan
# =========================================================================== #


def dry_plan(memsize: int) -> str:
    """The waves and a PREDICTED duration. Runs nothing; the k_mem values are
    predictions from the PREREG's quoted RSS, and the run recomputes them."""
    kmt = k_mem(memsize, int(PRED["r_training_gib"] * GIB))
    kmc = k_mem(memsize, int(PRED["r_core_gib"] * GIB))
    waves = full_plan(kmt, kmc)
    lines = [
        f"C0 dry plan (NOTHING RUN). hw.memsize {memsize} -> physical "
        f"{memsize / GIB:.1f} GiB",
        f"predicted k_mem_training {kmt} (R ~{PRED['r_training_gib']} GiB), "
        f"k_mem_core {kmc} (R ~{PRED['r_core_gib']} GiB) -- recomputed from ru_maxrss",
        "",
    ]
    total = 0.0
    n_jobs = 0
    suite = PRED["suite_s"] * len(BATTERY_THREADS)
    for i, w in enumerate(waves, 1):
        s = w[0]
        if s.part == "mps":  # the run times the suite between the core sweep and MPS
            total += suite
            lines.append(
                f"suite timings: pytest -rs --tb=no at threads {list(BATTERY_THREADS)}"
                f"  ~{suite:.0f} s"
            )
        if s.workload == "training" and s.device == "cpu":
            per = PRED["s_per_iter_training"] * s.iters
        elif s.workload == "core":
            per = PRED["s_per_sample_core"] * CORE_BATCH * s.iters
        else:
            per = PRED["s_per_iter_mps"] * s.iters
        load = len(w) * (s.threads or 1)
        wall = PRED["startup_s"] + per * max(1.0, load / PRED["perf_cores"])
        total += wall
        n_jobs += len(w)
        lines.append(
            f"wave {i:3d}: {len(w):2d} x {s.part:4s} {s.workload:8s} dev={s.device} "
            f"threads={s.threads or 'uncapped'} iters={s.iters} rep={s.rep}  "
            f"~{wall:5.0f} s  "
            f"-> {s.tag.rsplit('/job', 1)[0]}/job*"
        )
    lines += [
        "",
        f"{len(waves)} waves, {n_jobs} jobs (+{len(BATTERY_THREADS)} suite runs).",
        f"PREDICTED duration ~{total / 60:.0f} min (deadline {DEADLINE_HOURS} h). Brief: "
        f"'Expect under 1.5 h'. Prediction inputs: {PRED}.",
    ]
    return "\n".join(lines)


# =========================================================================== #
# CLI
# =========================================================================== #


def _real_measure(root: Path, sampler: Sampler) -> Callable[[float], Measure]:
    """The real measurement: jobs as subprocesses, the MPS job's tree watched by the
    1 Hz sampler while it runs."""

    def make(deadline_at: float) -> Measure:
        def wave(specs: list[JobSpec]) -> list[JobResult]:
            mps = specs[0].device == "mps"
            runner = Wave(
                root,
                deadline_at,
                on_start=(lambda pids: sampler.watch.extend(pids)) if mps else None,
                on_end=sampler.watch.clear,
            )
            return runner(specs)

        return Measure(
            wave=wave, suite=lambda n: time_suite(n, deadline_at), sampler=sampler
        )

    return make


def main(argv: list[str] | None = None) -> Exit:
    ap = ArgumentParser(prog="capacity-c0")
    ap.add_argument(
        "--dry-plan", action="store_true", help="print the waves; run nothing"
    )
    ap.add_argument(
        "--render-results", action="store_true", help="RESULTS.md from the ledger"
    )
    ap.add_argument("--run-id", default=RUN_ID)
    ap.add_argument(
        "--agent-pid", type=int, default=None, help="the claude process running C0"
    )
    # internal: one job
    ap.add_argument("--job", action="store_true", help=argparse_suppress())
    ap.add_argument("--workload", choices=("training", "core"), default="training")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--iters", type=int, default=TRAINING_ITERS)
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args(argv)

    if a.job:
        if a.out_dir is None:
            refuse(Exit.DID_NOT_RUN, "--job needs --out-dir")
        r = job_main(a.workload, a.device, a.iters, a.out_dir)
        (a.out_dir / "result.json").write_text(
            json.dumps(r, indent=2, default=str) + "\n"
        )
        return Exit.OK

    if a.dry_plan:
        print(dry_plan(int(_sysctl("hw.memsize"))))
        return Exit.OK

    root = ROOT / "runs" / a.run_id
    if a.render_results:
        path = root / "ledger.json"
        if not path.exists():
            refuse(Exit.DID_NOT_RUN, f"{path} is absent: C0 has not run")
        out = ROOT / "experiments" / "capacity-c0" / "RESULTS.md"
        out.write_text(render_results(json.loads(path.read_text())))
        print(f"wrote {out}")
        return Exit.OK

    return status(_run(a, root))


def argparse_suppress():
    import argparse

    return argparse.SUPPRESS


def _run(a, root: Path) -> Exit:
    import ledger as ledger_mod

    try:
        if sys.platform != "darwin":
            raise Refusal("C0 measures the Mac Studio (ADR-0007); this is not macOS")
        dirty = ledger_mod._dirty_source_paths()
        if dirty:
            raise Refusal(
                f"uncommitted source {dirty}: the ledger would refuse at the end"
            )
        if root.exists():
            raise Refusal(
                f"{root} exists. job_isolation needs fresh out_dirs: move it aside "
                f"(e.g. {root}.attempt-N) and re-run"
            )
        ruling = read_servers_ruling(ROOT / "docs" / "owner" / "rulings")
        sysm = System()
        procs = sysm.procs()
        agent_pid = a.agent_pid or find_agent_pid(procs, sysm.own_pid)
        if agent_pid is None:
            raise Refusal(
                "no `claude` ancestor process found: agent_sessions samples the claude "
                "process running C0. Pass --agent-pid PID"
            )
    except Refusal as e:
        refuse(Exit.DID_NOT_RUN, str(e))

    led = ledger_mod.Ledger(
        a.run_id, question="What lane capacities does this machine measure? (PREREG C0)"
    )
    led.command(
        " ".join(["uv run python", EXPERIMENT, *sys.argv[1:]]),
        exit_code=None,
        note="this run",
    )
    led.manifest(
        {
            "prereg": PREREG,
            "seed": SEED,
            "training": {"iters": TRAINING_ITERS, "config": "S0-03 CONFIG (imported)"},
            "core": {"iters": CORE_ITERS, "batch": CORE_BATCH},
            "det_t1_ks": DET_T1_KS,
            "det_reps": DET_REPS,
            "det_t5": [DET_T5_THREADS, DET_T5_CONCURRENT],
            "core_sweep_k": CORE_SWEEP_K,
            "core_reps": CORE_REPS,
            "battery_threads": BATTERY_THREADS,
            "mps_iters": MPS_ITERS,
            "deadline_hours": DEADLINE_HOURS,
            "agent_pid": agent_pid,
            "s003_config": _s003().CONFIG,
        }
    )
    led.run_meta(device="cpu+mps", seeds_actually_run=[SEED])
    led.note("c0.machine", machine(), how="sysctl")
    led.note(
        "c0.corpus_sha256",
        corpus_sha256(),
        how="sha256 of S0-03's training documents, seed 0",
    )
    led.note(
        "c0.agent.pid", agent_pid, how="--agent-pid, else the nearest `claude` ancestor"
    )

    sampler = Sampler(agent_pid, sysm.own_pid)
    sampler.start()
    try:
        code = execute(
            led,
            sysm,
            _real_measure(root, sampler),
            ruling=ruling,
            log_dir=root / "servers",
        )
    except Refusal as e:
        refuse(Exit.DID_NOT_RUN, str(e))
    except (DeadlineExceeded, JobFailed, Exception, KeyboardInterrupt) as e:
        traceback.print_exc()
        refuse(
            Exit.DID_NOT_RUN,
            f"C0 stopped before completion ({type(e).__name__}); ledger status partial",
        )
    finally:
        sampler.stop()
    print(f"ledger: {led.path}  status={led.doc['status']}  exit={int(code)}")
    return code


if __name__ == "__main__":
    run_main(main)
