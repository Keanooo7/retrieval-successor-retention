"""Scaffold timing -- does held-out retrieval precede or follow memorisation at N = 64?

Pre-registration: `experiments/scaffold-timing/PREREG.md`, committed alone at
167d650, ahead of this file. **Measurement only -- nothing is trained.** It reads the
corpus-size curve's saved checkpoints (gitignored, read-only, in the corpus-curve
worktree's `runs/corpus-size-curve/<nN>/seed<s>/ckpt-000{100..1000}.pt`, written by
the run at ec9332d) and measures each with **the corpus-size curve's own
``measure_checkpoint``, imported** -- S0-03's ``measure(sets=doc_sets(seed, N))``
on held-out documents [4096, 4160) and the training probe [0, 64).

Order (PREREG "The reproduction control (checked first; overrides everything)"):

1. **The reproduction control.** N = 64 at ckpt 300 and ckpt 1000, every seed. The
   re-measured tables are turned into ledger rows by the corpus-size curve's own
   ``write_rows`` (imported, so the key naming cannot drift), and **every**
   ``n64.ckpt300.*`` / ``n64.ckpt1000.*`` statistic key of
   `runs/corpus-size-curve/ledger.json` must match per seed within ``REPRO_TOL``.
   A failure, a missing key or a raise: exit 3, and **nothing further is measured**.
2. Every other (arm, seed, checkpoint): N = 64 first, then N = 512, then N = 4096.
3. The rule (N = 64 only): ``R(c)``, ``M(c)``, the sustained onsets and the
   BEFORE / WITH / AFTER / inconclusive table, row for row. Secondary readouts for
   every arm are recorded; none enters the classification.

Execution modes. **Sequential (default):** in this process at the corpus-size
parent's own torch thread count (its manifest's ``parent_torch_num_threads``), which
is how the ledger's numbers were produced. ``--parallel``: one child process per
seed at ``THREADS_PER_CHILD`` threads (12 CPU slots / 3 seeds). Whether a different
thread count reproduces the ledger within 1e-6 has NOT been measured; the control
decides it, and a failed control is exit 3, never a reason to re-run in the other
mode (PREREG: "Do not tune to make it match").

Usage::

    uv run --extra dev python experiments/scaffold-timing/run.py --dry-run
    uv run --extra dev python experiments/scaffold-timing/run.py
    uv run --extra dev python experiments/scaffold-timing/run.py --render-results

Exit codes (`rsr.exit_codes`): 0 a classification was reached (including the
PREREG's "inconclusive" when an onset does not exist) · 3 did not run or did not
complete (the reproduction control failed or was incomplete, a measurement raised,
a refused precondition).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rsr.exit_codes import ArgumentParser, Exit, refuse, run_main, status  # noqa: E402

EXPERIMENT = "experiments/scaffold-timing/run.py"
RUN_ID = "scaffold-timing"
PREREG = "experiments/scaffold-timing/PREREG.md"
PREREG_COMMIT = "167d650"
BRIEF_ERRORS = "experiments/scaffold-timing/BRIEF-ERRORS.md"


def _load(name: str, rel: str):
    """Import an experiment script by path. Imported, never copied."""
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CSC = _load("_scaffold_timing_csc", "experiments/corpus-size-curve/run.py")
RC = CSC.RC
S003 = CSC.S003

#: THE measurement: the corpus-size curve's function object, not a copy. PREREG
#: "instrument"; `tests/test_scaffold_timing.py` checks the identity.
measure_checkpoint = CSC.measure_checkpoint

# --------------------------------------------------------------------------- #
# 🔒 PREREG.md front matter, transcribed. Changing any is changing the PREREG.
# --------------------------------------------------------------------------- #

SEEDS = [0, 1, 2]
PRIMARY_ARM = 64
SECONDARY_ARMS = (512, 4096)
ARMS = (PRIMARY_ARM, *SECONDARY_ARMS)
CHECKPOINTS = (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)
CONTROL_CHECKPOINTS = (300, 1000)
DELTA = 0.03
REPRO_TOL = 1e-6
BUCKET = "gap_2_to_M"
DEVICE = CSC.DEVICE

#: PREREG "checkpoint_source" / "Condition": read-only, never written.
SOURCE_ROOT = (
    Path.home()
    / "retrieval-successor-retention"
    / ".worktrees"
    / "corpus-curve"
    / "runs"
    / "corpus-size-curve"
)
#: The corpus-size ledger as committed on this branch (26c590b).
REF_LEDGER = "runs/corpus-size-curve/ledger.json"
REF_MANIFEST = "runs/corpus-size-curve/manifest.json"
#: PREREG "checkpoint_source": the checkpoints were written by the run at ec9332d.
REF_GIT_SHA_PREFIX = "ec9332d"
CONTROL_PREFIXES = tuple(f"n{PRIMARY_ARM}.ckpt{c}." for c in CONTROL_CHECKPOINTS)

#: --parallel only: the machine's 12 deterministic CPU slots over 3 seeds.
THREADS_PER_CHILD = 4

QUESTION = (
    "In the corpus-size curve's N = 64 arm, does memory-dependent accuracy on UNSEEN "
    "documents appear before, with, or after memorisation of the training documents?"
)
DECISION_RULE = (
    "A failed or incomplete reproduction control, or a measurement that raised, makes "
    "the result inconclusive (exit 3). Otherwise, on the N = 64 arm: onset_R is the "
    "earliest checkpoint from which R holds at that and every later checkpoint "
    "through 1000; onset_M likewise for M. BEFORE if onset_R < onset_M; WITH if "
    "equal; AFTER if onset_R > onset_M; inconclusive if either onset does not exist."
)
#: PREREG "Author's expectation", verbatim; frozen into the manifest before any
#: measurement.
EXPECTED = (
    "AFTER, with onset_M at 200 or 300 and onset_R at 500–700. R rises steadily once "  # noqa: RUF001 -- verbatim
    "train accuracy passes ~0.5, rather than jumping. The train memory share rises "
    "before R does. At N = 512 and N = 4096 nothing reaches DELTA except possibly "
    "N = 512 seed 2 at 900–1000."  # noqa: RUF001 -- verbatim
)
#: PREREG "What each result commits the next run to (written now)".
NEXT_RUN = {
    "BEFORE": "arm A only (fresh stream from scratch)",
    "WITH": "arms A and B",
    "AFTER": "arms A and B",
    "inconclusive": "arms A and B",
}

Job = tuple[int, int, int]  # (N, seed, checkpoint)


# --------------------------------------------------------------------------- #
# where the checkpoints are
# --------------------------------------------------------------------------- #


def seed_dir(source_root: Path, n: int, seed: int) -> Path:
    return Path(source_root) / f"n{n}" / f"seed{seed}"


def ckpt_file(source_root: Path, job: Job) -> Path:
    n, s, c = job
    return RC.ckpt_path(seed_dir(source_root, n, s), c)


def control_jobs() -> list[Job]:
    """Measured first, before anything else (PREREG)."""
    return [(PRIMARY_ARM, s, c) for s in SEEDS for c in CONTROL_CHECKPOINTS]


def main_jobs() -> list[Job]:
    """Everything else: N = 64 first, then the secondary arms; the control's own
    measurements are reused, not repeated."""
    done = set(control_jobs())
    return [
        (n, s, c)
        for n in ARMS
        for s in SEEDS
        for c in CHECKPOINTS
        if (n, s, c) not in done
    ]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# the quantities (PREREG "Primary readout", "Secondary readouts")
# --------------------------------------------------------------------------- #


def _acc(m: dict, ds: str, cond: str) -> float:
    return float(m[ds][cond][BUCKET]["answer_acc"])


def r_quantity(m: dict) -> float:
    """heldout.live.answer_acc - heldout.slots_zeroed.answer_acc at gap_2_to_M."""
    return _acc(m, "heldout", "live") - _acc(m, "heldout", "slots_zeroed")


def m_quantity(m: dict) -> float:
    """train.live.answer_acc - heldout.live.answer_acc at gap_2_to_M."""
    return _acc(m, "train", "live") - _acc(m, "heldout", "live")


def train_memory_share(m: dict) -> float:
    """train.live - train.slots_zeroed accuracy: memorised answers routed through
    the slots."""
    return _acc(m, "train", "live") - _acc(m, "train", "slots_zeroed")


def side_channel_share(m: dict) -> float:
    """train.gate_zeroed - train.gate_zeroed_bos_off accuracy."""
    return _acc(m, "train", "gate_zeroed") - _acc(m, "train", "gate_zeroed_bos_off")


#: The derived per-seed readouts, by ledger-key suffix.
DERIVED = {
    "R_quantity": r_quantity,
    "M_quantity": m_quantity,
    "train_memory_share": train_memory_share,
    "side_channel_share": side_channel_share,
}


def holds(values: list[float]) -> bool:
    """R(c) or M(c): the quantity is >= DELTA on EVERY seed."""
    return len(values) == len(SEEDS) and all(v >= DELTA for v in values)


def sustained_onset(flags: dict[int, bool]) -> int | None:
    """The earliest checkpoint c such that the flag holds at c and at EVERY later
    checkpoint through the last; ``None`` if it fails at the last. A single noisy
    crossing that later lapses is not an onset. Every checkpoint must be present."""
    missing = [c for c in CHECKPOINTS if c not in flags]
    if missing:
        raise ValueError(f"no reading at checkpoints {missing}: an onset needs all")
    onset = None
    for c in reversed(CHECKPOINTS):
        if not flags[c]:
            break
        onset = c
    return onset


def classify(onset_r: int | None, onset_m: int | None) -> str:
    """PREREG decision table, rows 2-5."""
    if onset_r is None or onset_m is None:
        return "inconclusive"
    if onset_r < onset_m:
        return "BEFORE"
    if onset_r == onset_m:
        return "WITH"
    return "AFTER"


def quantities(per: dict[int, dict[int, dict]]) -> dict[str, dict[int, list[float]]]:
    """``{name: {c: [per-seed value]}}`` for every checkpoint measured on every seed."""
    out: dict[str, dict[int, list[float]]] = {k: {} for k in DERIVED}
    for c in CHECKPOINTS:
        if not all(c in per.get(s, {}) for s in SEEDS):
            continue
        for k, f in DERIVED.items():
            out[k][c] = [f(per[s][c]["s003"]) for s in SEEDS]
    return out


def primary(per64: dict[int, dict[int, dict]]) -> dict:
    """The N = 64 rule: R(c), M(c), their sustained onsets and the classification."""
    q = quantities(per64)
    r_flags = {c: holds(v) for c, v in q["R_quantity"].items()}
    m_flags = {c: holds(v) for c, v in q["M_quantity"].items()}
    onset_r = sustained_onset(r_flags)
    onset_m = sustained_onset(m_flags)
    return {
        "R_holds": r_flags,
        "M_holds": m_flags,
        "onset_R": onset_r,
        "onset_M": onset_m,
        "classification": classify(onset_r, onset_m),
    }


def seeds_reaching_delta(values: list[float]) -> list[int]:
    """PREREG secondary: 'any checkpoint where R's quantity reaches DELTA on any
    single seed is reported by name'."""
    return [s for s, v in zip(SEEDS, values, strict=True) if v >= DELTA]


# --------------------------------------------------------------------------- #
# the reproduction control
# --------------------------------------------------------------------------- #


class _Rows:
    """A ledger-shaped collector for the corpus-size curve's ``write_rows``."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def note(self, key: str, value: Any, *, how: str) -> None:
        self.rows.append({"key": key, "kind": "observation", "value": value})

    def stat(self, key: str, samples: list, *, how: str) -> None:
        self.rows.append({"key": key, "kind": "statistic", "samples": list(samples)})

    def verdict(self, **kw) -> None:
        pass


def reference_rows(doc: dict) -> dict[str, list[float]]:
    """Every ``n64.ckpt300.*`` / ``n64.ckpt1000.*`` STATISTIC row of the corpus-size
    ledger, samples in seed order."""
    if doc.get("seeds_actually_run") != SEEDS:
        raise ValueError(f"reference seeds {doc.get('seeds_actually_run')} != {SEEDS}")
    return {
        r["key"]: [float(x) for x in r["samples"]]
        for r in doc["rows"]
        if r.get("kind") == "statistic" and r["key"].startswith(CONTROL_PREFIXES)
    }


def remeasured_rows(per64: dict[int, dict[int, dict]]) -> dict[str, list[float]]:
    """The re-measured N = 64 ckpt-300/1000 tables as the corpus-size ledger's rows,
    written by the corpus-size curve's OWN ``write_rows`` (imported: the naming and
    the derived-row arithmetic are its, not a transcription)."""
    per = {
        s: {c: per64[s][c] for c in CONTROL_CHECKPOINTS if c in per64.get(s, {})}
        for s in SEEDS
    }
    arm = {
        "per": per,
        "stopped": None,
        "error": None,
        "elapsed_s": 0.0,
        "checkpoints_measured": {s: sorted(per[s]) for s in SEEDS},
    }
    res = {"arms": {PRIMARY_ARM: arm}, "control": {}, "not_run": {}, "error": None}
    col = _Rows()
    CSC.write_rows(col, res)
    return {
        r["key"]: [float(x) for x in r["samples"]]
        for r in col.rows
        if r["kind"] == "statistic" and r["key"].startswith(CONTROL_PREFIXES)
    }


def reproduction_control(
    reference: dict[str, list[float]],
    remeasured: dict[str, list[float]],
    tol: float = REPRO_TOL,
) -> dict:
    """PREREG: every reference key, per seed, ``|re-measured - ledger| <= tol``. A
    key absent from the re-measurement, or a non-finite difference, fails."""
    per_seed: dict[int, dict] = {
        s: {"ok": True, "n_keys_compared": 0, "max_abs_diff": 0.0, "failures": []}
        for s in SEEDS
    }
    missing = sorted(k for k in reference if len(remeasured.get(k, [])) != len(SEEDS))
    for key in sorted(reference):
        if key in missing:
            continue
        for i, s in enumerate(SEEDS):
            ref, got = float(reference[key][i]), float(remeasured[key][i])
            diff = abs(got - ref)
            p = per_seed[s]
            p["n_keys_compared"] += 1
            if not math.isfinite(diff) or diff > tol:
                p["ok"] = False
                p["failures"].append(
                    {"key": key, "reference": ref, "measured": got, "abs_diff": diff}
                )
            if math.isfinite(diff):
                p["max_abs_diff"] = max(p["max_abs_diff"], diff)
    if missing:
        for p in per_seed.values():
            p["ok"] = False
    return {
        "ok": bool(reference) and all(p["ok"] for p in per_seed.values()),
        "n_reference_keys": len(reference),
        "missing_keys": missing,
        "tolerance": tol,
        "per_seed": per_seed,
    }


# --------------------------------------------------------------------------- #
# executing measurements
# --------------------------------------------------------------------------- #

Executor = Callable[[list[Job], Path], tuple[dict[Job, dict], str | None]]


def run_sequential(
    jobs: list[Job],
    source_root: Path,
    measure_fn: Callable[[Path, int, int, int], dict] = measure_checkpoint,
    log: Callable[[str], None] = lambda s: print(s, flush=True),
) -> tuple[dict[Job, dict], str | None]:
    """Measure ``jobs`` in order in this process. Stops at the first raise and
    returns ``(results so far, error)``."""
    out: dict[Job, dict] = {}
    for n, s, c in jobs:
        try:
            out[(n, s, c)] = measure_fn(seed_dir(source_root, n, s), s, c, n)
        except Exception as e:
            return out, f"N={n} seed {s} ckpt{c}: {type(e).__name__}: {e}"
        log(f"  measured N={n} seed {s} ckpt{c}")
    return out, None


def _jobs_arg(jobs: list[Job]) -> str:
    return ",".join(f"{n}:{c}" for n, _s, c in jobs)


def _parse_jobs(seed: int, text: str) -> list[Job]:
    out = []
    for item in text.split(","):
        n, c = item.split(":")
        out.append((int(n), seed, int(c)))
    return out


def child_argv(seed: int, jobs: list[Job], source_root: Path, out: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--seed",
        str(seed),
        "--jobs",
        _jobs_arg(jobs),
        "--source-root",
        str(source_root),
        "--out",
        str(out),
        "--threads",
        str(THREADS_PER_CHILD),
    ]


def child_env(threads: int = THREADS_PER_CHILD) -> dict[str, str]:
    return {
        **os.environ,
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
    }


def child_main(
    seed: int,
    jobs: list[Job],
    source_root: Path,
    out: Path,
    threads: int,
    measure_fn: Callable = measure_checkpoint,
) -> Exit:
    """One seed's jobs in a child process; results written to ``out`` as JSON
    (floats round-trip exactly through ``json``)."""
    torch.set_num_threads(threads)
    res, err = run_sequential(jobs, source_root, measure_fn)
    out.write_text(
        json.dumps(
            {
                "seed": seed,
                "torch_num_threads": torch.get_num_threads(),
                "results": [[n, s, c, m] for (n, s, c), m in res.items()],
                "error": err,
            },
            default=str,
        )
        + "\n"
    )
    return Exit.DID_NOT_RUN if err else Exit.OK


class ParallelExecutor:
    """One child per seed (``--parallel``). ``spawn(argv, env)`` returns an object
    with ``wait() -> int``; injectable for the tests."""

    def __init__(self, work_dir: Path, spawn: Callable | None = None) -> None:
        self.work_dir = Path(work_dir)
        self.spawn = spawn or (
            lambda argv, env: subprocess.Popen(argv, cwd=ROOT, env=env)
        )
        self.commands: list[dict] = []
        self._phase = 0

    def __call__(
        self, jobs: list[Job], source_root: Path
    ) -> tuple[dict[Job, dict], str | None]:
        self._phase += 1
        self.work_dir.mkdir(parents=True, exist_ok=True)
        procs = {}
        for s in SEEDS:
            mine = [j for j in jobs if j[1] == s]
            if not mine:
                continue
            out = self.work_dir / f"phase{self._phase}-seed{s}.json"
            argv = child_argv(s, mine, source_root, out)
            procs[s] = (self.spawn(argv, child_env()), argv, out)
        results: dict[Job, dict] = {}
        errors = []
        for s, (p, argv, out) in procs.items():
            rc = p.wait()
            self.commands.append({"argv": argv, "exit_code": rc, "seed": s})
            doc = json.loads(out.read_text()) if out.exists() else None
            if doc is None:
                errors.append(f"seed {s} child exited {rc} and wrote no results")
                continue
            for n, s2, c, m in doc["results"]:
                results[(int(n), int(s2), int(c))] = m
            if doc["error"] or rc != 0:
                errors.append(doc["error"] or f"seed {s} child exited {rc}")
        return results, ("; ".join(errors) or None)


# --------------------------------------------------------------------------- #
# the run and the rule
# --------------------------------------------------------------------------- #


def _put(per: dict, results: dict[Job, dict]) -> None:
    for (n, s, c), m in results.items():
        per.setdefault(n, {s2: {} for s2 in SEEDS})[s][c] = m


def run(
    executor: Executor,
    source_root: Path,
    reference: dict[str, list[float]],
    log: Callable[[str], None] = lambda s: print(s, flush=True),
) -> dict:
    """The control first; everything else only if it passed."""
    res: dict = {
        "per": {},
        "control": None,
        "error": None,
        "stopped": None,
        "jobs_run": [],
    }
    t0 = time.time()
    jobs = control_jobs()
    got, err = executor(jobs, source_root)
    _put(res["per"], got)
    res["jobs_run"] += [list(j) for j in jobs if j in got]
    if err:
        res["error"], res["stopped"] = err, "a reproduction-control measurement raised"
        res["elapsed_s"] = time.time() - t0
        return res
    per64 = res["per"].get(PRIMARY_ARM, {})
    res["control"] = reproduction_control(reference, remeasured_rows(per64))
    log(f"reproduction control ok={res['control']['ok']}")
    if not res["control"]["ok"]:
        res["stopped"] = "reproduction control failed"
        res["elapsed_s"] = time.time() - t0
        return res
    jobs = main_jobs()
    got, err = executor(jobs, source_root)
    _put(res["per"], got)
    res["jobs_run"] += [list(j) for j in jobs if j in got]
    if err:
        res["error"], res["stopped"] = err, "a measurement raised"
    res["elapsed_s"] = time.time() - t0
    return res


def complete(per: dict) -> list[str]:
    """Every pre-registered (arm, seed, checkpoint) absent from ``per``."""
    return [
        f"n{n}.seed{s}.ckpt{c}"
        for n in ARMS
        for s in SEEDS
        for c in CHECKPOINTS
        if c not in per.get(n, {}).get(s, {})
    ]


def verdict(res: dict) -> dict:
    """PREREG decision table, row for row."""
    base = {"onset_R": None, "onset_M": None, "R_holds": {}, "M_holds": {}}
    if res.get("error"):
        return {
            **base,
            "classification": "inconclusive",
            "exit": Exit.DID_NOT_RUN,
            "detail": f"a measurement raised ({res['error']})",
        }
    control = res.get("control")
    if control is None:
        return {
            **base,
            "classification": "inconclusive",
            "exit": Exit.DID_NOT_RUN,
            "detail": "the reproduction control did not run",
        }
    if not control["ok"]:
        bad = [f"seed{s}" for s, p in control["per_seed"].items() if not p["ok"]]
        return {
            **base,
            "classification": "inconclusive",
            "exit": Exit.DID_NOT_RUN,
            "detail": "the reproduction control failed or was incomplete on "
            + ", ".join(bad)
            + "; nothing further was measured",
        }
    missing = complete(res["per"])
    if missing:
        return {
            **base,
            "classification": "inconclusive",
            "exit": Exit.DID_NOT_RUN,
            "detail": f"incomplete: not measured {missing}",
        }
    p = primary(res["per"][PRIMARY_ARM])
    if p["classification"] == "inconclusive":
        detail = f"no sustained onset: onset_R={p['onset_R']}, onset_M={p['onset_M']}"
    else:
        detail = (
            f"onset_R={p['onset_R']} vs onset_M={p['onset_M']}: {p['classification']}"
        )
    return {**p, "exit": Exit.OK, "detail": detail}


# --------------------------------------------------------------------------- #
# the ledger
# --------------------------------------------------------------------------- #


def _stat(led, key: str, xs: list, *, how: str) -> None:
    """CSC._stat's rule: fewer than every seed, or a None, is a note."""
    CSC._stat(led, key, xs, how=how)


def arm_rows(led, n: int, per: dict[int, dict[int, dict]]) -> None:
    """Every readout for one arm at every checkpoint, in the corpus-size ledger's
    naming (``n<N>.ckpt<c>.<set>.<condition>.<bucket>.<readout>``), plus the derived
    quantities this PREREG names."""
    how = (
        f"{CSC.EXPERIMENT}::measure_checkpoint (imported) -> {S003.EXPERIMENT}::measure "
        f"with sets=doc_sets(seed, N); per-seed samples in seed order"
    )
    for c in CHECKPOINTS:
        seeds = [s for s in SEEDS if c in per.get(s, {})]
        if not seeds:
            continue
        p = f"n{n}.ckpt{c}"
        led.note(f"{p}.seeds_measured", seeds, how="run()")
        led.note(
            f"{p}.stored_step",
            {f"seed{s}": per[s][c]["stored_step"] for s in seeds},
            how="rsr.train.checkpoint.load(...)['step'], checked == label",
        )
        tabs = [per[s][c]["s003"] for s in seeds]
        for ds in ("heldout", "train"):
            for cond in S003.CONDITIONS:
                for b in S003.BUCKETS:
                    for key in CSC.READOUTS:
                        _stat(
                            led,
                            f"{p}.{ds}.{cond}.{b}.{key}",
                            [t[ds][cond][b][key] for t in tabs],
                            how=f"{how}; set {ds}; condition {cond}; bucket {b}",
                        )
            for key in CSC.READOUTS:
                _stat(
                    led,
                    f"{p}.{ds}.slots_zeroed_minus_live.{BUCKET}.{key}",
                    [
                        t[ds]["slots_zeroed"][BUCKET][key] - t[ds]["live"][BUCKET][key]
                        for t in tabs
                    ],
                    how=f"per seed: {p}.{ds}.slots_zeroed - live, bucket {BUCKET}",
                )
        for key in CSC.READOUTS:
            _stat(
                led,
                f"{p}.train_minus_heldout.live.{BUCKET}.{key}",
                [
                    t["train"]["live"][BUCKET][key] - t["heldout"]["live"][BUCKET][key]
                    for t in tabs
                ],
                how=f"per seed: probe minus common held-out, live, {BUCKET}",
            )
        led.note(
            f"{p}.memory_gates",
            {f"seed{s}": per[s][c]["s003"]["memory_gates"] for s in seeds},
            how="trained checkpoint's memory_gate per C block",
        )
        for k, f in DERIVED.items():
            _stat(
                led,
                f"{p}.{k}",
                [f(t) for t in tabs],
                how=f"per seed, {BUCKET} answer_acc: {f.__doc__.split(chr(10))[0]}",
            )
        if seeds == SEEDS:
            rq = [r_quantity(t) for t in tabs]
            led.note(f"{p}.R_holds", holds(rq), how=f"R_quantity >= {DELTA} every seed")
            led.note(
                f"{p}.M_holds",
                holds([m_quantity(t) for t in tabs]),
                how=f"M_quantity >= {DELTA} every seed",
            )
            led.note(
                f"{p}.R_reaches_delta_on_seeds",
                seeds_reaching_delta(rq),
                how=f"seeds whose R_quantity >= {DELTA} (PREREG secondary)",
            )


def write_rows(led, res: dict, v: dict) -> None:
    for n in ARMS:
        if n in res["per"]:
            arm_rows(led, n, res["per"][n])
    c = res.get("control")
    per_seed = (c or {}).get("per_seed", {})
    for f in ("ok", "n_keys_compared", "max_abs_diff"):
        led.note(
            f"reproduction_control.{f}",
            [per_seed[s][f] if s in per_seed else None for s in SEEDS],
            how=f"per seed, seed order {SEEDS}: every n64.ckpt300/1000 statistic "
            f"key of {REF_LEDGER} vs the re-measurement through "
            f"{CSC.EXPERIMENT}::write_rows",
        )
    led.note(
        "reproduction_control.failures",
        {f"seed{s}": per_seed[s]["failures"] for s in per_seed},
        how="every (key, reference, measured, abs_diff) outside the tolerance",
    )
    led.note(
        "reproduction_control.missing_keys",
        (c or {}).get("missing_keys"),
        how="reference keys the re-measurement did not produce",
    )
    led.note(
        "reproduction_control.n_reference_keys",
        (c or {}).get("n_reference_keys"),
        how=f"statistic keys under {list(CONTROL_PREFIXES)} in {REF_LEDGER}",
    )
    led.note("reproduction_control.passed", (c or {}).get("ok"), how="all seeds ok")
    led.note("reproduction_tolerance", REPRO_TOL, how="PREREG thresholds")
    led.note("delta", DELTA, how="PREREG thresholds")
    led.note("onset_R", v["onset_R"], how="sustained_onset(R_holds), N=64")
    led.note("onset_M", v["onset_M"], how="sustained_onset(M_holds), N=64")
    led.note("classification", v["classification"], how="PREREG decision table")
    led.note("classification_detail", v["detail"], how="verdict()")
    led.note(
        "next_run_arms",
        NEXT_RUN[v["classification"]],
        how="PREREG 'What each result commits the next run to'",
    )
    led.note(
        "secondary_R_reaches_delta",
        [
            f"n{n}.ckpt{c}.seed{s}"
            for n in SECONDARY_ARMS
            for c in CHECKPOINTS
            if all(c in res["per"].get(n, {}).get(s2, {}) for s2 in SEEDS)
            for s in seeds_reaching_delta(
                [r_quantity(res["per"][n][s2][c]["s003"]) for s2 in SEEDS]
            )
        ],
        how=f"PREREG secondary: any single seed with R_quantity >= {DELTA}",
    )
    led.note("measurement_error", res.get("error"), how="run(); None if none")
    led.note("stopped_because", res.get("stopped"), how="run(); None if none")
    led.note("not_measured", complete(res["per"]), how="complete()")
    led.note("elapsed_s", res.get("elapsed_s"), how="wall clock of run()")
    led.note("arms_preregistered", list(ARMS), how="PREREG front matter")
    led.note("checkpoints_preregistered", list(CHECKPOINTS), how="PREREG front matter")
    led.note("decision_rule", DECISION_RULE, how="PREREG front matter, verbatim")
    led.note("expected", EXPECTED, how="PREREG 'Author's expectation', in the manifest")


def manifest(
    source_root: Path, mode: str, threads: int, ref_sha: str, ref_doc: dict
) -> dict:
    """Frozen before any measurement: every checkpoint's sha256 included."""
    all_jobs = control_jobs() + main_jobs()
    return {
        "prereg": PREREG,
        "prereg_commit": PREREG_COMMIT,
        "question": QUESTION,
        "decision_rule": DECISION_RULE,
        "expected": EXPECTED,
        "instrument": f"{CSC.EXPERIMENT}::measure_checkpoint (imported, identity)",
        "arms": list(ARMS),
        "primary_arm": PRIMARY_ARM,
        "secondary_arms": list(SECONDARY_ARMS),
        "seeds": SEEDS,
        "checkpoints": list(CHECKPOINTS),
        "control_checkpoints": list(CONTROL_CHECKPOINTS),
        "delta": DELTA,
        "reproduction_tolerance": REPRO_TOL,
        "bucket": BUCKET,
        "heldout_documents": list(CSC.HELDOUT),
        "probe_documents": list(CSC.PROBE),
        "device": DEVICE,
        "mode": mode,
        "torch_threads": threads,
        "source_root": str(source_root),
        "reference_ledger": REF_LEDGER,
        "reference_ledger_sha256": ref_sha,
        "reference_git_sha": ref_doc.get("provenance", {}).get("git_sha"),
        "job_order": [list(j) for j in all_jobs],
        "checkpoint_sha256": {
            str(ckpt_file(source_root, j).relative_to(source_root)): sha256_file(
                ckpt_file(source_root, j)
            )
            for j in all_jobs
        },
        "torch": torch.__version__,
        "python": sys.version.split()[0],
    }


# --------------------------------------------------------------------------- #
# RESULTS.md, rendered from the ledger
# --------------------------------------------------------------------------- #


def read_brief_errors(path: Path | None = None) -> str | None:
    path = path or ROOT / BRIEF_ERRORS
    return path.read_text() if path.exists() else None


COLUMNS = (
    "R_quantity",
    "R_holds",
    "M_quantity",
    "M_holds",
    "train_memory_share",
    "side_channel_share",
)


def render_results(doc: dict, brief_errors: str | None = None) -> str:
    """Every number beside its ledger key (`render_scoreboard.py --audit`). No number
    is typed here."""
    rows = RC._rows(doc)
    val, fmt = RC._val, RC._fmt
    prov = doc.get("provenance", {})
    rid = doc.get("run_id")
    out = [
        "# Scaffold timing -- RESULTS",
        "",
        f"<!-- GENERATED by `{EXPERIMENT}` from runs/{rid}/ledger.json. Do not edit: "
        "regenerate with `--render-results`. -->",
        "",
        f"Command: `uv run python {EXPERIMENT}`. Ledger `status` of run `{rid}`: "
        f"**{doc.get('status')}**.",
        "",
        f"Provenance of run `{rid}`: `provenance.git_sha` `{prov.get('git_sha')}` · "
        f"`device` {doc.get('device')} · hardware `provenance.machine` "
        f"`{prov.get('machine')}`, `provenance.platform` `{prov.get('platform')}` · "
        f"`seeds_actually_run` {doc.get('seeds_actually_run')} · `config_hash` "
        f"`{doc.get('config_hash')}`.",
        "",
        f"Decision rule of run `{rid}` (`decision_rule`, the PREREG's), sentence by "
        "sentence:",
        "",
        *[
            f"- run `{rid}` `decision_rule`: **{x.strip().removesuffix('.')}.**"
            for x in str(val(rows, "decision_rule") or "").split(". ")
            if x.strip()
        ],
        "",
        f"## Classification of run `{rid}`: `classification` "
        f"**{val(rows, 'classification')}**",
        "",
        f"`onset_R` {fmt(val(rows, 'onset_R'))} · `onset_M` {fmt(val(rows, 'onset_M'))} "
        f"· `delta` {fmt(val(rows, 'delta'))}.",
        "",
        f"Why, for run `{rid}` (`classification_detail`): "
        f"**{val(rows, 'classification_detail')}**",
        "",
        f"What it commits the next run to, for run `{rid}` (`next_run_arms`): "
        f"**{val(rows, 'next_run_arms')}**.",
        "",
        f"`measurement_error` {fmt(val(rows, 'measurement_error'))} · "
        f"`stopped_because` {fmt(val(rows, 'stopped_because'))} · `not_measured` "
        f"{fmt(val(rows, 'not_measured')) or '—'}.",
        "",
        "Whether the cognitive claim moved: **it cannot move here.** Nothing about RSR, "
        "eviction, ψ̂ or Kintsch & van Dijk; timing is correlation (PREREG, *What this "
        "does not establish*). No scaling claim.",
        "",
        f"Pre-registered expectation of run `{rid}` (`expected`, frozen in the manifest "
        "before any measurement), sentence by sentence:",
        "",
        *[
            f"- run `{rid}` `expected`: **{x.strip().removesuffix('.')}.**"
            for x in str(val(rows, "expected") or "").split(". ")
            if x.strip()
        ],
        "",
        "## The reproduction control (checked first)",
        "",
        f"`reproduction_tolerance` {val(rows, 'reproduction_tolerance')!r} absolute · "
        f"`reproduction_control.n_reference_keys` "
        f"{fmt(val(rows, 'reproduction_control.n_reference_keys'))} · "
        f"`reproduction_control.passed` "
        f"{fmt(val(rows, 'reproduction_control.passed'))}.",
        "",
        "| seed | `reproduction_control.ok` | `reproduction_control.n_keys_compared` "
        "| `reproduction_control.max_abs_diff` |",
        "|---|---|---|---|",
    ]
    cols = [
        val(rows, f"reproduction_control.{f}") or [None] * len(SEEDS)
        for f in ("ok", "n_keys_compared", "max_abs_diff")
    ]
    for i, s in enumerate(SEEDS):
        cells = [
            fmt(x[i]) if isinstance(x[i], bool | None | int) else repr(x[i]) for x in cols
        ]
        out.append(f"| seed{s} | " + " | ".join(cells) + " |")
    fails = val(rows, "reproduction_control.failures") or {}
    n_fail = sum(len(v) for v in fails.values())
    out += [
        "",
        f"`reproduction_control.missing_keys` "
        f"{fmt(val(rows, 'reproduction_control.missing_keys')) or '—'}.",
        "",
    ]
    if n_fail:
        out += [
            "Every key outside the tolerance (`reproduction_control.failures`):",
            "",
            "| seed | key | reference | measured | abs_diff |",
            "|---|---|---|---|---|",
        ]
        for s, fl in fails.items():
            for f in fl:
                out.append(
                    f"| {s} | `{f['key']}` | {f['reference']!r} | "
                    f"{f['measured']!r} | {f['abs_diff']!r} |"
                )
        out.append("")
    for n in ARMS:
        role = "primary" if n == PRIMARY_ARM else "secondary, no verdict"
        out += [
            f"## N = `n{n}` ({role}); per-seed samples in seed order, bucket `{BUCKET}`",
            "",
            "| arm.checkpoint | "
            + " | ".join(f"`{k}`" for k in COLUMNS)
            + " | `heldout.slots_zeroed_minus_live.gap_2_to_M.answer_brier_over_16` "
            "| `train.live.gap_2_to_M.answer_acc` |",
            "|---|" + "---|" * (len(COLUMNS) + 2),
        ]
        for c in CHECKPOINTS:
            p = f"n{n}.ckpt{c}"
            if f"{p}.seeds_measured" not in rows:
                out.append(f"| {p} | not measured |" + " |" * (len(COLUMNS) + 1))
                continue
            keys = [f"{p}.{k}" for k in COLUMNS] + [
                f"{p}.heldout.slots_zeroed_minus_live.{BUCKET}.answer_brier_over_16",
                f"{p}.train.live.{BUCKET}.answer_acc",
            ]
            out.append(f"| {p} | " + " | ".join(fmt(val(rows, k)) for k in keys) + " |")
        out.append("")
    out += ["Seeds on which the R quantity reaches `delta`, by checkpoint:", ""]
    any_reach = False
    for n in ARMS:
        for c in CHECKPOINTS:
            k = f"n{n}.ckpt{c}.R_reaches_delta_on_seeds"
            got = val(rows, k)
            if got:
                any_reach = True
                out.append(f"- `{k}` {got}")
    if not any_reach:
        out.append("- none")
    out += [
        "",
        f"`secondary_R_reaches_delta` (the secondary arms, by name): "
        f"{fmt(val(rows, 'secondary_R_reaches_delta')) or '—'}.",
        "",
        "Every other bucket, condition and readout is in the ledger under "
        "`n<N>.ckpt<checkpoint>.`.",
        "",
        "## BRIEF ERRORS",
        "",
        (brief_errors or "").strip()
        or f"Not yet written: the executor writes `{BRIEF_ERRORS}` after the run "
        "('none' if none) and re-renders.",
        "",
    ]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# --dry-run and CLI
# --------------------------------------------------------------------------- #


def dry_run(source_root: Path, parallel: bool) -> str:
    ref = json.loads((ROOT / REF_LEDGER).read_text())
    lines = [
        "scaffold-timing dry run (NOTHING MEASURED).",
        f"source root (read-only): {source_root}",
        f"reference ledger: {REF_LEDGER} sha256 {sha256_file(ROOT / REF_LEDGER)}; "
        f"{len(reference_rows(ref))} statistic keys under {list(CONTROL_PREFIXES)}",
        f"measurement: {CSC.EXPERIMENT}::measure_checkpoint (imported); device {DEVICE}",
        "mode: "
        + (
            f"parallel, one child per seed at {THREADS_PER_CHILD} threads"
            if parallel
            else "sequential in this process at the corpus-size parent's thread count"
        ),
        f"DELTA {DELTA}; reproduction tolerance {REPRO_TOL}",
        "phase 1 -- reproduction control (stop with exit 3 if it fails):",
    ]
    for j in control_jobs():
        f = ckpt_file(source_root, j)
        lines.append(f"  N={j[0]} seed {j[1]} ckpt{j[2]}  {f}  exists={f.exists()}")
    lines.append("phase 2 -- every other (arm, seed, checkpoint):")
    for j in main_jobs():
        f = ckpt_file(source_root, j)
        lines.append(f"  N={j[0]} seed {j[1]} ckpt{j[2]}  {f}  exists={f.exists()}")
    lines.append(
        f"total {len(control_jobs()) + len(main_jobs())} measurements; wall clock "
        "per measurement not measured (no prediction offered)"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> Exit:
    ap = ArgumentParser(prog="scaffold-timing")
    ap.add_argument(
        "--dry-run", action="store_true", help="list the plan; measure nothing"
    )
    ap.add_argument(
        "--render-results", action="store_true", help="RESULTS.md from the ledger"
    )
    ap.add_argument("--parallel", action="store_true", help="one child per seed")
    ap.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    ap.add_argument("--run-id", default=RUN_ID)
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", default="")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--threads", type=int, default=THREADS_PER_CHILD)
    a = ap.parse_args(argv)

    if a.child:
        if a.out is None or not a.jobs:
            refuse(Exit.DID_NOT_RUN, "--child needs --jobs and --out")
        return status(
            child_main(
                a.seed, _parse_jobs(a.seed, a.jobs), a.source_root, a.out, a.threads
            )
        )
    if a.dry_run:
        print(dry_run(a.source_root, a.parallel))
        return Exit.OK
    root = ROOT / "runs" / a.run_id
    results = ROOT / "experiments" / "scaffold-timing" / "RESULTS.md"
    if a.render_results:
        path = root / "ledger.json"
        if not path.exists():
            refuse(Exit.DID_NOT_RUN, f"{path} is absent: the measurement has not run")
        results.write_text(
            render_results(json.loads(path.read_text()), read_brief_errors())
        )
        print(f"wrote {results}")
        return Exit.OK
    return status(_run(a, root, results))


def _run(a, root: Path, results: Path) -> Exit:
    import ledger as ledger_mod

    dirty = ledger_mod._dirty_source_paths()
    if dirty:
        refuse(Exit.DID_NOT_RUN, f"uncommitted source {dirty}: the ledger would refuse")
    if root.exists():
        refuse(Exit.DID_NOT_RUN, f"{root} exists: move it aside and re-run")
    ref_path = ROOT / REF_LEDGER
    ref_sha = sha256_file(ref_path)
    src_ledger = Path(a.source_root) / "ledger.json"
    if not src_ledger.exists() or sha256_file(src_ledger) != ref_sha:
        refuse(
            Exit.DID_NOT_RUN,
            f"{src_ledger} is absent or differs from {REF_LEDGER}: "
            "the checkpoints may not be the ledger's",
        )
    ref_doc = json.loads(ref_path.read_text())
    if not str(ref_doc.get("provenance", {}).get("git_sha", "")).startswith(
        REF_GIT_SHA_PREFIX
    ):
        refuse(Exit.DID_NOT_RUN, f"{REF_LEDGER} was not written at {REF_GIT_SHA_PREFIX}")
    if ref_doc.get("device") != DEVICE:
        refuse(Exit.DID_NOT_RUN, f"{REF_LEDGER} device is {ref_doc.get('device')!r}")
    absent = [
        str(ckpt_file(a.source_root, j))
        for j in control_jobs() + main_jobs()
        if not ckpt_file(a.source_root, j).exists()
    ]
    if absent:
        refuse(Exit.DID_NOT_RUN, f"checkpoints absent: {absent}")
    reference = reference_rows(ref_doc)

    if a.parallel:
        executor: Executor = ParallelExecutor(root / "children")
        threads = THREADS_PER_CHILD
        mode = "parallel"
    else:
        threads = int(
            json.loads((ROOT / REF_MANIFEST).read_text())["parent_torch_num_threads"]
        )
        torch.set_num_threads(threads)
        executor = run_sequential
        mode = "sequential"

    led = ledger_mod.Ledger(a.run_id, question=QUESTION)
    led.command(
        " ".join(["uv run python", EXPERIMENT, *sys.argv[1:]]),
        exit_code=None,
        note="this run (the parent)",
    )
    m = led.manifest(manifest(a.source_root, mode, threads, ref_sha, ref_doc))
    print(f"manifest frozen: {m}", flush=True)
    led.note("reference_ledger_sha256", ref_sha, how=f"sha256 of {REF_LEDGER}")
    led.note(
        "torch_threads",
        torch.get_num_threads() if mode == "sequential" else THREADS_PER_CHILD,
        how=f"{mode} mode",
    )

    res = run(executor, a.source_root, reference)
    root.mkdir(parents=True, exist_ok=True)
    (root / "raw.json").write_text(json.dumps(res, indent=2, default=str) + "\n")
    for cmd in getattr(executor, "commands", []):
        rel = " ".join(["uv run python", EXPERIMENT, *cmd["argv"][2:]])
        led.command(rel, exit_code=cmd["exit_code"], note=f"seed {cmd['seed']} child")
    v = verdict(res)
    measured = sorted({s for n in res["per"] for s in res["per"][n] if res["per"][n][s]})
    led.run_meta(device=DEVICE, seeds_actually_run=measured)
    if res.get("error"):
        led.status("crashed")
    elif res.get("stopped") or complete(res["per"]):
        led.status("partial")
    else:
        led.status("ok")
    write_rows(led, res, v)
    path = led.write()
    results.write_text(render_results(json.loads(path.read_text()), read_brief_errors()))
    print(f"ledger: {path}  classification={v['classification']}  exit={int(v['exit'])}")
    return v["exit"]


if __name__ == "__main__":
    run_main(main)
