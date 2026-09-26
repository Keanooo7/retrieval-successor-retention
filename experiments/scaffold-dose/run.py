"""Scaffold dose -- how much memorisation unlocks learning from the stream?

Pre-registration: `experiments/scaffold-dose/PREREG.md`, committed alone at 4249567,
ahead of this file. **No new training code:** every arm is fresh-stream's arm-B
child (`experiments/fresh-stream/run.py`, imported, not copied) -- the corpus-size
N = 64 ``train()`` call with ``stream=`` and ``resume=`` added -- started from the
corpus-size N = 64 ``ckpt-{k:06d}.pt`` instead of ``ckpt-001000.pt`` and trained
``k -> k + 500``. ``train()`` in `src/rsr/train/loop.py` is not touched.

Order (PREREG "Controls"; each failure overrides everything after it):

0. **Before the manifest:** every start checkpoint ``n64/seed<s>/ckpt-{k:06d}.pt``
   exists, stores step ``k``, and has the sha256 recorded in
   `runs/scaffold-timing/manifest.json`. Any mismatch: refused, exit 3.
1. **Control 2, measurement path unchanged, FIRST:** each seed's ``ckpt-000600.pt``
   re-measured with the imported ``measure_checkpoint`` and turned into rows by
   scaffold-timing's own ``arm_rows``; every ``n64.ckpt600.*`` statistic key of
   `runs/scaffold-timing/ledger.json` must match per seed within ``REPRO_TOL``.
   Fail, missing key or a raise: exit 3, no arm runs.
2. **Control 3:** the stream ids of every arm, ``[4160 + 16 k, 4160 + 16 (k + 500))``,
   are disjoint from the probe / vocabulary documents ``[0, 64)`` and the held-out
   ``[4096, 4160)`` and never repeat; every word of those documents and of both
   measurement sets is in the seed's ``[0, 64)`` map (fresh-stream's closure,
   imported). Fail: exit 3, nothing trains.
3. **Arms** in the order ``600, 400, 300, 200, 100``, one at a time; three seeds as
   parallel children at 5 threads (fresh-stream's ``run_arm``, imported). Each child
   runs **control 1** (resume is exact, fresh-stream's ``ResumeCheck``) before its
   first step. The parent measures ``ckpt k + 500`` (n = 64) as it appears.
4. The rule: U(k), k*, the classification and the monotonicity note; every secondary.

The parent enforces the absolute ``DEADLINE``: at it the children are stopped,
whatever end checkpoints exist are measured, and no further arm starts.

Usage::

    uv run --extra dev python experiments/scaffold-dose/run.py --dry-run
    experiments/scaffold-dose/run.sh        # the real run; exit code -> run.rc
    uv run --extra dev python experiments/scaffold-dose/run.py --render-results

Exit codes (`rsr.exit_codes`): 0 a classification was reached · 3 did not run or
inconclusive (a refused precondition, a control failed, an arm unreadable).
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as _dt
import importlib.util
import json
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

EXPERIMENT = "experiments/scaffold-dose/run.py"
RUN_ID = "scaffold-dose"
PREREG = "experiments/scaffold-dose/PREREG.md"
PREREG_COMMIT = "4249567"
BRIEF_ERRORS = "experiments/scaffold-dose/BRIEF-ERRORS.md"


def _load(name: str, rel: str):
    """Import an experiment script by path. Imported, never copied."""
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: fresh-stream, whose arm-B child, controls, parent loop and readouts this run
#: reuses. ONE instance: its ST / CSC are the very modules scaffold-timing measured
#: with.
FS = _load("_scaffold_dose_fs", "experiments/fresh-stream/run.py")
ST = FS.ST
CSC = FS.CSC
RC = FS.RC
S003 = FS.S003

#: THE measurement (PREREG "instrument"): the corpus-size curve's function object,
#: not a copy; `tests/test_scaffold_dose.py` checks the identity.
measure_checkpoint = FS.measure_checkpoint

# --------------------------------------------------------------------------- #
# 🔒 PREREG.md, transcribed. Changing any is changing the PREREG.
# --------------------------------------------------------------------------- #

SEEDS = [0, 1, 2]
ARMS_K = (100, 200, 300, 400, 600)
ARM_ORDER = (600, 400, 300, 200, 100)
STEPS = 500
CKPT_EVERY = FS.CKPT_EVERY
N_TRAIN = FS.CONTROL_ARM  # 64: the arm the start checkpoints come from
STREAM = FS.STREAM  # step t trains documents [4160 + 16 t, 4160 + 16 t + 16)
BATCH = FS.BATCH
HELDOUT = FS.HELDOUT
PROBE = FS.PROBE
VOCAB_DOCUMENTS = FS.VOCAB_DOCUMENTS
#: "DELTA: 0.03 -- unchanged from scaffold-timing and fresh-stream": imported.
DELTA = FS.DELTA
C_THRESHOLD = FS.C_THRESHOLD
#: "stream_loss_threshold: ln 16 - 0.10 = 2.6726 -- fresh-stream's": imported.
STREAM_LOSS_THRESHOLD = FS.STREAM_LOSS_THRESHOLD
REPRO_TOL = 1e-6
DEADLINE = "2026-09-26T06:00:00-07:00"  # 06:00 America/Los_Angeles, PDT
#: Control 2 re-measures the k = 600 start checkpoint of every seed.
CONTROL_K = 600
CONTROL_PREFIX = f"n{N_TRAIN}.ckpt{CONTROL_K}."
BUCKET = FS.BUCKET
DEVICE = FS.DEVICE
THREADS_PER_SEED = FS.THREADS_PER_SEED

#: PREREG "Primary readout" table, by classification.
EARLY_K = (100, 200)
MEMORISATION_K = (300, 400)
ONSET_K = (600,)
READING = {
    "UNLOCK_AFTER_600": "more than retrieval onset is needed, or 500 steps is too "
    "few below k = 1000",
    "AT_RETRIEVAL_ONSET": "unlocking needs the held-out retrieval the model had "
    "already started to show",
    "AT_MEMORISATION": "unlocking comes with memorisation, before held-out "
    "retrieval is measurable",
    "EARLY": "little or no memorisation is needed; the scaffold may be generic early "
    "training on repeated data",
}

#: The start checkpoints: read-only, never written (PREREG "arm").
SOURCE_ROOT = FS.SOURCE_ROOT
REF_LEDGER = "runs/scaffold-timing/ledger.json"
REF_MANIFEST = "runs/scaffold-timing/manifest.json"

QUESTION = (
    "How much memorisation unlocks learning from a non-repeating stream? Resume the "
    "corpus-size N = 64 arm at step k and train 500 fresh-stream steps: which k "
    "unlocks memory-mediated retrieval on unseen documents?"
)
DECISION_RULE = (
    "A failed resume check, a measurement that raised, or a missing end checkpoint "
    "for any seed of any arm makes that arm's cell unreadable; the result is "
    "inconclusive if any arm is unreadable. Otherwise U(k) holds when R holds at "
    "ckpt k + 500 on every seed. k* is the smallest k in the sweep with U(k). The "
    "reading is from the table below."
)
#: PREREG "Author's expectation", verbatim (whitespace joined); frozen into the
#: manifest before any child starts.
EXPECTED = (
    "k* = 400: k = 600 unlocks on every seed; k = 400 on every seed but slowly; k = "
    "300 on one or two seeds; k = 200 and 100 do not. The unlock is fast (under 200 "
    "steps) wherever it happens."
)

POLL_S = FS.POLL_S
STOP_GRACE_S = FS.STOP_GRACE_S
#: Prediction for --dry-run only; no rule reads it (PREREG "Time": ~3.33 s/step
#: from the fresh-stream heartbeats).
PRED_S_PER_STEP = 3.33


class StartCheckpointRefused(RuntimeError):
    """A start checkpoint is absent, stores another step, or is not the file
    scaffold-timing measured (sha256)."""


def deadline_epoch() -> float:
    return _dt.datetime.fromisoformat(DEADLINE).timestamp()


def sha256_file(path: Path) -> str:
    return ST.sha256_file(path)


def arm_name(k: int) -> str:
    """The arm's directory and ledger prefix: ``k600`` etc."""
    return f"k{k}"


def end_step(k: int) -> int:
    return k + STEPS


def start_ckpt(k: int, seed: int, source_root: Path = SOURCE_ROOT) -> Path:
    """The corpus-size N = 64 ``ckpt-{k:06d}.pt`` of ``seed``."""
    return RC.ckpt_path(ST.seed_dir(source_root, N_TRAIN, seed), k)


@contextlib.contextmanager
def fresh_stream_with(**overrides):
    """fresh-stream's module with some of its PREREG constants set for one call
    (``RESUME_STEP`` for the child, ``CHECKPOINTS`` for its ledger rows), restored
    after. Every name must already exist there: a typo raises, never adds one."""
    old = {name: getattr(FS, name) for name in overrides}
    for name, v in overrides.items():
        setattr(FS, name, v)
    try:
        yield FS
    finally:
        for name, v in old.items():
            setattr(FS, name, v)


# --------------------------------------------------------------------------- #
# the start checkpoints (before the manifest)
# --------------------------------------------------------------------------- #


def verify_start_checkpoints(
    source_root: Path,
    ref_manifest: dict,
    *,
    sha_fn: Callable[[Path], str] = sha256_file,
    step_fn: Callable[[Path], int] = RC.stored_step,
) -> dict[str, dict[str, str]]:
    """PREREG "Condition": every start checkpoint's sha256 must match
    ``runs/scaffold-timing/manifest.json``; it must also exist and store step k.
    Returns ``{"k<k>": {"seed<s>": sha256}}``; raises ``StartCheckpointRefused``
    naming EVERY problem, not only the first."""
    recorded = ref_manifest.get("checkpoint_sha256") or {}
    out: dict[str, dict[str, str]] = {}
    problems: list[str] = []
    for k in ARMS_K:
        out[arm_name(k)] = {}
        for s in SEEDS:
            f = start_ckpt(k, s, source_root)
            rel = f"n{N_TRAIN}/seed{s}/{f.name}"
            if not f.exists():
                problems.append(f"{f} is absent")
                continue
            sha = sha_fn(f)
            out[arm_name(k)][f"seed{s}"] = sha
            if recorded.get(rel) != sha:
                problems.append(
                    f"{f} sha256 {sha} is not the one {REF_MANIFEST} recorded for "
                    f"{rel} ({recorded.get(rel)})"
                )
                continue
            if step_fn(f) != k:
                problems.append(f"{f} does not store step {k}")
    if problems:
        raise StartCheckpointRefused("; ".join(problems))
    return out


# --------------------------------------------------------------------------- #
# control 3: the stream ids each arm uses, and the vocabulary closure
# --------------------------------------------------------------------------- #


def arm_stream_ids(k: int) -> list[int]:
    """The document ids the arm trains on: ``STREAM.window(t)`` for t in
    ``[k, k + 500)``, i.e. ``[4160 + 16 k, 4160 + 16 (k + 500))``."""
    return [i for t in range(k, end_step(k)) for i in STREAM.window(t, BATCH)]


def arm_disjointness(k: int) -> dict:
    """PREREG control 3 for one arm: no id in ``PROBE`` (= the vocabulary
    documents, the N = 64 training corpus) or ``HELDOUT``, and no id repeats."""
    ids = arm_stream_ids(k)
    uniq = set(ids)
    in_probe = sorted(uniq & set(range(*PROBE)))
    in_vocab = sorted(uniq & set(range(*VOCAB_DOCUMENTS)))
    in_heldout = sorted(uniq & set(range(*HELDOUT)))
    repeats = len(ids) - len(uniq)
    return {
        "ok": bool(ids) and not (in_probe or in_vocab or in_heldout or repeats),
        "n_ids": len(ids),
        "first_id": min(ids) if ids else None,
        "last_id": max(ids) if ids else None,
        "ids_in_probe": in_probe[:10],
        "ids_in_vocab_documents": in_vocab[:10],
        "ids_in_heldout": in_heldout[:10],
        "repeats": repeats,
    }


def closure_iters() -> int:
    """fresh-stream's ``vocabulary_closure(seed, iters)`` covers the ids of steps
    ``[0, iters)``; the last step any arm trains is ``max(k) + 500 - 1``, so this
    covers every arm's ids (and those of the steps below the smallest k)."""
    return max(end_step(k) for k in ARMS_K)


def preflight(arms: tuple[int, ...] = ARMS_K, iters: int | None = None) -> dict:
    """Control 3 and the closure, before any step. Never raises: a failure is a
    record with ``ok`` False and the reason."""
    out: dict[str, Any] = {"ok": False, "error": None, "arms": {}, "closure": {}}
    try:
        for k in arms:
            out["arms"][arm_name(k)] = d = arm_disjointness(k)
            if not d["ok"]:
                out["error"] = (
                    f"arm {arm_name(k)}: the stream is not disjoint or repeats a document"
                )
                return out
        n = closure_iters() if iters is None else iters
        for s in SEEDS:
            out["closure"][s] = FS.vocabulary_closure(s, n)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    out["ok"] = True
    return out


# --------------------------------------------------------------------------- #
# one arm-seed, in its own child process: fresh-stream's arm B, from ckpt k
# --------------------------------------------------------------------------- #


def single(k: int, seed: int, out_dir: Path, *, source_root: Path = SOURCE_ROOT) -> dict:
    """fresh-stream's arm-B ``single()`` (the corpus-size N = 64 call + the stream +
    the resume, inside its ``ResumeCheck``) with the resume step set to ``k`` and the
    run ending at ``k + 500``. Nothing retyped."""
    if k not in ARMS_K:
        raise ValueError(f"k={k} is not one of {ARMS_K}")
    with fresh_stream_with(RESUME_STEP=k):
        r = FS.single("B", seed, out_dir, iters=end_step(k), source_root=source_root)
    r["arm"] = arm_name(k)
    r["k"] = k
    r["start_checkpoint"] = str(start_ckpt(k, seed, source_root))
    return r


def child_argv(k: int, seed: int, out_dir: Path, source_root: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single",
        "--k",
        str(k),
        "--seed",
        str(seed),
        "--out-dir",
        str(out_dir),
        "--source-root",
        str(source_root),
    ]


class Child(RC.Child):
    """The retrieval curve's child (5 OMP/MKL threads), launched on THIS script."""

    def __init__(self, k: int, seed: int, out_dir: Path, source_root: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.argv = child_argv(k, seed, out_dir, source_root)
        self._log = open(out_dir / "train.log", "w")  # noqa: SIM115 -- closed in wait
        self._p = subprocess.Popen(
            self.argv, cwd=ROOT, stdout=self._log, stderr=self._log, env=RC.child_env()
        )


# --------------------------------------------------------------------------- #
# control 2: the measurement path unchanged
# --------------------------------------------------------------------------- #


def reference_rows(doc: dict) -> dict[str, list[float]]:
    """Every ``n64.ckpt600.*`` STATISTIC row of the scaffold-timing ledger."""
    if doc.get("seeds_actually_run") != SEEDS:
        raise ValueError(f"reference seeds {doc.get('seeds_actually_run')} != {SEEDS}")
    return {
        r["key"]: [float(x) for x in r["samples"]]
        for r in doc["rows"]
        if r.get("kind") == "statistic" and r["key"].startswith(CONTROL_PREFIX)
    }


def remeasured_rows(per_control: dict[int, dict[int, dict]]) -> dict[str, list[float]]:
    """The re-measured ckpt-600 tables as scaffold-timing's ledger rows, written by
    scaffold-timing's OWN ``arm_rows`` (imported: naming and derived arithmetic are
    its), keeping the ``n64.ckpt600.`` statistic keys."""
    col = ST._Rows()
    ST.arm_rows(col, N_TRAIN, per_control)
    return {
        r["key"]: [float(x) for x in r["samples"]]
        for r in col.rows
        if r["kind"] == "statistic" and r["key"].startswith(CONTROL_PREFIX)
    }


def control_2(
    reference: dict[str, list[float]],
    source_root: Path,
    measure_fn: Callable[[Path, int, int, int], dict],
    log: Callable[[str], None] = lambda s: print(s, flush=True),
) -> dict:
    """PREREG control 2: re-measure each seed's start ``ckpt-000600.pt`` (read-only)
    and compare every reference key, per seed, within ``REPRO_TOL`` (scaffold-timing's
    ``reproduction_control``, imported). Never raises: a raise is ``ok`` False."""
    per: dict[int, dict[int, dict]] = {s: {} for s in SEEDS}
    try:
        for s in SEEDS:
            per[s][CONTROL_K] = measure_fn(
                ST.seed_dir(source_root, N_TRAIN, s), s, CONTROL_K, N_TRAIN
            )
            log(f"  control 2: measured seed {s} ckpt{CONTROL_K}")
        out = ST.reproduction_control(reference, remeasured_rows(per), REPRO_TOL)
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "per_seed": {}}
    out["error"] = None
    return out


# --------------------------------------------------------------------------- #
# the parent loop
# --------------------------------------------------------------------------- #


def read_resume_checks(root: Path, k: int) -> dict[int, dict | None]:
    out: dict[int, dict | None] = {}
    for s in SEEDS:
        f = root / arm_name(k) / f"seed{s}" / "resume_check.json"
        out[s] = json.loads(f.read_text()) if f.exists() else None
    return out


def run_all(
    root: Path,
    *,
    control_fn: Callable[[], dict],
    preflight_fn: Callable[[], dict],
    spawn: Callable[[int, int, Path], Any],
    measure_fn: Callable[[Path, int, int, int], dict],
    deadline: float,
    clock: Callable[[], float] = time.time,
    **kw,
) -> dict:
    """PREREG order: control 2 (measurement path); control 3 + closure; the arms
    600, 400, 300, 200, 100 through fresh-stream's ``run_arm``. A failure before the
    arms stops everything; a raised measurement stops every later arm; an arm whose
    start comes at or after the deadline is not run."""
    res: dict[str, Any] = {
        "control_2": None,
        "preflight": None,
        "arms": {},
        "resume_check": {},
        "not_run": {},
        "error": None,
        "stopped": None,
    }

    def skip_all(why: str) -> dict:
        res["stopped"] = why
        for k in ARM_ORDER:
            res["not_run"].setdefault(k, why)
        return res

    if clock() >= deadline:
        return skip_all("deadline")
    res["control_2"] = c2 = control_fn()
    if not c2.get("ok"):
        if c2.get("error"):
            res["error"] = c2["error"]
        return skip_all("control_2 (measurement path unchanged) failed or raised")
    res["preflight"] = pf = preflight_fn()
    if not pf.get("ok"):
        return skip_all(f"preflight failed before any step: {pf.get('error')}")
    for k in ARM_ORDER:
        if res["error"] is not None:
            res["not_run"][k] = "an earlier arm's measurement raised"
            continue
        if clock() >= deadline:
            res["not_run"][k] = "deadline"
            continue
        res["arms"][k] = FS.run_arm(
            arm_name(k),
            root,
            (end_step(k),),
            spawn=lambda s, d, k=k: spawn(k, s, d),
            measure_fn=measure_fn,
            deadline=deadline,
            clock=clock,
            **kw,
        )
        res["resume_check"][k] = read_resume_checks(root, k)
        res["error"] = res["error"] or res["arms"][k]["error"]
    return res


# --------------------------------------------------------------------------- #
# the rule (PREREG "Primary readout")
# --------------------------------------------------------------------------- #


def arm_cell(res: dict, k: int) -> dict:
    """One arm's cell: readable or not (and every reason), R and C at ckpt k + 500
    per seed, and U(k) -- None when the cell is unreadable."""
    end = end_step(k)
    arm = res.get("arms", {}).get(k)
    why: list[str] = []
    per: dict = {}
    if arm is None:
        why.append(f"not run ({(res.get('not_run') or {}).get(k)})")
    else:
        per = arm.get("per", {})
        if arm.get("error"):
            why.append(f"a measurement raised ({arm['error']})")
        rc = (res.get("resume_check") or {}).get(k) or {}
        bad = [f"seed{s}" for s in SEEDS if not (rc.get(s) or {}).get("ok")]
        if bad:
            why.append(
                f"control_1 (resume is exact) failed or missing on {', '.join(bad)}"
            )
        miss = [f"seed{s}" for s in SEEDS if end not in per.get(s, {})]
        if miss:
            why.append(f"ckpt{end} not measured for {', '.join(miss)}")
    ro = FS.arm_readout(per, (end,)).get(end)
    return {
        "k": k,
        "end_step": end,
        "readable": not why,
        "unreadable_because": why,
        "R_quantity": ro["R_quantity"] if ro else None,
        "R": ro["R"] if ro else None,
        "C_quantity": ro["C_quantity"] if ro else None,
        "C": ro["C"] if ro else None,
        "U": ro["R"] if (ro and not why) else None,
    }


def k_star(u: dict[int, bool | None]) -> int | None:
    """The SMALLEST k in the sweep with U(k); None if U holds nowhere."""
    hits = [k for k in ARMS_K if u.get(k) is True]
    return min(hits) if hits else None


def classify(kstar: int | None) -> str:
    """PREREG classification table, rows 2-5."""
    if kstar is None:
        return "UNLOCK_AFTER_600"
    if kstar in ONSET_K:
        return "AT_RETRIEVAL_ONSET"
    if kstar in MEMORISATION_K:
        return "AT_MEMORISATION"
    if kstar in EARLY_K:
        return "EARLY"
    raise ValueError(f"k*={kstar} is not in the sweep {ARMS_K}")


def monotonicity(u: dict[int, bool | None]) -> dict:
    """PREREG "Monotonicity is reported, not assumed": every pair ``(k1, k2)`` with
    ``k1 < k2``, U(k1) and not U(k2). Only readable cells take part."""
    ks = [k for k in ARMS_K if u.get(k) is not None]
    violations = [[k1, k2] for k1 in ks for k2 in ks if k1 < k2 and u[k1] and not u[k2]]
    return {"monotone": not violations, "violations": violations}


def verdict(res: dict) -> dict:
    """PREREG classification table, row for row. Every reason for inconclusive is
    listed, not only the first."""
    cells = {k: arm_cell(res, k) for k in ARMS_K}
    u = {k: c["U"] for k, c in cells.items()}
    base: dict[str, Any] = {"cells": cells, "U": u}
    why: list[str] = []
    c2 = res.get("control_2")
    if c2 is None:
        why.append("control_2 (measurement path unchanged) did not run")
    elif not c2.get("ok"):
        bad = [f"seed{s}" for s, p in (c2.get("per_seed") or {}).items() if not p["ok"]]
        why.append(
            "control_2 (measurement path unchanged) failed"
            + (f" on {', '.join(bad)}" if bad else "")
            + (f" ({c2['error']})" if c2.get("error") else "")
        )
    pf = res.get("preflight")
    if not pf or not pf.get("ok"):
        why.append(
            "control_3 (stream disjointness / vocabulary closure) failed or did not run"
            + (f" ({pf['error']})" if pf and pf.get("error") else "")
        )
    for k in ARM_ORDER:
        if not cells[k]["readable"]:
            why.append(
                f"arm {arm_name(k)} unreadable: "
                + "; ".join(cells[k]["unreadable_because"])
            )
    if why:
        return {
            **base,
            "k_star": None,
            "monotonicity": monotonicity(u),
            "classification": "inconclusive",
            "exit": Exit.DID_NOT_RUN,
            "detail": "; ".join(why),
        }
    ks = k_star(u)
    cls = classify(ks)
    held = [arm_name(k) for k in ARMS_K if u[k]]
    return {
        **base,
        "k_star": ks,
        "monotonicity": monotonicity(u),
        "classification": cls,
        "exit": Exit.OK,
        "detail": f"U holds for {', '.join(held) or 'no arm'}; k* "
        f"{arm_name(ks) if ks is not None else 'none'}: {cls} -- {READING[cls]}",
    }


# --------------------------------------------------------------------------- #
# the ledger
# --------------------------------------------------------------------------- #


def start_quantities(doc: dict) -> dict[int, dict[str, list[float]]]:
    """The start checkpoints' R and M quantities (and probe accuracy), as the
    scaffold-timing ledger holds them: ``{k: {name: [per seed]}}``."""
    rows = {r["key"]: r for r in doc["rows"] if r.get("kind") == "statistic"}
    names = (
        "R_quantity",
        "M_quantity",
        f"train.live.{BUCKET}.answer_acc",
        f"heldout.live.{BUCKET}.answer_acc",
    )
    return {
        k: {
            n: [float(x) for x in rows[f"n{N_TRAIN}.ckpt{k}.{n}"]["samples"]]
            for n in names
        }
        for k in ARMS_K
    }


def write_rows(led, res: dict, v: dict, start: dict[int, dict[str, list[float]]]) -> None:
    root = res.get("root")
    for k in ARM_ORDER:
        a = arm_name(k)
        arm = res["arms"].get(k)
        if arm is not None:
            # fresh-stream's own rows (every set / condition / bucket / readout,
            # memory gates, R, M, C, probe R), at this arm's one checkpoint.
            with fresh_stream_with(CHECKPOINTS={a: (end_step(k),)}):
                FS.arm_rows(led, a, arm["per"])
        for n, xs in start[k].items():
            led.note(
                f"{a}.start.{n}",
                xs,
                how=f"{REF_LEDGER} n{N_TRAIN}.ckpt{k}.{n} samples (this arm's start "
                "checkpoint), seed order",
            )
        cell = v["cells"][k]
        led.note(
            f"{a}.U", cell["U"], how="R at ckpt k + 500 on every seed; None if unreadable"
        )
        led.note(f"{a}.readable", cell["readable"], how="arm_cell()")
        led.note(f"{a}.unreadable_because", cell["unreadable_because"], how="arm_cell()")
        rc = (res.get("resume_check") or {}).get(k) or {}
        led.note(
            f"{a}.control_1.ok",
            [(rc.get(s) or {}).get("ok") for s in SEEDS],
            how="per seed: the child's resume_check.json (model and optimizer == the "
            "start checkpoint exactly, before the first step)",
        )
        for part in ("model", "optimizer"):
            led.note(
                f"{a}.control_1.{part}_n_tensors",
                [((rc.get(s) or {}).get(part) or {}).get("n_tensors") for s in SEEDS],
                how="per seed: tensors compared with torch.equal",
            )
        led.note(
            f"{a}.control_1.detail",
            {f"seed{s}": rc.get(s) for s in SEEDS},
            how="the children's resume_check.json, verbatim",
        )
        if arm is not None and root is not None:
            wins = {
                s: FS.stream_loss_windows(Path(root) / a / f"seed{s}" / "heartbeat.jsonl")
                for s in SEEDS
            }
            led.note(
                f"{a}.stream_answer_loss_window_means",
                {
                    f"seed{s}": {str(w0): x for w0, x in w.items()}
                    for s, w in wins.items()
                },
                how=f"heartbeat loss_answer_tokens, mean over each complete "
                f"{FS.STREAM_LOSS_WINDOW}-step window, keyed by its first step",
            )
            led.note(
                f"{a}.stream_answer_loss_first_window_below",
                {f"seed{s}": FS.first_window_below(w) for s, w in wins.items()},
                how="first window whose mean < stream_loss_threshold; None if none",
            )
        led.note(f"{a}.stopped_because", arm and arm["stopped"], how="run_arm()")
        led.note(f"{a}.measurement_error", arm and arm["error"], how="run_arm()")
        led.note(f"{a}.elapsed_s", arm and arm["elapsed_s"], how="parent wall clock")
        led.note(
            f"{a}.checkpoints_measured",
            arm and {f"seed{s}": arm["checkpoints_measured"][s] for s in SEEDS},
            how="run_arm()",
        )
    c2 = res.get("control_2") or {}
    per_seed = c2.get("per_seed") or {}
    for f in ("ok", "n_keys_compared", "max_abs_diff"):
        led.note(
            f"control_2.{f}",
            [per_seed[s][f] if s in per_seed else None for s in SEEDS],
            how=f"per seed, seed order {SEEDS}: every {CONTROL_PREFIX}* statistic key "
            f"of {REF_LEDGER} vs the re-measured start checkpoint, rows by "
            f"{ST.EXPERIMENT}::arm_rows",
        )
    led.note(
        "control_2.failures",
        {f"seed{s}": per_seed[s]["failures"] for s in per_seed},
        how="every (key, reference, measured, abs_diff) outside the tolerance",
    )
    led.note("control_2.missing_keys", c2.get("missing_keys"), how="control_2()")
    led.note(
        "control_2.n_reference_keys",
        c2.get("n_reference_keys"),
        how=f"statistic keys under {CONTROL_PREFIX} in {REF_LEDGER}",
    )
    led.note("control_2.error", c2.get("error"), how="control_2(); None if none")
    led.note("control_2.passed", c2.get("ok"), how="all seeds ok")
    pf = res.get("preflight") or {}
    led.note("preflight.ok", pf.get("ok"), how="per-arm disjointness + closure")
    led.note("preflight.error", pf.get("error"), how="preflight(); None if none")
    led.note(
        "stream_disjointness",
        pf.get("arms"),
        how=f"per arm: ids of STREAM.window(t, {BATCH}), t in [k, k + {STEPS}), vs "
        "PROBE, the vocabulary documents and HELDOUT",
    )
    led.note(
        "vocabulary_closure",
        {f"seed{s}": c for s, c in (pf.get("closure") or {}).items()},
        how=f"{FS.EXPERIMENT}::vocabulary_closure(seed, {closure_iters()}): the ids of "
        "steps [0, max k + 500) and both measurement sets",
    )
    led.note(
        "U", {arm_name(k): v["U"][k] for k in ARMS_K}, how="per arm; None if unreadable"
    )
    ks = v["k_star"]
    led.note("k_star", ks, how="smallest k with U(k); None if none or inconclusive")
    led.note("monotone", v["monotonicity"]["monotone"], how="monotonicity()")
    led.note(
        "monotonicity_violations",
        [[arm_name(a), arm_name(b)] for a, b in v["monotonicity"]["violations"]],
        how="every (k1 < k2) with U(k1) and not U(k2)",
    )
    led.note("arm_order", [arm_name(k) for k in ARM_ORDER], how="PREREG arm_order")
    led.note(
        "arms_not_run",
        {arm_name(k): w for k, w in res["not_run"].items()},
        how="run_all()",
    )
    led.note("stopped_because", res.get("stopped"), how="run_all(); None if none")
    led.note("measurement_error", res.get("error"), how="run_all(); None if none")
    led.note("classification", v["classification"], how="PREREG classification table")
    led.note("classification_detail", v["detail"], how="verdict()")
    led.note(
        "reading",
        READING.get(v["classification"]),
        how="PREREG 'Primary readout' table, the row of the classification",
    )
    led.note("delta", DELTA, how="PREREG thresholds (scaffold-timing's, imported)")
    led.note("c_threshold", C_THRESHOLD, how="fresh-stream's C threshold, imported")
    led.note(
        "stream_loss_threshold", STREAM_LOSS_THRESHOLD, how="fresh-stream's, imported"
    )
    led.note("reproduction_tolerance", REPRO_TOL, how="PREREG thresholds")
    led.note("steps_per_arm", STEPS, how="PREREG arm")
    led.note("decision_rule", DECISION_RULE, how="PREREG front matter, verbatim")
    led.note("expected", EXPECTED, how="PREREG 'Author's expectation', in the manifest")
    led.note("deadline", DEADLINE, how="PREREG thresholds")


def manifest(
    source_root: Path, parent_threads: int, ref_sha: str, start_sha: dict
) -> dict:
    """Frozen before any child starts."""
    return {
        **S003.CONFIG,
        "prereg": PREREG,
        "prereg_commit": PREREG_COMMIT,
        "question": QUESTION,
        "decision_rule": DECISION_RULE,
        "expected": EXPECTED,
        "reading": READING,
        "instrument": f"{CSC.EXPERIMENT}::measure_checkpoint (imported, identity), "
        f"n={N_TRAIN}",
        "child": f"{FS.EXPERIMENT}::single('B') with RESUME_STEP = k, "
        f"iters = k + {STEPS}",
        "arms_k": list(ARMS_K),
        "arm_order": list(ARM_ORDER),
        "seeds": SEEDS,
        "steps_per_arm": STEPS,
        "ckpt_every": CKPT_EVERY,
        "measured_checkpoint": {arm_name(k): end_step(k) for k in ARMS_K},
        "stream": dataclasses.asdict(STREAM),
        "stream_documents": {
            arm_name(k): [
                STREAM.window(k, BATCH).start,
                STREAM.window(end_step(k) - 1, BATCH).stop,
            ]
            for k in ARMS_K
        },
        "heldout_documents": list(HELDOUT),
        "probe_documents": list(PROBE),
        "vocab_documents": list(VOCAB_DOCUMENTS),
        "delta": DELTA,
        "c_threshold": C_THRESHOLD,
        "stream_loss_threshold": STREAM_LOSS_THRESHOLD,
        "reproduction_tolerance": REPRO_TOL,
        "control_2": f"measure_checkpoint of n{N_TRAIN}/seed<s>/ckpt-{CONTROL_K:06d}.pt "
        f"vs {REF_LEDGER} {CONTROL_PREFIX}* statistic keys",
        "deadline": DEADLINE,
        "device": DEVICE,
        "threads_per_seed": THREADS_PER_SEED,
        "seeds_trained_in_parallel": True,
        "arms_in_parallel": False,
        "parent_torch_num_threads": parent_threads,
        "source_root": str(source_root),
        "start_checkpoint_sha256": start_sha,
        "reference_ledger": REF_LEDGER,
        "reference_ledger_sha256": ref_sha,
        "torch": torch.__version__,
        "python": sys.version.split()[0],
    }


def execute(led, root: Path, *, results_path: Path | None, start: dict, **kw) -> Exit:
    res = run_all(root, **kw)
    res["root"] = str(root)
    # Written first, so a raise below still leaves every measurement on disk.
    root.mkdir(parents=True, exist_ok=True)
    (root / "raw.json").write_text(json.dumps(res, indent=2, default=str) + "\n")
    for k, arm in res["arms"].items():
        a = arm_name(k)
        for s in SEEDS:
            argv = arm["argv"].get(s)
            if argv:
                rel = " ".join(
                    ["uv run python"]
                    + [
                        str(Path(x).relative_to(ROOT)) if x.startswith(str(ROOT)) else x
                        for x in argv[1:]
                    ]
                )
                led.command(
                    rel, exit_code=arm["exit_codes"][s], note=f"{a} seed {s} training"
                )
            info = root / a / f"seed{s}" / "result.json"
            if info.exists():
                r = json.loads(info.read_text())
                led.note(
                    f"{a}.seed{s}.training",
                    {
                        x: r.get(x)
                        for x in (
                            "config_hash",
                            "steps_done",
                            "torch_num_threads",
                            "run_id",
                            "stream",
                            "start_checkpoint",
                        )
                    },
                    how="the child's result.json (train() return + torch threads)",
                )
    v = verdict(res)
    measured = sorted({s for a in res["arms"].values() for s in SEEDS if a["per"].get(s)})
    led.run_meta(
        device=DEVICE,
        seeds_actually_run=measured or None,
        steps_requested=max(end_step(k) for k in ARMS_K),
        steps_done=max(
            (
                max(a["per"][s])
                for a in res["arms"].values()
                for s in SEEDS
                if a["per"][s]
            ),
            default=0,
        ),
    )
    crashed = res["error"] is not None or any(
        a["failed_seeds"]
        or (a["stopped"] is None and any(rc != 0 for rc in a["exit_codes"].values()))
        for a in res["arms"].values()
    )
    complete = not res["not_run"] and all(
        a["stopped"] is None for a in res["arms"].values()
    )
    led.status("crashed" if crashed else ("ok" if complete else "partial"))
    write_rows(led, res, v, start)
    path = led.write()
    if results_path is not None:
        results_path.write_text(
            render_results(json.loads(path.read_text()), read_brief_errors())
        )
    print(f"ledger: {path}  classification={v['classification']}  exit={int(v['exit'])}")
    return v["exit"]


# --------------------------------------------------------------------------- #
# RESULTS.md, rendered from the ledger
# --------------------------------------------------------------------------- #


def read_brief_errors(path: Path | None = None) -> str | None:
    path = path or ROOT / BRIEF_ERRORS
    return path.read_text() if path.exists() else None


def render_results(doc: dict, brief_errors: str | None = None) -> str:
    """Every number beside its ledger key (`render_scoreboard.py --audit`). No number
    is typed here."""
    rows = RC._rows(doc)
    val, fmt = RC._val, RC._fmt
    prov = doc.get("provenance", {})
    rid = doc.get("run_id")
    out = [
        "# Scaffold dose -- RESULTS",
        "",
        f"<!-- GENERATED by `{EXPERIMENT}` from runs/{rid}/ledger.json. Do not edit: "
        "regenerate with `--render-results`. -->",
        "",
        f"Command: `experiments/scaffold-dose/run.sh` (the parent is `uv run python "
        f"{EXPERIMENT}`). Ledger `status` of run `{rid}`: **{doc.get('status')}**.",
        "",
        f"Provenance of run `{rid}`: `provenance.git_sha` `{prov.get('git_sha')}` · "
        f"`device` {doc.get('device')} · hardware `provenance.machine` "
        f"`{prov.get('machine')}`, `provenance.platform` `{prov.get('platform')}` · "
        f"`seeds_actually_run` {doc.get('seeds_actually_run')} · `config_hash` "
        f"`{doc.get('config_hash')}`.",
        "",
        f"The decision rule of run `{rid}` is its `decision_rule` row (the PREREG's, "
        "verbatim); it is not restated here.",
        "",
        f"## Classification of run `{rid}`: `classification` "
        f"**{val(rows, 'classification')}**",
        "",
        f"Why, for run `{rid}` (`classification_detail`): "
        f"**{val(rows, 'classification_detail')}**",
        "",
        f"`k_star` {fmt(val(rows, 'k_star'))} · `monotone` "
        f"{fmt(val(rows, 'monotone'))} · `monotonicity_violations` "
        f"{fmt(val(rows, 'monotonicity_violations')) or 'none'} · `measurement_error` "
        f"{fmt(val(rows, 'measurement_error'))} · `stopped_because` "
        f"{fmt(val(rows, 'stopped_because'))}.",
        "",
        "Arms not run (`arms_not_run`): "
        + (
            "; ".join(f"`{a}` {w}" for a, w in (val(rows, "arms_not_run") or {}).items())
            or "none"
        )
        + ".",
        "",
        "Whether the cognitive claim moved: **it cannot move here.** FIFO only, no "
        "retention loss; nothing about RSR, eviction, ψ̂ or Kintsch & van Dijk "
        "(PREREG, *What this does not establish*). No scaling claim.",
        "",
        f"The pre-registered expectation of run `{rid}` is its `expected` row, frozen "
        "in the manifest before any child ran; it is not restated here.",
        "",
        "## Per arm: U, and R at the start and at the end",
        "",
        "Per-seed samples in seed order; the start values are scaffold-timing's "
        "ledger rows for the start checkpoint (`<arm>.start.*`), the end values this "
        f"run's (`<arm>.ckpt<end>.*`); bucket `{BUCKET}`.",
        "",
    ]
    for k in ARM_ORDER:
        a = arm_name(k)
        e = f"{a}.ckpt{end_step(k)}"
        first = val(rows, f"{a}.stream_answer_loss_first_window_below") or {}
        out += [
            f"### Arm `{a}`",
            "",
            f"- `{a}.U` {fmt(val(rows, f'{a}.U'))} · `{a}.readable` "
            f"{fmt(val(rows, f'{a}.readable'))} · `{a}.unreadable_because` "
            f"{fmt(val(rows, f'{a}.unreadable_because')) or 'none'} · "
            f"`{a}.control_1.ok` {fmt(val(rows, f'{a}.control_1.ok'))}.",
            f"- `{a}.start.R_quantity` {fmt(val(rows, f'{a}.start.R_quantity'))} → "
            f"`{e}.R_quantity` {fmt(val(rows, f'{e}.R_quantity'))}.",
            f"- `{a}.start.M_quantity` {fmt(val(rows, f'{a}.start.M_quantity'))} → "
            f"`{e}.M_quantity` {fmt(val(rows, f'{e}.M_quantity'))}.",
            f"- `{e}.C_quantity` {fmt(val(rows, f'{e}.C_quantity'))} · `{e}.C_holds` "
            f"{fmt(val(rows, f'{e}.C_holds'))} · `{e}.probe_R_quantity` "
            f"{fmt(val(rows, f'{e}.probe_R_quantity'))}.",
            f"- `{e}.heldout.live.{BUCKET}.answer_acc` "
            f"{fmt(val(rows, f'{e}.heldout.live.{BUCKET}.answer_acc'))} · "
            f"`{e}.heldout.slots_zeroed.{BUCKET}.answer_acc` "
            f"{fmt(val(rows, f'{e}.heldout.slots_zeroed.{BUCKET}.answer_acc'))} · "
            f"`{e}.train.live.{BUCKET}.answer_acc` "
            f"{fmt(val(rows, f'{e}.train.live.{BUCKET}.answer_acc'))}.",
            "- Stream answer loss, the first complete window below "
            f"`stream_loss_threshold` {fmt(val(rows, 'stream_loss_threshold'))}: "
            + (
                " · ".join(
                    f"`{a}.stream_answer_loss_first_window_below.{s}` {fmt(w)}"
                    for s, w in sorted(first.items())
                )
                or "—"
            )
            + ".",
            f"- `{a}.stopped_because` {fmt(val(rows, f'{a}.stopped_because'))} · "
            f"`{a}.elapsed_s` {fmt(val(rows, f'{a}.elapsed_s'))}.",
            "",
        ]
    out += [
        "## Controls",
        "",
        f"`control_2` (measurement path unchanged, `{CONTROL_PREFIX}*`): "
        f"`control_2.passed` {fmt(val(rows, 'control_2.passed'))} · "
        f"`control_2.n_reference_keys` {fmt(val(rows, 'control_2.n_reference_keys'))} "
        f"· `control_2.ok` {fmt(val(rows, 'control_2.ok'))} · "
        f"`control_2.n_keys_compared` {fmt(val(rows, 'control_2.n_keys_compared'))} "
        f"· `control_2.max_abs_diff` {val(rows, 'control_2.max_abs_diff')!r} · "
        f"`reproduction_tolerance` {val(rows, 'reproduction_tolerance')!r} · "
        f"`control_2.error` {fmt(val(rows, 'control_2.error'))}.",
        "",
    ]
    fails = val(rows, "control_2.failures") or {}
    if any(fails.values()):
        out += [
            "Every key outside the tolerance (`control_2.failures`):",
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
    out += [
        f"`control_3`: `preflight.ok` {fmt(val(rows, 'preflight.ok'))} · "
        f"`preflight.error` {fmt(val(rows, 'preflight.error'))}. Per-arm stream ids "
        "and the closure are the `stream_disjointness` and `vocabulary_closure` rows.",
        "",
        "Every other bucket, condition and readout, the memory gates and the stream "
        "loss per window are in the ledger under `<arm>.ckpt<end>.` and "
        "`<arm>.stream_answer_loss_window_means`.",
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


def dry_run(source_root: Path, now: float | None = None) -> str:
    now = time.time() if now is None else now
    lines = [
        "scaffold-dose dry run (NOTHING TRAINED OR MEASURED).",
        f"control 2 first: {CSC.EXPERIMENT}::measure_checkpoint of "
        f"n{N_TRAIN}/seed<s>/ckpt-{CONTROL_K:06d}.pt x seeds {SEEDS} vs {REF_LEDGER} "
        f"{CONTROL_PREFIX}* within {REPRO_TOL}; fail -> exit 3, no arm",
        f"control 3: per-arm stream disjointness; vocabulary closure over steps "
        f"[0, {closure_iters()}) per seed + both measurement sets",
    ]
    for k in ARM_ORDER:
        d = arm_disjointness(k)
        argv = child_argv(
            k, 0, ROOT / "runs" / RUN_ID / arm_name(k) / "seed0", source_root
        )
        lines.append(
            f"arm {arm_name(k)}: resume ckpt{k} -> {end_step(k)}, ckpt_every "
            f"{CKPT_EVERY}; measure ckpt{end_step(k)}; stream ids "
            f"{d['first_id']}..{d['last_id']} ({d['n_ids']}, repeats {d['repeats']}, "
            f"disjoint {d['ok']}); child argv {argv[2:]}"
        )
        for s in SEEDS:
            f = start_ckpt(k, s, source_root)
            lines.append(f"  start seed {s}: {f}  exists={f.exists()}")
    try:
        ref_manifest = json.loads((ROOT / REF_MANIFEST).read_text())
        verify_start_checkpoints(source_root, ref_manifest)
        lines.append(
            f"start checkpoints: every sha256 == {REF_MANIFEST}, every step == k"
        )
    except (OSError, ValueError, StartCheckpointRefused) as e:
        lines.append(f"start checkpoints REFUSED (the run would exit 3): {e}")
    lines += [
        f"deadline {DEADLINE} ({(deadline_epoch() - now) / 3600:.2f} h from now)",
        f"PREDICTED ~{PRED_S_PER_STEP * STEPS * len(ARMS_K) / 3600:.2f} h of training "
        "(prediction, not premise: fresh-stream heartbeats, linear)",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> Exit:
    ap = ArgumentParser(prog="scaffold-dose")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; run nothing")
    ap.add_argument(
        "--render-results", action="store_true", help="RESULTS.md from the ledger"
    )
    ap.add_argument("--run-id", default=RUN_ID)
    ap.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--k", type=int, choices=ARMS_K, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args(argv)

    if a.single:
        if a.out_dir is None or a.k is None:
            refuse(Exit.DID_NOT_RUN, "--single needs --out-dir and --k")
        a.out_dir.mkdir(parents=True, exist_ok=True)
        r = single(a.k, a.seed, a.out_dir, source_root=a.source_root)
        (a.out_dir / "result.json").write_text(
            json.dumps(r, indent=2, default=str) + "\n"
        )
        return Exit.OK
    if a.dry_run:
        print(dry_run(a.source_root))
        return Exit.OK
    root = ROOT / "runs" / a.run_id
    results = ROOT / "experiments" / "scaffold-dose" / "RESULTS.md"
    if a.render_results:
        path = root / "ledger.json"
        if not path.exists():
            refuse(Exit.DID_NOT_RUN, f"{path} is absent: the run has not happened")
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
    if time.time() >= deadline_epoch():
        refuse(Exit.DID_NOT_RUN, f"the PREREG deadline {DEADLINE} has passed")
    ref_path = ROOT / REF_LEDGER
    ref_doc = json.loads(ref_path.read_text())
    if ref_doc.get("device") != DEVICE:
        refuse(Exit.DID_NOT_RUN, f"{REF_LEDGER} device is {ref_doc.get('device')!r}")
    reference = reference_rows(ref_doc)
    if not reference:
        refuse(Exit.DID_NOT_RUN, f"{REF_LEDGER} has no {CONTROL_PREFIX}* statistic keys")
    start = start_quantities(ref_doc)
    ref_manifest = json.loads((ROOT / REF_MANIFEST).read_text())
    try:
        start_sha = verify_start_checkpoints(a.source_root, ref_manifest)
    except StartCheckpointRefused as e:
        refuse(Exit.DID_NOT_RUN, f"start checkpoints refused: {e}")
    # Measured at the thread count the reference was measured at (scaffold-timing
    # sequential mode, its manifest's torch_threads), as fresh-stream did.
    torch.set_num_threads(int(ref_manifest["torch_threads"]))

    led = ledger_mod.Ledger(a.run_id, question=QUESTION)
    led.command(
        " ".join(["uv run python", EXPERIMENT, *sys.argv[1:]]),
        exit_code=None,
        note="this run (the parent)",
    )
    m = led.manifest(
        manifest(a.source_root, torch.get_num_threads(), sha256_file(ref_path), start_sha)
    )
    print(f"manifest frozen: {m}", flush=True)
    led.note("reference_ledger_sha256", sha256_file(ref_path), how=f"sha256 {REF_LEDGER}")
    led.note("start_checkpoint_sha256", start_sha, how="sha256 of every start checkpoint")
    led.note("parent_torch_threads", torch.get_num_threads(), how=f"{REF_MANIFEST}")
    return execute(
        led,
        root,
        results_path=results,
        start=start,
        control_fn=lambda: control_2(reference, a.source_root, measure_checkpoint),
        preflight_fn=preflight,
        spawn=lambda k, s, d: Child(k, s, d, a.source_root),
        measure_fn=measure_checkpoint,
        deadline=deadline_epoch(),
    )


if __name__ == "__main__":
    run_main(main)
