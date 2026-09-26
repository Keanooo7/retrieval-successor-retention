"""Fresh escape -- does a fresh start escape the chance plateau if trained longer?

Pre-registration: `experiments/fresh-escape/PREREG.md`, committed alone at d5b9a23,
ahead of this file. **No new training code:** the child is fresh-stream's resume
child (`experiments/fresh-stream/run.py`, imported, not copied) -- the corpus-size
N = 64 ``train()`` call with ``stream=`` and ``resume=`` added, inside fresh-stream's
``ResumeCheck`` -- started from fresh-stream's OWN arm-A ``ckpt-003000.pt`` and
trained ``3000 -> 9000`` on the same stream (step t trains documents
``[4160 + 16 t, 4160 + 16 t + 16)``). ``train()`` in `src/rsr/train/loop.py` is not
touched. Arm A's ``train()`` call and arm B's differ only in ``resume=``, so resuming
an arm-A checkpoint through the arm-B path is arm A continued.

Order (PREREG "Controls"; each failure overrides everything after it):

0. **Before the manifest (control 3, first half):** every start checkpoint
   ``A/seed<s>/ckpt-003000.pt`` exists, has the sha256 in the PREREG front matter
   and stores step 3000. Any mismatch: refused, exit 3.
1. **Control 1, measurement path unchanged, FIRST:** each seed's start checkpoint
   re-measured with the imported ``measure_checkpoint`` (n = 64) and turned into rows
   by fresh-stream's own ``arm_rows``; every ``A.ckpt3000.*`` statistic key of
   `runs/fresh-stream/ledger.json` must match per seed within ``REPRO_TOL``. Fail,
   missing key or a raise: exit 3, nothing trains.
2. **Control 3, second half:** the stream ids ``[4160 + 16*3000, 4160 + 16*9000)``
   are disjoint from the probe / vocabulary documents ``[0, 64)`` and the held-out
   ``[4096, 4160)`` and never repeat; every word of those documents and of both
   measurement sets is in the seed's ``[0, 64)`` map (fresh-stream's closure,
   imported, over exactly those ids). Fail: exit 3, nothing trains.
3. **The arm:** three seeds as parallel children at 5 threads (fresh-stream's
   ``run_arm``, imported). Each child runs **control 2** (resume is exact,
   fresh-stream's ``ResumeCheck``) before its first step. The parent measures ckpt
   4000 ... 9000 (n = 64) as they appear.
4. The rule: P, R(P), the stream-loss windows at or before P, the classification.

The parent enforces the absolute ``DEADLINE``: at it the children are stopped and
whatever listed checkpoints exist are measured (fresh-stream's ``run_arm``). Reading
of "measured on every seed before the deadline": a checkpoint written before the
children were stopped counts for P even if its measurement finishes after the
deadline. STIRRING reads this run's heartbeats (steps 3000 onward); fresh-stream's
own windows over steps 0-2999 were below the threshold on no seed
(``A.stream_answer_loss_first_window_below`` null on every seed).

Usage::

    uv run --extra dev python experiments/fresh-escape/run.py --dry-run
    experiments/fresh-escape/run.sh        # the real run; exit code -> run.rc
    uv run --extra dev python experiments/fresh-escape/run.py --render-results

Exit codes (`rsr.exit_codes`): 0 a classification was reached · 3 did not run or
inconclusive (a refused precondition, a control failed, a measurement raised, P <
6000).
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

EXPERIMENT = "experiments/fresh-escape/run.py"
RUN_ID = "fresh-escape"
PREREG = "experiments/fresh-escape/PREREG.md"
PREREG_COMMIT = "d5b9a23"
BRIEF_ERRORS = "experiments/fresh-escape/BRIEF-ERRORS.md"


def _load(name: str, rel: str):
    """Import an experiment script by path. Imported, never copied."""
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: fresh-stream, whose resume child, controls, parent loop and readouts this run
#: reuses. ONE instance: its ST / CSC are the very modules it measured with.
FS = _load("_fresh_escape_fs", "experiments/fresh-stream/run.py")
ST = FS.ST
CSC = FS.CSC
RC = FS.RC
S003 = FS.S003

#: THE measurement (PREREG "instrument"): the corpus-size curve's function object,
#: not a copy; `tests/test_fresh_escape.py` checks the identity.
measure_checkpoint = FS.measure_checkpoint

# --------------------------------------------------------------------------- #
# 🔒 PREREG.md, transcribed. Changing any is changing the PREREG.
# --------------------------------------------------------------------------- #

SEEDS = [0, 1, 2]
ARM = "A"  # fresh-stream arm A, continued: ledger prefix and directory
RESUME_STEP = 3000
END_STEP = 9000
CHECKPOINTS = (4000, 5000, 6000, 7000, 8000, 9000)
MIN_PRIMARY_CKPT = 6000
CKPT_EVERY = FS.CKPT_EVERY
N_TRAIN = FS.CONTROL_ARM  # 64: the vocabulary / measurement arm
STREAM = FS.STREAM  # step t trains documents [4160 + 16 t, 4160 + 16 t + 16)
BATCH = FS.BATCH
HELDOUT = FS.HELDOUT
PROBE = FS.PROBE
VOCAB_DOCUMENTS = FS.VOCAB_DOCUMENTS
#: "DELTA: 0.03 -- unchanged from scaffold-timing, fresh-stream and scaffold-dose".
DELTA = FS.DELTA
#: "C: 0.8875 = 0.9375 - 0.05 -- fresh-stream's": imported.
C_THRESHOLD = FS.C_THRESHOLD
#: "stream_loss_threshold: ln 16 - 0.10 = 2.6726 -- fresh-stream's": imported.
STREAM_LOSS_THRESHOLD = FS.STREAM_LOSS_THRESHOLD
STREAM_LOSS_WINDOW = FS.STREAM_LOSS_WINDOW  # "a 100-step window"
REPRO_TOL = 1e-6
DEADLINE = "2026-09-26T08:30:00-07:00"  # 08:30 America/Los_Angeles, PDT
BUCKET = FS.BUCKET
DEVICE = FS.DEVICE
THREADS_PER_SEED = FS.THREADS_PER_SEED

#: The start checkpoints: fresh-stream's arm-A outputs. PROTECTED: read only.
SOURCE_ROOT = (
    Path.home()
    / "retrieval-successor-retention"
    / ".worktrees"
    / "fresh-stream"
    / "runs"
    / "fresh-stream"
)
#: PREREG front matter ``start_checkpoint_sha256``.
START_SHA256 = {
    0: "f8adcebeb730d160d62ce3d46de0e173a82fe25efd4730a32bf733bf607a2e64",
    1: "9763cbc61f06496919058036113583b5f57a2aa923ab7d5e80b1b35882d4bb98",
    2: "f79c006f1a23f9d72cd9d8ba72299071ac2e738997d5a7965e1f42f6c76545fb",
}
REF_LEDGER = "runs/fresh-stream/ledger.json"
REF_MANIFEST = "runs/fresh-stream/manifest.json"
CONTROL_PREFIX = f"{ARM}.ckpt{RESUME_STEP}."

#: PREREG "Primary readout" table, by classification.
READING = {
    "ESCAPES": "a fresh start learns memory-mediated retrieval with enough steps; "
    "memorisation speeds it up rather than being required",
    "STIRRING": "learning has begun on at least one seed but has not reached R by P",
    "NO_ESCAPE": "no sign of learning in P steps of a non-repeating stream",
}

QUESTION = (
    "Does a fresh-start TG model escape the chance plateau on a non-repeating stream "
    "if it is simply trained longer? Continue fresh-stream arm A from its ckpt 3000 "
    "to ckpt 9000 on the same stream."
)
#: PREREG front matter ``decision_rule``, verbatim (the YAML fold joined).
DECISION_RULE = (
    "A failed control or a measurement that raised makes the result inconclusive "
    "(exit 3). The primary checkpoint P is the largest listed checkpoint measured on "
    "every seed before the deadline; if P < 6000 the result is inconclusive. ESCAPES "
    "if R(P) holds on every seed; STIRRING if R(P) fails but at least one seed's "
    "stream answer loss has a 100-step window below 2.6726 at or before P; NO_ESCAPE "
    "otherwise, reported as NO_ESCAPE by P."
)
#: PREREG "Author's expectation", verbatim (whitespace joined).
EXPECTED = (
    "NO_ESCAPE by 9000 (about 60 %). If anything moves, it is one seed, late, reading "
    "as STIRRING (about 30 %). ESCAPES about 10 %. The basis is the flat 2.78 plateau "
    "from step 500 to 3000 and the very slow creep."
)

POLL_S = FS.POLL_S
STOP_GRACE_S = FS.STOP_GRACE_S
#: Prediction for --dry-run only; no rule reads it (PREREG: "about 3.33 s per step").
PRED_S_PER_STEP = 3.33


class StartCheckpointRefused(RuntimeError):
    """A start checkpoint is absent, stores another step, or is not the file the
    PREREG front matter names (sha256)."""


def deadline_epoch() -> float:
    return _dt.datetime.fromisoformat(DEADLINE).timestamp()


def sha256_file(path: Path) -> str:
    return ST.sha256_file(path)


def start_ckpt(seed: int, source_root: Path = SOURCE_ROOT) -> Path:
    """fresh-stream arm A's ``A/seed<s>/ckpt-003000.pt``."""
    return RC.ckpt_path(Path(source_root) / ARM / f"seed{seed}", RESUME_STEP)


@contextlib.contextmanager
def fresh_stream_with(**overrides):
    """fresh-stream's module with some of its names set for one call, restored
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
# control 3: the start checkpoints (before the manifest)
# --------------------------------------------------------------------------- #


def verify_start_checkpoints(
    source_root: Path,
    *,
    sha_fn: Callable[[Path], str] = sha256_file,
    step_fn: Callable[[Path], int] = RC.stored_step,
) -> dict[str, str]:
    """PREREG control 3: every start checkpoint exists, its sha256 equals the front
    matter and it stores step 3000. Returns ``{"seed<s>": sha256}``; raises
    ``StartCheckpointRefused`` naming EVERY problem, not only the first. Reads only."""
    out: dict[str, str] = {}
    problems: list[str] = []
    for s in SEEDS:
        f = start_ckpt(s, source_root)
        if not f.exists():
            problems.append(f"{f} is absent")
            continue
        sha = sha_fn(f)
        out[f"seed{s}"] = sha
        if sha != START_SHA256[s]:
            problems.append(
                f"{f} sha256 {sha} is not the PREREG's ({START_SHA256[s]}) for seed {s}"
            )
            continue
        if step_fn(f) != RESUME_STEP:
            problems.append(f"{f} does not store step {RESUME_STEP}")
    if problems:
        raise StartCheckpointRefused("; ".join(problems))
    return out


# --------------------------------------------------------------------------- #
# control 3: the stream ids, and the vocabulary closure
# --------------------------------------------------------------------------- #


def escape_windows(iters: int = END_STEP) -> list[range]:
    """The document ids of every step ``t`` in ``[RESUME_STEP, iters)`` -- the steps
    this run trains. Substituted for fresh-stream's ``stream_windows`` (which starts
    at 0) while its closure runs, so the closure covers exactly the ids used."""
    return [STREAM.window(t, BATCH) for t in range(RESUME_STEP, iters)]


def stream_ids() -> list[int]:
    """``[4160 + 16*3000, 4160 + 16*9000)``, in step order."""
    return [i for w in escape_windows() for i in w]


def stream_disjointness() -> dict:
    """PREREG control 3: no id in ``PROBE`` (= the vocabulary documents) or
    ``HELDOUT``, and no id repeats."""
    ids = stream_ids()
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


def vocabulary_closure(seed: int) -> dict:
    """fresh-stream's ``vocabulary_closure``, imported, over this run's ids only."""
    with fresh_stream_with(stream_windows=escape_windows):
        return FS.vocabulary_closure(seed, END_STEP)


def preflight() -> dict:
    """Control 3 (stream) and the closure, before any step. Never raises: a failure
    is a record with ``ok`` False and the reason."""
    out: dict[str, Any] = {"ok": False, "error": None, "closure": {}}
    try:
        out["disjointness"] = d = stream_disjointness()
        if not d["ok"]:
            out["error"] = "the stream is not disjoint or repeats a document"
            return out
        for s in SEEDS:
            out["closure"][s] = vocabulary_closure(s)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    out["ok"] = True
    return out


# --------------------------------------------------------------------------- #
# one seed, in its own child process: fresh-stream's resume child, from ckpt 3000
# --------------------------------------------------------------------------- #


def single(seed: int, out_dir: Path, *, source_root: Path = SOURCE_ROOT) -> dict:
    """fresh-stream's resume ``single()`` (the corpus-size N = 64 call + the stream +
    ``resume=``, inside its ``ResumeCheck``) with the start checkpoint set to arm A's
    ``ckpt-003000.pt`` and the run ending at 9000. Nothing retyped."""
    with fresh_stream_with(
        start_ckpt=lambda s, root=source_root: start_ckpt(s, root),
        RESUME_STEP=RESUME_STEP,
    ):
        r = FS.single("B", seed, out_dir, iters=END_STEP, source_root=source_root)
    r["arm"] = ARM
    r["continued_from"] = "fresh-stream arm A"
    r["start_checkpoint"] = str(start_ckpt(seed, source_root))
    return r


def child_argv(seed: int, out_dir: Path, source_root: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single",
        "--seed",
        str(seed),
        "--out-dir",
        str(out_dir),
        "--source-root",
        str(source_root),
    ]


class Child(RC.Child):
    """The retrieval curve's child (5 OMP/MKL threads), launched on THIS script."""

    def __init__(self, seed: int, out_dir: Path, source_root: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.argv = child_argv(seed, out_dir, source_root)
        self._log = open(out_dir / "train.log", "w")  # noqa: SIM115 -- closed in wait
        self._p = subprocess.Popen(
            self.argv, cwd=ROOT, stdout=self._log, stderr=self._log, env=RC.child_env()
        )


# --------------------------------------------------------------------------- #
# control 1: the measurement path unchanged
# --------------------------------------------------------------------------- #


def reference_rows(doc: dict) -> dict[str, list[float]]:
    """Every ``A.ckpt3000.*`` STATISTIC row of the fresh-stream ledger."""
    if doc.get("seeds_actually_run") != SEEDS:
        raise ValueError(f"reference seeds {doc.get('seeds_actually_run')} != {SEEDS}")
    return {
        r["key"]: [float(x) for x in r["samples"]]
        for r in doc["rows"]
        if r.get("kind") == "statistic" and r["key"].startswith(CONTROL_PREFIX)
    }


def remeasured_rows(per_control: dict[int, dict[int, dict]]) -> dict[str, list[float]]:
    """The re-measured ckpt-3000 tables as fresh-stream's ledger rows, written by
    fresh-stream's OWN ``arm_rows`` (imported: naming and derived arithmetic are its),
    keeping the ``A.ckpt3000.`` statistic keys."""
    col = ST._Rows()
    with fresh_stream_with(CHECKPOINTS={ARM: (RESUME_STEP,)}):
        FS.arm_rows(col, ARM, per_control)
    return {
        r["key"]: [float(x) for x in r["samples"]]
        for r in col.rows
        if r["kind"] == "statistic" and r["key"].startswith(CONTROL_PREFIX)
    }


def control_1(
    reference: dict[str, list[float]],
    source_root: Path,
    measure_fn: Callable[[Path, int, int, int], dict],
    log: Callable[[str], None] = lambda s: print(s, flush=True),
) -> dict:
    """PREREG control 1: re-measure each seed's start ``ckpt-003000.pt`` (read-only)
    and compare every reference key, per seed, within ``REPRO_TOL`` (scaffold-timing's
    ``reproduction_control``, imported). Never raises: a raise is ``ok`` False."""
    per: dict[int, dict[int, dict]] = {s: {} for s in SEEDS}
    try:
        for s in SEEDS:
            per[s][RESUME_STEP] = measure_fn(
                start_ckpt(s, source_root).parent, s, RESUME_STEP, N_TRAIN
            )
            log(f"  control 1: measured seed {s} ckpt{RESUME_STEP}")
        out = ST.reproduction_control(reference, remeasured_rows(per), REPRO_TOL)
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "per_seed": {}}
    out["error"] = None
    return out


# --------------------------------------------------------------------------- #
# the parent loop
# --------------------------------------------------------------------------- #


def read_resume_checks(root: Path) -> dict[int, dict | None]:
    out: dict[int, dict | None] = {}
    for s in SEEDS:
        f = root / ARM / f"seed{s}" / "resume_check.json"
        out[s] = json.loads(f.read_text()) if f.exists() else None
    return out


def read_stream_windows(root: Path) -> dict[int, dict[int, float]]:
    """Per seed, fresh-stream's ``stream_loss_windows`` over this run's heartbeat."""
    return {
        s: FS.stream_loss_windows(root / ARM / f"seed{s}" / "heartbeat.jsonl")
        for s in SEEDS
    }


def run_all(
    root: Path,
    *,
    control_fn: Callable[[], dict],
    preflight_fn: Callable[[], dict],
    spawn: Callable[[int, Path], Any],
    measure_fn: Callable[[Path, int, int, int], dict],
    deadline: float,
    clock: Callable[[], float] = time.time,
    **kw,
) -> dict:
    """PREREG order: control 1 (measurement path); control 3 (stream) + closure; the
    arm through fresh-stream's ``run_arm``. A failure before the arm trains nothing."""
    res: dict[str, Any] = {
        "control_1": None,
        "preflight": None,
        "arm": None,
        "resume_check": {s: None for s in SEEDS},
        "stream_windows": {s: {} for s in SEEDS},
        "not_run": None,
        "error": None,
        "stopped": None,
    }

    def skip(why: str) -> dict:
        res["stopped"] = res["not_run"] = why
        return res

    if clock() >= deadline:
        return skip("deadline")
    res["control_1"] = c1 = control_fn()
    if not c1.get("ok"):
        if c1.get("error"):
            res["error"] = c1["error"]
        return skip("control_1 (measurement path unchanged) failed or raised")
    res["preflight"] = pf = preflight_fn()
    if not pf.get("ok"):
        return skip(f"preflight failed before any step: {pf.get('error')}")
    if clock() >= deadline:
        return skip("deadline")
    res["arm"] = arm = FS.run_arm(
        ARM,
        root,
        CHECKPOINTS,
        spawn=spawn,
        measure_fn=measure_fn,
        deadline=deadline,
        clock=clock,
        **kw,
    )
    res["error"] = arm["error"]
    res["stopped"] = arm["stopped"]
    res["resume_check"] = read_resume_checks(root)
    res["stream_windows"] = read_stream_windows(root)
    return res


# --------------------------------------------------------------------------- #
# the rule (PREREG "Primary readout")
# --------------------------------------------------------------------------- #


def primary_checkpoint(per: dict[int, dict[int, dict]]) -> int | None:
    """P: the LARGEST listed checkpoint measured on every seed; None if none."""
    done = [c for c in CHECKPOINTS if all(c in per.get(s, {}) for s in SEEDS)]
    return max(done) if done else None


def windows_at_or_before(windows: dict, p: int) -> dict:
    """The complete windows ``[w0, w0 + 100)`` that end at or before checkpoint P:
    ``w0 + 100 <= P``. Heartbeat step t is the t-th optimizer step (0-indexed) and
    ``ckpt-P`` holds the state after steps ``[0, P)``, so a window starting at P or
    later is training the checkpoint P has not seen."""
    return {int(w0): x for w0, x in windows.items() if int(w0) + STREAM_LOSS_WINDOW <= p}


def stirring(stream_windows: dict, p: int) -> dict:
    """Per seed, the first window at or before P with mean < 2.6726; ``any`` over
    seeds is the STIRRING condition."""
    first = {
        s: FS.first_window_below(windows_at_or_before(stream_windows.get(s) or {}, p))
        for s in SEEDS
    }
    return {
        "first_window_below": first,
        "any": any(w is not None for w in first.values()),
    }


def classify(p: int | None, r_at_p: bool | None, stirs: bool) -> str:
    """PREREG classification table, rows 2-5."""
    if p is None or p < MIN_PRIMARY_CKPT:
        return "inconclusive"
    if r_at_p:
        return "ESCAPES"
    if stirs:
        return "STIRRING"
    return "NO_ESCAPE"


def verdict(res: dict) -> dict:
    """PREREG classification table, row for row. Every reason for inconclusive is
    listed, not only the first."""
    per = (res.get("arm") or {}).get("per") or {}
    readout = FS.arm_readout(per, CHECKPOINTS)
    p = primary_checkpoint(per)
    ro = readout.get(p) if p is not None else None
    st = stirring(res.get("stream_windows") or {}, p) if p is not None else None
    base: dict[str, Any] = {
        "P": p,
        "readout": readout,
        "R_at_P": ro["R"] if ro else None,
        "C_at_P": ro["C"] if ro else None,
        "stirring": st,
    }
    why: list[str] = []
    c1 = res.get("control_1")
    if c1 is None:
        why.append("control_1 (measurement path unchanged) did not run")
    elif not c1.get("ok"):
        bad = [f"seed{s}" for s, q in (c1.get("per_seed") or {}).items() if not q["ok"]]
        why.append(
            "control_1 (measurement path unchanged) failed"
            + (f" on {', '.join(bad)}" if bad else "")
            + (f" ({c1['error']})" if c1.get("error") else "")
        )
    pf = res.get("preflight")
    if not pf or not pf.get("ok"):
        why.append(
            "control_3 (stream disjointness / vocabulary closure) failed or did not run"
            + (f" ({pf['error']})" if pf and pf.get("error") else "")
        )
    rc = res.get("resume_check") or {}
    bad2 = [f"seed{s}" for s in SEEDS if not (rc.get(s) or {}).get("ok")]
    if bad2:
        why.append(f"control_2 (resume is exact) failed or missing on {', '.join(bad2)}")
    if res.get("error"):
        why.append(f"a measurement raised ({res['error']})")
    if p is None or p < MIN_PRIMARY_CKPT:
        why.append(
            f"P = {p}: no listed checkpoint >= {MIN_PRIMARY_CKPT} measured on every seed"
        )
    if why:
        return {
            **base,
            "classification": "inconclusive",
            "reported_as": "inconclusive",
            "exit": Exit.DID_NOT_RUN,
            "detail": "; ".join(why),
        }
    cls = classify(p, ro["R"], st["any"])
    reported = f"NO_ESCAPE by {p}" if cls == "NO_ESCAPE" else cls
    return {
        **base,
        "classification": cls,
        "reported_as": reported,
        "exit": Exit.OK,
        "detail": f"P = {p}; R(P) {'holds' if ro['R'] else 'fails'}; a stream-loss "
        f"window below the threshold at or before P on "
        f"{[f'seed{s}' for s, w in st['first_window_below'].items() if w is not None]}"
        f": {reported} -- {READING[cls]}",
    }


# --------------------------------------------------------------------------- #
# the ledger
# --------------------------------------------------------------------------- #


def write_rows(led, res: dict, v: dict) -> None:
    arm = res.get("arm")
    if arm is not None:
        # fresh-stream's own rows (every set / condition / bucket / readout, memory
        # gates, R, M, C, probe R), at this run's listed checkpoints.
        with fresh_stream_with(CHECKPOINTS={ARM: CHECKPOINTS}):
            FS.arm_rows(led, ARM, arm["per"])
    wins = res.get("stream_windows") or {}
    led.note(
        "stream_answer_loss_window_means",
        {
            f"seed{s}": {str(w0): x for w0, x in (wins.get(s) or {}).items()}
            for s in SEEDS
        },
        how=f"heartbeat loss_answer_tokens (steps {RESUME_STEP} onward), mean over each "
        f"complete {STREAM_LOSS_WINDOW}-step window, keyed by its first step",
    )
    led.note(
        "stream_answer_loss_first_window_below",
        {f"seed{s}": FS.first_window_below(wins.get(s) or {}) for s in SEEDS},
        how="first window whose mean < stream_loss_threshold, any step; None if none",
    )
    st = v.get("stirring") or {}
    led.note(
        "stream_answer_loss_first_window_below_at_or_before_P",
        {f"seed{s}": w for s, w in (st.get("first_window_below") or {}).items()} or None,
        how="first window [w0, w0 + 100) with w0 + 100 <= P and mean < "
        "stream_loss_threshold; None if none or P is None",
    )
    led.note("stirring", st.get("any"), how="any seed has such a window; None if no P")
    rc = res.get("resume_check") or {}
    led.note(
        "control_2.ok",
        [(rc.get(s) or {}).get("ok") for s in SEEDS],
        how="per seed: the child's resume_check.json (model and optimizer == "
        "ckpt-003000.pt exactly, before the first step)",
    )
    for part in ("model", "optimizer"):
        led.note(
            f"control_2.{part}_n_tensors",
            [((rc.get(s) or {}).get(part) or {}).get("n_tensors") for s in SEEDS],
            how="per seed: tensors compared with torch.equal",
        )
    led.note(
        "control_2.detail",
        {f"seed{s}": rc.get(s) for s in SEEDS},
        how="the children's resume_check.json, verbatim",
    )
    c1 = res.get("control_1") or {}
    per_seed = c1.get("per_seed") or {}
    for f in ("ok", "n_keys_compared", "max_abs_diff"):
        led.note(
            f"control_1.{f}",
            [per_seed[s][f] if s in per_seed else None for s in SEEDS],
            how=f"per seed, seed order {SEEDS}: every {CONTROL_PREFIX}* statistic key "
            f"of {REF_LEDGER} vs the re-measured start checkpoint, rows by "
            f"{FS.EXPERIMENT}::arm_rows",
        )
    led.note(
        "control_1.failures",
        {f"seed{s}": per_seed[s]["failures"] for s in per_seed},
        how="every (key, reference, measured, abs_diff) outside the tolerance",
    )
    led.note("control_1.missing_keys", c1.get("missing_keys"), how="control_1()")
    led.note(
        "control_1.n_reference_keys",
        c1.get("n_reference_keys"),
        how=f"statistic keys under {CONTROL_PREFIX} in {REF_LEDGER}",
    )
    led.note("control_1.error", c1.get("error"), how="control_1(); None if none")
    led.note("control_1.passed", c1.get("ok"), how="all seeds ok")
    pf = res.get("preflight") or {}
    led.note("preflight.ok", pf.get("ok"), how="stream disjointness + closure")
    led.note("preflight.error", pf.get("error"), how="preflight(); None if none")
    led.note(
        "stream_disjointness",
        pf.get("disjointness"),
        how=f"ids of STREAM.window(t, {BATCH}), t in [{RESUME_STEP}, {END_STEP}), vs "
        "PROBE, the vocabulary documents and HELDOUT",
    )
    led.note(
        "vocabulary_closure",
        {f"seed{s}": c for s, c in (pf.get("closure") or {}).items()},
        how=f"{FS.EXPERIMENT}::vocabulary_closure over the ids of steps "
        f"[{RESUME_STEP}, {END_STEP}) and both measurement sets",
    )
    led.note(
        "R_by_checkpoint",
        {str(c): r["R"] for c, r in v["readout"].items()},
        how="R(c) per checkpoint measured on every seed",
    )
    led.note(
        "C_by_checkpoint",
        {str(c): r["C"] for c, r in v["readout"].items()},
        how="C(c) per checkpoint measured on every seed (reported, not classified)",
    )
    led.note("P", v["P"], how="largest listed checkpoint measured on every seed")
    led.note("R_at_P", v["R_at_P"], how="R(P); None if no P")
    led.note("C_at_P", v["C_at_P"], how="C(P), reported only; None if no P")
    led.note("stopped_because", res.get("stopped"), how="run_all(); None if none")
    led.note("not_run", res.get("not_run"), how="run_all(); None if the arm ran")
    led.note("measurement_error", res.get("error"), how="run_all(); None if none")
    led.note("elapsed_s", arm and arm["elapsed_s"], how="parent wall clock")
    led.note(
        "checkpoints_measured",
        arm and {f"seed{s}": arm["checkpoints_measured"][s] for s in SEEDS},
        how="run_arm()",
    )
    led.note("classification", v["classification"], how="PREREG classification table")
    led.note("classification_reported_as", v["reported_as"], how="PREREG decision rule")
    led.note("classification_detail", v["detail"], how="verdict()")
    led.note(
        "reading",
        READING.get(v["classification"]),
        how="PREREG 'Primary readout' table, the row of the classification",
    )
    led.note("delta", DELTA, how="PREREG thresholds (imported)")
    led.note("c_threshold", C_THRESHOLD, how="fresh-stream's C threshold, imported")
    led.note(
        "stream_loss_threshold", STREAM_LOSS_THRESHOLD, how="fresh-stream's, imported"
    )
    led.note("min_primary_ckpt", MIN_PRIMARY_CKPT, how="PREREG thresholds")
    led.note("reproduction_tolerance", REPRO_TOL, how="PREREG control 1")
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
        "child": f"{FS.EXPERIMENT}::single('B') with start_ckpt = fresh-stream "
        f"A/seed<s>/ckpt-{RESUME_STEP:06d}.pt, iters = {END_STEP}",
        "seeds": SEEDS,
        "resume_step": RESUME_STEP,
        "end_step": END_STEP,
        "ckpt_every": CKPT_EVERY,
        "checkpoints_measured": list(CHECKPOINTS),
        "min_primary_ckpt": MIN_PRIMARY_CKPT,
        "stream": dataclasses.asdict(STREAM),
        "stream_documents": [
            STREAM.window(RESUME_STEP, BATCH).start,
            STREAM.window(END_STEP - 1, BATCH).stop,
        ],
        "heldout_documents": list(HELDOUT),
        "probe_documents": list(PROBE),
        "vocab_documents": list(VOCAB_DOCUMENTS),
        "delta": DELTA,
        "c_threshold": C_THRESHOLD,
        "stream_loss_threshold": STREAM_LOSS_THRESHOLD,
        "stream_loss_window": STREAM_LOSS_WINDOW,
        "reproduction_tolerance": REPRO_TOL,
        "control_1": f"measure_checkpoint of A/seed<s>/ckpt-{RESUME_STEP:06d}.pt vs "
        f"{REF_LEDGER} {CONTROL_PREFIX}* statistic keys",
        "deadline": DEADLINE,
        "device": DEVICE,
        "threads_per_seed": THREADS_PER_SEED,
        "seeds_trained_in_parallel": True,
        "parent_torch_num_threads": parent_threads,
        "source_root": str(source_root),
        "start_checkpoint_sha256": start_sha,
        "reference_ledger": REF_LEDGER,
        "reference_ledger_sha256": ref_sha,
        "torch": torch.__version__,
        "python": sys.version.split()[0],
    }


def execute(led, root: Path, *, results_path: Path | None, **kw) -> Exit:
    res = run_all(root, **kw)
    res["root"] = str(root)
    # Written first, so a raise below still leaves every measurement on disk.
    root.mkdir(parents=True, exist_ok=True)
    (root / "raw.json").write_text(json.dumps(res, indent=2, default=str) + "\n")
    arm = res["arm"]
    if arm is not None:
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
                    rel, exit_code=arm["exit_codes"][s], note=f"seed {s} training"
                )
            info = root / ARM / f"seed{s}" / "result.json"
            if info.exists():
                r = json.loads(info.read_text())
                led.note(
                    f"seed{s}.training",
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
    per = (arm or {}).get("per") or {}
    measured = sorted(s for s in SEEDS if per.get(s))
    led.run_meta(
        device=DEVICE,
        seeds_actually_run=measured or None,
        steps_requested=END_STEP,
        steps_done=max((max(per[s]) for s in SEEDS if per.get(s)), default=0),
    )
    crashed = res["error"] is not None or bool(
        arm
        and (
            arm["failed_seeds"]
            or (
                arm["stopped"] is None and any(c != 0 for c in arm["exit_codes"].values())
            )
        )
    )
    complete = arm is not None and arm["stopped"] is None
    led.status("crashed" if crashed else ("ok" if complete else "partial"))
    write_rows(led, res, v)
    path = led.write()
    if results_path is not None:
        results_path.write_text(
            render_results(json.loads(path.read_text()), read_brief_errors())
        )
    print(f"ledger: {path}  classification={v['reported_as']}  exit={int(v['exit'])}")
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
        "# Fresh escape -- RESULTS",
        "",
        f"<!-- GENERATED by `{EXPERIMENT}` from runs/{rid}/ledger.json. Do not edit: "
        "regenerate with `--render-results`. -->",
        "",
        f"Command: `experiments/fresh-escape/run.sh` (the parent is `uv run python "
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
        f"## Classification of run `{rid}`: `classification_reported_as` "
        f"**{val(rows, 'classification_reported_as')}**",
        "",
        f"`classification` **{val(rows, 'classification')}**. Why, for run `{rid}` "
        f"(`classification_detail`): **{val(rows, 'classification_detail')}**",
        "",
        f"`P` {fmt(val(rows, 'P'))} · `R_at_P` {fmt(val(rows, 'R_at_P'))} · `C_at_P` "
        f"{fmt(val(rows, 'C_at_P'))} · `stirring` {fmt(val(rows, 'stirring'))} · "
        f"`measurement_error` {fmt(val(rows, 'measurement_error'))} · "
        f"`stopped_because` {fmt(val(rows, 'stopped_because'))} · `not_run` "
        f"{fmt(val(rows, 'not_run'))}.",
        "",
        "Whether the cognitive claim moved: **it cannot move here.** FIFO only, no "
        "retention loss; nothing about RSR, eviction, ψ̂ or Kintsch & van Dijk "
        "(PREREG, *What this does not establish*). No scaling claim.",
        "",
        f"The pre-registered expectation of run `{rid}` is its `expected` row, frozen "
        "in the manifest before any child ran; it is not restated here.",
        "",
        "## Per checkpoint",
        "",
        f"Per-seed samples in seed order; bucket `{BUCKET}`.",
        "",
    ]
    for c in CHECKPOINTS:
        e = f"{ARM}.ckpt{c}"
        if val(rows, f"{e}.R_quantity") is None:
            out += [f"- `{e}`: not measured on any seed.", ""]
            continue
        out += [
            f"- `{e}.R_quantity` {fmt(val(rows, f'{e}.R_quantity'))} · `{e}.R_holds` "
            f"{fmt(val(rows, f'{e}.R_holds'))} · `{e}.C_quantity` "
            f"{fmt(val(rows, f'{e}.C_quantity'))} · `{e}.C_holds` "
            f"{fmt(val(rows, f'{e}.C_holds'))} · `{e}.probe_R_quantity` "
            f"{fmt(val(rows, f'{e}.probe_R_quantity'))} · `{e}.M_quantity` "
            f"{fmt(val(rows, f'{e}.M_quantity'))}.",
            "",
        ]
    first = val(rows, "stream_answer_loss_first_window_below_at_or_before_P") or {}
    out += [
        "## Stream answer loss",
        "",
        f"Threshold `stream_loss_threshold` {fmt(val(rows, 'stream_loss_threshold'))}; "
        "the first complete window at or before P below it: "
        + (
            " · ".join(
                f"`stream_answer_loss_first_window_below_at_or_before_P.{s}` {fmt(w)}"
                for s, w in sorted(first.items())
            )
            or "—"
        )
        + ". Every window mean is in `stream_answer_loss_window_means`.",
        "",
        "## Controls",
        "",
        f"`control_1` (measurement path unchanged, `{CONTROL_PREFIX}*`): "
        f"`control_1.passed` {fmt(val(rows, 'control_1.passed'))} · "
        f"`control_1.n_reference_keys` {fmt(val(rows, 'control_1.n_reference_keys'))} "
        f"· `control_1.n_keys_compared` {fmt(val(rows, 'control_1.n_keys_compared'))} "
        f"· `control_1.max_abs_diff` {val(rows, 'control_1.max_abs_diff')!r} · "
        f"`reproduction_tolerance` {val(rows, 'reproduction_tolerance')!r} · "
        f"`control_1.error` {fmt(val(rows, 'control_1.error'))}.",
        "",
    ]
    fails = val(rows, "control_1.failures") or {}
    if any(fails.values()):
        out += [
            "Every key outside the tolerance (`control_1.failures`):",
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
        f"`control_2` (resume is exact): `control_2.ok` "
        f"{fmt(val(rows, 'control_2.ok'))}.",
        "",
        f"`control_3`: `preflight.ok` {fmt(val(rows, 'preflight.ok'))} · "
        f"`preflight.error` {fmt(val(rows, 'preflight.error'))}; the start-checkpoint "
        "sha256 values are the `start_checkpoint_sha256` row, the stream ids and the "
        "closure the `stream_disjointness` and `vocabulary_closure` rows.",
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
    """The plan, and the start checkpoints verified (sha256 + stored step), read
    only. Nothing is trained or measured and nothing is written."""
    now = time.time() if now is None else now
    d = stream_disjointness()
    lines = [
        "fresh-escape dry run (NOTHING TRAINED OR MEASURED).",
        f"control 1 first: {CSC.EXPERIMENT}::measure_checkpoint of "
        f"A/seed<s>/ckpt-{RESUME_STEP:06d}.pt x seeds {SEEDS} vs {REF_LEDGER} "
        f"{CONTROL_PREFIX}* within {REPRO_TOL}; fail -> exit 3, nothing trains",
        f"control 3: stream ids {d['first_id']}..{d['last_id']} ({d['n_ids']}, repeats "
        f"{d['repeats']}, disjoint {d['ok']}); vocabulary closure over those ids per "
        "seed + both measurement sets",
        f"arm {ARM}: resume ckpt{RESUME_STEP} -> {END_STEP}, ckpt_every {CKPT_EVERY}; "
        f"measure {list(CHECKPOINTS)}; child argv "
        f"{child_argv(0, ROOT / 'runs' / RUN_ID / ARM / 'seed0', source_root)[2:]}",
    ]
    for s in SEEDS:
        f = start_ckpt(s, source_root)
        lines.append(f"  start seed {s}: {f}  exists={f.exists()}")
    try:
        sha = verify_start_checkpoints(source_root)
        lines.append(
            f"start checkpoints: every sha256 == the PREREG front matter, every stored "
            f"step == {RESUME_STEP}: {sha}"
        )
    except (OSError, ValueError, StartCheckpointRefused) as e:
        lines.append(f"start checkpoints REFUSED (the run would exit 3): {e}")
    try:
        ref = reference_rows(json.loads((ROOT / REF_LEDGER).read_text()))
        lines.append(f"control 1 reference: {len(ref)} {CONTROL_PREFIX}* statistic keys")
    except (OSError, ValueError) as e:
        lines.append(f"control 1 reference UNREADABLE (the run would exit 3): {e}")
    lines += [
        f"deadline {DEADLINE} ({(deadline_epoch() - now) / 3600:.2f} h from now)",
        f"PREDICTED ~{PRED_S_PER_STEP * (END_STEP - RESUME_STEP) / 3600:.2f} h of "
        "training alone (prediction, not premise: PREREG, linear; longer under "
        "contention)",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> Exit:
    ap = ArgumentParser(prog="fresh-escape")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; run nothing")
    ap.add_argument(
        "--render-results", action="store_true", help="RESULTS.md from the ledger"
    )
    ap.add_argument("--run-id", default=RUN_ID)
    ap.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--seed", type=int, choices=SEEDS, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args(argv)

    if a.single:
        if a.out_dir is None or a.seed is None:
            refuse(Exit.DID_NOT_RUN, "--single needs --out-dir and --seed")
        a.out_dir.mkdir(parents=True, exist_ok=True)
        r = single(a.seed, a.out_dir, source_root=a.source_root)
        (a.out_dir / "result.json").write_text(
            json.dumps(r, indent=2, default=str) + "\n"
        )
        return Exit.OK
    if a.dry_run:
        print(dry_run(a.source_root))
        return Exit.OK
    root = ROOT / "runs" / a.run_id
    results = ROOT / "experiments" / "fresh-escape" / "RESULTS.md"
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
    try:
        start_sha = verify_start_checkpoints(a.source_root)
    except StartCheckpointRefused as e:
        refuse(Exit.DID_NOT_RUN, f"start checkpoints refused: {e}")
    # Measured at the thread count the reference was measured at (fresh-stream's
    # parent, its manifest's parent_torch_num_threads).
    ref_manifest = json.loads((ROOT / REF_MANIFEST).read_text())
    torch.set_num_threads(int(ref_manifest["parent_torch_num_threads"]))

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
        control_fn=lambda: control_1(reference, a.source_root, measure_checkpoint),
        preflight_fn=preflight,
        spawn=lambda s, d: Child(s, d, a.source_root),
        measure_fn=measure_checkpoint,
        deadline=deadline_epoch(),
    )


if __name__ == "__main__":
    run_main(main)
