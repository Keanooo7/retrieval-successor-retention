"""Fresh stream -- can the memory learn to retrieve without repeated documents?

Pre-registration: `experiments/fresh-stream/PREREG.md`, committed alone at 83128ee,
ahead of this file. **One change to the training code:** `train(stream=...)`
(`rsr.train.loop.DocumentStream`) replaces the with-replacement sampler by the
deterministic stream: step ``t`` trains on documents ``[4160 + 16 t, 4160 + 16 t +
16)`` of the seed's generator, generated per id, encoded with the vocabulary of
documents ``[0, 64)``. ``stream=None`` is today's path, byte for byte.

Order (PREREG "Reproduction controls (checked first; each overrides everything)"):

1. **Before any step:** control 3 (the stream is disjoint from ``[0, 64)`` and
   ``[4096, 4160)`` and never repeats an id) and the vocabulary closure (every word
   of every stream document ``[4160, 4160 + 16 * 3000)`` and of both measurement
   sets is in the seed's ``[0, 64)`` map). Either failing: exit 3, nothing trains.
2. **Control 1, default path unchanged.** The corpus-size N = 64 child, imported
   (``--single --n-documents 64``, which calls ``train()`` WITHOUT the stream),
   trained to step 100 on all three seeds; ``ckpt-000100.pt`` measured with the
   imported ``measure_checkpoint`` and turned into rows by scaffold-timing's own
   ``arm_rows``; every ``n64.ckpt100.*`` statistic key of
   `runs/scaffold-timing/ledger.json` must match per seed within ``REPRO_TOL``.
   Fail, missing key or a raise: exit 3, no arm runs.
3. **Arm A** (fresh init, steps 0 -> 3000 on the stream), then **arm B** (resume the
   corpus-size N = 64 ``ckpt-001000.pt``, steps 1000 -> 3000 on the stream). Each
   arm: three seeds as parallel children at 5 threads (the corpus-size setting);
   the parent measures each listed checkpoint as it appears.
4. **Control 2, resume is exact**, inside each arm-B child: the model and optimizer
   state after the load and before the first step must equal the tensors of
   ``ckpt-001000.pt`` (read again from disk) exactly. Recorded per seed
   (``resume_check.json``, then the ledger); a mismatch stops that child.
5. The rule: R and C per checkpoint, R3000 per arm, DATA_SUFFICES / SCAFFOLD / TRAP
   / NEITHER / inconclusive, the ``usable`` line, and every secondary readout.

The parent enforces the absolute ``DEADLINE``: at it the children are stopped and
whatever listed checkpoints exist are measured.

Usage::

    uv run --extra dev python experiments/fresh-stream/run.py --dry-run
    experiments/fresh-stream/run.sh        # the real run; exit code -> run.rc
    uv run --extra dev python experiments/fresh-stream/run.py --render-results

Exit codes (`rsr.exit_codes`): 0 a classification was reached · 3 did not run or
inconclusive (a control failed or was incomplete, a measurement raised, ckpt 3000
missing for any seed of either arm, a refused precondition).
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as _dt
import functools
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
from rsr.train.loop import DocumentStream  # noqa: E402

EXPERIMENT = "experiments/fresh-stream/run.py"
RUN_ID = "fresh-stream"
PREREG = "experiments/fresh-stream/PREREG.md"
PREREG_COMMIT = "83128ee"
BRIEF_ERRORS = "experiments/fresh-stream/BRIEF-ERRORS.md"


def _load(name: str, rel: str):
    """Import an experiment script by path. Imported, never copied."""
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ST = _load("_fresh_stream_st", "experiments/scaffold-timing/run.py")
#: The corpus-size module scaffold-timing measured with -- ONE instance, so the
#: measurement below is the very object scaffold-timing called.
CSC = ST.CSC
RC = CSC.RC
S003 = CSC.S003

#: THE measurement (PREREG "Instrument"): the corpus-size curve's function object,
#: not a copy; `tests/test_fresh_stream.py` checks the identity.
measure_checkpoint = CSC.measure_checkpoint

# --------------------------------------------------------------------------- #
# 🔒 PREREG.md, transcribed. Changing any is changing the PREREG.
# --------------------------------------------------------------------------- #

SEEDS = [0, 1, 2]
ARMS = ("A", "B")
CHECKPOINTS = {"A": (300, 1000, 1500, 2000, 2500, 3000), "B": (1500, 2000, 2500, 3000)}
FINAL = 3000
ITERS = 3000
RESUME_STEP = 1000
CKPT_EVERY = 100
BATCH = S003.CONFIG["batch"]
STREAM = DocumentStream(offset=4160, stride=16, vocab_documents=64)
HELDOUT = CSC.HELDOUT
PROBE = CSC.PROBE
VOCAB_DOCUMENTS = (0, STREAM.vocab_documents)
#: "DELTA: 0.03 accuracy per seed -- scaffold-timing's, unchanged": imported.
DELTA = ST.DELTA
BRIER_MARGIN = 0.05
BRIER16_UNIFORM = CSC.BRIER16_UNIFORM
C_THRESHOLD = BRIER16_UNIFORM - BRIER_MARGIN
REPRO_TOL = 1e-6
DEADLINE = "2026-09-26T06:00:00-07:00"  # 06:00 America/Los_Angeles, PDT
CONTROL_ARM = CSC.CONTROL_ARM  # N = 64, the default path
CONTROL_CHECKPOINT = 100
CONTROL_ITERS = CONTROL_CHECKPOINT
CONTROL_PREFIX = f"n{CONTROL_ARM}.ckpt{CONTROL_CHECKPOINT}."
BUCKET = "gap_2_to_M"
DEVICE = CSC.DEVICE
THREADS_PER_SEED = RC.THREADS_PER_SEED
CHANCE = S003.CHANCE
#: PREREG secondary: "the first 100-step window whose mean is below 2.7726 - 0.10".
STREAM_LOSS_WINDOW = 100
STREAM_LOSS_MARGIN = 0.10
STREAM_LOSS_THRESHOLD = CHANCE - STREAM_LOSS_MARGIN

#: Arm B's starting checkpoints: read-only, never written (PREREG "arms").
SOURCE_ROOT = ST.SOURCE_ROOT
REF_LEDGER = "runs/scaffold-timing/ledger.json"
REF_MANIFEST = "runs/scaffold-timing/manifest.json"

QUESTION = (
    "With no training document ever repeated, does TG's working memory come to carry "
    "answer information on unseen documents -- from scratch (arm A), and when started "
    "from a model that already acquired it by memorising 64 documents (arm B)?"
)
DECISION_RULE = (
    "A failed or incomplete reproduction control, a measurement that raised, or ckpt "
    "3000 missing for any seed of either arm makes the result inconclusive (exit 3). "
    "Otherwise read R3000 per arm (R at ckpt 3000 on every seed): A and B -> "
    "DATA_SUFFICES; B only -> SCAFFOLD; A only -> TRAP; neither -> NEITHER."
)
#: PREREG "Author's expectation", verbatim (whitespace joined); frozen into the
#: manifest before any child starts.
EXPECTED = (
    "- Arm A: the stream answer loss leaves the 2.78 plateau on at least one seed by "
    "step 3000 — N = 512 seed 2 began moving near step 900 with repetition — but "
    "R3000(A) holds on every seed somewhat less often than not (≈ 40 %). - Arm B: R "
    "survives (≈ 70 %) and does not grow much; held-out Brier improves from ~1.4 "
    "toward but not below 0.8875 as the memorised-document overconfidence washes out "
    "(C3000(B) ≈ 30 %). Probe accuracy falls from 1.0. - Most likely classification: "
    "**SCAFFOLD**. Next most likely: DATA_SUFFICES."
)

POLL_S = RC.POLL_S
STOP_GRACE_S = RC.STOP_GRACE_S
#: Prediction for --dry-run only; no rule reads it (PREREG "Time": corpus-size
#: heartbeats, 3570-3579 s per 1000 steps at 3 seeds x 5 threads).
PRED_S_PER_STEP = 3.579


class StepMismatch(RuntimeError):
    """A checkpoint's stored step is not the label it is measured under."""


class ResumeMismatch(RuntimeError):
    """Control 2 failed: the resumed state is not the checkpoint's."""


def deadline_epoch() -> float:
    return _dt.datetime.fromisoformat(DEADLINE).timestamp()


def sha256_file(path: Path) -> str:
    return ST.sha256_file(path)


def start_ckpt(seed: int, source_root: Path = SOURCE_ROOT) -> Path:
    """Arm B's start: the corpus-size N = 64 ``ckpt-001000.pt`` of ``seed``."""
    return RC.ckpt_path(ST.seed_dir(source_root, CONTROL_ARM, seed), RESUME_STEP)


# --------------------------------------------------------------------------- #
# the stream: disjointness (control 3) and vocabulary closure
# --------------------------------------------------------------------------- #


def stream_windows(iters: int = ITERS) -> list[range]:
    """The document ids of every step ``t`` in ``[0, iters)``."""
    return [STREAM.window(t, BATCH) for t in range(iters)]


def stream_disjointness(iters: int = ITERS) -> dict:
    """PREREG control 3: no stream id lies in ``PROBE`` (= the vocabulary documents)
    or ``HELDOUT``, and no id repeats. Also records that arms A and B see the same
    ids at every step in ``[RESUME_STEP, iters)`` -- both read ``STREAM.window``."""
    ids = [i for w in stream_windows(iters) for i in w]
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
        "first_window_of_B": list(STREAM.window(RESUME_STEP, BATCH)),
    }


def vocabulary_closure(seed: int, iters: int = ITERS, block: int = 4096) -> dict:
    """PREREG "Vocabulary": every word of every stream document the run will use and
    of both measurement sets is in the seed's ``[0, 64)`` map. Documents are made in
    blocks, never all at once. Raises ``ValueError`` (``check_vocabulary_closure``,
    the one definition ``train()`` also uses) on the first word outside the map."""
    from rsr.data.synthetic import SyntheticConfig, _generate_document
    from rsr.train.loop import build_vocab, check_vocabulary_closure

    cfg = SyntheticConfig(
        sentences_per_document=S003.CONFIG["steps_per_stream"], seed=seed
    )
    vmap = build_vocab([_generate_document(i, cfg) for i in range(*VOCAB_DOCUMENTS)])
    sets, set_vmap, _v = CSC.doc_sets(seed, CONTROL_ARM)
    if set_vmap != vmap:
        raise ValueError(f"seed {seed}: the measurement vocabulary is not [0, 64)'s")
    for docs in sets.values():
        check_vocabulary_closure(docs, vmap)
    ids = sorted({i for w in stream_windows(iters) for i in w})
    for k in range(0, len(ids), block):
        check_vocabulary_closure(
            [_generate_document(i, cfg) for i in ids[k : k + block]], vmap
        )
    return {
        "ok": True,
        "n_stream_documents": len(ids),
        "n_measurement_documents": sum(len(d) for d in sets.values()),
        "vocab_words": len(vmap),
    }


def preflight(iters: int = ITERS) -> dict:
    """Control 3 and the closure, before any step. Never raises: a failure is a
    record with ``ok`` False and the reason."""
    out: dict[str, Any] = {"ok": False, "error": None, "closure": {}}
    try:
        out["disjointness"] = stream_disjointness(iters)
        if not out["disjointness"]["ok"]:
            out["error"] = "the stream is not disjoint or repeats a document"
            return out
        for s in SEEDS:
            out["closure"][s] = vocabulary_closure(s, iters)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    out["ok"] = True
    return out


# --------------------------------------------------------------------------- #
# one training run, in its own child process
# --------------------------------------------------------------------------- #


def _same(a: Any, b: Any, path: str, acc: dict) -> None:
    """Exact recursive equality; tensors by dtype, shape and ``torch.equal``."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        acc["n_tensors"] += 1
        ok = (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and torch.equal(a, b)
        )
    elif isinstance(a, dict) and isinstance(b, dict):
        ok = sorted(map(str, a)) == sorted(map(str, b))
        if ok:
            for k in a:
                _same(a[k], b[k], f"{path}.{k}", acc)
    elif isinstance(a, list | tuple) and isinstance(b, list | tuple):
        ok = len(a) == len(b)
        if ok:
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                _same(x, y, f"{path}[{i}]", acc)
    else:
        ok = type(a) is type(b) and a == b
    if not ok:
        acc["mismatches"].append(path)


def compare_to_checkpoint(ckpt: Path, model, optimizer) -> dict:
    """PREREG control 2: the live model and optimizer state against the tensors of
    ``ckpt``, read again from disk. Exact: no tolerance."""
    ref = torch.load(ckpt, map_location="cpu", weights_only=False)
    out = {}
    for name, live, saved in (
        ("model", model.state_dict(), ref["model"]),
        ("optimizer", optimizer.state_dict(), ref["optimizer"]),
    ):
        acc: dict[str, Any] = {"n_tensors": 0, "mismatches": []}
        _same(live, saved, name, acc)
        out[name] = {
            "equal": not acc["mismatches"],
            "n_tensors": acc["n_tensors"],
            "mismatches": acc["mismatches"][:20],
        }
    return {
        "checkpoint": str(ckpt),
        "stored_step": int(ref["step"]),
        "ok": out["model"]["equal"] and out["optimizer"]["equal"],
        **out,
    }


class ResumeCheck:
    """Control 2 inside the arm-B child, around the unmodified ``train()``.

    ``rsr.train.checkpoint.load`` is wrapped to capture the model and optimizer it
    restores into; ``rsr.train.loop.run_policy_loop`` is wrapped so that at its
    FIRST call -- after the load, before the first forward and optimizer step --
    the state is compared with the checkpoint on disk. The result is written to
    ``out`` at once; a mismatch raises ``ResumeMismatch`` and the child stops.
    ``train()`` itself gains no hook (PREREG: the stream is the only change)."""

    def __init__(self, ckpt: Path, out: Path) -> None:
        self.ckpt, self.out = Path(ckpt), Path(out)
        self.loaded: tuple | None = None
        self.result: dict | None = None

    @contextlib.contextmanager
    def installed(self):
        import rsr.train.checkpoint as ck
        import rsr.train.loop as loop

        real_load, real_loop = ck.load, loop.run_policy_loop

        def load(path, **kw):
            payload = real_load(path, **kw)
            if Path(path) == self.ckpt:
                self.loaded = (kw.get("model"), kw.get("optimizer"))
            return payload

        def first(model, *a, **k):
            if self.result is None:
                self.check()
            return real_loop(model, *a, **k)

        ck.load, loop.run_policy_loop = load, first
        try:
            yield self
        finally:
            ck.load, loop.run_policy_loop = real_load, real_loop

    def check(self) -> dict:
        if self.loaded is None or None in self.loaded:
            self.result = {"ok": False, "error": "train() did not load the checkpoint"}
        else:
            self.result = compare_to_checkpoint(self.ckpt, *self.loaded)
        self.out.write_text(json.dumps(self.result, indent=2) + "\n")
        if not self.result["ok"]:
            raise ResumeMismatch(f"{self.ckpt}: {self.result}")
        return self.result


def single(
    arm: str,
    seed: int,
    out_dir: Path,
    *,
    iters: int = ITERS,
    source_root: Path = SOURCE_ROOT,
) -> dict:
    """The corpus-size N = 64 arm's own ``single()``, imported, with the stream added
    to its ``train()`` call (and, for arm B, the resume). Nothing else is retyped:
    every other argument is the one the N = 64 arm passed, so arm A's init is that
    arm's init for the seed."""
    import rsr.train.loop as loop

    if arm not in ARMS:
        raise ValueError(f"arm {arm!r} is not one of {ARMS}")
    extra: dict[str, Any] = {"stream": STREAM}
    check = None
    if arm == "B":
        extra["resume"] = str(start_ckpt(seed, source_root))
        check = ResumeCheck(start_ckpt(seed, source_root), out_dir / "resume_check.json")
    real = loop.train
    loop.train = functools.partial(real, **extra)
    try:
        with check.installed() if check else contextlib.nullcontext():
            r = CSC.single(seed, iters, CONTROL_ARM, out_dir)
    finally:
        loop.train = real
    r["arm"] = arm
    r["stream"] = dataclasses.asdict(STREAM)
    r["resume_check"] = check.result if check else None
    return r


def child_argv(arm: str, seed: int, out_dir: Path, source_root: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single",
        "--arm",
        arm,
        "--seed",
        str(seed),
        "--out-dir",
        str(out_dir),
        "--source-root",
        str(source_root),
    ]


class Child(RC.Child):
    """The retrieval curve's child (5 OMP/MKL threads), launched on THIS script."""

    def __init__(self, arm: str, seed: int, out_dir: Path, source_root: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.argv = child_argv(arm, seed, out_dir, source_root)
        self._log = open(out_dir / "train.log", "w")  # noqa: SIM115 -- closed in wait
        self._p = subprocess.Popen(
            self.argv, cwd=ROOT, stdout=self._log, stderr=self._log, env=RC.child_env()
        )


def control_child(seed: int, out_dir: Path):
    """Control 1: the corpus-size curve's N = 64 child, imported -- ``train()``
    WITHOUT the stream -- to step ``CONTROL_ITERS``."""
    return CSC.Child(seed, CONTROL_ARM, out_dir, iters=CONTROL_ITERS)


# --------------------------------------------------------------------------- #
# the parent loop
# --------------------------------------------------------------------------- #


def run_arm(
    arm: str,
    root: Path,
    checkpoints: tuple[int, ...],
    *,
    spawn: Callable[[int, Path], Any],
    measure_fn: Callable[[Path, int, int, int], dict],
    deadline: float,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    poll_s: float = POLL_S,
    log: Callable[[str], None] = lambda s: print(s, flush=True),
) -> dict:
    """One arm: three seeds in parallel children; each listed checkpoint measured in
    THIS process as it appears, lowest label first across seeds. The corpus-size
    ``run_arm`` with the arm's own checkpoints and no in-arm control."""
    t0 = clock()
    dirs = {s: root / arm / f"seed{s}" for s in SEEDS}
    procs = {s: spawn(s, dirs[s]) for s in SEEDS}
    pending = {s: list(checkpoints) for s in SEEDS}
    per: dict[int, dict[int, dict]] = {s: {} for s in SEEDS}

    def sweep() -> None:
        while True:
            ready = [
                (pending[s][0], s)
                for s in SEEDS
                if pending[s] and RC.ckpt_path(dirs[s], pending[s][0]).exists()
            ]
            if not ready:
                return
            label, s = min(ready)
            pending[s].pop(0)
            per[s][label] = measure_fn(dirs[s], s, label, CONTROL_ARM)
            log(f"  arm {arm} seed {s}: measured ckpt{label}")

    def stop_all() -> None:
        live = [p for p in procs.values() if p.poll() is None]
        for p in live:
            p.terminate()
        for p in live:
            try:
                p.wait(timeout=STOP_GRACE_S)
            except subprocess.TimeoutExpired:
                p.kill()

    stopped: str | None = None
    failed: list[int] = []
    error: str | None = None
    try:
        while True:
            sweep()
            if not any(pending.values()):
                break
            if clock() >= deadline:
                stop_all()
                sweep()
                stopped = "deadline"
                break
            dead = [s for s in SEEDS if pending[s] and procs[s].poll() is not None]
            if dead:
                sweep()
                failed = [s for s in SEEDS if pending[s] and procs[s].poll() is not None]
                if failed:
                    stopped = "training child exited early"
                    break
                continue
            sleep(poll_s)
    except (Exception, SystemExit) as e:
        error = f"{type(e).__name__}: {e}"
        stopped = "measurement raised"
        log(f"  arm {arm} measurement raised: {error}")
    finally:
        if stopped is None and error is None:
            # Every checkpoint measured: let the children write their footer and
            # result.json rather than SIGTERM a clean exit into a "crash".
            for p in procs.values():
                with contextlib.suppress(subprocess.TimeoutExpired):
                    p.wait(timeout=10 * STOP_GRACE_S)
        stop_all()
    exit_codes = {s: p.wait() for s, p in procs.items()}
    return {
        "arm": arm,
        "per": per,
        "stopped": stopped,
        "error": error,
        "failed_seeds": failed,
        "exit_codes": exit_codes,
        "argv": {s: getattr(p, "argv", None) for s, p in procs.items()},
        "checkpoints_measured": {s: sorted(per[s]) for s in SEEDS},
        "elapsed_s": clock() - t0,
    }


def remeasured_rows(per_control: dict[int, dict[int, dict]]) -> dict[str, list[float]]:
    """The control's ckpt-100 tables as scaffold-timing's ledger rows, written by
    scaffold-timing's OWN ``arm_rows`` (imported: naming and derived arithmetic are
    its), keeping the ``n64.ckpt100.`` statistic keys."""
    col = ST._Rows()
    ST.arm_rows(col, CONTROL_ARM, per_control)
    return {
        r["key"]: [float(x) for x in r["samples"]]
        for r in col.rows
        if r["kind"] == "statistic" and r["key"].startswith(CONTROL_PREFIX)
    }


def reference_rows(doc: dict) -> dict[str, list[float]]:
    """Every ``n64.ckpt100.*`` STATISTIC row of the scaffold-timing ledger."""
    if doc.get("seeds_actually_run") != SEEDS:
        raise ValueError(f"reference seeds {doc.get('seeds_actually_run')} != {SEEDS}")
    return {
        r["key"]: [float(x) for x in r["samples"]]
        for r in doc["rows"]
        if r.get("kind") == "statistic" and r["key"].startswith(CONTROL_PREFIX)
    }


def control_1(reference: dict[str, list[float]], per_control: dict) -> dict:
    """Scaffold-timing's ``reproduction_control``, imported, at THIS PREREG's
    tolerance: every reference key, per seed, within ``REPRO_TOL``; a missing key
    or no key at all fails."""
    return ST.reproduction_control(reference, remeasured_rows(per_control), REPRO_TOL)


def read_resume_checks(root: Path) -> dict[int, dict | None]:
    out: dict[int, dict | None] = {}
    for s in SEEDS:
        f = root / "B" / f"seed{s}" / "resume_check.json"
        out[s] = json.loads(f.read_text()) if f.exists() else None
    return out


def run_all(
    root: Path,
    *,
    preflight_fn: Callable[[], dict],
    spawn_control: Callable[[int, Path], Any],
    spawn: Callable[[str, int, Path], Any],
    measure_fn: Callable[[Path, int, int, int], dict],
    reference: dict[str, list[float]],
    deadline: float,
    clock: Callable[[], float] = time.time,
    **kw,
) -> dict:
    """PREREG order: control 3 + closure; control 1; arm A; arm B. A failure of
    anything before the arms stops everything; a raised measurement stops every later
    arm; an arm whose start comes at or after the deadline is not run."""
    res: dict[str, Any] = {
        "preflight": None,
        "control": None,
        "control_1": None,
        "arms": {},
        "resume_check": {s: None for s in SEEDS},
        "not_run": {},
        "error": None,
        "stopped": None,
    }
    common = {"measure_fn": measure_fn, "deadline": deadline, "clock": clock, **kw}

    def skip_all(why: str) -> dict:
        res["stopped"] = why
        for a in ARMS:
            res["not_run"].setdefault(a, why)
        return res

    res["preflight"] = pf = preflight_fn()
    if not pf["ok"]:
        return skip_all(f"preflight failed before any step: {pf['error']}")
    if clock() >= deadline:
        return skip_all("deadline")
    ctl = run_arm("control", root, (CONTROL_CHECKPOINT,), spawn=spawn_control, **common)
    res["control"] = ctl
    if ctl["error"]:
        res["error"] = ctl["error"]
        return skip_all("a control-1 measurement raised")
    res["control_1"] = c1 = control_1(reference, ctl["per"])
    if not c1["ok"]:
        return skip_all("control_1 (default path unchanged) failed or was incomplete")
    for a in ARMS:
        if res["error"] is not None:
            res["not_run"][a] = "an earlier arm's measurement raised"
            continue
        if clock() >= deadline:
            res["not_run"][a] = "deadline"
            continue
        res["arms"][a] = run_arm(
            a,
            root,
            CHECKPOINTS[a],
            spawn=lambda s, d, a=a: spawn(a, s, d),
            **common,
        )
        res["error"] = res["error"] or res["arms"][a]["error"]
    if "B" in res["arms"]:
        res["resume_check"] = read_resume_checks(root)
    return res


# --------------------------------------------------------------------------- #
# the rule (PREREG "Primary readout")
# --------------------------------------------------------------------------- #


def r_quantity(m: dict) -> float:
    """heldout.live.answer_acc - heldout.slots_zeroed.answer_acc at gap_2_to_M
    (scaffold-timing's, imported)."""
    return ST.r_quantity(m)


def c_quantity(m: dict) -> float:
    """heldout.live.answer_brier_over_16 at gap_2_to_M."""
    return float(m["heldout"]["live"][BUCKET]["answer_brier_over_16"])


def probe_r_quantity(m: dict) -> float:
    """R on the probe [0, 64): train.live - train.slots_zeroed answer_acc at
    gap_2_to_M. For arm A the probe is unseen: a second held-out read."""
    return ST.train_memory_share(m)


def r_holds(values: list[float]) -> bool:
    """R(c): the quantity is >= DELTA on EVERY seed (scaffold-timing's ``holds``)."""
    return ST.holds(values)


def c_holds(values: list[float]) -> bool:
    """C(c): Brier16(live) <= 0.9375 - BRIER_MARGIN on EVERY seed."""
    return len(values) == len(SEEDS) and all(v <= C_THRESHOLD for v in values)


def arm_readout(per: dict[int, dict[int, dict]], checkpoints) -> dict[int, dict]:
    """R and C at every listed checkpoint measured on every seed."""
    out = {}
    for c in checkpoints:
        if not all(c in per.get(s, {}) for s in SEEDS):
            continue
        tabs = [per[s][c]["s003"] for s in SEEDS]
        rq = [r_quantity(t) for t in tabs]
        cq = [c_quantity(t) for t in tabs]
        out[c] = {"R_quantity": rq, "R": r_holds(rq), "C_quantity": cq, "C": c_holds(cq)}
    return out


def classify(r_a: bool, r_b: bool) -> str:
    """PREREG classification table, rows 2-5."""
    if r_a and r_b:
        return "DATA_SUFFICES"
    if r_b:
        return "SCAFFOLD"
    if r_a:
        return "TRAP"
    return "NEITHER"


def missing_final(res: dict) -> list[str]:
    """``<arm>.seed<s>`` whose ckpt 3000 was not measured."""
    return [
        f"{a}.seed{s}"
        for a in ARMS
        for s in SEEDS
        if FINAL not in res["arms"].get(a, {}).get("per", {}).get(s, {})
    ]


def verdict(res: dict) -> dict:
    """PREREG classification table, row for row. Every reason for inconclusive is
    listed, not only the first."""
    base: dict[str, Any] = {"R3000": {}, "C3000": {}, "usable": {}}
    for a in ARMS:
        ro = arm_readout(res["arms"].get(a, {}).get("per", {}), (FINAL,)).get(FINAL)
        base["R3000"][a] = ro["R"] if ro else None
        base["C3000"][a] = ro["C"] if ro else None
        base["usable"][a] = (ro["R"] and ro["C"]) if ro else None
    why = []
    if res.get("error"):
        why.append(f"a measurement raised ({res['error']})")
    pf = res.get("preflight")
    if not pf or not pf.get("ok"):
        why.append(
            "the stream-disjointness / vocabulary-closure check failed or did not run"
            + (f" ({pf['error']})" if pf and pf.get("error") else "")
        )
    c1 = res.get("control_1")
    if c1 is None:
        why.append("control_1 (default path unchanged) did not run")
    elif not c1["ok"]:
        bad = [f"seed{s}" for s, p in c1["per_seed"].items() if not p["ok"]]
        why.append(f"control_1 (default path unchanged) failed on {', '.join(bad)}")
    rc = res.get("resume_check") or {}
    bad2 = [f"seed{s}" for s in SEEDS if not (rc.get(s) or {}).get("ok")]
    if bad2:
        why.append(f"control_2 (resume is exact) failed or missing on {', '.join(bad2)}")
    miss = missing_final(res)
    if miss:
        why.append(f"ckpt{FINAL} not measured for {', '.join(miss)}")
    if why:
        return {
            **base,
            "classification": "inconclusive",
            "exit": Exit.DID_NOT_RUN,
            "detail": "; ".join(why),
        }
    r_a, r_b = base["R3000"]["A"], base["R3000"]["B"]
    cls = classify(r_a, r_b)
    return {
        **base,
        "classification": cls,
        "exit": Exit.OK,
        "detail": f"R3000(A) {'holds' if r_a else 'fails'}, R3000(B) "
        f"{'holds' if r_b else 'fails'}: {cls}",
    }


# --------------------------------------------------------------------------- #
# secondary readouts from the heartbeats
# --------------------------------------------------------------------------- #


def stream_loss_windows(heartbeat: Path, window: int = STREAM_LOSS_WINDOW) -> dict:
    """``{window_start: mean loss_answer_tokens}`` over COMPLETE windows
    ``[k * window, (k + 1) * window)``; a window missing any step is left out."""
    by_step: dict[int, float] = {}
    if heartbeat.exists():
        for line in heartbeat.read_text().splitlines():
            with contextlib.suppress(json.JSONDecodeError):
                rec = json.loads(line)
                if rec.get("kind") == "beat" and "loss_answer_tokens" in rec:
                    by_step[int(rec["step"])] = float(rec["loss_answer_tokens"])
    out = {}
    for w0 in sorted({s - s % window for s in by_step}):
        steps = range(w0, w0 + window)
        if all(s in by_step for s in steps):
            out[w0] = sum(by_step[s] for s in steps) / window
    return out


def first_window_below(windows: dict, threshold: float = STREAM_LOSS_THRESHOLD):
    """The first complete window whose mean is below ``threshold``, or None."""
    for w0 in sorted(windows):
        if windows[w0] < threshold:
            return w0
    return None


# --------------------------------------------------------------------------- #
# the ledger
# --------------------------------------------------------------------------- #


def _stat(led, key: str, xs: list, *, how: str) -> None:
    """The corpus-size rule: fewer than every seed, or a None, is a note."""
    CSC._stat(led, key, xs, how=how)


def arm_rows(led, arm: str, per: dict[int, dict[int, dict]]) -> None:
    """Every readout at every listed checkpoint of one arm, in the corpus-size
    naming with the arm as prefix (``<arm>.ckpt<c>.<set>.<condition>.<bucket>.
    <readout>``), plus R, C and the scaffold-timing derived quantities."""
    how = (
        f"{CSC.EXPERIMENT}::measure_checkpoint (imported) -> {S003.EXPERIMENT}::measure "
        f"with sets=doc_sets(seed, 64); per-seed samples in seed order"
    )
    for c in CHECKPOINTS[arm]:
        seeds = [s for s in SEEDS if c in per.get(s, {})]
        if not seeds:
            continue
        p = f"{arm}.ckpt{c}"
        led.note(f"{p}.seeds_measured", seeds, how="run_arm()")
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
        led.note(
            f"{p}.memory_gates",
            {f"seed{s}": per[s][c]["s003"]["memory_gates"] for s in seeds},
            how="trained checkpoint's memory_gate per C block",
        )
        derived = {**ST.DERIVED, "probe_R_quantity": probe_r_quantity}
        for k, f in derived.items():
            _stat(
                led,
                f"{p}.{k}",
                [f(t) for t in tabs],
                how=f"per seed, {BUCKET}: {f.__doc__.split(chr(10))[0]}",
            )
        _stat(
            led,
            f"{p}.C_quantity",
            [c_quantity(t) for t in tabs],
            how=f"per seed: heldout.live.{BUCKET}.answer_brier_over_16",
        )
        if seeds == SEEDS:
            led.note(
                f"{p}.R_holds",
                r_holds([r_quantity(t) for t in tabs]),
                how=f"R_quantity >= {DELTA} every seed",
            )
            led.note(
                f"{p}.C_holds",
                c_holds([c_quantity(t) for t in tabs]),
                how=f"C_quantity <= {C_THRESHOLD} every seed",
            )
            led.note(
                f"{p}.probe_R_holds",
                r_holds([probe_r_quantity(t) for t in tabs]),
                how=f"probe_R_quantity >= {DELTA} every seed",
            )
            led.note(
                f"{p}.R_reaches_delta_on_seeds",
                ST.seeds_reaching_delta([r_quantity(t) for t in tabs]),
                how=f"seeds whose R_quantity >= {DELTA}",
            )


def scaffold_start(doc: dict) -> dict[str, list[float]]:
    """Arm B's ckpt-1000 values, as the scaffold-timing ledger holds them."""
    rows = {r["key"]: r for r in doc["rows"] if r.get("kind") == "statistic"}
    keys = {
        "R_quantity": "n64.ckpt1000.R_quantity",
        f"train.live.{BUCKET}.answer_acc": f"n64.ckpt1000.train.live.{BUCKET}.answer_acc",
        f"train.slots_zeroed.{BUCKET}.answer_acc": (
            f"n64.ckpt1000.train.slots_zeroed.{BUCKET}.answer_acc"
        ),
    }
    return {k: [float(x) for x in rows[v]["samples"]] for k, v in keys.items()}


def write_rows(led, res: dict, v: dict, start: dict[str, list[float]]) -> None:
    for a in ARMS:
        if a in res["arms"]:
            arm_rows(led, a, res["arms"][a]["per"])
    pf = res.get("preflight") or {}
    led.note("preflight.ok", pf.get("ok"), how="stream_disjointness + closure")
    led.note("preflight.error", pf.get("error"), how="preflight(); None if none")
    led.note(
        "stream_disjointness",
        pf.get("disjointness"),
        how=f"ids of STREAM.window(t, {BATCH}), t in [0, {ITERS}), vs PROBE, HELDOUT",
    )
    led.note(
        "vocabulary_closure",
        {f"seed{s}": c for s, c in (pf.get("closure") or {}).items()},
        how="rsr.train.loop.check_vocabulary_closure over the stream and both sets",
    )
    c1 = res.get("control_1") or {}
    per_seed = c1.get("per_seed", {})
    for f in ("ok", "n_keys_compared", "max_abs_diff"):
        led.note(
            f"control_1.{f}",
            [per_seed[s][f] if s in per_seed else None for s in SEEDS],
            how=f"per seed, seed order {SEEDS}: every {CONTROL_PREFIX}* statistic key "
            f"of {REF_LEDGER} vs the default-path re-train, rows by "
            f"{ST.EXPERIMENT}::arm_rows",
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
    led.note("control_1.passed", c1.get("ok"), how="all seeds ok")
    rc = res.get("resume_check") or {}
    led.note(
        "control_2.ok",
        [(rc.get(s) or {}).get("ok") for s in SEEDS],
        how="per seed: arm-B child's resume_check.json (model and optimizer == "
        "ckpt-001000.pt exactly, before the first step)",
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
        how="the arm-B children's resume_check.json, verbatim",
    )
    # B growth and forgetting references (the scaffold-timing ledger's ckpt 1000)
    for k, xs in start.items():
        led.note(
            f"B.start.{k}",
            xs,
            how=f"{REF_LEDGER} n64.ckpt1000.{k} samples (arm B's start checkpoint)",
        )
    b = res["arms"].get("B", {}).get("per", {})
    if all(FINAL in b.get(s, {}) for s in SEEDS):
        _stat(
            led,
            "B.growth_R_quantity",
            [
                r_quantity(b[s][FINAL]["s003"]) - start["R_quantity"][i]
                for i, s in enumerate(SEEDS)
            ],
            how="per seed: B.ckpt3000.R_quantity - B.start.R_quantity",
        )
    # stream answer loss, from the heartbeats
    root = res.get("root")
    for a in ARMS:
        if a not in res["arms"] or root is None:
            continue
        wins = {
            s: stream_loss_windows(Path(root) / a / f"seed{s}" / "heartbeat.jsonl")
            for s in SEEDS
        }
        led.note(
            f"{a}.stream_answer_loss_window_means",
            {f"seed{s}": {str(k): x for k, x in w.items()} for s, w in wins.items()},
            how=f"heartbeat loss_answer_tokens, mean over each complete "
            f"{STREAM_LOSS_WINDOW}-step window, keyed by its first step",
        )
        led.note(
            f"{a}.stream_answer_loss_first_window_below",
            {f"seed{s}": first_window_below(w) for s, w in wins.items()},
            how="first window whose mean < stream_loss_threshold; None if none",
        )
    led.note("stream_loss_threshold", STREAM_LOSS_THRESHOLD, how="CHANCE - 0.10")
    for a in ARMS:
        arm = res["arms"].get(a)
        led.note(f"{a}.stopped_because", arm and arm["stopped"], how="run_arm()")
        led.note(f"{a}.measurement_error", arm and arm["error"], how="run_arm()")
        led.note(f"{a}.elapsed_s", arm and arm["elapsed_s"], how="parent wall clock")
        led.note(
            f"{a}.checkpoints_measured",
            arm and {f"seed{s}": arm["checkpoints_measured"][s] for s in SEEDS},
            how="run_arm()",
        )
        led.note(f"{a}.R3000", v["R3000"][a], how="R_holds at ckpt3000; None if absent")
        led.note(f"{a}.C3000", v["C3000"][a], how="C_holds at ckpt3000; None if absent")
        led.note(f"{a}.usable", v["usable"][a], how="R3000 and C3000")
    led.note("arms_not_run", res.get("not_run"), how="run_all()")
    led.note("stopped_because", res.get("stopped"), how="run_all(); None if none")
    led.note("measurement_error", res.get("error"), how="run_all(); None if none")
    led.note("classification", v["classification"], how="PREREG classification table")
    led.note("classification_detail", v["detail"], how="verdict()")
    led.note("delta", DELTA, how="PREREG thresholds (scaffold-timing's, imported)")
    led.note("brier_margin", BRIER_MARGIN, how="PREREG thresholds")
    led.note("c_threshold", C_THRESHOLD, how="brier16_uniform - brier_margin")
    led.note("brier16_uniform", BRIER16_UNIFORM, how="1 - 1/16")
    led.note("reproduction_tolerance", REPRO_TOL, how="PREREG thresholds")
    led.note("chance_ln16", CHANCE, how="S0-03 run.py CHANCE, imported")
    led.note(
        "checkpoints_preregistered",
        {a: list(c) for a, c in CHECKPOINTS.items()},
        how="PREREG front matter",
    )
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
        "instrument": f"{CSC.EXPERIMENT}::measure_checkpoint (imported, identity), n=64",
        "arms": list(ARMS),
        "seeds": SEEDS,
        "iters": ITERS,
        "resume_step": RESUME_STEP,
        "ckpt_every": CKPT_EVERY,
        "checkpoints": {a: list(c) for a, c in CHECKPOINTS.items()},
        "stream": dataclasses.asdict(STREAM),
        "stream_documents": [STREAM.offset, STREAM.offset + STREAM.stride * ITERS],
        "heldout_documents": list(HELDOUT),
        "probe_documents": list(PROBE),
        "vocab_documents": list(VOCAB_DOCUMENTS),
        "delta": DELTA,
        "brier_margin": BRIER_MARGIN,
        "c_threshold": C_THRESHOLD,
        "reproduction_tolerance": REPRO_TOL,
        "control_1": f"{CSC.EXPERIMENT} --single --n-documents {CONTROL_ARM} --iters "
        f"{CONTROL_ITERS} (train() without the stream), ckpt {CONTROL_CHECKPOINT} vs "
        f"{REF_LEDGER} {CONTROL_PREFIX}* statistic keys",
        "stream_loss_threshold": STREAM_LOSS_THRESHOLD,
        "deadline": DEADLINE,
        "device": DEVICE,
        "threads_per_seed": THREADS_PER_SEED,
        "seeds_trained_in_parallel": True,
        "arms_in_parallel": False,
        "parent_torch_num_threads": parent_threads,
        "source_root": str(source_root),
        "arm_B_start_checkpoint_sha256": start_sha,
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
    runs = {"control": res["control"], **res["arms"]}
    for a, arm in runs.items():
        if not arm:
            continue
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
                        k: r.get(k)
                        for k in (
                            "config_hash",
                            "steps_done",
                            "torch_num_threads",
                            "run_id",
                            "stream",
                        )
                    },
                    how="the child's result.json (train() return + torch threads)",
                )
    v = verdict(res)
    measured = sorted({s for a in res["arms"].values() for s in SEEDS if a["per"].get(s)})
    led.run_meta(
        device=DEVICE,
        seeds_actually_run=measured or None,
        steps_requested=ITERS,
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


COLUMNS = (
    "R_quantity",
    "R_holds",
    "C_quantity",
    "C_holds",
    "probe_R_quantity",
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
        "# Fresh stream -- RESULTS",
        "",
        f"<!-- GENERATED by `{EXPERIMENT}` from runs/{rid}/ledger.json. Do not edit: "
        "regenerate with `--render-results`. -->",
        "",
        f"Command: `experiments/fresh-stream/run.sh` (the parent is `uv run python "
        f"{EXPERIMENT}`). Ledger `status` of run `{rid}`: **{doc.get('status')}**.",
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
        f"Why, for run `{rid}` (`classification_detail`): "
        f"**{val(rows, 'classification_detail')}**",
        "",
        "| arm | `R3000` | `C3000` | `usable` (R3000 and C3000) |",
        "|---|---|---|---|",
        *[
            f"| {a} | {fmt(val(rows, f'{a}.R3000'))} | {fmt(val(rows, f'{a}.C3000'))} | "
            f"{fmt(val(rows, f'{a}.usable'))} |"
            for a in ARMS
        ],
        "",
        "The RSR gate is read by the owner from the `usable` line; this run does not "
        "open it.",
        "",
        f"`delta` {fmt(val(rows, 'delta'))} · `c_threshold` "
        f"{fmt(val(rows, 'c_threshold'))} · `brier_margin` "
        f"{fmt(val(rows, 'brier_margin'))} · `measurement_error` "
        f"{fmt(val(rows, 'measurement_error'))} · `stopped_because` "
        f"{fmt(val(rows, 'stopped_because'))} · `arms_not_run` "
        f"{fmt(val(rows, 'arms_not_run')) or '—'}.",
        "",
        "Whether the cognitive claim moved: **it cannot move here.** FIFO only, no "
        "retention loss; nothing about RSR, eviction, ψ̂ or Kintsch & van Dijk "
        "(PREREG, *What this does not establish*). No scaling claim.",
        "",
        f"The pre-registered expectation of run `{rid}` is its `expected` row, frozen "
        "in the manifest before any child ran; it is not restated here.",
        "",
        "## Before any step: the stream and the vocabulary",
        "",
        f"`preflight.ok` {fmt(val(rows, 'preflight.ok'))} · `preflight.error` "
        f"{fmt(val(rows, 'preflight.error'))}.",
        "",
    ]
    dj = val(rows, "stream_disjointness") or {}
    out.append(
        " · ".join(
            f"`stream_disjointness.{k}` {fmt(dj.get(k)) or '—'}"
            for k in ("ok", "first_id", "last_id", "repeats", "n_ids")
        )
        + " · "
        + " · ".join(
            f"`stream_disjointness.{k}` {fmt(dj.get(k)) or 'none'}"
            for k in ("ids_in_probe", "ids_in_vocab_documents", "ids_in_heldout")
        )
        + "."
    )
    out.append("")
    cl = val(rows, "vocabulary_closure") or {}
    for s, c in sorted(cl.items()):
        out.append(
            "- "
            + " · ".join(
                f"`vocabulary_closure.{s}.{k}` {fmt(c.get(k))}"
                for k in (
                    "ok",
                    "n_stream_documents",
                    "n_measurement_documents",
                    "vocab_words",
                )
            )
            + "."
        )
    out += [
        "",
        "## `control_1`: the default path unchanged (`n64.ckpt100`, re-trained)",
        "",
        f"`reproduction_tolerance` {val(rows, 'reproduction_tolerance')!r} absolute · "
        f"`control_1.n_reference_keys` {fmt(val(rows, 'control_1.n_reference_keys'))} "
        f"· `control_1.passed` {fmt(val(rows, 'control_1.passed'))} · "
        f"`control_1.missing_keys` {fmt(val(rows, 'control_1.missing_keys')) or '—'}.",
        "",
        "| seed | `control_1.ok` | `control_1.n_keys_compared` | "
        "`control_1.max_abs_diff` |",
        "|---|---|---|---|",
    ]
    cols = [
        val(rows, f"control_1.{f}") or [None] * len(SEEDS)
        for f in ("ok", "n_keys_compared", "max_abs_diff")
    ]
    for i, s in enumerate(SEEDS):
        cells = [
            fmt(x[i]) if isinstance(x[i], bool | None | int) else repr(x[i]) for x in cols
        ]
        out.append(f"| seed{s} | " + " | ".join(cells) + " |")
    fails = val(rows, "control_1.failures") or {}
    if any(fails.values()):
        out += [
            "",
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
    out += [
        "",
        "## `control_2`: arm B's resume is exact",
        "",
        "| seed | `control_2.ok` | `control_2.model_n_tensors` | "
        "`control_2.optimizer_n_tensors` |",
        "|---|---|---|---|",
    ]
    cols = [
        val(rows, f"control_2.{f}") or [None] * len(SEEDS)
        for f in ("ok", "model_n_tensors", "optimizer_n_tensors")
    ]
    for i, s in enumerate(SEEDS):
        out.append(f"| seed{s} | " + " | ".join(fmt(x[i]) for x in cols) + " |")
    out.append("")
    for a in ARMS:
        out += [
            f"## Arm `{a}`; per-seed samples in seed order, bucket `{BUCKET}`",
            "",
            f"`{a}.stopped_because` {fmt(val(rows, f'{a}.stopped_because'))} · "
            f"`{a}.elapsed_s` {fmt(val(rows, f'{a}.elapsed_s'))}.",
            "",
            "| arm.checkpoint | "
            + " | ".join(f"`{k}`" for k in COLUMNS)
            + " | `heldout.live.gap_2_to_M.answer_acc` "
            "| `train.live.gap_2_to_M.answer_acc` "
            "| `train.slots_zeroed.gap_2_to_M.answer_acc` |",
            "|---|" + "---|" * (len(COLUMNS) + 3),
        ]
        for c in CHECKPOINTS[a]:
            p = f"{a}.ckpt{c}"
            if f"{p}.seeds_measured" not in rows:
                out.append(f"| {p} | not measured |" + " |" * (len(COLUMNS) + 2))
                continue
            keys = [f"{p}.{k}" for k in COLUMNS] + [
                f"{p}.heldout.live.{BUCKET}.answer_acc",
                f"{p}.train.live.{BUCKET}.answer_acc",
                f"{p}.train.slots_zeroed.{BUCKET}.answer_acc",
            ]
            out.append(f"| {p} | " + " | ".join(fmt(val(rows, k)) for k in keys) + " |")
        key = f"{a}.stream_answer_loss_first_window_below"
        first = val(rows, key) or {}
        out += [
            "",
            "Stream answer loss (heartbeat `loss_answer_tokens`), the first complete "
            f"window whose mean is below `stream_loss_threshold` "
            f"{fmt(val(rows, 'stream_loss_threshold'))}, keyed by its first step: "
            + (
                " · ".join(f"`{key}.{s}` {fmt(w)}" for s, w in sorted(first.items()))
                or "—"
            )
            + ".",
            "",
        ]
    out += [
        "## Arm B from its start (`B.start`: scaffold-timing's `n64.ckpt1000`)",
        "",
        f"`B.start.R_quantity` {fmt(val(rows, 'B.start.R_quantity'))} · "
        f"`B.growth_R_quantity` {fmt(val(rows, 'B.growth_R_quantity'))}.",
        "",
        f"`B.start.train.live.{BUCKET}.answer_acc` "
        f"{fmt(val(rows, f'B.start.train.live.{BUCKET}.answer_acc'))} · "
        f"`B.start.train.slots_zeroed.{BUCKET}.answer_acc` "
        f"{fmt(val(rows, f'B.start.train.slots_zeroed.{BUCKET}.answer_acc'))} "
        "(forgetting: the arm-B table's probe columns).",
        "",
        "Every other bucket, condition and readout, the memory gates and the stream "
        "loss per window are in the ledger under `<arm>.ckpt<checkpoint>.` and "
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
    dj = stream_disjointness()
    lines = [
        "fresh-stream dry run (NOTHING TRAINED OR MEASURED).",
        f"stream: {dataclasses.asdict(STREAM)}; step t trains on documents "
        f"[{STREAM.offset} + {STREAM.stride} t, ... + {BATCH}); ids "
        f"{dj['first_id']}..{dj['last_id']}, repeats {dj['repeats']}, disjoint "
        f"{dj['ok']}; arm B's first window {dj['first_window_of_B'][:3]}...",
        "before any step: control 3 (above) and the vocabulary closure over "
        f"{dj['n_ids']} stream documents per seed + both measurement sets",
        f"control 1: {CSC.EXPERIMENT} --single --n-documents {CONTROL_ARM} --iters "
        f"{CONTROL_ITERS} x seeds {SEEDS}; ckpt{CONTROL_CHECKPOINT} vs {REF_LEDGER} "
        f"{CONTROL_PREFIX}* within {REPRO_TOL}",
    ]
    for a in ARMS:
        argv = child_argv(a, 0, ROOT / "runs" / RUN_ID / a / "seed0", source_root)
        lines.append(
            f"arm {a}: {'fresh init' if a == 'A' else f'resume ckpt{RESUME_STEP}'} -> "
            f"{ITERS}, ckpt_every {CKPT_EVERY}; measure {list(CHECKPOINTS[a])}; "
            f"child argv {argv[2:]}"
        )
    for s in SEEDS:
        f = start_ckpt(s, source_root)
        lines.append(f"  arm B seed {s} start: {f}  exists={f.exists()}")
    lines += [
        f"deadline {DEADLINE} ({(deadline_epoch() - now) / 3600:.2f} h from now)",
        f"PREDICTED ~{PRED_S_PER_STEP * ITERS / 3600:.2f} h arm A, "
        f"~{PRED_S_PER_STEP * (ITERS - RESUME_STEP) / 3600:.2f} h arm B (prediction, "
        "not premise: corpus-size heartbeats, linear)",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> Exit:
    ap = ArgumentParser(prog="fresh-stream")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; run nothing")
    ap.add_argument(
        "--render-results", action="store_true", help="RESULTS.md from the ledger"
    )
    ap.add_argument("--run-id", default=RUN_ID)
    ap.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--arm", choices=ARMS, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args(argv)

    if a.single:
        if a.out_dir is None or a.arm is None:
            refuse(Exit.DID_NOT_RUN, "--single needs --out-dir and --arm")
        a.out_dir.mkdir(parents=True, exist_ok=True)
        r = single(a.arm, a.seed, a.out_dir, iters=a.iters, source_root=a.source_root)
        (a.out_dir / "result.json").write_text(
            json.dumps(r, indent=2, default=str) + "\n"
        )
        return Exit.OK
    if a.dry_run:
        print(dry_run(a.source_root))
        return Exit.OK
    root = ROOT / "runs" / a.run_id
    results = ROOT / "experiments" / "fresh-stream" / "RESULTS.md"
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
    start = scaffold_start(ref_doc)
    ref_manifest = json.loads((ROOT / REF_MANIFEST).read_text())
    start_sha = {}
    for s in SEEDS:
        f = start_ckpt(s, a.source_root)
        if not f.exists():
            refuse(Exit.DID_NOT_RUN, f"arm B's start checkpoint {f} is absent")
        start_sha[f"seed{s}"] = sha = sha256_file(f)
        rel = f"n{CONTROL_ARM}/seed{s}/{f.name}"
        if ref_manifest["checkpoint_sha256"].get(rel) != sha:
            refuse(
                Exit.DID_NOT_RUN,
                f"{f} sha256 {sha} is not the one {REF_MANIFEST} measured ({rel})",
            )
        if RC.stored_step(f) != RESUME_STEP:
            refuse(Exit.DID_NOT_RUN, f"{f} does not store step {RESUME_STEP}")
    # Measured at the thread count the reference was measured at (scaffold-timing
    # sequential mode, its manifest's torch_threads).
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
    led.note("arm_B_start_checkpoint_sha256", start_sha, how="sha256 of ckpt-001000.pt")
    led.note("parent_torch_threads", torch.get_num_threads(), how=f"{REF_MANIFEST}")
    return execute(
        led,
        root,
        results_path=results,
        start=start,
        preflight_fn=preflight,
        spawn_control=control_child,
        spawn=lambda arm, s, d: Child(arm, s, d, a.source_root),
        measure_fn=measure_checkpoint,
        reference=reference,
        deadline=deadline_epoch(),
    )


if __name__ == "__main__":
    run_main(main)
