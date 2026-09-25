"""The corpus-size curve -- does memory's held-out benefit survive a training set
too large to memorise?

Pre-registration: `experiments/corpus-size-curve/PREREG.md`, committed ahead of this
file. Brief: `docs/lab-notes/dispatch-corpus-size-curve.md`. **One factor changes:
the number of training documents, N.** Everything else is S0-03's configuration,
imported; the parent/child machinery is the retrieval curve's, imported where it
is reusable (`experiments/retrieval-curve/run.py`).

Arms ``ARMS`` run in order, one at a time. Per arm, seeds 0, 1, 2 train in parallel
child processes at 5 OMP/MKL threads (the retrieval curve's ``child_env``, imported)
to ``ITERS`` with ``ckpt_every=CKPT_EVERY``. The N = 64 arm calls ``train()``
WITHOUT ``n_documents`` -- the default path, byte for byte.

**The parent measures; a training process never does.** For each pre-registered
checkpoint as it appears:

1. the step stored IN the checkpoint must equal its label (``StepMismatch``);
2. S0-03's ``measure()``, imported, on the **common held-out set** (documents
   ``HELDOUT``, disjoint from every arm's ``[0, N)``) and the **memorisation probe**
   (documents ``PROBE``) via its ``sets=`` override;
3. N = 64 at ``CONTROL_CHECKPOINT`` only: S0-03's ``measure()`` with NO override,
   and the **reproduction control** against `runs/s0-03-rewardable-corpus/ledger.json`
   (the retrieval curve's, imported). A failure stops every arm: exit 3.

The verdict reads the N = 4096 arm only (``verdict()``, the PREREG's table row for
row). The parent enforces the absolute ``DEADLINE``.

Usage::

    uv run --extra dev python experiments/corpus-size-curve/run.py --dry-plan
    uv run --extra dev python experiments/corpus-size-curve/run.py
    uv run --extra dev python experiments/corpus-size-curve/run.py --render-results

Exit codes (`rsr.exit_codes`): 0 a verdict was reached · 3 did not run or did not
complete (control failed, a measurement raised, the deadline or a crash cut the
N = 4096 arm before a verdict, a refused precondition).
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
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

EXPERIMENT = "experiments/corpus-size-curve/run.py"
RUN_ID = "corpus-size-curve"
PREREG = "experiments/corpus-size-curve/PREREG.md"
BRIEF_ERRORS = "experiments/corpus-size-curve/BRIEF-ERRORS.md"


def _load(name: str, rel: str):
    """Import an experiment script by path. Imported, never copied."""
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RC = _load("_corpus_curve_rc", "experiments/retrieval-curve/run.py")
S003 = RC.S003

# --------------------------------------------------------------------------- #
# 🔒 PREREG.md front matter, transcribed. Changing any is changing the PREREG.
# --------------------------------------------------------------------------- #

SEEDS = [0, 1, 2]
ARMS = (64, 4096, 512)
PRIMARY_ARM = 4096
CONTROL_ARM = 64
CHECKPOINTS = (300, 1000)
CONTROL_CHECKPOINT = 300
ITERS = 1000
CKPT_EVERY = 100
REPRO_TOL = 1e-6
BRIER_MARGIN = 0.05
BRIER16_UNIFORM = 1.0 - 1.0 / 16.0
HELDOUT = (4096, 4160)
PROBE = (0, 64)
DEADLINE = "2026-09-25T07:30:00-07:00"
THREADS_PER_SEED = RC.THREADS_PER_SEED
DEVICE = RC.DEVICE
MARGIN = S003.MARGIN
CHANCE = S003.CHANCE
BUCKET = "gap_2_to_M"

FALSIFIER = (
    "With a training set too large to memorise (N = 4096 documents; S0-03's "
    "configuration otherwise), the working memory does not help held-out answers at "
    "2 <= gap <= M: at neither checkpoint (300, 1000) does every seed show "
    "Brier16(slots_zeroed) - Brier16(live) >= BRIER_MARGIN on the common held-out "
    "set with bar 1 holding."
)

#: PREREG "Author's expectation", condensed; frozen into the manifest before any
#: child starts.
EXPECTED = (
    "B fails at ckpt300 in every arm; at ckpt1000 the N=4096 arm shows B on every "
    "seed a little more often than not (~55%); N=64 shows held-out live NLL far above "
    "chance, N=4096 near chance. Main risk: 1000 iterations may be too few at N=4096 "
    "(PREREG 'Author's expectation', written before any data)."
)

POLL_S = RC.POLL_S
STOP_GRACE_S = RC.STOP_GRACE_S
#: Prediction for --dry-plan only; no rule reads it (retrieval-curve elapsed_s
#: 10934.9 s for 3000 iterations, PREREG "Deadline and scheduling").
PRED_S_PER_ITER = 10934.9 / 3000


class StepMismatch(RuntimeError):
    """A checkpoint's stored step is not the label it is measured under."""


def deadline_epoch() -> float:
    return _dt.datetime.fromisoformat(DEADLINE).timestamp()


# --------------------------------------------------------------------------- #
# document sets
# --------------------------------------------------------------------------- #


def _generate(seed: int, n: int):
    from rsr.data.synthetic import SyntheticConfig, generate

    S = S003.CONFIG["steps_per_stream"]
    return generate(SyntheticConfig(sentences_per_document=S, seed=seed, n_documents=n))


def doc_sets(seed: int, n_train: int):
    """``({"heldout": docs[HELDOUT], "train": docs[PROBE]}, vocab, V)`` -- the triple
    S0-03's ``measure(sets=...)`` takes. The vocabulary is built from the arm's own
    training documents ``[0, n_train)``, as ``train()`` builds it. Raises if the
    held-out set overlaps the training set or uses a word outside its vocabulary."""
    from rsr.train.loop import build_vocab

    if n_train > HELDOUT[0]:
        raise ValueError(f"N={n_train} overlaps the held-out documents {HELDOUT}")
    docs = _generate(seed, HELDOUT[1])
    tr = docs[:n_train]
    ho = docs[HELDOUT[0] : HELDOUT[1]]
    probe = docs[PROBE[0] : PROBE[1]]
    vmap = build_vocab(tr)
    missing = set(build_vocab(ho)) - set(vmap)
    if missing:
        raise SystemExit(f"seed {seed}: held-out words not in the vocab: {missing}")
    return {"heldout": ho, "train": probe}, vmap, 4 + len(vmap)


def corpus_sha256(seed: int, n: int) -> str:
    """sha256 of the seed's training documents at N, as ``train()`` generates them."""
    blob = json.dumps([dataclasses.asdict(d) for d in _generate(seed, n)], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# one training run, in its own child process
# --------------------------------------------------------------------------- #


def single(seed: int, iters: int, n: int, out_dir: Path) -> dict:
    """The retrieval curve's ``single()`` plus ``n_documents``. The control arm
    omits it: the default path, byte for byte."""
    from rsr.train.loop import train

    C = S003.CONFIG
    kw = {} if n == CONTROL_ARM else {"n_documents": n}
    r = train(
        d=C["d"],
        steps_per_stream=C["steps_per_stream"],
        batch=C["batch"],
        vocab=None,
        max_tokens=C["max_tokens"],
        memory_slots=C["memory_slots"],
        iters=iters,
        lr=C["lr"],
        seed=seed,
        device=DEVICE,
        out_dir=out_dir,
        beat_every=1,
        ckpt_every=CKPT_EVERY,
        policy_name=C["policy"],
        masked_loss=C["masked_loss"],
        srep_norm_reg_weight=C["srep_norm_reg_weight"],
        **kw,
    )
    r["seed"] = seed
    r["n_documents"] = n
    r["torch_num_threads"] = torch.get_num_threads()
    return r


def child_argv(seed: int, iters: int, n: int, out_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single",
        "--seed",
        str(seed),
        "--iters",
        str(iters),
        "--n-documents",
        str(n),
        "--out-dir",
        str(out_dir),
    ]


class Child(RC.Child):
    """The retrieval curve's child, launched on THIS script with ``--n-documents``."""

    def __init__(self, seed: int, n: int, out_dir: Path, iters: int = ITERS) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.argv = child_argv(seed, iters, n, out_dir)
        self._log = open(out_dir / "train.log", "w")  # noqa: SIM115 -- closed in wait
        self._p = subprocess.Popen(
            self.argv, cwd=ROOT, stdout=self._log, stderr=self._log, env=RC.child_env()
        )


# --------------------------------------------------------------------------- #
# the measurement (in the parent)
# --------------------------------------------------------------------------- #


def measure_checkpoint(seed_dir: Path, seed: int, label: int, n: int) -> dict:
    """Every readout for one (arm, seed, checkpoint), read from THAT checkpoint."""
    ckpt = RC.ckpt_path(seed_dir, label)
    step = RC.stored_step(ckpt)
    if step != label:
        raise StepMismatch(
            f"N={n} seed {seed}: {ckpt.name} stores step {step}, but is measured as "
            f"checkpoint {label}"
        )
    out = {
        "checkpoint": str(ckpt),
        "stored_step": step,
        "s003": S003.measure(ckpt, seed, DEVICE, sets=doc_sets(seed, n)),
    }
    if n == CONTROL_ARM and label == CONTROL_CHECKPOINT:
        out["s003_default"] = S003.measure(ckpt, seed, DEVICE)
    return out


# --------------------------------------------------------------------------- #
# the rule
# --------------------------------------------------------------------------- #


def brier_gap(m: dict) -> float:
    """Brier16(slots_zeroed) - Brier16(live) at gap_2_to_M, held-out."""
    h = m["heldout"]
    return (
        h["slots_zeroed"][BUCKET]["answer_brier_over_16"]
        - h["live"][BUCKET]["answer_brier_over_16"]
    )


def arm_rule(per_arm: dict[int, dict[int, dict]]) -> dict[int, dict]:
    """Per checkpoint measured on every seed: S0-03's ``decide()`` fed the HELD-OUT
    population only, and ``B`` (every seed's Brier gap >= BRIER_MARGIN)."""
    out = {}
    for c in CHECKPOINTS:
        if not all(c in per_arm.get(s, {}) for s in SEEDS):
            continue
        tables = {s: per_arm[s][c]["s003"] for s in SEEDS}
        d = RC.s003_decide(tables)
        gaps = [brier_gap(tables[s]) for s in SEEDS]
        out[c] = {
            "bar1": d["bar1_zeroed_not_below_chance"],
            "B": all(g >= BRIER_MARGIN for g in gaps),
            "brier_gap": gaps,
            "s003_decision": d,
        }
    return out


def verdict(arms: dict[int, dict], control: dict, error: str | None = None) -> dict:
    """PREREG "Primary readout and decision rule", row for row. ``arms[n]`` is
    ``run_arm``'s record (``per``, ``stopped``) or absent if never started."""
    if error is not None:
        return _v("inconclusive", f"a measurement raised ({error})", Exit.DID_NOT_RUN)
    failed = sorted(s for s, c in control.items() if not c["ok"])
    if failed:
        return _v(
            "inconclusive",
            "the N=64 ckpt-300 reproduction control failed on "
            + ", ".join(f"seed{s}" for s in failed)
            + ": this is not S0-03's configuration, nothing past it is read",
            Exit.DID_NOT_RUN,
        )
    if sorted(control) != SEEDS:
        return _v(
            "inconclusive",
            "the N=64 reproduction control never ran on every seed",
            Exit.DID_NOT_RUN,
        )
    prim = arms.get(PRIMARY_ARM)
    rule = arm_rule(prim["per"]) if prim else {}
    hit = [c for c, r in rule.items() if r["bar1"] and r["B"]]
    if hit:
        return _v(
            "falsified",
            "bar 1 and B hold on every seed at N=4096 "
            + ", ".join(f"ckpt{c}" for c in hit)
            + ": memory helps held-out answers at N=4096",
            Exit.OK,
            hit,
        )
    reached = CHECKPOINTS[-1] in rule
    if (
        reached
        and all(r["bar1"] for r in rule.values())
        and sorted(rule) == sorted(CHECKPOINTS)
    ):
        return _v(
            "survived",
            "N=4096 ckpt1000 reached; bar 1 holds and B fails at every checkpoint: no "
            "memory benefit on held-out answers at N=4096 by ckpt1000",
            Exit.OK,
        )
    if reached:
        bad = [c for c, r in rule.items() if not r["bar1"]]
        return _v(
            "inconclusive",
            "N=4096: bar 1 fails at "
            + ", ".join(f"ckpt{c}" for c in bad)
            + ", where B would be read; no checkpoint shows B with bar 1",
            Exit.OK,
        )
    why = prim["stopped"] if prim else "the arm never started"
    return _v(
        "inconclusive",
        f"the N=4096 arm stopped ({why}) before ckpt1000 without B shown; measured on "
        "every seed: " + (", ".join(f"ckpt{c}" for c in sorted(rule)) or "none"),
        Exit.DID_NOT_RUN,
    )


def _v(outcome: str, detail: str, exit_: Exit, hit: list | None = None) -> dict:
    return {
        "outcome": outcome,
        "detail": detail,
        "falsified_at": hit or [],
        "exit": exit_,
    }


# --------------------------------------------------------------------------- #
# the parent loop
# --------------------------------------------------------------------------- #


def run_arm(
    n: int,
    root: Path,
    *,
    spawn: Callable[[int, int, Path], Any],
    measure_fn: Callable[[Path, int, int, int], dict],
    reference: dict[int, float],
    deadline: float,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    poll_s: float = POLL_S,
    log: Callable[[str], None] = lambda s: print(s, flush=True),
) -> dict:
    """One arm: three seeds in parallel children; each pre-registered checkpoint
    measured in THIS process as it appears, lowest label first across seeds (so every
    control is read before any later checkpoint). The retrieval curve's
    ``run_curve``, with the arm's N and an absolute wall-clock deadline."""
    t0 = clock()
    dirs = {s: root / f"n{n}" / f"seed{s}" for s in SEEDS}
    procs = {s: spawn(s, n, dirs[s]) for s in SEEDS}
    pending = {s: list(CHECKPOINTS) for s in SEEDS}
    per: dict[int, dict[int, dict]] = {s: {} for s in SEEDS}
    control: dict[int, dict] = {}
    is_control = n == CONTROL_ARM

    def sweep() -> str | None:
        failed_control = False
        while True:
            ready = [
                (pending[s][0], s)
                for s in SEEDS
                if pending[s]
                and RC.ckpt_path(dirs[s], pending[s][0]).exists()
                and not (failed_control and pending[s][0] != CONTROL_CHECKPOINT)
            ]
            if not ready:
                return "reproduction control failed" if failed_control else None
            label, s = min(ready)
            pending[s].pop(0)
            per[s][label] = measure_fn(dirs[s], s, label, n)
            log(f"  N={n} seed {s}: measured ckpt{label}")
            if is_control and label == CONTROL_CHECKPOINT:
                control[s] = RC.reproduction_control(
                    s, per[s][label]["s003_default"], reference
                )
                log(f"  N={n} seed {s}: reproduction control {control[s]}")
                failed_control = failed_control or not control[s]["ok"]

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
            stopped = sweep()
            if stopped or not any(pending.values()):
                break
            if clock() >= deadline:
                stop_all()
                stopped = sweep() or "deadline"
                break
            dead = [s for s in SEEDS if pending[s] and procs[s].poll() is not None]
            if dead:
                stopped = sweep()
                failed = [s for s in SEEDS if pending[s] and procs[s].poll() is not None]
                if stopped or failed:
                    stopped = stopped or "training child exited early"
                    break
                continue
            sleep(poll_s)
    except (Exception, SystemExit) as e:
        error = f"{type(e).__name__}: {e}"
        stopped = "measurement raised"
        log(f"  N={n} measurement raised: {error}")
    finally:
        stop_all()
    exit_codes = {s: p.wait() for s, p in procs.items()}
    return {
        "n": n,
        "per": per,
        "control": control,
        "stopped": stopped,
        "error": error,
        "failed_seeds": failed,
        "exit_codes": exit_codes,
        "argv": {s: getattr(p, "argv", None) for s, p in procs.items()},
        "checkpoints_measured": {s: sorted(per[s]) for s in SEEDS},
        "elapsed_s": clock() - t0,
    }


def run_all(root: Path, **kw) -> dict:
    """The arms in PREREG order. A control failure or a raised measurement stops
    every later arm; an arm whose start comes at or after the deadline is not run."""
    arms: dict[int, dict] = {}
    not_run: dict[int, str] = {}
    control: dict[int, dict] = {}
    error: str | None = None
    clock = kw.get("clock", time.time)
    for n in ARMS:
        if error is not None:
            not_run[n] = "an earlier arm's measurement raised"
            continue
        if CONTROL_ARM in arms and (
            sorted(control) != SEEDS or any(not c["ok"] for c in control.values())
        ):
            not_run[n] = "the reproduction control failed or never ran on every seed"
            continue
        if clock() >= kw["deadline"]:
            not_run[n] = "deadline"
            continue
        arms[n] = run_arm(n, root, **kw)
        if n == CONTROL_ARM:
            control = arms[n]["control"]
        error = error or arms[n]["error"]
    return {"arms": arms, "not_run": not_run, "control": control, "error": error}


# --------------------------------------------------------------------------- #
# the ledger
# --------------------------------------------------------------------------- #


def _stat(led, key: str, xs: list, *, how: str) -> None:
    if len(xs) != len(SEEDS) or any(x is None for x in xs):
        led.note(key, xs, how=how + " [incomplete or contains None: not a statistic]")
        return
    led.stat(key, xs, how=how)


READOUTS = ("answer_brier_over_16", "answer_nll", "answer_acc", "answer_nll_over_16")


def write_rows(led, res: dict) -> dict:
    how = (
        f"{S003.EXPERIMENT}::measure (imported) with sets=doc_sets(seed, N): heldout = "
        f"docs {list(HELDOUT)}, train = docs {list(PROBE)}; the checkpoint whose stored "
        f"step equals its label; measured in the parent; per-seed samples in seed order"
    )
    for n, arm in res["arms"].items():
        per = arm["per"]
        for c in CHECKPOINTS:
            seeds = [s for s in SEEDS if c in per[s]]
            if not seeds:
                continue
            p = f"n{n}.ckpt{c}"
            led.note(f"{p}.seeds_measured", seeds, how="run_arm")
            led.note(
                f"{p}.stored_step",
                {f"seed{s}": per[s][c]["stored_step"] for s in seeds},
                how="rsr.train.checkpoint.load(...)['step'], checked == label",
            )
            for ds in ("heldout", "train"):
                for cond in S003.CONDITIONS:
                    for b in S003.BUCKETS:
                        for key in READOUTS:
                            _stat(
                                led,
                                f"{p}.{ds}.{cond}.{b}.{key}",
                                [per[s][c]["s003"][ds][cond][b][key] for s in seeds],
                                how=f"{how}; set {ds}; condition {cond}; bucket {b}",
                            )
                for key in READOUTS:
                    _stat(
                        led,
                        f"{p}.{ds}.slots_zeroed_minus_live.{BUCKET}.{key}",
                        [
                            per[s][c]["s003"][ds]["slots_zeroed"][BUCKET][key]
                            - per[s][c]["s003"][ds]["live"][BUCKET][key]
                            for s in seeds
                        ],
                        how=f"per seed: {p}.{ds}.slots_zeroed - live, bucket {BUCKET}",
                    )
            for key in READOUTS:
                _stat(
                    led,
                    f"{p}.train_minus_heldout.live.{BUCKET}.{key}",
                    [
                        per[s][c]["s003"]["train"]["live"][BUCKET][key]
                        - per[s][c]["s003"]["heldout"]["live"][BUCKET][key]
                        for s in seeds
                    ],
                    how=f"per seed: probe minus common held-out, live, {BUCKET}",
                )
            led.note(
                f"{p}.memory_gates",
                {f"seed{s}": per[s][c]["s003"]["memory_gates"] for s in seeds},
                how="trained checkpoint's memory_gate per C block",
            )
        for c, r in arm_rule(per).items():
            p = f"n{n}.ckpt{c}"
            led.note(f"{p}.bar1", r["bar1"], how="S0-03 decide() on the common held-out")
            led.note(f"{p}.B", r["B"], how=f"every seed brier_gap >= {BRIER_MARGIN}")
            led.note(f"{p}.s003_decision", r["s003_decision"], how="S0-03 decide()")
            led.note(
                f"{p}.s003_outcome",
                r["s003_decision"]["outcome"],
                how="S0-03's OWN label, about S0-03's falsifier -- not this verdict",
            )
        led.note(f"n{n}.stopped_because", arm["stopped"], how="run_arm")
        led.note(f"n{n}.measurement_error", arm["error"], how="run_arm; None if none")
        led.note(f"n{n}.elapsed_s", arm["elapsed_s"], how="parent wall clock")
        led.note(
            f"n{n}.checkpoints_measured",
            {f"seed{s}": arm["checkpoints_measured"][s] for s in SEEDS},
            how="run_arm",
        )

    control = res["control"]
    for f in ("measured", "reference", "abs_diff", "ok"):
        led.note(
            f"reproduction_control.{f}",
            [control[s][f] if s in control else None for s in SEEDS],
            how=(
                f"N={CONTROL_ARM} ckpt{CONTROL_CHECKPOINT}, S0-03 measure() with no "
                f"override, heldout.live.gap_2_to_M.answer_nll vs {RC.S003_LEDGER} "
                f"{RC.REPRO_KEY} samples; seed order {SEEDS}"
            ),
        )
    led.note("reproduction_tolerance", REPRO_TOL, how="PREREG thresholds")
    led.note(
        "arms_not_run", {f"n{n}": w for n, w in res["not_run"].items()}, how="run_all"
    )
    v = verdict(res["arms"], control, res["error"])
    led.note("arms_preregistered", list(ARMS), how="PREREG front matter")
    led.note("checkpoints_preregistered", list(CHECKPOINTS), how="PREREG thresholds")
    led.note("falsified_at_checkpoints", v["falsified_at"], how="verdict(), N=4096")
    led.note("measurement_error", res["error"], how="run_all; None if none")
    led.note("brier_margin", BRIER_MARGIN, how="PREREG thresholds")
    led.note("brier16_uniform", BRIER16_UNIFORM, how="1 - 1/16, reference only")
    led.note("margin", MARGIN, how="S0-03 run.py MARGIN, imported (bar 1)")
    led.note("chance_ln16", CHANCE, how="S0-03 run.py CHANCE, imported (bar 1)")
    led.note("deadline", DEADLINE, how="PREREG thresholds")
    led.note("expected", EXPECTED, how="manifest, frozen before the first child")
    led.verdict(falsifier=FALSIFIER, outcome=v["outcome"], detail=v["detail"])
    return v


def manifest(parent_threads: int) -> dict:
    return {
        **S003.CONFIG,
        "arms_n_documents": list(ARMS),
        "primary_arm": PRIMARY_ARM,
        "control_arm": CONTROL_ARM,
        "control_arm_passes_n_documents": False,
        "iters": ITERS,
        "ckpt_every": CKPT_EVERY,
        "checkpoints": list(CHECKPOINTS),
        "control_checkpoint": CONTROL_CHECKPOINT,
        "heldout_documents": list(HELDOUT),
        "probe_documents": list(PROBE),
        "device": DEVICE,
        "seeds": SEEDS,
        "threads_per_seed": THREADS_PER_SEED,
        "seeds_trained_in_parallel": True,
        "arms_in_parallel": False,
        "parent_torch_num_threads": parent_threads,
        "reproduction_tolerance": REPRO_TOL,
        "reproduction_reference": f"{RC.S003_LEDGER} rows[key={RC.REPRO_KEY}].samples",
        "deadline": DEADLINE,
        "brier_margin": BRIER_MARGIN,
        "margin": MARGIN,
        "chance_ln16": CHANCE,
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "prereg": PREREG,
        "falsifier": FALSIFIER,
        "expected": EXPECTED,
        "instrument": f"{S003.EXPERIMENT} measure(sets=...) + decide(); Brier16 added",
    }


def execute(led, root: Path, *, results_path: Path | None, **kw) -> Exit:
    res = run_all(root, **kw)
    for n, arm in res["arms"].items():
        for s in SEEDS:
            argv = arm["argv"].get(s)
            if argv:
                rel = " ".join(
                    ["uv run python", EXPERIMENT]
                    + [
                        str(Path(x).relative_to(ROOT)) if x.startswith(str(ROOT)) else x
                        for x in argv[2:]
                    ]
                )
                led.command(
                    rel, exit_code=arm["exit_codes"][s], note=f"N={n} seed {s} training"
                )
            info = root / f"n{n}" / f"seed{s}" / "result.json"
            if info.exists():
                r = json.loads(info.read_text())
                led.note(
                    f"n{n}.seed{s}.training",
                    {
                        k: r.get(k)
                        for k in (
                            "config_hash",
                            "steps_done",
                            "torch_num_threads",
                            "run_id",
                            "n_documents",
                        )
                    },
                    how="the child's result.json (train() return + torch threads)",
                )
    prim = res["arms"].get(PRIMARY_ARM)
    led.run_meta(
        device=DEVICE,
        seeds_actually_run=SEEDS,
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
    complete = (
        not res["not_run"]
        and all(a["stopped"] is None for a in res["arms"].values())
        and prim is not None
    )
    led.status("crashed" if crashed else ("ok" if complete else "partial"))
    v = write_rows(led, res)
    (root / "raw.json").write_text(json.dumps(res, indent=2, default=str) + "\n")
    path = led.write()
    if results_path is not None:
        results_path.write_text(
            render_results(json.loads(path.read_text()), read_brief_errors())
        )
    print(f"ledger: {path}  verdict={v['outcome']}  exit={int(v['exit'])}")
    return v["exit"]


# --------------------------------------------------------------------------- #
# RESULTS.md, rendered from the ledger
# --------------------------------------------------------------------------- #


def read_brief_errors(path: Path | None = None) -> str | None:
    path = path or ROOT / BRIEF_ERRORS
    return path.read_text() if path.exists() else None


def render_results(doc: dict, brief_errors: str | None = None) -> str:
    """Every number beside its ledger key (`render_scoreboard.py --audit`). No
    number is typed here."""
    rows = RC._rows(doc)
    val, fmt = RC._val, RC._fmt
    prov = doc.get("provenance", {})
    v = doc.get("verdict") or {}
    rid = doc.get("run_id")
    shas = val(rows, "corpus_sha256") or {}
    out = [
        "# The corpus-size curve -- RESULTS",
        "",
        f"<!-- GENERATED by `{EXPERIMENT}` from runs/{rid}/ledger.json. Do not edit: "
        f"regenerate with `--render-results`. -->",
        "",
        f"Command: `uv run python {EXPERIMENT}`. Ledger `status` of run `{rid}`: "
        f"**{doc.get('status')}**.",
        "",
        f"Provenance of run `{rid}`: `provenance.git_sha` `{prov.get('git_sha')}` · "
        f"`device` {doc.get('device')} · hardware `provenance.machine` "
        f"`{prov.get('machine')}`, `provenance.platform` `{prov.get('platform')}` · "
        f"`seeds_actually_run` {doc.get('seeds_actually_run')} · `corpus_sha256` "
        + " ".join(f"{k} `{x}`" for k, x in sorted(shas.items()))
        + ".",
        "",
        f"Falsifier of run `{rid}` (`verdict.falsifier`): **{v.get('falsifier')}**",
        "",
        f"## Verdict of run `{rid}`: `verdict.outcome` **{v.get('outcome')}**",
        "",
        f"Why, for run `{rid}` (`verdict.detail`): **{v.get('detail')}**",
        "",
        f"`falsified_at_checkpoints` {val(rows, 'falsified_at_checkpoints')} "
        f"(N=4096; empty unless falsified). `arms_preregistered` "
        f"{val(rows, 'arms_preregistered')}; `arms_not_run` {val(rows, 'arms_not_run')}.",
        "",
        "Whether the cognitive claim moved: **it cannot move here.** FIFO only; nothing "
        "about RSR, eviction policy, ψ̂ or Kintsch & van Dijk (PREREG, *What this does "
        "not establish*). No scaling claim.",
        "",
        f"Pre-registered expectation (`expected`, frozen in the manifest before any "
        f"child ran): **{val(rows, 'expected')}**",
        "",
        "## The reproduction control (N=64, ckpt300, S0-03's own held-out)",
        "",
        f"`reproduction_tolerance` {val(rows, 'reproduction_tolerance')!r} absolute.",
        "",
        "| seed | `reproduction_control.measured` | `reproduction_control.reference` "
        "| `reproduction_control.abs_diff` | `reproduction_control.ok` |",
        "|---|---|---|---|---|",
    ]
    cols = [
        val(rows, f"reproduction_control.{f}") or [None] * len(SEEDS)
        for f in ("measured", "reference", "abs_diff", "ok")
    ]
    for i, s in enumerate(SEEDS):
        cells = [
            fmt(c[i]) if isinstance(c[i], bool) or c[i] is None else repr(c[i])
            for c in cols
        ]
        out.append(f"| seed{s} | " + " | ".join(cells) + " |")
    out += [
        "",
        "## Per arm and checkpoint (common held-out unless named; per-seed samples in "
        "seed order)",
        "",
        f"`brier_margin` {val(rows, 'brier_margin')} · `brier16_uniform` "
        f"{val(rows, 'brier16_uniform')} · `margin` {val(rows, 'margin')} · "
        f"`chance_ln16` {val(rows, 'chance_ln16')}.",
        "",
        "| arm.checkpoint | `bar1` | `B` | "
        "`heldout.slots_zeroed_minus_live.gap_2_to_M.answer_brier_over_16` | "
        "`heldout.live.gap_2_to_M.answer_brier_over_16` | "
        "`heldout.live.gap_2_to_M.answer_acc` | "
        "`heldout.slots_zeroed.gap_2_to_M.answer_acc` | "
        "`heldout.live.gap_2_to_M.answer_nll` | "
        "`train.live.gap_2_to_M.answer_nll` (memorisation probe) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for n in ARMS:
        for c in CHECKPOINTS:
            p = f"n{n}.ckpt{c}"
            if f"{p}.seeds_measured" not in rows:
                out.append(f"| {p} | not measured | | | | | | | |")
                continue
            keys = [
                f"{p}.bar1",
                f"{p}.B",
                f"{p}.heldout.slots_zeroed_minus_live.{BUCKET}.answer_brier_over_16",
                f"{p}.heldout.live.{BUCKET}.answer_brier_over_16",
                f"{p}.heldout.live.{BUCKET}.answer_acc",
                f"{p}.heldout.slots_zeroed.{BUCKET}.answer_acc",
                f"{p}.heldout.live.{BUCKET}.answer_nll",
                f"{p}.train.live.{BUCKET}.answer_nll",
            ]
            out.append(f"| {p} | " + " | ".join(fmt(val(rows, k)) for k in keys) + " |")
    out += [
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
# --dry-plan and CLI
# --------------------------------------------------------------------------- #


def dry_plan(now: float | None = None) -> str:
    now = time.time() if now is None else now
    per_arm = PRED_S_PER_ITER * ITERS
    lines = [
        "corpus-size-curve dry plan (NOTHING RUN).",
        f"arms N={list(ARMS)} in order, one at a time; seeds {SEEDS} as parallel "
        f"children, OMP/MKL {THREADS_PER_SEED} each, device {DEVICE}",
        f"each arm to {ITERS} iters, ckpt_every {CKPT_EVERY}; measure at "
        f"{list(CHECKPOINTS)}; heldout docs {list(HELDOUT)}, probe docs {list(PROBE)}",
        f"N={CONTROL_ARM} calls train() WITHOUT n_documents; control at "
        f"ckpt{CONTROL_CHECKPOINT}: {RC.REPRO_KEY} within {REPRO_TOL} of "
        f"{RC.S003_LEDGER}: {RC.s003_reference()}",
        f"deadline {DEADLINE} ({(deadline_epoch() - now) / 3600:.2f} h from now)",
        f"PREDICTED ~{per_arm / 3600:.2f} h per arm, ~{len(ARMS) * per_arm / 3600:.2f} h "
        "total (prediction, not premise: retrieval-curve elapsed_s, linear)",
        f"child argv: {child_argv(0, ITERS, 4096, ROOT / 'runs' / RUN_ID / 'seed0')}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> Exit:
    ap = ArgumentParser(prog="corpus-size-curve")
    ap.add_argument("--dry-plan", action="store_true", help="print the plan; run nothing")
    ap.add_argument(
        "--render-results", action="store_true", help="RESULTS.md from the ledger"
    )
    ap.add_argument("--run-id", default=RUN_ID)
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--n-documents", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args(argv)

    if a.single:
        if a.out_dir is None or a.n_documents is None:
            refuse(Exit.DID_NOT_RUN, "--single needs --out-dir and --n-documents")
        a.out_dir.mkdir(parents=True, exist_ok=True)
        r = single(a.seed, a.iters, a.n_documents, a.out_dir)
        (a.out_dir / "result.json").write_text(
            json.dumps(r, indent=2, default=str) + "\n"
        )
        return Exit.OK

    if a.dry_plan:
        print(dry_plan())
        return Exit.OK

    root = ROOT / "runs" / a.run_id
    results = ROOT / "experiments" / "corpus-size-curve" / "RESULTS.md"
    if a.render_results:
        path = root / "ledger.json"
        if not path.exists():
            refuse(Exit.DID_NOT_RUN, f"{path} is absent: the curve has not run")
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
    ref_doc = json.loads((ROOT / RC.S003_LEDGER).read_text())
    if ref_doc.get("device") != DEVICE:
        refuse(Exit.DID_NOT_RUN, f"{RC.S003_LEDGER} device is {ref_doc.get('device')!r}")
    reference = RC.s003_reference()

    led = ledger_mod.Ledger(
        a.run_id,
        question=(
            "Does the working memory still help held-out answers at 2 <= gap <= M when "
            "S0-03's configuration is trained on 4096 documents instead of 64?"
        ),
    )
    led.command(
        " ".join(["uv run python", EXPERIMENT, *sys.argv[1:]]),
        exit_code=None,
        note="this run (the parent)",
    )
    m = led.manifest(manifest(torch.get_num_threads()))
    print(f"manifest frozen: {m}", flush=True)
    led.note(
        "corpus_sha256",
        {f"n{n}.seed{s}": corpus_sha256(s, n) for n in ARMS for s in SEEDS},
        how="sha256 of json(asdict(doc)) over the seed's training documents at N",
    )
    led.note("reproduction_reference", reference, how=f"{RC.S003_LEDGER} {RC.REPRO_KEY}")
    return execute(
        led,
        root,
        results_path=results,
        spawn=lambda s, n, d: Child(s, n, d),
        measure_fn=measure_checkpoint,
        reference=reference,
        deadline=deadline_epoch(),
    )


if __name__ == "__main__":
    run_main(main)
