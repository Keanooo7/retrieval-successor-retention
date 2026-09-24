"""The retrieval curve -- does longer training make the memory help answers?

Pre-registration: `experiments/retrieval-curve/PREREG.md`, committed ahead of this
file. Brief: `docs/lab-notes/dispatch-retrieval-curve.md`. **One factor changes:
training length.** Everything else is S0-03's configuration, imported.

Per seed (0, 1, 2), in its own child process at ``OMP_NUM_THREADS=5`` /
``MKL_NUM_THREADS=5`` (S0-03's ``threads_per_seed``, exactly the way S0-03's
``main()`` launches them): `rsr.train.loop.train` with S0-03's ``CONFIG`` to
``ITERS`` iterations, ``ckpt_every=CKPT_EVERY``. The loop writes
``ckpt-{step:06d}.pt`` at every multiple (`src/rsr/train/loop.py:497`) and never
prunes, so every pre-registered checkpoint survives.

**The parent measures; a training process never does** (brief do_not: the decisive
``measure()`` calls ``torch.manual_seed``). The parent polls the checkpoint files
(each is written atomically, so one that exists is complete) and, for each
pre-registered checkpoint as it appears:

1. checks the step stored IN the checkpoint equals its label (``StepMismatch``);
2. S0-03's ``measure()``, imported (`experiments/s0-03-rewardable-corpus/run.py`);
3. the decisive run's ``measure()`` with ``decoy_ckpt=None``, imported
   (`experiments/decisive-shuffle/run.py`) -- it builds the untrained same-seed
   decoy itself;
4. at ``CONTROL_CHECKPOINT`` only: the **reproduction control** against
   `runs/s0-03-rewardable-corpus/ledger.json`. A failure terminates every
   training child at once: verdict ``inconclusive``, exit 3, without waiting.

Then S0-03's ``decide()``, imported and unchanged, per checkpoint, fed the
held-out population only. The verdict is the PREREG's table (``verdict()``); S0-03's
own label is recorded under the separate key ``ckpt<c>.s003_outcome`` -- its
``survived`` is this PREREG's ``falsified`` and the two never share a key.

The parent enforces ``DEADLINE_HOURS`` (``train()`` has no deadline hook): at the
deadline it terminates the children and measures whichever pre-registered
checkpoints are already on disk.

Usage::

    uv run python experiments/retrieval-curve/run.py --dry-plan         # runs nothing
    uv run python experiments/retrieval-curve/run.py                    # the run
    uv run python experiments/retrieval-curve/run.py --render-results   # RESULTS.md

Exit codes (`rsr.exit_codes`): 0 a verdict was reached (``falsified``,
``survived``, or ``inconclusive`` because bar 1 failed with every checkpoint
measured) · 3 did not run or did not complete (the reproduction control failed,
the deadline or a crashed child cut the run before a verdict, a refused
precondition).
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import importlib.util
import json
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
from rsr.train import checkpoint as ck  # noqa: E402

EXPERIMENT = "experiments/retrieval-curve/run.py"
RUN_ID = "retrieval-curve"
PREREG = "experiments/retrieval-curve/PREREG.md"


def _load(name: str, rel: str):
    """Import an experiment script by path. Imported, never copied (brief bar 2)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S003 = _load("_retrieval_curve_s003", "experiments/s0-03-rewardable-corpus/run.py")
DECISIVE = _load("_retrieval_curve_decisive", "experiments/decisive-shuffle/run.py")

# --------------------------------------------------------------------------- #
# 🔒 PREREG.md `thresholds`, transcribed. MARGIN and CHANCE are S0-03's, imported.
# Changing any of these is changing the pre-registration (brief do_not).
# --------------------------------------------------------------------------- #

SEEDS = [0, 1, 2]
CHECKPOINTS = (300, 1000, 3000)
CONTROL_CHECKPOINT = 300
ITERS = CHECKPOINTS[-1]
#: Brief "Files in scope": ckpt_every=100, so 300, 1000 and 3000 are all written.
CKPT_EVERY = 100
REPRO_TOL = 1e-6
DEADLINE_HOURS = 8
THREADS_PER_SEED = 5
DEVICE = "cpu"
MARGIN = S003.MARGIN
CHANCE = S003.CHANCE

#: The reproduction control's reference: S0-03's ledger, the row looked up by key,
#: its samples in S0-03's seed order.
S003_LEDGER = "runs/s0-03-rewardable-corpus/ledger.json"
REPRO_KEY = "heldout.live.gap_2_to_M.answer_nll"

#: PREREG front matter `falsifier`, verbatim (YAML folded to one line).
FALSIFIER = (
    "Training S0-03's configuration (arm B: masked loss, hinge off) for longer does "
    "not make the working memory help answers at 2 <= gap <= M: at no checkpoint up "
    "to 3000 iterations does every seed show slots_zeroed - live answer NLL >= MARGIN "
    "on held-out documents with bar 1 holding."
)

#: PREREG "Author's expectation (written before any data)", condensed without its
#: counts so every sentence of RESULTS.md stays ledger-backed. Frozen into the
#: manifest before the first child starts.
EXPECTED = (
    "survived is at least as likely as falsified: the leading rival cause of "
    "chance-level retrieval is memorisation of the small training set rather than "
    "too little training, and longer training strengthens memorisation "
    "(PREREG 'Author's expectation', written before any data)."
)

#: Operational, not a threshold: how often the parent looks for a new checkpoint,
#: and how long a terminated child gets before it is killed.
POLL_S = 30.0
STOP_GRACE_S = 30.0

#: Brief "Budget": S0-03's wall for 300 iterations, scaled linearly. A PREDICTION
#: for --dry-plan only; no rule reads it.
BUDGET_NOTE = "prediction, not premise: linear in iterations from S0-03's wall clock"


class StepMismatch(RuntimeError):
    """A checkpoint's stored step is not the label it is measured under."""


# --------------------------------------------------------------------------- #
# one training run, in its own child process
# --------------------------------------------------------------------------- #


def single(seed: int, iters: int, out_dir: Path, *, ckpt_every: int = CKPT_EVERY):
    """S0-03's ``single()`` with one change: ``ckpt_every`` (and ``iters``). Every
    other argument is read off S0-03's ``CONFIG``, never retyped."""
    from rsr.train.loop import train

    C = S003.CONFIG
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
        ckpt_every=ckpt_every,
        policy_name=C["policy"],
        masked_loss=C["masked_loss"],
        srep_norm_reg_weight=C["srep_norm_reg_weight"],
    )
    r["seed"] = seed
    r["torch_num_threads"] = torch.get_num_threads()
    return r


def child_argv(seed: int, iters: int, out_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single",
        "--seed",
        str(seed),
        "--iters",
        str(iters),
        "--out-dir",
        str(out_dir),
    ]


def child_env() -> dict[str, str]:
    """S0-03's ``main()`` env, exactly: OMP and MKL at ``THREADS_PER_SEED``."""
    return {
        **os.environ,
        "OMP_NUM_THREADS": str(THREADS_PER_SEED),
        "MKL_NUM_THREADS": str(THREADS_PER_SEED),
    }


class Child:
    """One training child. ``poll``/``terminate``/``kill``/``wait`` as Popen's."""

    def __init__(self, seed: int, out_dir: Path, iters: int = ITERS) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.argv = child_argv(seed, iters, out_dir)
        self._log = open(out_dir / "train.log", "w")  # noqa: SIM115 -- closed in wait
        self._p = subprocess.Popen(
            self.argv, cwd=ROOT, stdout=self._log, stderr=self._log, env=child_env()
        )

    def poll(self):
        return self._p.poll()

    def terminate(self) -> None:
        self._p.terminate()

    def kill(self) -> None:
        self._p.kill()

    def wait(self, timeout: float | None = None):
        rc = self._p.wait(timeout=timeout)
        self._log.close()
        return rc


# --------------------------------------------------------------------------- #
# checkpoints and the measurement (in the parent)
# --------------------------------------------------------------------------- #


def ckpt_path(seed_dir: Path, step: int) -> Path:
    """The loop's own filename (`src/rsr/train/loop.py:497`)."""
    return seed_dir / f"ckpt-{step:06d}.pt"


def stored_step(ckpt: Path) -> int:
    """The step the training loop stored inside the checkpoint -- not its name."""
    return int(ck.load(ckpt, restore_rng=False)["step"])


def summarise_decisive(m: dict) -> dict:
    """The decisive run's secondary readouts, per batch: ratio, random_ratio and
    the cross-row cosine (trained and decoy), plus its controls."""
    out = {}
    for batch, x in m.items():
        ok, why = DECISIVE.controls_pass(x)
        out[batch] = {
            "ratio": DECISIVE.ratio(x),
            "random_ratio": DECISIVE.random_ratio(x),
            "cosine_trained": x["cosine_trained"]["mean_offdiag_cosine"],
            "cosine_decoy": x["cosine_decoy"]["mean_offdiag_cosine"],
            "controls_pass": ok,
            "controls_failed": why,
            "random_valid": DECISIVE.random_valid(x),
        }
    return out


def measure_checkpoint(seed_dir: Path, seed: int, label: int) -> dict:
    """Every readout for one (seed, checkpoint), read from THAT checkpoint. The
    stored step must equal the label (brief bar, mutation 3)."""
    ckpt = ckpt_path(seed_dir, label)
    step = stored_step(ckpt)
    if step != label:
        raise StepMismatch(
            f"seed {seed}: {ckpt.name} stores step {step}, but is measured as "
            f"checkpoint {label}"
        )
    s003 = S003.measure(ckpt, seed, DEVICE)
    dec = DECISIVE.measure(ckpt, seed, decoy_ckpt=None)
    return {
        "checkpoint": str(ckpt),
        "stored_step": step,
        "s003": s003,
        "decisive": summarise_decisive(dec),
    }


# --------------------------------------------------------------------------- #
# the reproduction control
# --------------------------------------------------------------------------- #


def s003_reference(path: Path | None = None) -> dict[int, float]:
    """S0-03's per-seed held-out live answer NLL at 2..M, by seed. The row is looked
    up by key; its samples are in S0-03's ``seeds_actually_run`` order."""
    path = path or ROOT / S003_LEDGER
    doc = json.loads(Path(path).read_text())
    rows = [r for r in doc["rows"] if r.get("key") == REPRO_KEY]
    if len(rows) != 1:
        raise ValueError(f"{path}: {len(rows)} rows keyed {REPRO_KEY!r}, need 1")
    seeds = doc["seeds_actually_run"]
    samples = rows[0]["samples"]
    if sorted(seeds) != SEEDS or len(samples) != len(seeds):
        raise ValueError(f"{path}: seeds {seeds} / {len(samples)} samples")
    return {int(s): float(x) for s, x in zip(seeds, samples, strict=True)}


def reproduction_control(seed: int, s003_measured: dict, reference: dict) -> dict:
    """PREREG "The reproduction control": |measured - S0-03| <= REPRO_TOL."""
    got = float(s003_measured["heldout"]["live"]["gap_2_to_M"]["answer_nll"])
    ref = float(reference[seed])
    diff = abs(got - ref)
    return {
        "seed": seed,
        "measured": got,
        "reference": ref,
        "abs_diff": diff,
        "tolerance": REPRO_TOL,
        "ok": diff <= REPRO_TOL,
    }


# --------------------------------------------------------------------------- #
# the rule
# --------------------------------------------------------------------------- #


def s003_decide(per_seed: dict[int, dict]) -> dict:
    """S0-03's ``decide()``, unchanged, fed the HELD-OUT population only. Its rule
    reads ``per[s]["heldout"]``; nothing else is handed to it."""
    feed = {s: {"heldout": m["heldout"]} for s, m in per_seed.items()}
    return S003.decide(feed)


def decisions(per: dict[int, dict[int, dict]]) -> dict[int, dict]:
    """``decide()`` at every checkpoint measured on every seed."""
    return {
        c: s003_decide({s: per[s][c]["s003"] for s in SEEDS})
        for c in CHECKPOINTS
        if all(c in per.get(s, {}) for s in SEEDS)
    }


def verdict(per: dict, control: dict, stopped: str | None) -> dict:
    """PREREG "Primary readout and decision rule", row for row.

    The reproduction control is checked first and overrides every row below it.
    """
    failed = sorted(s for s, c in control.items() if not c["ok"])
    if failed:
        return {
            "outcome": "inconclusive",
            "detail": (
                "the ckpt-300 reproduction control failed on "
                + ", ".join(f"seed{s}" for s in failed)
                + ": this is not S0-03's configuration, nothing past it is read"
            ),
            "falsified_at": [],
            "exit": Exit.DID_NOT_RUN,
        }
    if sorted(control) != SEEDS:
        return {
            "outcome": "inconclusive",
            "detail": f"the reproduction control never ran on every seed ({stopped})",
            "falsified_at": [],
            "exit": Exit.DID_NOT_RUN,
        }
    dec = decisions(per)
    hit = [
        c
        for c, d in dec.items()
        if d["bar1_zeroed_not_below_chance"] and d["h4_retrieval_shown"]
    ]
    if hit:
        return {
            "outcome": "falsified",
            "detail": (
                "bar 1 and H4 hold on every seed at "
                + ", ".join(f"ckpt{c}" for c in hit)
                + ": longer training makes memory help answers"
            ),
            "falsified_at": hit,
            "exit": Exit.OK,
        }
    reached = sorted(dec) == sorted(CHECKPOINTS)
    if reached and all(d["bar1_zeroed_not_below_chance"] for d in dec.values()):
        return {
            "outcome": "survived",
            "detail": (
                "ckpt3000 reached; bar 1 holds and H4 fails at every checkpoint: "
                "training length alone does not produce retrieval by ckpt3000"
            ),
            "falsified_at": [],
            "exit": Exit.OK,
        }
    if reached:
        bad = [c for c, d in dec.items() if not d["bar1_zeroed_not_below_chance"]]
        return {
            "outcome": "inconclusive",
            "detail": (
                "bar 1 fails at "
                + ", ".join(f"ckpt{c}" for c in bad)
                + ", where H4 would be read; no checkpoint shows retrieval"
            ),
            "falsified_at": [],
            "exit": Exit.OK,
        }
    return {
        "outcome": "inconclusive",
        "detail": (
            f"the run stopped ({stopped}) before ckpt3000 without retrieval shown; "
            "measured on every seed: "
            + (", ".join(f"ckpt{c}" for c in sorted(dec)) or "none")
        ),
        "falsified_at": [],
        "exit": Exit.DID_NOT_RUN,
    }


# --------------------------------------------------------------------------- #
# the parent loop
# --------------------------------------------------------------------------- #


def run_curve(
    root: Path,
    *,
    spawn: Callable[[int, Path], Any],
    measure_fn: Callable[[Path, int, int], dict],
    reference: dict[int, float],
    deadline_s: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    poll_s: float = POLL_S,
    log: Callable[[str], None] = lambda s: print(s, flush=True),
) -> dict:
    """Train the three seeds in parallel children; measure in THIS process each
    pre-registered checkpoint as it appears. Returns what happened; decides
    nothing."""
    t0 = clock()
    dirs = {s: root / f"seed{s}" for s in SEEDS}
    procs = {s: spawn(s, dirs[s]) for s in SEEDS}
    pending = {s: list(CHECKPOINTS) for s in SEEDS}
    per: dict[int, dict[int, dict]] = {s: {} for s in SEEDS}
    control: dict[int, dict] = {}

    def sweep() -> str | None:
        """Measure every pending checkpoint already on disk, lowest label first
        across seeds -- so every seed's control is read before any later
        checkpoint of any seed."""
        while True:
            ready = [
                (pending[s][0], s)
                for s in SEEDS
                if pending[s] and ckpt_path(dirs[s], pending[s][0]).exists()
            ]
            if not ready:
                return None
            label, s = min(ready)
            pending[s].pop(0)
            per[s][label] = measure_fn(dirs[s], s, label)
            log(f"  seed {s}: measured ckpt{label}")
            if label == CONTROL_CHECKPOINT:
                control[s] = reproduction_control(s, per[s][label]["s003"], reference)
                log(f"  seed {s}: reproduction control {control[s]}")
                if not control[s]["ok"]:
                    return "reproduction control failed"

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
    while True:
        stopped = sweep()
        if stopped:
            stop_all()
            break
        if not any(pending.values()):
            break
        if clock() - t0 >= deadline_s:
            stop_all()
            stopped = sweep() or "deadline"
            break
        dead = [s for s in SEEDS if pending[s] and procs[s].poll() is not None]
        if dead:
            # A child that exited normally has already written every checkpoint.
            stopped = sweep()
            failed = [s for s in SEEDS if pending[s] and procs[s].poll() is not None]
            if stopped or failed:
                stop_all()
                stopped = stopped or "training child exited early"
                break
            continue
        sleep(poll_s)
    exit_codes = {s: p.wait() for s, p in procs.items()}
    return {
        "per": per,
        "control": control,
        "stopped": stopped,
        "failed_seeds": failed,
        "exit_codes": exit_codes,
        "argv": {s: getattr(p, "argv", None) for s, p in procs.items()},
        "checkpoints_measured": {s: sorted(per[s]) for s in SEEDS},
        "elapsed_s": clock() - t0,
    }


# --------------------------------------------------------------------------- #
# provenance and the ledger
# --------------------------------------------------------------------------- #


def corpus_sha256(seed: int) -> str:
    """sha256 of the seed's training documents, as ``train()`` generates them."""
    from rsr.data.synthetic import SyntheticConfig, generate

    S = S003.CONFIG["steps_per_stream"]
    docs = generate(SyntheticConfig(sentences_per_document=S, seed=seed))
    blob = json.dumps([dataclasses.asdict(d) for d in docs], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def manifest(parent_threads: int) -> dict:
    return {
        **S003.CONFIG,
        "iters": ITERS,
        "ckpt_every": CKPT_EVERY,
        "checkpoints": list(CHECKPOINTS),
        "control_checkpoint": CONTROL_CHECKPOINT,
        "device": DEVICE,
        "seeds": SEEDS,
        "threads_per_seed": THREADS_PER_SEED,
        "seeds_trained_in_parallel": True,
        "parent_torch_num_threads": parent_threads,
        "parent_threads_note": (
            "the measuring parent's thread count is not pinned, as S0-03's was not"
        ),
        "reproduction_tolerance": REPRO_TOL,
        "reproduction_reference": f"{S003_LEDGER} rows[key={REPRO_KEY}].samples",
        "wall_deadline_hours": DEADLINE_HOURS,
        "margin": MARGIN,
        "chance_ln16": CHANCE,
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "prereg": PREREG,
        "falsifier": FALSIFIER,
        "expected": EXPECTED,
        "instrument": {
            "primary": "experiments/s0-03-rewardable-corpus/run.py measure + decide",
            "secondary": "experiments/decisive-shuffle/run.py measure, decoy_ckpt=None",
        },
    }


def _stat(led, key: str, xs: list, *, how: str) -> None:
    """A statistic over the three seeds, or a note when a seed is missing or a
    value is None: an incomplete list is not a statistic."""
    if len(xs) != len(SEEDS) or any(x is None for x in xs):
        led.note(key, xs, how=how + " [incomplete or contains None: not a statistic]")
        return
    led.stat(key, xs, how=how)


def write_rows(led, run: dict) -> dict:
    """Every per-checkpoint readout, then the control, then the verdict."""
    per, control = run["per"], run["control"]
    how = (
        f"{S003.EXPERIMENT}::measure (imported) on the checkpoint whose stored step "
        f"equals its label, measured in the parent; per-seed samples in seed order"
    )
    for c in CHECKPOINTS:
        seeds = [s for s in SEEDS if c in per[s]]
        if not seeds:
            continue
        p = f"ckpt{c}"
        led.note(f"{p}.seeds_measured", seeds, how="run_curve")
        led.note(
            f"{p}.stored_step",
            {f"seed{s}": per[s][c]["stored_step"] for s in seeds},
            how="rsr.train.checkpoint.load(...)['step'], checked == label",
        )
        for ds in ("heldout", "train"):
            for cond in S003.CONDITIONS:
                for b in S003.BUCKETS:
                    for key in ("answer_nll", "answer_acc", "answer_nll_over_16"):
                        _stat(
                            led,
                            f"{p}.{ds}.{cond}.{b}.{key}",
                            [per[s][c]["s003"][ds][cond][b][key] for s in seeds],
                            how=f"{how}; set {ds}; condition {cond}; bucket {b}",
                        )
                _stat(
                    led,
                    f"{p}.{ds}.{cond}.real_token_nll",
                    [per[s][c]["s003"][ds][cond]["real_token_nll"] for s in seeds],
                    how=f"{how}; set {ds}; condition {cond}; all real targets",
                )
            for b in ("all", "gap_eq_1", "gap_ge_2", "gap_2_to_M", "gap_gt_M"):
                v = [
                    (
                        per[s][c]["s003"][ds]["slots_zeroed"][b]["answer_nll"]
                        - per[s][c]["s003"][ds]["live"][b]["answer_nll"]
                    )
                    if per[s][c]["s003"][ds]["live"][b]["answer_nll"] is not None
                    and per[s][c]["s003"][ds]["slots_zeroed"][b]["answer_nll"] is not None
                    else None
                    for s in seeds
                ]
                _stat(
                    led,
                    f"{p}.{ds}.slots_zeroed_minus_live.{b}.answer_nll",
                    v,
                    how=f"per seed: {p}.{ds}.slots_zeroed - live, bucket {b}",
                )
        for batch in ("train", "heldout"):
            for key in ("ratio", "random_ratio", "cosine_trained", "cosine_decoy"):
                _stat(
                    led,
                    f"{p}.decisive.{batch}.{key}",
                    [per[s][c]["decisive"][batch][key] for s in seeds],
                    how=(
                        f"{DECISIVE.EXPERIMENT}::measure (imported), decoy_ckpt=None "
                        f"(untrained same-seed decoy); batch {batch}; field {key}"
                    ),
                )
            led.note(
                f"{p}.decisive.{batch}.controls",
                {
                    f"seed{s}": {
                        k: per[s][c]["decisive"][batch][k]
                        for k in ("controls_pass", "controls_failed", "random_valid")
                    }
                    for s in seeds
                },
                how=f"{DECISIVE.EXPERIMENT}::controls_pass / random_valid",
            )
        led.note(
            f"{p}.memory_gates",
            {f"seed{s}": per[s][c]["s003"]["memory_gates"] for s in seeds},
            how="trained checkpoint's memory_gate per C block",
        )

    for f in ("measured", "reference", "abs_diff", "ok"):
        led.note(
            f"reproduction_control.{f}",
            [control[s][f] if s in control else None for s in SEEDS],
            how=(
                f"ckpt{CONTROL_CHECKPOINT} heldout.live.gap_2_to_M.answer_nll vs "
                f"{S003_LEDGER} {REPRO_KEY} samples; seed order {SEEDS}"
            ),
        )
    led.note("reproduction_tolerance", REPRO_TOL, how="PREREG thresholds")

    dec = decisions(per)
    for c, d in dec.items():
        led.note(f"ckpt{c}.s003_decision", d, how=f"{S003.EXPERIMENT}::decide, heldout")
        led.note(
            f"ckpt{c}.s003_outcome",
            d["outcome"],
            how="S0-03's OWN label, about S0-03's falsifier -- not this verdict",
        )
        led.note(f"ckpt{c}.bar1", d["bar1_zeroed_not_below_chance"], how="decide()")
        led.note(f"ckpt{c}.h4", d["h4_retrieval_shown"], how="decide()")
    v = verdict(per, control, run["stopped"])
    led.note("checkpoints_preregistered", list(CHECKPOINTS), how="PREREG thresholds")
    led.note(
        "checkpoints_measured",
        {f"seed{s}": run["checkpoints_measured"][s] for s in SEEDS},
        how="run_curve",
    )
    led.note("checkpoints_measured_on_every_seed", sorted(dec), how="run_curve + decide")
    led.note("falsified_at_checkpoints", v["falsified_at"], how="verdict()")
    led.note("stopped_because", run["stopped"], how="run_curve")
    led.note("margin", MARGIN, how="S0-03 run.py MARGIN, imported")
    led.note("chance_ln16", CHANCE, how="S0-03 run.py CHANCE, imported")
    led.note("wall_deadline_hours", DEADLINE_HOURS, how="PREREG thresholds")
    led.note("elapsed_s", run["elapsed_s"], how="parent monotonic clock")
    led.note("expected", EXPECTED, how="manifest, frozen before the first child")
    led.verdict(falsifier=FALSIFIER, outcome=v["outcome"], detail=v["detail"])
    return v


def execute(
    led,
    root: Path,
    *,
    spawn: Callable[[int, Path], Any],
    measure_fn: Callable[[Path, int, int], dict],
    reference: dict[int, float],
    deadline_s: float,
    results_path: Path | None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Exit:
    """The whole run after the manifest: train, measure, decide, write, render."""
    run = run_curve(
        root,
        spawn=spawn,
        measure_fn=measure_fn,
        reference=reference,
        deadline_s=deadline_s,
        clock=clock,
        sleep=sleep,
    )
    for s in SEEDS:
        argv = run["argv"].get(s)
        if argv:
            rel = " ".join(
                [
                    "uv run python",
                    EXPERIMENT,
                    *[
                        str(Path(x).relative_to(ROOT)) if x.startswith(str(ROOT)) else x
                        for x in argv[2:]
                    ],
                ]
            )
            led.command(rel, exit_code=run["exit_codes"][s], note=f"seed {s} training")
        info = root / f"seed{s}" / "result.json"
        if info.exists():
            r = json.loads(info.read_text())
            led.note(
                f"seed{s}.training",
                {
                    k: r.get(k)
                    for k in (
                        "config_hash",
                        "steps_done",
                        "torch_num_threads",
                        "run_id",
                    )
                },
                how="the child's result.json (train() return + torch threads)",
            )
    done = [s for s in SEEDS if ITERS in run["per"][s]]
    led.run_meta(
        device=DEVICE,
        seeds_actually_run=SEEDS,
        steps_requested=ITERS,
        steps_done=max((max(run["per"][s]) for s in SEEDS if run["per"][s]), default=0),
    )
    if run["failed_seeds"]:
        led.status("crashed")
    else:
        led.status("ok" if len(done) == len(SEEDS) else "partial")
    v = write_rows(led, run)
    (root / "raw.json").write_text(json.dumps(run, indent=2, default=str) + "\n")
    path = led.write()
    if results_path is not None:
        results_path.write_text(render_results(json.loads(path.read_text())))
    print(f"ledger: {path}  verdict={v['outcome']}  exit={int(v['exit'])}")
    return v["exit"]


# --------------------------------------------------------------------------- #
# RESULTS.md, rendered from the ledger
# --------------------------------------------------------------------------- #


def _rows(doc: dict) -> dict:
    return {r["key"]: r for r in doc.get("rows", [])}


def _val(rows: dict, key: str) -> Any:
    r = rows.get(key)
    if r is None:
        return None
    return r["value"] if "value" in r else r.get("samples")


def _fmt(xs: Any) -> str:
    if xs is None:
        return "—"
    if isinstance(xs, list):
        return ", ".join(_fmt(x) for x in xs)
    if isinstance(xs, bool):
        return "yes" if xs else "no"
    if isinstance(xs, float):
        return f"{xs:.4f}"
    return str(xs)


def render_results(doc: dict) -> str:
    """Every number beside its ledger key, so `render_scoreboard.py --audit`
    backs it (done_when). No number is typed here."""
    rows = _rows(doc)
    prov = doc.get("provenance", {})
    v = doc.get("verdict") or {}
    rid = doc.get("run_id")
    shas = _val(rows, "corpus_sha256") or {}
    out = [
        "# The retrieval curve -- RESULTS",
        "",
        f"<!-- GENERATED by `{EXPERIMENT}` from runs/{rid}/ledger.json. Do not "
        f"edit: regenerate with `--render-results`. -->",
        "",
        f"Command: `uv run python {EXPERIMENT}`. Ledger `status` of run `{rid}`: "
        f"**{doc.get('status')}**.",
        "",
        f"Provenance of run `{rid}`: `provenance.git_sha` `{prov.get('git_sha')}` · "
        f"`device` {doc.get('device')} · `corpus_sha256` "
        + " ".join(f"{k} `{x}`" for k, x in sorted(shas.items()))
        + f" · `seeds_actually_run` {doc.get('seeds_actually_run')}.",
        "",
        f"Falsifier of run `{rid}` (`verdict.falsifier`): **{v.get('falsifier')}**",
        "",
        f"## Verdict of run `{rid}`: `verdict.outcome` **{v.get('outcome')}**",
        "",
        f"Why, for run `{rid}` (`verdict.detail`): **{v.get('detail')}**",
        "",
        f"`falsified_at_checkpoints` {_val(rows, 'falsified_at_checkpoints')} "
        f"(empty unless falsified). `checkpoints_measured_on_every_seed` "
        f"{_val(rows, 'checkpoints_measured_on_every_seed')} of "
        f"`checkpoints_preregistered` {_val(rows, 'checkpoints_preregistered')}. "
        f"`stopped_because`: {_val(rows, 'stopped_because') or 'nothing'}.",
        "",
        "Whether the cognitive claim moved: **it cannot move here.** This run asks "
        "only whether TG's memory, under FIFO, can be made to carry answers across "
        "sentences; it says nothing about RSR, eviction policy or Kintsch & van "
        "Dijk (PREREG, *What this does not establish*).",
        "",
        f"Pre-registered expectation (`expected`, frozen in the manifest before any "
        f"child ran): **{_val(rows, 'expected')}**",
        "",
        "## The reproduction control",
        "",
        f"`reproduction_tolerance` {_val(rows, 'reproduction_tolerance')!r} absolute, "
        f"against `{S003_LEDGER}` `{REPRO_KEY}`.",
        "",
        "| seed | `reproduction_control.measured` | `reproduction_control.reference` "
        "| `reproduction_control.abs_diff` | `reproduction_control.ok` |",
        "|---|---|---|---|---|",
    ]
    cols = [
        _val(rows, f"reproduction_control.{f}") or [None] * len(SEEDS)
        for f in ("measured", "reference", "abs_diff", "ok")
    ]
    for i, s in enumerate(SEEDS):
        cells = [
            _fmt(c[i]) if isinstance(c[i], bool) or c[i] is None else repr(c[i])
            for c in cols
        ]
        out.append(f"| seed{s} | " + " | ".join(cells) + " |")
    out += [
        "",
        "## Per checkpoint (held-out unless named; per-seed samples in seed order)",
        "",
        f"`margin` {_val(rows, 'margin')} · `chance_ln16` {_val(rows, 'chance_ln16')}.",
        "",
        "| checkpoint | `bar1` | `h4` | `s003_outcome` (S0-03's own label) "
        "| `heldout.slots_zeroed_minus_live.gap_2_to_M.answer_nll` "
        "| `heldout.live.gap_2_to_M.answer_nll` | `train.live.gap_2_to_M.answer_nll` "
        "| `heldout.live.gap_eq_1.answer_nll` (bos copy, not retrieval) "
        "| `decisive.train.ratio` |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in CHECKPOINTS:
        p = f"ckpt{c}"
        if f"{p}.seeds_measured" not in rows:
            out.append(f"| {p} | not measured | | | | | | | |")
            continue
        cells = [
            _fmt(_val(rows, f"{p}.bar1")),
            _fmt(_val(rows, f"{p}.h4")),
            _fmt(_val(rows, f"{p}.s003_outcome")),
            _fmt(
                _val(rows, f"{p}.heldout.slots_zeroed_minus_live.gap_2_to_M.answer_nll")
            ),
            _fmt(_val(rows, f"{p}.heldout.live.gap_2_to_M.answer_nll")),
            _fmt(_val(rows, f"{p}.train.live.gap_2_to_M.answer_nll")),
            _fmt(_val(rows, f"{p}.heldout.live.gap_eq_1.answer_nll")),
            _fmt(_val(rows, f"{p}.decisive.train.ratio")),
        ]
        out.append(f"| {p} | " + " | ".join(cells) + " |")
    out += [
        "",
        "Every other bucket, condition and the decisive run's `random_ratio` and "
        "cross-row cosine are in the ledger under `ckpt<checkpoint>.`.",
        "",
        "## BRIEF ERRORS",
        "",
        "Written by the executor after the run.",
        "",
    ]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# --dry-plan
# --------------------------------------------------------------------------- #


def dry_plan() -> str:
    ref = ROOT / S003_LEDGER
    doc = json.loads(ref.read_text())
    t = [
        _dt.datetime.strptime(doc[k], "%Y-%m-%dT%H:%M:%SZ")
        for k in ("started_utc", "finished_utc")
    ]
    wall = (t[1] - t[0]).total_seconds()
    pred = wall * ITERS / doc["steps_done"]
    lines = [
        "retrieval-curve dry plan (NOTHING RUN).",
        f"seeds {SEEDS} as parallel children, OMP/MKL threads {THREADS_PER_SEED} each, "
        f"device {DEVICE}",
        f"train to {ITERS} iters, ckpt_every {CKPT_EVERY}; measure in the parent at "
        f"{list(CHECKPOINTS)}",
        f"reproduction control at ckpt{CONTROL_CHECKPOINT}: {REPRO_KEY} within "
        f"{REPRO_TOL} of {S003_LEDGER}: {s003_reference()}",
        f"deadline {DEADLINE_HOURS} h (parent-enforced), poll every {POLL_S} s",
        f"PREDICTED wall ~{pred / 3600:.1f} h ({BUDGET_NOTE}; S0-03 took "
        f"{wall / 60:.1f} min for {doc['steps_done']} iters)",
        f"child argv: {child_argv(0, ITERS, ROOT / 'runs' / RUN_ID / 'seed0')}",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> Exit:
    ap = ArgumentParser(prog="retrieval-curve")
    ap.add_argument("--dry-plan", action="store_true", help="print the plan; run nothing")
    ap.add_argument(
        "--render-results", action="store_true", help="RESULTS.md from the ledger"
    )
    ap.add_argument("--run-id", default=RUN_ID)
    # internal: one training child
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args(argv)

    if a.single:
        if a.out_dir is None:
            refuse(Exit.DID_NOT_RUN, "--single needs --out-dir")
        a.out_dir.mkdir(parents=True, exist_ok=True)
        r = single(a.seed, a.iters, a.out_dir)
        (a.out_dir / "result.json").write_text(
            json.dumps(r, indent=2, default=str) + "\n"
        )
        return Exit.OK

    if a.dry_plan:
        print(dry_plan())
        return Exit.OK

    root = ROOT / "runs" / a.run_id
    results = ROOT / "experiments" / "retrieval-curve" / "RESULTS.md"
    if a.render_results:
        path = root / "ledger.json"
        if not path.exists():
            refuse(Exit.DID_NOT_RUN, f"{path} is absent: the curve has not run")
        results.write_text(render_results(json.loads(path.read_text())))
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
    ref_doc = json.loads((ROOT / S003_LEDGER).read_text())
    if ref_doc.get("device") != DEVICE:
        refuse(Exit.DID_NOT_RUN, f"{S003_LEDGER} device is {ref_doc.get('device')!r}")
    reference = s003_reference()

    led = ledger_mod.Ledger(
        a.run_id,
        question=(
            "Does training S0-03's configuration to 3000 iterations make the working "
            "memory help answers at 2 <= gap <= M on held-out documents?"
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
        {f"seed{s}": corpus_sha256(s) for s in SEEDS},
        how="sha256 of json(asdict(doc)) over the seed's training documents",
    )
    led.note("reproduction_reference", reference, how=f"{S003_LEDGER} {REPRO_KEY}")
    return execute(
        led,
        root,
        spawn=lambda s, d: Child(s, d),
        measure_fn=measure_checkpoint,
        reference=reference,
        deadline_s=DEADLINE_HOURS * 3600.0,
        results_path=results,
    )


if __name__ == "__main__":
    run_main(main)
