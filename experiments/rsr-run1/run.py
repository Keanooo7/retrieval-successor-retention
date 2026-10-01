"""RSR run 1 -- the measurement harness (PREREG `experiments/rsr-run1/PREREG.md`, T3).

Pre-registration: every layer of the PREREG -- base (`7c58777`), Amendment 1 (`6a4c5a8`),
2 (`7fcd3f9`), 3 (`58a9216`), 3b (`7196a52`) and 3c (`34f495f`), reviewed at `787875b`,
`c02cae7`, `7f7e2ad`, `3ef97c8`, `33ce631` and `de8f813` (``PREREG_COMMITS`` /
``REVIEW_COMMITS``; Amendment 4, the owner values, is added there when it lands).
**Descriptive. No gate, no pass/fail threshold.** The controls decide only whether the
run is readable; every contrast is read only as "CI above 0", "CI below 0" or "CI
includes 0".

The launch is NOT here: it is `experiments/rsr-run1/launch_run1.zsh`, committed
byte-identical to the launcher under ``LAUNCHER_HEADING`` (A3c.1, version 5; a test pins
it), run inside tmux (A3.8). This file:

* ``--preflight`` (A1.2 step 2, A1.8 m-3): C0 for seeds 0-2 before any model loads; the
  launcher is byte-identical to the PREREG's; the substrate checkpoints hash to their T0
  values; ``RSR_ORCH_ROOT`` / ``OUT`` / ``JOB_SUFFIX`` recorded, and a real launch refused
  unless the orchestrator root is the shared one and the dry-run variables are unset
  (review 7fcd3f91ca8f MINOR-A2-3).
* ``measure-seed --seed s`` (A1.6: one process per seed): C0-C10, the held-out harness on
  EVAL (eval mode, ``no_grad``, phi from ``__head__``, ``set_train_step``, observe off)
  for
  the three arms, the oracle (gamma_o = 0.9, A2.2; scoped controls C9 / C9b), LRU (observe
  on, A1.1 / T11), random x5 and the slots-zeroed floor under each arm; the intermediate
  phi snapshots (A1.8 m-9, DIAGNOSTIC); per-document answer vectors; C4 through
  fresh-stream's own ``measure_checkpoint``, imported unchanged.
* ``finalize``: the ONE headline (psiC - FIFO, all-query held-out accuracy) per seed and
  across seeds (points, mean, sd, sign count; A1.4), everything else labelled SECONDARY
  or DIAGNOSTIC; the ledger; ``RESULTS.md``.
* ``time-batch`` (T5) and ``probe-oracle`` (T7): build-time measurements only.

🔴 Every call into `feat/rsr-run1-ready` code goes through an ADAPTER (search
``ADAPTER``); they are the list to re-check on rebase.

Never: trains anything; calls `rsr.constants.record()`; fills §15; reads `loss` (the
analysis reads ``loss_lm``); names the rescaled target anything stronger; makes a scaling
claim; scores the author's expectation.

Exit codes (`rsr.exit_codes`): 0 every control passed in every seed · 3 did not run, or
a control failed (numbers written, marked not read) · 5 an arm measured INERT.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import gzip
import hashlib
import importlib.util
import inspect
import itertools
import json
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rsr.baselines.fifo import FIFOPolicy  # noqa: E402
from rsr.baselines.lru import LRUPolicy  # noqa: E402
from rsr.baselines.oracle import OraclePolicy  # noqa: E402
from rsr.data.synthetic import (  # noqa: E402
    SyntheticConfig,
    _generate_document,
    discounted_demand,
)
from rsr.exit_codes import ArgumentParser, Exit, run_main  # noqa: E402
from rsr.metrics.headroom import simulate  # noqa: E402
from rsr.model.tg import TGConfig, TGModel  # noqa: E402
from rsr.model.tg.policy_loop import run_policy_loop  # noqa: E402
from rsr.retention.instrumentation import rank_shift  # noqa: E402
from rsr.train import checkpoint as ck  # noqa: E402
from rsr.train import loop  # noqa: E402

EXPERIMENT = "experiments/rsr-run1/run.py"
LAUNCHER = "experiments/rsr-run1/launch_run1.zsh"
RUN_ID = "rsr-run1"
PREREG = "experiments/rsr-run1/PREREG.md"
#: Every PREREG layer that governs the run (review 6267fd645698 MINOR-6). Amendment 4 (the
#: owner values) is added here when it lands.
PREREG_COMMITS = {
    "base": "7c58777",
    "A1": "6a4c5a8",
    "A2": "7fcd3f9",
    "A3": "58a9216",
    "A3b": "7196a52",
    "A3c": "34f495f",
}
REVIEW_COMMITS = ("787875b", "c02cae7", "7f7e2ad", "3ef97c8", "33ce631", "de8f813")


def _load(name: str, rel: str):
    """Import an experiment script by path. Imported, never copied."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # its dataclasses resolve their module through this
    spec.loader.exec_module(mod)
    return mod


@functools.cache
def B2():
    """B2's harness (PREREG T3): `random_policy` (§7 seeding), `Scripted` (§9.1 replay)
    and the §9.2 bootstrap draw. Its `run_arm` is NOT used: it calls `run_policy_loop`
    without `eval_capture`, which an RSR policy needs (T3)."""
    return _load("_rsr_run1_b2", "experiments/b2-psi-probe/run.py")


@functools.cache
def FS():
    """fresh-stream (C4, T8): `measure_checkpoint` and `r_quantity`, unchanged."""
    return _load("_rsr_run1_fresh_stream", "experiments/fresh-stream/run.py")


# --------------------------------------------------------------------------- #
# 🔒 PREREG, transcribed. Changing any of these is changing the PREREG.
# --------------------------------------------------------------------------- #

ARMS = ("psiC", "fifo", "age")  # §5
RSR_ARMS = ("psiC", "age")
SEEDS = (0, 1, 2)  # A1.4
POOL = (993000, 993256)  # §2
EVAL = (995000, 997048)  # §2, §7.1
VOCAB_DOCUMENTS = 64  # §3
GEN_SEED_K = 1_000_003  # synthetic.py: per-document seed = seed * K + doc_id
#: §2 "The ranges checked were".
USED_RANGES: dict[str, tuple[int, int]] = {
    "armB_train_vocab_probe": (0, 64),
    "armB_stream_1000_2999": (20160, 52160),
    "fresh_stream_stream": (4160, 52160),
    "H64": (4096, 4160),
    "S003_heldout": (64, 128),
    "corpus_size": (0, 4096),
    "U": (64, 1088),
    "fresh_escape": (52160, 148160),
    "B5_continuation_cap": (52160, 884160),
    "E0d_a": (262144, 263168),
    "E0d_b": (900000, 920000),
    "B2_FIT_TRAIN": (920000, 936000),
    "B2_FIT_VAL": (936000, 937024),
    "B2_EVAL": (940000, 980000),
    "bakeoff_EVAL_NB": (990000, 992048),
}
BATCH = 16
D_MODEL = 128
S_STEPS = 48
M_SLOTS = 16
LR = 1e-3
BEAT_EVERY = 1
DEVICE = "cpu"  # §6
#: §3 and A1.4: one map for every seed.
EXPECTED_VOCAB_SHA256 = "afbaf994e9d1f617fdc2e68579bf5f55622ee17aee056d33d8667d218d245031"
_R = "/Users/keanooo7/retrieval-successor-retention"
_B = f"{_R}/.worktrees/fresh-stream/runs/fresh-stream/B"
#: A1.4's substrate table (and the launcher's SHA[...] lines).
SUBSTRATE: dict[int, dict[str, str]] = {
    s: {
        "checkpoint": f"{_B}/seed{s}/ckpt-003000.pt",
        "sha256": sha,
        "manifest": "/Users/keanooo7/rsr-substrate/2026-09-27/MANIFEST.sha256",
        "root": _R,
        "source_config": f"{_B}/seed{s}/result.json",
    }
    for s, sha in (
        (0, "0ee3f8a69b507d927c631eb85116e1cd9739ee4472c44eb7f145d739a118da60"),
        (1, "dadd1e08a3849c1211c8479394df4060f91cd2a6e188b97701b783f2068a3da8"),
        (2, "b507ebc573a54316f9d3c20337f49e664bb9c414388af925c5a8e87efd6b63b8"),
    )
}
MANIFEST_SHA256 = "33dd493d38067e88663b76e9b009e30211c99f75a8ed896a3808486f02fa565f"
#: C4: fresh-stream's ledger row, read by key and seed position (never retyped). A1.4's
#: table gives every seed's value; they are the cross-check that the row read is it.
C4_LEDGER = f"{_R}/.worktrees/fresh-stream/runs/fresh-stream/ledger.json"
C4_KEY = "B.ckpt3000.R_quantity"
C4_PREREG_VALUE = {0: 0.8597733527421951, 1: 0.8531856164336205, 2: 0.879943523555994}
C4_TOL = 1e-6
C4_N = 64  # measure_checkpoint's `n`: arm B's vocabulary documents
N_BOOT = 2000
BOOT_SEED_BASE = 20260927  # B2 PREREG §9.2's convention (front matter)
N_RANDOM = 5
ORACLE_GAMMA = 0.9  # A2.2: a construction parameter, decoupled from O1
C9_N_DOCS = 16  # A2.2: the first 16 EVAL documents, [995000, 995016)
C9_GAMMAS = (0.5, 0.9)
T7_GAMMAS = (0.01, 0.1, 0.5, 0.9, 0.97, 0.999)
DELTA_BLOCK = 500  # §7.3
FIRST_SCORED_STEPS = 500  # A1.8 m-2
O5_VALUES = ("eval", "train")  # A2.1
#: The shared orchestrator root (lanes.orch_root() with RSR_ORCH_ROOT unset: the main
#: checkout). A real launch's job records live under it.
SHARED_ORCH_ROOT = _R
LAUNCH_ENV_VARS = ("RSR_ORCH_ROOT", "OUT", "JOB_SUFFIX")  # A3c.1; MINOR-A2-3
#: A3 (not fixed by the PREREG): the held-out batch. Every arm shares it.
EVAL_BATCH = 16
HEADLINE = "acc.all.psiC_minus_fifo"  # A1.4
QUESTION = (
    "On fresh-stream arm B's ckpt-003000 with the transformer frozen, when psi_hat is "
    "trained online by L_MC on a fixed pool of 256 documents arm B never saw, how do "
    "its evictions and the frozen model's held-out answer accuracy and LM loss compare "
    "with FIFO and with a learned age-only head, arms that differ from it only in the "
    "eviction rule?"
)
FALSIFIER = (
    "NONE gated. Falsifier 1 (vacuity, spec §2 / §7.1) is described, never tested: no "
    "threshold applies to this run (PREREG front matter)."
)
EXPECTED = "PREREG §7.6 (author's expectation), quoted verbatim in RESULTS.md."
CONTRASTS = {
    "psiC_minus_fifo": ("psiC", "fifo"),
    "age_minus_fifo": ("age", "fifo"),
    "psiC_minus_age": ("psiC", "age"),
    "psiC_minus_lru": ("psiC", "lru"),  # A1.1
    "age_minus_lru": ("age", "lru"),
    "lru_minus_fifo": ("lru", "fifo"),  # A1.1: LRU's distance from FIFO
}
BUCKET_NAMES = ("all", "gap_eq_1", "gap_ge_2", "gap_2_to_M", "gap_gt_M")  # §7.1


def buckets_for(m: int) -> dict:
    """S0-03's bucket definitions at memory size ``m`` (pinned to S0-03's at 16)."""
    return {
        "all": lambda g: g >= 1,
        "gap_eq_1": lambda g: g == 1,
        "gap_ge_2": lambda g: g >= 2,
        "gap_2_to_M": lambda g: (g >= 2) & (g <= m),
        "gap_gt_M": lambda g: g > m,
    }


#: Launcher v3's recorded rule (A3.7), verbatim.
CKPT_RULE = "STEPS/3 if STEPS>0 and STEPS%3==0 else STEPS (A1.8 m-9)"


def derive_ckpt(steps: int) -> int:
    """A1.8 m-9: CKPT = STEPS / 3 when divisible by 3, else STEPS. Never typed."""
    return steps // 3 if steps > 0 and steps % 3 == 0 else steps


class ControlFailure(RuntimeError):
    """A PREREG §8 control failed: inconclusive, exit 3."""


class Refused(RuntimeError):
    """The harness refuses to start: exit 3, nothing run."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Substrate:
    checkpoint: str
    sha256: str
    manifest: str
    root: str
    source_config: str | None = None


@dataclass(frozen=True)
class OwnerValues:
    """A2.8's owner slots -- O1 gamma, O4 phi weight decay, R-1 steps, O5 frozen mode --
    plus the seed list (A1.4: 0, 1, 2; tests use fewer). **No defaults.**"""

    gamma: float
    phi_weight_decay: float
    iters: int
    o5: str
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class Run1Config:
    """What the harness reads that a test may shrink. `prereg_config()` builds the one
    the real run uses; the CLI has no flag that changes it.

    ``prereg_strict=False`` is for the test fixture only (a tree without C-2, launched
    directly, from a dirty worktree): it waives C2(a)'s logged pool ids, the slot job
    records and the clean-launch provenance. RESULTS prints it in bold."""

    seed: int
    substrate: Substrate
    iters: int
    d: int = D_MODEL
    batch: int = BATCH
    steps_per_stream: int = S_STEPS
    memory_slots: int = M_SLOTS
    lr: float = LR
    beat_every: int = BEAT_EVERY
    device: str = DEVICE
    pool: tuple[int, int] = POOL
    eval_range: tuple[int, int] = EVAL
    vocab_documents: int = VOCAB_DOCUMENTS
    eval_batch: int = EVAL_BATCH
    used_ranges: dict = field(default_factory=lambda: dict(USED_RANGES))
    expected_vocab_sha256: str | None = EXPECTED_VOCAB_SHA256
    c4_ledger: str | None = C4_LEDGER
    c4_key: str = C4_KEY
    c4_prereg_value: float | None = None
    n_boot: int = N_BOOT
    orch_root: str = SHARED_ORCH_ROOT
    #: The dry run's job-id suffix (A3c.1 `JOB_SUFFIX`); "" for the run. Set only by
    #: `measure-seed --dry-run`; C10 fails any launch that recorded one.
    job_suffix: str = ""
    #: The file C10 reads the owner's registered values from (A3.7a; `OWNER_TABLE_*`).
    #: Tests point it at a synthetic PREREG; C10 under `prereg_strict` requires PREREG.
    owner_prereg: str = PREREG
    prereg_strict: bool = True

    @property
    def ckpt_every(self) -> int:
        return derive_ckpt(self.iters)

    @property
    def snapshots(self) -> list[int]:
        """Every checkpoint step the arms write: CKPT, 2 CKPT, ..., iters. None at
        STEPS = 0 (CKPT = 0: the only dry run launcher v5 allows, A3c.1)."""
        if self.ckpt_every <= 0:
            return []
        return list(range(self.ckpt_every, self.iters + 1, self.ckpt_every))

    @property
    def pool_length(self) -> int:
        return self.pool[1] - self.pool[0]

    @property
    def steps_per_epoch(self) -> float:
        return self.pool_length / self.batch

    @property
    def t_warm_steps(self) -> float:
        """§6: one epoch = pool / batch steps (31(b)); T_warm_epochs = 1 (FROZEN)."""
        return self.steps_per_epoch

    def eval_ids(self) -> range:
        return range(*self.eval_range)

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def prereg_config(seed: int, iters: int) -> Run1Config:
    if seed not in SUBSTRATE:
        raise Refused(f"seed {seed}: no substrate is registered for it")
    return Run1Config(
        seed=seed,
        substrate=Substrate(**SUBSTRATE[seed]),
        iters=int(iters),
        c4_prereg_value=C4_PREREG_VALUE.get(seed),
    )


def validate_owner_values(ov: OwnerValues, *, dry_run: bool = False) -> None:
    """Refuse anything not ruled, or that the code would not actually run (A2.5).
    ``dry_run`` admits STEPS = 0, which is the only dry run launcher v5 allows (A3c.1;
    review 6267fd645698 MINOR-4); a real run needs STEPS >= 1."""
    for name in ("gamma", "phi_weight_decay", "iters", "o5", "seeds"):
        if getattr(ov, name, None) is None:
            raise Refused(f"owner value {name} is not given (PREREG A2.8): refused")
    g = float(ov.gamma)
    if g != g or not (0.0 <= g < 1.0):
        raise Refused(f"gamma={ov.gamma!r} is outside [0, 1) (spec §4.5)")
    if not ov.seeds or len(set(ov.seeds)) != len(ov.seeds):
        raise Refused(f"seeds={ov.seeds!r}: a non-empty list of distinct seeds")
    if int(ov.iters) != ov.iters or ov.iters < (0 if dry_run else 1):
        raise Refused(
            f"iters={ov.iters!r}: a positive integer run length (R-1)"
            + (", or 0 for a dry run" if dry_run else "")
        )
    if dry_run and ov.iters != 0:
        raise Refused(f"iters={ov.iters!r}: a dry run never trains (A3c.1: STEPS=0)")
    if ov.o5 not in O5_VALUES:
        raise Refused(f"O5 must be one of {O5_VALUES}, got {ov.o5!r} (A2.1)")
    if not frozen_mode_supported(ov.o5):
        raise Refused(f"O5={ov.o5}: this tree has no C-1 (frozen_model_mode) (A2.1)")
    wd = code_phi_weight_decay()
    if float(ov.phi_weight_decay) != wd:
        raise Refused(
            f"phi weight decay {ov.phi_weight_decay!r} was given, but the code pins "
            f"{wd!r} and has no C-3 (A1.9 O4): running it would report a decay that "
            f"did not run"
        )


# --------------------------------------------------------------------------- #
# ADAPTERS -- every call into feat/rsr-run1-ready (built against 211d6da; C-1/C-2
# shapes read from c02b16d, TO-CONFIRM T9/T10).
# --------------------------------------------------------------------------- #

#: ADAPTER (resume). PREREG §11.3: no resume. Stays False until an amendment says so.
RESUME_PERMITTED = False

#: ADAPTER (C-1, T9): what `frozen["frozen_model_mode"]` records for each O5 value
#: (`FROZEN_MODEL_MODE_RECORD` at c02b16d; "train (dropout live)" at 211d6da).
#: Read from the code (`loop.FROZEN_MODEL_MODE_RECORD`, c02b16d); a test pins the two
#: strings, so a change in the code's wording reddens rather than passes silently.
FROZEN_MODE_RECORDED = dict(loop.FROZEN_MODEL_MODE_RECORD)
#: ADAPTER (C-1, T9): the flag the launcher's C1_FLAGS carries for each O5 value.
C1_FLAGS = {"train": (), "eval": ("--frozen-model-mode", "eval")}
#: ADAPTER (C-2, T10): flags for the permutation pass (none at c02b16d: automatic in
#: pool mode) and the per-step field holding the step's pool ids.
C2_FLAGS: tuple[str, ...] = ()
POOL_IDS_FIELD = "pool_doc_ids"
#: ADAPTER: `max_tokens` -- `train()`'s default; the CLI has no flag (§6: 64).
CLI_MAX_TOKENS = 64
#: A3c.1 / A3c.5: every arm passes `--srep-norm-reg-weight 0.0` (arm B's training value).
#: A PREREG constant, not an owner slot: C10 requires it in each job's command (the token,
#: exactly as the launcher spells it) and in each job's recorded config (numerically).
SREP_NORM_REG_WEIGHT = 0.0
SREP_NORM_REG_WEIGHT_ARG = "0.0"


def frozen_mode_supported(o5: str) -> bool:
    """ADAPTER (C-1): the modes `train(frozen_model_mode=...)` accepts (c02b16d:
    `loop.FROZEN_MODEL_MODES`); "train" only on a tree without C-1."""
    if "frozen_model_mode" not in inspect.signature(loop.train).parameters:
        return o5 == "train"
    return o5 in loop.FROZEN_MODEL_MODES


def code_phi_weight_decay() -> float:
    """ADAPTER: the decoupled decay every AdamW group runs (`loop.py:602`)."""
    return float(loop.ADAMW_WEIGHT_DECAY)


def cli_argv(cfg: Run1Config, arm: str, owner: OwnerValues, out_dir: Path) -> list[str]:
    """ADAPTER (flags, T1, T17): one arm's command, exactly as launcher v5 builds it (a
    test compares the two token by token), including A3c.1's `--srep-norm-reg-weight
    0.0` after the C-1 / C-3 flags."""
    s = cfg.substrate
    rsr = [
        "--policy",
        "rsr",
        "--policy-override",
        "b_enabled=false",
        "--policy-override",
        "shadow_enabled=false",
        "--policy-override",
        "nu=0",
        "--policy-override",
        "beta=1",
        "--policy-override",
        f"gamma={owner.gamma!r}",
    ]
    per_arm = {
        "psiC": rsr,
        "fifo": ["--policy", "fifo"],
        "age": [*rsr, "--policy-override", "psi_override=age_only"],
    }
    if arm not in per_arm:
        raise ValueError(f"unknown arm {arm!r}; the arms are {ARMS}")
    return [
        ".venv/bin/python",
        "-m",
        "rsr.train.loop",
        "--d",
        str(cfg.d),
        "--batch",
        str(cfg.batch),
        "--iters",
        str(cfg.iters),
        "--steps-per-stream",
        str(cfg.steps_per_stream),
        "--memory-slots",
        str(cfg.memory_slots),
        "--lr",
        repr(cfg.lr),
        "--seed",
        str(cfg.seed),
        "--beat-every",
        str(cfg.beat_every),
        "--ckpt-every",
        str(cfg.ckpt_every),
        "--device",
        cfg.device,
        "--freeze-transformer",
        *C1_FLAGS[owner.o5],
        "--srep-norm-reg-weight",
        SREP_NORM_REG_WEIGHT_ARG,
        "--pool-offset",
        str(cfg.pool[0]),
        "--pool-length",
        str(cfg.pool_length),
        "--pool-vocab-documents",
        str(cfg.vocab_documents),
        *C2_FLAGS,
        "--warm-start",
        s.checkpoint,
        "--warm-start-sha256",
        s.sha256,
        "--warm-start-manifest",
        s.manifest,
        "--warm-start-root",
        s.root,
        "--warm-start-source-config",
        str(s.source_config),
        *per_arm[arm],
        "--out-dir",
        str(out_dir),
    ]


#: `orchestrator.lanes.THREAD_ENV_VARS`, so a direct (test) launch is capped as a slot is.
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "RSR_TORCH_THREADS",
)


def thread_env(threads: int) -> dict[str, str]:
    env = dict(os.environ)
    env.update(dict.fromkeys(THREAD_ENV_VARS, str(threads)))
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT / "scripts")])
    return env


def read_cli_result(log_path: Path) -> dict | None:
    """ADAPTER: the result dict `rsr.train.loop.main` prints last (indent=2). None if
    absent -- the run raised before printing (A1.2's rc-1 rows)."""
    p = Path(log_path)
    if not p.exists():
        return None
    text = p.read_text(errors="replace")
    dec = json.JSONDecoder()
    found = None
    i = text.find("{\n")
    while i != -1:
        if i == 0 or text[i - 1] == "\n":
            try:
                obj, _end = dec.raw_decode(text, i)
            except json.JSONDecodeError:
                obj = None
            if isinstance(obj, dict) and "run_id" in obj and "liveness" in obj:
                found = obj
        i = text.find("{\n", i + 1)
    return found


def read_heartbeat(out_dir: Path) -> dict[str, Any]:
    """ADAPTER: `heartbeat.jsonl` -> header, beats (by step), footer, crash, last line."""
    path = Path(out_dir) / "heartbeat.jsonl"
    if not path.exists():
        return {
            "exists": False,
            "header": None,
            "beats": [],
            "footer": None,
            "crash": None,
            "last": None,
        }
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    beats = [r for r in rows if r.get("kind") == "beat"]
    return {
        "exists": True,
        "header": next((r for r in rows if r.get("kind") == "header"), None),
        "beats": sorted(beats, key=lambda r: r["step"]),
        "footer": next((r for r in rows if r.get("kind") == "footer"), None),
        "crash": next((r for r in rows if r.get("kind") == "crash"), None),
        "last": rows[-1] if rows else None,
    }


def read_step_records(out_dir: Path) -> list[dict]:
    """ADAPTER: `diagnostics.jsonl`, one record per step (RSR arms always; every pool
    arm at c02b16d, which logs `pool_doc_ids` there). [] if the file is absent."""
    path = Path(out_dir) / loop.DIAGNOSTICS_FILE
    if not path.exists():
        return []
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return sorted(rows, key=lambda r: int(r["step"]))


def step_diagnostics(records: list[dict]) -> list[dict]:
    """ADAPTER (per-step §7.1 diagnostics; the warm/scored split): normalised, one per
    step that carries a `diag`. JSON histogram keys come back as strings; made ints."""
    out = []
    for r in records:
        d = r.get("diag")
        if d is None:
            continue
        warm = d.get("warm") or []
        if len(warm) != 1:
            raise ControlFailure(
                f"step {r['step']} has warm flags {warm!r}; a step is warm or scored"
            )

        def ints(h):
            return {int(k): int(v) for k, v in (h or {}).items()}

        out.append(
            {
                "step": int(r["step"]),
                "warm": bool(warm[0]),
                "shift": ints(d.get("shift_distribution")),
                "victim_age": ints(d.get("victim_age_distribution")),
                "partial_rho": d.get("partial_rho_score_age_given_content"),
                "spearman_rho": d.get("spearman_rho_score_age"),
            }
        )
    return out


def step_pool_ids(records: list[dict]) -> dict[int, list[int]]:
    """ADAPTER (C-2, T10): the pool ids each step read, from the step records."""
    return {
        int(r["step"]): [int(i) for i in r[POOL_IDS_FIELD]]
        for r in records
        if r.get(POOL_IDS_FIELD) is not None
    }


def batch_fingerprint(beat: dict) -> tuple:
    """ADAPTER (C2(b), A1.8 m-4): `(n_answer_targets, n_pad_targets)` of the step's
    batch. Both depend only on the drawn documents. `n_targets_scored` is NOT used: it is
    `batch x S x (L - 1)` every step."""
    n_pad = round(float(beat["pad_target_fraction"]) * float(beat["n_targets_scored"]))
    return (int(beat["n_answer_targets"]), int(n_pad))


def replay_batches(cfg: Run1Config, steps: int, sampling: str | None) -> list[list[int]]:
    """ADAPTER (sampler). The pool positions each step draws, replayed from
    `torch.Generator(device).manual_seed(seed)`, nothing else drawing from it:
    with replacement at 211d6da (`torch.randint`); per-epoch permutation at c02b16d
    (`PermutationEpochs`: a queue of `randperm(n)` drawn when used up)."""
    g = torch.Generator(device=cfg.device).manual_seed(cfg.seed)
    n, b = cfg.pool_length, cfg.batch
    if sampling is not None and "permutation" in sampling:
        queue: list[int] = []
        out = []
        for _ in range(steps):
            while len(queue) < b:
                queue += torch.randperm(n, generator=g, device=cfg.device).tolist()
            out.append(queue[:b])
            queue = queue[b:]
        return out
    return [
        torch.randint(0, n, (b,), generator=g, device=cfg.device).tolist()
        for _ in range(steps)
    ]


def arm_sampling(result: dict) -> str | None:
    """ADAPTER (sampler record; 262b5c5, review c02b16d MINOR-1): `frozen["sampling"]`,
    which `train()` now stamps for EVERY pool arm, FIFO included (at c02b16d only the RSR
    arms' `epoch` block carried it). Where an RSR arm carries both, they must agree;
    a disagreement is returned as a string that matches no sampler, so C2 fails."""
    conf = result["config"]
    top = conf.get("sampling")
    ep = (conf.get("epoch") or {}).get("sampling")
    if ep is not None and top != ep:
        return f"DISAGREE: frozen {top!r} vs epoch {ep!r}"
    return top


def checkpoint_path(out_dir: Path, step: int) -> Path:
    """ADAPTER: the loop's filename (`ckpt-{step:06d}.pt`)."""
    return Path(out_dir) / f"ckpt-{step:06d}.pt"


def read_checkpoint(path: Path) -> dict:
    """ADAPTER: a payload, restoring nothing. A quarantined checkpoint is refused."""
    return ck.load(path, restore_rng=False)


def model_from_payload(payload: dict) -> TGModel:
    """ADAPTER: the frozen model, rebuilt from the checkpoint's own frozen config."""
    tg = dict(payload["meta"]["config"]["tg"])
    tg["block_config"] = tuple(tg["block_config"])
    with torch.random.fork_rng(devices=[]):
        model = TGModel(TGConfig(**tg))
    model.load_state_dict(payload["model"])
    return model


def warm_start_record(result: dict) -> dict:
    """ADAPTER: the warm-start record `train()` stamps into `frozen` (C1)."""
    return dict(result["config"].get("warm_start") or {})


def head_state(payload: dict) -> dict | None:
    """ADAPTER: the trained phi (`checkpoint.py:444-445`)."""
    return (payload.get("policy") or {}).get("__head__")


def job_record(cfg: Run1Config, arm: str) -> dict | None:
    """ADAPTER (orchestrator): the slot's job record for the launcher's job id
    `rsr-run1-<arm>-s<seed>` in the shared root (`slot.py:150-152`)."""
    p = (
        Path(cfg.orch_root)
        / ".orchestrator"
        / "jobs"
        / f"{RUN_ID}-{job_key(arm, cfg.seed)}{cfg.job_suffix}.json"
    )
    return json.loads(p.read_text()) if p.exists() else None


def eval_policy(arm: str, payload: dict, cfg: Run1Config):
    """ADAPTER: an arm's policy at a checkpoint, for the held-out harness (§7.1). psiC /
    age: `build_policy` with the arm's own frozen overrides (read from the checkpoint),
    phi from ``__head__``, ``set_train_step(<the checkpoint's step>)``. FIFO: FIFO."""
    conf = payload["meta"]["config"]
    if arm == "fifo":
        if conf.get("policy") != "fifo":
            raise ControlFailure(
                f"the fifo arm's checkpoint is policy {conf['policy']!r}"
            )
        return FIFOPolicy()
    if conf.get("policy") != "rsr":
        raise ControlFailure(f"the {arm} arm's checkpoint is policy {conf['policy']!r}")
    ov = conf["policy_overrides"]
    if (ov.get("psi_override") == "age_only") is not (arm == "age"):
        raise ControlFailure(f"the {arm} arm's checkpoint has overrides {ov}")
    gen, _ = loop.phi_generator(int(conf["seed"]))  # never the global RNG
    pol = loop.build_policy(
        "rsr",
        d_model=int(conf["tg"]["D"]),
        steps_per_epoch=float(conf["epoch"]["steps_per_epoch"]),
        generator=gen,
        overrides=dict(ov),
        no_warmup_ablation=bool(conf.get("no_warmup_ablation", False)),
    )
    head = head_state(payload)
    if head is None or pol.head is None:
        raise ControlFailure(f"the {arm} arm's checkpoint carries no __head__")
    pol.head.load_state_dict(head)
    pol.set_train_step(int(payload["step"]))
    return pol


# --------------------------------------------------------------------------- #
# Provenance and environment
# --------------------------------------------------------------------------- #


def ledger_mod():
    import ledger as _ledger  # scripts/ledger.py

    return _ledger


#: Homebrew's git, never Apple's shim (CLAUDE.md); Apple silicon first, then Intel.
HOMEBREW_GIT = ("/opt/homebrew/bin/git", "/usr/local/bin/git")


def git_exe() -> str:
    """The git the harness runs (review 6267fd645698 MINOR-8): Homebrew's if present,
    else the PATH's, as an absolute path (Linux CI), else a clear refusal -- never a bare
    ``git`` that a failure would turn into an empty answer."""
    for p in HOMEBREW_GIT:
        if Path(p).is_file() and os.access(p, os.X_OK):
            return p
    found = shutil.which("git")
    if found:
        return os.path.abspath(found)  # not resolve(): keep the name argv[0] carries
    raise Refused(f"no git: none of {HOMEBREW_GIT} exists and none is on PATH")


def tree_state() -> dict[str, Any]:
    """HEAD and the full porcelain, as the launcher reads them (A1.5)."""
    try:
        exe = git_exe()
    except Refused as e:
        return {"sha": None, "dirty": None, "porcelain": None, "error": str(e)}
    try:
        head = subprocess.run(
            [exe, "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        st = subprocess.run(
            [exe, "-C", str(ROOT), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return {"sha": None, "dirty": None, "porcelain": None, "error": str(e)}
    if head.returncode != 0 or st.returncode != 0:
        err = (head.stderr or st.stderr).strip()
        return {"sha": None, "dirty": None, "porcelain": None, "error": err}
    return {
        "sha": head.stdout.strip(),
        "dirty": bool(st.stdout.strip()),
        "porcelain": st.stdout,
    }


def hardware() -> dict[str, Any]:
    def sysctl(key: str) -> str | None:
        try:
            r = subprocess.run(
                ["/usr/sbin/sysctl", "-n", key], capture_output=True, text=True, timeout=5
            )
            return r.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    return {
        "cpu": sysctl("machdep.cpu.brand_string") or platform.processor(),
        "memsize_bytes": sysctl("hw.memsize"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
    }


def launch_env(environ=None) -> dict[str, str | None]:
    environ = os.environ if environ is None else environ
    return {k: environ.get(k) for k in LAUNCH_ENV_VARS}


def env_problems(env: dict[str, str | None], *, dry_run: bool) -> list[str]:
    """Review 7fcd3f91ca8f MINOR-A2-3: a leftover RSR_ORCH_ROOT puts the real jobs in an
    isolated lane root; OUT / JOB_SUFFIX are dry-run hooks that must be unset. A dry run
    (A3c.1: `DRY_RUN=1`, STEPS=0) may set all three -- they are recorded, and C10 fails
    any launch that recorded them (review 6267fd645698 MINOR-4)."""
    if dry_run:
        return []
    out = []
    root = env.get("RSR_ORCH_ROOT")
    if root and Path(root).resolve() != Path(SHARED_ORCH_ROOT).resolve():
        out.append(f"RSR_ORCH_ROOT={root!r} is not the shared root {SHARED_ORCH_ROOT}")
    out += [
        f"{k}={env[k]!r} is set (a dry-run hook)"
        for k in ("OUT", "JOB_SUFFIX")
        if env.get(k)
    ]
    return out


#: T14: the PREREG heading the committed launcher is extracted from. Switching launcher
#: versions is this one line (and the expected sha in the test).
LAUNCHER_HEADING = "### A3c.1 Launcher, version 5"


#: A3.7a: "Each owner value is compared both with `launch-git.json` and with the value
#: the owner registered in Amendment 4" (review 6267fd645698 MAJOR-2). C10 reads the slot
#: table under this heading -- A2.8's, whose four `______` blanks Amendment 4 fills.
#: Switching to a table Amendment 4 prints under its own heading is this one line (and
#: the test's expected heading). Until a value is there, C10 fails: exit 3.
OWNER_TABLE_HEADING = "### A2.8 Owner slots: one value each (supersedes A1.9's table)"
#: The four rows, in this order, each first cell opening with the bolded slot name.
OWNER_TABLE_SLOTS = (("O1", "gamma"), ("O4", "phi_weight_decay"), ("R-1", "iters"),
                     ("O5", "o5"))  # fmt: skip
OWNER_TABLE_HEADER = ("| slot | value |", "|---|---|")
_DECIMAL = re.compile(r"[0-9]+(\.[0-9]+)?")
OWNER_VALUE_FORMAT = {
    "O1": (_DECIMAL, float),
    "O4": (_DECIMAL, float),
    "R-1": (re.compile(r"[1-9][0-9]*"), int),
    "O5": (re.compile(r"eval|train"), str),
}


def parse_owner_table(text: str, heading: str = OWNER_TABLE_HEADING) -> dict:
    """The owner's registered values, from the PREREG's slot table -- the exact format
    and nothing looser (MAJOR-2). Accepted, and only this:

    * exactly one line equal to ``heading``; blank lines; then the header
      ``| slot | value |`` and ``|---|---|``;
    * exactly four rows, in the order O1, O4, R-1, O5, each ``| **<slot>** <any text
      without a pipe> | `<value>` |`` -- the value cell is one backtick-quoted token
      and nothing else;
    * values: O1 and O4 a plain decimal (``0.9``, ``0.01``; no sign, exponent or
      unit), R-1 a positive integer without separators (``3000``), O5 ``eval`` or
      ``train``; a run of underscores (``______``) is a blank;
    * the table ends at the first line not starting with ``|``.

    Returns ``status`` ``filled`` (every value present and well formed), ``blank`` (a
    slot still ``______``), or ``malformed`` / ``absent``, with the problems. Only
    ``filled`` lets C10 pass."""
    out: dict[str, Any] = {"heading": heading, "values": {}, "raw": {}, "problems": []}
    lines = text.splitlines()
    at = [i for i, x in enumerate(lines) if x == heading]
    if len(at) != 1:
        out.update(status="absent" if not at else "malformed")
        out["problems"].append(f"heading {heading!r} occurs {len(at)} times, not once")
        return out
    i = at[0] + 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    table = []
    while i < len(lines) and lines[i].startswith("|"):
        table.append(lines[i])
        i += 1
    probs = out["problems"]
    if tuple(table[:2]) != OWNER_TABLE_HEADER:
        probs.append(f"the table header is {table[:2]!r}, not {OWNER_TABLE_HEADER!r}")
    rows = table[2:]
    if len(rows) != len(OWNER_TABLE_SLOTS):
        probs.append(f"{len(rows)} slot rows, not {len(OWNER_TABLE_SLOTS)}")
    blank = []
    for row, (slot, name) in zip(rows, OWNER_TABLE_SLOTS, strict=False):
        cells = row.split(" | ")
        if (len(cells) != 2 or not row.startswith(f"| **{slot}** ")
                or not row.endswith(" |")):  # fmt: skip
            probs.append(f"row {row!r} is not `| **{slot}** <text> | `<value>` |`")
            continue
        cell = cells[1][: -len(" |")]
        m = re.fullmatch(r"`([^`]*)`", cell)
        if m is None:
            probs.append(f"{slot}: the value cell {cell!r} is not one `quoted` token")
            continue
        raw = m.group(1)
        out["raw"][slot] = raw
        if re.fullmatch(r"_+", raw):
            blank.append(slot)
            continue
        pattern, cast = OWNER_VALUE_FORMAT[slot]
        if pattern.fullmatch(raw) is None:
            probs.append(f"{slot}: {raw!r} is not in the registered format")
            continue
        out["values"][name] = cast(raw)
    if probs:
        out["status"] = "malformed"
    elif blank:
        out["status"] = "blank"
        probs.append(
            f"owner slot(s) {blank} still blank (______): Amendment 4 fills them"
        )
    else:
        out["status"] = "filled"
    return out


def owner_table(cfg: Run1Config) -> dict:
    """`parse_owner_table` on the file ``cfg.owner_prereg`` names (PREREG for the run)."""
    path = Path(cfg.owner_prereg)
    path = path if path.is_absolute() else ROOT / path
    if not path.exists():
        return {"heading": OWNER_TABLE_HEADING, "status": "absent", "values": {},
                "raw": {}, "problems": [f"{path} does not exist"]}  # fmt: skip
    return {"file": str(path), **parse_owner_table(path.read_text())}


def launcher_text_from_prereg() -> str:
    """The launcher under ``LAUNCHER_HEADING``, from the PREREG: the committed file must
    equal it byte for byte (T14)."""
    t = (ROOT / PREREG).read_text()
    i = t.index(LAUNCHER_HEADING)
    j = t.index("```zsh\n", i) + len("```zsh\n")
    return t[j : t.index("\n```\n", j)] + "\n"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Documents, vocabulary, C0, preflight
# --------------------------------------------------------------------------- #


def syn_cfg(cfg: Run1Config) -> SyntheticConfig:
    return SyntheticConfig(sentences_per_document=cfg.steps_per_stream, seed=cfg.seed)


def vocab_map(cfg: Run1Config) -> dict[str, int]:
    """The map of documents ``[0, vocab_documents)`` at the seed (§3), as `train()`."""
    vdocs = loop.generate(
        SyntheticConfig(
            sentences_per_document=cfg.steps_per_stream,
            seed=cfg.seed,
            n_documents=cfg.vocab_documents,
        )
    )
    return loop.build_vocab(vdocs)


def docs_by_id(cfg: Run1Config, ids) -> list:
    sc = syn_cfg(cfg)
    return [_generate_document(i, sc) for i in ids]


def _overlap(a, b) -> bool:
    return max(a[0], b[0]) < min(a[1], b[1])


def check_c0(cfg: Run1Config) -> dict:
    """C0 (§8, §2, A1.8 m-3): pool and EVAL disjoint from every used range and from each
    other, below `1_000_003`, closed under the vocabulary, and the PREREG's map hash.
    Before any model loads. Raises `ControlFailure`."""
    out: dict[str, Any] = {}
    for name, rng in (("pool", cfg.pool), ("eval", cfg.eval_range)):
        out[f"{name}_overlaps"] = [
            k for k, r in cfg.used_ranges.items() if _overlap(rng, tuple(r))
        ]
        out[f"{name}_below_K"] = rng[1] <= GEN_SEED_K
    out["pool_eval_disjoint"] = not _overlap(cfg.pool, cfg.eval_range)
    vmap = vocab_map(cfg)
    out["vocab_sha256"] = loop.vocab_sha256(vmap)
    for name, rng in (("pool", cfg.pool), ("eval", cfg.eval_range)):
        try:
            loop.check_vocabulary_closure(docs_by_id(cfg, range(*rng)), vmap)
            out[f"{name}_closure"] = True
        except ValueError as e:
            out[f"{name}_closure"] = False
            out[f"{name}_closure_error"] = str(e)
    problems = [k for k in ("pool_overlaps", "eval_overlaps") if out[k]]
    problems += [
        k
        for k in (
            "pool_below_K",
            "eval_below_K",
            "pool_eval_disjoint",
            "pool_closure",
            "eval_closure",
        )
        if out[k] is not True
    ]
    want = cfg.expected_vocab_sha256
    if want is not None and out["vocab_sha256"] != want:
        problems.append("vocab_sha256")
    out["ok"] = not problems
    out["failed"] = problems
    if problems:
        raise ControlFailure(f"C0 failed: {problems} ({out})")
    return out


def preflight(*, dry_run: bool = False, environ=None) -> dict:
    """A1.2 step 2 / A1.8 m-3: run through the slot before any training command; rc 0
    required. Never loads a model and never reads an EVAL document through one."""
    out: dict[str, Any] = {"env": launch_env(environ), "dry_run": dry_run, "seeds": {}}
    out["env_problems"] = env_problems(out["env"], dry_run=dry_run)
    committed = (ROOT / LAUNCHER).read_text() if (ROOT / LAUNCHER).exists() else None
    out["launcher_heading"] = LAUNCHER_HEADING
    out["launcher_byte_identical_to_prereg"] = committed == launcher_text_from_prereg()
    man = Path(SUBSTRATE[0]["manifest"])
    out["manifest_sha256_ok"] = man.exists() and sha256_file(man) == MANIFEST_SHA256
    for s in SEEDS:
        cfg = prereg_config(s, 3)  # iters is not read by C0 or the hash
        row: dict[str, Any] = {}
        try:
            row["C0"] = check_c0(cfg)
        except ControlFailure as e:
            row["C0"] = {"ok": False, "error": str(e)}
        ckpt = Path(cfg.substrate.checkpoint)
        row["substrate_sha256_ok"] = (
            ckpt.exists() and sha256_file(ckpt) == cfg.substrate.sha256
        )
        out["seeds"][s] = row
    out["ok"] = (
        not out["env_problems"]
        and out["launcher_byte_identical_to_prereg"]
        and out["manifest_sha256_ok"]
        and all(r["C0"]["ok"] and r["substrate_sha256_ok"] for r in out["seeds"].values())
    )
    return out


# --------------------------------------------------------------------------- #
# The direct launch (tests ONLY -- the real launch and its dry run are launcher v5)
# --------------------------------------------------------------------------- #


def arm_dir(out_root: Path, arm: str, seed: int) -> Path:
    return Path(out_root) / arm / f"seed{seed}"


def job_key(arm: str, seed: int) -> str:
    return f"{arm}-s{seed}"  # the launcher's `k=$a-s$s`


def refuse_resume(out_dir: Path) -> None:
    """PREREG §11.3: an arm that has started is refused, never resumed."""
    if RESUME_PERMITTED:
        raise AssertionError("RESUME_PERMITTED is True: PREREG §11.3 needs an amendment")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise Refused(
            f"{out_dir} already holds {sorted(p.name for p in out_dir.iterdir())[:6]}: "
            "an "
            f"arm that has started is never resumed (PREREG §11.3)"
        )


def launch_direct(
    cfgs: list[Run1Config],
    owner: OwnerValues,
    out_root: Path,
    *,
    threads: int,
    allow_dirty: bool = False,
) -> dict[str, int]:
    """The arms through the same command line, WITHOUT the slot, into the launcher's
    layout (`launch-git.json`, `logs/<arm>-s<seed>.{log,rc}`, `<arm>/seed<s>/`). For the
    test fixture and dry runs; a real launch is `launch_run1.zsh` inside tmux."""
    validate_owner_values(owner)
    out_root = Path(out_root)
    if (out_root / "launch-git.json").exists():
        raise Refused(f"{out_root} was launched already (launch-git.json exists)")
    for c in cfgs:
        for arm in ARMS:
            refuse_resume(arm_dir(out_root, arm, c.seed))
    tree = tree_state()
    if tree["sha"] is None or (tree["dirty"] and not allow_dirty):
        raise Refused(f"not a clean committed head: {tree}")
    for c in cfgs:
        check_c0(c)  # before any model loads
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    (out_root / "launch-git.json").write_text(
        json.dumps(
            {
                "run_sha": tree["sha"],
                "dirty": tree["dirty"],
                "porcelain_lines": len((tree["porcelain"] or "").splitlines()),
                "gamma": repr(owner.gamma),
                "wd": repr(owner.phi_weight_decay),
                "steps": str(owner.iters),
                "ckpt": str(derive_ckpt(owner.iters)),
                "o5": owner.o5,
                "c1_flags": " ".join(C1_FLAGS[owner.o5]),
                "c2_flags": " ".join(C2_FLAGS),
                "c3_flags": "",
                "ckpt_rule": CKPT_RULE,
                "rsr_orch_root": launch_env()["RSR_ORCH_ROOT"] or "",
                "out": launch_env()["OUT"] or "",
                "job_suffix": launch_env()["JOB_SUFFIX"] or "",
                "dry_run": os.environ.get("DRY_RUN", ""),
                "launched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "launcher": "run.py launch_direct (tests / dry runs; no slot)",
                "env": launch_env(),
            }
        )
        + "\n"
    )
    rcs = {}
    for c in cfgs:
        for arm in ARMS:
            k = job_key(arm, c.seed)
            argv = cli_argv(c, arm, owner, arm_dir(out_root, arm, c.seed))
            with open(out_root / "logs" / f"{k}.log", "w") as log:
                rc = subprocess.run(
                    argv,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=thread_env(threads),
                ).returncode
            (out_root / "logs" / f"{k}.rc").write_text(f"{rc}\n")
            rcs[k] = rc
    return rcs


#: A3b.5: the order a worst arm / seed is taken in.
EXIT_RANK = {
    int(Exit.INERT): 3,
    int(Exit.DID_NOT_RUN): 2,
    int(Exit.FAIL): 1,
    int(Exit.OK): 0,
}


def arm_outcome(rc: int | None, hb: dict, result: dict | None) -> tuple[int, str, str]:
    """A3b.3's table for one arm: (exit, ledger status, required note). rc 3 is
    `did_not_run` whatever the heartbeat holds: with a completed footer it is train()'s
    liveness band `invalid` ("3, not a valid measurement", R-2026-09-22-inert-exit-5),
    which is not a crash (review 6267fd645698 MINOR-2)."""
    foot = (hb.get("footer") or {}).get("status")
    if rc == 0 and foot == "completed" and result is not None:
        return int(Exit.OK), "ok", "ARM_OK"
    if rc == 5:
        return int(Exit.INERT), "partial", "INERT"
    if rc == 1 and foot == "completed":
        return int(Exit.FAIL), "failed", "LIVENESS_INCONCLUSIVE"
    if rc == 1 and not hb.get("exists"):
        return int(Exit.DID_NOT_RUN), "did_not_run", "ARM_DID_NOT_RUN"
    if rc == 3:
        if foot == "completed":
            note = "ARM_DID_NOT_RUN (liveness invalid)"
        elif hb.get("exists"):
            note = "ARM_DID_NOT_RUN (rc 3)"
        else:
            note = "ARM_DID_NOT_RUN"
        return int(Exit.DID_NOT_RUN), "did_not_run", note
    if rc is None:
        return int(Exit.DID_NOT_RUN), "did_not_run", "ARM_DID_NOT_RUN (no rc file)"
    return int(Exit.DID_NOT_RUN), "crashed", "ARM_CRASHED"


def worst_outcome(cands: list[tuple[int, str, str]]) -> tuple[int, str, str] | None:
    """A3b.3 + A3b.5 + A3c.2: the worst of ``cands`` by exit, 5 > 3 > 1 > 0; among equal
    exits, the FIRST in the order given. Callers give arms in (psiC, fifo, age) order,
    which is A3c.2's tie rule. None if every candidate is OK."""
    bad = [c for c in cands if c[0] != int(Exit.OK)]
    if not bad:
        return None
    top = max(EXIT_RANK[c[0]] for c in bad)
    return next(c for c in bad if EXIT_RANK[c[0]] == top)


def arm_steps_done(rc: int | None, hb: dict, result: dict | None) -> int | None:
    """The steps one arm actually recorded (A3b.3: "as actually recorded. Neither may be
    cleared"; review 6267fd645698 MAJOR-1): the printed result's ``steps_done`` if the
    process printed one, else the last heartbeat beat's step + 1 (BEAT_EVERY = 1, so
    every step beats), else 0 for a process that ran -- it has an rc or a heartbeat --
    and recorded no step. None (absent) only for an arm with neither: never collected."""
    if result is not None and result.get("steps_done") is not None:
        return int(result["steps_done"])
    if hb.get("beats"):
        return int(hb["beats"][-1]["step"]) + 1
    if hb.get("exists") or rc is not None:
        return 0
    return None


def classify_rc(rc: int | None, job: dict | None, hb: dict, result: dict | None) -> str:
    """A1.2's reading rule: what an arm's rc means, disambiguated by the job record and
    the heartbeat."""
    if rc is None:
        return "no rc file: never collected"
    foot = (hb.get("footer") or {}).get("status")
    if rc == 0 and foot == "completed":
        return "ran, live" if result else "ran (no printed result)"
    if rc == 1 and foot == "completed":
        return "liveness inconclusive: the arm is not read"
    if rc == 5 and foot == "completed":
        return "INERT"
    if rc == 1 and not hb.get("exists"):
        return "did not run: exception before the heartbeat (e.g. WarmStartRefused)"
    if rc == 1 and hb.get("crash") is not None:
        return "crashed mid-loop (heartbeat crash line)"
    if rc == 3 and not hb.get("exists"):
        return "slot refused (no job record)" if job is None else "CLI usage error"
    if rc == 3 and foot == "completed":
        return "liveness invalid: not a valid measurement (rc 3)"
    if rc > 128:
        return f"killed by signal {rc - 128}"
    return f"rc {rc}: reported as itself"


# --------------------------------------------------------------------------- #
# The held-out harness (PREREG §7.1, §7.2, T3, T11, T13)
# --------------------------------------------------------------------------- #


class Recorder:
    """Wraps any policy for the held-out pass. Records the victims (`written_at`), each
    eviction's age and ADR-0006 displacement, and the IN-LOOP memory after every write
    -- read off the memory the loop holds (`on_write`'s `MemoryState`), not derived from
    the victims, so C7 compares two independent records.

    `observe` is refused unless ``forward_observe`` (LRU reads attention, T11); an RSR
    arm never observes in the held-out harness (§7.1)."""

    def __init__(self, inner, *, forward_observe: bool = False) -> None:
        self.inner = inner
        self.forward_observe = forward_observe
        self.name = getattr(inner, "name", type(inner).__name__)
        self._clear()

    def _clear(self) -> None:
        self.victims: dict[int, dict[int, int]] = {}
        self.evictions: dict[int, list[tuple[int, int, int]]] = {}
        self.after_write: dict[int, dict[int, list[int]]] = {}

    @property
    def needs_eval_context(self) -> bool:
        return bool(getattr(self.inner, "needs_eval_context", False))

    learns = False

    def to(self, device):
        if hasattr(self.inner, "to"):
            self.inner.to(device)
        return self

    def reset(self) -> None:
        self._clear()
        self.inner.reset()

    def select_eviction(self, slots, context, step: int) -> int:
        if not bool(slots.live.all()):
            raise ControlFailure(f"step {step}: asked to evict from a non-full memory")
        k = int(self.inner.select_eviction(slots, context, step))
        if not bool(slots.live[k]):
            raise ControlFailure(f"step {step}: {self.name} returned the dead slot {k}")
        _rank, disp = rank_shift(slots, k)
        w = int(slots.written_at[k])
        self.victims.setdefault(slots.row, {})[step] = w
        self.evictions.setdefault(slots.row, []).append((step, step - w, disp))
        return k

    def observe(self, slots, attn, step: int) -> None:
        if not self.forward_observe:
            raise RuntimeError("observe is off in the held-out harness (PREREG §7.1)")
        self.inner.observe(slots, attn, step)

    def on_write(self, slots, slot: int, step: int) -> None:
        live = slots.live.nonzero(as_tuple=True)[0]
        self.after_write.setdefault(slots.row, {})[step] = sorted(
            int(slots.written_at[j]) for j in live
        )
        self.inner.on_write(slots, slot, step)

    def resident_before(self, row: int, step: int) -> list[int]:
        """The memory the forward at ``step`` read: after the last write before it."""
        aw = self.after_write.get(row, {})
        prior = [s for s in aw if s < step]
        return aw[max(prior)] if prior else []


class PerRow:
    """T13: a row-correct batched wrapper -- one per-document policy per batch row,
    dispatched on `MemoryState.row`. C9b checks it hands each row its own document."""

    def __init__(self, policies: list, name: str) -> None:
        self.policies = policies
        self.name = name

    def select_eviction(self, slots, context, step: int) -> int:
        return self.policies[slots.row].select_eviction(slots, context, step)

    def observe(self, slots, attn, step: int) -> None:
        return None

    def on_write(self, slots, slot: int, step: int) -> None:
        self.policies[slots.row].on_write(slots, slot, step)

    def reset(self) -> None:
        for p in self.policies:
            p.reset()


class ZeroSlots(torch.nn.Module):
    """S0-03's ``slots_zeroed`` (`answer_readout`): every forward reads an all-zero
    memory, validity mask unchanged; writes unchanged. Under a policy, the policy still
    decides over the gestalts written (§7.2: the floor "under each arm")."""

    def __init__(self, model: TGModel) -> None:
        super().__init__()
        self.inner = model
        self.cfg = model.cfg

    @property
    def embed(self):
        return self.inner.embed

    @property
    def blocks(self):
        return self.inner.blocks

    def forward(self, ids_t, mask_t, kv, valid, bos_ctx, bos_valid, **kw):
        zero = torch.zeros_like(kv)
        return self.inner(ids_t, mask_t, zero, valid, bos_ctx, bos_valid, **kw)


def run_batch(
    model,
    policy,
    docs,
    vmap,
    cfg: Run1Config,
    *,
    zero_slots=False,
    decision_context="eval_forward",
    observe=False,
) -> list[dict]:
    """One held-out batch, eval mode, ``no_grad``. Per document: every answer target's
    `(t, gap, correct, nll)` (S0-03: argmax over the full vocabulary equals the answer),
    real-token NLL sum and count, the victims, each eviction's age and displacement,
    in-loop residency of every queried assert, and `|memory_t|` before each forward.

    ``decision_context="eval_forward"`` runs `run_policy_loop(eval_capture=True)` for a
    policy that `needs_eval_context`: training's decision path (ADR-0009 L2). With
    ``"step_forward"`` the decision reads the loop's own srep (here also eval-mode)."""
    if decision_context not in ("eval_forward", "step_forward"):
        raise ValueError(decision_context)
    S, L = cfg.steps_per_stream, int(model.cfg.max_sentence_tokens)
    ids, mask = loop.encode(docs, vmap, max_tokens=L, steps=S)
    tmask, gap = loop.answer_targets(docs, vmap, max_tokens=L, steps=S)
    dev = torch.device(cfg.device)
    ids, mask, tmask, gap = (x.to(dev) for x in (ids, mask, tmask, gap))
    B = len(docs)
    answers: list[list] = [[] for _ in range(B)]
    real_sum = [0.0] * B
    real_n = [0] * B

    def step_fn(t, out, ids_t, mask_t, row_valid):
        per = loop.lm_token_losses(out.logits, ids_t).view(B, L - 1)
        real = mask_t[:, 1:]
        sums = (per.double() * real.double()).sum(-1)
        ns = real.sum(-1)
        for b in range(B):
            real_sum[b] += float(sums[b])
            real_n[b] += int(ns[b])
        am = tmask[:, t, 1:]
        if bool(am.any()):
            pred = out.logits[:, :-1].argmax(-1)
            tgt = ids_t[:, 1:]
            for b, j in am.nonzero().tolist():
                ok = bool(pred[b, j] == tgt[b, j])
                answers[b].append((t, int(gap[b, t]), ok, float(per[b, j])))
        return torch.zeros((), device=dev)

    rec = Recorder(policy, forward_observe=observe)
    net = ZeroSlots(model) if zero_slots else model
    eval_ctx = rec.needs_eval_context and decision_context == "eval_forward"
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            lengths = torch.full((B,), S, device=dev)
            run_policy_loop(
                net,
                ids,
                mask,
                lengths,
                rec,
                step_fn=step_fn,
                observe=observe,
                eval_capture=eval_ctx,
            )
    finally:
        model.train(was)
    out = []
    for b, doc in enumerate(docs):
        assert_of = {q: a for a, q in doc.pairs}
        hits = [
            (q - assert_of[q], assert_of[q] in rec.resident_before(b, q))
            for q in sorted(assert_of)
        ]
        out.append(
            {
                "doc": doc.doc_id,
                "answers": answers[b],
                "real_sum": real_sum[b],
                "real_n": real_n[b],
                "victims": {str(k): v for k, v in sorted(rec.victims.get(b, {}).items())},
                "evictions": rec.evictions.get(b, []),
                "hits": hits,
                "n_live_before": [len(rec.resident_before(b, t)) for t in range(S)],
            }
        )
    return out


def residency_ok(doc, record: dict, m: int) -> bool:
    """C7 (B2 §9.1): in-loop residency of every queried assert equals the model-free
    `headroom.simulate` replay of the same victim sequence."""
    victims = {int(k): int(v) for k, v in record["victims"].items()}
    rep = simulate(doc, B2().Scripted(victims), m)["queries"]
    return [(q["gap"], q["hit"]) for q in rep] == [tuple(h) for h in record["hits"]]


def oracle_policy(docs: list, gamma: float = ORACLE_GAMMA) -> PerRow:
    """§7.2 / A2.2: `OraclePolicy(discounted_demand(doc, gamma_o))`, one per row."""
    return PerRow([OraclePolicy(discounted_demand(d, gamma)) for d in docs], "oracle")


def reference_policy(name: str, docs: list, seed: int):
    """Evaluation-only references (§7.2; A1.1 LRU)."""
    if name == "oracle":
        return oracle_policy(docs)
    if name == "lru":
        return LRUPolicy()  # keyed by (row, written_at); reset at each loop entry
    if name.startswith("random"):
        k = int(name[len("random") :])
        return PerRow([B2().random_policy(k, seed, d.doc_id) for d in docs], name)
    raise ValueError(name)


def heldout_passes(
    snapshot_steps: list[int] | None = None,
) -> list[tuple[str, str, bool]]:
    """Every held-out pass: (name, payload key, slots zeroed). `sz_fifo` is S0-03's
    floor; `sz_psiC` / `sz_age` are §7.2's "under each arm". `psiC@k` / `age@k` are the
    A1.8 m-9 snapshots (DIAGNOSTIC)."""
    passes = [(a, a, False) for a in ARMS]
    passes += [("oracle", "fifo", False), ("lru", "fifo", False)]
    passes += [(f"random{k}", "fifo", False) for k in range(N_RANDOM)]
    passes += [(f"sz_{a}", a, True) for a in ARMS]
    for k in snapshot_steps or []:
        passes += [(f"{a}@{k}", f"{a}@{k}", False) for a in RSR_ARMS]
    return passes


def arm_of(key: str) -> str:
    return key.split("@")[0]


def run_heldout(
    cfg: Run1Config,
    payloads: dict[str, dict],
    out_dir: Path,
    *,
    passes,
    limit: int | None = None,
    decision_context="eval_forward",
) -> dict:
    """Every held-out pass over EVAL, batched identically; per-document records to
    ``out_dir/<pass>.jsonl.gz``. T5: every batch of every pass is timed."""
    vmap = vocab_map(cfg)
    ids = list(cfg.eval_ids())[: limit or None]
    docs = docs_by_id(cfg, ids)
    loop.check_vocabulary_closure(docs, vmap)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = {k: model_from_payload(p).to(cfg.device) for k, p in payloads.items()}
    timing: dict[str, dict] = {}
    residency: dict[str, dict] = {}
    for name, src, zero in passes:
        recs: list[dict] = []
        secs: list[float] = []
        for i in range(0, len(docs), cfg.eval_batch):
            chunk = docs[i : i + cfg.eval_batch]
            if src == name or name.startswith("sz_"):
                pol = eval_policy(arm_of(src), payloads[src], cfg)
            else:
                pol = reference_policy(name, chunk, cfg.seed)
            t0 = time.perf_counter()
            recs += run_batch(
                models[src],
                pol,
                chunk,
                vmap,
                cfg,
                zero_slots=zero,
                decision_context=decision_context,
                observe=name == "lru",
            )
            secs.append(time.perf_counter() - t0)
        timing[name] = {
            "first_batch_s": secs[0] if secs else None,
            "batches": len(secs),
            "total_s": sum(secs),
            "eval_batch": cfg.eval_batch,
        }
        if not zero:
            bad = [
                r["doc"]
                for r, d in zip(recs, docs, strict=True)
                if not residency_ok(d, r, cfg.memory_slots)
            ]
            residency[name] = {
                "n_documents": len(recs),
                "mismatched_docs": bad[:20],
                "n_mismatched": len(bad),
            }
        with gzip.open(out_dir / f"{name}.jsonl.gz", "wt") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
    return {"timing": timing, "residency": residency, "doc_ids": ids}


def read_pass(out_dir: Path, name: str) -> list[dict]:
    with gzip.open(Path(out_dir) / f"{name}.jsonl.gz", "rt") as fh:
        return [json.loads(x) for x in fh if x.strip()]


# --------------------------------------------------------------------------- #
# C9 / C9b (A2.2, T13; review 7fcd3f91ca8f MINOR-A2-2) and the T7 probe
# --------------------------------------------------------------------------- #


def oracle_victims(doc, gamma: float, m: int) -> dict[int, int]:
    """The model-free oracle replay, from the document's OWN demand."""
    rec = Recorder(OraclePolicy(discounted_demand(doc, gamma)))
    simulate(doc, rec, m)
    return rec.victims.get(0, {})


def c9_oracle(model, cfg: Run1Config, vmap, *, policy_fn=oracle_policy) -> dict:
    """C9: on the first 16 EVAL documents, the model-read loop's oracle victims at
    gamma_o 0.5 and 0.9 are identical document by document. C9b: at 0.9 they equal the
    model-free replay built from each document's own demand -- which catches a batched
    wrapper handing a row another document's demand (C9 and C7 cannot). A failure
    invalidates ONLY the oracle-derived numbers (regret, headroom share)."""
    docs = docs_by_id(cfg, list(cfg.eval_ids())[:C9_N_DOCS])
    loop_v: dict[float, dict[int, dict]] = {}
    for g in C9_GAMMAS:
        loop_v[g] = {}
        for i in range(0, len(docs), cfg.eval_batch):
            chunk = docs[i : i + cfg.eval_batch]
            for r in run_batch(model, policy_fn(chunk, g), chunk, vmap, cfg):
                loop_v[g][r["doc"]] = {int(k): v for k, v in r["victims"].items()}
    lo, hi = C9_GAMMAS
    c9_bad = [d.doc_id for d in docs if loop_v[lo][d.doc_id] != loop_v[hi][d.doc_id]]
    c9b_bad = [
        d.doc_id
        for d in docs
        if loop_v[ORACLE_GAMMA][d.doc_id]
        != oracle_victims(d, ORACLE_GAMMA, cfg.memory_slots)
    ]
    return {
        "documents": [docs[0].doc_id, docs[-1].doc_id + 1] if docs else None,
        "C9": {"gammas": list(C9_GAMMAS), "differing_docs": c9_bad, "ok": not c9_bad},
        "C9b": {"gamma": ORACLE_GAMMA, "differing_docs": c9b_bad, "ok": not c9b_bad},
        "ok": bool(docs) and not c9_bad and not c9b_bad,
        "scope": "a failure marks oracle regret, headroom share and the oracle's own "
        "accuracy NOT READ; exit and status unchanged (A3.6)",
    }


def probe_oracle_invariance(docs, m: int, gammas=T7_GAMMAS) -> dict:
    """T7 (build-time, model-free): the oracle's victims at every gamma in ``gammas``."""
    differ = []
    for d in docs:
        ref = oracle_victims(d, gammas[0], m)
        differ += [(d.doc_id, g) for g in gammas[1:] if oracle_victims(d, g, m) != ref]
    return {
        "gammas": list(gammas),
        "n_documents": len(docs),
        "differ": differ[:20],
        "n_differ": len(differ),
        "ok": not differ,
    }


# --------------------------------------------------------------------------- #
# Controls from the arms' artefacts
# --------------------------------------------------------------------------- #


def c1_warm_start(results: dict[str, dict], cfg: Run1Config) -> dict:
    out = {}
    for arm, r in results.items():
        ws = warm_start_record(r)
        want = cfg.expected_vocab_sha256 or ws.get("run_vocab_sha256")
        keys = (
            "sha256",
            "source_vocab_sha256",
            "run_vocab_sha256",
            "vocab_checked",
            "seed_matches",
            "source_seed",
            "allow_unrecorded_vocabulary",
            "allow_seed_mismatch",
        )
        out[arm] = {k: ws.get(k) for k in keys}
        out[arm]["ok"] = (
            ws.get("sha256") == cfg.substrate.sha256
            and ws.get("source_vocab_sha256") == want
            and ws.get("run_vocab_sha256") == want
            and ws.get("vocab_checked") is True
            and ws.get("seed_matches") is True
            and ws.get("source_seed") == cfg.seed
            and not ws.get("allow_unrecorded_vocabulary")
            and not ws.get("allow_seed_mismatch")
        )
    return {"per_arm": out, "ok": bool(out) and all(v["ok"] for v in out.values())}


def pool_fingerprints(cfg: Run1Config) -> list[tuple[int, int]]:
    """Per pool document: (answer targets, padding targets), as `train()`'s tally counts
    them -- the inputs of the expected batch fingerprint."""
    vmap = vocab_map(cfg)
    docs = docs_by_id(cfg, range(*cfg.pool))
    L, S = CLI_MAX_TOKENS, cfg.steps_per_stream
    _ids, mask = loop.encode(docs, vmap, max_tokens=L, steps=S)
    tm, _ = loop.answer_targets(docs, vmap, max_tokens=L, steps=S)
    real = mask[:, :, 1:].sum(dim=(1, 2)).tolist()
    ans = tm[:, :, 1:].sum(dim=(1, 2)).tolist()
    return [(int(a), int(S * (L - 1) - r)) for a, r in zip(ans, real, strict=True)]


def c2_pairing(
    beats: dict[str, list[dict]],
    ids: dict[str, dict[int, list[int]]],
    cfg: Run1Config,
    sampling: str | None,
    *,
    samplings: dict[str, str | None] | None = None,
) -> dict:
    """C2 (A1.8 m-4).

    ``samplings`` (262b5c5): the sampler each arm's config records; when given, every
    arm must record the same one, and ``sampling`` (what the replay uses) is it.

    (a) PRIMARY: the logged per-step pool ids are identical across the arms at every
        step, and every full epoch's ids are the pool exactly once (C-2).
    (b) the per-step fingerprint (`n_answer_targets`, padding targets) is identical
        across arms AND equals the one the replayed sampler predicts.
    (c) `loss_lm` bitwise equal across arms for `k < T_warm`."""
    tw = cfg.t_warm_steps
    arms = list(beats)
    by = {a: {b["step"]: b for b in beats[a]} for a in arms}
    steps = sorted(set.intersection(*(set(v) for v in by.values()))) if by else []
    same_steps = all(set(v) == set(steps) for v in by.values())
    # (a)
    have_ids = bool(ids) and all(ids.get(a) for a in arms)
    a_cross, a_epoch = [], []
    if have_ids:
        a_cross = [k for k in steps if len({tuple(ids[a].get(k, ())) for a in arms}) != 1]
        spe = int(cfg.steps_per_epoch)
        first = ids[arms[0]]
        pool = list(range(*cfg.pool))
        for e in range(len(steps) // spe):
            seen = sorted(
                i for k in range(e * spe, (e + 1) * spe) for i in first.get(k, [])
            )
            if seen != pool:
                a_epoch.append(e)
    a_ok = have_ids and not a_cross and not a_epoch
    # (b)
    fp = pool_fingerprints(cfg)
    expected = replay_batches(cfg, steps[-1] + 1 if steps else 0, sampling)
    b_cross, b_replay = [], []
    for k in steps:
        prints = {a: batch_fingerprint(by[a][k]) for a in arms}
        if len(set(prints.values())) != 1:
            b_cross.append(k)
        want = (sum(fp[i][0] for i in expected[k]), sum(fp[i][1] for i in expected[k]))
        if any(p != want for p in prints.values()):
            b_replay.append(k)
        if have_ids:
            got = [i - cfg.pool[0] for i in ids[arms[0]].get(k, [])]
            if got != expected[k]:
                b_replay.append(k)
    # (c)
    lm = [k for k in steps if k < tw and len({by[a][k]["loss_lm"] for a in arms}) != 1]
    required_a = True  # C-2 is in the code (c02b16d): (a) is always required
    s_ok = samplings is None or (
        None not in samplings.values() and set(samplings.values()) == {sampling}
    )
    return {
        "steps_compared": len(steps),
        "warm_steps_compared": sum(1 for k in steps if k < tw),
        "a_pool_ids": {
            "present": have_ids,
            "required": required_a,
            "cross_arm_mismatch_steps": a_cross[:20],
            "epochs_not_exactly_the_pool": a_epoch[:20],
        },
        "b_fingerprint": {
            "cross_arm_mismatch_steps": b_cross[:20],
            "replay_mismatch_steps": sorted(set(b_replay))[:20],
            "sampling": sampling,
            "sampling_per_arm": samplings,
            "sampling_agrees": s_ok,
        },
        "c_loss_lm_warm_mismatch_steps": lm[:20],
        "same_step_sets": same_steps,
        "ok": bool(steps)
        and same_steps
        and not b_cross
        and not b_replay
        and not lm
        and a_ok
        and s_ok,
    }


def c3_frozen(payloads: dict[str, dict], substrate_model: dict) -> dict:
    out = {}
    for key, p in payloads.items():
        m = p["model"]
        keys_equal = set(m) == set(substrate_model)
        differ = [
            k
            for k in substrate_model
            if k in m and not torch.equal(m[k], substrate_model[k])
        ]
        out[key] = {
            "n_tensors": len(m),
            "keys_equal": keys_equal,
            "differ": differ[:10],
            "ok": keys_equal and not differ,
        }
    return {
        "per_checkpoint": out,
        "n_substrate_tensors": len(substrate_model),
        "ok": bool(out) and all(v["ok"] for v in out.values()),
    }


def load_substrate_model(cfg: Run1Config) -> tuple[dict, str]:
    """The substrate's `model` tensors, from bytes this harness hashed itself."""
    data = Path(cfg.substrate.checkpoint).read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != cfg.substrate.sha256:
        raise ControlFailure(f"substrate hashes to {got}, not {cfg.substrate.sha256}")
    return ck.load_bytes(data, source=cfg.substrate.checkpoint)["model"], got


def c5_liveness(results: dict[str, dict]) -> dict:
    """A2.3: `live` required in every arm; `inert` -> exit 5; the ratio is reported."""
    per = {
        a: {k: (r.get("liveness") or {}).get(k) for k in ("band", "ratio", "exit_code")}
        for a, r in results.items()
    }
    return {
        "per_arm": per,
        "ok": bool(per) and all(v["band"] == "live" for v in per.values()),
        "inert": [a for a, v in per.items() if v["band"] == "inert"],
        "ratio_note": "the ratio is reported, never a control (A2.3)",
    }


def c6_steps(results: dict[str, dict], hbs: dict[str, dict], cfg: Run1Config) -> dict:
    per = {}
    for a, r in results.items():
        foot = hbs[a]["footer"] or {}
        per[a] = {
            "steps_requested": r.get("steps_requested"),
            "steps_done": r.get("steps_done"),
            "status": foot.get("status"),
            "n_beats": len(hbs[a]["beats"]),
        }
        per[a]["ok"] = (
            r.get("steps_done") == cfg.iters
            and r.get("steps_requested") == cfg.iters
            and foot.get("status") == "completed"
            and hbs[a]["crash"] is None
        )
    return {"per_arm": per, "ok": bool(per) and all(v["ok"] for v in per.values())}


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def c8_finite(beats: dict[str, list[dict]]) -> dict:
    per = {}
    for a in RSR_ARMS:
        if a in beats:
            bad = [
                b["step"]
                for b in beats[a]
                if not (_finite(b.get("loss_mc")) and _finite(b.get("phi_grad_norm")))
            ]
            per[a] = {
                "n_steps": len(beats[a]),
                "nonfinite_steps": bad[:20],
                "ok": not bad,
            }
    return {"per_arm": per, "ok": bool(per) and all(v["ok"] for v in per.values())}


def _cmd_flag(cmd: list[str] | None, flag: str) -> str | None:
    if not cmd or flag not in cmd:
        return None
    i = cmd.index(flag)
    return cmd[i + 1] if i + 1 < len(cmd) else None


def _cmd_overrides(cmd: list[str] | None, key: str) -> list[str]:
    """Every ``--policy-override <key>=<v>`` value in a command, in order."""
    cmd = list(cmd or [])
    return [
        cmd[i + 1].split("=", 1)[1]
        for i, x in enumerate(cmd[:-1])
        if x == "--policy-override" and cmd[i + 1].startswith(f"{key}=")
    ]


def c10_launch_values(
    cfg: Run1Config,
    owner: OwnerValues,
    launch: dict,
    results: dict[str, dict],
    hbs: dict[str, dict],
    jobs: dict[str, dict | None],
    out_root: Path,
    *,
    rcs: dict[str, int | None] | None = None,
    registered: dict | None = None,
) -> dict:
    """C10 (A3.7a, extended by A3c.5; review 7fcd3f91ca8f MINOR-A2-1 / A2-4, review
    6267fd645698 MAJOR-2 / MINOR-5): each job ran the values the launcher recorded, the
    operator gave and the OWNER REGISTERED -- values, not flag presence -- and the
    provenance agrees. Exit 3 on any problem.

    * Registered: the PREREG's owner table (``parse_owner_table``) is ``filled``, and
      O1 / O4 / R-1 / O5 there equal the operator's values and launch-git.json's (a
      blank ``______`` fails: the run is not read before Amendment 4).
    * Launch: launch-git.json's slots equal the owner values; ckpt is the A1.8 m-9 rule
      on steps; c1_flags are O5's, c3_flags empty (O4 = 0.01 is the built value);
      rsr_orch_root empty or shared, out / job_suffix / dry_run empty; dirty false.
    * Per job, from the recorded config: `frozen_model_mode` is O5's; the optimizer's
      `weight_decay` is O4; iters is R-1; the RSR arms' label and override gamma are
      O1; `srep_norm_reg_weight` is 0.0 (A3c.5); the checkpoint files are the derived
      cadence; the heartbeat's git_sha is run_sha.
    * Per job, from the command (the job record's ``cmd``; in a non-strict fixture, the
      command `launch_direct` ran): ``--iters`` R-1, ``--seed`` the seed,
      ``--ckpt-every`` CKPT, ``--frozen-model-mode`` O5's flag, one ``gamma=`` override
      equal to O1 in the RSR arms and none in fifo, ``--srep-norm-reg-weight`` 0.0.
    * Strict (the run): each job record exists in the shared root under its exact id,
      with ``pinned_sha`` = run_sha, ``threads`` = 1 (§6, A1.6) and ``rc`` = the
      launcher's ``logs/<k>.rc``; ``cfg.owner_prereg`` is the PREREG itself."""
    ckpt = derive_ckpt(owner.iters)
    problems: list[str] = []
    rcs = rcs or {}

    def want(cond: bool, msg: str) -> None:
        if not cond:
            problems.append(msg)

    def num(x):
        if isinstance(x, bool):
            return None
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    reg = owner_table(cfg) if registered is None else registered
    if reg.get("status") != "filled":
        problems.append(
            f"owner table under {reg.get('heading')!r}: {reg.get('status')} "
            f"{reg.get('problems')} (A3.7a: the owner values are read from Amendment 4)"
        )
    else:
        rv = reg["values"]
        for name, mine, rec in (
            ("O1 gamma", owner.gamma, launch.get("gamma")),
            ("O4 wd", owner.phi_weight_decay, launch.get("wd")),
        ):
            key = "gamma" if name.startswith("O1") else "phi_weight_decay"
            want(rv[key] == mine, f"registered {name} {rv[key]!r} != given {mine!r}")
            want(rv[key] == num(rec), f"registered {name} {rv[key]!r} != launch {rec!r}")
        want(
            rv["iters"] == owner.iters,
            f"registered R-1 {rv['iters']!r} != given {owner.iters}",
        )
        want(
            str(rv["iters"]) == str(launch.get("steps")),
            f"registered R-1 {rv['iters']!r} != launch steps {launch.get('steps')!r}",
        )
        want(rv["o5"] == owner.o5, f"registered O5 {rv['o5']!r} != given {owner.o5!r}")
        want(
            rv["o5"] == launch.get("o5"),
            f"registered O5 {rv['o5']!r} != launch {launch.get('o5')!r}",
        )
    if cfg.prereg_strict:
        want(
            cfg.owner_prereg == PREREG,
            f"owner values read from {cfg.owner_prereg!r}, not the PREREG {PREREG}",
        )
    want(num(launch.get("gamma")) == owner.gamma, f"launch gamma {launch.get('gamma')!r}")
    want(
        num(launch.get("wd")) == owner.phi_weight_decay, f"launch wd {launch.get('wd')!r}"
    )
    want(
        str(launch.get("steps")) == str(owner.iters),
        f"launch steps {launch.get('steps')!r}",
    )
    want(
        str(launch.get("ckpt")) == str(ckpt),
        f"launch ckpt {launch.get('ckpt')!r} != {ckpt}",
    )
    want(launch.get("o5") == owner.o5, f"launch o5 {launch.get('o5')!r}")
    want(
        launch.get("ckpt_rule") == CKPT_RULE,
        f"launch ckpt_rule {launch.get('ckpt_rule')!r}",
    )
    steps_rec = str(launch.get("steps"))
    want(
        steps_rec.isdigit()
        and str(launch.get("ckpt")) == str(derive_ckpt(int(steps_rec))),
        f"launch ckpt {launch.get('ckpt')!r} is not the A3.7 rule applied to steps",
    )
    orch = launch.get("rsr_orch_root")
    want(
        orch is not None
        and (orch == "" or Path(orch).resolve() == Path(SHARED_ORCH_ROOT).resolve()),
        f"launch rsr_orch_root {orch!r} is not empty or the shared root",
    )
    for key in ("out", "job_suffix", "dry_run"):
        want(launch.get(key) == "", f"launch {key} {launch.get(key)!r}: a dry-run launch")
    want(
        launch.get("c1_flags", "") == " ".join(C1_FLAGS[owner.o5]),
        f"launch c1_flags {launch.get('c1_flags')!r}",
    )
    want(launch.get("c3_flags", "") == "", f"launch c3_flags {launch.get('c3_flags')!r}")
    want(
        Path(cfg.orch_root).resolve() == Path(SHARED_ORCH_ROOT).resolve()
        and cfg.job_suffix == "",
        f"job records read from {cfg.orch_root!r} with suffix {cfg.job_suffix!r}, not "
        f"the shared root under the exact ids",
    )
    sha = launch.get("run_sha")
    if cfg.prereg_strict:
        want(launch.get("dirty") is False, f"launch dirty {launch.get('dirty')!r}")
    per = {}
    for a, r in results.items():
        conf = r["config"]
        hb_sha = ((hbs[a].get("header") or {}).get("provenance") or {}).get("git_sha")
        job = jobs.get(a)
        if job is not None:
            cmd = job.get("cmd")
        elif not cfg.prereg_strict:
            cmd = cli_argv(cfg, a, owner, arm_dir(out_root, a, cfg.seed))
        else:
            cmd = None
        files = sorted(
            int(p.stem.split("-")[1])
            for p in arm_dir(out_root, a, cfg.seed).glob("ckpt-*.pt")
        )
        row = {
            "frozen_model_mode": conf.get("frozen_model_mode"),
            "weight_decay": (conf.get("optimizer") or {}).get("weight_decay"),
            "iters": conf.get("iters"),
            "label": conf.get("arm_label"),
            "gamma_override": (conf.get("policy_overrides") or {}).get("gamma"),
            "srep_norm_reg_weight": conf.get("srep_norm_reg_weight"),
            "checkpoint_steps": files,
            "heartbeat_git_sha": hb_sha,
            "job_pinned_sha": None if job is None else job.get("pinned_sha"),
            "job_threads": None if job is None else job.get("threads"),
            "job_rc": None if job is None else job.get("rc"),
            "logs_rc": rcs.get(a),
            "cmd_source": "job record"
            if job is not None
            else ("launch_direct (fixture)" if cmd else None),
            "cmd": cmd,
            "cmd_iters": _cmd_flag(cmd, "--iters"),
            "cmd_seed": _cmd_flag(cmd, "--seed"),
            "cmd_ckpt_every": _cmd_flag(cmd, "--ckpt-every"),
            "cmd_frozen_model_mode": _cmd_flag(cmd, "--frozen-model-mode"),
            "cmd_gamma": _cmd_overrides(cmd, "gamma"),
            "cmd_srep_norm_reg_weight": _cmd_flag(cmd, "--srep-norm-reg-weight"),
        }
        rp: list[str] = []
        if row["frozen_model_mode"] != FROZEN_MODE_RECORDED[owner.o5]:
            rp.append(
                f"frozen_model_mode {row['frozen_model_mode']!r} is not O5={owner.o5}"
            )
        if row["weight_decay"] != owner.phi_weight_decay:
            rp.append(f"weight_decay {row['weight_decay']!r}")
        if row["iters"] != owner.iters:
            rp.append(f"iters {row['iters']!r}")
        if a in RSR_ARMS:
            g = format(float(owner.gamma), "g")
            if (
                f"gamma={g}" not in (row["label"] or "")
                or row["gamma_override"] != owner.gamma
            ):
                rp.append(
                    f"gamma: label {row['label']!r}, override {row['gamma_override']!r}"
                )
        srep = row["srep_norm_reg_weight"]
        if not (_finite(srep) and srep == SREP_NORM_REG_WEIGHT):  # A3c.5: a number
            rp.append(f"config srep_norm_reg_weight {srep!r} != 0.0 (A3c.5)")
        if files != cfg.snapshots:
            rp.append(f"checkpoints {files} != the derived cadence {cfg.snapshots}")
        if hb_sha != sha:
            rp.append(f"heartbeat git_sha {hb_sha!r} != launch run_sha {sha!r}")
        if cmd is None:
            rp.append("no command to check (no job record)")
        else:
            if num(row["cmd_srep_norm_reg_weight"]) != SREP_NORM_REG_WEIGHT:
                rp.append(
                    f"command --srep-norm-reg-weight {row['cmd_srep_norm_reg_weight']!r}"
                    " != 0.0 (A3c.5)"
                )
            if row["cmd_iters"] != str(owner.iters):
                rp.append(f"command --iters {row['cmd_iters']!r} != {owner.iters}")
            if row["cmd_seed"] != str(cfg.seed):
                rp.append(f"command --seed {row['cmd_seed']!r} != {cfg.seed}")
            if row["cmd_ckpt_every"] != str(ckpt):
                rp.append(f"command --ckpt-every {row['cmd_ckpt_every']!r} != {ckpt}")
            fl = C1_FLAGS[owner.o5]
            want_fm = dict(zip(fl[0::2], fl[1::2], strict=True))
            if row["cmd_frozen_model_mode"] != want_fm.get("--frozen-model-mode"):
                rp.append(
                    f"command --frozen-model-mode {row['cmd_frozen_model_mode']!r} is "
                    f"not O5={owner.o5}'s"
                )
            want_g = [owner.gamma] if a in RSR_ARMS else []
            if [num(x) for x in row["cmd_gamma"]] != want_g:
                rp.append(f"command gamma overrides {row['cmd_gamma']!r} != {want_g}")
        if cfg.prereg_strict:
            if job is None:
                rp.append(
                    f"no job record {RUN_ID}-{job_key(a, cfg.seed)}{cfg.job_suffix} in "
                    f"{cfg.orch_root}"
                )
            else:
                if job.get("pinned_sha") != sha:
                    rp.append(f"job pinned_sha {job.get('pinned_sha')!r} != run_sha")
                if job.get("threads") != 1 or isinstance(job.get("threads"), bool):
                    rp.append(f"job threads {job.get('threads')!r} != 1 (§6, A1.6)")
                if (
                    job.get("rc") is None
                    or isinstance(job.get("rc"), bool)
                    or job.get("rc") != rcs.get(a)
                ):
                    rp.append(
                        f"job rc {job.get('rc')!r} != logs/{job_key(a, cfg.seed)}.rc "
                        f"{rcs.get(a)!r}"
                    )
        row["problems"] = rp
        row["ok"] = not rp
        per[a] = row
    env = launch.get("env")
    return {
        "derived_ckpt": ckpt,
        "registered": reg,
        "launch": {
            k: launch.get(k)
            for k in (
                "run_sha",
                "dirty",
                "gamma",
                "wd",
                "steps",
                "ckpt",
                "o5",
                "c1_flags",
                "c2_flags",
                "c3_flags",
                "rsr_orch_root",
                "out",
                "job_suffix",
                "dry_run",
            )
        },
        "launch_env": env,
        "jobs_read_from": {"orch_root": cfg.orch_root, "job_suffix": cfg.job_suffix},
        "launch_problems": problems,
        "per_arm": per,
        "ok": not problems and bool(per) and all(v["ok"] for v in per.values()),
    }


def c4_reference(cfg: Run1Config) -> float | None:
    """The ledger row, by key and by the seed's position in `seeds_actually_run`."""
    if cfg.c4_ledger is None:
        return None
    doc = json.loads(Path(cfg.c4_ledger).read_text())
    rows = [r for r in doc["rows"] if r.get("key") == cfg.c4_key]
    if len(rows) != 1:
        raise ControlFailure(f"{cfg.c4_ledger}: {len(rows)} rows keyed {cfg.c4_key!r}")
    val = float(rows[0]["samples"][doc["seeds_actually_run"].index(cfg.seed)])
    if cfg.c4_prereg_value is not None and val != cfg.c4_prereg_value:
        raise ControlFailure(
            f"C4 reference {val!r} is not the PREREG's {cfg.c4_prereg_value!r}"
        )
    return val


def c4_measure(fifo_dir: Path, cfg: Run1Config) -> dict:
    """C4 (T8): fresh-stream's instrument, imported unchanged, on the FIFO arm's final
    checkpoint: `measure_checkpoint(seed_dir, seed, iters, n=64)`, then `r_quantity`."""
    fs = FS()
    t0 = time.perf_counter()
    m = fs.measure_checkpoint(Path(fifo_dir), cfg.seed, cfg.iters, C4_N)
    r = float(fs.r_quantity(m["s003"]))
    ref = c4_reference(cfg)
    return {
        "R_quantity": r,
        "reference": ref,
        "abs_diff": None if ref is None else abs(r - ref),
        "tolerance": C4_TOL,
        "seconds": time.perf_counter() - t0,
        "torch_threads": torch.get_num_threads(),
        "ok": ref is not None and abs(r - ref) <= C4_TOL,
    }


# --------------------------------------------------------------------------- #
# Analysis: the paired per-document bootstrap (front matter; B2 §9.2)
# --------------------------------------------------------------------------- #


def boot_weights(n_docs: int, seed: int, n_boot: int) -> torch.Tensor:
    """B2 §9.2's resampling, drawn exactly as `B2.paired_bootstrap` draws it: one
    multiset of documents per replicate, shared by every arm (paired)."""
    g = torch.Generator().manual_seed(BOOT_SEED_BASE + seed)
    idx = torch.randint(0, n_docs, (n_boot, n_docs), generator=g)
    return torch.zeros(n_boot, n_docs, dtype=torch.float64).scatter_add_(
        1, idx, torch.ones(n_boot, n_docs, dtype=torch.float64)
    )


def pooled_replicates(per_doc: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """``per_doc`` ``[D, 2]`` = (denominator, numerator) -> pooled ratio per replicate."""
    x = per_doc.to(torch.float64)
    return (W @ x[:, 1]) / (W @ x[:, 0])


def ci_reading(lo: float | None, hi: float | None) -> str | None:
    """The only reading the PREREG allows (§7)."""
    if lo is None or hi is None:
        return None
    if lo > 0:
        return "CI above 0"
    if hi < 0:
        return "CI below 0"
    return "CI includes 0"


def _ci(rep: torch.Tensor) -> dict:
    ok = torch.isfinite(rep)
    r = rep[ok]
    if r.numel() == 0:
        return {"lo": None, "hi": None, "sd": None, "dropped": int((~ok).sum())}
    return {
        "lo": float(torch.quantile(r, 0.025)),
        "hi": float(torch.quantile(r, 0.975)),
        "sd": float(r.std()) if r.numel() > 1 else None,
        "dropped": int((~ok).sum()),
    }


def _point(x: torch.Tensor) -> float | None:
    den = float(x[:, 0].sum())
    return None if den == 0 else float(x[:, 1].sum()) / den


def contrast(per_doc: dict[str, torch.Tensor], a: str, b: str, W: torch.Tensor) -> dict:
    pa, pb = _point(per_doc[a]), _point(per_doc[b])
    c = _ci(pooled_replicates(per_doc[a], W) - pooled_replicates(per_doc[b], W))
    c["point"] = None if pa is None or pb is None else pa - pb
    c["reading"] = ci_reading(c["lo"], c["hi"])
    c["sd_exactly_zero"] = c["sd"] == 0.0
    return c


def level(per_doc: torch.Tensor, W: torch.Tensor) -> dict:
    c = _ci(pooled_replicates(per_doc, W))
    c["point"] = _point(per_doc)
    c["n"] = int(per_doc[:, 0].sum())
    return c


def share(per_doc: dict[str, torch.Tensor], arm: str, W: torch.Tensor) -> dict:
    """§7.2 headroom share; a replicate with a zero denominator is dropped, counted."""
    pa, pf, po = (_point(per_doc[k]) for k in (arm, "fifo", "oracle"))
    point = (pa - pf) / (po - pf) if None not in (pa, pf, po) and po != pf else None
    ra, rf, ro = (pooled_replicates(per_doc[k], W) for k in (arm, "fifo", "oracle"))
    c = _ci((ra - rf) / (ro - rf))
    c["point"] = point
    return c


def answer_counts(records: list[dict], sel) -> torch.Tensor:
    """``[D, 2]`` = (answers in the bucket, correct) per document, in record order."""
    rows = []
    for r in records:
        if not r["answers"]:
            rows.append([0.0, 0.0])
            continue
        g = torch.tensor([a[1] for a in r["answers"]], dtype=torch.long)
        ok = torch.tensor([a[2] for a in r["answers"]], dtype=torch.bool)
        m = sel(g)
        rows.append([float(m.sum()), float(ok[m].sum())])
    return torch.tensor(rows, dtype=torch.float64).reshape(-1, 2)


def loss_counts(records: list[dict], kind: str) -> torch.Tensor:
    """``[D, 2]`` = (targets, NLL sum): real-token (`real`) or answer-token (`answer`)."""
    if kind == "real":
        rows = [[r["real_n"], r["real_sum"]] for r in records]
    else:
        rows = [[len(r["answers"]), sum(a[3] for a in r["answers"])] for r in records]
    return torch.tensor(rows, dtype=torch.float64).reshape(-1, 2)


def analyse_heldout(out_dir: Path, cfg: Run1Config, passes) -> dict:
    """§7.1 and §7.2 from one seed's per-document records."""
    names = [p[0] for p in passes]
    recs = {n: read_pass(out_dir, n) for n in names}
    order = [r["doc"] for r in recs["fifo"]]
    for n, rs in recs.items():
        if [r["doc"] for r in rs] != order:
            raise ControlFailure(
                f"held-out pass {n} is not in the fifo pass's document order"
            )
    W = boot_weights(len(order), cfg.seed, cfg.n_boot)
    buckets = buckets_for(cfg.memory_slots)
    rnd = [f"random{k}" for k in range(N_RANDOM)]
    snaps = [n for n in names if "@" in n]
    out: dict[str, Any] = {
        "n_documents": len(order),
        "n_boot": cfg.n_boot,
        "boot_seed": BOOT_SEED_BASE + cfg.seed,
        "oracle_gamma": ORACLE_GAMMA,
    }
    acc: dict[str, dict] = {}
    con: dict[str, dict] = {}
    for bname in BUCKET_NAMES:
        per = {n: answer_counts(recs[n], buckets[bname]) for n in names}
        acc[bname] = {n: level(per[n], W) for n in names}
        rp = [_point(per[k]) for k in rnd]
        acc[bname]["random"] = {
            **_ci(torch.stack([pooled_replicates(per[k], W) for k in rnd]).mean(0)),
            "point": None if None in rp else sum(rp) / len(rp),
            "how": "mean over random0..4 of their pooled accuracies (B2 §9.2)",
        }
        for cname, (a, b) in CONTRASTS.items():
            con[f"acc.{bname}.{cname}"] = contrast(per, a, b, W)
        for arm in ARMS:
            con[f"oracle_regret.{bname}.{arm}"] = contrast(per, "oracle", arm, W)
        if bname == "gap_2_to_M":  # §7.2 R: acc_live(arm) - acc_slots_zeroed
            for arm in ARMS:
                con[f"R.{arm}"] = contrast(per, arm, "sz_fifo", W)
                con[f"R_own_floor.{arm}"] = contrast(per, arm, f"sz_{arm}", W)
            for arm in RSR_ARMS:
                key = f"floor_difference.sz_{arm}_minus_sz_fifo"
                con[key] = contrast(per, f"sz_{arm}", "sz_fifo", W)
        if bname == "all":
            out["headroom_share"] = {a: share(per, a, W) for a in RSR_ARMS}
            out["snapshots"] = {
                s: {
                    "level": acc[bname][s],
                    "minus_final_fifo": contrast(per, s, "fifo", W),
                }
                for s in snaps
            }
    out["accuracy"] = acc
    out["nll"] = {}
    for kind in ("real", "answer"):
        per = {n: loss_counts(recs[n], kind) for n in names}
        out["nll"][kind] = {n: level(per[n], W) for n in names}
        for cname, (a, b) in CONTRASTS.items():
            con[f"nll_{kind}.{cname}"] = contrast(per, a, b, W)
    out["contrasts"] = con
    ev = {}
    for n in (*ARMS, "oracle", "lru"):
        ages: Counter = Counter()
        disp: Counter = Counter()
        for r in recs[n]:
            for _t, age, dsp in r["evictions"]:
                ages[age] += 1
                disp[dsp] += 1
        tot = sum(disp.values())
        ev[n] = {
            "n_evictions": tot,
            "victim_age_distribution": dict(sorted(ages.items())),
            "displacement_distribution": dict(sorted(disp.items())),
            "fraction_oldest": (disp.get(0, 0) / tot) if tot else None,
        }
    out["heldout_evictions"] = ev
    fills = [n / cfg.memory_slots for r in recs["fifo"] for n in r["n_live_before"]]
    out["fill_fraction_heldout"] = sum(fills) / len(fills) if fills else None
    return out


# --------------------------------------------------------------------------- #
# Analysis: training time (§7.3) and the §7.1 diagnostics (§7.4, A1.8 m-2)
# --------------------------------------------------------------------------- #


def _summary(xs) -> dict:
    xs = [float(x) for x in xs if _finite(x)]
    if not xs:
        return {
            "n": 0,
            "mean": None,
            "sd": None,
            "median": None,
            "q25": None,
            "q75": None,
        }
    t = torch.tensor(xs, dtype=torch.float64)
    sd = float(t.std()) if len(xs) > 1 else None
    return {
        "n": len(xs),
        "mean": float(t.mean()),
        "sd": sd,
        "sd_exactly_zero": sd == 0.0,
        "median": float(t.quantile(0.5)),
        "q25": float(t.quantile(0.25)),
        "q75": float(t.quantile(0.75)),
        "min": float(t.min()),
        "max": float(t.max()),
    }


def _blocks(pairs, lo: int, hi: int, width: int = DELTA_BLOCK):
    out = []
    start = (lo // width) * width
    while start < hi:
        a, b = max(start, lo), min(start + width, hi)
        xs = [v for k, v in pairs if a <= k < b and _finite(v)]
        out.append(
            {"steps": [a, b], "n": len(xs), "mean": sum(xs) / len(xs) if xs else None}
        )
        start += width
    return out


def _mc(b: dict, key: str):
    return (b.get("mc") or {}).get(key)


def _g_by_t(beats: list[dict]) -> dict[int, float]:
    acc: dict[int, list[float]] = {}
    for b in beats:
        for t, v in (_mc(b, "g_mean_by_t") or {}).items():
            acc.setdefault(int(t), []).append(float(v))
    return {t: sum(v) / len(v) for t, v in sorted(acc.items())}


def analyse_training(
    hbs: dict[str, dict], diags: dict[str, list[dict]], cfg: Run1Config
) -> dict:
    """§7.3 (paired per step; in-sample for phi; SECONDARY) and §7.4 (DIAGNOSTIC, split
    warm / scored), plus A1.8 m-2's first scored block beside uniform random."""
    tw = cfg.t_warm_steps
    lm = {a: {b["step"]: b["loss_lm"] for b in hbs[a]["beats"]} for a in hbs}
    out: dict[str, Any] = {"t_warm_steps": tw, "iters": cfg.iters, "delta_loss_lm": {}}
    for a in RSR_ARMS:
        if a in lm and "fifo" in lm:
            pairs = [
                (k, lm[a][k] - lm["fifo"][k])
                for k in sorted(lm[a])
                if k in lm["fifo"] and k >= tw
            ]
            out["delta_loss_lm"][a] = {
                **_summary([v for _k, v in pairs]),
                "blocks": _blocks(pairs, math.ceil(tw), cfg.iters),
                "how": f"loss_lm(arm, k) - loss_lm(fifo, k), k in [{tw}, {cfg.iters})",
            }
    out["loss_lm"] = {a: _summary(list(v.values())) for a, v in lm.items()}
    out["rsr"] = {}
    for a in RSR_ARMS:
        if a not in hbs:
            continue
        beats = hbs[a]["beats"]
        warm = [b for b in beats if b["step"] < tw]
        post = [b for b in beats if b["step"] >= tw]
        fills = [
            _mc(b, "n_samples") / (_mc(b, "n_groups") * cfg.memory_slots)
            for b in beats
            if _mc(b, "n_groups")
        ]
        out["rsr"][a] = {
            "loss_mc_warm": _summary([b.get("loss_mc") for b in warm]),
            "loss_mc_post_warm": _summary([b.get("loss_mc") for b in post]),
            "loss_mc_blocks": _blocks(
                [(b["step"], b.get("loss_mc")) for b in beats], 0, cfg.iters
            ),
            "g_mean_by_t_warm": _g_by_t(warm),
            "g_mean_by_t_post_warm": _g_by_t(post),
            "corr_psi_age_warm": _summary([_mc(b, "corr_psi_age") for b in warm]),
            "corr_psi_age_post_warm": _summary([_mc(b, "corr_psi_age") for b in post]),
            "corr_psi_age_none_steps": sum(
                1 for b in beats if _mc(b, "corr_psi_age") is None
            ),
            "phi_grad_norm": _summary([b.get("phi_grad_norm") for b in beats]),
            "eval_forward_s": {
                **_summary([b.get("eval_forward_s") for b in beats]),
                "total": sum(b.get("eval_forward_s") or 0.0 for b in beats),
            },
            "fill_fraction_regression_rows": sum(fills) / len(fills) if fills else None,
        }
    for a in RSR_ARMS:
        if a in hbs and a in out["rsr"]:
            for phase, bs in (
                ("warm", [b for b in hbs[a]["beats"] if b["step"] < tw]),
                ("post_warm", [b for b in hbs[a]["beats"] if b["step"] >= tw]),
            ):
                tot: Counter = Counter()
                n = 0
                for b in bs:
                    if b.get("attribution"):
                        tot.update(b["attribution"])
                        n += 1
                out["rsr"][a][f"attribution_{phase}"] = (
                    {k: v / n for k, v in sorted(tot.items())} if n else None
                )
    out["step_s"] = {}
    for a, hb in hbs.items():
        el = [b["elapsed_s"] for b in hb["beats"]]
        out["step_s"][a] = _summary([y - x for x, y in itertools.pairwise(el)])
    out["diagnostics"] = {}
    for a, rows in diags.items():
        out["diagnostics"][a] = {}
        lo = math.ceil(tw)
        for phase, sel in (
            ("warm", lambda r: r["warm"]),
            ("scored", lambda r: not r["warm"]),
            (
                "first_scored_block",
                lambda r, lo=lo: (
                    not r["warm"] and lo <= r["step"] < lo + FIRST_SCORED_STEPS
                ),
            ),
        ):
            rs = [r for r in rows if sel(r)]
            shift: Counter = Counter()
            ages: Counter = Counter()
            for r in rs:
                shift.update(r["shift"])
                ages.update(r["victim_age"])
            tot = sum(shift.values())
            row = {
                "n_steps": len(rs),
                "n_evictions": tot,
                "victim_age_distribution": dict(sorted(ages.items())),
                "displacement_distribution": dict(sorted(shift.items())),
                "fraction_oldest": (shift.get(0, 0) / tot) if tot else None,
                "fraction_shifting": (1 - shift.get(0, 0) / tot) if tot else None,
                "mean_shift": (sum(k * v for k, v in shift.items()) / tot)
                if tot
                else None,
                "spearman_rho_score_age": _summary([r["spearman_rho"] for r in rs]),
                "partial_rho_diagnostic": _summary([r["partial_rho"] for r in rs]),
                "partial_rho_none_steps": sum(1 for r in rs if r["partial_rho"] is None),
                "warm_flag_matches_step": all((r["step"] < tw) is r["warm"] for r in rs),
            }
            if phase == "first_scored_block":
                m = cfg.memory_slots
                row["uniform_random_expected"] = {k: tot / m for k in range(m)}
                row["steps"] = [lo, lo + FIRST_SCORED_STEPS]
            out["diagnostics"][a][phase] = row
    return out


# --------------------------------------------------------------------------- #
# measure-seed, then finalize
# --------------------------------------------------------------------------- #


def measure_seed(
    cfg: Run1Config,
    owner: OwnerValues,
    out_root: Path,
    *,
    threads: int,
    limit: int | None = None,
    c4_fn=None,
    environ=None,
    dry_run: bool = False,
) -> dict:
    """Every control and number for one seed -> ``measure-seed{s}.json``. A failed
    control is recorded and the status says so; nothing is dropped."""
    torch.set_num_threads(threads)
    out_root = Path(out_root)
    env = launch_env(environ)
    if dry_run:  # A3c.1's dry run: its isolated root and job-id suffix are read (MINOR-4)
        cfg = dataclasses.replace(
            cfg,
            orch_root=env.get("RSR_ORCH_ROOT") or cfg.orch_root,
            job_suffix=env.get("JOB_SUFFIX") or "",
        )
    doc: dict[str, Any] = {
        "run_id": RUN_ID,
        "seed": cfg.seed,
        "config": cfg.as_dict(),
        "owner_values": dataclasses.asdict(owner),
        "tree": tree_state(),
        "hardware": hardware(),
        "measure_threads": torch.get_num_threads(),
        "measure_env": env,
        "controls": {},
        "scoped_controls": {},
        "status": None,
        "exit": None,
        "notes": [],
    }

    def done(code: Exit, status: str) -> dict:
        failed = [k for k, v in doc["controls"].items() if v.get("ok") is not True]
        doc.update(status=status, exit=int(code), controls_failed=failed)
        # A3b.3 / MAJOR-1: as recorded, per arm (set when the arms are read); a seed
        # whose launcher never ran has none, and is not a seed that ran.
        doc.setdefault("steps_done", {})
        doc.setdefault("seeds_actually_run", [])
        out_root.mkdir(parents=True, exist_ok=True)
        f = out_root / f"measure-seed{cfg.seed}.json"
        f.write_text(json.dumps(doc, indent=2, default=str) + "\n")
        return doc

    doc["dry_run"] = dry_run
    if cfg.prereg_strict and env_problems(env, dry_run=dry_run):
        doc["notes"].append(f"environment: {env_problems(env, dry_run=dry_run)}")
        return done(Exit.DID_NOT_RUN, "did_not_run")
    try:
        doc["controls"]["C0"] = check_c0(cfg)
    except ControlFailure as e:
        doc["controls"]["C0"] = {"ok": False, "error": str(e)}
        return done(Exit.DID_NOT_RUN, "did_not_run")
    lg = out_root / "launch-git.json"
    launch = json.loads(lg.read_text()) if lg.exists() else None
    doc["launch"] = launch
    if launch is None:
        doc["notes"].append(f"{lg} is missing: the launcher never ran here")
        return done(Exit.DID_NOT_RUN, "did_not_run")
    arms: dict[str, dict] = {}
    for a in ARMS:
        k = job_key(a, cfg.seed)
        rcf, logf = out_root / "logs" / f"{k}.rc", out_root / "logs" / f"{k}.log"
        rc = int(rcf.read_text().strip()) if rcf.exists() else None
        result = read_cli_result(logf)
        hb = read_heartbeat(arm_dir(out_root, a, cfg.seed))
        job = job_record(cfg, a)
        arms[a] = {
            "rc": rc,
            "result": result,
            "hb": hb,
            "job": job,
            "meaning": classify_rc(rc, job, hb, result),
            "log": str(logf),
        }
    doc["arms"] = {
        a: {
            "rc": v["rc"],
            "rc_meaning": v["meaning"],
            "log": v["log"],
            "run_id": (v["result"] or {}).get("run_id"),
            "config_hash": (v["result"] or {}).get("config_hash"),
            "arm_label": (v["result"] or {}).get("arm_label"),
            # The command as the job record holds it, verbatim (review NIT); the
            # reconstruction only where there is no record (the fixture).
            "argv": (v["job"] or {}).get("cmd")
            or cli_argv(cfg, a, owner, arm_dir(out_root, a, cfg.seed)),
            "argv_source": "job record" if (v["job"] or {}).get("cmd") else "cli_argv",
            "steps_done": arm_steps_done(v["rc"], v["hb"], v["result"]),
            "job_record": v["job"],
            "heartbeat_last": v["hb"]["last"],
        }
        for a, v in arms.items()
    }
    arm_status = {a: arm_outcome(arms[a]["rc"], arms[a]["hb"], arms[a]["result"])
                  for a in ARMS}  # fmt: skip
    doc["arm_outcomes"] = arm_status
    doc["steps_done"] = {
        a: v["steps_done"] for a, v in doc["arms"].items() if v["steps_done"] is not None
    }
    doc["seeds_actually_run"] = (
        [cfg.seed] if any(n >= 1 for n in doc["steps_done"].values()) else []
    )
    worst = worst_outcome([arm_status[a] for a in ARMS])
    if worst is not None:
        for a in ARMS:  # A3c.2: the notes list EVERY non-OK arm
            if arm_status[a][0] != int(Exit.OK):
                doc["notes"].append(f"{arm_status[a][2]}: {a}")
        # A3b.3 + A3b.5 + A3c.2: the worst arm by 5 > 3 > 1 > 0; a tie goes to the
        # first such arm in (psiC, fifo, age).
        code, status, _note = worst
        return done(Exit(code), status)
    results = {a: v["result"] for a, v in arms.items()}
    hbs = {a: v["hb"] for a, v in arms.items()}
    doc["frozen_model_mode"] = {
        a: r["config"].get("frozen_model_mode") for a, r in results.items()
    }
    doc["controls"]["C1"] = c1_warm_start(results, cfg)
    doc["controls"]["C5"] = c5_liveness(results)
    doc["controls"]["C6"] = c6_steps(results, hbs, cfg)
    doc["controls"]["C10"] = c10_launch_values(
        cfg,
        owner,
        launch,
        results,
        hbs,
        {a: v["job"] for a, v in arms.items()},
        out_root,
        rcs={a: v["rc"] for a, v in arms.items()},
    )
    # Review de8f813 MINOR-A3c-1: the seed takes the WORST of everything read so far,
    # not whichever check comes first. Candidates, in this order (the first wins a tie
    # of exits -- an order this harness chose; A3c.2 registers one only for arms):
    # C5 inert (5), C6 (3, crashed), C1 / C10 (3, CONTROL_FAILED), C5 inconclusive (1;
    # A3c.4). Any C5 / C6 finding stops the seed here; a C1 / C10 failure alone lets
    # the numbers be written, marked not read (A3b.3), and is applied at the end.
    c5 = doc["controls"]["C5"]
    unsure = [a for a, v in c5["per_arm"].items() if v["band"] == "inconclusive"]
    early_failed = [k for k in ("C1", "C10") if doc["controls"][k].get("ok") is not True]
    cands: list[tuple[int, str, str]] = []
    if c5["inert"]:
        cands.append((int(Exit.INERT), "partial", f"INERT: {c5['inert']}"))
    if not doc["controls"]["C6"]["ok"]:
        cands.append(
            (
                int(Exit.DID_NOT_RUN),
                "crashed",
                f"ARM_CRASHED: C6 {doc['controls']['C6']['per_arm']}",
            )
        )
    if early_failed:
        cands.append(
            (int(Exit.DID_NOT_RUN), "did_not_run", f"CONTROL_FAILED: {early_failed}")
        )
    if unsure:
        cands.append((int(Exit.FAIL), "failed", f"LIVENESS_INCONCLUSIVE: {unsure}"))
    stop = bool(c5["inert"] or unsure or not doc["controls"]["C6"]["ok"])
    if cfg.iters == 0:  # a dry run (A3c.1: STEPS = 0): nothing trained, nothing to read
        stop = True
        doc["notes"].append("DRY_RUN: STEPS 0, no step trained; no held-out pass")
        if not cands:
            cands.append((int(Exit.DID_NOT_RUN), "did_not_run", "DRY_RUN"))
    if stop:
        doc["notes"] += [c[2] for c in cands if c[2] != "DRY_RUN"]
        code, status, _note = worst_outcome(cands)
        return done(Exit(code), status)
    beats = {a: hbs[a]["beats"] for a in ARMS}
    steprecs = {a: read_step_records(arm_dir(out_root, a, cfg.seed)) for a in ARMS}
    samplings = {a: arm_sampling(results[a]) for a in ARMS}
    doc["controls"]["C2"] = c2_pairing(
        beats,
        {a: step_pool_ids(steprecs[a]) for a in ARMS},
        cfg,
        samplings["psiC"],
        samplings=samplings,
    )
    doc["controls"]["C8"] = c8_finite(beats)
    payloads = {
        a: read_checkpoint(checkpoint_path(arm_dir(out_root, a, cfg.seed), cfg.iters))
        for a in ARMS
    }
    snaps = [k for k in cfg.snapshots if k != cfg.iters]
    for k in snaps:
        for a in RSR_ARMS:
            payloads[f"{a}@{k}"] = read_checkpoint(
                checkpoint_path(arm_dir(out_root, a, cfg.seed), k)
            )
    sub_model, sub_sha = load_substrate_model(cfg)
    doc["controls"]["C3"] = {
        **c3_frozen(payloads, sub_model),
        "substrate_sha256": sub_sha,
    }
    del sub_model
    doc["checkpoint_bytes"] = {
        a: checkpoint_path(arm_dir(out_root, a, cfg.seed), cfg.iters).stat().st_size
        for a in ARMS
    }
    vmap = vocab_map(cfg)
    fifo_model = model_from_payload(payloads["fifo"]).to(cfg.device)
    doc["scoped_controls"]["C9_C9b"] = c9_oracle(fifo_model, cfg, vmap)
    passes = heldout_passes(snaps)
    hd = out_root / "heldout" / f"seed{cfg.seed}"
    ho = run_heldout(cfg, payloads, hd, passes=passes, limit=limit)
    ok7 = all(v["n_mismatched"] == 0 for v in ho["residency"].values())
    doc["controls"]["C7"] = {"per_pass": ho["residency"], "ok": ok7}
    n_batches = math.ceil((cfg.eval_range[1] - cfg.eval_range[0]) / cfg.eval_batch)
    doc["T5"] = {
        n: {
            "first_batch_s": t["first_batch_s"],
            "batches_run": t["batches"],
            "projected_full_eval_s": None
            if t["first_batch_s"] is None
            else t["first_batch_s"] * n_batches,
            "measured_total_s": t["total_s"],
        }
        for n, t in ho["timing"].items()
    }
    try:
        fifo_dir = arm_dir(out_root, "fifo", cfg.seed)
        doc["controls"]["C4"] = (c4_fn or c4_measure)(fifo_dir, cfg)
    except Exception as e:  # the control did not run: recorded, never a pass
        doc["controls"]["C4"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    doc["heldout"] = analyse_heldout(hd, cfg, passes)
    doc["oracle_numbers_read"] = doc["scoped_controls"]["C9_C9b"]["ok"]
    diags = {a: step_diagnostics(steprecs[a]) for a in RSR_ARMS}
    doc["training"] = analyse_training(hbs, diags, cfg)
    doc["training"]["run_level_histograms_agree"] = {
        a: run_level_agrees(
            results[a].get("diagnostics"), doc["training"]["diagnostics"][a]
        )
        for a in RSR_ARMS
    }
    doc["epoch"] = {a: results[a]["config"].get("epoch") for a in RSR_ARMS}
    failed = [k for k, v in doc["controls"].items() if v.get("ok") is not True]
    if failed:
        # A3b.3 / R-2026-09-22-inert-exit-5: "a control fails -> 3 DID_NOT_RUN". Training
        # ran: the note and steps_done / seeds_actually_run say so, and stay.
        doc["notes"].append(f"CONTROL_FAILED: {failed}")
        return done(Exit.DID_NOT_RUN, "did_not_run")
    return done(Exit.OK, "ok")


def run_level_agrees(phase_summary: dict | None, per_step: dict) -> bool | None:
    """ADAPTER (c02b16d's `phase_summary`): train()'s run-level displacement histograms,
    split warm / scored, equal the sums of the per-step ones (DIAGNOSTIC)."""
    if not phase_summary or "scored" not in phase_summary:
        return None
    for key, phase in (("fifo_warmup", "warm"), ("scored", "scored")):
        dist = phase_summary[key].get("shift_distribution") or {}
        if {int(k): int(v) for k, v in dist.items()} != per_step[phase][
            "displacement_distribution"
        ]:
            return False
    return True


def headline(docs: list[dict]) -> dict:
    """A1.4: the ONE contrast, per seed (within-seed CI) and across seeds (the points,
    their mean and sd, and how many are above / below / exactly 0; no CI, no test)."""
    per = {}
    for d in docs:
        c = ((d.get("heldout") or {}).get("contrasts") or {}).get(HEADLINE)
        per[d["seed"]] = None if c is None else {**c, "ci_scope": "within-seed only"}
    pts = [
        v["point"] for v in per.values() if v is not None and v.get("point") is not None
    ]
    n = len(pts)
    mean = sum(pts) / n if n else None
    sd = math.sqrt(sum((x - mean) ** 2 for x in pts) / (n - 1)) if n > 1 else None
    signs = {
        "above_0": sum(x > 0 for x in pts),
        "below_0": sum(x < 0 for x in pts),
        "exactly_0": sum(x == 0 for x in pts),
    }
    return {
        "contrast": HEADLINE,
        "per_seed": per,
        "across_seeds": {
            "n_seeds": n,
            "points": pts,
            "mean": mean,
            "sd": sd,
            "sd_exactly_zero": sd == 0.0,
            "sign_count": signs,
        },
    }


def finalize(
    docs: list[dict],
    owner: OwnerValues,
    out_root: Path,
    *,
    results_path: Path,
    write_ledger: bool = True,
) -> Exit:
    """Across seeds: the headline, RESULTS.md, the ledger. The exit is the worst seed's
    (INERT 5 over 3 over 0)."""
    # A3b.5: the worst seed by 5 > 3 > 1 > 0, ties to the lowest seed; both from it.
    worst = max(docs, key=lambda d: (EXIT_RANK[int(d["exit"])], -int(d["seed"])))
    code, status = Exit(worst["exit"]), worst["status"]
    run = {
        "run_id": RUN_ID,
        "seeds": [d["seed"] for d in docs],
        "exit": int(code),
        "status": status,
        "owner_values": dataclasses.asdict(owner),
        "headline": headline(docs),
        "tree": tree_state(),
        "prereg_strict": all(d["config"].get("prereg_strict", True) for d in docs),
    }
    (Path(out_root) / "run.json").write_text(
        json.dumps(run, indent=2, default=str) + "\n"
    )
    Path(results_path).write_text(render_results(run, docs))
    if write_ledger:
        write_run_ledger(run, docs, Path(out_root))
    return code


def write_run_ledger(run: dict, docs: list[dict], out_root: Path) -> Path:
    """Every number into `runs/rsr-run1/ledger.json` (scripts/ledger.py). The manifest is
    the launcher's `launch-git.json`, frozen before any slot call (A1.5)."""
    L = ledger_mod()
    led = L.Ledger(RUN_ID, question=QUESTION, runs_root=Path(out_root).parent)
    lg = Path(out_root) / "launch-git.json"
    if lg.exists():
        led.doc["config_hash"] = hashlib.sha256(lg.read_bytes()).hexdigest()
        led.doc["manifest_written_utc"] = json.loads(lg.read_text()).get("launched_utc")
    for d in docs:
        for a, r in (d.get("arms") or {}).items():
            note = f"arm {a} seed {d['seed']}: {r.get('rc_meaning')}"
            led.command(r["argv"], exit_code=r.get("rc"), note=note)
    led.command(
        [".venv/bin/python", EXPERIMENT, "finalize"],
        exit_code=run["exit"],
        note="this measurement",
    )
    # A3b.3 / review 6267fd645698 MAJOR-1: what was RECORDED, never cleared and never
    # read off C6 alone. Per job, the steps it recorded (`arm_steps_done`); a seed ran
    # only if one of its jobs recorded a step; `steps_done` is the least-trained job's
    # count, and the per-job counts are a row of their own.
    by_job = {
        f"{a}-s{d['seed']}": n
        for d in docs
        for a, n in sorted((d.get("steps_done") or {}).items())
    }
    ran = sorted({s for d in docs for s in d.get("seeds_actually_run") or []})
    led.run_meta(
        device=DEVICE,
        seeds_actually_run=ran,
        steps_requested=run["owner_values"]["iters"],
        steps_done=min(by_job.values()) if by_job else None,
    )
    led.note(
        "steps_done_by_job",
        by_job,
        how=f"{EXPERIMENT} measure-seed<s>.json steps_done (arm_steps_done: the printed "
        "result, else the last heartbeat beat + 1, else 0 for a process that ran)",
    )
    led.note(
        "prereg_layers",
        {
            "prereg": PREREG,
            "commits": PREREG_COMMITS,
            "reviews": list(REVIEW_COMMITS),
            "launcher": LAUNCHER_HEADING,
        },
        how=f"{EXPERIMENT} PREREG_COMMITS / REVIEW_COMMITS / LAUNCHER_HEADING",
    )
    led.note("headline", run["headline"], how=f"{EXPERIMENT} run.json")
    keys = (
        "controls",
        "scoped_controls",
        "heldout",
        "training",
        "T5",
        "checkpoint_bytes",
        "epoch",
        "arms",
        "launch",
        "oracle_numbers_read",
    )
    for d in docs:
        for key, val in _flatten(f"seed{d['seed']}", {k: d.get(k) for k in keys}):
            how = f"{EXPERIMENT} measure-seed{d['seed']}.json"
            unread = not d.get("oracle_numbers_read", True) and _oracle_derived(key)
            if unread:
                how = f"NOT READ (C9/C9b failed, A3.6). {how}"
            led.note(key, val, how=how)
            if unread:  # A3b.4: the value stays; the row says it is not read
                led.doc["rows"][-1].update(not_read=True, not_read_reason="C9/C9b failed")
    led.status(run["status"])
    return led.write()


#: A3b.4's three families, exactly (review 6267fd645698 MINOR-3): the oracle's own
#: accuracy level `acc.<bucket>.oracle`, `oracle_regret.*` and `headroom_share.*` -- in
#: the held-out numbers only. Never a control row (A3.6: "not ... any control") and
#: never the oracle's NLL.
_NOT_READ_KEYS = (
    re.compile(r"seed\d+\.heldout\.accuracy\.[^.]+\.oracle"),
    re.compile(r"seed\d+\.heldout\.contrasts\.oracle_regret\..+"),
    re.compile(r"seed\d+\.heldout\.headroom_share(\..+)?"),
)


def _oracle_derived(key: str) -> bool:
    """A3.6 / A3b.4: the ledger keys a C9/C9b failure marks NOT READ."""
    return any(r.fullmatch(key) for r in _NOT_READ_KEYS)


def _flatten(prefix: str, x: Any, depth: int = 0):
    if isinstance(x, dict) and depth < 4:
        for k, v in x.items():
            yield from _flatten(f"{prefix}.{k}", v, depth + 1)
    else:
        yield prefix, x


# --------------------------------------------------------------------------- #
# RESULTS.md
# --------------------------------------------------------------------------- #


def prereg_quote(after: str, start: str, stop: str) -> str:
    """A PREREG passage, read from the file (quoted, never retyped): from ``start`` to
    ``stop``, searching from the heading ``after``."""
    text = (ROOT / PREREG).read_text()
    i = text.index(start, text.index(after))
    return text[i : text.index(stop, i + len(start))].rstrip()


def scope_sentence() -> str:
    """A2.6's sentence, verbatim (it replaced A1.4's)."""
    return prereg_quote("### A2.6", "> This interval resamples", "\n\n")


def confound_text() -> str:
    """A2.1 item 2's named confound, verbatim, for O5 = train."""
    return prereg_quote("### A2.1", "2. **A named confound", "\n   - RESULTS prints")


def caveat_text() -> str:
    """§7.5's censored-target caveat, verbatim."""
    return prereg_quote(
        "### 7.5", "  > The shadow buffer is off", "- Every number is reported"
    )


def expectation_text() -> str:
    return prereg_quote(
        "## 7. Outcomes", "### 7.6 Author's expectation", "## 8. Controls"
    )


MULTIPLICITY = (
    "SECONDARY and DIAGNOSTIC intervals are unadjusted for multiplicity (A1.4); only the "
    "headline is designated."
)
LRU_CAVEAT = (
    "`LRUPolicy` falls back to the write step for a never-attended slot, so it degrades "
    "toward FIFO by design; its distance from FIFO is `acc.*.lru_minus_fifo` (A1.1)."
)
ATTRIBUTION_NOTE = (
    "The heartbeat's `attribution` is recorded for the RSR arms (RSRPolicy.attribution, "
    "feat/rsr-run1-ready c02b16d) and is None for FIFO, which has no terms. With b off "
    "and nu = 0 the scored decomposition has a single term, so the attribution only "
    "separates FIFO-warmup evictions from scored ones (A1.8 m-6's text predates "
    "c02b16d). Per-arm fractions are in the training section (DIAGNOSTIC)."
)
TIMING_CAVEAT = (
    "The §10 step_s projections were measured with one run-1 process beside the battery; "
    "per-step cost with ten concurrent single-thread processes was not measured before "
    "the run. The measured step_s and wall-clock are below (A1.6)."
)
BROKEN = " **sd exactly 0: broken, not clean**"


def _f(x, nd: int = 4) -> str:
    if x is None:
        return "None"
    if isinstance(x, (bool, int)):
        return str(x)
    if isinstance(x, float):
        return "nan" if x != x else f"{x:+.{nd}f}"
    return str(x)


def _ci_str(c: dict | None, *, reading: bool = True) -> str:
    if not c:
        return "not measured"
    flag = BROKEN if c.get("sd_exactly_zero") else ""
    rd = f" {c.get('reading')}" if reading and c.get("reading") else ""
    lo, hi = _f(c.get("lo")), _f(c.get("hi"))
    return f"{_f(c.get('point'))} [{lo}, {hi}]{rd}{flag}"


def _join(*parts: object) -> str:
    return "".join(str(p) for p in parts)


def _psic_confound(owner: dict) -> str:
    return (
        " -- **TRAIN/EVAL MEMORY SHIFT (psiC only)**"
        if owner.get("o5") == "train"
        else ""
    )


def render_results(run: dict, docs: list[dict]) -> str:
    """RESULTS.md (§14b as amended). Every number is read from `run.json` and the
    `measure-seed*.json` documents; nothing is typed."""
    L: list[str] = []
    ow = run["owner_values"]
    w = _confound_tagger(L.append, _psic_confound(ow))  # A2.1: every psiC line
    w("# RESULTS: RSR run 1 -- psi_hat trained online on frozen arm B ckpt3000\n")
    w(
        "**Descriptive. No gate, no pass/fail threshold.** Every contrast is read only "
        "as CI above 0 / CI below 0 / CI includes 0.\n"
    )
    if not run.get("prereg_strict", True):
        w("> **TEST FIXTURE (prereg_strict False): not a run-1 result.**\n")
    if run["status"] != "ok":
        w(
            _join(
                "> 🔴 **NOT READ.** status `",
                run["status"],
                "`, exit `",
                run["exit"],
                "`. Every number below is reported and none is read.\n",
            )
        )
    w("## 1. HEADLINE (the one designated contrast)\n")
    h = run["headline"]
    w(
        f"`{h['contrast']}`: psiC - FIFO, all-query held-out answer accuracy"
        f"{_psic_confound(ow)}.\n"
    )
    w(scope_sentence() + "\n")
    for s, c in h["per_seed"].items():
        w(f"- seed {s}: {_ci_str(c)} (within-seed){_psic_confound(ow)}")
    a = h["across_seeds"]
    pts = [round(x, 6) for x in a["points"]]
    w(
        _join(
            "- across seeds: n = ",
            a["n_seeds"],
            ", points ",
            pts,
            ", mean ",
            _f(a["mean"]),
            ", sd ",
            _f(a["sd"]),
            BROKEN if a["sd_exactly_zero"] else "",
            ", sign count ",
            a["sign_count"],
            " (no CI across seeds, no test)",
            _psic_confound(ow),
            "\n",
        )
    )
    if ow.get("o5") == "train":
        w("**Named confound (O5 = train), verbatim from A2.1:**\n")
        w(confound_text() + "\n")
    w(f"> {MULTIPLICITY}\n")
    w("## 2. Commands, exit codes, provenance\n")
    tree = run.get("tree") or {}
    w(
        _join(
            "- finalize -> exit `",
            run["exit"],
            "`, status `",
            run["status"],
            "`; tree `",
            tree.get("sha"),
            "` dirty `",
            tree.get("dirty"),
            "`; PREREG ",
            PREREG_COMMITS,
            ", reviews ",
            list(REVIEW_COMMITS),
        )
    )
    w(f"- owner values: `{ow}`; oracle_gamma `{ORACLE_GAMMA}` (A2.2)")
    for d in docs:
        reg = ((d.get("controls") or {}).get("C10") or {}).get("registered") or {}
        w(
            _join(
                "- seed ",
                d["seed"],
                ": owner values registered in the PREREG (`",
                reg.get("heading"),
                "`): status `",
                reg.get("status"),
                "`, values `",
                reg.get("values"),
                "` (C10 compares them, A3.7a); steps recorded per arm `",
                d.get("steps_done"),
                "`, seeds_actually_run `",
                d.get("seeds_actually_run"),
                "`",
            )
        )
    for d in docs:
        lg = d.get("launch") or {}
        w(
            _join(
                "- seed ",
                d["seed"],
                ": launch-git `",
                {
                    k: lg.get(k)
                    for k in (
                        "run_sha",
                        "dirty",
                        "gamma",
                        "wd",
                        "steps",
                        "ckpt",
                        "o5",
                        "c1_flags",
                        "c2_flags",
                        "launched_utc",
                        "env",
                    )
                },
                "`",
            )
        )
        for arm, r in (d.get("arms") or {}).items():
            job = r.get("job_record") or {}
            w(
                _join(
                    "  - **",
                    arm,
                    "** rc `",
                    r.get("rc"),
                    "` (",
                    r.get("rc_meaning"),
                    "), run_id `",
                    r.get("run_id"),
                    "`, config_hash `",
                    r.get("config_hash"),
                    "`, label `",
                    r.get("arm_label"),
                    "`, job pinned_sha `",
                    job.get("pinned_sha"),
                    "`, steps recorded `",
                    r.get("steps_done"),
                    "`",
                )
            )
            w(f"    - `{shlex.join(r.get('argv') or [])}` ({r.get('argv_source')})")
    hw = (docs[0].get("hardware") if docs else None) or {}
    w(
        _join(
            "\n**Hardware:** ",
            hw.get("cpu"),
            ", ",
            hw.get("platform"),
            ", CPU, torch ",
            hw.get("torch"),
            ", Python ",
            hw.get("python"),
            ".\n",
        )
    )
    w(f"{TIMING_CAVEAT}\n")
    for d in docs:
        w(
            _join(
                "## Seed ",
                d["seed"],
                " (status `",
                d["status"],
                "`, exit `",
                d["exit"],
                "`). Everything below is SECONDARY or DIAGNOSTIC.\n",
            )
        )
        _render_seed(d, w, ow)
    w("## What this does not show\n")
    w(
        "Not E1 and not E2 (ADR-0009 L6). Not evidence on the cognitive claim. No "
        "scaling claim. One corpus (synthetic S0-03), `M = 16`, one width, a frozen "
        "substrate, the "
        "shadow buffer off, `b` off. The rescaled target is reported as rescaled and "
        "nothing stronger. Training-time differences are in-sample for phi. Within-seed "
        "CIs do not estimate run-to-run or seed-to-seed variation.\n"
    )
    w(f"{ATTRIBUTION_NOTE}\n")
    if ow.get("o5") == "eval":
        w(
            "In-loop `loss_lm` values are eval-mode values and are not comparable with "
            "arm "
            "B's train-mode heartbeat losses (A1.3).\n"
        )
    w("## The author's expectation (PREREG §7.6, verbatim), and the outcome beside it\n")
    w(expectation_text())
    w(
        "\nThe outcome, as numbers in the expectation's terms (reported as text; never "
        "scored as met or missed, A1.8 m-10):\n"
    )
    for d in docs:
        _render_expectation_outcome(d, w)
    return "\n".join(L) + "\n"


EXPECTATION_KEYS = (
    "acc.all.psiC_minus_fifo",
    "acc.all.age_minus_fifo",
    "nll_real.psiC_minus_fifo",
    "nll_real.age_minus_fifo",
    "nll_real.psiC_minus_age",
)


def _render_expectation_outcome(d: dict, w) -> None:
    dg = (d.get("training") or {}).get("diagnostics") or {}
    con = (d.get("heldout") or {}).get("contrasts") or {}
    for arm in RSR_ARMS:
        s = (dg.get(arm) or {}).get("scored") or {}
        if s:
            w(
                _join(
                    "- seed ",
                    d["seed"],
                    " `",
                    arm,
                    "` scored: fraction_shifting ",
                    _f(s.get("fraction_shifting")),
                    ", fraction oldest ",
                    _f(s.get("fraction_oldest")),
                    ", raw rho median ",
                    _f(s["spearman_rho_score_age"]["median"]),
                    ", partial rho median ",
                    _f(s["partial_rho_diagnostic"]["median"]),
                )
            )
    for key in EXPECTATION_KEYS:
        if key in con:
            w(f"- seed {d['seed']} `{key}`: {_ci_str(con[key], reading=False)}")
    w(f"- seed {d['seed']} controls failed: `{d.get('controls_failed')}`")


CONTROL_KEYS = ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C10")


def _render_seed(d: dict, w, ow: dict) -> None:
    ctl = d.get("controls") or {}
    if d.get("notes"):
        w(f"Notes: {d['notes']}\n")
    w("### Controls (§8 as amended)\n")
    for k in CONTROL_KEYS:
        v = ctl.get(k)
        if v is None:
            w(f"- **{k}**: not run")
        else:
            w(f"- **{k}**: ok = `{v.get('ok')}` -- `{json.dumps(v, default=str)[:700]}`")
    sc = (d.get("scoped_controls") or {}).get("C9_C9b")
    if sc is not None:
        w(
            f"- **C9 / C9b** (scoped to the oracle numbers): ok = `{sc.get('ok')}` "
            f"-- `{sc}`"
        )
    w("\n### What ran\n")
    for a, v in ((ctl.get("C6") or {}).get("per_arm") or {}).items():
        w(
            _join(
                "- ",
                a,
                ": steps_requested `",
                v.get("steps_requested"),
                "`, steps_done `",
                v.get("steps_done"),
                "`, status `",
                v.get("status"),
                "`",
            )
        )
    for a, e in (d.get("epoch") or {}).items():
        if e:
            w(
                _join(
                    "- ",
                    a,
                    ": T_warm `",
                    e.get("t_warm_steps"),
                    "` steps = `",
                    e.get("t_warm_epochs"),
                    "` epoch; total epochs `",
                    e.get("total_epochs"),
                    "`; T_warm / total epochs `",
                    e.get("t_warm_fraction_of_total_epochs"),
                    "` (31(b)); sampling `",
                    e.get("sampling"),
                    "`",
                )
            )
    if d.get("frozen_model_mode"):
        w(f"- frozen model mode during training: `{d['frozen_model_mode']}`")
    if d.get("checkpoint_bytes"):
        w(f"- final checkpoint bytes (T6): `{d['checkpoint_bytes']}`")
    ho = d.get("heldout") or {}
    _render_heldout(ho, w, ow, d.get("oracle_numbers_read", True))
    _render_training(d.get("training") or {}, w)
    w("\n### §7.5 reporting (SECONDARY)\n")
    w(f"- fill fraction, held-out in-loop: {_f(ho.get('fill_fraction_heldout'))}")
    w("- The censored-target caveat, verbatim from the PREREG:\n")
    w(caveat_text())
    w("\n### T5 (held-out cost, measured in this run; DIAGNOSTIC)\n")
    for n, t in (d.get("T5") or {}).items():
        w(
            _join(
                "- `",
                n,
                "`: first batch ",
                _f(t["first_batch_s"], 2),
                " s; projected full EVAL ",
                _f(t["projected_full_eval_s"], 1),
                " s; measured ",
                _f(t["measured_total_s"], 1),
                " s over ",
                t["batches_run"],
                " batches",
            )
        )
    w("")


def _render_heldout(ho: dict, w, ow: dict, oracle_read: bool) -> None:
    acc = ho.get("accuracy") or {}
    conf = _psic_confound(ow)
    if acc:
        w(
            "\n### Held-out accuracy (§7.1; SECONDARY), EVAL n = "
            f"{ho.get('n_documents')}\n"
        )
        w(scope_sentence() + "\n")
        cols = ("psiC", "fifo", "age", "lru", "oracle", "random")
        w("| bucket | n | " + " | ".join(cols) + " |")
        w("|---|---|" + "---|" * len(cols))
        for b in BUCKET_NAMES:
            row = acc.get(b) or {}
            cells = " | ".join(
                _ci_str(row.get(n), reading=False)
                + (
                    " **NOT READ (C9/C9b failed)**"
                    if n == "oracle" and not oracle_read
                    else ""
                )
                for n in cols
            )
            w(f"| {b} | {(row.get('fifo') or {}).get('n')} | {cells} |")
        w(f"\n{LRU_CAVEAT}")
    con = ho.get("contrasts") or {}
    if not con:
        return
    w("\n### Contrasts (SECONDARY; the headline is §1)\n")
    for key in sorted(con):
        tag = " **(HEADLINE)**" if key == HEADLINE else ""
        psic = conf if "psiC" in key else ""
        unread = ""
        if key.startswith("oracle_regret.") and not oracle_read:
            unread = " **NOT READ (C9/C9b failed)**"
        w(f"- `{key}`{tag}{psic}{unread}: {_ci_str(con[key])}")
    for kind in ("real", "answer"):
        row = (ho.get("nll") or {}).get(kind) or {}
        names = ("psiC", "fifo", "age", "lru")
        lv = "; ".join(f"{n} {_f((row.get(n) or {}).get('point'))}" for n in names)
        w(f"- NLL `{kind}` level (pooled per target): {lv}")
    for a, s in (ho.get("headroom_share") or {}).items():
        unread = "" if oracle_read else " **NOT READ (C9/C9b failed)**"
        w(
            f"- headroom share `{a}` (all){unread}: {_ci_str(s, reading=False)}, "
            f"dropped {s.get('dropped')}"
        )
    for s, v in (ho.get("snapshots") or {}).items():
        w(
            f"- DIAGNOSTIC snapshot `{s}` (A1.8 m-9): accuracy "
            f"{_ci_str(v['level'], reading=False)}; "
            f"minus final fifo {_ci_str(v['minus_final_fifo'])}"
        )
    for n, e in (ho.get("heldout_evictions") or {}).items():
        w(
            _join(
                "- DIAGNOSTIC held-out evictions `",
                n,
                "`: n ",
                e["n_evictions"],
                ", fraction oldest ",
                _f(e["fraction_oldest"]),
                ", victim ages `",
                e["victim_age_distribution"],
                "`",
            )
        )


def _confound_tagger(w, conf: str):
    """A2.1 item 2 (O5 = train): the named confound beside every psiC entry -- each
    line whose first backticked name is a psiC arm, pass or contrast."""
    if not conf:
        return w

    def tagged(line: str) -> None:
        parts = line.split("`")
        name = parts[1] if line.startswith("- ") and len(parts) > 2 else ""
        psic = "psiC" in name and all(ch.isalnum() or ch in "._@-" for ch in name)
        if psic and conf not in line:
            line = line + conf
        w(line)

    return tagged


def _render_training(tr: dict, w) -> None:
    if not tr:
        return
    w("\n### Training time, on the pool (§7.3: SECONDARY; in-sample for phi)\n")
    for a, dl in (tr.get("delta_loss_lm") or {}).items():
        blocks = [(b["steps"], _f(b["mean"])) for b in dl.get("blocks", [])]
        w(
            _join(
                "- delta loss_lm `",
                a,
                "` - fifo, post-warm: mean ",
                _f(dl.get("mean")),
                ", sd ",
                _f(dl.get("sd")),
                BROKEN if dl.get("sd_exactly_zero") else "",
                "; 500-step blocks ",
                blocks,
            )
        )
    for a, r in (tr.get("rsr") or {}).items():
        w(
            _join(
                "- DIAGNOSTIC `",
                a,
                "`: L_MC warm mean ",
                _f(r["loss_mc_warm"]["mean"]),
                ", post-warm mean ",
                _f(r["loss_mc_post_warm"]["mean"]),
                "; corr_psi_age post-warm median ",
                _f(r["corr_psi_age_post_warm"]["median"]),
                " (None on ",
                r["corr_psi_age_none_steps"],
                " steps); phi grad norm median ",
                _f(r["phi_grad_norm"]["median"]),
                "; eval_forward_s total ",
                _f(r["eval_forward_s"]["total"], 1),
                "; fill fraction (regression rows) ",
                _f(r["fill_fraction_regression_rows"]),
            )
        )
    for a, r in (tr.get("rsr") or {}).items():
        w(
            f"- DIAGNOSTIC `{a}` attribution (mean per-step fraction): warm "
            f"`{r.get('attribution_warm')}`, post-warm `{r.get('attribution_post_warm')}`"
        )
    if tr.get("run_level_histograms_agree") is not None:
        w(
            "- run-level split histograms (train()'s phase_summary) equal the per-step "
            f"sums: `{tr['run_level_histograms_agree']}`"
        )
    w("\n### §7.1 diagnostics (§7.4: DIAGNOSTIC; never a gate; warm / scored split)\n")
    for a, ph in (tr.get("diagnostics") or {}).items():
        for p, v in ph.items():
            pr = v["partial_rho_diagnostic"]
            w(
                _join(
                    "- `",
                    a,
                    "` ",
                    p,
                    ": steps ",
                    v["n_steps"],
                    ", evictions ",
                    v["n_evictions"],
                    ", fraction oldest ",
                    _f(v["fraction_oldest"]),
                    ", fraction_shifting ",
                    _f(v["fraction_shifting"]),
                    ", mean_shift ",
                    _f(v["mean_shift"]),
                    "; raw Spearman rho(score, age) median ",
                    _f(v["spearman_rho_score_age"]["median"]),
                    "; partial rho (a diagnostic, never a gate) median ",
                    _f(pr["median"]),
                    " IQR [",
                    _f(pr["q25"]),
                    ", ",
                    _f(pr["q75"]),
                    "], None on ",
                    v["partial_rho_none_steps"],
                    " steps",
                )
            )
            w(
                _join(
                    "  - victim ages `",
                    v["victim_age_distribution"],
                    "`; displacement `",
                    v["displacement_distribution"],
                    "`",
                )
            )
            if "uniform_random_expected" in v:
                w(
                    _join(
                        "  - uniform random over the live slots would give `",
                        v["uniform_random_expected"],
                        "` (A1.8 m-2)",
                    )
                )
    w(
        "- Partial rho partials out the slot gestalts only (linear), not `c_t`; no "
        "verdict is drawn from any comparison with the spec's E2 value. Leading-edge "
        "statistic: not "
        "computed (the spec defines a baseline, not a statistic; `leading_edge.py` is a "
        "stub). Attention-share Gini: not available."
    )


# --------------------------------------------------------------------------- #
# Build-time measurement: T5 (timing only)
# --------------------------------------------------------------------------- #


def time_one_batch(cfg: Run1Config, doc_ids: list[int]) -> dict:
    """T5: one batch per pass on the REAL substrate, read-only, the process's threads.
    **Timing only**: no accuracy or loss is returned. RSR arms run a FRESH head (no
    trained phi exists; the cost does not depend on its values). Never EVAL (§14a)."""
    if any(cfg.eval_range[0] <= i < cfg.eval_range[1] for i in doc_ids):
        raise Refused("time-batch never runs an EVAL document (PREREG §14a)")
    data = Path(cfg.substrate.checkpoint).read_bytes()
    if hashlib.sha256(data).hexdigest() != cfg.substrate.sha256:
        raise Refused("substrate sha256 mismatch")
    payload = ck.load_bytes(data, source=cfg.substrate.checkpoint)
    vmap = vocab_map(cfg)
    tg = TGConfig(
        D=cfg.d,
        V=4 + len(vmap),
        F=int(cfg.d * 2.6875),
        max_sentence_tokens=64,
        max_sentences_in_short_term=cfg.memory_slots,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        eod_id=3,
    )
    with torch.random.fork_rng(devices=[]):
        model = TGModel(tg)
    model.load_state_dict(payload["model"])
    docs = docs_by_id(cfg, doc_ids)
    out: dict[str, Any] = {}
    for name in ("psiC", "fifo", "age", "oracle", "lru", "random0", "sz_fifo", "sz_psiC"):
        if name in ("psiC", "sz_psiC", "age"):
            ov = {
                "b_enabled": False,
                "shadow_enabled": False,
                "nu": 0.0,
                "beta": 1.0,
                "gamma": 0.5,
            }  # a timing placeholder only; not a registered value
            if name == "age":
                ov["psi_override"] = "age_only"
            pol = loop.build_policy(
                "rsr",
                d_model=cfg.d,
                steps_per_epoch=cfg.steps_per_epoch,
                generator=loop.phi_generator(cfg.seed)[0],
                overrides=ov,
            )
            pol.set_train_step(cfg.iters)
        elif name in ("fifo", "sz_fifo"):
            pol = FIFOPolicy()
        else:
            pol = reference_policy(name, docs, cfg.seed)
        t0 = time.perf_counter()
        run_batch(
            model,
            pol,
            docs,
            vmap,
            cfg,
            zero_slots=name.startswith("sz_"),
            observe=name == "lru",
        )
        out[name] = time.perf_counter() - t0
    t0 = time.perf_counter()
    for d in docs:
        simulate(d, FIFOPolicy(), cfg.memory_slots)
    out["simulate_per_doc_s"] = (time.perf_counter() - t0) / len(docs)
    out["threads"] = torch.get_num_threads()
    out["batch_docs"] = len(docs)
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _owner_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--gamma", type=float, required=True, help="O1")
    p.add_argument("--phi-weight-decay", type=float, required=True, help="O4")
    p.add_argument("--iters", type=int, required=True, help="R-1")
    p.add_argument("--o5", required=True, help="O5: eval or train (A2.1)")


def _owner(a, *, dry_run: bool = False) -> OwnerValues:
    ov = OwnerValues(a.gamma, a.phi_weight_decay, a.iters, a.o5, SEEDS)
    validate_owner_values(ov, dry_run=dry_run)
    return ov


def main(argv: list[str] | None = None) -> Exit:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--preflight"]:  # A1.2 step 2's literal spelling
        argv = ["preflight", *argv[1:]]
    p = ArgumentParser(prog="rsr-run1", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("preflight", help="C0 x 3 seeds, launcher, substrate, env")
    pp.add_argument("--dry-run", action="store_true", help="allow OUT / JOB_SUFFIX")
    ps = sub.add_parser("measure-seed", help="one seed -> measure-seed<s>.json")
    _owner_args(ps)
    ps.add_argument("--seed", type=int, required=True)
    ps.add_argument("--threads", type=int, required=True)
    ps.add_argument(
        "--dry-run",
        action="store_true",
        help="a launcher-v5 dry run only (A3.12, A3c.1): STEPS 0, its isolated "
        "RSR_ORCH_ROOT / JOB_SUFFIX read and recorded; C10 fails it by design",
    )
    ps.add_argument("--out-root", default=str(ROOT / "runs" / RUN_ID))
    pf = sub.add_parser("finalize", help="the three measure-seed jsons -> RESULTS")
    _owner_args(pf)
    pf.add_argument("--out-root", default=str(ROOT / "runs" / RUN_ID))
    pf.add_argument(
        "--results", default=str(ROOT / "experiments" / "rsr-run1" / "RESULTS.md")
    )
    pt = sub.add_parser("time-batch")
    pt.add_argument("--threads", type=int, required=True)
    pt.add_argument("--doc-start", type=int, default=POOL[0])
    po = sub.add_parser("probe-oracle")
    po.add_argument("--doc-start", type=int, default=POOL[0])
    po.add_argument("--n-docs", type=int, default=256)
    a = p.parse_args(argv)
    try:
        if a.cmd == "preflight":
            r = preflight(dry_run=a.dry_run)
            print(json.dumps(r, indent=2, default=str))
            return Exit.OK if r["ok"] else Exit.DID_NOT_RUN
        if a.cmd == "measure-seed":
            ow = _owner(a, dry_run=a.dry_run)
            if a.seed not in SEEDS:
                raise Refused(f"seed {a.seed} is not one of {SEEDS}")
            doc = measure_seed(
                prereg_config(a.seed, ow.iters),
                ow,
                Path(a.out_root),
                threads=a.threads,
                dry_run=a.dry_run,
            )
            return Exit(doc["exit"])
        if a.cmd == "finalize":
            ow = _owner(a)
            docs = []
            for s in SEEDS:
                f = Path(a.out_root) / f"measure-seed{s}.json"
                if not f.exists():
                    raise Refused(f"{f} is missing: run measure-seed --seed {s} first")
                docs.append(json.loads(f.read_text()))
            return finalize(docs, ow, Path(a.out_root), results_path=Path(a.results))
        if a.cmd == "time-batch":
            torch.set_num_threads(a.threads)
            cfg = prereg_config(0, 3000)
            ids = list(range(a.doc_start, a.doc_start + cfg.eval_batch))
            print(json.dumps(time_one_batch(cfg, ids), indent=2))
            return Exit.OK
        if a.cmd == "probe-oracle":
            cfg = prereg_config(0, 3000)
            docs = docs_by_id(cfg, range(a.doc_start, a.doc_start + a.n_docs))
            r = probe_oracle_invariance(docs, cfg.memory_slots)
            print(json.dumps(r, indent=2))
            return Exit.OK if r["ok"] else Exit.FAIL
    except Refused as e:
        print(f"DID NOT RUN: {e}", file=sys.stderr)
        return Exit.DID_NOT_RUN
    except ControlFailure as e:
        print(f"CONTROL FAILED: {e}", file=sys.stderr)
        return Exit.DID_NOT_RUN
    return Exit.DID_NOT_RUN


if __name__ == "__main__":
    run_main(main)
