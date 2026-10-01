"""RSR run 1 -- the measurement harness (PREREG `experiments/rsr-run1/PREREG.md`, T3-T8).

Pre-registration: `experiments/rsr-run1/PREREG.md`, committed alone at `7c58777`, ahead of
this file; reviewed at `787875b` (MERGE WITH FIXES, an amendment pending). **Descriptive.
No gate, no pass/fail threshold**: the controls C0-C8 decide only whether the run is
readable, and every contrast is read only as "CI above 0", "CI below 0" or
"CI includes 0".

What it does (PREREG §9: "Measurement ... come from run.py"):

* ``launch`` -- refuses without the owner-open values (O1 gamma, O4 phi weight decay, the
  run length; and the seed list): no defaults. Refuses a dirty tree (review MAJOR-5) and
  pins the launch to HEAD through the slot wrapper's ``--pinned-sha`` with a ``--wait``.
  Re-asserts C0 before any model loads, freezes the manifest, then starts every
  (arm, seed) through **the PREREG §9 command line** at a parameterised thread cap. Each
  (arm, seed) gets its own record and its own rc, classified (review MAJOR-2: an uncaught
  `train()` exception, `WarmStartRefused` included, is rc 1 with no heartbeat). A started
  arm is **refused, never resumed** (PREREG §11.3).
* ``measure`` -- per seed: C1-C8 and T7 from the arms' own artefacts; the held-out harness
  on EVAL (eval mode, ``no_grad``, final phi from ``__head__``, ``set_train_step(iters)``,
  observe off) for every arm and for the oracle, LRU, random x5 and the slots-zeroed floor
  under each arm; per-document answer vectors written for the paired bootstrap; C4 through
  fresh-stream's own ``measure_checkpoint`` (imported unchanged). Then across seeds: the
  ONE designated headline contrast per seed and across seeds, everything else labelled
  secondary or diagnostic; the ledger; ``RESULTS.md``.
* ``time-batch`` (T5) and ``probe-oracle`` (T7) -- build-time measurements only.

🔴 **Every call into `feat/rsr-run1-ready` code goes through an ADAPTER** (search for
``ADAPTER``). That branch is mid-fix; the adapters are the list to re-check on rebase.

Never: trains anything itself (the arms are the CLI's); calls `rsr.constants.record()`;
fills §15; reads `loss` (the analysis reads ``loss_lm``); calls the rescaled target by any
stronger name than "rescaled"; makes a scaling claim; scores the author's expectation.

Exit codes (`rsr.exit_codes`): 0 every control passed in every seed · 3 did not run, or a
control failed (inconclusive; numbers written, marked not read) · 5 an arm measured INERT.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
import os
import platform
import shlex
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
RUN_ID = "rsr-run1"
PREREG = "experiments/rsr-run1/PREREG.md"
PREREG_COMMIT = "7c58777"
REVIEW_COMMIT = "787875b"


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
    and the §9.2 bootstrap convention. Its `run_arm` is NOT used: it calls
    `run_policy_loop` without `eval_capture`, which an RSR policy needs (PREREG T3)."""
    return _load("_rsr_run1_b2", "experiments/b2-psi-probe/run.py")


@functools.cache
def FS():
    """fresh-stream (PREREG C4, T8): `measure_checkpoint` and `r_quantity`, unchanged."""
    return _load("_rsr_run1_fresh_stream", "experiments/fresh-stream/run.py")


# --------------------------------------------------------------------------- #
# 🔒 PREREG, transcribed. Changing any of these is changing the PREREG.
# --------------------------------------------------------------------------- #

ARMS = ("psiC", "fifo", "age")  # PREREG §5
RSR_ARMS = ("psiC", "age")
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
DEVICE = "cpu"  # §6 "Why CPU"
#: §3, and the review's cross-sha check (seed-invariant map).
EXPECTED_VOCAB_SHA256 = "afbaf994e9d1f617fdc2e68579bf5f55622ee17aee056d33d8667d218d245031"
_R = "/Users/keanooo7/retrieval-successor-retention"
_B = f"{_R}/.worktrees/fresh-stream/runs/fresh-stream/B"
#: §1 for seed 0. Seeds 1 and 2: the coordinator's amendment brief (3 seeds); their
#: sha256 are T0 manifest lines 130 and 150, read from that file (the PREREG's O2 row
#: quotes them truncated: `dadd1e08…3da8`, `b507ebc5…63b8`). PREREG-AMBIGUITY A1: the
#: amendment must restate them.
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
#: C4: fresh-stream's ledger row, read by key and seed position, never retyped. The
#: typed seed-0 value is only the cross-check that the row read is the one the PREREG
#: names; seeds 1 and 2 have no PREREG value (A1).
C4_LEDGER = f"{_R}/.worktrees/fresh-stream/runs/fresh-stream/ledger.json"
C4_KEY = "B.ckpt3000.R_quantity"
C4_PREREG_VALUE = {0: 0.8597733527421951}
C4_TOL = 1e-6
C4_N = 64  # measure_checkpoint's `n`: arm B's vocabulary documents
N_BOOT = 2000
BOOT_SEED_BASE = 20260927  # B2 PREREG §9.2's convention, reused (front matter)
N_RANDOM = 5
#: §7.2: Belady's MIN "for any gamma_o in (0, 1)" (T7). B2's primary gamma is the one
#: run; `probe_oracle_invariance` checks T7 on every EVAL document first (A4).
ORACLE_GAMMA = 0.9
T7_GAMMAS = (0.01, 0.1, 0.5, 0.9, 0.97, 0.999)
DELTA_BLOCK = 500  # §7.3
#: A3 (not fixed by the PREREG): the held-out batch; all arms share it.
EVAL_BATCH = 16
#: The ONE designated headline (review MAJOR-4; coordinator's amendment brief): psiC -
#: FIFO, all-query held-out answer accuracy. Everything else is secondary/diagnostic.
HEADLINE = "acc.all.psiC_minus_fifo"
SCOPE_SENTENCE = (
    "Every CI here is a within-seed CI: it is conditional on that seed's one trained "
    "phi, "
    "one data order, one dropout realisation and one substrate checkpoint, and it does "
    "not estimate run-to-run variance. Only the across-seed spread speaks to that."
)
MULTIPLICITY_SENTENCE = (
    "Apart from the one headline contrast, every CI is unadjusted for multiplicity and "
    "is secondary or diagnostic."
)
QUESTION = (
    "On fresh-stream arm B's ckpt-003000 with the transformer frozen, when psi_hat is "
    "trained online by L_MC on a fixed pool of 256 documents arm B never saw, how do its "
    "evictions and the frozen model's held-out answer accuracy and LM loss compare with "
    "FIFO and with a learned age-only head, arms that differ from it only in the "
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
    "psiC_minus_lru": ("psiC", "lru"),  # review MAJOR-1
    "age_minus_lru": ("age", "lru"),
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


class ControlFailure(RuntimeError):
    """A PREREG §8 control (or T7) failed: inconclusive, exit 3."""


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
    """Owner-open (PREREG §12 O1, O4; review R-1 run length) plus the seed list
    (O2, per the coordinator's amendment brief). **No defaults.**"""

    gamma: float
    seeds: tuple[int, ...]
    phi_weight_decay: float
    iters: int


@dataclass(frozen=True)
class Run1Config:
    """What the harness reads that a test may shrink. `prereg_config()` builds the one
    the real run uses; the CLI has no flag that changes it."""

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

    @property
    def ckpt_every(self) -> int:
        """§6: one checkpoint, at the end, holding the trained phi."""
        return self.iters

    @property
    def pool_length(self) -> int:
        return self.pool[1] - self.pool[0]

    @property
    def t_warm_steps(self) -> float:
        """§6: one epoch = pool / batch steps (31(b)); T_warm_epochs = 1 (FROZEN)."""
        return self.pool_length / self.batch

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


def validate_owner_values(ov: OwnerValues) -> None:
    """Refuse anything not ruled, or that the code would not actually run."""
    for name in ("gamma", "seeds", "phi_weight_decay", "iters"):
        if getattr(ov, name, None) is None:
            raise Refused(f"owner value {name} is not given (PREREG §12): refused")
    g = float(ov.gamma)
    if g != g or not (0.0 <= g < 1.0):
        raise Refused(f"gamma={ov.gamma!r} is outside [0, 1) (spec §4.5)")
    if not ov.seeds or len(set(ov.seeds)) != len(ov.seeds):
        raise Refused(f"seeds={ov.seeds!r}: a non-empty list of distinct seeds")
    if int(ov.iters) != ov.iters or ov.iters < 1:
        raise Refused(f"iters={ov.iters!r}: a positive integer run length")
    wd = code_phi_weight_decay()
    if float(ov.phi_weight_decay) != wd:
        raise Refused(
            f"phi weight decay {ov.phi_weight_decay!r} was given, but the code pins "
            f"{wd!r} (loop.ADAMW_WEIGHT_DECAY) and offers no other value (PREREG O4): "
            f"running it would report a decay that did not run. A code change and its "
            f"review come first."
        )


# --------------------------------------------------------------------------- #
# ADAPTERS -- every call into feat/rsr-run1-ready (built against 211d6da).
# --------------------------------------------------------------------------- #

#: ADAPTER (resume). PREREG §11.3: no resume; at 211d6da the data-order generator is
#: not checkpointed. Even if the fixed head checkpoints it, the PREREG forbids resume
#: until amended, so this stays False and `refuse_resume` asserts it.
RESUME_PERMITTED = False

#: ADAPTER: the arms' `max_tokens` -- `train()`'s default; the CLI has no flag for it
#: at 211d6da (PREREG §6: 64, arm B's).
CLI_MAX_TOKENS = 64

#: ADAPTER (flags). Extra flags the fixed head may require on every arm's command line
#: -- e.g. an eval-mode flag for the frozen transformer (review MAJOR-3) or a
#: `--srep-norm-reg-weight`. Empty at 211d6da; an amendment names them.
EXTRA_TRAIN_FLAGS: tuple[str, ...] = ()


def code_phi_weight_decay() -> float:
    """ADAPTER: the decoupled decay every AdamW group runs (`loop.py:602`)."""
    return float(loop.ADAMW_WEIGHT_DECAY)


def cli_argv(cfg: Run1Config, arm: str, gamma: float, out_dir: Path) -> list[str]:
    """ADAPTER (flags, T1): the PREREG §9 command for one arm, as an argv. ``<O1>`` is
    the ruled gamma, written with `repr` so the CLI parses back the same float."""
    s = cfg.substrate
    common = [
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
        "--pool-offset",
        str(cfg.pool[0]),
        "--pool-length",
        str(cfg.pool_length),
        "--pool-vocab-documents",
        str(cfg.vocab_documents),
        "--warm-start",
        s.checkpoint,
        "--warm-start-sha256",
        s.sha256,
        "--warm-start-manifest",
        s.manifest,
        "--warm-start-root",
        s.root,
    ]
    if s.source_config is not None:
        common += ["--warm-start-source-config", s.source_config]
    common += list(EXTRA_TRAIN_FLAGS)
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
        f"gamma={float(gamma)!r}",
    ]
    py = [str(ROOT / ".venv" / "bin" / "python"), "-m", "rsr.train.loop"]
    tail = ["--out-dir", str(out_dir)]
    if arm == "psiC":
        return py + common + rsr + tail
    if arm == "fifo":
        return py + common + ["--policy", "fifo"] + tail
    if arm == "age":
        return py + common + rsr + ["--policy-override", "psi_override=age_only"] + tail
    raise ValueError(f"unknown arm {arm!r}; the arms are {ARMS}")


def slot_prefix(
    threads: int, *, pinned_sha: str, wait_s: float, job_id: str
) -> list[str]:
    """The PREREG §9 `SLOT=(...)` wrapper, with the review's MAJOR-2/-5 fixes:
    `--wait` (a busy lane waits instead of refusing with 3) and `--pinned-sha` (HEAD
    must be this sha and the tree clean outside runs/, `slot.py:84-101`)."""
    return [
        "env",
        "PYTHONPATH=scripts",
        str(ROOT / ".venv" / "bin" / "python"),
        "-m",
        "orchestrator.slot",
        "run",
        "--lane",
        "cpu-det",
        "--slots",
        str(threads),
        "--wait",
        repr(float(wait_s)),
        "--pinned-sha",
        pinned_sha,
        "--job-id",
        job_id,
        "--cwd",
        str(ROOT),
        "--",
    ]


#: `orchestrator.lanes.THREAD_ENV_VARS`, set here too so a direct launch is capped
#: exactly as a slot launch is (T4).
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
    """ADAPTER: the result dict `rsr.train.loop.main` prints last
    (`json.dumps(r, indent=2)`). None if absent -- the run raised before printing."""
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
    """ADAPTER: `heartbeat.jsonl` -> header, beats (by step), footer, crash. A missing
    file is returned as ``exists: False`` (a refusal before the loop writes none)."""
    path = Path(out_dir) / "heartbeat.jsonl"
    if not path.exists():
        return {
            "exists": False,
            "header": None,
            "beats": [],
            "footer": None,
            "crash": None,
        }
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    beats = [r for r in rows if r.get("kind") == "beat"]
    return {
        "exists": True,
        "header": next((r for r in rows if r.get("kind") == "header"), None),
        "beats": sorted(beats, key=lambda r: r["step"]),
        "footer": next((r for r in rows if r.get("kind") == "footer"), None),
        "crash": next((r for r in rows if r.get("kind") == "crash"), None),
    }


def batch_fingerprint(beat: dict, cfg: Run1Config) -> tuple:
    """ADAPTER (C2; review MINOR-4). What identifies the batch a step drew. If the
    fixed head logs a hash of the drawn ids, that is used; at 211d6da it does not, so
    the fingerprint is `(n_answer_targets, n_pad_targets)`: both depend only on the drawn
    documents and not on the eviction rule. `n_targets_scored` is NOT used -- it is
    `batch x S x (L - 1)` every step and cannot see a batch mismatch."""
    for key in ("batch_ids_sha256", "batch_sha256", "batch_ids"):
        if beat.get(key) is not None:
            return (key, json.dumps(beat[key]))
    n_pad = round(float(beat["pad_target_fraction"]) * float(beat["n_targets_scored"]))
    return ("counts", int(beat["n_answer_targets"]), int(n_pad))


def replay_batches(cfg: Run1Config, steps: int) -> list[list[int]]:
    """ADAPTER (sampler; permutation epochs may replace it). The pool indices `train()`
    draws at each step at 211d6da: `torch.randint(0, len(pool), (batch,))` from
    `torch.Generator(device).manual_seed(seed)`, nothing else drawing from it
    (`loop.py:919`, `:1200-1202`)."""
    g = torch.Generator(device=cfg.device).manual_seed(cfg.seed)
    return [
        torch.randint(
            0, cfg.pool_length, (cfg.batch,), generator=g, device=cfg.device
        ).tolist()
        for _ in range(steps)
    ]


def read_step_diagnostics(out_dir: Path) -> list[dict]:
    """ADAPTER (per-step diagnostics; the warm/scored split): one normalised dict per
    line of `diagnostics.jsonl`. JSON histogram keys come back as strings; made ints."""
    path = Path(out_dir) / loop.DIAGNOSTICS_FILE
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        d = r["diag"]
        warm = d.get("warm") or []
        if len(warm) != 1:
            raise ControlFailure(
                f"{path}: step {r['step']} has warm flags {warm!r}; one optimizer step "
                f"is either warm or scored (§3.4)"
            )

        def ints(h):
            return {int(k): int(v) for k, v in (h or {}).items()}

        out.append(
            {
                "step": int(r["step"]),
                "warm": bool(warm[0]),
                "n_evictions": int(d.get("n_evictions", 0)),
                "shift": ints(d.get("shift_distribution")),
                "victim_age": ints(d.get("victim_age_distribution")),
                "partial_rho": d.get("partial_rho_score_age_given_content"),
                "spearman_rho": d.get("spearman_rho_score_age"),
            }
        )
    return sorted(out, key=lambda r: r["step"])


def final_checkpoint(out_dir: Path, iters: int) -> Path:
    """ADAPTER: the loop's filename (`ckpt-{step:06d}.pt`, `loop.py:1359`)."""
    return Path(out_dir) / f"ckpt-{iters:06d}.pt"


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


def frozen_model_mode(result: dict) -> str | None:
    """ADAPTER (review MAJOR-3): the training-time mode of the frozen model, as the
    arm recorded it ("train (dropout live)" at 211d6da)."""
    return result["config"].get("frozen_model_mode")


def head_state(payload: dict) -> dict | None:
    """ADAPTER: the trained phi (`checkpoint.py:444-445`)."""
    return (payload.get("policy") or {}).get("__head__")


def eval_policy(arm: str, payload: dict, cfg: Run1Config):
    """ADAPTER: an arm's FINAL policy (PREREG §7.1). psiC / age: `build_policy` with the
    arm's own frozen overrides (read from the checkpoint, never retyped), phi from
    ``__head__``, ``set_train_step(iters)``. FIFO: `FIFOPolicy`."""
    conf = payload["meta"]["config"]
    if arm == "fifo":
        if conf.get("policy") != "fifo":
            raise ControlFailure(
                f"the fifo arm's checkpoint is policy {conf['policy']!r}"
            )
        return FIFOPolicy()
    if conf.get("policy") != "rsr":
        raise ControlFailure(f"the {arm} arm's checkpoint is policy {conf['policy']!r}")
    want_age = arm == "age"
    if (conf["policy_overrides"].get("psi_override") == "age_only") is not want_age:
        raise ControlFailure(
            f"the {arm} arm's checkpoint has overrides {conf['policy_overrides']}"
        )
    gen, _ = loop.phi_generator(int(conf["seed"]))  # never the global RNG
    pol = loop.build_policy(
        "rsr",
        d_model=int(conf["tg"]["D"]),
        steps_per_epoch=float(conf["epoch"]["steps_per_epoch"]),
        generator=gen,
        overrides=dict(conf["policy_overrides"]),
        no_warmup_ablation=bool(conf.get("no_warmup_ablation", False)),
    )
    head = head_state(payload)
    if head is None or pol.head is None:
        raise ControlFailure(f"the {arm} arm's checkpoint carries no __head__")
    pol.head.load_state_dict(head)
    pol.set_train_step(int(cfg.iters))
    return pol


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


def ledger_mod():
    import ledger as _ledger  # scripts/ledger.py

    return _ledger


def tree_state() -> dict[str, Any]:
    """HEAD and whether anything outside runs/ is uncommitted (review MAJOR-5)."""
    exe = "/opt/homebrew/bin/git"
    try:
        head = subprocess.run(
            [exe, "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        st = subprocess.run(
            [
                exe,
                "-C",
                str(ROOT),
                "status",
                "--porcelain",
                "--",
                ":(top)",
                ":(top,exclude)runs",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return {"sha": None, "dirty": None, "porcelain": None, "error": str(e)}
    if head.returncode != 0 or st.returncode != 0:
        return {
            "sha": None,
            "dirty": None,
            "porcelain": None,
            "error": (head.stderr or st.stderr).strip(),
        }
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


def probe_threads(threads: int) -> int:
    """T4: the thread count torch reports in a child under this harness's thread env,
    the env the slot wrapper sets for a cpu-det job (`slot.py:167-170`)."""
    r = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.get_num_threads())"],
        env=thread_env(threads),
        capture_output=True,
        text=True,
        timeout=120,
    )
    return int(r.stdout.strip())


# --------------------------------------------------------------------------- #
# Documents, vocabulary, C0
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
    """C0 (§8, §2): pool and EVAL disjoint from every used range and from each other,
    below `1_000_003`, closed under the vocabulary, and the PREREG's map hash. Asserted
    before any model loads (review MINOR-3: before launch, too). Raises."""
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
    if (
        cfg.expected_vocab_sha256 is not None
        and out["vocab_sha256"] != cfg.expected_vocab_sha256
    ):
        problems.append("vocab_sha256")
    out["ok"] = not problems
    out["failed"] = problems
    if problems:
        raise ControlFailure(f"C0 failed: {problems} ({out})")
    return out


# --------------------------------------------------------------------------- #
# launch
# --------------------------------------------------------------------------- #


def arm_dir(out_root: Path, arm: str, seed: int) -> Path:
    return Path(out_root) / arm / f"seed{seed}"


def refuse_resume(out_dir: Path) -> None:
    """PREREG §11.3: an arm that has started is refused, never resumed."""
    if RESUME_PERMITTED:
        raise AssertionError("RESUME_PERMITTED is True: PREREG §11.3 needs an amendment")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise Refused(
            f"{out_dir} already holds {sorted(p.name for p in out_dir.iterdir())[:6]}: "
            f"an arm that has started is never resumed (PREREG §11.3 -- the data-order "
            f"generator is not checkpointed, so a resumed arm is not paired). A re-run "
            f"starts from step 0 under an amendment, with this attempt kept on record."
        )


def classify_rc(rc: int | None, result: dict | None, hb: dict) -> str:
    """Review MAJOR-2: what an arm's rc means, disambiguated by the heartbeat.

    rc 1 is EITHER an uncaught `train()` exception (`WarmStartRefused`, a closure or
    override `ValueError`, a crash after `hb.crash`) OR inconclusive liveness."""
    foot = (hb.get("footer") or {}).get("status")
    if rc is None:
        return "not finished"
    if result is not None and foot == "completed":
        band = (result.get("liveness") or {}).get("band")
        return {
            0: "completed, live",
            5: "completed, INERT",
            1: "completed, liveness inconclusive",
            3: "completed, liveness measurement invalid",
        }.get(rc, f"completed, rc {rc} (band {band})")
    if hb.get("crash") is not None:
        return f"crashed after start (heartbeat crash record), rc {rc}"
    if not hb.get("exists"):
        if rc == 1:
            return "refused before the loop: uncaught exception, no heartbeat (rc 1)"
        if rc == 3:
            return "slot refused or usage error, no heartbeat (rc 3)"
        return f"no heartbeat, rc {rc}"
    return f"incomplete heartbeat, no footer, rc {rc}"


def write_manifest(
    out_root: Path, cfgs: list[Run1Config], owner: OwnerValues, plan: dict
) -> Path:
    """Freeze the config before any arm starts (the ledger's `manifest()`)."""
    led = ledger_mod().Ledger(RUN_ID, question=QUESTION, runs_root=Path(out_root).parent)
    if led.path.parent.resolve() != Path(out_root).resolve():
        raise Refused(f"the ledger dir {led.path.parent} is not {out_root}")
    return led.manifest(
        {
            "experiment": EXPERIMENT,
            "prereg": PREREG,
            "prereg_commit": PREREG_COMMIT,
            "review_commit": REVIEW_COMMIT,
            "configs": [c.as_dict() for c in cfgs],
            "owner_values": dataclasses.asdict(owner),
            "plan": plan,
            "headline": HEADLINE,
            "falsifier": FALSIFIER,
            "expected": EXPECTED,
            "hardware": hardware(),
        }
    )


def launch_arm(cfg, owner, arm, out_root, *, launcher, threads, tree, wait_s):
    """Start one (arm, seed); the record is written before the child exists."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)  # review MAJOR-2: the log's directory
    out = arm_dir(out_root, arm, cfg.seed)
    refuse_resume(out)
    out.mkdir(parents=True, exist_ok=True)
    argv = cli_argv(cfg, arm, owner.gamma, out)
    job_id = f"{RUN_ID}-{arm}-seed{cfg.seed}"
    full = (
        slot_prefix(threads, pinned_sha=tree["sha"], wait_s=wait_s, job_id=job_id)
        if launcher == "slot"
        else []
    ) + argv
    record = {
        "arm": arm,
        "seed": cfg.seed,
        "argv": argv,
        "launcher": launcher,
        "full_argv": full,
        "threads": threads,
        "thread_env": {k: str(threads) for k in THREAD_ENV_VARS},
        "git_sha": tree["sha"],
        "git_dirty": tree["dirty"],
        "git_porcelain": tree["porcelain"],
        "hardware": hardware(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "started_epoch": time.time(),
        "job_id": job_id,
    }
    (out_root / f"launch-{arm}-seed{cfg.seed}.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    log = open(out_root / f"{arm}-seed{cfg.seed}.log", "w")  # noqa: SIM115 -- closed in finish
    proc = subprocess.Popen(
        full, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=thread_env(threads)
    )
    return proc, log, record


def finish_arm(proc, log, record: dict, out_root: Path) -> dict:
    rc = proc.wait()  # the child's own status, never a pipe's
    log.close()
    rec = dict(record)
    arm, seed = rec["arm"], rec["seed"]
    log_path = Path(out_root) / f"{arm}-seed{seed}.log"
    result = read_cli_result(log_path)
    hb = read_heartbeat(arm_dir(out_root, arm, seed))
    rec.update(
        rc=rc,
        rc_meaning=classify_rc(rc, result, hb),
        finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        wall_s=time.time() - rec["started_epoch"],
        log=str(log_path),
        result=result,
        run_id=None if result is None else result.get("run_id"),
        config_hash=None if result is None else result.get("config_hash"),
        arm_label=None if result is None else result.get("arm_label"),
        heartbeat_exists=hb["exists"],
    )
    (Path(out_root) / f"arm-{arm}-seed{seed}.json").write_text(
        json.dumps(rec, indent=2, default=str) + "\n"
    )
    return rec


def launch(
    cfgs: list[Run1Config],
    owner: OwnerValues,
    out_root: Path,
    *,
    arms=ARMS,
    launcher: str,
    threads: int,
    parallel: bool,
    wait_s: float = 0.0,
    allow_dirty: bool = False,
) -> dict[str, dict]:
    """C0 per seed, a clean pinned tree, the frozen manifest, then every (arm, seed)
    through the §9 command line. Returns ``{f"{arm}-seed{s}": record}``."""
    validate_owner_values(owner)
    if sorted(c.seed for c in cfgs) != sorted(owner.seeds):
        raise Refused(
            f"configs for seeds {[c.seed for c in cfgs]} != ruled {owner.seeds}"
        )
    if any(c.iters != owner.iters for c in cfgs):
        raise Refused("a config's iters is not the ruled run length")
    if launcher not in ("slot", "direct"):
        raise Refused(f"launcher={launcher!r}: 'slot' (PREREG §9) or 'direct' (tests)")
    if threads < 1:
        raise Refused(f"threads={threads}")
    tree = tree_state()
    if tree["sha"] is None or tree["dirty"] is None:
        raise Refused(f"git did not answer: {tree}")
    if tree["dirty"] and not allow_dirty:
        raise Refused(
            "the tree has uncommitted changes outside runs/ (review MAJOR-5): run 1 "
            f"launches only at a clean committed head.\n{tree['porcelain']}"
        )
    for c in cfgs:
        for arm in arms:
            refuse_resume(arm_dir(out_root, arm, c.seed))
    c0 = {c.seed: check_c0(c) for c in cfgs}  # before any model loads
    Path(out_root).mkdir(parents=True, exist_ok=True)
    plan = {
        "arms": list(arms),
        "seeds": [c.seed for c in cfgs],
        "launcher": launcher,
        "threads_per_arm": threads,
        "parallel": parallel,
        "slot_wait_s": wait_s,
        "tree": tree,
        "allow_dirty": allow_dirty,
        "c0": c0,
        "thread_probe": probe_threads(threads),
        "argv": {
            f"{a}-seed{c.seed}": cli_argv(c, a, owner.gamma, arm_dir(out_root, a, c.seed))
            for c in cfgs
            for a in arms
        },
    }
    write_manifest(out_root, cfgs, owner, plan)
    jobs = [(c, a) for c in cfgs for a in arms]
    recs: dict[str, dict] = {}
    kw = dict(launcher=launcher, threads=threads, tree=tree, wait_s=wait_s)
    if parallel:
        started = [launch_arm(c, owner, a, out_root, **kw) for c, a in jobs]
        for p, log, rec in started:
            r = finish_arm(p, log, rec, out_root)
            recs[f"{r['arm']}-seed{r['seed']}"] = r
    else:
        for c, a in jobs:
            r = finish_arm(*launch_arm(c, owner, a, out_root, **kw), out_root)
            recs[f"{a}-seed{c.seed}"] = r
    return recs


# --------------------------------------------------------------------------- #
# The held-out harness (PREREG §7.1, §7.2, T3)
# --------------------------------------------------------------------------- #


class Recorder:
    """Wraps any policy for the held-out pass. Records the victims (`written_at`), each
    eviction's age and ADR-0006 displacement, and the IN-LOOP memory after every write
    -- read off the memory the loop holds (`on_write`'s `MemoryState`), not derived from
    the victims, so C7 compares two independent records.

    `observe` is refused unless ``forward_observe`` (LRU reads attention; an RSR arm
    never observes in the held-out harness, PREREG §7.1)."""

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
    """One policy object per batch row (a per-document oracle or random arm)."""

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
    decides over the gestalts written (§7.2: the floor "is also measured under each
    arm")."""

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
        return self.inner(
            ids_t, mask_t, torch.zeros_like(kv), valid, bos_ctx, bos_valid, **kw
        )


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
    policy that `needs_eval_context` -- training's decision path (ADR-0009 L2). With
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
                answers[b].append(
                    (t, int(gap[b, t]), bool(pred[b, j] == tgt[b, j]), float(per[b, j]))
                )
        return torch.zeros((), device=dev)

    rec = Recorder(policy, forward_observe=observe)
    net = ZeroSlots(model) if zero_slots else model
    eval_ctx = rec.needs_eval_context and decision_context == "eval_forward"
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            run_policy_loop(
                net,
                ids,
                mask,
                torch.full((B,), S, device=dev),
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
        out.append(
            {
                "doc": doc.doc_id,
                "answers": answers[b],
                "real_sum": real_sum[b],
                "real_n": real_n[b],
                "victims": {str(k): v for k, v in sorted(rec.victims.get(b, {}).items())},
                "evictions": rec.evictions.get(b, []),
                "hits": [
                    (q - assert_of[q], assert_of[q] in rec.resident_before(b, q))
                    for q in sorted(assert_of)
                ],
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


def reference_policy(name: str, docs: list, seed: int):
    """Evaluation-only references (§7.2; review MAJOR-1 adds LRU)."""
    if name == "oracle":
        return PerRow(
            [OraclePolicy(discounted_demand(d, ORACLE_GAMMA)) for d in docs], "oracle"
        )
    if name == "lru":
        return LRUPolicy()  # keyed by (row, written_at); reset at each loop entry
    if name.startswith("random"):
        k = int(name[len("random") :])
        return PerRow([B2().random_policy(k, seed, d.doc_id) for d in docs], name)
    raise ValueError(name)


def heldout_passes() -> list[tuple[str, str, bool]]:
    """Every held-out pass: (name, the arm whose checkpoint supplies model and policy,
    slots zeroed). `sz_fifo` is S0-03's floor; `sz_psiC` / `sz_age` are §7.2's "under
    each arm"."""
    passes = [(a, a, False) for a in ARMS]
    passes += [("oracle", "fifo", False), ("lru", "fifo", False)]
    passes += [(f"random{k}", "fifo", False) for k in range(N_RANDOM)]
    passes += [(f"sz_{a}", a, True) for a in ARMS]
    return passes


def run_heldout(
    cfg: Run1Config,
    payloads: dict[str, dict],
    out_dir: Path,
    *,
    passes=None,
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
    models = {a: model_from_payload(p).to(cfg.device) for a, p in payloads.items()}
    timing: dict[str, dict] = {}
    residency: dict[str, dict] = {}
    for name, src, zero in passes or heldout_passes():
        recs: list[dict] = []
        secs: list[float] = []
        for i in range(0, len(docs), cfg.eval_batch):
            chunk = docs[i : i + cfg.eval_batch]
            if name in ARMS or name.startswith("sz_"):
                pol = eval_policy(src, payloads[src], cfg)
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
# T7: is the oracle the same rule at every gamma in (0, 1)?
# --------------------------------------------------------------------------- #


def oracle_victims(doc, gamma: float, m: int) -> dict[int, int]:
    rec = Recorder(OraclePolicy(discounted_demand(doc, gamma)))
    simulate(doc, rec, m)
    return rec.victims.get(0, {})


def probe_oracle_invariance(docs, m: int, gammas=T7_GAMMAS) -> dict:
    """T7: `OraclePolicy`'s eviction sequence is identical at every gamma in ``gammas``
    on every document (model-free, `headroom.simulate`)."""
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
        out[arm] = {
            k: ws.get(k)
            for k in (
                "sha256",
                "source_vocab_sha256",
                "run_vocab_sha256",
                "vocab_checked",
                "seed_matches",
                "source_seed",
                "allow_unrecorded_vocabulary",
                "allow_seed_mismatch",
            )
        }
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
    them (`loop.py:1173-1191`) -- the inputs of the expected batch fingerprint."""
    vmap = vocab_map(cfg)
    docs = docs_by_id(cfg, range(*cfg.pool))
    L = CLI_MAX_TOKENS
    _ids, mask = loop.encode(docs, vmap, max_tokens=L, steps=cfg.steps_per_stream)
    tm, _ = loop.answer_targets(docs, vmap, max_tokens=L, steps=cfg.steps_per_stream)
    n_t = cfg.steps_per_stream * (L - 1)
    real = mask[:, :, 1:].sum(dim=(1, 2))
    ans = tm[:, :, 1:].sum(dim=(1, 2))
    return [
        (int(a), int(n_t - r)) for a, r in zip(ans.tolist(), real.tolist(), strict=True)
    ]


def c2_pairing(beats: dict[str, list[dict]], cfg: Run1Config) -> dict:
    """C2 (review MINOR-4): every step drew the same batch in every arm, and that batch
    is the one the sampler replay predicts; `loss_lm` bitwise equal for `k < T_warm`.

    The batch check compares `batch_fingerprint` across arms AND against the
    replayed sampler's expected fingerprint (`replay_batches` + `pool_fingerprints`).
    It is a fingerprint, not an identity: two different 16-document batches could
    share one, so a mismatch is proof and a match is strong evidence, not proof. The
    warm-step bitwise `loss_lm` check is the stronger one, where it applies."""
    t_warm = cfg.t_warm_steps
    arms = list(beats)
    by = {a: {b["step"]: b for b in beats[a]} for a in arms}
    steps = sorted(set.intersection(*(set(v) for v in by.values()))) if by else []
    fp = pool_fingerprints(cfg)
    expected = replay_batches(cfg, len(steps) and steps[-1] + 1)
    cross, vs_replay, lm = [], [], []
    for k in steps:
        prints = {a: batch_fingerprint(by[a][k], cfg) for a in arms}
        if len(set(prints.values())) != 1:
            cross.append(k)
        if fp is not None:
            idx = expected[k]
            want = ("counts", sum(fp[i][0] for i in idx), sum(fp[i][1] for i in idx))
            if any(p[0] == "counts" and p != want for p in prints.values()):
                vs_replay.append(k)
        if k < t_warm and len({by[a][k]["loss_lm"] for a in arms}) != 1:
            lm.append(k)
    same_steps = all(set(v) == set(steps) for v in by.values())
    return {
        "steps_compared": len(steps),
        "warm_steps_compared": sum(1 for k in steps if k < t_warm),
        "fingerprint": "batch_fingerprint (ADAPTER)",
        "cross_arm_mismatch_steps": cross[:20],
        "replay_mismatch_steps": vs_replay[:20],
        "replay_checked": fp is not None,
        "loss_lm_warm_mismatch_steps": lm[:20],
        "same_step_sets": same_steps,
        "ok": bool(steps) and same_steps and not cross and not vs_replay and not lm,
    }


def c3_frozen(payloads: dict[str, dict], substrate_model: dict) -> dict:
    out = {}
    for arm, p in payloads.items():
        m = p["model"]
        keys_equal = set(m) == set(substrate_model)
        differ = [
            k
            for k in substrate_model
            if k in m and not torch.equal(m[k], substrate_model[k])
        ]
        out[arm] = {
            "n_tensors": len(m),
            "keys_equal": keys_equal,
            "differ": differ[:10],
            "ok": keys_equal and not differ,
        }
    return {
        "per_arm": out,
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
    per = {
        a: {k: (r.get("liveness") or {}).get(k) for k in ("band", "ratio", "exit_code")}
        for a, r in results.items()
    }
    return {
        "per_arm": per,
        "ok": bool(per) and all(v["band"] == "live" for v in per.values()),
        "inert": [a for a, v in per.items() if v["band"] == "inert"],
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


def analyse_heldout(out_dir: Path, cfg: Run1Config) -> dict:
    """§7.1 and §7.2 from the per-document records of one seed."""
    names = [p[0] for p in heldout_passes()]
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
    out: dict[str, Any] = {
        "n_documents": len(order),
        "n_boot": cfg.n_boot,
        "boot_seed": BOOT_SEED_BASE + cfg.seed,
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
                con[f"floor_difference.sz_{arm}_minus_sz_fifo"] = contrast(
                    per, f"sz_{arm}", "sz_fifo", W
                )
        if bname == "all":
            out["headroom_share"] = {a: share(per, a, W) for a in RSR_ARMS}
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
# Analysis: training time (§7.3) and the §7.1 diagnostics (§7.4)
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


def analyse_training(
    hbs: dict[str, dict], diags: dict[str, list[dict]], cfg: Run1Config
) -> dict:
    """§7.3 (paired per step; in-sample for phi; descriptive) and §7.4 (warm / scored)."""
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

        def g_by_t(bs):
            acc: dict[int, list[float]] = {}
            for b in bs:
                for t, v in ((b.get("mc") or {}).get("g_mean_by_t") or {}).items():
                    acc.setdefault(int(t), []).append(float(v))
            return {t: sum(v) / len(v) for t, v in sorted(acc.items())}

        def mc(b, key):
            return (b.get("mc") or {}).get(key)

        fills = [
            mc(b, "n_samples") / (mc(b, "n_groups") * cfg.memory_slots)
            for b in beats
            if mc(b, "n_groups")
        ]
        out["rsr"][a] = {
            "loss_mc_warm": _summary([b.get("loss_mc") for b in warm]),
            "loss_mc_post_warm": _summary([b.get("loss_mc") for b in post]),
            "loss_mc_blocks": _blocks(
                [(b["step"], b.get("loss_mc")) for b in beats], 0, cfg.iters
            ),
            "g_mean_by_t_warm": g_by_t(warm),
            "g_mean_by_t_post_warm": g_by_t(post),
            "corr_psi_age_warm": _summary([mc(b, "corr_psi_age") for b in warm]),
            "corr_psi_age_post_warm": _summary([mc(b, "corr_psi_age") for b in post]),
            "corr_psi_age_none_steps": sum(
                1 for b in beats if mc(b, "corr_psi_age") is None
            ),
            "phi_grad_norm": _summary([b.get("phi_grad_norm") for b in beats]),
            "eval_forward_s": {
                **_summary([b.get("eval_forward_s") for b in beats]),
                "total": sum(b.get("eval_forward_s") or 0.0 for b in beats),
            },
            "fill_fraction_regression_rows": sum(fills) / len(fills) if fills else None,
        }
    out["step_s"] = {}
    for a, hb in hbs.items():
        el = [b["elapsed_s"] for b in hb["beats"]]
        out["step_s"][a] = _summary([y - x for x, y in itertools.pairwise(el)])
    out["diagnostics"] = {}
    for a, rows in diags.items():
        out["diagnostics"][a] = {}
        for phase, sel in (("warm", True), ("scored", False)):
            rs = [r for r in rows if r["warm"] is sel]
            shift: Counter = Counter()
            ages: Counter = Counter()
            for r in rs:
                shift.update(r["shift"])
                ages.update(r["victim_age"])
            tot = sum(shift.values())
            out["diagnostics"][a][phase] = {
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
    return out


# --------------------------------------------------------------------------- #
# measure (per seed), then finalize (across seeds)
# --------------------------------------------------------------------------- #


def measure_seed(
    cfg: Run1Config,
    owner: OwnerValues,
    out_root: Path,
    *,
    threads: int,
    limit: int | None = None,
    c4_fn=None,
) -> dict:
    """Every control and number for one seed -> ``measure-seed{s}.json``. Never raises
    for a failed control: the failure is recorded and the status says so."""
    torch.set_num_threads(threads)
    out_root = Path(out_root)
    doc: dict[str, Any] = {
        "run_id": RUN_ID,
        "seed": cfg.seed,
        "config": cfg.as_dict(),
        "owner_values": dataclasses.asdict(owner),
        "tree": tree_state(),
        "hardware": hardware(),
        "measure_threads": torch.get_num_threads(),
        "controls": {},
        "status": None,
        "exit": None,
        "notes": [],
    }
    arms = {}
    for a in ARMS:
        p = out_root / f"arm-{a}-seed{cfg.seed}.json"
        arms[a] = json.loads(p.read_text()) if p.exists() else None
    doc["arms"] = {
        a: None
        if r is None
        else {
            k: r.get(k)
            for k in (
                "argv",
                "full_argv",
                "launcher",
                "threads",
                "rc",
                "rc_meaning",
                "run_id",
                "config_hash",
                "arm_label",
                "git_sha",
                "git_dirty",
                "wall_s",
                "started_utc",
                "finished_utc",
                "heartbeat_exists",
            )
        }
        for a, r in arms.items()
    }

    def done(code: Exit, status: str) -> dict:
        failed = [
            k
            for k, v in doc["controls"].items()
            if isinstance(v, dict) and v.get("ok") is not True
        ]
        doc.update(status=status, exit=int(code), controls_failed=failed)
        (out_root / f"measure-seed{cfg.seed}.json").write_text(
            json.dumps(doc, indent=2, default=str) + "\n"
        )
        return doc

    try:
        doc["controls"]["C0"] = check_c0(cfg)
    except ControlFailure as e:
        doc["controls"]["C0"] = {"ok": False, "error": str(e)}
        return done(Exit.DID_NOT_RUN, "did_not_run")
    missing = [a for a, r in arms.items() if r is None or r.get("result") is None]
    if missing:
        doc["notes"].append(
            "arms with no result: "
            f"{[(a, (arms[a] or {}).get('rc_meaning')) for a in missing]}"
        )
        return done(Exit.DID_NOT_RUN, "crashed")
    results = {a: r["result"] for a, r in arms.items()}
    hbs = {a: read_heartbeat(arm_dir(out_root, a, cfg.seed)) for a in ARMS}
    doc["frozen_model_mode"] = {a: frozen_model_mode(r) for a, r in results.items()}
    doc["controls"]["C1"] = c1_warm_start(results, cfg)
    doc["controls"]["C5"] = c5_liveness(results)
    doc["controls"]["C6"] = c6_steps(results, hbs, cfg)
    ran = {
        a: (results[a]["config"].get("policy_overrides") or {}).get("gamma")
        for a in RSR_ARMS
    }
    wd = {
        a: (results[a]["config"].get("optimizer") or {}).get("weight_decay") for a in ARMS
    }
    doc["controls"]["owner_values_ran"] = {
        "gamma": ran,
        "weight_decay": wd,
        "launched_clean": {a: (arms[a] or {}).get("git_dirty") is False for a in ARMS},
        "ok": all(v == owner.gamma for v in ran.values())
        and all(v == owner.phi_weight_decay for v in wd.values()),
    }
    if doc["controls"]["C5"]["inert"]:
        doc["notes"].append(
            "an arm measured INERT (R-2026-09-22-inert-exit-5): quarantined; "
            "no retention number is reported"
        )
        return done(Exit.INERT, "failed")
    if not doc["controls"]["C6"]["ok"]:
        return done(Exit.DID_NOT_RUN, "crashed")
    beats = {a: hbs[a]["beats"] for a in ARMS}
    doc["controls"]["C2"] = c2_pairing(beats, cfg)
    doc["controls"]["C8"] = c8_finite(beats)
    ckpts = {a: final_checkpoint(arm_dir(out_root, a, cfg.seed), cfg.iters) for a in ARMS}
    payloads = {a: read_checkpoint(p) for a, p in ckpts.items()}
    sub_model, sub_sha = load_substrate_model(cfg)
    doc["controls"]["C3"] = {
        **c3_frozen(payloads, sub_model),
        "substrate_sha256": sub_sha,
    }
    del sub_model
    doc["checkpoint_bytes"] = {a: p.stat().st_size for a, p in ckpts.items()}  # T6
    eval_docs = docs_by_id(cfg, list(cfg.eval_ids())[: limit or None])
    doc["controls"]["T7"] = probe_oracle_invariance(eval_docs, cfg.memory_slots)
    hd = out_root / "heldout" / f"seed{cfg.seed}"
    ho = run_heldout(cfg, payloads, hd, limit=limit)
    doc["controls"]["C7"] = {
        "per_pass": ho["residency"],
        "ok": all(v["n_mismatched"] == 0 for v in ho["residency"].values()),
    }
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
        doc["controls"]["C4"] = (c4_fn or c4_measure)(
            arm_dir(out_root, "fifo", cfg.seed), cfg
        )
    except Exception as e:  # the control did not run: recorded, never a pass
        doc["controls"]["C4"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    doc["heldout"] = analyse_heldout(hd, cfg)
    diags = {a: read_step_diagnostics(arm_dir(out_root, a, cfg.seed)) for a in RSR_ARMS}
    doc["training"] = analyse_training(hbs, diags, cfg)
    doc["epoch"] = {a: results[a]["config"].get("epoch") for a in RSR_ARMS}
    failed = [
        k
        for k, v in doc["controls"].items()
        if isinstance(v, dict) and v.get("ok") is not True
    ]
    if failed:
        doc["notes"].append(
            f"controls failed: {failed} -- inconclusive; numbers NOT read"
        )
        return done(Exit.DID_NOT_RUN, "failed")
    return done(Exit.OK, "ok")


def headline(docs: list[dict]) -> dict:
    """The ONE designated contrast (`HEADLINE`), per seed (within-seed CI, labelled so)
    and across seeds (per-seed points, mean, sd over seeds; no across-seed CI)."""
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
    return {
        "contrast": HEADLINE,
        "per_seed": per,
        "across_seeds": {
            "n_seeds": n,
            "points": pts,
            "mean": mean,
            "sd": sd,
            "sd_exactly_zero": sd == 0.0,
            "readings": [v.get("reading") for v in per.values() if v],
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
    (INERT 5 over did-not-run 3 over 0)."""
    codes = [Exit(d["exit"]) for d in docs]
    code = (
        Exit.INERT
        if Exit.INERT in codes
        else (Exit.DID_NOT_RUN if any(c != Exit.OK for c in codes) else Exit.OK)
    )
    status = (
        "ok" if code == Exit.OK else next(d["status"] for d in docs if d["exit"] != 0)
    )
    run = {
        "run_id": RUN_ID,
        "seeds": [d["seed"] for d in docs],
        "exit": int(code),
        "status": status,
        "owner_values": dataclasses.asdict(owner),
        "headline": headline(docs),
        "tree": tree_state(),
    }
    (Path(out_root) / "run.json").write_text(
        json.dumps(run, indent=2, default=str) + "\n"
    )
    Path(results_path).write_text(render_results(run, docs))
    if write_ledger:
        write_run_ledger(run, docs, Path(out_root))
    return code


def write_run_ledger(run: dict, docs: list[dict], out_root: Path) -> Path:
    """Every number into `runs/rsr-run1/ledger.json` (scripts/ledger.py)."""
    L = ledger_mod()
    led = L.Ledger(RUN_ID, question=QUESTION, runs_root=Path(out_root).parent)
    man = Path(out_root) / "manifest.json"
    if man.exists():  # the manifest was frozen at launch; its hash is adopted, not redone
        led.doc["config_hash"] = hashlib.sha256(man.read_bytes()).hexdigest()
        led.doc["manifest_written_utc"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(man.stat().st_mtime)
        )
    for d in docs:
        for a, r in (d.get("arms") or {}).items():
            if r is not None:
                led.command(
                    r["argv"],
                    exit_code=r.get("rc"),
                    note=f"arm {a} seed {d['seed']} via {r.get('launcher')}: "
                    f"{r.get('rc_meaning')}",
                )
    led.command(
        [str(ROOT / ".venv" / "bin" / "python"), EXPERIMENT, "measure"],
        exit_code=run["exit"],
        note="this measurement",
    )
    done = [
        ((d.get("controls") or {}).get("C6") or {}).get("per_arm") or {} for d in docs
    ]
    led.run_meta(
        device=DEVICE,
        seeds_actually_run=run["seeds"],
        steps_requested=run["owner_values"]["iters"],
        steps_done=min(
            (v.get("steps_done") or 0 for p in done for v in p.values()), default=0
        ),
    )
    led.note("headline", run["headline"], how=f"{EXPERIMENT} run.json")
    for d in docs:
        for key, val in _flatten(
            f"seed{d['seed']}",
            {
                k: d.get(k)
                for k in (
                    "controls",
                    "heldout",
                    "training",
                    "T5",
                    "checkpoint_bytes",
                    "epoch",
                    "arms",
                )
            },
        ):
            led.note(key, val, how=f"{EXPERIMENT} measure-seed{d['seed']}.json")
    led.status(run["status"])
    return led.write()


def _flatten(prefix: str, x: Any, depth: int = 0):
    if isinstance(x, dict) and depth < 4:
        for k, v in x.items():
            yield from _flatten(f"{prefix}.{k}", v, depth + 1)
    else:
        yield prefix, x


# --------------------------------------------------------------------------- #
# RESULTS.md
# --------------------------------------------------------------------------- #


def prereg_section(start: str, stop: str) -> str:
    """A PREREG passage, read from the file: quoted, never retyped."""
    text = (ROOT / PREREG).read_text()
    i = text.index(start)
    return text[i : text.index(stop, i + len(start))].rstrip()


def _f(x, nd: int = 4) -> str:
    if x is None:
        return "None"
    if isinstance(x, (bool, int)):
        return str(x)
    if isinstance(x, float):
        return "nan" if x != x else f"{x:+.{nd}f}"
    return str(x)


BROKEN = " **sd exactly 0: broken, not clean**"


def _ci_str(c: dict | None, *, reading: bool = True) -> str:
    if not c:
        return "not measured"
    flag = BROKEN if c.get("sd_exactly_zero") else ""
    rd = f" {c.get('reading')}" if reading and c.get("reading") else ""
    lo, hi = _f(c.get("lo")), _f(c.get("hi"))
    return f"{_f(c.get('point'))} [{lo}, {hi}]{rd}{flag}"


def _join(*parts: object) -> str:
    return "".join(str(p) for p in parts)


def render_results(run: dict, docs: list[dict]) -> str:
    """RESULTS.md (PREREG §14b). Every number is read from `run.json` and the
    `measure-seed*.json` documents; nothing is typed."""
    L: list[str] = []
    w = L.append
    w("# RESULTS: RSR run 1 -- psi_hat trained online on frozen arm B ckpt3000\n")
    w(
        "**Descriptive. No gate, no pass/fail threshold.** Every contrast is read "
        "only as CI above 0 / CI below 0 / CI includes 0.\n"
    )
    w(f"> {SCOPE_SENTENCE}\n>\n> {MULTIPLICITY_SENTENCE}\n")
    if run["status"] != "ok":
        w(
            _join(
                "> 🔴 **NOT READ.** status `", run["status"], "`, exit `", run["exit"],
                "`. Every number below is reported and none is read "
                "(PREREG §8, §11.7).\n",
            )
        )  # fmt: skip
    w("## 1. Headline (the one designated contrast)\n")
    h = run["headline"]
    w(f"`{h['contrast']}` (psiC - FIFO, all-query held-out answer accuracy):\n")
    for s, c in h["per_seed"].items():
        w(f"- seed {s}: {_ci_str(c)} (CI is within-seed only)")
    a = h["across_seeds"]
    pts = [round(x, 6) for x in a["points"]]
    w(
        _join(
            "- across seeds: n = ", a["n_seeds"], ", points ", pts, ", mean ",
            _f(a["mean"]), ", sd over seeds ", _f(a["sd"]),
            BROKEN if a["sd_exactly_zero"] else "", "\n",
        )
    )  # fmt: skip
    w("## 2. Commands, exit codes, provenance\n")
    tree = run.get("tree") or {}
    w(
        _join(
            "- measure: `.venv/bin/python ", EXPERIMENT, " measure` -> exit `",
            run["exit"], "`, status `", run["status"], "`; tree `", tree.get("sha"),
            "` dirty `", tree.get("dirty"), "`; PREREG `", PREREG_COMMIT,
            "`, review `", REVIEW_COMMIT, "`",
        )
    )  # fmt: skip
    w(f"- owner values: `{run['owner_values']}`")
    for d in docs:
        for arm, r in (d.get("arms") or {}).items():
            if r is None:
                w(f"- seed {d['seed']} **{arm}**: no record")
                continue
            w(
                _join(
                    "- seed ", d["seed"], " **", arm, "**: rc `", r.get("rc"), "` (",
                    r.get("rc_meaning"), "), run_id `", r.get("run_id"),
                    "`, config_hash `", r.get("config_hash"), "`, label `",
                    r.get("arm_label"), "`, git `", r.get("git_sha"), "` dirty `",
                    r.get("git_dirty"), "`, launcher `", r.get("launcher"),
                    "`, threads `", r.get("threads"), "`, wall `",
                    _f(r.get("wall_s"), 1), "` s",
                )
            )  # fmt: skip
            w(f"  - `{shlex.join(r.get('full_argv') or r.get('argv') or [])}`")
    hw = (docs[0].get("hardware") if docs else None) or {}
    w(
        _join(
            "\n**Hardware:** ", hw.get("cpu"), ", ", hw.get("platform"), ", CPU, torch ",
            hw.get("torch"), ", Python ", hw.get("python"), ".\n",
        )
    )  # fmt: skip
    for d in docs:
        w(
            _join(
                "## Seed ", d["seed"], " (status `", d["status"], "`, exit `", d["exit"],
                "`; everything in this section is secondary or diagnostic)\n",
            )
        )  # fmt: skip
        _render_seed(d, w)
    w("## What this does not show\n")
    w(
        "Not E1 and not E2 (ADR-0009 L6). Not evidence on the cognitive claim. No "
        "scaling claim. One corpus (synthetic S0-03), `M = 16`, one width, a frozen "
        "substrate, the shadow buffer off, `b` off. The rescaled target is reported as "
        "rescaled and nothing stronger. Training-time differences are in-sample for "
        "phi. Within-seed CIs do not estimate run-to-run variance. The frozen model's "
        "training-time mode is recorded per seed (`frozen_model_mode`); the held-out "
        "pass runs it in eval mode.\n"
    )
    w("## The author's expectation (PREREG §7.6, verbatim), and the outcome beside it\n")
    w(prereg_section("### 7.6 Author's expectation", "## 8. Controls"))
    w(
        "\nThe outcome, as numbers in the expectation's terms (reported as text; "
        "never scored as met or missed):\n"
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
                    "- seed ", d["seed"], " `", arm, "` scored: fraction_shifting ",
                    _f(s.get("fraction_shifting")), ", fraction oldest ",
                    _f(s.get("fraction_oldest")), ", raw rho median ",
                    _f(s["spearman_rho_score_age"]["median"]), ", partial rho median ",
                    _f(s["partial_rho_diagnostic"]["median"]),
                )
            )  # fmt: skip
    for key in EXPECTATION_KEYS:
        if key in con:
            w(f"- seed {d['seed']} `{key}`: {_ci_str(con[key], reading=False)}")
    w(f"- seed {d['seed']} controls failed: `{d.get('controls_failed')}`")


CONTROL_KEYS = ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "T7")


def _render_seed(d: dict, w) -> None:
    ctl = d.get("controls") or {}
    if d.get("notes"):
        w(f"Notes: {d['notes']}\n")
    w("### Controls (PREREG §8; T7)\n")
    for k in (*CONTROL_KEYS, "owner_values_ran"):
        v = ctl.get(k)
        if v is None:
            w(f"- **{k}**: not run")
        else:
            w(f"- **{k}**: ok = `{v.get('ok')}` -- `{json.dumps(v, default=str)[:700]}`")
    w("\n### What ran\n")
    for a, v in ((ctl.get("C6") or {}).get("per_arm") or {}).items():
        w(
            _join(
                "- ", a, ": steps_requested `", v.get("steps_requested"),
                "`, steps_done `", v.get("steps_done"), "`, status `",
                v.get("status"), "`",
            )
        )  # fmt: skip
    for a, e in (d.get("epoch") or {}).items():
        if e:
            w(
                _join(
                    "- ", a, ": T_warm `", e.get("t_warm_steps"), "` steps = `",
                    e.get("t_warm_epochs"), "` epoch; total epochs `",
                    e.get("total_epochs"), "`; T_warm / total epochs `",
                    e.get("t_warm_fraction_of_total_epochs"), "` (31(b)); sampling `",
                    e.get("sampling"), "`",
                )
            )  # fmt: skip
    if d.get("frozen_model_mode"):
        w(f"- frozen model mode during training: `{d['frozen_model_mode']}`")
    if d.get("checkpoint_bytes"):
        w(f"- final checkpoint bytes (T6): `{d['checkpoint_bytes']}`")
    ho = d.get("heldout") or {}
    _render_heldout(ho, w)
    _render_training(d.get("training") or {}, w)
    w("\n### §7.5 reporting\n")
    w(f"- fill fraction, held-out in-loop: {_f(ho.get('fill_fraction_heldout'))}")
    w("- The censored-target caveat, verbatim from the PREREG:\n")
    w(prereg_section("  > The shadow buffer is off", "- Every number is reported"))
    w("\n### T5 (held-out cost, measured in this run)\n")
    for n, t in (d.get("T5") or {}).items():
        w(
            _join(
                "- `", n, "`: first batch ", _f(t["first_batch_s"], 2),
                " s; projected full EVAL ", _f(t["projected_full_eval_s"], 1),
                " s; measured ", _f(t["measured_total_s"], 1), " s over ",
                t["batches_run"], " batches",
            )
        )  # fmt: skip
    w("")


def _render_heldout(ho: dict, w) -> None:
    acc = ho.get("accuracy") or {}
    if acc:
        w(f"\n### Held-out accuracy (§7.1), EVAL n = {ho.get('n_documents')} documents\n")
        w(
            "Pooled answer accuracy, model-read (argmax over the full vocabulary); 95% "
            "percentile CI over 2000 paired per-document resamples (within-seed).\n"
        )
        cols = ("psiC", "fifo", "age", "lru", "oracle", "random")
        w("| bucket | n | " + " | ".join(cols) + " |")
        w("|---|---|" + "---|" * len(cols))
        for b in BUCKET_NAMES:
            row = acc.get(b) or {}
            cells = " | ".join(_ci_str(row.get(n), reading=False) for n in cols)
            w(f"| {b} | {(row.get('fifo') or {}).get('n')} | {cells} |")
    con = ho.get("contrasts") or {}
    if not con:
        return
    w("\n### Contrasts (secondary; the headline is §1)\n")
    for key in sorted(con):
        tag = " **(HEADLINE)**" if key == HEADLINE else ""
        w(f"- `{key}`{tag}: {_ci_str(con[key])}")
    for kind in ("real", "answer"):
        row = (ho.get("nll") or {}).get(kind) or {}
        lv = "; ".join(
            f"{n} {_f((row.get(n) or {}).get('point'))}"
            for n in ("psiC", "fifo", "age", "lru")
        )
        w(f"- NLL `{kind}` level: {lv}")
    for a, s in (ho.get("headroom_share") or {}).items():
        w(
            f"- headroom share `{a}` (all): {_ci_str(s, reading=False)}, "
            f"dropped {s.get('dropped')}"
        )
    for n, e in (ho.get("heldout_evictions") or {}).items():
        w(
            _join(
                "- held-out evictions `", n, "`: n ", e["n_evictions"],
                ", fraction oldest ", _f(e["fraction_oldest"]), ", victim ages `",
                e["victim_age_distribution"], "`",
            )
        )  # fmt: skip


def _render_training(tr: dict, w) -> None:
    if not tr:
        return
    w("\n### Training time, on the pool (§7.3; in-sample for phi; descriptive)\n")
    for a, dl in (tr.get("delta_loss_lm") or {}).items():
        blocks = [(b["steps"], _f(b["mean"])) for b in dl.get("blocks", [])]
        w(
            _join(
                "- delta loss_lm `", a, "` - fifo, post-warm: mean ", _f(dl.get("mean")),
                ", sd ", _f(dl.get("sd")), BROKEN if dl.get("sd_exactly_zero") else "",
                "; 500-step blocks ", blocks,
            )
        )  # fmt: skip
    for a, r in (tr.get("rsr") or {}).items():
        w(
            _join(
                "- `", a, "`: L_MC warm mean ", _f(r["loss_mc_warm"]["mean"]),
                ", post-warm mean ", _f(r["loss_mc_post_warm"]["mean"]),
                "; corr_psi_age post-warm median ",
                _f(r["corr_psi_age_post_warm"]["median"]), " (None on ",
                r["corr_psi_age_none_steps"], " steps); phi grad norm median ",
                _f(r["phi_grad_norm"]["median"]), "; eval_forward_s total ",
                _f(r["eval_forward_s"]["total"], 1), "; fill fraction (regression rows) ",
                _f(r["fill_fraction_regression_rows"]),
            )
        )  # fmt: skip
    w("\n### §7.1 diagnostics (§7.4; logged, never a gate; split warm / scored)\n")
    for a, ph in (tr.get("diagnostics") or {}).items():
        for p, v in ph.items():
            pr = v["partial_rho_diagnostic"]
            w(
                _join(
                    "- `", a, "` ", p, ": steps ", v["n_steps"], ", evictions ",
                    v["n_evictions"], ", fraction oldest ", _f(v["fraction_oldest"]),
                    ", fraction_shifting ", _f(v["fraction_shifting"]), ", mean_shift ",
                    _f(v["mean_shift"]), "; raw Spearman rho(score, age) median ",
                    _f(v["spearman_rho_score_age"]["median"]),
                    "; **partial rho (a diagnostic, never a gate)** median ",
                    _f(pr["median"]), " IQR [", _f(pr["q25"]), ", ", _f(pr["q75"]),
                    "], None on ", v["partial_rho_none_steps"], " steps",
                )
            )  # fmt: skip
            w(
                _join(
                    "  - victim ages `", v["victim_age_distribution"],
                    "`; displacement `",
                    v["displacement_distribution"], "`",
                )
            )  # fmt: skip
    w(
        "- Partial rho partials out the slot gestalts only (linear), not `c_t`; no "
        "verdict is drawn from any comparison with the spec's E2 value. Leading-edge "
        "statistic: not computed (the spec defines a baseline, not a statistic; "
        "`leading_edge.py` is a stub). LRU runs as an evaluation-only reference "
        "(review MAJOR-1). Attention-share Gini and heartbeat attribution: not "
        "available."
    )


# --------------------------------------------------------------------------- #
# Build-time measurement: T5 (timing only)
# --------------------------------------------------------------------------- #


def time_one_batch(cfg: Run1Config, doc_ids: list[int]) -> dict:
    """T5: one batch per pass on the REAL substrate, read-only, the process's threads.

    **Timing only**: no accuracy or loss is returned or printed. RSR arms run a FRESH
    head (no trained phi exists before the run; the cost does not depend on its values).
    ``doc_ids`` must not be EVAL: EVAL stays unseen until the run (PREREG §14a)."""
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
    spe = cfg.pool_length / cfg.batch
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
                steps_per_epoch=spe,
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
    p.add_argument("--gamma", type=float, required=True, help="O1: owner-ruled")
    p.add_argument("--seeds", type=int, nargs="+", required=True, help="O2: ruled list")
    p.add_argument(
        "--phi-weight-decay", type=float, required=True, help="O4: owner-ruled"
    )
    p.add_argument("--iters", type=int, required=True, help="run length (review R-1)")


def main(argv: list[str] | None = None) -> Exit:
    p = ArgumentParser(prog="rsr-run1", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("launch", help="every (arm, seed) through the slot wrapper")
    _owner_args(pl)
    pl.add_argument("--threads", type=int, required=True)
    pl.add_argument("--slot-wait", type=float, required=True, help="seconds; slot --wait")
    pl.add_argument("--parallel", action="store_true")
    pl.add_argument("--out-root", default=str(ROOT / "runs" / RUN_ID))
    pm = sub.add_parser("measure")
    _owner_args(pm)
    pm.add_argument("--threads", type=int, required=True)
    pm.add_argument("--out-root", default=str(ROOT / "runs" / RUN_ID))
    pm.add_argument(
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
        if a.cmd in ("launch", "measure"):
            owner = OwnerValues(a.gamma, tuple(a.seeds), a.phi_weight_decay, a.iters)
            validate_owner_values(owner)
            cfgs = [prereg_config(s, owner.iters) for s in owner.seeds]
            if a.cmd == "launch":
                recs = launch(
                    cfgs,
                    owner,
                    Path(a.out_root),
                    launcher="slot",
                    threads=a.threads,
                    parallel=a.parallel,
                    wait_s=a.slot_wait,
                )
                print(
                    json.dumps(
                        {
                            f"rc_{k}": {"rc": v["rc"], "meaning": v["rc_meaning"]}
                            for k, v in recs.items()
                        },
                        indent=2,
                    )
                )
                return (
                    Exit.OK
                    if all(v["rc"] == 0 for v in recs.values())
                    else Exit.DID_NOT_RUN
                )
            docs = [
                measure_seed(c, owner, Path(a.out_root), threads=a.threads) for c in cfgs
            ]
            return finalize(docs, owner, Path(a.out_root), results_path=Path(a.results))
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
