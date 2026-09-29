"""B2: the ψ̂ probe (PLAN-v4 §2B B2), with E0h (falsifier 3c) as its own entry point.

Governing text: `experiments/b2-psi-probe/PREREG.md` -- the base (`e5ab398`),
Amendment 1 (`2ec60dd`, which **wins** wherever the two differ) and the B0
"already seen" addendum (`99a6626`). Every rule below cites the section it
transcribes. Nothing here is a threshold chosen after data: the only numbers this
build is allowed to add to the PREREG are **n** (TBD-1, TBD-3) and the **capture
cost** (TBD-2), each by a shown diff (A1.1).

What it is (§1): an offline, closed-form, fp64 ridge of the spec's bilinear head
`ψ̂ = s_iᵀ W c_t + uᵀ[s_i; c_t]` (§3.2.2, §3.3) onto the Monte-Carlo return G
(§3.4, correction 2), fitted on stop-grad `s_i` and `c_t` captured from a fresh FIFO
rollout of frozen fresh-stream arm B, then run **as an eviction rule through the
model** one document at a time. **Nothing is trained by gradient**: the transformer
and `W_sent` are read only (CLAUDE.md, "do not backpropagate the retention loss").

Structurally absent (§1, §4): age as a ψ̂ input (the bilinear arms), `b`, nu,
shadow, warmup, and `RSRPolicy` -- `ProbeArgminPolicy` below is local to this
experiment and sets no precedent for ADR-0009. No `rsr.constants.record()` call.

Entry points (exit codes: `rsr.exit_codes`)::

    run.py check                 # §3/A1.15 range, aliasing guards; no model (0 / 3)
    run.py size                  # TBD-1/TBD-2 sizing, FIT_VAL only (0 / 3)
    run.py fit                   # phase A: capture, fits, FIT_VAL ref/δ/power (0/1/3)
    run.py eval                  # phase B: EVAL arms; refuses while N_E is TBD (3)
    run.py e0h                   # E0h, its OWN rc (A1.3): 0/1/2/3

🔴 `eval` refuses (exit 3) while `N_E` is `None`: §9.4 item 6 has N_E written into
the PREREG by diff **before any EVAL document is run**.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import re
import resource
import subprocess
import sys
import time
import traceback
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rsr.baselines.fifo import FIFOPolicy  # noqa: E402
from rsr.baselines.oracle import OraclePolicy  # noqa: E402
from rsr.baselines.random_policy import RandomPolicy  # noqa: E402
from rsr.data.synthetic import (  # noqa: E402
    ANSWER_SYMBOLS,
    SyntheticConfig,
    _generate_document,
    discounted_demand,
)
from rsr.exit_codes import (  # noqa: E402
    ArgumentParser,
    Exit,
    did_not_run,
    run_main,
    status,
)
from rsr.metrics.headroom import simulate  # noqa: E402
from rsr.model.tg.model import init_memory  # noqa: E402
from rsr.model.tg.policy_loop import (  # noqa: E402
    cross_capture,
    run_policy_loop,
    trace_for_row,
    write_at,
)
from rsr.retention.instrumentation import rank_shift  # noqa: E402
from rsr.train.loop import answer_targets, build_vocab, encode  # noqa: E402


def _load(name: str, rel: str):
    """Import an experiment script by path. Imported, never copied."""
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: W10's harness (lookahead-room PREREG "Definitions"): `r_of`, the discounted
#: return, W6's fact/filler, the scripted replay, C1 against the fresh-stream ledger.
LR = _load("_b2_lookahead_room", "experiments/lookahead-room/run.py")
S003 = LR.S003
CSC = LR.CSC

EXPERIMENT = "experiments/b2-psi-probe/run.py"
PREREG = "experiments/b2-psi-probe/PREREG.md"
PREREG_COMMITS = {"base": "e5ab398", "amendment1": "2ec60dd", "b0_addendum": "99a6626"}
RUN_ID = "b2-psi-probe"
SIZING_RUN_ID = "b2-psi-probe-sizing"

# --------------------------------------------------------------------------- #
# 🔒 PREREG, transcribed. Changing any of these is changing the PREREG.
# --------------------------------------------------------------------------- #

SEEDS = [0, 1, 2]
CHECKPOINTS = (2500, 3000)
HEADLINE = 3000  # §2: ckpt3000 primary, ckpt2500 reported, never gating
CKPT_SHA256 = dict(LR.CKPT_SHA256)  # §2: the pins lookahead-room pinned
PRIMARY_GAMMA = 0.9
GAMMAS = (0.9, 0.0)  # §5: 0.9 primary, 0 secondary (E0h's input)
M_STEPS = S003.CONFIG["memory_slots"]  # asserted against the loaded model (§2)
S_STEPS = S003.CONFIG["steps_per_stream"]
L_TOKENS = S003.CONFIG["max_tokens"]

#: §3 / A1.15: B2's ranges (per seed), mutually disjoint and fresh.
RANGES: dict[str, tuple[int, int]] = {
    "FIT_TRAIN": (920000, 936000),
    "FIT_VAL": (936000, 937024),
    "EVAL": (940000, 980000),
}
#: §3: reserved for E0d, never used by B2.
E0D_RESERVED = (900000, 920000)
#: §3 "Ranges already used", plus A1.15's E0d range (run/e0d@5357ad2).
USED_RANGES: dict[str, tuple[int, int]] = {
    "probe_vocab_S003_n64_armB_0_999": (0, 64),
    "S003_heldout": (64, 128),
    "csc_N512": (0, 512),
    "csc_N4096": (0, 4096),
    "E_EXT_readability_lookahead_carryforward": (64, 1088),
    "P_H64_fresh_stream_heldout": (4096, 4160),
    "fresh_stream_stream_0_2999": (4160, 52160),
    "fresh_escape_stream_3000_8999": (52160, 148160),
    "B5_continuation_to_55000": (4160 + 16 * 3000, 4160 + 16 * 55000),
    "E0d_A1": (262144, 263168),
}
GEN_SEED_K = 1_000_003  # synthetic.py: generator seed = seed * K + doc_id
E0D_PREREG = ROOT / "experiments" / "e0d" / "PREREG.md"

RANKS = (0, 1, 2, 3)  # §5: rank 0 gating; ranks 1-3 the sensitivity arm
N_V = RANGES["FIT_VAL"][1] - RANGES["FIT_VAL"][0]  # 1024, all used
N_BOOT = 2000
BOOT_SEED_BASE = 20260927
LAMBDAS = [10.0**k for k in range(-6, 5)]  # §6: 1e-6 .. 1e4
ELIGIBLE_RESID = 1e-6  # A1.6: a grid point over this is ineligible (logged)
SELECTED_RESID = 1e-8  # A1.6: the selected lambda must meet this, else exit 1
SUM_TOL = 1e-5  # §11 control 5 (E0e's SUM_TOL)
IDENTITY_TOL = 1e-6  # §11 control 4 (W10's C4)
LOGIT_TOL = 1e-5  # A1.5 control
DELTA_FRACTION = 0.25  # §9.3
N_E_MIN, N_E_MAX = 1024, 40000  # §9.4 item 4
TIER1_BUDGET_H = 72.0  # A1.4
TIER2_BUDGET_H = 24.0  # A1.4 author's note
TIER2_MAX_DOCS = 4096  # A1.4
E0H_COLLINEAR = 0.90  # §10, PROPOSED (A1.8): not read until ratified
E0H_NOT_COLLINEAR = 0.49
N_DETERMINISM_DOCS = 8  # §11 control 8
N_RANDOM = 5  # §7: random x5

#: TBD-3 (§9.4 item 6): written into the PREREG by diff before any EVAL document
#: is run (PREREG "Addendum: TBD-3, N_E", 3a90fce), and then here. The floor 1024
#: of item 4 binds (largest required size 166; runs/b2-psi-probe-fit/ledger.json
#: key N_E.power). `run_eval` re-derives it from the phase-A files and refuses a
#: mismatch. `None` makes `eval` exit 3.
N_E: int | None = 1024

#: §9.4 item 1 as amended by A1.12: only these six contrasts size N_E.
GATING_CONTRASTS = (
    "psiU_minus_refU",
    "psiC_minus_refC",
    "psiU_minus_refU.gap_2_to_M",
    "psiC_minus_refC.gap_2_to_M",
    "psiU_minus_random",
    "psiC_minus_random",
)
#: The S0-03 buckets B2 reads (§9.1).
BUCKETS = {k: S003.BUCKETS[k] for k in ("all", "gap_2_to_M", "gap_gt_M", "gap_eq_M")}
KINDS = ("assert", "query", "filler")

QUESTION = (
    "Can the spec's context-conditioned value head psi_hat(s_i, c_t) = s_i^T W c_t + "
    "u^T [s_i ; c_t], fitted with no age input to the MC return G, produce an eviction "
    "rule that beats the best age-based reference through the model? Q1: kind prior; "
    "Q2: c_t beyond kind."
)
FALSIFIER = (
    "B2 is kill-gate evidence (PLAN-v4 §0.2), not a spec falsifier; E0h is falsifier "
    "3c and has its own entry point and rc (A1.3)."
)
EXPECTED = (
    "PREREG §13 (not edited; its F2 citation is corrected by the B0 addendum): "
    "HARMFUL ~40%, MIXED ~40%, NOT LEARNABLE ~10%, NOT RULED OUT ~10%; psi-C closer "
    "to FIFO than psi-U; age decodability substantial; Q2-WIN unlikely; E0h within-"
    "step R^2 > 0.49 likely."
)


class RangeError(RuntimeError):
    """A document id outside the range a step is allowed to read (§3). Exit 3."""


class ControlFailure(RuntimeError):
    """A §11 control failed (checkpoint sha, M/d, T0, closure, identity probe, Σr,
    residency, determinism, logits). Exit 3, and no classification."""


class RidgeFailure(RuntimeError):
    """A1.6: the selected lambda's residual > 1e-8, or no eligible grid point.
    **Exit 1**: the measurement itself is defective (§11 control 10)."""

    exit_code = 1


# --------------------------------------------------------------------------- #
# §3 / A1.15: ranges, disjointness, aliasing, closure
# --------------------------------------------------------------------------- #


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) < min(a[1], b[1])


def e0d_prereg_ranges(path: Path = E0D_PREREG) -> dict[str, tuple[int, int]]:
    """§3's conditional row: every `[a, b)` E0d's PREREG declares, if that file
    exists on this branch. A1.15 adds E0d's range explicitly because on
    `run/b2-psi-probe` the file does not exist and the row would skip silently."""
    if not path.exists():
        return {}
    out = {}
    for k, (a, b) in enumerate(
        re.findall(r"\[(\d[\d_,]*),\s*(\d[\d_,]*)\)", path.read_text())
    ):
        out[f"e0d_prereg_{k}"] = (
            int(a.replace(",", "").replace("_", "")),
            int(b.replace(",", "").replace("_", "")),
        )
    return out


def range_problems(
    ranges: dict[str, tuple[int, int]] | None = None,
    used: dict[str, tuple[int, int]] | None = None,
) -> list[str]:
    """Every §3 disjointness failure, as text. Empty is the only pass."""
    ranges = dict(RANGES if ranges is None else ranges)
    used = {**(USED_RANGES if used is None else used), **e0d_prereg_ranges()}
    own = {**ranges, "E0D_RESERVED": E0D_RESERVED}
    probs = []
    for name, (lo, hi) in own.items():
        if not lo < hi:
            probs.append(f"{name} {[lo, hi]} is empty")
    names = list(own)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            if _overlap(own[names[a]], own[names[b]]):
                probs.append(
                    f"{names[a]} {list(own[names[a]])} overlaps "
                    f"{names[b]} {list(own[names[b]])}"
                )
    for name, rng in ranges.items():
        for uname, urng in used.items():
            if _overlap(rng, urng):
                probs.append(f"{name} {list(rng)} overlaps used {uname} {list(urng)}")
    return probs


def aliasing_problems(
    ranges: dict[str, tuple[int, int]] | None = None,
    used: dict[str, tuple[int, int]] | None = None,
    seeds: list[int] = SEEDS,
) -> list[str]:
    """§3: B2's generator seeds `s*K + id` are disjoint from every used range's,
    for every seed pair (s, s'). Holds by construction below id K; still asserted."""
    ranges = dict(RANGES if ranges is None else ranges)
    used = {**(USED_RANGES if used is None else used), **e0d_prereg_ranges()}
    others = {**used, **ranges, "E0D_RESERVED": E0D_RESERVED}
    probs = []
    for name, (lo, hi) in ranges.items():
        for s in seeds:
            mine = (s * GEN_SEED_K + lo, s * GEN_SEED_K + hi)
            for oname, (olo, ohi) in others.items():
                for s2 in seeds:
                    if (oname, s2) == (name, s):
                        continue
                    theirs = (s2 * GEN_SEED_K + olo, s2 * GEN_SEED_K + ohi)
                    if _overlap(mine, theirs):
                        probs.append(f"{name}@seed{s} aliases {oname}@seed{s2}")
    return probs


def require_range(doc_ids, name: str) -> None:
    """The guard every fit / selection / evaluation step calls on its inputs."""
    ids = [int(i) for i in doc_ids]
    lo, hi = RANGES[name]
    if not ids:
        raise RangeError(f"{name}: no documents")
    bad = [i for i in ids if not lo <= i < hi]
    if bad:
        raise RangeError(
            f"{len(bad)} document id(s) outside {name} [{lo}, {hi}), e.g. {bad[:5]}"
        )


def doc_by_id(seed: int, doc_id: int, S: int | None = None):
    """The seed's generator document `doc_id` (synthetic.py:237), made by id so a
    far range never generates the ids below it."""
    cfg = SyntheticConfig(sentences_per_document=S or S_STEPS, seed=seed)
    return _generate_document(doc_id, cfg)


def docs_in(seed: int, name: str, n: int | None = None) -> list:
    lo, hi = RANGES[name]
    n = (hi - lo) if n is None else n
    if n > hi - lo:
        raise RangeError(f"{name}: {n} documents asked, {hi - lo} reserved")
    return [doc_by_id(seed, i) for i in range(lo, lo + n)]


def vocabulary_closure(docs, vmap: dict[str, int]) -> dict:
    missing = sorted(set(build_vocab(list(docs))) - set(vmap))
    return {"ok": not missing, "missing": missing[:20], "n_missing": len(missing)}


# --------------------------------------------------------------------------- #
# §6: sizes
# --------------------------------------------------------------------------- #


def p_of(d: int) -> int:
    """§6: `p = d² + 2d`, computed, never typed."""
    return d * d + 2 * d


def u_rows(S: int) -> int:
    return S * (S - 1) // 2


def c_rows(S: int, m: int) -> int:
    return sum(min(t, m) for t in range(1, S))


def n_f(p: int, c_per_doc: int | None = None) -> int:
    """A1.15 (TBD-1): `N_F = ⌈10·p / 632⌉` so that `n_C ≥ 10·p`."""
    c_per_doc = c_rows(S_STEPS, M_STEPS) if c_per_doc is None else c_per_doc
    return math.ceil(10 * p / c_per_doc)


# --------------------------------------------------------------------------- #
# §4: the feature maps, stop-grad, ProbeArgminPolicy
# --------------------------------------------------------------------------- #


def _stopgrad(x: Tensor) -> Tensor:
    """The one place a captured `s_i` or `c_t` leaves the transformer (§1, §5):
    no graph, no shared storage."""
    return x.detach().clone()


def bilinear_features(s_i: Tensor, c_t: Tensor) -> Tensor:
    """§4: `[vec(s_i c_tᵀ); s_i; c_t]` in fp64, row-major vec. **Takes `(s_i, c_t)`
    and nothing else** -- no age, no rank, no kind, no position (§6 "Inputs")."""
    s = s_i.reshape(-1, s_i.shape[-1]).to(torch.float64)
    c = c_t.reshape(-1, c_t.shape[-1]).to(torch.float64)
    outer = (s.unsqueeze(2) * c.unsqueeze(1)).reshape(s.shape[0], -1)
    return torch.cat([outer, s, c], dim=1)


def age_features(age: Tensor, S: int) -> Tensor:
    """§6: one-hot of age `t - i ∈ {1, …, S-1}` (the age-only arm only)."""
    age = torch.as_tensor(age, dtype=torch.long).reshape(-1)
    if bool(((age < 1) | (age > S - 1)).any()):
        raise ValueError(f"age outside [1, {S - 1}]: {age.tolist()[:8]}")
    return torch.nn.functional.one_hot(age - 1, S - 1).to(torch.float64)


def _status_of(doc, i: int, t: int) -> str | None:
    """Descriptive label for an assert (pending / querying / answered); never an
    input (§4 "The per-eviction log")."""
    c, _k = LR.slot_class(doc, i, t)
    return None if c == "filler" else c


def eviction_record(slots, k: int, step: int, *, doc, seed, arm, psi=None) -> dict:
    """§4's per-eviction log. `psi` is the full live ψ̂ vector (slot order) or None."""
    w = int(slots.written_at[k])
    victim_rank, displacement = rank_shift(slots, k)
    margin = None
    if psi is not None and len(psi) > 1:
        srt = sorted(psi)
        margin = srt[1] - srt[0]
    return {
        "doc": doc.doc_id,
        "step": step,
        "seed": seed,
        "arm": arm,
        "victim_slot": k,
        "written_at": w,
        "age": step - w,
        "rank": victim_rank,
        "rank_shift": [victim_rank, displacement],
        "kind": doc.sentences[w].kind,
        "status": _status_of(doc, w, step) if doc.sentences[w].kind == "assert" else None,
        "margin": margin,
        "psi": psi,
        "live_ages": [
            step - int(slots.written_at[j])
            for j in torch.nonzero(slots.live).flatten().tolist()
        ],
    }


class ProbeArgminPolicy:
    """§4: evict `argmin_k f(s_k, c_t)·w` over live slots, ties to the lowest slot
    index (the oldest), as `OraclePolicy`. Its only parameters are `w` (fp64) and
    the feature map. No `observe` state, no RNG, no `b`, nu, shadow or warmup.

    ``feature="bilinear"``: ψ̂ reads `MemoryState.gestalts[k]` and `context`
    (= `out.srep[row].detach()`) only. ``feature="age"``: the age-only arm, which
    reads age from `MemoryState` because it is an age arm by definition (§7).
    """

    name = "probe_argmin"

    def __init__(
        self, w: Tensor, feature: str, *, doc, model_seed: int, arm: str, S: int
    ):
        if feature not in ("bilinear", "age"):
            raise ValueError(f"feature={feature!r}")
        self.w = _stopgrad(w).to(torch.float64)
        self.feature = feature
        self.doc = doc
        self.doc_id = doc.doc_id
        self.model_seed = model_seed
        self.arm = arm
        self.S = S
        self.log: list[dict] = []

    def scores(self, slots, context: Tensor, step: int) -> Tensor:
        """ψ̂ over the live slots, in slot order."""
        live = torch.nonzero(slots.live).flatten()
        if self.feature == "bilinear":
            g = _stopgrad(slots.gestalts[live])
            c = _stopgrad(context).reshape(1, -1).expand(len(live), -1)
            X = bilinear_features(g, c)
            return X @ self.w
        age = step - slots.written_at[live]
        return age_features(age, self.S) @ self.w

    def select_eviction(self, slots, context: Tensor, step: int) -> int:
        live = torch.nonzero(slots.live).flatten().tolist()
        psi = self.scores(slots, context, step)
        if not bool(torch.isfinite(psi).all()):
            raise FloatingPointError(f"doc {self.doc_id} step {step}: non-finite ψ̂")
        vals = psi.tolist()
        j = 0
        for q in range(1, len(vals)):
            if vals[q] < vals[j]:  # strict: ties to the lowest slot, the oldest
                j = q
        k = live[j]
        self.log.append(
            eviction_record(
                slots, k, step, doc=self.doc, seed=self.model_seed, arm=self.arm, psi=vals
            )
        )
        return k

    def observe(self, slots, attn, step: int) -> None:
        return None

    def on_write(self, slots, slot: int, step: int) -> None:
        return None

    def reset(self) -> None:
        return None


def no_age_check(pol: ProbeArgminPolicy, gestalts: Tensor, context: Tensor) -> bool:
    """§11 control 9, at run time: ψ̂ does not move when `written_at` and the step
    are permuted with the gestalts held fixed."""
    from rsr.retention.policy import MemoryState

    n = gestalts.shape[0]
    live = torch.ones(n, dtype=torch.bool)
    a = MemoryState(gestalts=gestalts, written_at=torch.arange(n), live=live, step=n)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)) + 7
    b = MemoryState(gestalts=gestalts, written_at=perm, live=live, step=n + 40)
    return torch.equal(pol.scores(a, context, n), pol.scores(b, context, n + 40))


class Logged:
    """Wraps any arm: the same per-eviction fields (§4 "The other arms"), in-loop
    residency at every query step (§9.1), and §11 control 5's Σr check."""

    def __init__(self, inner, *, doc, model_seed, arm: str, check_sum: bool = False):
        self.inner = inner
        self.doc = doc
        self.doc_id = doc.doc_id
        self.model_seed = model_seed
        self.arm = arm
        self.check_sum = check_sum
        self.assert_of = {q: a for a, q in doc.pairs}
        self.name = getattr(inner, "name", arm)
        self.reset_state()

    def reset_state(self) -> None:
        self.log: list[dict] = []
        self.victims: dict[int, int] = {}
        self.hits: list[tuple[int, bool]] = []
        self.sum_worst = 0.0

    def reset(self) -> None:
        self.reset_state()
        self.inner.reset()

    def observe(self, slots, attn, step: int) -> None:
        live = torch.nonzero(slots.live).flatten().tolist()
        written = [int(slots.written_at[j]) for j in live]
        if step in self.assert_of:
            a = self.assert_of[step]
            self.hits.append((step - a, a in written))
        n_live = len(live)
        if self.check_sum and n_live == len(slots.live) and n_live > 0:
            r = LR.r_of(attn, n_live, len(slots.live))
            self.sum_worst = max(self.sum_worst, abs(float(r.sum()) - 1.0))
        self.inner.observe(slots, attn, step)

    def select_eviction(self, slots, context, step: int) -> int:
        if not bool(slots.live.all()):
            raise ControlFailure(f"step {step}: asked to evict from a non-full memory")
        n_before = len(getattr(self.inner, "log", []))
        k = int(self.inner.select_eviction(slots, context, step))
        if not bool(slots.live[k]):
            raise ControlFailure(f"step {step}: {self.arm} returned a dead slot {k}")
        inner_log = getattr(self.inner, "log", None)
        if inner_log is not None and len(inner_log) == n_before + 1:
            rec = inner_log[-1]
        else:
            rec = eviction_record(
                slots, k, step, doc=self.doc, seed=self.model_seed, arm=self.arm
            )
        self.log.append(rec)
        self.victims[step] = int(slots.written_at[k])
        return k

    def on_write(self, slots, slot: int, step: int) -> None:
        self.inner.on_write(slots, slot, step)


Scripted = LR.Scripted


class KindOracle:
    """§6 as amended by A1.10: evict a live slot of the lowest class mean
    `m_gamma(kind(i))`; ties within that class are broken uniformly at random with
    `random.Random(f"ko:{seed}:{doc_id}:{t}")` (``tie="random"``, Q2's comparator),
    or to the oldest (``tie="oldest"``, the secondary)."""

    name = "kind_oracle"

    def __init__(
        self, means: dict[str, float], *, doc, model_seed: int, tie: str, S: int
    ):
        if tie not in ("random", "oldest"):
            raise ValueError(f"tie={tie!r}")
        self.means = dict(means)
        self.doc = doc
        self.doc_id = doc.doc_id
        self.model_seed = model_seed
        self.tie = tie
        self.kinds = [s.kind for s in doc.sentences]

    def select_eviction(self, slots, context, step: int) -> int:
        live = torch.nonzero(slots.live).flatten().tolist()
        vals = {k: self.means[self.kinds[int(slots.written_at[k])]] for k in live}
        low = min(vals.values())
        ks = [k for k in live if vals[k] == low]
        if self.tie == "oldest":
            return ks[0]
        rng = random.Random(f"ko:{self.model_seed}:{self.doc_id}:{step}")
        return rng.choice(ks)

    def observe(self, slots, attn, step: int) -> None:
        return None

    def on_write(self, slots, slot: int, step: int) -> None:
        return None

    def reset(self) -> None:
        return None


def random_seed_int(k: int, seed: int, doc_id: int) -> int:
    """§7 (A1 R5): the first 8 bytes of `sha256(f"b2rand:{k}:{seed}:{doc_id}")`,
    read big-endian (the PREREG does not name a byte order; declared)."""
    h = hashlib.sha256(f"b2rand:{k}:{seed}:{doc_id}".encode()).digest()[:8]
    return int.from_bytes(h, "big")


def random_policy(k: int, seed: int, doc_id: int) -> RandomPolicy:
    """The spec's E1 random arm, uniform over **all** live slots (not F-X1's)."""
    g = torch.Generator().manual_seed(random_seed_int(k, seed, doc_id))
    pol = RandomPolicy(g)
    pol.doc_id = doc_id
    pol.model_seed = seed
    return pol


def factfiller_policy(doc, seed: int):
    """§7: W6's rule, `random.Random(f"ff:{seed}:{doc_id}")` fresh per document."""
    pol = LR.FactFiller(doc, random.Random(f"ff:{seed}:{doc.doc_id}"))
    pol.doc_id = doc.doc_id
    pol.model_seed = seed
    return pol


class CheckedOracle(OraclePolicy):
    """§7 / §11 control 6: `OraclePolicy(discounted_demand(doc, gamma))`, asserting that
    the demand it was handed is this document's at this gamma."""

    def __init__(self, doc, gamma: float, demand) -> None:
        want = discounted_demand(doc, gamma)
        got = [[float(x) for x in row] for row in demand]
        if got != want:  # §11 control 6: exit 3 (review F5), and never stripped by -O
            raise ControlFailure(
                f"oracle demand is not document {doc.doc_id}'s at gamma={gamma}"
            )
        super().__init__(demand)
        self.doc_id = doc.doc_id
        self.gamma = gamma


def fifo_policy(doc):
    pol = FIFOPolicy()
    pol.doc_id = doc.doc_id
    return pol


# --------------------------------------------------------------------------- #
# §5: the capture -- a fresh FIFO rollout with rank-0..3 probes
# --------------------------------------------------------------------------- #


@dataclass
class DocCapture:
    doc_id: int
    gest: Tensor  # [S, d] -- s_i, the gestalt FIFO wrote for sentence i
    ctx: Tensor  # [S, d] -- c_t, step t's out.srep
    D: dict[int, Tensor]  # rank -> [S, S] step-major demand-if-resident
    resident: Tensor  # [S, S] FIFO held sentence i at step t
    kinds: tuple[str, ...]
    sum_worst: float
    identity_worst: dict[int, float]
    answers: list = field(default_factory=list)
    hits: list = field(default_factory=list)
    n_probe_rows: int = 0
    seconds: float = 0.0


class _CaptureRecorder(FIFOPolicy):
    """FIFO eviction, unchanged; records `r` per resident, every written gestalt,
    and at each full-memory step the probes at every rank in `ranks` (§5)."""

    def __init__(
        self, model, doc, m: int, S: int, hook, ranks, probes: bool = True
    ) -> None:
        self.model, self.doc, self.m, self.S, self.hook = model, doc, m, S, hook
        self.ranks = tuple(ranks)
        self.probes = probes
        self.assert_of = {q: a for a, q in doc.pairs}
        self.reset()

    def reset(self) -> None:
        nan = float("nan")
        self.D = {
            r: torch.full((self.S, self.S), nan, dtype=torch.float64) for r in self.ranks
        }
        self.resident = torch.zeros(self.S, self.S, dtype=torch.bool)
        self.gest: dict[int, Tensor] = {}
        self.sum_worst = 0.0
        self.identity_worst = {r: 0.0 for r in self.ranks}
        self.n_probe_rows = 0
        self.hits: list[tuple[int, bool]] = []

    def observe(self, slots, attn, step: int) -> None:
        live = torch.nonzero(slots.live).flatten().tolist()
        written = [int(slots.written_at[j]) for j in live]
        if step in self.assert_of:
            a = self.assert_of[step]
            self.hits.append((step - a, a in written))
        if not live:
            return None
        r = LR.r_of(attn, len(live), self.m)
        for j, w in zip(live, written, strict=True):
            for rk in self.ranks:
                self.D[rk][step, w] = float(r[j])
            self.resident[step, w] = True
        if len(live) == self.m:
            self.sum_worst = max(self.sum_worst, abs(float(r.sum()) - 1.0))
            if written != list(range(step - self.m, step)):
                raise ControlFailure(f"step {step}: FIFO memory is {written}")
            if self.probes:
                self._probe(step, r)
        return None

    def _probe(self, step: int, r_fifo: Tensor) -> None:
        ids_t, mask_t, kv, valid, bos_ctx, bos_valid = self.hook.args[:6]
        cand = []
        for rk in self.ranks:
            cand.append((rk, step - self.m + rk, True))  # the identity probe (C4)
            cand += [(rk, i, False) for i in range(step - self.m)]
        P = len(cand)
        kvP = kv.expand(P, -1, -1).clone()
        for p, (rk, i, _ident) in enumerate(cand):
            kvP[p, rk] = self.gest[i]
        rep = (ids_t, mask_t, valid, bos_ctx, bos_valid)
        ids_p, mask_p, valid_p, bos_p, bosv_p = (
            x.expand(P, *x.shape[1:]).clone() for x in rep
        )
        self.hook.armed = False
        try:
            out = self.model(ids_p, mask_p, kvP, valid_p, bos_p, bosv_p, capture=True)
        finally:
            self.hook.armed = True
        cap = cross_capture(self.model, out, kvP, mask_p, valid_p)
        for p, (rk, i, ident) in enumerate(cand):
            rp = LR.r_of(trace_for_row(cap, valid_p, p, step), self.m, self.m)
            self.sum_worst = max(self.sum_worst, abs(float(rp.sum()) - 1.0))
            if ident:
                d = abs(float(rp[rk]) - float(r_fifo[rk]))
                self.identity_worst[rk] = max(self.identity_worst[rk], d)
            else:
                self.D[rk][step, i] = float(rp[rk])
        self.n_probe_rows += P

    def on_write(self, slots, slot: int, step: int) -> None:
        live = torch.nonzero(slots.live).flatten().tolist()
        ks = [j for j in live if int(slots.written_at[j]) == step]
        self.gest[step] = _stopgrad(slots.gestalts[ks[0]])
        return None


def _answer_sink(tm, gap_of: dict[int, int], answers: list, ctx: dict | None = None):
    """S0-03's `answer_acc` per answer target (argmax over the full vocabulary),
    and step t's `c_t = out.srep[0]` when `ctx` is given."""
    from rsr.train.loop import lm_token_losses

    def step_fn(t, out, ids_t, mask_t, row_valid):
        if ctx is not None:
            ctx[t] = _stopgrad(out.srep[0])
        am = tm[:, t, 1:]
        if am.any():
            B = ids_t.shape[0]
            per = lm_token_losses(out.logits, ids_t).view(B, ids_t.shape[1] - 1)
            lg = out.logits[:, :-1][am]
            tgt = ids_t[:, 1:][am]
            ok = lg.argmax(-1) == tgt
            for o, n in zip(ok.tolist(), per[am].tolist(), strict=True):
                answers.append((t, gap_of[t], bool(o), float(n)))
        return torch.zeros(())

    return step_fn


def capture_doc(model, doc, *, vmap, S: int, m: int, L: int, ranks=RANKS) -> DocCapture:
    """§5: one document, alone, FIFO, eval mode, `no_grad`, with the probes."""
    t0 = time.time()
    ids, mask = encode([doc], vmap, max_tokens=L, steps=S)
    tm, _gap = answer_targets([doc], vmap, max_tokens=L, steps=S)
    gap_of = {q: q - a for a, q in doc.pairs}
    hook = LR._LastCall()
    h = model.register_forward_pre_hook(hook, with_kwargs=True)
    rec = _CaptureRecorder(model, doc, m, S, hook, ranks)
    answers: list = []
    ctx: dict[int, Tensor] = {}
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            run_policy_loop(
                model,
                ids,
                mask,
                torch.full((1,), S),
                rec,
                step_fn=_answer_sink(tm, gap_of, answers, ctx),
                observe=True,
            )
    finally:
        h.remove()
        model.train(was)
    if sorted(rec.gest) != list(range(S)) or sorted(ctx) != list(range(S)):
        raise ControlFailure(f"doc {doc.doc_id}: not every sentence was written")
    return DocCapture(
        doc_id=doc.doc_id,
        gest=torch.stack([rec.gest[i] for i in range(S)]),
        ctx=torch.stack([ctx[t] for t in range(S)]),
        D=rec.D,
        resident=rec.resident,
        kinds=tuple(s.kind for s in doc.sentences),
        sum_worst=rec.sum_worst,
        identity_worst=rec.identity_worst,
        answers=answers,
        hits=rec.hits,
        n_probe_rows=rec.n_probe_rows,
        seconds=time.time() - t0,
    )


@torch.no_grad()
def probe_by_hand(model, doc, cap: DocCapture, t: int, i: int, r: int, *, vmap, m, L):
    """One probe recomputed from scratch (a test oracle for §5): FIFO to step t,
    slot r's gestalt replaced by sentence i's, `r` read at slot r."""
    cfg = model.cfg
    ids, mask = encode([doc], vmap, max_tokens=L, steps=t + 1)
    mem = init_memory(1, cfg)
    bos_ctx = torch.zeros(1, cfg.D)
    bos_valid = torch.zeros(1, dtype=torch.bool)
    for s in range(t):
        out = model(ids[:, s], mask[:, s], mem.kv, mem.valid, bos_ctx, bos_valid)
        mem = write_at(mem, out.srep, out.has_eos, torch.zeros(1, dtype=torch.long), s)
        if cfg.bos_replacement_mode == "copy":
            bos_ctx, bos_valid = out.srep, out.has_eos
    kv = mem.kv.clone()
    kv[0, r] = cap.gest[i]
    out = model(ids[:, t], mask[:, t], kv, mem.valid, bos_ctx, bos_valid, capture=True)
    c = cross_capture(model, out, kv, mask[:, t], mem.valid)
    return float(LR.r_of(trace_for_row(c, mem.valid, 0, t), m, m)[r])


# --------------------------------------------------------------------------- #
# §5 / A1.2: targets
# --------------------------------------------------------------------------- #


def returns(X: Tensor, gamma: float) -> Tensor:
    """§5: `G_gamma[t][i] = Σ_{k=0}^{S-1-t} gamma^k X[t+k][i]`, truncated at document end
    (W10's `discounted_returns`)."""
    return LR.discounted_returns(X, gamma)


def shifted(G: Tensor) -> Tensor:
    """A1.2: `G⁺[t] = G[t+1]` (the sum from k = 1), 0 at `t = S-1`; NaN where
    `G[t]` is NaN (sentence not yet written)."""
    out = torch.full_like(G, float("nan"))
    nxt = torch.zeros_like(G)
    nxt[:-1] = torch.nan_to_num(G[1:], nan=0.0)
    return torch.where(torch.isnan(G), out, nxt)


def literal(D: Tensor, resident: Tensor) -> Tensor:
    """(C): the FIFO rollout's literal `r`, 0 once FIFO has evicted the sentence."""
    lit = torch.where(resident, torch.nan_to_num(D, nan=0.0), torch.zeros_like(D))
    return lit.masked_fill(torch.isnan(D), float("nan"))


ROWSET_OF = {"U": "U", "U1": "U", "U2": "U", "U3": "U", "U+": "U", "C": "C", "C+": "C"}


def target_matrix(cap: DocCapture, arm: str, gamma: float | None) -> Tensor:
    if arm == "age":
        S = cap.D[0].shape[0]
        t = torch.arange(S).unsqueeze(1)
        i = torch.arange(S).unsqueeze(0)
        return torch.where(
            i < t,
            (t - i).to(torch.float64),
            torch.tensor(float("nan"), dtype=torch.float64),
        )
    if arm in ("U", "U+"):
        G = returns(cap.D[0], gamma)
    elif arm in ("U1", "U2", "U3"):
        G = returns(cap.D[int(arm[1])], gamma)
    elif arm in ("C", "C+"):
        G = returns(literal(cap.D[0], cap.resident), gamma)
    else:
        raise ValueError(f"arm={arm!r}")
    return shifted(G) if arm.endswith("+") else G


def row_index(cap: DocCapture, rowset: str, m: int) -> tuple[Tensor, Tensor]:
    """§5 rows: (U) every t in [1, S) and every i < t; (C) FIFO-resident i < t."""
    S = cap.D[0].shape[0]
    ts, is_ = [], []
    for t in range(1, S):
        for i in range(t):
            if rowset == "U" or bool(cap.resident[t, i]):
                ts.append(t)
                is_.append(i)
    if rowset not in ("U", "C"):
        raise ValueError(rowset)
    return torch.tensor(ts), torch.tensor(is_)


def design(cap: DocCapture, t: Tensor, i: Tensor, feature: str) -> Tensor:
    if feature == "bilinear":
        return bilinear_features(cap.gest[i], cap.ctx[t])
    return age_features(t - i, cap.D[0].shape[0])


# --------------------------------------------------------------------------- #
# §6 / A1.6 / A1.7: the closed-form fp64 ridge
# --------------------------------------------------------------------------- #


class Gram:
    """`XᵀX` and `XᵀY` accumulated over rows, fp64."""

    def __init__(self, p: int, k: int) -> None:
        self.xtx = torch.zeros(p, p, dtype=torch.float64)
        self.xty = torch.zeros(p, k, dtype=torch.float64)
        self.n = 0

    def add(self, X: Tensor, Y: Tensor) -> None:
        if X.requires_grad or Y.requires_grad:
            raise ValueError("a ridge input carries graph: stop-grad violated (§1)")
        X = X.to(torch.float64)
        Y = Y.to(torch.float64).reshape(X.shape[0], -1)
        if not (bool(torch.isfinite(X).all()) and bool(torch.isfinite(Y).all())):
            raise FloatingPointError("non-finite feature or target")
        self.xtx.addmm_(X.T, X)
        self.xty.addmm_(X.T, Y)
        self.n += X.shape[0]


def spectral(xtx: Tensor) -> tuple[Tensor, Tensor, float]:
    """A1.6: one eigendecomposition per Gram, fp64."""
    e, V = torch.linalg.eigh(xtx)
    return e, V, float(torch.trace(xtx))


def solve_path(xtx: Tensor, xty: Tensor, lambdas=LAMBDAS, spec=None) -> list[dict]:
    """Every grid λ solved spectrally: `λ' = λ·tr(XᵀX)/p`, `w = V (Vᵀb)/(e+λ')`,
    with §6's relative residual `‖(XᵀX + λ'I)w - Xᵀy‖ / ‖Xᵀy‖`."""
    e, V, tr = spectral(xtx) if spec is None else spec
    p = xtx.shape[0]
    b = xty.reshape(-1).to(torch.float64)
    Vb = V.T @ b
    bn = float(b.norm())
    out = []
    for lam in lambdas:
        lam_eff = lam * tr / p
        w = V @ (Vb / (e + lam_eff))
        res = xtx @ w + lam_eff * w - b
        resid = float(res.norm()) / bn if bn > 0 else float(res.norm())
        out.append(
            {
                "lam": lam,
                "lam_eff": lam_eff,
                "w": w,
                "resid": resid,
                "eligible": resid <= ELIGIBLE_RESID,
            }
        )
    return out


def select_lambda(path: list[dict], val_mse: list[float]) -> int:
    """A1.7: the minimum validation MSE among eligible grid points, ties to the
    larger λ; A1.6: the selected point's residual must be ≤ 1e-8, else exit 1."""
    elig = [k for k, pt in enumerate(path) if pt["eligible"]]
    if not elig:
        raise RidgeFailure("no eligible grid point (every residual > 1e-06)")
    best = None
    for k in elig:
        if (
            best is None
            or val_mse[k] < val_mse[best]
            or (val_mse[k] == val_mse[best] and path[k]["lam"] > path[best]["lam"])
        ):
            best = k
    if path[best]["resid"] > SELECTED_RESID:
        raise RidgeFailure(
            f"selected λ={path[best]['lam']:g}: residual {path[best]['resid']:.3g} "
            f"> {SELECTED_RESID:g}"
        )
    return best


def within_step_demean(v: Tensor, groups: Tensor) -> Tensor:
    """Subtract each (document, t) group's mean; `v` is [n] or [n, k]."""
    uniq, inv = torch.unique(groups, return_inverse=True)
    v2 = v.reshape(v.shape[0], -1).to(torch.float64)
    s = torch.zeros(len(uniq), v2.shape[1], dtype=torch.float64).index_add_(0, inv, v2)
    c = torch.zeros(len(uniq), dtype=torch.float64).index_add_(
        0, inv, torch.ones(len(inv), dtype=torch.float64)
    )
    out = v2 - (s / c.unsqueeze(1))[inv]
    return out.reshape(v.shape)


def _stack_targets(cap, specs, t, i) -> Tensor:
    return torch.stack([target_matrix(cap, a, g)[t, i] for a, g in specs], dim=1)


def fit_heads(
    train_caps: list[DocCapture],
    val_caps: list[DocCapture],
    rowset: str,
    feature: str,
    specs: list[tuple[str, float | None]],
    *,
    m: int,
    select_on: str = "demeaned",
) -> dict:
    """§6 fits sharing one design matrix: every `(arm, gamma)` in `specs` has the same
    rows (`rowset`), so one Gram and one eigendecomposition serve them all.

    λ is chosen on FIT_VAL (A1.7: within-step demeaned MSE over full-memory rows,
    raw MSE reported; ``select_on="raw"`` over all rows for the age-decodability
    predictor, A1.7's author's note). The final `w` is FIT_TRAIN's alone."""
    require_range([c.doc_id for c in train_caps], "FIT_TRAIN")
    require_range([c.doc_id for c in val_caps], "FIT_VAL")
    for a, _g in specs:
        if a != "age" and ROWSET_OF[a] != rowset:
            raise ValueError(f"arm {a} does not use the {rowset} rows")
    S, d = train_caps[0].gest.shape
    p = p_of(d) if feature == "bilinear" else S - 1
    G = Gram(p, len(specs))
    for cap in train_caps:
        t, i = row_index(cap, rowset, m)
        G.add(design(cap, t, i, feature), _stack_targets(cap, specs, t, i))
    spec = spectral(G.xtx)
    paths = [solve_path(G.xtx, G.xty[:, j], LAMBDAS, spec) for j in range(len(specs))]
    Ws = [torch.stack([pt["w"] for pt in path], dim=1) for path in paths]  # [p, nλ]
    k = len(LAMBDAS)
    sse_dm = [torch.zeros(k, dtype=torch.float64) for _ in specs]
    sse_raw = [torch.zeros(k, dtype=torch.float64) for _ in specs]
    ysum = [0.0 for _ in specs]
    yy = [0.0 for _ in specs]
    n_val = 0
    for cap in val_caps:
        t, i = row_index(cap, rowset, m)
        if select_on == "demeaned":
            keep = t >= m  # full-memory rows: the argmin's domain
            t, i = t[keep], i[keep]
        X = design(cap, t, i, feature)
        Y = _stack_targets(cap, specs, t, i)
        n_val += len(t)
        for j in range(len(specs)):
            pred = X @ Ws[j]
            y = Y[:, j]
            sse_raw[j] += ((pred - y.unsqueeze(1)) ** 2).sum(0)
            ysum[j] += float(y.sum())
            yy[j] += float((y * y).sum())
            if select_on == "demeaned":
                dp = within_step_demean(pred, t)
                dy = within_step_demean(y, t)
                sse_dm[j] += ((dp - dy.unsqueeze(1)) ** 2).sum(0)
    out = {}
    for j, key in enumerate(specs):
        raw = (sse_raw[j] / n_val).tolist()
        dm = (sse_dm[j] / n_val).tolist() if select_on == "demeaned" else None
        sel = select_lambda(paths[j], dm if dm is not None else raw)
        sst = yy[j] - ysum[j] ** 2 / n_val
        out[key] = {
            "w": paths[j][sel]["w"],
            "lam": paths[j][sel]["lam"],
            "selected": sel,
            "path": [{kk: v for kk, v in pt.items() if kk != "w"} for pt in paths[j]],
            "val_mse_demeaned": dm,
            "val_mse_raw": raw,
            # §6 / §8.1: sse_raw is already a sum over the n_val rows, so R² is
            # 1 - SSE/SST. Phase A (280a2ad) logged 1 - n_val*SSE/SST; its stored
            # values are corrected by erratum sidecar, never by a refit (P0.2).
            "val_r2_raw": [1.0 - float(x) / sst if sst > 0 else None for x in sse_raw[j]],
            "val_r2_form": R2_FORM,
            "n_train_rows": G.n,
            "n_val_rows": n_val,
            "train_range": "FIT_TRAIN",
            "val_range": "FIT_VAL",
            "rowset": rowset,
            "feature": feature,
        }
    return out


def age_decodability(train_caps, val_caps, rowset: str, *, m: int) -> dict:
    """§6 / §8.1: the bilinear form fitted to predict age; λ by **raw** validation
    MSE (A1.7 author's note), R² of raw age on FIT_VAL."""
    f = fit_heads(
        train_caps, val_caps, rowset, "bilinear", [("age", None)], m=m, select_on="raw"
    )[("age", None)]
    return {
        "r2": f["val_r2_raw"][f["selected"]],
        "r2_split": r2_split_by_index(val_caps, rowset, f["w"], m=m),
        "lam": f["lam"],
        "resid": f["path"][f["selected"]]["resid"],
        "lambda_rule": "raw_val_mse",
        "val_range": f["val_range"],
        "n_val_rows": f["n_val_rows"],
        "w": f["w"],
    }


def r2_split_by_index(val_caps, rowset: str, w: Tensor, *, m: int) -> dict:
    """Build review F11 (descriptive, no rule): the age predictor's FIT_VAL R² of raw
    age, split into rows with sentence index i < M (their srep was computed over an
    under-full memory, so s_i may carry the fill level) and i ≥ M. Each split's R²
    uses its own mean. Same fitted `w` (selected λ); nothing is refitted."""
    require_range([c.doc_id for c in val_caps], "FIT_VAL")
    acc = {k: [0, 0.0, 0.0, 0.0] for k in ("i_lt_M", "i_ge_M")}  # n, sse, Σy, Σy²
    for cap in val_caps:
        t, i = row_index(cap, rowset, m)
        pred = design(cap, t, i, "bilinear") @ w.to(torch.float64)
        y = (t - i).to(torch.float64)
        for k, sel in (("i_lt_M", i < m), ("i_ge_M", i >= m)):
            ys, ps = y[sel], pred[sel]
            a = acc[k]
            a[0] += len(ys)
            a[1] += float(((ps - ys) ** 2).sum())
            a[2] += float(ys.sum())
            a[3] += float((ys * ys).sum())
    out = {}
    for k, (n, sse, sy, syy) in acc.items():
        sst = syy - sy * sy / n if n else 0.0
        out[k] = {"r2": 1.0 - sse / sst if sst > 0 else None, "n": n}
    return out


def class_means(
    train_caps: list[DocCapture], gamma: float, *, m: int
) -> dict[str, float]:
    """§6 kind-oracle: `m_gamma(kind)` = mean `G_U,gamma[i][t]` over FIT_TRAIN U rows at
    full-memory steps, by the generator's kind of sentence i. Never from B0."""
    require_range([c.doc_id for c in train_caps], "FIT_TRAIN")
    acc: dict[str, list[float]] = {}
    for cap in train_caps:
        G = target_matrix(cap, "U", gamma)
        t, i = row_index(cap, "U", m)
        keep = t >= m
        for tt, ii in zip(t[keep].tolist(), i[keep].tolist(), strict=True):
            acc.setdefault(cap.kinds[ii], []).append(float(G[tt, ii]))
    return {k: sum(v) / len(v) for k, v in acc.items()}


# --------------------------------------------------------------------------- #
# §7: one arm, one document, through the model
# --------------------------------------------------------------------------- #


def run_arm(model, doc, policy, *, vmap, S: int, m: int, L: int, check_sum=True) -> dict:
    """B = 1, a fresh policy per document, through `run_policy_loop` (§2). The
    wrapper asserts the document it was built for is the one being run (§11.6), and
    in-loop residency must equal the model-free replay of its victims (§11.7)."""
    if getattr(policy, "doc_id", None) != doc.doc_id:
        raise ControlFailure(
            f"policy built for document {getattr(policy, 'doc_id', None)} run on "
            f"document {doc.doc_id}"
        )
    inst = (
        policy
        if isinstance(policy, Logged)
        else Logged(
            policy,
            doc=doc,
            model_seed=getattr(policy, "model_seed", None),
            arm=getattr(policy, "arm", getattr(policy, "name", "?")),
            check_sum=check_sum,
        )
    )
    ids, mask = encode([doc], vmap, max_tokens=L, steps=S)
    tm, _gap = answer_targets([doc], vmap, max_tokens=L, steps=S)
    gap_of = {q: q - a for a, q in doc.pairs}
    answers: list = []
    model.eval()
    with torch.no_grad():
        run_policy_loop(
            model,
            ids,
            mask,
            torch.full((1,), S),
            inst,
            step_fn=_answer_sink(tm, gap_of, answers),
            observe=True,
        )
    rep = simulate(doc, Scripted(inst.victims), m)["queries"]
    replay = [(q["gap"], q["hit"]) for q in rep]
    return {
        "doc": doc.doc_id,
        "answers": answers,
        "victims": inst.victims,
        "hits": inst.hits,
        "residency_ok": replay == inst.hits,
        "sum_worst": inst.sum_worst,
        "log": inst.log,
    }


def counts(answers, bucket: str) -> tuple[int, int]:
    sel = BUCKETS[bucket]
    rows = [ok for _t, g, ok, _n in answers if bool(sel(torch.tensor(g)))]
    return len(rows), sum(rows)


# --------------------------------------------------------------------------- #
# §9: bootstrap, ref, δ, power, ceiling
# --------------------------------------------------------------------------- #

_RANDOM_ARM = re.compile(r"random\d+")


def _rand_combine(xs: list[Tensor]) -> Tensor:
    """§9.2: the random arm is the mean over its 5 seeds of their pooled accuracies."""
    return torch.stack(xs).mean(0)


def paired_bootstrap(
    per_doc: dict[str, Tensor],
    contrasts: dict[str, tuple[str, str]],
    seed: int,
    n_boot: int = N_BOOT,
) -> dict[str, dict]:
    """§9.2: `per_doc[arm]` is `[D, 2]` (n answers, n correct) in ONE document order;
    one resampled multiset per replicate shared by every arm (paired); pooled
    accuracy per replicate; percentile 95% CI; `torch.Generator(20260927 + seed)`.
    The pseudo-arm ``"random"`` is `_rand_combine` over ``random0..random4``."""
    arms = list(per_doc)
    D = per_doc[arms[0]].shape[0]
    X = {a: per_doc[a].to(torch.float64).reshape(D, 2) for a in arms}
    g = torch.Generator().manual_seed(BOOT_SEED_BASE + seed)
    idx = torch.randint(0, D, (n_boot, D), generator=g)
    W = torch.zeros(n_boot, D, dtype=torch.float64).scatter_add_(
        1, idx, torch.ones(n_boot, D, dtype=torch.float64)
    )
    rep = {a: (W @ x[:, 1]) / (W @ x[:, 0]) for a, x in X.items()}
    pt = {a: x[:, 1].sum() / x[:, 0].sum() for a, x in X.items()}
    rnd = sorted(a for a in arms if _RANDOM_ARM.fullmatch(a))
    if rnd:
        rep["random"] = _rand_combine([rep[a] for a in rnd])
        pt["random"] = _rand_combine([pt[a] for a in rnd])
    out = {}
    for name, (a, b) in contrasts.items():
        r = rep[a] - rep[b]
        out[name] = {
            "point": float(pt[a] - pt[b]),
            "lo": float(torch.quantile(r, 0.025)),
            "hi": float(torch.quantile(r, 0.975)),
            "sd": float(r.std()),
        }
    return out


def select_ref(acc: dict[str, float], doc_ids) -> str:
    """§9.3: whichever of FIFO and that arm's age-only head has the higher all-query
    accuracy **on FIT_VAL**; a tie goes to FIFO. (A1.15: the review's R3 rule is
    not adopted.)"""
    require_range(doc_ids, "FIT_VAL")  # §9.3: ref on FIT_VAL, never EVAL
    return "age" if acc["age"] > acc["fifo"] else "fifo"


def delta_of(acc_oracle: float, acc_fifo: float) -> float:
    """§9.3: `δ = 0.25 x (acc_oracle(all) - acc_FIFO(all))` on FIT_VAL."""
    return DELTA_FRACTION * (acc_oracle - acc_fifo)


def deltas_by_gamma(d: dict[float, float]) -> dict[float, float]:
    """A1.11: at gamma = 0, δ is the same seed's gamma = 0.9 δ."""
    return {0.9: d[0.9], 0.0: d[0.9]}


def required_n(sigma: float, delta: float | None, n_v: int) -> float:
    """§9.4 item 3: `⌈N_V · (1.96 sigma / (δ/2))²⌉`; an undefined δ (≤ 0) needs
    infinitely many documents, so the cap binds (declared)."""
    if delta is None or not delta > 0:
        return math.inf
    return math.ceil(n_v * (1.96 * sigma / (delta / 2)) ** 2)


def n_eval(
    sigmas: dict[tuple[str, int], float], deltas: dict[int, float], n_v: int = N_V
) -> dict:
    """§9.4 items 4-5 with A1.12: the max over the six gating contrasts and the
    seeds, clipped to [1024, 40000]; UNDERPOWERED if the cap binds."""
    per = {
        f"{c}@seed{s}": required_n(sig, deltas[s], n_v)
        for (c, s), sig in sigmas.items()
        if c in GATING_CONTRASTS
    }
    raw = max(per.values())
    n = N_E_MAX if raw > N_E_MAX else max(N_E_MIN, int(raw))
    return {"N_E": n, "underpowered": raw > N_E_MAX, "raw_max": raw, "per": per}


def compute_ceiling(
    n_e: int,
    *,
    sec_per_arm_doc: float,
    arms: int,
    seeds: int,
    parallel: int,
    budget_h: float = TIER1_BUDGET_H,
) -> dict:
    """A1.4 item 7: if the projected wall time exceeds the budget, N_E is reduced to
    the largest value that fits and the run is labelled UNDERPOWERED."""

    def hours(n):
        return n * arms * seeds * sec_per_arm_doc / parallel / 3600.0

    if hours(n_e) <= budget_h:
        return {"N_E": n_e, "reduced": False, "projected_h": hours(n_e)}
    n = math.floor(budget_h * 3600.0 * parallel / (sec_per_arm_doc * arms * seeds))
    return {"N_E": n, "reduced": True, "projected_h": hours(n)}


def outcome_flags(point: float, lo: float, hi: float) -> list[str]:
    """A1.13: a CI that excludes its own estimate is flagged and UNRESOLVED."""
    return [] if lo <= point <= hi else ["CI_EXCLUDES_ESTIMATE"]


def outcome(point, lo, hi, delta, *, noninf_lo, rand_lo) -> str:
    """§9.5 as amended by A1.13: order WIN, LOSS, EQUIV, UNRESOLVED."""
    if delta is None or not delta > 0:
        return "UNRESOLVED"  # δ undefined
    if outcome_flags(point, lo, hi):
        return "UNRESOLVED"
    if lo > 0 and point >= delta and noninf_lo > -delta and rand_lo > 0:
        return "WIN"
    if hi < -delta:
        return "LOSS"
    if -delta < lo and hi < delta:
        return "EQUIV"
    return "UNRESOLVED"


CLASSES = {
    1: "HARMFUL",
    2: "NOT RULED OUT, CENSORED",
    3: "NOT RULED OUT, UNCENSORED ONLY",
    4: "ANOMALY (C > U)",
    5: "NOT LEARNABLE, offline",
    6: "MIXED / UNRESOLVED",
}


def classify(u: list[str], c: list[str]) -> tuple[int, str]:
    """§9.6: the 6-row precedence truth table; the first matching row wins."""
    if "LOSS" in u or "LOSS" in c:
        row = 1
    elif u.count("WIN") == 3 and c.count("WIN") == 3:
        row = 2
    elif u.count("WIN") == 3:
        row = 3
    elif c.count("WIN") == 3:
        row = 4
    elif u.count("EQUIV") == 3 and c.count("EQUIV") == 3:
        row = 5
    else:
        row = 6
    return row, CLASSES[row]


def q2_outcome(point, lo, hi, delta) -> str:
    """§9.7 (A1.13 applies): Q2-WIN / Q2-EQUIV / Q2-LOSS / Q2-UNRESOLVED."""
    if delta is None or not delta > 0 or outcome_flags(point, lo, hi):
        return "Q2-UNRESOLVED"
    if lo > 0 and point >= delta:
        return "Q2-WIN"
    if hi < -delta:
        return "Q2-LOSS"
    if -delta < lo and hi < delta:
        return "Q2-EQUIV"
    return "Q2-UNRESOLVED"


# --------------------------------------------------------------------------- #
# §10 / A1.3 / A1.5 / A1.8: E0h
# --------------------------------------------------------------------------- #


def recompute_logits(module, x: Tensor, mem_kv: Tensor, mem_valid: Tensor) -> Tensor:
    """A1.5: the pre-softmax cross-attention logit `query(x)·scale · key(mem + PE)`,
    `[B, H, Q, M]`, by the same ops as `CrossAttention.forward`, from the inputs a
    forward pre-hook captured. Nothing in the forward pass changes."""
    from rsr.model.tg.model import memory_positions, sinusoidal_key_pe

    cfg = module.cfg
    keys_src = mem_kv
    if cfg.stm_cross_pos_mode == "sinusoidal":
        keys_src = keys_src + sinusoidal_key_pe(
            memory_positions(mem_valid),
            cfg.D,
            cfg.stm_positional_weight,
            dtype=mem_kv.dtype,
        )
    q = module.query(x) * cfg.attention_logit_scale
    k = module.key(keys_src)
    return torch.einsum("bqhd,bkhd->bhqk", q, k)


class LogitHook:
    """A `register_forward_pre_hook` on every C block's `cross_attn` (A1.5)."""

    def __init__(self, model) -> None:
        self.calls: list[tuple] = []
        self._pending: list[Tensor] = []
        self._valid: Tensor | None = None
        self.handles = [
            b.cross_attn.register_forward_pre_hook(self._hook)
            for b in model.blocks
            if b.block_type == "C"
        ]

    def _hook(self, module, args):
        x, mem_kv, mem_valid = args[:3]
        with torch.no_grad():
            lg = recompute_logits(module, x, mem_kv, mem_valid)
        self._pending.append(lg)
        self._valid = mem_valid
        self.calls.append(tuple(lg.shape))
        return None

    def take(self) -> dict:
        out = {"logits": torch.stack(self._pending), "valid": self._valid}
        self._pending = []
        return out

    def wrap(self, step_fn, sink: list | None = None):
        """A step_fn that also hands each step's logits to `sink` (if any)."""

        def f(t, out, ids_t, mask_t, row_valid):
            lg = self.take()
            if sink is not None:
                sink.append((t, lg, out.cross_attention, mask_t))
            return step_fn(t, out, ids_t, mask_t, row_valid)

        return f

    def remove(self) -> None:
        for h in self.handles:
            h.remove()


def logit_control(taken: dict, cross_attention, mask_t: Tensor) -> float:
    """A1.5 control: per query token, before collapse, the softmax of the
    recomputed logits over allowed slots reproduces `out.cross_attention`."""
    worst = 0.0
    lg, valid = taken["logits"], taken["valid"]
    real = mask_t != 0  # [B, Q]
    for layer, att in zip(lg, cross_attention, strict=True):
        allowed = valid.view(valid.shape[0], 1, 1, -1).expand_as(layer)
        z = torch.where(
            allowed,
            layer.to(torch.float32),
            torch.full_like(layer, torch.finfo(torch.float32).min, dtype=torch.float32),
        )
        sm = torch.softmax(z, dim=-1).to(att.dtype)
        diff = (sm - att).abs()  # [B, H, Q, M]
        diff = diff.masked_fill(~real.view(real.shape[0], 1, -1, 1), 0.0)
        worst = max(worst, float(diff.max()))
    return worst


def collapse_mean(logits: Tensor, mask: Tensor) -> Tensor:
    """A1.5: the mean over real query tokens, `[B, H, Q, M] -> [B, H, M]`."""
    real = (mask != 0).to(logits.dtype)
    s = torch.einsum("bhqm,bq->bhm", logits, real)
    return s / real.sum(-1).view(-1, 1, 1)


def within_step_r2(y: Tensor, X: Tensor, groups: Tensor) -> float | None:
    """§10 primary: ψ̂ and every regressor demeaned within each (document, t)
    group, OLS without intercept, fp64. None when ψ̂'s within-step variance is 0."""
    yd = within_step_demean(y.to(torch.float64), groups)
    Xd = within_step_demean(X.to(torch.float64), groups)
    sst = float((yd * yd).sum())
    if sst == 0.0:
        return None
    b = torch.linalg.lstsq(Xd, yd.unsqueeze(1)).solution
    sse = float(((yd - (Xd @ b).squeeze(1)) ** 2).sum())
    return 1.0 - sse / sst


def pooled_r2(y: Tensor, X: Tensor) -> float | None:
    """§10 secondary: raw values, with an intercept."""
    y = y.to(torch.float64)
    Xi = torch.cat([X.to(torch.float64), torch.ones(len(y), 1, dtype=torch.float64)], 1)
    b = torch.linalg.lstsq(Xi, y.unsqueeze(1)).solution
    sst = float(((y - y.mean()) ** 2).sum())
    if sst == 0.0:
        return None
    return 1.0 - float(((y - (Xi @ b).squeeze(1)) ** 2).sum()) / sst


def e0h_classify(r2: dict[int, float | None]) -> str:
    """§10 as amended by A1.8. ``None`` is an UNINFORMATIVE seed, counted toward
    neither COLLINEAR nor NOT_COLLINEAR."""
    inf = {s: v for s, v in r2.items() if v is not None}
    if sum(v >= E0H_COLLINEAR for v in inf.values()) >= 2:
        return "COLLINEAR"
    if len(inf) >= 2 and all(v < E0H_NOT_COLLINEAR for v in inf.values()):
        return "NOT_COLLINEAR"
    return "INTERMEDIATE"


#: A1.3 / build review F10: the ruling that ratifies (or replaces) the E0h
#: thresholds, named here by an explicit edit once Brendan has written it (a path
#: relative to the repo root). A filename glob would let a ruling that REJECTS the
#: thresholds enable rc 1 with the old numbers. None: unratified, E0h exits 2.
E0H_RULING: str | None = None


def e0h_ratified(root: Path = ROOT) -> bool:
    """A1.3: E0H_RULING names a ruling file that exists. None does at this build;
    until one does, E0h exits 2 whatever the R² values are."""
    return E0H_RULING is not None and (root / E0H_RULING).is_file()


def e0h_rc(cls: str, *, ratified: bool) -> int:
    """A1.3 §10.1: 0 NOT_COLLINEAR; 1 COLLINEAR (only under a ratifying ruling);
    2 INTERMEDIATE, UNINFORMATIVE or unratified; 3 DID NOT RUN."""
    if cls == "DID_NOT_RUN":
        return 3
    if not ratified:
        return 2
    return {"COLLINEAR": 1, "NOT_COLLINEAR": 0}.get(cls, 2)


class R2Acc:
    """§10's two R² from streamed sufficient statistics, so E0h's rows (every
    FIFO-resident slot at every full-memory EVAL step) never have to be stored.

    Within-step: each (document, t) group is demeaned as it arrives (a group never
    spans two calls), then `XdᵀXd`, `Xdᵀyd`, `ydᵀyd` accumulate; OLS without
    intercept. Pooled: `[X 1]` with an intercept."""

    def __init__(self, k: int) -> None:
        self.k = k
        self.xx = torch.zeros(k, k, dtype=torch.float64)
        self.xy = torch.zeros(k, dtype=torch.float64)
        self.yy = 0.0
        self.px = torch.zeros(k + 1, k + 1, dtype=torch.float64)
        self.py = torch.zeros(k + 1, dtype=torch.float64)
        self.pyy = 0.0
        self.ys = 0.0
        self.n = 0

    def add(self, y: Tensor, X: Tensor, groups: Tensor) -> None:
        y = y.to(torch.float64)
        X = X.to(torch.float64)
        yd, Xd = within_step_demean(y, groups), within_step_demean(X, groups)
        self.xx += Xd.T @ Xd
        self.xy += Xd.T @ yd
        self.yy += float(yd @ yd)
        Xi = torch.cat([X, torch.ones(len(y), 1, dtype=torch.float64)], 1)
        self.px += Xi.T @ Xi
        self.py += Xi.T @ y
        self.pyy += float(y @ y)
        self.ys += float(y.sum())
        self.n += len(y)

    def within(self) -> float | None:
        if self.yy == 0.0:
            return None  # ψ̂'s within-step variance is 0: UNINFORMATIVE
        b = torch.linalg.lstsq(self.xx, self.xy.unsqueeze(1)).solution.squeeze(1)
        return 1.0 - (self.yy - 2 * float(b @ self.xy) + float(b @ self.xx @ b)) / self.yy

    def pooled(self) -> float | None:
        sst = self.pyy - self.ys**2 / self.n
        if sst == 0.0:
            return None
        b = torch.linalg.lstsq(self.px, self.py.unsqueeze(1)).solution.squeeze(1)
        sse = self.pyy - 2 * float(b @ self.py) + float(b @ self.px @ b)
        return 1.0 - sse / sst

    def state(self) -> dict:
        return {
            k: getattr(self, k)
            for k in ("k", "xx", "xy", "yy", "px", "py", "pyy", "ys", "n")
        }

    @classmethod
    def from_state(cls, st: dict) -> R2Acc:
        a = cls(st["k"])
        for k, v in st.items():
            setattr(a, k, v)
        return a

    def merge(self, other: R2Acc) -> R2Acc:
        """Add another accumulator's statistics (documents are disjoint groups).
        Folding per-document accumulators in document order performs the same
        floating-point additions, in the same order, as one accumulator fed those
        documents one `add` each (build review F1: EVAL is streamed per document)."""
        if self.k == 0:
            self.k = other.k
            self.xx = torch.zeros_like(other.xx)
            self.xy = torch.zeros_like(other.xy)
            self.px = torch.zeros_like(other.px)
            self.py = torch.zeros_like(other.py)
        self.xx += other.xx
        self.xy += other.xy
        self.yy += other.yy
        self.px += other.px
        self.py += other.py
        self.pyy += other.pyy
        self.ys += other.ys
        self.n += other.n
        return self


E0H_HEADS = ("U@0.0", "U+@0.0", "C@0.0")  # headline, A1.2 companion, reported
E0H_TARGET = "target_D"  # the reference: the gamma = 0 target D[i][t] itself


E0H_SOURCES = ("prehook_logit", "log_alpha")  # A1.5: `e0h.regressor_source`


def e0h_source_for(model) -> str:
    """A1.5: "prehook_logit" when the forward pre-hook can be built on this model
    (at least one C block's `cross_attn` takes it), else "log_alpha"."""
    try:
        hook = LogitHook(model)
    except Exception:
        return "log_alpha"
    n = len(hook.handles)
    hook.remove()
    return "prehook_logit" if n else "log_alpha"


def _alpha_wrap(step_fn, sink: list):
    """The log-alpha path's step_fn: hands each step's captured alpha to `sink`."""

    def f(t, out, ids_t, mask_t, row_valid):
        sink.append((t, None, out.cross_attention, mask_t))
        return step_fn(t, out, ids_t, mask_t, row_valid)

    return f


def fifo_with_e0h(
    model, doc, w_heads: dict[str, Tensor], *, vmap, S, m, L, source="prehook_logit"
) -> dict:
    """The FIFO arm's rollout of one EVAL document, instrumented for E0h (A1.4:
    E0h needs no arm-run of its own). Returns `run_arm`'s fields plus, per
    full-memory step, the E0h rows: the 6·H mean-collapsed pre-hook logits
    (A1.5) of every FIFO-resident slot, ψ̂ of each head in `w_heads` on
    `(s_i, c_t)`, and FIFO's own `r_i(t)` (the gamma = 0 target).

    ``source="log_alpha"`` (A1.5, only when the hook cannot be built): the
    regressors are the per-token `log alpha` collapsed by the same mean. Per query
    token `log alpha = logit - logsumexp`, and the second term is constant over
    slots, so within-step demeaning makes the primary exact; the pooled
    secondary is DID NOT RUN (`e0h_compute`). There is no logit control on this
    path (`logit_control_worst` is None)."""
    if source not in E0H_SOURCES:
        raise ValueError(f"source={source!r}")
    ids, mask = encode([doc], vmap, max_tokens=L, steps=S)
    tm, _ = answer_targets([doc], vmap, max_tokens=L, steps=S)
    gap_of = {q: q - a for a, q in doc.pairs}
    rec = _CaptureRecorder(model, doc, m, S, None, ranks=(0,), probes=False)
    inst = Logged(rec, doc=doc, model_seed=None, arm="fifo", check_sum=True)
    answers: list = []
    ctx: dict[int, Tensor] = {}
    steps: list = []
    sink = _answer_sink(tm, gap_of, answers, ctx)
    hook = LogitHook(model) if source == "prehook_logit" else None
    model.eval()
    try:
        with torch.no_grad():
            run_policy_loop(
                model,
                ids,
                mask,
                torch.full((1,), S),
                inst,
                step_fn=hook.wrap(sink, steps) if hook else _alpha_wrap(sink, steps),
                observe=True,
            )
    finally:
        if hook is not None:
            hook.remove()
    worst: float | None = 0.0 if hook is not None else None
    error = None
    rows = {"X": [], "groups": [], **{k: [] for k in w_heads}, E0H_TARGET: []}
    for t, lg, att, mask_t in steps:
        if t < m:
            continue
        if lg is not None:
            worst = max(worst, logit_control(lg, att, mask_t))
            reg = lg["logits"]
        else:
            real = (mask_t != 0).view(1, mask_t.shape[0], 1, -1, 1)
            la = torch.stack([a.to(torch.float64) for a in att]).log()
            if not bool(torch.isfinite(la.masked_select(real.expand_as(la))).all()):
                error = f"doc {doc.doc_id} step {t}: log alpha is not finite"
                break
            reg = la.masked_fill(~real.expand_as(la), 0.0)
        Lc, _B, H, Q, Mm = reg.shape
        col = collapse_mean(reg[:, 0].reshape(1, Lc * H, Q, Mm), mask_t)[0]
        i_idx = torch.arange(t - m, t)  # FIFO memory at t, oldest first
        rows["X"].append(col.T)  # [m, Lc*H]
        rows["groups"].append(torch.full((m,), t))
        f = bilinear_features(
            torch.stack([rec.gest[int(i)] for i in i_idx]),
            ctx[t].reshape(1, -1).expand(m, -1),
        )
        for k, w in w_heads.items():
            rows[k].append(f @ w.to(torch.float64))
        rows[E0H_TARGET].append(rec.D[0][t, i_idx])
    rep = simulate(doc, Scripted(inst.victims), m)["queries"]
    out = {
        "doc": doc.doc_id,
        "answers": answers,
        "victims": inst.victims,
        "hits": inst.hits,
        "residency_ok": [(q["gap"], q["hit"]) for q in rep] == inst.hits,
        "sum_worst": inst.sum_worst,
        "log": inst.log,
        "logit_control_worst": worst,
        "e0h_error": error,
    }
    ok = rows["X"] and error is None
    out["e0h"] = {k: torch.cat(v) for k, v in rows.items()} if ok else None
    return out


def e0h_parser() -> ArgumentParser:
    """Build review F2: phase A writes `runs/b2-psi-probe-fit/phaseA`
    (FIT_RUN_ID) and phase B writes `runs/b2-psi-probe/phaseB` (RUN_ID)."""
    ap = ArgumentParser(prog="run.py e0h")
    ap.add_argument("--fits", type=Path, default=ROOT / "runs" / FIT_RUN_ID / "phaseA")
    ap.add_argument("--eval-dir", type=Path, default=ROOT / "runs" / RUN_ID / "phaseB")
    ap.add_argument("--checkpoint", type=int, default=HEADLINE)
    return ap


def e0h_main(argv: list[str] | None = None) -> int:
    """A1.3: `run.py e0h`, its own rc and its own ledger keys (`e0h.*`). Reads the
    E0h accumulators the EVAL phase wrote from the Tier 1 FIFO rollout, and B2's
    FIT_VAL R² for ψ̂-U(gamma=0) (UNINFORMATIVE if ≤ 0). Either absent: exit 3. Any
    exception: exit 3, traceback kept. Exit 1 only as COLLINEAR under a ruling."""
    a = e0h_parser().parse_args(argv)
    try:
        need = [a.fits / f"ckpt{a.checkpoint}-seed{s}.pt" for s in SEEDS] + [
            a.eval_dir / f"ckpt{a.checkpoint}-seed{s}.e0h.pt" for s in SEEDS
        ]
        missing = [str(p) for p in need if not p.exists()]
        if missing:
            print(f"DID NOT RUN: E0h inputs absent: {missing[:3]}", file=sys.stderr)
            return 3
        result = e0h_compute(a.fits, a.eval_dir, a.checkpoint)
        (a.eval_dir / f"e0h-ckpt{a.checkpoint}.json").write_text(
            json.dumps(result, indent=1, default=str)
        )
        return result["rc"]
    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        try:  # beside the EVAL outputs; the phase-A record is never written to
            a.eval_dir.mkdir(parents=True, exist_ok=True)
            (a.eval_dir / "e0h_traceback.txt").write_text(tb)
        except OSError:
            pass
        return 3


R2_FORM = "1-sse/sst"
"""The FIT_VAL R² form a fit payload carries. A payload without it predates the fix
(phase A, 280a2ad: ``1 - n_val*SSE/SST``) and is read only through its erratum."""


def r2_from_logged_n_val_form(logged: float | None, n_val: int) -> float | None:
    """Erratum P0.2, exact: logged = 1 - n·SSE/SST, so R² = 1 - (1 - logged)/n."""
    return None if logged is None else 1.0 - (1.0 - float(logged)) / n_val


def r2_sidecar_path(fits: Path, label: int, seed: int) -> Path:
    return fits / f"ckpt{label}-seed{seed}.r2-corrected.json"


def selected_val_r2(head: dict, fits: Path, label: int, seed: int, key: str):
    """The selected λ's FIT_VAL R² of one head, correct whichever code wrote it: a
    payload carrying ``val_r2_form`` is read as is; a legacy one only through its
    erratum sidecar (the logged value is never read raw). Neither: ControlFailure,
    so E0h exits 3 rather than gate on a wrong R²."""
    if head.get("val_r2_form") == R2_FORM:
        return head["val_r2_raw"][head["selected"]]
    side = r2_sidecar_path(fits, label, seed)
    if not side.exists():
        raise ControlFailure(
            f"{key} at ckpt{label} seed {seed} has a pre-fix R² and no erratum {side}"
        )
    row = json.loads(side.read_text())["heads"][key]
    if row["original"] != head["val_r2_raw"][head["selected"]]:
        raise ControlFailure(f"erratum {side} does not match its payload for {key}")
    return row["corrected"]


def e0h_compute(fits: Path, eval_dir: Path, label: int) -> dict:
    """§10 per seed: within-step R² (primary) and pooled R² (secondary) of each E0h
    head and of the gamma = 0 target on the logits; the proposed class; the rc."""
    per, r2, sources = {}, {}, {}
    for seed in SEEDS:
        acc = torch.load(eval_dir / f"ckpt{label}-seed{seed}.e0h.pt")
        src = acc.get("regressor_source", "prehook_logit")
        sources[seed] = src
        if src not in E0H_SOURCES:
            return {"rc": 3, "why": f"seed {seed}: unknown regressor source {src!r}"}
        if acc.get("errors"):
            return {"rc": 3, "why": f"seed {seed}: E0h rows failed: {acc['errors'][:3]}"}
        worst = acc["logit_control_worst"]
        if src == "prehook_logit" and (worst is None or worst > LOGIT_TOL):
            return {
                "rc": 3,
                "why": f"A1.5 logit control failed on seed {seed}: worst {worst}",
            }
        vals = {k: R2Acc.from_state(st) for k, st in acc["acc"].items()}
        per[seed] = {  # A1.5: on log alpha only the within-step primary runs
            k: {
                "within": v.within(),
                "pooled": v.pooled() if src == "prehook_logit" else None,
                "n": v.n,
            }
            for k, v in vals.items()
        }
        head = torch.load(fits / f"ckpt{label}-seed{seed}.pt")["heads"]["U@0.0"]
        val_r2 = selected_val_r2(head, fits, label, seed, "U@0.0")
        w = per[seed]["U@0.0"]["within"]
        r2[seed] = w if (val_r2 is not None and val_r2 > 0 and w is not None) else None
    cls = e0h_classify(r2)
    ratified = e0h_ratified()
    inherited = [
        s
        for s in SEEDS
        if (per[s]["U@0.0"]["within"] or 0) >= E0H_COLLINEAR
        and (per[s]["U+@0.0"]["within"] or 0) < E0H_COLLINEAR
    ]
    return {
        "e0h.class_proposed": cls if ratified else None,
        "e0h.class_unratified_reading": cls,
        "e0h.ratified": ratified,
        "e0h.per_seed": per,
        "e0h.r2_headline": r2,
        "e0h.regressor_source": (
            sources[SEEDS[0]] if len(set(sources.values())) == 1 else sources
        ),
        "e0h.pooled": (
            "ran"
            if set(sources.values()) == {"prehook_logit"}
            else "DID NOT RUN (log alpha, A1.5)"
        ),
        "e0h.k0_inherited_seeds": inherited,
        "rc": e0h_rc(cls, ratified=ratified),
    }


# --------------------------------------------------------------------------- #
# the model side: load, controls, T0
# --------------------------------------------------------------------------- #


def fresh_stream_manifest_d(source: Path) -> int:
    """§2: `d` is read from `runs/fresh-stream/manifest.json` (key `d`)."""
    return int(json.loads((source / "manifest.json").read_text())["d"])


def load_checked(source: Path, label: int, seed: int):
    """§2: sha256-pinned, `restore_rng=False`, eval, M/S/d asserted. -> (model, vmap)"""
    ck = LR.ckpt_path(source, label, seed)
    got = LR.sha256(ck)
    if got != CKPT_SHA256[(label, seed)]:
        raise ControlFailure(f"checkpoint sha256 mismatch {ck}: {got}")
    _sets, vmap, V = CSC.doc_sets(seed, LR.VOCAB_DOCUMENTS)
    model = LR.load_model(ck, V)
    d_manifest = fresh_stream_manifest_d(source)
    D_ckpt = int(model.embed.weight.shape[1])
    if not (model.cfg.M == M_STEPS and D_ckpt == d_manifest == model.cfg.D):
        raise ControlFailure(
            f"M/d mismatch: M={model.cfg.M} D={D_ckpt} manifest d={d_manifest}"
        )
    return model, vmap


T0_RECORD = Path.home() / "rsr-substrate" / "2026-09-27" / "RESTORE.md"


def t0_manifest_path(record: Path = T0_RECORD) -> Path:
    """A1.15: the T0 manifest path is read from the T0 record. No
    `runs/t0-substrate/manifest.json` exists; its equivalent is the backup's
    RESTORE.md, whose `Backup root:` line names the directory (declared)."""
    if not record.exists():
        raise ControlFailure(f"no T0 record at {record}")
    if record.suffix == ".json":
        return Path(json.loads(record.read_text())["manifest"])
    mt = re.search(r"Backup root:\s*`([^`]+)`", record.read_text())
    if not mt:
        raise ControlFailure(f"T0 record {record} names no backup root")
    return Path(mt.group(1)) / "MANIFEST.sha256"


def t0_verify(record: Path = T0_RECORD) -> dict:
    """§2 / §11.1: re-verify every file of T0's manifest against the main checkout."""
    manifest = t0_manifest_path(record)
    git = "/opt/homebrew/bin/git" if Path("/opt/homebrew/bin/git").exists() else "git"
    common = subprocess.run(
        [git, "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    main = Path(common.stdout.strip()).parent
    bad, n = [], 0
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        h, rel = line.split(None, 1)
        n += 1
        if LR.sha256(main / rel.strip()) != h:
            bad.append(rel.strip())
    return {
        "ok": n > 0 and not bad,
        "n": n,
        "bad": bad[:10],
        "n_bad": len(bad),
        "manifest": str(manifest),
    }


def t0_checked(record: Path = T0_RECORD) -> dict:
    """`t0_verify` that reports a missing or unreadable T0 record as a failed check
    (the caller exits 3) instead of an uncaught exception (exit 1; review F5)."""
    try:
        return t0_verify(record)
    except (ControlFailure, OSError, ValueError, subprocess.CalledProcessError) as e:
        return {"ok": False, "why": str(e)}


def preflight() -> list[str]:
    """§11 control 2, before any model is loaded."""
    return [f"range: {p}" for p in range_problems()] + [
        f"aliasing: {p}" for p in aliasing_problems()
    ]


# --------------------------------------------------------------------------- #
# sizing (TBD-1, TBD-2): FIT_VAL only, as the PREREG allows
# --------------------------------------------------------------------------- #


def _peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS


def _timed(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t0


TIER1_ARM_KINDS = (
    "psi",
    "psi",
    "fifo",
    "age",
    "age",
    "random",
    "random",
    "random",
    "random",
    "random",
    "oracle",
    "kind",
)


def size_child(
    label: int, seed: int, source: Path, out: Path, n_docs: int, n_arm_docs: int
) -> int:
    """Capture the first `n_docs` FIT_VAL documents (timed), assert the row
    counts, and time one document of every arm type on `n_arm_docs` of them. No
    fit, no λ, no accuracy is computed or kept: the ψ̂ arms run with `w = 0`
    (timing only; every tie goes to the oldest)."""
    torch.set_num_threads(int(os.environ.get("RSR_B2_THREADS", "1")))
    docs = docs_in(seed, "FIT_VAL", n_docs)
    require_range([d.doc_id for d in docs], "FIT_VAL")
    try:
        model, vmap = load_checked(source, label, seed)
    except ControlFailure as e:
        print(e, file=sys.stderr)
        return 3
    clos = vocabulary_closure(docs, vmap)
    if not clos["ok"]:
        print(f"closure: {clos}", file=sys.stderr)
        return 3
    kw = {"vmap": vmap, "S": S_STEPS, "m": M_STEPS, "L": L_TOKENS}
    caps, cap_s = [], []
    for d in docs:
        c = capture_doc(model, d, **kw)
        caps.append(c)
        cap_s.append(c.seconds)
    rows = {
        "U": [len(row_index(c, "U", M_STEPS)[0]) for c in caps],
        "C": [len(row_index(c, "C", M_STEPS)[0]) for c in caps],
    }
    D = int(model.cfg.D)
    means = {"assert": 0.0, "query": 0.0, "filler": 0.0}
    arm_s: dict[str, list[float]] = {}
    wz = torch.zeros(p_of(D), dtype=torch.float64)
    wa = torch.zeros(S_STEPS - 1, dtype=torch.float64)
    for d in docs[:n_arm_docs]:
        makers = {
            "fifo": lambda d=d: fifo_policy(d),
            "random": lambda d=d: random_policy(0, seed, d.doc_id),
            "oracle": lambda d=d: CheckedOracle(d, 0.9, discounted_demand(d, 0.9)),
            "factfiller": lambda d=d: factfiller_policy(d, seed),
            "kind": lambda d=d: KindOracle(
                means, doc=d, model_seed=seed, tie="random", S=S_STEPS
            ),
            "psi": lambda d=d: ProbeArgminPolicy(
                wz, "bilinear", doc=d, model_seed=seed, arm="timing", S=S_STEPS
            ),
            "age": lambda d=d: ProbeArgminPolicy(
                wa, "age", doc=d, model_seed=seed, arm="timing", S=S_STEPS
            ),
        }
        for name, mk in makers.items():
            _r, s = _timed(
                run_arm, model, d, mk(), check_sum=name in ("psi", "age"), **kw
            )
            arm_s.setdefault(name, []).append(s)
        e0h = {"U@0.0": wz, "U+@0.0": wz, "C@0.0": wz}
        _r, s = _timed(fifo_with_e0h, model, d, e0h, **kw)
        arm_s.setdefault("fifo+e0h_hook", []).append(s)
    payload = {
        "checkpoint": label,
        "seed": seed,
        "range": "FIT_VAL",
        "doc_ids": [d.doc_id for d in docs],
        "D": D,
        "M": model.cfg.M,
        "S": S_STEPS,
        "capture_seconds": cap_s,
        "rows": rows,
        "identity_worst": {r: max(c.identity_worst[r] for c in caps) for r in RANKS},
        "sum_worst": max(c.sum_worst for c in caps),
        "n_probe_rows": [c.n_probe_rows for c in caps],
        "arm_seconds": arm_s,
        "threads": torch.get_num_threads(),
        "peak_rss_gb": _peak_rss_gb(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    torch.save(
        [
            {
                "doc_id": c.doc_id,
                "gest": c.gest,
                "ctx": c.ctx,
                "D": c.D,
                "resident": c.resident,
                "kinds": c.kinds,
            }
            for c in caps
        ],
        out.with_suffix(".caps.pt"),
    )
    return 0


def size_gram(caps_file: Path, n_docs: int, threads: int) -> dict:
    """TBD-2's Gram-accumulation and solve cost, on FIT_VAL captures: time the
    U and C `XᵀX` accumulation per document and one full-size fp64 `eigh`. The
    Gram and its eigenvectors are discarded; no `w` is solved."""
    torch.set_num_threads(threads)
    raw = torch.load(caps_file)[:n_docs]
    caps = [
        DocCapture(
            doc_id=r["doc_id"],
            gest=r["gest"],
            ctx=r["ctx"],
            D=r["D"],
            resident=r["resident"],
            kinds=r["kinds"],
            sum_worst=0.0,
            identity_worst={},
        )
        for r in raw
    ]
    require_range([c.doc_id for c in caps], "FIT_VAL")
    d = caps[0].gest.shape[1]
    p = p_of(d)
    out: dict[str, Any] = {"p": p, "threads": threads, "n_docs": len(caps)}
    G = Gram(p, 1)
    for rowset in ("U", "C"):
        secs = []
        for cap in caps:
            t, i = row_index(cap, rowset, M_STEPS)
            t0 = time.perf_counter()
            X = design(cap, t, i, "bilinear")
            G.add(X, torch.zeros(len(t), 1, dtype=torch.float64))
            secs.append(time.perf_counter() - t0)
        out[f"gram_{rowset}_seconds_per_doc"] = secs
    G.xtx.diagonal().add_(1e-6 * float(torch.trace(G.xtx)) / p)
    _e, s = _timed(spectral, G.xtx)
    out["eigh_seconds"] = s
    del _e
    out["peak_rss_gb"] = _peak_rss_gb()
    return out


def run_size(
    source: Path,
    root: Path,
    n_docs: int,
    n_arm_docs: int,
    gram_docs: int,
    gram_threads: int,
    parallel: int,
) -> Exit:  # pragma: no cover
    """The sizing run (TBD-1, TBD-2), with a ledger. FIT_VAL only."""
    from ledger import Ledger

    probs = preflight()
    if probs:
        return did_not_run("; ".join(probs))
    led = Ledger(
        SIZING_RUN_ID,
        question="B2 sizing: n and capture cost on FIT_VAL only "
        "(PREREG TBD-1, TBD-2, A1.4)",
        runs_root=root.parent,
    )
    led.run_meta(device="cpu", steps_requested=0, steps_done=0)
    led.manifest(
        {
            "run_id": SIZING_RUN_ID,
            "prereg": PREREG,
            "prereg_commits": PREREG_COMMITS,
            "range": "FIT_VAL",
            "range_ids": list(RANGES["FIT_VAL"]),
            "n_docs": n_docs,
            "n_arm_docs": n_arm_docs,
            "gram_docs": gram_docs,
            "gram_threads": gram_threads,
            "parallel_children": parallel,
            "child_threads": int(os.environ.get("RSR_B2_THREADS", "1")),
            "checkpoints": list(CHECKPOINTS),
            "seeds": SEEDS,
            "source": str(source),
            "checkpoint_sha256": {f"{c}.{s}": h for (c, s), h in CKPT_SHA256.items()},
            "expected": "capture ~1-4 s/doc at 1 thread; U=1128 and C=632 rows per doc; "
            "eigh at p=16640 minutes, not hours",
            "falsifier": "NONE: sizing only (PREREG fence item 1 and 2)",
        }
    )
    t0v = t0_verify()
    led.note("T0.start", t0v, how="run.py::t0_verify")
    if not t0v["ok"]:
        led.status("failed")
        led.write()
        return did_not_run(f"T0 manifest: {t0v}")
    jobs = [(c, s) for c in CHECKPOINTS for s in SEEDS]
    rcs, running = {}, {}
    (root / "logs").mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    while jobs or running:
        while jobs and len(running) < parallel:
            j = jobs.pop(0)
            argv = [
                sys.executable,
                str(Path(__file__).resolve()),
                "size-child",
                "--checkpoint",
                str(j[0]),
                "--seed",
                str(j[1]),
                "--source",
                str(source),
                "--out",
                str(root / "raw" / f"ckpt{j[0]}-seed{j[1]}.json"),
                "--n-docs",
                str(n_docs),
                "--n-arm-docs",
                str(n_arm_docs),
            ]
            log = open(root / "logs" / f"ckpt{j[0]}-seed{j[1]}.log", "w")  # noqa: SIM115
            running[j] = (
                subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT),
                argv,
            )
        for j, (p, argv) in list(running.items()):
            rc = p.poll()
            if rc is not None:
                rcs[j] = rc
                led.command(argv, exit_code=rc)
                del running[j]
        time.sleep(2)
    led.note("children_wall_seconds", time.time() - t_start, how="parent wall clock")
    failed = [j for j, rc in rcs.items() if rc != 0]
    pay = {}
    for (c, s), rc in sorted(rcs.items()):
        if rc == 0:
            pay[(c, s)] = json.loads((root / "raw" / f"ckpt{c}-seed{s}.json").read_text())
    gram = (
        size_gram(root / "raw" / f"ckpt{HEADLINE}-seed0.caps.pt", gram_docs, gram_threads)
        if (HEADLINE, 0) in pay
        else None
    )
    write_sizing_rows(led, pay, gram)
    t0e = t0_verify()
    led.note("T0.end", t0e, how="run.py::t0_verify")
    ok = (
        not failed
        and t0e["ok"]
        and all(
            set(v["rows"]["U"]) == {u_rows(S_STEPS)}
            and set(v["rows"]["C"]) == {c_rows(S_STEPS, M_STEPS)}
            and v["sum_worst"] <= SUM_TOL
            and max(v["identity_worst"].values()) <= IDENTITY_TOL
            for v in pay.values()
        )
    )
    led.run_meta(seeds_actually_run=sorted({s for (_c, s) in pay}))
    led.status("ok" if ok else "partial")
    led.verdict(
        falsifier="NONE: sizing",
        outcome="inconclusive",
        detail=f"sizing only; failed children {failed}; controls ok {ok}",
    )
    led.command(
        [sys.executable, EXPERIMENT, "size"], exit_code=0 if ok else 3, note="the parent"
    )
    led.write()
    return Exit.OK if ok else Exit.DID_NOT_RUN


def write_sizing_rows(led, pay: dict, gram: dict | None) -> None:  # pragma: no cover
    for c in CHECKPOINTS:
        ks = [(c, s) for s in SEEDS if (c, s) in pay]
        if len(ks) < 2:
            continue
        vs = [pay[k] for k in ks]
        how = "run.py::size_child, FIT_VAL; seed order " + ",".join(
            str(s) for _c, s in ks
        )
        led.stat(
            f"ckpt{c}.capture_s_per_doc.mean",
            [sum(v["capture_seconds"]) / len(v["capture_seconds"]) for v in vs],
            how=how,
        )
        led.stat(
            f"ckpt{c}.capture_s_per_doc.max",
            [max(v["capture_seconds"]) for v in vs],
            how=how,
        )
        led.stat(
            f"ckpt{c}.rows_U_per_doc",
            [sum(v["rows"]["U"]) / len(v["rows"]["U"]) for v in vs],
            how=how,
        )
        led.stat(
            f"ckpt{c}.rows_C_per_doc",
            [sum(v["rows"]["C"]) / len(v["rows"]["C"]) for v in vs],
            how=how,
        )
        led.stat(f"ckpt{c}.peak_rss_gb", [v["peak_rss_gb"] for v in vs], how=how)
        led.stat(f"ckpt{c}.sum_worst", [v["sum_worst"] for v in vs], how=how)
        for r in RANKS:
            led.stat(
                f"ckpt{c}.identity_worst.rank{r}",
                [
                    v["identity_worst"][str(r)]
                    if str(r) in v["identity_worst"]
                    else v["identity_worst"][r]
                    for v in vs
                ],
                how=how,
            )
        for arm in vs[0]["arm_seconds"]:
            led.stat(
                f"ckpt{c}.arm_s_per_doc.{arm}",
                [sum(v["arm_seconds"][arm]) / len(v["arm_seconds"][arm]) for v in vs],
                how=how,
            )
        led.stat(f"ckpt{c}.D", [v["D"] for v in vs], how=how)
        led.stat(f"ckpt{c}.n_docs", [len(v["doc_ids"]) for v in vs], how=how)
    if gram is not None:
        for k in ("gram_U_seconds_per_doc", "gram_C_seconds_per_doc"):
            led.stat(
                f"gram.{k}",
                gram[k],
                how="run.py::size_gram, ckpt3000 seed0 FIT_VAL captures",
            )
        led.note("gram.eigh_seconds", gram["eigh_seconds"], how="run.py::size_gram")
        led.note("gram.peak_rss_gb", gram["peak_rss_gb"], how="run.py::size_gram")
        led.note("gram.p", gram["p"], how="run.py::p_of")
        led.note("gram.threads", gram["threads"], how="run.py::size_gram")
    led.note(
        "N_F",
        n_f(p_of(next(iter(pay.values()))["D"])) if pay else None,
        how="run.py::n_f (A1.15: ceil(10 p / 632))",
    )


# --------------------------------------------------------------------------- #
# phase A (fit) and phase B (eval): the arms, per (checkpoint, seed)
# --------------------------------------------------------------------------- #

BILINEAR_U_SPECS = [
    ("U", 0.9),
    ("U", 0.0),
    ("U+", 0.9),
    ("U+", 0.0),
    ("U1", 0.9),
    ("U2", 0.9),
    ("U3", 0.9),
]
BILINEAR_C_SPECS = [("C", 0.9), ("C", 0.0), ("C+", 0.9), ("C+", 0.0)]
AGE_U_SPECS = [("U", 0.9), ("U", 0.0)]
AGE_C_SPECS = [("C", 0.9), ("C", 0.0)]
PSI_HEADS = ("U", "C", "U+", "C+", "U1", "U2", "U3")


def fit_all(train_caps, val_caps, *, m: int) -> dict:
    """Every §6 / A1.2 fit for one (checkpoint, seed). RidgeFailure propagates."""
    heads = {}
    for rowset, specs in (("U", BILINEAR_U_SPECS), ("C", BILINEAR_C_SPECS)):
        for (a, g), f in fit_heads(
            train_caps, val_caps, rowset, "bilinear", specs, m=m
        ).items():
            heads[f"{a}@{g}"] = f
    for rowset, specs in (("U", AGE_U_SPECS), ("C", AGE_C_SPECS)):
        for (a, g), f in fit_heads(
            train_caps, val_caps, rowset, "age", specs, m=m
        ).items():
            heads[f"age_{a}@{g}"] = f
    dec = {r: age_decodability(train_caps, val_caps, r, m=m) for r in ("U", "C")}
    means = {g: class_means(train_caps, g, m=m) for g in GAMMAS}
    return {"heads": heads, "age_decodability": dec, "class_means": means}


def _pearson(a: Tensor, b: Tensor) -> float | None:
    a, b = a.to(torch.float64), b.to(torch.float64)
    a, b = a - a.mean(), b - b.mean()
    den = float(a.norm() * b.norm())
    return float(a @ b) / den if den > 0 else None


def _ranks(x: Tensor) -> Tensor:
    """Average ranks (ties share their mean rank), for Spearman."""
    x = x.to(torch.float64)
    order = torch.argsort(x, stable=True)
    r = torch.empty_like(x)
    r[order] = torch.arange(len(x), dtype=torch.float64)
    uniq, inv = torch.unique(x, return_inverse=True)
    s = torch.zeros(len(uniq), dtype=torch.float64).index_add_(0, inv, r)
    c = torch.zeros(len(uniq), dtype=torch.float64).index_add_(0, inv, torch.ones_like(r))
    return (s / c)[inv]


def psi_age_corr(caps, w: Tensor, feature: str, *, m: int) -> dict:
    """§8.2 on FIT_VAL capture rows: corr(ψ̂, age) over live (FIFO-resident)
    slots at full-memory steps, Pearson and Spearman."""
    psi, age = [], []
    for cap in caps:
        t, i = row_index(cap, "C", m)
        keep = t >= m
        t, i = t[keep], i[keep]
        psi.append(design(cap, t, i, feature) @ w.to(torch.float64))
        age.append((t - i).to(torch.float64))
    p, a = torch.cat(psi), torch.cat(age)
    return {
        "pearson": _pearson(p, a),
        "spearman": _pearson(_ranks(p), _ranks(a)),
        "n": len(p),
    }


def arm_makers(fits: dict, seed: int, gamma: float, S: int, names=None) -> dict:
    """§7's arms at one gamma, each a `doc -> policy` factory (a fresh policy per
    document). `names` restricts the set."""
    h = fits["heads"]

    def psi(key, arm):
        return lambda d: ProbeArgminPolicy(
            h[key]["w"], "bilinear", doc=d, model_seed=seed, arm=arm, S=S
        )

    def age(key, arm):
        return lambda d: ProbeArgminPolicy(
            h[key]["w"], "age", doc=d, model_seed=seed, arm=arm, S=S
        )

    mk = {
        "psiU": psi(f"U@{gamma}", "psiU"),
        "psiC": psi(f"C@{gamma}", "psiC"),
        "psiU+": psi(f"U+@{gamma}", "psiU+"),
        "psiC+": psi(f"C+@{gamma}", "psiC+"),
        "fifo": fifo_policy,
        "ageU": age(f"age_U@{gamma}", "ageU"),
        "ageC": age(f"age_C@{gamma}", "ageC"),
        "oracle": lambda d: CheckedOracle(d, gamma, discounted_demand(d, gamma)),
        "kind": lambda d: KindOracle(
            fits["class_means"][gamma], doc=d, model_seed=seed, tie="random", S=S
        ),
        "kind_oldest": lambda d: KindOracle(
            fits["class_means"][gamma], doc=d, model_seed=seed, tie="oldest", S=S
        ),
        "factfiller": lambda d: factfiller_policy(d, seed),
    }
    for k in range(N_RANDOM):
        mk[f"random{k}"] = lambda d, k=k: random_policy(k, seed, d.doc_id)
    if gamma == 0.9:
        for r in (1, 2, 3):
            mk[f"psiU{r}"] = psi(f"U{r}@0.9", f"psiU{r}")
    return mk if names is None else {n: mk[n] for n in names}


#: A1.4's tiers. Tier 1 (ckpt3000, gamma = 0.9) on N_E documents; Tier 2 on the first
#: min(N_E, 4096). FIFO and random do not depend on gamma, so gamma = 0 reads them from
#: the gamma = 0.9 run on the same documents (paired).
RANDOMS = tuple(f"random{k}" for k in range(N_RANDOM))
TIER1 = ("psiU", "psiC", "fifo", "ageU", "ageC", *RANDOMS, "oracle", "kind")
TIER2_G09 = ("psiU1", "psiU2", "psiU3", "psiU+", "psiC+", "kind_oldest", "factfiller")
TIER2_G0 = (
    "psiU",
    "psiC",
    "ageU",
    "ageC",
    "oracle",
    "kind",
    "kind_oldest",
    "psiU+",
    "psiC+",
)


def tier_plan(label: int) -> dict[tuple[float, str], tuple[str, ...]]:
    """(gamma, tier) -> arms, per A1.4's table (28 arms at ckpt2500, 12 + 16 at 3000)."""
    if label == HEADLINE:
        return {
            (0.9, "tier1"): TIER1,
            (0.9, "tier2"): TIER2_G09,
            (0.0, "tier2"): TIER2_G0,
        }
    return {(0.9, "tier2"): TIER1 + TIER2_G09, (0.0, "tier2"): TIER2_G0}


def run_arms(model, docs, makers: dict, *, vmap, S, m, L, e0h_heads=None) -> dict:
    """Every arm on every document: `[D, 2]` counts per bucket, logs, controls. With
    `e0h_heads`, the FIFO arm is `fifo_with_e0h` and E0h's R² statistics stream."""
    per = {a: {b: [] for b in BUCKETS} for a in makers}
    logs: dict[str, list] = {a: [] for a in makers}
    ctrl = {
        "residency_ok": True,
        "sum_worst": 0.0,
        "logit_control_worst": 0.0,
        "doc_ids": [d.doc_id for d in docs],
    }
    e0h = None
    if e0h_heads is not None:
        e0h = {k: R2Acc(0) for k in (*e0h_heads, E0H_TARGET)}
    for d in docs:
        for a, mk in makers.items():
            if a == "fifo" and e0h_heads is not None:
                r = fifo_with_e0h(model, d, e0h_heads, vmap=vmap, S=S, m=m, L=L)
                ctrl["logit_control_worst"] = max(
                    ctrl["logit_control_worst"], r["logit_control_worst"]
                )
                rows = r["e0h"]
                for k in e0h:
                    if e0h[k].k == 0:
                        e0h[k] = R2Acc(rows["X"].shape[1])
                    e0h[k].add(rows[k], rows["X"], rows["groups"])
            else:
                r = run_arm(
                    model,
                    d,
                    mk(d),
                    vmap=vmap,
                    S=S,
                    m=m,
                    L=L,
                    check_sum=a.startswith(("psi", "age")),
                )
            for b in BUCKETS:
                per[a][b].append(counts(r["answers"], b))
            logs[a] += r["log"]
            ctrl["residency_ok"] &= r["residency_ok"]
            ctrl["sum_worst"] = max(ctrl["sum_worst"], r["sum_worst"])
    tens = {
        a: {b: torch.tensor(v, dtype=torch.long).reshape(-1, 2) for b, v in bb.items()}
        for a, bb in per.items()
    }
    return {"counts": tens, "logs": logs, "controls": ctrl, "e0h": e0h}


def pooled_acc(t: Tensor) -> float:
    return float(t[:, 1].sum()) / float(t[:, 0].sum())


def val_decisions(val_arms: dict[float, dict], seed: int, doc_ids) -> dict:
    """§9.3 / §9.4 on FIT_VAL for one (checkpoint, seed): ref per (arm, gamma), δ
    (gamma = 0 takes gamma = 0.9's, A1.11), and the paired-bootstrap SD of every §9.4
    contrast at gamma = 0.9 (the six gating ones size N_E; ψ̂-U - kind-oracle is
    reported)."""
    require_range(doc_ids, "FIT_VAL")
    out: dict[str, Any] = {"ref": {}, "acc": {}}
    for g, res in val_arms.items():
        acc = {a: pooled_acc(c["all"]) for a, c in res["counts"].items()}
        out["acc"][g] = acc
        for arm, age in (("U", "ageU"), ("C", "ageC")):
            if age in acc:
                out["ref"][f"{arm}@{g}"] = select_ref(
                    {"fifo": acc["fifo"], "age": acc[age]}, doc_ids
                )
    d09 = delta_of(out["acc"][0.9]["oracle"], out["acc"][0.9]["fifo"])
    out["delta"] = deltas_by_gamma({0.9: d09})
    res = val_arms[0.9]
    sig = {}
    for b, suf in (("all", ""), ("gap_2_to_M", ".gap_2_to_M")):
        per = {a: c[b] for a, c in res["counts"].items()}
        con = {}
        for arm, age in (("U", "ageU"), ("C", "ageC")):
            ref = "fifo" if out["ref"][f"{arm}@0.9"] == "fifo" else age
            con[f"psi{arm}_minus_ref{arm}{suf}"] = (f"psi{arm}", ref)
        if b == "all":
            con["psiU_minus_random"] = ("psiU", "random")
            con["psiC_minus_random"] = ("psiC", "random")
            con["psiU_minus_kind"] = ("psiU", "kind")
        for k, v in paired_bootstrap(per, con, seed).items():
            sig[k] = v["sd"]
    out["sigma"] = sig
    return out


def determinism(
    model, docs, makers: dict, first: dict[str, dict], *, vmap, S, m, L
) -> bool:
    """§11 control 8: re-running the first 8 FIT_VAL documents for every arm
    reproduces every victim and every `correct` flag."""
    for d in docs[:N_DETERMINISM_DOCS]:
        for a, mk in makers.items():
            r = run_arm(model, d, mk(d), vmap=vmap, S=S, m=m, L=L, check_sum=False)
            if (r["victims"], r["answers"]) != first[a][d.doc_id]:
                return False
    return True


def fit_core(model, train_docs, val_docs, *, seed, vmap, S, m, L) -> dict:
    """Phase A's measurement for one (checkpoint, seed), IO-free. Raises
    ControlFailure (exit 3) or RidgeFailure (exit 1)."""
    require_range([d.doc_id for d in train_docs], "FIT_TRAIN")
    require_range([d.doc_id for d in val_docs], "FIT_VAL")
    clos = vocabulary_closure(list(train_docs) + list(val_docs), vmap)
    if not clos["ok"]:
        raise ControlFailure(f"vocabulary closure: {clos}")
    kw = {"vmap": vmap, "S": S, "m": m, "L": L}
    t0 = time.time()
    tcaps = [capture_doc(model, d, **kw) for d in train_docs]
    vcaps = [capture_doc(model, d, **kw) for d in val_docs]
    cap_s = time.time() - t0
    ident = {r: max(c.identity_worst[r] for c in tcaps + vcaps) for r in RANKS}
    sumw = max(c.sum_worst for c in tcaps + vcaps)
    if max(ident.values()) > IDENTITY_TOL or sumw > SUM_TOL:
        raise ControlFailure(f"C4/C5: identity {ident} sum {sumw}")
    t0 = time.time()
    fits = fit_all(tcaps, vcaps, m=m)
    fit_s = time.time() - t0
    corr = {
        k: psi_age_corr(vcaps, f["w"], f["feature"], m=m)
        for k, f in fits["heads"].items()
    }
    first: dict[str, dict] = {}
    val_arms = {}
    t_arms = time.time()
    names = {0.9: (*TIER1, "kind_oldest", "factfiller"), 0.0: ("fifo", "ageU", "ageC")}
    for g in GAMMAS:
        mk = arm_makers(fits, seed, g, S, names[g])
        val_arms[g] = run_arms(model, val_docs, mk, **kw)
        c = val_arms[g]["controls"]
        if not c["residency_ok"] or c["sum_worst"] > SUM_TOL:
            raise ControlFailure(f"C5/C7 on FIT_VAL at gamma={g}: {c}")
        if g == 0.9:
            for a in mk:
                first[a] = {}
            for d in val_docs[:N_DETERMINISM_DOCS]:
                for a in mk:
                    r = run_arm(
                        model, d, mk[a](d), vmap=vmap, S=S, m=m, L=L, check_sum=False
                    )
                    first[a][d.doc_id] = (r["victims"], r["answers"])
            if not determinism(model, val_docs, mk, first, **kw):
                raise ControlFailure("C8: determinism")
    dec = val_decisions(val_arms, seed, [d.doc_id for d in val_docs])
    return {
        "fits": fits,
        "decisions": dec,
        "val_counts": {g: r["counts"] for g, r in val_arms.items()},
        "psi_age_corr_fit_val": corr,
        "controls": {"identity_worst": ident, "sum_worst": sumw, "closure": clos},
        "seconds": {"capture": cap_s, "fit": fit_s, "val_arms": time.time() - t_arms},
        "threads": torch.get_num_threads(),
        "n": {
            "N_F": len(train_docs),
            "n_U": sum(len(row_index(c, "U", m)[0]) for c in tcaps),
            "n_C": sum(len(row_index(c, "C", m)[0]) for c in tcaps),
        },
    }


# --------------------------------------------------------------------------- #
# EVAL, streamed (build review F1) and resumable: one unit = one (gamma, tier,
# document) of one (checkpoint, seed), written atomically before the next runs
# --------------------------------------------------------------------------- #

UNIT_FORMAT = 1


def unit_path(out_dir: Path, g: float, tier: str, doc_id: int) -> Path:
    return Path(out_dir) / "units" / f"g{g}.{tier}" / f"{doc_id}.pt.gz"


def save_unit(u: dict, path: Path) -> None:
    """Atomic: a temp file in the same directory, fsync, `os.replace`. A kill
    mid-write leaves only a `*.tmp<pid>` file, which a restart never reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    torch.save(u, buf)
    tmp = path.parent / f"{path.name}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(gzip.compress(buf.getvalue(), compresslevel=1))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_unit(path: Path) -> dict:
    try:
        raw = gzip.decompress(Path(path).read_bytes())
        return torch.load(io.BytesIO(raw), weights_only=True)
    except Exception as e:  # a unit that exists must be readable: exit 3, not a rerun
        raise ControlFailure(f"unreadable EVAL unit {path}: {e!r}") from e


def unit_logs(out_dir: Path, g: float, tier: str, doc_ids, arm: str) -> list[dict]:
    """One arm's per-eviction log over `doc_ids`, read back from the units."""
    out: list[dict] = []
    for d in doc_ids:
        out += load_unit(unit_path(out_dir, g, tier, d))["logs"][arm]
    return out


def _flush(pending: list) -> None:
    while pending:
        path, u = pending.pop(0)
        save_unit(u, path)


def code_sha256() -> str:
    """sha256 of this file: a unit written by other code is never resumed."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class AttributionAcc:
    """`attribution` (truth-table row 1's source, §4), accumulated online."""

    def __init__(self, S: int) -> None:
        self.S = S
        self.age = [0] * (S - 1)
        self.rank = [0] * (S - 1)
        self.kinds: dict[str, int] = {}
        self.margins = array("d")
        self.shifts: dict[int, int] = {}

    def add(self, log: list[dict]) -> AttributionAcc:
        for r in log:
            self.age[r["age"] - 1] += 1
            if 1 <= r["rank"] <= self.S - 1:
                self.rank[r["rank"] - 1] += 1
            k = r["kind"] if r["status"] is None else f"assert:{r['status']}"
            self.kinds[k] = self.kinds.get(k, 0) + 1
            if r["margin"] is not None:
                self.margins.append(r["margin"])
            sh = r["rank_shift"][1]
            self.shifts[sh] = self.shifts.get(sh, 0) + 1
        return self

    def result(self) -> dict:
        return {
            "age_hist": list(self.age),
            "rank_hist": list(self.rank),
            "kind_status": dict(self.kinds),
            "margin": LR.summary(list(self.margins)),
            "rank_shift": dict(sorted(self.shifts.items())),
        }


class PsiAgeAcc:
    """`psi_age_corr_in_loop` (§8.2), accumulated online: fp64 arrays, not dicts."""

    def __init__(self) -> None:
        self.psi = array("d")
        self.age = array("d")

    def add(self, log: list[dict]) -> PsiAgeAcc:
        for r in log:
            if r["psi"] is not None:
                self.psi.extend(r["psi"])
                self.age.extend(r["live_ages"])
        return self

    def result(self) -> dict:
        if not self.psi:
            return {"pearson": None, "spearman": None, "n": 0}
        p = torch.tensor(self.psi.tolist(), dtype=torch.float64)
        a = torch.tensor(self.age.tolist(), dtype=torch.float64)
        return {
            "pearson": _pearson(p, a),
            "spearman": _pearson(_ranks(p), _ranks(a)),
            "n": len(p),
        }


def run_unit(
    model, doc, makers: dict, *, vmap, S, m, L, e0h_heads=None, e0h_source=None
) -> dict:
    """Every arm of one tier on one document: counts per bucket, §11 controls 5
    and 7, the per-eviction logs, and (with `e0h_heads`) this document's E0h
    statistics from the FIFO rollout (A1.4, A1.5)."""
    u: dict[str, Any] = {
        "doc_id": doc.doc_id,
        "counts": {},
        "logs": {},
        "residency_ok": True,
        "sum_worst": 0.0,
        "logit_control_worst": None,
        "e0h": None,
        "e0h_error": None,
    }
    for a, mk in makers.items():
        if a == "fifo" and e0h_heads is not None:
            r = fifo_with_e0h(
                model, doc, e0h_heads, vmap=vmap, S=S, m=m, L=L, source=e0h_source
            )
            u["logit_control_worst"] = r["logit_control_worst"]
            u["e0h_error"] = r["e0h_error"]
            rows = r["e0h"]
            if rows is not None:
                st = {}
                for k in (*e0h_heads, E0H_TARGET):
                    acc = R2Acc(rows["X"].shape[1])
                    acc.add(rows[k], rows["X"], rows["groups"])
                    st[k] = acc.state()
                u["e0h"] = st
        else:
            r = run_arm(
                model,
                doc,
                mk(doc),
                vmap=vmap,
                S=S,
                m=m,
                L=L,
                check_sum=a.startswith(("psi", "age")),
            )
        u["counts"][a] = {b: list(counts(r["answers"], b)) for b in BUCKETS}
        u["logs"][a] = r["log"]
        u["residency_ok"] = bool(u["residency_ok"] and r["residency_ok"])
        u["sum_worst"] = max(u["sum_worst"], r["sum_worst"])
    return u


class TierAgg:
    """Folds units, in document order, into what `run_arms` returned (counts
    `[D, 2]` per arm and bucket, controls, E0h accumulators) plus the attribution
    and ψ̂-age statistics. Holds no per-eviction record (build review F1)."""

    def __init__(self, arms, S: int, e0h_source: str | None) -> None:
        self.arms = list(arms)
        self.per = {a: {b: [] for b in BUCKETS} for a in self.arms}
        self.attr = {a: AttributionAcc(S) for a in self.arms}
        self.corr = {a: PsiAgeAcc() for a in self.arms if a.startswith(("psi", "age"))}
        self.e0h_source = e0h_source
        self.ctrl = {
            "residency_ok": True,
            "sum_worst": 0.0,
            "logit_control_worst": None if e0h_source == "log_alpha" else 0.0,
            "doc_ids": [],
        }
        self.e0h: dict[str, R2Acc] | None = None
        self.e0h_errors: list[str] = []

    def add(self, u: dict) -> None:
        for a in self.arms:
            for b in BUCKETS:
                self.per[a][b].append(tuple(u["counts"][a][b]))
            self.attr[a].add(u["logs"][a])
            if a in self.corr:
                self.corr[a].add(u["logs"][a])
        c = self.ctrl
        c["residency_ok"] = bool(c["residency_ok"] and u["residency_ok"])
        c["sum_worst"] = max(c["sum_worst"], u["sum_worst"])
        c["doc_ids"].append(u["doc_id"])
        if u["logit_control_worst"] is not None and c["logit_control_worst"] is not None:
            c["logit_control_worst"] = max(
                c["logit_control_worst"], u["logit_control_worst"]
            )
        if u["e0h_error"]:
            self.e0h_errors.append(u["e0h_error"])
        if u["e0h"] is not None:
            if self.e0h is None:
                self.e0h = {k: R2Acc(0) for k in u["e0h"]}
            for k, st in u["e0h"].items():
                self.e0h[k].merge(R2Acc.from_state(st))

    def result(self) -> dict:
        return {
            "counts": {
                a: {
                    b: torch.tensor(v, dtype=torch.long).reshape(-1, 2)
                    for b, v in bb.items()
                }
                for a, bb in self.per.items()
            },
            "controls": self.ctrl,
            "e0h": self.e0h,
            "e0h_source": self.e0h_source,
            "e0h_errors": self.e0h_errors,
            "attribution": {a: v.result() for a, v in self.attr.items()},
            "psi_age_corr": {a: v.result() for a, v in self.corr.items()},
        }


def eval_tier(
    model,
    docs,
    makers: dict,
    *,
    g: float,
    tier: str,
    out_dir: Path,
    key: dict,
    vmap,
    S,
    m,
    L,
    e0h_heads=None,
    e0h_source=None,
    progress=None,
) -> dict:
    """One tier of one (checkpoint, seed), a document at a time. A document whose
    unit is on disk (written under the same `key`) is read, not run: a killed
    child loses at most the document it was running. §11 controls 5 and 7 are
    checked per document (exit 3 at once, not after the tier)."""
    agg = TierAgg(makers, S, e0h_source)
    pending: list = []
    n_resumed, n_run, t_run = 0, 0, 0.0
    for j, d in enumerate(docs):
        path = unit_path(out_dir, g, tier, d.doc_id)
        want = dict(key, doc_id=d.doc_id)
        if path.exists():
            u = load_unit(path)
            if u.get("key") != want:
                raise ControlFailure(
                    f"{path} was written under another configuration: "
                    f"{u.get('key')} != {want}"
                )
            n_resumed += 1
            sec = None
        else:
            t0 = time.perf_counter()
            u = run_unit(
                model,
                d,
                makers,
                vmap=vmap,
                S=S,
                m=m,
                L=L,
                e0h_heads=e0h_heads,
                e0h_source=e0h_source,
            )
            u["key"] = want
            if not u["residency_ok"] or u["sum_worst"] > SUM_TOL:
                raise ControlFailure(
                    f"EVAL controls at gamma={g} {tier}, document {d.doc_id}: "
                    f"residency_ok={u['residency_ok']} sum_worst={u['sum_worst']}"
                )
            pending.append((path, u))
            _flush(pending)  # F1: every unit reaches disk before the next document runs
            sec = time.perf_counter() - t0
            t_run += sec
            n_run += 1
        agg.add(u)
        del u
        if progress is not None:
            progress(g, tier, j + 1, len(docs), sec)
    _flush(pending)
    out = agg.result()
    out["n_docs"] = len(docs)
    out["n_resumed"] = n_resumed
    out["seconds"] = {"run": t_run, "n_run": n_run}
    return out


def eval_core(
    model,
    docs,
    fits: dict,
    *,
    seed,
    label,
    vmap,
    S,
    m,
    L,
    n_tier2: int,
    out_dir: Path,
    fits_id: str,
    progress=None,
) -> dict:
    """Phase B's measurement for one (checkpoint, seed) on EVAL, per A1.4's tiers,
    streamed to `out_dir` a document at a time and resumable (`eval_tier`). The
    FIFO rollout carries E0h's instrument (A1.4, A1.5). An E0h control failure
    is recorded for `run.py e0h` (exit 3 there), never raised here: A1.3, "B2's
    rc covers B2 alone" (build review F3)."""
    require_range([d.doc_id for d in docs], "EVAL")
    kw = {"vmap": vmap, "S": S, "m": m, "L": L}
    e0h_heads = {k: fits["heads"][k]["w"] for k in E0H_HEADS}
    source = e0h_source_for(model)
    base = {
        "format": UNIT_FORMAT,
        "label": label,
        "seed": seed,
        "fits_id": fits_id,
        "code_sha256": code_sha256(),
        "e0h_source": source,
        "S": S,
        "m": m,
        "L": L,
    }
    out: dict[str, Any] = {}
    for (g, tier), names in tier_plan(label).items():
        ds = docs if tier == "tier1" else docs[:n_tier2]
        want_e0h = "fifo" in names
        out[f"{g}.{tier}"] = eval_tier(
            model,
            ds,
            arm_makers(fits, seed, g, S, names),
            g=g,
            tier=tier,
            out_dir=out_dir,
            key=dict(base, gamma=g, tier=tier, arms=list(names)),
            e0h_heads=e0h_heads if want_e0h else None,
            e0h_source=source if want_e0h else None,
            progress=progress,
            **kw,
        )
    return out


def e0h_payload(out: dict) -> dict:
    """What `run.py e0h` reads (`ckpt{c}-seed{s}.e0h.pt`): the tier whose FIFO
    rollout carried the instrument."""
    r = next(r for r in out.values() if r["e0h_source"] is not None)
    return {
        "acc": {k: v.state() for k, v in (r["e0h"] or {}).items()},
        "logit_control_worst": r["controls"]["logit_control_worst"],
        "regressor_source": r["e0h_source"],
        "errors": list(r["e0h_errors"]),
        "n_docs": r["n_docs"],
    }


def contrasts_for(ref: dict[str, str]) -> dict[str, tuple[str, str]]:
    """The §9.5-9.7 / A1.2 contrasts at one gamma, given ref per arm (U / C)."""
    rU = "fifo" if ref["U"] == "fifo" else "ageU"
    rC = "fifo" if ref["C"] == "fifo" else "ageC"
    return {
        "psiU_minus_refU": ("psiU", rU),
        "psiC_minus_refC": ("psiC", rC),
        "psiU_minus_random": ("psiU", "random"),
        "psiC_minus_random": ("psiC", "random"),
        "psiU+_minus_refU": ("psiU+", rU),
        "psiC+_minus_refC": ("psiC+", rC),
        "psiU+_minus_random": ("psiU+", "random"),
        "psiC+_minus_random": ("psiC+", "random"),
        "psiU1_minus_refU": ("psiU1", rU),
        "psiU2_minus_refU": ("psiU2", rU),
        "psiU3_minus_refU": ("psiU3", rU),
        "psiU1_minus_random": ("psiU1", "random"),
        "psiU2_minus_random": ("psiU2", "random"),
        "psiU3_minus_random": ("psiU3", "random"),
        "q2_psiU_minus_kind": ("psiU", "kind"),
        "q2_psiC_minus_kind": ("psiC", "kind"),
        "oracle_minus_fifo": ("oracle", "fifo"),
    }


def outcomes_for(
    counts: dict[str, dict[str, Tensor]], ref: dict[str, str], delta, seed: int
) -> dict:
    """§9.5 per arm (ψ̂-U, ψ̂-C, the companions, U-r), §9.7 Q2, on one set of
    paired per-document counts. Contrasts whose arms are absent are skipped."""
    arms = set(counts) | (
        {"random"} if any(_RANDOM_ARM.fullmatch(a) for a in counts) else set()
    )
    con = {k: v for k, v in contrasts_for(ref).items() if v[0] in arms and v[1] in arms}
    boot = {}
    for b in ("all", "gap_2_to_M", "gap_gt_M", "gap_eq_M"):
        per = {a: c[b] for a, c in counts.items()}
        boot[b] = paired_bootstrap(per, con, seed) if con else {}
    res: dict[str, Any] = {"boot": boot, "ref": ref, "delta": delta, "outcome": {}}
    for arm, rkey in (
        ("psiU", "U"),
        ("psiC", "C"),
        ("psiU+", "U"),
        ("psiC+", "C"),
        ("psiU1", "U"),
        ("psiU2", "U"),
        ("psiU3", "U"),
    ):
        k = f"{arm}_minus_ref{rkey}"
        if k not in boot["all"]:
            continue
        a, n, rn = (
            boot["all"][k],
            boot["gap_2_to_M"][k],
            boot["all"][f"{arm}_minus_random"],
        )
        res["outcome"][arm] = {
            "label": outcome(
                a["point"], a["lo"], a["hi"], delta, noninf_lo=n["lo"], rand_lo=rn["lo"]
            ),
            "flags": outcome_flags(a["point"], a["lo"], a["hi"]),
        }
    for k in ("q2_psiU_minus_kind", "q2_psiC_minus_kind"):
        if k in boot["all"]:
            q = boot["all"][k]
            res["outcome"][k] = {"label": q2_outcome(q["point"], q["lo"], q["hi"], delta)}
    return res


def classification(per_seed: dict[int, dict]) -> dict:
    """§9.6 on the unshifted arms (gating at ckpt3000 gamma = 0.9), plus A1.2's
    non-gating companion class on (U⁺, C⁺)."""
    u = [per_seed[s]["outcome"]["psiU"]["label"] for s in SEEDS]
    c = [per_seed[s]["outcome"]["psiC"]["label"] for s in SEEDS]
    row, cls = classify(u, c)
    out = {"row": row, "class": cls, "psiU": u, "psiC": c}
    if all(
        "psiU+" in per_seed[s]["outcome"] and "psiC+" in per_seed[s]["outcome"]
        for s in SEEDS
    ):
        up = [per_seed[s]["outcome"]["psiU+"]["label"] for s in SEEDS]
        cp = [per_seed[s]["outcome"]["psiC+"]["label"] for s in SEEDS]
        out["companion"] = {
            "class": classify(up, cp)[1],
            "psiU+": up,
            "psiC+": cp,
            "label": "non-gating, companion",
        }
    return out


def classify_all(per: dict[int, dict[int, dict]]) -> dict:
    """Build review F6: every summary is classified. §9.6: the headline is
    ckpt3000 gamma = 0.9 Tier 1 (gating); ckpt2500 is reported, never gating, and a
    class that differs from the headline's is flagged **moving**. A1.2: the
    (U⁺, C⁺) companion at the headline (its arms are Tier 2), non-gating. §9.8:
    gamma = 0's class per checkpoint, labelled secondary."""
    by_key = {
        f"ckpt{c}.{key}": classification({s: per[c][s][key] for s in SEEDS})
        for c in per
        for key in per[c][SEEDS[0]]
    }
    head = dict(by_key[f"ckpt{HEADLINE}.0.9.tier1"], gating=True)
    out: dict[str, Any] = {
        "headline": head,
        "companion": by_key.get(f"ckpt{HEADLINE}.0.9.tier2", {}).get("companion"),
        "gamma0_secondary": {
            c: dict(by_key[f"ckpt{c}.0.0.tier2"], label="secondary")
            for c in per
            if f"ckpt{c}.0.0.tier2" in by_key
        },
        "by_key": by_key,
    }
    others = [c for c in per if c != HEADLINE]
    for c in others:
        out[f"ckpt{c}"] = dict(
            by_key[f"ckpt{c}.0.9.tier2"], label="reported, never gating"
        )
    out["moving"] = any(out[f"ckpt{c}"]["class"] != head["class"] for c in others)
    return out


def merge_counts(*runs: dict) -> dict:
    """Union of per-arm counts from runs over the SAME documents (paired)."""
    out: dict = {}
    for r in runs:
        for a, c in r.items():
            out.setdefault(a, c)
    return out


def summarise_eval(
    eval_out: dict, decisions: dict, seed: int, label: int, n_tier2: int
) -> dict:
    """Outcomes per gamma on the documents each arm ran on. Tier 1 contrasts use N_E
    documents; everything that involves a Tier 2 arm uses the first n_tier2."""
    s: dict[str, Any] = {}
    delta = decisions["delta"]
    for g in GAMMAS:
        ref = {"U": decisions["ref"][f"U@{g}"], "C": decisions["ref"][f"C@{g}"]}
        if label == HEADLINE and g == 0.9:
            t1 = eval_out["0.9.tier1"]["counts"]
            s[f"{g}.tier1"] = outcomes_for(t1, ref, delta[g], seed)
            cut = {a: {b: v[:n_tier2] for b, v in c.items()} for a, c in t1.items()}
            s[f"{g}.tier2"] = outcomes_for(
                merge_counts(eval_out["0.9.tier2"]["counts"], cut), ref, delta[g], seed
            )
        elif g == 0.9:
            s[f"{g}.tier2"] = outcomes_for(
                eval_out["0.9.tier2"]["counts"], ref, delta[g], seed
            )
        else:
            base = (
                eval_out["0.9.tier1"]["counts"]
                if label == HEADLINE
                else eval_out["0.9.tier2"]["counts"]
            )
            shared = {
                a: {b: v[:n_tier2] for b, v in base[a].items()}
                for a in ("fifo", *RANDOMS)
            }
            s[f"{g}.tier2"] = outcomes_for(
                merge_counts(eval_out["0.0.tier2"]["counts"], shared), ref, delta[g], seed
            )
    return s


def eviction_age_histogram(log: list[dict], S: int) -> list[int]:
    """§8.3: the victim's age, 1 .. S-1."""
    h = [0] * (S - 1)
    for rec in log:
        h[rec["age"] - 1] += 1
    return h


def psi_age_corr_in_loop(log: list[dict]) -> dict:
    """§8.2 in the arm's own EVAL rollout: over live slots at full-memory steps."""
    return PsiAgeAcc().add(log).result()


def attribution(log: list[dict], S: int) -> dict:
    """Truth-table row 1's attribution source (§4): victim age / rank, ψ̂ margin,
    rank shift and kind / status of the victims."""
    return AttributionAcc(S).add(log).result()


# --------------------------------------------------------------------------- #
# phase drivers (IO). Exercised end to end only on real data; the cores above
# are unit-tested on the tiny model.
# --------------------------------------------------------------------------- #


def control_c3(model, seed: int, label: int, vmap, source: Path) -> dict:
    """§11 control 3: on P = [4096, 4160), the harness's B = 1 FIFO rollout
    reproduces the fresh-stream ledger under W10 Amendment 1's rule."""
    sets, _vm, _V = CSC.doc_sets(seed, LR.VOCAB_DOCUMENTS)
    p_docs = list(sets["heldout"])
    sym_ids = torch.tensor([vmap[s] for s in ANSWER_SYMBOLS])
    answers = []
    for d in p_docs:
        r = run_arm(
            model,
            d,
            fifo_policy(d),
            vmap=vmap,
            S=S_STEPS,
            m=M_STEPS,
            L=L_TOKENS,
            check_sum=False,
        )
        answers += [(d.doc_id, t, g, ok, nll) for t, g, ok, nll in r["answers"]]
    batched = LR.batched_readout_ok(model, p_docs, vmap, sym_ids)
    return LR.control1(answers, batched, LR.reference_rows(source), label, seed)


def phase_fit_child(label, seed, source: Path, out_dir: Path) -> int:  # pragma: no cover
    """Phase A for one (checkpoint, seed)."""
    torch.set_num_threads(int(os.environ.get("RSR_B2_THREADS", "1")))
    wall0 = time.time()
    try:
        model, vmap = load_checked(source, label, seed)
        c3 = control_c3(model, seed, label, vmap, source)
        if not c3["ok"]:
            raise ControlFailure(f"C3: {json.dumps(c3)[:400]}")
        nf = n_f(p_of(int(model.cfg.D)))
        res = fit_core(
            model,
            docs_in(seed, "FIT_TRAIN", nf),
            docs_in(seed, "FIT_VAL"),
            seed=seed,
            vmap=vmap,
            S=S_STEPS,
            m=M_STEPS,
            L=L_TOKENS,
        )
    except (ControlFailure, RangeError) as e:
        print(f"CONTROL FAILED (exit 3): {e}", file=sys.stderr)
        return 3
    except RidgeFailure as e:
        print(f"RIDGE FAILURE (exit 1): {e}", file=sys.stderr)
        return 1
    res["controls"]["C3"] = c3
    res["seconds"]["child_wall"] = time.time() - wall0
    res["peak_rss_gb"] = _peak_rss_gb()
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "heads": res["fits"]["heads"],
            **{k: v for k, v in res.items() if k != "fits"},
            "class_means": res["fits"]["class_means"],
            "age_decodability": res["fits"]["age_decodability"],
        },
        out_dir / f"ckpt{label}-seed{seed}.pt",
    )
    return 0


def phase_a_rows(label: int, seed: int, a: dict) -> list[tuple[str, Any]]:
    """Phase A's per-child ledger rows (the brief for item B2-A): every head's
    selected λ, its residual and grid-edge flag, the age decodability (with build
    review F11's i < M / i ≥ M split), corr(ψ̂, age) on FIT_VAL, ref, δ, the FIT_VAL
    accuracies, the bootstrap SD of every §9.4 contrast, threads, seconds, controls."""
    pre = f"ckpt{label}.seed{seed}"
    rows: list[tuple[str, Any]] = []
    for k, h in a["heads"].items():
        sel = h["selected"]
        rows.append(
            (
                f"{pre}.head.{k}",
                {
                    "lam": h["lam"],
                    "lam_eff": h["path"][sel]["lam_eff"],
                    "resid": h["path"][sel]["resid"],
                    "grid_edge": sel in (0, len(h["path"]) - 1),
                    "n_eligible": sum(bool(pt["eligible"]) for pt in h["path"]),
                    "resid_grid": [pt["resid"] for pt in h["path"]],
                    "val_mse_demeaned": (
                        h["val_mse_demeaned"][sel] if h["val_mse_demeaned"] else None
                    ),
                    "val_mse_raw": h["val_mse_raw"][sel],
                    "val_r2_raw": h["val_r2_raw"][sel],
                    "n_train_rows": h["n_train_rows"],
                    "n_val_rows": h["n_val_rows"],
                },
            )
        )
    for r, dec in a["age_decodability"].items():
        rows.append(
            (
                f"{pre}.age_decodability.{r}",
                {k: v for k, v in dec.items() if k != "w"},
            )
        )
    for k, v in a["psi_age_corr_fit_val"].items():
        rows.append((f"{pre}.psi_age_corr_fit_val.{k}", v))
    d = a["decisions"]
    for k in ("ref", "delta", "acc", "sigma"):
        rows.append((f"{pre}.decisions.{k}", d[k]))
    rows.append((f"{pre}.class_means", a["class_means"]))
    rows.append((f"{pre}.n", a["n"]))
    rows.append((f"{pre}.seconds", a["seconds"]))
    rows.append((f"{pre}.threads", a.get("threads")))
    rows.append((f"{pre}.peak_rss_gb", a.get("peak_rss_gb")))
    rows.append((f"{pre}.controls", a["controls"]))
    return rows


def n_e_from_phase_a(phase_a: dict[int, dict]) -> dict:
    """§9.4 on the ckpt3000 phase-A outputs, per seed at gamma = 0.9."""
    sig = {
        (c, s): phase_a[s]["decisions"]["sigma"][c]
        for s in phase_a
        for c in phase_a[s]["decisions"]["sigma"]
    }
    deltas = {s: phase_a[s]["decisions"]["delta"][0.9] for s in phase_a}
    return n_eval(sig, deltas, N_V)


def _progress(tag: str):
    """One flushed line per EVAL document in the child's log (review F8: the
    per-document cost is measured as the run goes, not projected)."""

    def f(g, tier, j, n, sec):
        what = "resumed" if sec is None else f"{sec:.2f}s"
        print(
            f"{time.strftime('%H:%M:%S')} {tag} {g}.{tier} {j}/{n} {what} "
            f"peak_rss_gb={_peak_rss_gb():.2f}",
            file=sys.stderr,
            flush=True,
        )

    return f


def write_logs_jsonl(unit_dir: Path, path: Path, label: int, docs, n_tier2: int) -> int:
    """§4's per-eviction logs of every arm, one JSON line per eviction, streamed
    from the units (never held in RAM; review F1). Atomic. -> records written."""
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    n = 0
    with gzip.open(tmp, "wt") as f:
        for (g, tier), names in tier_plan(label).items():
            ds = docs if tier == "tier1" else docs[:n_tier2]
            for d in ds:
                u = load_unit(unit_path(unit_dir, g, tier, d.doc_id))
                for a in names:
                    for rec in u["logs"][a]:
                        f.write(json.dumps({"gamma": g, "tier": tier, **rec}) + "\n")
                        n += 1
    os.replace(tmp, path)
    return n


def _atomic_torch_save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def phase_eval_child(
    label, seed, source: Path, fits_dir: Path, out_dir: Path, n_e: int, n_tier2: int
) -> int:  # pragma: no cover
    """Phase B for one (checkpoint, seed). Units stream to
    `out_dir/ckpt{c}-seed{s}/units/`; a rerun resumes from them."""
    torch.set_num_threads(int(os.environ.get("RSR_B2_THREADS", "1")))
    wall0 = time.time()
    tag = f"ckpt{label}-seed{seed}"
    try:
        model, vmap = load_checked(source, label, seed)
        fpath = fits_dir / f"{tag}.pt"
        fits = torch.load(fpath)
        n = n_e if label == HEADLINE else n_tier2
        docs = docs_in(seed, "EVAL", n)
        clos = vocabulary_closure(docs, vmap)
        if not clos["ok"]:
            raise ControlFailure(f"closure {clos}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out = eval_core(
            model,
            docs,
            fits,
            seed=seed,
            label=label,
            vmap=vmap,
            S=S_STEPS,
            m=M_STEPS,
            L=L_TOKENS,
            n_tier2=n_tier2,
            out_dir=out_dir / tag,
            fits_id=LR.sha256(fpath),
            progress=_progress(tag),
        )
    except (ControlFailure, RangeError) as e:
        print(f"CONTROL FAILED (exit 3): {e}", file=sys.stderr)
        return 3
    _atomic_torch_save(e0h_payload(out), out_dir / f"{tag}.e0h.pt")
    summ = summarise_eval(out, fits["decisions"], seed, label, n_tier2)
    n_logs = write_logs_jsonl(
        out_dir / tag, out_dir / f"{tag}.logs.jsonl.gz", label, docs, n_tier2
    )
    _atomic_torch_save(
        {
            "summary": summ,
            "attribution": {
                f"{k}.{a}": v for k, r in out.items() for a, v in r["attribution"].items()
            },
            "psi_age_corr": {
                f"{k}.{a}": v
                for k, r in out.items()
                for a, v in r["psi_age_corr"].items()
            },
            "counts": {k: r["counts"] for k, r in out.items()},
            "controls": {
                k: {kk: vv for kk, vv in r["controls"].items() if kk != "doc_ids"}
                for k, r in out.items()
            },
            "e0h_source": e0h_payload(out)["regressor_source"],
            "n_log_records": n_logs,
            "n_docs": {k: r["n_docs"] for k, r in out.items()},
            "n_resumed": {k: r["n_resumed"] for k, r in out.items()},
            "seconds": {
                **{k: r["seconds"] for k, r in out.items()},
                "child_wall": time.time() - wall0,
            },
            "threads": torch.get_num_threads(),
            "peak_rss_gb": _peak_rss_gb(),
        },
        out_dir / f"{tag}.pt",
    )
    return 0


# --------------------------------------------------------------------------- #
# claims, CLI
# --------------------------------------------------------------------------- #


def write_claims(path: Path, claims: list[dict]) -> Path:
    """`runs/<id>/claims.json`: one `{claim, command, expected}` per checkable
    claim, each command runnable from a clean checkout (the verifier's input)."""
    for c in claims:
        missing = [
            k for k in ("claim", "command", "expected") if not isinstance(c.get(k), str)
        ]
        if missing:
            raise ValueError(f"claim {c!r} is missing {missing}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(claims, indent=2) + "\n")
    return path


#: TBD-2 (A1.4 item 7): the measured seconds per Tier 1 arm-document, written
#: here only after the PREREG diff that records it. `None` makes phase B refuse.
SEC_PER_ARM_DOC: float | None = None
FIT_RUN_ID = "b2-psi-probe-fit"
#: Measured by the sizing run (runs/b2-psi-probe-sizing, gram.peak_rss_gb = 9.6 GB
#: for one Gram + eigh at p = 16,640): six concurrent phase-A children would need
#: ~58 GB of 64. Three is the cap.
FIT_MAX_PARALLEL = 3


def _children(argvs: dict, root: Path, parallel: int, led) -> dict:  # pragma: no cover
    rcs, running = {}, {}
    (root / "logs").mkdir(parents=True, exist_ok=True)
    jobs = list(argvs)
    while jobs or running:
        while jobs and len(running) < parallel:
            j = jobs.pop(0)
            log = open(root / "logs" / f"{j}.log", "a")  # noqa: SIM115 (a rerun appends)
            running[j] = subprocess.Popen(
                argvs[j], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT
            )
        for j, proc in list(running.items()):
            rc = proc.poll()
            if rc is not None:
                rcs[j] = rc
                led.command(argvs[j], exit_code=rc)
                del running[j]
        time.sleep(2)
    return rcs


def parent_rc(rcs: dict, *, t0_ok: bool) -> Exit:
    """A child's 1 (a measurement defect) is the parent's 1; any other failure,
    or T0 failing, is DID NOT RUN (PREREG §12; review F5: rc 1 was mapped to 3)."""
    bad = {k: v for k, v in rcs.items() if v != 0}
    if any(v == 1 for v in bad.values()):
        return Exit.FAIL
    if bad or not t0_ok:
        return Exit.DID_NOT_RUN
    return Exit.OK


def _child_argv(cmd: str, c: int, s: int, source: Path, out: Path, extra=()) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        cmd,
        "--checkpoint",
        str(c),
        "--seed",
        str(s),
        "--source",
        str(source),
        "--out",
        str(out),
        *extra,
    ]


def run_fit(source: Path, runs_root: Path, parallel: int) -> Exit:  # pragma: no cover
    """Phase A (parent): preflight, T0, the six children, N_E by §9.4."""
    from ledger import Ledger

    probs = preflight()
    if probs:
        return did_not_run("; ".join(probs))
    root = runs_root / FIT_RUN_ID
    led = Ledger(
        FIT_RUN_ID,
        question=QUESTION + " [phase A: capture, fits, FIT_VAL]",
        runs_root=runs_root,
    )
    led.run_meta(device="cpu", steps_requested=0, steps_done=0)
    led.manifest(
        {
            "run_id": FIT_RUN_ID,
            "prereg": PREREG,
            "prereg_commits": PREREG_COMMITS,
            "ranges": RANGES,
            "used_ranges": USED_RANGES,
            "lambdas": LAMBDAS,
            "seeds": SEEDS,
            "checkpoints": list(CHECKPOINTS),
            "source": str(source),
            "expected": EXPECTED,
            "falsifier": FALSIFIER,
        }
    )
    t0 = t0_checked()
    led.note("T0.start", t0, how="run.py::t0_verify")
    if not t0["ok"]:
        led.status("failed")
        led.write()
        return did_not_run(f"T0: {t0}")
    argvs = {
        f"ckpt{c}-seed{s}": _child_argv("fit-child", c, s, source, root / "phaseA")
        for c in CHECKPOINTS
        for s in SEEDS
    }
    rcs = _children(argvs, root, min(parallel, FIT_MAX_PARALLEL), led)
    t0e = t0_checked()
    led.note("T0.end", t0e, how="run.py::t0_verify")
    rc = parent_rc(rcs, t0_ok=t0e["ok"])
    if rc == Exit.OK:
        a3 = {
            s: torch.load(root / "phaseA" / f"ckpt{HEADLINE}-seed{s}.pt") for s in SEEDS
        }
        ne = n_e_from_phase_a(a3)
        led.note("N_E.power", ne, how="run.py::n_eval (§9.4, A1.12)")
        for c in CHECKPOINTS:
            for s in SEEDS:
                a = torch.load(root / "phaseA" / f"ckpt{c}-seed{s}.pt")
                for k, v in phase_a_rows(c, s, a):
                    led.note(k, v, how=f"phaseA/ckpt{c}-seed{s}.pt")
        for s, v in a3.items():
            led.note(
                f"seed{s}.decisions",
                {k: v["decisions"][k] for k in ("ref", "delta", "acc")},
                how="run.py::val_decisions on FIT_VAL",
            )
    led.run_meta(seeds_actually_run=SEEDS if rc == Exit.OK else [])
    led.status("ok" if rc == Exit.OK else "failed")
    led.verdict(
        falsifier=FALSIFIER,
        outcome="inconclusive",
        detail=f"phase A only; children {rcs}",
    )
    led.command([sys.executable, EXPERIMENT, "fit"], exit_code=int(rc), note="the parent")
    led.write()
    return rc


def run_eval(source: Path, runs_root: Path, parallel: int) -> Exit:  # pragma: no cover
    """Phase B (parent): refuses unless N_E (TBD-3) and the TBD-2 cost are set and
    N_E is what phase A's files give (review F10); applies A1.4's ceiling; runs
    the children (each resumes from its units, so relaunching this command after a
    kill loses no completed document); classifies every summary (review F6)."""
    from ledger import Ledger

    if N_E is None or SEC_PER_ARM_DOC is None:
        return did_not_run("N_E (TBD-3) or the TBD-2 cost is not in the PREREG yet")
    fits = runs_root / FIT_RUN_ID / "phaseA"
    try:
        a3 = {s: torch.load(fits / f"ckpt{HEADLINE}-seed{s}.pt") for s in SEEDS}
    except OSError as e:
        return did_not_run(f"phase-A fits unreadable: {e}")
    ne_a = n_e_from_phase_a(a3)
    del a3
    if ne_a["N_E"] != N_E:
        return did_not_run(
            f"N_E = {N_E} in run.py, but phase A's files give {ne_a['N_E']} (§9.4)"
        )
    ceil1 = compute_ceiling(
        N_E,
        sec_per_arm_doc=SEC_PER_ARM_DOC,
        arms=len(TIER1),
        seeds=len(SEEDS),
        parallel=min(parallel, len(SEEDS)),  # one Tier 1 child per ckpt3000 seed
    )
    n_e = ceil1["N_E"]
    n2 = min(n_e, TIER2_MAX_DOCS)
    tier2_arms = (
        len(TIER2_G09) + len(TIER2_G0) + len(TIER1) + len(TIER2_G09) + len(TIER2_G0)
    )
    ceil2 = compute_ceiling(
        n2,
        sec_per_arm_doc=SEC_PER_ARM_DOC,
        arms=tier2_arms,
        seeds=len(SEEDS),
        parallel=parallel,
        budget_h=TIER2_BUDGET_H,
    )
    n2 = ceil2["N_E"]
    root = runs_root / RUN_ID
    prev = root / "manifest.json"
    resumed_from = None
    if prev.exists():  # a relaunch: the earlier manifest is kept, never overwritten
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        resumed_from = root / f"manifest.before-{stamp}.json"
        resumed_from.write_bytes(prev.read_bytes())
    led = Ledger(RUN_ID, question=QUESTION, runs_root=runs_root)
    led.run_meta(device="cpu", steps_requested=0, steps_done=0)
    led.manifest(
        {
            "run_id": RUN_ID,
            "prereg": PREREG,
            "prereg_commits": PREREG_COMMITS,
            "N_E_prereg": N_E,
            "N_E_run": n_e,
            "N_E_phase_a": ne_a,
            "N_tier2": n2,
            "ceiling": [ceil1, ceil2],
            "seeds": SEEDS,
            "checkpoints": list(CHECKPOINTS),
            "source": str(source),
            "fits": str(fits),
            "threads_per_child": os.environ.get("RSR_B2_THREADS", "1"),
            "parallel": parallel,
            "resume": "units under phaseB/ckpt{c}-seed{s}/units are read, not rerun, "
            "when their key (format, fits sha256, run.py sha256, arms, gamma, tier, "
            "document) matches; any other key is exit 3",
            "expected": EXPECTED,
            "falsifier": FALSIFIER,
        }
    )
    if resumed_from is not None:
        led.note("resumed", str(resumed_from), how="run.py::run_eval (relaunch)")
    probs = preflight()
    t0 = t0_checked()
    led.note("T0.start", t0, how="run.py::t0_verify")
    if probs or not t0["ok"]:
        led.status("failed")
        led.write()
        return did_not_run(f"preflight {probs} or T0 {t0}")
    argvs = {
        f"ckpt{c}-seed{s}": _child_argv(
            "eval-child",
            c,
            s,
            source,
            root / "phaseB",
            ("--fits", str(fits), "--n-e", str(n_e), "--n-tier2", str(n2)),
        )
        for c in CHECKPOINTS
        for s in SEEDS
    }
    rcs = _children(argvs, root, parallel, led)
    t0e = t0_checked()
    led.note("T0.end", t0e, how="run.py::t0_verify")
    rc = parent_rc(rcs, t0_ok=t0e["ok"])
    if rc != Exit.OK:
        led.run_meta(seeds_actually_run=[])
        led.status("failed")
        led.verdict(falsifier=FALSIFIER, outcome="inconclusive", detail=f"children {rcs}")
        led.write()
        return rc
    per = {
        c: {s: torch.load(root / "phaseB" / f"ckpt{c}-seed{s}.pt") for s in SEEDS}
        for c in CHECKPOINTS
    }
    cls = classify_all({c: {s: per[c][s]["summary"] for s in SEEDS} for c in CHECKPOINTS})
    head = cls["headline"]
    head["underpowered"] = ceil1["reduced"] or N_E >= N_E_MAX
    led.note(
        "classification", head, how="run.py::classify_all (§9.6), ckpt3000 gamma=0.9"
    )
    for k in ("companion", "moving", "gamma0_secondary", "by_key"):
        led.note(f"classification.{k}", cls[k], how="run.py::classify_all")
    for c in CHECKPOINTS:
        if c != HEADLINE:
            led.note(
                f"classification.ckpt{c}", cls[f"ckpt{c}"], how="run.py::classify_all"
            )
        for key in per[c][SEEDS[0]]["summary"]:
            led.note(
                f"ckpt{c}.{key}",
                {s: per[c][s]["summary"][key]["outcome"] for s in SEEDS},
                how="run.py::summarise_eval",
            )
        for s in SEEDS:
            led.note(
                f"ckpt{c}.seed{s}.cost",
                {
                    k: per[c][s][k]
                    for k in ("seconds", "peak_rss_gb", "threads", "n_resumed")
                },
                how=f"phaseB/ckpt{c}-seed{s}.pt",
            )
    led.run_meta(seeds_actually_run=SEEDS)
    led.status("ok")
    led.verdict(
        falsifier=FALSIFIER,
        outcome="inconclusive",
        detail=f"B2 class {head['class']} (recommendation to Brendan; not a spec kill)",
    )
    led.command(
        [sys.executable, EXPERIMENT, "eval"],
        exit_code=0,
        note="the parent; E0h is separate",
    )
    led.write()
    return Exit.OK


def main(argv: list[str] | None = None) -> Exit:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["e0h"]:  # build review F2: every e0h argument reaches e0h_main
        return status(e0h_main(argv[1:]))
    ap = ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "cmd",
        choices=[
            "check",
            "size",
            "size-child",
            "fit",
            "fit-child",
            "eval",
            "eval-child",
            "e0h",
        ],
    )
    ap.add_argument("--checkpoint", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--source", type=Path, default=None)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--fits", type=Path)
    ap.add_argument("--n-e", type=int)
    ap.add_argument("--n-tier2", type=int)
    ap.add_argument("--n-docs", type=int, default=32)
    ap.add_argument("--n-arm-docs", type=int, default=8)
    ap.add_argument("--gram-docs", type=int, default=16)
    ap.add_argument("--gram-threads", type=int, default=8)
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    a, rest = ap.parse_known_args(argv)
    if rest:
        return did_not_run(f"unknown arguments {rest}")
    if a.cmd == "check":
        probs = preflight()
        print(json.dumps({"problems": probs}))
        return did_not_run("; ".join(probs)) if probs else Exit.OK
    if a.cmd in ("eval", "eval-child") and N_E is None:
        return did_not_run(
            "N_E is TBD-3: §9.4 item 6 writes it into the PREREG by diff before any "
            "EVAL document is run"
        )
    source = a.source or LR.default_source()
    if a.cmd == "size-child":
        return status(
            size_child(a.checkpoint, a.seed, source, a.out, a.n_docs, a.n_arm_docs)
        )
    if a.cmd == "size":
        return status(
            run_size(
                source,
                a.runs_root / SIZING_RUN_ID,
                a.n_docs,
                a.n_arm_docs,
                a.gram_docs,
                a.gram_threads,
                a.parallel,
            )
        )
    if a.cmd == "fit-child":
        return status(phase_fit_child(a.checkpoint, a.seed, source, a.out))
    if a.cmd == "fit":
        return status(run_fit(source, a.runs_root, a.parallel))
    if a.cmd == "eval-child":
        return status(
            phase_eval_child(
                a.checkpoint, a.seed, source, a.fits, a.out, a.n_e, a.n_tier2
            )
        )
    return status(run_eval(source, a.runs_root, a.parallel))


if __name__ == "__main__":
    run_main(main)
