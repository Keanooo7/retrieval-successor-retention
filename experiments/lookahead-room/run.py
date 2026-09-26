"""Lookahead room (loop item W10) -- can 3b (RSR gamma=0.9 vs gamma=0) discriminate here?

Pre-registration: `experiments/lookahead-room/PREREG.md`, committed alone ahead of this
file. Read-only: it loads frozen fresh-stream arm B checkpoints and **trains nothing**.
It never calls `rsr.constants.record()`.

Spec anchors: §3.2.1 (`r_i`, gated, fill-rescaled -- `retrieval_demand(gated=True)`,
exactly as `experiments/e0e/run.py: r_of`), §3.4 and correction 2 (the discounted MC
return `G_i(t) = sum_k gamma^k r_i(t + k)` within the stream).

The instrument (PREREG *Definitions*):

* each document runs **alone** (B = 1) through the real
  `rsr.model.tg.policy_loop.run_policy_loop` under FIFO, eval mode, ``no_grad``,
  ``observe=True``; `r_i(t)` is recorded for every resident sentence;
* **demand-if-resident** ``D[t][i]``: for a sentence FIFO has already evicted, a
  *probe* -- the same step's forward on FIFO's memory with the rank-0 slot's gestalt
  replaced by sentence ``i``'s gestalt -- read at slot 0. The probe forward is a side
  computation: its inputs are captured by a forward pre-hook on the real call and it
  never feeds back into the rollout. Row 0 of every probe batch is the *identity
  probe* (control C4);
* target rules (``T0 = D``, ``G_0.9``, ``G_0.97``, ``Tnext``) evaluated by model-free
  residency through `rsr.metrics.headroom.simulate`;
* ``online_g0``: the gamma = 0 rule with the model in the loop (causal, exact).

🔴 **Caveat** (PREREG): the targets are from a FIFO world. The rule hit rates are an
upper-bound-style proxy of what each target could teach, not RSR's result, and the
hindsight is asymmetric in favour of gamma > 0.

Usage::

    uv run python experiments/lookahead-room/run.py --dry-run
    uv run python experiments/lookahead-room/run.py          # the run
    uv run python experiments/lookahead-room/run.py --render-results

Exit codes (`rsr.exit_codes`): 0 a classification was reached (PARTIAL included) ·
3 inconclusive / did not run.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rsr.baselines.fifo import FIFOPolicy  # noqa: E402
from rsr.baselines.oracle import OraclePolicy  # noqa: E402
from rsr.data.synthetic import ANSWER_SYMBOLS, discounted_demand  # noqa: E402
from rsr.exit_codes import ArgumentParser, Exit, run_main, status  # noqa: E402
from rsr.metrics.headroom import simulate  # noqa: E402
from rsr.model.tg.policy_loop import (  # noqa: E402
    cross_capture,
    run_policy_loop,
    trace_for_row,
)
from rsr.retention.reward import retrieval_demand  # noqa: E402
from rsr.train.loop import answer_targets, build_vocab, encode  # noqa: E402

EXPERIMENT = "experiments/lookahead-room/run.py"
RUN_ID = "lookahead-room"
PREREG = "experiments/lookahead-room/PREREG.md"
PREREG_COMMIT = "527daf4"


def _load(name: str, rel: str):
    """Import an experiment script by path. Imported, never copied."""
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CSC = _load("_lookahead_csc", "experiments/corpus-size-curve/run.py")
S003 = CSC.S003

# --------------------------------------------------------------------------- #
# 🔒 PREREG.md, transcribed. Changing any is changing the PREREG.
# --------------------------------------------------------------------------- #

SEEDS = [0, 1, 2]
CHECKPOINTS = (2500, 3000)
HEADLINE = 3000
CKPT_SHA256 = {
    (2500, 0): "26c4162e7dc6ef1f80dffa451e4ee75c56360bc94227533b38a45b62003e3176",
    (2500, 1): "327c3facde61ef9a7f4b9fcaeb31efcf40ab36d65a5d584ec97637db334bdadd",
    (2500, 2): "74b026f1dd3fe59eb473e501c37ed9bb7d836276270ec875939e5654d8e86ddd",
    (3000, 0): "0ee3f8a69b507d927c631eb85116e1cd9739ee4472c44eb7f145d739a118da60",
    (3000, 1): "dadd1e08a3849c1211c8479394df4060f91cd2a6e188b97701b783f2068a3da8",
    (3000, 2): "b507ebc573a54316f9d3c20337f49e664bb9c414388af925c5a8e87efd6b63b8",
}
M = S003.CONFIG["memory_slots"]
S = S003.CONFIG["steps_per_stream"]
L = S003.CONFIG["max_tokens"]
PROBE = CSC.PROBE  # (0, 64): vocabulary documents
P_SET = CSC.HELDOUT  # (4096, 4160)
E_SET = (64, 1088)
STREAM_B = (4160, 4160 + 16 * 3000)  # fresh-stream arms A/B
STREAM_ESCAPE = (4160 + 16 * 3000, 4160 + 16 * 9000)  # fresh-escape
VOCAB_DOCUMENTS = 64
GAMMAS = {"g09": 0.9, "g097": 0.97}
ORACLE_GAMMA = 0.97  # E-feas's; Belady MIN for every gamma in (0, 1)
RULES = ("rule_g0", "rule_g09", "rule_g097", "rule_next")
LITERAL = ("lit_g0", "lit_g09", "lit_g097")
REFS = ("fifo", "oracle", "pending_fifo", "factfiller")
ONLINE = ("online_g0",)
ALL_POLICIES = REFS + RULES + LITERAL + ONLINE
TESTABLE_MIN = 0.05
NO_ROOM_MAX = 0.02
N_BOOT = 2000
BOOT_SEED_BASE = 20260926
SUM_TOL = 1e-5  # E0e's
IDENTITY_TOL = 1e-6
ACC_TOL = 1e-12
NLL_TOL = 1e-4
C2_ORACLE = {0: 0.1852, 1: 0.1739, 2: 0.1896}
C2_KINDRAND = {0: 0.1498, 1: 0.1403, 2: 0.1577}
K_MAX_DISPLAY = 16  # pending k > 16 pooled for display
CLASSES = ("pending", "querying", "answered", "filler")

QUESTION = (
    "Can falsifier 3b (RSR at gamma=0.9 vs gamma=0) discriminate on the S0-03 corpus "
    "at M=16, i.e. does a gamma=0.9 demand target buy retention that the gamma=0 "
    "target cannot, on frozen fresh-stream arm B?"
)
FALSIFIER = (
    "none of spec section 2 directly; a D1 input. Claim tested: 3b has room here "
    "(room_3b = hit(rule_g09) - hit(rule_g0) >= 0.05 with CI lo > 0 every seed). "
    "TESTABLE_HERE -> survived, NO_ROOM -> falsified, PARTIAL/INCONCLUSIVE -> "
    "inconclusive."
)
EXPECTED = (
    "Pre-registered: pending-vs-filler AUC T0 ~0.6 (resident), G_0.9 ~0.75; rule_g0 "
    "near FIFO/random (0.79-0.82), rule_g09 ~0.90; headline ckpt3000 TESTABLE_HERE "
    "~45%, PARTIAL ~30%, NO_ROOM ~25%; literal variants ~FIFO; online_g0 within "
    "+/-0.03 of rule_g0."
)

OUTCOME = {
    "TESTABLE_HERE": "survived",
    "NO_ROOM": "falsified",
    "PARTIAL": "inconclusive",
    "INCONCLUSIVE": "inconclusive",
}


def default_source() -> Path:
    """`<main checkout>/.worktrees/fresh-stream/runs/fresh-stream`. Read-only."""
    env = os.environ.get("RSR_FRESH_STREAM_RUNS")
    if env:
        return Path(env)
    git = "/opt/homebrew/bin/git" if Path("/opt/homebrew/bin/git").exists() else "git"
    out = subprocess.run(
        [git, "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return Path(out).parent / ".worktrees" / "fresh-stream" / "runs" / "fresh-stream"


def ckpt_path(source: Path, label: int, seed: int) -> Path:
    return source / "B" / f"seed{seed}" / f"ckpt-{label:06d}.pt"


def sha256(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# documents (C6) -- W6's construction (retention-readability), re-stated here
# --------------------------------------------------------------------------- #


def disjointness(e_set: tuple[int, int] = E_SET) -> dict:
    others = {
        "probe": PROBE,
        "heldout_P": P_SET,
        "stream_fresh_stream": STREAM_B,
        "stream_fresh_escape": STREAM_ESCAPE,
    }
    overlaps = {
        k: [max(e_set[0], lo), min(e_set[1], hi)]
        for k, (lo, hi) in others.items()
        if max(e_set[0], lo) < min(e_set[1], hi)
    }
    return {"ok": not overlaps, "E": list(e_set), "overlaps": overlaps}


def documents(seed: int, e_set: tuple[int, int] = E_SET):
    """-> (U docs in order E then P, vmap, V, closure). The vocabulary map is
    `CSC.doc_sets(seed, 64)`'s, the one arm B was trained with."""
    sets, vmap, V = CSC.doc_sets(seed, VOCAB_DOCUMENTS)
    e_docs = CSC._generate(seed, e_set[1])[e_set[0] : e_set[1]]
    missing = sorted(set(build_vocab(e_docs)) - set(vmap))
    closure = {"ok": not missing, "missing": missing[:20], "n_missing": len(missing)}
    return tuple(e_docs) + tuple(sets["heldout"]), vmap, V, closure


# --------------------------------------------------------------------------- #
# the pure pieces (tests/test_lookahead_room.py drives each on hand-built inputs)
# --------------------------------------------------------------------------- #


def r_of(trace, n_live: int, m: int) -> torch.Tensor:
    """`[M]` §3.2.1's target, as E0e: norm-weighted, gated, fill-rescaled."""
    return retrieval_demand(trace, n_live=n_live, capacity=m, gated=True)


def slot_class(doc, i: int, t: int) -> tuple[str, int | None]:
    """PREREG *Definitions*: the class of sentence ``i`` at step ``t`` and, for an
    assert, ``k = q - t`` (steps until its query)."""
    q_of = {a: q for a, q in doc.pairs}
    if i not in q_of:
        return "filler", None
    k = q_of[i] - t
    if k > 0:
        return "pending", k
    if k == 0:
        return "querying", 0
    return "answered", k


def discounted_returns(D: torch.Tensor, gamma: float) -> torch.Tensor:
    """``G[t][i] = sum_{k=0}^{S-1-t} gamma^k D[t+k][i]`` within the stream.
    ``D`` is ``[S, S]`` step-major; entries with ``i >= t`` are NaN and are carried
    as NaN (never read by a rule: only sentences ``i < t`` are in memory at ``t``)."""
    n = D.shape[0]
    G = torch.full_like(D, float("nan"))
    acc = torch.zeros(D.shape[1], dtype=D.dtype)
    for t in range(n - 1, -1, -1):
        row = torch.nan_to_num(D[t], nan=0.0)
        acc = row + gamma * acc
        G[t] = torch.where(torch.isnan(D[t]), torch.full_like(acc, float("nan")), acc)
    return G


def next_step(D: torch.Tensor) -> torch.Tensor:
    """``Tnext[t][i] = D[t+1][i]``, 0 at the last step (descriptive)."""
    T = torch.zeros_like(D)
    T[:-1] = torch.nan_to_num(D[1:], nan=0.0)
    return torch.where(torch.isnan(D), torch.full_like(D, float("nan")), T)


class TargetPolicy:
    """Evict the live slot with the smallest ``target[step][written_at]``; ties to
    the lowest slot index (the oldest), as `OraclePolicy`. Uses the target AT the
    decision step (the gamma = 0 target is observed in that step's forward)."""

    name = "target"

    def __init__(self, target: torch.Tensor) -> None:
        self.target = target.tolist()

    def select_eviction(self, slots, context, step: int) -> int:
        best, best_v = -1, float("inf")
        for k in range(len(slots.live)):
            if not bool(slots.live[k]):
                continue
            v = self.target[step][int(slots.written_at[k])]
            if v != v:
                raise ValueError(f"step {step}: NaN target for a live sentence")
            if v < best_v:
                best, best_v = k, v
        return best

    def observe(self, slots, attn, step: int) -> None:
        return None

    def on_write(self, slots, slot: int, step: int) -> None:
        return None

    def reset(self) -> None:
        return None


class PendingFIFO:
    """The red team's causal rule: evict the oldest slot that is not a pending
    (asserted, query still ahead) fact. Knows the query schedule -- a reference."""

    name = "pending_fifo"

    def __init__(self, doc) -> None:
        self.kind = [s.kind for s in doc.sentences]
        self.q = {a: q for a, q in doc.pairs}

    def select_eviction(self, slots, context, step: int) -> int:
        for k in range(len(slots.live)):
            i = int(slots.written_at[k])
            if not (self.kind[i] == "assert" and self.q[i] > step):
                return k
        return 0

    def observe(self, slots, attn, step: int) -> None:
        return None

    def on_write(self, slots, slot: int, step: int) -> None:
        return None

    def reset(self) -> None:
        return None


class FactFiller:
    """Random non-assert victim (any live slot if all are asserts). W6's rule; the
    ``kindrand.py`` convention when handed a shared ``random.Random(seed)``."""

    name = "factfiller"

    def __init__(self, doc, rng: random.Random) -> None:
        self.kinds = [s.kind for s in doc.sentences]
        self.rng = rng

    def select_eviction(self, slots, context, step: int) -> int:
        live = [k for k in range(len(slots.live)) if bool(slots.live[k])]
        non = [k for k in live if self.kinds[int(slots.written_at[k])] != "assert"]
        return self.rng.choice(non if non else live)

    def observe(self, slots, attn, step: int) -> None:
        return None

    def on_write(self, slots, slot: int, step: int) -> None:
        return None

    def reset(self) -> None:
        return None


class Scripted:
    """Replays a victim sequence ``{step: written_at of the victim}`` (C5)."""

    name = "scripted"

    def __init__(self, victims: dict[int, int]) -> None:
        self.victims = victims

    def select_eviction(self, slots, context, step: int) -> int:
        w = self.victims[step]
        ks = [k for k in range(len(slots.live)) if int(slots.written_at[k]) == w]
        if len(ks) != 1:
            raise AssertionError(f"step {step}: victim {w} not resident exactly once")
        return ks[0]

    def observe(self, slots, attn, step: int) -> None:
        return None

    def on_write(self, slots, slot: int, step: int) -> None:
        return None

    def reset(self) -> None:
        return None


def auc(pos: list[float], neg: list[float]) -> float | None:
    """ROC-AUC of ranking ``pos`` above ``neg``; ties count 0.5. None if either empty."""
    if not pos or not neg:
        return None
    s = 0.0
    for p in pos:
        for n in neg:
            s += 1.0 if p > n else (0.5 if p == n else 0.0)
    return s / (len(pos) * len(neg))


def summary(xs: list[float]) -> dict:
    """n, mean, sd (sample), q10 / q50 / q90."""
    if not xs:
        return {"n": 0, "mean": None, "sd": None, "q10": None, "q50": None, "q90": None}
    t = torch.tensor(xs, dtype=torch.float64)
    q = torch.quantile(t, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64))
    return {
        "n": len(xs),
        "mean": float(t.mean()),
        "sd": float(t.std()) if len(xs) > 1 else None,
        "q10": float(q[0]),
        "q50": float(q[1]),
        "q90": float(q[2]),
    }


def classify(per_seed: dict[int, dict]) -> str:
    """🔒 PREREG decision rule over ``{seed: {"point", "lo", "hi"}}`` of room_3b."""
    if all(v["point"] >= TESTABLE_MIN and v["lo"] > 0 for v in per_seed.values()):
        return "TESTABLE_HERE"
    if all(v["hi"] < NO_ROOM_MAX for v in per_seed.values()):
        return "NO_ROOM"
    return "PARTIAL"


def paired_bootstrap(
    per_doc: dict[str, list[tuple[int, int]]],
    diffs: dict[str, tuple[str, str]],
    seed: int,
    n_boot: int = N_BOOT,
) -> dict[str, dict]:
    """``per_doc[policy] = [(n_queries, hits), ...]`` in one document order.
    ``diffs[name] = (a, b)`` -> hit(a) - hit(b), with hit = sum hits / sum n. One
    resampled document multiset per replicate, shared by every policy (paired)."""
    pols = list(per_doc)
    D = len(per_doc[pols[0]])
    X = {p: torch.tensor(per_doc[p], dtype=torch.float64).reshape(D, 2) for p in pols}
    g = torch.Generator().manual_seed(BOOT_SEED_BASE + seed)
    idx = torch.randint(0, D, (n_boot, D), generator=g)
    W = torch.zeros(n_boot, D, dtype=torch.float64).scatter_add_(
        1, idx, torch.ones(n_boot, D, dtype=torch.float64)
    )
    rep = {p: W @ x for p, x in X.items()}  # [R, 2]
    pt = {p: x.sum(0) for p, x in X.items()}

    def hit(v):
        return v[..., 1] / v[..., 0]

    out = {}
    for name, (a, b) in diffs.items():
        r = hit(rep[a]) - hit(rep[b])
        out[name] = {
            "point": float(hit(pt[a]) - hit(pt[b])),
            "lo": float(torch.quantile(r, 0.025)),
            "hi": float(torch.quantile(r, 0.975)),
        }
    return out


# --------------------------------------------------------------------------- #
# the model side: FIFO rollout with probes, and the online gamma = 0 rule
# --------------------------------------------------------------------------- #


class _LastCall:
    """Forward pre-hook: the positional inputs of the last real forward call."""

    def __init__(self) -> None:
        self.args: tuple | None = None
        self.armed = True

    def __call__(self, module, args, kwargs):
        if self.armed:
            self.args = args
        return None


class FifoProbeRecorder(FIFOPolicy):
    """FIFO eviction, unchanged. Records ``r`` per resident sentence per step, the
    gestalt of every written sentence, and -- at full-memory steps -- the probes."""

    def __init__(self, model, doc, m: int, n_steps: int, hook: _LastCall) -> None:
        self.model, self.doc, self.m, self.S, self.hook = model, doc, m, n_steps, hook
        self.assert_of = {q: a for a, q in doc.pairs}
        self.reset()

    def reset(self) -> None:
        self.D = torch.full((self.S, self.S), float("nan"), dtype=torch.float64)
        self.resident = torch.zeros(self.S, self.S, dtype=torch.bool)
        self.gest: dict[int, torch.Tensor] = {}
        self.sum_worst = 0.0
        self.identity_worst = 0.0
        self.n_probe_rows = 0
        self.hits: list[tuple[int, bool]] = []

    def observe(self, slots, attn, step: int) -> None:
        n_live = slots.n_live
        live = torch.nonzero(slots.live).flatten().tolist()
        written = [int(slots.written_at[j]) for j in live]
        if step in self.assert_of:
            a = self.assert_of[step]
            self.hits.append((step - a, a in written))
        if n_live == 0:
            return None
        r = r_of(attn, n_live, self.m)
        for j, w in zip(live, written, strict=True):
            self.D[step, w] = float(r[j])
            self.resident[step, w] = True
        if n_live == self.m:
            self.sum_worst = max(self.sum_worst, abs(float(r.sum()) - 1.0))
            if written != list(range(step - self.m, step)):
                raise AssertionError(f"step {step}: FIFO memory is {written}")
            self._probe(slots, step, float(r[0]))
        return None

    def _probe(self, slots, step: int, r0: float) -> None:
        ids_t, mask_t, kv, valid, bos_ctx, bos_valid = self.hook.args[:6]
        cand = [step - self.m, *range(step - self.m)]
        P = len(cand)
        kvP = kv.expand(P, -1, -1).clone()
        for p, i in enumerate(cand):
            kvP[p, 0] = self.gest[i]
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
        for p, i in enumerate(cand):
            rp = r_of(trace_for_row(cap, valid_p, p, step), self.m, self.m)
            self.sum_worst = max(self.sum_worst, abs(float(rp.sum()) - 1.0))
            if p == 0:
                self.identity_worst = max(self.identity_worst, abs(float(rp[0]) - r0))
            else:
                self.D[step, i] = float(rp[0])
        self.n_probe_rows += P

    def on_write(self, slots, slot: int, step: int) -> None:
        live = torch.nonzero(slots.live).flatten().tolist()
        ks = [j for j in live if int(slots.written_at[j]) == step]
        self.gest[step] = slots.gestalts[ks[0]].clone()
        return None


class OnlineG0(FIFOPolicy):
    """The gamma = 0 rule with the model in the loop: evict the argmin of this
    step's observed ``r_i(t)`` under the policy's own memory (ties: lowest slot)."""

    name = "online_g0"

    def __init__(self, doc, m: int) -> None:
        self.m = m
        self.assert_of = {q: a for a, q in doc.pairs}
        self.reset()

    def reset(self) -> None:
        self.last: tuple[int, list[float]] | None = None
        self.victims: dict[int, int] = {}
        self.hits: list[tuple[int, bool]] = []
        self.sum_worst = 0.0

    def observe(self, slots, attn, step: int) -> None:
        written = [int(x) for x in slots.written_at[slots.live].tolist()]
        if step in self.assert_of:
            a = self.assert_of[step]
            self.hits.append((step - a, a in written))
        if slots.n_live == 0:
            self.last = None
            return None
        r = r_of(attn, slots.n_live, self.m)
        if slots.n_live == self.m:
            self.sum_worst = max(self.sum_worst, abs(float(r.sum()) - 1.0))
        self.last = (step, [float(x) for x in r])
        return None

    def select_eviction(self, slots, context, step: int) -> int:
        if self.last is None or self.last[0] != step:
            raise AssertionError(f"step {step}: no r_i observed this step")
        r = self.last[1]
        best, best_v = -1, float("inf")
        for k in range(len(slots.live)):
            if bool(slots.live[k]) and r[k] < best_v:
                best, best_v = k, r[k]
        self.victims[step] = int(slots.written_at[best])
        return best


def _answer_step_fn(tm, sym_ids, sink: list):
    """Per answer target: (argmax-correct, NLL). S0-03's `answer_readout` math."""
    from rsr.train.loop import lm_token_losses

    def step_fn(t, out, ids_t, mask_t, row_valid):
        am = tm[:, t, 1:]
        if am.any():
            B = ids_t.shape[0]
            per = lm_token_losses(out.logits, ids_t).view(B, ids_t.shape[1] - 1)
            lg = out.logits[:, :-1][am]
            tgt = ids_t[:, 1:][am]
            ok = lg.argmax(-1) == tgt
            for o, n in zip(ok.tolist(), per[am].tolist(), strict=True):
                sink.append((t, bool(o), float(n)))
        return torch.zeros(())

    return step_fn


@torch.no_grad()
def run_fifo(model, doc, ids, mask, tm, sym_ids, m: int, n_steps: int) -> dict:
    """One document, alone, FIFO, with probes. -> D, residency, hits, answers."""
    hook = _LastCall()
    h = model.register_forward_pre_hook(hook, with_kwargs=True)
    rec = FifoProbeRecorder(model, doc, m, n_steps, hook)
    answers: list = []
    model.eval()
    try:
        run_policy_loop(
            model,
            ids,
            mask,
            torch.full((1,), n_steps),
            rec,
            step_fn=_answer_step_fn(tm, sym_ids, answers),
            observe=True,
        )
    finally:
        h.remove()
    return {
        "D": rec.D,
        "resident": rec.resident,
        "hits": rec.hits,
        "answers": answers,
        "sum_worst": rec.sum_worst,
        "identity_worst": rec.identity_worst,
        "n_probe_rows": rec.n_probe_rows,
    }


@torch.no_grad()
def run_online(model, doc, ids, mask, tm, sym_ids, m: int, n_steps: int) -> dict:
    pol = OnlineG0(doc, m)
    answers: list = []
    model.eval()
    run_policy_loop(
        model,
        ids,
        mask,
        torch.full((1,), n_steps),
        pol,
        step_fn=_answer_step_fn(tm, sym_ids, answers),
        observe=True,
    )
    return {
        "hits": pol.hits,
        "victims": pol.victims,
        "answers": answers,
        "sum_worst": pol.sum_worst,
    }


# --------------------------------------------------------------------------- #
# per document: targets, rules, classes, AUCs
# --------------------------------------------------------------------------- #


def targets(D: torch.Tensor, resident: torch.Tensor) -> dict[str, torch.Tensor]:
    """Every rule's target matrix from one document's ``D`` (PREREG)."""
    lit = torch.where(
        resident, torch.nan_to_num(D, nan=0.0), torch.zeros_like(D)
    ).masked_fill(torch.isnan(D), float("nan"))
    out = {
        "rule_g0": D,
        "rule_next": next_step(D),
        "lit_g0": lit,
    }
    for tag, g in GAMMAS.items():
        out[f"rule_{tag}"] = discounted_returns(D, g)
        out[f"lit_{tag}"] = discounted_returns(lit, g)
    return out


def rule_hits(doc, T: dict[str, torch.Tensor], m: int) -> dict[str, list]:
    """Model-free residency of every target rule: ``[(gap, hit), ...]`` per rule."""
    out = {}
    for name, tgt in T.items():
        qs = simulate(doc, TargetPolicy(tgt), m)["queries"]
        out[name] = [(q["gap"], q["hit"]) for q in qs]
    return out


def ref_hits(doc, seed: int, m: int) -> dict[str, list]:
    pols = {
        "fifo": FIFOPolicy(),
        "oracle": OraclePolicy(discounted_demand(doc, ORACLE_GAMMA)),
        "pending_fifo": PendingFIFO(doc),
        "factfiller": FactFiller(doc, random.Random(f"ff:{seed}:{doc.doc_id}")),
    }
    return {
        k: [(q["gap"], q["hit"]) for q in simulate(doc, p, m)["queries"]]
        for k, p in pols.items()
    }


AUC_TARGETS = ("rule_g0", "rule_g09", "rule_g097", "rule_next")


def class_and_auc(doc, D, resident, T, m: int, acc: dict) -> None:
    """Accumulate class values and per-step AUCs into ``acc`` (in place)."""
    n = D.shape[0]
    for t in range(m, n):
        res = [i for i in range(t) if bool(resident[t, i])]
        allc = list(range(t))
        cls = {i: slot_class(doc, i, t) for i in allc}
        for i in res:
            c, k = cls[i]
            acc["res"][c]["D"].append(float(D[t, i]))
            acc["res"][c]["G09"].append(float(T["rule_g09"][t, i]))
            acc["res"][c]["G097"].append(float(T["rule_g097"][t, i]))
            if c == "pending":
                acc["pend_k"].setdefault(min(k, K_MAX_DISPLAY + 1), []).append(
                    float(D[t, i])
                )
        for i in allc:
            if not bool(resident[t, i]):
                c, _ = cls[i]
                acc["probed"][c]["D"].append(float(D[t, i]))
        for scope, cands in (("res", res), ("all", allc)):
            pend = [i for i in cands if cls[i][0] == "pending"]
            fill = [i for i in cands if cls[i][0] == "filler"]
            ans = [i for i in cands if cls[i][0] == "answered"]
            for tn in AUC_TARGETS:
                v = T[tn][t]
                a = auc([float(v[i]) for i in pend], [float(v[i]) for i in fill])
                if a is not None:
                    acc["auc"][f"{scope}.pend_vs_fill.{tn}"].append(a)
                b = auc([float(v[i]) for i in pend], [float(v[i]) for i in ans])
                if b is not None:
                    acc["auc"][f"{scope}.pend_vs_ans.{tn}"].append(b)


def new_acc() -> dict:
    return {
        "res": {c: {"D": [], "G09": [], "G097": []} for c in CLASSES},
        "probed": {c: {"D": []} for c in CLASSES},
        "pend_k": {},
        "auc": {
            f"{s}.{p}.{tn}": []
            for s in ("res", "all")
            for p in ("pend_vs_fill", "pend_vs_ans")
            for tn in AUC_TARGETS
        },
    }


def summarise_acc(acc: dict) -> dict:
    return {
        "res": {c: {k: summary(v) for k, v in d.items()} for c, d in acc["res"].items()},
        "probed": {
            c: {k: summary(v) for k, v in d.items()} for c, d in acc["probed"].items()
        },
        "pend_k": {str(k): summary(v) for k, v in sorted(acc["pend_k"].items())},
        "auc": {
            k: {"mean": (sum(v) / len(v)) if v else None, "n_steps": len(v)}
            for k, v in acc["auc"].items()
        },
    }


# --------------------------------------------------------------------------- #
# C1: the fresh-stream ledger, reproduced from the FIFO rollout
# --------------------------------------------------------------------------- #


def reference_rows(source: Path) -> dict[str, dict]:
    doc = json.loads((source / "ledger.json").read_text())
    return {r["key"]: r for r in doc["rows"] if isinstance(r.get("key"), str)}


def control1(answers: list[tuple[int, bool, float]], ref, label: int, seed: int) -> dict:
    """``answers``: (gap, ok, nll) over P, FIFO rollout. Per S0-03 bucket."""
    gap = torch.tensor([a[0] for a in answers])
    ok = torch.tensor([a[1] for a in answers], dtype=torch.float64)
    nll = torch.tensor([a[2] for a in answers], dtype=torch.float64)
    worst_acc, worst_nll, bad, missing = 0.0, 0.0, [], []
    for b, sel in S003.BUCKETS.items():
        msk = sel(gap)
        if int(msk.sum()) == 0:
            continue
        for rd, mine, tol in (
            ("answer_acc", float(ok[msk].mean()), ACC_TOL),
            ("answer_nll", float(nll[msk].mean()), NLL_TOL),
        ):
            key = f"B.ckpt{label}.heldout.live.{b}.{rd}"
            row = ref.get(key)
            if row is None or len(row.get("samples", [])) <= SEEDS.index(seed):
                missing.append(key)
                continue
            theirs = row["samples"][SEEDS.index(seed)]
            if theirs is None or not (
                math.isfinite(mine) and math.isfinite(float(theirs))
            ):
                bad.append(key)
                continue
            d = abs(mine - float(theirs))
            if rd == "answer_acc":
                worst_acc = max(worst_acc, d)
            else:
                worst_nll = max(worst_nll, d)
            if d > tol:
                bad.append(key)
    return {
        "ok": not bad and not missing,
        "max_abs_acc_diff": worst_acc,
        "max_abs_nll_diff": worst_nll,
        "out_of_tolerance": bad,
        "missing": missing,
        "n_answers": len(answers),
    }


# --------------------------------------------------------------------------- #
# one (checkpoint, seed): the child's whole job
# --------------------------------------------------------------------------- #


def load_model(ckpt: Path, V: int):
    from rsr.model.tg import TGModel
    from rsr.train import checkpoint as ck

    model = TGModel(S003._cfg(V))
    ck.load(ckpt, model=model, restore_rng=False)
    model.eval()
    return model


def measure_doc(model, doc, vmap, sym_ids, seed: int, m: int, acc: dict) -> dict:
    ids, mask = encode([doc], vmap, max_tokens=L, steps=S)
    tm, _gap = answer_targets([doc], vmap, max_tokens=L, steps=S)
    fifo = run_fifo(model, doc, ids, mask, tm, sym_ids, m, S)
    online = run_online(model, doc, ids, mask, tm, sym_ids, m, S)
    T = targets(fifo["D"], fifo["resident"])
    hits = {**ref_hits(doc, seed, m), **rule_hits(doc, T, m)}
    hits["online_g0"] = online["hits"]
    replay = [
        (q["gap"], q["hit"])
        for q in simulate(doc, Scripted(online["victims"]), m)["queries"]
    ]
    fifo_mf = hits["fifo"]
    class_and_auc(doc, fifo["D"], fifo["resident"], T, m, acc)
    gap_of = {q: q - a for a, q in doc.pairs}
    return {
        "doc": doc.doc_id,
        "hits": hits,
        "c5_online_replay_ok": replay == online["hits"],
        "c_fifo_residency_ok": fifo["hits"] == fifo_mf,
        "sum_worst": max(fifo["sum_worst"], online["sum_worst"]),
        "identity_worst": fifo["identity_worst"],
        "n_probe_rows": fifo["n_probe_rows"],
        "fifo_answers": [(gap_of[t], ok, nll) for t, ok, nll in fifo["answers"]],
        "online_answers": [(gap_of[t], ok, nll) for t, ok, nll in online["answers"]],
        "D": fifo["D"],
    }


def child(label: int, seed: int, source: Path, out: Path, limit: int | None) -> int:
    torch.set_num_threads(int(os.environ.get("RSR_LOOKAHEAD_THREADS", "1")))
    t0 = time.time()
    docs, vmap, V, closure = documents(seed)
    if limit is not None:
        docs = docs[:limit]
    ck = ckpt_path(source, label, seed)
    got = sha256(ck)
    if got != CKPT_SHA256[(label, seed)]:
        print(f"sha256 mismatch {ck}: {got}", file=sys.stderr)
        return int(Exit.DID_NOT_RUN)
    model = load_model(ck, V)
    sym_ids = torch.tensor([vmap[s] for s in ANSWER_SYMBOLS])
    acc = new_acc()
    per_doc, raw_D = [], {}
    p_answers = []
    for n, doc in enumerate(docs):
        r = measure_doc(model, doc, vmap, sym_ids, seed, M, acc)
        raw_D[doc.doc_id] = r.pop("D")
        if P_SET[0] <= doc.doc_id < P_SET[1]:
            p_answers += r["fifo_answers"]
        per_doc.append(r)
        if n % 50 == 0:
            print(f"doc {n}/{len(docs)} {time.time() - t0:.0f}s", flush=True)
    ref = reference_rows(source)
    c1 = control1(p_answers, ref, label, seed) if p_answers else {"ok": False}
    payload = {
        "checkpoint": label,
        "seed": seed,
        "ckpt": str(ck),
        "ckpt_sha256": got,
        "closure": closure,
        "n_documents": len(docs),
        "control1": c1,
        "classes": summarise_acc(acc),
        "per_doc": per_doc,
        "seconds": time.time() - t0,
        "threads": torch.get_num_threads(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") as f:
        json.dump(payload, f)
    torch.save(raw_D, out.parent / (out.name.removesuffix(".json.gz") + ".D.pt"))
    return 0


# --------------------------------------------------------------------------- #
# the parent
# --------------------------------------------------------------------------- #


def raw_path(root: Path, label: int, seed: int) -> Path:
    return root / "raw" / f"B-ckpt{label}-seed{seed}.json.gz"


def child_argv(label: int, seed: int, source: Path, out: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single",
        "--checkpoint",
        str(label),
        "--seed",
        str(seed),
        "--source",
        str(source),
        "--out",
        str(out),
    ]


def run_children(source: Path, root: Path, parallel: int = 6) -> dict[tuple, int]:
    jobs = [(c, s) for c in CHECKPOINTS for s in SEEDS]
    rcs: dict[tuple, int] = {}
    running: dict[tuple, subprocess.Popen] = {}
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    while jobs or running:
        while jobs and len(running) < parallel:
            j = jobs.pop(0)
            log = open(logs / f"B-ckpt{j[0]}-seed{j[1]}.log", "w")  # noqa: SIM115
            running[j] = subprocess.Popen(
                child_argv(j[0], j[1], source, raw_path(root, *j)),
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=ROOT,
            )
        for j, p in list(running.items()):
            rc = p.poll()
            if rc is not None:
                rcs[j] = rc
                del running[j]
        time.sleep(2)
    return rcs


def control2(seed: int) -> dict:
    """C2, model-free on P (no model): oracle - FIFO, the kindrand.py convention,
    and oracle == pending_fifo victim for victim."""
    docs = CSC._generate(seed, P_SET[1])[P_SET[0] : P_SET[1]]

    def hr(make):
        qs = [q for d in docs for q in simulate(d, make(d), M)["queries"]]
        return sum(q["hit"] for q in qs) / len(qs)

    fifo = hr(lambda d: FIFOPolicy())
    oracle = hr(lambda d: OraclePolicy(discounted_demand(d, ORACLE_GAMMA)))
    rng = random.Random(seed)
    kind = hr(lambda d: FactFiller(d, rng))

    class Rec:
        def __init__(self, inner):
            self.inner, self.v = inner, []

        def select_eviction(self, slots, ctx, step):
            k = self.inner.select_eviction(slots, ctx, step)
            self.v.append(int(slots.written_at[k]))
            return k

        def reset(self):
            self.inner.reset()

    diff = total = 0
    for d in docs:
        a, b = Rec(OraclePolicy(discounted_demand(d, ORACLE_GAMMA))), Rec(PendingFIFO(d))
        simulate(d, a, M)
        simulate(d, b, M)
        total += len(a.v)
        diff += sum(x != y for x, y in zip(a.v, b.v, strict=True))
    got = {"oracle_minus_fifo": oracle - fifo, "kindrand_minus_fifo": kind - fifo}
    ok = (
        round(got["oracle_minus_fifo"], 4) == C2_ORACLE[seed]
        and round(got["kindrand_minus_fifo"], 4) == C2_KINDRAND[seed]
        and diff == 0
    )
    return {**got, "victims_differing": diff, "victims_total": total, "ok": ok}


def _rate(pairs: list, pred=lambda g: True) -> tuple[int, int]:
    sel = [h for g, h in pairs if pred(g)]
    return len(sel), sum(sel)


DIFFS = {
    "room_3b": ("rule_g09", "rule_g0"),
    "room_3b_097": ("rule_g097", "rule_g0"),
    "oracle_minus_fifo": ("oracle", "fifo"),
    "factfiller_minus_fifo": ("factfiller", "fifo"),
    "pending_fifo_minus_fifo": ("pending_fifo", "fifo"),
    "rule_g0_minus_fifo": ("rule_g0", "fifo"),
    "rule_g09_minus_fifo": ("rule_g09", "fifo"),
    "online_g0_minus_rule_g0": ("online_g0", "rule_g0"),
    "lit_room_3b": ("lit_g09", "lit_g0"),
    "next_minus_g0": ("rule_next", "rule_g0"),
}


def summarise_child(payload: dict) -> dict:
    seed = payload["seed"]
    docs = payload["per_doc"]
    out: dict[str, Any] = {}
    for setname, keep in (
        ("U", lambda d: True),
        ("P", lambda d: P_SET[0] <= d < P_SET[1]),
    ):
        sel = [d for d in docs if keep(d["doc"])]
        per_doc = {p: [_rate(d["hits"][p]) for d in sel] for p in ALL_POLICIES}
        hit = {}
        for p in ALL_POLICIES:
            n = sum(x[0] for x in per_doc[p])
            h = sum(x[1] for x in per_doc[p])
            hit[p] = h / n
            ng, hg = 0, 0
            for d in sel:
                a, b = _rate(d["hits"][p], lambda g: g > M)
                ng, hg = ng + a, hg + b
            hit[f"{p}.gap_gt_M"] = hg / ng if ng else float("nan")
        boot = paired_bootstrap(per_doc, DIFFS, seed)
        oa = [a for d in sel for a in d["online_answers"]]
        fa = [a for d in sel for a in d["fifo_answers"]]
        out[setname] = {
            "hit": hit,
            "diffs": boot,
            "n_documents": len(sel),
            "n_queries": sum(x[0] for x in per_doc["fifo"]),
            "acc_fifo": sum(a[1] for a in fa) / len(fa),
            "acc_online_g0": sum(a[1] for a in oa) / len(oa),
        }
    out["controls"] = {
        "C1": payload["control1"],
        "C3_sum_worst": max(d["sum_worst"] for d in docs),
        "C4_identity_worst": max(d["identity_worst"] for d in docs),
        "C5_online_replay_ok": all(d["c5_online_replay_ok"] for d in docs),
        "C5_fifo_residency_ok": all(d["c_fifo_residency_ok"] for d in docs),
        "closure": payload["closure"],
        "ckpt_sha256_ok": payload["ckpt_sha256"]
        == CKPT_SHA256[(payload["checkpoint"], seed)],
    }
    out["classes"] = payload["classes"]
    out["seconds"] = payload["seconds"]
    out["n_probe_rows"] = sum(d["n_probe_rows"] for d in docs)
    return out


def controls_failed_of(summ: dict, tag: str) -> list[str]:
    c = summ["controls"]
    f = []
    if not c["C1"].get("ok"):
        f.append(f"C1 failed on {tag}: {json.dumps(c['C1'])[:400]}")
    if not c["C3_sum_worst"] <= SUM_TOL:
        f.append(f"C3 failed on {tag}: worst |sum r - 1| {c['C3_sum_worst']}")
    if not c["C4_identity_worst"] <= IDENTITY_TOL:
        f.append(f"C4 failed on {tag}: identity probe worst {c['C4_identity_worst']}")
    if not (c["C5_online_replay_ok"] and c["C5_fifo_residency_ok"]):
        f.append(f"C5 failed on {tag}")
    if not (c["closure"]["ok"] and c["ckpt_sha256_ok"]):
        f.append(f"C6 failed on {tag}")
    return f


def decide(room: dict[int, dict[int, dict]], failed: list[str]) -> dict:
    """``room[label][seed]`` = room_3b {point, lo, hi} on U."""
    per_ckpt = {
        f"B.ckpt{c}": classify(room[c])
        for c in CHECKPOINTS
        if c in room and all(s in room[c] for s in SEEDS)
    }
    missing = [f"B.ckpt{c}" for c in CHECKPOINTS if f"B.ckpt{c}" not in per_ckpt]
    failed = list(failed) + ([f"missing: {missing}"] if missing else [])
    headline = "INCONCLUSIVE" if failed else per_ckpt[f"B.ckpt{HEADLINE}"]
    return {
        "classification": headline,
        "per_checkpoint": per_ckpt,
        "controls_failed": failed,
        "outcome": OUTCOME[headline],
        "exit": int(Exit.DID_NOT_RUN if failed else Exit.OK),
    }


def execute(led, source: Path, root: Path) -> Exit:
    disj = disjointness()
    led.note("C6.disjointness", disj, how="run.py::disjointness")
    failed: list[str] = []
    if not disj["ok"]:
        failed.append("C6 (disjointness) failed")
    c2 = {s: control2(s) for s in SEEDS}
    led.note("C2", c2, how="run.py::control2 (model-free, P)")
    for s, v in c2.items():
        if not v["ok"]:
            failed.append(f"C2 failed on seed {s}: {v}")
    rcs = run_children(source, root)
    summaries: dict[str, dict] = {}
    room: dict[int, dict[int, dict]] = {}
    for (c, s), rc in sorted(rcs.items()):
        led.command(child_argv(c, s, source, raw_path(root, c, s)), exit_code=rc)
        tag = f"B.ckpt{c}.seed{s}"
        if rc != 0:
            failed.append(f"child {tag} exited {rc}")
            continue
        with gzip.open(raw_path(root, c, s), "rt") as f:
            payload = json.load(f)
        summ = summarise_child(payload)
        failed += controls_failed_of(summ, tag)
        summaries[tag] = summ
        room.setdefault(c, {})[s] = summ["U"]["diffs"]["room_3b"]
    dec = decide(room, failed)
    write_rows(led, summaries, dec)
    (root / "summaries.json").write_text(json.dumps(summaries, indent=1, default=str))
    led.run_meta(
        seeds_actually_run=sorted({int(k.split("seed")[1]) for k in summaries}) or []
    )
    led.status("ok" if not dec["controls_failed"] else "partial")
    led.verdict(
        falsifier=FALSIFIER,
        outcome=dec["outcome"],
        detail=(
            f"classification {dec['classification']}; per checkpoint "
            f"{json.dumps(dec['per_checkpoint'], sort_keys=True)}; controls failed "
            f"{dec['controls_failed']}"
        ),
    )
    return Exit(dec["exit"])


def write_rows(led, summaries: dict, dec: dict) -> None:
    led.note("classification", dec["classification"], how="run.py::decide")
    led.note("per_checkpoint", dec["per_checkpoint"], how="run.py::decide")
    led.note("controls_failed", dec["controls_failed"], how="run.py::decide")
    for c in CHECKPOINTS:
        keys = [f"B.ckpt{c}.seed{s}" for s in SEEDS]
        if not all(k in summaries for k in keys):
            continue
        sm = [summaries[k] for k in keys]
        pre = f"B.ckpt{c}"
        how = "run.py::summarise_child; seed order 0,1,2"
        for setname in ("U", "P"):
            for p in sm[0][setname]["hit"]:
                led.stat(
                    f"{pre}.{setname}.hit.{p}",
                    [x[setname]["hit"][p] for x in sm],
                    how=how + " (model-free residency / in-loop for online_g0)",
                )
            for d in DIFFS:
                for f in ("point", "lo", "hi"):
                    led.stat(
                        f"{pre}.{setname}.{d}.{f}",
                        [x[setname]["diffs"][d][f] for x in sm],
                        how=how + " -> paired_bootstrap (per document, 2000)",
                    )
            for k in ("acc_fifo", "acc_online_g0", "n_documents", "n_queries"):
                led.stat(f"{pre}.{setname}.{k}", [x[setname][k] for x in sm], how=how)
        cl = [x["classes"] for x in sm]
        for scope in ("res", "probed"):
            for cls in CLASSES:
                for tgt in cl[0][scope][cls]:
                    for f in ("n", "mean", "sd", "q10", "q50", "q90"):
                        vals = [x[scope][cls][tgt][f] for x in cl]
                        if any(v is None for v in vals):
                            continue
                        led.stat(f"{pre}.class.{scope}.{cls}.{tgt}.{f}", vals, how=how)
        for k in cl[0]["pend_k"]:
            for f in ("n", "mean", "q50"):
                vals = [x["pend_k"].get(k, {}).get(f) for x in cl]
                if any(v is None for v in vals):
                    continue
                led.stat(f"{pre}.pend_k.{k}.{f}", vals, how=how)
        for k in cl[0]["auc"]:
            vals = [x["auc"][k]["mean"] for x in cl]
            if any(v is None for v in vals):
                continue
            led.stat(f"{pre}.auc.{k}", vals, how=how + " (mean over steps)")
            led.stat(
                f"{pre}.auc.{k}.n_steps", [x["auc"][k]["n_steps"] for x in cl], how=how
            )
        for k in ("C3_sum_worst", "C4_identity_worst"):
            led.stat(f"{pre}.{k}", [x["controls"][k] for x in sm], how=how)
        for f in ("max_abs_acc_diff", "max_abs_nll_diff"):
            led.stat(
                f"{pre}.C1.{f}",
                [x["controls"]["C1"].get(f, float("nan")) for x in sm],
                how=how,
            )
        led.stat(f"{pre}.seconds", [x["seconds"] for x in sm], how="child wall clock")
        led.stat(f"{pre}.n_probe_rows", [x["n_probe_rows"] for x in sm], how=how)
    for k, x in sorted(summaries.items()):
        led.note(f"{k}.controls", x["controls"], how="child payload")


def manifest_config(source: Path) -> dict:
    return {
        "run_id": RUN_ID,
        "prereg": PREREG,
        "prereg_commit": PREREG_COMMIT,
        "question": QUESTION,
        "falsifier": FALSIFIER,
        "expected": EXPECTED,
        "seeds": SEEDS,
        "checkpoints": [f"B.ckpt{c}" for c in CHECKPOINTS],
        "headline": f"B.ckpt{HEADLINE}",
        "policies": list(ALL_POLICIES),
        "gammas": GAMMAS,
        "oracle_gamma": ORACLE_GAMMA,
        "M": M,
        "S": S,
        "sets": {"P": list(P_SET), "E": list(E_SET), "U": "E then P"},
        "thresholds": {
            "TESTABLE_MIN": TESTABLE_MIN,
            "NO_ROOM_MAX": NO_ROOM_MAX,
            "N_BOOT": N_BOOT,
            "BOOT_SEED_BASE": BOOT_SEED_BASE,
            "SUM_TOL": SUM_TOL,
            "IDENTITY_TOL": IDENTITY_TOL,
            "ACC_TOL": ACC_TOL,
            "NLL_TOL": NLL_TOL,
        },
        "source": str(source),
        "source_ledger_sha256": sha256(source / "ledger.json"),
        "checkpoint_sha256": {
            f"B.ckpt{c}.seed{s}": h for (c, s), h in CKPT_SHA256.items()
        },
        "device": "cpu",
        "threads_per_child": int(os.environ.get("RSR_LOOKAHEAD_THREADS", "1")),
    }


# --------------------------------------------------------------------------- #
# RESULTS.md, rendered from the ledger
# --------------------------------------------------------------------------- #


def render_results(doc: dict) -> str:
    rows = {r["key"]: r for r in doc["rows"]}
    prov = doc["provenance"]

    def st(key: str, i: int) -> str:
        r = rows.get(key)
        if r is None:
            return "—"
        v = r["samples"][i]
        return "nan" if v != v else f"{v:.4f}"

    def ci(prefix: str, i: int) -> str:
        lo, hi = st(prefix + ".lo", i), st(prefix + ".hi", i)
        return f"{st(prefix + '.point', i)} [{lo}, {hi}]"

    out = [
        "# RESULTS — lookahead room (W10)",
        "",
        f"Rendered from `runs/{RUN_ID}/ledger.json` by `{EXPERIMENT} --render-results`. "
        f"PREREG `{PREREG}` (commit `{PREREG_COMMIT}`). Run sha "
        f"`{prov.get('git_sha')}`, platform `{prov.get('platform')}`, device `cpu`.",
        "",
        f"Command: `uv run python {EXPERIMENT}` (children under `commands` in the "
        "ledger).",
        "",
        f"**classification: {rows['classification']['value']}** (headline "
        f"B.ckpt{HEADLINE}); verdict outcome `{doc['verdict']['outcome']}`.",
        "",
        "Per checkpoint (`per_checkpoint`): "
        + ", ".join(f"`{k}` {v}" for k, v in rows["per_checkpoint"]["value"].items()),
        "",
        "Controls failed (`controls_failed`): "
        f"{rows['controls_failed']['value'] or 'none'}",
        "",
        "🔴 The rule hit rates are an upper-bound-style proxy from FIFO-world targets "
        "(demand-if-resident probes), not RSR's result; the discounted-return "
        "targets are hindsight. See PREREG.",
        "",
    ]
    for c in CHECKPOINTS:
        pre = f"B.ckpt{c}"
        if f"{pre}.U.room_3b.point" not in rows:
            continue
        out += [
            f"## {pre}",
            "",
            "Differences, set U (point [paired per-document percentile CI]):",
            "",
            "| key | seed0 | seed1 | seed2 |",
            "|---|---|---|---|",
        ]
        for d in DIFFS:
            k = f"{pre}.U.{d}"
            out.append(f"| `{k}` | " + " | ".join(ci(k, i) for i in range(3)) + " |")
        out += [
            "",
            "Hit rates (set U):",
            "",
            "| key | seed0 | seed1 | seed2 |",
            "|---|---|---|---|",
        ]
        for p in ALL_POLICIES:
            for suf in ("", ".gap_gt_M"):
                k = f"{pre}.U.hit.{p}{suf}"
                out.append(f"| `{k}` | {st(k, 0)} | {st(k, 1)} | {st(k, 2)} |")
        out += [
            "",
            "Hit rates (set P):",
            "",
            "| key | seed0 | seed1 | seed2 |",
            "|---|---|---|---|",
        ]
        for p in ALL_POLICIES:
            k = f"{pre}.P.hit.{p}"
            out.append(f"| `{k}` | {st(k, 0)} | {st(k, 1)} | {st(k, 2)} |")
        out += [
            "",
            "Live answer accuracy, FIFO vs online_g0 (set U):",
            "",
            "| key | seed0 | seed1 | seed2 |",
            "|---|---|---|---|",
        ]
        for k in (f"{pre}.U.acc_fifo", f"{pre}.U.acc_online_g0"):
            out.append(f"| `{k}` | {st(k, 0)} | {st(k, 1)} | {st(k, 2)} |")
        out += [
            "",
            "r_i (= D) by class, FIFO-resident slots at full-memory steps:",
            "",
            "| key | seed0 | seed1 | seed2 |",
            "|---|---|---|---|",
        ]
        for scope in ("res", "probed"):
            for cls in CLASSES:
                for tgt in ("D", "G09", "G097"):
                    for f in ("n", "mean", "sd", "q10", "q50", "q90"):
                        k = f"{pre}.class.{scope}.{cls}.{tgt}.{f}"
                        if k in rows:
                            out.append(
                                f"| `{k}` | {st(k, 0)} | {st(k, 1)} | {st(k, 2)} |"
                            )
        out += [
            "",
            "Pending asserts by k (steps to query; the last bin pools every k above M):",
            "",
            "| key | seed0 | seed1 | seed2 |",
            "|---|---|---|---|",
        ]
        for k in sorted(
            (r for r in rows if r.startswith(f"{pre}.pend_k.")),
            key=lambda r: (int(r.split(".")[3]), r),
        ):
            out.append(f"| `{k}` | {st(k, 0)} | {st(k, 1)} | {st(k, 2)} |")
        out += [
            "",
            "AUCs (mean over full-memory steps):",
            "",
            "| key | seed0 | seed1 | seed2 |",
            "|---|---|---|---|",
        ]
        for k in sorted(r for r in rows if r.startswith(f"{pre}.auc.")):
            out.append(f"| `{k}` | {st(k, 0)} | {st(k, 1)} | {st(k, 2)} |")
        out += [
            "",
            "Controls:",
            "",
            "| key | seed0 | seed1 | seed2 |",
            "|---|---|---|---|",
        ]
        for k in (
            f"{pre}.C1.max_abs_acc_diff",
            f"{pre}.C1.max_abs_nll_diff",
            f"{pre}.C3_sum_worst",
            f"{pre}.C4_identity_worst",
            f"{pre}.seconds",
        ):
            if k in rows:
                out.append(
                    f"| `{k}` | " + " | ".join(repr(v) for v in rows[k]["samples"]) + " |"
                )
        out.append("")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> Exit:
    ap = ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--checkpoint", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--source", type=Path, default=None)
    ap.add_argument("--out", type=Path)
    ap.add_argument(
        "--limit", type=int, default=None, help="timing only (PREREG fallback)"
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--render-results", action="store_true")
    ap.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    a = ap.parse_args(argv)
    source = a.source or default_source()
    if a.single:
        return status(child(a.checkpoint, a.seed, source, a.out, a.limit))
    root = a.runs_root / RUN_ID
    if a.render_results:
        doc = json.loads((root / "ledger.json").read_text())
        (ROOT / "experiments" / "lookahead-room" / "RESULTS.md").write_text(
            render_results(doc)
        )
        return Exit.OK
    if a.dry_run:
        print(json.dumps(disjointness()))
        for s in SEEDS:
            print(s, documents(s)[3])
        for c in CHECKPOINTS:
            for s in SEEDS:
                p = ckpt_path(source, c, s)
                print(p, sha256(p) == CKPT_SHA256[(c, s)])
        return Exit.OK
    from ledger import Ledger

    led = Ledger(RUN_ID, question=QUESTION, runs_root=a.runs_root)
    led.run_meta(device="cpu", steps_requested=0, steps_done=0)
    led.manifest(manifest_config(source))
    rc = execute(led, source, root)
    led.command(
        [sys.executable, EXPERIMENT] + (argv if argv is not None else sys.argv[1:]),
        exit_code=int(rc),
        note="the parent; its exit code is the run's",
    )
    led.write()
    return status(rc)


if __name__ == "__main__":
    run_main(main)
