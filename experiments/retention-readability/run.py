"""Retention readability -- can a FIFO-trained model read an old fact at a shifted rank?

Pre-registration: `experiments/retention-readability/PREREG.md`, committed alone at
ab7cbd2, ahead of this file. Read-only: it loads frozen fresh-stream checkpoints and
**trains nothing**. Not an E1 substrate ruling (D2) and not a headroom / kill-gate test
(E-feas is the kill gate and survived).

Why (ADR-0006): `P^(sent)` is added to memory keys by RANK, the position in the
oldest-first prefix. Under FIFO rank and age coincide, so a FIFO-trained model has only
ever seen a fact of gap ``g`` at rank ``M - g``. Any other eviction rule compacts the
prefix and puts kept facts at ranks and among neighbours FIFO never produced.

The instrument (PREREG "Instrument"):

* each document is run **alone** (B = 1) through the real
  `rsr.model.tg.policy_loop.run_policy_loop`, eval mode, ``no_grad``, ``lengths = S``,
  with a **fresh policy per document** wrapped in `DocPolicy`, which carries the
  document id it was built for (asserted against the tokens being run), checks the
  oracle's demand is that document's own, and records the live prefix after every
  write -- the memory the next sentence reads. `OraclePolicy` in a batched loop
  applies row 0's demand to every row (`policy_loop.py:358-365`); that is why B = 1;
* arms: FIFO, the oracle (Belady MIN, E-feas's gamma), and the fact/filler rule
  (evict a random non-assert slot; content-only, age-free, seeded per document);
* per answer: residency, rank index, correct, NLL, NLL over 16, Brier16.

Controls (each overrides everything; any failure -> inconclusive, exit 3):
1. harness FIFO bit-exact to ``answer_readout(cond="live")`` run on the same document
   alone (argmax equal, NLL max |diff| == 0.0);
2. the fresh-stream instrument reproduces the fresh-stream ledger's
   ``{B.ckpt2500,B.ckpt3000,A.ckpt3000}.heldout.live.*`` samples within 1e-6;
3. the per-document assertions (id, demand, B = 1);
4. harness residency == model-free `rsr.metrics.headroom.simulate` residency;
5. extended documents disjoint from every trained / probed range, vocabulary closed;
6. arm A's oracle - FIFO 95 % CI inside (-0.03, +0.03).

Usage::

    uv run python experiments/retention-readability/run.py --dry-run
    uv run python experiments/retention-readability/run.py          # the run
    uv run python experiments/retention-readability/run.py --render-results

Exit codes (`rsr.exit_codes`): 0 a classification was reached (MIXED included) ·
3 inconclusive / did not run (a control failed, a measurement raised, a checkpoint
missing).
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import os
import random
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

from rsr.baselines.fifo import FIFOPolicy  # noqa: E402
from rsr.baselines.oracle import OraclePolicy  # noqa: E402
from rsr.data.synthetic import ANSWER_SYMBOLS, discounted_demand  # noqa: E402
from rsr.exit_codes import ArgumentParser, Exit, run_main, status  # noqa: E402
from rsr.metrics.headroom import simulate  # noqa: E402
from rsr.model.tg.policy_loop import run_policy_loop  # noqa: E402
from rsr.train.loop import (  # noqa: E402
    answer_targets,
    build_vocab,
    encode,
    lm_token_losses,
)

EXPERIMENT = "experiments/retention-readability/run.py"
RUN_ID = "retention-readability"
PREREG = "experiments/retention-readability/PREREG.md"
PREREG_COMMIT = "ab7cbd2"


def _load(name: str, rel: str):
    """Import an experiment script by path. Imported, never copied."""
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CSC = _load("_readability_csc", "experiments/corpus-size-curve/run.py")
S003 = CSC.S003
EFEAS = _load("_readability_efeas", "experiments/efeas/run.py")

# --------------------------------------------------------------------------- #
# 🔒 PREREG.md, transcribed. Changing any is changing the PREREG.
# --------------------------------------------------------------------------- #

SEEDS = [0, 1, 2]
#: (fresh-stream arm, checkpoint). B.3000 is the headline; A.3000 the negative control.
CHECKPOINTS = (("B", 2500), ("B", 3000), ("A", 3000))
HEADLINE = ("B", 3000)
NULL_ARM = ("A", 3000)
ARMS = ("fifo", "oracle", "factfiller")
GAMMA = EFEAS.GAMMA  # 0.97, E-feas's
M = S003.CONFIG["memory_slots"]
S = S003.CONFIG["steps_per_stream"]
L = S003.CONFIG["max_tokens"]
PROBE = CSC.PROBE  # (0, 64)
P_SET = CSC.HELDOUT  # (4096, 4160)
E_SET = (64, 1088)
STREAM_B = (4160, 4160 + 16 * 3000)  # fresh-stream arms A/B
STREAM_ESCAPE = (4160 + 16 * 3000, 4160 + 16 * 9000)  # fresh-escape
VOCAB_DOCUMENTS = 64
RP_MAX = 0.03
READ_MARGIN = 0.10
A_NULL = 0.03
N_BOOT = 2000
BOOT_SEED_BASE = 20260926
REPRO_TOL = 1e-6
#: Control 1: the harness's FIFO NLL must equal answer_readout's EXACTLY.
CONTROL1_TOL = 0.0
LEDGER_BUCKETS = tuple(S003.BUCKETS)
LEDGER_READOUTS = (
    "answer_acc",
    "answer_nll",
    "answer_nll_over_16",
    "answer_brier_over_16",
)

QUESTION = (
    "Can fresh-stream arm B, trained only under FIFO eviction, read a fact that a "
    "different eviction rule kept in memory at a rank FIFO never put it at?"
)
FALSIFIER = (
    "none of spec section 2's; a D2 input. Claim tested: arm B, as trained, can read "
    "facts kept at shifted ranks (ADR-0006: memory keys are rank-indexed). "
    "READABLE_AT_SHIFTED_RANK -> survived, RANK_BOUND -> falsified, "
    "MIXED -> inconclusive."
)
EXPECTED = (
    "Pre-registered expectation: (a) NO_PENALTY at ckpt3000 about 60 percent; (b) "
    "READABLE about 60 percent; headline READABLE_AT_SHIFTED_RANK about 45 percent, "
    "MIXED about 40 percent, RANK_BOUND about 15 percent; the arm A null holds."
)


def default_source() -> Path:
    """`<main checkout>/.worktrees/fresh-stream/runs/fresh-stream`, found through git's
    common dir so it resolves from any worktree of this repository. Read-only."""
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


def ckpt_path(source: Path, arm: str, label: int, seed: int) -> Path:
    return source / arm / f"seed{seed}" / f"ckpt-{label:06d}.pt"


# --------------------------------------------------------------------------- #
# documents (control 5)
# --------------------------------------------------------------------------- #


def disjointness(e_set: tuple[int, int] = E_SET) -> dict:
    """Control 5, ranges: E against the probe, P, arm B's stream and fresh-escape's."""
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
    """-> (U docs in order E then P, vmap, V, closure dict). The map is
    `doc_sets(seed, 64)`'s, the one arm B's checkpoints were trained with."""
    sets, vmap, V = CSC.doc_sets(seed, VOCAB_DOCUMENTS)
    e_docs = CSC._generate(seed, e_set[1])[e_set[0] : e_set[1]]
    missing = sorted(set(build_vocab(e_docs)) - set(vmap))
    closure = {"ok": not missing, "missing": missing[:20], "n_missing": len(missing)}
    docs = tuple(e_docs) + tuple(sets["heldout"])
    return docs, vmap, V, closure


# --------------------------------------------------------------------------- #
# the policies
# --------------------------------------------------------------------------- #


class FactFillerPolicy:
    """Evict uniformly at random among live slots whose sentence is not an assert;
    if every live slot holds an assert, uniformly at random among all live slots.
    Content-only (the sentence's kind), age-free, no future."""

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


def factfiller_rng(seed: int, doc_id: int) -> random.Random:
    return random.Random(f"ff:{seed}:{doc_id}")


def make_inner(arm: str, doc, seed: int):
    """A FRESH policy for one document."""
    if arm == "fifo":
        return FIFOPolicy()
    if arm == "oracle":
        return OraclePolicy(discounted_demand(doc, GAMMA))
    if arm == "factfiller":
        return FactFillerPolicy(doc, factfiller_rng(seed, doc.doc_id))
    raise ValueError(f"unknown arm {arm!r}")


class DocPolicy:
    """One document's policy: the id it was built for, and the live prefix after
    every write (the memory the NEXT sentence's forward reads)."""

    def __init__(self, inner, doc_id: int) -> None:
        self.inner = inner
        self.doc_id = doc_id
        self.name = getattr(inner, "name", type(inner).__name__)
        self.prefix: list[int] = []

    def reset(self) -> None:
        self.inner.reset()
        self.prefix = []

    def select_eviction(self, slots, context, step: int) -> int:
        return self.inner.select_eviction(slots, context, step)

    def observe(self, slots, attn, step: int) -> None:
        return self.inner.observe(slots, attn, step)

    def on_write(self, slots, slot: int, step: int) -> None:
        self.inner.on_write(slots, slot, step)
        self.prefix = [int(x) for x in slots.written_at[slots.live].tolist()]


class DocumentMismatch(AssertionError):
    """Control 3: a policy built for one document was run on another."""


def check_policy(policy: DocPolicy, doc, n_rows: int) -> None:
    """Control 3, per document: B = 1, the policy's id is this document's, and an
    oracle's demand is this document's own."""
    if n_rows != 1:
        raise DocumentMismatch(f"B = {n_rows}: the harness runs one document at a time")
    if policy.doc_id != doc.doc_id:
        raise DocumentMismatch(
            f"policy built for document {policy.doc_id} run on document {doc.doc_id}"
        )
    if isinstance(policy.inner, OraclePolicy):
        own = [[float(x) for x in row] for row in discounted_demand(doc, GAMMA)]
        if policy.inner.demand != own:
            raise DocumentMismatch(
                f"oracle demand is not document {doc.doc_id}'s own discounted demand"
            )


# --------------------------------------------------------------------------- #
# one document through the model
# --------------------------------------------------------------------------- #


def _encode_one(doc, vmap):
    ids, mask = encode([doc], vmap, max_tokens=L, steps=S)
    tm, gap = answer_targets([doc], vmap, max_tokens=L, steps=S)
    return ids, mask, tm, gap


def _answer_stats(logits, ids_t, am, sym_ids):
    """S0-03's `answer_readout` per-answer quantities, computed the same way."""
    B = ids_t.shape[0]
    per = lm_token_losses(logits, ids_t).view(B, L - 1)
    lg = logits[:, :-1][am]
    tgt = ids_t[:, 1:][am]
    ok = lg.argmax(-1) == tgt
    lp16 = torch.log_softmax(lg[:, sym_ids], dim=-1)
    pos = (sym_ids.unsqueeze(0) == tgt.unsqueeze(1)).float().argmax(-1)
    nll16 = -lp16.gather(1, pos.unsqueeze(1)).squeeze(1)
    onehot = torch.nn.functional.one_hot(pos, len(sym_ids)).to(lp16.dtype)
    brier = (lp16.exp() - onehot).pow(2).sum(-1)
    return per[am], ok, nll16, brier


@torch.no_grad()
def run_document(model, doc, vmap, sym_ids, policy: DocPolicy) -> list[dict]:
    """One document, alone, through `run_policy_loop` under `policy`. One record per
    answer target."""
    ids, mask, tm, gap = _encode_one(doc, vmap)
    check_policy(policy, doc, ids.shape[0])
    assert_of = {q: a for a, q in doc.pairs}
    records: list[dict] = []

    def step_fn(t, out, ids_t, mask_t, row_valid):
        am = tm[:, t, 1:]
        if am.any():
            if int(am.sum()) != 1 or t not in assert_of:
                raise AssertionError(f"doc {doc.doc_id} step {t}: not one query answer")
            a = assert_of[t]
            if int(gap[0, t]) != t - a:
                raise AssertionError(f"doc {doc.doc_id} step {t}: gap {int(gap[0, t])}")
            nll, ok, nll16, brier = _answer_stats(out.logits, ids_t, am, sym_ids)
            prefix = policy.prefix
            resident = a in prefix
            records.append(
                {
                    "doc": doc.doc_id,
                    "q": t,
                    "a": a,
                    "gap": t - a,
                    "resident": resident,
                    "rank": prefix.index(a) if resident else -1,
                    "n_live": len(prefix),
                    "ok": bool(ok[0]),
                    "nll": float(nll[0]),
                    "nll16": float(nll16[0]),
                    "brier16": float(brier[0]),
                }
            )
        return torch.zeros(())

    was = model.training
    model.eval()
    try:
        run_policy_loop(model, ids, mask, torch.full((1,), S), policy, step_fn=step_fn)
    finally:
        model.train(was)
    return records


def readout_document(model, doc, vmap, sym_ids) -> dict:
    """S0-03's `answer_readout(cond="live")` on this document alone (B = 1)."""
    ids, mask, tm, gap = _encode_one(doc, vmap)
    return S003.answer_readout(model, ids, mask, tm, gap, sym_ids, cond="live")


def control1_compare(records: list[dict], readout: dict) -> dict:
    """Control 1 for one document: argmax mismatches and NLL max |diff|."""
    ok_h = [r["ok"] for r in records]
    ok_r = [bool(x) for x in readout["ok"].tolist()]
    nll_h = torch.tensor([r["nll"] for r in records], dtype=torch.float64)
    nll_r = readout["nll"].to(torch.float64)
    if len(ok_h) != len(ok_r):
        return {"n": len(ok_h), "ok_mismatches": None, "max_abs_nll_diff": None}
    return {
        "n": len(ok_h),
        "ok_mismatches": sum(x != y for x, y in zip(ok_h, ok_r, strict=True)),
        "max_abs_nll_diff": float((nll_h - nll_r).abs().max()) if ok_h else 0.0,
    }


def control1_ok(c: dict) -> bool:
    return (
        c["ok_mismatches"] == 0
        and c["max_abs_nll_diff"] is not None
        and c["max_abs_nll_diff"] <= CONTROL1_TOL
    )


def model_free_residency(arm: str, doc, seed: int) -> list[bool]:
    """Control 4: `simulate` under a fresh policy of the same arm and RNG."""
    return [q["hit"] for q in simulate(doc, make_inner(arm, doc, seed), M)["queries"]]


# --------------------------------------------------------------------------- #
# control 2: the fresh-stream ledger, reproduced
# --------------------------------------------------------------------------- #


def reference_rows(source: Path) -> dict[str, dict]:
    doc = json.loads((source / "ledger.json").read_text())
    return {r["key"]: r for r in doc["rows"] if isinstance(r.get("key"), str)}


def control2_compare(
    measured: dict, ref: dict[str, dict], arm: str, label: int, seed: int
) -> dict:
    """``measured`` is S0-03 ``measure()``'s output. Every heldout.live bucket x
    readout sample of the reference ledger, this seed, within REPRO_TOL."""
    worst, missing, bad = 0.0, [], []
    for b in LEDGER_BUCKETS:
        for rd in LEDGER_READOUTS:
            key = f"{arm}.ckpt{label}.heldout.live.{b}.{rd}"
            row = ref.get(key)
            mine = measured["heldout"]["live"][b][rd]
            if row is None or len(row.get("samples", [])) <= SEEDS.index(seed):
                missing.append(key)
                continue
            theirs = row["samples"][SEEDS.index(seed)]
            if theirs is None and mine is None:
                continue
            if theirs is None or mine is None:
                bad.append(key)
                continue
            d = abs(float(mine) - float(theirs))
            worst = max(worst, d)
            if d > REPRO_TOL:
                bad.append(key)
    return {
        "ok": not missing and not bad,
        "max_abs_diff": worst,
        "missing": missing,
        "out_of_tolerance": bad,
    }


# --------------------------------------------------------------------------- #
# one (arm, checkpoint, seed): the child's whole job
# --------------------------------------------------------------------------- #


def load_model(ckpt: Path, V: int):
    from rsr.model.tg import TGModel
    from rsr.train import checkpoint as ck

    model = TGModel(S003._cfg(V))
    ck.load(ckpt, model=model, restore_rng=False)
    model.eval()
    return model


def measure_one(
    model,
    docs,
    vmap,
    seed: int,
    *,
    p_ids: set[int] | None = None,
    arms: tuple[str, ...] = ARMS,
    progress: Callable[[int], None] | None = None,
) -> dict:
    """Every document of ``docs`` under every arm, with controls 1, 3 and 4."""
    sym_ids = torch.tensor([vmap[s] for s in ANSWER_SYMBOLS])
    rec: dict[str, list[dict]] = {a: [] for a in arms}
    c1 = {"n_documents": 0, "ok_mismatches": 0, "max_abs_nll_diff": 0.0, "failed": []}
    c4 = {"mismatches": 0, "failed_documents": [], "fifo_not_gap_le_M": 0}
    for i, doc in enumerate(docs):
        for arm in arms:
            pol = DocPolicy(make_inner(arm, doc, seed), doc.doc_id)
            r = run_document(model, doc, vmap, sym_ids, pol)
            mf = model_free_residency(arm, doc, seed)
            if [x["resident"] for x in r] != mf:
                c4["mismatches"] += 1
                c4["failed_documents"].append([arm, doc.doc_id])
            if arm == "fifo":
                c4["fifo_not_gap_le_M"] += sum(
                    x["resident"] != (x["gap"] <= M) for x in r
                )
                c = control1_compare(r, readout_document(model, doc, vmap, sym_ids))
                c1["n_documents"] += 1
                if not control1_ok(c):
                    c1["failed"].append([doc.doc_id, c])
                if c["ok_mismatches"] is not None:
                    c1["ok_mismatches"] += c["ok_mismatches"]
                if c["max_abs_nll_diff"] is not None:
                    c1["max_abs_nll_diff"] = max(
                        c1["max_abs_nll_diff"], c["max_abs_nll_diff"]
                    )
            rec[arm].extend(r)
        if progress is not None:
            progress(i)
    c1["ok"] = not c1["failed"]
    c4["ok"] = c4["mismatches"] == 0 and c4["fifo_not_gap_le_M"] == 0
    return {"records": rec, "control1": c1, "control4": c4}


def batch_diagnostic(model, p_docs, vmap, fifo_records: list[dict]) -> dict:
    """Batch-64 vs B = 1 FIFO on P: argmax flips and NLL max |diff|. Diagnostic."""
    sym_ids = torch.tensor([vmap[s] for s in ANSWER_SYMBOLS])
    ids, mask = encode(list(p_docs), vmap, max_tokens=L, steps=S)
    tm, gap = answer_targets(list(p_docs), vmap, max_tokens=L, steps=S)
    r = S003.answer_readout(model, ids, mask, tm, gap, sym_ids, cond="live")
    keys = []
    for t in range(S):
        for b, _pos in tm[:, t, 1:].nonzero().tolist():
            keys.append((p_docs[b].doc_id, t))
    batched = dict(
        zip(keys, zip(r["ok"].tolist(), r["nll"].tolist(), strict=True), strict=True)
    )
    flips, worst, n = 0, 0.0, 0
    for x in fifo_records:
        k = (x["doc"], x["q"])
        if k in batched:
            n += 1
            flips += bool(batched[k][0]) != x["ok"]
            worst = max(worst, abs(batched[k][1] - x["nll"]))
    return {"n": n, "argmax_flips": flips, "max_abs_nll_diff": worst}


def child(arm: str, label: int, seed: int, source: Path, out: Path) -> int:
    """One (fresh-stream arm, checkpoint, seed). Writes ``out`` (gzip JSON)."""
    torch.set_num_threads(int(os.environ.get("RSR_READABILITY_THREADS", "1")))
    t0 = time.time()
    docs, vmap, V, closure = documents(seed)
    ck = ckpt_path(source, arm, label, seed)
    model = load_model(ck, V)
    ref = reference_rows(source)
    measured = S003.measure(ck, seed, "cpu", sets=CSC.doc_sets(seed, VOCAB_DOCUMENTS))
    c2 = control2_compare(measured, ref, arm, label, seed)
    p_ids = set(range(*P_SET))
    res = measure_one(model, docs, vmap, seed)
    diag = batch_diagnostic(
        model,
        [d for d in docs if d.doc_id in p_ids],
        vmap,
        [x for x in res["records"]["fifo"] if x["doc"] in p_ids],
    )
    payload = {
        "arm": arm,
        "checkpoint": label,
        "seed": seed,
        "ckpt": str(ck),
        "n_documents": len(docs),
        "closure": closure,
        "control1": res["control1"],
        "control2": c2,
        "control4": res["control4"],
        "batch_diagnostic": diag,
        "records": {a: _columns(r) for a, r in res["records"].items()},
        "seconds": time.time() - t0,
        "threads": torch.get_num_threads(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") as f:
        json.dump(payload, f)
    return 0


def _columns(records: list[dict]) -> dict[str, list]:
    keys = (
        "doc",
        "q",
        "a",
        "gap",
        "resident",
        "rank",
        "n_live",
        "ok",
        "nll",
        "nll16",
        "brier16",
    )
    return {k: [r[k] for r in records] for k in keys}


def _rows(cols: dict[str, list]) -> list[dict]:
    keys = list(cols)
    return [
        dict(zip(keys, vals, strict=True)) for vals in zip(*cols.values(), strict=True)
    ]


# --------------------------------------------------------------------------- #
# aggregation: per-document sums, paired bootstrap
# --------------------------------------------------------------------------- #

BUCKETS = (
    "all",
    "gap_2_to_M",
    "gap_eq_M",
    "gap_gt_M",
    "rescued",
    "rescued_rank0",
    "gap_2_to_M_shifted",
)
STATS = ("n", "ok", "nll", "brier16", "resident")


def _in_bucket(r: dict, b: str) -> bool:
    g = r["gap"]
    if b == "all":
        return g >= 1
    if b == "gap_2_to_M":
        return 2 <= g <= M
    if b == "gap_eq_M":
        return g == M
    if b == "gap_gt_M":
        return g > M
    if b == "rescued":
        return g > M and r["resident"]
    if b == "rescued_rank0":
        return g > M and r["resident"] and r["rank"] == 0
    if b == "gap_2_to_M_shifted":
        return 2 <= g <= M and r["resident"] and r["rank"] != M - g
    raise ValueError(b)


def doc_sums(records: list[dict], doc_ids: list[int]) -> torch.Tensor:
    """``[D, len(BUCKETS), len(STATS)]`` float64 per-document sums."""
    pos = {d: i for i, d in enumerate(doc_ids)}
    X = torch.zeros(len(doc_ids), len(BUCKETS), len(STATS), dtype=torch.float64)
    for r in records:
        i = pos[r["doc"]]
        vals = (1.0, float(r["ok"]), r["nll"], r["brier16"], float(r["resident"]))
        for bi, b in enumerate(BUCKETS):
            if _in_bucket(r, b):
                for si, v in enumerate(vals):
                    X[i, bi, si] += v
    return X


def _mean(
    sums: dict[str, torch.Tensor], arm: str, bucket: str, stat: str
) -> torch.Tensor:
    s = sums[arm]
    n = s[..., BUCKETS.index(bucket), STATS.index("n")]
    v = s[..., BUCKETS.index(bucket), STATS.index(stat)]
    return torch.where(n > 0, v / n.clamp(min=1), torch.full_like(n, float("nan")))


def _diff(a_arm, a_b, b_arm, b_b, stat="ok"):
    return lambda s: _mean(s, a_arm, a_b, stat) - _mean(s, b_arm, b_b, stat)


#: The primary and descriptive readouts, each a function of per-arm sums.
READOUTS: dict[str, Callable[[dict], torch.Tensor]] = {
    # (a) rank penalty: positive = the oracle's shifted ranks cost accuracy
    "RP": _diff("fifo", "gap_2_to_M", "oracle", "gap_2_to_M"),
    # (b) rescued-fact readability
    "D_ref": _diff("oracle", "rescued", "fifo", "gap_eq_M"),
    "D_floor": _diff("oracle", "rescued", "fifo", "gap_gt_M"),
    # (c) model-based headroom, descriptive
    "H_model": _diff("oracle", "all", "fifo", "all"),
    "H_ff_model": _diff("factfiller", "all", "fifo", "all"),
    # secondary: rank-matched (b), NLL / Brier versions (lower is better)
    "D_ref_rank0": _diff("oracle", "rescued_rank0", "fifo", "gap_eq_M"),
    "RP_nll": _diff("oracle", "gap_2_to_M", "fifo", "gap_2_to_M", "nll"),
    "RP_brier16": _diff("oracle", "gap_2_to_M", "fifo", "gap_2_to_M", "brier16"),
    "D_ref_nll": _diff("oracle", "rescued", "fifo", "gap_eq_M", "nll"),
    "D_ref_brier16": _diff("oracle", "rescued", "fifo", "gap_eq_M", "brier16"),
}


def bootstrap(
    sums: dict[str, torch.Tensor], seed: int, n_boot: int = N_BOOT
) -> dict[str, dict]:
    """Point and paired percentile 95 % CI of every readout. ONE resampled multiset
    of documents per replicate, used for every arm (PREREG: paired)."""
    D = next(iter(sums.values())).shape[0]
    g = torch.Generator().manual_seed(BOOT_SEED_BASE + seed)
    idx = torch.randint(0, D, (n_boot, D), generator=g)
    W = torch.zeros(n_boot, D, dtype=torch.float64).scatter_add_(
        1, idx, torch.ones(n_boot, D, dtype=torch.float64)
    )
    rep = {a: torch.einsum("rd,dbk->rbk", W, x) for a, x in sums.items()}
    point = {a: x.sum(0) for a, x in sums.items()}
    out = {}
    for name, fn in READOUTS.items():
        p = float(fn(point))
        r = fn(rep)
        finite = r[~torch.isnan(r)]
        if len(finite) == 0:
            lo = hi = float("nan")
        else:
            lo = float(torch.quantile(finite, 0.025))
            hi = float(torch.quantile(finite, 0.975))
        out[name] = {
            "point": p,
            "lo": lo,
            "hi": hi,
            "nan_replicates": int(n_boot - len(finite)),
        }
    return out


def arm_means(sums: dict[str, torch.Tensor]) -> dict[str, dict]:
    point = {a: x.sum(0) for a, x in sums.items()}
    out: dict[str, dict] = {}
    for a in sums:
        for b in BUCKETS:
            n = float(point[a][BUCKETS.index(b), STATS.index("n")])
            out[f"{a}.{b}"] = {
                "n": int(n),
                "acc": float(_mean(point, a, b, "ok")),
                "nll": float(_mean(point, a, b, "nll")),
                "brier16": float(_mean(point, a, b, "brier16")),
                "resident_frac": float(_mean(point, a, b, "resident")),
            }
    return out


def rank_profile(records: list[dict], bucket: str) -> dict[int, dict]:
    """Accuracy by rank index for resident answers in ``bucket``."""
    out: dict[int, list[int]] = {}
    for r in records:
        if r["resident"] and _in_bucket(r, bucket):
            out.setdefault(r["rank"], [0, 0])
            out[r["rank"]][0] += 1
            out[r["rank"]][1] += int(r["ok"])
    return {k: {"n": n, "acc": c / n} for k, (n, c) in sorted(out.items())}


def model_free_hits(records: dict[str, list[dict]]) -> dict[str, float]:
    return {a: sum(r["resident"] for r in rs) / len(rs) for a, rs in records.items()}


# --------------------------------------------------------------------------- #
# 🔒 the rule (PREREG "Primary readouts"), transcribed
# --------------------------------------------------------------------------- #


def classify_a(per_seed: dict[int, dict]) -> str:
    """``per_seed[s] = {"point", "lo", "hi"}`` of RP."""
    if all(v["hi"] < RP_MAX for v in per_seed.values()):
        return "NO_PENALTY"
    if all(v["point"] >= RP_MAX and v["lo"] > 0 for v in per_seed.values()):
        return "PENALTY"
    return "MIXED_PENALTY"


def classify_b(d_ref: dict[int, dict], d_floor: dict[int, dict]) -> str:
    readable = all(v["lo"] > -READ_MARGIN for v in d_ref.values())
    unreadable = all(v["hi"] < READ_MARGIN for v in d_floor.values())
    if readable and unreadable:
        return "UNDISTINGUISHED"
    if readable:
        return "READABLE"
    if unreadable:
        return "UNREADABLE"
    return "MIXED_READ"


def combine(a: str, b: str) -> str:
    if a == "PENALTY" or b == "UNREADABLE":
        return "RANK_BOUND"
    if a == "NO_PENALTY" and b == "READABLE":
        return "READABLE_AT_SHIFTED_RANK"
    return "MIXED"


def a_null_ok(h_model: dict[int, dict]) -> bool:
    """Control 6: arm A's oracle - FIFO CI inside (-A_NULL, +A_NULL), every seed."""
    return all(v["lo"] > -A_NULL and v["hi"] < A_NULL for v in h_model.values())


OUTCOME = {
    "READABLE_AT_SHIFTED_RANK": "survived",
    "RANK_BOUND": "falsified",
    "MIXED": "inconclusive",
    "inconclusive": "inconclusive",
}


def decide(
    results: dict[tuple[str, int], dict[int, dict]], controls_failed: list[str]
) -> dict:
    """``results[(arm, label)][seed]`` = that child's readouts (``bootstrap`` output).
    -> per-checkpoint classes and the headline."""
    failed = list(controls_failed)
    missing = [
        f"{a}.ckpt{c}.seed{s}"
        for a, c in CHECKPOINTS
        for s in SEEDS
        if s not in results.get((a, c), {})
    ]
    if missing:
        failed.append(f"missing measurements: {missing}")
    null = None
    if not missing:
        null = {s: results[NULL_ARM][s]["H_model"] for s in SEEDS}
        if not a_null_ok(null):
            failed.append(f"control 6 (arm A null) failed: {null}")
    per_ckpt = {}
    for a, c in CHECKPOINTS:
        if (a, c) == NULL_ARM or any(s not in results.get((a, c), {}) for s in SEEDS):
            continue
        r = results[(a, c)]
        ca = classify_a({s: r[s]["RP"] for s in SEEDS})
        cb = classify_b(
            {s: r[s]["D_ref"] for s in SEEDS}, {s: r[s]["D_floor"] for s in SEEDS}
        )
        per_ckpt[f"{a}.ckpt{c}"] = {"a": ca, "b": cb, "combined": combine(ca, cb)}
    if failed:
        headline = "inconclusive"
    else:
        headline = per_ckpt[f"{HEADLINE[0]}.ckpt{HEADLINE[1]}"]["combined"]
    combos = {v["combined"] for v in per_ckpt.values()}
    return {
        "classification": headline,
        "per_checkpoint": per_ckpt,
        "moving": len(combos) > 1,
        "controls_failed": failed,
        "a_null": null,
        "outcome": OUTCOME[headline],
        "exit": int(Exit.DID_NOT_RUN if failed else Exit.OK),
    }


# --------------------------------------------------------------------------- #
# the parent
# --------------------------------------------------------------------------- #


def raw_path(root: Path, arm: str, label: int, seed: int) -> Path:
    return root / "raw" / f"{arm}-ckpt{label}-seed{seed}.json.gz"


def child_argv(arm: str, label: int, seed: int, source: Path, out: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single",
        "--arm",
        arm,
        "--checkpoint",
        str(label),
        "--seed",
        str(seed),
        "--source",
        str(source),
        "--out",
        str(out),
    ]


def run_children(source: Path, root: Path, parallel: int = 9) -> dict[tuple, int]:
    """Every (arm, checkpoint, seed) as its own child process; rc read directly."""
    jobs = [(a, c, s) for a, c in CHECKPOINTS for s in SEEDS]
    rcs: dict[tuple, int] = {}
    running: dict[tuple, subprocess.Popen] = {}
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    while jobs or running:
        while jobs and len(running) < parallel:
            j = jobs.pop(0)
            a, c, s = j
            log = open(logs / f"{a}-ckpt{c}-seed{s}.log", "w")  # noqa: SIM115
            running[j] = subprocess.Popen(
                child_argv(a, c, s, source, raw_path(root, a, c, s)),
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


def summarise_child(payload: dict) -> dict:
    """Readouts of one child's raw records."""
    seed = payload["seed"]
    recs = {a: _rows(c) for a, c in payload["records"].items()}
    doc_ids = sorted({r["doc"] for r in recs["fifo"]})
    p_ids = [d for d in doc_ids if P_SET[0] <= d < P_SET[1]]
    e_ids = [d for d in doc_ids if E_SET[0] <= d < E_SET[1]]
    out: dict[str, Any] = {}
    for name, ids in (("U", doc_ids), ("P", p_ids), ("E", e_ids)):
        keep = set(ids)
        sums = {
            a: doc_sums([r for r in rs if r["doc"] in keep], ids)
            for a, rs in recs.items()
        }
        out[name] = {
            "readouts": bootstrap(sums, seed),
            "means": arm_means(sums),
            "model_free_hit": model_free_hits(
                {a: [r for r in rs if r["doc"] in keep] for a, rs in recs.items()}
            ),
            "n_documents": len(ids),
        }
    out["rank_profile"] = {
        a: {b: rank_profile(rs, b) for b in ("gap_2_to_M", "rescued")}
        for a, rs in recs.items()
    }
    return out


def execute(led, source: Path, root: Path) -> Exit:
    disj = disjointness()
    led.note("control5.disjointness", disj, how="run.py::disjointness")
    closures = {}
    for s in SEEDS:
        closures[s] = documents(s)[3]
    led.note("control5.closure", closures, how="run.py::documents (vocab of [0, 64))")
    controls_failed: list[str] = []
    if not disj["ok"] or not all(c["ok"] for c in closures.values()):
        controls_failed.append("control 5 (documents) failed")
        return _finish(led, root, {}, controls_failed, {})
    missing = [
        str(ckpt_path(source, a, c, s))
        for a, c in CHECKPOINTS
        for s in SEEDS
        if not ckpt_path(source, a, c, s).exists()
    ]
    if missing:
        controls_failed.append(f"checkpoints missing: {missing}")
        return _finish(led, root, {}, controls_failed, {})

    rcs = run_children(source, root)
    results: dict[tuple, dict] = {}
    summaries: dict[str, dict] = {}
    for (a, c, s), rc in sorted(rcs.items()):
        led.command(child_argv(a, c, s, source, raw_path(root, a, c, s)), exit_code=rc)
        if rc != 0:
            controls_failed.append(f"child {a}.ckpt{c}.seed{s} exited {rc}")
            continue
        with gzip.open(raw_path(root, a, c, s), "rt") as f:
            payload = json.load(f)
        for k in ("control1", "control2", "control4"):
            if not payload[k]["ok"]:
                controls_failed.append(f"{k} failed on {a}.ckpt{c}.seed{s}")
        summ = summarise_child(payload)
        summ["controls"] = {k: payload[k] for k in ("control1", "control2", "control4")}
        summ["batch_diagnostic"] = payload["batch_diagnostic"]
        summ["seconds"] = payload["seconds"]
        summaries[f"{a}.ckpt{c}.seed{s}"] = summ
        results.setdefault((a, c), {})[s] = summ["U"]["readouts"]
    return _finish(led, root, results, controls_failed, summaries)


def _finish(led, root: Path, results, controls_failed, summaries) -> Exit:
    dec = decide(results, controls_failed)
    write_rows(led, results, summaries, dec)
    led.run_meta(
        seeds_actually_run=sorted({int(k.split("seed")[1]) for k in summaries}) or [],
    )
    led.status("ok" if not dec["controls_failed"] else "partial")
    led.verdict(
        falsifier=FALSIFIER,
        outcome=dec["outcome"],
        detail=(
            f"classification {dec['classification']}; per checkpoint "
            f"{json.dumps(dec['per_checkpoint'], sort_keys=True)}; moving "
            f"{dec['moving']}; controls failed {dec['controls_failed']}"
        ),
    )
    (root / "summaries.json").write_text(json.dumps(summaries, indent=1, default=str))
    return Exit(dec["exit"])


def write_rows(led, results, summaries, dec) -> None:
    led.note("classification", dec["classification"], how="run.py::decide")
    led.note("per_checkpoint", dec["per_checkpoint"], how="run.py::decide")
    led.note("moving", dec["moving"], how="run.py::decide")
    led.note("controls_failed", dec["controls_failed"], how="run.py::decide")
    how = "run.py::summarise_child -> bootstrap (paired, per document); seed order"
    for (a, c), per in sorted(results.items()):
        if any(s not in per for s in SEEDS):
            continue
        for name in READOUTS:
            for f in ("point", "lo", "hi"):
                led.stat(
                    f"{a}.ckpt{c}.U.{name}.{f}", [per[s][name][f] for s in SEEDS], how=how
                )
    for a, c in sorted(results):
        keys = [
            f"{a}.ckpt{c}.seed{s}" for s in SEEDS if f"{a}.ckpt{c}.seed{s}" in summaries
        ]
        if len(keys) != len(SEEDS):
            continue
        for setname in ("U", "P", "E"):
            for mkey in summaries[keys[0]][setname]["means"]:
                for f in ("acc", "nll", "brier16", "resident_frac", "n"):
                    led.stat(
                        f"{a}.ckpt{c}.{setname}.{mkey}.{f}",
                        [summaries[k][setname]["means"][mkey][f] for k in keys],
                        how=f"run.py::arm_means, set {setname}; seed order",
                    )
            for arm in ARMS:
                led.stat(
                    f"{a}.ckpt{c}.{setname}.model_free_hit.{arm}",
                    [summaries[k][setname]["model_free_hit"][arm] for k in keys],
                    how="harness residency (== simulate, control 4); seed order",
                )
            led.stat(
                f"{a}.ckpt{c}.{setname}.H_free",
                [
                    summaries[k][setname]["model_free_hit"]["oracle"]
                    - summaries[k][setname]["model_free_hit"]["fifo"]
                    for k in keys
                ],
                how="model-free oracle hit - FIFO hit on this set; seed order",
            )
            if setname != "U":
                for name in ("RP", "D_ref", "D_floor", "H_model"):
                    for f in ("point", "lo", "hi"):
                        led.stat(
                            f"{a}.ckpt{c}.{setname}.{name}.{f}",
                            [summaries[k][setname]["readouts"][name][f] for k in keys],
                            how=f"bootstrap on set {setname}; seed order",
                        )
    for k, summ in sorted(summaries.items()):
        led.note(f"{k}.controls", summ["controls"], how="child payload")
        led.note(f"{k}.batch_diagnostic", summ["batch_diagnostic"], how="child payload")
        led.note(f"{k}.rank_profile", summ["rank_profile"], how="run.py::rank_profile")
        led.note(f"{k}.seconds", summ["seconds"], how="child wall clock")


def manifest_config(source: Path) -> dict:
    return {
        "run_id": RUN_ID,
        "prereg": PREREG,
        "prereg_commit": PREREG_COMMIT,
        "question": QUESTION,
        "falsifier": FALSIFIER,
        "expected": EXPECTED,
        "seeds": SEEDS,
        "checkpoints": [f"{a}.ckpt{c}" for a, c in CHECKPOINTS],
        "arms": list(ARMS),
        "gamma": GAMMA,
        "M": M,
        "S": S,
        "sets": {"P": list(P_SET), "E": list(E_SET), "U": "E then P"},
        "thresholds": {
            "RP_MAX": RP_MAX,
            "READ_MARGIN": READ_MARGIN,
            "A_NULL": A_NULL,
            "N_BOOT": N_BOOT,
            "BOOT_SEED_BASE": BOOT_SEED_BASE,
            "REPRO_TOL": REPRO_TOL,
            "CONTROL1_TOL": CONTROL1_TOL,
        },
        "source": str(source),
        "source_ledger_sha256": _sha256(source / "ledger.json"),
        "checkpoint_sha256": {
            f"{a}.ckpt{c}.seed{s}": _sha256(ckpt_path(source, a, c, s))
            for a, c in CHECKPOINTS
            for s in SEEDS
        },
        "device": "cpu",
        "threads_per_child": int(os.environ.get("RSR_READABILITY_THREADS", "1")),
    }


def _sha256(p: Path) -> str | None:
    import hashlib

    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# RESULTS.md, rendered from the ledger
# --------------------------------------------------------------------------- #


def render_results(ledger_doc: dict) -> str:
    rows = {r["key"]: r for r in ledger_doc["rows"]}
    prov = ledger_doc["provenance"]

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
        "# RESULTS — retention readability",
        "",
        f"Rendered from `runs/{RUN_ID}/ledger.json` by `{EXPERIMENT} --render-results`. "
        f"PREREG `{PREREG}` (commit `{PREREG_COMMIT}`). Run sha `{prov.get('git_sha')}`, "
        f"platform `{prov.get('platform')}`, device `cpu`.",
        "",
        "Command: `uv run python experiments/retention-readability/run.py` (children "
        "listed under `commands` in the ledger).",
        "",
        f"**classification: {rows['classification']['value']}** (headline "
        f"B.ckpt3000); verdict outcome `{ledger_doc['verdict']['outcome']}`; moving "
        f"`{rows['moving']['value']}`.",
        "",
        "Per checkpoint (`per_checkpoint`):",
        "",
    ]
    for k, v in rows["per_checkpoint"]["value"].items():
        out.append(f"- `{k}`: (a) {v['a']}, (b) {v['b']}, combined {v['combined']}")
    out += [
        "",
        "Controls failed (`controls_failed`): "
        f"{rows['controls_failed']['value'] or 'none'}",
        "",
    ]
    for a, c in CHECKPOINTS:
        pre = f"{a}.ckpt{c}.U"
        if f"{pre}.RP.point" not in rows:
            continue
        out += [
            f"## {a}.ckpt{c}, set U (point [percentile CI], paired per-document bootstrap)",
            "",
            "| readout | seed0 | seed1 | seed2 |",
            "|---|---|---|---|",
        ]
        for name in READOUTS:
            out.append(
                f"| `{pre}.{name}` | "
                + " | ".join(ci(pre + "." + name, i) for i in range(3))
                + " |"
            )
        out += ["", "| accuracy | seed0 | seed1 | seed2 |", "|---|---|---|---|"]
        for arm in ARMS:
            for b in BUCKETS:
                key = f"{pre}.{arm}.{b}.acc"
                out.append(f"| `{key}` | {st(key, 0)} | {st(key, 1)} | {st(key, 2)} |")
        out += ["", "| model-free | seed0 | seed1 | seed2 |", "|---|---|---|---|"]
        for arm in ARMS:
            key = f"{pre}.model_free_hit.{arm}"
            out.append(f"| `{key}` | {st(key, 0)} | {st(key, 1)} | {st(key, 2)} |")
        key = f"{pre}.H_free"
        out.append(f"| `{key}` | {st(key, 0)} | {st(key, 1)} | {st(key, 2)} |")
        out.append("")
        for sub in ("P", "E"):
            p2 = f"{a}.ckpt{c}.{sub}"
            out += [
                f"Set {sub}:",
                "",
                "| readout | seed0 | seed1 | seed2 |",
                "|---|---|---|---|",
            ]
            for name in ("RP", "D_ref", "D_floor", "H_model"):
                out.append(
                    f"| `{p2}.{name}` | "
                    + " | ".join(ci(p2 + "." + name, i) for i in range(3))
                    + " |"
                )
            out.append("")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> Exit:
    ap = ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--arm", choices=["A", "B"])
    ap.add_argument("--checkpoint", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--source", type=Path, default=None)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--render-results", action="store_true")
    ap.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    a = ap.parse_args(argv)
    source = a.source or default_source()
    if a.single:
        return status(child(a.arm, a.checkpoint, a.seed, source, a.out))
    root = a.runs_root / RUN_ID
    if a.render_results:
        doc = json.loads((root / "ledger.json").read_text())
        (ROOT / "experiments" / "retention-readability" / "RESULTS.md").write_text(
            render_results(doc)
        )
        return Exit.OK
    if a.dry_run:
        print(json.dumps(disjointness()))
        for s in SEEDS:
            print(s, documents(s)[3])
        for x, c in CHECKPOINTS:
            for s in SEEDS:
                p = ckpt_path(source, x, c, s)
                print(p, p.exists())
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
