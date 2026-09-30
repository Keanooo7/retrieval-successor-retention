"""B0: zero-cost ceilings (PLAN-v4 §2B B0) -- descriptive, no gate.

Pre-registration: `experiments/b0-ceilings/PREREG.md`, committed alone ahead of this
file (fc26dbf). **Model-free**: it loads no checkpoint and runs no forward pass. It
reads only the lookahead-room-r2 demand tensors (`D.pt`, W10's demand-if-resident,
§3.2.1 gated and fill-rescaled) on the **already-inspected** U range, and never calls
`rsr.constants.record()`.

Rules (PREREG §4), each scored by model-free residency (`rsr.metrics.headroom.simulate`):

* ``ko``        -- kind-oracle, argmin ``E[G_gamma | kind]``; B2 A1.10's random tie rule;
* ``ko_oldest`` -- the same, ties to the oldest (secondary, as B2 A1.10);
* ``kb``        -- kind x age-band, argmin ``E[G_gamma | kind, band]``. **NOT a legal
  psi_hat** (it reads age; CLAUDE.md "Age is excluded from psi_hat") -- a reference;
* ``age``       -- the age-only rule, argmin ``E[G_gamma | age]``.

``G_gamma[t][i] = sum_{k>=0} gamma^k D[t+k][i]`` (§3.4, correction 2), truncated at
document end. Class means are **in-sample** on U (declared).

Usage::

    uv run python experiments/b0-ceilings/run.py            # the run
    uv run python experiments/b0-ceilings/run.py --dry-run  # controls C1/C2 only

Exit codes (`rsr.exit_codes`): 0 ran, every control passed · 1 a defect after the
controls passed · 3 did not run / a control failed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
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
from rsr.data.synthetic import discounted_demand  # noqa: E402
from rsr.exit_codes import ArgumentParser, Exit, run_main, status  # noqa: E402
from rsr.metrics.headroom import simulate  # noqa: E402

EXPERIMENT = "experiments/b0-ceilings/run.py"
RUN_ID = "b0-ceilings"
PREREG = "experiments/b0-ceilings/PREREG.md"
PREREG_COMMIT = "fc26dbf"


def _load(name: str, rel: str):
    """Import an experiment script by path. Imported, never copied."""
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LR = _load("_b0_lookahead_room", "experiments/lookahead-room/run.py")

# --------------------------------------------------------------------------- #
# 🔒 PREREG.md, transcribed. Changing any is changing the PREREG.
# --------------------------------------------------------------------------- #

SEEDS = [0, 1, 2]
CHECKPOINTS = (2500, 3000)
HEADLINE = 3000
GAMMAS = {"g09": 0.9, "g0": 0.0}
M = LR.M
S = LR.S
KINDS = ("assert", "query", "filler")
CLASSES = LR.CLASSES  # pending, querying, answered, filler
#: PREREG §3, fixed before any number: inclusive [lo, hi] ages.
AGE_BANDS = ((1, 4), (5, 8), (9, 12), (13, 16), (17, 24), (25, 47))
RULES = ("ko", "ko_oldest", "kb", "age")
HINDSIGHT = ("rule_g0", "rule_g09")
REFS = ("fifo", "factfiller", "oracle")
N_BOOT = 2000
BOOT_SEED_BASE = 20260927
MANIFEST = Path.home() / "rsr-substrate" / "2026-09-27" / "MANIFEST.sha256"
D_REL = ".worktrees/lookahead-room/runs/lookahead-room-r2/raw/B-ckpt{c}-seed{s}.D.pt"
REF_LEDGER = "runs/lookahead-room-r2/ledger.json"
#: The rules C3 reproduces against the reference ledger (PREREG §6).
C3_RULES = ("fifo", "factfiller", "oracle", "rule_g0", "rule_g09")

QUESTION = (
    "On the already-inspected U range, what do the kind-oracle argmin E[G|kind], the "
    "kind x age-band reference (not a legal psi_hat) and the age-only rule reach as "
    "model-free residency, against FIFO and fact/filler; and the F1/F2 tables."
)
FALSIFIER = "NONE: descriptive (PREREG §1); it feeds only B2's already-seen addendum"
EXPECTED = (
    "PREREG §8: ko cap ~0.8-1.1 on seeds 0 and 2, negative on seed 1 (ckpt3000, "
    "g0.9; same signs at g0); age cap in [-0.5, 0.5]; kb >= ko on seed 1; C3 exact."
)


# --------------------------------------------------------------------------- #
# provenance of the inputs: the T0 manifest (C1)
# --------------------------------------------------------------------------- #


def main_checkout() -> Path:
    """The main checkout (the manifest's paths are relative to it)."""
    git = "/opt/homebrew/bin/git" if Path("/opt/homebrew/bin/git").exists() else "git"
    out = subprocess.run(
        [git, "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return Path(out).parent


def d_rel(c: int, s: int) -> str:
    return D_REL.format(c=c, s=s)


def read_manifest(path: Path) -> dict[str, str]:
    """``shasum -a 256`` format: ``<hex>  <relative path>`` per line."""
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        h, rel = line.split(None, 1)
        out[rel.strip()] = h
    return out


def sha256(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(manifest: Path, rels: list[str], base: Path) -> dict:
    """C1: every file in ``rels`` has a manifest line and its sha256 matches it."""
    if not manifest.exists():
        return {"ok": False, "reason": f"manifest missing: {manifest}", "files": {}}
    want = read_manifest(manifest)
    files = {}
    for rel in rels:
        got = sha256(base / rel)
        files[rel] = {"want": want.get(rel), "got": got}
    ok = all(v["want"] is not None and v["got"] == v["want"] for v in files.values())
    return {"ok": ok, "manifest": str(manifest), "files": files}


# --------------------------------------------------------------------------- #
# the pure pieces (tests/test_b0_ceilings.py drives each on hand-built inputs)
# --------------------------------------------------------------------------- #


def age_band(age: int) -> int:
    """Index into ``AGE_BANDS`` of an age in [1, S-1]; raises outside it."""
    for b, (lo, hi) in enumerate(AGE_BANDS):
        if lo <= age <= hi:
            return b
    raise ValueError(f"age {age} is outside every band {AGE_BANDS}")


def row_mask(n: int, m: int = M) -> torch.Tensor:
    """``[n, n]`` bool, step-major: rows ``(t, i)``, ``t in [m, n)``, ``i < t``."""
    t = torch.arange(n).unsqueeze(1)
    i = torch.arange(n).unsqueeze(0)
    return (t >= m) & (i < t)


def check_shape(D: torch.Tensor, n: int = S) -> bool:
    """C2 for one document: ``[n, n]``, finite at ``i < t``, NaN at ``i >= t``."""
    if tuple(D.shape) != (n, n):
        return False
    t = torch.arange(n).unsqueeze(1)
    i = torch.arange(n).unsqueeze(0)
    past = i < t
    return bool(torch.isfinite(D[past]).all()) and bool(torch.isnan(D[~past]).all())


def new_sums() -> dict:
    return {
        "kind": {k: [0.0, 0] for k in KINDS},
        "kb": {(k, b): [0.0, 0] for k in KINDS for b in range(len(AGE_BANDS))},
        "age": {a: [0.0, 0] for a in range(1, S)},
    }


def accumulate(sums: dict, doc, G: torch.Tensor, m: int = M) -> tuple:
    """Add one document's rows to ``sums`` (in place). Returns its
    ``(sum G | assert, n assert, sum G | filler, n filler)`` for the F2 bootstrap."""
    n = G.shape[0]
    rows = row_mask(n, m).nonzero().tolist()
    Gl = G.tolist()
    kinds = [s.kind for s in doc.sentences]
    per = {"assert": [0.0, 0], "filler": [0.0, 0]}
    for t, i in rows:
        g = float(Gl[t][i])
        k = kinds[i]
        a = t - i
        sums["kind"][k][0] += g
        sums["kind"][k][1] += 1
        cell = sums["kb"][(k, age_band(a))]
        cell[0] += g
        cell[1] += 1
        sums["age"][a][0] += g
        sums["age"][a][1] += 1
        if k in per:
            per[k][0] += g
            per[k][1] += 1
    return (per["assert"][0], per["assert"][1], per["filler"][0], per["filler"][1])


def means(sums: dict) -> dict:
    """Class means from ``sums``; an empty cell is ``None``."""

    def mu(v):
        return v[0] / v[1] if v[1] else None

    return {
        "kind": {k: mu(v) for k, v in sums["kind"].items()},
        "kb": {kb: mu(v) for kb, v in sums["kb"].items()},
        "age": {a: mu(v) for a, v in sums["age"].items()},
        "n_kind": {k: v[1] for k, v in sums["kind"].items()},
        "n_kb": {kb: v[1] for kb, v in sums["kb"].items()},
        "n_age": {a: v[1] for a, v in sums["age"].items()},
    }


def tie_break(tied: list[int], seed: int, doc_id: int, t: int, tie: str) -> int:
    """B2 A1.10: uniformly at random with ``random.Random(f"ko:{seed}:{doc_id}:{t}")``
    over the tied slot indices in ascending order. ``tie="oldest"``: the lowest slot
    index, as ``OraclePolicy``."""
    ks = sorted(tied)
    if tie == "oldest":
        return ks[0]
    if tie != "random":
        raise ValueError(f"tie={tie!r}")
    return random.Random(f"ko:{seed}:{doc_id}:{t}").choice(ks)


class ScorePolicy:
    """Evict the live slot with the smallest ``score(i, t)`` (``i`` its sentence);
    ties by :func:`tie_break`. Stateless across steps."""

    name = "score"

    def __init__(self, score, *, seed: int, doc_id: int, tie: str) -> None:
        self.score = score
        self.seed = seed
        self.doc_id = doc_id
        self.tie = tie

    def select_eviction(self, slots, context, step: int) -> int:
        live = [k for k in range(len(slots.live)) if bool(slots.live[k])]
        vals = {k: self.score(int(slots.written_at[k]), step) for k in live}
        if any(v is None or v != v for v in vals.values()):
            raise ValueError(f"step {step}: undefined score for a live slot")
        lo = min(vals.values())
        tied = [k for k, v in vals.items() if v == lo]
        return tie_break(tied, self.seed, self.doc_id, step, self.tie)

    def observe(self, slots, attn, step: int) -> None:
        return None

    def on_write(self, slots, slot: int, step: int) -> None:
        return None

    def reset(self) -> None:
        return None


class Fallbacks:
    """Counts the ``kb`` decisions that scored a slot from an empty cell."""

    def __init__(self) -> None:
        self.n = 0


def scorers(doc, mu: dict, fb: Fallbacks | None = None) -> dict:
    """``{rule: score(i, t)}`` for ko / kb / age from one (seed, ckpt, gamma)'s means."""
    kinds = [s.kind for s in doc.sentences]

    def ko(i, t):
        return mu["kind"][kinds[i]]

    def kb(i, t):
        v = mu["kb"][(kinds[i], age_band(t - i))]
        if v is None:
            if fb is not None:
                fb.n += 1
            return mu["kind"][kinds[i]]
        return v

    def age(i, t):
        return mu["age"][t - i]

    return {"ko": ko, "kb": kb, "age": age}


def rule_policies(doc, mu: dict, seed: int, fb: Fallbacks | None = None) -> dict:
    sc = scorers(doc, mu, fb)
    mk = dict(seed=seed, doc_id=doc.doc_id)
    return {
        "ko": ScorePolicy(sc["ko"], tie="random", **mk),
        "ko_oldest": ScorePolicy(sc["ko"], tie="oldest", **mk),
        "kb": ScorePolicy(sc["kb"], tie="random", **mk),
        "age": ScorePolicy(sc["age"], tie="random", **mk),
    }


def pairs_of(doc, policy, m: int = M) -> list[tuple[int, bool]]:
    return [(q["gap"], q["hit"]) for q in simulate(doc, policy, m)["queries"]]


BUCKETS = {
    "all": lambda g: True,
    "gap_2_to_M": lambda g: 2 <= g <= M,
    "gap_gt_M": lambda g: g > M,
}


def rate(pairs: list, pred=lambda g: True) -> tuple[int, int]:
    sel = [h for g, h in pairs if pred(g)]
    return len(sel), sum(sel)


def pooled(per_doc: list[tuple[int, int]]) -> float:
    n = sum(x[0] for x in per_doc)
    h = sum(x[1] for x in per_doc)
    return h / n if n else float("nan")


def cap_point(hit: float, fifo: float, ff: float) -> float:
    """``(hit - FIFO) / (factfiller - FIFO)``; NaN if the denominator is <= 0."""
    den = ff - fifo
    return (hit - fifo) / den if den > 0 else float("nan")


def boot_weights(n_docs: int, seed: int, n_boot: int = N_BOOT) -> torch.Tensor:
    """``[n_boot, n_docs]`` multiplicities; one resampled multiset per replicate."""
    g = torch.Generator().manual_seed(BOOT_SEED_BASE + seed)
    idx = torch.randint(0, n_docs, (n_boot, n_docs), generator=g)
    return torch.zeros(n_boot, n_docs, dtype=torch.float64).scatter_add_(
        1, idx, torch.ones(n_boot, n_docs, dtype=torch.float64)
    )


def _q(x: torch.Tensor) -> tuple[float, float]:
    return float(torch.quantile(x, 0.025)), float(torch.quantile(x, 0.975))


def boot_rule(
    W: torch.Tensor,
    rule: list[tuple[int, int]],
    fifo: list[tuple[int, int]],
    ff: list[tuple[int, int]],
) -> dict:
    """Paired per-document bootstrap of ``hit - FIFO`` and of the cap."""

    def rep(x):
        v = W @ torch.tensor(x, dtype=torch.float64).reshape(-1, 2)
        return v[:, 1] / v[:, 0]

    hr, hf, hff = rep(rule), rep(fifo), rep(ff)
    diff = hr - hf
    den = hff - hf
    ok = den > 0
    cap = (diff[ok]) / den[ok]
    d_lo, d_hi = _q(diff)
    c_lo, c_hi = _q(cap) if int(ok.sum()) > 0 else (float("nan"), float("nan"))
    p_r, p_f, p_ff = pooled(rule), pooled(fifo), pooled(ff)
    return {
        "hit": p_r,
        "diff_fifo": {"point": p_r - p_f, "lo": d_lo, "hi": d_hi},
        "cap": {"point": cap_point(p_r, p_f, p_ff), "lo": c_lo, "hi": c_hi},
        "cap_dropped": int((~ok).sum()),
    }


def boot_f2(W: torch.Tensor, per_doc: list[tuple[float, int, float, int]]) -> dict:
    """``E[G | assert] - E[G | filler]``: ratio of pooled sums, paired over documents."""
    X = torch.tensor(per_doc, dtype=torch.float64).reshape(-1, 4)
    R = W @ X
    d = R[:, 0] / R[:, 1] - R[:, 2] / R[:, 3]
    P = X.sum(0)
    lo, hi = _q(d)
    return {"point": float(P[0] / P[1] - P[2] / P[3]), "lo": lo, "hi": hi}


def f1_accumulate(acc: dict, doc, D: torch.Tensor, m: int = M) -> None:
    """F1: ``D[t][i]`` by (age, class) over rows (in place)."""
    Dl = D.tolist()
    for t, i in row_mask(D.shape[0], m).nonzero().tolist():
        c, _ = LR.slot_class(doc, i, t)
        a = t - i
        v = float(Dl[t][i])
        for key in ((a, c), ("age_le_M" if a <= m else "age_gt_M", c)):
            cell = acc.setdefault(key, [0.0, 0])
            cell[0] += v
            cell[1] += 1


def class_G(acc: dict, doc, G: torch.Tensor, m: int = M) -> None:
    """F2: ``G[t][i]`` by the four classes over rows (in place)."""
    Gl = G.tolist()
    for t, i in row_mask(G.shape[0], m).nonzero().tolist():
        c, _ = LR.slot_class(doc, i, t)
        cell = acc.setdefault(c, [0.0, 0])
        cell[0] += float(Gl[t][i])
        cell[1] += 1


def c3_failures(mine: dict[str, float], ref_rows: dict, c: int, seed: int) -> list[str]:
    """C3: ``mine[<rule>[.gap_gt_M]]`` == the reference ledger's sample, exactly."""
    bad = []
    for rule in C3_RULES:
        for suf in ("", ".gap_gt_M"):
            key = f"B.ckpt{c}.U.hit.{rule}{suf}"
            row = ref_rows.get(key)
            if row is None:
                bad.append(f"{key}: absent from {REF_LEDGER}")
                continue
            theirs = row["samples"][SEEDS.index(seed)]
            got = mine.get(f"{rule}{suf}")
            if got is None or float(got) != float(theirs):
                bad.append(f"{key}: mine {got!r} != ledger {theirs!r}")
    return bad


# --------------------------------------------------------------------------- #
# one seed: everything
# --------------------------------------------------------------------------- #


def measure_seed(seed: int, Ds: dict[int, dict], limit: int | None = None) -> dict:
    """``Ds[c] = {doc_id: D}``. -> per-(ckpt, gamma) rule results, refs, F1, F2."""
    docs, _vmap, _V, closure = LR.documents(seed)
    if limit is not None:
        docs = docs[:limit]
    ids = [d.doc_id for d in docs]
    W = boot_weights(len(docs), seed)
    ref_pd = {r: [] for r in REFS}
    for doc in docs:
        pols = {
            "fifo": FIFOPolicy(),
            "oracle": OraclePolicy(discounted_demand(doc, LR.ORACLE_GAMMA)),
            "factfiller": LR.FactFiller(doc, random.Random(f"ff:{seed}:{doc.doc_id}")),
        }
        for r, p in pols.items():
            ref_pd[r].append(pairs_of(doc, p))
    refs = {
        r: {b: pooled([rate(x, f) for x in ref_pd[r]]) for b, f in BUCKETS.items()}
        for r in REFS
    }
    fifo_all = [rate(x) for x in ref_pd["fifo"]]
    ff_all = [rate(x) for x in ref_pd["factfiller"]]
    out: dict[str, Any] = {
        "n_documents": len(docs),
        "doc_ids_first_last": [ids[0], ids[-1]],
        "closure": closure,
        "refs": refs,
        "oracle": boot_rule(W, [rate(x) for x in ref_pd["oracle"]], fifo_all, ff_all),
        "ckpt": {},
    }
    for c, Dc in Ds.items():
        co: dict[str, Any] = {"gammas": {}, "hindsight": {}, "c3_mine": {}}
        # the hindsight rules (lookahead-room's, on D.pt): C3's other half
        T_pd = {h: [] for h in HINDSIGHT}
        f1: dict = {}
        for doc in docs:
            D = Dc[doc.doc_id]
            T = LR.targets(D, torch.zeros_like(D, dtype=torch.bool))
            for h in HINDSIGHT:
                T_pd[h].append(pairs_of(doc, LR.TargetPolicy(T[h])))
            f1_accumulate(f1, doc, D)
        for r, pd in {**ref_pd, **T_pd}.items():
            co["c3_mine"][r] = pooled([rate(x) for x in pd])
            co["c3_mine"][f"{r}.gap_gt_M"] = pooled(
                [rate(x, BUCKETS["gap_gt_M"]) for x in pd]
            )
        for h in HINDSIGHT:
            co["hindsight"][h] = boot_rule(
                W, [rate(x) for x in T_pd[h]], fifo_all, ff_all
            )
        co["F1"] = {
            f"{k[0]}|{k[1]}": {"mean": v[0] / v[1], "n": v[1]} for k, v in f1.items()
        }
        for gname, gamma in GAMMAS.items():
            sums = new_sums()
            f2_pd, cls = [], {}
            Gs = {}
            for doc in docs:
                G = LR.discounted_returns(Dc[doc.doc_id], gamma)
                Gs[doc.doc_id] = G
                f2_pd.append(accumulate(sums, doc, G))
                class_G(cls, doc, G)
            mu = means(sums)
            fb = Fallbacks()
            rule_pd = {r: [] for r in RULES}
            for doc in docs:
                for r, p in rule_policies(doc, mu, seed, fb).items():
                    rule_pd[r].append(pairs_of(doc, p))
            go: dict[str, Any] = {
                "means": {
                    "kind": mu["kind"],
                    "n_kind": mu["n_kind"],
                    "kb": {f"{k}|A{b + 1}": v for (k, b), v in mu["kb"].items()},
                    "n_kb": {f"{k}|A{b + 1}": v for (k, b), v in mu["n_kb"].items()},
                    "age": {str(a): v for a, v in mu["age"].items()},
                    "n_age": {str(a): v for a, v in mu["n_age"].items()},
                },
                "kb_empty_cells": sum(1 for v in mu["kb"].values() if v is None),
                "kb_fallback_decisions": fb.n,
                "E_G_class": {k: v[0] / v[1] for k, v in cls.items()},
                "assert_minus_filler": boot_f2(W, f2_pd),
                "rules": {},
            }
            for r in RULES:
                res = boot_rule(W, [rate(x) for x in rule_pd[r]], fifo_all, ff_all)
                res["by_bucket"] = {
                    b: pooled([rate(x, f) for x in rule_pd[r]])
                    for b, f in BUCKETS.items()
                }
                go["rules"][r] = res
            co["gammas"][gname] = go
        out["ckpt"][c] = co
    return out


def finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def defects(res: dict) -> list[str]:
    """Exit 1 (PREREG §7): a NaN in a class mean (kind / age) or in a hit rate."""
    bad = []
    for c, co in res["ckpt"].items():
        for g, go in co["gammas"].items():
            for k, v in go["means"]["kind"].items():
                if not finite(v):
                    bad.append(f"ckpt{c}.{g}.m_kind.{k} = {v}")
            for a, v in go["means"]["age"].items():
                if not finite(v):
                    bad.append(f"ckpt{c}.{g}.m_age.{a} = {v}")
            for r, rr in go["rules"].items():
                if not finite(rr["hit"]):
                    bad.append(f"ckpt{c}.{g}.hit.{r} = {rr['hit']}")
    return bad


# --------------------------------------------------------------------------- #
# ledger rows
# --------------------------------------------------------------------------- #


def write_rows(led, res: dict[int, dict]) -> None:
    how = "run.py::measure_seed; seed order 0,1,2"
    sd = [res[s] for s in SEEDS]
    for r in REFS:
        for b in BUCKETS:
            led.stat(f"U.hit.{r}.{b}", [x["refs"][r][b] for x in sd], how=how)
    for f in ("point", "lo", "hi"):
        led.stat(f"U.cap.oracle.{f}", [x["oracle"]["cap"][f] for x in sd], how=how)
    for c in CHECKPOINTS:
        pre = f"B.ckpt{c}"
        cs = [x["ckpt"][c] for x in sd]
        for h in HINDSIGHT:
            led.stat(f"{pre}.hit.{h}", [x["hindsight"][h]["hit"] for x in cs], how=how)
            for f in ("point", "lo", "hi"):
                led.stat(
                    f"{pre}.cap.{h}.{f}",
                    [x["hindsight"][h]["cap"][f] for x in cs],
                    how=how + " -> boot_rule (per document, 2000, paired)",
                )
        f1keys = sorted(set().union(*[x["F1"].keys() for x in cs]))
        led.note(
            f"{pre}.F1.D_by_age_class",
            {k: [x["F1"].get(k) for x in cs] for k in f1keys},
            how=how + " -> f1_accumulate; value per seed {mean, n} or null",
        )
        for k in f1keys:
            vals = [x["F1"].get(k) for x in cs]
            if all(v is not None for v in vals):
                led.stat(f"{pre}.F1.D.{k}.mean", [v["mean"] for v in vals], how=how)
                led.stat(f"{pre}.F1.D.{k}.n", [v["n"] for v in vals], how=how)
        for g in GAMMAS:
            gp = f"{pre}.{g}"
            gs = [x["gammas"][g] for x in cs]
            for r in RULES:
                rr = [x["rules"][r] for x in gs]
                for b in BUCKETS:
                    led.stat(
                        f"{gp}.hit.{r}.{b}", [x["by_bucket"][b] for x in rr], how=how
                    )
                for f in ("point", "lo", "hi"):
                    led.stat(
                        f"{gp}.cap.{r}.{f}",
                        [x["cap"][f] for x in rr],
                        how=how + " -> boot_rule (per document, 2000, paired)",
                    )
                    led.stat(
                        f"{gp}.diff_fifo.{r}.{f}",
                        [x["diff_fifo"][f] for x in rr],
                        how=how + " -> boot_rule (per document, 2000, paired)",
                    )
                led.stat(f"{gp}.cap.{r}.dropped", [x["cap_dropped"] for x in rr], how=how)
            for k in KINDS:
                led.stat(
                    f"{gp}.E_G.kind.{k}", [x["means"]["kind"][k] for x in gs], how=how
                )
                led.stat(
                    f"{gp}.E_G.kind.{k}.n", [x["means"]["n_kind"][k] for x in gs], how=how
                )
            for k in CLASSES:
                vals = [x["E_G_class"].get(k) for x in gs]
                if all(v is not None for v in vals):
                    led.stat(f"{gp}.E_G.class.{k}", vals, how=how)
            for f in ("point", "lo", "hi"):
                led.stat(
                    f"{gp}.F2.assert_minus_filler.{f}",
                    [x["assert_minus_filler"][f] for x in gs],
                    how=how + " -> boot_f2 (per document, 2000, paired)",
                )
            for kb in gs[0]["means"]["kb"]:
                vals = [x["means"]["kb"][kb] for x in gs]
                if all(v is not None for v in vals):
                    led.stat(f"{gp}.m_kb.{kb}", vals, how=how)
                led.stat(
                    f"{gp}.m_kb.{kb}.n", [x["means"]["n_kb"][kb] for x in gs], how=how
                )
            for a in gs[0]["means"]["age"]:
                led.stat(f"{gp}.m_age.{a}", [x["means"]["age"][a] for x in gs], how=how)
            led.stat(f"{gp}.kb_empty_cells", [x["kb_empty_cells"] for x in gs], how=how)
            led.stat(
                f"{gp}.kb_fallback_decisions",
                [x["kb_fallback_decisions"] for x in gs],
                how=how,
            )
    led.stat("n_documents", [x["n_documents"] for x in sd], how=how)


def manifest_config(manifest: Path) -> dict:
    return {
        "run_id": RUN_ID,
        "prereg": PREREG,
        "prereg_commit": PREREG_COMMIT,
        "question": QUESTION,
        "falsifier": FALSIFIER,
        "expected": EXPECTED,
        "seeds": SEEDS,
        "checkpoints": [f"B.ckpt{c}" for c in CHECKPOINTS],
        "gammas": GAMMAS,
        "M": M,
        "S": S,
        "age_bands": [list(b) for b in AGE_BANDS],
        "rules": list(RULES),
        "refs": list(REFS),
        "hindsight": list(HINDSIGHT),
        "tie_rule": (
            'B2 A1.10: random.Random(f"ko:{seed}:{doc_id}:{t}").choice(sorted(tied))'
        ),
        "rows": "t in [M, S), i < t (full-memory steps, every past sentence)",
        "documents": "U = [64, 1088) then [4096, 4160), lookahead-room documents(seed)",
        "already_inspected": True,
        "in_sample": True,
        "bootstrap": {"n": N_BOOT, "seed_base": BOOT_SEED_BASE, "paired": True},
        "inputs": {f"B.ckpt{c}.seed{s}": d_rel(c, s) for c in CHECKPOINTS for s in SEEDS},
        "t0_manifest": str(manifest),
        "t0_manifest_sha256": sha256(manifest),
        "reference_ledger": REF_LEDGER,
        "reference_ledger_sha256": sha256(ROOT / REF_LEDGER),
        "device": "cpu",
    }


def load_D(base: Path, c: int, s: int) -> dict:
    return torch.load(base / d_rel(c, s), weights_only=True)


def execute(led, manifest: Path, base: Path, limit: int | None) -> Exit:
    rels = [d_rel(c, s) for c in CHECKPOINTS for s in SEEDS]
    c1a = verify_manifest(manifest, rels, base)
    led.note("C1.start", c1a, how="run.py::verify_manifest before the first read")
    if not c1a["ok"]:
        led.status("did_not_run")
        return Exit.DID_NOT_RUN
    failed: list[str] = []
    ref_rows = {
        r["key"]: r
        for r in json.loads((ROOT / REF_LEDGER).read_text())["rows"]
        if isinstance(r.get("key"), str)
    }
    res: dict[int, dict] = {}
    for s in SEEDS:
        Ds = {c: load_D(base, c, s) for c in CHECKPOINTS}
        docs, *_ = LR.documents(s)
        want = sorted(d.doc_id for d in docs)
        for c, Dc in Ds.items():
            if sorted(Dc) != want:
                failed.append(f"C2 ckpt{c} seed{s}: doc ids differ from U")
            bad = [k for k, D in Dc.items() if not check_shape(D)]
            if bad:
                failed.append(f"C2 ckpt{c} seed{s}: {len(bad)} docs fail shape/NaN")
        if failed:
            break
        t0 = time.time()
        r = measure_seed(s, Ds, limit)
        r["seconds"] = time.time() - t0
        if not r["closure"]["ok"]:
            failed.append(f"closure seed{s}")
        if limit is None:
            for c in CHECKPOINTS:
                c3 = c3_failures(r["ckpt"][c]["c3_mine"], ref_rows, c, s)
                failed += [f"C3 seed{s}: {x}" for x in c3]
        res[s] = r
        print(f"seed {s} done {r['seconds']:.0f}s", flush=True)
    c1b = verify_manifest(manifest, rels, base)
    led.note("C1.end", c1b, how="run.py::verify_manifest after the last computation")
    if not c1b["ok"]:
        failed.append("C1 at end: manifest mismatch")
    if not failed and sorted(res) != SEEDS:
        failed.append(f"C4: seeds run {sorted(res)} != {SEEDS}")
    if not failed and limit is None and any(res[s]["n_documents"] != 1088 for s in res):
        failed.append("C4: not 1088 documents per seed")
    led.note("controls_failed", failed, how="run.py::execute")
    led.run_meta(seeds_actually_run=sorted(res))
    if failed:
        led.status("did_not_run")
        return Exit.DID_NOT_RUN
    for s in SEEDS:
        led.note(f"seed{s}.raw", res[s], how="run.py::measure_seed payload")
    bad = [f"seed{s}: {x}" for s in SEEDS for x in defects(res[s])]
    if bad:
        led.note("defects", bad, how="run.py::defects")
        led.status("failed")
        return Exit.FAIL
    write_rows(led, res)
    led.stat("seconds_per_seed", [res[s]["seconds"] for s in SEEDS], how="wall clock")
    led.status("ok")
    led.verdict(
        falsifier=FALSIFIER,
        outcome="inconclusive",
        detail=(
            "descriptive run (PREREG §1): no falsifier, no decision; "
            "controls C1-C4 passed"
        ),
    )
    return Exit.OK


def main(argv: list[str] | None = None) -> Exit:
    ap = ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--base", type=Path, default=None, help="default: the main checkout")
    ap.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    ap.add_argument("--limit", type=int, default=None, help="smoke only; skips C3/C4")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    base = a.base or main_checkout()
    if a.dry_run:
        rels = [d_rel(c, s) for c in CHECKPOINTS for s in SEEDS]
        v = verify_manifest(a.manifest, rels, base)
        print(json.dumps(v, indent=1))
        return Exit.OK if v["ok"] else Exit.DID_NOT_RUN
    from ledger import Ledger

    led = Ledger(RUN_ID, question=QUESTION, runs_root=a.runs_root)
    led.run_meta(device="cpu", steps_requested=0, steps_done=0)
    led.manifest(manifest_config(a.manifest))
    rc = execute(led, a.manifest, base, a.limit)
    led.command(
        [sys.executable, EXPERIMENT] + (argv if argv is not None else sys.argv[1:]),
        exit_code=int(rc),
        note="its exit code is the run's",
    )
    led.write()
    return status(rc)


if __name__ == "__main__":
    run_main(main)
