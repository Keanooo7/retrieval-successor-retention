"""B0-CF: the counterfactual-return ceiling -- descriptive.

Pre-registration: `experiments/cf-ceiling/PREREG.md` (0c5320e, committed alone before
any data). For each seed and checkpoint it measures the counterfactual demand tensor
Delta on the U range under FIFO (`cfdelta.cf_delta`: resident cells from
`rsr.metrics.loo.loo_delta_loss(mode="resample")`, evicted cells by reinstatement in
slot 0, D.pt's world), then scores the hindsight rule ``TargetPolicy(G^Delta_gamma)``
for gamma in {0, 0.5, 0.8, 0.9} by model-free residency, with the gamma sweep's cap
and paired per-document bootstrap. Loads checkpoints read-only, eval mode, no grad;
never calls ``rsr.constants.record()``.

**Imported, never re-implemented**: the gamma sweep's `run.py` (label, hindsight
target) and `ageshape.py` (pass-through recorder, Spearman, pooled median CI);
through them B0's (manifest, shape, `pairs_of`, `boot_weights`, `boot_rule`) and
lookahead-room's (`documents`, `discounted_returns`, `TargetPolicy`, `FactFiller`,
`load_model`); `rsr.metrics.loo`; `rsr.metrics.headroom.simulate`.

Usage (slot-wrapped)::

    .venv/bin/python experiments/cf-ceiling/run.py            # the run
    .venv/bin/python experiments/cf-ceiling/run.py --dry-run  # control C1 only

Exit codes (`rsr.exit_codes`): 0 ran, every control passed · 1 a defect after the
controls passed · 3 did not run / a control failed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
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
from rsr.metrics import loo  # noqa: E402
from rsr.train.loop import encode  # noqa: E402

EXPERIMENT = "experiments/cf-ceiling/run.py"
RUN_ID = "cf-ceiling"
PREREG = "experiments/cf-ceiling/PREREG.md"
PREREG_COMMIT = "0c5320e"
PROBE_COMMIT = "2767496"


def _load(name: str, rel: str):
    """Import an experiment script by path. Imported, never copied."""
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AS = _load("_cf_ageshape", "experiments/gamma-ceiling/ageshape.py")
GC = AS.GC
B0, LR = GC.B0, GC.LR
CF = _load("_cf_cfdelta", "experiments/cf-ceiling/cfdelta.py")

# --------------------------------------------------------------------------- #
# 🔒 PREREG.md, transcribed. Changing any is changing the PREREG.
# --------------------------------------------------------------------------- #

SEEDS = [0, 1, 2]
CHECKPOINTS = (3000, 2500)  # measured in this order
HEADLINE = 3000
GAMMAS = {"g0": 0.0, "g05": 0.5, "g08": 0.8, "g09": 0.9}
ARMS = ("cf", "cfshift")  # primary; sensitivity (DIAGNOSTIC)
RI_CONTROL = ("g0", "g09")
REFS = ("fifo", "factfiller", "oracle")
M = LR.M
S = LR.S
N_DOCS = 1088
BATCH = 32
MIN_COVERAGE = 0.80  # E0d's RESAMPLE_MIN_COVERAGE (its PREREG), PREREG §3
NLL_TOL = LR.NLL_TOL  # lookahead-room's, PREREG §6 C4
BUCKETS = B0.BUCKETS
MANIFEST = B0.MANIFEST
GC_LEDGER = "runs/gamma-ceiling/ledger.json"
CKPT_REL = ".worktrees/fresh-stream/runs/fresh-stream/B/seed{s}/ckpt-{c:06d}.pt"
CLASSES = LR.CLASSES

QUESTION = (
    "Does a hindsight rule on the counterfactual return G^Delta_gamma = sum_k gamma^k "
    "Delta_i(t+k) (Delta = resample-knockout LOO Delta-loss in D.pt's FIFO world) point "
    "at the fact/filler headroom where the r_i return does not (gamma sweep: SIGNAL)?"
)
FALSIFIER = (
    "NONE: descriptive (PREREG §1); the PM's pre-registered reading rule (PREREG §7) "
    "labels each gamma POINTS / PARTIAL / MISSES and the run CF-POINTS / CF-MIXED / "
    "CF-MISSES, as an input to an owner decision only"
)
EXPECTED = (
    "PREREG §9: Delta caps above the r_i caps at gamma >= 0.5 on every seed; gamma 0 "
    "MISSES; cap rises to 0.8-0.9; no POINTS on all seeds; predicted CF-MIXED (low "
    "confidence; CF-POINTS next); C3 exact."
)


# --------------------------------------------------------------------------- #
# the pure pieces (tests/test_cf_ceiling.py drives each)
# --------------------------------------------------------------------------- #


def label(points: list[float], los: list[float]) -> str:
    """PREREG §7 per gamma: the gamma sweep's ``label`` (same thresholds, verbatim)."""
    return GC.label(points, los)


def cf_class(labels: dict[str, str]) -> str:
    """PREREG §7: CF-POINTS if any gamma POINTS; CF-MISSES if every gamma MISSES;
    CF-MIXED otherwise."""
    if set(labels) != set(GAMMAS):
        raise ValueError(f"labels for {sorted(labels)} != {sorted(GAMMAS)}")
    vals = list(labels.values())
    if any(v == "POINTS" for v in vals):
        return "CF-POINTS"
    if all(v == "MISSES" for v in vals):
        return "CF-MISSES"
    return "CF-MIXED"


def batches(docs: list, size: int = BATCH) -> list[list]:
    """U in its order, in consecutive chunks of ``size`` (PREREG front matter)."""
    return [docs[i : i + size] for i in range(0, len(docs), size)]


def coverage(
    missing: torch.Tensor, slot_sentence: torch.Tensor, kind: torch.Tensor, m: int = M
) -> dict:
    """Resident donor coverage of one batch, by kind and over full-memory steps.
    ``missing`` ``[B, S, S]``, ``slot_sentence`` ``[B, S, M]``, ``kind`` ``[B, S]``.
    -> ``{"<kind>": [cells, missing], "full": [cells, missing], "all": [...]}``."""
    names = {v: k for k, v in loo.KIND.items()}
    out: dict[str, list[int]] = {}
    B, n, _ = slot_sentence.shape
    for b in range(B):
        for t in range(n):
            for j in range(slot_sentence.shape[2]):
                s = int(slot_sentence[b, t, j])
                if s < 0:
                    continue
                miss = int(bool(missing[b, t, s]))
                keys = [names[int(kind[b, s])], "all"] + (["full"] if t >= m else [])
                for k in keys:
                    c = out.setdefault(k, [0, 0])
                    c[0] += 1
                    c[1] += miss
    return out


def merge_counts(acc: dict, add: dict) -> None:
    for k, v in add.items():
        c = acc.setdefault(k, [0] * len(v))
        for i, x in enumerate(v):
            c[i] += x


def class_stats(doc, Delta: torch.Tensor, evicted: torch.Tensor, m: int = M) -> dict:
    """Sum, sum of squares and count of ``Delta[t][i]`` over full-memory steps
    ``t >= m``, ``i < t``, by ``slot_class`` x reading kind (PREREG §5)."""
    out: dict[str, list[float]] = {}
    n = Delta.shape[0]
    for t in range(m, n):
        for i in range(t):
            cls = LR.slot_class(doc, i, t)[0]
            k = f"{cls}|{'evicted' if bool(evicted[t, i]) else 'resident'}"
            v = float(Delta[t, i])
            c = out.setdefault(k, [0.0, 0.0, 0])
            c[0] += v
            c[1] += v * v
            c[2] += 1
    return out


def summarise_class(acc: dict) -> dict:
    out = {}
    for k, (s1, s2, n) in sorted(acc.items()):
        mean = s1 / n
        var = (s2 - n * mean * mean) / (n - 1) if n > 1 else float("nan")
        out[k] = {"mean": mean, "sd": math.sqrt(max(var, 0.0)), "n": int(n)}
    return out


def c3_failures(mine: dict, refs: dict, gc_rows: dict, c: int, seed: int) -> list[str]:
    """C3: the r_i arms gamma 0 and 0.9 reproduce the gamma sweep's ledger exactly
    (``==`` on the float): hit by bucket and cap point/lo/hi; and FIFO / factfiller
    all-query hit."""
    bad = []
    k = SEEDS.index(seed)

    def cmp(key: str, got) -> None:
        row = gc_rows.get(key)
        if row is None:
            bad.append(f"{key}: absent from {GC_LEDGER}")
            return
        theirs = row["samples"][k]
        if got is None or float(got) != float(theirs):
            bad.append(f"{key} seed{seed}: mine {got!r} != gamma sweep {theirs!r}")

    for g in RI_CONTROL:
        r = mine[g]
        for b in BUCKETS:
            cmp(f"B.ckpt{c}.{g}.hit.{b}", r["by_bucket"][b])
        for f in ("point", "lo", "hi"):
            cmp(f"B.ckpt{c}.{g}.cap.{f}", r["cap"][f])
    cmp("U.hit.fifo.all", refs["fifo"])
    cmp("U.hit.factfiller.all", refs["factfiller"])
    return bad


def instrument_failures(checks: dict, tol: float = NLL_TOL) -> list[str]:
    """C4 for one batch (PREREG §6)."""
    bad = []
    for k in ("live_loss_equal_loo", "slot_sentence_equal_loo", "fifo_memory_ok"):
        if checks.get(k) is not True:
            bad.append(f"C4 {k} = {checks.get(k)!r}")
    iw = checks.get("identity_worst")
    if not (isinstance(iw, float) and math.isfinite(iw) and iw <= tol):
        bad.append(f"C4 identity_worst = {iw!r} > {tol}")
    return bad


def file_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def delta_key(c: int, s: int, ckpt_sha: str, doc_ids: list[int]) -> str:
    """What a saved Delta depends on: the measuring code, the instrument, the
    checkpoint, the documents and the batch. A saved file is reused only on equality."""
    blob = json.dumps(
        {
            "cfdelta": file_sha(ROOT / "experiments/cf-ceiling/cfdelta.py"),
            "loo": file_sha(ROOT / "src/rsr/metrics/loo.py"),
            "ckpt": [c, s, ckpt_sha],
            "docs": doc_ids,
            "batch": BATCH,
            "donor_seed": s,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# measuring Delta for one (checkpoint, seed)
# --------------------------------------------------------------------------- #


def measure_delta(c: int, s: int, base: Path, raw_dir: Path, limit: int | None) -> dict:
    """-> ``{"Delta": {doc_id: [S, S]}, "shift": {...}, "evicted": {...},
    "coverage": ..., "class": ..., "failures": [...], "seconds", "raw", "reused"}``."""
    docs, vmap, V, closure = LR.documents(s)
    docs = list(docs if limit is None else docs[:limit])
    ck = base / CKPT_REL.format(c=c, s=s)
    ck_sha = B0.sha256(ck)
    key = delta_key(c, s, ck_sha, [d.doc_id for d in docs])
    raw = raw_dir / f"B-ckpt{c}-seed{s}.delta.pt"
    if limit is None and raw.exists():
        saved = torch.load(raw, weights_only=False)
        if saved.get("key") == key:
            saved["reused"] = True
            saved["raw"] = {"path": str(raw), "sha256": file_sha(raw)}
            return saved
    failures: list[str] = []
    if not closure["ok"]:
        failures.append(f"closure ckpt{c} seed{s}")
    if ck_sha != LR.CKPT_SHA256[(c, s)]:
        failures.append(f"C1 ckpt{c} seed{s}: sha256 {ck_sha} != lookahead-room's")
        return {"failures": failures}
    model = LR.load_model(ck, V)
    t0 = time.time()
    res: dict[str, Any] = {"Delta": {}, "shift": {}, "evicted": {}, "missing": {}}
    cov: dict = {}
    cls: dict = {}
    n_checks = {"identity_worst": 0.0, "batches": 0}
    for bi, chunk in enumerate(batches(docs)):
        ids, mask = encode(chunk, vmap, max_tokens=LR.L, steps=S)
        out = CF.cf_delta(model, chunk, ids, mask, seed=s)
        failures += [
            f"ckpt{c} seed{s} batch{bi}: {x}" for x in instrument_failures(out["checks"])
        ]
        n_checks["identity_worst"] = max(
            n_checks["identity_worst"], out["checks"]["identity_worst"]
        )
        n_checks["batches"] += 1
        ann = loo.stream_annotations(chunk, steps=S)
        merge_counts(cov, coverage(out["missing"], out["slot_sentence"], ann["kind"]))
        for b, doc in enumerate(chunk):
            D, Ds_ = out["Delta"][b].clone(), out["delta_shift"][b].clone()
            if not (B0.check_shape(D) and B0.check_shape(Ds_)):
                failures.append(f"C2 ckpt{c} seed{s} doc{doc.doc_id}: Delta shape/NaN")
            res["Delta"][doc.doc_id] = D
            res["shift"][doc.doc_id] = Ds_
            res["evicted"][doc.doc_id] = out["evicted"][b].clone()
            res["missing"][doc.doc_id] = out["missing"][b].clone()
            merge_counts(cls, class_stats(doc, D, out["evicted"][b]))
        print(
            f"ckpt{c} seed{s} batch {bi + 1}/{len(batches(docs))} "
            f"{time.time() - t0:.0f}s",
            flush=True,
        )
    full = cov.get("full", [0, 0])
    cov_full = 1 - full[1] / full[0] if full[0] else float("nan")
    if not (cov_full >= MIN_COVERAGE):
        failures.append(
            f"C5 ckpt{c} seed{s}: full-memory coverage {cov_full} < {MIN_COVERAGE}"
        )
    res.update(
        {
            "key": key,
            "ckpt": str(ck),
            "ckpt_sha256": ck_sha,
            "coverage": {
                k: {"cells": v[0], "missing": v[1], "coverage": 1 - v[1] / v[0]}
                for k, v in cov.items()
            },
            "class": summarise_class(cls),
            "checks": n_checks,
            "failures": failures,
            "seconds": time.time() - t0,
            "n_documents": len(docs),
            "reused": False,
        }
    )
    if limit is None and not failures:
        raw.parent.mkdir(parents=True, exist_ok=True)
        torch.save(res, raw)
        res["raw"] = {"path": str(raw), "sha256": file_sha(raw)}
    return res


# --------------------------------------------------------------------------- #
# scoring one seed (model-free)
# --------------------------------------------------------------------------- #


def score_arm(docs, targets: dict, W, fifo_all, ff_all) -> dict:
    """``TargetPolicy`` on ``targets[doc_id]`` through the gamma sweep's pass-through
    recorder: cap + bootstrap, hit by bucket, age-shape (DIAGNOSTIC), and the plain
    ``TargetPolicy`` hit for C6."""
    pd, rhos, idx, plain = [], [], [], []
    for j, doc in enumerate(docs):
        T = targets[doc.doc_id]
        pol = AS.AgeShapeRecorder(T)
        pd.append(B0.pairs_of(doc, pol))
        plain.append(B0.pairs_of(doc, LR.TargetPolicy(T)))
        Gm = torch.tensor([r[0] for r in pol.rows], dtype=torch.float64)
        Am = torch.tensor([r[1] for r in pol.rows], dtype=torch.float64)
        rhos.append(AS.spearman_rows(Gm, Am))
        idx.append(torch.full((len(pol.rows),), j, dtype=torch.long))
    res = B0.boot_rule(W, [B0.rate(x) for x in pd], fifo_all, ff_all)
    res["by_bucket"] = {
        b: B0.pooled([B0.rate(x, f) for x in pd]) for b, f in BUCKETS.items()
    }
    res["age_rho"] = AS.pooled_median_ci(torch.cat(rhos), torch.cat(idx), W)
    res["passthrough_ok"] = pd == plain
    return res


def score_ri(docs, D: dict, gamma: float, W, fifo_all, ff_all) -> dict:
    pd = [GC.hindsight_pairs(doc, D[doc.doc_id], gamma) for doc in docs]
    res = B0.boot_rule(W, [B0.rate(x) for x in pd], fifo_all, ff_all)
    res["by_bucket"] = {
        b: B0.pooled([B0.rate(x, f) for x in pd]) for b, f in BUCKETS.items()
    }
    return res


def score_seed(
    seed: int, deltas: dict[int, dict], Ds: dict[int, dict], limit: int | None
) -> dict:
    docs, *_ = LR.documents(seed)
    docs = list(docs if limit is None else docs[:limit])
    W = B0.boot_weights(len(docs), seed)
    ref_pd: dict[str, list] = {r: [] for r in REFS}
    for doc in docs:
        pols = {
            "fifo": FIFOPolicy(),
            "factfiller": LR.FactFiller(doc, random.Random(f"ff:{seed}:{doc.doc_id}")),
            "oracle": OraclePolicy(discounted_demand(doc, LR.ORACLE_GAMMA)),
        }
        for r, p in pols.items():
            ref_pd[r].append(B0.pairs_of(doc, p))
    fifo_all = [B0.rate(x) for x in ref_pd["fifo"]]
    ff_all = [B0.rate(x) for x in ref_pd["factfiller"]]
    out: dict[str, Any] = {
        "n_documents": len(docs),
        "refs": {
            r: {b: B0.pooled([B0.rate(x, f) for x in pd]) for b, f in BUCKETS.items()}
            for r, pd in ref_pd.items()
        },
        "oracle": B0.boot_rule(
            W, [B0.rate(x) for x in ref_pd["oracle"]], fifo_all, ff_all
        ),
        "ckpt": {},
    }
    for c in CHECKPOINTS:
        if c not in deltas:
            continue
        co: dict[str, Any] = {"cf": {}, "cfshift": {}, "ri": {}}
        for g, gamma in GAMMAS.items():
            for arm, src in (("cf", "Delta"), ("cfshift", "shift")):
                T = {
                    d: LR.discounted_returns(x, gamma) for d, x in deltas[c][src].items()
                }
                co[arm][g] = score_arm(docs, T, W, fifo_all, ff_all)
        for g in RI_CONTROL:
            co["ri"][g] = score_ri(docs, Ds[c], GAMMAS[g], W, fifo_all, ff_all)
        out["ckpt"][c] = co
    return out


def classify(res: dict[int, dict]) -> tuple[dict[str, str], str]:
    labels = {}
    for g in GAMMAS:
        rs = [res[s]["ckpt"][HEADLINE]["cf"][g]["cap"] for s in SEEDS]
        labels[g] = label([r["point"] for r in rs], [r["lo"] for r in rs])
    return labels, cf_class(labels)


def defects(res: dict) -> list[str]:
    """Exit 1 (PREREG §8): a NaN hit rate."""
    bad = []
    for c, co in res["ckpt"].items():
        for arm in ("cf", "cfshift", "ri"):
            for g, r in co[arm].items():
                for b, v in r["by_bucket"].items():
                    if not GC.finite(v):
                        bad.append(f"ckpt{c}.{arm}.{g}.hit.{b} = {v}")
    return bad


# --------------------------------------------------------------------------- #
# ledger rows
# --------------------------------------------------------------------------- #


def write_rows(led, res: dict[int, dict], meas: dict) -> None:
    how = "run.py::score_seed; seed order 0,1,2"
    boot = how + " -> B0.boot_rule (per document, 2000, paired)"
    sd = [res[s] for s in SEEDS]
    for r in REFS:
        for b in BUCKETS:
            led.stat(f"U.hit.{r}.{b}", [x["refs"][r][b] for x in sd], how=how)
    for f in ("point", "lo", "hi"):
        led.stat(f"U.cap.oracle.{f}", [x["oracle"]["cap"][f] for x in sd], how=boot)
    for c in CHECKPOINTS:
        for arm in ("cf", "cfshift", "ri"):
            for g in GAMMAS if arm != "ri" else RI_CONTROL:
                pre = f"B.ckpt{c}.{arm}.{g}"
                rr = [x["ckpt"][c][arm][g] for x in sd]
                for b in BUCKETS:
                    led.stat(f"{pre}.hit.{b}", [x["by_bucket"][b] for x in rr], how=how)
                for f in ("point", "lo", "hi"):
                    led.stat(f"{pre}.cap.{f}", [x["cap"][f] for x in rr], how=boot)
                    led.stat(
                        f"{pre}.diff_fifo.{f}", [x["diff_fifo"][f] for x in rr], how=boot
                    )
                led.stat(f"{pre}.cap.dropped", [x["cap_dropped"] for x in rr], how=how)
                if arm != "ri":
                    for f in (
                        "median",
                        "lo",
                        "hi",
                        "mean",
                        "n_steps",
                        "n_steps_undefined",
                    ):
                        led.stat(
                            f"{pre}.age_rho.{f}",
                            [x["age_rho"][f] for x in rr],
                            how="gamma-ceiling ageshape.pooled_median_ci (DIAGNOSTIC)",
                        )
        m = [meas[(c, s)] for s in SEEDS]
        for k in ("full", "all", "assert", "query", "filler"):
            led.stat(
                f"B.ckpt{c}.coverage.{k}",
                [x["coverage"].get(k, {}).get("coverage", float("nan")) for x in m],
                how="run.py::coverage over every batch",
            )
            led.stat(
                f"B.ckpt{c}.missing.{k}",
                [x["coverage"].get(k, {}).get("missing", 0) for x in m],
                how="run.py::coverage over every batch",
            )
        led.stat(
            f"B.ckpt{c}.identity_worst",
            [x["checks"]["identity_worst"] for x in m],
            how="cfdelta.cf_delta checks, max over batches",
        )
        led.stat(
            f"B.ckpt{c}.measure_seconds",
            [x["seconds"] for x in m],
            how="wall clock of the measurement that wrote the raw (raw notes: reused)",
        )
        for s, x in zip(SEEDS, m, strict=True):
            led.note(
                f"B.ckpt{c}.seed{s}.delta_by_class",
                x["class"],
                how="run.py::class_stats (PREREG §5)",
            )
            led.note(
                f"B.ckpt{c}.seed{s}.raw",
                {**x.get("raw", {}), "reused": x["reused"], "key": x["key"]},
                how="runs/cf-ceiling/raw (gitignored)",
            )
    led.stat("n_documents", [x["n_documents"] for x in sd], how=how)


def manifest_config(manifest: Path, base: Path) -> dict:
    return {
        "run_id": RUN_ID,
        "prereg": PREREG,
        "prereg_commit": PREREG_COMMIT,
        "probe_commit": PROBE_COMMIT,
        "question": QUESTION,
        "falsifier": FALSIFIER,
        "expected": EXPECTED,
        "seeds": SEEDS,
        "checkpoints": [f"B.ckpt{c}" for c in CHECKPOINTS],
        "headline_checkpoint": f"B.ckpt{HEADLINE}",
        "gammas": GAMMAS,
        "arms": {
            "cf": "primary, TargetPolicy(discounted_returns(Delta, gamma))",
            "cfshift": "DIAGNOSTIC sensitivity: evicted cells + Delta_res[t][0]",
            "ri": "C3 control, the gamma sweep's hindsight_target on D.pt",
        },
        "M": M,
        "S": S,
        "batch": BATCH,
        "delta": {
            "resident": "rsr.metrics.loo.loo_delta_loss(mode='resample', seed=seed)",
            "evicted": "L(FIFO memory) - L(slot 0 := gest_i), one batched forward",
            "missing_donor": "imputed 0, counted",
            "min_coverage": MIN_COVERAGE,
            "nll_tol": NLL_TOL,
        },
        "reading_rule": {
            "POINTS": f"cap CI lower >= {GC.POINTS_LO} on all 3 seeds (ckpt{HEADLINE})",
            "PARTIAL": f"not POINTS and cap point >= {GC.PARTIAL_POINT} on all 3 seeds",
            "MISSES": "otherwise",
            "class": "CF-POINTS if any POINTS; CF-MISSES if every MISSES; "
            "CF-MIXED otherwise",
        },
        "documents": "U = [64, 1088) then [4096, 4160), lookahead-room documents(seed)",
        "bootstrap": {"n": B0.N_BOOT, "seed_base": B0.BOOT_SEED_BASE, "paired": True},
        "checkpoints_files": {
            f"B.ckpt{c}.seed{s}": CKPT_REL.format(c=c, s=s)
            for c in CHECKPOINTS
            for s in SEEDS
        },
        "d_pt": {
            f"B.ckpt{c}.seed{s}": B0.d_rel(c, s) for c in CHECKPOINTS for s in SEEDS
        },
        "t0_manifest": str(manifest),
        "t0_manifest_sha256": B0.sha256(manifest),
        "gc_ledger": GC_LEDGER,
        "gc_ledger_sha256": B0.sha256(ROOT / GC_LEDGER),
        "cfdelta_sha256": file_sha(ROOT / "experiments/cf-ceiling/cfdelta.py"),
        "loo_sha256": file_sha(ROOT / "src/rsr/metrics/loo.py"),
        "threads": 1,
        "device": "cpu",
    }


def all_rels() -> list[str]:
    return [CKPT_REL.format(c=c, s=s) for c in CHECKPOINTS for s in SEEDS] + [
        B0.d_rel(c, s) for c in CHECKPOINTS for s in SEEDS
    ]


def execute(led, manifest: Path, base: Path, raw_dir: Path, limit: int | None) -> Exit:
    rels = all_rels()
    c1a = B0.verify_manifest(manifest, rels, base)
    led.note("C1.start", c1a, how="B0.verify_manifest before the first read")
    if not c1a["ok"]:
        led.status("did_not_run")
        return Exit.DID_NOT_RUN
    failed: list[str] = []
    gc_rows = {
        r["key"]: r
        for r in json.loads((ROOT / GC_LEDGER).read_text())["rows"]
        if isinstance(r.get("key"), str)
    }
    # C2 on the D.pt (the r_i control's input)
    Ds: dict[int, dict[int, dict]] = {}
    for s in SEEDS:
        docs, *_ = LR.documents(s)
        want = sorted(d.doc_id for d in docs)
        Ds[s] = {}
        for c in CHECKPOINTS:
            Dc = B0.load_D(base, c, s)
            if sorted(Dc) != want:
                failed.append(f"C2 D.pt ckpt{c} seed{s}: doc ids differ from U")
            if any(not B0.check_shape(D) for D in Dc.values()):
                failed.append(f"C2 D.pt ckpt{c} seed{s}: shape/NaN")
            Ds[s][c] = Dc
    meas: dict[tuple[int, int], dict] = {}
    if not failed:
        for c in CHECKPOINTS:
            for s in SEEDS:
                r = measure_delta(c, s, base, raw_dir, limit)
                failed += r["failures"]
                if r["failures"]:
                    break
                meas[(c, s)] = r
            if failed:
                break
    res: dict[int, dict] = {}
    if not failed:
        for s in SEEDS:
            t0 = time.time()
            deltas = {c: meas[(c, s)] for c in CHECKPOINTS}
            r = score_seed(s, deltas, Ds[s], limit)
            r["seconds"] = time.time() - t0
            for c in CHECKPOINTS:
                for arm in ARMS:
                    for g, x in r["ckpt"][c][arm].items():
                        if not x["passthrough_ok"]:
                            failed.append(f"C6 ckpt{c} {arm} {g} seed{s}: pass-through")
                if limit is None:
                    refs = {k: r["refs"][k]["all"] for k in ("fifo", "factfiller")}
                    failed += [
                        f"C3: {x}"
                        for x in c3_failures(r["ckpt"][c]["ri"], refs, gc_rows, c, s)
                    ]
            res[s] = r
            print(f"scored seed {s} {r['seconds']:.0f}s", flush=True)
    c1b = B0.verify_manifest(manifest, rels, base)
    led.note("C1.end", c1b, how="B0.verify_manifest after the last computation")
    if not c1b["ok"]:
        failed.append("C1 at end: manifest mismatch")
    if not failed and sorted(res) != SEEDS:
        failed.append(f"C6: seeds run {sorted(res)} != {SEEDS}")
    if not failed and limit is None and any(res[s]["n_documents"] != N_DOCS for s in res):
        failed.append(f"C6: not {N_DOCS} documents per seed")
    led.note("controls_failed", failed, how="run.py::execute")
    led.run_meta(seeds_actually_run=sorted(res))
    if failed:
        for x in failed:
            print(x, flush=True)
        led.status("did_not_run")
        return Exit.DID_NOT_RUN
    for s in SEEDS:
        led.note(f"seed{s}.scored", res[s], how="run.py::score_seed payload")
    bad = [f"seed{s}: {x}" for s in SEEDS for x in defects(res[s])]
    if bad:
        led.note("defects", bad, how="run.py::defects")
        led.status("failed")
        return Exit.FAIL
    write_rows(led, res, meas)
    labels, cls = classify(res)
    led.note(f"B.ckpt{HEADLINE}.label", labels, how="run.py::classify (PREREG §7)")
    led.note("cf_class", cls, how="run.py::classify (PREREG §7)")
    led.stat(
        "score_seconds_per_seed", [res[s]["seconds"] for s in SEEDS], how="wall clock"
    )
    led.status("ok")
    led.verdict(
        falsifier=FALSIFIER,
        outcome="inconclusive",
        detail=(
            f"descriptive run (PREREG §1): controls C1-C6 passed; reading rule "
            f"(PREREG §7) labels {labels}; class {cls}"
        ),
    )
    print(json.dumps({"labels": labels, "class": cls}), flush=True)
    return Exit.OK


def main(argv: list[str] | None = None) -> Exit:
    ap = ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--base", type=Path, default=None, help="default: the main checkout")
    ap.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="smoke only: first N docs per seed, skips C3, saves no raw",
    )
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    torch.set_num_threads(1)
    base = a.base or B0.main_checkout()
    if a.dry_run:
        v = B0.verify_manifest(a.manifest, all_rels(), base)
        print(json.dumps(v, indent=1))
        return Exit.OK if v["ok"] else Exit.DID_NOT_RUN
    from ledger import Ledger

    led = Ledger(RUN_ID, question=QUESTION, runs_root=a.runs_root)
    led.run_meta(device="cpu", steps_requested=0, steps_done=0)
    led.manifest(manifest_config(a.manifest, base))
    raw_dir = a.runs_root / RUN_ID / "raw"
    rc = execute(led, a.manifest, base, raw_dir, a.limit)
    led.command(
        [sys.executable, EXPERIMENT] + (argv if argv is not None else sys.argv[1:]),
        exit_code=int(rc),
        note="its exit code is the run's",
    )
    led.write()
    return status(rc)


if __name__ == "__main__":
    run_main(main)
