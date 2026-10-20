"""B0-gamma Amendment 1: age-shape of the target (DIAGNOSTIC); descriptive, no gate.

Amendment 1 of `experiments/gamma-ceiling/PREREG.md` (c4c1a25, committed alone, after the
main run c186ac6 and before any age-shape number). For each gamma x ckpt x seed: the
within-step Spearman rho between ``G_gamma[t][written_at]`` and slot age
``t - written_at`` over the 16 live slots of the hindsight rule's OWN memory at each of
its eviction decisions in ``simulate``; pooled as the lower median of per-step rho over
every defined step; per-document bootstrap 95% CI on B0's replicate matrix.

Age is an analysis variable only: it is read from ``slots.written_at`` by a pass-through
wrapper and never reaches the rule, whose choice is returned unchanged (control A1-C:
the wrapped rule's hit rate equals the main ledger's exactly).

Usage::

    uv run python experiments/gamma-ceiling/ageshape.py

Exit codes (`rsr.exit_codes`): 0 ran, every control passed · 1 a defect after the
controls passed · 3 did not run / a control failed.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rsr.exit_codes import ArgumentParser, Exit, run_main, status  # noqa: E402

EXPERIMENT = "experiments/gamma-ceiling/ageshape.py"
RUN_ID = "gamma-ceiling-ageshape"
AMENDMENT_COMMIT = "c4c1a25"
MAIN_LEDGER = "runs/gamma-ceiling/ledger.json"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GC = _load("_gc_run", "experiments/gamma-ceiling/run.py")
B0, LR = GC.B0, GC.LR
SEEDS, CHECKPOINTS, GAMMAS = GC.SEEDS, GC.CHECKPOINTS, GC.GAMMAS

QUESTION = (
    "Amendment 1 (DIAGNOSTIC): how age-ranked is each hindsight target G_gamma across "
    "the rule's own resident slots at its eviction steps (median per-step Spearman rho "
    "with slot age)?"
)
FALSIFIER = (
    "NONE: diagnostic only (PREREG Amendment 1); never gating, no threshold on rho, "
    "does not change the class rule"
)


# --------------------------------------------------------------------------- #
# the pure pieces (tests/test_gamma_ceiling.py drives each)
# --------------------------------------------------------------------------- #


class AgeShapeRecorder:
    """Pass-through around lookahead-room's ``TargetPolicy``: records, at each eviction
    decision, the live slots' ``(G, age)``; returns the wrapped choice unchanged."""

    name = "target"

    def __init__(self, target: torch.Tensor) -> None:
        self.inner = LR.TargetPolicy(target)
        self.rows: list[tuple[list[float], list[int]]] = []

    def select_eviction(self, slots, context, step: int) -> int:
        live = [k for k in range(len(slots.live)) if bool(slots.live[k])]
        w = [int(slots.written_at[k]) for k in live]
        self.rows.append(([self.inner.target[step][x] for x in w], [step - x for x in w]))
        return self.inner.select_eviction(slots, context, step)

    def observe(self, slots, attn, step: int) -> None:
        return self.inner.observe(slots, attn, step)

    def on_write(self, slots, slot: int, step: int) -> None:
        return self.inner.on_write(slots, slot, step)

    def reset(self) -> None:
        self.rows = []
        return self.inner.reset()


def avg_ranks(x: torch.Tensor) -> torch.Tensor:
    """Row-wise 1-based average ranks of ``x`` ``[n, k]`` (ties share their mean rank)."""
    less = (x.unsqueeze(1) < x.unsqueeze(2)).sum(-1).to(torch.float64)
    eq = (x.unsqueeze(1) == x.unsqueeze(2)).sum(-1).to(torch.float64)
    return less + (eq + 1.0) / 2.0


def spearman_rows(G: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Row-wise Spearman rho of ``G`` and ``A`` ``[n, k]``; NaN where a row has no
    rank variance."""
    rg = avg_ranks(G.to(torch.float64))
    ra = avg_ranks(A.to(torch.float64))
    rg = rg - rg.mean(1, keepdim=True)
    ra = ra - ra.mean(1, keepdim=True)
    num = (rg * ra).sum(1)
    den = torch.sqrt((rg * rg).sum(1) * (ra * ra).sum(1))
    return torch.where(den > 0, num / den, torch.full_like(num, float("nan")))


def weighted_lower_median(v_sorted: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``v_sorted`` ``[n]`` ascending; ``w`` ``[B, n]`` non-negative weights in that
    order. -> ``[B]``: the smallest value whose cumulative weight reaches half the
    total. With unit weights this is the lower median."""
    cum = torch.cumsum(w, dim=1)
    half = (cum[:, -1] / 2.0).unsqueeze(1)
    idx = torch.searchsorted(cum.contiguous(), half.contiguous()).squeeze(1)
    return v_sorted[idx.clamp(max=v_sorted.numel() - 1)]


def pooled_median_ci(
    rho: torch.Tensor, doc_idx: torch.Tensor, W: torch.Tensor, chunk: int = 250
) -> dict:
    """Lower median of the defined per-step rho and its per-document bootstrap CI."""
    ok = ~torch.isnan(rho)
    v, d = rho[ok], doc_idx[ok]
    order = torch.argsort(v, stable=True)
    v, d = v[order], d[order]
    point = float(weighted_lower_median(v, torch.ones(1, v.numel(), dtype=torch.float64)))
    reps = []
    for b in range(0, W.shape[0], chunk):
        reps.append(weighted_lower_median(v, W[b : b + chunk][:, d]))
    reps_t = torch.cat(reps)
    return {
        "median": point,
        "lo": float(torch.quantile(reps_t, 0.025)),
        "hi": float(torch.quantile(reps_t, 0.975)),
        "mean": float(v.mean()),
        "n_steps": int(v.numel()),
        "n_steps_undefined": int((~ok).sum()),
    }


# --------------------------------------------------------------------------- #
# one seed
# --------------------------------------------------------------------------- #


def measure_seed(seed: int, Ds: dict[int, dict]) -> dict:
    docs, *_ = LR.documents(seed)
    W = B0.boot_weights(len(docs), seed)
    out: dict[str, Any] = {"n_documents": len(docs), "ckpt": {}}
    for c, Dc in Ds.items():
        co = {}
        for g, gamma in GAMMAS.items():
            pd, rhos, idx = [], [], []
            for j, doc in enumerate(docs):
                pol = AgeShapeRecorder(GC.hindsight_target(Dc[doc.doc_id], gamma))
                pd.append(B0.pairs_of(doc, pol))
                Gm = torch.tensor([r[0] for r in pol.rows], dtype=torch.float64)
                Am = torch.tensor([r[1] for r in pol.rows], dtype=torch.float64)
                rhos.append(spearman_rows(Gm, Am))
                idx.append(torch.full((len(pol.rows),), j, dtype=torch.long))
            res = pooled_median_ci(torch.cat(rhos), torch.cat(idx), W)
            res["hit_all"] = B0.pooled([B0.rate(x) for x in pd])
            co[g] = res
        out["ckpt"][c] = co
    return out


def control_failures(res: dict, main_rows: dict, seed: int) -> list[str]:
    """A1-C: the wrapped rule's pooled hit equals the main ledger's, exactly."""
    bad = []
    k = SEEDS.index(seed)
    for c, co in res["ckpt"].items():
        for g, r in co.items():
            key = f"B.ckpt{c}.{g}.hit.all"
            row = main_rows.get(key)
            if row is None:
                bad.append(f"{key}: absent from {MAIN_LEDGER}")
            elif float(r["hit_all"]) != float(row["samples"][k]):
                bad.append(f"{key} seed{seed}: {r['hit_all']!r} != {row['samples'][k]!r}")
    return bad


def execute(led, manifest: Path, base: Path) -> Exit:
    rels = [B0.d_rel(c, s) for c in CHECKPOINTS for s in SEEDS]
    c1a = B0.verify_manifest(manifest, rels, base)
    led.note("C1.start", c1a, how="B0.verify_manifest before the first read")
    if not c1a["ok"]:
        led.status("did_not_run")
        return Exit.DID_NOT_RUN
    main_rows = {
        r["key"]: r
        for r in json.loads((ROOT / MAIN_LEDGER).read_text())["rows"]
        if isinstance(r.get("key"), str)
    }
    failed: list[str] = []
    res: dict[int, dict] = {}
    for s in SEEDS:
        Ds = {c: B0.load_D(base, c, s) for c in CHECKPOINTS}
        t0 = time.time()
        r = measure_seed(s, Ds)
        r["seconds"] = time.time() - t0
        failed += control_failures(r, main_rows, s)
        res[s] = r
        print(f"seed {s} done {r['seconds']:.0f}s", flush=True)
    c1b = B0.verify_manifest(manifest, rels, base)
    led.note("C1.end", c1b, how="B0.verify_manifest after the last computation")
    if not c1b["ok"]:
        failed.append("C1 at end: manifest mismatch")
    if sorted(res) != SEEDS or any(res[s]["n_documents"] != GC.N_DOCS for s in res):
        failed.append("C4: seeds or document count")
    led.note("controls_failed", failed, how="ageshape.py::execute")
    led.run_meta(seeds_actually_run=sorted(res))
    if failed:
        for x in failed:
            print(x, flush=True)
        led.status("did_not_run")
        return Exit.DID_NOT_RUN
    for s in SEEDS:
        led.note(f"seed{s}.raw", res[s], how="ageshape.py::measure_seed payload")
    how = "ageshape.py::measure_seed -> pooled_median_ci (per document, 2000, paired)"
    bad = []
    for c in CHECKPOINTS:
        for g in GAMMAS:
            pre = f"B.ckpt{c}.{g}.age_rho"
            rr = [res[s]["ckpt"][c][g] for s in SEEDS]
            for f in ("median", "lo", "hi", "mean", "n_steps", "n_steps_undefined"):
                vals = [x[f] for x in rr]
                if not all(
                    isinstance(v, (int, float)) and math.isfinite(v) for v in vals
                ):
                    bad.append(f"{pre}.{f} = {vals}")
                    continue
                led.stat(f"{pre}.{f}", vals, how=how)
    if bad:
        led.note("defects", bad, how="ageshape.py::execute")
        led.status("failed")
        return Exit.FAIL
    led.stat("seconds_per_seed", [res[s]["seconds"] for s in SEEDS], how="wall clock")
    led.status("ok")
    led.verdict(
        falsifier=FALSIFIER,
        outcome="inconclusive",
        detail="diagnostic (Amendment 1): controls C1 and A1-C passed; no threshold",
    )
    return Exit.OK


def main(argv: list[str] | None = None) -> Exit:
    ap = ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--manifest", type=Path, default=GC.MANIFEST)
    ap.add_argument("--base", type=Path, default=None, help="default: the main checkout")
    ap.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    a = ap.parse_args(argv)
    base = a.base or B0.main_checkout()
    from ledger import Ledger

    led = Ledger(RUN_ID, question=QUESTION, runs_root=a.runs_root)
    led.run_meta(device="cpu", steps_requested=0, steps_done=0)
    led.manifest(
        {
            "run_id": RUN_ID,
            "prereg": GC.PREREG + " (Amendment 1)",
            "amendment_commit": AMENDMENT_COMMIT,
            "written_after_main_run": "c186ac6",
            "question": QUESTION,
            "falsifier": FALSIFIER,
            "expected": "none registered (Amendment 1: caps already seen)",
            "seeds": SEEDS,
            "checkpoints": [f"B.ckpt{c}" for c in CHECKPOINTS],
            "gammas": GAMMAS,
            "resident_slots": "the hindsight rule's own 16 live slots at each eviction",
            "rho": "Spearman (average ranks) of G_gamma vs age = t - written_at",
            "pooling": "lower median over defined steps; weighted lower median per rep",
            "bootstrap": {"n": B0.N_BOOT, "seed_base": B0.BOOT_SEED_BASE},
            "inputs": {
                f"B.ckpt{c}.seed{s}": B0.d_rel(c, s) for c in CHECKPOINTS for s in SEEDS
            },
            "t0_manifest_sha256": B0.sha256(a.manifest),
            "main_ledger": MAIN_LEDGER,
            "main_ledger_sha256": B0.sha256(ROOT / MAIN_LEDGER),
            "device": "cpu",
        }
    )
    rc = execute(led, a.manifest, base)
    led.command(
        [sys.executable, EXPERIMENT] + (argv if argv is not None else sys.argv[1:]),
        exit_code=int(rc),
        note="its exit code is the run's",
    )
    led.write()
    return status(rc)


if __name__ == "__main__":
    run_main(main)
