"""B0-gamma: the hindsight target-ceiling sweep over the discount gamma -- descriptive.

Pre-registration: `experiments/gamma-ceiling/PREREG.md`, committed alone ahead of this
file (daabdb2). **Model-free**: it loads no checkpoint, runs no forward pass, and never
calls `rsr.constants.record()`. It reads only the lookahead-room-r2 demand tensors
(`D.pt`; PREREG §2 for how they were produced) on the already-inspected U range.

For each gamma in {0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0} it scores the hindsight rule
``TargetPolicy(G_gamma)`` (evict the live slot with the lowest realised discounted
return, §3.4 and correction 2; ties to the oldest) by model-free residency, and reports
its cap ``(hit - FIFO) / (factfiller - FIFO)`` with B0's paired per-document bootstrap.

**Imported, never re-implemented**: B0's module (`experiments/b0-ceilings/run.py`:
manifest check, shape check, references' rates, `boot_weights`, `boot_rule`) and,
through it, lookahead-room's (`documents`, `discounted_returns`, `TargetPolicy`,
`FactFiller`) and `rsr.metrics.headroom.simulate`.

Usage::

    uv run python experiments/gamma-ceiling/run.py            # the run
    uv run python experiments/gamma-ceiling/run.py --dry-run  # control C1 only

Exit codes (`rsr.exit_codes`): 0 ran, every control passed · 1 a defect after the
controls passed · 3 did not run / a control failed.
"""

from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rsr.baselines.fifo import FIFOPolicy  # noqa: E402
from rsr.exit_codes import ArgumentParser, Exit, run_main, status  # noqa: E402

EXPERIMENT = "experiments/gamma-ceiling/run.py"
RUN_ID = "gamma-ceiling"
PREREG = "experiments/gamma-ceiling/PREREG.md"
PREREG_COMMIT = "daabdb2"


def _load(name: str, rel: str):
    """Import an experiment script by path. Imported, never copied."""
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B0 = _load("_gc_b0_ceilings", "experiments/b0-ceilings/run.py")
LR = B0.LR

# --------------------------------------------------------------------------- #
# 🔒 PREREG.md, transcribed. Changing any is changing the PREREG.
# --------------------------------------------------------------------------- #

SEEDS = [0, 1, 2]
CHECKPOINTS = (3000, 2500)
HEADLINE = 3000
#: PREREG §4. Ordered; the keys are ledger names.
GAMMAS = {
    "g0": 0.0,
    "g05": 0.5,
    "g08": 0.8,
    "g09": 0.9,
    "g095": 0.95,
    "g099": 0.99,
    "g1": 1.0,
}
ALREADY_SEEN = ("g0", "g09")
M = LR.M
S = LR.S
N_DOCS = 1088
BUCKETS = B0.BUCKETS
MANIFEST = B0.MANIFEST
B0_LEDGER = "runs/b0-ceilings/ledger.json"
#: C3: our gamma key -> B0's hindsight rule name.
B0_RULE = {"g0": "rule_g0", "g09": "rule_g09"}
#: PREREG §6, verbatim thresholds.
POINTS_LO = 0.80
PARTIAL_POINT = 0.50

QUESTION = (
    "Is the hindsight rule's shortfall from the fact/filler headroom (B0: cap +0.184 / "
    "+0.134 / +0.537 at gamma 0.9, ckpt3000) a discount-horizon problem (some gamma in "
    "{0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0} points at the headroom) or a signal problem "
    "(none does)?"
)
FALSIFIER = (
    "NONE: descriptive (PREREG §1); the PM's pre-registered reading rule (PREREG §6) "
    "classifies each gamma POINTS / PARTIAL / MISSES and the sweep HORIZON / SIGNAL / "
    "MIXED, as an input to an owner decision only"
)
EXPECTED = (
    "PREREG §9: cap rises with gamma to 0.95-0.99, flat or lower at 1.0; "
    "no gamma POINTS; "
    "seed 1 lowest; predicted class SIGNAL (low confidence; MIXED next); C3 exact."
)


# --------------------------------------------------------------------------- #
# the pure pieces (tests/test_gamma_ceiling.py drives each)
# --------------------------------------------------------------------------- #


def _ge(x, thr: float) -> bool:
    """``x >= thr``; a NaN / None never satisfies it (PREREG §6 implementation note)."""
    return isinstance(x, (int, float)) and math.isfinite(x) and x >= thr


def label(points: list[float], los: list[float]) -> str:
    """PREREG §6, per gamma, over the three seeds' cap point and CI lower bound."""
    if len(points) != len(SEEDS) or len(los) != len(SEEDS):
        raise ValueError(f"need {len(SEEDS)} seeds, got {len(points)} / {len(los)}")
    if all(_ge(lo, POINTS_LO) for lo in los):
        return "POINTS"
    if all(_ge(p, PARTIAL_POINT) for p in points):
        return "PARTIAL"
    return "MISSES"


def sweep_class(labels: dict[str, str]) -> str:
    """PREREG §6: HORIZON if any gamma POINTS; SIGNAL if every gamma MISSES;
    MIXED otherwise."""
    if set(labels) != set(GAMMAS):
        raise ValueError(f"labels for {sorted(labels)} != {sorted(GAMMAS)}")
    vals = list(labels.values())
    if any(v == "POINTS" for v in vals):
        return "HORIZON"
    if all(v == "MISSES" for v in vals):
        return "SIGNAL"
    return "MIXED"


def hindsight_target(D, gamma: float):
    """``G_gamma`` of one document's ``D``: lookahead-room's ``discounted_returns``."""
    return LR.discounted_returns(D, gamma)


def hindsight_pairs(doc, D, gamma: float) -> list[tuple[int, bool]]:
    """Model-free residency of ``TargetPolicy(G_gamma)``: ``[(gap, hit), ...]``."""
    return B0.pairs_of(doc, LR.TargetPolicy(hindsight_target(D, gamma)))


def c3_failures(
    mine: dict[str, dict], refs: dict[str, float], b0_rows: dict, c: int, seed: int
) -> list[str]:
    """C3: gamma 0 and 0.9 reproduce B0's ledger exactly (``==`` on the float):
    hit and cap point/lo/hi of ``rule_g0`` / ``rule_g09``; and FIFO / factfiller hit."""
    bad = []
    k = SEEDS.index(seed)

    def cmp(key: str, got) -> None:
        row = b0_rows.get(key)
        if row is None:
            bad.append(f"{key}: absent from {B0_LEDGER}")
            return
        theirs = row["samples"][k]
        if got is None or float(got) != float(theirs):
            bad.append(f"{key} seed{seed}: mine {got!r} != B0 {theirs!r}")

    for g, rule in B0_RULE.items():
        r = mine[g]
        cmp(f"B.ckpt{c}.hit.{rule}", r["hit"])
        for f in ("point", "lo", "hi"):
            cmp(f"B.ckpt{c}.cap.{rule}.{f}", r["cap"][f])
    cmp("U.hit.fifo.all", refs["fifo"])
    cmp("U.hit.factfiller.all", refs["factfiller"])
    return bad


# --------------------------------------------------------------------------- #
# one seed
# --------------------------------------------------------------------------- #


def measure_seed(seed: int, Ds: dict[int, dict], limit: int | None = None) -> dict:
    """``Ds[c] = {doc_id: D}`` -> per-(ckpt, gamma) hindsight-rule results."""
    docs, _vmap, _V, closure = LR.documents(seed)
    if limit is not None:
        docs = docs[:limit]
    W = B0.boot_weights(len(docs), seed)
    ref_pd: dict[str, list] = {"fifo": [], "factfiller": []}
    for doc in docs:
        ref_pd["fifo"].append(B0.pairs_of(doc, FIFOPolicy()))
        ref_pd["factfiller"].append(
            B0.pairs_of(doc, LR.FactFiller(doc, random.Random(f"ff:{seed}:{doc.doc_id}")))
        )
    fifo_all = [B0.rate(x) for x in ref_pd["fifo"]]
    ff_all = [B0.rate(x) for x in ref_pd["factfiller"]]
    out: dict[str, Any] = {
        "n_documents": len(docs),
        "doc_ids_first_last": [docs[0].doc_id, docs[-1].doc_id],
        "closure": closure,
        "refs": {
            r: {b: B0.pooled([B0.rate(x, f) for x in pd]) for b, f in BUCKETS.items()}
            for r, pd in ref_pd.items()
        },
        "ckpt": {},
    }
    for c, Dc in Ds.items():
        co: dict[str, Any] = {}
        for g, gamma in GAMMAS.items():
            pd = [hindsight_pairs(doc, Dc[doc.doc_id], gamma) for doc in docs]
            res = B0.boot_rule(W, [B0.rate(x) for x in pd], fifo_all, ff_all)
            res["by_bucket"] = {
                b: B0.pooled([B0.rate(x, f) for x in pd]) for b, f in BUCKETS.items()
            }
            co[g] = res
        out["ckpt"][c] = co
    return out


def finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def defects(res: dict) -> list[str]:
    """Exit 1 (PREREG §8): a NaN hit rate."""
    bad = []
    for c, co in res["ckpt"].items():
        for g, r in co.items():
            for b, v in r["by_bucket"].items():
                if not finite(v):
                    bad.append(f"ckpt{c}.{g}.hit.{b} = {v}")
    return bad


def classify(res: dict[int, dict]) -> tuple[dict[str, str], str]:
    labels = {}
    for g in GAMMAS:
        rs = [res[s]["ckpt"][HEADLINE][g]["cap"] for s in SEEDS]
        labels[g] = label([r["point"] for r in rs], [r["lo"] for r in rs])
    return labels, sweep_class(labels)


# --------------------------------------------------------------------------- #
# ledger rows
# --------------------------------------------------------------------------- #


def write_rows(led, res: dict[int, dict]) -> None:
    how = "run.py::measure_seed; seed order 0,1,2"
    boot = how + " -> B0.boot_rule (per document, 2000, paired)"
    sd = [res[s] for s in SEEDS]
    for r in ("fifo", "factfiller"):
        for b in BUCKETS:
            led.stat(f"U.hit.{r}.{b}", [x["refs"][r][b] for x in sd], how=how)
    for c in CHECKPOINTS:
        for g in GAMMAS:
            pre = f"B.ckpt{c}.{g}"
            rr = [x["ckpt"][c][g] for x in sd]
            for b in BUCKETS:
                led.stat(f"{pre}.hit.{b}", [x["by_bucket"][b] for x in rr], how=how)
            for f in ("point", "lo", "hi"):
                led.stat(f"{pre}.cap.{f}", [x["cap"][f] for x in rr], how=boot)
                led.stat(
                    f"{pre}.diff_fifo.{f}", [x["diff_fifo"][f] for x in rr], how=boot
                )
            led.stat(f"{pre}.cap.dropped", [x["cap_dropped"] for x in rr], how=how)
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
        "headline_checkpoint": f"B.ckpt{HEADLINE}",
        "gammas": GAMMAS,
        "already_seen": list(ALREADY_SEEN),
        "M": M,
        "S": S,
        "rule": (
            "lookahead-room TargetPolicy on discounted_returns(D, gamma); ties oldest"
        ),
        "refs": ["fifo", "factfiller"],
        "reading_rule": {
            "POINTS": f"cap CI lower >= {POINTS_LO} on all 3 seeds (ckpt{HEADLINE})",
            "PARTIAL": f"not POINTS and cap point >= {PARTIAL_POINT} on all 3 seeds",
            "MISSES": "otherwise",
            "class": "HORIZON if any POINTS; SIGNAL if every MISSES; MIXED otherwise",
        },
        "documents": "U = [64, 1088) then [4096, 4160), lookahead-room documents(seed)",
        "already_inspected": True,
        "bootstrap": {
            "n": B0.N_BOOT,
            "seed_base": B0.BOOT_SEED_BASE,
            "paired": True,
            "weights": "B0.boot_weights (identical replicates to B0)",
        },
        "inputs": {
            f"B.ckpt{c}.seed{s}": B0.d_rel(c, s) for c in CHECKPOINTS for s in SEEDS
        },
        "t0_manifest": str(manifest),
        "t0_manifest_sha256": B0.sha256(manifest),
        "b0_ledger": B0_LEDGER,
        "b0_ledger_sha256": B0.sha256(ROOT / B0_LEDGER),
        "device": "cpu",
    }


def execute(led, manifest: Path, base: Path, limit: int | None) -> Exit:
    rels = [B0.d_rel(c, s) for c in CHECKPOINTS for s in SEEDS]
    c1a = B0.verify_manifest(manifest, rels, base)
    led.note("C1.start", c1a, how="B0.verify_manifest before the first read")
    if not c1a["ok"]:
        led.status("did_not_run")
        return Exit.DID_NOT_RUN
    failed: list[str] = []
    b0_rows = {
        r["key"]: r
        for r in json.loads((ROOT / B0_LEDGER).read_text())["rows"]
        if isinstance(r.get("key"), str)
    }
    res: dict[int, dict] = {}
    for s in SEEDS:
        Ds = {c: B0.load_D(base, c, s) for c in CHECKPOINTS}
        docs, *_ = LR.documents(s)
        want = sorted(d.doc_id for d in docs)
        for c, Dc in Ds.items():
            if sorted(Dc) != want:
                failed.append(f"C2 ckpt{c} seed{s}: doc ids differ from U")
            bad = [k for k, D in Dc.items() if not B0.check_shape(D)]
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
                refs = {k: r["refs"][k]["all"] for k in ("fifo", "factfiller")}
                c3 = c3_failures(r["ckpt"][c], refs, b0_rows, c, s)
                failed += [f"C3: {x}" for x in c3]
        res[s] = r
        print(f"seed {s} done {r['seconds']:.0f}s", flush=True)
    c1b = B0.verify_manifest(manifest, rels, base)
    led.note("C1.end", c1b, how="B0.verify_manifest after the last computation")
    if not c1b["ok"]:
        failed.append("C1 at end: manifest mismatch")
    if not failed and sorted(res) != SEEDS:
        failed.append(f"C4: seeds run {sorted(res)} != {SEEDS}")
    if not failed and limit is None and any(res[s]["n_documents"] != N_DOCS for s in res):
        failed.append(f"C4: not {N_DOCS} documents per seed")
    led.note("controls_failed", failed, how="run.py::execute")
    led.run_meta(seeds_actually_run=sorted(res))
    if failed:
        for x in failed:
            print(x, flush=True)
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
    labels, cls = classify(res)
    led.note(f"B.ckpt{HEADLINE}.label", labels, how="run.py::classify (PREREG §6)")
    led.note("sweep_class", cls, how="run.py::classify (PREREG §6)")
    led.stat("seconds_per_seed", [res[s]["seconds"] for s in SEEDS], how="wall clock")
    led.status("ok")
    led.verdict(
        falsifier=FALSIFIER,
        outcome="inconclusive",
        detail=(
            f"descriptive run (PREREG §1): controls C1-C4 passed; reading rule "
            f"(PREREG §6) labels {labels}; sweep class {cls}"
        ),
    )
    print(json.dumps({"labels": labels, "class": cls}), flush=True)
    return Exit.OK


def main(argv: list[str] | None = None) -> Exit:
    ap = ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--base", type=Path, default=None, help="default: the main checkout")
    ap.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    ap.add_argument("--limit", type=int, default=None, help="smoke only; skips C3/C4")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    base = a.base or B0.main_checkout()
    if a.dry_run:
        rels = [B0.d_rel(c, s) for c in CHECKPOINTS for s in SEEDS]
        v = B0.verify_manifest(a.manifest, rels, base)
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
