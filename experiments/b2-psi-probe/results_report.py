"""Generate `experiments/b2-psi-probe/RESULTS.md` for B2 phase B (EVAL) and E0h.

Every number in the output is read from, or computed by this script from, a file
the run wrote: `runs/b2-psi-probe/ledger.json`, the phase-B payloads
`runs/b2-psi-probe/phaseB/ckpt{c}-seed{s}.pt`, their per-eviction logs
`ckpt{c}-seed{s}.logs.jsonl.gz`, the E0h outputs `phaseB/e0h-ckpt{c}.json` and
`ckpt{c}-seed{s}.e0h.pt`, the phase-A ledger `runs/b2-psi-probe-fit/ledger.json`
(+ its R² erratum), and E0d's ledger read from `run/e0d` with git. None is typed.

Checks it performs before writing (it exits 1 and writes nothing if one fails):
- `run.summarise_eval` re-run on the payload counts reproduces every payload
  summary (points, CIs, labels) exactly;
- `run.classify_all` on those summaries reproduces the ledger's classification;
- the FIFO residency replay is 1.0 on `gap_2_to_M` and 0.0 on `gap_gt_M`.

Per-arm accuracy CIs use the same paired resample as `run.paired_bootstrap`
(`torch.Generator(20260927 + seed)`, 2000 replicates, percentile 95%).

Usage (slot-wrapped):
    PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run --lane cpu-det \
        --slots 1 -- .venv/bin/python experiments/b2-psi-probe/results_report.py
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "b2-psi-probe"
RUN = ROOT / "runs" / "b2-psi-probe"
PB = RUN / "phaseB"
FIT = ROOT / "runs" / "b2-psi-probe-fit"
GIT = "/opt/homebrew/bin/git"
E0D_REF = "run/e0d"
OUT = EXP / "RESULTS.md"

sys.path.insert(0, str(ROOT / "src"))
from rsr.exit_codes import Exit, run_main  # noqa: E402
_spec = importlib.util.spec_from_file_location("b2run", EXP / "run.py")
B2 = importlib.util.module_from_spec(_spec)
sys.modules["b2run"] = B2
_spec.loader.exec_module(B2)

SEEDS = B2.SEEDS
CKPTS = (3000, 2500)
M = B2.M_STEPS
S = B2.S_STEPS
BUCKETS_SHOWN = ("all", "gap_2_to_M", "gap_gt_M", "gap_eq_M")
#: Arms re-read from the per-eviction jsonl. That file's `arm` field is the inner
#: policy's own label, not the tier's arm name: the kind-oracle logs as
#: "kind_oracle" (and so does kind_oldest), and all five random arms log as
#: "random". Only arms whose label is unique within ckpt3000 gamma=0.9 Tier 1 are
#: re-read here; random is taken from the payload (`attribution`, keyed correctly).
ATTR_ARMS = ("psiU", "psiC", "fifo", "ageU", "ageC", "kind", "oracle")
LOG_LABEL = {"kind_oracle": "kind"}
AGE_BANDS = ((1, 1), (2, 4), (5, 8), (9, 12), (13, 15), (16, 16), (17, S - 1))
STATUSES = ("assert:pending", "assert:querying", "assert:answered", "query", "filler")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def f4(x) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def ci(lo, hi) -> str:
    return f"[{lo:+.4f}, {hi:+.4f}]"


def pct(n, d) -> str:
    return "n/a" if not d else f"{100.0 * n / d:.1f}%"


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #


def rows_of(ledger: dict) -> dict:
    return {r["key"]: r.get("value", r) for r in ledger["rows"] if "key" in r}


def load_inputs() -> dict:
    led = json.loads((RUN / "ledger.json").read_text())
    man = json.loads((RUN / "manifest.json").read_text())
    fit = json.loads((FIT / "ledger.json").read_text())
    r2c = json.loads((FIT / "ledger.r2-corrected.json").read_text())
    pay = {
        c: {s: torch.load(PB / f"ckpt{c}-seed{s}.pt", weights_only=False) for s in SEEDS}
        for c in CKPTS
    }
    e0h = {c: json.loads((PB / f"e0h-ckpt{c}.json").read_text()) for c in CKPTS}
    e0h_pt = {
        c: {
            s: torch.load(PB / f"ckpt{c}-seed{s}.e0h.pt", weights_only=False)
            for s in SEEDS
        }
        for c in CKPTS
    }
    e0d_sha = subprocess.run(
        [GIT, "rev-parse", "--short", E0D_REF], capture_output=True, text=True, check=True
    ).stdout.strip()
    e0d = json.loads(
        subprocess.run(
            [GIT, "show", f"{E0D_REF}:runs/e0d/ledger.json"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    hashes = {
        str(p.relative_to(ROOT)): sha256(p)
        for p in sorted(PB.glob("ckpt*-seed*.pt")) + [RUN / "ledger.json"]
    }
    return {
        "led": led,
        "rows": rows_of(led),
        "man": man,
        "fit": rows_of(fit),
        "fit_led": fit,
        "r2c": r2c["rows"],
        "pay": pay,
        "e0h": e0h,
        "e0h_pt": e0h_pt,
        "e0d": rows_of(e0d),
        "e0d_led": e0d,
        "e0d_sha": e0d_sha,
        "hashes": hashes,
    }


def decisions_of(fit_rows: dict, c: int, s: int) -> dict:
    d = fit_rows[f"ckpt{c}.seed{s}.decisions.delta"]
    return {
        "ref": fit_rows[f"ckpt{c}.seed{s}.decisions.ref"],
        "delta": {float(k): v for k, v in d.items()},
    }


# --------------------------------------------------------------------------- #
# the counts each summary used (mirrors run.summarise_eval)
# --------------------------------------------------------------------------- #


def counts_by_key(payload: dict, label: int, n_tier2: int) -> dict[str, dict]:
    cnt = payload["counts"]
    out = {}
    if label == B2.HEADLINE:
        t1 = cnt["0.9.tier1"]
        out["0.9.tier1"] = t1
        cut = {a: {b: v[:n_tier2] for b, v in c.items()} for a, c in t1.items()}
        out["0.9.tier2"] = B2.merge_counts(cnt["0.9.tier2"], cut)
        base = t1
    else:
        out["0.9.tier2"] = cnt["0.9.tier2"]
        base = cnt["0.9.tier2"]
    shared = {
        a: {b: v[:n_tier2] for b, v in base[a].items()} for a in ("fifo", *B2.RANDOMS)
    }
    out["0.0.tier2"] = B2.merge_counts(cnt["0.0.tier2"], shared)
    return out


def boot_weights(D: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(B2.BOOT_SEED_BASE + seed)
    idx = torch.randint(0, D, (B2.N_BOOT, D), generator=g)
    return torch.zeros(B2.N_BOOT, D, dtype=torch.float64).scatter_add_(
        1, idx, torch.ones(B2.N_BOOT, D, dtype=torch.float64)
    )


def arm_rates(per_doc: dict[str, torch.Tensor], seed: int) -> dict[str, dict]:
    """Point and paired percentile CI of each arm's pooled rate (n, k per doc)."""
    arms = list(per_doc)
    D = per_doc[arms[0]].shape[0]
    W = boot_weights(D, seed)
    X = {a: per_doc[a].to(torch.float64).reshape(D, 2) for a in arms}
    rep = {a: (W @ x[:, 1]) / (W @ x[:, 0]) for a, x in X.items()}
    pt = {a: float(x[:, 1].sum() / x[:, 0].sum()) for a, x in X.items()}
    n = {a: int(x[:, 0].sum()) for a, x in X.items()}
    rnd = sorted(a for a in arms if B2._RANDOM_ARM.fullmatch(a))
    if rnd:
        rep["random"] = torch.stack([rep[a] for a in rnd]).mean(0)
        pt["random"] = sum(pt[a] for a in rnd) / len(rnd)
        n["random"] = n[rnd[0]]
    return {
        a: {
            "point": pt[a],
            "lo": float(torch.quantile(rep[a], 0.025)),
            "hi": float(torch.quantile(rep[a], 0.975)),
            "n": n[a],
        }
        for a in rep
    }


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #


def check_summaries(I: dict) -> dict:
    n2 = I["man"]["N_tier2"]
    worst, labels_ok, per = 0.0, True, {}
    for c in CKPTS:
        per[c] = {}
        for s in SEEDS:
            p = I["pay"][c][s]
            ev = {k: {"counts": v} for k, v in p["counts"].items()}
            again = B2.summarise_eval(ev, decisions_of(I["fit"], c, s), s, c, n2)
            per[c][s] = again
            for key, summ in p["summary"].items():
                for b, cons in summ["boot"].items():
                    for name, v in cons.items():
                        w = again[key]["boot"][b][name]
                        for f in ("point", "lo", "hi", "sd"):
                            worst = max(worst, abs(v[f] - w[f]))
                for arm, o in summ["outcome"].items():
                    labels_ok &= again[key]["outcome"][arm]["label"] == o["label"]
    cls = B2.classify_all(per)
    head = I["rows"]["classification"]
    cls_ok = all(cls["headline"][k] == head[k] for k in ("row", "class", "psiU", "psiC"))
    cls_ok &= cls["moving"] == I["rows"]["classification.moving"]
    return {"worst_abs_diff": worst, "labels_ok": labels_ok, "class_ok": cls_ok}


# --------------------------------------------------------------------------- #
# per-eviction logs: attribution beyond the payload, and residency replay
# --------------------------------------------------------------------------- #


def band_of(age: int) -> str:
    for lo, hi in AGE_BANDS:
        if lo <= age <= hi:
            return f"{lo}" if lo == hi else f"{lo}-{hi}"
    raise ValueError(age)


def scan_logs(c: int, s: int, gamma: float, tier: str, arms) -> dict:
    """One pass over one child's per-eviction log for (gamma, tier)."""
    victims: dict[str, dict[int, dict[int, int]]] = {a: defaultdict(dict) for a in arms}
    joint = {a: Counter() for a in arms}
    margins = {a: defaultdict(list) for a in arms}
    spread = {a: [] for a in arms}
    with gzip.open(PB / f"ckpt{c}-seed{s}.logs.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            a = LOG_LABEL.get(r["arm"], r["arm"])
            if r["gamma"] != gamma or r["tier"] != tier or a not in victims:
                continue
            victims[a][r["doc"]][r["step"]] = r["written_at"]
            st = r["kind"] if r["status"] is None else f"assert:{r['status']}"
            joint[a][(st, band_of(r["age"]))] += 1
            if r["margin"] is not None:
                margins[a][st].append(r["margin"])
                spread[a].append(max(r["psi"]) - min(r["psi"]))
    return {
        "victims": victims,
        "joint": joint,
        "margins": margins,
        "spread": spread,
    }


def replay(seed: int, victims: dict[str, dict[int, dict[int, int]]], docs_ids) -> dict:
    """Model-free residency of each arm's logged victim sequence
    (`rsr.metrics.headroom.simulate`), per document per bucket, plus the query gap
    of every pending-assert victim (for "lost within gap M")."""
    per = {a: {b: [] for b in BUCKETS_SHOWN} for a in victims}
    pend_gap = {a: Counter() for a in victims}
    for d_id in docs_ids:
        doc = B2.doc_by_id(seed, d_id)
        q_of = dict(doc.pairs)
        for a, vv in victims.items():
            v = vv[d_id]
            q = B2.simulate(doc, B2.LR.Scripted(v), M)["queries"]
            g = torch.tensor([x["gap"] for x in q])
            h = torch.tensor([bool(x["hit"]) for x in q])
            for b in BUCKETS_SHOWN:
                sel = B2.BUCKETS[b](g)
                per[a][b].append((int(sel.sum()), int((h & sel).sum())))
            for t, w in v.items():
                if w in q_of and q_of[w] > t:
                    pend_gap[a]["le_M" if q_of[w] - w <= M else "gt_M"] += 1
    return {
        "counts": {
            a: {b: torch.tensor(x, dtype=torch.long) for b, x in bb.items()}
            for a, bb in per.items()
        },
        "pend_gap": pend_gap,
    }


def qsum(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    t = torch.tensor(xs, dtype=torch.float64)
    return {
        "n": len(xs),
        "q10": float(torch.quantile(t, 0.1)),
        "q50": float(torch.quantile(t, 0.5)),
        "q90": float(torch.quantile(t, 0.9)),
        "mean": float(t.mean()),
    }


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

ARM_NAME = {
    "psiU": "ψ̂-U",
    "psiC": "ψ̂-C",
    "psiU+": "ψ̂-U⁺",
    "psiC+": "ψ̂-C⁺",
    "psiU1": "ψ̂-U-1",
    "psiU2": "ψ̂-U-2",
    "psiU3": "ψ̂-U-3",
    "fifo": "FIFO",
    "ageU": "age-only (U)",
    "ageC": "age-only (C)",
    "random": "random (5-seed mean)",
    "oracle": "oracle",
    "kind": "kind-oracle (random tie)",
    "kind_oldest": "kind-oracle (oldest tie)",
    "factfiller": "fact/filler",
}
REF_ARM = {"fifo": "FIFO", "age": "age-only"}


def acc_table(rates: dict[str, dict], arms) -> list[str]:
    L = [
        "| arm | all | gap_2_to_M | gap_gt_M | gap_eq_M |",
        "|---|---|---|---|---|",
    ]
    for a in arms:
        if a not in rates["all"]:
            continue
        cells = [
            f"{f4(rates[b][a]['point'])} {ci(rates[b][a]['lo'], rates[b][a]['hi'])}"
            for b in BUCKETS_SHOWN
        ]
        L.append(f"| {ARM_NAME.get(a, a)} | " + " | ".join(cells) + " |")
    return L


def n_line(rates) -> str:
    a = next(iter(rates["all"]))
    return "Queries per bucket (pooled over documents): " + ", ".join(
        f"{b} {rates[b][a]['n']}" for b in BUCKETS_SHOWN
    )


def contrast_rows(summ: dict, seed: int, arms_ref) -> list[str]:
    out = []
    for arm, rk in arms_ref:
        k = f"{arm}_minus_ref{rk}"
        if k not in summ["boot"]["all"]:
            continue
        a = summ["boot"]["all"][k]
        n = summ["boot"]["gap_2_to_M"][k]
        g = summ["boot"]["gap_gt_M"][k]
        r = summ["boot"]["all"][f"{arm}_minus_random"]
        o = summ["outcome"][arm]
        out.append(
            f"| {seed} | {ARM_NAME[arm]} | {REF_ARM[summ['ref'][rk]]} | "
            f"{f4(summ['delta'])} | {a['point']:+.4f} {ci(a['lo'], a['hi'])} | "
            f"{n['point']:+.4f} {ci(n['lo'], n['hi'])} | "
            f"{g['point']:+.4f} {ci(g['lo'], g['hi'])} | "
            f"{r['point']:+.4f} {ci(r['lo'], r['hi'])} | **{o['label']}**"
            + (f" {o['flags']}" if o["flags"] else "")
            + " |"
        )
    return out


CONTRAST_HEAD = [
    "| seed | arm | ref (FIT_VAL) | δ | Δ all [95% CI] | Δ gap_2_to_M | Δ gap_gt_M | "
    "arm − random (all) | outcome |",
    "|---|---|---|---|---|---|---|---|---|",
]
ARMS_REF = (
    ("psiU", "U"),
    ("psiC", "C"),
    ("psiU+", "U"),
    ("psiC+", "C"),
    ("psiU1", "U"),
    ("psiU2", "U"),
    ("psiU3", "U"),
)


def q2_rows(summ, seed) -> list[str]:
    out = []
    for k, name in (("q2_psiU_minus_kind", "ψ̂-U"), ("q2_psiC_minus_kind", "ψ̂-C")):
        if k in summ["boot"]["all"]:
            q = summ["boot"]["all"][k]
            out.append(
                f"| {seed} | {name} − kind-oracle | {q['point']:+.4f} {ci(q['lo'], q['hi'])} "
                f"| {f4(summ['delta'])} | **{summ['outcome'][k]['label']}** |"
            )
    return out


def attr_payload_rows(att: dict, arms) -> list[str]:
    L = [
        "| arm | evictions | mean victim age | age 1 (newest) | age 16 (= M, FIFO's) | "
        "victim rank 1 (oldest) | pending-assert victims | querying-assert | "
        "answered-assert | query | filler | ψ̂ margin q10 / q50 / q90 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for a in arms:
        if a not in att:
            continue
        v = att[a]
        n = sum(v["age_hist"])
        mean_age = sum((i + 1) * c for i, c in enumerate(v["age_hist"])) / n
        ks = v["kind_status"]
        mg = v["margin"]
        m = (
            "—"
            if not mg.get("n")
            else f"{mg['q10']:.4f} / {mg['q50']:.4f} / {mg['q90']:.4f}"
        )
        L.append(
            f"| {ARM_NAME.get(a, a)} | {n} | {mean_age:.2f} | {pct(v['age_hist'][0], n)} | "
            f"{pct(v['age_hist'][M - 1], n)} | {pct(v['rank_hist'][0], n)} | "
            + " | ".join(pct(ks.get(k, 0), n) for k in STATUSES)
            + f" | {m} |"
        )
    return L


def main() -> Exit:
    I = load_inputs()
    led, rows, man = I["led"], I["rows"], I["man"]
    n2 = man["N_tier2"]
    chk = check_summaries(I)
    if chk["worst_abs_diff"] != 0.0 or not chk["labels_ok"] or not chk["class_ok"]:
        print(f"CHECK FAILED: {chk}", file=sys.stderr)
        return Exit.FAIL

    rates = {}  # (c, key, s) -> bucket -> arm -> stats
    for c in CKPTS:
        for s in SEEDS:
            cb = counts_by_key(I["pay"][c][s], c, n2)
            for key, cnt in cb.items():
                rates[(c, key, s)] = {
                    b: arm_rates({a: cc[b] for a, cc in cnt.items()}, s)
                    for b in BUCKETS_SHOWN
                }

    # logs: ckpt3000 gamma 0.9 tier1 (gating), every seed
    logs, res = {}, {}
    for s in SEEDS:
        logs[s] = scan_logs(3000, s, 0.9, "tier1", ATTR_ARMS)
        ids = sorted(logs[s]["victims"]["fifo"])
        res[s] = replay(s, logs[s]["victims"], ids)
        # the replayed-residency control: FIFO holds every gap <= M, none beyond
        rr = arm_rates({a: v["gap_2_to_M"] for a, v in res[s]["counts"].items()}, s)
        rg = arm_rates({a: v["gap_gt_M"] for a, v in res[s]["counts"].items()}, s)
        if rr["fifo"]["point"] != 1.0 or rg["fifo"]["point"] != 0.0:
            print(f"CHECK FAILED: FIFO replay seed {s}: {rr['fifo']} {rg['fifo']}")
            return Exit.FAIL
    resid = {
        s: {
            b: arm_rates({a: v[b] for a, v in res[s]["counts"].items()}, s)
            for b in BUCKETS_SHOWN
        }
        for s in SEEDS
    }

    L: list[str] = []
    w = L.append
    prov = led["provenance"]
    cls = rows["classification"]
    comp = rows["classification.companion"]
    g0 = rows["classification.gamma0_secondary"]
    c25 = rows["classification.ckpt2500"]
    w("# B2 — the ψ̂ probe: phase B (EVAL) results, and E0h")
    w("")
    w(
        "**Generated** by `experiments/b2-psi-probe/results_report.py` from the run's own "
        "files; no number below is typed. Written 2026-09-29 by a model (Claude Opus 5.5, "
        "an `rsr-researcher` session). **Not written by Brendan.** PREREG: "
        "`experiments/b2-psi-probe/PREREG.md` (base `"
        f"{man['prereg_commits']['base']}`, Amendment 1 `{man['prereg_commits']['amendment1']}`, "
        f"B0 addendum `{man['prereg_commits']['b0_addendum']}`, TBD-1/2/3 addenda). "
        "Phase A (fits, λ, ref, δ, N_E) is in `RESULTS-phaseA.md`."
    )
    w("")
    w("## 1. Headline")
    w("")
    w(
        f"- **Classification (§9.6, ckpt3000, γ = 0.9, gating): row {cls['row']}, "
        f"{cls['class']}.** ψ̂-U = {' / '.join(cls['psiU'])}; ψ̂-C = "
        f"{' / '.join(cls['psiC'])} (seeds 0 / 1 / 2). UNDERPOWERED: "
        f"{cls['underpowered']}."
    )
    w(
        "- **Row 1's rule: no build recommendation.** This report makes none. The "
        "attribution the row requires is §5."
    )
    losers = [
        f"{ARM_NAME[arm]} seed {s}"
        for arm in ("psiU", "psiC")
        for s, lab in zip(SEEDS, cls[arm])
        if lab == "LOSS"
    ]
    w(
        f"- **Which arm and which seed lose:** {', '.join(losers)}. Every other gating cell is "
        "EQUIV; none is WIN."
    )
    w(
        f"- Companion (U⁺, C⁺; non-gating, A1.2): {comp['class']} (ψ̂-U⁺ "
        f"{' / '.join(comp['psiU+'])}; ψ̂-C⁺ {' / '.join(comp['psiC+'])}). "
        f"ckpt2500 (reported, never gating): {c25['class']}. **moving** = "
        f"{rows['classification.moving']}. γ = 0 (secondary): ckpt3000 "
        f"{g0['3000']['class']}, ckpt2500 {g0['2500']['class']}."
    )
    e3 = I["e0h"][3000]
    w(
        "- **E0h (falsifier 3c, its own rc, A1.3): rc 2 (unratified)** at both "
        "checkpoints. Headline within-step R² of ψ̂-U(γ = 0) on the pre-hook logits, "
        "ckpt3000: "
        + " / ".join(f4(e3["e0h.r2_headline"][str(s)]) for s in SEEDS)
        + f". The unratified reading of the proposed thresholds is "
        f"{e3['e0h.class_unratified_reading']}; no class is read until Brendan rules (§10)."
    )
    w(
        "- **B2's rc is not E0h's rc.** B2 EVAL exited 0 (`b2-eval.rc`); E0h exited 2 at "
        "each checkpoint."
    )
    w("")
    w("### 1.1 The ledger's verdict field")
    w("")
    v = led["verdict"]
    w(
        f"`ledger.json` `verdict.outcome` = `{v['outcome']}`, detail `{v['detail']}`. This is "
        "the runner's intended mapping, not a defect: `run.py::run_eval` writes "
        '`outcome="inconclusive"` for **every** class once status is `ok`, because '
        "`scripts/ledger.py` admits only `survived | falsified | inconclusive` and B2 is not "
        "a spec falsifier (its `falsifier` field: B2 is kill-gate evidence, E0h is 3c). The "
        "scientific classification is the `classification` row. **Exit code:** PREREG §12 "
        "(file line 535) says HARMFUL is exit 0, and the parent exited 0; the two agree."
    )
    w("")

    # ---------------- headline table
    w("## 2. Headline table — ckpt3000, γ = 0.9, Tier 1 (gating), N_E documents per seed")
    w("")
    w(
        "Model-read answer accuracy (§9.1, S0-03's `correct`), pooled over queries, with "
        "the paired percentile 95% CI (2000 per-document resamples, generator "
        "`20260927 + seed`, the same resample as `run.paired_bootstrap`). "
        '"Hit rate" in the brief = this accuracy; model-free **residency** is §2.2.'
    )
    tier1_arms = ("psiU", "psiC", "fifo", "ageU", "ageC", "random", "oracle", "kind")
    for s in SEEDS:
        r = rates[(3000, "0.9.tier1", s)]
        w("")
        w(f"**Seed {s}.** {n_line(r)}.")
        w("")
        L.extend(acc_table(r, tier1_arms))
    w("")
    w("### 2.1 Gating contrasts (§9.5) and Q2 (§9.7), ckpt3000 γ = 0.9 Tier 1")
    w("")
    w(
        "Δ = arm − ref, paired. `ref` is chosen on FIT_VAL (§9.3; the phase-A ledger "
        "`decisions.ref`); δ = 0.25 × (oracle − FIFO) all-query accuracy on FIT_VAL "
        "(`decisions.delta`). Non-inferiority is the `gap_2_to_M` Δ lower bound > −δ; "
        "beats-random is the arm − random lower bound > 0."
    )
    w("")
    L.extend(CONTRAST_HEAD)
    for s in SEEDS:
        L.extend(
            contrast_rows(I["pay"][3000][s]["summary"]["0.9.tier1"], s, ARMS_REF[:2])
        )
    w("")
    w("| seed | Q2 contrast | Δ [95% CI] | δ | outcome |")
    w("|---|---|---|---|---|")
    for s in SEEDS:
        L.extend(q2_rows(I["pay"][3000][s]["summary"]["0.9.tier1"], s))
    w("")
    w("**EVAL headroom** (oracle − FIFO, all-query, reported; δ comes from FIT_VAL):")
    w("")
    for s in SEEDS:
        o = I["pay"][3000][s]["summary"]["0.9.tier1"]["boot"]["all"]["oracle_minus_fifo"]
        w(f"- seed {s}: {o['point']:+.4f} {ci(o['lo'], o['hi'])}")
    w("")
    w(
        "### 2.2 Residency (model-free replay of each arm's logged victims), ckpt3000 γ = 0.9 Tier 1"
    )
    w("")
    w(
        "`rsr.metrics.headroom.simulate` on each arm's logged `(step, written_at)` victims "
        "(`phaseB/ckpt3000-seed{s}.logs.jsonl.gz`); in-loop residency equalled this replay "
        "for every arm and document during the run (§11 control 7). Check: FIFO replays "
        "to 1.0000 on `gap_2_to_M` and 0.0000 on `gap_gt_M` on every seed."
    )
    for s in SEEDS:
        w("")
        w(f"**Seed {s}.**")
        w("")
        L.extend(acc_table(resid[s], tier1_arms))
    w("")

    # ---------------- every other cell
    w("## 3. Every other cell (per arm, seed and cell)")
    w("")
    for c, key, title in (
        (
            3000,
            "0.9.tier2",
            "ckpt3000, γ = 0.9, Tier 2 (first N_tier2 documents; companions and U-r, non-gating)",
        ),
        (3000, "0.0.tier2", "ckpt3000, γ = 0 (secondary, §9.8)"),
        (2500, "0.9.tier2", "ckpt2500, γ = 0.9 (reported, never gating)"),
        (2500, "0.0.tier2", "ckpt2500, γ = 0 (secondary)"),
    ):
        w(f"### {title}")
        w("")
        L.extend(CONTRAST_HEAD)
        for s in SEEDS:
            L.extend(contrast_rows(I["pay"][c][s]["summary"][key], s, ARMS_REF))
        w("")
        w("| seed | Q2 contrast | Δ [95% CI] | δ | outcome |")
        w("|---|---|---|---|---|")
        for s in SEEDS:
            L.extend(q2_rows(I["pay"][c][s]["summary"][key], s))
        w("")
        arms = [
            a
            for a in (
                "psiU",
                "psiC",
                "psiU+",
                "psiC+",
                "psiU1",
                "psiU2",
                "psiU3",
                "fifo",
                "ageU",
                "ageC",
                "random",
                "oracle",
                "kind",
                "kind_oldest",
                "factfiller",
            )
        ]
        for s in SEEDS:
            r = rates[(c, key, s)]
            w(f"<details><summary>Seed {s}: accuracy per arm ({n_line(r)})</summary>")
            w("")
            L.extend(acc_table(r, arms))
            w("")
            w("</details>")
            w("")
    w(
        "At γ = 0, δ is the γ = 0.9 δ of the same seed (A1.11). At ckpt3000 the γ = 0 and "
        "Tier 2 arms share FIFO and random with Tier 1's first N_tier2 documents (paired)."
    )
    w("")

    # ---------------- attribution
    w("## 4. Age controls (§8; reported, not gates)")
    w("")
    w(
        "| ckpt | seed | age decodability R² U | C | corr(ψ̂-U, age) in-loop Pearson / Spearman | corr(ψ̂-C, age) |"
    )
    w("|---|---|---|---|---|---|")
    for c in CKPTS:
        for s in SEEDS:
            dU = I["r2c"][f"ckpt{c}.seed{s}.age_decodability.U"]["corrected"]
            dC = I["r2c"][f"ckpt{c}.seed{s}.age_decodability.C"]["corrected"]
            pc = I["pay"][c][s]["psi_age_corr"]
            kU = "0.9.tier1.psiU" if c == 3000 else "0.9.tier2.psiU"
            kC = kU.replace("psiU", "psiC")
            w(
                f"| {c} | {s} | {f4(dU)} | {f4(dC)} | {f4(pc[kU]['pearson'])} / "
                f"{f4(pc[kU]['spearman'])} | {f4(pc[kC]['pearson'])} / {f4(pc[kC]['spearman'])} |"
            )
    w("")
    w(
        "Age decodability is the **corrected** FIT_VAL R² (`runs/b2-psi-probe-fit/"
        "ledger.r2-corrected.json`, erratum P0.2: the raw ledger value used "
        "`1 − n·SSE/SST`)."
    )
    w("")
    w("## 5. Row-1 attribution (PREREG §4, §9.6 row 1), against FIFO")
    w("")
    w(
        "Source: `ProbeArgminPolicy`'s per-eviction log, aggregated in the payload "
        "(`attribution`) and re-read from `ckpt{c}-seed{s}.logs.jsonl.gz`. One eviction per "
        "full-memory step: 32 per document. Percentages are of that arm's evictions. "
        "Victim rank is 1-based oldest-to-newest (`rank_shift`); displacement = rank − 1 "
        "(0 under FIFO always)."
    )
    for s in SEEDS:
        att = {
            a.split(".", 3)[3]: v
            for a, v in I["pay"][3000][s]["attribution"].items()
            if a.startswith("0.9.tier1.")
        }
        w("")
        w(
            f"### Seed {s} — ckpt3000 γ = 0.9 Tier 1 (ψ̂-U {cls['psiU'][s]}, ψ̂-C {cls['psiC'][s]})"
        )
        w("")
        L.extend(
            attr_payload_rows(
                att, ("psiU", "psiC", "fifo", "ageU", "ageC", "kind", "oracle", "random0")
            )
        )
        w("")
        # displacement
        for a in ("psiU", "psiC"):
            rs = att[a]["rank_shift"]
            n = sum(rs.values())
            w(
                f"- {ARM_NAME[a]} displacement (rank shift) distribution: "
                + ", ".join(f"{k}: {pct(v_, n)}" for k, v_ in rs.items())
                + "."
            )
        w("")
        # joint status x age band, from logs
        lg = logs[s]
        w(
            f"**Victim status × age band (from the log), seed {s}.** Cells are % of the arm's evictions."
        )
        w("")
        bands = [band_of(lo) for lo, _ in AGE_BANDS]
        w("| arm | status | " + " | ".join(f"age {b}" for b in bands) + " | total |")
        w("|---|---|" + "---|" * (len(bands) + 1))
        for a in ("psiU", "psiC", "fifo", "ageU"):
            n = sum(lg["joint"][a].values())
            for st in STATUSES:
                cells = [pct(lg["joint"][a].get((st, b), 0), n) for b in bands]
                tot = sum(lg["joint"][a].get((st, b), 0) for b in bands)
                w(
                    f"| {ARM_NAME[a]} | {st} | "
                    + " | ".join(cells)
                    + f" | {pct(tot, n)} |"
                )
        w("")
        w(
            f"**Pending-assert victims whose query is within gap M** (a query FIFO would "
            f"have answered from memory), seed {s}:"
        )
        w("")
        for a in ("psiU", "psiC", "fifo", "ageU", "kind", "oracle"):
            pg = res[s]["pend_gap"][a]
            w(
                f"- {ARM_NAME[a]}: {pg.get('le_M', 0)} with gap ≤ M, {pg.get('gt_M', 0)} "
                "with gap > M"
            )
        w("")
        w(f"**ψ̂ margin (second-lowest − lowest) by victim status, seed {s}:**")
        w("")
        w("| arm | status | n | q10 | median | q90 | mean |")
        w("|---|---|---|---|---|---|---|")
        for a in ("psiU", "psiC"):
            for st in STATUSES:
                q = qsum(lg["margins"][a].get(st, []))
                if q["n"]:
                    w(
                        f"| {ARM_NAME[a]} | {st} | {q['n']} | {q['q10']:.4f} | "
                        f"{q['q50']:.4f} | {q['q90']:.4f} | {q['mean']:.4f} |"
                    )
            sp = qsum(lg["spread"][a])
            w(
                f"| {ARM_NAME[a]} | *ψ̂ spread over live slots (max − min)* | {sp['n']} | "
                f"{sp['q10']:.4f} | {sp['q50']:.4f} | {sp['q90']:.4f} | {sp['mean']:.4f} |"
            )
        w("")
    w("### ckpt2500, γ = 0.9 (reported, never gating) — payload attribution")
    for s in SEEDS:
        att = {
            a.split(".", 3)[3]: v
            for a, v in I["pay"][2500][s]["attribution"].items()
            if a.startswith("0.9.tier2.")
        }
        w("")
        w(f"Seed {s} (ψ̂-U {c25['psiU'][s]}, ψ̂-C {c25['psiC'][s]}):")
        w("")
        L.extend(attr_payload_rows(att, ("psiU", "psiC", "fifo", "kind")))
    w("")

    # ---------------- summary numbers for the note
    def share(s, a, st):
        n = sum(logs[s]["joint"][a].values())
        return sum(v_ for (k, _b), v_ in logs[s]["joint"][a].items() if k == st) / n

    def lostM(s, a):
        return res[s]["pend_gap"][a].get("le_M", 0)

    w("## 6. Attribution summary and an interpretive note")
    w("")
    w("**Measured (from §5):**")
    w("")
    for s in SEEDS:
        a1 = I["pay"][3000][s]["attribution"]
        hu = a1["0.9.tier1.psiU"]["age_hist"]
        hc = a1["0.9.tier1.psiC"]["age_hist"]
        w(
            f"- Seed {s} (ψ̂-U {cls['psiU'][s]}): ψ̂-U evicts the newest slot (age 1) at "
            f"{pct(hu[0], sum(hu))} of steps and FIFO's age-16 slot at {pct(hu[M - 1], sum(hu))}; "
            f"ψ̂-C at {pct(hc[0], sum(hc))} / {pct(hc[M - 1], sum(hc))}. Pending-assert share of "
            f"victims: ψ̂-U {100 * share(s, 'psiU', 'assert:pending'):.1f}%, ψ̂-C "
            f"{100 * share(s, 'psiC', 'assert:pending'):.1f}%, FIFO "
            f"{100 * share(s, 'fifo', 'assert:pending'):.1f}%, oracle "
            f"{100 * share(s, 'oracle', 'assert:pending'):.1f}%. Pending victims whose "
            f"query is within gap M: ψ̂-U {lostM(s, 'psiU')}, ψ̂-C {lostM(s, 'psiC')}, "
            f"FIFO {lostM(s, 'fifo')}."
        )
    w("")
    for s in SEEDS:
        ra = rates[(3000, "0.9.tier1", s)]["all"]
        rs_ = resid[s]["all"]
        w(
            f"- Seed {s}, residency vs model-read accuracy (all queries): ψ̂-U "
            f"{f4(rs_['psiU']['point'])} resident / {f4(ra['psiU']['point'])} correct; "
            f"FIFO {f4(rs_['fifo']['point'])} / {f4(ra['fifo']['point'])}; ψ̂-C "
            f"{f4(rs_['psiC']['point'])} / {f4(ra['psiC']['point'])}. Correct ÷ resident: "
            f"ψ̂-U {ra['psiU']['point'] / rs_['psiU']['point']:.3f}, FIFO "
            f"{ra['fifo']['point'] / rs_['fifo']['point']:.3f}, ψ̂-C "
            f"{ra['psiC']['point'] / rs_['psiC']['point']:.3f}."
        )
    eq_noninf = []
    for s in SEEDS:
        su = I["pay"][3000][s]["summary"]["0.9.tier1"]
        for arm, rk in (("psiU", "U"), ("psiC", "C")):
            n = su["boot"]["gap_2_to_M"][f"{arm}_minus_ref{rk}"]
            if su["outcome"][arm]["label"] == "EQUIV" and n["lo"] <= -su["delta"]:
                eq_noninf.append(
                    f"{ARM_NAME[arm]} seed {s} (gap_2_to_M Δ lo {n['lo']:+.4f}, −δ {-su['delta']:+.4f})"
                )
    w(
        "- **EQUIV cells whose `gap_2_to_M` Δ fails the non-inferiority bound** (§9.5 "
        "checks non-inferiority only for WIN, so these are EQUIV as the rule is written; "
        "reported, not reclassified): "
        + ("; ".join(eq_noninf) if eq_noninf else "none")
        + "."
    )
    w("")
    lower_ratio = [
        x
        for x in SEEDS
        if rates[(3000, "0.9.tier1", x)]["all"]["psiU"]["point"]
        / resid[x]["all"]["psiU"]["point"]
        < rates[(3000, "0.9.tier1", x)]["all"]["fifo"]["point"]
        / resid[x]["all"]["fifo"]["point"]
    ]
    newest = [
        s
        for s in SEEDS
        if max(
            range(S - 1),
            key=lambda i: I["pay"][3000][s]["attribution"]["0.9.tier1.psiU"]["age_hist"][
                i
            ],
        )
        == 0
    ]
    more_pend = [
        s
        for s in SEEDS
        if share(s, "psiU", "assert:pending") > share(s, "fifo", "assert:pending")
    ]
    more_lost = [s for s in SEEDS if lostM(s, "psiU") > lostM(s, "fifo")]
    e0d_pop = lambda s: I["e0d"][f"seed{s}.analysis"]["populations"]["bos_excluded"]
    w(
        "**Interpretive note — INFERENCE, not a measurement.** E0d (read first, per PREREG "
        f"L80; `{E0D_REF}` @ `{I['e0d_sha']}`; labels {I['e0d']['e0d.labels']}, verdict "
        f"`{I['e0d_led']['verdict']['outcome']}`) found that the gated `r_i` ranks the "
        "LOO-critical slot well — AUROC_strat,pct (bos-excluded population) "
        + " / ".join(f4(e0d_pop(s)["a2"]["gated_resample"]["r"]["pct"]) for s in SEEDS)
        + " — while agreement at the bottom of the ranking is poor: ρ_Q "
        + " / ".join(f4(e0d_pop(s)["stats"]["gated"]["resample"]["rho_Q"]) for s in SEEDS)
        + ", bottom-1 agreement "
        + " / ".join(
            f4(e0d_pop(s)["stats"]["gated"]["resample"]["bottom1_Q"]) for s in SEEDS
        )
        + " against chance "
        + f4(e0d_pop(0)["stats"]["gated"]["resample"]["bottom1_chance"])
        + "; seed 1's sink probe (fraction of pairs with the assert above the read query) is "
        + f4(e0d_pop(1)["sink_probe"]["gated"]["assert_above_read_query"]["fraction"])
        + ". Eviction is an argmin, so it uses exactly the part of the ranking E0d found "
        "weak. The attribution is consistent with that reading: ψ̂-U's modal victim is the "
        f"newest slot (age 1) on seeds {newest}; it evicts a larger share of pending "
        f"asserts than FIFO on seeds {more_pend}; and more of its pending victims still "
        f"have their query within gap M than FIFO's on seeds {more_lost} — which is where "
        "a `gap_2_to_M` loss would come from; on the seed where ψ̂-U is EQUIV the same "
        "within-M losses are offset by `gap_gt_M` gains (§2.1). A second, separate "
        "signal: ψ̂-U turns residency into correct answers at a lower rate than FIFO on "
        f"seeds {lower_ratio} (and not on seeds {[x for x in SEEDS if x not in lower_ratio]}; "
        "the correct ÷ resident ratios above). Where it is lower, a slot kept out of FIFO "
        "order may be read less well: the ADR-0006 rank shift of ψ̂-U's evictions is large "
        "(§5), and a displaced slot's position no longer follows from its age. That is not "
        "tested here. ψ̂-C, whose censored target is age-shaped "
        "(§13), keeps most evictions at the old end and stays EQUIV. **The mechanism is an "
        "inference from co-occurring statistics; nothing here isolates it.** This report "
        "does not show that the bottom-of-ranking failure *causes* the LOSS, nor that a "
        "different target or form would avoid it, and it makes no build recommendation "
        "(row 1)."
    )
    w("")

    # ---------------- E0h
    w("## 7. E0h (falsifier 3c) — its own entry point and rc")
    w("")
    w(
        "Command (each, slot-wrapped): `PYTHONPATH=scripts .venv/bin/python -m "
        "orchestrator.slot run --lane cpu-det --slots 1 --wait 600 -- .venv/bin/python "
        "experiments/b2-psi-probe/run.py e0h [--checkpoint 2500]`. Output files "
        "`phaseB/e0h-ckpt3000.json`, `phaseB/e0h-ckpt2500.json`. Neither printed anything; "
        "both returned **rc 2** (`E0H_RULING` is `None`: unratified, A1.3)."
    )
    w("")
    w("| ckpt | seed | head | within-step R² | pooled R² | n rows |")
    w("|---|---|---|---|---|---|")
    for c in CKPTS:
        for s in SEEDS:
            for k, name in (
                ("U@0.0", "ψ̂-U(γ=0) **headline**"),
                ("C@0.0", "ψ̂-C(γ=0)"),
                ("U+@0.0", "ψ̂-U⁺(γ=0), k ≥ 1 companion"),
                ("target_D", "γ=0 target D itself (reference)"),
            ):
                v_ = I["e0h"][c]["e0h.per_seed"][str(s)][k]
                w(
                    f"| {c} | {s} | {name} | {f4(v_['within'])} | {f4(v_['pooled'])} | {v_['n']} |"
                )
    w("")
    for c in CKPTS:
        e = I["e0h"][c]
        worst = max(I["e0h_pt"][c][s]["logit_control_worst"] for s in SEEDS)
        w(
            f"- ckpt{c}: rc {e['rc']}; ratified {e['e0h.ratified']}; unratified reading "
            f"{e['e0h.class_unratified_reading']}; regressor source "
            f"`{e['e0h.regressor_source']}`; pooled {e['e0h.pooled']}; A1.5 logit control "
            f"worst {worst:.3g} (tolerance {B2.LOGIT_TOL}); k0-inherited seeds "
            f"{e['e0h.k0_inherited_seeds']}; E0h row errors "
            f"{sum(len(I['e0h_pt'][c][s]['errors']) for s in SEEDS)}."
        )
    w("")
    tgt = [e3["e0h.per_seed"][str(s)]["target_D"]["within"] for s in SEEDS]
    upl = [e3["e0h.per_seed"][str(s)]["U+@0.0"]["within"] for s in SEEDS]
    hr = [e3["e0h.r2_headline"][str(s)] for s in SEEDS]
    w(
        "Reading, **unclassified** (0.90 / 0.49 are PROPOSED, not ratified): at ckpt3000, "
        f"{sum(x >= B2.E0H_COLLINEAR for x in hr)} of 3 seeds reach 0.90 and "
        f"{sum(x >= B2.E0H_NOT_COLLINEAR for x in hr)} of 3 are at or above 0.49. The γ = 0 "
        "target itself (§10's reference) has within-step R² "
        + " / ".join(f4(x) for x in tgt)
        + " on the same logits, against ψ̂-U(γ=0)'s "
        + " / ".join(f4(x) for x in hr)
        + ". The k ≥ 1 companion ψ̂-U⁺(γ=0) is "
        + " / ".join(f4(x) for x in upl)
        + ". Brendan rules; no kill is read."
    )
    w("")

    # ---------------- §13
    w("## 8. Scoring §13 (the author's pre-data expectation)")
    w("")
    headU = cls["psiU"]
    near = []
    for s in SEEDS:
        su = I["pay"][3000][s]["summary"]["0.9.tier1"]["boot"]["all"]
        near.append(
            abs(su["psiC_minus_refC"]["point"]) < abs(su["psiU_minus_refU"]["point"])
        )
    q2 = [
        I["pay"][3000][s]["summary"]["0.9.tier1"]["outcome"]["q2_psiU_minus_kind"][
            "label"
        ]
        for s in SEEDS
    ]
    e3r = [e3["e0h.r2_headline"][str(s)] for s in SEEDS]
    agedU = [I["r2c"][f"ckpt3000.seed{s}.age_decodability.U"]["corrected"] for s in SEEDS]
    agedC = [I["r2c"][f"ckpt3000.seed{s}.age_decodability.C"]["corrected"] for s in SEEDS]
    w(
        f"- **Class odds** (HARMFUL ~40%, MIXED ~40%, NOT LEARNABLE ~10%, NOT RULED OUT ~10%). "
        f"Realised: **{cls['class']}**. Probability the author gave the realised class: 0.40 "
        "(log score ln 0.40 = −0.92; multi-class Brier over the four named bins = "
        "(1−0.4)² + 0.4² + 0.1² + 0.1² = 0.54). The modal pair (HARMFUL or MIXED, 80%) held."
    )
    w(
        f"- **ψ̂-U LOSS or UNRESOLVED on seed 1:** realised {headU[1]} — held. Its stated basis "
        "(F2's seed-1 reversal) was corrected by the B0 addendum (F2 compared asserts with a "
        "pool that counted queries as filler), and ψ̂-U also lost on seed 0, which the "
        "expectation did not single out. Right outcome; the stated reason is not supported."
    )
    w(
        "- **ψ̂-C closer to FIFO than ψ̂-U** (|Δ ψ̂-C − ref_C| < |Δ ψ̂-U − ref_U|, all-query): "
        + " / ".join(str(x) for x in near)
        + " — held on "
        + f"{sum(near)} of 3."
    )
    w(
        "- **Age decodability substantial:** corrected FIT_VAL R², ckpt3000, U "
        + " / ".join(f4(x) for x in agedU)
        + ", C "
        + " / ".join(f4(x) for x in agedC)
        + "."
    )
    q2win = [s for s in SEEDS if q2[s] == "Q2-WIN"]
    ko_below_fifo = [
        s
        for s in q2win
        if rates[(3000, "0.9.tier1", s)]["all"]["kind"]["point"]
        < rates[(3000, "0.9.tier1", s)]["all"]["fifo"]["point"]
    ]
    w(
        f"- **Q2-WIN unlikely on any seed:** realised {' / '.join(q2)} — "
        + (
            "held."
            if not q2win
            else f"**not held** (Q2-WIN on seeds {q2win}). On seeds {ko_below_fifo} the "
            "comparator itself scores below FIFO (kind-oracle all-query "
            + ", ".join(
                f"{f4(rates[(3000, '0.9.tier1', s)]['all']['kind']['point'])} vs FIFO "
                f"{f4(rates[(3000, '0.9.tier1', s)]['all']['fifo']['point'])}"
                for s in ko_below_fifo
            )
            + "): B0 had already shown the seed-1 class-mean order assert < query, so "
            "that Q2-WIN is a win over a comparator that evicts pending asserts. "
            "(Interpretation: it is weak evidence that ψ̂ uses `c_t` beyond kind.)"
        )
    )
    w(
        "- **E0h within-step R² for ψ̂-U(γ=0) above 0.49 likely:** realised "
        + " / ".join(f4(x) for x in e3r)
        + f" — above 0.49 on {sum(x >= 0.49 for x in e3r)} of 3; mostly not held. "
        'Reaching 0.90 ("genuinely open"): no seed.'
    )
    w("")

    # ---------------- controls, disclosure, provenance
    w("## 9. Controls (§11), T0, and disclosure")
    w("")
    t0s, t0e = rows["T0.start"], rows["T0.end"]
    w(
        f"- **T0** (`{t0s['manifest']}`): start ok={t0s['ok']} ({t0s['n']} files, "
        f"{t0s['n_bad']} bad); end ok={t0e['ok']} ({t0e['n']} files, {t0e['n_bad']} bad). "
        f"Run started {led['started_utc']}, finished {led['finished_utc']}."
    )
    for c in CKPTS:
        for s in SEEDS:
            ctl = I["pay"][c][s]["controls"]
            parts = [
                f"{k}: residency==replay {v_['residency_ok']}, Σr worst {v_['sum_worst']:.2e}"
                + (
                    f", logit ctrl worst {v_['logit_control_worst']:.2e}"
                    if v_.get("logit_control_worst") is not None
                    else ""
                )
                for k, v_ in ctl.items()
            ]
            w(f"- ckpt{c} seed {s} EVAL: " + "; ".join(parts) + ".")
    w(
        f"- Σr tolerance `SUM_TOL` = {B2.SUM_TOL}; every tier above is within it (a failure "
        "is exit 3 at the document, `eval_tier`). Controls 1, 2, 3, 4, 8, 9 and 10 ran in "
        "phase A (the fit ledger's `controls`, `head.*` residuals; see `RESULTS-phaseA.md`); "
        "phase B re-checked checkpoint sha256 on load (`load_checked`), ranges and closure."
    )
    w(
        f"- This report's own checks: `summarise_eval` re-run on the payload counts "
        f"reproduces every stored contrast (max |diff| = {chk['worst_abs_diff']}) and label "
        f"({chk['labels_ok']}); `classify_all` reproduces the ledger's classification "
        f"({chk['class_ok']}); FIFO's replayed residency is 1/0 on gap ≤ M / gap > M."
    )
    w("")
    w(
        "**Disclosure.** Before the official run, the phase-B fix session executed EVAL "
        "ids 940000–940031 and 940000–940001 in scratch, and computed within-step R² on 4 "
        "seed-0 EVAL documents. It reports reading no outcome. The official run recomputed "
        "everything: its units were written fresh (`n_resumed` = 0 for every tier and "
        "child, below), under the code sha256 in each unit's key."
    )
    w("")
    w("## 10. Runtime, hardware, command, SHA")
    w("")
    w(
        f"- **Code:** `{prov['git_sha']}` (dirty={prov['dirty']}), python {prov['python']}, "
        f"{prov['platform']}. Device `{led['device']}`. Config hash `{led['config_hash']}`."
    )
    w(
        "- **Hardware:** Mac Studio, Apple M4 Max, 64 GB (ADR-0007); CPU only; "
        f"{man['parallel']} children × {man['threads_per_child']} thread, 6 cpu-det slots."
    )
    w(
        f"- **N_E** = {man['N_E_run']} (PREREG TBD-3 {man['N_E_prereg']}); N_tier2 = {n2}; "
        f"seeds {led['seeds_actually_run']}."
    )
    w(
        "- **Command:** the parent `run.py eval` (recorded in `ledger.json` `commands` with "
        "its six children, below), launched by the manager at 15:28 PDT through "
        "`orchestrator.slot` on 6 cpu-det slots; the wrapper log `b2-eval.log` is empty and "
        "`b2-eval.rc` is 0. Child logs: `runs/b2-psi-probe/logs/`."
    )
    w("")
    w("| command | exit |")
    w("|---|---|")
    for cmd in led["commands"]:
        argv = cmd["argv"] if isinstance(cmd["argv"], str) else " ".join(cmd["argv"])
        short = argv.replace(str(ROOT) + "/", "")
        w(f"| `{short}` | {cmd.get('exit_code')} |")
    w("")
    w("| child | tier | seconds | documents run | resumed | child wall s | peak RSS GB |")
    w("|---|---|---|---|---|---|---|")
    for c in CKPTS:
        for s in SEEDS:
            cost = rows[f"ckpt{c}.seed{s}.cost"]
            for k, v_ in cost["seconds"].items():
                if k == "child_wall":
                    continue
                w(
                    f"| ckpt{c}-seed{s} | {k} | {v_['run']:.0f} | {v_['n_run']} | "
                    f"{cost['n_resumed'][k]} | {cost['seconds']['child_wall']:.0f} | "
                    f"{cost['peak_rss_gb']:.2f} |"
                )
    w("")
    w("**Input sha256** (this report was generated from these bytes):")
    w("")
    for k, h in I["hashes"].items():
        w(f"- `{k}`: `{h}`")
    w("")
    w(
        "Regenerate: `PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run --lane "
        "cpu-det --slots 1 -- .venv/bin/python experiments/b2-psi-probe/results_report.py`."
    )
    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT} ({len(L)} lines); checks {chk}")
    return Exit.OK


if __name__ == "__main__":
    run_main(main)
