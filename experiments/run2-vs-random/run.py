"""Step 1 after RSR run 2's FAILS: psiC against uniform-random eviction, and what each
eviction age costs.

DESCRIPTIVE, analysis only. No training, no model forward, no EVAL re-run. It reads
run 2's held-out per-document records (`heldout/seed{s}/{psiC,fifo,random0..4}.jsonl.gz`,
written by run 1's unchanged `measure_seed` at 728a42f) and, for reference, run 1's
records. Nothing here is pre-registered and nothing here gates.

SEEN before this script was written (2026-10-08): psiC - random point estimates per gap
bucket, derived by arithmetic from run 1's RESULTS tables (random, FIFO) and run 2's
psiC - FIFO contrasts (C12: run 2's fifo is run 1's fifo). This script turns them into
paired statistics and adds the eviction-age attribution.

Run 1's harness is reused by import, never copied (as experiments/rsr-run2/run.py does):
the bootstrap (`boot_weights`, run 1's paired per-document resampling, 2000 replicates,
seed 20260927 + seed), the buckets (`buckets_for(16)`), and the "random" column's
definition (the mean over random0..4 of their pooled accuracies, B2 §9.2).

Exit 0 when every pairing and consistency check passes; exit 3 (inconclusive; numbers
still written) otherwise.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rsr.exit_codes import ArgumentParser, Exit, run_main  # noqa: E402

EXPERIMENT = "experiments/run2-vs-random/run.py"
MAIN = ROOT.parents[1] if ROOT.parent.name == ".worktrees" else ROOT
RUN2_DEFAULT = MAIN / ".worktrees/rsr-run2-run/runs/rsr-run2"
RUN1_DEFAULT = MAIN / ".worktrees/rsr-run1-run/runs/rsr-run1"
SEEDS = (0, 1, 2)
M = 16  # run 1 / run 2 §6
A_MAX = 48  # run 2 PREREG Amendment 1
AGE_BUCKETS = (("1", 1, 1), ("2..4", 2, 4), ("5..M", 5, M), ("M+1..A_max", M + 1, A_MAX))
REPORT_BUCKETS = ("all", "gap_eq_1", "gap_2_to_M", "gap_gt_M")


def _load(name: str, rel: str):
    """Import an experiment script by path, as its own module instance. Never copied."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R1 = _load("_diag_r1", "experiments/rsr-run1/run.py")
RANDOM = tuple(f"random{k}" for k in range(R1.N_RANDOM))


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def hit_counts(records: list[dict], sel) -> torch.Tensor:
    """``[D, 2]`` = (queries in the bucket, queried assert resident) per document."""
    rows = []
    for r in records:
        if not r["hits"]:
            rows.append([0.0, 0.0])
            continue
        g = torch.tensor([h[0] for h in r["hits"]], dtype=torch.long)
        hit = torch.tensor([h[1] for h in r["hits"]], dtype=torch.bool)
        m = sel(g)
        rows.append([float(m.sum()), float(hit[m].sum())])
    return torch.tensor(rows, dtype=torch.float64).reshape(-1, 2)


def conditional_counts(records: list[dict], sel) -> dict:
    """Answer accuracy split by residency: (resident, resident and right, missed,
    missed and right) over the queries in the bucket. Point counts only."""
    c = Counter()
    for r in records:
        for (_t, gap, ok, _nll), (_g, hit) in zip(r["answers"], r["hits"], strict=True):
            if not bool(sel(torch.tensor(gap))):
                continue
            k = "resident" if hit else "missed"
            c[k] += 1
            c[f"{k}_right"] += bool(ok)
    return dict(c)


def _rate(c: dict, k: str) -> float | None:
    return c.get(f"{k}_right", 0) / c[k] if c.get(k) else None


def pairing_failures(recs: dict[str, list[dict]]) -> list[str]:
    """Every pass must hold the same documents, in order, with the same (t, gap) queries,
    and `hits` must align with `answers` query for query."""
    out = []
    ref = recs["fifo"]
    ref_docs = [r["doc"] for r in ref]
    ref_q = [[(a[0], a[1]) for a in r["answers"]] for r in ref]
    for n, rs in recs.items():
        if [r["doc"] for r in rs] != ref_docs:
            out.append(f"{n}: document order differs from fifo")
            continue
        if [[(a[0], a[1]) for a in r["answers"]] for r in rs] != ref_q:
            out.append(f"{n}: (t, gap) query sequence differs from fifo")
        bad = sum([a[1] for a in r["answers"]] != [h[0] for h in r["hits"]] for r in rs)
        if bad:
            out.append(f"{n}: hits not aligned with answers in {bad} documents")
    return out


def minus_random(per: dict[str, torch.Tensor], arm: str, W: torch.Tensor) -> dict:
    """``arm`` - random, where random is the mean over random0..4 of their pooled
    ratios (run 1's "random" column), paired per replicate."""
    rr = torch.stack([R1.pooled_replicates(per[k], W) for k in RANDOM]).mean(0)
    c = R1._ci(R1.pooled_replicates(per[arm], W) - rr)
    pa = R1._point(per[arm])
    pr = [R1._point(per[k]) for k in RANDOM]
    c["point"] = pa - sum(pr) / len(pr)
    c["arm"] = pa
    c["random"] = sum(pr) / len(pr)
    c["vs_each_random_min"] = min(pa - p for p in pr)
    c["vs_each_random_max"] = max(pa - p for p in pr)
    c["reading"] = R1.ci_reading(c["lo"], c["hi"])
    return c


def age_bucket(age: int) -> str | None:
    for name, lo, hi in AGE_BUCKETS:
        if lo <= age <= hi:
            return name
    return None


def eviction_cost(records: list[dict]) -> tuple[dict, list[str]]:
    """Attribute every residency miss to the eviction that removed the queried assert.

    A query at step t with gap g asks about the sentence written at step t - g; the
    forward reads the pre-write memory (`rsr.metrics.headroom.simulate`). `victims[e]`
    is the sentence evicted by the write at step e, so its age at eviction is e - victim
    (checked against the `evictions` rows' recorded ages)."""
    n_ev: Counter = Counter()
    miss: Counter = Counter()
    miss_2m: Counter = Counter()
    miss_gtm: Counter = Counter()
    miss_wrong: Counter = Counter()
    anomalies: list[str] = []
    totals = Counter()
    for r in records:
        evicted_at = {}
        for e, v in r["victims"].items():
            evicted_at[int(v)] = int(e)
        rec_ages = sorted((int(t), int(a)) for t, a, _ in r["evictions"])
        mine = sorted((e, e - v) for v, e in evicted_at.items())
        if rec_ages != mine:
            anomalies.append(f"doc {r['doc']}: victims ages != evictions ages")
        for _e, age in mine:
            n_ev[age_bucket(age)] += 1
        for (t, gap, ok, _nll), (_g, hit) in zip(r["answers"], r["hits"], strict=True):
            totals["queries"] += 1
            if hit:
                totals["hit"] += 1
                totals["hit_wrong"] += not ok
                continue
            totals["miss"] += 1
            totals["miss_right"] += bool(ok)
            a = t - gap
            if a not in evicted_at or not (a < evicted_at[a] <= t):
                anomalies.append(f"doc {r['doc']}: miss at t={t} gap={gap}, no eviction")
                continue
            b = age_bucket(evicted_at[a] - a)
            miss[b] += 1
            miss_wrong[b] += not ok
            if 2 <= gap <= M:
                miss_2m[b] += 1
            elif gap > M:
                miss_gtm[b] += 1
    rows = {}
    for name, _lo, _hi in AGE_BUCKETS:
        rows[name] = {
            "evictions": n_ev[name],
            "misses_caused": miss[name],
            "misses_caused_gap_2_to_M": miss_2m[name],
            "misses_caused_gap_gt_M": miss_gtm[name],
            "missed_and_answered_wrong": miss_wrong[name],
            "misses_per_eviction": miss[name] / n_ev[name] if n_ev[name] else None,
        }
    if None in n_ev:
        anomalies.append(f"{n_ev[None]} evictions outside every age bucket")
    return {"by_age": rows, "totals": dict(totals)}, anomalies


def mean_cost(costs: list[dict]) -> dict:
    """Average the five random passes' counts (each is one uniform-random replay)."""
    out = {"by_age": {}, "totals": {}}
    for name, _lo, _hi in AGE_BUCKETS:
        row = {}
        for k in (
            "evictions",
            "misses_caused",
            "misses_caused_gap_2_to_M",
            "misses_caused_gap_gt_M",
            "missed_and_answered_wrong",
        ):
            row[k] = sum(c["by_age"][name][k] for c in costs) / len(costs)
        row["misses_per_eviction"] = (
            row["misses_caused"] / row["evictions"] if row["evictions"] else None
        )
        out["by_age"][name] = row
    keys = set().union(*(c["totals"] for c in costs))
    out["totals"] = {
        k: sum(c["totals"].get(k, 0) for c in costs) / len(costs) for k in keys
    }
    return out


def analyse_seed(run2: Path, run1: Path, seed: int, n_boot: int) -> dict:
    d2 = run2 / "heldout" / f"seed{seed}"
    names = ("psiC", "fifo", *RANDOM)
    recs = {n: R1.read_pass(d2, n) for n in names}
    fails = pairing_failures(recs)
    r1 = {
        n: R1.read_pass(run1 / "heldout" / f"seed{seed}", n)
        for n in ("psiC", "fifo", *RANDOM)
    }
    fails += [f"run1 {x}" for x in pairing_failures(r1)]
    # run 2's fifo and random passes do not depend on phi: they must equal run 1's
    for n in ("fifo", *RANDOM):
        if recs[n] != r1[n]:
            fails.append(f"{n}: run 2 records differ from run 1's (expected identical)")
    W = R1.boot_weights(len(recs["fifo"]), seed, n_boot)
    buckets = R1.buckets_for(M)
    out = {"inputs": {f"run2/{n}": sha256(d2 / f"{n}.jsonl.gz") for n in names}}
    out["inputs"]["run1/psiC"] = sha256(
        run1 / "heldout" / f"seed{seed}" / "psiC.jsonl.gz"
    )
    out["n_documents"] = len(recs["fifo"])
    out["accuracy"], out["residency"] = {}, {}
    for b in REPORT_BUCKETS:
        per = {n: R1.answer_counts(recs[n], buckets[b]) for n in names}
        per1 = {n: R1.answer_counts(r1[n], buckets[b]) for n in ("psiC", *RANDOM)}
        out["accuracy"][b] = {
            "n_answers": int(per["fifo"][:, 0].sum()),
            "run2_psiC_minus_random": minus_random(per, "psiC", W),
            "fifo_minus_random": minus_random(per, "fifo", W),
            "run1_psiC_minus_random_SEEN": minus_random(per1, "psiC", W),
        }
        hp = {n: hit_counts(recs[n], buckets[b]) for n in names}
        out["residency"][b] = {
            "run2_psiC_minus_random": minus_random(hp, "psiC", W),
            "fifo_minus_random": minus_random(hp, "fifo", W),
        }
    out["conditional"] = {}
    for b in REPORT_BUCKETS:
        cc = {n: conditional_counts(recs[n], buckets[b]) for n in names}
        rnd = {
            k: sum(cc[n].get(k, 0) for n in RANDOM) / len(RANDOM)
            for k in ("resident", "resident_right", "missed", "missed_right")
        }
        out["conditional"][b] = {
            "psiC": cc["psiC"],
            "fifo": cc["fifo"],
            "random_mean": rnd,
        }
    cost, anomalies = {}, []
    for n in ("psiC", "fifo"):
        cost[n], an = eviction_cost(recs[n])
        anomalies += [f"{n}: {x}" for x in an]
    rc = []
    for n in RANDOM:
        c, an = eviction_cost(recs[n])
        rc.append(c)
        anomalies += [f"{n}: {x}" for x in an]
    cost["random_mean"] = mean_cost(rc)
    out["eviction_cost"] = cost
    out["pairing_failures"] = fails
    out["anomalies"] = anomalies[:50]
    out["n_anomalies"] = len(anomalies)
    return out


def fmt(c: dict) -> str:
    return f"{c['point']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"


def row(*cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def header(*cols) -> list[str]:
    return [row(*cols), row(*["---"] * len(cols))]


def render(res: dict, meta: dict) -> str:
    L = [
        "# RESULTS: run 2 psiC against uniform-random eviction (step 1 after FAILS)",
        "",
        "DESCRIPTIVE and SEEN (see run.py's docstring). Written by a model; not by "
        "Brendan. Nothing here is pre-registered and nothing gates. Generated by "
        "run.py; no number is typed.",
        "",
        f"- command: `{meta['command']}`",
        f"- git: `{meta['git_sha']}`, porcelain `{meta['porcelain']!r}`",
        f"- hardware: {meta['hardware']}; python {meta['python']}, torch {meta['torch']}",
        f"- inputs: run 2 `{meta['run2']}`, run 1 `{meta['run1']}`",
        f"- bootstrap: run 1's `boot_weights(2048, seed, {meta['n_boot']})`, paired "
        "per document; random = mean over random0..4 of their pooled ratios "
        "(run 1's column)",
        f"- exit `{meta['exit']}`",
        "",
        "## 1. Answer accuracy, run 2 psiC - random (95% CI)",
        "",
        *header(
            "seed",
            "bucket",
            "n answers",
            "psiC",
            "random",
            "psiC - random",
            "vs each random draw",
            "run 1 psiC - random (SEEN)",
            "fifo - random",
        ),
    ]
    for s, r in res.items():
        for b in REPORT_BUCKETS:
            a = r["accuracy"][b]
            p = a["run2_psiC_minus_random"]
            each = f"{p['vs_each_random_min']:+.4f} .. {p['vs_each_random_max']:+.4f}"
            L.append(
                row(
                    s,
                    b,
                    a["n_answers"],
                    f"{p['arm']:.4f}",
                    f"{p['random']:.4f}",
                    f"**{fmt(p)}**",
                    each,
                    fmt(a["run1_psiC_minus_random_SEEN"]),
                    fmt(a["fifo_minus_random"]),
                )
            )
    L += [
        "",
        "## 2. Residency (queried assert in memory at the query), psiC - random",
        "",
        *header("seed", "bucket", "psiC", "random", "psiC - random", "fifo - random"),
    ]
    for s, r in res.items():
        for b in REPORT_BUCKETS:
            h = r["residency"][b]
            p = h["run2_psiC_minus_random"]
            L.append(
                row(
                    s,
                    b,
                    f"{p['arm']:.4f}",
                    f"{p['random']:.4f}",
                    f"**{fmt(p)}**",
                    fmt(h["fifo_minus_random"]),
                )
            )
    L += [
        "",
        "## 2b. Accuracy given residency (point counts, no CI)",
        "",
        "`right given resident` = share of queries answered right when the queried "
        "assert was in memory; `right given missed` likewise when it was not.",
        "",
        *header(
            "seed",
            "bucket",
            "arm",
            "resident",
            "right given resident",
            "right given missed",
        ),
    ]
    for s, r in res.items():
        for b in ("all", "gap_2_to_M", "gap_gt_M"):
            for arm in ("psiC", "fifo", "random_mean"):
                c = r["conditional"][b][arm]
                rr, rm = _rate(c, "resident"), _rate(c, "missed")
                L.append(
                    row(
                        s,
                        b,
                        arm,
                        f"{c.get('resident', 0):.0f}",
                        "-" if rr is None else f"{rr:.4f}",
                        "-" if rm is None else f"{rm:.4f}",
                    )
                )
    L += [
        "",
        "## 3. What each eviction age costs",
        "",
        "Each residency miss is attributed to the eviction that removed the queried "
        "assert. Random rows are the mean over the five random passes. "
        "`per eviction` = misses caused / evictions.",
        "",
        *header(
            "seed",
            "arm",
            "victim age",
            "evictions",
            "misses caused",
            "of which gap 2..M",
            "of which gap > M",
            "per eviction",
        ),
    ]
    for s, r in res.items():
        for arm in ("psiC", "fifo", "random_mean"):
            for name, _lo, _hi in AGE_BUCKETS:
                c = r["eviction_cost"][arm]["by_age"][name]
                pe = c["misses_per_eviction"]
                L.append(
                    row(
                        s,
                        arm,
                        name,
                        f"{c['evictions']:.0f}",
                        f"{c['misses_caused']:.0f}",
                        f"{c['misses_caused_gap_2_to_M']:.0f}",
                        f"{c['misses_caused_gap_gt_M']:.0f}",
                        "-" if pe is None else f"{pe:.3f}",
                    )
                )
    L += [
        "",
        "### Share of psiC's gap 2..M residency misses caused by age-1 evictions",
        "",
    ]
    for s, r in res.items():
        by = r["eviction_cost"]["psiC"]["by_age"]
        tot = sum(by[n]["misses_caused_gap_2_to_M"] for n, _a, _b in AGE_BUCKETS)
        a1 = by["1"]["misses_caused_gap_2_to_M"]
        L.append(
            f"- seed {s}: {a1} / {tot} = {a1 / tot:.3f}"
            if tot
            else f"- seed {s}: no misses"
        )
    L += ["", "## 4. Checks", ""]
    for s, r in res.items():
        L.append(
            f"- seed {s}: pairing failures {r['pairing_failures'] or 'none'}; "
            f"anomalies {r['n_anomalies']}; documents {r['n_documents']}"
        )
        for arm in ("psiC", "fifo"):
            t = r["eviction_cost"][arm]["totals"]
            L.append(
                f"  - {arm}: queries {t.get('queries')}, resident {t.get('hit')}, "
                f"missed {t.get('miss')}, "
                f"missed but answered right {t.get('miss_right')}, "
                f"resident but answered wrong {t.get('hit_wrong')}"
            )
    L += [
        "",
        "## 5. What this does not show",
        "",
        "- Why psiC ranks as it does. The attribution counts which eviction removed "
        "a queried assert, not what the head saw.",
        "- Section 2b compares different subsets: the facts psiC keeps resident are "
        "not the facts random keeps, so `right given resident` mixes the reader's "
        "use of memory with which facts were kept.",
        "- No scaling claim, no cognitive claim. §15 untouched. Run 2's class (FAILS) "
        "is unchanged.",
        "",
    ]
    return "\n".join(L)


def _git(*args: str) -> str:
    p = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else f"<git rc {p.returncode}>"


def _hardware() -> str:
    def sysctl(k: str) -> str:
        p = subprocess.run(["sysctl", "-n", k], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else "?"

    mem = sysctl("hw.memsize")
    mem = f"{int(mem) / 2**30:.0f} GB" if mem.isdigit() else mem
    return f"{sysctl('machdep.cpu.brand_string')}, {mem}, {platform.platform()}"


def main(argv: list[str] | None = None) -> Exit:
    ap = ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run2", type=Path, default=RUN2_DEFAULT)
    ap.add_argument("--run1", type=Path, default=RUN1_DEFAULT)
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/run2-vs-random")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    res = {s: analyse_seed(args.run2, args.run1, s, args.n_boot) for s in SEEDS}
    bad = any(r["pairing_failures"] or r["n_anomalies"] for r in res.values())
    meta = {
        "command": " ".join([".venv/bin/python", EXPERIMENT, *(argv or sys.argv[1:])]),
        "git_sha": _git("rev-parse", "HEAD"),
        "porcelain": _git("status", "--porcelain"),
        "hardware": _hardware(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "run2": str(args.run2),
        "run1": str(args.run1),
        "n_boot": args.n_boot,
        "exit": int(Exit.DID_NOT_RUN if bad else Exit.OK),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps({"meta": meta, "seeds": res}, indent=1)
    )
    (args.out / "RESULTS.md").write_text(render(res, meta))
    print(f"exit {meta['exit']}; wrote {args.out / 'RESULTS.md'}")
    return Exit.DID_NOT_RUN if bad else Exit.OK


if __name__ == "__main__":
    run_main(main)
