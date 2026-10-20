"""Verifier's independent recompute for expire-span-referendum (A2 re-run).

Own numpy code over the raw per-document records. Does not import either harness.
Bootstrap: (a) an independent numpy bootstrap with the verifier's own seed (4000 reps);
(b) a replay of run 1's documented weights (torch.Generator seeded 20260927+seed,
randint(0, n, (2000, n)) counted per document) -- the draw only; the statistic is numpy.
"""
import gzip, hashlib, json, sys
from fractions import Fraction
from pathlib import Path
import numpy as np

A2 = Path("/Users/keanooo7/retrieval-successor-retention/.worktrees/es-referendum-a2-run/runs/expire-span-referendum")
R1 = Path("/Users/keanooo7/retrieval-successor-retention/.worktrees/rsr-run1-run/runs/rsr-run1")
L = json.loads((A2 / "ledger.json").read_text())
R1L = {r["key"]: r for r in json.loads((R1 / "ledger.json").read_text())["rows"] if isinstance(r.get("key"), str)}
DELTA = {"0": 0.038022, "1": 0.036757, "2": 0.031858}  # PREREG A1.1, typed from the table
out = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    with gzip.open(p, "rt") as f:
        return [json.loads(x) for x in f]


def counts(recs):
    # per document: (answers, correct); answers = [step, gap, correct, nll]
    return np.array([[len(r["answers"]), sum(1 for a in r["answers"] if a[2] is True)] for r in recs], dtype=np.float64)


def q(x, p):
    return float(np.quantile(x, p))  # linear interpolation


def replay_weights(n, seed, B=2000):
    import torch
    g = torch.Generator().manual_seed(20260927 + seed)
    idx = torch.randint(0, n, (B, n), generator=g).numpy()
    W = np.zeros((B, n))
    for b in range(B):
        W[b] = np.bincount(idx[b], minlength=n)
    return W


def own_weights(n, B=4000, seed=777001):
    rng = np.random.default_rng(seed)
    W = np.zeros((B, n))
    for b in range(B):
        W[b] = np.bincount(rng.integers(0, n, n), minlength=n)
    return W


def ratio(W, x):
    return (W @ x[:, 1]) / (W @ x[:, 0])


for s in (0, 1, 2):
    k = str(s)
    pes = A2 / f"eval/seed{s}/es.jsonl.gz"
    pps = R1 / f"heldout/seed{s}/psiC.jsonl.gz"
    pff = R1 / f"heldout/seed{s}/fifo.jsonl.gz"
    shas = {"es": sha(pes), "psiC": sha(pps), "fifo": sha(pff)}
    es, ps, ff = load(pes), load(pps), load(pff)
    docs = [[r["doc"] for r in a] for a in (es, ps, ff)]
    paired = docs[0] == docs[1] == docs[2] and docs[0] == list(range(995000, 997048))
    X = {"es": counts(es), "psiC": counts(ps), "fifo": counts(ff)}
    # exact pooled points
    pt = {a: Fraction(int(X[a][:, 1].sum()), int(X[a][:, 0].sum())) for a in X}
    ptf = {a: float(X[a][:, 1].sum()) / float(X[a][:, 0].sum()) for a in X}
    n = len(es)
    Wr = replay_weights(n, s)
    Wo = own_weights(n)
    res = {"shas": shas, "ledger_shas": L["seeds"][k]["sha256"], "paired_docs_eval_range": paired,
           "n_docs": n, "n_answers": {a: int(X[a][:, 0].sum()) for a in X},
           "same_answer_count_per_doc": bool((X["es"][:, 0] == X["psiC"][:, 0]).all() and (X["es"][:, 0] == X["fifo"][:, 0]).all())}
    for b in ("psiC", "fifo"):
        name = f"es_minus_{b}"
        led = L["seeds"][k]["con"]["acc.all." + name]
        point = ptf["es"] - ptf[b]
        rr = ratio(Wr, X["es"]) - ratio(Wr, X[b])
        ro = ratio(Wo, X["es"]) - ratio(Wo, X[b])
        res[name] = {
            "point": point, "point_exact": str(pt["es"] - pt[b]), "ledger_point": led["point"],
            "point_eq_ledger": point == led["point"],
            "replay": {"lo": q(rr, .025), "hi": q(rr, .975), "sd": float(rr.std(ddof=1))},
            "own": {"lo": q(ro, .025), "hi": q(ro, .975), "sd": float(ro.std(ddof=1))},
            "ledger": {x: led[x] for x in ("lo", "hi", "sd")},
        }
        e = res[name]
        e["replay_eq_ledger"] = e["replay"]["lo"] == led["lo"] and e["replay"]["hi"] == led["hi"]
        e["replay_absdiff"] = [abs(e["replay"]["lo"] - led["lo"]), abs(e["replay"]["hi"] - led["hi"])]
        e["own_absdiff_over_sd"] = [abs(e["own"]["lo"] - led["lo"]) / led["sd"], abs(e["own"]["hi"] - led["hi"]) / led["sd"]]
    # levels vs run 1's own ledger
    for a in ("psiC", "fifo"):
        mine = {"point": ptf[a], "n": int(X[a][:, 0].sum())}
        rep = ratio(Wr, X[a])
        mine.update(lo=q(rep, .025), hi=q(rep, .975))
        r1 = R1L[f"seed{s}.heldout.accuracy.all.{a}"]["value"]
        esl = L["seeds"][k]["acc"]["all"][a]
        res[f"level_{a}"] = {"mine": mine, "run1_ledger": r1, "es_ledger": esl,
                             "es_ledger_eq_run1": esl == r1,
                             "mine_point_eq_run1": mine["point"] == r1["point"] and mine["n"] == r1["n"],
                             "mine_replay_ci_eq_run1": mine["lo"] == r1["lo"] and mine["hi"] == r1["hi"]}
    out[k] = res

# Class, PREREG 7.2 + A1: order SUPERSEDES, LOSES, TIES, MIXED; use own-bootstrap CIs AND replay CIs
def classify(cis):
    if all(lo > 0 for lo, hi in cis.values()):
        return "SUPERSEDES"
    if all(hi < 0 for lo, hi in cis.values()):
        return "LOSES"
    if all(-DELTA[k] <= lo and hi <= DELTA[k] for k, (lo, hi) in cis.items()):
        return "TIES"
    return "MIXED"

out["class_own"] = classify({k: (out[k]["es_minus_psiC"]["own"]["lo"], out[k]["es_minus_psiC"]["own"]["hi"]) for k in "012"})
out["class_replay"] = classify({k: (out[k]["es_minus_psiC"]["replay"]["lo"], out[k]["es_minus_psiC"]["replay"]["hi"]) for k in "012"})
out["class_ledger"] = L["class"]
out["delta_ledger"] = L["delta"]
out["delta_eq_A1.1"] = L["delta"] == DELTA
json.dump(out, sys.stdout, indent=1)
