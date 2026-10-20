"""E0D-A2 steps 2-3: candidate primary statistics scored by SIMULATION ONLY.

Delta side: the real LOO Delta (resample, donor seed 0) measured in step 1 on set E
docs [64,128), seed 0, arm B ckpt3000, plus two donor-seed replicates (step 1b).
r_i side: SYNTHETIC scores only. No arm-B r_i is computed or read anywhere here.

Scores:
  PC      binary positive control, 1 on the A-cell (C10's true_demand), else 0
  ORACLE  an independent LOO replicate (donor seed 1): the best any r_i could do given
          LOO's own replicate noise
  AGE     the best age-only score: P(A-cell at rank k | Q-step), from generator structure
          (A-cell rank counts on these docs; no Delta used)
  RECENCY r = rank / M (newest highest), no content
  NULL    a plausible synthetic r shuffled within step
  SYN(a,c) r = softmax(a*1[A] + z), z = c*nscore(within-step rank of Delta_rep2) + sqrt(1-c^2)*eps
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from statistics import NormalDist
_ppf = np.vectorize(NormalDist().inv_cdf)

HERE = Path(__file__).parent
WT = Path("/Users/keanooo7/retrieval-successor-retention/.worktrees/e0d")
spec = importlib.util.spec_from_file_location("e0d_run", WT / "experiments/e0d/run.py")
m = importlib.util.module_from_spec(spec); sys.modules["e0d_run"] = m; spec.loader.exec_module(m)

c = dict(np.load(HERE / "step1_cells.npz"))
reps = dict(np.load(HERE / "step1b_reps.npz"))
M, S = 16, 48
rep1 = reps["dres_donor1"][c["doc"], c["t"], c["rank"]]
rep2 = reps["dres_donor2"][c["doc"], c["t"], c["rank"]]
d = c["dres"]
full = c["tfull"]
Q = full & c["q"]
A = Q & c["a"]
key = c["doc"] * (S + 1) + c["t"]
N_DOCS, N_TARGET = 64, 1024
rng = np.random.default_rng(20260927)

# ---- step structure (Q-steps) -------------------------------------------------
def grid(sel, col):
    k = key[sel]
    u, inv = np.unique(k, return_inverse=True)
    g = np.full((len(u), M), np.nan)
    g[inv, c["rank"][sel]] = col[sel]
    sd = np.zeros(len(u), dtype=int); sd[inv] = c["doc"][sel]
    return g, sd, u, inv

gA_steps = np.isin(key, np.unique(key[A]))  # cells of Q-steps whose assert is resident
QA = Q & gA_steps

# ---- statistics ----------------------------------------------------------------
def wspear(x, y, sel, W):
    f = m.WeightedSpearman(x[sel], y[sel]); dd = c["doc"][sel]
    return f(np.ones(sel.sum())), np.array([f(w[dd]) for w in W])

def step_top(r, dl, sel, W):
    """K2 top-1: argmax r == argmax Delta, per A-step (ties in r broken against us:
    a hit needs r's max to be unique). K3 AUC: P(r_top > r_j) over the other slots,
    ties 0.5, where top = argmax Delta."""
    rg, sd, _, _ = grid(sel, r); dg, _, _, _ = grid(sel, dl)
    ok = ~(np.isnan(rg) | np.isnan(dg))
    hit = np.full(len(rg), np.nan); auc = np.full(len(rg), np.nan)
    for j in range(len(rg)):
        o = ok[j]
        if o.sum() < 2:
            continue
        rr, dd = rg[j, o], dg[j, o]
        top = np.argmax(dd)
        mx = rr.max()
        hit[j] = float(rr[top] == mx and (rr == mx).sum() == 1)
        oth = np.delete(rr, top)
        auc[j] = float(((rr[top] > oth) + 0.5 * (rr[top] == oth)).mean())
    def agg(v):
        okv = ~np.isnan(v)
        pt = v[okv].mean()
        bs = []
        for w in W:
            ww = w[sd][okv]; bs.append((v[okv] * ww).sum() / ww.sum())
        return pt, np.array(bs)
    return agg(hit), agg(auc)

def step_rho(r, dl, sel, W):
    rg, sd, _, _ = grid(sel, r); dg, _, _, _ = grid(sel, dl)
    st = m.step_stats(rg, dg)["per_step_rho"]
    okv = ~np.isnan(st)
    pt = st[okv].mean()
    bs = [(st[okv] * w[sd][okv]).sum() / w[sd][okv].sum() for w in W]
    return pt, np.array(bs)

W = m.bootstrap_doc_weights(N_DOCS, 300, m.BOOT_SEED)
SCALE = np.sqrt(N_DOCS / N_TARGET)  # sd at n=1024 from the 64-doc bootstrap (believed: iid docs)

def all_stats(r, keep=None):
    keep = np.ones(len(d), bool) if keep is None else keep
    have = keep & ~np.isnan(d) & ~np.isnan(r)
    out = {}
    out["K0_rho_Q"] = wspear(r, d, Q & have, W)
    out["K4_rho_QA_pooled"] = wspear(r, d, QA & have, W)
    out["K6_rho_step_A"] = step_rho(r, d, QA & have, W)
    (h, a) = step_top(r, d, QA & have, W)
    out["K2_top1_A"] = h
    out["K3_auc_A"] = a
    return out

pc = A.astype(float)
# AGE: P(A at rank k | A-step), counted from structure (A mask), not from Delta
age_p = np.bincount(c["rank"][A], minlength=M) / A.sum()
age = age_p[c["rank"]] + 1e-9 * c["rank"]  # tiny tiebreak toward newer
recency = c["rank"] / M

def nscore_within_step(v, sel):
    g, _, u, inv = grid(sel, v)
    out = np.full(len(v), np.nan)
    rk = np.argsort(np.argsort(np.where(np.isnan(g), -np.inf, g), axis=1), axis=1)
    ns = _ppf((rk + 0.5) / M)
    idx = np.flatnonzero(sel)
    out[idx] = ns[inv, c["rank"][sel]]
    return out

def softmax_step(logit, sel):
    g, _, u, inv = grid(sel, logit)
    e = np.exp(g - np.nanmax(g, axis=1, keepdims=True)); e = np.where(np.isnan(e), 0, e)
    p = e / e.sum(1, keepdims=True)
    out = np.full(len(logit), np.nan); idx = np.flatnonzero(sel)
    out[idx] = p[inv, c["rank"][sel]]
    return out

zrep = nscore_within_step(np.nan_to_num(rep2, nan=0.0), full)

def syn(a, cc, seed):
    g = np.random.default_rng(seed)
    z = cc * zrep + np.sqrt(1 - cc * cc) * g.standard_normal(len(d))
    return softmax_step(a * A.astype(float) + z, full)

def shuffle_within_step(r, seed):
    g = np.random.default_rng(seed)
    out = r.copy()
    for k in np.unique(key[full]):
        idx = np.flatnonzero(full & (key == k))
        out[idx] = r[g.permutation(idx)]
    return out

scores = {
    "PC(binary)": pc,
    "ORACLE(LOO replicate)": np.nan_to_num(rep1, nan=np.nan),
    "AGE(best age-only)": age,
    "RECENCY": recency,
}
for a, cc in ((0, 0.0), (1, 0.0), (2, 0.0), (4, 0.0), (0, 0.5), (2, 0.5), (4, 0.5), (8, 0.0)):
    scores[f"SYN(a={a},c={cc})"] = syn(a, cc, 7)
scores["NULL(shuffled SYN(4,0.5))"] = shuffle_within_step(scores["SYN(a=4,c=0.5)"], 11)

keep_bos = m.bos_keep({"rank": c["rank"], "q_step": c["q"], "gap": c["gap"]}, M)
res = {"counts": {"Q_cells": int(Q.sum()), "A_cells": int(A.sum()), "QA_cells": int(QA.sum()),
                  "Q_steps": int(len(np.unique(key[Q]))), "A_steps": int(len(np.unique(key[A]))),
                  "p_A_in_Q": float(A.sum() / Q.sum()),
                  "sqrt3p(1-p)": float(np.sqrt(3 * (A.sum() / Q.sum()) * (1 - A.sum() / Q.sum())))},
       "age_p": age_p.tolist(), "stats": {}}
for excl in ("primary", "bos_excluded"):
    kp = None if excl == "primary" else keep_bos
    for nm, r in scores.items():
        st = all_stats(r, kp)
        res["stats"][f"{excl}|{nm}"] = {
            k: {"pt": float(pt), "sd64": float(np.std(bs)), "sd1024": float(np.std(bs) * SCALE)}
            for k, (pt, bs) in st.items()}
        print(excl, nm, {k: round(v[0], 4) for k, v in st.items()}, flush=True)

# normalised statistics
for excl in ("primary", "bos_excluded"):
    s = res["stats"]
    for nm in scores:
        for den in ("PC(binary)", "ORACLE(LOO replicate)"):
            v = s[f"{excl}|{nm}"]["K0_rho_Q"]["pt"] / s[f"{excl}|{den}"]["K0_rho_Q"]["pt"]
            s[f"{excl}|{nm}"][f"K1_rho_Q/{den}"] = {"pt": v}

# LOO reliability (no r_i): Spearman between replicates on Q cells, off-A cells
rel = {}
for nm, sel in (("Q", Q), ("Q_offA", Q & ~A), ("N", full & ~c["q"]), ("A", A)):
    ok = sel & ~np.isnan(d) & ~np.isnan(rep1) & ~np.isnan(rep2)
    rel[nm] = {"rho(d0,d1)": m.spearman(d[ok], rep1[ok]), "rho(d0,d2)": m.spearman(d[ok], rep2[ok]),
               "rho(d0,zero)": m.spearman(d[ok], c["dzero"][ok]), "n": int(ok.sum())}
res["loo_reliability"] = rel
print(json.dumps(rel, indent=1))
(HERE / "step3_out.json").write_text(json.dumps(res, indent=1))
