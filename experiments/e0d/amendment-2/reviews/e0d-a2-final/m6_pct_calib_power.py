"""E0D-AMD2-FINAL: AUROC_strat,pct calibration and power, per seed.

Delta side: real LOO resample Delta, set E docs [64, 128), per seed (seed 0: reviews/e0d-a2/
step1_cells.npz; seeds 1, 2: tau_cells_seed{1,2}.npz from m5). Labels use that seed's own frozen
TAU (m5_out.json). Score side: SYNTHETIC ONLY. No arm-B r_i is computed or read. The only
structure-derived score is true_demand (the A-cell indicator, C10's control).

This file is also the reference implementation of the A2.4 statistic, for the Part B tests.
"""
import hashlib
import json
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np

HERE = Path(__file__).resolve().parent
R0 = HERE.parent / "e0d-a2"
M = 16
A_STAR = 0.85
Phi = NormalDist().cdf
NB = 300  # multinomial redraws at n = 1024 docs
tau_out = json.loads((HERE / "m5_out.json").read_text())


def load(seed):
    p = R0 / "step1_cells.npz" if seed == 0 else HERE / f"tau_cells_seed{seed}.npz"
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha == tau_out[f"seed{seed}"]["cells_sha256"], (seed, sha)
    return dict(np.load(p)), sha


def grids(c, tau, excluded=True):
    """Step x rank grids of the A1.3-excluded population (operative) at full-memory Q-steps."""
    full = c["tfull"].astype(bool); q = c["q"].astype(bool)
    keep = full & q
    if excluded:
        keep &= (c["rank"] != M - 1) & ~(c["gap"] == 1)  # A1.3: rank M-1 and gap-1 Q-steps out
    key = c["doc"] * 49 + c["t"]
    u, inv = np.unique(key[keep], return_inverse=True)
    rk = c["rank"][keep]
    def G(col, fill=np.nan):
        g = np.full((len(u), M), fill, float); g[inv, rk] = col[keep]; return g
    elig = G(np.ones(len(keep)), 0.0) > 0
    d = G(c["dres"]); age = G((c["t"] - c["sent"]).astype(float))
    A = G((q & c["a"].astype(bool)).astype(float), 0.0) > 0
    sdoc = np.zeros(len(u), int); sdoc[inv] = c["doc"][keep]
    lab = elig & ~np.isnan(d)                  # a cell has a label iff its Delta has a value
    y = lab & (d > tau)                        # strict >, A2.2
    kt = y.sum(1)
    return dict(elig=elig, d=d, age=age, A=A, sdoc=sdoc, lab=lab, y=y, kt=kt)


def pct_grid(r, elig):
    """Within-step percentile: midrank among ALL eligible slots (a slot with a NaN Delta still
    counts in n_t; r itself must be finite), mapped to [0, 1] by (midrank - 1) / (n_t - 1)."""
    out = np.full(r.shape, np.nan)
    for j in range(r.shape[0]):
        ok = elig[j]; v = r[j, ok]
        if not np.all(np.isfinite(v)):
            raise ValueError("non-finite r on an eligible slot: MeasurementUndefined (exit 3)")
        n = ok.sum()
        _, ix, cnt = np.unique(v, return_inverse=True, return_counts=True)
        mid = (np.cumsum(cnt) - cnt) + (cnt + 1) / 2
        out[j, ok] = (mid[ix] - 1) / (n - 1)
    return out


def wauc(s, y, w):
    Wp = w[y].sum(); Wn = w[~y].sum()
    if Wp <= 0 or Wn <= 0:
        return np.nan, 0.0
    uv, ix = np.unique(s, return_inverse=True)
    wn = np.bincount(ix, weights=w * (~y), minlength=len(uv))
    below = np.cumsum(wn) - wn
    return float((w[y] * (below[ix[y]] + 0.5 * wn[ix[y]])).sum() / (Wp * Wn)), float(Wp)


def auroc_strat(score, P, dw):
    """Age-stratified AUROC, weight w_a = positives in bin a; a bin lacking a class is excluded."""
    cm = P["lab"]; sw = dw[P["sdoc"]]
    W = np.broadcast_to(sw[:, None], cm.shape)[cm]
    s = score[cm]; y = P["y"][cm]; a = P["age"][cm]
    num = den = 0.0; excl = []
    for b in range(1, M + 1):
        mb = a == b
        if not mb.any():
            continue
        au, wp = wauc(s[mb], y[mb], W[mb])
        if np.isnan(au):
            excl.append(b); continue
        num += wp * au; den += wp
    return (num / den if den > 0 else np.nan), excl


def softmax_steps(x, elig):
    x = np.where(elig, x, -np.inf); x = x - x.max(1, keepdims=True)
    e = np.where(elig, np.exp(x), 0.0); return e / e.sum(1, keepdims=True)


def run_seed(seed, rng):
    c, sha = load(seed)
    tau = tau_out[f"seed{seed}"]["dres"]["q0.995"]
    P = grids(c, tau); e = P["elig"]; y = P["y"].astype(float); age = np.nan_to_num(P["age"])
    has = (P["kt"] >= 1)[:, None]
    eps = rng.standard_normal(e.shape)
    # u(age): standardised log critical rate by age on this seed's labels
    lr = np.array([np.log((P["y"] & (P["age"] == b)).sum() + 1) - np.log(((P["lab"]) & (P["age"] == b)).sum() + 1)
                   for b in range(M + 1)])
    uu = (lr - lr[2:].mean()) / (lr[2:].std() + 1e-12)
    uage = uu[age.astype(int)]
    cases = {
        "perfect, binary y (tied)": np.where(e, y, np.nan),
        "perfect, softmax(10y+eps)": softmax_steps(10 * y + eps, e),
        "true_demand (A-cell, C10 control)": np.where(e, P["A"].astype(float), np.nan),
        "age-only f(age)": np.where(e, uage, np.nan),
        "recency -age": np.where(e, -age, np.nan),
        "within-step shuffle of softmax(1.5y+eps)": None,
        "pure step-concentration (flat on k>=1, peaked else)": softmax_steps(np.where(has, 0.0, 50.0) * eps, e),
    }
    base = softmax_steps(1.5 * y + eps, e)
    sh = base.copy()
    for j in range(sh.shape[0]):
        ok = np.flatnonzero(e[j]); sh[j, ok] = sh[j, rng.permutation(ok)]
    cases["within-step shuffle of softmax(1.5y+eps)"] = sh
    for a in (0.5, 1.0, 1.25, 1.4, 1.5, 2.0, 2.5, 3.0):
        cases[f"content a={a}"] = softmax_steps(a * y + eps, e)
    for a in (1.0, 2.0, 3.0):
        cases[f"content a={a} + age u(age)"] = softmax_steps(a * y + uage + eps, e)
    cases["content a=2.0, step temp flat-crit/peaked-else"] = softmax_steps(np.where(has, 0.05, 6.0) * (2.0 * y + eps), e)
    ct = rng.uniform(0.5, 2.0, size=(e.shape[0], 1))
    cases["content a=2.0 rescaled c_t*r"] = cases["content a=2.0"] * ct
    if seed == 0:
        reps = dict(np.load(R0 / "step1b_reps.npz"))
        full = c["tfull"].astype(bool); q = c["q"].astype(bool)
        keep = full & q & (c["rank"] != M - 1) & ~(c["gap"] == 1)
        key = c["doc"] * 49 + c["t"]; u, inv = np.unique(key[keep], return_inverse=True)
        g = np.full(e.shape, np.nan); g[inv, c["rank"][keep]] = reps["dres_donor1"][c["doc"], c["t"], c["rank"]][keep]
        cases["LOO replicate (donor seed 1), NaN->0"] = np.where(e, np.nan_to_num(g), np.nan)
    ones = np.ones(64)
    Wb = rng.multinomial(1024, np.ones(64) / 64, size=NB).astype(float)
    res = {"seed": seed, "cells_sha256": sha, "tau": tau,
           "n_Q_steps_excl": int(e.shape[0]), "n_t_values": sorted(set(int(x) for x in e.sum(1))),
           "k_t_hist": {str(k): int((P["kt"] == k).sum()) for k in range(int(P["kt"].max()) + 1)},
           "labelled_cells": int(P["lab"].sum()), "positives": int(P["y"].sum()),
           "steps_with_a_NaN_label_cell": int((e & ~P["lab"]).any(1).sum()), "cases": {}}
    for nm, sc in cases.items():
        sc = np.where(e, sc, np.nan)
        pg = pct_grid(sc, e)
        ap, excl = auroc_strat(pg, P, ones)
        ar, _ = auroc_strat(sc, P, ones)
        bs = np.array([auroc_strat(pg, P, w)[0] for w in Wb])
        sd = float(np.nanstd(bs))
        p1 = Phi((ap - A_STAR) / sd - 1.96) if sd > 0 else float(ap > A_STAR)
        res["cases"][nm] = {"AUROC_strat_pct": round(ap, 4), "AUROC_strat_raw": round(ar, 4),
                            "pct_sd_n1024": round(sd, 4),
                            "P(CI_lo>=0.85) one seed": round(p1, 3),
                            "excluded_bins": excl}
    # the ceiling of a continuous perfect proxy restricted to Q_crit steps (k>=1), reported only
    Pk = dict(P); Pk["lab"] = P["lab"] & has
    ap, _ = auroc_strat(pct_grid(np.where(e, cases["perfect, softmax(10y+eps)"], np.nan), e), Pk, ones)
    res["perfect_softmax_on_k>=1_steps_only"] = round(ap, 4)
    return res


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [0, 1, 2]
    out = {"numpy_version": np.__version__, "A_STAR": A_STAR, "NB": NB}
    for s in seeds:
        out[f"seed{s}"] = run_seed(s, np.random.default_rng(1000 + s))
    if len(seeds) > 1:
        names = out[f"seed{seeds[0]}"]["cases"].keys()
        out["all_seeds_power_product"] = {
            nm: round(float(np.prod([out[f"seed{s}"]["cases"][nm]["P(CI_lo>=0.85) one seed"]
                                     for s in seeds if nm in out[f"seed{s}"]["cases"]])), 3)
            for nm in names}
    print(json.dumps(out, indent=1))
    (HERE / f"m6_out_{''.join(map(str, seeds))}.json").write_text(json.dumps(out, indent=1))
