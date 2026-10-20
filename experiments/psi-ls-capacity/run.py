"""The exact least-squares optimum of psi_hat's function class on U (after diagnostic A).

PREREG `experiments/psi-ls-capacity/PREREG.md` (committed alone at 87dc2ec). U
documents only: no EVAL document is generated or read (C5, through diagnostic A's
`_seal_eval`). No gradient step, no optimizer, no learning rate, no muP choice.

psi_hat (`BilinearValueHead`, spec §3.2.2) is linear in its parameters:

    psi(s, c) = s^T W c / d + u_s.s / (2d) + u_c.c / (2d) = <A, s c^T> + a.s + b.c

and run 2's L_MC (ADR-0009 L9) is a squared error summed over cells divided by a
phi-free count, so its minimum over the class is ordinary least squares on the
16,640 features ``[vec(s c^T); s; c]`` (PREREG §0). The normal equations are assembled
from per-(document, t) sufficient statistics in float64 (PREREG §2), solved by `eigh`
with the minimum-norm cutoff, mapped back to (W, u), and every reading goes through the
head's own forward and offline-psi's functions (`val_loss`, `score_matrices`,
`replay_rates`, `reference_rates`, `B0.boot_rule`, `rho_psi_g`, `fit_label`).

Feature order: index ``a*d + b`` is ``s_a c_b`` (so ``A[a, b]``), then ``s`` (d),
then ``c`` (d).

Subcommands: ``run --seed S``, ``recompute {cap,solve} --seed S``, ``finalize``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rsr.exit_codes import ArgumentParser, Exit, run_main  # noqa: E402

EXPERIMENT = "experiments/psi-ls-capacity/run.py"
RUN_ID = "psi-ls-capacity"
PREREG_COMMIT = "87dc2ec"
PREREG_AMENDMENTS = {"A1": "f99f6ce"}
OUT_DEFAULT = ROOT / "runs" / RUN_ID


def _load(name: str, rel: str):
    """Import an experiment script by path, as its own module instance. Never copied."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


CV = _load("_lscap_cv", "experiments/offline-psi-converge/run.py")  # diagnostic A
OP = CV.OP  # offline-psi

SEEDS = OP.SEEDS
ARMS = OP.ARMS  # cf_all (primary), cf_live
PRIMARY = OP.PRIMARY
CHUNK = 32  # documents per assembly chunk (memory only; sums are order-fixed)
RIDGE_RHO = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)  # PREREG §3 item 4, descriptive
CUTOFF_SWEEP = (0.1, 0.01)  # Amendment 1 item 2: multipliers on the cutoff, descriptive
N_CHECK_XTX = 64  # C2
N_CHECK_XTY = 32  # C2
C2_TOL = 1e-9
C3_KKT_TOL = 1e-6
C3_FEASIBLE_SLACK = 1e-6
C3_QUAD_TOL = 1e-6
#: PREREG §1: diagnostic A's result.json, read for comparison and C3(b).
DIAG_A_OUT = Path(OP.R1._R) / ".worktrees/offline-psi-converge/runs/offline-psi-converge"
DIAG_A_SHA = {
    0: "8c88bd05f89a3bfaa192852bf1cd5251586bb1381a6ca3770f8596a2d7b2d965",
    1: "0bb3ee8c15bf95ea53262215008ef29561569188fcab4339cb2691794c421204",
    2: "496beeb4e98c64dc9f80305317fc9dc5177d85750f9ae05c69b09956826edfc6",
}
ControlFailure = CV.ControlFailure


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _finite(x) -> bool:
    return OP._finite(x)


# --------------------------------------------------------------------------- #
# PREREG §2: the normal equations from per-(document, t) sufficient statistics
# --------------------------------------------------------------------------- #


def n_features(d: int) -> int:
    return d * d + 2 * d


def normal_equations(
    S_: torch.Tensor, C_: torch.Tensor, Y: torch.Tensor, mask: torch.Tensor, chunk=CHUNK
) -> dict:
    """``X^T X``, ``X^T y``, ``y^T y`` over the masked cells ``[n, t, i]`` (cell
    features ``[vec(s_i c_t^T); s_i; c_t]``, target ``Y[n, t, i]``), float64.

    Per group (n, t): ``A = sum_i s_i s_i^T``, ``m = sum_i s_i``, ``g = sum_i y s_i``,
    ``gs = sum_i y``, ``cnt``. The bilinear block is ``sum_k A_k (x) c_k c_k^T``, one GEMM
    ``vec(A)^T vec(c c^T)`` with its indices permuted afterwards."""
    n, _t, d = S_.shape
    f64 = torch.float64
    P = torch.zeros(d * d, d * d, dtype=f64)  # [(a,a'), (b,b')]
    X_bs = torch.zeros(d, d, d, dtype=f64)  # [a, e, b] -> rows (a,b), col e
    X_bc = torch.zeros(d, d * d, dtype=f64)  # [a, (b,f)]
    X_ss = torch.zeros(d, d, dtype=f64)
    X_sc = torch.zeros(d, d, dtype=f64)
    X_cc = torch.zeros(d, d, dtype=f64)
    y_b = torch.zeros(d, d, dtype=f64)
    y_s = torch.zeros(d, dtype=f64)
    y_c = torch.zeros(d, dtype=f64)
    yy = 0.0
    cells = 0
    groups = 0
    for j in range(0, n, chunk):
        Sj = S_[j : j + chunk].to(f64)
        Cj = C_[j : j + chunk].to(f64)
        Mj = mask[j : j + chunk].to(f64)  # [k, T, I]
        Yj = torch.where(mask[j : j + chunk], Y[j : j + chunk].to(f64), 0.0)
        Sm = Mj.unsqueeze(-1) * Sj.unsqueeze(1)  # [k, T, I, d]
        A = (Sm.transpose(-1, -2) @ Sj.unsqueeze(1)).reshape(-1, d, d)  # [K, d, d]
        m = Sm.sum(2).reshape(-1, d)
        g = (Yj @ Sj).reshape(-1, d)  # sum_i y_ti s_i
        gs = Yj.sum(-1).reshape(-1)
        cnt = Mj.sum(-1).reshape(-1)
        c = Cj.reshape(-1, d)
        cc = (c.unsqueeze(-1) * c.unsqueeze(-2)).reshape(-1, d * d)
        P += A.reshape(-1, d * d).T @ cc
        X_bs += (A.reshape(-1, d * d).T @ c).reshape(d, d, d)
        X_bc += m.T @ cc
        X_ss += A.sum(0)
        X_sc += m.T @ c
        X_cc += (cnt.unsqueeze(-1) * c).T @ c
        y_b += g.T @ c
        y_s += g.sum(0)
        y_c += (gs.unsqueeze(-1) * c).sum(0)
        yy += float((Yj * Yj).sum())
        cells += int(mask[j : j + chunk].sum())
        groups += int(mask[j : j + chunk].any(-1).sum())
    p = n_features(d)
    XtX = torch.empty(p, p, dtype=f64)
    B = P.view(d, d, d, d).permute(0, 2, 1, 3).reshape(d * d, d * d)
    del P
    XtX[: d * d, : d * d] = B
    del B
    bs = X_bs.permute(0, 2, 1).reshape(d * d, d)  # rows (a,b), col e
    bc = X_bc.reshape(d, d, d).reshape(d * d, d)  # rows (a,b), col f
    XtX[: d * d, d * d : d * d + d] = bs
    XtX[: d * d, d * d + d :] = bc
    XtX[d * d : d * d + d, : d * d] = bs.T
    XtX[d * d + d :, : d * d] = bc.T
    XtX[d * d : d * d + d, d * d : d * d + d] = X_ss
    XtX[d * d : d * d + d, d * d + d :] = X_sc
    XtX[d * d + d :, d * d : d * d + d] = X_sc.T
    XtX[d * d + d :, d * d + d :] = X_cc
    Xty = torch.cat([y_b.reshape(-1), y_s, y_c])
    return {"XtX": XtX, "Xty": Xty, "yy": yy, "cells": cells, "groups": groups}


def xt_resid(
    S_: torch.Tensor, C_: torch.Tensor, R: torch.Tensor, mask: torch.Tensor, chunk=CHUNK
) -> torch.Tensor:
    """``X^T r`` straight from the cells (C3(a)), never through ``X^T X``."""
    n, _t, d = S_.shape
    f64 = torch.float64
    y_b = torch.zeros(d, d, dtype=f64)
    y_s = torch.zeros(d, dtype=f64)
    y_c = torch.zeros(d, dtype=f64)
    for j in range(0, n, chunk):
        Sj = S_[j : j + chunk].to(f64)
        Cj = C_[j : j + chunk].to(f64).reshape(-1, d)
        Rj = torch.where(mask[j : j + chunk], R[j : j + chunk].to(f64), 0.0)
        g = (Rj @ Sj).reshape(-1, d)
        y_b += g.T @ Cj
        y_s += g.sum(0)
        y_c += (Rj.sum(-1).reshape(-1, 1) * Cj).sum(0)
    return torch.cat([y_b.reshape(-1), y_s, y_c])


def feature(S_: torch.Tensor, C_: torch.Tensor, j: int) -> torch.Tensor:
    """Feature ``j`` at every ``[n, t, i]`` cell, float64 (C2's direct sums)."""
    n, T, d = S_.shape
    S64, C64 = S_.to(torch.float64), C_.to(torch.float64)
    if j < d * d:
        a, b = divmod(j, d)
        return C64[:, :, b].unsqueeze(-1) * S64[:, :, a].unsqueeze(1)
    if j < d * d + d:
        return S64[:, :, j - d * d].unsqueeze(1).expand(n, T, T)
    return C64[:, :, j - d * d - d].unsqueeze(-1).expand(n, T, T)


def block_of(j: int, d: int) -> int:
    return 0 if j < d * d else (1 if j < d * d + d else 2)


def c2_assembly(ne: dict, S_, C_, Y, mask, seed: int) -> dict:
    """C2: stratified entries of ``X^T X`` (each of the 6 block pairs) and ``X^T y`` (3
    blocks) against direct sums over the cells, ``|structured - direct| <= 1e-9 *
    sum|terms|``."""
    d = S_.shape[-1]
    g = torch.Generator().manual_seed(int(seed))
    lo = (0, d * d, d * d + d)
    hi = (d * d, d * d + d, d * d + 2 * d)

    def draw(blk):
        return int(torch.randint(lo[blk], hi[blk], (1,), generator=g))

    pairs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    m = mask.to(torch.float64)
    Y64 = torch.where(mask, Y.to(torch.float64), 0.0)
    worst = 0.0
    checked = []
    for q in range(N_CHECK_XTX):
        bi, bj = pairs[q % len(pairs)]
        r, c = draw(bi), draw(bj)
        prod = feature(S_, C_, r) * feature(S_, C_, c) * m
        direct, scale = float(prod.sum()), float(prod.abs().sum())
        err = abs(float(ne["XtX"][r, c]) - direct)
        worst = max(worst, err / scale if scale else err)
        checked.append(("XtX", r, c, err <= C2_TOL * scale))
    for q in range(N_CHECK_XTY):
        r = draw(q % 3)
        prod = feature(S_, C_, r) * Y64
        direct, scale = float(prod.sum()), float(prod.abs().sum())
        err = abs(float(ne["Xty"][r]) - direct)
        worst = max(worst, err / scale if scale else err)
        checked.append(("Xty", r, None, err <= C2_TOL * scale))
    return {
        "n_checked": len(checked),
        "worst_rel": worst,
        "ok": all(x[3] for x in checked),
    }


def solve(XtX: torch.Tensor, Xty: torch.Tensor) -> dict:
    """PREREG §2: minimum-norm least squares through ``eigh``; eigenvalues ``<= p * eps
    * lambda_max`` dropped. Returns the eigenbasis too, for the ridge path."""
    p = XtX.shape[0]
    lam, V = torch.linalg.eigh(XtX)
    cut = p * torch.finfo(torch.float64).eps * float(lam.max())
    keep = lam > cut
    z = V.T @ Xty
    w = V[:, keep] @ (z[keep] / lam[keep])
    return {
        "w": w,
        "lam": lam,
        "V": V,
        "z": z,
        "keep": keep,
        "spectrum": {
            "p": p,
            "rank": int(keep.sum()),
            "n_dropped": int((~keep).sum()),
            "lambda_max": float(lam.max()),
            "lambda_min_kept": float(lam[keep].min()),
            "lambda_min": float(lam.min()),
            "lambda_mean": float(lam.mean()),
            "cutoff": cut,
        },
    }


def cutoff_w(sol: dict, mult: float) -> tuple[torch.Tensor, int]:
    """Amendment 1 item 2: the minimum-norm solution at ``mult`` x the registered
    cutoff. -> (w, rank)."""
    keep = sol["lam"] > mult * sol["spectrum"]["cutoff"]
    w = sol["V"][:, keep] @ (sol["z"][keep] / sol["lam"][keep])
    return w, int(keep.sum())


def kkt_split(sol: dict, g: torch.Tensor, xty_norm: float) -> dict:
    """Amendment 1 item 1: ``X^T(Xw - y)`` (computed from the cells) relative to
    ``|X^T y|``, on the kept eigenvectors (gated), the dropped ones and in full
    (reported). A truncated min-norm solution leaves ``-V_drop^T X^T y`` by
    construction."""
    keep = sol["keep"]
    return {
        "kept": float((sol["V"][:, keep].T @ g).norm()) / xty_norm,
        "dropped": float((sol["V"][:, ~keep].T @ g).norm()) / xty_norm,
        "full": float(g.norm()) / xty_norm,
    }


def ridge_w(sol: dict, rho: float) -> torch.Tensor:
    lam_r = rho * sol["spectrum"]["lambda_mean"]
    return sol["V"] @ (sol["z"] / (sol["lam"] + lam_r))


def quad_sse(ne: dict, w: torch.Tensor) -> float:
    """``y^T y - 2 w^T X^T y + w^T X^T X w`` (C3(c))."""
    return float(ne["yy"] - 2.0 * (w @ ne["Xty"]) + w @ (ne["XtX"] @ w))


def head_state(w: torch.Tensor, d: int) -> dict:
    """(A, a, b) -> `BilinearValueHead`'s (W, u): ``W = d A``, ``u = 2d [a; b]``."""
    A = w[: d * d].reshape(d, d)
    return {"W": d * A, "u": 2 * d * w[d * d :]}


def ls_head(seed: int, state: dict):
    """offline-psi's head (run 2's psiC phi seed), float64, holding ``state``."""
    head = CV.new_head(seed).double()
    head.load_state_dict({k: v.to(torch.float64) for k, v in state.items()})
    head.requires_grad_(False)
    return head


# --------------------------------------------------------------------------- #
# PREREG §3: readings, through offline-psi's functions
# --------------------------------------------------------------------------- #


def cap_reading(head, inp: dict, positions: list[int], seed: int, *, rho=True) -> dict:
    """offline-psi PREREG §5's cap over the documents at ``positions``, with the paired
    per-document bootstrap ``boot_weights(len, seed, 2000)``; plus B0-CF's G
    (reference)."""
    pt = torch.tensor(positions, dtype=torch.long)
    docs = [inp["docs"][p] for p in positions]
    rr = OP.reference_rates(docs, seed)
    W = OP.B0.boot_weights(len(docs), seed, OP.R1.N_BOOT)
    P = OP.score_matrices(head, inp["S64"][pt], inp["C64"][pt])
    out = OP.B0.boot_rule(W, OP.replay_rates(docs, list(P)), rr["fifo"], rr["factfiller"])
    Gd = torch.stack([OP.target_of(inp["blob"]["Delta"][d.doc_id]) for d in docs])
    out["reference_G"] = OP.B0.boot_rule(
        W, OP.replay_rates(docs, list(Gd)), rr["fifo"], rr["factfiller"]
    )["cap"]
    if rho:
        out["rho_psi_G_fifo_resident"] = OP.rho_psi_g(P, Gd, W)
    return out


def cap_only(head, inp: dict, positions: list[int], seed: int) -> dict:
    pt = torch.tensor(positions, dtype=torch.long)
    docs = [inp["docs"][p] for p in positions]
    rr = OP.reference_rates(docs, seed)
    W = OP.B0.boot_weights(len(docs), seed, OP.R1.N_BOOT)
    P = OP.score_matrices(head, inp["S64"][pt], inp["C64"][pt])
    return OP.B0.boot_rule(
        W, OP.replay_rates(docs, list(P)), rr["fifo"], rr["factfiller"]
    )["cap"]


def load_inputs(seed: int) -> dict:
    """PREREG §1: diagnostic A's `load_inputs` (gestalt and Delta pins, C1 and C5), plus
    diagnostic A's result.json, sha-pinned."""
    inp = CV.load_inputs(seed)
    inp["S64"] = inp["S"].to(torch.float64)
    inp["C64"] = inp["C"].to(torch.float64)
    rp = DIAG_A_OUT / f"seed{seed}" / "result.json"
    rsha = sha256_file(rp)
    inp["diag_a"] = json.loads(rp.read_text())
    inp["c1"]["diag_a_sha256"] = rsha
    inp["c1"]["diag_a_ok"] = rsha == DIAG_A_SHA[seed]
    inp["c1"]["ok"] = inp["c1"]["ok"] and inp["c1"]["diag_a_ok"]
    return inp


def fit_arm(inp: dict, arm: str, seed: int) -> dict:
    """Assemble, solve, map to the head; the C2/C3 checks and the train L_MC values."""
    d = inp["S"].shape[-1]
    ft = OP.fit_tensors(inp["blob"], inp["docs"], arm)
    G, mask = ft["G"], ft["mask"]
    tp = torch.tensor(inp["train"], dtype=torch.long)
    S_t, C_t, G_t, M_t = inp["S64"][tp], inp["C64"][tp], G[tp], mask[tp]
    t0 = time.perf_counter()
    ne = normal_equations(S_t, C_t, G_t, M_t)
    t_asm = time.perf_counter() - t0
    c2 = c2_assembly(ne, S_t, C_t, G_t, M_t, seed)
    t0 = time.perf_counter()
    sol = solve(ne["XtX"], ne["Xty"])
    t_solve = time.perf_counter() - t0
    w = sol["w"]
    state = head_state(w, d)
    head = ls_head(seed, state)
    G64 = G.to(torch.float64)
    train_head = OP.val_loss(head, inp["S64"], inp["C64"], G64, mask, inp["train"])
    train_quad = quad_sse(ne, w) / ne["groups"]
    with torch.no_grad():
        R = torch.cat(
            [
                OP.psi_matrix(head, S_t[j : j + 16], C_t[j : j + 16])
                for j in range(0, len(tp), 16)
            ]
        ) - G_t.to(torch.float64)
    kkt = kkt_split(sol, xt_resid(S_t, C_t, R, M_t), float(ne["Xty"].norm()))
    da = inp["diag_a"]["arms"][arm]
    da_train = float(da["train_l_mc_at_selected"])
    c3 = {
        "kkt_rel_kept": kkt["kept"],  # gated (Amendment 1 item 1)
        "kkt_rel_dropped": kkt["dropped"],
        "kkt_rel_full": kkt["full"],
        "kkt_ok": kkt["kept"] <= C3_KKT_TOL,
        "train_l_mc_head": train_head,
        "train_l_mc_quad": train_quad,
        "quad_rel": abs(train_head - train_quad) / abs(train_quad),
        "diag_a_train_l_mc": da_train,
        "feasible_ok": train_head <= (1.0 + C3_FEASIBLE_SLACK) * da_train,
    }
    c3["quad_ok"] = c3["quad_rel"] <= C3_QUAD_TOL
    c3["ok"] = c3["kkt_ok"] and c3["feasible_ok"] and c3["quad_ok"]
    return {
        "ne": ne,
        "sol": sol,
        "state": state,
        "head": head,
        "G": G,
        "mask": mask,
        "c2": c2,
        "c3": c3,
        "seconds": {"assembly": t_asm, "eigh": t_solve},
    }


def run_seed(seed: int, out_root: Path, *, arms=ARMS, ridge=True) -> dict:
    sd = Path(out_root) / f"seed{seed}"
    out = sd / "result.json"
    if out.exists():
        raise ControlFailure(f"{out} exists; never overwrite a result")
    sd.mkdir(parents=True, exist_ok=True)
    doc: dict[str, Any] = {
        "run_id": RUN_ID,
        "seed": seed,
        "tree": OP.R1.tree_state(),
        "hardware": OP.R1.hardware(),
        "torch_threads": torch.get_num_threads(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "controls": {},
        "arms": {},
    }
    seal = CV._seal_eval()
    inp = load_inputs(seed)
    doc["controls"]["C1"] = inp["c1"]
    doc["controls"]["C5"] = {
        **inp["c5"],
        **seal,
        "ok": inp["c5"]["ok"] and len(seal["sealed"]) == 3,
    }
    c2, c3, c4 = {}, {}, {}
    for arm in arms:
        f = fit_arm(inp, arm, seed)
        c2[arm], c3[arm] = f["c2"], f["c3"]
        torch.save(f["state"], sd / f"phi-{arm}-ls.pt")
        head = f["head"]
        G64 = f["G"].to(torch.float64)
        val_l = OP.val_loss(head, inp["S64"], inp["C64"], G64, f["mask"], inp["val"])
        da = inp["diag_a"]["arms"][arm]
        a: dict[str, Any] = {
            "phi_sha256": sha256_file(sd / f"phi-{arm}-ls.pt"),
            "spectrum": f["sol"]["spectrum"],
            "cells_train": f["ne"]["cells"],
            "groups_train": f["ne"]["groups"],
            "seconds": f["seconds"],
            "train_l_mc": f["c3"]["train_l_mc_head"],
            "val_l_mc": val_l,
            "cap_val": cap_reading(head, inp, inp["val"], seed),
            "cap_train": cap_reading(head, inp, inp["train"], seed, rho=False),
            "diag_a": {
                "selected_step": da["selected"]["step"],
                "val_l_mc": da["selected"]["val_l_mc"],
                "train_l_mc": da["train_l_mc_at_selected"],
                "constant_val_l_mc": da["constant_baseline"]["val_l_mc"],
                "cap_point": da["cap_selected"]["cap"]["point"],
            },
            "norms": {
                "W": float(f["state"]["W"].norm()),
                "u": float(f["state"]["u"].norm()),
            },
        }
        a["optimizer_shortfall"] = {
            "train": a["diag_a"]["train_l_mc"] - a["train_l_mc"],
            "val": a["diag_a"]["val_l_mc"] - a["val_l_mc"],
        }
        if ridge:
            path = []
            for rho in RIDGE_RHO:
                hr = ls_head(seed, head_state(ridge_w(f["sol"], rho), inp["S"].shape[-1]))
                path.append(
                    {
                        "rho": rho,
                        "val_l_mc": OP.val_loss(
                            hr, inp["S64"], inp["C64"], G64, f["mask"], inp["val"]
                        ),
                        "cap_val": cap_only(hr, inp, inp["val"], seed),
                    }
                )
            a["ridge_path"] = path
            sweep = []
            for mult in CUTOFF_SWEEP:
                wc, rank = cutoff_w(f["sol"], mult)
                hc = ls_head(seed, head_state(wc, inp["S"].shape[-1]))
                sweep.append(
                    {
                        "mult": mult,
                        "rank": rank,
                        "train_l_mc": OP.val_loss(
                            hc, inp["S64"], inp["C64"], G64, f["mask"], inp["train"]
                        ),
                        "val_l_mc": OP.val_loss(
                            hc, inp["S64"], inp["C64"], G64, f["mask"], inp["val"]
                        ),
                        "cap_val": cap_only(hc, inp, inp["val"], seed),
                        "cap_train": cap_only(hc, inp, inp["train"], seed),
                    }
                )
            a["cutoff_sweep"] = sweep
        nums = [
            a["train_l_mc"],
            a["val_l_mc"],
            a["cap_val"]["cap"]["point"],
            a["cap_train"]["cap"]["point"],
            a["spectrum"]["lambda_max"],
            a["spectrum"]["lambda_min_kept"],
        ]
        c4[arm] = {"ok": all(_finite(x) for x in nums)}
        doc["arms"][arm] = a
        del f
    doc["controls"].update(C2=c2, C3=c3, C4=c4)
    flags = [
        doc["controls"]["C1"]["ok"],
        doc["controls"]["C5"]["ok"],
        *(v["ok"] for v in c2.values()),
        *(v["ok"] for v in c3.values()),
        *(v["ok"] for v in c4.values()),
    ]
    doc["controls_ok"] = all(flags)
    doc["exit"] = int(Exit.OK if doc["controls_ok"] else Exit.DID_NOT_RUN)
    doc["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out.write_text(json.dumps(doc, indent=1, default=float))
    return doc


# --------------------------------------------------------------------------- #
# PREREG §6: labels and route
# --------------------------------------------------------------------------- #


def route(val_label: str, insample_label: str, ok: bool, sensitive: bool = False) -> str:
    """PREREG §6; Amendment 1 item 3 adds "cutoff-sensitive" when the labels at x0.01
    differ from the registered cutoff's."""
    r = _route(val_label, insample_label, ok)
    return f"{r} [cutoff-sensitive]" if (sensitive and ok) else r


def _route(val_label: str, insample_label: str, ok: bool) -> str:
    if not ok:
        return "controls failed: inconclusive (exit 3)"
    if val_label in ("FIT-POINTS", "FIT-PARTIAL"):
        return (
            "U-val POINTS/PARTIAL: the class's L_MC optimum ranks usefully; run 2's "
            "optimizer stopped short of it (PREREG §6 row 1)"
        )
    if insample_label == "IN-SAMPLE-MISSES":
        return (
            "MISSES + IN-SAMPLE-MISSES: capacity under L_MC is the limit; diagnostic A's "
            "§6 row 2 stands (PREREG §6 row 2)"
        )
    return (
        "MISSES + in-sample POINTS/PARTIAL: expressible on seen documents, does not "
        "generalize from 932 (PREREG §6 row 3)"
    )


def fmt(c: dict) -> str:
    return f"{c['point']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"


def labels_of(res: dict, arm: str, mult: float | None = None) -> tuple[str, str]:
    """The two labels at the registered cutoff, or at ``mult`` x it (Amendment 1)."""

    def at(s, key):
        a = res[s]["arms"][arm]
        if mult is None:
            return a[key]["cap"]
        return next(x for x in a["cutoff_sweep"] if x["mult"] == mult)[key]

    val = OP.fit_label({s: at(s, "cap_val") for s in SEEDS})
    ins = OP.fit_label({s: at(s, "cap_train") for s in SEEDS})
    return val, ins.replace("FIT-", "IN-SAMPLE-")


def finalize(out_root: Path) -> dict:
    res = {
        s: json.loads((Path(out_root) / f"seed{s}" / "result.json").read_text())
        for s in SEEDS
    }
    ok = all(res[s]["controls_ok"] for s in SEEDS)
    val_label, ins_label = labels_of(res, PRIMARY)
    live_val, live_ins = labels_of(res, "cf_live")
    low = labels_of(res, PRIMARY, min(CUTOFF_SWEEP))
    sensitive = low != (val_label, ins_label)
    rt = route(val_label, ins_label, ok, sensitive)
    L = [
        "# RESULTS: psi-ls-capacity (least-squares optimum of psi_hat's class)",
        "",
        "Generated by run.py finalize; no number is typed. Written by a model; not by "
        f"Brendan. PREREG `{PREREG_COMMIT}` + Amendment 1 "
        f"`{PREREG_AMENDMENTS['A1']}`. U documents only; EVAL never read; no "
        "gradient step.",
        "",
        f"- command: `.venv/bin/python {EXPERIMENT} finalize`",
        f"- git: `{res[0]['tree']}`",
        f"- hardware: {res[0]['hardware']}",
        f"- exit `{0 if ok else 3}`",
        "",
        f"**U-val fit label (cf_all): {val_label}. In-sample label (cf_all): "
        f"{ins_label}.**",
        "",
        f"**Route:** {rt}",
        "",
        f"cf_live (reported, never routed): {live_val}, {live_ins}.",
        "",
        f"At x{min(CUTOFF_SWEEP):g} the cutoff (Amendment 1 item 3): {low[0]}, {low[1]}"
        + (" -- DIFFERS from the registered cutoff's labels." if sensitive else "."),
        "",
        "## Per seed and arm",
        "",
        "| seed | arm | rank / p | train L_MC: LS (diag A) | U-val L_MC: LS (diag A) "
        "| const-mean U-val | U-val cap: LS | diag A cap | U-train cap: LS "
        "| G ref U-val | G ref U-train | rho(psi,G) U-val |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in SEEDS:
        for arm in ARMS:
            a = res[s]["arms"][arm]
            rho = a["cap_val"]["rho_psi_G_fifo_resident"]
            L.append(
                f"| {s} | {arm} | {a['spectrum']['rank']} / {a['spectrum']['p']} | "
                f"{a['train_l_mc']:.4f} ({a['diag_a']['train_l_mc']:.4f}) | "
                f"{a['val_l_mc']:.4f} ({a['diag_a']['val_l_mc']:.4f}) | "
                f"{a['diag_a']['constant_val_l_mc']:.4f} | {fmt(a['cap_val']['cap'])} | "
                f"{a['diag_a']['cap_point']:+.4f} | {fmt(a['cap_train']['cap'])} | "
                f"{a['cap_val']['reference_G']['point']:+.4f} | "
                f"{a['cap_train']['reference_G']['point']:+.4f} | "
                f"{rho['median']:+.4f} [{rho['lo']:+.4f}, {rho['hi']:+.4f}] |"
            )
    L += [
        "",
        "## Ridge path (descriptive; never selected on)",
        "",
        "| seed | arm | rho | U-val L_MC | U-val cap |",
        "| --- | --- | --- | --- | --- |",
    ]
    for s in SEEDS:
        for arm in ARMS:
            for r in res[s]["arms"][arm].get("ridge_path", []):
                L.append(
                    f"| {s} | {arm} | {r['rho']:g} | {r['val_l_mc']:.4f} | "
                    f"{fmt(r['cap_val'])} |"
                )
    L += [
        "",
        "## Cutoff sensitivity (Amendment 1 item 2; descriptive)",
        "",
        "| seed | arm | cutoff x | rank | U-train L_MC | U-val L_MC | U-val cap "
        "| U-train cap |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in SEEDS:
        for arm in ARMS:
            a = res[s]["arms"][arm]
            L.append(
                f"| {s} | {arm} | 1 (registered) | {a['spectrum']['rank']} | "
                f"{a['train_l_mc']:.4f} | {a['val_l_mc']:.4f} | "
                f"{fmt(a['cap_val']['cap'])} | {fmt(a['cap_train']['cap'])} |"
            )
            for x in a.get("cutoff_sweep", []):
                L.append(
                    f"| {s} | {arm} | {x['mult']:g} | {x['rank']} | "
                    f"{x['train_l_mc']:.4f} | {x['val_l_mc']:.4f} | "
                    f"{fmt(x['cap_val'])} | {fmt(x['cap_train'])} |"
                )
    L += ["", "## Spectrum", "", "| seed | arm | spectrum |", "| --- | --- | --- |"]
    for s in SEEDS:
        for arm in ARMS:
            L.append(f"| {s} | {arm} | `{json.dumps(res[s]['arms'][arm]['spectrum'])}` |")
    L += ["", "## Controls", ""]
    for s in SEEDS:
        L.append(
            f"- seed {s}: ok `{res[s]['controls_ok']}`; {json.dumps(res[s]['controls'])}"
        )
    L += [
        "",
        "## Not shown",
        "",
        "- No EVAL number; no deployment; no change to offline-psi's, diagnostic A's or "
        "run 2's labels.",
        "- Not the best RANKING the class can express (PREREG §3 item 5): the fit "
        "minimises L_MC, a squared error over all of the arm's cells.",
        "- No lr or muP change (correction 15). No scaling or cognitive claim. "
        "§15 untouched.",
        "",
    ]
    out = Path(out_root)
    (out / "RESULTS.md").write_text("\n".join(L))
    summary = {
        "label_val": val_label,
        "label_insample": ins_label,
        "labels_at_x0.01": list(low),
        "cutoff_sensitive": sensitive,
        "cf_live": [live_val, live_ins],
        "route": rt,
        "controls_ok": ok,
        "caps_val": {
            s: res[s]["arms"][PRIMARY]["cap_val"]["cap"]["point"] for s in SEEDS
        },
        "caps_train": {
            s: res[s]["arms"][PRIMARY]["cap_train"]["cap"]["point"] for s in SEEDS
        },
    }
    claims = []
    for s in SEEDS:
        a = res[s]["arms"][PRIMARY]
        claims.append(
            {
                "claim": f"seed {s} cf_all U-val and U-train caps of the saved LS phi",
                "command": f".venv/bin/python {EXPERIMENT} recompute cap --seed {s}",
                "expected": f"{a['cap_val']['cap']['point']!r} "
                f"{a['cap_train']['cap']['point']!r}",
            }
        )
        claims.append(
            {
                "claim": f"seed {s} cf_all LS re-solved from the inputs: train and U-val "
                "L_MC, U-val cap",
                "command": f".venv/bin/python {EXPERIMENT} recompute solve --seed {s}",
                "expected": f"{a['train_l_mc']!r} {a['val_l_mc']!r} "
                f"{a['cap_val']['cap']['point']!r}",
            }
        )
    (out / "claims.json").write_text(json.dumps(claims, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def recompute(what: str, seed: int, out_root: Path) -> str:
    CV._seal_eval()
    inp = load_inputs(seed)
    if what == "cap":
        st = torch.load(
            Path(out_root) / f"seed{seed}" / f"phi-{PRIMARY}-ls.pt", weights_only=True
        )
        head = ls_head(seed, st)
        v = cap_only(head, inp, inp["val"], seed)["point"]
        t = cap_only(head, inp, inp["train"], seed)["point"]
        return f"{v!r} {t!r}"
    if what == "solve":
        f = fit_arm(inp, PRIMARY, seed)
        G64 = f["G"].to(torch.float64)
        v = OP.val_loss(f["head"], inp["S64"], inp["C64"], G64, f["mask"], inp["val"])
        cap = cap_only(f["head"], inp, inp["val"], seed)["point"]
        return f"{f['c3']['train_l_mc_head']!r} {v!r} {cap!r}"
    raise ValueError(what)


def main(argv: list[str] | None = None) -> Exit:
    p = ArgumentParser(prog=RUN_ID, description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--seed", type=int, required=True, choices=SEEDS)
    r.add_argument("--out-root", type=Path, default=OUT_DEFAULT)
    rc = sub.add_parser("recompute")
    rc.add_argument("what", choices=("cap", "solve"))
    rc.add_argument("--seed", type=int, required=True, choices=SEEDS)
    rc.add_argument("--out-root", type=Path, default=OUT_DEFAULT)
    f = sub.add_parser("finalize")
    f.add_argument("--out-root", type=Path, default=OUT_DEFAULT)
    a = p.parse_args(argv)
    if a.cmd == "run":
        d = run_seed(a.seed, a.out_root)
        print(f"seed {a.seed} exit {d['exit']} controls_ok {d['controls_ok']}")
        return Exit.OK if d["controls_ok"] else Exit.DID_NOT_RUN
    if a.cmd == "recompute":
        print(recompute(a.what, a.seed, a.out_root))
        return Exit.OK
    s = finalize(a.out_root)
    print(json.dumps(s))
    return Exit.OK if s["controls_ok"] else Exit.DID_NOT_RUN


if __name__ == "__main__":
    run_main(main)
