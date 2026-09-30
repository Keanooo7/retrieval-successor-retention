"""B1: is β inert for φ at real scale? (PREREG a0c2b30, PLAN-v4 §2B B1)

Trains the real `BilinearValueHead` (d = 128) on B2's rowset-C capture of fresh-stream
arm B ckpt3000, target C@0.9, for 3000 optimizer steps, under four optimizer treatments
of φ at two ε and three β, and reads max|Δφ|, open-loop eviction-argmin agreement and
the ε/√v̂ distribution against β = 1 of the same cell. MEASURE-THEN-DECIDE: the output is
evidence for ADR-0009 L7 and L8, never a build, a correction or an ADR edit.

    .venv/bin/python experiments/b1-beta-inertness/run.py check
    .venv/bin/python experiments/b1-beta-inertness/run.py run [--threads 3]

Exit: 0 measured (whatever the table says); 3 VOID or did not run (PREREG §8). Never 5:
`Exit.INERT` means memory without row content, which is not this "inert".
"""

from __future__ import annotations

import importlib.util
import json
import math
import resource
import sys
import time
from pathlib import Path

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rsr.exit_codes import (  # noqa: E402
    ArgumentParser,
    Exit,
    did_not_run,
    run_main,
    status,
)
from rsr.retention.value_head import BilinearValueHead  # noqa: E402


def _load(name: str, rel: str):
    """Import an experiment script by path. Imported, never copied."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


#: B2's capture, loader, rowset and target -- imported (PREREG §2).
B2 = _load("_b1_b2", "experiments/b2-psi-probe/run.py")
LR = B2.LR
ControlFailure = B2.ControlFailure

RUN_ID = "b1-beta-inertness"
EXPERIMENT = "experiments/b1-beta-inertness/run.py"
PREREG = "experiments/b1-beta-inertness/PREREG.md"
PREREG_COMMIT = "a0c2b30"

# --------------------------------------------------------------------------- #
# 🔒 PREREG, transcribed. Changing any of these is changing the PREREG.
# --------------------------------------------------------------------------- #

SEEDS = (0, 1, 2)
CHECKPOINT = 3000
GAMMA = 0.9  # §2: target C@0.9
BETA_SWEEP = (0.01, 0.1, 1.0)
EPSILONS = (1e-8, 1e-12)
ARMS = ("decoupled_wd", "phi_clip", "joint_clip", "coupled_l2")
CONTROL_P = "P"  # §5: beta as phi's LR multiplier (L7(a))
CONTROL_N = "N"  # §5: arm 1, eps 1e-8, beta 1, re-run
CONTROL_I = "I"  # §5: arm 1, eps 1e-8, beta 1, init seed 1100 + s (context)
ADAM_BETAS = (0.9, 0.95)  # §4, loop.py:434
WD = 0.01  # §4, loop.py:434 (global)
L2 = 0.01  # §4, CHOSEN
PHI_CLIP = 1.0  # §4, CHOSEN (the transformer's threshold)
JOINT_CLIP = 1.0  # §4, loop.py:578
CLIP_EPS = 1e-6  # torch.nn.utils.clip_grad_norm_'s denominator term
BASE_LR = 1e-3  # fresh-stream manifest `lr`
BASE_WIDTH = 128  # param_groups.py / value_head.py
K = 3000  # §3
SNAPS = (100, 300, 1000, 3000)  # §3
LOG_STEPS = (1, 100, 1000, 3000)  # §6
BATCH = 16  # §3, fresh-stream `batch`
TRAIN_LO, N_TRAIN = 920000, 256  # §2
ARGMIN_LO, N_ARGMIN = 936000, 64  # §2
INIT_SEED_BASE = 1000  # §2
ORDER_SEED_BASE = 2000  # §3
CONTEXT_INIT_BASE = 1100  # §5 I
G_T_STEPS = (2000, 3000)  # §4: arm B heartbeat steps [2000, 3000)
INERT_AGREE, INERT_REL = 0.99, 0.01  # §7
LIVE_AGREE, LIVE_REL = 0.95, 0.10  # §7
T0_N = 435

QUESTION = (
    "At real scale (d=128, B2's capture of arm B ckpt3000, G = C@0.9), does beta in "
    "{0.01, 0.1, 1} change phi or its eviction argmin under {decoupled wd}, {phi clip}, "
    "{joint clip}, {coupled L2}, each at eps in {1e-8, 1e-12}?"
)
FALSIFIER = (
    "ADR-0009 L7's premise ('beta is inert under AdamW + isolation'), not a spec "
    "falsifier: LIVE in arm decoupled_wd at eps 1e-8 falsifies it at real scale."
)
EXPECTED = (
    "PREREG §9: decoupled_wd INERT at 1e-12, INERT more likely than not at 1e-8 (else "
    "INTERMEDIATE via beta=0.01); phi_clip = decoupled_wd (clip never binds); joint_clip "
    "INERT on phi, m4 nonzero but tiny; coupled_l2 LIVE; P LIVE; N bit-identical."
)


# --------------------------------------------------------------------------- #
# §2: the data
# --------------------------------------------------------------------------- #


def pack(caps, m: int) -> dict:
    """B2 captures -> fp32 tensors: gest/ctx [N, S, d], G [N, S, S] (0 off-row), rows
    [N, S, S] (B2 rowset C), full [N, S] (|resident[t]| = m). Refuses graph."""
    gest, ctx, G, rows = [], [], [], []
    for cap in caps:
        for name, x in (("gest", cap.gest), ("ctx", cap.ctx)):
            if x.requires_grad or x.grad_fn is not None:
                raise ControlFailure(f"doc {cap.doc_id}: captured {name} carries graph")
        S = cap.gest.shape[0]
        t, i = B2.row_index(cap, "C", m)
        r = torch.zeros(S, S, dtype=torch.bool)
        r[t, i] = True
        g = B2.target_matrix(cap, "C", GAMMA)
        if not bool(torch.isfinite(g[r]).all()):
            raise ControlFailure(f"doc {cap.doc_id}: non-finite target on a row")
        gest.append(cap.gest.to(torch.float32))
        ctx.append(cap.ctx.to(torch.float32))
        G.append(torch.where(r, g, torch.zeros_like(g)).to(torch.float32))
        rows.append(r)
    rows_t = torch.stack(rows)
    return {
        "gest": torch.stack(gest),
        "ctx": torch.stack(ctx),
        "G": torch.stack(G),
        "rows": rows_t,
        "full": rows_t.sum(-1) == m,
        "doc_ids": [c.doc_id for c in caps],
    }


def read_g_T(path: Path) -> Tensor:
    """§4: arm B's logged pre-clip transformer grad norm at steps [2000, 3000), in step
    order. Refuses anything but exactly one value per step."""
    lo, hi = G_T_STEPS
    got: dict[int, float] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        b = json.loads(line)
        if b.get("kind") != "beat" or "grad_norm" not in b:
            continue
        s = int(b["step"])
        if lo <= s < hi:
            if s in got:
                raise ControlFailure(f"{path}: step {s} logged twice")
            got[s] = float(b["grad_norm"])
    if sorted(got) != list(range(lo, hi)):
        raise ControlFailure(f"{path}: {len(got)} of {hi - lo} steps in [{lo}, {hi})")
    return torch.tensor([got[s] for s in range(lo, hi)], dtype=torch.float64)


# --------------------------------------------------------------------------- #
# §3: psi-hat, L_MC, the minibatch order
# --------------------------------------------------------------------------- #


def psi_all(head: BilinearValueHead, gest: Tensor, ctx: Tensor) -> Tensor:
    """[N, S(t), S(i)]: psi[n, t, i] = head(s_i, c_t) of document n, through the head's
    own forward."""
    N, S, d = gest.shape
    g = gest.unsqueeze(1).expand(N, S, S, d).reshape(N * S, S, d)
    return head(g, ctx.reshape(N * S, d)).reshape(N, S, S)


def l_mc(psi: Tensor, G: Tensor, rows: Tensor) -> Tensor:
    """§3 / ADR-0009 L9: sum over resident slots, mean over (doc, t), t in [1, S)."""
    N, S = rows.shape[:2]
    diff = torch.where(rows, psi - G, torch.zeros_like(psi))
    return (diff * diff).sum() / (N * (S - 1))


def batch_order(n: int, batch: int, steps: int, *, seed: int) -> Tensor:
    """§3: `batch` documents per step, without replacement within an epoch, from a
    dedicated CPU generator -- identical for every beta, arm and eps of a seed."""
    if n % batch:
        raise ValueError(f"n={n} is not a multiple of batch={batch}")
    g = torch.Generator().manual_seed(seed)
    out: list[Tensor] = []
    while len(out) < steps:
        perm = torch.randperm(n, generator=g)
        out += [perm[k * batch : (k + 1) * batch] for k in range(n // batch)]
    return torch.stack(out[:steps])


# --------------------------------------------------------------------------- #
# §4 / §5: the arms and P
# --------------------------------------------------------------------------- #


def lr_of(d: int) -> float:
    """param_groups.py VALUE_HEAD_GROUP: base_lr / m, m = d / base_width."""
    return BASE_LR / (d / BASE_WIDTH)


def loss_scale(arm: str, beta: float) -> float:
    """beta multiplies L_MC in the four arms; in P it multiplies the LR instead."""
    return 1.0 if arm == CONTROL_P else beta


def make_optimizer(arm: str, params, *, eps: float, beta: float, lr: float):
    if arm == "coupled_l2":
        return torch.optim.Adam(params, lr=lr, betas=ADAM_BETAS, eps=eps, weight_decay=L2)
    lr_eff = beta * lr if arm == CONTROL_P else lr
    return torch.optim.AdamW(
        params, lr=lr_eff, betas=ADAM_BETAS, eps=eps, weight_decay=WD
    )


def _grad_norm(params) -> float:
    return math.sqrt(sum(float((p.grad.double() ** 2).sum()) for p in params))


def treat(arm: str, params, *, g_T: float | None) -> dict:
    """After backward of `loss_scale * L_MC`: the arm's clip, if any. Returns the
    pre-clip norm and, per arm, whether the phi clip bound (arm 2) or the joint
    coefficient and its deviation from the transformer-only one (arm 3, m4)."""
    n = _grad_norm(params)
    info: dict = {"gnorm": n}
    if arm == "phi_clip":
        torch.nn.utils.clip_grad_norm_(params, PHI_CLIP)
        info["clipped"] = bool(n > PHI_CLIP)
    elif arm == "joint_clip":
        if g_T is None:
            raise ValueError("joint_clip needs the transformer's grad norm")
        coef = min(1.0, JOINT_CLIP / (math.sqrt(g_T * g_T + n * n) + CLIP_EPS))
        for p in params:
            p.grad.mul_(coef)
        c_T = min(1.0, JOINT_CLIP / (g_T + CLIP_EPS))
        info["coef"] = coef
        info["dev"] = abs(coef - c_T) / c_T
    return info


# --------------------------------------------------------------------------- #
# §6: the metrics, §7: the rule
# --------------------------------------------------------------------------- #


def argmins(psi: Tensor, rows: Tensor, full: Tensor) -> Tensor:
    """Open-loop eviction argmin over resident slots at every full-memory (doc, t),
    ties to the lowest slot index; flattened in (doc, t) order."""
    x = psi.masked_fill(~rows, float("inf"))
    return x.argmin(-1)[full]


def eps_ratio(opt, eps: float) -> Tensor:
    """§6 m3: eps / sqrt(v / (1 - beta2^k)) for every coordinate."""
    out = []
    for group in opt.param_groups:
        b2 = group["betas"][1]
        for p in group["params"]:
            st = opt.state[p]
            k = float(st["step"])
            vhat = st["exp_avg_sq"].double() / (1.0 - b2**k)
            out.append((eps / vhat.sqrt()).flatten())
    return torch.cat(out)


def quantiles(x: Tensor) -> dict:
    x = x.double().flatten()
    q = torch.quantile(x, torch.tensor([0.01, 0.5, 0.99], dtype=torch.float64))
    return {
        "min": float(x.min()),
        "p01": float(q[0]),
        "p50": float(q[1]),
        "p99": float(q[2]),
        "max": float(x.max()),
        "frac_gt_0.01": float((x > 0.01).double().mean()),
        "frac_gt_0.1": float((x > 0.1).double().mean()),
    }


def classify(agreements, rels) -> str:
    """§7, over both beta != 1 and every seed of one (arm, eps) cell."""
    if min(agreements) < LIVE_AGREE or max(rels) > LIVE_REL:
        return "LIVE"
    if min(agreements) >= INERT_AGREE and max(rels) <= INERT_REL:
        return "INERT"
    return "INTERMEDIATE"


def reference_arm(arm: str) -> str:
    """P's beta = 1 is arm 1's beta = 1 run (§5); every other cell is its own."""
    return "decoupled_wd" if arm in (CONTROL_P, CONTROL_N, CONTROL_I) else arm


def compare(
    runs: dict, seed: int, arm: str, eps: float, beta: float, *, snap: int
) -> dict:
    a = runs[(seed, arm, eps, beta)]
    ref = runs[(seed, reference_arm(arm), eps, 1.0)]
    d = (a["phi"][snap] - ref["phi"][snap]).abs().max()
    m = ref["phi"][snap].abs().max()
    am, rm = a["argmin"][snap], ref["argmin"][snap]
    return {
        "max_abs": float(d),
        "rel": float(d / m),
        "agreement": float((am == rm).double().mean()),
        "n_decisions": int(am.numel()),
    }


# --------------------------------------------------------------------------- #
# one training run
# --------------------------------------------------------------------------- #


def _phi(ps) -> Tensor:
    return torch.cat([p.detach().flatten().clone() for p in ps])


def train_one(
    train: dict,
    held: dict,
    *,
    arm: str,
    eps: float,
    beta: float,
    steps: int = K,
    snaps=SNAPS,
    batch: int = BATCH,
    seed: int,
    init_seed: int,
    g_T: Tensor | None,
) -> dict:
    d = int(train["gest"].shape[-1])
    head = BilinearValueHead(
        d, base_width=BASE_WIDTH, generator=torch.Generator().manual_seed(init_seed)
    )
    ps = list(head.parameters())
    opt = make_optimizer(arm, ps, eps=eps, beta=beta, lr=lr_of(d))
    order = batch_order(len(train["doc_ids"]), batch, steps, seed=ORDER_SEED_BASE + seed)
    scale = loss_scale(arm, beta)
    rec: dict = {
        "arm": arm,
        "eps": eps,
        "beta": beta,
        "init_seed": init_seed,
        "lr": opt.param_groups[0]["lr"],
        "loss_scale": scale,
        "phi": {},
        "argmin": {},
        "loss": {},
        "grad_norm_unscaled": {},
    }
    clipped, devs, coefs = 0, [], []
    for k in range(steps):
        idx = order[k]
        psi = psi_all(head, train["gest"][idx], train["ctx"][idx])
        loss = l_mc(psi, train["G"][idx], train["rows"][idx])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                f"{arm} eps={eps} beta={beta} step {k + 1}: loss {loss}"
            )
        opt.zero_grad(set_to_none=True)
        (scale * loss).backward()
        gT = None if g_T is None else float(g_T[k % len(g_T)])
        info = treat(arm, ps, g_T=gT)
        opt.step()
        step = k + 1
        if arm == "phi_clip":
            clipped += int(info["clipped"])
        if arm == "joint_clip":
            devs.append(info["dev"])
            coefs.append(info["coef"])
        if step in LOG_STEPS:
            rec["loss"][step] = float(loss.detach())
            rec["grad_norm_unscaled"][step] = info["gnorm"] / scale
        if step in snaps:
            phi = _phi(ps)
            if not bool(torch.isfinite(phi).all()):
                raise FloatingPointError(f"{arm} eps={eps} beta={beta}: phi non-finite")
            rec["phi"][step] = phi
            with torch.no_grad():
                ph = psi_all(head, held["gest"], held["ctx"])
            rec["argmin"][step] = argmins(ph, held["rows"], held["full"])
    rec["eps_ratio"] = quantiles(eps_ratio(opt, eps))
    rec["phi_max_abs"] = float(_phi(ps).abs().max())
    if arm == "phi_clip":
        rec["clip_frac"] = clipped / steps
    if arm == "joint_clip":
        dv = torch.tensor(devs, dtype=torch.float64)
        rec["joint_dev"] = {"max": float(dv.max()), "median": float(dv.median())}
        rec["joint_coef"] = {"min": min(coefs), "max": max(coefs)}
    return rec


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _key(arm: str, eps: float, beta: float) -> str:
    return f"{arm}.eps{eps:g}.beta{beta:g}"


def seed_runs(s: int, source: Path) -> dict:
    """Capture, pack and every run of one substrate seed. -> summary dict (no tensors
    but the argmin/phi needed for comparison, which stay in `runs`)."""
    t0 = time.time()
    model, vmap = B2.load_checked(source, CHECKPOINT, s)
    S, M, L = B2.S_STEPS, B2.M_STEPS, B2.L_TOKENS
    train_docs = [B2.doc_by_id(s, i) for i in range(TRAIN_LO, TRAIN_LO + N_TRAIN)]
    held_docs = [B2.doc_by_id(s, i) for i in range(ARGMIN_LO, ARGMIN_LO + N_ARGMIN)]
    clos = B2.vocabulary_closure(train_docs + held_docs, vmap)
    if not clos["ok"]:
        raise ControlFailure(f"seed {s}: vocabulary closure {clos}")
    kw = {"vmap": vmap, "S": S, "m": M, "L": L, "ranks": (0,)}
    tcaps = [B2.capture_doc(model, d, **kw) for d in train_docs]
    hcaps = [B2.capture_doc(model, d, **kw) for d in held_docs]
    ident = max(c.identity_worst[0] for c in tcaps + hcaps)
    sumw = max(c.sum_worst for c in tcaps + hcaps)
    if ident > B2.IDENTITY_TOL or sumw > B2.SUM_TOL:
        raise ControlFailure(f"seed {s}: capture identity {ident} sum {sumw}")
    cap_s = time.time() - t0
    d = int(model.cfg.D)
    del model
    train, held = pack(tcaps, M), pack(hcaps, M)
    g_T = read_g_T(source / "B" / f"seed{s}" / "heartbeat.jsonl")
    rows = train["rows"]
    Gr = train["G"][rows].double()
    gn = torch.linalg.vector_norm(train["gest"].double(), dim=-1)
    cn = torch.linalg.vector_norm(train["ctx"].double(), dim=-1)
    data = {
        "d": d,
        "capture_seconds": cap_s,
        "identity_worst": ident,
        "sum_worst": sumw,
        "n_train_rows": int(rows.sum()),
        "n_decisions": int(held["full"].sum()),
        "G_rows": {
            "mean": float(Gr.mean()),
            "sd": float(Gr.std()),
            "min": float(Gr.min()),
            "max": float(Gr.max()),
        },
        "gest_norm": {"min": float(gn.min()), "max": float(gn.max())},
        "ctx_norm": {"min": float(cn.min()), "max": float(cn.max())},
        "g_T": {
            "median": float(g_T.median()),
            "min": float(g_T.min()),
            "max": float(g_T.max()),
        },
    }
    _log(f"seed {s}: captured {len(tcaps)}+{len(hcaps)} docs in {cap_s:.0f}s; {data}")

    runs: dict = {}
    t1 = time.time()
    base = {"seed": s, "init_seed": INIT_SEED_BASE + s}

    def go(tag: str, arm: str, eps: float, beta: float, **over) -> None:
        r = train_one(
            train, held, arm=arm, eps=eps, beta=beta, g_T=g_T, **{**base, **over}
        )
        runs[(s, tag, eps, beta)] = r
        _log(f"seed {s} {_key(tag, eps, beta)} done (phi max {r['phi_max_abs']:.4g})")

    for arm in ARMS:
        for eps in EPSILONS:
            for beta in BETA_SWEEP:
                go(arm, arm, eps, beta)
    for eps in EPSILONS:
        for beta in (0.01, 0.1):
            go(CONTROL_P, CONTROL_P, eps, beta)
    go(CONTROL_N, "decoupled_wd", 1e-8, 1.0)
    go(CONTROL_I, "decoupled_wd", 1e-8, 1.0, init_seed=CONTEXT_INIT_BASE + s)
    data["train_seconds"] = time.time() - t1
    return {"data": data, "runs": runs}


def _run_summary(r: dict) -> dict:
    keep = ("arm", "eps", "beta", "init_seed", "lr", "loss_scale", "loss")
    out = {k: r[k] for k in keep}
    out["grad_norm_unscaled"] = r["grad_norm_unscaled"]
    out["eps_ratio"] = r["eps_ratio"]
    out["phi_max_abs"] = r["phi_max_abs"]
    for k in ("clip_frac", "joint_dev", "joint_coef"):
        if k in r:
            out[k] = r[k]
    return out


def aggregate(all_runs: dict, seeds) -> dict:
    """§6 per (seed, cell), §7 per (arm, eps), P, N and I."""
    per: dict = {}
    cls: dict = {}
    for arm in (*ARMS, CONTROL_P):
        for eps in EPSILONS:
            agr, rel = [], []
            for s in seeds:
                for beta in (0.01, 0.1):
                    snaps = {
                        snap: compare(all_runs, s, arm, eps, beta, snap=snap)
                        for snap in SNAPS
                    }
                    per[f"seed{s}.{_key(arm, eps, beta)}"] = snaps
                    agr.append(snaps[K]["agreement"])
                    rel.append(snaps[K]["rel"])
            cls[f"{arm}.eps{eps:g}"] = {
                "class": classify(agr, rel),
                "min_agreement": min(agr),
                "max_rel": max(rel),
                "agreements": agr,
                "rels": rel,
            }
    n_ok, ctx = True, {}
    for s in seeds:
        c = compare(all_runs, s, CONTROL_N, 1e-8, 1.0, snap=K)
        per[f"seed{s}.N"] = c
        n_ok = n_ok and c["max_abs"] == 0.0 and c["agreement"] == 1.0
        ctx[f"seed{s}"] = compare(all_runs, s, CONTROL_I, 1e-8, 1.0, snap=K)
    void = []
    if not n_ok:
        void.append("N not bit-identical")
    for eps in EPSILONS:
        if cls[f"{CONTROL_P}.eps{eps:g}"]["class"] != "LIVE":
            void.append(f"P not LIVE at eps {eps:g}")
    return {"per_cell": per, "classes": cls, "N_ok": n_ok, "I": ctx, "void": void}


def _t0() -> dict:
    t = B2.t0_checked()
    t["ok"] = bool(t.get("ok")) and t.get("n") == T0_N
    return t


def run_all(runs_root: Path, threads: int, seeds=SEEDS) -> Exit:
    torch.set_num_threads(threads)
    t_start = time.time()
    t0_start = _t0()
    if not t0_start["ok"]:
        return did_not_run(f"T0 at start: {t0_start}")
    source = LR.default_source()
    from ledger import Ledger

    led = Ledger(RUN_ID, question=QUESTION, runs_root=runs_root)
    led.run_meta(device="cpu", steps_requested=K)
    led.manifest(
        {
            "run_id": RUN_ID,
            "prereg": PREREG,
            "prereg_commit": PREREG_COMMIT,
            "expected": EXPECTED,
            "falsifier": FALSIFIER,
            "source": str(source),
            "checkpoint": CHECKPOINT,
            "ckpt_sha256": {s: B2.CKPT_SHA256[(CHECKPOINT, s)] for s in seeds},
            "seeds": list(seeds),
            "gamma": GAMMA,
            "rowset": "C",
            "beta_sweep": BETA_SWEEP,
            "epsilons": EPSILONS,
            "arms": ARMS,
            "controls": {
                CONTROL_P: "decoupled_wd with lr = beta * lr_phi, loss unscaled",
                CONTROL_N: "decoupled_wd eps 1e-8 beta 1, re-run",
                CONTROL_I: (
                    f"decoupled_wd eps 1e-8 beta 1, init seed {CONTEXT_INIT_BASE}+s"
                ),
            },
            "adam_betas": ADAM_BETAS,
            "wd": WD,
            "l2_CHOSEN": L2,
            "phi_clip_CHOSEN": PHI_CLIP,
            "joint_clip": JOINT_CLIP,
            "base_lr": BASE_LR,
            "base_width": BASE_WIDTH,
            "steps": K,
            "snaps": SNAPS,
            "batch": BATCH,
            "train_docs": [TRAIN_LO, TRAIN_LO + N_TRAIN],
            "argmin_docs": [ARGMIN_LO, ARGMIN_LO + N_ARGMIN],
            "init_seed_base": INIT_SEED_BASE,
            "order_seed_base": ORDER_SEED_BASE,
            "g_T_steps": G_T_STEPS,
            "thresholds": {
                "inert_agree": INERT_AGREE,
                "inert_rel": INERT_REL,
                "live_agree": LIVE_AGREE,
                "live_rel": LIVE_REL,
            },
            "threads": threads,
            "torch": torch.__version__,
        }
    )
    all_runs: dict = {}
    data: dict = {}
    done = []
    try:
        for s in seeds:
            out = seed_runs(s, source)
            all_runs.update(out["runs"])
            data[s] = out["data"]
            done.append(s)
    except ControlFailure as e:
        led.status("did_not_run")
        led.note("control_failure", str(e), how="seed_runs raised ControlFailure")
        led.command([sys.executable, EXPERIMENT, "run"], exit_code=int(Exit.DID_NOT_RUN))
        led.write()
        return did_not_run(f"control: {e}")
    agg = aggregate(all_runs, done)
    t0_end = _t0()
    if not t0_end["ok"]:
        agg["void"].append(f"T0 at end: {t0_end}")

    led.run_meta(seeds_actually_run=done, steps_done=K)
    led.note("T0.start", t0_start, how="B2.t0_checked() before loading any model")
    led.note("T0.end", t0_end, how="B2.t0_checked() after every run")
    for s in done:
        led.note(
            f"data.seed{s}", data[s], how="seed_runs: capture controls, G/norm scales"
        )
    for (s, tag, eps, beta), r in sorted(all_runs.items(), key=lambda kv: str(kv[0])):
        led.note(f"run.seed{s}.{_key(tag, eps, beta)}", _run_summary(r), how="train_one")
    for k, v in agg["per_cell"].items():
        led.note(f"cmp.{k}", v, how="compare() against beta=1 of the same seed/arm/eps")
    for k, v in agg["classes"].items():
        led.note(f"class.{k}", v, how="classify() over beta in {0.01, 0.1} x seeds (§7)")
        led.stat(f"agreement.{k}", v["agreements"], how="per (seed, beta) at step 3000")
        led.stat(f"rel.{k}", v["rels"], how="per (seed, beta) at step 3000")
    led.note(
        "N_ok", agg["N_ok"], how="N vs decoupled_wd eps1e-8 beta1: max_abs==0, agree==1"
    )
    led.note(
        "I_context", agg["I"], how="init seed 1100+s vs 1000+s (context, not gating)"
    )
    led.note("void", agg["void"], how="PREREG §8")
    led.note(
        "peak_rss_gb",
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
        how="rusage",
    )
    led.note("wall_seconds", time.time() - t_start, how="time.time()")
    payload = {
        "phi": {str(k): r["phi"] for k, r in all_runs.items()},
        "argmin": {str(k): r["argmin"] for k, r in all_runs.items()},
    }
    torch.save(payload, runs_root / RUN_ID / "phi_argmin.pt")

    a1 = agg["classes"]["decoupled_wd.eps1e-08"]["class"]
    if agg["void"]:
        led.status("failed")
        led.verdict(
            falsifier=FALSIFIER,
            outcome="inconclusive",
            detail=f"VOID: {agg['void']}",
        )
        rc = Exit.DID_NOT_RUN
    else:
        led.status("ok")
        outcome = {"LIVE": "falsified", "INERT": "survived"}.get(a1, "inconclusive")
        led.verdict(
            falsifier=FALSIFIER,
            outcome=outcome,
            detail="; ".join(f"{k} {v['class']}" for k, v in agg["classes"].items()),
        )
        rc = Exit.OK
    led.command(
        [sys.executable, EXPERIMENT, "run", "--threads", str(threads)], exit_code=int(rc)
    )
    led.write()
    _log(f"wrote {led.path}; void={agg['void']}; rc={int(rc)}")
    return rc


def check() -> Exit:
    t = _t0()
    if not t["ok"]:
        return did_not_run(f"T0: {t}")
    source = LR.default_source()
    for s in SEEDS:
        ck = LR.ckpt_path(source, CHECKPOINT, s)
        if LR.sha256(ck) != B2.CKPT_SHA256[(CHECKPOINT, s)]:
            return did_not_run(f"checkpoint {ck} missing or mismatched")
        try:
            read_g_T(source / "B" / f"seed{s}" / "heartbeat.jsonl")
        except (ControlFailure, OSError) as e:
            return did_not_run(f"g_T: {e}")
    print(json.dumps({"T0": t, "source": str(source)}))
    return Exit.OK


def main(argv: list[str] | None = None) -> Exit:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("cmd", choices=["check", "run"])
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    a = ap.parse_args(argv)
    if a.cmd == "check":
        return status(check())
    return status(run_all(a.runs_root, a.threads))


if __name__ == "__main__":
    run_main(main)
