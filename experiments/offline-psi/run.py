"""Offline psi_hat on uncensored LOO returns: step 2 after RSR run 2's FAILS.

PREREG `experiments/offline-psi/PREREG.md`, committed ALONE at c200a08 (owner go
2026-10-08, PREREG §12). The PREREG wins over anything here.

Run 1's harness, run 2's harness, step 1's analysis, lookahead-room, b0-ceilings and
the gamma sweep's ``ageshape`` are reused **by import, never copied** (each loaded as a
private module instance). What this file adds:

* PREREG §1 the gestalt forward: one eval-mode ``no_grad`` forward per U document
  under FIFO memory on the frozen substrate, recording each written gestalt ``s_i``
  and each step's ``c_t`` (`GestaltRecorder`, `gestalt_forward`);
* PREREG §2-§3 the offline fit of run 2's ``BilinearValueHead`` on B0-CF's
  ``G = discounted_returns(Delta, 0.9)`` over the cf_all / cf_live cells, run 2's
  ``L_MC`` form and phi optimizer, U-val selection (`fit`, `select_step`);
* PREREG §5 the fit reading on U-val (model-free replay, B0-CF's cap and rule);
* PREREG §4 the deployment of the selected phi on run 2's EVAL through run 1's
  held-out ``run_batch``, and run 2's WIN / LOSS / EQUIV / UNRESOLVED label;
* PREREG §6 the secondary readings and PREREG §9's controls C1-C9.

Subcommands: ``preflight``, ``manifest``, ``measure-seed``, ``finalize``, ``time``
(PREREG §10's build-time measurements, EVAL never read) and ``recompute``
(claims.json).

Never: backpropagates into the transformer or ``W_sent`` (the gestalts are
precomputed tensors; only phi is in the optimizer, C4); gives psi_hat an age input;
lets ``b`` into anything (``b_enabled=False``); reads EVAL before the selected phi is
frozen (C9); fills §15; makes a scaling claim.

Exit codes (`rsr.exit_codes`): 0 every control passed; 3 did not run, a control
failed (numbers written, the labels marked not read), or a smoke (never the run).
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import math
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rsr.baselines.fifo import FIFOPolicy  # noqa: E402
from rsr.exit_codes import ArgumentParser, Exit, run_main, status  # noqa: E402
from rsr.model.tg.policy_loop import run_policy_loop  # noqa: E402
from rsr.mup.param_groups import VALUE_HEAD_GROUP, build_param_groups  # noqa: E402
from rsr.train import loop  # noqa: E402

EXPERIMENT = "experiments/offline-psi/run.py"
LAUNCHER = "experiments/offline-psi/launch.zsh"
RUN_ID = "offline-psi"
PREREG = "experiments/offline-psi/PREREG.md"
PREREG_COMMIT = "c200a08"


def _load(name: str, rel: str):
    """Import an experiment script by path, as its own module instance. Never copied."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


#: Run 1's harness (held-out procedure, bootstrap, buckets, SUBSTRATE pins).
R1 = _load("_offline_psi_r1", "experiments/rsr-run1/run.py")
#: Run 2's harness (§2 label and class, §5 age-1 share and newcomer label, the probe
#: wrapper, the ES source control C14, `_tg`). Its own run-1 instance is private to it.
R2 = _load("_offline_psi_r2", "experiments/rsr-run2/run.py")
#: Step 1 (`run2-vs-random`, 6ed6583): `minus_random`, `hit_counts`,
#: `conditional_counts`, `eviction_cost`, `mean_cost`.
STEP1 = _load("_offline_psi_step1", "experiments/run2-vs-random/run.py")
#: lookahead-room: `documents`, `discounted_returns`, `TargetPolicy`, `FactFiller`.
LR = _load("_offline_psi_lr", "experiments/lookahead-room/run.py")
#: The gamma sweep's age-shape module: `spearman_rows`, `pooled_median_ci`; through it
#: the gamma sweep (`label`, B0-CF's fit rule) and b0-ceilings (`pairs_of`, `rate`,
#: `pooled`, `cap_point`, `boot_rule`, `boot_weights`).
AS = _load("_offline_psi_ageshape", "experiments/gamma-ceiling/ageshape.py")
GC, B0 = AS.GC, AS.B0

# --------------------------------------------------------------------------- #
# 🔒 PREREG, transcribed. Changing any of these is changing the PREREG.
# --------------------------------------------------------------------------- #

SEEDS = (0, 1, 2)
M = R1.M_SLOTS  # 16
S = R1.S_STEPS  # 48
D_MODEL = R1.D_MODEL  # 128
#: PREREG §2: B0-CF's primary gamma, which is C3's gamma. A construction parameter of
#: the offline target (as run 1's ORACLE_GAMMA), never a registry read.
TARGET_GAMMA = 0.9
ARMS = ("cf_all", "cf_live")  # PREREG §2
PRIMARY = "cf_all"
FIT_BATCH = 16  # PREREG §3
FIT_STEPS = 3000  # PREREG §3: "up to 3000 optimizer steps, the same count as run 2"
VAL_EVERY = 50  # PREREG §3
VAL_MOD = 7  # PREREG §1: U-val = U-order positions p % 7 == 0
N_U = 1088
N_U_VAL = 156
EVAL = R1.EVAL  # (995000, 997048), run 2's
N_EVAL = EVAL[1] - EVAL[0]
RUN2_POOL = R1.POOL  # (993000, 993256)
B2_FIT_VAL = R1.USED_RANGES["B2_FIT_VAL"]  # (936000, 937024)
DELTA_FILE = "B-ckpt3000-seed{s}.delta.pt"
CKPT = 3000
#: PREREG §1, each `delta.pt`'s sha256 (C1; preflight checks each appears verbatim).
DELTA_SHA = {
    0: "1c047a363a8928c18fd017329b8aa6171debfb1046fb0c96e258d20364c87d72",
    1: "bf2753cef826f8d3c1647213d232c65a1e2167160be1a6c262fbea132b838700",
    2: "c880a2413fa2a4840724c7c005778df8cda77932f9db0fb6081db495514ce9c2",
}
#: PREREG §9 C2: B0-CF's ledger `B.ckpt3000.cf.g09.cap.point`, compared with `==`.
CF_CAP_KEY = "B.ckpt3000.cf.g09.cap.point"
CF_CAP = {0: 0.9161993769470408, 1: 0.9551669316375201, 2: 0.8736576121288696}
#: PREREG §4 delta_seed (C8: machine-read from B2's ledger and compared with `==`).
DELTA_KEY = "ckpt3000.seed{s}.decisions.delta"
DELTA_SEED = {0: 0.03802185560447721, 1: 0.03675691427360389, 2: 0.03185756551141167}
REPORT_BUCKETS = STEP1.REPORT_BUCKETS  # all, gap_eq_1, gap_2_to_M, gap_gt_M
FIT_LABELS = ("FIT-POINTS", "FIT-PARTIAL", "FIT-MISSES")
CONTROLS = ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9")
#: The order every seed runs in. C9: nothing reads EVAL before "select" has frozen phi.
PIPELINE = (
    "controls",
    "gestalts",
    "fit",
    "select",
    "fit_reading",
    "deploy",
    "score",
)

QUESTION = (
    "Trained offline on B0-CF's saved resample-knockout Delta (gamma 0.9 returns, never "
    "cut at eviction), can run 2's content-only bilinear psi_hat reproduce the "
    "hindsight ranking (cap on U-val) and beat FIFO when deployed frozen on run 2's "
    "EVAL?"
)
FALSIFIER = (
    "NONE: exploratory after FAILS (Tier D). One deploy label per seed (run 2's rule) "
    "and one fit label (B0-CF's rule); neither is a spec falsifier (PREREG front "
    "matter). The pair routes an owner decision (§8)."
)
EXPECTED = (
    "PREREG §11: fit FIT-PARTIAL most likely (cf_live fits its cells better, "
    "extrapolates worse); deploy MIXED or FAILS more likely than BEATS-FIFO; age-1 "
    "share well below run 2's (REDUCED or RESOLVED)."
)


class ControlFailure(RuntimeError):
    """A PREREG §9 control failed: inconclusive, exit 3."""


class Refused(RuntimeError):
    """The harness refuses to start: exit 3, nothing run."""


@dataclass(frozen=True)
class Refs:
    """Where this run reads what it did not produce. Tests and the smoke repoint it."""

    delta_dir: str = f"{R1._R}/.worktrees/cf-ceiling/runs/cf-ceiling/raw"
    cf_ledger: str = R2.REFS.cf_ledger
    b2_ledger: str = R2.REFS.b2_ledger
    run2_out: str = f"{R1._R}/.worktrees/rsr-run2-run/runs/rsr-run2"
    es_out: str = R2.REFS.es_out
    es_pinned: str = R2.REFS.es_pinned
    prereg: str = str(ROOT / PREREG)

    def r2(self):
        """Run 2's `Refs`, for its C14 (ES source) control."""
        return replace(R2.REFS, es_out=self.es_out, es_pinned=self.es_pinned)


REFS = Refs()


@dataclass(frozen=True)
class Plan:
    """What one seed runs. `plan()` builds the run's; only `--smoke` changes it."""

    fit_steps: int = FIT_STEPS
    val_every: int = VAL_EVERY
    u_limit: int | None = None  # smoke: the first N U documents
    standin: int | None = None  # smoke: N U-val documents stand in for EVAL
    n_boot: int = R1.N_BOOT
    smoke: bool = False


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def sha256_file(p) -> str:
    return R1.sha256_file(Path(p))


def ledger_rows(path: str) -> dict[str, Any]:
    return {r["key"]: r for r in json.loads(Path(path).read_text())["rows"]}


# --------------------------------------------------------------------------- #
# PREREG text checks (preflight, C1, C2, C8)
# --------------------------------------------------------------------------- #


def prereg_values(text: str) -> dict:
    """The numbers the PREREG states, read from its committed text."""
    sha = dict((int(s), h) for s, h in re.findall(r"seed (\d): `([0-9a-f]{64})`", text))
    m = re.search(r"The values are\s+([0-9.]+)\s*/\s*([0-9.]+)\s*/\s*([0-9.]+)", text)
    delta = {} if m is None else {s: float(m.group(s + 1)) for s in SEEDS}
    m = re.search(r"exactly\s+\(([0-9.]+)\s*/\s*([0-9.]+)\s*/\s*([0-9.]+)\)", text)
    cap = {} if m is None else {s: float(m.group(s + 1)) for s in SEEDS}
    return {"delta_sha": sha, "delta_seed": delta, "cf_cap": cap}


def prereg_check(refs: Refs = REFS) -> dict:
    """Every PREREG number this file transcribes equals the committed text's."""
    got = prereg_values(Path(refs.prereg).read_text())
    out = {
        "delta_sha": got["delta_sha"] == DELTA_SHA,
        "delta_seed": got["delta_seed"] == DELTA_SEED,
        "cf_cap": got["cf_cap"] == CF_CAP,
    }
    out["ok"] = all(out.values())
    return out


# --------------------------------------------------------------------------- #
# §1 documents, the U split, disjointness (C3), Delta (C1), the target (§2)
# --------------------------------------------------------------------------- #


def u_documents(seed: int) -> tuple[list, dict]:
    """U in lookahead-room's order (ids [64, 1088) then [4096, 4160)) and its map."""
    docs, vmap, _V, closure = LR.documents(seed)
    if not closure["ok"]:
        raise ControlFailure(f"U is not closed under the vocabulary: {closure}")
    return list(docs), vmap


def u_split(n: int) -> tuple[list[int], list[int]]:
    """PREREG §1: U-val = the U-order positions p with p % 7 == 0; U-train the rest."""
    val = [p for p in range(n) if p % VAL_MOD == 0]
    train = [p for p in range(n) if p % VAL_MOD != 0]
    return train, val


def c3_disjoint(train_ids, val_ids) -> dict:
    """C3: U-train, U-val, EVAL, run 2's pool and B2's FIT_VAL share no id."""
    sets = {
        "U_train": set(train_ids),
        "U_val": set(val_ids),
        "EVAL": set(range(*EVAL)),
        "run2_pool": set(range(*RUN2_POOL)),
        "B2_FIT_VAL": set(range(*B2_FIT_VAL)),
    }
    names = list(sets)
    overlaps = {
        f"{a}&{b}": len(sets[a] & sets[b])
        for i, a in enumerate(names)
        for b in names[i + 1 :]
        if sets[a] & sets[b]
    }
    return {
        "sizes": {k: len(v) for k, v in sets.items()},
        "overlaps": overlaps,
        "ok": not overlaps,
    }


def delta_path(seed: int, refs: Refs = REFS) -> Path:
    return Path(refs.delta_dir) / DELTA_FILE.format(s=seed)


def load_delta(seed: int, refs: Refs = REFS) -> tuple[dict, str]:
    """B0-CF's saved Delta, read (never written): the blob and its sha256."""
    p = delta_path(seed, refs)
    data = p.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    import io

    return torch.load(io.BytesIO(data), weights_only=False), sha


def c1_delta(blob: dict, sha: str, seed: int, ids: list[int], s: int = S) -> dict:
    """C1: sha256 pinned; every U id present in U order; ``[48, 48]``; finite strictly
    below the diagonal and NaN on and above it."""
    out: dict[str, Any] = {"sha256": sha, "registered": DELTA_SHA.get(seed)}
    out["sha_ok"] = sha == DELTA_SHA.get(seed)
    Dm = blob.get("Delta") or {}
    out["n_ids"] = len(Dm)
    out["ids_ok"] = list(Dm) == list(ids)
    below = torch.ones(s, s, dtype=torch.bool).tril(-1)
    bad = []
    for d in ids:
        x = Dm.get(d)
        if x is None or tuple(x.shape) != (s, s):
            bad.append(d)
            continue
        if not bool(torch.isfinite(x[below]).all()) or not bool(
            torch.isnan(x[~below]).all()
        ):
            bad.append(d)
    out["bad_docs"] = bad[:20]
    out["n_bad"] = len(bad)
    out["ok"] = out["sha_ok"] and out["ids_ok"] and not bad
    return out


def target_of(D: torch.Tensor, gamma: float = TARGET_GAMMA) -> torch.Tensor:
    """PREREG §2: ``G[t][i] = sum_k gamma^k Delta[t+k][i]``, lookahead-room's
    `discounted_returns`, imported. Never cut at eviction; truncated at the end."""
    return LR.discounted_returns(D, gamma)


def cell_mask(arm: str, s: int = S, m: int = M) -> torch.Tensor:
    """PREREG §2: ``[t, i]`` regression cells. cf_all: every ``i < t`` (1 <= t < S);
    cf_live: ``t - m <= i < t``, the cells FIFO's memory holds."""
    t = torch.arange(s).unsqueeze(1)
    i = torch.arange(s).unsqueeze(0)
    if arm == "cf_all":
        return i < t
    if arm == "cf_live":
        return (i < t) & (i >= t - m)
    raise ValueError(f"unknown arm {arm!r}; the arms are {ARMS}")


# --------------------------------------------------------------------------- #
# §1 the gestalt forward (C5)
# --------------------------------------------------------------------------- #


class GestaltRecorder:
    """FIFO for the gestalt forward, recording ``c_t`` (the decision context the eval
    forward hands the policy, ADR-0009 L2), the in-loop memory at each full step, and
    the gestalt each write puts in its slot. Decisions are FIFO's, unchanged."""

    name = "fifo+gestalts"
    learns = False
    needs_eval_context = True  # run_policy_loop's eval_capture: c_t is eval_forward's

    def __init__(self) -> None:
        self.inner = FIFOPolicy()
        self._clear()

    def _clear(self) -> None:
        self.ctx: dict[tuple[int, int], torch.Tensor] = {}
        self.full_mem: dict[tuple[int, int], list[int]] = {}
        self.written: dict[tuple[int, int], torch.Tensor] = {}
        self.after: dict[tuple[int, int], list[int]] = {}

    def to(self, device):
        return self

    def reset(self) -> None:
        self._clear()
        self.inner.reset()

    def select_eviction(self, slots, context, step: int) -> int:
        key = (int(slots.row), int(step))
        self.ctx[key] = context.detach().clone()
        self.full_mem[key] = sorted(int(x) for x in slots.written_at[slots.live].tolist())
        return int(self.inner.select_eviction(slots, context, step))

    def observe(self, slots, attn, step: int) -> None:
        raise RuntimeError("observe is off in the gestalt forward (PREREG §1)")

    def on_write(self, slots, slot: int, step: int) -> None:
        key = (int(slots.row), int(step))
        self.written[key] = slots.gestalts[slot].detach().clone()
        self.after[key] = sorted(int(x) for x in slots.written_at[slots.live].tolist())
        self.inner.on_write(slots, slot, step)


def gestalt_forward(
    model, docs: list, vmap: dict, *, batch: int = R1.EVAL_BATCH, m: int = M
) -> dict:
    """PREREG §1: one eval-mode, ``no_grad`` forward per document under FIFO memory.

    -> ``{"s": [N, S, d], "c": [N, S, d], "c5": {...}, "seconds"}``. ``s[n, i]`` is the
    gestalt sentence ``i``'s write put in memory; ``c[n, t]`` is the step-``t``
    sentence gestalt from the forward (eval mode, so the eval forward's: checked
    bit-equal to the decision context at every full step)."""
    n_s, L = S, int(model.cfg.max_sentence_tokens)
    d = int(model.cfg.D)
    S_out = torch.zeros(len(docs), n_s, d)
    C_out = torch.zeros(len(docs), n_s, d)
    problems: list[str] = []
    n_ctx_checked = 0
    t0 = time.perf_counter()
    was = model.training
    model.eval()
    try:
        for i0 in range(0, len(docs), batch):
            chunk = docs[i0 : i0 + batch]
            B = len(chunk)
            ids, mask = loop.encode(chunk, vmap, max_tokens=L, steps=n_s)
            srep = torch.zeros(B, n_s, d)

            def step_fn(t, out, ids_t, mask_t, row_valid, srep=srep):
                srep[:, t] = out.srep.detach()
                return torch.zeros(())

            rec = GestaltRecorder()
            with torch.no_grad():
                run_policy_loop(
                    model,
                    ids,
                    mask,
                    torch.full((B,), n_s),
                    rec,
                    step_fn=step_fn,
                    observe=False,
                    eval_capture=True,
                )
            for b, doc in enumerate(chunk):
                for t in range(n_s):
                    w = rec.written.get((b, t))
                    if w is None:
                        problems.append(f"doc {doc.doc_id}: no write at step {t}")
                        continue
                    S_out[i0 + b, t] = w
                    if not torch.equal(w, srep[b, t]):
                        problems.append(f"doc {doc.doc_id} t={t}: s_t != srep_t")
                    want = list(range(max(0, t - m + 1), t + 1))
                    if rec.after.get((b, t)) != want:
                        problems.append(f"doc {doc.doc_id} t={t}: memory after write")
                    if t >= m:
                        if rec.full_mem.get((b, t)) != list(range(t - m, t)):
                            problems.append(f"doc {doc.doc_id} t={t}: full memory")
                        ctx = rec.ctx.get((b, t))
                        n_ctx_checked += 1
                        if ctx is None or not torch.equal(ctx, srep[b, t]):
                            problems.append(f"doc {doc.doc_id} t={t}: c_t != context")
                C_out[i0 + b] = srep[b]
    finally:
        model.train(was)
    c5 = {
        "n_documents": len(docs),
        "full_steps_checked": n_ctx_checked,
        "problems": problems[:20],
        "n_problems": len(problems),
        "ok": not problems and len(docs) > 0,
    }
    return {"s": S_out, "c": C_out, "c5": c5, "seconds": time.perf_counter() - t0}


# --------------------------------------------------------------------------- #
# §3 the fit
# --------------------------------------------------------------------------- #


def new_policy(seed: int, overrides: dict, steps_per_epoch: float, **kw):
    """Run 2's psiC construction (`train()`'s path): ``build_policy("rsr")`` with phi
    from ``phi_generator(seed)`` -- so seed ``s`` starts from run 2's initial phi."""
    gen, phi_seed = loop.phi_generator(seed)
    pol = loop.build_policy(
        "rsr",
        d_model=D_MODEL,
        steps_per_epoch=steps_per_epoch,
        generator=gen,
        overrides=dict(overrides),
        **kw,
    )
    return pol, phi_seed


def phi_optimizer(frozen_model, head, lr: float):
    """Run 2's phi optimizer as `rsr.train.loop` builds it: muP groups (the frozen
    transformer contributes none), AdamW, decoupled decay, betas and eps from loop."""
    groups = build_param_groups(
        frozen_model, head, base_lr=lr, d_model=D_MODEL, base_width=128
    )
    opt = torch.optim.AdamW(
        groups,
        betas=loop.ADAMW_BETAS,
        weight_decay=loop.ADAMW_WEIGHT_DECAY,
        eps=loop.ADAMW_EPS,
    )
    return opt


def optimizer_holds_only_phi(opt, head) -> dict:
    """C4 (part): the optimizer's parameters are exactly phi = {W, u}."""
    names = [g.get("name") for g in opt.param_groups]
    in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
    phi = {id(p) for p in head.parameters()}
    return {
        "groups": names,
        "n_params": len(in_opt),
        "ok": names == [VALUE_HEAD_GROUP] and in_opt == phi,
    }


def psi_matrix(head, Sg: torch.Tensor, Cg: torch.Tensor) -> torch.Tensor:
    """``[B, S, d]`` gestalts and contexts -> ``[B, S(t), S(i)]`` with
    ``P[b, t, i] = psi_hat(s_i, c_t)``, through the head's own forward."""
    B, n, d = Sg.shape
    g = Sg.unsqueeze(1).expand(B, n, n, d).reshape(B * n, n, d)
    return head(g, Cg.reshape(B * n, d)).reshape(B, n, n)


def l_mc(
    P: torch.Tensor, G: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """Run 2's L_MC form (ADR-0009 L9): squared error summed over the cells of each
    (document, t), averaged over the (document, t) pairs holding at least one cell.
    -> (loss, n_groups). ``G`` must be finite wherever ``mask`` is set."""
    err = torch.where(mask, (P - G).pow(2), torch.zeros_like(P))
    groups = int(mask.any(-1).sum())
    return err.sum() / groups, groups


def fit_tensors(blob: dict, docs: list, arm: str) -> dict:
    """``G`` (float32, 0 off-cell) and the arm's cell mask per U document."""
    Gs, masks = [], []
    cm = cell_mask(arm)
    for doc in docs:
        G = target_of(blob["Delta"][doc.doc_id])
        ok = ~torch.isnan(G)
        Gs.append(torch.nan_to_num(G, nan=0.0).to(torch.float32))
        masks.append(cm & ok)
    return {"G": torch.stack(Gs), "mask": torch.stack(masks)}


def val_loss(
    head,
    S_: torch.Tensor,
    C_: torch.Tensor,
    G: torch.Tensor,
    mask: torch.Tensor,
    pos: list[int],
    chunk: int = 16,
) -> float:
    """U-val L_MC over ``pos``: sum over cells / (document, t) groups, all at once."""
    tot, groups = 0.0, 0
    with torch.no_grad():
        for j in range(0, len(pos), chunk):
            p = torch.tensor(pos[j : j + chunk], dtype=torch.long)
            P = psi_matrix(head, S_[p], C_[p])
            err = torch.where(mask[p], (P - G[p]).pow(2), torch.zeros_like(P))
            tot += float(err.double().sum())
            groups += int(mask[p].any(-1).sum())
    return tot / groups if groups else float("nan")


def select_step(history: list[dict]) -> dict | None:
    """PREREG §3: the lowest U-val L_MC, ties to the earlier step. Non-finite losses are
    never selected. Reads ``history`` only (U-val; C9)."""
    best = None
    for h in sorted(history, key=lambda r: r["step"]):
        v = h["val_l_mc"]
        if not _finite(v):
            continue
        if best is None or v < best["val_l_mc"]:
            best = h
    return best


def fit(
    head,
    opt,
    data: dict,
    train_pos: list[int],
    val_pos: list[int],
    *,
    seed: int,
    steps: int,
    every: int,
    batch: int = FIT_BATCH,
) -> dict:
    """PREREG §3: batches of ``batch`` U-train documents in permutation epochs from a
    private generator seeded ``seed`` (`loop.PermutationEpochs`, imported); U-val L_MC
    every ``every`` steps; the selected state is the lowest U-val L_MC, ties earliest.

    ``data``: ``s``, ``c`` ``[N, S, d]`` (precomputed tensors: no path to the
    transformer exists), ``G`` ``[N, S, S]``, ``mask``. -> history, the selected step
    and its phi state, C7's per-step finiteness."""
    S_, C_, G, mask = data["s"], data["c"], data["G"], data["mask"]
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    sampler = loop.PermutationEpochs(len(train_pos), gen, "cpu")
    tp = torch.tensor(train_pos, dtype=torch.long)
    params = list(head.parameters())
    history: list[dict] = []
    bad: list[int] = []
    kept: dict | None = None
    t0 = time.perf_counter()
    val0 = val_loss(head, S_, C_, G, mask, val_pos)
    for k in range(1, steps + 1):
        pos = tp[sampler.take(batch)]
        P = psi_matrix(head, S_[pos], C_[pos])
        loss, _groups = l_mc(P, G[pos], mask[pos])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = loop._grad_norm(params)
        lv = float(loss.detach())
        if not (_finite(lv) and _finite(gn)):
            bad.append(k)
        opt.step()
        if k % every == 0:
            v = val_loss(head, S_, C_, G, mask, val_pos)
            row = {"step": k, "val_l_mc": v, "train_l_mc": lv, "grad_norm": gn}
            history.append(row)
            if _finite(v) and (kept is None or v < kept["val_l_mc"]):
                kept = {
                    "step": k,
                    "val_l_mc": v,
                    "state": {
                        n: x.detach().clone() for n, x in head.state_dict().items()
                    },
                }
    sel = select_step(history)
    if (sel is None) != (kept is None) or (
        sel is not None and sel["step"] != kept["step"]
    ):
        raise ControlFailure(
            f"fit: kept state {kept and kept['step']} != selection {sel}"
        )
    return {
        "history": history,
        "val_l_mc_step0": val0,
        "selected": None
        if sel is None
        else {"step": sel["step"], "val_l_mc": sel["val_l_mc"]},
        "state": None if kept is None else kept["state"],
        "c7_bad_steps": bad[:20],
        "c7_n_bad": len(bad),
        "epochs_drawn": sampler.epochs_drawn,
        "seconds": time.perf_counter() - t0,
        "steps": steps,
    }


def state_sha(state: dict) -> str:
    """sha256 over phi's tensors, in name order (the frozen selected phi's identity)."""
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode())
        h.update(state[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# §5 the fit reading (U-val, model-free)
# --------------------------------------------------------------------------- #


def score_matrices(head, S_: torch.Tensor, C_: torch.Tensor) -> torch.Tensor:
    """``[N, S, S]`` float64 ``psi_hat(s_i, c_t)``, NaN at ``i >= t`` (no sentence
    ``i >= t`` is in memory at ``t``)."""
    out = []
    with torch.no_grad():
        for j in range(0, S_.shape[0], 16):
            out.append(psi_matrix(head, S_[j : j + 16], C_[j : j + 16]).double())
    P = torch.cat(out) if out else torch.zeros(0, S, S, dtype=torch.float64)
    below = torch.ones(S, S, dtype=torch.bool).tril(-1)
    return torch.where(below, P, torch.full_like(P, float("nan")))


def replay_rates(docs: list, mats, m: int = M) -> list[tuple[int, int]]:
    """Model-free residency replay (`headroom.simulate`): evict the live slot with the
    smallest ``mat[t][written_at]``, ties to the oldest (lookahead-room's
    `TargetPolicy`). -> per-document (queries, hits), all queries."""
    return [
        B0.rate(B0.pairs_of(d, LR.TargetPolicy(x), m))
        for d, x in zip(docs, mats, strict=True)
    ]


def reference_rates(docs: list, seed: int, m: int = M) -> dict[str, list]:
    """FIFO and B0-CF's FactFiller (seeded ``random.Random(f"ff:{seed}:{doc_id}")``)."""
    return {
        "fifo": [B0.rate(B0.pairs_of(d, FIFOPolicy(), m)) for d in docs],
        "factfiller": [
            B0.rate(
                B0.pairs_of(
                    d, LR.FactFiller(d, random.Random(f"ff:{seed}:{d.doc_id}")), m
                )
            )
            for d in docs
        ],
    }


def c2_reproduce(
    docs: list, blob: dict, seed: int, refs_rates: dict | None = None, m: int = M
) -> dict:
    """C2: ``discounted_returns(Delta, 0.9)`` replayed by `TargetPolicy` on the FULL U
    reproduces B0-CF's cap point exactly (`==` on the float)."""
    rr = refs_rates or reference_rates(docs, seed, m)
    rule = replay_rates(docs, [target_of(blob["Delta"][d.doc_id]) for d in docs], m)
    point = B0.cap_point(
        B0.pooled(rule), B0.pooled(rr["fifo"]), B0.pooled(rr["factfiller"])
    )
    return {
        "point": point,
        "registered": CF_CAP.get(seed),
        "n_documents": len(docs),
        "hit": B0.pooled(rule),
        "rates": rule,
    }


def rho_psi_g(P: torch.Tensor, G: torch.Tensor, W: torch.Tensor, m: int = M) -> dict:
    """PREREG §5: within-step Spearman rho(psi_hat, G) over the FIFO-resident cells
    ``t - m <= i < t`` at each of FIFO's eviction steps (``t >= m``), pooled as the lower
    median with the per-document bootstrap (`spearman_rows`, `pooled_median_ci`)."""
    rhos, idx = [], []
    n = P.shape[1]
    for j in range(P.shape[0]):
        rows_p = torch.stack([P[j, t, t - m : t] for t in range(m, n)])
        rows_g = torch.stack([G[j, t, t - m : t] for t in range(m, n)])
        rhos.append(AS.spearman_rows(rows_p, rows_g))
        idx.append(torch.full((n - m,), j, dtype=torch.long))
    return AS.pooled_median_ci(torch.cat(rhos), torch.cat(idx), W)


def fit_label(caps: dict[int, dict]) -> str:
    """PREREG §5 (B0-CF §7's rule, verbatim, through the gamma sweep's `label`):
    POINTS if every seed's cap CI lower >= 0.80; PARTIAL if not, and every point >=
    0.50; MISSES otherwise."""
    pts = [caps[s]["point"] for s in SEEDS]
    los = [caps[s]["lo"] for s in SEEDS]
    return "FIT-" + GC.label(pts, los)


# --------------------------------------------------------------------------- #
# §4 deployment
# --------------------------------------------------------------------------- #


def deploy_policy(state: dict, seed: int, overrides: dict, steps_per_epoch: float):
    """The frozen selected phi in RSR's eviction argmin with T_warm = 0 (PREREG §4):
    run 2's psiC overrides (b off, nu 0, shadow off) plus the declared no-warmup
    switch, so no FIFO warmup is applied at evaluation."""
    ov = {**overrides, "t_warm": 0.0}
    pol, _ = new_policy(seed, ov, steps_per_epoch, no_warmup_ablation=True)
    pol.head.load_state_dict(state)
    if pol.config.t_warm != 0.0 or pol.config.b_enabled or pol.config.nu != 0.0:
        raise ControlFailure(f"deploy policy config {pol.config}")
    return pol


def deploy_guard(selection: dict | None, phi_file: Path, out_file: Path) -> dict:
    """C9, before anything reads EVAL: a selection exists, the frozen phi on disk is
    the selected one (sha256), and the EVAL output does not exist yet."""
    if selection is None or not selection.get("phi_sha256"):
        raise Refused("no selected phi: EVAL is read only by the frozen selected phi")
    if not Path(phi_file).exists():
        raise Refused(f"{phi_file} is missing")
    got = state_sha(torch.load(phi_file, weights_only=True))
    if got != selection["phi_sha256"]:
        raise Refused(
            f"{phi_file} hashes to {got}, not the selected {selection['phi_sha256']}"
        )
    if Path(out_file).exists():
        raise Refused(f"{out_file} exists: EVAL is read once per seed (C9)")
    return {"phi_sha256": got, "ok": True}


def deploy_pass(model, cfg, docs: list, vmap: dict, make_policy) -> dict:
    """Run 1's held-out procedure: blocks of ``cfg.eval_batch`` documents through
    `R1.run_batch` (eval mode, ``no_grad``, ``eval_forward`` decision context), each
    block a fresh policy wrapped in run 2's `ProbePolicy` (records raw psi_hat and
    ages at every decision; the decision is the inner policy's)."""
    loop.check_vocabulary_closure(docs, vmap)
    recs: list[dict] = []
    decisions: list[tuple] = []
    secs: list[float] = []
    for i in range(0, len(docs), cfg.eval_batch):
        chunk = docs[i : i + cfg.eval_batch]
        pol = R2.ProbePolicy(make_policy())
        t0 = time.perf_counter()
        recs += R1.run_batch(model, pol, chunk, vmap, cfg)
        secs.append(time.perf_counter() - t0)
        decisions += [(i + d[0], *d[1:]) for d in pol.decisions]
    bad = [
        r["doc"]
        for r, d in zip(recs, docs, strict=True)
        if not R1.residency_ok(d, r, cfg.memory_slots)
    ]
    return {
        "records": recs,
        "decisions": decisions,
        "residency": {
            "n_documents": len(recs),
            "mismatched": bad[:20],
            "n_mismatched": len(bad),
        },
        "timing": {
            "batches": len(secs),
            "total_s": sum(secs),
            "first_batch_s": secs[0] if secs else None,
        },
    }


def write_records(path: Path, recs: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    return sha256_file(path)


# --------------------------------------------------------------------------- #
# §4 / §6 scoring (pure; reads records)
# --------------------------------------------------------------------------- #


def pairing(a: list[dict], b: list[dict]) -> dict:
    """C6: the same ``(doc, [(t, gap)])`` sequence, for every document; the lengths are
    asserted equal (not a prefix; run 2 review F5)."""
    return {
        "n": [len(a), len(b)],
        "ok": len(a) == len(b) and len(a) > 0 and R2._tg(a) == R2._tg(b),
    }


def headline(cf: list[dict], fifo: list[dict], W: torch.Tensor, delta) -> dict:
    """PREREG §4: acc.all(cf_all) - acc.all(fifo), run 1's paired bootstrap, run 2's
    per-seed label against delta_seed."""
    b = R1.buckets_for(M)["all"]
    per = {"cf_all": R1.answer_counts(cf, b), "fifo": R1.answer_counts(fifo, b)}
    return R2.label_contrast(R1.contrast(per, "cf_all", "fifo", W), delta)


def deploy_class(labels: dict[int, str]) -> str:
    """PREREG §4: BEATS-FIFO iff WIN on all 3 seeds; FAILS iff LOSS on any; MIXED."""
    return R2.run_class(labels, SEEDS)


def rho_psi_age(decisions: list[tuple], W: torch.Tensor) -> dict:
    """PREREG §6 item 6: within-step Spearman rho(psi_hat, age) at every EVAL decision,
    pooled as the lower median with the per-document bootstrap (descriptive)."""
    if not decisions:
        return {"n_steps": 0}
    P = torch.stack([d[2] for d in decisions])
    A = torch.stack([d[4] for d in decisions]).to(torch.float64)
    idx = torch.tensor([d[0] for d in decisions], dtype=torch.long)
    return AS.pooled_median_ci(AS.spearman_rows(P, A), idx, W)


def secondary(
    recs: dict[str, list[dict]], W: torch.Tensor, es: list[dict] | None
) -> dict:
    """PREREG §6 items 1-4 (descriptive, unadjusted): cf_all - random (step 1's
    definition), - psiC(run 2), - pinned ES when its scoped controls pass; by gap
    bucket, for accuracy and residency; accuracy given residency; eviction-age cost."""
    buckets = R1.buckets_for(M)
    out: dict[str, Any] = {"accuracy": {}, "residency": {}}
    names = [n for n in recs]
    for b in REPORT_BUCKETS:
        per = {n: R1.answer_counts(recs[n], buckets[b]) for n in names}
        hp = {n: STEP1.hit_counts(recs[n], buckets[b]) for n in names}
        for kind, src in (("accuracy", per), ("residency", hp)):
            row = {
                "n": int(src["fifo"][:, 0].sum()),
                "levels": {n: R1.level(src[n], W) for n in ("cf_all", "fifo", "psiC")},
                "minus_fifo": R1.contrast(src, "cf_all", "fifo", W),
                "minus_random": STEP1.minus_random(src, "cf_all", W),
                "minus_psiC_run2": R1.contrast(src, "cf_all", "psiC", W),
            }
            if es is not None:
                esp = (
                    R1.answer_counts(es, buckets[b])
                    if kind == "accuracy"
                    else STEP1.hit_counts(es, buckets[b])
                )
                row["minus_es"] = R1.contrast({**src, "es": esp}, "cf_all", "es", W)
            out[kind][b] = row
    out["conditional"] = {}
    for b in ("all", "gap_2_to_M", "gap_gt_M"):
        cc = {n: STEP1.conditional_counts(recs[n], buckets[b]) for n in names}
        rnd = {
            k: sum(cc[n].get(k, 0) for n in STEP1.RANDOM) / len(STEP1.RANDOM)
            for k in ("resident", "resident_right", "missed", "missed_right")
        }
        out["conditional"][b] = {
            "cf_all": cc["cf_all"],
            "fifo": cc["fifo"],
            "psiC": cc["psiC"],
            "random_mean": rnd,
        }
    cost, anomalies = {}, []
    for n in ("cf_all", "psiC", "fifo"):
        cost[n], an = STEP1.eviction_cost(recs[n])
        anomalies += [f"{n}: {x}" for x in an]
    rc = []
    for n in STEP1.RANDOM:
        c, an = STEP1.eviction_cost(recs[n])
        rc.append(c)
        anomalies += [f"{n}: {x}" for x in an]
    cost["random_mean"] = STEP1.mean_cost(rc)
    out["eviction_cost"] = cost
    out["eviction_cost_anomalies"] = anomalies[:20]
    out["n_eviction_cost_anomalies"] = len(anomalies)
    return out


def read_delta_seed(seed: int, refs: Refs = REFS) -> dict:
    """C8: delta_seed machine-read from B2's ledger, `==` PREREG §4's value."""
    out: dict[str, Any] = {"ledger": refs.b2_ledger, "key": DELTA_KEY.format(s=seed)}
    try:
        val = float(ledger_rows(refs.b2_ledger)[out["key"]]["value"]["0.9"])
        out.update(delta=val, registered=DELTA_SEED[seed], ok=val == DELTA_SEED[seed])
    except Exception as e:  # did not run: recorded, never a pass
        out.update(delta=None, ok=False, error=f"{type(e).__name__}: {e}")
    return out


def read_cf_cap(seed: int, refs: Refs = REFS) -> float | None:
    try:
        return float(
            ledger_rows(refs.cf_ledger)[CF_CAP_KEY]["samples"][SEEDS.index(seed)]
        )
    except Exception:
        return None


def c2_control(c2: dict, ledger_value: float | None) -> dict:
    """C2: computed == B0-CF's ledger == PREREG, all as floats with `==`."""
    return {
        "computed": c2["point"],
        "ledger": ledger_value,
        "registered": c2["registered"],
        "n_documents": c2["n_documents"],
        "ok": c2["n_documents"] == N_U
        and c2["point"] == ledger_value
        and ledger_value == c2["registered"],
    }


def c9_order(events: list[str]) -> dict:
    """C9: the pipeline ran in PIPELINE's order, so EVAL ("deploy") came after the
    selection had frozen phi, exactly once."""
    order_ok = [e for e in events if e in PIPELINE] == [
        p for p in PIPELINE if p in events
    ]
    once = events.count("deploy") <= 1
    after = "deploy" not in events or (
        "select" in events and events.index("select") < events.index("deploy")
    )
    return {"events": events, "ok": order_ok and once and after}


def run_pipeline(steps: dict, events: list[str]) -> dict:
    """Run ``steps[name](ctx)`` in PIPELINE's order, logging each. ``ctx`` carries each
    step's return under its name, so "deploy" can only see what "select" froze."""
    ctx: dict[str, Any] = {}
    for name in PIPELINE:
        if name not in steps:
            continue
        events.append(name)
        ctx[name] = steps[name](ctx)
    return ctx


# --------------------------------------------------------------------------- #
# measure-seed
# --------------------------------------------------------------------------- #


def seed_dir(out_root: Path, seed: int) -> Path:
    return Path(out_root) / f"seed{seed}"


def heldout_dir(out_root: Path, seed: int) -> Path:
    return Path(out_root) / "heldout" / f"seed{seed}"


def run2_records(refs: Refs, seed: int, names) -> dict[str, list[dict]]:
    hd = Path(refs.run2_out) / "heldout" / f"seed{seed}"
    return {n: R1.read_pass(hd, n) for n in names}


def load_frozen(cfg) -> tuple[Any, dict, str]:
    """The substrate model, from bytes hashed here (run 1's pin), frozen and in eval."""
    data = Path(cfg.substrate.checkpoint).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if sha != cfg.substrate.sha256:
        raise ControlFailure(f"substrate hashes to {sha}, not {cfg.substrate.sha256}")
    from rsr.train import checkpoint as ck

    payload = ck.load_bytes(data, source=cfg.substrate.checkpoint)
    model = R1.model_from_payload(payload)
    model.requires_grad_(False)  # D2(b): nothing reaches the transformer or W_sent
    model.eval()
    return model, payload, sha


def run2_psic_config(refs: Refs, seed: int) -> dict:
    """Run 2's psiC arm's frozen config (overrides, epoch, lr), from its checkpoint."""
    p = Path(refs.run2_out) / "psiC" / f"seed{seed}" / f"ckpt-{CKPT:06d}.pt"
    payload = R1.read_checkpoint(p)
    return {"path": str(p), "payload": payload, "conf": payload["meta"]["config"]}


def c4_frozen(model, substrate_state: dict, cfg, run2_payloads: dict) -> dict:
    """C4 (part): the substrate file still hashes to its pin; the model's tensors are
    bit-equal to the substrate's (and to run 2's psiC / fifo checkpoints' -- the model
    run 2's FIFO records came from); no transformer tensor has a gradient."""
    now = sha256_file(cfg.substrate.checkpoint)
    st = model.state_dict()
    equal = set(st) == set(substrate_state) and all(
        torch.equal(st[k], substrate_state[k]) for k in substrate_state
    )
    same_run2 = (
        R1.c3_frozen(run2_payloads, substrate_state) if run2_payloads else {"ok": False}
    )
    no_grad = all(p.grad is None and not p.requires_grad for p in model.parameters())
    return {
        "substrate_sha_after": now,
        "substrate_sha_ok": now == cfg.substrate.sha256,
        "state_bit_equal": equal,
        "run2_checkpoints_equal_substrate": same_run2,
        "no_transformer_grad": no_grad,
        "ok": now == cfg.substrate.sha256
        and equal
        and bool(same_run2.get("ok"))
        and no_grad,
    }


def plan_docs(seed: int, plan: Plan) -> tuple[list, dict, list[int], list[int]]:
    docs, vmap = u_documents(seed)
    if plan.u_limit is not None:
        docs = docs[: plan.u_limit]
    train, val = u_split(len(docs))
    return docs, vmap, train, val


def measure_seed(
    seed: int,
    out_root: Path,
    *,
    threads: int,
    plan: Plan | None = None,
    refs: Refs = REFS,
) -> dict:
    """One seed, in PIPELINE's order. Writes ``measure-seed<s>.json``; returns it."""
    plan = Plan() if plan is None else plan
    torch.set_num_threads(int(threads))
    out_root = Path(out_root)
    sd, hd = seed_dir(out_root, seed), heldout_dir(out_root, seed)
    cf_out = hd / f"{PRIMARY}.jsonl.gz"
    if cf_out.exists():
        raise Refused(f"{cf_out} exists: EVAL is read once per seed (C9); never re-run")
    sd.mkdir(parents=True, exist_ok=True)
    cfg = R1.prereg_config(seed, CKPT)
    doc: dict[str, Any] = {
        "run_id": RUN_ID,
        "seed": seed,
        "plan": asdict(plan),
        "refs": asdict(refs),
        "threads": threads,
        "tree": R1.tree_state(),
        "hardware": R1.hardware(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "controls": {},
        "notes": [],
        "timing": {},
    }
    events: list[str] = []
    state: dict[str, Any] = {}

    def controls(_ctx):
        doc["controls"]["prereg_text"] = prereg_check(refs)
        full_docs, vmap = u_documents(seed)
        docs = full_docs if plan.u_limit is None else full_docs[: plan.u_limit]
        train, val = u_split(len(docs))
        state.update(docs=docs, vmap=vmap, train=train, val=val)
        doc["u"] = {"n": len(docs), "n_train": len(train), "n_val": len(val)}
        ids = [d.doc_id for d in docs]
        doc["controls"]["C3"] = c3_disjoint(
            [ids[p] for p in train], [ids[p] for p in val]
        )
        blob, dsha = load_delta(seed, refs)
        state["blob"] = blob
        doc["controls"]["C1"] = c1_delta(blob, dsha, seed, [d.doc_id for d in full_docs])
        t0 = time.perf_counter()
        rr = reference_rates(full_docs, seed)
        c2 = c2_reproduce(full_docs, blob, seed, rr)
        doc["controls"]["C2"] = c2_control(c2, read_cf_cap(seed, refs))
        doc["timing"]["c2_s"] = time.perf_counter() - t0
        pos = {d.doc_id: j for j, d in enumerate(full_docs)}
        state["ref_rates_val"] = {
            k: [v[pos[docs[p].doc_id]] for p in val] for k, v in rr.items()
        }
        doc["controls"]["C8"] = read_delta_seed(seed, refs)
        vm_run1 = R1.vocab_map(cfg)
        doc["vocab"] = {
            "equals_run1_map": vm_run1 == state["vmap"],
            "sha256": loop.vocab_sha256(state["vmap"]),
            "run1_registered": R1.EXPECTED_VOCAB_SHA256,
        }
        if (
            not doc["vocab"]["equals_run1_map"]
            or doc["vocab"]["sha256"] != R1.EXPECTED_VOCAB_SHA256
        ):
            doc["notes"].append("vocabulary map differs from run 1's")
        return None

    def gestalts(_ctx):
        model, _payload, sha = load_frozen(cfg)
        state.update(
            model=model,
            substrate_state={k: v.clone() for k, v in model.state_dict().items()},
        )
        doc["substrate"] = {"checkpoint": cfg.substrate.checkpoint, "sha256_before": sha}
        g = gestalt_forward(model, state["docs"], state["vmap"])
        doc["controls"]["C5"] = g["c5"]
        doc["timing"]["gestalt_forward_s"] = g["seconds"]
        torch.save(
            {"s": g["s"], "c": g["c"], "ids": [d.doc_id for d in state["docs"]]},
            sd / "gestalts.pt",
        )
        doc["gestalts"] = {
            "path": str(sd / "gestalts.pt"),
            "sha256": sha256_file(sd / "gestalts.pt"),
        }
        state.update(S=g["s"], C=g["c"])
        return None

    def fit_all(_ctx):
        r2 = run2_psic_config(refs, seed)
        conf = r2["conf"]
        ov = dict(conf["policy_overrides"])
        spe = float(conf["epoch"]["steps_per_epoch"])
        lr = float(conf["lr"])
        state.update(overrides=ov, spe=spe, run2_payload=r2["payload"])
        doc["run2_psiC_config"] = {
            "path": r2["path"],
            "overrides": ov,
            "steps_per_epoch": spe,
            "lr": lr,
            "seed": conf["seed"],
        }
        if lr != R1.LR or int(conf["seed"]) != seed or ov.get("b_enabled") is not False:
            doc["notes"].append(f"run 2's psiC config is not run 1's: lr {lr}, ov {ov}")
        fits: dict[str, Any] = {}
        for arm in ARMS:
            pol, phi_seed = new_policy(seed, ov, spe)
            head = pol.head
            init_sha = state_sha(head.state_dict())
            opt = phi_optimizer(state["model"], head, lr)
            only = optimizer_holds_only_phi(opt, head)
            data = {
                "s": state["S"],
                "c": state["C"],
                **fit_tensors(state["blob"], state["docs"], arm),
            }
            state[f"data_{arm}"] = data
            r = fit(
                head,
                opt,
                data,
                state["train"],
                state["val"],
                seed=seed,
                steps=plan.fit_steps,
                every=plan.val_every,
            )
            fits[arm] = {
                **{k: v for k, v in r.items() if k != "state"},
                "phi_seed": phi_seed,
                "phi_init_sha256": init_sha,
                "optimizer": only,
                "optimizer_record": loop.optimizer_record(),
            }
            state[f"state_{arm}"] = r["state"]
            state[f"head_{arm}"] = head
        doc["fit"] = fits
        doc["timing"]["fit_s"] = {a: fits[a]["seconds"] for a in ARMS}
        return None

    def select(_ctx):
        sel = {}
        for arm in ARMS:
            st = state[f"state_{arm}"]
            if st is None:
                sel[arm] = None
                continue
            path = sd / f"phi-{arm}.pt"
            torch.save(st, path)
            sel[arm] = {
                **doc["fit"][arm]["selected"],
                "phi_sha256": state_sha(st),
                "path": str(path),
            }
            state[f"head_{arm}"].load_state_dict(st)  # the frozen selected phi
        doc["selection"] = sel
        return sel

    def fit_reading(_ctx):
        docs, val = state["docs"], state["val"]
        vdocs = [docs[p] for p in val]
        vp = torch.tensor(val, dtype=torch.long)
        Wv = B0.boot_weights(len(vdocs), seed, plan.n_boot)
        rr = state["ref_rates_val"]
        Gv = torch.stack([target_of(state["blob"]["Delta"][d.doc_id]) for d in vdocs])
        out: dict[str, Any] = {"n_documents": len(vdocs)}
        ref = B0.boot_rule(
            Wv, replay_rates(vdocs, list(Gv)), rr["fifo"], rr["factfiller"]
        )
        out["reference_G"] = {
            **ref,
            "note": "B0-CF's own G on U-val: beside, never contrasted",
        }
        for arm in ARMS:
            if state[f"state_{arm}"] is None:
                out[arm] = {"omitted": "no finite U-val L_MC: nothing selected"}
                continue
            head = state[f"head_{arm}"]
            P = score_matrices(head, state["S"][vp], state["C"][vp])
            r = B0.boot_rule(
                Wv, replay_rates(vdocs, list(P)), rr["fifo"], rr["factfiller"]
            )
            r["rho_psi_G_fifo_resident"] = rho_psi_g(P, Gv, Wv)
            r["val_l_mc_selected"] = doc["selection"][arm]["val_l_mc"]
            r["role"] = (
                "PRIMARY (fit label)" if arm == PRIMARY else "SECONDARY: diagnostics only"
            )
            out[arm] = r
        doc["fit_reading"] = out
        return None

    def deploy(ctx):
        sel = ctx["select"].get(PRIMARY)
        guard = deploy_guard(sel, sd / f"phi-{PRIMARY}.pt", cf_out)
        state_p = torch.load(sd / f"phi-{PRIMARY}.pt", weights_only=True)
        model = state["model"]
        if plan.standin is not None:
            # SMOKE: U-val documents stand in for EVAL; EVAL is never generated.
            ddocs = [state["docs"][p] for p in state["val"][: plan.standin]]
        else:
            ddocs = R1.docs_by_id(cfg, list(cfg.eval_ids()))

        def make():
            return deploy_policy(state_p, seed, state["overrides"], state["spe"])

        r = deploy_pass(model, cfg, ddocs, R1.vocab_map(cfg), make)
        doc["deploy"] = {
            "guard": guard,
            "n_documents": len(ddocs),
            "doc_range": [ddocs[0].doc_id, ddocs[-1].doc_id] if ddocs else None,
            "standin": plan.standin is not None,
            "residency": r["residency"],
            "timing": r["timing"],
        }
        doc["deploy"]["records_sha256"] = write_records(cf_out, r["records"])
        torch.save(
            {
                "doc": torch.tensor([d[0] for d in r["decisions"]]),
                "step": torch.tensor([d[1] for d in r["decisions"]]),
                "psi": torch.stack([d[2] for d in r["decisions"]])
                if r["decisions"]
                else None,
                "age": torch.stack([d[4] for d in r["decisions"]])
                if r["decisions"]
                else None,
            },
            hd / f"{PRIMARY}.decisions.pt",
        )
        if plan.standin is not None:
            write_standin(model, cfg, ddocs, seed, state["run2_payload"], out_root)
        state.update(cf_recs=r["records"], decisions=r["decisions"])
        doc["timing"]["deploy_s"] = r["timing"]["total_s"]
        return None

    def score(_ctx):
        doc["heldout"] = score_seed(
            state["cf_recs"],
            state["decisions"],
            seed,
            refs if plan.standin is None else standin_refs(refs, out_root),
            plan.n_boot,
            doc["controls"]["C8"].get("delta"),
        )
        return None

    steps = {
        "controls": controls,
        "gestalts": gestalts,
        "fit": fit_all,
        "select": select,
        "fit_reading": fit_reading,
        "deploy": deploy,
        "score": score,
    }
    try:
        run_pipeline(steps, events)
    finally:
        doc["events"] = list(events)
    # Controls assembled after the pipeline (C4, C6, C7, C9).
    run2p = {"run2_psiC": state["run2_payload"]}
    fifo_ck = Path(refs.run2_out) / "fifo" / f"seed{seed}" / f"ckpt-{CKPT:06d}.pt"
    if fifo_ck.exists():
        run2p["run2_fifo"] = R1.read_checkpoint(fifo_ck)
    c4 = c4_frozen(state["model"], state["substrate_state"], cfg, run2p)
    c4["optimizer_only_phi"] = {a: doc["fit"][a]["optimizer"]["ok"] for a in ARMS}
    c4["ok"] = c4["ok"] and all(c4["optimizer_only_phi"].values())
    doc["controls"]["C4"] = c4
    h = doc.get("heldout") or {}
    doc["controls"]["C6"] = h.get("C6", {"ok": False, "error": "no held-out reading"})
    psi_ok = bool(state.get("decisions")) and all(
        bool(torch.isfinite(d[2]).all()) for d in state["decisions"]
    )
    doc["controls"]["C7"] = {
        "fit_bad_steps": {a: doc["fit"][a]["c7_n_bad"] for a in ARMS},
        "psi_finite_every_eval_decision": psi_ok,
        "n_eval_decisions": len(state.get("decisions") or []),
        "ok": psi_ok and all(doc["fit"][a]["c7_n_bad"] == 0 for a in ARMS),
    }
    doc["controls"]["C9"] = {
        **c9_order(events),
        "guard": (doc.get("deploy") or {}).get("guard"),
    }
    doc["controls"]["C9"]["ok"] = doc["controls"]["C9"]["ok"] and bool(
        ((doc.get("deploy") or {}).get("guard") or {}).get("ok")
    )
    doc["controls"]["C1"]["ok"] = (
        doc["controls"]["C1"]["ok"] and doc["controls"]["prereg_text"]["ok"]
    )
    doc["controls"]["C2"]["ok"] = (
        doc["controls"]["C2"]["ok"] and doc["controls"]["prereg_text"]["ok"]
    )
    doc["controls"]["C8"]["ok"] = (
        doc["controls"]["C8"]["ok"] and doc["controls"]["prereg_text"]["ok"]
    )
    doc["controls"]["C5"]["ok"] = (
        doc["controls"]["C5"]["ok"]
        and bool(doc["vocab"]["equals_run1_map"])
        and doc["vocab"]["sha256"] == R1.EXPECTED_VOCAB_SHA256
    )
    doc["controls"]["residency_C7_run1"] = {
        **doc["deploy"]["residency"],
        "ok": doc["deploy"]["residency"]["n_mismatched"] == 0,
    }
    failed = [
        k
        for k, v in doc["controls"].items()
        if k != "prereg_text" and v.get("ok") is not True
    ]
    doc["controls_failed"] = failed
    if plan.smoke:
        doc["notes"].append("SMOKE: never the run (PREREG §10); every label is not read")
    bad = bool(failed) or plan.smoke
    doc["exit"] = int(Exit.DID_NOT_RUN if bad else Exit.OK)
    doc["status"] = "did_not_run" if bad else "ok"
    doc["seeds_actually_run"] = [seed]
    doc["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out_root / f"measure-seed{seed}.json").write_text(
        json.dumps(doc, indent=1, default=str) + "\n"
    )
    return doc


def standin_refs(refs: Refs, out_root: Path) -> Refs:
    """SMOKE: run 2's records are replaced by passes over the stand-in documents."""
    return replace(refs, run2_out=str(Path(out_root) / "standin"))


def write_standin(
    model, cfg, docs: list, seed: int, run2_payload: dict, out_root: Path
) -> None:
    """SMOKE only: fifo, run 2's psiC and random0..4 over the stand-in documents, by run
    1's `run_heldout` passes (its own policies), so every scoring path runs without
    EVAL. Written under ``<out_root>/standin``; never read by a real run."""
    hd = Path(out_root) / "standin" / "heldout" / f"seed{seed}"
    vmap = R1.vocab_map(cfg)
    passes = {
        "fifo": lambda chunk: FIFOPolicy(),
        "psiC": lambda chunk: R1.eval_policy("psiC", run2_payload, cfg),
    }
    for k in range(R1.N_RANDOM):
        passes[f"random{k}"] = lambda chunk, k=k: R1.reference_policy(
            f"random{k}", chunk, cfg.seed
        )
    for name, mk in passes.items():
        recs = []
        for i in range(0, len(docs), cfg.eval_batch):
            chunk = docs[i : i + cfg.eval_batch]
            recs += R1.run_batch(model, mk(chunk), chunk, vmap, cfg)
        write_records(hd / f"{name}.jsonl.gz", recs)


def score_seed(
    cf_recs: list[dict], decisions: list[tuple], seed: int, refs: Refs, n_boot: int, delta
) -> dict:
    """PREREG §4 headline and label; §6 items 1-6; C6 (pairing with run 2's fifo)."""
    other = run2_records(refs, seed, ("fifo", "psiC", *STEP1.RANDOM))
    recs = {"cf_all": cf_recs, **other}
    h: dict[str, Any] = {"n_documents": len(cf_recs)}
    h["C6"] = pairing(cf_recs, other["fifo"])
    h["C6"]["all_passes_paired"] = STEP1.pairing_failures(recs)
    h["C6"]["ok"] = h["C6"]["ok"] and not h["C6"]["all_passes_paired"]
    W = R1.boot_weights(len(cf_recs), seed, n_boot)
    h["headline"] = headline(cf_recs, other["fifo"], W, delta)
    # §6 item 1: the pinned ES, scoped by run 2's C14 and a pairing check (as C15).
    sc: dict[str, Any] = {"C14": R2.c14_es_source(seed, refs.r2())}
    es = None
    try:
        es_all = R1.read_pass(Path(refs.es_out) / "eval" / f"seed{seed}", "es")
        order = [r["doc"] for r in cf_recs]
        es_f = [r for r in es_all if r["doc"] in set(order)]
        sc["pairing"] = pairing(cf_recs, es_f)
        if sc["C14"].get("ok") and sc["pairing"]["ok"]:
            es = es_f
    except Exception as e:
        sc["pairing"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    h["es_scoped_controls"] = {
        k: {kk: vv for kk, vv in v.items() if kk != "eval_json"} for k, v in sc.items()
    }
    if es is None:
        h["es_omitted"] = "scoped control(s) failed (run 2 PREREG §4 item 4)"
    h["secondary"] = secondary(recs, W, es)
    h["age1_share"] = {
        "cf_all": R2.age1_share(cf_recs),
        "psiC_run2_SEEN": R2.age1_share(other["psiC"]),
    }
    h["rho_psi_age"] = rho_psi_age(decisions, W)
    h["n_boot"] = n_boot
    return h


# --------------------------------------------------------------------------- #
# finalize: labels, RESULTS.md, ledger, claims.json
# --------------------------------------------------------------------------- #


def summarize(docs: list[dict]) -> dict:
    labels, caps, shares = {}, {}, {}
    for d in docs:
        s = d["seed"]
        hl = ((d.get("heldout") or {}).get("headline")) or {}
        if hl:
            labels[s] = hl["label"]
        fr = ((d.get("fit_reading") or {}).get(PRIMARY)) or {}
        if "cap" in fr:
            caps[s] = fr["cap"]
        shares[s] = (
            ((d.get("heldout") or {}).get("age1_share") or {}).get("cf_all") or {}
        ).get("share")
    readable = len(docs) == len(SEEDS) and all(
        int(d["exit"]) == int(Exit.OK) for d in docs
    )
    fl = fit_label(caps) if set(caps) == set(SEEDS) else "FIT-MISSES (incomplete)"
    cls = deploy_class(labels)
    return {
        "deploy_labels": labels,
        "deploy_class": cls,
        "fit_label": fl,
        "fit_caps": caps,
        "newcomer": R2.newcomer_label(shares, SEEDS),
        "age1_share_cf_all": shares,
        "readable": readable,
        "readout_row": readout_row(fl, cls),
    }


def readout_row(fit: str, cls: str) -> str:
    """PREREG §8's table: which row the pair lands in (routes an owner decision)."""
    if fit.startswith("FIT-MISSES"):
        return "MISSES / any: content+age arm or a richer head (both change the thesis)"
    if cls == "BEATS-FIFO":
        return "POINTS-or-PARTIAL / BEATS-FIFO: shadow buffer vs protected recency window"
    return (
        "POINTS-or-PARTIAL / MIXED-or-FAILS: §6 items 3-4 to tell the two readings apart"
    )


def _fmt(x, nd: int = 4) -> str:
    return "None" if not _finite(x) else f"{x:+.{nd}f}"


def _ci(c: dict | None) -> str:
    if not c or "point" not in c:
        return str((c or {}).get("omitted", "n/a"))
    return f"{_fmt(c.get('point'))} [{_fmt(c.get('lo'))}, {_fmt(c.get('hi'))}]"


def render_results(run: dict, docs: list[dict]) -> str:
    """RESULTS.md, generated: every number is read from the measure-seed documents."""
    s = run["summary"]
    row, header = STEP1.row, STEP1.header
    w: list[str] = []
    p = w.append
    p("# RESULTS: offline psi_hat on uncensored LOO returns (step 2 after FAILS)\n")
    p(f"Generated by `{EXPERIMENT} finalize`; no number is typed. Written by a model;")
    p(f"not by Brendan. PREREG `{PREREG}` @ `{PREREG_COMMIT}` (+ amendments).\n")
    p(f"- command: `{run['command']}`")
    p(f"- git: `{run['tree'].get('sha')}`, dirty `{run['tree'].get('dirty')}`")
    hw = docs[0].get("hardware") or {}
    threads = [d.get("threads") for d in docs]
    p(f"- hardware: {hw.get('cpu')}, {hw.get('memsize_bytes')} B, {hw.get('platform')}")
    p(f"- python {hw.get('python')}, torch {hw.get('torch')}; threads {threads}")
    p(f"- exit `{run['exit']}`, status `{run['status']}`")
    if any((d.get("plan") or {}).get("smoke") for d in docs):
        p("\n**SMOKE: not the run. No label below is read.**")
    if not s["readable"]:
        p("\n**NOT READ: a seed's control failed, a seed is missing, or a smoke.**")
    p("\n## 1. Fit (PREREG §5, U-val): cap of the fitted head\n")
    w += header("seed", "arm", "step", "U-val L_MC", "cap [95% CI]", "rho(psi, G) [CI]")
    for d in docs:
        fr = d.get("fit_reading") or {}
        for arm in ARMS:
            r = fr.get(arm) or {}
            sel = (d.get("selection") or {}).get(arm) or {}
            rho = r.get("rho_psi_G_fifo_resident") or {}
            rho_s = f"{_fmt(rho.get('median'))} [{_fmt(rho.get('lo'))}, "
            rho_s += f"{_fmt(rho.get('hi'))}]"
            p(
                row(
                    d["seed"],
                    arm,
                    sel.get("step"),
                    sel.get("val_l_mc"),
                    _ci(r.get("cap")),
                    rho_s,
                )
            )
        ref = fr.get("reference_G") or {}
        p(row(d["seed"], "B0-CF's G (beside)", "-", "-", _ci(ref.get("cap")), "-"))
    p(f"\n**Fit label (cf_all): {s['fit_label']}**\n")
    p("## 2. Deployment (PREREG §4): acc.all(cf_all) - acc.all(fifo) on EVAL\n")
    w += header("seed", "delta_seed", "point [95% CI]", "label")
    for d in docs:
        hl = (d.get("heldout") or {}).get("headline") or {}
        p(row(d["seed"], hl.get("delta"), _ci(hl), f"**{hl.get('label', 'n/a')}**"))
    p(f"\n**Deploy class: {s['deploy_class']}**. PREREG §8 row: {s['readout_row']}.")
    p(f"\nNewcomer label (§6 item 5, cf_all age-1 share): **{s['newcomer']}**")
    p(f"(shares {s['age1_share_cf_all']}; run 2's psiC shares are already seen).\n")
    p("## 3. Secondary (PREREG §6; descriptive, unadjusted)\n")
    for d in docs:
        h = d.get("heldout") or {}
        sec = h.get("secondary") or {}
        p(f"### Seed {d['seed']}\n")
        w += header("bucket", "kind", "- fifo", "- random", "- psiC(run 2)", "- ES")
        for kind in ("accuracy", "residency"):
            for b in REPORT_BUCKETS:
                r = (sec.get(kind) or {}).get(b) or {}
                es = _ci(r["minus_es"]) if "minus_es" in r else h.get("es_omitted")
                p(
                    row(
                        b,
                        kind,
                        _ci(r.get("minus_fifo")),
                        _ci(r.get("minus_random")),
                        _ci(r.get("minus_psiC_run2")),
                        es,
                    )
                )
        p("\nAccuracy given residency (point counts; the arms keep different facts):\n")
        for b, cond in (sec.get("conditional") or {}).items():
            for arm, c in cond.items():
                n_r, n_m = c.get("resident"), c.get("missed")
                rr = c.get("resident_right", 0) / n_r if n_r else None
                rm = c.get("missed_right", 0) / n_m if n_m else None
                p(f"- {b} {arm}: resident {n_r}, right|resident {_fmt(rr)}, ")
                w[-1] += f"right|missed {_fmt(rm)}"
        p("\nEviction-age cost (evictions, misses caused):\n")
        for arm, c in (sec.get("eviction_cost") or {}).items():
            by = c.get("by_age") or {}
            cells = [
                f"age {k}: {v['evictions']:.0f}, {v['misses_caused']:.0f}"
                for k, v in by.items()
            ]
            p(f"- {arm}: " + "; ".join(cells))
        ra = h.get("rho_psi_age") or {}
        p(f"\n- rho(psi_hat, age) at EVAL decisions: {_fmt(ra.get('median'))} ")
        w[-1] += f"[{_fmt(ra.get('lo'))}, {_fmt(ra.get('hi'))}], n {ra.get('n_steps')}\n"
    p("## 4. Controls (PREREG §9)\n")
    for d in docs:
        c = d.get("controls") or {}
        oks = ", ".join(
            f"{k} `{(c.get(k) or {}).get('ok')}`"
            for k in (*CONTROLS, "residency_C7_run1")
        )
        p(f"- seed {d['seed']}: exit `{d['exit']}`; failed `{d.get('controls_failed')}`")
        p(f"  - {oks}")
        c2 = c.get("C2") or {}
        p(f"  - C2: computed {c2.get('computed')!r}, ledger {c2.get('ledger')!r}")
        p(f"  - timing: {d.get('timing')}")
    p("\n## 5. What this run does not show (PREREG §13)\n")
    p("Not an online result: a WIN would show the head can hold a winning ranking, not")
    p("that RSR's training loop finds it. Delta and the gestalts are FIFO-world. No")
    p("scaling claim, no cognitive claim; §15 untouched. Run 2's class is unchanged.")
    return "\n".join(w) + "\n"


def write_ledger(run: dict, docs: list[dict], out_root: Path) -> Path:
    L = R1.ledger_mod()
    out_root = Path(out_root)
    if out_root.name != RUN_ID:
        raise Refused(
            f"the ledger lands in runs_root/{RUN_ID}: out-root must end in {RUN_ID}"
        )
    led = L.Ledger(RUN_ID, question=QUESTION, runs_root=out_root.parent)
    man = out_root / "manifest.json"
    if man.exists():
        led.doc["config_hash"] = hashlib.sha256(man.read_bytes()).hexdigest()
        led.doc["manifest_written_utc"] = json.loads(man.read_text()).get("written_utc")
    for d in docs:
        led.command(
            [
                ".venv/bin/python",
                EXPERIMENT,
                "measure-seed",
                "--seed",
                str(d["seed"]),
                "--threads",
                str(d["threads"]),
                "--out-root",
                str(out_root),
            ],
            exit_code=d["exit"],
            note=f"seed {d['seed']}",
        )
    led.command(
        [".venv/bin/python", EXPERIMENT, "finalize", "--out-root", str(out_root)],
        exit_code=run["exit"],
        note="this measurement",
    )
    led.run_meta(
        device=R1.DEVICE,
        seeds_actually_run=sorted(
            {s for d in docs for s in d.get("seeds_actually_run") or []}
        ),
        steps_requested=FIT_STEPS,
        steps_done=min(
            int((d.get("fit") or {}).get(PRIMARY, {}).get("steps") or 0) for d in docs
        ),
    )
    s = run["summary"]
    led.note("summary", s, how=f"{EXPERIMENT} finalize")
    led.note("prereg", {"path": PREREG, "commit": PREREG_COMMIT}, how=EXPERIMENT)
    if len(docs) >= 2:

        def col(f):
            return [f(d) for d in docs]

        for key, f in (
            ("deploy.headline.point", lambda d: d["heldout"]["headline"]["point"]),
            ("deploy.headline.lo", lambda d: d["heldout"]["headline"]["lo"]),
            ("deploy.headline.hi", lambda d: d["heldout"]["headline"]["hi"]),
            (
                "fit.cf_all.cap.point",
                lambda d: d["fit_reading"]["cf_all"]["cap"]["point"],
            ),
            ("fit.cf_all.cap.lo", lambda d: d["fit_reading"]["cf_all"]["cap"]["lo"]),
            ("fit.cf_all.cap.hi", lambda d: d["fit_reading"]["cf_all"]["cap"]["hi"]),
            (
                "fit.reference_G.cap.point",
                lambda d: d["fit_reading"]["reference_G"]["cap"]["point"],
            ),
            ("c2.cap.point", lambda d: d["controls"]["C2"]["computed"]),
            (
                "deploy.age1_share.cf_all",
                lambda d: d["heldout"]["age1_share"]["cf_all"]["share"],
            ),
        ):
            try:
                led.stat(
                    key, col(f), how=f"{EXPERIMENT} measure-seed<s>.json, seeds in order"
                )
            except (KeyError, TypeError, ValueError) as e:
                led.note(key, {"omitted": f"{type(e).__name__}: {e}"}, how=EXPERIMENT)
    for d in docs:
        keys = (
            "controls",
            "selection",
            "fit_reading",
            "heldout",
            "timing",
            "deploy",
            "u",
        )
        for key, val in R1._flatten(f"seed{d['seed']}", {k: d.get(k) for k in keys}):
            led.note(key, val, how=f"{EXPERIMENT} measure-seed{d['seed']}.json")
        for arm in ARMS:
            f = (d.get("fit") or {}).get(arm) or {}
            led.note(
                f"seed{d['seed']}.fit.{arm}.history", f.get("history"), how=EXPERIMENT
            )
    led.status(run["status"])
    return led.write()


def recompute_line(kind: str, x: dict) -> str:
    if kind in ("headline", "cap"):
        lab = f" {x['label']}" if "label" in x else ""
        return f"{x['point']:.6f} {x['lo']:.6f} {x['hi']:.6f}{lab}"
    if kind == "age1":
        return f"{x['n_age1']} {x['n_evictions']}"
    if kind == "refit":
        return f"{x['step']} {x['val_l_mc']:.9e} {x['phi_sha256']}"
    raise ValueError(kind)


def write_claims(run: dict, docs: list[dict], out_root: Path) -> Path:
    """``<out_root>/claims.json`` for `orchestrator.verify draw/check`: every command
    RECOMPUTES its number from the saved per-seed artefacts (records, gestalts, the
    frozen phi, Delta), never from this run's ledger or measure json numbers."""
    base = (
        "PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run --lane cpu-det "
        f"--slots 1 -- .venv/bin/python {EXPERIMENT} recompute --out-root {out_root}"
    )
    claims = []
    for d in docs:
        s = d["seed"]
        t = d.get("threads", 1)
        c2 = (d.get("controls") or {}).get("C2") or {}
        claims.append(
            {
                "claim": f"seed {s} C2: TargetPolicy on B0-CF's G over full U, cap point",
                "command": f"{base} c2 --seed {s}",
                "expected": repr(c2.get("computed")),
            }
        )
        fr = ((d.get("fit_reading") or {}).get(PRIMARY)) or {}
        if "cap" in fr:
            claims.append(
                {
                    "claim": f"seed {s} fit reading: cf_all cap on U-val",
                    "command": f"{base} fitcap --seed {s}",
                    "expected": recompute_line("cap", fr["cap"]),
                }
            )
        sel = (d.get("selection") or {}).get(PRIMARY)
        if sel:
            claims.append(
                {
                    "claim": f"seed {s} cf_all refit: selected step, U-val L_MC, phi sha",
                    "command": f"{base} refit --seed {s} --threads {t}",
                    "expected": recompute_line("refit", sel),
                }
            )
        h = d.get("heldout") or {}
        if h.get("headline"):
            claims.append(
                {
                    "claim": f"seed {s} headline cf_all - FIFO and label",
                    "command": f"{base} headline --seed {s}",
                    "expected": recompute_line("headline", h["headline"]),
                }
            )
            claims.append(
                {
                    "claim": f"seed {s} cf_all age-1 eviction share",
                    "command": f"{base} age1 --seed {s}",
                    "expected": recompute_line("age1", h["age1_share"]["cf_all"]),
                }
            )
    sm = run["summary"]
    claims.append(
        {
            "claim": "deploy class, fit label, newcomer label",
            "command": f"{base} class",
            "expected": f"{sm['deploy_class']} {sm['fit_label']} {sm['newcomer']}",
        }
    )
    p = Path(out_root) / "claims.json"
    p.write_text(json.dumps(claims, indent=1) + "\n")
    return p


def finalize(
    docs: list[dict],
    out_root: Path,
    *,
    results_path: Path,
    command: str,
    write_ledger_too: bool = True,
) -> Exit:
    worst = max(docs, key=lambda d: (R1.EXIT_RANK[int(d["exit"])], -int(d["seed"])))
    code = Exit(worst["exit"]) if len(docs) == len(SEEDS) else Exit.DID_NOT_RUN
    run = {
        "run_id": RUN_ID,
        "seeds": [d["seed"] for d in docs],
        "exit": int(code),
        "status": worst["status"] if code == Exit(worst["exit"]) else "did_not_run",
        "summary": summarize(docs),
        "tree": R1.tree_state(),
        "command": command,
    }
    out_root = Path(out_root)
    (out_root / "run.json").write_text(json.dumps(run, indent=1, default=str) + "\n")
    Path(results_path).write_text(render_results(run, docs))
    if write_ledger_too:
        write_ledger(run, docs, out_root)
        write_claims(run, docs, out_root)
    return code


# --------------------------------------------------------------------------- #
# recompute (claims.json)
# --------------------------------------------------------------------------- #


def _measure(out_root: Path, seed: int) -> dict:
    return json.loads((Path(out_root) / f"measure-seed{seed}.json").read_text())


def _refs_of(d: dict) -> Refs:
    return Refs(**d["refs"])


def recompute(what: str, out_root: Path, seed: int | None, threads: int = 1) -> str:
    """Recompute one claimed number from saved artefacts (claims.json)."""
    out_root = Path(out_root)
    if what == "class":
        docs = [_measure(out_root, s) for s in SEEDS]
        labels, caps, shares = {}, {}, {}
        for d in docs:
            s = d["seed"]
            labels[s] = recompute("headline", out_root, s).split()[3]
            pt, lo, hi = (float(x) for x in recompute("fitcap", out_root, s).split())
            caps[s] = {"point": pt, "lo": lo, "hi": hi}
            n1, n = recompute("age1", out_root, s).split()
            shares[s] = int(n1) / int(n) if int(n) else None
        return (
            f"{deploy_class(labels)} {fit_label(caps)} {R2.newcomer_label(shares, SEEDS)}"
        )
    d = _measure(out_root, seed)
    refs = _refs_of(d)
    plan = Plan(**d["plan"])
    if what == "c2":
        torch.set_num_threads(threads)
        blob, _ = load_delta(seed, refs)
        return repr(c2_reproduce(u_documents(seed)[0], blob, seed)["point"])
    if what in ("fitcap", "refit"):
        torch.set_num_threads(threads)
        docs, _vmap, train, val = plan_docs(seed, plan)
        g = torch.load(seed_dir(out_root, seed) / "gestalts.pt", weights_only=False)
        if g["ids"] != [x.doc_id for x in docs]:
            raise ControlFailure("gestalts.pt does not hold this seed's U documents")
        blob, _ = load_delta(seed, refs)
        conf = run2_psic_config(refs, seed)["conf"]
        ov, spe = dict(conf["policy_overrides"]), float(conf["epoch"]["steps_per_epoch"])
        if what == "refit":
            cfg = R1.prereg_config(seed, CKPT)
            model, _p, _sha = load_frozen(cfg)
            pol, _ = new_policy(seed, ov, spe)
            opt = phi_optimizer(model, pol.head, float(conf["lr"]))
            data = {"s": g["s"], "c": g["c"], **fit_tensors(blob, docs, PRIMARY)}
            r = fit(
                pol.head,
                opt,
                data,
                train,
                val,
                seed=seed,
                steps=plan.fit_steps,
                every=plan.val_every,
            )
            return recompute_line(
                "refit", {**r["selected"], "phi_sha256": state_sha(r["state"])}
            )
        pol, _ = new_policy(seed, ov, spe)
        pol.head.load_state_dict(
            torch.load(seed_dir(out_root, seed) / f"phi-{PRIMARY}.pt", weights_only=True)
        )
        vdocs = [docs[p] for p in val]
        vp = torch.tensor(val, dtype=torch.long)
        P = score_matrices(pol.head, g["s"][vp], g["c"][vp])
        rr = reference_rates(vdocs, seed)
        W = B0.boot_weights(len(vdocs), seed, plan.n_boot)
        r = B0.boot_rule(W, replay_rates(vdocs, list(P)), rr["fifo"], rr["factfiller"])
        return recompute_line("cap", r["cap"])
    hd = heldout_dir(out_root, seed)
    cf = R1.read_pass(hd, PRIMARY)
    if what == "age1":
        return recompute_line("age1", R2.age1_share(cf))
    if what == "headline":
        r = refs if not plan.standin else standin_refs(refs, out_root)
        fifo = run2_records(r, seed, ("fifo",))["fifo"]
        W = R1.boot_weights(len(cf), seed, plan.n_boot)
        return recompute_line(
            "headline", headline(cf, fifo, W, read_delta_seed(seed, refs)["delta"])
        )
    raise ValueError(what)


# --------------------------------------------------------------------------- #
# preflight, manifest, time (PREREG §10), CLI
# --------------------------------------------------------------------------- #


def preflight(refs: Refs = REFS) -> dict:
    """Before anything runs: the PREREG numbers, every input's sha256 / presence, the
    substrate pins, run 2's records and checkpoints, delta, the C2 ledger value."""
    out: dict[str, Any] = {"prereg_text": prereg_check(refs), "seeds": {}}
    out["launcher_exists"] = (ROOT / LAUNCHER).exists()
    for s in SEEDS:
        cfg = R1.prereg_config(s, CKPT)
        row: dict[str, Any] = {}
        p = delta_path(s, refs)
        row["delta_sha_ok"] = p.exists() and sha256_file(p) == DELTA_SHA[s]
        ck = Path(cfg.substrate.checkpoint)
        row["substrate_sha_ok"] = ck.exists() and sha256_file(ck) == cfg.substrate.sha256
        row["C8"] = read_delta_seed(s, refs)
        row["cf_cap_ledger_ok"] = read_cf_cap(s, refs) == CF_CAP[s]
        hd = Path(refs.run2_out) / "heldout" / f"seed{s}"
        row["run2_records"] = all(
            (hd / f"{n}.jsonl.gz").exists() for n in ("fifo", "psiC", *STEP1.RANDOM)
        )
        row["run2_psiC_ckpt"] = (
            Path(refs.run2_out) / "psiC" / f"seed{s}" / f"ckpt-{CKPT:06d}.pt"
        ).exists()
        row["eval_output_absent"] = True  # checked per out-root by measure-seed (C9)
        row["C14"] = R2.c14_es_source(s, refs.r2()).get("ok")
        out["seeds"][s] = row
    out["ok"] = out["prereg_text"]["ok"] and all(
        r["delta_sha_ok"]
        and r["substrate_sha_ok"]
        and r["C8"]["ok"]
        and r["cf_cap_ledger_ok"]
        and r["run2_records"]
        and r["run2_psiC_ckpt"]
        for r in out["seeds"].values()
    )
    return out


def manifest_config(refs: Refs = REFS) -> dict:
    """The frozen config (written before any seed runs; its sha256 is the config hash)."""
    return {
        "run_id": RUN_ID,
        "prereg": PREREG,
        "prereg_commit": PREREG_COMMIT,
        "prereg_sha256": sha256_file(refs.prereg),
        "seeds": list(SEEDS),
        "device": R1.DEVICE,
        "constants": {
            "M": M,
            "S": S,
            "d": D_MODEL,
            "target_gamma": TARGET_GAMMA,
            "arms": list(ARMS),
            "fit_batch": FIT_BATCH,
            "fit_steps": FIT_STEPS,
            "val_every": VAL_EVERY,
            "val_mod": VAL_MOD,
            "eval": list(EVAL),
            "n_boot": R1.N_BOOT,
            "optimizer": loop.optimizer_record(),
            "lr": R1.LR,
        },
        "inputs": {
            "delta_sha256": DELTA_SHA,
            "substrate": {s: R1.SUBSTRATE[s]["sha256"] for s in SEEDS},
            "cf_cap": CF_CAP,
            "delta_seed": DELTA_SEED,
            "refs": asdict(refs),
        },
        "expected": EXPECTED,
        "falsifier": FALSIFIER,
    }


def write_manifest(out_root: Path, refs: Refs = REFS, *, smoke: bool = False) -> dict:
    out_root = Path(out_root)
    p = out_root / "manifest.json"
    if p.exists() and not smoke:
        raise Refused(f"{p} exists: the manifest is frozen once, before the run")
    tree = R1.tree_state()
    if not smoke and (tree.get("dirty") or not tree.get("sha")):
        raise Refused(f"the tree is dirty or unreadable: {tree}")
    doc = {
        **manifest_config(refs),
        "git_sha": tree.get("sha"),
        "git_dirty": tree.get("dirty"),
        "smoke": smoke,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    return doc


def check_manifest(out_root: Path, refs: Refs = REFS) -> dict:
    """measure-seed refuses unless the frozen manifest matches this code's config and
    the tree it was written at is the tree running."""
    p = Path(out_root) / "manifest.json"
    if not p.exists():
        raise Refused(f"{p} is missing: run `{EXPERIMENT} manifest` first")
    man = json.loads(p.read_text())
    want = json.loads(json.dumps(manifest_config(refs), sort_keys=True, default=str))
    have = {k: man.get(k) for k in want}
    if have != want:
        raise Refused(f"{p} does not match this code's config")
    tree = R1.tree_state()
    if not man.get("smoke") and man.get("git_sha") != tree.get("sha"):
        raise Refused(f"manifest at {man.get('git_sha')}, tree at {tree.get('sha')}")
    return man


def time_build(seed: int, threads: int, refs: Refs = REFS) -> dict:
    """PREREG §10, EVAL never read: the gestalt forward over 32 U documents; 50 fit
    steps (cf_all) and one U-val L_MC evaluation; one 64-document deploy block through
    the deploy code path on 64 U documents (same S, batch and code; EVAL unread).
    Nothing is written."""
    torch.set_num_threads(int(threads))
    cfg = R1.prereg_config(seed, CKPT)
    docs, vmap = u_documents(seed)
    blob, _ = load_delta(seed, refs)
    model, _payload, _sha = load_frozen(cfg)
    out: dict[str, Any] = {"seed": seed, "threads": threads}
    g = gestalt_forward(model, docs[:32], vmap)
    out["gestalt_forward_32_s"] = g["seconds"]
    out["C5_ok_on_sample"] = g["c5"]["ok"]
    conf = run2_psic_config(refs, seed)["conf"]
    ov, spe = dict(conf["policy_overrides"]), float(conf["epoch"]["steps_per_epoch"])
    pol, _ = new_policy(seed, ov, spe)
    opt = phi_optimizer(model, pol.head, float(conf["lr"]))
    data = {"s": g["s"], "c": g["c"], **fit_tensors(blob, docs[:32], PRIMARY)}
    train, val = u_split(32)
    t0 = time.perf_counter()
    r = fit(pol.head, opt, data, train, val, seed=seed, steps=50, every=50)
    out["fit_50_steps_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    val_loss(pol.head, data["s"], data["c"], data["G"], data["mask"], list(range(32)))
    out["val_l_mc_32_docs_s"] = time.perf_counter() - t0
    state_p = r["state"]
    t0 = time.perf_counter()
    rr = reference_rates(docs, seed)
    c2_reproduce(docs, blob, seed, rr)
    out["c2_full_U_s"] = time.perf_counter() - t0
    blk = docs[32:96]
    t0 = time.perf_counter()
    dp = deploy_pass(
        model, cfg, blk, R1.vocab_map(cfg), lambda: deploy_policy(state_p, seed, ov, spe)
    )
    out["deploy_block_64_s"] = time.perf_counter() - t0
    out["deploy_block_doc_ids"] = [blk[0].doc_id, blk[-1].doc_id]
    out["deploy_residency_ok"] = dp["residency"]["n_mismatched"] == 0
    # Projection, per seed: two arms x 3000 steps, 61 U-val evaluations per arm.
    per_doc_g = out["gestalt_forward_32_s"] / 32
    per_step = out["fit_50_steps_s"] / 50  # includes one val eval of 4-5 docs
    per_val = out["val_l_mc_32_docs_s"] / 32 * N_U_VAL
    per_deploy_doc = out["deploy_block_64_s"] / 64
    proj = {
        "gestalt_forward_s": per_doc_g * N_U,
        "fit_s": len(ARMS)
        * (FIT_STEPS * per_step + (FIT_STEPS // VAL_EVERY + 1) * per_val),
        "c2_s": out["c2_full_U_s"],
        "deploy_s": per_deploy_doc * N_EVAL,
    }
    proj["seed_total_s"] = sum(proj.values())
    proj["three_seeds_sequential_h"] = 3 * proj["seed_total_s"] / 3600
    out["projection"] = proj
    return out


def _plan(a) -> Plan:
    if not a.smoke:
        if any(x is not None for x in (a.fit_steps, a.u_limit, a.standin, a.val_every)):
            raise Refused(
                "--fit-steps / --u-limit / --standin / --val-every are smoke-only"
            )
        return Plan()
    if a.standin is None:
        raise Refused("a smoke never reads EVAL: give --standin N (U-val docs stand in)")
    return Plan(
        fit_steps=a.fit_steps or 30,
        val_every=a.val_every or 10,
        u_limit=a.u_limit,
        standin=a.standin,
        smoke=True,
    )


def main(argv: list[str] | None = None) -> Exit:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = ArgumentParser(prog="offline-psi", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preflight")
    pm = sub.add_parser("manifest")
    pm.add_argument("--out-root", default=str(ROOT / "runs" / RUN_ID))
    pm.add_argument("--smoke", action="store_true")
    ps = sub.add_parser("measure-seed")
    ps.add_argument("--seed", type=int, required=True)
    ps.add_argument("--threads", type=int, required=True)
    ps.add_argument("--out-root", default=str(ROOT / "runs" / RUN_ID))
    ps.add_argument("--smoke", action="store_true")
    ps.add_argument("--fit-steps", type=int, default=None)
    ps.add_argument("--val-every", type=int, default=None)
    ps.add_argument("--u-limit", type=int, default=None)
    ps.add_argument("--standin", type=int, default=None)
    pf = sub.add_parser("finalize")
    pf.add_argument("--out-root", default=str(ROOT / "runs" / RUN_ID))
    pf.add_argument("--results", default=None)
    pt = sub.add_parser("time")
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--threads", type=int, required=True)
    pr = sub.add_parser("recompute")
    pr.add_argument("--out-root", required=True)
    pr.add_argument(
        "what", choices=("c2", "fitcap", "refit", "headline", "age1", "class")
    )
    pr.add_argument("--seed", type=int, default=None)
    pr.add_argument("--threads", type=int, default=1)
    a = p.parse_args(argv)
    try:
        if a.cmd == "preflight":
            r = preflight()
            print(json.dumps(r, indent=1, default=str))
            return Exit.OK if r["ok"] else Exit.DID_NOT_RUN
        if a.cmd == "manifest":
            out = Path(a.out_root).resolve()
            if not a.smoke and out != (ROOT / "runs" / RUN_ID).resolve():
                raise Refused(f"the run's manifest lives in runs/{RUN_ID}")
            write_manifest(out, smoke=a.smoke)
            print(f"wrote {out / 'manifest.json'}")
            return Exit.OK
        if a.cmd == "measure-seed":
            plan = _plan(a)
            out = Path(a.out_root).resolve()
            real = (ROOT / "runs" / RUN_ID).resolve()
            if plan.smoke and out == real:
                raise Refused(f"a smoke never writes runs/{RUN_ID}")
            if not plan.smoke and out != real:
                raise Refused(f"the run writes runs/{RUN_ID} only")
            if a.seed not in SEEDS:
                raise Refused(f"seed {a.seed} is not one of {SEEDS}")
            check_manifest(out)
            doc = measure_seed(a.seed, out, threads=a.threads, plan=plan)
            print(f"seed {a.seed}: exit {doc['exit']}; failed {doc['controls_failed']}")
            return status(doc["exit"])
        if a.cmd == "finalize":
            out = Path(a.out_root)
            docs = []
            for s in SEEDS:
                f = out / f"measure-seed{s}.json"
                if not f.exists():
                    raise Refused(f"{f} is missing: run measure-seed --seed {s} first")
                docs.append(json.loads(f.read_text()))
            res = Path(a.results) if a.results else out / "RESULTS.md"
            cmd = " ".join([".venv/bin/python", EXPERIMENT, *argv])
            return status(finalize(docs, out, results_path=res, command=cmd))
        if a.cmd == "time":
            r = time_build(a.seed, a.threads)
            print(json.dumps(r, indent=1, default=str))
            return Exit.OK
        if a.cmd == "recompute":
            print(recompute(a.what, Path(a.out_root), a.seed, a.threads))
            return Exit.OK
    except Refused as e:
        print(f"DID NOT RUN: {e}", file=sys.stderr)
        return Exit.DID_NOT_RUN
    except ControlFailure as e:
        print(f"CONTROL FAILED: {e}", file=sys.stderr)
        return Exit.DID_NOT_RUN
    return Exit.DID_NOT_RUN


if __name__ == "__main__":
    run_main(main)
