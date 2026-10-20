"""Phase 2a (offline): fit with the ranking loss, select on TUNE, deploy on EVAL.

PREREG ``experiments/rsr-run3-ranking/PREREG.md`` (committed alone at 32475b5) §1-§4,
§6, with the PM's §12 decisions (2a pairs over offline-psi's ``cf_all`` cells). Run id
``rsr-run3-2a``.

**offline-psi's, diagnostic A's and run 2's code are imported, never copied.**
Diagnostic A (``experiments/offline-psi-converge/run.py``) is loaded as ``CV``; through
it offline-psi (``CV.OP``) and, through that, run 1 (``OP.R1``), run 2 (``OP.R2``), the
step-1 module, lookahead-room and B0. From them, unchanged:

* inputs: diagnostic A's ``load_inputs`` (offline-psi's gestalts and B0-CF's Delta,
  sha-checked) and offline-psi's ``fit_tensors`` / ``target_of`` / ``cell_mask``;
* the head and optimizer: offline-psi's ``new_policy`` (run 2's psiC overrides and phi
  init, ``phi_generator(seed)``) and ``phi_optimizer`` (muP groups, AdamW);
* the fit loop's order: offline-psi's ``fit`` (sampler draw, ``psi_matrix``, loss,
  ``zero_grad``, ``backward``, grad norm, ``step``), with the loss switched (``mc``:
  offline-psi's ``l_mc``; ``rank``: ``rsr.retention.rank_loss``);
* deployment: offline-psi's ``load_frozen``, ``deploy_policy`` (T_warm = 0),
  ``deploy_pass`` (run 1's held-out procedure, ``eval_forward``, blocks of 16, run 2's
  ``ProbePolicy``), ``deploy_guard`` and ``score_seed`` (headline, run 2's label, the
  secondary contrasts, age-1 share, rho(psi, age));
* the TUNE cap: run 2's ``hit_counts``, ``factfiller_hits`` and ``ceiling``;
  ``headroom.simulate`` with ``FIFOPolicy``; run 1's ``boot_weights``.

Stages (each its own process, its own R-SEAL; ``run.py 2a <stage>``):
``launch-checks``, ``fit --seed s (--rc2 | --omega w)``, ``select --seed s``,
``deploy --seed s``, ``finalize``, and ``time`` (Amendment 2a-S's smoke: U documents
only). Exit codes (``rsr.exit_codes``, PREREG §6): 0 ran and every control passed;
1 a defect after the controls passed; 3 did not run, or a control failed.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load_here(name: str, file: str):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / file)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


SS = _load_here("run3_stage_seal", "stage_seal.py")
C = SS.C  # phase 1's common.py (pfc_common): pins, manifest check, census, ledger

import torch  # noqa: E402

from rsr.baselines.fifo import FIFOPolicy  # noqa: E402
from rsr.exit_codes import Exit  # noqa: E402
from rsr.metrics.headroom import simulate  # noqa: E402
from rsr.retention.rank_loss import (  # noqa: E402
    LOSS_MC,
    LOSS_RANK,
    argmin_agreement,
    rank_loss_groups,
)
from rsr.retention.value_head import BilinearValueHead  # noqa: E402
from rsr.train import loop  # noqa: E402

RUN_ID = "rsr-run3-2a"
EXPERIMENT = "experiments/rsr-run3-ranking/run.py"
PREREG_REL = "experiments/rsr-run3-ranking/PREREG.md"
PREREG_PATH = ROOT / PREREG_REL
PREREG_COMMIT = "32475b5"
OUT_DEFAULT = ROOT / "runs" / RUN_ID

# --------------------------------------------------------------------------- #
# 🔒 PREREG, transcribed (checked against the committed text by `prereg_check`)
# --------------------------------------------------------------------------- #

OMEGAS = (2.0, 8.0, 32.0)  # §2.2 grid
RANK_K = 3  # §2.2
J_EVERY = 250  # §4.2
CONV_MIN_STEP = 2000  # §4.2: first check k >= 2000
CONV_WINDOW = 2000  # §4.2: best(k - 2000)
CONV_REL = 1e-3  # §4.2: (1 - 10^-3)
STEP_CAP = 200_000  # §4.2 (diagnostic A's cap)
SAVE_STEPS = (3000, 10_000)  # §4.2, where reached
ARM = "cf_all"  # §2.1, PM decision §12 item 2
RC2_ARMS = ("cf_all", "cf_live")  # §4.7 R-C2
RC2_STEP = 3000  # §4.7 R-C2
N_BOOT = 2000  # §4.3, §4.4
TUNE_NAME, EVAL2_NAME = "TUNE", "EVAL2"
CONVERGED, CAP_HIT = "CONVERGED", "CAP-HIT"
WIN_ALL = "WIN on all 3 seeds"


def prereg_values(text: str) -> dict:
    """The numbers above, read from the committed PREREG's text."""
    out: dict[str, Any] = {}
    m = re.search(r"\*\*`ω ∈ \{([0-9, ]+)\}`\*\*", text)
    out["omegas"] = None if m is None else tuple(float(x) for x in m.group(1).split(","))
    m = re.search(r"\| `k` \| \*\*(\d+)\*\*, fixed", text)
    out["k"] = None if m is None else int(m.group(1))
    m = re.search(r"Every (\d+) steps,\s*evaluate the full-batch", text)
    out["j_every"] = None if m is None else int(m.group(1))
    m = re.search(
        "first check `k \u2265 (\\d+)` with `best\\(k\\) \u2265 \\(1 \u2212 "
        "10\u207b([\u00b9\u00b2\u00b3])\\) \u00b7 best\\(k \u2212 (\\d+)\\)`",
        text,
    )
    if m is None:
        out.update(conv_min=None, conv_rel=None, conv_window=None)
    else:
        out.update(
            conv_min=int(m.group(1)),
            conv_rel=10.0 ** -(1 + "\u00b9\u00b2\u00b3".index(m.group(2))),
            conv_window=int(m.group(3)),
        )
    m = re.search(r"by \*\*([\d,]+)\*\* steps \(diagnostic A's cap\)", text)
    out["cap"] = None if m is None else int(m.group(1).replace(",", ""))
    m = re.search(r"at `K_conv` and at steps (\d+) and ([\d,]+) where reached", text)
    out["save"] = (
        None if m is None else (int(m.group(1)), int(m.group(2).replace(",", "")))
    )
    return out


def prereg_check(path: Path = PREREG_PATH) -> dict:
    got = prereg_values(Path(path).read_text())
    want = {
        "omegas": OMEGAS,
        "k": RANK_K,
        "j_every": J_EVERY,
        "conv_min": CONV_MIN_STEP,
        "conv_rel": CONV_REL,
        "conv_window": CONV_WINDOW,
        "cap": STEP_CAP,
        "save": SAVE_STEPS,
    }
    rows = {k: {"prereg": got[k], "code": v, "ok": got[k] == v} for k, v in want.items()}
    return {"values": rows, "ok": all(r["ok"] for r in rows.values())}


def registered_seeds(path: Path = PREREG_PATH) -> tuple[int, ...]:
    return C.registered_seeds(path)


# --------------------------------------------------------------------------- #
# The harness modules, imported (never copied), after the seal is installed
# --------------------------------------------------------------------------- #

_H: dict[str, Any] = {}


def harness() -> dict[str, Any]:
    """``{"CV", "OP", "R1", "R2"}``. Install a stage seal before calling: the modules
    are loaded after it, and the seal's rebinding is re-run after the load."""
    if _H:
        return _H
    if SS._STATE["stage"] is None:
        raise C.ControlFailure("harness() before a stage seal is installed (R-SEAL)")
    CV = C.load_module("_run3_cv", "experiments/offline-psi-converge/run.py")
    OP = CV.OP
    _H.update(CV=CV, OP=OP, R1=OP.R1, R2=OP.R2)
    SS.rebind()
    return _H


# --------------------------------------------------------------------------- #
# §2 the loss on offline-psi's [B, S(t), S(i)] matrices; §4.2 the objective J
# --------------------------------------------------------------------------- #


def as_groups(P: torch.Tensor, G: torch.Tensor, mask: torch.Tensor):
    """``[B, S, S]`` -> ``[B*S, S]``: one group per (document, t), its pair set the
    cells ``mask[b, t, :]`` (offline-psi's ``cf_all``: every ``i < t`` with finite G)."""
    B, n, _ = P.shape
    return P.reshape(B * n, n), G.reshape(B * n, n), mask.reshape(B * n, n)


def rank_matrix_loss(P, G, mask, omega: float, k: int = RANK_K):
    s, g, v = as_groups(P, G, mask)
    return rank_loss_groups(s, g, v, k=k, omega=omega)


def batch_loss(loss: str, P, G, mask, omega: float | None):
    """The switch: ``mc`` is offline-psi's ``l_mc`` (L9), ``rank`` is §2.1's loss."""
    if loss == LOSS_MC:
        return harness()["OP"].l_mc(P, G, mask)[0]
    if loss == LOSS_RANK:
        r = rank_matrix_loss(P, G, mask, float(omega))
        return r.loss
    raise ValueError(f"loss={loss!r}")


def full_objective(
    head, S_, C_, G, mask, positions: list[int], omega: float, chunk: int = 16
) -> dict:
    """§4.2 ``J``: the §2.1 batch loss over EVERY group of ``positions`` (the mean of
    ``l_g`` over groups with a pair), ``no_grad``, no RNG consumed."""
    OP = harness()["OP"]
    tot = 0.0
    n_with = n_no = 0
    with torch.no_grad():
        for j in range(0, len(positions), chunk):
            p = torch.tensor(positions[j : j + chunk], dtype=torch.long)
            P = OP.psi_matrix(head, S_[p], C_[p])
            r = rank_matrix_loss(P, G[p], mask[p], omega)
            if r.n_groups_with_pair:
                tot += float(r.per_group[r.has_pair].double().sum())
            n_with += r.n_groups_with_pair
            n_no += r.n_groups_no_pair
    return {
        "J": tot / n_with if n_with else float("nan"),
        "groups_with_pair": n_with,
        "groups_no_pair": n_no,
    }


def best_at(history: list[dict], j: int) -> float | None:
    v = [h["J"] for h in history if h["step"] <= j and _finite(h["J"])]
    return min(v) if v else None


def converged(history: list[dict], k: int) -> bool:
    """§4.2: CONVERGED at the first check ``k >= 2000`` with ``best(k) >= (1 - 1e-3)
    best(k - 2000)``; ``best(j)`` the lowest finite ``J`` at or before ``j``. The
    history holds ``J`` at step 0 and every 250 steps."""
    if k < CONV_MIN_STEP:
        return False
    now, then = best_at(history, k), best_at(history, k - CONV_WINDOW)
    if now is None or then is None:
        return False
    return now >= (1.0 - CONV_REL) * then


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _snap(head) -> dict:
    return {n: x.detach().clone() for n, x in head.state_dict().items()}


def fit_loop(
    head,
    opt,
    data: dict,
    train_pos: list[int],
    *,
    seed: int,
    loss: str,
    omega: float | None,
    cap: int,
    every: int | None = J_EVERY,
    save_steps=SAVE_STEPS,
    stop_on_convergence: bool = True,
    on_step=None,
) -> dict:
    """offline-psi's ``fit`` loop, op for op (sampler draw from a private generator
    seeded ``seed``, ``psi_matrix``, loss, ``zero_grad``, ``backward``, grad norm,
    ``step``), with the loss switched and §4.2's schedule. ``J`` (rank only) at step 0
    and every ``every`` steps; stop at CONVERGED or at ``cap``. Returns the history,
    ``K_conv``, the label, phi at ``K_conv`` and at the save steps reached, R-C4's
    finiteness record and the per-step wall time."""
    OP = harness()["OP"]
    S_, C_, G, mask = data["s"], data["c"], data["G"], data["mask"]
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    sampler = loop.PermutationEpochs(len(train_pos), gen, "cpu")
    tp = torch.tensor(train_pos, dtype=torch.long)
    params = list(head.parameters())
    history: list[dict] = []
    saved: dict[int, dict] = {}
    bad: list[int] = []
    stop = None
    t_steps = 0.0
    t_j = 0.0
    t0 = time.perf_counter()
    if loss == LOSS_RANK and every:
        tj = time.perf_counter()
        j0 = full_objective(head, S_, C_, G, mask, train_pos, float(omega))
        t_j += time.perf_counter() - tj
        history.append({"step": 0, **j0, "batch_loss": None, "grad_norm": None})
        if not _finite(j0["J"]):
            bad.append(0)
    k = 0
    for k in range(1, cap + 1):
        ts = time.perf_counter()
        pos = tp[sampler.take(OP.FIT_BATCH)]
        P = OP.psi_matrix(head, S_[pos], C_[pos])
        lt = batch_loss(loss, P, G[pos], mask[pos], omega)
        opt.zero_grad(set_to_none=True)
        lt.backward()
        gn = loop._grad_norm(params)
        lv = float(lt.detach())
        if not (_finite(lv) and _finite(gn)):
            bad.append(k)
        opt.step()
        t_steps += time.perf_counter() - ts
        if k in save_steps:
            saved[k] = _snap(head)
        if on_step is not None:
            on_step(k, head)
        if loss == LOSS_RANK and every and k % every == 0:
            tj = time.perf_counter()
            jk = full_objective(head, S_, C_, G, mask, train_pos, float(omega))
            t_j += time.perf_counter() - tj
            history.append({"step": k, **jk, "batch_loss": lv, "grad_norm": gn})
            if not _finite(jk["J"]):
                bad.append(k)
            if stop_on_convergence and converged(history, k):
                stop = k
                break
    k_conv = stop if stop is not None else k
    return {
        "history": history,
        "label": CONVERGED if stop is not None else CAP_HIT,
        "k_conv": k_conv,
        "steps_done": k,
        "state": _snap(head),
        "saved": saved,
        "rc4_n_bad": len(bad),
        "rc4_bad_steps": bad[:20],
        "epochs_drawn": sampler.epochs_drawn,
        "seconds": time.perf_counter() - t0,
        "seconds_steps": t_steps,
        "seconds_J": t_j,
        "n_J": len(history),
    }


# --------------------------------------------------------------------------- #
# inputs, heads, the descriptive U caps (§4.6 items 2, 3)
# --------------------------------------------------------------------------- #


def rank_overrides(base: dict, omega: float) -> dict:
    """Run 2's psiC overrides plus §0.3's labelled switch (the label gains
    ``loss=rank,rank_k=3,rank_omega=<omega>``)."""
    return {**base, "loss": LOSS_RANK, "rank_k": RANK_K, "rank_omega": float(omega)}


def make_head(seed: int, overrides: dict, spe: float):
    """offline-psi's ``new_policy``: run 2's psiC construction, phi from
    ``phi_generator(seed)`` -- the same initial phi for every omega."""
    pol, phi_seed = harness()["OP"].new_policy(seed, overrides, spe)
    return pol, phi_seed


def u_caps(head, inp: dict, positions: list[int], seed: int, rr: dict, n_boot: int):
    """offline-psi PREREG §5's FIFO-world cap (``score_matrices``, ``replay_rates``,
    B0's ``boot_rule``) over ``positions`` of U; with rho(psi, G) when asked."""
    OP = harness()["OP"]
    vp = torch.tensor(positions, dtype=torch.long)
    docs = [inp["docs"][p] for p in positions]
    W = OP.B0.boot_weights(len(docs), seed, n_boot)
    P = OP.score_matrices(head, inp["S"][vp], inp["C"][vp])
    fifo = [rr["fifo"][p] for p in positions]
    ff = [rr["factfiller"][p] for p in positions]
    out = OP.B0.boot_rule(W, OP.replay_rates(docs, list(P)), fifo, ff)
    return out, P, W


def rho_psi_g_val(P, inp: dict, positions: list[int], W) -> dict:
    OP = harness()["OP"]
    docs = [inp["docs"][p] for p in positions]
    Gv = torch.stack([OP.target_of(inp["blob"]["Delta"][d.doc_id]) for d in docs])
    return OP.rho_psi_g(P, Gv, W)


# --------------------------------------------------------------------------- #
# files, events, the stage prologue
# --------------------------------------------------------------------------- #


def wtag(omega: float) -> str:
    return f"omega{float(omega):g}"


def seed_dir(out_root: Path, seed: int) -> Path:
    return Path(out_root) / f"seed{seed}"


def fit_file(out_root: Path, seed: int, omega: float) -> Path:
    return seed_dir(out_root, seed) / f"fit-{wtag(omega)}.json"


def phi_file(out_root: Path, seed: int, omega: float, what: str) -> Path:
    return seed_dir(out_root, seed) / f"phi-{wtag(omega)}-{what}.pt"


def selection_file(out_root: Path, seed: int) -> Path:
    return seed_dir(out_root, seed) / "selection.json"


def eval_out(out_root: Path, seed: int) -> Path:
    return Path(out_root) / "heldout" / f"seed{seed}" / "psiR2a.jsonl.gz"


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def event(out_root: Path, stage: str, what: str, **kw) -> None:
    """R-C6: an append-only stage log (``events.jsonl``)."""
    Path(out_root).mkdir(parents=True, exist_ok=True)
    row = {"utc": utc(), "t": time.time(), "pid": os.getpid(), "stage": stage}
    row.update(what=what, **kw)
    with open(Path(out_root) / "events.jsonl", "a") as fh:
        fh.write(json.dumps(row) + "\n")


def read_events(out_root: Path) -> list[dict]:
    p = Path(out_root) / "events.jsonl"
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def sha256_file(p) -> str:
    return C.sha256_file(p)


def state_sha(state: dict) -> str:
    return harness()["OP"].state_sha(state)


def write_json(p: Path, doc: dict) -> str:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    return sha256_file(p)


def _strip_tune(r: dict) -> dict:
    """§4.3: the selection reads hits only; no accuracy statistic is computed on TUNE.
    TUNE records are written without the answer and loss fields."""
    return {k: v for k, v in r.items() if k not in ("answers", "real_sum", "real_n")}


def open_stage(stage: str, doc: dict) -> dict:
    """R-SEAL (a)-(c) for this process, R-C3's ``record`` refusal, the harness load
    and the rebinding; the seal state goes into ``doc["controls"]``."""
    ctl = doc.setdefault("controls", {})
    seal = SS.install("2a", stage)
    ctl["R-SEAL.install"] = {**seal, "ok": True}
    ctl["R-C3.record_refused"] = C.forbid_constants_record()
    harness()
    leaks = C.seal_leaks()
    ctl["R-SEAL.leaks"] = {"leaks": leaks, "ok": not leaks}
    ctl["prereg_text"] = prereg_check()
    doc["tree"] = {"head": C.git_head(), "dirty": C.git_dirty_paths()}
    return ctl


def close_stage(doc: dict, named: dict | None = None) -> None:
    doc["controls"]["R-SEAL.census"] = SS.census(named)


def finish(doc: dict, path: Path) -> int:
    ctl = doc.get("controls", {})
    failed = sorted(
        k for k, v in ctl.items() if not (isinstance(v, dict) and v.get("ok"))
    )
    doc["controls_failed"] = failed
    code = int(Exit.DID_NOT_RUN) if failed or doc.get("error") else int(Exit.OK)
    if doc.get("smoke"):
        code = int(Exit.DID_NOT_RUN)
        doc.setdefault("notes", []).append("SMOKE: never the run; nothing is read")
    doc["exit"] = code
    doc["status"] = "ok" if code == 0 else "did_not_run"
    doc["finished_utc"] = utc()
    write_json(path, doc)
    return code


def substrate_checks(seed: int, ctl: dict, when: str) -> None:
    """R-C1 / R-C3: the substrate manifest (``shasum -c`` rc read directly; its own
    sha256) and the substrate checkpoint's pin."""
    R1 = harness()["R1"]
    sub = R1.prereg_config(seed, harness()["OP"].CKPT).substrate
    ctl[f"R-C1.manifest_{when}"] = C.manifest_check(
        sub.manifest, R1.MANIFEST_SHA256, sub.root
    )
    ctl[f"R-C3.substrate_pin_{when}"] = C.checkpoint_pin(sub.checkpoint, sub.sha256)


def load_inputs(seed: int, ctl: dict, pins) -> dict:
    """§4.1 / R-C1: diagnostic A's ``load_inputs`` (gestalts and Delta, sha-checked
    against their pins), and run 2's psiC config (overrides, epoch, lr)."""
    CV, OP = harness()["CV"], harness()["OP"]
    inp = CV.load_inputs(seed)
    ctl["R-C1.inputs"] = {**inp["c1"], "ok": bool(inp["c1"]["ok"])}
    pins.pin(CV.OP_OUT / f"seed{seed}" / "gestalts.pt")
    pins.pin(OP.delta_path(seed))
    ov, spe, lr = CV.run2_conf(seed)
    pins.pin(OP.run2_psic_config(OP.REFS, seed)["path"])
    inp.update(overrides=ov, spe=spe, lr=lr)
    return inp


def check_u_only(inp: dict) -> dict:
    """U documents only in a fit (§4.1): no id in EVAL or a sealed range."""
    R1 = harness()["R1"]
    ids = set(inp["ids"])
    bad = {
        "EVAL": len(ids & set(range(*R1.EVAL))),
        **{k: len(ids & set(range(*v))) for k, v in SS.sealed_ranges().items()},
    }
    return {"overlaps": bad, "ok": not any(bad.values())}


# --------------------------------------------------------------------------- #
# stage: fit (--rc2 | --omega w)
# --------------------------------------------------------------------------- #


def rc2_seed(seed: int, out_root: Path) -> int:
    """R-C2: 2a's fit path with ``loss="mc"`` on cf_all and cf_live reproduces
    offline-psi's recorded U-val L_MC at step 3000, ``==`` on the float."""
    out_root = Path(out_root)
    path = seed_dir(out_root, seed) / "rc2.json"
    doc: dict[str, Any] = {"run_id": RUN_ID, "stage": "fit", "part": "rc2", "seed": seed}
    doc["started_utc"] = utc()
    pins = C.Pins()
    try:
        ctl = open_stage("fit", doc)
        event(out_root, "fit", "start", seed=seed, part="rc2")
        OP, CV = harness()["OP"], harness()["CV"]
        inp = load_inputs(seed, ctl, pins)
        ctl["U_only"] = check_u_only(inp)
        model, _p, _sha = OP.load_frozen(OP.R1.prereg_config(seed, OP.CKPT))
        rows = {}
        for arm in RC2_ARMS:
            pol, _ = make_head(seed, inp["overrides"], inp["spe"])
            head = pol.head
            opt = OP.phi_optimizer(model, head, inp["lr"])
            data = {
                "s": inp["S"],
                "c": inp["C"],
                **OP.fit_tensors(inp["blob"], inp["docs"], arm),
            }
            r = fit_loop(
                head,
                opt,
                data,
                inp["train"],
                seed=seed,
                loss=LOSS_MC,
                omega=None,
                cap=RC2_STEP,
                every=None,
                save_steps=(),
                stop_on_convergence=False,
            )
            got = OP.val_loss(
                head, inp["S"], inp["C"], data["G"], data["mask"], inp["val"]
            )
            pins.pin(CV.OP_OUT / f"measure-seed{seed}.json")
            want = CV.op_history_value(seed, arm, RC2_STEP)
            rows[arm] = {
                "val_l_mc_step3000": got,
                "offline_psi_recorded": want,
                "equal": got == want,
                "seconds": r["seconds"],
                "rc4_n_bad": r["rc4_n_bad"],
            }
        ctl["R-C2"] = {"arms": rows, "ok": all(v["equal"] for v in rows.values())}
        ctl["R-C1.pins_end"] = pins.recheck()
        close_stage(doc, {"U": (64, 1088)})
    except Exception as e:  # did not run: recorded, never a pass
        import traceback

        doc["error"] = f"{type(e).__name__}: {e}"
        doc["traceback"] = traceback.format_exc()
    code = finish(doc, path)
    event(out_root, "fit", "end", seed=seed, part="rc2", exit=code)
    return code


def fit_seed_omega(
    seed: int,
    omega: float,
    out_root: Path,
    *,
    cap: int = STEP_CAP,
    smoke: bool = False,
    n_boot: int = N_BOOT,
) -> int:
    """§4.2: one grid configuration. Fit on U-train (cf_all pairs), J every 250 steps,
    CONVERGED / CAP-HIT, phi at ``K_conv`` (and 3000, 10,000 where reached); §4.6
    items 1-3 (descriptive): the J trajectory, the FIFO-world caps on U-val and
    U-train, rho(psi, G) on U-val."""
    if float(omega) not in OMEGAS:
        raise C.ControlFailure(f"omega={omega} is not in the registered grid {OMEGAS}")
    out_root = Path(out_root)
    path = fit_file(out_root, seed, omega)
    if path.exists():
        raise C.ControlFailure(f"{path} exists: a fit is never re-run into the same root")
    doc: dict[str, Any] = {
        "run_id": RUN_ID,
        "stage": "fit",
        "seed": seed,
        "omega": float(omega),
        "k": RANK_K,
        "cap": cap,
        "smoke": smoke,
        "started_utc": utc(),
    }
    pins = C.Pins()
    try:
        ctl = open_stage("fit", doc)
        event(out_root, "fit", "start", seed=seed, omega=float(omega))
        OP = harness()["OP"]
        substrate_checks(seed, ctl, "start")
        inp = load_inputs(seed, ctl, pins)
        ctl["U_only"] = check_u_only(inp)
        model, _payloads, _sha = OP.load_frozen(OP.R1.prereg_config(seed, OP.CKPT))
        ov = rank_overrides(inp["overrides"], omega)
        pol, phi_seed = make_head(seed, ov, inp["spe"])
        head = pol.head
        doc["policy"] = {
            "overrides": ov,
            "label": loop.arm_label("rsr", ov),
            "config": {k: str(v) for k, v in vars(pol.config).items()},
            "phi_seed": phi_seed,
            "phi_init_sha256": state_sha(head.state_dict()),
            "steps_per_epoch": inp["spe"],
            "lr": inp["lr"],
        }
        opt = OP.phi_optimizer(model, head, inp["lr"])
        only = OP.optimizer_holds_only_phi(opt, head)
        ctl["R-C3.frozen"] = {
            "optimizer": only,
            "head_is_content_bilinear": isinstance(head, BilinearValueHead),
            "no_model_grad": C.no_grads(model),
            "ok": bool(only["ok"]) and isinstance(head, BilinearValueHead),
        }
        data = {
            "s": inp["S"],
            "c": inp["C"],
            **OP.fit_tensors(inp["blob"], inp["docs"], ARM),
        }
        r = fit_loop(
            head, opt, data, inp["train"], seed=seed, loss=LOSS_RANK, omega=omega, cap=cap
        )
        ctl["R-C3.frozen"]["no_model_grad_after"] = C.no_grads(model)
        ctl["R-C3.frozen"]["ok"] = (
            ctl["R-C3.frozen"]["ok"] and ctl["R-C3.frozen"]["no_model_grad_after"]["ok"]
        )
        ctl["R-C4.fit"] = {
            "n_bad": r["rc4_n_bad"],
            "bad_steps": r["rc4_bad_steps"],
            "ok": r["rc4_n_bad"] == 0,
        }
        files = {}
        torch.save(r["state"], phi_file(out_root, seed, omega, "kconv"))
        files["kconv"] = {
            "step": r["k_conv"],
            "path": str(phi_file(out_root, seed, omega, "kconv")),
            "phi_sha256": state_sha(r["state"]),
        }
        for st, snap in sorted(r["saved"].items()):
            p = phi_file(out_root, seed, omega, f"step{st}")
            torch.save(snap, p)
            files[f"step{st}"] = {
                "step": st,
                "path": str(p),
                "phi_sha256": state_sha(snap),
            }
        doc["fit"] = {k: v for k, v in r.items() if k not in ("state", "saved")}
        doc["phi"] = files
        # §4.6 items 2, 3 (descriptive): FIFO-world caps on U-val, U-train; rho on U-val
        rr = OP.reference_rates(inp["docs"], seed)
        desc: dict[str, Any] = {}
        for name, st in [
            ("kconv", r["state"]),
            *[(f"step{s}", v) for s, v in sorted(r["saved"].items())],
        ]:
            h2 = BilinearValueHead(OP.D_MODEL, generator=torch.Generator().manual_seed(0))
            h2.load_state_dict(st)
            val, Pv, Wv = u_caps(h2, inp, inp["val"], seed, rr, n_boot)
            row = {"U_val": val, "U_val_label": "spent for selection; descriptive here"}
            if name == "kconv":
                row["rho_psi_G_U_val"] = rho_psi_g_val(Pv, inp, inp["val"], Wv)
                row["U_train"], _P, _W = u_caps(h2, inp, inp["train"], seed, rr, n_boot)
                gv = data["G"][torch.tensor(inp["val"])]
                mv = data["mask"][torch.tensor(inp["val"])]
                Pm = OP.psi_matrix(
                    h2,
                    inp["S"][torch.tensor(inp["val"])],
                    inp["C"][torch.tensor(inp["val"])],
                )
                row["argmin_agreement_U_val"] = argmin_agreement(
                    *as_groups(Pm.detach(), gv, mv)
                )
            desc[name] = row
        doc["descriptive"] = desc
        substrate_checks(seed, ctl, "end")
        ctl["R-C1.pins_end"] = pins.recheck()
        close_stage(doc, {"U": (64, 1088)})
    except Exception as e:  # did not run: recorded, never a pass
        import traceback

        doc["error"] = f"{type(e).__name__}: {e}"
        doc["traceback"] = traceback.format_exc()
    code = finish(doc, path)
    event(out_root, "fit", "end", seed=seed, omega=float(omega), exit=code)
    return code


# --------------------------------------------------------------------------- #
# stage: select (TUNE's first read)
# --------------------------------------------------------------------------- #


def fifo_hits(docs: list, m: int) -> list[list]:
    """§4.3 ``hit_FIFO``: model-free ``headroom.simulate`` with ``FIFOPolicy``."""
    return [
        [(q["gap"], q["hit"]) for q in simulate(d, FIFOPolicy(), m)["queries"]]
        for d in docs
    ]


def tune_caps(hits: dict[str, list], seed: int, m: int, n_boot: int = N_BOOT) -> dict:
    """§4.3: run 2's ceiling share per omega, all queries pooled per document, paired
    across the omegas, FIFO and FactFiller (``R1.boot_weights(512, seed, 2000)``)."""
    R1, R2 = harness()["R1"], harness()["R2"]
    n = len(hits["fifo"])
    W = R1.boot_weights(n, seed, n_boot)
    sel = R1.buckets_for(m)["all"]
    per = {k: R2.hit_counts(v, sel) for k, v in hits.items()}
    out = {}
    for w in OMEGAS:
        out[wtag(w)] = {
            "cap": R2.ceiling(per, wtag(w), W),
            "hit": R1.level(per[wtag(w)], W),
        }
    out["fifo_hit"] = R1.level(per["fifo"], W)
    out["factfiller_hit"] = R1.level(per["factfiller"], W)
    out["n_documents"] = n
    return out


def choose(caps: dict) -> dict:
    """§4.3: the omega with the highest cap point; ties to the smaller omega. A seed with
    every cap point None is exit 3."""
    best = None
    for w in sorted(OMEGAS):
        pt = caps[wtag(w)]["cap"]["point"]
        if pt is None or not _finite(pt):
            continue
        if best is None or pt > best[1]:
            best = (w, pt)
    if best is None:
        return {"omega": None, "ok": False, "why": "every cap point is None"}
    return {"omega": best[0], "cap_point": best[1], "ok": True}


def tune_closure(seed: int) -> dict:
    """R-SEAL (e): run 1's C0 logic on TUNE, before any model loads: disjoint from
    every ``USED_RANGES`` entry and from POOL, upper bound <= ``GEN_SEED_K``, closed
    under the seed's vocabulary map. Also recorded: disjoint from EVAL and EVAL2."""
    R1, OP = harness()["R1"], harness()["OP"]
    import dataclasses

    tune = SS.sealed_ranges()[TUNE_NAME]
    cfg = dataclasses.replace(R1.prereg_config(seed, OP.CKPT), eval_range=tune)
    try:
        c0 = R1.check_c0(cfg)
    except R1.ControlFailure as e:
        return {"ok": False, "error": str(e)}
    extra = {
        "disjoint_EVAL": not R1._overlap(tune, R1.EVAL),
        "disjoint_EVAL2": not R1._overlap(tune, SS.sealed_ranges()[EVAL2_NAME]),
    }
    return {
        **c0,
        **extra,
        "range": list(tune),
        "ok": bool(c0["ok"]) and all(extra.values()),
    }


def select_seed(
    seed: int, out_root: Path, *, n_boot: int = N_BOOT, tune_docs=None
) -> int:
    """§4.3: TUNE's first read. Each omega's phi at ``K_conv``, frozen, deployed on the
    512 TUNE documents (offline-psi's ``deploy_pass``, T_warm = 0); the cap; the
    selection, written and hashed before ``deploy`` can run. ``tune_docs`` is for the
    test fixture only (stand-in documents; TUNE is never generated then)."""
    out_root = Path(out_root)
    path = seed_dir(out_root, seed) / "select.json"
    if path.exists() or selection_file(out_root, seed).exists():
        raise C.ControlFailure(f"{path}: TUNE is read once per seed")
    doc: dict[str, Any] = {"run_id": RUN_ID, "stage": "select", "seed": seed}
    doc["started_utc"] = utc()
    pins = C.Pins()
    try:
        ctl = open_stage("select", doc)
        event(out_root, "select", "start", seed=seed)
        OP, R1, R2 = harness()["OP"], harness()["R1"], harness()["R2"]
        fits = {}
        for w in OMEGAS:
            f = json.loads(fit_file(out_root, seed, w).read_text())
            fits[wtag(w)] = f
        ctl["R-C6.fits_done"] = {
            "exits": {k: v.get("exit") for k, v in fits.items()},
            "ok": all(v.get("exit") == 0 for v in fits.values()),
        }
        if not ctl["R-C6.fits_done"]["ok"]:
            raise C.ControlFailure("a fit did not exit 0: TUNE is not opened")
        cfg = R1.prereg_config(seed, OP.CKPT)
        if tune_docs is None:
            ctl["R-SEAL.closure_TUNE"] = tune_closure(seed)  # before any model loads
            if not ctl["R-SEAL.closure_TUNE"]["ok"]:
                raise C.ControlFailure("TUNE closure failed: TUNE is not read further")
            tune = SS.sealed_ranges()[TUNE_NAME]
            docs = R1.docs_by_id(cfg, list(range(*tune)))
        else:
            docs = list(tune_docs)
            doc["fixture"] = True
        substrate_checks(seed, ctl, "start")
        inp_ov, spe, _lr = harness()["CV"].run2_conf(seed)
        model, _payloads, _sha = OP.load_frozen(cfg)
        vmap = R1.vocab_map(cfg)
        m = cfg.memory_slots
        hits: dict[str, list] = {}
        passes = {}
        finite = True
        for w in OMEGAS:
            f = fits[wtag(w)]
            ph = Path(f["phi"]["kconv"]["path"])
            st = torch.load(ph, weights_only=True)
            if state_sha(st) != f["phi"]["kconv"]["phi_sha256"]:
                raise C.ControlFailure(f"{ph} does not hash to its fit record")
            ov = rank_overrides(inp_ov, w)

            def make(st=st, ov=ov):
                return OP.deploy_policy(st, seed, ov, spe)

            r = OP.deploy_pass(model, cfg, docs, vmap, make)
            recs = [_strip_tune(x) for x in r["records"]]
            p = Path(out_root) / "tune" / f"seed{seed}" / f"{wtag(w)}.jsonl.gz"
            if p.exists():
                raise C.ControlFailure(f"{p} exists: TUNE is read once")
            passes[wtag(w)] = {
                "records_sha256": OP.write_records(p, recs),
                "path": str(p),
                "residency": r["residency"],
                "timing": r["timing"],
                "k_conv": f["fit"]["k_conv"],
                "label": f["fit"]["label"],
                "phi_sha256": f["phi"]["kconv"]["phi_sha256"],
            }
            finite = finite and all(
                bool(torch.isfinite(d[2]).all()) for d in r["decisions"]
            )
            hits[wtag(w)] = [x["hits"] for x in r["records"]]
        hits["fifo"] = fifo_hits(docs, m)
        hits["factfiller"] = R2.factfiller_hits(docs, seed, m)
        # the model-free references, kept so the caps recompute with TUNE sealed
        refp = Path(out_root) / "tune" / f"seed{seed}" / "references.json"
        doc["references_sha256"] = write_json(
            refp,
            {
                "doc": [d.doc_id for d in docs],
                "fifo": hits["fifo"],
                "factfiller": hits["factfiller"],
            },
        )
        caps = tune_caps(hits, seed, m, n_boot)
        pick = choose(caps)
        doc["passes"] = passes
        doc["caps"] = caps
        doc["n_documents"] = len(docs)
        doc["doc_range"] = [docs[0].doc_id, docs[-1].doc_id] if docs else None
        ctl["R-C7.tune"] = {
            "mismatched": {k: v["residency"]["n_mismatched"] for k, v in passes.items()},
            "ok": all(v["residency"]["n_mismatched"] == 0 for v in passes.values()),
        }
        ctl["R-C4.tune_psi_finite"] = {"ok": bool(finite)}
        ctl["selection"] = pick
        if pick["ok"]:
            w = pick["omega"]
            src = phi_file(out_root, seed, w, "kconv")
            dst = seed_dir(out_root, seed) / "phi-selected.pt"
            shutil.copyfile(src, dst)
            sel = {
                "seed": seed,
                "omega_s": w,
                "k_conv": passes[wtag(w)]["k_conv"],
                "convergence": passes[wtag(w)]["label"],
                "phi_sha256": state_sha(torch.load(dst, weights_only=True)),
                "phi_path": str(dst),
                "cap_point": pick["cap_point"],
                "rule": "highest TUNE cap point; ties to the smaller omega (§4.3)",
                "written_utc": utc(),
            }
            sha = write_json(selection_file(out_root, seed), sel)
            (seed_dir(out_root, seed) / "selection.sha256").write_text(sha + "\n")
            doc["selection"] = {**sel, "file_sha256": sha}
        substrate_checks(seed, ctl, "end")
        ctl["R-C1.pins_end"] = (
            pins.recheck() if pins.first else {"ok": True, "n_files": 0}
        )
        close_stage(doc, {"EVAL": tuple(R1.EVAL)})
    except Exception as e:  # did not run: recorded, never a pass
        import traceback

        doc["error"] = f"{type(e).__name__}: {e}"
        doc["traceback"] = traceback.format_exc()
    code = finish(doc, path)
    event(out_root, "select", "end", seed=seed, exit=code)
    return code


# --------------------------------------------------------------------------- #
# stage: deploy (EVAL, already used; the headline)
# --------------------------------------------------------------------------- #


def selection_guard(out_root: Path, seed: int) -> dict:
    """§4.3 / R-C6 before EVAL is read: the selection file exists and hashes to the
    sha written beside it; offline-psi's ``deploy_guard`` (the frozen phi on disk is
    the selected one; the EVAL output does not exist yet)."""
    OP = harness()["OP"]
    sf = selection_file(out_root, seed)
    if not sf.exists():
        raise OP.Refused(f"{sf} is missing: EVAL is read only by the selected phi")
    want = (seed_dir(out_root, seed) / "selection.sha256").read_text().strip()
    got = sha256_file(sf)
    if got != want:
        raise OP.Refused(f"{sf} hashes to {got}, not the recorded {want}")
    sel = json.loads(sf.read_text())
    g = OP.deploy_guard(sel, Path(sel["phi_path"]), eval_out(out_root, seed))
    return {"selection": sel, "selection_sha256": got, "deploy_guard": g, "ok": True}


def deploy_seed(seed: int, out_root: Path, *, n_boot: int = N_BOOT) -> int:
    """§4.4 / §4.6 item 4: the selected phi, frozen, deployed once on EVAL."""
    out_root = Path(out_root)
    path = seed_dir(out_root, seed) / "deploy.json"
    doc: dict[str, Any] = {"run_id": RUN_ID, "stage": "deploy", "seed": seed}
    doc["started_utc"] = utc()
    pins = C.Pins()
    try:
        ctl = open_stage("deploy", doc)
        OP, R1, R2, CV = (harness()[k] for k in ("OP", "R1", "R2", "CV"))
        ctl["R-C6.guard"] = selection_guard(out_root, seed)
        event(out_root, "deploy", "start", seed=seed)
        sel = ctl["R-C6.guard"]["selection"]
        cfg = R1.prereg_config(seed, OP.CKPT)
        substrate_checks(seed, ctl, "start")
        refs = OP.REFS
        hd2 = Path(refs.run2_out) / "heldout" / f"seed{seed}"
        for n in ("fifo", "psiC", *OP.STEP1.RANDOM):
            pins.pin(hd2 / f"{n}.jsonl.gz")
        pins.pin(Path(refs.es_out) / "eval" / f"seed{seed}" / "es.jsonl.gz")
        pins.pin(CV.OP_OUT / "heldout" / f"seed{seed}" / "cf_all.jsonl.gz")
        dl = R2.read_delta(seed)
        ctl["R-C9"] = dl
        inp_ov, spe, _lr = CV.run2_conf(seed)
        ov = rank_overrides(inp_ov, sel["omega_s"])
        st = torch.load(sel["phi_path"], weights_only=True)
        model, _payloads, _sha = OP.load_frozen(cfg)
        docs = R1.docs_by_id(cfg, list(cfg.eval_ids()))

        def make():
            return OP.deploy_policy(st, seed, ov, spe)

        r = OP.deploy_pass(model, cfg, docs, R1.vocab_map(cfg), make)
        eo = eval_out(out_root, seed)
        doc["deploy"] = {
            "n_documents": len(docs),
            "doc_range": [docs[0].doc_id, docs[-1].doc_id],
            "records_sha256": OP.write_records(eo, r["records"]),
            "residency": r["residency"],
            "timing": r["timing"],
            "omega_s": sel["omega_s"],
            "phi_sha256": sel["phi_sha256"],
            "label_of_arm": loop.arm_label("rsr", {**ov, "t_warm": 0.0}),
        }
        torch.save(
            {
                "doc": torch.tensor([d[0] for d in r["decisions"]]),
                "step": torch.tensor([d[1] for d in r["decisions"]]),
                "psi": torch.stack([d[2] for d in r["decisions"]]),
                "age": torch.stack([d[4] for d in r["decisions"]]),
            },
            eo.parent / "psiR2a.decisions.pt",
        )
        ctl["R-C7.eval"] = {**r["residency"], "ok": r["residency"]["n_mismatched"] == 0}
        ctl["R-C4.eval_psi_finite"] = {
            "n": len(r["decisions"]),
            "ok": bool(r["decisions"])
            and all(bool(torch.isfinite(d[2]).all()) for d in r["decisions"]),
        }
        # §4.4 headline and §4.6 item 4, through offline-psi's score_seed (its arm
        # "cf_all" IS this head; renamed below to psiR2a so nothing reads otherwise)
        sc = OP.score_seed(
            r["records"], r["decisions"], seed, refs, n_boot, dl.get("delta")
        )
        ctl["R-C8"] = {**sc["C6"], "ok": bool(sc["C6"]["ok"])}
        doc["heldout"] = rename_arm(sc)
        doc["heldout"]["extra"] = eval_extra(r["records"], docs, seed, cfg, n_boot)
        substrate_checks(seed, ctl, "end")
        ctl["R-C1.pins_end"] = pins.recheck()
        close_stage(doc, {"EVAL": tuple(R1.EVAL)})
    except Exception as e:  # did not run: recorded, never a pass
        import traceback

        doc["error"] = f"{type(e).__name__}: {e}"
        doc["traceback"] = traceback.format_exc()
    code = finish(doc, path)
    event(out_root, "deploy", "end", seed=seed, exit=code)
    return code


def rename_arm(x: Any, old: str = "cf_all", new: str = "psiR2a") -> Any:
    """offline-psi's ``score_seed`` names the deployed head ``cf_all``; here that head
    is 2a's. Keys only (values are numbers and labels)."""
    if isinstance(x, dict):
        return {
            (k.replace(old, new) if isinstance(k, str) else k): rename_arm(v, old, new)
            for k, v in x.items()
        }
    if isinstance(x, list):
        return [rename_arm(v, old, new) for v in x]
    return x


def eval_extra(recs: list[dict], docs: list, seed: int, cfg, n_boot: int) -> dict:
    """§4.6 item 4 beyond offline-psi's secondary block: psiR2a - offline-psi cf_all
    (the loss-only contrast, PM §12 item 2), hit rate and the FactFiller ceiling
    share."""
    OP, R1, R2, CV = (harness()[k] for k in ("OP", "R1", "R2", "CV"))
    m = cfg.memory_slots
    W = R1.boot_weights(len(recs), seed, n_boot)
    out: dict[str, Any] = {}
    cf = R1.read_pass(CV.OP_OUT / "heldout" / f"seed{seed}", "cf_all")
    out["pairing_offline_psi_cf_all"] = OP.pairing(recs, cf)
    if out["pairing_offline_psi_cf_all"]["ok"]:
        bk = R1.buckets_for(m)
        out["minus_offline_psi_cf_all"] = {
            b: R1.contrast(
                {
                    "psiR2a": R1.answer_counts(recs, bk[b]),
                    "op": R1.answer_counts(cf, bk[b]),
                },
                "psiR2a",
                "op",
                W,
            )
            for b in ("all", "gap_2_to_M", "gap_gt_M")
        }
    fifo = R1.read_pass(Path(OP.REFS.run2_out) / "heldout" / f"seed{seed}", "fifo")
    hits = {
        "psiR2a": [r["hits"] for r in recs],
        "fifo": [r["hits"] for r in fifo],
        "factfiller": R2.factfiller_hits(docs, seed, m),
    }
    hb = {}
    for b in ("all", "gap_2_to_M", "gap_gt_M"):
        per = {k: R2.hit_counts(v, R1.buckets_for(m)[b]) for k, v in hits.items()}
        hb[b] = {
            "hit": {k: R1.level(p, W) for k, p in per.items()},
            "ceiling_share": R2.ceiling(per, "psiR2a", W),
        }
    out["hit_rate"] = hb
    return out


# --------------------------------------------------------------------------- #
# stage: launch-checks (G-C3, R-SEAL (d), the frozen manifest)
# --------------------------------------------------------------------------- #

QUESTION = (
    "PREREG rsr-run3-ranking §4 (2a): fitted offline on B0-CF's uncensored FIFO-world "
    "LOO returns with a within-step pairwise ranking loss instead of L9's squared "
    "error, and selected on the sealed TUNE split by the cap, does run 2's "
    "content-only bilinear psi_hat beat FIFO when deployed frozen on the already-used "
    "EVAL?"
)
FALSIFIER = (
    "NONE as a falsifier of the thesis; this is the PLAN §2 phase-2 kill-gate input: "
    "2a WIN on all 3 seeds gates 2b; anything else stops phase 2 at Gate A."
)


def prereg_passage(start: str, stop: str) -> str:
    text = PREREG_PATH.read_text()
    i = text.index(start)
    return text[i : text.index(stop, i + len(start))].strip()


def expectation_text() -> str:
    return prereg_passage("- 2a fit:", "- If 2a is all-WIN")


def manifest_config(seeds) -> dict:
    OP, R1 = harness()["OP"], harness()["R1"]
    return {
        "run_id": RUN_ID,
        "prereg": PREREG_REL,
        "prereg_commit": PREREG_COMMIT,
        "seeds": list(seeds),
        "omegas": list(OMEGAS),
        "rank_k": RANK_K,
        "convergence": {
            "j_every": J_EVERY,
            "min_step": CONV_MIN_STEP,
            "window": CONV_WINDOW,
            "rel": CONV_REL,
            "cap": STEP_CAP,
            "j_at_step_0": True,
        },
        "save_steps": list(SAVE_STEPS),
        "pairs": ARM,
        "fit_set": "U-train (offline-psi u_split)",
        "fit_batch": OP.FIT_BATCH,
        "rc2_arms": list(RC2_ARMS),
        "rc2_step": RC2_STEP,
        "tune_from": SS.SEALED_SOURCE,
        "sealed": {k: list(v) for k, v in SS.sealed_ranges().items()},
        "stage_seals": {k: list(v) for k, v in SS.STAGES["2a"].items()},
        "eval_range": list(R1.EVAL),
        "eval_batch": R1.EVAL_BATCH,
        "memory_slots": R1.M_SLOTS,
        "steps_per_stream": R1.S_STEPS,
        "device": "cpu",
        "threads_per_process": 1,
        "n_boot": N_BOOT,
        "boot_seed_base": R1.BOOT_SEED_BASE,
        "delta_sha": {str(k): v for k, v in OP.DELTA_SHA.items()},
        "gestalt_sha": {str(k): v for k, v in harness()["CV"].GESTALT_SHA.items()},
        "substrate": {str(s): v for s, v in R1.SUBSTRATE.items()},
        "manifest_sha256": R1.MANIFEST_SHA256,
        "expected": expectation_text(),
        "falsifier": FALSIFIER,
    }


def launch_checks(out_root: Path, *, suite_log: Path, suite_rc: Path, suite_head: Path):
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rep: dict[str, Any] = {"run_id": RUN_ID, "launched_utc": utc(), "controls": {}}
    ctl = open_stage("launch-checks", rep)
    rep["head"] = rep["tree"]["head"]
    try:
        rc_txt = Path(suite_rc).read_text().strip()
        rc = int(rc_txt) if rc_txt.lstrip("-").isdigit() else None
        cen = C.suite_census(Path(suite_log).read_text(), rc)
        sh = Path(suite_head).read_text().strip()
        cen.update(head=sh, head_is_launch_head=sh == rep["head"], log=str(suite_log))
        cen["ok"] = bool(cen["ok"]) and cen["head_is_launch_head"]
    except OSError as e:
        cen = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    ctl["G-C3"] = cen
    ctl["R-SEAL.search"] = SS.branch_search()
    ctl["clean_tree"] = {"dirty": rep["tree"]["dirty"], "ok": rep["tree"]["dirty"] == []}
    seeds = registered_seeds()
    L = C.ledger_mod()
    led = L.Ledger(RUN_ID, question=QUESTION, runs_root=out_root.parent)
    if led.path.parent != out_root:
        ctl["out_root"] = {"ok": False, "why": f"{out_root} is not runs/{RUN_ID}"}
    else:
        led.manifest(manifest_config(seeds))
        rep["manifest"] = {
            "path": str(out_root / "manifest.json"),
            "config_hash": led.doc["config_hash"],
            "manifest_written_utc": led.doc["manifest_written_utc"],
        }
    close_stage(rep)
    return finish(rep, out_root / "launch.json")


# --------------------------------------------------------------------------- #
# Amendment 2a-S: the timing smoke (U documents only)
# --------------------------------------------------------------------------- #


def time_smoke(seed: int, out: Path, *, steps: int = 50, n_deploy: int = 64) -> dict:
    """§4.8: ``steps`` fit steps per omega on ``seed``, one J evaluation, one
    ``n_deploy``-document deploy block on U documents (never TUNE or EVAL); plus the
    L_MC step (R-C2), the descriptive U caps, the manifest check and the input loads,
    each timed. Writes ``out/time-seed<s>.json``; never a result."""
    out = Path(out)
    doc: dict[str, Any] = {"run_id": RUN_ID, "stage": "time", "seed": seed, "smoke": True}
    doc["started_utc"] = utc()
    doc["loadavg_start"] = os.getloadavg()
    torch.set_num_threads(1)
    doc["threads"] = torch.get_num_threads()
    ctl = open_stage("time", doc)
    OP = harness()["OP"]
    pins = C.Pins()
    t = {}
    t0 = time.perf_counter()
    substrate_checks(seed, ctl, "start")
    t["manifest_check_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    inp = load_inputs(seed, ctl, pins)
    model, _p, _sha = OP.load_frozen(OP.R1.prereg_config(seed, OP.CKPT))
    t["inputs_and_model_s"] = time.perf_counter() - t0
    ctl["U_only"] = check_u_only(inp)
    per: dict[str, Any] = {}
    for w in [None, *OMEGAS]:
        loss = LOSS_MC if w is None else LOSS_RANK
        ov = inp["overrides"] if w is None else rank_overrides(inp["overrides"], w)
        pol, _ = make_head(seed, ov, inp["spe"])
        head = pol.head
        opt = OP.phi_optimizer(model, head, inp["lr"])
        data = {
            "s": inp["S"],
            "c": inp["C"],
            **OP.fit_tensors(inp["blob"], inp["docs"], ARM),
        }
        stamps: list[float] = []
        last = [time.perf_counter()]

        def on_step(_k, _h, stamps=stamps, last=last):
            now = time.perf_counter()
            stamps.append(now - last[0])
            last[0] = now

        last[0] = time.perf_counter()
        r = fit_loop(
            head,
            opt,
            data,
            inp["train"],
            seed=seed,
            loss=loss,
            omega=w,
            cap=steps,
            every=None,
            save_steps=(),
            stop_on_convergence=False,
            on_step=on_step,
        )
        row = {
            "steps": r["steps_done"],
            "step_s_median": float(torch.tensor(stamps).median()),
            "step_s_mean": sum(stamps) / len(stamps),
            "rc4_n_bad": r["rc4_n_bad"],
        }
        if w is not None:
            tj = time.perf_counter()
            j = full_objective(
                head, inp["S"], inp["C"], data["G"], data["mask"], inp["train"], w
            )
            row["J_s"] = time.perf_counter() - tj
            row["J"] = j
        per["mc" if w is None else wtag(w)] = row
    doc["fit"] = per
    # descriptive U caps (one configuration's worth)
    rr_t = time.perf_counter()
    rr = OP.reference_rates(inp["docs"], seed)
    t["reference_rates_s"] = time.perf_counter() - rr_t
    ut = time.perf_counter()
    u_caps(head, inp, inp["val"], seed, rr, N_BOOT)
    t["u_val_cap_s"] = time.perf_counter() - ut
    ut = time.perf_counter()
    u_caps(head, inp, inp["train"], seed, rr, N_BOOT)
    t["u_train_cap_s"] = time.perf_counter() - ut
    # one n_deploy-document deploy block on U documents (stand-ins; never TUNE/EVAL)
    cfg = OP.R1.prereg_config(seed, OP.CKPT)
    ddocs = [inp["docs"][p] for p in inp["val"][:n_deploy]]
    st = {k: v.clone() for k, v in head.state_dict().items()}
    ov = rank_overrides(inp["overrides"], OMEGAS[-1])
    dt = time.perf_counter()
    r = OP.deploy_pass(
        model,
        cfg,
        ddocs,
        OP.R1.vocab_map(cfg),
        lambda: OP.deploy_policy(st, seed, ov, inp["spe"]),
    )
    t["deploy_block_s"] = time.perf_counter() - dt
    t["deploy_block_docs"] = len(ddocs)
    t["deploy_block_residency_mismatched"] = r["residency"]["n_mismatched"]
    ft = time.perf_counter()
    fifo_hits(ddocs, cfg.memory_slots)
    harness()["R2"].factfiller_hits(ddocs, seed, cfg.memory_slots)
    t["references_block_s"] = time.perf_counter() - ft
    doc["timing"] = t
    doc["loadavg_end"] = os.getloadavg()
    close_stage(doc, {"U": (64, 1088)})
    doc["projection"] = projection(doc)
    finish(doc, out / f"time-seed{seed}.json")
    return doc


def projection(doc: dict) -> dict:
    """Amendment 2a-S's projection: 9 fits to the 200,000-step worst case (J at step 0
    and every 250 steps), 9 TUNE deploys of 512 documents, 3 EVAL deploys of 2048, and
    what else runs (R-C2's 2 x 3000 L_MC steps per seed, the descriptive U caps, the
    manifest checks). Serial CPU-seconds, and the wall time of the launch layout
    (9 fits concurrently, one slot each, then 3 selects, then 3 deploys)."""
    f, t = doc["fit"], doc["timing"]
    n_j = STEP_CAP // J_EVERY + 1
    per_fit = {}
    for w in OMEGAS:
        r = f[wtag(w)]
        per_fit[wtag(w)] = r["step_s_median"] * STEP_CAP + r["J_s"] * n_j
    worst_fit = max(per_fit.values())
    per_doc = t["deploy_block_s"] / t["deploy_block_docs"]
    ref_doc = t["references_block_s"] / t["deploy_block_docs"]
    tune_one = 512 * per_doc
    eval_one = 2048 * per_doc
    rc2_one = 2 * RC2_STEP * f["mc"]["step_s_median"]
    caps_one = t["u_val_cap_s"] + t["u_train_cap_s"] + t["reference_rates_s"]
    man = t["manifest_check_s"]
    io = t["inputs_and_model_s"]
    fits_cpu = sum(per_fit.values()) * len(registered_seeds())
    serial = (
        fits_cpu
        + 9 * (caps_one + 2 * man + io)
        + 9 * tune_one
        + 3 * 512 * ref_doc
        + 3 * eval_one
        + 3 * (rc2_one + io)
        + 6 * (2 * man + io)
    )
    wall = (
        max(worst_fit + caps_one + 2 * man + io, rc2_one + io)
        + (3 * tune_one + 512 * ref_doc + 2 * man + io)
        + (eval_one + 2 * man + io)
    )
    return {
        "per_fit_worst_case_s": per_fit,
        "worst_fit_s": worst_fit,
        "n_J_per_fit": n_j,
        "deploy_s_per_document": per_doc,
        "tune_deploy_one_s": tune_one,
        "eval_deploy_one_s": eval_one,
        "rc2_one_seed_s": rc2_one,
        "u_caps_one_config_s": caps_one,
        "serial_cpu_s": serial,
        "serial_cpu_h": serial / 3600,
        "wall_layout_s": wall,
        "wall_layout_h": wall / 3600,
        "layout": "9 fits + 3 R-C2 concurrently (12 slots), then 3 selects, then 3 "
        "deploys (3 slots each stage); 1 thread per process",
        "over_8h_serial": serial / 3600 > 8,
        "over_8h_wall_layout": wall / 3600 > 8,
    }


# --------------------------------------------------------------------------- #
# stage: finalize (ledger, claims.json, RESULTS.md); recompute
# --------------------------------------------------------------------------- #


def _read(p: Path) -> dict | None:
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def stage_docs(out_root: Path, seeds) -> dict:
    out = {"launch": _read(Path(out_root) / "launch.json"), "seeds": {}}
    for s in seeds:
        sd = seed_dir(out_root, s)
        out["seeds"][s] = {
            "rc2": _read(sd / "rc2.json"),
            "fits": {wtag(w): _read(fit_file(out_root, s, w)) for w in OMEGAS},
            "select": _read(sd / "select.json"),
            "deploy": _read(sd / "deploy.json"),
        }
    return out


def _ok(d: dict | None, key: str) -> bool:
    return bool(d and (d.get("controls") or {}).get(key, {}).get("ok"))


def order_check(events: list[dict], seed: int) -> dict:
    """R-C6: per seed, every fit ended (exit 0) before select started; select ended
    before deploy started; deploy started exactly once."""
    ev = [e for e in events if e.get("seed") == seed]

    def times(stage, what):
        return [e["t"] for e in ev if e["stage"] == stage and e["what"] == what]

    fit_end = times("fit", "end")
    sel_start, sel_end = times("select", "start"), times("select", "end")
    dep_start = times("deploy", "start")
    ok = (
        len(fit_end) >= len(OMEGAS) + 1
        and len(sel_start) == 1
        and len(sel_end) == 1
        and len(dep_start) == 1
        and max(fit_end) < sel_start[0]
        and sel_end[0] < dep_start[0]
    )
    return {
        "n_fit_end": len(fit_end),
        "n_select_start": len(sel_start),
        "n_deploy_start": len(dep_start),
        "ok": bool(ok),
    }


def summarize(out_root: Path, seeds) -> dict:
    st = stage_docs(out_root, seeds)
    events = read_events(out_root)
    launch = st["launch"]
    rc: dict[str, Any] = {}
    every_doc: list[dict] = [launch] if launch else []
    for s in seeds:
        sd = st["seeds"][s]
        every_doc += [d for d in (sd["rc2"], sd["select"], sd["deploy"]) if d]
        every_doc += [d for d in sd["fits"].values() if d]
    n_expected = 1 + len(seeds) * (3 + len(OMEGAS))

    def all_ok(prefix):
        rows = [
            (d.get("stage"), d.get("seed"), d.get("omega"), k, bool(v.get("ok")))
            for d in every_doc
            for k, v in (d.get("controls") or {}).items()
            if k.startswith(prefix)
        ]
        return {"n": len(rows), "failed": [r for r in rows if not r[-1]][:20]}

    def control(prefix, extra=True):
        a = all_ok(prefix)
        a["ok"] = a["n"] > 0 and not a["failed"] and bool(extra)
        return a

    rc["stage_docs"] = {
        "n": len(every_doc),
        "expected": n_expected,
        "ok": len(every_doc) == n_expected,
    }
    rc["R-SEAL"] = control("R-SEAL")
    rc["R-C1"] = control("R-C1")
    rc["R-C2"] = control(
        "R-C2", len([d for d in every_doc if d.get("part") == "rc2"]) == len(seeds)
    )
    rc["R-C3"] = control("R-C3")
    rc["R-C4"] = control("R-C4")
    grid_ok = True
    recompute_ok = True
    for s in seeds:
        sd = st["seeds"][s]
        for w in OMEGAS:
            f = sd["fits"][wtag(w)]
            sel = sd["select"]
            grid_ok = grid_ok and bool(
                f
                and f.get("fit", {}).get("history")
                and f["fit"].get("k_conv")
                and sel
                and (sel.get("caps") or {}).get(wtag(w))
            )
        sel = sd["select"]
        if sel and sel.get("caps"):
            recompute_ok = (
                recompute_ok and choose(sel["caps"]) == sel["controls"]["selection"]
            )
        else:
            recompute_ok = False
    rc["R-C5"] = {
        "grid_complete": grid_ok,
        "selection_recomputes": recompute_ok,
        "ok": grid_ok and recompute_ok,
    }
    orders = {s: order_check(events, s) for s in seeds}
    guards = {s: _ok(st["seeds"][s]["deploy"], "R-C6.guard") for s in seeds}
    rc["R-C6"] = {
        "order": orders,
        "guards": guards,
        "ok": all(o["ok"] for o in orders.values()) and all(guards.values()),
    }
    rc["R-C7"] = control("R-C7")
    rc["R-C8"] = control("R-C8")
    rc["R-C9"] = control("R-C9")
    rc["G-C3"] = {**((launch or {}).get("controls", {}).get("G-C3") or {})}
    rc["G-C3"]["ok"] = bool(rc["G-C3"].get("ok"))
    rc["prereg_text"] = control("prereg_text")
    exits = [d.get("exit") for d in every_doc]
    rc["stage_exits"] = {"exits": exits, "ok": all(e == 0 for e in exits) and bool(exits)}
    failed = sorted(k for k, v in rc.items() if not v.get("ok"))
    labels: dict[int, Any] = {}
    per_seed: dict[int, Any] = {}
    for s in seeds:
        dep = st["seeds"][s]["deploy"] or {}
        h = (dep.get("heldout") or {}).get("headline") or {}
        labels[s] = h.get("label")
        sel = st["seeds"][s]["select"] or {}
        per_seed[s] = {
            "headline": h,
            "selection": sel.get("selection"),
            "configs": {
                wtag(w): {
                    "label": (st["seeds"][s]["fits"][wtag(w)] or {})
                    .get("fit", {})
                    .get("label"),
                    "k_conv": (st["seeds"][s]["fits"][wtag(w)] or {})
                    .get("fit", {})
                    .get("k_conv"),
                    "tune_cap": ((sel.get("caps") or {}).get(wtag(w)) or {}).get("cap"),
                    "u_val_cap": (
                        (st["seeds"][s]["fits"][wtag(w)] or {}).get("descriptive") or {}
                    )
                    .get("kconv", {})
                    .get("U_val"),
                }
                for w in OMEGAS
            },
        }
    code = int(Exit.DID_NOT_RUN) if failed else int(Exit.OK)
    if code == 0:
        all_win = all(labels[s] == "WIN" for s in seeds)
        reading = (
            "2a WIN on all 3 seeds: Amendment 2b-S, then 2b (PREREG §4.5, §7)"
            if all_win
            else "2a not WIN on all 3 seeds: phase 2 stops (structurally hard); Gate A"
        )
        if any(
            not _finite(((per_seed[s]["headline"]) or {}).get("point")) for s in seeds
        ):
            code = int(Exit.FAIL)
            reading = "a reported statistic is not finite: defect (exit 1)"
    else:
        all_win = None
        reading = "a control failed: no label is read; Gate A as a control failure (§4.5)"
    shares = {
        s: ((st["seeds"][s]["deploy"] or {}).get("heldout") or {})
        .get("age1_share", {})
        .get("psiR2a", {})
        .get("share")
        for s in seeds
    }
    try:
        newcomer = harness()["R2"].newcomer_label(shares, seeds)
    except Exception as e:
        newcomer = f"not computed: {type(e).__name__}: {e}"
    return {
        "run_id": RUN_ID,
        "exit": code,
        "status": {0: "ok", 1: "failed", 3: "did_not_run"}[code],
        "controls": rc,
        "controls_failed": failed,
        "labels": labels if code == 0 else {s: "NOT READ" for s in seeds},
        "labels_raw_not_read": None if code == 0 else labels,
        "gate_to_2b": all_win,
        "reading": reading,
        "per_seed": per_seed,
        "newcomer_label": newcomer,
        "seeds_actually_run": [s for s in seeds if st["seeds"][s]["deploy"]],
    }


def _f(x, nd: int = 4) -> str:
    return "—" if not _finite(x) else f"{x:+.{nd}f}"


def _ci(c: dict | None) -> str:
    if not c:
        return "—"
    return f"{_f(c.get('point'))} [{_f(c.get('lo'))}, {_f(c.get('hi'))}]"


def render_results(sm: dict, out_root: Path, seeds, led_doc: dict) -> str:
    st = stage_docs(out_root, seeds)
    w: list[str] = []
    a = w.append
    a(f"# RESULTS: {RUN_ID} (RSR run 3, phase 2a)\n")
    a(
        f"Generated by `{EXPERIMENT} 2a finalize`; no number is typed. Written by a "
        f"model; not by Brendan. PREREG `{PREREG_COMMIT}`.\n"
    )
    a(
        f"- git: `{led_doc.get('provenance', {}).get('git_sha')}`; exit `{sm['exit']}` "
        f"({sm['status']}); seeds actually run {sm['seeds_actually_run']}"
    )
    a(f"- manifest config hash `{led_doc.get('config_hash')}`\n")
    a(f"**Reading (PREREG §4.5):** {sm['reading']}\n")
    a("## Headline per seed (EVAL, already used): acc.all(psiR-2a) - acc.all(fifo)\n")
    a("| seed | ω_s | K_conv | point [95% CI] | δ_seed | label |")
    a("| --- | --- | --- | --- | --- | --- |")
    for s in seeds:
        p = sm["per_seed"][s]
        sel = p.get("selection") or {}
        h = p.get("headline") or {}
        a(
            f"| {s} | {sel.get('omega_s')} | {sel.get('k_conv')} | {_ci(h)} | "
            f"{_f(h.get('delta'))} | {sm['labels'][s]} |"
        )
    a("\n## Every configuration fitted (3 ω x 3 seeds; §4.3, §4.6 items 1-2)\n")
    a(
        "| seed | ω | convergence | K_conv | TUNE cap [95% CI] "
        "| U-val FIFO-world cap (descriptive) |"
    )
    a("| --- | --- | --- | --- | --- | --- |")
    for s in seeds:
        for ww in OMEGAS:
            c = sm["per_seed"][s]["configs"][wtag(ww)]
            a(
                f"| {s} | {ww:g} | {c['label']} | {c['k_conv']} | {_ci(c['tune_cap'])} | "
                f"{_ci(c['u_val_cap'])} |"
            )
    a("\n## Controls (§4.7)\n")
    a("| id | ok |")
    a("| --- | --- |")
    for k, v in sm["controls"].items():
        a(f"| {k} | {v.get('ok')} |")
    a(f"\nNewcomer label (run 2 §5 thresholds, descriptive): {sm['newcomer_label']}\n")
    a("## Secondary (§4.6 item 4; descriptive, unadjusted, never gates)\n")
    for s in seeds:
        dep = st["seeds"][s]["deploy"] or {}
        h = dep.get("heldout") or {}
        sec = (h.get("secondary") or {}).get("accuracy") or {}
        ex = h.get("extra") or {}
        a(f"### seed {s}\n")
        for b, row in sec.items():
            a(
                f"- acc {b}: -fifo {_ci(row.get('minus_fifo'))}; -random "
                f"{_ci(row.get('minus_random'))}; "
                f"-psiC(run 2) {_ci(row.get('minus_psiC_run2'))}"
                + (f"; -ES {_ci(row.get('minus_es'))}" if row.get("minus_es") else "")
            )
        for b, c in (ex.get("minus_offline_psi_cf_all") or {}).items():
            a(f"- acc {b}: -offline-psi cf_all {_ci(c)}")
        for b, row in (ex.get("hit_rate") or {}).items():
            a(f"- hit {b}: ceiling share {_ci(row.get('ceiling_share'))}")
        a(f"- age-1 share: {(h.get('age1_share') or {}).get('psiR2a')}")
        a(f"- rho(psi_hat, age) at EVAL decisions: {_ci(h.get('rho_psi_age'))}\n")
    a("## Expectation (PREREG §11, quoted)\n")
    a(expectation_text() + "\n")
    a("## What this does NOT show (PREREG §9)\n")
    a(prereg_passage("- **2a WIN** shows", "- **2b BEATS-FIFO**") + "\n")
    return "\n".join(w) + "\n"


def recompute(what: str, out_root: Path, seed: int) -> str:
    """claims.json: recomputed from the records on disk (TUNE and EVAL stay sealed:
    the TUNE references were stored at select)."""
    SS.install("2a", "finalize")
    R1, R2, OP = harness()["R1"], harness()["R2"], harness()["OP"]
    out_root = Path(out_root)
    if what == "headline":
        recs = R1.read_pass(eval_out(out_root, seed).parent, "psiR2a")
        fifo = R1.read_pass(Path(OP.REFS.run2_out) / "heldout" / f"seed{seed}", "fifo")
        W = R1.boot_weights(len(recs), seed, N_BOOT)
        h = OP.headline(recs, fifo, W, R2.read_delta(seed)["delta"])
        return f"{h['point']:.6f} {h['lo']:.6f} {h['hi']:.6f} {h['label']}"
    if what == "selection":
        ref = json.loads(
            (out_root / "tune" / f"seed{seed}" / "references.json").read_text()
        )
        hits = {"fifo": ref["fifo"], "factfiller": ref["factfiller"]}
        for ww in OMEGAS:
            p = out_root / "tune" / f"seed{seed}"
            hits[wtag(ww)] = [r["hits"] for r in R1.read_pass(p, wtag(ww))]
        caps = tune_caps(hits, seed, R1.M_SLOTS)
        pick = choose(caps)
        pts = " ".join(
            f"{wtag(ww)}={caps[wtag(ww)]['cap']['point']:.6f}" for ww in OMEGAS
        )
        return f"omega_s={pick['omega']:g} {pts}"
    if what == "kconv":
        kc = {
            ww: json.loads(fit_file(out_root, seed, ww).read_text())["fit"]["k_conv"]
            for ww in OMEGAS
        }
        return " ".join(f"{wtag(ww)}={kc[ww]}" for ww in OMEGAS)
    if what == "phi":
        sel = json.loads(selection_file(out_root, seed).read_text())
        return state_sha(torch.load(sel["phi_path"], weights_only=True))
    if what == "delta":
        return f"{R2.read_delta(seed)['delta']!r}"
    raise ValueError(what)


def write_claims(out_root: Path, seeds) -> Path:
    slot = (
        "PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run --lane cpu-det "
        "--slots 1 --wait 3600 -- .venv/bin/python"
    )
    base = f"{slot} {EXPERIMENT} 2a recompute --out-root {Path(out_root).resolve()}"
    claims = []
    for s in seeds:
        for what, claim in (
            (
                "headline",
                f"seed {s}: acc.all(psiR-2a) - acc.all(fifo) on EVAL, CI, label",
            ),
            ("selection", f"seed {s}: omega_s and the TUNE cap points of the 3 configs"),
            ("kconv", f"seed {s}: K_conv of the 3 configs"),
            ("phi", f"seed {s}: sha256 of the selected frozen phi"),
            ("delta", f"seed {s}: delta_seed machine-read (R-C9)"),
        ):
            claims.append(
                {
                    "claim": claim,
                    "command": f"{base} --what {what} --seed {s}",
                    "expected": recompute(what, out_root, s),
                }
            )
    p = Path(out_root) / "claims.json"
    p.write_text(json.dumps(claims, indent=2) + "\n")
    return p


def finalize(out_root: Path, *, argv_self: list[str]) -> int:
    out_root = Path(out_root)
    doc: dict[str, Any] = {"stage": "finalize"}
    open_stage("finalize", doc)
    seeds = registered_seeds()
    sm = summarize(out_root, seeds)
    L = C.ledger_mod()
    led = L.Ledger(RUN_ID, question=QUESTION, runs_root=out_root.parent)
    launch = _read(out_root / "launch.json")
    if launch and launch.get("manifest"):
        led.doc["config_hash"] = launch["manifest"]["config_hash"]
        led.doc["manifest_written_utc"] = launch["manifest"]["manifest_written_utc"]
    py = ".venv/bin/python"

    def rc_of(name):
        p = out_root / "logs" / f"{name}.rc"
        t = p.read_text().strip() if p.exists() else ""
        return int(t) if t.lstrip("-").isdigit() else None

    led.command([py, EXPERIMENT, "2a", "launch-checks"], exit_code=rc_of("launch-checks"))
    for s in seeds:
        led.command(
            [py, EXPERIMENT, "2a", "fit", "--seed", str(s), "--rc2"],
            exit_code=rc_of(f"fit-s{s}-rc2"),
        )
        for ww in OMEGAS:
            led.command(
                [py, EXPERIMENT, "2a", "fit", "--seed", str(s), "--omega", f"{ww:g}"],
                exit_code=rc_of(f"fit-s{s}-{wtag(ww)}"),
            )
        led.command(
            [py, EXPERIMENT, "2a", "select", "--seed", str(s)],
            exit_code=rc_of(f"select-s{s}"),
        )
        led.command(
            [py, EXPERIMENT, "2a", "deploy", "--seed", str(s)],
            exit_code=rc_of(f"deploy-s{s}"),
        )
    led.command([py, EXPERIMENT, *argv_self], exit_code=sm["exit"], note="this finalize")
    kc = [c["k_conv"] for s in seeds for c in sm["per_seed"][s]["configs"].values()]
    led.run_meta(
        device="cpu",
        seeds_actually_run=sm["seeds_actually_run"],
        steps_requested=STEP_CAP,
        steps_done=max(kc) if kc and None not in kc else None,
    )
    led.note(
        "units",
        "steps_* are fit optimizer steps (max K_conv over the 9 fits)",
        how="phase2a",
    )
    led.note("summary", sm, how=f"{EXPERIMENT} 2a finalize")
    led.note("prereg", {"path": PREREG_REL, "commit": PREREG_COMMIT}, how="phase2a")
    for s in seeds:
        h = sm["per_seed"][s]["headline"] or {}
        for f in ("point", "lo", "hi"):
            led.note(
                f"seed{s}.headline.{f}",
                h.get(f),
                how=f"runs/{RUN_ID}/seed{s}/deploy.json",
            )
        led.note(f"seed{s}.label", sm["labels"][s], how="run 2's seed_label (§4.4)")
        led.note(
            f"seed{s}.selection",
            sm["per_seed"][s]["selection"],
            how=f"runs/{RUN_ID}/seed{s}/selection.json",
        )
        for name, c in sm["per_seed"][s]["configs"].items():
            for k, v in c.items():
                led.note(
                    f"seed{s}.{name}.{k}", v, how=f"runs/{RUN_ID}/seed{s}/fit-{name}.json"
                )
    pts = [((sm["per_seed"][s]["headline"]) or {}).get("point") for s in seeds]
    if len(pts) >= 2 and all(_finite(x) for x in pts):
        led.stat("headline.point", pts, how="per seed 0,1,2; never averaged into a claim")
    led.status(sm["status"])
    led.write()
    (out_root / "summary.json").write_text(json.dumps(sm, indent=1, default=str) + "\n")
    (out_root / "RESULTS.md").write_text(render_results(sm, out_root, seeds, led.doc))
    if sm["exit"] in (0, 1):
        write_claims(out_root, seeds)
    return sm["exit"]


# --------------------------------------------------------------------------- #
# CLI: run.py 2a <stage> ...
# --------------------------------------------------------------------------- #


def main(argv: list[str]) -> int:
    from rsr.exit_codes import ArgumentParser

    torch.set_num_threads(1)  # §0 convention: 1 thread per process (R-C2's `==`)
    p = ArgumentParser(prog="run.py 2a", description="PREREG rsr-run3-ranking §4")
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("launch-checks")
    pl.add_argument("--out-root", default=str(OUT_DEFAULT))
    for name in ("suite-log", "suite-rc", "suite-head"):
        pl.add_argument(f"--{name}", default=None)
    pf = sub.add_parser("fit")
    pf.add_argument("--seed", type=int, required=True)
    g = pf.add_mutually_exclusive_group(required=True)
    g.add_argument("--rc2", action="store_true")
    g.add_argument("--omega", type=float)
    pf.add_argument("--out-root", default=str(OUT_DEFAULT))
    for name in ("select", "deploy"):
        ps = sub.add_parser(name)
        ps.add_argument("--seed", type=int, required=True)
        ps.add_argument("--out-root", default=str(OUT_DEFAULT))
    pz = sub.add_parser("finalize")
    pz.add_argument("--out-root", default=str(OUT_DEFAULT))
    pt = sub.add_parser("time")
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--steps", type=int, default=50)
    pt.add_argument("--out", required=True)
    pr = sub.add_parser("recompute")
    pr.add_argument("--out-root", required=True)
    pr.add_argument("--what", required=True)
    pr.add_argument("--seed", type=int, required=True)
    a = p.parse_args(argv)
    seeds = registered_seeds()
    if getattr(a, "seed", None) is not None and a.seed not in seeds:
        print(f"DID NOT RUN: seed {a.seed} is not registered ({seeds})", file=sys.stderr)
        return 3
    if a.cmd == "launch-checks":
        logs = Path(a.out_root) / "logs"
        return launch_checks(
            Path(a.out_root),
            suite_log=Path(a.suite_log or logs / "suite.log"),
            suite_rc=Path(a.suite_rc or logs / "suite.rc"),
            suite_head=Path(a.suite_head or logs / "suite.head"),
        )
    if a.cmd == "fit":
        if a.rc2:
            return rc2_seed(a.seed, Path(a.out_root))
        return fit_seed_omega(a.seed, a.omega, Path(a.out_root))
    if a.cmd == "select":
        return select_seed(a.seed, Path(a.out_root))
    if a.cmd == "deploy":
        return deploy_seed(a.seed, Path(a.out_root))
    if a.cmd == "finalize":
        return finalize(Path(a.out_root), argv_self=["2a", "finalize", *argv[1:]])
    if a.cmd == "time":
        d = time_smoke(a.seed, Path(a.out), steps=a.steps)
        print(
            json.dumps(
                {"projection": d["projection"], "fit": d["fit"], "timing": d["timing"]},
                indent=1,
                default=str,
            )
        )
        return int(Exit.OK) if not d["controls_failed"] else int(Exit.DID_NOT_RUN)
    if a.cmd == "recompute":
        print(recompute(a.what, Path(a.out_root), a.seed))
        return 0
    return 3
