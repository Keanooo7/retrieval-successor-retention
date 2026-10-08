"""Diagnostic A after offline-psi: train offline-psi's psi_hat to convergence on U.

PREREG `experiments/offline-psi-converge/PREREG.md` (committed alone at 58783a3).
U documents only: no EVAL document is generated or read (C5). offline-psi's module is
imported and defines its deploy path; C5 is enforced by `_seal_eval`, which replaces that
path and run 1's EVAL document generator with functions that raise, before anything runs.

offline-psi's harness is reused by import, never copied: the U documents and split, the
Delta loader, the targets and cell masks (`target_of`, `fit_tensors`), the head and
its init (`new_policy`, run 2's psiC phi seed), the optimizer (`phi_optimizer`, run 2's
lr), the loss
(`psi_matrix`, `l_mc`), U-val L_MC (`val_loss`), selection (`select_step`), and the fit
reading (`score_matrices`, `replay_rates`, `reference_rates`, `rho_psi_g`, `fit_label`).
Only the schedule changes (PREREG §2): a 200,000-step cap, U-val every 500 steps, and a
convergence rule. The training loop below is offline-psi's `fit` loop with that rule
added;
C2 checks its trajectory equals offline-psi's at step 3000.

Subcommands: ``run --seed S``, ``recompute {cap,c2} --seed S``, ``finalize``.
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
from rsr.train import loop  # noqa: E402

EXPERIMENT = "experiments/offline-psi-converge/run.py"
RUN_ID = "offline-psi-converge"
PREREG_COMMIT = "58783a3"
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


OP = _load("_converge_op", "experiments/offline-psi/run.py")

SEEDS = OP.SEEDS
ARMS = OP.ARMS  # cf_all (primary), cf_live
PRIMARY = OP.PRIMARY
STEP_CAP = 200_000  # PREREG §2
VAL_EVERY = 500  # PREREG §2
CONV_MIN_STEP = 20_000  # PREREG §2
CONV_WINDOW = 10_000  # PREREG §2
CONV_REL = 0.001  # PREREG §2
SAVE_STEPS = (3000, 10_000, 30_000, 100_000)  # PREREG §2
REPRO_STEP = 3000  # C2
#: PREREG §1: offline-psi's gestalts, read (sha256 = its measure-seed `gestalts.sha256`).
OP_OUT = Path(OP.R1._R) / ".worktrees/offline-psi/runs/offline-psi"
GESTALT_SHA = {
    0: "211244f1469b90972e3b1d3b6635b3ee9eb095e78b932dc396bba64df3f5562b",
    1: "3a18e2917b385a6339ef82e92535953105877cf295bbaf9f6495a4749a6a0c68",
    2: "74e3bec7e5366d631480f9e3ac9a07916926da21ffd60be57568acb3b3723706",
}
EVAL_IDS = range(995000, 997048)


class ControlFailure(RuntimeError):
    pass


def _seal_eval() -> dict:
    """C5: make every EVAL entry point raise. Reading PREREG C5's "imports no deploy path"
    as "calls none": offline-psi's module (imported for its fit functions) defines one."""

    def refuse(*_a, **_k):
        raise ControlFailure("C5: an EVAL/deploy path was called")

    sealed = []
    for mod, name in ((OP, "deploy_pass"), (OP, "deploy_policy"), (OP.R1, "docs_by_id")):
        if hasattr(mod, name):
            setattr(mod, name, refuse)
            sealed.append(f"{mod.__name__}.{name}")
    return {"sealed": sealed}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def converged(history: list[dict], k: int) -> bool:
    """PREREG §2: at a check with ``k >= 20,000``, converged if best(k) >= (1 - 0.001) *
    best(k - 10,000), where best(j) is the lowest finite U-val L_MC at or before j."""
    if k < CONV_MIN_STEP:
        return False

    def best(j):
        v = [
            h["val_l_mc"] for h in history if h["step"] <= j and OP._finite(h["val_l_mc"])
        ]
        return min(v) if v else None

    now, then = best(k), best(k - CONV_WINDOW)
    if now is None or then is None:
        return False
    return now >= (1.0 - CONV_REL) * then


def fit_converge(
    head,
    opt,
    data: dict,
    train_pos,
    val_pos,
    *,
    seed: int,
    cap: int,
    every: int,
    save_steps=SAVE_STEPS,
) -> dict:
    """offline-psi's `fit` loop (sampler, loss, step, grad norm, in that order), with
    PREREG §2's longer schedule and convergence rule."""
    S_, C_, G, mask = data["s"], data["c"], data["G"], data["mask"]
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    sampler = loop.PermutationEpochs(len(train_pos), gen, "cpu")
    tp = torch.tensor(train_pos, dtype=torch.long)
    params = list(head.parameters())
    history: list[dict] = []
    saved: dict[int, dict] = {}
    bad: list[int] = []
    kept = None
    stop = None
    t0 = time.perf_counter()
    val0 = OP.val_loss(head, S_, C_, G, mask, val_pos)
    for k in range(1, cap + 1):
        pos = tp[sampler.take(OP.FIT_BATCH)]
        P = OP.psi_matrix(head, S_[pos], C_[pos])
        loss, _ = OP.l_mc(P, G[pos], mask[pos])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = loop._grad_norm(params)
        lv = float(loss.detach())
        if not (OP._finite(lv) and OP._finite(gn)):
            bad.append(k)
        opt.step()
        snap = None
        if k in save_steps or k % every == 0:
            snap = {n: x.detach().clone() for n, x in head.state_dict().items()}
        if k in save_steps:
            saved[k] = snap
        if k % every == 0:
            v = OP.val_loss(head, S_, C_, G, mask, val_pos)
            history.append({"step": k, "val_l_mc": v, "train_l_mc": lv, "grad_norm": gn})
            if OP._finite(v) and (kept is None or v < kept["val_l_mc"]):
                kept = {"step": k, "val_l_mc": v, "state": snap}
            if converged(history, k):
                stop = k
                break
    sel = OP.select_step(history)
    if (sel is None) != (kept is None) or (
        sel is not None and sel["step"] != kept["step"]
    ):
        raise ControlFailure(f"kept {kept and kept['step']} != selection {sel}")
    return {
        "history": history,
        "val_l_mc_step0": val0,
        "selected": None
        if sel is None
        else {"step": sel["step"], "val_l_mc": sel["val_l_mc"]},
        "state": None if kept is None else kept["state"],
        "saved": saved,
        "convergence": "CONVERGED" if stop is not None else "CAP-HIT",
        "stopped_at": stop if stop is not None else cap,
        "c4_n_bad": len(bad),
        "c4_bad_steps": bad[:20],
        "seconds": time.perf_counter() - t0,
    }


def load_inputs(seed: int) -> dict:
    """PREREG §1 and C1: offline-psi's gestalts and B0-CF's Delta, sha-checked."""
    gp = OP_OUT / f"seed{seed}" / "gestalts.pt"
    gsha = sha256_file(gp)
    g = torch.load(gp, weights_only=False)
    docs, _vmap = OP.u_documents(seed)
    ids = [d.doc_id for d in docs]
    blob, dsha = OP.load_delta(seed)
    c1 = {
        "gestalt_sha256": gsha,
        "gestalt_ok": gsha == GESTALT_SHA[seed] and g["ids"] == ids,
        "delta_sha256": dsha,
        "delta_ok": dsha == OP.DELTA_SHA[seed],
    }
    c1["ok"] = c1["gestalt_ok"] and c1["delta_ok"]
    c5 = {
        "ok": not (set(ids) & set(EVAL_IDS)),
        "eval_ids_in_u": len(set(ids) & set(EVAL_IDS)),
    }
    train, val = OP.u_split(len(docs))
    return {
        "docs": docs,
        "ids": ids,
        "S": g["s"],
        "C": g["c"],
        "blob": blob,
        "train": train,
        "val": val,
        "c1": c1,
        "c5": c5,
    }


def run2_conf(seed: int) -> tuple[dict, float, float]:
    conf = OP.run2_psic_config(OP.REFS, seed)["conf"]
    return (
        dict(conf["policy_overrides"]),
        float(conf["epoch"]["steps_per_epoch"]),
        float(conf["lr"]),
    )


def new_head(seed: int):
    """offline-psi's head at run 2's psiC phi seed (no optimizer, no model)."""
    ov, spe, _lr = run2_conf(seed)
    return OP.new_policy(seed, ov, spe)[0].head


def make_head_and_opt(seed: int):
    """offline-psi's construction: run 2's psiC config (overrides, epoch, lr) and phi
    seed;
    the optimizer over the frozen substrate model's muP groups (phi only)."""
    ov, spe, lr = run2_conf(seed)
    cfg = OP.R1.prereg_config(seed, OP.CKPT)
    model, _payloads, _sha = OP.load_frozen(cfg)
    pol, phi_seed = OP.new_policy(seed, ov, spe)
    head = pol.head
    opt = OP.phi_optimizer(model, head, lr)
    return (
        head,
        opt,
        {
            "lr": lr,
            "phi_seed": phi_seed,
            "only_phi": OP.optimizer_holds_only_phi(opt, head),
        },
    )


def op_history_value(seed: int, arm: str, step: int) -> float | None:
    d = json.loads((OP_OUT / f"measure-seed{seed}.json").read_text())
    for h in d["fit"][arm]["history"]:
        if h["step"] == step:
            return h["val_l_mc"]
    return None


def cap_of(head_state: dict, seed: int, inp: dict, rr: dict, W) -> dict:
    """offline-psi PREREG §5's cap on U-val for one phi state."""
    head = new_head(seed)
    head.load_state_dict(head_state)
    vp = torch.tensor(inp["val"], dtype=torch.long)
    vdocs = [inp["docs"][p] for p in inp["val"]]
    P = OP.score_matrices(head, inp["S"][vp], inp["C"][vp])
    r = OP.B0.boot_rule(W, OP.replay_rates(vdocs, list(P)), rr["fifo"], rr["factfiller"])
    Gv = torch.stack([OP.target_of(inp["blob"]["Delta"][d.doc_id]) for d in vdocs])
    r["rho_psi_G_fifo_resident"] = OP.rho_psi_g(P, Gv, W)
    return r


def constant_baseline(data: dict, train, val) -> dict:
    """PREREG §3 item 4: U-val L_MC of the constant U-train mean G over the arm's
    cells."""
    tp, vp = torch.tensor(train), torch.tensor(val)
    m = data["mask"][tp]
    c = float(data["G"][tp][m].double().mean())
    P = torch.full_like(data["G"][vp], c)
    loss, _ = OP.l_mc(P.double(), data["G"][vp].double(), data["mask"][vp])
    return {"mean_G_train": c, "val_l_mc": float(loss)}


def run_seed(
    seed: int,
    out_root: Path,
    *,
    cap: int = STEP_CAP,
    every: int = VAL_EVERY,
    threads: int = 1,
) -> dict:
    torch.set_num_threads(threads)
    sd = Path(out_root) / f"seed{seed}"
    out = sd / "result.json"
    if out.exists():
        raise ControlFailure(f"{out} exists; never overwrite a result")
    sd.mkdir(parents=True, exist_ok=True)
    doc: dict[str, Any] = {
        "run_id": RUN_ID,
        "seed": seed,
        "cap": cap,
        "every": every,
        "tree": OP.R1.tree_state(),
        "hardware": OP.R1.hardware(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "controls": {},
        "arms": {},
    }
    seal = _seal_eval()
    inp = load_inputs(seed)
    doc["controls"]["C1"] = inp["c1"]
    doc["controls"]["C5"] = {**inp["c5"], **seal}
    vdocs = [inp["docs"][p] for p in inp["val"]]
    rr = OP.reference_rates(vdocs, seed)
    W = OP.B0.boot_weights(len(vdocs), seed, OP.R1.N_BOOT)
    Gv = torch.stack([OP.target_of(inp["blob"]["Delta"][d.doc_id]) for d in vdocs])
    doc["reference_G"] = OP.B0.boot_rule(
        W, OP.replay_rates(vdocs, list(Gv)), rr["fifo"], rr["factfiller"]
    )
    c2, c3, c4 = {}, {}, {}
    for arm in ARMS:
        head, opt, meta = make_head_and_opt(seed)
        c3[arm] = meta["only_phi"]
        data = {
            "s": inp["S"],
            "c": inp["C"],
            **OP.fit_tensors(inp["blob"], inp["docs"], arm),
        }
        r = fit_converge(
            head, opt, data, inp["train"], inp["val"], seed=seed, cap=cap, every=every
        )
        c4[arm] = {"n_bad": r["c4_n_bad"], "ok": r["c4_n_bad"] == 0}
        mine = next(
            (h["val_l_mc"] for h in r["history"] if h["step"] == REPRO_STEP), None
        )
        ref = op_history_value(seed, arm, REPRO_STEP)
        c2[arm] = {
            "mine": mine,
            "offline_psi": ref,
            "ok": mine is not None and ref is not None and mine == ref,
        }
        arm_doc: dict[str, Any] = {
            k: r[k]
            for k in (
                "history",
                "val_l_mc_step0",
                "selected",
                "convergence",
                "stopped_at",
                "seconds",
            )
        }
        arm_doc["lr"], arm_doc["phi_seed"] = meta["lr"], meta["phi_seed"]
        if r["state"] is not None:
            torch.save(r["state"], sd / f"phi-{arm}-selected.pt")
            arm_doc["selected"]["phi_sha256"] = sha256_file(sd / f"phi-{arm}-selected.pt")
            arm_doc["cap_selected"] = cap_of(r["state"], seed, inp, rr, W)
            head.load_state_dict(r["state"])
            arm_doc["train_l_mc_at_selected"] = OP.val_loss(
                head, data["s"], data["c"], data["G"], data["mask"], inp["train"]
            )  # full U-train L_MC at the selected phi
        arm_doc["cap_trajectory"] = {}
        for k, st in sorted(r["saved"].items()):
            torch.save(st, sd / f"phi-{arm}-step{k}.pt")
            arm_doc["cap_trajectory"][k] = cap_of(st, seed, inp, rr, W)
        arm_doc["constant_baseline"] = constant_baseline(data, inp["train"], inp["val"])
        doc["arms"][arm] = arm_doc
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


def fmt(c: dict) -> str:
    return f"{c['point']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"


def finalize(out_root: Path) -> dict:
    res = {
        s: json.loads((Path(out_root) / f"seed{s}" / "result.json").read_text())
        for s in SEEDS
    }
    caps = {s: res[s]["arms"][PRIMARY]["cap_selected"]["cap"] for s in SEEDS}
    label = OP.fit_label(caps)
    conv = {s: res[s]["arms"][PRIMARY]["convergence"] for s in SEEDS}
    ok = all(res[s]["controls_ok"] for s in SEEDS)
    if not ok:
        route = "controls failed: inconclusive (exit 3)"
    elif any(v == "CAP-HIT" for v in conv.values()):
        route = "CAP-HIT on a seed: diagnostic B (correction 15) next (PREREG §6 row 3)"
    elif label in ("FIT-POINTS", "FIT-PARTIAL"):
        route = "CONVERGED + POINTS/PARTIAL: the head can express it (PREREG §6 row 1)"
    else:
        route = (
            "CONVERGED + MISSES: offline-psi's §8 MISSES route stands (PREREG §6 row 2)"
        )
    L = [
        "# RESULTS: offline-psi-converge (diagnostic A)",
        "",
        "Generated by run.py finalize; no number is typed. Written by a model; not by "
        f"Brendan. PREREG `{PREREG_COMMIT}`. U documents only; EVAL never read.",
        "",
        f"- command: `.venv/bin/python {EXPERIMENT} finalize`",
        f"- git: `{res[0]['tree']}`",
        f"- hardware: {res[0]['hardware']}",
        f"- exit `{0 if ok else 3}`",
        "",
        f"**Fit label (cf_all, selected step): {label}**. Convergence: "
        + ", ".join(f"seed {s} {conv[s]}" for s in SEEDS),
        "",
        f"**Route:** {route}",
        "",
        "## Per seed and arm",
        "",
        "| seed | arm | convergence | stopped at | selected step | U-val L_MC sel "
        "| train L_MC sel | const-mean L_MC | step-0 L_MC | cap at selected "
        "| rho(psi,G) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in SEEDS:
        for arm in ARMS:
            a = res[s]["arms"][arm]
            cs = a.get("cap_selected")
            c = cs["cap"] if cs else None
            rho = cs["rho_psi_G_fifo_resident"] if cs else None
            L.append(
                f"| {s} | {arm} | {a['convergence']} | {a['stopped_at']} | "
                f"{a['selected']['step']} | {a['selected']['val_l_mc']:.4f} | "
                f"{a.get('train_l_mc_at_selected', float('nan')):.4f} | "
                f"{a['constant_baseline']['val_l_mc']:.4f} | {a['val_l_mc_step0']:.4f} | "
                f"{fmt(c) if c else '-'} | "
                f"{rho['median']:+.4f} [{rho['lo']:+.4f}, {rho['hi']:+.4f}] |"
            )
    L += [
        "",
        "## Cap trajectory (cf_all; U-val)",
        "",
        "| seed | step | cap [95% CI] |",
        "| --- | --- | --- |",
    ]
    for s in SEEDS:
        for k, c in res[s]["arms"][PRIMARY]["cap_trajectory"].items():
            L.append(f"| {s} | {k} | {fmt(c['cap'])} |")
        sel = res[s]["arms"][PRIMARY]
        L.append(
            f"| {s} | selected {sel['selected']['step']} | "
            f"{fmt(sel['cap_selected']['cap'])} |"
        )
        L.append(f"| {s} | B0-CF's G (reference) | {fmt(res[s]['reference_G']['cap'])} |")
    L += ["", "## Controls", ""]
    for s in SEEDS:
        L.append(
            f"- seed {s}: ok `{res[s]['controls_ok']}`; {json.dumps(res[s]['controls'])}"
        )
    L += [
        "",
        "## Not shown",
        "",
        "- No EVAL number; no deployment; no change to offline-psi's or run 2's labels.",
        "- No lr or muP change (correction 15). No scaling or cognitive claim. "
        "§15 untouched.",
        "",
    ]
    out = Path(out_root)
    (out / "RESULTS.md").write_text("\n".join(L))
    summary = {
        "label": label,
        "convergence": conv,
        "route": route,
        "controls_ok": ok,
        "caps": {s: caps[s]["point"] for s in SEEDS},
    }
    claims = []
    for s in SEEDS:
        a = res[s]["arms"][PRIMARY]
        claims.append(
            {
                "claim": f"seed {s} cf_all cap at the selected phi",
                "command": f".venv/bin/python {EXPERIMENT} recompute cap --seed {s}",
                "expected": f"{a['selected']['step']} {caps[s]['point']!r}",
            }
        )
        claims.append(
            {
                "claim": f"seed {s} C2 trajectory reproduction (cf_all, step 3000)",
                "command": f".venv/bin/python {EXPERIMENT} recompute c2 --seed {s}",
                "expected": repr(res[s]["controls"]["C2"][PRIMARY]["mine"]),
            }
        )
    (out / "claims.json").write_text(
        json.dumps({"run_id": RUN_ID, "claims": claims}, indent=1)
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def recompute(what: str, seed: int, out_root: Path) -> str:
    res = json.loads((Path(out_root) / f"seed{seed}" / "result.json").read_text())
    a = res["arms"][PRIMARY]
    if what == "cap":
        inp = load_inputs(seed)
        vdocs = [inp["docs"][p] for p in inp["val"]]
        rr = OP.reference_rates(vdocs, seed)
        W = OP.B0.boot_weights(len(vdocs), seed, OP.R1.N_BOOT)
        st = torch.load(
            Path(out_root) / f"seed{seed}" / f"phi-{PRIMARY}-selected.pt",
            weights_only=True,
        )
        return f"{a['selected']['step']} {cap_of(st, seed, inp, rr, W)['cap']['point']!r}"
    if what == "c2":
        inp = load_inputs(seed)
        head, opt, _ = make_head_and_opt(seed)
        data = {
            "s": inp["S"],
            "c": inp["C"],
            **OP.fit_tensors(inp["blob"], inp["docs"], PRIMARY),
        }
        r = fit_converge(
            head,
            opt,
            data,
            inp["train"],
            inp["val"],
            seed=seed,
            cap=REPRO_STEP,
            every=VAL_EVERY,
            save_steps=(),
        )
        return repr(next(h["val_l_mc"] for h in r["history"] if h["step"] == REPRO_STEP))
    raise ValueError(what)


def main(argv: list[str] | None = None) -> Exit:
    p = ArgumentParser(prog=RUN_ID, description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--seed", type=int, required=True, choices=SEEDS)
    r.add_argument("--out-root", type=Path, default=OUT_DEFAULT)
    r.add_argument("--cap", type=int, default=STEP_CAP)
    r.add_argument("--threads", type=int, default=1)
    rc = sub.add_parser("recompute")
    rc.add_argument("what", choices=("cap", "c2"))
    rc.add_argument("--seed", type=int, required=True, choices=SEEDS)
    rc.add_argument("--out-root", type=Path, default=OUT_DEFAULT)
    f = sub.add_parser("finalize")
    f.add_argument("--out-root", type=Path, default=OUT_DEFAULT)
    a = p.parse_args(argv)
    if a.cmd == "run":
        d = run_seed(a.seed, a.out_root, cap=a.cap, threads=a.threads)
        print(f"seed {a.seed} exit {d['exit']} controls_ok {d['controls_ok']}")
        return Exit.OK if d["controls_ok"] else Exit.DID_NOT_RUN
    if a.cmd == "recompute":
        torch.set_num_threads(1)
        print(recompute(a.what, a.seed, a.out_root))
        return Exit.OK
    s = finalize(a.out_root)
    print(json.dumps(s))
    return Exit.OK if s["controls_ok"] else Exit.DID_NOT_RUN


if __name__ == "__main__":
    run_main(main)
