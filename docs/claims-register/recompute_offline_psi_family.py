"""Recompute offline-psi / diagnostic A (offline-psi-converge) / LS (psi-ls-capacity)
readings for one seed from their saved artefacts, through each run's OWN reading code,
and print every leaf as one JSON line ``{"<dotted path>": value}``. The dotted paths
mirror the run's own record: offline-psi's ledger keys without the ``seed<s>.`` prefix
(``fit_reading.*``, ``heldout.secondary.accuracy.all.*``), and diagnostic A's / LS's
``seed<s>/result.json`` paths (``arms.<arm>.*``, ``reference_G.*``).

Usage: recompute_offline_psi_family.py
           --exp offline-psi|offline-psi-converge|psi-ls-capacity --seed S
       recompute_offline_psi_family.py --exp offline-psi-converge|psi-ls-capacity --labels

Reads saved phi states, gestalts and per-document records from
``~/rsr-runs-archive/<exp>/runs/<exp>/`` and the runs' own pinned inputs (B0-CF's Delta,
run 2's FIFO-world records), all read-only. Writes nothing. No EVAL document is
generated: the offline-psi deploy numbers are re-read from its archived EVAL records,
and diagnostic A / LS seal EVAL exactly as their runs did (``_seal_eval``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, C.ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def flatten(x, prefix: str, out: dict) -> None:
    if isinstance(x, dict):
        for k, v in x.items():
            flatten(v, f"{prefix}.{k}" if prefix else str(k), out)
    else:
        out[prefix] = x


def arch(exp: str) -> Path:
    return C.ARCHIVE / exp / "runs" / exp


def offline_psi(seed: int) -> dict:
    OP = _load("_cr_op", "experiments/offline-psi/run.py")
    root = arch("offline-psi")
    d = json.loads((root / f"measure-seed{seed}.json").read_text())
    refs = OP.Refs(**d["refs"])
    plan = OP.Plan(**d["plan"])
    out: dict = {}
    # fit reading on U-val (PREREG §5), both arms: offline-psi's fit_reading code path
    docs, _vmap, _train, val = OP.plan_docs(seed, plan)
    g = torch.load(root / f"seed{seed}" / "gestalts.pt", weights_only=False)
    if g["ids"] != [x.doc_id for x in docs]:
        raise SystemExit("gestalts.pt does not hold this seed's U documents")
    blob, _ = OP.load_delta(seed, refs)
    conf = OP.run2_psic_config(refs, seed)["conf"]
    ov, spe = dict(conf["policy_overrides"]), float(conf["epoch"]["steps_per_epoch"])
    vdocs = [docs[p] for p in val]
    vp = torch.tensor(val, dtype=torch.long)
    Wv = OP.B0.boot_weights(len(vdocs), seed, plan.n_boot)
    rr = OP.reference_rates(vdocs, seed)
    Gv = torch.stack([OP.target_of(blob["Delta"][x.doc_id]) for x in vdocs])
    fr: dict = {
        "reference_G": OP.B0.boot_rule(
            Wv, OP.replay_rates(vdocs, list(Gv)), rr["fifo"], rr["factfiller"]
        )
    }
    for arm in OP.ARMS:
        pol, _ = OP.new_policy(seed, ov, spe)
        pol.head.load_state_dict(
            torch.load(root / f"seed{seed}" / f"phi-{arm}.pt", weights_only=True)
        )
        P = OP.score_matrices(pol.head, g["s"][vp], g["c"][vp])
        r = OP.B0.boot_rule(
            Wv, OP.replay_rates(vdocs, list(P)), rr["fifo"], rr["factfiller"]
        )
        r["rho_psi_G_fifo_resident"] = OP.rho_psi_g(P, Gv, Wv)
        fr[arm] = r
    flatten(fr, "fit_reading", out)
    # deploy secondary (PREREG §6, descriptive): offline-psi's `secondary` on the
    # archived cf_all EVAL records and run 2's records; ES omitted (es=None).
    cf = OP.R1.read_pass(OP.heldout_dir(root, seed), OP.PRIMARY)
    other = OP.run2_records(refs, seed, ("fifo", "psiC", *OP.STEP1.RANDOM))
    W = OP.R1.boot_weights(len(cf), seed, plan.n_boot)
    sec = OP.secondary({"cf_all": cf, **other}, W, None)
    flatten({"accuracy": {"all": sec["accuracy"]["all"]}}, "heldout.secondary", out)
    return out


def converge(seed: int) -> dict:
    CV = _load("_cr_cv", "experiments/offline-psi-converge/run.py")
    OP = CV.OP
    root = arch("offline-psi-converge") / f"seed{seed}"
    CV._seal_eval()
    inp = CV.load_inputs(seed)
    vdocs = [inp["docs"][p] for p in inp["val"]]
    rr = OP.reference_rates(vdocs, seed)
    W = OP.B0.boot_weights(len(vdocs), seed, OP.R1.N_BOOT)
    Gv = torch.stack([OP.target_of(inp["blob"]["Delta"][d.doc_id]) for d in vdocs])
    res: dict = {
        "reference_G": OP.B0.boot_rule(
            W, OP.replay_rates(vdocs, list(Gv)), rr["fifo"], rr["factfiller"]
        ),
        "arms": {},
    }
    for arm in CV.ARMS:
        data = {
            "s": inp["S"],
            "c": inp["C"],
            **OP.fit_tensors(inp["blob"], inp["docs"], arm),
        }
        st = torch.load(root / f"phi-{arm}-selected.pt", weights_only=True)
        head = CV.new_head(seed)
        head.load_state_dict(st)
        a = {
            "cap_selected": CV.cap_of(st, seed, inp, rr, W),
            "selected": {
                "val_l_mc": OP.val_loss(
                    head, data["s"], data["c"], data["G"], data["mask"], inp["val"]
                )
            },
            "train_l_mc_at_selected": OP.val_loss(
                head, data["s"], data["c"], data["G"], data["mask"], inp["train"]
            ),
            "constant_baseline": CV.constant_baseline(data, inp["train"], inp["val"]),
            "cap_trajectory": {},
        }
        for p in sorted(root.glob(f"phi-{arm}-step*.pt")):
            k = p.stem.split("step")[-1]
            a["cap_trajectory"][k] = CV.cap_of(
                torch.load(p, weights_only=True), seed, inp, rr, W
            )
        res["arms"][arm] = a
    out: dict = {}
    flatten(res, "", out)
    return out


def ls(seed: int) -> dict:
    LS = _load("_cr_ls", "experiments/psi-ls-capacity/run.py")
    OP = LS.OP
    root = arch("psi-ls-capacity") / f"seed{seed}"
    LS.CV._seal_eval()
    inp = LS.load_inputs(seed)
    res: dict = {"arms": {}}
    for arm in LS.ARMS:
        st = torch.load(root / f"phi-{arm}-ls.pt", weights_only=True)
        head = LS.ls_head(seed, st)
        ft = OP.fit_tensors(inp["blob"], inp["docs"], arm)
        G64 = ft["G"].to(torch.float64)
        res["arms"][arm] = {
            "train_l_mc": OP.val_loss(
                head, inp["S64"], inp["C64"], G64, ft["mask"], inp["train"]
            ),
            "val_l_mc": OP.val_loss(
                head, inp["S64"], inp["C64"], G64, ft["mask"], inp["val"]
            ),
            "cap_val": LS.cap_reading(head, inp, inp["val"], seed),
            "cap_train": LS.cap_reading(head, inp, inp["train"], seed, rho=False),
        }
    out: dict = {}
    flatten(res, "", out)
    return out


def labels(exp: str) -> dict:
    """The run's own label rules applied to RECOMPUTED caps (all 3 seeds): offline-psi's
    ``fit_label`` (B0-CF §7 via the gamma sweep's ``label``). Diagnostic A also re-runs
    its convergence rule (``converged``) and selection (``select_step``) over the
    archived U-val L_MC trajectory (the run's raw ``history``)."""
    seeds = (0, 1, 2)
    if exp == "offline-psi-converge":
        CV = _load("_cr_cv", "experiments/offline-psi-converge/run.py")
        recs = {s: converge(s) for s in seeds}
        caps = {
            s: {
                f: recs[s][f"arms.cf_all.cap_selected.cap.{f}"]
                for f in ("point", "lo", "hi")
            }
            for s in seeds
        }
        out = {"fit_label": CV.OP.fit_label(caps)}
        for s in seeds:
            res = json.loads((arch(exp) / f"seed{s}" / "result.json").read_text())
            for arm in CV.ARMS:
                h = res["arms"][arm]["history"]
                stop = next(
                    (
                        x["step"]
                        for i, x in enumerate(h)
                        if CV.converged(h[: i + 1], x["step"])
                    ),
                    None,
                )
                sel = CV.OP.select_step(h)
                out[f"{arm}.convergence.{s}"] = (
                    "CONVERGED" if stop is not None else "CAP-HIT"
                )
                out[f"{arm}.stopped_at.{s}"] = stop
                out[f"{arm}.selected_step.{s}"] = sel["step"]
        return out
    if exp == "psi-ls-capacity":
        LS = _load("_cr_ls", "experiments/psi-ls-capacity/run.py")
        recs = {s: ls(s) for s in seeds}
        out = {}
        for arm in LS.ARMS:
            res = {
                s: {
                    "arms": {
                        arm: {
                            k: {
                                "cap": {
                                    f: recs[s][f"arms.{arm}.{k}.cap.{f}"]
                                    for f in ("point", "lo", "hi")
                                }
                            }
                            for k in ("cap_val", "cap_train")
                        }
                    }
                }
                for s in seeds
            }
            v, i = LS.labels_of(res, arm)
            out[f"{arm}.label_val"], out[f"{arm}.label_insample"] = v, i
        return out
    raise SystemExit(f"--labels is not defined for {exp}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exp",
        required=True,
        choices=("offline-psi", "offline-psi-converge", "psi-ls-capacity"),
    )
    ap.add_argument("--seed", type=int)
    ap.add_argument("--labels", action="store_true")
    a = ap.parse_args(argv)
    torch.set_num_threads(1)
    if a.labels:
        print(json.dumps(labels(a.exp), default=float))
        return 0
    if a.seed is None:
        ap.error("--seed is required without --labels")
    fn = {
        "offline-psi": offline_psi,
        "offline-psi-converge": converge,
        "psi-ls-capacity": ls,
    }[a.exp]
    print(json.dumps(fn(a.seed), default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
