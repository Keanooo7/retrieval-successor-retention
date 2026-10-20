"""LS (experiments/psi-ls-capacity; PREREG 87dc2ec + Amendment 1 f99f6ce, committed
2026-10-08 17:36 / 17:56 PDT, before seed 0 started 2026-10-09T01:04Z; results 90c2720;
verified at 320c58a). The exact least-squares (L_MC) optimum of psi_hat's content-only
bilinear class on U-train, fitted on FIFO-world Delta (not on-policy). U documents only,
EVAL sealed. No committed ledger: the record is the archived ``seed<s>/result.json`` /
``summary.json``."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from _ofam import SEEDS, TBOOT, UBOOT, labels_cmd, rec_row, record, src

EXPERIMENT = "psi-ls-capacity"
EXP = EXPERIMENT
ARCH = "/Users/keanooo7/rsr-runs-archive/psi-ls-capacity/runs/psi-ls-capacity"
OWN = f"{C.SLOT}{C.PY} experiments/psi-ls-capacity/run.py recompute"
P = f"{C.PREREG} (PREREG 87dc2ec §3 + Amendment 1 f99f6ce, before the run)"
PD = f"{C.PREREG} (PREREG 87dc2ec §3, descriptive)"
U_CAP = (
    C.UNSEEN,
    "no cap and no U-val number of the LS head existed before launch "
    "(PREREG §7, Amendment 1)",
)
S_TRAIN = (
    C.SEEN,
    "U-train L_MC at the registered cutoff was computed in the pre-launch "
    "smoke and declared (Amendment 1)",
)
RIDGE_COST = (
    "not re-derived (cost): the ridge path is computed only inside `run`, from the "
    "eigendecomposition of the 16,640-dim normal matrix (eigh 350-400 s per arm and "
    "seed, "
    "plus assembly); 6 solves, about 40 min; no recompute path exists"
)


def rows() -> list[C.Row]:
    out: list[C.Row] = []
    solve = []
    for s in SEEDS:
        a = record(EXP, f"seed{s}/result.json")["arms"]["cf_all"]
        solve.append(
            C.Check(
                label=f"s{s} own recompute solve (re-solves LS from the inputs, ~6 min)",
                cmd=f"{OWN} solve --seed {s} --out-root {ARCH}",
                expected=(
                    f"{a['train_l_mc']!r} {a['val_l_mc']!r} "
                    f"{a['cap_val']['cap']['point']!r}"
                ),
            )
        )
        solve.append(
            C.Check(
                label=f"s{s} own recompute cap (saved LS phi)",
                cmd=f"{OWN} cap --seed {s} --out-root {ARCH}",
                expected=(
                    f"{a['cap_val']['cap']['point']!r} {a['cap_train']['cap']['point']!r}"
                ),
            )
        )
    out.append(
        rec_row(
            EXP,
            "ls.cf_all.cap_val",
            "arms.cf_all.cap_val.cap",
            "cap on U-val of the exact least-squares optimum of psi_hat's class (cf_all; "
            "fitted on FIFO-world Delta, not on-policy)",
            tier="headline",
            prereg=P,
            seen=U_CAP[0],
            seen_why=U_CAP[1],
            extra_checks=solve,
        )
    )
    out.append(
        rec_row(
            EXP,
            "ls.cf_all.cap_train",
            "arms.cf_all.cap_train.cap",
            "in-sample cap on U-train (the 932 documents it was fitted to), cf_all LS "
            "head",
            tier="headline",
            prereg=P,
            seen=U_CAP[0],
            seen_why=U_CAP[1],
            ci_method=TBOOT,
        )
    )
    summ = record(EXP, "summary.json")
    out.append(
        C.Row(
            id="ls.labels",
            experiment=EXP,
            tier="headline",
            claim=(
                f"labels: U-val {summ['label_val']}, in-sample "
                f"{summ['label_insample']} "
                f"(cf_live: {' / '.join(summ['cf_live'])}); cutoff-sensitive "
                f"{summ['cutoff_sensitive']}"
            ),
            seeds=SEEDS,
            values=[summ["label_val"]] * 3,
            source=src(
                EXP,
                "summary.json",
                "label_val, label_insample, cf_live, cutoff_sensitive",
            ),
            prereg=P,
            seen=U_CAP[0],
            seen_why=U_CAP[1],
            note="one label over all 3 seeds, by B0-CF §7's rule on the per-seed caps",
            checks=[
                C.Check(
                    label="U-val label (own labels_of on recomputed caps)",
                    cmd=labels_cmd(EXP),
                    expected=summ["label_val"],
                    pick="cf_all.label_val",
                ),
                C.Check(
                    label="in-sample label",
                    cmd=labels_cmd(EXP),
                    expected=summ["label_insample"],
                    pick="cf_all.label_insample",
                ),
                C.Check(
                    label="cf_live U-val label",
                    cmd=labels_cmd(EXP),
                    expected=summ["cf_live"][0],
                    pick="cf_live.label_val",
                ),
                C.Check(
                    label="cf_live in-sample label",
                    cmd=labels_cmd(EXP),
                    expected=summ["cf_live"][1],
                    pick="cf_live.label_insample",
                ),
            ],
        )
    )
    out.append(
        rec_row(
            EXP,
            "ls.cf_all.val_l_mc",
            "arms.cf_all.val_l_mc",
            "U-val L_MC of the LS head, cf_all",
            prereg=P,
            seen=U_CAP[0],
            seen_why=U_CAP[1],
            with_ci=False,
        )
    )
    out.append(
        rec_row(
            EXP,
            "ls.cf_all.train_l_mc",
            "arms.cf_all.train_l_mc",
            "U-train L_MC of the LS head (the class's optimum on U-train), cf_all",
            prereg=P,
            seen=S_TRAIN[0],
            seen_why=S_TRAIN[1],
            with_ci=False,
        )
    )
    out.append(
        rec_row(
            EXP,
            "ls.cf_all.rho_psi_G",
            "arms.cf_all.cap_val.rho_psi_G_fifo_resident",
            "within-step Spearman rho(psi_hat, G) over FIFO-resident slots on U-val, LS "
            "cf_all "
            "(median over steps)",
            prereg=PD,
            seen=U_CAP[0],
            seen_why=U_CAP[1],
            point="median",
        )
    )
    for k, claim in (
        ("cap_val", "cap on U-val"),
        ("cap_train", "in-sample cap on U-train"),
    ):
        out.append(
            rec_row(
                EXP,
                f"ls.cf_live.{k}",
                f"arms.cf_live.{k}.cap",
                f"{claim}, cf_live LS head (secondary arm)",
                prereg=PD,
                seen=U_CAP[0],
                seen_why=U_CAP[1],
                ci_method=TBOOT if k == "cap_train" else UBOOT,
            )
        )
    # Ridge path (PREREG §3 item 4, descriptive; RESULTS F1 hypothesis, F10 U-val spent)
    for arm in ("cf_all", "cf_live"):
        rho = []
        vals_cap, cis, vals_l = [], [], []
        for s in SEEDS:
            rp = record(EXP, f"seed{s}/result.json")["arms"][arm]["ridge_path"]
            hi = max(rp, key=lambda x: x["rho"])
            lo = min(rp, key=lambda x: x["rho"])
            rho.append((lo["rho"], hi["rho"]))
            vals_cap.append(hi["cap_val"]["point"])
            cis.append((hi["cap_val"]["lo"], hi["cap_val"]["hi"]))
            vals_l.append(hi["val_l_mc"])
        out.append(
            C.Row(
                id=f"ls.{arm}.ridge.cap@max_rho",
                experiment=EXP,
                tier="secondary",
                claim=f"ridge path, {arm}: U-val cap at the largest ridge strength rho = "
                f"{rho[0][1]!r} (grid {rho[0][0]!r}..{rho[0][1]!r})",
                seeds=SEEDS,
                values=vals_cap,
                ci=cis,
                ci_method=UBOOT,
                source=src(
                    EXP,
                    "seed{s}/result.json",
                    f"arms.{arm}.ridge_path[rho=max].cap_val.{{point,lo,hi}}",
                ),
                prereg=PD,
                seen=C.UNSEEN,
                seen_why="first read in this run (descriptive); U-val is now spent (F10)",
                rederive_note=RIDGE_COST,
            )
        )
        out.append(
            C.Row(
                id=f"ls.{arm}.ridge.val_l_mc@max_rho",
                experiment=EXP,
                tier="secondary",
                claim=f"ridge path, {arm}: U-val L_MC at rho = {rho[0][1]!r}",
                seeds=SEEDS,
                values=vals_l,
                source=src(
                    EXP, "seed{s}/result.json", f"arms.{arm}.ridge_path[rho=max].val_l_mc"
                ),
                prereg=PD,
                seen=C.UNSEEN,
                seen_why="first read in this run (descriptive); U-val is now spent (F10)",
                rederive_note=RIDGE_COST,
            )
        )
    return out
