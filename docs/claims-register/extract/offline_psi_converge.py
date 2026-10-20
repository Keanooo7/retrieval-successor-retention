"""Diagnostic A (experiments/offline-psi-converge; PREREG 58783a3, committed
2026-10-08 13:18 PDT, before seed 0 started 2026-10-08T22:59Z; results f7ac595;
verified at 9c998d3). offline-psi's head trained on U to convergence (U-val L_MC
rule), same data, optimizer and lr; U documents only, EVAL sealed. No committed ledger:
the record is the archived ``seed<s>/result.json`` / ``summary.json``."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from _ofam import SEEDS, labels_cmd, rec_row, record, src

EXPERIMENT = "offline-psi-converge"
EXP = EXPERIMENT
ARCH = "/Users/keanooo7/rsr-runs-archive/offline-psi-converge/runs/offline-psi-converge"
OWN = f"{C.SLOT}{C.PY} experiments/offline-psi-converge/run.py recompute"
P = f"{C.PREREG} (PREREG 58783a3 §3, before the run)"
PD = f"{C.PREREG} (PREREG 58783a3 §3, descriptive)"
U = (
    C.UNSEEN,
    "no fit beyond step 3000 existed when the PREREG was committed (PREREG §7)",
)
S3000 = (
    C.SEEN,
    "step 3000 reproduces offline-psi's selected head (C2), read in offline-psi",
)


def rows() -> list[C.Row]:
    out: list[C.Row] = []
    own = []
    for s in SEEDS:
        a = record(EXP, f"seed{s}/result.json")["arms"]["cf_all"]
        own.append(
            C.Check(
                label=f"s{s} own recompute cap",
                cmd=f"{OWN} cap --seed {s} --out-root {ARCH}",
                expected=f"{a['selected']['step']} {a['cap_selected']['cap']['point']!r}",
            )
        )
    out.append(
        rec_row(
            EXP,
            "diagA.cf_all.cap",
            "arms.cf_all.cap_selected.cap",
            "fit cap on U-val of offline-psi's cf_all head trained to convergence "
            "(selected step)",
            tier="headline",
            prereg=P,
            seen=U[0],
            seen_why=U[1],
            extra_checks=own,
        )
    )
    summ = record(EXP, "summary.json")
    conv = [summ["convergence"][str(s)] for s in SEEDS]
    lab_checks = [
        C.Check(
            label="fit label (own fit_label on recomputed caps)",
            cmd=labels_cmd(EXP),
            expected=summ["label"],
            pick="fit_label",
        )
    ]
    for s in SEEDS:
        a = record(EXP, f"seed{s}/result.json")["arms"]["cf_all"]
        lab_checks += [
            C.Check(
                label=f"s{s} convergence (own rule over archived history)",
                cmd=labels_cmd(EXP),
                expected=a["convergence"],
                pick=f"cf_all.convergence.{s}",
            ),
            C.Check(
                label=f"s{s} stopped_at",
                cmd=labels_cmd(EXP),
                expected=C.fmt_exact(a["stopped_at"]),
                pick=f"cf_all.stopped_at.{s}",
            ),
            C.Check(
                label=f"s{s} selected step",
                cmd=labels_cmd(EXP),
                expected=C.fmt_exact(a["selected"]["step"]),
                pick=f"cf_all.selected_step.{s}",
            ),
        ]
    out.append(
        C.Row(
            id="diagA.label",
            experiment=EXP,
            tier="headline",
            claim=f"per-seed convergence verdict (cf_all); fit label {summ['label']}",
            seeds=SEEDS,
            values=conv,
            source=src(EXP, "summary.json", "convergence.<s>, label")
            + "; seed<s>/result.json :: "
            "arms.cf_all.{convergence,stopped_at,selected.step}",
            prereg=P,
            seen=U[0],
            seen_why=U[1],
            checks=lab_checks,
        )
    )
    out.append(
        rec_row(
            EXP,
            "diagA.cf_live.cap",
            "arms.cf_live.cap_selected.cap",
            "fit cap on U-val, cf_live head trained to convergence (secondary arm)",
            prereg=PD,
            seen=U[0],
            seen_why=U[1],
        )
    )
    for arm in ("cf_all", "cf_live"):
        out.append(
            rec_row(
                EXP,
                f"diagA.{arm}.val_l_mc",
                f"arms.{arm}.selected.val_l_mc",
                f"U-val L_MC at the selected step, {arm}",
                prereg=P,
                seen=U[0],
                seen_why=U[1],
                with_ci=False,
            )
        )
        out.append(
            rec_row(
                EXP,
                f"diagA.{arm}.train_l_mc",
                f"arms.{arm}.train_l_mc_at_selected",
                f"U-train L_MC at the selected phi, {arm}",
                prereg=PD,
                seen=U[0],
                seen_why=U[1],
                with_ci=False,
            )
        )
        out.append(
            rec_row(
                EXP,
                f"diagA.{arm}.const_val_l_mc",
                f"arms.{arm}.constant_baseline.val_l_mc",
                f"U-val L_MC of a constant predictor (U-train mean G), {arm}",
                prereg=PD,
                seen=U[0],
                seen_why=U[1],
                with_ci=False,
            )
        )
    out.append(
        rec_row(
            EXP,
            "diagA.cf_all.rho_psi_G",
            "arms.cf_all.cap_selected.rho_psi_G_fifo_resident",
            "within-step Spearman rho(psi_hat, G) over FIFO-resident slots on U-val, "
            "cf_all at "
            "the selected phi (median over steps)",
            prereg=PD,
            seen=U[0],
            seen_why=U[1],
            point="median",
        )
    )
    out.append(
        rec_row(
            EXP,
            "diagA.cf_all.cap@3000",
            "arms.cf_all.cap_trajectory.3000.cap",
            "cap trajectory at step 3000, cf_all (= offline-psi's selected head)",
            prereg=PD,
            seen=S3000[0],
            seen_why=S3000[1],
        )
    )
    for k in ("10000", "30000"):
        out.append(
            rec_row(
                EXP,
                f"diagA.cf_all.cap@{k}",
                f"arms.cf_all.cap_trajectory.{k}.cap",
                f"cap trajectory at step {k}, cf_all",
                prereg=PD,
                seen=U[0],
                seen_why=U[1],
            )
        )
    return out
