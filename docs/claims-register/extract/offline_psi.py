"""offline-psi (experiments/offline-psi; PREREG c200a08 + Amendment 1 66682e5, both
committed 2026-10-08 before the run that started 2026-10-08T20:07Z; results 6d481f0;
verified at dc6d1bd). Fit the content-only head offline on B0-CF's uncensored
FIFO-world Delta (U-train), select on U-val L_MC, deploy frozen on run 2's EVAL
[995000, 997048)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from _ofam import SEEDS, UBOOT, fam_cmd

EXPERIMENT = "offline-psi"
RUN = "offline-psi"
ARCH = "/Users/keanooo7/rsr-runs-archive/offline-psi/runs/offline-psi"
OWN = f"{C.SLOT}{C.PY} experiments/offline-psi/run.py recompute --out-root {ARCH}"
EBOOT = (
    "paired per-document bootstrap percentile, n_boot 2000, 2048 EVAL docs, "
    "seed 20260927+s"
)
P4 = f"{C.PREREG} (PREREG c200a08 §4, before the run)"
P5 = f"{C.PREREG} (PREREG c200a08 §5, before the run)"
P5S = f"{C.PREREG} (PREREG c200a08 §5, secondary: diagnostics only)"
P6 = f"{C.PREREG} (PREREG c200a08 §6, descriptive, gates nothing)"
U_EVAL = (
    C.UNSEEN,
    "first read of the frozen cf_all head on EVAL [995000,997048); "
    "EVAL had fed run 1/run 2 for other policies (PREREG §14), never this head",
)
U_FIT = (
    C.UNSEEN,
    "no psi_hat had been fitted to Delta when the PREREG was committed "
    "(PREREG §14); U-val first read for this head here",
)


def _own(p, lo, hi, label=None) -> str:
    return f"{p:.6f} {lo:.6f} {hi:.6f}" + (f" {label}" if label else "")


def _seed_stat(
    rid,
    base,
    claim,
    *,
    tier="secondary",
    prereg,
    seen,
    ci_method,
    pick_base,
    own=None,
    own_label_key=None,
    note=None,
    point="point",
) -> C.Row:
    """Per-seed statistic at ledger ``seed<s>.<base>.{point,lo,hi}`` (scalar rows, or a
    dict row then sub-path); raw checks pick ``<pick_base>.<f>`` from the family
    recompute, and/or the run's own recompute ``own`` subcommand."""
    vals, cis, checks = [], [], []
    for s in SEEDS:
        got = {}
        for f in (point, "lo", "hi"):
            full = f"seed{s}.{base}.{f}"
            v = _leaf(full)
            got[f] = v
            if pick_base:
                checks.append(
                    C.Check(
                        label=f"s{s} {f}",
                        cmd=fam_cmd(RUN, s),
                        expected=C.fmt_exact(v),
                        pick=f"{pick_base}.{f}",
                    )
                )
        vals.append(got[point])
        cis.append((got["lo"], got["hi"]))
        if own:
            lab = _leaf(f"seed{s}.{own_label_key}") if own_label_key else None
            checks.append(
                C.Check(
                    label=f"s{s} own recompute {own}",
                    cmd=f"{OWN} {own} --seed {s}",
                    expected=_own(got[point], got["lo"], got["hi"], lab),
                )
            )
    return C.Row(
        id=rid,
        experiment=EXPERIMENT,
        tier=tier,
        claim=claim,
        seeds=SEEDS,
        values=vals,
        ci=cis,
        ci_method=ci_method,
        source=f"runs/{RUN}/ledger.json :: seed{{s}}.{base}.{{{point},lo,hi}}",
        prereg=prereg,
        seen=seen[0],
        seen_why=seen[1],
        checks=checks,
        note=note,
    )


def _leaf(full: str):
    """Committed-ledger value at a dotted key: the longest key prefix that is a row,
    then the remainder as a sub-path."""
    rows = C.lrows(RUN)
    parts = full.split(".")
    for i in range(len(parts), 0, -1):
        k = ".".join(parts[:i])
        if k in rows:
            return C.dig(rows[k], parts[i:]) if i < len(parts) else rows[k]
    raise KeyError(full)


def rows() -> list[C.Row]:
    L = C.lrows(RUN)
    sm = L["summary"]
    out: list[C.Row] = []
    out.append(
        _seed_stat(
            "ofpsi.deploy.cf_all-fifo",
            "heldout.headline",
            "offline-fit psi_hat (cf_all) deployed frozen - FIFO, all-query EVAL "
            "accuracy "
            "(the deploy headline)",
            tier="headline",
            prereg=P4,
            seen=U_EVAL,
            ci_method=EBOOT,
            pick_base="heldout.secondary.accuracy.all.minus_fifo",
            own="headline",
            own_label_key="heldout.headline.label",
        )
    )
    labels = [L[f"seed{s}.heldout.headline.label"] for s in SEEDS]
    out.append(
        C.Row(
            id="ofpsi.deploy.label",
            experiment=EXPERIMENT,
            tier="headline",
            claim=f"per-seed deploy label vs delta_seed (run 2's rule); deploy class "
            f"{sm['deploy_class']}, fit label {sm['fit_label']}, {sm['newcomer']}",
            seeds=SEEDS,
            values=labels,
            source=f"runs/{RUN}/ledger.json :: seed{{s}}.heldout.headline.label; summary."
            "{deploy_class,fit_label,newcomer}",
            prereg=P4,
            seen=U_EVAL[0],
            seen_why=U_EVAL[1],
            checks=[
                C.Check(
                    label="class + fit label + newcomer",
                    cmd=f"{OWN} class",
                    expected=f"{sm['deploy_class']} {sm['fit_label']} {sm['newcomer']}",
                )
            ],
        )
    )
    out.append(
        _seed_stat(
            "ofpsi.fit.cf_all.cap",
            "fit_reading.cf_all.cap",
            "fit cap of the selected cf_all head on U-val (B0-CF §7 cap; fit label "
            "input)",
            tier="headline",
            prereg=P5,
            seen=U_FIT,
            ci_method=UBOOT,
            pick_base="fit_reading.cf_all.cap",
            own="fitcap",
        )
    )
    out.append(
        _seed_stat(
            "ofpsi.fit.cf_live.cap",
            "fit_reading.cf_live.cap",
            "fit cap of the selected cf_live head on U-val",
            prereg=P5S,
            seen=U_FIT,
            ci_method=UBOOT,
            pick_base="fit_reading.cf_live.cap",
        )
    )
    out.append(
        _seed_stat(
            "ofpsi.fit.reference_G.cap",
            "fit_reading.reference_G.cap",
            "B0-CF's own G replayed as the ranking on U-val (reference cap; beside, "
            "never contrasted)",
            prereg=P5,
            seen=(
                C.SEEN,
                "B0-CF's own G: its cap over all of U was read in "
                "cf-ceiling (and is this run's C2); the U-val subset is new",
            ),
            ci_method=UBOOT,
            pick_base="fit_reading.reference_G.cap",
        )
    )
    out.append(
        _seed_stat(
            "ofpsi.fit.cf_all.rho_psi_G",
            "fit_reading.cf_all.rho_psi_G_fifo_resident",
            "within-step Spearman rho(psi_hat, G) over FIFO-resident slots on U-val, "
            "cf_all "
            "(median over steps)",
            prereg=P5,
            seen=U_FIT,
            ci_method=(
                "bootstrap percentile of the median over the U-val documents, n_boot 2000"
            ),
            pick_base="fit_reading.cf_all.rho_psi_G_fifo_resident",
            point="median",
        )
    )
    c2 = [L[f"seed{s}.controls.C2.computed"] for s in SEEDS]
    out.append(
        C.Row(
            id="ofpsi.c2.cap",
            experiment=EXPERIMENT,
            tier="secondary",
            claim=(
                "C2: B0-CF's G as the ranking over all of U, cap point (reproduces "
                "cf-ceiling's "
                "g0.9 cap)"
            ),
            seeds=SEEDS,
            values=c2,
            source=f"runs/{RUN}/ledger.json :: seed{{s}}.controls.C2.computed",
            prereg=f"{C.PREREG} (PREREG c200a08, control C2)",
            seen=C.SEEN,
            seen_why="cf-ceiling's B.ckpt3000.cf.g09.cap.point, read before this PREREG",
            checks=[
                C.Check(
                    label=f"s{s} own recompute c2",
                    cmd=f"{OWN} c2 --seed {s}",
                    expected=C.fmt_exact(c2[s]),
                )
                for s in SEEDS
            ],
        )
    )
    vl = [L[f"seed{s}.selection.cf_all.val_l_mc"] for s in SEEDS]
    out.append(
        C.Row(
            id="ofpsi.sel.cf_all.val_l_mc",
            experiment=EXPERIMENT,
            tier="secondary",
            claim="U-val L_MC of the selected cf_all head (selected step "
            + "/".join(str(L[f"seed{s}.selection.cf_all.step"]) for s in SEEDS)
            + ")",
            seeds=SEEDS,
            values=vl,
            source=(
                f"runs/{RUN}/ledger.json :: "
                "seed{s}.selection.cf_all.{step,val_l_mc,phi_sha256}"
            ),
            prereg=P5,
            seen=U_FIT[0],
            seen_why=U_FIT[1],
            checks=[
                C.Check(
                    label=f"s{s} own recompute refit (re-fits phi from scratch)",
                    cmd=f"{OWN} refit --seed {s} --threads 1",
                    expected=f"{L[f'seed{s}.selection.cf_all.step']} {vl[s]:.9e} "
                    f"{L[f'seed{s}.selection.cf_all.phi_sha256']}",
                )
                for s in SEEDS
            ],
        )
    )
    sh = [L[f"seed{s}.heldout.age1_share.cf_all.share"] for s in SEEDS]
    out.append(
        C.Row(
            id="ofpsi.deploy.age1_share",
            experiment=EXPERIMENT,
            tier="secondary",
            claim="share of EVAL evictions at age 1, cf_all (n_age1 / n_evictions)",
            seeds=SEEDS,
            values=sh,
            source=(
                f"runs/{RUN}/ledger.json :: "
                "seed{s}.heldout.age1_share.cf_all.{n_age1,n_evictions,share}"
            ),
            prereg=P6,
            seen=U_EVAL[0],
            seen_why=U_EVAL[1],
            checks=[
                C.Check(
                    label=f"s{s} own recompute age1",
                    cmd=f"{OWN} age1 --seed {s}",
                    expected=f"{L[f'seed{s}.heldout.age1_share.cf_all.n_age1']} "
                    f"{L[f'seed{s}.heldout.age1_share.cf_all.n_evictions']}",
                )
                for s in SEEDS
            ],
        )
    )
    for rid, k, claim in (
        ("ofpsi.deploy.acc.cf_all", "levels.cf_all", "EVAL all-query accuracy, cf_all"),
        (
            "ofpsi.deploy.acc.cf_all-random",
            "minus_random",
            "cf_all - random (mean of 5 draws), EVAL all-query accuracy",
        ),
        (
            "ofpsi.deploy.acc.cf_all-psiC_run2",
            "minus_psiC_run2",
            "cf_all - run 2's psiC, EVAL all-query accuracy",
        ),
    ):
        out.append(
            _seed_stat(
                rid,
                f"heldout.secondary.accuracy.all.{k}",
                claim,
                prereg=P6,
                seen=U_EVAL,
                ci_method=EBOOT,
                pick_base=f"heldout.secondary.accuracy.all.{k}",
            )
        )
    return out
