"""Run 2 (experiments/rsr-run2; PREREG a406d39 + Amendment 1 d8dfd14, committed
2026-10-07 before the 2026-10-08 run; verified at 9c432b6)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from _heldout import SEEDS, stat_row

EXPERIMENT = "rsr-run2"
RUN = "rsr-run2"
ARCH = "/Users/keanooo7/rsr-runs-archive/rsr-run2-run/runs/rsr-run2"
OWN = f"{C.SLOT}{C.PY} experiments/rsr-run2/run.py recompute --out-root {ARCH}"
P = f"{C.PREREG} (PREREG a406d39, before the run)"
P_SEC = f"{C.PREREG} (PREREG a406d39 §4, secondary, gates nothing)"
U = (C.UNSEEN, "run 2's psiC/age arms are new; read first by this PREREG'd run")
S_EVAL = (
    C.SEEN,
    "policy-independent pass on EVAL [995000,997048); the same pass was read in run 1 "
    "(run 2's C12 checks it equals run 1's)",
)


def _own_line(fmt6: tuple, label: str) -> str:
    return " ".join(f"{x:.6f}" for x in fmt6) + f" {label}"


def rows() -> list[C.Row]:
    L = C.lrows(RUN)
    out: list[C.Row] = []
    hp = "contrasts.acc.all.psiC_minus_fifo"
    r = stat_row(
        RUN,
        "run2.acc.psiC-fifo",
        hp,
        "psiC - FIFO all-query held-out accuracy (the headline)",
        tier="headline",
        prereg=P,
        seen=U[0],
        seen_why=U[1],
    )
    # the run's own recompute path as a second, independent raw check
    for s in SEEDS:
        h = (
            L[f"seed{s}.run2.heldout.headline.point"],
            L[f"seed{s}.run2.heldout.headline.lo"],
            L[f"seed{s}.run2.heldout.headline.hi"],
        )
        r.checks.append(
            C.Check(
                label=f"s{s} own recompute headline",
                cmd=f"{OWN} headline --seed {s}",
                expected=_own_line(h, L[f"seed{s}.run2.heldout.headline.label"]),
            )
        )
    out.append(r)
    labels = [L[f"seed{s}.run2.heldout.headline.label"] for s in SEEDS]
    deltas = [L[f"seed{s}.run2.heldout.headline.delta"] for s in SEEDS]
    summ = L["summary"]
    out.append(
        C.Row(
            id="run2.label",
            experiment=EXPERIMENT,
            tier="headline",
            claim=(
                "per-seed WIN/LOSS/EQUIV label vs delta_seed; run class "
                f"{summ['class']} "
                f"(newcomer {summ['newcomer']})"
            ),
            seeds=SEEDS,
            values=labels,
            source=f"runs/{RUN}/ledger.json :: "
            "seed{s}.run2.heldout.headline.label; summary.class",
            prereg=P,
            seen=U[0],
            seen_why=U[1],
            checks=[
                C.Check(
                    label="class + newcomer",
                    cmd=f"{OWN} class",
                    expected=f"{summ['class']} {summ['newcomer']}",
                )
            ],
        )
    )
    out.append(
        C.Row(
            id="run2.delta",
            experiment=EXPERIMENT,
            tier="secondary",
            claim="delta_seed (the EQUIV margin) used by the label rule",
            seeds=SEEDS,
            values=deltas,
            source=(
                "runs/b2-psi-probe-fit/ledger.json :: "
                "ckpt3000.seed{s}.decisions.delta "
                f"(copied to runs/{RUN}/ledger.json :: seed{{s}}.controls.Cdelta.delta)"
            ),
            prereg=P,
            seen=C.SEEN,
            seen_why="B2's measured delta, fixed before run 1 and reused",
            checks=[
                C.ledger_check(
                    "b2-psi-probe-fit", f"ckpt3000.seed{s}.decisions.delta", None, f"s{s}"
                )
                for s in SEEDS
            ],
        )
    )
    for arm, why in (
        ("psiC", U),
        ("age", U),
        ("fifo", S_EVAL),
        ("oracle", S_EVAL),
        ("lru", S_EVAL),
        ("random", S_EVAL),
    ):
        out.append(
            stat_row(
                RUN,
                f"run2.acc.{arm}",
                f"accuracy.all.{arm}",
                f"held-out all-query accuracy, {arm}"
                + (
                    " (mean of 5 random draws' pooled accuracies)"
                    if arm == "random"
                    else ""
                ),
                prereg=P,
                seen=why[0],
                seen_why=why[1],
            )
        )
    out.append(
        stat_row(
            RUN,
            "run2.acc.age-fifo",
            "contrasts.acc.all.age_minus_fifo",
            "learned age-only head - FIFO accuracy",
            prereg=P,
            seen=U[0],
            seen_why=U[1],
            note="sd exactly 0: the age head's records are identical to FIFO's",
        )
    )
    out.append(
        stat_row(
            RUN,
            "run2.acc.psiC-fifo.gap_2_to_M",
            "contrasts.acc.gap_2_to_M.psiC_minus_fifo",
            "psiC - FIFO accuracy, gap 2..M bucket",
            prereg=P,
            seen=U[0],
            seen_why=U[1],
        )
    )
    out.append(
        stat_row(
            RUN,
            "run2.acc.psiC-fifo.gap_gt_M",
            "contrasts.acc.gap_gt_M.psiC_minus_fifo",
            "psiC - FIFO accuracy, gap > M bucket",
            prereg=P,
            seen=U[0],
            seen_why=U[1],
        )
    )
    for arm, why in (("psiC", U), ("fifo", S_EVAL), ("age", U), ("oracle", S_EVAL)):
        out.append(
            stat_row(
                RUN,
                f"run2.nll.real.{arm}",
                f"nll.real.{arm}",
                f"held-out real-token NLL, {arm}",
                prereg=P,
                seen=why[0],
                seen_why=why[1],
            )
        )
    for k in range(5):
        out.append(
            stat_row(
                RUN,
                f"run2.nll.real.random{k}",
                f"nll.real.random{k}",
                f"held-out real-token NLL, random draw {k}",
                prereg=P,
                seen=S_EVAL[0],
                seen_why=S_EVAL[1],
            )
        )
    out.append(
        stat_row(
            RUN,
            "run2.nll.real.psiC-fifo",
            "contrasts.nll_real.psiC_minus_fifo",
            "psiC - FIFO real-token NLL (positive = psiC worse)",
            prereg=P,
            seen=U[0],
            seen_why=U[1],
        )
    )
    out.append(
        stat_row(
            RUN,
            "run2.headroom_share.psiC",
            "headroom_share.psiC",
            "headroom share (psiC - FIFO) / (oracle - FIFO), all queries",
            prereg=P,
            seen=U[0],
            seen_why=U[1],
        )
    )
    # run-2-specific numbers (seed<s>.run2.heldout.*): own recompute where it exists
    es = [
        (L[f"seed{s}.run2.heldout.psiC_minus_es.all.{f}"])
        for s in SEEDS
        for f in ("point", "lo", "hi")
    ]
    out.append(
        C.Row(
            id="run2.acc.psiC-es",
            experiment=EXPERIMENT,
            tier="secondary",
            claim="psiC - pinned Expire-Span (L48-k1e-2) accuracy, all queries; labels "
            + "/".join(L[f"seed{s}.run2.heldout.psiC_minus_es.all.label"] for s in SEEDS),
            seeds=SEEDS,
            values=es[0::3],
            ci=list(zip(es[1::3], es[2::3], strict=True)),
            ci_method="paired per-document bootstrap percentile, n_boot 2000",
            source=(
                f"runs/{RUN}/ledger.json :: seed{{s}}.run2.heldout.psiC_minus_es.all.*"
            ),
            prereg=P_SEC,
            seen=U[0],
            seen_why=U[1],
            checks=[
                C.Check(
                    label=f"s{s} own recompute es",
                    cmd=f"{OWN} es --seed {s}",
                    expected=_own_line(
                        tuple(es[3 * s : 3 * s + 3]),
                        L[f"seed{s}.run2.heldout.psiC_minus_es.all.label"],
                    ),
                )
                for s in SEEDS
            ],
        )
    )
    rr = [
        (L[f"seed{s}.run2.heldout.psiC_run2_minus_run1.{f}"])
        for s in SEEDS
        for f in ("point", "lo", "hi")
    ]
    out.append(
        C.Row(
            id="run2.acc.psiC_run2-run1",
            experiment=EXPERIMENT,
            tier="secondary",
            claim="psiC(run 2) - psiC(run 1) accuracy, all queries",
            seeds=SEEDS,
            values=rr[0::3],
            ci=list(zip(rr[1::3], rr[2::3], strict=True)),
            ci_method="paired per-document bootstrap percentile, n_boot 2000",
            source=(
                f"runs/{RUN}/ledger.json :: seed{{s}}.run2.heldout.psiC_run2_minus_run1.*"
            ),
            prereg=P_SEC,
            seen=U[0],
            seen_why=U[1],
            checks=[
                C.archive_ledger_check(
                    f"{ARCH}/ledger.json",
                    RUN,
                    f"seed{s}.run2.heldout.psiC_run2_minus_run1.{f}",
                    None,
                    f"s{s} {f}",
                )
                for s in SEEDS
                for f in ("point", "lo", "hi")
            ],
        )
    )
    for arm, key, why in (
        ("psiC", "age1_share.psiC", U),
        (
            "run 1 psiC",
            "age1_share_run1_psiC_SEEN",
            (
                C.SEEN,
                "run 1's psiC records, read in run 1; re-tabulated here (PREREG §5)",
            ),
        ),
    ):
        n1 = [L[f"seed{s}.run2.heldout.{key}.n_age1"] for s in SEEDS]
        nn = [L[f"seed{s}.run2.heldout.{key}.n_evictions"] for s in SEEDS]
        sh = [L[f"seed{s}.run2.heldout.{key}.share"] for s in SEEDS]
        checks = (
            [
                C.Check(
                    label=f"s{s} own recompute age1",
                    cmd=f"{OWN} age1 --seed {s}",
                    expected=f"{n1[s]} {nn[s]}",
                )
                for s in SEEDS
            ]
            if arm == "psiC"
            else [
                C.archive_ledger_check(
                    f"{ARCH}/ledger.json",
                    RUN,
                    f"seed{s}.run2.heldout.{key}.share",
                    None,
                    f"s{s}",
                )
                for s in SEEDS
            ]
        )
        out.append(
            C.Row(
                id=f"run2.age1_share.{'psiC' if arm == 'psiC' else 'run1_psiC'}",
                experiment=EXPERIMENT,
                tier="secondary",
                claim=(
                    f"share of held-out evictions at age 1, {arm} (n_age1 / n_evictions)"
                ),
                seeds=SEEDS,
                values=sh,
                source=f"runs/{RUN}/ledger.json :: seed{{s}}.run2.heldout.{key}.share",
                prereg=P_SEC,
                seen=why[0],
                seen_why=why[1],
                checks=checks,
            )
        )
    return out
