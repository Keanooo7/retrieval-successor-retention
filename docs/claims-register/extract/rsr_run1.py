"""Run 1 (experiments/rsr-run1; PREREG 7c58777 + Amendments A1..A4, the last 43de9f5,
all committed 2026-10-01; launched 2026-10-06 at ed6302a; verified at 157f1fb).

Run 1 is DESCRIPTIVE (RESULTS §1: "No gate, no pass/fail threshold"): its headline is
read only as CI above / below / includes 0; it has no WIN/LOSS label and no class.
EVAL [995000, 997048) was fresh at run 1 (PREREG: "never trained on by any arm"), so
every held-out number here was first read by this run."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from _heldout import SEEDS, stat_row

EXPERIMENT = "rsr-run1"
RUN = "rsr-run1"
ARCH = "/Users/keanooo7/rsr-runs-archive/rsr-run1-run/runs/rsr-run1"
P = f"{C.PREREG} (PREREG 7c58777..A4 43de9f5, before the run; descriptive, no gate)"
U = (C.UNSEEN, "EVAL [995000,997048) fresh at run 1; first read by this PREREG'd run")


def rows() -> list[C.Row]:
    out: list[C.Row] = []
    r = stat_row(
        RUN,
        "run1.acc.psiC-fifo",
        "contrasts.acc.all.psiC_minus_fifo",
        "psiC - FIFO all-query held-out accuracy (run 1's one designated contrast; "
        "read only as CI above/below/includes 0)",
        tier="headline",
        prereg=P,
        seen=U[0],
        seen_why=U[1],
    )
    # the ledger's own `headline` row restates the per-seed contrast: read it from the
    # run's own out-root ledger in the archive (not the committed copy)
    for s in SEEDS:
        for f in ("point", "lo", "hi"):
            r.checks.append(
                C.archive_ledger_check(
                    f"{ARCH}/ledger.json",
                    RUN,
                    "headline",
                    f"per_seed.{s}.{f}",
                    f"s{s} headline.{f} (archive ledger)",
                )
            )
    out.append(r)
    for arm in ("psiC", "age", "fifo", "oracle", "lru", "random"):
        out.append(
            stat_row(
                RUN,
                f"run1.acc.{arm}",
                f"accuracy.all.{arm}",
                f"held-out all-query accuracy, {arm}"
                + (
                    " (mean of 5 random draws' pooled accuracies)"
                    if arm == "random"
                    else ""
                ),
                prereg=P,
                seen=U[0],
                seen_why=U[1],
            )
        )
    out.append(
        stat_row(
            RUN,
            "run1.acc.age-fifo",
            "contrasts.acc.all.age_minus_fifo",
            "learned age-only head - FIFO accuracy",
            prereg=P,
            seen=U[0],
            seen_why=U[1],
        )
    )
    out.append(
        stat_row(
            RUN,
            "run1.acc.psiC-fifo.gap_2_to_M",
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
            "run1.acc.psiC-fifo.gap_gt_M",
            "contrasts.acc.gap_gt_M.psiC_minus_fifo",
            "psiC - FIFO accuracy, gap > M bucket",
            prereg=P,
            seen=U[0],
            seen_why=U[1],
        )
    )
    for arm in ("psiC", "fifo", "age", "oracle"):
        out.append(
            stat_row(
                RUN,
                f"run1.nll.real.{arm}",
                f"nll.real.{arm}",
                f"held-out real-token NLL, {arm}",
                prereg=P,
                seen=U[0],
                seen_why=U[1],
            )
        )
    out.append(
        stat_row(
            RUN,
            "run1.nll.real.psiC-fifo",
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
            "run1.headroom_share.psiC",
            "headroom_share.psiC",
            "headroom share (psiC - FIFO) / (oracle - FIFO), all queries",
            prereg=P,
            seen=U[0],
            seen_why=U[1],
        )
    )
    return out
