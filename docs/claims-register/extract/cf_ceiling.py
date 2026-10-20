"""B0-CF counterfactual-return ceiling (experiments/cf-ceiling; PREREG 0c5320e committed
alone 2026-10-06 00:23 PDT, before any Delta on U existed; run at 03859e2,
2026-10-06T10:26Z). Descriptive (Tier D); its reading rule fed owner option C3."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from _ceilings import cap_row, label_row

EXPERIMENT = "cf-ceiling"
RUN = "cf-ceiling"
P = f"{C.PREREG} (PREREG 0c5320e §7 reading rule, descriptive)"
U = (
    C.UNSEEN,
    "no Delta on U existed when the PREREG was committed (U itself was already inspected "
    "for r_i, not for Delta)",
)
GAMMAS = (("g0", "0"), ("g05", "0.5"), ("g08", "0.8"), ("g09", "0.9"))


def rows() -> list[C.Row]:
    out = [
        cap_row(
            RUN,
            "cf",
            f"cf.cap.{g}",
            f"B.ckpt3000.cf.{g}.cap",
            f"hindsight TargetPolicy(G^Delta_gamma) cap, primary arm, ckpt3000, gamma {gm} "
            "(FIFO-world LOO-resample Delta)",
            tier="headline",
            prereg=P,
            seen=U[0],
            seen_why=U[1],
        )
        for g, gm in GAMMAS
    ]
    out.append(
        label_row(
            RUN,
            "cf",
            "cf.labels",
            "per-gamma reading-rule labels at ckpt3000 (PREREG §7)",
            prereg=P,
            seen=U[0],
            seen_why=U[1],
            class_key="cf_class",
        )
    )
    out.append(
        cap_row(
            RUN,
            "cf",
            "cf.cap.cfshift.g09",
            "B.ckpt3000.cfshift.g09.cap",
            "sensitivity arm G^{Delta shift} cap, ckpt3000, gamma 0.9 (DIAGNOSTIC, unlabelled)",
            tier="secondary",
            prereg=f"{C.PREREG} (PREREG 0c5320e, diagnostic arm)",
            seen=U[0],
            seen_why=U[1],
        )
    )
    return out
