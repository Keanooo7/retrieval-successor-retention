"""B0-gamma hindsight target-ceiling sweep (experiments/gamma-ceiling; PREREG daabdb2
committed alone 2026-10-05 23:44 PDT, before the run at 02a7eeb, 2026-10-06T06:47Z).
Descriptive; gamma 0 and 0.9 had already been seen in B0 (PREREG front matter
``already_seen``), the other five gammas had not."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from _ceilings import cap_row, label_row

EXPERIMENT = "gamma-ceiling"
RUN = "gamma-ceiling"
P = f"{C.PREREG} (PREREG daabdb2 §6 reading rule, descriptive)"
GAMMAS = (
    ("g0", "0"),
    ("g05", "0.5"),
    ("g08", "0.8"),
    ("g09", "0.9"),
    ("g095", "0.95"),
    ("g099", "0.99"),
    ("g1", "1.0"),
)
SEEN_B0 = (C.SEEN, "gamma 0 and 0.9 already seen in B0 (PREREG already_seen)")
NOT_SEEN = (C.UNSEEN, "PREREG already_seen: this gamma NOT seen by anyone before the run")


def rows() -> list[C.Row]:
    out = []
    for g, gm in GAMMAS:
        sw = SEEN_B0 if g in ("g0", "g09") else NOT_SEEN
        out.append(
            cap_row(
                RUN,
                "gamma",
                f"gamma.cap.{g}",
                f"B.ckpt3000.{g}.cap",
                "hindsight TargetPolicy(G_gamma on r_i demand D) cap, ckpt3000, gamma "
                f"{gm}",
                tier="headline",
                prereg=P,
                seen=sw[0],
                seen_why=sw[1],
            )
        )
    out.append(
        label_row(
            RUN,
            "gamma",
            "gamma.labels",
            "per-gamma reading-rule labels at ckpt3000 (PREREG §6)",
            prereg=P,
            seen=C.SEEN,
            seen_why=(
                "the class reads all 7 gammas; gamma 0 and 0.9 were already seen (B0)"
            ),
            class_key="sweep_class",
        )
    )
    return out
