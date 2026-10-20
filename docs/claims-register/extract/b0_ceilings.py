"""B0 zero-cost ceilings (experiments/b0-ceilings; PREREG fc26dbf committed alone
2026-09-27 13:49 PDT, before the run at bb40c40, 2026-09-27T21:21Z). Descriptive, no
gate; model-free residency on the U range, which PREREG §2 declares ALREADY INSPECTED,
with class means estimated in-sample."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from _ceilings import cap_row

EXPERIMENT = "b0-ceilings"
RUN = "b0-ceilings"
P = f"{C.PREREG} (PREREG fc26dbf, descriptive, no gate)"
SEEN_U = (
    C.SEEN,
    "U range declared ALREADY INSPECTED (PREREG §2), class means in-sample; the "
    "ko/kb-vs-least-squares comparison (PLAN §1) was made after these caps were seen",
)
SEEN_REF = (
    C.SEEN,
    "U-range reference hit rates already read in lookahead-room-r2 (B0 C3)",
)


def rows() -> list[C.Row]:
    out: list[C.Row] = []
    g = "B.ckpt3000.g09"
    for rule, tier, claim, note in (
        (
            "ko",
            "headline",
            "kind-oracle (ko, content-only, random tie) cap = (hit - FIFO)/(FactFiller - FIFO), "
            "ckpt3000, gamma 0.9",
            None,
        ),
        (
            "kb",
            "headline",
            "kind x age-band (kb) cap, ckpt3000, gamma 0.9",
            "kb reads age: NOT a legal psi_hat",
        ),
        (
            "ko_oldest",
            "secondary",
            "kind-oracle, ties to oldest (ko_oldest) cap, ckpt3000, gamma 0.9",
            None,
        ),
        ("age", "secondary", "age-only rule cap, ckpt3000, gamma 0.9", "reads age only"),
    ):
        out.append(
            cap_row(
                RUN,
                "b0",
                f"b0.cap.{rule}.g09",
                f"{g}.cap.{rule}",
                claim,
                tier=tier,
                prereg=P,
                seen=SEEN_U[0],
                seen_why=SEEN_U[1],
                note=note,
            )
        )
    for h, gm in (("rule_g09", "0.9"), ("rule_g0", "0")):
        out.append(
            cap_row(
                RUN,
                "b0",
                f"b0.cap.{h}",
                f"B.ckpt3000.cap.{h}",
                f"lookahead-room hindsight rule TargetPolicy(G_r_i, gamma {gm}) cap, ckpt3000",
                tier="secondary",
                prereg=P,
                seen=C.SEEN,
                seen_why="already read in lookahead-room-r2 (reproduced as B0 C3)",
            )
        )
    for ref in ("fifo", "factfiller"):
        out.append(
            cap_row(
                RUN,
                "b0",
                f"b0.hit.{ref}",
                f"U.hit.{ref}.all",
                f"U-range all-query residency hit rate, {ref} (the cap's reference)",
                tier="secondary",
                prereg=P,
                seen=SEEN_REF[0],
                seen_why=SEEN_REF[1],
                with_ci=False,
            )
        )
    out.append(
        cap_row(
            RUN,
            "b0",
            "b0.cap.oracle",
            "U.cap.oracle",
            "oracle (discounted demand, gamma_o 0.97) cap on U",
            tier="secondary",
            prereg=P,
            seen=SEEN_REF[0],
            seen_why=SEEN_REF[1],
        )
    )
    return out
