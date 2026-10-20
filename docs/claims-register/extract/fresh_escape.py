"""Fresh escape (experiments/fresh-escape; PREREG d5b9a23, committed 2026-09-26 04:37Z,
before the 05:10Z run). FIFO only; arm A of fresh-stream continued from its ckpt 3000
to 9000 on fresh stream ids [52160, 148160). Held-out [4096, 4160), probe [0, 64),
bucket gap_2_to_M. No CI was computed: per-seed point values; the PREREG's rule is "on
every seed"."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from _raw import SEEDS, L, check, none_as, samples_row  # noqa: E402

EXPERIMENT = "fresh-escape"
RUN = EXPERIMENT
P = f"{C.PREREG} (PREREG d5b9a23 classification rule, before the run)"
P_SEC = f"{C.PREREG} (PREREG readout, before the run; gates nothing)"
U = (
    C.UNSEEN,
    "steps 3000-9000 are new training; read first by this PREREG'd run (its design "
    "followed fresh-stream's arm-A plateau, a different quantity)",
)
HELD = "held-out [4096,4160), gap_2_to_M"


def rows() -> list[C.Row]:
    led = L(RUN)
    p = led["P"]
    out: list[C.Row] = []
    k_st = "stream_answer_loss_first_window_below_at_or_before_P"
    out.append(C.Row(
        id="fe.classification", experiment=EXPERIMENT, tier="headline",
        claim=f"classification **{led['classification_reported_as']}** (run-level, seeds 0-2: "
              f"{led['classification_detail']}); value per seed = first stream-loss window "
              f"below threshold at or before P = {p} (STIRRING iff any seed has one)",
        seeds=SEEDS,
        values=[none_as(led[k_st][f"seed{s}"], "none") for s in SEEDS],
        source=f"runs/{RUN}/ledger.json :: classification_reported_as; {k_st}; P; R_at_P",
        prereg=P, seen=U[0], seen_why=U[1],
        checks=[check(RUN, k) for k in
                ("classification_reported_as", "classification_detail", "P", "R_at_P",
                 "stirring", k_st)],
    ))
    out.append(samples_row(
        RUN, "fe.R_at_P", f"A.ckpt{p}.R_quantity",
        f"R(P = {p}) = live - slots-zeroed answer accuracy, {HELD}; holds iff >= delta "
        f"{led['delta']} every seed: **{'holds' if led['R_at_P'] else 'fails'}**",
        tier="headline", prereg=P, seen=U[0], seen_why=U[1],
    ))
    out.append(samples_row(
        RUN, "fe.C_at_P", f"A.ckpt{p}.C_quantity",
        f"C(P = {p}) = live answer Brier/16, {HELD}; holds iff <= {led['c_threshold']} "
        f"every seed: **{'holds' if led['C_at_P'] else 'fails'}**",
        prereg=P, seen=U[0], seen_why=U[1],
    ))
    out.append(samples_row(
        RUN, "fe.probe_R_at_P", f"A.ckpt{p}.probe_R_quantity",
        f"R on the probe [0,64) at P = {p}", prereg=P_SEC, seen=U[0], seen_why=U[1],
    ))
    out.append(samples_row(
        RUN, "fe.M_at_P", f"A.ckpt{p}.M_quantity",
        f"M_quantity (scaffold-timing's derived readout) at P = {p}, {HELD}",
        prereg=P_SEC, seen=U[0], seen_why=U[1],
    ))
    return out
