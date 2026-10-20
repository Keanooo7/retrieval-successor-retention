"""Fresh stream (experiments/fresh-stream; PREREG 83128ee, committed 2026-09-25 19:03Z,
before the 19:58Z run). FIFO only. Data: a non-repeating training stream, document ids
[4160, 52160); held-out set [4096, 4160) (PRIMARY, unseen by both arms); probe [0, 64);
bucket gap_2_to_M. No CI was computed: each number is a per-seed point value; the
PREREG's rule is "on every seed"."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from _raw import SEEDS, L, check, none_as, samples_row  # noqa: E402

EXPERIMENT = "fresh-stream"
RUN = EXPERIMENT
P = f"{C.PREREG} (PREREG 83128ee classification rule, before the run)"
P_SEC = f"{C.PREREG} (PREREG secondary readout, before the run; gates nothing)"
U = (
    C.UNSEEN,
    "new trainings (arm A fresh start; arm B continued on the stream); read first by "
    "this PREREG'd run",
)
HELD = "held-out [4096,4160), gap_2_to_M"


def rows() -> list[C.Row]:
    led = L(RUN)
    d = led["delta"]
    ct = led["c_threshold"]
    out: list[C.Row] = []
    for arm in ("A", "B"):
        out.append(samples_row(
            RUN, f"fs.R3000.{arm}", f"{arm}.ckpt3000.R_quantity",
            f"arm {arm} R(3000) = live - slots-zeroed answer accuracy, {HELD}; "
            f"R holds iff >= delta {d} on every seed: "
            f"**{'holds' if led[f'{arm}.R3000'] else 'fails'}**",
            tier="headline", prereg=P, seen=U[0], seen_why=U[1],
        ))
    reach = {a: led[f"{a}.ckpt3000.R_reaches_delta_on_seeds"] for a in ("A", "B")}
    out.append(C.Row(
        id="fs.classification", experiment=EXPERIMENT, tier="headline",
        claim=f"classification **{led['classification']}** (run-level, seeds 0-2: "
              f"{led['classification_detail']}); value per seed = whether R(3000) >= delta",
        seeds=SEEDS,
        values=[
            " / ".join(f"{a} {'>=d' if s in reach[a] else '<d'}" for a in ("A", "B"))
            for s in SEEDS
        ],
        source=f"runs/{RUN}/ledger.json :: classification; "
               "{A,B}.ckpt3000.R_reaches_delta_on_seeds",
        prereg=P, seen=U[0], seen_why=U[1],
        checks=[check(RUN, "classification"), check(RUN, "classification_detail")]
        + [check(RUN, f"{a}.ckpt3000.R_reaches_delta_on_seeds") for a in ("A", "B")],
    ))
    for arm in ("A", "B"):
        out.append(samples_row(
            RUN, f"fs.C3000.{arm}", f"{arm}.ckpt3000.C_quantity",
            f"arm {arm} C(3000) = live answer Brier/16, {HELD}; C holds iff <= {ct} "
            f"every seed: **{'holds' if led[f'{arm}.C3000'] else 'fails'}**",
            prereg=P, seen=U[0], seen_why=U[1],
        ))
        out.append(samples_row(
            RUN, f"fs.heldout_acc.{arm}", f"{arm}.ckpt3000.heldout.live.gap_2_to_M.answer_acc",
            f"arm {arm} held-out answer accuracy, memory live, ckpt 3000, {HELD}",
            prereg=P_SEC, seen=U[0], seen_why=U[1],
        ))
        out.append(samples_row(
            RUN, f"fs.probe_R.{arm}", f"{arm}.ckpt3000.probe_R_quantity",
            f"arm {arm} R on the probe [0,64) at ckpt 3000 (unseen by A; memorised by B)",
            prereg=P_SEC, seen=U[0], seen_why=U[1],
        ))
    out.append(samples_row(
        RUN, "fs.side_channel_share.B", "B.ckpt3000.side_channel_share",
        f"arm B side-channel share at ckpt 3000, {HELD}",
        prereg=P_SEC, seen=U[0], seen_why=U[1],
    ))
    out.append(C.Row(
        id="fs.B.start_R", experiment=EXPERIMENT, tier="secondary",
        claim=f"arm B's start: scaffold-timing n64.ckpt1000 R_quantity, {HELD}",
        seeds=SEEDS, values=led["B.start.R_quantity"],
        ci_method="none computed",
        source=f"runs/{RUN}/ledger.json :: B.start.R_quantity "
               "(copied from runs/scaffold-timing/ledger.json :: n64.ckpt1000.R_quantity)",
        prereg=P_SEC,
        seen=C.SEEN, seen_why="scaffold-timing's value, read in that run before this PREREG",
        checks=[check(RUN, "B.start.R_quantity", "samples s0-s2")],
    ))
    out.append(samples_row(
        RUN, "fs.B.growth_R", "B.growth_R_quantity",
        "arm B growth in R: R(3000) - R(start)", prereg=P_SEC, seen=U[0], seen_why=U[1],
    ))
    for arm in ("A", "B"):
        k = f"{arm}.stream_answer_loss_first_window_below"
        out.append(C.Row(
            id=f"fs.stream_loss_first_below.{arm}", experiment=EXPERIMENT, tier="secondary",
            claim=f"arm {arm}: first 100-step stream window whose mean answer loss < "
                  f"{led['stream_loss_threshold']:.4f} (ln16 - 0.10), keyed by first step",
            seeds=SEEDS,
            values=[none_as(led[k][f"seed{s}"], "none") for s in SEEDS],
            ci_method="none computed",
            source=f"runs/{RUN}/ledger.json :: {k} (from the archived heartbeats)",
            prereg=P_SEC, seen=U[0], seen_why=U[1],
            checks=[check(RUN, k, "per-seed dict")],
        ))
    return out
