"""Decisive shuffle (experiments/decisive-shuffle; PREREG 4242d5e + Amendment 1 7d64c2c,
both committed 2026-09-21 before the 11:28Z run). Data: S0-03's rewardable corpus;
the PRIMARY batch is #17's measurement batch (the first 8 training documents); the
descriptive held-out batch is S0-03 held-out documents 64..71. No CI was computed:
each number is a per-seed point value over one 8-document batch."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from _raw import SEEDS, L, check, module_constant, samples_row  # noqa: E402

EXPERIMENT = "decisive-shuffle"
RUN = EXPERIMENT
P = f"{C.PREREG} (PREREG 4242d5e decision rule, before the run)"
P_A1 = f"{C.PREREG} (Amendment 1 7d64c2c, descriptive, no verdict; before the run)"
P_SEC = f"{C.PREREG} (PREREG Secondary 1/2, descriptive; before the run)"
U = (C.UNSEEN, "PREREG committed 'before any S0-03 number exists'; first read by this run")
TRAIN = "train batch = first 8 S0-03 training docs (#17's batch)"
HELD = "held-out batch = S0-03 held-out docs 64..71"


def rows() -> list[C.Row]:
    led = L(RUN)
    rel = "experiments/decisive-shuffle/run.py"
    inert = module_constant(rel, "INERT_MAX_RATIO")
    live = module_constant(rel, "LIVE_MIN_RATIO")
    out: list[C.Row] = []
    for arm in ("A", "B", "C"):
        v = led[f"arm{arm}.verdict"]["outcome"]
        out.append(samples_row(
            RUN, f"ds.ratio.{arm}", f"arm{arm}.ratio",
            f"arm {arm} shuffle ratio A(trained)/A(untrained decoy), {TRAIN}; "
            f"arm verdict **{v}** (inert iff every seed <= {inert}, live iff any seed >= {live})",
            tier="headline", prereg=P, seen=U[0], seen_why=U[1],
        ))

    def band(x: float) -> str:
        return f"<={inert}" if x <= inert else (f">={live}" if x >= live else "between")

    per_seed = [
        " / ".join(f"{a} {band(led[f'arm{a}.ratio'][i])}" for a in ("A", "B", "C"))
        for i, _s in enumerate(SEEDS)
    ]
    ver = C.ledger(RUN)["verdict"]
    verdicts = {a: led[f"arm{a}.verdict"]["outcome"] for a in ("A", "B", "C")}
    out.append(C.Row(
        id="ds.falsifier", experiment=EXPERIMENT, tier="headline",
        claim=f"falsifier outcome **{ver['outcome']}** (run-level, seeds 0-2); per-arm "
              f"verdicts {verdicts}; value per seed = each arm's ratio band "
              f"(PREREG thresholds read from run.py)",
        seeds=SEEDS, values=per_seed,
        source=f"runs/{RUN}/ledger.json :: verdict.outcome; arm{{A,B,C}}.verdict; "
               "arm{A,B,C}.ratio",
        prereg=P, seen=U[0], seen_why=U[1],
        checks=[check(RUN, "__verdict__", "run verdict")]
        + [check(RUN, f"arm{a}.verdict", f"arm {a} verdict") for a in ("A", "B", "C")],
    ))
    for arm in ("A", "B", "C"):
        out.append(samples_row(
            RUN, f"ds.heldout_ratio.{arm}", f"arm{arm}.heldout.ratio",
            f"arm {arm} shuffle ratio, {HELD}", prereg=P_A1, seen=U[0], seen_why=U[1],
        ))
    out.append(samples_row(
        RUN, "ds.random_ratio.A.train", "armA.train.random_ratio",
        f"arm A matched-norm random-replacement ratio (Secondary 2), {TRAIN}",
        prereg=P_SEC, seen=U[0], seen_why=U[1],
    ))
    out.append(samples_row(
        RUN, "ds.cosine_trained.A.train", "armA.train.cosine_trained",
        f"arm A mean off-diagonal cross-row memory cosine (Secondary 2), {TRAIN}",
        prereg=P_SEC, seen=U[0], seen_why=U[1],
    ))
    for arm in ("B", "C"):
        for b in ("answer_all", "gap_1"):
            out.append(samples_row(
                RUN, f"ds.answer_nll.{arm}.{b}", f"arm{arm}.heldout.answer.{b}.honest_nll",
                f"arm {arm} answer-token NLL with own memory (nats), bucket {b}, {HELD}",
                prereg=P_A1, seen=U[0], seen_why=U[1],
            ))
    return out
