"""The environment canary: one fixed config, one fixed seed, re-run on a schedule.

**What it is for.** Over a long unattended session the thing most likely to
invalidate a night's work is not a wrong experiment -- it is the floor moving. A
torch cache rebuilt, a different MPS allocator state, a machine that thermally
throttled, an accidental edit to the data path. Every result after the move is
suspect and there is no way to tell from the results themselves.

So: the same config, the same seed, re-run every fourth cycle. If the number moves,
the environment moved, and the manager says so rather than reasoning past it.

**The canary's number is the loss sequence, not the throughput.** Throughput moves
with whatever else is on the GPU -- a second run, a browser, Spotlight -- and would
false-alarm every time two things overlapped. The loss at a fixed seed is a property
of the code and the data path, which is what we actually need to hold still.
Throughput is recorded anyway, as context for a loss move, never as the trigger.

Tolerance: MPS reductions are not bit-reproducible across runs in general, so an
exact-equality canary would cry wolf. The threshold is committed here, **before the
first reading**, at `1e-4` relative on every beat -- far tighter than any real
regression and far looser than float noise. A move outside it is reported as a move.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "src"))

from ledger import Ledger  # noqa: E402

# -- the frozen canary config. Do not tune these. ----------------------------- #
CONFIG = dict(
    d=128,
    steps_per_stream=48,
    batch=8,
    iters=6,
    seed=0,
    memory_slots=16,
    lr=1e-3,
    device="mps",
    beat_every=1,
    ckpt_every=0,
)
REL_TOL = 1e-4  # committed before the first reading
BASELINE = _REPO / "runs" / "canary" / "baseline.json"


def _losses(hb: Path) -> list[float]:
    out = []
    for line in hb.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("kind") == "beat":
            out.append(float(rec["loss"]))
    return out


def run(cycle: int) -> dict:
    from rsr.train.loop import train

    out_dir = _REPO / "runs" / "canary" / f"cycle-{cycle:02d}"
    train(out_dir=out_dir, **CONFIG)
    losses = _losses(out_dir / "heartbeat.jsonl")

    led = Ledger(
        f"canary/cycle-{cycle:02d}",
        cycle=cycle,
        question="has the environment moved since the canary baseline?",
    )
    led.command(
        f".venv/bin/python scripts/canary.py {cycle}",
        exit_code=0,
        note="frozen config, seed 0; see CONFIG in scripts/canary.py",
    )
    led.note("losses", losses, how="heartbeat.jsonl beat records, field 'loss'")
    led.note("config", CONFIG, how="frozen literal in scripts/canary.py")
    led.note(
        "rel_tol", REL_TOL, how="committed in scripts/canary.py before the first reading"
    )

    if not BASELINE.exists():
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(
            json.dumps({"cycle": cycle, "losses": losses}, indent=2) + "\n"
        )
        led.note(
            "role", "baseline", how="baseline.json did not exist; this reading defines it"
        )
        led.verdict(
            falsifier="the environment has not moved",
            outcome="inconclusive",
            detail="first reading: this IS the baseline, nothing to compare against yet",
        )
        verdict, moved = "baseline", []
    else:
        base = json.loads(BASELINE.read_text())["losses"]
        n = min(len(base), len(losses))
        moved = [
            {
                "beat": i,
                "baseline": base[i],
                "now": losses[i],
                "rel": abs(losses[i] - base[i]) / max(abs(base[i]), 1e-12),
            }
            for i in range(n)
            if abs(losses[i] - base[i]) / max(abs(base[i]), 1e-12) > REL_TOL
        ]
        led.note("baseline_losses", base, how="runs/canary/baseline.json")
        led.note(
            "beats_outside_tol", moved, how="elementwise |now-base|/|base| vs REL_TOL"
        )
        if len(base) != len(losses):
            led.note(
                "length_mismatch",
                {"baseline": len(base), "now": len(losses)},
                how="len of the two beat sequences",
            )
        led.verdict(
            falsifier="the environment has not moved since the canary baseline",
            outcome="falsified" if moved else "survived",
            detail=(
                f"{len(moved)} of {n} beats outside rel_tol={REL_TOL}"
                if moved
                else f"all {n} beats within rel_tol={REL_TOL}"
            ),
        )
        verdict = "MOVED" if moved else "held"

    path = led.write()
    return {"verdict": verdict, "moved": moved, "losses": losses, "ledger": str(path)}


if __name__ == "__main__":
    cycle = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    r = run(cycle)
    print(json.dumps({k: v for k, v in r.items() if k != "losses"}, indent=2))
    print("losses:", [round(x, 6) for x in r["losses"]])
    if r["verdict"] == "MOVED":
        sys.exit(2)
