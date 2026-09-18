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

## Two repairs from cycle 0 of the 2026-09-19 run

🔴 **A first reading exits 3, not 0.** It used to write `outcome: "inconclusive"`,
return verdict `"baseline"`, and exit **0** -- the exact `3`-collapsing-to-`0`
shape, in the one script that had no exit-3 branch at all. `3` means *did not run*
and it is not *found nothing*.

🔴 **A length mismatch is a `MOVED`, not a comparison over the overlap.** It used to
note the mismatch and compare the shared prefix, so a run that produced 2 of 6 beats
could report "held" -- a truncated run reading as a clean environment.

Exit codes follow the run protocol: `0` pass · `1` real failure · `2` nothing to
compare · `3` did not run.
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
CONFIG = dict(d=128, steps_per_stream=48, batch=8, iters=6, seed=0,
              memory_slots=16, lr=1e-3, device="mps", beat_every=1, ckpt_every=0)
REL_TOL = 1e-4          # committed before the first reading
BASELINE = _REPO / "runs" / "canary" / "baseline.json"


#: `0` pass · `1` real failure · `2` nothing to compare · `3` did not run.
#: 🔴 A first reading is `3`. It never collapses to `0`.
EXIT_CODES = {"held": 0, "MOVED": 1, "baseline": 3}


def exit_code_for(verdict: str) -> int:
    """The process exit status for a canary verdict.

    A `baseline` reading did not compare anything against anything: there was no
    baseline to compare to. That is *did not run*, and `runs/canary/cycle-04`'s own
    ledger says `inconclusive` while the process exited 0 -- which is how the night
    reported "3 canaries, all held" over 2 survived and 1 inconclusive.
    """
    try:
        return EXIT_CODES[verdict]
    except KeyError:
        raise ValueError(f"unknown canary verdict {verdict!r}") from None


def compare(baseline: list[float], losses: list[float]) -> tuple[str, list[dict], str]:
    """`(verdict, beats_outside_tol, detail)`.

    🔴 **Unequal lengths are a move.** Comparing over the overlap lets a run that
    produced 2 of 6 beats report "held": the beats it did produce match, and the
    four it never reached are simply not looked at. A canary that cannot see a
    truncated run is not an environment check.
    """
    n = min(len(baseline), len(losses))
    if len(baseline) != len(losses):
        return "MOVED", [], (
            f"beat-count length mismatch: baseline has {len(baseline)}, this run "
            f"produced {len(losses)}. A short run is a moved environment, not a "
            f"held one -- the overlap is not compared."
        )
    moved = [
        {"beat": i, "baseline": baseline[i], "now": losses[i],
         "rel": abs(losses[i] - baseline[i]) / max(abs(baseline[i]), 1e-12)}
        for i in range(n)
        if abs(losses[i] - baseline[i]) / max(abs(baseline[i]), 1e-12) > REL_TOL
    ]
    if moved:
        return "MOVED", moved, f"{len(moved)} of {n} beats outside rel_tol={REL_TOL}"
    return "held", [], f"all {n} beats within rel_tol={REL_TOL}"


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

    run_id = f"canary/cycle-{cycle:02d}"
    led = Ledger(run_id, cycle=cycle,
                 question="has the environment moved since the canary baseline?")
    led.manifest(CONFIG)
    led.run_meta(device=CONFIG["device"], seeds_actually_run=[CONFIG["seed"]],
                 steps_requested=CONFIG["iters"], steps_done=len(losses))
    led.command(f".venv/bin/python scripts/canary.py {cycle}", exit_code=0,
                note="frozen config, seed 0; see CONFIG in scripts/canary.py")
    led.note("losses", losses, how="heartbeat.jsonl beat records, field 'loss'")
    led.note("config", CONFIG, how="frozen literal in scripts/canary.py")
    led.note("rel_tol", REL_TOL,
             how="committed in scripts/canary.py before the first reading")

    if not BASELINE.exists():
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(
            json.dumps({"cycle": cycle, "losses": losses}, indent=2) + "\n")
        led.note("role", "baseline",
                 how="baseline.json did not exist; this reading defines it")
        led.verdict(falsifier="the environment has not moved",
                    outcome="inconclusive",
                    detail="first reading: this IS the baseline, nothing to compare "
                           "against yet. Exits 3 (did not run), never 0.")
        led.status("partial")
        verdict, moved, detail = "baseline", [], "first reading"
    else:
        base = json.loads(BASELINE.read_text())["losses"]
        verdict, moved, detail = compare(base, losses)
        led.note("baseline_losses", base, how="runs/canary/baseline.json")
        led.note("beats_outside_tol", moved,
                 how="elementwise |now-base|/|base| vs REL_TOL")
        led.note("beat_counts", {"baseline": len(base), "now": len(losses)},
                 how="len of the two beat sequences")
        led.verdict(
            falsifier="the environment has not moved since the canary baseline",
            outcome="falsified" if verdict == "MOVED" else "survived",
            detail=detail,
        )
        led.status("ok")

    path = led.write()
    return {"verdict": verdict, "moved": moved, "losses": losses,
            "detail": detail, "ledger": str(path),
            "exit_code": exit_code_for(verdict)}


if __name__ == "__main__":
    cycle = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    r = run(cycle)
    print(json.dumps({k: v for k, v in r.items() if k != "losses"}, indent=2))
    print("losses:", [round(x, 6) for x in r["losses"]])
    sys.exit(r["exit_code"])
