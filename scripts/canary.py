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

🔴 **A first reading exits 2 -- not 0, and not 3.** It used to write
`outcome: "inconclusive"`, return verdict `"baseline"`, and exit **0**. Cycle 0
"fixed" that to **3**, which is the same collapse inverted (S0-05): a first reading
*ran* -- it trained, read the beats and wrote `baseline.json` -- and what it could
not do is *compare*, because there was nothing to compare against. That is `2`,
`docs/gates.md`'s "no recorded floor for this key". `3` is reserved for a canary
that did not run.

🔴 **A length mismatch is a `MOVED`, not a comparison over the overlap.** It used to
note the mismatch and compare the shared prefix, so a run that produced 2 of 6 beats
could report "held" -- a truncated run reading as a clean environment.

Exit codes follow the run protocol, all five of them (`rsr.exit_codes.Exit`,
`docs/gates.md`): `0` held · `1` MOVED · `2` nothing to compare · `3` did not run ·
`4` unbanked rise. This script emits `0`, `1` and `2`. It has no floor to bank, so
`4` is unreachable here, not forgotten; a crash inside `train()` propagates as an
uncaught exception (`1`), because a fixed-seed fixed-config run that crashes is an
environment that moved. (This docstring listed four codes until S0-05 and silently
dropped `4`.)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "src"))

from ledger import Ledger  # noqa: E402

from rsr.exit_codes import Exit, run_main  # noqa: E402

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


#: `0` held · `1` MOVED · `2` nothing to compare. See `rsr.exit_codes.Exit`.
#: 🔴 A first reading is `2`: it ran and had nothing to compare. Not `0`, not `3`.
EXIT_CODES = {"held": Exit.OK, "MOVED": Exit.FAIL, "baseline": Exit.UNKNOWN}


def exit_code_for(verdict: str) -> Exit:
    """The process exit status for a canary verdict.

    A `baseline` reading ran and compared nothing against anything: there was no
    baseline to compare to. That is *nothing to compare* (`2`). It is not a pass --
    `runs/canary/cycle-04`'s own ledger says `inconclusive` while the process exited
    0, which is how the night reported "3 canaries, all held" over 2 survived and 1
    inconclusive -- and it is not *did not run* either, which is what cycle 0's fix
    mapped it to (S0-05; `docs/gates.md`: "`2` and `3` are separate deliberately").
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
        return (
            "MOVED",
            [],
            (
                f"beat-count length mismatch: baseline has {len(baseline)}, this run "
                f"produced {len(losses)}. A short run is a moved environment, not a "
                f"held one -- the overlap is not compared."
            ),
        )
    moved = [
        {
            "beat": i,
            "baseline": baseline[i],
            "now": losses[i],
            "rel": abs(losses[i] - baseline[i]) / max(abs(baseline[i]), 1e-12),
        }
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
    led = Ledger(
        run_id,
        cycle=cycle,
        question="has the environment moved since the canary baseline?",
    )
    led.manifest(CONFIG)
    led.run_meta(
        device=CONFIG["device"],
        seeds_actually_run=[CONFIG["seed"]],
        steps_requested=CONFIG["iters"],
        steps_done=len(losses),
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
            detail="first reading: this IS the baseline, nothing to compare "
            "against yet. Exits 2 (nothing to compare): never 0, and not 3 -- "
            "it ran.",
        )
        led.status("partial")
        verdict, moved, detail = "baseline", [], "first reading"
    else:
        base = json.loads(BASELINE.read_text())["losses"]
        verdict, moved, detail = compare(base, losses)
        led.note("baseline_losses", base, how="runs/canary/baseline.json")
        led.note(
            "beats_outside_tol", moved, how="elementwise |now-base|/|base| vs REL_TOL"
        )
        led.note(
            "beat_counts",
            {"baseline": len(base), "now": len(losses)},
            how="len of the two beat sequences",
        )
        led.verdict(
            falsifier="the environment has not moved since the canary baseline",
            outcome="falsified" if verdict == "MOVED" else "survived",
            detail=detail,
        )
        led.status("ok")

    # 🔴 The row is written here, after the verdict, so its exit code is the one
    # this process returns -- READ from the verdict, not typed. It used to be written
    # above the verdict as the literal `exit_code=0`, so a MOVED canary exiting 1
    # filed a ledger saying 0. A hardcoded 0 is worse than `null`: it looks measured
    # (scripts/ledger.py made `exit_code` required because `null` hid this).
    code = exit_code_for(verdict)
    led.command(
        f".venv/bin/python scripts/canary.py {cycle}",
        exit_code=int(code),
        note="frozen config, seed 0; see CONFIG in scripts/canary.py. The exit code "
        "is the one run() returns for this verdict, recorded before sys.exit().",
    )
    path = led.write()
    return {
        "verdict": verdict,
        "moved": moved,
        "losses": losses,
        "detail": detail,
        "ledger": str(path),
        "exit_code": code,
    }


if __name__ == "__main__":
    cycle = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    r = run(cycle)
    print(json.dumps({k: v for k, v in r.items() if k != "losses"}, indent=2))
    print("losses:", [round(x, 6) for x in r["losses"]])
    run_main(lambda: r["exit_code"])
