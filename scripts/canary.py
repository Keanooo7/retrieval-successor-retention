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

## The split, 2026-09-22: a CPU canary, and a code check in front of both

🔴 **The canary could not tell "the code moved" from "the machine moved".** At
`945b501` every one of its 6 beats read MOVED, and the same script at `7254080`, same
machine, same torch, reproduced `runs/canary/baseline.json` bit for bit
(`docs/lab-notes/for-brendan-2026-09-21.md`, `docs/lab-notes/canary-2026-09-21/`).
Code between the two shas changed the loss path. A canary that exits `1` for that is
reporting a code change as an environment move -- the one thing it exists not to do.

Two repairs:

1. **`loss_path_hash`.** sha256 over every source file the loss path actually
   imported (each `sys.modules` entry whose file is under `src/rsr/`, plus this
   script), sorted by repo-relative path. It is written into a baseline when the
   baseline is written. On a reading, **a hash that differs from the baseline's is
   `2` UNKNOWN, "not comparable: code changed" -- never `1`**, and the command that
   runs the baseline's own sha in a pinned worktree is printed: that is the
   environment check, and it is the only one a code change leaves available. A
   baseline written before the hash existed is also `2`, with a message -- it cannot
   say which code it measured. Existing baseline files are **not rewritten**;
   re-baselining is the owner's call.
2. **`--device cpu`**, bit-exact: `rel_tol = 0`, one thread, its own baseline at
   `runs/canary/baseline-cpu.json`. CPU at one thread is deterministic on this
   machine (E0b's argument, and the old-sha reading reproduced exactly), so any
   difference at all is a move. **The MPS configuration is frozen as it was** --
   same `CONFIG`, same `REL_TOL`, same `baseline.json`.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
#: Where the code that runs lives. Separate from `_REPO` (where runs are written),
#: which tests repoint at a tmp dir: the hash must always be over the code that ran.
_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "src"))

from ledger import Ledger  # noqa: E402

from rsr.exit_codes import ArgumentParser, Exit, run_main, status  # noqa: E402

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

# -- the CPU variant, committed 2026-09-22 before its first reading. ----------- #
# Identical to CONFIG except the device. Bit-exact: any difference is a move.
CONFIG_CPU = {**CONFIG, "device": "cpu"}
REL_TOL_CPU = 0.0
CPU_THREADS = 1
BASELINE_CPU = _REPO / "runs" / "canary" / "baseline-cpu.json"

DEVICES = ("mps", "cpu")


#: `0` held · `1` MOVED · `2` nothing to compare. See `rsr.exit_codes.Exit`.
#: 🔴 A first reading is `2`: it ran and had nothing to compare. Not `0`, not `3`.
#: 🔴 `not_comparable` is `2` as well: the code under the loss path changed since the
#: baseline, so there is nothing *comparable* to compare against. Never `1`.
EXIT_CODES = {
    "held": Exit.OK,
    "MOVED": Exit.FAIL,
    "baseline": Exit.UNKNOWN,
    "not_comparable": Exit.UNKNOWN,
}


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


def settings(device: str) -> tuple[dict, float, Path]:
    """`(config, rel_tol, baseline_path)` for one variant. Read at call time, so a
    test that repoints `BASELINE` / `BASELINE_CPU` is honoured."""
    if device == "mps":
        return CONFIG, REL_TOL, BASELINE
    if device == "cpu":
        return CONFIG_CPU, REL_TOL_CPU, BASELINE_CPU
    raise ValueError(f"unknown canary device {device!r}; one of {DEVICES}")


def compare(
    baseline: list[float], losses: list[float], rel_tol: float = REL_TOL
) -> tuple[str, list[dict], str]:
    """`(verdict, beats_outside_tol, detail)`.

    🔴 **Unequal lengths are a move.** Comparing over the overlap lets a run that
    produced 2 of 6 beats report "held": the beats it did produce match, and the
    four it never reached are simply not looked at. A canary that cannot see a
    truncated run is not an environment check.

    At `rel_tol == 0` (the CPU variant) the comparison is float equality: any
    non-zero relative difference is outside tolerance.
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
        if abs(losses[i] - baseline[i]) / max(abs(baseline[i]), 1e-12) > rel_tol
    ]
    if moved:
        return "MOVED", moved, f"{len(moved)} of {n} beats outside rel_tol={rel_tol}"
    return "held", [], f"all {n} beats within rel_tol={rel_tol}"


# --------------------------------------------------------------------------- #
# loss_path_hash -- is the code under the loss the code the baseline measured?
# --------------------------------------------------------------------------- #


def loss_path_files(modules: dict | None = None, repo: Path | None = None) -> list[str]:
    """Repo-relative paths of every imported module file under `src/rsr/`, plus
    this script, sorted. Read off `sys.modules`, i.e. what was *actually*
    imported -- not a hand-kept list, which would rot the first time `loop.py`
    grew an import."""
    repo = repo or _CODE
    modules = sys.modules if modules is None else modules
    src = (repo / "src" / "rsr").resolve()
    out = {"scripts/canary.py"}
    for mod in list(modules.values()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        p = Path(f).resolve()
        if p.suffix == ".py" and p.is_relative_to(src):
            out.add(p.relative_to(repo.resolve()).as_posix())
    return sorted(out)


def loss_path_hash(files: list[str], repo: Path | None = None) -> str:
    """sha256 over `(path, bytes)` of each file, in sorted path order. The path is
    hashed too, so moving a file is a change even when its bytes are not."""
    repo = repo or _CODE
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(rel.encode() + b"\0")
        h.update((repo / rel).read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _baseline_sha(base: dict, runs_root: Path) -> str | None:
    """The sha the baseline was measured at: recorded in it, or -- for a baseline
    written before that field existed -- read off the ledger of the cycle that
    wrote it. Derived, never typed."""
    if base.get("git_sha"):
        return base["git_sha"]
    cycle = base.get("cycle")
    if not isinstance(cycle, int):
        return None
    led = runs_root / "canary" / f"cycle-{cycle:02d}" / "ledger.json"
    try:
        return json.loads(led.read_text())["provenance"]["git_sha"] or None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def old_sha_command(sha: str | None, cycle: int, device: str) -> str:
    """The environment check a code change leaves available: the baseline's own
    sha, in a pinned worktree, against its own committed baseline. It must hold."""
    pin = sha or "<SHA-THAT-WROTE-THE-BASELINE>"
    wt = f"/tmp/rsr-canary-{pin[:12]}"
    return (
        f"/opt/homebrew/bin/git -C {_REPO} worktree add --detach {wt} {pin} && "
        f"cd {wt} && uv sync --frozen --extra dev && "
        f".venv/bin/python scripts/canary.py {cycle} --device {device}"
    )


def comparability(
    base: dict, now_hash: str, *, cycle: int, device: str, runs_root: Path
) -> tuple[bool, str]:
    """`(comparable, why)`. A baseline without a hash, or with a different one, is
    not comparable -- `2`, never `1`."""
    sha = _baseline_sha(base, runs_root)
    cmd = old_sha_command(sha, cycle, device)
    then = base.get("loss_path_hash")
    if then is None:
        return False, (
            "not comparable: the baseline records no loss_path_hash, so it cannot say "
            "which code it measured. Re-baselining is the owner's call. Environment "
            f"check -- run the baseline's own sha in a pinned worktree: {cmd}"
        )
    if then != now_hash:
        return False, (
            f"not comparable: code changed (loss_path_hash {then[:12]} in the "
            f"baseline, {now_hash[:12]} now). Environment check -- run the "
            f"baseline's own sha in a pinned worktree, which must hold: {cmd}"
        )
    return True, f"loss_path_hash {now_hash[:12]} matches the baseline"


def _losses(hb: Path) -> list[float]:
    out = []
    for line in hb.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("kind") == "beat":
            out.append(float(rec["loss"]))
    return out


def run(cycle: int, device: str = "mps") -> dict:
    import torch

    from rsr.train.loop import train

    config, rel_tol, baseline_path = settings(device)
    suffix = "" if device == "mps" else f"-{device}"
    out_dir = _REPO / "runs" / "canary" / f"cycle-{cycle:02d}{suffix}"

    prev_threads = torch.get_num_threads()
    if device == "cpu":
        torch.set_num_threads(CPU_THREADS)
    try:
        threads = torch.get_num_threads()
        train(out_dir=out_dir, **config)
    finally:
        torch.set_num_threads(prev_threads)
    losses = _losses(out_dir / "heartbeat.jsonl")

    # After train(): every lazy import on the loss path has happened by now.
    files = loss_path_files()
    now_hash = loss_path_hash(files)

    run_id = f"canary/cycle-{cycle:02d}{suffix}"
    led = Ledger(
        run_id,
        cycle=cycle,
        question="has the environment moved since the canary baseline?",
    )
    led.manifest(config)
    led.run_meta(
        device=config["device"],
        seeds_actually_run=[config["seed"]],
        steps_requested=config["iters"],
        steps_done=len(losses),
    )
    led.note("losses", losses, how="heartbeat.jsonl beat records, field 'loss'")
    led.note("config", config, how="frozen literal in scripts/canary.py")
    led.note(
        "rel_tol", rel_tol, how="committed in scripts/canary.py before the first reading"
    )
    led.note("torch_num_threads", threads, how="torch.get_num_threads() during train()")
    led.note(
        "loss_path_hash",
        now_hash,
        how="sha256 over sys.modules files under src/rsr + scripts/canary.py, sorted",
    )
    led.note("loss_path_files", files, how="canary.loss_path_files() after train()")

    try:
        base_rel = baseline_path.relative_to(_REPO).as_posix()
    except ValueError:
        base_rel = str(baseline_path)
    if not baseline_path.exists():
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(
            json.dumps(
                {
                    "cycle": cycle,
                    "device": config["device"],
                    "rel_tol": rel_tol,
                    "git_sha": led.doc["provenance"].get("git_sha"),
                    "losses": losses,
                    "loss_path_hash": now_hash,
                    "loss_path_files": files,
                },
                indent=2,
            )
            + "\n"
        )
        led.note(
            "role", "baseline", how=f"{base_rel} did not exist; this reading defines it"
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
        base_doc = json.loads(baseline_path.read_text())
        base = base_doc["losses"]
        led.note("baseline_losses", base, how=base_rel)
        ok, why = comparability(
            base_doc, now_hash, cycle=cycle, device=device, runs_root=_REPO / "runs"
        )
        if not ok:
            verdict, moved, detail = "not_comparable", [], why
            led.note(
                "baseline_loss_path_hash",
                base_doc.get("loss_path_hash"),
                how=f"{base_rel}, field loss_path_hash (null: predates the field)",
            )
            led.verdict(
                falsifier="the environment has not moved since the canary baseline",
                outcome="inconclusive",
                detail=why,
            )
            led.status("partial")
        else:
            verdict, moved, detail = compare(base, losses, rel_tol)
            led.note(
                "beats_outside_tol",
                moved,
                how="elementwise |now-base|/|base| vs rel_tol",
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
    dev_arg = "" if device == "mps" else f" --device {device}"
    led.command(
        f".venv/bin/python scripts/canary.py {cycle}{dev_arg}",
        exit_code=int(code),
        note="frozen config, seed 0; see CONFIG / CONFIG_CPU in scripts/canary.py. "
        "The exit code is the one run() returns for this verdict, recorded before "
        "sys.exit().",
    )
    path = led.write()
    return {
        "verdict": verdict,
        "moved": moved,
        "losses": losses,
        "detail": detail,
        "device": device,
        "loss_path_hash": now_hash,
        "ledger": str(path),
        "exit_code": code,
    }


def main(argv: list[str] | None = None) -> Exit:
    ap = ArgumentParser(description="the environment canary")
    ap.add_argument("cycle", nargs="?", type=int, default=0)
    ap.add_argument(
        "--device",
        choices=DEVICES,
        default="mps",
        help="mps: the frozen 1e-4 canary; cpu: bit-exact, 1 thread, own baseline",
    )
    args = ap.parse_args(argv)
    r = run(args.cycle, args.device)
    print(json.dumps({k: v for k, v in r.items() if k != "losses"}, indent=2))
    print("losses:", [round(x, 6) for x in r["losses"]])
    if r["verdict"] == "not_comparable":
        print(f"\nUNKNOWN (2): {r['detail']}", file=sys.stderr)
    return status(r["exit_code"])


if __name__ == "__main__":
    run_main(main)
