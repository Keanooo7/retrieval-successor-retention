"""Run ONE claude session recorded in `.orchestrator/cycles/<n>.json`, and record
how it ended in `<n>.result.json`.

    PYTHONPATH=scripts .venv/bin/python -m orchestrator.session <n>

The launcher (`dispatch.py`, `tick.py`) writes the cycle record -- argv, cwd,
kind, item -- and starts this module, detached for a researcher and in the
foreground for a manager cycle. This module is the only writer of the result:

    {rc, end, cause, cost_usd, is_error, subtype, launched, result_head}

🔴 `rc` is the child's own `returncode`, normalised so a SIGTERM reads `143`. It
is never read through a pipe or a shell (CLAUDE.md). If THIS process is killed,
no result is written, and `reconcile.py` reads "pid dead, no result" as a session
that died unreported -- a failure, never a pass.

Spend: `total_cost_usd` from claude's `--output-format json` is appended to
`.orchestrator/state/spend.jsonl`. **An unparseable or missing cost is charged as
the whole per-cycle cap** (`RSR_CYCLE_CAP`): a spend cap that a crash can blind
is not a cap.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from orchestrator import loopcore as lc
from rsr.exit_codes import ArgumentParser, Exit, run_main

#: A researcher that handed its compute to `submit.py` ends its report with this.
LAUNCHED = re.compile(r"^\s*status:\s*LAUNCHED\b", re.MULTILINE)


def parse_output(text: str) -> dict:
    """Pull cost / error fields out of claude's JSON output. Tolerates a stream of
    JSON lines (the last object with a `total_cost_usd` wins) and garbage."""
    candidates = []
    try:
        candidates.append(json.loads(text))
    except (json.JSONDecodeError, ValueError):
        for line in text.splitlines():
            try:
                candidates.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                continue
    obj: dict = {}
    for c in candidates:
        if isinstance(c, dict) and ("total_cost_usd" in c or "result" in c):
            obj = c
    cost = obj.get("total_cost_usd")
    result = obj.get("result")
    return {
        "cost_usd": float(cost) if isinstance(cost, int | float) else None,
        "is_error": bool(obj.get("is_error", False)),
        "subtype": obj.get("subtype"),
        "result_text": result if isinstance(result, str) else "",
    }


def cause_of(rc: int | None, parsed: dict) -> str | None:
    """The failure's cause, or None for a clean exit. Two equal causes park an item
    (stop rule "two failures with the same cause")."""
    if rc == 0 and not parsed.get("is_error"):
        return None
    cause = f"rc={rc}"
    if parsed.get("subtype") and parsed.get("subtype") != "success":
        cause += f":{parsed['subtype']}"
    elif parsed.get("is_error"):
        cause += ":is_error"
    return cause


def execute(root: Path, n: int) -> int:
    """Run cycle `n` to completion and write its result. Returns the child's rc."""
    rec = lc.load_json(lc.cycle_path(root, n), None)
    if rec is None:
        raise FileNotFoundError(f"no cycle record {n}")
    cycles_dir = lc.sub(root, "cycles")
    out_path = cycles_dir / f"{n}.out.json"
    err_path = cycles_dir / f"{n}.err.log"
    cfg = lc.load_config(root)
    env = lc.child_env(root, **{k: str(v) for k, v in rec.get("env", {}).items()})
    with out_path.open("w") as out, err_path.open("w") as err:
        try:
            proc = subprocess.run(
                rec["argv"],
                cwd=rec["cwd"],
                stdout=out,
                stderr=err,
                stdin=subprocess.DEVNULL,
                env=env,
                check=False,
            )
            rc = lc.norm_rc(proc.returncode)
        except OSError as e:  # the binary is missing: did not run
            err.write(f"DID NOT RUN: {e}\n")
            rc = int(Exit.DID_NOT_RUN)
    parsed = parse_output(out_path.read_text())
    cost = parsed["cost_usd"]
    charged = cost
    if charged is None:
        charged = lc.cap(cfg, "RSR_CYCLE_CAP") or 0.0
    result = {
        "n": n,
        "rc": rc,
        "end": lc.now().isoformat(timespec="seconds"),
        "cause": cause_of(rc, parsed),
        "route": lc.route(rc),
        "cost_usd": cost,
        "cost_charged_usd": charged,
        "is_error": parsed["is_error"],
        "subtype": parsed["subtype"],
        "launched": bool(LAUNCHED.search(parsed["result_text"])),
        "result_head": parsed["result_text"][:4000],
        "session_pid": os.getpid(),
    }
    lc.append_jsonl(
        lc.spend_path(root),
        {
            "night": rec.get("night"),
            "cycle": n,
            "kind": rec.get("kind"),
            "item": rec.get("item"),
            "cost_usd": charged,
            "cost_reported": cost is not None,
            "rc": rc,
        },
    )
    lc.write_json(lc.result_path(root, n), result)
    return rc


def launch_detached(root: Path, n: int) -> int:
    """Start `python -m orchestrator.session <n>` in its own session (survives the
    tick) and stamp its pid into the cycle record. Returns the pid."""
    log = lc.sub(root, "logs") / f"session-{n}.log"
    with log.open("a") as f:
        proc = subprocess.Popen(
            [sys.executable, "-m", "orchestrator.session", str(n)],
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=lc.child_env(root),
            start_new_session=True,
        )
    path = lc.cycle_path(root, n)
    rec = lc.load_json(path, {})
    rec["pid"] = proc.pid
    lc.write_json(path, rec)
    return proc.pid


def main() -> Exit:
    ap = ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("n", type=int, help="cycle number under .orchestrator/cycles/")
    args = ap.parse_args()
    root = lc.root()
    try:
        rc = execute(root, args.n)
    except FileNotFoundError as e:
        print(f"DID NOT RUN: {e}", file=sys.stderr)
        return Exit.DID_NOT_RUN
    return Exit.OK if rc == 0 else Exit.FAIL


if __name__ == "__main__":
    run_main(main)
