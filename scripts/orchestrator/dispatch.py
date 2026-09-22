"""Dispatch ONE ready item to a fresh researcher, in its own worktree at the base.

    PYTHONPATH=scripts .venv/bin/python -m orchestrator.dispatch <item_id>
    PYTHONPATH=scripts .venv/bin/python -m orchestrator.dispatch <item_id> \\
        --collect <job_id>

Preconditions, each a refusal (exit 3) when unmet:

* a night is open (`.orchestrator/state/night.json`) -- the base is its `base_sha`;
* the item is in `workqueue ready --json --base <base_sha>`;
* `orchestrator.lint_brief <brief> --base <base_sha>` exits 0 -- a brief with
  findings is not dispatched (rsr-manager.md: the brief is committed, and correct,
  before the run);
* the spend caps are set (`RSR_CYCLE_CAP`; an owner decision, ops/orchestrator.env);
* `.claude/settings.researcher.json` exists in the worktree.

Then: `.worktrees/<item_id>` on branch `run/<item_id>` at `base_sha` (reused, never
recreated, on a retry -- a failed attempt's tree is evidence), and the researcher is
launched **detached** through `orchestrator.session`, whose record lands in
`.orchestrator/cycles/<n>.json`:

    $RSR_CLAUDE_BIN -p "Execute <brief path>" --agent rsr-researcher
        --settings .claude/settings.researcher.json --permission-mode dontAsk
        --output-format json --max-budget-usd $RSR_CYCLE_CAP

The prompt is a verb and a path. **Never a paraphrase of the brief** (rsr-manager.md:
"a paraphrase is a second, divergent copy, and every copy rots"). `--collect`
spawns the fresh "collector" researcher for a finished job: the same brief, plus
the path of the job record.

Exit: 0 launched (or DRY_RUN) · 1 git refused the worktree · 3 a precondition.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from orchestrator import loopcore as lc
from orchestrator import session
from rsr.exit_codes import ArgumentParser, Exit, run_main, status

SETTINGS = ".claude/settings.researcher.json"


def ready_items(root: Path, base: str) -> tuple[list[dict] | None, str]:
    proc = lc.run_orch(root, "workqueue", "ready", "--json", "--base", base)
    if proc.returncode != 0:
        return None, f"workqueue ready rc={proc.returncode}: {proc.stderr.strip()}"
    try:
        doc = json.loads(proc.stdout or "")
    except json.JSONDecodeError as e:
        return None, f"workqueue ready printed non-JSON: {e}"
    # workqueue's documented shape: {"base_sha": ..., "ready": [row, ...]}, each row
    # carrying the computed `ready` flag. 🔴 The first integration parsed a bare list:
    # iterating the dict yielded its keys, all dropped as non-dicts, so every item read
    # "not in the ready set" and a full queue looked idle. An unexpected shape is now
    # an unreadable queue (None), never an empty one.
    if not isinstance(doc, dict) or not isinstance(doc.get("ready"), list):
        return None, f"workqueue ready printed an unexpected shape: {type(doc).__name__}"
    if doc.get("base_sha") not in (None, base) and base:
        return (
            None,
            f"workqueue ready answered for base {doc.get('base_sha')}, not {base}",
        )
    rows = [i for i in doc["ready"] if isinstance(i, dict) and i.get("ready") is True]
    return rows, ""


def ensure_worktree(root: Path, item: str, base: str) -> tuple[Exit, Path, str]:
    wt = root / ".worktrees" / item
    branch = f"run/{item}"
    if wt.exists():
        head = lc.git(wt, "rev-parse", "--abbrev-ref", "HEAD")
        if head.returncode != 0 or head.stdout.strip() != branch:
            return Exit.DID_NOT_RUN, wt, f"{wt} exists and is not on {branch}"
        return Exit.OK, wt, f"reusing {wt} ({branch})"
    have = lc.git(root, "rev-parse", "--verify", f"refs/heads/{branch}")
    wt.parent.mkdir(parents=True, exist_ok=True)
    if have.returncode == 0:
        anc = lc.git(root, "merge-base", "--is-ancestor", base, have.stdout.strip())
        if anc.returncode != 0:
            return Exit.DID_NOT_RUN, wt, f"{branch} exists and does not descend from base"
        proc = lc.git(root, "worktree", "add", str(wt), branch)
    else:
        proc = lc.git(root, "worktree", "add", "-b", branch, str(wt), base)
    if proc.returncode != 0:
        return Exit.FAIL, wt, f"git worktree add failed: {proc.stderr.strip()}"
    return Exit.OK, wt, f"created {wt} on {branch} at {base[:12]}"


def sync_venv(wt: Path) -> tuple[Exit, str]:
    """`uv sync --frozen --extra dev` in the job worktree (`RSR_UV_BIN` overrides)."""
    uv = os.environ.get("RSR_UV_BIN") or "uv"
    try:
        proc = subprocess.run(
            [uv, "sync", "--frozen", "--extra", "dev"],
            cwd=wt,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as e:
        return Exit.DID_NOT_RUN, f"uv not runnable in {wt}: {e}"
    if proc.returncode != 0:
        return (
            Exit.DID_NOT_RUN,
            f"`uv sync --frozen` in {wt} -> rc={proc.returncode}\n{proc.stderr.strip()}",
        )
    return Exit.OK, ""


def dispatch(
    root: Path, item: str, *, collect_job: str | None = None, dry_run: bool = False
) -> tuple[Exit, str]:
    night = lc.read_night(root)
    if not night or night.get("closed"):
        return Exit.DID_NOT_RUN, "no open night (orchestrator.night open)"
    base = night["base_sha"]
    cfg = lc.load_config(root)
    if lc.cap(cfg, "RSR_CYCLE_CAP") is None or lc.cap(cfg, "RSR_NIGHT_CAP") is None:
        return Exit.DID_NOT_RUN, "spend caps are UNSET (ops/orchestrator.env)"
    p = lc.load_pipeline(root)
    if collect_job:
        info = p["items"].get(item)
        if not info:
            return Exit.DID_NOT_RUN, f"{item} was never dispatched; nothing to collect"
        brief = info["brief"]
    else:
        items, err = ready_items(root, base)
        if items is None:
            return Exit.DID_NOT_RUN, err
        match = [i for i in items if i.get("id") == item]
        if not match:
            return Exit.DID_NOT_RUN, f"{item} is not in the ready set at {base[:12]}"
        info = match[0]
        brief = info.get("brief")
        if not brief:
            return Exit.DID_NOT_RUN, f"{item} has no brief path"
        lint = lc.run_orch(root, "lint_brief", brief, "--base", base)
        if lint.returncode != 0:
            return (
                Exit.DID_NOT_RUN,
                f"lint_brief {brief} --base {base[:12]} -> rc={lint.returncode}; "
                f"a brief with findings is not dispatched\n{lint.stdout}{lint.stderr}",
            )
    prompt = f"Execute {brief}"
    kind = "researcher"
    if collect_job:
        kind = "collector"
        job_rec = lc.sub(root, "jobs") / f"{collect_job}.json"
        prompt += f" -- you are the collector for job record {job_rec}"
    argv = lc.claude_argv(cfg, prompt, "rsr-researcher", SETTINGS)
    if dry_run:
        return Exit.OK, f"DRY_RUN: would launch {kind} for {item}: {argv}"
    code, wt, msg = ensure_worktree(root, item, base)
    if code != Exit.OK:
        return code, msg
    if not (wt / SETTINGS).exists():
        return Exit.DID_NOT_RUN, f"{wt / SETTINGS} is missing"
    # The hooks in SETTINGS run `$CLAUDE_PROJECT_DIR/.venv/bin/python` and fail
    # closed: a worktree without its own venv blocks every tool call.
    code, why = sync_venv(wt)
    if code != Exit.OK:
        return code, why
    n, _ = lc.new_cycle(
        root,
        {
            "kind": kind,
            "item": item,
            "job": collect_job,
            "argv": argv,
            "cwd": str(wt),
            "night": night["date"],
            "base_sha": base,
            "start": lc.now().isoformat(timespec="seconds"),
            "pid": None,
            # RSR_RUN_ID: hooks.py's stop/blinding checks key on it; the run id
            # is the item id (merge.py reads runs/<item>/verification.json).
            "env": {"RSR_ITEM_ID": item, "RSR_RUN_ID": item, "RSR_BASE_SHA": base},
        },
    )
    pid = session.launch_detached(root, n)
    p = lc.load_pipeline(root)
    p["items"].setdefault(item, {}).update(
        {
            "brief": brief,
            "lane": info.get("lane"),
            "slots": info.get("slots"),
            "worktree": str(wt),
            "branch": f"run/{item}",
        }
    )
    if collect_job:
        p["items"][item].setdefault("collector_cycles", {})[collect_job] = n
    lc.save_pipeline(root, p)
    if not collect_job:
        proc = lc.run_orch(root, "workqueue", "set-status", item, "running")
        if proc.returncode != 0:
            lc.log(f"dispatch: set-status {item} running rc={proc.returncode}")
    return Exit.OK, f"{msg}; launched {kind} cycle {n} (pid {pid}) for {item}"


def main() -> Exit:
    ap = ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("item_id")
    ap.add_argument("--collect", metavar="JOB_ID", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dry = args.dry_run or os.environ.get("DRY_RUN") == "1"
    code, msg = dispatch(lc.root(), args.item_id, collect_job=args.collect, dry_run=dry)
    print(msg, file=sys.stdout if code == Exit.OK else sys.stderr)
    return status(code)


if __name__ == "__main__":
    run_main(main)
