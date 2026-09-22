"""Merge `run/<item_id>` into `night/<date>` -- and nowhere else -- behind two gates.

    PYTHONPATH=scripts .venv/bin/python -m orchestrator.merge <item_id>

R-2026-09-22-night-branch: unattended work merges only into the night branch, with
a merge commit, in the manager's worktree (`.worktrees/_mgr`). **This module never
names `main`.** The owner merges the night branch once each morning.

Gate 1 -- **verification**: `runs/<item_id>/verification.json` with
`status: ok`, committed on the night branch (where the manager's cycle commits it)
or on the run branch. Missing or `inconclusive` -> refused (3); `failed` -> 1.

Gate 2 -- **the frozen-diff guard**, over `git diff <base_sha>..run/<item_id>`.
Trips on any change to:

* `preregistration/**` -- signed, not re-openable;
* an **existing** `runs/<id>/` directory (one present at `base_sha`) -- new run
  directories are allowed; an old record is never edited;
* `runs/canary/baseline*` -- the canary's reference;
* `docs/owner/**` and `docs/spec/**`;
* a **FROZEN** definition in `src/rsr/constants.py` -- compared definition by
  definition (AST source segments), so a changed line inside a FROZEN entry, a
  FROZEN entry removed, one reclassified, or a new one added all trip it; an
  unparseable file trips it too (fail closed).

A trip is the most serious rejection there is (rsr-manager.md, "Frozen things":
*"it stops the night"*): `.orchestrator/HALT` is written with the reason, the
owner is notified, and the exit is 1.

Exit: 0 merged · 1 guard trip, failed verification, or a merge conflict (aborted)
· 3 a precondition (no night, no run branch, no verification, a dirty manager
worktree, a run branch not descended from the base).
"""

from __future__ import annotations

import ast
import fnmatch
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from orchestrator import loopcore as lc
from orchestrator import notify
from orchestrator.night import mgr_worktree
from rsr.exit_codes import ArgumentParser, Exit, run_main, status

CONSTANTS = "src/rsr/constants.py"

#: Paths no unattended branch may change at all.
FROZEN_GLOBS = (
    "preregistration/*",
    "docs/owner/*",
    "docs/spec/*",
    "runs/canary/baseline*",
)


def frozen_definitions(source: str | None) -> dict[str, str] | None:
    """{name: source segment} of every `Definition(..., klass=Klass.FROZEN, ...)`.
    None when the source does not parse -- the caller treats that as a trip."""
    if source is None:
        return {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Definition"
        ):
            continue
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        klass = kw.get("klass")
        if not (isinstance(klass, ast.Attribute) and klass.attr == "FROZEN"):
            continue
        name = kw.get("name")
        key = name.value if isinstance(name, ast.Constant) else ast.unparse(node)[:60]
        out[str(key)] = ast.get_source_segment(source, node) or ast.unparse(node)
    return out


def _show(root: Path, rev: str, path: str) -> str | None:
    proc = lc.git(root, "show", f"{rev}:{path}")
    return proc.stdout if proc.returncode == 0 else None


def frozen_violations(root: Path, base: str, head: str) -> list[str]:
    diff = lc.git(root, "diff", "--name-status", "--no-renames", f"{base}..{head}")
    if diff.returncode != 0:
        return [f"git diff {base[:12]}..{head} failed -- cannot clear the guard"]
    base_runs = {
        line.split("/")[1]
        for line in lc.git(
            root, "ls-tree", "-r", "--name-only", base, "--", "runs"
        ).stdout.splitlines()
        if line.count("/") >= 2
    }
    out = []
    for line in diff.stdout.splitlines():
        if not line.strip():
            continue
        st, path = line.split("\t", 1)
        if any(fnmatch.fnmatch(path, g) for g in FROZEN_GLOBS):
            out.append(f"{st} {path}: a frozen path")
        elif path.startswith("runs/") and path.count("/") >= 2:
            run = path.split("/")[1]
            if run in base_runs:
                out.append(
                    f"{st} {path}: an existing run record (runs/{run}/ is at base)"
                )
        elif path == CONSTANTS:
            before = frozen_definitions(_show(root, base, CONSTANTS))
            after = frozen_definitions(_show(root, head, CONSTANTS))
            if before is None or after is None:
                out.append(f"{st} {path}: does not parse -- FROZEN entries unverifiable")
                continue
            for name in sorted(set(before) | set(after)):
                if name not in after:
                    out.append(f"{path}: FROZEN {name!r} removed or reclassified")
                elif name not in before:
                    out.append(f"{path}: FROZEN {name!r} added")
                elif before[name] != after[name]:
                    out.append(f"{path}: FROZEN {name!r} changed")
    return out


def read_verification(root: Path, item: str) -> tuple[dict | None, str]:
    """(record, where) from the night branch first, then the run branch."""
    night = lc.read_night(root) or {}
    rel = f"runs/{item}/verification.json"
    for rev in (night.get("branch"), f"run/{item}"):
        if not rev:
            continue
        text = _show(root, rev, rel)
        if text is None:
            continue
        try:
            return json.loads(text), f"{rev}:{rel}"
        except json.JSONDecodeError:
            return {"status": "unparseable"}, f"{rev}:{rel}"
    return None, rel


def trip(root: Path, item: str, found: list[str], *, dry_run: bool) -> tuple[Exit, str]:
    reason = f"frozen-diff guard tripped merging {item}: " + "; ".join(found)
    if not dry_run:
        lc.halt(root, reason)
        body = "\n".join(
            [f"The merge of `run/{item}` into the night branch was refused.", ""]
            + [f"- {v}" for v in found]
            + [
                "",
                "`.orchestrator/HALT` is written; the loop stays stopped until you",
                "remove it.",
            ]
        )
        notify.post(root, f"HALT: frozen-diff guard tripped ({item})", body)
    return Exit.FAIL, reason


def merge_item(root: Path, item: str, *, dry_run: bool = False) -> tuple[Exit, str]:
    night = lc.read_night(root)
    if not night or night.get("closed"):
        return Exit.DID_NOT_RUN, "no open night"
    base, branch = night["base_sha"], night["branch"]
    run = f"run/{item}"
    head = lc.git(root, "rev-parse", "--verify", f"refs/heads/{run}")
    if head.returncode != 0:
        return Exit.DID_NOT_RUN, f"no branch {run}"
    head_sha = head.stdout.strip()
    if lc.git(root, "merge-base", "--is-ancestor", base, head_sha).returncode != 0:
        return Exit.DID_NOT_RUN, f"{run} does not descend from base {base[:12]}"
    rec, where = read_verification(root, item)
    if rec is None:
        return Exit.DID_NOT_RUN, f"no verification record ({where}) -- not merged"
    vstatus = rec.get("status")
    if vstatus == "failed":
        return Exit.FAIL, f"verification failed ({where}) -- not merged"
    if vstatus != "ok":
        return Exit.DID_NOT_RUN, f"verification is {vstatus!r} ({where}) -- not merged"
    found = frozen_violations(root, base, head_sha)
    if found:
        return trip(root, item, found, dry_run=dry_run)
    if dry_run:
        return Exit.OK, f"DRY_RUN: would merge {run} ({head_sha[:12]}) into {branch}"
    wt = mgr_worktree(root)
    if not wt.exists():
        return Exit.DID_NOT_RUN, f"no manager worktree at {wt}"
    on = lc.git(wt, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if on != branch:
        return Exit.DID_NOT_RUN, f"{wt} is on {on!r}, not {branch}"
    if lc.git(wt, "status", "--porcelain", "--untracked-files=no").stdout.strip():
        return Exit.DID_NOT_RUN, f"{wt} has uncommitted changes"
    msg = f"merge(night {night['date']}): {run} -- verified ({where})"
    proc = lc.git(wt, "merge", "--no-ff", "--no-edit", "-m", msg, head_sha)
    if proc.returncode != 0:
        lc.git(wt, "merge", "--abort")
        return Exit.FAIL, f"merge of {run} conflicted and was aborted: {proc.stdout}"
    return Exit.OK, f"merged {run} ({head_sha[:12]}) into {branch}"


def main() -> Exit:
    ap = ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("item_id")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dry = args.dry_run or os.environ.get("DRY_RUN") == "1"
    code, msg = merge_item(lc.root(), args.item_id, dry_run=dry)
    print(msg, file=sys.stdout if code == Exit.OK else sys.stderr)
    return status(code)


if __name__ == "__main__":
    run_main(main)
