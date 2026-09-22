"""Open, close and report a night: `night/<date>` cut from a pinned `origin/main`.

    PYTHONPATH=scripts .venv/bin/python -m orchestrator.night open [--date D]
    PYTHONPATH=scripts .venv/bin/python -m orchestrator.night close
    PYTHONPATH=scripts .venv/bin/python -m orchestrator.night status

R-2026-09-22-night-branch: unattended work merges only into `night/<date>` at a
pinned base; **`main` never moves during a night**; the owner merges once each
morning. This module creates branches and a worktree; it never checks out, resets,
merges into or pushes `main`, and `close` writes the morning digest -- it does not
merge anything anywhere.

`open`:
  1. `git fetch origin`;
  2. pin `base_sha = origin/main` (read from git, never typed);
  3. `git branch night/<date> <base_sha>`;
  4. `.worktrees/_mgr` checked out on the night branch (the manager's checkout);
  5. write `.orchestrator/state/night.json`.

`close`: write the morning digest (`.orchestrator/state/digest-<date>.md` and,
committed on the night branch, `docs/lab-notes/morning-<date>.md` if absent), and
stamp `closed` in `night.json`.

Exit: 0 done · 2 `status` with no night recorded · 3 refused (fetch failed, a
dirty manager worktree, a night branch at a different base).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from orchestrator import loopcore as lc
from rsr.exit_codes import ArgumentParser, Exit, run_main, status

MGR = "_mgr"


def mgr_worktree(root: Path) -> Path:
    return root / ".worktrees" / MGR


def open_night(root: Path, date: str, window: lc.Window) -> tuple[Exit, str]:
    night = lc.read_night(root)
    if night and night.get("date") == date and not night.get("closed"):
        return Exit.OK, f"night {date} already open at {night['base_sha']}"
    fetch = lc.git(root, "fetch", "origin")
    if fetch.returncode != 0:
        return Exit.DID_NOT_RUN, f"git fetch origin failed (rc={fetch.returncode})"
    base = lc.git(root, "rev-parse", "--verify", "origin/main^{commit}")
    if base.returncode != 0:
        return Exit.DID_NOT_RUN, "origin/main does not resolve"
    base_sha = base.stdout.strip()
    branch = f"night/{date}"
    have = lc.git(root, "rev-parse", "--verify", f"refs/heads/{branch}")
    if have.returncode == 0:
        at = have.stdout.strip()
        anc = lc.git(root, "merge-base", "--is-ancestor", base_sha, at)
        if anc.returncode != 0:
            return (
                Exit.DID_NOT_RUN,
                f"{branch} exists at {at[:12]}, which does not descend from "
                f"origin/main {base_sha[:12]} -- refusing to adopt it",
            )
    else:
        lc.git(root, "branch", branch, base_sha, check=True)
    wt = mgr_worktree(root)
    if not wt.exists():
        wt.parent.mkdir(parents=True, exist_ok=True)
        lc.git(root, "worktree", "add", str(wt), branch, check=True)
    else:
        dirty = lc.git(wt, "status", "--porcelain")
        if dirty.returncode != 0 or dirty.stdout.strip():
            return Exit.DID_NOT_RUN, f"{wt} is dirty; the owner must clean it first"
        lc.git(wt, "checkout", branch, check=True)
    lc.write_json(
        lc.night_path(root),
        {
            "date": date,
            "base_sha": base_sha,
            "branch": branch,
            "window": {
                "last_dispatch": window.last_dispatch,
                "park_by": window.park_by,
                "digest_by": window.digest_by,
                "opens": window.opens,
            },
            "opened_at": lc.now().isoformat(timespec="seconds"),
            "closed": None,
        },
    )
    return Exit.OK, f"opened {branch} at {base_sha}"


# --------------------------------------------------------------------------- #
# the digest
# --------------------------------------------------------------------------- #


def build_digest(root: Path) -> str:
    """The morning status. **Stable for stable state** -- no wall-clock stamp in
    the body -- so notify's content-hash dedupe posts an unchanged digest once."""
    night = lc.read_night(root) or {}
    date = night.get("date", "?")
    p = lc.load_pipeline(root)
    cfg = lc.load_config(root)
    out = [f"# Morning digest -- night {date}", ""]
    out += [
        f"- base: `{night.get('base_sha', '?')}`",
        f"- branch: `{night.get('branch', '?')}` (the owner merges it; nothing "
        f"unattended touches `main`)",
    ]
    h = lc.halted(root)
    out.append(f"- HALT: {h}" if h else "- HALT: none")
    spent = lc.night_spend(root, date)
    out.append(
        f"- spend: ${spent:.2f} of night cap {cfg.get('RSR_NIGHT_CAP', 'UNSET')} "
        f"(per-cycle cap {cfg.get('RSR_CYCLE_CAP', 'UNSET')})"
    )
    out += ["", "## Items", ""]
    out.append(f"- merged into the night branch: {sorted(p['merged']) or 'none'}")
    out.append(f"- awaiting verification: {sorted(p['awaiting']) or 'none'}")
    out.append(f"- collecting (job finished, not collected): {p['collecting'] or 'none'}")
    if p["parked"]:
        out.append("- parked:")
        out += [f"  - `{k}`: {v}" for k, v in sorted(p["parked"].items())]
    else:
        out.append("- parked: none")
    if p["failures"]:
        out.append("- failures by cause:")
        out += [f"  - `{k}`: {v}" for k, v in sorted(p["failures"].items())]
    out += ["", "## Work queue (`orchestrator.workqueue digest`)", ""]
    wq = lc.run_orch(root, "workqueue", "digest")
    if wq.returncode == 0:
        out.append(wq.stdout.rstrip() or "(empty)")
    else:
        out.append(f"(workqueue digest did not run: rc={wq.returncode})")
    lanes = lc.run_orch(root, "lanes", "status", "--json")
    out += ["", "## Lanes", ""]
    out.append(
        f"```\n{lanes.stdout.strip()}\n```"
        if lanes.returncode == 0
        else f"(lanes status did not run: rc={lanes.returncode})"
    )
    owner = mgr_worktree(root) / "docs" / "lab-notes" / f"for-brendan-{date}.md"
    out += [
        "",
        "## For the owner (decisions batched here, per R-2026-09-22-owner-out-of-loop)",
        "",
    ]
    out.append(owner.read_text().rstrip() if owner.exists() else "none escalated")
    return "\n".join(out) + "\n"


def close_night(root: Path) -> tuple[Exit, str]:
    night = lc.read_night(root)
    if not night:
        return Exit.UNKNOWN, "no night recorded"
    date = night["date"]
    digest = build_digest(root)
    (lc.sub(root, "state") / f"digest-{date}.md").write_text(digest)
    wt = mgr_worktree(root)
    rel = Path("docs") / "lab-notes" / f"morning-{date}.md"
    if wt.exists() and not (wt / rel).exists():
        (wt / rel).parent.mkdir(parents=True, exist_ok=True)
        (wt / rel).write_text(
            digest + "\n<!-- skeleton written by orchestrator.night close; the "
            "manager's morning document replaces it -->\n"
        )
        lc.git(wt, "add", "--", str(rel), check=True)
        commit = lc.git(
            wt, "commit", "-m", f"night({date}): morning digest skeleton", "--", str(rel)
        )
        if commit.returncode != 0:
            return Exit.FAIL, f"could not commit the digest skeleton: {commit.stderr}"
    night["closed"] = lc.now().isoformat(timespec="seconds")
    lc.write_json(lc.night_path(root), night)
    return Exit.OK, f"closed night {date}"


def main() -> Exit:
    ap = ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("action", choices=["open", "close", "status"])
    ap.add_argument("--date", default=None, help="night date (default: from the clock)")
    args = ap.parse_args()
    root = lc.root()
    cfg = lc.load_config(root)
    window = lc.Window.from_config(cfg)
    if args.action == "status":
        night = lc.read_night(root)
        if not night:
            print("no night recorded")
            return Exit.UNKNOWN
        print(build_digest(root))
        return Exit.OK
    if os.environ.get("DRY_RUN") == "1":
        print(f"DRY_RUN: would {args.action} a night")
        return Exit.OK
    if args.action == "open":
        code, msg = open_night(root, args.date or window.night_date(lc.now()), window)
    else:
        code, msg = close_night(root)
    print(msg, file=sys.stdout if code == Exit.OK else sys.stderr)
    return status(code)


if __name__ == "__main__":
    run_main(main)
