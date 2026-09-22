"""Hand a long computation to a lane, detached, so the researcher can exit.

    PYTHONPATH=scripts .venv/bin/python -m orchestrator.submit \\
        --item I --lane L --slots K [--peak-gb G] [--job-id J] -- <cmd...>

Run by a researcher from inside its worktree. Refuses (exit 3) unless:

* the tree's HEAD resolves -- it is the **pinned sha** the job runs at, stamped from
  git and never typed;
* the tree is **clean**, untracked files included (rsr-manager.md,
  Reproducibility: "the code that ran is committed");
* `uv sync --frozen` succeeded in that tree (`RSR_UV_BIN` overrides `uv`).

Then it starts `orchestrator.slot run --lane L --slots K --job-id J --item I
--pinned-sha S [--peak-gb G] -- <cmd>` in a new session (nohup-equivalent: it
survives the researcher), appends the launch to
`.orchestrator/state/submitted.jsonl`, and prints the job id on stdout. The
researcher then ends its report with `status: LAUNCHED`; when the job's record
reads `done`/`crashed`, the tick spawns a fresh collector with the same brief and
the job record (dispatch.py --collect).

Exit: 0 launched (job id on stdout) · 3 a precondition.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from orchestrator import loopcore as lc
from rsr.exit_codes import ArgumentParser, Exit, run_main, status


def submit(
    root: Path,
    cwd: Path,
    *,
    item: str,
    lane: str,
    slots: int,
    cmd: list[str],
    peak_gb: str | None = None,
    job_id: str | None = None,
    dry_run: bool = False,
) -> tuple[Exit, str]:
    if not cmd:
        return Exit.DID_NOT_RUN, "no command after `--`"
    head = lc.git(cwd, "rev-parse", "--verify", "HEAD^{commit}")
    if head.returncode != 0:
        return Exit.DID_NOT_RUN, f"{cwd} has no HEAD to pin"
    sha = head.stdout.strip()
    dirty = lc.git(cwd, "status", "--porcelain", "--untracked-files=all")
    if dirty.returncode != 0:
        return Exit.DID_NOT_RUN, f"git status failed in {cwd}"
    if dirty.stdout.strip():
        return (
            Exit.DID_NOT_RUN,
            f"the tree is not clean -- commit first; the code that runs must be "
            f"committed:\n{dirty.stdout}",
        )
    uv = os.environ.get("RSR_UV_BIN") or "uv"
    try:
        sync = subprocess.run(
            [uv, "sync", "--frozen", "--extra", "dev"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        sync_rc = sync.returncode
    except FileNotFoundError:
        sync_rc = None
    if sync_rc != 0:
        return Exit.DID_NOT_RUN, f"`uv sync --frozen` did not succeed (rc={sync_rc})"
    job_id = job_id or f"{item}-{lc.now().strftime('%Y%m%dT%H%M%S')}"
    argv = [
        *lc.orch_cmd("slot"),
        "run", "--lane", lane, "--slots", str(slots), "--job-id", job_id,
        "--item", item, "--pinned-sha", sha,
    ]  # fmt: skip
    if peak_gb is not None:
        argv += ["--peak-gb", str(peak_gb)]
    argv += ["--", *cmd]
    if dry_run:
        return Exit.OK, f"DRY_RUN: would launch {argv}"
    log = lc.sub(root, "logs") / f"job-{job_id}.log"
    with log.open("a") as f:
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=lc.child_env(root),
            start_new_session=True,
        )
    lc.append_jsonl(
        lc.sub(root, "state") / "submitted.jsonl",
        {
            "job_id": job_id,
            "item": item,
            "lane": lane,
            "slots": slots,
            "pid": proc.pid,
            "cwd": str(cwd),
            "pinned_sha": sha,
            "cmd": cmd,
            "start": lc.now().isoformat(timespec="seconds"),
        },
    )
    return Exit.OK, job_id


def main() -> Exit:
    argv = sys.argv[1:]
    if "--" in argv:
        cut = argv.index("--")
        argv, cmd = argv[:cut], argv[cut + 1 :]
    else:
        cmd = []
    ap = ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--item", required=True)
    ap.add_argument("--lane", required=True)
    ap.add_argument("--slots", type=int, required=True)
    ap.add_argument("--peak-gb", default=None)
    ap.add_argument("--job-id", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    code, msg = submit(
        lc.root(),
        Path.cwd(),
        item=args.item,
        lane=args.lane,
        slots=args.slots,
        cmd=cmd,
        peak_gb=args.peak_gb,
        job_id=args.job_id,
        dry_run=args.dry_run or os.environ.get("DRY_RUN") == "1",
    )
    print(msg, file=sys.stdout if code == Exit.OK else sys.stderr)
    return status(code)


if __name__ == "__main__":
    run_main(main)
