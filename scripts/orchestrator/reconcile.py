"""Reconcile recorded state with the processes that actually exist.

    PYTHONPATH=scripts .venv/bin/python -m orchestrator.reconcile [--dry-run]

Deterministic, idempotent, and **it never touches a worktree** -- a crashed run's
tree is evidence, and deleting it is the owner's call.

Jobs (`.orchestrator/jobs/<job_id>.json`, written by `orchestrator.slot`):

* `status: running` and the pid is dead -> the record is rewritten `crashed`
  (with `crashed_by: reconcile`) and the item goes to `collecting`;
* `done` / `crashed` and not yet collected -> the item goes to `collecting`, once;
* a `submit.py` launch whose slot process died before writing any job record ->
  a `crashed` record is written for it, then as above.

Sessions (`.orchestrator/cycles/<n>.json` + `<n>.result.json`, see session.py):

* exited 0 (and not `is_error`) -> a researcher or collector's item goes to
  `verifying`; a researcher that reported `status: LAUNCHED` stays `running`
  (its job carries it);
* exited != 0 -- including 143, a SIGTERM -- or died with no result -> the item
  goes back to `ready` with `--attempt` (a collector's back to `collecting`).
  **Two failures with the same cause -> `parked`** (stop rule, dispatch
  2026-09-21 overnight "Two failures with the same cause"). `RSR_MAX_ATTEMPTS`
  (default 3) failures of any causes also park.
* a manager cycle failing twice with the same cause writes `.orchestrator/HALT`.

Exit: 0 reconciled · 3 a `workqueue set-status` call failed (state left as-is for
the next pass).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from orchestrator import loopcore as lc
from rsr.exit_codes import ArgumentParser, Exit, run_main, status


class Reconciler:
    def __init__(self, root: Path, *, dry_run: bool = False):
        self.root = root
        self.dry = dry_run
        self.cfg = lc.load_config(root)
        self.p = lc.load_pipeline(root)
        self.actions: list[str] = []
        self.errors: list[str] = []
        self.max_attempts = int(self.cfg.get("RSR_MAX_ATTEMPTS", "3"))

    # -- queue ------------------------------------------------------------- #

    def set_status(self, item: str, new: str, *, attempt: bool = False) -> None:
        args = ["set-status", item, new] + (["--attempt"] if attempt else [])
        self.actions.append(f"workqueue {' '.join(args)}")
        if self.dry:
            return
        proc = lc.run_orch(self.root, "workqueue", *args)
        if proc.returncode != 0:
            err = proc.stderr.strip()
            self.errors.append(
                f"workqueue {' '.join(args)} -> rc={proc.returncode} {err}"
            )

    def park(self, item: str, reason: str) -> None:
        self.p["parked"][item] = reason
        if item in self.p["awaiting"]:
            self.p["awaiting"].remove(item)
        self.set_status(item, "parked")

    # -- jobs -------------------------------------------------------------- #

    def jobs(self) -> None:
        jobs_dir = lc.sub(self.root, "jobs")
        recorded = {p.stem for p in jobs_dir.glob("*.json")}
        for sub in lc.read_jsonl(lc.sub(self.root, "state") / "submitted.jsonl"):
            jid = sub.get("job_id")
            if jid and jid not in recorded and not lc.pid_alive(sub.get("pid")):
                rec = {
                    "job_id": jid,
                    "item_id": sub.get("item"),
                    "lane": sub.get("lane"),
                    "slots": sub.get("slots"),
                    "cmd": sub.get("cmd"),
                    "cwd": sub.get("cwd"),
                    "pid": sub.get("pid"),
                    "pinned_sha": sub.get("pinned_sha"),
                    "start": sub.get("start"),
                    "end": lc.now().isoformat(timespec="seconds"),
                    "rc": None,
                    "status": "crashed",
                    "crashed_by": "reconcile: slot died before writing a job record",
                }
                self.actions.append(f"job {jid}: no record and pid dead -> crashed")
                if not self.dry:
                    lc.write_json(jobs_dir / f"{jid}.json", rec)
                recorded.add(jid)
        for path in sorted(jobs_dir.glob("*.json")):
            rec = lc.load_json(path, {})
            jid = rec.get("job_id", path.stem)
            item = rec.get("item_id")
            st = rec.get("status")
            if st == "running":
                if lc.pid_alive(rec.get("pid")):
                    continue
                rec["status"] = "crashed"
                rec["end"] = rec.get("end") or lc.now().isoformat(timespec="seconds")
                rec["crashed_by"] = "reconcile: status running, pid dead"
                self.actions.append(f"job {jid}: running with dead pid -> crashed")
                if not self.dry:
                    lc.write_json(path, rec)
                st = "crashed"
            if st in ("done", "crashed") and item:
                if jid in self.p["collected"] or jid in self.p["collecting"]:
                    continue
                self.p["collecting"][jid] = item
                self.p["manager_due"] = True
                self.set_status(item, "collecting")

    # -- sessions ---------------------------------------------------------- #

    def sessions(self) -> None:
        for n, rec, res in lc.cycles(self.root):
            if n in self.p["reconciled"]:
                continue
            if res is None:
                if lc.pid_alive(rec.get("pid")):
                    continue  # still running
                res = {"rc": None, "cause": "session died without a result (pid dead)"}
            kind = rec.get("kind")
            item = rec.get("item")
            cause = res.get("cause")
            rc = res.get("rc")
            self.actions.append(f"cycle {n} ({kind}, {item}): rc={rc} -> {lc.route(rc)}")
            if kind == "manager":
                self.manager(cause)
            elif item:
                if cause is None:
                    self.success(kind, item, rec, res)
                else:
                    self.failure(kind, item, rec, cause)
            self.p["reconciled"].append(n)
            self.p["manager_due"] = True

    def manager(self, cause: str | None) -> None:
        hist = self.p["failures"].setdefault("_manager", [])
        if cause is None:
            hist.clear()
            return
        if hist and hist[-1] == cause:
            reason = f"two consecutive manager cycles failed with the same cause: {cause}"
            self.actions.append(f"HALT: {reason}")
            if not self.dry:
                lc.halt(self.root, reason)
        hist.append(cause)

    def success(self, kind: str, item: str, rec: dict, res: dict) -> None:
        if kind == "collector":
            job = rec.get("job")
            if job:
                self.p["collecting"].pop(job, None)
                if job not in self.p["collected"]:
                    self.p["collected"].append(job)
        elif res.get("launched"):
            self.actions.append(f"{item}: researcher LAUNCHED a job; stays running")
            return
        if item not in self.p["awaiting"]:
            self.p["awaiting"].append(item)
        self.set_status(item, "verifying")

    def failure(self, kind: str, item: str, rec: dict, cause: str) -> None:
        hist = self.p["failures"].setdefault(item, [])
        repeat = cause in hist
        hist.append(cause)
        if repeat:
            self.park(item, f"two failures with the same cause ({cause})")
        elif len(hist) >= self.max_attempts:
            self.park(item, f"{len(hist)} failed attempts: {hist}")
        elif kind == "collector":
            self.set_status(item, "collecting", attempt=True)
        else:
            self.set_status(item, "ready", attempt=True)

    def run(self) -> Exit:
        self.jobs()
        self.sessions()
        # A failed set-status leaves the pipeline unsaved, so the next pass re-derives
        # the same transitions rather than recording one the queue never took.
        if not self.dry and not self.errors:
            lc.save_pipeline(self.root, self.p)
        for a in self.actions:
            lc.log(("DRY_RUN would: " if self.dry else "reconcile: ") + a)
        for e in self.errors:
            print(f"reconcile: {e}", file=sys.stderr)
        return Exit.DID_NOT_RUN if self.errors else Exit.OK


def reconcile(root: Path, *, dry_run: bool = False) -> Exit:
    return Reconciler(root, dry_run=dry_run).run()


def main() -> Exit:
    ap = ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dry = args.dry_run or os.environ.get("DRY_RUN") == "1"
    return status(reconcile(lc.root(), dry_run=dry))


if __name__ == "__main__":
    run_main(main)
