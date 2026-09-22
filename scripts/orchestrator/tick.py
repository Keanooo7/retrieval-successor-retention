"""ONE deterministic pass of the loop. Started every 600 s by launchd via tick.zsh.

    PYTHONPATH=scripts .venv/bin/python -m orchestrator.tick

In order, and each step can end the pass:

 1. `.orchestrator/HALT` present -> exit 0, touching nothing.
 2. Spend caps (`RSR_CYCLE_CAP`, `RSR_NIGHT_CAP`) `UNSET` or not positive -> notify
    once, exit **3**. The caps are an owner decision; the loop does not guess one.
 3. Take the tick lock (non-blocking flock); held by another tick -> exit 0.
 4. Clock (`RSR_NOW` pins it): in the window and no night open for tonight ->
    `night open`. Past `digest_by` -> close the open night (digest + notify), exit.
 5. `reconcile` (jobs and sessions against live pids).
 6. Night spend >= `RSR_NIGHT_CAP` -> HALT, notify, exit.
 7. Merge every awaiting item whose `verification.json` is `ok` (merge.py; a
    frozen-diff trip HALTs); park the `failed`/`inconclusive` ones.
 8. Past `park_by` -> park everything in flight, write + notify the digest, exit.
    Nothing is killed: a running job is left to finish and is reconciled later.
 9. Nothing ready, running, collecting or awaiting verification -> write the
    digest, notify ONCE (per distinct state), exit **without starting claude**.
10. Before `last_dispatch`: dispatch ready items up to `RSR_MAX_SESSIONS` live
    researchers. Until `park_by`: spawn a collector for each finished job.
11. One manager cycle, in the foreground, in `.worktrees/_mgr`, under
    `caffeinate -i -s` when it exists -- if one is due (a
    researcher or job changed state since the last one, or
    `RSR_MANAGER_INTERVAL_MIN` has passed -- a manager cycle ending is not itself a
    reason for another):

        $RSR_CLAUDE_BIN -p "Run one cycle per .orchestrator/cycle-protocol.md"
            --agent rsr-manager --settings .claude/settings.orchestrator.json
            --permission-mode dontAsk --output-format json
            --max-budget-usd $RSR_CYCLE_CAP

    Its `total_cost_usd` is appended to `.orchestrator/state/spend.jsonl`; the
    night total reaching the cap writes HALT.

`DRY_RUN=1`: every step logs what it would do; nothing is launched, no branch is
created, no status is changed, nothing is posted.

Exit: 0 a pass completed (including HALT, lock held, idle) · 1 a merge guard trip
or refused merge · 3 caps UNSET, or a night could not be opened.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from orchestrator import dispatch, merge, night, notify, reconcile, session
from orchestrator import loopcore as lc
from rsr.exit_codes import ArgumentParser, Exit, run_main, status

MANAGER_PROMPT = "Run one cycle per .orchestrator/cycle-protocol.md"
MANAGER_SETTINGS = ".claude/settings.orchestrator.json"


class Tick:
    def __init__(self, root: Path):
        self.root = root
        self.cfg = lc.load_config(root)
        self.dry = self.cfg.get("DRY_RUN") == "1"
        self.window = lc.Window.from_config(self.cfg)
        self.now = lc.now()

    def say(self, msg: str) -> None:
        lc.log(("DRY_RUN: " if self.dry else "") + msg)

    def notify(self, title: str, body: str) -> None:
        notify.post(self.root, title, body, dry_run=self.dry)

    # ------------------------------------------------------------------ #

    def run(self) -> Exit:
        h = lc.halted(self.root)
        if h:
            self.say(f"HALT present -- nothing done: {h.splitlines()[-1]}")
            return Exit.OK
        cycle_cap = lc.cap(self.cfg, "RSR_CYCLE_CAP")
        night_cap = lc.cap(self.cfg, "RSR_NIGHT_CAP")
        if cycle_cap is None or night_cap is None:
            msg = (
                "The orchestrator refuses to run: RSR_CYCLE_CAP="
                f"{self.cfg.get('RSR_CYCLE_CAP', 'UNSET')} RSR_NIGHT_CAP="
                f"{self.cfg.get('RSR_NIGHT_CAP', 'UNSET')}. The spend caps are an "
                "owner decision -- set both in ops/orchestrator.env."
            )
            print(f"DID NOT RUN: {msg}", file=sys.stderr)
            self.notify("Orchestrator refused: spend caps UNSET", msg)
            return Exit.DID_NOT_RUN
        self.night_cap = night_cap
        with lc.tick_lock(self.root) as held:
            if not held:
                self.say("another tick holds the lock -- nothing done")
                return Exit.OK
            return self.locked_pass()

    def locked_pass(self) -> Exit:
        phase = self.window.phase(self.now)
        date = self.window.night_date(self.now)
        n = lc.read_night(self.root)
        self.say(f"phase={phase} night={date}")
        if phase == "closed":
            if n and not n.get("closed"):
                return self.close(n)
            return Exit.OK
        if not n or n.get("date") != date or n.get("closed"):
            if n and n.get("date") == date and n.get("closed"):
                return Exit.OK  # tonight is already closed
            if self.dry:
                self.say(f"would open night/{date}")
                return Exit.OK
            code, msg = night.open_night(self.root, date, self.window)
            self.say(msg)
            if code != Exit.OK:
                self.notify(f"Night {date} could not open", msg)
                return Exit.DID_NOT_RUN
        self.date = date
        reconcile.reconcile(self.root, dry_run=self.dry)
        if lc.halted(self.root):
            self.notify("HALT", lc.halted(self.root) or "")
            return Exit.OK
        if self.over_budget():
            return Exit.OK
        code = self.merges()
        if code != Exit.OK:
            return code
        if phase == "parking":
            self.park_all()
            self.post_digest("parked")
            return Exit.OK
        ready = self.ready()
        if ready is None:
            return Exit.DID_NOT_RUN
        if not ready and not self.active():
            self.idle()
            return Exit.OK
        if phase == "open":
            self.dispatch_ready(ready)
        self.collectors()
        self.manager_cycle()
        return Exit.OK

    # ------------------------------------------------------------------ #

    def over_budget(self) -> bool:
        spent = lc.night_spend(self.root, self.date)
        if spent < self.night_cap:
            return False
        reason = f"night spend ${spent:.2f} reached RSR_NIGHT_CAP ${self.night_cap:.2f}"
        self.say(f"HALT: {reason}")
        if not self.dry:
            lc.halt(self.root, reason)
        self.notify("HALT: night spend cap reached", reason)
        return True

    def ready(self) -> list[dict] | None:
        """The ready set, or None when the queue cannot be read -- which is NOT an
        empty queue, and must not produce an "idle" digest."""
        n = lc.read_night(self.root) or {}
        items, err = dispatch.ready_items(self.root, n.get("base_sha", ""))
        if items is None:
            self.say(f"ready set unavailable: {err}")
            self.notify("Work queue unreadable", err)
            return None
        p = lc.load_pipeline(self.root)
        return [i for i in items if i.get("id") not in p["parked"]]

    def active(self) -> bool:
        p = lc.load_pipeline(self.root)
        running_jobs = [
            rec
            for path in lc.sub(self.root, "jobs").glob("*.json")
            if (rec := lc.load_json(path, {})).get("status") == "running"
        ]
        return bool(
            lc.live_sessions(self.root)
            or running_jobs
            or p["collecting"]
            or p["awaiting"]
        )

    def merges(self) -> Exit:
        p = lc.load_pipeline(self.root)
        for item in list(p["awaiting"]):
            rec, where = merge.read_verification(self.root, item)
            if rec is None:
                continue
            st = rec.get("status")
            if st == "ok":
                code, msg = merge.merge_item(self.root, item, dry_run=self.dry)
                self.say(msg)
                if self.dry:
                    continue
                p = lc.load_pipeline(self.root)
                if code == Exit.OK:
                    p["awaiting"].remove(item)
                    p["merged"].append(item)
                    lc.save_pipeline(self.root, p)
                    lc.run_orch(self.root, "workqueue", "set-status", item, "merged")
                    self.notify(f"Merged {item} into the night branch", msg)
                    continue
                if lc.halted(self.root):
                    return Exit.FAIL
                self.park(p, item, f"merge refused: {msg}")
            elif st in ("failed", "inconclusive", "unparseable"):
                self.say(f"{item}: verification {st} ({where}) -> parked")
                if not self.dry:
                    self.park(lc.load_pipeline(self.root), item, f"verification {st}")
        return Exit.OK

    def park(self, p: dict, item: str, reason: str) -> None:
        p["parked"][item] = reason
        if item in p["awaiting"]:
            p["awaiting"].remove(item)
        lc.save_pipeline(self.root, p)
        lc.run_orch(self.root, "workqueue", "set-status", item, "parked")

    def park_all(self) -> None:
        p = lc.load_pipeline(self.root)
        inflight = set(p["awaiting"]) | set(p["collecting"].values())
        inflight |= {
            rec.get("item") for rec in lc.live_sessions(self.root) if rec.get("item")
        }
        for item in sorted(inflight - set(p["parked"]) - set(p["merged"])):
            self.say(f"park_by passed: parking {item}")
            if not self.dry:
                self.park(lc.load_pipeline(self.root), item, "wall clock: park_by passed")

    def post_digest(self, why: str) -> None:
        body = night.build_digest(self.root)
        if not self.dry:
            (lc.sub(self.root, "state") / f"digest-{self.date}.md").write_text(body)
        self.notify(f"Digest, night {self.date} ({why})", body)

    def idle(self) -> None:
        p = lc.load_pipeline(self.root)
        sig = hashlib.sha256(
            json.dumps(
                [
                    self.date,
                    p["merged"],
                    p["parked"],
                    p["failures"],
                    lc.halted(self.root),
                ],
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if p.get("idle_signature") == sig:
            self.say("idle; digest already posted for this state")
            return
        self.say("idle: nothing ready, running, collecting or awaiting -- digest only")
        self.post_digest("idle")
        if not self.dry:
            p["idle_signature"] = sig
            lc.save_pipeline(self.root, p)

    def close(self, n: dict) -> Exit:
        self.date = n["date"]
        if self.dry:
            self.say(f"would close night {self.date}")
            return Exit.OK
        code, msg = night.close_night(self.root)
        self.say(msg)
        self.post_digest("closed")
        return status(code)

    # ------------------------------------------------------------------ #

    def dispatch_ready(self, ready: list[dict]) -> None:
        cap_n = int(self.cfg.get("RSR_MAX_SESSIONS", "2"))
        p = lc.load_pipeline(self.root)
        busy = {r.get("item") for r in lc.live_sessions(self.root)}
        for item in ready:
            iid = item.get("id")
            live = len(lc.live_sessions(self.root, "researcher"))
            if live >= cap_n:
                self.say(f"{live} researchers live (RSR_MAX_SESSIONS={cap_n}); waiting")
                return
            if not iid or iid in busy or iid in p["awaiting"]:
                continue
            code, msg = dispatch.dispatch(self.root, iid, dry_run=self.dry)
            self.say(msg)
            if code != Exit.OK and not self.dry:
                self.park(lc.load_pipeline(self.root), iid, f"dispatch refused: {msg}")

    def collectors(self) -> None:
        p = lc.load_pipeline(self.root)
        live = {(r.get("item"), r.get("job")) for r in lc.live_sessions(self.root)}
        for job, item in sorted(p["collecting"].items()):
            if (item, job) in live or item in p["parked"]:
                continue
            _code, msg = dispatch.dispatch(
                self.root, item, collect_job=job, dry_run=self.dry
            )
            self.say(msg)

    def manager_due(self) -> bool:
        p = lc.load_pipeline(self.root)
        if lc.live_sessions(self.root, "manager"):
            return False
        if p["manager_due"]:  # reconcile sets it when a researcher/job changed
            return True
        last = p.get("last_manager_end")
        if not last:
            return True
        minutes = float(self.cfg.get("RSR_MANAGER_INTERVAL_MIN", "60"))
        return self.now - dt.datetime.fromisoformat(last) >= dt.timedelta(minutes=minutes)

    def manager_cycle(self) -> None:
        if not self.manager_due():
            self.say("no manager cycle due")
            return
        wt = night.mgr_worktree(self.root)
        argv = lc.claude_argv(self.cfg, MANAGER_PROMPT, "rsr-manager", MANAGER_SETTINGS)
        if shutil.which("caffeinate") and self.cfg.get("RSR_NO_CAFFEINATE") != "1":
            argv = ["caffeinate", "-i", "-s", *argv]
        if self.dry:
            self.say(f"would run a manager cycle in {wt}: {argv}")
            return
        n_, _ = lc.new_cycle(
            self.root,
            {
                "kind": "manager",
                "item": None,
                "job": None,
                "argv": argv,
                "cwd": str(wt),
                "night": self.date,
                "start": lc.now().isoformat(timespec="seconds"),
                "pid": None,
                "env": {},
            },
        )
        path = lc.cycle_path(self.root, n_)
        rec = lc.load_json(path, {})
        rec["pid"] = os.getpid()  # foreground: the tick is the session
        lc.write_json(path, rec)
        rc = session.execute(self.root, n_)
        self.say(f"manager cycle {n_} exited rc={rc} -> {lc.route(rc)}")
        p = lc.load_pipeline(self.root)
        p["manager_due"] = False
        p["last_manager_end"] = lc.now().isoformat(timespec="seconds")
        lc.save_pipeline(self.root, p)
        reconcile.reconcile(self.root)
        self.over_budget()


def main() -> Exit:
    ArgumentParser(description=__doc__.split("\n", 1)[0]).parse_args()
    return status(Tick(lc.root()).run())


if __name__ == "__main__":
    run_main(main)
