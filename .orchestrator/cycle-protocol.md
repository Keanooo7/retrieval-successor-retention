# One manager cycle

You were started by `scripts/orchestrator/tick.py` with *"Run one cycle per
.orchestrator/cycle-protocol.md"*. You are `rsr-manager` (`.claude/agents/rsr-manager.md`
-- read it; everything there still binds). Your checkout is `.worktrees/_mgr`, on
`night/<date>`. The main checkout is `$RSR_ORCH_ROOT`; the loop's state is under
`$RSR_ORCH_ROOT/.orchestrator/`.

**One cycle, then exit.** The tick starts the next one. A cycle that does not end
is a cycle the spend cap ends for you.

## What you may not do -- the loop depends on each

- 🔴 **Never launch a process that outlives you.** No researcher, no job, no
  `nohup`, no `&`, no `orchestrator.dispatch`, no `orchestrator.submit`. The tick
  dispatches; you decide. The one exception is `rsr-manager.md`'s: re-executing a
  researcher's claim to check it, in the foreground, and exiting when it exits.
- 🔴 **Never merge to `main`, push `main`, or move it.** Not by merge, reset, rebase
  or push (R-2026-09-22-night-branch). Merges into the night branch are done by
  `orchestrator.merge` from the tick, behind the verification gate and the
  frozen-diff guard -- not by you.
- **Never edit** `preregistration/`, an existing `runs/*/` file, `runs/canary/baseline*`,
  `docs/owner/`, `docs/spec/`, or a FROZEN entry of `src/rsr/constants.py`. The
  merge guard HALTs the whole loop on any of them.
- **Never contact the owner directly.** Anything for the owner goes in
  `docs/lab-notes/for-brendan-<date>.md`, one line each; the tick batches it into
  the digest (R-2026-09-22-owner-out-of-loop).
- **Never fill §15**, and do not restate or suggest answers to it.

## The cycle

1. **Orient (cheap).** `cat $RSR_ORCH_ROOT/.orchestrator/state/night.json` and
   `.../state/pipeline.json`. Items in `awaiting` wait on you; `collecting` jobs have
   a collector coming; `parked` items carry their reason.
2. **Review every item awaiting verification.** Read the researcher's report
   (the leading header of `.orchestrator/outbox/researcher.md` in that item's
   worktree, `.worktrees/<item>`) and apply `rsr-manager.md`'s rejection table
   row by row. For each item write `runs/<item>/verification.json` **in this
   checkout** and commit it on the night branch:

   ```json
   {"status": "ok" | "failed" | "inconclusive",
    "item": "<item>", "run_branch_sha": "<git rev-parse run/<item>>",
    "reexecuted": ["<the exact command>", "..."], "exit_codes": {"<cmd>": 0},
    "rejections": ["<table row>: <why>"], "reviewer": "rsr-manager",
    "note": "<one line>"}
   ```

   `ok` only when no rejection row applies **and** you re-executed at least one of
   its claims and got its number. `failed`: a rejection row applies. `inconclusive`:
   you could not re-execute (say why) -- it parks the item for the owner. The tick
   merges `ok` items and parks the rest.
3. **Re-execute at least one claim** this cycle (`rsr-manager.md`, "Verify by a
   different path"). Capture every exit code on the same line, never through a pipe
   (`cmd > out 2>&1; rc=$?`). 🔴 **An empty re-execution record is a failed cycle.**
4. **At most ONE new brief.** If the queue needs one and a falsifier justifies it:
   write `docs/lab-notes/dispatch-<id>.md` (falsifier, files in scope, bar, done
   when, do NOT), run `PYTHONPATH=$RSR_ORCH_ROOT/scripts python -m
   orchestrator.lint_brief <brief> --base <base_sha>` and fix every finding, then
   commit the brief **alone, in its own commit**. 📌 A brief written tonight is on
   the night branch, not at the pinned base, so it cannot be dispatched tonight: it
   becomes dispatchable after the owner merges the night. Say so in the scoreboard
   row. Refuse to write one for a config that already ran unchanged, one with no
   falsifier, or one waiting on an owner decision.
5. **Escalate, don't decide.** Anything in `rsr-manager.md`'s "Escalate rather than
   decide" list goes to `docs/lab-notes/for-brendan-<date>.md`, committed. Keep
   working on everything that does not depend on the answer.
6. **Scoreboard row.** Append to `docs/lab-notes/overnight-<date>.md`: run_id,
   falsifier, expected, observed, verdict, what-was-verified-by-re-execution.
   *"We learned nothing this cycle"* is a writable row.
7. **Commit, then exit 0.** Everything you wrote is committed on the night branch
   (`/opt/homebrew/bin/git`, never bare `git`); an uncommitted file is invisible to
   the merge and to the owner.

## Stop rules (dispatch-2026-09-21-overnight.md) as they apply to a cycle

- A frozen constant, a pre-registered threshold, or an arm changed to make a gate
  pass -> `verification.json` `failed`, a line to the owner, and write
  `$RSR_ORCH_ROOT/.orchestrator/HALT` with the reason. That stops the night.
- The canary moves -> mark every result since the last good reading suspect in the
  scoreboard; two moves in a row -> HALT with the reason.
- A declared dependency unmet -> refuse that brief in writing; continue.
- Two failures with the same cause -> the tick parks the item; do not re-brief it.
- Wall clock: no new brief after `last_dispatch`; after `park_by` the tick starts no
  cycle at all.
