# ops/ -- running the orchestrator unattended

Everything here is **written, not installed**. Nothing in the repository loads the
launchd job; you do, by hand, when you decide to.

| File | What it is |
|---|---|
| `orchestrator.env` | Caps, the night window, loop limits. Read by `scripts/orchestrator/loopcore.py`. |
| `launchd/com.keanooo7.rsr.tick.plist` | Runs `scripts/orchestrator/tick.zsh` every 600 s, and once at load. |

Rulings this machinery implements: **R-2026-09-22-night-branch** (unattended work
merges only into `night/<date>` at a pinned base; `main` never moves during a night;
you merge once each morning) and **R-2026-09-22-owner-out-of-loop** (you hear from it
only on the pinned GitHub issue labelled `rsr-digest`; decisions that are yours are
batched into the digest).

## Before installing

1. **Set the spend caps.** `RSR_CYCLE_CAP` and `RSR_NIGHT_CAP` in
   `ops/orchestrator.env` ship as `UNSET`, and the tick refuses (exit 3, one notice)
   until both are positive numbers. That is deliberate: the caps are your decision.
2. Check the window (`RSR_WINDOW_OPENS`, `RSR_LAST_DISPATCH`, `RSR_PARK_BY`,
   `RSR_DIGEST_BY`). `RSR_WINDOW_OPENS=21:00` is an assumption; the other three are the
   2026-09-21 overnight dispatch's stop rules.
3. `.claude/settings.orchestrator.json` (manager) and `.claude/settings.researcher.json`
   (researchers) must exist on `main` -- every session is started with one of them.
4. `gh auth status` must be logged in as you (the digest issue is posted with it).
5. `uv sync --frozen --extra dev` in the repository; the tick uses `.venv/bin/python`.
6. Rehearse without launching anything:

   ```zsh
   DRY_RUN=1 zsh scripts/orchestrator/tick.zsh; rc=$?; print "rc=$rc"
   ```

## Install

```zsh
cd ~/retrieval-successor-retention
mkdir -p .orchestrator/logs            # launchd will not create the log directory
cp ops/launchd/com.keanooo7.rsr.tick.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.keanooo7.rsr.tick.plist; rc=$?; print "rc=$rc"
launchctl print gui/$(id -u)/com.keanooo7.rsr.tick | head -20
```

## Uninstall

```zsh
launchctl bootout gui/$(id -u)/com.keanooo7.rsr.tick; rc=$?; print "rc=$rc"
rm ~/Library/LaunchAgents/com.keanooo7.rsr.tick.plist
```

Uninstalling stops future ticks. It does not kill a researcher or a job already
running (they run in their own sessions); `.orchestrator/jobs/*.json` and
`.orchestrator/cycles/*.json` name their pids.

## HALT -- stopping it without uninstalling

```zsh
echo "owner: stopped by hand $(date -Iseconds)" >> .orchestrator/HALT     # stop
cat .orchestrator/HALT                                                    # why it stopped
rm .orchestrator/HALT                                                     # resume
```

While `.orchestrator/HALT` exists every tick exits 0 at its first line. The loop
writes HALT itself when:

- the **frozen-diff guard** trips on a merge (`preregistration/`, an existing
  `runs/<id>/`, `runs/canary/baseline*`, `docs/owner/`, `docs/spec/`, or a FROZEN
  entry of `src/rsr/constants.py` changed) -- the most serious rejection; read the
  branch before removing HALT;
- the night's spend reaches `RSR_NIGHT_CAP`;
- two consecutive manager cycles fail with the same cause.

Each writes one line of reason and posts one notice.

## Each morning

The digest is on the `rsr-digest` issue and in `.orchestrator/state/digest-<date>.md`;
a skeleton is committed on the night branch as `docs/lab-notes/morning-<date>.md`.
Merge `night/<date>` into `main` yourself, once. Nothing unattended does.
