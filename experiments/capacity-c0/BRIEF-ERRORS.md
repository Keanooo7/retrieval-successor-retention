# C0 -- BRIEF ERRORS

Hand-written by the executor after the run (2026-09-24, Studio lead session).
`RESULTS.md` beside this file is generated and says "Do not edit", so its BRIEF
ERRORS placeholder is answered here instead. Brief:
`docs/lab-notes/dispatch-capacity-c0.md`.

1. **The owner's LLM servers are launchd daemons with `KeepAlive: true`.** The
   brief and `R-2026-09-22-mlx-servers` assume C0 can stop them itself. It cannot:
   `/Library/LaunchDaemons/com.vanta.{main,fast,embed}.plist` respawned all three
   within the check (attempt 1, `runs/capacity-c0.attempt-1/`, exit 3 --
   run.py's respawn detection worked as designed). Stopping them needs
   `sudo launchctl bootout system/com.vanta.<job>`, which an agent cannot run.
   Brendan ran it before attempt 2. `c0.stopped_servers` is therefore `[]` and
   `c0.restarted_servers` is `[]`: C0 neither stopped nor restarted anything on
   the run that counts. Restarting is `sudo launchctl bootstrap system
   /Library/LaunchDaemons/com.vanta.<job>.plist`, by the owner, after RSR work
   releases the machine.
2. **No preflight for `.venv/bin/pytest`.** `time_suite` runs
   `ROOT/.venv/bin/pytest`; pytest is in the `dev` optional extra, which `uv run`
   does not install in a fresh worktree. `--dry-plan` passed and attempt 2 ran
   11 minutes of waves before failing (`runs/capacity-c0.attempt-2/`, exit 3).
   Fixed in the environment with `uv sync --extra dev`, and attempt 3 was run as
   `uv run --extra dev python experiments/capacity-c0/run.py` (plain `uv run`
   re-syncs and removes the extra again). The command line in `RESULTS.md` omits
   `--extra dev`.
3. **`lanes generate` writes to the main checkout, not the worktree.** Run from
   `.worktrees/c0-run`, it wrote `/Users/keanooo7/retrieval-successor-retention/
   ops/lanes.json`. The committed `ops/lanes.json` was produced by the same
   command with `RSR_ORCH_ROOT` set to the worktree; the two are identical apart
   from `generated_utc`.
4. **Anchor drift.** The brief's `suite_threads` anchor said `:1956`; it is
   `:1976` at fee5f04 (fixed on `docs/record-fixes-2026-09-24`). Its two `scope`
   findings and `premises[1]` are true of any base after #38 merged.
5. **Predicted vs measured wall.** `--dry-plan` predicted ~37 min; attempt 3 took
   16 min 49 s (14:08:48 -> 14:25:37 -0700). A prediction, not a premise; noted
   because the brief quotes it.
