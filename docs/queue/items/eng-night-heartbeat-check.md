---
id: eng-night-heartbeat-check
title: A missing night heartbeat fails the morning check (silence is never a pass)
class: eng
sprint: 0
workstream: W0
lane: agent-only
slots: 1
requires: []
status: blocked
attempts: 0
source: scripts/orchestrator/night.py:158; scripts/orchestrator/tick.py (no per-pass heartbeat is written)
---
# A missing night heartbeat fails the morning check

**Failure mode (hidden):** the nightly launchd run does not start at all (Mac asleep, nobody
logged in, the job unloaded) and leaves nothing behind. Today nothing detects that:

- `tick` writes no per-pass heartbeat;
- `night close` (`scripts/orchestrator/night.py:158`) returns `2` "no night recorded" only if it
  runs, and a loop that never started never runs it.

**Task (failure-finding):**

1. `tick` appends one line per pass to `.orchestrator/state/heartbeat-<night>.jsonl` (time,
   phase, exit code), including passes that refuse (`3`) or find HALT.
2. A morning check, `orchestrator.morning check --night <date>`, exits `1` when that file is
   missing or empty, or when its last line is older than the window's `park_by`; `0` only when
   the night demonstrably ran; `3` if it cannot read the state. Never `0` on absence.
3. The check must be runnable **outside the loop** (the owner's side, or a second machine
   reading the pushed night branch), because a loop that never started cannot report its own
   absence. Its result is the first line of the morning digest.

**Bar:** mutations -- the heartbeat write skipped; a missing file reading as `0`; a stale file
reading as `0` -- each reddening only its own test.

**Provenance:** raised in the owner's proxy review (RCM, "Nightly launchd run -- doesn't start
and leaves nothing behind -- hidden -- failure-finding"), relayed into the interactive session
of 2026-09-22. That review is not an owner ruling; this is an engineering item, not a decision.
