---
id: eng-s003-run-exit-protocol
title: 's0-03 run.py: precondition refusals exit 3 through rsr.exit_codes'
class: eng
sprint: 0
workstream: W0
lane: agent-only
slots: 1
requires: []
status: blocked
attempts: 0
source: docs/lab-notes/for-brendan-2026-09-21.md:13; experiments/s0-03-rewardable-corpus/run.py:190
---
# s0-03 run.py: precondition refusals exit 3 through rsr.exit_codes

`raise SystemExit(f"seed {seed}: held-out words not in the vocab ...")` at `:190` exits `1` (real failure) for a precondition refusal, and `:566` is `sys.exit(main())`, which skips `status()`'s bool/None refusal (S0-05). Use `refuse(Exit.DID_NOT_RUN, ...)` and `run_main(main)`.

**Done when:** the refusal exits `3`, pinned by a test, with a battery entry.
