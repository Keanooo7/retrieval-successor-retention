---
id: eng-s002-write-ledger-typed-exit
title: 's0-02 write_ledger.py: exit codes are recorded, not typed'
class: eng
sprint: 0
workstream: W0
lane: agent-only
slots: 1
requires: []
status: blocked
attempts: 0
source: docs/lab-notes/for-brendan-2026-09-21.md:14; experiments/s0-02/write_ledger.py:48
---
# s0-02 write_ledger.py: exit codes are recorded, not typed

`write_ledger.py` writes `exit_code=0` as a literal on seven command rows (`:48,55,61,74,83,88,101`). A typed exit code is a number no process produced -- the fabrication class the scoreboard audit exists to catch. Record the captured `$?` of each command (CLAUDE.md, *Reading an exit code*) or mark the row as not re-executed.

**Done when:** no ledger row carries an exit code that was not captured.
