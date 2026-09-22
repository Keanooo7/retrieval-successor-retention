---
id: eng-s002-qtok-exit-protocol
title: 's0-02 measure_qtok_collapse.py: precondition refusals exit 3'
class: eng
sprint: 0
workstream: W0
lane: agent-only
slots: 1
requires: []
status: blocked
attempts: 0
source: docs/lab-notes/for-brendan-2026-09-21.md:13; experiments/s0-02/measure_qtok_collapse.py:131
---
# s0-02 measure_qtok_collapse.py: precondition refusals exit 3

Three `raise SystemExit("...")` precondition refusals (`:131`, `:139`, `:141`) exit `1`; `:421` is `raise SystemExit(main())`. Convert to `refuse(Exit.DID_NOT_RUN, ...)` and `run_main(main)`.

**Done when:** each refusal exits `3` under test.
