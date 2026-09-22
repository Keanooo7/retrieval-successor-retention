---
id: eng-test-cycle99-hardcode
title: 'test_evidence_machinery: stop hard-coding runs/canary/cycle-99'
class: eng
sprint: 0
workstream: W0
lane: agent-only
slots: 1
requires: []
status: blocked
attempts: 0
source: docs/lab-notes/for-brendan-2026-09-21.md:15; tests/test_evidence_machinery.py:217
---
# test_evidence_machinery: stop hard-coding runs/canary/cycle-99

`test_a_later_canary_reading_does_not_redden_the_09_18_tally` writes `runs/canary/cycle-99/ledger.json` into a copy of `runs/`; if a real `cycle-99` ever exists the test errors for a reason unrelated to what it tests. Pick a cycle number guaranteed absent (max existing + 1).

**Done when:** the test holds with a real `cycle-99` present.
