---
id: eng-loo
title: 'metrics/loo.py: the LOO metric (§3.2.1''s arbiter)'
class: eng
sprint: 2
workstream: W2
lane: agent-only
slots: 1
requires: []
status: done
attempts: 0
source: src/rsr/metrics/loo.py:22; docs/ROADMAP.md:134
---
# metrics/loo.py: the LOO metric (§3.2.1's arbiter)

Stub code item. Code may land before its sprint's gate (the experiment items that consume it carry the `sprint_gate`). Replace every `raise NotImplementedError` in the file with the implementation the cited spec section describes, tests first; add a mutation-battery entry that reddens only the new tests.

Spec §3.2.1. ROADMAP:134: the same machinery serves the oracle and the shadow scorer -- build once, use three times. E0d consumes it.

**Status `done` (2026-09-27, docs sync at `b21f3c3`).** Merged in #48. `src/rsr/metrics/loo.py`
is 800 lines and 0 `raise NotImplementedError`: resample and zero knockouts, and `L_PAIRS`. Its
mutation-battery entries are in the 245/245 RC 0 battery (`docs/mutation-battery.md`). E0d's
instrument now exists. The oracle and shadow-scorer uses have not been built on it.
