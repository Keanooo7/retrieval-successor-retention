---
id: eng-loo
title: 'metrics/loo.py: the LOO metric (§3.2.1''s arbiter)'
class: eng
sprint: 2
workstream: W2
lane: agent-only
slots: 1
requires: []
status: blocked
attempts: 0
source: src/rsr/metrics/loo.py:22; docs/ROADMAP.md:95
---
# metrics/loo.py: the LOO metric (§3.2.1's arbiter)

Stub code item. Code may land before its sprint's gate (the experiment items that consume it carry the `sprint_gate`). Replace every `raise NotImplementedError` in the file with the implementation the cited spec section describes, tests first; add a mutation-battery entry that reddens only the new tests.

Spec §3.2.1. ROADMAP:95: the same machinery serves the oracle and the shadow scorer -- build once, use three times. E0d consumes it.
