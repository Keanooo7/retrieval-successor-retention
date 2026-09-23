---
id: eng-shadow
title: 'retention/shadow.py: the shadow buffer'
class: eng
sprint: 2
workstream: W2
lane: agent-only
slots: 1
requires:
- item_done: eng-loo
status: blocked
attempts: 0
source: src/rsr/retention/shadow.py:40; docs/ROADMAP.md:134
---
# retention/shadow.py: the shadow buffer

Stub code item. Code may land before its sprint's gate (the experiment items that consume it carry the `sprint_gate`). Replace every `raise NotImplementedError` in the file with the implementation the cited spec section describes, tests first; add a mutation-battery entry that reddens only the new tests.

Spec §3.4; `K` is read from `rsr.constants`, never hardcoded. Built on the LOO machinery (ROADMAP:134).
