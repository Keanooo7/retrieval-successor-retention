---
id: eng-bias
title: 'retention/bias.py: ProtectionBias'
class: eng
sprint: 2
workstream: W2
lane: agent-only
slots: 1
requires:
- ruling: R-*-bias-interface
status: blocked
attempts: 0
source: src/rsr/retention/bias.py:60; docs/RESEARCH-CONTEXT.md:1068
---
# retention/bias.py: ProtectionBias

Stub code item. Code may land before its sprint's gate (the experiment items that consume it carry the `sprint_gate`). Replace every `raise NotImplementedError` in the file with the implementation the cited spec section describes, tests first; add a mutation-battery entry that reddens only the new tests.

Spec §3.5. **Blocked on the owner** (RESEARCH-CONTEXT §12 item 3): which side of the `ProtectionBias` / `_score` interface moves. Not worked around. `gamma_b` and `tau` come from `rsr.constants` (E0e); `b` never enters `psi_hat` or any differentiable path.
