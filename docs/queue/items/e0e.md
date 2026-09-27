---
id: e0e
title: 'E0e: FIFO run + EMA bookkeeping -> E_lifetime -> gamma_b'
class: exp
sprint: 2
workstream: W2
lane: cpu-det
slots: 1
requires:
- sprint_gate: S0
- ruling: R-*-sprint2-start
- prereg: experiments/e0e/PREREG.md
status: parked
attempts: 0
source: docs/ROADMAP.md:135; experiments/e0e/run.py:16
---
# E0e: FIFO run + EMA bookkeeping -> E_lifetime -> gamma_b

ROADMAP:135: the cheapest experiment in the project; it auto-unlocks `gamma_b` via the registry. RESEARCH-CONTEXT §12 item 2: `E[lifetime]` is itself policy-dependent -- the PREREG must say which policy's lifetime is measured.

**Status `parked` (2026-09-27, docs sync at `b21f3c3`): RAN; values NOT recorded.** Merged in #51.
`experiments/e0e/RESULTS.md` and `runs/e0e/ledger.json` are verified
(`runs/e0e/verification.json`): `tau` 0.27555, `E_lifetime` 13.1667, `gamma_b` 0.30380. The run
made none of its `record()` calls (ledger row `would_be_record_calls_NOT_MADE`), so the registry
still refuses `gamma_b`. The item is `parked`, not `done`, because recording the values is its
purpose, and that waits on owner decisions: D2, which substrate E1's constants come from; the `τ`
rule; and `γ_b` falling outside §3.5's range (RESEARCH-CONTEXT §12 items 8–10). Nothing requires
`item_done: e0e`.
