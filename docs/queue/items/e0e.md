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
status: blocked
attempts: 0
source: docs/ROADMAP.md:135; experiments/e0e/run.py:16
---
# E0e: FIFO run + EMA bookkeeping -> E_lifetime -> gamma_b

ROADMAP:96: the cheapest experiment in the project; it auto-unlocks `gamma_b` via the registry. RESEARCH-CONTEXT §12 item 2: `E[lifetime]` is itself policy-dependent -- the PREREG must say which policy's lifetime is measured.
