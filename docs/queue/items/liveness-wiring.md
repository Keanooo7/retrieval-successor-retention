---
id: liveness-wiring
title: Wire memory liveness into every training run
class: exp
sprint: 0
workstream: W2
lane: cpu-det
slots: 1
requires:
- ruling: R-2026-09-22-liveness-go
- prereg: experiments/liveness-wiring/PREREG.md
brief: docs/lab-notes/dispatch-2026-09-21-liveness-wiring.md
status: blocked
attempts: 0
source: docs/lab-notes/dispatch-2026-09-21-liveness-wiring.md:1
---
# Wire memory liveness into every training run

The brief is written and was held for the owner's go; `R-2026-09-22-liveness-go` gives it. Every anchor in the brief is re-derived at spawn time (the brief's own line 4).
