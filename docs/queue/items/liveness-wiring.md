---
id: liveness-wiring
title: Wire memory liveness into every training run (exit 5 INERT, quarantined checkpoints)
class: eng
sprint: 0
workstream: W0
lane: cpu-det
slots: 1
requires:
- ruling: R-2026-09-22-liveness-go
- item_done: capacity-c0
brief: docs/lab-notes/dispatch-liveness-wiring.md
status: blocked
attempts: 0
source: docs/lab-notes/dispatch-2026-09-21-liveness-wiring.md:1
---
# Wire memory liveness into every training run

Held for the owner's go; `R-2026-09-22-liveness-go` gives it. The brief
(`dispatch-liveness-wiring.md`) supersedes the 2026-09-21 one and carries the corrected
exit mapping of `R-2026-09-22-inert-exit-5` (pre-registered bands: live 0, inert 5,
inconclusive 1, invalid 3).

**Class `eng`, not `exp` (integration, 2026-09-22):** it implements a decision rule the
decisive run already pre-registered (`experiments/decisive-shuffle/PREREG.md:33-45`,
Amendment 1) and runs CPU smoke runs only; it tests no new hypothesis, so it needs no
PREREG of its own. It waits on `capacity-c0` because the `cpu-det` lane does not exist
until C0 has generated `ops/lanes.json`.
