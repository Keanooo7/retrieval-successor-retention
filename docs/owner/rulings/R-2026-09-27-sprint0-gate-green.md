---
id: R-2026-09-27-sprint0-gate-green
date: 2026-09-27
stated_in: 'Written by the interactive RSR PM session on 2026-09-27, as R-2026-09-22-s0-green-criterion directs ("When `liveness-wiring` is `done`, an interactive session writes `R-<date>-sprint0-gate-green.md` citing this criterion; no further owner judgement is needed, because the criterion is already decided here"). Brendan approved the 2026-09-27 plan (PLAN-v4) in that session.'
---
**The Sprint 0 gate is green,** under `R-2026-09-22-s0-green-criterion`.

- **The criterion:** the gate is green when the `liveness-wiring` queue item is `done`.
- **The evidence:** `docs/queue/items/liveness-wiring.md` reads `status: done` at `941a68e`. Liveness-wiring merged via PR #50 (`Exit.INERT = 5`); its content is in main.
- **Scope:** this opens exactly the `{sprint_gate: S0}` predicate. It does **not** satisfy `ruling: R-*-retrieval-shown`. That ruling is Brendan's own, and E0d stays held until it exists.
