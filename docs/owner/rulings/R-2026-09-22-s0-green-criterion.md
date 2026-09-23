---
id: R-2026-09-22-s0-green-criterion
date: 2026-09-22
stated_in: 'MacBook owner-proxy session 2026-09-22 — Brendan, verbatim: "go with your recommendations, make sure you look at the other agent defs and see what would be better and just go with it. go with your recomendations" -- recorded in the Studio interactive session 2026-09-22 under R-2026-09-22-proxy-go (Brendan: "if the other window gives the go then you do it dont wait for me")'
---
**B1 answered as a criterion, not as the gate itself.** The Sprint 0 gate
(`docs/ROADMAP.md`, "the shuffle control, as a permanent instrument, green") is green
**when the `liveness-wiring` queue item is `done`** — not before, and not later than that.

- The decisive run (`runs/decisive-shuffle/`) meets the gate's first half: the shuffle reads
  live on arms B and C. The gate's second half — *"a run where memory contributes ≈0 is
  refused, not filed"* — does not exist until liveness wiring lands.
- **Retrieval is not part of this gate.** Memory that helps answers is a stronger claim than
  the gate makes; it is tested by `retrieval-curve` and gates E0d instead
  (`R-2026-09-22-sprint2-start`).

🔴 **Why this file is not named `sprint0-gate`.** The queue's `{sprint_gate: S0}`
predicate is satisfied by the mere existence of a signed `R-<date>-sprint0-gate*` file
(`scripts/orchestrator/workqueue.py`). Writing that file now would open Sprint 0 before its
refusal half exists — a hidden failure. **When `liveness-wiring` is `done`, an interactive
session writes `R-<date>-sprint0-gate-green.md` citing this criterion**; no further owner
judgement is needed, because the criterion is already decided here.
