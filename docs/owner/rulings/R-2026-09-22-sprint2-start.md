---
id: R-2026-09-22-sprint2-start
date: 2026-09-22
stated_in: 'MacBook owner-proxy session 2026-09-22 — Brendan, verbatim: "go with your recommendations, make sure you look at the other agent defs and see what would be better and just go with it. go with your recomendations" -- recorded in the Studio interactive session 2026-09-22 under R-2026-09-22-proxy-go (Brendan: "if the other window gives the go then you do it dont wait for me")'
---
**B2: Sprint 2 may start (once the Sprint 0 gate is green), limited to the instruments
whose validity does not depend on retrieval.**

| Item | Starts under this ruling? | Why |
|---|---|---|
| `e0e` (FIFO + EMA → `E[lifetime]` → `γ_b`) | ✅ yes | model-free bookkeeping on FIFO; retrieval does not enter it |
| `e0a-rerun` (μP coordinate check, unit-norm gestalts) | ✅ yes | a width-scaling property of the optimizer, not of memory use |
| code items (`eng-loo`, `eng-shadow`, `eng-leading-edge`, …) | ✅ yes | code may land before its sprint's gate |
| `e0d` (LOO answers) | ⛔ **held** | LOO measures how much each retained slot is worth. While held-out answer NLL at `2 ≤ gap ≤ M` sits at chance (`runs/decisive-shuffle/`, `runs/s0-03-rewardable-corpus/`), every slot is worth ≈0 and E0d cannot answer. **E0d additionally requires `R-*-retrieval-shown`**, written when `retrieval-curve` (or a successor) returns retrieval shown. |

The held half is enforced by a `requires` line on `docs/queue/items/e0d.md`, committed
separately, not by this prose.
