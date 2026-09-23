# Proposed owner rulings — 2026-09-22 (B1, B2, B3), NOT YET RULINGS

**Not in `docs/owner/rulings/` on purpose.** A file there satisfies queue predicates the moment it
lands, and the README says rulings are written in a session where Brendan states them. Brendan
approved these in the MacBook owner-proxy session ("go with your recommendations"). **The Studio's
interactive session records them as rulings once Brendan confirms them there. Copy each block
below to `docs/owner/rulings/<id>.md`, unchanged apart from `stated_in`.**

⚠️ **B1 is a criterion, and must NOT be filed as `R-*-sprint0-gate*`.** The `sprint_gate: S0`
predicate is satisfied by that filename existing, so filing it under that name opens Sprint 0
before liveness wiring lands.


---8<--- docs/owner/rulings/R-2026-09-22-s0-green-criterion.md

---
id: R-2026-09-22-s0-green-criterion
date: 2026-09-22
stated_in: 'MacBook owner-proxy session 2026-09-22 — Brendan, verbatim: "go with your recommendations, make sure you look at the other agent defs and see what would be better and just go with it. go with your recomendations"'
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

---8<--- docs/owner/rulings/R-2026-09-22-sprint2-start.md

---
id: R-2026-09-22-sprint2-start
date: 2026-09-22
stated_in: 'MacBook owner-proxy session 2026-09-22 — Brendan, verbatim: "go with your recommendations, make sure you look at the other agent defs and see what would be better and just go with it. go with your recomendations"'
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

---8<--- docs/owner/rulings/R-2026-09-22-hinge.md

---
id: R-2026-09-22-hinge
date: 2026-09-22
stated_in: 'MacBook owner-proxy session 2026-09-22 — Brendan, verbatim: "go with your recommendations, make sure you look at the other agent defs and see what would be better and just go with it. go with your recomendations"'
---
**B3: the srep-norm hinge is OFF (`srep_norm_reg_weight = 0.0`, arm B's setting) for new
experiments.**

- Evidence: `armB.ratio` 9.33 ± 4.68 vs `armC.ratio` 7.74 ± 6.06 (`runs/decisive-shuffle/`) —
  three seeds cannot separate them. Off is the simpler configuration and is the one S0-03 and
  `retrieval-curve` already use.
- **Scope:** experiments pass `srep_norm_reg_weight=0.0` explicitly. `TGConfig`'s code default
  (`0.01`) is **not** changed by this ruling: changing it would change every future config
  hash and needs its own eng item.
- **Revisit** only with a pre-registered comparison of B vs C at ≥ 5 seeds, PREREG committed
  first.
