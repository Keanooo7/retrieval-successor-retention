---
id: R-2026-09-22-inert-exit-5
date: 2026-09-22
stated_in: 'interactive planning session, 2026-09-22 — Brendan: "it is better to build better systems than be stuck with old ones we micro fix. see if the new code 5 is better"; evaluated, recommended, plan approved.'
---
"Ran and inert" gets a new exit code **`5 INERT`**, not `1`.

- Reason: `docs/gates.md` / `src/rsr/exit_codes.py` exist because distinct facts
  sharing a code were conflated (2↔3, 3→0). "Trained, but memory carries no
  row-specific content" is not "the check caught a failure"; it means no eviction
  rule has anything to act on. The orchestrator routes on the code alone: `1` →
  regression/debug item; `5` → model/training-config investigation, and retention
  experiments on that checkpoint are refused.
- **Mapping — follows the pre-registered bands exactly**
  (`experiments/decisive-shuffle/PREREG.md:35-38,45`), no new thresholds:

  | Amendment 1 verdict | exit |
  |---|---|
  | live — any seed `ratio ≥ 0.1` | `0 OK` |
  | inert — every seed `ratio ≤ 0.01` | `5 INERT` |
  | inconclusive — otherwise (the 0.01–0.1 band) | `1 FAIL` — liveness not demonstrated; the ledger records the band label. Never `0`, never `5`. |
  | a control fails / `ratio == 1.0` exactly / measurement raised | `3 DID_NOT_RUN` — the measurement is not valid |

- Cross-row cosine (arm A's row-agnostic memory) is **recorded** but has **no
  threshold**; proposing one needs its own PREREG committed first.
- Scope of change: `Exit.INERT = 5`, `status()` range 0–5, `tests/test_exit_codes.py`,
  `docs/gates.md` table, battery entries. Implemented by the liveness-wiring brief.

**Erratum (2026-09-22, same session).** The first draft of this file and of the
approved plan said the trigger was "`ratio < 0.1`". That was a transcription error by
the assistant: it folded Amendment 1's `inconclusive` band into `inert`, which the
PREREG forbids ("reported as such, not rounded to either outcome", `:45`). Caught by
build agent C before anything shipped. The `1` for the inconclusive band is the
assistant's recommended default under R-2026-09-22-owner-out-of-loop and is listed in
the owner digest as revisable.
