---
id: R-2026-09-30-rsr-run-1-scope
date: 2026-09-30
stated_in: 'Interactive PM session, 2026-09-30 ~19:30 PDT. Brendan answered four questions in the session (AskUserQuestion), verbatim choices: N0 = "Build it now"; D2 part 1 = "Arm B ckpt3000 (Recommended)"; D2 part 2 = first "I dont know what do you think provide contextual reasoning", then, after the PM''s reasoning, "Freeze for run 1 (Recommended)"; 31(b) = "One pass over the doc pool".'
evidence: 'Evidence brief (Claude Doc "RSR owner decisions: evidence brief", https://claude.ai/code/artifact/ac5e9833-70cf-4e73-8ab7-da4b85a66ab1); docs/results/2026-09-30-verified.md @ 3493fc8 (B5 rows 35-38; B2 rows 5-8); RESEARCH-CONTEXT §6 rows fresh-stream, retention-readability, carry-forward, E0e (@ 898628a).'
drafted_by: 'Claude Opus 5.5 (PM session). NOT written by Brendan. Brendan commits it, or edits it, to docs/owner/rulings/; an agent write there is blocked by design.'
---
# RSR run 1: build the learning path; substrate, transformer and epoch

## Decisions

| id | Decision | Ruling |
|---|---|---|
| **N0** | Build ADR-0009 slice 1 (the ψ̂ learning path) | **Build it now.** Scope as `R-2026-09-30-adr0009-signoff` (L1, L2, L6, L8 clip, L9–L13): shadow buffer off and refused, `b` off, the §7.1 diagnostics and the censored-target caveat in every RESULTS. Building is not running: run 1 still needs its own PREREG. |
| **D2 (a)** | Substrate | **Resume from fresh-stream arm B, `ckpt3000`.** It is the only substrate with retrieval on unseen documents (R 0.87 / 0.85 / 0.89 at step 4000), and it reads facts at ranks FIFO never produces (rank penalty 0.0016 / 0.0033 / 0.0084). Arm A (fresh start) does not escape by step 9000. |
| **D2 (b)** | Transformer during run 1 | **Frozen** (no `L_NTP` update) **for run 1**. "Training continues" is kept as a later arm, pre-registered before it runs. Reason: arms then differ only in the eviction rule; B5 shows LM loss still falling 5.4–6.2 % per 1000 steps at step 9000, which would let the FIFO and RSR arms' transformers diverge. Consequence: ADR-0009's T4 bitwise isolation test is vacuous under freezing, so isolation also gets a unit test in a non-frozen configuration. |
| **31(b)** | What "one epoch" is for `T_warm` (§3.4) | **One pass over the document pool**: 256 documents seen, i.e. 256 / batch optimizer steps (16 at batch 16). `T_warm` is reported as a fraction of total epochs under this definition. |

## Unchanged and still open

- E0e's constants (`tau`, `gamma_b`) stay unrecorded: `b` is off in slice 1 (L12), so they are not read.
- `ν` and `β`: still owed by E1. Run 1 sets them by labelled override (L12), declared in its PREREG; no value is recorded in the registry.
- ADR-0009 L3, L4, L5, L7 (premise falsified by B1), N1, N3, N4: unchanged.
- §15: untouched.
