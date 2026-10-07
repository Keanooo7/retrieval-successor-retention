---
id: R-2026-10-06-expire-span-q1-q2-q9
date: 2026-10-06
stated_in: 'Interactive PM session a3d27993, 2026-10-06 ~00:40 PDT, via the PM question form; answers verbatim in the body.'
committed: 'Brendan, 2026-10-07 interactive session, chose "Yes, commit both (Rec.)": committed as written; only this front matter was added.'
drafted_by: 'Claude Opus 5.5 (PM session). NOT written by Brendan.'
---
# R-2026-10-06-expire-span-q1-q2-q9 (DRAFT, for Brendan to commit to docs/owner/rulings/)

**Ruled by:** Brendan, interactive PM session a3d27993, 2026-10-06 at about 00:40 PDT. The selections were made through the PM's question form. The answers are recorded verbatim:

- **q1, the gradient path:** "predictor (Recommended)". `ExpireSpanConfig.grad_path = "predictor"`. Only the span predictor's `w, b` receive gradient from the LM loss through the span mask. The gestalt is detached at the predictor's input.
- **q2, the tuning budget:** "Equal configs+steps (Rec.)".
  - Expire-Span gets as many configs as RSR's grid: 9 = 3 α × 3 max-span L. The ramp R, the init and structured dropout come from [P7].
  - It gets the same optimizer steps × seeds as RSR.
  - The winner is selected on a validation split. The test split is read once.
  - All of this is pre-registered before any run.
- **q9, the optimizer group:** "Same as RSR's φ (Rec.)". `w, b` use exactly φ's optimizer in RSR run 1: AdamW, φ's learning rate, decoupled weight decay 0.01, ε 1e-8.

**Not ruled here:** q3–q8 and q10 of ADR-0010. They are fixed by [P7] where the ADR says so; otherwise the PREREG declares them, and they stay open for review.

**PM note, not part of the ruling. CORRECTED 2026-10-06 ~01:00 by measurement.**
The PM expected `through_gestalt` ≡ `predictor` on a frozen substrate. That is **REFUTED**.

- Over 5 frozen steps, `w, b` differ: max |Δw| = 8.75e-5.
- The reason: `w, b` shape each step's mask, so they sit upstream of every later gestalt, and TG writes gestalts to memory with their gradient graph.
- The two paths become bit-equal only with `detach_sreps_for_memory=True`. Both cases are pinned in tests on `feat/expire-span-arm`, at `dade81d` / `087a486`.

So q1 = `predictor` is a substantive choice, not a no-op.
