---
id: R-2026-10-01-rsr-run-1-owner-values
date: 2026-10-01
stated_in: 'Interactive PM session, 2026-10-01 ~11:10 PDT. The PM presented four owner slots (O1, O4, R-1, O5 of experiments/rsr-run1/PREREG.md) with evidence and a recommendation for each; Brendan replied, verbatim: "take the recomendations".'
evidence: 'experiments/rsr-run1/PREREG.md on prereg/rsr-run1 through Amendment 3d (fe0e01f; reviews 787875b, c02cae7, 7f7e2ad, 3ef97c8, 33ce631, de8f813, b543e96); docs/results/2026-09-30-verified.md row 7 (B2 psiC at gamma 0.9 EQUIV to FIFO, +0.0063/+0.0085/+0.0094); review 787875b (observable discounted weight 79.4% at gamma 0.9 vs 36.7% at 0.97; phi shrink factor 0.970 at wd 0.01); docs/spec-corrections.md item 20.'
drafted_by: 'Claude Opus 5.5 (PM session). NOT written by Brendan. Brendan commits it, or edits it, to docs/owner/rulings/; an agent write there is blocked by design.'
---
# RSR run 1: owner values (O1, O4, R-1, O5)

## Decisions

| id | Decision | Ruling |
|---|---|---|
| **O1** | γ for both RSR arms | **0.9.** The γ at which B2 measured ψ̂-C offline on this substrate (EQUIV to FIFO on all three seeds); horizon ≈ 10 < M = 16; 79.4 % of the discounted weight is observable under FIFO censoring. The oracle reference stays at γ_o = 0.9 independently (A2.2). |
| **O4** | φ weight decay | **0.01, decoupled** (as built). Shrinks φ by a factor of 0.970 over the run. ADR-0009 L8's decay clause stays unsigned for later runs; this ruling covers run 1 only. |
| **R-1** | Run length | **3000 steps** (187.5 epochs of the 256-document pool), **plus one additional DIAGNOSTIC held-out evaluation of φ at step 160 (10 epochs)**, so a few-epoch reading (spec §3.4's presumption, correction 31(b)) exists alongside the full-length one. The step-160 reading is diagnostic, never the headline. |
| **O5** | Correction 20 under a frozen transformer | **eval** — the frozen transformer runs with dropout off in every arm, for training and evaluation. Correction 20's "train mode for the LM loss" is read as applying to an LM loss that is optimised; under D2(b) the LM loss is only reported. This reading is for frozen-transformer runs; a run that trains the transformer uses train mode for the LM loss. |

## Unchanged

- R-2026-09-30-rsr-run-1-scope (N0, D2(a), D2(b), 31(b)) stands.
- PM decisions recorded in the PREREG (not owner-reserved): seeds 0, 1, 2; run alongside the battery; `srep_norm_reg_weight` 0.0.
- §15: untouched.
