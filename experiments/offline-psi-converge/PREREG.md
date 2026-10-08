---
run_id: offline-psi-converge
item: "Diagnostic A after offline-psi: train the same psi_hat to convergence on U, at the same learning rate"
question: >-
  offline-psi's fit was still improving when its 3000-step schedule ended (selected step 3000 on
  every seed and arm). Trained on the same data, with the same optimizer and lr, until U-val L_MC
  stops improving, does the content-only bilinear psi_hat reach a useful ranking (the PREREG §5
  fit cap of offline-psi), or not?
class: "DIAGNOSTIC (Tier D). U documents only. No EVAL read, no deployment, no spec falsifier. One fit label and one convergence label per seed; together they route an OWNER decision (§6)."
not: "Not a re-run of offline-psi's headline. Does not change offline-psi's labels (FIT-MISSES, FAILS) or run 2's class. Changes no learning rate and no muP multiplier (correction 15 is Brendan's). No scaling claim. No cognitive claim. §15 untouched."
seeds: "[0, 1, 2]"
status: "IN FORCE as of the commit that adds this file, ALONE, before any code. Owner go 2026-10-08: \"go ahead with diagnostic A and the run 2 log check\"."
---

# PREREG: offline ψ̂ trained to convergence (diagnostic A)

**Drafted 2026-10-08 by a model** (Claude Opus 5.5, an interactive session with Brendan). **Not
written by Brendan.** He approved diagnostic A in these words: *"go ahead with diagnostic A and the run 2 log check"*,
given in reply to the PM's report proposing "the same head and learning rate, run until the U-val
loss stops falling … no μP question". From this commit on, amendments are appended, dated and
committed alone.

## 0. Why

offline-psi (`prereg/offline-psi`; RESULTS `6d481f0`; rsr-verifier ok `dc6d1bd` on
`verify/offline-psi`) labelled the fit FIT-MISSES. But selection landed on the **last allowed step,
3000, on every seed and both arms**, with U-val L_MC still falling. Seed 0's cf_all went from 1.830
at step 0 to 1.631 at step 3000, falling about 0.0006 relative per 50 steps at the end, with a
gradient norm of about 0.005.

So FIT-MISSES does not separate two explanations:

- the head **cannot express** the ranking;
- the head **was not trained long enough** to express it.

This run separates them by training longer. It changes nothing else.

## 1. Data: offline-psi's, read, never recomputed

- **Gestalts:** `.worktrees/offline-psi/runs/offline-psi/seed{s}/gestalts.pt` (`s`, `c`
  `[1088, 48, 128]`, `ids`). sha256 per seed, equal to offline-psi's `measure-seed{s}.json`
  `gestalts.sha256`:
  - seed 0: `211244f1469b90972e3b1d3b6635b3ee9eb095e78b932dc396bba64df3f5562b`
  - seed 1: `3a18e2917b385a6339ef82e92535953105877cf295bbaf9f6495a4749a6a0c68`
  - seed 2: `74e3bec7e5366d631480f9e3ac9a07916926da21ffd60be57568acb3b3723706`
- **Δ:** B0-CF's `delta.pt`, with the same sha256 pins as offline-psi PREREG §1.
- **Targets and split:** offline-psi's. G is `discounted_returns(Δ, 0.9)`. The cell sets are
  cf_all (primary) and cf_live (secondary). U-train / U-val is positions `p % 7` (932 / 156).
  All of these come through offline-psi's harness functions, imported (`target_of`, `cell_mask`,
  `fit_tensors`, `u_documents`, `u_split`).
- **EVAL is not read.** No deployment.

## 2. The fit: offline-psi's, longer

Everything is offline-psi PREREG §3's, through its harness functions, imported. That covers the
head and its init (`new_policy`, run 2's psiC φ seed), the optimizer (`phi_optimizer`, run 2's lr
1e-3, φ weight decay 0.01, μP groups), the batch of 16, the permutation-epoch sampler seeded
`seed`, and the loss (`l_mc`). Only these change:

- **Step cap: 200,000** optimizer steps, against offline-psi's 3000.
- **U-val L_MC every 500 steps.** Evaluating U-val consumes no RNG, so the training trajectory is
  offline-psi's for the first 3000 steps (C2 checks this).
- **Convergence rule.** Let `best(k)` be the lowest U-val L_MC at or before step k. At each check
  with `k ≥ 20,000`, the fit is **CONVERGED** at k if `best(k) ≥ (1 − 0.001) · best(k − 10,000)`:
  less than 0.1% relative improvement over the last 10,000 steps. Training stops at the first such
  k. If it never stops before 200,000, the label is **CAP-HIT**.
- **Selection:** the lowest U-val L_MC over all checks, ties to the earlier step. This is
  offline-psi's rule (`select_step`).
- **Saved states:** φ at steps 3000, 10,000, 30,000 and 100,000 (where reached), and at the
  selected step.

## 3. Readings

1. **Fit cap on U-val** at the selected step, by offline-psi PREREG §5's definition and code: the
   model-free replay `replay_rates` against `reference_rates`, the cap from `B0.boot_rule`, and
   the paired per-document bootstrap over the 156 U-val documents.
2. **Fit label, offline-psi PREREG §5's rule, verbatim:**
   - **FIT-POINTS** if the cap CI lower is ≥ 0.80 on all 3 seeds;
   - **FIT-PARTIAL** if not, and the point is ≥ 0.50 on all 3 seeds;
   - **FIT-MISSES** otherwise.
3. **Convergence label per seed:** CONVERGED (with the step) or CAP-HIT, for cf_all. cf_live is
   reported beside it.
4. **Descriptive, never labelled:**
   - the U-val cap at each saved step (the trajectory);
   - train L_MC against U-val L_MC at the selected step (overfitting);
   - the within-step ρ(ψ̂, G) (offline-psi's `rho_psi_g`) at the selected step;
   - the L_MC of a constant predictor equal to U-train's mean G over the arm's cells, on U-val, as
     a scale for how much of the loss any content model removes;
   - B0-CF's own G replayed on U-val (the reference row, as offline-psi).

## 4. Controls (any failure: inconclusive, exit 3; numbers still reported)

| id | control | expected |
|---|---|---|
| C1 | Gestalt and Δ files hash to §1's values and offline-psi's pins | true |
| C2 | **Trajectory reproduction:** for each seed and arm, U-val L_MC at step 3000 equals offline-psi's recorded `fit.<arm>.history` value at step 3000 (`==` on the float; the cf_all values are 1.6307810744736677 / 1.5766100367590767 / 1.790370677418465) | `==` |
| C3 | Only φ is in the optimizer (offline-psi's `optimizer_holds_only_phi`); no age input | true |
| C4 | L_MC and the gradient norm are finite at every step | true |
| C5 | No EVAL document id (995000–997047) is generated or read; the run imports no deploy path | true |

## 5. Compute

- offline-psi measured 50 fit steps at 0.107 s (its Amendment 1). So 200,000 steps is at most
  about 7 min per arm and seed, plus U-val checks.
- **Worst case:** 6 fits of about 8 min each. Seeds run as three parallel cpu-det slot jobs.
- **Stop line:** 2 h wall. Past it, the builder stops and reports.

## 6. Readout (routes an OWNER decision)

| cf_all convergence | Fit label | Reading | Goes to Brendan with |
|---|---|---|---|
| CONVERGED, all seeds | POINTS or PARTIAL | The head can express a useful ranking. offline-psi's 3000 steps were too few at this lr, and runs 1 and 2 likely were too. | An online run with a longer φ schedule, or diagnostic B (the lr and the μP multiplier, correction 15) to make it affordable. |
| CONVERGED, all seeds | MISSES | Even fully trained, the content-only bilinear head cannot express G's ranking. offline-psi's §8 MISSES route stands. | The content+age arm or a richer head (both change the thesis). |
| CAP-HIT on any seed | any | The optimisation is too slow even at 200,000 steps. Capacity is still unresolved. | Diagnostic B (correction 15) as the next measurement. |

## 7. Seen already (declared)

- offline-psi's RESULTS, histories (steps 50–3000) and verification. Run 2's RESULTS.
- **The run 2 log check (same day, before this file).**
  - Run 2's ψC L_MC fell for about 2000 steps, then was flat or slightly rising over the last 1000:
    slope +0.007 to +0.010 per 1000 steps.
  - Its φ norm kept growing almost linearly: ‖W‖ about 88 / 146 / 186 at steps 1000 / 2000 / 3000
    on seed 0.
  - Its gradient norm decayed from 0.005 to 0.002.
  - The age-only head's L_MC was flat from step 250, with a gradient norm about 0.07.
- No fit beyond step 3000 exists.

## 8. Drafter's expectation (allowed to be wrong)

- **Convergence:** CONVERGED before 200,000, because the loss decay looked smooth.
- **Fit label:** FIT-PARTIAL is somewhat more likely than MISSES. At step 3000, ρ(ψ̂, G) was
  already 0.14–0.21 with the cap near 0. Rank correlation has to rise a long way before the cap
  moves, because the cap only rewards getting the bottom of each step's ranking right.
