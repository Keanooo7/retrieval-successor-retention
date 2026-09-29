---
id: R-2026-09-27-e0d-statistic-amended
date: 2026-09-27
supersedes_in_part: R-2026-09-27-e0d-statistic (the harm gate and the form of the AUROC gated at 0.85)
stated_in: 'Interactive PM session, 2026-09-27. Brendan, verbatim: "just review the options for the questions you gave me agaist the project roadmap the plan it built out you should make the desisions that get us there effectivly and with proper reaserch intent that we have designed". The PM decided under that delegation; the rationale is in DIGEST cycle 17.'
---
# E0d primary statistic: amended

## What changes

1. **The eviction-harm rate `H` no longer gates.**
   - It is reported as the **"argmin-hit rate"**, with its limit stated. Only 1.9% of LOO-critical slots are still needed after their step, so `H` does not measure eviction damage.
   - The eviction-consequence question belongs to **B2** (the closed-loop answer accuracy of learned eviction), under B2's own pre-registered rule.
   - `HARM_RATIO_STAR` is withdrawn as a gate.
2. **`AUROC_STAR: 0.85` now applies to `AUROC_strat,pct`.** This is the age-stratified AUROC computed on within-step percentile ranks of `r_i`.
   - It controls age exactly (one stratum per age) and removes step-level attention concentration.
   - Measured on synthetic data: the pure step-level artefact scores 0.502; the controls score 0.967 and 0.968.
   - The raw `AUROC_strat` is reported, not gated.

## What is unchanged

- Each seed must pass separately; pooled results are reported and never gate.
- Gated `r_i` is primary. The resample LOO knockout is primary.
- LOO remains the truth (§3.2.1).

## τ

`τ` is the q = 0.995 quantile of off-answer-cell |Δ_resample| on Q-steps. It is computed **per seed** on the already-inspected set E `[64,128)` and frozen in Amendment 2, before any data from `[262144, 263168)` is read.
