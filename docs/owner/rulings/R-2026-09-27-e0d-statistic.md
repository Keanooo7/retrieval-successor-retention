---
id: R-2026-09-27-e0d-statistic
date: 2026-09-27
stated_in: 'Interactive PM session, 2026-09-27. Brendan adopted the E0d round-table synthesis (~/Downloads/E0d_RoundTable_Synthesis.md, "Use the round table report") and answered four ruling questions verbatim: AUROC bar "0.85 (Recommended)"; harm ratio "ρ = 0.5 (Recommended)"; seed rule "Each seed separately (Recommended)"; r_i variant "Gated (Recommended)".'
---
# E0d's primary statistic and thresholds

## The statistic

E0d's primary gate is the round-table **Rank 1** design:
- **AUROC_strat.** Age-stratified AUROC of `r_i` for LOO-critical cells, `y = 1[LOO_Δ > τ]`. It is computed on full-memory query-step cells and weighted by each stratum's critical-cell share.
- **H.** The eviction-harm rate: how often the slot at `argmin r_i` is critical, over the steps with at least one critical cell.

## The thresholds and rules

| Item | Ruling |
|---|---|
| AUROC bar | **`AUROC_STAR: 0.85`**, per seed, through the 95% cluster-bootstrap CI |
| Harm bar | **`HARM_RATIO_STAR: 0.5`**. Pass if H ≤ 0.5 × H_age-random, per seed, through the CI |
| Seed rule | **Each seed separately.** E0d passes only if every seed passes. Pooled results are reported and never gate. |
| Primary `r_i` | **Gated** (correction 17). Raw `r_i` is reported as a secondary. |
| Primary knockout | Resample (the in-distribution LOO of spec §3.2.1). Zero is reported as a secondary. |

## Scope

- LOO remains the truth; §3.2.1 is unchanged.
- The secondaries are round-table Rank 2 and are never gating.
- The τ rule and the age bins are fixed in Amendment 2, before any data.
- This ruling satisfies E0d's C8 ratification requirement once Amendment 2 renames that check from ρ\* to these keys.
