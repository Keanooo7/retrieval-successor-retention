---
id: R-2026-10-07-c3-loo-target
date: 2026-10-07
stated_in: 'Interactive PM session, 2026-10-07. Brendan, verbatim: "You will Decision C3 to pivot RSR Run 2 to the resample-knockout LOO target, the model has a direct, verified pathway to eliminate newcomer eviction and become a viable, high-performing memory mode". With it he pasted the PM recommendation from MORNING.md §3 item 1 (LOO at γ=0.9 on run 1''s harness; pre-registered newcomer diagnostics, the age-1 eviction share and a ψ(s,c) self-alignment probe; shadow buffer off as in run 1).'
evidence: 'B0-CF, class CF-POINTS: run/cf-ceiling 651fe63, rsr-verifier CONFIRMED 3a202b0 (γ=0.9 cap 0.916 / 0.955 / 0.874, CI lower 0.895 / 0.935 / 0.852). γ sweep, class SIGNAL (every r_i γ MISSES): verifier 7457f43. RSR run 1, ψC − FIFO −0.205 / −0.228 / −0.168, 38% of evictions at age 1 (seed 0): verifier 157f1fb.'
drafted_by: 'Claude Opus 5.5 (PM session). NOT written by Brendan. Committed on branch owner/2026-10-07, not on main; Brendan moves main.'
---
# C3: RSR run 2 trains on the resample-knockout LOO return

## Ruling

| id | Decision | Ruling |
|---|---|---|
| **C3** | The retention training target for RSR run 2 (spec §3.2.1, option C3; owner-only) | **The resample-knockout leave-one-out return at γ = 0.9.** `G^Δ_i(t) = Σ_k γ^k Δ_i(t+k)`, where `Δ_i` is `src/rsr/metrics/loo.py`'s `loo_delta_loss(mode="resample")` Δ-loss (E0d's instrument), and replaces the gated, share-of-M rescaled `r_i` return of run 1. |
| **Harness** | What else changes | **Nothing.** Run 2 is run 1's harness (`run/rsr-run1` @ `ed6302a`) with only the target swapped, so that ψC(run 2) − ψC(run 1) is attributable to the target: frozen arm B `ckpt3000`, `M = 16`, `S = 48`, `ν = 0`, `β = 1`, `b` off, **shadow buffer off**, the same arms, pool, seeds and steps unless the PREREG states a measured reason. |
| **Newcomer diagnostics** | Added, pre-registered | **The age-1 eviction share** and **a ψ̂(s_i, c_t) self-alignment probe**, both registered in run 2's PREREG before any run-2 data. They are diagnostics and not gates unless the PREREG makes one a gate. |

## What this ruling does not do

- It does not say the LOO target removes newcomer eviction. B0-CF is a hindsight ceiling in FIFO's world, not a trained ψ̂. Whether a learned ψ̂ on this target still evicts newcomers is what run 2 measures.
- It does not rule on falsifier 6 (the Expire-Span referendum, class SUPERSEDES), on E0i, or on ADR-0010 q1/q2/q9.
- It does not change the spec text, `docs/spec-corrections.md`, or any FROZEN constant. γ = 0.9 reaches run 2 by labelled `--policy-override`, as in run 1 (ADR-0009 L12); no value is recorded in the registry.
- §15: untouched.

## Unchanged prohibitions that bind the new target

- `Δ` is a **target** (`sg[G]`, §3.3). It is formed under `no_grad` in eval mode. No retention gradient reaches the transformer or `W_sent`; only `φ` learns.
- Age is excluded from ψ̂C. The age-only head stays a separate, labelled arm.
- `b` stays off and outside every differentiable path.
