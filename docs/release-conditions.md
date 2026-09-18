# §16 release conditions

**Approved:** weeks 1–4. E0a–E0i, E1, E2. Roughly $100 of compute, no reputational
exposure.

**Not approved:** the PG-19 block, the four-figure spend, and everything
downstream.

Conditions 1–4 and 8 decide whether the project finishes.
**Conditions 5, 6 and 7 decide whether it is worth finishing.**

| # | Condition | Source | Status | Evidence |
|---|---|---|---|---|
| 1 | E0i histogram shows adequate `(40, 64]` population against a pre-registered threshold | D-2 | ☐ open | threshold: [`preregistration/e0i_threshold.md`](../preregistration/e0i_threshold.md) · result: `experiments/e0i/RESULTS.md` |
| 2 | `γ_b` rederived from measured lifetime; §3.5 demonstrated to move the argmin | D-1 | ☐ open | `experiments/e0e/RESULTS.md` (Sprint 2). Registry refuses `γ_b` until `E[lifetime]` is logged |
| 3 | Expire-Span implemented and run on synthetic **at the week-4 gate** | D-4 | ☐ open | week 4. **No demotion path** — its success invalidates the design, not the result |
| 4 | E7 small-`M` model funded (§4.1) **or** E7 cut and falsifier 4 withdrawn in week 1 | D-5 | ☐ open | [`ADR-0005`](decisions/ADR-0005-e7-stimulus-set.md) · `experiments/e0g/RESULTS.md` |
| 5 | Marginal/redundancy scoring specified and ρ(score, LOO Δloss) reported with and without | D-3 | ☐ open | `experiments/e0d/RESULTS.md`, then E1's `ν` sweep |
| 6 | §11 rewritten with the leading-edge strategy as named ancestor **and implemented baseline** | S-1 | ☐ open | `src/rsr/baselines/leading_edge.py` (stub) |
| 7 | **§15.2 non-empty, derived by the author, defended under push-back in person** | §15 | ☐ open | **Brendan only.** §15 states it must not be filled by a reviewer, an advisor, or a model. Agents may schedule this work and may not do it |
| 8 | ~~**Parallel GPU capacity reserved, not just total hours priced**~~ | §4.1 | **n/a — condition dissolved** | [`ADR-0007`](decisions/ADR-0007-all-training-on-the-mac-studio.md): no GPU is rented; everything runs on the Mac Studio. The condition guarded against pricing hours without securing capacity, and there is now no capacity to secure. Replaced by the `(S, d, batch)` ceiling **measured on this machine** — `experiments/e0c/RESULTS.md` |

## Notes that change how these are read

**Condition 8 is satisfied in Sprint 1 by a *hold*, not a reservation.** The
reservation *is* the four-figure commitment §16 declines to approve. A cancellable
hold with a movable start date satisfies the condition without pre-empting the
gate. Start date is **week 6**, not week 5 — see correction 7.

**Condition 8's sizing is larger than §4.1 states.** §4.1 computes 768 GPU-h for
weeks 5–7 and derives "floor 3" from it. §8 also schedules the E7 `M = 8` model
(72 h) and A1 into the same window → **842 GPU-h**, which is 3.21 GPUs at 65%
utilization with a 25% rerun margin. **Hold the floor at 4.** See correction 8.

**Condition 1 has two floors, not one.** The training-side and evaluation-side
questions have different failure modes and different fixes; an evaluation-side
shortfall is a day of inference, a training-side shortfall is a red light. See
the pre-registration.

**Condition 7 is the one agents must not touch.** §15 is a placeholder and must
not be filled by a model. Do not restate the pointer it leaves open, and do not
suggest answers to it.
