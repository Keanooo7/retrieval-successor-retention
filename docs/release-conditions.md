# §16 release conditions

**Approved:** weeks 1–4. E0a–E0i, E1, E2. Roughly $100 of compute, no reputational
exposure.

**Not approved:** the PG-19 block, the four-figure spend, and everything
downstream.

Conditions 1–4 and 8 decide whether the project finishes.
**Conditions 5, 6 and 7 decide whether it is worth finishing.**

| # | Condition | Source | Status | Evidence |
|---|---|---|---|---|
| 1 | E0i histogram shows adequate `(40, 64]` population against a pre-registered threshold | D-2 | ☐ open — threshold **signed** (`d43960a`); E0i **exit 3, did not run** | threshold: [`preregistration/e0i_threshold.md`](../preregistration/e0i_threshold.md), signed by Brendan 2026-10-07 at `d43960a` (histogram-not-computed SHA `475a409`). It stays exit 3 until T3 measures `p` on the annotations (that file, "E0i remains exit 3") · result: `experiments/e0i/RESULTS.md` (**NOT RUN**) |
| 2 | `γ_b` rederived from measured lifetime; §3.5 demonstrated to move the argmin | D-1 | ☐ open — E0e measured, not recorded | `experiments/e0e/RESULTS.md` (`82014e6`), `runs/e0e/ledger.json`, verified: `γ_b` measured on fresh-stream arm B's FIFO run, but **no `record()` call was made** (which substrate E1's constants come from is D2, open). Registry refuses `γ_b` until `E[lifetime]` is logged. §3.5 moving the argmin: not demonstrated |
| 3 | Expire-Span implemented and run on synthetic **at the week-4 gate** | D-4 | ◐ **implemented and run**; falsifier-6 ruling deferred to E1 | Implemented `331d4af` (fixes `3399967`, `60a11ed`; [ADR-0010](decisions/ADR-0010-expire-span-baseline.md) PROPOSED, its q1/q2/q9 ruled in `R-2026-10-06-expire-span-q1-q2-q9`). Referendum on synthetic (frozen arm B): PREREG `3d5ed87`, Amendment 2 `13fc843`, run head `99646cb`; class **SUPERSEDES** against run 1's ψC; rsr-verifier record `3a241fc` (`verify/expire-span-referendum`); [`experiments/expire-span-referendum/RESULTS.md`](../experiments/expire-span-referendum/RESULTS.md); rows `es.*` in [`docs/claims-register.md`](claims-register.md). Owner ruling `6d3e5e3` (`R-2026-10-07-falsifier-6-deferred-to-e1`) defers falsifier 6 to gate E1 and pins config `L48-k1e-2`. **No demotion path** — its success invalidates the design, not the result |
| 4 | E7 small-`M` model funded (§4.1) **or** E7 cut and falsifier 4 withdrawn in week 1 — **and sentence length partialled out of the E7 correlation** | D-5 · S-7 | ☐ open | [`ADR-0005`](decisions/ADR-0005-e7-stimulus-set.md) · `experiments/e0g/RESULTS.md` · [correction 24 (S-7)](spec-corrections.md#24--s-7-sentence-length-is-an-uncontrolled-confound-in-e7) |
| 5 | Marginal/redundancy scoring specified and ρ(score, LOO Δloss) reported with and without | D-3 | ☐ open | `experiments/e0d/RESULTS.md` (class AGREE, `3bdb018`; verified at `run/e0d@0e51139`) reports Spearman ρ of gated `r_i` against resample LOO Δloss per seed with CIs (claims-register row `e0d.rho_Q.A13`): the target **without** any marginal/redundancy term. No marginal/redundancy score is specified, and E1's `ν` sweep has not run |
| 6 | §11 rewritten with the leading-edge strategy as named ancestor **and implemented baseline** | S-1 | ☐ open | `src/rsr/baselines/leading_edge.py` (still a stub that raises, checked 2026-10-20); [ADR-0011](decisions/ADR-0011-leading-edge-baseline.md) PROPOSED (`13177c6`), not accepted |
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

**Condition 4 now carries an analysis requirement, not only a funding decision.**
TG normalizes every gestalt to unit norm, so a 4-word and a 43-word sentence enter
memory at identical strength, while human encoding quality falls with sentence
length. The two systems therefore forget for different reasons, and **a correlation
between what RSR drops and what people fail to recall can be produced by sentence
length alone.** E7 must partial sentence length out of the correlation exactly as
§10.1 already partials out serial position. If the correlation survives it is
stronger; if it does not, RSR is tracking length rather than structural importance —
a vacuity failure **no falsifier names**, and one the age-based §7.1 gate
structurally cannot see, because length and age are independent. Condition 4 is not
met by funding the model alone. See **correction 24 (S-7)**.

**Condition 7 is the one agents must not touch.** §15 is a placeholder and must
not be filled by a model. Do not restate the pointer it leaves open, and do not
suggest answers to it.
