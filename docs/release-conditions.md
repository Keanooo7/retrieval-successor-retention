# §16 release conditions

**Approved:** weeks 1–4. E0a–E0i, E1, E2. Roughly $100 of compute.
**Not approved:** the PG-19 block, the four-figure spend, and everything downstream.

Conditions 1–4 and 8 decide whether the project **finishes**. Conditions 5, 6 and 7 decide whether
it is **worth finishing**.

| # | Condition | Source | Status | Evidence |
|---|---|---|---|---|
| 1 | E0i histogram shows adequate `(40, 64]` population against a **pre-registered** threshold | D-2 | ⬜ open | threshold: `preregistration/e0i_threshold.md` · histogram: `experiments/e0i/` |
| 2 | `γ_b` rederived from measured lifetime; §3.5 demonstrated to move the argmin | D-1 | ⬜ open | `experiments/e0e/` → registry `gamma_b`; attribution via `RSRPolicy.attribution()` |
| 3 | Expire-Span implemented and run on synthetic **at the week-4 gate** | D-4 | ⬜ open | `src/rsr/baselines/stubs.py` → Sprint 3. ⚠️ see **B-3**: it must be *tuned* |
| 4 | E7 small-`M` model funded, **or** E7 cut and falsifier 4 withdrawn — in week 1 | D-5 | ⬜ open | `experiments/e0g/RESULTS.md` |
| 5 | Marginal/redundancy scoring specified; ρ(score, LOO Δloss) reported with and without | D-3 | 🟡 partial | `ν` term implemented in `RSRPolicy.score`; ρ needs E0d |
| 6 | §11 rewritten with the leading-edge strategy as named ancestor **and implemented baseline** | S-1 | ⬜ open | `LeadingEdgePolicy` is a stub. ⚠️ [P11] still unverified |
| 7 | **§15.2 non-empty**, derived by the author, defended under push-back in person | §15 | ⬜ open | **see the note below — this condition is unsatisfiable as written** |
| 8 | Parallel GPU capacity **reserved**, not just total hours priced | §4.1 | ⬜ open | T8. A cancellable hold, not a booking. ⚠️ see **B-2** |

---

## Condition 7 is unsatisfiable as written, and is not being quietly reinterpreted

§16 requires §15.2 *"defended under push-back **in person**."* There is one person on this project.
That clause has no referent.

**What is NOT happening:**
- No agent drafts §15.2, or a candidate for it, or a hint toward it. §15 forbids it in terms, and
  the prohibition is the point: this document became correct largely *because* it was corrected, and
  a load-bearing claim supplied by a model would reproduce that failure exactly one level up.
- The condition is not being marked satisfied by a substitute.

**What is available instead, labelled as what it is:** an adversarial refuter pass over a §15.2 the
**owner has already written** — the shape of the `/spec-review` skill in the owner's vault, where
independent reviewers are followed by a refuter that kills the weak findings.

🔴 **That is weaker than the condition and must be recorded as a substitute, never as the condition
met.** An adversarial agent is not a person, and recording it as one is the failure §15 exists to
name. If the project is ever assessed by someone else, condition 7 becomes satisfiable and should be
satisfied properly.

---

## Two conditions the verification pass changed

**Condition 3 (Expire-Span) is at risk of being adjudicated unfairly.** Falsifier 6 — *Expire-Span ≥
RSR on synthetic* — is the project's **stop condition**. [P7] carries a ramp length `R`, a loss
coefficient `α`, and *requires* structured dropout. The spec runs it once, untuned, against RSR's
9-config `γ`×`β` grid plus a `ν` sweep. An untuned Expire-Span losing is uninformative; a tuned one
winning ends the project. **Equal search budget per arm, reported per arm** — or the verdict is
provisional and must say so. See `docs/spec-corrections.md` B-3.

**Condition 8's dollar figure is not what the spec says it is.** §4.1 calls it "a four-figure spend"
without ever giving a number. At commodity rates the *experiments* are three figures
(~$400–$690 on demand); it becomes four figures only under **continuous reservation**
(~$820–$1,410). So the unapproved item is the **billing model**, not the experiment count — which is
exactly why week 1 buys a cancellable hold and the gate converts it. And the underlying GPU-hour
total itself needs re-deriving: see B-2.
