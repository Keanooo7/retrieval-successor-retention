# Pre-registration — E0i reintroduction-gap threshold

> **AWAITING SIGNATURE (§6). NOT IN FORCE UNTIL SIGNED.**
>
> **No histogram has been computed, and none may be before the signed version is
> committed.** A threshold registered after seeing the data is not a threshold, and
> this gate protects 576 GPU-hours.
>
> **The numbers below are final** — the two competing derivations are reconciled
> per **D-G** and are not open for renegotiation. What remains is Brendan's
> signature, not a decision.
>
> *Commit note: an earlier draft of this file was swept into the Phase 0 repair
> commit by a `git add -A`. It carried the pre-D-G numbers and was never in force.
> This commit is the substantive pre-registration, and it still precedes any
> histogram.*

The spec calls this threshold pre-registered (§6 E0i, defect D-2, §16 condition 1)
and never gives the number. This supplies it.

## 0. The distinction the spec does not draw

D-2 asks one question — is `(40, 64]` adequately populated? — but there are
**two**, with different floors, different failure modes and different fixes.
Answer both, or the gate is ambiguous.

| | Training-side | Evaluation-side |
|---|---|---|
| Question | Can the policy **learn** long-gap retention? | Can we **measure** a difference? |
| Data | Training stream slices at `S = 80` | Held-out documents, **unsliced** |
| Why they differ | §3.1 fact 3: *"Streams are a training-time slicing device. **Validation and test documents are never sliced.**"* There is no `S = 80` ceiling on gaps at evaluation. D-2's "only 16 stream positions can host a `k = 64` reintroduction" is a statement about **training density only**. | |
| Failure means | The mechanism has no signal to fit | The dependent measure has no power |
| Fix if short | Enlarge the training corpus, or raise `S` — and `S` is already at the memory ceiling. **Expensive.** | **Enlarge the eval split.** Costs inference, not training. **Cheap.** |

Reporting one number for both is how a gate gets fudged: an eval-side shortfall
looks fatal when it is a day of inference, and a training-side shortfall looks
survivable when it is not.

## 1. Evaluation-side floor — the gate

Design: **paired.** The same held-out reintroduction events are scored under every
arm; only the retention policy differs. Paired is far more powerful than unpaired
and it is what E3 actually runs.

> ### 🔴 PASS
>
> ```
> n_raw × p²  ≥  150   in EVERY ONE of   (40, 48],  (48, 56],  (56, 64]
>          AND
> n_raw × p²  ≥  600   pooled across     (40, 64]
> ```
>
> drawn from **≥ 30 distinct documents**.

`n_raw` is the raw count of detected reintroductions in the bin. `p` is the coref
system's **precision**, measured on 100 hand-annotated reintroductions (T3).

### Why `p²` and not `p`

False-positive reintroductions carry no true effect, so they dilute the observed
effect by `p` — and the required `n` scales as `1/δ²`. Both independent derivations
agreed on this and it is settled: **the exponent is `p²`.** This is what makes T3's
precision measurement load-bearing rather than decorative, which is the hole D-2
names and then answers only with a population check that cannot see precision.

🔴 **An unmeasured `p` is exit 3 — "did not run". It is not a pass.** In ML a run
that produced no metric looks exactly like a run that produced a bad one, and `3`
must never collapse into `0`. There is no default `p`, and `p = 1` is not a
conservative assumption — it is the most permissive one available.

### Why the per-bin constraint binds, and the pooled figure is subsumed

Two derivations produced two numbers. They are reconciled here, and the
reconciliation is **D-G**, not a compromise:

| | Derivation | Floor | MDE at `σ_d = 0.30` |
|---|---|---|---|
| Per-bin | 150 per bin over three bins | binds | **2.0%** |
| Pooled (Studio) | 400 | subsumed | 1.2% |
| Pooled (adopted) | **600** | binds jointly | **1.0%** |

**The per-bin constraint binds** because a pooled-only gate can pass on an empty
top bin — and `(56, 64]` is where E3's claim lives. Pooling 600 events that are 550
short-gap and 50 long-gap would satisfy a pooled floor while leaving the headline
window unpowered, and the failure would surface as a null in week 9 that looks like
a finding about retention.

The Studio's pooled `≥ 400` is **subsumed**: 600 is stricter, and three bins at 150
already imply 450 pooled, so the pooled figure binds only on the distribution across
bins, which is the one thing per-bin floors do not constrain.

**2.0% per-bin is the number that matters**, and it is set against [P2]'s own 2–4%
headline effect — the only benchmark available. A gate that could not detect the
base paper's own effect size would not be a gate.

### `σ_d = 0.30` is an assumption, and it is labelled as one

The MDEs above assume a paired difference SD of 0.30 nats/token. That is an
assumption, not a measurement, and it is the only soft number in this section.

**When a real paired run produces a `σ_d`, recompute the MDE and report it beside
the headline.** Both the E0e FIFO run and the E0d subsample can produce one before
E3.

🔴 **The event threshold does not move retroactively.** `150 / 600` are the gate; a
measured `σ_d` changes what those counts *buy* in power, and that revised figure is
reported. It does not re-open the counts. A threshold that rescales after the data
arrives is not a pre-registration.

### Prior derivation, kept for the record

The earlier evaluation-side derivation is retained so the reconciliation is
auditable rather than asserted:

```
assumed per-sentence mean loss SD        σ      = 0.40 nats/token
assumed inter-arm correlation            ρ      = 0.90
paired difference SD    σ_d = σ·√(2(1−ρ))       = 0.40 × 0.447 = 0.179
target detectable difference             δ      = 0.05 nats/token
α = 0.05 two-sided, power = 0.80  →  (1.96+0.84)² = 7.85

n = 7.85 × (σ_d/δ)²  = 7.85 × (0.179/0.05)²      = 101
× DEFF = 1 + (m̄−1)·ICC,  m̄ ≈ 10 events/book, ICC ≈ 0.10   = 1.9   → 192
× 2, for the k-interaction across sub-buckets                      → 384  ("400")
```

It assumed `σ_d = 0.179`; D-G's `0.30` is the more conservative figure and is what
the MDEs above use. The ≥30-document requirement survives from it unchanged — it is
what keeps the design effect near the assumed 1.9.

### Bucket edges

`(40, 48]`, `(48, 56]`, `(56, 64]` — three bins of width 8, superseding the earlier
two bins of width 12. **The `(40, 64]` window may not be re-sliced to pass.**

## 2. Training-side floor — judgment, flagged as such

Weaker derivation than §1, and it should be read that way. The value head fits a
sum over events; if `(40, 64]` events are a negligible share of that sum, the head
optimizes short gaps and the long-gap behaviour is a tail it never sees a gradient
for. Absolute count matters too — the bilinear head is `d² + 2d` ≈ 16.6K parameters
at `d = 128`.

> **FLOOR: `(40, 64]` events are ≥ 2% of all reintroduction events in the training
> slices, AND ≥ 5,000 in absolute count.**

The 2% share is the meaningful constraint; the 5,000 is a sanity floor. Both are
Brendan's, not the spec's. Move them if you disagree — but move them **now**,
before the histogram.

## 3. Response ladder — ordered, pre-registered

A gate with only one response gets fudged. If a floor is missed, work down this
list in order and **record which rung was used.**

**Evaluation-side shortfall:**

1. **Enlarge the held-out eval split.** Inference-only; does not touch the training
   budget or the 30M-token training corpus. This is the expected fix and it is
   cheap.
2. **Narrow the window to `(40, 56]`** — i.e. drop the `(56, 64]` bin — where
   density is higher. Restate E3's claim to that window in the pre-registration,
   **before running**. Note what this costs: the dropped bin is the one the
   headline lives in, so this rung changes the claim, it does not rescue it.
3. **Accept reduced power**, state the achieved MDE in the paper, and drop the
   sub-bucket interaction test while keeping the main effect.
4. Only then: **E3's headline claim is not testable at this scale.** Report that
   and stop.

**Training-side shortfall:** there is **no cheap rung.** Enlarging the training
corpus or raising `S` both cost real money, and `S` is at the memory ceiling. A
training-side miss is a genuine red light and goes to Brendan as a
**stop-and-decide**, not to the ladder.

## 4. Also pre-registered here

- **The PG-19 subset-selection rule.** Which books enter the 30M-token subset, and
  the held-out split, fixed before any histogram. Picking books after seeing the gap
  distribution would make the gate unfalsifiable by a different route.
- **Bucket edges**, as stated in §1: `(40, 48]`, `(48, 56]`, `(56, 64]`. The
  `(40, 64]` window may not be re-sliced to pass.

## 5. Closed — the coref-attenuation term is IN

This section was open. **D-G closes it: the `p²` term is folded into §1's gate**,
not parked as a limitation.

The errata that got it there, recorded because the near-miss is instructive: the
first derivation wrote the attenuation as `1/p`. The reintroduction events a coref
system reports are diluted by precision `p` in the *effect*, and required `n` scales
as `1/δ²`, so the correct exponent is **`p²`**. Both sessions' derivations agreed on
this once stated; they disagreed only on the resulting count, which is what D-G
reconciles above.

Consequence: **T3 is on the critical path for E0i.** Without a measured `p` the gate
cannot be evaluated at all — exit 3, not exit 0.

## 6. Signature

Everything above is final and reconciled (D-G). This is a signature, not a
decision.

```
Threshold set by:  ______________________    Date: __________

Histogram not computed before this commit:   git SHA ______________
```

**On signing, record in `GATE-1.md`:** the signing SHA, and that `p` is unmeasured
until T3 reports — so E0i's current status is **exit 3, did not run**, and must be
reported that way rather than as a pending pass.
