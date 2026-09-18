# Pre-registration — E0i reintroduction-gap threshold

> **UNSIGNED DRAFT. NOT IN FORCE. DO NOT COMMIT UNTIL SIGNED.**
>
> **No histogram may be computed before the signed version is committed.** A
> threshold registered after seeing the data is not a threshold, and this gate
> protects 576 GPU-hours.

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

## 1. Evaluation-side floor — derived

Design: **paired.** The same held-out reintroduction events are scored under every
arm; only the retention policy differs. Paired is far more powerful than unpaired
and it is what E3 actually runs.

```
assumed per-sentence mean loss SD        σ      = 0.40 nats/token
assumed inter-arm correlation            ρ      = 0.90
paired difference SD    σ_d = σ·√(2(1−ρ))       = 0.40 × 0.447 = 0.179
target detectable difference             δ      = 0.05 nats/token
α = 0.05 two-sided, power = 0.80  →  (1.96+0.84)² = 7.85

n = 7.85 × (σ_d/δ)²  = 7.85 × (0.179/0.05)²      = 101

× design effect for clustering within books
  DEFF = 1 + (m̄−1)·ICC,  m̄ ≈ 10 events/book, ICC ≈ 0.10   = 1.9   → 192

× 2, because the k-interaction across two sub-buckets is
  "the strongest form of the result" (§6, E3)                     → 384
```

> **FLOOR: ≥ 400 reintroduction events in `(40, 64]` in the held-out evaluation
> split, with ≥ 150 in each of `(40, 52]` and `(52, 64]`, drawn from ≥ 30 distinct
> documents.**

The ≥30-document requirement is what keeps the design effect near the assumed 1.9.

**σ and ρ are assumptions, and they are measurable before E3.** Measure both on the
E0e FIFO run and the E0d subsample. If the measured paired SD exceeds 0.179, the
floor rescales as

```
400 × (σ_d,measured / 0.179)²
```

and is **re-derived before the histogram is consulted again.** That clause is what
makes this a pre-registration rather than a guess with a number on it.

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
2. **Narrow the window to `(40, 56]`**, where density is higher. Restate E3's claim
   to that window in the pre-registration, **before running**.
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
- **Bucket edges**, as stated in §1. The `(40, 64]` window may not be re-sliced to
  pass.

## 5. Open — Brendan's call before signing

**The coref-attenuation term is not currently in §1.** T3 measures the coref
system's precision `p` on 100 hand-annotated reintroductions. False-positive
reintroductions carry no true effect, so they dilute `δ` by roughly `p`, and the
required `n` scales as `1/p²`. The gate would then read

```
N_observed × p²  ≥  400
```

rather than `N_observed ≥ 400`. This is what would make T3's precision measurement
load-bearing rather than decorative — which is the hole D-2 names and then answers
only with a population check that cannot see precision.

**Add it, drop it, or park it as a stated limitation in GATE-1.** Not folded in
without a decision.

## 6. Signature

```
Threshold set by:  ______________________    Date: __________

Histogram not computed before this commit:   git SHA ______________
```
