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
> n_raw × p_LCB²  ≥  150   in EVERY ONE of   (40, 48],  (48, 56],  (56, 64]
>              AND
> n_raw × p_LCB²  ≥  600   pooled across     (40, 64]
> ```
>
> drawn from **≥ 30 distinct documents**.

`n_raw` is the raw count of detected reintroductions in the bin. **`p_LCB` is the
one-sided 95% Wilson score lower bound** on the coref system's precision, measured
on 100 hand-annotated reintroductions (T3) — **not the point estimate**. §1.3 gives
the reason and the arithmetic.

**These are raw-event floors.** They are not effective sample sizes: events cluster
within books, and §1.2 states the design effect and the honest minimum detectable
effect that follows from it.

### Why `p²` and not `p`

False-positive reintroductions carry no true effect, so they dilute the observed
effect by `p` — and the required `n` scales as `1/δ²`. Both independent derivations
agreed on this and it is settled: **the exponent is 2.** This is what makes T3's
precision measurement load-bearing rather than decorative, which is the hole D-2
names and then answers only with a population check that cannot see precision.

The exponent is also **why the quantity that enters is a lower bound and not a point
estimate** (§1.3): squaring roughly doubles the sampling error of `p`, so a gate
built on `p̂` is systematically more permissive than it looks.

🔴 **An unmeasured `p` is exit 3 — "did not run". It is not a pass.** In ML a run
that produced no metric looks exactly like a run that produced a bad one, and `3`
must never collapse into `0`. There is no default `p`, and `p = 1` is not a
conservative assumption — it is the most permissive one available.

### Why the per-bin constraint binds, and the pooled figure is subsumed

Two derivations produced two numbers. They are reconciled here, and the
reconciliation is **D-G**, not a compromise:

| | Derivation | Floor | MDE, uncorrected | **MDE with DEFF** |
|---|---|---|---|---|
| Per-bin | 150 per bin over three bins | binds | 2.0% | **2.76%** |
| Pooled (Studio) | 400 | subsumed | 1.2% | 1.66% |
| Pooled (adopted) | **600** | binds jointly | 1.0% | **1.38%** |

**The right-hand column is the one to quote.** See §1.2 — the uncorrected figures
are optimistic by ~38%.

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

### 1.2 The design effect is derived, and now it is applied

The retained derivation below carries `DEFF = 1 + (m̄−1)·ICC` with `m̄ ≈ 10` events
per book and `ICC ≈ 0.10`, giving **1.9** — and the D-G floors were then stated
without it. That made the advertised MDEs optimistic by `√1.9 = 1.378`, i.e. ~38%.

```
per-bin   n_raw 150  ->  effective  78.9   MDE 2.0% -> 2.76%
pooled    n_raw 600  ->  effective 315.8   MDE 1.0% -> 1.38%
```

**The floors do not move. The stated MDEs do.** D-G's own rule is that the event
threshold does not move retroactively; correcting an arithmetic statement *about*
the threshold is not the same act, and leaving it uncorrected would mean the
document advertises power it does not have.

150 and 600 are therefore **raw-event floors**, and the effective counts are **~79
per bin** and **~316 pooled**. Both figures go in the writeup; quoting the raw count
as though it were the effective one is the error this section exists to prevent.

> 🔴 **What 2.76% per bin actually buys, stated plainly.** [P2]'s headline effect is
> 2–4%. A per-bin MDE of 2.76% detects the **upper** part of that band and would
> **miss an effect at the bottom of it**. That is not a reason to raise the floor,
> and it is a reason to be precise about what each half of the gate does:
>
> * the **per-bin** floor exists to stop an **empty top bin** — `(56, 64]` is where
>   E3's claim lives, and a pooled-only gate cannot see it;
> * the **pooled** floor at **1.38%** carries the power for the headline effect and
>   covers the whole of [P2]'s band.
>
> Reporting the per-bin figure as though it powered the main effect would be the
> same conflation §0 exists to prevent, one level down.

### 1.3 `p` enters as a lower confidence bound, not a point estimate

`p` is measured on **100 annotations** and enters **squared**, so its sampling error
is roughly doubled on the way in. Using the point estimate makes the gate pass
about half the time it should not, precisely at the margin where it matters.

**Definition: `p_LCB` is the one-sided 95% Wilson score lower bound at `n = 100`.**
One-sided because the gate is one-directional — the question is only whether
precision is at least high enough. Wilson rather than the normal approximation
because the latter misbehaves as `p̂` approaches 1, which is the region T3 is most
likely to land in.

| `p̂` | `p_LCB` | `p̂²` | `p_LCB²` | raw needed @ point | **raw needed @ LCB** | penalty |
|---|---|---|---|---|---|---|
| 0.95 | 0.901 | 0.902 | 0.812 | 167 | **185** | 1.11× |
| 0.90 | 0.840 | 0.810 | 0.705 | 186 | **213** | 1.15× |
| 0.85 | 0.782 | 0.722 | 0.612 | 208 | **246** | 1.18× |
| 0.80 | 0.727 | 0.640 | 0.528 | 235 | **285** | 1.21× |
| 0.70 | 0.620 | 0.490 | 0.385 | 307 | **391** | 1.27× |

At the likely operating point `p̂ = 0.90`: `p_LCB = 0.840`, so the per-bin floor
needs **213 raw events**, not 186.

*(A two-sided 95% bound would be stricter still — `p_LCB = 0.826`, 221 raw events at
`p̂ = 0.90`. One-sided is the honest match to a one-directional gate; the two-sided
figures are recorded here so the choice is visible rather than assumed.)*

**Compounding with §1.2:** at `p̂ = 0.90`, clearing the per-bin floor takes 213 raw
events, which is ~112 effective after the design effect. Both corrections point the
same way and neither cancels the other.

### `σ_d = 0.30` is an assumption, and it is labelled as one

The MDEs above assume a paired difference SD of 0.30 nats/token. That is an
assumption, not a measurement, and it is the only soft number in this section.

**When a real paired run produces a `σ_d`, recompute the MDE and report it beside
the headline.** Both the E0e FIFO run and the E0d subsample can produce one before
E3.

**And it is the conservative direction, which is worth one line.** With `σ = 0.40`,
`σ_d = σ√(2(1−ρ))` inverts to `ρ = 0.719` — so `σ_d = 0.30` implicitly assumes an
inter-arm correlation of **~0.72, not the 0.90** the retained derivation below
assumed. A paired design with less correlation between arms is a *weaker* design,
so the assumption costs power rather than borrowing it. Nothing here depends on
0.72 being right; it is stated so the two numbers in this document are not read as
independent.

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
   cheap. **Size it against `p_LCB`, not `p̂`** (§1.3): at `p̂ = 0.90` the per-bin
   floor needs 213 raw events, not 186, and enlarging to the point-estimate figure
   would leave the gate failing after the work was done.
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

This section was open. **D-G closes it: the precision term is folded into §1's
gate**, not parked as a limitation — and it enters as `p_LCB²`, a one-sided 95%
Wilson lower bound, per §1.3.

The errata that got it there, recorded because the near-miss is instructive: the
first derivation wrote the attenuation as `1/p`. The reintroduction events a coref
system reports are diluted by precision `p` in the *effect*, and required `n` scales
as `1/δ²`, so the correct exponent is **2**. Both sessions' derivations agreed on
this once stated; they disagreed only on the resulting count, which is what D-G
reconciles above.

Consequence: **T3 is on the critical path for E0i.** Without a measured `p` the gate
cannot be evaluated at all — exit 3, not exit 0.

## 6. Signature

Everything above is final and reconciled (D-G, plus the two corrections in §1.2 and
§1.3, both of which make the gate stricter). This is a signature, not a decision.

🔴 **Unsigned. It must be signed by a person, and that person is not a model.**

The only thing this signature does is record that *someone committed to a number
before seeing the data*. A name written here by an agent would make the document
look committed-to while binding nobody, and a later reader — or a committee — would
read it as Brendan's commitment. That is the failure, and it is the same one as
writing an advisor review in an advisor's voice. The blank is the point.

```
Threshold set by:  Brendan Keane                Date: 2026-09-17

Histogram not computed before this commit:   git SHA 990f6792fe912a800e6ce79220e728c05879fd94
```

### To sign

```bash
/opt/homebrew/bin/git -C ~/retrieval-successor-retention rev-parse HEAD
```

Put that SHA on the second line, a name and today's date on the first, and commit.
**The commit is the proof of ordering** — that is the whole mechanism, and it is why
the signature and the SHA belong in the same commit rather than in a file that
existed beforehand.

### Verified at the time of writing: no histogram exists

Checked, with the commands, so the claim is not an assertion:

```
experiments/e0i/RESULTS.md    **Status: NOT RUN.**
experiments/e0i/run.py        raise NotImplementedError("E0I is not implemented yet.")
ls data runs outputs          No such file or directory  (x3)
import fastcoref              ModuleNotFoundError
ls measurements               No such file or directory
```

No corpus, no coref tooling, no ledger, no run. There is nothing from which a gap
histogram could have been computed.

### Signed does not mean passed

**E0i remains exit 3 — "did not run" — until T3 measures `p`.** A signed
pre-registration with no measured precision cannot be evaluated at all: `p_LCB` is
undefined, and there is no default. `p = 1` is the most permissive assumption
available, not a conservative one. `GATE-1.md` reports it as exit 3 and must
continue to until T3 lands.
