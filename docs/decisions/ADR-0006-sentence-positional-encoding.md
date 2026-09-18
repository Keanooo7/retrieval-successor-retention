# ADR-0006 — `P^(sent)` stays rank-indexed; the confound it creates is measured

- **Status:** accepted
- **Date:** 2026-09-17
- **Decision:** D-D of `~/rsr-gauntlet-2026-09-17.md`
- **Relates to:** [P2] §2.2; spec §3.2.2, §3.4, §3.7, §7.1; ADR-0001 (the code is
  the reference), correction 15
- **Written before:** any line of `RSRPolicy`. That ordering is the point — the
  instrumentation below is a constraint on how the policy is built, not a report
  on how it behaved.

## Context

### What TG actually does

[P2] §2.2 builds the memory cross-attention as

```
K_M = s_{t−Mt}, …, s_{t−1}  +  P^(sent)_{1:Mt}
V_M = s_{t−Mt}, …, s_{t−1}                      # no positional term
```

over a memory *"ordered from oldest to most recent."* Three properties, each
load-bearing below:

1. `P^(sent)` is **sinusoidal** and fixed, not learned.
2. It is added to the **keys only**. `V_M` gets none.
3. Its index is the slot's **position in the memory ordering** — its rank — not
   the slot's age and not its absolute write step.

Under stock TG these last two distinctions are invisible, because TG evicts the
oldest slot. Rank order and age order are the same order, always, so `P^(sent)`
can be read as "recency" with no loss.

### What RSR breaks

RSR evicts by predicted future demand, so it can evict a **middle** slot. Every
slot behind the victim moves up one rank. Its key — and therefore its
cross-attention logit against every query — changes, although the slot itself did
not change and no time passed for it.

**A slot's positional encoding becomes a function of which other slots the policy
killed.** That is a second difference between the RSR arm and the FIFO arm, on top
of the eviction rule. E3 attributes its delta to retention; this says part of the
delta is bookkeeping.

### Why E0b cannot catch it

This is the part that decides the ADR.

E0b exists to guarantee that the FIFO and RSR arms differ *only* in the eviction
rule (§3.7, `tests/test_reduction.py`). It cannot see this confound, and not by
accident of implementation: **under the §3.7 reduction the policy is FIFO.**
`ψ̂ ≡ −a_i` makes the argmin select the oldest slot, so no middle slot is ever
evicted, so no rank ever shifts, so rank and age-order coincide exactly as they do
in stock TG. E0b passes, correctly, and says nothing.

**The confound is structurally invisible in the reduction and present in every arm
the project cares about.** A gate that is green precisely where the hazard is
absent is not evidence about the hazard.

## Decision

**Keep `P^(sent)` rank-indexed.** Then make the confound observable rather than
arguable, with three pieces of instrumentation that ship *with* the policy.

Rank-indexing is kept for one reason and it is sufficient: **it is what the
released TG code does, and ADR-0001/D-A make the code the reference by
construction.** `tests/test_fidelity.py` compares this repo's TG against tensors
extracted from that code. An absolute-age index would be a silent transcription
divergence — the fidelity harness would redden, and the honest fix would be to
change RSR back, having spent the fixture budget to learn it.

Changing the indexing would also be a change to TG, and §3.1 is explicit that the
base model is unmodified: *"the entire contribution of this document is replacing
'the oldest entry is removed.'"* Re-indexing the keys is a second modification,
and it would be one made on the basis of an argument rather than a measurement.

### 1. Per-eviction rank-shift logging — mandatory, every RSR run

At every eviction, record how many live slots had their rank changed by it:

```
shift_count(t) = |{ j : j is live, rank_after(j) ≠ rank_before(j) }|
```

Under FIFO this is identically 0: the victim is at rank 1 and everything behind it
keeps its place in the ordering. Under RSR it is the number of slots behind the
victim. **Report the distribution, not the mean** — a policy that evicts the oldest
slot 95% of the time and the newest 5% of the time has a small mean and a real
confound.

Two derived quantities go in the heartbeat (Phase 3.8):

- the fraction of evictions with `shift_count > 0` — how often the confound fires;
- the mean rank displacement per surviving slot per stream — how far a slot's key
  drifts over its lifetime for reasons unrelated to itself.

This is the cheap half of the instrumentation and it is not optional: §3.4 already
requires per-eviction decision attribution from the first policy run, and this is
one more field on a record that must exist anyway.

### 2. A `P^(sent)`-ablated arm in E1 — bounds the confound's size

One additional E1 arm with `P^(sent)` **zeroed** (keys carry no positional term),
run for both FIFO and RSR.

It answers the only question that matters quantitatively: *how much of the RSR−FIFO
delta survives when the confound is removed entirely?* The ablation is not the
headline configuration — removing sentence position from the memory keys is a
strictly weaker model and will very likely cost loss in both arms. The comparison
is the **difference of differences**:

```
Δ_confounded  = loss(FIFO)         − loss(RSR)
Δ_ablated     = loss(FIFO, no P)   − loss(RSR, no P)
```

If `Δ_ablated ≈ Δ_confounded`, the rank shifting contributes nothing and the
headline stands as stated. If `Δ_ablated` is substantially smaller, the gap is an
upper bound on what re-indexing bought, and it is reported as such **in the
headline sentence**, not in a limitations paragraph.

This is a bound, not a decomposition. Zeroing `P^(sent)` removes the confound and
also removes the legitimate positional signal; it cannot separate them. Stated so
that nobody later reads it as an effect size.

### 3. An absolute-age-indexed variant behind a config switch — default OFF

```
sent_pos_index: Literal["rank", "absolute_age"] = "rank"
```

`"absolute_age"` indexes `P^(sent)` by `t − written_at(i)` clipped to `A_max`, so a
slot's key depends only on the slot. It is a research variant, not a fix: it
changes TG, it breaks fidelity against the pinned reference, and it must be
reported as a modified base model wherever it is used.

**It must be `"rank"` in `RSRConfig.reduction_to_tg()`, and a test asserts that.**
With `"absolute_age"` the reduction is no longer bit-exact TG and E0b would be
testing a model the fidelity harness never saw. This is the §3.7 off-switch
discipline applied to a term that is not in the eviction score — the rule is that
*every* configurable divergence from TG has a documented off-switch set in the
reduction, and this one is a divergence in the base model, which is worse.

## Two consequences that must appear in the writeup

### (a) It lowers the E0h collinearity prior — and is independent support for D-C

The cross-attention logit for slot `i` is

```
q · (s_i + P^(sent)_{rank(i)})  =  q·s_i  +  q·P^(sent)_{rank(i)}
```

The second term is a function of position that **ψ̂ structurally cannot contain**:
§3.2.2 excludes age from `ψ̂`, and `ψ̂`'s only inputs are `s_i` and `c_t`.

So attention — and therefore `r_i`, and therefore ψ̂'s regression target — carries a
positional component ψ̂ has no way to represent. E0h asks whether `ψ̂` and age are
collinear. **Some of the collinearity the estimator could exhibit is bounded away
from it by the architecture**: ψ̂ can only fit the positional part of `r_i` to the
extent that position happens to correlate with gestalt content. The prior on
observing high collinearity should be **lower** than it would be if ψ̂ could see
position, and E0h's threshold should be read against that.

This is independent support for **D-C** (`c_t` is the current sentence gestalt).
A running-context `c_t` accumulates over the stream and is therefore weakly
time-indexed: it would give ψ̂ an indirect positional channel, partially restoring
exactly the collinearity this argument removes. The gestalt reading keeps both
arguments to the bilinear form unit-norm *and* keeps ψ̂ positionally blind. The two
decisions were made on different grounds and agree.

### (b) §7.1's vacuity gate must not convict the estimator of the architecture's own contamination

`r_i` is built from `α`, and `α` is a softmax over logits that already contain
`P^(sent)`. **The retention target is therefore itself partly a function of slot
position.** A perfect estimator of a positionally-contaminated target is
positionally contaminated.

§7.1 gates on whether the eviction *score* is a function of age (partial
correlation, ρ < 0.7, against an age-only head, and "RSR must beat LRU"). Written
into §7.1's report, verbatim:

> The retention target `r_i` inherits a positional component from `P^(sent)`
> (ADR-0006). A non-zero age correlation in `ψ̂` is therefore expected at some
> level and is not by itself evidence that the estimator collapsed onto recency.
> The gate is on the *magnitude* and on the behavioural tests — the age-only-head
> comparison and RSR-vs-LRU — which compare estimators fitting **the same**
> contaminated target and so difference the contamination out.

The two behavioural tests survive this argument intact; the raw partial correlation
is the one that needs the caveat. That is why §7.1 lists three tests and gates on
the full score rather than on `ψ̂` alone.

**The `P^(sent)`-ablated arm (2) is also the clean read of §7.1's ρ**, since in that
arm the target carries no positional term at all. Report ρ for both arms.

## Status of the alternatives

**Absolute-age indexing as the default** — rejected above: a change to TG, made
from an argument, that breaks fidelity against the reference the project chose as
its ground truth (D-A). Kept as a switch so the question stays answerable.

**Re-encoding keys only on eviction steps, or freezing a slot's `P` at write time**
— both are absolute-age indexing with extra steps, and `P` frozen at write time is
exactly `P^(sent)_{rank at write}`, which drifts from every other slot's convention
as the memory turns over. Not pursued.

**Declaring the confound negligible because RSR will mostly evict old slots** — that
is instrumentation (1) stated as an assumption instead of a measurement. It may
well turn out true. It is not assumable in advance, and if it is true the logging
costs one integer per eviction to prove it.

## Consequences for the build

- `MemoryState` must expose a stable ordering and the policy protocol must report
  the victim's rank, so `shift_count` is computable without the policy reaching
  into the memory.
- The eviction record grows a `rank_shift` field (§3.4's decision attribution).
- `RSRConfig` grows `sent_pos_index`, defaulting to `"rank"`, pinned to `"rank"` in
  `reduction_to_tg()`, and covered by the off-switch enumeration test.
- E1's arm list grows one arm (`P^(sent)` zeroed) — priced into E1, not into E3.
- `docs/code-vs-paper.md` records that rank-indexing is the **code's** behaviour and
  that the paper's `P^(sent)_{1:Mt}` over an oldest-to-newest ordering is the only
  statement of it (D-A).
