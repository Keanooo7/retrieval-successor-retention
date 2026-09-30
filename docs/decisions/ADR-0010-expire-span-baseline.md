# ADR-0010 — Expire-Span [P7] ported onto TG: the gradient path, the tuning budget, and every choice the spec leaves open

- **Status:** **proposed** — pending Brendan's sign-off. **Not accepted.** The code that
  lands with this ADR (`feat/expire-span`) implements every option below behind
  **required** config fields with no defaults, so nothing here is decided by the code.
- **Date:** 2026-09-29
- **Decision:** the 2026-09-29 build brief, "implement the Expire-Span baseline [P7] for
  real" — the BUILD, not the referendum run.
- **Base:** `main` at `f7a6b10`. Code: `src/rsr/baselines/expire_span.py`,
  `CrossAttention`'s `mem_weight` in `src/rsr/model/tg/model.py`, the `memory_weight`
  hook in `src/rsr/model/tg/policy_loop.py`, `tests/test_expire_span.py`.
- **Relates to:** spec §5.4, §2 ("a learned *static* span"), §7 falsifier 6, defect D-4;
  `docs/release-conditions.md` condition 3 (week-4 gate, **no demotion path**);
  `docs/RESEARCH-CONTEXT.md` §4.8 and **B-3**; ADR-0006 (rank-indexed `P^(sent)`);
  ADR-0009 (the ψ̂ learning path, proposed, on `docs/adr-learning-path`).

## Source status

**Revised 2026-09-29.** The [P7] paper is now in the corpus:
`~/research-corpus/sources/memory-retention/sukhbaatar-2021-expire-span.md` (arXiv:2105.06548
**v2**, ICML 2021; PDF sha256 `0d49299b…0f3c`). `P7:L<n>` below is a line in that file. The
line-by-line verification of this branch at `ac46567` is
`~/Documents/RSR-2026-09-29-day/reports/P7-verification.md`. The first draft of this ADR was
written before the source existed on disk, and it marked every paper detail UNVERIFIED. The
table below replaces that one.

| Detail | [P7] | Status in this port |
|---|---|---|
| Span `e_i = L·σ(w·h_i + b)` | Eq. 3, P7:L259–265 | **MATCH**, per layer (see below) |
| Remaining span `r_ti = e_i − (t − i)` | P7:L267–268 | **MATCH**. Age is in sentence steps, not tokens |
| Soft mask `m = max(0, min(1, 1 + r/R))` | Eq. 5, P7:L284–286 | **MATCH** |
| Attention `a′_ti = m_ti a_ti / Σ_j m_tj a_tj` | Eq. 4, P7:L272–279 | **MATCH**. An all-zero row outputs exactly 0, which the paper does not specify |
| Heads share one span per layer | P7:L355–356 | **MATCH** |
| **Spans per layer** ("done independently for each layer") | P7:L81–82, P7:L355 | **FIXED 2026-09-29.** Verification 4b was a MISMATCH: one span was shared by every layer |
| Aux loss `L_task + α Σ_i e_i / T` | Eq. 7, P7:L330–335 | **The form follows §4.2's timing; the scale against Eq. 7 is open (q3).** Each memory is charged `α·e` at each of the ~`R` steps it spends on the ramp, whereas Eq. 7 charges `Σ_i e_i` once, so `α`'s effective scale is coupled to `R`. Eq. 7's `/T` is absorbed because the loop's LM term is a sum over `T` steps. Batch averaging is not in the paper. The loss is charged in training only and only for rows still inside their stream (review MAJOR-1 and MINOR-1) |
| **Aux loss timing**: charged while `0 < m < 1` | §4.2 "Loss Computation", P7:L368–382 | **FIXED 2026-09-29.** Verification 5c was a MISMATCH: the charge was once at admission, which the paper reports "empirically results in poor performance" |
| **Structured dropout**: per batch `l ∼ U(0, L)`, `a_ti = 0` for `t − i > l`, training only | §4.2 "Regularization", P7:L387–392; App. A.2, P7:L1055–1064 | **FIXED 2026-09-29.** Verification 6 was a MISMATCH: per-slot Bernoulli(`p`). It is now one cutoff per sentence step, and the config field is an on/off switch |
| Bias `b` initialised negative | App. A.1, P7:L1013–1015 | **MATCH**, now enforced: `init_span_fraction < 0.5` is validated |
| `w = 0` init | not in paper (the official code does the same) | port choice (q8) |
| Large-`L` variant, Eq. 8 | P7:L398–404 | not implemented. The paper makes it conditional on very large `L` |
| Gradient into `h_i` (detach or not) | **not in paper** | q1 |
| Hard capacity and an eviction rule | **not in paper**; deletion is by expiry only (P7:L289–291, L365) | port choice (q5, q6) |
| Static span, a function of the memory alone | spec §2 | sourced |
| Trained by the LM loss directly; no demotion path | spec §5.4, D-4, falsifier 6 | sourced |

## Context

§5.4 makes Expire-Span **the referendum, not a comparison**: soft, differentiable expiry
trains retention by the LM loss directly, and if it matches or beats RSR on synthetic
(falsifier 6) the reward proxy, the MC/TD machinery, the shadow buffer and the warmup
were working around a non-differentiability that did not need to exist. That verdict
**invalidates the design, not the result.**

[P7] was written for a token-level transformer with an unbounded memory that simply stops
attending to expired entries. TG has a hard `M`-slot memory of sentence gestalts. A port
therefore has to make choices [P7] never faced, and each is listed below. **Where the paper
speaks, the port matches it**. This is a PM ruling under delegation on 2026-09-29: the paper
is the authority for its own baseline. It is recorded in the day digest
`~/Documents/RSR-2026-09-29-day/DIGEST.md`: cycle 26 for the paper-authority ruling, and
cycle 28 for the α ratio (no literal `/T`). **That file is outside this repository.**

### What the build implements

```
e_li    = L · sigmoid(w_l · h_i + b_l)      one (w_l, b_l) per cross-attention layer l
r_li(t) = e_li − (t − written_at_i)
m_li(t) = clamp(1 + r_li(t)/R, 0, 1) · live_i · [age_i ≤ ℓ_t if training & dropout on]
                                            ℓ_t ∼ U(0, L), one per sentence step
a_i     ∝ m_li · a_i  (renormalised)        layer l's C block, every head, every query
loss   += α · Σ_{l,i : 0 < m_li < 1} e_li / B
evict   = argmin_live max_l r_li(t)         only when memory is full (q6)
```

- The span is **recomputed each step** from the occupant's gestalt, with age taken from
  `written_at`. Reset on admission (gauntlet 0.4) therefore holds **by construction**:
  there is no per-slot cache to go stale, and `on_write` is a no-op under either loop
  contract. On `night/2026-09-30`, `fix/on-write-slot-index` makes `run_policy_loop`
  pass `on_write` the newcomer's post-write slot and gives `MemoryState` a `row`. This
  branch predates that merge; before it the loop passed the victim index, which would have
  been wrong for any cache keyed on it.
- `w = 0` and `b = logit(init_span_fraction) < 0` at construction. **No global RNG draw**
  (the E0b trap). Every gestalt starts with the same span in every layer, so **Expire-Span
  still evicts exactly FIFO at initialisation**
  (`test_at_zero_weight_init_expire_span_evicts_exactly_fifo`).
- A policy with no `memory_weight` runs the model with exactly the old arguments, so E0b
  and `test_fidelity.py` pass unchanged. The model refuses a mask whose layer count is not
  the number of C blocks.

### Paper-faithfulness fixes (2026-09-29, after the source landed)

1. **4b, per-layer spans.** `weight` is `[n_C, d]` and `bias` is `[n_C]`. The mask is
   `[n_C, B, M]`, and the l-th C block reads layer `l`. Pinned by
   `test_each_cross_layer_gets_its_own_span_and_mask` and
   `test_each_c_block_reads_its_own_layer_of_the_mask`.
2. **5c, aux-loss timing.** `α · e_li` is charged at every step where `0 < m_li < 1` (the
   pre-dropout mask). The ramp condition selects which spans are charged but is not
   differentiated. Pinned by `test_the_span_loss_charges_memories_on_the_ramp_only` and
   `test_the_span_loss_carries_gradient_only_through_the_span`. A mutation restores the
   charge-at-admission rule.
3. **6, structured dropout.** One `ℓ ∼ U(0, L)` per sentence step, drawn from the private CPU
   generator and shared by every row, layer and head. Every memory with age `> ℓ` gets
   weight 0, in training only (the **model's** mode). The config field is now the switch
   `structured_dropout: bool`. Pinned by
   `test_structured_dropout_drops_every_memory_older_than_one_cutoff`.
4. **7a, negative bias.** `init_span_fraction ∈ (0, 0.5)` is validated.

## Decisions for Brendan

### q1 — The gradient path. 🔴 NOT DECIDED HERE

`ExpireSpanConfig.grad_path` is required and has three values:

| Value | `w, b` get LM-loss gradient | Span gradient reaches gestalt → transformer / `W_sent` | `α` loss reaches transformer |
|---|---|---|---|
| `none` | no (mask built under `no_grad`) | no | no |
| `predictor` | **yes** | no (gestalt detached at the predictor input) | no |
| `through_gestalt` | **yes** | **yes** | **yes** |

All three put the mask in the forward pass, so in all three the transformer's own LM
gradients are *shaped by* the spans. That is the same kind of effect any eviction rule has
on the forward, and it is not a gradient path from the retention objective.

Considerations, not a recommendation:

- **CLAUDE.md's rule is about RSR.** "Do not backpropagate the retention loss into the
  transformer or `W_sent`. Only `φ` receives gradient." The rule governs RSR's `φ`.
  Whether a *baseline* may do otherwise is exactly this question. `through_gestalt` sends
  both the LM-via-span gradient and the `α` span loss into the transformer.
- **Falsifier 6 asks whether gradients supersede RSR.** B-3's principle is that the
  referendum's unfairness must not run in the project's favour. The strongest
  Expire-Span is the informative one, which argues against handicapping it by a
  restriction RSR accepted for its own reasons.
- **E3's discipline is that arms "differ only in the eviction rule" (§3.7).**
  `through_gestalt` also changes the representations every arm is scored on. So a win
  could come from better gestalts rather than better retention, and E1 could not tell
  the two apart. `predictor` keeps the transformer's gradient sources identical in kind
  to RSR's.
- **`none` is an ablation, not a candidate.** It gives fixed spans with soft expiry, and
  it isolates "does learning the span matter at all".
- Running both `predictor` and `through_gestalt` doubles the referendum's budget under
  q2, and it has to be pre-registered as two arms, not as a choice made after the data.

**What the source says (context, not a decision).** [P7] never mentions detach or
stop-gradient, and never states whether `∂e_i/∂h_i` is propagated. Its only direct words
are that memories are forgotten "in a gradually differentiable way to retain end-to-end
training with backpropagation" (P7:L78–81), which is about the mask, and it names only `w`
and `b` as trainable (P7:L264). The official code (`facebookresearch/transformer-sequential`
@ `9650fc9d`) **is not the paper**. It detaches its whole hidden-state cache at every block
boundary (`trainer.py:89–95`), and it computes spans from the cache plus the current
block, without detaching them. So in that code the span gradient reaches hidden states and
the transformer for memories in the current block, and reaches only `w, b` for older
memories. That is a consequence of truncated BPTT, not of a predictor-side stop-gradient.
`predictor` is stricter than the code. `through_gestalt` is looser, because TG retains
the graph across the whole stream. The closest port of the *code* would be
`through_gestalt` truncated at a block boundary TG does not have. None of this settles q1
(verification report §2).

Pinned by tests. `test_model_gradients_are_identical_under_none_and_predictor` is bit
equality, with `α > 0`. `test_through_gestalt_changes_the_model_gradients` and
`test_predictor_path_trains_the_predictor_but_not_the_gestalt` pin the other two.

### q2 — The tuning budget (B-3). NOT DECIDED HERE

The spec runs Expire-Span **once, untuned**, against RSR's 9-config `γ×β` grid plus a `ν`
sweep. *"An untuned Expire-Span losing is uninformative; a tuned one winning ends the
project."* B-3 names the fix, an equal-budget protocol, and leaves it open. The protocol
needs all of the following, **in a PREREG committed before any run**:

1. **The grid.** Which of `L`, `R`, `α`, structured dropout on/off, and init fraction are tuned, and over
   which values. None of them is registered in `src/rsr/constants.py`, and the code
   supplies no default.
2. **"Equal".** Equal configs (9 + `|ν|`), equal optimizer steps × seeds, or equal
   wall-clock on this machine. They differ, because Expire-Span's per-step cost is not
   RSR's.
3. **Selection.** Which split and metric pick the winning config, and the rule that the
   test split is read once.
4. **Seeds**, and whether a spread criterion joins the mean.
5. **Whether `grad_path` (q1) is inside the budget or outside it.**
6. **Order.** ADR-0009's ψ̂ learning path is proposed, not built. RSR cannot yet be tuned
   at all, so an equal budget has nothing to be equal *to* until it lands.

### q3 — The span loss: timing FIXED to [P7]; the scale is still open

The timing now follows [P7] §4.2: `α · e_li` is charged at every step where
`0 < m_li < 1` (P7:L368–382), summed over layers and slots and divided by batch rows.
Two questions remain open:

- **Normalisation.** Eq. 7 is `L_task + α Σ_i e_i / T`. The loop's LM term is a *sum* of
  per-step means over the `T` sentence steps, so the `/T` is absorbed by multiplying both
  sides of Eq. 7 by `T`. That is verification 5b. A literal extra `/T` would weight `α`
  `T` times lower than the paper does.
- **Repeated charging.** A memory on the ramp is charged at each of the roughly `R` steps it
  spends there. The paper is silent on this. The official code (context only) divides by
  `R` ("each memory has R losses applied"). Should `α`'s scale absorb that factor, or should
  the port divide by `R`?

### q4 — Structured dropout: FIXED to [P7]'s form; whether to use it is open

The form now follows [P7] §4.2: one `ℓ ∼ U(0, L)` per sentence step, with every memory of
age `> ℓ` dropped in training (P7:L387–392). "Per batch" becomes per sentence step, because
TG's step is the unit that a batch of queries attends at. The paper has no rate parameter,
so the config field is a switch. The paper *proposes* this as regularisation and shows one
overfitting failure without it (App. A.2, P7:L1055–1064). RESEARCH-CONTEXT B-3's
"requires" overstates that (verification 6b), but correcting B-3 is Brendan's call.
**Open:** is the switch on for the referendum, or inside the q2 grid? The eviction
decision uses the undropped span.

### q5 — Expired slots that capacity has not yet forced out

**A deliberate deviation, left open for Brendan.** [P7] deletes a memory once it expires
("once a memory is expired, it can be permanently deleted", P7:L290–291). The port keeps
them: they receive zero attention but hold their rank, and so their place in
`P^(sent)`. Freeing them early would compact the prefix and shift every younger slot's
rank. That is the ADR-0006 confound, now triggered by expiry rather than by eviction. It
would also let Expire-Span run at an effective capacity below `M`. Keep or free?

### q6 — The eviction rule under a hard capacity

**[P7] has no capacity and no eviction rule** (verification §3). Memory size is an
emergent average that the aux loss minimises, and deletion is by expiry only (P7:L289–291,
L365). This is a **deliberate deviation, left open for Brendan.** The port evicts
`argmin_live max_l r_li`: the slot whose *longest* remaining span over layers is least. A
slot that any layer still reads is alive, by analogy with "a memory cannot be removed if it
is used by any of the heads" (P7:L353–355), which [P7] says of heads, not layers. Ties go to
the lowest index (the oldest). The undropped span decides. The alternatives are:
- min over layers;
- mean over layers;
- the lowest current mask;
- the shortest `e` regardless of age.

Is max-over-layers least-remaining the port? (Guarded by
`test_a_slot_one_layer_still_reads_is_not_the_victim` and a mutation to `amin`.)

### q7 — The span's input

`h_i` is the **gestalt as stored**. It is unit norm, without the positional term, which
only the keys receive. [P7] predicts from a token's hidden state. In TG the gestalt is
the only per-memory representation, so the choice is forced, but it should be stated.

### q8 — Initialisation

`w = 0`, `b = logit(init_span_fraction)`, with the fraction validated to lie in `(0, 0.5)`
so that `b < 0`, as [P7] App. A.1 requires ("we initialize the bias term b with a negative
value", P7:L1013–1015). The paper gives no value. The paper's reason is GPU memory; ours is
FIFO at initialisation and no RNG draw (E0b). `w = 0` is not in the paper; the official code
does the same. The fraction's value is q2's.

### q9 — The optimizer group for `w, b` (μP, §4.3)

`build_param_groups` does not know Expire-Span's parameters. Expire-Span is **not wired
into `train()` or `--policy`** (`POLICIES` is unchanged), deliberately, until q1, q2 and
q9 are ruled. `w` is a `d`-vector read against a unit-norm input. Which μP group does it
belong to, and at what learning rate?

### q10 — Registration

`L`, `R` (both in sentence steps), `α` and the init fraction are per scope, like `M`
and `S`. Should they enter `src/rsr/constants.py`, for example as `CONDITIONAL` and
logged by the B-3 tuning run, so that the D-1 guard covers them as it covers `ν`, `β`
and `γ`? Until then the config has no defaults, which is the same guard in weaker form.

## Consequences

- Whether release condition 3 is met is the owner's call. The facts: the policy is
  implemented on this branch; it is **not** wired into `train()` or `--policy`; it has not
  been run on synthetic beyond a one-step wiring smoke. The condition's "run on synthetic
  at the week-4 gate" half would additionally need q1, q2 and q9 ruled, a PREREG, and a
  real run. If this ADR and the build are accepted, the build would meet the condition's
  "implemented" half.
- Nothing about the referendum's outcome is claimed. The build's smoke is one training
  step, and it is a wiring check, not a result.
