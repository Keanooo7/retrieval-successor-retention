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

## ⚠️ Source status

**The [P7] paper (Sukhbaatar et al. 2021) is not in `~/research-corpus`.** A search found
only two handoff notes that mention Expire-Span, and no source. `RESEARCH-CONTEXT.md` §2
marks [P7] "✅ checked", but the checked text is not on disk anywhere this build could
read. Every formula below that comes from the paper rather than from the spec is marked
**UNVERIFIED**: it is the paper as recalled, and it must be checked against the primary
source before the referendum is pre-registered.

| Detail | Source | Status |
|---|---|---|
| Span `e_i = L·σ(w·h_i + b)` | paper, as recalled | UNVERIFIED |
| Remaining span `r_i(t) = e_i − (t − i)` | paper, as recalled | UNVERIFIED |
| Soft mask `m = clamp(1 + r/R, 0, 1)`, ramp length `R` | paper as recalled; `R` named in RESEARCH-CONTEXT B-3 | form UNVERIFIED |
| Attention `a_i ∝ m_i·exp(s_i)` (softmax, reweight, renormalise) | paper, as recalled | UNVERIFIED |
| Auxiliary span loss `α·Σ e_i` | `α` named in B-3; form as recalled | form and normalisation UNVERIFIED |
| "Requires structured dropout" | RESEARCH-CONTEXT B-3 | **the requirement is sourced; its form is UNVERIFIED** |
| Span init; `w` init | not recalled | chosen here, see q8 |
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
therefore has to make choices [P7] never faced, and each is listed below.

### What the build implements

```
e_i    = L · sigmoid(w · h_i + b)          h_i = the slot's gestalt (unit norm)
r_i(t) = e_i − (t − written_at_i)
m_i(t) = clamp(1 + r_i(t)/R, 0, 1) · live_i · keep_i(t)     keep: structured dropout
a_i    ∝ m_i · exp(s_i)                    every C block, every head, every query
loss  += α · Σ_{i admitted at t−1} e_i / B
evict  = argmin_live r_i(t)                only when memory is full
```

- The span is **recomputed each step** from the occupant's gestalt, with age taken from
  `written_at`. Reset on admission (gauntlet 0.4) therefore holds **by construction**:
  there is no per-slot cache to go stale. (`run_policy_loop` passes `on_write` the victim
  index, which is not where the newcomer lands after `write_at`'s compaction, and is `0`
  for a row that was not full. A cache keyed on it would be wrong both ways.)
- `w = 0` and `b = logit(init_span_fraction)` at construction. **No global RNG draw**
  (the E0b trap), and every gestalt starts with the same span, so **Expire-Span evicts
  exactly FIFO at initialisation** (`test_at_zero_weight_init_expire_span_evicts_exactly_fifo`).
- A policy with no `memory_weight` runs the model with exactly the old arguments: E0b and
  `test_fidelity.py` pass unchanged (44 passed, 0 skipped).

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

Pinned by tests. `test_model_gradients_are_identical_under_none_and_predictor` is bit
equality, with `α > 0`. `test_through_gestalt_changes_the_model_gradients` and
`test_predictor_path_trains_the_predictor_but_not_the_gestalt` pin the other two.

### q2 — The tuning budget (B-3). NOT DECIDED HERE

The spec runs Expire-Span **once, untuned**, against RSR's 9-config `γ×β` grid plus a `ν`
sweep. *"An untuned Expire-Span losing is uninformative; a tuned one winning ends the
project."* B-3 names the fix, an equal-budget protocol, and leaves it open. The protocol
needs all of the following, **in a PREREG committed before any run**:

1. **The grid.** Which of `L`, `R`, `α`, dropout rate, init fraction are tuned, and over
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

### q3 — The span loss: form and normalisation. UNVERIFIED

Implemented as `α · Σ e_i` over gestalts admitted at `t − 1`, divided by batch rows, and
added to that step's objective by the loop, so each gestalt pays once, at the first step
it can be read. The alternative is charging every live slot every step, which penalises
span × lifetime and is a stronger objective (a mutation guards the difference).
[P7]'s exact normalisation, per token or per memory and with or without a `/L`, is not
recalled reliably. Does the port charge per admission, and at what scale relative to the
per-sentence-step LM loss?

### q4 — Structured dropout: form and rate. Requirement sourced, form UNVERIFIED

Implemented as a whole slot's mask zeroed with probability `p`, one draw per `(row, slot)`
per step, shared by every layer, head and query. It runs in training only, gated by the
**model's** mode rather than the policy's. There is no `1/(1−p)` rescale, because
renormalising the attention makes any rescale cancel. It uses a private CPU generator
seeded by `cfg.seed`, whose state round-trips through checkpoints, and the draw moves to
the memory's device, so the path is the same on CPU and CUDA. Does this match [P7]'s
structured dropout, and what is `p`?

### q5 — Expired slots that capacity has not yet forced out

They are **kept**: they receive zero attention but hold their rank, and so their place in
`P^(sent)`. Freeing them early would compact the prefix and shift every younger slot's
rank. That is the ADR-0006 confound, now triggered by expiry rather than by eviction. It
would also let Expire-Span run at an effective capacity below `M`. Keep or free?

### q6 — The eviction rule under a hard capacity

[P7] has no capacity. The port evicts `argmin r_i` among live slots, and ties go to the
lowest index (the oldest). The eviction uses the undropped span: dropout affects
attention, never who is evicted. Alternatives are the lowest current mask `m_i` (which
differs only once slots saturate at 0 or 1) or the shortest `e_i` regardless of age.
Is least-remaining the port?

### q7 — The span's input

`h_i` is the **gestalt as stored**. It is unit norm, without the positional term, which
only the keys receive. [P7] predicts from a token's hidden state. In TG the gestalt is
the only per-memory representation, so the choice is forced, but it should be stated.

### q8 — Initialisation

`w = 0`, `b = logit(init_span_fraction)`. This is chosen for E0b (no RNG draw) and
because it makes Expire-Span start as FIFO and move only as far as training moves it.
[P7]'s init is not recalled. The fraction is a required field.

### q9 — The optimizer group for `w, b` (μP, §4.3)

`build_param_groups` does not know Expire-Span's parameters. Expire-Span is **not wired
into `train()` or `--policy`** (`POLICIES` is unchanged), deliberately, until q1, q2 and
q9 are ruled. `w` is a `d`-vector read against a unit-norm input. Which μP group does it
belong to, and at what learning rate?

### q10 — Registration

`L`, `R` (both in sentence steps), `α`, `p` and the init fraction are per scope, like `M`
and `S`. Should they enter `src/rsr/constants.py`, for example as `CONDITIONAL` and
logged by the B-3 tuning run, so that the D-1 guard covers them as it covers `ν`, `β`
and `γ`? Until then the config has no defaults, which is the same guard in weaker form.

## Consequences

- Release condition 3's **"implemented"** half is met by the build, pending review. Its
  **"run on synthetic at the week-4 gate"** half needs q1, q2 and q9 ruled, a PREREG, and
  the smoke replaced by a real run.
- Nothing about the referendum's outcome is claimed. The build's smoke is one training
  step, and it is a wiring check, not a result.
