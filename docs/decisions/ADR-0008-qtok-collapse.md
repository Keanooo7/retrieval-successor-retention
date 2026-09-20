# ADR-0008 — `Q_tok` collapses by **summing over real query tokens**; EOS-only is the live alternative

- **Status:** accepted
- **Date:** 2026-09-20
- **Decision:** dispatch `docs/lab-notes/dispatch-S0-02-capture-bridge.md`, "The
  decision this brief must make, and record"
- **Relates to:** spec §3.2.1; `docs/spec-corrections.md` corrections 17, 18, 20;
  ADR-0006; defects D-6, D-E, D-F
- **Written before:** any use of `r_i` as a training target. `reward.py` has had
  zero callers in `src/` since it was written, which is exactly why this could be
  decided in the open rather than discovered in a commit message.

## Context

§3.2.1 defines the retention target as

```
r_i(t) = Σ_{l,h} ‖ g_mem^(l) · α_{l,h,i} · W_O^(l,h) v_{l,h,i} ‖₂
```

(the `g_mem` factor is correction 17). `AttentionTrace.alpha` is therefore typed
`[L, H, M]` — one number per cross-attention layer, head and slot.

**The model does not produce that.** `CrossAttention.forward`
(`src/rsr/model/tg/model.py:335`) stores `self.last_attention` with shape
`[B, H, Q_tok, M]`: one attention distribution **per query token of the sentence**.
There are `Q_tok` of them and §3.2.1 wants one. The spec never says how the token
axis disappears, and the three obvious answers are not the same measurement.

This matters beyond tidiness: `r_i` is what E0d correlates against leave-one-out
Δloss to decide *whether `r_i` is a confound at all*. A collapse chosen silently
would be a free parameter inside the experiment that is supposed to validate the
quantity.

### Does the spec imply an answer?

**No, and it is worth being explicit that it does not.** §3.2.1 writes
`α_{l,h,i}` with three indices and never introduces a token index, so the notation
presupposes the collapse has already happened and says nothing about it. The
nearest thing to guidance is §3.2.1's own instruction to *"report the per-layer
profile once before collapsing to a scalar"* — an argument that aggregating over a
heterogeneous axis is a claim, not a formality. That argument applies one axis
over, which is why this ADR exists rather than a line in a commit message. It does
not, on its own, pick a value.

## Decision

**Sum `α` over the query positions where `mask != 0`.**

```python
alpha[l, h, i] = Σ_{q : mask[q] ≠ 0} att[l][h, q, i]
```

Implemented as `Q_TOK_COLLAPSE = "sum_over_real_query_tokens"` in
`src/rsr/model/tg/policy_loop.py`. Any other value raises rather than being
silently accepted.

### Why sum: for this axis the collapse is algebra, not a choice

`W_O^(l,h) v_{l,h,i}` **does not depend on `q`**. The values are read from the
memory (`v = self.value(mem_kv)`), the output projection is linear, and the memory
does not change within a step. So the total cross-attention increment that slot
`i` contributes to layer `l`'s residual stream, summed over the sentence, is

```
Σ_q increment_q[i]  =  Σ_h ( Σ_q α_{l,h,q,i} ) · W_O^(l,h) v_{l,h,i}
```

— exactly, with the summed `α` factored out. Summing over `q` is not one of
several ways to characterise slot `i`'s use; it **is** slot `i`'s contribution to
the sentence, which is what §3.2.1's norm is a norm of.

Verified numerically rather than asserted: reconstructing the block's cross output
from `att` and `v`, summing it over the real query positions, subtracting
`Q_real · attn_out_proj.bias`, and comparing against
`einsum("hm,hmd->d", alpha, wo_v)` agrees to **≤ 1.49 × 10⁻⁷ on values up to
8.23 × 10⁻¹** — float32 epsilon — on all six cross-attention layers. The check is
`tests/test_capture_bridge.py::test_the_collapse_reproduces_the_real_increment`;
every number in this ADR comes from that file's fixture (`D=32, H=2, M=5`,
`Q_tok = 10` with a 5-token PAD tail, so `Q_real = 5`, row 0, 4 of 5 slots live).

> 📌 **Provenance (added 2026-09-20d, task B1).** Every figure below is now
> produced by `experiments/s0-02/measure_qtok_collapse.py` into
> `runs/s0-02-capture-bridge/qtok_collapse.json` and recorded as
> `qtok_collapse.*` rows in `runs/s0-02-capture-bridge/ledger.json`. Until then
> they were **prose**: `grep -c "1.49\|2.98\|0.0383"` over that ledger returned
> `0`, and the three CI tests that pin this ADR assert *thresholds*
> (`atol=1e-5`, `atol=1e-6`, not-`allclose` at `atol=1e-3`), which is stronger
> in one way — they fail when the property breaks, not when a decimal moves —
> and no substitute in the way that matters. The producer **imports** the
> fixture from `tests/test_capture_bridge.py` rather than re-typing it, so the
> sentence above is checkable, and it re-derives every parameter and raises on
> drift. **It raised on its first run**, which is how the `Q_tok` and PAD-tail
> figures in that sentence were corrected: `TGConfig.L` is
> `1 + max_sentence_tokens + sentence_tail_len = 1 + 8 + 1 = 10`
> (`src/rsr/model/tg/config.py:128-130`), and `_sentence` masks `mask[0, 5:] = 0`.
> This ADR had read `max_sentence_tokens` as the query-axis length and derived
> the tail as `8 − 5`. `Q_real = 5` is right either way — which is exactly why it
> survived — and **no measured number changes.**
>
> `ledger.json`'s `qtok_collapse.adr_published_vs_measured` row carries this
> ADR's published figures against the produced ones, so "the numbers are real"
> is a row rather than something a reader takes on trust. **8 of 9 agree
> exactly**, including all three headline figures; the ninth is the
> `contribution()` slip corrected below.

Two deliberate exclusions, both recorded because both are the kind of thing that
gets quietly added back:

- **PAD query positions are excluded.** Their residual stream is discarded — the
  LM objective masks them (cycle 1's finding: the padded loss halved real-token
  NLL) — so a PAD position's attention is not evidence that a slot was used. This
  is a real decision and not algebra: PAD queries are *not* masked out of the
  softmax, only PAD **keys** are, so including them would import a full attention
  distribution per pad token.
- **`attn_out_proj.bias` is excluded.** §3.2.1 is `W_O v`: a matrix applied to a
  vector. The bias is one vector added once per query position, shared across
  every head and every slot, so attributing it to slots credits all of them
  equally with something none of them caused.

## Two alternatives, and what distinguishes them

### 1. Mean over real tokens — *the same decision, not a different one*

`mean = sum / Q_real`, and `Q_real` is a single scalar shared by every `(l, h, i)`
in the step. §3.2.1's fill-level rescale is
`share_i = raw_i / Σ_j raw_j`, so any uniform positive scaling of `α` **cancels
exactly**. Measured on a real forward pass:

| quantity | sum-collapse | mean-collapse |
|---|---|---|
| `r_i` (slots 0–3) | `0.27818, 0.17073, 0.19087, 0.16022` | `0.27818, 0.17073, 0.19087, 0.16022` |
| max abs difference in `r_i` | — | **2.98 × 10⁻⁸** |
| `contribution()` (unnormalised) | `8.5662, 5.2574, 5.8776, 4.9337` | `1.7132, 1.0515, 1.1755, 0.9867` |

> 📌 Slot 2's sum-collapse entry read `5.8777` until B1 re-derived it. The value
> is `5.877645…`, which rounds to `5.8776`. Nothing depends on it — no threshold,
> no test, no conclusion — and it is corrected rather than left because an
> uncorrected fourth decimal in a cited table is how a reader learns the table
> was never re-derived.

The unnormalised diagnostic differs by exactly `Q_real = 5`; the target does not
differ at all. **So this axis of the brief's worry is discharged: mean-vs-sum
cannot change E0d's answer**, because E0d correlates `r_i`, and `r_i` is
invariant. It changes only `contribution()` / `contribution_per_layer()`, which
are reported per-layer profiles — and there the sum is the interpretable one,
since it is the actual residual-stream increment in model units.

Sum is chosen over mean anyway, because the sum is the quantity with a meaning and
the mean is the sum divided by a bookkeeping constant.

### 2. The EOS position only — *a genuinely different measurement*

Read `α` at the single position where the gestalt is formed. This does **not**
cancel, because one query row's distribution over slots is not the sentence's
aggregate. Measured on the same forward pass, same row:

| | slot 0 | slot 1 | slot 2 | slot 3 |
|---|---|---|---|---|
| `r_i`, sum-collapse | 0.2782 | 0.1707 | 0.1909 | 0.1602 |
| `r_i`, EOS-only | 0.2399 | 0.1961 | 0.2269 | 0.1372 |

**max abs difference 0.0383**, ~14% of the largest entry — on an untrained model,
where there is no learned retrieval structure for the two to disagree about yet.

⚠️ **Stated precisely, because the weaker claim is the true one:** on this row the
two happen to rank the slots identically (`0, 2, 1, 3` both ways). The difference
is in the magnitudes, not yet in the order. Spearman ρ is rank-based, so a
one-row rank agreement is *not* evidence that E0d would score them the same — it
is evidence that a single untrained row is too small a sample to tell, which is
why the discriminator below is E0d over a held-out subsample and not this table.

🔴 **A second weakness in this table, found by B1 and worse than the first.**
Row 0's `[EOS]` is at **query position 9, which row 0's mask marks PAD.**
`_sentence` writes `ids[:, -1] = eos_id` at index 9 and then sets
`mask[0, 5:] = 0`. So on the one row this ADR measures, the sum-collapse
**excludes the single position the EOS-collapse reads**: the two share no query
position at all, and `0.0383` is a divergence guaranteed by the fixture rather
than found in the attention. The conclusion — EOS-only does not cancel — is
still right, and this table is no longer what shows it.

`qtok_collapse.eos_only_supplementary_row` measures the comparison this section
means to make. **Row 1** has a full mask, so its `[EOS]` *is* one of the
positions the sum collapses over; it has 2 live slots rather than 4. There:

| | slot 0 | slot 1 |
|---|---|---|
| `r_i`, sum-collapse | 0.1932 | 0.2068 |
| `r_i`, EOS-only | 0.1960 | 0.2040 |

**max abs difference 0.00286**, 1.4% of the largest entry. EOS-only still
differs, so the decision stands; the margin is a **thirteenth** of what row 0
advertises. Both orderings are `1, 0`. Spearman ρ is recorded as `null` rather
than `1.0` — over two points it is ±1 by arithmetic and carries no information.

**Which row this section should publish is the owner's call.** Both are in
`runs/s0-02-capture-bridge/qtok_collapse.json`; neither is deleted here, because
`0.0383` has already been quoted and a retracted number must stay visible as a
retraction.

The case for EOS-only is not empty: the gestalt is read at `[EOS]`
(`model.py`'s `hit.to(torch.int32).argmax`), it is the gestalt that propagates to
the next step, and a slot that shaped *it* shaped the recurrence. The case
against, and the reason it loses here: §3.2.1's `r_i` is the retrieval demand of
the whole step, the LM loss is computed at every real position, and restricting to
one position discards the retrieval that produced most of the loss the memory
exists to reduce.

### What would distinguish them — the experiment, not the argument

**E0d already is the discriminator.** It computes Spearman ρ between `r_i` and
leave-one-out Δloss on a held-out subsample, and §3.2.1's truth rule is that **LOO
is truth**. So:

> E0d reports ρ for `sum_over_real_query_tokens` and for `eos_only`, alongside
> the gated/raw pair correction 17 already requires. Four rows, not one. If
> EOS-only correlates better with LOO Δloss, this ADR is wrong and is superseded
> by the measurement.

That is a cheap addition — both collapses come from the *same* captured
`[B, H, Q, M]` tensor, so it is two reductions of one forward pass, not two runs.
A second, weaker discriminator is available for free in the same capture: the
attention-sink diagnostic [P14] is sharper under EOS-only, because sink mass
concentrates on particular query positions, so a large `α`/`‖W_O v‖` divergence
under EOS-only and not under the sum is evidence the EOS row is reading a sink.

**Not implemented today.** `cross_capture(collapse=...)` raises on anything but
the decided value, deliberately: an unused branch that nobody runs is how a second
specification survives. Adding `eos_only` is a config enum plus the E0d row above,
in the cycle that runs E0d.

## Consequences

- `r_i` is **insensitive to sentence length only after the rescale**. Before it,
  `contribution()` scales with `Q_real`, so a per-layer profile compared **across
  sentences of different lengths is not comparable** and must be normalised or
  reported with `n_real_query_tokens` beside it. `CrossCapture` carries that field
  for exactly this reason. This is a fresh instance of correction 24's S-7
  (sentence length as an uncontrolled confound), one layer down.
- The collapse is **not** where the vacuity risk lives. Correction 18's six-layer
  profile and correction 20's eval-mode requirement are both upstream of it and
  both already enforced in code.
- The decision is reversible in one line plus an E0d row, and this file is the
  place that says so.

## Status of the alternatives

| Alternative | Status |
|---|---|
| mean over real tokens | **Equivalent** for `r_i`; rejected only for the unnormalised profile |
| mean/sum over **all** tokens incl. PAD | Rejected: PAD queries are not key-masked, so they import a full distribution each |
| EOS position only | **Open, and scheduled** — an E0d row, not a branch in the critical path |
| `W_O`-free `α` sum | Rejected by D-6, and `scripts/mutation_battery.py` now proves the test catches it |
