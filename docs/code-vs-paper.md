# Code vs paper — where the released TG diverges from arXiv:2512.25026

**Standing rule (D-A): the code is the reference.** The fidelity harness compares
against tensors from the released implementation, so the code is the ground truth
by construction — you cannot validate fidelity against a paper. Every divergence
found is recorded here rather than resolved in favour of whichever source reads
better.

The upstream README says so first:

> *"This version differs slightly from the version of the model described in
> (arXiv:2512.25026). It achieves comparable results to those reported in the paper
> when trained with 12M text tokens."*

That sentence is why this file exists. It names no divergence, so every one of them
has to be found.

## Provenance discipline

**The pinned tree is vendored and verified** at
`f220b1098d24a02c94907043d6205c113b31ebb6` (gauntlet 2.1, 2026-09-17). Every row
below that was **relay** has been re-read from the source and upgraded. Rows are
marked by how they are known:

| Mark | Meaning |
|---|---|
| **source** | read directly from the pinned tree by a session that had it |
| **relay** | asserted by a session that had the tree; not re-read here |
| **paper** | read from the paper or from a session's quotation of it |
| **open** | a question this file is holding, not an answer |

A **relay** row is not evidence. Every relay row from the pre-vendoring draft was
re-verified at 2.1; **all six were confirmed, none corrected.** Rows 9–12 are new —
they were not in any audit, and they were found by reading the tree rather than by
being told about it. New relay rows may appear again as other sessions report; the
mark is what keeps them honest.

---

## 1 — The gestalt is L2-normalised to unit length. The paper describes no normalisation.

| | |
|---|---|
| Code | `tg/models/tg_srep_head.py`: `srep_BxD = raw_BxD / denom_Bx1 * cfg.srep_norm_target`, with `srep_norm_target: float = 1.0` in `tg_config.py`. The file's own comment calls it equivalent to `F.normalize(v, p=2, dim=0)`. A genuine full-vector L2 normalise. |
| Paper | `s_t = W_sent H^(ℓs)_iEOS`. Zero hits for any normalisation of the gestalt in 20 pages. |
| Mark | **source** — `tg/models/tg_srep_head.py`; `srep_norm_target` at `tg_config.py:149` |
| Resolution | **The code wins: ‖s‖₂ = 1.0 exactly.** Coordinates are O(1/√d), not Θ(1). |

⚠️ **The near-miss to record.** The paper's line *"`s_t` lives in the same
`d`-dimensional space as token hidden states"* is a claim about **dimensionality,
not norm**. It has been read as a magnitude guarantee. It is not one, and it is a
subtler slip than a terminology collision because the sentence is true either way.

**Consequence, and it is live.** §4.3 derives the `1/d` bilinear multiplier assuming
Θ(1) coordinates. Under unit norm the trained-regime output is Θ(1/d) — it *decays*
with width, in exactly the regime μP governs. **The multiplier is NOT changed here**
(§15.3 makes that derivation the author's). E0a measures the coordinate scale under
both input regimes and the ADR follows the measurement. See correction 15.

## 2 — `P^(sent)` is rank-indexed, not age-indexed

| | |
|---|---|
| Paper | §2.2: `K_M = s_{t−Mt},…,s_{t−1} + P^(sent)_{1:Mt}` — sinusoidal, **keys only**, `V_M` gets none — over a memory *"ordered from oldest to most recent."* |
| Code | Rank order over the memory ordering. |
| Mark | **paper** for the formula; **source** for the code — `tg/models/tg_cross_attention.py:65-70` and the `NOTE` at `:158` |
| Resolution | **Kept rank-indexed** (ADR-0006), with the confound instrumented rather than argued. |

Not a divergence between the two sources — they agree. It is recorded here because
it is a divergence between **what the index means under FIFO and what it means under
RSR**: under FIFO rank and age coincide, and RSR breaks that. See
[ADR-0006](decisions/ADR-0006-sentence-positional-encoding.md).

## 3 — `r_i` must include the memory gate `g_mem`

| | |
|---|---|
| Paper | [P2] puts a learnable scalar `g_mem` on each cross-attention layer, scaling the increment **before** the residual add. App. C measures it growing over training and larger in deeper layers. |
| Spec | §3.2.1's `r_i` omits it. |
| Mark | **source** — `tg/models/tg_model.py:311-314`, `memory_gate_init = 1.0` |
| Resolution | **D-E: `r_i` includes it**, and is computed both ways (gated and raw) and reported against LOO Δloss in E0d. §3.2.1's truth rule stands: if they disagree, **LOO is truth.** |

```
r_i(t) = Σ_{l,h} ‖ g_mem^(l) · α_{l,h,i} · W_O^(l,h) v_{l,h,i} ‖₂
```

D-6's argument for keeping `W_O` — *"precisely where head-specific rescaling lives"*
— applies verbatim to `g_mem`, which is where **layer**-specific rescaling lives.
And because App. C measures the gates growing over training, the weighting is
**non-stationary**: `r_i` collected at epoch 1 and at epoch 12 are not the same
measurement.

## 4 — Cross-attention is on six layers, not twelve

| | |
|---|---|
| Code | Alternating self/cross blocks, `S,C,S,C,…` over 12 layers → cross-attention at `ℓ ∈ {2,4,6,8,10,12}`. |
| Mark | **source** — `DEFAULT_BLOCK_CONFIG = ('S', 'C') * 6` at `tg_config.py:70` |
| Resolution | **The per-layer `r_i` profile has SIX entries, not twelve** (D-E). Cross blocks are 1,3,5,7,9,11 zero-indexed. |

§3.2.1 requires the per-layer profile be reported once before collapsing to a
scalar. A twelve-entry profile would be six real rows and six zeros, and the zeros
would be read as a depth finding.

## 5 — Attention dropout is 0.2, so `r_i` collection needs an explicit mode

| | |
|---|---|
| Code | `attn_dropout: float = 0.2` at `tg_config.py:134`. |
| Mark | **source** |
| Resolution | **D-F: `r_i` is collected in EVAL mode**; the LM loss is computed in train mode. Written into the code as an explicit mode switch, never an ambient default. |

With dropout live, `α` is stochastically zeroed, and the zeroing is *policy
relevant* — a slot can score zero demand because a mask fell on it. That is noise
the policy would learn from.

## 6 — RESOLVED: the gestalt layer is 6 zero-indexed, which is the spec's 7

| | |
|---|---|
| Paper / spec | §5.1, citing [P2]: layer **7**. |
| Code | `srep_extraction_layer: int = 6  # 0-indexed block whose output feeds the head` (`tg_config.py:146`) |
| Mark | **source** |
| Resolution | **Not a divergence.** The off-by-one was the explanation, and it was verified rather than guessed: `srep_layer_idx` returns `min(srep_extraction_layer, N - 1)`, so block 6 zero-indexed is the 7th block. The transcription indexes from 0 and extracts at block 6. |

The instruction in `PINS.md` was *"it must not be guessed"*, and the guess would have
been right — which is exactly the kind of near-miss that makes guessing feel safe.

## 7 — The paper's headline numbers are not reachable from the release

The kickoff's planned smoke test — reproduce 29.8 test PPL and 21 sentence-steps/sec
— measures a difference nobody has characterised, because of the README line at the
top of this file. Replaced by `tests/test_fidelity.py`. See correction 14.

**Also inherited:** do not let the spec's `21 sent/sec` into a capacity plan. It was
measured at `d_model = 768` / 85.6M parameters, and E0c measures the real number on
the target device (gauntlet 2.7).

## 8 — Known-broken code in the pinned tree

Recorded in `third_party/PINS.md` and repeated here because they are code-vs-paper
gaps too:

- `tg/Rough/preprocess_corpus.py` does not run — it imports a `src_recurrent`
  package that exists nowhere in the tree. RSR writes its own preprocessing. The
  script does tell us the segmenter: SaT `sat-3l-sm`, with an explicit refusal to
  fall back to regex.
- The **gist baseline is buggy upstream**, per the tree's own `EXPERIMENTS.md`: the
  mask indexes the gist flag on the query axis instead of the key axis and so grants
  no cross-sentence access at all. Does not affect RSR — TG itself is what we
  transcribe — but it is a caution against treating [P2]'s reported comparisons as
  settled.

---

# Rows found by reading the vendored tree (2026-09-17)

None of these were in any audit. They are here because vendoring at 2.1 was the
first time anyone on this machine read the source, and the transcription (2.4) has
to reproduce them.

## 9 — The S_REP head is not `W_sent`; it is LayerNorm → dropout → MLP → normalize

| | |
|---|---|
| Paper | `s_t = W_sent H^(ℓs)_iEOS` — a single linear map. |
| Code | `SrepHead.__call__`: `nn.LayerNorm(name='ln_srep')` → `nn.Dropout(srep_dropout_now)` → `_head(...)` → L2 normalize. |
| Mark | **source** — `tg/models/tg_srep_head.py` |

At the default `srep_head_depth = 1` the MLP *is* a single `nn.Dense(D)`, so the
paper's `W_sent` is recoverable — **but the LayerNorm and the 0.15 dropout in front
of it are not in the paper at all.** The transcription must include both, and the
dropout is warm-in scaled (`dropout_scale`), so it is not a constant.

`srep_pool_mode: 'eos'` matches the paper's `[EOS]` pooling. `'mean_pool'` exists as
a switch and is not the default.

## 10 — There is an auxiliary hinge penalty on the gestalt norm

| | |
|---|---|
| Paper | silent — it describes no normalization at all (row 1). |
| Code | `srep_norm_penalty()`: squared hinge outside `[1.0 − 0.1, 1.0 + 0.1]`, weighted `srep_norm_reg_weight = 0.01` in `tg_loss.py`. |
| Mark | **source** |

Worth stating plainly because it looks redundant and is not: the penalty is on
`raw_norm_B`, the **pre**-normalization norm, and the output is hard-normalized
afterwards regardless. So the head is *also* being trained to produce vectors that
are already near unit length before the divide.

**Consequence for E0a.** Correction 15 / D-B says the gestalt has unit norm and
therefore `O(1/√d)` coordinates. This row says the pre-normalization vector is
pushed toward unit norm too, so the head's raw output does not grow with width
either. E0a's coordinate check should read **both** — `raw_BxD` and `srep_BxD` —
because a μP violation upstream of the divide is invisible downstream of it.

## 11 — The sinusoidal positional encoding is itself L2-normalized

| | |
|---|---|
| Paper | `P^(sent)_{1:Mt}`, sinusoidal, no normalization mentioned. |
| Code | `sinusoidal_key_pe`: even dims sin, odd dims cos, **the whole vector L2-normalized** (eps 1e-6), zeroed on invalid slots, scaled by `stm_positional_weight = 1.0`. |
| Mark | **source** — `tg/models/tg_cross_attention.py:77-108` |

So the positional term added to a key has norm 1, and the gestalt it is added to has
norm 1 (row 1). **The two are the same magnitude by construction** — the positional
signal is not a small perturbation on the content signal, it is half of the key. That
raises, not lowers, the stakes on ADR-0006: a rank displacement changes a term that
is comparable in size to the content itself.

## 12 — D-D's ablated arm is a config flag, not new code

| | |
|---|---|
| Code | `stm_cross_pos_mode: 'sinusoidal' | 'none'` and `stm_positional_weight: float = 1.0`. |
| Mark | **source** — `tg_config.py:103-105` |

ADR-0006 item 2 asks for a `P^(sent)`-ablated E1 arm to bound the confound. The
reference already has the switch: `stm_cross_pos_mode='none'`, or equivalently
`stm_positional_weight=0.0`. **The transcription must carry both knobs**, and the
arm then costs a config line rather than a code path — which also means it cannot
drift from the main arm.

## 13 — Confirmations of [P2] facts the spec leans on

| | |
|---|---|
| `detach_sreps_for_memory: bool = False` | with the comment *"MUST stay False for the recurrence to train: gradients have to flow from later sentences back through memory."* Confirms §3.1 fact 1 and the no-detach rule the fidelity gradient fixtures exist for (2.3). |
| `stm_backprop_window: Optional[int] = None` | truncated BPTT through the STM **exists as a config axis**, defaulting to unlimited. Note CLAUDE.md's prohibition: a truncated-BPTT window is **not** "consolidation", and this switch is where that mislabelling would attach. |
| `max_sentences_in_short_term: int = 40` | `M = 40`, matching §5.1's corpora value. |
| `memory_mode: 'external'` | cross-attention to memory, `O(n·M)`. The `in_context` alternative forces every block to type `S` and is not the configuration being transcribed. |
