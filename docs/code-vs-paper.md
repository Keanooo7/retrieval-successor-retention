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

🔴 **The pinned tree is not vendored on this machine yet** (`third_party/PINS.md`,
gauntlet 2.1). Rows below are therefore marked by how they are known:

| Mark | Meaning |
|---|---|
| **source** | read directly from the pinned tree by a session that had it |
| **relay** | asserted by a session that had the tree; not re-read here |
| **paper** | read from the paper or from a session's quotation of it |
| **open** | a question this file is holding, not an answer |

A **relay** row is not evidence. Each one is re-verified against the vendored tree
at 2.1 and its mark upgraded to **source** or the row is corrected. Recording them
now is what makes that re-verification a checklist rather than a rediscovery.

---

## 1 — The gestalt is L2-normalised to unit length. The paper describes no normalisation.

| | |
|---|---|
| Code | `tg/models/tg_srep_head.py`: `srep_BxD = raw_BxD / denom_Bx1 * cfg.srep_norm_target`, with `srep_norm_target: float = 1.0` in `tg_config.py`. The file's own comment calls it equivalent to `F.normalize(v, p=2, dim=0)`. A genuine full-vector L2 normalise. |
| Paper | `s_t = W_sent H^(ℓs)_iEOS`. Zero hits for any normalisation of the gestalt in 20 pages. |
| Mark | **relay** (D-B quotes the source lines; not re-read here) |
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
| Mark | **paper** for the formula; **relay** for the code |
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
| Mark | **paper** |
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
| Mark | **relay** (reference defaults in `third_party/PINS.md`) |
| Resolution | **The per-layer `r_i` profile has SIX entries, not twelve** (D-E). |

§3.2.1 requires the per-layer profile be reported once before collapsing to a
scalar. A twelve-entry profile would be six real rows and six zeros, and the zeros
would be read as a depth finding.

## 5 — Attention dropout is 0.2, so `r_i` collection needs an explicit mode

| | |
|---|---|
| Code | `attn_dropout: float = 0.2` in `tg_config.py`. |
| Mark | **relay** |
| Resolution | **D-F: `r_i` is collected in EVAL mode**; the LM loss is computed in train mode. Written into the code as an explicit mode switch, never an ambient default. |

With dropout live, `α` is stochastically zeroed, and the zeroing is *policy
relevant* — a slot can score zero demand because a mask fell on it. That is noise
the policy would learn from.

## 6 — OPEN: the gestalt layer is 6 or 7

| | |
|---|---|
| Paper / spec | §5.1, citing [P2]: layer **7**. |
| Code | Reference README: `S_REP` extracted at layer **6**. |
| Mark | **open** |
| Resolution | **Unresolved. Resolve against `tg_srep_head.py` during the transcription (2.4) and record the answer here.** Off-by-one in layer indexing is the likely explanation and **it must not be guessed.** |

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
