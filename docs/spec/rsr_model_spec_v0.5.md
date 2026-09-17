# Retrieval-Successor Retention (RSR)

**A prospective retention policy for the Thought Gestalt architecture**

Model specification **v0.5** — revised against third-pass advisor review of v0.4

Author: Brendan Keane (concept) · Spec drafted with Claude
Date: 2026-09-14

---

## 0. Changelog from v0.1

Two blocking defects, both confirmed against [P2] directly.

| # | v0.1 defect | Status | Fix |
|---|---|---|---|
| **B1** | TD target used `ψ̂_φ(s_i)` on both sides. Since `s_i` is fixed after write and `ψ̂` was content-only, the fixed point collapsed to `r_i/(1−γ)` — a rescaled *instantaneous* reward. No temporal credit assignment. "Successor" was decorative. | **Confirmed** | §3.2–3.3 rewritten with context-conditioned value `ψ̂(s_i, c_t)`. The state that transitions is the reader, not the slot. |
| **B2** | §3.6 claimed value-based retention causes unbounded activation memory. | **Confirmed wrong.** [P2] App. A: memory size "limits which sentence vectors are available to cross-attention in the forward pass, it does not limit the backward flow of gradients … maximum backward depth is bounded by the maximum stream length S." Evicting a slot does **not** free its graph. | §3.6 rewritten as truncated BPTT, orthogonal to retention. **CLS claim deleted entirely** — the mapping was inverted (see §3.6). |

Non-blocking but load-bearing:

| Area | Change |
|---|---|
| Reward definition | `r_i` is now norm-weighted contribution, validated against leave-one-out Δloss (§3.2) |
| Censored feedback | Shadow buffer added (§3.4) — the policy was training only on data it chose to collect |
| Baselines | **H2O (accumulated attention) and Expire-Span added.** These define the contribution bar; FIFO/LRU are a floor (§5.3, §11) |
| Stream length | `S` is now an explicit per-experiment parameter with a memory budget. **This was the largest hole in v0.1** (§5.2) |
| Memory capacity | Split: `M = 16` synthetic, `M = 40` corpora, with GPT-2's context window matched (§5.1) |
| Bias loop | `ψ̂` z-scored; `ū` is an EMA rate, not a cumulative sum; `τ` measured before freezing (§3.5) |
| Vacuity gate | Now on the full eviction score, ρ < 0.7, partial correlation, plus an age-only head arm (§7.1) |
| `γ` | Constrained by `1/(1−γ) ≤ min(S, A_max)`. 0.99 dropped (§4.5) |
| Scaling fit | **Demoted from headline to hygiene.** A 3× width range at ≤21M cannot resolve slope from intercept (§6) |
| Primary result | Reintroduction loss + oracle regret + **human narrative-recall gradient** (§6, §10) |
| Framing | Retroactive prioritization (§10) — the biological case for demand-after-encoding |
| §2 | Mnemonic-traditions section removed from the spec; reduced to a provenance note (§12) |

### v0.2 → v0.4 (second-pass review)

Weeks 1–4 were approved unchanged; E0a–E2 are unaffected by everything below.

| # | Defect | Fix |
|---|---|---|
| **C1** | **E7 inoperable (a).** At `M = 40`, a 10–30 sentence passage never fills memory. No policy evicts, every arm retains byte-identical contents, and the experiment returns a tautological null. | E7 gets its own `M = 8` (§5.1) **plus a capacity-mismatch control** (§10.1) |
| **C2** | **Underfull mask deleted the signal it protected.** Excluding `\|memory\| < M` removes 50% of E3's steps and 100% of E7's. | **Rescale the target**, do not down-weight the loss (§3.2.1) |
| **C3** | **FIFO is a degenerate control for E7.** Survival time is `min(M, S−i)`, a deterministic function of serial position, which the partial correlation zeroes out by construction. | Controls are H2O and LRU (§10.1) |
| **C4** | **No control for the word "future."** RSR adds context conditioning *and* lookahead over H2O; no arm separated them. | `γ = 0` arm required in E1 and E3; falsifier 3b; **plus an attention-logit control** (§2) |
| **C5** | **No eviction warmup.** An untrained head evicts near-randomly during the period when those evictions shape the gestalts it depends on. | FIFO until `T_warm`, `ψ̂` trains passively (§3.4) |
| **C6** | **μP prescription for a bilinear head unwritten.** "Readout-like" is not a prescription. | Explicit init/multiplier/LR table with the correct scaling argument (§4.3) |
| **C7** | **No compute budget.** v0.1 died of an unpriced plan; v0.2 still had no number. | GPU-hours, dollars, and a fixed cut order (§4.1) |
| **C8** | E7 stimulus set unnamed; 10 of 14 citations unverified; Expire-Span scheduled as an afternoon. | E0f, E0g added; Expire-Span gets its own slot with a demotion path (§8) |
| **C9** | **The reviewer's objections are not the author's contribution.** v0.2 is correct largely where the review is. | §15 — unfilled by design |

### v0.4 → v0.5 (third-pass review)

**Verdict on v0.4: conditionally approved through week 4 only; the PG-19 block is not approved.** Weeks 1–4 proceed. Everything downstream is gated on the conditions in §16.

| # | Defect | Fix |
|---|---|---|
| **D-1** | **§3.5 was numerically inert.** `γ_b = 0.001` against a stream-bounded lifetime `≤ S = 80` caps \|`b`\| at **0.08 SD** of a z-scored `ψ̂`, and `b` resets at stream boundaries. The anti-collapse loop could not move the argmin. It was indistinguishable from `b ≡ 0`, which is also the §3.7 reduction condition — A5 would have reported "no effect" in week 7 and the loop would have been called unnecessary rather than absent. | `γ_b` derived from measured lifetime in E0e (§3.5); A5 becomes a real sweep; **plus a decision-attribution check**, because the fix reopens the v0.2 worry that the controller *is* the policy |
| **D-2** | **The primary metric had no power estimate and no pipeline.** At `S = 80` only **16** stream positions can host a `k = 64` reintroduction, and detecting reintroductions at all requires coreference over PG-19 — an unnamed dependency with its own error rate feeding straight into the dependent measure. | **E0i**, a week-1 CPU kill gate with a pre-registered events-per-bucket floor (§6) |
| **D-3** | **Slot value was scored independently; retention value is not separable.** Redundant slots split attention mass and both look evictable; unique-but-modest slots score below popular-but-redundant ones. Set value under LOO is submodular, so greedy argmin over independent scores is not an approximation to it — under redundancy it is anti-correlated with it. E0d was built to detect this and no response was specified. | Marginal scoring: redundancy penalty in the eviction rule (§3.4); falsifier 5 |
| **D-4** | **Expire-Span was a stretch goal with a demotion path.** It is the one baseline whose *success invalidates the design rather than the result*: soft differentiable expiry trains retention by the LM loss directly and makes §3.2–3.4's entire apparatus unnecessary. | Moved to the **week-4 gate** on synthetic. **Demotion path deleted** (§6, §8) |
| **D-5** | **E7's capacity mismatch had an unfunded resolution.** A precondition whose failure mode is not in the budget is not a precondition. | Second PG-19 model trained at `M = 8, d = 128`, **priced** (§4.1, §5.1) |
| **D-6** | **`r_i` was not faithful to [P5].** `‖α·v‖` omits `W_O`, which is where head-specific rescaling lives; layer-summing conflates contributions to different residual-stream positions. | `‖α · W_O v‖`, per-layer profile reported once before collapsing (§3.2.1) |
| **D-7** | **§3.2.1 overclaimed "unbiased."** The rescale removes the *mechanical* fill-level bias under a uniform-competitor assumption; it still credits absent competitors and biases genuinely important stream-initial slots downward. | Claim narrowed to what it does (§3.2.1) |
| **D-8** | **TD(0) was unmotivated and carried the one instability the document flags.** Finite horizon `S ≤ 80`, `γ ≤ 0.97`, no gradient to the transformer, and only the *policy* needs to act online — the target does not. | **Monte-Carlo / λ-return is the default; TD(0) is the ablation** (§3.3). §7.7 collapses |
| **D-9** | **E1 was treated as a result.** The synthetic generator constructs the gap structure that makes lookahead pay, so `RSR > RSR(γ=0)` there is evidence about the optimizer, not about language. | E1 restated as a **kill gate, not a result** (§6) |
| **S-1** | **[P11]'s leading-edge strategy — a hand-specified retention policy over a capacity-limited discourse buffer, selecting by structural importance, validated against human recall — was cited only for the behavioural effect.** RSR is a learned version of it. Omitting it is a scholarship failure. | Reframing adopted in §2; named ancestor and **implemented baseline** in §11 |
| **S-4** | Hard eviction is a poor idealization of graded, cue-dependent, interference-based forgetting. | Stated as a limitation with a named successor (§13) |
| **Self-audit** | **`K = M` is too small.** A slot evicted shortly after write needs shadow coverage out to the longest gap the claim targets; at `M = 40` with gaps to 64, `K = 40` censors exactly the events E3 is about. Same disease as D-1, found by applying D-1's test to the other frozen constants. | `K ≥ max target gap` (§3.4, §4.5) |

**Not fixed, by design:** §15.2. See §15.

---

## 1. Provenance and epistemic status

| Tag | Meaning |
|---|---|
| **[E]** | Established — demonstrated in a cited source |
| **[X]** | Extension — mechanically straightforward given [E], untested |
| **[S]** | Speculative — original claim, no supporting result |

### Sources

| Ref | Source | Status |
|---|---|---|
| **[P1]** | Cho & McClelland (2026), *Capturing rapid learning in an extended successor representation theory of the cognitive map* | **Preprint, not peer reviewed** |
| **[P2]** | Borazjanizadeh & McClelland (2026), *Modeling Language as a Sequence of Thoughts* (TG), arXiv:2512.25026v2 | **Under review** |
| **[P3]** | Dayan (1993), successor representation | Published |
| **[P4]** | Yang & Hu et al., Tensor Programs V (μP) | Published |
| **[P5]** | Kobayashi et al. (2020), *Attention is Not Only a Weight* | Published |
| **[P6]** | Zhang et al. (2023), **H2O: Heavy-Hitter Oracle** | Published |
| **[P7]** | Sukhbaatar et al. (2021), **Expire-Span** | Published |
| **[P8]** | Rae et al. (2020), **Compressive Transformer** (introduced PG-19) | Published |
| **[P9]** | Graves et al. (2016), **DNC** | Published |
| **[P10]** | Liu et al. (2023), **Scissorhands** | Published |
| **[P11]** | Kintsch & van Dijk (1978); Thorndyke (1977) — levels effect in narrative recall | Published |
| **[P12]** | Dunsmoor et al. (2015, *Nature*); Braun, Wimmer & Shohamy (2018) — retroactive memory prioritization | Published |
| **[P13]** | Frey & Morris (1997) — synaptic tagging and capture | Published |
| **[P14]** | Xiao et al. (2023), StreamingLLM — attention sinks | Published |

> **[P5]–[P14] are verified in week 1 (E0f), not "before citing."** Ten of fourteen references — including every support for the cognitive claim in §10 and every baseline in §11 — were unchecked at drafting. This is an afternoon of work and the cheapest risk retirement in the project. **This note is deleted from the document once E0f is logged**; it cannot still be here at week 12.

### Attribution

[P1] and [P2] are McClelland's. **Nothing in §3 is.** Where this document says "the framework predicts," it means the framework *as extended here*.

### Relationship to Neural Priming v0.2

Superseded. Cut: prediction-error salience (novelty trap), the AIMD controller, the successor-feature value readout, the analogy claim, offline replay. Retained: the diagnosis that TG's memory is managed by a policy nobody chose on purpose, and the reduction-to-identity discipline.

---

## 2. The claim

> **Kintsch & van Dijk (1978) specified retention by structural importance over a capacity-limited discourse buffer — the leading-edge strategy — and it reproduced the human recall gradient. Does a system trained only to predict the next sentence *discover* that policy? RSR is the test.**

That is the question the project asks. The mechanism below is how it is asked.

> **Mechanism.** TG evicts FIFO. Retain instead by predicted future demand, conditioned on where the discourse currently is.

**Why this framing and not the previous one.** v0.4 led with the engineering delta. At ≤21M parameters and 30M tokens that delta will not interest a systems audience regardless of outcome, and §2 itself states the prior that context conditioning, not lookahead, does the work — a proposal whose author expects the headline mechanism to be inert. Under the comprehension framing the same experiments answer a question that does not depend on which mechanism wins: [P11]'s leading-edge strategy becomes an implemented baseline rather than a citation, E7 becomes the point rather than a garnish, and **a null on E3 is survivable**.

The engineering delta remains, as the secondary claim:

> **Prior learned-eviction methods score memories by accumulated *past* attention ([P6]) or by a learned *static* span ([P7]). RSR predicts *future* retrieval demand as a function of the current discourse state.**

**That sentence names two mechanisms, and the experiment must separate them.**

1. **Context conditioning** — scoring a slot by relevance to where the discourse is *now*.
2. **Temporal lookahead** — the discounted sum over *future* steps.

H2O controls for neither individually. The separating control is **`γ = 0`**, which keeps (1) and removes (2). It is a required arm in E1 and E3.

> **The prior should be that (1) does the work.** Current-context relevance is a far easier signal than multi-step prediction, and `ψ̂ = s_iᵀWc_t` is *architecturally the same functional form as the cross-attention logit itself*. It can fit "what is being attended to right now" almost for free.

That similarity motivates a second, cheaper control the `γ = 0` arm does not provide: **regress `ψ̂(γ=0)` on the model's own current-step cross-attention logits.** If they are near-collinear, the estimator is re-deriving the attention mechanism rather than learning anything about retention, and `γ = 0` is not a neutral control but a reimplementation of the forward pass. Report the R² (E0h).

If RSR does not beat accumulated-attention eviction, the finding is that predicting the future adds nothing over measuring the past. That is a legitimate result and a different paper. **It must be known by week 4, not week 11** (§6, E1).

### Falsifiers

1. **Vacuity.** Eviction score is a monotone function of slot age (§7.1). RSR is FIFO with extra parameters.
2. **No headroom.** Oracle retention ≈ FIFO on the target corpus (E-feas). The corpus cannot test the claim.
3. **No prospective advantage.** RSR ≈ H2O. Past attention already contains everything future demand would add.
3b. **No lookahead advantage.** RSR(`γ = 0.9`) ≈ RSR(`γ = 0`). The gain is current-context relevance, not prospection. Publishable, but §2's delta sentence is then false and must be rewritten before the abstract is.
3c. **No mechanism at all.** `ψ̂(γ=0)` is collinear with the current cross-attention logit (E0h). The estimator reproduces the forward pass.
5. **Redundancy dominates.** The eviction score is anti-correlated with LOO Δloss wherever live slots overlap in content (E0d, §3.4). Independent scoring is then not an approximation to marginal value but an inversion of it, and greedy argmin is the wrong rule regardless of how good `ψ̂` is.
6. **Superseded by gradients.** Expire-Span, ported into TG, matches or beats RSR on synthetic (E1). The reward proxy, the MC/TD machinery, the shadow buffer and the warmup then exist to work around a non-differentiability that a soft-expiry formulation simply does not have, and the contribution is a more complicated route to what the LM loss already provides. **This falsifier invalidates the design, not the result.**
4. **No behavioural signature.** RSR's retention choices track human narrative recall no better than FIFO (§6, E7). The result is then an engineering note, not a cognitive-science contribution.

---

## 3. Architecture

### 3.1 Base model (unmodified TG) **[E]**

From [P2]: a recurrent transformer processing one sentence per step. Tokens attend causally within the sentence and reach prior sentences only through cross-attention to a working memory of gestalt vectors. Each gestalt is `s_t = W_sent · H^(7)[EOS]`, appended **without detaching its computation graph**.

Three facts from [P2] that constrain everything below:

1. **Memory capacity `M` bounds the forward view only.** Gradients flow backward through the stored graphs of *evicted* sentences via surviving slots' links ([P2] App. A, Fig. 6).
2. **Backward depth is bounded by stream length `S`**, not by `M`. Memory is reset (stop-gradient) at each stream boundary.
3. **Streams are a training-time slicing device.** Validation and test documents are never sliced.

**The entire contribution of this document is replacing "the oldest entry is removed."**

### 3.2 Retrieval demand and the retention value **[X]**

#### 3.2.1 Defining `r_i` — what counts as "being retrieved"

v0.1 used summed, normalized cross-attention probability. That is wrong, and the ways it is wrong are documented:

- Attention weight is a relative allocation over a softmax, not evidence of use. A head with nothing to retrieve must still place its mass somewhere — the attention-sink phenomenon [P14].
- A slot can receive large α while contributing almost nothing to the residual stream.

**Definition.** `r_i(t)` is the **norm-weighted contribution** of slot *i* at step *t* [P5]:

```
r_i(t) = Σ_{l,h} ‖ α_{l,h,i} · W_O^{(l,h)} v_{l,h,i} ‖₂      (then normalized across live slots)
```

> **`W_O` is not optional.** [P5] measures the norm of the *transformed* vector including the output projection, and `W_O` is precisely where head-specific rescaling lives. Dropping it reintroduces the confound the norm-weighting was adopted to remove.

> **Report the per-layer profile once before collapsing to a scalar.** Summing across layers conflates contributions to different residual-stream positions. [P2] App. C–D show memory gates and cross-attention gradient share are strongly depth-stratified, so the layer sum is not obviously the right aggregate.

**Validation requirement (E0d, week 1–2).** On a held-out subsample, ablate slot *i* and measure Δ next-sentence loss (leave-one-out). Report Spearman ρ(`r_i`, LOO Δloss). **If they disagree, LOO is truth and `r_i` is a confound** — and §3.4's redundancy term is the specified response, not a shrug. This is the same computation as the oracle (§5.3), so it is built once and used twice.

**Normalization caveat.** Because `r` is a share, `Σ_i ψ_i` is pinned near `1/(1−γ)`. Acceptable for a relative eviction rule — state it explicitly in the paper.

**Underfull steps: rescale the target, do not mask and do not merely down-weight.** Early in a stream few slots exist and each receives enormous normalized mass for purely structural reasons; uncorrected, the estimator learns "stream-initial content is valuable."

v0.2's fix — excluding steps where `|memory| < M` — has a blast radius nobody priced: at `M = 40, S = 80` it deletes **half of E3's training signal**, and on E7's short passages it deletes **100%**.

The obvious repair, down-weighting those steps by `|memory_t|/M`, is also wrong, and the reason matters. Down-weighting a **biased target** reduces how hard the estimator fits it; it does not remove the bias. A slot that is one of three necessarily receives ~1/3 of the mass regardless of usefulness, and the target still says so. The magnitudes happen to be reciprocal — inflation `M/|memory_t|` against weight `|memory_t|/M` — which is why down-weighting *approximately* works and why the error is easy to miss.

Correct the target instead:

```
share_i(t) = ‖·‖_i(t) / Σ_j ‖·‖_j(t)          # share of live slots
r_i(t)     = share_i(t) · |memory_t| / M      # share-of-M-equivalent
```

This removes the **mechanical fill-level bias** — the component that comes from `|memory_t|` rather than from the slot — and puts every step on a common scale, so the loss needs no weighting. `w_t = |memory_t|/M` may be retained as variance reduction, but it is no longer the correction.

> **It does not make the target unbiased, and the document must not say so.** The rescale credits a slot as though it had competed against `M` slots when only `|memory_t|` existed, which biases genuinely important stream-initial content *downward*. That is a smaller and better-understood error than the one it replaces, and it is the honest description. "Unbiased" is the word a reviewer will test.

Report the effective fill fraction `mean(|memory_t|/M)` per experiment in §5.2 — ≈0.76 at `M = 40, S = 80`, versus the 0.51 that v0.2's mask would have retained.

#### 3.2.2 The value function — context-conditioned **[X]**

v0.1's content-only value was degenerate. The correction is the conceptual core of v0.2.

**A successor representation requires a state that transitions. A memory slot has no successor slot — slots do not lead anywhere. The state that transitions is the reader.**

```
ψ(s_i, c_t) = E[ Σ_{k=0}^{∞} γ^k · r_i(t+k)  |  context c_t ]
```

where `c_t` is the current sentence gestalt (or a running context vector). `ψ` now reads as: *how much will this memory be needed, given where the discourse currently is.* That is the quantity the model actually wants, and it makes the mechanism a prediction about the unfolding narrative rather than a popularity score.

**Estimator.** Bilinear, with an MLP variant as ablation A6:

```
ψ̂_φ(s_i, c_t) = s_iᵀ W c_t + u ᵀ[s_i ; c_t]        φ = {W, u}
```

Cost remains `O(M·d)` per step.

> **Age is still excluded** from `ψ̂`. Supplying it invites collapse onto recency and makes the vacuity failure mode invisible rather than merely possible. An age-only head and a content+age head are both retained as baseline arms (§7.1).

### 3.3 Learning rule **[X]**

**Monte-Carlo return is the default. TD(0) is the ablation.** The reasoning is in the box below the equations.

TD(0) form, retained as ablation A8, with a genuine bootstrap because `c` moves:

```
L_TD = Σ_{i ∈ M}  ( ψ̂_φ(s_i, c_t)  −  sg[ r_i(t) + γ · ψ̂_φ(s_i, c_{t+1}) ] )²
```

**Corrected fixed point.** `ψ̂*(s_i, c) = E_{c'|c}[ r_i + γ·ψ̂*(s_i, c') ]` — a Bellman equation over the context process.

**Default form — Monte-Carlo / λ-return.** Streams are finite (`S ≤ 80`), `γ ≤ 0.97`, no gradient reaches the transformer, and only the *policy* must act online; the *target* need not. So compute the realized discounted return directly on completed streams and regress:

```
G_i(t) = Σ_{k=0}^{S−t} γ^k · r_i(t+k)
L_MC   = Σ_{i ∈ M} ( ψ̂_φ(s_i, c_t) − sg[ G_i(t) ] )²
```

`λ` interpolates: `λ = 1` is MC, `λ = 0` is TD(0). Default `λ = 1`; sweep `λ` once on synthetic only if MC variance is limiting.

> **Why the default flipped.** v0.4 carried a bootstrapped estimator on a representation that is itself training — a known-unstable combination — and then listed that instability as a failure mode to monitor (§7.7). Over an 80-step finite horizon there is no reason to pay for it. MC removes the bootstrap, the moving target, the EMA target copy, and the divergence risk outright. The cost is variance, plus the censoring already handled by the shadow buffer. **If TD(0) wins on sample efficiency in A8, it is kept with a measured reason rather than because it sounds like reinforcement learning.**

> **Reversion check against v0.1.** v0.1's degeneracy was *algebraic*: the same function of the same argument appeared on both sides of a bootstrap, collapsing the fixed point to `r_i(t)/(1−γ)`. MC has no bootstrap, so that collapse cannot recur by construction. But v0.4's defense sentence — *"there is no closed-form regression that produces this"* — **becomes false under MC and is withdrawn**: MC *is* a regression. What distinguishes it from v0.1 is the target, not the estimator class. `G_i(t)` is a realized discounted sum over actual future steps; v0.1's was a rescaling of the instantaneous reward. Anyone re-deriving this must check the target, not the presence of a bootstrap.

**Do not backpropagate `L_TD` into the transformer or `W_sent`.** Only `φ` receives its gradient. Two reasons: [P2] §4 documents that auxiliary sentence-level objectives are brittle here, and it preserves exact reduction to TG (§3.7). Note `c_t` enters `ψ̂` with a stop-gradient on the transformer side.

```
L = L_NTP + β · L_TD
```

### 3.4 Eviction, and the censored-feedback problem **[X]**

```
i* = argmin_i [ z(ψ̂_φ(s_i, c_t)) + b_i − ν · max_{j≠i} cos(s_i, s_j) ]
```

where `z(·)` is a z-score across live slots at the current step (§3.5). Subject to the hard bound in §3.6.

**Eviction warmup — required.** `ψ̂` is a function of representations that are themselves being trained, and [P2] states that early sentence representations are "largely uninformative." An untrained head therefore evicts near-randomly during exactly the period when those evictions shape the gestalts it later depends on.

```
t < T_warm :  eviction is FIFO;  ψ̂ trains passively on realized r_i (β active, policy inert)
t ≥ T_warm :  eviction switches to argmin[ z(ψ̂) + b ]
```

`T_warm` = one epoch, fixed, and **stated as a fraction of total epochs** in the writeup — one epoch of three is a different experiment from one of twelve. This also gives a clean on-policy/off-policy boundary for §7.2: everything before `T_warm` is unbiased FIFO-collected data. If the TD residual (§7.7) misbehaves after handover, add an EMA target copy of `φ` before touching `β`.

**The censoring problem.** `r_i` is observable only for slots *in memory*. A slot evicted at step 5 never demonstrates that it would have been retrieved at step 40. This is bandit feedback with survivorship bias: the policy generates its own training distribution, so early mistakes are self-confirming and invisible. §3.5 addresses the *forward* rich-get-richer loop; this is a distinct loop in the *learning signal*.

**Shadow buffer.** Retain the last `K` evicted gestalts outside the forward pass. Compute their would-be contribution scores post-hoc — the queries already exist, so cost is one extra `O(K·d)` matmul — and use them as counterfactual targets, down-weighted by `λ_shadow`. This is also how the PG-19 oracle is computed (§5.3): build once, use twice.

> **`K = M` is wrong, and it is D-1's disease in a different constant.** A slot can be evicted at any point after write, so observing whether it *would* have been retrieved at the longest gap the claim targets requires shadow coverage out to that gap — not out to `M`. At `M = 40` with E3's target window `(40, 64]`, a slot evicted shortly after write needs ~64 steps of coverage; `K = 40` censors exactly the events E3 exists to measure. **Set `K ≥ max target gap`:** `K = 64` on PG-19, `K = 40` on synthetic (max generated gap 40). Cost is `O(K·d)`, still negligible.

**Marginal value, not independent value (D-3).** `ψ̂` scores each slot in isolation, but what makes a memory worth keeping is the loss it saves *that no other retained slot can save*. Two slots carrying the same proposition split the attention mass, both look half as valuable, and both become evictable — losing the information entirely. Set value under leave-one-out is **submodular**, so greedy argmin over independent scores is not an approximation to marginal value; where slots are redundant it is anti-correlated with it.

The minimum viable response is the `− ν · max_{j≠i} cos(s_i, s_j)` term in the eviction rule above: `O(M²d)`, negligible at `M ≤ 40`, `ν` swept once on synthetic. **Report ρ(score, LOO Δloss) with and without it.** If the redundancy term closes the gap E0d finds, that is a result in itself.

*Cognitive resonance, stated and not leaned on: gist-based retention discards redundant surface detail while preserving distinct propositions. Independent-slot scoring cannot produce that; marginal scoring can.*

**Decision attribution — required from the first policy run.** The eviction score now has three terms, two of which were added to fix failures of the first. Log, per eviction, which term determined the argmin: the fraction of decisions where `b` flips the choice relative to `z(ψ̂)` alone, the fraction where the redundancy term does, and the fraction where all three agree. **If `b` flips a large share, the balance controller is the policy** — which is the v0.2 objection that z-scoring was introduced to prevent, reopened by D-1's fix. Same test for `ν`. Report the decomposition in the paper.

`K = M`, `λ_shadow = 0.5`, both swept once in A7.

### 3.5 Anti-collapse: the gradient-free protection bias **[X]**

**The failure.** Retention determines what is attendable; attention trains retention. A retained slot is attended, raising its value, retaining it. Fixed point: a few slots monopolize memory. Known as router collapse in sparse MoE, and as over-representation in [P1] Figs. 3f–g.

**Mechanism.** A scalar bias `b_i`, added **inside the eviction argmin only**, never into `ψ̂` and never into any differentiable path:

```
b_i ← b_i − γ_b      if ū_i > (1+τ)/M
b_i ← b_i + γ_b      if ū_i < (1−τ)/M
b_i ← clip(b_i, −b_max, +b_max)
```

If `b` entered `ψ̂`, it would produce gradients pushing toward balanced retention at the expense of language-modeling loss, and would corrupt the estimate itself — an under-attended slot would receive an inflated predicted *demand*, not merely an improved chance of survival. Balance must be a zero-gradient control loop, not a competing objective.

**Three corrections from review:**

1. **Scale is not free.** `ψ̂`'s raw output range is unspecified; at share-normalized reward it can reach `1/(1−γ)`. If `ψ̂ ≫ b`, the bias is rounding error; if `ψ̂ ≪ b`, **the policy is the balance controller, not the value estimate — and it might still beat FIFO for reasons unrelated to the hypothesis.** Fix: z-score `ψ̂` across live slots each step. `b_max = 1.0` then means one standard deviation and is interpretable.
2. **`ū` must be a rate, not a cumulative sum.** A raw running sum accumulates with age by construction, so the loop preferentially marks old slots evictable and smuggles recency back in through the anti-collapse mechanism. Use an EMA of per-step share, **half-life = `E[lifetime]/4`, measured in E0e** — not the frozen 16 of v0.4, for the reason in point 4.

4. **`γ_b` is a timescale, not a magic number — and v0.4's value made this entire section inert.** With `γ_b = 0.001`, `b_max = 1.0`, and a slot lifetime bounded by the stream (`S ≤ 80`, with `b` reset at stream boundaries), the maximum attainable \|`b`\| over a slot's *entire life* is `0.001 × 80 =` **0.08 standard deviations** of a z-scored `ψ̂`. Realized lifetimes are shorter, so the true figure is smaller. The control loop could not move the argmin. It was operationally identical to `b ≡ 0`, which is also the §3.7 reduction condition — so A5 would have reported "no effect," and the conclusion would have been that the loop is *unnecessary* rather than *absent*.

   Require `b` to reach `O(b_max)` within the expected slot lifetime, not within training. Updates fire only when `ū` is outside the `τ` band, empirically on order a quarter of steps:

   ```
   γ_b  ≈  b_max / (0.25 · E[lifetime])
   ```

   At `E[lifetime]` of 20 / 40 / 80 that is 0.20 / 0.10 / 0.05 — **order 0.05–0.1, not 0.001.** `E[lifetime]` comes free from the same E0e FIFO run that measures the `ū` distribution. **A5 becomes a genuine sweep over `γ_b`, not a sensitivity check around a frozen value**, and it is paired with the decision-attribution decomposition in §3.4, because a `γ_b` large enough to matter is also large enough to take over.
3. **`τ` is measured, not frozen.** Attention over slots is heavy-tailed; `ū` may essentially never sit inside a ±25% band, leaving `b` saturated for most slots most of the time. **Measure the distribution of `ū` on a FIFO run (E0e) before freezing `τ`.** Freezing an unmeasured constant is how v0.2-NP died.

### 3.6 Slot lifetime and backpropagation depth **[X]** — rewritten

> **v0.1's premise was false.** It claimed value-based retention causes unbounded activation memory. Under [P2], evicting a slot does not free its graph: surviving slots retain links to the sentences they attended to at formation, and backward depth is bounded by stream length `S` with memory reset at stream boundaries. Under vanilla TG the retained graph is already the whole stream. **RSR cannot cause an activation blowup, and slot lifetime is already bounded by `S`.**

Two things follow.

**(a) `A_max` must be set relative to `S`, not chosen freely.** With `S = 30`, sweeping `A_max ∈ {16, 32, 64}` leaves two values inert and produces a flat curve that is an artifact, not a finding. Constraint: `A_max ≤ S`, and `A_max` is swept only where `S` makes it bind.

**(b) `A_detach` is truncated BPTT, not consolidation.** It is an intervention available to vanilla TG equally and is **orthogonal to retention**. [P2]'s own ablation shows truncation is expensive: detaching gestalts at write costs 29.8 → 35.0 test PPL (throughput 21 → 24). Keep it only as ablation arm A4, to characterize the reach-versus-gradient-quality trade.

> **The CLS claim is deleted.** It was inverted. In CLS, consolidation is the process by which an experience *acquires* the ability to shape slow-learning cortical weights through replay. `detach` does the opposite: the item stays retrievable but permanently loses the ability to shape any weights at all. That is the *failure* of consolidation with retrieval preserved. Calling a truncated-BPTT window "consolidation" is the kind of move that makes cognitive scientists distrust modelling papers.

**The real bill, running the opposite direction.** E3 measures reintroduction loss at gaps `k` up to `A_max`. **Gaps longer than `S` do not exist in training data.** A headline experiment at `k ≈ 64` therefore requires `S ≳ 80` — roughly 2.7× [P2]'s starting curriculum, with correspondingly deeper graphs. *That* is the activation-memory bill, and v0.1 had it backwards. See §5.2.

### 3.7 Reduction to exact TG **[X]**

Set `ψ̂ ≡ −a_i`, `b ≡ 0`, **`ν = 0`**, `β = 0`, `A_max = M`, warmup irrelevant (`T_warm = ∞`), shadow buffer off. Eviction becomes `argmin_i(−a_i)` = oldest slot. Bit-exact TG.

> **`ν = 0` is new in v0.5 and is part of the reduction condition.** Every term added to the eviction score must have a documented off-switch, or §3.7 silently stops being a reduction to TG and E0b stops testing what it claims to test.

**Verified numerically, not argued** (E0b). **Construct the value head after the model, or seed it from a separate RNG stream** — otherwise instantiating `φ` consumes draws, shifts data order and dropout masks, and E0b fails for a reason unrelated to the mechanism.

---

## 4. The five axes and their bills

Every mechanism cuts a coupling; the cut relocates the cost rather than deleting it.

| Axis | RSR's weld | RSR's bill |
|---|---|---|
| **Compute** | Experiments ∝ questions asked | Runs that fit in 12 weeks |
| **Memory** | Activation memory ∝ **stream length S** | E3 needs `S ≳ 80`; deeper graphs than [P2] ever trained |
| **Signal** | HP optimum ∝ width (μP cuts it) | Value head breaks transfer if mis-grouped |
| **Iteration** | Wall-clock per run on MPS | Sets the compute axis |
| **Attribution** | Effects ∝ mechanisms | Every knob is a tax |

### 4.1 Compute

Retention head: `O(M·d)` forward, `O(M·d)` TD, `O(K·d)` shadow. Against a 12-layer transformer, ~2–4%. Negligible per step.

The binding constraint is **experiment count**. §6 collapses the v0.1 sweep from 27 configurations to 3 by freezing `γ` and `β` on synthetic before touching PG-19.

**Budget, in hours and dollars — required, because v0.1 died of an unpriced plan.** [P2]: 21 sent./sec on one A40 at `S ≈ 30`. E3 runs a backward chain ≈2.7× deeper at `S = 80` with a correspondingly smaller batch, so assume **≤ 7 sent./sec** pending E0c. A 30M-token corpus is ≈1.2M sentence steps, ≈48 h per run.

| Line item | Runs | Est./run | Est. total |
|---|---|---|---|
| E3 PG-19 `S=80` — RSR, FIFO, H2O, RSR(`γ=0`) × 3 seeds | 12 | 48 h | **576** |
| **A2, A4 ablations on PG-19** (`A_max`, `A_detach` — the reach/gradient trade only exists at `S = 80`) | 4 | 48 h | **192** |
| **E7 model at `M = 8, d = 128, S = 30`** — RSR / H2O / LRU × 2 seeds (D-5) | 6 | ~12 h | ~72 |
| E4 WikiText, `S` curriculum | 4 | ~16 h | ~64 |
| E5 width sweep (hygiene) | 8 | ~10 h | ~80 |
| E1/E2 synthetic — `γ`×`β` grid, `ν` sweep, Expire-Span, full baselines | ~28 | ~2 h | ~56 |
| E-feas, E6, A1/A3/A5–A8 (synthetic or eval-only) | ~15 | ~2 h | ~30 |
| **Total** | | | **1,070 GPU-h** |

**Parallel capacity, which is the thing actually reserved.** Pricing the total says nothing about schedulability. Weeks 5–7 are `3 × 7 × 24 =` **504 wall-clock hours** and must contain E3 (576) plus A2/A4 (192) = **768 GPU-h**:

| Assumption | GPUs required |
|---|---|
| 100% utilization, zero failed runs | 1.52 |
| 85% utilization, +25% rerun margin | 2.23 |
| 65% utilization, +25% rerun margin | 2.93 |

**Reserve 4, floor 3, continuously, from week 5.** A single GPU cannot run weeks 5–7 at any utilization. Name the provider and reserve in week 1.

> v0.2 had no number at all. An intermediate draft budgeted ≈760 h by omitting the ablations, E-feas, E6, E7 — a ~30% undercount driven entirely by assuming ablations are cheap. **A2 and A4 are not cheap: they are PG-19 runs at `S = 80`.** v0.4's ≈985 h then omitted the E7 small-`M` model, whose precondition it had already made load-bearing.

At commodity A40/A6000 rates this is a four-figure spend. **Cut order, fixed in advance:** E5 (hygiene, §6) → A4 on PG-19, demoted to synthetic → seeds 3→2 on E4 → width `d = 384` → **`S` never**. Name the provider and reserve capacity in week 1.

### 4.2 Memory — corrected

Activation memory is set by `S × batch × L × d × layers`, not by slot lifetime. RSR adds nothing to it.

**But E3 requires `S ≳ 80` against [P2]'s starting `S = 30`.** This is the project's real memory bill and it is not optional — without it the headline gaps do not exist in the data.

**Budgeting protocol (E0c, week 1).** Measure peak resident memory at `S ∈ {30, 50, 80}` × `d ∈ {128, 256}` at `M = 40`, fit the trend, and fix the maximum feasible `(S, d, batch)` triple before any science runs. If `S = 80` does not fit at any usable width on 64 GB, **`S` wins and `d` is cut** — the experiment is defined by `S`, and width is only hygiene (§6).

### 4.3 Signal — μP

**Three bills, all of which fail silently:**

1. **The value head is readout-like — and "readout-like" is not a prescription for a bilinear form.** Write it down.

   Read `ψ̂ = s_iᵀWc_t` as two stages: `h = Wc_t` is a hidden matrix producing Θ(1) coordinates, and `s_iᵀh` is a dot product of two `d`-dimensional Θ(1) vectors. That dot product is **Θ(√d) at init** (uncorrelated) and **Θ(d) after training** (once `s`, `W`, `c` are correlated). μP is governed by the trained regime, so the output needs an explicit **`1/d` multiplier**.

   | Param | Init | Multiplier | LR |
   |---|---|---|---|
   | `W` (`d×d`, bilinear) | `Var = 1/d` | **`1/d`** on the bilinear output | hidden-matrix rule (`∝ 1/d`) |
   | `u` (`2d → 1`, linear) | `Var = 1/fan_in` | **`1/fan_in`** (output rule) | output rule (`∝ 1/d`) |

   > Do not justify the multiplier by "the form sums `d²` terms and scales as `Θ(d)` under standard init." At init it is `Θ(√d)`; `Θ(d)` is the *correlated* regime. The prescription is the same either way, but the wrong argument gives the wrong multiplier for a head of slightly different shape, which is exactly how this failure mode propagates.

   Both live in their own μP parameter group, separate from the transformer. **E0a must check the coordinate scale of `ψ̂`'s scalar output specifically**, not only transformer activations — this is the failure that surfaces in week 9 with no error message.
2. **μP transfers across width only.** `M`, `S`, and depth are non-width axes and must be fixed within each comparison (§5.1).
3. **Recurrence through a retained graph is untested territory for μP.** E0a is a check, not a proof. **Run it twice — bare TG, then TG with the value head attached.**

**Rejected: Muon.** μP and Muon are both width-correct parameterizations; stacking them double-corrects, and the matrix-method advantage under equally-tuned baselines is small and shrinking with scale. One correction: μP + AdamW.

### 4.4 Iteration

[P2] reports 21 sentence-steps/sec on a single A40 *with* the retained graph. A 30M-token corpus is ≈1.2M sentence steps, ≈16 h per run at A40 speed. **MPS will be materially slower** — no fused attention kernel, large retained graph.

**Rent compute in week 1, not week 8.** A throughput measurement taken in week 1 and left unused for seven weeks is not a plan. The Mac Studio's role is the coordinate check, the reduction test, the synthetic corpus, and interactive debugging. The corpus runs go to rented hardware from week 3.

### 4.5 Attribution

| Knob | Disposition | Reason |
|---|---|---|
| `γ` | **Swept {0.5, 0.9, 0.97} on synthetic only, then frozen.** `γ = 0` is a **control arm, not a sweep value** — carried into E1 and E3 alongside the frozen `γ`. | Constraint `1/(1−γ) ≤ min(S, A_max)` binds on **`A_max`, not `S`** — horizons are 2 / 10 / 33 / 100 at γ = 0.5 / 0.9 / 0.97 / 0.99, and `A_max ∈ {16, 32, 64}`. So **`γ = 0.97` is legal only at `A_max = 64`; `γ` and `A_max` are not independent and must be swept jointly.** `γ = 0.99` stays dropped: horizon 100 exceeds `S = 80` itself. v0.4's `γ ≤ 0.9` left an order of magnitude of horizon unused at `S = 80`. **`γ = 0` isolates context conditioning from lookahead (§2) and is the only arm that licenses the word "future" in the abstract.** |
| `β` | **Swept {0.01, 0.1, 1.0} on synthetic only, then frozen** | Collapses the corpus sweep from 27 configs to 3 |
| `A_max` | Swept only where `S` makes it bind | §3.6(a) |
| `A_detach` | Ablation arm A4 only | Not a tuned hyperparameter; orthogonal to retention |
| `γ_b` | **Derived in E0e** from `b_max / (0.25·E[lifetime])`, then swept in A5 | **Frozen at 0.001 the section was inert — 0.08 SD maximum over a full stream (D-1).** A constant that cannot move the decision it governs is not a frozen hyperparameter, it is a disabled mechanism. |
| `b_max` | Frozen 1.0 | Interpretable as one SD because `ψ̂` is z-scored |
| `ν` (redundancy) | **Swept once on synthetic** | D-3. `ν = 0` is also part of the §3.7 reduction condition |
| `λ` (return) | Frozen 1.0 (Monte Carlo) | TD(0) is A8; sweep `λ` only if MC variance is limiting (§3.3) |
| `τ` | **Set from E0e measurement** | Never freeze an unmeasured constant |
| `K` | **`≥ max target gap`: 64 on PG-19, 40 on synthetic** | Not `M`. `K = M = 40` censors exactly the `(40, 64]` events E3 measures (self-audit, §3.4) |
| `λ_shadow` | Frozen 0.5, checked once in A7 | Counterfactual targets are estimates; down-weighting them is a stated judgement, not a measurement |

**Rule:** any result not attributable to a single named mechanism does not go in the writeup.

---

## 5. Configuration

### 5.1 Model and memory capacity

| Parameter | Value | Note |
|---|---|---|
| Layers | 12 | [P2] |
| `d_model` sweep | 128, 192, 256, 384 | ≈0.3M–21M non-embedding params |
| Base width for tuning | 128 | Coordinate-check anchor |
| Gestalt layer | 7, `<EOS>` position | [P2] |
| `L_max` | 64 tokens/sentence | [P2] |
| Optimizer | AdamW, μP-parameterized | Value head in its own μP group |
| Tokenizer | GPT-2 BPE | Shared across corpora and baselines |
| Value head | Bilinear `s_iᵀWc_t + uᵀ[s_i;c_t]` | MLP variant = A6 |

**Memory capacity is split, not fixed globally:**

| Corpus | `M` | Rationale |
|---|---|---|
| Synthetic | **16** | Maximum eviction pressure; gap distribution is controlled exactly |
| PG-19, WikiText-103 | **40** | [P2]'s value, chosen because `40 × ~25 tokens ≈ 1024` ≈ GPT-2's context |
| **Narrative recall (E7)** | **8**, with its **own trained model** (`d = 128`, `S = 30`, PG-19), not an eval-time reconfiguration | **`M = 40` was a v0.2 error.** Human-normed passages run 10–30 sentences, so a 40-slot memory never fills, nothing is evicted, and every arm retains byte-identical contents — a tautological null. `M = 8` makes the passage exert real eviction pressure, which is also the condition under which humans show the levels effect. Sweep {8, 16} once. **Requires the mismatch control in §10.1.** |

> **`M = 16` on the corpora was a v0.1 error.** It gives an effective context of ~400 tokens, leaves a 14-step window in which to demonstrate a long-range effect, and puts every number off the map relative to [P2]. μP requires only that `M` be fixed *within* a comparison.

> **Match GPT-2's context window to `M × 25` tokens in every comparison.** Otherwise a 400-token-context TG is being compared to a 1024-token-context GPT-2 and TG will look far worse than published for reasons that have nothing to do with the mechanism.

### 5.2 Stream-length schedule — explicit per experiment

> **The largest hole in v0.1 was that `S` was never named.** Gaps longer than `S` do not exist in training data, so `S` silently caps the entire headline experiment.

| Experiment | `S` | Rationale |
|---|---|---|
| E0a–E0e | 30 | [P2]'s starting curriculum; diagnostics only |
| E1, E2 (synthetic) | 48 | Generator produces gaps up to 40; `S` must exceed max gap |
| E3 (PG-19, headline) | **80, fixed, no curriculum** | Required for gaps `k ∈ (40, 64]` to exist at all |
| E4 (WikiText) | 30 → +12/5 epochs | [P2]'s curriculum exactly, for comparability |
| E7 (narrative recall) | Passage length, uncut | Human-normed passages are short. **Evaluation only** — the model is trained on PG-19 and probed here, which is what makes §10.1's mismatch control necessary. |

Activation memory is budgeted from `S` in E0c (§4.2). **If `S = 80` does not fit, `S` wins and width is cut.**

**Fill fraction.** Per §3.2.1, report `mean(|memory_t|/M)` for every experiment: ≈0.76 at `M = 40, S = 80`. Under the target rescaling every step contributes unbiased signal, so this is a descriptive statistic, not a discount on the usable data — which is why the rescale was chosen over the mask or the weight.

### 5.3 Corpora

**1. Demand-controlled synthetic — the gate.** Documents of `L` sentences; a fact asserted at *i* is queried at *i+k*, with *k* from a mixture of short-range geometric and heavy tail, max gap 40. Ground-truth discounted demand is known per slot per step, so `ψ̂` is measurable against truth and eviction regret against oracle — directly, not through perplexity. Cheapest and most diagnostic artifact in the project.

**2. PG-19 (30M-token subset) — where the claim lives.** Book-length documents; entities reintroduced after long, naturally distributed gaps. Cite [P8], which introduced the benchmark and whose answer to this problem is "compress rather than evict" — address that in related work.

**3. WikiText-103 (30M-token subset) — comparability only.** Reproduce TG vs GPT-2 at `M = 40` with [P2]'s curriculum. **A null here is the expected result:** encyclopedic dependencies are local, future-demand is near-flat, and FIFO is near-optimal when everything decays uniformly.

**4. Human-normed narrative passages — the cognitive-science arm.** See §6 E7 and §10.

> **The stimulus set must be named and confirmed in week 1 (E0g).** v0.2 scheduled a *primary* experiment for week 8 against a dataset whose existence was an open question in its own §14. Candidates in order of preference: naturalistic free-recall corpora with event-level annotations (Chen et al. 2017 *Sherlock* recall data; the *Narratives* collection), which give per-event recall probability over passages long enough to exert memory pressure; then Thorndyke (1977) story-grammar materials; then Kintsch & van Dijk recall protocols. **Verify availability, licensing, and whether importance is scored independently of serial position before week 2, and report what was found regardless.** If nothing suitable exists, E7 is cut and falsifier 4 (§2) is withdrawn rather than left unfalsifiable.

### 5.4 Baselines

| Baseline | Role |
|---|---|
| GPT-2, context matched to `M × 25` | Floor |
| TG / FIFO | Identical to RSR under §3.7 reduction |
| LRU | **The recency policy. Failing to beat it = §7.1 vacuity, stated behaviourally.** |
| **H2O — accumulated attention [P6]** | **The bar.** RSR's delta is *predicted future* vs *accumulated past* demand. **Implement both halves — heavy hitters *and* the recent window, at H2O's 50/50 split of `M`, stated in the paper.** A heavy-hitter-only H2O is a crippled comparator and a reviewer will catch it. |
| **RSR, `γ = 0`** | **The separating control.** Context conditioning without lookahead (§2). Distinguishes relevance-to-now from prediction-of-future. |
| **Expire-Span — learned soft expiry [P7]** | **The referendum, not a comparison.** Retention becomes differentiable, so the LM loss trains it directly and §3.2–3.4's apparatus is unnecessary. **Runs at the week-4 gate on synthetic. No demotion path** (D-4, falsifier 6). |
| **Leading-edge strategy [P11]** | **The named ancestor.** A hand-specified retention rule — keep the most recent propositions plus those highest in the macrostructure, with reinstatement search on failure. It is a rule, not a model: an afternoon to implement, and the direct comparator for §2's question. |
| Random eviction | Sanity floor |
| **Oracle demand** | Upper bound. Synthetic: exact. PG-19: offline via the shadow-buffer machinery (§3.4). |

**Run the oracle first on each corpus as a feasibility probe (E-feas).** If oracle ≈ FIFO, that corpus cannot exhibit the effect. One day, saves six weeks.

### 5.5 Metrics

| Metric | Role |
|---|---|
| **Reintroduction loss vs gap `k`** | **Primary.** Loss on sentences reintroducing an entity last mentioned `k` steps prior, bucketed. |
| **Oracle regret** (synthetic) | **Primary.** Ground-truth demand available. |
| **Human recall-gradient agreement** (§10) | **Primary.** The cognitive-science claim. |
| Partial ρ(eviction score, age) controlling for content | **Gate.** §7.1 |
| ρ(`r_i`, LOO Δloss) | **Gate.** §3.2.1 |
| Attention-share Gini across slots | Collapse monitor |
| Reversal margin | Secondary, [P2]'s probe |
| Test PPL, lexical tokens only | Comparability |
| `L(N)`, `L(D)` fits | **Hygiene, not a claim** (§6) |

---

## 6. Experiments

**Nothing after E2 is attempted before E2 passes.**

| # | Experiment | Prediction | Kills it? |
|---|---|---|---|
| **E0a** | μP coordinate check — bare TG, then TG + value head | Activation RMS width-invariant through the recurrence | **Yes** |
| **E0b** | Reduction test vs TG, separate RNG for `φ` | Bit-exact identical loss curve | **Yes** |
| **E0c** | Memory/throughput at `S ∈ {30,50,80}` × `d ∈ {128,256}`, `M = 40` | Fixes max feasible `(S, d, batch)` | Resizes everything |
| **E0d** | `r_i` validation vs leave-one-out Δloss, held-out subsample | High ρ. **If not, LOO is truth and `r_i` is a confound.** | **Yes** |
| **E0e** | Distribution of `ū` on a FIFO run | Sets `τ` | No |
| **E0f** | Verify [P5]–[P14] against primary sources | Citations hold | No, but it is an afternoon |
| **E0g** | **Name and obtain the E7 stimulus set** (§5.3) | Recency-controlled recall norms exist and are usable | **Yes, for E7** — if none exists, E7 is cut and falsifier 4 withdrawn |
| **E0h** | Regress `ψ̂(γ=0)` on current cross-attention logits (§2) | Not collinear | **Yes** — collinear means the head reproduces the forward pass |
| **E0i** | **Coref/mention pipeline over the PG-19 subset, offline, CPU.** Histogram of reintroduction gaps and events-per-bucket at `S = 80`, against a **pre-registered** minimum | `(40, 64]` adequately populated | **Yes — kill gate.** Only 16 stream positions can host a `k = 64` gap. One CPU-day instead of 576 GPU-h. |
| **E-feas** | Oracle vs FIFO, offline, per corpus | Headroom on synthetic and PG-19; little on WikiText | **Yes, per corpus** |
| **E1** | Synthetic: RSR vs FIFO / LRU / **H2O** / **Expire-Span** / **leading-edge** / **RSR(`γ=0`)** / random / oracle, plus the `ν` sweep | **RSR > RSR(`γ=0`) > H2O**, and **RSR ≥ Expire-Span**. If RSR ≈ RSR(`γ=0`) the win is context relevance, not prospection (falsifier 3b). If Expire-Span ≥ RSR, **stop and write a different thesis in week 5** (falsifier 6). | **Yes — kill gate only** |

> **E1 is a kill gate, not a result (D-9).** The synthetic generator constructs the gap distribution that makes lookahead pay, so `RSR > RSR(γ=0)` there shows the estimator can learn a structure that was built in. **That is evidence about the optimizer, not about language.** The claim lives or dies on E3 and E7. The week-4 decision point can only ever return a red light or a *permission to continue* — never a green light on the hypothesis. Stating this matters because a project that forgets it spends 768 GPU-hours on a mechanism that works on its own test harness.
| **E2** | Vacuity gate on full eviction score | Partial ρ < 0.7; content+age ≫ age-only | **Yes** |
| **E3** | PG-19 at `S = 80`: reintroduction loss vs `k`. Arms: RSR, FIFO, H2O, **RSR(`γ=0`)** | RSR flatter than FIFO, H2O **and RSR(`γ=0`)** for `40 < k ≤ 64`; no PPL regression. **Lookahead should matter *more* at large `k` — that interaction is the strongest form of the result.** | No |
| **E4** | WikiText comparability, `M = 40`, [P2] curriculum | RSR ≈ TG. **Null expected.** | No |
| **E7** | **Human narrative-recall gradient** (§10), at **`M = 8`**, controls **H2O and LRU** (not FIFO), after the capacity-mismatch precondition | RSR's retention tracks the levels effect; H2O and LRU do not, after partialling out serial position | No — but it is the difference between a cognitive-science result and an engineering note |
| **E6** | Reversal probe | Margin improves with retention | No |
| **E5** | μP width sweep, Kaplan fits | **Hygiene.** See below. | No |
| **A1–A7** | No bias loop / `A_max = M` / content+age head / `A_detach` / bias sensitivity / MLP head / shadow-buffer params | §4.5 | No |

> **E5 is demoted.** A Kaplan-style fit over a 3× width range (128→384) at ≤21M parameters **will not resolve a slope difference from an intercept shift.** v0.1's falsifier 4 was not a falsifier — it was the near-certain outcome. The μP work stays, because it is how you tune once instead of four times, but the scaling-law headline was the most expensive and least defensible part of the project. **Report the fits as hygiene; make no scaling claim.**

---

## 7. Predicted failure modes

### 7.1 The eviction score is recency — most likely

**Gate on the full score `z(ψ̂) + b`, not on `ψ̂` alone** — given the `ū` age confound (§3.5), gating on `ψ̂` measures the wrong object.

**Three tests, all cheap:**
1. **Partial correlation**, not raw ρ: does content carry information about future retrieval *beyond* age? **Gate at ρ < 0.7.** v0.1's 0.9 was far too permissive — Spearman 0.85 with age is essentially recency.
2. **Age-only value head** as an explicit arm. If content+age ≈ age-only, content contributes nothing, in one cheap run.
3. **RSR must beat LRU.** LRU *is* the recency policy.

Report the number regardless of outcome.

### 7.2 Censored feedback

The policy trains only on data it chose to collect; early mistakes are self-confirming and invisible. Addressed by the shadow buffer (§3.4). **Residual risk:** shadow targets are counterfactual estimates, not observations — a slot's would-be attention in a memory it is not actually in is an approximation. Report the gap between shadow-estimated and oracle demand on synthetic, where both are available.

### 7.3 `r_i` measures the wrong construct

Addressed by norm-weighting and E0d. **Residual risk:** norm-weighted contribution is still correlational. LOO is the causal quantity and is only affordable on a subsample.

### 7.4 Retention collapse

Addressed by §3.5. **Residual risk:** the bias loop is a control system with no convergence proof in this setting. Monitor attention-share Gini; rising Gini with flat loss is collapse in progress.

### 7.5 RSR ≈ H2O

Not a bug — a result. Predicting the future adds nothing over measuring the past. **Must be known by week 4** (E1), which is why H2O is in the synthetic baseline set rather than added late.

### 7.6 `S = 80` does not fit

Then E3 cannot be run as specified. Fallback: cut width to `d = 128`, then reduce `A_max` to 40 and target gaps `(16, 40]` at `M = 16` on PG-19, accepting reduced comparability. Decide in week 1 from E0c, not in week 6.

### 7.7 Estimator variance (was: TD instability)

**Mostly retired by D-8.** Under the Monte-Carlo default there is no bootstrap, no moving target, and no divergence risk from a representation that is itself training. The residual risks are the ones MC trades for: **return variance** over an 80-step horizon, and the censoring already handled by the shadow buffer (§3.4, §7.2).

Monitor return variance per bucket; if it limits learning, move `λ` below 1 before touching `β`. **The old failure mode returns only inside ablation A8**, where TD(0) is run deliberately — clamp `ψ̂`'s output and monitor the TD residual there, and there only.

---

## 8. Timeline — 12 weeks

| Weeks | Work | Gate |
|---|---|---|
| 1 | **Protected:** E0a (×2, incl. `ψ̂` output coordinate check), E0c memory/throughput at `S = 80`, E0d `r_i` validation, **E0g** E7 stimulus set, **E0i** reintroduction histogram (CPU). **Reserve 4 GPUs against §4.1, not just hours.** | **E0a, E0d, E0g, E0i** |
| 2 | **Moved here from week 1 so week 1 is a week:** E0e (`ū` distribution → `τ`, `E[lifetime]` → `γ_b` and EMA half-life), E0f citations. Implement §3.2–3.5 + shadow buffer + warmup + redundancy term. E0b reduction test. Synthetic generator. | **E0b** |
| 3 | E-feas oracle probe, all corpora. Implement H2O (**both halves**), **Expire-Span**, and the **leading-edge strategy**. E0h collinearity check. | **E-feas, E0h** |
| 4 | **E1 synthetic — full baseline set including H2O, Expire-Span, leading-edge, RSR(`γ=0`), `ν` sweep. E2 vacuity gate. A5 `γ_b` sweep + decision attribution. Freeze `γ`, `β`, `ν`.** | **E1, E2 — the decision point, and the §16 release gate** |
| 5–7 | E3 PG-19 at `S = 80`, four arms, **on ≥3 GPUs continuously**. A1, A2 ablations. Train the `M = 8, d = 128` model for E7. | — |
| 8 | E7 on its own `M = 8` model — in-distribution, so the mismatch check is a robustness report rather than a gate. E4 WikiText comparability. | — |
| 9–10 | E5 width sweep (hygiene). Final runs. | — |
| 11 | E6 reversal probe. A3–A7. | — |
| 12 | Writeup including §11 related work and §13. | — |

Five gates inside four weeks.

---

## 9. What would make this a cognitive-science result

**As written, RSR without §10 is a cache-eviction paper** — better eviction in a bounded buffer, evaluated by perplexity and reintroduction loss, competing against H2O and Expire-Span. It might be a good one. It is not a contribution to the study of memory.

Two additions change that, and both are cheap.

---

## 10. The behavioural signature and the biological case

### 10.1 The levels effect — E7

[P11]: people reading narrative retain propositions high in the macrostructure and lose peripheral detail, **largely independently of recency**. That is the human behavioural signature of exactly what RSR proposes — retention by structural importance rather than by position.

**Experiment E7.** Take narrative passages with established human recall norms (**stimulus set confirmed in E0g, §5.3**). Run the PG-19-trained model over them **at `M = 8`** (§5.1) so that eviction pressure exists. Ask whether RSR's retention choices track the human recall gradient. Does RSR keep the plot-central sentence and drop the descriptive aside, the way people do?

| Measure | Definition |
|---|---|
| Agreement | Rank correlation between per-sentence survival time under RSR and normed human recall probability |
| **Controls** | **H2O and LRU. Not FIFO** — under FIFO, survival time is the deterministic function `min(M, S−i)` of serial position, so the raw correlation re-measures position and the partial correlation zeroes it out by construction. "RSR beats FIFO" here is true trivially and means nothing. |
| Partial | Controlling for serial position — the levels effect is *not* recency, and neither should the result be |
| **Unit mapping** | Human norms are scored per proposition or clause; TG slots are SaT-segmented sentences. Aggregate propositions to their containing sentence (max or mean importance, **pre-registered**) and report the fraction of sentences carrying mixed high/low-importance propositions. Above ~30%, the correlation is between two different things and must be reported as such. |

**Two preconditions, both checked before the result is interpreted.**

1. **Capacity mismatch — introduced by the `M = 8` fix and not addressed by it.** The model is trained at `M = 40` and evaluated at `M = 8`. Its cross-attention, its learned memory gates ([P2] App. C, which grow over training and are larger in deeper layers), and everything else calibrated over 40 slots are off-distribution at 8. **Any RSR-vs-H2O difference at `M = 8` could be a distribution-shift artifact rather than a retention difference.** Precondition: run TG/FIFO on the E7 passages at `M ∈ {8, 16, 40}` and confirm perplexity degrades gracefully. If it collapses at 8, E7 requires either brief fine-tuning at `M = 8` or a separately trained small-`M` model — **and that cost is not in the §4.1 budget.** Decide in week 1 from E0g, not in week 8.
2. **Comprehension floor.** A ≤21M-parameter model on 30M tokens may not represent the passage well enough for its retention choices to mean anything. Does next-sentence loss distinguish plot-central from peripheral content at all? **If not, E7 measures noise and the null is reported as uninformative, not as evidence against the levels effect.**

> This is the difference between *"my eviction policy has lower perplexity"* and *"a policy trained on predicted retrieval demand reproduces the human recall gradient over narrative, and FIFO does not."* The second is a result. The first is an engineering note.

### 10.2 Retroactive prioritization — the framing, and the answer to an objection

v0.1's open question 3 asked whether retention should be salience-weighted *at encoding* (as in [P1]) or demand-weighted *after* encoding, and framed the latter as a weakness of RSR. **It is not a weakness. There is direct evidence for retroactive prioritization in human memory.**

- [P12] Dunsmoor et al. (2015): memories for stimuli encoded *before* a category became behaviourally significant are selectively strengthened afterward. Braun, Wimmer & Shohamy (2018): the analogous effect for reward.
- [P13] Frey & Morris, synaptic tagging and capture: the cellular mechanism — an encoded trace can be held in a tagged, labile state and later stabilized if subsequent events establish its relevance.

**The structural mapping, which v0.4 had and did not state.** Because `ψ̂(s_i, c_t)` is **recomputed at every step as `c_t` moves**, a stored trace's retention value *rises* when the discourse later turns toward it. The gestalt is the tag; re-scoring under new context is the capture. **No accumulated-past-attention method has this property at all** — H2O's score is a monotone accumulation that cannot be revised downward by later context, let alone upward by it.

> *Attribution: this mapping was supplied by review, not derived here. It is therefore a correction like any other and — per §15.2 — cannot serve as the author's original contribution, however well it fits.*

**This is a biological existence proof for the thing RSR does and encoding-time salience does not.** It moves the contribution from "an SR pointed somewhere unusual" to "a computational account of prospective retention, with a behavioural signature to test against."

### 10.3 On circularity

v0.1's open question 1 worried that deriving demand from the model's own attention is circular. **Overstated.** The brain has no oracle on future retrieval demand either; it uses proxies — arousal, novelty, reward, relevance — that correlate with future need. A learned estimator of realized demand is a proxy of the same type, learned rather than evolved.

**The defensible version of the circularity worry is the censored-feedback problem (§3.4, §7.2)**, which is a real loop in the learning signal and which v0.1 did not notice.

---

## 11. Related work — required

v0.1 had FIFO, LRU, and random as baselines. That is a floor, not a bar.

| Work | Relation to RSR |
|---|---|
| **Leading-edge strategy, Kintsch & van Dijk 1978 [P11]** | **The named ancestor, and §2's framing.** The 1978 model contains a capacity-limited short-term buffer with an explicit retention policy — keep the most recent propositions plus those highest in the macrostructure, discard the rest, with reinstatement search when a needed proposition has been dropped. That is hand-specified retention by structural importance over a bounded buffer, validated against human recall. **RSR is a learned version of it.** v0.4 cited [P11] only for the behavioural effect, which a committee in this area catches in the first ten minutes and reads as not having read one's own citation. **Implemented as a baseline, not merely cited.** |
| **H2O [P6]** | Evicts by *accumulated attention* — RSR's reward signal used directly as a policy. **The secondary delta is predicted-future vs accumulated-past. State it as secondary.** |
| **Expire-Span [P7]** | Learned per-memory retention span in a transformer, long-range LM evaluation. Nearest neighbour. |
| **Scissorhands [P10]** | Persistence-of-importance hypothesis — directly tests whether past attention predicts future attention, which is RSR's empirical bet. Cite as the reason H2O is a strong baseline. |
| **DNC [P9]** | Usage-based allocation with least-used writes. Closest ancestor; state how RSR differs (predicted future demand vs observed usage). |
| **Compressive Transformer [P8]** | Introduced PG-19. Its answer is "compress rather than evict" — address directly. |
| Adaptive Attention Span; Memorizing Transformers ([P2] ref 25) | Adjacent design space |

---

## 12. Provenance note on the mnemonic-traditions material

v0.1 contained a §2 arguing that Vedic *pāṭha* recitation, ring composition, and liturgical cycles converged on retention-by-anticipated-recurrence.

**Removed from the specification.** In a model spec it reads as post-hoc justification, and the table asserted a convergence claim that stating "this is not evidence" does not neutralize. Ring composition exists for performance and structural reasons unrelated to buffer eviction. §3 stands on [P3] and [P2] without it.

**Retained, in one paragraph, in the discussion section of the final writeup, labelled as origin rather than support.** Where an idea came from is a legitimate thing to state; it simply cannot do argumentative work. One deferred item remains recorded: bidirectional rehearsal of relational pairs (*krama*, *jaṭā*, *ghana*) as a replay curriculum against the reversal curse. Out of scope for v0.2.

---

## 13. Where the evidence stops

*Required in the final writeup.*

- **[P1] is a preprint; [P2] is under review.** Both may change materially.
- **`M` differs by corpus** (16 synthetic / 40 corpora). No cross-corpus comparison of absolute numbers is valid.
- **`S = 80` for E3 exceeds [P2]'s trained curriculum.** TG's own behaviour at that depth is uncharacterized; the FIFO baseline must be re-run at `S = 80` and may itself degrade.
- **μP through a retained-graph recurrence is not established in the literature.** E0a is a check for this configuration at these widths, not a general result.
- **No scaling claim.** E5 is hygiene (§6). The width range cannot resolve slope from intercept.
- **`r_i` is norm-weighted attention, validated against LOO on a subsample only.** Correlational, not causal, outside that subsample.
- **Shadow-buffer targets are counterfactual estimates**, not observations.
- **Scale.** ≤21M non-embedding parameters, ≤30M tokens. Nothing licenses a production-scale claim. *(v0.1 said 85M — the width sweep tops out near 21M; an 85M final run happens only if compute allows.)*
- **Hard eviction is a poor idealization of the phenomenon.** Human forgetting is graded, cue-dependent, and interference-driven; nothing in this tradition deletes. `argmin`-delete is inherited from TG and the §3.7 reduction depends on it, but "which slot is deleted" is a weaker dependent measure than "how much of each slot survives," and a reviewer from this field will ask why a model of memory has a delete instruction. **The graded version — soft retention weights, or [P8]'s compress-rather-than-evict — is the obvious successor and is named as such rather than discovered in Q&A.**
- **Single-seed risk.** ≥3 seeds on E1 and E3; **2 on E7** (§4.1) — stated because it was cut for budget, not because two suffice.
- **[P5]–[P14] were not verified against primary sources during drafting.**

---

## 14. Open questions

*v0.1's Q2 and Q3 are now resolved — Q2 by the context-conditioning fix (§3.2.2), Q3 by the retroactive-prioritization literature (§10.2), which answers it in RSR's favour.*

1. **Is norm-weighted cross-attention an adequate operationalization of retrieval demand,** given that the causally correct quantity (leave-one-out Δloss) is affordable only on a subsample?
2. **Does the levels effect have a recency-controlled operationalization** suitable for E7, or do the classic norms confound structural importance with serial position? **No longer a standing question — it is resolved or E7 is cut, in week 1 (E0g).** A primary experiment cannot be scheduled against a dataset that may not exist.
3. **Is the shadow buffer a fair counterfactual?** A slot's would-be attention in a memory it is not in is an estimate. Is there a better correction for censored feedback in this setting?
4. **Does prospective retention predict anything the persistence-of-importance hypothesis [P10] does not?** If past attention predicts future attention nearly perfectly, RSR's delta over H2O is small by construction and the interesting regime is wherever that prediction fails. Where does it fail?
5. **Bidirectional rehearsal.** Is there existing work treating mnemonic traditions as evidence about retention schedules rather than as cultural artifacts?

---

---

## 15. The author's contribution — deliberately unfilled

*This section is a placeholder and must not be filled by a reviewer, an advisor, or a model.*

Across v0.1 → v0.4 this document has become correct largely where it was corrected. The degenerate TD fixed point, the inverted CLS mapping, the stream-length hole, the H2O bar, the `γ = 0` control, the E7 capacity bug — every one arrived from outside. That is a good revision record and a poor research record, and the difference matters here because the second is what is being assessed.

Two requirements follow, and neither is spec work.

**15.1 Rederivability.** Every correction in §0's changelogs must be rederivable from the primary sources without reference to the review that produced it. The test is not "can I recite that the CLS mapping was inverted" but "can I derive, from [P4] and from what `detach` does to the gradient, why it is inverted, when challenged by someone who thinks it is not." A committee will probe exactly the places where the reasoning arrived from elsewhere, because those places read differently. Work through §3.3, §3.6, §7.1 and §10 against [P1]–[P4] directly before week 4.

**15.2 One load-bearing original claim.** v0.4 needs at least one substantive commitment that was not suggested to the author, and that the author defends under push-back rather than absorbing.

The most likely site is §10.2. Retroactive prioritization currently rests on handed-over citations used as a rhetorical shield against an objection. Built out from the primary literature it could instead become a *positive* prediction that distinguishes RSR from every baseline in §11: if human memory retroactively strengthens traces whose relevance is established by *later* events, then a retention policy with lookahead should show a specific, measurable signature that accumulated-past-attention methods cannot — and that signature should be derivable in advance, stated as a falsifiable prediction, and tested in E3 **without new runs**. Neither the review nor this document has derived it.

**Two things offered in the v0.4 review are disqualified from filling this slot, and it matters that they are named.**

- **The redundancy/marginal-value term (D-3, §3.4)** is a substantive design commitment, but it was supplied. It is a correction.
- **The tag-and-capture structural mapping (§10.2)** is the sharpest paragraph in §10 and it was supplied too.

Adopting either as the original contribution would reproduce, one level up, exactly the failure this section exists to name. Both are kept as corrections and attributed as such.

**What remains open is a pointer, deliberately left unanswered here:**

> Which slots is H2O *structurally incapable* of retaining, regardless of tuning? Characterize that population. What is the corresponding human paradigm? And should the size of the RSR–H2O gap vary with a quantity already measurable in E3?

That pointer is not an answer and must not be treated as one. **If working it through takes ten minutes, it was always the author's. If it takes a week, that week is the thesis.**

**Until §15.2 contains something, this specification is a well-audited implementation of other people's objections.** That is worth a working model. It is not yet worth a thesis.

**15.3 One thing that is already the author's, and should be noticed.** §4.3's μP footnote — why the `Θ(√d)`-at-init versus `Θ(d)`-when-correlated distinction matters, and why the wrong argument yields the wrong multiplier for a differently-shaped head — was derived here, not supplied. It is one worked example of the rederivability §15.1 asks for. The standard is not that nothing may be corrected; it is that the load-bearing claim must survive push-back in the author's own hands.

---

## 16. Release conditions

**Approved:** weeks 1–4. E0a–E0i, E1, E2. Roughly $100 of compute, no reputational exposure.

**Not approved:** the PG-19 block, the four-figure spend, and everything downstream.

| # | Condition | Source |
|---|---|---|
| 1 | E0i histogram shows adequate `(40, 64]` population against a pre-registered threshold | D-2 |
| 2 | `γ_b` rederived from measured lifetime; §3.5 demonstrated to move the argmin | D-1 |
| 3 | Expire-Span implemented and run on synthetic **at the week-4 gate** | D-4 |
| 4 | E7 small-`M` model funded (§4.1) or E7 cut and falsifier 4 withdrawn in week 1 | D-5 |
| 5 | Marginal/redundancy scoring specified and ρ(score, LOO Δloss) reported with and without | D-3 |
| 6 | §11 rewritten with the leading-edge strategy as named ancestor and implemented baseline | S-1 |
| 7 | **§15.2 non-empty, derived by the author, defended under push-back in person** | §15 |
| 8 | Parallel GPU capacity reserved, not just total hours priced | §4.1 |

Conditions 1–4 and 8 decide whether the project finishes. **Conditions 5, 6 and 7 decide whether it is worth finishing.**

---

*Where the record is thin — μP under recurrence, the `γ_b` timescale, the adequacy of `r_i`, the fairness of shadow targets, the fill-level rescale's residual bias, and whether greedy argmin over marginal scores approximates the submodular optimum at all — this document says so rather than smoothing it over.*
