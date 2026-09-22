# RESEARCH CONTEXT — Retrieval-Successor Retention

**The single orientation document for this project.** It exists so that a session which reads
nothing else still knows what is being claimed, what has been decided, what has been measured, what
is broken, and what it is forbidden to touch — and so that none of that drifts from the original
spec and research as sessions turn over.

**Written:** 2026-09-18, against `d1c221f`, by a session that read every document in the table in
§16, both primary-source PDFs (`arXiv:2512.25026v2`, McClelland & Rogers 2003), and re-ran the suite.

> 🔴 **This file is an index with the load-bearing facts inlined. It is not authoritative over the
> documents it indexes.** Where it disagrees with `docs/spec-corrections.md`, an ADR, a
> `preregistration/` file or a ledger, **those win and this file is stale text** — fix it.
> The one thing it is authoritative about is §17, the defects found in the record itself.

---

## 0. How to use this file

Read §1–§3 always. Then jump:

| You are about to… | Read |
|---|---|
| implement anything | §4, then `docs/spec-corrections.md` **in full** |
| quote a number | §9, §11 — and resolve it to a ledger key or a command |
| run an experiment | §6, §10, §14 |
| change a constant, threshold, bucket edge or arm | §12, §13 — almost certainly **stop** |
| know what to work on next | **`docs/ROADMAP.md`** — sprints, gates, and the AttentionTrace capture bridge that gates E0d/E0e/E0h/observe/H2O/oracle/shadow at once |
| write a brief | §14, `docs/lab-notes/overnight-2026-09-19.md` §6 |
| reason about hardware or cost | §5 |

🔴 **Three things are never yours.** Spec §15 (§13.1), the E0i signature (§13.2), and the six owner
decisions in §12. They are not stylistic preferences; each has a stated reason.

---

## 1. The claim

> **Kintsch & van Dijk (1978) specified retention by structural importance over a capacity-limited
> discourse buffer — the leading-edge strategy — and it reproduced the human recall gradient.
> Does a system trained only to predict the next sentence *discover* that policy? RSR is the test.**

**Mechanism.** Thought Gestalt (TG) is a recurrent transformer that compresses each sentence to one
vector (a *gestalt*) and writes it to a fixed working memory of `M` slots; later sentences reach
earlier ones only via cross-attention to that memory. When memory is full, **TG evicts the oldest
slot.** Nobody chose that on purpose; it is a default. RSR replaces that one line: evict the slot
with the lowest **predicted future retrieval demand, conditioned on where the discourse is now**.

🔴 **The primary claim is cognitive, not engineering.** The engineering delta — *predicted future*
demand beats *accumulated past* demand — is **secondary, and the spec says so** (§2). **Drifting
toward the engineering framing is itself drift, even when every number is honest.** Under the
cognitive framing a null on E3 is survivable and E7 is the point; under the engineering framing the
project is a cache-eviction paper at a scale no systems audience will care about (§9 of the spec).

### The mechanism names two things and the experiment must separate them

1. **Context conditioning** — scoring a slot by relevance to where the discourse is *now*.
2. **Temporal lookahead** — the discounted sum over *future* steps.

H2O controls for neither individually. The separating control is **`γ = 0`**, required in E1 and E3.
**The spec's own prior is that (1) does the work** — `ψ̂ = s_iᵀWc_t` is architecturally the same
functional form as the cross-attention logit, so it can fit "what is being attended to now" almost
for free. E0h is the cheaper check: regress `ψ̂(γ=0)` on the current cross-attention logits and
report R².

### Falsifiers (spec §2)

⚠️ They are numbered `1, 2, 3, 3b, 3c, 5, 6, 4` — **falsifier 4 appears last, after 6** (correction
11). §16 condition 4 cites "falsifier 4" by number. **Cite by name as well as number.**

| # | Name | Dies if |
|---|---|---|
| 1 | **Vacuity** | eviction score is a monotone function of slot age. RSR is FIFO with extra parameters |
| 2 | **No headroom** | oracle ≈ FIFO on the corpus (E-feas). The corpus cannot test the claim |
| 3 | **No prospective advantage** | RSR ≈ H2O. Past attention already contains what future demand would add |
| 3b | **No lookahead advantage** | RSR(γ=0.9) ≈ RSR(γ=0). Publishable, but §2's delta sentence is then false |
| 3c | **No mechanism at all** | `ψ̂(γ=0)` collinear with the current cross-attention logit (E0h) |
| 5 | **Redundancy dominates** | score anti-correlated with LOO Δloss wherever slots overlap (E0d) |
| 6 | **Superseded by gradients** | Expire-Span matches or beats RSR on synthetic. 🔴 **This one invalidates the design, not the result** |
| 4 | **No behavioural signature** | RSR's retention tracks human narrative recall no better than FIFO (E7). Result becomes an engineering note |

---

## 2. Sources, and what has actually been checked

`[P1]`–`[P14]` are the spec's own numbering. **Ten of fourteen were unchecked at drafting** — §1 of
the spec says so and calls the check (E0f) *"the cheapest risk retirement in the project."*

> 📌 **E0f pass 2 ran 2026-09-20. Thirteen of fourteen are now checked; [P9] (DNC) is the only one
> left.** The pass was not clean. **[P11] — Kintsch & van Dijk, which is §2's entire framing — is
> wrong in the spec in three places**, one of them load-bearing for E7's design: the levels effect
> is *not* "largely independent of recency" (the strategy emphasizes recency, and a
> levels-plus-primacy variant fits 23% worse than leading-edge on the same data); the strategy runs
> on the **microstructure** coherence graph, not the macrostructure; and it is **Kintsch & Vipond
> (1978)**, whom KvD credit. Also: §15.1 cites [P4] for an argument that is in [P2]; §10.2 omits
> that its own biological evidence is **delay-dependent and absent in immediate test**, which is
> the condition RSR operates in; and §11 is missing arXiv:2605.24585, the nearest prior work.
> **Corrections 25–30. Detail in `docs/citation-audit.md`.**

| Ref | Source | Status | Checked against primary source? |
|---|---|---|---|
| **[P1]** | Cho & McClelland (2026), extended successor-representation theory of the cognitive map | Preprint, not peer reviewed | ◐ v3 read; **Figs. 3f–g not read** |
| **[P2]** | **Borazjanizadeh & McClelland (2026), *Modeling Language as a Sequence of Thoughts*, arXiv:2512.25026v2** | Under review — **the base model** | ✅ **read in full** |
| **[P3]** | Dayan (1993), successor representation | Published | ✅ read 2026-09-20 |
| **[P4]** | Yang & Hu et al., **Tensor Programs V (μP)** | Published | ✅ read 2026-09-20. ⚠️ **§15.1 cites it for an argument that is in [P2] App. A** — μP says nothing about gradient flow through a detached write (correction 28) |
| **[P5]** | Kobayashi et al. (2020), *Attention is Not Only a Weight* | Published | ✅ checked |
| **[P6]** | Zhang et al. (2023), **H2O: Heavy-Hitter Oracle** | Published — the bar | ✅ checked |
| **[P7]** | Sukhbaatar et al. (2021), **Expire-Span** | Published — falsifier 6 | ✅ checked |
| **[P8]** | Rae et al., **Compressive Transformer** (introduced PG-19) | Published | ✅ read 2026-09-20. ⚠️ **misdated** — arXiv:1911.05507, posted **2019-11-13** (ICLR 2020), not "2020" (correction 12) |
| **[P9]** | Graves et al. (2016), DNC | Published | ❌ **the one outstanding** |
| **[P10]** | Liu et al. (2023), Scissorhands | Published | ◐ title/claim; **body not read** |
| **[P11]** | **Kintsch & van Dijk (1978); Thorndyke (1977)** — levels effect | Published — **the cognitive claim** | 🔴 **READ 2026-09-20 — AND THE SPEC IS WRONG IN THREE PLACES.** Corrections 25–27: the levels effect is not "largely independent of recency"; the strategy runs on the microstructure coherence graph, not the macrostructure; and it is Kintsch & Vipond (1978). Thorndyke is a **story grammar**, a different hierarchy, and [P11] bundles the two. **TG does not cite KvD at all** |
| **[P12]** | Dunsmoor et al. (2015, *Nature*); Braun, Wimmer & Shohamy (2018) | Published | 🔴 **read — corr. 29** |
| **[P13]** | Frey & Morris (1997), synaptic tagging and capture | Published | ✅ read 2026-09-20 |
| **[P14]** | Xiao et al. (2023), StreamingLLM — attention sinks | Published | ◐ title/claim; body not read |

⚠️ **§1's verification note stays in the spec until that list is empty.** It is not discharged, and
the spec says it *"cannot still be here at week 12."*

### An additional primary source, supplied by the owner 2026-09-18

> **McClelland, J. L. & Rogers, T. T. (2003). *The parallel distributed processing approach to
> semantic cognition.* Nature Reviews Neuroscience 4, 310–322. doi:10.1038/nrn1076.**
> Local copy: `~/Downloads/McCRogers03.pdf` (13 pp).

🔴 **It is not `[P1]`–`[P14]`, it is not on E0f's outstanding list, and the owner has not stated
what role it plays.** Whether it becomes `[P15]` is the owner's call, not an agent's. Recorded here
with what it actually contains, because three of its contents touch decisions already in this record:

| What it contains | Where it touches this project |
|---|---|
| **Complementary Learning Systems** (McClelland, McNaughton & O'Reilly, its ref 63): a slow-learning neocortical semantic system complemented by a fast-learning medial-temporal system; **catastrophic interference** is the problem CLS solves | **§3.6 deleted a CLS claim** because the mapping was inverted, and CLAUDE.md forbids calling a truncated-BPTT window "consolidation." **§15.1 requires every changelog correction be rederivable from primary sources without reference to the review that produced it.** This is a primary source for that rederivation. Agents *"may schedule the work"* — this is scheduling it, and nothing more |
| **Graded degradation**: perturbing internal representations loses **specific/idiosyncratic properties first and general ones last**, and over-attributes superordinate-typical properties (four legs drawn on a swan) | §13's stated limitation that *"hard eviction is a poor idealization … human forgetting is graded, cue-dependent and interference-driven; nothing in this tradition deletes."* This paper is the primary-source form of that limitation, and it names the named successor (graded retention) the spec already points at |
| **Coherent covariation**: properties that covary coherently across items drive larger representational change, become *central*, and are learned faster — `canary HAS wings` is learned faster than `canary IS yellow` despite equal training frequency. **Importance is learned, not innate** | The structural analogue of RSR's bet — that *what a system will need later* is learnable from experience rather than hand-specified. ⚠️ **Stated as an observation. It is not a result, not a prediction, and nothing in the spec derives from it** |
| Representations learned from **event sequences**, where *"the task is to learn to predict each step from information extracted from previous steps"* and *"the representations assigned to items are affected by the consequences of the item's occurrence"* | The same shape as TG's own objective. ⚠️ Recorded; no claim drawn |

📌 **The lineage is real and already in the record.** [P2] states it is *"inspired by the Sentence
Gestalt model [21]"* — St. John & McClelland (1990) — and **correction 24 (S-7)** already flags that
TG's unit-norm gestalts are *"a departure from the Sentence Gestalt lineage … where magnitude
carried strength,"* recorded as a question for the authors and explicitly **not a code change.**

---

## 3. The base model — and there are two of them

**The paper and the released code are different models, and the code says so.**

> *"This version differs slightly from the version of the model described in (arXiv:2512.25026). It
> achieves comparable results to those reported in the paper when trained with 12M text tokens."*
> — `jlmcc94303/ThoughtGestaltCode` README

🔴 **D-A: the CODE is the reference.** The fidelity harness compares against tensors extracted from
the released implementation, so the code is ground truth **by construction — you cannot validate
fidelity against a paper.** Every divergence found is *recorded* in `docs/code-vs-paper.md`, not
resolved in favour of whichever source reads better.

| | |
|---|---|
| Paper | arXiv:2512.25026, v1 2025-12-31, v2 2026-01-12. Nasim Borazjanizadeh & James McClelland, Stanford; *"authors contributed equally"* |
| Code | `github.com/jlmcc94303/ThoughtGestaltCode` — **JAX/Flax**, Apache-2.0, 0 stars, on McClelland's personal account, fork-by-import of `google-deepmind/nanodo` |
| Pin | `f220b1098d24a02c94907043d6205c113b31ebb6` (2026-09-02), vendored read-only at `third_party/ThoughtGestaltCode/` |
| Discoverability | **Effectively none.** The abs page carries no code link; GitHub returns 0 results for `borazjanizadeh` or `2512.25026`. It surfaced only on a repo search for `"thought gestalt"` |

### Verified firsthand in `arXiv:2512.25026v2` (this session, not relayed)

| Fact | Evidence |
|---|---|
| TG evicts FIFO | §2.2: *"if working memory is full, the oldest entry is removed."* The string `FIFO` appears **0** times; `oldest` appears 4 |
| `s_t = W_sent H^(ℓs)_iEOS`, `ℓs = 7`, `W_sent ∈ R^{d×d}`, **depth 1** | §2.2 |
| **The paper describes no normalization of the gestalt** | the string `normali` appears **0** times in 20 pages. Its line *"`s_t` lives in the same `d`-dimensional space as token hidden states"* is a claim about **dimensionality, not norm** |
| `P^(sent)` is added to **keys only**; `V_M` gets none; memory is *"ordered from oldest to most recent"* | §2.2 |
| `M = 40` chosen to approximate GPT-2's 1024-token window; corpus averages **~25 tokens/sentence** | §2.2 |
| A **scalar learnable memory gate `g_mem` per cross-attention layer**, scaling the increment before the residual add | §2.2 |
| Curriculum `S = 30`, **+12 every 5 epochs** | §2.3 |
| **Trains on WikiText-103.** `PG-19` **0** hits, `Gutenberg` **0** hits, `WikiText` 3 | §3.1 |
| Headline effect **2–4%** PPL (23.2 vs 24.0 at 50M); `m_D ≈ 1.05–1.08`; `m_N ≈ 1.33–1.42×` | §3.2 |
| Parameter sweep `N ≈ 0.34M → 21.3M`, `d_model ∈ {48, 96, 192, 384}`, depth fixed at 12 | §3.2 |
| *"Early in training, sentence representations are largely uninformative"* | §2.3 — the warmup citation holds |
| **Context seeding:** `X^(0)_iBOS ← s_{t−1}` — the previous gestalt **replaces** the static `<BOS>` embedding | §2.2. **This is a real architectural feature, not a bug** — see §10.2 |
| **EOS down-weighting:** weight 1.0 for epoch 1, then **0.05**; special-token targets excluded from reported perplexities and checkpoint selection | §2.3, §3.1 |
| App. A, verbatim | *"While the memory size M limits which sentence vectors are available to cross-attention in the forward pass, it does not limit the backward flow of gradients… Thus, the maximum backward depth is bounded by the maximum stream length S."* |

### [P2]'s ablation table (§3.5, 30M tokens, A40, sent./sec)

| Ablation | Test PPL | Throughput | N |
|---|---|---|---|
| **TG (baseline)** | **29.8** | 21 | 85.6M |
| Layer type: Self → Cross | 29.4 | 17 | 114M |
| Layer type: parallel Self & Cross | 29.7 | 17 | 114M |
| 🔴 **No working memory (all layers self-attention)** | **45.8** | 26 | 85.6M |
| **Detach sentence reps at memory write** | **35.0** | 24 | 85.6M |
| In-context working memory | 30.2 | 19 | 85.6M |
| Sentence rep from last layer (12) | 30.3 | 21 | 85.6M |
| No context seeding (static `<BOS>`) | 30.2 | 21 | 85.6M |
| No EOS down-weighting | 30.4 | 21 | 85.6M |
| No stream curriculum (fixed `S=40`) | 30.5 | 21 | 85.6M |
| Max sentence length = 32 | 30.4 | 22 | 85.6M |

🔑 **Read the first two rows against §10.3.** In [P2], removing the working memory costs **29.8 → 45.8
PPL, +54%** — the largest ablation in the paper. In this repo's trained model, deleting all 16 slots
costs **≈ 0%**. Those two numbers are not directly comparable (different corpora, scales and
objectives) and **the gap is the thing the project currently has to explain.**
*(Update 2026-09-22: the "≈ 0%" is the old unmasked-objective model. Under the masked objective the
decisive run (#34) reads the memory as live but does not show retrieval. See §10.3.)*

### The divergences that matter (`docs/code-vs-paper.md`, all marked **source**)

1. **The gestalt is L2-normalized to exactly 1.0.** `srep_BxD = raw_BxD / denom * srep_norm_target`,
   `srep_norm_target = 1.0`. So `‖s‖₂ = 1` and coordinates are `O(1/√d)`, **not `Θ(1)`** —
   which is what §4.3's μP derivation assumes. **Under the prescribed `1/d` multiplier the trained-
   regime output is `Θ(1/d)`: it decays with width, in exactly the regime μP governs.**
   🔴 **The multiplier is NOT changed. §15.3 makes that derivation the author's. E0a measures it.**
2. **An auxiliary squared-hinge penalty on the pre-normalization norm**, margin 0.1, weight 0.01,
   holding the raw norm in `[0.9, 1.1]`. **The paper is silent on it.** So the head is *also* trained
   to emit near-unit vectors before the divide — E0a's coordinate check must read **both** `raw_BxD`
   and `srep_BxD`, because a μP violation upstream of the divide is invisible downstream of it.
3. **`P^(sent)` is rank-indexed, not age-indexed** → ADR-0006, §4.4 below.
4. **Cross-attention lives on six layers, not twelve** (`('S','C')*6`), so the per-layer `r_i`
   profile has **six entries**. A twelve-entry profile is six real rows and six zeros, and the zeros
   read as a depth finding.
5. **`attn_dropout = 0.2`** → `r_i` must be collected in **eval mode** (D-F).
6. **The S_REP head is not `W_sent`** — it is LayerNorm → dropout(0.15, warm-in scaled) → MLP →
   normalize. At `srep_head_depth = 1` the MLP *is* a single Dense, so the paper's `W_sent` is
   recoverable, **but the LayerNorm and the dropout are not in the paper at all.**
7. **The sinusoidal positional encoding is itself L2-normalized.** So the positional term added to a
   key has norm 1 and the gestalt it is added to has norm 1 — **the two are the same magnitude by
   construction.** That raises the stakes on ADR-0006: a rank displacement moves a term comparable
   in size to the content itself.
8. **Gestalt layer 6 (0-indexed) == the spec's layer 7 (1-indexed).** Not a divergence. Verified
   against `srep_layer_idx`, not guessed — *"and the guess would have been right, which is exactly
   the kind of near-miss that makes guessing feel safe."*
9. **Known-broken upstream:** `tg/Rough/preprocess_corpus.py` imports a `src_recurrent` package that
   **exists nowhere in the tree**, so the corpus path is not runnable as published (RSR writes its
   own; it does tell us the segmenter is SaT `sat-3l-sm` with an explicit refusal to fall back to
   regex). And `EXPERIMENTS.md` admits the paper's **gist baseline is buggy** — the mask indexes the
   gist flag on the query axis instead of the key axis *"and so grants no cross-sentence access at
   all."* A caution against treating [P2]'s reported comparisons as settled.

---

## 4. The mechanism, as actually decided

**Read `docs/spec-corrections.md` before implementing anything. It has 24 entries and it OVERRIDES
the spec body.** The spec is on its fifth revision; several passages were superseded by a correction
and never rewritten. What follows is the decided form, with the correction that produced it.

### 4.1 Reward — what counts as "being retrieved"

```
r_i(t) = Σ_{l,h} ‖ g_mem^(l) · α_{l,h,i} · W_O^(l,h) v_{l,h,i} ‖₂        # then normalized across live slots
share_i(t) = ‖·‖_i / Σ_j ‖·‖_j
r_i(t)     = share_i(t) · |memory_t| / M                                  # share-of-M-equivalent
```

- **`W_O` is not optional** (D-6) — it is where head-specific rescaling lives.
- **`g_mem` is not optional** (correction 17 / D-E) — it is where *layer*-specific rescaling lives,
  and App. C measures the gates **growing over training**, so the weighting is **non-stationary**:
  `r_i` at epoch 1 and epoch 12 are not the same measurement. **Compute both ways** (gated and raw)
  and report both against LOO Δloss in E0d.
- **Six-entry per-layer profile**, reported once before collapsing (correction 18).
- **Eval mode** for the retention target, train mode for the LM loss — an explicit switch, never an
  ambient default (correction 20 / D-F).
- 🔴 **Rescale the target; do not mask, and do not merely down-weight.** v0.2's `|memory| < M` mask
  deletes **half of E3's** training signal and **100% of E7's**. Down-weighting a *biased* target
  reduces how hard the estimator fits it; it does not remove the bias.
- 🔴 **And do not call the rescaled target "unbiased"** (D-7). It credits absent competitors and
  biases genuinely important stream-initial slots **downward**. That is the honest description, and
  "unbiased" is the word a reviewer will test.
- **Truth rule (§3.2.1):** if `r_i` and LOO Δloss disagree, **LOO is truth and `r_i` is a confound**,
  and §3.4's redundancy term is the specified response — not a shrug.

### 4.2 Value — context-conditioned

> **A successor representation requires a state that transitions. A memory slot has no successor
> slot. The state that transitions is the reader.**

```
ψ(s_i, c_t) = E[ Σ_k γ^k · r_i(t+k) | c_t ]
ψ̂_φ(s_i, c_t) = s_iᵀ W c_t + uᵀ[s_i ; c_t]          φ = {W, u}
```

- **`c_t` is the CURRENT SENTENCE GESTALT** (correction 19 / D-C). §3.2.2's *"or a running context
  vector"* is a clean ablation, **not a branch in the critical path** — written as an enum with one
  value implemented and the other raising. Three reasons: simplest reading; both bilinear arguments
  become unit-norm so the μP analysis is single-valued; and it keeps `ψ̂` **positionally blind**,
  which ADR-0006 arrives at independently.
- 🔴 **Age is excluded from `ψ̂`.** Supplying it invites collapse onto recency and makes the vacuity
  failure mode **invisible rather than merely possible.** Age-only and content+age heads are
  separate baseline arms (A3), never this one.
- **μP:** `W` (`d×d`) init `Var = 1/d`, **`1/d` multiplier** on the bilinear output, hidden-matrix LR;
  `u` (`2d→1`) init `Var = 1/fan_in`, output multiplier and LR. **Its own parameter group**,
  separate from the transformer. ⚠️ **Do not justify the multiplier with the `Θ(d)`-at-init
  argument** — at init the bilinear form is `Θ(√d)`; `Θ(d)` is the *correlated* regime. Same
  prescription, wrong argument, and the wrong argument gives the wrong multiplier for a
  differently-shaped head. Measured: `experiments/e0a/RESULTS.md`.

### 4.3 Learning rule

🔴 **Monte-Carlo (`λ = 1`) is the default. TD(0) is ablation A8** (D-8, correction 2).

```
G_i(t) = Σ_{k=0}^{S−t} γ^k · r_i(t+k)
L_MC   = Σ_i ( ψ̂_φ(s_i, c_t) − sg[ G_i(t) ] )²
L      = L_NTP + β · L_MC
```

- **No EMA target copy of `φ` on the default path** (correction 3) — that sentence is live only
  inside A8. MC has no bootstrap, so there is no moving target to stabilize.
- 🔴 **Do not backpropagate the retention loss into the transformer or `W_sent`.** Only `φ` receives
  gradient; `c_t` enters `ψ̂` with a stop-gradient on the transformer side. ⚠️ **One of the two
  stated justifications is unsupported** — S-5: the spec attributes *"auxiliary sentence-level
  objectives are brittle"* to [P2] §4 as a measurement, but [P2] says it in **related work, citing
  the NSP literature**, and TG deliberately has **no** auxiliary loss, so it never measured one.
  The rule stands on its other reason (it preserves §3.7's reduction).
- **Reversion check against v0.1:** v0.1's degeneracy was *algebraic* — the same function of the
  same argument on both sides of a bootstrap. MC has no bootstrap, so that cannot recur by
  construction. But v0.4's defence *"there is no closed-form regression that produces this"* is
  **false under MC and is withdrawn.** MC *is* a regression. **What distinguishes it is the target,
  not the estimator class.** Anyone re-deriving this must check the target.

### 4.4 Eviction

```
i* = argmin_i [ z(ψ̂_φ(s_i, c_t)) + b_i − ν · max_{j≠i} cos(s_i, s_j) ]
```

- **Warmup:** `t < T_warm` → FIFO, `ψ̂` trains passively; `t ≥ T_warm` → the score. `T_warm` is **a
  float number of steps, one representation everywhere** (correction 21 / D-I); `T_warm_epochs`
  (FROZEN 1.0) survives only for the report.
- 🔴 **`T_warm = 0` in `reduction_to_tg()`, not `∞`** (correction 16). At `∞` the FIFO branch is
  taken forever, so **under the §3.7 reduction no eviction ever reaches the score and E0b certifies
  FIFO against FIFO** — the gate that exists to prove RSR reduces to TG passing without executing
  the thing being reduced.
- **Shadow buffer:** `K ≥ max target gap` — **64 on PG-19, 40 on synthetic. Not `K = M`**
  (correction 1). At `M = 40` with E3's window `(40, 64]`, `K = 40` **censors exactly the events E3
  exists to measure.**
- **Redundancy (D-3):** set value under LOO is **submodular**, so greedy argmin over *independent*
  scores is not an approximation to marginal value — under redundancy it is **anti-correlated** with
  it. ⚠️ See §12.4 and §10.4: `ν` has been measured and **cannot arbitrate within a near-duplicate
  pair at any `ν`.**
- **Decision attribution is required from the first policy run** — per eviction, which term
  determined the argmin. *"If `b` flips a large share, the balance controller is the policy."*

### 4.5 Anti-collapse — the gradient-free protection bias

```
b_i ← b_i − γ_b   if ū_i > (1+τ)/M
b_i ← b_i + γ_b   if ū_i < (1−τ)/M
b_i ← clip(b_i, −b_max, +b_max)
```

- **`b` never enters `ψ̂` or any differentiable path.** Balance is a zero-gradient control loop, not
  a competing objective. If `b` entered `ψ̂` an under-attended slot would receive an inflated
  predicted *demand*, not merely an improved chance of survival.
- **`ψ̂` is z-scored across live slots each step**, so `b_max = 1.0` means one SD and is interpretable.
- 🔴 **§3.5 is headed "Three corrections" and lists FOUR, numbered 1, 2, 4, 3** (correction 4). All
  four apply. Item "4" is the one that matters:
  ```
  γ_b ≈ b_max / (0.25 · E[lifetime])        order 0.05–0.1, DERIVED IN E0e, never frozen at 0.001
  ```
  At `γ_b = 0.001` with a stream-bounded lifetime, max attainable `|b|` over a slot's whole life is
  **0.08 SD** — the loop cannot control the argmin, and `b ≡ 0` is *also* the §3.7 reduction
  condition, so A5 would have reported "no effect" and the loop would have been called *unnecessary*
  rather than *absent*.
- **`ū` is an EMA rate, not a cumulative sum** — a running sum accumulates with age by construction
  and smuggles recency back in through the anti-collapse mechanism. Half-life `E[lifetime]/4`,
  measured in E0e.
- **`τ` is measured, not frozen.** Attention over slots is heavy-tailed. *Freezing an unmeasured
  constant is how v0.2-NP died.*
- ⚠️ **S-2:** H2O's own App. B.2 documents the same age bias for accumulated attention **and reports
  that the averaged-score fix degraded performance for them.** §3.5's EMA-rate fix now has a
  published counter-datapoint in the nearest-neighbour method. The cases differ (`ū` feeds a
  zero-gradient controller, not the policy) **but the difference has to be argued.**

### 4.6 Positional encoding — ADR-0006, the confound E0b structurally cannot see

Under FIFO, rank order and age order are **the same order, always**, so `P^(sent)` reads as
"recency" with no loss. **RSR can evict a middle slot, and every slot behind it moves up one rank** —
its key, and therefore its cross-attention logit against every query, changes although the slot did
not change and no time passed for it.

> **A slot's positional encoding becomes a function of which other slots the policy killed.**

🔴 **E0b cannot catch it, and not by accident of implementation: under the §3.7 reduction the policy
*is* FIFO.** No middle slot is ever evicted, no rank ever shifts. **The confound is structurally
invisible in the reduction and present in every arm the project cares about. A gate that is green
precisely where the hazard is absent is not evidence about the hazard.**

**Decision: keep rank-indexing** (it is what the code does, and D-A makes the code the reference),
and make the confound *observable*:

1. **Per-eviction rank-shift logging**, mandatory. `displacement(t) = victim_rank(t) − 1` — the
   number of **older** slots that survive an eviction FIFO would have spent on the oldest.
   ⚠️ **The first draft of this metric was wrong and the implementation caught it:** defined as "how
   many live slots had their rank changed," it reported 7 shifted slots on 100 of 100 evictions
   under the reduction — **it measured the sliding window, not the policy.** Under the shipped
   definition it is identically 0 under FIFO. Measured: `{0: 100}` under the reduction;
   `fraction_shifting = 0.93` with the learned head on. **Report the distribution, not the mean.**
2. **A `P^(sent)`-ablated arm in E1** (both FIFO and RSR) to bound the confound, read as a
   difference of differences. **It is a bound, not a decomposition** — zeroing `P` removes the
   confound *and* the legitimate positional signal.
3. **An absolute-age variant behind a config switch, default OFF**, pinned to `"rank"` in
   `reduction_to_tg()` with a test asserting it.

**Two consequences that must appear in the writeup:**
- It **lowers the E0h collinearity prior** — the logit's positional term is something `ψ̂`
  structurally cannot contain, so `ψ̂` can only fit it insofar as position correlates with content.
- 🔴 **§7.1's vacuity gate must not convict the estimator of the architecture's own contamination.**
  `r_i` is built from `α`, and `α` is a softmax over logits that already contain `P^(sent)` — **the
  retention target is itself partly a function of slot position.** A non-zero age correlation is
  *expected*. The gate is on the **magnitude** and on the two behavioural tests (age-only head,
  RSR-vs-LRU), which compare estimators fitting the *same* contaminated target and so difference the
  contamination out.

### 4.7 Reduction to exact TG (§3.7)

Set `ψ̂ ≡ −a_i`, `b ≡ 0`, **`ν = 0`**, `β = 0`, `A_max = M`, **`T_warm = 0`**, shadow buffer off,
`sent_pos_index = "rank"`, `c_t` pinned. Eviction becomes `argmin(−a_i)` = oldest slot. Bit-exact TG.

🔴 **Every term added to the eviction score must have a documented off-switch**, or §3.7 silently
stops being a reduction and E0b stops testing what it claims to test.
`tests/test_reduction.py::test_every_config_field_has_an_off_switch` fails the moment someone adds a
term without one. ⚠️ **Construct `φ` after the model, or from a separate RNG stream** — otherwise
instantiating the head consumes draws, shifts data order and dropout masks, and E0b fails for a
reason unrelated to the mechanism.

### 4.8 Baselines — a bar, not a floor

| Baseline | Role |
|---|---|
| GPT-2, context matched to `M × 25` tokens | Floor |
| TG / FIFO | Identical to RSR under §3.7 |
| **LRU** | **The recency policy. Failing to beat it = §7.1 vacuity, stated behaviourally** |
| **H2O [P6]** | **The bar.** Implement **both halves** — heavy hitters *and* the recent window, at H2O's 50/50 split. A heavy-hitter-only H2O is a crippled comparator and a reviewer will catch it (App. C.6: 2.85%→22.75% degradation) |
| **RSR, `γ = 0`** | **The separating control** |
| **Expire-Span [P7]** | **The referendum, not a comparison. No demotion path** (D-4) |
| **Leading-edge strategy [P11]** | **The named ancestor — implemented, not merely cited** (S-1) |
| Random | Sanity floor |
| **Oracle demand** | Upper bound. Synthetic exact; PG-19 offline via the shadow-buffer machinery |

🔴 **B-3 — the referendum is not a fair fight, and the unfairness runs the wrong way.** [P7] has a
ramp length `R`, a loss coefficient `α`, and **requires structured dropout**. The spec runs it
**once, untuned**, against RSR's 9-config grid plus a `ν` sweep. **An untuned Expire-Span losing is
uninformative; a tuned one winning ends the project.** The asymmetry protects the project, which is
the wrong direction for a test whose purpose is to kill it. *Fix: an equal-budget protocol.*

⚠️ **S-1 — §11 must engage H2O's Theorem 4.4.** H2O *"formulate[s] the KV cache eviction as a
dynamic submodular problem and prove[s] a theoretical guarantee"* for greedy eviction (Lemma 3.1,
Thm 4.4, proofs App. D) — while §3.4's D-3 argues *against* greedy independent scoring **on
submodularity grounds**, and §11's H2O entry says only *"evicts by accumulated attention."* This is
S-1's scholarship failure repeated on a second paper, and **falsifier 5's "regardless of how good
`ψ̂` is" is stated more strongly than the published theory supports.**

📌 **H2O is an *inference-time* KV-cache method.** TG evicts gestalt slots *during training*, where
evicted slots retain their computation graphs. **Porting it across that regime boundary is a design
decision with no analogue in the source** and belongs in §13's limitations as one.

---

## 5. Scope, budget, hardware

🔴 **Only weeks 1–4 are approved (§16). Everything downstream is a projection, not a permission.**

Do not, without the owner saying so in this session: train on PG-19 or WikiText; run anything at
`S = 80` beyond E0c's profiling steps; spend more than **~$100 of compute total**.

### ADR-0007 — everything trains on the Mac Studio; procurement is dropped

🔴 **No outsourced compute, at all.** No GPU rented, no provider named, no quote requested, no hold
taken, nothing priced. **Procurement is out of scope, not deferred within it** — not a blocker, not
a dependency, not a line item. Reconsidered only *after* the model demonstrates its effect locally.

This **supersedes corrections 7, 8 and 10 and ADR-0002 Part A**, all of which reason about a rented
card. Read them as historical. The rental entered the plan for three things, and all three dissolved:

1. **The golden-tensor extraction.** Thought to need a GPU because `jax-metal` is dead. It does not:
   `jax-metal` is the **Metal GPU backend**, and `jaxlib` ships `macosx_11_0_arm64` **CPU** wheels.
   *"No GPU path" is not "no path."* Seconds of CPU work on a 2.48M-parameter model — and CPU is the
   **correct** target, because fixtures must be byte-reproducible. **Done**, in a throwaway venv, so
   JAX never enters `pyproject.toml`.
2. **The capacity budget.** Correction 10's point was real — sizing against 64 GB for a run that
   would happen on a 48 GB rental. **With no rental the asymmetry disappears:** the machine that
   sizes the run is the machine that runs it.
3. **Parallel capacity for weeks 5–7.** That work is **outside the approved scope.** Pricing capacity
   for unapproved work is what §16 declined, and the honest response is to stop pricing it.

**Nothing is lost.** §16 approves E0a–E0i, E1, E2, and **E1/E2 are synthetic at `M = 16, S = 48`**,
which fits several times over. The rental was only ever attached to the block §16 does *not* approve.

**Do not raise `iogpu.wired_limit_mb`** — it is 0, the system default. **MPS stays best-effort, not
required**; all code must still run on CUDA without modification (a portability property, not a
procurement plan). **§13 gains an entry: every throughput and capacity number in this project was
measured on a single M4 Max**, and the paper must not imply anything about other hardware.

---

## 6. The experiment ladder, and where it actually stands

**Nothing after E2 is attempted before E2 passes.**

| # | What | Kill gate? | Status |
|---|---|---|---|
| **E0a** | μP coordinate check — bare TG, then TG + value head | **Yes** | ⚠️ **PARTIAL / DEFERRED.** Value-head half ran (5 seeds) on the **MacBook**; its `run.py` and `coord_check` **are not in this tree** and the APIs are incompatible. And its input premise is wrong for real TG — see §9 |
| **E0b** | Reduction test vs TG, separate RNG for `φ` | **Yes** | ✅ **GREEN, bit-exact** |
| **E0c** | Memory/throughput ceiling on the Studio | Resizes everything | ✅ **RUN** |
| **E0d** | `r_i` vs leave-one-out Δloss | **Yes** | ☐ not run. ⚠️ `metrics/loo.py` is a stub that raises — §3.2.1's truth rule has **no implemented arbiter** |
| **E0e** | `ū` distribution on a FIFO run → `τ`, `E[lifetime]` → `γ_b` | No | ☐ not run. ⚠️ **E0e MEASURES; it does not freeze `γ_b`** — the scope question is §12.2 |
| **E0f** | Verify [P5]–[P14] against primary sources | No | ◐ **pass 2 run 2026-09-20 — 13 of 14 done, [P9] outstanding.** Not clean: corrections 25–30, including three on [P11], the cognitive claim (§2) |
| **E0g** | Name and obtain the E7 stimulus set | **Yes, for E7** | ✅ **PASS** |
| **E0h** | Regress `ψ̂(γ=0)` on current cross-attention logits | **Yes** | ☐ not run |
| **E0i** | Coref histogram over the PG-19 subset, CPU | **Yes — kill gate** | 🔴 **exit 3 — DID NOT RUN.** Pre-registration is **final and UNSIGNED** |
| **E-feas** | Oracle vs FIFO per corpus | **Yes, per corpus** | ☐ not run |
| **E1** | Synthetic, full baseline set + `ν` sweep | **Yes — kill gate only** | ☐ not run |
| **E2** | Vacuity gate on the full eviction score | **Yes** | ☐ not run |
| E3 | PG-19 at `S = 80`, reintroduction loss vs `k` | No | **unapproved** |
| E4 | WikiText comparability | No | **unapproved** |
| E5 | μP width sweep | No | **unapproved.** **Hygiene — make no scaling claim** |
| E6 | Reversal probe | No | **unapproved** |
| E7 | Human narrative-recall gradient | No | **unapproved** |

> 🔴 **E1 is a kill gate, not a result (D-9).** The synthetic generator *constructs* the gap
> structure that makes lookahead pay, so `RSR > RSR(γ=0)` there is **evidence about the optimizer,
> not about language.** The week-4 decision point can only return a red light or a **permission to
> continue** — never a green light on the hypothesis. *A project that forgets this spends 768
> GPU-hours on a mechanism that works on its own test harness.*

> **E5 is demoted.** A Kaplan fit over a 3× width range at ≤21M parameters **cannot resolve a slope
> difference from an intercept shift** — v0.1's falsifier 4 was not a falsifier, it was the
> near-certain outcome. ⚠️ **S-4: [P2] already ran it**, over a *wider* range (48→384), and published
> the fits (`α ≈ 0.081` vs `0.080`, *"consistent with an intercept shift"*) — which is E5's own
> stated conclusion, already in print.

### E0i's pre-registered threshold — the gate in front of the headline experiment

> 🔴 **PASS:** `n_raw × p_LCB² ≥ 150` in **every one** of `(40,48]`, `(48,56]`, `(56,64]`
> **AND** `n_raw × p_LCB² ≥ 600` pooled across `(40,64]`, from **≥ 30 distinct documents**.

- **`p_LCB` is a one-sided 95% Wilson lower bound** on coref precision measured on 100 hand-annotated
  reintroductions — **not the point estimate.** It enters **squared**, which roughly doubles its
  sampling error, so a gate built on `p̂` is systematically more permissive than it looks. At
  `p̂ = 0.90`, `p_LCB = 0.840` and the per-bin floor needs **213 raw events, not 186**.
- **Why `p²` and not `p`:** false positives dilute the observed effect by `p`, and required `n`
  scales as `1/δ²`. Both independent derivations agreed once stated. **The exponent is 2.**
- **Why per-bin binds:** a pooled-only gate can pass on an **empty top bin**, and `(56,64]` is where
  E3's claim lives. Pooling 600 events that are 550 short-gap and 50 long-gap satisfies a pooled
  floor while leaving the headline window unpowered.
- **Design effect applied:** `DEFF = 1 + (m̄−1)·ICC ≈ 1.9`. 150 raw → **~79 effective**; 600 raw →
  **~316 effective**. **The floors do not move; the stated MDEs do** — 2.0%→**2.76%** per bin,
  1.0%→**1.38%** pooled. ⚠️ Per-bin 2.76% detects the **upper** part of [P2]'s 2–4% band and would
  **miss an effect at the bottom of it**; the **pooled** floor carries the power for the headline.
- **Two floors, not one.** Training-side (can the policy *learn*?) and evaluation-side (can we
  *measure*?) have different failure modes and different fixes: an eval-side shortfall is a day of
  inference; a training-side shortfall is a red light with **no cheap rung**.
- **Response ladder is pre-registered and ordered.** Record which rung was used.
- 🔴 **The event threshold does not move retroactively.** A measured `σ_d` changes what the counts
  *buy* in power; it does not re-open the counts. **The `(40,64]` window may not be re-sliced to pass.**
- 🔴 **Signed will not change the status.** A signature records that a number was committed to before
  the data; it supplies no measurement. **E0i stays exit 3 until T3 reports `p`.**

---

## 7. The decision record

| ADR | Decision | Status |
|---|---|---|
| **0001** | Vendor the JAX reference pinned and read-only; **reimplement TG in PyTorch**; validate against golden tensors | accepted; amended by 0007 |
| **0002** | **Part A** rented hardware · **Part B** the fidelity tolerance | **A superseded by 0007. B accepted and DONE** |
| **0003** | **fastcoref** is the pipeline of record (MIT); **maverick** is an agreement/ceiling check only (CC BY-NC-SA); SaT for segmentation | accepted |
| **0004** | **`A_max = S = 48` on synthetic**, non-binding by construction | accepted |
| **0005** | The E7 stimulus set = **NFRD** (Raccah et al. 2024), CC0 | **proposed** — pending owner sign-off and the "in hand" exit |
| **0006** | **`P^(sent)` stays rank-indexed**; the confound is measured, not argued | accepted |
| **0007** | **Everything trains on the Mac Studio; procurement is dropped** | accepted |

📌 **ADR-0001 is deliberately left standing, unedited, as a record of an error.** It concluded no TG
code existed and proposed a 3–6 week reimplementation. **The code existed.** ADR-0002 supersedes it.
*The instrument was the failure, not the effort:* a web index was searched, returned a different 2018
model with a similar name, and that single hit was read as an answer rather than as information about
the instrument. GitHub's own repository index found it on one query. 🔑 **A search that finds nothing
tells you nothing unless you have shown the search can find something.**

### The fidelity tolerance (ADR-0002 Part B) — committed before any fixture existed

| Quantity | rtol | atol | Achieved (fraction of allowance used) |
|---|---|---|---|
| Per-layer activations | `1e-4` | `1e-5` | **0.115** |
| Logits | `1e-4` | `1e-5` | 0.083 |
| Gestalt vectors | `1e-4` | `1e-5` | 0.015 |
| Cross-attention weights | `1e-4` | `1e-5` | 0.003 |
| **Gradients** (206 arrays) | **`1e-3`** | **`1e-4`** | **0.006** worst, 0.0001 median |

Total loss is **bit-identical**: JAX `125.3105468750`, torch `125.3105468750`, `|Δ| = 0.000e+00`.
**Nothing was relaxed.** `git log` shows the tolerance commit preceding the fixture commit, **and
that ordering is the evidence, not the sentence saying so.**

- **Gradients get exactly one order more room**, and the reasons are properties of the comparison,
  not of the transcription: gradients accumulate error over 12 blocks × `S` steps, and **JAX and
  PyTorch reduce in different orders** (floating-point addition is not associative). **One order,
  not two** — a looser allowance would start absorbing the retained-graph divergence the gradient
  fixtures exist to catch, and *a wrong graph is off by a large factor or by everything, not by 1e-3*.
- 🔴 **Gradient fixtures are mandatory.** Gestalts are appended **without detaching the graph**, and
  backward depth is bounded by `S` not `M`. **JAX functional autodiff and PyTorch retained-graph
  semantics diverge exactly there** — a transcription matching every forward quantity while
  retaining the wrong graph passes a forward-only check and **survives to week 7, where it surfaces
  as a gradient-flow difference in E3 that looks like a finding.**
- 📌 **One real transcription bug was found by this harness rather than by inspection:** the fixture
  sidecar did not record token ids, so `from_reference_dict` defaulted `eos_id` to GPT-2's 50259
  while the extraction used 510. **Every activation matched to 1e-7 and the gestalt was off by
  0.25** — the gestalt is read at the first `[EOS]`, so a wrong id reads position 0, and memory is
  written only when a sentence *has* one, so memory stayed empty for all 20 steps. **Defaulting a
  field that changes behaviour was the defect.** The loader now refuses a fixture without ids.
- 🔴 **If the transcription cannot meet a tolerance: record the achieved value and the reason. Do
  not silently relax.** A miss that cannot be explained mechanically is a transcription bug.

---

## 8. §16 release conditions

**Approved:** weeks 1–4, E0a–E0i, E1, E2, ~$100 of compute, no reputational exposure.
**Not approved:** the PG-19 block, the four-figure spend, everything downstream.

**Conditions 1–4 and 8 decide whether the project finishes. Conditions 5, 6 and 7 decide whether it
is worth finishing.**

| # | Condition | Status |
|---|---|---|
| 1 | E0i histogram adequate against the pre-registered threshold | ☐ open — **exit 3** |
| 2 | `γ_b` rederived from measured lifetime; §3.5 demonstrated to move the argmin | ☐ open — E0e |
| 3 | Expire-Span implemented and run on synthetic at the week-4 gate | ☐ open — **no demotion path** |
| 4 | E7 model funded **or** E7 cut and falsifier 4 withdrawn — **and sentence length partialled out** | ☐ open |
| 5 | Marginal/redundancy scoring specified; ρ(score, LOO Δloss) reported with and without | ☐ open |
| 6 | §11 rewritten with the leading-edge strategy as named ancestor **and implemented baseline** | ☐ open |
| 7 | **§15.2 non-empty, derived by the author, defended under push-back in person** | ☐ open — **owner only** |
| 8 | ~~Parallel GPU capacity reserved~~ | **n/a — dissolved by ADR-0007** |

### Condition 4 carries an analysis requirement, not only a funding decision — correction 24 (S-7)

TG normalizes every gestalt to unit norm, so **a 4-word sentence and a 43-word sentence enter memory
at identical strength.** Human encoding quality **falls with sentence length**. So the two systems
being correlated in E7 forget for different reasons — human forgetting is partly *encoding*, TG's is
*retention* only.

> **A correlation between what RSR drops and what people fail to recall can be produced by sentence
> length alone, with no retention policy involved.** That is a confound in the **primary claim**.

**Fix:** partial sentence length out of the E7 correlation, exactly as §10.1 already partials out
serial position. If the correlation **survives** it is *stronger*. If it does not, **RSR is tracking
sentence length rather than structural importance** — a vacuity failure **no falsifier names**, and
one the age-based §7.1 gate **structurally cannot see**, because length and age are independent, so
a length-tracking policy beats LRU comfortably while having rediscovered nothing.

⚠️ **Invisible in current work; do not go looking for it now.** Both measured: the synthetic corpus
is **mean 4.14 words/sentence, max 6, zero over 8** (n = 3072 at `sentences_per_document=48, seed=0`;
distribution `{2:91, 3:244, 4:1914, 5:784, 6:39}`) — the whole corpus sits at the flat top of the
human curve, so **the confound has no variance to act through**. And `max_sentence_tokens = 64`
**truncates rather than degrades** — a cliff, where the human curve is a gradient.
🔴 **Do not change the μP multiplier, `srep_norm_target`, or any frozen constant on the strength of
this.** It is a regressor in one analysis.

### E0g — the stimulus set (PASS), with three findings that change the experiment

**Naturalistic Free Recall Dataset** (Raccah, Chen, Gureckis, Poeppel & Vo, 2024, *Sci Data* 11:1317),
**CC0**, osf.io/h2pkv — 4 spoken narratives, 229 participants, human-reviewed transcripts of stimuli
*and* recalls, per-event recall probability with bootstrapped CIs. **CC0 collapses "identified" and
"in hand" into one exit** — no data-use agreement, no calendar risk. It beats the spec's own ranked
first choice (Sherlock, N=17, **audiovisual — a text LM cannot read the stimulus**).

1. 🔑 **It ships a position-independent importance measure.** *Semantic centrality* — a semantic
   similarity graph, per Lee & Chen, with *"a significantly positive effect … on the likelihood of
   recall (p < 0.001, β = 0.27)"* and published code. **Position-independent by construction:
   reorder the events and it is unchanged.** §5.3's criterion is met out of the box. *(An earlier
   draft of E0g concluded the opposite — "no naturalistic free-recall corpus does" — by reading
   Methods and Data Records and stopping before the usage notes. Corrected in place.)*
   ⚠️ **One precision:** the paper does not state that the reported β **controlled for** serial
   position. So centrality is position-independent as a *construct*; whether β is a
   position-controlled *estimate* is not established. **E7 still partials position out.**
2. 🔴 **`baseball` is a likely PG-19 contaminant.** It is chapter 1 of *Baseball Joe in the Big
   League* (Chadwick, **1915**), Gutenberg **#27584** — inside PG-19's pre-1919 window — and it has
   the **highest** mean recall rate of the four, making it the story most likely to produce a
   convincing but meaningless result. **Check the subset for #27584 before training, not after.**
   📌 Weaker but not gone: correction 13 establishes [P2] trains on WikiText, so PG-19 is RSR's own
   choice — **it is RSR's exposure, not inherited.**
3. **The unit mapping runs the opposite way from §10.1's assumption.** NFRD is **coarser** — one
   recall probability per *event*, each spanning several sentences — so every sentence inside an
   event inherits one value. **Effective N is the number of events (~100–150 across all four), not
   ~300 sentences.** State it before E7 runs, not in review. And three of four stories exert real
   eviction pressure at `M = 40` (~84–96 sentences), which **may remove the `M = 8` second model
   from the budget entirely** — ⚠️ **but that is estimated from word counts at [P2]'s ~25
   words/sentence ratio and must be measured under the project's own SaT segmentation before it is
   banked.**

---

## 9. What has actually been measured, with provenance

🔴 **Every number below carries where it came from. A number without provenance does not go in a
writeup, and a number that cannot be resolved to a ledger key or a command is a claim, not a
measurement.**

### Suite — re-measured by this session

```
$ .venv/bin/pytest -rs --tb=no          # worktree at d1c221f, 2026-09-18
passed=287 failed=0 skipped=0 errors=0
287 passed, 1 warning in 28.55s         # exit 0
```

**Zero skips.** Both former blockers are closed: `test_fidelity.py` is green forward *and* gradients
within ADR-0002's committed tolerance, and `test_reduction.py` (E0b) is green and **bit-exact** —
stock TG, the §3.7 reduction and the explicit FIFO policy all give `125.310546875`, and the learned
head does not.

### E0c — capacity and throughput (`experiments/e0c/RESULTS.md`, Mac Studio, MPS, 3 repeats)

| policy | `d` | `S` | batch | peak GB | sent/s | h per 1.2M steps |
|---|---|---|---|---|---|---|
| FIFO | 128 | 80 | 16 | 32.2 | 374±8 | 0.9 |
| **RSR (learned head)** | **128** | **80** | **16** | **31.8** | **310±2** | **1.1** |
| FIFO | 128 | 48 | 16 | 20.3 | 392±0 | 0.9 |
| **RSR (learned head)** | **128** | **48** | **16** | **20.3** | **357±2** | **0.9** |
| FIFO | 384 | 80 | 8 | 31.0 | 197±0 | 1.7 |
| **RSR (learned head)** | **384** | **80** | **8** | **31.0** | **171±4** | **1.9** |

> **RECOMMENDED CONFIG: `d = 128, S = 80, batch = 16`** — 31.8 GB, 310 sent/s, ~1.1 h per run.

- **`S = 80` fits at every width in scope.** §4.2's tie-break (*"if `S = 80` does not fit, `S` wins
  and `d` is cut"*) **never binds.** The constraint that binds is batch size at `d = 384`.
- **Memory is identical across policies to within 0.4 GB.** RSR adds compute, not memory.
- **Most of the RSR cost is the Python dispatch loop, not the value head:** `neg_age` runs no head
  and still costs 15%; the bilinear head adds 2–5% on top. **If throughput needs recovering, batch
  the policy across rows — do not simplify `ψ̂`.**
- 🔴 **The `neg_age` row is not RSR.** Quoting it as "the RSR arm" reports dispatch overhead under
  the name of the mechanism.
- 🔴 **How the ceiling announces itself.** `d=384, S=80, batch=64` asks for **68.80 GB on a 64 GB
  machine.** Metal does not refuse it — the run "succeeds": 1,024,668 swapouts, 16 GB of swap, and
  **throughput inverts** (batch 32 → 1162 sent/s; batch 64 → 807 sent/s: *twice the work, 30%
  slower*). **A configuration that reports `fitted: true` while running slower than half its batch
  size has not fitted.** A memory number alone would have called 68.80 GB a pass.

🔴 **Correction 23: §4.1 assumed ≤7 sent/sec and 48 h per run. Measured is 171–392 sent/s and ~1.1 h
— off by ~44× in our favour.** The arithmetic was sound; it was **anchored to a rented A40 at
`d_model = 768` / 85.6M params**, roughly 4× wider than anything here. **The error is the anchoring,
not the division.** §4.1's 1,070 GPU-h total, its parallel-capacity table and its cut order are
superseded. ⚠️ **What the measurement does not cover:** random tokens rather than real text (real
batching uses token-budget bucketing, so lengths and effective batch vary); **no optimizer step** in
the timing; one data shape per row; MPS only. **It is a measurement of the model's throughput, not
of a training loop's.**

### E0a — μP coordinate check (PARTIAL, and its premise is wrong for real TG)

Value-head half, 5 seeds, `d ∈ {128,192,256,384}`, `n_live = 40`, 8 AdamW steps:

| quantity | mean | sd |
|---|---|---|
| drift at init | 0.419 | 0.107 |
| drift at step 8 | **0.120** | 0.034 |
| `ψ̂` RMS ratio `d=384/d=128` at init, **predicted** by `Θ(1/√d)` | `1/√3 = 0.577` | |
| **measured** | **0.581** | — a 0.7% match |

> 🔴 **The E0a verdict is read at `t ≥ 1`. Step 0 is recorded and never gated on.** At init the
> output is *not* width-invariant, **and that is correct.** Anyone who "fixed" it by switching the
> multiplier to `1/√d` would have tuned the head for the init regime and broken it for the trained
> one — exactly the failure §4.3's footnote warns about, met from the other side.

🔴 **What does NOT stand: that this validates §4.3's `1/d` prescription for TG.** The experiment
feeds `torch.randn` gestalts — `Θ(1)` coordinates, norm `~√d`. **TG's real gestalts are unit-norm**,
coordinates `O(1/√d)`. The arithmetic is right *under an input assumption TG violates.*
**When E0a actually runs:** pin `c_t` first (it cannot measure an underspecified object), re-run with
unit-norm inputs, and **report both input regimes side by side — the difference between them is the
finding.** 📌 A synthetic "correlated regime" via `pinv(W)` was tried and is **not a substitute** —
it does not control correlation magnitude and gave non-monotone results. **Real optimizer steps, or
nothing.**

⚠️ **The code that produced these numbers is not in this tree.** It lives on `macbook-local-2026-09-18`;
this tree's `experiments/e0a/run.py` and `src/rsr/mup/coord_check.py` are **stubs that raise**, with
incompatible APIs. **Gate status: DEFERRED, not passed.**

### Mutation battery — `docs/mutation-battery.md`, **17/17 gates PROVEN**

> **A new check is not believed until a mutation has shown it red — and the mutation must redden
> *only* it. If nothing reddens it, the check adds nothing and that is the finding.**

Two findings the battery produced and that were **recorded rather than fixed away**:

1. **`LRUPolicy.on_write` is defensive, not load-bearing.** Neutering it reddened **nothing** —
   `select_eviction` takes `max(written_at, last_used)` and a stale record is by construction older
   than the new occupant's write step. **The hook stays because H2O needs it** (accumulated attention
   is not monotone in write time). The original test asserted the hook was load-bearing and **was
   vacuous; it was replaced, not deleted.**
2. **Defaulting `nu` breaks the module, not the test** — a default before a non-default makes the
   dataclass invalid, so the module fails to import. A **stronger** guarantee, but not one the test
   can demonstrate, so the battery mutates the last non-defaulted field instead.

📌 The first version of the reduction suite **had a hole that only a mutation exposed**: dropping the
`nu` clause from `is_reduction` left the whole suite green. With that hole, a policy at `ν = 0.5`
would report itself as the §3.7 reduction, `observe` would silently return `None`, and **an arm
supposed to be learning would quietly learn nothing while still being called RSR.**

📌 **`tests/conftest.py`'s all-skipped tripwire cannot be mutated from inside the suite it guards.**
Proved separately by running a suite in which every test skips: `FAILED: only 0 tests passed, floor
is 100`, exit 1, against the real suite's exit 0.

---

## 10. What is broken right now

🔴 **Fix the instrument, then measure. Two of these make any retraining experiment uninterpretable
before it starts.** All five `loop.py` defects and the corpus defect were **re-verified at `d1c221f`
by this session**, at source.

### 10.1 `src/rsr/train/loop.py` — five confirmed defects, all ours

**491 lines of `rsr.train` have no test that imports them.** *(Re-derived 2026-09-20:
`src/rsr/train/` is 852 lines — `__init__` 25, `checkpoint` 361, `heartbeat` 159, `loop` 307 — and
only `checkpoint.py` is imported by a test. The figure read **365** until today, which was `loop`
228 + `heartbeat` 137 before 37 over-long prose lines in those two files were rewrapped to get ruff
green. **The rewrap is why the number moved; nothing was added and nothing was tested.** A derived
number in prose rots the moment anyone touches the file it describes, and this one rotted inside a
day.)* Nothing in `tests/` imports `rsr.train` except
`checkpoint`.

| # | Defect | Verified at `d1c221f` | Consequence |
|---|---|---|---|
| 1 | **The loss scores padding.** `F.cross_entropy(…, reduction="mean")` with no `ignore_index` and no mask, although `step_fn` receives `mask_t` | `loop.py:164`; `grep -c ignore_index` → **0** | **93.4% of 193,536 scored targets are PAD.** Final loss 1.1066 nats is *worse than a context-free unigram table* (0.5044). **Every perplexity figure in every ledger is uninterpretable.** The biggest one |
| 2 | **The RSR arm is FIFO.** `policy = FIFOPolicy()` unconditionally while `policy_name` is stamped into `run_id`, the frozen config and the heartbeat | `loop.py:138`, `:116`, `:142`, `:144`; **no `--policy` in any `add_argument`** | **A run whose heartbeat reads `"policy": "rsr"` IS FIFO.** Two runs differing only in `policy_name` are bit-identical on CPU, and **no CLI run could ever have been RSR** |
| 3 | **The `srep_norm` hinge is missing from the objective.** `step_fn` is `cross_entropy` alone | `loop.py:164`; **0 occurrences of `srep_norm`** in the file | Pre-normalization gestalt norm reaches **20.5** against the reference's `[0.9, 1.1]`. **The shipped loop optimizes an objective that is not TG's.** `test_fidelity.py` **structurally cannot catch it** — it loads reference weights and compares forward/gradients, **never the training objective** |
| 4 | **Heartbeat attribution is silently null.** `policy.attribution()` — the real method is `attribution_counts()` | `loop.py:184`; `rsr.py:450` | `hasattr` is False, `attr` is always `None`. The eviction-attribution field never records anything |
| 5 | **`--vocab` defaults to 50257** against a 156-word corpus | `loop.py:211`; `train()` already supports `vocab=None` to derive | Trains a 6.4M-parameter output layer that is **99.7% unreachable** |

🔴 **Write the tests first for this one.** The file has no tests, has already shipped five defects,
and **three of them are silent — they produce a plausible loss curve.** The test for defect 2 is four
lines: construct two policies, assert the run's `policy` field and the constructed object agree.

⚠️ **On defect 3's fix, the honest support is the paired statistic**, not a mean difference smaller
than the within-arm spread: **5 of 5 seeds move the same direction, sign-test p = 0.031.** And
**0.780 does not *land* in `[0.9, 1.1]`** — 0 of 384 sentences reach the lower edge on the seed
quoted. **The hinge *binds*; it does not *land*.** Raising `srep_norm_reg_weight` above 0.01 was
never tried.

### 10.2 The synthetic corpus is not rewardable

`src/rsr/data/synthetic.py` stores each query's answer **out of band** (`Sentence.answer`, set at
`:227`, carried at `:238`) and **it never enters the token stream.** Every query reads *"What does
Hal-6 measures?"* **with no object.**

> 🔴 **So the cross-entropy objective has no target token that requires retrieving the asserted
> fact.** On this corpus the memory is not merely unused — **it is barely *rewardable*.**

**Until a target token requires retrieval, "the memory is inert" is a finding about the corpus, not
about TG.** This is upstream of all three causes the overnight loop proposed.

> 📌 **Update 2026-09-22.** S0-03 puts the answer in the token stream (`SyntheticConfig.answer_in_stream`,
> default `True`), per `experiments/s0-03-rewardable-corpus/RESULTS.md` and
> `experiments/efeas/RESULTS-s003.md`. The S0-03 ledger verdict is `inconclusive`, "retrieval not
> shown (not both at chance)" (`runs/s0-03-rewardable-corpus/ledger.json`). The paragraph above
> describes the pre-S0-03 corpus.

### 10.3 The model this repo trains has no usable memory — *superseded 2026-09-22; kept as history*

> 📌 **Update 2026-09-22: the decisive experiment below has run. The heading above describes the
> old unmasked-objective model and no longer describes the repo's trained model.** The decisive run
> is #34: `experiments/decisive-shuffle/RESULTS.md`, `runs/decisive-shuffle/ledger.json`,
> `provenance.git_sha` `90438f3`, three seeds per arm, CPU. It re-ran the shuffle control on
> S0-03's rewardable corpus under three objectives. Every figure here was read from that ledger:
>
> | arm | objective | `arm{X}.ratio` (train batch, mean ± sd) | verdict |
> |---|---|---|---|
> | A | unmasked, hinge off (#17's config, new corpus) | **0.0030 ± 0.0023** | inert |
> | B | masked, hinge off | **9.33 ± 4.68** | live |
> | C | masked, hinge at `TGConfig` default | **7.74 ± 6.06** | live |
>
> - **The memory path is not broken** (ledger `verdict.outcome` `falsified` for the inert
>   hypothesis). The cause of #17's null was the objective, not the wiring. Arm A shows that the
>   corpus change alone was not enough.
> - 🔴 **Live is not retrieval.** Held-out answer-token NLL, with the model's own memory intact,
>   is **2.904 ± 0.057** (B) and **2.852 ± 0.088** (C) (`arm{B,C}.heldout.answer.answer_all.honest_nll`).
>   That is at or above chance `ln 16 ≈ 2.773`. ⚠️ Per bucket, `gap_1` is below chance
>   (B 2.725, C 2.609), and gap = 1 is readable through context seeding outside the memory (RESULTS
>   caveats). `gap_2_to_M` and `gap_gt_M` are 2.88–2.94, above chance. The memory is used, and it
>   is not shown to retrieve the right answer.
> - **Arm A's memory is row-agnostic, not ignored.** Its trained memories are almost exactly
>   collinear across rows: `armA.train.cosine_trained` **0.9995 ± 0.0005**, against a decoy's
>   0.565. Matched-norm random replacement moves tokens at `armA.train.random_ratio`
>   **2.10 ± 1.11**. So the readout uses memory, but the contents carry nothing row-specific. This
>   is the PREREG's own descriptive reading ("random ≥ 0.1 while the shuffle ≤ 0.01").
> - **What follows for the text below.** The S0-04 / #17 finding (`runs/shuffle-control/ledger.json`)
>   still stands **for the model it measured**: unmasked loss, pre-S0-03 corpus. The decisive
>   experiment's "Still inert ⇒ … broken" branch **did not occur**. The "Live ⇒ the cause was the
>   objective" branch did, **with the retrieval caveat above**, which that sentence did not
>   anticipate.
> - ⚠️ The ledger records `python: 3.14.6`, which predates the 3.12 pin (`5563b62`). See the
>   provenance note in the RESULTS file.

**The original 2026-09-20 text follows, unedited.**

✅ **Verified by a stronger test than the loop ran, and re-derived from committed code on
2026-09-20** (`runs/shuffle-control/ledger.json`, verdict `survived`). Hand a document's row
**another document's entire 16-slot memory** at every step: max per-token delta 1.2e-4 nats, min
−1.9e-4, across 384 sentences. The re-derivation's seed 0 gives **1.17e-4 / −1.91e-4**, with
`memory_gate` 0.949–0.989 and 384/384 sentences with EOS. **Tokens move, but only slightly:** mean
|per-token Δ| is **7.8e-4 ± 2.7e-4 of what a live, untrained memory produces** at the same seed,
over three seeds (`ratio_to_live_decoy`). 🔴 **Two phrases this paragraph used to carry are
retracted (§11):** *"the loss moves by exactly 0.0"* (the signed mean is −1.46e-8 on seed 0,
not zero) and *"no single token moves"* (1,100–1,254 of ~1,580 real tokens move on each seed).
Zeroed slot contents cost +0.018%. The gestalt cloud collapses: uncentered participation
ratio **1.0001 of 128**.

**It is not the known dead-memory trap** — `has_eos` is true for 384/384 sentences, all rows fill
16/16 slots, and the trained `memory_gate` scalars are **0.949–0.989**, not gated off.

🔑 **Against [P2]'s own ablation, removing the working memory costs 29.8 → 45.8 PPL (+54%). Here it
costs ≈ 0%.** Not directly comparable — different corpora, scales and objectives, and §10.1–10.2 are
sufficient to explain it — **but the gap is the thing to close, and it is what makes the shuffle
control decisive rather than merely interesting.**

⚠️ **"Deleting the entire memory" names an ablation that was not performed.** The ablation sets
`cfg.use_memory=False`, **but the `bos_replacement_mode == "copy"` block sits *outside* that guard**,
so the previous sentence's gestalt still lands at token position 0 — this is [P2]'s **context
seeding** (§3, verified), a real architectural feature. Turning *that* off costs **+31%** of the NLL.
The audit closed the gap: replacing it with **another row's** gestalt costs **+0.0002%**, so it
carries no information either — **a required constant placeholder, not a channel.** *The conclusion
survives; the stated method does not.*

🔴 **The decisive experiment, and both outcomes are decisive.** Repair the loop, make the corpus
rewardable, re-run the ablation with **the shuffle control as the primary readout** (give a row
another document's memory — **not** `use_memory=False`). Still inert ⇒ **the memory path itself is
broken**, a wiring or gate problem, and *every E1/E3 reading would be about a model whose memory does
nothing.* Live ⇒ the cause was the objective and the fix is known.
**This must be settled before E1 is read as evidence about anything.**

### 10.4 Settled by measurement — do not reopen

- ✅ **`ν` cannot arbitrate between two near-copies, at any `ν`.** The max-cosine penalty is **equal
  within a mutual-nearest-neighbour pair to float32 precision** (`|Δpenalty| = 4.102e-08`), so the
  within-pair score gap is **invariant in `ν` to nine significant figures.** `score_i(ν)` is affine
  in `ν` with slope `−p_i`, and pair members share `p_i` exactly. Raising `ν` only makes the decision
  land *inside* such a pair more often (**3.1% → 83.1% at `ν = 4`**), where it is then a coin flip:
  `P(argmin evicts the needed member) = 0.49860 ± 0.01613` against chance 0.5, **bit-identical at
  `ν` = 0, 1, 4.** `ν` does not make the choice better; it makes the choice **happen 24.8× more
  often.** 🔴 **So `ν` is not the lever for D-3 and tuning it is wasted effort.** A redundancy fix
  must be **asymmetric within the pair** — LOO marginal value, or a tie-break on age/demand — or
  `max_cos` must be applied as **set-level selection** rather than a per-slot additive penalty.
- ✅ **`b` and `ν` are not commensurable.** `_score` z-scores `ψ̂`, adds `b` **on the z scale**, then
  subtracts `ν·max_cos` **raw**. ⚠️ **The mechanism first reported for this was wrong:** neither term
  is z-scored, and the conversion factor is `SD(max_cos)`, not `SD(ψ̂)` (which cancels). **This is a
  spec defect in §3.4, not a code defect — do not send anyone to patch `rsr.py`.**
- ✅ **`EvictionRecord.attribution` is a config echo**, so §3.4's own decision-attribution test is
  **not computable from any run's log.** Measured: `"+b"` on **1.0000 ± 0.0000** of evictions in
  every `b_enabled=True` arm, *including `γ_b = 0` where `b` changed 0 of 2400 victims*; mutual
  information with ground truth **0.0000 bits** against **0.97340 ± 0.00683** available.
  **Treat that gate as NOT IMPLEMENTED.** The cheapest sufficient fix is one field —
  `victim_without: dict[str, int]`, the argmin with each optional term omitted — which also gives
  §3.7's off-switch discipline a measurement counterpart. 📌 **Do not reach for logging the pre-bias
  margin instead: measured, it is *worse* than the margin already logged** (AUC 0.643 vs 0.686).
- ✅ **The tie is expensive when the answer lives in the residual.** Which member of a `cos = 0.90`
  pair dies moves retrieval hit rate **1.0 → 0.0** (paired Δ 1.0, null 0), at every cosine 0.90–0.999
  and at `M = 16` and 40. **But §3.4's defence is right in the other regime:** if the query needs the
  **shared topic**, the two victims differ by **0.00001**. 🔑 **The honest bracket is 1e-5 to 1.0, and
  the load-bearing unmeasured quantity is what fraction of real near-duplicate pairs carry a
  future-needed distinction in the residual. Nothing has measured that on text.**
- ✅ **The 03:27 escalation was correctly retracted.** ⚠️ Its *stated mechanism* is wrong — "α is a
  function of the pre-query state" is false; α *does* take the query. **What survives is a theorem,
  not a finding: where the need is not identifiable from the input, no retention target computed
  from the input can identify it.** That is a statement about what `ψ̂` can be *asked* to do, not a
  defect in `r_i`.
- ⚠️ **E0d as currently designed would report a null that means nothing on this corpus at this
  budget.** In the trained model **both sides are ≈ 0** — LOO Δ magnitudes ~1e-7 nats, proxy-vs-LOO
  sign agreement 0.510 ± 0.016 — so the correlation is uninformative **because the model is
  degenerate**, not because the target is unfaithful. **§3.2.1's rule is silent on the case where
  they agree and both are uninformative, which is the case that obtains.**

### 10.5 The evidence machinery

- 🔴 **Of the overnight run's 13 ledgers, 9 cannot be re-executed from any commit.** Seven
  `commands[].argv` point at a session-scoped scratch directory; one is a literal placeholder; one
  names a file not in the tree. **There is no path from any commit to most of those numbers.**
  📌 **The sharpest part decides the design:** `cycle-srep-hinge` has `git_dirty: false` and is
  *still* unreproducible, because its modified trainer was never committed. **So the gate is not
  "the tree is clean" — it is "the code that ran is committed."**
- ⚠️ **`docs/gates.md` describes a gate system this repo does not implement.** There is no
  `src/rsr/gates/` `Exit` enum, no `check_floor`, no `.rsr/` directory, and `REGISTRY.unset()` —
  which its last ratchet row names — **does not exist in `src/rsr/constants.py`.** It arrived by a
  port from the other repo. **Treat it as a specification of intent, not a description of this tree.
  Do not report a ratchet as checked because `gates.md` lists it.**
- ⚠️ **`scripts/mutation_battery.py`: `off_gate` is computed, stored, printed — and never enters the
  `unproven` filter.** So clause 2 of the discipline (*"the mutation must redden only it"*) is
  enforced by nothing: a mutation reddening 11 unrelated tests still scores `PROVEN`.
- ⚠️ **`scripts/canary.py`: a first reading exits 0**, writing `outcome: "inconclusive"` and verdict
  `"baseline"` — **the exact 3-collapsing-to-0 shape, in the one script with no exit-3 branch.** And
  a length mismatch is compared over the overlap only, so a run producing 2 of 6 beats can report
  "held."
- ⚠️ **`_git()` returns `""` when git fails**, so `git_dirty` becomes `bool("") → false`: **a git
  failure is currently indistinguishable from a clean tree.** `src/rsr/constants.py:571` already does
  this correctly — return `{"git_sha": None, "dirty": None, "provenance_error": <str>}`.

### 10.6 Stub census — measured at `d1c221f`

| Still a stub that raises | lines / raises |
|---|---|
| `baselines/h2o.py` · `expire_span.py` · `leading_edge.py` · `oracle.py` | 46/4 · 42/4 · 44/4 · 36/4 |
| `retention/bias.py` · `retention/shadow.py` | 72/4 · 46/3 |
| `metrics/loo.py` · `reintroduction.py` · `vacuity.py` · `gini.py` | 22/1 · 18/1 · 25/1 · 14/1 |
| `data/coref.py` · `data/pg19.py` | 29/2 · 24/1 |
| `mup/coord_check.py` | 42/1 — **but its docstring carries the load-bearing E0a reading rule** |
| `retention/rsr.py` | 456/3 — **real**, 3 raises remaining |
| `experiments/{e0a,e0d,e0e,e0f,e0g,e0h,e0i}/run.py` | 11 lines each, 1 raise each |

**So: four of the eight baselines do not exist, the shadow buffer and bias loop do not exist, and
the LOO metric — §3.2.1's own arbiter of truth — does not exist.**

---

## 11. Retracted numbers — do not quote these

The overnight run of 2026-09-18 was honest and well-instrumented, and **its ledgers are trustworthy
while its prose is not.** An audit re-derived the load-bearing claims. These failed:

| Do not cite | Why | What is true |
|---|---|---|
| **`1.5476 ± 0.0171`** as the hinge-off masked NLL | `grep -rn "1.5476" runs/` returns **0 hits**. It exists in no ledger or metrics file on **either** machine | The real hinge-off arm is **`1.5945 ± 0.0482` (n=2)**, key `I300_HINGEOFF_eval_masked_real_token_nll`. **The error runs *against* the claimant's interest** — the hinge helps ~4× more than reported — so it is transcription, not motivation |
| **"cross-attention KL from uniform 0.0013 nats"** as evidence of an inert memory | **Non-diagnostic and directionally backwards.** Untrained measures 0.0016–0.0034, trained 0.0099–0.0139 — the trained model is **4.4× *farther* from uniform**, monotone across every seed. Near-uniform cross-attention is this architecture's default at this scale. Cycle 13's own ledger carries a row named `KL_IS_A_WEAK_DISCRIMINATOR` | **Quote the shuffle control instead** (§10.3) |
| **"3 canaries, all held"** | `runs/canary/cycle-04/ledger.json` says `"outcome": "inconclusive"` — it **wrote the baseline rather than comparing against one** | **2 survived, 1 inconclusive.** The environmental conclusion still holds: the audit re-executed `smoke2` at HEAD ten commits downstream and got bit-identical loss, grad_norm, loss_sum and ppl |
| **"proven to learn, 10.82 → 1.11 from chance"** | That run used `V = 50257` against a **156-word** corpus. `10.817 ≈ ln(50257)`; true chance is `ln(160) = 5.075`. **The first ~5.75 of the 9.71 nats is the model learning that ~50,100 output tokens are dead** | **The learning is real** — final ppl **3.024** against true-chance **160** — but the honest interval is **5.075 → 1.107** |
| **"12.8 pp"** as the order-statistic effect on `b`'s authority | That arm also varied `γ_b` **and** the EMA half-life | The isolated effect is **~17 pp** |
| **"50 escalations"** | it was **54** | — |
| **"the loss moves by exactly 0.0"** and **"no single token moves"** (the shuffle control, §10.3) | Re-derived from committed code on 2026-09-20: the signed mean Δ is **−1.46e-8** on seed 0, not zero, and **1,100–1,254 of ~1,580** real tokens move on each seed. And the signed mean is **the wrong statistic**: a live, untrained memory moves it by only ~5e-5 relative, because swapped memory pushes tokens both ways (`experiments/shuffle-control/PREREG.md`, Amendment 1) | **The per-token figures reproduce** (seed 0: max **1.17e-4**, min **−1.91e-4**, gates 0.949–0.989). The claim holds as: mean \|per-token Δ\| is **7.8e-4 ± 2.7e-4** of a live memory's (`runs/shuffle-control/ledger.json`, `ratio_to_live_decoy`) |
| **"every number traces to a ledger row"** | **false**, and it fails first at the canary headline itself — cycle 4's four advertising numbers appear in no ledger | — |

🔴 **If you are about to repeat a number from `docs/lab-notes/overnight-2026-09-18.md` or
`morning-2026-09-18.md`, resolve it to a ledger key first. If it has no key it is a claim, not a
measurement, and it does not go in any output.** Join on `run_id`, **not on `cycle`** — the
experiment ledgers' `cycle` field runs one behind the scoreboard's numbering, two runs collide on
`4`, and one is `null`.

---

## 12. Decisions only the owner can make — do not make them, do not work around them

**Measure around them; do not resolve them.** All six are the owner's.

1. 🔴 **The `b_max` invariant.** `b_max` is FROZEN at 1.0 and it **really is** one SD of `ψ̂`
   (measured **0.98280 ± 0.00360**, theory 0.98111). **But the argmin turns on the gap between the
   two smallest of `M` draws, not on the spread.** So `b_max` is ~**3.25×** the median top-2 margin
   at `M = 40` and ~**1.9×** at `M = 8` (swing ~1.68×), and **92.3% ± 0.5%** of decisions have a
   margin `b_max` alone could cross. The mechanism is **pure order statistics** — reproduced from
   first principles, so it is **not fixable by rescaling**. Four options are costed; a fifth is that
   `b_max` should not be frozen at all. ⚠️ An `M`-keyed fix **cannot work**: the gap it would be
   tuned to close is itself a function of redundancy, and redundancy's *sign* flips with `M`. And
   mean-max-cosine **is not a sufficient statistic** for redundancy — two constructions at the same
   achieved cosine differ in `b_max`/margin by **1.484 ± 0.046×**; what shrinks the margin is the
   **upper tail** of the pairwise-cosine distribution, not its mean.
   🔑 **All options need one sentence naming the invariant** — equal flip rate? equal `b_max`/margin?
   equal fraction of decisions `b` could cross? **The missing input is the definition of
   "comparable", and no further measurement supplies it.** The harness returns the number for any
   candidate in ~80 s.
2. 🔴 **Correction 4 contradicts itself at the `M` that D-5 funds.** `γ_b = b_max/(0.25·E[lifetime])`
   with `E[lt] = M` gives **0.1 at `M = 40`**, 0.25 at `M = 16`, **0.5 at `M = 8`** — **5× over the
   top of its own stated "order 0.05–0.1."** The audit's read: this is a **scoping bug, not a range
   bug.** `γ_b` is per-corpus exactly as `M`, `K`, `S` and `A_max` already are, but it is derived
   with no scope, so **the registry will silently hand E7's `M = 8` model a value ~5.7× too small.**
   ⚠️ And `E[lifetime]` is itself **policy-dependent** (26.49 ± 0.53 under a learned head, not
   `M = 40`), so E0e's design carries a circularity.
3. **Which side of the `ProtectionBias` / `_score` interface moves.** `_score` calls
   `self.bias.b(slots)`; `ProtectionBias` declares `update`/`values`/`reset` and **no `b`** — a
   `# type: ignore` is why nothing caught it, and `bias: object | None` pins no protocol, so nothing
   will catch the next drift either. **Evidence favours `ProtectionBias`** (§3.5 updates `b_i` from
   `ū_i` and its own previous value — nothing makes it a function of memory state), **but the spec
   names no accessor.**
4. **§3.4's `− ν · max cos` is unclamped.** ⚠️ The "two sign traps" report is **overstated** — both
   named scenarios provably cannot change an eviction. **But there is a reachable version:** at
   `n_live ≥ 3`, any slot whose best peer-cosine is *negative* gets a **protective bonus**, and that
   **does** flip victims. Decide in E1's `ν` sweep whether the term should be `− ν · max(0, cos)`.
5. **Per-step vs global normalizer in `_score`** — they differ on **5.79% ± 0.93%** of victims.
   **Neither is wrong.**
6. **Confirm S-7 was actually sent.** It is the only §16 release-gate edit an unattended loop made
   (`docs/release-conditions.md` condition 4, commit `f1ea1a0`). It *tightens* rather than loosens,
   and **nothing in the repo records the authorisation beyond the loop's own prose.**

📌 **Also owner-only, and already flagged:** ADR-0005's sign-off on departing from the spec's ranked
stimulus list, and pinning `c_t` before E0a runs (§4.2 — the decision is made as D-C, the
confirmation is not).

---

## 13. Standing prohibitions — honor these literally

### 13.1 §15

🔴 **Do not fill §15.** It reads: *"This section is a placeholder and must not be filled by a
reviewer, an advisor, or a model."* **You are a model.** If you find yourself drafting a candidate
for §15.2, **stop.** You may **schedule** the work; you may not do it. **Do not restate the pointer
§15 leaves open, and do not suggest answers to it.**

**This is not a style preference.** Every correction in the spec arrived from outside; that is a good
revision record and a poor research record, and **the difference matters because the second is what
is being assessed.** One supplied by an agent would reproduce that failure one level up. §15 itself
names two strong candidates offered in review — the redundancy term, and the tag-and-capture mapping
— and **disqualifies both, explicitly, because they were supplied.**

📌 **§15.1 (rederivability) is different and is work the spec assigns:** every changelog correction
must be rederivable from the primary sources **without reference to the review that produced it**.
§15.3 already names one worked example that is the author's own — §4.3's `Θ(√d)`-at-init versus
`Θ(d)`-when-correlated footnote. **No session has touched §15. Keep it that way.**

### 13.2 The pre-registration

🔴 **Do not sign `preregistration/e0i_threshold.md`.** It says **"AWAITING SIGNATURE — NOT IN FORCE
UNTIL SIGNED"** and *"it must be signed by a person, and that person is not a model."*
⚠️ **`docs/lab-notes/HANDOFF-2026-09-18.md`'s document table calls it "signed." The handoff is wrong
and the file is right.**

*A name written there by an agent would make the document look committed-to while binding nobody,
and a later reader — or a committee — would read it as the owner's commitment. **The blank is the
point.***

### 13.3 The rest

- **Do not backpropagate the retention loss into the transformer or `W_sent`.** Only `φ`.
- **Age is excluded from `ψ̂`.** **`b` never enters `ψ̂`** or any differentiable path.
- **Make no scaling claim anywhere.** E5 is hygiene.
- **Do not call a truncated-BPTT window "consolidation."** The CLS mapping was inverted and the
  claim is deleted. 📌 `stm_backprop_window` in the reference is exactly where that mislabelling
  would attach.
- **Do not down-weight underfull steps as the bias correction** — rescale the target — **and do not
  call the rescaled target "unbiased."**
- **Do not justify the μP multiplier with the `Θ(d)`-at-init argument.**
- 🔴 **Never change a frozen constant, a bucket edge, a threshold, or drop an arm to make a gate
  pass. If a gate fails, that *is* the result** — and changing one is the most serious thing you can
  do here.
- **Never invent a constant.** `beta`, `nu`, `gamma`, `tau` come from the registry, which **raises**
  and names the experiment that owes the value. **If it raises, the answer is "E0e has not run," not
  a default.**
- **Do not start E1**, and do not tune `γ_b` or `ν`. **All of it sits above a memory that does nothing.**
- **Mention unrelated problems; don't fix them.** Surface them in the return; don't fold cleanup in.

---

## 14. Verification discipline

The three invariants the repo is built on:

1. **§3.7's reduction is reachable by configuration alone**, never by a separate code path.
2. **`src/rsr/constants.py` refuses unmeasured reads.** A `MEASURED`/`DERIVED`/`CONDITIONAL` constant
   read before its source experiment logged a value **raises, naming the experiment.** *This exists
   because the predecessor project died from freezing a number nobody had measured.*
3. **One `RetentionPolicy` protocol for every arm**, so no comparison depends on which code path ran.

And the rules that have each already been paid for:

- 🔴 **Literal command output, or it did not happen.** "Tests pass" is an adjective; `287 passed,
  0 skipped` is a result. **A gate you skipped is a gate that failed.**
- 🔴 **A skipped test is not a passing test.** Report skips separately, always.
- 🔴 **"Did not run" is not "found nothing." Exit 3 is not exit 0**, and `2` is not a pass either.
  *In ML this is the sharpest version of the problem it can be: a run that produced no metric looks
  exactly like a run that produced a bad one.* **Two false clean bills of health have already been
  produced on this project this way.**
- 🔴 **A new gate is not believed green until a mutation has shown it red — and the mutation must
  redden only it.** If nothing reddens it, **the gate adds nothing, and that is the finding.**
- 🔴 **A negative result must say where it searched, and with what command.** *One confident "not
  found" proposed a 3–6 week reimplementation for code that existed.*
- 🔴 **Report spread, never a bare mean. An sd of exactly 0.0000 across seeds is not a clean result,
  it is a broken one** — that is how a five-seed mean from a one-seed run was caught.
  *And the "fix" for it made the code request five seeds while an inner function reset the seed
  every time, so all five were byte-identical. The sd caught that too.*
- 🔴 **Never read `$?` after a pipe.** You get the pipe's status. This shell is zsh: `PIPESTATUS` is
  not defined — it is `$pipestatus[1]`. **This project has filed a false clean bill of health from
  exactly this**, in the same hour a document about that class of error was written.
- **Numbers are measured, never typed.** A human retyping a count is the step where one digit changes.
- **A `provenance:` line in every return:** `<git sha> · <device> · <dataset hash> · <seeds>`.
  *An absolute path is not provenance; a sha is — and in ML the same claim needs three more fields.*
- **Every experiment writes `RESULTS.md`** next to its `run.py`: the command, the SHA, the hardware,
  the seeds, and the numbers — **including the ones that came out wrong.**
- **Pre-registrations and briefs are committed BEFORE the thing they govern.** `git log` is what
  makes that checkable, **which is the entire point.**
- **A researcher's report is not evidence; the ledger is.**
- **"We learned nothing this cycle" is a writable row.** *A loop that cannot report a null night will
  manufacture a result instead.*
- 🔴 **Documented configuration is not evidence of what was actually run.** *A system can be fully
  disabled and still produce confident output — E0c's first numbers timed a model whose memory was
  never written, and produced a perfectly plausible loss.* **Count the thing you assume is happening.**
- ⚠️ **On the Studio, use `/opt/homebrew/bin/git`, never bare `git`.** Apple's git blocks on an Xcode
  licence prompt and fails as an **empty answer rather than an error.**

📌 **Known error rate, recorded because it calibrates how much to trust a brief:** of the overnight
manager's 11 briefs, **7 contained an error a researcher caught** — a false premise from reading code
in isolation, a ratio compared against a flip-rate, a contradiction carried verbatim from the
previous cycle, **two consecutive cycles where the question was a theorem rather than a
measurement**, a threshold invented and attributed to the spec, a stub described as drivable.
🔑 **The most valuable thing a researcher did all night was refuse the conclusion the brief set up
for them.** Make correcting the brief a requirement, not a courtesy.

---

## 15. Machines, repos, branches

| | |
|---|---|
| **Mac Studio** — M4 Max, 64 GB | `<studio>` (Tailscale). **The training machine. All training happens here.** Repo at `~/retrieval-successor-retention` |
| **MacBook Pro** — 16 GB | Advisory. 🔴 **No memory or throughput number measured here transfers.** |
| Remote | `github.com/Keanooo7/retrieval-successor-retention` |

🔴 **Two histories, no common ancestor.** The Studio's is trunk (`origin/main`); the MacBook's 13
commits were scaffolded independently and are preserved as `origin/macbook-local-2026-09-18`.
`git rev-list --left-right --count` returns no merge base — **they cannot be merged by fast-forward**,
which is why the MacBook's evidence was **ported by file** (PR #1), not merged.

⚠️ **Checked 2026-09-18: the MacBook root checkout's local branch named `main` IS the MacBook
history (`df508ee`), not `origin/main`.** A session that trusts the branch name there is working
against a tree missing `code-vs-paper.md`, `mutation-battery.md`, ADR-0003…0007, `third_party/`, 14
test files and every `runs/` ledger. **Check `git rev-parse origin/main`, not the branch name.**

⚠️ **There is no orchestrator on the Studio and nothing there reads `ORCH_PROJECT`.** Earlier copies
of the role files and the launch prompt opened with `export ORCH_PROJECT=rsr`; it was carried over
from the Cleaning-app adapter and is inert — `grep -rn ORCH_PROJECT` over the repo returns **prose
only**. *Recorded because an instruction that looks operational and does nothing is precisely how a
prompt bends.* The **vault** is where the lane/claim machinery lives — see
`Projects/RSR/RESEARCH-CONTEXT.md` in the Memory Palace.

📌 **One risk nothing can protect against:** the Studio has **FileVault** enabled. If power flickers
it stops at a password screen and is unreachable until someone types it in physically.

---

## 16. The map — where everything lives

| Need | Read |
|---|---|
| 🔴 **Corrections that OVERRIDE the spec** | `docs/spec-corrections.md` — **24 entries, read before the spec** |
| What has and has not been checked against sources | `docs/citation-audit.md` (⚠️ see §17) |
| The spec | `docs/spec/rsr_model_spec_v0.5.md` (793 lines) |
| Where the released TG diverges from the paper | `docs/code-vs-paper.md` |
| Decisions | `docs/decisions/ADR-0001…0007` |
| The ratchets and exit codes | `docs/gates.md` (⚠️ intent, not this tree — §10.5) |
| Mutation evidence | `docs/mutation-battery.md` |
| §16 conditions with evidence links | `docs/release-conditions.md` |
| The E0i threshold | `preregistration/e0i_threshold.md` |
| Sprint 1 report | `experiments/GATE-1.md` |
| Per-experiment results | `experiments/e0*/RESULTS.md` |
| The pinned TG reference and its known defects | `third_party/PINS.md` |
| 🔴 **Where the project stands, with the audit's corrections** | `docs/lab-notes/HANDOFF-2026-09-18.md` |
| The overnight record | `docs/lab-notes/overnight-2026-09-18.md`, `morning-2026-09-18.md`, `for-brendan-2026-09-18.md` (⚠️ §11) |
| The next run's contract | `docs/lab-notes/overnight-2026-09-19.md` |
| Two sessions compared, and who was right | `docs/decision-review.md` |
| The path to a first training run | `docs/gauntlet-to-first-training.md` |
| A plain-language account of the project | `docs/RSR-end-of-day-2026-09-17.md` |
| Standing rules | `CLAUDE.md` |
| Lanes, claims, `device:mps0` | **vault** — `Projects/RSR/panes/README.md` |

---

## 17. Defects found in this record while writing this file

**Recorded, not fixed** — per CLAUDE.md, *"mention unrelated problems, don't fix them."* Both were
verified against `~/Downloads/2512.25026v2.pdf` directly.

### 17.1 🔴 `docs/citation-audit.md`'s S-3 promotes a scoped superlative to an unscoped one

S-3 quotes [P2] as *"removing the stream curriculum produces the largest drop (PPL 30.5) …
supporting the role of controlling backpropagation depth"* and concludes **"it is the largest of
[P2]'s ablation drops."**

**It is not.** The paper's sentence is scoped by its own preceding clause. §3.5, verbatim:

> *"The remaining ablations … all increase test perplexity by 0.4–0.7 PPL (about 1–2.4%) relative to
> baseline. **Among these**, removing the stream curriculum produces the largest drop (PPL 30.5)…"*

**"Among these" is the minor ablations only.** [P2]'s actual largest drops are **No working memory,
29.8 → 45.8**, and **Detach sentence reps, 29.8 → 35.0**. The elision in the audit's quote is exactly
where the scoping lived.

**S-3's substantive point is unaffected** — E3 does specify `S = 80, fixed, no curriculum`, the
curriculum ablation is a real degradation, and §13's instinct that the FIFO baseline must be re-run
at `S = 80` stands. **Only the characterisation is false**, and it is the kind a reviewer checks.
**Filing it as a correction is whoever owns `docs/citation-audit.md`'s call, not this file's.**

### 17.2 🔴 CI is red on trunk for **two different reasons**, and only one of them is real

**Measured 2026-09-18 on `d1c221f`, with real exit codes read directly and never after a pipe.**

**(a) Every CI run on trunk is an outage artefact, not a build result.** `gh run list` shows
`failure` on the last six runs — **at 3s, 4s, 3s, 3s, 3s and 5s.** That is far too short to have
installed uv, built a venv, installed torch and run 287 tests. The annotation on run `35381181139`:

> *"The job was not started because recent account payments have failed or your spending limit needs
> to be increased."*

🔑 **Read the run duration before you read the colour. A job that dies in seconds started no step; a
job that runs for minutes is a real result.** Do not read these reds as failures, and **do not read
them as passes either** — nothing has been checked by CI.

**(b) But the lint genuinely is red, locally, and it will surface the moment billing is restored:**

```
$ .venv/bin/ruff check                    # exactly what CI runs — no path args
exit=1 · Found 57 errors                  # 55 E501 line-too-long · 1 RUF022 · 1 RUF100

$ .venv/bin/ruff format --check
exit=1 · 4 files would be reformatted, 118 already formatted
```

The four files needing reformat are **`scripts/canary.py`, `scripts/ledger.py`,
`src/rsr/train/heartbeat.py`, `src/rsr/train/loop.py`**, and the 57 `ruff check` errors are
concentrated in the same three of those plus `canary.py`.

⚠️ **Those are precisely the files §10.1 and §10.5 already name as defective and untested.** They are
the evidence-machinery and training-loop files, and **no CI job has started since they were added**,
so nothing mechanical has looked at them. The suite is unaffected: `pytest -rs` is **287 passed,
0 skipped, exit 0**, measured twice this session.

📌 **This is not introduced by this file's commit** — it touches only Markdown, and the same two
commands return the same exit codes on an unmodified `origin/main` checkout.

### 17.3 `experiments/GATE-1.md` quotes a suite count from a different commit

GATE-1 reports `passed=239 failed=0 skipped=0 errors=0` *"at `3e19dcb`."* Measured at `d1c221f` by
this session: **`passed=287`.** Not a defect — the tree moved — but **GATE-1 carries a stale literal
with a sha attached**, and the sha is what makes it checkable. Re-measure before the report ships.

📌 **And one non-defect worth stating, because it would otherwise get re-litigated:** the copy of the
spec the owner supplied on 2026-09-18 (`~/Downloads/rsr_model_spec_v0.5 (1).md`) is **byte-identical**
to `docs/spec/rsr_model_spec_v0.5.md`. `diff` returns empty. **There is no spec drift.**
