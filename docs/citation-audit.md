# Citation audit — E0f, run early

**Date:** 2026-09-17 · **Scheduled for:** week 2 (§8) · **Run in:** week 1

§1 of the spec says ten of fourteen references were unchecked at drafting, calls the check "the
cheapest risk retirement in the project," and adds: *"This note is deleted from the document once
E0f is logged; it cannot still be here at week 12."*

It was run early because the spec's GPU budget, its baseline definitions and two of its kill gates
all descend from claims about [P2], [P5] and [P6]. Building on an unchecked number is how a week-2
task becomes a week-9 rewrite.

**Scope of this pass:** [P2], [P5], [P6], [P7]. The primary sources were fetched and read.
[P1], [P3], [P4], [P8]–[P14] are **not yet checked** and are listed at the bottom as outstanding.
This audit does not discharge §1's note until they are.

---

## [P2] Borazjanizadeh & McClelland — *Modeling Language as a Sequence of Thoughts*

arXiv:2512.25026**v2**, submitted 2025-12-31, revised 2026-01-12. Under review. Stanford.
Fetched 2026-09-17; 20 pages; all quotes below are from `pdftotext -layout` of that file.

### Verified — quote matches the spec

| Spec | Verdict |
|---|---|
| §3.1 / B2's fix: memory bounds the forward view, not backward gradients | ✅ **Verbatim.** App. A: *"While the memory size M limits which sentence vectors are available to cross-attention in the forward pass, it does not limit the backward flow of gradients. … Thus, the maximum backward depth is bounded by the maximum stream length S."* The spec quotes this correctly and B2 is sound. |
| §3.6: detaching at write costs 29.8 → 35.0 PPL, throughput 21 → 24 | ✅ **Exact.** Ablation table: `TG (Baseline) 29.8 / 21 / 85.6M`; `Detach sentence reps at memory write 35.0 / 24 / 85.6M`. Body: *"considerably worsens test perplexity (29.8 → 35.0); despite increasing the throughput modestly (21 → 24) due to the reduced backpropagation depth."* |
| §3.4: [P2] states early sentence representations are "largely uninformative" | ✅ **Verbatim.** §2.3: *"Early in training, sentence representations are largely uninformative; long sentence streams therefore increase compute and optimization difficulty without providing useful long-range credit assignment."* |
| §5.2: E4 curriculum `30 → +12 / 5 epochs` | ✅ **Exact.** §2.3: *"we start at S=30 sentences and increase by +12 every 5 epochs."* |
| §5.1: `M = 40` chosen to match GPT-2's 1024-token context | ✅ **Exact.** *"we set M = 40 to approximate the 1024-token context window of the GPT-2 baseline."* |
| §5.1: `L_max = 64`; gestalt from layer 7 `<EOS>`; 12 layers | ✅ All three confirmed. |
| §4.1 / §4.4: 21 sent./sec on a single A40 | ✅ *"Throughput is measured on a single NVIDIA A40 GPU with no parallelization."* **But see B-2 — the configuration matters.** |
| §3.1: TG evicts the oldest entry (FIFO) | ✅ *"the oldest entry is removed."* |
| §3.2.1: memory gates are depth-stratified ([P2] App. C–D) | ✅ App. C: *"gates are larger in higher layers (e.g., layers 9–11) than in lower layers. This depth-wise stratification suggests higher layers rely more [on memory]."* Also: *"Gates grow over training."* |
| §6 E6: [P2]'s reversal probe | ✅ Father–son **in-context** reversal probe, §3.4. Note it is an inference-time phenomenon, distinct from the training-time reversal curse — the spec's E6 row does not make that distinction. |

### Corrections — the spec states these wrongly

**B-2 · The throughput anchor is off-configuration.**
The `29.8 PPL / 21 sent./sec` row is at **`d_model = 768`, `N ≈ 85.6M` non-embedding parameters**:
*"N ≈ 85M non-embedding parameters (12 layers, d_model = 768) for both models."*
The spec's §4.1 scales its whole budget from that 21 sent./sec, then assumes ≤7 sent./sec at `S = 80`.
But RSR trains at `d ≤ 384` — **≤21.3M parameters, roughly 4× narrower**. Every hour and dollar in the
1,070 GPU-h total inherits an anchor measured on a model the project never trains.
→ E0c must report sent./sec **at the widths that will actually run**, and the budget re-derived from
that measurement rather than scaled from 21.

**B-4 · §5.1's parameter range is mis-transcribed.**
The spec pairs `d_model ∈ {128, 192, 256, 384}` with *"≈0.3M–21M non-embedding params."*
[P2]'s actual sweep: *"vary model size from N ≈ 0.34M to 21.3M non-embedding parameters. We hold
depth fixed at 12 layers and scale width (d_model ∈ {48, 96, 192, 384})."*
**0.34M is `d = 48`, not `d = 128`.** At 12 layers (≈144·d² non-embedding) the spec's own set spans
**≈2.4M – 21.3M**. Checks out against [P2]'s three reported points: 48→0.33M, 384→21.2M, 768→84.9M.

**S-4 · E5 partially re-runs a published experiment.**
[P2] already ran the width sweep the spec calls E5, over a **wider** range (48→384, 8×, vs E5's
128→384, 3×), and published the fits: *"Power-law fits to test loss yield nearly identical exponents
(TG α ≈ 0.081; GPT-2 α ≈ 0.080), again consistent with an intercept shift."*
That is E5's own stated conclusion, already in print. Either extend down to `d ∈ {48, 96}` — those
runs are cheap — or cut E5 and cite Fig 2(b).

**S-3 · E3 adopts [P2]'s worst measured design ablation.**
Ablation table: `No stream curriculum (fixed S=40) → 30.5` — and the body: *"removing the stream
curriculum produces the largest drop (PPL 30.5) … supporting the role of controlling
backpropagation depth for stable optimization."*
E3 specifies `S = 80, fixed, **no curriculum**` — the same choice at 2.7× the stream length.
The spec's §13 instinct (*"the FIFO baseline must be re-run at S = 80 and may itself degrade"*) is
right and understated: there is a published measurement pointing the same way, and it is the largest
of [P2]'s ablation drops.
Further: *"no curriculum is required for gaps `k ∈ (40,64]` to exist"* is **not established**. The
curriculum `30 → +12/5ep` reaches 78 then 90, so a curriculum terminating at ≥80 supplies the same
gaps in late epochs while keeping the measured stability benefit.

**S-5 · §3.3 cites a measurement that does not exist.**
The spec: *"[P2] §4 documents that auxiliary sentence-level objectives are brittle **here**."*
[P2] says it in **related work**, citing the NSP literature: *"auxiliary sentence-level objectives
are often brittle and can even hurt generalization if not tuned carefully [31, 32]."* And TG
deliberately has **no** auxiliary loss — *"does not use a frozen encoder or any separate
sentence-level loss"* — so it never measured one.
The stop-gradient rule is still correct for its other reason (it preserves §3.7's reduction). One of
its two justifications is unsupported and must not be quoted as [P2]'s finding.

**S-6 · No code-availability statement.** See `docs/decisions/ADR-0001-tg-base.md`.

### Facts worth having that the spec does not record

- TG's own headline effect is small: *"a 2–4% reduction in perplexity"*, and `m_D ≈ 1.05–1.08`
  ("GPT-2 requires about 5–8% more training tokens"). RSR's delta is measured against a baseline
  whose own advantage is that size — which is independent support for the ≥3-seed requirement.
- [P2] reports 23.2 test PPL at D = 50M tokens. The project uses 30M-token subsets.
- Batch construction is **uniform token-budget bucketing** (App. B.1), grouping streams by sentence
  count. Any throughput or memory measurement (E0c) that does not replicate it is not comparable.

---

## [P5] Kobayashi et al. 2020 — *Attention is Not Only a Weight*

EMNLP 2020, arXiv:2004.10102.

| Spec | Verdict |
|---|---|
| D-6: `‖α·v‖` omits `W_O`, "which is where head-specific rescaling lives" | ✅ **Correct.** The norm-based measure is `‖α f(x)‖` where the transformation covers `W_V` **and** `W_O`. D-6's fix is faithful to the source. |
| §3.2.1: attention weight alone is not evidence of use | ✅ Correct, and it is this paper's central finding. |

**B-1 · The successor instrument the spec does not name.**
The same authors published *Incorporating Residual and Normalization Layers into Analysis of Masked
Language Models* (EMNLP 2021, arXiv:2109.07152), which extends the analysis "from solely attention
patterns to the whole attention block, including multi-head attention, residual connection, and
layer normalization" — and finds that **token-to-token interaction via attention has less impact on
intermediate representations than previously assumed.** That is a correction to the 2020 method the
spec adopts, by its own authors.

This is not abstract here. [P2] gives each cross-attention layer a **scalar learnable memory gate**:
*"Each cross-attention layer has a scalar, learnable memory gate `g_mem` that scales the
cross-attention increment before it is added back via the residual path."* App. C measures that
these gates **grow over training** and are **larger in deeper layers**.

`r_i = Σ_{l,h} ‖α · W_O v‖₂` measures upstream of `g_mem`. So the layer sum mis-weights layers by a
factor that is both depth-stratified and **moving during training** — and E0d, a kill gate, is
exactly where that bites.
→ E0d computes `r_i` **both ways**, with and without `g_mem^(l)`, and reports both against LOO Δloss.
The spec already requires the per-layer profile before collapsing; this is what to weight it by.

---

## [P6] Zhang et al. 2023 — *H2O: Heavy-Hitter Oracle*

NeurIPS 2023. Fetched the proceedings PDF 2026-09-17.

| Spec | Verdict |
|---|---|
| §5.4: implement both halves, "at H2O's 50/50 split of `M`" | ✅ **Supported.** App. C: *"H2O-256-256 means maintaining 256 Heavy-Hitters and 256 local tokens."* Reference pseudocode: *"select K heavy hitters and K recent tokens."* The "20% heavy hitters" in the abstract is the **total cache budget**, a different axis — do not confuse them. |
| §5.4: "a heavy-hitter-only H2O is a crippled comparator" | ✅ **Confirmed with a magnitude.** App. C.6 / Table 9: *"only retaining the embeddings of H2 or local tokens can't maintain a similar performance as the model using full embeddings, with a performance degradation from 2.85% to 22.75%."* |

**S-1 · §11 omits H2O's central theoretical machinery, and §3.4 argues against it unaware.**
H2O's abstract: *"We formulate the KV cache eviction as a dynamic submodular problem and prove
(under mild assumptions) a theoretical guarantee for our novel eviction algorithm."* Lemma 3.1:
*"Assuming the attention scheme is submodular, then greedily constructing the set S_i … satisfies the
near-optimal property in terms of submodular."* Theorem 4.4 extends it to the cache-limited case;
App. D carries the proofs.

The spec's §3.4 (D-3) argues that *"greedy argmin over independent scores is not an approximation to
[marginal value]"* on submodularity grounds — and §11's entire H2O entry reads *"evicts by
accumulated attention."* **This is S-1's scholarship failure repeated on a second paper:** citing a
work for its mechanism while its central formalism is the direct ancestor of your own argument.
The two are not necessarily in conflict — H2O's guarantee is over attention-score coverage, not LOO
loss, and under stated assumptions — but §11 must engage Theorem 4.4, and falsifier 5's
"regardless of how good ψ̂ is" is stated more strongly than the published theory supports.

**S-2 · H2O's own limitations section documents the age bias — and reports the spec's fix failing.**
App. B.2: *"In H2O, employing the accumulated attention score to evict KV embeddings can lead to a
potential bias favoring the least recent tokens. This bias arises because most previous tokens have
a higher number of attention scores, resulting in a higher accumulated attention score and,
consequently, a greater likelihood of being retained. To address this concern, we conducted an
additional experiment utilizing the averaged attention score to determine which KV embeddings should
be retained. **However, this alternative approach resulted in performance degradation.**"*

Two consequences:
1. §3.5's point 2 derives the same bias for `ū` and fixes it with an EMA **rate**. That fix now has a
   published counter-datapoint in the nearest-neighbour method and must be addressed, not ignored.
   (The cases differ — `ū` feeds a zero-gradient balance controller, not the policy — but the
   difference has to be *argued*, and a reviewer who knows H2O will ask.)
2. **E2's vacuity gate is one-sided.** It gates RSR's score on age partial-ρ. H2O's bias runs
   *anti*-recency. → report age partial-ρ for **every arm**, not just RSR's. It costs nothing and it
   is the only way the vacuity number is interpretable.

**Adaptation, not port.** H2O is an *inference-time* KV-cache method — *"formulate its deployment in
LLM generation as a variant of submodular maximization"*, scores computed "at each decoding step."
TG evicts gestalt slots *during training*, where evicted slots retain their computation graphs.
Porting H2O across that regime boundary is a design decision with no analogue in the source, and it
belongs in §13's limitations as one.

---

## [P7] Sukhbaatar et al. 2021 — *Not All Memories are Created Equal: Learning to Expire*

arXiv:2105.06548.

**B-3 · Expire-Span has hyperparameters and a required regularizer, and the spec budgets neither.**
Expire-Span carries a **ramp length `R`** and a **loss coefficient `α`**, and requires **structured
dropout** — randomly shortening the memory during training — because *without regularizing the model
memory size during training, the model can easily overfit.*

The spec runs it **once, untuned**, at the week-4 gate, against RSR's `γ`×`β` grid (9 configs) plus a
`ν` sweep. Falsifier 6 — *"Expire-Span ≥ RSR on synthetic → stop and write a different thesis"* — is
**the project's stop condition**, and it is adjudicated under a budget asymmetry that favours RSR.

An untuned Expire-Span losing is uninformative; a tuned one winning ends the project. The asymmetry
runs in the direction that protects the project, which is the wrong direction for a referendum.
→ See `docs/spec-corrections.md` B-3 for the fix (equal-budget protocol).

---

## Outstanding — not yet checked

[P1] Cho & McClelland · [P3] Dayan 1993 · [P4] Tensor Programs V · [P8] Compressive Transformer ·
[P9] DNC · [P10] Scissorhands · [P11] Kintsch & van Dijk 1978 / Thorndyke 1977 ·
[P12] Dunsmoor et al. 2015 / Braun et al. 2018 · [P13] Frey & Morris 1997 · [P14] StreamingLLM.

**[P11] is the highest-value remaining check** — it is now §2's framing, the named ancestor, and an
*implemented baseline*, so the implementation depends on reading it. **[P4] is second** — §4.3 is the
one passage the spec claims as the author's own (§15.3), and §15.1 requires deriving it from [P4]
directly.

⚠️ **§1's verification note stays in the spec until this list is empty.** It is not discharged.
