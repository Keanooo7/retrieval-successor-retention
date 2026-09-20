# Cognitive grounding — what the literature licenses, and what it does not

**Written 2026-09-20, from primary sources read that day (E0f pass 2).** Companion to
`docs/citation-audit.md`, which records the verification; this file records the *reasoning* the
verification permits.

> KEY: **The purpose of this document is to make the cognitive claim defensible to someone who
> knows this literature.** The base model is McClelland's, the framing citation is a 1978 paper in
> *Psychological Review*, and both invite a reader who will check. Every section below is organised
> the same way: **what the source says · what RSR may therefore claim · what RSR may not claim.**
> The third column is the one that matters, and §3 exists entirely because a claim in that column
> was once in the first.

---

## 1. The lineage is real and it is documented in TG itself

TG names the **Sentence Gestalt** model as an ancestor, in its own words (arXiv:2512.25026v2,
Introduction):

> *"TG is also inspired by the Sentence Gestalt model [21], which incrementally maps a word sequence
> onto a single event representation from which role–filler structure can be decoded. TG similarly
> uses sentence boundaries as a structural proxy for thought boundaries."*

[21] is **St. John, M. F., & McClelland, J. L. (1990), "Learning and applying contextual constraints
in sentence comprehension," *Artificial Intelligence* 46(1–2), 217–257** — a PDP model that
comprehends by constraint satisfaction, holding a single distributed "Sentence Gestalt" vector that
is updated word by word and from which role–filler pairs are decoded by query.

**May claim:** RSR operates inside a lineage that runs St. John & McClelland (1990) → the N400
modelling work (Rabovsky, Hansen & McClelland 2018) → TG (2026), and the object it manages — one
vector per sentence, standing for an event — is that lineage's object, not an engineering
convenience.

⚠️ **May not claim:** that TG inherits the Kintsch & van Dijk framing. **It does not cite them.**
TG's discourse-memory citations are the *situation-model* tradition — Radvansky & Zacks 2011,
Jarvella 1971, Zwaan 2016. Adding the KvD connection is a contribution of this project. Describing
it as inherited is a false claim, and an easy one to check.

---

## 2. Kintsch & van Dijk 1978 — what the leading-edge strategy actually is

Read in full 2026-09-20. **Three of the spec's statements about it were wrong; see corrections
25–27.** The corrected picture:

- The strategy walks the **microstructure coherence graph** — built from *argument overlap* between
  propositions, cycle by cycle — **not** the macrostructure. KvD say so explicitly: *"there is no
  claim that topmost propositions are always most important or relevant in a more intuitive sense."*
- It **"emphasizes recency and frequency."** Recency is a selection criterion, not a nuisance.
- It is **Kintsch & Vipond (1978)**'s proposal; KvD implement and validate it.
- Recall probability is **`1 − (1 − p)^k`** for `k` cycles survived — so **survival time in the
  buffer is the model's own predictor of recall.**
- The fitted buffer is **`s ≈ 1–4`**. `s = 10` fails outright.

**May claim:** RSR is a *learned* version of a hand-specified 1978 retention policy over a
capacity-limited buffer, and the two can be compared directly, because both produce an ordering over
sentences and the 1978 model's ordering **is** its recall prediction. The published ranking on human
data — leading-edge > levels+primacy (+23%) > recency-only (+43%) > random (χ²(34) = 113.77) — gives
RSR a yardstick with numbers on it.

⚠️ **May not claim:** that beating FIFO or LRU on recall correlation shows anything about the levels
effect. Under FIFO, survival time is the deterministic `min(M, S−i)`, so the partial correlation
zeroes it by construction; and a *pure-structural* policy is, on KvD's own data, **worse** than the
recency+structure hybrid. The comparator that means something is the leading-edge strategy itself,
which is why §16 condition 6 requires it as an **implemented baseline** rather than a citation.

⚠️ **Also may not claim:** that Thorndyke (1977) supports the same hierarchy. Thorndyke's is a
**story grammar** (setting / theme / plot / resolution). KvD's is an argument-overlap graph. They
rank differently. Pick one, say which, report the correlation between them.

---

## 3. Complementary Learning Systems — what would be a misuse, stated so nobody re-adds it

**McClelland, McNaughton & O'Reilly (1995)**, *Psychological Review* 102(3), 419–457; updated by
**Kumaran, Hassabis & McClelland (2016)**, *TiCS* 20(7), 512–534.

CLS is a theory about **two weight-change systems distinguished by learning rate and
representational overlap**: a hippocampus that learns fast with *sparse, conjunctive* codes, and a
neocortex that learns slowly with overlapping distributed codes and must interleave. The engine of
the argument is catastrophic interference, and its central tradeoff is stated in the paper:
*"reducing overlap avoids catastrophic interference at the cost of a dramatic reduction in the
exploitation of shared structure."*

🔴 **§3.6 of the spec once contained a CLS claim and it was deleted because the mapping was
inverted.** That deletion was correct, and this section exists so the claim does not come back. Four
specific misuses:

1. **Calling TG's slot memory "the hippocampus."** TG's memory holds **activations** with a live
   computation graph. The CLS hippocampus is a *rapid synaptic store* with pattern separation. A
   dense vector in a 40-slot FIFO has no sparsity, no pattern separation and no plasticity. The only
   structural analogue of the cortical side in TG is the weight matrices.
2. **Calling eviction "forgetting" in the CLS sense.** CLS forgetting is interference between
   overlapping distributed codes, or failed consolidation. `argmin`-delete is neither.
3. **Calling anything "consolidation" that is not replay-driven interleaved transfer into slow
   weights.** `CLAUDE.md` already forbids calling a truncated-BPTT window consolidation. `detach`
   removes an item's ability to shape weights while keeping it retrievable — the *inverse* of
   consolidation.
4. **Invoking catastrophic interference.** It is a claim about sequential training on
   non-interleaved item sets. It says nothing about which of 40 activation slots survives a step.

✅ **The one CLS-family link that is defensible, and it is McClelland's own.** The 2016 update
broadens the role of replay, noting it *"allows goal-dependent weighting of experience statistics"*
so that *"the neocortex is not a slave to the statistics of its environment."* And **McClelland,
McNaughton & Lampinen (2020)**, *Phil. Trans. R. Soc. B* 375:20190637, goes further and is
*prescriptive*: material that projects onto known dimensions can be learned rapidly without
interleaving, while a genuinely new dimension needs gradual interleaved learning, so **interleaving
should be focused on the locally relevant branch**.

That is a normative claim about *what to preserve and replay*, at the right level of description,
from the author of the base model. **If the project wants a McClelland-grounded claim about
prioritised retention, this is the citation — not CLS 1995.** One sentence, and stop.

---

## 4. Coherent covariation — an analogy about structure, not an identity

**McClelland & Rogers (2003)**, *Nature Reviews Neuroscience* 4, 310–322. The paper's own
definition: *"Consistent co-occurrence of a set of properties across different objects … having
wings, having feathers, having hollow bones and being able to fly all consistently co-occur in
birds."*

What it predicts: progressive differentiation (superordinate distinctions first), overextension
errors, and — the relevant one — *"it determines the strength with which a given feature contributes
to representational change in a single learning episode. Properties that covary together generate
larger weight changes throughout the network."* Hence: *"The model's sensitivity to coherent
covariation can therefore explain why some sets of properties are easier to learn and remember than
others."*

**May claim:** there is a PDP precedent, from the same lab, for *retention value being a structural
property that is learned rather than stipulated*. That is the structural analogue of RSR's bet.

⚠️ **May not claim** that `ψ̂` *is* coherent covariation. Coherent covariation is a statistic of the
whole training ensemble that governs weight change; `ψ̂` is a context-conditional, per-step estimate
of future retrieval demand. They agree only in that importance is **structural rather than
positional**. Say that much and no more.

---

## 5. Retroactive prioritisation — the claim survives, the timescale does not

See **correction 29**. [P12] and [P13] support the *computational* claim that a stored trace's
retention value can be **revised upward by later context**, and that no accumulated-past-attention
method (H2O) has that property, since its score is a monotone accumulation.

🔴 They do **not** support it on RSR's timescale. Dunsmoor et al. (2015) observed the enhancement
*"following a period of consolidation"* and explicitly **not** *"in an immediate memory test."* Braun
et al. (2018) needed 24 hours. RSR recomputes `ψ̂` every step. **§13 must state the disanalogy**
rather than leave a reader to find it.

Frey & Morris (1997) is faithfully glossed, but the tag **decays in under three hours** and capture
is competition for a limited protein pool triggered at another input. "The gestalt is the tag,
re-scoring is the capture" is an analogy about revisability. It should be labelled as one.

---

## 6. The conceptual move that is actually this project's own

A successor representation is defined over a state that **transitions**. A memory slot does not
transition — it sits until it is evicted. The move that makes an SR definable here:

> **The state that transitions is the reader.**

`ψ̂(s_i, c_t)` is a value function over (slot, reader-state) pairs, and `c_t` — the current gestalt —
is what moves. This is not in [P1], [P2] or [P3]; it is what the spec's §1 means when it says
*"[P1] and [P2] are McClelland's. Nothing in §3 is."*

⚠️ It is also where the project's own prior cuts against it. §2: `ψ̂ = s_iᵀWc_t` is *"architecturally
the same functional form as the cross-attention logit"*, so it may fit "what is being attended to
now" almost for free — which is falsifier 3c and experiment E0h. **The design's most distinctive
idea is also its most likely failure mode, and the spec says so before any data exists.**

---

## 7. Nearest prior work

**Immertreu, Schilling, Kinfe & Krauss (2026)**, arXiv:2605.24585 — an SR trained on natural
language (WikiText-103, multi-horizon future-distribution prediction) from which word-class
structure emerges unsupervised, syntactic at short horizons and semantic at long ones. The closest
published work to RSR's premise. Belongs in §11; see correction 30.

For SR in general, cite **Carvalho, Tomov, de Cothi, Barry & Gershman (2024)**, *Neural Computation*
36(11), the current review of record, alongside Dayan (1993).

🔴 **"Successor heads" (ICLR 2024) are unrelated** — attention heads that increment ordinal tokens.
The paper does not mention successor representations. Name collision only.

📌 **No paper applying a successor representation specifically to transformer KV-cache or
working-memory eviction was found.** State it as "to our knowledge" and re-check before submission.

---

## What this document does not do

It does not make the cognitive claim true. Nothing in the experiment ladder that would test it has
run: E0i did not run, E7 is unapproved, and the model the whole thing sits on currently has a
**working memory that does nothing** (`docs/RESEARCH-CONTEXT.md` §10.3 — a shuffle control moves the
loss by exactly 0.0). This file establishes only that the claim is **stated in a form the literature
supports**, and marks the places where a reader who knows that literature would otherwise have been
right to object.
