# ADR-0011: the leading-edge baseline runs on the microstructure argument-overlap graph, one sentence per unit, `s = M`

- **Status:** **PROPOSED.** Drafted by an agent (RSR roadmap item C7, PLAN-v4 §2D). **Brendan
  ratifies.** Nothing here is in force until he does. No code, no runs.
- **Date:** 2026-09-30
- **Branch / base:** `docs/adr-leading-edge` from `main` `f7a6b10`
- **Relates to:** spec §5.4, §10.1, §11; `docs/spec-corrections.md` corrections **24, 25, 26, 27**;
  `docs/RESEARCH-CONTEXT.md` §4.8; release condition 6 (`docs/release-conditions.md:19`);
  `docs/citation-audit.md` [P11]; `docs/cognitive-grounding.md` §2;
  `docs/lab-notes/dispatch-leading-edge.md` (the existing implementation brief, `0752f61`);
  queue item `eng-leading-edge`; ADR-0003 (coref), ADR-0005 (E7 stimuli), ADR-0006 (rank order).
- **Depends on, not yet on `main`:** the `on_write` contract on `fix/on-write-slot-index`
  (`f43fbe1`: `MemoryState.row`, and `on_write`'s `slot` is the newcomer's index). Every
  per-row statement below assumes that branch lands as written.
- **Written before:** any leading-edge code. `src/rsr/baselines/leading_edge.py` is still the
  four-`raise` stub.

---

## 0. Source status. Read this first

🔴 **The primary source is not in `~/research-corpus`.** Searched 2026-09-30 by filename and full
text (`kintsch`, `van dijk`, `vipond`, `leading edge`). The only hit is the 09-29 handoff naming
this task. **This session has not read Kintsch & van Dijk (1978).**

So every detail about the paper below is taken from what earlier sessions quoted into this repo:

- `docs/citation-audit.md:201–258`, which says the source was "read in full" from a scan at
  `https://www.cl.cam.ac.uk/teaching/1516/R216/Towards.pdf` (*Psychological Review* 85(5),
  363–394), and quotes p. 379 verbatim.
- `docs/spec-corrections.md` corrections 25–27, quoting the same pages.
- `docs/cognitive-grounding.md` §2.

Every such detail is marked **[UNVERIFIED: repo quote]**. It is a transcription by an earlier
session, and this session has not checked it against the page. **Kintsch & Vipond (1978)** has never
been read by anyone on this project. It exists only as an in-text citation inside KvD
(correction 27; `docs/citation-audit.md:369`), and is marked **[UNVERIFIED: never read]**.
Anything this ADR says about how the graph is built, beyond the p. 379 selection rule, is **this
ADR's interpretation**, and is labelled as one.

---

## 1. Context

§5.4 and §11 make the leading-edge strategy an **implemented baseline**, "the named ancestor". §16
condition 6 requires it. The spec describes it wrongly in three places:

| Spec says | Corrections say | Correction |
|---|---|---|
| keep what is "highest in the **macro**structure" (lines 517, 686) | the rule walks the **micro**structure coherence graph, built by **argument overlap**, cycle by cycle. The macro-operators are a separate, later mechanism | **26** |
| reinstatement search "when a needed proposition has been dropped" | the search triggers on **absent argument overlap between the input set and the buffer**, a cycle-level coherence failure | **26** |
| "Kintsch & van Dijk's leading-edge strategy" | "originally proposed by **Kintsch and Vipond (1978)**". KvD 1978 **implements and validates** it | **27** |
| the levels effect holds "largely independently of recency" (line 641) | the strategy "emphasizes recency and frequency", and recency is one of its two criteria | **25** |

The stub's docstring (`src/rsr/baselines/leading_edge.py:4–5`) still carries the first two
errors. This ADR does not edit it (no code). The implementing PR must.

PLAN-v4's red team (REDTEAM-v2 X-3) ruled this "expensive to reverse, because it defines the named
comparator. It is not TECH-DECIDE." Hence this ADR.

### The rule, as quoted [UNVERIFIED: repo quote, `docs/citation-audit.md:209–214`, p. 379]

> *"The 'leading-edge strategy,' originally proposed by Kintsch and Vipond (1978), does exactly
> that. It consists of the following scheme. Start with the top proposition in Figure 1 and pick up
> all propositions along the graph's lower edge, as long as each is more recent than the previous
> one (i.e., the index numbers increase); next, go to the highest level possible and pick
> propositions in order of their recency (i.e., highest numbers first); stop whenever s
> propositions have been selected."*

And the disclaimer that settles correction 26 [UNVERIFIED: repo quote, p. 379]:

> *"Note that the procedure is strictly formal: There is no claim that topmost propositions are
> always most important or relevant in a more intuitive sense. That will be taken care of with the
> macro-operations described below."*

---

## 2. Decision (proposed)

### D1. Attribution

Every document and docstring cites the strategy as **Kintsch & Vipond (1978), as implemented and
validated by Kintsch & van Dijk (1978)**. Both are cited. Kintsch & Vipond is marked unread wherever
it is cited, until someone reads it. **No text may say TG inherits the KvD connection.** TG does not
cite KvD (correction 27; `docs/cognitive-grounding.md:35`).

### D2. Which hierarchy: the KvD microstructure argument-overlap graph, and only that

The baseline is defined on **the coherence graph over propositions, built from argument overlap**.
It is not built on:

- the **macrostructure** (correction 26). No macro-operator (deletion, generalization,
  construction) appears anywhere in this baseline;
- **Thorndyke (1977)'s story grammar** (setting / theme / plot / resolution). That is a different
  hierarchy and produces a different ranking (correction 27);
- the **semantic-centrality** graph that ships with E0g's chosen stimulus set (NFRD, Lee & Chen;
  `RESEARCH-CONTEXT.md` §8). This is a **third** hierarchy. Correction 27 names two, and this ADR
  adds the third because E0g already picked NFRD.

📌 Correction 27 requires **E0g's stimulus choice to name one hierarchy too**, and to report the
correlation between them. That is an E0g/E7 decision, not this baseline's. It is listed as **Q1**
(OWNER-ONLY). This ADR decides only that *the baseline's* hierarchy is KvD's, because the
leading-edge strategy is defined on that graph and on no other.

### D3. Unit: one sentence is one graph node and one slot

TG's memory unit is a sentence gestalt (spec §3). `select_eviction` chooses a **slot**, and a slot holds
one sentence. The baseline therefore runs on a graph whose nodes are **sentences**. Each node carries
the **union of the argument sets of the propositions in that sentence**.

- **Synthetic corpus:** exact. Every assert, query and filler is one clause with one predicate, so
  one sentence is one proposition (this ADR, §3).
- **Natural text (E7, unapproved):** an approximation. A sentence holds a number of propositions,
  unmeasured on this project (see D5 for why this matters to `s`). The approximation merges a
  sentence's propositions into one node, and so gives them one level and one fate. That is the same
  mixed-importance problem §10.1's "unit mapping" row already names for human norms. It is listed
  as **Q4**.

### D4. The cycle is the TG step, and the protocol forces one departure

One KvD input cycle maps to one TG sentence step. At step `t`, `run_policy_loop`
(`fix/on-write-slot-index`, `policy_loop.py`) does the following, in this order:

1. The **forward** over sentence `t` reads the pre-write memory. This is the "buffer".
2. `observe` runs (only if `observe=True`).
3. If the row is full, `select_eviction(slots, c_t, t)`.
4. Sentence `t` is written (`on_write`, always, per written row).

So at eviction time the policy may use sentence `t`'s propositions, which are the cycle's **input
set**, together with the live slots. That matches a KvD cycle, where selection for the next cycle
happens after the input is integrated [interpretation].

🔴 **Forced departure, stated so nobody discovers it later:** in KvD the `s` selected units are
chosen from buffer ∪ input, so the input can itself be dropped [interpretation of "stop whenever s
propositions have been selected"; the quote does not say the input is protected]. In TG the newcomer
is **always written**. §3.1 changes only the eviction rule, and `write_at` places the newcomer at
`M − 1` or at `k`. The baseline therefore:

- builds the leading-edge pick order over live ∪ {t} with `s = M`;
- evicts the **live** slot that ranks lowest, where never-picked ranks below every picked unit and
  ties go to the oldest (`written_at` smallest);
- **counts** the steps on which the unconstrained rule would have left `t` unpicked
  (`forced_newcomer`). That count is the size of the departure, reported per run, and is never
  folded into a score.

### D5. The `s ↔ M` mapping: two roles, never conflated

KvD's `s` [UNVERIFIED: repo quote, `docs/citation-audit.md:231–233`]: `s = 4` was "arbitrarily
chosen" and fits; `s = 1, 2, 3` give minimum chi-squares "only slightly larger"; at `s = 10` "the
minimum chi-square increases drastically, and the model no longer can fit the data".

**`s` is counted in propositions. `M` is counted in sentence slots.** The baseline has two roles, and
they take `s` differently:

| Role | Where | `s` | Unit | Why |
|---|---|---|---|---|
| **(a) Eviction arm.** A `RetentionPolicy` compared against FIFO, LRU, H2O, RSR and oracle at matched capacity | synthetic (approved), and the E3/E7 arm tables | **`s := M`** (16 synthetic, 8 or 16 for E7's own model) | sentence | Arms must differ **only in the eviction rule** (§3.1, E0b's premise). An LE arm at `s = 4` inside a 16-slot memory would differ in capacity too. `M` comes from the scope (`synthetic`, `corpora`, `e7`); it is never a global value (CLAUDE.md, constants registry) |
| **(b) Human-recall model.** The comparator ordering in correction 25's primary E7 number (3) | E7 only (unapproved) | KvD's fitted range, **`s ∈ {1, 2, 3, 4}`**, all reported, none selected after the fact | **proposition** | That is the model whose survival time *is* KvD's recall prediction (D7). It needs propositional coding of the E7 passages (§7) |

Consequences, stated rather than discovered:

1. **Role (a) on synthetic runs at `s = 16`**, well past the `s = 10` at which the human fit fails.
   So the synthetic arm is *the leading-edge rule at the project's capacity*. It is **not the
   human-fitted model**, and no synthetic result may be described as reproducing the 1978 fit.
2. 🔴 **Correction 26's 📌 ("`s ≈ 1–4` is a better argument for evaluating E7 at `M = 8`")
   compares propositions to sentences.** At [P2]'s ~25 words/sentence (correction 24), one E7
   sentence plausibly carries more than one proposition. That count is **unmeasured** (BELIEVED, NOT
   VERIFIED), so `s = 4` propositions may be fewer sentences than 4, never mind 8. This ADR does not
   edit correction 26. It is flagged for Brendan as **Q3**, because the corrections file wins and
   only he amends it. Until the propositions-per-sentence ratio is measured on E7's own SaT
   segmentation, **`s ≈ 1–4` supports "a small `M`", not "`M = 8`" specifically.**
3. `dispatch-leading-edge.md` says `s ≈ 1–4` is "for the fixture and for E7 at `M = 8`". The fixture
   half is right (it is KvD's own example, in propositions). The E7 half conflates roles (a) and (b)
   and should be reworded when the brief is revised.
4. `s` is **not** a `constants.py` entry. In role (a) it is `M`, which the registry already scopes.
   In role (b) it is swept over the published range. No new constant is invented.

### D6. The reinstatement trigger: counted, never acted on

Correction 26: the trigger is **absent argument overlap between the input set and the buffer**.
Operationally, at step `t` with `t > 0`:

```
fires(t)  :=  args(t) ≠ ∅   and   args(t) ∩ ⋃_{live slots k} args(written_at[k]) = ∅
```

- **Buffer** means the **live slots of the pre-write memory**, the ones the forward just attended to.
  It does not mean the whole graph. 🔴 `dispatch-leading-edge.md` counts a failure as "sharing no
  argument with **any unit in the graph**" in its construction section, and as "sharing no argument
  with **the buffer**" in its `do_not`. These are different quantities once anything has been
  evicted. Correction 26 names the buffer, and this ADR adopts the buffer.
- **An empty argument set neither fires nor satisfies.** Otherwise, under a coding that gives
  fillers `∅`, every filler fires and the count measures filler density. This is a proposal
  (**Q7**, TECH).
- **Not acted on.** TG has no long-term store that a policy can pull from. An evicted gestalt leaves
  the forward view, and `RetentionPolicy` cannot write (its docstring: "It does not own the memory,
  write gestalts, or touch the transformer"). Re-admitting an evicted gestalt would change the write
  mechanism, and with it §3.1 and E0b's premise. So the trigger is a **per-run diagnostic count**
  (`reinstatement_triggers`, alongside `forced_newcomer`), and the baseline is named in every table
  as **"leading-edge, no reinstatement"**. Whether a reinstatement analogue is ever in scope is
  **Q6** (OWNER-ONLY).
- Nothing here claims what KvD do with reinstatement counts beyond the trigger condition. The repo
  quotes nothing more.

### D7. Recall model: survival time is the predictor

[UNVERIFIED: repo quote, `docs/citation-audit.md:235–237`]: *"if a proposition is selected k − 1
times for inclusion in the short-term memory buffer, it has k chances of being stored in long-term
memory, and hence, its reproduction probability will be 1 − (1 − p)^k."*

Mapped onto the loop in D4, with sentence `i` written at step `i` and evicted at step `e` (after
step `e`'s forward):

- `i` is processed once as input. That is 1 chance.
- `i` is present in memory for the forwards of steps `i+1 … e`. That is `e − i` carry-overs, i.e.
  `k − 1`.
- So **`k_i = 1 + (e − i)`**. A sentence never evicted is censored at the stream end:
  `k_i = 1 + (S − 1 − i)`, and the end of the passage is the end of the text.
- Under FIFO this gives `k_i = 1 + min(M, S − 1 − i)`, matching correction 25's
  "`min(M, S−i)`" up to the off-by-one of where "survival" starts. The implementing brief must fix
  one convention, and ADR-0011 proposes the `k` above because it is the quoted formula's own count.

`1 − (1 − p)^k` is strictly increasing in `k` for `p ∈ (0, 1)`. So **every rank-based measure
(E7's (1)–(3)) depends on `k` alone, and `p` never enters.** `p` is a fitted parameter of KvD's
model. **It is not given a value here, not put in `constants.py`, and not hardcoded.** It matters
only for predicting absolute recall levels, which no approved or proposed measure does.

KvD's "emphasizes recency and frequency" [UNVERIFIED: repo quote] needs no extra term in a slot
memory. "Frequency" in the `k` sense is exactly repeated selection, which is survival.

### D8. Plugging into `RetentionPolicy`: a separate arm, not a score term

The leading-edge strategy is **a baseline policy, not a term in RSR's eviction score**. So:

- **No off-switch in `RSRConfig.reduction_to_tg()`**, because nothing is added to RSR's score. The
  CLAUDE.md rule ("every term added to the eviction score needs a documented off-switch") governs
  terms inside `ψ̂ + b`. It does not govern separate arms. §3.7's reduction and `tests/test_reduction.py`
  are untouched. E0b is unaffected by construction, and the implementing PR must not import
  anything from `rsr.retention.rsr`.
- **No gradient, no parameters, no `.to()`.** It reads neither `slots.gestalts` nor `context`. It
  reads `slots.live`, `slots.written_at`, `slots.row`, `step`, and an offline per-sentence argument
  table. It is allowed to use recency, as LRU and FIFO are. The age exclusion applies to `ψ̂` only.
- **No RNG.** Constructing or running it draws nothing from any generator, because `test_reduction`'s
  docstring names RNG ordering as the likeliest cross-arm contaminant. Ties break by `written_at`.

**Inputs.** The policy needs a side channel that the protocol does not carry. Precedent:
`OraclePolicy(demand)` takes one document's ground truth at construction and maps slots to
sentences through `written_at`, which is the sentence index because the loop writes sentence `t` at
step `t`. The leading-edge policy follows that pattern, but **keyed by row**, because `OraclePolicy`
predates `MemoryState.row` and is built one per document:

```python
LeadingEdgePolicy(arguments: Callable[[int, int], frozenset[str]])   # (row, sentence_index) -> args
```

(`dispatch-leading-edge.md` proposes `Callable[[int], frozenset[str]]`, a single stream. That
signature was written before `f43fbe1`, and one policy object now serves every row, so it must
become per-row. Whether a single-document convenience constructor is also offered is **Q9**, TECH.)

**State, per the `on_write` contract (`f43fbe1`).**

| State | Keyed by | Lifetime |
|---|---|---|
| the coherence graph (nodes, levels, edges), **including evicted nodes** | `row`, then **occupant** = `written_at` (sentence index) | cleared in `reset()` |
| `reinstatement_triggers`, `forced_newcomer` counters | `row` | read out by the caller before `reset()` |
| nothing | slot index | never, because compaction shifts indices (ADR-0006, `write_at`) |

- **Ingestion is lazy and idempotent**: "advance row `r`'s graph to step `t`". It is called from both
  `select_eviction` (which needs sentence `t`, still unwritten) and `on_write`. It is **never** called
  from `observe` alone, because `observe` runs only when `observe=True`. A policy that ingested only
  there would silently become a different policy with `observe=False`, and that is gauntlet 0.4's
  exact shape.
- `on_write(slots, slot, step)` asserts `slots.written_at[slot] == step` (the contract's guarantee),
  ingests, and does nothing else. Graph state is per sentence, and a sentence occupies one slot for
  life and never returns. So "a new occupant starts fresh" (the stub's `on_write` docstring) holds
  trivially.
- `observe` is a **documented no-op**, because the 1978 rule reads no attention.
- `reset()` clears every row's graph and counters. `run_policy_loop` calls it once on entry.

**Tests the implementing brief should require** (listed, not written): per-row independence (row A's
graph cannot move row B's victim); compaction invariance (the same stream gives the same victims
whether a mid-memory eviction preceded or not); `observe=True` and `observe=False` give byte-identical
eviction sequences; no global-RNG draw (compare `torch.random.get_rng_state()` before and after);
never evicts a dead slot; and the existing brief's fidelity-fixture and distinguishable-from-FIFO
bars.

---

## 3. Inputs on the synthetic corpus (computable offline, now)

From `src/rsr/data/synthetic.py` at `f7a6b10`. Sentences are `"{E} {P} {O}."` (assert),
`"What does {E} {P}?"` plus ` {answer_symbol(O)}` when `answer_in_stream=True` (query), or one of
eight fixed fillers (`:128–137`). `E` comes from 64 names, `P` from 16 predicates, `O` from 16
objects.

**Proposition.** One per sentence: assert `P(E, O)`; query `P(E, ?)`; filler, the filler's own
clause. The coding is a fixed table over the generator's tuples (table below). It involves **no parser
and no coreference model**, because entities are exact strings and there are no pronouns.

**Argument.** A referent slot of the proposition. **Predicates are not arguments.** Correction 26
says argument overlap. REDTEAM-v2 X-3 counted "entity or predicate" overlap, and this ADR does not
adopt that. Two unrelated facts sharing `owns` are not coherent in KvD's sense, on the literal
reading of "argument" [interpretation; KvD's coding manual is not in the repo].

| Kind | Arguments (proposed primary) | Alternatives, and why not primary |
|---|---|---|
| assert | `{E, O}` | none |
| query | `{E}` | `{E, O}` since the answer symbol is in the text under S0-03. Not primary: the answer is appended after the question, the fact is queried once, and the eviction that decides the answer happens *before* this step's forward. It changes graph levels and trigger counts, not what the query could retrieve. **Q8** |
| filler | the filler's own nominal referent(s), hand-coded once for the 8 fillers (e.g. `The room stayed quiet.` → `{room}`; `Time passed.` → `{time}`; `No one spoke.` → `∅`) | `∅` for every filler, which is what `dispatch-leading-edge.md` specifies. **Not primary**, because deleting filler referents because they are never queried puts *the generator's demand structure* into a baseline that is supposed to be demand-blind. Correction 26: the procedure is "strictly formal". **Q7** |

🔴 **Two consequences of the literal coding, both computable offline and neither measured here:**

1. **Repeated fillers overlap each other.** There are eight fillers and many filler positions per
   document, so the same filler recurs, and on the literal coding repeats share their argument.
   REDTEAM-v2 X-3's "fillers carry no shared arguments" is true only under the `∅` coding.
2. **Asserts are mostly isolated until their query.** Two facts share an entity with probability
   1/64 per pair, and an object with probability 1/16 per pair (uniform draws, `:255–257`). So
   before its query arrives, an assert typically overlaps nothing, or overlaps via an object string
   shared with an unrelated fact. **Prediction (BELIEVED, NOT VERIFIED): on synthetic, the graph is
   near-flat. Most units sit at the top level, and phase 2 of the rule ("highest numbers first")
   dominates, so the arm behaves close to a recency policy.** `dispatch-leading-edge.md` reports an
   "independent simulation" in the same direction under its `∅` coding: 741 assert-evicted-while-
   filler-live events for leading-edge against 605 for FIFO, out of 6,144, seeds 0–2, `M = 16`. That
   figure has **no ledger** and is not re-checked here.

**Object identity.** `"a brass key"` in two facts is an indefinite NP, so the same string may not be
the same referent. The proposed primary treats string identity as the same referent (it is the
simplest formal rule), and flags the indefinites. **Q8**.

**Where the coding lives.** A pure function (`sentence_arguments(Sentence) -> frozenset[str]`),
parsing `Sentence.text` against the generator's own tuples. **Not** a new `Sentence` field:
`to_bytes` serialises `asdict(Sentence)`, so a new field changes the corpus bytes and every dataset
hash in `runs/`. This agrees with `dispatch-leading-edge.md` ("the generator is untouched").

**Computable offline on synthetic, without a model:** argument sets; the graph; levels; the pick
order at any `s`; every eviction of role (a) given the write sequence (writes are fixed: every
sentence is written); `reinstatement_triggers`; `forced_newcomer`; `k_i`; and agreement with FIFO,
LRU-by-proxy and oracle victims. The graph and the rule need no gestalt. Only the comparison
*against RSR's* survival ordering needs a trained model.

---

## 4. Natural text (E7, **unapproved**; design only)

What would need parsing, stated so its cost is not an afternoon:

| Need | Why | Tooling in the repo |
|---|---|---|
| Sentence segmentation | slots are SaT sentences (§10.1) | SaT, per ADR-0005/E0g |
| **Propositionalisation**: predicates and argument slots per clause | the graph's nodes (role b) and the sentence argument unions (role a) | **none**. KvD's propositions were hand-coded [UNVERIFIED: repo does not quote their coding procedure]. An automatic proposition extractor would be a new dependency with its own error rate feeding the comparator (D-2's argument, ADR-0003) |
| **Coreference**, so that argument identity survives pronouns and descriptions | argument overlap *is* referent identity | fastcoref / maverick (ADR-0003), accepted for E0i. Not validated for this use |
| Propositions per sentence | to relate role (b)'s `s` to role (a)'s `M` (D5.2) | measurable once the two rows above exist |

Nothing in this section is authorised by this ADR. E7 is C6 (owner), outside §16's weeks 1–4.

---

## 5. Consequences

- The baseline's meaning is fixed before its code exists, so the implementing PR cannot quietly pick
  the macrostructure reading, or FIFO under another name.
- The "no reinstatement" and "newcomer forced" departures are named in the arm's own label and
  counted per run. A reader can then tell how far the arm is from the 1978 procedure on each corpus.
- On synthetic the arm is expected to sit close to recency (§3, prediction). **If it does, that is
  a finding about the corpus**, whose facts barely cohere, and not about the strategy. E7-type text
  is where the rule has a graph to walk. That is unapproved.
- E7's primary number (correction 25 (3)) needs role (b), and role (b) needs propositional coding
  of the stimuli. That cost belongs in C6's decision, not after it.

---

## 6. Open questions

| # | Question | Class |
|---|---|---|
| **Q1** | Which importance hierarchy does E0g/E7 name as "structural importance": KvD argument overlap, Thorndyke's story grammar, or NFRD's shipped semantic centrality? Correction 27 requires picking one and reporting the correlation between them, and NFRD makes it three | **OWNER-ONLY** |
| **Q2** | Ratify D5's two roles: `s := M` for the eviction arm, and `s ∈ {1…4}` propositions for the E7 comparator | **OWNER-ONLY** |
| **Q3** | Correction 26's 📌 compares `s` (propositions) to `M = 8` (sentences). Amend or annotate it? Only Brendan edits the corrections file | **OWNER-ONLY** |
| **Q4** | For E7, is a sentence-level node (the union of its propositions' arguments) acceptable for role (a), and what mixed-importance fraction voids it (§10.1 already uses ~30% for the human side)? | **OWNER-ONLY** |
| **Q5** | Does E7 role (b) use hand-coded propositions (KvD's own practice, slow) or an automatic extractor (new dependency, unmeasured error)? Part of C6's cost | **OWNER-ONLY** |
| **Q6** | Is a reinstatement *analogue* (re-admitting an evicted gestalt, whose graph is retained per `MemoryState`'s docstring) ever in scope? This ADR says no: it changes the write path, §3.1 and E0b | **OWNER-ONLY** |
| **Q7** | Filler argument coding: literal nominal referents (proposed) or `∅` (the current brief). Also: an empty argument set neither fires nor satisfies the trigger | TECH |
| **Q8** | Query arguments `{E}` (proposed) or `{E, O}`; object identity by string, given indefinite NPs | TECH |
| **Q9** | Constructor shape: per-row `arguments(row, i)` only, or also a per-document convenience matching `OraclePolicy`'s one-policy-per-document use in `experiments/*/run.py` | TECH |
| **Q10** | Graph construction beyond the p. 379 selection rule: which unit is "top"; where a new unit attaches (highest-level overlap **in the buffer**, or anywhere in the graph including evicted units); what "lower edge" and "highest level possible" mean when a level is exhausted; how a disconnected input is placed. **None of this is quoted in the repo.** The implementing brief must quote KvD pp. ~370–379 for each, or label it as interpretation | TECH (blocked on the source) |
| **Q11** | Survival convention: `k = 1 + (e − i)` (proposed, from the quoted formula) against correction 25's `min(M, S−i)`. Fix one convention for E7's three numbers | TECH |
| **Q12** | Obtain the KvD 1978 PDF into `~/research-corpus` and read Kintsch & Vipond (1978) once. Until then every paper detail here is a repo quote | TECH (acquisition), with the reading's conclusions owner-reviewed |

---

## 7. UNVERIFIED items (the complete list)

1. The p. 379 selection-rule quote. [repo quote, `docs/citation-audit.md:209–214`]
2. The "strictly formal … macro-operations described below" disclaimer. [repo quote, correction 26]
3. "Originally proposed by Kintsch and Vipond (1978)". [repo quote, correction 27]
4. **Kintsch & Vipond (1978) itself**: title, venue, content. [never read by anyone on the project]
5. The `s` fits (`s = 4` fits; 1–3 slightly worse; `s = 10` fails). [repo quote, `docs/citation-audit.md:231–233`]
6. The recall formula `1 − (1 − p)^k` and the "`k − 1` times … `k` chances" wording. [repo quote, `:235–237`]
7. "Emphasizes recency and frequency". [repo quote, correction 25]
8. Footnote 6's ranking: recency-only +43%, levels+primacy +23%, random χ²(34) = 113.77. [repo quote, correction 25]
9. The reinstatement-trigger wording, "if there exists some argument overlap between the input set
   and the contents of the short-term memory buffer". [repo quote, `docs/citation-audit.md:224–225`]
10. Everything about how KvD **build** the graph (top node, attachment, levels, disconnected input),
    and how they code propositions and arguments, including whether predicates or embedded
    propositions can be arguments. [not quoted in the repo at all; D3, §3 and Q10 are interpretation]
11. That the input set can itself go unselected in KvD's scheme (D4's "forced departure"). [interpretation of the quote]
12. Thorndyke (1977) beyond its abstract. [`docs/citation-audit.md:248`: "full text not read"]
13. That an E7 sentence carries more than one proposition on average (D5.2). [unmeasured]
14. The dispatch brief's 741-vs-605 simulation. [no ledger]
15. That synthetic graphs are near-flat and the arm is near-recency (this ADR, §3). [prediction, not measured]
16. The `on_write` contract (`MemoryState.row`, newcomer `slot`) is on `fix/on-write-slot-index`,
    **not on `main`** at `f7a6b10`. This ADR's D8 assumes it lands unchanged.

---

## 8. What this ADR does not do

It does not edit the spec, §15, `docs/spec-corrections.md`, `constants.py`, `docs/owner/rulings/`,
the stub, or the dispatch brief. It proposes no value for `p` and adds no constant. It runs nothing.
It makes no claim about human cognition beyond the quoted sources, and it does not say that RSR does
or will reproduce the 1978 fit.
