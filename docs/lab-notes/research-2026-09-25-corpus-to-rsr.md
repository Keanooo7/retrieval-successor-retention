# Research corpus → RSR: candidate mechanisms for the 2026-09-24 measured problems

- **Written:** 2026-09-25, by a model (Claude, overnight session, Track R / R2) working under Brendan's go for the overnight plan (`~/research-corpus/handoffs/2026-09-24-overnight-rsr.md`). **Brendan did not write this, and nothing in it is an owner decision.**
- **Task definition:** `~/research-corpus/handoffs/2026-09-24-overnight-rsr.md:65-104` (R2).
- **Corpus state:** 317 component cards. 316 were added in `b723a44` ("corpus: 316 component cards from 15 sources"), plus the earlier `muon-optimizer` card. Corpus HEAD is `1e3ebff`, which adds only the Phase-0 plan and no cards.
- **RSR state:** `main` working tree (clean at `340dd11`). The run data comes from branch `run/retrieval-curve-2026-09-24` (`runs/retrieval-curve/ledger.json`, `experiments/retrieval-curve/BRIEF-ERRORS.md`, `experiments/retrieval-curve/PREREG.md`).

**Citation discipline.** Corpus citations use a card id plus `sources/<file>:<lines>`. I opened the source lines for every card this note relies on. RSR citations use `path:line`, and ledger values are cited by `key`. `docs/spec-corrections.md` overrides the spec. **Transfer (inference)** marks every step from a card to RSR. **UNVERIFIED -- inference** marks anything that neither the corpus nor the repo states. **PROPOSAL -- owner decision** marks anything that would change the spec, a preregistration, or a gate. This note does not touch spec §15 (`docs/spec/rsr_model_spec_v0.5.md:738` onward). It makes no scaling claim. It does not propose gradient from the retention loss into the transformer or `W_sent`, and it keeps age and `b` out of `ψ̂`.

Abbreviations for source files:
- `CAL` = `sources/decision-models-series/03_Calibration_and_Uncertainty.md`
- `EVAL` = `sources/decision-models-series/06_Evaluation_and_Epistemics.md`
- `FND` = `sources/decision-models-series/01_Foundations_Decision_Models.md`
- `PAR` = `sources/decision-models-series/02_Parallel_Decoding_and_Structured_Outputs.md`
- `RL` = `sources/decision-models-series/04_Reinforcement_Learning_for_Calibrated_Decisions.md`
- `ECON` = `sources/decision-models-series/05_Inference_Economics_and_Systems.md`
- `SERV` = `sources/frontier-model-series/06servinginfrastructure.md`
- `HYB` = `sources/frontier-model-series/02hybridattentioncsahca.md`
- `MUON` = `sources/frontier-model-series/04muonoptimizer.md`

---

## Problem 1 — Memorisation

### Measured facts

| Quantity | ckpt300 | ckpt1000 | ckpt3000 | Source |
|---|---|---|---|---|
| train, live, gap 2..M, answer NLL (per seed) | 2.656, 2.499, 2.497 | 0.00281, 0.00244, 0.00222 | **5.46e-5, 3.88e-5, 6.73e-5** | `ckpt*.train.live.gap_2_to_M.answer_nll` |
| train, live, answer acc | 0.128, 0.197, 0.198 | 1.0, 1.0, 1.0 | 1.0, 1.0, 1.0 | `ckpt*.train.live.gap_2_to_M.answer_acc` |
| held-out, live, answer NLL (mean) | 2.881 | 6.365 | 8.136 | `ckpt*.heldout.live.gap_2_to_M.answer_nll` |
| held-out, live, answer acc (mean) | 0.082 | 0.152 | 0.164 | `ckpt*.heldout.live.gap_2_to_M.answer_acc` |

- The training set is 64 documents. That is about 75 passes at 300 iterations and about 750 at 3000 (`experiments/retrieval-curve/PREREG.md:105-106`, run branch).
- The preregistration names memorisation of the 64 training documents as "the leading rival cause" (`PREREG.md:111-112`).

### Candidate cards

| id | What it is | Source | Transfer (inference) |
|---|---|---|---|
| `modern-nn-overconfidence` | Guo et al.: modern networks are overconfident, mainly because they overfit the loss rather than the error. Error plateaus while cross-entropy keeps falling as probabilities are pushed toward one. The cited contributing factors are capacity, less weight decay, and batch norm. | `CAL:196-206` ("It has stopped learning to be *right* and started learning to be *certain*. That certainty does not transfer to held-out data", `CAL:204`) | The trajectory has this shape. Train NLL goes to about 0 at train accuracy 1.0. Held-out accuracy rises and then flattens (0.082 → 0.152 → 0.164), while held-out NLL grows (2.88 → 6.36 → 8.14). The card describes the mechanism. It does not show that the RSR model has this cause. The cause is untested here. |
| `final-checkpoint-optimizer-ranking` | Loss curves cross during LR decay, so rankings taken at intermediate checkpoints can invert. | `MUON:77` | Any comparison of arms on this substrate depends on the checkpoint, because held-out NLL is still moving at ckpt3000. The card is about optimizers, and applying it to eviction arms is an analogy. |

### What the corpus does not cover

Only one card, `modern-nn-overconfidence`, has `memorization` as a problem. The corpus says nothing about:
- training-set size or number of passes as a memorisation control
- deduplication, early stopping or dropout
- how to tell retrieval apart from memorisation in a memory-augmented model

A text search for "early stop" matches only a speculative-decoding card, and "epoch" matches nothing. The corpus-size question that Track T is testing has no corpus support.

---

## Problem 2 — Overconfidence (held-out NLL above chance; the rule fired on an NLL difference)

### Measured facts

- `chance_ln16` = 2.7726, from 16 answer symbols (`PREREG.md:18`).
- `ckpt3000.heldout.live.gap_2_to_M.answer_nll` = 8.1453, 7.8372, 8.4254.
- `ckpt1000.heldout.live.gap_2_to_M.answer_nll` = 6.2774, 6.2781, 6.5387.
- Both are above chance and still rising (BRIEF-ERRORS item 4).
- The decision rule is: on all three seeds, bar 1 holds and `slots_zeroed − live` answer NLL ≥ `margin` = 0.10 nats (H4) (`PREREG.md:6-7`, `:75-76`, `:84`; margin value `PREREG.md:17`; ledger `margin`). It returned `falsified` at `falsified_at_checkpoints` = [1000, 3000]. The ledger `verdict.detail` reads "longer training makes memory help answers".
- The difference being tested is `ckpt1000.heldout.slots_zeroed_minus_live.gap_2_to_M.answer_nll` = 1.175, 1.713, 1.330 (mean 1.406) and `ckpt3000…` = 1.905, 2.499, 2.382 (mean 2.262).
- BRIEF-ERRORS item 4: the difference "measures zeroing the memory making an already-overconfident held-out loss worse". Bar one "cannot bind when every held-out loss is above chance". "How to read this is the owner's decision. No threshold was changed."

### Candidate cards

| id | What it is | Source | Transfer (inference) |
|---|---|---|---|
| `logarithmic-score` | Log score = cross-entropy. Strictly proper but unbounded below, so "a single confident mistake can dominate everything else" and variance is large. | `CAL:146-148`, `CAL:162` | Answer NLL is the log score. Held-out NLL of 8.1 at a chance level of 2.77 fits the case the card describes, where a few confident errors dominate the mean. That is inference: the per-item NLL distribution is not in the ledger and was not checked. |
| `modern-nn-overconfidence` | Overfitting the loss rather than the error. | `CAL:200-204` | Held-out NLL rose by 1.77 nats from ckpt1000 to ckpt3000 while held-out accuracy rose by 0.012 (arithmetic on ledger means). The card calls this pattern "accuracy is respectable while its confidence is inflated" (`CAL:204`). |
| `brier-score` | Strictly proper and **bounded**: confident mistakes cost a lot, but a finite amount. | `CAL:142-144`, `CAL:162` | A bounded proper score on the 16-way answer distribution would limit how much any single confident wrong answer can add to a memory-ablation difference. |
| `temperature-scaling` | A single scalar fitted on held-out data. It "preserves the ranking … cannot change accuracy". It fixes uniform overconfidence but not selective overconfidence, and it degrades under shift. | `CAL:218-234` | Fitting T on a split separate from the held-out split scored, then recomputing `slots_zeroed − live`, would show how much of the 2.26-nat gap is miscalibration. Caveats: the fit must use a separate split (`EVAL:201`), and a single T cannot fix selective overconfidence (`CAL:232`). |
| `murphy-decomposition` | A proper score splits into reliability, resolution and uncertainty terms. Recalibration moves only reliability: "Recalibrating a model never makes it smarter." | `CAL:172-188` | This separates the two claims mixed together in the `falsified` verdict: "memory adds information" (resolution) and "memory ablation hurts an overconfident model" (reliability). Accuracy and a recalibrated score measure the first. Raw NLL mixes both. |
| `reliability-diagram`, `aggregate-ece-masks-subdomain-miscalibration` | Plot confidence against accuracy with bucket counts shown. Never report a single calibration number. Use a proper score (Brier for unordered categories). | `CAL:46-52`, `EVAL:285`, `EVAL:295` | Report a reliability diagram of the held-out answer distribution, with bucket populations, per checkpoint and condition. |
| `accuracy-not-proper` | Accuracy ignores confidence, so accuracy-only *training* produces confidently wrong models. | `CAL:132-134` | This limits recommendation (f). Accuracy is a reasonable readout next to NLL, not in place of a proper score. See (f). |

### What the corpus does not cover

The corpus says nothing about overconfidence under *memory ablation*. All its calibration material concerns classifiers and decision heads. It also has nothing on decision rules that threshold a *difference* of two NLLs, or on how an above-chance baseline disables a "not below chance" bar. The account of why bar one cannot bind comes from BRIEF-ERRORS item 4, not from the corpus.

---

## Problem 3 — Held-out accuracy depends on memory

### Measured facts

| Held-out, gap 2..M, answer acc | ckpt300 | ckpt1000 | ckpt3000 |
|---|---|---|---|
| live (per seed) | 0.071, 0.105, 0.069 | **0.1485, 0.1646, 0.1431** | 0.150, 0.190, 0.153 |
| slots_zeroed (per seed) | 0.076, 0.084, 0.066 | **0.0636, 0.0633, 0.0715** | 0.082, 0.062, 0.069 |

(Ledger keys: `ckpt*.heldout.{live,slots_zeroed}.gap_2_to_M.answer_acc`.)

- Chance accuracy is 1/16 = 0.0625 (arithmetic from `PREREG.md:18`, "16 answer symbols").
- Zeroing the slots returns accuracy to about chance. With live memory, mean accuracy is about 2.4× chance at ckpt1000 (0.152) and 2.6× at ckpt3000 (0.164). Per seed the ratio ranges from 2.3× to 3.0× (ckpt1000: 2.38, 2.63, 2.29; ckpt3000: 2.40, 3.04, 2.44). Live accuracy is above slots_zeroed accuracy on every seed at both checkpoints.
- At ckpt300 the live/zeroed difference looks inside seed noise (UNVERIFIED -- inference from the per-seed values; seed 2 has live < zeroed; no test was run).
- Item counts per bucket are **not in the ledger**, so no significance test is reported here.

### Candidate cards

| id | What it is | Source | Transfer (inference) |
|---|---|---|---|
| `randomised-blocks-and-paired-tests` | Randomised blocks. Paired tests on the same items: "McNemar's test is the standard choice for comparing classifiers, because it uses the cases where the two systems disagree". Fit on one split, evaluate on another. Publish the logs. | `EVAL:201` | live and slots_zeroed are two conditions scored on the same held-out items, which is the paired design the card describes. A per-item McNemar table for live against zeroed would turn "accuracy is memory-dependent" into a tested claim. That needs per-item logs, which were not checked. |
| `read-the-whole-table-baselines` | Read headline accuracies against simple baselines. A result that ties a trivial baseline is itself the finding. | `EVAL:265-267`, `EVAL:283` | The relevant baselines are the slots_zeroed condition and 1/16. Accuracy of 0.15 is well above both, but 0.15 is still low in absolute terms. |
| `murphy-decomposition` | Resolution is the information a forecast earns. | `CAL:180-184` | Accuracy above chance under live memory and at chance under zeroing is a resolution statement, and it does not depend on the overconfidence in Problem 2. |

### What the corpus does not cover

The corpus says nothing about attributing an accuracy gain to a working memory rather than to memorised documents. That is the Track T question, and it is not addressed.

---

## Problem 4 — D-3, independent slot scoring

### Facts (spec and repository)

- D-3: "Slot value was scored independently; retention value is not separable … greedy argmin over independent scores is not an approximation to it — under redundancy it is anti-correlated with it" (`docs/spec/rsr_model_spec_v0.5.md:62`).
- "`ψ̂` scores each slot in isolation, but what makes a memory worth keeping is the loss it saves *that no other retained slot can save*" (`:288`).
- Current fix: `i* = argmin_i [ z(ψ̂_φ(s_i, c_t)) + b_i − ν · max_{j≠i} cos(s_i, s_j) ]` (`:268`). This is the "minimum viable response", with `ν` swept once on synthetic (`:290`, `:442`). `ν = 0` is part of the §3.7 reduction condition (`:344-346`).
- Falsifier 5 is "Redundancy dominates" (`:150`). Decision attribution is required for `ν` (`:294`).
- The spec's closing line flags as thin "whether greedy argmin over marginal scores approximates the submodular optimum at all" (`:792`).
- The spec's own test of D-3 is ρ(score, LOO Δloss) against E0d (`:189`, `:290`). **E0d has not run.** `src/rsr/metrics/loo.py:21-22` raises `NotImplementedError`, and `docs/RESEARCH-CONTEXT.md:506` records that "§3.2.1's truth rule has **no implemented arbiter**". **D-3 is therefore unmeasured in this repo.**
- `cos(s_i, s_j)` is a plain dot product on unit vectors (`docs/spec-corrections.md:350-351`).

### Candidate cards

| id | What it is | Source | Transfer (inference) |
|---|---|---|---|
| `independent-question-evaluation` | Questions over one shared state are "evaluated simultaneously and independently". No question can condition on another's answer. The source calls this "both the source of the economic advantage and the source of the most serious statistical risk". | `FND:120-128` | This is the same structure as D-3. `ψ̂(s_i, c_t)` is a per-item score against a shared context, with no item seeing its siblings. The corpus names the risk. It does not measure it for memory. |
| `calibration-non-composition` | Calibration is marginal, and chain correctness is joint: "Marginal calibration says nothing whatsoever about joint behaviour". Errors that share causes are positively correlated. | `CAL:296-304` | This is the corpus's closest general statement that per-item quality does not add up to set quality. The card's subject is chained decisions, not retention sets. |
| `decomposition-with-learned-aggregator`, `learned-aggregator-for-decomposed-signals` | Do not multiply per-question probabilities as if they were independent. Learn the combiner on a separate split so it "absorbs the correlation structure". The cheap model went 89.4% → 95.0%, and the strong model went 94.2% → 93.2% (worse). The questions were written after reading the labelling scheme, and the result is unreplicated. | `EVAL:229-261`, `CAL:370-372` | This supports (a) in principle: a learned joint combiner can model correlation that independence ignores. It also carries a warning. When decomposition replaced a strong model's internal integration, performance fell (`EVAL:253-255`). The evidence is one unreplicated study on phishing classification. |
| `distractor-experiment`, `irrelevant-option-shared-denominator-probe` | A probe where one hypothesis predicts **exactly zero** change: add an irrelevant option and check whether the relative preference between the original two moves. It moved, which shows a shared denominator. The probe design is the transferable part. The result rests on one unreplicated study. | `EVAL:177-187`, `PAR:328-336`, `EVAL:303` | This is a direct diagnostic for any set-scoring head. Insert a duplicate of slot *j* and measure the change in the score gap between two *other* slots. An independent `ψ̂` predicts exactly zero by construction. A set-scorer that uses redundancy must move. It is cheap, and it tests whether (a) does what it claims. |
| `lightning-indexer` (**rejected as a model for set scoring**) | A learned cheap scorer `I_{t,s} = Σ_h w_{t,h}·ReLU(q_{t,h}·k_s)`. Its top-k chooses what dense attention reads. | `HYB:74-78` | UNVERIFIED -- inference: no other card found in this pass is a learned, query-conditioned memory selector. It scores each past token independently and selects rather than evicts, so it is an example of the independent scoring that D-3 criticises, not a remedy. |
| `shared-expert-isolation` (**rejected**) | An always-on expert absorbs common knowledge so routed experts need not relearn it. | `sources/frontier-model-series/01mixtureofexperts.md:71` | The card is tagged `redundancy`, but it concerns MoE parameter duplication, not redundancy between retained items. It does not transfer. |

### What the corpus does not cover

The corpus has nothing on:
- submodular set functions, greedy selection guarantees, or marginal-gain scoring
- Deep Sets or Set Transformer-style permutation-invariant scoring
- learned cache replacement

The eviction material (`radix-attention`, `vllm-automatic-prefix-caching`, `SERV:26-28`) is LRU for prefix caches, which RSR already has as a baseline. `learned-gated-kv-pooling` (`HYB:4`, "nothing is evicted, everything is compressed") is the compress-rather-than-evict alternative. The spec already names that as the obvious successor (`:718`), and §11 lists the Compressive Transformer's "compress rather than evict" (`:691`), and it is outside the approved weeks 1–4.

---

## Recommendations (a)–(g)

### (a) Score slots as a set (slot-to-slot attention before the value readout) in place of the ν term

**Verdict: partly supported in principle. Not addressed for memory eviction. Nothing in the corpus shows set scoring beating a redundancy penalty.**
- For:
  - `independent-question-evaluation` (`FND:120-128`) names independent evaluation as a statistical risk.
  - `calibration-non-composition` (`CAL:296-304`) says marginal quality says nothing about joint quality.
  - `decomposition-with-learned-aggregator` (`EVAL:239-241`) says a learned combiner absorbs the correlation that independence assumptions miss.
- Against or limiting:
  - The same card's strong-model result got worse (`EVAL:233`, `EVAL:253-255`). It is one unreplicated study (`EVAL:259-261`).
  - The one learned memory selector found in this pass scores each token independently (`lightning-indexer`, `HYB:74`); that no other exists is UNVERIFIED -- inference.
- Transfer (inference):
  - Set scoring changes *what each score sees*. It does not change the argmin over scores. Greedy argmin over any per-slot score still faces the submodularity gap that the spec flags at `:792`.
  - Constraints, all from the spec or CLAUDE.md:
    - (i) The head must read `s_i`, which carries no positional term, and not the memory keys. Keys carry the rank-indexed `P^(sent)` (`docs/decisions/ADR-0006-sentence-positional-encoding.md:19-20,28-30`), so reading them would bring age or rank into `ψ̂`, contrary to `:232`.
    - (ii) `b` stays outside the head (`:302`).
    - (iii) The head needs a documented off-switch in `RSRConfig.reduction_to_tg()` so that §3.7 and E0b keep holding (`:346`).
    - (iv) Decision attribution applies as it does for `ν` (`:294`).
  - The distractor probe above is the cheap acceptance test.
  - Because D-3 is unmeasured (E0d has not run), there is not yet evidence that `ν` is insufficient.
- **PROPOSAL -- owner decision.** Replacing `ν` changes §3.4's eviction rule, the §3.7 reduction condition and the E1 `ν` sweep (`docs/spec-corrections.md:66`). It needs a changelog block and Brendan. A version that runs alongside the existing arms, adding set scoring as an arm while keeping `ν`, is also a spec change.

### (b) Add Belady's MIN and Parrot to spec §11 as prior art

**Verdict: not addressed by the corpus.**
- The corpus has no card and no source line for either term. `grep -ril "belady\|parrot"` over `sources/` and `components/` returns nothing, and `corpus.py find --text` returns 0 matches for each.
- The spec has no mention of either. `grep -n -i "belady\|parrot" docs/spec/rsr_model_spec_v0.5.md` exits 1, and so does the same grep over `docs/spec-corrections.md` and `docs/RESEARCH-CONTEXT.md`.
- This note does not describe either work, because it has no citable source for them.
- The nearest repo precedent is correction 30 (`docs/spec-corrections.md:722-743`), which adds missing prior work to §11 from a checked primary source. It also records that no paper applying an SR to KV-cache or working-memory eviction was found (`:738-740`).
- **PROPOSAL -- owner decision.** A §11 addition is a spec change. It should follow the E0f pattern: a primary-source check before citation, then a changelog block.

### (c) Batch the leave-one-out Δloss as M+1 masked copies in one forward pass

**Verdict: not addressed by the corpus for LOO. There is support by analogy only.**
- `teacher-forcing` (`PAR:41`): when all inputs are known in advance, every position can be computed "in one batched pass". The M+1 ablations of a step are all known in advance.
- `shared-state-isolated-question-branches` and `shared-state-prefix-reuse-architecture` (`PAR:314-324`, `ECON:199-209`): the state is encoded once and branches attend to it without seeing their siblings. About 1500 questions ran in about 600 ms. That is an inferred architecture from one investigator, unreplicated (`PAR:314`).
- `exact-prefix-kv-reuse` (`SERV:24`): a shared prefix can reuse KV bit-exactly. This is **rejected as a transfer**. Masking slot *i* changes memory cross-attention, so whether any per-layer state is shareable across the M+1 copies depends on TG's layer order. UNVERIFIED -- inference, not checked.
- Transfer (inference):
  - Batching changes throughput. It does not change the O(M) cost per step, so the spec's "only affordable on a subsample" (`:589`; also `src/rsr/metrics/loo.py:13`, and `:715` "validated against LOO on a subsample only") still applies at a different constant.
  - For E0d, batched and sequential LOO should be shown equal to a stated tolerance on CPU before either is used as truth, in the spirit of E0b's bit-exactness.
  - `loo.py` is still a stub (`src/rsr/metrics/loo.py:21-22`).
- **Not a spec change** if it implements §3.2.1's definition exactly. It is an implementation choice inside E0d, which is week 1–2 and approved.

### (d) Make "content+age ≫ age-only" in E2 a preregistered held-out skill score

**Verdict: the evaluation hygiene is supported. "Skill score" as a term is not addressed by the corpus.**
- Spec E2 reads "Partial ρ < 0.7; content+age ≫ age-only" and is a kill gate (`:558`, also `:578`). "≫" has no threshold.
- Supporting cards:
  - `randomised-blocks-and-paired-tests` (`EVAL:201`): fit on one split, evaluate on another, and use paired tests on the same items.
  - `read-the-whole-table-baselines` (`EVAL:265-267`): read results against the simple baseline. The age-only head is RSR's simple baseline.
  - `equal-tuning-optimizer-baselines` (`MUON:75`): an under-tuned baseline is "the single largest source of inflated … wins". The age-only arm needs the same tuning budget as the content+age arm. Transfer (inference).
  - `final-checkpoint-optimizer-ranking` (`MUON:77`): fix the comparison checkpoint in advance.
  - `distractor-experiment` (`EVAL:187`): pick a readout where one hypothesis predicts exactly zero.
  - `base-rate-forecaster-trap` and `murphy-decomposition` (`CAL:62-68`, `CAL:180-184`): the value of the content+age head is its *resolution* beyond the reference forecaster. Transfer (inference): the age-only head plays the role of the base-rate forecaster.
- The corpus's proper scoring rules are defined for *predicted distributions* (`CAL:124-126`). `ψ̂` is a scalar regression head trained by MSE (`:249-250`, `L_MC` per `docs/spec-corrections.md:33`). Whether a proper-score skill metric carries over to a regression head is **not addressed by the corpus** (UNVERIFIED -- inference).
- **PROPOSAL -- owner decision.** This defines a threshold for E2, a week-4 kill gate (`:558`, 'Kills it? Yes'). E2 sits in §16's approved weeks 1–4 (`:773`), but it is not itself one of §16's eight release conditions. It must be committed as its own preregistration commit before E2 runs (CLAUDE.md, "Pre-registration commits land before the experiment they govern").

### (e) Fix the training substrate (corpus size / stopping) before running eviction-policy arms

**Verdict: supported in direction, via the overconfidence mechanism. The specific remedies (corpus size, early stopping) are not addressed by the corpus.**
- `modern-nn-overconfidence` (`CAL:200-210`): an overfit model's confidence "does not transfer to held-out data", and "any system that uses model confidence as a signal inherits this defect".
- Transfer (inference): RSR's eviction arms are judged by next-sentence loss, and E0d's truth rule is a loss difference (`:189`). On a substrate whose held-out NLL is rising while its accuracy is flat (Problems 1–2), both would be read through inflated confidence.
- `confounded-in-family-comparison` (`sources/frontier-model-series/01mixtureofexperts.md:46`) warns against attributing a difference to one component when several changed. That is an analogy for running policy arms while the substrate is also changing.
- The corpus has nothing on data quantity, deduplication or stopping rules, so the choice of *fix* has no corpus support. Track T measures it.
- **PROPOSAL -- owner decision.** This reorders gates relative to spec §8 (`:611` onward). It is labelled a proposal although the substrate work itself sits in approved weeks 1–4 on the Studio.

### (f) Read accuracy, not NLL, where the loss is dominated by overconfidence

**Verdict: supported as an added readout. Contradicted as a *replacement* for a proper score.**
- For:
  - The log score is unbounded and dominated by confident mistakes (`logarithmic-score`, `CAL:148`).
  - Accuracy is invariant to a ranking-preserving rescale (`temperature-scaling`, `CAL:230`), so it isolates the information memory adds.
  - Ledger: the memory effect on accuracy is flat from ckpt1000 to ckpt3000 (live − zeroed means 0.086 → 0.093). The NLL difference grew 1.41 → 2.26 nats (arithmetic on the Problem 2–3 keys).
- Against replacement:
  - `accuracy-not-proper` (`CAL:132-134`): accuracy "is completely indifferent to whether you said fifty-one percent or ninety-nine percent".
  - `aggregate-ece-masks-subdomain-miscalibration` (`EVAL:295`): use a proper scoring rule and reliability diagrams, "never a single calibration-error number".
- The corpus's own recommendation for unordered categories is Brier (`CAL:162`).
- Transfer (inference): report accuracy **and** a bounded proper score (Brier on the 16-way answer distribution) **and** NLL after held-out temperature fitting, alongside raw NLL. That lets reliability and resolution be read separately (`murphy-decomposition`, `CAL:172-188`).
- **PROPOSAL -- owner decision.** Changing the primary readout of the retrieval-curve rule, or of Track T's preregistration, is a preregistration change. BRIEF-ERRORS item 4 already reserves the reading of this run to the owner, and no threshold was changed.

### (g) "Calibrating ψ̂ doesn't matter because argmin ignores monotone rescaling" against "base-model calibration matters for the Δloss targets"

**Verdict: both claims hold, about different objects. As worded, the first is contradicted by the spec's own score. The second is supported by the corpus and by a ledger instance, but it is about E0d's truth quantity, not `ψ̂`'s training target.**

**Claim 1 (argmin invariance):**
- Supported in the narrow case. Dividing all scores by a positive number "cannot change which option is highest, or reorder anything" (`temperature-scaling`, `CAL:230`). Recalibration moves only reliability, never resolution (`CAL:188`).
- The RSR decision is not `argmin ψ̂`. It is `argmin[z(ψ̂) + b − ν·max cos]` (`:268`).
- Transfer (inference, arithmetic): the per-step z-score (`:271`) already removes any increasing *affine* rescale of `ψ̂`. A non-affine monotone transform changes the z-scores and therefore the trade-off against `b` and `ν`. The spec says this itself: "**Scale is not free**" (`:314`).
- So claim 1 holds exactly only when `b ≡ 0` and `ν = 0`.
- Separately, miscalibration that varies across slots (`CAL:232`, "selective overconfidence") is not a monotone transform at all. It reorders slots, and it is a ranking error that no rescaling argument covers.

**Claim 2 (base-model calibration matters for the Δloss targets):**
- The object needs a correction. `ψ̂` is trained on `G_i = Σ γ^k r_i`, built from attention-contribution shares (`:249-250`; `docs/spec-corrections.md:33`), **not on Δloss**. LOO Δloss is E0d's truth (`:189`, `:548`) and the oracle, "built once and used twice" (`src/rsr/metrics/loo.py:7-8`; spec `:189`).
- The claim therefore bears on the arbiter of `r_i` and of D-3, and on any future retention loss built on Δloss.
- Corpus mechanism:
  - The log score is unbounded, and one confident mistake can dominate (`CAL:148`).
  - Overfit models carry inflated held-out confidence (`CAL:200-204`).
  - `synthetic-label-calibration-target` (`RL:302-306`): "The calibration target is the label". Whatever generates the target defines what gets fitted, and its systematic errors are "invisible to the evaluation too". Transfer (inference): an LOO Δloss produced by an overconfident TG inherits TG's miscalibration.
- Ledger instance: the all-slots ablation (`slots_zeroed − live`) is the same construction as LOO applied to the whole memory rather than one slot. That is inference, not LOO itself. Its NLL value grew 1.41 → 2.26 nats between ckpt1000 and ckpt3000, while the accuracy effect barely moved (0.086 → 0.093).
- E0d reports Spearman ρ (`:189`). A rank statistic is unchanged by a *global* monotone transform of the Δloss column. Overconfidence that varies per item can reorder the column. (Spearman's invariance is not in the corpus: UNVERIFIED -- inference.)

**Resolution offered:**
- Compute E0d's LOO Δloss under NLL and under a bounded proper score (Brier). Also compute it under NLL after a temperature fitted on a separate split (`CAL:218`, `EVAL:201`).
- Report ρ for all of them. If they disagree, that is a finding about the arbiter, not about `r_i`.
- Temperature scaling degrades under shift (`CAL:234`).
- **PROPOSAL -- owner decision**, because it touches §3.2.1's truth rule.

---

## Relevance limits

- **The corpus is mostly about LLM decision models and frontier serving.** It has 15 sources: `decision-models-series/01–06` (calibration, scoring rules, evaluation epistemics, a closed "decision model" product), `frontier-model-series/01–08` (MoE, hybrid attention, Muon, serving, quantisation, one vendor model), and a kit index. None is about recurrent memory models, working memory, discourse, or cognitive retention.
- **Memory eviction coverage is thin.** The `memory` model part has five cards, all serving-side KV/prefix caches (`radix-attention`, `shadowradix`, `shadowradix-dual-lock-eviction`, `shadowradix-unverified-extensions`, `hisparse-cpu-kv-offload`). `vllm-automatic-prefix-caching` is tagged `kv-cache`/`inference-systems`, not `memory`, and its LRU free-queue is at `SERV:26`. In the cited lines (`SERV:26-28`, the dual-lock card), the eviction policies are LRU or refcount/lock based; `hisparse-cpu-kv-offload` and `shadowradix-unverified-extensions` were not checked for this (UNVERIFIED -- inference). None is learned, value-predictive or set-aware.
- **Not addressed by the corpus:**
  - learned cache replacement
  - Belady's MIN or Parrot
  - Expire-Span, H2O, Scissorhands, the Compressive Transformer and DNC (all named in spec §11)
  - successor representations
  - submodular selection
  - memorisation versus retrieval in memory-augmented models
- **What does transfer is epistemic, not architectural:**
  - proper and bounded scoring rules
  - the overconfidence mechanism
  - recalibration moves reliability only
  - marginal versus joint
  - paired and split evaluation
  - the zero-effect probe design
- **Evidence grade.** Several cards used here are `source-claimed-unverified`: the learned-aggregator result (`decomposition-with-learned-aggregator`, `learned-aggregator-for-decomposed-signals`), the distractor probe (`distractor-experiment`, `irrelevant-option-shared-denominator-probe`), and the shared-state architecture (`shared-state-isolated-question-branches`, `shared-state-prefix-reuse-architecture`). `synthetic-label-calibration-target`, used in (g), has status `open-question`. The sources themselves call the Jev-related findings unreplicated and single-investigator (`PAR:314`, `EVAL:303`, `EVAL:259-261`). They are weak priors, not evidence about RSR.

---

## Decisions for Brendan (batched)

1. **Reading the retrieval-curve verdict** (BRIEF-ERRORS item 4). Does `falsified`, which fired on an NLL difference that rose with overconfidence, stand as preregistered, with accuracy and a bounded score reported as secondary? No threshold has been changed. — (f), (g)
2. **Readouts for the next preregistration (Track T and later).** Should accuracy, Brier on the 16-way answer distribution, and held-out temperature-fitted NLL be co-reported or co-primary with raw NLL? **PROPOSAL.** — (f)
3. **E0d's truth quantity.** Should LOO Δloss be computed under NLL only, or also under Brier and temperature-fitted NLL, with ρ reported for each? **PROPOSAL** (§3.2.1 truth rule). — (g)
4. **Set scoring in place of, or alongside, `ν`.** This needs a changelog block, a reduction off-switch, decision attribution, and the duplicate-slot zero-effect probe as its acceptance test. Suggested ordering: after E0d measures D-3, since D-3 is currently unmeasured. **PROPOSAL.** — (a)
5. **§11 additions (Belady's MIN, Parrot).** The corpus has nothing on either. If wanted, they need a primary-source check in the E0f/correction-30 pattern, then a changelog. **PROPOSAL.** — (b)
6. **E2's "≫".** Should a numeric held-out threshold be preregistered for content+age over age-only, with equal tuning and a fixed comparison checkpoint, in its own commit before E2? **PROPOSAL.** — (d)
7. **Gate order.** Should substrate generalisation (Track T's outcome) come before any eviction-policy arm? **PROPOSAL.** — (e)
8. **No decision needed:** batching LOO as M+1 masked copies is an implementation choice within E0d, provided batched and sequential results are shown equal to a stated tolerance on CPU first. — (c)
