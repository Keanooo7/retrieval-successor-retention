# E0g — name and obtain the E7 stimulus set

| | |
|---|---|
| Kill gate | **Yes, for E7** (§6). Decides §16 condition 4 and falsifier 4 |
| Status | ✅ **PASS — a usable set exists, and it is CC0** |
| Date | 2026-09-17 |
| Method | Literature and repository search. No data downloaded yet — see "identified vs in hand" |

§5.3 required this named and confirmed in week 1, because *"v0.2 scheduled a primary experiment for
week 8 against a dataset whose existence was an open question in its own §14."* And §5.3 says to
**report what was found regardless of outcome**. This is that report.

---

## Recommendation: the Naturalistic Free Recall Dataset (NFRD)

**Raccah, O., Chen, P., Gureckis, T. M., Poeppel, D. & Vo, V. A. (2024).** *The "Naturalistic Free
Recall" dataset: four stories, hundreds of participants, and high-fidelity transcriptions.*
**Scientific Data 11:1317.** doi:10.1038/s41597-024-04082-6

| | |
|---|---|
| **Repository** | **https://osf.io/h2pkv/** (doi:10.17605/OSF.IO/H2PKV) |
| **Licence** | 🟢 **CC0 1.0 Universal** — public domain dedication |
| **Data-use agreement** | 🟢 **None required** |

> 🔑 **CC0 removes the calendar risk that was E0g's real danger.** The plan artifact flagged that
> "identified" and "in hand" are two different week-1 exits and that a data-use agreement is
> calendar time nobody controls. CC0 collapses them: there is nothing to negotiate, no institution
> to wait on, and no approval that can arrive in week 9.

### Contents

| Story | Source | Words | Audio (s) | Participants |
|---|---|---|---|---|
| `pieman` | The Moth Radio Hour | 948 | 489 | 116 |
| `eyespy` | The Moth Radio Hour | 2,318 | 779 | 116 |
| `oregontrail` | The Moth Radio Hour | 2,389 | 743 | 113 |
| `baseball` | LibriVox — *Baseball Joe in the Big League* | 2,088 | 768 | 113 |

229 participants total, each heard two stories and recalled each aloud for ≥4 minutes.
Transcription was human-reviewed, not raw ASR.

### Why it beats the spec's first choice

§5.3 ranked *Sherlock* (Chen et al. 2017) first. NFRD dominates it for this use:

| | Sherlock | **NFRD** |
|---|---|---|
| Stimulus | audiovisual film | **spoken narrative with text transcript** — a text LM can read it |
| Participants | 17 | **229** |
| Licence | per-repository, varies | **CC0** |
| Recall measure | scene-level description | **per-event recall probability with bootstrapped 95% CIs** |

A text language model cannot read a film. Sherlock's recall norms are about a stimulus the model
never receives, which is a unit-mapping problem far worse than the one §10.1 already anticipates.

### How recall probability is defined

Events are segmented by LDA topic vectors + HMM, **with boundaries manually adjusted to the nearest
sentence boundary** — which is why this maps onto TG at all, since TG's slots are SaT-segmented
sentences. Segmentation was validated against 205 independent human boundary annotators on
`pieman` (F1 = 0.72).

Recall probability = **percentage of participants who recalled each event**, bootstrapped 10,000×
for CIs, with a permutation test for significance. Mean recall rates: Pieman 37.5%, Eyespy 31.1%,
Oregontrail 30.6%, Baseball 43.6% — so there is real variance to correlate against, not a ceiling.

---

## 🔴 Three findings that change the experiment

### 1. The premise behind C1 and D-5 is false for this stimulus set — E7 may not need a second model

§5.1's C1 fix reads: *"Human-normed passages run 10–30 sentences, so a 40-slot memory never fills,
nothing is evicted, and every arm retains byte-identical contents — a tautological null."* That
premise does not hold here.

At [P2]'s ~25 words/sentence, the NFRD stories are approximately:

| Story | Words | ≈ Sentences | vs `M = 40` |
|---|---|---|---|
| pieman | 948 | ≈ 38 | marginal |
| eyespy | 2,318 | **≈ 93** | **2.3× — real eviction pressure** |
| oregontrail | 2,389 | **≈ 96** | **2.4×** |
| baseball | 2,088 | **≈ 84** | **2.1×** |

**Three of the four stories exert genuine eviction pressure at `M = 40`.** If E7 runs on those, it
runs on the *same PG-19 model as E3*, and three things follow:

- The `M = 8` second model (D-5) is **not needed** → **≈72 GPU-hours and a whole second training
  run come off the budget.**
- §10.1's capacity-mismatch precondition **disappears** rather than being argued about — there is no
  mismatch when train and eval are both `M = 40`. That also settles the §10.1-vs-§8 conflict the
  plan artifact carries as unresolved: it becomes moot.
- **§16 condition 4 resolves as satisfied without spending anything.** E7 survives; falsifier 4
  stays live; the funding requirement is void.

⚠️ **Verify before banking this.** Sentence counts above are *estimated* from word counts at [P2]'s
ratio. Measure them on the actual transcripts under the same SaT segmentation the model uses. Spoken
narrative tends to shorter sentences than book prose, which would give *more* sentences and only
strengthens the conclusion — but it must be measured, not assumed. `pieman` is marginal at ≈38 and
should be treated as the `M = 8` case if one is wanted.

### 2. 🔴 `baseball` is a likely PG-19 training-set contaminant. Exclude or verify.

The `baseball` stimulus is the first chapter of ***Baseball Joe in the Big League* by Lester
Chadwick, published 1915**, taken from LibriVox; the text is Project Gutenberg eBook **#27584**.

**PG-19 is Project Gutenberg books published before 1919.** A 1915 Gutenberg title is squarely
inside that window.

If the E3 model trained on a PG-19 subset containing this book, E7's retention behaviour on it is
not a generalization test — the model may have memorised the passage. And `baseball` has the
**highest** mean recall rate (43.6%), so it is the story most likely to look like a clean result.

> **Do:** check the PG-19 subset for Gutenberg ID 27584 and for the *Baseball Joe* series before
> training, not after. If present, exclude it from training or drop `baseball` from E7. Record which
> in the E7 pre-registration. `pieman`'s transcript also came from a prior neuroimaging dataset —
> lower risk, but check its provenance too.

### 3. The unit mapping runs the opposite way from what §10.1 assumes

§10.1 anticipates human norms scored **per proposition** (finer than a sentence) and prescribes
aggregating propositions *up* to their containing sentence, reporting the fraction of sentences
carrying mixed high/low-importance propositions.

NFRD is **coarser**: one recall probability per *event*, and each event spans several sentences. So
the mapping is the inverse — every sentence inside an event inherits that event's single value.

Two consequences:

- The "fraction of sentences with mixed importance" check is **not the right diagnostic here**. The
  replacement is the **distribution of sentences per event**, and the analysis must cluster by event
  or treat the event as the unit.
- **Effective N is the number of events, not sentences.** Roughly 20–40 events per story, so on the
  order of **100–150 independent recall values across all four**, not ~300 sentences. That is the
  real sample size for the rank correlation and it must be stated before E7 runs, not discovered in
  review.

---

## Does importance come scored independently of serial position?

§5.3 asks this directly. **No — and no naturalistic free-recall corpus does.**

Recall probability confounds structural importance with position, and the NFRD paper measures the
position effects itself: probability-of-first-recall is significantly elevated for early events in
all four stories, and lag-CRP shows significant clustering at lag 1. The paper's own permutation
test explicitly *assumes* "the base rate of recalling each event is independent of its position in
the story" — an assumption, stated as one.

**This is not a blocker; it is what §10.1's partial correlation is for**: *"Controlling for serial
position — the levels effect is not recency, and neither should the result be."* NFRD supplies
exactly the serial-position data needed to partial it out.

> ⚠️ **But it must be reported as residual variance, not assumed to exist.** E7 should report how
> much recall-probability variance survives partialling out serial position **before** correlating
> anything with the model. If little survives, E7 measures position and the null is **uninformative**
> — the same disposition §10.1 already assigns to a failed comprehension floor.

---

## Alternatives, recorded

| Set | Verdict |
|---|---|
| **Sherlock** (Chen et al. 2017) — OpenNeuro `ds001132`, Figshare | Available, but **audiovisual** and N=17. A text LM cannot read the stimulus. **Not recommended.** |
| **Narratives** (Nastase et al. 2021, *Sci Data* 8:250) — OpenNeuro | 27 spoken stories, 345 subjects, word-level timestamps. Strong stimulus resource, but it is an **fMRI** collection; free-recall norms are not its product. Useful as a *source of additional passages*, not of recall probabilities. |
| **Thorndyke (1977)** | The right construct — *"recall probability of individual facts depended on the structural centrality of the facts"*, which is importance independent of position, exactly what E7 wants. But the materials are 1977 print; per-proposition norms are in figures, not a dataset. **Viable only as a hand-digitised supplement**, and that is a scoped task, not a download. |
| **Kintsch & van Dijk recall protocols** | Same problem, less tractable. Last resort. |

📌 Thorndyke remains worth digitising *later* precisely because it has the position-independent
importance scoring NFRD lacks. It would be the cleanest possible control — and it is small enough to
hand-enter. Not week 1.

---

## Gate status

✅ **PASS.** E7 is not cut. Falsifier 4 stays live. §16 condition 4 is satisfiable, and on current
evidence satisfiable **without funding a second model**.

## Outstanding — "identified" vs "in hand"

- [ ] Download from OSF and checksum. (CC0 — no agreement, no wait.)
- [ ] **Measure** sentence counts per story under the project's SaT segmentation. Confirm or refute
      finding 1 before removing the `M = 8` model from the budget.
- [ ] **Check PG-19 for Gutenberg #27584** and the *Baseball Joe* series. Before training.
- [ ] Count events per story; fix the analysis unit; state effective N.
- [ ] Pre-register the event→sentence aggregation rule and the position-partialling procedure
      **before** looking at any model output (§10.1 requires it pre-registered).
