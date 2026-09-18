# cycle-dup-victim-cost — the tie is **not** cheap. Which near-copy dies flips a ground-truth retrieval from 1.000 to 0.000, the argmin chooses between them at exactly chance, and `ν` multiplies how often that coin is flipped by **24.8×**

- **Falsifier.** *"A tie that `ν` creates is cheap — when the two lowest-scoring slots are
  near-copies at cosine ≥ 0.9, which one is evicted does not change what the model can
  later retrieve."*
- **Verdict: FALSIFIED**, on the cost limb, in a synthetic task that has a ground truth.
  1. **The choice changes retrievability completely.** In the ground-truth unit — a 2AFC
     retrieval hit, **chance 0.5** — evicting the *not-needed* member of a `cos = 0.90`
     pair leaves the later query resolving correctly at **1.00000 ± 0.00000**; evicting
     the *needed* member leaves it at **0.00000 ± 0.00000**. Paired delta
     **1.00000 ± 0.00000** against a null of **0**. The wrong choice does not merely lose
     the answer, it lands **below the 0.5 chance floor**, because the surviving near-copy
     answers confidently with the *other* detail. Rows `partB_hit_{KEEP,EVICT}needed_wq0.9`,
     `partB_hit_DELTA_wq0.9`.
  2. **Nothing in the eviction rule resolves it.** The argmin restricted to the pair evicts
     the needed member at **0.49860 ± 0.01613** against chance **0.5** (n = 2000 × 5), and
     the per-trial indicator is **bit-identical at `ν` = 0, 1 and 4**
     (`partF_within_pair_victim_is_nu_INVARIANT_exact: True`) — because the max-cosine
     penalty is **exactly equal in float32 within the pair on 80000/80000 pairs**
     (`partF_LEAK_highn_maxcos_tie` = 1.00000 ± 0.00000).
  3. **`ν` multiplies the exposure.** `P(the unrestricted argmin's victim is a pair
     member)` goes **0.03490 ± 0.00466 → 0.85350 ± 0.00896** as `ν` goes 0 → 4, so the
     expected hit-loss *per eviction* goes **0.01743 ± 0.00266 → 0.42552 ± 0.01290**, an
     inflation of **24.79910 ± 3.24326×** (`partF_expected_hitloss_inflation_nu4_over_nu0`).
     `ν` does not break the tie; it manufactures more of them, and each one is now a
     measured coin flip on a total loss.
- **The brief contains an error and I am stating it — two, in fact.**
  1. **The falsifier's own operationalisation is a category error.** The brief writes:
     *"The claim dies if which member … **does** change downstream retrievability at a
     rate above chance"*, and rule 6 says *"a rate that beats zero is not a finding when
     chance is 0.5."* Those apply to **item 2** (does the argmin *pick* the needed member —
     a 2AFC, chance 0.5). They do **not** apply to **item 3**, the cost question, which is
     a **paired difference** whose null is **0**, not 0.5. There is no 0.5 baseline for
     "does the choice matter". The two questions have different nulls and the brief fuses
     them into one sentence. Both are answered separately below, each against its own null.
  2. **Items 1 and 2 are in tension, and satisfying item 1 makes item 2 a theorem rather
     than a measurement.** If the generator genuinely does not leak (item 1), then the
     memory and `c_t` are *exchangeable* in the two pair members, and **every** function of
     the pre-query state picks each with probability exactly 0.5 — by symmetry, before any
     code runs. Item 2's above-chance measurement can only be non-trivial if the need is
     predictable from the discourse state, which is RSR's whole premise — and that would
     require a **trained `ψ̂`**, which cannot exist here because `RSRPolicy.observe()`
     raises (`phi_is_untrained_observe_raises`). So item 2 is blocked from both sides. I
     resolved it by demoting item 2 to a **verification of the leak control** and adding
     an explicit **positive control** where the signal is deliberately injected, which is
     the only way "0.5" carries information rather than being a tautology.
  3. **Smaller:** the brief says *"`src/rsr/metrics/loo.py` exists"*. The **file** exists;
     `loo_delta_loss` is a `NotImplementedError` stub (row `repo_loo_harness_status`), so
     the repo's LOO path **cannot be driven at all**. The LOO here is mine, in this task's
     unit, and is labelled as a substitute everywhere.
  4. Also recorded: **the `cos ≥ 0.9` threshold is the brief's, not the spec's.** A grep of
     `docs/spec/rsr_model_spec_v0.5.md` finds no near-duplicate cosine threshold anywhere.
     Row `bypass_cos_threshold_0.9`.
- Ledger: `runs/cycle-dup-victim-cost/ledger.json` — **159 rows; every number in this file
  is one of them.** 41 statistic rows have `sd == 0.0`, all 41 classified by a classifier
  running in the same process; **`sd_zero_rows_unclassified` is `[]`** (§8 below).
- Provenance stamped by the ledger itself: `git_sha 03f71c6` (the brief's baseline),
  `git_dirty: true` — that is **this untracked run directory only**.
  `/opt/homebrew/bin/git status --porcelain` shows nothing but `?? runs/cycle-dup-victim-cost/`;
  `git diff --stat HEAD` is empty. `src/`, `tests/`, `docs/` untouched; nothing committed.
  Python 3.12.13, macOS-26.6.2-arm64, CPU, 1 thread, **42.3 s wall**.
- Scratch harness (throwaway, per the rules):
  `…/scratchpad/run_dup_victim_cost.py` + `…/scratchpad/lib_dupcost.py`.
- **Geometry read from the registry, never supplied:** `M = 40` (`pg19_e3`), `M = 16`
  (`synthetic`), `S = 48` (`synthetic`), `b_max = 1.0` (FROZEN).
  `get("nu","synthetic")` **raises `UnmeasuredConstant`, naming E1** (row
  `nu_registry_read_raises`) and `get("S","e7")` **raises `ScopeRequired`** (row
  `S_e7_registry_read_raises`). **Every `ν` here is a deliberate, visible bypass; no `ν`
  value is reported as a measurement of that constant.**
- **Other labelled bypasses**, all in the ledger: `d_model = 384`, `L = 6` cross-attention
  layers (correction 18), `H = 8` heads, softmax sharpness 12.0, `g_mem = linspace(0.3,1.2,6)`
  (TG-learned, App. C's depth profile, supplied), `W_K` **tied to** `W_Q`, and the `0.9`
  cosine threshold.
- **§13.** One synthetic generator, dimensionless rates and paired differences within it.
  **There are no PG-19 gestalts in this repo and none were used**; there is no corpus and
  no trained TG. Row `section_13_scope_statement`, written before any number.

---

## 1 — The task, and what it is a model of

A memory of `M` slots. Two of them are a **near-duplicate pair**:

```
u, a, b   orthonormal (u = shared topic, a/b = the two discriminating details)
eps       = sqrt(1/c - 1)
s_A       = (u + eps*a)/sqrt(1+eps^2)      s_B = (u + eps*b)/sqrt(1+eps^2)
cos(s_A, s_B) = 1/(1+eps^2) = c            EXACTLY
```

The other `M−2` slots are i.i.d. unit gestalts. All gestalts are unit-norm (correction 15).
A fair coin, drawn from an independent stream, names **one detail** — `a` or `b` — as the
one a later query will need. Eviction happens first, at which point the policy sees only
the memory and `c_t`. The query arrives afterwards.

The **only** thing that makes this task non-vacuous is that a pair at `cos = 0.9` is not
two copies of one thing. Row `orthogonal_residual_norm_at_cos_0.9` = **0.43589**: at the
brief's own threshold, **44% of a unit gestalt's length is not shared with its near-copy.**
That residual is where a discriminating detail lives, and evicting the wrong member deletes
it. This is arithmetic, not a construction choice — it holds for *any* pair at that cosine.

**What is a construction choice, and is the honest boundary of this cycle:** whether real
sentence gestalts at `cos ≥ 0.9` in fact differ in a detail that a later query needs, or
differ only in paraphrase noise. This experiment cannot answer that. See §9.

## 2 — Ground-truth unit, and why it is a 2AFC

After eviction the reader gets back the attention-weighted memory mixture
`m = Σ_i α_i s_i`, averaged over the 6 × 8 cross-attention heads. A **hit** is
`⟨m, d_target⟩ > ⟨m, d_other⟩` — the needed detail is more present in what came back than
the near-copy's detail. **Chance is 0.5.** This is a genuine two-alternative forced choice,
not a rate against zero, and it is stated in the ledger (`ground_truth_unit`) before the
numbers.

The cross-attention head is **measured, not asserted, to be retrieval-competent**:
`partB_alpha_mass_best_distractor_wq0.9` = **0.00062 ± 0.00001**, i.e. essentially none of
the attention lands outside the pair.

## 3 — The cost of a wrong choice. Paired, null = 0.

`M = 40`, pair cosine 0.90, 2000 trials × 5 seeds. `wq` is the query's weight on the shared
topic; `wq = 1.0` is a **purely topic-level query** ("tell me about this topic") whose
answer nonetheless lives in the detail.

| `wq` | hit, **kept** needed | hit, **evicted** needed | paired Δ (null 0) | α on needed | α on other |
|---|---|---|---|---|---|
| 0.70 | 1.00000 ± 0.00000 | 0.00000 ± 0.00000 | **1.00000 ± 0.00000** | 0.90375 ± 0.00020 | 0.07912 ± 0.00017 |
| 0.90 | 1.00000 ± 0.00000 | 0.00000 ± 0.00000 | **1.00000 ± 0.00000** | 0.80871 ± 0.00041 | 0.18593 ± 0.00042 |
| 0.98 | 1.00000 ± 0.00000 | 0.00000 ± 0.00000 | **1.00000 ± 0.00000** | 0.65792 ± 0.00066 | 0.33782 ± 0.00067 |
| 1.00 | 1.00000 ± 0.00000 | 0.00000 ± 0.00000 | **1.00000 ± 0.00000** | 0.49754 ± 0.00084 | 0.49739 ± 0.00084 |

Continuous unit (`⟨m, d_target⟩`, `wq = 0.9`): **0.31410 ± 0.00002** keeping the needed
member, **0.00032 ± 0.00001** evicting it — a factor of **980**.

**The negative control, which is what makes this a *detail* effect and not a "one fewer
slot" effect.** Recovery of the **shared topic** `u` barely moves:

| `wq` | shared-topic recovery, kept | evicted | difference |
|---|---|---|---|
| 0.90 | 0.94222 ± 0.00008 | 0.92369 ± 0.00023 | 0.0185 |
| 1.00 | 0.93877 ± 0.00013 | 0.93876 ± 0.00012 | **0.00001** |

Rows `partB_NEGCONTROL_shared_topic_recovery_{KEEP,EVICT}_wq*`. **If the later query had
needed the shared content — which is exactly §3.4's defence of the redundancy term — the
tie would indeed be cheap: the cost is 1e-5.** It is the *residual*, not the pair, that
makes the victim choice expensive.

**Cosine sweep** (`wq = 0.9`), achieved cosines exact by construction:

| pair cos | 0.900 | 0.950 | 0.990 | 0.999 |
|---|---|---|---|---|
| paired hit Δ | 1.00000 | 1.00000 | 1.00000 | 1.00000 |

Rows `partB_cos_sweep_hit_DELTA_*`. The cost does **not** decay as the pair approaches
identity, because the readout is a 2AFC on the residual and the residual, however small,
is the only thing that carries the answer. Tightening the near-duplicate threshold is
therefore **not** a mitigation. `partB_M16_hit_DELTA` = 1.00000 ± 0.00000: the same at
`M = 16` (`synthetic`).

## 4 — The argmin picks at chance. Chance is 0.5, stated.

Restricted to the pair, `P(the argmin evicts the needed member)`, `n = 2000 × 5`:

| `ν` | 0 | 1 | 4 |
|---|---|---|---|
| P(evicts needed) | **0.49860 ± 0.01613** | **0.49860 ± 0.01613** | **0.49860 ± 0.01613** |
| P(victim ∈ pair) | 0.03490 ± 0.00466 | — | 0.85350 ± 0.00896 |

`0.49860` is **0.09 sd from chance**. It is identical across `ν` **to the bit**, not merely
to the reported precision: `partF_within_pair_victim_is_nu_INVARIANT_exact` is `True`, a
`torch.equal` on the per-trial indicator across every trial of every seed. The mechanism is
`partC_within_pair_maxcos_penalty_abs_diff` = **0.00000 ± 0.00000** and
`partF_LEAK_highn_maxcos_tie` = **1.00000** (80000/80000 pairs, float32 bit-equality). This
is cycle 9's finding in its strongest available form.

**Expected cost per eviction** = `P(victim ∈ pair) × P(evicts needed | in pair) × Δhit`:

| `ν` | 0 | 4 | inflation |
|---|---|---|---|
| expected hit-loss per eviction | 0.01743 ± 0.00266 | 0.42552 ± 0.01290 | **24.79910 ± 3.24326×** |

Every number in `partA_selftest_vec_score_matches_RSRPolicy_nu{0,1,4}` is
**1.00000 ± 0.00000** — the vectorised `argmin[(ψ̂−mean)/std(unbiased=False) − ν·max_cos]`
reproduces `rsr.retention.rsr.RSRPolicy.select_eviction`'s victim on every memory, which is
what licenses every argmin number here.

## 5 — How I stopped the generator leaking the answer, and the control that proves it

**The proof.** `a` and `b` are the Gram-Schmidt orthonormalisation of two i.i.d. Gaussians
in `u`'s orthogonal complement — i.e. a **uniform random orthonormal 2-frame**, whose law is
invariant under swapping the two vectors. `s_A` and `s_B` are the same function of `(u,a)`
and `(u,b)`. The pair's slot indices are a uniform permutation. The target coin is drawn
from an independent stream. `c_t` is built from `u` and a fresh noise vector, never from the
target. The joint law of `(memory, c_t)` is therefore **exchangeable in the two members**,
so *every* function of the pre-query state evicts each with probability exactly **0.5**.

**The empirical control, at n = 80000 pairs (20 seeds × 4000 blind trials):**

| rule (evict-the-needed rate; chance 0.5) | rate | pooled z vs 0.5 |
|---|---|---|
| `cos(s_i, c_t)` — evict the one less similar to context | 0.49641 ± 0.00758 | **−2.03** |
| `ψ̂` — the (untrained) bilinear value head | 0.49862 ± 0.00645 | **−0.78** |
| max-cosine | degenerate — see below | — |

Rows `partF_LEAK_highn_*`, `partF_LEAK_highn_*_pooled_z_vs_chance`. **No geometry-only rule
beats chance. The generator does not leak.**

**The max-cosine rule is degenerate, not protective.** `partC_LEAK_blind_rule_maxcos_evicts_needed`
reads **0.00000 ± 0.00000**, and that is an artifact I am flagging rather than banking: the
strict `>` is being evaluated on a bit-exact tie, so it is always `False`. It must **not**
be read as "the redundancy term protects the needed member". The classification is in
`sd_zero_rows_classified`.

**The positive control — the row that makes "0.5" mean something.** Re-run with `c_t`
deliberately carrying the target detail:

| rule | blind `c_t` | **leaky `c_t`** |
|---|---|---|
| `cos(s_i, c_t)` | 0.51200 ± 0.01117 (0.49641 at n=80000) | **0.00000 ± 0.00000** |
| `ψ̂` (untrained) | 0.49860 ± 0.01613 | 0.48660 ± 0.02175 |

Rows `partC_LEAK_{blind,leaky}_rule_*`. The harness **can** detect a far-above-chance rate
when the signal is present — the context rule goes from chance to never evicting the needed
member. And the **untrained `ψ̂` stays at chance even when the signal is right there in
`c_t`**, which locates the blockage precisely: it is the untrained head, not the geometry.
This is the strongest statement item 2 of the brief admits given `observe()` raises.

## 6 — Proxy vs LOO. §3.2.1 reached, with one substitution I am naming.

**The repo's LOO harness could not be driven**: `rsr.metrics.loo.loo_delta_loss` raises
`NotImplementedError("E0d / kickoff T6. Spec section 3.2.1.")` (row
`repo_loo_harness_status`). There is also no trained TG, so "Δ next-sentence loss" — LOO's
spec definition — does not exist in this repo at all. **What I computed is a genuine
leave-one-out in this task's own ground-truth unit** (ablate slot `i`, re-softmax, measure
the change in the 2AFC margin), for every one of the `M = 40` slots. That is a **labelled
substitute for `rsr.metrics.loo`, not that function.**

**The proxy is the real one.** `r_i` is
`rsr.retention.reward.retrieval_demand(trace, n_live=40, capacity=40, gated=True)` called on
a real `AttentionTrace` with `alpha [6,8,40]`, `wo_v [6,8,40,384]`, `gate [6]`,
`eval_mode=True` (correction 20 — the function refuses a train-mode trace), six
cross-attention layers (correction 18), memory gate included (correction 17). `gated=False`
is reported alongside, as correction 17 requires.

| | `wq = 0.9` (query carries the detail) | `wq = 1.0` (topic-level query) |
|---|---|---|
| **LOO** discriminates the pair (chance 0.5) | **1.00000 ± 0.00000** | **1.00000 ± 0.00000** |
| **`r_i`** discriminates the pair (chance 0.5) | 1.00000 ± 0.00000 | **0.50300 ± 0.03493** |
| `r_i` **ungated** discriminates | 1.00000 ± 0.00000 | 0.49800 ± 0.04791 |
| `r_i(needed) / r_i(other)` | 4.39488 ± 0.03314 | **1.00635 ± 0.00680** |
| sign agreement `r_i` vs LOO, within pair | 1.00000 ± 0.00000 | **0.50300 ± 0.03493** |
| P(needed is the **global** LOO argmax; chance 1/40) | 1.00000 ± 0.00000 | 1.00000 ± 0.00000 |
| P(needed is the **global** `r_i` argmax; chance 1/40) | 1.00000 ± 0.00000 | **0.50300 ± 0.03493** |
| Spearman ρ(`r_i`, LOO) over all 40 slots | −0.59951 ± 0.00513 | −0.00279 ± 0.00733 |

**§3.2.1's verdict, applied.** At `wq = 1.0` the proxy and LOO **disagree**: LOO identifies
the needed member on **every** trial (and it is the argmax of all 40 slots), while `r_i` —
gated and raw — sits at **chance**, and the two agree on sign only 50.3% of the time.
§3.2.1's rule is that **LOO is truth and `r_i` is the confound**, and that §3.4's redundancy
term is the specified response. §4 above shows that response is `ν`-invariant inside the
pair, so **the specified response does not reach this failure.**

The mechanism is not subtle: `r_i` is a norm-weighted function of the attention `α`, and at
a topic-level query the two near-copies receive **α = 0.49754 vs 0.49739** — equal to three
decimal places. **A retention target built from attention cannot see a distinction the
attention does not make.** No amount of training `ψ̂` fixes that, because `ψ̂` is regressed
onto `r_i`.

*Read the Spearman rows with their diagnostic, not alone.* `partD_spearman_ri_vs_LOO_DISTRACTORS_ONLY_wq0.9`
= −0.70356 ± 0.00599: among the 38 non-pair slots, `r_i` is the slot's small attention mass,
and removing such a slot renormalises that mass **onto** the pair, which *helps* — so `r_i`
and LOO Δloss run opposite there by a property of softmax attention, and 38 of 40 ranks
dominate the correlation. The all-slots ρ is that effect, not a statement about the pair.
I am flagging it rather than headlining it.

## 7 — Re-execution

One command, self-contained, reads the snippet **out of the ledger it wrote** and prints its
own numbers (~3.5 s, CPU):

```bash
cd /Users/keanooo7/retrieval-successor-retention && .venv/bin/python -c "
import json
rows = json.load(open('runs/cycle-dup-victim-cost/ledger.json'))['rows']
exec(next(r['value'] for r in rows if r['key'] == 'partE_reexec_snippet'))
"
```

It prints, for `wq ∈ {0.9, 1.0}` at 400 trials × 3 seeds, with `CHANCE … IS 0.500` on the
first line: hit rates of **1.0000 ± 0.0000** (kept needed) and **0.0000 ± 0.0000** (evicted
needed); `argmin EVICTS needed member` **0.5183 ± 0.0449** at every one of `ν = 0, 1, 4`;
`P(victim in pair)` **0.0342 → 0.1425 → 0.8508**; `LOO … discriminates` **1.0000 ± 0.0000**;
and `r_i … discriminates` **1.0000 ± 0.0000** at `wq = 0.9` versus **0.5050 ± 0.0200** at
`wq = 1.0`, with `r_i(needed)/r_i(not-needed)` **4.3819 → 1.0059**.

Those exact values are ledger rows `partE_snippet_*`, written by executing that same text in
the harness process. They are at 3 seeds × 400 trials and so are noisier than the 5 × 2000
`partB`/`partC`/`partD` rows — `0.5183 ± 0.0449` versus `0.49860 ± 0.01613`, both at chance.

## 8 — `sd == 0.0` rows

41 statistic rows carry `sd` exactly 0.0. **All 41 are classified in the same process that
wrote them** (row `sd_zero_rows_classified`); `sd_zero_rows_unclassified` is **`[]`**. The
categories, with exact counts:

| category | rows | why zero |
|---|---|---|
| `partA_selftest_*` | 3 | an identity: the vectorised score reproduces `RSRPolicy`'s victim on every trial |
| `hit_KEEPneeded_*` (incl. 2 snippet mirrors) | 6 | **ceiling** — 1.0 on every trial; keeping the needed member always leaves its detail in memory |
| `partB_hit_EVICTneeded_*` | 4 | **floor** — 0.0 on every trial; the surviving near-copy answers with the wrong detail |
| `partE_snippet_hit_EVICTneeded_*` | 2 | same floor, snippet mirror |
| `hit_DELTA_*`, `cos_sweep_hit_DELTA_*`, `M16_hit_DELTA` | 9 | difference of a saturated ceiling and a saturated floor |
| `partB_cos_sweep_achieved_cos_0.95` | 1 | pair cosine exact by construction, `1/(1+eps²)`. The other three cosine rows have tiny non-zero sd and are not in this table |
| `partC_within_pair_maxcos_penalty_abs_diff` | 1 | **bit-exact float32 tie** — this is the measurement itself |
| `partF_LEAK_highn_maxcos_tie` | 1 | tie rate exactly 1.0 in every seed, 80000/80000 |
| `LEAK_*_rule_maxcos_*` | 2 | **degenerate strict-`>` artifact on that bit-exact tie**, explicitly *not* a protective rule |
| `LEAK_leaky_rule_cosctx` | 1 | the **positive control saturating** — which is the point of it |
| `LOO_discriminates_pair_*`, `partE_snippet_LOO_disc_*`, `p_needed_is_argmax_LOO_*` | 6 | **ceiling** — LOO identifies the needed member on every trial |
| `ri_*` / `sign_agreement` / `argmax_ri` at `wq = 0.9` | 5 | **ceiling at `wq<1`** — the query carries the detail, so `r_i` lands on the needed member every time. Contrast the `wq = 1.0` rows, which are at chance with non-zero sd |

None is a seed that failed to reach the RNG. The five seeds do vary where they can: e.g.
`partC_p_argmin_EVICTS_needed_*` `sd = 0.01613`, `partF_LEAK_highn_cosctx` `sd = 0.00758`.

## 9 — What this does **not** establish

- **It is a constructed task, not a corpus.** It establishes that the **mechanism** of harm
  exists — that at `cos = 0.9` the victim choice can be total — and that nothing in the
  §3.4 rule resolves it. It says **nothing** about the **rate** on real text. §13.
- **The load-bearing unmeasured quantity is the fraction of real near-duplicate pairs whose
  residual carries a future-needed distinction.** My negative control shows the honest
  bracket: if the answer lives in the *shared* content the cost is **1e-5**; if it lives in
  the *residual* the cost is **1.0**. Real text is somewhere between and this cycle cannot
  say where. Everything about corpora here is **conjecture the experiment does not establish.**
- **The attention head is not trained.** `W_K` is tied to `W_Q` so the head retrieves by
  content at all; `g_mem` is supplied; `ψ̂` is a freshly initialised `BilinearValueHead` and
  `RSRPolicy.observe()` raises. A *trained* `ψ̂` might read the need out of `c_t` where
  discourse makes it predictable — §5's positive control shows the signal is exploitable in
  principle by a rule that looks for it. **This cycle cannot rule that out and does not.**
- **LOO here is not `rsr.metrics.loo.loo_delta_loss`**, which is a stub, and not Δ
  next-sentence loss, which needs a trained TG. It is leave-one-out in this task's unit.
- **The 2AFC is a discrimination metric.** It saturates at 1.0/0.0 by design, which is what
  makes it legible, and is also why it cannot express *graded* harm. The continuous recovery
  rows (`partB_recovery_*`, factor 980) carry the graded version.
- No claim about `ν`'s correct value: `ν` is MEASURED by E1 and unmeasured; every `ν` here
  is a labelled bypass and the result is a curve, not a setting.

## 10 — The next falsifier I would name

> **"`r_i`'s within-pair blindness is an artifact of an untrained attention head: in a
> trained TG, two gestalts at `cos ≥ 0.9` receive measurably *different* cross-attention
> when the query needs one of them, so §3.2.1's target does discriminate and the §6 result
> does not survive contact with a real model."**

Why it is the right one: §6 is the only result here that indicts a *component of the spec*
rather than the brief's premise, and it is the one most exposed to my having faked the
attention. It is also the cheapest thing in the queue that is **reachable without a corpus**
— it needs the PyTorch TG transcription (gauntlet 2.4) and a synthetic stream, not PG-19 and
not the exports. Concretely: run the same planted pair through a trained TG's real
cross-attention and read `α` on the two members at a topic-level query. If they separate,
§6 dies and `r_i` is exonerated; if they stay at 0.497/0.497, then §3.2.1's target is blind
to exactly the distinction D-3 says the policy must make, and **`ν` is the wrong response to
D-3 at the level of the target, not just at the level of the argmin.** Either way it
converts the single most consequential number in this cycle from synthetic to measured.
