# cycle-zscore-denominator — `_score`'s per-step z-scoring is the right denominator, and the fill-period bias is never exercised

- **Falsifier.** *"`_score`'s z-scoring puts `b` on the scale §3.5 item 1 claims it
  does — that is, `b_max = 1.0` really is one standard deviation of `ψ̂` as the
  policy actually sees it."*
- **Verdict: SURVIVED.** At the only `n_live` a decision ever occurs at, the
  per-step normalizer is **0.98038 ± 0.01012** of `ψ̂`'s true sd, against a
  normal-theory prediction of **0.98111**. §3.5 item 1's sentence is correct to
  ~2%.
- Ledger: `runs/cycle-zscore-denominator/ledger.json` — **229 rows, every number
  below is one of them.** 16 statistic rows have `sd == 0.0`, all 16 classified by
  key in `sd_zero_rows_reasons`; **`sd_zero_rows_unclassified` is `[]`**.
- Provenance stamped by the ledger itself: `git_sha bc8441a` (the brief's
  baseline), `git_dirty: true` — that is this untracked run directory only; `git
  status --porcelain` shows nothing but `?? runs/cycle-zscore-denominator/`.
  `src/`, `tests/`, `docs/` untouched, nothing committed. Python 3.12.13,
  macOS-26.6.2-arm64, CPU, 1 thread, **11.6 s wall**.
- Scratch harness (throwaway, per the rules):
  `/private/tmp/claude-501/-Users-keanooo7-second-brain/1bdd6660-ee4c-49c5-99a7-28bf139ef2c9/scratchpad/run_zscore_denominator.py`,
  reusing cycle 2's `scratch_bias.py` and cycle 3's driver/shared-head design.
- Geometry **read** from the registry: `M = 40`, `S = 80` (`pg19_e3`, FROZEN),
  `b_max = 1.0` (FROZEN). `d_model = 384`.
- **Deliberate visible bypasses**, labelled as such in the ledger and *not*
  measurements of those constants: `γ_b ∈ {0.001, 0.05, 0.1}` (DERIVED, E0e),
  `τ = 0.25`, EMA half-life `10.0`, hand-built `RSRConfig`, the `b(slots)`
  accessor carried from cycle 2, and part 4's `α`/`κ` perturbations.
- **`φ` is untrained and `ū` is synthetic.** `observe()` still raises
  (`src/rsr/retention/rsr.py:433`); ledger rows `phi_is_untrained` and
  `u_bar_is_synthetic`.

---

## 0 — The brief's premise about `n_live` is false, and that is the load-bearing finding

The brief says the per-step sd is *"a population sd over `n = n_live`, which is
small early in a stream."* **It is never small at a decision.**
`src/rsr/model/tg/policy_loop.py:145` is

```python
if bool(mem.valid[row].all()):
    victim = policy.select_eviction(slots, out.srep[row].detach(), t)
```

so the policy is consulted **only when the row's memory is full**. Measured, not
read off — through the **real** `run_policy_loop` with a real `TGModel`
(`M = 8`, batch 2, 24 sentences, 3 seeds):

| row | value |
|---|---|
| `part0_distinct_n_live_over_real_evictions` | **`[8]`** — one value, and it is `M` |
| `part0_min_n_live_at_a_real_eviction` | **8.0000 ± 0.0000** |
| `part0_frac_real_evictions_at_n_live_eq_M` | **1.0000 ± 0.0000** |
| `part0_n_evictions_per_seed` | 32.0000 ± 0.0000 |

Memory fills 0 → `M` monotonically and never un-fills, so `EvictionRecord.n_live`
is a constant. Consequences: the `unbiased=False` small-`n` bias is **real in the
estimator and never reached by a decision**; and `EvictionRecord.n_live` is a
third config-echo field alongside cycle 3's `attribution` — it cannot vary.

Brief items 2 and 3 are therefore measured below as properties of the estimator,
with that boundary stated, rather than as properties of the policy.

## 1 — Per-step normalizer vs pooled across-step sd: the ratio **is** ≠ 1, and the gap is entirely a per-step constant

5 seeds × 12 streams × `S = 80` steps; raw (pre-z) `ψ̂` at every step with
`n_live ≥ 2`; 480 full-memory steps/seed.

| row | value |
|---|---|
| `part1_perstep_sd_pop_mean_full_memory` | 1.380e-4 ± 0.02e-4 |
| `part1_pooled_raw_sd_across_steps_full_memory` | 1.480e-4 ± 0.02e-4 |
| **`part1_ratio_perstep_over_pooled_full_memory`** | **0.93260 ± 0.00665** |
| `part1_ratio_perstep_over_pooled_all_steps` | 0.90995 ± 0.00908 |
| `part1_frac_pooled_variance_from_between_step_means` | **0.11833 ± 0.01237** |
| `part1_variance_decomposition_relative_residual` | **0.000000 ± 0.000000** |
| `part1_perstep_sd_coefficient_of_variation_full` | 0.11698 ± 0.00479 |
| `part1_sigma_true_coefficient_of_variation_full` | 0.03574 ± 0.00043 |
| `part1_n_steps_with_sd_exactly_zero` | **0.0000 ± 0.0000** |

So the brief's item-1 prediction is confirmed as an arithmetic fact: the per-step
normalizer is **6.7% below** the pooled across-step sd, systematically, in every
seed. **But the inference the brief draws from it does not follow, and I am saying
so plainly.**

`1 − 0.93260² = 0.1303`, and `frac_pooled_variance_from_between_step_means` is
`0.1183`; the decomposition residual is **exactly zero to every reported digit**.
The entire gap is `Var` of the *per-step mean* of `ψ̂`. A per-step mean is a
constant added to **all** live slots at that step, and `_score` subtracts it one
operation earlier (`finite.mean()`, `rsr.py:371`). A common additive offset cannot
change an `argmin`. **Pooling across steps measures a quantity the argmin is blind
to, so a ratio ≠ 1 in that comparison is not evidence that `b_max = 1.0` is
mis-scaled.** Measured, not argued — §3 below swaps the denominator and counts
victims.

The right comparison is against `σ_true`, the sd of `ψ̂(s, c_t)` over the
gestalt-generating distribution at fixed `(φ, c_t)`. That is available in **closed
form** here: `ψ̂` is a *linear functional* of `s`
(`ψ̂ = s·w_eff + const`, `w_eff = (Wc)/d + u_s/2d`), and `E[s sᵀ] = I/d` for `s`
uniform on the unit sphere — which correction 15 says every gestalt is — so
`σ_true = ‖w_eff‖/√d` exactly. Checked against 200,000 fresh gestalts:
`selftest_sigma_true_closed_form_vs_monte_carlo_rel_dev = 0.00107 ± 0.00073`
against an expected MC noise floor of 0.00158.

| row | value |
|---|---|
| **`part1_ratio_perstep_sd_over_sigma_true_full_memory`** | **0.98038 ± 0.01012** |
| `part1_ratio_perstep_sd_over_sigma_true_all_steps` | 0.95837 ± 0.00858 |
| `part1_ratio_unbiased_sd_over_sigma_true_full_memory` | 0.99287 ± 0.01025 |
| `part1_theory_pop_ratio_at_n_eq_M` (closed form, not a measurement) | 0.98111 |

## 2 — The `n_live` sweep, 2 → `M`

96 contexts/seed × up to 30 **disjoint** slot groups per `n` (one `randperm` per
`(context, n)`, so no within-group duplicate can manufacture a tie), 5 seeds,
`n = 2…40`. `part2_within_step_excess_kurtosis_of_raw_psi = −0.01378 ± 0.00361`,
which is why normal theory is the right comparison.

| `n_live` | `sd_pop / σ_true` (measured) | `c4(n)·√((n−1)/n)` | `sd_pop / sd_unbiased` | median top-2 z-margin | `b_max` / that margin |
|---|---|---|---|---|---|
| 2 | **0.56406 ± 0.00552** | 0.56419 | 0.70711 | **2.00000 ± 0.00000** | 0.500 |
| 3 | 0.72646 ± 0.01288 | 0.72360 | 0.81650 | 1.22963 ± 0.00751 | 0.813 |
| 4 | 0.79916 ± 0.00774 | 0.79788 | 0.86603 | 0.85028 ± 0.01992 | 1.176 |
| 5 | 0.84292 ± 0.00392 | 0.84075 | 0.89443 | 0.69807 ± 0.00989 | 1.433 |
| 6 | 0.87089 ± 0.00495 | 0.86863 | 0.91287 | 0.61537 ± 0.00600 | 1.625 |
| 8 | 0.90369 ± 0.00117 | 0.90270 | 0.93541 | 0.51403 ± 0.01457 | 1.945 |
| 10 | 0.92381 ± 0.00368 | 0.92275 | 0.94868 | 0.46808 ± 0.00649 | 2.137 |
| 15 | 0.95019 ± 0.00307 | 0.94901 | 0.96609 | 0.39832 ± 0.00363 | 2.511 |
| 20 | 0.96187 ± 0.00109 | 0.96195 | 0.97468 | 0.36715 ± 0.00946 | 2.725 |
| 30 | 0.97361 ± 0.00265 | 0.97475 | 0.98319 | 0.33449 ± 0.01154 | 2.990 |
| **40 = `M`** | **0.98280 ± 0.00360** | **0.98111** | **0.98742** | **0.30794 ± 0.00427** | **3.248** |

(Full 39-value curves are in the ledger, one `stat` row per `n` for each of the
three quantities; `part2_theory_*_curve` notes hold the closed forms.)

**Item 2 answered.** The bias is large and exactly as the closed form says: at
`n_live = 2` the per-step sd is **56.4%** of `σ_true`, a **1.77× understatement**.
It is within 2% of `σ_true` by `n_live ≈ 20` and within 1.8% at `M`. Measurement
tracks `c4(n)·√((n−1)/n)` to within 0.3% at every `n` — this is ordinary
small-sample sd bias and nothing about `ψ̂` or `φ` complicates it.

**And it is inert.** Per §0 above, no eviction occurs at any `n_live < M`, so the
56% figure describes the estimator, not the policy.

**Item 3 answered, and the answer is that the question has no random part.**
`std(unbiased=False)/std(unbiased=True) = √((n−1)/n)` *identically*, so every
`part2_sd_pop_over_sd_unbiased_n*` row has `sd == 0.0` by algebra — itemized as
such in `sd_zero_rows_reasons`. At `n_live = 2` the correction is 29.3%; at the
`n_live` actually seen (`= M = 40`) it is **1.258%**. The brief asked whether it
matters at the `n_live` actually seen rather than asserting it from the formula —
§3 counts the decisions.

**One algebraic curiosity worth recording.** At `n_live = 2` the population sd is
`|x₁−x₂|/2`, so the two z-scores are **exactly ∓1** and the top-2 margin is
**exactly 2.0 = 2·b_max** at every draw (hence the zero sd, and
`frac_top2_z_margin_below_2bmax_n02 = 0.0000 ± 0.0000`). `b`'s maximum attainable
spread is `2·b_max`, so at `n_live = 2` `b` can **only** flip the argmin at exact
double saturation. The place the normalizer is most biased is the place `b` is
weakest. Both facts are decision-irrelevant for the same reason.

## 3 — Swap the denominator and count victims

Cycle 2/3's design exactly: a `b`-free driver arm owns the memory so the
`MemoryState` sequence is identical across `γ_b`; all arms share one
`BilinearValueHead` so `ψ̂` is bit-identical. 5 seeds × 12 streams × (80−40) =
**480 evictions/seed**.

`part3_selftest_hand_score_matches_policy_victim_gamma_b_* = 1.0000 ± 0.0000` in
all three arms — the hand-recomputed `argmin[(ψ̂−mean)/std(unbiased=False) + b]`
reproduces `RSRPolicy`'s own victim on all 2400 evictions, which is what licenses
the two counterfactuals.

| `γ_b` | victims changed by `unbiased=True` | victims changed by a **fixed global** normalizer |
|---|---|---|
| 0.001 | **0.0000 ± 0.0000** | 0.00417 ± 0.00295 |
| 0.05 | 0.00417 ± 0.00417 | **0.05792 ± 0.00925** |
| 0.1 | 0.00333 ± 0.00186 | 0.04292 ± 0.00700 |

- **`unbiased=False` vs `True` is decision-noise: ≤ 0.42% of victims.** A 1.258%
  uniform rescale of every z-gap almost never crosses one. The `γ_b = 0.001` zero
  is a **measured** zero, not a dead seed — the same arm's global-normalizer row
  has `sd > 0` on the same 2400 evictions.
- **The per-step-vs-global choice is *not* decision-neutral: 4.3–5.8%.** That is
  small but real, and it is the honest residue of §1. It is not the between-step
  *mean* (that cancels); it is that the per-step *sd* itself wobbles
  (`CV = 0.117`), of which `σ_true`'s own drift with `c_t` accounts for only
  `CV = 0.036` — the rest is sampling noise in a 40-sample sd. Per-step
  normalization makes `b`'s strength stationary **relative to the local spread**,
  which is what item 1 wants; a global normalizer would make it stationary in raw
  units. Neither is mis-scaled; they differ on ~5% of victims. **This is a design
  question for Brendan, not a defect, and I am not calling it either way.**

**Cross-cycle replication.** `part3_driver_top2_score_margin_median = 0.33388 ±
0.02135`. Cycle 2 measured **0.3326 ± 0.0207** on the same stream design, from an
independently written harness. The harness is the same harness.

## 4 — What `b_max = 1.0` is actually worth

**Two different numbers, and conflating them is how §3.5 item 1 reads as a
protection when it is not one.**

**(a) In `ψ̂`-sd units — item 1's own claim — it is right.**

| `n_live` | `b_max = 1.0` in true `ψ̂` sd |
|---|---|
| 2 | 0.56406 ± 0.00552 |
| 3 | 0.72646 ± 0.01288 |
| 5 | 0.84292 ± 0.00392 |
| 10 | 0.92381 ± 0.00368 |
| 20 | 0.96187 ± 0.00109 |
| **40 = the only value a decision sees** | **0.98280 ± 0.00360** |

(Rows `part2_bmax_in_true_psi_sd_at_n*`; the stream measurement
`part1_ratio_perstep_sd_over_sigma_true_full_memory = 0.98038 ± 0.01012` is the
same number from a different construction.) **`b_max = 1.0` = 0.98 true SD,
±0.004.** Item 1's sentence is accurate.

**(b) In decision units it is 3.25×, and *that* is what §3.5 item 1's defence
actually needs to be about.**

`part2_bmax_over_median_top2_z_margin_at_n40 = 3.24786 ± 0.04523`;
`part2_frac_top2_z_margin_below_bmax_n40 = 0.92278 ± 0.00453`. One SD is **three
and a quarter times** the median gap the `argmin` turns on, and **92.3%** of
decisions have a margin `b_max` alone could cross. Item 1's argument — *"z-scored,
which is what makes `b_max = 1.0` mean one standard deviation"* — is a true
statement that does no protective work, because one SD of 40 samples is enormous
next to the gap between the two *smallest* of them. That is order statistics, not
a scaling error.

## 5 — `φ` sensitivity: the calibration cannot be moved by `φ` at all

**Uniform weight-scale drift is *exactly* invariant.** `α ∈ {0.25, 1, 4}` on both
`W` and `u`: `part4_alpha_invariance_max_abs_deviation = 0.0` —
`part4_bmax_in_true_psi_sd_at_nM_alpha_{0.25,1,4}` are all `0.981437 ± 0.002932`
and the margins all `0.305851 ± 0.003810`, bit-identical. `ψ̂` is homogeneous of
degree 1 in `φ`, so the scale cancels in `(ψ̂ − mean)/sd`. (These rows' zero sd
*across* `α` is itemized; the across-**seed** sd inside each `α` is 0.0029.)

**Rank-1 low-rank drift moves nothing either.** `W += κ·abᵀ`, `κ ∈ {0, 20, 80}`
(κ ≈ √d ≈ 19.6 is parity with the base bilinear term): conversion
0.98144 → 0.97943 → 0.97691, margin 0.30585 → 0.31342 → 0.31161, excess kurtosis
−0.012 → −0.015 → −0.018. There is a reason it cannot move: **for any `φ` of this
head's form, `ψ̂` is a linear functional of `s`**, so for a spherical gestalt
distribution its across-slot law is near-Gaussian with `sd = ‖w_eff‖/√d` whatever
`W` and `u` are. **`φ` can change `ψ̂`'s scale and direction; it cannot change the
sd-to-margin relation.**

**The lever is the gestalt distribution.** 25% of each group of `M` slots replaced
by a near-copy of another member of the *same* group, 2000 groups/seed, 3 seeds:

| arm | mean max cosine in group | `b_max` in true sd | median top-2 z-margin | `b_max` / margin |
|---|---|---|---|---|
| iid (part 2) | 0.165 (cycle 3) | 0.98280 ± 0.00360 | 0.30794 ± 0.00427 | 3.248 ± 0.045 |
| `loose` | 0.18332 ± 0.00026 | 0.98069 ± 0.00213 | 0.31427 ± 0.00556 | 3.183 ± 0.056 |
| `tight` | **0.42423 ± 0.00098** | 0.97514 ± 0.00196 | **0.22696 ± 0.00495** | **4.408 ± 0.095** |

The sd conversion moves **0.8%**; the decision ratio moves **+36%**. ⚠️ **A
correction to cycle 3's construction while I was here:** cycle 3's literal
`gest[idx] + 0.15*randn(d)` has noise norm `0.15·√384 = 2.94`, which *dominates*
the unit-norm gestalt — pair cosine ≈ 0.32, not ≈ 1. The `loose` row reproduces
that and it barely perturbs anything; `tight` scales the noise per coordinate and
is the real manipulation. Cycle 3's 0.38 mean-max-cosine came from duplicates
*accumulating over a stream*, which my per-group construction does not model.

## 6 — The `sd = 0.0000` rows

16 rows, all classified by key in `sd_zero_rows_reasons` by a classifier in the
same process; `sd_zero_rows_unclassified` is `[]`. Categories:

- **Structural counts with no RNG** — `part0_n_evictions_per_seed` (32),
  `part1_n_steps_scored_per_seed` (936), `part1_n_steps_at_full_memory_per_seed`
  (480), `part3_n_evictions_per_seed_per_arm` (480), `part2_trials_per_seed_at_nM`.
- **Structural *and the finding*** — `part0_min_n_live_at_a_real_eviction` (8.0)
  and `part0_frac_real_evictions_at_n_live_eq_M` (1.0): `policy_loop.py:145` makes
  `n_live == M` at every decision, so no seed can move it.
- **Algebraic identities** — every `part2_sd_pop_over_sd_unbiased_n*` row
  (`√((n−1)/n)` is deterministic — which *is* the answer to item 3); and the four
  `n_live = 2` rows (`top2_z_margin_median_n02` = exactly 2.0,
  `frac_below_bmax_n02` = `frac_below_2bmax_n02` = 0, `bmax_over_margin_n02` =
  0.5), all from `sd_pop(2) = |x₁−x₂|/2`.
- **Exact invariance, and a finding** — the `part4_*_alpha_*` rows: degree-1
  homogeneity of `ψ̂` in `φ`.
- **Self-tests at their required value** —
  `part3_selftest_hand_score_matches_policy_victim_gamma_b_*` = 1.0, three rows.
- **Measured zeros** — `part1_n_steps_with_sd_exactly_zero` (`rsr.py:370`'s
  `if sd > 0` guard never fired in 4680 steps) and
  `part3_frac_victims_changed_by_unbiased_sd_gamma_b_0.001`.

**The seed does reach the RNG.** Every ratio, margin, victim-change fraction and
`σ_true` CV row has `sd > 0`; `part3_driver_top2_score_margin_median` replicates
cycle 2's independently written harness to 0.4%.

## 7 — Re-execution

The headline in the smallest standalone form — the per-step normalizer against
`ψ̂`'s true sd, at the worst `n_live` and at the only one a decision sees:

```bash
cd /Users/keanooo7/retrieval-successor-retention && .venv/bin/python -c "
import sys, math, torch; sys.path.insert(0,'src')
from rsr.retention.value_head import BilinearValueHead
d=384
for n in (2, 40):
    rs=[]
    for seed in range(3):
        g=torch.Generator().manual_seed(1000+seed)
        h=BilinearValueHead(d, generator=g)
        pool=torch.randn(8192,d,generator=g); pool=pool/pool.norm(dim=-1,keepdim=True)
        acc=[]
        with torch.no_grad():
            for _ in range(96):
                c=torch.randn(d,generator=g); c=c/c.norm()
                w=(h.W@c)*h.bilinear_multiplier + h.u[:d]*h.linear_multiplier
                st=float(w.norm()/math.sqrt(d))                  # sigma_true, exact
                psi=h(pool,c)
                k=min(30, 8192//n)
                v=psi[torch.randperm(8192,generator=g)[:k*n]].view(k,n).double()
                acc += (v.std(dim=1,unbiased=False)/st).tolist() # rsr.py:369 verbatim
        rs.append(sum(acc)/len(acc))
    m=sum(rs)/len(rs); sd=math.sqrt(sum((x-m)**2 for x in rs)/(len(rs)-1))
    c4=math.sqrt(2/(n-1))*math.exp(math.lgamma(n/2)-math.lgamma((n-1)/2))
    print(f'n_live={n:2d}  measured {m:.5f} +/- {sd:.5f}   theory {c4*math.sqrt((n-1)/n):.5f}')
"
```

Prints:

```
n_live= 2  measured 0.56298 +/- 0.01165   theory 0.56419
n_live=40  measured 0.98104 +/- 0.00232   theory 0.98111
```

(Independent RNG ordering from the harness, so it lands within sd of the ledger's
5-seed 0.56406 ± 0.00552 and 0.98280 ± 0.00360 rather than reproducing them bit
for bit.)

Full experiment, 11.6 s, rewrites this directory's `ledger.json`:

```bash
cd /Users/keanooo7/retrieval-successor-retention && .venv/bin/python \
  /private/tmp/claude-501/-Users-keanooo7-second-brain/1bdd6660-ee4c-49c5-99a7-28bf139ef2c9/scratchpad/run_zscore_denominator.py
```

## 8 — What this does NOT establish

- **`φ` is untrained, and that is a real limit.** `observe()` raises;
  `run_policy_loop` never calls it. Two things a trained `φ` could change, and one
  it provably cannot:
  - **Cannot:** the sd-to-margin relation, *as long as the head keeps this form*.
    `ψ̂` is linear in `s`, so for spherical gestalts its across-slot law is
    near-Gaussian with `sd = ‖w_eff‖/√d` for **any** `W, u`. Both perturbation
    arms confirm it numerically.
  - **Could:** training does not act on `φ` alone. Under a trained `φ` the
    *gestalts* are also different — §5 shows the gestalt distribution is the lever,
    and a `ψ̂` that has learned to separate one or two slots sharply from 38 others
    is exactly the non-Gaussian across-slot law this measurement cannot produce.
    A bimodal `ψ̂` would raise the top-2 gap and **weaken** `b`; a `ψ̂` with a
    near-tied lower tail would **strengthen** it. `b_max` in *sd* units would stay
    ≈0.98; `b_max` in *margin* units could move in either direction.
  - **Could:** `E[lifetime]`, hence correction 4's `γ_b` formula, under a
    non-FIFO realized lifetime distribution. Not touched here.
  - Per `src/rsr/mup/coord_check.py`'s own note, a synthetically "correlated" `φ`
    does not substitute for real optimizer steps, and part 4 does not pretend to.
    **Nothing was trained.**
- **Nothing about `γ_b`, `τ`, `b_max`, `ν` or `E[lifetime]` as constants.** E0e's
  and E1's. Every value used is a labelled bypass.
- **`ū` is synthetic.** Part 3's victim-change fractions are conditional on a
  log-normal-salience softmax and `τ = 0.25`, as in cycles 2–3. Parts 0, 1, 2 and
  4 do not involve `ū` at all and are unconditional on it.
- **One gestalt distribution family.** iid spherical, plus the two duplicate arms.
  Real PG-19 gestalts are neither iid nor isotropic, and §5 shows that is the
  axis that matters. ADR-0005 / E7 stimuli would change this.
- **`n_live < M` is unreachable *in the current loop*.** If a future change lets
  slots die mid-stream, or queries the policy on an underfull memory, the §2 curve
  becomes live and `b_max` drops toward 0.56 true SD.
- **Nothing about whether `b` helps.** No language model ran, no loss was
  computed, no `r_i` was collected. "Changed the victim" is not "made it worse".
- **`ν = 0` and `t_warm = 0` throughout**; `psi_override="neg_age"` never run.
  §3.7's reduction path is untouched.
- **CPU only**, one thread. No MPS, no device-placement claim.
- **Cycle-1 walls still stand**: `measurements_ledger_exists: false`, re-checked.
