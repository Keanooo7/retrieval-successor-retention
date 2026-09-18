# cycle-bmax-scope-dependence — a single FROZEN `b_max` is **not** comparable across the `M` values this project trains

- **Falsifier.** *"A single FROZEN `b_max` gives `b` comparable strength at every `M`
  the project trains."*
- **Verdict: FALSIFIED**, on both the order statistic and the realized decision count.
  - The dimensionless ratio `b_max / median-top-2-z-margin` is **1.88772 ± 0.05003**
    at `M = 8`, **2.59245 ± 0.05256** at `M = 16`, **3.17490 ± 0.08201** at
    `M = 40` — a swing of **1.68244 ± 0.04855** across exactly the two `M` the
    project trains. The brief predicted ~1.67 from cycle 4's two points; measured
    independently here on a denser sweep, it is 1.682.
  - Realized victim-flip rate, `γ_b` set by **correction 4's own formula at each
    `M`** (`E[lifetime] = M`, i.e. `γ_b = 4/M` = 0.5 / 0.25 / 0.1): **0.46713 ±
    0.01702** → **0.53359 ± 0.02150** → **0.59500 ± 0.01289**. A **12.8 ± 2.4
    percentage-point** spread, `1.27535 ± 0.06147`×.
  - With `γ_b` **held** at the `M = 40` value 0.1: **0.32269 ± 0.01287** →
    **0.48542 ± 0.01047** → **0.59500 ± 0.01289**. `1.84636 ± 0.08705`×.
- Ledger: `runs/cycle-bmax-scope-dependence/ledger.json` — **186 rows, every number
  below is one of them.** 21 statistic rows have `sd == 0.0`, all 21 classified by
  key in `sd_zero_rows_reasons`; **`sd_zero_rows_unclassified` is `[]`**.
- Provenance stamped by the ledger itself: `git_sha ecbc7b6` (the brief's baseline),
  **`git_dirty: false`**. `src/`, `tests/`, `docs/` untouched; nothing committed.
  Python 3.12.13, macOS-26.6.2-arm64, CPU, 1 thread, **7.6 s wall**.
- Scratch harness (throwaway, per the rules):
  `/private/tmp/claude-501/-Users-keanooo7-second-brain/1bdd6660-ee4c-49c5-99a7-28bf139ef2c9/scratchpad/run_bmax_scope.py`,
  reusing cycle 2/3/4's driver + shared-head counterfactual design and
  `scratch_bias.CallSiteBias`.
- **Geometry read from the registry**, never supplied: `b_max = 1.0` (FROZEN),
  `M = 8` (`e7`), `M = 16` (`synthetic`), `M = 40` (`corpora`, `pg19_e3`),
  `S = 48` (`synthetic`), `S = 80` (`pg19_e3`). `d_model = 384`.
- **Deliberate visible bypasses**, labelled as such in the ledger and *not*
  measurements of those constants: every `γ_b` (DERIVED, E0e), `τ = 0.25`, the EMA
  half-lives, the hand-built `RSRConfig`, the `b(slots)` accessor carried from cycle
  2, and **`S = 80` held fixed across all three `M`** in part 3 (part 4 is the
  control for that choice).
- **`φ` is untrained and `ū` is synthetic.** `observe()` still raises
  (`src/rsr/retention/rsr.py:433`); rows `phi_is_untrained`, `u_bar_is_synthetic`.

---

## 0 — Two corrections to the brief, one of them load-bearing

**1. `M = 16` does have a registry scope.** The brief says *"`M = 16` may not have a
registry scope — if so, say so rather than inventing one."* It does:
`rsr.constants.get("M", "synthetic")` returns **16**, FROZEN, spec §5.1 (*"16
synthetic (maximum eviction pressure, exact gap control)"*). All three `M` in this
experiment are registry reads; none is invented. Row
`brief_claim_M16_may_have_no_registry_scope__CORRECTED`.

What *is* missing is `S` for the `M = 8` scope: `get("S", "e7")` **raises
`ScopeRequired`** — §5.2 sets E7's stream length from the data (*"evaluation only,
at passage length, uncut"*). So the small-`M` model's stream length has to be
supplied, and it is, as a labelled bypass. Rows `S_e7_registry_read`,
`S_corpora_registry_read` (also raises).

**2. Correction 4's formula and correction 4's stated range disagree away from
`M = 40`.** `γ_b ≈ b_max/(0.25·E[lifetime])` at `E[lifetime] = M` gives **0.1 at
`M = 40`** — the top of the stated *"order 0.05–0.1"* — but **0.25 at `M = 16`** and
**0.5 at `M = 8`**, i.e. **5× the top of its own stated range** at the `M` that D-5 /
correction 6 funds. The range and the formula were both written for `M = 40`. Row
`part3_gamma_b_formula_M`.

## 1 — §13 statement, made before any number

The registry attaches `M = 8` to `e7`, `M = 16` to `synthetic` and `M = 40` to
`corpora`/`pg19_e3`. **This experiment does not compare those corpora.** Part 1
compares a **dimensionless ratio as a function of the integer `n`** — a pure
order-statistic property of `n` numbers, carrying no corpus units at all. Parts 3–4
compare flip rates **within one synthetic generator** (iid unit-norm spherical
gestalts per correction 15, synthetic `ū`) evaluated at three *capacities*. No
absolute number from one corpus is set against an absolute number from another. Row
`section_13_scope_statement`.

## 2 — The order statistic, measured twice

5 seeds. **(a) `ψ̂` construction:** 96 contexts × up to 30 **disjoint** slot groups
per `n`, drawn from one `randperm` of a pool of 8192 unit gestalts, z-scored with
`std(unbiased=False)` — `rsr.py:369` verbatim — and the gap taken between the two
*smallest* z-scores, the same definition as `EvictionRecord.score_margin`
(`rsr.py:406`). **(b) Normal-theory yardstick:** 40,000 **exact iid N(0,1)** draws
per `n`, no head, no gestalt, no `ψ̂`. Cycle 4's measured within-step excess
kurtosis of `−0.01378 ± 0.00361` is what licenses (b) as the yardstick.

| `n` | median top-2 z-margin | **`b_max` / that margin** (`ψ̂`) | **normal theory** | frac. margin < `b_max` |
|---|---|---|---|---|
| 4 | 0.84828 ± 0.01896 | 1.17931 ± 0.02573 | 1.16226 ± 0.00308 | 0.57458 ± 0.00908 |
| 5 | 0.68625 ± 0.01697 | 1.45790 ± 0.03595 | 1.43310 ± 0.00806 | 0.66701 ± 0.01397 |
| 6 | 0.60499 ± 0.02388 | 1.65498 ± 0.06514 | 1.63007 ± 0.00773 | 0.71951 ± 0.01270 |
| **8 = `M` (e7)** | **0.53004 ± 0.01410** | **1.88772 ± 0.05003** | **1.92608 ± 0.00754** | **0.77938 ± 0.01402** |
| 10 | 0.46236 ± 0.01411 | 2.16438 ± 0.06513 | 2.13821 ± 0.01333 | 0.82021 ± 0.00618 |
| 12 | 0.43196 ± 0.00951 | 2.31592 ± 0.05120 | 2.30285 ± 0.01077 | 0.83938 ± 0.00532 |
| **16 = `M` (synthetic)** | **0.38586 ± 0.00780** | **2.59245 ± 0.05256** | **2.53593 ± 0.01184** | **0.86979 ± 0.00503** |
| 20 | 0.37217 ± 0.00754 | 2.68782 ± 0.05583 | 2.71258 ± 0.02043 | 0.88486 ± 0.00510 |
| 24 | 0.34879 ± 0.01039 | 2.86900 ± 0.08272 | 2.84727 ± 0.01791 | 0.89313 ± 0.00546 |
| 32 | 0.32509 ± 0.01095 | 3.07892 ± 0.10431 | 3.04549 ± 0.01542 | 0.90819 ± 0.00655 |
| **40 = `M` (corpora)** | **0.31514 ± 0.00825** | **3.17490 ± 0.08201** | **3.19050 ± 0.02180** | **0.91979 ± 0.00485** |
| 48 | 0.30120 ± 0.00379 | 3.32043 ± 0.04234 | 3.29949 ± 0.01746 | 0.92507 ± 0.00312 |
| 64 | 0.28364 ± 0.00901 | 3.52836 ± 0.10829 | 3.49506 ± 0.01475 | 0.93340 ± 0.00502 |

**Swings.** `part1_ratio_swing_M40_over_M8 = 1.68244 ± 0.04855`;
`M16/M8 = 1.37415 ± 0.04815`; `M40/M16 = 1.22517 ± 0.04325`. On the exact-normal
yardstick the same swing is `1.65652 ± 0.01608`.

**Normal theory is an excellent yardstick and it is the mechanism, not a
coincidence.** `part1_psi_minus_normal_ratio_abs_dev_n*` is ≤ 0.0994 at every `n` —
around 2–3% of the ratio — and the two curves bracket each other rather than one
sitting above the other. The consequence matters for interpretation: **this swing is
a property of the integer `n`, not of RSR.** It is how far apart the two smallest of
`n` z-scores are. Nothing about `φ`, `ψ̂`, TG or the corpus enters it.

**Cross-cycle replication.** Cycle 4's independently written sweep, with a different
`n` list and therefore different `randperm` consumption, gave `1.945` at `n = 8` and
`3.24786 ± 0.04523` at `n = 40`, and `frac_below_bmax_n40 = 0.92278 ± 0.00453`.
Here: `1.88772 ± 0.05003`, `3.17490 ± 0.08201`, `0.91979 ± 0.00485`. Agreement
within ~1 sd at `n = 40` and ~1.1 sd at `n = 8`. Row
`part1_cycle4_replication_targets`.

**The ceiling, for completeness.** `2·b_max` is `b`'s maximum attainable spread:
`part1_frac_top2_z_margin_below_2bmax` is `0.98660 ± 0.00236` (8), `0.99410 ±
0.00141` (16), `0.99840 ± 0.00072` (40). At every `M` the ceiling is essentially
100%; it is the *unilateral* reach column that separates them.

## 3 — `E[lifetime]` is policy-dependent, and the gap itself grows with `M`

12 streams × `S = 80` per seed, 5 seeds. The FIFO column is a **real `FIFOPolicy`
driving its own memory** over the same context stream — not a statistic read out of
the learned driver's memory. (An earlier version of this harness did the latter and
mislabelled it "FIFO"; that row is kept under its true name,
`part2_oldest_live_slot_age_at_decision_M*`, because it is real and interesting — it
is just not FIFO's lifetime. Recorded because §12.4 means recording the wrong one
too.)

| `M` | realized mean lifetime (learned head) | FIFO | realized / FIFO | oldest live slot at a decision | longest single lifetime |
|---|---|---|---|---|---|
| 8 | **7.37708 ± 0.07148** | **8.00000 ± 0.00000** | 0.92214 ± 0.00893 | 24.28125 ± 1.97850 | 59.6 ± 3.4 |
| 16 | **13.31693 ± 0.32914** | **16.00000 ± 0.00000** | 0.83231 ± 0.02057 | 41.83906 ± 1.61322 | 71.4 ± 6.2 |
| 40 | **26.23583 ± 0.47394** | **40.00000 ± 0.00000** | 0.65590 ± 0.01185 | 58.75625 ± 0.21375 | 76.0 ± 2.0 |

The FIFO rows are **exactly `M`**, with `sd = 0.0` by construction (one write per
step into a full memory ⇒ the oldest slot's age is `M`), which is the check that
`E[lifetime] = M` *is* the FIFO parameterization of correction 4's formula. The
`M = 40` realized figure **replicates the brief's own 26.49 ± 0.53** at 26.236 ±
0.474.

**A third `M`-dependence, not in the brief.** `realized / FIFO` is not a constant:
**0.922 → 0.832 → 0.656**. The two parameterizations of correction 4's formula
therefore diverge *more* at larger `M`, so "which `E[lifetime]`" is itself a
scope-dependent question, not a global 0.66 correction factor.

## 4 — Realized victim-flip rates, three `γ_b` parameterizations

Cycle 2/3/4's design exactly: a `b`-free driver arm (`b_enabled=False`, `ν=0`) owns
the memory so the `MemoryState` sequence is identical, all arms share one
`BilinearValueHead` so `ψ̂` is bit-identical, `S = 80` for all three `M`, 12
streams × 5 seeds. A **flip** = the `b`-enabled arm's victim differs from the
`b`-free driver's on the same state. EMA half-life is `E[lifetime]/4` with the
*same* `E[lifetime]` that set that arm's `γ_b`, per correction 4's own pairing.

`part3_selftest_hand_score_matches_policy_victim_M*_* = 1.00000 ± 0.00000` in **all
nine arms** — the hand-recomputed `argmin[(ψ̂−mean)/std(unbiased=False) + b]`
reproduces `RSRPolicy`'s own victim on all 10,560 evictions per arm (31,680 arm-evictions in total). That is what
licenses the counterfactual.

| `M` | `γ_b` | arm | **frac. victims `b` changed** | frac. `b` saturated | mean `|b|` |
|---|---|---|---|---|---|
| 8 | 0.5 | `formula_M` (`E[lt]=M`) | **0.46713 ± 0.01702** | 0.64661 ± 0.01741 | 0.71884 ± 0.01549 |
| 16 | 0.25 | `formula_M` | **0.53359 ± 0.02150** | 0.59113 ± 0.01304 | 0.67523 ± 0.01119 |
| 40 | 0.1 | `formula_M` | **0.59500 ± 0.01289** | 0.54339 ± 0.01010 | 0.64103 ± 0.01065 |
| 8 | 0.54222 | `formula_realized` | 0.47269 ± 0.01905 | 0.65773 ± 0.01979 | 0.73944 ± 0.01607 |
| 16 | 0.30037 | `formula_realized` | 0.54766 ± 0.01736 | 0.61279 ± 0.01386 | 0.72008 ± 0.01227 |
| 40 | 0.15246 | `formula_realized` | 0.61000 ± 0.00925 | 0.65118 ± 0.00900 | 0.73735 ± 0.00800 |
| 8 | 0.1 | `fixed_0.1` | **0.32269 ± 0.01287** | 0.19919 ± 0.02198 | 0.42486 ± 0.02068 |
| 16 | 0.1 | `fixed_0.1` | **0.48542 ± 0.01047** | 0.37822 ± 0.01492 | 0.54790 ± 0.00987 |
| 40 | 0.1 | `fixed_0.1` | **0.59500 ± 0.01289** | 0.54339 ± 0.01010 | 0.64103 ± 0.01065 |

Swings, per seed then averaged:

| arm | `M=40 / M=8` | `M=40 − M=8` (pp/100) |
|---|---|---|
| `formula_M` | **1.27535 ± 0.06147** | **0.12787 ± 0.02406** |
| `formula_realized` | 1.29205 ± 0.05177 | 0.13731 ± 0.01894 |
| `fixed_0.1` | **1.84636 ± 0.08705** | **0.27231 ± 0.01929** |

**Which parameterization is used where.** `formula_M` is correction 4's formula fed
`E[lifetime] = M`, the **FIFO** value the registry's `E_lifetime` is defined as
(*"Expected slot lifetime under FIFO"*). `formula_realized` is the same formula fed
part 3's **measured realized** lifetime under the learned-head driver. **Both are
reported; neither is a measurement of `E_lifetime` or `γ_b`.** They differ by 0.5–1.5
pp in flip rate and by up to 1.5× in `γ_b` — the flip rate is far less sensitive to
`γ_b` than `γ_b` is to the lifetime choice, because `b` is saturated most of the
time either way.

**Three readings, in order of how much they carry.**

1. **The rates are not comparable under the formula.** 46.7% vs 59.5% is a 12.8 pp
   gap with seed sd ≈ 2 pp — about 5 sd. `b`'s realized authority over the argmin
   differs materially between the PG-19 model at `M = 40` and the E7 small-`M` model
   at `M = 8` that D-5 / correction 6 funds. **The falsifier dies here.**
2. **But correction 4's `1/M` scaling is doing real work, and it over-corrects.** Hold
   `γ_b` at 0.1 and the swing nearly doubles, to `1.84636 ± 0.08705`. So the formula's
   `M`-dependence pulls the small-`M` rate **up** (32.3% → 46.7%) and closes roughly
   half the gap the order statistic opens. It is a partial compensation for a
   mechanism it was not derived to compensate for: correction 4's argument is about a
   *timescale* (reach `O(b_max)` within a lifetime), and it lands on `1/M`, while the
   order statistic wants roughly the inverse of the ratio column — about `1/1.68`
   between `M = 8` and `M = 40`, not `1/5`.
3. **The direction is the opposite of what a saturation reading predicts.** At
   `M = 8, γ_b = 0.5`, `b` saturates *more* (64.7% of slot-decisions at `±b_max` vs
   54.3% at `M = 40`) and `mean |b|` is *higher* (0.719 vs 0.641), yet `b` flips
   *fewer* victims. That is the order statistic beating the magnitude: a bigger `b`
   against a wider gap still moves less. Any future defence of a frozen `b_max` that
   argues from `|b|` reaching `b_max` is arguing about the wrong quantity.

## 5 — `S` is not driving it (part 4)

`M = 16`'s registry scope is `synthetic`, whose own `S` is **48**, not the 80 part 3
holds fixed. Re-run at `S = 48`, everything else identical:

| arm | flip rate at `S = 48` | `S=48 − S=80` |
|---|---|---|
| `formula_M` | 0.52917 ± 0.03794 | −0.00443 ± 0.04643 |
| `formula_realized` | 0.54740 ± 0.03680 | −0.00026 ± 0.03812 |
| `fixed_0.1` | 0.47344 ± 0.04111 | −0.01198 ± 0.03742 |

All three differences are within one sd of zero, on half as many evictions per seed
(384 vs 768). The part-3 fixed-`S` choice is not producing the result.

**Cross-cycle replication, and a bit-exact one.**
`part3_driver_top2_score_margin_median_M40 = 0.33388 ± 0.02135` — **identical to
cycle 4's `part3_driver_top2_score_margin_median`** to all five digits, same seeds
and same stream design, and cycle 2 measured `0.3326 ± 0.0207` from an independently
written harness. `part3_frac_victims_flipped_by_b_M40_formula_M = 0.59500` sits at
the top of the brief's replicated 56–60% band.

Note also that the *live-memory* margin is systematically **larger** than the
iid-draw margin at the same capacity — 0.53070 vs 0.53004 at `M = 8`, 0.39253 vs
0.38586 at 16, 0.33388 vs 0.31514 at 40 — so the iid part-1 ratio is, if anything, a
mild over-estimate of `b`'s reach in a live memory. It does not change the ordering.

## 6 — The `sd = 0.0000` rows

21 rows, all classified by key in `sd_zero_rows_reasons` by a classifier in the same
process; `sd_zero_rows_unclassified` is `[]`. Categories:

- **Structural counts with no RNG** — `part1_psi_trials_per_seed_at_n40`,
  `part1_normal_trials_per_seed_at_n40`, `part2_n_evictions_per_seed_M*`,
  `part3_n_evictions_per_seed_per_arm_M*`, `part4_n_evictions_per_seed_per_arm_M16_S48`.
- **Algebraic, and it is the point** — `part2_fifo_mean_slot_lifetime_M{08,16,40}` =
  exactly 8 / 16 / 40. A real `FIFOPolicy` with one write per step into a full memory
  evicts a slot of age exactly `M` every time, which is the check that
  `E[lifetime] = M` is the FIFO parameterization.
- **Self-tests at their required value 1.0** —
  `part3_selftest_hand_score_matches_policy_victim_M*_*`, nine rows. A value below
  1.0 would invalidate the flip counts.

**The seed does reach the RNG.** Every ratio, margin, lifetime, flip-rate,
saturation and swing row has `sd > 0`; the `M = 40` driver margin replicates cycle 4
to five digits and cycle 2's independent harness to 0.4%.

## 7 — What this implies for the registry (options and costs; **not a pick**)

`b_max` is one FROZEN value (`src/rsr/constants.py`, `Definition(name="b_max", …,
value=1.0)`), with no `scoped=` dict, while `M` and `S` have one and §13 forbids a
global reading of either. Three honest options:

**(A) `b_max` becomes per-scope, like `M` and `S`.**
*Cost:* mechanically cheapest — `value=1.0` becomes `scoped={…}` and every call site
must pass a scope, exactly as `get("M")` already refuses a scope-free read. But
`b_max`'s own spec note (§4.5) is *"interpretable as one standard deviation because
`ψ̂` is z-scored across live slots"*, and cycle 4 measured that reading to be
**correct** (`0.98280 ± 0.00360` of true sd at `M = 40`). A per-scope `b_max` gives
up that one-line interpretation: it would have to be justified as "whatever equalizes
decision authority," i.e. a *measured* or *derived* constant rather than a frozen
one, which moves it out of FROZEN and into something E0e (or a new experiment) owns.
It also creates a second frozen-per-scope knob that `A5`'s `γ_b` sweep now has to be
crossed with.
*What would decide it:* a target for what "comparable strength" means. Equalizing the
ratio column would put `b_max ≈ 0.59` at `M = 8` against 1.0 at `M = 40`; equalizing
the **flip rate** is a different number again, and this cycle can supply both once
someone says which is the invariant.

**(B) `γ_b`'s formula absorbs the `M` dependence.**
*Cost:* the smallest diff — `_derive_gamma_b` gains a factor — but it changes a
formula §3.5 item 4 states and correction 4 pins, and `γ_b` is already DERIVED from
`(b_max, E_lifetime)`, so adding an `M` dependence adds a third dependency and makes
`γ_b` scoped too (`E_lifetime` is already effectively per-scope). Measured here, this
option is **already half-realized by accident**: `γ_b = 4/M` closes ~half the gap
(1.85× → 1.28×), so the fix is a re-tuning of an existing exponent, not a new
mechanism. The risk is that it conflates two different jobs in one constant — a
*timescale* (how fast `b` reaches `O(b_max)`) and a *decision authority* (how much of
the argmin `b` owns) — and correction 4's derivation is explicitly about the first.
*What would decide it:* whether the residual 1.28× is acceptable, and whether anyone
is willing to have `γ_b` no longer be readable as "reach `b_max` in a quarter of a
lifetime."

**(C) The spec accepts that `b` is stronger at large `M` and says so.**
*Cost:* zero code, and it is the only option that costs no constant. But it must be
written into **§13 as a limitation** and into the E7 writeup, because falsifier 4 /
release condition 4 compares the `M = 8` E7 model's behaviour against humans while
the `M = 40` model carries the primary claim — and this cycle shows the anti-collapse
loop has ~13 pp less authority over the argmin in the E7 model. If that is accepted,
it must be *stated*, not discovered when the two models disagree.
*What would decide it:* whether `b`'s job is load-bearing for the E7 claim at all. If
`b` is only anti-collapse hygiene and E7's correlation does not depend on it, (C) is
free. If E7's retained-set depends on `b` having comparable authority, (C) is not
available.

**What none of the three changes.** The mechanism is the order statistic, which is a
property of `n` and is untouched by any registry edit: the margin between the two
smallest of `n` z-scores shrinks with `n` while `b_max` does not. Only (A) and (B)
can compensate for it; (C) declares it out of scope.

## 8 — Re-execution

The headline in the smallest standalone form — the order statistic, on exact
normals, at the three `M` values read from the registry:

```bash
cd /Users/keanooo7/retrieval-successor-retention && .venv/bin/python -c "
import sys, math, torch; sys.path.insert(0,'src')
from rsr import constants as C
b_max = C.get('b_max')                                  # FROZEN registry read
Ms = [C.get('M','e7'), C.get('M','synthetic'), C.get('M','pg19_e3')]
for n in Ms:
    rs=[]
    for seed in range(3):
        g=torch.Generator().manual_seed(7000+seed)
        v=torch.randn(40000,n,generator=g,dtype=torch.float64)
        z=(v-v.mean(1,keepdim=True))/v.std(1,unbiased=False).unsqueeze(1)  # rsr.py:369
        zs=z.sort(1).values
        m=(zs[:,1]-zs[:,0]).median().item()
        rs.append(b_max/m)
    mu=sum(rs)/3; sd=math.sqrt(sum((x-mu)**2 for x in rs)/2)
    print(f'M={n:2d}  b_max/median_top2_z_margin = {mu:.4f} +/- {sd:.4f}')
"
```

Prints:

```
M= 8  b_max/median_top2_z_margin = 1.9329 +/- 0.0075
M=16  b_max/median_top2_z_margin = 2.5404 +/- 0.0127
M=40  b_max/median_top2_z_margin = 3.1824 +/- 0.0279
```

i.e. a **1.65× swing** across the two `M` the project trains, from a single
`torch.randn` — no head, no policy, no gestalt. (3 seeds, and it drops the
median-over-many-groups structure, so it lands within sd of the ledger's 5-seed
normal rows `1.92608 ± 0.00754` / `2.53593 ± 0.01184` / `3.19050 ± 0.02180` rather
than reproducing them bit for bit.)

Full experiment, 7.6 s, rewrites this directory's `ledger.json`:

```bash
cd /Users/keanooo7/retrieval-successor-retention && .venv/bin/python \
  /private/tmp/claude-501/-Users-keanooo7-second-brain/1bdd6660-ee4c-49c5-99a7-28bf139ef2c9/scratchpad/run_bmax_scope.py
```

## 9 — What this does NOT establish

- **`φ` is untrained.** `observe()` raises; `run_policy_loop` never calls it. Cycle 4
  established that the sd-to-margin relation is invariant to `φ`'s **weight scale**
  (exactly, by degree-1 homogeneity) and near-invariant to rank-1 drift, *as long as
  the head keeps this form* — `ψ̂` is a linear functional of `s`, so for spherical
  gestalts its across-slot law is near-Gaussian whatever `W, u` are. What a trained
  `φ` changes is the **gestalt distribution**, and cycle 4 measured that as the lever
  (a duplicate-heavy memory moved the `M = 40` ratio +36%). A trained `ψ̂` with a
  bimodal across-slot law would widen the top-2 gap and **weaken** `b`; one with a
  near-tied lower tail would **strengthen** it. **Whether the `M`-dependence survives
  training is not established here** — only that it is present, is ~1.68× wide, and
  is a property of `n` under the near-Gaussian law that is currently measured to hold
  (excess kurtosis `−0.014 ± 0.004`, cycle 4).
- **`ū` is synthetic.** Every part-3/4 flip rate is conditional on a log-normal
  salience softmax and `τ = 0.25`. **Part 1 and part 2 do not involve `ū` at all.** A
  real utilisation signal could change the flip rates in either direction; it cannot
  change the margin distribution, which is where the `M`-dependence lives.
- **Nothing about `γ_b`, `τ`, `b_max`, `E_lifetime` or `ν` as constants.** E0e's and
  E1's. Every `γ_b` used is a labelled evaluation of correction 4's formula at a
  hand-supplied `E[lifetime]`, not a measurement of `γ_b`.
- **"Changed the victim" is not "made it worse."** No language model ran, no loss was
  computed, no `r_i` was collected, no LOO Δloss. This cycle counts decisions, not
  outcomes. Whether the `M = 8` model's weaker `b` is a *problem* requires E0e/E3.
- **One gestalt distribution family.** iid spherical. Real PG-19 gestalts are neither
  iid nor isotropic, and cycle 4 showed that is the axis that matters.
- **`ν = 0`, `t_warm = 0`, `β = 0`, `γ = 0` throughout**, `psi_override=None`
  everywhere. §3.7's reduction path is untouched. The redundancy term `−ν·max cos`
  is off, and it is the one other term in `_score` that could interact with `M`.
- **`S` for the `M = 8` scope was supplied, not read.** `get("S","e7")` raises. Part 4
  controls `S` at `M = 16` only; there is no `S`-control at `M = 8` because the
  registry has no value to control against.
- **CPU only**, 1 thread. No MPS, no device claim.
- **Cycle-1 wall still stands**: `measurements_ledger_exists: false`, re-checked.
- **The known defects are unchanged and untouched**: `ProtectionBias`'s declared
  surface still disagrees with `rsr.py:377`'s call site; `attribution` and `n_live`
  are still config-echo fields; `run_policy_loop` still never trains `ψ̂`.
