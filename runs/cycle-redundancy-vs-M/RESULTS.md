# cycle-redundancy-vs-M — `b`'s decision strength is **not** a function of `M` alone, and "redundancy" is not one axis either

- **Falsifier.** *"`b`'s realized decision strength is a function of `M` alone — so any
  `M`-indexed fix (per-scope `b_max`, or an `M` factor in `γ_b`) is a complete fix."*
- **Verdict: FALSIFIED**, on three counts, one of which is not in the brief:
  1. At **fixed `M` = 40**, gestalt redundancy moves `b_max / median-top-2-z-margin` by
     **2.37102 ± 0.21026** (live memory, achieved mean-max-cosine
     **0.73083 ± 0.00980**) and **4.03253 ± 0.11747** (fixed-membership groups, achieved
     mean-max-cosine 0.98040 ± 0.00038), against a whole-`M`-range effect of
     **1.66915 ± 0.12036** (order statistic) / **1.65205 ± 0.17053** (live memory). The
     crossing — the achieved mean-max-cosine at which redundancy at fixed `M` matches the
     **entire** `M = 8 → M = 40` effect — is **0.54132 ± 0.03193** (order statistic) and
     **0.54260 ± 0.04367** (live memory): two constructions locating it in the same place.
  2. The `M = 8 → M = 40` **gap itself moves with redundancy**. The realized flip-rate gap
     an `M`-keyed `b_max` would be tuned to close is **1.19266 ± 0.05566** at the
     near-duplicate control and **1.49851 ± 0.11214** at the highest reachable redundancy.
     One `M`-keyed constant cannot close a gap that is a function of the other variable.
  3. **The sign flips with `M`.** Redundancy *weakens* `b`'s realized authority at `M = 8`
     (flip-rate effect **0.88401 ± 0.05446**) and *strengthens* it at `M = 40`
     (**1.10803 ± 0.03838**) — straddling 1.0, about 3 sd apart.
- **The brief's framing contains two errors and I am stating both.** (1) It says cycle 4's
  +36% redundancy swing was *"a swing larger than the entire `M = 8` → `M = 40` swing"* —
  **it is not**: 1.35714 against 1.68244, about half (row
  `brief_claim_checked__redundancy_swing_larger_than_whole_M_swing`, recomputed here from
  cycle 4's and cycle 5's own ledger rows). (2) It asks for the redundancy axis to be
  measured as achieved mean-max-cosine; **that is not a sufficient statistic.** Two
  constructions at the *same* achieved mean-max-cosine 0.51621 ± 0.00064 give ratios
  differing by **1.48406 ± 0.04645×** (§7). The verdict does not rest on either error.
- Ledger: `runs/cycle-redundancy-vs-M/ledger.json` — **838 rows; every number in this
  file is one of them**, including the cross-cycle comparisons, which are **read at
  runtime from the two prior ledgers** (row `prior_cycle_rows_read_from_their_ledgers`)
  rather than retyped. 94 statistic rows have `sd == 0.0`, all 94 classified by key in
  `sd_zero_rows_reasons`; **`sd_zero_rows_unclassified` is `[]`** (§10).
- Provenance stamped by the ledger itself: `git_sha 550a174` (the brief's baseline),
  `git_dirty: true` — that is **this untracked run directory only**;
  `/opt/homebrew/bin/git status --porcelain` shows nothing but
  `?? runs/cycle-redundancy-vs-M/`, and `git diff --stat HEAD` is empty. `src/`, `tests/`,
  `docs/` untouched; nothing committed. Python 3.12.13, macOS-26.6.2-arm64, CPU, 1 thread,
  **76.0 s wall**.
- Scratch harness (throwaway, per the rules):
  `/private/tmp/claude-501/-Users-keanooo7-second-brain/1bdd6660-ee4c-49c5-99a7-28bf139ef2c9/scratchpad/run_redundancy_vs_M.py`,
  reusing cycle 5's driver + shared-head counterfactual design and cycle 2's
  `scratch_bias.CallSiteBias`.
- **Geometry read from the registry**, never supplied: `b_max = 1.0` (FROZEN), `M = 8`
  (`e7`), `M = 16` (`synthetic`), `M = 40` (`pg19_e3`), `S = 48` (`synthetic`), `S = 80`
  (`pg19_e3`). `d_model = 384`. `get("S", "e7")` **raises `ScopeRequired`** — row
  `S_e7_registry_read_raises`; the stream length at `M = 8` is supplied and labelled.
- **Deliberate visible bypasses**, labelled as such in the ledger and *not* measurements
  of those constants: every `γ_b` (DERIVED, E0e), `τ = 0.25`, the EMA half-lives, the
  hand-built `RSRConfig`, the `b(slots)` accessor carried from cycle 2, `d_model = 384`,
  and `S = 80` held across all three `M`.
- **`ν = 0` everywhere**, so the score's *own* redundancy term `−ν·max cos` is OFF. That is
  deliberate and it is a real boundary: this cycle measures how redundancy moves the
  **margin**, not how the `ν` term would respond to it. `ν` is MEASURED (E1). §12.
- **`φ` is untrained and `ū` is synthetic.** `observe()` still raises; rows
  `phi_is_untrained`, `u_bar_is_synthetic`.
- **§13.** This is a comparison of a **dimensionless ratio** and of flip rates **within one
  synthetic generator**. No absolute number from one corpus is set against one from
  another. **There are no PG-19 gestalts in this repo and none were used.** Every sentence
  about what a real corpus's redundancy might be is marked as conjecture this experiment
  does not establish. Row `section_13_scope_statement`, written before any number.

---

## 0 — Design, in one paragraph

Two readouts on a crossed grid. The **order statistic** builds 800 independent full
memories of `n` gestalts per cell, scores each against one context with a single
`BilinearValueHead`, and takes the gap between the two **smallest** z-scores using
`std(unbiased=False)` — `rsr.py:369` and `rsr.py:406` verbatim, the same quantity as
`EvictionRecord.score_margin`. The **live memory** runs cycle 5's counterfactual: one
`b_enabled=False`, `ν=0` driver **owns** the memory so the `MemoryState` sequence is
identical across arms, every arm shares **one** `BilinearValueHead` so `ψ̂` is
bit-identical, 12 streams × `S = 80` per seed, and a *flip* is an eviction where the
`b`-enabled arm's victim differs from the driver's on the same state. 5 seeds throughout,
mean ± sd over seeds. Achieved redundancy in the live memory is measured by **calling
`rsr.py::RSRPolicy._max_cosine` on the driver's own `MemoryState`** and averaging over
live slots and decisions. Two `γ_b` arms: `formula_M` = correction 4's own
`b_max/(0.25·M)` = `4/M`, and `fixed_0.1` = that formula's value at `M = 40`.

**The redundancy construction heeds the brief's warning.** `gestalt + 0.15·randn(384)` has
noise norm 2.94 against a unit-norm gestalt and perturbs nothing; it is never used here.
Three constructions are, and **every cell reports its achieved mean-max-cosine, never the
nominal knob**:

| name | construction | knob |
|---|---|---|
| **broad** (parts A, A2, B) | `normalize((1−w)·centre_k + w·idio)`, `max(2, M//2)` topics, two members per topic in expectation | `w` |
| **near-duplicate** (parts E, F) | a fraction `f` of slots replaced by `ρ·g_src + √(1−ρ²)·e_perp`, `e_perp` orthogonalized against `g_src`, so `cos(g_dup, g_src) = ρ` **exactly** | `ρ`, `f` |
| **anisotropic** (part C) | `x ~ N(0, diag(k^−α))` then L2-normalized (correction 15 preserved: still unit norm) | `α`, reported as achieved participation ratio |

## 1 — The two numbers side by side. This is the falsifier's verdict.

On `b_max / median top-2 z-margin`, the dimensionless decision-strength ratio:

| | order statistic | live memory |
|---|---|---|
| **`M` effect**, `M = 8 → 40`, iid | **1.66915 ± 0.12036** | **1.65205 ± 0.17053** |
| **redundancy** at fixed `M = 40`, **broad** (achieved cos 0.62066 ± 0.00076 / 0.62364 ± 0.00194) | 1.14431 ± 0.08211 | 1.17951 ± 0.10620 |
| **redundancy** at fixed `M = 40`, **near-duplicate** (achieved cos 0.98040 ± 0.00038 / 0.73083 ± 0.00980) | **4.03253 ± 0.11747** | **2.37102 ± 0.21026** |
| **crossing**: achieved mean-max-cosine where redundancy = the whole `M` effect | **0.54132 ± 0.03193** | **0.54260 ± 0.04367** |

Rows `partD_M_effect_on_ratio_{orderstat,livemem}_M40_over_M08_iid`,
`partD_redundancy_effect_on_ratio_{orderstat,livemem}_M40_w0.4`,
`partD_dup_redundancy_effect_on_ratio_{orderstat,livemem}_M40_maxreach`,
`partD_dup_crossing_achieved_mean_max_cos_{orderstat,livemem}_M40` (5/5 seeds reached the
crossing in both — `partD_dup_crossing_seeds_reached`).

The `M`-effect row **replicates cycle 5's `part1_ratio_swing_M40_over_M8 = 1.68244 ±
0.04855`** from an independently written grid, at 1.66915 ± 0.12036.

**Below an achieved mean-max-cosine of ≈ 0.54 `M` is the bigger lever; above it,
redundancy is.** An `M`-keyed `b_max` is calibrated against a variable that stops being
the dominant one at a redundancy level that two different constructions locate at the same
place.

**On the realized flip rate the comparison goes the other way**, and both numbers belong
in the report:

| | `formula_M` (`γ_b = 4/M`) | `fixed_0.1` |
|---|---|---|
| `M` effect, `M = 8 → 40`, iid | **1.31354 ± 0.06434** | **1.91208 ± 0.10354** |
| redundancy at fixed `M = 40`, furthest reachable | **1.10803 ± 0.03838** | 1.10803 ± 0.03838 |
| seeds where redundancy reached the `M` effect | **0 of 5** | — |

Rows `partD_M_effect_on_flip_M40_over_M08_iid_*`,
`partD_dup_redundancy_effect_on_flip_livemem_M40_maxreach_*`,
`partD_dup_flip_crossing_seeds_reached`. (`fixed_0.1` *is* `formula_M` at `M = 40` by
construction, hence the identical redundancy row. Cycle 5's own values for the same two
`M` swings, read from its ledger: 1.27535 ± 0.06147 and 1.84636 ± 0.08705.)

**Why the readouts disagree — measured, not argued.** `M` acts on the flip rate through
**three** channels: the margin, the number of competing slots, and `γ_b` itself (`4/M`).
Redundancy acts through the margin **only**. At `M = 40` with `γ_b = 0.1`, `b` is
saturated on more than half of slot-decisions (cycle 5) and the flip rate sits at
0.58–0.64, so a −17% margin buys +6.6% flips. **The flip rate is a low-sensitivity
instrument for margin changes at these `γ_b`**, and any calibration target defined on it
inherits that. This is a finding, not a caveat: if "comparable decision authority" is
defined as a flip rate, the target barely moves for a large change in the thing it is
meant to measure.

## 2 — The `M` × redundancy grid, live memory, **broad** construction

`partB_selftest_hand_score_matches_policy_victim_* = 1.00000 ± 0.00000` in **all 36 arms**
— the hand-recomputed `argmin[(ψ̂−mean)/std(unbiased=False) + b]` reproduces `RSRPolicy`'s
own victim on every eviction, which is what licenses the flip counts.

| `M` | `w` | achieved mean-max-cos | median top-2 z-margin | `b_max`/margin | flip `formula_M` | flip `fixed_0.1` |
|---|---|---|---|---|---|---|
| 8 | 1.0 | 0.07010 ± 0.00180 | 0.53216 ± 0.01096 | 1.87978 ± 0.03881 | 0.46157 ± 0.01256 | 0.31713 ± 0.00781 |
| 8 | 0.8 | 0.09251 ± 0.00136 | 0.51874 ± 0.02627 | 1.93168 ± 0.09726 | 0.46875 ± 0.00835 | 0.32477 ± 0.01159 |
| 8 | 0.7 | 0.16282 ± 0.00105 | 0.52052 ± 0.01815 | 1.92304 ± 0.06741 | 0.46806 ± 0.01862 | 0.32755 ± 0.01568 |
| 8 | 0.6 | 0.29261 ± 0.00113 | 0.52271 ± 0.01869 | 1.91504 ± 0.06833 | 0.45764 ± 0.02384 | 0.30741 ± 0.01987 |
| 8 | 0.5 | 0.45767 ± 0.00238 | 0.53657 ± 0.01777 | 1.86532 ± 0.06097 | 0.44560 ± 0.01661 | 0.30764 ± 0.01404 |
| 8 | 0.4 | 0.62178 ± 0.00334 | 0.54155 ± 0.02088 | 1.84874 ± 0.07054 | 0.43727 ± 0.02825 | 0.30741 ± 0.01262 |
| 16 | 1.0 | 0.08952 ± 0.00071 | 0.40110 ± 0.03181 | 2.50581 ± 0.20028 | 0.53125 ± 0.01788 | 0.47813 ± 0.02319 |
| 16 | 0.8 | 0.10555 ± 0.00074 | 0.39903 ± 0.01915 | 2.51056 ± 0.11618 | 0.52812 ± 0.02868 | 0.47760 ± 0.02024 |
| 16 | 0.7 | 0.16921 ± 0.00333 | 0.38989 ± 0.01205 | 2.56675 ± 0.07732 | 0.53984 ± 0.01961 | 0.49401 ± 0.02046 |
| 16 | 0.6 | 0.29788 ± 0.00306 | 0.39465 ± 0.02025 | 2.53917 ± 0.12799 | 0.53438 ± 0.01775 | 0.48307 ± 0.01977 |
| 16 | 0.5 | 0.46187 ± 0.00258 | 0.38504 ± 0.02449 | 2.60534 ± 0.16213 | 0.53281 ± 0.01701 | 0.47969 ± 0.01185 |
| 16 | 0.4 | 0.62629 ± 0.00387 | 0.35934 ± 0.02229 | 2.79164 ± 0.17696 | 0.53073 ± 0.00763 | 0.47656 ± 0.00829 |
| 40 | 1.0 | 0.11068 ± 0.00105 | 0.32476 ± 0.03227 | 3.10417 ± 0.31578 | 0.60583 ± 0.02142 | 0.60583 ± 0.02142 |
| 40 | 0.8 | 0.11972 ± 0.00160 | 0.31034 ± 0.01349 | 3.22738 ± 0.14827 | 0.58500 ± 0.02847 | 0.58500 ± 0.02847 |
| 40 | 0.7 | 0.17166 ± 0.00194 | 0.31448 ± 0.02336 | 3.19412 ± 0.23944 | 0.59042 ± 0.01448 | 0.59042 ± 0.01448 |
| 40 | 0.6 | 0.29858 ± 0.00184 | 0.31021 ± 0.02623 | 3.24152 ± 0.26457 | 0.60708 ± 0.02071 | 0.60708 ± 0.02071 |
| 40 | 0.5 | 0.46143 ± 0.00333 | 0.29959 ± 0.00848 | 3.34003 ± 0.09509 | 0.60167 ± 0.01714 | 0.60167 ± 0.01714 |
| 40 | 0.4 | 0.62364 ± 0.00194 | 0.27550 ± 0.01610 | 3.63955 ± 0.21081 | 0.61167 ± 0.00432 | 0.61167 ± 0.00432 |

**Read the `M` = 8 block against the `M` = 40 block.** Broad correlation pushes the ratio
*down* at `M` = 8 (1.87978 → 1.84874) and *up* at `M` = 40 (3.10417 → 3.63955), and pushes
the flip rate *down* at `M` = 8 (0.46157 → 0.43727) while leaving `M` = 40 flat-to-up. The
interaction is already present in the mildest construction.

**Why broad correlation does so little, and it is mechanical.** A component shared by
*every* live slot is a **common additive offset** in `ψ̂`, and `_score` subtracts the
per-step mean (`finite.mean()`, `rsr.py:371`) one operation before z-scoring. An `argmin`
is blind to it. Cycle 4 found the identical mechanism from the other side (the per-step
mean's variance was the entire per-step-vs-pooled gap). With `max(2, M//2)` topics the
shared part is a topic-level random effect rather than a global one, so it is not
*entirely* absorbed — hence 1.14×, not 1.00× — but it is mostly absorbed. **The brief's
premise that "the margin distribution already moved +36% at fixed `M`" does not generalize
to redundancy as such; it was a property of cycle 4's near-duplicate structure.**

## 3 — The `M` × redundancy grid, live memory, **near-duplicate** construction

Exact pair cosine `ρ`, fraction `f`. Part E (`f = 0.25`, `ρ` swept) and part F
(`ρ = 0.99`, `f` swept) are points on one curve through the same structure family, read
against achieved mean-max-cosine. All 48 self-test rows (`partEf_selftest_*`,
`partFf_selftest_*`) are `1.00000 ± 0.00000`.

| `M` | knob | achieved mean-max-cos | median top-2 z-margin | `b_max`/margin | flip `formula_M` | flip `fixed_0.1` |
|---|---|---|---|---|---|---|
| 8 | `ρ=0` | 0.06873 ± 0.00203 | 0.53284 ± 0.01963 | 1.87880 ± 0.06975 | 0.48657 ± 0.00977 | 0.33912 ± 0.01317 |
| 8 | `ρ=0.5` | 0.17700 ± 0.01088 | 0.53126 ± 0.02041 | 1.88457 ± 0.07352 | 0.47755 ± 0.01414 | 0.32454 ± 0.01150 |
| 8 | `ρ=0.8` | 0.27911 ± 0.01372 | 0.51571 ± 0.03686 | 1.94706 ± 0.13958 | 0.46944 ± 0.01409 | 0.32685 ± 0.01626 |
| 8 | `ρ=0.9` | 0.33847 ± 0.01252 | 0.52240 ± 0.02740 | 1.91861 ± 0.10431 | 0.45810 ± 0.01607 | 0.32014 ± 0.02376 |
| 8 | `ρ=0.95` | 0.36458 ± 0.01272 | 0.49239 ± 0.01336 | 2.03209 ± 0.05347 | 0.47477 ± 0.02032 | 0.34028 ± 0.01776 |
| 8 | `ρ=0.99` | 0.40103 ± 0.02419 | 0.46484 ± 0.02087 | 2.15487 ± 0.09988 | 0.47130 ± 0.01809 | 0.34398 ± 0.01967 |
| 8 | `f=0.5` | 0.66702 ± 0.02236 | — | 2.25889 ± 0.22742 | 0.43079 ± 0.01978 | 0.35370 ± 0.02081 |
| 8 | `f=0.75` | 0.84402 ± 0.02133 | — | 2.34858 ± 0.29677 | **0.43009 ± 0.02705** | 0.37662 ± 0.03175 |
| 16 | `ρ=0` | 0.08880 ± 0.00178 | 0.39754 ± 0.01793 | 2.51958 ± 0.11346 | 0.52839 ± 0.02456 | 0.47552 ± 0.02767 |
| 16 | `ρ=0.5` | 0.19617 ± 0.00149 | 0.38794 ± 0.01574 | 2.58104 ± 0.10349 | 0.52917 ± 0.03421 | 0.47552 ± 0.03762 |
| 16 | `ρ=0.8` | 0.30279 ± 0.02253 | 0.38471 ± 0.01747 | 2.60361 ± 0.11778 | 0.53151 ± 0.02160 | 0.48411 ± 0.01645 |
| 16 | `ρ=0.9` | 0.35143 ± 0.01687 | 0.36231 ± 0.01547 | 2.76406 ± 0.11599 | 0.53516 ± 0.01473 | 0.48177 ± 0.01375 |
| 16 | `ρ=0.95` | 0.37733 ± 0.01027 | 0.34863 ± 0.01592 | 2.87313 ± 0.13092 | 0.53854 ± 0.00886 | 0.48177 ± 0.01554 |
| 16 | `ρ=0.99` | 0.40504 ± 0.01393 | 0.30810 ± 0.00735 | 3.24723 ± 0.07776 | 0.52812 ± 0.02766 | 0.48854 ± 0.02488 |
| 16 | `f=0.5` | 0.65648 ± 0.00769 | — | 3.93936 ± 0.27343 | 0.55599 ± 0.02513 | 0.52370 ± 0.02661 |
| 16 | `f=0.75` | 0.84253 ± 0.02212 | — | 4.89233 ± 0.60012 | 0.56198 ± 0.01567 | 0.54896 ± 0.02589 |
| 40 | `ρ=0` | 0.10948 ± 0.00126 | 0.31764 ± 0.01936 | 3.15777 ± 0.19603 | 0.58000 ± 0.02007 | 0.58000 ± 0.02007 |
| 40 | `ρ=0.5` | 0.20299 ± 0.00580 | 0.30648 ± 0.02118 | 3.27515 ± 0.22172 | 0.59125 ± 0.03147 | 0.59125 ± 0.03147 |
| 40 | `ρ=0.8` | 0.28628 ± 0.01225 | 0.31710 ± 0.01949 | 3.16275 ± 0.18738 | 0.59792 ± 0.03599 | 0.59792 ± 0.03599 |
| 40 | `ρ=0.9` | 0.31922 ± 0.01503 | 0.30554 ± 0.02217 | 3.28587 ± 0.22411 | 0.59208 ± 0.02489 | 0.59208 ± 0.02489 |
| 40 | `ρ=0.95` | 0.33770 ± 0.01383 | 0.29015 ± 0.01882 | 3.45872 ± 0.23659 | 0.59042 ± 0.03071 | 0.59042 ± 0.03071 |
| 40 | `ρ=0.99` | 0.35583 ± 0.01650 | 0.26520 ± 0.02208 | 3.79386 ± 0.34735 | 0.58917 ± 0.03292 | 0.58917 ± 0.03292 |
| 40 | `f=0.5` | 0.56601 ± 0.00910 | — | 5.47793 ± 0.52423 | 0.61792 ± 0.02050 | 0.61792 ± 0.02050 |
| 40 | `f=0.75` | 0.73083 ± 0.00980 | — | **7.50246 ± 0.99917** | **0.64208 ± 0.00839** | 0.64208 ± 0.00839 |

The order-statistic version of the same grid goes further, because a fixed-membership group
cannot lose a duplicate's source to eviction. At `n = 40`:

| knob | achieved mean-max-cos | `b_max`/margin |
|---|---|---|
| `ρ=0` | 0.10928 ± 0.00017 | 3.22344 ± 0.09828 |
| `ρ=0.5` | 0.29083 ± 0.00031 | 3.21525 ± 0.16612 |
| `ρ=0.8` | 0.42975 ± 0.00036 | 3.41251 ± 0.16496 |
| `ρ=0.9` | 0.47592 ± 0.00039 | 3.70550 ± 0.21585 |
| `ρ=0.95` | 0.49823 ± 0.00049 | 4.07985 ± 0.13627 |
| `ρ=0.99` | 0.51621 ± 0.00064 | 5.23194 ± 0.15544 |
| `f=0.5` | 0.83087 ± 0.00132 | 8.26462 ± 0.30082 |
| `f=0.75` | 0.98040 ± 0.00038 | 12.99493 ± 0.43138 |

The live-memory stream tops out lower (achieved cos 0.73083 vs 0.98040) because a
near-copy's source can be evicted before the copy is scored — worth knowing on its own:
**a live memory under a value-based policy is harder to saturate with duplicates than a
static sample of the same generator.**

**The two rows that carry the verdict, in the smallest form.** At `M` = 40 the
near-duplicate family moves `b_max`/margin from 3.15777 ± 0.19603 to 7.50246 ± 0.99917 —
a factor of **2.37102 ± 0.21026** — while the entire `M = 8 → M = 40` range moves it by
**1.65205 ± 0.17053**.

## 4 — The interaction. This is what kills the *fix*, not just the claim.

| measured at | `M = 8 → 40` effect on `b_max`/margin | `M = 8 → 40` effect on flip rate (`formula_M`) |
|---|---|---|
| iid (part B, `w = 1`) | 1.66915 ± 0.12036 (os) / 1.65205 ± 0.17053 (lm) | 1.31354 ± 0.06434 |
| near-dup control (`ρ = 0`) | 1.67089 ± 0.10944 (os) / 1.68269 ± 0.12209 (lm) | 1.19266 ± 0.05566 |
| `ρ = 0.99`, `f = 0.25` | 1.83519 ± 0.08298 (os) | 1.25281 ± 0.10505 |
| `ρ = 0.99`, `f = 0.5` | 1.36264 ± 0.13368 (os) | 1.43734 ± 0.09367 |
| `ρ = 0.99`, `f = 0.75` | 2.05065 ± 0.13253 (os) / **3.24214 ± 0.63011** (lm) | **1.49851 ± 0.11214** |
| broad, `w = 0.4` | 1.70227 ± 0.12479 (os) | — |

Rows `partD_M_effect_on_{ratio,flip}_at_{rho,f,w}*`. The three iid-ish baselines on the
ratio (1.66915, 1.67089, and cycle 5's 1.68244) and on the flip rate (1.31354, 1.19266,
and cycle 5's 1.27535) agree within noise; that spread is the harness's own seed-to-seed
variation and is the right yardstick for reading the rest of the column. **The flip-rate
gap grows from 1.19266 ± 0.05566 to 1.49851 ± 0.11214 across the redundancy range** —
non-overlapping at 1 sd each.

**And the sign of redundancy's effect on realized flips depends on `M`:**

| `M` | flip-rate effect at `ρ = 0.99`, `f = 0.5` | at `f = 0.75` |
|---|---|---|
| 8 | 0.88548 ± 0.04100 | **0.88401 ± 0.05446** |
| 16 | 1.05398 ± 0.06700 | 1.06609 ± 0.07082 |
| 40 | 1.06583 ± 0.03310 | **1.10803 ± 0.03838** |

Rows `partD_dup_redundancy_effect_on_flip_M*_f*_formula_M`. `M = 8` and `M = 40` straddle
1.0 and are ~3 sd apart. **A correction indexed on `M` alone would have to push the
small-`M` rate by a different amount at every redundancy, and in the opposite direction
from the one it pushes at `M = 40`.** That is what makes an `M`-keyed constant a partial
fix rather than a complete one, and it is measured, not inferred.

## 5 — Anisotropy, reached, with achieved effective dimension

Gestalts from a decaying spectrum then normalized — still unit norm, so correction 15
holds and only the direction law changes. **Achieved effective dimension** is the
participation ratio `(tr S)²/‖S‖²_F` of the second-moment matrix of 8000 normalized
samples, with the finite-`N` bias removed exactly: for unit rows
`E‖Ŝ‖²_F = 1/N + (1−1/N)‖S‖²_F`, and the estimator inverts that. `α` is reported only as
the knob, per the brief.

| `α` | **achieved participation ratio** | analytic PR of the pre-normalization spectrum | achieved mean-max-cos (`n = 40`) | `b_max`/margin (`n = 40`) | `b_max`/margin (`n = 8`) |
|---|---|---|---|---|---|
| 0.0 | **384.02721 ± 0.11063** | 384.0 | 0.10968 ± 0.00018 | 3.16845 ± 0.07619 | 1.91271 ± 0.10546 |
| 0.5 | **222.65502 ± 0.87799** | 218.3421 | 0.14192 ± 0.00021 | 3.27319 ± 0.06456 | 1.85003 ± 0.08894 |
| 1.0 | **32.63748 ± 0.41211** | 25.9569 | 0.33714 ± 0.00077 | 3.37819 ± 0.16516 | 1.93385 ± 0.05164 |
| 1.5 | **8.17544 ± 0.11283** | 5.2427 | 0.59776 ± 0.00020 | 3.82622 ± 0.36115 | 2.14781 ± 0.03556 |
| 2.0 | **4.10527 ± 0.03810** | 2.4921 | 0.76141 ± 0.00093 | 4.52231 ± 0.87321 | 2.36070 ± 0.24099 |

Live-memory flip runs on an anisotropic stream at `M = 40`:

| `α` | achieved mean-max-cos (live) | `b_max`/driver margin | flip `formula_M` |
|---|---|---|---|
| 0.0 | 0.11024 ± 0.00096 | 3.00488 ± 0.19063 | **0.59500 ± 0.01289** |
| 1.0 | 0.33416 ± 0.00181 | 3.36664 ± 0.19378 | 0.58958 ± 0.02425 |
| 2.0 | 0.77075 ± 0.00983 | 4.70504 ± 0.55795 | 0.65000 ± 0.03659 |

The `α = 0` flip row is **0.59500 ± 0.01289, identical in every digit to cycle 5's
`part3_frac_victims_flipped_by_b_M40_formula_M = 0.59500 ± 0.01289`** (read from its
ledger by this process), from a differently written stream generator. That is the
strongest cross-cycle check in this run.

**Anisotropy behaves like broad correlation, not like duplication.** Collapsing the
effective dimension from 384 to **4.1** — far past anything a 384-d encoder would
plausibly produce — moves the `M = 40` ratio only 1.43× (order statistic:
4.52231/3.16845) / 1.57× (live memory: 4.70504/3.00488), still below the 1.65–1.67 `M`
effect, and moves the flip rate 1.09×. It does so at an achieved mean-max-cosine of
0.76–0.77, *higher* than the near-duplicate live cell (0.73083) that moved the ratio
2.37×. **Third construction, same conclusion: the achieved mean-max-cosine does not
determine the margin.**

## 6 — Context correlation is not the driver either (part A2)

In a live memory `c_t` is the sentence just written, so it is correlated with the memory;
part A's independent context is a simplification. Repeating the whole order-statistic grid
with `c_t` drawn from one of the group's own topics moves nothing outside noise — at
`n = 40`, `w = 1.0`: 3.26519 ± 0.10996 (independent) vs 3.16033 ± 0.08999
(topic-correlated); at `w = 0.4`: 3.73214 ± 0.21833 vs 3.65550 ± 0.24046. Rows
`partA2_bmax_over_median_top2_z_margin_*`. Part B's live memory carries the correlated
context by construction anyway, and agrees with part A throughout.

## 7 — Mean-max-cosine is **not** a sufficient statistic, and that corrects the brief

For each near-duplicate cell at `M = 40`, the **topic-model** ratio was interpolated to the
**same achieved mean-max-cosine** and the two divided.

| near-dup cell | matched achieved mean-max-cos | near-dup ratio ÷ topic-model ratio at that cosine |
|---|---|---|
| `ρ = 0.5` | 0.29083 ± 0.00031 | 1.02269 ± 0.09633 |
| `ρ = 0.8` | 0.42975 ± 0.00036 | 1.01556 ± 0.06709 |
| `ρ = 0.9` | 0.47592 ± 0.00039 | 1.07801 ± 0.09474 |
| `ρ = 0.95` | 0.49823 ± 0.00049 | 1.17081 ± 0.07769 |
| `ρ = 0.99` | 0.51621 ± 0.00064 | **1.48406 ± 0.04645** |

Rows `partD_sufficiency_dup_over_topic_ratio_at_matched_cos_rho*`,
`partD_sufficiency_matched_cos_rho*`. Below an achieved cosine of ≈ 0.43 the two structures
agree to 2% and the mean **is** an adequate summary; above ≈ 0.48 they diverge, reaching
48% at 0.51621. §5's anisotropic arm is a third point off the same curve in the other
direction.

**Consequence.** The brief is right not to trust the nominal knob, but the measured mean is
not sufficient either: **what shrinks the top-2 margin is a near-tie between two slots,
i.e. the extreme upper tail of the pairwise-cosine distribution, not its typical value.**
Any registry option keyed off a measured redundancy scalar would have to key off a tail
statistic — the fraction of slots whose max cosine exceeds a threshold, or the expected
minimum `ψ̂` gap itself — not the mean.

## 8 — What would decide the registry question (options and costs; **not a pick**)

Cycle 5 tabled three options: (A) per-scope `b_max`, (B) an `M` factor in `γ_b`, (C) accept
it and write it into §13. This cycle changes what each costs, and adds a fourth.

**(A) `b_max` becomes per-scope, keyed on `M`.** *Costs more than cycle 5 thought.* It
closes the `M = 8` ↔ `M = 40` gap at one redundancy and opens it at another: the flip-rate
gap it would be tuned against is 1.19266 ± 0.05566 at the near-duplicate control and
1.49851 ± 0.11214 at high redundancy, and the *direction* of redundancy's effect differs
between `M = 8` and `M = 40` (§4). It also still gives up `b_max`'s one-line reading as one
SD of `ψ̂`, which cycle 4 measured to be correct (`part2_bmax_in_true_psi_sd_at_n40 =
0.98280 ± 0.00360`, read from its ledger).
*What would settle it:* a decision about **which redundancy the constant is tuned at**,
plus an argument that the residual mis-tuning elsewhere is acceptable. Both are statements
about the corpus, and there is no measurement of the corpus.

**(B) an `M` factor in `γ_b`.** Cycle 5 measured `γ_b = 4/M` as already closing about half
the `M` gap by accident. This cycle neither improves nor damages that: `γ_b` has no
redundancy dependence to exploit, and §1 shows the flip-rate readout `γ_b` acts on is a
low-sensitivity instrument to begin with. *Cost:* smallest diff; keeps the gap-closing it
already does. *What would settle it:* whether cycle 5's residual 1.27535 ± 0.06147 is
acceptable once §4's redundancy-dependence is added on top, and whether anyone accepts
`γ_b` no longer reading as "reach `b_max` in a quarter of a lifetime".

**(C) `b_max` normalized by a *measured* margin.** `b_max_eff = κ · median top-2 z-margin`,
estimated online from the `EvictionRecord.score_margin` stream the policy already logs.
*This is the only option that compensates for both axes at once*, because the margin is the
variable both `M` and redundancy act **through** — and §7's insufficiency problem
disappears, because the margin does not have to be *predicted* from a geometry statistic.
*Costs:* (i) `b_max` stops being FROZEN and becomes an online statistic, i.e. a second
feedback path alongside `b` itself, with a stability question this cycle did not test;
(ii) `b` stops being comparable across *steps* within a run, only across decisions of
equal difficulty, which changes what an attribution record means; (iii) it needs a warm-up
before the margin estimate exists, interacting with `T_warm`; (iv) §3.5 item 1's sentence
is replaced rather than amended. *What would settle it:* run this cycle's grid with
`b_max` divided by a trailing median margin and check the flip rate is flat across `M`
**and** across the redundancy sweep. One cycle on this harness.

**(D) `b_max` should not be frozen at all** — stated plainly as an option, as the brief
asks. On the evidence of cycles 4–6, `b_max` is a FROZEN constant whose only justified
reading ("one SD of `ψ̂`") is measured-correct and does no protective work (cycle 4), whose
decision-relevant value is 1.88–12.99× the margin depending on `M` and geometry (§2, §3),
and which therefore behaves as a **free parameter that was never swept**. *Cost:* it moves
`b_max` from FROZEN into MEASURED or CONDITIONAL, so `RSRConfig.from_registry` raises on it
until an experiment logs a value — by design (§4.5), the same mechanism that prevented
defect D-1 — and it adds a sweep dimension to E0e, crossed with `γ_b`.

**What all four need and none of them has.** One sentence naming the invariant: equal flip
rate? equal `b_max`/margin? equal fraction of decisions `b` could cross? This harness can
supply all three for any candidate in ~80 s. **The measurement that is missing is the
invariant, not more numbers. Nothing here picks.**

## 9 — Re-execution

The headline in the smallest standalone form: the `M` effect and the fixed-`M` redundancy
effect side by side, from registry reads, **no policy, no bias, no stream**. This snippet
is embedded verbatim in the harness as part G, so what it prints **is** three ledger rows.

```bash
cd /Users/keanooo7/retrieval-successor-retention && .venv/bin/python -c "
import sys, math, torch; sys.path.insert(0,'src')
from rsr import constants as C
from rsr.retention.value_head import BilinearValueHead
b_max = C.get('b_max')                       # FROZEN registry read
M8, M40 = C.get('M','e7'), C.get('M','pg19_e3')
d, G = 384, 800
def unit(x): return x/x.norm(dim=-1,keepdim=True)
def dup(g, rho, frac, gen):                  # exact pair cosine rho
    n=g.shape[1]; nd=int(round(frac*n)); nb=n-nd
    src=torch.randint(nb,(g.shape[0],nd),generator=gen)
    gs=torch.gather(g[:,:nb,:],1,src.unsqueeze(-1).expand(-1,-1,d)).clone()
    e=unit(torch.randn(g.shape[0],nd,d,generator=gen))
    e=unit(e-(e*gs).sum(-1,keepdim=True)*gs)
    g[:,nb:,:]=rho*gs+math.sqrt(1-rho*rho)*e; return g
def ratio(n, rho, frac, seed):
    gen=torch.Generator().manual_seed(4000+seed)
    h=BilinearValueHead(d,generator=gen); ms=[]
    with torch.no_grad():
        for _ in range(G//100):
            g=unit(torch.randn(100,n,d,generator=gen))
            if frac>0: g=dup(g,rho,frac,gen)
            v=h(g,unit(torch.randn(100,d,generator=gen))).double()
            z=(v-v.mean(1,keepdim=True))/v.std(1,unbiased=False).unsqueeze(1)  # rsr.py:369
            zs=z.sort(1).values; ms+= (zs[:,1]-zs[:,0]).tolist()              # rsr.py:406
    ms.sort(); return b_max/ms[len(ms)//2]
def ms_(xs):
    m=sum(xs)/len(xs); return m, math.sqrt(sum((x-m)**2 for x in xs)/(len(xs)-1))
Meff=[ratio(M40,0,0,s)/ratio(M8,0,0,s) for s in range(3)]
Reff=[ratio(M40,0.99,0.5,s)/ratio(M40,0,0,s) for s in range(3)]
print('M effect          M=%d -> M=%d, iid          : %.4f +/- %.4f' % (M8,M40,*ms_(Meff)))
print('redundancy effect M=%d, near-dup rho=0.99 f=.5: %.4f +/- %.4f' % (M40,*ms_(Reff)))
print('redundancy / M                               : %.4f' % (ms_(Reff)[0]/ms_(Meff)[0]))
"
```

Prints, bit-exactly — rows `partG_reexec_M_effect_on_ratio_M40_over_M08_iid`
(1.64205 ± 0.07139), `partG_reexec_redundancy_effect_on_ratio_M40_rho0.99_f0.5`
(2.59278 ± 0.10736) and `partG_reexec_redundancy_over_M` (1.579):

```
M effect          M=8 -> M=40, iid          : 1.6420 +/- 0.0714
redundancy effect M=40, near-dup rho=0.99 f=.5: 2.5928 +/- 0.1074
redundancy / M                               : 1.5790
```

**A redundancy manipulation at fixed `M` moves `b`'s decision strength 1.58× as far as the
entire `M` range the project trains.** That is the falsifier, in one command, in ~25 s.

Full experiment, 76 s, rewrites this directory's `ledger.json`:

```bash
cd /Users/keanooo7/retrieval-successor-retention && .venv/bin/python \
  /private/tmp/claude-501/-Users-keanooo7-second-brain/1bdd6660-ee4c-49c5-99a7-28bf139ef2c9/scratchpad/run_redundancy_vs_M.py
```

## 10 — The `sd = 0.0000` rows

94 rows, all classified by key in `sd_zero_rows_reasons` by a classifier in the same
process; **`sd_zero_rows_unclassified` is `[]`**. Two categories, no others:

- **Self-tests at their required value 1.0 — 90 rows.**
  `partB_selftest_hand_score_matches_policy_victim_*` (36),
  `partEf_selftest_hand_score_matches_policy_victim_*` (36),
  `partFf_selftest_hand_score_matches_policy_victim_*` (12),
  `partC_flip_selftest_*` (6). Each is the fraction of evictions where the hand-recomputed
  `argmin[(ψ̂−mean)/std(unbiased=False)+b]` equals `RSRPolicy`'s own victim. A value below
  1.0 would invalidate every flip count in this run; 1.0 across all seeds is their licence.
- **Structural counts with no RNG — 4 rows.**
  `partB_n_evictions_per_seed_per_arm_M08` = 864.00000, `_M16` = 768.00000,
  `_M40` = 480.00000 (all `12 × (80 − M)`), and `partA_groups_per_cell` = 800.00000.

**The seed does reach the RNG.** Every cosine, margin, ratio, flip-rate, saturation,
crossing, participation-ratio and swing row has `sd > 0`. Independent evidence: the `α = 0`
anisotropic flip row reproduces cycle 5's `M = 40` flip rate in every digit
(0.59500 ± 0.01289) from a different generator, and the `M` effect on the ratio reproduces
cycle 5's 1.68244 ± 0.04855 at 1.66915 ± 0.12036.

## 11 — What this does NOT establish

- **Nothing about PG-19.** There are no PG-19 gestalts in this repo, none were used, and
  `φ` was never trained on text. Every number here is one synthetic generator. **What a
  real corpus's achieved gestalt redundancy is, and whether it sits above or below this
  cycle's crossing of 0.54132 ± 0.03193, is not measured and is not measurable from
  anything in this repo.** For orientation only, and *as conjecture this experiment does
  not establish*: cycle 4's `tight` arm reached
  `part4_mean_max_cosine_within_group_redundant_tight = 0.42423 ± 0.00098` and its `loose`
  arm 0.18332 ± 0.00026 — both **below** 0.54132. If a real corpus sits there, the `M` axis
  is the bigger one in practice and an `M`-keyed fix is a good approximation. If real
  gestalts carry a heavier near-duplicate tail, it is not. **That is exactly the
  measurement neither this cycle nor any prior one has.** §13 forbids reading a synthetic
  number as a corpus number, and this bullet is the boundary.
- **Mean-max-cosine is the wrong summary above ≈ 0.48** (§7), so even a *measured* corpus
  mean-max-cosine would not settle the question. The tail statistic that would is named in
  §7 and was not measured here.
- **`φ` is untrained.** Cycles 4 and 5 established that the sd-to-margin relation is
  invariant to `φ`'s weight scale (exactly, by degree-1 homogeneity) and near-invariant to
  rank-1 drift, *as long as the head keeps this form*: `ψ̂` is a linear functional of `s`,
  so its across-slot law is near-Gaussian for spherical gestalts whatever `W, u` are. What
  training changes is the **gestalt distribution** — precisely this cycle's axis, explored
  synthetically rather than by training. A trained `ψ̂` with a bimodal across-slot law
  would widen the top-2 gap and **weaken** `b`; one with a near-tied lower tail would
  **strengthen** it. **Whether the crossing at 0.54132 survives training is not
  established.**
- **`ū` is synthetic.** Every flip rate is conditional on a log-normal salience softmax and
  `τ = 0.25`. Parts A, A2 and the order-statistic halves of E/F and C do not involve `ū` at
  all. A real utilisation signal could move the flip rates either way; it cannot move the
  margin distribution, which is where both effects live.
- **`ν = 0`.** The score's own `−ν·max cos` term is off, so this cycle does not measure the
  mechanism §3.4 built *for* redundancy. With `ν > 0` the two interact and the result could
  differ in either direction. §12.
- **"Changed the victim" is not "made it worse."** No language model ran, no loss was
  computed, no `r_i` was collected, no LOO Δloss. And under high redundancy a flip is often
  a choice between two *near-equivalent* slots, so the flip rate arguably **over**-states
  harm exactly where redundancy is high — the opposite direction from the ratio. Neither
  effect was quantified.
- **Nothing about `γ_b`, `τ`, `b_max`, `E_lifetime` or `ν` as constants.** E0e's and E1's.
  Every `γ_b` used is a labelled evaluation of correction 4's formula at a hand-supplied
  `E[lifetime]`.
- **`S = 80` for all three `M`**, supplied not read at `M = 8` (`get("S","e7")` raises).
  Cycle 5 part 4 is the only `S`-control and it exists only at `M = 16`.
- **`f = 0.75` is not a plausible corpus.** Three quarters of slots being near-copies of
  the other quarter is a stress test, present because the live-memory stream tops out at
  achieved cosine 0.73083 and the crossing had to be bracketed. The *crossing* at 0.54132
  is interpolated between cells at 0.35583 and 0.56601 and does not depend on the
  `f = 0.75` point.
- **One head form, one `d`.** `d_model = 384`, `BilinearValueHead` at init. No `d` sweep.
- **CPU only**, one thread. No MPS, no device claim.
- **Cycle-1 walls still stand**: no constant was measured, nothing was committed, `src/`
  `tests/` `docs/` untouched.
- **The known defects are unchanged and untouched**: `ProtectionBias`'s declared surface
  still disagrees with its call site in `_score`; `attribution` and `n_live` are still
  config-echo fields; `run_policy_loop` still never trains `ψ̂`.

## 12 — The single next falsifier I would name

> **"A `ν > 0` redundancy term in the score removes the redundancy sensitivity that §3.4
> exists to remove, so once `ν` is on, `b`'s decision strength really is a function of `M`
> alone."**

It is the right next one because it is the *repair hypothesis* implied by everything above.
§3.4's `−ν·max cos` is the one term in `_score` aimed at exactly the geometry this cycle
showed dominates, and it has been **off in every cycle so far** (2–6 all ran `ν = 0`). If
`ν > 0` flattens the redundancy axis, options (A)/(B) survive and the registry question is
small. If it does not — or if it flattens the margin while *inflating* the flip rate, which
is plausible because subtracting a term that is itself near-equal across a duplicate pair
can create ties rather than break them — then the margin-keyed option (C) is the only
complete one and §3.4 needs re-deriving.

It is **reachable now**: this exact harness with `nu` non-zero in the driver and the arms,
on the same grid, ~80 s. The one blocker is honest and small — `ν` is MEASURED by E1 and
the registry will refuse the read, so the cycle must sweep `ν` as a **labelled visible
bypass** against a `max cos` term whose own range is measured in §2–§5 above, and report a
*curve*, never a value for `ν`. That is the same discipline cycles 2–6 used for `γ_b`, so
it costs nothing new.
