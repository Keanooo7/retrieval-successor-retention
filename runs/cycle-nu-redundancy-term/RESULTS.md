# cycle-nu-redundancy-term — `ν > 0` does not flatten the redundancy axis. In a live memory it **inflates** it, and the mechanism is exactly the tie creation the brief asked me to look for

- **Falsifier.** *"A `ν > 0` redundancy term removes the redundancy sensitivity that §3.4
  exists to remove — so once `ν` is on, `b`'s decision strength really is a function of `M`
  alone."*
- **Verdict: FALSIFIED**, on three counts.
  1. **In a live memory `ν` makes it worse, not better.** At `M = 40`, `b_max / median
     top-2 score margin`, the redundancy swing (`ρ = 0.99, f = 0.75` over iid) is
     **2.37102 ± 0.21026** at `ν = 0` and **4.16593 ± 0.48548** at `ν = 4` — a **1.76×
     inflation** — while the `M` swing over the same range is untouched (**1.68269 ±
     0.12209** → **1.67943 ± 0.12218**). The flip-rate redundancy swing also rises,
     **1.10803 ± 0.03838 → 1.20914 ± 0.03203**, against an essentially flat `M` flip swing
     (1.19266 ± 0.05566 → 1.21233 ± 0.02551).
  2. **In a fixed-membership sample `ν` does nothing at all.** Paired per seed on the same
     800 groups, the redundancy swing at `ν` relative to `ν = 0` is **1.02535 ± 0.06828**
     at `ν = 8` — i.e. 1.0 to within noise across the entire sweep. A flattening would have
     driven this to `1/4.09 = 0.24`.
  3. **The mechanism is tie creation, and it is an identity, not a tendency.** The
     max-cosine penalty is **equal within a constructed near-duplicate pair to float32
     precision** — mean `|Δpenalty| = 4.102e-08 ± 4.342e-10` at `n = 40, ρ = 0.99,
     f = 0.25` — so the within-pair **score** gap is **invariant in `ν` to nine
     significant figures**: seed 0 reads `0.1169522568` at `ν = 0` and `0.1169522510` at
     `ν = 4` (rows `partF_E2_within_pair_score_gap_samples_nu*`), and the ratio row
     `partF_E2_within_pair_score_gap_inflation_nu4_over_nu0` is `1.00000 ± 0.00000`.
     `ν` cannot separate the pair it is aimed at. What it does instead is push
     **both** members to the bottom: the fraction of decisions whose two lowest-scoring
     slots are a near-duplicate pair goes **0.2513 ± 0.0088 → 0.8408 ± 0.0092** as `ν`
     goes 0 → 4, and `b_max/margin` goes **5.1834 ± 0.5373 → 12.2316 ± 0.3355`.
- **The brief contains an error and I am stating it.** Item 4 of *THE FALSIFIER* offers as
  the interesting second outcome that `ν` "flattens the **margin** while **inflating** the
  flip rate", and gives as its mechanism that "subtracting a term that is near-equal across
  a duplicate pair can create ties rather than break them". **Those two are contradictory.**
  Tie creation *shrinks* the margin — that is what a tie is — so it raises `b_max/margin`
  and raises the flip rate together; it cannot flatten the margin. The measurement
  confirms the mechanism and refutes the outcome as worded: at `M = 40, f = 0.25`, `ν` 0→4
  moves the median margin **0.26520 ± 0.02208 → 0.08767 ± 0.00508** *and* the flip rate
  **0.58917 ± 0.03292 → 0.61542 ± 0.02364** — the same direction, not opposite ones. The
  wording is inherited from cycle 6's own §12, so it is the previous researcher's slip
  carried forward, not a new one. The verdict does not rest on it.
- **A second, smaller framing point.** The brief says to measure "the top-2 **z-margin**
  after the `ν` term is applied". After the `ν` term the quantity is no longer on a `z`
  scale: `_score` z-scores `ψ̂` and then subtracts `ν·max cos` **raw**. Everything here is
  reported as the top-2 **score** margin, which is what `EvictionRecord.score_margin`
  actually records and what the argmin turns on.
- Ledger: `runs/cycle-nu-redundancy-term/ledger.json` — **2091 rows; every number in this
  file is one of them.** 237 statistic rows have `sd == 0.0`, all 237 classified by a
  classifier in the same process; **`sd_zero_rows_unclassified` is `[]`** (§10).
- Provenance stamped by the ledger itself: `git_sha 6674cea` (the brief's baseline),
  `git_dirty: true` — that is **this untracked run directory only**.
  `/opt/homebrew/bin/git status --porcelain` shows nothing but
  `?? runs/cycle-nu-redundancy-term/`; `git diff --stat HEAD` is empty. `src/`, `tests/`,
  `docs/` untouched; nothing committed. Python 3.12.13, macOS-26.6.2-arm64, CPU, 1 thread,
  **65.0 s wall**.
- Scratch harness (throwaway, per the rules):
  `/private/tmp/claude-501/-Users-keanooo7-second-brain/1bdd6660-ee4c-49c5-99a7-28bf139ef2c9/scratchpad/run_nu_redundancy.py`,
  reusing cycle 6's driver + shared-head counterfactual design, cycle 6's exact-cosine
  near-duplicate construction, and cycle 2's `scratch_bias.CallSiteBias`.
- **Geometry read from the registry**, never supplied: `b_max = 1.0` (FROZEN), `M = 8`
  (`e7`), `M = 16` (`synthetic`), `M = 40` (`pg19_e3`), `S = 48` (`synthetic`), `S = 80`
  (`pg19_e3`). `get("S","e7")` **raises `ScopeRequired`** (row `S_e7_registry_read_raises`);
  the stream length at `M = 8` is supplied and labelled. `get("nu","synthetic")` **raises
  `UnmeasuredConstant`, naming E1** (row `nu_registry_read_raises`) — so **every `ν` in
  this run is a deliberate, visible bypass and NO value of `ν` is reported as a
  measurement of that constant.** The result is a curve.
- **Other deliberate visible bypasses**, labelled in the ledger: every `γ_b` (DERIVED,
  E0e), `τ = 0.25`, the EMA half-lives, the hand-built `RSRConfig`, the `b(slots)`
  accessor carried from cycle 2, `d_model = 384`, and `S = 80` held across all three `M`.
- **`φ` is untrained and `ū` is synthetic.** `observe()` still raises; rows
  `phi_is_untrained`, `u_bar_is_synthetic`.
- **§13.** Dimensionless ratios and flip rates **within one synthetic generator**. No
  absolute number from one corpus is set against one from another. **There are no PG-19
  gestalts in this repo and none were used.** Row `section_13_scope_statement`, written
  before any number.

---

## 0 — Design, in one paragraph

Two readouts. The **order statistic** builds 800 independent full memories of `n` gestalts
per cell, scores each against one context with a single `BilinearValueHead`, and combines
as `score = (ψ̂ − mean)/std(unbiased=False) − ν·max_cos` — `rsr.py::_score` verbatim for a
full memory — then takes the gap between the two lowest scores, the same quantity as
`EvictionRecord.score_margin`. **`ν` changes only a post-hoc combination of `(ψ̂, max_cos)`,
so one draw serves every `ν`: the `ν` curve is *paired* on identical groups, identical
`ψ̂`, identical cosines.** The **live memory** runs cycle 6's counterfactual: one `ν = 0`,
`b`-free driver **owns** the memory so the `MemoryState` sequence is identical across every
arm and every `ν`; every arm shares **one** `BilinearValueHead` so `ψ̂` is bit-identical;
12 streams × `S = 80` per seed; for each `ν` there is a `b`-off arm (its margin and victim)
and a `b`-on arm, and a **flip** is an eviction where the `b`-on arm's victim differs from
the `b`-**off** arm *at the same `ν`*. 5 seeds throughout, mean ± sd over seeds.

`ν` sweep: `{0, 0.125, 0.25, 0.5, 1, 2, 4, 8}` (order statistic), `{0, 0.5, 1, 2, 4}` (live
memory). The range is set against the max-cosine term's **own measured range** (cycle 6:
achieved mean-max-cos 0.07 iid to 0.98 saturated), so `ν·Δ(max cos)` spans 0 to ≈`0.9ν`
against a z-scored `ψ̂` of sd 1. `ν = 1` therefore puts the term's full swing at roughly
**one SD of `ψ̂`, i.e. exactly `b_max`'s own reach**, and `ν = 8` is deep in the regime
where the term dominates the argmin outright. Constructions (achieved mean-max-cosine
reported in every cell, never the nominal knob):

| name | construction | achieved mean-max-cos at `n = 40` |
|---|---|---|
| **iid** | `ρ = 0` through the duplicate code path (control) | 0.10937 ± 0.00025 |
| **broad_w0.4** | topic model, `max(2, n//2)` topics — the **flat control** | 0.62042 ± 0.00047 |
| **dup ρ=0.9 f=0.25** | exact pair cosine `ρ` on a fraction `f` | 0.47565 ± 0.00023 |
| **dup ρ=0.99 f=0.25** | " | 0.51654 ± 0.00024 |
| **dup ρ=0.99 f=0.5** | " | 0.82970 ± 0.00199 |
| **dup ρ=0.99 f=0.75** | " | 0.98023 ± 0.00023 |

All 77 self-test rows (`partB_selftest_*`, `partC_selftest_*`) are **1.00000 ± 0.00000**:
the hand-recomputed `argmin[(ψ̂−mean)/std(unbiased=False) + b − ν·max_cos]` reproduces
`RSRPolicy`'s own victim on **every** eviction, which is what licenses every flip count —
and this version of the self-test includes the `ν` term, so it is also the proof the term
is live.

## 1 — The `ν` curve. Redundancy sensitivity as a function of `ν`.

**Order statistic, `n = 40`, paired per seed on identical groups.** "REL" is the
redundancy swing at `ν` divided by the redundancy swing at `ν = 0`, seed by seed, so the
seed noise cancels. A surviving falsifier would drive REL to `1/4.093 = 0.244`.

| `ν` | redundancy swing `f=0.75`/iid | **REL to `ν=0`** | `M` swing 40/8 iid | **REL to `ν=0`** | flat control (broad/iid) |
|---|---|---|---|---|---|
| 0 | 4.09255 ± 0.10986 | 1.00000 ± 0.00000 | 1.61821 ± 0.10832 | 1.00000 ± 0.00000 | 1.14737 ± 0.04263 |
| 0.125 | 4.09958 ± 0.10061 | 1.00180 ± 0.00873 | 1.62437 ± 0.09921 | 1.00413 ± 0.00692 | 1.15538 ± 0.03429 |
| 0.25 | 4.12731 ± 0.10654 | 1.00876 ± 0.02491 | 1.63157 ± 0.09950 | 1.00863 ± 0.01329 | 1.15722 ± 0.05145 |
| 0.5 | 4.15527 ± 0.10600 | 1.01558 ± 0.02385 | 1.63018 ± 0.10281 | 1.00768 ± 0.01458 | 1.17058 ± 0.04088 |
| 1 | 4.13099 ± 0.07184 | 1.00975 ± 0.02141 | 1.64316 ± 0.10352 | 1.01573 ± 0.01894 | 1.17015 ± 0.06118 |
| 2 | 4.15533 ± 0.07806 | 1.01563 ± 0.01911 | 1.63738 ± 0.10156 | 1.01223 ± 0.01648 | 1.16983 ± 0.06770 |
| 4 | 4.13404 ± 0.13577 | 1.01076 ± 0.04395 | 1.64972 ± 0.08256 | 1.02104 ± 0.04339 | 1.15422 ± 0.05598 |
| 8 | 4.19261 ± 0.22974 | **1.02535 ± 0.06828** | 1.64965 ± 0.09805 | 1.02170 ± 0.06845 | 1.17040 ± 0.11917 |

Rows `partD_orderstat_redundancy_swing_{ratio,RELATIVE_to_nu0}_*`,
`partD_orderstat_M_swing_*`, `partD_orderstat_redundancy_swing_flat_control_*`.
**The curve is flat. `ν` at eight times the scale of `b_max` moves the redundancy
sensitivity of a fixed-membership sample by 2.5% ± 6.8%.**

**Live memory, `M = 40`.** Here the curve is not flat — it goes the wrong way.

| `ν` | red. swing on `b_max`/margin (`f=0.75`/iid) | `M` swing on the ratio | red. swing on the **flip rate** | `M` swing on the flip rate |
|---|---|---|---|---|
| 0 | **2.37102 ± 0.21026** | 1.68269 ± 0.12209 | **1.10803 ± 0.03838** | 1.19266 ± 0.05566 |
| 0.5 | 3.07131 ± 0.24711 | 1.68251 ± 0.12929 | 1.15562 ± 0.05091 | 1.19109 ± 0.04419 |
| 1 | 3.67317 ± 0.31152 | 1.66607 ± 0.12742 | 1.16976 ± 0.04991 | 1.19522 ± 0.03961 |
| 2 | 4.00536 ± 0.49241 | 1.69807 ± 0.12460 | 1.20525 ± 0.04212 | 1.20664 ± 0.03275 |
| 4 | **4.16593 ± 0.48548** | **1.67943 ± 0.12218** | **1.20914 ± 0.03203** | 1.21233 ± 0.02551 |

Rows `partD_livemem_{redundancy,M}_swing_{ratio,flip}_*`. **`ν` inflates the redundancy
swing on the ratio by 1.757× and on the flip rate by 1.091×, and leaves the `M` swing
alone.** This is the opposite of the repair the falsifier proposes.

**Why the two readouts disagree — measured, not argued.** The order-statistic `f = 0.75`
cell has achieved mean-max-cos **0.98023**: *nearly every* slot is in a near-duplicate
pair, so `ν·max_cos` is a near-**common additive offset** and the argmin is blind to it —
the same mechanism cycle 6 found for broad correlation. The row that proves it is
`partA_nonpair_abs_penalty_diff_n40_dup_rho0.99_f0.75 = 0.01922 ± 0.00068`: the penalty
differs by 2% of a cosine unit even between *unrelated* slots there. In the live memory the
same nominal `f` reaches only **0.73083**, because a near-copy's source can be evicted
before the copy is scored, so the memory is a **mixture** of duplicated and unduplicated
slots — and a mixture is exactly what `ν` can sort. The order statistic is not a null; it
is a **saturation** result, and it says `ν`'s effect is a function of the *dispersion* of
`max_cos`, not its level.

## 2 — The tie-creation result. Measured, not inferred.

**The penalty is equal within a pair to float32 precision.** For a constructed
(source, copy) pair at exact cosine `ρ`, both members' nearest live neighbour *is each
other*, so both receive the identical penalty:

| construction, `n = 40` | `|Δpenalty|` **within** a pair | `|Δpenalty|` between random slots | `|Δz(ψ̂)|` within a pair | between random slots |
|---|---|---|---|---|
| `ρ=0.9, f=0.25` | **0.00000 ± 0.00000** | 0.40094 ± 0.00183 | 0.36955 ± 0.00389 | 1.12464 ± 0.01450 |
| `ρ=0.99, f=0.25` | **0.00000 ± 0.00000** | 0.44645 ± 0.00204 | 0.11727 ± 0.00144 | 1.12265 ± 0.01484 |
| `ρ=0.99, f=0.5` | **0.00000 ± 0.00000** | 0.26231 ± 0.00478 | 0.12244 ± 0.00144 | 1.11502 ± 0.00487 |
| `ρ=0.99, f=0.75` | **0.00000 ± 0.00000** | 0.01922 ± 0.00068 | 0.13273 ± 0.00121 | 1.10557 ± 0.00568 |

Rows `partA_{pair,nonpair}_abs_{penalty_diff,z_gap}_n40_*`. The zeros are not rounded:
the raw per-seed samples for `ρ=0.99, f=0.25` are
`[4.081e-08, 4.072e-08, 4.151e-08, 4.107e-08, 4.104e-08]` — float32 epsilon on a Gram
matrix, itemized in §10.

**Consequence, measured directly.** The within-pair **score** gap is therefore identical at
every `ν`, and the ledger carries it as eight separate rows that agree to five decimals:

| `ν` | 0 | 0.125 | 0.25 | 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|---|---|
| `partA_pair_abs_score_gap_n40_dup_rho0.99_f0.25` | 0.11727 ± 0.00144 | 0.11727 | 0.11727 | 0.11727 | 0.11727 | 0.11727 | 0.11727 | 0.11727 |

To be exact rather than rounded, the raw per-seed values from the re-execution snippet
(rows `partF_E2_within_pair_score_gap_samples_nu*`, seed 0 shown):

| `ν` | 0 | 0.25 | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|---|---|
| within-pair \|score gap\| | 0.1169522568 | 0.1169522564 | 0.1169522561 | 0.1169522553 | 0.1169522539 | 0.1169522510 |

**Nine significant figures of agreement across a 4× range of `ν`.** The drift in the tenth
digit is `ν` times the float32 noise in the Gram matrix, and the paired ratio rows
`partF_E2_within_pair_score_gap_inflation_nu*_over_nu0` are `1.00000 ± 0.00000` at every
`ν`. This is the falsifier's cause of death: the term is **rank-preserving inside the pair
that sets the margin.**

**So `ν` cannot break the tie. It selects for it.** Fraction of decisions whose two
lowest-scoring slots are a near-duplicate pair (pairwise cosine ≥ 0.9), order statistic,
`n = 40`:

| `ν` | `f=0.25` | `f=0.5` | `f=0.75` | iid | broad (flat control) |
|---|---|---|---|---|---|
| 0 | **0.25500 ± 0.00940** | 0.53750 ± 0.01218 | 0.84050 ± 0.00629 | 0.00000 | 0.00000 |
| 0.5 | 0.42900 ± 0.01673 | 0.67400 ± 0.01659 | 0.85725 ± 0.00709 | 0.00000 | 0.00000 |
| 1 | 0.60150 ± 0.00675 | 0.75625 ± 0.01627 | 0.86675 ± 0.00763 | 0.00000 | 0.00000 |
| 2 | 0.79600 ± 0.01288 | 0.81950 ± 0.01360 | 0.87100 ± 0.00768 | 0.00000 | 0.00000 |
| 4 | **0.84650 ± 0.01144** | 0.82850 ± 0.01604 | 0.87325 ± 0.00808 | 0.00000 | 0.00000 |

Rows `partA_frac_bottom2_cos_ge_0.9_*` and, on the labelled `(source, copy)` identity
rather than a cosine threshold, `partA_frac_bottom2_is_labelled_pair_*` (0.24475 ± 0.00894
→ 0.80725 ± 0.01257 over the same range at `f = 0.25`). **The iid and broad columns are
identically zero at every `ν`**: `ν` creates no ties where the generator planted none,
which is the control that rules out an artifact of the statistic.

**And in the live memory, where it costs something.** `M = 40`, `ρ = 0.99, f = 0.25`
(achieved mean-max-cos 0.35583 ± 0.01650):

| `ν` | median bottom-2 cosine | frac bottom-2 cos ≥ 0.9 | median score margin | `b_max`/margin | flip rate | `ν`'s own authority |
|---|---|---|---|---|---|---|
| 0 | 0.01679 ± 0.00402 | 0.11917 ± 0.01191 | **0.26520 ± 0.02208** | **3.79386 ± 0.34735** | 0.58917 ± 0.03292 | 0 (is the driver) |
| 0.5 | 0.02969 ± 0.00379 | 0.24125 ± 0.01682 | 0.19316 ± 0.01605 | 5.20473 ± 0.41893 | 0.61625 ± 0.03116 | 0.15542 ± 0.02050 |
| 1 | 0.06721 ± 0.01501 | 0.41625 ± 0.02317 | 0.15156 ± 0.01267 | 6.63786 ± 0.59640 | 0.63333 ± 0.03055 | 0.34583 ± 0.02509 |
| 2 | **0.99000 ± 0.00000** | 0.70917 ± 0.01339 | 0.10165 ± 0.00504 | 9.85721 ± 0.48577 | 0.62500 ± 0.02814 | 0.67167 ± 0.01463 |
| 4 | **0.99000 ± 0.00000** | 0.87000 ± 0.00815 | **0.08767 ± 0.00508** | **11.43746 ± 0.67145** | 0.61542 ± 0.02364 | 0.83833 ± 0.01715 |

Rows `partB_*_M40_dup_rho0.99_f0.25_nu*`. **The median pairwise cosine of the two
lowest-scoring slots reaches `ρ` exactly** — by `ν = 2` the *typical* eviction decision is
a coin-flip between two slots at cosine 0.99. `b_max/margin` triples. The last column is
`ν`'s own authority: the fraction of evictions where the `ν` arm's victim differs from the
`ν = 0` driver's on the same state, which reaches 0.838 — `ν` is not a small perturbation
at these settings, it is running the policy.

The `f = 0.75` live cell is the same story one notch further: `b_max`/margin
**7.50246 ± 0.99917 → 13.17335 ± 0.95179**, flip rate **0.64208 ± 0.00839 → 0.70542 ±
0.00543**. Both move up together, which is the arithmetic point §0's bullet makes against
the brief's "flattens the margin while inflating the flip rate".

**The flat control behaves.** On `broad_w0.4` at `M = 40`, `ν` 0→4 leaves the ratio at
3.63955 ± 0.21081 → 3.68073 ± 0.31420 and the flip rate at 0.61167 ± 0.00432 → 0.61125 ±
0.02055, while still changing 26% of victims. **`ν` reorders slots under broad correlation
without changing the decision difficulty at all** — it is only near-duplicate *pairs* that
convert into ties.

## 3 — `b`'s authority at `ν = 0` vs the largest `ν`, side by side

| readout | swing | `ν = 0` | largest `ν` | direction |
|---|---|---|---|---|
| order statistic, `b_max`/margin | `M`: `n=40` / `n=8`, iid | 1.61821 ± 0.10832 | 1.64965 ± 0.09805 (`ν=8`) | unchanged |
| order statistic, `b_max`/margin | redundancy: `f=0.75` / iid at `n=40` | 4.09255 ± 0.10986 | 4.19261 ± 0.22974 (`ν=8`) | unchanged |
| live memory, `b_max`/margin | `M`: `M=40` / `M=8`, iid | 1.68269 ± 0.12209 | 1.67943 ± 0.12218 (`ν=4`) | unchanged |
| live memory, `b_max`/margin | redundancy: `f=0.75` / iid at `M=40` | **2.37102 ± 0.21026** | **4.16593 ± 0.48548** (`ν=4`) | **worse, 1.757×** |
| live memory, flip rate | `M`: `M=40` / `M=8`, iid | 1.19266 ± 0.05566 | 1.21233 ± 0.02551 (`ν=4`) | unchanged |
| live memory, flip rate | redundancy: `f=0.75` / iid at `M=40` | **1.10803 ± 0.03838** | **1.20914 ± 0.03203** (`ν=4`) | **worse, 1.091×** |

Every ratio in that column is also its own ledger row, formed **per seed** and then
averaged so it carries an sd rather than being a ratio of two means (part F):

| row | value |
|---|---|
| `partF_livemem_redundancy_swing_OVER_M_swing_ratio_nu0` | **1.41550 ± 0.16527** |
| `partF_livemem_redundancy_swing_OVER_M_swing_ratio_nu0.5` | 1.83967 ± 0.25919 |
| `partF_livemem_redundancy_swing_OVER_M_swing_ratio_nu1` | 2.22542 ± 0.35313 |
| `partF_livemem_redundancy_swing_OVER_M_swing_ratio_nu2` | 2.38153 ± 0.44063 |
| `partF_livemem_redundancy_swing_OVER_M_swing_ratio_nu4` | **2.50528 ± 0.46000** |
| `partF_nu_inflation_livemem_redundancy_swing_ratio_nu4_over_nu0` | **1.76779 ± 0.25171** |
| `partF_nu_inflation_livemem_redundancy_swing_flip_nu4_over_nu0` | **1.09159 ± 0.01743** |
| `partF_nu_inflation_livemem_M_swing_ratio_nu4_over_nu0` (the control) | **0.99849 ± 0.03338** |

**At `ν = 0` the redundancy swing on the ratio is 1.41550 ± 0.16527 times the `M` swing; at
`ν = 4` it is 2.50528 ± 0.46000 times it.** Turning on the term §3.4 built for redundancy
nearly **doubles** redundancy's lead over `M` as the variable that sets `b`'s decision
strength, while leaving the `M` axis itself at 0.99849 ± 0.03338 of where it was.

**Per-cell, `ν`'s effect is confined to cells that contain near-duplicate pairs** — which
is the same statement as §2's control, seen from the `b_max`/margin side:

| cell at `M = 40` | `b_max`/margin at `ν=4` ÷ at `ν=0` |
|---|---|
| iid | **1.00646 ± 0.03008** |
| broad_w0.4 (flat control) | **1.01050 ± 0.04313** |
| `ρ=0.99, f=0.25` | **3.03887 ± 0.36234** |
| `ρ=0.99, f=0.75` | 1.77749 ± 0.24394 |

Rows `partF_nu_inflation_bmax_over_margin_M40_*`. The `f = 0.25` cell is the worst because
it has the most **dispersion** in `max_cos`; `f = 0.75` is partly saturated. And the tie
creation that produces it: `partF_nu_inflation_frac_bottom2_neardup_M40_dup_rho0.99_f0.25_nu4_over_nu0
= 7.36156 ± 0.77031`.

**Cross-cycle check, read from the prior ledgers at runtime rather than retyped** (part G;
row `prior_cycle_rows_read_from_their_ledgers`). The `ν = 0` column **replicates cycle 6
exactly**, because the `ν = 0` `b`-off arm *is* the driver:

| this cycle | cycle 6, read from its ledger |
|---|---|
| `partB_bmax_over_margin_M40_iid_nu0 = 3.15777 ± 0.19603` | `partG_cycle6_livemem_M40_iid_ratio_read_from_cycle6 = 3.15777 ± 0.19603` |
| `partD_livemem_redundancy_swing_ratio_M40_..._nu0 = 2.37102 ± 0.21026` | `partG_cycle6_livemem_redundancy_swing_M40_read_from_cycle6 = 2.37102 ± 0.21026` |
| `partD_livemem_M_swing_flip_M40_over_M08_iid_nu0 = 1.19266 ± 0.05566` | `partG_cycle6_livemem_M_swing_flip_rho0_read_from_cycle6 = 1.19266 ± 0.05566` |

Identical in every digit, from a driver written independently of cycle 6's.

**The brief's `b_max/margin = 1.888 → 2.592 → 3.175` triple is cycle 5's ORDER-STATISTIC
triple**, located by reading cycle 5's ledger in this process:
`partG_briefs_triple_M08_read_from_cycle5 = 1.88772 ± 0.05003`,
`partG_briefs_triple_M16_read_from_cycle5 = 2.59245 ± 0.05256`,
`partG_briefs_triple_M40_read_from_cycle5 = 3.17490 ± 0.08201`. My independently written
order statistic gives 1.93564 ± 0.13662 / 2.70354 ± 0.19543 / 3.12195 ± 0.11036 and my live
memory 1.87880 ± 0.06975 / 2.51958 ± 0.11346 / 3.15777 ± 0.19603 — all consistent. **The
brief does not say which readout its triple is**, and the two differ by up to 7% at
`M = 16`; recorded as `partG_briefs_triple_provenance` so a later cycle does not mix them.

## 4 — Control: the state distribution is not doing the work

Part B scores every `ν` on states generated by a `ν = 0` driver, which is what makes the
counterfactual clean. Part C inverts it: at `M = 40` the `ν = 4` `b`-off arm **owns** its
own memory.

| | part B (`ν=0` driver owns memory) | part C (`ν=4` arm owns memory) |
|---|---|---|
| iid, `b_max`/margin | 3.17756 ± 0.20781 | 3.13115 ± 0.24882 |
| iid, frac bottom-2 cos ≥ 0.9 | 0.00000 | 0.00000 |
| `ρ=0.99,f=0.5`, `b_max`/margin | 12.01926 ± 0.56252 | 9.12480 ± 0.37869 |
| `ρ=0.99,f=0.5`, median bottom-2 cos | 0.99000 ± 0.00000 | 0.99000 ± 0.00000 |
| `ρ=0.99,f=0.5`, frac bottom-2 cos ≥ 0.9 | 0.78625 ± 0.03168 | 0.66208 ± 0.03672 |
| `ρ=0.99,f=0.5`, flip rate | 0.66708 ± 0.01512 | 0.55208 ± 0.02200 |

Rows `partC_selfdriven_*`. **The effect survives self-driving and is somewhat smaller**
(`b_max`/margin 9.12480 ± 0.37869 rather than 12.01926 ± 0.56252; the paired inflation over
part B's `ν = 0` value in the same cell is
`partF_partC_selfdriven_inflation_over_nu0_M40_dup_rho0.99_f0.5 = 1.67873 ± 0.18481`).
That is the expected direction: a `ν`-driven policy evicts duplicates, so it partly cleans
its own memory. It reduces the inflation; it does not remove it, and the median bottom-2
cosine is still pinned at `ρ`. All part C self-tests are 1.00000 ± 0.00000.

## 5 — `_max_cosine`'s dead-slot handling: **correct for live slots**, with two sign
behaviours that are not defects but are traps

Audited against `ref_max_cos`, a brute-force double loop written independently in the
harness that computes, for each **live** `i`, `max` over **live** `j ≠ i` of `cos(s_i,s_j)`
— i.e. what §3.4 intends.

| probe | row | result |
|---|---|---|
| 200 random partly-dead memories, `n ∈ [2,12]`, live rows vs the reference | `partX_X1_max_abs_dev_from_reference_on_live_slots` | **7.45e-08** (float32 eps) |
| an **exact duplicate of live slot 0 planted in a dead slot** — does the dead column leak? | `partX_X2_live_slot0_maxcos_with_exact_duplicate_in_a_DEAD_slot` | **0.03327**, not 1.0 |
| " | `partX_X2_dead_column_leaked` | **False** |
| dead rows' returned value | `partX_X2_dead_rows_returned` | `[0.0, 0.0, 0.0]` |
| NaN / 1e9 / zero garbage in dead rows | `partX_X3_live_rows_finite_with_nan_in_dead_rows` | **True** |
| " — do live rows change? | `partX_X3_live_rows_unchanged_by_dead_row_garbage` | **0.0**, bit-identical |
| does the term reach the score? | `partX_X6_attribution_nu0` / `_nu2` | `"psi"` → `"psi-nu"` |
| " | `partX_X6_score_diff_equals_minus_nu_maxcos` | **1.19e-07** |

**Verdict: correct.** `slots.live.unsqueeze(0)` broadcasts over rows and masks **columns**,
and `max(dim=1)` reduces over columns, so dead `j` are excluded; `fill_diagonal_(-1)`
excludes `j = i`; dead rows are returned as 0 and are overwritten with `+inf` by `_score`
anyway. Nothing leaks, including NaN.

**Two behaviours I report and do not fix**, both consequences of `-1.0` being used as the
"absent" fill in a quantity whose natural range includes negatives:

1. `partX_X4_maxcos_when_n_live_is_1 = -1.0`. A live slot with **no live neighbour** scores
   `−ν·(−1) = +ν` — a **bonus of `ν`**, not a zero penalty. Unreachable in eviction (the
   memory is full when a victim is chosen), but it is the value any future caller reading a
   "redundancy" statistic off this method would get.
2. `partX_X5_maxcos_for_antipodal_pair = -1.0`. Two live slots at `cos = −1` each score a
   **bonus of `+ν`**. §3.4 calls the term a redundancy *penalty*; as implemented it rewards
   anti-correlation exactly as strongly as it punishes duplication. At `d = 384` with unit
   gestalts this never bites — iid `max_cos` is 0.109 ± 0.0003 at `n = 40` — but it is a
   spec question, not a code bug, and it is the kind of thing that becomes reachable once
   gestalts are trained and anisotropic.

**One interface observation outside `_max_cosine` itself**, recorded as row
`nu_term_is_not_z_scored`: `_score` z-scores `ψ̂` and adds `b` **on the z scale**, then
subtracts `ν·max_cos` **raw**. So `b_max` is in units of SD(`ψ̂`) — §3.5 item 1's whole
point — while `ν` is in units of cosine. The two are **not commensurable by construction**,
and the conversion factor between them is the very gestalt geometry `ν` is supposed to
neutralize. That is the structural reason a single frozen `ν` cannot transfer across
corpora, and it is upstream of everything in §1–§3.

## 6 — Re-execution

**The tie-creation result, self-contained, ~1 s.** This text is embedded verbatim in the
harness as part E2 and executed there, so what it prints **is** twenty ledger rows
(`partE2_reexec_*`).

```bash
cd /Users/keanooo7/retrieval-successor-retention && .venv/bin/python -c "
import sys, math, torch; sys.path.insert(0,'src')
from rsr import constants as C
from rsr.retention.value_head import BilinearValueHead
b_max = C.get('b_max')                      # FROZEN registry read
n = C.get('M','pg19_e3')                    # 40, registry read
d, G, RHO, FRAC = 384, 800, 0.99, 0.25
NUS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]       # LABELLED BYPASS: nu is MEASURED (E1)
def unit(x): return x/x.norm(dim=-1,keepdim=True)
def cell(seed):
    gen=torch.Generator().manual_seed(4000+seed); h=BilinearValueHead(d,generator=gen)
    nd=max(1,int(round(FRAC*n))); nb=n-nd
    marg={nu:[] for nu in NUS}; near={nu:[] for nu in NUS}; gap={nu:[] for nu in NUS}; pen=[]
    with torch.no_grad():
        for _ in range(G//100):
            g=unit(torch.randn(100,n,d,generator=gen))
            src=torch.randint(nb,(100,nd),generator=gen)
            gs=torch.gather(g[:,:nb,:],1,src.unsqueeze(-1).expand(-1,-1,d)).clone()
            e=unit(torch.randn(100,nd,d,generator=gen))
            e=unit(e-(e*gs).sum(-1,keepdim=True)*gs)
            g[:,nb:,:]=RHO*gs+math.sqrt(1-RHO*RHO)*e          # cos(dup,src) = RHO exactly
            v=h(g,unit(torch.randn(100,d,generator=gen))).double()
            z=(v-v.mean(1,keepdim=True))/v.std(1,unbiased=False).unsqueeze(1)   # rsr.py _score
            gn=torch.nn.functional.normalize(g,dim=-1); gram=(gn@gn.transpose(1,2)).double()
            mc=gram.clone(); mc.diagonal(dim1=1,dim2=2).fill_(-1.); mc=mc.max(2).values  # _max_cosine
            di=torch.arange(nb,n); ar=torch.arange(100)
            pen += (mc.gather(1,di.unsqueeze(0).expand(100,-1))-mc.gather(1,src)).abs().flatten().tolist()
            for nu in NUS:
                sc=z-nu*mc; o=sc.argsort(1); i0,i1=o[:,0],o[:,1]
                marg[nu]+= (sc[ar,i1]-sc[ar,i0]).tolist()
                near[nu]+= (gram[ar,i0,i1]>=0.9).double().tolist()
                gap[nu] += (sc.gather(1,di.unsqueeze(0).expand(100,-1))-sc.gather(1,src)).abs().flatten().tolist()
    r={}
    for nu in NUS:
        m=sorted(marg[nu])
        r[nu]=(b_max/m[len(m)//2], sum(near[nu])/len(near[nu]), sum(gap[nu])/len(gap[nu]))
    return r, sum(pen)/len(pen)
def ms(xs):
    m=sum(xs)/len(xs); return m, math.sqrt(sum((x-m)**2 for x in xs)/(len(xs)-1))
A={s:cell(s) for s in range(3)}
print('within-pair |max-cos penalty difference|  : %.3e +/- %.3e  (exactly 0 up to float32)'
      % ms([A[s][1] for s in range(3)]))
print('nu      b_max/margin        frac bottom-2 is a near-dup pair   within-pair |score gap|')
for nu in NUS:
    print('%-7g %.4f +/- %.4f   %.4f +/- %.4f                  %.5f +/- %.5f'
          % (nu, *ms([A[s][0][nu][0] for s in range(3)]),
                 *ms([A[s][0][nu][1] for s in range(3)]),
                 *ms([A[s][0][nu][2] for s in range(3)])))
"
```

Prints, bit-exactly (rows `partE2_reexec_within_pair_abs_penalty_diff`,
`partE2_reexec_bmax_over_margin_nu*`, `partE2_reexec_frac_bottom2_neardup_pair_nu*`,
`partE2_reexec_within_pair_abs_score_gap_nu*`):

```
within-pair |max-cos penalty difference|  : 4.102e-08 +/- 4.342e-10  (exactly 0 up to float32)
nu      b_max/margin        frac bottom-2 is a near-dup pair   within-pair |score gap|
0       5.1834 +/- 0.5373   0.2513 +/- 0.0088                  0.11729 +/- 0.00180
0.25    6.2121 +/- 0.5483   0.3438 +/- 0.0187                  0.11729 +/- 0.00180
0.5     7.3474 +/- 0.4341   0.4292 +/- 0.0201                  0.11729 +/- 0.00180
1       9.8547 +/- 0.4636   0.6000 +/- 0.0087                  0.11729 +/- 0.00180
2       11.8776 +/- 0.2153   0.7933 +/- 0.0171                  0.11729 +/- 0.00180
4       12.2316 +/- 0.3355   0.8408 +/- 0.0092                  0.11729 +/- 0.00180
```

**Three columns, one conclusion.** `b`'s over-reach grows
`partF_E2_bmax_over_margin_inflation_nu4_over_nu0 = 2.37332 ± 0.19867`; the fraction of
decisions that are a coin-flip between two near-copies grows
`partF_E2_frac_bottom2_neardup_inflation_nu4_over_nu0 = 3.34870 ± 0.09351`; and the thing
`ν` was supposed to change — the score gap *inside* the pair — does not move in the fifth decimal, at any `ν`,
because the penalty it subtracts is the same number for both members.

The second, saturated-cell snippet is also embedded and runs in ~1.4 s (rows
`partE_reexec_*`), printing the `ν`-flat order statistic from §1.

Full experiment, 65.0 s, rewrites this directory's `ledger.json`:

```bash
cd /Users/keanooo7/retrieval-successor-retention && .venv/bin/python \
  /private/tmp/claude-501/-Users-keanooo7-second-brain/1bdd6660-ee4c-49c5-99a7-28bf139ef2c9/scratchpad/run_nu_redundancy.py
```

## 7 — What this changes for the registry question

Cycle 6 tabled four options: (A) per-scope `b_max` keyed on `M`, (B) an `M` factor in
`γ_b`, (C) `b_max` normalized by a *measured* margin, (D) `b_max` should not be frozen.
This cycle removes the hope that `ν` makes (A)/(B) sufficient, and it does so structurally
rather than numerically: **`ν` is rank-preserving within exactly the pairs that set the
margin** (§2), so no value of it can flatten that axis. It also adds a cost to `ν` itself
that §3.4 does not price: a positive `ν` **manufactures** low-margin decisions in a
redundant memory, which is the regime where `b` is most over-powered. **Nothing here picks
an option, and nothing here is a recommendation about `ν` — that is E1's.**

## 8 — The `sd = 0.0000` rows

237 rows, all classified by key in `sd_zero_rows_reasons` by a classifier in the same
process; **`sd_zero_rows_unclassified` is `[]`**. Eight categories, no others:

| count | reason |
|---|---|
| 79 | **every seed measured exactly 0.0; the event never occurred in any draw.** Overwhelmingly the `frac_bottom2_cos_ge_0.9` rows on the **iid** and **broad** constructions — at `d = 384` those generators never produce a pairwise cosine at the threshold. **This is the control that makes §2 a finding rather than an artifact.** |
| 77 | **self-tests at their required value 1.0** — the hand-recomputed `argmin[z(ψ̂)+b−ν·max_cos]` equals `RSRPolicy`'s victim on every eviction in every seed. Below 1.0 would invalidate every flip count. |
| 24 | `partA_frac_bottom2_is_labelled_pair_*` on the **topic** construction: it plants no labelled pairs, so the quantity is 0.0 by construction. |
| 18 | `..._nu0` compared with itself (`frac_victim_changed_vs_nu0` at `ν = 0`): identically 0 by construction. |
| 18 | `median_bottom2_cos` rows whose distribution has a **near-atom at the constructed pair cosine `ρ`**: the median falls inside that atom, so it is identical across seeds by construction of the generator, not by a dead seed. (These are the `0.99000 ± 0.00000` cells in §2 — the finding itself.) |
| 15 | the `ν = 0` `b`-off arm **is** the driver, so `frac_victim_changed_vs_nu0_driver` can never differ from 0. |
| 3 | structural counts with no RNG: `partB_n_evictions_per_seed_per_arm_M{08,16,40}` = 864 / 768 / 480, all `12 × (80 − M)`. |
| 3 | `..._RELATIVE_to_nu0` and `tie_creation_..._over_nu0` at `ν = 0`: identically 1.0. |

**The `pair_abs_penalty_diff = 0.00000 ± 0.00000` rows are *not* in this list**, because
their sd is not exactly zero — the raw samples are ~4.1e-08 with sd ~4.3e-10. They round to
zero at five decimals and are reported that way, with the raw samples quoted in §2.

**The seed does reach the RNG.** Every margin, ratio, flip-rate, cosine, swing and
saturation row has `sd > 0`, and the `ν = 0` column reproduces cycle 6's live-memory
numbers in every digit from an independently written driver.

## 9 — What this does NOT establish

- **Nothing about PG-19, and nothing about any real corpus.** There are no PG-19 gestalts
  in this repo, none were used, and `φ` was never trained on text. Every number is one
  synthetic generator with hand-planted duplicate pairs at an exactly controlled cosine.
  **Whether real gestalts contain near-duplicate pairs at all, at what rate, and at what
  cosine, is not measured and is not measurable from anything in this repo.** *As conjecture
  this experiment does not establish*: if a real memory's `max_cos` distribution is
  **unimodal and tight** — every slot equally (un)redundant — then `ν` is a common offset
  and §1's order-statistic column is the right picture, i.e. `ν` does nothing either way.
  If it has a **heavy upper tail with a spread** — some pairs near-tied, most not — then
  §2's live-memory column is, and `ν` costs margin. **`ν`'s entire effect is a function of
  the dispersion of `max_cos`, not its level, and nothing here measures that on real text.**
  §13 forbids reading a synthetic number as a corpus number; this bullet is the boundary.
- **This does not measure `ν`.** Not one value here is a measurement of that constant; E1
  owns it. The registry refused the read in this process and the refusal is a ledger row.
  Likewise every `γ_b`, `τ`, `b_max`-as-reach and `E[lifetime]` is a labelled bypass.
- **"Created a tie" is not "made it worse."** No language model ran, no loss was computed,
  no `r_i` was collected, no LOO Δloss. A tie between two slots at cosine 0.99 may be a
  choice between two *near-equivalent* memories, in which case losing the coin flip costs
  little — which is §3.4's own defence and is exactly the thing D-3 says is **not** true
  under a submodular set value. **Neither direction was quantified here**, and it is the
  single largest gap in this cycle.
- **`φ` is untrained.** `ψ̂` is a linear functional of `s`, so its across-slot law is
  near-Gaussian for spherical gestalts whatever `W, u` are, and cycles 4–5 showed the
  sd-to-margin relation is invariant to `φ`'s scale. What training changes is the gestalt
  distribution — this cycle's axis, explored synthetically. **Two things training could
  change that this cycle cannot see:** a trained `ψ̂` might assign *different* values to
  two near-parallel gestalts (shrinking the within-pair `z` gap further, making the tie
  worse) or systematically similar ones (no change); and training changes `max_cos`'s
  dispersion, which §9 bullet 1 identifies as the whole lever.
- **`ū` is synthetic.** Every flip rate is conditional on a log-normal salience softmax and
  `τ = 0.25`. The order statistic and every margin, cosine and tie-creation row do not
  involve `ū` at all. A real utilisation signal could move the flip rates either way; it
  cannot move the margin distribution, where the effect lives.
- **Part B scores `ν` on `ν = 0` states.** Part C shows the effect survives self-driving at
  `M = 40` and is ~24% smaller there (§4). A **fully** self-driven grid across all `M` and
  all `ν` was not run.
- **`S = 80` for all three `M`**, supplied not read at `M = 8` (`get("S","e7")` raises).
- **One head form, one `d`.** `d_model = 384`, `BilinearValueHead` at init. No `d` sweep —
  and `max_cos` for iid vectors is a function of `d`, so the iid baseline of 0.109 is a
  `d = 384` number.
- **One duplicate topology.** Every planted pair is a *pair*. Triples, chains and
  many-to-one duplication would give the source a different `max_cos` from each copy and
  could break the exact-equality result in §2. Not tested.
- **CPU only**, one thread. No MPS, no device claim.
- **Cycle-1 walls still stand**: no constant was measured, nothing was committed, `src/`
  `tests/` `docs/` untouched. `_max_cosine` was audited and **not modified**.
- **The known defects are unchanged and untouched**: `ProtectionBias`'s declared surface
  still disagrees with its call site in `_score`; `attribution` and `n_live` are still
  config-echo fields; `run_policy_loop` still never trains `ψ̂`; `observe()` still raises.

## 10 — The single next falsifier I would name

> **"A tie that `ν` creates is cheap: when the two lowest-scoring slots are near-copies at
> cosine ≥ 0.9, which one is evicted does not change the loss — so `b`'s inflated
> `b_max`/margin over those decisions is authority over a choice that does not matter."**

It is the right next one because it is the **only** remaining defence of §3.4's term after
this cycle, and because it is the assumption every "changed the victim" number in cycles
2–7 has quietly leaned on. If the tie really is cheap, then §2's 3.35× rise in coin-flip
decisions is a *feature* — `ν` is routing `b`'s over-reach onto harmless decisions — and
the registry question shrinks to the non-redundant slots. If it is not cheap, then `ν`
converts `b` from a mis-calibrated constant into a mis-calibrated constant **operating
precisely where the policy is most uncertain**, and D-3's submodularity argument says it is
not cheap: two slots carrying the same proposition are individually redundant but jointly
necessary, so evicting *either* is fine while evicting the *wrong* one relative to what the
reader will need next is not.

It is **reachable, but not on this harness**, and that is the honest part. It needs a real
`r_i` — an `AttentionTrace` and a leave-one-out Δloss (§3.2.1's truth rule) — which means
the PyTorch TG transcription (gauntlet 2.4) and `observe()`, both of which currently raise.
The **cheapest reachable proxy**, and what I would actually run next, is a synthetic
retrieval task with a known ground-truth demand signal: plant a near-duplicate pair where
exactly **one** member is the one a later query needs, and measure whether the RSR argmin
under `ν > 0` picks the needed one above chance. That is one cycle on a generator this
harness already has, and it distinguishes "cheap tie" from "coin flip on a real decision"
without a language model. It would also be the **first** cycle in this series in which a
victim choice has a right answer.
