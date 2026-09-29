# E0d Amendment 2: final draft, for independent adversarial review

> **Status: DRAFT. NOT COMMITTED. NOT REVIEWED.** Nothing here has been written to
> `experiments/e0d/PREREG.md`. An independent adversarial review comes first. Part A below is the
> text that would be appended, verbatim, after A1.10. Parts B–E sit outside the PREREG: the runner
> test plan, the open issues for Brendan, the fix map, and a proposed spec-correction note that
> Brendan writes himself if he wants it.

**Written 2026-09-27 by a model** (Claude Opus 5.5, RSR researcher subagent, item E0D-AMD2).
- **Governing design:** `~/Downloads/E0d_RoundTable_Synthesis.md`, Rank 1 (§4) with the Rank 2
  secondaries (§5) and the §6A calibration.
- **Rulings:** `~/Documents/RSR-2026-09-27-plan/owner-drafts/R-2026-09-27-e0d-statistic.md`,
  quoted verbatim where used.
- **Base:** PREREG at `run/e0d` `268b947` (base, Amendment 1 and the A1.10 erratum); runner at
  `4918f2d`.

**Data hygiene.**
- Every number here comes from set E, docs `[64, 128)`, seed 0, arm B ckpt3000. That set was
  already inspected.
- The only data read is the saved LOO cells: `reviews/e0d-a2/step1_cells.npz`, sha256
  `1338de4bdc5fbf9d33c8bd37abb34532535d2cd1e77f18e4deb87c357a65f15c` (recomputed for this draft),
  and its replicate file `step1b_reps.npz`.
- One script (`m3_oi.py`) regenerated the *structure* of docs `[64, 128)` seed 0 from the generator,
  with `assert HI <= 262144`.
- **No model was loaded, no forward was run, and no `r_i` of any kind was computed.** Every `r_i`
  below is synthetic.
- **No document of `[262144, 263168)` was generated, read or touched.**
- Scripts and logs are in `reviews/e0d-a2-final/` (`m1`–`m4`). Every rc was 0, captured as
  `cmd > log 2>&1; rc=$?`. `m1`–`m3c` ran through `orchestrator.slot run --lane cpu-det`.
  `m4_tau.py` is a 20-line numpy quantile script and ran bare.

---

## Part A: the amendment text (append-only)

```
## Amendment 2 (2026-09-27, pre-data)

**Written 2026-09-27 by a model** (Claude Opus 5.5, RSR researcher subagent, item E0D-AMD2).
**Append-only.** Nothing above this heading has been edited. Where this section and the text
above disagree, **this section governs**, including over Amendment 1 and over the front matter's
`thresholds.RHO_STAR`, `requires` and `prediction` fields.
```

### A2.0 Authority, pre-data status, and what does not change

- **Authority.**
  - `R-2026-09-27-e0d-statistic` (Brendan). It adopts the E0d round-table synthesis, Rank 1, and
    rules:
    - `AUROC_STAR: 0.85`;
    - `HARM_RATIO_STAR: 0.5` ("Pass if H ≤ 0.5 × H_age-random, per seed, through the CI");
    - "Each seed separately. E0d passes only if every seed passes. Pooled results are reported
      and never gate.";
    - primary `r_i` gated; primary knockout resample.
  - Its scope clause says: "The τ rule and the age bins are fixed in Amendment 2, before any
    data." This amendment does that.
- **LOO remains the truth.** Spec §3.2.1 is unchanged. This amendment changes the *comparison
  statistic* (lever (a)), not the ground truth (lever (b)). It departs from §3.2.1's named
  statistic ("Report Spearman ρ"). That departure is declared in A2.12.
- **Pre-data.**
  - At this commit, no document of `D_E0d = [262144, 263168)` has been generated, run or
    inspected.
  - No E0d measurement exists.
  - No arm-B `r_i` has been computed on any document for the purpose of this amendment.
  - The constants frozen below were computed on set E `[64, 128)`, seed 0: already-inspected data
    that is disjoint from `D_E0d`. They were computed from LOO Δ only.
- **Unchanged:**
  - §1 (substrate) and §2 (documents, disjointness, closure);
  - A1.1's strata (Q-steps, A-cells, N-steps);
  - A1.1's LOO-flat rule and §5's `R_FLAT`;
  - A1.3 (bos-copy exclusions; the exit follows the excluded population);
  - A1.4 (any exception exits 3; exit 1 only after `e0d.class` is written);
  - A1.5–A1.7;
  - controls C1–C7 and C9;
  - the cluster bootstrap (per-document, 2000 resamples, generator seed `20260927`, 95%
    percentile, the same indices for every statistic of a seed).

### A2.1 Retired from gating (kept as reported secondaries)

- **`RHO_STAR` and every ρ-based decision rule are retired from gating.** This covers A1.1's ρ_Q
  and ρ_Q,rank, §4 and A1.8, the ρ\* sensitivity values {0.3, 0.7}, and A1's §7 table.
  - All of them are still computed and reported, through the same code and bootstrap.
  - The A1 per-seed label (the A1 §7 table at ρ\* = 0.5) is also reported, under the name
    `label_A1`. It changes no exit.
- **C10 as a ρ rule is retired from gating.** C10's binary positive control (`true_demand`, 1 on
  the A-cell) is still computed on every ρ statistic and reported, with its CEILING reading.
  - The same control is re-used as the positive control of the new statistics (A2.8, row 0).
    There, it can only turn a failure into `CEILING` (exit 2). It can never produce a pass or a
    kill.
- **ρ_pool, ρ_step and bottom-1 agreement** remain secondaries, as in A1.1.

### A2.2 The critical label and τ (round table 4.2(d))

- **Critical cell.** `y(t, i) = 1[Δ_resample(t, i) > τ]`.
  - Δ is `loo_delta_loss(...)["delta"]`, as in §3: the change in sentence `t`'s mean real-target
    NLL when slot `i` alone is knocked out.
  - It is **not** answer-token NLL. The round table's §4.1 line "answer-token NLL" does not
    describe `loo.py`, and τ below was computed on the sentence-mean Δ.
  - A cell whose Δ_resample is NaN (no resample donor) has no label. The treatment of those cells
    is in A2.4 and A2.6.
- **The rule.**
  - τ = the q = 0.995 quantile (numpy's default linear, type-7 interpolation) of |Δ_resample|;
  - the population is full-memory Q-step cells that are not A-cells and have a value;
  - it is computed once, on the already-inspected sample, and frozen.
  - The measured fact that selects this branch of 4.2(d): the off-A Δ is continuous, with **0
    exact zeros in 31,846** full-memory off-A cells (`reviews/e0d-a2/step1_out.json`). So
    τ = 0⁺ does not apply.
- **The sample.**
  - `reviews/e0d-a2/step1_cells.npz`, sha256
    `1338de4bdc5fbf9d33c8bd37abb34532535d2cd1e77f18e4deb87c357a65f15c`.
  - Set E docs `[64, 128)`, seed 0, arm B ckpt3000 (sha256 `0ee3f8a6…a118da60`), resample donor
    seed 0.
  - n = 13,376 values.
- **Frozen values.**

  | constant | value | role |
  |---|---|---|
  | `TAU_RESAMPLE` | **0.4157434984576128** nats/sentence (q = 0.995) | **primary** |
  | `TAU_ZERO` | 0.3447678400575726 (the same rule on \|Δ_zero\|, n = 13,481) | the zero-knockout secondary's label |
  | sensitivity, resample | q = 0.99 → 0.3109549731015192; q = 0.999 → 0.7129988655444408 | reported, never gating |
  | sensitivity, zero | q = 0.99 → 0.24913927197480037; q = 0.999 → 0.5874972405433865 | reported, never gating |

- **τ is a constant, not a rule re-run on `D_E0d`.** The runner reads these values. It never
  recomputes a quantile on `D_E0d`.
- **Where τ lands on the sample.**
  - A document-cluster bootstrap of τ (2000 resamples) gives a 95% interval of [0.378, 0.472].
    The two sensitivity values (0.311, 0.713) bracket it.
  - At τ = 0.4157: 683 critical cells among the 14,028 Q-cells that have a value (0.0487).
    - Of these, 631 are A-cells, out of 652 A-cells that have a value.
    - The other 52 are not A-cells.
  - Across τ's interval, the count moves from 701 (A 636) to 665 (A 625). τ mostly moves the
    non-A tail.
- **Limitation (declared).** τ comes from **seed 0 and 64 documents only**, and it is applied to
  seeds 1 and 2 unchanged. Whether a per-seed τ, or a larger sample, is warranted is open issue
  OI-4 (Part C). This amendment registers the single value.

### A2.3 Age strata (round table 4.2(a), 7.5)

- **Age.** `a(t, i) = t − slot_sentence(t, i)`, the number of steps since the slot's sentence was
  read.
- **Bins.** 16 bins, one per age: `B = {{1}, {2}, …, {16}}`. They are fixed here, before data,
  and never chosen from Δ or `r_i`.
- **Measured structure** (set E sample, all 2,048 full-memory steps):
  - every full-memory step holds **exactly one slot of each age 1–16**;
  - write-order rank is `i = M − a` on every full-memory cell.
  - The runner asserts both, per seed, on `D_E0d`, and exits 3 on violation (a new control,
    C11).
- **Consequence, stated explicitly: `AUROC_strat` is a cross-step statistic within age.**
  - With one slot per age per step, no pair of cells is both within-step and within-age.
  - Every comparison inside `AUROC_a` is between cells of the same age at *different* steps.
  - The within-step question is carried by H (A2.6) and by the within-step companion (A2.5).
- **Under A1.3 (the operative population).** Rank M−1 is excluded, so bin {1} is empty. That
  bin is logged as excluded, and `AUROC_strat` runs over bins 2–16.

### A2.4 The primary statistic `AUROC_strat`

- **Population.** Q-step cells (A1.1: full memory, sentence `t` a query), in the A1.3-excluded
  population, with a Δ_resample value.
- **Score.** Gated `r_i` (A2.9), used as emitted.
  - At full memory it is a share: Σ_i r_i(t) = 1 over the 16 live slots.
  - It is **not** re-normalised after the A1.3 exclusion.
  - `AUROC_strat` compares values across steps, so it is not invariant to per-step rescaling
    (Part B, test T4).
- **Definition.** For each bin `a` with at least one positive (y = 1) and one negative (y = 0),
  compute:
  - `AUROC_a` = the Mann–Whitney probability that a positive's `r` exceeds a negative's `r`;
    ties count ½.
  - Then `AUROC_strat = Σ_a w_a · AUROC_a / Σ_a w_a`, with `w_a` = the number of positives in
    bin `a`.
  - A bin without both classes is **excluded and logged** (bin, positives, negatives). It never
    counts as 0 or ½.
- **Bootstrap.** Each cell carries its document's bootstrap count. `AUROC_a` and `w_a` are
  computed on the weighted sample: the resample's statistic, without materialising it.
- **Undefined.** If no bin has both classes, `AUROC_strat` has no value. That is
  `MeasurementUndefined`, exit 3 (the runner's existing rule), unless the LOO-flat rule already
  labelled the seed.
- **Null and reference values.**
  - **Null: `r` shuffled within step**, across the step's eligible slots. Expected 0.5.
  - The round table's §4.3 null ("shuffle r across cells within (step, age_bin)") is the
    identity here, because each (step, age) holds one cell. It is replaced.
  - An age-only score gives exactly 0.5, since it is tied within every bin.
  - Set E sample (bos-excluded; synthetic scores on real Δ):
    - `true_demand` positive control: 0.9585;
    - LOO replicate (donor seed 1) as score: 0.9885;
    - age-only and recency: 0.5000 exactly;
    - within-step shuffle: 0.495.

### A2.5 The within-step companion (round table 4.2(a); reported secondary)

- **Per-cell percentile.** At each Q-step `t`, over its eligible cells (the A1.3 population, with
  a value), `pct(t, i) = (midrank of r_i(t) − 1) / (n_t − 1)`, in [0, 1].
- **Age means.** `m_a` = the mean of `pct` over **all** Q-step cells of age `a` (critical or not).
- **Per-step value.** For each Q-step with at least one critical cell, `v_t` = the mean over its
  critical cells of `pct(t, i) − m_{a(t,i)}`.
- **`C_ws`** = the mean of `v_t` over those steps. It uses the same bootstrap; `m_a` is
  re-estimated in each resample.
- **Reference values.**
  - Null (within-step shuffle): 0.
  - Age-only score: exactly 0.
  - Set E sample: `true_demand` 0.447; LOO replicate 0.440.
- **Status.** Reported, never gating. Whether it should be required to agree in direction with
  the primary is open issue OI-2.

### A2.6 H and its baselines (round table 4.2(b))

- **`Q_crit`.**
  - These are the Q-steps (A1.1) with at least one critical cell (`k_t ≥ 1`) among their
    eligible slots, where every eligible slot has a Δ_resample value.
  - The round table calls this set "Q". It is renamed here because A1.1's "Q-steps" is the
    larger set.
  - Steps dropped for a missing value are counted and reported.
  - Set E sample (bos-excluded):
    - 720 Q-steps;
    - 498 of them with `k_t ≥ 1`;
    - 443 of those complete.
- **Eligible slots.**
  - In the A1.3 population (operative), `n_t = M − 1 = 15`: ranks 0 to M−2.
  - In the all-cells population, `n_t = M`.
- **H** = the mean over `Q_crit` of `h_t`.
  - `h_t` = the fraction of the step's tied minima of `r` (over eligible slots) that are
    critical. It is 1 or 0 when the argmin is unique.
  - The ruling calls H the "eviction-harm rate". Open issue OI-1 disputes that name. The symbol
    and the ruled bar stand.
- **`H_random`** = the mean over `Q_crit` of `k_t / n_t`: uniform random eviction over the
  eligible slots, on the same steps.
  - *This departs from the PM's "k_t / M"* in the operative population, where `n_t = 15`. On the
    set E sample, k_t/M gives 0.0649 and k_t/n_t gives 0.0694.
- **`H_age-random`** = random eviction matched to the age distribution of `r_i`'s own evictions.
  - `π(a)` = the mean over `Q_crit` of the fraction of the step's tied minima at age `a`.
  - `H_age-random` = the mean over `Q_crit` of `Σ_a π(a) · y(t, a)`.
  - Both `π` and `H_age-random` are re-estimated in each bootstrap resample.
  - It is the age-only null. If `r_i`'s argmin is independent of `y` within age,
    `E[H] = H_age-random`.
  - An age-only score gives `H = H_age-random` exactly (Part B, T3).
- **The ratio.** `R_H = H / H_age-random`, computed within each resample.
  - If `H_age-random = 0` in the point estimate or in any resample, `R_H` has no CI (the runner's
    `percentile_ci` NaN rule). The seed's harm reading is then `HARM_UNDEFINED` (A2.8).
  - This happens only when `r_i` evicts only at ages that are never critical, which is harmless
    but entirely attributable to age.
- **Age-uncontrolled analogue** (A2.8 row 6 only): `R_H,uniform = H / H_random`.
- **Set E sample (bos-excluded; synthetic scores):**

  | score | H | H_age-random | R_H |
  |---|---|---|---|
  | `true_demand` | 0.0059 | 0.0653 | 0.091 |
  | LOO replicate | 0.0023 | 0.0832 | 0.027 |
  | age-only | 0.0045 | 0.0045 | 1.000 exactly |
  | recency | 0.0384 | 0.0384 | 1.000 exactly |
  | within-step shuffle | 0.0497 | 0.0633 | 0.78 |

  `H_random` = 0.0694 for every score.

### A2.7 Reported secondaries (never gating)

Each of the following is reported per seed. The pooled-over-seeds value is reported too, never
gates, and uses a bootstrap stratified by seed.

1. **The full 2×2 grid**, with the primary cell marked:
   - gated or ungated `r_i`;
   - resample (`TAU_RESAMPLE`) or zero (`TAU_ZERO`) knockout;
   - every A2.4–A2.6 statistic and the A2.8 label.
2. **The all-cells population**, without A1.3's exclusions. `BOS_SENSITIVE` is as in A1.3.
3. **τ sensitivity:** every label recomputed at q = 0.99 and q = 0.999.
4. **Per-bin detail.** Each `AUROC_a` with its positive and negative counts, and the excluded
   bins.
5. **Other age and eviction baselines:**
   - `AUROC_unstrat`, the same statistic without strata;
   - `R_H,uniform`;
   - `H_FIFO`, the H of evicting the oldest slot;
   - `H_age-oracle` = min over `a` of the mean over `Q_crit` of `y(t, a)`: the best fixed-age
     rule, an oracle bound on any age-only policy.
6. **Every A1 ρ statistic** (A2.1), C10's ρ ceiling, and `label_A1`.
7. **Round table §5.1 (retained-set recall):** not reported. E0d evicts at most one slot per
   step, so it equals 1 − H restricted to `Q_crit` (the round table's own condition).
8. **Round table §5.2 (closed-loop replay):** **deferred, not run in E0d.** Replaying with
   `r_i`-, random-, age- or LOO-driven eviction is non-FIFO eviction, which C9 / PLAN-v4 T3
   forbid in this run. It is proposed as a separate item (Part C, OI-1).
9. **Round table §5.3 (sink/inversion probe):**
   - the `r_i` share by sentence kind (assert, query, filler; pending and spent asserts
     separately) at Q-steps, per seed;
   - the round table's §4.5 seed-1 diagnostic. Its text ("the fact slot outranks its own question
     slot" at answer steps) cannot be computed as written: at a Q-step the question is sentence
     `t` itself, not a slot. It is registered as: at full-memory steps where a fact's assert and
     its already-read query are both resident, the fraction with `r_assert > r_query`.
10. **Round table §5.4:** R-EVICT's system-specific numbers are cited nowhere in the E0d report
    as evidence.

### A2.8 Per-seed labels, classification, exits (round table 4.2(c), 4.2(f))

**Thresholds** (ruled, read from the C8 ruling, never typed):
- `A* = AUROC_STAR`;
- `ρ_H* = HARM_RATIO_STAR`.

Every reading is through the seed's 95% CI, never a point estimate.

**Per-seed label.** The primary cell is gated `r_i` × resample, in the A1.3-excluded population.
The first matching row wins. This table replaces A1's §7 table.

| # | condition | label |
|---|---|---|
| 0 | C10′: the `true_demand` control's `AUROC_strat` CI upper < A\*, **or** its `R_H` CI lower > ρ_H\*, **or** its `R_H` CI is undefined | `CEILING` |
| 1 | LOO flat (A1.1) and `r_i` flat (§5) | `DEGENERATE` |
| 2 | LOO flat, `r_i` not flat | `LOO_UNINFORMATIVE` |
| 3 | `AUROC_strat` CI upper < 0.5 | `INVERTED` |
| 4 | `AUROC_strat` CI lower ≥ A\* **and** `R_H` CI upper ≤ ρ_H\* | `AGREE` |
| 5 | a primary bar is decisively failed (`AUROC_strat` CI upper < A\*, or `R_H` CI lower > ρ_H\*), **and** both age-uncontrolled analogues pass decisively (`AUROC_unstrat` CI lower ≥ A\* and `R_H,uniform` CI upper ≤ ρ_H\*) | `AGREE_VIA_RANK` |
| 6 | `AUROC_strat` CI upper < A\*, **or** `R_H` CI lower > ρ_H\* | `DISAGREE` |
| 7 | `R_H` CI undefined | `HARM_UNDEFINED` |
| 8 | otherwise (a CI straddles a bar) | `UNRESOLVED` |

- **Row 0.** On the set E sample, the control reads `AUROC_strat` 0.9585 (sd 0.0015 at n = 1024)
  and `R_H` 0.091 (sd 0.003). So row 0 is expected not to fire. It exists so that a kill is only
  ever read where a perfect-label score could have passed.
- **Row 5 keeps A1.2 in force.** It is agreement with LOO that disappears once age is controlled.
  A1.2 ruled that this is not the §3.2.1 kill.
  - The label keeps A1's name, since write-order rank is age under FIFO. Its class keeps A1's
    name, `RECENCY_ONLY`, exit 2.
  - Row 5 can never produce a pass.
- **Row 6: either bar failed decisively is `DISAGREE`.** The ruling makes both bars primary, and
  a seed that decisively fails one cannot pass at any sample size. The ledger records which bar
  failed (`AUROC`, `HARM`, or both).
- **Row 3: `INVERTED` reads the AUROC only.**

**§8 classification over the 3 seeds** (first match wins). Rows 0–6 are A1's, restated.
`CEILING` and `HARM_UNDEFINED` count as `UNRESOLVED`.

| # | condition | class (`e0d.class`) | exit |
|---|---|---|---|
| 0 | any control C1–C11 fails, any measurement raises, any checkpoint missing | `inconclusive` | **3** |
| 1 | ≥ 2 seeds `DEGENERATE` or `LOO_UNINFORMATIVE` | `DEGENERATE_UNINFORMATIVE` | **2** |
| 2 | **all 3 seeds `AGREE`** | `AGREE` | **0** |
| 3 | all 3 seeds `INVERTED` | `CONFOUND_INVERTED` | **1** |
| 4 | all 3 seeds in {`DISAGREE`, `INVERTED`} | `CONFOUND` | **1** |
| 5 | all 3 seeds in {`DISAGREE`, `INVERTED`, `AGREE_VIA_RANK`}, at least one `AGREE_VIA_RANK` | `RECENCY_ONLY` | **2** |
| 6 | otherwise | `MIXED_UNRESOLVED` | **2** |

- **Seed rule (the ruling).** A pass (exit 0) needs every seed `AGREE`. Pooled results never
  gate.
- **What the rule does not change.**
  - A kill (exit 1) still needs every seed in {`DISAGREE`, `INVERTED`}, as in A1.
  - Two seeds `AGREE` and one `DISAGREE` is `MIXED_UNRESOLVED`, exit 2.
- **Unchanged reporting:**
  - `GATE_SENSITIVE` (the ungated class differs; the exit follows gated);
  - the zero knockout never changes the exit;
  - `BOS_SENSITIVE` (the exit follows the excluded population);
  - A1's reporting requirements for the flat conditions.
- **Exit discipline.** A1.4 unchanged: exit 1 only after `e0d.class ∈ {CONFOUND,
  CONFOUND_INVERTED}` is written, and the verifier checks that rc and class agree.

### A2.9 Primary cell (round table 4.2(e))

- **Gated `r_i` × resample LOO**, per the ruling ("Primary `r_i`: Gated … Primary knockout:
  Resample").
- The other three cells are secondaries (A2.7 item 1).

### A2.10 Seed rule (round table 4.2(f))

- Each seed is labelled separately (A2.8). There is no pooled or seed-term alternative.
- **The inferential unit is 3 seeds of one training lineage.** The document bootstrap captures
  within-seed variance only, and the report says so.

### A2.11 Power (round table 4.2(c))

**Method (`m2_power.py`).**
- Real Δ_resample and labels come from the set E sample.
- Scores are synthetic: `softmax_step(a · y + b · u(age) + ε)`, with u = the standardised log
  critical rate by age.
- The spread at n = 1024 documents comes from 300 multinomial redraws of 1024 documents out of
  the 64, i.e. the cluster bootstrap at E0d's n.
- Power for "CI lower ≥ bar" is approximated as `Φ((θ − bar)/sd − 1.96)`. For "CI upper ≤ bar"
  the sign is mirrored.

**Results** (bos-excluded population):

| quantity | sd at n = 1024 | 80%-power detectable value |
|---|---|---|
| `AUROC_strat` | 0.0004–0.0049 over all synthetic scores (`m2.log`, `m3.log`, `m3b.log`) | true `AUROC_strat` ≥ 0.85 + 2.80 × 0.0049 = **0.864** (worst sd) |
| `R_H` | 0.003–0.072 (it grows as `H_age-random` shrinks) | true `R_H` ≤ 0.50 − 2.80 × 0.072 = **0.30** (worst sd); ≈ 0.39–0.44 at sd 0.02–0.04 |
| `C_ws` | 0.0001–0.0034 | – |

- Example: a synthetic score with true `AUROC_strat` 0.860 (sd 0.0022) passes the AUROC bar on
  one seed with power 0.996, so on all three with ≈ 0.99, assuming independent seeds and equal
  truths.
- **False pass.** At a true value exactly on the bar, one seed passes with probability ≤ 0.025.
- **Believed, not verified:**
  - that 64 documents represent `D_E0d`'s distribution;
  - that documents are i.i.d.;
  - the normal approximation;
  - that the synthetic family spans the real `r_i`'s shape.
  - Seed-to-seed spread is not modelled.

### A2.12 Declared departure from §3.2.1's named statistic

- §3.2.1 says "Report Spearman ρ(`r_i`, LOO Δloss)", and §6's prediction is "High ρ".
  - This amendment gates E0d on `AUROC_strat` and `R_H`.
  - It keeps LOO as truth, with the label `y` a threshold of LOO Δ.
  - It still reports Spearman ρ (A2.1).
- **Reason.** With about 5% signal-bearing cells, a perfect-label score reaches a Spearman ρ of
  at most `√(3p(1−p))`. On real Δ that measured 0.3633, and the round table checked it
  analytically and by simulation. So "High ρ" is unreachable by construction.
- Whether this departure needs a spec correction is Brendan's call. Part E has a proposed note;
  `docs/spec-corrections.md` is not edited here.

### A2.13 C8 authority (replaces the ρ\* ruling requirement) and C11

- **C8 now requires**, in `docs/owner/rulings/`, committed and unmodified:
  - `R-*-retrieval-shown*`;
  - `R-*-sprint0-gate*`;
  - **`R-*-e0d-statistic*`**, which **replaces `R-*-rho-star*`**.
- **The `R-*-e0d-statistic*` ruling must state exactly one value each of:**
  - `AUROC_STAR`, in (0.5, 1);
  - `HARM_RATIO_STAR`, in (0, 1).
  - Each is a `KEY: <x>` or `KEY = <x>` line, with optional backticks or asterisks, in the form
    `_RHO_LINE` already accepts.
- **Refusals.**
  - A missing file or key, more than one distinct value, or a value out of range exits 3 before
    any `D_E0d` document is generated.
  - An `R-*-rho-star*` file is neither required nor read.
- **Where the ruling is now.** At this draft, `R-2026-09-27-e0d-statistic.md` exists only in
  `~/Documents/RSR-2026-09-27-plan/owner-drafts/`, not in `docs/owner/rulings/`. Brendan commits
  it.
- **C11 (new, exit 3):** at every full-memory step of every seed, the eligible slots hold exactly
  one slot per age, and `rank = M − age` (A2.3).

### A2.14 Registered prediction

- No new class prediction is registered.
- §9's class prediction (CONFOUND; second most likely MIXED) is scored against the A2.8
  classification.
- Its numeric parts are scored against the ρ secondaries, as A1.8 already arranged.

---

## Part B: test plan for the runner update (tests first; mutation-proven)

### B.1 The calibration fixture (round table §6A)

A pure function, `synthetic_ledger(seed, n_docs, noise="continuous"|"tied")`. It builds per-cell
arrays shaped like `cells_from` output: `doc, t, rank, sentence, full, q_step, a_cell, gap,
d_resample, d_zero`. It must be deterministic (numpy `default_rng(seed)`), use no model, and run
in under 2 s at `n_docs = 64`.

**Matched to the set E sample (`m1.log`):**
- **Shape.** M = 16, S = 48. At `t ≥ 16`, one slot per age 1–16 with `rank = 16 − age`.
- **Q-steps.** About 43% of full-memory steps: 884 of 2048.
- **`k_t` per Q-step.** 0: 26%, 1: 72%, 2: 2%, 3 or more: < 1%.
- **Critical cells by age** (a U shape; counts over 64 docs): 166, 115, 92, 81, 38, 27, 24, 19,
  11, 14, 2, 22, 14, 19, 20, 19.
- **Critical Δ.** Log-normal, median 1.2 and q10 0.62 nats. About 92% of critical cells are
  A-cells.
- **Off-critical Δ,** in one of two modes:
  - **continuous:** |Δ| log-normal matched to q50 0.0094, q90 0.081, q99 0.31, with sign
    positive in 58% of cells;
  - **tied:** exactly 0.
- **Missing values.** 0.8% of cells get a NaN `d_resample`, independent of `y`.
- **The gap-1 A-cell** is at rank 15, so A1.3's exclusions have something to remove.

**Injected proxies** (the functions return an `r` grid; each step's shares sum to 1):
- **perfect:** `softmax(10·y + ε)`;
- **within-step shuffle** of a plausible score;
- **age-only:** `f(age)`, with f the log critical rate by age;
- **recency:** `−age`;
- **graded content:** `softmax(a·y + ε)` for `a ∈ {0.5, 1, 1.5, 2, 3}`;
- **age-steered:** `softmax(a·y + u(age) + ε)`;
- **step-temperature** (OI-2): `softmax(T_t · (a·y + ε))`, with `T_t` = 0.05 on steps with a
  critical cell and 6 elsewhere, and the reverse;
- **pure step-level:** `a = 0`, flat on critical steps, one-hot elsewhere;
- **rescaled:** `c_t · r`, with `c_t ~ U(0.5, 2)` per step.

### B.2 Tests (each named for the rule it pins)

| # | test | expected | pins |
|---|---|---|---|
| T1 | perfect proxy, both noise modes | `AUROC_strat == 1.0`, `H == 0`, `R_H == 0`, `C_ws > 0.4` | 4.4(1) |
| T2 | within-step shuffle, 1024 docs | \|`AUROC_strat` − 0.5\| < 4·sd; \|`H − H_random`\| < 4·sd; `C_ws` within 4·sd of 0 | 4.4(2) |
| T3 | age-only and recency | `AUROC_strat == 0.5` **exactly**; `H == H_age-random` exactly, so `R_H == 1.0`; `C_ws == 0` within 1e-12 | 4.4(3), A2.6 |
| T4a | rescaled `c_t·r` | H, `R_H`, `C_ws` and argmins unchanged, bit for bit | 4.4(4), re-stated |
| T4b | rescaled `c_t·r` | `AUROC_strat` **changes** on the fixture (a pinned counter-example). A future "normalise inside" edit then fails loudly instead of passing silently. | 4.4(4); A2.4 |
| T4c | input already sums to 1 per step | `r / Σ_step r` is the identity, bit for bit | 4.4(4) |
| T5 | a bin with positives but no negatives, and one with neither | excluded, logged with counts, and `w` excludes it; `AUROC_strat` equals the hand value on the rest | 4.4(5) |
| T6 | hand-built steps with `k ∈ {1, 2, 3}` of `n_t ∈ {15, 16}` | `H_random == mean(k/n_t)` exactly; under A1.3, `n_t = 15`, not 16 | 4.4(6), A2.6 |
| T7 | Spearman guard, binary proxy, continuous noise | Spearman ≤ `√(3p(1−p)) + 0.01` | 4.4(7) |
| T8 | ties at the argmin (two slots tied at min, one critical) | `h_t == 0.5`; `π` splits ½ / ½ | A2.6 |
| T9 | `H_age-random == 0` in one bootstrap resample | the `R_H` CI is NaN, so the label is `HARM_UNDEFINED`, never `AGREE` or `DISAGREE` | A2.6, A2.8 r7 |
| T10 | a NaN Δ on one eligible slot of a `k ≥ 1` step | the step leaves `Q_crit` and is counted; AUROC drops only that cell | A2.4, A2.6 |
| T11 | τ | the runner's constants equal A2.2's values; no quantile is computed on the run's cells (a guard test monkeypatches `np.quantile` in the labelling path to raise) | A2.2 |
| T12 | label table | a table-driven test builds one CI tuple per A2.8 row, including boundary equality (`== A*`, `== ρ_H*`), and asserts first-match order: row 4 beats row 5 beats row 6; row 3 beats row 6; row 0 beats all | A2.8 |
| T13 | classification | every §8 row is reachable; exit codes 0/1/2 match; `CEILING` and `HARM_UNDEFINED` count as `UNRESOLVED`; `AGREE, AGREE, DISAGREE` gives `MIXED_UNRESOLVED` | A2.8 |
| T14 | C8 | passes with a committed `R-x-e0d-statistic.md` holding one `AUROC_STAR` and one `HARM_RATIO_STAR`; exits 3 on a missing file, a missing key, two values, `AUROC_STAR ≤ 0.5`, or `HARM_RATIO_STAR ≥ 1`; an `R-*-rho-star*` file alone does not satisfy it | A2.13 |
| T15 | C11 | a fixture with a duplicated age at one step exits 3 | A2.13 |
| T16 | A1.3 population | `n_t == 15`, bin 1 excluded, gap-1 Q-steps absent | A2.3 |
| T17 | bootstrap | one `W` for every statistic of a seed; `π` and `m_a` re-estimated per resample (a test with a document that holds all of age 11's positives) | A2.4–A2.6 |
| T18 | OI-2 fixture | pure step-level: `AUROC_strat` in (0.55, 0.70) and `C_ws` within 4·sd of 0; step-temperature: `C_ws` and H equal to the T = 1 case bit for bit, while `AUROC_strat` differs | OI-2 evidence, as a regression |
| T19 | graded content, `a = 0.5` | `R_H` point < 0.5 while `AUROC_strat` < 0.7. This documents OI-1's weak harm bar as a known property. | OI-1 |

**Mutations for the battery** (each must be killed by at least one test above):
- ties at 0 or 1 instead of ½;
- `w_a` = all cells instead of positives;
- an excluded bin counted as 0.5;
- `n_t = M` under A1.3;
- argmin to argmax;
- π taken from all steps instead of `Q_crit`;
- π not re-estimated per resample;
- `R_H` computed as a ratio of means across resamples;
- `y` using `≥ τ`;
- τ recomputed on the run's cells;
- `CEILING` not counted as `UNRESOLVED`;
- rows 5 and 6 swapped;
- `DISAGREE` on AUROC only;
- C8 still accepting `R-*-rho-star*`;
- C11 disabled;
- pooled across seeds gating.

---

## Part C: OPEN ISSUES for Brendan (none resolved by changing a ruling)

The operative text in Part A follows the ruling as written. Each issue gives evidence and options,
with exact text for any option that would change Part A.

### OI-1. H is near-tautological on this corpus, and "harm" is a misnomer

**Evidence.**
- **The critical cell is the slot being read at `t`.** In the A1.3 population, 475 of 517
  critical cells are A-cells: the queried assert at its own query step. In the all-cells
  population it is 631 of 683.
  - The generator writes exactly one query per fact (`synthetic.py::_generate_document`;
    `true_demand` has one 1 per fact).
  - So an A-cell has **no demand after `t`**.
  - Of all critical cells, only **10 of 517 (1.9%)** are pending asserts, i.e. needed after `t`.
    All-cells: 12 of 683 (1.8%) (`m3.log`).
- **Eviction at step `t` happens after step `t`'s read.** At a query step, evicting the slot LOO
  marks as critical costs almost nothing going forward. This is the k = 0 issue of B2 A1.2.
- **`r_i(t)` is the read's own attention-weighted contribution.** Any `r_i` that reflects
  retrieval puts share on the slot being read, so its argmin almost never lands there.
- **In simulation, the harm bar is passed by scores that fail the AUROC bar badly** (point values
  on 64 documents):

  | score | `AUROC_strat` | `R_H` | population |
  |---|---|---|---|
  | content a = 0.5 | 0.650 | 0.297 | bos-excluded |
  | content a = 0.5 | 0.643 | 0.424 | all cells |
  | content a = 1.0 | 0.745 | 0.176 | bos-excluded |
  | content a = 1.0 | 0.766 | 0.089 | all cells |

  - Sources: `m3b.log` (a = 0.5) and `m3.log` (a = 1.0).
  - `R_H` failed only when an age term pushed the argmin toward rarely-critical ages. Examples:
    `m2.log`, bos-excluded, a = 0.5 and a = 1.0 with an age term: `R_H` 0.75 and 0.62, on about
    5–6 argmin hits in 443 steps.
  - So H rarely decides the outcome. When it does, it turns on a handful of events.
- **What is not measured.** Nothing here uses a real `r_i`. How close the real H sits to 0 is
  unknown until the run.

**Options** (Brendan's; none is applied in Part A):
- **(a) Keep H as ruled and rename it** "argmin-hit rate", with a stated scope.
  - Replace "eviction-harm rate" in A2.6 with: "**argmin-hit rate**: how often `argmin r_i(t)`
    lands on a slot whose knockout changes step `t`'s own loss. It is a contemporaneous
    agreement statistic. It is **not** a measure of what eviction costs later: on S0-03, 98% of
    critical cells have no later demand."
  - The gate is unchanged.
- **(b) Demote H to a secondary.**
  - This changes the ruled gate, so it needs a new ruling.
  - A2.8 row 4 would become "`AUROC_strat` CI lower ≥ A\*". Row 6 would become "`AUROC_strat`
    CI upper < A\*". Rows 5 and 7 would drop their `R_H` terms, and `HARM_RATIO_STAR` would
    leave C8.
- **(c) Keep H, and add the real eviction-consequence measure as its own item.**
  - This is round table §5.2's closed-loop replay: answer-token NLL on held-out documents, with
    eviction driven by `r_i`, random choice, an age-only rule, and a LOO oracle.
  - It **cannot run inside E0d**, because C9/T3 fix FIFO. It is a separate item, and a candidate
    truth for a future lever-(b) correction.
  - (c) combines with (a) or (b).

### OI-2. `AUROC_strat` is cross-step, so step-level concentration moves it

**Evidence** (`m3.log`, `m3b.log`, `m3c.log`; bos-excluded; synthetic scores).
- **Pure step-level signal.** The score is flat on steps with a critical cell and peaked on a
  random slot elsewhere, with no within-step content.
  - `AUROC_strat` = 0.596 (flat vs peaked) and 0.645 (flat vs one-hot).
  - The analytic cap is 0.669: shares sum to 1 per step, so a step-level signal can only beat
    negatives on steps with no critical cell.
  - `C_ws` stays at its null (−0.030 and −0.001, at 64 documents).
  - **Pure step-level concentration cannot reach 0.85, and the companion catches it.**
- **Step-level concentration on top of real content** (per-step temperature on a fixed
  within-step score, so `C_ws` and H are identical by construction):

  | content | T = 1 | flat on critical / peaked elsewhere | peaked on critical / flat elsewhere |
  |---|---|---|---|
  | a = 0.5 | 0.650 | 0.711 (T .2/3) · 0.728 (T .05/6) | 0.551 |
  | a = 1.0 | 0.773 | 0.796 · 0.810 | 0.682 |

  - A step-level effect moves `AUROC_strat` by about +0.08 or −0.10, while the within-step
    information is unchanged.
  - **A direction-only requirement on `C_ws` does not catch this.** `C_ws` is 0.148 and 0.264,
    well above its null, because the content is real.
- **A scale-free alternative.** `AUROC_strat,pct`: the same statistic computed on within-step
  percentiles `pct(t, i)` instead of raw shares.
  - Pure step-level: 0.502.
  - a = 0.5: 0.654 under every temperature.
  - a = 1.0: 0.772 under every temperature.
  - `true_demand`: 0.967. LOO replicate: 0.968.
  - sd at 1024 documents: 0.001–0.004.

**Options:**
- **(a) As ruled.** `C_ws` stays a reported secondary (Part A).
- **(b) The manager's proposal.** `AGREE` additionally needs `C_ws` CI lower > 0.
  - A2.8 row 4 gains "and `C_ws` CI lower > 0".
  - A new row before row 6 reads: "`C_ws` CI upper ≤ 0, with the primary bars met:
    `AGREE_VIA_STEP`, counted as `UNRESOLVED`".
  - Effect: it catches pure step-level passes, which cannot reach 0.85 anyway. It does **not**
    catch step-level inflation of a real but sub-bar content signal.
- **(c)** `AGREE` additionally needs `AUROC_strat,pct` CI lower ≥ A\*.
  - This catches both cases above, and the controls reach it: 0.967 and 0.968.
  - Both (b) and (c) add a gating condition, so either needs Brendan's ruling.
- **(d)** Report `AUROC_strat,pct` and the gap `AUROC_strat − AUROC_strat,pct` as a secondary,
  and flag `STEP_SENSITIVE` when the pct version's label differs. This is the reporting-only
  analogue of `GATE_SENSITIVE`.
- (d) needs no ruling, and I would add it in any case. Part A does not include it yet.

### OI-3. Departure from the spec's named statistic

Declared in A2.12. The spec text is §3.2.1, "Report Spearman ρ(`r_i`, LOO Δloss) … If they
disagree, LOO is truth", and §6's "High ρ".
- LOO stays truth. The gating statistic changes.
- Brendan decides whether a correction is needed. Part E is a proposed note for him to write
  himself, if he wants one. **`docs/spec-corrections.md` is not edited.**

### OI-4. τ comes from seed 0 and 64 documents only

**Evidence.**
- τ's bootstrap interval on the sample is [0.378, 0.472].
- Across it, the critical count moves by 36 cells in 700 (5%). The movement is almost all in the
  non-A tail (65 to 40 cells). The A-cells are far above τ: their q10 is 0.62, and 97% of A-cells
  are critical at τ.
- The registered sensitivities (0.311, 0.713) bracket that interval.
- Seeds 1 and 2's off-A tails are **unmeasured**. The seeds are known to differ: on B0 kind
  means, seed 1's facts rank below their own questions.

**Options:**
- **(a) As drafted.** One frozen τ, with the q = 0.99 and q = 0.999 sensitivities.
- **(b) Per-seed τ.**
  - Run LOO (resample and zero; **no `r_i`, no capture**) on docs `[64, 128)` for seeds 1 and 2.
    The cost is two LOO passes over 64 documents per seed, on the cpu-det lane.
  - Freeze `TAU_RESAMPLE[s]` and `TAU_ZERO[s]` per seed, by the same rule, before any `D_E0d`
    data.
- **(c) A larger sample.** Up to all of set E, `[64, 1088)`, on 3 seeds.

**My view (flagged; not decided).**
- (b) is warranted. It is cheap, it stays pre-data, and it removes an assumption about seeds 1
  and 2 that nothing has checked.
- (c) buys little: τ's interval already moves the label by only about 5%.
- If (b) runs, A2.2's table gains two rows per seed. The rest of Part A is unchanged.

---

## Part D: fix map and disagreements

### D.1 Round-table item → paragraph

| Round table | Answered in |
|---|---|
| 4.1 definitions (scope, cell, truth, label, score, age) | A2.2, A2.3, A2.4 (the "answer-token NLL" truth line corrected to `loo.py`'s sentence-mean Δ) |
| **4.2(a)** within- vs cross-step; companion | A2.3 (cross-step within age, stated explicitly; one slot per age measured); A2.5 (companion); OI-2 |
| **4.2(b)** Q, H_random, H_age-random | A2.6 (`Q_crit`; `H_random` = k/n_t; `H_age-random` = π-matched) |
| **4.2(c)** thresholds as a ratio; power | A2.8 (ruled `AUROC_STAR`, `HARM_RATIO_STAR`, through the CI); A2.11 (power); A2.13 (C8) |
| **4.2(d)** τ | A2.2 (continuous, so a q-quantile; q = 0.995 frozen, with sha); OI-4 |
| **4.2(e)** primary cell | A2.9 |
| **4.2(f)** seed rule | A2.8 (every seed must pass), A2.10 |
| 4.3 computation and nulls | A2.4 (null corrected: the within-(step, bin) shuffle is the identity); A2.6; A2.5 |
| 4.4 tests 1–7 | Part B, T1–T7 (T4 re-stated as T4a–c) |
| 4.5 report | A2.7 (items 1–5, 9) |
| 5.1 retained-set recall | A2.7 item 7 (not applicable at one eviction per step) |
| 5.2 closed-loop replay | A2.7 item 8 (deferred: conflicts with C9/T3); OI-1(c) |
| 5.3 sink/inversion | A2.7 item 9 (with the seed-1 diagnostic re-defined) |
| 5.4 evidence caveat | A2.7 item 10 |
| 6A calibration fixture | Part B.1 |
| 6B redundancy demo | not included: optional, and off the critical path |
| 7 blockers 1, 2, 5 | A2.2, A2.3 (measured) |
| 7 blockers 3, 4 | the ruling |
| 7 blocker 6 | this draft goes to review uncommitted |
| 8 constraints | A2.0 (LOO stays truth; §3.2.1 untouched); age used only for strata, never as a ψ̂ input; tests first (Part B) |

### D.2 Where I disagree with the PM's choices (with evidence)

1. **`H_random = mean k_t / M` is wrong in the operative population.** Under A1.3 there are 15
   eligible slots, so uniform eviction harms with probability `k_t / 15`. On the sample, the
   values are 0.0649 (k/M) against 0.0694 (k/n_t). A2.6 uses `k_t / n_t`. In the all-cells
   population the two agree (0.0651).
2. **`R_H` needs a rule for a zero denominator, and ties need a rule.** `H_age-random` = 0 is
   reachable, whenever `r_i` evicts only at never-critical ages; bin 11 has 2 critical cells in
   713 on the sample. A2.6 and A2.8 add `HARM_UNDEFINED` (exit 2), and a tie rule for the argmin.
3. **The round table's AUROC null is the identity** at one slot per age. It is replaced by the
   within-step shuffle (A2.4).
4. **The round table's test 4.4(4) is vacuous or false as written.** On shares it is the
   identity. Under per-step rescaling `AUROC_strat` does change, because it is cross-step. It is
   re-stated as T4a–c.
5. **The round table's truth line** ("answer-token NLL") does not match `loo.py`, which uses
   sentence-mean NLL. τ was computed on the latter. A2.2 says so.
6. **NaN resample cells.** 106 of 884 Q-steps have at least one eligible slot with no donor. A2.6
   keeps H on complete steps only (443 of 498 `Q_crit` steps, bos-excluded) and reports the
   count. Imputing, or dropping only the argmin slot, would change the eviction decision.
7. **Additions not in the PM's list**, each flagged for the reviewer:
   - **C10′ (row 0).** It can only prevent a kill.
   - **Row 5 (`AGREE_VIA_RANK`, redefined).** It keeps A1.2 in force. Without it, age-only
     agreement would become exit 1 again under an age-stratified primary. A1.2 retired exactly
     that.
   - **Row 6 fires on either bar.**
   - **C11.**
   - **`TAU_ZERO`** for the zero-knockout secondary.
8. **The ruling is still a draft.** `R-2026-09-27-e0d-statistic.md` is in `owner-drafts/`. C8
   needs it committed in `docs/owner/rulings/`.
9. **The manager's four points** are OI-1 to OI-4. On OI-2 specifically: the evidence says the
   proposed direction-only requirement does not catch the case it targets. Option (c) does.

---

## Part E: proposed spec-correction note (for Brendan to write himself; not applied)

> **Correction NN — E0d's gating statistic (lever (a); LOO remains truth).** Spec §3.2.1's
> "Report Spearman ρ(`r_i`, LOO Δloss)" and §6's E0d prediction "High ρ" no longer define E0d's
> pass/kill reading. E0d gates on (i) the age-stratified AUROC of `r_i` for LOO-critical cells,
> `y = 1[Δ_resample > τ]`, and (ii) the rate at which `argmin r_i` lands on a LOO-critical slot,
> relative to age-matched random eviction. Thresholds, τ and age bins are those of
> R-2026-09-27-e0d-statistic and E0d PREREG Amendment 2. Spearman ρ is still reported. **Why:**
> with about 5% signal-bearing cells, a perfect-label score's Spearman ρ against continuous LOO Δ
> is capped at √(3p(1−p)) ≈ 0.36 (measured 0.363). "High ρ" was therefore unreachable by
> construction. §3.2.1's clause "If they disagree, LOO is truth and `r_i` is a confound" is
> unchanged.
