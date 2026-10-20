# E0d Amendment 2: final draft v2, for independent adversarial review

> **Status: DRAFT. NOT COMMITTED. NOT REVIEWED.** Nothing here has been written to
> `experiments/e0d/PREREG.md`. An independent adversarial review comes next. Part A is the text
> that would be appended, verbatim, after A1.10. Parts B–E sit outside the PREREG: the runner test
> plan, what the decisions settled and what they left, the fix map, and a spec-correction note for
> Brendan to write himself.

**Written 2026-09-27 by a model** (Claude Opus 5.5, RSR researcher subagent, item E0D-AMD2-FINAL).
**Completed 2026-09-29 by a second session of the same model** (item E0D-AMD2-FINAL, resumed). The
first session was cut off by an API rate limit after `m5` finished and before `m6` ran. The
resuming session:
- did not re-run `m5`;
- recomputed the three cells-file sha256;
- ran `m6` and `m7`;
- filled every placeholder in the partial draft by script, from `m5_out.json`, `m6_out_012.json`
  and `m7.log`;
- wrote D.2 and D.3.

The partial draft is archived at
`reviews/superseded/E0D-AMENDMENT-2-FINAL-v2.partial-2026-09-27T1912.md`.
- **Supersedes** `reviews/E0D-AMENDMENT-2-FINAL-DRAFT.md` (left untouched).
- **Decisions applied:** `owner-drafts/R-2026-09-27-e0d-statistic-amended.md` (made by the PM under
  Brendan's delegation; rationale in `DIGEST.md`, cycle 17), on top of
  `owner-drafts/R-2026-09-27-e0d-statistic.md`. Neither is committed in `docs/owner/rulings/`.
- **Base:** PREREG at `run/e0d` `268b947` (base, Amendment 1, the A1.10 erratum); runner at
  `4918f2d`.

**Data hygiene.**
- Every number comes from set E, docs `[64, 128)`, arm B (fresh-stream) ckpt3000, seeds 0, 1, 2.
  That set was already inspected (seed 0) or is inside the already-inspected range E `[64, 1088)`
  (seeds 1, 2); it is disjoint from `D_E0d = [262144, 263168)`.
- Seed 0: the saved cells `reviews/e0d-a2/step1_cells.npz` (sha256 in A2.2).
- Seeds 1 and 2: **new** LOO passes (resample and zero; donor seed = model seed), run for this
  draft by `reviews/e0d-a2-final/m5_tau_seeds.py` through `orchestrator.slot run --lane cpu-det`.
  **LOO Δ only: no `r_i` of any kind was computed, no capture forward was run.** The script
  asserts `HI = 128 ≤ 262144` and that `e0d_documents` refuses any range touching `D_E0d`.
- Every score in the calibration and power sections is synthetic, except `true_demand` (the
  A-cell indicator, from document structure).
- **No document of `[262144, 263168)` was generated, read or touched.**
- Scripts, logs, `.npz` files and their sha256 are in `reviews/e0d-a2-final/` (`m5`, `m6`; the
  earlier `m1`–`m4` are unchanged). rc for each run is recorded in Part D.3.

---

## Part A: the amendment text (append-only)

```
## Amendment 2 (2026-09-27, pre-data)

**Written 2026-09-27 by a model** (Claude Opus 5.5, RSR researcher subagent, item E0D-AMD2-FINAL).
**Append-only.** Nothing above this heading has been edited. Where this section and the text
above disagree, **this section governs**, including over Amendment 1 and over the front matter's
`thresholds.RHO_STAR`, `requires` and `prediction` fields.
```

### A2.0 Authority, pre-data status, and what does not change

- **Authority.**
  - `R-2026-09-27-e0d-statistic` (Brendan): adopts the round-table synthesis, rules
    `AUROC_STAR: 0.85`, "Each seed separately. E0d passes only if every seed passes. Pooled
    results are reported and never gate.", primary `r_i` gated, primary knockout resample; and
    "The τ rule and the age bins are fixed in Amendment 2, before any data."
  - `R-2026-09-27-e0d-statistic-amended` (decided by the PM under Brendan's delegation),
    superseding the first ruling in part:
    - "The eviction-harm rate `H` no longer gates." It is reported as the argmin-hit rate.
      `HARM_RATIO_STAR` is withdrawn as a gate.
    - "`AUROC_STAR: 0.85` now applies to `AUROC_strat,pct`." The raw `AUROC_strat` is reported,
      not gated.
    - τ is "computed **per seed** on the already-inspected set E `[64,128)` and frozen in
      Amendment 2, before any data from `[262144, 263168)` is read."
- **LOO remains the truth.** Spec §3.2.1 is unchanged. This amendment changes the *comparison
  statistic*, not the ground truth. The departure from §3.2.1's named statistic ("Report Spearman
  ρ") is declared in A2.12.
- **Pre-data.**
  - At this commit no document of `D_E0d = [262144, 263168)` has been generated, run or inspected,
    and no E0d measurement exists.
  - No arm-B `r_i` has been computed on any document for this amendment.
  - The constants below were computed on set E `[64, 128)` from LOO Δ only.
- **Unchanged:** §1 and §2; A1.1's strata (Q-steps, A-cells, N-steps); A1.1's LOO-flat rule and
  §5's `R_FLAT`; A1.3 (bos-copy exclusions; the exit follows the excluded population); A1.4 (any
  exception exits 3; exit 1 only after `e0d.class` is written); A1.5–A1.7; controls C1–C7 and C9;
  the cluster bootstrap (per-document, 2000 resamples, generator seed `20260927`, 95% percentile,
  the same indices for every statistic of a seed).

### A2.1 Retired from gating (kept as reported secondaries)

- **`RHO_STAR` and every ρ-based rule** (A1.1's ρ_Q and ρ_Q,rank, §4, A1.8, the ρ\* sensitivity
  values {0.3, 0.7}, A1's §7 table). All are still computed and reported through the same code and
  bootstrap. The A1 per-seed label is reported as `label_A1` and changes no exit.
- **C10 as a ρ rule.** Its positive control (`true_demand`, 1 on the A-cell) is still computed on
  every ρ statistic and reported with its ρ ceiling. The same control is the CEILING control of
  A2.8, row 0.
- **H, `R_H`, `H_age-random` and `HARM_RATIO_STAR`.** H is reported as the **argmin-hit rate**
  (A2.6). No label, class or exit reads it.
- `ρ_pool`, `ρ_step` and bottom-1 agreement remain secondaries, as in A1.1.

### A2.2 The critical label and τ, per seed

- **Critical cell.** `y(t, i) = 1[Δ_resample(t, i) > τ_s]`, strict `>`, with `τ_s` the frozen
  value of the cell's seed `s`.
  - Δ is `loo_delta_loss(...)["delta"]` (§3): the change in sentence `t`'s mean real-target NLL
    when slot `i` alone is knocked out. It is not answer-token NLL; τ was computed on this Δ.
  - A cell whose Δ_resample is NaN (no resample donor) **has no label** (A2.4).
- **The rule.** `τ_s` = the q = 0.995 quantile (numpy default, linear / type 7) of
  |Δ_resample| over full-memory Q-step cells of seed `s` that are not A-cells and have a value;
  computed once on set E `[64, 128)` and frozen. The off-A Δ is continuous (exact zeros: see the
  table), so τ = 0⁺ does not apply.
- **The samples.** Arm B ckpt3000 of each seed (sha256 as C1), resample donor seed = model seed,
  64 documents per seed.

  | seed | cells file | sha256 |
  |---|---|---|
  | 0 | `reviews/e0d-a2/step1_cells.npz` | `1338de4bdc5fbf9d33c8bd37abb34532535d2cd1e77f18e4deb87c357a65f15c` |
  | 1 | `reviews/e0d-a2-final/tau_cells_seed1.npz` | `b683dd1ba1c105a52edc440dd300e2ada9c090e1fcee4fa22740756935f6cd6c` |
  | 2 | `reviews/e0d-a2-final/tau_cells_seed2.npz` | `4c55cc601c70bee4aebf79c331fe6984601feb61a63df0ed99c77e4991156197` |

- **Frozen values** (nats per sentence; full precision is the registered value):

  | seed | ckpt sha256 | `TAU_RESAMPLE[s]` (q = 0.995, **primary**) | 95% doc-bootstrap CI | `TAU_ZERO[s]` (q = 0.995) | 95% CI | n (res / zero) |
  |---|---|---|---|---|---|---|
  | 0 | `0ee3f8a6…a118da60` | **0.4157434984576128** | [0.378, 0.472] | 0.3447678400575726 | [0.315, 0.393] | 13,376 / 13,481 |
  | 1 | `dadd1e08…068a3da8` | **0.4315741845071321** | [0.377, 0.464] | 0.3300262098312375 | [0.292, 0.387] | 12,842 / 12,969 |
  | 2 | `b507ebc5…fd6b63b8` | **0.4884946896703897** | [0.435, 0.519] | 0.386515489417143 | [0.347, 0.422] | 13,398 / 13,492 |

  **Sensitivities (reported, never gating):**

  | seed | resample q = 0.99 | resample q = 0.999 | zero q = 0.99 | zero q = 0.999 |
  |---|---|---|---|---|
  | 0 | 0.3109549731015192 | 0.7129988655444408 | 0.24913927197480037 | 0.5874972405433865 |
  | 1 | 0.310573765520007 | 0.7854374831784502 | 0.24216222953796368 | 0.6492233513593663 |
  | 2 | 0.3491311189141148 | 0.7694723016164862 | 0.2818768095970196 | 0.6450255164741889 |

  - Exact zeros in the off-A |Δ| population: 0 in every seed and both knockouts (`n_exact_zero`), so τ = 0⁺ does not apply.
  - Quantile: numpy 2.5.3, default `linear` (type 7). Source: `reviews/e0d-a2-final/m5_out.json` (`m5.rc` = 0). The full-precision values are the registered values; each is bound to its seed's cells-file sha256 above.

- **τ is a constant, not a rule re-run on `D_E0d`.** The runner reads `TAU_RESAMPLE[s]` and
  `TAU_ZERO[s]`. It never computes a quantile on `D_E0d` cells.
- **Where τ lands** (reported, not registered):

  | seed | knockout | Q-cells with a value | critical at τ | of which A-cells (A-cells with a value) | critical non-A |
  |---|---|---|---|---|---|
  | 0 | resample | 14,028 | 683 | 631 (652) | 52 |
  | 0 | zero | 14,144 | 673 | 646 (663) | 27 |
  | 1 | resample | 13,443 | 629 | 589 (601) | 40 |
  | 1 | zero | 13,584 | 620 | 609 (615) | 11 |
  | 2 | resample | 14,046 | 661 | 614 (648) | 47 |
  | 2 | zero | 14,144 | 657 | 629 (652) | 28 |

  This table is all-cells (before A1.3). In the A1.3 population the resample positives are 517 / 506 / 512 (seeds 0 / 1 / 2; `m6_out_012.json`).

### A2.3 Age strata

- **Age.** `a(t, i) = t − slot_sentence(t, i)`.
- **Bins.** 16 bins, one per age: `B = {{1}, {2}, …, {16}}`, fixed before data and never chosen
  from Δ or `r_i`.
- **Measured structure** (set E, every full-memory step, all three seeds): every full-memory step
  holds exactly one slot of each age 1–16, and write-order rank is `i = M − a`. The runner asserts
  both, per seed, on `D_E0d`, and exits 3 on violation (C11, A2.13).
- **Consequence.** No two cells are both within-step and within-age, so every comparison inside a
  bin is between cells of the same age at different steps. That is why the gated statistic is
  computed on within-step percentiles (A2.4): the percentile carries each cell's within-step
  standing across steps, and removes any step-level scale.
- **Under A1.3** rank M−1 is excluded, so bin {1} is empty; it is logged as excluded and the
  statistic runs over bins 2–16.

### A2.4 The primary statistic `AUROC_strat,pct`

**Population.** The cells of A1.1's Q-steps (full memory, sentence `t` a query) in the A1.3
population (operative: ranks 0 … M−2, gap-1 Q-steps excluded), per seed. At each such step the
**eligible slots** are those cells; under A1.3 `n_t = M − 1 = 15`, asserted by C11.

**Score.** Gated `r_i(t)` (A2.9), as emitted. A non-finite `r_i` on an eligible slot is
`MeasurementUndefined`, exit 3.

**Step 1, the within-step percentile.** At each step `t`, over **all** its `n_t` eligible slots:
- `mr(t, i)` = the midrank of `r_i(t)` among them (rank 1 = smallest; slots tied on `r` share the
  mean of the ranks they span);
- `pct(t, i) = (mr(t, i) − 1) / (n_t − 1)`, in [0, 1].
- **NaN donors.** A slot whose Δ_resample is NaN **still counts** in `n_t` and still receives a
  `pct`. The percentile is a property of the eviction candidates' scores, not of the labels, so a
  missing label never shifts another slot's percentile. The step is **not** dropped.

**Step 2, the label.** `y(t, i)` as A2.2 on cells with a Δ value. A cell with a NaN Δ has no label
and leaves steps 3–4 only.

**Step 3, per age bin.** For each bin `a` with at least one positive and one negative labelled
cell: `AUROC_a` = the Mann–Whitney probability that a positive's `pct` exceeds a negative's `pct`,
over all positive–negative pairs in the bin (across steps); **equal `pct` counts ½**. A bin
without both classes is excluded and logged (bin, positives, negatives); it never counts as 0 or ½.

**Step 4.** `AUROC_strat,pct = Σ_a w_a · AUROC_a / Σ_a w_a`, `w_a` = the number of positives in
bin `a`.

**Bootstrap.** Each cell carries its document's bootstrap count; `AUROC_a` and `w_a` are computed
on the weighted sample. `pct` is computed once per step on the original data (a document's steps
are resampled whole, so a step's percentiles never change within a resample).

**Undefined.** If no bin has both classes, the statistic has no value: `MeasurementUndefined`,
exit 3, unless the LOO-flat rule already labelled the seed.

**Properties** (each pinned by a Part B test):
- invariant to any strictly increasing transform of `r` applied per step, including per-step
  rescaling and temperature (T4);
- an age-only score (any function of age alone) gives **exactly 0.5**: with one slot per age per
  step, its `pct` is a function of age, so it is tied inside every bin (T3);
- a pure step-level signal (no within-step content) gives 0.5 in expectation (T18);
- the binary perfect proxy `r = y` gives **exactly 1.0** (T1a). A perfect proxy with continuous
  noise does **not** reach 1.0: on a Q-step with no critical cell some negative holds `pct = 1`,
  and it ties or beats positives of its age. Its ceiling on set E is in A2.11 (T1b).

**Reported values on set E** (A2.11 has the full table): the table below.

| score | seed 0 pct (raw) | seed 1 pct (raw) | seed 2 pct (raw) | sd of pct at n = 1024, seeds 0/1/2 |
|---|---|---|---|---|
| perfect proxy, binary `r = y` | 1.0000 (1.0000) | 1.0000 (1.0000) | 1.0000 (1.0000) | 0.0000 / 0.0000 / 0.0000 |
| perfect proxy, continuous `softmax(10y+ε)` | 0.9872 (0.9992) | 0.9875 (0.9993) | 0.9876 (0.9996) | 0.0002 / 0.0002 / 0.0002 |
| `true_demand` (C10′ control) | 0.9672 (0.9585) | 0.9731 (0.9639) | 0.9750 (0.9599) | 0.0015 / 0.0013 / 0.0011 |
| LOO replicate, donor seed 1 (seed 0 only) | 0.9675 (0.9885) | – | – | 0.0011 / – / – |
| age-only `f(age)` | 0.5000 (0.5000) | 0.5000 (0.5000) | 0.5000 (0.5000) | 0.0000 / 0.0000 / 0.0000 |
| recency `−age` | 0.5000 (0.5000) | 0.5000 (0.5000) | 0.5000 (0.5000) | 0.0000 / 0.0000 / 0.0000 |
| pure step-concentration (no within-step content) | 0.5008 (0.6458) | 0.4963 (0.6534) | 0.4995 (0.6534) | 0.0007 / 0.0007 / 0.0010 |
| within-step shuffle (null) | 0.5063 (0.4995) | 0.5087 (0.4946) | 0.4683 (0.4621) | 0.0034 / 0.0032 / 0.0036 |
| content `a = 2.0` | 0.9203 (0.9326) | 0.9176 (0.9366) | 0.9070 (0.9284) | 0.0014 / 0.0013 / 0.0016 |
| content `a = 2.0`, step temperature 0.05 / 6 | 0.9203 (0.9176) | 0.9176 (0.9217) | 0.9070 (0.9115) | 0.0014 / 0.0013 / 0.0016 |
| content `a = 2.0`, rescaled `c_t · r` | 0.9203 (0.9196) | 0.9176 (0.9200) | 0.9070 (0.9124) | 0.0014 / 0.0013 / 0.0016 |
| content `a = 2.0` + age term `u(age)` | 0.8915 (0.9340) | 0.8750 (0.9376) | 0.8702 (0.9302) | 0.0014 / 0.0015 / 0.0017 |

- Point values are on each seed's 64 documents. The sd column is the spread at E0d's n = 1024 (300 multinomial redraws); at the 64-document fixture size the sd is about 4× larger.
- Perfect continuous proxy restricted to steps with `k_t ≥ 1`: 0.9985 / 0.9989 / 0.9992 (the k = 0 steps hold it below 1.0; T1c).
- Source: `reviews/e0d-a2-final/m6_out_012.json` (`m6.rc` = 0).

### A2.5 Reported secondaries of the AUROC family (never gating)

- **`AUROC_strat`** (raw): the A2.4 procedure with `r_i` in place of `pct` (the first ruling's
  form). It is cross-step on raw shares, so step-level concentration moves it.
- **`STEP_SENSITIVE`**: flagged when the per-seed label computed with `AUROC_strat` in place of
  `AUROC_strat,pct` differs from the gated label. It changes no exit (the analogue of
  `GATE_SENSITIVE`).
- **`AUROC_unstrat,pct`**: the A2.4 procedure with a single bin holding every age. Used by A2.8
  row 5 only.
- **`C_ws`** (the within-step companion), as in the v1 draft: `pct` as A2.4 step 1 over the
  eligible slots; `m_a` = the mean `pct` over all labelled Q-step cells of age `a`; `v_t` = the
  mean over a step's critical cells of `pct − m_a`; `C_ws` = the mean of `v_t` over steps with at
  least one critical cell; `m_a` re-estimated per resample. Null 0; an age-only score gives exactly
  0.

### A2.6 The argmin-hit rate H (reported, never gating)

- **Definition, as a name with its limit.** H is the **argmin-hit rate**: how often
  `argmin r_i(t)` lands on a slot whose knockout changes step `t`'s own loss by more than `τ_s`.
  It is a contemporaneous agreement statistic. **It is not a measure of what eviction costs
  later:** on set E seed 0, only 10 of 517 critical cells in the A1.3 population (1.9%) are
  pending asserts, i.e. needed after `t`; the rest are the queried assert at its own query step,
  which has no later demand (`m3.log`).
- **Eviction consequence is B2's question**, answered closed-loop under B2's own pre-registered
  rule: `experiments/b2-psi-probe/PREREG.md` on branch `run/b2-psi-probe` (last PREREG commit
  `ed7fce8`), §7 (arms: ψ̂-U, ψ̂-C, FIFO, age-only, random ×5, fact/filler, oracle, kind-oracle),
  §9.5–9.6 (per-seed outcome and classification on all-query answer accuracy, model-read). **Scope
  note:** B2 evicts by ψ̂, the learned head of spec §3.2.2; it has no arm that evicts by
  `argmin r_i`. E0d reports H; nothing pre-registered measures the closed-loop consequence of
  `r_i`-driven eviction.
- **Computation** (unchanged from v1; kept because it is reported):
  - `Q_crit` = the Q-steps with `k_t ≥ 1` critical cells whose every eligible slot has a
    Δ_resample value; steps dropped for a missing value are counted and reported.
  - `h_t` = the fraction of the step's tied minima of `r` that are critical (½ for a two-way tie
    with one critical).
  - `H` = mean over `Q_crit` of `h_t`; `H_random` = mean over `Q_crit` of `k_t / n_t` (`n_t` = 15
    under A1.3); `H_age-random` = the age-matched random-eviction baseline (π from `r_i`'s own tied
    minima on `Q_crit`, re-estimated per resample); `R_H = H / H_age-random`; `R_H,uniform =
    H / H_random`; `H_FIFO`; `H_age-oracle`.
  - If `H_age-random = 0` in the point estimate or any resample, `R_H` is reported as undefined.
    There is no `HARM_UNDEFINED` label any more.

### A2.7 Other reported secondaries (never gating)

Each per seed; the pooled-over-seeds value is reported with a seed-stratified bootstrap and never
gates.
1. **The 2×2 grid** (gated / ungated `r_i` × resample / zero knockout), every A2.4–A2.6 statistic
   and the A2.8 label in each cell, primary cell marked. The zero column uses `TAU_ZERO[s]`.
2. **The all-cells population** (no A1.3 exclusions); `BOS_SENSITIVE` as in A1.3.
3. **τ sensitivity:** every label recomputed at the q = 0.99 and q = 0.999 values of A2.2.
4. **Per-bin detail:** each `AUROC_a` (pct and raw) with its positive and negative counts, and the
   excluded bins.
5. **Every A1 ρ statistic**, C10's ρ ceiling, and `label_A1`.
6. **Round table §5.1 (retained-set recall):** not reported (one eviction per step makes it
   1 − H on `Q_crit`).
7. **Round table §5.2 (closed-loop replay):** not run in E0d (C9 / PLAN-v4 T3 fix FIFO). See A2.6
   for where the consequence question goes.
8. **Round table §5.3 (sink/inversion probe):** the `r_i` share by sentence kind (assert, query,
   filler; pending and spent asserts separately) at Q-steps; and, at full-memory steps where a
   fact's assert and its already-read query are both resident, the fraction with
   `r_assert > r_query`.
9. **Round table §5.4:** R-EVICT's numbers are cited nowhere in the E0d report as evidence.

### A2.8 Per-seed labels, classification, exits

**Threshold:** `A* = AUROC_STAR`, read from the C8 ruling(s), never typed. Every reading is through
the seed's 95% CI, never a point estimate. The primary cell is gated `r_i` × resample in the A1.3
population. **First matching row wins. This table replaces A1's §7 table and the v1 draft's.**

| # | condition | label |
|---|---|---|
| 0 | C10′: the `true_demand` control's `AUROC_strat,pct` CI upper < A\* | `CEILING` |
| 1 | LOO flat (A1.1) and `r_i` flat (§5) | `DEGENERATE` |
| 2 | LOO flat, `r_i` not flat | `LOO_UNINFORMATIVE` |
| 3 | `AUROC_strat,pct` CI upper < 0.5 | `INVERTED` |
| 4 | `AUROC_strat,pct` CI lower ≥ A\* | `AGREE` |
| 5 | `AUROC_strat,pct` CI upper < A\* **and** `AUROC_unstrat,pct` CI lower ≥ A\* | `AGREE_VIA_RANK` |
| 6 | `AUROC_strat,pct` CI upper < A\* | `DISAGREE` |
| 7 | otherwise (the CI contains A\*) | `UNRESOLVED` |

- **Row 0 (CEILING) is kept, and why.** The pct form's reachable maximum is below 1 whenever the
  score is continuous (A2.4), and the `true_demand` control's value depends on each seed's own
  label composition (critical non-A cells score 0 under it). Row 0 turns a kill into exit 2 only
  where a perfect-structure score could not have passed on that seed's labels. It can never
  produce a pass. Measured on set E: the `true_demand` control reads `AUROC_strat,pct` 0.9672 / 0.9731 / 0.9750 on seeds 0 / 1 / 2 (sd at n = 1024: 0.0015 / 0.0013 / 0.0011), about 80 sd above the bar. So row 0 is expected not to fire; it
  costs nothing when it does not.
- **Row 5 keeps A1.2 in force:** agreement that exists only without age control is not the
  §3.2.1 kill. Class `RECENCY_ONLY`, exit 2. It can never produce a pass.
- **Row 3 (`INVERTED`)** reads the gated statistic's CI upper < 0.5: a score that ranks critical
  slots *below* their same-age, same-standing peers.
- **Removed:** the v1 rows that read `R_H` (its term in rows 0, 4, 5, 6) and `HARM_UNDEFINED`.

**§8 classification over the 3 seeds** (first match wins). `CEILING` counts as `UNRESOLVED`.

| # | condition | class (`e0d.class`) | exit |
|---|---|---|---|
| 0 | any control C1–C11 fails, any measurement raises, any checkpoint missing | `inconclusive` | **3** |
| 1 | ≥ 2 seeds `DEGENERATE` or `LOO_UNINFORMATIVE` | `DEGENERATE_UNINFORMATIVE` | **2** |
| 2 | **all 3 seeds `AGREE`** | `AGREE` | **0** |
| 3 | all 3 seeds `INVERTED` | `CONFOUND_INVERTED` | **1** |
| 4 | all 3 seeds in {`DISAGREE`, `INVERTED`} | `CONFOUND` | **1** |
| 5 | all 3 seeds in {`DISAGREE`, `INVERTED`, `AGREE_VIA_RANK`}, at least one `AGREE_VIA_RANK` | `RECENCY_ONLY` | **2** |
| 6 | otherwise | `MIXED_UNRESOLVED` | **2** |

- **Seed rule (the ruling).** A pass needs every seed `AGREE`. Pooled results never gate.
- A kill (exit 1) needs every seed in {`DISAGREE`, `INVERTED`}. `AGREE, AGREE, DISAGREE` is
  `MIXED_UNRESOLVED`, exit 2.
- **Unchanged reporting:** `GATE_SENSITIVE` (ungated class differs; exit follows gated);
  `STEP_SENSITIVE` (A2.5); the zero knockout never changes the exit; `BOS_SENSITIVE`; A1's
  reporting for the flat conditions.
- **Exit discipline.** A1.4 unchanged: exit 1 only after `e0d.class ∈ {CONFOUND,
  CONFOUND_INVERTED}` is written; the verifier checks that rc and class agree.

### A2.9 Primary cell

Gated `r_i` × resample LOO, per the ruling. The other three cells are secondaries (A2.7 item 1).

### A2.10 Seed rule

Each seed is labelled separately, with its own `τ_s`. There is no pooled or seed-term alternative.
The inferential unit is 3 seeds of one training lineage; the document bootstrap captures
within-seed variance only, and the report says so.

### A2.11 Power (the single gate: `AUROC_strat,pct` ≥ 0.85, per seed, through the CI)

**Method (`m6_pct_calib_power.py`).**
- Labels: each seed's real Δ_resample on set E `[64, 128)` and its own frozen `τ_s`.
- Scores: synthetic, per step `softmax(a·y + ε)` (content), with and without an age term
  `u(age)` (standardised log critical rate by age on that seed); plus the calibration scores.
- Spread at n = 1024 documents: the sd over 300 multinomial redraws of 1024 documents from the 64
  (the cluster bootstrap at E0d's n).
- One-seed power for "CI lower ≥ 0.85" ≈ `Φ((θ − 0.85)/sd − 1.96)`; three-seed power is the
  product over seeds (independent seeds, each at its own θ and sd).

**Results** (A1.3 population):

| seed | τ_s | positives (A1.3) | Q-steps (A1.3) | `k_t` = 0 / 1 / 2 / 3 | steps with a NaN-donor eligible slot | pct sd, content scores near the bar | 80% one-seed detectable θ | 80% all-three detectable θ | worst sd of any score → all-three θ |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.41574 | 517 | 720 | 222 / 481 / 15 / 2 | 80 | 0.0014–0.0026 | 0.8573 | 0.8589 | 0.0034 → 0.8616 |
| 1 | 0.43157 | 506 | 724 | 232 / 478 / 14 / 0 | 106 | 0.0013–0.0023 | 0.8564 | 0.8579 | 0.0032 → 0.8610 |
| 2 | 0.48849 | 512 | 740 | 239 / 490 / 11 / 0 | 79 | 0.0016–0.0027 | 0.8576 | 0.8592 | 0.0036 → 0.8623 |

**The graded content family** (`softmax(a·y + ε)`): `AUROC_strat,pct` and one-seed power `P(CI_lo ≥ 0.85)`.

| a | seed 0 | seed 1 | seed 2 | all three (product) |
|---|---|---|---|---|
| 0.5 | 0.6419 (P 0.000) | 0.6490 (P 0.000) | 0.6310 (P 0.000) | 0.000 |
| 1.0 | 0.7629 (P 0.000) | 0.7729 (P 0.000) | 0.7470 (P 0.000) | 0.000 |
| 1.25 | 0.8149 (P 0.000) | 0.8187 (P 0.000) | 0.7979 (P 0.000) | 0.000 |
| 1.4 | 0.8414 (P 0.000) | 0.8442 (P 0.000) | 0.8214 (P 0.000) | 0.000 |
| 1.5 | 0.8596 (P 0.999) | 0.8579 (P 0.991) | 0.8384 (P 0.000) | 0.000 |
| 2.0 | 0.9203 (P 1.000) | 0.9176 (P 1.000) | 0.9070 (P 1.000) | 1.000 |
| 2.5 | 0.9510 (P 1.000) | 0.9541 (P 1.000) | 0.9516 (P 1.000) | 1.000 |
| 3.0 | 0.9709 (P 1.000) | 0.9749 (P 1.000) | 0.9734 (P 1.000) | 1.000 |
| 1.0 + age term | 0.7553 (P 0.000) | 0.7461 (P 0.000) | 0.7293 (P 0.000) | 0.000 |
| 2.0 + age term | 0.8915 (P 1.000) | 0.8750 (P 1.000) | 0.8702 (P 1.000) | 1.000 |
| 3.0 + age term | 0.9420 (P 1.000) | 0.9366 (P 1.000) | 0.9343 (P 1.000) | 1.000 |

- **Detectable effect.** With the largest sd among content scores in [0.75, 0.95] (0.0026 / 0.0023 / 0.0027 on seeds 0 / 1 / 2), one seed whose true `AUROC_strat,pct` is at least about **0.857** passes with 80% power. All three seeds pass together with 80% power when each true value is at least about **0.859** (z = 3.423; equal truths, independent seeds). With the worst sd of any score (0.0036), the figure is **0.862**.
- **The bar is steep, not soft.** One-seed power goes from 0.000 to at least 0.99 between true values of about 0.842 and 0.858. The design resolves the bar to about ±0.01. It does not resolve seed-to-seed differences in the real `r_i`, which the bootstrap does not model.
- **The same synthetic content strength is not equally hard on every seed.** At `a = 1.5`, seeds 0 and 1 pass (0.8596, 0.8579) and seed 2 fails (0.8384). Seed 2 has the largest τ (0.488), and so a different positive set. This is a property of per-seed labels. It is stated here so that it is not read later as a seed effect of `r_i`.
- **An age term in `r_i` lowers the pct statistic.** At `a = 2.0`, adding `u(age)` moves pct from 0.9203 / 0.9176 / 0.9070 down to 0.8915 / 0.8750 / 0.8702, while raw rises slightly. Within a step, the age component reorders the slots. Within a bin every cell carries the same age shift, so the content contrast is compressed. See D.2 item 3.
- The "LOO replicate" product in `m6_out_012.json` (1.0) is over seed 0 only; it is not a three-seed figure.

- **False pass.** At a true value exactly on the bar, one seed passes with probability ≤ 0.025;
  all three with ≤ 0.025³ if independent.
- **Believed, not verified:** that 64 documents per seed represent `D_E0d`; that documents are
  i.i.d.; the normal approximation; that the synthetic family spans the real `r_i`'s shape.
  Seed-to-seed spread of the real `r_i` is not modelled.

### A2.12 Declared departure from §3.2.1's named statistic

- §3.2.1 says "Report Spearman ρ(`r_i`, LOO Δloss)", and §6's prediction is "High ρ". This
  amendment gates E0d on `AUROC_strat,pct` alone. LOO stays truth (`y` thresholds LOO Δ).
  Spearman ρ is still reported (A2.1).
- **Reason.** With about 5% signal-bearing cells, a perfect-label score reaches Spearman ρ of at
  most `√(3p(1−p))`, measured 0.3633 on real Δ; "High ρ" is unreachable by construction.
- Whether a spec correction is needed is Brendan's call (Part E). `docs/spec-corrections.md` is
  not edited here.

### A2.13 C8 authority and C11

- **C8 now requires**, in `docs/owner/rulings/`, committed and unmodified: `R-*-retrieval-shown*`;
  `R-*-sprint0-gate*`; and **every `R-*-e0d-statistic*` file** (the ruling and its amendment),
  which **replace `R-*-rho-star*`**.
- **Across all `R-*-e0d-statistic*` files, exactly one distinct `AUROC_STAR` value, in (0.5, 1).**
  `HARM_RATIO_STAR` is **not required and not read**; its presence in the first ruling is not an
  error.
- **Key form.** `AUROC_STAR`, optional backticks or asterisks, `:` or `=`, a number. **Unlike
  `_RHO_LINE`, the match is not anchored at line start**: it is
  `(?<![A-Za-z0-9_])[`*]*AUROC_STAR[`*]*\s*[:=]\s*([0-9]*\.?[0-9]+)`. Reason: in both ruling
  drafts the key sits inside a table cell or a numbered list item, and the line-anchored form
  `_RHO_LINE` uses matches **neither** file (measured; Part D.2 item 2). Both files match the
  unanchored form with 0.85.
- **Refusals.** A missing file, no `AUROC_STAR`, more than one distinct value, or a value out of
  range exits 3 before any `D_E0d` document is generated. An `R-*-rho-star*` file is neither
  required nor read.
- **Where the rulings are now.** Both files are in `~/Documents/RSR-2026-09-27-plan/owner-drafts/`,
  not in `docs/owner/rulings/`. Brendan commits them.
- **C11 (new, exit 3):** at every full-memory step of every seed, the eligible slots hold exactly
  one slot per age and `rank = M − age` (A2.3); under A1.3, `n_t = 15`.
- **C12 (new, exit 3):** the runner's `TAU_RESAMPLE` and `TAU_ZERO` tables hold exactly seeds
  {0, 1, 2} and equal A2.2's values; a seed without a frozen τ is not run.

### A2.14 Registered prediction

No new class prediction is registered. §9's class prediction (CONFOUND; second most likely MIXED)
is scored against the A2.8 classification; its numeric parts against the ρ secondaries, as A1.8
arranged.

---

## Part B: test plan for the runner update (tests first; mutation-proven)

### B.1 The calibration fixture

A pure function `synthetic_ledger(seed, n_docs, noise="continuous"|"tied", k_profile="matched"|
"one_per_step")` returning per-cell arrays shaped like `cells_from` output (`doc, t, rank,
sentence, full, q_step, a_cell, gap, d_resample, d_zero`). Deterministic (numpy
`default_rng(seed)`), no model, under 2 s at `n_docs = 64`.

**Matched to set E seed 0 (`m1.log`):** M = 16, S = 48, one slot per age 1–16 at `t ≥ 16` with
`rank = 16 − age`; Q-steps ≈ 43% of full-memory steps; `k_t` per Q-step 0: 26%, 1: 72%, 2: 2%,
≥ 3: < 1%; critical cells by age U-shaped (166, 115, 92, 81, 38, 27, 24, 19, 11, 14, 2, 22, 14,
19, 20, 19 over 64 docs); critical Δ log-normal (median 1.2, q10 0.62), ≈ 92% A-cells; off-critical
Δ continuous (|Δ| q50 0.0094, q90 0.081, q99 0.31, 58% positive) or tied at 0; 0.8% NaN
`d_resample` independent of `y`; a gap-1 A-cell at rank 15. `k_profile="one_per_step"` makes every
Q-step hold exactly one critical cell.

**Injected proxies** (each returns an `r` grid; each step's shares sum to 1 unless stated):
perfect binary `r = y` (tied); perfect continuous `softmax(10·y + ε)`; within-step shuffle of a
plausible score; age-only `f(age)` (f = log critical rate by age) and recency `−age`; graded
content `softmax(a·y + ε)`, `a ∈ {0.5, 1, 1.5, 2, 3}`; age-steered `softmax(a·y + u(age) + ε)`;
step-temperature `softmax(T_t·(a·y + ε))`, `T_t` = 0.05 on steps with a critical cell and 6
elsewhere, and the reverse; pure step-concentration (`a = 0`: flat on steps with a critical cell,
peaked on a random slot elsewhere); rescaled `c_t · r`, `c_t ~ U(0.5, 2)` per step.

### B.2 Tests (each named for the rule it pins)

| # | test | expected | pins |
|---|---|---|---|
| T1a | **perfect proxy, binary `r = y`**, both noise modes, matched `k` profile | `AUROC_strat,pct == 1.0` exactly | calibration (decision 6) |
| T1b | perfect proxy, continuous `softmax(10y+ε)`, `k_profile="one_per_step"` | `AUROC_strat,pct == 1.0` exactly | calibration (decision 6) |
| T1c | perfect continuous proxy, matched profile | `AUROC_strat,pct` equals the hand-computed ceiling (the k = 0 steps' `pct = 1` negatives tie or beat positives), and is < 1; documents A2.4's ceiling | A2.4 |
| T2 | within-step shuffle, 1024 docs | \|`AUROC_strat,pct` − 0.5\| < 4·sd; `C_ws` within 4·sd of 0 | null |
| T3 | **age-only** `f(age)` and recency, both populations | `AUROC_strat,pct == 0.5` **exactly**; raw `AUROC_strat == 0.5` exactly; `C_ws == 0` within 1e-12; `R_H == 1.0` exactly | calibration (decision 6); A2.4 |
| T4a | rescaled `c_t·r` and step-temperature | `AUROC_strat,pct`, `C_ws`, H, argmins unchanged, bit for bit | A2.4 invariance |
| T4b | rescaled `c_t·r` | raw `AUROC_strat` **changes** (pinned counter-example), so `STEP_SENSITIVE` is computable | A2.5 |
| T5 | a bin with positives but no negatives, and one with neither | excluded, logged with counts, `w` excludes it; value equals the hand value on the rest | A2.4 step 3 |
| T6 | hand-built steps, `k ∈ {1,2,3}` of `n_t ∈ {15,16}` | `H_random == mean(k/n_t)` exactly; `n_t = 15` under A1.3 | A2.6 (reported) |
| T7 | Spearman guard, binary proxy, continuous noise | Spearman ≤ `√(3p(1−p)) + 0.01` | A2.12 |
| T8 | ties in `r` within a step: two slots tied, one critical | both get the same midrank `pct`; in AUROC equal `pct` counts ½; hand value matches | A2.4 step 1, 3 |
| T9 | a NaN Δ on one eligible slot of a step | `n_t` still 15 and every other slot's `pct` is unchanged vs the same step with that Δ filled; only that cell leaves the AUROC; the step stays | A2.4 NaN rule |
| T10 | a non-finite `r` on an eligible slot | `MeasurementUndefined`, exit 3 | A2.4 |
| T11 | τ | runner constants equal A2.2's per-seed values; the labelling path uses `τ_s` of the cell's seed; no quantile computed on run cells (`np.quantile` monkeypatched to raise in the labelling path) | A2.2, C12 |
| T12 | label table | one CI tuple per A2.8 row, including boundary equality (`== A*`, CI upper == 0.5); first-match order: row 0 beats all; row 3 beats row 6; row 4 beats row 5 beats row 6 | A2.8 |
| T13 | classification | every §8 row reachable; exits 0/1/2 match; `CEILING` counts as `UNRESOLVED`; `AGREE, AGREE, DISAGREE` → `MIXED_UNRESOLVED` | A2.8 |
| T14 | C8 | passes on fixture copies of **the two real ruling texts** (key in a table cell and in a list item); exits 3 on a missing file, no key, two distinct values, `AUROC_STAR ≤ 0.5` or `≥ 1`; `HARM_RATIO_STAR` absent passes; an `R-*-rho-star*` file alone fails | A2.13 |
| T15 | C11 | a duplicated age at one step exits 3 | A2.13 |
| T16 | A1.3 population | `n_t == 15`, bin 1 excluded, gap-1 Q-steps absent | A2.3 |
| T17 | bootstrap | one `W` for every statistic of a seed; `pct` not recomputed per resample; `m_a` and π re-estimated per resample | A2.4–A2.6 |
| T18 | **pure step-concentration** | `AUROC_strat,pct` within 4·sd of **0.5**, while raw `AUROC_strat` is in (0.55, 0.70) | calibration (decision 6) |
| T19 | H is not read by any gate | a fixture where H = 0 and one where H = 1, same `AUROC_strat,pct` CI: identical label, class and exit | A2.1, A2.6 |

**Mutations for the battery** (each killed by at least one test): ties at 0 or 1 instead of ½;
`pct` by ordinal rank instead of midrank; `pct` over labelled slots only (NaN cells dropped from
`n_t`); step dropped when any Δ is NaN; `pct` recomputed per resample; `w_a` = all cells; an
excluded bin counted as 0.5; `n_t = M` under A1.3; gating on raw `AUROC_strat`; `y` using `≥ τ`;
seed 0's τ used for every seed; τ recomputed on run cells; H or `R_H` re-entering the label;
`CEILING` not counted as `UNRESOLVED`; rows 5 and 6 swapped; C8 anchored at line start (fails on
the real rulings); C8 still requiring `HARM_RATIO_STAR`; C8 accepting `R-*-rho-star*`; C11
disabled; pooled across seeds gating.

---

## Part C: the decisions, and what they settle

| Open issue (v1) | Decision | Where applied |
|---|---|---|
| OI-1: H near-tautological; "harm" a misnomer | H does not gate; reported as the argmin-hit rate with its limit; consequence deferred to B2 | A2.1, A2.6, A2.8 (R_H terms and `HARM_UNDEFINED` removed), A2.13 (`HARM_RATIO_STAR` out of C8) |
| OI-2: raw `AUROC_strat` moved by step-level concentration | gate on `AUROC_strat,pct` at 0.85, per seed, through the CI; raw reported | A2.4, A2.5, A2.8, A2.11 |
| OI-3: departure from §3.2.1's named statistic | Part E note for Brendan | A2.12, Part E |
| OI-4: τ from seed 0 only | per-seed τ, measured on set E `[64,128)` | A2.2, A2.10, C12 |

**Still open (not settled by the decisions; for the reviewer):**
- **The consequence question for `r_i` itself.** B2 measures ψ̂-driven eviction closed-loop, not
  `argmin r_i` eviction (A2.6 scope note). The decision's cross-reference holds for the project's
  learned policy; it does not cover the E0d signal as an eviction rule.
- **The pct ceiling.** A continuous score cannot reach 1.0 on the matched population (A2.4, T1c);
  the measured ceiling is in A2.11. It is far above 0.85, so the gate stays reachable.

---

## Part D: fix map, disagreements, and run record

### D.1 v1 disagreements with the PM: status after the decisions

1. **`H_random = mean k_t / n_t`, not `k_t / M`.** Applied (A2.6). **Moot for gating:** H no longer
   gates; it matters only for the reported `R_H,uniform`.
2. **Zero denominator of `R_H` and the argmin tie rule.** Tie rule applied to H (A2.6).
   `HARM_UNDEFINED` is **moot and removed**; an undefined `R_H` is reported as undefined.
3. **The round table's AUROC null is the identity** at one slot per age. Still applies; the null is
   the within-step shuffle (T2).
4. **The round table's test 4.4(4).** Still applies, re-stated: the gated pct form is invariant to
   per-step rescaling (T4a); the raw form is not (T4b).
5. **Truth line** ("answer-token NLL" vs `loo.py`'s sentence-mean NLL). Still applies (A2.2).
6. **NaN resample cells.** Re-stated for the gate: under the pct form a NaN donor drops only its own
   cell from the AUROC, never the step and never another slot's percentile (A2.4). The
   complete-step rule remains for the reported H only.
7. **Additions not in the PM's list:** C10′ (kept, justified in A2.8); row 5 `AGREE_VIA_RANK`
   (re-based on `AUROC_unstrat,pct`); `TAU_ZERO[s]`; C11; **C12 (new)**; `STEP_SENSITIVE` (new,
   reporting only). "Row 6 fires on either bar" is **moot** (one bar).

### D.2 Anything that contradicts a decision (reported, not overridden)

None of the evidence below overrides a decision. Each item is reported for the reviewer and the PM.

1. **"Perfect proxy at 1.0" holds only for a binary or one-critical-per-step proxy.** The binary
   proxy `r = y` scores exactly 1.0000 on every seed. A perfect proxy with continuous noise,
   `softmax(10y + ε)`, scores 0.9872 / 0.9875 / 0.9876 on the real labels. On Q-steps with no
   critical cell (222 / 232 / 239 of 720 / 724 / 740 steps), some negative holds `pct = 1`. It ties
   or beats positives of its age. Restricted to `k_t ≥ 1` steps the same proxy scores 0.9985 /
   0.9989 / 0.9992. The gate stays reachable, 0.137 above the bar. Part B pins 1.0 on the binary
   proxy (T1a) and on the one-per-step profile (T1b), and pins the ceiling below 1 separately
   (T1c). The decision's calibration line should be read that way.
2. **The ruling's rationale sentence "It controls age exactly" is not what distinguishes the pct
   form.** The raw `AUROC_strat` also gives exactly 0.5000 for age-only and recency scores, on
   every seed. What only the pct form does is remove step-level concentration. The pure
   step-concentration score gives pct 0.5008 / 0.4963 / 0.4995 against raw 0.6458 / 0.6534 /
   0.6534. The rescaled and step-temperature scores leave pct unchanged to 4 decimals (0.9203 /
   0.9176 / 0.9070) while raw moves. So the decision's evidence holds. The rationale's first
   clause is true of both forms.
3. **The pct form attenuates content when `r_i` also carries age.** At content `a = 2.0`:
   - without an age term, pct is 0.9203 / 0.9176 / 0.9070;
   - with `+ u(age)`, pct is 0.8915 / 0.8750 / 0.8702;
   - raw is 0.9340 / 0.9376 / 0.9302 with the age term, 0.9326 / 0.9366 / 0.9284 without.

   At `a = 1.0` the drop is 0.008–0.018. All of these scores stay above 0.85 at `a = 2`, but an
   `r_i` with real content near the bar *and* an age component would read `DISAGREE` under pct
   where raw reads `AGREE`. `STEP_SENSITIVE` (A2.5) makes that visible; it changes no exit.
   **Bearing on the decision:** it makes the gate more conservative for age-steered `r_i`. It does
   not make it permissive. Whether that is intended is the PM's to confirm. This draft does not
   change it.
4. **The consequence question is not fully covered by B2.** B2's PREREG (`run/b2-psi-probe`,
   `ed7fce8`, §7) evicts by `argmin ψ̂` (the fitted probe), FIFO, age-only, random, oracle and
   kind-oracle. It has **no arm that evicts by `argmin r_i`**. The deferral covers the project's
   learned policy, not E0d's `r_i` used as an eviction rule. Nothing pre-registered measures the
   latter. A2.6 states this scope. The deferral stands as decided.
5. **The C8 key form of the runner cannot read either ruling.**
   - The runner's `_RHO_LINE` is anchored at line start: `^\s*[`*]*KEY…`, in `run.py` at
     `4918f2d`, line 165.
   - A copy of it keyed on `AUROC_STAR` finds **no match** in either ruling draft. The key sits in
     a table cell in `R-2026-09-27-e0d-statistic.md` and in a list item in `-amended.md`.
   - The unanchored form in A2.13 finds `0.85` in both files. `HARM_RATIO_STAR: 0.5` is found only
     in the first file.
   - This was measured by this researcher on 2026-09-29, re-checking the predecessor's claim.
   - This does not contradict a decision. It means C8 as written would exit 3 on the ratified
     rulings. A2.13 therefore changes the regex, and T14 pins it on copies of the real texts.
6. **A null near the edge of its band.**
   - The within-step shuffle scores 0.5063 / 0.5087 / 0.4683.
   - Seed 2 is about 2.2 sd below 0.5 at the 64-document sd (≈ 4 × 0.0036). That is within the
     4-sd band T2 uses, but T2 must use the sd at the fixture's own n, not the n = 1024 sd. A
     reading of "8.8 sd" against the 1024 sd is wrong.
   - This is recorded so the test is written correctly. It is not evidence against the statistic.

### D.3 Run record

| script | what | how run | rc | output |
|---|---|---|---|---|
| `m1`–`m4` | v1 draft measurements (seed 0) | unchanged; see v1 draft | 0 each | `m1.log`–`m4.log` |
| `m5_tau_seeds.py` | seeds 1, 2 LOO (resample + zero) on set E `[64,128)`; per-seed τ for 0, 1, 2 | `orchestrator.slot run --lane cpu-det` (predecessor; two earlier slot-busy / slot-wait attempts are in `e0d-a2-final/superseded/`) | `m5.rc`: 0 | `m5_out.json`, `m5.log`, `tau_cells_seed{1,2}.npz` |
| `m6_pct_calib_power.py 0 1 2` | pct calibration and power per seed; synthetic scores on real per-seed labels; no model | `orchestrator.slot run --lane cpu-det --slots 1 --wait 3600 --item E0D-AMD2-FINAL` (this researcher, 2026-09-29) | `m6.rc`: 0 | `m6_out_012.json`, `m6.log` |
| `m7_power_summary.py` | arithmetic on `m6_out_012.json` only (z-values, detectable θ) | bare `.venv/bin/python` (no data read, as `m4`) | `m7.rc`: 0 | `m7.log` |

**Cells-file sha256, recomputed by this researcher with `shasum -a 256` (rc 0) and equal to
`m5_out.json`:**
- seed 0 `e0d-a2/step1_cells.npz`: `1338de4bdc5fbf9d33c8bd37abb34532535d2cd1e77f18e4deb87c357a65f15c`
- seed 1 `e0d-a2-final/tau_cells_seed1.npz`: `b683dd1ba1c105a52edc440dd300e2ada9c090e1fcee4fa22740756935f6cd6c`
- seed 2 `e0d-a2-final/tau_cells_seed2.npz`: `4c55cc601c70bee4aebf79c331fe6984601feb61a63df0ed99c77e4991156197`

`m6` also asserts each sha against `m5_out.json` on load (`load()`); it did not raise.

**Housekeeping.** The predecessor's partial v2 (placeholders unfilled, 2026-09-27 19:12) was
moved, not deleted, to `reviews/superseded/E0D-AMENDMENT-2-FINAL-v2.partial-2026-09-27T1912.md`.
`E0D-AMENDMENT-2-FINAL-DRAFT.md` (v1) is untouched.

---

## Part E: spec-correction note (for Brendan to write himself; not applied)

> **Correction NN — E0d gates on stratified within-step-percentile AUROC; Spearman ρ reported.**
> Spec §3.2.1's "Report Spearman ρ(`r_i`, LOO Δloss)" and §6's E0d prediction "High ρ" no longer
> define E0d's pass/kill reading. E0d gates, per seed, on `AUROC_strat,pct`: the age-stratified
> AUROC, for LOO-critical cells `y = 1[Δ_resample > τ_s]`, of each slot's within-step percentile
> rank of gated `r_i` (bar 0.85, through the 95% document-cluster bootstrap CI; every seed must
> pass). Spearman ρ is reported, never gating. The rate at which `argmin r_i` lands on a critical
> slot is reported as the argmin-hit rate and does not gate; eviction consequence is measured
> closed-loop by B2. The thresholds, the per-seed τ and the age bins are those of
> R-2026-09-27-e0d-statistic, its amendment, and E0d PREREG Amendment 2. **Why:** with about 5%
> signal-bearing cells, a perfect-label score's Spearman ρ against continuous LOO Δ is capped at
> √(3p(1−p)) ≈ 0.36 (measured 0.363), so "High ρ" was unreachable by construction; and eviction is
> a within-step argmin, so the gated statistic reads each slot's within-step standing, which a
> step-level concentration of `r_i` cannot move. §3.2.1's clause "If they disagree, LOO is truth
> and `r_i` is a confound" is unchanged.
