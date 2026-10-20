---
run_id: newcomer-bakeoff
item: "ADR-0009 sign-off pack §4.3 (~/Documents/RSR-2026-09-29-day/ADR-0009-SIGNOFF-PACK.md), the newcomer bake-off. Extends B2 (run/b2-psi-probe @ b1c34a6)."
question: >-
  Offline, at ckpt3000 and gamma = 0.9, which newcomer design avoids the harm B2 found in psi-U
  (it evicts new pending asserts before their query)? Arms: psi-C, psi-U, psi-U + a hard grace rule
  (g = 1, 2, 4), psi fitted to the R1 shadow target (K = 40, lambda_shadow = 0.5, rank 0), psi-U+ and
  psi-C+, and references.
class: "MEASURE-THEN-DECIDE. Offline probe evidence only. No build recommendation is made by this run, whatever it shows."
substrate: "fresh-stream arm B, ckpt3000 only, FROZEN, read-only (T0)."
seeds: [0, 1, 2]
gamma: 0.9
documents:
  FIT_TRAIN: "[920000, 920264) per seed -- B2's phase-A fit documents, reused (B2 TBD-1)"
  FIT_VAL:   "[936000, 937024) per seed -- B2's phase-A validation documents, reused"
  EVAL_NB:   "[990000, 992048) per seed -- FRESH; N_E = 2048, all used"
thresholds:
  delta: "B2's phase-A delta per seed, unchanged (runs/b2-psi-probe-fit, decisions.delta[0.9])"
  outcome_rule: "B2 PREREG §9.5 as amended by A1.13, unchanged"
  bootstrap: "2000 per-document resamples, paired across every arm, percentile 95% CI, torch.Generator seeded 20260927 + seed (B2 §9.2)"
---

# PREREG: the newcomer bake-off (offline, ckpt3000, γ = 0.9)

**Written 2026-09-29 by a model** (Claude Opus 5.5, an `rsr-researcher` session, overnight
dispatch). **Not written by Brendan.** It is committed **alone**, before any bake-off code and
before any bake-off computation. Nothing in it may change after its data exists; a defect found
later is reported as a defect, never fixed here.

**Sources.** ADR-0009 sign-off pack §4 (§4.3 is the design, §1.2 and §2.1–2.2 the evidence);
ADR-0009 (`origin/docs/adr-learning-path` @ `238663a`, L3/L4 table rows and "The shadow buffer's
role"); B2's PREREG, RESULTS, `run.py` and `results_report.py` at `b1c34a6`.

## 0. Three statements that no number in this run can change

1. **Whether the grace rule is acceptable under the age prohibition is OWNER-ONLY, whatever the
   numbers say.** CLAUDE.md's letter excludes age from ψ̂; its stated reason ("makes the vacuity
   failure mode invisible") covers any path by which age sets the decision. Arm (c) reads age in
   the argmin. Measuring it is not a ruling that it is allowed.
2. **The result is offline probe evidence.** Every ψ̂ is a closed-form ridge on FIFO-world
   captures, run as a fixed eviction rule. It says **nothing about on-policy behaviour after
   `T_warm`**, where the censoring is the learned policy's own (§7.2), and nothing about online MC
   training at μP rates. B2's asymmetry holds: an offline harm is evidence; an offline safety is
   necessary, not sufficient.
3. **No build recommendation is made by this run.** The decision table below says which arm
   lands where. It does not say what to build, sign or implement. The L3/L4 items stay Brendan's.

Also out of scope: Kintsch & van Dijk, E7, any scaling claim, any `rsr.constants.record()` call,
any change to a frozen constant, the spec, `spec-corrections.md`, §15 or `docs/owner/rulings`.

## 1. Substrate and what is reused

- **Checkpoints:** fresh-stream arm B `B/seed{0,1,2}/ckpt-003000.pt`, loaded exactly as B2 does
  (`load_checked`: sha256-pinned to B2 §2's ckpt3000 pins, `restore_rng=False`, eval, `no_grad`,
  `M = 16`, `S = 48`, `d = 128` asserted from the checkpoint and the fresh-stream manifest).
- **T0.** `shasum -a 256 -c ~/rsr-substrate/2026-09-27/MANIFEST.sha256`, run from the repo root,
  must give rc 0 and 435 OK **at the start and at the end** of the run (also checked in-process by
  B2's `t0_checked`). Any failure is exit 3.
- **B2's phase-A fits are reused, not refitted,** for ψ̂-U, ψ̂-C, ψ̂-U⁺, ψ̂-C⁺ and the age-only heads
  (U) and (C), with B2's selected λ, B2's `ref` choices, B2's δ and B2's kind-oracle class means.
  They come from `FIT_TRAIN`/`FIT_VAL`, not from B2's EVAL. The files are pinned by sha256
  (mismatch → exit 3):

  | file (`.worktrees/b2-psi-probe/runs/b2-psi-probe-fit/phaseA/`) | sha256 |
  |---|---|
  | `ckpt3000-seed0.pt` | `0227d80a2f77d52ff84b4311250636ce55e1509ed5a89fb833b3f406336a4846` |
  | `ckpt3000-seed1.pt` | `cdb2c3640a584dcac3dbb22a20025df21f0aefa0f238cce1d83d219e4c98c0db` |
  | `ckpt3000-seed2.pt` | `5ed0142cb0601046e790c470607f65273f802f9edf0899944f0867ac76861cfe` |

- **B2's code is imported, never copied** (as B2 imports lookahead-room): `ProbeArgminPolicy`,
  `capture_doc`, the ridge (`fit_heads` machinery: Gram, spectral path, A1.6 residual rules, A1.7
  within-step-demeaned λ selection), `run_arm`, `Logged`, `KindOracle`, `CheckedOracle`,
  `random_policy`, `paired_bootstrap`, `outcome`, `classify`.
- **The tree** is `run/b2-psi-probe` @ `b1c34a6` merged with `main` (`f7a6b10`) and
  `origin/fix/on-write-slot-index` (`6b4907f`, the LRU fix). Control C6 (§8) proves the merge did
  not change any B2 arm.

## 2. Document ranges

| Name | Ids (per seed) | Size | Use |
|---|---|---|---|
| `FIT_TRAIN` | `[920000, 920264)` | 264 | the (d) fits (same documents as B2's heads) |
| `FIT_VAL` | `[936000, 937024)` | 1024 | λ for the (d) fits; `ref` for the (d) arms; controls C5, C6, C8 |
| **`EVAL_NB`** | **`[990000, 992048)`** | **2048, all used** | every arm's in-loop evaluation |

- **`EVAL_NB` is fresh.** No document in it has been generated, run or inspected by any
  experiment. It is disjoint from every range any committed PREREG on any branch claims, checked
  by grep over every `PREREG*` on every local and `origin/` branch at this commit. In particular:
  B2's **claimed** `EVAL [940000, 980000)` (its outcomes on `[940000, 941024)` have been read, so
  that range is not reused), B2's reserved `FIT_TRAIN [920000, 936000)`, B2's `FIT_VAL
  [936000, 937024)` (used here only for fitting, as B2 did), E0d's reserved `[900000, 920000)`,
  E0d's `D_E0d [262144, 263168)`, and every row of B2's `USED_RANGES` (up to B5's continuation,
  which ends at 884,160).
- **No cross-seed aliasing:** all ids are below `1_000_003`, so the generator seeds
  `s·1_000_003 + id` are disjoint across seeds by construction; still asserted (exit 3).
- **Vocabulary closure** of every `EVAL_NB` document in the seed's `[0, 64)` map is asserted.
- **Why `N_E = 2048` and not B2's 1024.** B2's §9.4 power rule sized only B2's six gating
  contrasts (largest required n = 166, floor 1024). This run's decision questions are
  **between-arm** contrasts (grace − C, R1 − C, R1 − U) that §9.4 never sized. 2048 is fixed
  here, before any data, and is never revised.
- **`FIT_TRAIN`/`FIT_VAL` are reused and declared already used** (B2 fitted on them; their FIT_VAL
  accuracies are in B2's phase-A ledger). No outcome is read on them; they serve fitting, λ, `ref`
  and controls only, exactly as in B2.

## 3. Arms (all at ckpt3000, γ = 0.9, three seeds, on `EVAL_NB`, one document at a time, B = 1, a fresh policy per document, through `run_policy_loop`)

| Key | Arm | Definition |
|---|---|---|
| `psiC` | **(a) ψ̂-C** | `ProbeArgminPolicy`, B2's `C@0.9` fit (reused) |
| `psiU` | **(b) ψ̂-U** | `ProbeArgminPolicy`, B2's `U@0.9` fit (reused) |
| `graceU1`, `graceU2`, `graceU4` | **(c) ψ̂-U + hard grace**, g ∈ {1, 2, 4} | §4 |
| `psiR1` | **(d) ψ̂-R1** | `ProbeArgminPolicy` with the §5 fit to the R1 shadow target, λ_shadow = 0.5 |
| `psiR1L1` | *(d′) companion, non-gating:* ψ̂-R1 at λ_shadow = 1 | as (d), λ_shadow replaced by 1 (§5). **Added by this PREREG** (not in pack §4.3): it holds the rows fixed so that the counterfactual weight can be read as a dose (C = 0, R1 = 0.5, R1λ1 = 1), separately from U's row set. It does not change λ_shadow's FROZEN value; it is a diagnostic arm. |
| `psiU+`, `psiC+` | **(e)** ψ̂-U⁺, ψ̂-C⁺ | B2's `U+@0.9`, `C+@0.9` fits (reused; B2 A1.2) |
| `fifo` | (f) FIFO | `rsr.baselines.fifo.FIFOPolicy` |
| `lru` | (f) LRU | `rsr.baselines.lru.LRUPolicy` after the `fix/on-write-slot-index` merge, fresh per document |
| `ageU`, `ageC` | (f) age-only heads | B2's `age_U@0.9`, `age_C@0.9` (reused) |
| `ageR1`, `ageR1L1` | (f) age-only heads for (d), (d′) | §5, the one-hot-age ridge on the (d) / (d′) rows and target |
| `random0..4` | (f) random ×5 | B2's `random_policy(k, seed, doc_id)` (uniform over all live slots; B2's sha256 seeding) |
| `kind` | (f) kind-oracle | B2's `KindOracle`, B2's FIT_TRAIN class means at γ = 0.9, random tie (A1.10) |
| `oracle` | (f) oracle | `CheckedOracle(doc, 0.9, discounted_demand(doc, 0.9))` |

24 arm-runs per (document, seed). Every arm goes through B2's `Logged` wrapper (in-loop residency
and Σr controls), extended only to log, per eviction, each live slot's content label (§6.3).

## 4. Arm (c): the hard grace rule, a wrapper outside ψ̂

- **Rule.** At a full-memory step `t`, with live slots `k` of age `a_k = t − written_at_k`
  (the newest slot has age 1; B2's convention), a slot is **ineligible** iff `a_k ≤ g`. The victim
  is `argmin` of the unchanged inner ψ̂-U score over **eligible** slots, ties to the lowest slot
  index (B2's rule). If every live slot is ineligible, all are eligible. (At `M = 16` with
  distinct ages and `g ≤ 4` that fallback cannot trigger; it is implemented and tested anyway.)
- **Convention, declared.** The pack writes "slots younger than `g`… (age `< g`)" and also "a
  `g = 1` rule would have blocked 36.8% / 23.4% / 45.9% of ψ̂-U's decisions", which are B2's
  **age-1** victim shares. Under B2's convention (newest = age 1), `age < 1` protects nothing.
  This PREREG uses the reading that matches the pack's own numbers: **the `g` newest slots are
  protected** (`age ≤ g`).
- **Where it lives.** An experiment-local wrapper around the ψ̂-U `ProbeArgminPolicy`. It takes
  the inner score vector (already stop-grad, fp64) and applies a boolean mask in the argmin. It
  is **not** a term in the score, it is not added to `RSRPolicy` or `RSRConfig`, and it sets no
  precedent (the pack: a grace rule in RSR would need its own `reduction_to_tg()` switch, a
  `REDUCTION_SWITCHES` entry and an ADR amendment — none of that is done or proposed here).
- **Off-switch.** `g = 0` makes the wrapper return exactly the inner policy's victim; a test and a
  run-time control (C9) assert decision identity with the inner policy.
- **No differentiable path.** The wrapper never reads a tensor that requires grad, never builds
  a graph, and returns a Python int. ψ̂ itself is unchanged: its score vector is asserted equal to
  the inner policy's, and B2's no-age check (control 9) still passes on the inner head.
- **Per eviction it logs** the inner (unmasked) argmin, the chosen victim, and whether grace
  **flipped** the victim.

## 5. Arm (d): the R1 shadow target, computed offline from captures

The shadow buffer (`src/rsr/retention/shadow.py`) is a stub that raises. The target is therefore
computed offline from a FIFO capture, which is exactly B2's §5 capture (`capture_doc`, ranks 0–3,
eval, `no_grad`), re-run on `FIT_TRAIN` and `FIT_VAL` because B2 saved its fits but not its
captures.

**Definitions** (ADR-0009 L3 option R1: "the shadow `r̃` continues the return past eviction,
`G = Σ_observed γ^k r + λ_shadow Σ_shadow γ^k r̃`"; L4 (a)+(i): single insertion at the write-order
rank; pack §4.3 (d): `K = 40`, `λ_shadow = 0.5`). For one document, with `D0[t, i]` B2's
demand-if-resident at rank 0 and `resident[t, i]` B2's FIFO residency, both step-major:

- `t_e(i)` = the first step at which FIFO no longer holds `i` (FIFO holds `i` at `t_e(i) − 1`,
  reads it there, then evicts it).
- **Shadow window:** at step `t ≥ t_e(i)`, `i` is in a depth-`K` buffer iff fewer than `K`
  evictions happened after `i`'s, i.e. `n_ev(i, t) < K`, where `n_ev(i, t)` counts FIFO evictions
  at steps in `[t_e(i), t)` after `i`'s own. Under FIFO at full memory this is `t − t_e(i) < K`.
- **The shadow `r̃_i(t)`** is `D0[t, i]` — B2's rank-0 probe: `i`'s gestalt inserted **singly** in
  place of the oldest resident's, which is the write-order rank an evicted (older than every
  resident) slot would hold. That is L4 (a)+(i).
- `X_R1[t, i] = D0[t, i]` if `resident[t, i]`; `= λ_shadow · D0[t, i]` if not resident and in the
  shadow window; `= 0` otherwise; NaN for `i ≥ t`.
- **Target:** `G_R1 = discounted_returns(X_R1, 0.9)` (B2's `returns`, sum from `k = 0`,
  truncated at document end).
- **Rows: B2's C rows** (every `t ∈ [1, S)`, every FIFO-resident `i < t`). R1 re-labels the
  **pre-eviction** samples (ADR-0009: "only R1 un-censors the pre-eviction samples that the
  decision at `t_e` reads"); adding post-eviction rows is R2, which is not this arm.
- **(d′):** identical, with `λ_shadow = 1`.
- **Declared: `K` is inert on this corpus.** With `S = 48` and FIFO evicting at age 16, the shadow
  window never exceeds 31 steps, so `K = 40` truncates nothing. The count of truncated cells is
  computed and recorded (expected 0). The R1 target here is therefore "C's target plus half of the
  probe-continued demand after eviction".
- **Declared: how (d) differs from U.** U regresses on `D0` at weight 1 **on U's rows** (which add
  every post-eviction `(i, t)`); (d) regresses on the same demand at weight 0.5 **on C's rows**.
  (d′) isolates the weight (C rows, weight 1); U − (d′) is then the row-set effect.

**The fit.** B2's §6 procedure, unchanged: bilinear features `[vec(s_i c_tᵀ); s_i; c_t]`
(`p = 16,640`), fp64 closed-form ridge, one eigendecomposition of the C-row Gram on `FIT_TRAIN`
(264 documents), the λ grid `10⁻⁶ … 10⁴`, `λ' = λ·tr/p`, A1.6's residual rules (grid points with
residual > 1e-6 ineligible; the selected λ must be ≤ 1e-8, else **exit 1**), A1.7's λ by minimum
**within-step demeaned** FIT_VAL MSE over full-memory rows, ties to the larger λ, final `w` from
FIT_TRAIN alone. The same Gram also refits `C@0.9` for control C5. The age-only heads `ageR1`,
`ageR1L1` are B2's one-hot-age ridge on the same rows and targets.

**`ref` for (d) and (d′)** (B2 §9.3, unchanged): on FIT_VAL, whichever of FIFO and that arm's
age-only head has the higher all-query accuracy; a tie goes to FIFO.

## 6. Readouts (all pre-registered; every one reported whatever it shows)

### 6.1 B2's outcome rule, per arm vs its reference

For each arm X and seed s: Δ = all-query accuracy(X) − accuracy(ref_X) on `EVAL_NB`, paired 95%
CI `[lo, hi]`, with B2's δ_s. Outcome by B2 §9.5 + A1.13, **unchanged**: CI excluding its own
estimate → UNRESOLVED; **WIN** iff `lo > 0`, `Δ ≥ δ`, the `gap_2_to_M` Δ lower bound `> −δ` and
the arm − random-mean lower bound `> 0`; **LOSS** iff `hi < −δ`; **EQUIV** iff `−δ < lo` and
`hi < δ`; else **UNRESOLVED**. Order WIN, LOSS, EQUIV, UNRESOLVED.

| Arm | ref |
|---|---|
| ψ̂-U, (c) grace g ∈ {1, 2, 4}, ψ̂-U⁺ | B2's `ref_U` (seed 0: age-only (U); seeds 1, 2: FIFO) |
| ψ̂-C, ψ̂-C⁺ | B2's `ref_C` (FIFO on every seed) |
| (d) ψ̂-R1, (d′) ψ̂-R1λ1 | `ref_R1`, `ref_R1L1` from FIT_VAL (§5) |

Also reported: every arm's accuracy on all / `gap_2_to_M` / `gap_gt_M` / `gap_eq_M`, the
model-free residency of every arm, and **B2's §9.6 truth-table class recomputed on (ψ̂-U, ψ̂-C)**
and on (ψ̂-U⁺, ψ̂-C⁺) on `EVAL_NB`, labelled **replication, non-binding** (it does not replace
B2's classification).

### 6.2 Newcomer and pending-assert evictions (per arm, per seed)

- **age-1 pending-assert evictions:** count of evictions whose victim has age 1 and is an assert
  whose query is still ahead (B2's `status == "pending"`), and its share of all evictions.
- **pending victims within gap M:** count of evictions of a pending assert `w` (query `q > t`)
  with `q − w ≤ M` — B2's `results_report.replay` definition (`pend_gap["le_M"]`).
- The victim age histogram (1 … S−1), the victim kind/status counts, the ψ̂ margin summary (B2's
  attribution).

### 6.3 §7.1 vacuity on the EFFECTIVE rule

Rows: every live slot at every full-memory eviction step of the arm's own `EVAL_NB` rollout.
Variables: `y` = 1 if the slot is the victim, else 0; `a` = its age; `c` = its **content label**
∈ {`assert:pending`, `assert:querying`, `assert:answered`, `query`, `filler`} (the generator's
kind, and an assert's status at that step: B2's descriptive labels, never an input to any ψ̂).

- **Partial ρ(victim indicator, age | content)**, primary: rank `y` and `a` over all rows of that
  (arm, seed) (average ranks); demean both ranks and the one-hot of `c` within each
  (document, step); regress each demeaned rank on the demeaned one-hot (OLS, no intercept); ρ is
  the Pearson correlation of the two residuals.
- Also reported: the same without content (within-step Spearman), and, for every ψ̂ arm, the
  **score-level** partial ρ(ψ̂, age | content) with ψ̂ in place of `y` (the spec §7.1 test's own
  object). For a grace arm the score is the inner ψ̂-U, so the gap between its score-level and
  decision-level ρ is exactly the part of the rule §7.1's score test cannot see.
- **No threshold is read.** The spec's 0.7 is a line on a score; a binary victim indicator has a
  different scale (FIFO, pure recency, cannot reach 1). FIFO's and LRU's values are reported beside
  every arm as the recency references.
- **Age-only-head contrast** (§7.1 test 2): Δ(X − age-only head of X's target) all-query, paired CI:
  (b), (c), (e: U⁺) against `ageU`; (a), (e: C⁺) against `ageC`; (d) against `ageR1`; (d′) against
  `ageR1L1`. Labelled BETTER (`lo > 0` and `Δ ≥ δ`), SAME (`−δ < lo`, `hi < δ`), WORSE (`hi < −δ`),
  else UNRESOLVED (A1.13 applies).
- **LRU contrast** (§7.1 test 3): Δ(X − LRU) all-query, paired CI, same four labels; "beats LRU"
  is reported as `lo > 0`.

### 6.4 Grace flip share (c)

Per g and seed: the share of the grace arm's own evictions where the eligible-argmin victim
differs from the unmasked ψ̂-U argmin on the same state. Reported with its count. No threshold is
invented: RESEARCH-CONTEXT §4.4's "if grace flips a large share, grace is the policy" is quoted
beside it and not converted into a number.

### 6.5 ADR-0006 displacement histogram

Per arm and seed, the histogram of `displacement = victim_rank − 1` from
`rsr.retention.instrumentation.rank_shift` (FIFO is identically 0).

## 7. The decision table (per seed with B2's δ_s; A1.13 applies to every CI)

Per-seed contrast labels, for a paired contrast X − Y: **BETTER** (`lo > 0` and `Δ ≥ δ`),
**SAME** (`−δ < lo` and `hi < δ`), **WORSE** (`hi < −δ`), else **UNRESOLVED**.

**DQ1. Does grace buy anything over (a)?** For each g, the label of `grace_g − ψ̂-C` on each seed.
- **YES** if some g is BETTER on ≥ 2 seeds and WORSE on none.
- **NO** if no g is BETTER on any seed.
- **MIXED** otherwise.

Reported beside it, not part of the rule: each grace arm's §9.5 outcome vs `ref_U` (does grace
remove ψ̂-U's LOSS?), its flip share, and its decision-level vs score-level partial ρ.

**DQ2. Does (d) land near (a) (safe) or near (b) (harmful)?** Per seed:
- **NEAR-C** if `R1 − C` is SAME and `R1 − U` is not SAME;
- **NEAR-U** if `R1 − U` is SAME and `R1 − C` is not SAME;
- **INDISTINGUISHABLE** if both are SAME (U and C within δ of each other on that seed);
- **BETWEEN** if `R1 − U` has `lo > 0` and `C − R1` has `lo > 0` and neither is SAME;
- else **UNRESOLVED**.

Class, first match wins:
1. **HARMFUL (near b)** if (d)'s §9.5 outcome is LOSS on any seed (B2's row-1 criterion), or
   NEAR-U on ≥ 2 seeds.
2. **SAFE (near a)** if (d) is LOSS on no seed and every seed is NEAR-C or INDISTINGUISHABLE.
3. **INTERMEDIATE** otherwise.

Also reported: the position `π_s = (acc_R1 − acc_U) / (acc_C − acc_U)` with its bootstrap
percentile CI (0 = at U, 1 = at C; flagged undefined when the denominator's CI contains 0).
**What it settles for L3/L4:** only whether R1 at `λ_shadow = 0.5`, rank 0 (L4 (a)+(i)) carries
ψ̂-U's offline harm. It decides no L-item; it is the measurement the pack names as their decider.

**DQ3. Does the harm travel with the counterfactual target?** Harm seeds `H` = the seeds on which
ψ̂-U is LOSS on `EVAL_NB` (§6.1). If `H` is empty, DQ3 is **NOT ASSESSABLE**. For `s ∈ H`:
- **TRAVELS_s** if `R1λ1 − C` is WORSE (full-weight counterfactual on C's own rows is worse than C
  by more than δ) **and** the point estimates are ordered `acc_C ≥ acc_R1 ≥ acc_R1λ1`;
- **STAYS_s** (does not travel) if `R1λ1 − C` is SAME;
- else unresolved on that seed.

Class: **YES** if TRAVELS on more than half of `H`; **NO** if STAYS on more than half of `H`;
**MIXED** otherwise. Reported beside it: the row-set effect `U − R1λ1`; ψ̂-U vs ψ̂-U⁺ (the k = 0
term); and the §6.2 counts along C → R1 → R1λ1 → U.

## 8. Controls (any failure → exit 3 and no class; C10 → exit 1)

1. **Checkpoints and T0.** Checkpoint sha256 = B2's pins; T0 re-verified at start and end
   (in-process `t0_checked`, and the literal `shasum -c`, rc 0 and 435 OK, recorded by the operator).
2. **Ranges.** §2's disjointness, aliasing and closure, before any model is loaded.
3. **B2's fits.** The three phase-A files match §1's sha256; their `decisions.ref`,
   `decisions.delta` and heads are read, never recomputed.
4. **Capture.** B2 §11 controls 4–5 on the new capture: identity probe ≤ 1e-6 at ranks 0–3;
   `|Σr − 1| ≤ 1e-5` at every full-memory step and probe row.
5. **The (d) pipeline is B2's.** Refitting `C@0.9` on the new capture with the §5 Gram must
   (i) select B2's λ, (ii) reproduce B2's within-step demeaned FIT_VAL MSE at that λ to relative
   1e-6, and (iii) give the same argmin as B2's stored `C@0.9` on ≥ 99.9% of FIT_VAL full-memory
   steps. (The max |w − w_B2| and whether it is bit-identical are reported.)
6. **The merged tree reproduces B2.** On FIT_VAL, FIFO, ψ̂-C and ψ̂-U (B2's stored `w`) give pooled
   all-query accuracies **exactly equal** (`==`) to B2's `decisions.acc[0.9]`.
7. **In-loop.** Every arm: residency equals the model-free replay (B2 §11.7); Σr ≤ 1e-5 on the
   ψ̂ and age arms; one policy per document; the oracle's demand is the document's; ψ̂ finite.
8. **Determinism.** The first 8 FIT_VAL documents, rerun for `psiR1`, `psiR1L1`, `graceU1/2/4`
   and `lru`, reproduce every victim and every `correct` flag.
9. **No age in ψ̂; grace outside ψ̂.** B2's no-age check on the (d)/(d′) heads; the grace
   wrapper's score vector equals the inner ψ̂-U's; `g = 0` reproduces ψ̂-U's victims on the first
   8 FIT_VAL documents.
10. **Ridge residual** (A1.6): the selected λ's relative residual ≤ 1e-8, else **exit 1**.

**Exit codes.** 0: ran and every control passed; a classification was reached (every DQ outcome,
including HARMFUL, NO or NOT ASSESSABLE, is exit 0). 1: a measurement defect (ridge residual,
non-finite ψ̂ or target, a raise inside a measurement). 3: did not run, or a control failed.
Exits 2 and 5 are not used. `$?` is read directly (`cmd > log 2>&1; rc=$?`).

**Crash safety.** EVAL writes one unit per (seed, document) atomically (temp file, fsync,
`os.replace`) under a key that includes the code's sha256 and the fits' sha256; a relaunch reads
matching units and never reruns them; a unit under any other key is exit 3.

## 9. Author's expectation (before any bake-off number; allowed to be wrong)

- **Replication:** ψ̂-U LOSS on seeds 0 and 1 again, ψ̂-C EQUIV on all three (B2's pattern);
  about 70%.
- **DQ1 (grace over (a)):** NO about 60%, MIXED 30%, YES 10%. Grace should cut ψ̂-U's age-1
  evictions to zero by construction; the expectation is that the harm partly moves to ages 2–5
  (g = 1) and that g = 4 approaches, but does not beat, ψ̂-C. Flip shares of roughly 25–45% at
  g = 1 (B2's age-1 shares).
- **DQ2 ((d)):** INTERMEDIATE 40%, HARMFUL 35%, SAFE 25%. Reason: the half-weight continuation
  mostly raises the targets of old resident slots (ages 12–16), which is the "old is valuable"
  direction the pack's reading 3 blames.
- **DQ3:** YES 55%, NO 25%, MIXED or NOT ASSESSABLE 20%.
- **Vacuity:** every grace arm's decision-level partial ρ exceeds its score-level ρ's recency
  signal; ψ̂-C's decision-level ρ is closer to FIFO's than ψ̂-U's.

## 10. "Already seen" declaration

The author has read, before this commit: the sign-off pack in full; ADR-0009 (L3–L6 rows and the
shadow-buffer section); B2's PREREG, RESULTS (every number in it, including its per-seed
attribution and its EVAL documents' outcomes), RESULTS-phaseA and phase-A files (refs, δ, λ, FIT_VAL
accuracies); B0's ceilings as quoted in the pack and B2's addendum; E0d's digest as quoted in the
pack; the first record of B2's ckpt3000 seed-0 per-eviction log (to check it holds full ψ̂
vectors: it does, but grace changes every later memory state, so the grace arms need fresh
rollouts regardless).

**Not seen by anyone:** any document of `EVAL_NB = [990000, 992048)` for any seed, any R1 or R1λ1
fit, any grace rollout, and any LRU rollout on this tree.

## 11. What this does not establish

Nothing about on-policy behaviour after `T_warm`, online MC, μP rates, `b`, ν, β or the real
shadow buffer (L5's query stash is not exercised: the probe recomputes the forward). One corpus
(S0-03), `M = 16`, one width, FIFO-trained checkpoints, one checkpoint. No scaling claim. No build
recommendation. The acceptability of the grace rule is OWNER-ONLY.

## Erratum E1 (2026-09-29, pre-data; append-only)

**Written by a model** (the same session), before any bake-off computation: no `runs/newcomer-bakeoff*`
exists, no `EVAL_NB` document has been generated, and nothing has been fitted. Nothing above this
heading is edited.

- **§3's sentence "24 arm-runs per (document, seed)" is a miscount of §3's own table.** The table
  lists 22 keys: `psiC`, `psiU`, `graceU1`, `graceU2`, `graceU4`, `psiR1`, `psiR1L1`, `psiU+`,
  `psiC+`, `fifo`, `lru`, `ageU`, `ageC`, `ageR1`, `ageR1L1`, `random0..4` (5), `kind`, `oracle`.
  **The table governs: 22 arm-runs.** No arm is added or removed by this erratum.
- Found by the build's own test (`test_the_arm_set_is_the_prereg_table_s_22`), which asserts the
  table's 22 keys.
