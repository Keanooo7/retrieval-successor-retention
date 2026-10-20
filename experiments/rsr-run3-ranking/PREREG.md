---
run_id: "rsr-run3-ranking (two run ids: rsr-run3-2a for the offline fit and deploy, rsr-run3-2b for the on-policy run)"
item: "PLAN-2026-10-19 phase 2: the ranking-loss test. Attempt 3 at a learned content-only eviction policy, after RSR run 1 and run 2"
question: >-
  (2a) Fitted offline on B0-CF's uncensored FIFO-world LOO returns with a within-step pairwise
  ranking loss instead of L9's squared error, and selected on the sealed TUNE split by the cap, does
  run 2's content-only bilinear psi_hat beat FIFO when deployed frozen on the already-used EVAL?
  (2b, only if 2a is WIN on all 3 seeds) Trained on-policy with run 2's harness, changed only in the
  loss, does it beat FIFO on the fresh sealed EVAL2?
class: "GATED (Tier G, review policy v2): one review of this PREREG and the code together, at most one confirming round; full suite at the exact launch head; rsr-verifier once per run (2a, 2b). 2a's per-seed labels gate 2b; 2b's class (BEATS-FIFO / FAILS / MIXED) goes to Gate A. Every other number is descriptive."
not: "Not E1, not E2, not a thesis change (R-2026-10-19-l9-ranking-loss, ruling item 5). Does not re-open run 1, run 2, offline-psi, diagnostic A or the LS check, and changes none of their labels. No age in psi_hat. No gradient into the transformer or W_sent. b off, shadow off. No scaling claim. No cognitive claim. §15 and release condition 7 untouched. Falsifier 6 stays deferred to E1."
attempt_context: "PLAN §2 counts phase 2 (2a and 2b together) as attempt 3 at a learned content-only policy; run 1 and run 2 (on-policy, gated) were attempts 1 and 2. Three further OFFLINE fits of the same class under L9's squared error exist and are disclosed beside the count, not inside it: offline-psi (deployed on EVAL), diagnostic A and the LS check (fit only, U documents). Every configuration phase 2 fits is reported (§4.3, §5.2)."
seeds: "[0, 1, 2]"
sealed_ranges_source: "experiments/post-fails-controls/PREREG.md front matter, keys sealed.TUNE and sealed.EVAL2 (machine-read; never retyped in code)"
labels: "SEEN / UNSEEN per §8"
status: "COMMITTED ALONE by the PM on 2026-10-20, before any phase-2 code or data (PM review §12). From that commit on, amendments are appended, dated and committed alone; the text above them is never edited."
---

# PREREG: RSR run 3, the ranking-loss test (phase 2)

**Drafted 2026-10-20 by a model** (Claude Opus 5.5, an `rsr-researcher` session dispatched by the
PM). **Not written by Brendan.** Authority: PLAN-2026-10-19 §0 (phases 0, 1 and 2 GO; phase 2 under
Gate P2) and §2 (phase 2, "Fixed now"); the ruling draft
`~/Documents/RSR-2026-10-19-review/R-2026-10-19-l9-ranking-loss.DRAFT.md` (the L9 change, for this
test only). Every place where this file fills a choice PLAN left open, or departs from it, is listed
in §10 with its reason; none loosens PLAN.

**Gate P2 has been applied by the PM.** Item 1a is verified (`verify/post-fails-controls-1a` @
`2f62a93`); FactFiller's reading is `WIN_ALL_SEEDS` (`runs/post-fails-controls-1a/ledger.json`, key
`summary`, `readings.factfiller.pattern`; `experiments/post-fails-controls/RESULTS-1a.md:28`). So
phase 2a runs.

Base: `build/pfc-1a` @ `a1d7934` (phase-1 PREREG with Amendment 1a-S, item 1a and its RESULTS).

## 0. Authority, and the one change

### 0.1 Fixed by PLAN §2 (verbatim; this file cannot loosen it)

> - **Same as run 2 except the loss:**
>   - the content-only `BilinearValueHead` (16,640 features);
>   - no age in ψ̂;
>   - no gradient into the transformer or `W_sent`;
>   - frozen arm B ckpt3000;
>   - LOO-resample Δ at γ = 0.9;
>   - b off, shadow off.
> - **The only change is the loss.**
>   - Primary: a within-step pairwise logistic loss over slot pairs ordered by G, with the pairs that
>     involve the step's true bottom-k up-weighted.
>   - At most ONE secondary loss, pre-specified.
>   - The PREREG fixes k, the weighting and a small hyperparameter grid.
>   - The selection criterion on the **fresh tuning split** is the cap, never EVAL.
> - **The default config must reproduce run 2.** A test asserts that the loss switch off is
>   bit-equal to run 2's L_MC path, the same pattern as run 2's test against run 1.
> - **Attempt count:** this is attempt 3 at a learned content-only policy (after run 1 and run 2).
>   Every configuration fitted is reported.

### 0.2 Unchanged from run 2, by citation

Everything `experiments/rsr-run2/PREREG.md` (and, through it, run 1's PREREG and amendments) fixes
binds here wherever this file is silent: substrate and pins (fresh-stream arm B `ckpt-003000.pt`,
run 1's SUBSTRATE, `experiments/rsr-run1/run.py:186-201`), pool `[993000, 993256)`
(`experiments/rsr-run1/run.py:152`), vocabulary map, constants (d 128, batch 16, S 48, M 16, lr 1e-3,
φ weight decay 0.01, ν 0, β 1, b off, shadow off, γ 0.9, O5 eval; `experiments/rsr-run2/PREREG.md:42-49`),
φ init (`loop.phi_generator(seed)`), μP groups, permutation-epoch sampling, `T_warm` 16, the target
`loo_resample` (`experiments/rsr-run2/PREREG.md:52-61`), the held-out procedure, the bootstrap
(`R1.boot_weights(n, seed, 2000)`, `experiments/rsr-run1/run.py:2345`), δ_seed
(`experiments/rsr-run2/PREREG.md:95-99`, machine-read by run 2's `read_delta`), the label rule
(`experiments/rsr-run2/run.py:206-219`) and the class rule (`experiments/rsr-run2/run.py:222-231`),
and launch discipline (clean pinned head, tmux, rc files).

### 0.3 The one change

A new retention-loss switch, `loss ∈ {"mc", "rank"}`, **default `"mc"`**. With `"mc"` every code path
is run 2's, bit for bit (§3 T1). With `"rank"` the squared error at
`src/rsr/retention/rsr.py:892` (`RSRPolicy.retention_loss`, L9: unweighted, sum-then-mean; signed in
`docs/owner/rulings/R-2026-09-30-adr0009-signoff.md`) is replaced by §2's loss, computed on the same
rows (`mc_targets`, `rsr.py:756`), through the same detach site (`rsr.py:886-891`). Nothing else in
the training step changes. The switch is label-visible (ADR-0009 L12 as signed): RSR arm labels gain
`loss=rank,rank_k=3,rank_omega=<ω>` after run 2's `target=loo_resample`.

**One further difference is forced by PLAN and is not a loss change:** 2b's step count is fixed by
§5.3's rule from a smoke, not copied from run 2. A matched-step descriptive contrast at k = 3000 is
reported (§5.6 item 3).

## 1. Document ranges and seals

| range | ids (per seed's generator) | used by phase 2 | status at phase 2 |
|---|---|---|---|
| U-train | offline-psi `u_split` train positions of U (`experiments/offline-psi/run.py:273-277`) | 2a fit | already inspected |
| U-val | offline-psi `u_split` val positions | 2a descriptive only (§4.6) and control R-C2 | **spent for selection** (psi-ls-capacity F10, `experiments/psi-ls-capacity/RESULTS.md:140-141`) |
| POOL | `[993000, 993256)` | 2b training; 2b smoke | used (run 1, run 2) |
| EVAL | `[995000, 997048)` | 2a headline; 2b ES reproduction control on its first 32 ids | **already used** (run 1, run 2, Expire-Span, offline-psi, run2-vs-random, phase-1 1a) |
| **TUNE** | `[997200, 997712)`, 512 per seed (`experiments/post-fails-controls/PREREG.md:16,64`) | 2a selection | **sealed; first read at 2a's select stage** |
| **EVAL2** | `[984000, 986048)`, 2048 per seed (`experiments/post-fails-controls/PREREG.md:17,65`) | 2b headline | **sealed; first read at 2b's measure stage** |

**Seal (control R-SEAL, every stage; failure = that stage's exit 3).** Phase 1's G-C1 pattern
(`experiments/post-fails-controls/PREREG.md:126`), extended to stages:

- (a) Every document constructor phase 2 can reach (`rsr.data.synthetic._generate_document`,
  `generate`, and through them `R1.docs_by_id`, lookahead-room's `documents`, the ES harness's
  document path) is wrapped for the whole process by a guard that raises `SealedRange` for any id in
  a range sealed **at that stage**:

  | stage | TUNE | EVAL2 |
  |---|---|---|
  | 2a `fit` | sealed | sealed |
  | 2a `select` (TUNE deploys) | **open** | sealed |
  | 2a `deploy` (EVAL) and `finalize` | sealed | sealed |
  | 2b `smoke`, `train` | sealed | sealed |
  | 2b `measure` (after every training process has exited with a completed footer) | sealed | **open** |
  | 2b `finalize` | sealed | sealed |

- (b) Each stage ledgers the ids it generated, by range (count, min, max), and asserts 0 in every
  range sealed at that stage.
- (c) The bounds are machine-read from the phase-1 PREREG front matter (`sealed_ranges_source`).
- (d) At each launch, phase 1's §1.1 branch search is re-run and ledgered; a new hit inside TUNE or
  EVAL2 is exit 3 and goes to the PM.
- (e) **Closure at first read** (phase-1 PREREG §1, last bullet): when a range is first opened,
  run 1's C0 logic (`R1.check_c0`, `experiments/rsr-run1/run.py:1124-1145`) is applied to it before
  any model loads: disjoint from every `USED_RANGES` entry and from POOL, upper bound ≤ `GEN_SEED_K`,
  and closed under each seed's vocabulary map (`loop.check_vocabulary_closure`). A failure is exit 3,
  and the range is not read further.
- (f) TUNE is never read in 2b. EVAL2 is never read in 2a.

## 2. The loss (`loss = "rank"`)

### 2.1 Definitions

A **group** is one `(row, t)`: a document row at step `t`. Its **pair set** `L` is:
- in **2a** (offline): `L = {i : 0 ≤ i < t}` with finite `G`, i.e. offline-psi's **`cf_all`** cells,
  its owner-approved primary (`experiments/offline-psi/run.py:348-357`). PM decision at commit (§12
  item 2): with `cf_all`, 2a differs from offline-psi's primary fit in the loss only, and the head is
  trained to rank slots older than M, which a content-based policy keeps at deployment;
- in **2b** (on-policy): the policy's own pre-write memory at `t`, exactly the rows `mc_targets`
  returns for that `(row, t)` (`rsr.py:756-800`).

**G.** `G_i` is the return regressed in the corresponding L9 path, detached:
- 2a: `G = discounted_returns(Δ, 0.9)[t][i]` from B0-CF's saved resample-knockout Δ, uncensored,
  FIFO-world, truncated at document end (offline-psi `target_of`, `experiments/offline-psi/run.py:342-345`);
- 2b: `G_i(t) = Σ_k 0.9^k Δ_i(t+k)`, run 2's on-policy censored return (`mc_returns` over the
  `written_at` grid, zeros where not live; missing donor or no real target imputed 0;
  `rsr.py:693-731, 756-800`).

**Scores.** `ŝ_i = ψ̂(s_i, c_t)` from the content-only `BilinearValueHead(d_model=128)`; `s` and `c`
detached at `rsr.py:886-891` (2b) or precomputed tensors (2a).

**Pair set.** `P = {(i, j) ∈ L × L : G_i > G_j}`, compared as float32 values as stored. **Ties**
(`G_i == G_j` exactly) form no pair. A group with `|P| = 0` contributes nothing and is counted
(`groups_no_pair`).

**Bottom-k membership (exact, no RNG; phase 1's 1c definition, `experiments/post-fails-controls/PREREG.md:343-347`).**
With `k = 3` and `n = |L|`: `b_j = 1` if `G_j` is strictly below the k-th smallest value of `G` over
`L`; `b_j = (k − #strictly below) / #tied` if `G_j` equals it; `b_j = 0` otherwise. If `n ≤ k`,
`b_j = 1` for all `j`.

**Pair weight.** `w_ij = 1 + (ω − 1) · max(b_i, b_j)`, with `ω` from the grid (§2.2).

**Group loss, normalised per group.**

    ℓ_g = Σ_{(i,j) ∈ P} w_ij · softplus(ŝ_j − ŝ_i)  /  Σ_{(i,j) ∈ P} w_ij

**Batch loss.** The mean of `ℓ_g` over the groups of the batch with `|P| ≥ 1`. This keeps L9's
structure (one term per `(row, t)`, then a mean over `(row, t)`); only the per-group functional
changes. Every step with a pair is included, underfull steps too: nothing is down-weighted by fill
(CLAUDE.md), and the step set is L9's, restricted to groups that have a pair.

**Stop-gradient and isolation.** Unchanged from run 2: only φ = {W, u} receives gradient. The age arm
(`psi_override = age_only`) uses the same loss on its age-only head (§5.2).

### 2.2 Fixed constants and the grid

| constant | value | reason |
|---|---|---|
| `k` | **3**, fixed | The eviction is the bottom-1 of a 16-slot live set, but a single-cell Δ is a noisy single-knockout difference; k = 3 is phase 1's registered bottom-k (1c) and the bottom fifth of a full memory. Fixed, not tuned, per PLAN. |
| weighting form | `w_ij = 1 + (ω − 1)·max(b_i, b_j)`, fixed | "Pairs that involve the step's true bottom-k" (PLAN), with exact fractional membership so ties at the boundary need no RNG. |
| **grid** | **`ω ∈ {2, 8, 32}`**: 3 configurations per seed, 9 fits in 2a | One dimension only. In a full 16-slot memory with no ties, 42 of 120 pairs touch the bottom-3; their share of the group's weight is `42ω / (78 + 42ω)`, so the grid spans "bottom pairs are half the weight" to "bottom pairs are nearly all of it". |
| temperature | none (the logit is `ŝ_i − ŝ_j`) | A temperature is a multiplier on ψ̂'s output, i.e. a μP output multiplier; correction 15 (`docs/spec-corrections.md` §15) and the lr are the owner's. |
| optimizer | run 2's: AdamW, lr 1e-3, φ weight decay 0.01, μP groups (`experiments/offline-psi/run.py:506-518`) | "The only change is the loss" (PLAN). Not a grid dimension. |
| secondary loss | **none** | (i) An L_MC anchor would re-add the objective the LS ridge path suggests is misaligned with the cap (`experiments/psi-ls-capacity/RESULTS.md:132-139`, a hypothesis there); (ii) the argmin is scale-free and weight decay already bounds ψ̂'s scale; (iii) a listwise bottom-1 term duplicates what the bottom-k weighting targets; (iv) one change keeps the attempt interpretable. |

## 3. Tests (written first; `tests/test_rsr_run3_ranking.py`; the full suite must pass at the exact launch head of each run)

- **T1 switch off = run 2 (PLAN; pattern `experiments/rsr-run2/PREREG.md:63-68`).**
  (a) Unit: with `loss="mc"`, `RSRPolicy.retention_loss` returns a tensor bitwise equal to run 2's
  on fixed rows (both RSR arms). (b) End-to-end: run 2's psiC seed-0 command for 20 steps at
  `728a42f` (run 2's launch head) and at the build head give identical heartbeats (apart from
  run 2's `VOLATILE_KEYS`), diagnostics, eviction records, checkpoint tensors and run_id. `src/`
  differs from `728a42f` in `policy_loop.py` and `train/loop.py` (Expire-Span wiring), so this is a
  behavioural comparison of outputs, not a code diff. (c) `RSRConfig.reduction_to_tg()` is unchanged
  and `tests/test_reduction.py` passes. (d) 2a's fit path with `loss="mc"` on cf_all and on cf_live reproduces
  offline-psi's U-val L_MC at step 3000 (control R-C2, run as a test on a 2-document fixture too).
- **T2 isolation.** With `loss="rank"`, after `backward()` on both RSR arms, every transformer
  parameter and `W_sent` has `grad is None`; the optimizer holds only φ (offline-psi's
  `optimizer_holds_only_phi`).
- **T3 no age.** `BilinearValueHead.forward(gestalts, context)` takes no age; its feature count is
  `d² + 2d` (16,640 at d = 128); with `(s, c)` fixed, permuting `written_at`/ages leaves every score
  and the rank loss unchanged; the content arm's rank path never reads `t − key`.
- **T4 the loss.** Hand-computed fixtures for `ℓ_g` (no ties; ties at the k-th value; ties
  elsewhere; `n ≤ k`; a group with no pair, excluded and counted); invariance to adding a constant to
  every score of a group; invariance of the batch loss to the order of rows; `ω = 1` reduces to the
  unweighted mean over pairs.
- **T5 sanity.** On a planted fixture whose `G` is a known bilinear function of `(s, c)`, the fit
  drives the training objective down and the within-group argmin agreement to 1.
- **T6 seals.** The guard raises `SealedRange` on TUNE and EVAL2 ids at every stage where §1 seals
  them, and lets them through only where §1 opens them.
- **T7 config.** RSR arms under `loss="rank"` have `b_enabled = False`, shadow off, `ν = 0`,
  `target = loo_resample`, γ 0.9, and labels containing `loss=rank`.

## 4. Phase 2a (offline): fit, select on TUNE, deploy on EVAL

### 4.1 Data (read, never recomputed)

- **Δ:** `.worktrees/cf-ceiling/runs/cf-ceiling/raw/B-ckpt3000-seed{s}.delta.pt`, key `Delta`, sha256
  pinned as `experiments/offline-psi/PREREG.md:49-52`.
- **FIFO-world gestalts** `s_i`, `c_t` on U: `.worktrees/offline-psi/runs/offline-psi/seed{s}/gestalts.pt`,
  through diagnostic A's `load_inputs`, sha256 pinned as `experiments/offline-psi-converge/PREREG.md:43-45`.
- **Fit set: U-train only (932 documents per seed).** U-val is not fitted on and selects nothing; it
  is kept as a descriptive held-out split so the ranking fit's U-val cap stands beside offline-psi's,
  diagnostic A's and the LS fit's on the same documents (§4.6). Rejected alternative: fitting on all
  of U would add U-val's documents but lose the like-for-like comparison with the three L_MC fits.
- **Cells:** the 2a pair sets of §2.1 (`cf_all`: every `i < t` with finite `G`), every `t ∈ [1, 48)` with a pair.

### 4.2 The fit (offline-psi's harness, imported; only the loss changes)

- Head `new_policy(seed, …)` with run 2's psiC overrides and φ init (`experiments/offline-psi/run.py:491-503`);
  optimizer `phi_optimizer` (`:506-518`); batches of 16 U-train documents in permutation epochs from
  a private generator seeded `seed` (`:598-628`). The same initial φ and sampler for every ω.
- **Convergence rule (registered; addresses diagnostic A's finding that run 2's optimizer needs
  ~10⁵ steps on L_MC, `experiments/offline-psi-converge/RESULTS.md:18,20,22`).** Every 250 steps,
  evaluate the full-batch training objective `J(k)` (the §2.1 batch loss over every U-train group,
  `no_grad`, no RNG consumed). Let `best(k)` be the lowest `J` at or before `k`. The fit is
  **CONVERGED** at the first check `k ≥ 2000` with `best(k) ≥ (1 − 10⁻³) · best(k − 2000)` and stops
  there; `K_conv = k`. If no check meets it by **200,000** steps (diagnostic A's cap), the label is
  **CAP-HIT** and `K_conv = 200,000`.
- **The fitted head is φ at `K_conv`.** No held-out split chooses the step (U-val is spent; TUNE has
  no Δ).
- φ is saved at `K_conv` and at steps 3000 and 10,000 where reached (descriptive trajectory).

### 4.3 Selection on TUNE: the cap (TUNE's first read)

For each seed and each ω, the fitted head is deployed **frozen** on the 512 TUNE documents through
the deploy path 2a uses on EVAL (offline-psi `deploy_pass` with `deploy_policy`: run 1's held-out
procedure, `eval_forward` decision context, blocks of 16, `T_warm = 0`;
`experiments/offline-psi/run.py:770-834`). This is **on-policy residency**: the head's own memory,
its own gestalts.

- **Cap on TUNE** = run 2's ceiling share (`experiments/rsr-run2/PREREG.md:109-115`; code
  `experiments/rsr-run2/run.py:479-517`, `ceiling`):
  `(hit_ψR − hit_FIFO) / (hit_FactFiller − hit_FIFO)`, all queries, pooled per document.
  - `hit_ψR`: residency of the queried assert at its query step under the head's own victim
    sequence (the deploy records' `hits`, checked by run 1's residency replay, R-C7);
  - `hit_FIFO`: model-free `headroom.simulate` with `FIFOPolicy`, M = 16;
  - `hit_FactFiller`: run 2's `factfiller_hits`, seeded `random.Random(f"ff:{seed}:{doc_id}")`;
  - CI: `R1.boot_weights(512, seed, 2000)`, paired across the three ω, FIFO and FactFiller; a
    replicate with denominator ≤ 0 is dropped and counted (B0's rule).
- **Selection, per seed:** `ω_s` = the ω with the highest cap point; ties to the smaller ω. The
  selection reads `hits` only. No accuracy statistic is computed on TUNE.
- The selection (`ω_s`, `K_conv`, φ sha256) is written to the ledger and hashed **before** the
  `deploy` stage can run (offline-psi's `deploy_guard` / C9 order, `:782-797, 972-983`).
- If every ω has a `None` cap point for a seed (pooled denominator ≤ 0), that seed is exit 3.

Every configuration fitted (3 ω × 3 seeds) is reported with its convergence label, `K_conv`, TUNE cap
and the §4.6 descriptive caps, whether or not selected.

### 4.4 Headline: deployment on EVAL (already used; run 2's rule)

- The selected φ per seed, frozen, deployed once on EVAL `[995000, 997048)`, 2048 documents per
  seed, by the same path. **Labelled "EVAL, already used"** (comparable to offline-psi and run 2).
- **Headline per seed:** `acc.all(ψR-2a) − acc.all(fifo)`, FIFO from run 2's records
  (`.worktrees/rsr-run2-run/runs/rsr-run2/heldout/seed{s}/fifo.jsonl.gz`, sha256 ledgered at first
  read), `R1.contrast` with `R1.boot_weights(2048, seed, 2000)`.
- **Label:** run 2's `seed_label`, imported: WIN if CI lower > 0 and point ≥ +δ_seed; LOSS if CI
  upper < 0 and point ≤ −δ_seed; EQUIV if `−δ_seed < lo` and `hi < δ_seed`; UNRESOLVED otherwise.

### 4.5 2a reading and the gate to 2b (PLAN §2, verbatim, then tightened)

> **Proceed to 2b iff WIN on all 3 seeds.** Otherwise stop: the result is "structurally hard", and
> the next stop is Gate A.

Tightened: if 2a exits 3 (any control red), no label is read, 2b does not run, and the PM takes 2a to
Gate A as a **control failure**, not as "structurally hard" and not as a result.

### 4.6 Secondary (descriptive, unadjusted; never gates)

1. Per configuration: convergence label, `K_conv`, `J` trajectory; TUNE cap with CI.
2. Per configuration: the **FIFO-world** cap on U-val and on U-train by offline-psi PREREG §5's
   code (`replay_rates`, `reference_rates`, B0's `boot_rule`), printed beside offline-psi's,
   diagnostic A's and the LS fit's U-val caps (SEEN; `experiments/offline-psi/RESULTS.md:14-24`,
   `experiments/offline-psi-converge/RESULTS.md:16-23`, `experiments/psi-ls-capacity/RESULTS.md:20-27`).
   U-val is labelled "spent for selection; descriptive here". Not contrasted.
3. Within-step Spearman ρ(ψ̂, G) on U-val live cells (offline-psi `rho_psi_g`).
4. On EVAL, for the selected head: gap buckets `gap_eq_1`, `gap_2_to_M`, `gap_gt_M`; hit rate and
   ceiling share; age-1 share and the newcomer label (run 2 §5 thresholds); ρ(ψ̂, age) at EVAL
   decisions; paired contrasts against offline-psi cf_all, run 2 ψC, the mean of random0..4, and
   pinned Expire-Span `L48-k1e-2`, from their saved EVAL records (sources and sha256 as offline-psi
   PREREG §6 and run 2 PREREG §4 item 4).

### 4.7 Controls (failure: exit 3; numbers written and marked NOT READ)

| id | control | expected |
|---|---|---|
| R-SEAL | §1 (a)-(f) for 2a's stages; TUNE closure at the `select` stage before any model loads | true |
| R-C1 | Inputs pinned: Δ and gestalt sha256 (§4.1); `~/rsr-substrate/2026-09-27/MANIFEST.sha256` `shasum -c` rc 0 at start and end, its own sha256 = `MANIFEST_SHA256` (`experiments/rsr-run1/run.py:201`); every run-2 / offline-psi / ES record file read hashes to its pin at first read and at the end | true |
| R-C2 | **Harness identity:** 2a's fit path with `loss="mc"` on cf_all and on cf_live reproduces offline-psi's recorded U-val L_MC at step 3000, per seed and arm, `==` on the float (diagnostic A C2's pattern, `experiments/offline-psi-converge/PREREG.md:96`; values from offline-psi's `measure-seed{s}.json` `fit.{cf_all,cf_live}.history`) | `==` |
| R-C3 | Frozen and age-free: only φ in the optimizer; no age input; substrate checkpoint sha256 equals its pin before and after; `rsr.constants.record()` never called | true |
| R-C4 | Finite: `J`, the batch loss and the φ gradient norm at every step; ψ̂ at every TUNE and EVAL decision (float32) | true |
| R-C5 | Every grid configuration is in the ledger (3 per seed) with history, `K_conv`, TUNE cap; the selection is recomputable from the ledger by §4.3's rule | true |
| R-C6 | Order: `fit` → `select` → `deploy` per seed; EVAL read once per seed, by the frozen selected φ (offline-psi `deploy_guard`; a re-run refuses if the EVAL output exists) | true |
| R-C7 | Residency: every TUNE and EVAL deploy pass's in-loop residency equals the model-free replay of its own victims (run 1 C7, `R1.residency_ok`) | 0 mismatched |
| R-C8 | Pairing: the EVAL records' `(doc, [(t, gap)])` sequence equals run 2's fifo pass for all 2048 documents (length asserted) | equal |
| R-C9 | δ_seed machine-read and equal to run 2 PREREG §2.1's table (`read_delta`) | `==` |
| G-C3 | Full suite at the exact launch head: rc 0 read directly; passed / failed / skipped / errors ledgered; any skipped test named, never counted as a pass | rc 0 |

### 4.8 Timing (placeholder)

Filled by **Amendment 2a-S**, committed alone before 2a launches: 50 fit steps per ω on seed 0,
one `J` evaluation, and one 64-document deploy block on U documents (never TUNE or EVAL), with the
projection for 9 fits to the 200,000-step worst case plus 9 TUNE deploys and 3 EVAL deploys. **A
projection above 8 h stops 2a and returns to the PM.**

## 5. Phase 2b (on-policy run 3): only if 2a is WIN on all 3 seeds

### 5.1 Harness

Run 2's harness at `728a42f` (`experiments/rsr-run2/run.py`, `experiments/rsr-run1/run.py`,
imported, never copied), with the loss switch of §0.3 and nothing else changed.

### 5.2 Arms

| arm | what | role |
|---|---|---|
| **psiR** | run 2's psiC (content-only bilinear ψ̂, `target=loo_resample`) with `loss=rank`, `rank_k=3`, `rank_omega=ω_s` | headline |
| **ageR** | run 2's age arm with the same loss and `ω_s` | descriptive baseline (CLAUDE.md: the age-only head is a separate arm); gates nothing |
| **fifo** | run 1's command, byte for byte, as in run 2 | the reader and the FIFO reference |

`ω_s` is machine-read from 2a's ledger (selection row, sha256 of the ledger pinned at 2b launch).
Overrides otherwise run 2's (`experiments/rsr-run2/PREREG.md:17`).

### 5.3 Step count and projection (PLAN: fixed here, justified by a smoke)

PLAN's premise "run 2 was still improving at 3000" is not what the record shows for run 2 (§10 item
B2); what the record does show is that run 2's optimizer needs on the order of 10⁵ steps to converge
offline on L_MC (diagnostic A). So 3000 is not reused by default. The rule:

    K_2b = the smallest multiple of 1500 that is ≥ max(3000, max_s K_conv(s, ω_s))

where `K_conv(s, ω_s)` is 2a's convergence step for the selected configuration of seed `s` (§4.2),
read from 2a's ledger. Reason: 2a fits the same head, loss, ω, optimizer, lr and batch size (16
documents × 48 steps per optimizer step) as 2b; on-policy targets are noisier, censored and
non-stationary, so 2a's `K_conv` is a **lower-bound proxy** for the steps 2b needs, and 2b never
trains fewer steps than run 2. Multiples of 1500 keep run 1's `derive_ckpt` (K/3) integral.

**Smoke S-2b (after 2a's verification, before 2b launches; POOL documents only):**
- the launch shape: psiR, ageR and fifo × seeds 0/1/2 concurrently (9 processes, 1 thread each, one
  cpu-det slot each), each seed with its `ω_s`, for **48 optimizer steps** (16 warm + 32 on-policy),
  in a scratch out-root;
- measured per RSR process: median `step_s`, `loo_s`, `eval_forward_s` and the rank-loss time over
  the on-policy steps `k ∈ [16, 48)`; the load average at start and end; the slot occupancy;
- one held-out pass of 64 already-used EVAL documents (`[995000, 995064)`) through the psiR payload,
  for the per-pass cost (never EVAL2).

**Amendment 2b-S**, committed alone before 2b launches, states `K_2b` computed by the rule above from
2a's ledger (not chosen), the smoke's numbers, and the projection
`K_2b × (max over RSR processes of median step_s) + held-out passes`.
- **Stop line: 24 h** (run 2's, `experiments/rsr-run2/PREREG.md:50,213`). If the projection exceeds
  it, 2b does **not** launch and returns to the PM with the numbers. `K_2b` is never shortened to
  fit the line, and the lr is not changed to fit it (PLAN: the only change is the loss).
- PLAN's "about 13 h or more" is an expectation, not the stop line.

### 5.4 Held-out (EVAL2's first read)

After every training process has exited with a completed footer, run 2's `measure-seed` per seed,
pointed at **EVAL2** (2048 documents per seed), at the final φ (`k = K_2b`): run 1's held-out
passes for the three arms and the references (oracle, LRU, random0..4, slots-zeroed, the A1.8 m-9
snapshots, A4.2's diagnostic step 160), plus:

- **psiR@3000** if `K_2b > 3000` (the harness also saves φ at k = 3000; training is unaffected,
  as `derive_every`'s proof, `experiments/rsr-run1/run.py:276-284`);
- **run 2's ψC** final φ (`.worktrees/rsr-run2-run/runs/rsr-run2/psiC/seed{s}/ckpt-003000.pt`,
  sha256 ledgered) through the same held-out path: descriptive;
- **2a's selected head** (T_warm = 0 deploy path): descriptive;
- **pinned Expire-Span `L48-k1e-2`** (`~/rsr-substrate/2026-10-07-es-pinned/`) through the ES
  referendum's `es_pass` (`experiments/expire-span-referendum/run.py:379-405`): secondary;
- FactFiller hits, model-free (run 2's `factfiller_hits`).

### 5.5 Headline and class (run 2's, verbatim; PLAN §2)

- **Headline per seed:** `acc.all(psiR) − acc.all(fifo)` on EVAL2, run 1's paired per-document
  bootstrap (`R1.boot_weights(2048, seed, 2000)`), δ_seed from B2's ledger.
- Labels: WIN / LOSS / EQUIV / UNRESOLVED (§4.4).
- **Class: BEATS-FIFO iff WIN on all 3 seeds; any LOSS → FAILS; anything else → MIXED, which goes to
  the owner.** Every class goes to Gate A (§7). A BEATS-FIFO is a kill-gate pass on one synthetic
  corpus at one width, not a result about natural text (D-9, `docs/RESEARCH-CONTEXT.md:533`).
- If 2b exits 3, no class is read; the PM takes it to Gate A as a control failure.

### 5.6 Secondary (descriptive, unadjusted; never gates)

1. **psiR − pinned Expire-Span**, all-query and by bucket, labelled with run 2's rule against
   δ_seed (run 2 PREREG §4 item 4's pattern). It gates nothing. **Falsifier 6 stays deferred to E1**
   (`docs/owner/rulings/R-2026-10-07-falsifier-6-deferred-to-e1.md`).
2. **psiR − ψC(run 2)** on EVAL2, paired: same harness, different loss (and step count if
   `K_2b > 3000`).
3. **psiR@3000 − ψC(run 2)** (only if `K_2b > 3000`): the matched-step, loss-only contrast.
4. **psiR − 2a's head** on EVAL2: on-policy training against the offline fit of the same loss. This
   is the **LS FIFO-world caveat's answer** (`~/Documents/RSR-2026-10-19-review/PLAN.md` §1: "it was
   fitted on FIFO-world Δ, not on-policy"): 2b fits on its own memory's Δ.
5. ageR − fifo; psiR − ageR; hit rates and ceiling share; gap buckets; LRU; oracle regret (run 1's
   contrasts, as reported in run 2).
6. Age-1 share and the newcomer label (run 2 §5 thresholds); the self-alignment probe (run 2 §5 and
   its Amendment 1 buckets); the target diagnostics (run 2 §6).
7. **Training convergence (descriptive label, never gates).** From the heartbeats, the per-step rank
   loss of psiR: the mean over the last 10% of steps minus the mean over the preceding 10%, with a
   95% CI from a block bootstrap over 16-step blocks (seeded `torch.Generator().manual_seed(seed)`,
   2000 replicates). **STILL-IMPROVING** if the CI upper bound < 0, else **PLATEAU**.

### 5.7 Controls (failure: exit 3 unless scoped)

Run 2's controls C0–C13 as registered (`experiments/rsr-run2/PREREG.md:181-193`, including run 1's
C0–C10 as amended), applied unchanged with EVAL2 as the held-out range, plus:

| id | control | expected |
|---|---|---|
| C11′ | C11 extended: both RSR arms' labels contain `target=loo_resample` **and** `loss=rank`; every beat has `mc.target == "loo_resample"`, `mc.loss == "rank"`, a finite rank loss and a finite `loo_s` | true |
| C12 | fifo is run 1's fifo (run_id, config hash, heartbeats, diagnostics, held-out records on the documents run 1 read) | identical |
| C-K | `K_2b` in `launch.json` equals §5.3's rule applied to 2a's ledger; `steps_done == K_2b` for both RSR arms on every seed | true |
| C-ω | each seed's `ω_s` equals 2a's selection row; 2a's ledger sha256 equals the one pinned in Amendment 2b-S | true |
| R-SEAL | §1 for 2b's stages; EVAL2 closure (C0 logic) at the `measure` stage before any model loads; TUNE never generated | true |
| C-pair | every EVAL2 pass's `(doc, [(t, gap)])` equals the fifo pass's, all 2048 documents | equal |
| C-res | run 1 C7 residency on every EVAL2 pass | 0 mismatched |
| C-ES (scoped to §5.6 item 1) | `shasum -c` of the ES manifest rc 0 (run 2 C14); `es_pass` on EVAL `[995000, 995032)` reproduces the referendum's saved `es.jsonl.gz` records record for record before EVAL2 is read. On failure the ES contrast is omitted, the failure printed, and the headline and class stand | true |
| G-C3 | full suite at the exact launch head, as §4.7 | rc 0 |

### 5.8 Timing

Placeholder until Amendment 2b-S (§5.3).

## 6. Exit codes, outputs, verification

- **Exit codes** (`rsr.exit_codes`), per run: **0** ran and every control passed, whatever the
  numbers are; **1** a defect after the controls passed (a NaN in a reported statistic, a raise in
  the computation); **3** did not run, or a control failed (numbers still written, marked NOT READ;
  no label is read). `$?` is read directly, never after a pipe. Exit 5 is not used.
- **Ledgers:** `runs/rsr-run3-2a/` and `runs/rsr-run3-2b/`, each with `manifest.json` (frozen before
  the run starts, git sha stamped from git, config hash, seeds, document ranges, the sealed-range
  source), `ledger.json` (`scripts/ledger.py`), `claims.json`, `measure-seed{s}.json`, raw records.
- **RESULTS** are generated by each run's `finalize` from the ledger at its default `runs/` path; no
  number is typed. The copy into `experiments/rsr-run3-ranking/` is a later, separate commit.
- **rsr-verifier** once per run, before the PM reads 2a's gate or 2b's class.

## 7. Routing, kill rule, Gate A

- 2a not WIN on all 3 seeds → phase 2 stops ("structurally hard") → Gate A.
- 2a exit 3 → phase 2 stops (control failure) → Gate A.
- 2a WIN on all 3 seeds → Amendment 2b-S → 2b (or a stop-line return to the PM) → Gate A.
- **Kill rule (PLAN §2, verbatim):** *"if 1c drops finding 1 AND phase 2 fails (2a not all-WIN, 2b not
  BEATS-FIFO, or Gate P2 stopped it), write the short negative-results paper scoped to the synthetic
  setting, and stop."*
- **Gate A is the stop after phase 2, whatever phase 2 shows.** Phase 3 is never started from here.

## 8. SEEN / UNSEEN

**SEEN (every number this file cites or relies on).** Run 1's and run 2's RESULTS and ledgers
(including δ_seed, `experiments/rsr-run2/PREREG.md:95-99`); offline-psi's RESULTS (fit and deploy);
diagnostic A's RESULTS (convergence steps and caps); the LS check's RESULTS (caps, ridge path, F8,
F10, F11); B0's and B0-CF's RESULTS (including the `ko` caps, `experiments/b0-ceilings/RESULTS.md:42`);
the Expire-Span referendum's RESULTS; run2-vs-random's RESULTS; phase-1 item 1a's RESULTS
(`experiments/post-fails-controls/RESULTS-1a.md`); the run-2 log check
(`experiments/offline-psi-converge/PREREG.md:119-123`). Whatever of phase-1 items 1b–1e has reported
by the commit of this file is SEEN; the PM lists it at commit (§11).

**UNSEEN (produced by phase 2).** Every rank-loss fit, `K_conv` and convergence label; every TUNE
document and cap (TUNE has never been generated); 2a's EVAL deployment and labels; every 2b number;
every EVAL2 document and number (EVAL2 has never been generated).

## 9. What this does NOT show

- **2a WIN** shows that the content-only class, fitted offline with a ranking loss on FIFO-world
  returns, *can* hold a ranking that beats FIFO through this reader on already-used documents. It
  does not show that RSR's online loop finds it (that is 2b), and it is not out-of-sample in
  documents (EVAL is already used; TUNE selected the configuration).
- **2a not-WIN** does not show that no member of the class ranks well: it is one loss family, one
  grid, one optimizer at run 2's lr.
- **2b BEATS-FIFO** is a kill-gate pass (D-9) on one synthetic corpus, one width, three seeds; not
  E1, not a claim about natural text, not a test of the leading-edge strategy. **2b FAILS** does not
  separate the loss from on-policy censoring (shadow off) or from the step budget.
- Neither shows that squared error was "the" binding constraint in run 2; a 2b WIN with psiR@3000
  above ψC(run 2) is consistent with that hypothesis, not proof of it.
- No scaling claim. No cognitive claim. Falsifier 6 is not ruled on. §15 and release condition 7
  untouched.

## 10. Choices this file makes that PLAN left open, and brief/PLAN errors found

**Choices (each with its reason in the section cited):** k = 3 (§2.2); weighting form and the ω grid
`{2, 8, 32}` (§2.2); no secondary loss (§2.2); no temperature (§2.2); pairs in 2a over offline-psi's
`cf_all` cells (§2.1; the drafter proposed FIFO-live pairs, and the PM decided `cf_all` at commit,
§12 item 2), and in 2b over the policy's own live memory; ties form no pair (§2.1); per-group normalisation (§2.1); fit set
U-train (§4.1); convergence rule on the training objective, window 2000, tolerance 10⁻³, cap
200,000 (§4.2); TUNE cap = on-policy deployed ceiling share, not offline-psi's FIFO-world replay
(§4.3; reason: it measures the object 2a deploys); per-seed selection (§4.3); 2b step rule and smoke
(§5.3); 2b stop line 24 h (§5.3); the age arm kept, the fifo arm kept as run 1's command (§5.2);
four descriptive EVAL2 passes (§5.4).

**Errors found:**
- **B1.** ADR-0009 is not in `docs/decisions/` at this base: it exists only on
  `docs/adr-learning-path` (`238663a`, PROPOSED, `docs/decisions/ADR-0009-rsr-learning-path.md`; L9 at
  line 374), with L9 signed in `docs/owner/rulings/R-2026-09-30-adr0009-signoff.md`. This file cites
  the ruling.
- **B2.** PLAN §2's "Run 2 was still improving at 3000 steps" conflicts with the run-2 log check:
  run 2's ψC L_MC was flat or slightly rising over its last 1000 steps while ‖W‖ kept growing
  (`experiments/offline-psi-converge/PREREG.md:119-123`). It was **offline-psi's** fit that was still
  improving at 3000 (`experiments/offline-psi-converge/PREREG.md:5,27`). The smoke requirement stands
  either way; §5.3's rule rests on diagnostic A, not on that sentence.

## 11. Drafter's expectation (before any phase-2 number; allowed to be wrong)

- 2a fit: CONVERGED well before 200,000 steps for every ω; the ranking fit's FIFO-world U-val cap
  above the LS fit's on at least two seeds.
- 2a deploy: not WIN on all 3 seeds. Seed 1 is the likeliest miss, because its kind means order
  assert below query (`experiments/b0-ceilings/RESULTS.md:57-62`; the kind-oracle's LOSS there,
  `experiments/post-fails-controls/RESULTS-1a.md:29`), so a head that learns kind but not status
  inherits that order. Modal outcome: phase 2 stops at 2a.
- If 2a is all-WIN: `K_2b` above 3000, and a projection near or over the 24 h line.

## 12. PM review and decisions at commit (2026-10-20, before any phase-2 code or data)

Read in full by the PM (Claude Opus 5.5, the session executing PLAN-2026-10-19). Before commit, the
PM edited §2.1 (2a pair set), §3 T1(d), §4.1 (cells), §4.7 R-C2 and §10 for item 2 below; nothing
else above was changed from the draft. Each decision keeps or tightens PLAN §2.

1. **Feasibility of 2b.** If Amendment 2b-S projects past the 24 h stop line, 2b does **not**
   launch, and phase 2 goes to **Gate A** with the numbers and an owner question: a longer compute
   budget, or an lr / μP ruling (correction 15). Changing the lr is not "only the loss", so the PM
   never decides it. A stop-line return is reported as "not run (budget)", never as a result.
2. **2a pairs over `cf_all`, not FIFO-live.** PLAN §2 says "slot pairs ordered by G" and does not
   say "live". With `cf_all`, 2a is offline-psi's owner-approved primary fit with **only the loss
   changed**. So 2a's EVAL contrast against offline-psi cf_all (§4.6 item 4) isolates the loss, and
   the head learns to rank slots older than M, which deployment exercises. The bottom-k membership
   `b_j` is taken over the same pair set. R-C2 checks both the cf_all and the cf_live L9 paths.
3. **The on-policy TUNE cap (§4.3) is accepted.** It measures the object 2a deploys.
4. **ageR is kept** (§5.2). It is the age-only baseline arm CLAUDE.md requires, and it keeps
   "same as run 2" literal. Its 3 processes fit within the 12 cpu-det slots.
5. **The 24 h stop line (§5.3) is accepted.** PLAN's "about 13 h or more" is an expectation.
6. **SEEN at commit:** of phase 1, only item 1a (RESULTS-1a.md, verified at `2f62a93`). Items 1b–1e
   have not reported; none of their numbers is seen by this file. They do not gate phase 2.
7. **§10 B1 and B2 are accepted as errors in PLAN/brief.** ADR-0009 lives only on
   `docs/adr-learning-path`, and L9 is signed in the cited ruling. "Run 2 still improving at 3000"
   was offline-psi's fit, not run 2's. Both go to the PLAN errata (phase 0.4).
