---
run_id: rsr-run2
item: "RSR run 2: run 1's harness with the retention target swapped to the on-policy resample-knockout LOO return (owner ruling C3, R-2026-10-07-c3-loo-target)"
question: >-
  On fresh-stream arm B's ckpt-003000 with the transformer frozen, when psi_hat is trained online
  on G^Delta_i(t) = sum_k 0.9^k Delta_i(t+k), Delta the on-policy resample-knockout LOO delta loss,
  does the trained eviction rule beat FIFO on held-out all-query answer accuracy, and does it still
  evict newcomers?
class: "GATED (Tier G). One headline per seed, labelled WIN / LOSS / EQUIV / UNRESOLVED; the class BEATS-FIFO / FAILS / MIXED routes ROADMAP step 5 / step 6 / owner. Every other number is descriptive."
not: "Not E1, not E2. No scaling claim. No cognitive claim. §15 untouched."
substrate: "fresh-stream arm B ckpt-003000.pt, seeds 0/1/2, the sha256 pins of run 1 (experiments/rsr-run1/run.py SUBSTRATE); transformer and W_sent frozen, eval mode (O5 = eval)"
seeds: "[0, 1, 2]"
arms: {psiC: "RSR, content-only bilinear psi_hat, target=loo_resample", fifo: "FIFOPolicy (no target; run 1's command, unchanged)", age: "RSR, learned age-only head, target=loo_resample"}
pool: "[993000, 993256), 256 documents (run 1 §2)"
eval: "[995000, 997048), 2048 documents (run 1 §2, §7.1)"
steps: "3000 optimizer steps per arm; T_warm = 16 steps = 1 epoch (31(b))"
overrides: "b_enabled=false, shadow_enabled=false, nu=0, beta=1, gamma=0.9, target=loo_resample (RSR arms)"
bootstrap: "run 1's: 2000 per-document resamples, paired across arms, percentile 95% CI, torch.Generator seeded 20260927 + seed (boot_weights(2048, seed, 2000))"
falsifier: "No spec §2 falsifier is decided here. Falsifier 1 (vacuity) is watched, descriptively (§6). Falsifier 6 (Expire-Span) is DEFERRED TO E1 (R-2026-10-07-falsifier-6-deferred-to-e1); run 2 reports E1's number early (§4 item 4) and does not gate on it."
---

# PREREG: RSR run 2, the LOO target

**Drafted 2026-10-07 by a model** (Claude Opus 5.5, `rsr-researcher` session, PM-dispatched).
**Not written by Brendan.** Under review policy v2 Tier G this file is committed **alone**, before
any run-2 data. Amendments are appended and dated; the text above them is never edited.

The base is `feat/rsr-run2` @ `10f29c7`: run 1's harness `run/rsr-run1` @ `ed6302a`, the target
code `e67645b` (+ placement fix `10f29c7`) and its tests `424eb48`, and `owner/2026-10-07` @ `6d3e5e3` merged (rulings C3
`63d15b6`, run 1 owner values and ADR-0010 q1/q2/q9 `8fb7030`, falsifier 6 deferred to E1
`6d3e5e3`).

## 0. Authority, and the one change

- **C3** (`docs/owner/rulings/R-2026-10-07-c3-loo-target.md`): the target is
  `G^Δ_i(t) = Σ_k γ^k Δ_i(t+k)`, γ = 0.9, `Δ` from `rsr.metrics.loo.loo_delta_loss(mode="resample")`.
  It replaces run 1's gated, share-of-M rescaled `r_i` return. **Nothing else changes.**
- **The one change on the command line:** both RSR arms add
  `--policy-override target=loo_resample` to run 1's command (`experiments/rsr-run1/run.py`
  `cli_argv`). The arm label then ends `...,shadow=off,target=loo_resample]`, and the config hash
  differs from run 1's. The fifo arm's command is run 1's, byte for byte.
- **Unchanged from run 1, by citation** (`experiments/rsr-run1/PREREG.md` and its amendments,
  which bind here wherever this file is silent): substrate and pins (§1, A1.4), pool (§2), EVAL (§2),
  vocabulary map hash (§3), seeds 0/1/2 (A1.4), arms (§5), constants (§6: d 128, batch 16, S 48,
  M 16, lr 1e-3, ν 0, β 1, b off, shadow off, γ 0.9 by override, φ weight decay 0.01, O5 eval),
  permutation-epoch sampling (A1.7), T_warm 16 (31(b)), held-out procedure (§7.1, A3.2–A3.4:
  final φ at k = 3000, `eval_forward` decision context, blocks of 16 documents, pooled
  per-target definitions), the bootstrap (front matter), launcher discipline (A1.5: clean, pinned
  head; A2.4: tmux).
- No change was forced by a measurement (§8): the projection fits inside 24 h.

## 1. The target, as built (code `e67645b`)

| | Choice | Code |
|---|---|---|
| on-policy | `Δ_i(t)` is measured over the **policy's own** pre-write memory at step t, in eval mode under `no_grad`, against the base loss of the **same eval forward** whose gestalt is `c_t` for the decision and the regression (ADR-0009 L1(a), L2). One knockout forward per occupied rank, every row at once. | `src/rsr/retention/loo_target.py` `step_delta`; `src/rsr/model/tg/policy_loop.py` `run_policy_loop(loo=...)`; `EvalForward.logits` |
| (a) donors | `pick_donor`'s rule, unchanged: same kind, **same rank** (slot index in the oldest-first prefix, which is what `P^(sent)` indexes, ADR-0006), different document, same step, never the displaced sentence's fact key. On-policy the donor pool is the other rows' **policy** memories at this step, so same rank no longer implies same age. `donor_age_gap = |age(donor) − age(own)|` is logged. Donor seed `loo_seed(seed, k) = seed·10⁷ + k`: fresh draws each optimizer step. | `loo_target.pick_donor_fast` (list version of `pick_donor`, equality-tested), `loo_seed` |
| (b) missing | No donor → NaN → imputed 0 in the return, counted by status × kind × fill. No real target (padding) → NaN → 0, counted separately. Control C9 (§7) applies E0d's C6 floor. | `RSRPolicy._loo_term`, `loo_coverage` (`src/rsr/retention/rsr.py`) |
| (c) underfull | **No rescale and no down-weighting.** §3.2.1's `share·|memory|/M` corrects a share's mechanical fill-level inflation; `Δ` is a loss difference with nothing normalised across slots. L_MC stays ADR-0009 L9's unweighted sum-then-mean. Fill fraction reported. **Flagged to the PM (§10 item 2).** | `RSRPolicy._loo_term` docstring |
| (d) censoring | Unchanged: the return is cut at eviction (`mc_targets` zeros where not live); shadow buffer off by ruling. Exposed by §6's `g_by_evict_age`. | `RSRPolicy.mc_targets`, `_target_diagnostics` |
| age arm | Regresses its age-only table on the **same** `G^Δ`. | `RSRPolicy.retention_loss` (age branch unchanged) |

Tests (`tests/test_loo_target.py`, 15): on a FIFO memory the loop's `Δ`, donors and NaNs are
**bit-equal** to `loo_delta_loss(mode="resample")`; on a non-FIFO memory every `Δ` cell equals a
hand single-knockout; no gradient reaches any transformer parameter (both RSR arms); the default
config regresses on run 1's `r_i` bit for bit. End-to-end: run 1's psiC seed-0 command for 20 steps
at `ed6302a` and at this head gives identical heartbeats, diagnostics, 10241 eviction records,
218 checkpoint tensors and run_id.

## 2. Headline and decision rule (registered verbatim; the PM's, under Brendan's delegation)

- **Headline, per seed:** `acc.all(psiC) − acc.all(fifo)` on held-out EVAL, run 1's paired
  per-document bootstrap 95% CI. `δ_seed` = the δ run 1 used per seed (see §2.1).
  - Per-seed label:
    - **WIN** if CI lower > 0 and point ≥ +δ_seed;
    - **LOSS** if CI upper < 0 and point ≤ −δ_seed;
    - **EQUIV** if the CI lies within (−δ_seed, +δ_seed);
    - **UNRESOLVED** otherwise.
  - Class:
    - **BEATS-FIFO** iff WIN on all 3 seeds → ROADMAP step 5 (E1).
    - **FAILS** iff LOSS on any seed → ROADMAP step 6 (write-up).
    - **MIXED** otherwise → owner.

The four labels are mutually exclusive as written (WIN needs point ≥ δ > hi under EQUIV; LOSS
symmetric), so no precedence is needed. "Within (−δ, +δ)" is read as `−δ < lo` and `hi < δ`.

### 2.1 δ_seed — PM decision (the brief's "run 1's δ" was an error)

**Run 1 used no δ.** Its PREREG §7: *"It makes no superiority, equivalence or non-inferiority
claim, and it uses no δ."* The only per-seed δ in run 1's lineage is **B2 phase A's**
`δ_s = 0.25 × (acc_oracle(all) − acc_FIFO(all))` on FIT_VAL `[936000, 937024)`, γ_o = 0.9 (B2 PREREG
§9.3). Run 1's PREREG cites it for its expectation (§7.6, A1.8 m-1), the newcomer bake-off used it,
and the Expire-Span referendum registered it as its δ (its PREREG A1.1). Proposed:

| seed | δ_seed | source (machine-read, never retyped by the harness) |
|---|---|---|
| 0 | 0.03802185560447721 | `.worktrees/b2-psi-probe/runs/b2-psi-probe-fit/ledger.json`, key `ckpt3000.seed0.decisions.delta`, `["0.9"]` |
| 1 | 0.03675691427360389 | same, `ckpt3000.seed1.decisions.delta["0.9"]` |
| 2 | 0.03185756551141167 | same, `ckpt3000.seed2.decisions.delta["0.9"]` |

## 3. Primary measurement

As run 1 §7.1, A3.2–A3.4: all-query answer accuracy (argmax over the vocabulary equals the answer
token, pooled per answer target) on EVAL, per arm, final φ at k = 3000, scored rule. Held-out
real-token and answer-token NLL beside it.

## 4. Secondary (descriptive, unadjusted for multiplicity)

1. **Hit rate** (B0-CF's definition: the queried assert is resident at the query step, model-free
   residency replay of the arm's own victim sequence, pooled over EVAL) per arm, and the **ceiling
   share** `(hit_ψC − hit_FIFO) / (hit_factfiller − hit_FIFO)`, point and paired CI, with
   `FactFiller` seeded `random.Random(f"ff:{seed}:{doc_id}")` as in B0-CF. Reported **beside**
   B0-CF's γ = 0.9 cap (+0.916 / +0.955 / +0.874, `run/cf-ceiling` ledger
   `B.ckpt3000.cf.g09.cap.point`). The two are not the same quantity: B0-CF is a hindsight rule
   in FIFO's world on U; this is a trained head on-policy on EVAL. Not contrasted.
2. **Gap buckets** `gap_2_to_M` and `gap_gt_M`: accuracy per arm and the psiC − fifo contrast.
3. **ψC(run 2) − ψC(run 1)**, all-query, paired on EVAL documents with the same weights. Run 1's
   per-document records are read from `.worktrees/rsr-run1-run/runs/rsr-run1/heldout/seed{s}`;
   the contrast is computed only if their document ids and per-document query counts equal run 2's
   (else omitted, and why).
4. **ψC(run 2) − pinned Expire-Span L48-k1e-2**, all-query, per seed: a **pre-registered
   SECONDARY contrast** (`R-2026-10-07-falsifier-6-deferred-to-e1`). Same paired per-document
   bootstrap as the headline (`boot_weights(2048, seed, 2000)`), and the **same WIN / LOSS /
   EQUIV / UNRESOLVED labels against δ_seed** (§2). Also by bucket `gap_2_to_M`, `gap_gt_M`.
   - **It does not gate run 2.** Run 2's class is the FIFO rule of §2 only. This is the number
     E1 will rule on, reported early. E1's own PREREG sets "beat" numerically, and it may not be
     weaker than the spec's "RSR ≥ Expire-Span". Nothing here pre-empts that.
   - **Source: the referendum's saved per-document EVAL vectors, read, not recomputed** (no ES
     training, no ES re-evaluation): `.worktrees/es-referendum-a2-run/runs/expire-span-referendum/eval/seed{s}/es.jsonl.gz`,
     config `L48-k1e-2` (`eval.json`), sha256 seed 0 `f6a3d97f25c520c84091562002e719c489d5824ba53b484f2ad892ff12d194e1`,
     seed 1 `d647767927b7c3655a8d473fdef990c534315ed461271c1a109d466dcd3f8db2`,
     seed 2 `6ded09f44abdefd90d68c9dfcd2c8d904f7f1e8dc07c6f934078b4425ab5a74a`, each equal to its `eval.json` `records_sha256` (checked
     2026-10-07). Record format is run 1's (`doc`, `answers` = `[t, gap, ok, nll]`, `hits`,
     `victims`), 2048 documents `995000…997047` in order; the `(doc, [(t, gap)])` sequence equals
     run 1's fifo pass on every seed (37853 / 37870 / 37836 answers; checked 2026-10-07).
   - These vectors were produced from the checkpoints the ruling pins
     (`~/rsr-substrate/2026-10-07-es-pinned/`, `MANIFEST.sha256`, `shasum -c` rc 0 on
     2026-10-07). The harness re-checks that manifest at evaluate time; a mismatch is exit 3 (C14).
   - Pairing controls C14/C15 (§7) are **scoped to this contrast** (PM decision, §10 item 6): if
     either fails, this contrast is not reported, the failure is printed, and the headline and class
     of §2 stand. A C15 failure that also breaks C12 is a run-level failure through C12.
5. Run 1's other contrasts (age − fifo, psiC − age, LRU, oracle regret, headroom share, R per arm,
   §7.3 in-loop Δloss) are reported as in run 1.

## 5. Newcomer diagnostics (labels, never gates)

- **Age-1 eviction share**: of the scored held-out evictions of each arm and seed, the fraction
  whose victim has age 1 (run 1 key `seed{s}.heldout.heldout_evictions.<arm>.victim_age_distribution`).
  - **NEWCOMER-RESOLVED** if ≤ 0.05 on all seeds;
  - **NEWCOMER-PERSISTS** if ≥ 0.20 on any seed;
  - else **NEWCOMER-REDUCED**.
  - The comparison is run 1's psiC value (seed 0: 24924 / 65536 = 0.380), **labelled already
    seen**. Applied to psiC; reported for age and fifo (0 by construction for fifo).
- **Self-alignment probe**, at every scored held-out eviction decision of psiC (run 2), over the
  live slots:
  1. within-step Spearman ρ between ψ̂(s_i, c_t) and cos(s_i, c_t); pooled as the **lower median**
     over decisions, with the per-document bootstrap CI (resample documents, recompute the lower
     median);
  2. split by age bucket {1, 2..4, 5..M}: within each decision, ψ̂ and cos are converted to
     within-step percentile ranks over the live slots; per bucket, the Pearson correlation of the
     two percentile ranks over the pooled (decision, slot) cells in that bucket, with the same
     bootstrap, and each bucket's mean ψ̂-percentile and mean cos-percentile beside it (§10 item 4:
     this definition is the drafter's);
  3. the share of victims whose cos(s_i, c_t) is the step's maximum.
  - Computed for run 2 **and** retroactively for run 1's psiC `ckpt-003000.pt` (all seeds),
    **labelled seen**.

## 6. Target diagnostics (training pool; logged every step by the code at `e67645b`; never gates)

From each beat's `mc` block (`target_diag`, `loo`), aggregated by sums over steps, split at T_warm
(k < 16 FIFO-collected; k ≥ 16 on-policy):

- mean `G` by age at observation (`g_by_age`);
- mean `G` by the slot's **eventual age at eviction**, or `kept` (`g_by_evict_age`) — where
  censoring's self-confirmation shows (a slot evicted young has its return cut young);
- within-step ρ(G, age) (`rho_g_age`: median, quartiles, n per step), the falsifier-1 watch.
  B0-CF's LOO age shape was negative; reported, no threshold;
- donor coverage by kind and fill (`loo.counts`), the donor-age-gap mean and zero share;
- `loo_s` (the knockouts' measured cost) beside `eval_forward_s` and `step_s`.

## 7. Controls (each failure: inconclusive, exit 3; numbers still reported)

Run 1's §8 controls C0–C10 as amended, applied unchanged, plus:

| id | control | expected |
|---|---|---|
| C2′ | pairing as run 1 C2; for k < 16, `loss_lm` bitwise equal across all three arms | equal (the knockouts touch no training forward, no RNG) |
| C9 | **LOO coverage** (E0d C6): per RSR arm and seed, over the whole run, `Σ full_covered / Σ full_cells ≥ 0.80` | measured 0.989–0.991 over 22–24 steps, every arm and seed |
| C11 | **target stamped**: both RSR arms' labels contain `target=loo_resample`; every beat has `mc.target == "loo_resample"` and a finite `loo_s` | true |
| C12 | **fifo is run 1's fifo**: run 2's fifo arm has run 1's run_id and config_hash per seed (e.g. seed 0 `fifo-d128-s48-b16-050c2dd027ecd1c5`), identical heartbeats and diagnostics apart from timing and provenance fields, and identical held-out per-document records | identical |
| C13 | Δ is finite wherever status is OK; `loss_mc` and `phi_grad_norm` finite every step (extends run 1 C8) | true |
| C14 | **ES source pinned**: `shasum -a 256 -c MANIFEST.sha256` in `~/rsr-substrate/2026-10-07-es-pinned/` rc 0; each `es.jsonl.gz` sha256 equals §4 item 4's and its `eval.json` `records_sha256`; `eval.json` `config == "L48-k1e-2"` | true |
| C15 | **FIFO vector equals the referendum's FIFO vector**: the referendum ran no FIFO pass of its own; it paired ES against run 1's `heldout/seed{s}/fifo.jsonl.gz` (sha256 seed 0 `b05387509be57d12587b4577ca49fd3443f07ebaec735db52715d2335d8caa1e`, seed 1 `819bd8af12a70c194fc17ddc58d9d3967a0e2c8d6bff4bee5128d9e578420413`, seed 2 `3a7ad2cecb55e1096bf981aa42d72dd2eac1758a6a0d75a039c3537fa60a2cb5`; `experiments/expire-span-referendum/run.py` `paired_seed`). Run 2's fifo EVAL records must equal those, record for record (doc, answers incl. ok and nll, hits, victims), and the `(doc, [(t, gap)])` sequence of run 2's psiC pass must equal ES's | equal (C12 implies the first) |

## 8. Compute (measured, then projected)

Measured at `424eb48` (the placement fix `10f29c7` changes no arithmetic), 1 thread per process, cpu-det lane via `orchestrator.slot run --slots 1`,
run 1's real config, seed 0 unless stated. See the return note for the literal numbers; the
summary:

- **One pair** (psiC + age seed 0 concurrently, 24 steps each, load average 1.72 before, 3.20 after):
  scored steps (k = 16…23, 8 each) `step_s` median 11.92 s psiC / 11.86 s age; of that `loo_s`
  10.15 / 10.13 s and `eval_forward_s` 0.79 s. Warm steps (16 each) 11.72 / 11.70 s.
- **Nine concurrent, the launch shape** (psiC, age, fifo × seeds 0/1/2, 22 steps each, load average
  2.86 at start, 7.40 at end): scored steps (k = 16…21, 6 per process) `step_s` median
  15.89–16.02 s across the six RSR processes; `loo_s` 13.58–13.72 s; `eval_forward_s` 1.06 s.
  Run 1's psiC, same config and r_i target, measured 2.23 s median over 3000 steps in its own
  launch (its heartbeat).

**Projection (an extrapolation from 6 scored steps per process):** the six RSR arm-runs and three
fifo arm-runs in parallel on 9 of the 12 cpu-det slots: 3000 × ≈ 16.0 s ≈ **13.3 h** wall per RSR
arm (all six concurrently), fifo far shorter; plus held-out passes (run 1's, unchanged in cost
except the new probes, not measured). Under the 24 h stop line, so no option is proposed.

## 9. Author's expectation (before any run-2 number; allowed to be wrong)

- psiC: the age-1 share falls well below run 1's 0.38 (NEWCOMER-REDUCED more likely than RESOLVED),
  because a just-written slot's Δ is measured directly rather than through the share-of-M target,
  but censoring still cuts the return of anything evicted young; ρ(G, age) within step negative on
  the warm steps, as B0-CF's FIFO-world shape. Headline: EQUIV or UNRESOLVED more likely than WIN;
  LOSS on any seed less likely than in run 1 (whose points were −0.17 to −0.23). Class MIXED is the
  modal guess.
- age: learns `G` decreasing with age under censoring and evicts the oldest, as in run 1.
- C12 holds; C9 ≥ 0.98 on every seed.

## 10. Decided by the PM before commit (2026-10-07, under Brendan's delegation of technical rulings)

1. **δ_seed: B2 phase A's**, as tabled in §2.1. It was re-read from the ledger by the PM on
   2026-10-07 (`rows[162/204/246]`, keys `ckpt3000.seed{0,1,2}.decisions.delta`, `["0.9"]`).
   Reason: it is the δ the Expire-Span referendum registered (its A1.1). So §2's headline, §4 item 4's
   ES contrast, and E1's later reading all use one δ. Run 1 had none to inherit.
2. **(c) No rescale of Δ.** CLAUDE.md's rule corrects a **share's** mechanical fill-level
   inflation, and `Δ` is a loss difference, not a share. Nothing here down-weights underfull steps,
   so the rule's prohibition is honoured as written. This is a reading, not a spec conflict. It is
   reported to Brendan, not added to spec-corrections.
3. **Coverage floor: whole-run, as E0d's C6** (C9), and reported by kind. No per-kind gate: a
   per-kind floor is a new threshold with no source, and the measured coverage is 0.989–0.991.
4. **Probe item 2's definition is accepted as written** (§5).
5. **The harness** (`experiments/rsr-run2/run.py`) is built against this committed file. It is
   reviewed with the code in one Tier-G review (at most 2 rounds).
6. **C14/C15 are contrast-scoped** (§4 item 4). They are secondary controls and do not void the
   headline.
7. **The suite must be green at the launch head.** The 7 `tests/test_rsr_run1_harness.py`
   failures (shared orchestrator root, also failing at `ed6302a`) are fixed at the root, by
   isolating the orchestrator root in tests, before launch. They are not waived as "known".

## 11. What this run does not show

- No scaling claim (one width, d = 128). No cognitive claim: nothing here bears on the
  leading-edge strategy or on human recall. §15 untouched.
- B0-CF's cap is a hindsight ceiling in FIFO's world. A WIN here would not show that ψ̂ reaches
  it; a ceiling share is reported beside it, not tested against it.
- Run 2 does not rule on falsifier 6: it is deferred to E1. The pinned-Expire-Span contrast (§4 item 4) is reported early and gates nothing here.
- The return is censored at the policy's own evictions (shadow off). Any target diagnostic read as
  "what Δ says about age" is read through that censoring.

## 12. Seen already (declared)

Run 1's RESULTS (headline, victim-age histograms, the age arm ≡ FIFO on held-out), B0-CF's
RESULTS, the newcomer bake-off's RESULTS, the Expire-Span referendum's RESULTS (headline and
levels) and the first record of its seed-0 `es.jsonl.gz` (format check), and the cost runs of §8 (their `loss_mc`, coverage and timing; no held-out
number of run 2 exists).

## AMENDMENT 1 (2026-10-07): a fourth age bucket in the self-alignment probe

Written by the PM session (Claude Opus 5.5) under Brendan's delegation of technical rulings, and
committed **alone**, before any run-2 training. The text above is not edited. **No run-2 data
exists at this commit.** The only run-2 numbers anyone has seen are the build's 30-step,
32-document smoke run in a scratch directory, and it is not a result.

**Found by** the harness build at `9c053f6`. §5 probe item 2's age buckets are {1, 2..4, 5..M}.
On-policy slots can be older than `M` (FIFO never keeps one, but a learned rule can), and the
buckets leave those slots out. In the smoke, 5468 of 16384 run-2 cells and 7001 of 16384 run-1
retroactive cells fell outside every bucket. So the split, as written, ignores about a third of the
cells, and it ignores exactly the old slots that a non-recency rule keeps.

**Change.** Probe item 2's buckets become **{1, 2..4, 5..M, M+1..A_max}**, with `A_max = 48`, read
from the registry as in run 1 §6. The statistic, the bootstrap and the reporting for the new bucket
are those of the other three. `cells_older_than_M` is still reported, and it now equals the new
bucket's count, which is checked. Nothing else changes. The probe remains a diagnostic and never a
gate.
