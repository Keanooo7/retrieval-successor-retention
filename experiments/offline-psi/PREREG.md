---
run_id: offline-psi
item: "Step 2 after RSR run 2's FAILS: fit the content-only psi_hat offline on uncensored FIFO-world LOO returns, then deploy it frozen on EVAL"
question: >-
  Trained offline on B0-CF's saved resample-knockout Delta (gamma 0.9 returns, never cut at
  eviction) instead of online on its own censored returns, can run 2's content-only bilinear
  psi_hat (a) reproduce the ranking whose hindsight cap was 0.87-0.96, and (b) beat FIFO when
  deployed frozen on run 2's EVAL?
class: "EXPLORATORY after FAILS (Tier D). One deployment headline per seed, labelled with run 2's WIN / LOSS / EQUIV / UNRESOLVED rule; one fit label per seed. Neither label is a spec falsifier. The pair routes an OWNER decision (§8); it routes nothing by itself."
not: "Not E1, not E2, not a re-run of run 2. Does not change run 2's class (FAILS). No scaling claim. No cognitive claim. §15 untouched."
substrate: "fresh-stream arm B ckpt-003000.pt, seeds 0/1/2, run 1's SUBSTRATE sha256 pins (experiments/rsr-run1/run.py:186-200); transformer and W_sent frozen, eval mode"
seeds: "[0, 1, 2]"
arms: {cf_all: "PRIMARY. BilinearValueHead fitted offline on G over every (t, i<t) cell", cf_live: "SECONDARY, fit diagnostics only. Same, over FIFO-resident cells (t-M <= i < t) only", fifo: "run 2's fifo records, read, not recomputed", random: "run 2's random0..4 records, read"}
status: "IN FORCE as of the commit that adds this file, ALONE, before any fitting code exists and before any fit or EVAL number. Owner go 2026-10-08 (§12)."
---

# PREREG: offline ψ̂ on uncensored LOO returns (step 2 after run 2's FAILS)

**Drafted 2026-10-08 by a model** (Claude Opus 5.5, an interactive session with Brendan).
**Not written by Brendan.** He approved it as drafted (§12). From this commit on, amendments are
appended and dated, and the text above them is never edited.

## 0. Why this run, and what it is allowed to decide

Run 2 (C3, the on-policy LOO target) is a verified FAILS. Its ψC is below FIFO on every seed
(−0.072 / −0.045 / −0.083). Step 1 (`experiments/run2-vs-random/RESULTS.md` @ `6ed6583`) found it
is also below **uniform-random eviction** on every seed: −0.047 / −0.022 / −0.038 on all
queries, every CI below 0.

The same target, scored in hindsight and in FIFO's world (B0-CF, `run/cf-ceiling` @ `14a27f6`),
reaches 0.916 / 0.955 / 0.874 of the fact/filler gain. Run 2 differs from B0-CF in two ways at
once:

1. **The return.** Run 2's return is censored: it is cut at the policy's own evictions, with
   the shadow buffer off. B0-CF's is not cut.
2. **The learner.** Run 2 has a learned head where B0-CF read G directly.

This run separates the two. It keeps the head and the target, and removes the censoring and the
on-policy loop.

**Status.** This is exploratory work after a pre-registered FAILS. **It needs Brendan's go.** It
does not re-open run 2. Whatever it shows, ROADMAP step 6 (the write-up) proceeds. Its result
feeds an owner decision on which online change, if any, is worth a run.

## 1. Data

- **Δ: B0-CF's saved arrays, read, never recomputed.**
  - Files: `.worktrees/cf-ceiling/runs/cf-ceiling/raw/B-ckpt3000-seed{s}.delta.pt`, key `Delta`.
  - sha256 per seed:
    - seed 0: `1c047a363a8928c18fd017329b8aa6171debfb1046fb0c96e258d20364c87d72`
    - seed 1: `bf2753cef826f8d3c1647213d232c65a1e2167160be1a6c262fbea132b838700`
    - seed 2: `c880a2413fa2a4840724c7c005778df8cda77932f9db0fb6081db495514ce9c2`
  - Each file holds 1088 U documents (ids `[64, 1088)` then `[4096, 4160)`, lookahead-room's
    `documents(seed)`). Each document has a `[48, 48]` step-major array, finite at `i < t` and NaN
    at `i ≥ t`.
  - Missing donors are already imputed 0 and are flagged in `missing`.
  - Cells in `evicted` hold B0-CF's re-insertion Δ: the sentence put into slot 0 of FIFO's memory.
    This is the asymmetry declared in B0-CF's PREREG §3.
  - Checked 2026-10-08 on seed 0, document 64: 0 NaN below the diagonal; 496 of 1128 cells were
    evicted cells.
- **Substrate:** run 1's SUBSTRATE pins, which equal B0-CF's `ckpt_sha256` (seed 0 `0ee3f8a6…`,
  checked 2026-10-08).
- **Gestalts.** These are not saved by B0-CF and are computed once, here.
  - One eval-mode, `no_grad` forward per U document, under **FIFO memory** (FIFO's world, as B0-CF).
  - It records each sentence's written gestalt `s_i` and each step's decision context `c_t`.
  - `c_t` is the current sentence gestalt (correction 19), from the same forward that run 1's
    `eval_forward` uses.
  - No knockouts are run.
- **The U split, fixed here.**
  - U-val: the documents at positions `p % 7 == 0` in U order, which gives 156.
  - U-train: the other 932.
  - Fitting reads U-train only. Model selection reads U-val only.
- **EVAL:** run 2's, `[995000, 997048)`, 2048 documents per seed. It is read once, by the frozen
  selected head, after selection. No EVAL number influences any choice.
- **Disjointness:** U, U-val, EVAL, run 2's pool `[993000, 993256)` and B2's FIT_VAL
  `[936000, 937024)` are pairwise disjoint by id. This is asserted (C3).

## 2. The target

`G[t][i] = Σ_{k=0}^{S−1−t} 0.9^k Δ[t+k][i]`, computed with lookahead-room's `discounted_returns`,
imported. This is B0-CF's primary target at γ = 0.9, which is C3's γ.

- It is **not cut at eviction.** That is the whole point of this run.
- It is truncated at the document's end, as in B0-CF.
- **No rescale and no down-weighting**, as in run 2 PREREG §10 item 2.

Regression cells:

- **cf_all (PRIMARY):** every `(t, i)` with `1 ≤ t < 48` and `i < t`. The deployed head scores
  slots of any age, and B0-CF's cap replay read G at every age.
- **cf_live (SECONDARY):** only `t − 16 ≤ i < t`, the cells FIFO's memory holds. This is the
  regression set closest to run 2's. It gets fit diagnostics only (§5) and no EVAL pass.

## 3. The fit

- **Head:** `rsr.retention.value_head.BilinearValueHead(d_model=128)`, with run 2's class, μP
  multipliers and init. φ is seeded per seed exactly as run 2's psiC arm seeds it, so seed `s`
  starts from run 2's initial φ.
- **Inputs:** `s_i` and `c_t` only. **No age input** (CLAUDE.md), `b` off, `ν = 0`.
- **Gradient:** only φ = {W, u} has gradient. The gestalts are precomputed tensors, so no
  gradient path to the transformer or `W_sent` exists (C4 still checks it).
- **Loss:** run 2's `L_MC` form: `Σ_{i in the cell set at t} (ψ̂(s_i, c_t) − G[t][i])²`,
  summed per (document, t) and averaged over the (document, t) pairs that hold at least one cell.
- **Optimizer:** run 2's φ optimizer as constructed in `rsr.train.loop`: AdamW, lr 1e-3, φ weight
  decay 0.01, betas and eps from `loop.py`.
- **Schedule:**
  - Batches of 16 U-train documents, sampled in permutation epochs from a private generator
    seeded `seed`.
  - Up to 3000 optimizer steps, the same count as run 2.
  - U-val `L_MC` is evaluated every 50 steps.
  - **The selected φ is the one with the lowest U-val `L_MC`**, with ties going to the earlier
    step.
  - This is fixed now. The schedule is not tuned after any fit is seen.

## 4. Headline: deployment on EVAL (run 2's rule, verbatim)

- **Procedure:** run 1's held-out procedure, unchanged (run 1 PREREG §7.1, A3.2–A3.4).
  - The selected φ is frozen, with an `eval_forward` decision context and blocks of 16 documents.
  - The answer accuracy is pooled per answer target.
  - The policy is RSR's eviction argmin over ψ̂ with `T_warm = 0`. The head is trained already,
    and no FIFO warmup is applied at evaluation.
- **Headline, per seed:** `acc.all(cf_all) − acc.all(fifo)`, with run 1's paired per-document
  bootstrap (`boot_weights(2048, seed, 2000)`). FIFO's records are run 2's, read.
- **Per-seed label (run 2 PREREG §2, verbatim):**
  - **WIN** if CI lower > 0 and point ≥ +δ_seed;
  - **LOSS** if CI upper < 0 and point ≤ −δ_seed;
  - **EQUIV** if `−δ_seed < lo` and `hi < δ_seed`;
  - **UNRESOLVED** otherwise.
- **δ_seed:** B2 phase A's, machine-read from
  `.worktrees/b2-psi-probe/runs/b2-psi-probe-fit/ledger.json` keys
  `ckpt3000.seed{s}.decisions.delta["0.9"]`. The values are 0.03802185560447721 /
  0.03675691427360389 / 0.03185756551141167, as in run 2 §2.1.
- **Deploy class:**
  - **BEATS-FIFO** iff WIN on all 3 seeds;
  - **FAILS** iff LOSS on any seed;
  - **MIXED** otherwise.

## 5. Fit reading (U-val; B0-CF's reading rule, reused)

- **The quantity:** the fitted head's cap on U-val.
  - Use model-free residency replay (`rsr.metrics.headroom.simulate`, `M = 16`).
  - Evict the live slot with the smallest ψ̂(s_i, c_t), with ties going to the oldest slot.
  - Score with FIFO-world gestalts.
  - `cap = (hit − hit_FIFO) / (hit_factfiller − hit_FIFO)` on all queries, which is B0-CF §4's
    definition. `FactFiller` is seeded `random.Random(f"ff:{seed}:{doc_id}")`.
  - Use the paired per-document bootstrap over the 156 U-val documents.
- **Fit label per seed, B0-CF PREREG §7's rule, verbatim:**
  - **FIT-POINTS** if the cap CI lower ≥ 0.80 on all 3 seeds;
  - **FIT-PARTIAL** if not POINTS, and the point ≥ 0.50 on all 3 seeds;
  - **FIT-MISSES** otherwise.
- **The reference row:** B0-CF's own G replayed on the same 156 U-val documents. This is the
  ceiling the head is fitting toward. It is computed here and reported beside the fit, never
  contrasted with it.
- **Also reported for cf_all and cf_live:** U-val `L_MC` at the selected step, and the within-step
  Spearman ρ(ψ̂, G) over FIFO-resident cells, pooled as the lower median with the per-document
  bootstrap CI (the γ sweep's `spearman_rows` / `pooled_median_ci`, imported).

## 6. Secondary (descriptive, unadjusted for multiplicity)

All of these use run 2's EVAL documents and the same bootstrap.

1. **Against other arms:**
   - cf_all − random, using step 1's definition (the mean of random0..4, run 2's records);
   - cf_all − ψC(run 2), paired;
   - cf_all − the pinned Expire-Span L48-k1e-2 (`R-2026-10-07-falsifier-6-deferred-to-e1`;
     sources and sha256 as run 2 PREREG §4 item 4).

   None of these gates anything.
2. **Gap buckets** `gap_eq_1`, `gap_2_to_M`, `gap_gt_M`: accuracy and residency for each
   contrast in item 1.
3. **Accuracy given residency** (step 1's §2b), with the same caveat: the arms keep different
   facts, so this mixes how the reader uses memory with which facts were kept.
4. **The eviction-age cost table** (step 1's §3), using step 1's `eviction_cost`, imported.
5. **Newcomer label** with run 2 PREREG §5's thresholds, applied to cf_all:
   - RESOLVED if the age-1 share is ≤ 0.05 on all seeds;
   - PERSISTS if it is ≥ 0.20 on any seed;
   - REDUCED otherwise.

   Run 2's ψC shares (0.162 / 0.151 / 0.187) are labelled already seen.
6. **ρ(ψ̂, age) at EVAL decisions:** the within-step Spearman, as for the probe in run 2 §5,
   descriptive.

## 7. What each outcome means for the censoring hypothesis

The hypothesis is the PM's from before run 2's launch, and it is not measured. It says that
cutting the return at the policy's own evictions makes newcomer eviction self-confirming.

- **Consistent with it:** cf_all's age-1 share is far below run 2's ψC, and cf_all beats ψC(run 2).
- **Against it:** cf_all also evicts newcomers at run 2's rate, even though it never trained on
  a censored return.

Neither outcome proves it. A run with the shadow buffer on would be the online test.

## 8. Readout (routes an OWNER decision, not a run)

| Fit (§5) | Deploy (§4) | Reading | Goes to Brendan with |
|---|---|---|---|
| POINTS or PARTIAL | BEATS-FIFO | The head can express a winning ranking. Online censoring, or the on-policy loop, was the bottleneck. | the choice between the shadow buffer (spec §3.4, correction 1) and a protected recency window, for one online run |
| POINTS or PARTIAL | MIXED or FAILS | The ranking fits in FIFO's world but does not survive deployment. Either its own memory moves it off-distribution, or the reader fails on resident facts (step 1 §2b). | §6 items 3–4, to tell those two apart |
| MISSES | any | The content-only bilinear head cannot express G's ranking even without censoring. The shadow buffer alone would not fix run 2. | the content+age arm or a richer head; both change the thesis |

## 9. Controls (any failure: inconclusive, exit 3; numbers still reported, marked not read)

| id | control | expected |
|---|---|---|
| C1 | Δ pinned: each `delta.pt` sha256 equals §1's; each holds 1088 ids `[48, 48]`, finite below the diagonal | true |
| C2 | Target reproduction: `discounted_returns(Δ, 0.9)` replayed by lookahead-room's `TargetPolicy` on the **full** U reproduces B0-CF's ledger `B.ckpt3000.cf.g09.cap.point` exactly (0.9161993769470408 / 0.9551669316375201 / 0.8736576121288696) | `==` on the float |
| C3 | Disjointness: U-train, U-val, EVAL, run 2's pool and B2's FIT_VAL share no document id | true |
| C4 | Frozen substrate: checkpoint sha256 equals run 1's SUBSTRATE pin before and after; only φ parameters are in the optimizer; transformer state_dict bit-equal before and after | true |
| C5 | Gestalt forward is FIFO's world: the forward's memory at every full step is `range(t − 16, t)` | true |
| C6 | Pairing with run 2: cf_all's EVAL records have the same `(doc, [(t, gap)])` sequence as run 2's fifo pass, for all 2048 documents (length asserted, not a prefix; run 2 review F5) | equal |
| C7 | Finite: `L_MC` and the φ gradient norm are finite at every step; ψ̂ is finite at every EVAL decision | true |
| C8 | δ machine-read and equal to §4's table | `==` |
| C9 | Selection used U-val only: the harness reads EVAL only through the frozen, selected φ, once per seed (a re-run refuses if the EVAL output exists) | true |

## 10. Compute: measure first, then project

- The builder measures the following before this PREREG's numbers are relied on, and records
  them in a dated amendment written before launch:
  - the gestalt forward over a 32-document U sample;
  - 50 fit steps;
  - one 64-document EVAL block.
- **Stop line:** if the projected wall time for three seeds exceeds 8 h on the cpu-det lane, the
  builder returns options to Brendan instead of launching.
- **Expectation, unmeasured:**
  - The fit is cheap, because Δ is precomputed and there are no knockouts.
  - The cost is dominated by the one EVAL pass per seed. Run 2's projection was about 57 min per
    seed, and that included many more passes than this run has.

## 11. Drafter's expectation (before any number; allowed to be wrong)

- **Fit:** FIT-PARTIAL is the most likely label. A 16.6K-parameter bilinear head fitting a
  hindsight table on 932 documents will lose part of the 0.87–0.96 cap. cf_live will fit its own
  cells better than cf_all but extrapolate worse.
- **Deploy:** MIXED or FAILS are more likely than BEATS-FIFO.
  - The age-1 share should fall well below run 2's (REDUCED or RESOLVED). If the censoring
    hypothesis holds, the 2..M loss falls with it.
  - Step 1 found that ψC's right-given-resident beyond M was 0.68–0.71, against 0.79–0.82 for
    random. If that is a reader-side effect, it will cap cf_all's gain beyond M whatever the
    ranking.

## 12. Decisions for Brendan, ruled before this commit

**Ruled 2026-10-08, interactive session.** Brendan, verbatim: *"go ahead, commit the prereg and
build step 2"*, given in reply to the draft and its six items below. Each item is read as
approved as drafted. Nothing below was changed after his reply.

1. **Go or no-go** on exploratory work after FAILS. The write-up proceeds either way.
2. **cf_all as primary** (every past sentence), rather than cf_live (FIFO-resident only).
3. **The fit labels** borrow B0-CF §7's POINTS / PARTIAL thresholds (0.80 CI lower, 0.50 point).
   They are reused, not derived for this question.
4. **The U-val split** (`p % 7 == 0`, 156 documents). This makes the cap CIs wider than B0-CF's.
5. **The 8 h stop line.**
6. **Tier D review** (at most 2 rounds, policy v2), since nothing here gates a spec falsifier.
   Tier G applies if you want the deploy label treated as gating.

## 13. What this run does not show

- **Not an online result.** A WIN would show that the head *can* hold a winning ranking. It would
  not show that RSR's training loop finds it.
- **Not a test of the thesis.** No claim about the leading-edge strategy or human recall. No
  scaling claim. §15 untouched.
- **Δ is FIFO-world Δ, and the gestalts are FIFO-world gestalts.** The fit never sees the
  memories its own evictions create. §8's middle row is exactly that gap.

## 14. Seen already (declared)

- Run 2's RESULTS and verification.
- Step 1's RESULTS (`6ed6583`).
- B0-CF's RESULTS and ledger.
- Seed 0, document 64 of B0-CF's Δ: a structure check, 0 NaN below the diagonal and 496 evicted
  cells; no statistic.
- No ψ̂ has been fitted to Δ, and no number in §4–§6 exists.

## Amendment 1 (2026-10-08): §10's compute, measured before launch

**Drafted by a model** (the builder session, Claude Opus 5.5), not by Brendan. It adds
measurements only; nothing above it changes. No EVAL document has been generated or
read. The build's tests and smokes fitted heads for 20-50 steps on at most 40 U documents
(scratch out-roots, U-val documents standing in for EVAL, labelled SMOKE, never read);
no fit at §3's schedule and no U-val cap over the 156 U-val documents exists.

- **Command:** `PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run --lane cpu-det
  --slots 1 -- .venv/bin/python experiments/offline-psi/run.py time --seed 0 --threads 1`,
  exit 0, at `9dfa85e` (clean tree), Apple M4 Max, 64 GB, macOS 26.6.2, python 3.12.13,
  torch 2.14.0, CPU, 1 thread. Load average 1.4-1.6 from another session throughout.
- **EVAL was not read.** The "64-document EVAL block" was timed on 64 U documents (ids
  96..159) through the same deploy code path (`deploy_pass`: run 1's `run_batch`, blocks
  of 16, `eval_forward` context, run 2's `ProbePolicy`), with S = 48 and the same batch.
  Its cost does not depend on which synthetic documents it reads. Nothing was written.
- **Measured (seed 0):**
  - gestalt forward, 32 U documents: 2.657 s (C5 held on the sample);
  - 50 fit steps (cf_all, batch 16, including one U-val evaluation): 0.107 s;
  - one U-val L_MC evaluation over 32 documents: 0.0021 s;
  - C2 on the full U (model-free): 6.854 s;
  - one 64-document deploy block: 5.786 s (run 1's residency check held).
- **Projection per seed:** gestalt forward 1088 documents 90 s; two fits (cf_all, cf_live)
  of 3000 steps with 61 U-val evaluations each 14 s; C2 7 s; EVAL 2048 documents 185 s;
  total 296 s. **Three seeds, sequential: 0.25 h**, before loading, the fit reading and
  scoring (each model-free or a few seconds in the smoke). Far under the 8 h stop line.
