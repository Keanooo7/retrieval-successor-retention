---
run_id: expire-span-referendum
item: "Falsifier 6's referendum on the frozen arm-B substrate: a tuned Expire-Span [P7] against RSR run 1's psiC and FIFO"
question: >-
  On frozen arm B ckpt3000, does a tuned Expire-Span (grad_path predictor) match or beat RSR's
  psiC from run 1 and FIFO on held-out answer accuracy? This is falsifier 6's referendum on this
  substrate.
class: "GATING (review Tier G): feeds an owner decision. SUPERSEDES triggers falsifier 6 for the owner."
not: "Not E1 and not the week-4 synthetic gate's full protocol. Not evidence on the cognitive claim. No scaling claim."
substrate: "fresh-stream arm B ckpt-003000.pt per seed, run 1's sha pins, weights only, transformer and W_sent frozen, eval mode"
seeds: "[0, 1, 2], run 1's"
arm: "expire_span, grad_path=predictor (q1), w and b in phi's optimizer group (q9), 9 configs = 3 alpha x 3 L (q2)"
pool: "[993000, 993256), run 1's, per-epoch permutation pass"
val: "[994000, 994512) of each seed's generator, 512 documents, selection only"
eval: "[995000, 997048), run 1's EVAL, read ONCE, for the selected config only"
steps: "3000 optimizer steps per config x seed, run 1's"
bootstrap: "run 1's: 2000 per-document resamples, torch.Generator seeded 20260927 + seed, the same weights run 1's arms use"
falsifier: "6 (spec §7): SUPERSEDES = ES - psiC lower CI > 0 on all 3 seeds"
---

# PREREG: the Expire-Span referendum on frozen arm B

**Written 2026-10-06 by a model** (Claude Opus 5.5, an implementation session dispatched by
the PM). **Not written by Brendan.** This file is committed **alone**, before the wiring and
the harness it governs, and before any referendum step. Amendments are appended, dated, and
name what prompted them; the text above them is never edited.

The base of this commit is `ed6302a` (`run/rsr-run1`, RSR run 1's launch head). "P7:L*n*" is
line *n* of `~/research-corpus/sources/memory-retention/sukhbaatar-2021-expire-span.md`
(arXiv:2105.06548v2). "Run 1 §x" is a section of `experiments/rsr-run1/PREREG.md` at `ed6302a`.

## 0. Authority, and what this run is not

- **The ruling.** Brendan's ruling on ADR-0010 q1, q2 and q9, made 2026-10-06 ≈ 00:40 PDT in the
  PM's question form and recorded verbatim in
  `~/Documents/RSR-2026-10-05-night/R-2026-10-06-expire-span-q1-q2-q9.DRAFT.md`. ⚠️ It is a
  **DRAFT for Brendan to commit** to `docs/owner/rulings/`; it is not in the repository at
  `ed6302a`. Its three answers:
  - **q1 = `predictor`.** Only the span predictor's `w, b` receive gradient; the gestalt is
    detached at the predictor's input.
  - **q2 = equal configs + steps.** 9 configs = 3 α × 3 max-span `L`. The ramp `R`, the init
    and structured dropout come from [P7]. The same optimizer steps × seeds as RSR. Selection
    on a validation split; the test split is read once; all pre-registered.
  - **q9 = φ's optimizer.** AdamW, φ's learning rate, decoupled weight decay 0.01, ε 1e-8.
- **ADR-0010 is PROPOSED.** q3–q8 and q10 are not ruled. Where the ADR says [P7] fixes them,
  the build follows [P7]. Otherwise this PREREG declares them (§4, §11), and they stay open
  for review.
- **Review.** `~/Documents/RSR-2026-10-05-night/REVIEW-POLICY.md` Tier G: one review of this
  PREREG and the code together, at most one confirming round. The full suite passes on the
  exact launch head; `rsr-verifier` runs before merge.
- **Not E1.** Not the week-4 synthetic gate's full protocol (release condition 3) — it is the
  referendum *on this substrate*: frozen arm B, run 1's pool, run 1's arms.
- **Not evidence on the cognitive claim.** One corpus, `M = 16`, `d = 128`. **No scaling claim.**
- Spec §15, the spec, `docs/spec-corrections.md`, every frozen constant and every ruling are
  untouched. No `rsr.constants.record()` call is made.

## 1. The question, and why it is the referendum

Spec §5.4 and defect D-4 make Expire-Span the referendum: soft, differentiable expiry trains
retention by the LM loss directly. If it matches or beats RSR, the reward proxy, the MC
machinery, the shadow buffer and the warmup were working around a non-differentiability that
did not need to exist. That verdict **invalidates the design, not the result** (ADR-0010
Context). Falsifier 6 (spec §7) is that outcome. B-3 (RESEARCH-CONTEXT): "an untuned
Expire-Span losing is uninformative; a tuned one winning ends the project", so the
referendum's unfairness must not run in the project's favour.

## 2. What is shared with RSR run 1 (unchanged)

Every setting below is run 1's, read at `ed6302a` from `experiments/rsr-run1/launch_run1.zsh`
(sha256 `cb0bef7f7d9efdd8e3b0a0286d385116aec19a19e8353f5e84d1b6ed4f823f0f`, **not edited**) and
from run 1's own `launch-git.json` (run_sha `ed6302a`, gamma 0.9, wd 0.01, steps 3000, o5 eval,
c1_flags `--frozen-model-mode eval`).

| item | value | source |
|---|---|---|
| substrate | arm B `ckpt-003000.pt`, seed *s* for seed *s*; sha256 `0ee3f8a6…da60` (0), `dadd1e08…a3da8` (1), `b507ebc5…63b8` (2); T0 manifest `33dd493d…565f` | run 1 §1, A1.4 |
| frozen | `--freeze-transformer --frozen-model-mode eval`: no update to the transformer or `W_sent`, dropout off | run 1 D2(b), A1.3 (C-1), A4.1 O5 |
| `srep_norm_reg_weight` | 0.0 | run 1 A3c.1 |
| pool | `[993000, 993256)`, vocabulary of `[0, 64)`, per-epoch permutation pass (C-2) | run 1 §2, A1.7 |
| seeds | 0, 1, 2; the data-order generator is the run's seed, so every step reads run 1's batches (control E4) | run 1 A1.4 |
| steps | 3000 per config × seed | run 1 R-1 |
| `d`, `M`, `S`, batch, `max_tokens`, lr | 128, 16, 48, 16, 64, `1e-3` | run 1 §6 |
| device, threads | CPU, one `cpu-det` slot = 1 thread per job | run 1 §6, A1.6 |
| EVAL | `[995000, 997048)`, 2048 documents, held-out batch 16, eval mode, `no_grad` | run 1 §7.1, A3.3 |
| bootstrap | 2000 per-document resamples, paired, percentile 95% CI, generator `20260927 + seed` — **the same weights** as run 1's arms (`boot_weights(n, seed, 2000)`) | run 1 front matter, A1.4 |

**One difference, with no numeric effect:** `--ckpt-every 3000` (run 1 uses 40, for its
step-160 DIAGNOSTIC φ reading, A4.2). Only the step-3000 checkpoint is read here. Saving a
checkpoint draws no RNG.

## 3. The arm

`--policy expire_span` (`src/rsr/baselines/expire_span.py` at `ed6302a`, blob `a3c4e62`,
merged at `898628a`):

```
e_li    = L · sigmoid(w_l · h_i + b_l)       one (w_l, b_l) per C block (six)   Eq. 3, P7:L259-265; P7:L81-82
r_li(t) = e_li − (t − written_at_i)                                             P7:L267-268
m_li(t) = clamp(1 + r_li(t)/R, 0, 1)                                            Eq. 5, P7:L284
a_i     ∝ m_li · a_i, renormalised, in layer l's cross-attention                Eq. 4, P7:L272-279
loss   += α · Σ_{l,i : 0 < m_li < 1} e_li / B, at every training step           Eq. 7, P7:L330-335; timing P7:L368-382
evict   = argmin_live max_l r_li(t), only when memory is full                   ADR-0010 q6 (port choice)
```

- **q1 (ruled): `grad_path = predictor`.** The gestalt enters the predictor detached.
- **q9 (ruled): φ's optimizer.** `w, b` go to `build_param_groups` where φ's head goes: the
  `value_head` group, lr `base_lr / m = 1e-3` (`m = 128/128`), one AdamW, betas (0.9, 0.95),
  decoupled weight decay 0.01, ε 1e-8, no clip (ADR-0009 L8). The model is frozen, so it is
  the only group.
- **The objective under freeze.** `train()` normally optimises nothing of the model under
  `--freeze-transformer`. For this arm it optimises `loss` = the LM loss (reaching `w, b`
  only through the mask) + the span loss, which is [P7] Eq. 7's `L_task + α·(span)`.
- **Training-only parts run at training steps.** The span loss and structured dropout are
  training-only in [P7] (P7:L330-335, L387-392). Run 1's arms keep the frozen model in eval
  mode during training (O5). The built loop read the *model's* mode, so under O5 it would
  have trained this arm with `α = 0` and no structured dropout. **Code change W-1** (in the
  wiring commit after this one) adds `run_policy_loop(policy_training=...)`; `train()`
  passes `True`; every evaluation caller leaves it `None` (= the model's mode, so eval
  passes get the mask and nothing else).
- **Port choices fixed as built, not re-opened here:** q5 (an expired slot keeps its rank and
  gets zero attention), q6 (max-over-layers least-remaining eviction, ties to the oldest), q7
  (the input is the stored gestalt), q8 (`w = 0`).

## 4. The grid: 9 configs = 3 α × 3 `L` (q2)

**Units.** [P7] counts `L` and `R` in tokens. Here one memory is one sentence step, so `L`
and `R` are in sentence steps, scaled against `M = 16` and `S = 48` (run 1 §6).

**`L` ∈ {16, 32, 48} = {M, 2M, S}.**
- The upper end, `S = 48`: [P7] sets the maximum span to the task's longest horizon. In the
  Corridor task "the corridor length is sampled from U(3, 200)" (P7:L1084-1085) and "we set
  the maximum span L to 200" (P7:L1087, L1113). A stream of `S` sentence steps has no horizon
  longer than `S` (the generator's longest gap is `max_gap = 40 < S`, spec §5.2).
- The lower end, `M = 16`: the smallest cap at which a span can still reach FIFO's horizon. A
  full memory holds ages 1…16; with `L < M` no span could keep a memory as long as FIFO does.
- The middle, `2M = 32`: the midpoint.

**`R = 0.16 · L` ∈ {2.56, 5.12, 7.68}: from [P7], not tuned.** [P7]'s two settings with `L`
in the hundreds give `R/L = 16/200 = 0.08` (Corridor, P7:L1113-1114) and `R/L = 16/100 = 0.16`
(Multi-Room Portal, "L = 100, R = 16 and α = 1e-6", P7:L1133). Its long-span settings give far
smaller ratios (`R = 128` at `L = 32k`, P7:L1183-1184; `R = 128` at `L = 8k`/`16k`,
P7:L1196-1197). The ratio is the dimensionless quantity, so `R` scales with `L`. **0.16 is
chosen over 0.08** because the mask has non-zero gradient only on the ramp, `0 < m < 1`
(P7:L285-287), and ages are whole sentence steps: at `L = 16`, 0.08 gives `R = 1.28`, about
one step on the ramp per memory, and 0.16 gives `R = 2.56`. Flagged for review (§11).

**α through `κ = α · L · R` ∈ {1e-4, 1e-3, 1e-2}: `α = κ / (L · R)`, to 3 significant figures.**
- [P7]'s Eq. 7 charges `α Σ_i e_i / T` (P7:L332-335): at the maximum span, a memory costs
  `α · L` against a per-token task loss. That product is dimensionless, and across [P7]'s
  settings it spans about two decades: Portal `1e-6 × 100 = 1e-4` (P7:L1133); Corridor
  `5e-6 × 200 = 1e-3` (P7:L1113-1114); PG-19 `1e-6 × 8k ≈ 8e-3` and `3e-7 × 16k ≈ 4.8e-3`
  (P7:L1196-1197); Enwik8 large `3e-7 × 32k ≈ 9.8e-3` (P7:L1183-1184). The grid's three
  values are the decade points from Portal to Enwik8.
- **The extra `/R`** (ADR-0010 q3, open). The port charges `α · e` at *every* step a memory is
  on the ramp, about `R` steps, where Eq. 7 charges it once. The official code divides by `R`
  ("each memory has R losses applied"; P7-verification row 5c, code, not the paper). Folding
  that factor into α (`α = κ / (L·R)`) makes a memory's total charge `≈ κ · e / L`, [P7]'s own
  scale, with no code change. The loop's LM term is a sum over steps of per-token means, so
  Eq. 7's `/T` is already absorbed (ADR-0010 q3, verification 5b).

### 4.1 The grid (machine-read)

`run.py grid` reads this table and the next; the launcher reads `run.py grid`. Nothing else
types these values. `run.py` checks every row against `R = ramp_fraction · L` and
`α = κ / (L · R)` to 3 significant figures, and that the rows are the full 3 × 3 cross product.

| config | L (max_span) | R (ramp) | κ | α (loss_coef) |
|---|---|---|---|---|
| L16-k1e-4 | 16 | 2.56 | 1e-4 | 2.44e-6 |
| L16-k1e-3 | 16 | 2.56 | 1e-3 | 2.44e-5 |
| L16-k1e-2 | 16 | 2.56 | 1e-2 | 2.44e-4 |
| L32-k1e-4 | 32 | 5.12 | 1e-4 | 6.10e-7 |
| L32-k1e-3 | 32 | 5.12 | 1e-3 | 6.10e-6 |
| L32-k1e-2 | 32 | 5.12 | 1e-2 | 6.10e-5 |
| L48-k1e-4 | 48 | 7.68 | 1e-4 | 2.71e-7 |
| L48-k1e-3 | 48 | 7.68 | 1e-3 | 2.71e-6 |
| L48-k1e-2 | 48 | 7.68 | 1e-2 | 2.71e-5 |

### 4.2 Fixed fields (machine-read)

| field | value |
|---|---|
| grad_path | predictor |
| structured_dropout | true |
| init_span_fraction | 0.1 |
| ramp_fraction | 0.16 |

- **`grad_path = predictor`**: q1 as ruled. `run.py` refuses any other value.
- **Structured dropout ON** in every config. [P7] proposes it "as a form of regularization …
  For each batch, we sample l∼U(0,L) and set a_ti = 0 for all t−i > l only during training"
  (P7:L387-392), and shows a model overfitting without it (App. A.2, P7:L1055-1064). The port
  draws one cutoff per sentence step from a private CPU generator seeded by the run's seed.
- **`init_span_fraction = 0.1`**: `w = 0`, `b = logit(0.1) ≈ −2.197`. [P7] says only "we
  initialize the bias term b with a negative value" (App. A.1, P7:L1013-1015) and gives no
  number. 0.1 is the official code's default (`b = −log(1/p − 1)`, `p = 0.1`;
  P7-verification row 7b), the only value the authors published. ⚠️ It **handicaps the arm
  at initialisation** (§11 item 4): the initial span is `0.1 · L`, so at step 0 a memory gets
  zero attention past age `0.1·L + R` = 4.16, 8.32 and 12.48 steps at `L` = 16, 32, 48,
  where FIFO reads all 16.
- **`ramp_fraction = 0.16`**: §4's `R / L`.

## 5. Splits

| split | ids | documents | role |
|---|---|---|---|
| POOL | `[993000, 993256)` | 256 | training (run 1's) |
| **VAL** | **`[994000, 994512)`** | **512 per seed** | **selection only (§6.3)** |
| EVAL | `[995000, 997048)` | 2048 per seed | the comparison, read once (§6.4) |

Every id is drawn from **seed s's generator** (generator seed `s · 1_000_003 + id`), as in run 1
A1.4. VAL is declared here, before any data. It was checked by this session against run 1's
`USED_RANGES` (`experiments/rsr-run1/run.py` at `ed6302a`, run 1 §2's list: arm B's training
ranges, fresh-stream, H64, S0-03 held-out, corpus-size, U, fresh-escape, B5's cap, E0d, B2's
FIT_TRAIN / FIT_VAL / EVAL, the bake-off's EVAL_NB), its POOL and its EVAL. A `git grep` of
every local branch for integer ids in `[992000, 995000)` found no claim on `[994000, 994512)`;
the hits were float digits, hashes, run 1's own pool and a test's pool offset. The script
(`val_proof.py`, scratchpad) generated VAL documents only, never EVAL's; rc 0; its output,
literally:

```
VAL (994000, 994512) overlaps used ranges: []
VAL vs POOL (993000, 993256) disjoint: True
VAL vs EVAL (995000, 997048) disjoint: True
max id < 1_000_003: True
seed 0: vocab_sha256=afbaf994e9d1f617fdc2e68579bf5f55622ee17aee056d33d8667d218d245031 VAL closure: True documents=512 first_id=994000 last_id=994511 queries=9472
seed 1: vocab_sha256=afbaf994e9d1f617fdc2e68579bf5f55622ee17aee056d33d8667d218d245031 VAL closure: True documents=512 first_id=994000 last_id=994511 queries=9413
seed 2: vocab_sha256=afbaf994e9d1f617fdc2e68579bf5f55622ee17aee056d33d8667d218d245031 VAL closure: True documents=512 first_id=994000 last_id=994511 queries=9497
```

`run.py preflight` and `run.py validate` re-assert all of it (E0) before any model loads.

## 6. Procedure

1. **Train** (`launch_es.zsh`, §9): 27 jobs = seeds {0, 1, 2} × the 9 configs, 3000 steps each,
   on run 1's pool and substrate. Nothing is selected, read or compared during training.
2. **Validate** (`run.py validate --seed s`, per seed): the controls E0–E5 (§8), then each
   config's step-3000 Expire-Span on VAL — eval mode, `no_grad`, run 1's `run_batch` in
   run 1's held-out batches of 16, the mask applied (E7). The reading is **all-query answer
   accuracy** (S0-03's definition, run 1 §7.1 item 1): `(correct, answers)` pooled over the
   seed's 512 VAL documents.
3. **Select** (`run.py select`): **one** config for all three seeds — the one with the highest
   mean over the three seeds of its per-seed VAL all-query accuracy. Computed in exact
   rational arithmetic, so a tie is an exact tie. **Tie rule: the lowest row of §4.1** (`L`
   ascending, then `κ` ascending). Every config's mean and per-seed values are recorded.
4. **Evaluate** (`run.py evaluate --seed s`): the selected config's step-3000 checkpoint at
   seed *s*, on EVAL, **once**. The harness writes an `EVAL_READ` marker before the first EVAL
   document is generated and refuses a second read. It refuses to run before `selection.json`
   exists and while any §7.3 cell reads OPEN. No other config ever reads EVAL.
5. **Finalize** (`run.py finalize`): §7.

## 7. Outcomes and classification, registered before any data

### 7.1 The comparison

Per seed *s*, on EVAL, **paired by document** with RSR run 1's measured passes:
- run 1's **psiC** and **fifo** per-document records, `heldout/seed<s>/psiC.jsonl.gz` and
  `fifo.jsonl.gz` under run 1's output (`.worktrees/rsr-run1-run/runs/rsr-run1`), read-only,
  their sha256 recorded;
- the referendum's **ES** records for the same documents in the same order (E6).

**The headline: ES − psiC, all-query answer accuracy**, point and paired per-document bootstrap
95% CI, using `boot_weights(2048, s, 2000)`: run 1's own resample weights. **Secondary**
(reported, never classified): ES − fifo on the same statistic; both contrasts per S0-03 bucket
(`gap_eq_1`, `gap_ge_2`, `gap_2_to_M`, `gap_gt_M`); real-token and answer-token NLL.
Secondary intervals are unadjusted for multiplicity. Run 1's scope sentence (A1.4) applies to
every within-seed CI.

### 7.2 Pre-stated classification (PM decision under delegation; copied verbatim)

- SUPERSEDES: ES − ψC lower CI > 0 on all 3 seeds.
- TIES: |ES − ψC| CI within ±δ, where δ is run 1's own δ per seed.
- LOSES: ES − ψC upper CI < 0 on all seeds.
- otherwise MIXED.
- **SUPERSEDES triggers falsifier 6 for the owner.**

**Reading notes, flagged for review (§11) — not amendments to the PM's text.** The harness
needs a definite reading, and these are the ones `run.py classify` implements:
- The classes are tested **in the order SUPERSEDES, LOSES, TIES, MIXED; the first match wins.**
  With δ > 0, SUPERSEDES and TIES can both hold (a CI inside `(0, δ]`); SUPERSEDES first
  resolves that toward the outcome that ends the project, which is B-3's direction.
- **TIES** is read as: at every seed, the whole ES − ψC CI lies in `[−δ_s, δ_s]`.
- The CI is the §7.1 headline's percentile 95% CI.

### 7.3 δ per seed (machine-read; PM-OPEN)

🔴 **"Run 1's own δ" does not exist.** Run 1's PREREG registers none: §7 says every contrast is
read only as "CI above 0", "CI below 0" or "CI includes 0" and "it uses no δ" (run 1
PREREG, line 304), and A1.4 says "There is no CI across seeds, no test, and no pass or fail".
The only `DELTA` in run 1's harness is `DELTA_BLOCK = 500`, a block width in steps (§7.3 there),
not a tolerance. **This PREREG does not invent δ.** Until an amendment, committed alone and
before any EVAL read, fills the table below, TIES cannot be assigned, and
`run.py evaluate` refuses to read EVAL.

| seed | δ |
|---|---|
| 0 | OPEN |
| 1 | OPEN |
| 2 | OPEN |

## 8. Controls (any failure: the seed is inconclusive, exit 3; numbers kept, marked not read)

| control | what | where |
|---|---|---|
| E0 | VAL disjoint from POOL, EVAL and run 1's ranges; ids `< 1_000_003`; closure; map hash `afbaf994…5031` | `check_val` |
| E1 | the launcher's rc is 0 and `steps_done = 3000` | `check_training` |
| E2 | the run's frozen config is its §4.1 row: the overrides, `grad_path = predictor`, the substrate sha pin, frozen, eval mode, `srep_norm_reg_weight = 0.0`, the pool, the seed, 3000 steps | `check_training` |
| E3 | the model in the step-3000 checkpoint is bit-equal to the substrate's (nothing upstream trained) | run 1's `c3_frozen` |
| E4 | every step read the same pool ids as run 1's **fifo** arm at that seed (same data order) | run 1's `step_pool_ids` |
| E5 | every logged `loss` and `loss_lm` is finite | heartbeat |
| E6 | ES, psiC and fifo EVAL records are EVAL's documents in the same order | `paired_seed` |
| E7 | the Expire-Span mask was read at every sentence step of every held-out batch (run 1's `Recorder` does not forward it; `ESRecorder` does) | `es_pass` |
| E8 | in-loop residency equals the model-free replay of the victims (run 1's C7) | `es_pass` |

## 9. Commands

```zsh
# preflight (E0; no model loads; EVAL never generated)
.venv/bin/python experiments/expire-span-referendum/run.py preflight
# training: inside tmux, from a dedicated run worktree at the reviewed head, after the PM's go
/bin/zsh experiments/expire-span-referendum/launch_es.zsh
# after training
for s in 0 1 2; do .venv/bin/python experiments/expire-span-referendum/run.py validate --seed $s; done
.venv/bin/python experiments/expire-span-referendum/run.py select
# only after an amendment fills §7.3, and after run 1's measure-seed has written its passes
for s in 0 1 2; do .venv/bin/python experiments/expire-span-referendum/run.py evaluate --seed $s; done
.venv/bin/python experiments/expire-span-referendum/run.py finalize
```

`launch_es.zsh` (in the wiring commit): refuses an unclean head; reads the 9 configs from
`run.py grid`; pins the three substrate sha256s exactly as run 1's launcher does; runs each job
through `orchestrator.slot run --lane cpu-det --slots 1 --pinned-sha <head>`; records one rc
per config × seed. `DRY_RUN=1` runs every job at `--iters 0`.

## 10. Cost: measured, then projected

**Measured by this session**, 2026-10-06 ≈ 00:28 PDT, on the uncommitted wiring that the next
commit lands (no EVAL or VAL document touched):

- **Training step.** One config (`L32-k1e-3`), seed 0, the full run command (§2 settings,
  substrate pin, pool) at `--iters 20`, `OMP_NUM_THREADS=1` and `torch.set_num_threads(1)`,
  rc 0, wall 63 s. `step_s` from the heartbeat: median **2.599 s**, mean 2.644 s (first step
  3.028 s). Load average ≈ 10 throughout: run 1's nine jobs and another researcher's run
  shared the CPU. For scale, run 1's own recent median `step_s` at the same time: psiC 2.28 s,
  age 2.26 s, fifo 1.12 s.
- **Held-out batch.** The probe's step-20 checkpoint on 16-document batches of **POOL**
  documents (training data): 1.284, 1.175, 1.224 s per batch; FIFO 1.091 s.

**Projected (an extrapolation, labelled so):**
- Training: 3000 × 2.6 s ≈ 7,800 s ≈ 2.17 h per job; **27 jobs ≈ 210,600 s ≈ 58.5 CPU-h.**
- VAL: 27 passes × 32 batches × 1.2 s ≈ 1,040 s. EVAL: 3 × 128 × 1.2 s ≈ 460 s.
- **Wall:** the `cpu-det` lane has 12 slots (`ops/lanes.json`). With all 12 free, 27 jobs run
  in 3 waves ≈ 3 × 2.17 h ≈ **6.5 h**, plus ≈ 0.4 h of measurement, if `step_s` holds at 12
  concurrent jobs (not measured). While run 1 holds 9 slots, only 3 are free.
- 🔴 **This exceeds the brief's ≈ 10 h in CPU-hours (≈ 59 CPU-h); wall time is ≈ 7 h with the
  whole lane.** The launch waits for the PM's decision on cost.
- Disk: ≈ 9.9 MB per step-3000 checkpoint (the probe's), ≈ 0.3 GB for 27.

## 11. Open, flagged for the Tier G review

1. **δ (§7.3) is undefined** — the PM's classification refers to a δ run 1 does not have. Needs
   the PM (or Brendan) before EVAL. The harness refuses EVAL until it is filled.
2. **The α axis is `κ = α L R`**, folding q3's repeated-charge factor into α (§4). Open under
   q3; the alternative (the port divides by `R`) is a code change.
3. **`R / L = 0.16`** (Portal) rather than 0.08 (Corridor) (§4).
4. **The init (0.1) handicaps the arm at step 0** (§4.2). [P7] requires only `b < 0`, and the
   code enforces `init_span_fraction < 0.5`, so no init in range starts the arm at FIFO's
   16-step horizon at `L = 16`. B-3 says the unfairness must not favour the project; this one
   does at initialisation. 3000 steps of LM gradient on the ramp is what [P7] relies on to
   lengthen spans ("the model quickly learns to expire", P7:L1013-1016).
5. **One config across seeds** (§6.3), not one per seed.
6. **The classification's precedence and TIES reading** (§7.2 notes).
7. **The tuning budget is not symmetric with psiC's.** Run 1's psiC was not tuned on any
   validation split (`γ = 0.9` is the owner's O1, `ν = 0`, `β = 1` its PREREG's). Expire-Span
   gets 9 configs and a VAL selection, which is q2 as ruled.
8. **`through_gestalt` is not tested here** (q1 = predictor). The wiring commit adds the
   bit-equality test the ruling's PM note asks for; its result is reported there and in the
   build report, and does not change this run.

## 12. What this author has seen before committing (declared)

- No VAL or EVAL accuracy, loss or record of any arm. No run-1 held-out record (none exists
  yet: run 1 is training).
- The probe's training-pool numbers at step 19: `loss_lm` 1.4738, `loss` 1.4743 (printed while
  checking that the span loss is charged), and its step times.
- Run 1's in-progress `launch-git.json`, its heartbeat step times (above), and the tail of
  psiC seed 0's heartbeat (one step's §7.1 diagnostics, including a partial ρ of −0.415).
- The bit-equality measurement on a tiny CPU model (d = 32, 8 documents, 5 steps): `predictor`
  and `through_gestalt` gave **different** `w, b` (max |Δw| 8.75e-05, max |Δb| 3.10e-06).

## AMENDMENT 1 (2026-10-06): δ per seed

**PM decision under delegation**, prompted by the Tier-G review of `087a486`, findings B-1 and
B-2 (`docs/reviews/feat-expire-span-arm/087a486cdff8.md` at `59b57cf`, branch
`review/expire-span-arm-087a486`). Committed alone, before any VAL or EVAL read. §7.3 stays as
written (OPEN); this amendment supplies its values. What is measured is unchanged; how TIES is
classified becomes defined.

- **δ_s is B2's pre-registered δ** (`experiments/b2-psi-probe/PREREG.md` §9.3, "Reference and
  minimum effect (on `FIT_VAL`, never on `EVAL`)", as amended by A1.11; on `night/2026-09-30` at
  `94a7ceb`): `δ_s = 0.25 × (acc_oracle(all) − acc_FIFO(all))`, γ_o = 0.9, all-query answer
  accuracy read through the model, on B2's FIT_VAL `[936000, 937024)`, frozen arm B
  `ckpt-003000.pt` seed s with the same sha256 pins as §2, M = 16, S = 48.
- **The values are copied from B2 phase A, not recomputed** on this run's VAL, on run 1's EVAL,
  or on any referendum or run-1 data. Read by this session from
  `runs/b2-psi-probe-fit/ledger.json` on `night/2026-09-30` (blob `510afff`, written by commit
  `280a2ad` "run(b2-psi-probe-fit): phase A record…", ledger status `ok`), rows
  `seed0.decisions`, `seed1.decisions`, `seed2.decisions`, field `value.delta["0.9"]`:
  `0.03802185560447721`, `0.03675691427360389`, `0.03185756551141167`. They equal
  `experiments/b2-psi-probe/RESULTS-phaseA.md` "ref and δ (§9.3, A1.11), FIT_VAL, ckpt3000":
  0.038022 / 0.036757 / 0.031858. **The registered values are those six-decimal figures**
  (the table below), the PM's set.
- δ_s is applied per seed to §7.1's headline: the ES − ψC all-query contrast on run 1's EVAL
  `[995000, 997048)` and its paired percentile 95% CI `[lo_s, hi_s]`.
- **TIES** holds iff `−δ_s ≤ lo_s` and `hi_s ≤ δ_s` at every seed s ∈ {0, 1, 2}. The
  precedence is unchanged: SUPERSEDES, LOSES, TIES, MIXED; first match wins.
- δ is **not** recomputed on EVAL, because B2 §9.3 says "never on `EVAL`".

### A1.1 δ per seed (machine-read)

| seed | δ |
|---|---|
| 0 | 0.038022 |
| 1 | 0.036757 |
| 2 | 0.031858 |

## AMENDMENT 2 (2026-10-06): a NaN in Expire-Span's backward pass; a full re-run

Written by a model (Claude Opus 5.5): an implementation session drafted it, and the PM (session
a3d27993) set the re-run scope in A2.4 under delegation. It is appended below Amendment 1 and
committed **alone**, before any re-run step it governs. The text above it is not edited.

**Prompted by** control E5 (§8) failing at seed 1: `run.py validate --seed 1` exited 3 ("every
logged loss and loss_lm is finite"). By §8, seed 1 is **inconclusive**. `run.py select` refused
(rc 3), and **EVAL has not been read**.

**What this amendment changes.** Only the code that produced the seed-1 numbers, and which jobs
run again. *What* is measured and *how* it is classified (§6, §7, A1) do not change. It is an
amendment because a reported number (seed 1's VAL accuracy, and through it §6.3's selection) will
come from a different code path than the one the launch head ran.

### A2.1 The defect

- **Where:** `src/rsr/model/tg/model.py:286` at `c59d1af`, in `_reweight_attention`
  (Expire-Span's Eq. 4 reweighting, P7:L272-279):
  `return (a / total.clamp(min=torch.finfo(torch.float32).tiny)).to(att.dtype)`.
- **When it triggers:** a cross-attention row in which *every* weight is exactly 0. In this run
  that happens when structured dropout's cutoff `l ~ U(0, L)` falls below 1. Every memory then has
  age ≥ 1 and is dropped (P7:L387-392). This hits about 1/L of training sentence steps (seed 0,
  `L = 16`: 601 of the first 9,400 sentence steps with memory).
- **Mechanism:** the forward is correct: `a = 0`, so the output is 0. But the backward of the
  division is `grad / tiny = grad · 2**126`. That overflows float32 to `inf` once the upstream
  gradient on the attention probability exceeds about 3.97 in magnitude. `MulBackward0` of
  `a = att · weight` then computes `inf · 0 = NaN`, which reaches the span predictor's `w, b`
  through the mask. One Adam step makes `w, b` NaN, and every later forward is NaN.
- **Measured, seed 1, `L16-k1e-4`:**
  - A deterministic CPU replay of the launcher's exact command at `c59d1af` (orchestrator
    `cpu-det`, 1 slot) matches the original heartbeat **bit for bit**, NaNs included, through
    step 131. Fields compared: `loss`, `loss_lm`, `es_grad_norm`, `es_weight_norm`, `es_bias`.
  - Gradient hooks put the first non-finite gradient at **train step 130, sentence step 19**. At
    that step the dropout cutoff was 0.7067, every weight in the row was 0, and 2 entries of
    `∂/∂a` were `+inf`. The largest finite entry was `3.38e38`, which corresponds to an upstream
    `|grad| ≈ 3.97`.
  - `torch.autograd.set_detect_anomaly(True)` independently names `MulBackward0` at
    `a = att * weight` as the first op to return NaN.
  - `es_grad_norm` is NaN at step 130 while `loss` is finite; `loss` is NaN from step 131. This is
    the same signature the PM measured in 5 of the 9 seed-1 configs.
- **Why seed 1 and not seeds 0 and 2:** the empty-row path runs on every seed about once per L
  sentence steps. It overflows only when an empty row coincides with a large upstream attention
  gradient. Seed 1's data order and dropout stream, which every config at a seed shares, reached
  that coincidence near steps 122–136. Seeds 0 and 2 passed E5 in all 9 configs, so their
  empty-row gradients never overflowed.

### A2.2 The fix

- **Fix commit:** `35eb103` on `fix/expire-span-nan-backward` (parent `c59d1af`). One commit:
  `src/rsr/model/tg/model.py` and `tests/test_expire_span.py`.
- **The change:** `torch.where(total > 0, a / total.clamp(min=tiny), 0)`. An empty row's output
  stays the constant 0, and its gradient is now 0 instead of `grad · 2**126`. A row with
  `total > 0` is unchanged, forward and backward.
- **Tests (red without the fix):**
  - `test_an_all_zero_weight_row_has_a_finite_zero_gradient` uses an empty row with upstream
    gradient ×10. It **fails at `c59d1af`** (`assert torch.isfinite(att.grad).all()`) and
    passes at `35eb103`.
  - `test_a_row_with_any_weight_is_bit_identical_to_the_plain_quotient` checks forward and
    backward against `a / sum(a)` with `torch.equal`.
- **Suite at `35eb103`:**
  - Targeted files (`test_reduction`, `test_fidelity`, `test_expire_span`,
    `test_expire_span_arm`, `test_expire_span_referendum`): `passed=143 failed=0 skipped=0
    errors=0`, rc 0.
  - Full suite, serial, `RSR_ORCH_ROOT` unset: `passed=2154 failed=7 skipped=0 errors=0`. The 7
    failures are the known environmental failures in `test_rsr_run1_harness.py`.

### A2.3 Evidence that clean seeds are unchanged (bit-identity)

Each run uses the launcher's exact command, CPU, orchestrator `cpu-det` 1 slot, 200 steps, run
before the fix (`c59d1af`) and after it (`35eb103`). Exact equality was checked on `loss`,
`loss_lm`, `es_grad_norm`, `es_weight_norm` and `es_bias` at every step 0–199:

| job | pre vs post | original referendum heartbeat vs post |
|---|---|---|
| seed 0, `L16-k1e-4` | **BIT-IDENTICAL** (200 beats) | **BIT-IDENTICAL** (200 beats) |
| seed 2, `L32-k1e-3` | **BIT-IDENTICAL** (200 beats) | **BIT-IDENTICAL** (200 beats) |

The changed path was exercised in these 200 steps: 601 (seed 0) and 298 (seed 2) sentence steps
had every memory dropped. The old finite `grad · 2**126` on those rows was always multiplied by
the dropout mask's 0 before reaching `w, b`. That is why removing it changes no bit.

**Scope of the evidence:** the measurement covers 200 of 3,000 steps. Extending it to step 3000
is an argument, not a measurement. The fix changes `w, b` only when an empty row's gradient was
non-finite, which E5 rules out on seeds 0 and 2. The one exception would be a memory whose pre-clamp
mask is exactly `0.0` while it is live and undropped in an otherwise empty row. If the PM wants it
measured, the verifier can replay one clean job to step 3000 at `35eb103` and compare the step-3000
checkpoint's `w, b` with the original's.

Seed 1, `L16-k1e-4`, at `35eb103`, 200 steps:
- bit-identical to the original through step 129;
- step 130 is `es_grad_norm = 5.38` (was NaN);
- no non-finite `loss`, `loss_lm`, `es_grad_norm` or `es_bias` at any step 0–199.

### A2.4 Re-run scope (PM decision under delegation)

- **All 27 jobs re-run** at a run head that contains `35eb103`. Everything else is unchanged:
  §2, §4.1, §4.2, §6.1, 3000 steps, and run 1's sha pins.
- **Where:** a fresh dedicated run worktree, so the output root `runs/expire-span-referendum/` in
  that worktree starts empty. The original outputs, in `.worktrees/es-referendum-run`, are
  untouched (A2.5).
- **Job ids** carry an attempt tag, `es-ref-a2-<cfg>-s<seed>`. This is the only launcher change.
  The orchestrator refuses a reused job id, and the original job records are kept, not deleted.
- **Why all 27 and not seed 1 only:**
  - A2.3 measured bit-identity on clean seeds over 200 of 3000 steps, and extended it to 3000 by
    argument. A full re-run turns that argument into a measurement.
  - A seed-1-only re-run would need harness changes to read two output roots.
  - Cost: about 6.5 h of otherwise idle CPU.
- **Control A2-C (DIAGNOSTIC, reported, never gating):** for the 18 seed-0 and seed-2 jobs,
  compare the new heartbeats against the original ones at every one of steps 0–2999, on the A2.3
  fields, and compare the step-3000 checkpoints' `w, b`.
  - Expected: bit-identical.
  - Any difference is reported in RESULTS, with the first differing step. It does not change which
    outputs are read: §6 reads **only** the re-run's outputs, for all three seeds.

### A2.5 The original outputs are kept

- All original outputs of the 27-job launch at `c59d1af` are **kept, not deleted**. Seed 1 is
  marked inconclusive per §8. Seeds 0 and 2 are superseded by the re-run, and serve only as
  control A2-C's reference. This covers checkpoints, heartbeats, logs, rc files, and whatever
  `validate --seed 1` wrote.
- **None of their VAL numbers are read**, and they never enter §6.3's selection or §7.

### A2.6 EVAL is still unread

- **EVAL `[995000, 997048)` has not been read** by any config, and no `EVAL_READ` marker exists.
  The §6.4 single read stands for the selected config only, after the re-run, validation and
  selection.
- This session read no VAL accuracy and no EVAL data. It read only the training heartbeats of the
  jobs named above, on the fields listed in A2.3.

### A2.7 Unchanged

**No selection rule, grid or threshold changes:**
- §4.1's 9 configs and §4.2's fixed fields;
- §6.3's selection (highest three-seed mean VAL all-query accuracy, exact rationals, lowest-row
  tie rule);
- the controls E0–E8 (§8);
- the §7.2 classification and its precedence;
- the δ table (A1.1).
