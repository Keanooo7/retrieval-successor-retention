# B2 ψ̂ probe: phase A results (capture, fits, FIT_VAL decisions)

**Written 2026-09-29 by a model** (Claude Opus 5.5, an `rsr-researcher` session, item B2-A-WRAP).
**Not written by Brendan.** The fit was **not re-run** for this document. Every number below is
read from `runs/b2-psi-probe-fit/ledger.json` (the key is given), or is arithmetic on ledger values,
and is labelled as arithmetic where it is. Checkable claims are in `runs/b2-psi-probe-fit/claims.json`.

**Phase A is FIT_VAL only.** No EVAL document was run. Nothing here is a B2 outcome: the ledger
verdict is `inconclusive` ("phase A only"). The FIT_VAL contrasts below are the inputs §9.3 and §9.4
use (ref, δ, σ). They are not the §9.5 outcome, which is read on EVAL.

## Provenance

| | |
|---|---|
| Command (parent, as recorded in the ledger) | `.venv/bin/python experiments/b2-psi-probe/run.py fit` (`--parallel` default 6, capped to `FIT_MAX_PARALLEL = 3`) |
| Threads | `RSR_B2_THREADS = 4` per child (ledger `ckpt{c}.seed{s}.threads` = 4 on all six). The env assignment itself is not in the ledger; the recorded `torch.get_num_threads()` is. |
| Children | `run.py fit-child --checkpoint {2500,3000} --seed {0,1,2} --source .worktrees/fresh-stream/runs/fresh-stream --out runs/b2-psi-probe-fit/phaseA`, all six exit 0 |
| Git SHA | `a94b228c633eeb1e66914acacd8c4717b033dc8d` (committed 2026-09-28T01:33:43Z, 36 s before the run started) |
| `dirty` | **`true`**. See "Provenance note" below. |
| Hardware | Mac Studio, Apple M4 Max, 64 GB; device `cpu`; Python 3.12.13; macOS 26.6.2 arm64 |
| Config hash | `f239d981a036c0f5b0c2ac8b4e103a7219c00fb9c93ba859f60e8a590eb45126` (= sha256 of `manifest.json`) |
| Seeds actually run | 0, 1, 2; checkpoints 2500 and 3000 (arm B of fresh-stream; ckpt3000 is the headline, §2) |
| Wall time | started 2026-09-28T01:34:19Z, finished 06:47:53Z: **5 h 13 m 34 s** in two waves of three children (ckpt2500 first, then ckpt3000) |
| T0 | `T0.start` and `T0.end` both `ok`, 435 files, 0 bad. Re-checked 2026-09-29 from the main checkout root: `shasum -a 256 -c ~/rsr-substrate/2026-09-27/MANIFEST.sha256` → 435 `OK`, 0 other lines, **rc 0** |
| Logs | `runs/b2-psi-probe-fit/logs/*.log` are all **0 bytes**, and so is the parent's fit log in the session scratchpad. The ledger is the only record of the run. |

**Provenance note (`dirty = true`).** `git_provenance()` runs `git status --porcelain`, which counts
untracked files. At the start of the run the tree held one untracked file outside the source roots,
`.orchestrator/outbox/b2-psi-probe-prereg.md` (created 2026-09-27 13:23 local, five hours before the
run). `Ledger.write()` refuses to write if anything under `src/`, `scripts/` or `experiments/` is
uncommitted, and it wrote, so those roots were clean at the end of the run. `git diff --stat
a94b228` in the worktree is empty. The flag is therefore explained by an untracked non-source file,
not by uncommitted code. A change made and reverted during the run could not be detected by either
check.

**phaseA payloads** (gitignored `.pt`, not committed; sha256 for reference):

| file | sha256 |
|---|---|
| ckpt2500-seed0.pt | `3f56c36fa188f53471ac30897de2dbb6b57aac53a9c359ea8f0b58b22579022c` |
| ckpt2500-seed1.pt | `9b93f40b6b5ba21354a245f8d8bd14578963ee5169c0eb69aa694e2250e48f77` |
| ckpt2500-seed2.pt | `9549b460efa27012d6dda1bcb87040d88395594b944290f4c93289a991c2404b` |
| ckpt3000-seed0.pt | `0227d80a2f77d52ff84b4311250636ce55e1509ed5a89fb833b3f406336a4846` |
| ckpt3000-seed1.pt | `cdb2c3640a584dcac3dbb22a20025df21f0aefa0f238cce1d83d219e4c98c0db` |
| ckpt3000-seed2.pt | `5ed0142cb0601046e790c470607f65273f802f9edf0899944f0867ac76861cfe` |

## 🔴 A reporting defect in the logged R² (found while writing this)

> **Superseded 2026-09-29 by [Erratum P0.2](#erratum-p02-2026-09-29-the-logged-r²-fixed-and-corrected-without-a-refit)** at the end of this file: the formula is fixed in `run.py`, the stored values are corrected in sidecars, and the correction is checked against the split code to 5.6e-16. The text below is kept as written.

**Every pooled `val_r2_raw` in the ledger, and `age_decodability.{U,C}.r2`, is wrong by a factor of
`n_val`.** In `fit_heads` (`run.py`), `sse_raw` is a **sum** over rows, and the R² is computed as
`1.0 - float(x) * n_val / sst`. The correct form is `1 - x / sst`. The logged values are therefore
around −10⁵ to −10⁶ (for example `ckpt3000.seed0.age_decodability.U.r2` = −201898.7).

- **Correction (arithmetic on ledger values, exact):** `R²_true = 1 − (1 − R²_logged) / n_val`, where
  `n_val` is the row's `n_val_rows`.
- **Independent check.** The F11 split R² (`r2_split_by_index`) is computed by different code with
  the correct formula. The corrected pooled R² of the age predictor falls between its two split
  values, or next to them, in every case (for example ckpt3000 seed 0 U: corrected pooled 0.8252,
  split 0.8269 / 0.6585).
- **What it does not touch.** λ selection (MSE, not R²), ref, δ, σ and N_E do not read R². The
  **residuals, λs, accuracies and N_E below are unaffected.**
- **What it does touch: E0h.** `e0h_compute` treats a seed as informative only if the U@0.0 head's
  `val_r2_raw > 0`. The stored values are −351034.9 / −432716.5 / −418435.6 (ckpt3000), so **every seed
  would be UNINFORMATIVE and E0h would classify INTERMEDIATE whatever its within-step R².** The
  corrected values are 0.6599 / 0.5808 / 0.5946, which are > 0. This must be fixed before `run.py
  e0h` is run. It does not need a refit: the correction is closed-form from the stored numbers.
- The test `test_age_decodability_is_raw_mse_lambda_and_r2_on_fit_val` asserts only `r2 <= 1.0`, so it
  passes on the wrong value.

Every R² below is given as **corrected (logged)** or as the split value, which is correct as logged.

## TBD-3: N_E (§9.4, A1.12)

Ledger `N_E.power`: **`N_E = 1024`, `underpowered = False`, `raw_max = 166`.**

- **The binding contrast is the floor, not a contrast.** The largest required size over the six
  gating contrasts and three seeds is **166**, from `psiU_minus_refU.gap_2_to_M@seed0`. That is below
  `N_E_MIN = 1024`, so §9.4 item 4's clip to `[1024, 40000]` sets `N_E = 1024`.
- Per contrast and seed (ckpt3000, γ = 0.9, `N_V = 1024`):

| contrast | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| ψ̂U − ref_U | 114 | 90 | 130 |
| ψ̂C − ref_C | 34 | 23 | 74 |
| ψ̂U − random | 87 | 68 | 97 |
| ψ̂C − random | 37 | 42 | 59 |
| ψ̂U − ref_U, gap_2_to_M | **166** | 140 | 156 |
| ψ̂C − ref_C, gap_2_to_M | 36 | 27 | 58 |

- Check by hand for the binding cell: σ = 0.0038942 (`ckpt3000.seed0.decisions.sigma`), δ = 0.0380219,
  `⌈1024 · (1.96 · 0.0038942 / 0.0190109)²⌉ = ⌈165.0⌉ = 166`.
- TBD-2 put Tier 1 at 1.870 s per (document, seed), so `N_E = 1024` projects to about 0.53 h for Tier 1.
  That is a projection from TBD-2, not a measurement.
- `run.py`'s own `N_E` constant is still `None`, so `run.py eval` still exits 3 until it is set.

## Ridge fits: λ and residuals

All heads: 11 of 11 grid points eligible (`n_eligible = 11`, residual ≤ 1e-6 at every λ). The worst
selected-λ relative residual over every head, checkpoint and seed is **7.4e-13** (`ckpt2500.seed1.head.C+@0.9`),
far under A1.6's 1e-8. Row counts: `n_train_rows` 297,792 (U) / 166,848 (C), N_F = 264; `n_val_rows`
1,032,192 (U heads, full-memory rows) / 524,288 (C heads).

Selected λ (grid-edge flag `*` means λ sits at an end of the 1e-6 … 1e4 grid), ckpt3000, seed 0 / 1 / 2:

| head | λ | FIT_VAL demeaned MSE | FIT_VAL R² raw, corrected |
|---|---|---|---|
| U@0.9 | 1e-5 / 1e-6* / 1e-4 | 0.00692 / 0.00496 / 0.01099 | 0.709 / 0.732 / 0.509 |
| U@0.0 | 1e-6* / 1e-6* / 1e-6* | 0.000427 / 0.000418 / 0.000655 | 0.660 / 0.581 / 0.595 |
| U+@0.9 | 1e-4 / 1e-5 / 1e-4 | 0.00704 / 0.00465 / 0.01098 | 0.712 / 0.747 / 0.507 |
| U+@0.0 | 1e-5 / 1e-6* / 1e-4 | 0.000776 / 0.000660 / 0.001148 | 0.232 / 0.209 / 0.175 |
| U1@0.9 | 1e-5 / 1e-6* / 1e-4 | 0.00540 / 0.00494 / 0.01048 | 0.696 / 0.732 / 0.517 |
| U2@0.9 | 1e-5 / 1e-6* / 1e-5 | 0.00490 / 0.00480 / 0.01000 | 0.689 / 0.731 / 0.539 |
| U3@0.9 | 1e-5 / 1e-6* / 1e-5 | 0.00464 / 0.00460 / 0.00951 | 0.690 / 0.727 / 0.552 |
| C@0.9 | 1e-6* ×3 | 0.00969 / 0.00867 / 0.01424 | 0.489 / 0.575 / 0.383 |
| C@0.0 | 1e-6* ×3 | 0.000437 / 0.000486 / 0.000657 | 0.582 / 0.552 / 0.573 |
| C+@0.9 | 1e-6* ×3 | 0.01067 / 0.00850 / 0.01505 | 0.477 / 0.572 / 0.345 |
| C+@0.0 | 1e-6* / 1e-6* / 1e-5 | 0.000906 / 0.000823 / 0.001224 | 0.176 / 0.239 / 0.188 |
| age_U@0.9 | 1e4* ×3 | 0.01999 / 0.01430 / 0.02369 | −4.96 / −4.93 / −4.36 |
| age_U@0.0 | 1e-2 ×3 | 0.000989 / 0.000882 / 0.001504 | 0.202 / 0.140 / 0.104 |
| age_C@0.9 | 1e-1 ×3 | 0.00958 / 0.00862 / 0.01545 | 0.389 / 0.514 / 0.313 |
| age_C@0.0 | 1e-3 / 1e-6* / 1e-6* | 0.00100 / 0.00091 / 0.00150 | 0.106 / 0.230 / 0.092 |

ckpt2500 is in the ledger under `ckpt2500.seed{s}.head.*` and has the same pattern. Its one difference
in kind is `ckpt2500.seed0.head.age_U@0.9`, which selects λ = 1e-6 (the other five age_U@0.9 fits select
1e4) and has corrected R² 0.114 and corr(ψ̂, age) **+0.776** (the other five are −0.72 to −0.90).

Notes:
- **Many heads select λ at the lower grid edge (1e-6)**: every C@0.9 and C+@0.9 head, and U@0.0 on all
  three ckpt3000 seeds. The optimum may lie below the grid. The rule (A1.7) was followed as written.
- **age_U@0.9 selects λ = 1e4 (upper edge) on 5 of 6 fits** and its raw R² is about −4 to −5. The
  selection is on within-step demeaned MSE, which ignores a per-step offset. A heavily shrunk one-hot
  head with no intercept predicts about 0, so its raw MSE is about E[G²]. Its argmin order comes from
  the shape of `w`, not its level.

## ref and δ (§9.3, A1.11), FIT_VAL, ckpt3000

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| ref U@0.9 | **age** | fifo | fifo |
| ref C@0.9 | fifo | fifo | fifo |
| ref U@0.0 | **age** | fifo | fifo |
| ref C@0.0 | **age** | fifo | fifo |
| δ (γ 0.9 = γ 0) | 0.038022 | 0.036757 | 0.031858 |

δ mean 0.0355, sd 0.0033. At ckpt2500, the ref for U@0.9 is age / fifo / fifo; U@0.0 and C@0.0 are age / fifo / age;
C@0.9 is fifo ×3. δ is 0.037266 / 0.036426 / 0.037180.

- **The age-only head is FIFO on the C arm.** `ageC` accuracy equals `fifo` exactly at γ 0.9 on
  every seed and checkpoint, and `corr(age_C@0.9, age)` has Spearman −1.0 on all six fits. On seeds
  1 and 2 at ckpt3000, `ageU` also equals `fifo` exactly. Where ref = fifo came from a **tie**, the
  rule sends it to FIFO (§9.3).
- The age ref wins by small margins: seed 0 ageU 0.770994 vs fifo 0.770039 (γ 0.9).

## FIT_VAL all-query accuracy (γ = 0.9, ckpt3000; key `seed{s}.decisions.acc`)

| arm | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| ψ̂U | 0.6953 | 0.6869 | 0.7731 |
| ψ̂C | 0.7790 | 0.7754 | 0.8010 |
| FIFO | 0.7700 | 0.7663 | 0.7971 |
| ageU | 0.7710 | 0.7663 | 0.7971 |
| ageC | 0.7700 | 0.7663 | 0.7971 |
| random, 5-seed mean | 0.7499 | 0.7442 | 0.7511 |
| oracle | 0.9221 | 0.9134 | 0.9246 |
| kind-oracle | 0.8439 | **0.6354** | 0.8437 |
| kind-oldest | 0.8572 | **0.6791** | 0.8580 |
| fact/filler | 0.8444 | 0.8385 | 0.8406 |

FIT_VAL point differences (arithmetic, no CI; §9.5 reads these on EVAL, not here), ckpt3000:
- ψ̂U − ref_U: −0.0757 / −0.0795 / −0.0240 (mean −0.060, sd 0.031). That is roughly −0.7δ to −2.2δ.
- ψ̂C − ref_C: +0.0090 / +0.0091 / +0.0039 (mean +0.0073, sd 0.0029).
- ψ̂U − random mean: −0.055 / −0.057 / +0.022. ψ̂C − random mean: +0.029 / +0.031 / +0.050.
- oracle − FIFO: 0.152 / 0.147 / 0.127.
- ckpt2500: ψ̂U − ref_U −0.092 / −0.084 / −0.034; ψ̂C − ref_C +0.005 / +0.008 / +0.022.

**The kind-oracle is bimodal across seeds.** It scores 0.84 on seeds 0 and 2 at ckpt3000 but 0.635 on
seed 1. At ckpt2500 it scores 0.638 / 0.634 / 0.839. The FIT_TRAIN class means explain it. On the low
seeds the query class has the highest mean return, above assert
(ckpt3000 seed 1: assert 0.406, query 0.505, filler 0.342), so the kind-oracle evicts asserts before
queries. On the high seeds assert ranks above query (seed 0: 0.575 / 0.535 / 0.457). This is Q2's
comparator, so the ψ̂U − kind contrast (−0.149 / +0.051 / −0.071) changes sign with it.

## Age decodability (§8.1) with the F11 split

The bilinear form is fitted to predict age `t − i`, λ by raw FIT_VAL MSE. The split R² values are
correct as logged. Pooled values are corrected (see above). M-split row counts: U 647,168 (i < M) /
507,904 (i ≥ M); C 262,144 / 385,024.

| ckpt | rows | seed | pooled R² (corrected) | i < M | i ≥ M | λ |
|---|---|---|---|---|---|---|
| 3000 | U | 0 / 1 / 2 | 0.825 / 0.882 / 0.743 | 0.827 / 0.883 / 0.755 | 0.659 / 0.768 / 0.465 | 1e-6* ×3 |
| 3000 | C | 0 / 1 / 2 | 0.571 / 0.686 / 0.431 | 0.540 / 0.692 / 0.512 | 0.585 / 0.676 / 0.365 | 1e-6* ×3 |
| 2500 | U | 0 / 1 / 2 | 0.837 / 0.886 / 0.769 | 0.837 / 0.890 / 0.781 | 0.685 / 0.766 / 0.519 | 1e-6* ×3 |
| 2500 | C | 0 / 1 / 2 | 0.582 / 0.682 / 0.477 | 0.539 / 0.709 / 0.553 | 0.605 / 0.658 / 0.415 | 1e-6* ×3 |

- On U rows, age is more decodable for i < M than for i ≥ M on every seed: at ckpt3000 the mean is
  0.822 against 0.631. That matches F11's route, where the fill level is in `s_i` for early sentences.
- Age is still decodable at i ≥ M: R² from 0.37 to 0.77. It is not only the fill-level channel.
- The spread across seeds is large (i ≥ M, U, ckpt3000: sd 0.153).

## corr(ψ̂, age) on FIT_VAL (Pearson; n = 524,288 each; key `psi_age_corr_fit_val.*`)

| head | ckpt3000 s0 / s1 / s2 | ckpt2500 s0 / s1 / s2 |
|---|---|---|
| U@0.9 | +0.203 / +0.065 / +0.153 | +0.217 / +0.055 / +0.152 |
| U@0.0 | +0.142 / −0.170 / +0.055 | +0.134 / −0.156 / +0.027 |
| U+@0.9 | +0.188 / +0.092 / +0.151 | +0.206 / +0.084 / +0.155 |
| U1 / U2 / U3 @0.9 | +0.137 / +0.098 / +0.073 (s0) | see ledger |
| C@0.9 | −0.395 / −0.560 / −0.272 | −0.412 / −0.512 / −0.302 |
| C@0.0 | +0.008 / −0.306 / −0.009 | −0.061 / −0.306 / −0.057 |
| C+@0.9 | −0.426 / −0.558 / −0.305 | −0.438 / −0.518 / −0.332 |
| age_U@0.9 | −0.719 / −0.898 / −0.827 | **+0.776** / −0.887 / −0.814 |
| age_C@0.9 | −0.985 / −0.992 / −0.976 | −0.993 / −0.992 / −0.981 |

ψ̂-C is anti-correlated with age: a lower ψ̂ goes with an older slot, which is FIFO-like. ψ̂-U is weakly
**positively** correlated with age, the reverse of FIFO. The ordering agrees with the manifest's
expectation "psi-C closer to FIFO than psi-U". The sign of ψ̂-U was not pre-registered. The spread
across seeds is again large.

## Controls (per child; key `ckpt{c}.seed{s}.controls`)

All six children pass:
- identity probe worst |Δ| ≤ 5.96e-8;
- Σr worst 2.38e-7;
- vocabulary closure ok, 0 missing;
- C3 ok, with 0 argmax mismatches, max |Δacc| ≤ 2.95e-8 and max |ΔNLL| ≤ 5.53e-7, over 1188–1192 answers.

## Cost (per child; key `ckpt{c}.seed{s}.seconds`, `peak_rss_gb`)

| | capture s | fit s | FIT_VAL arms s | child wall h | peak RSS GB |
|---|---|---|---|---|---|
| ckpt3000 s0 / s1 / s2 | 2509 / 2636 / 2658 | 3013 / 2949 / 2956 | 3664 / 3670 / 3647 | 2.57 / 2.59 / 2.59 | 9.92 / 10.10 / 9.86 |
| ckpt2500 s0 / s1 / s2 | 2481 / 2513 / 2504 | 3093 / 3169 / 3199 | 3645 / 3745 / 3708 | 2.58 / 2.63 / 2.63 | 9.94 / 9.86 / 10.27 |

Review F7 guessed about 5–7 h per child at 1 thread. At 4 threads with 3 in parallel, the measured
figure is 2.6 h per child and 5.2 h in total.

## Class means (FIT_TRAIN, full-memory U rows; key `class_means`)

| ckpt / seed | γ 0.9: assert / query / filler |
|---|---|
| 3000 / 0 | 0.575 / 0.535 / 0.457 |
| 3000 / 1 | 0.406 / 0.505 / 0.342 |
| 3000 / 2 | 0.586 / 0.477 / 0.322 |
| 2500 / 0 | 0.592 / 0.594 / 0.415 |
| 2500 / 1 | 0.387 / 0.568 / 0.268 |
| 2500 / 2 | 0.571 / 0.521 / 0.262 |

## Erratum P0.2 (2026-09-29): the logged R² fixed and corrected without a refit

Everything above is kept as written. This section supersedes the defect section and the two table
entries named below. It is an erratum to a derived statistic; PREREG.md is untouched.

**Fix.** `fit_heads` now logs `1 - SSE/SST` and stamps `val_r2_form = "1-sse/sst"` on every fit.
It was the only site with the defect: `age_decodability` reads `fit_heads`'s value, and
`r2_split_by_index`, `within_step_r2` and `R2Acc` already used `1 - SSE/SST`. The test
`test_val_r2_raw_is_one_minus_sse_over_sst` pins the logged R² to an independent computation. At
3a90fce it failed (`-662.398` logged against `-2.6855` independent on the tiny fit). Battery
mutation `b2-psi-probe: the FIT_VAL R² multiplies a summed SSE by n_val again` puts the defect
back and turns that test red.

**Correction, no refit.** `experiments/b2-psi-probe/r2_erratum.py` writes
`runs/b2-psi-probe-fit/phaseA/ckpt{c}-seed{s}.r2-corrected.json` (every head, every λ on the path,
and both age-decodability rows) and `runs/b2-psi-probe-fit/ledger.r2-corrected.json` (102 ledger
rows). Each entry records the original value, `n_val_rows`, the corrected value and the formula
`R2_true = 1 - (1 - R2_logged) / n_val_rows`. The `.pt` payloads and `ledger.json` are unchanged.

**Independent check.** The age target `t - i` depends only on the row set, so the FIT_VAL SST can
be derived from the row structure. The U rows are every `i < t`. The C rows are the FIFO-resident
ones, `t - i <= M`, and this structure is confirmed because it reproduces the logged row counts and
both split counts exactly. The split code (F11) is separate code with the correct formula. From
it, `SSE_lt + SSE_ge` is the pooled SSE, and `1 - (SSE_lt + SSE_ge)/SST` equals the corrected
pooled R² on all 12 age-decodability rows (2 checkpoints × 3 seeds × U, C). The worst absolute
difference is **5.6e-16**. The ψ̂ heads' targets depend on the model, so for those heads the
correction is arithmetic only.

**The E0h input (U@0.0, selected λ), read through the sidecar:**

| ckpt | seed | logged | n_val_rows | corrected | E0h gate `val_r2 > 0` |
|---|---|---|---|---|---|
| 3000 | 0 | −351034.9223 | 1,032,192 | 0.659912 | informative |
| 3000 | 1 | −432716.4940 | 1,032,192 | 0.580778 | informative |
| 3000 | 2 | −418435.6110 | 1,032,192 | 0.594614 | informative |
| 2500 | 0 | −343001.1841 | 1,032,192 | 0.667695 | informative |
| 2500 | 1 | −358523.8695 | 1,032,192 | 0.652657 | informative |
| 2500 | 2 | −392399.0954 | 1,032,192 | 0.619838 | informative |

`e0h_compute` now reads a pre-fix payload only through this sidecar, and only when the sidecar's
`original` matches the payload. If the sidecar is missing or does not match, it raises and exits 3.
**E0h itself has not run.** `run.py e0h` needs the EVAL-phase accumulators
`phaseB/ckpt3000-seed{s}.e0h.pt`, and phase B has not run: `e0h_main` prints `DID NOT RUN: E0h
inputs absent` and exits 3. No E0h class or within-step R² exists. When it does run, the thresholds
are still unratified (`e0h_ratified() = False`), so rc is 2 whatever the values.

**Two rounding slips in the head table above** (ckpt3000 seed 0; the sidecar values govern):
C+@0.9 is 0.4765, printed as 0.477 where it should be 0.476. age_C@0.9 is 0.3885, printed as 0.389
where it should be 0.388. Every other R² in the two tables matches the sidecars to the printed
precision.
