---
run_id: b5-convergence
roadmap: "PLAN-v4 §2B B5 (substrate convergence), a robustness secondary"
question: >-
  Does fresh-stream arm B's stream answer loss reach a plateau after its ckpt 3000 if the same
  run is continued on the same never-repeating stream, and if so at which common step, and does
  that checkpoint still retrieve through memory?
seeds: [0, 1, 2]
arm: >-
  resume a COPY of .worktrees/fresh-stream/runs/fresh-stream/B/seed<s>/ckpt-003000.pt (model,
  optimizer, policy, RNG) into runs/b5-convergence/B/seed<s>/; train steps 3000 -> 9000 on
  fresh-stream's stream (step t trains documents [4160 + 16 t, 4160 + 16 t + 16)); 3 seeds as
  parallel children x 4 threads = 12 threads (cpu-det, --slots 12)
start_checkpoints: "the three B ckpt-003000.pt files; sha256 must equal their line in ~/rsr-substrate/2026-09-27/MANIFEST.sha256"
window: 250 steps
plateau_rule: >-
  per seed, at evaluation step E (a multiple of 250): OLS of the 16 window means of heartbeat
  loss_answer_tokens over [E - 4000, E) on window midpoints in thousands of steps; the 95% CI
  (Student t, 14 df) of the slope, divided by the mean of the 16 window means and times 100,
  lies strictly inside (-1, +1) % per 1000 steps
common_step: "E* = the first E at which the plateau rule holds on every seed"
cap: 9000
controls_at_candidate: "held-out gap_2_to_M live.answer_acc - slots_zeroed.answer_acc >= 0.03 on every seed (fresh-stream's R)"
decision_rule: >-
  A failed control or a measurement that raised makes the result inconclusive (exit 3). E* is
  the first evaluation step E in {7000, 7250, ..., 9000} at which the plateau rule holds on every
  seed. PLATEAU at E* if E* exists and R(E*) holds on every seed; PLATEAU_WITHOUT_RETRIEVAL at E*
  if E* exists and R(E*) fails on some seed; NO_PLATEAU by 9000 if no E* exists by the cap. In
  every case ckpt3000 stays primary; making another checkpoint the substrate is D2, owner-only.
---

# PREREG: B5, substrate convergence (is arm B still learning after ckpt 3000?)

**Written 2026-09-30, about 02:30 PDT, by a model (Claude Opus 5.5, a researcher session on the
Studio), dispatched from PLAN-v4 §2B B5. Not written by Brendan.** It is committed alone, before
any B5 code or data. Amendments are appended, dated, and name the data that prompted them. **No
amendment may be made after B5 data exists.**

## Status of this experiment

B5 is a **robustness secondary. ckpt3000 stays primary for everything**
(`R-2026-09-27-retrieval-shown` is scoped to `B.ckpt3000`). Nothing this run finds moves the
substrate. **Making any later checkpoint the substrate is D2, and D2 is owner-only.** This run
reports; it does not decide.

## Everything already seen (before this PREREG)

- `R-2026-09-27-retrieval-shown`: B.ckpt3000 held-out gap 2..16 answer accuracy is live
  0.92068 / 0.90859 / 0.93220 against `slots_zeroed` 0.06091 / 0.05540 / 0.05226.
- Arm B's heartbeats (`.worktrees/fresh-stream/runs/fresh-stream/B/seed<s>/heartbeat.jsonl`, read
  only) give the stream answer loss, as 250-step window means:

  | seed | 1000 | 1250 | 1500 | 1750 | 2000 | 2250 | 2500 | 2750 |
  |---|---|---|---|---|---|---|---|---|
  | 0 | 2.3000 | 1.3944 | 1.0373 | 0.9433 | 0.9080 | 0.8891 | 0.8666 | 0.8537 |
  | 1 | 1.8966 | 1.1084 | 0.9873 | 0.9398 | 0.9238 | 0.8940 | 0.8770 | 0.8626 |
  | 2 | 2.2577 | 1.6018 | 1.2183 | 1.0439 | 0.9482 | 0.8945 | 0.8553 | 0.8091 |

  Over steps 2000–2999, the OLS slope is −8.34 / −8.93 / −21.15 % per 1000 steps. **The loss is
  still falling at step 3000**, which is why B5 exists.
- fresh-escape (arm A, 3000 → 9000) ran at 3 × 5 threads with no measured s/step at 4 threads;
  3.46 s/step was measured at 5 threads (PLAN-v4).

## The plateau rule (replaces v2's noise-floor rule)

- **Series.** The heartbeat key `loss_answer_tokens` per optimizer step: the answer-token NLL on
  that step's training documents. The stream never repeats a document, so every step's documents
  are unseen when it is taken.
- **Windows.** Window `w` covers steps `[250 w, 250 w + 250)`. Its value is the mean over all 250
  steps; a window missing any step does not exist. Only B5's own steps (`≥ 3000`) are used, so
  every window comes from one run at one thread count.
- **Evaluation step E.** A multiple of 250. The rule reads the 16 windows covering `[E − 4000, E)`.
  These are exactly the steps that produced `ckpt-E`, which holds the state after steps `[0, E)`.
  The first E with 16 own windows is **7000**, so `E ∈ {7000, 7250, …, 9000}` (9 evaluations).
- **Statistic.** OLS of the 16 window means `y` on the window midpoints `x`, in thousands of steps.
  The slope is `b`, with `SE = sqrt(RSS / 14 / Sxx)` and `CI = b ± t(0.975, 14) · SE`, where
  `t = 2.1447866879`. The relative CI is `100 · CI / mean(y)`, in % per 1000 steps.
- **Holds for seed s at E** iff the relative CI lies strictly inside `(−1, +1)`.
- **Common step.** `E*` is the **first E at which the rule holds on every seed**, as one common
  step, not a per-seed step. The candidate substrate checkpoint is `ckpt-E*` on every seed.

### Power calculation (the rule's expected half-width)

This assumes independent window residuals, measured on arm B's steps 2000–2999 above (pre-existing
data):

| seed | per-step SD / mean | lag-1 autocorr. (detrended) | window SD / mean = ÷√250 | half-width |
|---|---|---|---|---|
| 0 | 0.1021 | 0.012 | 0.646 % | 0.300 %/1000 |
| 1 | 0.1009 | −0.015 | 0.638 % | 0.297 %/1000 |
| 2 | 0.1156 | 0.029 | 0.731 % | 0.340 %/1000 |

The midpoints are 0.25 apart, so `Sxx = 0.25² · Σ_{i=0}^{15}(i − 7.5)² = 0.0625 · 340 = 21.25`,
and `√Sxx = 4.610`. The half-width is `t · σ_w / √Sxx = 2.1448 / 4.610 · σ_w = 0.4653 σ_w`,
which gives **0.297–0.340 % per 1000 steps**, matching PLAN-v4's 0.30–0.34. The rule therefore
fires only when `|b|/mean ≲ 0.66–0.70 %` per 1000 steps.

Caveats, stated before the data:
- The CI uses the fitted residuals, not the assumed σ_w. Curvature, a still-decelerating decline,
  leaves structured residuals that inflate RSS and widen the CI, so it cannot make the rule fire
  early.
- Positive autocorrelation of the window residuals would make the OLS SE too small. The measured
  per-step lag-1 is about 0, and a 250-step window averages over it.
- The relative scale divides by the window mean. The loss is lower later, which raises the
  relative noise.

## Cap

**Step 9000.** Sixteen own windows first exist at 7000, which is at or below 9000, so the
"or later if 16 windows need it" clause does not extend the cap. Training continues to 9000
whether or not E* is found earlier. E* is a first-crossing, so later data cannot change it, and
the full trajectory is measured.

## Controls (checked first; each overrides everything)

1. **T0.** `shasum -a 256 -c ~/rsr-substrate/2026-09-27/MANIFEST.sha256` from the repo root
   returns rc 0 with 435 OK, at the start and at the end. **The run never writes into a protected
   directory.** The code refuses any output directory that is inside a directory named in the T0
   manifest, or inside `.worktrees/fresh-stream`.
2. **Start checkpoints.** Each `B/seed<s>/ckpt-003000.pt` has the sha256 of its T0 manifest line
   and stores step 3000. It is **copied** into `runs/b5-convergence/B/seed<s>/`, and the copy's
   sha256 equals the source's. The source is only read.
3. **Measurement path unchanged.** Each copied start checkpoint is re-measured with fresh-stream's
   instrument (corpus-size `measure_checkpoint`, n = 64, at fresh-stream's parent thread count).
   Every `B.ckpt3000.*` statistic key of `runs/fresh-stream/ledger.json` must match per seed
   within 1e-6. On failure, a missing key or a raise: exit 3, and nothing trains.
4. **Stream.** The ids of steps `[3000, 9000)` are disjoint from held-out `[4096, 4160)` and probe
   `[0, 64)` and never repeat. Vocabulary closure holds per seed over them and both measurement
   sets (fresh-stream's functions).
5. **Resume is exact.** Every child segment compares its model and optimizer state after load
   with the checkpoint it loaded, exactly, before its first step (fresh-stream's `ResumeCheck`).
6. **Resume proof (before launch; one seed).**
   - **(a)** From arm B's own `ckpt-002900.pt` (a copy), train 100 steps at **5 threads**, the
     original's count. The result must equal arm B's `ckpt-003000.pt` exactly (model, optimizer,
     RNG). This is the original trajectory, where that is checkable.
   - **(b)** The same at **4 threads**. This is reported, not gated: a mismatch means B5's
     4-thread continuation is numerically distinct from what 5 threads would have produced,
     which is stated and not hidden.
   - **(c)** At 4 threads, resume its own `ckpt-002950.pt` to 3000. The result must equal the
     uninterrupted 4-thread `ckpt-003000.pt` exactly. This is the kill-and-resume B5 relies on.

   If **(c)** fails, B5 is not launched. If **(a)** fails, it is reported with its cause and the
   launch still proceeds, because B5 runs at 4 threads and (c) is what its resumability rests on.

## Readouts

- **Primary: the controls at the candidate checkpoint.** At `ckpt-E*`, on every seed, held-out
  `gap_2_to_M`: **R(E*)** = `live.answer_acc − slots_zeroed.answer_acc ≥ 0.03` on every seed.
  This is fresh-stream's R and DELTA, the same readout behind `R-2026-09-27-retrieval-shown`.
  **C** = `live.answer_brier_over_16`, reported only.
- **Trajectory (secondary, no verdict).** Also measured at 4000, 5000, 6000, 7000, 8000 and 9000:
  R, C, probe R, and every S0-03 quantity.
- **Rule trace.** Per seed and E: the slope, the CI, the window mean, and whether the rule holds.
  Every window mean is kept in the ledger.

| condition | result |
|---|---|
| a control fails; a measurement raised | `inconclusive`, exit 3 |
| E* exists (≤ 9000) and R(E*) on every seed | **PLATEAU at E*** — the stream answer loss is flat within ±1 %/1000 steps on every seed from E*, and that checkpoint still retrieves through memory |
| E* exists and R(E*) fails on some seed | **PLATEAU_WITHOUT_RETRIEVAL at E*** — flat, but not a retrieval substrate |
| no E* by 9000 | **NO_PLATEAU by 9000** — this is itself the result (fallback) |

**Fallback.** No plateau by the cap is the result. Under every row, **ckpt3000 stays primary**,
and choosing another substrate checkpoint is **D2, owner-only**.

**Before the cap.** PLATEAU may be declared once E* is found and `ckpt-E*` is measured on every
seed, because E* cannot move afterwards. NO_PLATEAU can be declared only at 9000. A run stopped
earlier (the session window ends, a kill) is reported **in progress**, with the step reached and
the rule trace; it is resumable from its own latest checkpoint.

## What counts as done

- Every seed has `steps_done = 9000`.
- Controls 1–5 pass, and 6(c) passed before launch.
- Every listed checkpoint (and `ckpt-E*`) is measured on every seed.
- `runs/b5-convergence/ledger.json` has `status: ok`.
- `experiments/b5-convergence/RESULTS.md` is rendered from that ledger.
- T0 is verified at the end.

A status RESULTS rendered from a `partial` ledger is **not** done.

## Author's expectation (before any number from this run; allowed to be wrong)

- **NO_PLATEAU by 9000: about 55 %.** At 3000 the decline is 8–21 % per 1000 steps. The rule
  needs it below about 0.7 %, sustained over a 4000-step window, which is a 10–30× slowdown.
- **PLATEAU: about 40 %,** most likely late (E* ≥ 8000).
- **PLATEAU_WITHOUT_RETRIEVAL: about 5 %.** Retrieval at ckpt3000 is large (R ≈ 0.86), and
  continued training on the same objective is unlikely to remove it.

## What this does not establish

- A NO_PLATEAU bounds only this budget, learning rate and thread count.
- A PLATEAU says the training loss is flat, not that the model is optimal. It does not change
  the substrate (D2).
- Nothing about RSR, eviction, ψ̂ or Kintsch & van Dijk: FIFO only, no retention loss.
- No scaling claim.
