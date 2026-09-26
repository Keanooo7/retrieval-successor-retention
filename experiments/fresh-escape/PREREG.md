---
run_id: fresh-escape
question: >-
  Does a fresh-start TG model escape the chance plateau on a non-repeating stream if it is simply
  trained longer? Continue fresh-stream arm A from its ckpt 3000 to ckpt 9000 on the same stream.
seeds: [0, 1, 2]
arm: "resume .worktrees/fresh-stream/runs/fresh-stream/A/seed<s>/ckpt-003000.pt (model, optimizer, policy, RNG); train steps 3000 -> 9000 on fresh-stream's stream (step t trains documents [4160 + 16 t, 4160 + 16 t + 16)); 3 seeds as parallel children x 5 threads"
start_checkpoint_sha256:
  seed0: f8adcebeb730d160d62ce3d46de0e173a82fe25efd4730a32bf733bf607a2e64
  seed1: 9763cbc61f06496919058036113583b5f57a2aa923ab7d5e80b1b35882d4bb98
  seed2: f79c006f1a23f9d72cd9d8ba72299071ac2e738997d5a7965e1f42f6c76545fb
checkpoints_measured: [4000, 5000, 6000, 7000, 8000, 9000]
instrument: "corpus-size measure_checkpoint (n = 64), imported -- held-out [4096, 4160), probe [0, 64)"
thresholds:
  DELTA: "0.03 -- unchanged from scaffold-timing, fresh-stream and scaffold-dose"
  C: "0.8875 = 0.9375 - 0.05 -- fresh-stream's"
  stream_loss_threshold: "ln 16 - 0.10 = 2.6726 -- fresh-stream's"
  MIN_PRIMARY_CKPT: 6000
  deadline: "2026-09-26 08:30 America/Los_Angeles, absolute, parent-enforced"
decision_rule: >-
  A failed control or a measurement that raised makes the result inconclusive (exit 3). The
  primary checkpoint P is the largest listed checkpoint measured on every seed before the
  deadline; if P < 6000 the result is inconclusive. ESCAPES if R(P) holds on every seed;
  STIRRING if R(P) fails but at least one seed's stream answer loss has a 100-step window below
  2.6726 at or before P; NO_ESCAPE otherwise, reported as NO_ESCAPE by P.
---

# PREREG — fresh escape: is memorisation necessary, or does it only make learning faster?

**Written 2026-09-25, ~21:45 PDT, by a model (Claude Opus 5.5, interactive session on the Studio),
at Brendan's request for an overnight run ("make sure there is a test running over night").
Not written by Brendan.** Committed alone, before any code for it. Amendments are appended, dated,
and name the data that prompted them.

## Everything already seen

- **fresh-stream** (`runs/fresh-stream/ledger.json`, merged in PR #45): arm A, a fresh model on a
  stream of never-repeated documents, sat at chance for 3000 steps. `A.ckpt3000.R_quantity` =
  0.0085 / 0.0 / −0.0014; `A.ckpt3000.C_quantity` = 0.9379 / 0.9374 / 0.9378;
  `A.stream_answer_loss_first_window_below` null on every seed. The stream answer loss drifted
  from about 2.785 (steps 300–499) to about 2.779 (steps 2000–2999), a creep of roughly 0.001 per
  several hundred steps. Arm B (memorised start) retrieved at R ≈ 0.85–0.88.
- **scaffold-dose** (PR #45): resuming the memorised run at step 300 or later unlocks retrieval
  within 100–300 stream steps; at step 100 nothing moves, and at step 200 one seed moves.
- **corpus-size** (PR #44): N = 512 seed 2, with repetition but no memorisation, showed a first
  movement near step 900–1000.

## Why this run

Today's headline says memorisation is *sufficient* to unlock the memory. The open question is
whether it is *necessary*, or whether a fresh start would also get there with more training.
This changes one factor: training length. Everything else is fresh-stream arm A, continued
exactly from where it stopped, on the same stream (the next documents in order, all unseen).

## Controls (checked first; each overrides everything)

1. **Measurement path unchanged.** Re-measure `A/seed<s>/ckpt-003000.pt` for every seed and
   compare every statistic key with `runs/fresh-stream/ledger.json` `A.ckpt3000.*` within 1e-6.
   Fail → exit 3, nothing trains.
2. **Resume is exact.** Each child's model and optimizer state after load equals the
   checkpoint's tensors exactly (fresh-stream's resume check, imported).
3. **Start checkpoints.** sha256 of each start checkpoint equals the front matter; stored step is
   3000. **Stream disjointness and vocabulary closure** over the ids used,
   `[4160 + 16·3000, 4160 + 16·9000)`, against held-out `[4096, 4160)` and probe `[0, 64)`.

## Primary readout

At each measured checkpoint c, held-out `gap_2_to_M`, per seed:
**R(c)** = `live.answer_acc − slots_zeroed.answer_acc ≥ 0.03` on every seed.
**C(c)** = `live.answer_brier_over_16 ≤ 0.8875` on every seed (reported, not classified).

| condition | result |
|---|---|
| a control fails; a measurement raised | `inconclusive`, exit 3 |
| P (largest checkpoint measured on every seed before the deadline) < 6000 | `inconclusive` |
| R(P) | **ESCAPES** — a fresh start learns memory-mediated retrieval with enough steps; memorisation speeds it up rather than being required |
| not R(P), and some seed's stream loss has a 100-step window < 2.6726 at or before P | **STIRRING** — learning has begun on at least one seed but has not reached R by P |
| otherwise | **NO_ESCAPE by P** — no sign of learning in P steps of a non-repeating stream |

The deadline is absolute because this run shares the machine with the full mutation battery;
wall time is a prediction (about 3.33 s per step alone, so about 5.6 h for 6000 steps, longer
under contention).

## Secondary readouts (no verdict)

R, C and every S0-03 quantity at every measured checkpoint; the stream answer loss per 100-step
window for steps 3000 onward and the first window below 2.6726 per seed; probe accuracy (the
probe is unseen by arm A); memory gates; side-channel share.

## Author's expectation (before any number from this run; allowed to be wrong)

NO_ESCAPE by 9000 (about 60 %). If anything moves, it is one seed, late, reading as STIRRING
(about 30 %). ESCAPES about 10 %. The basis is the flat 2.78 plateau from step 500 to 3000 and
the very slow creep.

## What this does not establish

A NO_ESCAPE says nothing about longer runs, other learning rates or other architectures; it
bounds only this budget. ESCAPES would weaken "memorisation is necessary" but leave "memorisation
is sufficient and much faster" standing. Nothing about RSR, eviction, ψ̂ or Kintsch & van Dijk.
No scaling claim.
