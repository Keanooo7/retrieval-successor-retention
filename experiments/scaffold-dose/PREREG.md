---
run_id: scaffold-dose
question: >-
  How much memorisation unlocks learning from a non-repeating stream? Resume the corpus-size
  N = 64 arm at step k and train 500 fresh-stream steps: which k unlocks memory-mediated
  retrieval on unseen documents?
seeds: [0, 1, 2]
arms_k: [100, 200, 300, 400, 600]
arm: "resume runs/corpus-size-curve/n64/seed<s>/ckpt-{k:06d}.pt (model, optimizer, policy, RNG); train steps k -> k + 500 on the fresh-stream stream (step t trains documents [4160 + 16 t, 4160 + 16 t + 16)); measure ckpt k + 500"
arm_order: "600, 400, 300, 200, 100 -- one arm at a time, 3 seeds as parallel children x 5 threads"
instrument: "fresh-stream's train(stream=...) and measurement path, imported (corpus-size measure_checkpoint, n = 64)"
thresholds:
  DELTA: "0.03 -- unchanged from scaffold-timing and fresh-stream"
  stream_loss_threshold: "ln 16 - 0.10 = 2.6726 -- fresh-stream's"
  reproduction_tolerance: "exact (resume) / 1e-6 (measurement)"
  deadline: "2026-09-26 06:00 America/Los_Angeles, absolute, parent-enforced"
decision_rule: >-
  A failed resume check, a measurement that raised, or a missing end checkpoint for any seed of
  any arm makes that arm's cell unreadable; the result is inconclusive if any arm is unreadable.
  Otherwise U(k) holds when R holds at ckpt k + 500 on every seed. k* is the smallest k in the
  sweep with U(k). The reading is from the table below.
---

# PREREG — scaffold dose: how much memorisation unlocks learning from the stream?

**Written 2026-09-25, ~17:55 PDT, by a model (Claude Opus 5.5, interactive session on the
Studio), under Brendan's go to run step 3 and "continue based on what you believe to be the next
steps". Not written by Brendan.** Committed alone, before any code for it. Amendments are
appended, dated, and name the data that prompted them.

## Everything already seen

- **fresh-stream** (`runs/fresh-stream/ledger.json`, `6d143be`, PREREG `83128ee`):
  **SCAFFOLD.** Arm A (fresh init, 3000 steps of a non-repeating stream) never left chance:
  `A.stream_answer_loss_first_window_below` null on every seed, `A.R3000` false. Arm B (N = 64
  ckpt 1000, then 2000 stream steps) retrieves on unseen documents: `B.ckpt3000.R_quantity`
  0.8598 / 0.8532 / 0.8799, `B.C3000` true, and its stream loss crossed the threshold in the
  first window (`B.stream_answer_loss_first_window_below` 1100 / 1000 / 1000). So at k = 1000
  the unlock happens within ≤ 200 steps. Held-out `gap_gt_M` stays near chance (the leak check).
- **scaffold-timing** (`24baf97`): `onset_M` = 300, `onset_R` = 600 at N = 64. Training-probe
  accuracy by checkpoint is in that ledger (`n64.ckpt<k>.train.live.gap_2_to_M.answer_acc`).

## Condition

Identical to fresh-stream arm B except the start checkpoint (step k) and the length (500 steps).
Every stream document is unseen: ids start at 4160 + 16k ≥ 5760, disjoint from held-out
`[4096, 4160)` and probe `[0, 64)`. The start checkpoints' sha256 must match
`runs/scaffold-timing/manifest.json`. 500 steps is chosen because the k = 1000 unlock completed
inside 200 steps; it is a budget, and a non-unlock at 500 steps is read as "not within 500".

## Controls

1. **Resume is exact** per arm and seed: model and optimizer tensors after load equal the
   checkpoint's, exactly (fresh-stream's control 2 check, imported).
2. **Measurement path unchanged:** before the arms, re-measure the k = 600 start checkpoint for
   every seed and compare every statistic key with `runs/scaffold-timing/ledger.json`
   `n64.ckpt600.*` within 1e-6. Fail → exit 3, no arm runs.
3. Stream disjointness and vocabulary closure, as fresh-stream.

No new training code is needed: `train(stream=...)` from `e45e4fe` is used unchanged, and
fresh-stream's control 1 already proved its default path.

## Primary readout

At ckpt k + 500, held-out `[4096, 4160)`, `gap_2_to_M`, per seed:
**R** = `live.answer_acc − slots_zeroed.answer_acc ≥ DELTA`. **U(k)** = R on every seed.
k* = the smallest k with U(k).

| result | reading |
|---|---|
| any arm unreadable | `inconclusive`, exit 3 |
| U(k) for no k in the sweep | **UNLOCK_AFTER_600** — more than retrieval onset is needed, or 500 steps is too few below k = 1000 |
| k* = 600 | **AT_RETRIEVAL_ONSET** — unlocking needs the held-out retrieval the model had already started to show |
| k* ∈ {300, 400} | **AT_MEMORISATION** — unlocking comes with memorisation, before held-out retrieval is measurable |
| k* ∈ {100, 200} | **EARLY** — little or no memorisation is needed; the scaffold may be generic early training on repeated data |

**Monotonicity** is reported, not assumed: if U(k) holds at some k and fails at a larger k, the
report names it, and k* is still the smallest k.

## Secondary readouts (no verdict)

Per arm and seed: stream answer loss per 100-step window and the first window below 2.6726;
C (held-out live Brier16 ≤ 0.8875); held-out accuracy live / slots_zeroed / gate_zeroed_bos_off
in every bucket; probe accuracy; memory gates; the start checkpoint's R and M quantities (from
the scaffold-timing ledger) beside the end values.

## Author's expectation (before any number from this run; allowed to be wrong)

k* = 400: k = 600 unlocks on every seed; k = 400 on every seed but slowly; k = 300 on one or
two seeds; k = 200 and 100 do not. The unlock is fast (under 200 steps) wherever it happens.

## Time

5 arms × 500 steps at ~3.33 s/step (fresh-stream heartbeats) ≈ 2.3 h, plus ~15 min of
measurement. A prediction.

## What this does not establish

Nothing about RSR, eviction, ψ̂ or Kintsch & van Dijk. It is a curriculum fact about TG at d =
128 on the synthetic corpus. It does not show why memorisation unlocks learning, only how much
is needed. The RSR gate stays the owner's reading.
