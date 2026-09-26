---
run_id: fresh-stream
question: >-
  With no training document ever repeated, does TG's working memory come to carry answer
  information on unseen documents -- from scratch (arm A), and when started from a model that
  already acquired it by memorising 64 documents (arm B)?
seeds: [0, 1, 2]
arms:
  A: "fresh init (same init as the N = 64 arm of each seed), steps 0 -> 3000 on the stream"
  B: "resume runs/corpus-size-curve/n64/seed<s>/ckpt-001000.pt (model, optimizer, policy, RNG), steps 1000 -> 3000 on the stream"
arm_order: "A then B, one arm at a time, 3 seeds as parallel children x 5 threads"
stream: "step t trains on documents [4160 + 16 t, 4160 + 16 t + 16) of the seed's generator; no document repeats; A and B see identical documents at every step t in [1000, 3000)"
heldout: "documents [4096, 4160) (PRIMARY, unseen by both arms)"
probe: "documents [0, 64) -- unseen by A (a second held-out set); memorised by B (forgetting readout)"
checkpoints_measured: {A: [300, 1000, 1500, 2000, 2500, 3000], B: [1500, 2000, 2500, 3000]}
thresholds:
  DELTA: "0.03 accuracy per seed -- scaffold-timing's, unchanged"
  BRIER_MARGIN: "0.05 -- corpus-size curve's, unchanged; C threshold is 0.9375 - 0.05 = 0.8875"
  reproduction_tolerance: "1e-6 absolute"
  deadline: "2026-09-26 06:00 America/Los_Angeles, absolute, parent-enforced"
decision_rule: >-
  A failed or incomplete reproduction control, a measurement that raised, or ckpt 3000 missing
  for any seed of either arm makes the result inconclusive (exit 3). Otherwise read R3000 per
  arm (R at ckpt 3000 on every seed): A and B -> DATA_SUFFICES; B only -> SCAFFOLD; A only ->
  TRAP; neither -> NEITHER.
---

# PREREG — fresh stream: can the memory learn to retrieve without repeated documents?

**Written 2026-09-25, ~12:10 PDT, by a model (Claude Opus 5.5, interactive session on the
Studio), under Brendan's explicit go to run step 3 and continue without checking back. Not
written by Brendan.** Committed alone, ahead of any code for it. Amendments are appended, dated,
and name the data that prompted them; the text above is never edited.

## What is already known (every number below has been seen by the author)

- **Scaffold timing** (`runs/scaffold-timing/ledger.json`, `24baf97`, PREREG `167d650`,
  independently verified): at N = 64, **AFTER** — `onset_M` 300, `onset_R` 600. Unseen-document
  retrieval (R) is sustained only once training accuracy is ≥ 0.9 on seeds 0 and 2
  (`n64.ckpt600.train.live.gap_2_to_M.answer_acc` 0.9109 / 0.9815 / 0.9831). The train
  memory share rises before R. That PREREG committed this run to **both** arms.
- **Against the scaffold being necessary:** N = 512 seed 2 reaches R 0.0367 at ckpt 1000 with no
  memorisation (`n512.ckpt1000.R_quantity`; train accuracy never above 0.092). Its training
  heartbeat's answer loss, 100-step mean, is 2.757 over steps 900–999, below ln 16 = 2.7726
  (`runs/corpus-size-curve/n512/seed2/heartbeat.jsonl`, `loss_answer_tokens`). N = 4096 by
  contrast sits at 2.780–2.783 from step 500 on (all seeds), and reaches R on no seed.
- N = 4096 at 1000 steps saw each document ~3.9 times; its null is uninformative (corpus-size
  BRIEF-ERRORS).

## Condition

S0-03's `CONFIG`, imported, as in the corpus-size curve (`d=128, steps_per_stream=48, batch=16,
max_tokens=64, memory_slots=16, lr=1e-3, policy="fifo", masked_loss=True,
srep_norm_reg_weight=0.0`). Constant learning rate (no schedule exists in `train()`), so
resuming at step 1000 does not change the optimiser's schedule.

**The only change to the training code:** `train()` gains an optional argument (name chosen by
the implementation) that replaces the with-replacement sampler (`torch.randint`,
`src/rsr/train/loop.py`) by the deterministic stream above. Documents are generated per id with
the generator's per-document function (prefix-stable: seed `seed · 1_000_003 + doc_id`,
`src/rsr/data/synthetic.py:234-238`), lazily or in blocks, never all held at once if that would
exceed memory. Its default (`None`) is today's path, byte for byte.

**Vocabulary.** The token map is S0-03's, built from documents `[0, 64)` (so arm B's embedding
matrix is the one its checkpoint was trained with). Before training, the run proves every word
of every stream document it will use (`[4160, 4160 + 16 · 3000)`) and of both measurement sets
is in that map; if not, exit 3 before any step.

**Arm B** calls `train(resume=<ckpt-001000.pt>, iters=3000, <stream>)`. Model, optimizer,
policy and RNG are restored. It trains steps 1000–2999, the same documents as arm A at those
steps.

**Arm A** calls `train(iters=3000, <stream>)` from `torch.manual_seed(seed)`: the same
initialisation as the corpus-size N = 64 arm of that seed.

Threads: 5 per seed, three seeds as parallel child processes (the corpus-size training
setting). Arms run A then B. Checkpoints are written every 100 steps; the listed ones are
measured.

## Instrument

The corpus-size curve's `measure_checkpoint(seed_dir, seed, label, n=64)`, imported (the same
object scaffold-timing used): S0-03's `measure(sets=doc_sets(seed, 64))`, held-out
`[4096, 4160)`, probe `[0, 64)`, conditions `live`, `slots_zeroed`, `gate_zeroed`,
`gate_zeroed_bos_off`, every bucket, Brier16, accuracy, NLL.

## Reproduction controls (checked first; each overrides everything)

1. **Default path unchanged.** Before either arm, train the corpus-size N = 64 configuration
   through the modified `train()` **without** the stream argument, to step 100, all 3 seeds
   (~6 min). Its `ckpt-000100.pt`, measured, must equal `runs/scaffold-timing/ledger.json`'s
   `n64.ckpt100.*` statistic keys per seed within 1e-6. Fail → exit 3, no arm runs.
2. **Resume is exact.** Arm B's model and optimizer state after load, before its first step,
   must equal the tensors in `ckpt-001000.pt` exactly (checked in the run, per seed, recorded).
3. **Stream is disjoint.** The run asserts no stream document id lies in `[0, 64)` or
   `[4096, 4160)` and no id repeats.

## Primary readout

Per arm, per measured checkpoint c, on the held-out set at `gap_2_to_M`:
- **R(c)**: `heldout.live.answer_acc − heldout.slots_zeroed.answer_acc ≥ DELTA` on **every** seed.
- **C(c)**: `heldout.live.answer_brier_over_16 ≤ 0.8875` on **every** seed (better than the
  uniform forecast by BRIER_MARGIN: calibrated enough to be usable).

**Classification**, from R at ckpt 3000 (R3000) of each arm:

| condition | result |
|---|---|
| a control fails or is incomplete; a measurement raised; ckpt 3000 missing for any seed of either arm | `inconclusive`, exit 3 |
| R3000(A) and R3000(B) | **DATA_SUFFICES** — repetition is not needed; the substrate learns retrieval from a non-repeating stream |
| R3000(B), not R3000(A) | **SCAFFOLD** — retrieval acquired by memorising survives a non-repeating stream, but is not acquired from it in 3000 steps |
| R3000(A), not R3000(B) | **TRAP** — the memorised start prevents what a fresh start learns |
| neither | **NEITHER** — not acquired from the stream, and the memorisation-acquired retrieval does not survive it |

A second line is reported per arm: **usable** if R3000 and C3000 both hold. The RSR gate
("memory helps on unseen documents") is read by the owner from this line; this run does not
open it.

⚠️ **Arm B starts with R already true** (scaffold-timing: `n64.ckpt1000.R_quantity` 0.0609 /
0.1122 / 0.0720). So R3000(B) reads *survival* of retrieval acquired by memorisation, not new
acquisition. Growth is reported (below), not classified.

## Secondary readouts (reported, no verdict)

- R, C and every S0-03 quantity at every measured checkpoint of both arms.
- **B growth:** `R_quantity(3000) − R_quantity(1000)` per seed, where the ckpt-1000 value is the
  scaffold-timing ledger's.
- **B forgetting:** probe (training-document) accuracy, live and slots zeroed, 1500 → 3000.
- **A on the probe**, which is unseen by A: R computed on the probe, as a second held-out read.
- **Stream answer loss:** `loss_answer_tokens`, 100-step means, both arms, from the heartbeats.
  In arm A every training document is unseen when trained on, so this is an on-line
  unseen-document loss. Report the first 100-step window whose mean is below 2.7726 − 0.10 on
  each seed, if any.
- Side-channel share and memory gates, as in scaffold-timing.

## Author's expectation (before any number from this run; allowed to be wrong)

- Arm A: the stream answer loss leaves the 2.78 plateau on at least one seed by step 3000 —
  N = 512 seed 2 began moving near step 900 with repetition — but R3000(A) holds on every seed
  somewhat less often than not (≈ 40 %).
- Arm B: R survives (≈ 70 %) and does not grow much; held-out Brier improves from ~1.4 toward
  but not below 0.8875 as the memorised-document overconfidence washes out (C3000(B) ≈ 30 %).
  Probe accuracy falls from 1.0.
- Most likely classification: **SCAFFOLD**. Next most likely: DATA_SUFFICES.

## Time

Predicted from the corpus-size heartbeats (3570–3579 s per 1000 steps, 3 seeds × 5 threads):
arm A ≈ 3.0 h, arm B ≈ 2.0 h, control ≈ 6 min, measurement ≈ 15 min. A prediction, not a
premise. The deadline is absolute (front matter); at the deadline the parent stops the children
and measures whatever listed checkpoints exist; a missing ckpt 3000 makes the result
inconclusive.

## What this does not establish

Nothing about RSR, eviction, ψ̂ or Kintsch & van Dijk. FIFO only; no retention loss. No scaling
claim. SCAFFOLD would say memorisation is sufficient to bootstrap retrieval here, not that it is
necessary in general (N = 512 seed 2 is a counter-instance at one seed). 3000 steps is one
budget; NEITHER at 3000 does not show the architecture cannot learn the task.
