---
run_id: retrieval-curve
falsifier: >-
  Training S0-03's configuration (arm B: masked loss, hinge off) for longer does not make
  the working memory help answers at 2 <= gap <= M: at no checkpoint up to 3000
  iterations does every seed show slots_zeroed - live answer NLL >= MARGIN on held-out
  documents with bar 1 holding.
decision_rule: >-
  If the ckpt-300 reproduction control fails, the verdict is inconclusive (exit 3) whatever the
  other checkpoints show. Otherwise: S0-03's decide() (experiments/s0-03-rewardable-corpus/run.py), unchanged, applied at each
  checkpoint in {300, 1000, 3000}. falsified if ANY checkpoint has bar 1 AND H4 on all three
  seeds. survived if 3000 was reached and H4 fails at every checkpoint with bar 1 holding.
  inconclusive otherwise (bar 1 fails where H4 would be read, the deadline cut the run before
  3000 without retrieval shown, or the ckpt-300 reproduction control fails).
seeds: [0, 1, 2]
thresholds:
  MARGIN: "0.10 nats -- S0-03's, imported, not re-chosen (s0-03 run.py MARGIN)"
  CHANCE: "ln 16 = 2.7726 (16 answer symbols)"
  checkpoints: [300, 1000, 3000]
  reproduction_tolerance: "1e-6 absolute on each seed's heldout live gap_2_to_M answer_nll at ckpt 300"
  wall_deadline_hours: 8
  threads: "5 per seed (OMP/MKL), 3 seeds as parallel child processes -- S0-03's setting (runs/s0-03-rewardable-corpus/manifest.json threads_per_seed); CPU float reductions are bit-reproducible per thread count, not across thread counts"
---

# PREREG — the retrieval curve: does longer training make memory help answers?

**Written 2026-09-22 by the MacBook owner-proxy window, before any run it governs.** The
author has read `runs/decisive-shuffle/` and `runs/s0-03-rewardable-corpus/` (both at
300 iterations) and **no number from any longer run exists anywhere**. This file is
committed alone, ahead of the brief and of any code. The text above an amendment is never
edited; amendments are appended, dated, naming the data that prompted them.

**run_id:** `retrieval-curve` · **device:** CPU (so the ckpt-300 control can be exact) ·
**seeds:** `0, 1, 2`.

## Question

The decisive run showed the memory is **live** under the masked objective (`armB.ratio`
9.33 ± 4.68) but does **not help answers**: held-out answer NLL at `2 ≤ gap ≤ M` sits at
chance, and S0-03's H4 failed (`heldout.slots_zeroed_minus_live.gap_2_to_M.answer_nll`
0.068 ± 0.092, against `MARGIN` 0.10). Both measurements were at **300 iterations**, and the
overnight report lists "behaviour past 300 iterations" as not established
(`docs/lab-notes/overnight-2026-09-21.md:97`).

This run changes **one factor: training length.** Everything else is S0-03's configuration.

The gap-1 bucket (below chance in both runs) is **not** evidence of retrieval: at `gap = 1`,
`bos_replacement_mode="copy"` carries the previous sentence outside the memory
(`experiments/decisive-shuffle/PREREG.md`, Amendment 1, finding 2). It is reported and
never read as retrieval.

## Condition

S0-03's `CONFIG` (`experiments/s0-03-rewardable-corpus/run.py`), imported, not retyped:
shuffle-control's `d=128, steps_per_stream=48, batch=16, max_tokens=64, memory_slots=16,
lr=1e-3, policy="fifo"`, with `masked_loss=True`, `srep_norm_reg_weight=0.0` (S0-03's
setting; also proposed ruling `R-2026-09-22-hinge`, `docs/lab-notes/proposed-rulings-2026-09-22.md`),
and S0-03's rewardable synthetic corpus. **Only `iters` changes.** **Threads are pinned to S0-03's:
5 per seed, three seeds as parallel child processes.** A different thread count changes CPU float
reductions, so it would fail the reproduction control for a reason unrelated to the question (the
decisive run's arm B shares S0-03's `config_hash` but ran at 1 thread, and its final losses differ
from S0-03's). S0-03's provenance is `dirty: true` at `851103e`, recorded here so a reproduction
failure is not blamed on it silently.
Checkpoints are measured at **300, 1000 and 3000** iterations.

## Instrument — reused, no new implementation

- **Primary:** S0-03's `measure()` and `decide()`, imported. `MARGIN`, `CHANCE`, the buckets
  and the held-out population (docs 64..127 of each seed's generator stream) are S0-03's.
- **Secondary:** the decisive run's `measure()` (shuffle `ratio` with an untrained decoy at the
  same seed, `random_ratio`, cross-row cosine), imported, at each checkpoint.

## Primary readout and decision rule

At each checkpoint `c`, apply S0-03's `decide()` unchanged to the three seeds → `bar1(c)`,
`H4(c)` (every seed `slots_zeroed − live ≥ 0.10` at `gap_2_to_M`, held-out), and S0-03's
own outcome label, recorded under the separate key `s003_outcome`. ⚠️ S0-03's labels are about
**its** falsifier: its `survived` (H4 and bar 2) corresponds to **this** PREREG's `falsified`. The two
must never share a key.

| condition | verdict on the falsifier |
|---|---|
| the ckpt-300 reproduction control fails (below) — **checked first; overrides every row below** | `inconclusive` — the run is not S0-03's configuration; exit 3 |
| **any** checkpoint with `bar1(c)` **and** `H4(c)` | **`falsified`** — longer training makes memory help answers |
| 3000 reached, `bar1(c)` holds and `H4(c)` fails at every checkpoint | **`survived`** — training length alone does not produce retrieval by 3000 |
| otherwise (bar 1 fails where H4 would be read; deadline before 3000 with no H4) | `inconclusive`, with the reason |

⚠️ **Three looks.** "Any checkpoint" is a looser rule than one pre-chosen checkpoint. The
all-three-seeds requirement at `MARGIN` is kept strict for that reason, and the report says
which checkpoint carried a `falsified` verdict.

## The reproduction control (the part only this run can get wrong)

The 300-iteration checkpoint must reproduce `runs/s0-03-rewardable-corpus/ledger.json`
`heldout.live.gap_2_to_M.answer_nll` **per seed, within 1e-6 absolute**. If it does not, the
curve is not a continuation of S0-03, and nothing past it is interpretable. **Stop, exit 3,
report both sets of samples.** Do not tune to make it match. (S0-03's ledger records Python
3.14.6; this run is 3.12 after `R-2026-09-22-python-312`. E-feas and the old-sha canary held
bit-exact across that change; if this control is the first thing that does not, that is
itself the finding.)

## Secondary readouts (reported, no verdict)

Per checkpoint and seed: live and slots-zeroed answer NLL and accuracy in every S0-03 bucket
(`gap_eq_1` separately), the **train-document** answer NLL (memorisation: the training set is
64 documents, ~75 passes at 300 iterations, ~750 at 3000), and the decisive run's `ratio`,
`random_ratio` and cross-row cosine.

## Author's expectation (written before any data; allowed to be wrong)

`survived` is at least as likely as `falsified`. The leading rival cause of chance-level
retrieval is **memorisation of 64 training documents** rather than too little training, and
longer training strengthens memorisation. If `survived`, the next candidates, each needing
its own PREREG, are more training documents, and `bos_replacement_mode` off (the gap-1
shortcut may absorb the gradient that would teach the memory path).

## What this does not establish

Anything about RSR, eviction policy or Kintsch & van Dijk. It asks only whether TG's memory,
under FIFO, can be made to carry answers across 2..M sentences. **That is the precondition for
E0d (proposed ruling `R-2026-09-22-sprint2-start`, `docs/lab-notes/proposed-rulings-2026-09-22.md`) and for every cognitive experiment after it.**
