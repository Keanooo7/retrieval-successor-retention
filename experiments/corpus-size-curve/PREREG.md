---
run_id: corpus-size-curve
falsifier: >-
  With a training set too large to memorise (N = 4096 documents; S0-03's configuration
  otherwise), the working memory does not help held-out answers at 2 <= gap <= M: at
  neither checkpoint (300, 1000) does every seed show Brier16(slots_zeroed) - Brier16(live)
  >= BRIER_MARGIN on the common held-out set with bar 1 holding.
decision_rule: >-
  A measurement that raised, or a failed or incomplete ckpt-300 reproduction control on the
  N=64 arm, makes the verdict inconclusive (exit 3) whatever else was measured. Otherwise,
  on the N=4096 arm only: falsified if ANY checkpoint in {300, 1000} has bar 1 AND B on all
  three seeds. survived if ckpt 1000 was measured on every seed and B fails at every
  checkpoint with bar 1 holding. inconclusive otherwise (bar 1 fails where B would be read;
  the deadline or a crash cut the N=4096 arm before ckpt 1000 without B shown).
seeds: [0, 1, 2]
arms_n_documents: [64, 4096, 512]
arm_order: "64 -> 4096 -> 512, one arm at a time"
thresholds:
  BRIER_MARGIN: "0.05 -- per seed, Brier16(slots_zeroed) - Brier16(live), heldout, gap_2_to_M"
  MARGIN: "0.10 nats -- S0-03's, imported (bar 1 only)"
  CHANCE: "ln 16 = 2.7726 (bar 1 only)"
  brier16_uniform: "0.9375 = 1 - 1/16, the Brier of the uniform forecast (reference, not a threshold)"
  checkpoints: [300, 1000]
  iters_per_arm: 1000
  reproduction_tolerance: "1e-6 absolute on each seed's heldout live gap_2_to_M answer_nll at ckpt 300, N=64 arm, S0-03's measure() with its own default held-out set"
  deadline: "2026-09-25 07:30 America/Los_Angeles, absolute, parent-enforced"
  threads: "5 per seed (OMP/MKL), 3 seeds as parallel child processes -- S0-03's and the retrieval curve's setting"
common_heldout: "documents [4096, 4160) of each seed's generator, identical for every arm"
memorisation_probe: "documents [0, 64) of each seed's generator (in every arm's training set)"
---

# PREREG — the corpus-size curve: does memory's held-out benefit survive a training set too large to memorise?

**Written 2026-09-24, ~23:30 PDT, by a model (Claude Opus 5.5, the Studio overnight
session), under Brendan's go for "Research + corpus-size run"
(`~/research-corpus/handoffs/2026-09-24-overnight-rsr.md`). Not written by Brendan.** It is
committed alone, ahead of the brief and of any code. **No number from any corpus size
other than 64 exists anywhere.** The author has read the retrieval curve's ledger
(`runs/retrieval-curve/ledger.json`, branch `run/retrieval-curve-2026-09-24`, PR #43), which
is N = 64 only. The text above an amendment is never edited. Amendments are appended, dated,
and name the data that prompted them.

**run_id:** `corpus-size-curve` · **device:** CPU · **seeds:** `0, 1, 2`.

## Question

The retrieval curve could not separate retrieval from memorisation. It trained on 64
documents, about 750 passes by step 3000 (`experiments/retrieval-curve/PREREG.md:105-106`).
At ckpt3000 the training documents' live answer NLL at `2 ≤ gap ≤ M` is ≈ 0 (5.46e-5,
3.88e-5, 6.73e-5), while held-out it is 8.1453, 7.8372, 8.4254 against chance ln 16 = 2.7726.
Held-out accuracy at ckpt1000 is memory-dependent: live 0.1485, 0.1646, 0.1431 against
slots-zeroed 0.0636, 0.0633, 0.0715. All of these are retrieval-curve ledger keys
`ckpt3000.{train,heldout}.live.gap_2_to_M.answer_nll` and
`ckpt1000.heldout.{live,slots_zeroed}.gap_2_to_M.answer_acc`.

This run changes **one factor: the number of training documents, N.** The question is whether
the memory's held-out benefit is still there when N is large enough that the model cannot
memorise the training set.

## Condition

S0-03's `CONFIG` (`experiments/s0-03-rewardable-corpus/run.py`), imported, not retyped
(`d=128, steps_per_stream=48, batch=16, max_tokens=64, memory_slots=16, lr=1e-3,
policy="fifo", masked_loss=True, srep_norm_reg_weight=0.0`), and S0-03's rewardable
synthetic corpus. **Only `n_documents` of the training corpus changes.** `train()` gains an
optional `n_documents` argument. Its default (`None`) is today's code path, and the N = 64
arm calls `train()` **without** it.

- **Arms:** N ∈ {64, 4096, 512}, run in that order, one arm at a time, each to **1000
  iterations**, with checkpoints measured at **300 and 1000**.
- **Threads:** 5 per seed, three seeds as parallel child processes. This is S0-03's and the
  retrieval curve's setting.
- **The vocabulary is the same in every arm.** It is the sorted word set, and it is identical
  for 64 and 4160 documents on seeds 0-2 (checked before this PREREG; a test pins it). So `V`
  and every token id are shared across arms. Per iteration, only which documents the batch
  indices point at differs.

## Document sets

`generate` is prefix-stable. Each document draws from its own RNG, seeded
`seed · 1_000_003 + doc_id` (`src/rsr/data/synthetic.py:235-238`). So document *i* is the
same for every `n_documents > i`, and a test pins it.

- **Common held-out (PRIMARY):** documents `[4096, 4160)` of each seed's generator. They are
  disjoint from every arm's training set `[0, N)` with N ≤ 4096, identical across arms, and
  64 documents (S0-03's held-out size). A test proves the disjointness.
- **Memorisation probe:** documents `[0, 64)`. Every arm trains on them, and N = 64 trains on
  nothing else.
- **S0-03's own held-out** (`[64, 128)`) is measured **only** in the N = 64 arm, **only** for
  the reproduction control. For N ≥ 128 those documents are training data.

## Instrument

- S0-03's `measure()` and `decide()`, imported. `measure()` gains an optional override of its
  document sets, whose default leaves S0-03 unchanged. `answer_readout` gains one more
  per-target number, the **Brier score over the 16 answer symbols**. That is new measurement
  code with its own test.
- **Brier16:** for an answer target with true symbol `y`, take the model's logits at that
  position restricted to the 16 answer-symbol ids, and `p = softmax(logits[sym_ids])`.
  Then `Brier16 = Σ_k (p_k − 1[k = y])²`. It ranges over [0, 2]; the uniform forecast
  scores 0.9375. This is the same renormalisation S0-03's `answer_nll_over_16` already uses.
- **Not measured:** the decisive run's shuffle `ratio`. Its `measure()` builds its own 64-doc
  sets, which are training data for N ≥ 128.

## Why Brier is primary, not NLL or accuracy

- **NLL** is unbounded. At N = 64 held-out answer NLL rose to ~8 nats, about 3× chance, while
  accuracy stayed above slots-zeroed. A difference of NLLs there mostly measures
  overconfidence, not retrieval. Research corpus: the logarithmic score is "unbounded below …
  a single confident mistake can dominate everything else"
  (`~/research-corpus/sources/decision-models-series/03_Calibration_and_Uncertainty.md:148`).
- **Accuracy** "looks only at whether the top choice was right" and "is not proper" (`:132-134`).
- **Brier** is "strictly proper and **bounded**" (`:144`). The corpus's guidance is "Brier for
  unordered categories, when you want bounded and stable" (`:162`), and the 16 answer symbols
  are unordered.
- **BRIER_MARGIN = 0.05** was chosen before any Brier number existed anywhere in this project.
  It is a round number, about 5 % of the uniform forecast's 0.9375. It is **not derived** and
  is allowed to be wrong. It is recorded in BRIEF-ERRORS if so, and never changed after data.

## Primary readout and decision rule

**Only the N = 4096 arm carries the verdict.** At each checkpoint `c ∈ {300, 1000}`, on the
common held-out set:
- `bar1(c)` is S0-03's `decide()` bar 1, unchanged. With memory zeroed (slots or gate), answer
  NLL at `gap ≥ 2` is ≥ CHANCE − MARGIN on every seed: the answer doesn't leak without memory.
- `B(c)` holds when every seed has `Brier16(slots_zeroed) − Brier16(live) ≥ BRIER_MARGIN` at
  `gap_2_to_M`.

| condition | verdict |
|---|---|
| a measurement raised | `inconclusive`: exit 3 |
| the N = 64 arm's ckpt-300 reproduction control fails on any seed, or never ran on every seed. **Checked first; overrides every row below.** | `inconclusive`: this is not S0-03's configuration; exit 3; stop every arm |
| N = 4096: **any** `c` with `bar1(c)` **and** `B(c)` | **`falsified`**: memory helps held-out answers at N = 4096 |
| N = 4096: ckpt 1000 measured on every seed, `bar1` holds and `B` fails at every `c` | **`survived`**: no memory benefit on held-out answers at N = 4096 by ckpt 1000 |
| otherwise (bar 1 fails where B would be read; the N = 4096 arm stopped before ckpt 1000) | `inconclusive`, with the reason |

⚠️ **Two looks.** "Any checkpoint" is looser than one pre-chosen checkpoint, so the
all-three-seeds requirement is kept. The report names which checkpoint carried a
`falsified`.

⚠️ **Polarity, as in the retrieval curve.** `falsified` means **memory helps** (the falsifier
is false). S0-03's own `decide()` label is recorded under `s003_outcome` and never under the
verdict key.

## The reproduction control

The N = 64 arm's ckpt-300 checkpoint, measured by S0-03's `measure()` with **no override**,
must equal `runs/s0-03-rewardable-corpus/ledger.json` `heldout.live.gap_2_to_M.answer_nll`
**per seed within 1e-6 absolute**. This is the retrieval curve's control. It proves the new
`n_documents` argument and the measurement changes left the default path unchanged. If it
fails: **stop every arm, exit 3, report both sets of samples.** Don't tune to make it match,
and don't rerun a changed design without a new PREREG commit.

## Secondary readouts (reported, no verdict)

For every arm, checkpoint and seed, on the common held-out set and the memorisation probe:
live and slots-zeroed Brier16, answer NLL, accuracy and NLL-over-16 in every S0-03 bucket
(`gap_eq_1` separately, not read as retrieval). Also: `slots_zeroed − live` for Brier16, NLL
and accuracy at `gap_2_to_M`; the train-minus-held-out live gap in Brier16, NLL and accuracy
(the memorisation readout); S0-03's `decide()` fields; and the memory gates. The N = 64 and
N = 512 arms get the same `bar1`/`B` values, **reported, no verdict read from them**.

## Deadline and scheduling

The arms run in the order 64 → 4096 → 512. The parent stops at the absolute deadline,
**2026-09-25 07:30 America/Los_Angeles**. At the deadline it terminates the children and
measures whatever pre-registered checkpoints are on disk. An arm that never started is
recorded `not run`. The predicted wall time is about 1 h per arm (the retrieval curve took
10934.9 s for 3000 iterations), which is a prediction and not a premise.

## Author's expectation (written before any data; allowed to be wrong)

- At **ckpt 300**, `B` fails in every arm. S0-03's H4 failed at 300 on N = 64, and 300
  iterations x batch 16 is ~75 passes over N = 64 but only ~1.2 over N = 4096.
- At **ckpt 1000**, the N = 4096 arm shows `B` on every seed a little more often than not
  (≈ 55 %). The N = 64 arm shows the retrieval curve's pattern: held-out live NLL far above
  chance and a Brier that is poor in both conditions. The N = 4096 arm's held-out live NLL
  stays near or below chance, because it cannot memorise, so it cannot become overconfident
  on specific documents.
- The main way this could be wrong: 1000 iterations at N = 4096 may simply be too few for the
  memory path to learn anything (`survived` for lack of training, not lack of memory). The
  report must say so if `survived` comes with near-chance live accuracy.

## What this does not establish

Nothing about RSR, eviction policy, ψ̂, or Kintsch & van Dijk. The run is FIFO only, and it
touches no retention loss. It makes no scaling claim: N is a data-size factor at one width,
and nothing here is a slope. The question is only whether TG's memory helps held-out answers
once memorisation is taken away, which is a precondition for every eviction-policy arm.
