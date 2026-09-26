---
run_id: scaffold-timing
question: >-
  In the corpus-size curve's N = 64 arm, does memory-dependent accuracy on UNSEEN documents
  appear before, with, or after memorisation of the training documents?
kind: measurement only -- no training. Reads the corpus-size curve's saved checkpoints.
checkpoint_source: "runs/corpus-size-curve/<arm>/seed<s>/ckpt-000{100..1000}.pt, written by the run at ec9332d (ledger committed in 26c590b)"
seeds: [0, 1, 2]
arms: {primary: [64], secondary: [512, 4096]}
checkpoints: [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
thresholds:
  DELTA: "0.03 accuracy, per seed (see 'Why DELTA = 0.03')"
  reproduction_tolerance: "1e-6 absolute, every ledger key re-measured at ckpt 300 and ckpt 1000 (N = 64, all seeds)"
instrument: "the corpus-size curve's measurement path, imported: S0-03 measure(sets=doc_sets(seed, 64)) -- held-out = docs [4096, 4160), train probe = docs [0, 64)"
decision_rule: >-
  A failed or incomplete reproduction control, or a measurement that raised, makes the result
  inconclusive (exit 3). Otherwise, on the N = 64 arm: onset_R is the earliest checkpoint from
  which R holds at that and every later checkpoint through 1000; onset_M likewise for M.
  BEFORE if onset_R < onset_M; WITH if equal; AFTER if onset_R > onset_M; inconclusive if
  either onset does not exist.
---

# PREREG — scaffold timing: does retrieval on unseen documents come before or after memorisation?

**Written 2026-09-25, ~11:00 PDT, by a model (Claude Opus 5.5, interactive session on the
Studio), under Brendan's explicit go ("go on step 2, write the PREREG … do not check back with
me"). Not written by Brendan.** Committed alone, ahead of any code. The text above an amendment
is never edited; amendments are appended, dated, and name the data that prompted them.

## Why this exists

The corpus-size curve (PR #44, ledger `runs/corpus-size-curve/ledger.json` in `26c590b`) showed
at N = 64, ckpt 1000:

- held-out (unseen) documents, `gap_2_to_M`: accuracy **live 0.1232, 0.1662, 0.1201** against
  **slots zeroed 0.0623, 0.0540, 0.0480** (chance 0.0625). The memory carries answer
  information on documents never trained on — weak resolution, badly calibrated
  (`heldout.live…answer_brier_over_16` 1.44 / 1.38 / 1.46 against uniform 0.9375);
- training-probe documents: accuracy **live 1.0** on every seed, and **slots zeroed 0.629,
  0.503, 0.729** — most memorised answers do not need the memory.

At N = 512 and N = 4096 nothing was learned by ckpt 1000. One hypothesis (UNVERIFIED —
inference) is that **memorisation is the scaffold** through which the model learns to use its
memory at all. If so, a training stream with no repeated documents (the planned next run)
removes the scaffold along with the memorisation, and a null there would be ambiguous.

This measurement asks the timing question on checkpoints that already exist. It decides which
arms the next run needs (below).

## ⚠️ Data the author has already seen — this measurement is largely pre-determined

The corpus-size ledger already holds ckpt 300 and ckpt 1000 for the N = 64 arm. From those
keys (`n64.ckpt{300,1000}.{heldout,train}.{live,slots_zeroed}.gap_2_to_M.answer_acc`):

| ckpt | R quantity (held-out live − slots zeroed) | M quantity (train live − held-out live) |
|---|---|---|
| 300 | 0.0142, 0.0111, 0.0085 | 0.0500, 0.1226, 0.1511 |
| 1000 | 0.0609, 0.1122, 0.0721 | 0.8768, 0.8338, 0.8799 |

With DELTA = 0.03, M already holds at 300 on every seed and R fails at 300 on every seed.
**So the classification will be AFTER unless M fails on some seed at some checkpoint in
400–900** (which would move onset_M past 300). The author states this before any new number
exists. The new information is: (1) the onsets at 100-step resolution (is onset_M at 100, 200
or 300; is onset_R at 400 or 900?), (2) whether R rises gradually alongside M or appears only
after M saturates, and (3) the secondary readouts, especially when memorised performance starts
to route through the slots.

## Condition

No training. For each arm ∈ {64, 512, 4096}, seed ∈ {0, 1, 2} and checkpoint
c ∈ {100, 200, …, 1000}, load `ckpt-{c:06d}.pt` from the corpus-size curve's run directory
(read-only; the checkpoints are gitignored and live in
`~/retrieval-successor-retention/.worktrees/corpus-curve/runs/corpus-size-curve/`) and measure
it with the **same function the corpus-size run used**, imported, not retyped:
S0-03's `measure(sets=doc_sets(seed, N))` with the corpus-size curve's `doc_sets`. Held-out is
documents `[4096, 4160)`; the training probe is `[0, 64)`. Device CPU. Every condition the
corpus-size run measured (`live`, `slots_zeroed`, `gate_zeroed`, `gate_zeroed_bos_off`) and every
bucket is recorded.

## The reproduction control (checked first; overrides everything)

Re-measure N = 64 at ckpt 300 and ckpt 1000 on every seed. **Every** statistic key the
corpus-size ledger holds for `n64.ckpt300.*` and `n64.ckpt1000.*` (per seed) must equal the
re-measured value within **1e-6 absolute**. This proves the checkpoints on disk are the ones the
ledger came from and the measurement path is unchanged. If it fails on any key: **exit 3,
inconclusive, report both values, measure nothing further.** Do not tune to make it match.

## Primary readout (N = 64 only)

Per seed, at `gap_2_to_M`:
- **R(c)** — retrieval on unseen documents: `heldout.live.answer_acc − heldout.slots_zeroed.answer_acc ≥ DELTA` on **every** seed.
- **M(c)** — memorisation: `train.live.answer_acc − heldout.live.answer_acc ≥ DELTA` on **every** seed.

`onset_R` = the earliest c such that R holds at c and at every later checkpoint through 1000
(sustained onset — a single noisy crossing does not count). `onset_M` likewise.

| condition | result |
|---|---|
| reproduction control fails or incomplete, or a measurement raised | `inconclusive`, exit 3 |
| `onset_R` or `onset_M` does not exist | `inconclusive` |
| `onset_R < onset_M` | **BEFORE** — retrieval on unseen documents is not scaffolded by memorisation |
| `onset_R = onset_M` | **WITH** |
| `onset_R > onset_M` | **AFTER** |

## Why DELTA = 0.03

Chosen **after** seeing the two checkpoints above, and disclosed as such. It sits about twice
the ckpt-300 R values (0.009–0.014, which look like noise) and about half the smallest ckpt-1000
R value (0.061). Held-out is 64 documents; per-bucket target counts are not in the ledger, so no
formal power figure is offered. The same DELTA is used for M so the two onsets are read on one
scale. It is allowed to be wrong; if so that is recorded, never changed after data.

## What each result commits the next run to (written now)

The next run (step 3, its own PREREG, written after this result) trains on a stream in which no
document repeats.
- **BEFORE** → arm A only (fresh stream from scratch). Memorisation is not needed to learn
  retrieval, so a null in arm A is readable as "cannot learn without repetition".
- **WITH, AFTER, or inconclusive** → arms **A and B** (B: resume from the N = 64 ckpt 1000,
  optimizer and RNG included, then train on the fresh stream). A null in A alone would not
  separate "cannot learn the binding" from "cannot learn it without a memorisation scaffold".

## Secondary readouts (reported, no verdict)

For every arm × seed × checkpoint: accuracy, Brier16, NLL for `live`, `slots_zeroed`,
`gate_zeroed`, `gate_zeroed_bos_off`, held-out and train probe, every bucket. Derived per seed:
- R and M quantities (the table above, at all 10 checkpoints);
- **train memory share**: `train.live − train.slots_zeroed` accuracy — when memorised answers
  start routing through the slots;
- **side-channel share**: `train.gate_zeroed − train.gate_zeroed_bos_off` accuracy;
- held-out Brier16 `slots_zeroed − live` (the corpus-size curve's readout);
- N = 512 and N = 4096: the same quantities. Any checkpoint where R's quantity reaches DELTA on
  any single seed is reported by name (the corpus-size ledger shows N = 512 seed 2 moving at
  ckpt 1000, Brier difference 0.0160).

## Author's expectation (before any new number; allowed to be wrong)

AFTER, with onset_M at 200 or 300 and onset_R at 500–700. R rises steadily once train accuracy
passes ~0.5, rather than jumping. The train memory share rises before R does. At N = 512 and
N = 4096 nothing reaches DELTA except possibly N = 512 seed 2 at 900–1000.

## What this does not establish

Nothing about RSR, eviction, ψ̂, or Kintsch & van Dijk. Timing is correlation: AFTER is
consistent with memorisation scaffolding retrieval, and does not show it. That is what step 3's
arm B is for. No scaling claim.
