# Decisive run — is the TG working memory live on the rewardable corpus?

**Pre-registration:** [`PREREG.md`](PREREG.md) (+ Amendment 1), committed ahead of `run.py`.
**Ledger:** `runs/decisive-shuffle/ledger.json`, the source of every number below (rendered from it, not typed).
**Manifest:** `runs/decisive-shuffle/manifest.json`, frozen before the first seed; ledger `config_hash` `2a299e4afc2dc074ff375d4a38ee8684ae9a7cc3b5bd638e166185bbda487c8d`.

**Command:** `uv run python experiments/decisive-shuffle/run.py` (exit `0`). Per-(arm, seed) training commands are in the ledger's `commands` (all exit `0`, all reproducible at the sha).
**Git SHA:** `90438f3d2a9c1998cca37069a7b4f7ded649a4b9` (ledger `provenance.git_sha`; `dirty: true` is `git status --porcelain` being non-empty; the one untracked path at launch was `uv.lock`, and `Ledger.write()` refuses a dirty `src/`, `scripts/` or `experiments/`, which it did not). The code under `src/` and `experiments/` is unchanged between this sha and the PR head; later commits touch only tests, the battery, this file and the ledger.
**Hardware:** Mac Studio M4 Max, **CPU only**. torch `2.14.0`, Python `3.14.6`. Nine trainings ran as concurrent subprocesses, `torch.set_num_threads(1)` in each (ledger `training_threads_per_process`); the measuring process also ran at one thread.
**Seeds actually run:** 0, 1, 2 in every arm; `steps_done` 300 of 300; status `ok`.

## Verdict

| arm | objective | verdict (PREREG rule, train batch) |
|---|---|---|
| A | unmasked, hinge off (#17's config, new corpus) | **inert** |
| B | masked, hinge off | **live** |
| C | masked, hinge at the `TGConfig` default | **live** |

**Which §10.3 outcome occurred: live on B and C.** The memory path is not broken. With the masked objective, the trained model's tokens move under a memory swap at several times the live decoy's rate on every seed. Arm A shows that the corpus change alone was not enough: under the unmasked objective, the shuffle reads inert.

**Read this with Secondary 2 before concluding anything about arm A or #17.** Arm A's memories are almost exactly collinear across rows (cross-row cosine ≈ 0.9995). Replacing them with matched-norm random vectors moves tokens at `>=0.1` of the decoy's rate on every seed. By the PREREG's own descriptive reading, *"random ≥ 0.1 while the shuffle ≤ 0.01 → memory is read but carries no row-specific content."* Arm A's null therefore says the memory contents carry nothing row-specific. It does not say the readout ignores memory.

**What "live" does not establish.** The answer-token NLL of B and C on held-out docs, with the model's own memory intact, is about 2.9 nats per answer token in every gap bucket. That is at or above ln 16, chance over the answer symbols. The memory is used, and the answer tokens are the most memory-sensitive targets, with shuffle ratios several times the whole-token ratio. Neither fact shows that the model retrieves the right answer. This run does not test retrieval accuracy.

## Primary: `ratio` per arm (train batch, #17's measurement batch)

| arm | ratio seed 0 / 1 / 2 | mean ± sd | sd_exactly_zero | verdict |
|---|---|---|---|---|
| A | 0.0004656 / 0.003647 / 0.004921 | 0.003011 ± 0.002295 | False | **inert** |
| B | 6.742 / 14.73 / 6.502 | 9.325 ± 4.683 | False | **live** |
| C | 2.989 / 14.56 / 5.672 | 7.741 ± 6.058 | False | **live** |

**Overall:** `falsified` -- per arm {'A': 'inert', 'B': 'live', 'C': 'live'}; live on at least one arm: the cause was the objective or the corpus; arm A (corpus alone) is 'inert'

## Controls, every seed of every arm

| arm | seed | batch | memory-disabled A (==0) | own-memory A (==0) | decoy A (>0) | random-self A (==0) |
|---|---|---|---|---|---|---|
| A | 0 | train | 0.0 | 0.0 | 0.01021 | 0.0 |
| A | 0 | heldout | 0.0 | 0.0 | 0.009318 | 0.0 |
| A | 1 | train | 0.0 | 0.0 | 0.005432 | 0.0 |
| A | 1 | heldout | 0.0 | 0.0 | 0.0063 | 0.0 |
| A | 2 | train | 0.0 | 0.0 | 0.006189 | 0.0 |
| A | 2 | heldout | 0.0 | 0.0 | 0.006456 | 0.0 |
| B | 0 | train | 0.0 | 0.0 | 0.01021 | 0.0 |
| B | 0 | heldout | 0.0 | 0.0 | 0.009318 | 0.0 |
| B | 1 | train | 0.0 | 0.0 | 0.005432 | 0.0 |
| B | 1 | heldout | 0.0 | 0.0 | 0.0063 | 0.0 |
| B | 2 | train | 0.0 | 0.0 | 0.006189 | 0.0 |
| B | 2 | heldout | 0.0 | 0.0 | 0.006456 | 0.0 |
| C | 0 | train | 0.0 | 0.0 | 0.01021 | 0.0 |
| C | 0 | heldout | 0.0 | 0.0 | 0.009318 | 0.0 |
| C | 1 | train | 0.0 | 0.0 | 0.005432 | 0.0 |
| C | 1 | heldout | 0.0 | 0.0 | 0.0063 | 0.0 |
| C | 2 | train | 0.0 | 0.0 | 0.006189 | 0.0 |
| C | 2 | heldout | 0.0 | 0.0 | 0.006456 | 0.0 |

## Held-out batch (the first 8 documents of S0-03's held-out set), descriptive, no verdict

| arm | ratio seed 0 / 1 / 2 | mean ± sd | sd_exactly_zero |
|---|---|---|---|
| A | 0.0006886 / 0.003368 / 0.004055 | 0.002704 ± 0.001779 | False |
| B | 7.239 / 13.3 / 6.096 | 8.877 ± 3.869 | False |
| C | 3.103 / 15.65 / 6.579 | 8.444 ± 6.478 | False |

## Secondary 1: answer tokens only, held-out, by gap (descriptive)

| arm | bucket | n targets seed 0 / 1 / 2 | ratio seed 0 / 1 / 2 | mean ± sd | trained answer NLL, own memory (mean ± sd) |
|---|---|---|---|---|---|
| A | `answer_all` | 150 / 145 / 147 | 0.000402 / 0.003292 / 0.003763 | 0.002486 ± 0.00182 | 2.803 ± 0.06157 |
| A | `gap_1` | 34 / 26 / 18 | 0.0003816 / 0.001222 / 0.0033 | 0.001635 ± 0.001502 | 2.746 ± 0.01624 |
| A | `gap_2_to_M` | 85 / 87 / 98 | 0.0003569 / 0.00415 / 0.004086 | 0.002864 ± 0.002172 | 2.82 ± 0.0815 |
| A | `gap_gt_M` | 31 / 32 / 31 | 0.000642 / 0.004576 / 0.003074 | 0.002764 ± 0.001985 | 2.802 ± 0.05617 |
| B | `answer_all` | 150 / 145 / 147 | 18.98 / 46.94 / 12.05 | 25.99 ± 18.47 | 2.904 ± 0.05702 |
| B | `gap_1` | 34 / 26 / 18 | 16.39 / 25.75 / 12.9 | 18.35 ± 6.648 | 2.725 ± 0.109 |
| B | `gap_2_to_M` | 85 / 87 / 98 | 17.93 / 57.27 / 12.71 | 29.31 ± 24.36 | 2.945 ± 0.07261 |
| B | `gap_gt_M` | 31 / 32 / 31 | 28.49 / 54.43 / 9.592 | 30.84 ± 22.51 | 2.93 ± 0.1347 |
| C | `answer_all` | 150 / 145 / 147 | 5.315 / 68.35 / 17.85 | 30.5 ± 33.37 | 2.852 ± 0.08828 |
| C | `gap_1` | 34 / 26 / 18 | 4.674 / 37.35 / 18.83 | 20.28 ± 16.38 | 2.609 ± 0.206 |
| C | `gap_2_to_M` | 85 / 87 / 98 | 5.274 / 80.48 / 18.46 | 34.74 ± 40.16 | 2.883 ± 0.1445 |
| C | `gap_gt_M` | 31 / 32 / 31 | 6.7 / 90.13 / 15.49 | 37.44 ± 45.84 | 2.919 ± 0.1788 |

## Secondary 2: rival-hypothesis discriminators (descriptive)

| arm | batch | random_ratio seed 0 / 1 / 2 | mean ± sd | band per seed | cosine trained (mean ± sd) | cosine decoy (mean ± sd) | decoy random/shuffle (mean ± sd) |
|---|---|---|---|---|---|---|---|
| A | train | 1.075 / 3.28 / 1.938 | 2.098 ± 1.111 | >=0.1 / >=0.1 / >=0.1 | 0.9995 ± 0.0005338 | 0.565 ± 0.2042 | 3.129 ± 1.208 |
| A | heldout | 1.184 / 2.831 / 1.8 | 1.938 ± 0.8325 | >=0.1 / >=0.1 / >=0.1 | 0.9995 ± 0.000536 | 0.5572 ± 0.2195 | 2.952 ± 0.9291 |
| B | train | 10.37 / 20.02 / 11.85 | 14.08 ± 5.199 | >=0.1 / >=0.1 / >=0.1 | 0.4833 ± 0.06134 | 0.565 ± 0.2042 | 3.129 ± 1.208 |
| B | heldout | 11.98 / 14.85 / 11.01 | 12.61 ± 1.997 | >=0.1 / >=0.1 / >=0.1 | 0.4506 ± 0.03407 | 0.5572 ± 0.2195 | 2.952 ± 0.9291 |
| C | train | 5.89 / 21.96 / 11.84 | 13.23 ± 8.124 | >=0.1 / >=0.1 / >=0.1 | 0.4357 ± 0.07591 | 0.565 ± 0.2042 | 3.129 ± 1.208 |
| C | heldout | 6.838 / 16.54 / 10.97 | 11.45 ± 4.868 | >=0.1 / >=0.1 / >=0.1 | 0.3999 ± 0.06075 | 0.5572 ± 0.2195 | 2.952 ± 0.9291 |

## Other readouts (train batch)

| arm | honest real-token NLL (mean ± sd) | signed mean Δ (mean ± sd) | tokens moved (per seed) | A trained (mean ± sd) | A decoy (mean ± sd) |
|---|---|---|---|---|---|
| A | 1.681 ± 0.02436 | 0.000000086 ± 0.0000009102 | 1240 / 1345 / 1294 | 0.00001834 ± 0.00001291 | 0.007278 ± 0.002569 |
| B | 1.573 ± 0.03259 | 0.01194 ± 0.00668 | 1700 / 1699 / 1678 | 0.06304 ± 0.02051 | 0.007278 ± 0.002569 |
| C | 1.567 ± 0.02932 | 0.009052 ± 0.008463 | 1699 / 1702 / 1680 | 0.04824 ± 0.02682 | 0.007278 ± 0.002569 |

Memory gates (train batch, seed 0, per cross-attention block):

- A: 0.9766, 0.9647, 0.969, 0.9754, 0.9785, 0.987
- B: 0.9373, 0.9617, 0.9951, 1.01, 1.021, 1.018
- C: 0.9081, 0.9668, 0.9898, 1.005, 1.022, 1.022

## Notes, caveats, and what was not done

- **Primary population.** The PREREG decision rule is applied to #17's measurement batch, the first 8 **training** documents. The held-out rows (the first 8 documents of S0-03's held-out set) are descriptive only, per Amendment 1, and they agree in band on every arm.
- **Decoy.** The decoy is the untrained model at the seed, so it is the same model for all three arms. Identical decoy columns across arms are expected, not a defect.
- **Ratios above 1.** In B and C, the trained model's `A` exceeds the random-init decoy's. A near-uniform untrained model moves little when its memory changes; a trained model with confident predictions moves more. The PREREG bands are one-sided (`≥ 0.1` is live), so this does not change a verdict. It does mean the ratio cannot be read as a fraction of a maximum.
- **gap = 1** is reported separately because `bos_replacement_mode="copy"` carries the previous sentence outside the memory. The shuffle leaves `bos_ctx` honest, so a gap-1 answer can still be read through that path.
- **Controls.** Memory-disabled and own-memory read exactly `0.0`, the decoy read `> 0`, and random-replacement-with-self read exactly `0.0` on every seed of every arm, on both batches. No `sd_exactly_zero` flag is set on any statistic row.
- **Signed mean Δ is not the bar** (#17 Amendment 1). It is reported in the table above and carries no verdict.
- Nothing here tests whether RSR discovers Kintsch & van Dijk's leading-edge strategy. It establishes that, under the masked objective on this corpus, TG has a memory that a retention policy could act on.
