# S0-03 — results

**Command:** `uv run python experiments/s0-03-rewardable-corpus/run.py --device cpu`
**Git sha:** `851103efd3d1c676b6ae844a923813bd9b1c1da9` (the PREREG commit; the run
ran on it with a clean source tree). **Hardware:** Mac Studio, CPU, torch 2.14.0,
Python 3.14.6; the three seeds trained in parallel, 5 threads each.
**Ledger:** `runs/s0-03-rewardable-corpus/ledger.json` (config hash
`662c339b…4a78da`). **Seeds:** 0, 1, 2 (actually run); steps 300/300.
Every number below is a ledger key, shown as mean ± sd over the 3 seeds. No row has
`sd_exactly_zero` (0 of 234).

**Verdict (PREREG rule): `inconclusive` — "retrieval not shown (not both at
chance)".** Bar 1 passes, bar 2 passes as registered, and H4 fails. This does **not**
pass the brief's bar. It routes to the decisive run.

Chance = `ln 16 = 2.7726`. The primary set is `heldout` (manifest `measure_heldout`, never
trained on).

## Answer-token NLL, held-out, nats per answer token

| bucket | live | slots zeroed | gate zeroed | gate zeroed + bos off |
|---|---|---|---|---|
| gap = 1 | 2.744 ± 0.152 | 2.804 ± 0.087 | 2.803 ± 0.092 | 2.927 ± 0.099 |
| gap ≥ 2 | 2.913 ± 0.078 | **2.959 ± 0.080** | **2.959 ± 0.078** | 2.937 ± 0.080 |
| 2 ≤ gap ≤ M | **2.881 ± 0.096** | **2.950 ± 0.081** | 2.950 ± 0.080 | 2.927 ± 0.077 |
| gap < M | **2.847 ± 0.106** | 2.914 ± 0.068 | 2.914 ± 0.068 | 2.928 ± 0.081 |
| gap > M | **3.011 ± 0.087** | 2.986 ± 0.075 | 2.987 ± 0.072 | 2.969 ± 0.090 |

Keys: `heldout.<condition>.<bucket>.answer_nll`.

- **Bar 1** (zeroed ≥ `ln16 − 0.10` at gap ≥ 2, every seed): **pass**. Both zeroings
  score *above* chance, not below it (slots zeroed: 2.867 / 3.004 / 3.005).
- **H4** (`slots_zeroed − live` ≥ 0.10 at 2 ≤ gap ≤ M, every seed): **fail**. The
  difference is `0.068 ± 0.092`, with samples `0.023 / 0.174 / 0.008`
  (`heldout.slots_zeroed_minus_live.gap_2_to_M.answer_nll`). It is positive on all
  three seeds, but only seed 1 clears the margin. Live memory is also *above*
  chance, so this is not the "both at chance" branch either.
- **Bar 2** (live gap > M minus gap < M > 0, every seed): **pass**, `0.164 ± 0.144`
  (`0.099 / 0.329 / 0.064`). **Confound:** with the slots zeroed the same contrast is
  still `0.073 ± 0.034` (`heldout.slots_zeroed.gap_gt_M_minus_gap_lt_M.answer_nll`).
  So part of it does not come from the memory. The likely cause is position: a
  gap > M query sits late in its document, and gap < M includes gap 1.
- **At gap > M, live is worse than zeroed:** `slots_zeroed − live = −0.025 ± 0.023`
  (all three seeds negative). Where FIFO has evicted the assert, reading the memory
  hurts slightly.
- **Accuracy**, 2 ≤ gap ≤ M: live 0.082 ± 0.021, zeroed 0.076 ± 0.009. 1/16 = 0.0625.
- **Renormalised over the 16 symbols**, 2 ≤ gap ≤ M: live 2.873 ± 0.094, zeroed
  2.942 ± 0.079. Above chance even among the 16 symbols, so this is not mass
  leaking to non-answer tokens.

## Training documents: memorisation, which is why held-out is primary

| bucket | live | slots zeroed |
|---|---|---|
| gap ≥ 2 | 2.561 ± 0.089 | 2.636 ± 0.104 (`2.699 / 2.692 / 2.516`) |
| 2 ≤ gap ≤ M | 2.551 ± 0.092 | 2.642 ± 0.117 |

On the documents it trained on, the model beats chance **with the memory zeroed**
(seed 2 by 0.26 nats). `(entity, predicate) → object` is being read out of the
weights. Scored on the training documents, bar 1 would fail on seed 2.

## Corpus

- Pairs at `gap > M` (M = 16): held-out `0.192 ± 0.008`, train `0.181 ± 0.012`
  (`*.fraction_of_pairs_gap_gt_M`).
- Answer targets as a share of real targets in the last training batch:
  0.085 / 0.085 / 0.086 (`seed*.train_final.answer_target_fraction`).
- Held-out answer targets per seed: 1203 / 1181 / 1194. The `gap = M` bucket holds
  only 20 / 6 / 9 of them, so its numbers carry no weight.
- Real-token NLL, live: held-out `1.684 ± 0.012`, train `1.560 ± 0.011`.
