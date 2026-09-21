# PREREG — S0-03: is the rewardable synthetic corpus testing retrieval?

**Written 2026-09-21 by the S0-03 researcher, before any full run.** The only
numbers seen before this commit are (a) corpus statistics, which are model-free
(`gap > M` fraction 0.195 / 0.174 / 0.174 at seeds 0 / 1 / 2; answer targets 8.5%
of real targets at seed 0), and (b) a 2-iteration plumbing smoke whose answer NLL
was ≈ 5.1–5.2 nats, i.e. ≈ `ln V` for an untrained model, deleted afterwards. No
threshold below was chosen with a trained-model number in view.

Brief: `docs/lab-notes/dispatch-S0-03-rewardable-corpus.md`. Script:
`experiments/s0-03-rewardable-corpus/run.py`; `decide()` there is this rule.

## Condition

`experiments/shuffle-control/run.py`'s `CONFIG` loaded from that file (d=128,
steps_per_stream=48, batch=16, max_tokens=64, memory_slots=16, iters=300, lr=1e-3,
policy=fifo), with `masked_loss=True` and `srep_norm_reg_weight=0.0` (the brief),
on the S0-03 corpus (`SyntheticConfig` defaults, `answer_in_stream=True`). Seeds
0, 1, 2. CPU. The three seeds train in parallel processes, 5 threads each.

## Readout

Answer-token NLL (full softmax, nats per answer token), trained checkpoint,
`eval()`, FIFO, one pass per stream. Also reported: argmax accuracy and NLL
renormalised over the 16 answer symbols. Chance = `ln 16 = 2.7726`.

- **Sets:** `heldout` = documents 64..127 of the seed's generator, never trained
  on — **primary**, because 300 iterations × 16 streams is 75 passes over the 64
  training documents and `(entity, predicate) → object` can be memorised from them
  without any memory. `train` = documents 0..63, reported beside it.
- **Conditions:** `live`; `slots_zeroed` (every slot's content zeros, validity as
  in the honest loop — the brief's literal "all M slots zeroed"); `gate_zeroed`
  (every `memory_gate` = 0); `gate_zeroed_bos_off` (diagnostic).
- **Buckets:** `gap_eq_1`, `gap_ge_2`, `gap_2_to_M`, `gap_lt_M`, `gap_eq_M`,
  `gap_le_M`, `gap_gt_M`, `all`. **`gap = 1` is set aside in the bars** because
  `bos_replacement_mode="copy"` puts the previous sentence's gestalt at token 0,
  outside the memory, so zeroing the memory does not remove the assert at gap 1.
- Under FIFO with M = 16 and a write on every sentence, the assert is in memory at
  its query iff `gap <= M`.

## Rule

`MARGIN = 0.10` nats — the one tolerance the brief does not give; it is used
identically in all three clauses. On `heldout`, every seed:

1. **Bar 1 (zeroed at chance):** `slots_zeroed` and `gate_zeroed` answer NLL at
   `gap_ge_2` are both `>= ln16 − 0.10`. If not, the answer is being read from
   somewhere other than the memory → verdict **falsified** (the corpus does not
   test retrieval).
2. **Bar 2 (eviction bites):** `live` answer NLL at `gap_gt_M` minus at `gap_lt_M`
   is `> 0`.
3. **H4 (retrieval shown):** `slots_zeroed − live` answer NLL at `gap_2_to_M` is
   `>= 0.10`.

Verdict: bar 1 fails → falsified. Bar 1 + H4 + bar 2 → survived. Bar 1 + H4
without bar 2 → inconclusive ("eviction does not bite"). Bar 1 without H4 →
inconclusive: "corpus built, retrieval not shown" if live and slots-zeroed are both
within 0.10 of chance at `gap_2_to_M`, otherwise "retrieval not shown (not both at
chance)". **Neither inconclusive branch is a pass of the brief's bar**; per the
brief's H4 addendum they route to the decisive run.

## Expected (written before the run)

Bar 1 passes. Gap-1 answers fall below chance live via the bos-copy path. At
`2 <= gap <= M`, live is at most modestly below zeroed after 300 iterations;
`gap > M` ≈ chance. Most likely verdict: corpus built, retrieval not shown beyond
gap 1.

## Falsifier addressed

"The S0-03 corpus's answer tokens cannot be predicted without the memory" (bar 1),
and "FIFO eviction raises answer-token loss at `gap > M`" (bar 2).
