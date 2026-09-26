---
run_id: retention-readability
question: >-
  Can fresh-stream arm B -- a TG model trained ONLY under FIFO eviction -- read a fact that a
  different eviction rule kept in memory at a rank FIFO never put it at? Two parts: (a) on
  facts FIFO also keeps (gap 2..M), does moving them to other ranks cost accuracy; (b) on facts
  only the other rule keeps (gap > M), are they read like FIFO reads its oldest resident fact?
seeds: [0, 1, 2]
checkpoints:
  B: [2500, 3000]            # fresh-stream arm B, FROZEN, read-only. 3000 is primary.
  A: [3000]                  # fresh-stream arm A: inert memory, the negative control
source: ".worktrees/fresh-stream/runs/fresh-stream/{A,B}/seed<s>/ckpt-00XX00.pt (read-only)"
arms:
  fifo: "rsr.baselines.fifo.FIFOPolicy"
  oracle: "rsr.baselines.oracle.OraclePolicy(discounted_demand(doc, 0.97)) -- E-feas's GAMMA, one fresh policy per document"
  factfiller: "evict uniformly at random among live slots whose sentence is not an assert; if every live slot is an assert, uniformly at random among all live slots. No age, no future. RNG random.Random(f'ff:{seed}:{doc_id}'), fresh per document"
documents:
  P: "[4096, 4160) via experiments/corpus-size-curve/run.py::doc_sets(seed, 64)['heldout'] (fresh-stream's held-out set)"
  E: "[64, 1088) of the same seed's generator (1024 documents, unseen by arms A and B)"
  U: "E union P, 1088 documents -- the PRIMARY set for every readout"
vocabulary: "doc_sets(seed, 64)'s map, built from documents [0, 64); closure over E asserted"
thresholds:
  RP_MAX: "0.03 accuracy (scaffold-timing / fresh-stream DELTA, unchanged)"
  READ_MARGIN: "0.10 accuracy"
  A_NULL: "0.03 accuracy: arm A's oracle - FIFO must have its 95% CI inside (-0.03, +0.03)"
  bootstrap: "2000 per-document resamples of U, paired across arms, percentile 95% CI, torch.Generator seeded 20260926 + seed"
  reproduction_tolerance: "1e-6 absolute (fresh-stream's REPRO_TOL)"
decision_rule: >-
  Any control failing, any measurement raising, or any listed checkpoint missing -> inconclusive
  (exit 3). Otherwise, per arm-B checkpoint: (a) NO_PENALTY / PENALTY / MIXED_PENALTY and
  (b) READABLE / UNREADABLE / UNDISTINGUISHED / MIXED_READ, combined into
  READABLE_AT_SHIFTED_RANK / RANK_BOUND / MIXED. ckpt 3000 is the headline; ckpt 2500 is
  classified by the same rule and reported beside it.
---

# PREREG — retention readability: can a FIFO-trained model read an old fact kept at a shifted rank?

**Written 2026-09-26 by a model (Claude Opus 5.5, an `rsr-researcher` subagent of the
interactive day-loop manager), under Brendan's Stage-1 gate opened today in the interactive
session ("You decide I am ok with what is required…"; thresholds: "Loop sets, pre-data"). Not
written by Brendan.** Committed alone, ahead of any code for it. Amendments are appended, dated,
and name the data that prompted them; the text above is never edited.

**Scope.** Read-only measurement on frozen fresh-stream checkpoints. **Trains nothing.** It is
**not** an E1 substrate ruling (that is D2, Brendan's) and **not** a headroom or kill-gate test:
the kill gate is E-feas, which survived (`experiments/efeas/RESULTS.md`). It is one input to D2:
whether arm B, as trained, can use what a non-FIFO eviction rule leaves in its memory.

## Why the question exists

`P^(sent)` is added to memory keys by **rank** — position in the oldest-first prefix — not by
age (ADR-0006). Under FIFO rank and age are the same order, so a FIFO-trained model has only ever
seen a fact of gap `g` at rank index `M − g`, and has seen a fact at rank index 0 only when its
gap is exactly `M`. Any other eviction rule compacts the prefix: kept facts move to ranks, and
sit among neighbours, that FIFO never produced. If the model reads memory partly by rank, a
better eviction rule could be worth less through this model than it is model-free.

## What is already known (every number below has been seen by the author)

From the day's red team (`~/Documents/RSR-2026-09-26-day/redteam-scripts/`), re-run by the
day's researcher, on arm B's held-out set P and model-free unless marked:

- At M = 16 on this corpus at most **9** facts are ever pending, on every seed; the oracle is
  **decision-identical** to the causal rule "evict the oldest non-pending-assert slot" (0 of
  6144 victims differ). So the oracle is not a prospective policy here; it is a stand-in for
  "a rule that keeps facts and compacts the prefix".
- Model-free hit rate on P: FIFO 0.8148 / 0.8261 / 0.8104; oracle 1.0000 on every seed;
  held-out model-free headroom **0.1852 / 0.1739 / 0.1896** (mean 0.183). Fact/filler with
  random tie-break: +0.1498 / +0.1403 / +0.1577 over FIFO.
- On P the oracle keeps every gap 2..M fact resident (706/706, 722/722, 708/708), so on gap
  2..M an accuracy difference between oracle and FIFO is pure rank/neighbour effect.
- Oracle-rescued gap > M facts sit at rank index 0 (164/220, 147/207, 145/226) or 1
  (40, 45, 60), rarely 2–4. Fact/filler's rescued facts spread over ranks 0–12.
- Through the model (arm B ckpt 3000, P, FIFO, `answer_readout` live): all 0.7761 / 0.7824 /
  0.7978; gap > M 0.0591 / 0.0966 / 0.1549; FIFO's only native rank-0 evidence, gap = M:
  14/18, 16/18, 17/17 (0.7778 / 0.8889 / 1.0). Per-gap 10..16 accuracies 0.64–1.0 on n = 4–24.
  ckpt 2500 gap = M: 0.8889 / 0.6111 / 0.9412 (fresh-stream ledger).
- Arm A ckpt 3000 is at chance everywhere (live all 0.0606 / 0.0714 / 0.0562; live and
  slots-zeroed equal to 3 decimals on gap 2..M).
- `run_policy_loop` with `FIFOPolicy`, batch 64, `lengths = S`, eval + no_grad, is bit-exact
  to `answer_readout(cond="live")` on seed 0 ckpt 3000 (argmax equal; NLL max |diff| 0.0).
- While building the instrument (before this file): `answer_readout` run **one document at a
  time** on P, seed 0 ckpt 3000, gives identical argmax to the batch-64 readout and NLL max
  |diff| 1.12e-5 (float32 reduction order). So batch size changes NLL in the 5th decimal; the
  harness is B = 1 and its bit-exact control is against `answer_readout` also run at B = 1.
- `OraclePolicy` inside a batched `run_policy_loop` applies row 0's demand to every row
  (`policy_loop.py:358-365`; `MemoryState` has no document id). Hence B = 1.

No accuracy of any document of E under any arm, and no accuracy under the oracle or fact/filler
arm through any model, has been seen.

## Instrument

An experiment-local harness, `experiments/retention-readability/run.py`, on CPU, eval mode,
`torch.no_grad()`:

- Model: `TGModel(S0-03 _cfg(V))`, `checkpoint.load(..., restore_rng=False)`, V and the token
  map from `doc_sets(seed, 64)`.
- Each document is run **alone** (B = 1) through `rsr.model.tg.policy_loop.run_policy_loop`
  with `lengths = S = 48`, a **fresh policy per document**, wrapped so that the wrapper (i)
  carries the document id it was built for and the harness asserts it equals the id of the
  document whose tokens are being run; (ii) for the oracle, asserts its demand matrix equals
  `discounted_demand(doc, 0.97)` recomputed from that document; (iii) records the memory's
  `written_at` prefix after every write (`on_write`), which is the memory the next sentence
  reads.
- Per answer target (one per query sentence), logged: document id, query index `q`, assert index
  `a`, gap `q − a`, arm, **resident** (is `a` in the live prefix the query step reads), **rank
  index** (its position in that prefix, 0 = oldest; `null` if not resident), live-slot count,
  **correct** (argmax over the full vocabulary equals the answer token — S0-03's `answer_acc`),
  **NLL** (full vocabulary), NLL over the 16 answer symbols, **Brier16** (S0-03's definition).
- Buckets as S0-03: `gap_2_to_M` (2 ≤ gap ≤ 16), `gap_eq_M` (gap = 16), `gap_gt_M` (gap > 16),
  `all`.

## Controls (checked first; each overrides everything)

1. **Harness FIFO is bit-exact.** For every (arm, checkpoint, seed) and every document of U:
   the harness's FIFO arm vs `answer_readout(cond="live")` run on that document alone — argmax
   correctness identical on every answer and NLL max |diff| **== 0.0**.
2. **Ledger reproduction.** `S0-03 measure(ckpt, seed, "cpu", sets=doc_sets(seed, 64))` (the
   fresh-stream instrument, batched as it was run) reproduces every
   `{B.ckpt2500, B.ckpt3000, A.ckpt3000}.heldout.live.<bucket>.<readout>` sample of
   `.worktrees/fresh-stream/runs/fresh-stream/ledger.json` — all 8 S0-03 buckets × 4 readouts
   (`answer_acc`, `answer_nll`, `answer_nll_over_16`, `answer_brier_over_16`) — within 1e-6,
   per seed. Missing key = fail. The max |diff| is reported.
3. **One document, its own demand.** The per-document assertions of the instrument (id match,
   oracle demand match, B = 1) hold for every run; any assertion raising is a failed control.
4. **Harness memory = model-free memory.** For every document and arm, per-query residency
   through the model equals `rsr.metrics.headroom.simulate(doc, <same policy, fresh, same
   RNG>, 16)`'s `hit`, and FIFO residency equals `gap ≤ 16`.
5. **Documents.** E's ids are disjoint from the probe [0, 64), P [4096, 4160), arm B's stream
   [4160, 52160) and fresh-escape's stream [52160, 148160); every word of every E document is
   in the [0, 64) map. Either failing: exit 3 before any model is loaded.
6. **Arm A null.** Arm A ckpt 3000, on U: the 95% bootstrap CI of `acc_oracle(all) −
   acc_FIFO(all)` lies inside (−0.03, +0.03) on every seed. Arm A's memory is inert; a
   difference there is an instrument artefact, and it voids the arm-B readouts.

## Primary readouts (arm B, per checkpoint c ∈ {2500, 3000}, per seed, on U)

Every readout carries a 95% percentile CI from 2000 per-document bootstrap resamples of U,
**paired**: one resampled multiset of documents per replicate, every arm and bucket computed on
it. Accuracy is pooled over the answers of the resampled documents.

**(a) Rank penalty on facts FIFO keeps.** `RP = acc_FIFO(gap_2_to_M) − acc_oracle(gap_2_to_M)`.
Positive = the oracle's shifted ranks cost accuracy. Reported beside it: the fraction of those
answers under the oracle whose rank index differs from FIFO's `M − gap` (expected large), and
oracle residency on gap 2..M (expected 1.0).

- **NO_PENALTY**: on every seed, CI upper bound < RP_MAX = 0.03.
- **PENALTY**: on every seed, point ≥ 0.03 and CI lower bound > 0.
- **MIXED_PENALTY**: otherwise.

**(b) Readability of rescued facts.** Rescued = `gap_gt_M` answers whose assert is resident
under the oracle. `RA = acc_oracle(rescued)`. Against FIFO's own evidence at rank 0,
`REF = acc_FIFO(gap_eq_M)`, and FIFO's not-retained level, `FLOOR = acc_FIFO(gap_gt_M)`:
`D_ref = RA − REF`, `D_floor = RA − FLOOR`, each with its paired CI.

- **READABLE**: on every seed, `D_ref` CI lower bound > −READ_MARGIN = −0.10.
- **UNREADABLE**: on every seed, `D_floor` CI upper bound < +0.10.
- **UNDISTINGUISHED**: both READABLE and UNREADABLE hold (REF and FLOOR too close to separate).
- **MIXED_READ**: neither.

**Combined, per checkpoint** (headline = ckpt 3000):

| (a) | (b) | result |
|---|---|---|
| any control fails / a raise / a checkpoint missing | — | `inconclusive`, exit 3 |
| NO_PENALTY | READABLE | **READABLE_AT_SHIFTED_RANK** — this FIFO-trained model reads facts at ranks FIFO never produced about as well as it reads them where FIFO puts them |
| PENALTY | any | **RANK_BOUND** — what a better eviction rule keeps is worth less through this model than model-free; arm B's reading is tied to FIFO's rank layout |
| any | UNREADABLE | **RANK_BOUND** (as above) |
| any other combination (incl. UNDISTINGUISHED) | | **MIXED** — both parts reported; no summary claim |

The ledger verdict maps
READABLE_AT_SHIFTED_RANK → `survived`, RANK_BOUND → `falsified`, MIXED/inconclusive →
`inconclusive`, for the claim "arm B, as trained, can read facts kept at shifted ranks". If
ckpt 2500 and ckpt 3000 classify differently, both are reported and the result is flagged
**moving** (arm B has not converged; its loss is still falling).

**(c) Model-based headroom — descriptive, not a gate, never "the kill gate".**
`H_model = acc_oracle(all) − acc_FIFO(all)` with CI, printed next to the model-free
`H_free = hit_oracle − hit_FIFO` computed in the run on the same documents U (and next to
0.183, P's model-free mean). Also `acc_factfiller(all) − acc_FIFO(all)` beside its model-free
counterpart. The ratio `H_model / H_free` is reported, not classified.

**(d) Arm A null.** Control 6 above; its numbers are reported in full either way.

## Secondary readouts (reported, no verdict)

- Accuracy, NLL, Brier16 by rank index per arm and per gap bucket (the rank profile).
- Rank-matched (b): oracle-rescued answers at rank index 0 only vs FIFO `gap_eq_M` (also rank 0).
- Fact/filler's rescued facts, which spread over ranks 0–12: accuracy by rank index.
- Every primary readout on P alone (the set with the red team's numbers) and on E alone.
- NLL and Brier16 versions of (a) and (b).
- ckpt 2500 → 3000 change of every primary readout per seed.
- Batch-64 vs B = 1 FIFO on P: count of argmax flips and NLL max |diff| (diagnostic only).

## Author's expectation (before any number from this run; allowed to be wrong)

The task is content-addressed (the query names the entity and predicate), rank enters only as a
key bias, and FIFO already reads rank 0 at 0.78–1.0 on P. Expected: (a) NO_PENALTY at ckpt 3000
(≈ 60 %), the likeliest alternative being a small penalty from a fact-denser memory (more
same-kind distractors), not from rank; (b) READABLE (≈ 60 %), with RA a few points under REF;
headline READABLE_AT_SHIFTED_RANK ≈ 45 %, MIXED ≈ 40 %, RANK_BOUND ≈ 15 %. H_model a large
fraction (≥ 0.7) of H_free. Arm A null holds (≈ 95 %). ckpt 2500 noisier than 3000 on (b).

## What this does not establish

Nothing about ψ̂, the prospective term, or Kintsch & van Dijk: at M = 16 the oracle here is a
causal fact/filler rule (D1). Not a headroom test and not the kill gate. Not an E1 substrate
ruling (D2). One corpus, M = 16, one model width, FIFO-trained checkpoints only; no scaling
claim. READABLE says this model can read these ranks, not that RSR would learn to put facts
there. The oracle memory differs from FIFO's in neighbours as well as ranks; (a) measures the
two together and does not separate them.

## Time

Measured cost ≈ 0.15 s per document per pass at B = 1. Four passes per document (FIFO harness,
FIFO `answer_readout`, oracle, fact/filler) × 1088 documents × 9 (checkpoint, seed) pairs ≈
1.6 CPU-hours, three seeds as parallel processes ≈ 35 min. A prediction, not a premise.
