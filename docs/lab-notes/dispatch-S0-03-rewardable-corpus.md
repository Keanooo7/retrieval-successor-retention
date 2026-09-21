# S0-03 — make the synthetic corpus rewardable

**Baseline:** `3a458adbe28b65fdfa0ed0039fd01fb4becb86ae` (re-baselined 2026-09-21 by the manager; was `c647af4`). **Lane:** researcher · CPU. **Sprint:** 0. **Depends on:** S0-01 (landed, #15).
**Worktree:** `.worktrees/s0-03` on branch `s0/s0-03` from `origin/main`. One agent, one worktree.
**run_id:** `s0-03-rewardable-corpus`.

> **Re-baseline, 2026-09-21 — every citation was re-derived with a command at `3a458ad`:**
> - `grep -n answer src/rsr/data/synthetic.py`: the answer is set at `:227` (`kinds[j], fact_of[j], answers[j] = "query", fact_id, obj`) and carried at `:238` (`Sentence(... answer=answers[i])`). The field is declared at `:168`, and the query text is built at `:228` (`f"What does {entity} {predicate}?"`). **Unchanged.**
> - `_OBJECTS` (`:89`) holds **16** objects of 2–3 words each (measured: `len(_OBJECTS) == 16`). The `ln(16)` chance figure below is right **only for single-symbol objects**. At word level, chance differs per position, so state it per position if you choose word level.
> - `tests/test_synthetic.py:176` is `test_byte_identical_across_two_separate_processes`. **Unchanged.**
> - `src/rsr/train/loop.py`: `encode` at `:101`, `step_fn` at `:331`, and `masked_loss` defaulting to `True` at `:235`.
> - The "1,180 queries" figure is seed 0's `seed0.n_queries` in `runs/efeas-synthetic/ledger.json`.

## Hypothesis (not instruction)

`src/rsr/data/synthetic.py` stores each query's answer **out of band** — `Sentence.answer`, set
at `:227`, carried at `:238` — and it **never enters the token stream**. Every query reads
*"What does Hal-6 measures?"* with no object. Verified: across 1,180 queries, no word of any answer
appears in its query's text.

**So the cross-entropy objective has no target token that requires retrieving the asserted fact.**
If that is right, then "the memory is inert" is currently a finding about the corpus, not about TG,
and it is upstream of every cause the overnight loop proposed.

## Files in scope

`src/rsr/data/synthetic.py` · `src/rsr/train/loop.py` (the `encode`/`step_fn` path only) ·
`tests/test_synthetic.py`.

## What to build

1. **Emit the answer as a target span** — either appended to the query sentence or as an
   `answer_span` on `Sentence`. Objects are 2–3 words; decide word-level or single-symbol and say
   which.
2. **A `target_mask` alongside `ids`.** `encode()` returns a *padding* mask today. Without a
   separate supervision mask the answer tokens are ~6 of 193,536 targets and drown.
3. **Bucket the answer-token loss by gap.** That bucketing is what makes this an eviction test
   rather than a memorization test.
4. **A generator invariant** that a reportable fraction of assert→query pairs has `gap > M`
   (`M = 16` on synthetic). Report the fraction.

## Bar

🔴 **A memory-ablation control, and it is the bar, not a nice-to-have.** With all `M` slots zeroed,
answer-token loss must be **near chance** — `ln(16) ≈ 2.77` per object symbol if you use single
symbols. **If it is not, the model is reading the answer from somewhere else and the corpus is still
not testing retrieval.** Report the number with spread.

Second: the loss on answer tokens at `gap > M` must be **worse than at `gap < M`** under FIFO.
If it is not, eviction is not biting and the gap distribution is wrong.

**Added 2026-09-21 by the manager (overnight dispatch, H4: the zeroed-only bar is vacuous on its own, because an inert memory also predicts it):**

> Report answer-token loss with memory **live** and with all `M` slots **zeroed**, the gap between them, and the same split at `gap > M` vs `gap < M` under FIFO. **If live and zeroed are both at chance, the corpus is built but retrieval is not shown.** That is a valid return that routes to the decisive run, **not a pass of this bar.**

**Required in the return:**
- Every figure is a ledger key in `runs/s0-03-rewardable-corpus/ledger.json`, with a `how`.
- Seeds `0, 1, 2`, with the sd, on **CPU**. Report `sd_exactly_zero`. An sd of exactly `0.0` is a broken run, not a clean one.
- The `gap > M` fraction.
- The manifest records the **torch version**.
- For training, use `experiments/shuffle-control/run.py`'s `CONFIG` (300 iters, batch 16, 16 slots, 48 steps per stream) with `masked_loss=True` and `srep_norm_reg_weight=0.0`, unless you say why not in `BRIEF ERRORS`. **Do not run any shuffle-control / liveness readout.** That is the decisive run's measurement, and it is pre-registered separately.

**Mutation bar.** The corpus code is new, so a code mutation is trivially "only this test". **Mutate the fixture instead:** make the generator emit the answer out of band again (the pre-S0-03 behaviour), and require a named test to redden. Quote the node id.

## Done when

Both bar numbers reported (with the live/zeroed split above), `pytest -rs` green (census line + `$?` quoted), `ruff check` / `ruff format --check` exit `0`, `mutation_battery.py --check` exits `0`, the determinism digest test still passes
(`test_byte_identical_across_two_separate_processes` compares two live subprocesses, not a stored
golden, so a schema change is safe), and a PR **opened**. The manager verifies and merges; do not merge.

## Do NOT

- Do not touch anything under an existing `runs/*/`, or anything in `preregistration/`.
- Do not run in the main checkout.

- Do not touch PG-19 or coref. That is S1-01 and it is a different lane.
- Do not tune `γ_b`, `ν`, or start E1.
- Do not report the old whole-corpus `Δ masked NLL = +0.03%` as evidence of anything. It is the
  null this brief exists to make measurable.

## Report

Standard block, `BRIEF ERRORS`, both bar numbers with spread, and the fraction of pairs at
`gap > M`.
