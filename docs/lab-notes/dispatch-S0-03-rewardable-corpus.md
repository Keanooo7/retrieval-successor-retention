# S0-03 — make the synthetic corpus rewardable

**Baseline:** `c647af4290f7a45be2edf947b7973816dcc2af8b`. **Lane:** researcher · CPU. **Sprint:** 0. **Depends on:** S0-01.

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

## Done when

Both bar numbers reported, `pytest -rs` green, the determinism digest test still passes
(`test_byte_identical_across_two_separate_processes` compares two live subprocesses, not a stored
golden, so a schema change is safe), and a PR merged.

## Do NOT

- Do not touch PG-19 or coref. That is S1-01 and it is a different lane.
- Do not tune `γ_b`, `ν`, or start E1.
- Do not report the old whole-corpus `Δ masked NLL = +0.03%` as evidence of anything. It is the
  null this brief exists to make measurable.

## Report

Standard block, `BRIEF ERRORS`, both bar numbers with spread, and the fraction of pairs at
`gap > M`.
