# Vendored source pins

## ThoughtGestaltCode

The reference implementation of [P2], Thought Gestalt.

| | |
|---|---|
| Upstream | https://github.com/jlmcc94303/ThoughtGestaltCode |
| Licence | Apache-2.0 |
| Pinned commit | `f220b109` — "Add corpus preprocessing and dataset subset scripts.", 2026-09-02 |
| Release commit | `a0079d08` — "Release Thought Gestalt: sentence-level recurrent LM built on NanoDO", 2026-08-21 |
| Framework | **JAX / Flax**, built on [`google-deepmind/nanodo`](https://github.com/google-deepmind/nanodo) (Apache-2.0) |
| Paper | https://arxiv.org/abs/2512.25026 |

**Read-only. Never imported, never installed, never a dependency.** See
[ADR-0001](../docs/decisions/ADR-0001-tg-base.md).

This directory exists for two purposes:

1. **A source reference** to transcribe the PyTorch TG from.
2. **A one-time golden-tensor extraction**, which runs on rented hardware. JAX is
   never installed on the development machine — `jax-metal`'s last release was
   v0.1.1 on 2024-10-08 and current JAX requires Python ≥3.12, which the reference
   forbids, so there is no working GPU path for it on Apple silicon anyway.

### Known defects in the pinned tree

Recorded so nobody rediscovers them:

- **The release is not the paper's model.** README: *"This version differs slightly
  from the version of the model described in (arXiv:2512.25026). It achieves
  comparable results to those reported in the paper when trained with 12M text
  tokens."* The kickoff's smoke test against 29.8 test PPL / 21 sentence-steps/sec
  is therefore unreachable; see correction 14.
- **`tg/Rough/preprocess_corpus.py` does not run.** It imports
  `src_recurrent.pipelines.data.io`, `src_recurrent.core.tokenizer_setup` and
  `src_recurrent.core.data.sentence_splitter`; **`src_recurrent` does not exist
  anywhere in the tree.** RSR writes its own preprocessing. The script does tell us
  the segmenter: SaT `sat-3l-sm`, with an explicit refusal to fall back to regex.
- **The gist baseline is buggy upstream.** `EXPERIMENTS.md`: *"The gist mask here
  is derived from the paper's description rather than from the original
  implementation, whose mask indexes the gist flag on the query axis instead of the
  key axis and so grants no cross-sentence access at all."* That does not affect
  RSR — TG itself is what we transcribe — but it is a caution against treating
  [P2]'s reported comparisons as settled.

### Reference defaults, for the transcription

12 layers · `d_model` 768 · 12 heads · SwiGLU · alternating self/cross-attention
blocks (`S,C,S,C,…`) · `[BOS]` + ≤64 tokens + `[EOS]` padded to 66 · working memory
**40** sentence vectors · `S_REP` extracted at **layer 6**, L2-normalized · AdamW,
peak LR 2.5e-4, cosine decay, linear warmup from 0 · token-budget bucketing at
20,000 supervised tokens/step · stream curriculum 30 sentences, **+12 every 5
epochs**.

Note the spec says the gestalt layer is **7** (§5.1, citing [P2]) while the
reference README says **6**. Resolve during transcription against `tg_srep_head.py`
and record the answer here. Off-by-one in layer indexing is the likely explanation
and it must not be guessed.

**Sprint 1 transcribes `d = 128` only** (ADR-0001 D5). `d` stays a config axis;
other widths instantiate at E0a and E5.

### Vendoring

Not yet vendored — pending the first step of the Sprint 1 critical path. Vendor with:

```bash
git clone https://github.com/jlmcc94303/ThoughtGestaltCode third_party/ThoughtGestaltCode
git -C third_party/ThoughtGestaltCode checkout f220b109
rm -rf third_party/ThoughtGestaltCode/.git
```
