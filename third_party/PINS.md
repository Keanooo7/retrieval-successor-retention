# Vendored source pins

## ThoughtGestaltCode

The reference implementation of [P2], Thought Gestalt.

| | |
|---|---|
| Upstream | https://github.com/jlmcc94303/ThoughtGestaltCode |
| Licence | Apache-2.0 |
| Pinned commit | `f220b1098d24a02c94907043d6205c113b31ebb6` — "Add corpus preprocessing and dataset subset scripts.", 2026-09-02 |
| Release commit | `a0079d08` — "Release Thought Gestalt: sentence-level recurrent LM built on NanoDO", 2026-08-21 |
| Framework | **JAX / Flax**, built on [`google-deepmind/nanodo`](https://github.com/google-deepmind/nanodo) (Apache-2.0) |
| Paper | https://arxiv.org/abs/2512.25026 |

**Read-only. Never imported, never installed, never a dependency.** See
[ADR-0001](../docs/decisions/ADR-0001-tg-base.md).

This directory exists for two purposes:

1. **A source reference** to transcribe the PyTorch TG from.
2. **A one-time golden-tensor extraction**, which has **run, here, on CPU** — see
   the extraction section below. `jax-metal` (the Metal *GPU* backend) is indeed
   dead, but `jaxlib` ships `macosx_11_0_arm64` **CPU** wheels, and CPU is the
   correct target because the fixtures must be byte-reproducible. JAX is installed
   only in a throwaway venv and is never a project dependency.

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

### RESOLVED — the gestalt layer is 6 **0-indexed**, which is the spec's 7

`tg/models/tg_config.py:146`:

```python
srep_extraction_layer: int = 6  # 0-indexed block whose output feeds the head
```

and `srep_layer_idx` returns `min(srep_extraction_layer, N - 1)`. **The spec's
"layer 7" (§5.1, 1-indexed) and the README's "layer 6" (0-indexed) are the same
block.** The off-by-one was the explanation; it was verified against the source, not
guessed. The transcription indexes from 0 and extracts at block 6.

**Sprint 1 transcribes `d = 128` only** (ADR-0001 D5). `d` stays a config axis;
other widths instantiate at E0a and E5.

### Vendoring — DONE, 2026-09-17 (gauntlet 2.1)

```bash
git clone https://github.com/jlmcc94303/ThoughtGestaltCode third_party/ThoughtGestaltCode
git -C third_party/ThoughtGestaltCode checkout f220b109
rm -rf third_party/ThoughtGestaltCode/.git
```

**Verified against the pin before `.git` was removed:**

```
$ git -C third_party/ThoughtGestaltCode rev-parse HEAD
f220b1098d24a02c94907043d6205c113b31ebb6
```

42 tracked files, 292 KB after removing `.git`. **Read-only.** JAX is not installed
here and is not a dependency; this tree is a source reference and the origin of a
one-time tensor extraction, which has run here on CPU in a throwaway venv.

Because `.git` is gone, the sha cannot be re-derived from the tree. It is recorded
above and in the vendoring commit message; re-verify by re-cloning if it ever
matters.

### Golden-tensor extraction — DONE, 2026-09-17 (gauntlet 2.3)

**JAX runs here, on CPU.** `jaxlib` ships `macosx_11_0_arm64` CPU wheels; only
`jax-metal`, the Metal *GPU* backend, is dead. CPU is the right target anyway —
fixtures must be byte-reproducible, which is ADR-0001 D3's own argument for running
E0b on CPU.

**JAX is still not a project dependency and never enters `pyproject.toml`.** The
extraction runs in a throwaway venv that is created, used, and deleted:

```bash
uv venv --python 3.12 /tmp/rsr-jaxenv
uv pip install --python /tmp/rsr-jaxenv/bin/python \
    "jax==0.7.2" "jaxlib==0.7.2" "flax>=0.8.2" "numpy>=1.26"
PYTHONPATH=third_party/ThoughtGestaltCode \
    /tmp/rsr-jaxenv/bin/python scripts/extract_golden_tensors.py \
    --out tests/fixtures/tg_d128_seed0.npz
rm -rf /tmp/rsr-jaxenv
```

Produced, on `Darwin arm64`, `TFRT_CPU_0`, jax 0.7.2 / python 3.12.13:

```
parameters: 2,478,278  (D=128 H=2 N=12 V=512 M=8)
loss (explicit loop)     = 125.3105468750
loss (run_sentence_loop) = 125.3105468750      |gap| = 0.000e+00
wrote tests/fixtures/tg_d128_seed0.npz (23.48 MB, 438 arrays)
sha256 79edafe967efff900ab3fe2d10a15cd7757398289a89f672588aad5efd2cd4e8
steps at full memory (i.e. evicting): 12 of 20
gestalt L2 norms: min=1.000000 max=1.000000
```

**Byte-reproducible**, verified by re-running under a different `PYTHONHASHSEED` in
a fresh process: identical sha256.

Three choices in the extraction that are not the reference's defaults, each for a
stated reason:

- **`M = 8`, not 40.** 🔴 The most important one. At `M = 40` a 20-step extraction
  never fills memory, so `push_memory` never takes its roll-and-evict branch and the
  gradient path *through eviction* — the path where JAX functional autodiff and
  PyTorch retained-graph semantics actually diverge — would be missing from the
  fixtures while the fixtures looked complete. At `M = 8`, 12 of 20 steps evict.
- **`V = 512`, `L = 16`.** Keeps the fixture to 23 MB. Shapes are parametric; the
  mechanism is not.
- **`H = 2`.** Preserves the reference's head dimension of 64 at `D = 128` (the
  reference is `D = 768 / H = 12`). Head *count* is the width axis under μP; head
  *dim* is not.

The generator runs the sentence loop twice — once explicitly to capture per-step
intermediates, once through the reference's own `run_sentence_loop` for the loss and
gradients — and **aborts if the two losses disagree.** A hand-written loop inside the
thing that validates transcriptions is itself a transcription, and an unchecked one
there is the worst place to put it. Measured gap: exactly zero.

Cross-attention probabilities are read through `capture_intermediates`, via the
`nn.Dropout` submodule that the softmax is piped through (the identity at
`deterministic=True`). **Nothing in the vendored tree was instrumented or modified.**

### What the vendored source settled

Six claims that were previously **relayed** from a session that had the tree are now
read directly here. See `docs/code-vs-paper.md`, where each row carries its
provenance mark:

| Claim | Where | Verdict |
|---|---|---|
| Gestalt is L2-normalized, `srep_norm_target = 1.0` | `tg_srep_head.py`, `tg_config.py:149` | confirmed |
| `P^(sent)` is rank-indexed, keys only | `tg_cross_attention.py:65-70,158` | confirmed |
| `memory_gate` is a per-layer learnable scalar, applied before the residual add | `tg_model.py:311-314` | confirmed |
| `attn_dropout = 0.2` | `tg_config.py:134` | confirmed |
| Cross-attention on six layers | `DEFAULT_BLOCK_CONFIG = ('S','C') * 6` | confirmed — blocks 1,3,5,7,9,11 (0-indexed) |
| Gestalt layer 6 vs 7 | `tg_config.py:146` | resolved: same block, different base |

And four the audit had not seen at all, now recorded as new rows in
`docs/code-vs-paper.md`: the S_REP head is `LayerNorm -> dropout -> MLP -> L2
normalize` rather than a bare `W_sent`; there is an auxiliary **norm hinge penalty**
on the *pre*-normalization norm; the sinusoidal positional encoding is **itself**
L2-normalized; and `stm_cross_pos_mode`/`stm_positional_weight` give D-D's
`P^(sent)`-ablated arm as a config flag rather than new code.
