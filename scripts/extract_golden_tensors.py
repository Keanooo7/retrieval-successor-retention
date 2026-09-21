"""Extract the golden tensors from the pinned JAX reference (gauntlet 2.3).

**Runs in a THROWAWAY venv, never in the project venv.** JAX is not a project
dependency and must never enter `pyproject.toml`; `jaxlib` ships CPU wheels for
`macosx_11_0_arm64`, so the extraction runs here. Only `jax-metal` (the Metal GPU
backend) is dead, and the GPU is not wanted: **CPU is the better choice because the
fixtures have to be byte-reproducible**, which is ADR-0001 D3's own argument for
running E0b on CPU.

Regenerate with exactly this (recorded in `third_party/PINS.md` too):

    uv venv --python 3.12 /tmp/rsr-jaxenv
    uv pip install --python /tmp/rsr-jaxenv/bin/python \
        "jax==0.7.2" "jaxlib==0.7.2" "flax>=0.8.2" "numpy>=1.26"
    PYTHONPATH=third_party/ThoughtGestaltCode \
        /tmp/rsr-jaxenv/bin/python scripts/extract_golden_tensors.py \
        --out tests/fixtures/tg_d128_seed0.npz
    rm -rf /tmp/rsr-jaxenv

## What is captured, and why each one

Forward (tolerance `rtol=1e-4, atol=1e-5`, ADR-0002):

* **per-layer activations** -- every block's output, all 12, every step
* **gestalt vectors** -- `srep_BxD` per step, unit-norm by construction
* **cross-attention weights** -- captured through the `nn.Dropout` submodule inside
  `TgCrossAttn`, which with `deterministic=True` is the identity on the softmax.
  The vendored tree is **read-only**, so nothing is instrumented; Flax's
  `capture_intermediates` reads submodule outputs without touching the source.
* **logits**

Gradients (tolerance `rtol=1e-3, atol=1e-4`):

* **`W_sent`** -- `srep_head/proj` kernel and bias
* **transformer parameters** -- embeddings, every block, the output LayerNorm

## The positive control (gauntlet 2.3)

> *"a deliberately detached variant must **fail** the gradient check."*

The generator also runs the whole thing with `detach_sreps_for_memory=True` -- the
reference's own ablation switch, whose comment reads *"MUST stay False for the
recurrence to train: gradients have to flow from later sentences back through
memory"* -- and records, in the sidecar, how far that moves the gradients.

It is a control on the **tolerance**, not on the transcription: it establishes that
D-H's `rtol=1e-3, atol=1e-4` can actually separate a retained graph from a severed
one. Without it, "the gradients match" is compatible with "the gradients are
insensitive to the thing we are checking."

The control on the transcription is the other half, and it lives in
`tests/test_fidelity.py`: a PyTorch model built with detach ON must **fail** against
these fixtures.

## The two configuration choices that are not the reference's, and why

1. **`M = 8`, not 40.** With 20 sentence steps and `M = 40` the memory never fills,
   so `push_memory` never takes its roll-and-evict branch and **the fixtures would
   never exercise eviction at all** -- which is precisely the path where JAX
   functional autodiff and PyTorch retained-graph semantics diverge. At `M = 8`,
   steps 8..19 are evictions. This is the whole point of the gradient fixtures.
2. **`V = 512`, `L = 16`, `H = 2`.** A small vocabulary and short sentences keep the
   fixture small; `H = 2` preserves the reference's head dimension of 64 at
   `D = 128` (the reference is `D = 768 / H = 12`). Head *count* is the width axis
   under muP, head *dim* is not.

Everything else is the reference default, including `block_config = ('S','C')*6`,
`srep_extraction_layer = 6`, `srep_norm_target = 1.0` and `stm_cross_pos_mode =
'sinusoidal'`.

Dropout is **off** (`deterministic=True`), per ADR-0002: "matched dtype and matched
dropout (deterministic, `srep_dropout` disabled) before any tolerance is argued
about."

## The self-check

The generator runs the sentence loop **twice**: once explicitly, to capture
per-step intermediates, and once through the reference's own `run_sentence_loop`,
to compute the loss and its gradients. **If the two losses disagree the generator
aborts.** A hand-written loop in a fixture generator is a transcription, and an
unchecked transcription in the thing that validates transcriptions is the worst
place to put one.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import re
import sys
from pathlib import Path

# Stdlib-only import, so it works inside the throwaway JAX venv (no torch there).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rsr.exit_codes import ArgumentParser, Exit, refuse, run_main

# 🔴 Missing JAX/Flax/the vendored tree is DID NOT RUN (3). Uncaught, the
# ImportError exited 1 -- a real failure -- which is what this tool reported in the
# project venv, where JAX is correctly absent (S0-05).
try:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from tg.models import tg_model
    from tg.models.tg_config import TgConfig
except ImportError as _exc:
    refuse(
        Exit.DID_NOT_RUN,
        f"{_exc}. This tool runs in a throwaway JAX venv with "
        f"PYTHONPATH=third_party/ThoughtGestaltCode, never in the project venv; "
        f"see this file's docstring and third_party/PINS.md.",
    )

D_MODEL = 128
N_HEADS = 2  # preserves the reference's head_dim = 64
N_BLOCKS = 12
VOCAB = 512
MEMORY_SLOTS = 8  # small ON PURPOSE: 20 steps must include evictions
SENTENCE_TOKENS = 14  # + BOS + EOS = 16
N_STEPS = 20
BATCH = 2
SEED = 0

PAD_ID, BOS_ID, EOS_ID, EOD_ID = 508, 509, 510, 511


def build_config() -> TgConfig:
    return TgConfig(
        D=D_MODEL,
        H=N_HEADS,
        N=N_BLOCKS,
        V=VOCAB,
        max_sentence_tokens=SENTENCE_TOKENS,
        max_sentences_in_short_term=MEMORY_SLOTS,
        pad_id=PAD_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        eod_id=EOD_ID,
        fsdp_enabled=False,  # no mesh in a single-process extraction
    )


def build_batch(cfg: TgConfig, rng: np.random.Generator):
    """Sentences of the full span, every one ending in [EOS] so every step writes."""
    span = cfg.L
    ids = rng.integers(0, 500, size=(BATCH, N_STEPS, span), dtype=np.int32)
    ids[:, :, 0] = cfg.bos_id
    ids[:, :, -1] = cfg.eos_id
    mask = np.ones((BATCH, N_STEPS, span), dtype=np.int32)
    lengths = np.full((BATCH,), N_STEPS, dtype=np.int32)
    return jnp.asarray(ids), jnp.asarray(mask), jnp.asarray(lengths)


def _next_token_loss(logits_BxLxV, ids_BxL, mask_BxL, row_valid_B):
    """Mean next-token cross-entropy over valid positions, summed over steps."""
    logp = jax.nn.log_softmax(logits_BxLxV[:, :-1].astype(jnp.float32), axis=-1)
    tgt = ids_BxL[:, 1:]
    picked = jnp.take_along_axis(logp, tgt[..., None], axis=-1)[..., 0]
    valid = (mask_BxL[:, 1:] == 1) & row_valid_B[:, None]
    return -(picked * valid).sum() / jnp.maximum(valid.sum(), 1)


def loss_through_reference_loop(params, model, cfg, ids, mask, lengths):
    """The reference's own `run_sentence_loop`. This is the gradient path."""

    def step_fn(t, out, ids_BxL, mask_BxL, row_valid_B):
        del t
        return _next_token_loss(out.logits_BxLxV, ids_BxL, mask_BxL, row_valid_B)

    return tg_model.run_sentence_loop(
        model.apply,
        {"params": params},
        ids,
        mask,
        lengths,
        cfg,
        step_fn=step_fn,
        deterministic=True,
    )


def explicit_loop(params, model, cfg, ids, mask, lengths):
    """Mirror of `run_sentence_loop`, kept only to capture per-step intermediates.

    Validated against the real loop by comparing the accumulated loss; see the
    module docstring.
    """
    mem = tg_model.init_memory(BATCH, cfg)
    bos_ctx = jnp.zeros((BATCH, cfg.D), dtype=cfg.dtype)
    bos_valid = jnp.zeros((BATCH,), dtype=jnp.bool_)
    captured: list[dict] = []
    total = 0.0

    for t in range(N_STEPS):
        ids_t, mask_t = ids[:, t], mask[:, t]
        row_valid = t < lengths
        out, state = model.apply(
            {"params": params},
            ids_t,
            mask_t,
            mem.kv_BxMxD,
            mem.valid_BxM,
            bos_ctx,
            bos_valid,
            True,
            mem.step_BxM,
            capture_intermediates=True,
        )
        total = total + _next_token_loss(out.logits_BxLxV, ids_t, mask_t, row_valid)
        captured.append(
            {
                "logits": np.asarray(out.logits_BxLxV),
                "gestalt": np.asarray(out.srep_BxD),
                "srep_raw_norm": np.asarray(out.srep_raw_norm_B),
                "mem_valid": np.asarray(mem.valid_BxM),
                "n_live": int(np.asarray(mem.valid_BxM)[0].sum()),
                "intermediates": state["intermediates"],
            }
        )
        write = row_valid & out.has_eos_B
        mem = tg_model.push_memory(mem, out.srep_BxD, write, step=t)
        bos_ctx = out.srep_BxD
        bos_valid = write & ((t + 1) < lengths)
    return total, captured


def _flat(tree, prefix=""):
    """Flatten Flax's intermediates dict to `path -> array`."""
    out = {}
    for key, value in tree.items():
        path = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flat(value, path))
        elif isinstance(value, tuple):
            for i, v in enumerate(value):
                if hasattr(v, "shape"):
                    out[f"{path}[{i}]" if len(value) > 1 else path] = np.asarray(v)
        elif hasattr(value, "shape"):
            out[path] = np.asarray(value)
    return out


def main() -> Exit:
    ap = ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = build_config()
    model = tg_model.ThoughtGestaltDo(cfg)
    rng = np.random.default_rng(SEED)
    ids, mask, lengths = build_batch(cfg, rng)

    init_key = jax.random.PRNGKey(SEED)
    variables = model.init(
        {"params": init_key},
        ids[:, 0],
        mask[:, 0],
        jnp.zeros((BATCH, cfg.M, cfg.D)),
        jnp.zeros((BATCH, cfg.M), dtype=jnp.bool_),
        jnp.zeros((BATCH, cfg.D)),
        jnp.zeros((BATCH,), dtype=jnp.bool_),
        True,
        jnp.full((BATCH, cfg.M), -1, dtype=jnp.int32),
    )
    params = variables["params"]
    n_params = sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(params))
    print(
        f"parameters: {n_params:,}  (D={cfg.D} H={cfg.H} N={cfg.N} V={cfg.V} M={cfg.M})"
    )

    # --- forward, with intermediates --------------------------------------- #
    loss_explicit, captured = explicit_loop(params, model, cfg, ids, mask, lengths)

    # --- the self-check: the reference's own loop must agree ---------------- #
    loss_reference = loss_through_reference_loop(params, model, cfg, ids, mask, lengths)
    gap = float(abs(loss_explicit - loss_reference))
    print(f"loss (explicit loop)  = {float(loss_explicit):.10f}")
    print(f"loss (run_sentence_loop) = {float(loss_reference):.10f}")
    print(f"|gap| = {gap:.3e}")
    if gap > 1e-6:
        print(
            "ABORT: the capture loop disagrees with the reference's own loop. "
            "The fixtures would encode the generator's bug, not the model.",
            file=sys.stderr,
        )
        return Exit.FAIL  # a self-check failed: a real finding

    # --- gradients, through the reference loop ------------------------------ #
    grads = jax.grad(loss_through_reference_loop)(params, model, cfg, ids, mask, lengths)
    flat_grads = {
        "/".join(str(k.key) for k in path): np.asarray(v)
        for path, v in jax.tree_util.tree_flatten_with_path(grads)[0]
    }

    # --- the positive control: sever the S_REP -> STM graph ------------------ #
    cfg_det = dataclasses.replace(cfg, detach_sreps_for_memory=True)
    model_det = tg_model.ThoughtGestaltDo(cfg_det)
    loss_det = loss_through_reference_loop(params, model_det, cfg_det, ids, mask, lengths)
    grads_det = jax.grad(loss_through_reference_loop)(
        params, model_det, cfg_det, ids, mask, lengths
    )
    flat_det = {
        "/".join(str(k.key) for k in path): np.asarray(v)
        for path, v in jax.tree_util.tree_flatten_with_path(grads_det)[0]
    }
    g_rtol, g_atol = 1e-3, 1e-4  # ADR-0002's gradient tolerance
    outside, worst_key, worst_rel = 0, None, 0.0
    for key, g in flat_grads.items():
        if np.allclose(g, flat_det[key], rtol=g_rtol, atol=g_atol):
            continue
        outside += 1
        denom = float(np.abs(flat_det[key]).max()) or 1.0
        rel = float(np.abs(g - flat_det[key]).max()) / denom
        if rel > worst_rel:
            worst_key, worst_rel = key, rel
    control = {
        "description": (
            "detach_sreps_for_memory=True severs the S_REP -> STM graph. The "
            "forward pass is unchanged; only the backward pass is. If the loss "
            "below is identical and the gradient count is large, D-H's gradient "
            "tolerance can separate a retained graph from a severed one."
        ),
        "loss_retained": float(loss_reference),
        "loss_detached": float(loss_det),
        "forward_identical": float(loss_reference) == float(loss_det),
        "gradient_arrays": len(flat_grads),
        "arrays_outside_gradient_tolerance": outside,
        "tolerance": {"rtol": g_rtol, "atol": g_atol},
        "worst_array": worst_key,
        "worst_relative_delta": worst_rel,
    }
    print(
        f"positive control: forward loss identical="
        f"{control['forward_identical']}, {outside}/{len(flat_grads)} gradient "
        f"arrays outside D-H, worst {worst_key} at {worst_rel:.3f}"
    )
    if not control["forward_identical"] or outside < len(flat_grads) // 4:
        print(
            "ABORT: the positive control is not a control. Either detaching moved "
            "the forward pass (so the comparison is not isolating the graph), or "
            "it barely moved the gradients (so the tolerance cannot detect it).",
            file=sys.stderr,
        )
        return Exit.FAIL  # a self-check failed: a real finding

    payload: dict[str, np.ndarray] = {}

    # Forward quantities, stacked over steps.
    payload["logits"] = np.stack([c["logits"] for c in captured])
    payload["gestalts"] = np.stack([c["gestalt"] for c in captured])
    payload["gestalt_raw_norms"] = np.stack([c["srep_raw_norm"] for c in captured])
    payload["memory_valid"] = np.stack([c["mem_valid"] for c in captured])

    per_step_flat = [_flat(c["intermediates"]) for c in captured]

    # Only the block OUTPUTS, not every Dense inside them. `startswith("blocks_")`
    # matches every nested path and silently captured 582 arrays / 85 MB on the
    # first run; the per-layer profile §3.2.1 asks for is the block output.
    block_re = re.compile(r"^blocks_(\d+)/__call__$")
    block_keys = sorted(
        (k for k in per_step_flat[0] if block_re.match(k)),
        key=lambda k: int(block_re.match(k).group(1)),
    )
    for key in block_keys:
        payload[f"activations/{key}"] = np.stack([s[key] for s in per_step_flat])

    # The cross-attention probabilities. `TgCrossAttn` returns only its output, and
    # the vendored tree is read-only -- but the softmax is piped through an
    # `nn.Dropout` submodule, which at `deterministic=True` is the identity, so
    # `capture_intermediates` reads the weights without instrumenting anything.
    xattn_re = re.compile(r"^blocks_(\d+)/cross_attn/Dropout_\d+/__call__$")
    xattn_keys = sorted(
        (k for k in per_step_flat[0] if xattn_re.match(k)),
        key=lambda k: int(xattn_re.match(k).group(1)),
    )
    if len(xattn_keys) != 6:
        print(
            f"ABORT: expected 6 cross-attention layers (block_config ('S','C')*6), "
            f"captured {len(xattn_keys)}: {xattn_keys}",
            file=sys.stderr,
        )
        return Exit.FAIL  # a self-check failed: a real finding
    for key in xattn_keys:
        payload[f"cross_attention/{key}"] = np.stack([s[key] for s in per_step_flat])

    for name, g in flat_grads.items():
        payload[f"grad/{name}"] = g

    # Inputs, so the fixture is self-contained and the PyTorch side needs no RNG.
    payload["input_ids"] = np.asarray(ids)
    payload["input_mask"] = np.asarray(mask)
    payload["input_lengths"] = np.asarray(lengths)
    payload["loss"] = np.asarray(float(loss_reference))
    for name, p in {
        "/".join(str(k.key) for k in path): np.asarray(v)
        for path, v in jax.tree_util.tree_flatten_with_path(params)[0]
    }.items():
        payload[f"param/{name}"] = p

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()

    n_evictions = sum(
        1 for i in range(1, len(captured)) if captured[i]["n_live"] == cfg.M
    )
    meta = {
        "jax": jax.__version__,
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "device": str(jax.devices()[0]),
        "tg_pin": "f220b1098d24a02c94907043d6205c113b31ebb6",
        "config": {
            "D": cfg.D,
            "H": cfg.H,
            "N": cfg.N,
            "V": cfg.V,
            "M": cfg.M,
            "L": cfg.L,
            "steps": N_STEPS,
            "batch": BATCH,
            "seed": SEED,
            "srep_extraction_layer": cfg.srep_extraction_layer,
            "srep_norm_target": cfg.srep_norm_target,
            # 🔴 The token ids are part of the model's behaviour, not of the data.
            # The gestalt is read at the first [EOS], and memory is written only
            # when a sentence HAS one -- so a consumer that defaults `eos_id`
            # silently reads position 0 and never writes memory. It cost one
            # debugging cycle on 2026-09-17; the config now carries them and
            # `TGConfig.from_reference_dict` refuses a fixture without them.
            "pad_id": cfg.pad_id,
            "bos_id": cfg.bos_id,
            "eos_id": cfg.eos_id,
            "eod_id": cfg.eod_id,
            "block_config": list(cfg.block_config),
            "deterministic": True,
        },
        "n_params": n_params,
        "loss": float(loss_reference),
        "arrays": len(payload),
        "bytes": args.out.stat().st_size,
        "sha256": digest,
        "steps_with_full_memory": n_evictions,
        "detach_positive_control": control,
        "activation_keys": block_keys,
        "cross_attention_keys": xattn_keys,
        "grad_keys": sorted(flat_grads),
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")

    print(
        f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB, "
        f"{len(payload)} arrays)"
    )
    print(f"sha256 {digest}")
    print(f"steps at full memory (i.e. evicting): {n_evictions} of {N_STEPS}")
    norms = np.linalg.norm(payload["gestalts"], axis=-1)
    print(f"gestalt L2 norms: min={norms.min():.6f} max={norms.max():.6f}  (expect 1.0)")
    return Exit.OK


if __name__ == "__main__":
    run_main(main)
