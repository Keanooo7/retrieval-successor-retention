"""Thought Gestalt in PyTorch -- a transcription of the pinned JAX reference.

**The reference is the code, not the paper** (D-A): the released implementation's
own README says it *"differs slightly from the version of the model described in
(arXiv:2512.25026)"*, so `tests/test_fidelity.py` compares this module against
tensors extracted from `third_party/ThoughtGestaltCode` at
`f220b1098d24a02c94907043d6205c113b31ebb6`, to a tolerance committed in ADR-0002
before those tensors existed.

Module and parameter names mirror the Flax tree exactly (`blocks_3/ln_mem/scale`,
`srep_head/proj/kernel`, ...) so `loading.py` maps weights by name rather than by
position. A positional mapping is right until someone inserts a layer.

## The five places a transcription of *this* model goes wrong

1. **The S_REP -> STM path must NOT be detached.** The reference's own comment:
   *"MUST stay False for the recurrence to train: gradients have to flow from later
   sentences back through memory."* A detached variant matches every forward
   quantity and fails only on gradients -- measured, on the reference itself:
   identical loss (`125.3105468750`), 113 of 206 gradient arrays outside D-H.
2. **`push_memory` rolls.** Occupied slots live in an oldest-first **prefix**; when
   full the buffer shifts left and the newest slot is overwritten. That is why
   `P^(sent)` indexes *rank* (ADR-0006), and it is the branch the fixtures are
   built to exercise (`M = 8`, 20 steps, 12 evictions).
3. **The gestalt is L2-normalised to unit length**, and the head is
   `LayerNorm -> dropout -> Dense -> normalize`, not a bare `W_sent`
   (`docs/code-vs-paper.md` rows 1 and 9).
4. **Positional encoding goes on the memory KEYS only.** `V_M` sees the raw
   gestalts. The encoding is itself L2-normalised, so it has the same magnitude as
   the unit-norm gestalt it is added to (row 11).
5. **Softmax in float32**, then back to the working dtype, and masked with
   `finfo(float32).min` -- not `-inf`, which produces NaN on an all-masked row.

## What is deliberately absent

Block types `SC`/`CS`/`P`, multi-EOS, in-context memory, `mean_pool` pooling and
muP's `8_over_d` attention scale. `TGConfig` **raises** on each rather than
accepting and ignoring it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from rsr.model.tg.config import TGConfig

KERNEL_STD = 0.02
"""`cfg.kernel_init = nn.initializers.normal(0.02)` in the reference, applied to
every `nn.Dense`/`nn.Embed`. Attention **in**-projections are the exception and use
`xavier_uniform`; see `DenseGeneralIn.reset_parameters`."""

__all__ = [
    "KERNEL_STD",
    "Memory",
    "StepOutput",
    "TGModel",
    "init_memory",
    "memory_positions",
    "push_memory",
    "run_sentence_loop",
    "sinusoidal_key_pe",
]


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #


@dataclass
class Memory:
    """Up to `M` gestalts, **oldest -> newest in a prefix**.

    The prefix ordering is not incidental: it is what makes `arange(M)` the
    positional index, and therefore what makes `P^(sent)` rank-indexed (ADR-0006).
    """

    kv: Tensor  # [B, M, D]
    valid: Tensor  # [B, M] bool
    step: Tensor  # [B, M] int64, -1 where empty


def init_memory(batch: int, cfg: TGConfig, *, device=None, dtype=None) -> Memory:
    dtype = dtype or torch.float32
    return Memory(
        kv=torch.zeros(batch, cfg.M, cfg.D, device=device, dtype=dtype),
        valid=torch.zeros(batch, cfg.M, device=device, dtype=torch.bool),
        step=torch.full((batch, cfg.M), -1, device=device, dtype=torch.long),
    )


def push_memory(
    mem: Memory,
    srep: Tensor,
    write: Tensor,
    step: int = 0,
    backprop_window: int | None = None,
) -> Memory:
    """Append one gestalt per row, **rolling** when full. Rows with `write=False`
    are untouched.

    🔴 No `detach` anywhere on this path. Sentence `t+1`'s loss must reach sentence
    `t` through this tensor; severing it is invisible in the forward pass.
    """
    kv, valid = mem.kv, mem.valid
    m = kv.shape[1]
    srep = srep.to(kv.dtype)

    k = valid.to(torch.int64).sum(dim=-1)  # [B] occupied count
    full = (k >= m).unsqueeze(-1)  # [B, 1]

    kv_rolled = torch.cat([kv[:, 1:], torch.zeros_like(kv[:, :1])], dim=1)
    valid_rolled = torch.cat([valid[:, 1:], torch.zeros_like(valid[:, :1])], dim=1)
    step_rolled = torch.cat(
        [mem.step[:, 1:], torch.full_like(mem.step[:, :1], -1)], dim=1
    )

    kv_base = torch.where(full.unsqueeze(-1), kv_rolled, kv)
    valid_base = torch.where(full, valid_rolled, valid)
    step_base = torch.where(full, step_rolled, mem.step)

    slot = torch.where(full.squeeze(-1), torch.full_like(k, m - 1), k)  # [B]
    onehot = torch.nn.functional.one_hot(slot, m).to(kv.dtype)  # [B, M]
    kv_new = kv_base * (1.0 - onehot.unsqueeze(-1)) + srep.unsqueeze(
        1
    ) * onehot.unsqueeze(-1)
    is_slot = onehot > 0
    valid_new = valid_base | is_slot
    step_new = torch.where(is_slot, torch.full_like(mem.step, step), step_base)

    if backprop_window is not None:
        # The reference's deque is newest-first and detaches at index >= window;
        # this prefix is oldest-first, so slot j of k has age k-1-j.
        k_new = valid_new.to(torch.int64).sum(dim=-1, keepdim=True)
        age = (k_new - 1) - torch.arange(m, device=kv.device).unsqueeze(0)
        keep = valid_new & (age < int(backprop_window))
        kv_new = torch.where(keep.unsqueeze(-1), kv_new, kv_new.detach())

    write_m = write.unsqueeze(-1).expand_as(valid)
    return Memory(
        kv=torch.where(write_m.unsqueeze(-1), kv_new, kv),
        valid=torch.where(write_m, valid_new, valid),
        step=torch.where(write_m, step_new, mem.step),
    )


def memory_positions(valid: Tensor) -> Tensor:
    """`[B, M]` -- each slot's position **oldest -> newest**; -1 where empty.

    ADR-0006: this is rank in the memory ordering, not age. Under FIFO the two
    coincide; under a learned policy they do not.
    """
    idx = torch.arange(valid.shape[-1], device=valid.device).unsqueeze(0)
    return torch.where(valid, idx, torch.full_like(idx, -1).expand_as(valid))


def sinusoidal_key_pe(
    positions: Tensor, d: int, positional_weight: float = 1.0, dtype=torch.float32
) -> Tensor:
    """L2-normalised sinusoidal encoding for the memory keys.

    Even dims sin, odd dims cos, **the whole vector L2-normalised** (eps 1e-6),
    zeroed on invalid slots, then scaled. Computed in float32 regardless of
    `dtype`.

    The normalisation is the reference's, not the paper's, and it matters: the
    encoding then has norm 1, and so does the gestalt it is added to, so the
    positional term is **half the key** rather than a small perturbation
    (`docs/code-vs-paper.md` row 11).
    """
    if d % 2:
        raise ValueError("memory_dim must be even for sinusoidal STM positions")
    valid = positions >= 0
    inv_freq = torch.exp(
        torch.arange(0, d, 2, device=positions.device, dtype=torch.float32)
        * -(math.log(10000.0) / d)
    )
    clamped = positions.clamp(min=0).to(torch.float32)
    angles = clamped.unsqueeze(-1) * inv_freq.view(1, 1, -1)
    pe = torch.stack([angles.sin(), angles.cos()], dim=-1).reshape(*positions.shape, d)
    pe = pe / pe.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    pe = pe * valid.unsqueeze(-1).to(pe.dtype)
    return (pe * positional_weight).to(dtype)


# --------------------------------------------------------------------------- #
# Layers
# --------------------------------------------------------------------------- #


class DenseGeneralIn(nn.Module):
    """Flax `DenseGeneral(axis=-1, features=(H, Dh))`: kernel `[D, H, Dh]`."""

    def __init__(self, d: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.kernel = nn.Parameter(torch.empty(d, heads, head_dim))
        self.bias = nn.Parameter(torch.zeros(heads, head_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """`attn_in_proj` -> `xavier_uniform`, with `fan_in = D`, `fan_out = H·Dh`.

        **Measured from the reference's own initial weights**, not derived: the
        fixture holds an untrained model, so `param/blocks_0/self_attn/query/kernel`
        IS the initialisation. Its std is 0.0884 across all six layers, and
        `sqrt(2/(D + H·Dh)) = sqrt(2/256) = 0.0884`. Reasoning from Flax's
        `_compute_fans` instead gives fan_in 256 / fan_out 8192 and a std of
        0.0154 -- wrong by 5.7x, and nothing would have failed.
        """
        d, heads, head_dim = self.kernel.shape
        nn.init.xavier_uniform_(self.kernel.view(d, heads * head_dim))
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:  # [..., D] -> [..., H, Dh]
        return torch.einsum("...i,ihd->...hd", x, self.kernel) + self.bias


class DenseGeneralOut(nn.Module):
    """Flax `DenseGeneral(axis=(-2,-1), features=D)`: kernel `[H, Dh, D]`."""

    def __init__(self, heads: int, head_dim: int, d: int) -> None:
        super().__init__()
        self.kernel = nn.Parameter(torch.empty(heads, head_dim, d))
        self.bias = nn.Parameter(torch.zeros(d))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """`attn_out_proj` -> `kernel_init` = `normal(0, 0.02)`. Measured: 0.0200."""
        nn.init.normal_(self.kernel, mean=0.0, std=KERNEL_STD)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:  # [..., H, Dh] -> [..., D]
        return torch.einsum("...hk,hkd->...d", x, self.kernel) + self.bias


class Dense(nn.Module):
    """Flax `nn.Dense`: kernel `[in, out]` -- the transpose of `nn.Linear`."""

    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.kernel = nn.Parameter(torch.empty(d_in, d_out))
        self.bias = nn.Parameter(torch.zeros(d_out))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """`mlp_kernel` and `head` -> `normal(0, 0.02)`. Measured: 0.0200."""
        nn.init.normal_(self.kernel, mean=0.0, std=KERNEL_STD)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.kernel + self.bias


def _layer_norm(d: int, eps: float) -> nn.LayerNorm:
    ln = nn.LayerNorm(d, eps=eps)
    return ln


def _masked_softmax(logits: Tensor, allowed: Tensor, out_dtype) -> Tensor:
    """Softmax in float32 over `finfo.min`-masked logits, cast back.

    `finfo(float32).min` rather than `-inf`: a row with no allowed key softmaxes
    to a finite uniform distribution instead of NaN, which is what the reference
    relies on for an empty memory (its output is zeroed separately).
    """
    logits = logits.to(torch.float32)
    neg_inf = torch.finfo(torch.float32).min
    logits = torch.where(allowed, logits, torch.full_like(logits, neg_inf))
    return torch.softmax(logits, dim=-1).to(out_dtype)


class SelfAttention(nn.Module):
    """Causal self-attention over the tokens of one sentence."""

    def __init__(self, cfg: TGConfig) -> None:
        super().__init__()
        self.cfg = cfg
        dh = cfg.head_dim
        self.query = DenseGeneralIn(cfg.D, cfg.H, dh)
        self.key = DenseGeneralIn(cfg.D, cfg.H, dh)
        self.value = DenseGeneralIn(cfg.D, cfg.H, dh)
        self.attn_out_proj = DenseGeneralOut(cfg.H, dh, cfg.D)
        self.attn_drop = nn.Dropout(cfg.attn_dropout)

    def forward(self, x: Tensor, key_pad: Tensor) -> Tensor:
        cfg = self.cfg
        q = self.query(x) * cfg.attention_logit_scale
        k, v = self.key(x), self.value(x)
        att = torch.einsum("bqhd,bkhd->bhqk", q, k)
        length = x.shape[1]
        causal = torch.tril(
            torch.ones(1, 1, length, length, dtype=torch.bool, device=x.device)
        )
        allowed = causal & (~key_pad).view(key_pad.shape[0], 1, 1, -1)
        att = self.attn_drop(_masked_softmax(att, allowed, x.dtype))
        out = torch.einsum("bhqk,bkhd->bqhd", att, v)
        return self.attn_out_proj(out)


class CrossAttention(nn.Module):
    """Sentence tokens (queries) -> memory (keys/values). Non-causal."""

    def __init__(self, cfg: TGConfig) -> None:
        super().__init__()
        self.cfg = cfg
        dh = cfg.head_dim
        self.query = DenseGeneralIn(cfg.D, cfg.H, dh)
        self.key = DenseGeneralIn(cfg.D, cfg.H, dh)
        self.value = DenseGeneralIn(cfg.D, cfg.H, dh)
        self.attn_out_proj = DenseGeneralOut(cfg.H, dh, cfg.D)
        self.attn_drop = nn.Dropout(cfg.attn_dropout)
        self.last_attention: Tensor | None = None
        """Set on every forward, for the `r_i` capture (§3.2.1). Not a buffer --
        it carries graph and must not be checkpointed."""

    def forward(self, x: Tensor, mem_kv: Tensor, mem_valid: Tensor) -> Tensor:
        cfg = self.cfg
        keys_src = mem_kv
        if cfg.stm_cross_pos_mode == "sinusoidal":
            pos = memory_positions(mem_valid)
            keys_src = keys_src + sinusoidal_key_pe(
                pos, cfg.D, cfg.stm_positional_weight, dtype=mem_kv.dtype
            )
        q = self.query(x) * cfg.attention_logit_scale
        k = self.key(keys_src)
        # NOTE: values see the RAW memory vectors, without the positional term.
        v = self.value(mem_kv)
        att = torch.einsum("bqhd,bkhd->bhqk", q, k)
        allowed = mem_valid.view(mem_valid.shape[0], 1, 1, -1).expand_as(att)
        att = self.attn_drop(_masked_softmax(att, allowed, x.dtype))
        self.last_attention = att
        out = torch.einsum("bhqk,bkhd->bqhd", att, v)
        out = self.attn_out_proj(out)
        # A row with no memory contributes exactly zero, so a `C` block degenerates
        # to FFN-only rather than adding the uniform-softmax artifact.
        row_has_mem = mem_valid.any(dim=-1).view(-1, 1, 1).to(out.dtype)
        return out * row_has_mem


class Mlp(nn.Module):
    """SwiGLU feed-forward. `Dense_0 -> split -> silu(a)*b -> Dense_1`."""

    def __init__(self, cfg: TGConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.ffn_activation.lower() == "swiglu":
            self.Dense_0 = Dense(cfg.D, 2 * cfg.F)
        else:
            self.Dense_0 = Dense(cfg.D, cfg.F)
        self.Dense_1 = Dense(cfg.F, cfg.D)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = self.Dense_0(x)
        if self.cfg.ffn_activation.lower() == "swiglu":
            a, b = h.chunk(2, dim=-1)
            h = torch.nn.functional.silu(a) * b
        else:
            # Exact (erf) gelu, matching torch's nn.GELU() default, not tanh.
            h = torch.nn.functional.gelu(h, approximate="none")
        h = self.drop(h)
        return self.drop(self.Dense_1(h))


class Block(nn.Module):
    """Pre-LN block, type `S` (self-attention) or `C` (cross-attention).

    There is **no self gate**: the reference freezes it at a constant 1.0 with
    `requires_grad=False`, so it is absent rather than parameterised. `memory_gate`
    is the learnable one, and D-E puts it in `r_i`.
    """

    def __init__(self, cfg: TGConfig, block_type: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.block_type = block_type
        eps = cfg.layer_norm_epsilon
        if block_type == "S":
            self.ln_self = _layer_norm(cfg.D, eps)
            self.self_attn = SelfAttention(cfg)
        else:
            self.ln_mem = _layer_norm(cfg.D, eps)
            self.cross_attn = CrossAttention(cfg)
            self.memory_gate = nn.Parameter(torch.tensor(float(cfg.memory_gate_init)))
        self.ln_ffn = _layer_norm(cfg.D, eps)
        self.mlp = Mlp(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor, key_pad: Tensor, mem_kv: Tensor, mem_valid: Tensor):
        if self.block_type == "S":
            x = x + self.drop(self.self_attn(self.ln_self(x), key_pad))
        else:
            m = self.cross_attn(self.ln_mem(x), mem_kv, mem_valid)
            x = x + self.drop(self.memory_gate * m)
        return x + self.mlp(self.ln_ffn(x))


class SrepHead(nn.Module):
    """`LayerNorm -> dropout -> Dense -> L2 normalize`, with the norm hinge.

    **Not a bare `W_sent`** (`docs/code-vs-paper.md` row 9): the LayerNorm and the
    0.15 dropout in front of the projection are in the code and not in the paper.
    """

    def __init__(self, cfg: TGConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.ln_srep = _layer_norm(cfg.D, cfg.layer_norm_epsilon)
        self.drop = nn.Dropout(cfg.srep_dropout_now)
        self.proj = Dense(cfg.D, cfg.D)

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        cfg = self.cfg
        raw = self.proj(self.drop(self.ln_srep(h)))
        raw_norm = raw.to(torch.float32).norm(dim=-1)
        lower, upper = (
            cfg.srep_norm_target - cfg.srep_norm_margin,
            (cfg.srep_norm_target + cfg.srep_norm_margin),
        )
        penalty = (lower - raw_norm).clamp(min=0) ** 2 + (raw_norm - upper).clamp(
            min=0
        ) ** 2
        denom = raw_norm.clamp(min=1e-12).unsqueeze(-1).to(raw.dtype)
        srep = raw / denom * cfg.srep_norm_target
        return srep, raw_norm, penalty


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #


@dataclass
class StepOutput:
    logits: Tensor  # [B, L, V]
    srep: Tensor  # [B, D]
    srep_raw_norm: Tensor  # [B]
    srep_norm_penalty: Tensor  # [B]
    has_eos: Tensor  # [B] bool
    cross_attention: tuple[Tensor, ...] = ()
    """Per cross-attention layer, `[B, H, L, M]`. **Six entries, not twelve**
    (D-E)."""

    activations: tuple[Tensor, ...] = ()
    """Per block output, `[B, L, D]`. Twelve entries."""


class TGModel(nn.Module):
    """TG over one sentence, conditioned on the memory."""

    def __init__(self, cfg: TGConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.V, cfg.D)
        self.pos_embed = nn.Embedding(cfg.L, cfg.D)
        self.blocks = nn.ModuleList(Block(cfg, cfg.block_config[i]) for i in range(cfg.N))
        self.out_ln = _layer_norm(cfg.D, cfg.layer_norm_epsilon)
        self.srep_head = SrepHead(cfg)
        self.embed_drop = nn.Dropout(cfg.dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """🔴 Initialise every parameter.

        The first version of this class did not, and `nn.Parameter(torch.empty(...))`
        is **uninitialised memory**: a freshly constructed model had `3e36` in one
        attention kernel and zeros in another, differing between runs under a fixed
        `torch.manual_seed`. Every test passed, because `test_fidelity.py` and E0b
        both **load** the reference weights over the top -- so the defect was
        invisible to everything except an actual training run, where it would have
        surfaced as divergence and been blamed on the learning rate.

        Submodules initialise themselves in their own `reset_parameters`; this
        covers what the module owns directly.
        """
        nn.init.normal_(self.embed.weight, mean=0.0, std=KERNEL_STD)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=KERNEL_STD)
        for block in self.blocks:
            if block.block_type == "C":
                nn.init.constant_(block.memory_gate, self.cfg.memory_gate_init)

    def forward(
        self,
        ids: Tensor,
        mask: Tensor,
        mem_kv: Tensor,
        mem_valid: Tensor,
        bos_ctx: Tensor,
        bos_ctx_valid: Tensor,
        *,
        capture: bool = False,
    ) -> StepOutput:
        cfg = self.cfg
        tok = self.embed(ids)

        if cfg.bos_replacement_mode == "copy":
            # The previous sentence's gestalt replaces this sentence's [BOS]
            # embedding, BEFORE positional embeddings are added.
            bos_slot = torch.where(
                bos_ctx_valid.unsqueeze(-1), bos_ctx.to(tok.dtype), tok[:, 0, :]
            )
            tok = torch.cat([bos_slot.unsqueeze(1), tok[:, 1:, :]], dim=1)

        pos = self.pos_embed(torch.arange(ids.shape[1], device=ids.device)).unsqueeze(0)
        h = self.embed_drop(tok + pos)
        key_pad = mask == 0

        h_srep = None
        activations = []
        cross = []
        for i, block in enumerate(self.blocks):
            h = block(h, key_pad, mem_kv, mem_valid)
            if capture:
                activations.append(h)
                if block.block_type == "C":
                    cross.append(block.cross_attn.last_attention)
            if i == cfg.srep_layer_idx:
                h_srep = h

        h = self.out_ln(h)
        if h_srep is None:
            h_srep = h
        # Tied readout: the reference uses `embed.attend`, in float32.
        logits = h.to(torch.float32) @ self.embed.weight.to(torch.float32).T

        # The gestalt is read at the first [EOS]; argmax returns 0 when absent,
        # which is the reference's own fallback.
        hit = ids == cfg.eos_id
        has_eos = hit.any(dim=-1)
        position = hit.to(torch.int32).argmax(dim=-1)
        h_tok = h_srep.gather(
            1, position.view(-1, 1, 1).expand(-1, 1, h_srep.shape[-1])
        ).squeeze(1)
        srep, raw_norm, penalty = self.srep_head(h_tok)
        return StepOutput(
            logits=logits,
            srep=srep,
            srep_raw_norm=raw_norm,
            srep_norm_penalty=penalty,
            has_eos=has_eos,
            cross_attention=tuple(cross),
            activations=tuple(activations),
        )


def run_sentence_loop(
    model: TGModel,
    sentences: Tensor,  # [B, T, L]
    masks: Tensor,  # [B, T, L]
    lengths: Tensor,  # [B]
    *,
    step_fn,
    capture: bool = False,
):
    """Run every sentence of a batch of documents, threading the memory.

    🔴 **No `detach` on the gestalt -> memory path** unless
    `cfg.detach_sreps_for_memory` is set, which is the reference's ablation and is
    the positive control `tests/test_fidelity.py` uses. Sentence `t+1`'s loss must
    reach sentence `t` through this tensor.
    """
    cfg = model.cfg
    batch, steps, _ = sentences.shape
    device = sentences.device
    mem = init_memory(batch, cfg, device=device, dtype=model.embed.weight.dtype)
    bos_ctx = torch.zeros(batch, cfg.D, device=device, dtype=model.embed.weight.dtype)
    bos_valid = torch.zeros(batch, dtype=torch.bool, device=device)
    acc = None
    captured: list[StepOutput] = []

    for t in range(steps):
        ids_t, mask_t = sentences[:, t], masks[:, t]
        row_valid = torch.full((batch,), t, device=device) < lengths
        out = model(ids_t, mask_t, mem.kv, mem.valid, bos_ctx, bos_valid, capture=capture)
        if capture:
            captured.append(out)
        contrib = step_fn(t, out, ids_t, mask_t, row_valid)
        acc = contrib if acc is None else acc + contrib

        srep_mem = out.srep
        if cfg.detach_sreps_for_memory:
            srep_mem = srep_mem.detach()
        write = row_valid & out.has_eos
        if cfg.use_memory:
            mem = push_memory(
                mem, srep_mem, write, step=t, backprop_window=cfg.stm_backprop_window
            )
        if cfg.bos_replacement_mode == "copy":
            bos_ctx = (
                out.srep.detach()
                if (cfg.bos_context_detach or cfg.detach_sreps_for_memory)
                else out.srep
            )
            bos_valid = write & (torch.full((batch,), t + 1, device=device) < lengths)
    return (acc, captured) if capture else acc
