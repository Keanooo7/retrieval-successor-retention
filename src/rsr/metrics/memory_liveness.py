"""The shuffle control -- does a row's own memory matter to its loss? (S0-04)

Hand every row **another document's entire memory** and re-read the real-token
loss. If the loss does not move, the model is not using what its memory holds,
whatever the attention weights look like. It is the instrument behind
`docs/RESEARCH-CONTEXT.md` §10.3, which until this module existed in prose only:
the 2026-09-18 audit ran it from a scratch script that was never committed.

Lifted from `experiments/cycle-01-masked-loss/run.py::shuffle_control` and
extended in three ways that cycle's review asked for:

1. **Per-token deltas, not only per-step means.** §10.3's claim is that *no single
   token moves*, which a mean over a step cannot show.
2. **The facts that rule out the known dead-memory trap** travel with the delta:
   how many sentences carry an EOS (a sentence without one is never written),
   how full the memory is, and the trained `memory_gate` scalars.
3. **Two controls, because a null from an instrument never seen to read non-null
   is not evidence** (the manager's rejection of cycle 1, and S0-04 Bar 1-2):

   * `memory_disabled` -- the same model with every `memory_gate` zeroed. The
     cross-attention output is multiplied by 0, so the delta **must be exactly
     0.0**. It is the reading the instrument gives when memory is absent.
   * a **live** memory -- the caller supplies it. `tests/test_shuffle_control.py`
     uses a randomly initialised model, whose memory genuinely differs row to row
     and genuinely reaches the logits, so the delta must be **non-zero**. An
     instrument that silently no-ops -- a permutation never applied, a guard that
     never fires, an eval path that skips memory -- fails there.

`bos_ctx` is left honest on purpose, as in cycle 1: `bos_replacement_mode ==
"copy"` puts the previous sentence's gestalt at token 0 *outside* the
`use_memory` guard, so disturbing it would measure that path instead of
cross-attention (§10.3's corrected ablation).
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor

from rsr.model.tg import TGModel
from rsr.model.tg.model import init_memory
from rsr.model.tg.policy_loop import write_at
from rsr.train.loop import lm_token_losses


def derangement(n: int, *, device=None) -> Tensor:
    """Row `i` receives row `i - 1`'s memory. Roll-by-1 has no fixed point for
    `n >= 2`, so every row gets **another** document's memory, never its own."""
    if n < 2:
        raise ValueError(f"a derangement needs at least 2 rows, got {n}")
    return torch.roll(torch.arange(n, device=device), 1)


def memory_gates(model: TGModel) -> list[float]:
    """The learnable `memory_gate` scalar of every cross-attention block."""
    return [float(b.memory_gate.detach()) for b in model.blocks if b.block_type == "C"]


def with_memory_disabled(model: TGModel) -> TGModel:
    """A copy of `model` whose cross-attention contributes nothing: every
    `memory_gate` is zeroed. The original is not touched."""
    off = copy.deepcopy(model)
    with torch.no_grad():
        for b in off.blocks:
            if b.block_type == "C":
                b.memory_gate.zero_()
    return off


def shuffle_control(
    model: TGModel, ids: Tensor, mask: Tensor, *, perm: Tensor | None = None
) -> dict:
    """Real-token loss with each row's own memory, vs. another row's.

    Two passes over one batch with one model, in `eval()` and under `no_grad`.
    Pass A threads the batch through the memory loop honestly -- FIFO, with the
    write and `bos_ctx` rules of `run_policy_loop` (`policy_loop.py`) -- and
    snapshots the memory at every step; pass B replays
    those exact snapshots **row-permuted by `perm`**. Memory contents are
    identical between the passes -- only whose they are changes.

    `ids`, `mask`: `[B, S, L]`, as `rsr.train.loop.encode` returns them.
    """
    cfg = model.cfg
    was_training = model.training
    model.eval()
    device = ids.device
    B, S, _ = ids.shape
    perm = derangement(B, device=device) if perm is None else perm.to(device)
    dtype = model.embed.weight.dtype

    snaps, bos_snaps = [], []
    honest_tok, shuffled_tok = [], []
    honest_step, shuffled_step = [], []
    n_eos = n_valid_sentences = 0
    try:
        with torch.no_grad():
            mem = init_memory(B, cfg, device=device, dtype=dtype)
            bos_ctx = torch.zeros(B, cfg.D, device=device, dtype=dtype)
            bos_valid = torch.zeros(B, dtype=torch.bool, device=device)
            # As `rsr.train.loop.train` calls `run_policy_loop`: every row runs
            # the full stream, so `row_valid` is `t < S` for all of them.
            lengths = torch.full((B,), S, device=device)
            for t in range(S):
                ids_t, mask_t = ids[:, t], mask[:, t]
                row_valid = torch.full((B,), t, device=device) < lengths
                snaps.append((mem.kv.clone(), mem.valid.clone()))
                bos_snaps.append((bos_ctx.clone(), bos_valid.clone()))
                out = model(ids_t, mask_t, mem.kv, mem.valid, bos_ctx, bos_valid)
                real = mask_t[:, 1:].reshape(-1)
                per = lm_token_losses(out.logits, ids_t)
                honest_tok.append(per[real])
                honest_step.append(per[real].mean() if real.any() else per.new_zeros(()))
                is_sentence = row_valid & mask_t.any(dim=-1)
                n_eos += int((is_sentence & out.has_eos).sum())
                n_valid_sentences += int(is_sentence.sum())
                write = row_valid & out.has_eos
                victim = torch.zeros(B, dtype=torch.long, device=device)  # FIFO
                if cfg.use_memory:
                    mem = write_at(mem, out.srep, write, victim, t)
                if cfg.bos_replacement_mode == "copy":
                    bos_ctx = out.srep
                    bos_valid = write & (torch.full((B,), t + 1, device=device) < lengths)

            for t in range(S):
                ids_t, mask_t = ids[:, t], mask[:, t]
                kv, valid = snaps[t]
                bc, bv = bos_snaps[t]
                out = model(ids_t, mask_t, kv[perm], valid[perm], bc, bv)
                real = mask_t[:, 1:].reshape(-1)
                per = lm_token_losses(out.logits, ids_t)
                shuffled_tok.append(per[real])
                shuffled_step.append(
                    per[real].mean() if real.any() else per.new_zeros(())
                )
    finally:
        model.train(was_training)

    h = torch.cat(honest_tok).double()
    s = torch.cat(shuffled_tok).double()
    d = s - h
    step_d = [float(y - x) for x, y in zip(honest_step, shuffled_step, strict=True)]
    hm, sm = float(h.mean()), float(s.mean())
    return {
        "loss_real_tokens_honest_memory": hm,
        "loss_real_tokens_shuffled_memory": sm,
        "delta_nats_per_token": sm - hm,
        "delta_relative": (sm - hm) / hm if hm else None,
        "delta_exactly_zero": sm == hm,
        # Mean |per-token delta|: a swapped memory pushes tokens both ways, and
        # the signed mean cancels them -- on a live random-init memory the signed
        # mean moves ~5e-5 relative while single tokens move ~0.1 nats.
        "mean_abs_token_delta": float(d.abs().mean()),
        "max_token_delta": float(d.max()),
        "min_token_delta": float(d.min()),
        "n_tokens_moved": int((d != 0).sum()),
        "n_real_tokens": int(d.numel()),
        "max_step_abs_delta": max(abs(x) for x in step_d),
        "n_sentences": n_valid_sentences,
        "n_sentences_with_eos": n_eos,
        "final_slots_filled_per_row": [int(v) for v in mem.valid.sum(dim=1)],
        "memory_slots": cfg.M,
        "memory_gates": memory_gates(model),
        "permutation": [int(p) for p in perm],
    }
