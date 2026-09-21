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
from collections.abc import Callable

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
    model: TGModel,
    ids: Tensor,
    mask: Tensor,
    *,
    perm: Tensor | None = None,
    token_masks: dict[str, Tensor] | None = None,
    replace: Callable[[Tensor, Tensor, int], Tensor] | None = None,
) -> dict:
    """Real-token loss with each row's own memory, vs. another row's.

    Two passes over one batch with one model, in `eval()` and under `no_grad`.
    Pass A threads the batch through the memory loop honestly -- FIFO, with the
    write and `bos_ctx` rules of `run_policy_loop` (`policy_loop.py`) -- and
    snapshots the memory at every step; pass B replays
    those exact snapshots **row-permuted by `perm`**. Memory contents are
    identical between the passes -- only whose they are changes.

    `ids`, `mask`: `[B, S, L]`, as `rsr.train.loop.encode` returns them.

    Additive arguments (decisive run, `experiments/decisive-shuffle/PREREG.md`
    *Secondary readouts*). Both default to `None`, which is the behaviour above
    byte for byte and adds no key to the result:

    * `token_masks` -- `{name: [B, S, L] bool}` supervision masks in `encode()`'s
      token frame (e.g. `rsr.train.loop.answer_targets`). Each is scored the way
      `mask` is, `[..., 1:]` against the shifted targets, and the result gains
      `by_mask[name]` with the same statistics restricted to it. A mask that
      selects a non-real target raises: it would be scoring padding.
    * `replace(kv, valid, t) -> kv` -- applied in pass B to the (row-permuted)
      snapshot before the forward. It is how `random_replacement` and
      `cross_row_cosine` reuse this one loop instead of a second copy of it.
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
    sel: dict[str, list[Tensor]] = {}
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
                if token_masks is not None:
                    for name, tm in token_masks.items():
                        sel_full = tm[:, t, 1:].reshape(-1).to(device)
                        if bool((sel_full & ~real).any()):
                            raise ValueError(
                                f"token mask {name!r} selects a non-real target at "
                                f"step {t}: it would score padding"
                            )
                        sel.setdefault(name, []).append(sel_full[real])
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
                if replace is None:
                    out = model(ids_t, mask_t, kv[perm], valid[perm], bc, bv)
                else:
                    kv_b = replace(kv[perm], valid[perm], t)
                    out = model(ids_t, mask_t, kv_b, valid[perm], bc, bv)
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
    extra: dict = {}
    if token_masks is not None:
        extra["by_mask"] = {}
        for name in token_masks:
            m_ = torch.cat(sel[name]) if sel.get(name) else d.new_zeros(0).bool()
            dm, hsel, ssel = d[m_], h[m_], s[m_]
            n_ = int(dm.numel())
            extra["by_mask"][name] = {
                "n_targets": n_,
                "loss_honest_memory": float(hsel.mean()) if n_ else None,
                "loss_replaced_memory": float(ssel.mean()) if n_ else None,
                "delta_nats_per_token": float(dm.mean()) if n_ else None,
                "mean_abs_token_delta": float(dm.abs().mean()) if n_ else None,
                "delta_exactly_zero": bool((dm == 0).all()) if n_ else None,
                "n_tokens_moved": int((dm != 0).sum()),
            }
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
        **extra,
    }


def random_replacement(
    model: TGModel,
    ids: Tensor,
    mask: Tensor,
    *,
    generator: torch.Generator,
    self_replace: bool = False,
    token_masks: dict[str, Tensor] | None = None,
) -> dict:
    """Matched-norm random replacement (decisive PREREG, *Secondary 2*).

    Pass B replays each row's **own** position in the stream but with every
    memory slot replaced by an i.i.d. Gaussian vector rescaled to that slot's L2
    norm, per slot and per step. Draws come from `generator`, which the caller
    seeds independently of the model, so the model's RNG stream is untouched.

    It separates the two rivals the shuffle cannot: a readout that ignores memory
    reads ~0 here, while a readout that uses memory whose contents are the same in
    every row reads ~0 under the shuffle but not here.

    `self_replace=True` is the function's own control: every slot is "replaced"
    by itself, so the reading **must be exactly 0.0** -- otherwise the delta comes
    from the replacement path, not from the memory's contents.
    """

    def _rand(kv: Tensor, valid: Tensor, t: int) -> Tensor:
        if self_replace:
            return kv.clone()
        r = torch.randn(
            kv.shape, generator=generator, dtype=kv.dtype, device=generator.device
        ).to(kv.device)
        norm_r = r.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(kv.dtype).tiny)
        return r / norm_r * kv.norm(dim=-1, keepdim=True)

    B = ids.shape[0]
    out = shuffle_control(
        model,
        ids,
        mask,
        perm=torch.arange(B, device=ids.device),
        token_masks=token_masks,
        replace=_rand,
    )
    out["replacement"] = "self" if self_replace else "matched_norm_gaussian"
    return out


def cross_row_cosine(model: TGModel, ids: Tensor, mask: Tensor) -> dict:
    """Mean off-diagonal cosine similarity between rows' memory vectors at matched
    slot and step (decisive PREREG, *Secondary 2*). Descriptive only.

    Reads the honest pass's snapshots through `shuffle_control`'s own loop (the
    `replace` hook records them and returns them unchanged). Only `(t, slot)`
    pairs valid in **every** row enter; each contributes the mean of its
    `B * (B - 1)` off-diagonal cosines, and the result is their mean.
    """
    recorded: list[tuple[Tensor, Tensor]] = []

    def _record(kv: Tensor, valid: Tensor, t: int) -> Tensor:
        recorded.append((kv.clone(), valid.clone()))
        return kv

    B = ids.shape[0]
    if B < 2:
        raise ValueError(f"cross-row cosine needs at least 2 rows, got {B}")
    shuffle_control(
        model, ids, mask, perm=torch.arange(B, device=ids.device), replace=_record
    )
    vals: list[float] = []
    off = ~torch.eye(B, dtype=torch.bool, device=ids.device)
    for kv, valid in recorded:
        for j in range(kv.shape[1]):
            if not bool(valid[:, j].all()):
                continue
            x = torch.nn.functional.normalize(kv[:, j].double(), dim=-1)
            c = x @ x.T
            vals.append(float(c[off].mean()))
    return {
        "mean_offdiag_cosine": sum(vals) / len(vals) if vals else None,
        "min_pair_mean": min(vals) if vals else None,
        "max_pair_mean": max(vals) if vals else None,
        "n_slot_steps": len(vals),
    }
