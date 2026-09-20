"""The sentence loop with a pluggable retention policy (spec §3.7, gauntlet 2.6).

`rsr.model.tg.model.run_sentence_loop` is the **transcription** and evicts the way
the reference does: `push_memory` rolls, so the oldest slot dies. That is stock TG
and it must stay that way -- `tests/test_fidelity.py` compares against it.

This module is the RSR loop. It is identical except for **one line**: which slot is
overwritten. Section 3.1: *"the entire contribution of this document is replacing
'the oldest entry is removed.'"*

## Why the two loops share `write_at` rather than diverging

E0b's claim is that **E3's FIFO and RSR arms differ only in the eviction rule**.
That is a claim about code paths, not about numbers, and the strongest form of it
is that there is only one code path. So the policy chooses an index and
`write_at` does the same thing with it either way:

* the memory stays **oldest-first in a prefix** (`push_memory`'s invariant, and
  what makes `P^(sent)` rank-indexed -- ADR-0006);
* evicting the oldest slot and re-packing is **exactly `push_memory`'s roll**, so
  the FIFO path through here is bit-identical to the transcription's;
* evicting a middle slot compacts the prefix, which is precisely the rank
  displacement ADR-0006 instruments.

🔴 **No `detach` on the gestalt -> memory path.** Sentence `t+1`'s loss must reach
sentence `t` through it, and severing it is invisible in the forward pass --
measured on the reference: identical loss, 113 of 206 gradient arrays moved.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rsr.model.tg.model import Memory, StepOutput, TGModel, init_memory
from rsr.retention.policy import AttentionTrace, MemoryState

__all__ = [
    "Q_TOK_COLLAPSE",
    "CrossCapture",
    "cross_capture",
    "memory_state",
    "run_policy_loop",
    "trace_for_row",
    "write_at",
]

Q_TOK_COLLAPSE = "sum_over_real_query_tokens"
"""How `alpha`'s `[B, H, Q_tok, M]` becomes `AttentionTrace.alpha`'s `[L, H, M]`.

🔴 **ADR-0008 is the decision, and it is a decision, not an implementation
detail.** Read it before changing this string. The one-line version: `W_O v_i` does
not depend on the query position, so the cross-attention increment slot `i` makes
to the sentence's residual stream, summed over the sentence, is exactly
`(sum_q alpha[h,q,i]) . W_O v[h,i]` -- the sum is algebra, not a choice. PAD query
positions are excluded because their residual stream is discarded.

**Mean is the same decision.** `mean = sum / Q_real` with one `Q_real` for every
`(l, h, i)`, so it cancels in section 3.2.1's `share_i = raw_i / sum_j raw_j` and
changes only the *unnormalised* `contribution()` diagnostic. **EOS-only does not
cancel** and is the real alternative; ADR-0008 says what would distinguish them.
"""


def write_at(
    mem: Memory, srep: Tensor, write: Tensor, victim: Tensor, step: int
) -> Memory:
    """Insert `srep`, evicting slot `victim` in rows that are full.

    Keeps the oldest-first prefix by shifting the slots **after** the victim left
    and writing the newcomer at the end. For `victim = 0` that is exactly
    `push_memory`'s roll, which is what makes the §3.7 reduction bit-exact rather
    than merely close.
    """
    kv, valid = mem.kv, mem.valid
    batch, m, _ = kv.shape
    device = kv.device

    k = valid.to(torch.int64).sum(dim=-1)  # occupied count
    full = k >= m
    # Where not full the newcomer appends at slot k and nothing is evicted; where
    # full it takes the last slot after everything behind the victim shifts left.
    victim = torch.where(full, victim, k)

    idx = torch.arange(m, device=device).unsqueeze(0).expand(batch, m)
    # Gather source index: slots before the victim stay, slots at/after it pull
    # from one position later. The final slot is overwritten by the newcomer.
    src = torch.where(idx < victim.unsqueeze(-1), idx, (idx + 1).clamp(max=m - 1))
    kv_shift = kv.gather(1, src.unsqueeze(-1).expand(batch, m, kv.shape[-1]))
    valid_shift = valid.gather(1, src)
    step_shift = mem.step.gather(1, src)

    # Rows that are not full do not shift at all.
    kv_base = torch.where(full.view(-1, 1, 1), kv_shift, kv)
    valid_base = torch.where(full.unsqueeze(-1), valid_shift, valid)
    step_base = torch.where(full.unsqueeze(-1), step_shift, mem.step)

    slot = torch.where(full, torch.full_like(k, m - 1), k)
    onehot = torch.nn.functional.one_hot(slot, m).to(kv.dtype)
    kv_new = kv_base * (1.0 - onehot.unsqueeze(-1)) + srep.unsqueeze(
        1
    ) * onehot.unsqueeze(-1)
    is_slot = onehot > 0
    valid_new = valid_base | is_slot
    step_new = torch.where(is_slot, torch.full_like(mem.step, step), step_base)

    write_m = write.unsqueeze(-1).expand_as(valid)
    return Memory(
        kv=torch.where(write_m.unsqueeze(-1), kv_new, kv),
        valid=torch.where(write_m, valid_new, valid),
        step=torch.where(write_m, step_new, mem.step),
    )


def memory_state(mem: Memory, step: int, row: int = 0) -> MemoryState:
    """One row of the model's `Memory` as the policy protocol's `MemoryState`.

    The policy sees **detached** gestalts: §3.3 permits gradient to `phi` only, and
    a policy that cannot reach the transformer cannot violate that by accident.
    """
    return MemoryState(
        gestalts=mem.kv[row].detach(),
        written_at=mem.step[row].clone(),
        live=mem.valid[row].clone(),
        step=step,
    )


@dataclass
class CrossCapture:
    """One step of cross-attention for the **whole batch**, ready to slice per row.

    Built per step because `AttentionTrace` is per row (`live` is `[M]`), and the
    expensive part -- `W_O v` -- is row-shared, so it is computed once here rather
    than `B` times in `trace_for_row`.

    🔴 **Everything here is detached.** §3.3 permits gradient to `phi` only, and
    `r_i` is a *target*, not a differentiable path. `memory_state` detaches the
    gestalts for the same reason.
    """

    alpha: Tensor
    """`[L, B, H, M]`, collapsed over `Q_tok` per `Q_TOK_COLLAPSE` and zeroed on
    dead slots."""

    wo_v: Tensor
    """`[L, B, H, M, D]` -- `W_O^(l,h) v_{l,h,i}`, the value vector after the head's
    output projection.

    **The projection's bias is excluded.** §3.2.1 is `W_O v`, a matrix applied to a
    vector; `attn_out_proj.bias` is one vector added once per query position,
    shared by every head and every slot, so attributing it to a slot would credit
    every slot equally with something no slot caused.
    """

    gate: Tensor
    """`[L]` TG's `g_mem` per cross-attention layer (correction 17 / D-E)."""

    eval_mode: bool
    """Whether `alpha` was produced with attention dropout off (D-F)."""

    n_real_query_tokens: Tensor
    """`[B]` how many query positions the collapse summed over. Reported, not used:
    it is the scalar that makes sum and mean the same decision (ADR-0008), and a
    row where it is 0 is a row whose `alpha` means nothing."""


def cross_capture(
    model: TGModel,
    out: StepOutput,
    mem_kv: Tensor,
    mask: Tensor,
    mem_valid: Tensor,
    *,
    collapse: str = Q_TOK_COLLAPSE,
) -> CrossCapture:
    """The §3.2.1 capture bridge: a real forward pass -> `AttentionTrace` inputs.

    `out` must come from `model(..., capture=True)`; `mem_kv` and `mem_valid` must
    be the memory the forward pass **attended over**, i.e. before that step's
    write. Passing the post-write memory silently measures the wrong step.

    Three things this function exists to produce, none of which the model surfaced
    before:

    * `alpha` as `[L, H, M]` -- see `Q_TOK_COLLAPSE` and ADR-0008;
    * `wo_v`, which **never exists as a tensor in the forward pass**: `att @ v` is
      contracted over slots *before* `attn_out_proj` is applied, so the per-slot
      per-head projected value has to be recomputed here;
    * `gate`, TG's `g_mem`, which is an `nn.Parameter` on the block and was not
      surfaced anywhere.

    🔴 **It recomputes `v` rather than having the forward pass stash it.** The
    forward pass must not change (`test_fidelity.py`), and a tensor stashed on
    every forward is not free when capture is off. `self.value` is linear and
    dropout-free, so the recompute is exact.
    """
    if collapse != Q_TOK_COLLAPSE:
        raise ValueError(
            f"collapse={collapse!r}: only {Q_TOK_COLLAPSE!r} is implemented. "
            f"ADR-0008 records the decision and names `eos_only` as the live "
            f"alternative; implementing it is a config branch plus an E0d row, "
            f"not an edit here."
        )
    if not out.cross_attention:
        raise ValueError(
            "StepOutput.cross_attention is empty: the forward pass ran with "
            "capture=False, so nothing was collected. `r_i` cannot be recovered "
            "after the fact -- re-run the step with capture=True."
        )

    blocks = [b for b in model.blocks if b.block_type == "C"]
    if len(blocks) != len(out.cross_attention):
        raise ValueError(
            f"{len(blocks)} cross-attention blocks but "
            f"{len(out.cross_attention)} captured attention tensors. D-E: the "
            f"profile has one row per C block and the count is the check."
        )

    q_real = (mask != 0).to(out.cross_attention[0].dtype)  # [B, Q]
    live = mem_valid.unsqueeze(1)  # [B, 1, M]

    alphas, wo_vs, gates = [], [], []
    # 🔴 `no_grad`, not `.detach()` on the way out. `self.value` reads a parameter
    # that requires grad, so detaching only its *input* leaves `wo_v` attached to
    # the transformer -- which is a path from the retention target into `W_sent`
    # and `self.value`, and §3.3 permits gradient to `phi` alone. The first draft
    # of this function detached `mem_kv` and the kernel and still leaked.
    with torch.no_grad():
        for block, att in zip(blocks, out.cross_attention, strict=True):
            # att: [B, H, Q, M] -> [B, H, M]. Sum over REAL query positions only.
            a = torch.einsum("bhqm,bq->bhm", att, q_real)
            # A dead slot's attention is masked to ~0 already, except on an
            # all-masked row, where `_masked_softmax` returns a finite uniform
            # distribution on purpose. Zero it so `contribution()` -- which does
            # not see `live` -- is not reading a softmax artefact as retrieval.
            alphas.append(torch.where(live, a, torch.zeros_like(a)))

            v = block.cross_attn.value(mem_kv)  # [B, M, H, Dh]
            wo = block.cross_attn.attn_out_proj.kernel  # [H, Dh, D]
            # 🔴 `W_O` applied PER HEAD and PER SLOT. This is the tensor D-6 is
            # about, and it is the one the forward pass destroys by contracting
            # over slots first. Dropping it here is shape-compatible with
            # `reward.contribution` (the norm is over the last axis either way),
            # so it fails silently -- which is why `scripts/mutation_battery.py`
            # carries a mutation for it.
            wo_vs.append(torch.einsum("bmhk,hkd->bhmd", v, wo))

            gates.append(block.memory_gate.reshape(()))

        return CrossCapture(
            alpha=torch.stack(alphas),
            wo_v=torch.stack(wo_vs),
            gate=torch.stack(gates),
            eval_mode=(not model.training) or model.cfg.attn_dropout == 0.0,
            n_real_query_tokens=q_real.sum(dim=-1),
        )


def trace_for_row(
    cap: CrossCapture, mem_valid: Tensor, row: int, step: int
) -> AttentionTrace:
    """One row of a `CrossCapture` as the protocol's `AttentionTrace`.

    `mem_valid` is the **pre-write** validity mask, matching `cap`.
    """
    return AttentionTrace(
        alpha=cap.alpha[:, row],
        wo_v=cap.wo_v[:, row],
        live=mem_valid[row].clone(),
        step=step,
        eval_mode=cap.eval_mode,
        gate=cap.gate,
    )


def run_policy_loop(
    model: TGModel,
    sentences: Tensor,
    masks: Tensor,
    lengths: Tensor,
    policy,
    *,
    step_fn,
    capture: bool = False,
    observe: bool = False,
):
    """`run_sentence_loop`, with the eviction rule supplied by `policy`.

    `policy.select_eviction(slots, context, step)` is consulted **only when the
    row's memory is full**, which is the same condition under which the
    transcription's `push_memory` rolls.

    `observe=True` turns on the §3.2.1 capture bridge: each step builds an
    `AttentionTrace` per live row from the forward pass that just ran and hands it
    to `policy.observe`. **Off by default and free when off** -- with `observe` and
    `capture` both false the model is called with `capture=False` and not one extra
    tensor is formed.

    🔴 **`observe` runs BEFORE the eviction and the write**, because the attention
    it describes was paid over the pre-write memory. Moving it below `write_at`
    hands the policy a trace whose `live` mask belongs to the next step.

    `policy.reset()` is called once on entry. This call **is** a stream: the memory
    is initialised fresh at the top, and [P2] fact 2 resets memory at each stream
    boundary -- §3.5's `b` resets with it, which is the whole of defect D-1's
    mechanism. `src/rsr/train/loop.py` never reset between streams.
    """
    cfg = model.cfg
    batch, steps, _ = sentences.shape
    device = sentences.device
    dtype = model.embed.weight.dtype
    mem = init_memory(batch, cfg, device=device, dtype=dtype)
    bos_ctx = torch.zeros(batch, cfg.D, device=device, dtype=dtype)
    bos_valid = torch.zeros(batch, dtype=torch.bool, device=device)
    acc = None
    captured: list[StepOutput] = []
    # A policy that owns parameters is device-bound; the memory's device wins.
    if hasattr(policy, "to"):
        policy.to(device)
    # [P2] fact 2: the memory is reset at each stream boundary, and this call is a
    # stream. Anything the policy keeps per stream -- LRU's `last_used`, H2O's
    # accumulated attention, §3.5's `b` -- dies here rather than leaking forward.
    policy.reset()

    for t in range(steps):
        ids_t, mask_t = sentences[:, t], masks[:, t]
        row_valid = torch.full((batch,), t, device=device) < lengths
        out = model(
            ids_t,
            mask_t,
            mem.kv,
            mem.valid,
            bos_ctx,
            bos_valid,
            capture=capture or observe,
        )
        if capture:
            captured.append(out)
        if observe:
            # Pre-write memory, deliberately: this is what the forward attended to.
            cap = cross_capture(model, out, mem.kv, mask_t, mem.valid)
            for row in range(batch):
                if bool(row_valid[row]):
                    policy.observe(
                        memory_state(mem, t, row),
                        trace_for_row(cap, mem.valid, row, t),
                        t,
                    )
        contrib = step_fn(t, out, ids_t, mask_t, row_valid)
        acc = contrib if acc is None else acc + contrib

        srep_mem = out.srep.detach() if cfg.detach_sreps_for_memory else out.srep
        write = row_valid & out.has_eos

        victims = []
        for row in range(batch):
            if bool(mem.valid[row].all()):
                slots = memory_state(mem, t, row)
                victims.append(policy.select_eviction(slots, out.srep[row].detach(), t))
            else:
                victims.append(0)  # unused: the row is not full
        victim = torch.tensor(victims, device=device, dtype=torch.long)

        if cfg.use_memory:
            mem = write_at(mem, srep_mem, write, victim, t)
            for row in range(batch):
                if bool(write[row]):
                    policy.on_write(memory_state(mem, t, row), int(victim[row]), t)

        if cfg.bos_replacement_mode == "copy":
            bos_ctx = (
                out.srep.detach()
                if (cfg.bos_context_detach or cfg.detach_sreps_for_memory)
                else out.srep
            )
            bos_valid = write & (torch.full((batch,), t + 1, device=device) < lengths)

    return (acc, captured) if capture else acc
