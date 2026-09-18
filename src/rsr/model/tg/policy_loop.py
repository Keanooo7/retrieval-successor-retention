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

import torch
from torch import Tensor

from rsr.model.tg.model import Memory, StepOutput, TGModel, init_memory
from rsr.retention.policy import MemoryState

__all__ = ["memory_state", "run_policy_loop", "write_at"]


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


def run_policy_loop(
    model: TGModel,
    sentences: Tensor,
    masks: Tensor,
    lengths: Tensor,
    policy,
    *,
    step_fn,
    capture: bool = False,
):
    """`run_sentence_loop`, with the eviction rule supplied by `policy`.

    `policy.select_eviction(slots, context, step)` is consulted **only when the
    row's memory is full**, which is the same condition under which the
    transcription's `push_memory` rolls.
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

    for t in range(steps):
        ids_t, mask_t = sentences[:, t], masks[:, t]
        row_valid = torch.full((batch,), t, device=device) < lengths
        out = model(ids_t, mask_t, mem.kv, mem.valid, bos_ctx, bos_valid, capture=capture)
        if capture:
            captured.append(out)
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
