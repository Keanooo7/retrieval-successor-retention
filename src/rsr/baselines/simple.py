"""The floor policies: FIFO, LRU, random.

§5.4 calls FIFO/LRU/random a **floor, not a bar** -- the bar is H2O and Expire-Span. They are here
because they are cheap, because §3.7's reduction target *is* FIFO, and because **LRU is the recency
policy**: failing to beat it is §7.1 vacuity stated behaviourally.
"""

from __future__ import annotations

import torch
from torch import Tensor

from rsr.retention.policy import AttentionTrace, MemoryState

__all__ = ["FIFOPolicy", "LRUPolicy", "RandomPolicy"]


class FIFOPolicy:
    """Evict the oldest slot. This is stock TG ([P2]: "the oldest entry is removed").

    §3.7's reduction must reproduce this **by configuration**, and E0b asserts the resulting loss
    curve is bit-identical. If RSR-under-reduction and this policy ever disagree on a single
    eviction, the reduction is broken and E0b is measuring nothing.
    """

    name = "fifo"

    def select_eviction(self, slots: MemoryState, context: Tensor, step: int) -> int:
        return int(torch.argmax(slots.ages))

    def observe(self, slots: MemoryState, attn: AttentionTrace, step: int) -> None:
        return None

    def reset(self) -> None:
        return None


class LRUPolicy:
    """Evict the slot least recently *attended to*.

    Distinct from FIFO: FIFO uses time since write, LRU uses time since use. On E7 this matters --
    §10.1 rules FIFO out as a control there because under FIFO survival time is the deterministic
    function ``min(M, S-i)`` of serial position, so the partial correlation zeroes it out by
    construction. LRU does not have that degeneracy, which is why it is a control and FIFO is not.
    """

    name = "lru"

    def __init__(self) -> None:
        self._last_used: dict[int, int] = {}

    def select_eviction(self, slots: MemoryState, context: Tensor, step: int) -> int:
        ids = slots.slot_ids.tolist()
        # A slot never attended to is maximally stale; fall back to its age.
        staleness = [
            step - self._last_used.get(sid, -int(slots.ages[i])) for i, sid in enumerate(ids)
        ]
        return int(max(range(len(staleness)), key=lambda i: staleness[i]))

    def observe(self, slots: MemoryState, attn: AttentionTrace, step: int) -> None:
        # "Used" = carried non-trivial norm-weighted contribution this step, not merely attended.
        # Attention weight alone is not evidence of use ([P14], attention sinks).
        per_slot = attn.contributions.sum(dim=(0, 1))
        if per_slot.numel() == 0:
            return
        top = int(torch.argmax(per_slot))
        self._last_used[int(slots.slot_ids[top])] = step

    def reset(self) -> None:
        self._last_used.clear()


class RandomPolicy:
    """Sanity floor. Carries its own generator so it cannot perturb the model's RNG stream.

    Same discipline as §3.7's warning about constructing the value head: consuming draws from the
    shared stream shifts data order and dropout masks, and then a comparison fails for a reason
    unrelated to the mechanism.
    """

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self._gen = torch.Generator().manual_seed(seed)

    def select_eviction(self, slots: MemoryState, context: Tensor, step: int) -> int:
        return int(torch.randint(0, slots.n_live, (1,), generator=self._gen))

    def observe(self, slots: MemoryState, attn: AttentionTrace, step: int) -> None:
        return None

    def reset(self) -> None:
        return None
