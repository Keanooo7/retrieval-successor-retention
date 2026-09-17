"""LRU -- the recency policy (spec section 5.4).

**Failing to beat LRU is section 7.1 vacuity, stated behaviourally.** LRU *is*
recency; if RSR cannot beat it, the eviction score is a monotone function of slot
age with extra parameters (falsifier 1).

Also one of E7's two legitimate controls, with H2O (section 10.1).
"""

from __future__ import annotations

import torch
from torch import Tensor

from rsr.retention.policy import AttentionTrace, MemoryState

_KEY = "lru.last_used"


class LRUPolicy:
    """Evict the slot least recently attended to."""

    name = "lru"

    def select_eviction(self, slots: MemoryState, context: Tensor, step: int) -> int:
        last = slots.accum.get(_KEY)
        if last is None:
            last = slots.written_at.clone()
        masked = torch.where(slots.live, last, torch.full_like(last, 2**62))
        return int(masked.argmin().item())

    def observe(self, slots: MemoryState, attn: AttentionTrace, step: int) -> None:
        last = slots.accum.setdefault(_KEY, slots.written_at.clone())
        # "Used" = received the largest share of this step's attention mass,
        # summed over layers and heads. Raw probability, not norm-weighted: LRU is
        # a recency baseline and must not quietly become a weak H2O.
        share = attn.alpha.sum(dim=(0, 1))
        winner = int(torch.where(attn.live, share, torch.full_like(share, -1)).argmax())
        last[winner] = step

    def reset(self) -> None:
        return None
