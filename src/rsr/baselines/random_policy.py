"""Random eviction -- the sanity floor (spec section 5.4)."""

from __future__ import annotations

import torch
from torch import Tensor

from rsr.retention.policy import AttentionTrace, MemoryState


class RandomPolicy:
    """Evict a uniformly random live slot."""

    name = "random"

    def __init__(self, generator: torch.Generator | None = None) -> None:
        self.generator = generator

    def select_eviction(self, slots: MemoryState, context: Tensor, step: int) -> int:
        live = slots.live.nonzero(as_tuple=True)[0]
        idx = torch.randint(len(live), (1,), generator=self.generator, device=live.device)
        return int(live[idx].item())

    def observe(self, slots: MemoryState, attn: AttentionTrace, step: int) -> None:
        return None

    def reset(self) -> None:
        return None
