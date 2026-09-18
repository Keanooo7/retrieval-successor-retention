"""FIFO -- stock TG's policy, and the arm section 3.7 reduces to.

[P2] evicts the oldest entry. **The entire contribution of the RSR document is
replacing that one line** (section 3.1).

Implemented for real, not stubbed, because `tests/test_reduction.py` compares
against it and because it is E0b's reference behaviour.

Note section 10.1's warning: FIFO is a **degenerate control for E7** (defect C3).
Survival time under FIFO is `min(M, S - i)`, a deterministic function of serial
position, so a raw correlation re-measures position and the partial correlation
zeroes it out by construction. "RSR beats FIFO" on E7 is true trivially and means
nothing. E7's controls are H2O and LRU.
"""

from __future__ import annotations

import torch
from torch import Tensor

from rsr.retention.policy import AttentionTrace, MemoryState


class FIFOPolicy:
    """Evict the slot written earliest."""

    name = "fifo"

    def select_eviction(self, slots: MemoryState, context: Tensor, step: int) -> int:
        written = torch.where(
            slots.live, slots.written_at, torch.full_like(slots.written_at, 2**62)
        )
        return int(written.argmin().item())

    def observe(self, slots: MemoryState, attn: AttentionTrace, step: int) -> None:
        return None

    def on_write(self, slots: MemoryState, slot: int, step: int) -> None:
        """No-op, and genuinely so: FIFO's only state is `written_at`, which the
        memory owns and the write updates. Stated rather than omitted -- gauntlet
        0.4 was a policy whose state quietly did not survive the write."""
        return None

    def reset(self) -> None:
        return None
