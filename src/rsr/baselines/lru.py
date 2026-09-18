"""LRU -- the recency policy (spec section 5.4).

**Failing to beat LRU is section 7.1 vacuity, stated behaviourally.** LRU *is*
recency; if RSR cannot beat it, the eviction score is a monotone function of slot
age with extra parameters (falsifier 1).

Also one of E7's two legitimate controls, with H2O (section 10.1).

## Gauntlet 0.4 -- this comparator was crippled in RSR's favour

`last_used` used to live in `MemoryState.accum`. A training loop builds a fresh
`MemoryState` each step, so the record was erased every step, and under the natural
call order -- evict, write, forward, observe -- **LRU's eviction sequence was
byte-identical to FIFO's.** Measured, on the pre-fix tree:

```
slot 0 receives ALL the attention, every step
LRU  evicted [0, 1, 2, 3, 0, 1, 2, 3]
FIFO evicted [0, 1, 2, 3, 0, 1, 2, 3]      identical
```

Under the other call order -- observe first, on the same object -- it kept exactly
one step of memory, which is not LRU either.

The state now lives **on the policy**, and `on_write` invalidates a slot's history
when a new gestalt takes the slot. `tests/test_policies.py` asserts LRU and FIFO
choose *different* slots in a real eviction loop; if they cannot be made to differ,
LRU is not implemented.
"""

from __future__ import annotations

import torch
from torch import Tensor

from rsr.retention.policy import AttentionTrace, MemoryState


class LRUPolicy:
    """Evict the slot least recently attended to.

    `_last_used[i]` is the step at which slot `i` last won the attention mass, or
    the step it was written if it never has. A slot that has never been attended is
    therefore evicted in write order, which is what makes LRU degrade *gracefully*
    toward FIFO rather than silently *into* it.
    """

    name = "lru"

    def __init__(self) -> None:
        self._last_used: dict[int, int] = {}

    def select_eviction(self, slots: MemoryState, context: Tensor, step: int) -> int:
        last = slots.written_at.clone()
        for slot, when in self._last_used.items():
            if slot < last.shape[0]:
                last[slot] = max(int(last[slot].item()), when)
        masked = torch.where(slots.live, last, torch.full_like(last, 2**62))
        return int(masked.argmin().item())

    def observe(self, slots: MemoryState, attn: AttentionTrace, step: int) -> None:
        # "Used" = received the largest share of this step's attention mass,
        # summed over layers and heads. Raw probability, not norm-weighted: LRU is
        # a recency baseline and must not quietly become a weak H2O.
        share = attn.alpha.sum(dim=(0, 1))
        winner = int(torch.where(attn.live, share, torch.full_like(share, -1)).argmax())
        self._last_used[winner] = step

    def on_write(self, slots: MemoryState, slot: int, step: int) -> None:
        """A new gestalt took the slot; the previous tenant's history is void.

        Without this the slot inherits the last-used time of whatever used to live
        there, and a heavily-attended evictee protects its replacement.
        """
        self._last_used[slot] = step

    def reset(self) -> None:
        self._last_used.clear()
