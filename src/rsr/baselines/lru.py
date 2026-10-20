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

    `_last_used[row][w]` is the step at which the occupant written at step `w` in
    batch row `row` last won the attention mass. A slot whose occupant has never
    been attended falls back to its write step, so a never-attended memory is
    evicted in write order -- which is what makes LRU degrade *gracefully* toward
    FIFO rather than silently *into* it.

    ## Why `(row, written_at)` and not slot index (2026-09-29)

    * **Row.** One policy object serves the whole batch and is called once per
      row. A single dict let row A's attention to its slot 1 protect row B's
      slot 1.
    * **Occupant, not index.** `write_at` compacts the prefix behind the victim,
      so every slot after a mid-memory victim moves one to the left. A record
      keyed by index stayed where it was while its occupant moved -- after one
      eviction, the recency of slot `j` described the occupant now at `j - 1`.
      `written_at` is unique among a row's live slots (a row writes at most once
      per step) and it moves with the occupant, because the memory owns it.
    """

    name = "lru"

    def __init__(self) -> None:
        self._last_used: dict[int, dict[int, int]] = {}

    def select_eviction(self, slots: MemoryState, context: Tensor, step: int) -> int:
        seen = self._last_used.get(slots.row, {})
        last = slots.written_at.clone()
        for i in range(last.shape[0]):
            if bool(slots.live[i]):
                when = seen.get(int(slots.written_at[i]))
                if when is not None:
                    last[i] = max(int(last[i].item()), when)
        masked = torch.where(slots.live, last, torch.full_like(last, 2**62))
        return int(masked.argmin().item())

    def observe(self, slots: MemoryState, attn: AttentionTrace, step: int) -> None:
        # "Used" = received the largest share of this step's attention mass,
        # summed over layers and heads. Raw probability, not norm-weighted: LRU is
        # a recency baseline and must not quietly become a weak H2O.
        share = attn.alpha.sum(dim=(0, 1))
        winner = int(torch.where(attn.live, share, torch.full_like(share, -1)).argmax())
        occupant = int(slots.written_at[winner])
        self._last_used.setdefault(slots.row, {})[occupant] = step

    def on_write(self, slots: MemoryState, slot: int, step: int) -> None:
        """A new gestalt took `slot`; start its record and drop evicted tenants'.

        Keyed by occupant, a stale record cannot attach to the newcomer (its
        `written_at` is new), so the hook's job is bookkeeping: seed the newcomer
        at its write step and prune records whose occupant is no longer live, so
        the per-row dict stays bounded by `M` rather than growing with `S`.
        """
        live = {int(w) for w in slots.written_at[slots.live].tolist()}
        seen = self._last_used.setdefault(slots.row, {})
        for w in [w for w in seen if w not in live]:
            del seen[w]
        seen[int(slots.written_at[slot])] = step

    def reset(self) -> None:
        self._last_used.clear()
