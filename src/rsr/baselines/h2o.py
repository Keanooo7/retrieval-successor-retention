"""H2O -- accumulated-attention eviction [P6] (spec section 5.4).

**The bar, not a floor.** RSR's secondary delta is *predicted future* against
*accumulated past* demand. If RSR does not beat H2O, the finding is that
predicting the future adds nothing over measuring the past -- a legitimate result
and a different paper. It must be known by week 4 (section 7.5).

**Implement both halves -- heavy hitters AND the recent window, at H2O's 50/50
split of M, stated in the paper.** A heavy-hitter-only H2O is a crippled
comparator and a reviewer will catch it.

🔴 **H2O inherits gauntlet 0.4's root cause if `on_write` is ignored.** Accumulated
attention is per *slot*, and a slot changes occupant. Without zeroing on admission a
new gestalt inherits the heat of whatever used to live there, so the heavy-hitter
set becomes partly an artifact of slot reuse -- and because H2O's own bias runs
*anti*-recency ([P6] App. B.2) while the reuse artifact runs pro-recency, the two
would not even cancel in a legible direction. The hook now exists on the protocol;
the implementation must use it.

**Sprint 2 -- typed stub.** Kickoff scope boundary: Sprint 1 implements sections
3.2-3.5 only as far as E0a needs, and stubs the rest against the interfaces.
"""

from __future__ import annotations

from torch import Tensor

from rsr.retention.policy import AttentionTrace, MemoryState


class H2OPolicy:
    name = "h2o"

    def select_eviction(self, slots: MemoryState, context: Tensor, step: int) -> int:
        raise NotImplementedError("h2o: Sprint 2. See spec section 5.4 and section 11.")

    def observe(self, slots: MemoryState, attn: AttentionTrace, step: int) -> None:
        raise NotImplementedError("h2o: Sprint 2. See spec section 5.4 and section 11.")

    def on_write(self, slots: MemoryState, slot: int, step: int) -> None:
        """Must zero this slot's accumulated attention. See the module docstring --
        this is the half of H2O that gauntlet 0.4 predicted would be got wrong."""
        raise NotImplementedError("h2o: Sprint 2. Accumulated attention resets here.")

    def reset(self) -> None:
        raise NotImplementedError("h2o: Sprint 2. See spec section 5.4 and section 11.")
