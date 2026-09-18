"""Oracle demand -- the upper bound (spec section 5.4).

Synthetic: exact, because ground-truth discounted demand is known per slot per
step. PG-19: offline via the shadow-buffer machinery (section 3.4) -- **build
once, use twice.**

**Run the oracle first on each corpus as a feasibility probe (E-feas).** If oracle
~= FIFO, that corpus cannot exhibit the effect. One day, saves six weeks.

**Sprint 2 -- typed stub.** Kickoff scope boundary: Sprint 1 implements sections
3.2-3.5 only as far as E0a needs, and stubs the rest against the interfaces.
"""

from __future__ import annotations

from torch import Tensor

from rsr.retention.policy import AttentionTrace, MemoryState


class OraclePolicy:
    name = "oracle"

    def select_eviction(self, slots: MemoryState, context: Tensor, step: int) -> int:
        raise NotImplementedError("oracle: Sprint 2. See spec section 5.4, E-feas.")

    def observe(self, slots: MemoryState, attn: AttentionTrace, step: int) -> None:
        raise NotImplementedError("oracle: Sprint 2. See spec section 5.4, E-feas.")

    def on_write(self, slots: MemoryState, slot: int, step: int) -> None:
        """The oracle's per-slot true demand is re-derived for the new occupant
        (gauntlet 0.4)."""
        raise NotImplementedError("Sprint 2.")

    def reset(self) -> None:
        raise NotImplementedError("oracle: Sprint 2. See spec section 5.4, E-feas.")
