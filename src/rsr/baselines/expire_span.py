"""Expire-Span -- learned soft expiry [P7] (spec section 5.4).

**The referendum, not a comparison** (defect D-4, falsifier 6). Soft differentiable
expiry trains retention by the LM loss directly and makes sections 3.2-3.4's entire
apparatus unnecessary. It is the one baseline whose *success invalidates the design
rather than the result*.

Runs at the **week-4 gate on synthetic**. The demotion path was deleted.

**Sprint 2 -- typed stub.** Kickoff scope boundary: Sprint 1 implements sections
3.2-3.5 only as far as E0a needs, and stubs the rest against the interfaces.
"""

from __future__ import annotations

from torch import Tensor

from rsr.retention.policy import AttentionTrace, MemoryState


class ExpireSpanPolicy:
    name = "expire_span"

    def select_eviction(self, slots: MemoryState, context: Tensor, step: int) -> int:
        raise NotImplementedError(
            "expire_span: Sprint 2. See spec section 5.4, defect D-4."
        )

    def observe(self, slots: MemoryState, attn: AttentionTrace, step: int) -> None:
        raise NotImplementedError(
            "expire_span: Sprint 2. See spec section 5.4, defect D-4."
        )

    def reset(self) -> None:
        raise NotImplementedError(
            "expire_span: Sprint 2. See spec section 5.4, defect D-4."
        )
