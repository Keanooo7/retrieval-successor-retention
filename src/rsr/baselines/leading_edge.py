"""Leading-edge strategy, Kintsch & van Dijk 1978 [P11] (spec section 11).

**The named ancestor, and section 2's framing.** A hand-specified retention rule:
keep the most recent propositions plus those highest in the macrostructure,
discard the rest, with reinstatement search when a needed proposition has been
dropped. RSR is a learned version of it.

v0.4 cited [P11] only for the behavioural effect, which a committee in this area
catches in the first ten minutes and reads as not having read one's own citation.
**Implemented as a baseline, not merely cited.** An afternoon of work.

**Sprint 2 -- typed stub.** Kickoff scope boundary: Sprint 1 implements sections
3.2-3.5 only as far as E0a needs, and stubs the rest against the interfaces.
"""

from __future__ import annotations

from torch import Tensor

from rsr.retention.policy import AttentionTrace, MemoryState


class LeadingEdgePolicy:
    name = "leading_edge"

    def select_eviction(self, slots: MemoryState, context: Tensor, step: int) -> int:
        raise NotImplementedError(
            "leading_edge: Sprint 2. See spec section 11, defect S-1."
        )

    def observe(self, slots: MemoryState, attn: AttentionTrace, step: int) -> None:
        raise NotImplementedError(
            "leading_edge: Sprint 2. See spec section 11, defect S-1."
        )

    def reset(self) -> None:
        raise NotImplementedError(
            "leading_edge: Sprint 2. See spec section 11, defect S-1."
        )
