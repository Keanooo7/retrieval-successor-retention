"""The shadow buffer -- censored feedback (spec section 3.4). **Sprint 2 stub.**

`r_i` is observable only for slots *in memory*. A slot evicted at step 5 never
demonstrates that it would have been retrieved at step 40. This is bandit feedback
with survivorship bias: **the policy generates its own training distribution, so
early mistakes are self-confirming and invisible.**

Retain the last `K` evicted gestalts outside the forward pass, compute their
would-be contribution scores post-hoc -- the queries already exist, so cost is one
extra `O(K*d)` matmul -- and use them as counterfactual targets, down-weighted by
`lambda_shadow`.

**`K = M` is wrong** (correction 1, the self-audit). A slot can be evicted at any
point after write, so observing whether it *would* have been retrieved at the
longest gap the claim targets requires shadow coverage out to **that gap**, not
out to `M`. At `M = 40` with E3's window `(40, 64]`, `K = 40` censors exactly the
events E3 exists to measure. Read `K` from `rsr.constants`, which refuses the
wrong value by construction.

**This is also how the PG-19 oracle is computed (section 5.4): build once, use
twice.** The same machinery serves E-feas.

Residual risk, to be reported (section 7.2): shadow targets are counterfactual
*estimates*, not observations -- a slot's would-be attention in a memory it is not
actually in is an approximation. Report the gap between shadow-estimated and
oracle demand on synthetic, where both are available.
"""

from __future__ import annotations

from torch import Tensor

__all__ = ["ShadowBuffer"]


class ShadowBuffer:
    """Last `K` evicted gestalts, scored counterfactually."""

    def __init__(self, k: int, d_model: int) -> None:
        raise NotImplementedError("Sprint 2. Spec section 3.4; K from rsr.constants.")

    def push(self, gestalt: Tensor, evicted_at: int) -> None:
        raise NotImplementedError("Sprint 2. Spec section 3.4.")

    def counterfactual_targets(self, queries: Tensor, step: int) -> Tensor:
        raise NotImplementedError("Sprint 2. Spec section 3.4.")
