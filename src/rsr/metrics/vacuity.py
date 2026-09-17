"""E2 -- the vacuity gate (spec section 7.1). **The most likely failure mode.**

**Gate on the full eviction score `z(psi_hat) + b - nu * ...`, not on `psi_hat`
alone.** Given the `u_bar` age confound (section 3.5), gating on `psi_hat`
measures the wrong object.

Three tests, all cheap:

1. **Partial correlation**, not raw rho: does content carry information about
   future retrieval *beyond* age? **Gate at rho < 0.7.** v0.1's 0.9 was far too
   permissive -- Spearman 0.85 with age is essentially recency.
2. **Age-only value head** as an explicit arm. If content+age ~= age-only, content
   contributes nothing, in one cheap run.
3. **RSR must beat LRU.** LRU *is* the recency policy.

Report the number regardless of outcome.
"""

from __future__ import annotations

__all__ = ["partial_correlation_with_age"]


def partial_correlation_with_age(*args, **kwargs):
    raise NotImplementedError("E2. Spec section 7.1.")
