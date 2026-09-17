"""Leave-one-out delta-loss -- the causal quantity (spec sections 3.2.1, 5.4).

Ablate slot `i`, measure the change in next-sentence loss. This is **truth** where
`r_i` is a correlational proxy: "if they disagree, LOO is truth and `r_i` is a
confound" (E0d).

**This is the same computation as the oracle (section 5.4), so it is built once
and used twice** -- three times, in fact: E0d validates `r_i` against it, E-feas
probes oracle-vs-FIFO headroom with it, and the shadow buffer scores
counterfactuals through it. Kickoff T6: "Build the LOO harness as the shared
oracle."

Affordable only on a subsample; section 13 records that limit.
"""

from __future__ import annotations

__all__ = ["loo_delta_loss"]


def loo_delta_loss(*args, **kwargs):
    raise NotImplementedError("E0d / kickoff T6. Spec section 3.2.1.")
