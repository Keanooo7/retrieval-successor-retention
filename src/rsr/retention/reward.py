"""Retrieval demand `r_i` (spec section 3.2.1). **Sprint 2 -- typed stub.**

    r_i(t) = sum_{l,h} || alpha_{l,h,i} . W_O^{(l,h)} v_{l,h,i} ||_2

then normalized across live slots.

Three things that are easy to get wrong and are the whole point of the definition:

1. **`W_O` is not optional** (defect D-6). [P5] measures the norm of the
   *transformed* vector including the output projection, and `W_O` is where
   head-specific rescaling lives. Dropping it reintroduces the confound the
   norm-weighting was adopted to remove.

2. **Attention weight is not evidence of use.** It is a relative allocation over a
   softmax; a head with nothing to retrieve must still place its mass somewhere
   -- the attention-sink phenomenon [P14]. A slot can receive large `alpha` while
   contributing almost nothing to the residual stream. That is why v0.1's summed
   normalized attention probability was wrong.

3. **Underfull steps: rescale the target. Do not mask, and do not merely
   down-weight** (defect C2, and the kickoff's prohibition list).

       share_i(t) = ||.||_i(t) / sum_j ||.||_j(t)
       r_i(t)     = share_i(t) * |memory_t| / M

   Masking `|memory| < M` deletes half of E3's training signal and 100% of E7's.
   Down-weighting by `|memory_t|/M` reduces how hard the estimator fits a biased
   target; it does not remove the bias. The magnitudes are reciprocal, which is
   why down-weighting approximately works and why the error is easy to miss.

   **Do not call the rescaled target "unbiased."** It removes the *mechanical
   fill-level bias*; it still credits absent competitors and biases genuinely
   important stream-initial slots downward. That is the honest description, and
   "unbiased" is the word a reviewer will test.

Validation is E0d (kickoff T6): Spearman rho(`r_i`, leave-one-out delta-loss) on a
held-out subsample. **If they disagree, LOO is truth and `r_i` is a confound**, and
section 3.4's redundancy term is the specified response -- not a shrug.
"""

from __future__ import annotations

from torch import Tensor

from rsr.retention.policy import AttentionTrace

__all__ = ["contribution", "contribution_per_layer", "rescale_to_share_of_M"]


def contribution_per_layer(attn: AttentionTrace) -> Tensor:
    """`[L, M]` norm-weighted contribution, **before** collapsing across layers.

    Section 3.2.1 requires this be reported once before the scalar is used:
    summing across layers conflates contributions to different residual-stream
    positions, and [P2] App. C-D show memory gates and cross-attention gradient
    share are strongly depth-stratified, so the layer sum is not obviously the
    right aggregate.
    """
    raise NotImplementedError("E0d / Sprint 2. Spec section 3.2.1.")


def contribution(attn: AttentionTrace) -> Tensor:
    """`[M]` the scalar `sum_{l,h} || alpha . W_O v ||_2`, unnormalized."""
    raise NotImplementedError("E0d / Sprint 2. Spec section 3.2.1.")


def rescale_to_share_of_M(raw: Tensor, n_live: int, capacity: int) -> Tensor:
    """Share of live slots, rescaled to share-of-M-equivalent. See module docstring."""
    raise NotImplementedError("Sprint 2. Spec section 3.2.1.")
