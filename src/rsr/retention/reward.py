"""Retrieval demand `r_i` (spec section 3.2.1).

    r_i(t) = sum_{l,h} || g_mem^(l) . alpha_{l,h,i} . W_O^{(l,h)} v_{l,h,i} ||_2

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

4. **The memory gate `g_mem` is in the formula** (D-E). [P2] puts a learnable
   scalar on each cross-attention layer, scaling the increment *before* the
   residual add, and App. C measures it **growing over training and larger in
   deeper layers**. D-6's argument for `W_O` -- "precisely where head-specific
   rescaling lives" -- applies verbatim to `g_mem`, which is where *layer*-specific
   rescaling lives. Because the gates grow, the weighting is **non-stationary**:
   `r_i` collected at epoch 1 and at epoch 12 are not the same measurement.

   **Compute it both ways** -- gated and raw -- and report both against LOO
   delta-loss in E0d. Section 3.2.1's truth rule is unchanged: if they disagree,
   **LOO is truth**.

5. **Collected in EVAL mode** (D-F). `attn_dropout = 0.2` in the reference
   `tg_config.py`, so in training `alpha` is stochastically zeroed and a slot can
   score zero demand because a mask fell on it. That is noise the policy would
   learn from, and it is noise *in a policy-relevant direction*. Eval mode for the
   retention target; train mode for the LM loss. `AttentionTrace.eval_mode` is a
   required field so the two cannot be confused by an ambient default.

6. **The per-layer profile has SIX entries, not twelve** (D-E). TG alternates
   self/cross blocks, so cross-attention lives on half the layers.

Validation is E0d (kickoff T6): Spearman rho(`r_i`, leave-one-out delta-loss) on a
held-out subsample. **If they disagree, LOO is truth and `r_i` is a confound**, and
section 3.4's redundancy term is the specified response -- not a shrug.
"""

from __future__ import annotations

import torch
from torch import Tensor

from rsr.retention.policy import AttentionTrace

__all__ = [
    "TrainModeTrace",
    "contribution",
    "contribution_per_layer",
    "rescale_to_share_of_M",
    "retrieval_demand",
]


class TrainModeTrace(RuntimeError):
    """A retention target was asked for from a trace captured with dropout live.

    D-F. Not a warning: an on-policy `r_i` collected under `attn_dropout = 0.2` is
    noisy in a policy-relevant way, and the failure is silent -- the numbers look
    fine and the policy learns from mask draws.
    """


def _require_eval(attn: AttentionTrace) -> None:
    if not attn.eval_mode:
        raise TrainModeTrace(
            "r_i must be collected in eval mode (D-F). attn_dropout = 0.2 in the "
            "reference config, so alpha is stochastically zeroed during training "
            "and a slot can score zero demand because a mask fell on it. Capture "
            "the trace under `model.eval()` (or torch.no_grad + dropout off) and "
            "set eval_mode=True. Train mode is for the LM loss, not for this."
        )


def contribution_per_layer(attn: AttentionTrace, *, gated: bool = True) -> Tensor:
    """`[L, M]` norm-weighted contribution, **before** collapsing across layers.

    Section 3.2.1 requires this be reported once before the scalar is used:
    summing across layers conflates contributions to different residual-stream
    positions, and [P2] App. C-D show memory gates and cross-attention gradient
    share are strongly depth-stratified, so the layer sum is not obviously the
    right aggregate.

    `L` is the number of **cross-attention** layers -- six under TG's alternating
    blocks, not twelve (D-E).

    `gated=True` includes `g_mem^(l)`; `gated=False` is the raw form. E0d reports
    both against LOO delta-loss, and LOO is truth if they disagree.
    """
    _require_eval(attn)
    # || g . alpha . W_O v ||_2 over d_model, keeping layer and head separate.
    scaled = attn.wo_v * attn.alpha.unsqueeze(-1)
    if gated:
        if attn.gate is None:
            raise ValueError(
                "gated=True needs AttentionTrace.gate ([L], TG's g_mem). D-E puts "
                "the memory gate in r_i; capture it or ask for gated=False "
                "explicitly and say so in the writeup."
            )
        scaled = scaled * attn.gate.reshape(-1, 1, 1, 1)
    per_head = scaled.norm(dim=-1)  # [L, H, M]
    return per_head.sum(dim=1)  # [L, M]


def contribution(attn: AttentionTrace, *, gated: bool = True) -> Tensor:
    """`[M]` the scalar `sum_{l,h} || g . alpha . W_O v ||_2`, unnormalized."""
    return contribution_per_layer(attn, gated=gated).sum(dim=0)


def rescale_to_share_of_M(raw: Tensor, n_live: int, capacity: int) -> Tensor:
    """Share of live slots, rescaled to share-of-M-equivalent. See module docstring.

        share_i = raw_i / sum_j raw_j          (over live slots)
        r_i     = share_i * n_live / capacity

    **Not unbiased**, and the code must not say it is: it credits a slot as though
    it had competed against `M` slots when only `n_live` existed, which biases
    genuinely important stream-initial content downward. Smaller and better
    understood than the fill-level bias it replaces, which is the honest
    description.
    """
    total = raw.sum()
    if total <= 0:
        # No retrieval at all this step. A uniform share would assert that every
        # slot was equally used, which is the opposite of what was observed.
        return torch.zeros_like(raw)
    return raw / total * (n_live / capacity)


def retrieval_demand(
    attn: AttentionTrace, *, n_live: int, capacity: int, gated: bool = True
) -> Tensor:
    """`[M]` the section 3.2.1 target: norm-weighted, gated, fill-level rescaled."""
    raw = contribution(attn, gated=gated)
    raw = torch.where(attn.live, raw, torch.zeros_like(raw))
    return rescale_to_share_of_M(raw, n_live, capacity)
