"""The context-conditioned retention value head, in muP.

    psi-hat_phi(s_i, c_t) = s_i^T W c_t + u^T [s_i ; c_t]        phi = {W, u}

§3.2.2: "A successor representation requires a state that transitions. A memory slot has no
successor slot -- slots do not lead anywhere. **The state that transitions is the reader.**" So the
value is conditioned on ``c_t``, the current discourse state, not on the slot alone.

**Age is deliberately absent** (§3.2.2). Supplying it invites collapse onto recency and makes the
vacuity failure mode invisible rather than merely possible. An age-only head and a content+age head
exist as separate baseline arms (A3), never as this one.

muP, per §4.3
-------------
Read ``psi-hat = s_i^T W c_t`` as two stages: ``h = W c_t`` is a hidden matrix producing Theta(1)
coordinates, and ``s_i^T h`` is a dot product of two d-dimensional Theta(1) vectors.

That dot product is **Theta(sqrt(d)) at init** (the vectors are uncorrelated) and **Theta(d) after
training** (once ``s``, ``W`` and ``c`` are correlated). muP is governed by the trained regime, so
the output needs an explicit ``1/d`` multiplier.

⚠️ Do **not** justify the multiplier with "the form sums d^2 terms and scales as Theta(d) under
standard init." At init it is Theta(sqrt(d)). The prescription is the same either way, but the
wrong argument gives the wrong multiplier for a head of slightly different shape, which is exactly
how this failure mode propagates.

| param              | init          | multiplier      | LR                         |
|--------------------|---------------|-----------------|----------------------------|
| ``W`` (d x d)      | Var = 1/d     | ``1/d``         | hidden-matrix rule (1/d)   |
| ``u`` (2d -> 1)    | Var = 1/fan_in| ``1/fan_in``    | output rule (1/d)          |

Both live in their own muP parameter group, separate from the transformer. **E0a must check the
coordinate scale of this head's scalar output specifically**, not only transformer activations --
that is the failure that otherwise surfaces in week 9 with no error message.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

__all__ = ["BilinearValueHead", "zscore_live"]


class BilinearValueHead(nn.Module):
    """psi-hat over every live slot at one step. Cost is O(M*d).

    Returns **raw** scores. Z-scoring is applied at the eviction argmin (§3.5), not here, because
    ``b_max = 1.0`` is only interpretable as "one standard deviation" if the z-score is taken across
    the live slots at the current step -- which this module cannot know is the full set.
    """

    def __init__(self, d: int, *, base_d: int = 128, device=None, dtype=None) -> None:
        super().__init__()
        self.d = d
        self.base_d = base_d
        factory = {"device": device, "dtype": dtype}

        self.W = nn.Parameter(torch.empty(d, d, **factory))
        self.u = nn.Parameter(torch.empty(2 * d, **factory))

        # muP multipliers. Held as buffers so they move with the module and appear in state_dict --
        # a multiplier that lives in a config file drifts away from the weights it scales.
        self.register_buffer("bilinear_mult", torch.tensor(1.0 / d), persistent=False)
        self.register_buffer("linear_mult", torch.tensor(1.0 / (2 * d)), persistent=False)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        # W: hidden matrix, Var = 1/fan_in = 1/d.
        self.W.normal_(mean=0.0, std=1.0 / math.sqrt(self.d))
        # u: output-like, Var = 1/fan_in with fan_in = 2d.
        self.u.normal_(mean=0.0, std=1.0 / math.sqrt(2 * self.d))

    def forward(self, gestalts: Tensor, context: Tensor) -> Tensor:
        """``gestalts`` (n_live, d); ``context`` (d,) or (1, d). Returns (n_live,)."""
        if context.ndim == 2:
            if context.shape[0] != 1:
                raise ValueError(f"context must be a single vector, got {tuple(context.shape)}")
            context = context.squeeze(0)
        if gestalts.shape[-1] != self.d or context.shape[-1] != self.d:
            raise ValueError(
                f"width mismatch: head d={self.d}, gestalts d={gestalts.shape[-1]}, "
                f"context d={context.shape[-1]}"
            )

        # h = W c  -- a hidden matrix producing Theta(1) coordinates.
        h = self.W @ context
        # s_i^T h -- Theta(sqrt(d)) at init, Theta(d) correlated. The 1/d multiplier targets the
        # trained regime, which is the one muP is governed by.
        bilinear = (gestalts @ h) * self.bilinear_mult

        ctx = context.expand(gestalts.shape[0], -1)
        pair = torch.cat([gestalts, ctx], dim=-1)
        linear = (pair @ self.u) * self.linear_mult

        return bilinear + linear

    def mup_param_groups(self, lr: float) -> list[dict]:
        """Parameter groups for AdamW, scaled by the muP rules.

        muP transfers across **width only** (§4.3). ``M``, ``S`` and depth are non-width axes and
        must be held fixed within any comparison. Both groups are separate from the transformer's --
        mis-grouping is the failure that surfaces with no error message.
        """
        scale = self.base_d / self.d
        return [
            {"params": [self.W], "lr": lr * scale, "mup_rule": "hidden_matrix", "name": "psi.W"},
            {"params": [self.u], "lr": lr * scale, "mup_rule": "output", "name": "psi.u"},
        ]

    def extra_repr(self) -> str:
        return f"d={self.d}, base_d={self.base_d}, bilinear_mult=1/{self.d}"


def zscore_live(scores: Tensor, *, eps: float = 1e-6) -> Tensor:
    """Z-score across the live slots at the current step (§3.5).

    Without this, ``psi-hat``'s raw range is unspecified -- at share-normalised reward it can reach
    ``1/(1-gamma)``. If ``psi-hat >> b`` the anti-collapse bias is rounding error; if
    ``psi-hat << b`` **the policy is the balance controller, not the value estimate, and it might
    still beat FIFO for reasons unrelated to the hypothesis.**

    A single live slot has no spread; it is returned as zero rather than as a division by ~0, since
    with one slot there is no eviction choice to make anyway.
    """
    if scores.numel() <= 1:
        return torch.zeros_like(scores)
    std = scores.std(unbiased=False)
    return (scores - scores.mean()) / (std + eps)
