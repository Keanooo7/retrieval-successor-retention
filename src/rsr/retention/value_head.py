"""The context-conditioned value head `psi_hat` (spec sections 3.2.2, 4.3).

    psi_hat_phi(s_i, c_t) = s_i^T W c_t + u^T [s_i ; c_t]        phi = {W, u}

**Why context-conditioned.** v0.1's value was content-only, and the fixed point
collapsed to `r_i / (1 - gamma)` -- a rescaled *instantaneous* reward, with no
temporal credit assignment. "Successor" was decorative (defect B1).

The correction is the conceptual core of the model:

> A successor representation requires a state that transitions. A memory slot has
> no successor slot -- slots do not lead anywhere. **The state that transitions is
> the reader.**

So `psi_hat` reads as: *how much will this memory be needed, given where the
discourse currently is.* Cost is `O(M*d)` per step.

**Age is excluded, deliberately** (section 3.2.2). Supplying it invites collapse
onto recency and makes the vacuity failure mode (section 7.1) invisible rather
than merely possible. An age-only head and a content+age head are retained as
separate baseline arms; they are not this class.

**The anti-collapse bias `b` never enters here** (section 3.5). If it did, it
would produce gradients pushing toward balanced retention at the expense of
language-modeling loss, and would corrupt the estimate itself -- an under-attended
slot would receive an inflated predicted *demand*, not merely an improved chance
of survival. Balance is a zero-gradient control loop applied inside the eviction
argmin, not a competing objective.

This module is Sprint 1 scope (kickoff T7): E0a needs a value head to exist in
week 1, while the rest of sections 3.2-3.5 is week-2 work.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

__all__ = ["BilinearValueHead"]


class BilinearValueHead(nn.Module):
    """`psi_hat` with its muP parameterization.

    The muP prescription for a bilinear form is written out because "readout-like"
    is not a prescription (defect C6), and because getting it wrong fails silently
    in week 9 with no error message.

    Read `psi_hat = s_i^T W c_t` as two stages: `h = W c_t` is a hidden matrix
    producing Theta(1) coordinates, and `s_i^T h` is a dot product of two
    `d`-dimensional Theta(1) vectors. That dot product is **Theta(sqrt(d)) at
    init** (uncorrelated) and **Theta(d) after training** (once `s`, `W` and `c`
    are correlated). muP is governed by the trained regime, so the output needs an
    explicit `1/d` multiplier.

    | Param | Init | Multiplier | LR rule |
    |---|---|---|---|
    | `W` (`d x d`, bilinear) | `Var = 1/d` | `1/d` on the bilinear | hidden-matrix |
    | `u` (`2d -> 1`, linear) | `Var = 1/fan_in` | `1/fan_in` | output rule |

    Both LRs follow their muP rule, `~ 1/d`. Both live in their own muP
    parameter group, separate from the transformer.

    **Do not justify the multiplier by "the form sums `d^2` terms and scales as
    Theta(d) under standard init."** At init it is Theta(sqrt(d)); Theta(d) is the
    *correlated* regime. The prescription is the same either way, but the wrong
    argument gives the wrong multiplier for a head of slightly different shape,
    which is exactly how this failure mode propagates.

    E0a must check the coordinate scale of this module's **scalar output**
    specifically, not only transformer activations.

    **Open, and flagged rather than resolved: correction 15.** The reference TG
    L2-normalizes every gestalt to unit norm (`tg_srep_head.py`), so `s_i` -- and
    `c_t`, if it is a gestalt -- have coordinates of order `1/sqrt(d)`, **not
    Theta(1)**. Section 4.3's derivation assumes Theta(1) coordinates, so its
    premise does not hold here and the `1/d` multiplier may leave `psi_hat`
    decaying with width in precisely the trained regime muP governs. The
    multiplier below is the spec's, kept deliberately as written until **E0a
    measures** which one holds the coordinate scale flat. Do not change it from
    the algebra alone.
    """

    def __init__(
        self,
        d_model: int,
        *,
        base_width: int = 128,
        generator: torch.Generator | None = None,
    ) -> None:
        """`generator` seeds `phi` from a stream of its own.

        Section 3.7 / E0b: **construct the value head after the model, or seed
        it from a separate RNG stream.** Otherwise instantiating `phi` consumes
        draws from the global generator, shifts data order and dropout masks,
        and the reduction test fails for a reason unrelated to the mechanism.

        Passing a generator here is the safer of the two routes, because it
        does not depend on construction order staying correct as the codebase
        changes.
        """
        super().__init__()
        self.d_model = d_model
        self.base_width = base_width

        self.W = nn.Parameter(torch.empty(d_model, d_model))
        self.u = nn.Parameter(torch.empty(2 * d_model))

        # Output multipliers, applied in forward rather than folded into init, so
        # that the coordinate check in E0a can read them and so that the muP
        # parameter grouping in rsr.mup stays a property of the optimizer rather
        # than of the weights.
        self.bilinear_multiplier = 1.0 / d_model
        self.linear_multiplier = 1.0 / (2 * d_model)  # = 1/fan_in

        self.reset_parameters(generator)

    def reset_parameters(self, generator: torch.Generator | None = None) -> None:
        """Initialize `phi`.

        `generator` exists for E0b. Section 3.7: **construct the value head after
        the model, or seed it from a separate RNG stream** -- otherwise
        instantiating `phi` consumes draws, shifts data order and dropout masks,
        and the reduction test fails for a reason unrelated to the mechanism.
        """
        with torch.no_grad():
            self.W.normal_(0.0, math.sqrt(1.0 / self.d_model), generator=generator)
            self.u.normal_(0.0, math.sqrt(1.0 / (2 * self.d_model)), generator=generator)

    def forward(self, gestalts: Tensor, context: Tensor) -> Tensor:
        """Score every slot against the current discourse state.

        Args:
            gestalts: `[M, d]` or `[B, M, d]` -- the slot gestalts `s_i`.
            context:  `[d]` or `[B, d]` -- the current sentence gestalt `c_t`.

                **`context` must already be stop-gradiented on the transformer
                side** (section 3.3). This module does not detach it for you:
                doing so silently would hide a caller that meant to backpropagate,
                and section 3.3's rule -- only `phi` receives gradient, never the
                transformer or `W_sent` -- is load-bearing for both [P2]'s
                brittleness finding and the section 3.7 reduction.

        Returns:
            `[M]` or `[B, M]` -- `psi_hat` per slot. Unnormalized; the eviction
            rule z-scores across live slots (section 3.5 item 1), which is what
            makes `b_max = 1.0` mean one standard deviation.
        """
        batched = gestalts.dim() == 3
        if not batched:
            gestalts = gestalts.unsqueeze(0)
            context = context.unsqueeze(0)

        # h = W c_t  ->  [B, d]
        h = context @ self.W.T
        bilinear = torch.einsum("bmd,bd->bm", gestalts, h) * self.bilinear_multiplier

        # u^T [s_i ; c_t], split so the context half is broadcast over slots.
        u_s, u_c = self.u[: self.d_model], self.u[self.d_model :]
        linear = (gestalts @ u_s) + (context @ u_c).unsqueeze(-1)
        linear = linear * self.linear_multiplier

        out = bilinear + linear
        return out if batched else out.squeeze(0)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, base_width={self.base_width}, "
            f"bilinear_multiplier={self.bilinear_multiplier:.3g}, "
            f"linear_multiplier={self.linear_multiplier:.3g}"
        )
