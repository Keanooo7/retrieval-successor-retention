"""Anti-collapse: the gradient-free protection bias (section 3.5). **Sprint 2 stub.**

**The failure.** Retention determines what is attendable; attention trains
retention. A retained slot is attended, raising its value, retaining it. Fixed
point: a few slots monopolize memory. Known as router collapse in sparse MoE, and
as over-representation in [P1] Figs. 3f-g.

    b_i <- b_i - gamma_b     if u_bar_i > (1 + tau) / M
    b_i <- b_i + gamma_b     if u_bar_i < (1 - tau) / M
    b_i <- clip(b_i, -b_max, +b_max)

**`b` is added inside the eviction argmin only -- never into `psi_hat`, never into
any differentiable path.** If `b` entered `psi_hat` it would produce gradients
pushing toward balanced retention at the expense of language-modeling loss, and
would corrupt the estimate itself: an under-attended slot would receive an inflated
predicted *demand*, not merely an improved chance of survival. Balance must be a
zero-gradient control loop, not a competing objective.

Four corrections apply -- section 3.5 is headed "Three corrections" and lists four,
numbered 1, 2, 4, 3 (correction 4):

1. **Scale is not free.** z-score `psi_hat` across live slots each step, so
   `b_max = 1.0` means one standard deviation and is interpretable. If
   `psi_hat >> b` the bias is rounding error; if `psi_hat << b` **the policy is
   the balance controller, not the value estimate** -- and it might still beat
   FIFO for reasons unrelated to the hypothesis.
2. **`u_bar` must be an EMA rate, not a cumulative sum.** A raw running sum
   accumulates with age by construction, so the loop preferentially marks old
   slots evictable and smuggles recency back in through the anti-collapse
   mechanism. Half-life = `E[lifetime] / 4`, measured in E0e.
4. **`gamma_b` is a timescale.** `gamma_b ~= b_max / (0.25 * E[lifetime])`, order
   0.05-0.1. **Never 0.001** -- defect D-1: at 0.001 the maximum attainable |b|
   over a slot's entire life is 0.08 SD, the loop cannot move the argmin, and A5
   would have reported "no effect" in week 7, concluding the loop is *unnecessary*
   rather than *absent*. Read it from `rsr.constants`, which refuses it until E0e
   has logged `E[lifetime]`.
3. **`tau` is measured, not frozen.** Attention over slots is heavy-tailed;
   `u_bar` may essentially never sit inside a +/-25% band, leaving `b` saturated
   for most slots most of the time. Measure the distribution on a FIFO run (E0e)
   before freezing. Freezing an unmeasured constant is how v0.2-NP died.

**Decision attribution is required from the first policy run** (section 3.4). The
eviction score has three terms, two added to fix failures of the first. Log, per
eviction, which term determined the argmin. **If `b` flips a large share of
decisions, the balance controller is the policy** -- the v0.2 objection that
z-scoring was introduced to prevent, reopened by D-1's fix. Same test for `nu`.
"""

from __future__ import annotations

from torch import Tensor

__all__ = ["ProtectionBias"]


class ProtectionBias:
    """The zero-gradient balance controller."""

    def __init__(self, capacity: int, b_max: float, gamma_b: float, tau: float) -> None:
        raise NotImplementedError(
            "Sprint 2. Spec section 3.5; gamma_b and tau from rsr.constants (E0e)."
        )

    def update(self, u_bar: Tensor) -> None:
        raise NotImplementedError("Sprint 2. Spec section 3.5.")

    def values(self) -> Tensor:
        raise NotImplementedError("Sprint 2. Spec section 3.5.")

    def reset(self) -> None:
        """Reset at stream boundaries -- [P2] fact 2, and the reason D-1 bit."""
        raise NotImplementedError("Sprint 2. Spec section 3.5.")
