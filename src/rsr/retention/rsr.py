"""The RSR policy (spec sections 3.3, 3.4). **Sprint 2 -- typed stub.**

Eviction rule:

    i* = argmin_i [ z(psi_hat_phi(s_i, c_t)) + b_i - nu * max_{j != i} cos(s_i, s_j) ]

**Marginal value, not independent value** (defect D-3). `psi_hat` scores each slot
in isolation, but what makes a memory worth keeping is the loss it saves *that no
other retained slot can save*. Two slots carrying the same proposition split the
attention mass, both look half as valuable, and both become evictable -- losing
the information entirely. Set value under leave-one-out is **submodular**, so
greedy argmin over independent scores is not an approximation to marginal value;
where slots are redundant it is anti-correlated with it (falsifier 5).

**Eviction warmup is required** (defect C5):

    t <  T_warm : eviction is FIFO; psi_hat trains passively on realized r_i
    t >= T_warm : eviction switches to argmin[ z(psi_hat) + b - nu * ... ]

`psi_hat` is a function of representations that are themselves training, and [P2]
states early sentence representations are "largely uninformative". An untrained
head evicts near-randomly during exactly the period when those evictions shape the
gestalts it later depends on. Warmup also gives a clean on-policy/off-policy
boundary: everything before `T_warm` is unbiased FIFO-collected data.

**Learning rule: Monte-Carlo is the default; TD(0) is ablation A8** (correction 2,
defect D-8):

    G_i(t) = sum_{k=0}^{S-t} gamma^k * r_i(t+k)
    L_MC   = sum_i ( psi_hat_phi(s_i, c_t) - sg[ G_i(t) ] )^2

Streams are finite (`S <= 80`), `gamma <= 0.97`, no gradient reaches the
transformer, and only the *policy* must act online -- the *target* need not. MC
removes the bootstrap, the moving target, the EMA target copy and the divergence
risk outright. The cost is variance, plus censoring, which the shadow buffer
handles.

**Reversion check against v0.1.** v0.1's degeneracy was *algebraic*: the same
function of the same argument on both sides of a bootstrap, collapsing the fixed
point to `r_i(t)/(1-gamma)`. MC has no bootstrap, so that collapse cannot recur by
construction. But v0.4's defence -- "there is no closed-form regression that
produces this" -- **is false under MC and is withdrawn**: MC *is* a regression.
What distinguishes it from v0.1 is the target, not the estimator class. Anyone
re-deriving this must check the target, not the presence of a bootstrap.

**Section 3.7 reduction, reachable by configuration alone:** `psi_override="neg_age"`,
`b_enabled=False`, `nu=0.0`, `beta=0.0`, `A_max=M`, `T_warm=inf`, `shadow=False`.
Eviction becomes `argmin(-a_i)` = the oldest slot. Bit-exact TG.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["RSRConfig", "RSRPolicy"]


@dataclass(frozen=True)
class RSRConfig:
    """Every term in the eviction score has an off-switch here.

    Without one, section 3.7 silently stops being a reduction to TG and E0b stops
    testing what it claims to test. `reduction_to_tg()` returns the exact config.
    """

    psi_override: str | None = None
    """None = use the learned head. "neg_age" = section 3.7's `psi_hat == -a_i`."""

    b_enabled: bool = True
    nu: float = 0.0
    beta: float = 1.0
    gamma: float | None = None
    t_warm: float = float("inf")
    shadow_enabled: bool = True
    a_max: int | None = None

    @classmethod
    def reduction_to_tg(cls, capacity: int) -> RSRConfig:
        """Section 3.7's settings. Consumed by `tests/test_reduction.py`."""
        return cls(
            psi_override="neg_age",
            b_enabled=False,
            nu=0.0,
            beta=0.0,
            gamma=0.0,
            t_warm=float("inf"),
            shadow_enabled=False,
            a_max=capacity,
        )


class RSRPolicy:
    name = "rsr"

    def __init__(self, config: RSRConfig, d_model: int) -> None:
        raise NotImplementedError("Sprint 2. Spec sections 3.3-3.5.")
