"""Retention configuration -- and §3.7's reduction, expressed as a config rather than a code path.

§3.7: set ``psi-hat = -a_i``, ``b = 0``, ``nu = 0``, ``beta = 0``, ``A_max = M``,
``T_warm = inf``, shadow buffer off, and eviction becomes ``argmin_i(-a_i)`` = the oldest slot.
Bit-exact TG.

> "**``nu = 0`` is new in v0.5 and is part of the reduction condition.** Every term added to the
> eviction score must have a documented off-switch, or §3.7 silently stops being a reduction to TG
> and E0b stops testing what it claims to test."

That sentence is the reason this file exists. Every term in the eviction score has a field here, and
``RetentionConfig.reduction_to_tg()`` sets all of them. A term added to the score without a switch
here breaks the reduction silently, so ``tests/test_reduction.py`` also asserts that the switch set
is complete.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

__all__ = ["PsiSource", "RetentionConfig"]

PsiSource = Literal["head", "negative_age"]


@dataclass(frozen=True)
class RetentionConfig:
    """Every knob that can change an eviction decision.

    ``psi_source``
        ``"head"`` uses the learned bilinear head. ``"negative_age"`` is §3.7's ``psi-hat = -a_i``.
    ``use_bias``
        §3.5's gradient-free anti-collapse loop. ``b`` enters the **argmin only**, never ``psi-hat``
        and never any differentiable path -- if it entered ``psi-hat`` an under-attended slot would
        receive an inflated predicted *demand*, not merely an improved chance of survival.
    ``nu``
        §3.4's redundancy penalty on ``max_j cos(s_i, s_j)``. Set value under LOO is submodular, so
        greedy argmin over *independent* scores is not an approximation to marginal value -- under
        redundancy it is anti-correlated with it. (See spec-corrections S-1: [P6] has a submodular
        formulation with a greedy near-optimality theorem that §11 must engage.)
    ``beta``
        Weight on the retention loss in ``L = L_NTP + beta * L_MC``. Note ``L_MC``, not ``L_TD`` --
        correction C-2; the spec body still composes the ablation's loss.
    ``t_warm``
        §3.4. Before it, eviction is FIFO and ``psi-hat`` trains passively on realized ``r_i``. An
        untrained head otherwise evicts near-randomly during exactly the period when those evictions
        shape the gestalts it later depends on. Report as a **fraction of total epochs** -- one of
        three is a different experiment from one of twelve.
    ``a_max``
        Hard upper bound on slot age. §3.6a: ``A_max <= S``, and swept only where ``S`` makes it
        bind. ``None`` means unbounded.
    ``use_shadow``
        §3.4's shadow buffer, depth ``K >= max target gap`` -- **not** ``M`` (correction C-1).
    """

    psi_source: PsiSource = "head"
    use_bias: bool = True
    nu: float = 0.0
    beta: float = 0.1
    t_warm: float = 1.0
    a_max: int | None = None
    use_shadow: bool = True
    zscore: bool = True

    @classmethod
    def reduction_to_tg(cls, *, capacity: int) -> RetentionConfig:
        """§3.7. Under this config the policy must be bit-identical to stock TG.

        ``a_max = capacity`` rather than ``None`` because §3.7 names ``A_max = M`` explicitly. With
        ``M`` slots no live slot can exceed age ``M`` under FIFO anyway, so the bound is inert --
        which is the point: it is switched off by being set to a value that cannot bind.
        """
        return cls(
            psi_source="negative_age",
            use_bias=False,
            nu=0.0,
            beta=0.0,
            t_warm=math.inf,
            a_max=capacity,
            use_shadow=False,
            zscore=True,
        )

    @property
    def is_reduction(self) -> bool:
        """True iff every term that can move the argmin is switched off.

        Deliberately does **not** check ``zscore``: a z-score is a strictly monotone transform of
        the live scores, so it cannot reorder them and cannot change an argmin. Leaving it on under
        reduction is what makes the reduction test exercise the real scoring path instead of a
        special case.
        """
        return (
            self.psi_source == "negative_age"
            and not self.use_bias
            and self.nu == 0.0
            and self.beta == 0.0
            and not self.use_shadow
        )

    def switches(self) -> dict[str, object]:
        """Every field that can change an eviction decision, for the completeness assertion."""
        return {
            "psi_source": self.psi_source,
            "use_bias": self.use_bias,
            "nu": self.nu,
            "beta": self.beta,
            "t_warm": self.t_warm,
            "a_max": self.a_max,
            "use_shadow": self.use_shadow,
        }
