"""Fidelity tolerances, committed before any fixture exists (ADR-0002, D-H).

**Read by both the fixture generator and `tests/test_fidelity.py`.** One definition,
so the golden tensors cannot be generated against one number and asserted against
another -- which is the quiet way a committed tolerance stops being one.

A tolerance chosen after seeing the mismatch is not a tolerance, and the evidence
for that is the git history: the commit that introduced this file **precedes** the
commit that introduces `tests/fixtures/`. See ADR-0002 Part B.

If the transcription cannot meet these, **ADR-0002 records the achieved value and
the reason. It does not silently relax.**
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["FORWARD", "GRADIENT", "Tolerance", "for_quantity"]


@dataclass(frozen=True)
class Tolerance:
    rtol: float
    atol: float
    why: str

    def as_kwargs(self) -> dict[str, float]:
        """For `torch.testing.assert_close` / `numpy.allclose`."""
        return {"rtol": self.rtol, "atol": self.atol}


FORWARD = Tolerance(
    rtol=1e-4,
    atol=1e-5,
    why=(
        "float32, matched dtype, dropout deterministic and srep_dropout disabled. "
        "Gestalts are unit-norm by construction (srep_norm_target = 1.0), so atol "
        "is directly interpretable as a fraction of the vector's own length; "
        "cross-attention rows are a simplex where a uniform entry at M = 40 is "
        "0.025, so 1e-5 is ~0.04% of a typical value."
    ),
)

GRADIENT = Tolerance(
    rtol=1e-3,
    atol=1e-4,
    why=(
        "Exactly one order looser than FORWARD, decided in advance. (1) Gradients "
        "accumulate error: a layer-0 gradient is a product of Jacobians over 12 "
        "blocks AND over S sentence steps, a path the forward never traverses. "
        "(2) JAX and PyTorch reduce in different orders, floating-point addition is "
        "not associative, and the backward pass performs far more reductions over "
        "longer axes -- a discrepancy between two correct implementations, not an "
        "error in either. One order and not two: a looser allowance would start "
        "absorbing the retained-graph divergence these fixtures exist to catch, and "
        "a wrong graph is off by a large factor or by everything, not by 1e-3."
    ),
)

_FORWARD_QUANTITIES = frozenset({"activations", "gestalts", "cross_attention", "logits"})
_GRADIENT_QUANTITIES = frozenset({"grad_w_sent", "grad_transformer"})


def for_quantity(name: str) -> Tolerance:
    """The committed tolerance for a named fixture quantity.

    Raises on an unknown name rather than defaulting: a new quantity needs a
    tolerance decided in ADR-0002, not inherited from whichever branch it lands in.
    """
    if name in _FORWARD_QUANTITIES:
        return FORWARD
    if name in _GRADIENT_QUANTITIES:
        return GRADIENT
    known = ", ".join(sorted(_FORWARD_QUANTITIES | _GRADIENT_QUANTITIES))
    raise KeyError(
        f"{name!r} has no committed tolerance. Decide one in ADR-0002 and add it "
        f"here BEFORE generating a fixture for it. Known quantities: {known}."
    )
