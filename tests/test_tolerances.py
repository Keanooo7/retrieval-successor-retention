"""The committed fidelity tolerances (ADR-0002 Part B, D-H, gauntlet 2.2).

These tests exist so the numbers cannot drift after the fixtures are generated.
They are cheap and they are the entire mechanism: a tolerance is only a tolerance
if changing it is visible.
"""

from __future__ import annotations

import pytest

from rsr.model.tg.tolerances import FORWARD, GRADIENT, for_quantity


def test_the_committed_forward_tolerance():
    assert (FORWARD.rtol, FORWARD.atol) == (1e-4, 1e-5)


def test_the_committed_gradient_tolerance():
    assert (GRADIENT.rtol, GRADIENT.atol) == (1e-3, 1e-4)


def test_gradients_are_exactly_one_order_looser():
    """ADR-0002 says one order and gives the reason. Two orders would start
    absorbing the retained-graph divergence the fixtures exist to catch."""
    assert GRADIENT.rtol == pytest.approx(FORWARD.rtol * 10)
    assert GRADIENT.atol == pytest.approx(FORWARD.atol * 10)


@pytest.mark.parametrize(
    "quantity,expected",
    [
        ("activations", FORWARD),
        ("gestalts", FORWARD),
        ("cross_attention", FORWARD),
        ("logits", FORWARD),
        ("grad_w_sent", GRADIENT),
        ("grad_transformer", GRADIENT),
    ],
)
def test_every_fixture_quantity_has_a_committed_tolerance(quantity, expected):
    assert for_quantity(quantity) is expected


def test_an_unknown_quantity_raises_rather_than_defaulting():
    """A new quantity needs a decision in ADR-0002, not whichever branch it lands
    in."""
    with pytest.raises(KeyError, match="ADR-0002"):
        for_quantity("some_new_tensor")


def test_the_reasons_are_recorded_beside_the_numbers():
    """ "Say how much looser and why, in advance." The why travels with the value so
    it cannot be lost when the table is copied."""
    assert "reduce in different orders" in GRADIENT.why
    assert "float32" in FORWARD.why


def test_no_fixture_exists_yet():
    """Gauntlet 2.2's ordering, asserted rather than asserted-about.

    If this ever fails, the fixtures were generated in the same commit as (or
    before) the tolerance, and `git log` no longer proves what ADR-0002 claims.
    Delete this test in the commit that adds the fixtures, and say so there.
    """
    from pathlib import Path

    fixtures = Path(__file__).parent / "fixtures"
    assert not fixtures.exists() or not list(fixtures.glob("*.npz")), (
        "golden tensors exist; this test has served its purpose and its removal "
        "belongs in the same commit that adds them"
    )
