"""The bilinear value head (spec sections 3.2.2, 4.3; kickoff T7)."""

from __future__ import annotations

import pytest
import torch

from rsr.retention.value_head import BilinearValueHead


def test_shapes_unbatched_and_batched():
    h = BilinearValueHead(32)
    assert h(torch.randn(8, 32), torch.randn(32)).shape == (8,)
    assert h(torch.randn(4, 8, 32), torch.randn(4, 32)).shape == (4, 8)


def test_multipliers_follow_the_muP_table():
    """Section 4.3: `1/d` on the bilinear output, `1/fan_in` on the linear."""
    h = BilinearValueHead(64)
    assert h.bilinear_multiplier == pytest.approx(1 / 64)
    assert h.linear_multiplier == pytest.approx(1 / 128)  # fan_in = 2d


def test_only_phi_holds_parameters():
    """Section 3.3: only `phi = {W, u}` receives gradient."""
    h = BilinearValueHead(16)
    assert {n for n, _ in h.named_parameters()} == {"W", "u"}


def test_age_is_not_an_input():
    """Section 3.2.2: age is excluded. Supplying it invites collapse onto recency
    and makes the section 7.1 vacuity failure invisible rather than merely
    possible. The signature is the enforcement."""
    import inspect

    params = set(inspect.signature(BilinearValueHead.forward).parameters)
    assert params == {"self", "gestalts", "context"}


def test_does_not_detach_context_for_you():
    """Section 3.3: `c_t` enters with a stop-gradient on the transformer side, but
    the *caller* applies it. Detaching silently here would hide a caller that
    meant to backpropagate into the transformer -- which section 3.3 forbids."""
    h = BilinearValueHead(16)
    c = torch.randn(16, requires_grad=True)
    out = h(torch.randn(4, 16), c)
    out.sum().backward()
    assert c.grad is not None, "head must not silently detach its context argument"


def test_separate_rng_stream_reproduces_phi_exactly():
    """Section 3.7 / E0b: construct `phi` after the model, or seed it from a
    separate RNG stream. Otherwise instantiating the value head consumes draws,
    shifts data order and dropout masks, and the reduction test fails for a reason
    unrelated to the mechanism."""
    a, b = BilinearValueHead(16), BilinearValueHead(16)
    a.reset_parameters(torch.Generator().manual_seed(7))
    b.reset_parameters(torch.Generator().manual_seed(7))
    assert torch.equal(a.W, b.W) and torch.equal(a.u, b.u)


def test_construction_with_a_generator_leaves_the_global_stream_untouched():
    """The E0b trap, tested from the side that actually bites.

    Building `phi` from the global generator consumes draws, which shifts data
    order and dropout masks downstream -- so the reduction test fails for a reason
    that has nothing to do with the eviction rule. Passing a generator must leave
    the global stream exactly where it was.
    """
    torch.manual_seed(0)
    baseline = torch.randn(3)

    torch.manual_seed(0)
    BilinearValueHead(16, generator=torch.Generator().manual_seed(7))
    with_generator = torch.randn(3)

    assert torch.equal(baseline, with_generator), (
        "constructing the value head consumed global RNG draws; this is the E0b "
        "failure mode described in spec section 3.7"
    )


def test_construction_without_a_generator_does_consume_global_draws():
    """The converse, asserted so the hazard is documented rather than assumed.

    This is *not* a bug in the head -- section 3.7 permits either route, the other
    being 'construct the value head after the model'. It is recorded here so that
    anyone who reaches for the bare constructor inside a seeded run knows what they
    are paying for.
    """
    torch.manual_seed(0)
    baseline = torch.randn(3)

    torch.manual_seed(0)
    BilinearValueHead(16)
    after_bare_construction = torch.randn(3)

    assert not torch.equal(baseline, after_bare_construction)
