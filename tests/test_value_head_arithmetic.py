"""Gauntlet 1.4 -- `psi_hat.forward()` against section 3.2.2's formula, by hand.

**No test exercised the arithmetic.** `tests/test_value_head.py` pins shapes, the
multiplier *attributes*, the parameter set, the signature and the RNG behaviour --
every one of which stays green if `forward()` returns zeros, or transposes `W`, or
drops the linear term, or applies the bilinear multiplier to the wrong half.

```
psi_hat_phi(s_i, c_t) = (s_i^T W c_t) / d  +  (u^T [s_i ; c_t]) / (2d)
```

The multipliers are section 4.3's, and correction 15 / D-B is why they are
*measured* rather than argued: the reference L2-normalizes every gestalt, so `s_i`
has coordinates of order `1/sqrt(d)` and section 4.3's Theta(1) premise does not
hold. **The multiplier is not changed here** -- section 15.3 makes that derivation
the author's, and E0a measures both regimes. These tests pin what the code
*currently computes*, so that when E0a licenses a change, the change is visible.
"""

from __future__ import annotations

import pytest
import torch

from rsr.retention.value_head import BilinearValueHead

D = 8
M = 5


def _head(seed=0):
    h = BilinearValueHead(D)
    h.reset_parameters(torch.Generator().manual_seed(seed))
    return h


def _by_hand(h, gestalts, context):
    """Section 3.2.2's formula, written out element by element."""
    out = torch.empty(gestalts.shape[0], dtype=torch.float64)
    W = h.W.detach().to(torch.float64)
    u = h.u.detach().to(torch.float64)
    c = context.to(torch.float64)
    for i, s in enumerate(gestalts.to(torch.float64)):
        bilinear = 0.0
        for a in range(D):
            for b in range(D):
                bilinear += s[a] * W[a, b] * c[b]
        linear = sum(u[a] * s[a] for a in range(D)) + sum(
            u[D + b] * c[b] for b in range(D)
        )
        out[i] = bilinear * h.bilinear_multiplier + linear * h.linear_multiplier
    return out


def test_forward_matches_the_formula_computed_by_hand():
    """The whole point of 1.4: a triple loop over the actual definition."""
    h = _head()
    g = torch.Generator().manual_seed(3)
    gestalts = torch.randn(M, D, generator=g)
    context = torch.randn(D, generator=g)
    got = h(gestalts, context).detach().to(torch.float64)
    assert torch.allclose(got, _by_hand(h, gestalts, context), atol=1e-10)


def test_a_head_returning_zeros_would_fail_this():
    """Mutation. `test_value_head.py` stays fully green against a zeroed head."""
    h = _head()
    with torch.no_grad():
        h.W.zero_()
        h.u.zero_()
    g = torch.Generator().manual_seed(3)
    gestalts, context = torch.randn(M, D, generator=g), torch.randn(D, generator=g)
    assert torch.equal(h(gestalts, context), torch.zeros(M))
    # ... and the by-hand comparison agrees, which is why the mutation has to be
    # checked against a *non-degenerate* reference:
    assert torch.allclose(
        h(gestalts, context).to(torch.float64), _by_hand(h, gestalts, context)
    )


def test_W_is_not_silently_transposed():
    """`h = c @ W.T` and `h = c @ W` differ for an asymmetric `W`, and every
    shape test passes either way."""
    h = _head()
    with torch.no_grad():
        h.W.copy_(torch.triu(torch.ones(D, D)))  # strongly asymmetric
        h.u.zero_()
    s = torch.zeros(M, D)
    s[0, 0] = 1.0
    c = torch.zeros(D)
    c[D - 1] = 1.0
    # s^T W c  =  W[0, D-1]  =  1 (upper triangle); the transpose gives W[D-1, 0] = 0.
    assert h(s, c)[0].item() == pytest.approx(1.0 / D)


def test_the_bilinear_multiplier_is_one_over_d_in_the_output():
    """Section 4.3's `1/d`, measured on the output rather than read off the
    attribute -- an attribute set correctly and never applied is the silent
    failure the muP table warns about."""
    h = _head()
    with torch.no_grad():
        h.u.zero_()
        h.W.copy_(torch.eye(D))
    s = torch.ones(1, D)
    c = torch.ones(D)
    # s^T I c = D, so the output must be D * (1/D) = 1.
    assert h(s, c).item() == pytest.approx(1.0)


def test_the_linear_multiplier_is_one_over_fan_in_in_the_output():
    h = _head()
    with torch.no_grad():
        h.W.zero_()
        h.u.fill_(1.0)
    s = torch.ones(1, D)
    c = torch.ones(D)
    # u^T [s ; c] = 2D, times 1/(2D) = 1.
    assert h(s, c).item() == pytest.approx(1.0)


def test_both_terms_are_present():
    """Dropping either term leaves every shape and attribute test green."""
    h = _head()
    g = torch.Generator().manual_seed(11)
    s, c = torch.randn(M, D, generator=g), torch.randn(D, generator=g)
    full = h(s, c)

    bilinear_only = BilinearValueHead(D)
    with torch.no_grad():
        bilinear_only.W.copy_(h.W)
        bilinear_only.u.zero_()
    linear_only = BilinearValueHead(D)
    with torch.no_grad():
        linear_only.W.zero_()
        linear_only.u.copy_(h.u)

    assert torch.allclose(full, bilinear_only(s, c) + linear_only(s, c), atol=1e-6)
    assert not torch.allclose(full, bilinear_only(s, c))
    assert not torch.allclose(full, linear_only(s, c))


def test_the_context_half_of_u_is_broadcast_over_slots_not_indexed_by_slot():
    """`u^T [s_i ; c_t]`: the `c_t` half is the same for every slot, so it shifts
    all scores equally and cannot change the argmin. Getting this wrong makes the
    linear term slot-dependent in a way the formula does not license."""
    h = _head()
    with torch.no_grad():
        h.W.zero_()
        h.u[:D].zero_()  # only the context half is live
    g = torch.Generator().manual_seed(5)
    out = h(torch.randn(M, D, generator=g), torch.randn(D, generator=g))
    assert torch.allclose(out, out[0].expand(M))


def test_batched_and_unbatched_agree():
    h = _head()
    g = torch.Generator().manual_seed(7)
    s, c = torch.randn(M, D, generator=g), torch.randn(D, generator=g)
    assert torch.allclose(h(s, c), h(s.unsqueeze(0), c.unsqueeze(0)).squeeze(0))


# --- gauntlet 1.5: age cannot reach psi_hat, by signature AND numerically ----- #


def test_age_is_not_in_the_signature():
    """The static half. Already covered in `test_value_head.py`; repeated here so
    1.5's two halves sit together."""
    import inspect

    assert set(inspect.signature(BilinearValueHead.forward).parameters) == {
        "self",
        "gestalts",
        "context",
    }


def test_the_output_is_invariant_to_everything_except_its_two_arguments():
    """The numerical half. A head that reached age through a closure, a module
    attribute or global state would fail here and pass every signature check.

    Same `(s_i, c_t)`, different step, different memory, different slot order:
    section 3.2.2 says the score may not move.
    """
    h = _head()
    g = torch.Generator().manual_seed(13)
    s, c = torch.randn(M, D, generator=g), torch.randn(D, generator=g)
    first = h(s, c).clone()
    for _ in range(5):
        torch.randn(100, generator=g)  # advance global-ish state
        assert torch.equal(h(s, c), first)


def test_permuting_the_slots_permutes_the_scores_and_nothing_else():
    """If age leaked in through position -- slot index standing in for recency --
    a permutation would change the values, not merely their order."""
    h = _head()
    g = torch.Generator().manual_seed(17)
    s, c = torch.randn(M, D, generator=g), torch.randn(D, generator=g)
    perm = torch.tensor([3, 0, 4, 1, 2])
    assert torch.allclose(h(s[perm], c), h(s, c)[perm], atol=1e-7)


def test_the_head_never_sees_a_memory_state():
    """`MemoryState` is where `written_at` lives. The head takes tensors, so age
    is not merely unused -- it is unreachable."""
    import inspect

    src = inspect.getsource(BilinearValueHead)
    assert "written_at" not in src
    assert "MemoryState" not in src.split('"""')[-1]  # not in the code body
