"""E0b -- reduction to exact TG (spec section 3.7). **Kill gate.**

Set `psi_hat == -a_i`, `b == 0`, `nu = 0`, `beta = 0`, `A_max = M`, warmup
irrelevant (`T_warm = inf`), shadow buffer off. Eviction becomes `argmin(-a_i)` =
the oldest slot. **Bit-exact TG.**

Verified numerically, not argued.

---

## What this test is for, and what it is not for

**It is intra-repo by design.** It compares RSR-under-the-section-3.7-switches
against *this repository's* PyTorch TG -- not against the published model, and not
against the JAX reference. That is deliberate and it is what section 3.7 actually
needs: the guarantee that **E3's FIFO and RSR arms differ only in the eviction
rule**, so that any difference between them is attributable to retention and
nothing else.

External fidelity is a separate claim with a separate test. See
`tests/test_fidelity.py` and ADR-0001.

## The RNG trap -- read this before debugging a failure here

**Construct `phi` after the model, or seed it from a separate RNG stream.**

Otherwise instantiating the value head consumes draws from the global generator,
which shifts data order and dropout masks, and **this test fails for a reason that
has nothing to do with the mechanism.** Section 3.7 says so explicitly, and it is
written down here so that nobody spends a day debugging it as a mechanism failure.

`BilinearValueHead(d, generator=...)` is the safe route;
`tests/test_value_head.py` pins both halves of that behaviour.

## Why every added term needs an off-switch

`nu = 0` is new in v0.5 and is part of the reduction condition. Every term added
to the eviction score must have a documented off-switch, **or section 3.7 silently
stops being a reduction to TG and this test stops testing what it claims to
test.** `RSRConfig.reduction_to_tg()` is the single place those switches are set.
"""

from __future__ import annotations

import pytest

from rsr.retention.rsr import RSRConfig

pytestmark = pytest.mark.skip(
    reason=(
        "Blocked on the PyTorch TG transcription (ADR-0001, Sprint 1 critical "
        "path). This test is NOT passing and must not be reported as passing in "
        "GATE-1 -- it is unrun. Unskip with the transcription."
    )
)


def test_reduction_config_disables_every_added_term():
    """The section 3.7 switch set, pinned so a new term cannot be added silently."""
    cfg = RSRConfig.reduction_to_tg(capacity=40)
    assert cfg.psi_override == "neg_age"
    assert cfg.b_enabled is False
    assert cfg.nu == 0.0
    assert cfg.beta == 0.0
    assert cfg.t_warm == float("inf")
    assert cfg.shadow_enabled is False
    assert cfg.a_max == 40


def test_loss_curve_is_bit_exact_against_stock_tg():
    """The gate itself. Identical loss curve, not 'close'."""
    raise NotImplementedError("Needs the PyTorch TG. ADR-0001.")
