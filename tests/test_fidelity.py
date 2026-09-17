"""The PyTorch TG against golden tensors from the pinned JAX reference.

**New in Sprint 1; not in the spec.** It exists because Task 0's answer was not
the one the kickoff anticipated. See ADR-0001 and ADR-0002, and
`docs/spec-corrections.md` correction 14.

The kickoff planned to vendor TG and "reproduce its reported numbers (29.8 test
PPL, 21 sentence-steps/sec)" as a smoke test. The released code's own README says
it *"differs slightly from the version of the model described in"* the paper, so
that smoke test would be measuring a difference nobody has characterized. This
harness replaces it.

## Protocol

- Golden tensors are extracted **once**, from the pinned JAX reference, on
  **rented hardware**. JAX is never installed on the development machine
  (ADR-0001).
- Fixed seed, fixed batch, `d = 128`, ~20 sentence steps.
- Fixtures cover per-layer activations, gestalt vectors, cross-attention weights,
  logits, **and gradients w.r.t. `W_sent` and the transformer parameters**.
- **The tolerance is committed in ADR-0002 BEFORE the fixtures are generated.**
  Same discipline as the E0i pre-registration: a tolerance chosen after seeing the
  mismatch is not a tolerance.

## Why the gradient fixtures are mandatory

Gestalts are appended **without detaching the computation graph** ([P2], spec
section 3.1), and backward depth is bounded by stream length `S` rather than by
memory capacity `M` -- evicting a slot does not free its graph (section 3.6,
[P2] App. A).

**JAX functional autodiff and PyTorch retained-graph semantics diverge exactly
there.** A transcription that matches on every forward quantity while retaining
the wrong graph will pass a forward-only fidelity check and survive to week 7,
where it surfaces as a gradient-flow difference in E3 that looks like a finding.
Forward agreement is necessary and not sufficient.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"

pytestmark = pytest.mark.skip(
    reason=(
        "Blocked on (a) ADR-0002's tolerance being committed and (b) the golden "
        "tensor extraction on rented hardware, then (c) the PyTorch TG "
        "transcription. NOT passing -- unrun. Must not be reported as passing in "
        "GATE-1."
    )
)


def test_golden_fixtures_are_present():
    assert (FIXTURES / "tg_d128_seed0.npz").exists()


def test_forward_matches_jax_reference():
    raise NotImplementedError("ADR-0002.")


def test_gestalt_vectors_match_jax_reference():
    raise NotImplementedError("ADR-0002.")


def test_cross_attention_weights_match_jax_reference():
    raise NotImplementedError("ADR-0002.")


def test_gradients_wrt_w_sent_match_jax_reference():
    """Not optional. See the module docstring: this is where JAX functional
    autodiff and PyTorch retained-graph semantics diverge."""
    raise NotImplementedError("ADR-0002.")


def test_gradients_wrt_transformer_params_match_jax_reference():
    raise NotImplementedError("ADR-0002.")
