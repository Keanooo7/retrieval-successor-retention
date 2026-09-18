"""Retrieval demand `r_i` (spec section 3.2.1; D-E, D-F).

Three properties that are easy to lose and silent when lost: the memory gate is in
the formula, the trace must be an eval-mode capture, and the fill-level correction
is a **rescale of the target**, not a weight on the loss.
"""

from __future__ import annotations

import pytest
import torch

from rsr.retention.policy import AttentionTrace
from rsr.retention.reward import (
    TrainModeTrace,
    contribution,
    contribution_per_layer,
    rescale_to_share_of_M,
    retrieval_demand,
)

L, H, M, D = 6, 4, 8, 16  # six cross-attention layers, not twelve (D-E)


def _trace(*, eval_mode=True, gate=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    alpha = torch.rand(L, H, M, generator=g)
    alpha = alpha / alpha.sum(dim=-1, keepdim=True)
    return AttentionTrace(
        alpha=alpha,
        wo_v=torch.randn(L, H, M, D, generator=g),
        live=torch.ones(M, dtype=torch.bool),
        step=10,
        eval_mode=eval_mode,
        gate=torch.linspace(0.2, 1.4, L) if gate is None else gate,
    )


def test_the_per_layer_profile_has_six_entries():
    """D-E. TG alternates self/cross blocks, so cross-attention is on half the 12
    layers. A twelve-entry profile is six real rows and six zeros, and the zeros
    would be read as a depth finding."""
    assert contribution_per_layer(_trace()).shape == (L, M)


def test_the_memory_gate_changes_the_answer():
    """D-E. `g_mem` is where layer-specific rescaling lives, exactly as `W_O` is
    where head-specific rescaling lives -- and App. C measures it growing over
    training, so the weighting is non-stationary."""
    tr = _trace()
    gated = contribution(tr, gated=True)
    raw = contribution(tr, gated=False)
    assert not torch.allclose(gated, raw)


def test_both_forms_are_computable_from_one_trace():
    """D-E: "compute r_i both ways and report both against LOO Delta-loss in E0d."
    If they disagree, section 3.2.1's rule stands: LOO is truth."""
    tr = _trace()
    for gated in (True, False):
        assert retrieval_demand(tr, n_live=M, capacity=M, gated=gated).shape == (M,)


def test_gated_without_a_captured_gate_is_refused():
    tr = _trace(gate=torch.ones(L))
    tr = AttentionTrace(
        alpha=tr.alpha, wo_v=tr.wo_v, live=tr.live, step=tr.step, eval_mode=True
    )
    with pytest.raises(ValueError, match="g_mem"):
        contribution(tr, gated=True)
    contribution(tr, gated=False)  # the raw form is still available, explicitly


def test_a_train_mode_trace_is_refused():
    """D-F. `attn_dropout = 0.2`, so in train mode a slot can score zero demand
    because a mask fell on it -- noise in a policy-relevant direction."""
    with pytest.raises(TrainModeTrace, match="eval mode"):
        contribution(_trace(eval_mode=False))


def test_eval_mode_is_a_required_field():
    """ "An explicit mode switch, not an ambient default" (D-F)."""
    import dataclasses

    f = {x.name: x for x in dataclasses.fields(AttentionTrace)}["eval_mode"]
    assert f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING


def test_W_O_is_not_optional():
    """Defect D-6. `r_i` is the norm of the *transformed* vector; a slot can take
    large alpha while contributing almost nothing to the residual stream [P14]."""
    g = torch.Generator().manual_seed(1)
    alpha = torch.zeros(L, H, M)
    alpha[:, :, 0] = 0.9  # slot 0 takes almost all the attention ...
    alpha[:, :, 1] = 0.1
    wo_v = torch.randn(L, H, M, D, generator=g) * 0.001
    wo_v[:, :, 1] = 10.0  # ... but slot 1 carries the content
    tr = AttentionTrace(
        alpha=alpha,
        wo_v=wo_v,
        live=torch.ones(M, dtype=torch.bool),
        step=3,
        eval_mode=True,
        gate=torch.ones(L),
    )
    r = contribution(tr)
    assert r[1] > r[0], "attention probability is not evidence of use"


def test_underfull_steps_are_rescaled_not_masked():
    """Section 3.2.1 / the prohibition list: rescale the target.

    At `n_live = 4` of `M = 8`, a slot taking a third of the live mass is credited
    a third *of half* -- share-of-M-equivalent -- instead of a third of one.
    """
    raw = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    r = rescale_to_share_of_M(raw, n_live=4, capacity=8)
    assert r.sum() == pytest.approx(0.5)
    assert r[0] == pytest.approx(1 / 3 * 0.5)


def test_a_full_memory_is_unchanged_by_the_rescale():
    raw = torch.rand(M, generator=torch.Generator().manual_seed(2))
    assert rescale_to_share_of_M(raw, n_live=M, capacity=M).sum() == pytest.approx(1.0)


def test_a_step_with_no_retrieval_credits_nobody():
    """A uniform share would assert every slot was equally used, which is the
    opposite of what was observed."""
    assert torch.equal(rescale_to_share_of_M(torch.zeros(M), 4, M), torch.zeros(M))


def test_dead_slots_get_no_demand():
    tr = _trace()
    tr = AttentionTrace(
        alpha=tr.alpha,
        wo_v=tr.wo_v,
        live=torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.bool),
        step=tr.step,
        eval_mode=True,
        gate=tr.gate,
    )
    r = retrieval_demand(tr, n_live=4, capacity=M)
    assert torch.equal(r[4:], torch.zeros(4))
