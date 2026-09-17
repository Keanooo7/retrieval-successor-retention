"""Every arm implements one interface, so no comparison depends on which code path ran."""

from __future__ import annotations

import pytest
import torch

from rsr.baselines import (
    ExpireSpanPolicy,
    FIFOPolicy,
    H2OPolicy,
    LeadingEdgePolicy,
    LRUPolicy,
    OraclePolicy,
    RandomPolicy,
)
from rsr.retention.config import RetentionConfig
from rsr.retention.policy import AttentionTrace, MemoryState, RetentionPolicy
from rsr.retention.rsr import RSRPolicy
from rsr.retention.value_head import BilinearValueHead

M, D = 16, 32

IMPLEMENTED = [FIFOPolicy(), LRUPolicy(), RandomPolicy()]
STUBS = [H2OPolicy(), ExpireSpanPolicy(), LeadingEdgePolicy(), OraclePolicy()]


def _state(n=M):
    torch.manual_seed(0)
    return MemoryState(
        gestalts=torch.randn(n, D),
        ages=torch.arange(n, 0, -1),
        capacity=M,
    )


def _trace(n=M):
    return AttentionTrace(
        weights=torch.rand(12, 4, n),
        contributions=torch.rand(12, 4, n),
        memory_gates=torch.rand(12),
    )


@pytest.mark.parametrize("policy", IMPLEMENTED + STUBS, ids=lambda p: p.name)
def test_conforms_to_protocol(policy):
    assert isinstance(policy, RetentionPolicy)


@pytest.mark.parametrize("policy", IMPLEMENTED, ids=lambda p: p.name)
def test_selects_a_live_slot(policy):
    state = _state()
    idx = policy.select_eviction(state, torch.randn(D), 0)
    assert 0 <= idx < state.n_live


@pytest.mark.parametrize("policy", STUBS, ids=lambda p: p.name)
def test_stubs_raise_rather_than_return_something_plausible(policy):
    """A stub that returned 0 would look like a working FIFO and silently become a baseline."""
    with pytest.raises(NotImplementedError):
        policy.select_eviction(_state(), torch.randn(D), 0)


def test_fifo_picks_the_oldest():
    state = _state()
    assert FIFOPolicy().select_eviction(state, torch.randn(D), 0) == int(
        torch.argmax(state.ages)
    )


def test_age_is_not_reachable_by_the_value_head():
    """§3.2.2: age is excluded from psi-hat. Supplying it "invites collapse onto recency and makes
    the vacuity failure mode invisible rather than merely possible."

    The head's signature takes gestalts and context only -- there is no argument through which age
    could arrive. Asserted so that adding one is a test failure, not a quiet design change.
    """
    import inspect

    params = set(inspect.signature(BilinearValueHead.forward).parameters) - {"self"}
    assert params == {"gestalts", "context"}, f"value head gained an argument: {params}"


def test_memory_state_rejects_mismatched_ages():
    with pytest.raises(ValueError, match="ages"):
        MemoryState(gestalts=torch.randn(4, D), ages=torch.arange(3), capacity=M)


def test_attention_trace_rejects_mismatched_gates():
    with pytest.raises(ValueError, match="memory_gates"):
        AttentionTrace(
            weights=torch.rand(12, 4, M),
            contributions=torch.rand(12, 4, M),
            memory_gates=torch.rand(11),
        )


def test_per_layer_profile_gated_requires_gates():
    """Correction B-1: r_i measured upstream of TG's g_mem mis-weights layers by a factor that
    grows over training and is larger in deeper layers. Asking for the gated profile without the
    gates must fail loudly rather than silently return the ungated one."""
    trace = AttentionTrace(weights=torch.rand(12, 4, M), contributions=torch.rand(12, 4, M))
    with pytest.raises(ValueError, match="memory_gates"):
        trace.per_layer_profile(gated=True)


def test_gated_and_ungated_profiles_differ():
    """If they did not, B-1 would be moot. They do, because the gates are not all 1."""
    trace = _trace()
    assert not torch.allclose(
        trace.per_layer_profile(gated=False), trace.per_layer_profile(gated=True)
    )


def test_rsr_observe_raises_until_sprint_2():
    cfg = RetentionConfig(psi_source="head", t_warm=0.0)
    policy = RSRPolicy(cfg, BilinearValueHead(D))
    with pytest.raises(NotImplementedError, match="Sprint 2"):
        policy.observe(_state(), _trace(), 0)


def test_decision_attribution_is_recorded_from_the_first_run():
    """§3.4 requires it "from the first policy run", not added later."""
    cfg = RetentionConfig(psi_source="head", t_warm=0.0, nu=0.3, use_bias=True)
    policy = RSRPolicy(cfg, BilinearValueHead(D))
    for step in range(10):
        policy.select_eviction(_state(), torch.randn(D), step)
    attr = policy.attribution()
    assert attr["n"] == 10
    assert 0.0 <= attr["bias_flipped"] <= 1.0
    assert 0.0 <= attr["redundancy_flipped"] <= 1.0
