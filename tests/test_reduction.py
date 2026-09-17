"""E0b -- the reduction to exact TG. A CI test, not a script (kickoff decision 3).

§3.7: with ``psi-hat = -a_i``, ``b = 0``, ``nu = 0``, ``beta = 0``, ``A_max = M``,
``T_warm = inf`` and the shadow buffer off, eviction becomes ``argmin_i(-a_i)`` = the oldest slot,
and the loss curve must be **bit-identical** to stock TG.

🔴 **THE RNG TRAP, WRITTEN HERE SO NOBODY DEBUGS IT AS A MECHANISM FAILURE.**

§3.7: *"Construct the value head after the model, or seed it from a separate RNG stream --
otherwise instantiating ``phi`` consumes draws, shifts data order and dropout masks, and E0b fails
for a reason unrelated to the mechanism."*

If the full-model reduction test fails, check that first. The symptom is a loss curve that diverges
from step 1 with no eviction ever differing -- which looks exactly like a broken mechanism and is
not one.

⚠️ **Scope.** The full-model half of E0b is blocked on ADR-0001 (no TG implementation yet). What
runs here is the **policy-level** reduction: under the §3.7 config, RSRPolicy's eviction choices are
identical to FIFO's on every input. That is a necessary condition for the loss curves to match, and
it is testable today. ``test_full_model_reduction`` is the outstanding half and is skipped with a
reason, never silently absent.
"""

from __future__ import annotations

import math

import pytest
import torch

from rsr.baselines import FIFOPolicy
from rsr.retention.config import RetentionConfig
from rsr.retention.policy import MemoryState
from rsr.retention.rsr import RSRPolicy

M, D = 40, 64
TRIALS = 200


def _states(seed: int = 0):
    torch.manual_seed(seed)
    for trial in range(TRIALS):
        n = int(torch.randint(2, M + 1, (1,)))
        ages = torch.randperm(500)[:n].sort(descending=True).values
        yield MemoryState(
            gestalts=torch.randn(n, D), ages=ages, capacity=M, step=trial
        ), torch.randn(D)


def test_reduction_config_switches_everything_off():
    cfg = RetentionConfig.reduction_to_tg(capacity=M)
    assert cfg.is_reduction
    assert cfg.psi_source == "negative_age"
    assert cfg.use_bias is False
    assert cfg.nu == 0.0, "nu=0 is part of the reduction condition and is new in v0.5"
    assert cfg.beta == 0.0
    assert cfg.use_shadow is False
    assert cfg.t_warm == math.inf
    assert cfg.a_max == M


def test_every_score_term_has_an_off_switch():
    """§3.7: "Every term added to the eviction score must have a documented off-switch, or §3.7
    silently stops being a reduction to TG and E0b stops testing what it claims to test."

    This asserts the switch set is *complete*: it fails when someone adds a term to the score
    without adding its switch, which is the moment the reduction silently stops being one.
    """
    cfg = RetentionConfig(psi_source="head", nu=0.5)
    head = _head()
    policy = RSRPolicy(cfg, head)
    state, ctx = next(_states())
    _, terms = policy.score(state, ctx)

    scoring_terms = set(terms) - {"psi", "z_psi", "total"}
    switchable = {"b": "use_bias", "redundancy": "nu"}
    unswitchable = scoring_terms - set(switchable)
    assert not unswitchable, (
        f"score terms with no off-switch in RetentionConfig: {sorted(unswitchable)}. "
        "Add one, or §3.7 is no longer a reduction and E0b tests nothing."
    )


def _head():
    from rsr.retention.value_head import BilinearValueHead

    torch.manual_seed(1234)
    return BilinearValueHead(D)


def test_reduction_matches_fifo_via_the_score_path():
    """The load-bearing one.

    ``T_warm`` is set to 0 **deliberately**, so eviction goes through the real scoring path
    (``z(psi-hat) + b - nu*cos``) rather than through the warmup's FIFO shortcut. Testing only the
    full reduction config would route every call through the warmup branch and the test would pass
    without ever evaluating the score -- green, and vacuous.
    """
    cfg = RetentionConfig(
        psi_source="negative_age",
        use_bias=False,
        nu=0.0,
        beta=0.0,
        t_warm=0.0,
        a_max=None,
        use_shadow=False,
    )
    rsr, fifo = RSRPolicy(cfg), FIFOPolicy()
    for state, ctx in _states():
        assert rsr.select_eviction(state, ctx, state.step) == fifo.select_eviction(
            state, ctx, state.step
        )


def test_reduction_matches_fifo_under_the_full_37_config():
    cfg = RetentionConfig.reduction_to_tg(capacity=M)
    rsr, fifo = RSRPolicy(cfg), FIFOPolicy()
    for state, ctx in _states(seed=7):
        assert rsr.select_eviction(state, ctx, state.step) == fifo.select_eviction(
            state, ctx, state.step
        )


def test_reduction_policy_holds_no_state():
    """Under §3.7 the policy must learn nothing. A policy that accumulated state could perturb a
    run even while choosing the same slot, and 'bit-identical' would stop being true."""
    cfg = RetentionConfig.reduction_to_tg(capacity=M)
    rsr = RSRPolicy(cfg)
    from rsr.retention.policy import AttentionTrace

    state, _ = next(_states())
    trace = AttentionTrace(
        weights=torch.rand(12, 4, state.n_live),
        contributions=torch.rand(12, 4, state.n_live),
        memory_gates=torch.rand(12),
    )
    assert rsr.observe(state, trace, 0) is None
    assert rsr._bias == {}


def test_zscore_cannot_reorder():
    """``is_reduction`` deliberately ignores ``zscore``, on the grounds that a z-score is strictly
    monotone and cannot change an argmin. If that is false the reduction argument is false, so it
    is asserted rather than assumed."""
    from rsr.retention.value_head import zscore_live

    torch.manual_seed(3)
    for _ in range(200):
        x = torch.randn(int(torch.randint(2, 41, (1,))))
        assert torch.equal(torch.argsort(x), torch.argsort(zscore_live(x)))


@pytest.mark.skip(
    reason="BLOCKED: no TG implementation. See docs/decisions/ADR-0001-tg-base.md. "
    "This half asserts a bit-exact LOSS CURVE, which needs a model; the policy-level "
    "reduction above is a necessary condition for it, not a substitute."
)
def test_full_model_reduction_bit_exact_loss_curve():
    raise AssertionError("unreachable until src/rsr/model/tg/ exists")


def test_is_reduction_is_pinned_against_every_switch():
    """``is_reduction`` gates whether ``observe`` is allowed to be a no-op, so a hole in it lets a
    partially-configured RSR masquerade as stock TG and learn nothing without saying so.

    🔑 **This test exists because a mutation found its absence.** Deleting the ``nu`` clause from
    ``is_reduction`` left the whole suite green: four other mutations reddened the reduction tests,
    that one reddened nothing. A gate not shown red by a mutation that reddens only it adds
    nothing, so the mutation set is written out here and each switch is flipped individually.
    """
    base = RetentionConfig.reduction_to_tg(capacity=M)
    assert base.is_reduction

    from dataclasses import replace

    mutations = {
        "psi_source": "head",
        "use_bias": True,
        "nu": 0.5,
        "beta": 0.1,
        "use_shadow": True,
    }
    for field, value in mutations.items():
        mutated = replace(base, **{field: value})
        assert not mutated.is_reduction, (
            f"is_reduction stayed True with {field}={value!r}. That switch is not checked, so a "
            "policy configured this way would report itself as the §3.7 reduction and observe() "
            "would silently no-op instead of learning."
        )


def test_observe_refuses_to_no_op_outside_the_reduction():
    """The consequence of the hole above, asserted directly at the behaviour."""
    from dataclasses import replace

    from rsr.retention.policy import AttentionTrace
    from rsr.retention.value_head import BilinearValueHead

    cfg = replace(RetentionConfig.reduction_to_tg(capacity=M), nu=0.5)
    policy = RSRPolicy(cfg, BilinearValueHead(D))
    state, _ = next(_states())
    trace = AttentionTrace(
        weights=torch.rand(12, 4, state.n_live),
        contributions=torch.rand(12, 4, state.n_live),
    )
    with pytest.raises(NotImplementedError):
        policy.observe(state, trace, 0)
