"""The retention-policy interface (kickoff scaffolding item 1)."""

from __future__ import annotations

import torch

from rsr.baselines.fifo import FIFOPolicy
from rsr.baselines.lru import LRUPolicy
from rsr.baselines.random_policy import RandomPolicy
from rsr.retention.policy import AttentionTrace, MemoryState, RetentionPolicy


def _state(capacity=4, d=8, step=10, written=(0, 3, 1, 2), live=(1, 1, 1, 1)):
    return MemoryState(
        gestalts=torch.randn(capacity, d),
        written_at=torch.tensor(written, dtype=torch.long),
        live=torch.tensor(live, dtype=torch.bool),
        step=step,
    )


def _trace(capacity=4, d=8, layers=2, heads=2, step=10, live=(1, 1, 1, 1), used=None):
    """`layers` is the count of CROSS-attention layers -- six in TG, not twelve
    (D-E). `eval_mode` is required, never defaulted (D-F)."""
    alpha = torch.rand(layers, heads, capacity)
    if used is not None:
        alpha[:] = 0.0
        alpha[:, :, used] = 1.0
    return AttentionTrace(
        alpha=alpha,
        wo_v=torch.randn(layers, heads, capacity, d),
        live=torch.tensor(live, dtype=torch.bool),
        step=step,
        eval_mode=True,
        gate=torch.ones(layers),
    )


def test_all_implemented_policies_satisfy_the_protocol():
    for p in (FIFOPolicy(), LRUPolicy(), RandomPolicy()):
        assert isinstance(p, RetentionPolicy), p.name


def test_fifo_evicts_the_oldest_slot():
    """Stock TG's rule, and the arm section 3.7 reduces to."""
    assert FIFOPolicy().select_eviction(_state(), torch.randn(8), 10) == 0


def test_fifo_ignores_dead_slots():
    s = _state(written=(0, 3, 1, 2), live=(0, 1, 1, 1))
    assert FIFOPolicy().select_eviction(s, torch.randn(8), 10) == 2


def test_fifo_survival_is_a_deterministic_function_of_serial_position():
    """Defect C3, recorded as a test so E7 cannot quietly use FIFO as a control.

    Under FIFO survival time is `min(M, S - i)`. A raw correlation with human
    recall therefore re-measures serial position, and the partial correlation
    zeroes it out by construction. E7's controls are H2O and LRU.
    """
    p, capacity, s_len = FIFOPolicy(), 4, 12
    survival = {}
    written_at = list(range(capacity))
    live = [True] * capacity
    for step in range(capacity, s_len):
        st = MemoryState(
            gestalts=torch.zeros(capacity, 2),
            written_at=torch.tensor(written_at),
            live=torch.tensor(live),
            step=step,
        )
        victim = p.select_eviction(st, torch.zeros(2), step)
        survival[written_at[victim]] = step - written_at[victim]
        written_at[victim] = step
    assert set(survival.values()) == {capacity}, (
        "FIFO survival must be exactly M for every interior slot"
    )


def test_lru_evicts_the_least_recently_attended_slot():
    p = LRUPolicy()
    s = _state()
    p.observe(s, _trace(used=3), 10)  # slot 3 is the only one used
    assert p.select_eviction(s, torch.randn(8), 10) != 3


# --------------------------------------------------------------------------- #
# Gauntlet 0.4 -- LRU was silently FIFO. Its state lived in `MemoryState.accum`,
# and a training loop builds a fresh `MemoryState` every step.
#
# "If they cannot be made to differ, LRU is not implemented."
# --------------------------------------------------------------------------- #


def _eviction_loop(policy, attended, n=8, capacity=4, observe_first=False):
    """A real loop: fresh `MemoryState` per step, `on_write` after the write."""
    written, victims = list(range(capacity)), []
    for step in range(capacity, capacity + n):
        st = MemoryState(
            gestalts=torch.zeros(capacity, 2),
            written_at=torch.tensor(written, dtype=torch.long),
            live=torch.ones(capacity, dtype=torch.bool),
            step=step,
        )
        if observe_first:
            policy.observe(
                st, _trace(capacity=capacity, d=2, used=attended, step=step), step
            )
        victim = policy.select_eviction(st, torch.zeros(2), step)
        victims.append(victim)
        written[victim] = step
        policy.on_write(st, victim, step)
        if not observe_first:
            policy.observe(
                st, _trace(capacity=capacity, d=2, used=attended, step=step), step
            )
    return victims


def test_lru_and_fifo_choose_different_slots_in_a_real_eviction_loop():
    """Gauntlet 0.4's PASS criterion, in both call orders.

    Pre-fix, under the natural order -- evict, write, forward, observe -- the two
    sequences were byte-identical:

        LRU  [0, 1, 2, 3, 0, 1, 2, 3]
        FIFO [0, 1, 2, 3, 0, 1, 2, 3]

    Section 7.1 makes "RSR must beat LRU" the behavioural vacuity test and section
    10.1 makes LRU one of E7's two legitimate controls, so a crippled LRU is a
    comparator rigged in RSR's favour.
    """
    fifo = _eviction_loop(FIFOPolicy(), attended=0)
    for observe_first in (False, True):
        lru = _eviction_loop(LRUPolicy(), attended=0, observe_first=observe_first)
        assert lru != fifo, f"LRU == FIFO with observe_first={observe_first}"


def test_lru_remembers_an_attention_event_older_than_one_step():
    """The property that distinguishes LRU from FIFO-with-a-one-step-reprieve.

    Pre-fix, `last_used` was rebuilt from `written_at` every step, so a slot
    attended at t=4 was evicted at t=5 as though it never had been.
    """
    p = LRUPolicy()
    capacity, written = 4, [0, 1, 2, 3]
    st = MemoryState(
        gestalts=torch.zeros(capacity, 2),
        written_at=torch.tensor(written, dtype=torch.long),
        live=torch.ones(capacity, dtype=torch.bool),
        step=4,
    )
    p.observe(st, _trace(capacity=capacity, d=2, used=0, step=4), 4)
    later = MemoryState(  # a FRESH state: the accum-based version forgot here
        gestalts=torch.zeros(capacity, 2),
        written_at=torch.tensor(written, dtype=torch.long),
        live=torch.ones(capacity, dtype=torch.bool),
        step=5,
    )
    assert p.select_eviction(later, torch.zeros(2), 5) != 0


def test_on_write_voids_the_previous_tenant_s_history():
    """A new gestalt must not inherit the slot's last-used time. H2O inherits this
    same hazard for accumulated attention -- hence the hook on the protocol."""
    p = LRUPolicy()
    capacity, written = 4, [0, 1, 2, 3]
    st = MemoryState(
        gestalts=torch.zeros(capacity, 2),
        written_at=torch.tensor(written, dtype=torch.long),
        live=torch.ones(capacity, dtype=torch.bool),
        step=9,
    )
    p.observe(st, _trace(capacity=capacity, d=2, used=1, step=9), 9)
    assert p.select_eviction(st, torch.zeros(2), 9) != 1  # slot 1 is protected
    p.on_write(st, 1, 10)  # ... until a new gestalt takes the slot
    fresh = MemoryState(
        gestalts=torch.zeros(capacity, 2),
        written_at=torch.tensor([0, 10, 2, 3], dtype=torch.long),
        live=torch.ones(capacity, dtype=torch.bool),
        step=11,
    )
    assert p.select_eviction(fresh, torch.zeros(2), 11) == 0  # oldest, as it should


def test_reset_clears_lru_state_at_a_stream_boundary():
    """[P2] fact 2: memory is reset at each stream boundary. Anything that must
    survive one belongs on the policy -- and anything that must NOT, must go."""
    p = LRUPolicy()
    capacity = 4
    st = MemoryState(
        gestalts=torch.zeros(capacity, 2),
        written_at=torch.arange(capacity, dtype=torch.long),
        live=torch.ones(capacity, dtype=torch.bool),
        step=9,
    )
    p.observe(st, _trace(capacity=capacity, d=2, used=0, step=9), 9)
    p.reset()
    assert p.select_eviction(st, torch.zeros(2), 9) == 0  # back to write order


def test_every_policy_implements_the_admission_hook():
    """Gauntlet 0.4's root cause: the protocol had no write/admission hook, so no
    policy was ever told a slot changed occupant."""
    from rsr.baselines.expire_span import ExpireSpanPolicy
    from rsr.baselines.h2o import H2OPolicy
    from rsr.baselines.leading_edge import LeadingEdgePolicy
    from rsr.baselines.oracle import OraclePolicy

    for cls in (
        FIFOPolicy,
        LRUPolicy,
        RandomPolicy,
        H2OPolicy,
        ExpireSpanPolicy,
        LeadingEdgePolicy,
        OraclePolicy,
    ):
        assert hasattr(cls, "on_write"), cls.__name__


def test_random_is_reproducible_under_a_seeded_generator():
    g = torch.Generator().manual_seed(0)
    a = [RandomPolicy(g).select_eviction(_state(), torch.randn(8), 10) for _ in range(5)]
    g = torch.Generator().manual_seed(0)
    b = [RandomPolicy(g).select_eviction(_state(), torch.randn(8), 10) for _ in range(5)]
    assert a == b


def test_attention_trace_keeps_layers_and_heads_separate():
    """Section 3.2.1 requires the per-layer profile before collapsing to a scalar,
    and `W_O` must be included -- so the trace cannot pre-collapse either axis."""
    tr = _trace(layers=3, heads=4, capacity=5, d=16)
    assert tr.alpha.shape == (3, 4, 5)
    assert tr.wo_v.shape == (3, 4, 5, 16)


def test_memory_state_reports_fill_fraction():
    """Section 5.2 requires mean(|memory_t|/M) per experiment."""
    assert _state(live=(1, 1, 1, 0)).fill_fraction() == 0.75


def test_memory_state_marks_dead_slots_in_ages():
    ages = _state(written=(0, 3, 1, 2), live=(1, 0, 1, 1)).ages()
    assert ages.tolist() == [10, -1, 9, 8]
