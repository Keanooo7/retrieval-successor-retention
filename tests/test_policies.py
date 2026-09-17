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


def _trace(capacity=4, d=8, layers=2, heads=2, step=10, live=(1, 1, 1, 1)):
    return AttentionTrace(
        alpha=torch.rand(layers, heads, capacity),
        wo_v=torch.randn(layers, heads, capacity, d),
        live=torch.tensor(live, dtype=torch.bool),
        step=step,
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
    tr = _trace()
    tr.alpha[:] = 0.0
    tr.alpha[:, :, 3] = 1.0  # slot 3 is the only one used
    p.observe(s, tr, 10)
    assert p.select_eviction(s, torch.randn(8), 10) != 3


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
