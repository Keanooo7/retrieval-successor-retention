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


def test_lru_on_write_is_defensive_and_the_mutation_battery_says_so():
    """🔴 **A finding, recorded rather than hidden.**

    The 1.7 battery neutered `LRUPolicy.on_write` and **nothing reddened**. The
    original version of this test asserted the hook was load-bearing; it was not,
    and the test was vacuous.

    The reason is in `select_eviction`: it takes `max(written_at[i], last_used[i])`,
    and a stale `last_used` is by construction *older* than the new occupant's write
    step -- a slot can only be attended at or after the step it was written. So the
    `max` already discards the previous tenant's history, and `on_write` is
    redundant **for LRU specifically**.

    It is kept, for two reasons:

    1. The hook is on the protocol because **H2O needs it** -- accumulated attention
       is not monotone in write time, so a stale accumulator survives the `max`
       trick and corrupts the heavy-hitter set. That is proved below, on a stand-in,
       rather than asserted about code that does not exist yet.
    2. If anyone simplifies the `max` away -- and `last = written_at.clone()` looks
       like dead weight until you know why it is there -- `on_write` becomes
       load-bearing immediately.

    This test pins the *argument*, since the behaviour cannot be pinned.
    """
    p = LRUPolicy()
    st = MemoryState(
        gestalts=torch.zeros(4, 2),
        written_at=torch.tensor([0, 25, 2, 3], dtype=torch.long),
        live=torch.ones(4, dtype=torch.bool),
        step=26,
    )
    p._last_used[1] = 20  # a stale record from the previous tenant of slot 1
    # The max() discards it without help: the slot's own write step is later.
    assert p.select_eviction(st, torch.zeros(2), 26) == 0
    p.on_write(st, 1, 25)
    assert p.select_eviction(st, torch.zeros(2), 26) == 0


class _Accumulator:
    """H2O's shape, in miniature: a per-slot sum that is NOT monotone in write step.

    Stands in for `H2OPolicy` so gauntlet 0.4's root-cause fix is provable today.
    """

    name = "accumulator"

    def __init__(self, use_hook=True):
        self.total = {}
        self.use_hook = use_hook

    def select_eviction(self, slots, context, step):
        scores = torch.tensor([self.total.get(i, 0.0) for i in range(slots.capacity)])
        scores = torch.where(slots.live, scores, torch.full_like(scores, float("inf")))
        return int(scores.argmin().item())

    def observe(self, slots, attn, step):
        share = attn.alpha.sum(dim=(0, 1))
        for i in range(share.shape[0]):
            self.total[i] = self.total.get(i, 0.0) + float(share[i])

    def on_write(self, slots, slot, step):
        if self.use_hook:
            self.total[slot] = 0.0

    def reset(self):
        self.total.clear()


def test_an_accumulating_policy_is_corrupted_without_the_admission_hook():
    """Gauntlet 0.4's root cause, proved on the policy shape that actually needs it.

    A slot accumulates heat, is evicted, and a new gestalt takes the slot. Without
    `on_write` the newcomer inherits the heat and is protected by attention it never
    received -- so the heavy-hitter set becomes partly an artifact of slot reuse.
    """
    st = MemoryState(
        gestalts=torch.zeros(4, 2),
        written_at=torch.arange(4, dtype=torch.long),
        live=torch.ones(4, dtype=torch.bool),
        step=4,
    )
    # Every slot gets some attention; slot 2 gets ten times as much. Without a
    # baseline the totals tie at zero and the comparison cannot discriminate.
    alpha = torch.full((2, 2, 4), 0.1)
    alpha[:, :, 2] = 1.0
    heat = AttentionTrace(
        alpha=alpha,
        wo_v=torch.zeros(2, 2, 4, 2),
        live=torch.ones(4, dtype=torch.bool),
        step=0,
        eval_mode=True,
        gate=torch.ones(2),
    )

    results = {}
    for use_hook in (True, False):
        p = _Accumulator(use_hook=use_hook)
        for step in range(4, 9):  # slot 2 accumulates the most attention
            p.observe(st, heat, step)
        p.on_write(st, 2, 9)  # slot 2 is overwritten by a brand-new gestalt
        results[use_hook] = p.select_eviction(st, torch.zeros(2), 10)
    assert results[True] == 2, "a fresh occupant has no accumulated attention"
    assert results[False] != 2, "without the hook it inherits the previous tenant's"


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
