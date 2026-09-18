"""ADR-0006 / D-D -- the rank-shift instrumentation.

The confound: `P^(sent)` indexes **rank in the memory ordering**, not age. Under
FIFO the two coincide; under RSR, evicting a middle slot means a slot's positional
encoding becomes a function of which other slots the policy killed.

🔴 **E0b structurally cannot catch it** -- under section 3.7's reduction the policy
*is* FIFO, so no rank is ever displaced and the test passes while saying nothing.
These tests are the substitute.
"""

from __future__ import annotations

import pytest
import torch

from rsr.retention.instrumentation import EvictionRecord, RankShiftLog, rank_shift
from rsr.retention.policy import MemoryState


def _state(written, live=None, step=100):
    n = len(written)
    return MemoryState(
        gestalts=torch.zeros(n, 2),
        written_at=torch.tensor(written, dtype=torch.long),
        live=torch.ones(n, dtype=torch.bool)
        if live is None
        else torch.tensor(live, dtype=torch.bool),
        step=step,
    )


def test_ranks_are_the_memory_ordering_oldest_first():
    """[P2] section 2.2: "ordered from oldest to most recent"."""
    assert _state([5, 1, 9, 3]).ranks().tolist() == [3, 1, 4, 2]


def test_dead_slots_have_no_rank():
    assert _state([5, 1, 9, 3], live=[1, 0, 1, 1]).ranks().tolist() == [2, -1, 3, 1]


def test_evicting_the_oldest_displaces_nothing():
    """FIFO's case, and the property that makes this a measure of the confound
    rather than of the sliding window."""
    assert rank_shift(_state([5, 1, 9, 3]), victim=1) == (1, 0)


def test_evicting_a_middle_slot_displaces_the_slots_older_than_it():
    """Each older survivor outlives an eviction FIFO would have spent on the
    oldest, and everything behind it now carries an index its age does not imply."""
    assert rank_shift(_state([5, 1, 9, 3]), victim=0) == (3, 2)


def test_evicting_the_newest_displaces_the_most():
    assert rank_shift(_state([5, 1, 9, 3]), victim=2) == (4, 3)


def test_evicting_a_dead_slot_is_an_error():
    with pytest.raises(ValueError, match="not live"):
        rank_shift(_state([5, 1, 9, 3], live=[1, 0, 1, 1]), victim=1)


def _record(shift, rank, step=0):
    return EvictionRecord(
        step=step,
        victim=0,
        victim_rank=rank,
        rank_shift=shift,
        n_live=8,
        policy="rsr",
        warm=False,
        attribution="psi",
    )


def test_the_log_reports_a_distribution_not_only_a_mean():
    """ADR-0006: a policy that evicts the oldest slot 95% of the time and the
    newest 5% of the time has a small mean and a real confound."""
    log = RankShiftLog()
    for _ in range(95):
        log.add(_record(0, 1))
    for _ in range(5):
        log.add(_record(7, 8))
    assert log.distribution() == {0: 95, 7: 5}
    assert log.mean_shift == pytest.approx(0.35)
    assert log.fraction_shifting == pytest.approx(0.05)


def test_a_pure_fifo_run_reports_zero_fraction_shifting():
    log = RankShiftLog()
    for _ in range(50):
        log.add(_record(0, 1))
    assert log.fraction_shifting == 0.0
    assert log.summary()["shift_distribution"] == {0: 50}


def test_an_empty_log_does_not_divide_by_zero():
    log = RankShiftLog()
    assert log.fraction_shifting == 0.0 and log.mean_shift == 0.0


def test_the_eviction_record_carries_section_3_4_attribution():
    """Section 3.4 requires per-eviction decision attribution from the first policy
    run -- and gauntlet 0.2 is a run whose records would all read `fifo_warmup`."""
    import dataclasses

    names = {f.name for f in dataclasses.fields(EvictionRecord)}
    assert {"attribution", "warm", "rank_shift", "victim_rank", "score_margin"} <= names
