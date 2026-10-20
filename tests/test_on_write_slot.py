"""`on_write` must name the newcomer's slot, and per-slot state must be per row.

2026-09-29, before E1 uses per-slot baselines. Three defects, one test each
(plus one that pins the compaction semantics the fix derives the index from):

1. `run_policy_loop` called `policy.on_write(..., int(victim[row]), t)` AFTER
   `write_at`, which compacts the prefix and writes the newcomer at the **end**
   (spec §3.1: only the eviction rule changes; ADR-0006: the prefix stays
   oldest-first). On a full row the newcomer is at `M - 1`, not at the victim.
2. On a row that was **not** full, `victim` was the placeholder `0`, so every
   underfull write reported slot 0. The newcomer is at `k`, the occupied count.
3. `LRUPolicy._last_used` was one dict shared by every batch row, while
   `observe` / `select_eviction` are called once per row -- row A's attention
   protected row B's slot. And it was keyed by slot *index*, which `write_at`'s
   compaction shifts: a recency record stayed at an index while its occupant
   moved one to the left.

Gauntlet 0.4 is the lineage: the hook exists so per-slot state follows the
occupant. A hook that names the wrong slot is the same defect one level down.
"""

from __future__ import annotations

import torch

from rsr.baselines.lru import LRUPolicy
from rsr.model.tg import TGConfig, TGModel
from rsr.model.tg.model import Memory, init_memory
from rsr.model.tg.policy_loop import memory_state, run_policy_loop, write_at
from rsr.retention.policy import AttentionTrace

M = 3


def _cfg() -> TGConfig:
    return TGConfig(
        D=32,
        H=2,
        V=64,
        max_sentence_tokens=8,
        max_sentences_in_short_term=M,
        pad_id=60,
        bos_id=61,
        eos_id=62,
        eod_id=63,
    )


class _Recorder:
    """Evicts a MIDDLE slot so compaction is non-trivial (victim 0 would make the
    shift a pure roll and could hide an off-by-compaction), and records every
    `on_write` together with the memory it was shown."""

    name = "recorder"

    def __init__(self) -> None:
        self.writes: list[tuple[int, int, int, list[int], list[bool]]] = []
        self.full_at: set[tuple[int, int]] = set()

    def select_eviction(self, slots, context, step):
        self.full_at.add((getattr(slots, "row", -1), step))
        return 1

    def observe(self, slots, attn, step):
        return None

    def on_write(self, slots, slot, step):
        self.writes.append(
            (
                getattr(slots, "row", -1),
                slot,
                step,
                slots.written_at.tolist(),
                slots.live.tolist(),
            )
        )

    def reset(self):
        return None


def _drive(policy, batch=2, steps=7):
    torch.manual_seed(0)
    cfg = _cfg()
    model = TGModel(cfg).eval()
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 59, (batch, steps, cfg.L), generator=g)
    ids[..., 0] = cfg.bos_id
    ids[..., -1] = cfg.eos_id
    mask = torch.ones(batch, steps, cfg.L, dtype=torch.long)
    lengths = torch.full((batch,), steps, dtype=torch.long)

    def step_fn(t, out, ids_t, mask_t, row_valid):
        return out.logits.new_zeros(())

    with torch.no_grad():
        run_policy_loop(model, ids, mask, lengths, policy, step_fn=step_fn)


def test_on_write_receives_the_newcomers_slot_after_an_eviction():
    """Full row, middle victim: the newcomer lands at `M - 1` after compaction."""
    rec = _Recorder()
    _drive(rec)
    full_writes = [w for w in rec.writes if w[2] >= M]
    assert full_writes, "the fixture must reach evictions"
    for _row, slot, step, written, live in full_writes:
        assert live[slot] and written[slot] == step, (
            f"on_write named slot {slot} at step {step}, but that slot holds the "
            f"gestalt written at {written[slot]}; memory written_at={written}"
        )
        assert slot == M - 1


def test_on_write_receives_the_newly_filled_slot_on_a_non_full_row():
    """Underfull row: the newcomer appends at `k`, never the placeholder 0."""
    rec = _Recorder()
    _drive(rec)
    fill_writes = [w for w in rec.writes if w[2] < M]
    assert len(fill_writes) == 2 * M  # two rows, M underfull writes each
    for _row, slot, step, written, live in fill_writes:
        assert slot == step, f"step {step} filled slot {step}, on_write said {slot}"
        assert live[slot] and written[slot] == step


def test_on_write_memory_state_carries_its_row():
    """The row a callback is about is on the `MemoryState` it receives."""
    rec = _Recorder()
    _drive(rec)
    assert sorted({w[0] for w in rec.writes}) == [0, 1]
    assert {r for r, _ in rec.full_at} == {0, 1}


def _mem(written_rows) -> Memory:
    batch = len(written_rows)
    mem = init_memory(batch, _cfg())
    mem.kv = torch.zeros(batch, M, _cfg().D)
    mem.valid = torch.ones(batch, M, dtype=torch.bool)
    mem.step = torch.tensor(written_rows, dtype=torch.long)
    return mem


def _used(slot, step, capacity=M):
    alpha = torch.zeros(1, 1, capacity)
    alpha[..., slot] = 1.0
    return AttentionTrace(
        alpha=alpha,
        wo_v=torch.zeros(1, 1, capacity, 2),
        live=torch.ones(capacity, dtype=torch.bool),
        step=step,
        eval_mode=True,
    )


def test_lru_state_is_per_batch_row():
    """Row A's attention to its slot 1 must not protect row B's slot 1.

    Both rows hold occupants written at (0, 1, 2) -- the realistic case, since
    every row writes at every step, so neither slot index nor write step tells
    the rows apart. Row A uses its slot 1; row B uses its slot 0. Row B's LRU
    victim is its slot 1 (written at 1, never used by row B). A shared record
    protects it with row A's use and evicts slot 2 instead.
    """
    mem = _mem([[0, 1, 2], [0, 1, 2]])
    p = LRUPolicy()
    p.observe(memory_state(mem, 5, 0), _used(1, 5), 5)  # row A uses its slot 1
    p.observe(memory_state(mem, 5, 1), _used(0, 5), 5)  # row B uses its slot 0
    victim_b = p.select_eviction(memory_state(mem, 5, 1), torch.zeros(2), 5)
    assert victim_b == 1, (
        f"row B evicted slot {victim_b}; its slot 1 (written at 1, never used by "
        f"row B) is the LRU victim -- row A's use of ITS slot 1 protected it"
    )


def test_lru_recency_follows_the_occupant_through_compaction():
    """`write_at` shifts every slot behind the victim one to the left. A recency
    record keyed by slot index stays put while its occupant moves.

    Occupants written at (0, 1, 2). Step 3: occupant 1 is used, occupant 0 is
    evicted, occupants become (1, 2, 3). Step 4: the newcomer (3) is used. The
    LRU victim is occupant 2 (last touched when written, at 2), NOT occupant 1
    (used at 3) -- whose record, keyed by index, was left behind at slot 1.
    """
    mem = _mem([[0, 1, 2]])
    p = LRUPolicy()
    # Loop order in run_policy_loop: observe -> select -> write_at -> on_write.
    p.observe(memory_state(mem, 3, 0), _used(1, 3), 3)
    victim = p.select_eviction(memory_state(mem, 3, 0), torch.zeros(2), 3)
    assert victim == 0
    mem = write_at(
        mem, torch.zeros(1, _cfg().D), torch.tensor([True]), torch.tensor([victim]), 3
    )
    assert mem.step[0].tolist() == [1, 2, 3]
    p.on_write(memory_state(mem, 3, 0), M - 1, 3)

    p.observe(memory_state(mem, 4, 0), _used(2, 4), 4)  # the newcomer is used
    victim = p.select_eviction(memory_state(mem, 4, 0), torch.zeros(2), 4)
    assert int(mem.step[0, victim]) == 2, (
        f"evicted the occupant written at {int(mem.step[0, victim])}; the one "
        f"written at 2 was least recently used (occupant 1 was used at step 3)"
    )
