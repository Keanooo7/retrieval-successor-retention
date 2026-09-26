"""`T_warm` is counted in training steps, not in sentences (spec §3.4; correction 31).

§3.4's warmup exists because `psi_hat` is untrained **early in training**: "an
untrained head evicts near-randomly during exactly the period when those evictions
shape the gestalts it later depends on." Correction 21 made `T_warm` a float number
of steps and said the dispatch "compares it against `t`". In `run_policy_loop`, `t`
is the **sentence index inside one stream** (0..S-1), and that is what
`select_eviction` received. Two failures follow, one on each side of `S`:

* `T_warm >= S`: `t < T_warm` holds at every sentence of every stream, so the arm
  evicts FIFO for its whole life -- gauntlet 0.1's `T_warm = inf` again, this time
  by mixed units. `train()` passed `steps_per_epoch=iters`, so every real run was
  on this side.
* `T_warm < S`: the first `T_warm` sentences of **every** stream are FIFO and the
  rest are scored, from the first optimizer step on -- a per-stream prefix, not a
  warmup, and no protection for the untrained head at all.

Found by reading on 2026-09-26 (the day-roadmap red team); these tests are the
first execution of it. They go through `train()`, the real call site, because the
defect lives in which counter the caller hands the policy -- a test that fed
`select_eviction` a hand-picked `step` would have encoded the bug, which is what
`tests/test_config.py`'s two warmup tests did.
"""

from __future__ import annotations

import pytest
import torch

import rsr.train.loop as loop
from rsr.retention.policy import MemoryState
from rsr.retention.rsr import RSRConfig, RSRPolicy

MEMORY_SLOTS = 4
STEPS_PER_STREAM = 48
BATCH = 2
EVICTIONS_PER_ITER = (STEPS_PER_STREAM - MEMORY_SLOTS) * BATCH


def _neg_age(t_warm: float) -> RSRConfig:
    """§3.7's score (`psi_hat == -a_i`), so no head and no registry, with only the
    warmup switched on. Every non-warm eviction is attributed `neg_age`."""
    return RSRConfig(
        nu=0.0,
        beta=0.0,
        gamma=0.0,
        t_warm=t_warm,
        a_max=MEMORY_SLOTS,
        psi_override="neg_age",
        b_enabled=False,
        shadow_enabled=False,
    )


def _train_with(monkeypatch, tmp_path, policy: RSRPolicy, iters: int) -> list[list]:
    """Run the real `train()` with `policy` as its eviction rule; return the
    eviction records split per optimizer step."""
    monkeypatch.setattr(loop, "build_policy", lambda *a, **k: policy)
    loop.train(
        d=32,
        steps_per_stream=STEPS_PER_STREAM,
        batch=BATCH,
        memory_slots=MEMORY_SLOTS,
        iters=iters,
        device="cpu",
        out_dir=tmp_path,
        ckpt_every=10_000,
        beat_every=10_000,
    )
    recs = policy.records
    assert len(recs) == iters * EVICTIONS_PER_ITER, (
        f"{len(recs)} evictions over {iters} steps; expected {EVICTIONS_PER_ITER} "
        f"per step -- the memory must fill for this test to test anything"
    )
    return [
        recs[i * EVICTIONS_PER_ITER : (i + 1) * EVICTIONS_PER_ITER] for i in range(iters)
    ]


def test_warmup_below_S_covers_whole_training_steps(monkeypatch, tmp_path):
    """`T_warm = 2` optimizer steps: steps 0 and 1 are all FIFO, steps 2 and 3 are
    all scored. Under the defect the first eviction is at sentence `M = 4 >= 2`, so
    optimizer step 0 is already entirely scored -- no warmup at all."""
    per_step = _train_with(monkeypatch, tmp_path, RSRPolicy(_neg_age(2.0), d_model=32), 4)
    for it, recs in enumerate(per_step):
        want = "fifo_warmup" if it < 2 else "neg_age"
        got = {r.attribution for r in recs}
        assert got == {want}, f"optimizer step {it}: {got}, want {{{want!r}}}"
        assert {r.train_step for r in recs} == {it}, "the record must carry k"


def test_warm_status_never_flips_inside_one_stream(monkeypatch, tmp_path):
    """The invariant the defect broke, stated directly: warmup is a property of
    training progress, so every eviction of one stream shares it. `T_warm = 10`
    sits between `M = 4` and `S = 48`, the range where the defect flips mid-stream
    (sentences 4..9 FIFO, 10..47 scored). An M-or-below value would pass vacuously."""
    assert MEMORY_SLOTS < 10 < STEPS_PER_STREAM
    per_step = _train_with(
        monkeypatch, tmp_path, RSRPolicy(_neg_age(10.0), d_model=32), 2
    )
    for it, recs in enumerate(per_step):
        assert len({r.warm for r in recs}) == 1, (
            f"optimizer step {it}: warm flips inside one stream: {[r.warm for r in recs]}"
        )


def _full(step: int) -> MemoryState:
    return MemoryState(
        gestalts=torch.zeros(MEMORY_SLOTS, 4),
        written_at=torch.arange(step - MEMORY_SLOTS, step),
        live=torch.ones(MEMORY_SLOTS, dtype=torch.bool),
        step=step,
    )


def test_warmup_longer_than_a_stream_ends():
    """The other side of `S`: `T_warm = 60 > S = 48`. Under the defect `t < 60`
    holds at every sentence, so the arm never leaves warmup (every real run: train()
    passed `steps_per_epoch=iters`). Driven through the dispatch directly -- 61
    optimizer steps of `train()` cost over a minute; the two tests above already
    prove `train()` hands the policy its step."""
    policy = RSRPolicy(_neg_age(60.0), d_model=4)
    policy.set_train_step(59)
    for t in range(MEMORY_SLOTS, STEPS_PER_STREAM):
        policy.select_eviction(_full(t), torch.zeros(4), t)
    assert policy.attribution_counts() == {"fifo_warmup": STEPS_PER_STREAM - MEMORY_SLOTS}
    policy.set_train_step(60)
    policy.select_eviction(_full(MEMORY_SLOTS), torch.zeros(4), MEMORY_SLOTS)
    assert policy.attribution_counts()["neg_age"] == 1


def test_a_warmup_policy_refuses_to_guess_its_training_step():
    """Loud, not silent: with `T_warm > 0` and nobody having said which training
    step it is, `select_eviction` raises rather than defaulting -- a default of 0
    would make every eval-time use of a trained policy silently FIFO."""
    policy = RSRPolicy(_neg_age(1.0), d_model=4)
    st = MemoryState(
        gestalts=torch.zeros(MEMORY_SLOTS, 4),
        written_at=torch.arange(MEMORY_SLOTS),
        live=torch.ones(MEMORY_SLOTS, dtype=torch.bool),
        step=MEMORY_SLOTS,
    )
    with pytest.raises(RuntimeError, match="set_train_step"):
        policy.select_eviction(st, torch.zeros(4), MEMORY_SLOTS)


def test_the_reduction_needs_no_training_step():
    """§3.7 / correction 16: `T_warm = 0` means never warm, so E0b's direct calls
    to `run_policy_loop` stay valid without a training counter."""
    policy = RSRPolicy(RSRConfig.reduction_to_tg(capacity=MEMORY_SLOTS), d_model=4)
    st = MemoryState(
        gestalts=torch.zeros(MEMORY_SLOTS, 4),
        written_at=torch.arange(MEMORY_SLOTS),
        live=torch.ones(MEMORY_SLOTS, dtype=torch.bool),
        step=MEMORY_SLOTS,
    )
    assert policy.select_eviction(st, torch.zeros(4), MEMORY_SLOTS) == 0
    assert policy.attribution_counts() == {"neg_age": 1}


def test_build_policy_refuses_rsr_without_an_epoch():
    """Defect (b), correction 31: `train()` passed `steps_per_epoch=iters`, making
    §3.4's "one epoch" the entire run. It now passes None, and what an epoch is on
    that loop is an owner decision, so `build_policy` refuses and names it. (That
    `train()` reaches this refusal is `test_train_loop.py`'s S0-01 test.)"""
    with pytest.raises(ValueError, match="correction 31"):
        loop.build_policy("rsr", d_model=4, steps_per_epoch=None)
