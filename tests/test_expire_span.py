"""Expire-Span [P7] -- the referendum baseline (spec §5.4, D-4, falsifier 6).

Written before the implementation, and revised before the paper-faithfulness fixes
(2026-09-29, after the [P7] source reached the corpus). "P7:L<n>" is a line of
`~/research-corpus/sources/memory-retention/sukhbaatar-2021-expire-span.md`.

What is pinned:

* **forward semantics** -- span `e = L·sigmoid(w·h + b)` (Eq. 3, P7:L259), remaining
  span `r = e - age` (P7:L267), mask `m = clamp(1 + r/R, 0, 1)` (Eq. 5, P7:L284),
  the mask reweighting attention (Eq. 4, P7:L272);
* **per-layer spans** -- "done independently for each layer" (P7:L81-82); heads in
  a layer share one span (P7:L355);
* **eviction** under TG's hard capacity -- a port choice (ADR-0010 q6), not [P7];
* **span reset on admission** (gauntlet 0.4);
* **the span loss** charged on memories on the ramp, `0 < m < 1` (P7:L368-382);
* **structured dropout** -- one `l ~ U(0, L)` per step, drop every memory older
  than `l`, training only (P7:L387-392);
* **the gradient switch** -- what each setting does; which to use is ADR-0010 q1;
* **determinism** -- construction draws nothing from the global RNG (the E0b trap).
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from rsr.baselines.expire_span import GRAD_PATHS, ExpireSpanConfig, ExpireSpanPolicy
from rsr.baselines.fifo import FIFOPolicy
from rsr.model.tg import TGConfig, TGModel
from rsr.model.tg.model import init_memory
from rsr.model.tg.policy_loop import run_policy_loop
from rsr.retention.policy import MemoryState, RetentionPolicy
from rsr.train.checkpoint import load_policy_state, policy_state_dict

D = 8
NL = 3  # cross-attention layers in the unit tests; TG itself has six


def _cfg(**over) -> ExpireSpanConfig:
    # e = 8 · sigmoid(logit(0.25)) = 2 at init.
    base = dict(
        max_span=8.0,
        ramp=2.0,
        loss_coef=0.0,
        structured_dropout=False,
        grad_path="predictor",
        init_span_fraction=0.25,
        seed=0,
    )
    base.update(over)
    return ExpireSpanConfig(**base)


def _policy(n_layers=NL, d=D, **over) -> ExpireSpanPolicy:
    return ExpireSpanPolicy(_cfg(**over), d_model=d, n_layers=n_layers)


def _state(written, step, *, d=D, live=None, gestalts=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    m = len(written)
    return MemoryState(
        gestalts=gestalts if gestalts is not None else torch.randn(m, d, generator=g),
        written_at=torch.tensor(written, dtype=torch.long),
        live=torch.ones(m, dtype=torch.bool)
        if live is None
        else torch.tensor(live, dtype=torch.bool),
        step=step,
    )


def _set_weight(policy: ExpireSpanPolicy, seed: int = 5, scale: float = 3.0) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        policy.weight.copy_(torch.randn(policy.weight.shape, generator=g) * scale)


# --------------------------------------------------------------------------- #
# Config: nothing is invented
# --------------------------------------------------------------------------- #


def test_the_config_has_no_defaults():
    """`R`, `alpha`, `L`, the init and the gradient path are all unregistered in
    `src/rsr/constants.py` and open under B-3. A default here would be an agent
    choosing a hyperparameter of the referendum."""
    with pytest.raises(TypeError):
        ExpireSpanConfig()  # type: ignore[call-arg]
    for f in dataclasses.fields(ExpireSpanConfig):
        assert f.default is dataclasses.MISSING, f.name
        assert f.default_factory is dataclasses.MISSING, f.name


@pytest.mark.parametrize(
    "over",
    [
        {"grad_path": "lm"},
        {"ramp": 0.0},
        {"max_span": 0.0},
        {"loss_coef": -1.0},
        {"init_span_fraction": 0.0},
        {"init_span_fraction": 0.5},
        {"init_span_fraction": 0.9},
        {"structured_dropout": 0.1},
    ],
)
def test_the_config_refuses_out_of_range_values(over):
    """`init_span_fraction < 0.5` because [P7] initialises `b` negative (App. A.1,
    P7:L1013-1015). `structured_dropout` is a switch: [P7]'s form has no rate."""
    with pytest.raises(ValueError):
        _cfg(**over)


def test_the_bias_starts_negative():
    assert float(_policy(init_span_fraction=0.1).bias.max()) < 0.0


def test_the_gradient_paths_are_exactly_three():
    assert GRAD_PATHS == ("none", "predictor", "through_gestalt")


def test_expire_span_satisfies_the_protocol():
    assert isinstance(_policy(), RetentionPolicy)


# --------------------------------------------------------------------------- #
# Determinism: the E0b RNG trap
# --------------------------------------------------------------------------- #


def test_construction_does_not_consume_the_global_rng():
    """`tests/test_reduction.py`'s RNG note: a baseline whose constructor draws from
    the global generator shifts data order and dropout masks for every arm built
    after it."""
    torch.manual_seed(0)
    expect = torch.randn(4)
    torch.manual_seed(0)
    _policy(structured_dropout=True)
    assert torch.equal(expect, torch.randn(4))


def test_structured_dropout_does_not_consume_the_global_rng():
    p = _policy(structured_dropout=True)
    mem = _memory(batch=2, m=4)
    torch.manual_seed(0)
    expect = torch.randn(4)
    torch.manual_seed(0)
    p.memory_weight(mem.kv, mem.valid, mem.step, 6, training=True)
    assert torch.equal(expect, torch.randn(4))


def test_zero_weight_init_gives_every_gestalt_and_layer_the_same_initial_span():
    p = _policy(max_span=10.0, init_span_fraction=0.25)
    e = p.spans(torch.randn(5, D))
    torch.testing.assert_close(e, torch.full((NL, 5), 2.5))


# --------------------------------------------------------------------------- #
# Forward semantics
# --------------------------------------------------------------------------- #


def test_spans_lie_in_zero_to_max_span():
    p = _policy(max_span=6.0)
    _set_weight(p, scale=50.0)
    e = p.spans(torch.randn(64, D, generator=torch.Generator().manual_seed(1))).detach()
    assert float(e.min()) >= 0.0 and float(e.max()) <= 6.0
    assert float(e.max()) > 5.0 and float(e.min()) < 1.0  # both ends reachable


def test_the_mask_is_the_clamped_ramp():
    """Eq. 5: `m = max(0, min(1, 1 + r/R))` (P7:L284)."""
    p = _policy(ramp=2.0)
    r = torch.tensor([3.0, 0.0, -0.5, -1.0, -2.0, -5.0])
    torch.testing.assert_close(
        p.mask_from_remaining(r), torch.tensor([1.0, 1.0, 0.75, 0.5, 0.0, 0.0])
    )


def test_memory_weight_is_zero_on_dead_slots_and_follows_age():
    p = _policy(ramp=2.0)
    mem = _memory(batch=1, m=4, written=[[0, 1, 2, -1]], valid=[[1, 1, 1, 0]])
    w, _ = p.memory_weight(mem.kv, mem.valid, mem.step, 4, training=False)
    assert w.shape == (NL, 1, 4)
    # e = 2 everywhere. ages 4,3,2 -> r = -2,-1,0 -> m = 0, 0.5, 1; dead -> 0.
    for layer in range(NL):
        torch.testing.assert_close(w[layer], torch.tensor([[0.0, 0.5, 1.0, 0.0]]))


def test_each_cross_layer_gets_its_own_span_and_mask():
    """[P7]: "done independently for each layer, allowing different layers to
    specialize at different time-scales" (P7:L81-82)."""
    p = _policy()
    with torch.no_grad():
        p.weight.zero_()
        p.bias.copy_(torch.tensor([-3.0, -1.0, -0.2]))
    mem = _memory(batch=1, m=4, written=[[0, 1, 2, 3]])
    e = p.spans(mem.kv)
    assert e.shape == (NL, 1, 4)
    assert len({round(float(e[layer, 0, 0]), 4) for layer in range(NL)}) == NL
    w, _ = p.memory_weight(mem.kv, mem.valid, mem.step, 6, training=False)
    assert not torch.equal(w[0], w[2])


def test_a_zero_weight_slot_receives_no_attention_and_rows_renormalise():
    """Eq. 4: `a'_i = m_i a_i / Σ_j m_j a_j` (P7:L272-279). A slot at `m = 0`
    gets exactly zero attention in that layer, and the rest still sum to one."""
    cfg = _tg_cfg()
    torch.manual_seed(0)
    model = TGModel(cfg).eval()
    ids, mask, _ = _stream(cfg, steps=1)
    b = ids.shape[0]
    mem = _memory(batch=b, m=cfg.M, d=cfg.D, seed=4)
    weight = torch.ones(_n_cross(cfg), b, cfg.M)
    weight[:, :, 1] = 0.0
    weight[:, :, 2] = 0.25
    out = _forward(model, cfg, ids, mask, mem, capture=True, mem_weight=weight)
    for att in out.cross_attention:
        assert float(att[..., 1].detach().abs().max()) == 0.0
        torch.testing.assert_close(att.sum(-1), torch.ones_like(att.sum(-1)))
    plain = _forward(model, cfg, ids, mask, mem, capture=True)
    assert not torch.allclose(plain.cross_attention[0], out.cross_attention[0])


def test_each_c_block_reads_its_own_layer_of_the_mask():
    cfg = _tg_cfg()
    torch.manual_seed(0)
    model = TGModel(cfg).eval()
    ids, mask, _ = _stream(cfg, steps=1)
    b = ids.shape[0]
    mem = _memory(batch=b, m=cfg.M, d=cfg.D, seed=4)
    weight = torch.ones(_n_cross(cfg), b, cfg.M)
    weight[0, :, 1] = 0.0  # only the FIRST cross-attention layer drops slot 1
    out = _forward(model, cfg, ids, mask, mem, capture=True, mem_weight=weight)
    assert float(out.cross_attention[0][..., 1].detach().abs().max()) == 0.0
    assert float(out.cross_attention[1][..., 1].detach().min()) > 0.0


def test_a_mask_with_the_wrong_layer_count_is_refused():
    cfg = _tg_cfg()
    torch.manual_seed(0)
    model = TGModel(cfg).eval()
    ids, mask, _ = _stream(cfg, steps=1)
    mem = _memory(batch=ids.shape[0], m=cfg.M, d=cfg.D, seed=4)
    with pytest.raises(ValueError):
        _forward(model, cfg, ids, mask, mem, mem_weight=torch.ones(1, 2, cfg.M))


def test_an_all_ones_weight_reproduces_the_unweighted_forward():
    cfg = _tg_cfg()
    torch.manual_seed(0)
    model = TGModel(cfg).eval()
    ids, mask, _ = _stream(cfg, steps=1)
    b = ids.shape[0]
    mem = _memory(batch=b, m=cfg.M, d=cfg.D, seed=4)
    a = _forward(model, cfg, ids, mask, mem)
    ones = torch.ones(_n_cross(cfg), b, cfg.M)
    c = _forward(model, cfg, ids, mask, mem, mem_weight=ones)
    torch.testing.assert_close(a.logits, c.logits, rtol=1e-5, atol=1e-5)


def test_the_loop_passes_the_mask_to_the_model():
    """If `run_policy_loop` computed the mask and never handed it to the model, the
    LM loss would never see the spans and nothing would train them."""
    cfg = _tg_cfg()
    ids, mask, lengths = _stream(cfg, steps=10)
    torch.manual_seed(0)
    model = TGModel(cfg).eval()
    # Spans so short that every slot is fully expired by the time it is read.
    short = ExpireSpanPolicy(
        _cfg(max_span=0.5, ramp=0.1, init_span_fraction=0.01),
        d_model=cfg.D,
        n_layers=_n_cross(cfg),
    )
    loss_short = run_policy_loop(model, ids, mask, lengths, short, step_fn=_loss)
    loss_fifo = run_policy_loop(model, ids, mask, lengths, FIFOPolicy(), step_fn=_loss)
    assert loss_short.item() != loss_fifo.item()


# --------------------------------------------------------------------------- #
# Eviction semantics (a port choice: [P7] has no capacity -- ADR-0010 q6)
# --------------------------------------------------------------------------- #


def test_expire_span_evicts_the_slot_with_least_remaining_span():
    """Not the oldest: a young slot with a short span dies before an old slot with
    a long one. Across layers a slot's life is its LONGEST remaining span -- no
    layer may still be reading it."""
    p = _policy(n_layers=2, max_span=10.0)
    gestalts = torch.zeros(4, D)
    gestalts[:, 0] = torch.tensor([5.0, -5.0, 5.0, 5.0])  # slot 1: short span
    with torch.no_grad():
        p.weight.zero_()
        p.weight[:, 0] = 1.0
    st = _state([0, 5, 2, 3], step=6, gestalts=gestalts)
    assert p.select_eviction(st, torch.zeros(D), 6) == 1
    assert FIFOPolicy().select_eviction(st, torch.zeros(D), 6) == 0  # it differs


def test_a_slot_one_layer_still_reads_is_not_the_victim():
    """Slot 0 is short-lived in layer 0 but long-lived in layer 1; slot 1 is
    middling in both. The slot's life is the max over layers, so slot 1 dies."""
    p = _policy(n_layers=2, max_span=10.0)
    gestalts = torch.zeros(3, D)
    gestalts[0, 0], gestalts[0, 1] = -5.0, 5.0
    gestalts[1, 0], gestalts[1, 1] = 0.0, 0.0
    gestalts[2, 0], gestalts[2, 1] = 5.0, 5.0
    with torch.no_grad():
        p.weight.zero_()
        p.weight[0, 0] = 1.0  # layer 0 reads feature 0
        p.weight[1, 1] = 1.0  # layer 1 reads feature 1
    st = _state([0, 0, 0], step=1, gestalts=gestalts)
    assert p.select_eviction(st, torch.zeros(D), 1) == 1


def test_eviction_ignores_dead_slots():
    p = _policy()
    st = _state([0, 1, 2, 3], step=9, live=[0, 1, 1, 1])
    assert p.select_eviction(st, torch.zeros(D), 9) == 1


def test_at_zero_weight_init_expire_span_evicts_exactly_fifo():
    """Every gestalt, in every layer, has the same span at init, so the least
    remaining is the oldest. Expire-Span starts as TG and moves only as far as
    training moves it."""
    p, f = _policy(), FIFOPolicy()
    written = [0, 1, 2, 3]
    for step in range(4, 30):
        st = _state(written, step, seed=step)
        v = p.select_eviction(st, torch.zeros(D), step)
        assert v == f.select_eviction(st, torch.zeros(D), step)
        written[v] = step
        p.on_write(st, v, step)


def test_span_resets_on_admission():
    """Gauntlet 0.4. A slot whose previous tenant was old and expired takes a new
    gestalt: the newcomer's remaining span is its OWN full span and its mask is 1.
    It must not inherit the evictee's age -- which is how LRU silently became FIFO."""
    p = _policy(ramp=2.0)
    mem = _memory(batch=1, m=3, written=[[0, 1, 2]], valid=[[1, 1, 1]])
    w_old, _ = p.memory_weight(mem.kv, mem.valid, mem.step, 7, training=False)
    assert float(w_old[:, 0, 0].max()) == 0.0  # slot 0's tenant, age 7, expired
    mem.step[0, 0] = 7
    mem.kv[0, 0] = torch.randn(D, generator=torch.Generator().manual_seed(9))
    w_new, _ = p.memory_weight(mem.kv, mem.valid, mem.step, 8, training=False)
    assert float(w_new[:, 0, 0].min()) == 1.0
    st = MemoryState(
        gestalts=mem.kv[0], written_at=mem.step[0], live=mem.valid[0], step=8
    )
    assert p.select_eviction(st, torch.zeros(D), 8) != 0


def test_on_write_observe_and_reset_leave_the_decision_unchanged():
    """No cross-step per-slot state: the hooks are no-ops and stay no-ops."""
    p = _policy()
    _set_weight(p)
    st = _state([0, 1, 2, 3], step=6)
    before = p.select_eviction(st, torch.zeros(D), 6)
    p.on_write(st, 2, 6)
    p.reset()
    assert p.select_eviction(st, torch.zeros(D), 6) == before


# --------------------------------------------------------------------------- #
# Structured dropout: [P7] §4.2 "Regularization" (P7:L387-392)
# --------------------------------------------------------------------------- #


LONG = dict(max_span=200.0, init_span_fraction=0.4)
"""e = 80 at init: no memory of age <= 60 has expired, so a zero weight in
training is structured dropout and nothing else."""


def _ages(mem, step):
    return (step - mem.step).expand(NL, *mem.step.shape)


def test_structured_dropout_drops_every_memory_older_than_one_cutoff():
    """ "For each batch, we sample l ~ U(0, L) and set a_ti = 0 for all t - i > l
    only during training." One cutoff per step, shared by every row and layer:
    the dropped set is exactly the memories older than it."""
    p = _policy(structured_dropout=True, **LONG)
    step = 60
    written = [[step - a for a in (1, 5, 10, 20, 40, 59)]] * 4
    mem = _memory(batch=4, m=6, written=written)
    ages = _ages(mem, step)
    # Nothing has expired (e = 80 > 59 + R), so every zero below is dropout.
    w_eval, _ = p.memory_weight(mem.kv, mem.valid, mem.step, step, training=False)
    assert bool((w_eval == 1.0).all())
    saw_partial = False
    for _ in range(30):
        w, _ = p.memory_weight(mem.kv, mem.valid, mem.step, step, training=True)
        dropped = w == 0.0
        if dropped.any():
            cutoff = int(ages[dropped].min())
            assert bool(dropped[ages >= cutoff].all()), "older memory kept"
        kept = ~dropped
        if kept.any() and dropped.any():
            assert int(ages[kept].max()) < int(ages[dropped].min())
            saw_partial = True
    assert saw_partial


def test_structured_dropout_is_off_at_eval_and_when_switched_off():
    mem = _memory(batch=4, m=6, written=[[0, 5, 10, 20, 40, 59]] * 4)
    long = dict(max_span=1000.0, init_span_fraction=0.4)  # e = 400
    on = _policy(structured_dropout=True, **long)
    off = _policy(structured_dropout=False, **long)
    w_eval, _ = on.memory_weight(mem.kv, mem.valid, mem.step, 60, training=False)
    assert bool((w_eval == 1.0).all())  # long spans: nothing expired
    for _ in range(10):
        w_off, _ = off.memory_weight(mem.kv, mem.valid, mem.step, 60, training=True)
        assert bool((w_off == 1.0).all())


def test_structured_dropout_is_reproducible_under_its_seed():
    mem = _memory(batch=2, m=6, written=[[0, 5, 10, 20, 40, 59]] * 2)

    def draws(seed):
        p = _policy(structured_dropout=True, seed=seed, **LONG)
        return torch.stack(
            [
                p.memory_weight(mem.kv, mem.valid, mem.step, 60, training=True)[0]
                for _ in range(8)
            ]
        )

    assert torch.equal(draws(1), draws(1))
    assert not torch.equal(draws(1), draws(2))


def test_the_checkpoint_round_trips_the_dropout_generator():
    """A resume that restarted the dropout stream would replay the same cutoffs."""
    mem = _memory(batch=2, m=6, written=[[0, 5, 10, 20, 40, 59]] * 2)
    a = _policy(structured_dropout=True, seed=3, **LONG)
    a.memory_weight(mem.kv, mem.valid, mem.step, 60, training=True)
    state = policy_state_dict(a)
    expect = torch.stack(
        [
            a.memory_weight(mem.kv, mem.valid, mem.step, 60, training=True)[0]
            for _ in range(4)
        ]
    )
    b = _policy(structured_dropout=True, seed=3, **LONG)
    load_policy_state(b, state)
    got = torch.stack(
        [
            b.memory_weight(mem.kv, mem.valid, mem.step, 60, training=True)[0]
            for _ in range(4)
        ]
    )
    assert torch.equal(got, expect)


# --------------------------------------------------------------------------- #
# The span loss: [P7] §4.2 "Loss Computation" (P7:L368-382)
# --------------------------------------------------------------------------- #


def test_the_span_loss_charges_memories_on_the_ramp_only():
    """ "we compute the auxiliary loss on e_i at the same time as negative gradients
    when 0 < m_ti < 1" (P7:L380-382). e = 2, R = 2, so the ramp is 2 < age < 4:
    age 1 -> m = 1 (not charged), age 3 -> m = 0.5 (charged), age 4 -> m = 0 (not
    charged), dead -> not charged."""
    p = _policy(loss_coef=0.5, ramp=2.0)
    mem = _memory(
        batch=2, m=3, written=[[3, 1, 0], [1, 1, -1]], valid=[[1, 1, 1], [1, 1, 0]]
    )
    _, aux = p.memory_weight(mem.kv, mem.valid, mem.step, 4, training=False)
    # Charged: row 0 slot 1, row 1 slots 0 and 1 (all age 3); every layer; e = 2.
    expect = 0.5 * (3 * NL * 2.0) / 2
    torch.testing.assert_close(aux, torch.tensor(expect))
    _, aux_none = p.memory_weight(mem.kv, mem.valid, mem.step, 7, training=False)
    # step 7: ages 4/6/7 and 6/6 -> every m = 0: nothing on the ramp.
    assert float(aux_none) == 0.0


def test_the_span_loss_carries_gradient_only_through_the_span():
    """The ramp condition selects which spans are charged; it is not itself
    differentiated. d(aux)/d(b) = alpha * L * sig'(b) / B for one charged memory."""
    p = _policy(loss_coef=1.0, ramp=2.0, n_layers=1)
    mem = _memory(batch=1, m=2, written=[[1, 3]])  # ages 3 (charged) and 1
    _, aux = p.memory_weight(mem.kv, mem.valid, mem.step, 4, training=False)
    aux.backward()
    # e = 8 sig(b), sig(b) = 0.25, so de/db = 8 * 0.25 * 0.75.
    torch.testing.assert_close(p.bias.grad, torch.tensor([8 * 0.25 * 0.75]))


def test_the_loop_adds_the_span_loss():
    cfg = _tg_cfg()
    ids, mask, lengths = _stream(cfg, steps=8)
    torch.manual_seed(0)
    model = TGModel(cfg).eval()
    nc = _n_cross(cfg)
    lo = run_policy_loop(
        model,
        ids,
        mask,
        lengths,
        ExpireSpanPolicy(_cfg(loss_coef=0.0), cfg.D, nc),
        step_fn=_loss,
    )
    hi = run_policy_loop(
        model,
        ids,
        mask,
        lengths,
        ExpireSpanPolicy(_cfg(loss_coef=1.0), cfg.D, nc),
        step_fn=_loss,
    )
    assert hi.item() > lo.item()


# --------------------------------------------------------------------------- #
# The gradient switch (ADR-0010 q1 -- the setting is the owner's)
# --------------------------------------------------------------------------- #


def _weight_grads(grad_path):
    p = _policy(grad_path=grad_path, loss_coef=0.3)
    _set_weight(p)
    mem = _memory(batch=2, m=4)
    kv = mem.kv.clone().requires_grad_(True)
    w, aux = p.memory_weight(kv, mem.valid, mem.step, 5, training=False)
    total = w.sum() + aux
    if total.requires_grad:
        total.backward()
    return p, kv


def test_grad_path_none_sends_no_gradient_anywhere():
    p, kv = _weight_grads("none")
    assert p.weight.grad is None and p.bias.grad is None
    assert kv.grad is None


def test_predictor_path_trains_the_predictor_but_not_the_gestalt():
    """`predictor`: the LM loss reaches `w, b`; the gestalt -- and so the
    transformer and `W_sent` -- gets nothing through the span."""
    p, kv = _weight_grads("predictor")
    assert p.weight.grad is not None and float(p.weight.grad.abs().sum()) > 0.0
    assert kv.grad is None or float(kv.grad.abs().sum()) == 0.0


def test_through_gestalt_path_reaches_the_gestalt():
    p, kv = _weight_grads("through_gestalt")
    assert float(p.weight.grad.abs().sum()) > 0.0
    assert kv.grad is not None and float(kv.grad.abs().sum()) > 0.0


def _model_run(grad_path, *, loss_coef=0.0):
    cfg = _tg_cfg()
    ids, mask, lengths = _stream(cfg, steps=10)
    torch.manual_seed(0)
    model = TGModel(cfg).eval()
    p = ExpireSpanPolicy(
        _cfg(grad_path=grad_path, loss_coef=loss_coef),
        d_model=cfg.D,
        n_layers=_n_cross(cfg),
    )
    _set_weight(p, scale=1.0)
    loss = run_policy_loop(model, ids, mask, lengths, p, step_fn=_loss)
    loss.backward()
    grads = {n: q.grad.clone() for n, q in model.named_parameters() if q.grad is not None}
    return loss, p, grads


def test_the_lm_loss_reaches_the_span_predictor_through_the_model():
    """The mechanism itself: with `alpha = 0`, any gradient on `w` came from the LM
    loss through the mask in cross-attention -- in more than one layer."""
    _, p, _ = _model_run("predictor", loss_coef=0.0)
    assert p.weight.grad is not None
    assert int((p.weight.grad.abs().sum(dim=1) > 0).sum()) > 1


def test_model_gradients_are_identical_under_none_and_predictor():
    """Neither setting sends span gradient into the transformer, and the forward is
    the same, so every model gradient must agree to the bit -- with `alpha > 0`."""
    l0, p0, g0 = _model_run("none", loss_coef=0.5)
    l1, p1, g1 = _model_run("predictor", loss_coef=0.5)
    assert l0.item() == l1.item()
    assert g0.keys() == g1.keys()
    for n in g0:
        assert torch.equal(g0[n], g1[n]), n
    assert p0.weight.grad is None
    assert float(p1.weight.grad.abs().sum()) > 0.0


def test_through_gestalt_changes_the_model_gradients():
    _, _, g1 = _model_run("predictor", loss_coef=0.5)
    _, _, g2 = _model_run("through_gestalt", loss_coef=0.5)
    assert any(not torch.equal(g1[n], g2[n]) for n in g1)


# --------------------------------------------------------------------------- #
# End-to-end determinism, and the FIFO path is untouched
# --------------------------------------------------------------------------- #


def test_two_runs_under_one_seed_are_bit_equal():
    cfg = _tg_cfg()
    ids, mask, lengths = _stream(cfg, steps=12)

    def once():
        torch.manual_seed(0)
        model = TGModel(cfg).train()
        p = ExpireSpanPolicy(
            _cfg(structured_dropout=True, loss_coef=0.1, seed=4),
            d_model=cfg.D,
            n_layers=_n_cross(cfg),
        )
        _set_weight(p, scale=1.0)
        loss = run_policy_loop(model, ids, mask, lengths, p, step_fn=_loss)
        return loss.item(), list(p.victims)

    a, b = once(), once()
    assert a == b
    assert len(a[1]) > 0  # memory filled and evictions actually happened


def test_fifo_through_the_loop_does_not_call_a_mask():
    """The E0b path: a policy without `memory_weight` runs the model exactly as
    before. `tests/test_reduction.py` is the bit-exact proof; this is the cheap
    tripwire."""
    assert not hasattr(FIFOPolicy(), "memory_weight")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _tg_cfg() -> TGConfig:
    return TGConfig(
        D=32,
        H=2,
        V=64,
        max_sentence_tokens=8,
        max_sentences_in_short_term=4,
        pad_id=60,
        bos_id=61,
        eos_id=62,
        eod_id=63,
    )


def _n_cross(cfg: TGConfig) -> int:
    return sum(b == "C" for b in cfg.block_config)


def _forward(model, cfg, ids, mask, mem, **kw):
    b = ids.shape[0]
    return model(
        ids[:, 0],
        mask[:, 0],
        mem.kv,
        mem.valid,
        torch.zeros(b, cfg.D),
        torch.zeros(b, dtype=torch.bool),
        **kw,
    )


def _stream(cfg: TGConfig, steps: int, seed: int = 3, batch: int = 2):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 59, (batch, steps, cfg.L), generator=g)
    ids[:, :, 0] = cfg.bos_id
    ids[:, :, -1] = cfg.eos_id
    mask = torch.ones(batch, steps, cfg.L, dtype=torch.long)
    lengths = torch.full((batch,), steps)
    return ids, mask, lengths


def _loss(t, out, ids, mask, row_valid):
    logp = torch.log_softmax(out.logits[:, :-1].to(torch.float32), dim=-1)
    picked = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    valid = (mask[:, 1:] == 1) & row_valid.unsqueeze(-1)
    return -(picked * valid).sum() / valid.sum().clamp(min=1)


class _Mem:
    def __init__(self, kv, valid, step):
        self.kv, self.valid, self.step = kv, valid, step


def _memory(batch, m, *, d=D, written=None, valid=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    kv = torch.randn(batch, m, d, generator=g)
    if written is None:
        written = [list(range(m))] * batch
    if valid is None:
        valid = [[1] * m] * batch
    return _Mem(
        kv,
        torch.tensor(valid, dtype=torch.bool),
        torch.tensor(written, dtype=torch.long),
    )


def test_the_memory_helper_matches_init_memory_layout():
    """The helper stands in for `Memory`; keep its dtypes honest."""
    ref = init_memory(2, _tg_cfg())
    mem = _memory(batch=2, m=4, d=32)
    assert mem.kv.dtype == ref.kv.dtype
    assert mem.valid.dtype == ref.valid.dtype
    assert mem.step.dtype == ref.step.dtype
