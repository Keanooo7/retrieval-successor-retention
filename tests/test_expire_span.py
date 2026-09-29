"""Expire-Span [P7] -- the referendum baseline (spec §5.4, D-4, falsifier 6).

Written before the implementation. What these tests pin, in the order the build
brief (2026-09-29) lists it:

* **forward / eviction semantics** -- the span `e_i = L·sigmoid(w·h_i + b)`, the
  remaining span `r_i = e_i - age_i`, the soft mask `m_i = clamp(1 + r_i/R, 0, 1)`,
  the mask entering cross-attention, and eviction of the least-remaining slot when
  memory is full;
* **span reset on admission** (gauntlet 0.4) -- a new occupant's span is its own,
  whatever the slot held before and however old that was;
* **the gradient switch** -- `none` / `predictor` / `through_gestalt`. Which one
  the referendum uses is an owner decision (ADR-0010, PROPOSED); these tests pin
  what each setting does, not which is right;
* **structured dropout** ([P7] requires it; B-3), whole slots, training only;
* **determinism** -- construction draws nothing from the global RNG (the E0b trap,
  `tests/test_reduction.py` docstring), and two runs under one seed are bit-equal.

Paper-derived formulas are UNVERIFIED: the [P7] source is not in
`~/research-corpus`, and ADR-0010 lists each such detail.
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


def _cfg(**over) -> ExpireSpanConfig:
    base = dict(
        max_span=4.0,
        ramp=2.0,
        loss_coef=0.0,
        dropout=0.0,
        grad_path="predictor",
        init_span_fraction=0.5,
        seed=0,
    )
    base.update(over)
    return ExpireSpanConfig(**base)


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
    """`R`, `alpha`, `L`, the dropout rate, the init and the gradient path are all
    unregistered in `src/rsr/constants.py` and open under B-3. A default here would
    be an agent choosing a hyperparameter of the referendum."""
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
        {"dropout": 1.0},
        {"dropout": -0.1},
        {"loss_coef": -1.0},
        {"init_span_fraction": 0.0},
        {"init_span_fraction": 1.0},
    ],
)
def test_the_config_refuses_out_of_range_values(over):
    with pytest.raises(ValueError):
        _cfg(**over)


def test_the_gradient_paths_are_exactly_three():
    assert GRAD_PATHS == ("none", "predictor", "through_gestalt")


def test_expire_span_satisfies_the_protocol():
    assert isinstance(ExpireSpanPolicy(_cfg(), d_model=D), RetentionPolicy)


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
    ExpireSpanPolicy(_cfg(dropout=0.5), d_model=D)
    assert torch.equal(expect, torch.randn(4))


def test_structured_dropout_does_not_consume_the_global_rng():
    p = ExpireSpanPolicy(_cfg(dropout=0.5), d_model=D)
    mem = _memory(batch=2, m=4)
    torch.manual_seed(0)
    expect = torch.randn(4)
    torch.manual_seed(0)
    p.memory_weight(mem.kv, mem.valid, mem.step, 6, training=True)
    assert torch.equal(expect, torch.randn(4))


def test_zero_weight_init_gives_every_gestalt_the_same_initial_span():
    p = ExpireSpanPolicy(_cfg(max_span=10.0, init_span_fraction=0.25), d_model=D)
    e = p.spans(torch.randn(5, D))
    torch.testing.assert_close(e, torch.full((5,), 2.5))


# --------------------------------------------------------------------------- #
# Forward semantics
# --------------------------------------------------------------------------- #


def test_spans_lie_in_zero_to_max_span():
    p = ExpireSpanPolicy(_cfg(max_span=6.0), d_model=D)
    _set_weight(p, scale=50.0)
    e = p.spans(torch.randn(64, D, generator=torch.Generator().manual_seed(1))).detach()
    assert float(e.min()) >= 0.0 and float(e.max()) <= 6.0
    assert float(e.max()) > 5.0 and float(e.min()) < 1.0  # both ends reachable


def test_the_mask_is_the_clamped_ramp():
    """`m = clamp(1 + r/R, 0, 1)`: 1 while unexpired, linear over the last `R`
    steps past expiry, 0 after."""
    p = ExpireSpanPolicy(_cfg(ramp=2.0), d_model=D)
    r = torch.tensor([3.0, 0.0, -0.5, -1.0, -2.0, -5.0])
    torch.testing.assert_close(
        p.mask_from_remaining(r), torch.tensor([1.0, 1.0, 0.75, 0.5, 0.0, 0.0])
    )


def test_memory_weight_is_zero_on_dead_slots_and_follows_age():
    p = ExpireSpanPolicy(_cfg(max_span=4.0, ramp=2.0, init_span_fraction=0.5), d_model=D)
    mem = _memory(batch=1, m=4, written=[[0, 1, 2, -1]], valid=[[1, 1, 1, 0]])
    w, _ = p.memory_weight(mem.kv, mem.valid, mem.step, 4, training=False)
    # e = 2 for every slot. ages 4,3,2 -> r = -2,-1,0 -> m = 0, 0.5, 1; dead -> 0.
    torch.testing.assert_close(w, torch.tensor([[0.0, 0.5, 1.0, 0.0]]))


def test_a_zero_weight_slot_receives_no_attention_and_rows_renormalise():
    """The mask enters cross-attention as `a_i ∝ m_i·exp(s_i)`. A slot at `m = 0`
    gets exactly zero attention in every layer and head, and the rest still sum
    to one."""
    cfg = _tg_cfg()
    torch.manual_seed(0)
    model = TGModel(cfg).eval()
    ids, mask, _ = _stream(cfg, steps=1)
    mem = _memory(batch=ids.shape[0], m=cfg.M, d=cfg.D, seed=4)
    weight = torch.ones(ids.shape[0], cfg.M)
    weight[:, 1] = 0.0
    weight[:, 2] = 0.25
    bos = torch.zeros(ids.shape[0], cfg.D)
    out = model(
        ids[:, 0],
        mask[:, 0],
        mem.kv,
        mem.valid,
        bos,
        torch.zeros(ids.shape[0]).bool(),
        capture=True,
        mem_weight=weight,
    )
    for att in out.cross_attention:
        assert float(att[..., 1].abs().max()) == 0.0
        torch.testing.assert_close(att.sum(-1), torch.ones_like(att.sum(-1)))
    plain = model(
        ids[:, 0],
        mask[:, 0],
        mem.kv,
        mem.valid,
        bos,
        torch.zeros(ids.shape[0]).bool(),
        capture=True,
    )
    assert not torch.allclose(plain.cross_attention[0], out.cross_attention[0])


def test_an_all_ones_weight_reproduces_the_unweighted_forward():
    cfg = _tg_cfg()
    torch.manual_seed(0)
    model = TGModel(cfg).eval()
    ids, mask, _ = _stream(cfg, steps=1)
    mem = _memory(batch=ids.shape[0], m=cfg.M, d=cfg.D, seed=4)
    bos = torch.zeros(ids.shape[0], cfg.D)
    nb = torch.zeros(ids.shape[0]).bool()
    a = model(ids[:, 0], mask[:, 0], mem.kv, mem.valid, bos, nb)
    b = model(
        ids[:, 0],
        mask[:, 0],
        mem.kv,
        mem.valid,
        bos,
        nb,
        mem_weight=torch.ones(ids.shape[0], cfg.M),
    )
    torch.testing.assert_close(a.logits, b.logits, rtol=1e-5, atol=1e-5)


def test_the_loop_passes_the_mask_to_the_model():
    """If `run_policy_loop` computed the mask and never handed it to the model, the
    LM loss would never see the spans and nothing would train them."""
    cfg = _tg_cfg()
    ids, mask, lengths = _stream(cfg, steps=10)
    torch.manual_seed(0)
    model = TGModel(cfg).eval()
    # Spans so short that every slot is fully expired by the time it is read:
    # with the mask applied the model sees no memory at all.
    short = ExpireSpanPolicy(
        _cfg(max_span=0.5, ramp=0.1, init_span_fraction=0.01), d_model=cfg.D
    )
    loss_short = run_policy_loop(model, ids, mask, lengths, short, step_fn=_loss)
    loss_fifo = run_policy_loop(model, ids, mask, lengths, FIFOPolicy(), step_fn=_loss)
    assert loss_short.item() != loss_fifo.item()


# --------------------------------------------------------------------------- #
# Eviction semantics
# --------------------------------------------------------------------------- #


def test_expire_span_evicts_the_slot_with_least_remaining_span():
    """Not the oldest: a young slot with a short span dies before an old slot with
    a long one."""
    p = ExpireSpanPolicy(_cfg(max_span=10.0), d_model=D)
    gestalts = torch.zeros(4, D)
    gestalts[:, 0] = torch.tensor([5.0, -5.0, 5.0, 5.0])  # slot 1: short span
    with torch.no_grad():
        p.weight.zero_()
        p.weight[0] = 1.0
    st = _state([0, 5, 2, 3], step=6, gestalts=gestalts)
    e = p.spans(gestalts)
    r = e - (6 - st.written_at).to(e.dtype)
    assert int(r.argmin()) == 1
    assert p.select_eviction(st, torch.zeros(D), 6) == 1
    assert FIFOPolicy().select_eviction(st, torch.zeros(D), 6) == 0  # it differs


def test_eviction_ignores_dead_slots():
    p = ExpireSpanPolicy(_cfg(), d_model=D)
    st = _state([0, 1, 2, 3], step=9, live=[0, 1, 1, 1])
    assert p.select_eviction(st, torch.zeros(D), 9) == 1


def test_at_zero_weight_init_expire_span_evicts_exactly_fifo():
    """Every gestalt has the same span at init, so the least remaining is the
    oldest. Expire-Span starts as TG and moves only as far as training moves it."""
    p, f = ExpireSpanPolicy(_cfg(), d_model=D), FIFOPolicy()
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
    p = ExpireSpanPolicy(_cfg(max_span=4.0, ramp=2.0), d_model=D)
    mem = _memory(batch=1, m=3, written=[[0, 1, 2]], valid=[[1, 1, 1]])
    w_old, _ = p.memory_weight(mem.kv, mem.valid, mem.step, 7, training=False)
    assert float(w_old[0, 0]) == 0.0  # slot 0's tenant, age 7, has expired
    # Slot 0 is rewritten at step 7 by a new gestalt; at step 8 it has age 1.
    mem.step[0, 0] = 7
    mem.kv[0, 0] = torch.randn(D, generator=torch.Generator().manual_seed(9))
    w_new, _ = p.memory_weight(mem.kv, mem.valid, mem.step, 8, training=False)
    assert float(w_new[0, 0]) == 1.0
    st = MemoryState(
        gestalts=mem.kv[0], written_at=mem.step[0], live=mem.valid[0], step=8
    )
    assert p.select_eviction(st, torch.zeros(D), 8) != 0


def test_on_write_observe_and_reset_leave_the_decision_unchanged():
    """Expire-Span keeps no cross-step per-slot state: the span is a function of the
    occupant's gestalt and the age of `written_at`, both owned by the memory. The
    hooks are no-ops and say so; this pins that they stay no-ops."""
    p = ExpireSpanPolicy(_cfg(), d_model=D)
    _set_weight(p)
    st = _state([0, 1, 2, 3], step=6)
    before = p.select_eviction(st, torch.zeros(D), 6)
    p.on_write(st, 2, 6)
    p.reset()
    assert p.select_eviction(st, torch.zeros(D), 6) == before


# --------------------------------------------------------------------------- #
# Structured dropout
# --------------------------------------------------------------------------- #


def test_structured_dropout_drops_whole_slots_in_training_only():
    """[P7] requires structured dropout (RESEARCH-CONTEXT B-3). Whole slots, one
    draw per `(row, slot)` shared by every layer and head; never at eval."""
    p = ExpireSpanPolicy(_cfg(dropout=0.5, max_span=100.0), d_model=D)
    mem = _memory(batch=8, m=6)
    w_eval, _ = p.memory_weight(mem.kv, mem.valid, mem.step, 6, training=False)
    assert bool((w_eval == 1.0).all())  # long spans: nothing expired
    w_train, _ = p.memory_weight(mem.kv, mem.valid, mem.step, 6, training=True)
    assert w_train.shape == (8, 6)
    dropped = w_train == 0.0
    assert 0 < int(dropped.sum()) < dropped.numel()
    assert bool(((w_train == 0.0) | (w_train == 1.0)).all())


def test_structured_dropout_is_reproducible_under_its_seed():
    mem = _memory(batch=8, m=6)

    def draw(seed):
        p = ExpireSpanPolicy(_cfg(dropout=0.5, max_span=100.0, seed=seed), d_model=D)
        return p.memory_weight(mem.kv, mem.valid, mem.step, 6, training=True)[0]

    assert torch.equal(draw(1), draw(1))
    assert not torch.equal(draw(1), draw(2))


def test_the_checkpoint_round_trips_the_dropout_generator():
    """A resume that restarted the dropout stream would replay the same masks."""
    mem = _memory(batch=8, m=6)
    a = ExpireSpanPolicy(_cfg(dropout=0.5, max_span=100.0, seed=3), d_model=D)
    a.memory_weight(mem.kv, mem.valid, mem.step, 6, training=True)
    state = policy_state_dict(a)
    expect = a.memory_weight(mem.kv, mem.valid, mem.step, 6, training=True)[0]
    b = ExpireSpanPolicy(_cfg(dropout=0.5, max_span=100.0, seed=3), d_model=D)
    load_policy_state(b, state)
    assert torch.equal(
        b.memory_weight(mem.kv, mem.valid, mem.step, 6, training=True)[0], expect
    )


# --------------------------------------------------------------------------- #
# The auxiliary span loss
# --------------------------------------------------------------------------- #


def test_the_span_loss_charges_each_admission_once():
    """`alpha · Σ e_i` over gestalts admitted at the previous step, per row: each
    gestalt is charged once, at the first step it can be read."""
    p = ExpireSpanPolicy(_cfg(loss_coef=0.5, max_span=4.0), d_model=D)
    mem = _memory(
        batch=2, m=3, written=[[0, 1, 2], [0, 2, -1]], valid=[[1, 1, 1], [1, 1, 0]]
    )
    _, aux = p.memory_weight(mem.kv, mem.valid, mem.step, 3, training=False)
    e = p.spans(mem.kv).detach()
    expect = 0.5 * (e[0, 2] + e[1, 1]) / 2
    torch.testing.assert_close(aux, expect)
    _, aux_none = p.memory_weight(mem.kv, mem.valid, mem.step, 5, training=False)
    assert float(aux_none) == 0.0


def test_the_loop_adds_the_span_loss():
    cfg = _tg_cfg()
    ids, mask, lengths = _stream(cfg, steps=8)
    torch.manual_seed(0)
    model = TGModel(cfg).eval()
    lo = run_policy_loop(
        model,
        ids,
        mask,
        lengths,
        ExpireSpanPolicy(_cfg(loss_coef=0.0), cfg.D),
        step_fn=_loss,
    )
    hi = run_policy_loop(
        model,
        ids,
        mask,
        lengths,
        ExpireSpanPolicy(_cfg(loss_coef=1.0), cfg.D),
        step_fn=_loss,
    )
    assert hi.item() > lo.item()


# --------------------------------------------------------------------------- #
# The gradient switch (ADR-0010, PROPOSED -- the setting is the owner's)
# --------------------------------------------------------------------------- #


def _weight_grads(grad_path):
    p = ExpireSpanPolicy(_cfg(grad_path=grad_path, loss_coef=0.3), d_model=D)
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
    p = ExpireSpanPolicy(_cfg(grad_path=grad_path, loss_coef=loss_coef), d_model=cfg.D)
    _set_weight(p, scale=1.0)
    loss = run_policy_loop(model, ids, mask, lengths, p, step_fn=_loss)
    loss.backward()
    grads = {n: q.grad.clone() for n, q in model.named_parameters() if q.grad is not None}
    return loss, p, grads


def test_the_lm_loss_reaches_the_span_predictor_through_the_model():
    """The mechanism itself: with `alpha = 0`, any gradient on `w` came from the LM
    loss through the mask in cross-attention."""
    _, p, _ = _model_run("predictor", loss_coef=0.0)
    assert p.weight.grad is not None and float(p.weight.grad.abs().sum()) > 0.0


def test_model_gradients_are_identical_under_none_and_predictor():
    """Neither setting sends span gradient into the transformer, and the forward is
    the same, so every model gradient must agree to the bit -- with `alpha > 0` too."""
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
        p = ExpireSpanPolicy(_cfg(dropout=0.3, loss_coef=0.1, seed=4), d_model=cfg.D)
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
