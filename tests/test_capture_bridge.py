"""The §3.2.1 capture bridge: a real forward pass -> `AttentionTrace` -> `r_i`.

`src/rsr/retention/reward.py` was 160 lines with 11 passing tests and **zero
callers in `src/`**, because the model physically could not produce its input:
`alpha` came out `[B, H, Q_tok, M]` instead of `[L, H, M]`, `W_O v` never existed
as a tensor at all (the forward pass contracts over slots *before* applying
`attn_out_proj`), and `g_mem` was an `nn.Parameter` nobody surfaced.

Every test here drives a real `TGModel` forward. None builds a hand-made trace --
that is what `tests/test_reward.py` is for, and the gap between the two is the
thing this file exists to close.

🔴 **The load-bearing test is `test_W_O_is_load_bearing_in_the_capture`.** Dropping
`W_O` from the bridge is **shape-compatible** with `reward.contribution` (the norm
is over the last axis, whether that axis is `D` or `Dh`), so it fails silently and
reintroduces exactly the confound D-6 adopted norm-weighting to remove.
`scripts/mutation_battery.py` carries the mutation that proves this file catches
it.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from rsr.baselines.fifo import FIFOPolicy
from rsr.model.tg import TGConfig, TGModel
from rsr.model.tg.model import init_memory
from rsr.model.tg.policy_loop import (
    Q_TOK_COLLAPSE,
    cross_capture,
    run_policy_loop,
    trace_for_row,
)
from rsr.retention.policy import AttentionTrace, MemoryState
from rsr.retention.reward import (
    TrainModeTrace,
    contribution,
    contribution_per_layer,
    retrieval_demand,
)

B, M = 3, 5


def _cfg(**over) -> TGConfig:
    base = dict(
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
    base.update(over)
    return TGConfig(**base)


def _sentence(cfg: TGConfig, seed: int = 1):
    """One step of input, with a PAD tail on row 0 so the mask is not all-ones."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 59, (B, cfg.L), generator=g)
    ids[:, 0] = cfg.bos_id
    ids[:, -1] = cfg.eos_id
    mask = torch.ones(B, cfg.L, dtype=torch.long)
    mask[0, 5:] = 0
    return ids, mask


def _memory(cfg: TGConfig, seed: int = 2):
    """A partly-filled memory: row 0 has 4 of 5 live, row 1 has 2, row 2 has none.

    Underfull rows are the case §3.2.1's rescale exists for, and an all-empty row
    is the one where `_masked_softmax` returns a finite uniform distribution on
    purpose -- the artefact the bridge has to zero rather than read as retrieval.
    """
    mem = init_memory(B, cfg, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    mem.kv = torch.randn(B, M, cfg.D, generator=g)
    mem.valid = torch.tensor(
        [[1, 1, 1, 1, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.bool
    )
    return mem


def _capture(cfg=None, *, train=False, seed=1):
    cfg = cfg or _cfg()
    torch.manual_seed(0)
    model = TGModel(cfg)
    model.train(train)
    mem = _memory(cfg)
    ids, mask = _sentence(cfg, seed=seed)
    bos = torch.zeros(B, cfg.D)
    bos_valid = torch.zeros(B, dtype=torch.bool)
    with torch.no_grad():
        out = model(ids, mask, mem.kv, mem.valid, bos, bos_valid, capture=True)
        cap = cross_capture(model, out, mem.kv, mask, mem.valid)
    return model, mem, ids, mask, out, cap


# --- bar 1: r_i end to end on a real forward pass ---------------------------- #


def test_retrieval_demand_runs_on_a_real_forward_pass():
    """Bar item 1. Not a hand-built fixture: `TGModel` -> `AttentionTrace` -> `r_i`.

    The §3.2.1 rescale means a row's `r_i` sums to `n_live / M`, so the assertion
    is a real property of the target and not merely "it returned a tensor".
    """
    cfg = _cfg()
    _, mem, _, _, _, cap = _capture(cfg)
    for row, n_live in ((0, 4), (1, 2)):
        tr = trace_for_row(cap, mem.valid, row, step=0)
        r = retrieval_demand(tr, n_live=n_live, capacity=M)
        assert r.shape == (M,)
        assert torch.isfinite(r).all()
        assert r.sum().item() == pytest.approx(n_live / M, abs=1e-5)
        assert (r > 0).sum().item() == n_live, "every live slot took some demand"
        assert torch.equal(r[~mem.valid[row]], torch.zeros(M - n_live))


def test_the_trace_has_the_shapes_the_reward_module_declares():
    """`[L, H, M]` and `[L, H, M, d_model]` -- and `L` is **six**, not twelve (D-E).

    TG alternates `S,C,S,C,...`, so a twelve-entry profile would be six real rows
    and six zeros, and the zeros would be read as a depth finding.
    """
    cfg = _cfg()
    model, mem, _, _, _, cap = _capture(cfg)
    n_cross = sum(1 for b in model.blocks if b.block_type == "C")
    assert n_cross == 6
    tr = trace_for_row(cap, mem.valid, 0, step=0)
    assert tr.alpha.shape == (6, cfg.H, M)
    assert tr.wo_v.shape == (6, cfg.H, M, cfg.D)
    assert tr.gate.shape == (6,)
    assert contribution_per_layer(tr).shape == (6, M)


def test_the_captured_gate_is_the_models_memory_gate():
    """Correction 17 / D-E: `g_mem` is in the formula, and it is *this* parameter."""
    cfg = _cfg()
    model, mem, ids, mask, _, _ = _capture(cfg)
    # `memory_gate_init` is 1.0, so an untrained model has six identical gates and
    # gated == raw by arithmetic rather than by capture. [P2] App. C measures the
    # gates GROWING over training and LARGER IN DEEPER LAYERS, so a depth-
    # stratified set is the case that has to work -- and it is the case that makes
    # `r_i` non-stationary between epoch 1 and epoch 12 (correction 17).
    with torch.no_grad():
        for i, block in enumerate(b for b in model.blocks if b.block_type == "C"):
            block.memory_gate.fill_(0.2 + 0.2 * i)
    bos, bosv = torch.zeros(B, cfg.D), torch.zeros(B, dtype=torch.bool)
    with torch.no_grad():
        out = model(ids, mask, mem.kv, mem.valid, bos, bosv, capture=True)
        cap = cross_capture(model, out, mem.kv, mask, mem.valid)

    gates = [b.memory_gate for b in model.blocks if b.block_type == "C"]
    assert torch.equal(cap.gate, torch.stack(gates).detach())
    assert cap.gate.tolist() == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    tr = trace_for_row(cap, mem.valid, 0, step=0)
    assert not torch.allclose(contribution(tr, gated=True), contribution(tr, gated=False))


# --- bar 3 / D-6: W_O is load bearing ---------------------------------------- #


def test_W_O_is_load_bearing_in_the_capture():
    """🔴 Bar item 3. Perturb `attn_out_proj` and `r_i` **must** move -- and must
    move *through the capture*, not around it.

    §3.2.1: *"`W_O` is not optional... dropping it reintroduces the confound the
    norm-weighting was adopted to remove."*

    ⚠️ **The obvious version of this test is vacuous, and it was written that way
    first.** Perturbing *every* cross-attention layer's `attn_out_proj` changes
    `r_i` even when the bridge captures raw `v`, because layer `l`'s output enters
    the residual stream and moves layer `l+1`'s **attention**. Run against the
    `W_O dropped from the capture path` mutation, that version stayed green: it
    was detecting "the model changed", not "`W_O` is inside `r_i`".

    So the perturbation is confined to the **last** cross-attention block. Nothing
    downstream of it attends to memory, so `alpha` is provably unchanged on all
    six layers -- asserted below -- and any movement in `r_i` has exactly one
    route left: `wo_v`. The intervention is head-asymmetric because `W_O` is where
    head-specific rescaling lives (D-6); a uniform scale would cancel in the
    share-of-live rescale and prove nothing.
    """
    cfg = _cfg()
    model, mem, ids, mask, _, cap_before = _capture(cfg)
    r_before = retrieval_demand(
        trace_for_row(cap_before, mem.valid, 0, 0), n_live=4, capacity=M
    )

    cross_idx = [i for i, b in enumerate(model.blocks) if b.block_type == "C"]
    last = model.blocks[cross_idx[-1]]
    assert cross_idx[-1] == len(model.blocks) - 1, (
        "the last cross block must be the last block, or a later block could "
        "carry the perturbation back into an attention distribution"
    )
    with torch.no_grad():
        last.cross_attn.attn_out_proj.kernel[0] *= 40.0
        last.cross_attn.attn_out_proj.kernel[1] *= 0.02

    bos, bosv = torch.zeros(B, cfg.D), torch.zeros(B, dtype=torch.bool)
    with torch.no_grad():
        out = model(ids, mask, mem.kv, mem.valid, bos, bosv, capture=True)
        cap_after = cross_capture(model, out, mem.kv, mask, mem.valid)

    assert torch.equal(cap_before.alpha, cap_after.alpha), (
        "the perturbation leaked into an attention distribution, so this test "
        "cannot attribute a change in r_i to W_O"
    )
    r_after = retrieval_demand(
        trace_for_row(cap_after, mem.valid, 0, 0), n_live=4, capacity=M
    )
    assert not torch.allclose(r_before, r_after, atol=1e-4), (
        "`W_O` is not in the capture path: rescaling one output projection per "
        "head, with every alpha provably unchanged, left r_i untouched -- so the "
        "capture is norm-weighted raw attention and the reward is the confound "
        "D-6 removed"
    )


def test_wo_v_is_the_projected_value_not_the_raw_value():
    """The shape check the silent failure would survive, made explicit.

    Dropping `W_O` yields `[L, H, M, Dh]`, and `reward.contribution` norms over the
    last axis either way -- so nothing downstream would notice.
    """
    cfg = _cfg(D=32, H=2)
    model, mem, _, _, _, cap = _capture(cfg)
    assert cfg.head_dim != cfg.D, "the shapes must differ or this check is vacuous"
    assert cap.wo_v.shape[-1] == cfg.D
    block = next(b for b in model.blocks if b.block_type == "C")
    v = block.cross_attn.value(mem.kv)
    expected = torch.einsum("bmhk,hkd->bhmd", v, block.cross_attn.attn_out_proj.kernel)
    assert torch.allclose(cap.wo_v[0], expected, atol=1e-6)


# --- ADR-0008: the Q_tok collapse -------------------------------------------- #


def test_the_collapse_reproduces_the_real_increment():
    """🔴 ADR-0008's premise, checked rather than asserted.

    `W_O v_i` does not depend on the query position, so the cross-attention
    increment slot `i` makes over the whole sentence is exactly
    `(sum_q alpha_q) . W_O v_i`. If that identity does not hold, "sum" is not
    algebra and the ADR's argument collapses with it.
    """
    cfg = _cfg()
    model, mem, _, mask, out, cap = _capture(cfg)
    row = 0
    q_real = mask[row] != 0
    blocks = [b for b in model.blocks if b.block_type == "C"]
    for li, (block, att) in enumerate(zip(blocks, out.cross_attention, strict=True)):
        with torch.no_grad():
            v = block.cross_attn.value(mem.kv)
            o = block.cross_attn.attn_out_proj(torch.einsum("bhqm,bmhk->bqhk", att, v))
            lhs = o[row][q_real].sum(0) - q_real.sum() * (
                block.cross_attn.attn_out_proj.bias
            )
        rhs = torch.einsum("hm,hmd->d", cap.alpha[li, row], cap.wo_v[li, row])
        assert torch.allclose(lhs, rhs, atol=1e-5), f"layer {li}"


def test_mean_and_sum_collapse_give_the_same_r_i():
    """ADR-0008. `mean = sum / Q_real`, one scalar for every `(l, h, i)`, so it
    cancels in `share_i = raw_i / sum_j raw_j`. The brief expected this axis to
    change `r_i`; measured, it does not -- only the unnormalised diagnostic moves.
    """
    cfg = _cfg()
    _, mem, _, mask, _, cap = _capture(cfg)
    row, n_real = 0, int((mask[0] != 0).sum())
    tr = trace_for_row(cap, mem.valid, row, 0)
    tr_mean = dataclasses.replace(tr, alpha=tr.alpha / n_real)

    r_sum = retrieval_demand(tr, n_live=4, capacity=M)
    r_mean = retrieval_demand(tr_mean, n_live=4, capacity=M)
    assert torch.allclose(r_sum, r_mean, atol=1e-6)
    # ... and the unnormalised profile differs by exactly that scalar.
    assert torch.allclose(contribution(tr), contribution(tr_mean) * n_real, rtol=1e-5)


def test_eos_only_collapse_is_a_different_measurement():
    """ADR-0008's real fork. If this ever stops differing, the ADR's "open,
    scheduled" alternative has become a no-op and the E0d row can be dropped."""
    cfg = _cfg()
    _, mem, ids, _, out, cap = _capture(cfg)
    row = 0
    eos = int((ids[row] == cfg.eos_id).to(torch.int32).argmax())
    alpha_eos = torch.stack([a[row, :, eos, :] for a in out.cross_attention])
    alpha_eos = torch.where(
        mem.valid[row].view(1, 1, -1), alpha_eos, torch.zeros_like(alpha_eos)
    )
    tr = trace_for_row(cap, mem.valid, row, 0)
    r_sum = retrieval_demand(tr, n_live=4, capacity=M)
    r_eos = retrieval_demand(
        dataclasses.replace(tr, alpha=alpha_eos), n_live=4, capacity=M
    )
    assert not torch.allclose(r_sum, r_eos, atol=1e-3)


def test_pad_query_positions_do_not_contribute():
    """PAD **keys** are masked; PAD **queries** are not, so every pad position
    still emits a full distribution over slots. Including them would credit slots
    for attention paid by positions whose residual stream is discarded."""
    cfg = _cfg()
    model, mem, ids, mask, _, _ = _capture(cfg)
    bos, bosv = torch.zeros(B, cfg.D), torch.zeros(B, dtype=torch.bool)
    with torch.no_grad():
        out = model(ids, mask, mem.kv, mem.valid, bos, bosv, capture=True)
        cap_masked = cross_capture(model, out, mem.kv, mask, mem.valid)
        cap_all = cross_capture(model, out, mem.kv, torch.ones_like(mask), mem.valid)
    assert int((mask[0] == 0).sum()) > 0, "row 0 must have PAD or this is vacuous"
    assert not torch.allclose(cap_masked.alpha[:, 0], cap_all.alpha[:, 0])
    # Rows with no PAD are untouched, which is what makes the difference the mask.
    assert torch.allclose(cap_masked.alpha[:, 1], cap_all.alpha[:, 1])
    assert cap_masked.n_real_query_tokens.tolist() == [5, cfg.L, cfg.L]


def test_an_unimplemented_collapse_raises_rather_than_falling_back():
    cfg = _cfg()
    model, mem, _, mask, out, _ = _capture(cfg)
    with pytest.raises(ValueError, match="ADR-0008"):
        cross_capture(model, out, mem.kv, mask, mem.valid, collapse="eos_only")
    assert Q_TOK_COLLAPSE == "sum_over_real_query_tokens"


# --- the artefacts a capture must not read as retrieval ---------------------- #


def test_an_empty_memory_row_credits_nobody():
    """`_masked_softmax` returns a finite **uniform** distribution on an all-masked
    row by design (the output is zeroed separately by `row_has_mem`). Read
    naively, that is 1/M of "retrieval" on five slots that do not exist."""
    cfg = _cfg()
    _, mem, _, _, _, cap = _capture(cfg)
    assert not mem.valid[2].any()
    tr = trace_for_row(cap, mem.valid, 2, step=0)
    assert torch.equal(tr.alpha, torch.zeros_like(tr.alpha))
    assert torch.equal(contribution(tr), torch.zeros(M))
    assert torch.equal(retrieval_demand(tr, n_live=0, capacity=M), torch.zeros(M))


def test_a_train_mode_capture_is_refused_downstream():
    """Correction 20 / D-F. `attn_dropout = 0.2` in the reference config, so in
    train mode a slot can score zero demand because a mask fell on it. The bridge
    does not decide -- it reports `eval_mode` honestly and `reward` refuses."""
    cfg = _cfg(attn_dropout=0.2)
    _, mem, _, _, _, cap = _capture(cfg, train=True)
    assert cap.eval_mode is False
    with pytest.raises(TrainModeTrace):
        retrieval_demand(trace_for_row(cap, mem.valid, 0, 0), n_live=4, capacity=M)


def test_dropout_free_train_mode_is_still_an_eval_capture():
    """`eval_mode` is a claim about whether dropout ran, not about `model.training`.
    With `attn_dropout = 0` there is no mask to fall, so refusing would be a false
    negative -- and a false negative here is a capture nobody can collect."""
    cfg = _cfg(attn_dropout=0.0)
    _, mem, _, _, _, cap = _capture(cfg, train=True)
    assert cap.eval_mode is True
    retrieval_demand(trace_for_row(cap, mem.valid, 0, 0), n_live=4, capacity=M)


def test_a_forward_without_capture_is_refused_not_guessed():
    cfg = _cfg()
    torch.manual_seed(0)
    model = TGModel(cfg).eval()
    mem = _memory(cfg)
    ids, mask = _sentence(cfg)
    bos, bosv = torch.zeros(B, cfg.D), torch.zeros(B, dtype=torch.bool)
    with torch.no_grad():
        out = model(ids, mask, mem.kv, mem.valid, bos, bosv, capture=False)
    with pytest.raises(ValueError, match="capture=False"):
        cross_capture(model, out, mem.kv, mask, mem.valid)


def test_the_trace_carries_no_gradient_to_the_transformer():
    """§3.3: only `phi` receives gradient. A trace holding live autograd history
    is a path from the retention target into `W_sent` and the transformer."""
    cfg = _cfg()
    torch.manual_seed(0)
    model = TGModel(cfg).eval()
    mem = _memory(cfg)
    ids, mask = _sentence(cfg)
    bos, bosv = torch.zeros(B, cfg.D), torch.zeros(B, dtype=torch.bool)
    out = model(ids, mask, mem.kv, mem.valid, bos, bosv, capture=True)  # no no_grad
    assert out.cross_attention[0].requires_grad, "the forward must still build a graph"
    cap = cross_capture(model, out, mem.kv, mask, mem.valid)
    for t in (cap.alpha, cap.wo_v, cap.gate):
        assert not t.requires_grad


# --- bar 4: observe() is reached --------------------------------------------- #


class RecordingPolicy:
    """FIFO, plus a log of every `observe` / `on_write` / `reset` it received."""

    name = "recording"

    def __init__(self) -> None:
        self.observed: list[tuple[int, int, bool]] = []
        self.resets = 0
        self._fifo = FIFOPolicy()

    def select_eviction(self, slots: MemoryState, context, step: int) -> int:
        return self._fifo.select_eviction(slots, context, step)

    def observe(self, slots: MemoryState, attn: AttentionTrace, step: int) -> None:
        assert isinstance(attn, AttentionTrace)
        assert attn.alpha.shape[0] == 6  # six cross layers, not twelve (D-E)
        assert torch.equal(attn.live, slots.live), (
            "the trace and the memory state must describe the same step: `live` "
            "is what the attention was masked with"
        )
        self.observed.append((step, slots.n_live, attn.eval_mode))

    def on_write(self, slots: MemoryState, slot: int, step: int) -> None:
        return None

    def reset(self) -> None:
        self.resets += 1


def _stream(cfg: TGConfig, steps: int = 4, seed: int = 3):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 59, (B, steps, cfg.L), generator=g)
    ids[:, :, 0] = cfg.bos_id
    ids[:, :, -1] = cfg.eos_id
    mask = torch.ones(B, steps, cfg.L, dtype=torch.long)
    lengths = torch.tensor([steps, steps, steps - 1])
    return ids, mask, lengths


def _loss(t, out, ids, mask, row_valid):
    return out.logits.float().pow(2).mean()


def test_observe_is_called_once_per_live_row_per_step():
    """🔴 Bar item 4. Deleting the `policy.observe(...)` call site in
    `run_policy_loop` turns this red, and nothing else in the suite notices --
    which is precisely why `git grep '\\.observe(' -- src/` returned zero hits
    while `reward.py` sat there with eleven passing tests and no caller."""
    cfg = _cfg()
    torch.manual_seed(0)
    model = TGModel(cfg).eval()
    ids, mask, lengths = _stream(cfg, steps=4)
    policy = RecordingPolicy()
    run_policy_loop(model, ids, mask, lengths, policy, step_fn=_loss, observe=True)
    # rows 0 and 1 run 4 steps, row 2 runs 3 -> 11 observations.
    assert len(policy.observed) == 11
    assert [s for s, _, _ in policy.observed].count(3) == 2
    assert all(ev for _, _, ev in policy.observed)


def test_observe_is_off_by_default():
    """Bar item 2's first half. A policy that raises in `observe` must survive a
    default run, or "off by default" is a claim rather than a property."""

    class Exploding(RecordingPolicy):
        def observe(self, slots, attn, step):
            raise AssertionError("observe must not be called when observe=False")

    cfg = _cfg()
    torch.manual_seed(0)
    model = TGModel(cfg).eval()
    ids, mask, lengths = _stream(cfg)
    run_policy_loop(model, ids, mask, lengths, Exploding(), step_fn=_loss)


def test_observe_sees_the_pre_write_memory():
    """The trace describes attention paid **before** this step's write. Moving the
    call below `write_at` hands the policy a `live` mask belonging to the next
    step, and the resulting `r_i` credits a slot that was not there."""
    cfg = _cfg()
    torch.manual_seed(0)
    model = TGModel(cfg).eval()
    ids, mask, lengths = _stream(cfg, steps=4)
    policy = RecordingPolicy()
    run_policy_loop(model, ids, mask, lengths, policy, step_fn=_loss, observe=True)
    first = [n for s, n, _ in policy.observed if s == 0]
    assert first == [0, 0, 0], "step 0 attends over an empty memory"
    second = [n for s, n, _ in policy.observed if s == 1]
    assert second == [1, 1, 1], "step 1 sees the one gestalt step 0 wrote"


def test_the_stream_boundary_resets_the_policy():
    """[P2] fact 2, and defect D-1's mechanism: §3.5's `b` resets with the memory.
    `run_policy_loop` initialises a fresh memory, so the call IS a stream boundary
    and the policy has to be told. `src/rsr/train/loop.py` never told it."""
    cfg = _cfg()
    torch.manual_seed(0)
    model = TGModel(cfg).eval()
    ids, mask, lengths = _stream(cfg)
    policy = RecordingPolicy()
    run_policy_loop(model, ids, mask, lengths, policy, step_fn=_loss)
    assert policy.resets == 1
    run_policy_loop(model, ids, mask, lengths, policy, step_fn=_loss)
    assert policy.resets == 2


def test_observing_does_not_change_the_loss():
    """Capture is observation. If it moves a number, it is a defect (the brief's
    "Do NOT"). Same seed, same loss, bit for bit."""
    cfg = _cfg()
    torch.manual_seed(0)
    model = TGModel(cfg).eval()
    ids, mask, lengths = _stream(cfg)
    a = float(
        run_policy_loop(model, ids, mask, lengths, RecordingPolicy(), step_fn=_loss)
    )
    b = float(
        run_policy_loop(
            model, ids, mask, lengths, RecordingPolicy(), step_fn=_loss, observe=True
        )
    )
    assert a == b
