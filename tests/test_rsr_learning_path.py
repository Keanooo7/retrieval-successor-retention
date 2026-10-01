"""ADR-0009 slice 1 -- the psi-hat learning path, its gates (T1-T13) and the run-1 build.

`observe` -> realized `r_i` (eval mode) -> Monte-Carlo return -> `L_MC` on `phi` only.
Signed items (owner draft `R-2026-09-30-adr0009-signoff`): L1(a), L2, L6(a), L8 clip,
L9, L10(a), L11, L12, L13(a). Run-1 scope (owner draft `R-2026-09-30-rsr-run-1-scope`):
31(b) "one pass over the doc pool", and a frozen-transformer option.

🔴 **ADR-0009 T4 is vacuous under a frozen transformer**, so every isolation test here
(T1, T3, T4) runs the transformer **un-frozen**, in train mode, with dropout live.
The frozen option has its own test, which asserts the weights do not move.

Nothing here trains for evaluation. `nu`, `beta`, `gamma` are MEASURED and owed by E1:
every RSR arm below supplies them as labelled overrides (L12), and the tests that
rely on the registry refusing assert that it does.
"""

from __future__ import annotations

import json
import time

import pytest
import torch
import torch.nn.functional as F

import rsr.constants as C
import rsr.retention.rsr as rsr_mod
import rsr.train.loop as loop
from rsr.baselines.fifo import FIFOPolicy
from rsr.model.tg import TGConfig, TGModel
from rsr.model.tg.model import init_memory
from rsr.model.tg.policy_loop import eval_forward, run_policy_loop
from rsr.retention.mc import mc_returns
from rsr.retention.policy import AttentionTrace, MemoryState
from rsr.retention.rsr import CAPACITY, REDUCTION_SWITCHES, RSRConfig, RSRPolicy
from rsr.retention.value_head import BilinearValueHead

# train()-level shapes: the S0-01 tiny config (`tests/test_train_loop.py`).
TINY = dict(
    d=32,
    steps_per_stream=48,
    batch=2,
    max_tokens=16,
    memory_slots=4,
    iters=1,
    device="cpu",
    beat_every=1,
    ckpt_every=0,
)
#: A learning arm's labelled overrides (ADR-0009 L12). Not values for any run:
#: gamma = 0.5 only so that G differs from r; beta = 1 so L_MC is computed.
OV = dict(nu=0.0, beta=1.0, gamma=0.5, b_enabled=False, shadow_enabled=False)

# run_policy_loop-level shapes.
B, M, S, D = 2, 4, 12, 32


def _cfg() -> TGConfig:
    return TGConfig(
        D=D,
        H=2,
        V=64,
        max_sentence_tokens=8,
        max_sentences_in_short_term=M,
        pad_id=60,
        bos_id=61,
        eos_id=62,
        eod_id=63,
    )


def _model(seed: int = 0) -> TGModel:
    torch.manual_seed(seed)
    model = TGModel(_cfg())
    model.train()  # dropout live: attn 0.2, srep 0.15 -- the training condition
    assert model.cfg.attn_dropout > 0.0
    return model


def _stream(cfg: TGConfig, *, seed: int = 0, batch: int = B, lengths=None):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 59, (batch, S, cfg.L), generator=g)
    ids[..., 0] = cfg.bos_id
    ids[..., -1] = cfg.eos_id
    mask = torch.ones(batch, S, cfg.L, dtype=torch.long)
    lengths = torch.full((batch,), S) if lengths is None else torch.tensor(lengths)
    return ids, mask, lengths


def _lm(t, out, ids, mask, row_valid):
    return loop.lm_loss(out.logits, ids, mask)


def _learned(*, t_warm=0.0, beta=1.0, gamma=0.5) -> RSRConfig:
    return RSRConfig(
        nu=0.0,
        beta=beta,
        gamma=gamma,
        t_warm=t_warm,
        a_max=M,
        psi_override=None,
        b_enabled=False,
        shadow_enabled=False,
    )


def _policy(**kw) -> RSRPolicy:
    return RSRPolicy(
        _learned(**kw), d_model=D, generator=torch.Generator().manual_seed(7)
    )


def _run(model, policy, ids, mask, lengths, *, observe=True, eval_capture=True, **kw):
    torch.manual_seed(1)  # dropout masks
    return run_policy_loop(
        model,
        ids,
        mask,
        lengths,
        policy,
        step_fn=_lm,
        observe=observe,
        eval_capture=eval_capture,
        **kw,
    )


def _ckpt(out_dir) -> dict:
    """The last checkpoint a train() wrote, quarantined or not."""
    paths = sorted(out_dir.rglob("ckpt-*.pt"))
    assert paths, f"no checkpoint under {out_dir}"
    return torch.load(paths[-1], map_location="cpu", weights_only=False)


def _spy_build_policy(monkeypatch) -> list:
    built, real = [], loop.build_policy

    def spy(*a, **k):
        built.append((k, real(*a, **k)))
        return built[-1][1]

    monkeypatch.setattr(loop, "build_policy", spy)
    return built


def _spy_loop_calls(monkeypatch) -> list:
    calls, real = [], loop.run_policy_loop

    def spy(model, ids, mask, lengths, policy, **kw):
        out = real(model, ids, mask, lengths, policy, **kw)
        calls.append({"ids": ids.clone(), "training_after": model.training, **kw})
        return out

    monkeypatch.setattr(loop, "run_policy_loop", spy)
    return calls


# --------------------------------------------------------------------------- #
# Gradient isolation (ADR-0009 §2) -- NON-frozen, train mode, dropout live
# --------------------------------------------------------------------------- #


def test_l_mc_reaches_phi_and_nothing_else():
    """T1. `L_MC.backward()` on a real TGModel: every model gradient None, every
    `phi` gradient nonzero. The named parameters below are the ones the
    prohibition is about (W_sent, cross-attention values, W_O, g_mem)."""
    model = _model()
    policy = _policy()
    _run(model, policy, *_stream(model.cfg))
    l_mc = policy.retention_loss()
    assert l_mc is not None and l_mc.requires_grad
    l_mc.backward()
    names = [n for n, _ in model.named_parameters()]
    for needle in ("srep_head", "cross_attn.value", "attn_out_proj", "memory_gate"):
        assert any(needle in n for n in names), needle
    reached = [n for n, p in model.named_parameters() if p.grad is not None]
    assert reached == [], reached
    for p in policy.head.parameters():
        assert p.grad is not None and float(p.grad.abs().sum()) > 0.0


def test_the_lm_loss_never_reaches_phi():
    """T2. `L_NTP.backward()` alone: `phi` is not in the LM path."""
    model = _model()
    policy = _policy()
    loss = _run(model, policy, *_stream(model.cfg))
    loss.backward()
    assert all(p.grad is None for p in policy.head.parameters())
    assert any(p.grad is not None for p in model.parameters())  # not vacuous


def test_the_policys_own_detach_holds_for_attached_inputs():
    """T3. `retention_loss` fed ATTACHED `s` and `c` directly: still no model grad.

    The loop detaches upstream (`memory_state`, the `no_grad` eval forward), which
    would mask the policy's own detach -- so this test bypasses the loop. Without
    it, dropping the policy's detach is an equivalent mutant (ADR-0009 §2)."""
    model = _model()
    cfg = model.cfg
    n_c = sum(b.block_type == "C" for b in model.blocks)
    gestalts = F.normalize(model.embed.weight[:M], dim=-1)
    context = F.normalize(model.embed.weight[M], dim=-1)
    assert gestalts.requires_grad and context.requires_grad  # attached, on purpose
    policy = _policy()
    g = torch.Generator().manual_seed(3)
    for step in range(M, M + 3):
        slots = MemoryState(
            gestalts=gestalts,
            written_at=torch.arange(M),
            live=torch.ones(M, dtype=torch.bool),
            step=step,
        )
        trace = AttentionTrace(
            alpha=torch.rand(n_c, cfg.H, M, generator=g),
            wo_v=torch.randn(n_c, cfg.H, M, D, generator=g),
            live=torch.ones(M, dtype=torch.bool),
            step=step,
            eval_mode=True,
            gate=torch.ones(n_c),
            context=context,
        )
        policy.observe(slots, trace, step)
    policy.retention_loss().backward()
    reached = [n for n, p in model.named_parameters() if p.grad is not None]
    assert reached == [], reached
    assert all(p.grad is not None for p in policy.head.parameters())


def test_one_step_moves_the_transformer_identically_with_and_without_l_mc(
    tmp_path, monkeypatch
):
    """T4, NON-frozen. One optimizer step through train(): beta = 1 vs L_MC not
    computed (beta = 0, L13). Transformer weights bitwise equal -- this covers the
    clip and the optimizer, not only autograd -- and `phi` is never in the clip set.
    (Under the frozen option this would be vacuous; R-2026-09-30-rsr-run-1-scope.)"""
    clipped: list[set[int]] = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def clip_spy(params, *a, **k):
        params = list(params)
        clipped.append({id(p) for p in params})
        return real_clip(params, *a, **k)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip_spy)
    built = _spy_build_policy(monkeypatch)
    for name, beta in (("on", 1.0), ("off", 0.0)):
        loop.train(
            policy_name="rsr",
            policy_overrides={**OV, "beta": beta},
            out_dir=tmp_path / name,
            **{**TINY, "ckpt_every": 1},
        )
    on, off = _ckpt(tmp_path / "on"), _ckpt(tmp_path / "off")
    assert on["model"].keys() == off["model"].keys()
    moved = [k for k in on["model"] if not torch.equal(on["model"][k], off["model"][k])]
    assert moved == [], moved
    # L_MC did act in the beta = 1 arm (else the equality above is vacuous):
    assert not torch.equal(on["policy"]["__head__"]["W"], off["policy"]["__head__"]["W"])
    assert len(clipped) == 2
    for (_, policy), ids in zip(built, clipped, strict=True):
        assert not ids & {id(p) for p in policy.head.parameters()}, "phi in the clip"


# --------------------------------------------------------------------------- #
# The return (T5)
# --------------------------------------------------------------------------- #


def test_mc_return_matches_the_closed_form():
    """T5. G(t) = sum_{k=0}^{T-1-t} gamma^k r(t+k), checked against the sum."""
    g = torch.Generator().manual_seed(0)
    r = torch.rand(3, 7, generator=g, dtype=torch.float64)
    gamma = 0.7
    want = torch.zeros_like(r)
    for t in range(7):
        for k in range(7 - t):
            want[:, t] += gamma**k * r[:, t + k]
    assert torch.allclose(mc_returns(r, gamma), want, rtol=0, atol=1e-12)
    assert torch.equal(mc_returns(r, gamma)[:, -1], r[:, -1])  # truncated at T


def test_gamma_zero_is_the_present_reward_exactly():
    """T5. gamma = 0 (the separating control arm): G = r, as a case, not 0 ** 0."""
    r = torch.rand(2, 5, generator=torch.Generator().manual_seed(1))
    assert torch.equal(mc_returns(r, 0.0), r)


def test_g_is_keyed_by_written_at_through_a_middle_slot_eviction(monkeypatch):
    """T5. A middle-slot eviction compacts the prefix, so slot INDEX 1 holds key 1
    at t=3 and key 2 at t=4. G must follow the gestalt (written_at), not the index.

    r is injected through `retrieval_demand` so the arithmetic is exact."""
    monkeypatch.setattr(
        rsr_mod, "retrieval_demand", lambda attn, **_: attn.alpha[0, 0].clone()
    )
    policy = RSRPolicy(_learned(gamma=0.5), d_model=2)

    def obs(step, written, r):
        live = torch.ones(3, dtype=torch.bool)
        policy.observe(
            MemoryState(
                gestalts=torch.zeros(3, 2),
                written_at=torch.tensor(written),
                live=live,
                step=step,
            ),
            AttentionTrace(
                alpha=torch.tensor(r).view(1, 1, 3),
                wo_v=torch.zeros(1, 1, 3, 2),
                live=live,
                step=step,
                eval_mode=True,
                context=torch.zeros(2),
            ),
            step,
        )

    obs(3, [0, 1, 2], [0.1, 0.2, 0.3])
    obs(4, [0, 2, 3], [0.4, 0.5, 0.6])  # key 1 evicted from the MIDDLE; 3 written
    obs(5, [0, 2, 3], [0.7, 0.8, 0.9])
    rows = policy.mc_targets()
    got = {
        (int(t), int(k)): float(g)
        for t, k, g in zip(rows["t"], rows["key"], rows["target"], strict=True)
    }
    want = {
        (3, 0): 0.1 + 0.5 * 0.4 + 0.25 * 0.7,
        (3, 1): 0.2,  # evicted at 4: censored (no shadow buffer, L6(a))
        (3, 2): 0.3 + 0.5 * 0.5 + 0.25 * 0.8,
        (4, 0): 0.4 + 0.5 * 0.7,
        (4, 2): 0.5 + 0.5 * 0.8,
        (4, 3): 0.6 + 0.5 * 0.9,
        (5, 0): 0.7,
        (5, 2): 0.8,
        (5, 3): 0.9,
    }
    assert got.keys() == want.keys()
    for k, v in want.items():
        assert got[k] == pytest.approx(v, abs=1e-6), k


def test_the_return_is_truncated_at_each_rows_length():
    """T5. A row of length T_b is observed only at t < T_b, and its return stops
    there: at t = T_b - 1 the target is the present r alone."""
    model = _model()
    policy = _policy()
    ids, mask, lengths = _stream(model.cfg, lengths=[S, S - 4])
    _run(model, policy, ids, mask, lengths)
    rows = policy.mc_targets()
    for b, t_b in enumerate((S, S - 4)):
        mine = rows["row"] == b
        assert int(rows["t"][mine].max()) == t_b - 1, b
        last = mine & (rows["t"] == t_b - 1)
        assert torch.equal(rows["target"][last], rows["r"][last]), b


# --------------------------------------------------------------------------- #
# Eval mode and RNG (T6), rows (T7), warmup (T8)
# --------------------------------------------------------------------------- #


def test_every_trace_reaching_observe_in_train_is_eval_mode(tmp_path, monkeypatch):
    """T6, through train(): every trace observe receives is eval-mode (correction
    20) and carries c_t (L10(a)); the model is in train mode after the loop; and
    the beat records L_MC tagged with k and warm, mean G by t, and L1's cost."""
    seen: list[tuple[bool, bool]] = []
    real = RSRPolicy.observe

    def spy(self, slots, attn, step):
        seen.append((attn.eval_mode, attn.context is not None))
        return real(self, slots, attn, step)

    monkeypatch.setattr(RSRPolicy, "observe", spy)
    calls = _spy_loop_calls(monkeypatch)
    r = loop.train(policy_name="rsr", policy_overrides=OV, out_dir=tmp_path, **TINY)
    assert seen and all(e and c for e, c in seen), seen[:3]
    assert calls[0]["observe"] is True and calls[0]["eval_capture"] is True
    assert all(c["training_after"] for c in calls)
    beats = [
        json.loads(x)
        for x in (tmp_path / "heartbeat.jsonl").read_text().splitlines()
        if '"beat"' in x
    ]
    beat = beats[-1]
    assert beat["loss_mc"] is not None and beat["loss_mc"] > 0
    assert beat["train_step"] == 0 and beat["mc"]["warm"] is True  # 0 < T_warm
    assert beat["mc"]["g_mean_by_t"] and beat["eval_forward_s"] > 0
    assert r["final"]["eval_forward_s"] > 0


def test_the_eval_forward_draws_no_rng_and_restores_the_mode():
    """T6. The second forward consumes nothing from the global or a data
    generator, returns an eval-mode capture, and leaves the model training."""
    model = _model()
    cfg = model.cfg
    ids, mask, _ = _stream(cfg)
    mem = init_memory(B, cfg)
    mem.kv.normal_()
    mem.valid[:] = True
    bos = torch.zeros(B, cfg.D)
    data = torch.Generator().manual_seed(5)
    g0, d0 = torch.get_rng_state(), data.get_state()
    ev = eval_forward(
        model,
        ids[:, 0],
        mask[:, 0],
        mem,
        bos,
        torch.zeros(B, dtype=torch.bool),
        capture=True,
    )
    assert torch.equal(torch.get_rng_state(), g0)
    assert torch.equal(data.get_state(), d0)
    assert model.training is True
    assert ev.cap is not None and ev.cap.eval_mode is True
    assert not ev.srep.requires_grad and ev.seconds > 0


def test_the_training_forward_is_unchanged_by_the_eval_forward():
    """T6. Same seed, FIFO: the loss with the eval forward on is the loss without
    it, bit for bit -- no dropout mask moved."""
    model = _model()
    stream = _stream(model.cfg)
    off = _run(model, FIFOPolicy(), *stream, observe=False, eval_capture=False)
    on = _run(model, FIFOPolicy(), *stream, observe=False, eval_capture=True)
    assert on.item() == off.item()


def test_the_decision_and_the_regression_read_one_context():
    """L2 (signed): the c_t `select_eviction` reads is the c_t `observe` was handed,
    the eval gestalt -- one tensor for both, never the train-mode srep."""
    model = _model()
    policy = _policy()
    decided, observed = {}, {}
    real_sel, real_obs = policy.select_eviction, policy.observe

    def sel(slots, context, step):
        decided[(slots.row, step)] = context.clone()
        return real_sel(slots, context, step)

    def obs(slots, attn, step):
        observed[(slots.row, step)] = attn.context.clone()
        return real_obs(slots, attn, step)

    policy.select_eviction, policy.observe = sel, obs
    _run(model, policy, *_stream(model.cfg))
    assert decided, "memory never filled: the test would be vacuous"
    for key, ctx in decided.items():
        assert torch.equal(ctx, observed[key]), key


def test_observe_refuses_a_trace_without_its_context():
    """L10(a): an r_i with no c_t from the same forward is refused."""
    policy = _policy()
    live = torch.ones(M, dtype=torch.bool)
    st = MemoryState(
        gestalts=torch.zeros(M, D), written_at=torch.arange(M), live=live, step=M
    )
    tr = AttentionTrace(
        alpha=torch.ones(1, 1, M),
        wo_v=torch.ones(1, 1, M, D),
        live=live,
        step=M,
        eval_mode=True,
        gate=torch.ones(1),
    )
    with pytest.raises(ValueError, match="context"):
        policy.observe(st, tr, M)


def test_two_rows_keep_two_buffers():
    """T7, the row-0 bug class. A batch of two different streams yields each row's
    own targets: equal to what the row produces run alone, and not each other's.
    (Eval mode, so batch size does not change dropout masks.)"""
    model = _model()
    model.eval()
    ids, mask, lengths = _stream(model.cfg)
    both = _policy()
    _run(model, both, ids, mask, lengths)
    rows = both.mc_targets()
    assert set(rows["row"].tolist()) == {0, 1}
    per_row = []
    for b in range(B):
        alone = _policy()
        _run(model, alone, ids[b : b + 1], mask[b : b + 1], lengths[b : b + 1])
        mine = rows["row"] == b
        want = alone.mc_targets()
        assert torch.equal(rows["t"][mine], want["t"]), b
        assert torch.equal(rows["key"][mine], want["key"]), b
        assert torch.allclose(rows["target"][mine], want["target"], atol=1e-5), b
        per_row.append(rows["target"][mine])
    assert not torch.allclose(per_row[0], per_row[1])


def test_warmup_evicts_fifo_and_phi_still_trains():
    """T8, §3.4: k < T_warm -> every eviction fifo_warmup AND phi gets a nonzero
    gradient (passive training); k >= T_warm -> scored."""
    model = _model()
    warm = _policy(t_warm=2.0)
    warm.set_train_step(1)
    _run(model, warm, *_stream(model.cfg))
    assert set(warm.attribution_counts()) == {"fifo_warmup"}
    warm.retention_loss().backward()
    assert all(float(p.grad.abs().sum()) > 0 for p in warm.head.parameters())
    assert warm.last_mc_stats["warm"] is True and warm.last_mc_stats["train_step"] == 1
    scored = _policy(t_warm=2.0)
    scored.set_train_step(2)
    _run(model, scored, *_stream(model.cfg))
    assert "fifo_warmup" not in scored.attribution_counts()
    assert scored.attribution_counts().get("psi", 0) > 0


# --------------------------------------------------------------------------- #
# phi's RNG (T10, L11)
# --------------------------------------------------------------------------- #


def test_fifo_and_rsr_arms_see_the_same_first_batch(tmp_path, monkeypatch):
    """T10. Same seed: the RSR arm's first stream batch is the FIFO arm's. phi is
    built from its own CPU generator (L11), so it draws nothing from `gen`."""
    calls = _spy_loop_calls(monkeypatch)
    loop.train(policy_name="fifo", out_dir=tmp_path / "fifo", **TINY)
    loop.train(policy_name="rsr", policy_overrides=OV, out_dir=tmp_path / "rsr", **TINY)
    assert torch.equal(calls[0]["ids"], calls[1]["ids"])


def test_phi_is_initialised_from_its_own_seeded_stream(tmp_path, monkeypatch):
    """L11: the head train() builds is BilinearValueHead(d) drawn from
    `phi_generator(seed)` -- (seed, "phi"), CPU -- and the global RNG is untouched."""
    built = _spy_build_policy(monkeypatch)
    loop.train(
        policy_name="rsr", policy_overrides=OV, out_dir=tmp_path, **{**TINY, "iters": 0}
    )
    ((_, policy),) = built
    want = BilinearValueHead(TINY["d"], generator=loop.phi_generator(0)[0])
    for (n, p), (_, q) in zip(
        policy.head.named_parameters(), want.named_parameters(), strict=True
    ):
        assert torch.equal(p.detach().cpu(), q.detach()), n
    s0, s1 = loop.phi_generator(0)[1], loop.phi_generator(1)[1]
    assert s0 != s1 and s0 == loop.phi_generator(0)[1]
    assert loop.phi_generator(0)[0].device.type == "cpu"


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is not available on this host: the RSR-arm-on-MPS half of T10 "
    "DID NOT RUN here and is not a pass",
)
def test_the_rsr_arm_builds_and_steps_on_mps(tmp_path):
    """T10, L11's second defect: phi built from an MPS generator raised
    `Expected a 'cpu' device type for generator`. Built from its CPU stream, then
    moved, the arm constructs and takes a step on MPS."""
    r = loop.train(
        policy_name="rsr",
        policy_overrides=OV,
        out_dir=tmp_path,
        **{**TINY, "device": "mps"},
    )
    assert r["steps_done"] == 1 and r["final"]["loss_mc"] is not None


# --------------------------------------------------------------------------- #
# The overrides path (T11, L12)
# --------------------------------------------------------------------------- #


def _build(**ov):
    flag = ov.pop("_ablation", False)
    return loop.build_policy(
        "rsr", d_model=8, steps_per_epoch=4.0, overrides=ov, no_warmup_ablation=flag
    )


def test_the_ledger_is_empty_for_the_e1_constants():
    """The premise of every override test: E1 has not logged nu, beta, gamma."""
    for name in ("nu", "beta", "gamma"):
        with pytest.raises(C.UnmeasuredConstant, match="E1"):
            C.REGISTRY.get(name, "synthetic")


def test_override_arms_build_on_an_empty_ledger():
    """T11. Every MEASURED field overridden: the arm builds, nothing recorded."""
    policy = _build(**OV)
    cfg = policy.config
    assert (cfg.nu, cfg.beta, cfg.gamma, cfg.b_enabled, cfg.shadow_enabled) == (
        0.0,
        1.0,
        0.5,
        False,
        False,
    )
    assert cfg.t_warm == 4.0  # T_warm_epochs (FROZEN 1.0) x steps_per_epoch
    with pytest.raises(C.UnmeasuredConstant):
        C.REGISTRY.get("beta", "synthetic")  # still unrecorded


def test_a_field_left_unoverridden_still_raises():
    """T11. Overrides are visible departures, not defaults (D-1)."""
    ov = {k: v for k, v in OV.items() if k != "beta"}
    with pytest.raises(C.UnmeasuredConstant, match="beta"):
        _build(**ov)


def test_b_enabled_false_is_required():
    """L12: no ProtectionBias exists, so b must be off and say so."""
    with pytest.raises(ValueError, match="b_enabled=False is required"):
        _build(**{k: v for k, v in OV.items() if k != "b_enabled"})


def test_shadow_enabled_false_must_be_passed_explicitly():
    """L6(a)/L12: the dataclass default is True; the arm must say False."""
    with pytest.raises(ValueError, match="shadow_enabled=False must be passed"):
        _build(**{k: v for k, v in OV.items() if k != "shadow_enabled"})


def test_shadow_enabled_true_is_refused_by_the_policy():
    """T11, L6(a): a switch that is set and does nothing is worse than no switch."""
    cfg = RSRConfig(
        nu=0.0,
        beta=1.0,
        gamma=0.5,
        t_warm=0.0,
        a_max=M,
        psi_override=None,
        b_enabled=False,
        shadow_enabled=True,
    )
    with pytest.raises(NotImplementedError, match="shadow"):
        RSRPolicy(cfg, d_model=D)


def test_a_t_warm_override_needs_the_ablation_declaration():
    """T11, L12: otherwise an override decides 31(b) by the back door."""
    with pytest.raises(ValueError, match="no-warmup ablation"):
        _build(**OV, t_warm=0.0)
    with pytest.raises(ValueError, match=r"t_warm=0\.0"):
        _build(**OV, t_warm=3.0, _ablation=True)
    with pytest.raises(ValueError, match=r"t_warm=0\.0"):
        _build(**OV, _ablation=True)
    assert _build(**OV, t_warm=0.0, _ablation=True).config.t_warm == 0.0


def test_unknown_overrides_and_fifo_overrides_are_refused():
    with pytest.raises(ValueError, match="unknown RSR override"):
        _build(**OV, tau=0.3)
    with pytest.raises(ValueError, match="FIFO"):
        loop.build_policy("fifo", d_model=8, steps_per_epoch=4.0, overrides={"nu": 0.0})


def test_overrides_are_stamped_into_frozen_hash_and_label(tmp_path):
    """T11. The label and the hash differ per override set, and the overrides are
    in the frozen config (so in the run_id's hash)."""
    kw = {**TINY, "iters": 0}
    a = loop.train(policy_name="rsr", policy_overrides=OV, out_dir=tmp_path / "a", **kw)
    b = loop.train(
        policy_name="rsr",
        policy_overrides={**OV, "beta": 0.5},
        out_dir=tmp_path / "b",
        **kw,
    )
    assert a["arm_label"] == "rsr[b=off,beta=1,gamma=0.5,nu=0,shadow=off]"
    assert b["arm_label"] == "rsr[b=off,beta=0.5,gamma=0.5,nu=0,shadow=off]"
    assert a["config"]["policy_overrides"] == OV
    assert a["config"]["arm_label"] == a["arm_label"]
    assert a["config_hash"] != b["config_hash"] and a["run_id"] != b["run_id"]
    assert a["config"]["optimizer"]["eps"] == loop.ADAMW_EPS
    hdr = json.loads((tmp_path / "a" / "heartbeat.jsonl").read_text().splitlines()[0])
    assert (
        hdr["optimizer"]["eps"] == loop.ADAMW_EPS and hdr["arm_label"] == a["arm_label"]
    )


def test_the_adamw_eps_is_torchs_default_and_recorded():
    """L7/L8 after B1: eps left at the AdamW default, read from torch, never typed."""
    import inspect

    default = inspect.signature(torch.optim.AdamW).parameters["eps"].default
    assert default == loop.ADAMW_EPS
    assert loop.optimizer_record()["eps"] == default
    assert loop.optimizer_record()["phi_in_transformer_clip"] is False


def test_cli_overrides_ablation_and_freeze_reach_train(monkeypatch, tmp_path):
    seen: dict = {}
    monkeypatch.setattr(
        "rsr.train.loop.train",
        lambda **kw: seen.update(kw) or {"run_id": "x", "liveness": {"band": "live"}},
    )
    argv = [
        "--out-dir",
        str(tmp_path),
        "--policy",
        "rsr",
        "--policy-override",
        "beta=0.1",
        "--policy-override",
        "b_enabled=false",
        "--policy-override",
        "psi_override=neg_age",
        "--policy-override",
        "a_max=4",
        "--no-warmup-ablation",
        "--freeze-transformer",
    ]
    assert loop.main(argv) == 0
    assert seen["policy_overrides"] == {
        "beta": 0.1,
        "b_enabled": False,
        "psi_override": "neg_age",
        "a_max": 4,
    }
    assert seen["no_warmup_ablation"] is True and seen["freeze_transformer"] is True
    seen.clear()
    assert loop.main(["--out-dir", str(tmp_path)]) == 0
    assert seen["policy_overrides"] is None and seen["freeze_transformer"] is False


# --------------------------------------------------------------------------- #
# beta = 0 (T12, L13) and the reduction (T13)
# --------------------------------------------------------------------------- #


def test_beta_zero_leaves_phi_bitwise_at_init(tmp_path, monkeypatch):
    """T12, L13(a): beta = 0 means L_MC is NOT computed -- observe is off, phi's
    gradients stay None, AdamW skips phi (its decoupled decay too), and phi is
    bitwise its init after a step."""
    calls = _spy_loop_calls(monkeypatch)
    loop.train(
        policy_name="rsr",
        policy_overrides={**OV, "beta": 0.0},
        out_dir=tmp_path,
        **{**TINY, "ckpt_every": 1},
    )
    assert calls[0]["observe"] is False and calls[0]["eval_capture"] is True
    init = BilinearValueHead(TINY["d"], generator=loop.phi_generator(0)[0]).state_dict()
    head = _ckpt(tmp_path)["policy"]["__head__"]
    for k, v in init.items():
        assert torch.equal(head[k].cpu(), v), k


def test_the_reduction_through_train_has_no_head_no_observe_and_no_loss(
    tmp_path, monkeypatch
):
    """T13. §3.7's switches as L12 overrides (t_warm = 0 is the declared no-warmup
    value, correction 16): no head, observe and the eval forward off,
    retention_loss() None -- and the loss is the FIFO arm's, bit for bit."""
    red = {
        k: (TINY["memory_slots"] if v is CAPACITY else v)
        for k, v in REDUCTION_SWITCHES.items()
    }
    built = _spy_build_policy(monkeypatch)
    calls = _spy_loop_calls(monkeypatch)
    r = loop.train(
        policy_name="rsr",
        policy_overrides=red,
        no_warmup_ablation=True,
        out_dir=tmp_path / "red",
        **TINY,
    )
    f = loop.train(policy_name="fifo", out_dir=tmp_path / "fifo", **TINY)
    policy = built[0][1]
    assert policy.head is None and policy.retention_loss() is None
    assert calls[0]["observe"] is False and calls[0]["eval_capture"] is False
    assert policy.attribution_counts().get("neg_age", 0) > 0  # the score path ran
    assert r["final"]["loss"] == f["final"]["loss"]


# --------------------------------------------------------------------------- #
# 31(b): the epoch, derived from the pool
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_docs", [8, 16])
def test_steps_per_epoch_is_derived_from_the_pool(tmp_path, monkeypatch, n_docs):
    """31(b), R-2026-09-30-rsr-run-1-scope (owner draft): one epoch is one pass
    over the document pool, n_docs / batch optimizer steps -- from the pool train()
    built, never typed and never `iters`. T_warm (§3.4: one epoch) follows, and is
    reported as a fraction of total epochs."""
    built = _spy_build_policy(monkeypatch)
    r = loop.train(
        policy_name="rsr",
        policy_overrides=OV,
        n_documents=n_docs,
        out_dir=tmp_path,
        **{**TINY, "iters": 3},
    )
    spe = n_docs / TINY["batch"]
    assert built[0][0]["steps_per_epoch"] == spe
    ep = r["config"]["epoch"]
    assert ep["pool_documents"] == n_docs and ep["steps_per_epoch"] == spe
    assert ep["t_warm_steps"] == spe and ep["t_warm_epochs"] == 1.0
    assert ep["total_epochs"] == 3 / spe
    assert ep["t_warm_fraction_of_total_epochs"] == spe / 3


def test_an_rsr_arm_on_a_stream_is_refused(tmp_path):
    """A never-repeating stream has no pool, so no epoch under the 31(b) ruling."""
    with pytest.raises(ValueError, match="POOL"):
        loop.train(
            policy_name="rsr",
            policy_overrides=OV,
            stream=loop.DocumentStream(offset=4160, stride=16),
            out_dir=tmp_path,
            **TINY,
        )


# --------------------------------------------------------------------------- #
# The frozen-transformer option (R-2026-09-30-rsr-run-1-scope D2(b))
# --------------------------------------------------------------------------- #


def test_the_frozen_option_is_recorded_and_moves_no_transformer_weight(
    tmp_path, monkeypatch
):
    """D2(b): no L_NTP update. Recorded in frozen/hash/label; after two steps every
    transformer parameter is bitwise its init, phi has learned, and the optimizer
    holds phi's group only."""
    inits: list[dict] = []
    real_model = loop.TGModel

    def model_spy(cfg):
        m = real_model(cfg)
        inits.append({k: v.detach().clone() for k, v in m.state_dict().items()})
        return m

    monkeypatch.setattr(loop, "TGModel", model_spy)
    groups: list = []
    real_groups = loop.build_param_groups
    monkeypatch.setattr(
        loop,
        "build_param_groups",
        lambda *a, **k: groups.append(real_groups(*a, **k)) or groups[-1],
    )
    kw = {**TINY, "iters": 2, "ckpt_every": 2}
    r = loop.train(
        policy_name="rsr",
        policy_overrides=OV,
        freeze_transformer=True,
        out_dir=tmp_path / "frozen",
        **kw,
    )
    ck = _ckpt(tmp_path / "frozen")
    moved = [k for k, v in inits[0].items() if not torch.equal(ck["model"][k], v)]
    assert moved == [], moved
    assert [g["name"] for g in groups[0]] == ["value_head"]
    phi0 = BilinearValueHead(TINY["d"], generator=loop.phi_generator(0)[0])
    assert not torch.equal(ck["policy"]["__head__"]["W"], phi0.W.detach())
    assert r["config"]["freeze_transformer"] is True
    assert r["arm_label"].endswith("+frozen")
    live = loop.train(
        policy_name="rsr",
        policy_overrides=OV,
        out_dir=tmp_path / "live",
        **{**TINY, "iters": 0},
    )
    assert live["config"]["freeze_transformer"] is False
    assert live["config_hash"] != r["config_hash"]


def test_a_frozen_fifo_arm_runs_without_an_optimizer(tmp_path):
    """The run-1 FIFO arm on a frozen substrate has nothing to optimise: it runs,
    builds no optimizer, and is labelled."""
    r = loop.train(
        policy_name="fifo",
        freeze_transformer=True,
        out_dir=tmp_path,
        **{**TINY, "ckpt_every": 1},
    )
    assert r["steps_done"] == 1 and r["arm_label"] == "fifo+frozen"
    assert r["config"]["freeze_transformer"] is True
    assert _ckpt(tmp_path)["optimizer"] is None
    hdr = json.loads((tmp_path / "heartbeat.jsonl").read_text().splitlines()[0])
    assert hdr["optimizer"] is None


# --------------------------------------------------------------------------- #
# L1(a)'s cost, measured (signed: "its cost is measured in the PR")
# --------------------------------------------------------------------------- #


def test_l1_eval_forward_cost_is_measured(capsys):
    """Measures, does not gate: wall time of one stream with and without the second
    forward, and with the full learning path. Timing is not asserted (it is the
    machine's, not the code's); the printed line is the report's L1 figure, at
    THESE shapes only -- run 1's beats log `eval_forward_s` at its own."""
    model = _model()
    stream = _stream(model.cfg)
    reps = 3

    def wall(fn) -> float:
        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return best

    base = wall(
        lambda: _run(model, FIFOPolicy(), *stream, observe=False, eval_capture=False)
    )
    timings: dict = {}
    l1 = wall(
        lambda: _run(
            model,
            FIFOPolicy(),
            *stream,
            observe=False,
            eval_capture=True,
            timings=timings,
        )
    )

    def learn():
        p = _policy()
        _run(model, p, *stream)
        p.retention_loss()

    full = wall(learn)
    # The denominator a training step actually pays: forward AND backward.
    step = wall(
        lambda: _run(
            model, FIFOPolicy(), *stream, observe=False, eval_capture=False
        ).backward()
    )
    assert timings["eval_forward_s"] > 0
    with capsys.disabled():
        print(
            f"\nL1-COST shapes B={B} S={S} M={M} D={D} N={model.cfg.N} cpu "
            f"threads={torch.get_num_threads()} best-of-{reps}: stream fwd "
            f"{base:.4f}s; +eval fwd {l1:.4f}s ({(l1 / base - 1) * 100:+.1f}% of "
            f"fwd); +observe+L_MC {full:.4f}s; fwd+bwd step {step:.4f}s, so the "
            f"eval fwd adds {(l1 - base) / step * 100:+.1f}% of a fwd+bwd step"
        )
