"""Expire-Span wired into `train()` as a referendum arm (spec §5.4, falsifier 6).

ADR-0010 q1, q2 and q9 as ruled in R-2026-10-06-expire-span-q1-q2-q9 (draft):
`grad_path` is required with no default (q1 = predictor for the referendum), every
config field is a labelled override the referendum PREREG declares (q2), and `w, b`
take exactly phi's optimizer group and settings (q9).

What is pinned here:

* `POLICIES` / `build_policy` -- registered; every `ExpireSpanConfig` field but `seed`
  is required as an override, `grad_path` included; nothing extra is accepted;
  construction draws nothing from the global RNG (the E0b trap, `test_reduction.py`'s
  docstring).
* `build_param_groups` via `train()` -- `w, b` land in phi's `value_head` group with
  phi's lr, betas, decoupled decay and eps.
* W-1 -- the frozen arm's eval-mode model (run 1's O5) does NOT switch off the span
  loss or structured dropout at training steps; an eval caller still gets neither.
* the FIFO and psiC arms are unchanged by the wiring: run ids, labels and the global
  RNG state after training are the values `ed6302a` (pre-wiring) produced.
* the ruling's PM note -- "with the model frozen, `predictor` and `through_gestalt`
  update `w, b` identically" -- measured: they do NOT, and why.
"""

from __future__ import annotations

import hashlib

import pytest
import torch

from rsr.baselines.expire_span import ExpireSpanConfig, ExpireSpanPolicy
from rsr.model.tg import TGConfig, TGModel
from rsr.model.tg.policy_loop import run_policy_loop
from rsr.mup.param_groups import VALUE_HEAD_GROUP, build_param_groups
from rsr.train import checkpoint as ck
from rsr.train import loop
from rsr.train.loop import ES_OVERRIDE_FIELDS, POLICIES, build_policy, train

ES_OV = dict(
    max_span=8.0,
    ramp=2.0,
    loss_coef=1e-3,
    structured_dropout=True,
    grad_path="predictor",
    init_span_fraction=0.1,
)
PSIC = dict(b_enabled=False, shadow_enabled=False, nu=0.0, beta=1.0, gamma=0.9)
TINY = dict(
    d=32, steps_per_stream=48, batch=2, memory_slots=4, device="cpu", n_documents=8
)
FROZEN_EVAL = dict(freeze_transformer=True, frozen_model_mode="eval")


def _es(**over):
    return build_policy(
        "expire_span",
        d_model=16,
        steps_per_epoch=None,
        overrides={**ES_OV, **over},
        seed=0,
        n_cross_layers=3,
    )


# --------------------------------------------------------------------------- #
# Registration and build_policy
# --------------------------------------------------------------------------- #


def test_expire_span_is_a_registered_policy_and_a_cli_choice(tmp_path):
    """The CLI accepts the name, and with no overrides refuses rather than defaulting
    `grad_path` (or anything else)."""
    assert "expire_span" in POLICIES
    argv = ["--policy", "expire_span", "--iters", "0", "--device", "cpu"]
    with pytest.raises(ValueError, match="grad_path"):
        loop.main([*argv, "--out-dir", str(tmp_path)])


def test_every_config_field_but_seed_is_a_required_override():
    assert set(ES_OVERRIDE_FIELDS) == {
        "max_span",
        "ramp",
        "loss_coef",
        "structured_dropout",
        "grad_path",
        "init_span_fraction",
    }


@pytest.mark.parametrize("missing", ["grad_path", "max_span", "loss_coef"])
def test_build_policy_refuses_a_missing_override_grad_path_included(missing):
    """q1 is the owner's: no default `grad_path`, and no default anything else."""
    ov = {k: v for k, v in ES_OV.items() if k != missing}
    with pytest.raises(ValueError, match=missing):
        build_policy(
            "expire_span",
            d_model=16,
            steps_per_epoch=None,
            overrides=ov,
            seed=0,
            n_cross_layers=3,
        )


def test_build_policy_refuses_an_unknown_override_and_an_invalid_grad_path():
    with pytest.raises(ValueError, match="unknown"):
        _es(beta=1.0)
    with pytest.raises(ValueError, match="grad_path"):
        _es(grad_path="lm")


def test_build_policy_needs_the_seed_and_the_layer_count():
    for kw in ({"seed": None, "n_cross_layers": 3}, {"seed": 0, "n_cross_layers": None}):
        with pytest.raises(ValueError, match="seed"):
            build_policy(
                "expire_span", d_model=16, steps_per_epoch=None, overrides=ES_OV, **kw
            )


def test_build_policy_refuses_the_no_warmup_declaration():
    with pytest.raises(ValueError, match="no warmup"):
        build_policy(
            "expire_span",
            d_model=16,
            steps_per_epoch=None,
            overrides=ES_OV,
            no_warmup_ablation=True,
            seed=0,
            n_cross_layers=3,
        )


def test_build_policy_builds_the_config_it_was_given():
    p = _es()
    assert isinstance(p, ExpireSpanPolicy)
    assert p.cfg == ExpireSpanConfig(**ES_OV, seed=0)
    assert p.n_layers == 3 and tuple(p.weight.shape) == (3, 16)


def test_building_expire_span_draws_nothing_from_the_global_rng():
    """`test_reduction.py`'s RNG trap: a draw here would shift data order and dropout
    for every arm after it."""
    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()
    _es()
    assert torch.equal(torch.get_rng_state(), before)


# --------------------------------------------------------------------------- #
# train(): q9's optimizer group, the objective, W-1
# --------------------------------------------------------------------------- #


def _train_es(tmp_path, name, *, iters=2, ov=None, **kw):
    out = tmp_path / name
    r = train(
        **TINY,
        iters=iters,
        out_dir=out,
        ckpt_every=iters,
        policy_name="expire_span",
        policy_overrides={**ES_OV, **(ov or {})},
        srep_norm_reg_weight=0.0,  # run 1's; so `loss - loss_lm` is the span loss
        **{**FROZEN_EVAL, **kw},
    )
    return r, ck.load(out / f"ckpt-{iters:06d}.pt", restore_rng=False)


def test_w_and_b_take_exactly_phis_optimizer_group_and_settings(tmp_path):
    """q9 (ruled): the `value_head` group, lr = base_lr / m, AdamW (0.9, 0.95),
    decoupled decay 0.01, eps 1e-8 -- the group a psiC arm's phi gets at the same d."""
    _r, es = _train_es(tmp_path, "es", iters=1)
    out = tmp_path / "psic"
    train(
        **TINY,
        iters=1,
        out_dir=out,
        ckpt_every=1,
        policy_name="rsr",
        policy_overrides=PSIC,
        **FROZEN_EVAL,
    )
    rsr = ck.load(out / "ckpt-000001.pt", restore_rng=False)
    (g_es,) = es["optimizer"]["param_groups"]
    (g_rsr,) = rsr["optimizer"]["param_groups"]
    assert g_es["name"] == g_rsr["name"] == VALUE_HEAD_GROUP
    for k in ("lr", "betas", "weight_decay", "eps"):
        assert g_es[k] == g_rsr[k], k
    assert g_es["lr"] == pytest.approx(1e-3 / (32 / 128))
    assert len(g_es["params"]) == 2  # w and b, nothing of the frozen model


def test_build_param_groups_puts_the_policy_where_phi_goes():
    cfg = _tg_cfg()
    model = TGModel(cfg)
    model.requires_grad_(False)
    pol = ExpireSpanPolicy(ExpireSpanConfig(**ES_OV, seed=0), cfg.D, 2)
    (g,) = build_param_groups(model, pol, base_lr=1e-3, d_model=cfg.D, base_width=128)
    assert g["name"] == VALUE_HEAD_GROUP
    assert {id(p) for p in g["params"]} == {id(pol.weight), id(pol.bias)}


def test_the_frozen_arm_trains_w_b_and_leaves_the_model_bit_equal(tmp_path):
    r, es = _train_es(tmp_path, "es", iters=2)
    pol = ExpireSpanPolicy(
        ExpireSpanConfig(**ES_OV, seed=0), 32, r["config"]["expire_span_n_layers"]
    )
    assert not torch.equal(es["policy"]["bias"], pol.bias.detach())
    assert not torch.equal(es["policy"]["weight"], pol.weight.detach())
    src = tmp_path / "fifo"
    train(**TINY, iters=1, out_dir=src, ckpt_every=1, policy_name="fifo", **FROZEN_EVAL)
    fifo = ck.load(src / "ckpt-000001.pt", restore_rng=False)
    for k, v in fifo["model"].items():  # the same init at the same seed: frozen
        assert torch.equal(es["model"][k], v), k


def test_the_config_and_overrides_are_stamped_into_the_hash_and_label(tmp_path):
    r, _ = _train_es(tmp_path, "es", iters=1)
    assert r["config"]["policy"] == "expire_span"
    assert r["config"]["policy_overrides"] == dict(sorted(ES_OV.items()))
    assert r["config"]["expire_span"] == {**ES_OV, "seed": 0}
    assert r["arm_label"].startswith("expire_span[grad_path=predictor,")
    assert r["arm_label"].endswith("+frozen[eval]")
    r2, _ = _train_es(tmp_path, "es2", iters=1, ov={"loss_coef": 2e-3})
    assert r2["config_hash"] != r["config_hash"]


def test_w1_eval_mode_frozen_training_still_charges_the_span_loss(tmp_path):
    """W-1: under run 1's O5 the frozen model is in eval mode at every training step.
    The span loss must still be charged (`loss` > `loss_lm` with alpha > 0) and the
    structured-dropout generator must still advance."""
    r, es = _train_es(tmp_path, "es", iters=2)
    f = r["final"]
    assert f["loss"] - f["loss_lm"] > 1e-3  # >> float32-vs-float64 accumulation
    fresh = ExpireSpanPolicy(ExpireSpanConfig(**ES_OV, seed=0), 32, 1)
    assert not torch.equal(
        es["policy"]["_extra_state"]["generator"], fresh.get_extra_state()["generator"]
    )
    r0, _ = _train_es(tmp_path, "es0", iters=2, ov={"loss_coef": 0.0})
    assert abs(r0["final"]["loss"] - r0["final"]["loss_lm"]) < 1e-5


def test_w1_an_eval_caller_gets_the_mask_but_no_loss_and_no_dropout():
    cfg = _tg_cfg()
    torch.manual_seed(0)
    model = TGModel(cfg).eval()
    pol = ExpireSpanPolicy(ExpireSpanConfig(**ES_OV, seed=0), cfg.D, _n_cross(cfg))
    ids, mask, lengths = _stream(cfg, 6)
    gen0 = pol.get_extra_state()["generator"].clone()
    zero = lambda *a: torch.zeros(())  # noqa: E731
    with torch.no_grad():
        acc = run_policy_loop(model, ids, mask, lengths, pol, step_fn=zero)
    assert float(acc) == 0.0  # policy_training=None -> the model's mode: no span loss
    assert torch.equal(pol.get_extra_state()["generator"], gen0)  # no dropout draw
    with torch.no_grad():
        acc = run_policy_loop(
            model, ids, mask, lengths, pol, step_fn=zero, policy_training=True
        )
    assert float(acc) > 0.0
    assert not torch.equal(pol.get_extra_state()["generator"], gen0)


# --------------------------------------------------------------------------- #
# FIFO and psiC are unchanged by the wiring
# --------------------------------------------------------------------------- #

#: Measured at `ed6302a` (before this wiring) with the identical calls, and again
#: after it: equal. The run id is the config hash; the RNG digest is the global CPU
#: generator's state after `train()` -- the same draws, in the same order.
PRE_WIRING = {
    "fifo": (
        "fifo-d32-s48-b2-9602104390e9d161",
        "fifo",
        "2a46e6f41f75e6c97d7194b38e8a5fda8ac31b7662693e4363058d2077cce830",
    ),
    "psiC": (
        "rsr-d32-s48-b2-206f453525bb479f",
        "rsr[b=off,beta=1,gamma=0.9,nu=0,shadow=off]",
        "2a46e6f41f75e6c97d7194b38e8a5fda8ac31b7662693e4363058d2077cce830",
    ),
    "fifo_frozen_eval": (
        "fifo-d32-s48-b2-517632c54ebf9de0",
        "fifo+frozen[eval]",
        "68cb7af797a8f3f61209d9bf16fa94786d66390d7b72dfb3c17a78aadd74e7d6",
    ),
    "psiC_frozen_eval": (
        "rsr-d32-s48-b2-3e9f49e4d0ed3ba9",
        "rsr[b=off,beta=1,gamma=0.9,nu=0,shadow=off]+frozen[eval]",
        "68cb7af797a8f3f61209d9bf16fa94786d66390d7b72dfb3c17a78aadd74e7d6",
    ),
}


@pytest.mark.parametrize("arm", sorted(PRE_WIRING))
def test_fifo_and_psic_arms_are_unchanged_by_the_wiring(tmp_path, arm):
    pol, ov = ("rsr", PSIC) if arm.startswith("psiC") else ("fifo", None)
    kw = FROZEN_EVAL if arm.endswith("frozen_eval") else {}
    r = train(
        **TINY,
        iters=2,
        out_dir=tmp_path / arm,
        ckpt_every=0,
        policy_name=pol,
        policy_overrides=ov,
        **kw,
    )
    rng = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    assert (r["run_id"], r["arm_label"], rng) == PRE_WIRING[arm]
    assert "expire_span" not in r["config"]


# --------------------------------------------------------------------------- #
# The ruling's PM note: predictor vs through_gestalt with the model frozen
# --------------------------------------------------------------------------- #


def test_frozen_predictor_and_through_gestalt_do_NOT_update_w_b_identically(tmp_path):
    """FINDING (measured; the ruling's PM note expected bit-equality). Five frozen
    steps on `train()`'s own path: the two paths give different `w, b`.

    Why: nothing of the MODEL trains, but `w, b` are themselves upstream of every
    later gestalt. Step t's mask shapes step t's cross-attention, so the srep written
    to memory at t depends on `w, b`; TG writes it with its graph
    (`detach_sreps_for_memory = False`). Under `through_gestalt` a later span is
    computed from that gestalt and carries the extra term de/dh * dh/d(w, b);
    `predictor` detaches it. The control below removes exactly that term."""
    got = {}
    for gp in ("predictor", "through_gestalt"):
        _r, p = _train_es(tmp_path, gp, iters=5, ov={"grad_path": gp})
        got[gp] = (p["policy"]["weight"], p["policy"]["bias"])
    (w0, b0), (w1, b1) = got["predictor"], got["through_gestalt"]
    assert not torch.equal(w0, w1)
    assert not torch.equal(b0, b1)


@pytest.mark.parametrize("detach", [False, True])
def test_the_paths_differ_only_through_the_gestalts_dependence_on_w_b(detach):
    """The control: with sreps written to memory detached (TG's
    `detach_sreps_for_memory`), the five frozen updates ARE bit-equal; with TG's
    default they are not."""
    got = {}
    for gp in ("predictor", "through_gestalt"):
        got[gp] = _five_frozen_updates(gp, detach)
    same = all(
        torch.equal(a, b)
        for a, b in zip(got["predictor"], got["through_gestalt"], strict=True)
    )
    assert same is detach


def _five_frozen_updates(grad_path: str, detach: bool):
    """`train()`'s ES step in miniature: frozen eval-mode model, phi's AdamW group,
    `loss` (LM + span loss) as the objective, `policy_training=True`."""
    cfg = _tg_cfg(detach_sreps_for_memory=detach)
    torch.manual_seed(0)
    model = TGModel(cfg)
    model.requires_grad_(False)
    model.eval()
    pol = ExpireSpanPolicy(
        ExpireSpanConfig(**{**ES_OV, "grad_path": grad_path}, seed=0),
        cfg.D,
        _n_cross(cfg),
    )
    groups = build_param_groups(model, pol, base_lr=1e-2, d_model=cfg.D, base_width=128)
    opt = torch.optim.AdamW(
        groups,
        betas=loop.ADAMW_BETAS,
        weight_decay=loop.ADAMW_WEIGHT_DECAY,
        eps=loop.ADAMW_EPS,
    )
    for step in range(5):
        ids, mask, lengths = _stream(cfg, 8, seed=step)
        loss = run_policy_loop(
            model, ids, mask, lengths, pol, step_fn=_loss, policy_training=True
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return pol.weight.detach().clone(), pol.bias.detach().clone()


# --------------------------------------------------------------------------- #
# Helpers (the shapes of tests/test_expire_span.py's)
# --------------------------------------------------------------------------- #


def _tg_cfg(**over) -> TGConfig:
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
        **over,
    )


def _n_cross(cfg: TGConfig) -> int:
    return sum(b == "C" for b in cfg.block_config)


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
