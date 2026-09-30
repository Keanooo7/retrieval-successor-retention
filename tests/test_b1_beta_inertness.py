"""experiments/b1-beta-inertness/run.py: the L_MC reduction, the four optimizer arms and
the two controls, the three metrics and the inertness rule, on hand-built tensors and a
tiny untrained model.

Governing text: experiments/b1-beta-inertness/PREREG.md (a0c2b30). No checkpoint is
read. The tiny fixture generates S = 12 documents by id inside B2's FIT ranges only.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch

from rsr.data.synthetic import SyntheticConfig, _generate_document
from rsr.model.tg import TGConfig, TGModel
from rsr.retention.value_head import BilinearValueHead
from rsr.train.loop import build_vocab

ROOT = Path(__file__).resolve().parents[1]
TS, TM, TL, TD = 12, 4, 16, 8


def _load(name, rel):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def b1():
    return _load("_b1_run_under_test", "experiments/b1-beta-inertness/run.py")


def _tiny_docs(ids):
    cfg = SyntheticConfig(sentences_per_document=TS, max_gap=8, heavy_tail_min=5, seed=0)
    return [_generate_document(i, cfg) for i in ids]


@pytest.fixture(scope="module")
def tiny(b1):
    train = _tiny_docs(range(920000, 920006))
    held = _tiny_docs(range(936000, 936003))
    vmap = build_vocab(train + held)
    torch.manual_seed(0)
    cfg = TGConfig(
        D=TD,
        V=4 + len(vmap),
        F=22,
        max_sentence_tokens=TL,
        max_sentences_in_short_term=TM,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        eod_id=3,
    )
    model = TGModel(cfg).eval()
    kw = {"vmap": vmap, "S": TS, "m": TM, "L": TL, "ranks": (0,)}
    tcaps = [b1.B2.capture_doc(model, d, **kw) for d in train]
    hcaps = [b1.B2.capture_doc(model, d, **kw) for d in held]
    return {
        "tcaps": tcaps,
        "hcaps": hcaps,
        "train": b1.pack(tcaps, TM),
        "held": b1.pack(hcaps, TM),
    }


# --------------------------------------------------------------------------- #
# the data: B2's rowset C and target C@0.9, stop-gradded
# --------------------------------------------------------------------------- #


def test_rows_are_b2_rowset_c_and_target_is_c_at_09(b1, tiny):
    """PREREG §2: rows = `row_index(cap, "C", m)`, target = `target_matrix(cap, "C", 0.9)`."""
    p = tiny["train"]
    for n, cap in enumerate(tiny["tcaps"]):
        t, i = b1.B2.row_index(cap, "C", TM)
        want = torch.zeros(TS, TS, dtype=torch.bool)
        want[t, i] = True
        assert torch.equal(p["rows"][n], want)
        G = b1.B2.target_matrix(cap, "C", 0.9)
        assert torch.allclose(p["G"][n][want], G[want].to(torch.float32))
        assert bool((p["G"][n][~want] == 0).all())


def test_full_memory_steps_are_those_with_m_resident(b1, tiny):
    """PREREG §6 m2: decisions only where |resident[t]| = M."""
    p = tiny["train"]
    assert torch.equal(p["full"], p["rows"].sum(-1) == TM)
    assert bool(p["full"][:, TM:].all()) and not bool(p["full"][:, :TM].any())


def test_pack_refuses_inputs_that_carry_graph(b1, tiny):
    """PREREG §2: no gradient can reach the transformer; `pack` refuses a captured
    s_i or c_t that requires grad."""
    cap = tiny["tcaps"][0]
    bad = type(cap)(**{**cap.__dict__, "gest": cap.gest.clone().requires_grad_(True)})
    with pytest.raises(b1.ControlFailure):
        b1.pack([bad], TM)


# --------------------------------------------------------------------------- #
# psi-hat and L_MC
# --------------------------------------------------------------------------- #


def test_psi_all_is_the_head_at_every_t_and_i(b1, tiny):
    p = tiny["train"]
    head = BilinearValueHead(TD, generator=torch.Generator().manual_seed(0))
    psi = b1.psi_all(head, p["gest"][:2], p["ctx"][:2])
    for n in range(2):
        for t in (1, 5, TS - 1):
            want = head(p["gest"][n], p["ctx"][n, t])  # [S] over i
            assert torch.allclose(psi[n, t], want, atol=1e-6)


def test_l_mc_sums_over_live_slots_and_means_over_doc_t(b1):
    """ADR-0009 L9 as PREREG §3 fixes it: sum over resident slots, mean over (doc, t)
    for t in [1, S)."""
    N, S = 2, 4
    psi = torch.zeros(N, S, S)
    G = torch.zeros(N, S, S)
    rows = torch.zeros(N, S, S, dtype=torch.bool)
    rows[0, 1, 0] = rows[0, 2, 0] = rows[0, 2, 1] = True
    rows[1, 3, 2] = True
    G[0, 1, 0], G[0, 2, 0], G[0, 2, 1], G[1, 3, 2] = 1.0, 2.0, 3.0, 4.0
    G[1, 0, 0] = 100.0  # not a row: must not count
    want = (1 + 4 + 9 + 16) / (N * (S - 1))
    assert b1.l_mc(psi, G, rows).item() == pytest.approx(want)


# --------------------------------------------------------------------------- #
# the arms and the controls
# --------------------------------------------------------------------------- #


def _head():
    return BilinearValueHead(TD, generator=torch.Generator().manual_seed(0))


def test_beta_scales_the_loss_and_never_the_lr_in_the_four_arms(b1):
    for arm in b1.ARMS:
        assert b1.loss_scale(arm, 0.01) == 0.01
        opt = b1.make_optimizer(arm, list(_head().parameters()), eps=1e-8, beta=0.01, lr=1e-3)
        assert opt.param_groups[0]["lr"] == 1e-3


def test_positive_control_scales_the_lr_and_not_the_loss(b1):
    """PREREG §5 P: L7(a)'s reading, lr = beta * 1e-3, gradient unscaled."""
    assert b1.loss_scale(b1.CONTROL_P, 0.01) == 1.0
    opt = b1.make_optimizer(
        b1.CONTROL_P, list(_head().parameters()), eps=1e-8, beta=0.01, lr=1e-3
    )
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-5)
    assert type(opt) is torch.optim.AdamW


def test_coupled_l2_is_adam_with_l2_and_the_others_are_adamw_decoupled(b1):
    ps = list(_head().parameters())
    for arm in b1.ARMS:
        opt = b1.make_optimizer(arm, ps, eps=1e-12, beta=1.0, lr=1e-3)
        g = opt.param_groups[0]
        assert g["betas"] == (0.9, 0.95) and g["eps"] == 1e-12 and g["weight_decay"] == 0.01
        if arm == "coupled_l2":
            assert type(opt) is torch.optim.Adam
        else:
            assert type(opt) is torch.optim.AdamW


def _grads(ps, norm):
    torch.manual_seed(1)
    gs = [torch.randn_like(p) for p in ps]
    tot = math.sqrt(sum(float((g * g).sum()) for g in gs))
    for p, g in zip(ps, gs, strict=True):
        p.grad = g * (norm / tot)


def _norm(ps):
    return math.sqrt(sum(float((p.grad * p.grad).sum()) for p in ps))


def test_phi_clip_binds_on_phi_norm_at_one(b1):
    ps = list(_head().parameters())
    _grads(ps, 5.0)
    info = b1.treat("phi_clip", ps, g_T=None)
    assert info["clipped"] is True and _norm(ps) == pytest.approx(1.0, rel=1e-5)
    _grads(ps, 0.5)
    info = b1.treat("phi_clip", ps, g_T=None)
    assert info["clipped"] is False and _norm(ps) == pytest.approx(0.5, rel=1e-6)
    _grads(ps, 5.0)
    b1.treat("decoupled_wd", ps, g_T=None)
    assert _norm(ps) == pytest.approx(5.0, rel=1e-6)


def test_joint_clip_uses_the_joint_norm_with_the_transformers(b1):
    """PREREG §5 arm 3: coefficient min(1, 1/sqrt(g_T^2 + |g_phi|^2)); m4 is its
    relative deviation from the transformer-only coefficient."""
    ps = list(_head().parameters())
    _grads(ps, 5.0)
    info = b1.treat("joint_clip", ps, g_T=12.0)
    coef = 1.0 / (13.0 + 1e-6)
    assert info["coef"] == pytest.approx(coef, rel=1e-6)
    assert _norm(ps) == pytest.approx(5.0 * coef, rel=1e-5)
    c_T = 1.0 / (12.0 + 1e-6)
    assert info["dev"] == pytest.approx(abs(coef - c_T) / c_T, rel=1e-6)


def test_batch_order_is_a_permutation_per_epoch_and_seed_determined(b1):
    o = b1.batch_order(8, 4, 6, seed=2000)
    assert o.shape == (6, 4)
    for e in range(3):
        assert sorted(o[2 * e : 2 * e + 2].flatten().tolist()) == list(range(8))
    assert torch.equal(o, b1.batch_order(8, 4, 6, seed=2000))
    assert not torch.equal(o, b1.batch_order(8, 4, 6, seed=2001))


# --------------------------------------------------------------------------- #
# the metrics and the rule
# --------------------------------------------------------------------------- #


def test_argmins_only_over_resident_slots_at_full_steps(b1):
    N, S = 1, 5
    psi = torch.zeros(N, S, S)
    rows = torch.zeros(N, S, S, dtype=torch.bool)
    full = torch.zeros(N, S, dtype=torch.bool)
    # step 3 is full with slots {1, 2}; slot 0 has the lowest psi but is not resident
    rows[0, 3, 1] = rows[0, 3, 2] = True
    full[0, 3] = True
    psi[0, 3] = torch.tensor([-9.0, 0.5, 0.2, -8.0, -7.0])
    # step 2 is not full: must not be read
    rows[0, 2, 0] = True
    psi[0, 2, 0] = -1.0
    a = b1.argmins(psi, rows, full)
    assert a.tolist() == [2]


def test_eps_ratio_uses_the_bias_corrected_second_moment(b1):
    """PREREG §6 m3: eps / sqrt(v / (1 - 0.95^k))."""
    p = torch.nn.Parameter(torch.zeros(3))
    opt = torch.optim.AdamW([p], lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    p.grad = torch.tensor([1e-3, 2e-3, 4e-3])
    opt.step()
    r = b1.eps_ratio(opt, 1e-8)
    assert torch.allclose(r, 1e-8 / torch.tensor([1e-3, 2e-3, 4e-3]), rtol=1e-4)


def test_classify_thresholds_are_the_prereg_ones(b1):
    """PREREG §7."""
    c = b1.classify
    assert c([0.99, 1.0], [0.01, 0.0]) == "INERT"
    assert c([0.9899, 1.0], [0.0]) == "INTERMEDIATE"
    assert c([0.95], [0.0]) == "INTERMEDIATE"
    assert c([0.9499], [0.0]) == "LIVE"
    assert c([1.0], [0.0100001]) == "INTERMEDIATE"
    assert c([1.0], [0.1]) == "INTERMEDIATE"
    assert c([1.0], [0.1000001]) == "LIVE"


def test_compare_reads_every_cell_against_its_own_beta_one(b1):
    """PREREG §6: against beta = 1 of the same seed, arm and eps."""

    def run(val, am):
        return {"phi": {3000: torch.tensor([val, 1.0])}, "argmin": {3000: torch.tensor(am)}}

    runs = {
        (0, "decoupled_wd", 1e-8, 1.0): run(1.0, [0, 1]),
        (0, "decoupled_wd", 1e-8, 0.1): run(1.0, [0, 1]),
        (0, "coupled_l2", 1e-8, 1.0): run(2.0, [1, 1]),
        (0, "coupled_l2", 1e-8, 0.1): run(2.0, [1, 1]),
    }
    out = b1.compare(runs, 0, "coupled_l2", 1e-8, 0.1, snap=3000)
    assert out["max_abs"] == 0.0 and out["agreement"] == 1.0


# --------------------------------------------------------------------------- #
# the transformer norm
# --------------------------------------------------------------------------- #


def test_read_g_t_takes_steps_2000_to_2999_and_refuses_a_gap(b1, tmp_path):
    hb = tmp_path / "heartbeat.jsonl"
    lines = [json.dumps({"kind": "header"})]
    lines += [
        json.dumps({"kind": "beat", "step": s, "grad_norm": float(s)})
        for s in range(1000, 3000)
    ]
    hb.write_text("\n".join(lines) + "\n")
    g = b1.read_g_T(hb)
    assert g.shape == (1000,) and g[0].item() == 2000.0 and g[-1].item() == 2999.0
    hb.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(b1.ControlFailure):
        b1.read_g_T(hb)


# --------------------------------------------------------------------------- #
# end to end, tiny: N is bit-identical, beta is inert at eps ~ 0, P is not
# --------------------------------------------------------------------------- #


def test_train_one_is_deterministic_and_beta_inert_without_eps_but_p_is_not(b1, tiny):
    kw = {
        "steps": 40,
        "snaps": (40,),
        "batch": 2,
        "seed": 0,
        "init_seed": 1000,
        "g_T": None,
    }
    tr, ho = tiny["train"], tiny["held"]
    r1 = b1.train_one(tr, ho, arm="decoupled_wd", eps=1e-30, beta=1.0, **kw)
    r1b = b1.train_one(tr, ho, arm="decoupled_wd", eps=1e-30, beta=1.0, **kw)
    assert torch.equal(r1["phi"][40], r1b["phi"][40])
    r01 = b1.train_one(tr, ho, arm="decoupled_wd", eps=1e-30, beta=0.01, **kw)
    d = (r01["phi"][40] - r1["phi"][40]).abs().max() / r1["phi"][40].abs().max()
    assert float(d) < 1e-4
    rp = b1.train_one(tr, ho, arm=b1.CONTROL_P, eps=1e-30, beta=0.01, **kw)
    dp = (rp["phi"][40] - r1["phi"][40]).abs().max() / r1["phi"][40].abs().max()
    assert float(dp) > 1e-3
