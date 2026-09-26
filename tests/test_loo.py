"""`rsr.metrics.loo` -- the leave-one-out slot-knockout instrument (spec §3.2.1).

§3.2.1: "ablate slot *i* and measure Δ next-sentence loss (leave-one-out) ... If
they disagree, LOO is truth." These tests prove the INSTRUMENT, on a tiny random
model; no experiment is run here.

The load-bearing ones:

* the no-knockout path reproduces S0-03's `answer_readout(cond="live")`
  bit-exactly (max |diff| 0.0), so any knockout delta is the knockout's and not a
  second, subtly different readout's;
* a knockout changes exactly one slot of exactly one row, and nothing else in the
  memory the model attends over;
* a resample donor is the same kind, a different document, the same rank;
* a missing control target is recorded as missing, never substituted.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsr.data.synthetic import ANSWER_SYMBOLS, SyntheticConfig, generate  # noqa: E402
from rsr.metrics import loo  # noqa: E402
from rsr.model.tg import TGConfig, TGModel  # noqa: E402
from rsr.train.loop import answer_targets, build_vocab, encode  # noqa: E402

S, L, M = 16, 10, 4


def _s003():
    spec = importlib.util.spec_from_file_location(
        "s003_run_for_loo", ROOT / "experiments" / "s0-03-rewardable-corpus" / "run.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _docs(n=8, seed=0):
    return generate(
        SyntheticConfig(
            n_documents=n,
            sentences_per_document=S,
            max_gap=12,
            heavy_tail_min=6,
            seed=seed,
        )
    )


def _world(n=8, seed=0, model_seed=0):
    docs = _docs(n, seed)
    # The vocab is built over a larger prefix so every answer symbol has an id
    # whatever n is (the 16-way renormalisation needs all 16).
    vmap = build_vocab(_docs(64, seed))
    V = 4 + len(vmap)
    torch.manual_seed(model_seed)
    cfg = TGConfig(
        D=16,
        H=2,
        N=4,
        V=V,
        block_config=("S", "C", "S", "C"),
        srep_extraction_layer=2,
        max_sentence_tokens=L,
        max_sentences_in_short_term=M,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        eod_id=3,
    )
    model = TGModel(cfg)
    ids, mask = encode(docs, vmap, max_tokens=L, steps=S)
    tmask, gap = answer_targets(docs, vmap, max_tokens=L, steps=S)
    sym = torch.tensor([vmap[s] for s in ANSWER_SYMBOLS])
    return model, docs, ids, mask, tmask, gap, sym


class _Spy(torch.nn.Module):
    """Wraps a model and records the memory kv of every forward call."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.cfg = inner.cfg
        self.embed = inner.embed
        self.calls: list[tuple[torch.Tensor, torch.Tensor, bool, bool]] = []

    def forward(self, ids_t, mask_t, kv, valid, bos_ctx, bv):
        self.calls.append(
            (kv.detach().clone(), valid.clone(), self.training, torch.is_grad_enabled())
        )
        return self.inner(ids_t, mask_t, kv, valid, bos_ctx, bv)


# --------------------------------------------------------------------------- #
# exactness control: live == answer_readout(cond="live"), bit for bit
# --------------------------------------------------------------------------- #


def test_live_path_is_bit_exact_to_answer_readout():
    model, docs, ids, mask, tmask, gap, sym = _world()
    ref = _s003().answer_readout(model, ids, mask, tmask, gap, sym, cond="live")
    r = loo.loo_readout(model, docs, ids, mask, tmask, gap, sym, seed=0)
    assert r["gap"].shape == ref["gap"].shape and r["gap"].numel() > 0
    assert torch.equal(r["gap"], ref["gap"])
    assert torch.equal(r["live_ok"], ref["ok"].double())
    for mine, theirs in (("live_nll", "nll"), ("live_nll16", "nll16")):
        assert float((r[mine] - ref[theirs]).abs().max()) == 0.0, mine
    assert float((r["live_brier16"] - ref["brier16"]).abs().max()) == 0.0
    assert r["real_token_nll"] == ref["real_token_nll"]


def test_records_are_ordered_step_major_then_row():
    model, docs, ids, mask, tmask, gap, sym = _world()
    r = loo.loo_readout(model, docs, ids, mask, tmask, gap, sym, seed=0)
    key = r["t"] * 10_000 + r["row"]
    assert torch.equal(key, key.sort().values)
    for i in range(r["t"].numel()):
        b, t = int(r["row"][i]), int(r["t"][i])
        assert docs[b].sentences[t].kind == "query"
        assert int(r["gap"][i]) == int(gap[b, t])
        assert int(r["doc_id"][i]) == docs[b].doc_id


# --------------------------------------------------------------------------- #
# the one code path: knockout_kv
# --------------------------------------------------------------------------- #


def test_knockout_kv_zero_changes_exactly_the_target_slot_of_the_target_row():
    g = torch.Generator().manual_seed(1)
    kv = torch.randn(3, M, 8, generator=g)
    before = kv.clone()
    slot = torch.tensor([-1, 2, -1])
    out = loo.knockout_kv(kv, slot, mode="zero")
    assert torch.equal(out[1, 2], torch.zeros(8))
    keep = torch.ones(3, M, dtype=torch.bool)
    keep[1, 2] = False
    assert torch.equal(out[keep], kv[keep])
    assert torch.equal(kv, before)  # input not mutated in place
    assert not torch.equal(out, kv)


def test_knockout_kv_replace_writes_the_replacement_only_there():
    g = torch.Generator().manual_seed(2)
    kv = torch.randn(2, M, 8, generator=g)
    rep = torch.randn(2, 8, generator=g)
    before = kv.clone()
    out = loo.knockout_kv(kv, torch.tensor([0, -1]), mode="replace", replacement=rep)
    assert torch.equal(out[0, 0], rep[0])
    assert torch.equal(out[0, 1:], kv[0, 1:])
    assert torch.equal(out[1], kv[1])
    assert torch.equal(kv, before)


def test_knockout_kv_rejects_unknown_mode_and_missing_replacement():
    kv = torch.zeros(1, M, 4)
    with pytest.raises(ValueError):
        loo.knockout_kv(kv, torch.tensor([0]), mode="mean")
    with pytest.raises(ValueError):
        loo.knockout_kv(kv, torch.tensor([0]), mode="replace")


def test_readout_knockout_forwards_differ_from_live_only_at_the_target():
    """Through loo_readout itself: every knockout forward's kv equals the live kv
    except at (row, own/ctrl rank) of rows that carry a target at that step, and
    the validity mask is never touched."""
    model, docs, ids, mask, tmask, gap, sym = _world()
    spy = _Spy(model)
    r = loo.loo_readout(spy, docs, ids, mask, tmask, gap, sym, seed=0)
    n_cond = len(loo.CONDITIONS)
    # Group calls by step: a step with targets makes n_cond calls, else 1.
    steps_with = sorted(set(r["t"].tolist()))
    calls = iter(spy.calls)
    checked = 0
    for t in range(S):
        live_kv, live_valid, *_ = next(calls)
        if t not in steps_with:
            continue
        sel = r["t"] == t
        per = {c: next(calls) for c in loo.CONDITIONS[1:]}
        assert len(per) == n_cond - 1
        for cond, (kv, valid, *_rest) in per.items():
            assert torch.equal(valid, live_valid), cond
            diff = (kv != live_kv).any(-1)  # [B, M]
            want = torch.zeros_like(diff)
            if cond == "all_slots_zeroed":
                want = live_valid & (live_kv != 0).any(-1)
                assert torch.equal(kv, torch.zeros_like(kv))
                assert torch.equal(diff, want)
                continue
            rank_key = "ctrl_rank" if cond == "ctrl_resample" else "own_rank"
            for i in sel.nonzero().flatten().tolist():
                b, rk = int(r["row"][i]), int(r[rank_key][i])
                applied = not math.isnan(float(r[f"{cond}_nll"][i]))
                if applied:
                    want[b, rk] = True
            assert torch.equal(diff, want), (cond, t)
            checked += int(want.sum())
    assert checked > 0


# --------------------------------------------------------------------------- #
# targets and donors
# --------------------------------------------------------------------------- #


def test_own_slot_is_the_slot_holding_t_minus_gap():
    model, docs, ids, mask, tmask, gap, sym = _world()
    spy = _Spy(model)
    r = loo.loo_readout(spy, docs, ids, mask, tmask, gap, sym, seed=0)
    for i in range(r["t"].numel()):
        t, g = int(r["t"][i]), int(r["gap"][i])
        a = t - g
        # FIFO, one write per sentence: the assert is resident iff g <= M.
        assert bool(r["own_resident"][i]) == (g <= M)
        if r["own_resident"][i]:
            k = min(t, M)  # occupied count before step t's write
            assert int(r["own_rank"][i]) == k - g
            assert int(r["own_sentence"][i]) == a
        else:
            assert int(r["own_rank"][i]) == -1
            assert math.isnan(float(r["own_zero_nll"][i]))
            assert math.isnan(float(r["own_resample_nll"][i]))
            assert int(r["ctrl_status"][i]) == loo.CTRL_OWN_NOT_RESIDENT


def test_resample_donor_is_same_kind_different_doc_same_rank():
    model, docs, ids, mask, tmask, gap, sym = _world()
    r = loo.loo_readout(model, docs, ids, mask, tmask, gap, sym, seed=0)
    n_checked = 0
    for who in ("own", "ctrl"):
        for i in range(r["t"].numel()):
            dr = int(r[f"{who}_donor_row"][i])
            if dr < 0:
                continue
            b, t = int(r["row"][i]), int(r["t"][i])
            ds = int(r[f"{who}_donor_sentence"][i])
            target_s = int(r[f"{who}_sentence"][i])
            assert dr != b
            assert docs[dr].doc_id != docs[b].doc_id
            assert docs[dr].sentences[ds].kind == docs[b].sentences[target_s].kind
            assert docs[dr].sentences[ds].kind == "assert"
            # same rank: FIFO memory at step t holds sentences t-k .. t-1 in order
            k = min(t, M)
            assert ds - (t - k) == int(r[f"{who}_rank"][i])
            assert ds < t
            n_checked += 1
    assert n_checked > 0


def test_pick_donor_filters_kind_doc_rank_and_key():
    ann = {
        "kind": torch.tensor([[1, 0], [1, 1], [0, 1], [1, 1]]),
        "fact_key": torch.tensor([[5, -1], [5, 7], [-1, 9], [8, 8]]),
        "doc_id": torch.tensor([10, 11, 12, 10]),
        "query_of": torch.full((4, 2), -1),
    }
    mem_step = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]])
    valid = torch.ones(4, 2, dtype=torch.bool)
    seen = set()
    for s in range(40):
        d = loo.pick_donor(
            ann,
            mem_step,
            valid,
            row=0,
            rank=1,
            want_kind=1,
            exclude_keys={7},
            seed=s,
            t=2,
            role=0,
        )
        seen.add(d)
    # row 0 is the target; row 1 rank 1 has key 7 (excluded); row 2 rank 1 is an
    # assert with key 9; row 3 is the same doc_id as row 0.
    assert seen == {(2, 1)}
    none = loo.pick_donor(
        ann,
        mem_step,
        valid,
        row=0,
        rank=0,
        want_kind=2,
        exclude_keys=set(),
        seed=0,
        t=2,
        role=0,
    )
    assert none is None


def test_missing_control_is_recorded_not_substituted():
    """A control target exists only at own_rank +- 1 holding a pending assert. When
    none does, ctrl_* is NaN and ctrl_status says why -- never another slot."""
    model, docs, ids, mask, tmask, gap, sym = _world()
    ann = loo.stream_annotations(docs, steps=S)
    r = loo.loo_readout(model, docs, ids, mask, tmask, gap, sym, seed=0)
    statuses = set(r["ctrl_status"].tolist())
    assert loo.CTRL_NO_PENDING_ADJACENT in statuses  # the fixture exercises it
    for i in range(r["t"].numel()):
        st = int(r["ctrl_status"][i])
        b, t = int(r["row"][i]), int(r["t"][i])
        if st != loo.CTRL_PRESENT:
            assert int(r["ctrl_rank"][i]) == -1
            assert int(r["ctrl_sentence"][i]) == -1
            assert math.isnan(float(r["ctrl_resample_nll"][i]))
            assert math.isnan(float(r["ctrl_resample_ok"][i]))
        if st == loo.CTRL_NO_PENDING_ADJACENT:
            own = int(r["own_rank"][i])
            k = min(t, M)
            for rk in (own - 1, own + 1):
                if 0 <= rk < k:
                    s = t - k + rk
                    pending = (
                        int(ann["kind"][b, s]) == 1 and int(ann["query_of"][b, s]) > t
                    )
                    assert not pending or int(ann["fact_key"][b, s]) == int(
                        ann["fact_key"][b, t]
                    )
        if st == loo.CTRL_PRESENT:
            cs = int(r["ctrl_sentence"][i])
            assert abs(int(r["ctrl_rank"][i]) - int(r["own_rank"][i])) == 1
            assert int(ann["kind"][b, cs]) == 1
            assert int(ann["query_of"][b, cs]) > t


def test_missing_donor_is_recorded_not_substituted():
    """B = 1: there is no other document, so no resample donor can exist."""
    model, docs, ids, mask, tmask, gap, sym = _world()
    r = loo.loo_readout(
        model, docs[:1], ids[:1], mask[:1], tmask[:1], gap[:1], sym, seed=0
    )
    assert r["t"].numel() > 0
    assert (r["own_donor_row"] == -1).all()
    assert torch.isnan(r["own_resample_nll"]).all()
    assert torch.isnan(r["ctrl_resample_nll"]).all()
    res = r["own_resident"]
    assert (~torch.isnan(r["own_zero_nll"][res])).all()


# --------------------------------------------------------------------------- #
# modes, determinism, batching
# --------------------------------------------------------------------------- #


def test_every_forward_is_eval_mode_and_no_grad_and_mode_is_restored():
    model, docs, ids, mask, tmask, gap, sym = _world()
    spy = _Spy(model)
    spy.train()
    loo.loo_readout(spy, docs, ids, mask, tmask, gap, sym, seed=0)
    assert spy.calls
    assert all(not training for _kv, _v, training, _g in spy.calls)
    assert all(not grad for _kv, _v, _t, grad in spy.calls)
    assert spy.training and model.training


def test_deterministic_under_a_seed_and_seed_moves_donors():
    model, docs, ids, mask, tmask, gap, sym = _world()
    a = loo.loo_readout(model, docs, ids, mask, tmask, gap, sym, seed=3)
    b = loo.loo_readout(model, docs, ids, mask, tmask, gap, sym, seed=3)
    for k, v in a.items():
        if isinstance(v, torch.Tensor):
            assert torch.equal(v.nan_to_num(-7.0), b[k].nan_to_num(-7.0)), k
    others = [
        loo.loo_readout(model, docs, ids, mask, tmask, gap, sym, seed=s)
        for s in (4, 5, 6)
    ]
    assert any(not torch.equal(a["own_donor_row"], o["own_donor_row"]) for o in others)


def test_donor_choice_does_not_depend_on_global_rng():
    model, docs, ids, mask, tmask, gap, sym = _world()
    torch.manual_seed(0)
    a = loo.loo_readout(model, docs, ids, mask, tmask, gap, sym, seed=3)
    torch.manual_seed(12345)
    b = loo.loo_readout(model, docs, ids, mask, tmask, gap, sym, seed=3)
    assert torch.equal(a["own_donor_row"], b["own_donor_row"])
    assert torch.equal(a["ctrl_donor_row"], b["ctrl_donor_row"])


def test_batched_equals_row_by_row_for_row_local_conditions():
    """B > 1 with per-row targets: each row's live / own_zero / all_slots_zeroed
    record equals the one computed with that row alone (resample depends on the
    batch as donor pool, so it is excluded by design)."""
    model, docs, ids, mask, tmask, gap, sym = _world(n=4)
    full = loo.loo_readout(model, docs, ids, mask, tmask, gap, sym, seed=0)
    for b in range(4):
        one = loo.loo_readout(
            model,
            docs[b : b + 1],
            ids[b : b + 1],
            mask[b : b + 1],
            tmask[b : b + 1],
            gap[b : b + 1],
            sym,
            seed=0,
        )
        sel = full["row"] == b
        assert int(sel.sum()) == one["t"].numel() > 0
        assert torch.equal(full["t"][sel], one["t"])
        assert torch.equal(full["own_rank"][sel], one["own_rank"])
        for c in ("live", "own_zero", "all_slots_zeroed"):
            x, y = full[f"{c}_nll"][sel], one[f"{c}_nll"]
            assert torch.allclose(x, y, atol=1e-5, equal_nan=True), (b, c)


def test_knockouts_move_the_answer_nll_on_a_live_memory():
    """A random-init model has g_mem = 1, so emptying the memory must move the
    answer NLL -- a knockout that changes nothing downstream is not wired in."""
    model, docs, ids, mask, tmask, gap, sym = _world()
    r = loo.loo_readout(model, docs, ids, mask, tmask, gap, sym, seed=0)
    res = r["own_resident"]
    assert (r["all_slots_zeroed_nll"] != r["live_nll"]).any()
    assert (r["own_zero_nll"][res] != r["live_nll"][res]).any()
    ok = ~torch.isnan(r["own_resample_nll"])
    assert ok.any() and (r["own_resample_nll"][ok] != r["live_nll"][ok]).any()


def test_rejects_a_model_without_memory():
    model, docs, ids, mask, tmask, gap, sym = _world()
    model.cfg = TGConfig(**{**model.cfg.__dict__, "use_memory": False})
    with pytest.raises(ValueError, match="use_memory"):
        loo.loo_readout(model, docs, ids, mask, tmask, gap, sym, seed=0)


# --------------------------------------------------------------------------- #
# E0d's per-slot delta loss (the stub's original entry point)
# --------------------------------------------------------------------------- #


def test_loo_delta_loss_matches_a_direct_single_knockout():
    model, docs, ids, mask, *_ = _world(n=3)
    out = loo.loo_delta_loss(model, docs, ids, mask, mode="zero", seed=0)
    assert out["delta"].shape == (3, S, M)
    # Step 0 has an empty memory: every slot is NaN.
    assert torch.isnan(out["delta"][:, 0]).all()
    # Step 2 has two occupied slots, ranks 0 and 1; ranks 2, 3 NaN.
    assert torch.isnan(out["delta"][:, 2, 2:]).all()
    assert not torch.isnan(out["delta"][:, 2, :2]).any()
    assert torch.equal(out["slot_sentence"][0, 5], torch.tensor([1, 2, 3, 4]))
    # Recompute one cell by hand via the same knockout primitive.
    ref = loo.loo_delta_loss(model, docs, ids, mask, mode="zero", seed=0, steps=(5,))
    assert torch.equal(
        ref["delta"][:, 5].nan_to_num(9.0), out["delta"][:, 5].nan_to_num(9.0)
    )
    assert torch.isnan(ref["delta"][:, 4]).all()
    assert (out["delta"][:, 5] != 0).any()


def test_loo_delta_loss_step_one_by_hand():
    """Step 1, slot 0 (sentence 0's gestalt), recomputed from the model directly:
    delta = mean real-target NLL of sentence 1 with that slot zeroed, minus live."""
    from rsr.model.tg.model import init_memory
    from rsr.model.tg.policy_loop import write_at
    from rsr.train.loop import lm_token_losses

    model, docs, ids, mask, *_ = _world(n=3)
    out = loo.loo_delta_loss(model, docs, ids, mask, mode="zero", seed=0)
    model.eval()
    with torch.no_grad():
        B = ids.shape[0]
        mem = init_memory(B, model.cfg)
        z = torch.zeros(B, model.cfg.D)
        o0 = model(ids[:, 0], mask[:, 0], mem.kv, mem.valid, z, torch.zeros(B).bool())
        mem = write_at(mem, o0.srep, o0.has_eos, torch.zeros(B).long(), 0)

        def sent_loss(kv):
            o = model(ids[:, 1], mask[:, 1], kv, mem.valid, o0.srep, o0.has_eos)
            per = lm_token_losses(o.logits, ids[:, 1]).view(B, -1).double()
            real = mask[:, 1, 1:]
            return (per * real).sum(-1) / real.sum(-1)

        want = sent_loss(torch.zeros_like(mem.kv)) - sent_loss(mem.kv)
    assert torch.allclose(out["delta"][:, 1, 0], want, atol=1e-12, rtol=0)
    assert torch.allclose(out["live_loss"][:, 1], sent_loss(mem.kv), atol=1e-12, rtol=0)


def test_loo_delta_loss_resample_records_donor_or_nan():
    model, docs, ids, mask, *_ = _world(n=6)
    out = loo.loo_delta_loss(model, docs, ids, mask, mode="resample", seed=0)
    dr = out["donor_row"]
    has = dr >= 0
    assert has.any()
    assert torch.isnan(out["delta"][~has]).all()
    assert not torch.isnan(out["delta"][has]).any()
    with pytest.raises(ValueError):
        loo.loo_delta_loss(model, docs, ids, mask, mode="mean", seed=0)
