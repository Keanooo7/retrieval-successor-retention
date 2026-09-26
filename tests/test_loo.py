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

import dataclasses
import importlib.util
import math
import sys
from pathlib import Path
from typing import NamedTuple

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsr.data.synthetic import (  # noqa: E402
    ANSWER_SYMBOLS,
    SyntheticConfig,
    answer_symbol,
    generate,
)
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


def _vocab(seed=0):
    # Built over a larger prefix so every answer symbol has an id whatever n is
    # (the 16-way renormalisation needs all 16).
    return build_vocab(_docs(64, seed))


def _encode(docs, seed=0):
    vmap = _vocab(seed)
    ids, mask = encode(docs, vmap, max_tokens=L, steps=S)
    tmask, gap = answer_targets(docs, vmap, max_tokens=L, steps=S)
    sym = torch.tensor([vmap[s] for s in ANSWER_SYMBOLS])
    return ids, mask, tmask, gap, sym


def _model(seed=0, model_seed=0):
    V = 4 + len(_vocab(seed))
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
    return TGModel(cfg)


def _world(n=8, seed=0, model_seed=0):
    docs = _docs(n, seed)
    return (_model(seed, model_seed), docs, *_encode(docs, seed))


_OBJ = tuple(sym.replace("_", " ") for sym in ANSWER_SYMBOLS)


def _key_of(doc, q):
    return doc.sentences[q].text.split("?", 1)[0][len("What does ") :]


def _set_fact(doc, a, q, key, obj):
    """Rewrite fact (a, q) of ``doc`` to ask ``key`` with answer ``obj``."""
    sents = list(doc.sentences)
    sents[a] = dataclasses.replace(sents[a], text=f"{key} {obj}.")
    sents[q] = dataclasses.replace(
        sents[q], text=f"What does {key}? {answer_symbol(obj)}", answer=obj
    )
    return dataclasses.replace(doc, sentences=tuple(sents))


def _twin(doc, new_id):
    """The same document, same keys, every answer object rotated by one."""
    for a, q in doc.pairs:
        obj = doc.sentences[q].answer
        doc = _set_fact(doc, a, q, _key_of(doc, q), _OBJ[(_OBJ.index(obj) + 1) % 16])
    return dataclasses.replace(doc, doc_id=new_id)


class _Call(NamedTuple):
    kv: torch.Tensor
    valid: torch.Tensor
    bv: torch.Tensor
    training: bool
    grad: bool


class _Spy(torch.nn.Module):
    """Wraps a model and records the memory inputs of every forward call."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.cfg = inner.cfg
        self.embed = inner.embed
        self.calls: list[_Call] = []

    def forward(self, ids_t, mask_t, kv, valid, bos_ctx, bv):
        self.calls.append(
            _Call(
                kv.detach().clone(),
                valid.clone(),
                bv.clone(),
                self.training,
                torch.is_grad_enabled(),
            )
        )
        return self.inner(ids_t, mask_t, kv, valid, bos_ctx, bv)


def _calls_by_step(spy, r):
    """-> {t: {condition: _Call}} for steps with targets; {t: {"live": ...}} else."""
    it = iter(spy.calls)
    steps_with = set(r["t"].tolist())
    out = {}
    for t in range(S):
        if t in steps_with:
            out[t] = {c: next(it) for c in loo.ALL_CONDITIONS}
        else:
            out[t] = {"live": next(it)}
    assert next(it, None) is None
    return out


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
    """Through loo_readout itself: every single-slot knockout forward's kv equals
    the live kv except at (row, own/ctrl rank) of rows where it was applied, and
    the validity mask is never touched."""
    model, docs, ids, mask, tmask, gap, sym = _world()
    spy = _Spy(model)
    r = loo.loo_readout(spy, docs, ids, mask, tmask, gap, sym, seed=0)
    by = _calls_by_step(spy, r)
    checked = 0
    for t, calls in by.items():
        if len(calls) == 1:
            continue
        live = calls["live"]
        sel = (r["t"] == t).nonzero().flatten().tolist()
        for cond in ("own_zero", "own_resample", "ctrl_resample"):
            c = calls[cond]
            assert torch.equal(c.valid, live.valid), cond
            diff = (c.kv != live.kv).any(-1)  # [B, M]
            want = torch.zeros_like(diff)
            rank_key = "ctrl_rank" if cond == "ctrl_resample" else "own_rank"
            for i in sel:
                if not math.isnan(float(r[f"{cond}_nll"][i])):
                    want[int(r["row"][i]), int(r[rank_key][i])] = True
            assert torch.equal(diff, want), (cond, t)
            checked += int(want.sum())
        z = calls["all_slots_zeroed"]
        assert torch.equal(z.kv, torch.zeros_like(z.kv))
        assert torch.equal(z.valid, live.valid)
    assert checked > 0


def test_all_slots_resample_swaps_whole_rows_for_a_different_documents_memory():
    model, docs, ids, mask, tmask, gap, sym = _world()
    spy = _Spy(model)
    r = loo.loo_readout(spy, docs, ids, mask, tmask, gap, sym, seed=0)
    ann = loo.stream_annotations(docs, steps=S)
    by = _calls_by_step(spy, r)
    n = 0
    for i in range(r["t"].numel()):
        b, t = int(r["row"][i]), int(r["t"][i])
        live, c = by[t]["live"], by[t]["all_slots_resample"]
        d = int(r["all_donor_row"][i])
        assert torch.equal(c.valid, live.valid)
        if d < 0:
            assert math.isnan(float(r["all_slots_resample_nll"][i]))
            assert torch.equal(c.kv[b], live.kv[b])
            continue
        n += 1
        assert d != b and docs[d].doc_id != docs[b].doc_id
        assert torch.equal(live.valid[d], live.valid[b])
        assert torch.equal(c.kv[b], live.kv[d])  # the whole memory, same step
        # the donor memory holds no assert of the queried question
        held = range(max(0, t - M), t)  # FIFO, one write per sentence
        qk = int(ann["fact_key"][b, t])
        assert all(
            not (int(ann["kind"][d, s]) == 1 and int(ann["fact_key"][d, s]) == qk)
            for s in held
        )
    assert n > 0


def test_memory_off_masks_every_slot_for_the_query_forward_only():
    model, docs, ids, mask, tmask, gap, sym = _world()
    spy = _Spy(model)
    r = loo.loo_readout(spy, docs, ids, mask, tmask, gap, sym, seed=0)
    by = _calls_by_step(spy, r)
    for calls in by.values():
        if len(calls) == 1:
            continue
        c, live = calls["memory_off"], calls["live"]
        assert not bool(c.valid.any())
        assert torch.equal(c.kv, live.kv)
        assert torch.equal(c.bv, live.bv)
    assert not torch.isnan(r["memory_off_nll"]).any()


def test_bos_off_variants_drop_only_the_bos_context():
    """M2: at gap 1 the bos-copy context IS the assert's gestalt, so every
    condition has a *_bos_off twin: identical memory, bos flag off."""
    model, docs, ids, mask, tmask, gap, sym = _world()
    spy = _Spy(model)
    r = loo.loo_readout(spy, docs, ids, mask, tmask, gap, sym, seed=0)
    by = _calls_by_step(spy, r)
    want = loo.CONDITIONS + tuple(f"{c}_bos_off" for c in loo.CONDITIONS)
    assert want == loo.ALL_CONDITIONS
    for calls in by.values():
        if len(calls) == 1:
            continue
        live = calls["live"]
        for c in loo.CONDITIONS:
            on, off = calls[c], calls[f"{c}_bos_off"]
            assert torch.equal(on.bv, live.bv), c
            assert not bool(off.bv.any()), c
            assert torch.equal(on.kv, off.kv) and torch.equal(on.valid, off.valid), c
    assert bool(by[1]["live"].bv.any())  # the fixture has a live bos context
    g1 = r["gap"] == 1
    assert g1.any()
    assert (r["live_bos_off_nll"][g1] != r["live_nll"][g1]).any()


def test_no_knockout_is_written_back():
    """The live memory trajectory under loo_readout is answer_readout's, step by
    step: no knockout leaks into the memory later steps attend over."""
    model, docs, ids, mask, tmask, gap, sym = _world()
    ref_spy = _Spy(model)
    _s003().answer_readout(ref_spy, ids, mask, tmask, gap, sym, cond="live")
    spy = _Spy(model)
    r = loo.loo_readout(spy, docs, ids, mask, tmask, gap, sym, seed=0)
    by = _calls_by_step(spy, r)
    assert len(ref_spy.calls) == S
    for t in range(S):
        assert torch.equal(by[t]["live"].kv, ref_spy.calls[t].kv), t
        assert torch.equal(by[t]["live"].valid, ref_spy.calls[t].valid), t
        assert torch.equal(by[t]["live"].bv, ref_spy.calls[t].bv), t


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
    assert all(not c.training for c in spy.calls)
    assert all(not c.grad for c in spy.calls)
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


def _live_states(model, ids, mask):
    """The honest FIFO loop, by hand: (kv, valid, bos_ctx, bos_valid) per step."""
    from rsr.model.tg.model import init_memory
    from rsr.model.tg.policy_loop import write_at

    B = ids.shape[0]
    model.eval()
    out = []
    with torch.no_grad():
        mem = init_memory(B, model.cfg)
        bos = torch.zeros(B, model.cfg.D)
        bv = torch.zeros(B, dtype=torch.bool)
        for t in range(S):
            out.append((mem.kv.clone(), mem.valid.clone(), bos.clone(), bv.clone()))
            o = model(ids[:, t], mask[:, t], mem.kv, mem.valid, bos, bv)
            mem = write_at(mem, o.srep, o.has_eos, torch.zeros(B).long(), t)
            bos, bv = o.srep, o.has_eos & (t + 1 < S)
    return out


def test_every_condition_matches_a_hand_recomputation():
    """Each condition's answer NLL equals a forward built by direct indexing of
    the live memory (not through knockout_kv): the knockout is the stated one,
    at the stated slot, with the stated donor -- for all 14 conditions."""
    from rsr.train.loop import lm_token_losses

    model, docs, ids, mask, tmask, gap, sym = _world()
    r = loo.loo_readout(model, docs, ids, mask, tmask, gap, sym, seed=0)
    states = _live_states(model, ids, mask)
    B = ids.shape[0]
    applied = dict.fromkeys(loo.ALL_CONDITIONS, 0)
    with torch.no_grad():
        for i in range(r["t"].numel()):
            b, t = int(r["row"][i]), int(r["t"][i])
            kv0, valid0, bos, bv0 = states[t]
            for cond in loo.ALL_CONDITIONS:
                base = cond.removesuffix("_bos_off")
                kv, valid, bv = kv0.clone(), valid0.clone(), bv0.clone()
                if cond.endswith("_bos_off"):
                    bv[b] = False
                own, ctrl = int(r["own_rank"][i]), int(r["ctrl_rank"][i])
                if base == "own_zero":
                    if own < 0:
                        continue
                    kv[b, own] = 0.0
                elif base == "own_resample":
                    d = int(r["own_donor_row"][i])
                    if d < 0:
                        continue
                    kv[b, own] = kv0[d, own]
                elif base == "ctrl_resample":
                    d = int(r["ctrl_donor_row"][i])
                    if d < 0:
                        continue
                    kv[b, ctrl] = kv0[d, ctrl]
                elif base == "all_slots_zeroed":
                    kv[b] = 0.0
                elif base == "all_slots_resample":
                    d = int(r["all_donor_row"][i])
                    if d < 0:
                        continue
                    kv[b] = kv0[d]
                elif base == "memory_off":
                    valid[b] = False
                o = model(ids[:, t], mask[:, t], kv, valid, bos, bv)
                per = lm_token_losses(o.logits, ids[:, t]).view(B, L - 1)
                want = float(per[b][tmask[b, t, 1:]][0])
                got = float(r[f"{cond}_nll"][i])
                assert got == pytest.approx(want, abs=1e-5, rel=0), (cond, i)
                applied[cond] += 1
    assert all(n > 0 for n in applied.values()), applied
    res = r["own_resident"]
    # and the knockout is not a no-op on this live memory
    assert (r["own_zero_nll"][res] - r["live_nll"][res]).abs().max() > 1e-4


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


# --------------------------------------------------------------------------- #
# W4-fix: donor exclusions, flags, statuses (review of PR #48)
# --------------------------------------------------------------------------- #


def test_pick_donor_is_called_with_the_required_exclusions(monkeypatch):
    """Own donors exclude the queried key and the answer object; control donors
    exclude the queried key, the control's own key and the answer object."""
    model, docs, ids, mask, tmask, gap, sym = _world()
    ann = loo.stream_annotations(docs, steps=S)
    seen = []
    real = loo.pick_donor

    def spy(ann_, step, valid, **kw):
        seen.append(kw)
        return real(ann_, step, valid, **kw)

    monkeypatch.setattr(loo, "pick_donor", spy)
    r = loo.loo_readout(model, docs, ids, mask, tmask, gap, sym, seed=0)
    rec = {(int(r["row"][i]), int(r["t"][i])): i for i in range(r["t"].numel())}
    roles = set()
    for kw in seen:
        b, t = kw["row"], kw["t"]
        i = rec[(b, t)]
        qk, qo = int(ann["fact_key"][b, t]), int(ann["object_id"][b, t])
        assert qk in kw["exclude_keys"] and qo in kw["exclude_objects"], kw
        if kw["role"] == loo._ROLE_CTRL:
            ck = int(ann["fact_key"][b, int(r["ctrl_sentence"][i])])
            assert ck in kw["exclude_keys"], kw
        roles.add(kw["role"])
    assert roles == {loo._ROLE_OWN, loo._ROLE_CTRL}


def test_own_donor_excludes_the_queried_key():
    """[A, twin(A)]: the twin holds the same questions with other answers, so its
    same-rank assert is the only candidate and must be refused."""
    base = _docs(1)[0]
    docs = (base, _twin(base, 999))
    model = _model()
    ids, mask, tmask, gap, sym = _encode(docs)
    r = loo.loo_readout(model, docs, ids, mask, tmask, gap, sym, seed=0)
    assert int(r["own_resident"].sum()) > 0
    assert (r["own_donor_row"] == -1).all()
    assert torch.isnan(r["own_resample_nll"]).all()


def test_all_slots_resample_donor_excludes_the_queried_key():
    base = _docs(1)[0]
    docs = (base, _twin(base, 999))
    model = _model()
    ids, mask, tmask, gap, sym = _encode(docs)
    r = loo.loo_readout(model, docs, ids, mask, tmask, gap, sym, seed=0)
    res = r["own_resident"]
    assert res.any()
    # a resident own assert has its twin (same key) in the twin's memory
    assert (r["all_donor_row"][res] == -1).all()


def _ctrl_rekeyed_world():
    """A record with a control present; every adjacent pending assert rewritten
    to ask the query's own question. -> (model, docs, tensors..., (b, t))."""
    model, docs, ids, mask, tmask, gap, sym = _world()
    r = loo.loo_readout(model, docs, ids, mask, tmask, gap, sym, seed=0)
    ann = loo.stream_annotations(docs, steps=S)
    i = int((r["ctrl_status"] == loo.CTRL_PRESENT).nonzero()[0])
    b, t, own = int(r["row"][i]), int(r["t"][i]), int(r["own_rank"][i])
    k = min(t, M)
    doc = docs[b]
    qa = {a: q for a, q in doc.pairs}
    n = 0
    for rk in (own - 1, own + 1):
        if 0 <= rk < k:
            s = t - k + rk
            if int(ann["kind"][b, s]) == 1 and int(ann["query_of"][b, s]) > t:
                q = qa[s]
                doc = _set_fact(doc, s, q, _key_of(doc, t), doc.sentences[q].answer)
                n += 1
    assert n > 0
    docs = tuple(doc if j == b else d for j, d in enumerate(docs))
    return (model, docs, *_encode(docs), (b, t))


def test_control_is_never_of_the_queried_key():
    model, docs, ids, mask, tmask, gap, sym, (b, t) = _ctrl_rekeyed_world()
    r = loo.loo_readout(model, docs, ids, mask, tmask, gap, sym, seed=0)
    i = int(((r["row"] == b) & (r["t"] == t)).nonzero()[0])
    assert int(r["ctrl_status"][i]) == loo.CTRL_NO_PENDING_ADJACENT
    ann = loo.stream_annotations(docs, steps=S)
    for j in (r["ctrl_status"] == loo.CTRL_PRESENT).nonzero().flatten().tolist():
        bj, tj = int(r["row"][j]), int(r["t"][j])
        cs = int(r["ctrl_sentence"][j])
        assert int(ann["fact_key"][bj, cs]) != int(ann["fact_key"][bj, tj])


def test_duplicate_key_in_document_is_flagged():
    model, docs, ids, mask, tmask, gap, sym, (b, t) = _ctrl_rekeyed_world()
    r = loo.loo_readout(model, docs, ids, mask, tmask, gap, sym, seed=0)
    i = int(((r["row"] == b) & (r["t"] == t)).nonzero()[0])
    assert bool(r["dup_key_in_doc"][i])
    m0, d0, *rest = _world()
    r0 = loo.loo_readout(m0, d0, *rest, seed=0)
    ann = loo.stream_annotations(d0, steps=S)
    for j in range(r0["t"].numel()):
        bj, tj = int(r0["row"][j]), int(r0["t"][j])
        key = _key_of(d0[bj], tj)
        n = sum(1 for a, q in d0[bj].pairs if _key_of(d0[bj], q) == key)
        assert bool(r0["dup_key_in_doc"][j]) == (n > 1)
    assert int(ann["fact_key"][b, t]) >= 0


def test_object_flags_and_the_donor_object_exclusion():
    model, docs, ids, mask, tmask, gap, sym = _world()
    r = loo.loo_readout(model, docs, ids, mask, tmask, gap, sym, seed=0)
    ann = loo.stream_annotations(docs, steps=S)
    obj = ann["object_id"]
    n_ctrl_same = 0
    for i in range(r["t"].numel()):
        b, t = int(r["row"][i]), int(r["t"][i])
        ans = int(obj[b, t])
        assert int(r["answer_object"][i]) == ans
        for who in ("own", "ctrl"):
            d = int(r[f"{who}_donor_row"][i])
            flag = bool(r[f"{who}_donor_same_object"][i])
            if d < 0:
                assert not flag
                continue
            ds = int(r[f"{who}_donor_sentence"][i])
            assert int(obj[d, ds]) != ans  # excluded
            assert not flag
        cs = int(r["ctrl_sentence"][i])
        same = cs >= 0 and int(obj[b, cs]) == ans
        assert bool(r["ctrl_same_object"][i]) == same
        n_ctrl_same += same
        d = int(r["all_donor_row"][i])
        held = range(max(0, t - M), t)
        has = d >= 0 and any(
            int(ann["kind"][d, s]) == 1 and int(obj[d, s]) == ans for s in held
        )
        assert bool(r["all_donor_has_object"][i]) == has
    assert n_ctrl_same > 0  # the fixture exercises the flag


def test_own_status_distinguishes_evicted_from_never_written():
    model, docs, ids, mask, tmask, gap, sym = _world()
    r0 = loo.loo_readout(model, docs, ids, mask, tmask, gap, sym, seed=0)
    i = int(r0["own_resident"].nonzero()[0])
    b, t, g = int(r0["row"][i]), int(r0["t"][i]), int(r0["gap"][i])
    ids, mask = ids.clone(), mask.clone()
    ids[b, t - g] = 0  # the assert sentence becomes padding: no EOS, no write
    mask[b, t - g] = False
    r = loo.loo_readout(model, docs, ids, mask, tmask, gap, sym, seed=0)
    j = int(((r["row"] == b) & (r["t"] == t)).nonzero()[0])
    assert int(r["own_status"][j]) == loo.OWN_NEVER_WRITTEN
    assert not bool(r["own_resident"][j])
    for k in range(r0["t"].numel()):
        st = int(r0["own_status"][k])
        if bool(r0["own_resident"][k]):
            assert st == loo.OWN_RESIDENT
        else:
            assert st == loo.OWN_EVICTED and int(r0["gap"][k]) > M


def test_seed_mix_has_no_role_collision():
    assert loo._mix(0, 0, 1, 0) != loo._mix(0, 0, 0, 131)
    vals = {
        loo._mix(s, d, t, role)
        for s in range(2)
        for d in range(3)
        for t in range(4)
        for role in (0, 1, 2, 131, 147, 400)
    }
    assert len(vals) == 2 * 3 * 4 * 6


def test_loo_delta_loss_zero_mode_needs_no_annotations():
    model, docs, ids, mask, *_ = _world(n=3)
    a = loo.loo_delta_loss(model, docs, ids, mask, mode="zero", seed=0)
    b = loo.loo_delta_loss(model, None, ids, mask, mode="zero", seed=0)
    assert torch.equal(a["delta"].nan_to_num(9.0), b["delta"].nan_to_num(9.0))
    with pytest.raises(ValueError):
        loo.loo_delta_loss(model, None, ids, mask, mode="resample", seed=0)


def test_loo_delta_loss_padding_step_is_nan_not_zero():
    model, docs, ids, mask, *_ = _world(n=3)
    ids, mask = ids.clone(), mask.clone()
    ids[0, 5] = 0
    mask[0, 5] = False
    out = loo.loo_delta_loss(model, docs, ids, mask, mode="zero", seed=0)
    assert math.isnan(float(out["live_loss"][0, 5]))
    assert torch.isnan(out["delta"][0, 5]).all()
    assert not math.isnan(float(out["live_loss"][1, 5]))


def test_loo_delta_loss_resample_excludes_the_slots_own_key():
    """[A, twin(A)]: an assert/query slot's only donor is its twin (same key), so
    it gets none; a filler slot (no key) still gets one."""
    base = _docs(1)[0]
    docs = (base, _twin(base, 999))
    model = _model()
    ids, mask, *_ = _encode(docs)
    out = loo.loo_delta_loss(model, docs, ids, mask, mode="resample", seed=0)
    ann = loo.stream_annotations(docs, steps=S)
    ss = out["slot_sentence"]
    n_fill = 0
    for b in range(2):
        for t in range(S):
            for i in range(M):
                s = int(ss[b, t, i])
                if s < 0:
                    continue
                keyed = int(ann["fact_key"][b, s]) >= 0
                assert (int(out["donor_row"][b, t, i]) >= 0) == (not keyed)
                n_fill += not keyed
    assert n_fill > 0
