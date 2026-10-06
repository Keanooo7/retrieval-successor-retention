"""experiments/cf-ceiling (B0-CF): the Delta instrument on a tiny random model, and the
pure pieces of run.py on hand-built inputs. No checkpoint, no D.pt, and no T0 manifest
is read here."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

from rsr.data.synthetic import SyntheticConfig, generate
from rsr.metrics import loo
from rsr.model.tg import TGConfig, TGModel
from rsr.model.tg.model import init_memory
from rsr.model.tg.policy_loop import write_at
from rsr.train.loop import build_vocab, encode

ROOT = Path(__file__).resolve().parents[1]
NAN = float("nan")
S, L, M = 16, 10, 4


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cf():
    return _load("cf_ceiling_cfdelta", "experiments/cf-ceiling/cfdelta.py")


@pytest.fixture(scope="module")
def run():
    return _load("cf_ceiling_run", "experiments/cf-ceiling/run.py")


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


@pytest.fixture(scope="module")
def world():
    docs = _docs(8)
    vmap = build_vocab(_docs(64))
    ids, mask = encode(docs, vmap, max_tokens=L, steps=S)
    torch.manual_seed(0)
    cfg = TGConfig(
        D=16,
        H=2,
        N=4,
        V=4 + len(vmap),
        block_config=("S", "C", "S", "C"),
        srep_extraction_layer=2,
        max_sentence_tokens=L,
        max_sentences_in_short_term=M,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        eod_id=3,
    )
    return TGModel(cfg), docs, ids, mask


@pytest.fixture(scope="module")
def measured(cf, world):
    model, docs, ids, mask = world
    return cf.cf_delta(model, docs, ids, mask, seed=0)


def _past(n):
    t = torch.arange(n).unsqueeze(1)
    i = torch.arange(n).unsqueeze(0)
    return i < t


# --------------------------------------------------------------------------- #
# the instrument (cfdelta.py)
# --------------------------------------------------------------------------- #


def test_internal_checks_hold_on_a_random_model(measured):
    c = measured["checks"]
    assert c["live_loss_equal_loo"] is True
    assert c["slot_sentence_equal_loo"] is True
    assert c["fifo_memory_ok"] is True
    assert c["identity_worst"] <= 1e-6
    B = measured["Delta"].shape[0]
    assert c["n_evicted_cells"] == B * sum(t - M for t in range(M, S))
    assert c["n_resident_cells"] == B * sum(min(t, M) for t in range(S))


def test_delta_layout_is_d_pts(measured):
    past = _past(S)
    for X in (measured["Delta"], measured["delta_shift"]):
        for b in range(X.shape[0]):
            assert torch.isfinite(X[b][past]).all()
            assert torch.isnan(X[b][~past]).all()


def test_resident_cells_are_loos_delta_and_missing_is_imputed_zero(measured):
    d, ss = measured["delta_res"], measured["slot_sentence"]
    B = d.shape[0]
    n_checked = 0
    for b in range(B):
        for t in range(S):
            for j in range(M):
                s = int(ss[b, t, j])
                if s < 0:
                    continue
                v = float(d[b, t, j])
                got = float(measured["Delta"][b, t, s])
                if math.isnan(v):
                    assert got == 0.0 and bool(measured["missing"][b, t, s])
                else:
                    assert got == v and not bool(measured["missing"][b, t, s])
                    n_checked += 1
    assert n_checked > 0


def test_evicted_cell_equals_an_independent_reinstatement_forward(cf, world, measured):
    """Re-derive Delta[b, t, i] for an evicted i by an independent FIFO loop: the
    loss with FIFO's memory minus the loss with slot 0 := sentence i's gestalt."""
    model, _, ids, mask = world
    model.eval()
    B = ids.shape[0]
    t_star, i_star = S - 1, 2  # i < t - M: evicted
    with torch.no_grad():
        cfg = model.cfg
        mem = init_memory(B, cfg, device=ids.device, dtype=model.embed.weight.dtype)
        bos_ctx = torch.zeros(B, cfg.D)
        bos_valid = torch.zeros(B, dtype=torch.bool)
        gest = {}
        for t in range(S):
            if t == t_star:
                kv = mem.kv.clone()
                kv[:, 0] = gest[i_star]
                live = model(ids[:, t], mask[:, t], mem.kv, mem.valid, bos_ctx, bos_valid)
                rein = model(ids[:, t], mask[:, t], kv, mem.valid, bos_ctx, bos_valid)
                sl = cf._sent_loss
                want = sl(live.logits, ids[:, t], mask[:, t]) - sl(
                    rein.logits, ids[:, t], mask[:, t]
                )
                break
            out = model(ids[:, t], mask[:, t], mem.kv, mem.valid, bos_ctx, bos_valid)
            gest[t] = out.srep
            mem = write_at(
                mem, out.srep, out.has_eos, torch.zeros(B, dtype=torch.long), t
            )
            bos_ctx, bos_valid = out.srep, out.has_eos & (t + 1 < S)
    got = measured["Delta"][:, t_star, i_star]
    assert bool(measured["evicted"][:, t_star, i_star].all())
    assert torch.allclose(got, want, atol=1e-6, rtol=0)


def test_shift_rebaselines_only_evicted_cells_by_slot0s_resident_delta(measured):
    D, Sh, ev = measured["Delta"], measured["delta_shift"], measured["evicted"]
    r0 = torch.nan_to_num(measured["delta_res"][:, :, 0], nan=0.0)
    past = _past(S)
    for b in range(D.shape[0]):
        res = past & ~ev[b]
        assert torch.equal(Sh[b][res], D[b][res])
        for t in range(M, S):
            for i in range(t - M):
                assert float(Sh[b, t, i]) == float(D[b, t, i] + r0[b, t])


def test_assemble_on_a_hand_built_input(cf):
    # B=1, S=4, M=2. Steps 0..1 underfull, 2..3 full; FIFO memory [t-2, t-1].
    ss = torch.tensor([[[-1, -1], [0, -1], [0, 1], [1, 2]]])
    dr = torch.full((1, 4, 2), NAN, dtype=torch.float64)
    dr[0, 1, 0] = 0.5
    dr[0, 2, 0], dr[0, 2, 1] = 0.25, NAN  # missing donor for sentence 1 at t=2
    dr[0, 3, 0], dr[0, 3, 1] = 0.125, 1.0
    probe = {2: torch.tensor([[3.0]]), 3: torch.tensor([[3.0, 2.0]])}  # t=3: i=0 -> 2.0
    out = cf.assemble(dr, ss, probe, 2)
    D = out["Delta"][0]
    assert D[1, 0] == 0.5 and D[2, 0] == 0.25 and D[2, 1] == 0.0
    assert bool(out["missing"][0, 2, 1]) and int(out["missing"].sum()) == 1
    assert D[3, 1] == 0.125 and D[3, 2] == 1.0
    assert D[3, 0] == 1.0 and bool(out["evicted"][0, 3, 0])  # 3.0 - 2.0
    assert out["delta_shift"][0, 3, 0] == 1.125  # + Delta_res[3][slot 0]
    assert int(out["evicted"].sum()) == 1
    assert torch.isnan(D[0]).all() and torch.isnan(D[1, 1:]).all()


# --------------------------------------------------------------------------- #
# run.py's pure pieces
# --------------------------------------------------------------------------- #


def test_prereg_constants(run):
    assert run.GAMMAS == {"g0": 0.0, "g05": 0.5, "g08": 0.8, "g09": 0.9}
    assert run.SEEDS == [0, 1, 2] and run.CHECKPOINTS == (3000, 2500)
    assert run.HEADLINE == 3000 and run.BATCH == 32 and run.N_DOCS == 1088
    assert run.MIN_COVERAGE == 0.80 and run.RI_CONTROL == ("g0", "g09")
    assert run.GC.POINTS_LO == 0.80 and run.GC.PARTIAL_POINT == 0.50


@pytest.mark.parametrize(
    ("points", "los", "want"),
    [
        ([0.9, 0.9, 0.9], [0.8, 0.81, 0.85], "POINTS"),
        ([0.9, 0.9, 0.9], [0.8, 0.79, 0.85], "PARTIAL"),
        ([0.5, 0.6, 0.7], [0.1, 0.2, 0.3], "PARTIAL"),
        ([0.5, 0.49, 0.7], [0.1, 0.2, 0.3], "MISSES"),
        ([0.9, 0.9, NAN], [0.85, 0.85, NAN], "MISSES"),
    ],
)
def test_label(run, points, los, want):
    assert run.label(points, los) == want


def test_cf_class(run):
    base = dict.fromkeys(run.GAMMAS, "MISSES")
    assert run.cf_class(base) == "CF-MISSES"
    assert run.cf_class({**base, "g08": "PARTIAL"}) == "CF-MIXED"
    assert run.cf_class({**base, "g05": "PARTIAL", "g09": "POINTS"}) == "CF-POINTS"
    with pytest.raises(ValueError):
        run.cf_class({"g0": "MISSES"})


def test_batches_keep_u_order_in_34_chunks_of_32(run):
    xs = list(range(1088))
    bs = run.batches(xs)
    assert len(bs) == 34 and all(len(b) == 32 for b in bs)
    assert [x for b in bs for x in b] == xs


def test_coverage_counts_by_kind_and_full_memory(run):
    ss = torch.tensor([[[0, -1], [0, 1], [1, 2]]])  # S=3, M=2; full at t=2
    kind = torch.tensor([[0, 1, 2]])  # filler, assert, query
    missing = torch.zeros(1, 3, 3, dtype=torch.bool)
    missing[0, 2, 2] = True
    c = run.coverage(missing, ss, kind, m=2)
    assert c["all"] == [5, 1] and c["full"] == [2, 1]
    assert c["query"] == [1, 1] and c["assert"] == [2, 0] and c["filler"] == [2, 0]


def test_instrument_failures(run):
    ok = {
        "live_loss_equal_loo": True,
        "slot_sentence_equal_loo": True,
        "fifo_memory_ok": True,
        "identity_worst": 0.0,
    }
    assert run.instrument_failures(ok) == []
    assert run.instrument_failures({**ok, "identity_worst": 1e-3})
    assert run.instrument_failures({**ok, "identity_worst": NAN})
    assert run.instrument_failures({**ok, "fifo_memory_ok": False})


def test_c3_exact_equality_and_any_difference(run):
    rows = {}
    mine = {}
    for g in run.RI_CONTROL:
        mine[g] = {
            "by_bucket": {b: 0.5 for b in run.BUCKETS},
            "cap": {"point": 0.1, "lo": 0.0, "hi": 0.2},
        }
        for b in run.BUCKETS:
            rows[f"B.ckpt3000.{g}.hit.{b}"] = {"samples": [0.5, 0.5, 0.5]}
        for f, v in (("point", 0.1), ("lo", 0.0), ("hi", 0.2)):
            rows[f"B.ckpt3000.{g}.cap.{f}"] = {"samples": [v] * 3}
    rows["U.hit.fifo.all"] = {"samples": [0.8] * 3}
    rows["U.hit.factfiller.all"] = {"samples": [0.9] * 3}
    refs = {"fifo": 0.8, "factfiller": 0.9}
    assert run.c3_failures(mine, refs, rows, 3000, 1) == []
    mine["g09"]["cap"]["lo"] = 0.0 + 1e-15
    assert len(run.c3_failures(mine, refs, rows, 3000, 1)) == 1
    assert run.c3_failures(mine, refs, {}, 3000, 1)  # absent rows fail


def test_gamma_sweep_ledger_has_every_row_c3_reads(run):
    import json

    keys = {
        r["key"]
        for r in json.loads((ROOT / run.GC_LEDGER).read_text())["rows"]
        if isinstance(r.get("key"), str)
    }
    for c in run.CHECKPOINTS:
        for g in run.RI_CONTROL:
            for b in run.BUCKETS:
                assert f"B.ckpt{c}.{g}.hit.{b}" in keys
            for f in ("point", "lo", "hi"):
                assert f"B.ckpt{c}.{g}.cap.{f}" in keys
    assert {"U.hit.fifo.all", "U.hit.factfiller.all"} <= keys


def test_class_stats_cover_every_full_memory_cell(run):
    doc = generate(SyntheticConfig(sentences_per_document=48, seed=0))[0]
    D = torch.zeros(48, 48, dtype=torch.float64)
    ev = torch.zeros(48, 48, dtype=torch.bool)
    for t in range(16, 48):
        ev[t, : t - 16] = True
    st = run.class_stats(doc, D, ev, m=16)
    assert sum(v[2] for v in st.values()) == sum(range(16, 48))
    assert sum(v[2] for k, v in st.items() if k.endswith("|evicted")) == sum(
        t - 16 for t in range(16, 48)
    )


def test_delta_key_depends_on_documents_and_checkpoint(run):
    a = run.delta_key(3000, 0, "x", [1, 2])
    assert a == run.delta_key(3000, 0, "x", [1, 2])
    assert a != run.delta_key(3000, 0, "x", [1, 3])
    assert a != run.delta_key(3000, 0, "y", [1, 2])
    assert a != run.delta_key(2500, 0, "x", [1, 2])


def test_manifest_rels_name_checkpoints_and_d_pt_in_t0(run):
    rels = run.all_rels()
    assert len(rels) == 12
    assert ".worktrees/fresh-stream/runs/fresh-stream/B/seed1/ckpt-002500.pt" in rels


class _Led:
    def __init__(self):
        self.notes, self.st = {}, None

    def note(self, k, v, *, how):
        self.notes[k] = v

    def status(self, s):
        self.st = s

    def run_meta(self, **kw):
        pass


def test_c1_missing_manifest_is_did_not_run_before_any_read(run, tmp_path):
    led = _Led()
    rc = run.execute(led, tmp_path / "absent.sha256", tmp_path, tmp_path / "raw", None)
    assert rc == run.Exit.DID_NOT_RUN and led.st == "did_not_run"
    assert led.notes["C1.start"]["ok"] is False and "C1.end" not in led.notes


def test_c1_is_checked_again_at_the_end(run, tmp_path, monkeypatch):
    """T0 manifest at start AND end: a mismatch at the end is exit 3."""
    calls = []

    def vm(manifest, rels, base):
        calls.append(len(rels))
        return {"ok": len(calls) == 1}

    class _Doc:
        def __init__(self, i):
            self.doc_id = i

    past = _past(48)
    good = torch.where(past, torch.zeros(48, 48, dtype=torch.float64), NAN)
    monkeypatch.setattr(run.B0, "verify_manifest", vm)
    monkeypatch.setattr(run.LR, "documents", lambda s: ([_Doc(0)], None, 0, {"ok": True}))
    monkeypatch.setattr(run.B0, "load_D", lambda base, c, s: {0: good})
    monkeypatch.setattr(run, "measure_delta", lambda c, s, b, r, lim: {"failures": []})
    monkeypatch.setattr(run, "c3_failures", lambda *a: [])

    def fake_score(seed, deltas, Ds, limit):
        arm = {g: {"passthrough_ok": True} for g in run.GAMMAS}
        return {
            "n_documents": run.N_DOCS,
            "refs": {"fifo": {"all": 0.8}, "factfiller": {"all": 0.9}},
            "ckpt": {c: {"cf": arm, "cfshift": arm, "ri": {}} for c in run.CHECKPOINTS},
        }

    monkeypatch.setattr(run, "score_seed", fake_score)
    led = _Led()
    rc = run.execute(led, tmp_path / "m", tmp_path, tmp_path / "raw", None)
    assert len(calls) == 2 and all(n == 12 for n in calls)
    assert rc == run.Exit.DID_NOT_RUN
    assert "C1 at end: manifest mismatch" in led.notes["controls_failed"]
    assert loo.KIND  # the instrument module imported, not copied
