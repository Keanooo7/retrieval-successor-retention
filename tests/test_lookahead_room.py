"""experiments/lookahead-room/run.py (W10): the pure pieces on hand-built inputs, and
the probe machinery on a tiny untrained model. No checkpoint is read here."""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

from rsr.baselines.fifo import FIFOPolicy
from rsr.data.synthetic import SyntheticConfig, generate
from rsr.metrics.headroom import simulate
from rsr.model.tg import TGConfig, TGModel
from rsr.model.tg.policy_loop import run_policy_loop
from rsr.train.loop import answer_targets, build_vocab, encode

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments" / "lookahead-room" / "run.py"
NAN = float("nan")


@pytest.fixture(scope="module")
def lr():
    spec = importlib.util.spec_from_file_location("lookahead_room_run", RUN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lookahead_room_run"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def doc():
    return generate(SyntheticConfig(sentences_per_document=48, seed=0))[0]


# --------------------------------------------------------------------------- #
# targets
# --------------------------------------------------------------------------- #


def test_the_discounted_return_sums_the_future_within_the_stream(lr):
    """S = 3, one sentence column (i = 0) defined from t = 1: G[1] = 1 + g*2,
    G[2] = 2; the undefined entry stays NaN."""
    D = torch.tensor([[NAN], [1.0], [2.0]], dtype=torch.float64)
    G = lr.discounted_returns(D, 0.5)
    assert math.isnan(G[0, 0])
    assert G[1, 0] == pytest.approx(2.0)
    assert G[2, 0] == pytest.approx(2.0)
    assert torch.equal(lr.discounted_returns(D, 0.0)[1:], D[1:])


def test_next_step_is_the_following_row_and_zero_at_the_end(lr):
    D = torch.tensor([[NAN], [1.0], [3.0]], dtype=torch.float64)
    T = lr.next_step(D)
    assert math.isnan(T[0, 0])
    assert T[1, 0] == 3.0 and T[2, 0] == 0.0


def test_the_literal_target_is_zero_where_fifo_did_not_hold_the_sentence(lr):
    D = torch.tensor([[NAN, NAN], [0.4, NAN], [0.7, 0.3]], dtype=torch.float64)
    res = torch.tensor([[False, False], [True, False], [False, True]])
    T = lr.targets(D, res)
    assert T["lit_g0"][2, 0] == 0.0 and T["lit_g0"][2, 1] == pytest.approx(0.3)
    assert T["rule_g0"][2, 0] == pytest.approx(0.7)
    assert T["rule_g09"][1, 0] == pytest.approx(0.4 + 0.9 * 0.7)
    assert T["lit_g09"][1, 0] == pytest.approx(0.4)


def test_a_target_rule_evicts_the_argmin_and_ties_go_to_the_oldest(lr, doc):
    """Target = written step: the oldest has the smallest target -> FIFO exactly.
    A constant target ties everywhere -> also FIFO (ties to slot 0)."""
    Sd = len(doc.sentences)
    idx = torch.arange(Sd, dtype=torch.float64).expand(Sd, Sd).clone()
    fifo = simulate(doc, FIFOPolicy(), 16)["queries"]
    assert simulate(doc, lr.TargetPolicy(idx), 16)["queries"] == fifo
    const = torch.ones(Sd, Sd, dtype=torch.float64)
    assert simulate(doc, lr.TargetPolicy(const), 16)["queries"] == fifo
    newest_first = simulate(doc, lr.TargetPolicy(-idx), 16)["queries"]
    assert newest_first != fifo


def test_a_nan_target_for_a_live_sentence_raises(lr, doc):
    Sd = len(doc.sentences)
    T = torch.full((Sd, Sd), NAN, dtype=torch.float64)
    with pytest.raises(ValueError, match="NaN target"):
        simulate(doc, lr.TargetPolicy(T), 16)


# --------------------------------------------------------------------------- #
# classes, AUC, summary
# --------------------------------------------------------------------------- #


def test_slot_classes_follow_the_query_schedule(lr, doc):
    a, q = doc.pairs[0]
    assert lr.slot_class(doc, a, q - 2) == ("pending", 2)
    assert lr.slot_class(doc, a, q) == ("querying", 0)
    assert lr.slot_class(doc, a, q + 1) == ("answered", -1)
    fillers = [s.index for s in doc.sentences if s.kind == "filler"]
    assert lr.slot_class(doc, fillers[0], 47) == ("filler", None)
    assert lr.slot_class(doc, q, 47)[0] == "filler"  # a written query is not a fact


def test_auc_counts_ties_as_half(lr):
    assert lr.auc([2.0], [1.0]) == 1.0
    assert lr.auc([1.0], [2.0]) == 0.0
    assert lr.auc([1.0], [1.0]) == 0.5
    assert lr.auc([1.0, 3.0], [2.0]) == 0.5
    assert lr.auc([], [1.0]) is None


# --------------------------------------------------------------------------- #
# the decision rule and the bootstrap
# --------------------------------------------------------------------------- #


def _ci(p, lo, hi):
    return {"point": p, "lo": lo, "hi": hi}


def test_the_rule_is_the_prereg_rule(lr):
    ok = {s: _ci(0.05, 0.001, 0.1) for s in (0, 1, 2)}
    assert lr.classify(ok) == "TESTABLE_HERE"
    assert lr.classify({**ok, 2: _ci(0.049, 0.01, 0.09)}) == "PARTIAL"
    assert lr.classify({**ok, 1: _ci(0.06, 0.0, 0.1)}) == "PARTIAL"
    none = {s: _ci(0.0, -0.01, 0.0199) for s in (0, 1, 2)}
    assert lr.classify(none) == "NO_ROOM"
    assert lr.classify({**none, 0: _ci(0.0, -0.01, 0.02)}) == "PARTIAL"
    neg = {s: _ci(-0.1, -0.2, -0.05) for s in (0, 1, 2)}
    assert lr.classify(neg) == "NO_ROOM"


def test_a_failed_control_makes_the_run_inconclusive(lr):
    room = {c: {s: _ci(0.1, 0.05, 0.2) for s in (0, 1, 2)} for c in (2500, 3000)}
    d = lr.decide(room, [])
    assert d["classification"] == "TESTABLE_HERE" and d["exit"] == 0
    d = lr.decide(room, ["C4 failed"])
    assert d["classification"] == "INCONCLUSIVE" and d["exit"] == 3
    d = lr.decide({3000: room[3000]}, [])
    assert d["classification"] == "INCONCLUSIVE"


def test_the_bootstrap_is_paired_and_ratio_of_sums(lr):
    a = [(10, 5), (2, 2), (4, 0)]
    b = [(10, 5), (2, 2), (4, 0)]
    c = [(10, 10), (2, 2), (4, 4)]
    out = lr.paired_bootstrap(
        {"a": a, "b": b, "c": c}, {"ab": ("a", "b"), "ca": ("c", "a")}, 0
    )
    assert out["ab"] == {"point": 0.0, "lo": 0.0, "hi": 0.0}
    assert out["ca"]["point"] == pytest.approx(1.0 - 7 / 16)
    assert 0.0 <= out["ca"]["lo"] < out["ca"]["point"] < out["ca"]["hi"] <= 1.0


# --------------------------------------------------------------------------- #
# the model side, on a tiny untrained model
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def tiny():
    S, M, L = 12, 4, 16
    docs = generate(SyntheticConfig(sentences_per_document=48, seed=0))[:1]
    vocab = build_vocab(docs)
    ids, mask = encode(docs, vocab, max_tokens=L, steps=S)
    tm, _ = answer_targets(docs, vocab, max_tokens=L, steps=S)
    torch.manual_seed(0)
    cfg = TGConfig(
        D=32, V=4 + len(vocab), F=86, max_sentence_tokens=L,
        max_sentences_in_short_term=M, pad_id=0, bos_id=1, eos_id=2, eod_id=3,
    )  # fmt: skip
    return TGModel(cfg).eval(), docs[0], ids, mask, tm, S, M


def test_probes_fill_demand_for_every_past_sentence_and_the_identity_probe_holds(
    lr, tiny
):
    model, d, ids, mask, tm, S, M = tiny
    r = lr.run_fifo(model, d, ids, mask, tm, None, M, S)
    D, res = r["D"], r["resident"]
    for t in range(S):
        assert not torch.isnan(D[t, :t]).any(), t
        assert torch.isnan(D[t, t:]).all(), t
        want = list(range(max(0, t - M), t))
        assert torch.nonzero(res[t]).flatten().tolist() == want
    assert r["identity_worst"] <= lr.IDENTITY_TOL
    assert r["sum_worst"] <= lr.SUM_TOL
    assert r["n_probe_rows"] == sum(1 + (t - M) for t in range(M, S))


def test_the_probes_never_feed_back_into_the_fifo_rollout(lr, tiny):
    """Answers and NLLs with probes on == a plain FIFO loop with capture off."""
    model, d, ids, mask, tm, S, M = tiny
    with_probes = lr.run_fifo(model, d, ids, mask, tm, None, M, S)["answers"]
    plain: list = []
    with torch.no_grad():
        run_policy_loop(
            model, ids, mask, torch.full((1,), S), FIFOPolicy(),
            step_fn=lr._answer_step_fn(tm, None, plain),
        )  # fmt: skip
    assert len(plain) > 0
    assert with_probes == plain


def test_a_probe_reads_the_swapped_in_sentence_not_the_resident_one(lr, tiny):
    """At a full step, the probe for an evicted sentence differs from the identity
    probe's value (else the swap did nothing)."""
    model, d, ids, mask, tm, S, M = tiny
    r = lr.run_fifo(model, d, ids, mask, tm, None, M, S)
    t = S - 1
    probed = [float(r["D"][t, i]) for i in range(t - M)]
    assert any(abs(p - float(r["D"][t, t - M])) > 1e-6 for p in probed)


def test_online_g0_evicts_its_own_argmin_and_replays_model_free(lr, tiny):
    model, d, ids, mask, tm, S, M = tiny
    o = lr.run_online(model, d, ids, mask, tm, None, M, S)
    assert len(o["victims"]) == S - M
    assert o["sum_worst"] <= lr.SUM_TOL
    # the replay runs the full 48-sentence doc; compare the steps the model ran
    sub = type(d)(doc_id=d.doc_id, sentences=d.sentences[:S],
                  pairs=tuple(p for p in d.pairs if p[1] < S))  # fmt: skip
    rep = simulate(sub, lr.Scripted(o["victims"]), M)["queries"]
    assert [(q["gap"], q["hit"]) for q in rep] == o["hits"]


# --------------------------------------------------------------------------- #
# controls and hygiene
# --------------------------------------------------------------------------- #


def test_control2_reproduces_the_red_team_on_seed_0(lr):
    c = lr.control2(0)
    assert c["ok"], c
    assert c["victims_differing"] == 0


def test_the_extended_set_is_disjoint(lr):
    assert lr.disjointness()["ok"]
    assert not lr.disjointness((4000, 4200))["ok"]


def test_the_run_never_calls_record(lr):
    tree = ast.parse(RUN.read_text())
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and getattr(n.func, "attr", getattr(n.func, "id", None)) == "record"
    ]  # fmt: skip
    assert calls == []
