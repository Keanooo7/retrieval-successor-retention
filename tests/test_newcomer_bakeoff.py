"""experiments/newcomer-bakeoff (run.py + analysis.py): the grace wrapper, the R1
target, the content-labelled log, the readouts and the decision table, on
hand-built inputs and a tiny untrained model.

Governing text: experiments/newcomer-bakeoff/PREREG.md (801635c). No checkpoint is
read, and no document of EVAL_NB = [990000, 992048) is generated here: the tiny
fixtures use S = 12 documents with ids inside B2's FIT ranges only.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

from rsr.data.synthetic import SyntheticConfig, _generate_document
from rsr.model.tg import TGConfig, TGModel
from rsr.retention.policy import MemoryState
from rsr.train.loop import build_vocab

ROOT = Path(__file__).resolve().parents[1]
NAN = float("nan")
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
def nb():
    return _load("_nb_run_under_test", "experiments/newcomer-bakeoff/run.py")


@pytest.fixture(scope="module")
def an(nb):
    return nb.AN


@pytest.fixture(scope="module")
def b2(nb):
    return nb.B2


def _tiny_docs(ids):
    cfg = SyntheticConfig(sentences_per_document=TS, max_gap=8, heavy_tail_min=5, seed=0)
    return [_generate_document(i, cfg) for i in ids]


@pytest.fixture(scope="module")
def tiny():
    train = _tiny_docs(range(920000, 920004))
    val = _tiny_docs(range(936000, 936003))
    vmap = build_vocab(train + val)
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
    return {"model": TGModel(cfg).eval(), "vmap": vmap, "train": train, "val": val}


@pytest.fixture(scope="module")
def caps(b2, tiny):
    kw = {"vmap": tiny["vmap"], "S": TS, "m": TM, "L": TL}
    return {
        "train": [b2.capture_doc(tiny["model"], d, **kw) for d in tiny["train"]],
        "val": [b2.capture_doc(tiny["model"], d, **kw) for d in tiny["val"]],
    }


def _state(gestalts, written_at, step):
    n = gestalts.shape[0]
    return MemoryState(
        gestalts=gestalts,
        written_at=torch.as_tensor(written_at, dtype=torch.long),
        live=torch.ones(n, dtype=torch.bool),
        step=step,
    )


def _probe(b2, doc, w):
    return b2.ProbeArgminPolicy(w, "bilinear", doc=doc, model_seed=0, arm="psiU", S=TS)


def _bilinear_w_scoring_first_coord(d, weights):
    """A w whose psi for slot k is `weights · s_k` through the `s_i` block."""
    w = torch.zeros(d * d + 2 * d, dtype=torch.float64)
    w[d * d : d * d + d] = torch.as_tensor(weights, dtype=torch.float64)
    return w


# --------------------------------------------------------------------------- #
# §2 ranges; §5 registry reads
# --------------------------------------------------------------------------- #


def test_eval_nb_is_fresh_and_disjoint_from_every_claimed_range(nb):
    assert nb.EVAL_NB == (990000, 992048) and nb.N_E == 2048
    assert nb.range_problems() == []
    # B2's claimed EVAL is avoided even where B2 never ran a document
    probs = nb.range_problems((979000, 981048))
    assert any("B2_EVAL" in p for p in probs)
    assert any("E0d" in p or "E0D" in p for p in nb.range_problems((900000, 902048)))


def test_eval_nb_past_the_seed_stride_is_refused(nb):
    probs = nb.range_problems((999000, 1001048))
    assert any("alias" in p for p in probs)


def test_the_eval_guard_refuses_an_id_outside_eval_nb(nb, b2):
    with pytest.raises(b2.RangeError):
        nb.require_eval_nb([940000])
    nb.require_eval_nb([990000, 992047])


def test_shards_tile_eval_nb_in_order(nb):
    b = [nb.shard_bounds(j, 4) for j in range(4)]
    assert b[0][0] == 990000 and b[-1][1] == 992048
    assert all(b[j][1] == b[j + 1][0] for j in range(3))


def test_k_and_lambda_shadow_are_read_from_the_registry(nb):
    from rsr import constants as RC

    assert nb.K_SHADOW == RC.get("K", "synthetic") == 40
    assert nb.LAMBDA_SHADOW == RC.get("lambda_shadow") == 0.5
    assert nb.R1_LAMBDAS == {"R1": 0.5, "R1L1": 1.0}


def test_the_arm_set_is_the_prereg_table_s_22(an):
    # PREREG §3 table: 22 arms (its "24 arm-runs" line is a miscount; erratum E1)
    assert len(an.ARMS) == 22 and len(set(an.ARMS)) == 22
    for a in (
        "psiC",
        "psiU",
        "graceU1",
        "graceU2",
        "graceU4",
        "psiR1",
        "psiR1L1",
        "psiU+",
        "psiC+",
        "fifo",
        "lru",
        "ageU",
        "ageC",
        "ageR1",
        "ageR1L1",
        "kind",
        "oracle",
        *(f"random{k}" for k in range(5)),
    ):
        assert a in an.ARMS


# --------------------------------------------------------------------------- #
# §5: the R1 target
# --------------------------------------------------------------------------- #


def _eq_nan(a, b):
    return torch.equal(torch.isnan(a), torch.isnan(b)) and torch.equal(
        torch.nan_to_num(a), torch.nan_to_num(b)
    )


def test_r1_at_zero_weight_is_the_censored_target(nb, b2, caps):
    for cap in caps["train"]:
        X, _ = nb.r1_X(cap.D[0], cap.resident, 0.0, nb.K_SHADOW)
        assert _eq_nan(X, b2.literal(cap.D[0], cap.resident))


def test_r1_at_full_weight_and_open_window_is_u_s_demand(nb, caps):
    for cap in caps["train"]:
        X, n = nb.r1_X(cap.D[0], cap.resident, 1.0, 10**6)
        assert n == 0 and _eq_nan(X, cap.D[0])


def test_r1_at_half_weight_is_resident_plus_half_the_shadow(nb, caps):
    for cap in caps["train"]:
        D0, res = cap.D[0], cap.resident
        X, _ = nb.r1_X(D0, res, 0.5, 10**6)
        want = torch.where(res, torch.nan_to_num(D0), 0.5 * torch.nan_to_num(D0))
        want = want.masked_fill(torch.isnan(D0), NAN)
        assert _eq_nan(X, want)
        # and it is not U's demand wherever an evicted sentence has demand
        gone = ~res & ~torch.isnan(D0) & (torch.nan_to_num(D0) != 0)
        assert bool(gone.any()) and not _eq_nan(X, D0)


def test_r1_window_truncates_at_k(nb, caps):
    cap = caps["train"][0]
    D0, res = cap.D[0], cap.resident
    S = D0.shape[0]
    X, n = nb.r1_X(D0, res, 0.5, 2)
    for i in range(S):
        t_e = next((t for t in range(i + 1, S) if not bool(res[t, i])), None)
        if t_e is None:
            continue
        # FIFO drops i right after it has been read at age M (B2's convention)
        assert t_e == i + TM + 1
        for t in range(t_e, S):
            want = 0.5 * float(D0[t, i]) if t - t_e < 2 else 0.0
            assert float(X[t, i]) == pytest.approx(want, abs=0)
    assert n == sum(max(0, S - (i + TM + 1) - 2) for i in range(S))


def test_the_r1l1_target_equals_u_s_target_on_c_rows(nb, b2, caps):
    nb.install_r1_targets()
    for cap in caps["train"]:
        t, i = b2.row_index(cap, "C", TM)
        a = nb.target_matrix(cap, "R1L1", 0.9)[t, i]
        u = b2.target_matrix(cap, "U", 0.9)[t, i]
        assert torch.equal(a, u)


def test_install_keeps_every_b2_target_and_puts_r1_on_c_rows(nb, b2, caps):
    nb.install_r1_targets()
    assert b2.target_matrix is nb.target_matrix
    assert b2.ROWSET_OF["R1"] == "C" and b2.ROWSET_OF["R1L1"] == "C"
    for arm in ("U", "C", "U+", "C+", "U1"):
        assert _eq_nan(
            b2.target_matrix(caps["train"][0], arm, 0.9),
            nb._B2_TARGET(caps["train"][0], arm, 0.9),
        )


def test_a_joint_fit_reproduces_b2_s_own_c_fit(nb, b2, caps):
    nb.install_r1_targets()
    alone = b2.fit_heads(caps["train"], caps["val"], "C", "bilinear", [("C", 0.9)], m=TM)
    joint = b2.fit_heads(
        caps["train"],
        caps["val"],
        "C",
        "bilinear",
        [("C", 0.9), ("R1", 0.9), ("R1L1", 0.9)],
        m=TM,
    )
    # the same Gram and rule; X^T Y of a wider Y may differ in the last bits
    assert torch.allclose(
        alone[("C", 0.9)]["w"], joint[("C", 0.9)]["w"], rtol=1e-9, atol=1e-12
    )
    assert alone[("C", 0.9)]["lam"] == joint[("C", 0.9)]["lam"]
    assert joint[("R1", 0.9)]["rowset"] == "C"


# --------------------------------------------------------------------------- #
# §4: the grace wrapper
# --------------------------------------------------------------------------- #


def test_grace_g0_is_the_inner_policy(nb, b2, tiny):
    doc = tiny["val"][0]
    gen = torch.Generator().manual_seed(3)
    for _trial in range(20):
        w = torch.randn(TD * TD + 2 * TD, generator=gen, dtype=torch.float64)
        g = torch.randn(TM, TD, generator=gen)
        c = torch.randn(TD, generator=gen)
        st = _state(g, [5, 6, 7, 8], 9)
        inner = _probe(b2, doc, w)
        want = inner.select_eviction(st, c, 9)
        got = nb.GraceArgmin(_probe(b2, doc, w), 0, arm="g0").select_eviction(st, c, 9)
        assert got == want


def test_grace_protects_the_g_newest(nb, b2, tiny):
    doc = tiny["val"][0]
    g = torch.eye(TM, TD)
    # psi_k = weights[k] through s_k = e_k: newest (slot 3, age 1) lowest, then slot 2
    w = _bilinear_w_scoring_first_coord(TD, [0.4, 0.3, 0.2, 0.1] + [0.0] * (TD - 4))
    st = _state(g, [5, 6, 7, 8], 9)  # ages 4, 3, 2, 1
    c = torch.zeros(TD)
    assert nb.GraceArgmin(_probe(b2, doc, w), 0, arm="x").select_eviction(st, c, 9) == 3
    assert nb.GraceArgmin(_probe(b2, doc, w), 1, arm="x").select_eviction(st, c, 9) == 2
    assert nb.GraceArgmin(_probe(b2, doc, w), 2, arm="x").select_eviction(st, c, 9) == 1
    pol = nb.GraceArgmin(_probe(b2, doc, w), 3, arm="x")
    assert pol.select_eviction(st, c, 9) == 0
    rec = pol.log[-1]
    assert rec["flipped"] is True and rec["unmasked_slot"] == 3 and rec["age"] == 4


def test_grace_falls_back_when_every_slot_is_protected(nb, b2, tiny):
    doc = tiny["val"][0]
    g = torch.eye(TM, TD)
    w = _bilinear_w_scoring_first_coord(TD, [0.4, 0.1, 0.2, 0.3] + [0.0] * (TD - 4))
    st = _state(g, [5, 6, 7, 8], 9)  # ages 4..1, all <= 4
    pol = nb.GraceArgmin(_probe(b2, doc, w), 4, arm="x")
    assert pol.select_eviction(st, torch.zeros(TD), 9) == 1
    assert pol.log[-1]["flipped"] is False


def test_grace_logs_the_inner_score_unchanged_and_age_never_enters_it(nb, b2, tiny):
    doc = tiny["val"][0]
    gen = torch.Generator().manual_seed(5)
    w = torch.randn(TD * TD + 2 * TD, generator=gen, dtype=torch.float64)
    g = torch.randn(TM, TD, generator=gen)
    c = torch.randn(TD, generator=gen)
    a, b = _state(g, [5, 6, 7, 8], 9), _state(g, [1, 3, 7, 8], 9)
    pa = nb.GraceArgmin(_probe(b2, doc, w), 2, arm="x")
    pb = nb.GraceArgmin(_probe(b2, doc, w), 2, arm="x")
    pa.select_eviction(a, c, 9)
    pb.select_eviction(b, c, 9)
    inner = _probe(b2, doc, w).scores(a, c, 9).tolist()
    assert pa.log[-1]["psi"] == inner == pb.log[-1]["psi"]


def test_grace_builds_no_graph_and_wraps_only_a_bilinear_probe(nb, b2, tiny):
    doc = tiny["val"][0]
    g = torch.randn(TM, TD, requires_grad=True)
    c = torch.randn(TD, requires_grad=True)
    w = torch.randn(TD * TD + 2 * TD, dtype=torch.float64)
    pol = nb.GraceArgmin(_probe(b2, doc, w), 1, arm="x")
    k = pol.select_eviction(_state(g, [5, 6, 7, 8], 9), c, 9)
    assert isinstance(k, int) and g.grad is None and c.grad is None
    age = b2.ProbeArgminPolicy(
        torch.zeros(TS - 1), "age", doc=doc, model_seed=0, arm="a", S=TS
    )
    with pytest.raises(TypeError):
        nb.GraceArgmin(age, 1, arm="x")
    with pytest.raises(ValueError):
        nb.GraceArgmin(_probe(b2, doc, w), -1, arm="x")


# --------------------------------------------------------------------------- #
# §3 / §6.3: the logged content labels, end to end on the tiny model
# --------------------------------------------------------------------------- #


def _fake_fits(b2, caps, tiny):
    """B2-shaped fits on the tiny data, made by B2's own code."""
    heads = {}
    for rowset, specs in (
        ("U", [("U", 0.9), ("U+", 0.9)]),
        ("C", [("C", 0.9), ("C+", 0.9)]),
    ):
        for (a, g), f in b2.fit_heads(
            caps["train"], caps["val"], rowset, "bilinear", specs, m=TM
        ).items():
            heads[f"{a}@{g}"] = f
    for rowset in ("U", "C"):
        f = b2.fit_heads(caps["train"], caps["val"], rowset, "age", [(rowset, 0.9)], m=TM)
        heads[f"age_{rowset}@0.9"] = f[(rowset, 0.9)]
    means = {0.9: b2.class_means(caps["train"], 0.9, m=TM)}
    kw = {"vmap": tiny["vmap"], "S": TS, "m": TM, "L": TL}
    acc = {}
    for a, key in (("fifo", None), ("psiC", "C@0.9"), ("psiU", "U@0.9")):
        n = c = 0
        for d in tiny["val"]:
            pol = (
                b2.fifo_policy(d)
                if key is None
                else b2.ProbeArgminPolicy(
                    heads[key]["w"], "bilinear", doc=d, model_seed=0, arm=a, S=TS
                )
            )
            r = b2.run_arm(tiny["model"], d, pol, **kw)
            nn, cc = b2.counts(r["answers"], "all")
            n, c = n + nn, c + cc
        acc[a] = float(c) / float(n)
    return {
        "heads": heads,
        "class_means": means,
        "decisions": {
            "acc": {0.9: acc},
            "ref": {"U@0.9": "fifo", "C@0.9": "fifo"},
            "delta": {0.9: 0.03},
        },
    }


@pytest.fixture(scope="module")
def fitted(nb, b2, caps, tiny):
    b2fits = _fake_fits(b2, caps, tiny)
    res = nb.fit_core(
        tiny["model"],
        tiny["train"],
        tiny["val"],
        b2fits,
        seed=0,
        vmap=tiny["vmap"],
        S=TS,
        m=TM,
        L=TL,
    )
    return {"b2": b2fits, "nb": res}


def test_fit_core_passes_every_control_on_b2_s_own_fits(fitted):
    c = fitted["nb"]["controls"]
    assert c["C5"]["ok"] and c["C5"]["same_lambda"] and c["C5"]["rate"] == 1.0
    assert c["C5"]["w_max_abs_diff"] < 1e-9
    assert all(v["equal"] for v in c["C6"].values())
    assert c["C8"] is True and all(c["C9_grace_off"].values())
    assert set(fitted["nb"]["heads"]) == {
        "R1@0.9",
        "R1L1@0.9",
        "age_R1@0.9",
        "age_R1L1@0.9",
    }
    assert fitted["nb"]["heads"]["R1@0.9"]["rowset"] == "C"
    assert set(fitted["nb"]["refs"]) == {"R1", "R1L1"}


def test_fit_core_refuses_when_the_c_refit_is_not_b2_s(nb, b2, caps, tiny, fitted):
    bad = {**fitted["b2"], "heads": dict(fitted["b2"]["heads"])}
    h = dict(bad["heads"]["C@0.9"])
    h["val_mse_demeaned"] = [x * 1.01 for x in h["val_mse_demeaned"]]
    bad["heads"]["C@0.9"] = h
    with pytest.raises(nb.ControlFailure, match="C5"):
        nb.fit_core(
            tiny["model"],
            tiny["train"],
            tiny["val"],
            bad,
            seed=0,
            vmap=tiny["vmap"],
            S=TS,
            m=TM,
            L=TL,
        )


def test_fit_core_refuses_when_the_tree_does_not_reproduce_b2_s_fit_val(nb, tiny, fitted):
    bad = dict(fitted["b2"])
    acc = dict(bad["decisions"]["acc"][0.9])
    acc["fifo"] += 1e-9
    bad["decisions"] = {**bad["decisions"], "acc": {0.9: acc}}
    with pytest.raises(nb.ControlFailure, match="C6"):
        nb.fit_core(
            tiny["model"],
            tiny["train"],
            tiny["val"],
            bad,
            seed=0,
            vmap=tiny["vmap"],
            S=TS,
            m=TM,
            L=TL,
        )


def test_a_unit_runs_every_arm_and_logs_content_and_victim_position(nb, an, tiny, fitted):
    mk = nb.arm_makers(fitted["b2"], fitted["nb"], 0, TS)
    d = tiny["val"][0]
    u = nb.run_unit(tiny["model"], d, mk, seed=0, vmap=tiny["vmap"], S=TS, m=TM, L=TL)
    assert set(u["logs"]) == set(an.ARMS) and u["residency_ok"]
    for a in an.ARMS:
        assert len(u["logs"][a]) == TS - TM
        for rec in u["logs"][a]:
            assert len(rec["live_content"]) == len(rec["live_ages"]) == TM
            assert rec["live_ages"][rec["victim_pos"]] == rec["age"]
            t = rec["step"]
            ws = [t - x for x in rec["live_ages"]]
            want = [nb.content_of(d, w, t) for w in ws]
            assert rec["live_content"] == want
    assert all(r["victim_pos"] == 0 for r in u["logs"]["fifo"])
    assert all("flipped" in r for r in u["logs"]["graceU1"])
    assert all(r["age"] > 1 for r in u["logs"]["graceU1"])
    assert all(r["age"] > 2 for r in u["logs"]["graceU2"])


def test_units_are_atomic_resumable_and_keyed(nb, tiny, fitted, tmp_path):
    mk = nb.arm_makers(fitted["b2"], fitted["nb"], 0, TS, names=("fifo", "graceU1"))
    docs = tiny["val"][:2]
    key = {"k": 1}
    kw = {"seed": 0, "out": tmp_path, "vmap": tiny["vmap"], "S": TS, "m": TM, "L": TL}
    r1 = nb.eval_shard(tiny["model"], docs, mk, key=key, **kw)
    assert r1 == {"n_run": 2, "n_resumed": 0}
    assert not list(tmp_path.rglob("*.tmp*"))
    r2 = nb.eval_shard(tiny["model"], docs, mk, key=key, **kw)
    assert r2 == {"n_run": 0, "n_resumed": 2}
    with pytest.raises(nb.ControlFailure, match="written under"):
        nb.eval_shard(tiny["model"], docs, mk, key={"k": 2}, **kw)


# --------------------------------------------------------------------------- #
# analysis: labels, the decision table, partial rho, the bootstrap, aggregation
# --------------------------------------------------------------------------- #


def test_contrast_labels(an):
    d = 0.03
    assert an.contrast_label(0.05, 0.01, 0.08, d) == "BETTER"
    assert an.contrast_label(0.02, 0.01, 0.025, d) == "SAME"
    assert an.contrast_label(-0.05, -0.08, -0.04, d) == "WORSE"
    assert an.contrast_label(0.02, 0.005, 0.04, d) == "UNRESOLVED"
    assert an.contrast_label(0.05, 0.06, 0.08, d) == "UNRESOLVED"  # A1.13
    assert an.contrast_label(0.05, 0.01, 0.08, None) == "UNRESOLVED"


def test_dq1(an):
    assert an.dq1({1: ["BETTER", "BETTER", "SAME"], 2: ["SAME"] * 3}) == "YES"
    assert an.dq1({1: ["BETTER", "BETTER", "WORSE"], 2: ["SAME"] * 3}) == "MIXED"
    assert an.dq1({1: ["SAME", "WORSE", "UNRESOLVED"], 2: ["WORSE"] * 3}) == "NO"
    assert an.dq1({1: ["BETTER", "SAME", "SAME"]}) == "MIXED"


def test_dq2(an):
    assert an.dq2_seed("SAME", "SAME", 0, 0) == "INDISTINGUISHABLE"
    assert an.dq2_seed("SAME", "WORSE", 0, 0) == "NEAR-C"
    assert an.dq2_seed("WORSE", "SAME", 0, 0) == "NEAR-U"
    assert an.dq2_seed("WORSE", "BETTER", 0.01, 0.01) == "BETWEEN"
    assert an.dq2_seed("WORSE", "BETTER", 0.01, -0.01) == "UNRESOLVED"
    assert an.dq2(["EQUIV", "LOSS", "EQUIV"], ["NEAR-C"] * 3) == "HARMFUL (near b)"
    assert an.dq2(["EQUIV"] * 3, ["NEAR-U", "NEAR-U", "NEAR-C"]) == "HARMFUL (near b)"
    assert (
        an.dq2(["EQUIV"] * 3, ["NEAR-C", "INDISTINGUISHABLE", "NEAR-C"])
        == "SAFE (near a)"
    )
    assert an.dq2(["EQUIV"] * 3, ["NEAR-C", "BETWEEN", "NEAR-C"]) == "INTERMEDIATE"


def test_dq3(an):
    p = lambda lab, c, r, r1: {"label": lab, "acc_C": c, "acc_R1": r, "acc_R1L1": r1}  # noqa: E731
    assert (
        an.dq3(["EQUIV"] * 3, [p("WORSE", 1, 0.9, 0.8)] * 3)["class"] == "NOT ASSESSABLE"
    )
    r = an.dq3(
        ["LOSS", "LOSS", "EQUIV"],
        [p("WORSE", 0.8, 0.75, 0.7), p("WORSE", 0.8, 0.79, 0.7), p("SAME", 1, 1, 1)],
    )
    assert r["class"] == "YES" and r["harm_seeds"] == [0, 1]
    assert r["per_seed"][0]["label"] == "TRAVELS"
    assert r["per_seed"][0]["contrast_label"] == "WORSE"
    r = an.dq3(
        ["LOSS", "LOSS", "EQUIV"],
        [p("WORSE", 0.8, 0.85, 0.7), p("SAME", 0.8, 0.8, 0.8), p("SAME", 1, 1, 1)],
    )
    assert r["class"] == "MIXED"  # seed 0 not monotone; seed 1 STAYS -> 1 of 2 each
    r = an.dq3(
        ["LOSS", "EQUIV", "EQUIV"], [p("SAME", 0.8, 0.8, 0.8)] + [p("SAME", 1, 1, 1)] * 2
    )
    assert r["class"] == "NO"


def test_partial_rho_removes_an_age_signal_carried_by_content(an):
    gen = torch.Generator().manual_seed(0)
    G, n = 400, 8
    groups = torch.arange(G).repeat_interleave(n)
    age = torch.arange(1, n + 1, dtype=torch.float64).repeat(G)
    # content is a function of age (old slots are 'filler' = 4), and the victim
    # is a random 'filler' slot: age reaches the decision only through content
    c = torch.where(age > n / 2, 4, 0)
    y = torch.zeros(G * n)
    for g in range(G):
        k = int(torch.randint(n // 2, n, (1,), generator=gen))
        y[g * n + k] = 1
    r = an.partial_rho(y, age, c, groups)
    assert r["raw"] > 0.2 and abs(r["partial"]) < 0.05
    # a rule that reads age directly keeps its partial correlation
    y2 = (age == n).to(torch.float64)
    r2 = an.partial_rho(y2, age, c, groups)
    assert r2["partial"] > 0.2


def test_boot_reps_are_b2_s_replicates(an, b2):
    gen = torch.Generator().manual_seed(1)
    per = {}
    for a in ("x", "y", "random0", "random1"):
        n = torch.randint(5, 10, (50, 1), generator=gen)
        c = torch.minimum(n, torch.randint(0, 10, (50, 1), generator=gen))
        per[a] = torch.cat([n, c], 1)
    got = an.boot_reps(per, 2, n_boot=300)
    want = b2.paired_bootstrap(
        per, {"d": ("x", "y"), "r": ("x", "random")}, 2, n_boot=300
    )
    d = got["rep"]["x"] - got["rep"]["y"]
    assert float(torch.quantile(d, 0.025)) == want["d"]["lo"]
    assert float(torch.quantile(d, 0.975)) == want["d"]["hi"]
    rr = got["rep"]["x"] - got["rep"]["random"]
    assert float(torch.quantile(rr, 0.975)) == want["r"]["hi"]


def test_ratio_ci_is_undefined_when_the_denominator_straddles_zero(an):
    rep = {
        "a": torch.tensor([0.5, 0.6]),
        "b": torch.tensor([0.4, 0.4]),
        "c": torch.tensor([0.45, 0.35]),
    }
    reps = {"rep": rep, "point": {"a": 0.55, "b": 0.4, "c": 0.4}}
    assert an.ratio_ci(reps, ("a", "b"), ("c", "b"))["defined"] is False


def test_aggregate_counts_newcomer_pending_and_flips(an, tiny):
    d = tiny["val"][0]
    q_of = dict(d.pairs)
    a_pend = next(a for a, q in d.pairs if q - a >= 2)
    rec = lambda t, w, flipped=None: {  # noqa: E731
        "step": t,
        "written_at": w,
        "age": t - w,
        "rank": 1,
        "rank_shift": [1, 0],
        "kind": d.sentences[w].kind,
        "status": ("pending" if q_of.get(w, -1) > t else "answered")
        if d.sentences[w].kind == "assert"
        else None,
        "margin": None,
        "psi": None,
        "live_ages": [t - w, 99],
        "live_content": [0, 4],
        "victim_pos": 0,
        **({} if flipped is None else {"flipped": flipped}),
    }
    logs = [rec(a_pend + 1, a_pend, True), rec(a_pend + 1, a_pend, False)]
    u = {
        "doc_id": d.doc_id,
        "counts": {"x": {b: [2, 1] for b in an.BUCKETS}},
        "hits": {"x": [(3, True), (20, False)]},
        "logs": {"x": logs},
    }
    out = an.aggregate_seed([u], [d], ["x"], S=TS, m=TM)
    r = out["readouts"]["x"]
    assert r["age1"] == 2 and r["age1_pending"] == 2 and r["pending_victims"] == 2
    le = q_of[a_pend] - a_pend <= TM
    assert r["pending_le_M"] == (2 if le else 0)
    assert r["grace_flips"] == 1 and r["grace_flip_share"] == 0.5
    assert r["displacement_hist"] == {0: 2}
    assert out["residency"]["x"]["all"].tolist() == [[2, 1]]


def test_content_labels(an):
    assert an.content_label("assert", "pending") == "assert:pending"
    assert an.content_label("query", None) == "query"
    with pytest.raises(ValueError):
        an.content_label("assert", None)
    assert len(an.CONTENT_LABELS) == 5


def test_seed_outcomes_reads_b2_s_rule_against_each_arm_s_ref(an):
    gen = torch.Generator().manual_seed(4)
    counts = {}
    for a in an.ARMS:
        n = torch.full((60, 1), 10)
        c = torch.randint(6, 10, (60, 1), generator=gen)
        counts[a] = {b: torch.cat([n, c], 1) for b in an.BUCKETS}
    refs = {"U": "age", "C": "fifo", "R1": "fifo", "R1L1": "age"}
    out = an.seed_outcomes(counts, refs, 0.03, 0)
    assert out["outcome"]["psiU"]["ref"] == "ageU"
    assert out["outcome"]["graceU2"]["ref"] == "ageU"
    assert out["outcome"]["psiC+"]["ref"] == "fifo"
    assert out["outcome"]["psiR1L1"]["ref"] == "ageR1L1"
    for v in out["outcome"].values():
        assert v["label"] in ("WIN", "LOSS", "EQUIV", "UNRESOLVED")
    dec = an.decisions({s: out for s in (0, 1, 2)})
    assert dec["DQ1"]["class"] in ("YES", "NO", "MIXED")
    assert dec["replication"]["label"].startswith("replication")
    assert math.isfinite(out["acc"]["all"]["psiC"]["point"])
