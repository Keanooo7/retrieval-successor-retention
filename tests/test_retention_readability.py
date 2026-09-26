"""Retention readability's own gates (PREREG at ab7cbd2).

`experiments/retention-readability/PREREG.md`.

The harness is exercised on a tiny random-init TG (d=32, two blocks) over real S0-03
documents -- the instrument's mechanics do not depend on the model being trained. The
rule is fed synthetic per-seed readouts. Mutation-battery gates
(`scripts/mutation_battery.py`, ``# --- retention-readability ---``):

* ``test_document_id_mismatch_raises`` -- the per-document id assertion dropped;
* ``test_oracle_demand_of_another_document_raises`` -- the oracle demand check dropped;
* ``test_rank_index_is_M_minus_gap_under_fifo`` -- rank read newest-first;
* ``test_paired_bootstrap_identical_arms_have_zero_width`` -- the bootstrap unpaired;
* ``test_classify_a_table`` -- PENALTY on any seed instead of every seed;
* ``test_control1_tolerance_is_exact`` -- control 1's tolerance loosened;
* ``test_decide_a_null_failure_is_inconclusive`` -- the arm A null ignored.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rsr.baselines.oracle import OraclePolicy  # noqa: E402
from rsr.data.synthetic import ANSWER_SYMBOLS, discounted_demand  # noqa: E402
from rsr.model.tg import TGConfig, TGModel  # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location(
        "retention_readability_run",
        ROOT / "experiments" / "retention-readability" / "run.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


run = _load()
SEEDS = run.SEEDS


@pytest.fixture(scope="module")
def world():
    """Seed 0's vocabulary, a few E documents and a tiny random TG."""
    docs, vmap, V, closure = run.documents(0, e_set=(64, 70))
    torch.manual_seed(0)
    cfg = TGConfig(
        D=32,
        H=1,
        N=2,
        V=V,
        max_sentence_tokens=run.L,
        max_sentences_in_short_term=run.M,
        block_config=("S", "C"),
        pad_id=0,
        bos_id=1,
        eos_id=2,
        eod_id=3,
    )
    model = TGModel(cfg).eval()
    sym = torch.tensor([vmap[s] for s in ANSWER_SYMBOLS])
    return {"docs": docs, "vmap": vmap, "model": model, "sym": sym, "closure": closure}


def _run(world, arm, doc, seed=0):
    pol = run.DocPolicy(run.make_inner(arm, doc, seed), doc.doc_id)
    return run.run_document(world["model"], doc, world["vmap"], world["sym"], pol)


# --------------------------------------------------------------------------- #
# the instrument
# --------------------------------------------------------------------------- #


def test_documents_are_E_then_P_and_closed(world):
    ids = [d.doc_id for d in world["docs"]]
    assert ids == list(range(64, 70)) + list(range(4096, 4160))
    assert world["closure"]["ok"]


def test_fifo_harness_bit_exact_to_answer_readout(world):
    for doc in world["docs"][:3]:
        r = _run(world, "fifo", doc)
        c = run.control1_compare(
            r, run.readout_document(world["model"], doc, world["vmap"], world["sym"])
        )
        assert c["n"] == len(doc.pairs) and run.control1_ok(c), c


def test_rank_index_is_M_minus_gap_under_fifo(world):
    seen = 0
    for doc in world["docs"][:6]:
        for x in _run(world, "fifo", doc):
            if x["gap"] <= run.M:
                assert x["resident"]
                if x["n_live"] == run.M:
                    assert x["rank"] == run.M - x["gap"], x
                    seen += 1
            else:
                assert not x["resident"] and x["rank"] == -1
    assert seen > 10


def test_residency_matches_simulate_for_every_arm(world):
    for doc in world["docs"][:4]:
        for arm in run.ARMS:
            r = _run(world, arm, doc)
            assert [x["resident"] for x in r] == run.model_free_residency(arm, doc, 0)


def test_oracle_keeps_every_queried_fact(world):
    for doc in world["docs"][:6]:
        assert all(x["resident"] for x in _run(world, "oracle", doc))


def test_document_id_mismatch_raises(world):
    a, b = world["docs"][0], world["docs"][1]
    pol = run.DocPolicy(run.make_inner("fifo", a, 0), a.doc_id)
    with pytest.raises(run.DocumentMismatch):
        run.run_document(world["model"], b, world["vmap"], world["sym"], pol)


def test_oracle_demand_of_another_document_raises(world):
    a, b = world["docs"][0], world["docs"][1]
    assert discounted_demand(a, run.GAMMA) != discounted_demand(b, run.GAMMA)
    pol = run.DocPolicy(OraclePolicy(discounted_demand(a, run.GAMMA)), b.doc_id)
    with pytest.raises(run.DocumentMismatch):
        run.run_document(world["model"], b, world["vmap"], world["sym"], pol)


def test_batched_rows_are_refused(world):
    doc = world["docs"][0]
    pol = run.DocPolicy(run.make_inner("fifo", doc, 0), doc.doc_id)
    with pytest.raises(run.DocumentMismatch):
        run.check_policy(pol, doc, 2)


def test_factfiller_evicts_no_assert_while_a_filler_is_live(world):
    from rsr.retention.policy import MemoryState

    doc = world["docs"][0]
    kinds = [s.kind for s in doc.sentences]
    asserts = [i for i, k in enumerate(kinds) if k == "assert"]
    others = [i for i, k in enumerate(kinds) if k != "assert"]
    written = torch.tensor(asserts[:3] + others[:1])
    slots = MemoryState(
        gestalts=torch.zeros(4, 1),
        written_at=written,
        live=torch.ones(4, dtype=torch.bool),
        step=40,
    )
    pol = run.FactFillerPolicy(doc, random.Random(0))
    assert {pol.select_eviction(slots, None, 40) for _ in range(20)} == {3}
    a = [run.make_inner("factfiller", doc, 1).rng.random() for _ in range(2)]
    assert a[0] == a[1]  # fresh per document, seeded by (seed, doc id)


# --------------------------------------------------------------------------- #
# controls
# --------------------------------------------------------------------------- #


def test_control1_tolerance_is_exact():
    ok = {"n": 3, "ok_mismatches": 0, "max_abs_nll_diff": 0.0}
    assert run.control1_ok(ok)
    assert not run.control1_ok({**ok, "max_abs_nll_diff": 1e-9})
    assert not run.control1_ok({**ok, "ok_mismatches": 1})


def _measured(v=0.5):
    return {
        "heldout": {
            "live": {b: {r: v for r in run.LEDGER_READOUTS} for b in run.LEDGER_BUCKETS}
        }
    }


def _ref(v=0.5, drop=None):
    rows = {}
    for b in run.LEDGER_BUCKETS:
        for r in run.LEDGER_READOUTS:
            k = f"B.ckpt3000.heldout.live.{b}.{r}"
            if k != drop:
                rows[k] = {"key": k, "samples": [v, v, v]}
    return rows


def test_control2_tolerance_and_missing_keys():
    assert run.control2_compare(_measured(0.5), _ref(0.5), "B", 3000, 1)["ok"]
    assert run.control2_compare(_measured(0.5 + 5e-7), _ref(0.5), "B", 3000, 1)["ok"]
    bad = run.control2_compare(_measured(0.5 + 2e-6), _ref(0.5), "B", 3000, 1)
    assert not bad["ok"] and bad["out_of_tolerance"]
    k = "B.ckpt3000.heldout.live.all.answer_acc"
    miss = run.control2_compare(_measured(0.5), _ref(0.5, drop=k), "B", 3000, 1)
    assert not miss["ok"] and miss["missing"] == [k]


def test_control2_fails_on_nan_on_either_or_both_sides():
    """`abs(nan - nan) > tol` is False, so a NaN on both sides used to pass, and
    `max(worst, nan)` kept `max_abs_diff` at 0.0. A NaN is never a reproduction."""
    nan, k = float("nan"), "B.ckpt3000.heldout.live.all.answer_acc"
    for mine, theirs in ((nan, nan), (nan, 0.5), (0.5, nan)):
        m = _measured(0.5)
        b, rd = k.split(".")[-2:]
        m["heldout"]["live"][b][rd] = mine
        ref = _ref(0.5)
        ref[k]["samples"] = [theirs] * 3
        got = run.control2_compare(m, ref, "B", 3000, 1)
        assert not got["ok"], (mine, theirs, got)
        assert got["out_of_tolerance"] == [k], (mine, theirs, got)


def test_disjointness_catches_overlap():
    assert run.disjointness()["ok"]
    for bad in ((0, 100), (4000, 4100), (5000, 6000), (60000, 60001)):
        assert not run.disjointness(bad)["ok"], bad


def test_thresholds_are_the_preregs():
    text = (ROOT / run.PREREG).read_text()
    fm = yaml.safe_load(text.split("---")[1])
    th = fm["thresholds"]
    assert th["RP_MAX"].startswith(f"{run.RP_MAX} ")
    assert th["READ_MARGIN"].startswith(f"{run.READ_MARGIN:.2f} ")
    assert th["A_NULL"].startswith(f"{run.A_NULL} ")
    assert th["reproduction_tolerance"].startswith("1e-6") and run.REPRO_TOL == 1e-6
    assert f"{run.N_BOOT} per-document" in th["bootstrap"]
    assert f"{run.BOOT_SEED_BASE} + seed" in th["bootstrap"]
    assert fm["seeds"] == run.SEEDS
    assert fm["checkpoints"] == {"B": [2500, 3000], "A": [3000]}
    assert run.E_SET == (64, 1088) and run.P_SET == (4096, 4160)
    assert run.GAMMA == 0.97


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #


def _recs(ok_by_doc, gap=5, resident=True):
    return [
        {
            "doc": d,
            "gap": gap,
            "resident": resident,
            "rank": 0,
            "ok": ok,
            "nll": float(not ok),
            "brier16": float(not ok),
        }
        for d, oks in ok_by_doc.items()
        for ok in oks
    ]


def test_paired_bootstrap_identical_arms_have_zero_width():
    g = random.Random(0)
    by_doc = {d: [g.random() < 0.6 for _ in range(g.randint(1, 6))] for d in range(40)}
    ids = list(range(40))
    x = run.doc_sums(_recs(by_doc), ids)
    out = run.bootstrap({"fifo": x, "oracle": x.clone(), "factfiller": x.clone()}, 0, 500)
    assert out["RP"]["point"] == 0.0
    assert out["RP"]["lo"] == 0.0 and out["RP"]["hi"] == 0.0
    assert out["H_model"]["lo"] == 0.0 and out["H_model"]["hi"] == 0.0


def test_bootstrap_ci_brackets_a_real_difference():
    ids = list(range(60))
    f = run.doc_sums(_recs({d: [True, True, False] for d in ids}), ids)
    o = run.doc_sums(_recs({d: [True, False, False] for d in ids}), ids)
    out = run.bootstrap({"fifo": f, "oracle": o, "factfiller": f}, 1, 300)
    assert out["RP"]["point"] == pytest.approx(1 / 3)
    assert out["RP"]["lo"] <= 1 / 3 <= out["RP"]["hi"]


def test_buckets():
    r = {"gap": 20, "resident": True, "rank": 0}
    assert run._in_bucket(r, "rescued") and run._in_bucket(r, "rescued_rank0")
    assert not run._in_bucket({**r, "resident": False}, "rescued")
    s = {"gap": 5, "resident": True, "rank": run.M - 5}
    assert run._in_bucket(s, "gap_2_to_M") and not run._in_bucket(s, "gap_2_to_M_shifted")
    assert run._in_bucket({**s, "rank": 0}, "gap_2_to_M_shifted")
    assert run._in_bucket({"gap": run.M}, "gap_eq_M")


# --------------------------------------------------------------------------- #
# 🔒 the rule
# --------------------------------------------------------------------------- #


def _ci(p, lo, hi):
    return {"point": p, "lo": lo, "hi": hi}


def test_classify_a_table():
    no = {s: _ci(0.0, -0.01, 0.02) for s in SEEDS}
    assert run.classify_a(no) == "NO_PENALTY"
    pen = {s: _ci(0.05, 0.01, 0.09) for s in SEEDS}
    assert run.classify_a(pen) == "PENALTY"
    one = {0: _ci(0.05, 0.01, 0.09), 1: _ci(0.0, -0.02, 0.02), 2: _ci(0.0, -0.02, 0.02)}
    assert run.classify_a(one) == "MIXED_PENALTY"
    # point >= 0.03 but CI touching 0 is not PENALTY
    assert run.classify_a({s: _ci(0.04, -0.001, 0.08) for s in SEEDS}) == "MIXED_PENALTY"
    # the upper bound AT 0.03 is not NO_PENALTY
    assert run.classify_a({s: _ci(0.0, -0.01, 0.03) for s in SEEDS}) == "MIXED_PENALTY"


def test_classify_b_table():
    good = {s: _ci(-0.02, -0.08, 0.03) for s in SEEDS}
    far = {s: _ci(0.7, 0.6, 0.8) for s in SEEDS}
    near = {s: _ci(0.02, -0.03, 0.07) for s in SEEDS}
    bad_ref = {s: _ci(-0.6, -0.7, -0.5) for s in SEEDS}
    assert run.classify_b(good, far) == "READABLE"
    assert run.classify_b(bad_ref, near) == "UNREADABLE"
    assert run.classify_b(good, near) == "UNDISTINGUISHED"
    assert run.classify_b(bad_ref, far) == "MIXED_READ"
    one = dict(good)
    one[2] = _ci(-0.2, -0.3, -0.1)
    assert run.classify_b(one, far) == "MIXED_READ"


def test_combine_table():
    assert run.combine("NO_PENALTY", "READABLE") == "READABLE_AT_SHIFTED_RANK"
    for b in ("READABLE", "UNREADABLE", "UNDISTINGUISHED", "MIXED_READ"):
        assert run.combine("PENALTY", b) == "RANK_BOUND"
    for a in ("NO_PENALTY", "MIXED_PENALTY"):
        assert run.combine(a, "UNREADABLE") == "RANK_BOUND"
    assert run.combine("NO_PENALTY", "UNDISTINGUISHED") == "MIXED"
    assert run.combine("MIXED_PENALTY", "READABLE") == "MIXED"
    assert run.combine("NO_PENALTY", "MIXED_READ") == "MIXED"


def _results(null=0.0, rp=0.0):
    good = {
        "RP": _ci(rp, rp - 0.01, rp + 0.01),
        "D_ref": _ci(-0.02, -0.08, 0.03),
        "D_floor": _ci(0.7, 0.6, 0.8),
        "H_model": _ci(0.15, 0.13, 0.17),
    }
    a = {**good, "H_model": _ci(null, null - 0.01, null + 0.01)}
    return {
        ("B", 2500): {s: good for s in SEEDS},
        ("B", 3000): {s: good for s in SEEDS},
        ("A", 3000): {s: a for s in SEEDS},
    }


def test_decide_happy_path():
    d = run.decide(_results(), [])
    assert d["classification"] == "READABLE_AT_SHIFTED_RANK"
    assert d["outcome"] == "survived" and d["exit"] == 0 and not d["moving"]
    assert set(d["per_checkpoint"]) == {"B.ckpt2500", "B.ckpt3000"}


def test_decide_a_null_failure_is_inconclusive():
    d = run.decide(_results(null=0.05), [])
    assert d["classification"] == "inconclusive" and d["exit"] == 3
    assert d["outcome"] == "inconclusive"


def test_decide_control_failure_and_missing_are_inconclusive():
    d = run.decide(_results(), ["control1 failed on B.ckpt3000.seed0"])
    assert d["classification"] == "inconclusive" and d["exit"] == 3
    r = _results()
    del r[("B", 2500)][1]
    d = run.decide(r, [])
    assert d["classification"] == "inconclusive" and d["exit"] == 3


def test_decide_rank_bound_is_falsified_and_moving_is_flagged():
    r = _results()
    r[("B", 3000)] = {
        s: {**r[("B", 3000)][s], "RP": _ci(0.06, 0.03, 0.09)} for s in SEEDS
    }
    d = run.decide(r, [])
    assert d["classification"] == "RANK_BOUND" and d["outcome"] == "falsified"
    assert d["moving"] and d["exit"] == 0
