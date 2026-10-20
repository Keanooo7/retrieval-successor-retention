"""PREREG post-fails-controls §3, item 1a (``experiments/post-fails-controls/item_1a.py``)
and the phase-1 dispatcher (``run.py``).

A naming-only split of PREREG §0's ``tests/test_post_fails_controls.py``.

🔴 **Nothing here reads the real substrate, run 2's records or any EVAL document.**
The reader is a tiny random TG (the real block types, D = 32, N = 2) built in-process;
the documents are U's ids 64 and 65 at seed 0 (already inspected by B0, CF,
offline-psi) and the vocabulary documents [0, 64). The four passes run through the
real ``R1.run_heldout`` / ``run_batch`` path the run uses.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments" / "post-fails-controls"
U_IDS = (64, 66)
KM = {"assert": 0.5, "query": 0.4, "filler": 0.3}  # test-only class means


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def RUN():
    return _load("pfc_run_dispatcher", HERE / "run.py")


@pytest.fixture(scope="module")
def M(RUN):
    return RUN.load_item("1a")


@pytest.fixture(scope="module")
def H(M):
    return M.harness()


@pytest.fixture(scope="module")
def tiny(M, H, tmp_path_factory):
    """Four passes over U ids 64, 65 through ``R1.run_heldout`` with a tiny reader."""
    from rsr.model.tg import TGConfig, TGModel

    R1 = H["R1"]
    sub = R1.Substrate(checkpoint="-", sha256="-", manifest="-", root="-")
    cfg = R1.Run1Config(seed=0, substrate=sub, iters=0, eval_range=U_IDS)
    vmap = R1.vocab_map(cfg)
    tg = TGConfig(
        D=32,
        H=2,
        N=2,
        V=max(vmap.values()) + 1,
        max_sentence_tokens=16,
        max_sentences_in_short_term=cfg.memory_slots,
        block_config=("S", "C"),
        srep_extraction_layer=1,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        eod_id=3,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = TGModel(tg)
    tgd = dataclasses.asdict(tg)
    tgd["block_config"] = list(tgd["block_config"])
    payload = {
        "meta": {"config": {"tg": tgd, "policy": "fifo"}},
        "model": model.state_dict(),
    }
    M.set_kind_means(0, KM)
    out = tmp_path_factory.mktemp("pfc1a")
    passes = [(a, "fifo", False) for a in M.PASS_ORDER]
    ho = R1.run_heldout(cfg, {"fifo": payload}, out, passes=passes)
    ho2 = R1.run_heldout(cfg, {"fifo": payload}, out / "again", passes=passes[:1])
    recs = {a: R1.read_pass(out, a) for a in M.PASS_ORDER}
    return {
        "cfg": cfg,
        "ho": ho,
        "recs": recs,
        "again": R1.read_pass(out / "again", "fifo"),
        "docs": R1.docs_by_id(cfg, range(*U_IDS)),
        "ho2": ho2,
    }


# --------------------------------------------------------------------------- #
# the dispatcher
# --------------------------------------------------------------------------- #


def test_dispatcher_absent_module_exits_3(RUN, tmp_path, capsys):
    assert int(RUN.main(["1a", "measure", "--seed", "0"], here=tmp_path)) == 3
    assert "absent" in capsys.readouterr().err
    assert int(RUN.main(["finalize", "1b"], here=tmp_path)) == 3
    assert int(RUN.main(["finalize"])) == 3
    assert int(RUN.main(["1z"])) == 3
    assert int(RUN.main([])) == 3
    assert RUN.load_item("1z") is None


def test_dispatcher_loads_item_1a_with_the_shared_common(RUN, M):
    assert M.C is sys.modules["pfc_common"]
    assert callable(M.main) and callable(M.finalize)


# --------------------------------------------------------------------------- #
# the passes through the real held-out path
# --------------------------------------------------------------------------- #


def test_every_pass_ran_and_residency_holds(M, tiny):
    for a in M.PASS_ORDER:
        assert len(tiny["recs"][a]) == 2
        assert M.residency(tiny["ho"], a, 2)["ok"], a


def test_factfiller_in_loop_equals_model_free(M, H, tiny):
    R1, R2 = H["R1"], H["R2"]
    c = M.factfiller_check(R1, R2, tiny["recs"]["factfiller"], tiny["docs"], 0, 16)
    assert c["ok"], c
    assert c["pooled_hit_in_loop"] == c["pooled_hit_model_free"]
    assert all(r["victims"] for r in tiny["recs"]["factfiller"])  # memory filled


def test_kindoracle_is_b0s_ko_rule_in_the_loop(M, H, tiny):
    """The in-loop victims equal B0's ``ScorePolicy`` (ko, random ties keyed by the
    step) replayed model-free on the same document."""
    from rsr.metrics.headroom import simulate

    R1, B0 = H["R1"], H["B0"]
    mu = {"kind": dict(KM)}
    for d, r in zip(tiny["docs"], tiny["recs"]["kindoracle"], strict=True):
        rec = R1.Recorder(M.kind_oracle(B0, d, mu, 0))
        simulate(d, rec, 16)
        assert r["victims"]
        assert {str(k): v for k, v in rec.victims.get(0, {}).items()} == r["victims"]
    # and it differs from FIFO's victims (the rule is not FIFO in disguise)
    assert [r["victims"] for r in tiny["recs"]["kindoracle"]] != [
        r["victims"] for r in tiny["recs"]["fifo"]
    ]


def test_kindoracle_score_has_no_age(M, H, tiny):
    B0 = H["B0"]
    d = tiny["docs"][0]
    ko = B0.scorers(d, {"kind": dict(KM)})["ko"]
    for i in range(10):
        assert len({ko(i, t) for t in range(i + 1, 48)}) == 1


def test_records_equal_is_record_for_record(M, tiny):
    a, b = tiny["recs"]["fifo"], tiny["again"]
    assert M.records_equal(a, b)["ok"]  # the pass is deterministic
    bad = json.loads(json.dumps(b))
    bad[1]["answers"][0][3] += 1e-12
    r = M.records_equal(a, bad)
    assert not r["ok"] and r["first_mismatches"][0]["fields"] == ["answers"]
    assert not M.records_equal(a[:1], b)["ok"]  # a prefix is not equal
    assert M.records_equal(a[:1], b, prefix=True)["ok"]  # smoke only, flagged
    assert not M.records_equal([], [])["ok"]  # nothing compared is not a pass


def test_pairing(M, H, tiny):
    R2 = H["R2"]
    for a in M.PASS_ORDER:
        assert M.pairing(R2, tiny["recs"][a], tiny["recs"]["fifo"])["ok"]
    assert not M.pairing(R2, tiny["recs"]["fifo"][:1], tiny["recs"]["fifo"])["ok"]


# --------------------------------------------------------------------------- #
# statistics, gated by A-C2; the reading rule
# --------------------------------------------------------------------------- #


def test_no_statistic_before_a_c2(M, H, tiny):
    recs = {a: tiny["recs"][a] for a in ("fifo", *M.NEW_ARMS)}
    for gate in (None, {}, {"ok": False}):
        with pytest.raises(M.C.ControlFailure):
            M.seed_stats(H, recs, 0, 0.01, a_c2=gate, n_boot=20, m=16)
    st = M.seed_stats(H, recs, 0, 0.01, a_c2={"ok": True}, n_boot=20, m=16)
    assert set(st["headline"]) == set(M.NEW_ARMS)
    for a in M.NEW_ARMS:
        assert st["headline"][a]["label"] in ("WIN", "LOSS", "EQUIV", "UNRESOLVED")
        assert st["contrast"]["all"][a]["point"] == pytest.approx(
            st["accuracy"]["all"][a]["point"] - st["accuracy"]["all"]["fifo"]["point"]
        )
    assert set(st["age1_share"]) == {"fifo", *M.NEW_ARMS}
    bad = json.loads(json.dumps(st))
    bad["headline"]["factfiller"]["lo"] = float("nan")
    assert M.stats_defects(bad)
    swapped = {**recs, "factfiller": list(reversed(recs["factfiller"]))}
    with pytest.raises(M.C.ControlFailure):
        M.seed_stats(H, swapped, 0, 0.01, a_c2={"ok": True}, n_boot=20, m=16)


def test_labels_are_run2s_seed_label(H):
    R2 = H["R2"]
    assert R2.seed_label(0.05, 0.02, 0.08, 0.01) == "WIN"
    assert R2.seed_label(-0.05, -0.08, -0.02, 0.01) == "LOSS"
    assert R2.seed_label(0.0, -0.005, 0.005, 0.01) == "EQUIV"
    assert R2.seed_label(None, 0, 0, 0.01) == "UNRESOLVED"


@pytest.mark.parametrize(
    ("labels", "pattern"),
    [
        ({0: "WIN", 1: "WIN", 2: "WIN"}, "WIN_ALL_SEEDS"),
        ({0: "WIN", 1: "LOSS", 2: "WIN"}, "LOSS_ANY_SEED"),
        ({0: "WIN", 1: "EQUIV", 2: "WIN"}, "INDETERMINATE"),
        ({0: "WIN", 1: "UNRESOLVED", 2: "WIN"}, "INDETERMINATE"),
        ({0: "WIN", 2: "WIN"}, "INDETERMINATE"),
        ({0: "LOSS", 1: None, 2: "UNRESOLVED"}, "LOSS_ANY_SEED"),
    ],
)
def test_reading_rule(M, labels, pattern):
    assert M.reading(labels, (0, 1, 2))["pattern"] == pattern


def _doc(seed, exit_code, label, head="h"):
    return {
        "seed": seed,
        "exit": exit_code,
        "git_head": head,
        "smoke": False,
        "seeds_actually_run": [seed],
        "documents_done": 2,
        "stats": {
            "headline": {a: {"label": label} for a in ("factfiller", "kindoracle")}
        },
    }


def test_summarize_not_read_unless_every_control_passed(M, tmp_path):
    (tmp_path / "logs").mkdir()
    for s in (0, 1, 2):
        (tmp_path / "logs" / f"1a-s{s}.rc").write_text("0\n")
    launch = {"exit": 0, "head": "h"}
    docs = {s: _doc(s, 0, "WIN") for s in (0, 1, 2)}
    sm = M.summarize(docs, launch, (0, 1, 2), tmp_path)
    assert sm["exit"] == 0 and sm["readings"]["factfiller"]["pattern"] == "WIN_ALL_SEEDS"
    docs[1] = _doc(1, 3, "WIN")
    (tmp_path / "logs" / "1a-s1.rc").write_text("3\n")
    sm = M.summarize(docs, launch, (0, 1, 2), tmp_path)
    assert sm["exit"] == 3 and sm["readings"]["factfiller"]["pattern"] == "NOT_READ"
    assert sm["labels"]["factfiller"]["1"] == "WIN"  # written, not read
    docs = {s: _doc(s, 0, "WIN") for s in (0, 1)}  # a seed missing
    assert M.summarize(docs, launch, (0, 1, 2), tmp_path)["exit"] == 3
    docs = {s: _doc(s, 0, "WIN", head=str(s)) for s in (0, 1, 2)}  # heads differ
    assert M.summarize(docs, launch, (0, 1, 2), tmp_path)["exit"] == 3
    docs = {s: _doc(s, 0, "WIN") for s in (0, 1, 2)}
    assert M.summarize(docs, None, (0, 1, 2), tmp_path)["exit"] == 3  # no launch checks
    (tmp_path / "logs" / "1a-s2.rc").write_text("1\n")  # the process said otherwise
    assert M.summarize(docs, launch, (0, 1, 2), tmp_path)["exit"] == 3


def test_kind_means_read_from_the_b0_ledger(M, tmp_path):
    rows = [
        {"key": f"K.{k}", "samples": [0.1 + i, 0.2 + i, 0.3 + i]}
        for i, k in enumerate(M.KINDS)
    ]
    p = tmp_path / "ledger.json"
    p.write_text(json.dumps({"rows": rows}))
    refs = M.Refs(b0_ledger=str(p), b0_key="K.{k}")
    r = M.read_kind_means(1, refs)
    assert r["ok"] and r["m_kind"] == {"assert": 0.2, "query": 1.2, "filler": 2.2}
    assert r["order_ascending"] == ["assert", "query", "filler"]
    assert M._KIND_MEANS[1] == r["m_kind"]
    M.set_kind_means(0, KM)  # restore the fixture's
