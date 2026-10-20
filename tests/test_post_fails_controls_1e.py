"""Item 1e (post-fails-controls PREREG §7): the registered groups, the group knockout's
pass-through, and the statistics, on hand-built inputs; the seal on 1e's paths."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


E = _load("_pfc_item_1e", "experiments/post-fails-controls/item_1e.py")


@dataclass
class Sent:
    kind: str
    text: str


@dataclass
class Doc:
    doc_id: int
    sentences: tuple
    pairs: tuple


def toy(doc_id=9):
    sents = [Sent("filler", f"f{i % 3}") for i in range(48)]
    pairs = [(20, 25), (10, 40)]
    for a, q in pairs:
        sents[a] = Sent("assert", f"A{a}")
        sents[q] = Sent("query", f"Q{q}")
    return Doc(doc_id, tuple(sents), tuple(pairs))


def test_groups_registered_types_and_counts():
    d = toy()
    t = 30  # resident 14..29: assert 20 and its query 25; fillers f0-f2
    gs = E.groups_for(d, t, seed=0)
    by = {}
    for typ, mem in gs:
        by.setdefault(typ, []).append(mem)
    ranks = {i: i - (t - 16) for i in range(t - 16, t)}
    assert by["G2_fact_pairs"] == [tuple(sorted((ranks[20], ranks[25])))]
    # 14 resident fillers over 3 texts: sizes 5/5/4 -> 10 + 10 + 6 duplicate pairs
    fill = [i for i in range(t - 16, t) if d.sentences[i].kind == "filler"]
    cnt = {}
    for i in fill:
        cnt[d.sentences[i].text] = cnt.get(d.sentences[i].text, 0) + 1
    assert len(by["G1_dup_fillers"]) == sum(n * (n - 1) // 2 for n in cnt.values())
    assert len(by["G3_random_pairs"]) == 8 and len(set(by["G3_random_pairs"])) == 8
    assert [len(x) for x in by["G4_random_sets"]] == [4, 4, 8]
    assert gs == E.groups_for(d, t, seed=0)  # deterministic
    assert gs != E.groups_for(d, t, seed=1)


def test_g2_needs_both_resident():
    d = toy()
    gs = E.groups_for(d, 41, seed=0)  # assert 10 evicted by t = 27; query 40 resident
    assert not [g for g in gs if g[0] == "G2_fact_pairs"]


def test_group_kv_replaces_members_and_passes_the_rest_through():
    g = torch.Generator().manual_seed(0)
    kv = torch.randn(3, 16, 4, generator=g)
    members = [[1, 5], [], [0, 2, 15]]
    reps = [
        [torch.full((4,), 7.0), torch.full((4,), 8.0)],
        [],
        [torch.full((4,), 1.0), torch.full((4,), 2.0), torch.full((4,), 3.0)],
    ]
    out, ok = E.group_kv(kv, members, reps)
    assert ok
    assert torch.equal(out[0, 1], torch.full((4,), 7.0))
    assert torch.equal(out[0, 5], torch.full((4,), 8.0))
    assert torch.equal(out[1], kv[1])
    assert torch.equal(out[2, 15], torch.full((4,), 3.0))
    assert torch.equal(out[0, 0], kv[0, 0])


def test_statistics_on_hand_records():
    d = toy()
    docs = [d, toy(10)]
    recs = [
        {
            "doc_pos": 0,
            "t": 25,
            "type": "G3_random_pairs",
            "dG": 1.0,
            "sum_single": 0.5,
            "singles": [0.25, 0.25],
            "skipped": False,
        },  # t = 25 is a query: Q
        {
            "doc_pos": 1,
            "t": 30,
            "type": "G3_random_pairs",
            "dG": -2.0,
            "sum_single": -1.0,
            "singles": [-0.5, -0.5],
            "skipped": False,
        },
        {"doc_pos": 1, "t": 31, "type": "G3_random_pairs", "skipped": True},
    ]
    X = E.per_doc_sums(recs, docs, "G3_random_pairs", "Q")
    assert X[0].tolist() == [1.0, 1.0, 0.5, 0.5, 0.5, 1.0] and X[1].sum() == 0
    X = E.per_doc_sums(recs, docs, "G3_random_pairs", "other")
    assert X[1].tolist() == [1.0, -2.0, -1.0, -1.0, 1.0, 2.0]
    W = torch.ones(5, 2, dtype=torch.float64)
    st = E.stats_of(X, W)
    assert st["n_groups"] == 1
    assert st["interaction_share"]["point"] == 0.5
    assert st["mean_I"]["point"] == -1.0
    s = E.summarize(recs, docs, seed=0, n_boot=10)
    assert s["G3_random_pairs"]["skipped"] == 1


def test_seal_guards_1e_document_path():
    cm = E.IC.common()
    cm.install_seal()
    for lo, _hi in cm.sealed_ranges().values():
        with pytest.raises(cm.SealedRange):
            E.LR.documents(0, e_set=(lo, lo + 2))
