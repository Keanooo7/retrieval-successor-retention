"""Item 1b (post-fails-controls PREREG §4): the split, the cells, the probes on planted
fixtures (B-C5's test half), balanced accuracy, the reading rule, the class means and
the 5-class status rule. Hand-built inputs only."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

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


B = _load("_pfc_item_1b", "experiments/post-fails-controls/item_1b.py")


@dataclass
class Sent:
    kind: str


@dataclass
class Doc:
    doc_id: int
    sentences: tuple
    pairs: tuple


def toy_doc(doc_id=7):
    kinds = ["filler"] * 48
    pairs = [(2, 20), (18, 30), (25, 40)]
    for a, q in pairs:
        kinds[a], kinds[q] = "assert", "query"
    return Doc(doc_id, tuple(Sent(k) for k in kinds), tuple(pairs))


def test_p_split_is_j_mod_5_in_u_train_order():
    train = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
    fit, test = B.p_split(train)
    assert test == [1, 6, 12] and fit == [2, 3, 4, 5, 8, 9, 10, 11]
    assert not set(fit) & set(test)


def test_status_cells_labels_and_querying_excluded():
    d = toy_doc()
    idx, y, nq = B.status_cells([d], [0])
    cells = {
        (int(t), int(i)): int(v)
        for (_, t, i), v in zip(idx.tolist(), y.tolist(), strict=True)
    }
    # assert 2 (query 20): resident at t in [16, 19), pending; gone by its query
    assert cells[(16, 2)] == 0 and cells[(17, 2)] == 0
    assert (18, 2) in cells and (20, 2) not in cells
    # assert 18 (query 30): answered once t > 30
    assert cells[(31, 18)] == 1 and cells[(29, 18)] == 0
    assert (30, 18) not in cells
    assert nq == 2  # (30, 18) and (40, 25); assert 2 is evicted by t = 19
    assert all(t - 16 <= i < t for (_, t, i) in idx.tolist())


def test_class5():
    d = toy_doc()
    assert B.class5(d, 2, 10) == "assert_pending"
    assert B.class5(d, 2, 20) == "assert_querying"
    assert B.class5(d, 2, 21) == "assert_answered"
    assert B.class5(d, 20, 30) == "query"
    assert B.class5(d, 0, 30) == "filler"


def _planted(n, d, g, rule):
    X = torch.randn(n, d, generator=g, dtype=torch.float64)
    return X, rule(X)


def test_planted_linear_rules_reach_balanced_accuracy_one():
    """B-C5 (test half): each probe form recovers a known linear rule exactly."""
    g = torch.Generator().manual_seed(0)
    # multinomial (kind): 3 classes by a linear argmax
    Wt = torch.randn(3, 6, generator=g, dtype=torch.float64)
    X, y = _planted(600, 6, g, lambda X: (X @ Wt.T).argmax(1))
    p = B.Probe([6], 3, bilinear=False)
    fit = B.fit_probe(p, [X], y, 1e-6)
    assert fit["finite"]
    assert bool((p.predict([X]) == y).all())
    # binary from s only
    w = torch.randn(5, generator=g, dtype=torch.float64)
    X, y = _planted(500, 5, g, lambda X: (X @ w > 0).long())
    p = B.Probe([5], 2, bilinear=False)
    B.fit_probe(p, [X], y, 1e-6)
    assert bool((p.predict([X]) == y).all())
    # bilinear [s; c; s (x) c]: label = sign(s^T A c)
    A = torch.randn(4, 4, generator=g, dtype=torch.float64)
    s_ = torch.randn(800, 4, generator=g, dtype=torch.float64)
    c_ = torch.randn(800, 4, generator=g, dtype=torch.float64)
    y = (torch.einsum("nd,de,ne->n", s_, A, c_) > 0).long()
    p = B.Probe([4, 4], 2, bilinear=True)
    B.fit_probe(p, [s_, c_], y, 1e-7)
    assert bool((p.predict([s_, c_]) == y).all())


def test_balanced_accuracy_and_its_bootstrap():
    # two docs; class 0: 3 cells (2 right), class 1: 1 cell (1 right)
    pred = torch.tensor([0, 0, 1, 1])
    y = torch.tensor([0, 0, 0, 1])
    doc_of = torch.tensor([5, 5, 9, 9])
    counts = B.per_doc_recall_counts(pred, y, doc_of, [5, 9], 2)
    W = torch.ones(10, 2, dtype=torch.float64)
    ba = B.balanced_accuracy(counts, W)
    assert abs(ba["point"] - (2 / 3 + 1) / 2) < 1e-15
    assert ba["lo"] == ba["hi"] == ba["point"]


def test_probe_label_rule():
    ok = {s: True for s in B.SEEDS}

    def ci(lo, hi):
        return {"lo": lo, "hi": hi}

    assert B.probe_label({s: ci(0.96, 0.99) for s in B.SEEDS}, 0.95, ok) == "DECODABLE"
    c = {0: ci(0.96, 0.99), 1: ci(0.90, 0.94), 2: ci(0.96, 0.99)}
    assert B.probe_label(c, 0.95, ok) == "NOT-DECODABLE"
    c = {0: ci(0.94, 0.97), 1: ci(0.96, 0.99), 2: ci(0.96, 0.99)}
    assert B.probe_label(c, 0.95, ok) == "UNRESOLVED"
    conv = {0: True, 1: False, 2: True}
    assert B.probe_label({s: ci(0.96, 0.99) for s in B.SEEDS}, 0.95, conv) == "UNRESOLVED"


def test_class_means_pool_rows_row_weighted():
    d = toy_doc(3)
    G = torch.full((48, 48), float("nan"), dtype=torch.float64)
    for t in range(48):
        G[t, :t] = 1.0
    G[16:, 2] = 5.0
    mu = B.class_means([d], {3: G}, [0])
    assert mu["n_kind"]["assert"] + mu["n_kind"]["query"] + mu["n_kind"]["filler"] == sum(
        t for t in range(16, 48)
    )
    # assert 2 contributes 32 rows of 5; asserts 18 (30 rows: t in [19, 48)) and 25
    n18, n25 = 48 - 19, 48 - 26
    want = (32 * 5 + n18 + n25) / (32 + n18 + n25)
    assert abs(mu["kind"]["assert"] - want) < 1e-12
    assert mu["n_class"]["assert_querying"] == 3  # (20, 2), (30, 18), (40, 25)
