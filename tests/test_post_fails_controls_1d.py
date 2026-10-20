"""Item 1d (post-fails-controls PREREG §6): the grid, the configs, the counts, and the
seal on the modified configs (D-C3 / G-C1(c))."""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
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


D = _load("_pfc_item_1d", "experiments/post-fails-controls/item_1d.py")


def test_grid_is_the_registered_30_points_with_the_default_base():
    g = D.grid()
    assert len(g) == 30 and len(set(g)) == 30
    assert D.BASE in g
    from rsr.data.synthetic import SyntheticConfig

    dflt = SyntheticConfig()
    assert (dflt.geometric_p, dflt.heavy_tail_weight) == D.BASE


def test_configs_differ_from_run1_only_in_the_two_knobs():
    base = D.R1.syn_cfg(D.R1.prereg_config(1, 3000))
    for p, w in D.grid():
        c = D.config(1, p, w)
        a, b = dataclasses.asdict(c), dataclasses.asdict(base)
        diff = {k for k in a if a[k] != b[k]}
        assert diff <= {"geometric_p", "heavy_tail_weight"}
    assert D.config(1, *D.BASE) == base


def test_d_c2_every_config_valid():
    assert D.d_c2()["ok"]


def test_counts_and_pooling():
    hits = [[[1, True], [20, False], [5, True]], [[17, True]]]
    c = D.per_doc_counts(hits, D.BUCKETS["all"])
    assert c.tolist() == [[3.0, 2.0], [1.0, 1.0]]
    assert D.pooled(c) == 3 / 4
    c = D.per_doc_counts(hits, D.BUCKETS["gap_gt_M"])
    assert c.tolist() == [[1.0, 0.0], [1.0, 1.0]]
    assert D.per_doc_counts(hits, D.BUCKETS["gap_2_to_M"]).tolist() == [
        [1.0, 1.0],
        [0.0, 0.0],
    ]


def test_measure_point_small_and_record_format():
    ids = list(range(10, 14))  # not sealed, not EVAL: a unit-test fixture
    r = D.measure_point(0, 0.5, 0.2, ids, D.R1.boot_weights(len(ids), 0, 50))
    h = r["hits"]["fifo"][0]
    assert all(isinstance(x[0], int) and isinstance(x[1], bool) for x in h)
    s = r["summary"]
    assert s["n_documents"] == 4
    assert set(s["contrast"]["all"]) == {"factfiller_minus_fifo", "oracle_minus_fifo"}
    o = s["hit"]["all"]["oracle"]
    assert o >= s["hit"]["all"]["fifo"] - 1e-12  # Belady dominates FIFO on residency


def test_seal_covers_the_modified_configs():
    cm = D.IC.common()
    cm.install_seal()
    for lo, _hi in cm.sealed_ranges().values():
        for p, w in D.grid():
            with pytest.raises(cm.SealedRange):
                D.documents(D.config(2, p, w), [lo])
    assert torch.is_tensor(D.R1.boot_weights(2, 0, 2))
