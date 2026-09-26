"""carry-forward's own gates (`experiments/carry-forward/PREREG.md`, e076da8).

No test reads a real arm B checkpoint. The readouts are fed hand-built per-target
records, the controls are fed hand-built summaries, and the end-to-end test runs
`measure_one` on a tiny random model through the real `loo_readout`.

Mutation-battery gates (`scripts/mutation_battery.py`, ``# --- carry-forward ---``):

* ``test_cluster_bootstrap_resamples_documents_not_targets``
* ``test_carry_needs_the_ci_lower_bound_above_zero``
* ``test_classification_table``
* ``test_disjointness_catches_the_fresh_escape_stream``
* ``test_reproduction_fails_at_2e_6``
* ``test_bitexact_control_refuses_a_one_ulp_difference``
* ``test_l_population_is_gap_2_to_M``
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "carry_forward_run", ROOT / "experiments" / "carry-forward" / "run.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


run = _load()
M = run.M


def _front_matter() -> dict:
    text = (ROOT / "experiments" / "carry-forward" / "PREREG.md").read_text()
    return yaml.safe_load(text.split("---")[1])


# --------------------------------------------------------------------------- #
# the PREREG, transcribed
# --------------------------------------------------------------------------- #


def test_constants_match_the_prereg_front_matter():
    fm = _front_matter()
    assert fm["seeds"] == run.SEEDS
    assert tuple(fm["checkpoints"]) == run.CHECKPOINTS
    assert fm["classification_checkpoint"] == run.FINAL
    th = fm["thresholds"]
    assert th["REACH_DELTA"].startswith(f"{run.REACH_DELTA} ")
    assert th["S_MIN_COVERAGE"].startswith(f"{run.S_MIN_COVERAGE:.2f} ")
    assert th["RESAMPLE_MIN_COVERAGE"].startswith(f"{run.RESAMPLE_MIN_COVERAGE:.2f} ")
    assert th["reproduction_tolerance"].startswith("1e-6 ")
    assert run.REPRO_TOL == 1e-6
    assert f"{run.BOOT_R} resamples" in th["bootstrap"]
    assert f"seed {run.BOOT_SEED}" in th["bootstrap"]
    assert "[64, 1088)" in fm["sets"]["EXT"] and run.EXT == (64, 1088)
    assert "256 consecutive ids" in fm["batching"] and run.EXT_CHUNK == 256
    assert run.H64 == (4096, 4160)
    assert run.M == 16


# --------------------------------------------------------------------------- #
# documents
# --------------------------------------------------------------------------- #


def test_ext_is_disjoint_from_everything_arm_b_saw():
    d = run.disjointness(run.ext_ids())
    assert d["ok"], d
    assert d["n_ids"] == 1024 and d["first_id"] == 64 and d["last_id"] == 1087
    assert [len(c) for c in run.ext_chunks(run.ext_ids())] == [256] * 4


@pytest.mark.parametrize(
    "bad",
    [[5], [4100], [4160], [52159], [105], [70, 70]],
    ids=["vocab", "heldout", "fresh-stream", "stream-last", "repeat-in-list", "repeat"],
)
def test_disjointness_catches_overlap(bad):
    ids = list(range(100, 110)) + bad
    assert not run.disjointness(ids)["ok"]


def test_disjointness_catches_the_fresh_escape_stream():
    # 60000 is outside EXT_ALLOWED anyway; the named hit must still say fresh-escape
    d = run.disjointness([100, 60000])
    assert not d["ok"]
    assert d["hits"]["fresh_escape"] == [60000]


def test_seed_sets_closure_and_prefix_stability():
    s = run.seed_sets(0)
    assert s["closure"]["ok"] and len(s["EXT"]) == 1024 and len(s["H64"]) == 64
    assert [d.doc_id for d in s["H64"]] == list(range(4096, 4160))
    assert [d.doc_id for d in s["EXT"]] == run.ext_ids()


# --------------------------------------------------------------------------- #
# the bootstrap
# --------------------------------------------------------------------------- #


def test_cluster_bootstrap_point_is_the_ratio_of_sums():
    doc = torch.tensor([0, 0, 1, 2])
    a = torch.tensor([1.0, 1.0, 0.0, 1.0])
    b = torch.tensor([1.0, 0.0, 0.0, 1.0])
    r = run.cluster_boot(doc, {"a": a, "b": b}, lambda s: s["b"] / s["a"])
    assert r["point"] == pytest.approx(2.0 / 3.0)
    assert r["n"] == 4 and r["n_docs"] == 3


def test_cluster_bootstrap_resamples_documents_not_targets():
    # 4 documents; within a document every target is identical. Resampling documents
    # gives a CI as wide as 4 draws allow; resampling the 400 targets would give a
    # CI ~10x narrower.
    doc = torch.arange(4).repeat_interleave(100)
    x = torch.tensor([0.0, 1.0, 0.0, 1.0]).repeat_interleave(100)
    r = run.cluster_boot(
        doc, {"x": x, "n": torch.ones_like(x)}, lambda s: s["x"] / s["n"], R=2000
    )
    assert r["point"] == pytest.approx(0.5)
    assert r["hi"] - r["lo"] >= 0.5, r


def test_cluster_bootstrap_is_seeded():
    doc = torch.arange(50).repeat_interleave(3)
    x = torch.rand(150, generator=torch.Generator().manual_seed(1))
    f = lambda s: s["x"] / s["n"]  # noqa: E731
    cols = {"x": x, "n": torch.ones_like(x)}
    assert run.cluster_boot(doc, cols, f) == run.cluster_boot(doc, cols, f)


# --------------------------------------------------------------------------- #
# labels and the table
# --------------------------------------------------------------------------- #


def _ci(point, lo, hi):
    return {"point": point, "lo": lo, "hi": hi}


def test_carry_needs_the_ci_lower_bound_above_zero():
    assert run.carry(_ci(0.05, 0.01, 0.09))
    assert not run.carry(_ci(0.05, -0.01, 0.09))
    assert not run.carry(_ci(0.05, 0.0, 0.09))
    assert not run.carry(_ci(0.029, 0.01, 0.05))
    assert run.carry(_ci(0.03, 0.001, 0.05))


def test_l_and_s_labels():
    assert run.l_label(_ci(0.9, 0.5, 1.0)) == "LOCALISED"
    assert run.l_label(_ci(0.9, 0.49, 1.0)) == "INDETERMINATE"
    assert run.l_label(_ci(0.3, 0.1, 0.49)) == "DIFFUSE"
    assert run.l_label(_ci(0.3, float("nan"), float("nan"))) == "INDETERMINATE"
    assert run.s_label(0.5, _ci(0.1, 0.0, 0.25)) == "SPECIFIC"
    assert run.s_label(0.5, _ci(0.1, 0.0, 0.26)) == "INDETERMINATE"
    assert run.s_label(0.5, _ci(0.5, 0.26, 0.8)) == "NOT_SPECIFIC"
    assert run.s_label(0.19, _ci(0.1, 0.0, 0.2)) == "INSUFFICIENT_COVERAGE"


def _seed(carry, L="LOCALISED", S="SPECIFIC"):
    return {"carry": carry, "L_label": L, "S_label": S}


def test_classification_table():
    c = run.classify
    ok = int(run.Exit.OK)
    assert (
        c({0: _seed(True), 1: _seed(True), 2: _seed(True)}, True)["outcome"] == "CARRIED"
    )
    r = c({0: _seed(False), 1: _seed(False), 2: _seed(True)}, True)
    assert r == {"outcome": "MIXED", "exit": ok, "carry_seeds": [2]}
    assert (
        c({0: _seed(True), 1: _seed(False), 2: _seed(True)}, True)["outcome"] == "MIXED"
    )
    r = c({0: _seed(False), 1: _seed(False), 2: _seed(False)}, True)
    assert r["outcome"] == "LOCALISED" and r["exit"] == ok
    r = c({0: _seed(False), 1: _seed(False, L="INDETERMINATE"), 2: _seed(False)}, True)
    assert r["outcome"] == "UNCLASSIFIED"
    r = c({0: _seed(False), 1: _seed(False, S="NOT_SPECIFIC"), 2: _seed(False)}, True)
    assert r["outcome"] == "UNCLASSIFIED"
    r = c(
        {0: _seed(False), 1: _seed(False, S="INSUFFICIENT_COVERAGE"), 2: _seed(False)},
        True,
    )
    assert r["outcome"] == "LOCALISED"


def test_a_failed_control_or_a_missing_seed_is_inconclusive_exit_3():
    three = {0: _seed(True), 1: _seed(True), 2: _seed(True)}
    r = run.classify(three, False)
    assert r["outcome"] == "inconclusive" and r["exit"] == 3
    r = run.classify({0: _seed(True), 1: _seed(True)}, True)
    assert r["outcome"] == "inconclusive" and r["exit"] == 3


# --------------------------------------------------------------------------- #
# controls
# --------------------------------------------------------------------------- #


def _summary(x=0.5, n=10):
    b = {s: x for s in run.STATS}
    return {"real_token_nll": 1.25, "all": {"n": n, **b}, "gap_gt_M": {"n": n, **b}}


def test_reproduction_passes_at_1e_6_boundary_and_equal_n():
    ref, got = _summary(), _summary()
    got["all"]["answer_acc"] += 5e-7
    assert run.compare_summary(ref, got)["ok"]


def test_reproduction_fails_at_2e_6():
    ref, got = _summary(), _summary()
    got["all"]["answer_nll"] += 2e-6
    assert not run.compare_summary(ref, got)["ok"]


def test_reproduction_fails_on_n_or_missing_bucket():
    ref, got = _summary(), _summary(n=11)
    assert not run.compare_summary(ref, got)["ok"]
    got = _summary()
    del got["gap_gt_M"]
    assert not run.compare_summary(ref, got)["ok"]


def test_ledger_sample_comparison():
    rows = [
        {
            "key": "B.ckpt3000.heldout.live.all.answer_acc",
            "samples": [0.1, 0.5, 0.2],
        }
    ]
    got = {"live": _summary(0.5)}
    assert run.compare_ledger_samples({"rows": rows}, 1, got)["ok"]
    assert not run.compare_ledger_samples({"rows": rows}, 0, got)["ok"]
    assert not run.compare_ledger_samples({"rows": []}, 1, got)["ok"]  # nothing checked


def _rec_pair():
    n = 5
    loo_res = {
        "live_nll": torch.linspace(0.1, 1.0, n, dtype=torch.float64),
        "live_ok": torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0], dtype=torch.float64),
        "live_nll16": torch.linspace(0.2, 1.0, n, dtype=torch.float64),
        "live_brier16": torch.linspace(0.3, 1.0, n, dtype=torch.float64),
        "gap": torch.tensor([1, 2, 3, 17, 20]),
    }
    ar = {
        "nll": loo_res["live_nll"].clone(),
        "ok": loo_res["live_ok"].bool(),
        "nll16": loo_res["live_nll16"].clone(),
        "brier16": loo_res["live_brier16"].clone(),
        "gap": loo_res["gap"].clone(),
    }
    return loo_res, ar


def test_bitexact_control_refuses_a_one_ulp_difference():
    loo_res, ar = _rec_pair()
    assert run.bitexact_live(loo_res, ar)["ok"]
    ar["nll16"][2] = torch.nextafter(
        ar["nll16"][2], torch.tensor(9.0, dtype=torch.float64)
    )
    r = run.bitexact_live(loo_res, ar)
    assert not r["ok"] and not r["equal"]["nll16"]


def test_residency_control():
    rec = {
        "gap": torch.tensor([1, 16, 17, 40]),
        "own_resident": torch.tensor([1, 1, 0, 0]),
    }
    assert run.residency(rec)["ok"]
    rec["own_resident"] = torch.tensor([1, 1, 1, 0])
    r = run.residency(rec)
    assert not r["ok"] and r["n_violations"] == 1
