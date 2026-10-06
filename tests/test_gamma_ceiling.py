"""experiments/gamma-ceiling/run.py (B0-gamma): the pure pieces on hand-built inputs.
No D.pt, no checkpoint and no T0 manifest is read here."""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

from rsr.data.synthetic import SyntheticConfig, generate

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments" / "gamma-ceiling" / "run.py"
NAN = float("nan")


@pytest.fixture(scope="module")
def gc():
    spec = importlib.util.spec_from_file_location("gamma_ceiling_run", RUN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gamma_ceiling_run"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def doc():
    return generate(SyntheticConfig(sentences_per_document=48, seed=0))[0]


def _rand_D(n: int = 48, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    D = torch.rand(n, n, generator=g, dtype=torch.float64)
    t = torch.arange(n).unsqueeze(1)
    i = torch.arange(n).unsqueeze(0)
    return D.masked_fill(i >= t, NAN)


# --------------------------------------------------------------------------- #
# the PREREG transcription
# --------------------------------------------------------------------------- #


def test_gammas_are_the_prereg_seven_in_order(gc):
    assert list(gc.GAMMAS.values()) == [0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0]
    assert gc.ALREADY_SEEN == ("g0", "g09")
    assert gc.HEADLINE == 3000 and set(gc.CHECKPOINTS) == {2500, 3000}
    assert gc.SEEDS == [0, 1, 2]
    assert (gc.POINTS_LO, gc.PARTIAL_POINT) == (0.80, 0.50)


def test_reuses_b0_and_lookahead_room_and_reimplements_nothing(gc):
    """The brief: reuse B0's code by import; do not re-implement simulate."""
    tree = ast.parse(RUN.read_text())
    defined = {
        n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))
    }
    for name in (
        "simulate",
        "boot_rule",
        "boot_weights",
        "discounted_returns",
        "TargetPolicy",
        "FactFiller",
        "verify_manifest",
        "check_shape",
        "pooled",
    ):
        assert name not in defined, name
    assert gc.B0.simulate.__module__ == "rsr.metrics.headroom"
    assert gc.LR is gc.B0.LR


# --------------------------------------------------------------------------- #
# the reading rule (PREREG §6)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("points", "los", "want"),
    [
        ([0.9, 0.95, 1.0], [0.80, 0.81, 0.9], "POINTS"),
        ([0.9, 0.95, 1.0], [0.80, 0.7999, 0.9], "PARTIAL"),
        ([0.5, 0.5, 0.5], [0.1, 0.1, 0.1], "PARTIAL"),
        ([0.5, 0.4999, 0.9], [0.1, 0.1, 0.85], "MISSES"),
        ([0.184, 0.134, 0.537], [0.147, 0.103, 0.508], "MISSES"),
        ([NAN, 0.9, 0.9], [NAN, 0.85, 0.85], "MISSES"),
        ([0.9, 0.9, 0.9], [NAN, 0.85, 0.85], "PARTIAL"),
    ],
)
def test_label_follows_the_prereg_rule(gc, points, los, want):
    assert gc.label(points, los) == want


def test_label_refuses_the_wrong_number_of_seeds(gc):
    with pytest.raises(ValueError):
        gc.label([0.9, 0.9], [0.9, 0.9])


def test_sweep_class(gc):
    keys = list(gc.GAMMAS)
    allm = {k: "MISSES" for k in keys}
    assert gc.sweep_class(allm) == "SIGNAL"
    assert gc.sweep_class({**allm, "g099": "PARTIAL"}) == "MIXED"
    assert gc.sweep_class({**allm, "g099": "PARTIAL", "g1": "POINTS"}) == "HORIZON"
    with pytest.raises(ValueError):
        gc.sweep_class({"g0": "MISSES"})


# --------------------------------------------------------------------------- #
# the returns and the rule
# --------------------------------------------------------------------------- #


def test_gamma0_target_equals_D_exactly(gc):
    """C3 rests on G_0 == D bit-for-bit (B0's rule_g0 is TargetPolicy on D itself)."""
    D = _rand_D()
    G = gc.hindsight_target(D, 0.0)
    past = ~torch.isnan(D)
    assert torch.equal(G[past], D[past])
    assert torch.isnan(G[~past]).all()


def test_gamma1_target_is_the_undiscounted_sum_to_document_end(gc):
    D = _rand_D(seed=1)
    G = gc.hindsight_target(D, 1.0)
    want = torch.flip(torch.cumsum(torch.flip(torch.nan_to_num(D, nan=0.0), [0]), 0), [0])
    past = ~torch.isnan(D)
    assert torch.allclose(G[past], want[past], rtol=0, atol=1e-12)


def test_gamma09_target_matches_lookahead_rooms_rule_g09(gc):
    D = _rand_D(seed=2)
    T = gc.LR.targets(D, torch.zeros_like(D, dtype=torch.bool))
    G = gc.hindsight_target(D, 0.9)
    assert torch.equal(
        torch.nan_to_num(G, nan=-1.0), torch.nan_to_num(T["rule_g09"], nan=-1.0)
    )


def test_hindsight_pairs_gamma0_equals_b0_rule_g0(gc, doc):
    D = _rand_D(seed=3)
    want = gc.B0.pairs_of(doc, gc.LR.TargetPolicy(D))
    assert gc.hindsight_pairs(doc, D, 0.0) == want
    assert len(want) == len(doc.pairs)


def test_hindsight_rule_keeps_the_sentence_with_future_demand(gc, doc):
    """A sentence whose only demand lies beyond the next step is evicted at gamma 0 and
    kept at gamma 1.0 -- the horizon is what the sweep varies."""
    n = 48
    D = torch.full((n, n), 0.5, dtype=torch.float64)
    t = torch.arange(n).unsqueeze(1)
    i = torch.arange(n).unsqueeze(0)
    D = D.masked_fill(i >= t, NAN)
    D[:, 0] = torch.where(
        torch.arange(n) > 0, torch.tensor(0.0, dtype=torch.float64), NAN
    )
    D[40, 0] = 100.0  # sentence 0 is demanded only at step 40
    G0 = gc.hindsight_target(D, 0.0)
    G1 = gc.hindsight_target(D, 1.0)
    pol0, pol1 = gc.LR.TargetPolicy(G0), gc.LR.TargetPolicy(G1)
    from types import SimpleNamespace

    slots = SimpleNamespace(
        live=torch.ones(16, dtype=torch.bool), written_at=torch.arange(16)
    )
    assert pol0.select_eviction(slots, None, 16) == 0
    assert pol1.select_eviction(slots, None, 16) != 0


# --------------------------------------------------------------------------- #
# C3
# --------------------------------------------------------------------------- #


def _b0_rows(vals: dict[str, list[float]]) -> dict:
    return {k: {"key": k, "samples": v} for k, v in vals.items()}


def _mine(h0, h9):
    def r(h):
        return {"hit": h, "cap": {"point": h + 1, "lo": h + 2, "hi": h + 3}}

    return {"g0": r(h0), "g09": r(h9)}


def test_c3_passes_on_exact_equality_and_fails_on_any_difference(gc):
    vals = {
        "B.ckpt3000.hit.rule_g0": [0.1, 0.2, 0.3],
        "B.ckpt3000.cap.rule_g0.point": [1.1, 1.2, 1.3],
        "B.ckpt3000.cap.rule_g0.lo": [2.1, 2.2, 2.3],
        "B.ckpt3000.cap.rule_g0.hi": [3.1, 3.2, 3.3],
        "B.ckpt3000.hit.rule_g09": [0.4, 0.5, 0.6],
        "B.ckpt3000.cap.rule_g09.point": [1.4, 1.5, 1.6],
        "B.ckpt3000.cap.rule_g09.lo": [2.4, 2.5, 2.6],
        "B.ckpt3000.cap.rule_g09.hi": [3.4, 3.5, 3.6],
        "U.hit.fifo.all": [0.8, 0.81, 0.82],
        "U.hit.factfiller.all": [0.96, 0.97, 0.98],
    }
    rows = _b0_rows(vals)
    mine = _mine(0.2, 0.5)
    refs = {"fifo": 0.81, "factfiller": 0.97}
    assert gc.c3_failures(mine, refs, rows, 3000, 1) == []
    mine["g09"]["cap"]["lo"] = 2.5 + 1e-15
    bad = gc.c3_failures(mine, refs, rows, 3000, 1)
    assert len(bad) == 1 and "rule_g09.lo" in bad[0]
    assert gc.c3_failures(
        _mine(0.2, 0.5), {"fifo": 0.8, "factfiller": 0.97}, rows, 3000, 1
    )
    assert any(
        "absent" in x for x in gc.c3_failures(_mine(0.2, 0.5), refs, rows, 2500, 1)
    )


def test_b0_ledger_has_every_row_c3_reads(gc):
    """The committed B0 ledger carries the C3 keys (a missing key would be exit 3)."""
    import json

    rows = {r["key"] for r in json.loads((ROOT / gc.B0_LEDGER).read_text())["rows"]}
    for c in gc.CHECKPOINTS:
        for rule in gc.B0_RULE.values():
            assert f"B.ckpt{c}.hit.{rule}" in rows
            for f in ("point", "lo", "hi"):
                assert f"B.ckpt{c}.cap.{rule}.{f}" in rows
    assert {"U.hit.fifo.all", "U.hit.factfiller.all"} <= rows


def test_defects_flags_a_nan_hit(gc):
    res = {"ckpt": {3000: {"g0": {"by_bucket": {"all": 0.8, "gap_gt_M": NAN}}}}}
    assert gc.defects(res) == ["ckpt3000.g0.hit.gap_gt_M = nan"]
    assert math.isnan(NAN)
