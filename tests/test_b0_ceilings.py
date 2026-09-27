"""experiments/b0-ceilings/run.py (PLAN-v4 B0): the pure pieces on hand-built inputs.
No D.pt, no checkpoint and no T0 manifest is read here."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import math
import random
import sys
from pathlib import Path

import pytest
import torch

from rsr.data.synthetic import SyntheticConfig, generate
from rsr.model.tg.model import init_memory
from rsr.model.tg.policy_loop import memory_state, write_at

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments" / "b0-ceilings" / "run.py"
NAN = float("nan")


@pytest.fixture(scope="module")
def b0():
    spec = importlib.util.spec_from_file_location("b0_ceilings_run", RUN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["b0_ceilings_run"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def doc():
    return generate(SyntheticConfig(sentences_per_document=48, seed=0))[0]


def _full_memory(written: list[int], step: int):
    """A full ``len(written)``-slot MemoryState; slot k holds sentence written[k]."""
    from types import SimpleNamespace

    m = len(written)
    mem = init_memory(1, SimpleNamespace(M=m, D=1))
    for k, w in enumerate(written):
        mem = write_at(
            mem, torch.zeros(1, 1), torch.ones(1, dtype=torch.bool), torch.tensor([k]), w
        )
    return memory_state(mem, step, 0)


# --------------------------------------------------------------------------- #
# the tie rule (B2 A1.10) and the rules
# --------------------------------------------------------------------------- #


def test_random_ties_use_b2s_a1_10_generator_exactly(b0):
    """A1.10: random.Random(f"ko:{seed}:{doc_id}:{t}").choice over the tied slots in
    ascending order -- the same draw B2's kind-oracle makes, and not the oldest."""
    tied = [9, 2, 5, 11]
    picks = set()
    for t in range(16, 48):
        want = random.Random(f"ko:1:777:{t}").choice(sorted(tied))
        got = b0.tie_break(tied, 1, 777, t, "random")
        assert got == want
        picks.add(got)
    assert len(picks) > 1, "the random tie never left one slot: not random"
    assert b0.tie_break(tied, 1, 777, 20, "oldest") == 2


def test_the_kind_oracle_evicts_the_lowest_class_and_ties_randomly(b0, doc):
    """With filler the lowest class, the victim is a filler slot, and which one is
    A1.10's draw among the filler slots."""
    kinds = [s.kind for s in doc.sentences]
    t = 30
    written = list(range(t - 16, t))
    slots = _full_memory(written, t)
    mu = {"kind": {"assert": 0.9, "query": 0.5, "filler": 0.1}}
    pol = b0.rule_policies(doc, {**mu, "kb": {}, "age": {}}, seed=2)["ko"]
    v = pol.select_eviction(slots, None, t)
    fill = [k for k, w in enumerate(written) if kinds[w] == "filler"]
    assert fill, "fixture has no filler in the window"
    assert v == random.Random(f"ko:2:{doc.doc_id}:{t}").choice(sorted(fill))
    old = b0.rule_policies(doc, {**mu, "kb": {}, "age": {}}, seed=2)["ko_oldest"]
    assert old.select_eviction(slots, None, t) == min(fill)


def test_the_age_rule_and_the_band_rule_read_age_at_the_decision_step(b0, doc):
    t = 40
    written = list(range(t - 16, t))
    slots = _full_memory(written, t)
    age = {a: 1.0 for a in range(1, 48)}
    age[7] = 0.0  # the slot holding sentence t - 7
    mu = {"kind": {"assert": 1.0, "query": 1.0, "filler": 1.0}, "kb": {}, "age": age}
    v = b0.rule_policies(doc, mu, seed=0)["age"].select_eviction(slots, None, t)
    assert written[v] == t - 7
    kinds = [s.kind for s in doc.sentences]
    kb = {(k, b): 1.0 for k in b0.KINDS for b in range(len(b0.AGE_BANDS))}
    kb[(kinds[t - 14], b0.age_band(14))] = -1.0  # band A4 = [13, 16] of that kind
    mu["kb"] = kb
    v = b0.rule_policies(doc, mu, seed=0)["kb"].select_eviction(slots, None, t)
    cand = [
        k
        for k, w in enumerate(written)
        if 13 <= t - w <= 16 and kinds[w] == kinds[t - 14]
    ]
    assert v in cand


def test_an_empty_band_cell_falls_back_to_the_kind_mean_and_is_counted(b0, doc):
    kinds = [s.kind for s in doc.sentences]
    mu = {
        "kind": {"assert": 0.3, "query": 0.2, "filler": 0.1},
        "kb": {(k, b): None for k in b0.KINDS for b in range(6)},
        "age": {},
    }
    fb = b0.Fallbacks()
    sc = b0.scorers(doc, mu, fb)
    assert sc["kb"](3, 20) == mu["kind"][kinds[3]]
    assert fb.n == 1


def test_age_bands_are_the_preregistered_edges(b0):
    assert b0.AGE_BANDS == ((1, 4), (5, 8), (9, 12), (13, 16), (17, 24), (25, 47))
    for a, b in (
        (1, 0),
        (4, 0),
        (5, 1),
        (12, 2),
        (13, 3),
        (16, 3),
        (17, 4),
        (24, 4),
        (25, 5),
        (47, 5),
    ):
        assert b0.age_band(a) == b
    for a in (0, 48):
        with pytest.raises(ValueError):
            b0.age_band(a)


# --------------------------------------------------------------------------- #
# rows and class means
# --------------------------------------------------------------------------- #


def test_rows_are_full_memory_steps_and_past_sentences_only(b0):
    r = b0.row_mask(48, 16)
    assert int(r.sum()) == sum(range(16, 48))  # 1008
    assert not r[:16].any()
    assert r[16, 15] and not r[16, 16] and r[47, 0]


def test_class_means_pool_rows_by_kind_band_and_age(b0, doc):
    """G = t - i + 100 * [kind is assert]: every mean is checkable by hand."""
    kinds = [s.kind for s in doc.sentences]
    G = torch.full((48, 48), NAN, dtype=torch.float64)
    for t in range(48):
        for i in range(t):
            G[t, i] = (t - i) + (100.0 if kinds[i] == "assert" else 0.0)
    sums = b0.new_sums()
    per = b0.accumulate(sums, doc, G)
    mu = b0.means(sums)
    rows = [(t, i) for t in range(16, 48) for i in range(t)]
    for k in b0.KINDS:
        xs = [float(G[t, i]) for t, i in rows if kinds[i] == k]
        assert mu["n_kind"][k] == len(xs)
        assert mu["kind"][k] == pytest.approx(sum(xs) / len(xs))
    a7 = [float(G[t, i]) for t, i in rows if t - i == 7]
    assert mu["age"][7] == pytest.approx(sum(a7) / len(a7))
    band = [float(G[t, i]) for t, i in rows if 17 <= t - i <= 24 and kinds[i] == "filler"]
    assert mu["kb"][("filler", 4)] == pytest.approx(sum(band) / len(band))
    xa = [float(G[t, i]) for t, i in rows if kinds[i] == "assert"]
    xf = [float(G[t, i]) for t, i in rows if kinds[i] == "filler"]
    assert per == pytest.approx((sum(xa), len(xa), sum(xf), len(xf)))


def test_the_shape_control_wants_finite_past_and_nan_future(b0):
    D = torch.full((48, 48), NAN, dtype=torch.float64)
    for t in range(48):
        D[t, :t] = 0.5
    assert b0.check_shape(D)
    bad = D.clone()
    bad[20, 3] = NAN
    assert not b0.check_shape(bad)
    bad = D.clone()
    bad[3, 20] = 0.0
    assert not b0.check_shape(bad)
    assert not b0.check_shape(D[:47, :47])


# --------------------------------------------------------------------------- #
# cap and bootstrap
# --------------------------------------------------------------------------- #


def test_the_cap_is_gain_over_fifo_as_a_share_of_factfillers(b0):
    assert b0.cap_point(0.85, 0.80, 0.90) == pytest.approx(0.5)
    assert b0.cap_point(0.75, 0.80, 0.90) == pytest.approx(-0.5)
    assert math.isnan(b0.cap_point(0.85, 0.80, 0.80))


def test_the_bootstrap_is_paired_and_the_point_is_the_pooled_cap(b0):
    n = 50
    fifo = [(10, 8)] * n
    ff = [(10, 9)] * n
    rule = [(10, 8)] * (n // 2) + [(10, 10)] * (n // 2)
    W = b0.boot_weights(n, 0, n_boot=400)
    assert torch.equal(W, b0.boot_weights(n, 0, n_boot=400))
    assert torch.equal(W.sum(1), torch.full((400,), float(n), dtype=torch.float64))
    r = b0.boot_rule(W, rule, fifo, ff)
    assert r["hit"] == pytest.approx(0.9)
    assert r["cap"]["point"] == pytest.approx(1.0)
    assert r["cap"]["lo"] < 1.0 < r["cap"]["hi"]
    assert r["cap_dropped"] == 0
    same = b0.boot_rule(W, ff, fifo, ff)  # the reference against itself: cap == 1 always
    assert same["cap"]["lo"] == pytest.approx(1.0) and same["cap"]["hi"] == pytest.approx(
        1.0
    )


def test_a_replicate_with_no_factfiller_gain_is_dropped_and_counted(b0):
    fifo = [(10, 8)] * 4
    r = b0.boot_rule(b0.boot_weights(4, 0, n_boot=50), fifo, fifo, fifo)
    assert r["cap_dropped"] == 50
    assert math.isnan(r["cap"]["point"]) and math.isnan(r["cap"]["lo"])


def test_f2_difference_is_a_ratio_of_pooled_sums(b0):
    per = [(3.0, 3, 1.0, 2), (1.0, 1, 4.0, 2)]
    r = b0.boot_f2(b0.boot_weights(2, 0, n_boot=100), per)
    assert r["point"] == pytest.approx(4.0 / 4 - 5.0 / 4)


# --------------------------------------------------------------------------- #
# controls
# --------------------------------------------------------------------------- #


def test_the_manifest_check_catches_a_changed_or_unlisted_file(b0, tmp_path):
    f = tmp_path / "a" / "x.D.pt"
    f.parent.mkdir()
    f.write_bytes(b"demand")
    h = hashlib.sha256(b"demand").hexdigest()
    man = tmp_path / "MANIFEST.sha256"
    man.write_text(f"{h}  a/x.D.pt\n")
    assert b0.verify_manifest(man, ["a/x.D.pt"], tmp_path)["ok"]
    f.write_bytes(b"demand!")
    assert not b0.verify_manifest(man, ["a/x.D.pt"], tmp_path)["ok"]
    f.write_bytes(b"demand")
    assert not b0.verify_manifest(man, ["a/x.D.pt", "a/y.D.pt"], tmp_path)["ok"]
    assert not b0.verify_manifest(tmp_path / "absent", ["a/x.D.pt"], tmp_path)["ok"]


def test_a_manifest_mismatch_at_start_is_exit_3_and_reads_nothing(
    b0, tmp_path, monkeypatch
):
    man = tmp_path / "MANIFEST.sha256"
    man.write_text("")
    monkeypatch.setattr(b0, "load_D", lambda *a: pytest.fail("read a D.pt"))

    class Led:
        def __init__(self):
            self.st = None

        def note(self, *a, **k):
            pass

        def status(self, s):
            self.st = s

    led = Led()
    assert b0.execute(led, man, tmp_path, None) == b0.Exit.DID_NOT_RUN
    assert led.st == "did_not_run"


def test_c3_is_exact_equality_against_the_reference_ledger(b0):
    rows = {}
    mine = {}
    for r in b0.C3_RULES:
        for suf in ("", ".gap_gt_M"):
            rows[f"B.ckpt3000.U.hit.{r}{suf}"] = {"samples": [0.1, 0.8093, 0.3]}
            mine[f"{r}{suf}"] = 0.8093
    assert b0.c3_failures(mine, rows, 3000, 1) == []
    mine["rule_g09"] = 0.8093 + 1e-15
    assert len(b0.c3_failures(mine, rows, 3000, 1)) == 1
    del rows["B.ckpt3000.U.hit.oracle"]
    assert len(b0.c3_failures(mine, rows, 3000, 1)) == 2


def test_the_run_never_calls_record_and_loads_no_model(b0):
    tree = ast.parse(RUN.read_text())
    names = {
        n.attr if isinstance(n, ast.Attribute) else getattr(n, "id", None)
        for n in ast.walk(tree)
        if isinstance(n, (ast.Attribute, ast.Name))
    }
    assert "record" not in names
    assert "load_model" not in names and "TGModel" not in names
