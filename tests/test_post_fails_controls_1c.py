"""Item 1c (post-fails-controls PREREG §5): the exact agreement metric, the
strictly-after reference, the ES score, the reading rule, and the seal on 1c's
document paths. Hand-built inputs only (no data on disk except the seal test, which
generates no document)."""

from __future__ import annotations

import importlib.util
import random
import sys
from fractions import Fraction
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


C = _load("_pfc_item_1c", "experiments/post-fails-controls/item_1c.py")


def frac(units) -> Fraction:
    return Fraction(int(units), C.UNITS)


def brute(x: list[float], f: list[float], k: int) -> Fraction:
    """Expected overlap of the two bottom-k sets under independent uniform
    tie-breaking, by enumerating every tie-broken bottom-k set."""
    import itertools

    def sets(v):
        n = len(v)
        out = []
        for perm in itertools.permutations(range(n)):
            ranked = sorted(range(n), key=lambda j: (v[j], perm.index(j)))
            out.append(frozenset(ranked[:k]))
        return out

    a, b = sets(x), sets(f)
    tot = sum(len(sa & sb) for sa in a for sb in b)
    return Fraction(tot, len(a) * len(b) * k)


def test_kset_units_ties_by_hand():
    P, nt = C.kset_units(torch.tensor([[3.0, 1.0, 2.0, 1.0]]), 1)
    assert P.tolist() == [[0, 1, 0, 1]] and nt.tolist() == [2]
    P, nt = C.kset_units(torch.tensor([[3.0, 1.0, 2.0, 1.0]]), 3)
    assert P.tolist() == [[0, 1, 1, 1]] and nt.tolist() == [1]
    P, nt = C.kset_units(torch.tensor([[0.0, 0.0, 0.0, 0.0]]), 3)
    assert P.tolist() == [[3, 3, 3, 3]] and nt.tolist() == [4]


def test_kset_units_refuses_nan():
    with pytest.raises(ValueError):
        C.kset_units(torch.tensor([[0.0, float("nan")]]), 1)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_agreement_matches_brute_force_with_ties(k):
    rng = random.Random(7)
    for _ in range(30):
        x = [float(rng.choice([0, 1, 2])) for _ in range(5)]
        f = [float(rng.choice([0, 1, 1, 3])) for _ in range(5)]
        got = C.agreement_units(torch.tensor([x]), torch.tensor([f]), k, top=False)
        assert frac(got[0]) == brute(x, f, k)
        got = C.agreement_units(torch.tensor([x]), torch.tensor([f]), k, top=True)
        assert frac(got[0]) == brute([-v for v in x], [-v for v in f], k)


def test_agreement_extremes_and_uniform_is_chance_exactly():
    f = torch.arange(16, dtype=torch.float64).unsqueeze(0)
    for k in (1, 3):
        assert frac(C.agreement_units(f, f, k, top=False)[0]) == 1
        assert frac(C.agreement_units(f, f, k, top=True)[0]) == 1
        assert frac(C.agreement_units(-f, f, k, top=False)[0]) == 0
        u = torch.zeros_like(f)
        assert C.agreement_units(u, f, k, top=False)[0] == C.chance_units(16, k)
        assert frac(C.chance_units(16, k)) == Fraction(k, 16)
        assert frac(C.chance_units(15, k)) == Fraction(k, 15)


def test_reference_is_strictly_after_t():
    """F_i(t) = G[t+1][i]: Delta[t] (the same-step read) never enters."""
    n = 48
    G = torch.arange(n).unsqueeze(1).double().expand(n, n).clone()  # G[t][i] = t
    T, I_ = C.decision_index()
    F = C.reference_F(G, T, I_)
    assert torch.equal(F[:, 0], (T + 1).double())
    assert T.tolist() == list(range(16, 47))
    assert I_[0].tolist() == list(range(0, 16)) and I_[-1].tolist() == list(range(30, 46))


def test_discounted_return_at_t_plus_1_excludes_same_step():
    LR = C.LR
    n = 48
    D = torch.full((n, n), float("nan"), dtype=torch.float64)
    for t in range(n):
        D[t, :t] = 0.0
    D[20, 5] = 1.0  # demand read at step 20 only
    G = LR.discounted_returns(D, 0.9)
    T, I_ = C.decision_index()
    F = C.reference_F(G, T, I_)
    # at t = 20 the read is the same step: excluded; at t = 19 it is 0.9^0 ahead
    j20, j19 = 20 - 16, 19 - 16
    assert F[j20, 5 - (20 - 16)].item() == 0.0
    assert F[j19, 5 - (19 - 16)].item() == 1.0


def test_pop_cols_bos_excluded_drops_age_one():
    assert C.pop_cols("all_live") == list(range(16))
    assert C.pop_cols("bos_excluded") == list(range(15))  # slot 15 holds t-1


def test_es_scores_argmin_matches_policy():
    from rsr.baselines.expire_span import ExpireSpanConfig, ExpireSpanPolicy

    pol = ExpireSpanPolicy(
        ExpireSpanConfig(
            max_span=48.0,
            ramp=7.68,
            loss_coef=0.0,
            structured_dropout=False,
            grad_path="predictor",
            init_span_fraction=0.1,
            seed=0,
        ),
        d_model=8,
        n_layers=3,
    )
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        pol.weight.copy_(torch.randn(3, 8, generator=g))
    Sg = torch.randn(48, 8, generator=g)
    T, I_ = C.decision_index()
    sc = C.es_scores(pol.spans(Sg), T, I_)
    for j, t in enumerate(T.tolist()):
        ms = C.memory_state(Sg, t)
        assert int(torch.argmin(sc[j])) == pol.select_eviction(ms, torch.zeros(8), t)


def _conds(ok=True):
    one = {
        "ceiling_bottom_lo_above_chance": True,
        "r_i_top_lo_above_chance": True,
        "r_i_bottom_hi_at_or_below_chance": ok,
    }
    return {"1": dict(one), "3": dict(one)}


def test_label_rule():
    assert C.label({s: _conds() for s in C.SEEDS}, True) == "SURVIVES"
    c = {s: _conds() for s in C.SEEDS}
    c[2]["3"]["r_i_top_lo_above_chance"] = False  # k = 3 alone fails: tightened rule
    assert C.label(c, True) == "DROPPED"
    assert C.label({s: _conds() for s in C.SEEDS}, False) == "DROPPED"  # exit 3
    assert C.label({0: _conds(), 1: _conds()}, True) == "DROPPED"  # a missing seed
    assert C.label(None, True) == "DROPPED"


def test_seed_conditions_reads_the_ci_bounds():
    def leaf(lo, hi):
        return {"point": 0.0, "lo": lo, "hi": hi}

    res = {
        "U": {
            "all_live": {
                "all": {
                    "ceiling": {"bottom": {"1": leaf(0.01, 0.1), "3": leaf(0.0, 0.1)}},
                    "r_i": {
                        "top": {"1": leaf(0.2, 0.3), "3": leaf(0.2, 0.3)},
                        "bottom": {"1": leaf(-0.1, 0.0), "3": leaf(-0.1, 0.01)},
                    },
                }
            }
        }
    }
    c = C.seed_conditions(res)
    assert c["1"] == {
        "ceiling_bottom_lo_above_chance": True,
        "r_i_top_lo_above_chance": True,
        "r_i_bottom_hi_at_or_below_chance": True,
    }
    assert c["3"]["ceiling_bottom_lo_above_chance"] is False  # lo == chance: not above
    assert c["3"]["r_i_bottom_hi_at_or_below_chance"] is False


def test_units_table_uniform_point_zero_and_ci_degenerate():
    g = torch.Generator().manual_seed(1)
    N, nd, m = 12, 31, 16
    F = torch.randn(N, nd, m, generator=g, dtype=torch.float64)
    Q = torch.rand(N, nd, generator=g) < 0.3
    scores = {"uniform": torch.zeros(N, nd, m, dtype=torch.float64), "ref": F.clone()}
    U = C.units_table(scores, F, Q)
    W = C.R1.boot_weights(N, 0, 200)
    for pop in C.POPS:
        for ss in C.STEPSETS:
            for k in C.KS:
                b = C.boot_minus_chance(U[("uniform", "bottom", k, pop, ss)], W)
                assert b["point"] == 0.0 and b["lo"] == 0.0 and b["hi"] == 0.0
                r = C.boot_minus_chance(U[("ref", "top", k, pop, ss)], W)
                n = len(C.pop_cols(pop))
                assert r["agreement"] == 1.0 and r["chance"] == k / n
    nQ = int(Q.sum())
    assert int(U[("ref", "bottom", 1, "all_live", "Q")][:, 2].sum()) == nQ


def test_c_c4_flags_a_tied_reference():
    N, nd, m = 2, 31, 16
    F = torch.arange(m, dtype=torch.float64).expand(N, nd, m).clone()
    U = C.units_table(
        {"uniform": torch.zeros_like(F)}, F, torch.zeros(N, nd, dtype=torch.bool)
    )
    assert C.c_c4(U, F)["ok"]
    F2 = F.clone()
    F2[0, 0, 1] = F2[0, 0, 0]  # a tie at the bottom-1 boundary: self-agreement < 1
    U2 = C.units_table(
        {"uniform": torch.zeros_like(F2)}, F2, torch.zeros(N, nd, dtype=torch.bool)
    )
    assert not C.c_c4(U2, F2)["self_is_one"]


def test_seal_guards_1c_document_paths():
    """G-C1(c): with the guard installed, every document constructor 1c can reach
    raises on a sealed id (TUNE and EVAL2 lower bounds, read from the PREREG)."""
    cm = C.common()
    cm.install_seal()
    ranges = cm.sealed_ranges()
    assert set(ranges) == {"TUNE", "EVAL2"}
    from rsr.data import synthetic as syn

    for lo, _hi in ranges.values():
        with pytest.raises(cm.SealedRange):
            syn._generate_document(lo, syn.SyntheticConfig(seed=0))
        with pytest.raises(cm.SealedRange):
            C.R1.docs_by_id(C.R1.prereg_config(0, 3000), [lo])
        with pytest.raises(cm.SealedRange):
            C.LR.documents(1, e_set=(lo, lo + 1))
    assert not cm.seal_leaks()


def _bs(paths, *, errors=(), rc=0, n_refs=3, allowed=()):
    hits = [
        {
            "path": p,
            "token": "997200",
            "value": 997200,
            "range": "TUNE",
            "allowed": p in allowed,
            "refs": ["refs/heads/x"],
        }
        for p in paths
    ]
    return {
        "for_each_ref_rc": rc,
        "n_refs": n_refs,
        "errors": list(errors),
        "hits": hits,
        "new_hits": [h for h in hits if not h["allowed"]],
        "ok": False,
    }


def test_gc1e_allows_only_prereg_citations():
    """Amendment G-C1e: a citation in a committed experiments/*/PREREG*.md is allowed;
    any other hit (code, test, run artefact, other document) is still new: exit 3."""
    ok = C.apply_gc1e(
        _bs(
            [
                "experiments/rsr-run3-ranking/PREREG.md",
                "experiments/post-fails-controls/PREREG.md",
            ],
            allowed=("experiments/post-fails-controls/PREREG.md",),
        )
    )
    assert ok["ok"] and ok["n_new_hits"] == 0
    assert [h["path"] for h in ok["allowed_by_G-C1e"]] == [
        "experiments/rsr-run3-ranking/PREREG.md"
    ]
    assert C.apply_gc1e(_bs(["experiments/x/PREREG-amend.md"]))["ok"]
    for bad in (
        "experiments/rsr-run3-ranking/run.py",
        "tests/test_x.py",
        "runs/x/ledger.json",
        "docs/PLAN.md",
        "experiments/x/RESULTS.md",
        "experiments/a/b/PREREG.md",
        "PREREG.md",
        "experiments/x/notes-PREREG.md",
    ):
        r = C.apply_gc1e(_bs(["experiments/rsr-run3-ranking/PREREG.md", bad]))
        assert not r["ok"] and [h["path"] for h in r["new_hits"]] == [bad], bad
    assert not C.apply_gc1e(_bs([], errors=[{"ref": "r", "rc": 128}]))["ok"]
    assert not C.apply_gc1e(_bs([], rc=1))["ok"]
    assert not C.apply_gc1e(_bs([], n_refs=0))["ok"]
