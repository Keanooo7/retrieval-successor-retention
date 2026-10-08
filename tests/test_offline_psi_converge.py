"""Diagnostic A (`experiments/offline-psi-converge/run.py`), PREREG 58783a3.

The pieces that decide a label, on hand-built inputs: the convergence rule (PREREG §2),
the trajectory identity with offline-psi's `fit` loop (C2's premise), and the EVAL seal
(C5)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsr.retention.value_head import BilinearValueHead  # noqa: E402


def _load():
    name = "_test_offline_psi_converge"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "experiments/offline-psi-converge/run.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


CV = _load()
OP = CV.OP
RUN2_OV = {
    "b_enabled": False,
    "beta": 1.0,
    "gamma": 0.9,
    "nu": 0.0,
    "shadow_enabled": False,
    "target": "loo_resample",
}


def _hist(vals: dict[int, float]) -> list[dict]:
    return [{"step": k, "val_l_mc": v} for k, v in sorted(vals.items())]


def test_never_converged_before_20000_steps():
    h = _hist({k: 1.0 for k in range(500, 20_000, 500)})
    assert not any(CV.converged(h, k) for k in range(500, 20_000, 500))


def test_converged_when_best_improves_less_than_0_1_percent_in_10000_steps():
    flat = _hist({k: 1.0 for k in range(500, 20_001, 500)})
    assert CV.converged(flat, 20_000)
    # best(20000) = 0.9985 vs best(10000) = 1.0: 0.15% better -> not converged
    better = _hist({**{k: 1.0 for k in range(500, 10_001, 500)}, 20_000: 0.9985})
    assert not CV.converged(better, 20_000)
    # 0.05% better -> converged
    small = _hist({**{k: 1.0 for k in range(500, 10_001, 500)}, 20_000: 0.9995})
    assert CV.converged(small, 20_000)


def test_converged_uses_the_running_best_not_the_last_value():
    # a late spike does not count as a plateau-breaker, nor does it hide one
    h = _hist({**{k: 1.0 for k in range(500, 10_001, 500)}, 15_000: 0.5, 20_000: 2.0})
    assert not CV.converged(h, 20_000)  # best(20000)=0.5 vs best(10000)=1.0


def test_non_finite_values_are_ignored():
    h = _hist({**{k: 1.0 for k in range(500, 20_001, 500)}, 20_000: float("nan")})
    assert CV.converged(h, 20_000)


def _frozen_dummy():
    m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    m.requires_grad_(False)
    return m


def _fit_data(n=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    s = torch.nn.functional.normalize(torch.randn(n, 48, 128, generator=g), dim=-1)
    c = torch.nn.functional.normalize(torch.randn(n, 48, 128, generator=g), dim=-1)
    teacher = BilinearValueHead(128, generator=torch.Generator().manual_seed(99))
    with torch.no_grad():
        G = OP.psi_matrix(teacher, s, c) * 50.0
    mask = OP.cell_mask("cf_all").expand(n, 48, 48).clone()
    return {"s": s, "c": c, "G": G, "mask": mask}


def test_trajectory_equals_offline_psis_fit_at_shared_steps():
    """C2's premise: the loop is offline-psi's; checking U-val more or less often changes
    nothing (no RNG), so the shared checks are bit-equal and the phi state is too."""
    data = _fit_data()
    train, val = OP.u_split(12)
    p1, _ = OP.new_policy(0, RUN2_OV, 16.0)
    o1 = OP.phi_optimizer(_frozen_dummy(), p1.head, 1e-2)
    ref = OP.fit(p1.head, o1, data, train, val, seed=0, steps=12, every=3)
    p2, _ = OP.new_policy(0, RUN2_OV, 16.0)
    o2 = OP.phi_optimizer(_frozen_dummy(), p2.head, 1e-2)
    mine = CV.fit_converge(
        p2.head, o2, data, train, val, seed=0, cap=12, every=6, save_steps=(12,)
    )
    by_step = {h["step"]: h["val_l_mc"] for h in ref["history"]}
    for h in mine["history"]:
        assert h["val_l_mc"] == by_step[h["step"]]
    assert OP.state_sha(mine["saved"][12]) == OP.state_sha(p1.head.state_dict())
    assert mine["convergence"] == "CAP-HIT" and mine["stopped_at"] == 12


def test_fit_stops_at_the_first_converged_check(monkeypatch):
    monkeypatch.setattr(CV, "converged", lambda history, k: k >= 6)
    data = _fit_data()
    train, val = OP.u_split(12)
    p, _ = OP.new_policy(0, RUN2_OV, 16.0)
    o = OP.phi_optimizer(_frozen_dummy(), p.head, 1e-2)
    r = CV.fit_converge(
        p.head, o, data, train, val, seed=0, cap=30, every=3, save_steps=()
    )
    assert r["convergence"] == "CONVERGED" and r["stopped_at"] == 6
    assert [h["step"] for h in r["history"]] == [3, 6]


def test_eval_seal_makes_every_deploy_entry_raise(monkeypatch):
    for mod, name in ((OP, "deploy_pass"), (OP, "deploy_policy"), (OP.R1, "docs_by_id")):
        monkeypatch.setattr(mod, name, getattr(mod, name))  # restored after the test
    seal = CV._seal_eval()
    assert len(seal["sealed"]) == 3
    with pytest.raises(CV.ControlFailure):
        OP.deploy_pass(None, None, [], {}, None)
    with pytest.raises(CV.ControlFailure):
        OP.R1.docs_by_id(None, [995000])


def test_u_never_contains_an_eval_id():
    ids = [d.doc_id for d in OP.u_documents(0)[0]]
    assert not set(ids) & set(CV.EVAL_IDS)
    assert len(ids) == 1088
