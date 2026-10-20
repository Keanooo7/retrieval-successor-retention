"""psi-ls-capacity (`experiments/psi-ls-capacity/run.py`), PREREG 87dc2ec.

The pieces that decide a label, on hand-built inputs at a small width: the structured
normal equations against an explicit design matrix (C2's premise), the solve (planted
head recovered; minimum norm when rank-deficient), the map back to `BilinearValueHead`,
the residual check (C3(a)), and the readout routing (PREREG §6)."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsr.retention.value_head import BilinearValueHead  # noqa: E402


def _load():
    name = "_test_psi_ls_capacity"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "experiments/psi-ls-capacity/run.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


LS = _load()
OP = LS.OP
D, N, T = 5, 7, 9


def _data(seed=0, d=D, n=N, t=T, null_dims=0):
    g = torch.Generator().manual_seed(seed)
    S = torch.nn.functional.normalize(
        torch.randn(n, t, d, generator=g, dtype=torch.float64), dim=-1
    )
    C = torch.nn.functional.normalize(
        torch.randn(n, t, d, generator=g, dtype=torch.float64), dim=-1
    )
    if null_dims:
        S[..., d - null_dims :] = 0.0
    mask = (torch.rand(n, t, t, generator=g) < 0.6) & OP.cell_mask("cf_all", s=t, m=4)
    Y = torch.randn(n, t, t, generator=g, dtype=torch.float64)
    Y = torch.where(mask, Y, torch.full_like(Y, float("nan")))  # off-cell never read
    return S, C, Y, mask


def _explicit(S, C, Y, mask):
    d = S.shape[-1]
    cols = [LS.feature(S, C, j)[mask] for j in range(LS.n_features(d))]
    X = torch.stack(cols, dim=1)
    return X, Y[mask]


def test_normal_equations_equal_the_explicit_design_matrix():
    S, C, Y, mask = _data()
    ne = LS.normal_equations(S, C, Y, mask, chunk=3)  # several chunks
    X, y = _explicit(S, C, Y, mask)
    assert torch.allclose(ne["XtX"], X.T @ X, rtol=0, atol=1e-12)
    assert torch.allclose(ne["Xty"], X.T @ y, rtol=0, atol=1e-12)
    assert abs(ne["yy"] - float(y @ y)) < 1e-10
    assert ne["cells"] == int(mask.sum())
    assert ne["groups"] == int(mask.any(-1).sum())


def test_head_state_maps_w_onto_the_heads_own_forward():
    S, C, Y, mask = _data(1)
    X, _ = _explicit(S, C, Y, mask)
    w = torch.randn(
        LS.n_features(D), generator=torch.Generator().manual_seed(3), dtype=torch.float64
    )
    head = BilinearValueHead(D).double()
    head.load_state_dict(LS.head_state(w, D))
    P = OP.psi_matrix(head, S, C)
    assert torch.allclose(P[mask], X @ w, rtol=0, atol=1e-12)


def test_solve_recovers_a_planted_head_exactly():
    S, C, _, _ = _data(2, n=40)
    mask = OP.cell_mask("cf_all", s=T, m=4).expand(40, T, T).clone()
    teacher = BilinearValueHead(D, generator=torch.Generator().manual_seed(9)).double()
    with torch.no_grad():
        Y = OP.psi_matrix(teacher, S, C)
    ne = LS.normal_equations(S, C, Y, mask)
    sol = LS.solve(ne["XtX"], ne["Xty"])
    assert sol["spectrum"]["rank"] == LS.n_features(D)
    st = LS.head_state(sol["w"], D)
    assert torch.allclose(st["W"], teacher.W.detach(), atol=1e-8)
    assert torch.allclose(st["u"], teacher.u.detach(), atol=1e-8)
    assert LS.quad_sse(ne, sol["w"]) < 1e-12


def test_rank_deficient_solve_is_minimum_norm_and_stationary():
    S, C, Y, mask = _data(4, n=30, null_dims=2)
    ne = LS.normal_equations(S, C, Y, mask)
    sol = LS.solve(ne["XtX"], ne["Xty"])
    p = LS.n_features(D)
    assert sol["spectrum"]["n_dropped"] > 0 and sol["spectrum"]["rank"] < p
    w = sol["w"]
    A = w[: D * D].reshape(D, D)
    assert A[D - 2 :].abs().max() < 1e-10  # s's null coordinates carry no weight
    assert w[D * D + D - 2 : D * D + D].abs().max() < 1e-10
    grad = ne["XtX"] @ w - ne["Xty"]
    assert float(grad.norm() / ne["Xty"].norm()) < 1e-9
    X, y = _explicit(S, C, Y, mask)
    ref = torch.linalg.lstsq(X, y, driver="gelsd").solution
    assert torch.allclose(X @ w, X @ ref, atol=1e-9)


def test_xt_resid_from_cells_equals_the_normal_equation_gradient():
    S, C, Y, mask = _data(5)
    ne = LS.normal_equations(S, C, Y, mask)
    w = torch.randn(
        LS.n_features(D), generator=torch.Generator().manual_seed(1), dtype=torch.float64
    )
    head = BilinearValueHead(D).double()
    head.load_state_dict(LS.head_state(w, D))
    with torch.no_grad():
        R = OP.psi_matrix(head, S, C) - torch.nan_to_num(Y)
    assert torch.allclose(
        LS.xt_resid(S, C, R, mask, chunk=2), ne["XtX"] @ w - ne["Xty"], atol=1e-11
    )


def test_quad_sse_equals_the_direct_sum_of_squares():
    S, C, Y, mask = _data(6)
    ne = LS.normal_equations(S, C, Y, mask)
    X, y = _explicit(S, C, Y, mask)
    w = torch.randn(
        LS.n_features(D), generator=torch.Generator().manual_seed(2), dtype=torch.float64
    )
    assert abs(LS.quad_sse(ne, w) - float(((X @ w - y) ** 2).sum())) < 1e-9


def test_ridge_path_shrinks_and_tends_to_least_squares():
    S, C, Y, mask = _data(7, n=40)
    ne = LS.normal_equations(S, C, Y, mask)
    sol = LS.solve(ne["XtX"], ne["Xty"])
    assert torch.allclose(LS.ridge_w(sol, 1e-14), sol["w"], atol=1e-6)
    norms = [float(LS.ridge_w(sol, r).norm()) for r in LS.RIDGE_RHO]
    assert norms == sorted(norms, reverse=True)


def test_c2_assembly_passes_clean_and_flags_a_corrupted_block():
    S, C, Y, mask = _data(8)
    ne = LS.normal_equations(S, C, Y, mask)
    assert LS.c2_assembly(ne, S, C, Y, mask, seed=0)["ok"]
    bad = dict(ne, XtX=ne["XtX"].clone())
    bad["XtX"][D * D + D :, D * D + D :] += 1e-3  # the c-c block
    assert not LS.c2_assembly(bad, S, C, Y, mask, seed=0)["ok"]


def test_route_follows_prereg_section_6():
    assert "row 1" in LS.route("FIT-PARTIAL", "IN-SAMPLE-MISSES", True)
    assert "row 1" in LS.route("FIT-POINTS", "IN-SAMPLE-POINTS", True)
    assert "row 2" in LS.route("FIT-MISSES", "IN-SAMPLE-MISSES", True)
    assert "row 3" in LS.route("FIT-MISSES", "IN-SAMPLE-PARTIAL", True)
    assert "inconclusive" in LS.route("FIT-POINTS", "IN-SAMPLE-POINTS", False)


def test_every_entry_point_seals_eval_first():
    src = (ROOT / "experiments/psi-ls-capacity/run.py").read_text()
    for fn in ("run_seed", "recompute"):
        body = src.split(f"def {fn}(", 1)[1].split("\ndef ", 1)[0]
        assert body.index("CV._seal_eval()") < body.index("load_inputs(seed)")


def test_launcher_assigns_pids_through_a_scalar_and_checks_the_tree():
    text = (ROOT / "experiments/psi-ls-capacity/launch.zsh").read_text()
    code = "\n".join(ln.split("#", 1)[0] for ln in text.splitlines()[1:])
    assert not re.search(r"\w+\[[^\]]*\]=\$!", code)
    assert "p=$!; pid[$s]=$p" in text
    assert "git status --porcelain" in code


def test_kkt_split_is_zero_on_the_kept_subspace_and_flags_truncation():
    """Amendment 1 item 1: a truncated min-norm solution is stationary on the kept
    eigenvectors only; the dropped part is ``-V_drop^T X^T y`` by construction."""
    S, C, Y, mask = _data(10, n=40)
    ne = LS.normal_equations(S, C, Y, mask)
    sol = LS.solve(ne["XtX"], ne["Xty"])
    sol["keep"] = torch.arange(len(sol["lam"])) >= len(sol["lam"]) // 2  # top half
    keep = sol["keep"]
    w = sol["V"][:, keep] @ (sol["z"][keep] / sol["lam"][keep])
    head = BilinearValueHead(D).double()
    head.load_state_dict(LS.head_state(w, D))
    with torch.no_grad():
        R = OP.psi_matrix(head, S, C) - torch.nan_to_num(Y)
    k = LS.kkt_split(sol, LS.xt_resid(S, C, R, mask), float(ne["Xty"].norm()))
    assert k["kept"] < 1e-10
    want = float(sol["z"][~keep].norm() / ne["Xty"].norm())
    assert abs(k["dropped"] - want) < 1e-10 and k["dropped"] > 1e-3
    assert abs(k["full"] ** 2 - k["kept"] ** 2 - k["dropped"] ** 2) < 1e-10


def test_cutoff_w_at_multiplier_one_is_the_registered_solution():
    S, C, Y, mask = _data(11, n=30, null_dims=2)
    ne = LS.normal_equations(S, C, Y, mask)
    sol = LS.solve(ne["XtX"], ne["Xty"])
    w1, r1 = LS.cutoff_w(sol, 1.0)
    assert torch.equal(w1, sol["w"]) and r1 == sol["spectrum"]["rank"]
    _, r_lo = LS.cutoff_w(sol, 0.01)
    _, r_hi = LS.cutoff_w(sol, 1e6)
    assert r_hi <= r1 <= r_lo


def _fake_res(points_reg, points_low):
    def cap(p):
        return {"point": p, "lo": p - 0.05, "hi": p + 0.05}

    return {
        s: {
            "arms": {
                LS.PRIMARY: {
                    "cap_val": {"cap": cap(points_reg[s])},
                    "cap_train": {"cap": cap(points_reg[s])},
                    "cutoff_sweep": [
                        {
                            "mult": m,
                            "cap_val": cap(points_low[s]),
                            "cap_train": cap(points_low[s]),
                        }
                        for m in LS.CUTOFF_SWEEP
                    ],
                }
            }
        }
        for s in LS.SEEDS
    }


def test_labels_at_the_low_cutoff_drive_the_sensitivity_caveat():
    res = _fake_res({0: 0.3, 1: 0.4, 2: 0.2}, {0: 0.6, 1: 0.7, 2: 0.55})
    assert LS.labels_of(res, LS.PRIMARY) == ("FIT-MISSES", "IN-SAMPLE-MISSES")
    low = LS.labels_of(res, LS.PRIMARY, min(LS.CUTOFF_SWEEP))
    assert low == ("FIT-PARTIAL", "IN-SAMPLE-PARTIAL")
    r = LS.route("FIT-MISSES", "IN-SAMPLE-MISSES", True, sensitive=True)
    assert "row 2" in r and r.endswith("[cutoff-sensitive]")
    assert "sensitive" not in LS.route("FIT-MISSES", "IN-SAMPLE-MISSES", True)
