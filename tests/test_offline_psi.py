"""Offline psi_hat (step 2 after run 2's FAILS): the harness's pure pieces.

`experiments/offline-psi/run.py` against its PREREG (`experiments/offline-psi/PREREG.md`,
c200a08). Every test runs on hand-built or generated inputs -- no B0-CF Delta, no
checkpoint, no EVAL document -- so it runs in CI. The load-bearing ones:

* the target is B0-CF's uncut return (a cell beyond FIFO's horizon keeps its credit);
* C2's arithmetic reproduces an independently pooled cap, compared with ``==``;
* the fit gives gradient to phi only and the optimizer holds phi only (C4);
* the selection reads U-val L_MC only: lowest, ties to the earlier step (§3);
* EVAL is read once, only after selection froze phi (C9), and never twice;
* pairing (C6) is the full ``(doc, [(t, gap)])`` sequence, never a prefix;
* the fit and deploy labels at their boundaries (§4, §5).

Mutants applied by hand at build time (2026-10-08, at 30c3763), each run against this
file alone; every one KILLED (the test that reddened in brackets):

M1 G truncated at FIFO eviction, the censoring put back [target_is_never_cut, c2_...];
M2 PIPELINE reordered so EVAL is read before selection [pipeline_puts_eval_after_...];
M3 the deploy guard skips the frozen-phi sha256 check [deploy_guard_refuses_...];
M4 pairing compares a prefix [pairing_is_the_full_sequence_never_a_prefix];
M5 selection ties go to the later step [select_step_lowest_ties_earliest_...];
M6 L_MC averaged over cells, not (doc, t) groups [l_mc_is_run_2s_sum_then_mean_...];
M7 cf_live one cell too wide [cell_masks_count_cf_all_and_cf_live];
M8 U-val by p % 7 == 1 [u_split_is_positions_mod_7, c3_disjointness_...];
M9 the phi-only check accepts extra trainable tensors [fit_gives_gradient_to_phi_...];
M10 C2 compared with a 1e-9 tolerance instead of == [c2_control_is_float_equality_...].
They are not in `scripts/mutation_battery.py` (neither run 2's nor step 1's harness
is); adding them there is the battery owner's call.
"""

from __future__ import annotations

import importlib.util
import math
import random
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rsr.data.synthetic import SyntheticConfig, generate  # noqa: E402
from rsr.exit_codes import Exit  # noqa: E402
from rsr.metrics.headroom import simulate  # noqa: E402
from rsr.model.tg import TGConfig, TGModel  # noqa: E402
from rsr.retention.value_head import BilinearValueHead  # noqa: E402
from rsr.train import loop  # noqa: E402


def _load():
    name = "_test_offline_psi"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "experiments/offline-psi/run.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


OP = _load()

#: Run 2's psiC overrides (its checkpoint's `policy_overrides`).
RUN2_OV = {
    "b_enabled": False,
    "beta": 1.0,
    "gamma": 0.9,
    "nu": 0.0,
    "shadow_enabled": False,
    "target": "loo_resample",
}


def _docs48(n: int = 6, seed: int = 0):
    return generate(SyntheticConfig(n_documents=n, sentences_per_document=48, seed=seed))


def _delta(doc, s: int = 48) -> torch.Tensor:
    """A hand-built Delta: credit to an assert while its query is still ahead, else a
    small per-doc random value; finite below the diagonal, NaN on and above it."""
    g = torch.Generator().manual_seed(doc.doc_id)
    D = torch.rand(s, s, generator=g, dtype=torch.float64) * 0.01
    q_of = {a: q for a, q in doc.pairs}
    for t in range(s):
        for i in range(t):
            if i in q_of and q_of[i] >= t:
                D[t, i] += 1.0
    below = torch.ones(s, s, dtype=torch.bool).tril(-1)
    return torch.where(below, D, torch.full_like(D, float("nan")))


# --------------------------------------------------------------------------- #
# §2 the target and the cells
# --------------------------------------------------------------------------- #


def test_target_is_never_cut_at_fifo_eviction():
    """PREREG §2: G is NOT cut at eviction. Sentence 0 leaves FIFO's memory at step
    16, but Delta[40][0] still reaches G[20][0] with weight 0.9^20. MUTANT (killed):
    truncating G at FIFO eviction (zeroing Delta[t][i] for t - i > M)."""
    D = torch.full((48, 48), float("nan"), dtype=torch.float64)
    below = torch.ones(48, 48, dtype=torch.bool).tril(-1)
    D[below] = 0.0
    D[40, 0] = 1.0
    G = OP.target_of(D)
    assert G[20, 0].item() == pytest.approx(0.9**20, rel=1e-12)
    assert G[40, 0].item() == 1.0
    assert G[41, 0].item() == 0.0
    assert torch.isnan(G[0, 0]) and torch.isnan(G[5, 7])
    assert torch.equal(
        torch.nan_to_num(G, nan=-1.0),
        torch.nan_to_num(OP.LR.discounted_returns(D, 0.9), nan=-1.0),
    )


def test_cell_masks_count_cf_all_and_cf_live():
    """cf_all: every i < t (1128 cells); cf_live: t - 16 <= i < t (120 + 32 * 16)."""
    a, b = OP.cell_mask("cf_all"), OP.cell_mask("cf_live")
    assert int(a.sum()) == 48 * 47 // 2
    assert int(b.sum()) == sum(range(1, 16)) + 32 * 16
    assert bool((b & ~a).sum() == 0)
    assert bool(b[30, 14]) and not bool(b[30, 13]) and bool(a[30, 13])
    assert not bool(a[5, 5]) and not bool(a[0].any())
    with pytest.raises(ValueError):
        OP.cell_mask("cf_some")


def test_l_mc_is_run_2s_sum_then_mean_over_groups():
    """Sum over the cells of each (doc, t), mean over the (doc, t) holding a cell."""
    P = torch.zeros(1, 3, 3)
    G = torch.tensor([[[0.0, 0, 0], [2.0, 0, 0], [1.0, 3.0, 0]]])
    mask = torch.tensor([[[0, 0, 0], [1, 0, 0], [1, 1, 0]]], dtype=torch.bool)
    loss, groups = OP.l_mc(P, G, mask)
    assert groups == 2
    assert float(loss) == pytest.approx((4.0 + 1.0 + 9.0) / 2)


def test_psi_matrix_is_the_heads_own_slot_scoring():
    """P[b, t, i] equals the head scoring slot s_i against c_t exactly as the
    deployed policy does (one call per decision over its slots)."""
    head = BilinearValueHead(128, generator=torch.Generator().manual_seed(3))
    Sg, Cg = torch.randn(2, 48, 128), torch.randn(2, 48, 128)
    P = OP.psi_matrix(head, Sg, Cg)
    with torch.no_grad():
        for b, t in ((0, 20), (1, 47), (1, 3)):
            want = head(Sg[b, t - 3 : t], Cg[b, t])
            assert torch.allclose(P[b, t, t - 3 : t], want, rtol=0, atol=1e-6)


# --------------------------------------------------------------------------- #
# §1 the U split and disjointness (C3)
# --------------------------------------------------------------------------- #


def test_u_split_is_positions_mod_7():
    train, val = OP.u_split(1088)
    assert len(val) == OP.N_U_VAL == 156 and len(train) == 932
    assert all(p % 7 == 0 for p in val) and all(p % 7 for p in train)
    assert sorted(train + val) == list(range(1088))


def test_c3_disjointness_passes_on_u_and_fails_on_an_overlap():
    ids = list(range(64, 1088)) + list(range(4096, 4160))
    train, val = OP.u_split(len(ids))
    ok = OP.c3_disjoint([ids[p] for p in train], [ids[p] for p in val])
    assert ok["ok"] and ok["sizes"]["U_val"] == 156
    bad = OP.c3_disjoint([ids[p] for p in train] + [995001], [ids[p] for p in val])
    assert not bad["ok"] and "U_train&EVAL" in bad["overlaps"]
    leak = OP.c3_disjoint([ids[p] for p in train], [ids[p] for p in val] + [ids[1]])
    assert not leak["ok"] and "U_train&U_val" in leak["overlaps"]


def test_the_prereg_numbers_are_the_committed_texts():
    """C1 / C2 / C8 compare against numbers transcribed here; each must equal the
    committed PREREG's text (a typo in either reddens)."""
    assert OP.prereg_check()["ok"]
    got = OP.prereg_values(Path(OP.REFS.prereg).read_text())
    assert got["cf_cap"][1] == 0.9551669316375201
    bad = OP.prereg_values("The values are 0.1 / 0.2 / 0.3")
    assert bad["delta_seed"] != OP.DELTA_SEED


# --------------------------------------------------------------------------- #
# C1 and C2 on a fixture path
# --------------------------------------------------------------------------- #


def test_c1_reads_a_fixture_delta_and_refuses_a_bad_one(tmp_path):
    docs = _docs48(3)
    blob = {"Delta": {d.doc_id: _delta(d) for d in docs}}
    torch.save(blob, tmp_path / OP.DELTA_FILE.format(s=0))
    refs = OP.Refs(delta_dir=str(tmp_path))
    got, sha = OP.load_delta(0, refs)
    ids = [d.doc_id for d in docs]
    c1 = OP.c1_delta(got, sha, 0, ids)
    assert c1["ids_ok"] and c1["n_bad"] == 0
    assert not c1["ok"]  # the fixture's sha256 is not PREREG §1's
    c1x = OP.c1_delta(got, OP.DELTA_SHA[0], 0, ids)
    assert c1x["ok"]
    got["Delta"][ids[1]][30, 2] = float("nan")
    assert OP.c1_delta(got, OP.DELTA_SHA[0], 0, ids)["n_bad"] == 1
    assert not OP.c1_delta(got, OP.DELTA_SHA[0], 0, list(reversed(ids)))["ids_ok"]


def test_c2_reproduces_an_independently_pooled_cap_exactly():
    """C2's arithmetic on a fixture: `c2_reproduce` equals the cap pooled from
    lookahead-room's own `rule_hits` / `ref_hits` (a second route through the
    replay), compared with `==`."""
    docs = _docs48(8, seed=1)
    blob = {"Delta": {d.doc_id: _delta(d) for d in docs}}
    got = OP.c2_reproduce(docs, blob, seed=1)
    h = {"rule": [0, 0], "fifo": [0, 0], "factfiller": [0, 0]}
    for d in docs:
        G = OP.LR.discounted_returns(blob["Delta"][d.doc_id], 0.9)
        rows = {"rule": OP.LR.rule_hits(d, {"x": G}, 16)["x"], **OP.LR.ref_hits(d, 1, 16)}
        for k in h:
            h[k][0] += len(rows[k])
            h[k][1] += sum(x[1] for x in rows[k])
    r, f, ff = (h[k][1] / h[k][0] for k in ("rule", "fifo", "factfiller"))
    assert got["point"] == (r - f) / (ff - f)
    assert got["point"] > 0.5  # the pending-credit fixture points at the facts


def test_c2_control_is_float_equality_three_ways():
    c2 = {"point": OP.CF_CAP[0], "registered": OP.CF_CAP[0], "n_documents": OP.N_U}
    assert OP.c2_control(c2, OP.CF_CAP[0])["ok"]
    nudged = math.nextafter(OP.CF_CAP[0], 1.0)
    assert not OP.c2_control({**c2, "point": nudged}, OP.CF_CAP[0])["ok"]
    assert not OP.c2_control(c2, nudged)["ok"]
    assert not OP.c2_control(c2, None)["ok"]
    assert not OP.c2_control({**c2, "n_documents": 1087}, OP.CF_CAP[0])["ok"]


# --------------------------------------------------------------------------- #
# §3 the fit: phi only, selection, determinism
# --------------------------------------------------------------------------- #


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


def test_new_policy_head_is_run_2s_initial_phi():
    """Seed s starts from the phi `train()` builds: phi_generator(s) feeds the head
    first (nothing draws before it); seeds differ."""
    pol, phi_seed = OP.new_policy(0, RUN2_OV, 16.0)
    gen, want_seed = loop.phi_generator(0)
    ref = BilinearValueHead(128, generator=gen)
    assert phi_seed == want_seed
    assert torch.equal(pol.head.W, ref.W) and torch.equal(pol.head.u, ref.u)
    pol1, _ = OP.new_policy(1, RUN2_OV, 16.0)
    assert not torch.equal(pol1.head.W, pol.head.W)


def test_fit_gives_gradient_to_phi_only_and_the_optimizer_holds_phi_only():
    """C4: the optimizer is exactly phi = {W, u} (one value_head group, AdamW with
    loop's betas / eps / decoupled decay); the frozen model contributes no group and
    never receives a gradient; the gestalts are plain tensors."""
    model = _frozen_dummy()
    pol, _ = OP.new_policy(0, RUN2_OV, 16.0)
    opt = OP.phi_optimizer(model, pol.head, 1e-3)
    only = OP.optimizer_holds_only_phi(opt, pol.head)
    assert only["ok"] and only["groups"] == ["value_head"] and only["n_params"] == 2
    g = opt.param_groups[0]
    assert g["betas"] == loop.ADAMW_BETAS and g["eps"] == loop.ADAMW_EPS
    assert g["weight_decay"] == loop.ADAMW_WEIGHT_DECAY and g["lr"] == 1e-3
    data = _fit_data()
    train, val = OP.u_split(12)
    r = OP.fit(pol.head, opt, data, train, val, seed=0, steps=4, every=2)
    assert all(p.grad is not None for p in pol.head.parameters())
    assert all(p.grad is None for p in model.parameters())
    assert not data["s"].requires_grad and data["s"].grad is None
    assert r["c7_n_bad"] == 0
    live = torch.nn.Linear(4, 4)  # a model that still has trainable tensors
    opt2 = OP.phi_optimizer(live, pol.head, 1e-3)
    assert not OP.optimizer_holds_only_phi(opt2, pol.head)["ok"]


def test_select_step_lowest_ties_earliest_never_nonfinite():
    h = [
        {"step": 50, "val_l_mc": 3.0},
        {"step": 100, "val_l_mc": 2.0},
        {"step": 150, "val_l_mc": 2.0},
        {"step": 200, "val_l_mc": float("nan")},
        {"step": 250, "val_l_mc": 2.5},
    ]
    assert OP.select_step(h)["step"] == 100
    assert OP.select_step(list(reversed(h)))["step"] == 100
    assert OP.select_step([{"step": 50, "val_l_mc": float("nan")}]) is None
    assert OP.select_step([]) is None
    assert OP.select_step([{"step": 50, "val_l_mc": float("inf")}]) is None


def test_fit_keeps_the_selected_state_and_is_deterministic():
    """The kept phi is the lowest U-val L_MC's (`select_step`), and a re-fit at the
    same seed reproduces it bit for bit (the refit claim relies on this)."""
    data = _fit_data()
    train, val = OP.u_split(12)
    out = []
    for _ in range(2):
        pol, _ = OP.new_policy(0, RUN2_OV, 16.0)
        opt = OP.phi_optimizer(_frozen_dummy(), pol.head, 1e-2)
        out.append(OP.fit(pol.head, opt, data, train, val, seed=0, steps=12, every=3))
    a, b = out
    sel = OP.select_step(a["history"])
    assert a["selected"] == {"step": sel["step"], "val_l_mc": sel["val_l_mc"]}
    assert [h["step"] for h in a["history"]] == [3, 6, 9, 12]
    assert OP.state_sha(a["state"]) == OP.state_sha(b["state"])
    assert a["history"][-1]["val_l_mc"] < a["val_l_mc_step0"]


def test_the_fit_reads_u_train_only_for_gradient_and_u_val_only_for_selection():
    """Poisoning a U-val document's G cannot change any gradient step (only the
    selection's numbers); poisoning a U-train document changes the steps."""
    data = _fit_data()
    train, val = OP.u_split(12)

    def run(d):
        pol, _ = OP.new_policy(0, RUN2_OV, 16.0)
        opt = OP.phi_optimizer(_frozen_dummy(), pol.head, 1e-2)
        r = OP.fit(pol.head, opt, d, train, val, seed=0, steps=6, every=6)
        return OP.state_sha({k: v for k, v in pol.head.state_dict().items()}), r

    base, _ = run(data)
    pv = {**data, "G": data["G"].clone()}
    pv["G"][val[0]] += 100.0
    same, _ = run(pv)
    assert same == base
    pt = {**data, "G": data["G"].clone()}
    pt["G"][train[0]] += 100.0
    diff, _ = run(pt)
    assert diff != base


# --------------------------------------------------------------------------- #
# §1 the gestalt forward (C5) on a tiny model
# --------------------------------------------------------------------------- #


def _tiny_world(n=5, s=16, m=4, L=10):
    docs = generate(
        SyntheticConfig(
            n_documents=n, sentences_per_document=s, max_gap=12, heavy_tail_min=6, seed=0
        )
    )
    vmap = loop.build_vocab(
        generate(
            SyntheticConfig(
                n_documents=64,
                sentences_per_document=s,
                max_gap=12,
                heavy_tail_min=6,
                seed=0,
            )
        )
    )
    torch.manual_seed(0)
    cfg = TGConfig(
        D=16,
        H=2,
        N=4,
        V=4 + len(vmap),
        block_config=("S", "C", "S", "C"),
        srep_extraction_layer=2,
        max_sentence_tokens=L,
        max_sentences_in_short_term=m,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        eod_id=3,
    )
    return TGModel(cfg), docs, vmap


def test_gestalt_forward_is_fifos_world_and_records_c_t():
    model, docs, vmap = _tiny_world()
    model.train()
    g = OP.gestalt_forward(model, docs, vmap, batch=2, m=4, steps=16)
    assert g["c5"]["ok"], g["c5"]
    assert g["c5"]["full_steps_checked"] == len(docs) * (16 - 4)
    assert g["s"].shape == (5, 16, 16) and torch.equal(g["s"], g["c"])
    assert model.training  # the mode is restored
    g2 = OP.gestalt_forward(model, docs, vmap, batch=5, m=4, steps=16)
    assert torch.allclose(g["s"], g2["s"], atol=1e-5)


def test_c5_fails_when_the_forward_is_not_fifo(monkeypatch):
    model, docs, vmap = _tiny_world(n=2)

    class NotFifo(OP.FIFOPolicy):
        def select_eviction(self, slots, context, step):
            return 1

    monkeypatch.setattr(OP, "FIFOPolicy", NotFifo)
    g = OP.gestalt_forward(model, docs, vmap, batch=2, m=4, steps=16)
    assert not g["c5"]["ok"] and g["c5"]["n_problems"] > 0


# --------------------------------------------------------------------------- #
# §5 the fit reading
# --------------------------------------------------------------------------- #


def test_replay_of_g_is_c2s_rule_and_rho_is_signed():
    docs = _docs48(4, seed=2)
    blob = {"Delta": {d.doc_id: _delta(d) for d in docs}}
    Gs = [OP.target_of(blob["Delta"][d.doc_id]) for d in docs]
    assert OP.replay_rates(docs, Gs) == OP.c2_reproduce(docs, blob, 2)["rates"]
    W = OP.B0.boot_weights(4, 0, 50)
    G = torch.stack(Gs)
    assert OP.rho_psi_g(G, G, W)["median"] == pytest.approx(1.0)
    assert OP.rho_psi_g(-G, G, W)["median"] == pytest.approx(-1.0)


def test_fit_label_boundaries():
    def caps(points, los):
        return {
            s: {"point": p, "lo": lo, "hi": 1.0}
            for s, p, lo in zip((0, 1, 2), points, los, strict=True)
        }

    assert OP.fit_label(caps([0.9, 0.9, 0.9], [0.80, 0.81, 0.95])) == "FIT-POINTS"
    assert OP.fit_label(caps([0.9, 0.9, 0.9], [0.80, 0.79, 0.95])) == "FIT-PARTIAL"
    assert OP.fit_label(caps([0.50, 0.6, 0.7], [0.1, 0.1, 0.1])) == "FIT-PARTIAL"
    assert OP.fit_label(caps([0.49, 0.9, 0.9], [0.1, 0.9, 0.9])) == "FIT-MISSES"
    nan = float("nan")
    assert OP.fit_label(caps([0.9, 0.9, 0.9], [nan, 0.9, 0.9])) == "FIT-PARTIAL"
    assert OP.fit_label(caps([nan, 0.9, 0.9], [nan, 0.9, 0.9])) == "FIT-MISSES"


# --------------------------------------------------------------------------- #
# §4 deployment: T_warm = 0, the label, the class, EVAL once (C9), pairing (C6)
# --------------------------------------------------------------------------- #


def test_deploy_policy_is_the_frozen_phi_with_no_warmup():
    pol0, _ = OP.new_policy(0, RUN2_OV, 16.0)
    state = {k: v.clone() + 0.5 for k, v in pol0.head.state_dict().items()}
    pol = OP.deploy_policy(state, 0, RUN2_OV, 16.0)
    assert pol.config.t_warm == 0.0 and not pol.config.b_enabled and pol.config.nu == 0
    assert OP.state_sha(pol.head.state_dict()) == OP.state_sha(state)
    from rsr.retention.policy import MemoryState

    g = torch.nn.functional.normalize(torch.randn(16, 128), dim=-1)
    c = torch.nn.functional.normalize(torch.randn(128), dim=-1)
    ms = MemoryState(
        gestalts=g,
        written_at=torch.arange(16),
        live=torch.ones(16, dtype=torch.bool),
        step=16,
        row=0,
    )
    k = pol.select_eviction(ms, c, 16)  # no set_train_step: T_warm = 0, never FIFO
    with torch.no_grad():
        assert k == int(pol.head(g, c).argmin())


def _recs(n, ok_fn, gaps=(1, 3, 20)):
    out = []
    for j in range(n):
        ans = [(5 + i, gp, bool(ok_fn(j, i)), 0.1) for i, gp in enumerate(gaps)]
        out.append(
            {"doc": 995000 + j, "answers": ans, "hits": [(gp, True) for gp in gaps]}
        )
    return out


def test_headline_label_and_class():
    W = OP.R1.boot_weights(40, 0, 200)
    good, bad = _recs(40, lambda j, i: True), _recs(40, lambda j, i: False)
    assert OP.headline(good, bad, W, 0.038)["label"] == "WIN"
    assert OP.headline(bad, good, W, 0.038)["label"] == "LOSS"
    assert OP.headline(good, good, W, 0.038)["label"] == "EQUIV"
    assert OP.headline(good, bad, W, None)["label"] == "UNRESOLVED"
    mixed = _recs(40, lambda j, i: (j + i) % 2 == 0)
    assert OP.headline(mixed, good, W, 0.038)["label"] in ("LOSS", "UNRESOLVED")
    assert OP.deploy_class({0: "WIN", 1: "WIN", 2: "WIN"}) == "BEATS-FIFO"
    assert OP.deploy_class({0: "WIN", 1: "LOSS", 2: "WIN"}) == "FAILS"
    assert OP.deploy_class({0: "WIN", 1: "EQUIV", 2: "WIN"}) == "MIXED"
    assert OP.deploy_class({0: "WIN", 1: "WIN"}) == "MIXED"


def test_pairing_is_the_full_sequence_never_a_prefix():
    a = _recs(5, lambda j, i: True)
    assert OP.pairing(a, _recs(5, lambda j, i: False))["ok"]  # same (doc, t, gap)
    assert not OP.pairing(a[:4], a)["ok"]
    assert not OP.pairing(a, a[:4])["ok"]
    assert not OP.pairing(a, _recs(5, lambda j, i: True, gaps=(1, 3, 21)))["ok"]
    assert not OP.pairing([], [])["ok"]


def test_deploy_guard_refuses_without_selection_mismatch_or_existing_output(tmp_path):
    pol, _ = OP.new_policy(0, RUN2_OV, 16.0)
    st = pol.head.state_dict()
    phi = tmp_path / "phi-cf_all.pt"
    torch.save(st, phi)
    out = tmp_path / "heldout" / "cf_all.jsonl.gz"
    sel = {"step": 50, "val_l_mc": 1.0, "phi_sha256": OP.state_sha(st)}
    assert OP.deploy_guard(sel, phi, out)["ok"]
    with pytest.raises(OP.Refused):
        OP.deploy_guard(None, phi, out)
    with pytest.raises(OP.Refused):
        OP.deploy_guard({**sel, "phi_sha256": "0" * 64}, phi, out)
    out.parent.mkdir(parents=True)
    out.write_text("")
    with pytest.raises(OP.Refused, match="read once"):
        OP.deploy_guard(sel, phi, out)


def test_measure_seed_refuses_a_rerun_once_eval_output_exists(tmp_path):
    """C9: EVAL is read once per seed -- refused before anything loads."""
    hd = OP.heldout_dir(tmp_path, 0)
    hd.mkdir(parents=True)
    (hd / "cf_all.jsonl.gz").write_text("")
    with pytest.raises(OP.Refused, match="once"):
        OP.measure_seed(0, tmp_path, threads=1)


def test_pipeline_puts_eval_after_selection():
    """C9: "deploy" (the only EVAL read) runs after "select" froze phi, sees its
    output, and runs once. MUTANT (killed): EVAL read before selection (PIPELINE
    reordered)."""
    assert OP.PIPELINE.index("select") < OP.PIPELINE.index("deploy")
    assert OP.PIPELINE.index("fit") < OP.PIPELINE.index("select")
    seen = {}
    steps = {n: (lambda ctx, n=n: n) for n in OP.PIPELINE}
    steps["deploy"] = lambda ctx: seen.setdefault("select_output", ctx.get("select"))
    events: list[str] = []
    OP.run_pipeline(dict(reversed(list(steps.items()))), events)
    assert events == list(OP.PIPELINE)
    assert seen["select_output"] == "select"
    assert OP.c9_order(events)["ok"]
    swapped = ["controls", "gestalts", "fit", "deploy", "select", "score"]
    assert not OP.c9_order(swapped)["ok"]
    assert not OP.c9_order([*OP.PIPELINE, "deploy"])["ok"]


# --------------------------------------------------------------------------- #
# §6 secondary readings on hand-built records
# --------------------------------------------------------------------------- #


def test_rho_psi_age_reads_the_probe_decisions():
    W = OP.R1.boot_weights(2, 0, 50)
    ages = torch.arange(16, 0, -1)
    dec = [(0, 16, ages.double(), None, ages, 0), (1, 17, -ages.double(), None, ages, 15)]
    r = OP.rho_psi_age(dec, W)
    assert r["n_steps"] == 2 and r["median"] == pytest.approx(-1.0)


# --------------------------------------------------------------------------- #
# the CLI's refusals
# --------------------------------------------------------------------------- #


def test_smoke_only_flags_and_out_roots_are_refused():
    base = ["measure-seed", "--seed", "0", "--threads", "1"]
    assert OP.main([*base, "--fit-steps", "5"]) == Exit.DID_NOT_RUN
    assert OP.main([*base, "--smoke"]) == Exit.DID_NOT_RUN  # a smoke needs --standin
    real = str(ROOT / "runs" / OP.RUN_ID)
    assert OP.main([*base, "--smoke", "--standin", "2", "--out-root", real]) == (
        Exit.DID_NOT_RUN
    )
    assert (
        OP.main([*base, "--out-root", "/tmp/elsewhere/offline-psi"]) == Exit.DID_NOT_RUN
    )


def test_manifest_is_required_and_must_match(tmp_path):
    with pytest.raises(OP.Refused, match="missing"):
        OP.check_manifest(tmp_path)
    OP.write_manifest(tmp_path, smoke=True)
    assert OP.check_manifest(tmp_path)["run_id"] == OP.RUN_ID
    p = tmp_path / "manifest.json"
    p.write_text(p.read_text().replace('"fit_steps": 3000', '"fit_steps": 30'))
    with pytest.raises(OP.Refused, match="does not match"):
        OP.check_manifest(tmp_path)


def test_factfiller_reference_is_seeded_per_document():
    """B0-CF's FactFiller seeding: random.Random(f"ff:{seed}:{doc_id}")."""
    docs = _docs48(2)
    r = OP.reference_rates(docs, 3)["factfiller"]
    want = [
        OP.B0.rate(
            [
                (q["gap"], q["hit"])
                for q in simulate(
                    d, OP.LR.FactFiller(d, random.Random(f"ff:3:{d.doc_id}")), 16
                )["queries"]
            ]
        )
        for d in docs
    ]
    assert r == want
