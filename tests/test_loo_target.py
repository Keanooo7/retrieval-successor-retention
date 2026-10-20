"""The on-policy resample-knockout LOO target (owner ruling C3, 2026-10-07).

`rsr.retention.loo_target` forms E0d's `Delta` inside `run_policy_loop`, over the
policy's own memory; `RSRPolicy(target="loo_resample")` regresses on its MC return.
The load-bearing tests:

* on a FIFO memory, the loop's `Delta` is **bit-equal** to
  `rsr.metrics.loo.loo_delta_loss(mode="resample")` -- same donors, same NaNs;
* on an on-policy (non-FIFO) memory, each `Delta` cell is the single knockout it
  claims to be, recomputed by hand;
* no retention gradient reaches any transformer parameter (§3.3, CLAUDE.md);
* the default config regresses on run 1's `r_i`, exactly, and needs no LOO.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsr.data.synthetic import SyntheticConfig, generate  # noqa: E402
from rsr.metrics import loo  # noqa: E402
from rsr.model.tg import TGConfig, TGModel  # noqa: E402
from rsr.model.tg.policy_loop import run_policy_loop  # noqa: E402
from rsr.retention import loo_target as LT  # noqa: E402
from rsr.retention.reward import retrieval_demand  # noqa: E402
from rsr.retention.rsr import (  # noqa: E402
    AGE_ONLY,
    REDUCTION_SWITCHES,
    RSRConfig,
    RSRPolicy,
)
from rsr.train import loop  # noqa: E402

S, L, M = 16, 10, 4
D = 16


def _docs(n=8, seed=0):
    return generate(
        SyntheticConfig(
            n_documents=n,
            sentences_per_document=S,
            max_gap=12,
            heavy_tail_min=6,
            seed=seed,
        )
    )


def _world(n=8, seed=0, model_seed=0):
    docs = _docs(n, seed)
    vmap = loop.build_vocab(_docs(64, seed))
    ids, mask = loop.encode(docs, vmap, max_tokens=L, steps=S)
    torch.manual_seed(model_seed)
    cfg = TGConfig(
        D=D,
        H=2,
        N=4,
        V=4 + len(vmap),
        block_config=("S", "C", "S", "C"),
        srep_extraction_layer=2,
        max_sentence_tokens=L,
        max_sentences_in_short_term=M,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        eod_id=3,
    )
    return TGModel(cfg), docs, ids, mask


def _cfg(*, target=LT.TARGET_LOO_RESAMPLE, t_warm=0.0, psi=None) -> RSRConfig:
    return RSRConfig(
        nu=0.0,
        beta=1.0,
        gamma=0.9,
        t_warm=t_warm,
        a_max=M,
        psi_override=psi,
        b_enabled=False,
        shadow_enabled=False,
        target=target,
    )


def _policy(**kw) -> RSRPolicy:
    p = RSRPolicy(_cfg(**kw), d_model=D, generator=torch.Generator().manual_seed(7))
    p.set_train_step(0)
    return p


def _lm(t, out, ids, mask, row_valid):
    return loop.lm_loss(out.logits, ids, mask)


class _SpyTarget(LT.LooTarget):
    """Records every `LooStep` and the memory it was measured on."""

    def __init__(self, ann, seed):
        super().__init__(ann=ann, seed=seed)
        self.steps: list[LT.LooStep] = []
        self.mems: list = []

    def __call__(self, model, ids_t, mask_t, mem, bos_ctx, bos_valid, logits, t):
        out = super().__call__(model, ids_t, mask_t, mem, bos_ctx, bos_valid, logits, t)
        self.steps.append(out)
        self.mems.append((mem, bos_ctx.clone(), bos_valid.clone()))
        return out


def _run(model, docs, ids, mask, policy, *, seed=123):
    spy = _SpyTarget(loo.stream_annotations(docs, steps=S), seed)
    lengths = torch.full((ids.shape[0],), S)
    run_policy_loop(
        model,
        ids,
        mask,
        lengths,
        policy,
        step_fn=_lm,
        observe=True,
        eval_capture=True,
        loo=spy,
    )
    return spy


# --------------------------------------------------------------------------- #
# donor choice
# --------------------------------------------------------------------------- #


def test_pick_donor_fast_is_pick_donor():
    """Same candidates, same order, same seeded draw, on random memories."""
    rng = random.Random(0)
    for trial in range(300):
        B, Mm, Ss = rng.randint(1, 6), rng.randint(1, 5), rng.randint(2, 9)
        ann = {
            "kind": torch.randint(
                0, 3, (B, Ss), generator=torch.Generator().manual_seed(trial)
            ),
            "fact_key": torch.randint(
                -1, 4, (B, Ss), generator=torch.Generator().manual_seed(trial + 1)
            ),
            "doc_id": torch.tensor([rng.randint(0, 3) for _ in range(B)]),
        }
        step = torch.randint(
            0, Ss, (B, Mm), generator=torch.Generator().manual_seed(trial + 2)
        )
        valid = (
            torch.rand(B, Mm, generator=torch.Generator().manual_seed(trial + 3)) > 0.3
        )
        row, rank = rng.randrange(B), rng.randrange(Mm)
        kw = dict(
            row=row,
            rank=rank,
            want_kind=rng.randint(0, 2),
            exclude_keys={rng.randint(-1, 3)},
            seed=rng.randint(0, 10**9),
            t=rng.randint(0, 50),
            role=16 + rank,
        )
        want = loo.pick_donor(ann, step, valid, **kw)
        got = LT.pick_donor_fast(
            ann["kind"].tolist(),
            ann["fact_key"].tolist(),
            ann["doc_id"].tolist(),
            step.tolist(),
            valid.tolist(),
            **kw,
        )
        assert got == want, (trial, got, want)


def test_loo_seed_is_injective_and_bounded():
    seen = {LT.loo_seed(s, k) for s in range(3) for k in range(0, 3000, 7)}
    assert len(seen) == 3 * len(range(0, 3000, 7))
    with pytest.raises(ValueError):
        LT.loo_seed(0, 10**7)


# --------------------------------------------------------------------------- #
# exactness: FIFO memory -> loo_delta_loss, bit for bit
# --------------------------------------------------------------------------- #


def test_fifo_memory_delta_is_bit_equal_to_loo_delta_loss():
    """The policy in warmup evicts FIFO, so its memory IS loo_delta_loss's; the
    loop's Delta, donors and NaNs must then be loo_delta_loss's exactly."""
    model, docs, ids, mask = _world(n=6)
    model.eval()
    ref = loo.loo_delta_loss(model, docs, ids, mask, mode="resample", seed=123)
    pol = _policy(t_warm=1e9)  # k = 0 < T_warm: FIFO throughout
    spy = _run(model, docs, ids, mask, pol, seed=123)
    assert len(spy.steps) == S
    got = torch.stack([s.delta for s in spy.steps], dim=1)  # [B, S, M]
    assert torch.equal(torch.isnan(got), torch.isnan(ref["delta"]))
    fin = ~torch.isnan(got)
    assert fin.any()
    assert torch.equal(got[fin], ref["delta"][fin])  # bit-equal, not close
    dr = torch.stack([s.donor_row for s in spy.steps], dim=1)
    ds = torch.stack([s.donor_sentence for s in spy.steps], dim=1)
    assert torch.equal(dr, ref["donor_row"])
    assert torch.equal(ds, ref["donor_sentence"])
    assert all(r.victim == 0 for r in pol.records)


def test_policy_target_is_the_imputed_delta_at_live_cells():
    """What observe() buffers is Delta with NaN -> 0 (choice (b)), per live slot."""
    model, docs, ids, mask = _world(n=6)
    model.eval()
    pol = _policy(t_warm=1e9)
    spy = _run(model, docs, ids, mask, pol)
    for row, obs in pol._mc.items():
        for o in obs:
            st = spy.steps[o.step]
            live = st.status[row] >= 0
            want = torch.nan_to_num(st.delta[row][live], nan=0.0).float()
            assert torch.equal(o.r.cpu(), want)


# --------------------------------------------------------------------------- #
# on-policy: a memory FIFO would not hold
# --------------------------------------------------------------------------- #


def test_on_policy_delta_is_the_knockout_it_claims_on_a_non_fifo_memory():
    model, docs, ids, mask = _world(n=6)
    model.eval()
    pol = _policy(t_warm=0.0)  # scored from k = 0: the head decides
    spy = _run(model, docs, ids, mask, pol)
    assert any(r.victim != 0 for r in pol.records), "memory never left FIFO"
    checked = 0
    for t, (st, (mem, bos_ctx, bos_valid)) in enumerate(
        zip(spy.steps, spy.mems, strict=True)
    ):
        if t < M:
            continue
        ids_t, mask_t = ids[:, t], mask[:, t]
        with torch.no_grad():
            base = LT._sent_loss(
                model(ids_t, mask_t, mem.kv, mem.valid, bos_ctx, bos_valid).logits,
                ids_t,
                mask_t,
            )
            for b in range(ids.shape[0]):
                for i in range(M):
                    if int(st.status[b, i]) != LT.STATUS_OK:
                        continue
                    kv = mem.kv.clone()
                    kv[b, i] = mem.kv[int(st.donor_row[b, i]), i]
                    o = model(ids_t, mask_t, kv, mem.valid, bos_ctx, bos_valid)
                    want = LT._sent_loss(o.logits, ids_t, mask_t)[b] - base[b]
                    assert float(st.delta[b, i]) == pytest.approx(float(want), abs=1e-9)
                    # same rank, different doc, same kind (choice (a))
                    dr, dsn = int(st.donor_row[b, i]), int(st.donor_sentence[b, i])
                    assert int(mem.step[dr, i]) == dsn
                    assert int(
                        loo.stream_annotations(docs, steps=S)["doc_id"][dr]
                    ) != int(docs[b].doc_id)
                    gap = abs(int(mem.step[dr, i]) - int(mem.step[b, i]))
                    assert int(st.donor_age_gap[b, i]) == gap
                    checked += 1
    assert checked > 20


def test_missing_donors_are_imputed_zero_and_counted_by_kind():
    """B = 1: no other document, so no donor exists anywhere."""
    model, docs, ids, mask = _world(n=1)
    model.eval()
    pol = _policy(t_warm=1e9)
    _run(model, docs, ids, mask, pol)
    for o in pol._mc[0]:
        assert torch.equal(o.r, torch.zeros_like(o.r))
    cov = pol.loo_coverage()
    n_live = sum(o.keys.numel() for o in pol._mc[0])
    assert sum(cov["counts"].values()) == n_live
    assert all(".no_donor." in k for k in cov["counts"])
    assert cov["full_coverage"] == 0.0


def test_coverage_counts_every_live_cell_once():
    model, docs, ids, mask = _world(n=6)
    model.eval()
    pol = _policy(t_warm=0.0)
    _run(model, docs, ids, mask, pol)
    cov = pol.loo_coverage()
    n_live = sum(o.keys.numel() for obs in pol._mc.values() for o in obs)
    assert sum(cov["counts"].values()) == n_live
    full = sum(v for k, v in cov["counts"].items() if k.startswith("full."))
    assert cov["full_cells"] == full
    assert 0.0 <= cov["full_coverage"] <= 1.0
    loss = pol.retention_loss()
    assert loss is not None
    st = pol.last_mc_stats
    assert st["target"] == LT.TARGET_LOO_RESAMPLE
    assert st["loo"]["full_cells"] == full
    diag = st["target_diag"]
    assert sum(n for _, n in diag["g_by_age"].values()) == st["n_samples"]
    assert sum(n for _, n in diag["g_by_evict_age"].values()) == st["n_samples"]
    assert diag["rho_g_age"]["n"] > 0


# --------------------------------------------------------------------------- #
# §3.3: no retention gradient into the transformer
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("psi", [None, AGE_ONLY])
def test_no_gradient_reaches_any_transformer_parameter(psi):
    model, docs, ids, mask = _world(n=6)
    model.train()  # grads ON for the model; dropout live in the training forward
    for p in model.parameters():
        p.requires_grad_(True)
        p.grad = None
    pol = _policy(t_warm=0.0, psi=psi)
    spy = _run(model, docs, ids, mask, pol)
    assert model.training  # the knockouts ran in eval and restored the mode
    for st in spy.steps:
        assert not st.delta.requires_grad
    loss = pol.retention_loss()
    loss.backward()
    assert all(p.grad is None for p in model.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in pol.head.parameters()
    )


def test_mode_is_restored_after_the_knockouts():
    model, docs, ids, mask = _world(n=4)
    model.train()
    _run(model, docs, ids, mask, _policy(t_warm=0.0))
    assert model.training


# --------------------------------------------------------------------------- #
# the default is run 1
# --------------------------------------------------------------------------- #


def test_default_target_is_run_1s_r_i_exactly():
    """No `target` given -> `r_i`, the gated share-of-M target, bit for bit, and the
    loop needs no LOO (`needs_loo` False)."""
    cfg = RSRConfig(
        nu=0.0,
        beta=1.0,
        gamma=0.9,
        t_warm=0.0,
        a_max=M,
        b_enabled=False,
        shadow_enabled=False,
    )
    assert cfg.target == LT.TARGET_R_I
    assert REDUCTION_SWITCHES["target"] == LT.TARGET_R_I
    seen = []

    class Rec(RSRPolicy):
        def observe(self, slots, attn, step):
            seen.append((slots, attn))
            super().observe(slots, attn, step)

    pol = Rec(cfg, d_model=D, generator=torch.Generator().manual_seed(7))
    pol.set_train_step(0)
    assert not pol.needs_loo
    model, _docs4, ids, mask = _world(n=4)
    model.eval()
    run_policy_loop(
        model,
        ids,
        mask,
        torch.full((4,), S),
        pol,
        step_fn=_lm,
        observe=True,
        eval_capture=True,
    )
    obs = iter(o for row in sorted(pol._mc) for o in pol._mc[row])
    by_row_step = {(int(sl.row), int(sl.step)): (sl, at) for sl, at in seen}
    n = 0
    for row in sorted(pol._mc):
        for o in pol._mc[row]:
            sl, at = by_row_step[(row, o.step)]
            assert at.loo is None
            want = retrieval_demand(
                at, n_live=sl.n_live, capacity=sl.capacity, gated=True
            )
            assert torch.equal(o.r, want[sl.live.nonzero(as_tuple=True)[0]])
            n += 1
    assert n > 0
    del obs


def test_loo_target_refuses_a_trace_without_loo():
    model, _docs4, ids, mask = _world(n=4)
    pol = _policy(t_warm=0.0)
    with pytest.raises(ValueError, match="carries no LOO"):
        run_policy_loop(
            model,
            ids,
            mask,
            torch.full((4,), S),
            pol,
            step_fn=_lm,
            observe=True,
            eval_capture=True,
        )


def test_loop_refuses_loo_without_eval_capture_and_observe():
    model, docs, ids, mask = _world(n=4)
    tgt = LT.LooTarget(ann=loo.stream_annotations(docs, steps=S), seed=0)
    with pytest.raises(ValueError, match="needs eval_capture"):
        run_policy_loop(
            model,
            ids,
            mask,
            torch.full((4,), S),
            _policy(),
            step_fn=_lm,
            observe=True,
            loo=tgt,
        )


def test_unknown_target_is_refused():
    with pytest.raises(ValueError, match="not a known retention target"):
        RSRPolicy(_cfg(target="loo_zero"), d_model=D)


def test_target_override_is_in_the_label_and_not_in_run_1s():
    run1 = {
        "b_enabled": False,
        "shadow_enabled": False,
        "nu": 0.0,
        "beta": 1.0,
        "gamma": 0.9,
    }
    a = loop.arm_label("rsr", run1)
    b = loop.arm_label("rsr", {**run1, "target": "loo_resample"})
    assert a == "rsr[b=off,beta=1,gamma=0.9,nu=0,shadow=off]"  # run 1's, unchanged
    assert b == "rsr[b=off,beta=1,gamma=0.9,nu=0,shadow=off,target=loo_resample]"
    assert loop.parse_overrides(["target=loo_resample"]) == {"target": "loo_resample"}
