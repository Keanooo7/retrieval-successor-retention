"""The shuffle control has to be seen reading both ways before its null means anything.

`rsr.metrics.memory_liveness.shuffle_control` exists to say *"this model does not
use its memory"* -- a null. The RSR manager rejected cycle 1's use of it because
**an instrument never observed to read non-zero cannot report an absence**: a
permutation that never applies, a replay that never reaches the forward, an eval
path that skips memory all read exactly the `~0` a dead memory does.

So three readings are pinned here, on a tiny randomly initialised model where the
answer is known by construction:

* **live memory -> non-zero.** At init every row's memory holds its own
  document's gestalts and `memory_gate` is 1.0, so another row's memory changes
  the logits. If the instrument no-ops, this goes red.
* **memory disabled -> exactly 0.0.** Zeroing `memory_gate` multiplies the
  cross-attention output by 0. If this reads non-zero, the delta is coming from
  somewhere other than memory.
* **identity permutation -> exactly 0.0.** Replaying each row's *own* snapshots
  reproduces the honest pass bit for bit, so the live reading above is caused by
  *whose* memory it is, not by replay drift.

Mutation this file answers (`scripts/mutation_battery.py`): make `derangement`
return the identity. Only this file goes red.
"""

from __future__ import annotations

import torch

from rsr.data.synthetic import SyntheticConfig, generate
from rsr.metrics.memory_liveness import (
    derangement,
    memory_gates,
    shuffle_control,
    with_memory_disabled,
)
from rsr.model.tg import TGConfig, TGModel
from rsr.train.loop import build_vocab, encode

S, L, M, N_DOCS = 6, 16, 4, 4


def _batch():
    # The generator needs `max_gap < sentences_per_document`; encode only the
    # first `S` sentences of each document.
    docs = generate(
        SyntheticConfig(n_documents=N_DOCS, sentences_per_document=48, seed=0)
    )
    vmap = build_vocab(docs)
    ids, mask = encode(docs, vmap, max_tokens=L, steps=S)
    return ids[:N_DOCS], mask[:N_DOCS], 4 + len(vmap)


def _model(V: int) -> TGModel:
    torch.manual_seed(0)
    cfg = TGConfig(
        D=32,
        H=2,
        V=V,
        max_sentence_tokens=L,
        max_sentences_in_short_term=M,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        eod_id=3,
    )
    return TGModel(cfg)


def test_derangement_gives_every_row_another_rows_memory():
    for n in range(2, 9):
        p = derangement(n)
        assert sorted(p.tolist()) == list(range(n))
        assert all(int(p[i]) != i for i in range(n)), p.tolist()


def test_live_memory_moves_the_loss():
    ids, mask, V = _batch()
    r = shuffle_control(_model(V), ids, mask)

    # The premise: memory was actually written and is full, so the reading is
    # about a live memory rather than an empty one.
    assert r["n_sentences_with_eos"] == r["n_sentences"] == N_DOCS * S
    assert r["final_slots_filled_per_row"] == [M] * N_DOCS
    assert r["memory_gates"] and all(g == 1.0 for g in r["memory_gates"])

    assert not r["delta_exactly_zero"]
    assert r["n_tokens_moved"] > 0
    assert max(abs(r["max_token_delta"]), abs(r["min_token_delta"])) > 1e-4
    assert r["mean_abs_token_delta"] > 0.0


def test_disabled_memory_reads_exactly_zero():
    ids, mask, V = _batch()
    model = _model(V)
    r = shuffle_control(with_memory_disabled(model), ids, mask)

    assert r["memory_gates"] and all(g == 0.0 for g in r["memory_gates"])
    assert r["delta_exactly_zero"], r["delta_nats_per_token"]
    assert r["n_tokens_moved"] == 0
    assert r["mean_abs_token_delta"] == 0.0
    # The copy is what was disabled, not the model handed in.
    assert all(g == 1.0 for g in memory_gates(model))


def test_replaying_each_rows_own_memory_reads_exactly_zero():
    ids, mask, V = _batch()
    r = shuffle_control(_model(V), ids, mask, perm=torch.arange(N_DOCS))

    assert r["delta_exactly_zero"], r["delta_nats_per_token"]
    assert r["n_tokens_moved"] == 0


def test_the_control_leaves_the_models_mode_as_it_found_it():
    ids, mask, V = _batch()
    model = _model(V)
    model.train()
    shuffle_control(model, ids, mask)
    assert model.training


# --------------------------------------------------------------------------- #
# decisive run additions (experiments/decisive-shuffle/PREREG.md, Secondary 1-2).
# Additive: none of the tests above is touched.
# --------------------------------------------------------------------------- #


def test_random_replacement_with_self_reads_exactly_zero():
    from rsr.metrics.memory_liveness import random_replacement

    ids, mask, V = _batch()
    g = torch.Generator().manual_seed(10_000)
    r = random_replacement(_model(V), ids, mask, generator=g, self_replace=True)
    assert r["replacement"] == "self"
    assert r["delta_exactly_zero"], r["delta_nats_per_token"]
    assert r["n_tokens_moved"] == 0
    assert r["mean_abs_token_delta"] == 0.0


def test_matched_norm_random_replacement_moves_a_live_memory():
    from rsr.metrics.memory_liveness import random_replacement

    ids, mask, V = _batch()
    g = torch.Generator().manual_seed(10_000)
    r = random_replacement(_model(V), ids, mask, generator=g)
    assert r["replacement"] == "matched_norm_gaussian"
    assert r["n_tokens_moved"] > 0
    assert r["mean_abs_token_delta"] > 0.0


def test_random_replacement_preserves_each_slots_norm_and_draws_off_the_model_rng():
    from rsr.metrics import memory_liveness as ml

    ids, mask, V = _batch()
    model = _model(V)
    seen = []
    orig = ml.shuffle_control

    def spy(*a, replace=None, **k):
        def wrapped(kv, valid, t):
            out = replace(kv, valid, t)
            seen.append((kv.norm(dim=-1), out.norm(dim=-1), torch.equal(out, kv)))
            return out

        return orig(*a, replace=wrapped, **k)

    state = torch.get_rng_state()
    ml.shuffle_control = spy
    try:
        ml.random_replacement(model, ids, mask, generator=torch.Generator())
    finally:
        ml.shuffle_control = orig
    assert torch.equal(torch.get_rng_state(), state)  # the model's RNG untouched
    assert seen
    for n_in, n_out, unchanged in seen:
        if bool((n_in > 0).any()):
            assert not unchanged  # a non-empty memory was actually replaced
        assert torch.allclose(n_in, n_out, rtol=1e-5, atol=1e-6)


def test_token_mask_equal_to_the_real_mask_reproduces_the_whole_reading():
    ids, mask, V = _batch()
    r = shuffle_control(_model(V), ids, mask, token_masks={"all": mask.clone()})
    b = r["by_mask"]["all"]
    # `mask[..., 1:]` scores exactly the real targets, so the two agree exactly.
    assert b["n_targets"] == r["n_real_tokens"]
    assert b["mean_abs_token_delta"] == r["mean_abs_token_delta"]
    assert b["loss_honest_memory"] == r["loss_real_tokens_honest_memory"]


def test_token_mask_on_padding_raises_and_default_adds_no_key():
    import pytest

    ids, mask, V = _batch()
    model = _model(V)
    assert "by_mask" not in shuffle_control(model, ids, mask)
    pad = ~mask
    with pytest.raises(ValueError, match="non-real target"):
        shuffle_control(model, ids, mask, token_masks={"pad": pad})
    empty = shuffle_control(model, ids, mask, token_masks={"none": mask & False})
    assert empty["by_mask"]["none"]["n_targets"] == 0
    assert empty["by_mask"]["none"]["mean_abs_token_delta"] is None


def test_cross_row_cosine_is_one_for_identical_rows_and_bounded_otherwise():
    from rsr.metrics.memory_liveness import cross_row_cosine

    ids, mask, V = _batch()
    model = _model(V)
    c = cross_row_cosine(model, ids, mask)
    assert c["n_slot_steps"] > 0
    assert -1.0 <= c["mean_offdiag_cosine"] < 1.0 - 1e-6
    same = cross_row_cosine(model, ids[:1].expand_as(ids), mask[:1].expand_as(mask))
    assert abs(same["mean_offdiag_cosine"] - 1.0) < 1e-9
