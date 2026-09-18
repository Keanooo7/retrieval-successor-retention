"""The PyTorch TG against golden tensors from the pinned JAX reference.

**New in Sprint 1; not in the spec.** It exists because Task 0's answer was not
the one the kickoff anticipated. See ADR-0001 and ADR-0002, and
`docs/spec-corrections.md` correction 14.

The kickoff planned to vendor TG and "reproduce its reported numbers (29.8 test
PPL, 21 sentence-steps/sec)" as a smoke test. The released code's own README says
it *"differs slightly from the version of the model described in"* the paper, so
that smoke test would be measuring a difference nobody has characterized. This
harness replaces it.

## Protocol

- Golden tensors are extracted **once**, from the pinned JAX reference, by
  `scripts/extract_golden_tensors.py` **in a throwaway venv on CPU**. JAX is not a
  project dependency and never enters `pyproject.toml`; `jaxlib` ships CPU wheels
  for `macosx_11_0_arm64`, and CPU is the right target anyway because the fixtures
  must be byte-reproducible -- ADR-0001 D3's own argument for E0b.
- Fixed seed, fixed batch, `d = 128`, 20 sentence steps.
- Fixtures cover per-layer activations, gestalt vectors, cross-attention weights,
  logits, **and gradients w.r.t. `W_sent` and the transformer parameters**.
- **The tolerance was committed in ADR-0002 BEFORE the fixtures were generated**,
  and `tests/test_tolerances.py` asserts that ordering against `git log`.

## Why the gradient fixtures are mandatory

Gestalts are appended **without detaching the computation graph** ([P2], spec
section 3.1), and backward depth is bounded by stream length `S` rather than by
memory capacity `M` -- evicting a slot does not free its graph (section 3.6,
[P2] App. A).

**JAX functional autodiff and PyTorch retained-graph semantics diverge exactly
there.** A transcription that matches on every forward quantity while retaining
the wrong graph will pass a forward-only fidelity check and survive to week 7,
where it surfaces as a gradient-flow difference in E3 that looks like a finding.
Forward agreement is necessary and not sufficient.

**The fixture is built to make that reachable.** `M = 8` with 20 steps, so **12 of
the 20 steps evict** -- at the reference's `M = 40` the memory would never fill and
the eviction path, which is the path that diverges, would never be exercised.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
NPZ = FIXTURES / "tg_d128_seed0.npz"
META = FIXTURES / "tg_d128_seed0.json"


@pytest.fixture(scope="module")
def golden():
    if not NPZ.exists():
        pytest.skip(f"{NPZ} is absent; run scripts/extract_golden_tensors.py")
    return np.load(NPZ)


@pytest.fixture(scope="module")
def meta():
    if not META.exists():
        pytest.skip(f"{META} is absent")
    return json.loads(META.read_text())


# --------------------------------------------------------------------------- #
# The fixture itself. These run today: they check that the extraction captured
# what it claims to have captured, which is a separate question from whether the
# transcription reproduces it.
# --------------------------------------------------------------------------- #


def test_golden_fixtures_are_present():
    assert NPZ.exists(), "run scripts/extract_golden_tensors.py (see its docstring)"
    assert META.exists(), "the sidecar records jax version, device and the TG pin"


def test_the_fixture_records_its_own_provenance(meta):
    """Section 12.4: documented configuration is not evidence of what was run."""
    assert meta["tg_pin"] == "f220b1098d24a02c94907043d6205c113b31ebb6"
    assert meta["device"].lower().startswith("tfrt_cpu"), meta["device"]
    assert meta["config"]["deterministic"] is True
    assert meta["jax"] and meta["python"] and meta["platform"]


def test_the_fixture_matches_its_recorded_sha256(meta):
    """The sidecar's digest is what makes "byte-reproducible" checkable later."""
    import hashlib

    assert hashlib.sha256(NPZ.read_bytes()).hexdigest() == meta["sha256"]


def test_the_memory_actually_fills_so_eviction_is_exercised(meta, golden):
    """🔴 The single most important property of this fixture.

    At the reference's `M = 40` a 20-step extraction never fills memory, so
    `push_memory` never takes its roll-and-evict branch -- and the gradient path
    through eviction, which is exactly where JAX and PyTorch diverge, would be
    absent from the fixtures while the fixtures looked complete.
    """
    assert meta["config"]["M"] == 8
    assert meta["steps_with_full_memory"] == 12
    valid = golden["memory_valid"]  # [T, B, M]
    assert valid[0].sum() == 0, "memory should start empty"
    assert valid[-1].all(), "memory should be full by the last step"


def test_gestalts_are_unit_norm(golden):
    """D-B / correction 15, verified numerically on the reference's own output:
    `srep_norm_target = 1.0` and the head does a genuine full-vector L2 normalize,
    so coordinates are O(1/sqrt(d)), NOT Theta(1) as section 4.3 assumed."""
    norms = np.linalg.norm(golden["gestalts"], axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-6), (norms.min(), norms.max())


def test_the_per_layer_profile_has_twelve_blocks_and_six_cross_attentions(golden):
    """D-E: cross-attention is on half the blocks. A twelve-entry `r_i` profile
    would be six real rows and six zeros, and the zeros read as a depth finding."""
    activations = [k for k in golden.files if k.startswith("activations/")]
    cross = [k for k in golden.files if k.startswith("cross_attention/")]
    assert len(activations) == 12, sorted(activations)
    assert len(cross) == 6, sorted(cross)


def test_cross_attention_rows_are_a_simplex_over_live_slots(golden):
    """Captured through the `nn.Dropout` submodule, which at `deterministic=True`
    is the identity on the softmax -- so these must still sum to 1."""
    key = sorted(k for k in golden.files if k.startswith("cross_attention/"))[0]
    att = golden[key]  # [T, B, H, L, M]
    sums = att.sum(axis=-1)
    assert np.allclose(sums[1:], 1.0, atol=1e-5), sums[1:].min()


def test_gradient_fixtures_cover_w_sent_and_the_transformer(golden):
    """Not optional. See the module docstring: this is where JAX functional
    autodiff and PyTorch retained-graph semantics diverge."""
    grads = [k for k in golden.files if k.startswith("grad/")]
    assert any("srep_head" in k and "proj" in k for k in grads), "W_sent gradient missing"
    assert any("embed" in k for k in grads)
    assert sum("blocks_" in k for k in grads) > 100
    assert len(grads) == 206


def test_no_gradient_is_all_zero(golden):
    """An all-zero gradient is what a severed graph looks like, and it is the
    failure the retained-graph argument predicts. Checked on the fixture so the
    reference side is known good before the transcription is blamed."""
    zeros = [k for k in golden.files if k.startswith("grad/") and not np.any(golden[k])]
    assert not zeros, f"all-zero gradients in the reference extraction: {zeros}"


def test_the_fixture_carries_its_own_inputs(golden):
    """Self-contained: the PyTorch side replays these ids rather than re-deriving
    them from an RNG, so a seeding difference cannot masquerade as a mismatch."""
    for key in ("input_ids", "input_mask", "input_lengths", "loss"):
        assert key in golden.files
    assert golden["input_ids"].shape == (2, 20, 16)


def test_the_fixture_carries_the_reference_parameters(golden):
    """The PyTorch model is loaded FROM these, so the comparison is of
    computation, not of initialization."""
    params = [k for k in golden.files if k.startswith("param/")]
    assert len(params) == 206
    total = sum(golden[k].size for k in params)
    assert total == 2_478_278


def test_the_positive_control_shows_the_tolerance_can_see_a_severed_graph(meta):
    """Gauntlet 2.3: "a deliberately detached variant must **fail** the gradient
    check."

    This half is a control on the **tolerance**. The extraction reruns everything
    with `detach_sreps_for_memory=True` -- the reference's own ablation, whose
    comment reads *"MUST stay False for the recurrence to train"* -- and records
    what that does.

    🔴 **The forward loss is bit-identical.** `125.3105468750` either way. Only the
    backward pass changes. That is the failure mode ADR-0002 is built around, made
    numerical: a transcription with the wrong graph retention matches every forward
    quantity, passes a forward-only check, and surfaces in week 7 as a
    gradient-flow difference in E3 that looks like a finding.

    Without this, "the gradients match" would be compatible with "the gradients
    are insensitive to the thing being checked."
    """
    control = meta["detach_positive_control"]
    assert control["forward_identical"] is True
    assert control["loss_retained"] == control["loss_detached"]
    assert control["tolerance"] == {"rtol": 1e-3, "atol": 1e-4}
    # 113 of 206 on the recorded run. The floor is a quarter, so the assertion is
    # about the tolerance being able to see it, not about the exact number.
    assert control["arrays_outside_gradient_tolerance"] >= control["gradient_arrays"] // 4
    assert control["worst_relative_delta"] > 1.0, control["worst_relative_delta"]


# --------------------------------------------------------------------------- #
# The comparison itself (gauntlet 2.4). PyTorch TG vs the golden tensors, to
# ADR-0002's committed tolerance.
# --------------------------------------------------------------------------- #

import dataclasses  # noqa: E402

import torch  # noqa: E402

from rsr.model.tg import (  # noqa: E402
    TGConfig,
    TGModel,
    load_reference_params,
    run_sentence_loop,
)
from rsr.model.tg.tolerances import FORWARD, GRADIENT  # noqa: E402


def _next_token_loss(logits, ids, mask, row_valid):
    """The same loss the extraction used. Mean over valid positions, per step."""
    logp = torch.log_softmax(logits[:, :-1].to(torch.float32), dim=-1)
    tgt = ids[:, 1:]
    picked = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    valid = (mask[:, 1:] == 1) & row_valid.unsqueeze(-1)
    return -(picked * valid).sum() / valid.sum().clamp(min=1)


def _build(meta, golden, *, detach=False):
    cfg = TGConfig.from_reference_dict(meta["config"])
    if detach:
        cfg = dataclasses.replace(cfg, detach_sreps_for_memory=True)
    model = TGModel(cfg)
    load_reference_params(model, {k: golden[k] for k in golden.files})
    model.eval()  # ADR-0002: dropout deterministic, srep_dropout disabled
    return cfg, model


def _run(model, golden, *, capture=True):
    ids = torch.as_tensor(golden["input_ids"], dtype=torch.long)
    mask = torch.as_tensor(golden["input_mask"], dtype=torch.long)
    lengths = torch.as_tensor(golden["input_lengths"], dtype=torch.long)
    return run_sentence_loop(
        model,
        ids,
        mask,
        lengths,
        step_fn=lambda t, out, i, m, rv: _next_token_loss(out.logits, i, m, rv),
        capture=capture,
    )


@pytest.fixture(scope="module")
def transcription(meta, golden):
    cfg, model = _build(meta, golden)
    loss, steps = _run(model, golden)
    return cfg, model, loss, steps


def _assert_close(got, want, tol, what):
    torch.testing.assert_close(
        got,
        torch.as_tensor(np.asarray(want), dtype=got.dtype),
        rtol=tol.rtol,
        atol=tol.atol,
        msg=lambda m: (
            f"{what}: {m}\n\ntolerance is ADR-0002's, committed before "
            f"these fixtures existed. If it cannot be met, ADR-0002 records the "
            f"ACHIEVED value and the cause. It does not silently relax."
        ),
    )


def test_the_loss_matches_the_jax_reference(transcription, golden):
    """The single number that has to agree before anything else is diagnosable."""
    _, _, loss, _ = transcription
    _assert_close(loss, golden["loss"], FORWARD, "total loss")


def test_forward_matches_jax_reference(transcription, golden):
    """Per-layer activations, all 12 blocks, all 20 steps."""
    _, _, _, steps = transcription
    keys = sorted(
        (k for k in golden.files if k.startswith("activations/")),
        key=lambda k: int(k.split("blocks_")[1].split("/")[0]),
    )
    assert len(keys) == 12
    for layer, key in enumerate(keys):
        want = golden[key]  # [T, B, L, D]
        got = torch.stack([s.activations[layer] for s in steps])
        _assert_close(got, want, FORWARD, f"activations, block {layer}")


def test_gestalt_vectors_match_jax_reference(transcription, golden):
    _, _, _, steps = transcription
    got = torch.stack([s.srep for s in steps])
    _assert_close(got, golden["gestalts"], FORWARD, "gestalts")
    assert torch.allclose(got.norm(dim=-1), torch.ones(()), atol=1e-6)


def test_cross_attention_weights_match_jax_reference(transcription, golden):
    """Six layers, not twelve (D-E). Each `[T, B, H, L, M]`."""
    _, _, _, steps = transcription
    keys = sorted(
        (k for k in golden.files if k.startswith("cross_attention/")),
        key=lambda k: int(k.split("blocks_")[1].split("/")[0]),
    )
    assert len(keys) == 6
    for layer, key in enumerate(keys):
        got = torch.stack([s.cross_attention[layer] for s in steps])
        _assert_close(got, golden[key], FORWARD, f"cross-attention, layer {layer}")


def test_logits_match_jax_reference(transcription, golden):
    _, _, _, steps = transcription
    got = torch.stack([s.logits for s in steps])
    _assert_close(got, golden["logits"], FORWARD, "logits")


# --- gradients -------------------------------------------------------------- #


@pytest.fixture(scope="module")
def gradients(meta, golden):
    _, model = _build(meta, golden)
    loss = _run(model, golden, capture=False)
    loss.backward()
    return {name: p.grad for name, p in model.named_parameters()}


def _reference_grads(golden):
    from rsr.model.tg.loading import reference_name_to_torch

    return {
        reference_name_to_torch(k[len("grad/") :]): golden[k]
        for k in golden.files
        if k.startswith("grad/")
    }


def test_gradients_wrt_w_sent_match_jax_reference(gradients, golden):
    """Not optional. See the module docstring: this is where JAX functional
    autodiff and PyTorch retained-graph semantics diverge."""
    want = _reference_grads(golden)
    for name in ("srep_head.proj.kernel", "srep_head.proj.bias"):
        assert gradients[name] is not None, f"{name} received no gradient"
        _assert_close(gradients[name], want[name], GRADIENT, name)


def test_gradients_wrt_transformer_params_match_jax_reference(gradients, golden):
    want = _reference_grads(golden)
    assert len(want) == 206
    for name, reference in want.items():
        assert gradients[name] is not None, f"{name} received no gradient"
        _assert_close(gradients[name], reference, GRADIENT, name)


def test_the_memory_gate_receives_gradient(gradients):
    """D-E's parameter, and the one the positive control hits hardest."""
    gates = {n: g for n, g in gradients.items() if n.endswith("memory_gate")}
    assert len(gates) == 6, sorted(gates)  # six cross-attention layers
    assert all(g is not None and g.abs() > 0 for g in gates.values())


def test_a_detached_pytorch_model_fails_the_gradient_check(meta, golden):
    """🔴 Gauntlet 2.3's positive control, on the transcription.

    A PyTorch TG built with `detach_sreps_for_memory=True` must **fail** against
    these fixtures. If it passed, the gradient check would be measuring nothing
    about graph retention -- and a forward-only match with the wrong graph is the
    failure that survives to week 7 and looks like a finding in E3.

    The forward pass is expected to still match exactly: detaching changes only
    the backward. That is asserted here too, because it is what makes the control
    a control rather than a broken model.
    """
    _, model = _build(meta, golden, detach=True)
    loss = _run(model, golden, capture=False)
    _assert_close(loss, golden["loss"], FORWARD, "detached forward loss")

    loss.backward()
    want = _reference_grads(golden)
    mismatched = [
        name
        for name, p in model.named_parameters()
        if p.grad is not None
        and not torch.allclose(
            p.grad,
            torch.as_tensor(np.asarray(want[name]), dtype=p.grad.dtype),
            rtol=GRADIENT.rtol,
            atol=GRADIENT.atol,
        )
    ]
    assert len(mismatched) >= len(want) // 4, (
        f"only {len(mismatched)} of {len(want)} gradients moved when the "
        f"gestalt->memory graph was severed. The gradient check is not testing "
        f"graph retention."
    )
