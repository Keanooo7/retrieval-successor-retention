"""E0b -- reduction to exact TG (spec section 3.7). **Kill gate.**

Set `psi_hat == -a_i`, `b == 0`, `nu = 0`, `beta = 0`, `A_max = M`, `T_warm = 0`,
shadow buffer off. Eviction becomes `argmin(-a_i)` = the oldest slot. **Bit-exact
TG.**

Verified numerically, not argued.

---

## What this test is for, and what it is not for

**It is intra-repo by design.** It compares RSR-under-the-section-3.7-switches
against *this repository's* PyTorch TG -- not against the published model, and not
against the JAX reference. That is deliberate and it is what section 3.7 actually
needs: the guarantee that **E3's FIFO and RSR arms differ only in the eviction
rule**, so that any difference between them is attributable to retention and
nothing else.

External fidelity is a separate claim with a separate test. See
`tests/test_fidelity.py` and ADR-0001.

## Gauntlet 0.1 -- `T_warm` was `inf`, and that made this gate vacuous

Dispatch is `t < T_warm -> FIFO`. `reduction_to_tg()` set `T_warm = inf`, so under
the reduction **no eviction ever reached the score** and E0b certified FIFO against
FIFO. The spec's own §3.7 says "warmup irrelevant (`T_warm = ∞`)", and it is
irrelevant *to the outcome* -- but only because `psi_hat == -a_i` already selects
the oldest slot. Setting it to `inf` bypasses the thing being reduced. It is now
`0.0`, so the reduction runs **through the score path**, and
`test_reduction_runs_through_the_score_path` asserts exactly that. See
`docs/spec-corrections.md` correction 16.

## Gauntlet 0.6 -- the off-switch test was skipped under a reason false for it

`pytestmark` skipped the whole module for want of the PyTorch TG. The off-switch
test needs no TG and would have passed on day one, so **the §3.7 off-switch
contract was enforced by nothing that runs.** The skip is now on the one test that
genuinely needs the transcription.

## Why every added term needs an off-switch

`RSRConfig.REDUCTION_SWITCHES` is the single enumeration, and
`test_every_config_field_has_an_off_switch` asserts it covers every field of
`RSRConfig`. **Add a term with no entry and that test reddens** -- which is the
mechanical form of "every term added to the eviction score must have a documented
off-switch, or section 3.7 silently stops being a reduction to TG and this test
stops testing what it claims to test."

## The RNG trap -- read this before debugging a failure here

**Construct `phi` after the model, or seed it from a separate RNG stream.**

Otherwise instantiating the value head consumes draws from the global generator,
which shifts data order and dropout masks, and **this test fails for a reason that
has nothing to do with the mechanism.** Section 3.7 says so explicitly, and it is
written down here so that nobody spends a day debugging it as a mechanism failure.

`BilinearValueHead(d, generator=...)` is the safe route;
`tests/test_value_head.py` pins both halves of that behaviour.
"""

from __future__ import annotations

from dataclasses import fields, replace

import pytest
import torch

from rsr.baselines.fifo import FIFOPolicy
from rsr.retention.policy import MemoryState
from rsr.retention.rsr import (
    CAPACITY,
    REDUCTION_SWITCHES,
    ContextSource,
    RSRConfig,
    RSRPolicy,
    SentPosIndex,
)

M = 8


def _state(written, step, d=4, generator=None):
    return MemoryState(
        gestalts=torch.randn(len(written), d, generator=generator),
        written_at=torch.tensor(written, dtype=torch.long),
        live=torch.ones(len(written), dtype=torch.bool),
        step=step,
    )


def _run(policy, n=100, d=4, seed=0):
    """Drive a policy through `n` evictions and return the victim sequence."""
    g = torch.Generator().manual_seed(seed)
    written, victims = list(range(M)), []
    for step in range(M, M + n):
        s = _state(written, step, d=d, generator=g)
        v = policy.select_eviction(s, torch.randn(d, generator=g), step)
        victims.append(v)
        written[v] = step
        policy.on_write(s, v, step)
    return victims


# --------------------------------------------------------------------------- #
# The off-switch contract. No TG required -- gauntlet 0.6.
# --------------------------------------------------------------------------- #


def test_reduction_config_disables_every_added_term():
    """The section 3.7 switch set, pinned so a new term cannot be added silently."""
    cfg = RSRConfig.reduction_to_tg(capacity=40)
    assert cfg.psi_override == "neg_age"
    assert cfg.b_enabled is False
    assert cfg.nu == 0.0
    assert cfg.beta == 0.0
    assert cfg.gamma == 0.0
    assert cfg.shadow_enabled is False
    assert cfg.a_max == 40


def test_every_config_field_has_an_off_switch():
    """**The mutation target.** Add a term to `RSRConfig` without registering its
    off-switch in `REDUCTION_SWITCHES` and this reddens -- and nothing else does.

    Without it, section 3.7 stops being a reduction the moment someone adds a term,
    and E0b keeps passing while testing a different model.
    """
    assert {f.name for f in fields(RSRConfig)} == set(REDUCTION_SWITCHES), (
        "every field of RSRConfig must have an entry in REDUCTION_SWITCHES "
        "(spec section 3.7). A term with no documented off-switch means the "
        "reduction is no longer a reduction."
    )


def test_t_warm_is_zero_not_infinity():
    """Gauntlet 0.1. `inf` routed every eviction to FIFO, so the reduction never
    executed the score path and E0b certified FIFO against FIFO."""
    assert RSRConfig.reduction_to_tg(capacity=M).t_warm == 0.0


def test_the_reduction_pins_the_positional_index_to_rank():
    """ADR-0006 / D-D. `absolute_age` changes the base model, so under it E0b
    would test something the fidelity harness never saw."""
    assert REDUCTION_SWITCHES["sent_pos_index"] is SentPosIndex.RANK
    assert RSRConfig.reduction_to_tg(capacity=M).sent_pos_index is SentPosIndex.RANK


def test_the_reduction_pins_the_context_source():
    """D-C. `c_t` is the current sentence gestalt."""
    assert (
        RSRConfig.reduction_to_tg(capacity=M).context_source
        is ContextSource.CURRENT_SENTENCE_GESTALT
    )


def test_a_max_tracks_capacity():
    assert REDUCTION_SWITCHES["a_max"] is CAPACITY
    assert RSRConfig.reduction_to_tg(capacity=40).a_max == 40


def test_is_reduction_detects_a_single_flipped_switch():
    """Each switch on its own must break the reduction.

    The brief records a real hole of exactly this shape on the MacBook repo:
    dropping the `nu` clause from `is_reduction` reddened *nothing*, which would
    have let a policy with `nu = 0.5` report itself as stock TG.
    """
    base = RSRConfig.reduction_to_tg(capacity=M)
    assert base.is_reduction(M)
    flips = {
        "psi_override": None,
        "b_enabled": True,
        "nu": 0.5,
        "beta": 1.0,
        "gamma": 0.9,
        "t_warm": float("inf"),
        "shadow_enabled": True,
        "a_max": M + 1,
        "context_source": ContextSource.RUNNING_CONTEXT,
        "sent_pos_index": SentPosIndex.ABSOLUTE_AGE,
    }
    assert set(flips) == set(REDUCTION_SWITCHES), "a switch has no flip case here"
    for name, value in flips.items():
        assert not replace(base, **{name: value}).is_reduction(M), (
            f"flipping {name} left is_reduction() True"
        )


# --------------------------------------------------------------------------- #
# The reduction, through the score path. Needs no TG either -- the eviction rule
# is what §3.7 reduces, and it is testable against FIFO directly.
# --------------------------------------------------------------------------- #


def test_reduction_runs_through_the_score_path():
    """Gauntlet 0.1's PASS. Not one eviction may be attributed to FIFO warmup."""
    p = RSRPolicy(RSRConfig.reduction_to_tg(capacity=M), d_model=4)
    _run(p)
    counts = p.attribution_counts()
    assert counts == {"neg_age": 100}, counts


def test_reduction_agrees_with_fifo_on_every_eviction():
    """`argmin(-a_i)` is the oldest slot. Identical sequences, not 'close'."""
    rsr = _run(RSRPolicy(RSRConfig.reduction_to_tg(capacity=M), d_model=4))
    fifo = _run(FIFOPolicy())
    assert rsr == fifo


def test_switching_psi_source_to_the_learned_head_reddens_the_reduction():
    """**The mutation.** If the learned head also matched FIFO, the gate above
    would be measuring nothing."""
    mutated = replace(RSRConfig.reduction_to_tg(capacity=M), psi_override=None)
    p = RSRPolicy(mutated, d_model=4, generator=torch.Generator().manual_seed(1))
    assert _run(p) != _run(FIFOPolicy())
    assert p.attribution_counts() == {"psi": 100}


def test_the_reduction_shifts_no_ranks():
    """ADR-0006: displacement is identically 0 under FIFO, which is what makes it
    a measure of the confound rather than of the sliding window. It is also why
    E0b structurally cannot see the confound -- under the reduction the policy
    *is* FIFO."""
    p = RSRPolicy(RSRConfig.reduction_to_tg(capacity=M), d_model=4)
    _run(p)
    assert p.rank_shifts.distribution() == {0: 100}
    assert p.rank_shifts.fraction_shifting == 0.0


def test_the_learned_head_does_shift_ranks():
    """The other half: the confound is real in the arm that matters."""
    mutated = replace(RSRConfig.reduction_to_tg(capacity=M), psi_override=None)
    p = RSRPolicy(mutated, d_model=4, generator=torch.Generator().manual_seed(1))
    _run(p)
    assert p.rank_shifts.fraction_shifting > 0.5


# --------------------------------------------------------------------------- #
# The gate itself (gauntlet 2.6): bit-exact against the PyTorch TG.
# --------------------------------------------------------------------------- #

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from rsr.model.tg import (  # noqa: E402
    TGConfig,
    TGModel,
    load_reference_params,
    run_sentence_loop,
)
from rsr.model.tg.policy_loop import run_policy_loop  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


def _loss(t, out, ids, mask, row_valid):
    logp = torch.log_softmax(out.logits[:, :-1].to(torch.float32), dim=-1)
    picked = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    valid = (mask[:, 1:] == 1) & row_valid.unsqueeze(-1)
    return -(picked * valid).sum() / valid.sum().clamp(min=1)


@pytest.fixture(scope="module")
def tg():
    """The PyTorch TG at the fixture's configuration, with reference weights.

    Real weights rather than a fresh init: a reduction that holds only at
    initialisation would be a statement about small numbers, not about the
    eviction rule.
    """
    npz, sidecar = FIXTURES / "tg_d128_seed0.npz", FIXTURES / "tg_d128_seed0.json"
    if not npz.exists():
        pytest.skip("golden fixtures absent; run scripts/extract_golden_tensors.py")
    golden = np.load(npz)
    cfg = TGConfig.from_reference_dict(json.loads(sidecar.read_text())["config"])
    model = TGModel(cfg)
    load_reference_params(model, {k: golden[k] for k in golden.files})
    model.eval()
    ids = torch.as_tensor(golden["input_ids"], dtype=torch.long)
    mask = torch.as_tensor(golden["input_mask"], dtype=torch.long)
    lengths = torch.as_tensor(golden["input_lengths"], dtype=torch.long)
    return cfg, model, ids, mask, lengths


def _stock(tg):
    """Stock TG: the transcription's own loop, whose `push_memory` evicts the
    oldest slot. This is the arm E3 calls FIFO."""
    _, model, ids, mask, lengths = tg
    return run_sentence_loop(model, ids, mask, lengths, step_fn=_loss)


def _under_policy(tg, policy):
    _, model, ids, mask, lengths = tg
    return run_policy_loop(model, ids, mask, lengths, policy, step_fn=_loss)


def test_the_fixture_configuration_actually_fills_memory(tg):
    """A reduction test on a memory that never fills tests nothing: no eviction
    ever happens, so the eviction rule is never consulted. `M = 8` over 20 steps
    gives 12 evictions."""
    cfg, _, _, _, lengths = tg
    assert int(lengths.max()) > cfg.M


def test_loss_curve_is_bit_exact_against_stock_tg(tg):
    """🔴 **The gate.** Identical loss, not 'close'.

    Under §3.7's switches `psi_hat == -a_i`, so the argmin selects the oldest slot
    and `write_at` reduces to `push_memory`'s roll. The two loops then differ in
    no operation at all, and the losses must agree to the last bit -- not to a
    tolerance, which is what `test_fidelity.py` is for.
    """
    cfg, _, _, _, _ = tg
    policy = RSRPolicy(RSRConfig.reduction_to_tg(capacity=cfg.M), d_model=cfg.D)
    stock, reduced = _stock(tg), _under_policy(tg, policy)
    assert reduced.item() == stock.item(), (
        f"stock {stock.item()!r} vs reduction {reduced.item()!r}; "
        f"delta {abs(reduced.item() - stock.item()):.3e}"
    )


def test_the_reduction_ran_through_the_score_path(tg):
    """Gauntlet 0.1 again, now on the real model: every eviction attributed to the
    score, none to FIFO warmup. `T_warm = inf` used to make this all warmup."""
    cfg, _, _, _, _ = tg
    policy = RSRPolicy(RSRConfig.reduction_to_tg(capacity=cfg.M), d_model=cfg.D)
    _under_policy(tg, policy)
    counts = policy.attribution_counts()
    assert counts.get("fifo_warmup", 0) == 0, counts
    assert counts["neg_age"] == 24, counts  # 12 evicting steps x 2 rows


def test_the_reduction_displaces_no_ranks_on_the_real_model(tg):
    """ADR-0006. Under the reduction the policy *is* FIFO, so displacement is
    identically 0 -- which is exactly why E0b cannot see the P^(sent) confound and
    why the instrumentation exists separately."""
    cfg, _, _, _, _ = tg
    policy = RSRPolicy(RSRConfig.reduction_to_tg(capacity=cfg.M), d_model=cfg.D)
    _under_policy(tg, policy)
    assert policy.rank_shifts.distribution() == {0: 24}


def test_fifo_policy_also_reproduces_stock_tg(tg):
    """The other direction: the explicit FIFO baseline must agree too.

    If it did not, the divergence would be in `write_at` rather than in the score,
    and the gate above would be passing for the wrong reason.
    """
    assert _under_policy(tg, FIFOPolicy()).item() == _stock(tg).item()


def test_a_learned_head_changes_the_loss(tg):
    """**The mutation.** If the learned head also matched stock TG bit for bit, the
    gate would be measuring the harness, not the eviction rule."""
    cfg, _, _, _, _ = tg
    mutated = replace(RSRConfig.reduction_to_tg(capacity=cfg.M), psi_override=None)
    policy = RSRPolicy(mutated, d_model=cfg.D, generator=torch.Generator().manual_seed(1))
    assert _under_policy(tg, policy).item() != _stock(tg).item()


def test_the_learned_head_does_evict_middle_slots(tg):
    """And the mutation is the interesting kind: it displaces ranks, which is the
    confound ADR-0006 instruments and E0b structurally cannot see."""
    cfg, _, _, _, _ = tg
    mutated = replace(RSRConfig.reduction_to_tg(capacity=cfg.M), psi_override=None)
    policy = RSRPolicy(mutated, d_model=cfg.D, generator=torch.Generator().manual_seed(1))
    _under_policy(tg, policy)
    assert policy.rank_shifts.fraction_shifting > 0.0


def test_phi_construction_does_not_perturb_the_global_rng(tg):
    """The RNG trap, on the real model. Constructing `phi` from the global
    generator consumes draws, shifts data order and dropout masks, and fails this
    file for a reason unrelated to the mechanism (§3.7)."""
    cfg, _, _, _, _ = tg
    torch.manual_seed(0)
    baseline = torch.randn(4)
    torch.manual_seed(0)
    RSRPolicy(
        replace(RSRConfig.reduction_to_tg(capacity=cfg.M), psi_override=None),
        d_model=cfg.D,
        generator=torch.Generator().manual_seed(7),
    )
    assert torch.equal(baseline, torch.randn(4))
