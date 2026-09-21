"""S0-01 -- the three `rsr.train.loop` defects, each pinned by a test that was red first.

Brief: `docs/lab-notes/dispatch-S0-01-loop-defects.md` (amended twice, then revalidated
by the manager at `75c0f62`). Measured at `3973d1a`, `loop.py` 401 lines.

The three defects, as the brief predicts them and as they re-measured:

* **(b)** `loop.py:214` constructed `FIFOPolicy()` unconditionally while `policy_name`
  (`:184`, default `"fifo"`) was stamped into the frozen config (`:224`) and the
  `run_id` (`:228`), and `main()`'s flag table (`:364-375`) had no `--policy`. A caller
  writing `train(policy_name="rsr")` got a directory, a heartbeat and a `run_id` all
  reading `rsr-...` over a run in which FIFO evicted every slot. **Silent**: the loss
  curve is a perfectly plausible FIFO curve, because it is one.
* **(c)** `StepOutput.srep_norm_penalty` is computed at `model.py:424` and was dropped
  on the floor -- `grep -c srep_norm src/rsr/train/loop.py` returned **0**.
  `docs/spec-corrections.md` correction 15 item 4 is explicit that *"the hinge penalty
  is in TG's loss"*; §3.1 requires the base model be unmodified, so a loop that omits
  it is not training TG. **Silent**: the penalty is ~0.1 at init and the weight is
  0.01, so the omission moves the loss in the third decimal and nothing screams.
* **(e)** `--vocab` defaulted to `50257` (`:369`) against a corpus of **156** unique
  words. Not silent -- it is a 314x embedding table -- but it made the derived path
  (`V = vocab if vocab else 4 + len(build_vocab(probe))`) unreachable **at the
  default**, which is the smaller and the true claim. `--vocab 0` always reached it.

Two further items the brief listed separately, on purpose, and which are handled
differently here: `value_head=None` into `build_param_groups` is an `xfail(strict=True)`
rather than a speculative fix (nothing in this tree constructs a value head yet), and
`RSRConfig.from_registry`'s eager `kw` build (`rsr.py:236-243`) is fixed, because it was
*measured* to raise through an explicit override rather than merely suspected of it.

🔴 Defect (a) (the masked loss) belongs to `dispatch-cycle-01-masked-loss.md` and defect
(d) (`policy.attribution()` vs `attribution_counts()`) belongs to cycle 4, downstream of
(b). Neither is touched here.
"""

from __future__ import annotations

import pytest
import torch

import rsr.constants as C
from rsr.baselines.fifo import FIFOPolicy
from rsr.data.synthetic import SyntheticConfig, generate
from rsr.retention.rsr import RSRConfig
from rsr.train.loop import build_policy, build_vocab, main, train

# `sentences_per_document` cannot go below the generator's `max_gap=40`
# (`synthetic.py:143`), so 48 is the floor for a real stream, not a taste.
TINY = dict(
    d=32,
    steps_per_stream=48,
    batch=2,
    max_tokens=16,
    memory_slots=4,
    iters=1,
    device="cpu",
    beat_every=1,
    ckpt_every=0,
)


# The CLI is exercised through `main` with `train` monkeypatched, never by rebuilding
# the flag table here -- a test that rebuilds the parser stops testing the shipped one.

# --------------------------------------------------------------------------- #
# (b) -- the policy the run says it used is the policy it used
# --------------------------------------------------------------------------- #


def test_build_policy_fifo_returns_a_fifo_policy():
    assert isinstance(build_policy("fifo", d_model=32, steps_per_epoch=1.0), FIFOPolicy)


def test_build_policy_rsr_is_never_a_fifo_policy():
    """The brief's four-line test: ask for `rsr`, and never receive FIFO.

    On today's empty ledger the honest outcome is a *loud refusal* -- `nu`, `beta` and
    `gamma` are MEASURED and E1 has not run -- not a substitution. Both outcomes pass;
    silently returning FIFO does not. That is the whole defect.
    """
    try:
        policy = build_policy("rsr", d_model=32, steps_per_epoch=1.0)
    except C.UnmeasuredConstant as e:
        assert "E1" in str(e), "the refusal must name the experiment that unblocks it"
        return
    assert not isinstance(policy, FIFOPolicy)
    assert getattr(policy, "name", None) == "rsr"


def test_an_unknown_policy_name_is_refused_rather_than_defaulted():
    """`policy_name="h2o"` must not quietly become FIFO with an `h2o-...` run_id."""
    with pytest.raises(ValueError, match="h2o"):
        build_policy("h2o", d_model=32, steps_per_epoch=1.0)


def test_train_does_not_stamp_a_policy_it_did_not_build(tmp_path):
    """The silent failure itself: a `run_id` reading `rsr-` over a FIFO run.

    Before the fix this returned normally and `r["run_id"].startswith("rsr-")` was
    True while `FIFOPolicy` had evicted every slot.
    """
    with pytest.raises(C.UnmeasuredConstant):
        train(policy_name="rsr", out_dir=tmp_path / "rsr", **TINY)


def test_the_policy_is_selectable_from_the_command_line(monkeypatch, tmp_path):
    """`--policy` exists, reaches `train`, and defaults to `fifo`."""
    seen: dict[str, object] = {}

    def fake_train(**kw):
        seen.update(kw)
        return {"run_id": "x"}

    monkeypatch.setattr("rsr.train.loop.train", fake_train)
    assert main(["--out-dir", str(tmp_path), "--policy", "rsr"]) == 0
    assert seen["policy_name"] == "rsr"

    seen.clear()
    assert main(["--out-dir", str(tmp_path)]) == 0
    assert seen["policy_name"] == "fifo"


# --------------------------------------------------------------------------- #
# (c) -- the srep-norm hinge is in the objective, and has an off-switch
# --------------------------------------------------------------------------- #


def test_the_hinge_reaches_the_objective(tmp_path):
    """With the hinge on, the optimised loss is strictly above the LM loss.

    `loss_srep_hinge` is reported unweighted so the term is auditable independently
    of its weight; `loss` is the quantity actually minimised.
    """
    r = train(out_dir=tmp_path / "on", srep_norm_reg_weight=0.5, seed=0, **TINY)
    f = r["final"]
    assert f["loss_srep_hinge"] > 0.0, (
        "the hinge is zero -- either the raw gestalt norms are already inside "
        "[0.9, 1.1] at init, in which case this test is vacuous and says so, or the "
        "penalty is not being read"
    )
    assert f["loss"] > f["loss_lm"]
    assert f["loss"] == pytest.approx(f["loss_lm"] + 0.5 * f["loss_srep_hinge"], rel=1e-5)


def test_the_hinge_has_a_documented_off_switch(tmp_path):
    """Weight 0 restores the pre-fix objective -- CLAUDE.md's off-switch rule.

    ⚠️ `rel=1e-6`, not `abs=0`, and the reason is worth stating rather than hiding
    behind a loose tolerance. At `w_srep = 0` the code takes the branch `loss = lm`,
    so the two are the *same expression*; they still disagree at ~2e-7 because
    `loss` is a float32 tensor accumulated across 48 sentence steps by
    `run_policy_loop` while `loss_lm` is a Python-float sum of the same 48 terms.
    That is accumulation order, not a residual penalty -- which the hinge-is-nonzero
    assertion below pins, since a 0.5-weighted hinge would show up at 1e-2.
    """
    r = train(out_dir=tmp_path / "off", srep_norm_reg_weight=0.0, seed=0, **TINY)
    f = r["final"]
    assert f["loss"] == pytest.approx(f["loss_lm"], rel=1e-6)
    assert f["loss_srep_hinge"] > 1e-3, (
        "the hinge is still computed and reported when the weight is 0 -- an "
        "off-switch that hid the measurement would make the arm unauditable"
    )
    on = train(out_dir=tmp_path / "on2", srep_norm_reg_weight=0.5, seed=0, **TINY)
    assert on["final"]["loss"] > f["loss"] + 1e-3, (
        "the weight must change the objective; if these agree the off-switch is "
        "switching nothing and this test is vacuous"
    )


def test_perplexity_is_a_perplexity_and_not_a_penalised_loss(tmp_path):
    """Adding a regulariser to `loss` must not silently contaminate `ppl`.

    `ppl` was `exp(loss / steps_per_stream)`. Had it been left there, the hinge would
    have entered a number reported as a perplexity -- a new silent defect shipped
    inside the fix for an old one.
    """
    import json

    r = train(out_dir=tmp_path / "ppl", srep_norm_reg_weight=0.5, seed=0, **TINY)
    beats = [
        json.loads(ln)
        for ln in (tmp_path / "ppl" / "heartbeat.jsonl").read_text().splitlines()
    ]
    beat = [b for b in beats if b.get("record") == "beat" or "ppl" in b][-1]
    assert beat["ppl"] == pytest.approx(
        float(torch.exp(torch.tensor(r["final"]["loss_lm"]))), rel=1e-5
    )


def test_the_hinge_is_on_by_default(tmp_path):
    """Correction 15 item 4: the hinge is part of TG's loss. Off is the arm, not the norm.

    🔴 **This test was vacuous when first written and the mutation battery caught it.**
    It asserted `loss > loss_lm`, which survives the "hinge back out of the objective"
    mutation: with `loss = lm` the two still differ by ~2.1e-7, because `loss` is a
    float32 tensor accumulated across 48 steps by `run_policy_loop` while `loss_lm` is
    a Python-float sum of the same terms. A strict `>` against float noise is not an
    assertion about the hinge; it is an assertion that two accumulation orders
    disagree, which they always do.

    The margin below is **measured, not chosen**. At the default weight the hinge
    contributes `0.01 * 0.0623 = 6.236e-4`; the noise it must be separated from is
    2.1e-7, three thousand times smaller. `1e-4` sits between them with room on both
    sides, and the exact-decomposition assertion is what actually pins the term.
    """
    r = train(out_dir=tmp_path / "dflt", seed=0, **TINY)
    f = r["final"]
    assert f["srep_norm_reg_weight"] == 0.01  # TGConfig's value, not this loop's
    assert f["loss"] - f["loss_lm"] > 1e-4
    assert f["loss"] == pytest.approx(
        f["loss_lm"] + f["srep_norm_reg_weight"] * f["loss_srep_hinge"], rel=1e-5
    )


# --------------------------------------------------------------------------- #
# (e) -- the derived vocabulary is reachable at the default
# --------------------------------------------------------------------------- #


def test_the_corpus_holds_156_unique_words():
    """The brief predicted 156 and did not verify it. Measured, and now pinned."""
    docs = generate(SyntheticConfig(sentences_per_document=48, seed=0))
    assert len(build_vocab(docs)) == 156


def test_the_cli_vocab_default_reaches_the_derived_path(monkeypatch, tmp_path):
    """A falsy default is what `V = vocab if vocab else ...` needs. 50257 is not falsy."""
    seen: dict[str, object] = {}

    def fake_train(**kw):
        seen.update(kw)
        return {"run_id": "x"}

    monkeypatch.setattr("rsr.train.loop.train", fake_train)
    assert main(["--out-dir", str(tmp_path)]) == 0
    assert not seen["vocab"], (
        f"--vocab defaults to {seen['vocab']!r}, so the derived path is unreachable "
        f"at the default and every CLI run allocates a 50257-row embedding for a "
        f"156-word corpus"
    )


def test_an_explicit_vocab_still_overrides_the_derived_path(monkeypatch, tmp_path):
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        "rsr.train.loop.train", lambda **kw: seen.update(kw) or {"run_id": "x"}
    )
    assert main(["--out-dir", str(tmp_path), "--vocab", "1024"]) == 0
    assert seen["vocab"] == 1024


def test_the_derived_vocabulary_is_what_the_model_is_built_with(tmp_path):
    r = train(out_dir=tmp_path / "v", vocab=None, seed=0, **TINY)
    assert r["config"]["tg"]["V"] == 160  # 4 specials + 156 words


# --------------------------------------------------------------------------- #
# Listed separately by the brief, and handled separately here
# --------------------------------------------------------------------------- #


def test_from_registry_honours_an_override_without_reading_the_registry():
    """`rsr.py:236-243` built `kw` eagerly, so an override could not save you.

    Measured on the empty ledger before the fix: `from_registry(..., nu=0.0,
    gamma=0.0)` raised `UnmeasuredConstant` on `nu` -- the value the caller had just
    supplied. The `gamma = 0` control arm and A2's `A_max = M` are exactly the arms
    the docstring says `overrides` exists for, and neither could be built.

    🔴 This is not a default. A key absent from `overrides` still raises, and that is
    asserted below so the fix cannot decay into one.
    """
    cfg = RSRConfig.from_registry(
        "synthetic",
        steps_per_epoch=1.0,
        nu=0.0,
        beta=0.0,
        gamma=0.0,
        t_warm=0.0,
        a_max=16,
    )
    assert (cfg.nu, cfg.beta, cfg.gamma, cfg.t_warm, cfg.a_max) == (
        0.0,
        0.0,
        0.0,
        0.0,
        16,
    )


def test_from_registry_still_raises_for_a_field_left_unoverridden():
    """The registry's refusal is the mechanism (D-1). Overriding four fields of five
    must still raise on the fifth, naming its experiment."""
    with pytest.raises(C.UnmeasuredConstant, match="beta"):
        RSRConfig.from_registry(
            "synthetic", steps_per_epoch=1.0, nu=0.0, gamma=0.0, t_warm=0.0, a_max=16
        )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "S0-01 'less certain', deliberately left failing rather than speculatively "
        "fixed: loop.py:210 passes value_head=None into build_param_groups, so psi-hat "
        "would train outside its muP group. Nothing in this tree constructs a value "
        "head in the loop yet -- RSRPolicy is unreachable while E1 is unlogged -- so "
        "there is no head to pass and a fix here would be fiction. This xfail is the "
        "record; it turns green the day the head is wired, and strict=True means it "
        "cannot pass unnoticed."
    ),
)
def test_the_value_head_lands_in_its_own_mup_group(tmp_path):
    from rsr.model.tg import TGConfig, TGModel
    from rsr.mup.param_groups import build_param_groups

    cfg = TGConfig(
        D=32,
        V=160,
        F=86,
        max_sentence_tokens=16,
        max_sentences_in_short_term=4,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        eod_id=3,
    )
    model = TGModel(cfg)
    groups = build_param_groups(model, None, base_lr=1e-3, d_model=32, base_width=128)
    names = {g.get("name") for g in groups}
    assert "value_head" in names
