"""The Expire-Span referendum's harness and launcher
(`experiments/expire-span-referendum/`, PREREG committed alone at 3d5ed87).

Pinned: the grid is the PREREG's (read, checked, never typed); VAL is disjoint; EVAL
is refused while δ is OPEN, before selection and on a second read; the classification
and the selection rule; the Expire-Span pass applies the mask that run 1's `Recorder`
would drop; the launcher's pins are run 1's and run 1's launcher is untouched.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest
import torch

from rsr.baselines.expire_span import ExpireSpanConfig, ExpireSpanPolicy
from rsr.model.tg import TGConfig, TGModel
from rsr.train import checkpoint as ck
from rsr.train.loop import train

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/expire-span-referendum/run.py"
LAUNCHER = ROOT / "experiments/expire-span-referendum/launch_es.zsh"


@pytest.fixture(scope="module")
def esr():
    name = "_test_expire_span_referendum"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, RUN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# The grid, read from the PREREG
# --------------------------------------------------------------------------- #

#: §4.1 as committed at 3d5ed87: (config, L, R, kappa, alpha).
PREREG_GRID = [
    ("L16-k1e-4", 16.0, 2.56, 1e-4, 2.44e-6),
    ("L16-k1e-3", 16.0, 2.56, 1e-3, 2.44e-5),
    ("L16-k1e-2", 16.0, 2.56, 1e-2, 2.44e-4),
    ("L32-k1e-4", 32.0, 5.12, 1e-4, 6.10e-7),
    ("L32-k1e-3", 32.0, 5.12, 1e-3, 6.10e-6),
    ("L32-k1e-2", 32.0, 5.12, 1e-2, 6.10e-5),
    ("L48-k1e-4", 48.0, 7.68, 1e-4, 2.71e-7),
    ("L48-k1e-3", 48.0, 7.68, 1e-3, 2.71e-6),
    ("L48-k1e-2", 48.0, 7.68, 1e-2, 2.71e-5),
]


def test_the_grid_is_the_prereg_table(esr):
    got = [(c["config"], c["L"], c["R"], c["kappa"], c["alpha"]) for c in esr.grid()]
    assert got == PREREG_GRID


def test_every_config_parses_to_a_buildable_expire_span_with_the_fixed_fields(esr):
    for c in esr.grid():
        ov = esr.parsed_overrides(c)
        assert ov["grad_path"] == "predictor"
        assert ov["structured_dropout"] is True
        assert ov["init_span_fraction"] == 0.1
        ExpireSpanConfig(**ov, seed=0)  # validates


def _text(esr, **swap):
    t = esr.PREREG.read_text()
    for old, new in swap.items():
        assert t.count(old) == 1, old
        t = t.replace(old, new)
    return t


def test_a_row_whose_alpha_is_not_kappa_over_LR_is_refused(esr):
    row = "| L32-k1e-3 | 32 | 5.12 | 1e-3 |"
    bad = _text(esr, **{f"{row} 6.10e-6 |": f"{row} 6.10e-5 |"})
    with pytest.raises(esr.Refused, match="kappa"):
        esr.grid(bad)


def test_a_ramp_off_the_registered_fraction_is_refused(esr):
    bad = _text(esr, **{"| L48-k1e-4 | 48 | 7.68 |": "| L48-k1e-4 | 48 | 3.84 |"})
    with pytest.raises(esr.Refused, match="R ="):
        esr.grid(bad)


def test_a_grad_path_other_than_the_ruled_one_is_refused(esr):
    bad = _text(esr, **{"| grad_path | predictor |": "| grad_path | through_gestalt |"})
    with pytest.raises(esr.Refused, match="q1"):
        esr.grid(bad)


def test_the_grid_subcommand_prints_nine_launcher_lines(esr, capsys):
    with pytest.raises(SystemExit) as e:
        from rsr.exit_codes import run_main

        run_main(lambda: esr.main(["grid"]))
    assert e.value.code == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 9
    assert lines[0] == (
        "L16-k1e-4 max_span=16 ramp=2.56 loss_coef=2.44e-6 structured_dropout=true "
        "grad_path=predictor init_span_fraction=0.1"
    )


# --------------------------------------------------------------------------- #
# Splits, δ, EVAL refusals
# --------------------------------------------------------------------------- #


def test_val_is_disjoint_and_closed(esr):
    out = esr.check_val(seeds=(0,))
    assert out["ok"] and out["overlaps"] == []
    assert esr.VAL == (994000, 994512)


def test_delta_is_open_in_the_committed_prereg(esr):
    assert esr.delta_per_seed() is None


def _delta_text(esr, vals=("0.01", "0.01", "0.01")):
    t = esr.PREREG.read_text()
    for s, v in zip((0, 1, 2), vals, strict=True):
        t = t.replace(f"| {s} | OPEN |", f"| {s} | {v} |")
    return t


def test_a_filled_delta_table_is_read_per_seed(esr):
    assert esr.delta_per_seed(_delta_text(esr, ("0.01", "0.02", "0.03"))) == {
        0: 0.01,
        1: 0.02,
        2: 0.03,
    }


def test_evaluate_refuses_eval_while_delta_is_open(esr, tmp_path):
    with pytest.raises(esr.Refused, match="OPEN"):
        esr.evaluate(0, tmp_path)
    assert not (tmp_path / "eval").exists()


def test_evaluate_refuses_eval_before_selection(esr, tmp_path):
    with pytest.raises(esr.Refused, match="selection"):
        esr.evaluate(0, tmp_path, delta_text=_delta_text(esr))
    assert not (tmp_path / "eval").exists()


def test_evaluate_refuses_a_second_eval_read(esr, tmp_path):
    (tmp_path / "selection.json").write_text(json.dumps({"selected": "L16-k1e-4"}))
    m = esr.eval_marker(tmp_path, 0)
    m.parent.mkdir(parents=True)
    m.write_text("{}")
    with pytest.raises(esr.Refused, match="read once"):
        esr.evaluate(0, tmp_path, delta_text=_delta_text(esr))


# --------------------------------------------------------------------------- #
# Classification (§7.2) and selection (§6.3)
# --------------------------------------------------------------------------- #


def _ci(*pairs):
    return {s: {"lo": lo, "hi": hi} for s, (lo, hi) in enumerate(pairs)}


D = {0: 0.02, 1: 0.02, 2: 0.02}


def test_classify_supersedes_loses_ties_mixed(esr):
    assert esr.classify(_ci((0.03, 0.05), (0.04, 0.06), (0.03, 0.04)), D) == "SUPERSEDES"
    assert esr.classify(_ci((-0.05, -0.01), (-0.06, -0.04), (-0.03, -0.02)), D) == "LOSES"
    assert esr.classify(_ci((-0.01, 0.01), (-0.02, 0.0), (0.0, 0.02)), D) == "TIES"
    assert esr.classify(_ci((-0.01, 0.03), (-0.02, 0.0), (0.0, 0.02)), D) == "MIXED"
    assert esr.classify(_ci((0.01, 0.05), (-0.01, 0.02), (0.03, 0.04)), D) == "MIXED"


def test_supersedes_takes_precedence_over_ties(esr):
    """A CI inside (0, δ] on every seed satisfies both; §7.2's notes: first match."""
    assert (
        esr.classify(_ci((0.001, 0.01), (0.002, 0.015), (0.001, 0.02)), D) == "SUPERSEDES"
    )


def test_classify_refuses_without_delta(esr):
    with pytest.raises(esr.Refused):
        esr.classify(_ci((0.03, 0.05), (0.04, 0.06), (0.03, 0.04)), None)


def _summaries(acc_by_config, ok=True):
    """acc_by_config: config -> [(correct, answers)] per seed."""
    out = {}
    for s in (0, 1, 2):
        out[s] = {
            "configs": {
                c: {"controls": {"ok": ok}, "correct": v[s][0], "answers": v[s][1]}
                for c, v in acc_by_config.items()
            }
        }
    return out


def test_selection_takes_the_highest_mean_and_breaks_exact_ties_to_the_lowest_row(esr):
    grid = esr.grid()
    ids = [c["config"] for c in grid]
    base = {c: [(50, 100)] * 3 for c in ids}
    base[ids[4]] = [(60, 100)] * 3
    assert esr.selection(_summaries(base), grid)["selected"] == ids[4]
    # an EXACT tie (1/2 = 2/4) between rows 7 and 2 -> row 2
    tie = {c: [(10, 100)] * 3 for c in ids}
    tie[ids[7]] = [(1, 2), (1, 2), (1, 2)]
    tie[ids[2]] = [(2, 4), (3, 6), (50, 100)]
    sel = esr.selection(_summaries(tie), grid)
    assert sel["selected"] == ids[2] and sel["tied_at_max"] == [ids[2], ids[7]]
    assert sel["table"][2]["mean_exact"] == str(Fraction(1, 2))


def test_selection_refuses_a_failed_control(esr):
    grid = esr.grid()
    s = _summaries({c["config"]: [(1, 2)] * 3 for c in grid}, ok=False)
    with pytest.raises(esr.Refused):
        esr.selection(s, grid)


# --------------------------------------------------------------------------- #
# The Expire-Span pass: the mask run 1's Recorder would drop
# --------------------------------------------------------------------------- #


def _tiny_model_and_policy(esr):
    r1 = esr.R1()
    cfg = r1.prereg_config(0, 3000)
    vmap = r1.vocab_map(cfg)
    tg = TGConfig(
        D=32,
        H=2,
        V=4 + len(vmap),
        max_sentence_tokens=64,
        max_sentences_in_short_term=16,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        eod_id=3,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = TGModel(tg)
    es_cfg = ExpireSpanConfig(
        max_span=8.0,
        ramp=2.0,
        loss_coef=0.0,
        structured_dropout=True,
        grad_path="predictor",
        init_span_fraction=0.1,
        seed=0,
    )
    pol = ExpireSpanPolicy(es_cfg, 32, sum(b == "C" for b in tg.block_config))
    docs = r1.docs_by_id(cfg, range(993000, 993016))  # POOL: training data
    return r1, cfg, vmap, model, pol, docs


def test_the_es_pass_applies_the_mask_at_every_step(esr):
    r1, cfg, vmap, model, pol, docs = _tiny_model_and_policy(esr)
    recs = esr.es_pass(model, pol, docs, vmap, cfg)
    assert pol.weight_calls == cfg.steps_per_stream  # E7: one batch of 16
    plain = r1.run_batch(model, pol, docs, vmap, cfg)  # run 1's Recorder: no mask
    assert [r["doc"] for r in recs] == [r["doc"] for r in plain]
    assert [r["victims"] for r in recs] == [r["victims"] for r in plain]
    nll = lambda rs: [a[3] for r in rs for a in r["answers"]]  # noqa: E731
    assert nll(recs) != nll(plain)  # the mask changed what the model read
    assert r1.Recorder.__name__ == "Recorder"  # restored after the pass


def test_the_es_pass_draws_no_dropout_at_eval(esr):
    _r1, cfg, vmap, model, pol, docs = _tiny_model_and_policy(esr)
    g0 = pol.get_extra_state()["generator"].clone()
    esr.es_pass(model, pol, docs, vmap, cfg)
    assert torch.equal(pol.get_extra_state()["generator"], g0)


def test_es_policy_from_payload_restores_the_trained_w_b(esr, tmp_path):
    ov = dict(
        max_span=8.0,
        ramp=2.0,
        loss_coef=1e-3,
        structured_dropout=True,
        grad_path="predictor",
        init_span_fraction=0.1,
    )
    train(
        d=32,
        steps_per_stream=48,
        batch=2,
        memory_slots=4,
        device="cpu",
        n_documents=8,
        iters=2,
        out_dir=tmp_path,
        ckpt_every=2,
        policy_name="expire_span",
        policy_overrides=ov,
        freeze_transformer=True,
        frozen_model_mode="eval",
    )
    payload = ck.load(tmp_path / "ckpt-000002.pt", restore_rng=False)
    pol = esr.es_policy_from_payload(payload)
    assert torch.equal(pol.weight.detach(), payload["policy"]["weight"])
    assert torch.equal(pol.bias.detach(), payload["policy"]["bias"])
    assert pol.cfg.grad_path == "predictor"


# --------------------------------------------------------------------------- #
# The launcher
# --------------------------------------------------------------------------- #


def test_the_launcher_pins_run_1s_substrate_and_reads_the_grid(esr):
    text = LAUNCHER.read_text()
    r1 = esr.R1()
    for s, sub in r1.SUBSTRATE.items():
        assert f"SHA[{s}]={sub['sha256']}" in text
    for flag in (
        "--freeze-transformer --frozen-model-mode eval",
        "--srep-norm-reg-weight 0.0",
        "--pool-offset 993000 --pool-length 256 --pool-vocab-documents 64",
        "--lr 0.001",
        "--policy expire_span",
        "run.py grid",
        "--lane cpu-det --slots 1",
        "--pinned-sha $RUN_SHA",
    ):
        assert flag in text, flag
    code = "\n".join(x for x in text.splitlines() if not x.lstrip().startswith("#"))
    for forbidden in ("995000", "997048", "994000", "launch_run1", "loss_coef=", "2.44e"):
        assert forbidden not in code, forbidden


def test_run_1s_launcher_is_untouched(esr):
    run1 = ROOT / "experiments/rsr-run1/launch_run1.zsh"
    assert hashlib.sha256(run1.read_bytes()).hexdigest() == esr.RUN1_LAUNCHER_SHA256
