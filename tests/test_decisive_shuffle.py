"""The decisive run's own gates (`experiments/decisive-shuffle/PREREG.md`, *Mutation
bar*).

The experiment script is new, so it is not believed until these have been seen
red (`scripts/mutation_battery.py`):

* **Arm swap.** `ARMS` must be the PREREG's table, and a checkpoint whose config
  is another arm's must be refused by `verify_arm` against the frozen manifest
  entry -- not against the script's `ARMS`, which is what a swap would edit.
* **Decoy aliasing.** Pointing the decoy at the trained checkpoint makes the
  decoy's reading the trained model's own, so `ratio == 1.0` exactly, and the
  rule must return `inconclusive` rather than `live` (1.0 >= 0.1).
* **The decision rule** is PREREG's, band for band, including the partial case.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "decisive_run", ROOT / "experiments" / "decisive-shuffle" / "run.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run = _load()


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #

#: PREREG.md "Condition", table "Arms", copied by hand -- the independent copy a
#: swap in the script cannot also edit.
PREREG_ARMS = {
    "A": {"masked_loss": False, "srep_norm_reg_weight": 0.0},
    "B": {"masked_loss": True, "srep_norm_reg_weight": 0.0},
    "C": {"masked_loss": True, "srep_norm_reg_weight": None},
}


def test_arms_are_the_preregistered_table():
    assert run.ARMS == PREREG_ARMS
    assert run.SEEDS == [0, 1, 2]
    assert run.resolved_arm("C")["srep_norm_reg_weight"] == 0.01


def _result(arm_settings: dict, iters: int = 300) -> dict:
    """A `train()` result as `run_trainings` sees it, for the given arm settings."""
    cfg = {
        "tg": {"D": 128, "max_sentences_in_short_term": 16},
        "iters": iters,
        "batch": 16,
        "steps_per_stream": 48,
        "lr": 1e-3,
        "seed": 0,
        "policy": "fifo",
        "device": "cpu",
        "masked_loss": arm_settings["masked_loss"],
        "srep_norm_reg_weight": arm_settings["srep_norm_reg_weight"],
    }
    return {
        "config": cfg,
        "config_hash": run._config_hash(cfg),
        "steps_requested": iters,
    }


def test_every_arm_passes_against_its_own_frozen_entry():
    frozen = run.manifest_arms()
    for arm in run.ARMS:
        run.verify_arm(frozen, arm, _result(run.resolved_arm(arm)))


@pytest.mark.parametrize(("arm", "impostor"), [("A", "B"), ("B", "C"), ("C", "A")])
def test_an_arm_swap_is_refused_by_the_manifest_check(arm, impostor):
    frozen = run.manifest_arms()
    with pytest.raises(run.ArmMismatch, match="arm hash"):
        run.verify_arm(frozen, arm, _result(run.resolved_arm(impostor)))


def test_a_config_that_does_not_hash_to_its_own_stamp_is_refused():
    frozen = run.manifest_arms()
    r = _result(run.resolved_arm("B"))
    r["config"]["lr"] = 2e-3  # edited after the hash was stamped
    with pytest.raises(run.ArmMismatch):
        run.verify_arm(frozen, "B", r)


# --------------------------------------------------------------------------- #
# decision rule
# --------------------------------------------------------------------------- #


def _m(a_trained: float, a_decoy: float, *, disabled=0.0, own=0.0) -> dict:
    return {
        "reading": {"mean_abs_token_delta": a_trained},
        "control_live_decoy": {"mean_abs_token_delta": a_decoy},
        "control_memory_disabled": {"delta_exactly_zero": disabled == 0.0},
        "control_own_memory": {"delta_exactly_zero": own == 0.0},
    }


def test_decide_arm_bands():
    d = run.decide_arm
    assert d({s: _m(0.001, 1.0) for s in range(3)})[0] == "inert"
    assert d({0: _m(0.001, 1.0), 1: _m(0.2, 1.0), 2: _m(0.001, 1.0)})[0] == "live"
    assert d({0: _m(0.001, 1.0), 1: _m(0.05, 1.0), 2: _m(0.001, 1.0)})[0] == (
        "inconclusive"
    )
    # the thresholds are inclusive, as PREREG writes them
    assert d({s: _m(0.01, 1.0) for s in range(3)})[0] == "inert"
    assert d({0: _m(0.1, 1.0), 1: _m(0.0, 1.0), 2: _m(0.0, 1.0)})[0] == "live"


def test_a_failed_control_or_dead_decoy_is_inconclusive_whatever_the_ratio():
    d = run.decide_arm
    assert d({0: _m(0.5, 1.0, disabled=1e-9), 1: _m(0.5, 1), 2: _m(0.5, 1)})[0] == (
        "inconclusive"
    )
    assert d({0: _m(0.5, 1.0, own=1e-9), 1: _m(0.5, 1), 2: _m(0.5, 1)})[0] == (
        "inconclusive"
    )
    assert d({0: _m(0.5, 0.0), 1: _m(0.5, 1), 2: _m(0.5, 1)})[0] == "inconclusive"


def test_fewer_than_three_seeds_is_inconclusive_partial_never_rounded():
    assert run.decide_arm({0: _m(0.5, 1.0), 1: _m(0.5, 1.0)})[0] == (
        "inconclusive (partial)"
    )
    assert run.decide_arm({0: _m(0.0, 1.0)})[0] == "inconclusive (partial)"


def test_ratio_exactly_one_is_inconclusive_not_live():
    assert run.decide_arm({s: _m(0.3, 0.3) for s in range(3)})[0] == "inconclusive"


def test_overall_reads_b_and_c():
    assert run.overall({"A": "inert", "B": "inert", "C": "inert"}).startswith("inert")
    assert "arm A" in run.overall({"A": "inert", "B": "live", "C": "inert"})
    assert run.overall({"A": "live", "B": "inconclusive", "C": "inert"}).startswith(
        "live"
    )
    assert run.overall({"A": "inert", "B": "inconclusive", "C": "inert"}).startswith(
        "neither"
    )


# --------------------------------------------------------------------------- #
# decoy aliasing, end to end through measure()
# --------------------------------------------------------------------------- #


def test_the_decoy_pointed_at_the_trained_checkpoint_reads_ratio_one_and_inconclusive(
    tmp_path,
):
    from rsr.model.tg import TGModel
    from rsr.train import checkpoint as ck

    _, V = run.batches(0)
    torch.manual_seed(123)  # not the decoy's seed: a genuinely different model
    model = TGModel(run._cfg(V))
    p = ck.save(tmp_path / "ckpt.pt", step=0, model=model)

    m = run.measure(p, 0, decoy_ckpt=p)["train"]
    a_t = m["reading"]["mean_abs_token_delta"]
    a_d = m["control_live_decoy"]["mean_abs_token_delta"]
    assert a_t == a_d  # the decoy IS the trained model
    outcome, detail = run.decide_arm({s: m for s in (0, 1, 2)})
    assert outcome == "inconclusive", detail
    if a_d:
        assert run.ratio(m) == 1.0
