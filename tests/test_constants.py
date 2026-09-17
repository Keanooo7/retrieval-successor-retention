"""The constants registry must actually refuse. §4.5: "Never freeze an unmeasured constant."""

from __future__ import annotations

import pytest

from rsr.constants import (
    Class,
    ConstantError,
    Registry,
    UnknownScopeError,
    UnmeasuredConstantError,
)


@pytest.fixture
def reg(tmp_path):
    return Registry(store=tmp_path / "measurements.json")


def test_frozen_constants_read_immediately(reg):
    assert reg.get("M", "synthetic") == 16
    assert reg.get("M", "corpora") == 40
    assert reg.get("M", "e7") == 8
    assert reg.get("b_max") == 1.0
    assert reg.get("lambda_return") == 1.0, "C-2: Monte Carlo is the default, TD(0) is A8"


def test_K_is_not_M(reg):
    """Correction C-1. §3.4's closing line still says K = M; the self-audit overrides it.

    K = M = 40 censors exactly the (40, 64] events E3 exists to measure. This is the single line
    most likely to be copied into an implementation, so it is pinned by a test.
    """
    assert reg.get("K", "pg19") == 64
    assert reg.get("K", "synthetic") == 40
    assert reg.get("K", "pg19") != reg.get("M", "corpora")


def test_measured_constant_raises_before_its_experiment(reg):
    with pytest.raises(UnmeasuredConstantError) as e:
        reg.get("tau")
    msg = str(e.value)
    assert "E0e" in msg, "the refusal must name the experiment that owes the value"
    assert "MEASURED" in msg


def test_derived_constant_names_its_missing_input(reg):
    with pytest.raises(UnmeasuredConstantError) as e:
        reg.get("gamma_b")
    msg = str(e.value)
    assert "E_lifetime" in msg
    assert "b_max / (0.25 * E_lifetime)" in msg


def test_gamma_b_derives_once_lifetime_is_measured(reg):
    reg.record("E_lifetime", 40.0, experiment="E0e", evidence="experiments/e0e/RESULTS.md")
    got = reg.get("gamma_b")
    assert got == pytest.approx(0.1), "D-1: order 0.05-0.1, NOT 0.001"
    assert got > 0.01, "at 0.001 the whole anti-collapse section is inert (0.08 SD over a stream)"


def test_ema_half_life_derives_from_lifetime(reg):
    reg.record("E_lifetime", 40.0, experiment="E0e", evidence="cmd: rsr e0e")
    assert reg.get("ema_half_life") == pytest.approx(10.0)


def test_recording_requires_evidence(reg):
    with pytest.raises(ConstantError, match="evidence"):
        reg.record("tau", 0.25, experiment="E0e", evidence="   ")


def test_frozen_constants_cannot_be_recorded(reg):
    with pytest.raises(ConstantError, match="FROZEN"):
        reg.record("b_max", 2.0, experiment="E0e", evidence="x")


def test_scoped_constant_refuses_a_bare_read(reg):
    with pytest.raises(UnknownScopeError):
        reg.get("M")


def test_A_max_synthetic_is_unset_and_says_why(reg):
    """Correction B-5. Synthetic is S=48, M=16; if A_max defaults to M then gamma=0.97 (horizon 33)
    is illegal on the only corpus gamma is swept on. The registry holds it unset so the sweep
    cannot quietly run against a default."""
    with pytest.raises(UnmeasuredConstantError) as e:
        reg.get("A_max_synthetic")
    assert "0.97" in str(e.value)


def test_unset_list_is_the_ratchet(reg):
    before = reg.unset()
    assert "tau" in before and "E_lifetime" in before
    reg.record("E_lifetime", 40.0, experiment="E0e", evidence="x")
    after = reg.unset()
    assert len(after) < len(before), "recording a measurement must lower the count; it is a ratchet"
    assert "gamma_b" not in after, "a DERIVED constant becomes set when its inputs do"


def test_measurements_persist_across_registry_instances(tmp_path):
    store = tmp_path / "m.json"
    Registry(store=store).record("tau", 0.31, experiment="E0e", evidence="experiments/e0e/")
    assert Registry(store=store).get("tau") == 0.31


def test_every_measured_constant_names_an_experiment():
    from rsr.constants import _CONSTANTS

    for c in _CONSTANTS:
        if c.cls in (Class.MEASURED, Class.DERIVED, Class.CONDITIONAL):
            assert c.experiment, f"{c.name} is {c.cls} but names no experiment to owe it to"
