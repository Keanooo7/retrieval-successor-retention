"""The registry must refuse unmeasured reads (spec section 4.5, kickoff item 2).

Section 4.5's rule -- "never freeze an unmeasured constant" -- is stated as
discipline. These tests are what make it mechanical.
"""

from __future__ import annotations

import json

import pytest

from rsr import constants as C


@pytest.fixture
def registry(tmp_path):
    return C.Registry(definitions=C.REGISTRY.definitions, ledger_path=tmp_path / "l.json")


def test_measured_constant_raises_and_names_its_experiment(registry):
    with pytest.raises(C.UnmeasuredConstant) as exc:
        registry.get("tau")
    assert "E0e" in str(exc.value)


def test_derived_constant_raises_through_its_dependency(registry):
    """`gamma_b` depends on `E_lifetime`, which E0e measures."""
    with pytest.raises(C.UnmeasuredConstant) as exc:
        registry.get("gamma_b")
    assert "E0e" in str(exc.value)


def test_gamma_b_is_order_point_zero_five_to_point_one_not_point_zero_zero_one():
    """Defect D-1. At 0.001 the loop cannot move the argmin it governs.

    Section 3.5 item 4 gives 0.20 / 0.10 / 0.05 at E[lifetime] of 20 / 40 / 80.
    """
    for lifetime, expected in ((20, 0.20), (40, 0.10), (80, 0.05)):
        got = C._derive_gamma_b(1.0, lifetime)
        assert got == pytest.approx(expected)
        assert got > 0.001 * 10, "gamma_b collapsed toward the inert v0.4 value"


def test_recording_then_reading_a_measured_constant(registry):
    registry.record("E_lifetime", 40.0, experiment="E0e", git_sha="deadbeef", run_id="r1")
    assert registry.get("E_lifetime") == 40.0
    assert registry.get("gamma_b") == pytest.approx(0.10)


def test_ledger_appends_and_keeps_provenance(registry):
    registry.record("tau", 0.25, experiment="E0e", git_sha="aaa", run_id="r1")
    registry.record("tau", 0.31, experiment="E0e", git_sha="bbb", run_id="r2")
    assert registry.get("tau") == 0.31  # most recent wins
    records = json.loads(registry.ledger_path.read_text())
    assert [r["git_sha"] for r in records] == ["aaa", "bbb"]


def test_cannot_record_against_the_wrong_experiment(registry):
    with pytest.raises(C.ConstantError, match="E0e"):
        registry.record("tau", 0.25, experiment="E1", git_sha="x", run_id="r")


def test_cannot_record_over_a_frozen_constant(registry):
    with pytest.raises(C.ConstantError, match="FROZEN"):
        registry.record("b_max", 2.0, experiment="E0e", git_sha="x", run_id="r")


# --- scoping: section 5.1, 5.2, and section 13's no-cross-corpus rule --------- #


def test_M_has_no_global_value(registry):
    """Section 13: no cross-corpus comparison of absolute numbers is valid."""
    with pytest.raises(C.ScopeRequired):
        registry.get("M")


@pytest.mark.parametrize(
    "scope,expected", [("synthetic", 16), ("corpora", 40), ("e7", 8)]
)
def test_M_per_scope(registry, scope, expected):
    assert registry.get("M", scope) == expected


@pytest.mark.parametrize("scope,expected", [("synthetic", 48), ("pg19_e3", 80)])
def test_S_per_scope(registry, scope, expected):
    assert registry.get("S", scope) == expected


def test_K_is_max_target_gap_not_M(registry):
    """Correction 1 / the self-audit. `K = M = 40` censors the (40, 64] events."""
    assert registry.get("K", "pg19") == 64
    assert registry.get("K", "synthetic") == 40
    assert registry.get("K", "pg19") > registry.get("M", "corpora")


# --- A_max is CONDITIONAL on S: section 3.6(a) ------------------------------- #


def test_A_max_requires_S(registry):
    with pytest.raises(C.ScopeRequired, match="S"):
        registry.get("A_max", "synthetic")


def test_A_max_equals_S_on_synthetic(registry):
    """ADR-0004 / correction 9. Non-binding, which is what makes gamma=0.97 legal."""
    assert registry.get("A_max", "synthetic", S=48) == 48


def test_A_max_above_S_is_refused(registry):
    with pytest.raises(C.ConstraintViolation, match="exceeds S"):
        registry.get("A_max", "pg19_e3", S=48)


# --- the gamma horizon constraint: section 4.5 ------------------------------- #


def test_gamma_097_is_legal_on_synthetic_with_A_max_equal_S():
    """Correction 9. Reading section 3.6(a)'s {16,32,64} as a prescription is what
    made this look illegal. It is not."""
    C.check_gamma_horizon(0.97, s=48, a_max=48)


def test_gamma_097_is_legal_on_the_E3_headline():
    C.check_gamma_horizon(0.97, s=80, a_max=64)


def test_gamma_097_refused_when_A_max_binds_below_the_horizon():
    with pytest.raises(C.ConstraintViolation, match="horizon"):
        C.check_gamma_horizon(0.97, s=48, a_max=32)


def test_gamma_099_stays_dropped():
    """Horizon 100 exceeds S = 80 itself (section 4.5)."""
    with pytest.raises(C.ConstraintViolation):
        C.check_gamma_horizon(0.99, s=80, a_max=64)


def test_gamma_zero_is_a_control_arm_and_always_legal():
    """Section 2: the separating control. Context conditioning without lookahead."""
    C.check_gamma_horizon(0.0, s=48, a_max=16)
