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
    registry.record("E_lifetime", 40.0, experiment="E0e", run_id="r1")
    assert registry.get("E_lifetime") == 40.0
    assert registry.get("gamma_b") == pytest.approx(0.10)


def test_ledger_appends_and_keeps_provenance(registry):
    registry.record("tau", 0.25, experiment="E0e", run_id="r1")
    registry.record("tau", 0.31, experiment="E0e", run_id="r2")
    assert registry.get("tau") == 0.31  # most recent wins
    records = json.loads(registry.ledger_path.read_text())
    assert [r["run_id"] for r in records] == ["r1", "r2"]
    # Gauntlet 0.7 defect 7: the sha is stamped from git inside the measured tree.
    # A caller-typed sha records a belief; section 12.4 wants a fact.
    assert all(r["git_sha"] and len(r["git_sha"]) == 40 for r in records)
    assert all("dirty" in r for r in records)


def test_cannot_record_against_the_wrong_experiment(registry):
    with pytest.raises(C.ConstantError, match="E0e"):
        registry.record("tau", 0.25, experiment="E1", run_id="r")


def test_cannot_record_over_a_frozen_constant(registry):
    with pytest.raises(C.ConstantError, match="FROZEN"):
        registry.record("b_max", 2.0, experiment="E0e", run_id="r")


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


# --------------------------------------------------------------------------- #
# Gauntlet 0.7 -- the seven registry defects, one test each.
#
# Each of these reproduced against 83bdf57 before the fix. They are written as
# regression tests rather than as a changelog entry because the defects share a
# shape -- a guard that trusts its caller -- and that shape comes back.
# --------------------------------------------------------------------------- #


def test_scope_free_measured_read_does_not_leak_across_scopes(registry):
    """Defect 1. `get("tau")` used to return a value recorded on synthetic."""
    registry.record("tau", 0.11, experiment="E0e", run_id="r", scope="synthetic")
    with pytest.raises(C.ScopeRequired, match="per scope"):
        registry.get("tau")
    assert registry.get("tau", "synthetic") == 0.11


def test_derived_constants_cannot_be_recorded(registry):
    """Defect 2. `record("gamma_b", 999.0)` succeeded and `get()` ignored it.

    A ledger row nothing reads is worse than a refused write: it looks like
    provenance.
    """
    with pytest.raises(C.ConstantError, match="DERIVED"):
        registry.record("gamma_b", 999.0, experiment="E0e", run_id="r")


def test_K_cannot_be_recorded_against_an_arbitrary_experiment(registry):
    """Defect 5. `K` has no `source_experiment`, so the experiment check never
    fired and any name was accepted. Subsumed by the DERIVED refusal."""
    with pytest.raises(C.ConstantError, match="DERIVED"):
        registry.record(
            "K", 7, experiment="totally-made-up", run_id="r", scope="synthetic"
        )


def test_conditional_constants_cannot_be_recorded(registry):
    """The same hole on the other class: T_warm is computed from its context."""
    with pytest.raises(C.ConstantError, match="CONDITIONAL"):
        registry.record("T_warm", 1200.0, experiment="E0e", run_id="r")


def test_A_max_does_not_trust_a_caller_supplied_S(registry):
    """Defect 3. `get("A_max", "synthetic", S=10**9)` returned 48 and validated
    nothing -- the guard compared against whatever arrived."""
    with pytest.raises(C.ConstraintViolation, match="registry holds"):
        registry.get("A_max", "synthetic", S=10**9)


def test_A_max_with_S_none_raises_inside_the_constant_error_hierarchy(registry):
    """Defect 4. It raised a bare `TypeError`, so a caller catching registry
    refusals missed it -- and a missed refusal is a silently defaulted constant."""
    with pytest.raises(C.ConstantError):
        registry.get("A_max", "synthetic", S=None)


def test_every_scoped_definition_covers_the_whole_vocabulary():
    """Defect 6. E4, E7 and diagnostics had no entry in several definitions, so
    the read raised `UnknownScope` -- which reads as "you typed it wrong" when the
    truth is "the spec never says." Different problems, different fixes."""
    for name, defn in C.REGISTRY.definitions.items():
        if defn.scoped is None:
            continue
        assert set(defn.scoped) == set(C.SCOPES), (
            f"{name} covers {sorted(defn.scoped)}, vocabulary is {sorted(C.SCOPES)}"
        )


def test_an_unspecified_scope_says_so_rather_than_implying_a_typo(registry):
    with pytest.raises(C.UnknownScope, match="not specified for scope"):
        registry.get("K", "wikitext_e4")


def test_recording_under_an_unknown_scope_is_refused(registry):
    with pytest.raises(C.UnknownScope, match="scope vocabulary"):
        registry.record("tau", 0.2, experiment="E0e", run_id="r", scope="not_a_scope")


def test_git_provenance_is_stamped_not_supplied(registry):
    """Defect 7. There is no `git_sha` parameter to get wrong."""
    import inspect

    assert "git_sha" not in inspect.signature(C.Registry.record).parameters
    stamp = C.git_provenance()
    assert stamp["git_sha"] is None or len(stamp["git_sha"]) == 40


# --- D-I: T_warm has one representation, a float number of steps ------------- #


def test_T_warm_is_a_float_number_of_steps(registry):
    """D-I. The registry held the string "one_epoch" and `RSRConfig` held a float,
    with nothing converting between them."""
    got = registry.get("T_warm", steps_per_epoch=1200)
    assert isinstance(got, float)
    assert got == 1200.0


def test_T_warm_epochs_stays_readable_for_the_report(registry):
    """Section 3.4 requires it be reported as a fraction of total epochs -- one of
    three is a different experiment from one of twelve."""
    assert registry.get("T_warm_epochs") == 1.0


def test_T_warm_refuses_to_guess_steps_per_epoch(registry):
    with pytest.raises(C.ScopeRequired, match="steps_per_epoch"):
        registry.get("T_warm")


def test_T_warm_refuses_a_nonsense_steps_per_epoch(registry):
    with pytest.raises(C.ConstraintViolation):
        registry.get("T_warm", steps_per_epoch=0)


def test_no_string_representation_of_T_warm_survives():
    """D-I: "pick the float ... and delete the string"."""
    for name in ("T_warm", "T_warm_epochs"):
        assert not isinstance(C.REGISTRY.definition(name).value, str)
