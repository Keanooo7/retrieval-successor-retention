"""Cycle 0 of the 2026-09-19 run: the evidence machinery, tested.

**Why this file exists.** Of the 13 ledgers the 2026-09-18 overnight loop wrote,
**9 cannot be re-executed from any commit**: seven `commands[].argv` entries point
at a session-scoped scratch directory belonging to a different session, one is the
literal placeholder `<scratch>/train_arm.py`, and one is a bare
`run_trained_alpha.py` that is not in the tree. There is no path from any commit to
most of last night's numbers.

The sharpest case decides the design. `runs/cycle-srep-hinge/ledger.json` has
`git_dirty: false` and is *still* unreproducible, because its modified trainer was
never committed. **So the gate is not "the tree is clean" -- it is "the code that
ran is committed."** A clean-tree check would have passed that ledger.

Every test here is paired with an entry in `scripts/mutation_battery.py` that
reddens it and nothing else. A check no mutation has shown red is a check that adds
nothing, and the battery is where that claim is settled.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

import canary  # noqa: E402
import ledger as ledger_mod  # noqa: E402
import mutation_battery  # noqa: E402
import render_scoreboard  # noqa: E402
from ledger import Ledger, Unreproducible  # noqa: E402

# The literal argv out of runs/cycle-bias-interface/ledger.json, 2026-09-18. A gate
# that only fails on a synthetic input is a gate you have not tested.
HISTORICAL_SCRATCH_ARGV = (
    ".venv/bin/python /private/tmp/claude-501/-Users-keanooo7-second-brain/"
    "1bdd6660-ee4c-49c5-99a7-28bf139ef2c9/scratchpad/run_bias_interface.py"
)


@pytest.fixture
def led(tmp_path, monkeypatch):
    """A ledger whose provenance is clean and whose `runs/` is disposable."""
    monkeypatch.setattr(ledger_mod, "_dirty_source_paths", lambda: [])
    return Ledger("t", question="q?", runs_root=tmp_path)


# --------------------------------------------------------------------------- #
# 5.1 -- scripts/ledger.py
# --------------------------------------------------------------------------- #


def test_command_refuses_the_real_historical_scratch_argv(led):
    """The motivating input, verbatim, not a synthetic stand-in."""
    with pytest.raises(Unreproducible) as exc:
        led.command(HISTORICAL_SCRATCH_ARGV, exit_code=0)
    assert "run_bias_interface.py" in str(exc.value)


def test_command_refuses_an_entry_point_not_in_the_tree(led):
    with pytest.raises(Unreproducible):
        led.command(".venv/bin/python run_trained_alpha.py --trials 400", exit_code=0)


def test_command_refuses_a_command_with_no_file_entry_point_at_all(led):
    """`python -c '...'` and the `<scratch>` placeholder are both this shape."""
    with pytest.raises(Unreproducible):
        led.command([".venv/bin/python", "-c", "import rsr; print(1)"], exit_code=0)
    with pytest.raises(Unreproducible):
        led.command(".venv/bin/python <scratch>/train_arm.py --hinge 0", exit_code=0)


def test_command_accepts_a_committed_entry_point(led):
    led.command(".venv/bin/python scripts/canary.py 4", exit_code=0)
    assert led.doc["commands"][0]["reproducible"] is True


def test_a_declared_tool_entry_point_is_accepted(led):
    """`pytest` carries no code of its own; tests/ and src/ are committed. A gate
    that forces the manager's own verification through the escape hatch -- which
    caps the verdict -- is a gate that gets routed around."""
    led.command("uv run pytest -rs", exit_code=0)
    assert led.doc["commands"][0]["reproducible"] is True
    assert "pytest" in ledger_mod.TOOL_ENTRY_POINTS


def test_command_refuses_an_undeclared_tool_entry_point(led):
    with pytest.raises(Unreproducible):
        led.command("uv run some-other-tool --go", exit_code=0)


def test_the_escape_hatch_caps_the_verdict_at_inconclusive(led):
    """A number you cannot re-run is not evidence that survived."""
    led.command(HISTORICAL_SCRATCH_ARGV, exit_code=0, allow_unreproducible=True)
    assert led.doc["commands"][0]["reproducible"] is False
    led.verdict(falsifier="f", outcome="survived", detail="d")
    assert led.doc["verdict"]["outcome"] == "inconclusive"
    assert led.doc["verdict"]["capped_from"] == "survived"


def test_exit_code_is_required(led):
    """Seven of last night's 13 ledgers carry `exit_code: null` -- they never
    learned whether the process exited 0."""
    with pytest.raises(TypeError):
        led.command(".venv/bin/python scripts/canary.py 4")  # type: ignore[call-arg]


def test_a_git_failure_is_not_recorded_as_a_clean_tree(monkeypatch, tmp_path):
    """`bool("")` is `False`: the old `_git()` made a git failure indistinguishable
    from a clean tree. Copy `constants.git_provenance`'s shape instead."""
    monkeypatch.setattr(ledger_mod, "_git_exe", lambda: None)
    prov = ledger_mod.git_provenance()
    assert prov["git_sha"] is None
    assert prov["dirty"] is None
    assert prov["provenance_error"]


def test_write_refuses_a_dirty_source_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ledger_mod, "_dirty_source_paths", lambda: ["src/rsr/train/loop.py"]
    )
    led = Ledger("t", question="q?", runs_root=tmp_path)
    led.status("ok")
    with pytest.raises(ledger_mod.DirtyTree) as exc:
        led.write()
    assert "src/rsr/train/loop.py" in str(exc.value)


def test_a_dirty_runs_directory_is_fine(tmp_path, monkeypatch):
    """`runs/` churns by design; only `src/`, `scripts/` and `experiments/` gate."""
    monkeypatch.setattr(
        ledger_mod,
        "_porcelain",
        lambda: " M runs/canary/baseline.json\n?? runs/x/ledger.json\n",
    )
    assert ledger_mod._dirty_source_paths() == []


def test_a_dirty_source_path_is_detected_through_porcelain(monkeypatch):
    monkeypatch.setattr(
        ledger_mod,
        "_porcelain",
        lambda: " M src/rsr/train/loop.py\n M runs/canary/baseline.json\n",
    )
    assert ledger_mod._dirty_source_paths() == ["src/rsr/train/loop.py"]


def test_the_reconciliation_fields_are_written(led, tmp_path):
    """The role files check for these and nothing wrote them; the two evidence
    formats disagreed as a result."""
    led.run_meta(device="cpu", seeds_actually_run=[0, 1], steps_requested=100,
                 steps_done=100)
    led.status("ok")
    led.manifest({"d": 128, "seed": 0})
    p = led.write()
    doc = json.loads(p.read_text())
    assert doc["device"] == "cpu"
    assert doc["seeds_actually_run"] == [0, 1]
    assert doc["steps_requested"] == 100
    assert doc["steps_done"] == 100
    assert doc["status"] == "ok"
    assert len(doc["config_hash"]) == 64
    assert (p.parent / "manifest.json").exists()
    assert json.loads((p.parent / "manifest.json").read_text())["d"] == 128


def test_write_refuses_without_a_status(led):
    with pytest.raises(ledger_mod.MissingStatus):
        led.write()


# --------------------------------------------------------------------------- #
# 5.2 -- scripts/render_scoreboard.py
# --------------------------------------------------------------------------- #


def test_the_scoreboard_reproduces_the_true_canary_tally():
    """The prose said "3 canaries, all held". `runs/canary/cycle-04/ledger.json`
    says `inconclusive` -- it wrote the baseline rather than comparing against one.
    2 survived, 1 inconclusive."""
    board = render_scoreboard.build(_REPO / "runs")
    canaries = [r for r in board.rows if r["run_id"].startswith("canary/")]
    assert len(canaries) == 3
    outcomes = sorted(r["outcome"] for r in canaries)
    assert outcomes == ["inconclusive", "survived", "survived"]


def test_the_scoreboard_joins_on_run_id_and_refuses_cycle(tmp_path):
    """`cycle` is not a usable join key: the experiment ledgers run one behind the
    scoreboard's numbering, `cycle-zscore-denominator` collides with
    `canary/cycle-04` on 4, and `cycle-arm-identity` has `cycle: null`."""
    board = render_scoreboard.build(_REPO / "runs")
    assert len(board.rows) == len({r["run_id"] for r in board.rows})
    assert 4 in board.cycle_collisions
    assert sorted(board.cycle_collisions[4]) == [
        "canary/cycle-04",
        "cycle-zscore-denominator",
    ]


def test_the_scoreboard_refuses_the_prose_verdict_for_cycle_4():
    """"cycle 4 survived" is unresolvable: the two ledgers at cycle 4 disagree."""
    ok, why = render_scoreboard.check_claim(_REPO / "runs", "cycle-4", "survived")
    assert ok is False
    assert "join key" in why or "ambiguous" in why


def test_a_claim_that_matches_its_ledger_is_accepted():
    ok, _ = render_scoreboard.check_claim(
        _REPO / "runs", "canary/cycle-08", "survived"
    )
    assert ok is True
    ok, why = render_scoreboard.check_claim(
        _REPO / "runs", "canary/cycle-04", "survived"
    )
    assert ok is False
    assert "inconclusive" in why


def test_every_scoreboard_count_is_a_len_not_a_typed_number():
    board = render_scoreboard.build(_REPO / "runs")
    assert board.tally["total"] == len(board.rows)
    assert board.tally["falsified"] == len(
        [r for r in board.rows if r["outcome"] == "falsified"]
    )
    assert sum(board.tally[k] for k in ("survived", "falsified", "inconclusive")) == (
        board.tally["total"] - board.tally["no_verdict"]
    )


def test_the_rendered_artefact_passes_its_own_audit(tmp_path):
    """§5.2's claim, made checkable: *a number that is not a ledger key cannot
    appear in the output*. If the renderer prints something the audit cannot back,
    one of the two is wrong -- and on the first run it was the audit, which backed
    ledger *values* but not the `len()`-derived counts the board prints."""
    board = render_scoreboard.build(_REPO / "runs")
    text = render_scoreboard.render(board, runs_dir=_REPO / "runs")
    assert render_scoreboard.audit_prose(_REPO / "runs", text) == []


def test_the_audit_flags_a_number_that_is_in_no_ledger():
    """`1.5476` is the fabrication that motivated this script: three documents
    carry it and `grep -rn 1.5476 runs/` returns zero hits."""
    unbacked = render_scoreboard.audit_prose(
        _REPO / "runs", "the hinge-off masked NLL was 1.5476 nats"
    )
    assert "1.5476" in unbacked


def test_the_audit_passes_a_number_that_is_in_a_ledger():
    backed = render_scoreboard.audit_prose(
        _REPO / "runs", "the canary rel_tol is 0.0001"
    )
    assert backed == []


# --------------------------------------------------------------------------- #
# 5.3 -- scripts/mutation_battery.py
# --------------------------------------------------------------------------- #


def test_an_off_gate_failure_makes_a_mutation_unproven():
    """Clause 2 of the stated discipline -- "the mutation must redden *only* it" --
    was enforced by nothing: `off_gate` was computed, stored, printed, and never
    entered the `unproven` filter."""
    rows = [
        {"mutation": "m", "gate": "g", "reddened_gate": True,
         "off_gate": ["tests/test_other.py::test_x"], "off_gate_allowed": []},
    ]
    assert mutation_battery.unproven(rows) == rows


def test_a_declared_coupling_does_not_make_a_mutation_unproven():
    rows = [
        {"mutation": "m", "gate": "g", "reddened_gate": True,
         "off_gate": ["tests/t.py::x"],
         "off_gate_allowed": [("tests/t.py::x", "why")]},
    ]
    assert mutation_battery.unproven(rows) == []


def test_a_mutated_suite_run_does_not_clobber_the_census(monkeypatch):
    """`tests/conftest.py` writes `test-count.json` every session, so an unredirected
    battery leaves the *last mutated run's* census on disk -- `passed=315 failed=1`
    after a green 316-test suite, which is what CI asserts on and what a ledger row
    was written from before the mismatch was caught.

    ⚠️ **The delenv is load-bearing.** Without it this test reads the redirect the
    battery itself put in the subprocess environment, so its own mutation reddens
    nothing and the gate scores `ADDS NOTHING` -- a check shadowed by the harness
    that runs it.
    """
    monkeypatch.delenv("RSR_TEST_COUNT", raising=False)
    env = mutation_battery._suite_env()
    assert env.get("RSR_TEST_COUNT") == mutation_battery.SCRATCH_COUNT
    assert env.get("RSR_TEST_COUNT") != "test-count.json"


def test_the_mutation_table_is_well_formed_without_a_runtime_repair():
    """`MUTATIONS[7]` was field-shifted -- its `gate` was duplicated into `path`,
    and `_fix_derived_entry()` patched it back at runtime. Fix the literal or
    delete the repair; do not leave both.

    ⚠️ **This deliberately does not check that each `old` anchor is present in the
    tree.** The battery applies one mutation at a time and runs *this suite*
    against the mutated tree, so an anchor check here reddens under every mutation
    by construction -- it is a tripwire, not a gate, and it made all 32 mutations
    score `LEAKS` the first time clause 2 was enforced. Staleness is
    `mutation_battery.apply()`'s job and it already raises on it.
    """
    assert not hasattr(mutation_battery, "_fix_derived_entry")
    for m in mutation_battery.MUTATIONS:
        assert m.gate.startswith("test_"), m.name
        assert m.path.endswith(".py"), m.name
        assert (_REPO / m.path).exists(), f"{m.name}: {m.path}"
        assert m.old and m.new and m.old != m.new, m.name
        assert m.why, m.name


# --------------------------------------------------------------------------- #
# 5.4 -- scripts/canary.py
# --------------------------------------------------------------------------- #


def test_a_first_canary_reading_exits_3_not_0():
    """The one script with no exit-3 branch, and the exact 3-collapsing-to-0 shape:
    it wrote `outcome: inconclusive`, returned verdict `baseline`, and exited 0."""
    assert canary.exit_code_for("baseline") == 3
    assert canary.exit_code_for("held") == 0
    assert canary.exit_code_for("MOVED") == 1


def test_a_length_mismatch_is_a_move():
    """Compared over the overlap only, a run that produced 2 of 6 beats reports
    "held"."""
    verdict, _moved, detail = canary.compare(
        [1.0, 0.9, 0.8, 0.7, 0.6, 0.5], [1.0, 0.9]
    )
    assert verdict == "MOVED"
    assert "length" in detail.lower()


def test_an_equal_length_match_still_holds():
    verdict, moved, _ = canary.compare([1.0, 0.9], [1.0, 0.9])
    assert verdict == "held"
    assert moved == []


def test_a_beat_outside_tolerance_is_a_move():
    verdict, moved, _ = canary.compare([1.0, 0.9], [1.0, 0.95])
    assert verdict == "MOVED"
    assert len(moved) == 1
