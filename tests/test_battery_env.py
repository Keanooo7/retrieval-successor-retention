"""I6 (PLAN-v4 §4; red-team m-2): the battery records every ``RSR_*`` variable in its
environment and refuses to run on one that is not declared.

On 2026-09-26/27 the battery's verdicts changed with an ambient
``RSR_BATTERY_THREADS``, and the only thing the battery recorded was the thread
source. It passes its whole environment to pytest, so every other ``RSR_*``
variable was an input nobody wrote down. Guards, each with a mutation in
``scripts/mutation_battery.py`` (block ``# --- I6: RSR_* environment ---``):

* a stray ``RSR_FOO=1`` is refused with exit 3 and named, before any suite runs;
* a declared variable is recorded with its value, in the stdout header and in
  every row of the ``--json`` record;
* every ``RSR_*`` name the tree reads is declared (the inventory), so the list
  cannot go stale the way the red team's "twelve" did;
* what the battery sets per shard and what ``orchestrator.slot`` sets when it
  launches the battery are declared: neither launch is refused.

⚠️ Every test deletes the ambient ``RSR_*`` variables first: under the battery this
suite runs with the shard's own ``RSR_ORCH_ROOT``, ``RSR_BATTERY_PROBE`` and
``RSR_TEST_COUNT`` set.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "tests"))

import battery_env  # noqa: E402
import battery_isolation as iso  # noqa: E402
import mutation_battery as mb  # noqa: E402
from orchestrator import lanes  # noqa: E402

from _battery_fakes import use_fake_shard  # noqa: E402
from test_orch_lanes import make_root  # noqa: E402


@pytest.fixture(autouse=True)
def _no_ambient_rsr(monkeypatch):
    for name in [k for k in os.environ if k.startswith("RSR_")]:
        monkeypatch.delenv(name)
    for v in lanes.THREAD_ENV_VARS:
        monkeypatch.delenv(v, raising=False)


def _no_suite(*a, **k):
    raise AssertionError("a suite ran: the environment check must come first")


def _one_mutation(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    monkeypatch.setattr(mb, "ROOT", tmp_path)
    monkeypatch.setattr(
        mb,
        "MUTATIONS",
        (mb.Mutation("m", "test_gate", "src/a.py", "x = 1", "x = 2", "w"),),
    )
    use_fake_shard(monkeypatch, mb, tmp_path)


# ------------------------------------------------------------------ the refusal


def test_stray_variable_is_refused_with_exit_3_and_named(monkeypatch, capsys):
    monkeypatch.setenv("RSR_FOO", "1")
    with pytest.raises(SystemExit) as e:
        battery_env.check()
    assert e.value.code == 3
    err = capsys.readouterr().err
    assert err.startswith("DID NOT RUN: ")
    assert "RSR_FOO='1'" in err


def test_stray_variable_names_every_offender_and_the_nearest_declared_name(
    monkeypatch, capsys
):
    # the typo this gate exists for: the operator believes the cap is set
    monkeypatch.setenv("RSR_BATTERY_THREAD", "4")
    monkeypatch.setenv("RSR_FOO", "1")
    monkeypatch.setenv("RSR_GIT", "/opt/homebrew/bin/git")  # declared: not named
    with pytest.raises(SystemExit) as e:
        battery_env.check()
    assert e.value.code == 3
    err = capsys.readouterr().err
    assert "RSR_BATTERY_THREAD='4' (did you mean RSR_BATTERY_THREADS?)" in err
    assert "RSR_FOO='1'" in err
    assert "RSR_GIT" not in err


@pytest.mark.parametrize("mode", [[], ["--check"], ["--check-anchors"]])
def test_stray_variable_stops_main_before_any_suite(tmp_path, monkeypatch, capsys, mode):
    _one_mutation(tmp_path, monkeypatch)
    monkeypatch.setattr(mb, "run_suite", _no_suite)
    monkeypatch.setattr(mb, "apply", _no_suite)
    monkeypatch.setattr(mb, "open_workspace", _no_suite)
    monkeypatch.setenv("RSR_FOO", "1")
    monkeypatch.setattr(sys, "argv", ["mutation_battery.py", *mode])
    with pytest.raises(SystemExit) as e:
        mb.main()
    assert e.value.code == 3
    assert "RSR_FOO='1'" in capsys.readouterr().err
    assert (tmp_path / "src" / "a.py").read_text() == "x = 1\n"


# ------------------------------------------------------------------- the record


def test_declared_variable_is_recorded_with_its_value(monkeypatch):
    monkeypatch.setenv("RSR_BATTERY_THREADS", "4")
    monkeypatch.setenv("RSR_GIT", "/opt/homebrew/bin/git")
    monkeypatch.setenv("NOT_RSR", "x")
    env = battery_env.check()
    assert env == {"RSR_BATTERY_THREADS": "4", "RSR_GIT": "/opt/homebrew/bin/git"}
    lines = battery_env.header(env).splitlines()
    assert lines[0].startswith("RSR_* environment: 2 set")
    assert any(ln.startswith("  RSR_BATTERY_THREADS='4'") for ln in lines)
    assert any(ln.startswith("  RSR_GIT='/opt/homebrew/bin/git'") for ln in lines)


def test_declared_variable_is_recorded_as_none_when_none_is_set():
    assert battery_env.check() == {}
    assert battery_env.header({}) == "RSR_* environment: none set"
    assert battery_env.stamp([{"mutation": "m"}]) == [{"mutation": "m", "rsr_env": {}}]


def test_declared_variable_is_recorded_in_the_header_and_every_json_row(
    tmp_path, monkeypatch, capsys
):
    _one_mutation(tmp_path, monkeypatch)
    runs = iter([{}, {"tests/t.py::test_gate": "AssertionError: gate"}])
    monkeypatch.setattr(mb, "run_suite", lambda *a: next(runs))
    monkeypatch.setenv("RSR_BATTERY_THREADS", "4")
    out = tmp_path / "v.json"
    monkeypatch.setattr(sys, "argv", ["mutation_battery.py", "--json", str(out)])
    assert mb.main() == mb.Exit.OK
    stdout = capsys.readouterr().out
    assert "RSR_* environment: 1 set" in stdout
    assert "  RSR_BATTERY_THREADS='4'" in stdout
    # the header comes before the first verdict line
    assert stdout.index("RSR_* environment") < stdout.index("PROVEN")
    (row,) = json.loads(out.read_text())
    assert row["rsr_env"] == {"RSR_BATTERY_THREADS": "4"}
    assert row["verdict"] == "PROVEN"


# ----------------------------------------------------------------- the inventory


def test_inventory_every_rsr_name_the_tree_reads_is_declared():
    """The declared list is measured, not copied: a name that appears in ``src/``,
    ``scripts/`` or ``experiments/`` and is not declared fails here, naming where."""
    found = battery_env.inventory(_REPO)
    assert len(found) > 12, (
        found
    )  # the red team's count; a scan that finds less is broken
    assert "RSR_BATTERY_THREADS" in found and "RSR_LEDGER" in found
    missing = {
        name: sites[:3]
        for name, sites in found.items()
        if not battery_env.is_declared(name) and name not in battery_env.NOT_VARIABLES
    }
    assert missing == {}, (
        f"RSR_* names read in the tree but not declared in scripts/battery_env.py: "
        f"{missing}"
    )


def test_inventory_finds_a_name_by_file_and_line(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "a.py").write_text('import os\nos.environ["RSR_NEW"]\n')
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "t.py").write_text('"RSR_ONLY_IN_A_TEST"\n')
    assert battery_env.inventory(tmp_path) == {"RSR_NEW": ["scripts/a.py:2"]}


# ------------------------------------------ what the battery and the slot set


def test_launch_what_the_battery_sets_per_shard_is_declared(tmp_path, monkeypatch):
    """A battery nested in a shard's suite (``tests/test_battery_isolation.py``)
    inherits the shard env; it must not be refused by its own parent's variables."""
    monkeypatch.setenv("RSR_BATTERY_THREADS", "2")  # so the thread variables are set
    shard = iso.Shard(tmp_path, "0" * 40, 0, tmp_path / "probe.json")
    suite = iso.shard_env(shard, mb._suite_env())
    set_by_battery = {k for k in suite if k.startswith("RSR_")} - {"RSR_BATTERY_THREADS"}
    assert set_by_battery == {
        "RSR_BATTERY_PROBE",
        "RSR_ORCH_ROOT",
        "RSR_TEST_COUNT",
        "RSR_TORCH_THREADS",
    }
    assert battery_env.undeclared(suite) == []
    assert set(battery_env.REPLACED_IN_SUITE) <= set_by_battery


def test_launch_through_the_slot_wrapper_is_not_refused(tmp_path):
    """The integration battery is launched with ``orchestrator.slot run --lane
    battery``; the wrapper forces ``RSR_TORCH_THREADS`` on its child."""
    root = make_root(tmp_path)
    code = (
        "import json, sys; sys.path.insert(0, sys.argv[1]); import battery_env; "
        "print(json.dumps(battery_env.check()))"
    )
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith("RSR_")},
        "RSR_ORCH_ROOT": str(root),
        "PYTHONPATH": str(_REPO / "scripts"),
    }
    r = subprocess.run(
        [sys.executable, "-m", "orchestrator.slot", "run", "--lane", "battery", "--"]
        + [sys.executable, "-c", code, str(_REPO / "scripts")],
        env=env,
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    seen = json.loads(r.stdout)
    assert seen["RSR_ORCH_ROOT"] == str(root)
    assert int(seen["RSR_TORCH_THREADS"]) >= 1
