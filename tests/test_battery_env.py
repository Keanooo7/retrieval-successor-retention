"""I6 (PLAN-v4 §4; red-team m-2): the battery records every ``RSR_*`` variable in its
environment and refuses to run on one that is not declared.

On 2026-09-26/27 the battery's verdicts changed with an ambient
``RSR_BATTERY_THREADS``, and the only thing the battery recorded was the thread
source. It passes its whole environment to pytest, so every other ``RSR_*``
variable was an input nobody wrote down. Guards, each with a mutation in
``scripts/mutation_battery.py`` (block ``# --- I6: RSR_* environment ---``):

* a stray ``RSR_FOO=1`` is refused with exit 3 and named, before any suite runs --
  by ``main()`` and by ``_run_isolated`` for a driver that does not go through it;
* ``RSR_ORCH_CMD_<MODULE>`` is declared only for a module that exists;
* a declared variable is recorded with its value: the battery's own environment
  and the one each shard's pytest receives, in stdout and on every ``--json`` row;
* the suite header marks what the battery replaced;
* a row's recorded environment is never overwritten by a later stamp;
* every ``RSR_*`` name the tree reads is declared (the inventory).

Fix round 1 (review of 41a9f2b, BLOCKER-2): these tests must not duplicate gates
that already have mutations -- which names the battery puts in the suite env
(``test_the_suite_env_is_the_shards_own``,
``test_the_env_override_caps_every_thread_pool``, the census redirect) and that
the slot forces the thread env
(``test_cpu_det_forces_exactly_slots_threads``). They assert only what I6 adds:
whatever is set is declared, and what is recorded is what was set.

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


def _one_mutation(tmp_path, monkeypatch) -> iso.Shard:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    monkeypatch.setattr(mb, "ROOT", tmp_path)
    monkeypatch.setattr(
        mb,
        "MUTATIONS",
        (mb.Mutation("m", "test_gate", "src/a.py", "x = 1", "x = 2", "w"),),
    )
    return use_fake_shard(monkeypatch, mb, tmp_path)


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


@pytest.mark.parametrize(
    "name", ["RSR_ORCH_CMD_NO_SUCH_MODULE", "RSR_ORCH_CMD_slot", "RSR_ORCH_CMD_"]
)
def test_stray_variable_orch_cmd_for_no_module_is_refused(monkeypatch, name):
    """Review MINOR-5: `loopcore.orch_cmd` reads ``RSR_ORCH_CMD_`` + the module name
    upper-cased; any other suffix stubs nothing and is a typo."""
    assert battery_env.check({"RSR_ORCH_CMD_SLOT": "x"}) == {"RSR_ORCH_CMD_SLOT": "x"}
    monkeypatch.setenv(name, "x")
    with pytest.raises(SystemExit) as e:
        battery_env.check()
    assert e.value.code == 3


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


def test_stray_variable_stops_a_driver_that_skips_main(tmp_path, monkeypatch, capsys):
    """A driver that runs shards through ``_run_isolated`` without ``main()`` (I2's
    sharded driver is one) is refused before the baseline suite too."""
    _one_mutation(tmp_path, monkeypatch)
    monkeypatch.setattr(mb, "run_suite", _no_suite)
    monkeypatch.setattr(mb, "apply", _no_suite)
    monkeypatch.setenv("RSR_FOO", "1")
    args = mb.ArgumentParser().parse_args([])
    args.json = args.markdown = None
    with pytest.raises(SystemExit) as e:
        mb._run_isolated(args)
    assert e.value.code == 3
    assert "RSR_FOO='1'" in capsys.readouterr().err


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
    assert battery_env.stamp({"mutation": "m"}, {}, {}) == {
        "mutation": "m",
        "rsr_env": {},
        "rsr_env_suite": {},
    }


def test_declared_variable_is_recorded_in_the_header_and_every_json_row(
    tmp_path, monkeypatch, capsys
):
    shard = _one_mutation(tmp_path, monkeypatch)
    runs = iter([{}, {"tests/t.py::test_gate": "AssertionError: gate"}])
    monkeypatch.setattr(mb, "run_suite", lambda *a: next(runs))
    monkeypatch.setenv("RSR_BATTERY_THREADS", "4")
    out = tmp_path / "v.json"
    monkeypatch.setattr(sys, "argv", ["mutation_battery.py", "--json", str(out)])
    assert mb.main() == mb.Exit.OK
    stdout = capsys.readouterr().out
    assert "RSR_* environment: 1 set" in stdout
    assert "  RSR_BATTERY_THREADS='4'" in stdout
    assert "RSR_* environment of each shard's suite:" in stdout
    # both headers come before the first verdict line
    assert stdout.index("of each shard's suite") < stdout.index("PROVEN")
    (row,) = json.loads(out.read_text())
    assert row["rsr_env"] == {"RSR_BATTERY_THREADS": "4"}
    # what pytest received: built by the same functions run_suite uses, so this
    # holds whatever those functions set (their own gates test that)
    suite = battery_env.present(iso.shard_env(shard, mb._suite_env()))
    assert row["rsr_env_suite"] == suite
    assert suite["RSR_BATTERY_THREADS"] == "4"
    assert row["verdict"] == "PROVEN"


def test_suite_header_marks_what_the_battery_set_replaced_or_removed():
    """Review MAJOR-3: ambient ``RSR_TORCH_THREADS=14`` under a cap of 2 -- the suite
    ran with 2, and the record must say so rather than print 14."""
    own = {"RSR_TORCH_THREADS": "14", "RSR_GIT": "/g", "RSR_LEDGER": "/l"}
    suite = {"RSR_TORCH_THREADS": "2", "RSR_GIT": "/g", "RSR_ORCH_ROOT": "/shard"}
    lines = battery_env.suite_header(own, suite).splitlines()
    assert lines[0] == "RSR_* environment of each shard's suite: 3 set"
    assert (
        "  RSR_TORCH_THREADS='2' [replaced by the battery; its own value is '14']"
        in lines
    )
    assert "  RSR_ORCH_ROOT='/shard' [set by the battery]" in lines
    assert "  RSR_GIT='/g'" in lines
    assert "  RSR_LEDGER [removed by the battery; its own value is '/l']" in lines


def test_stamp_keeps_a_rows_existing_environment():
    """Review MAJOR-4: a driver merging rows from several batteries must not stamp
    its own environment over each row's."""
    row = {"mutation": "m", "rsr_env": {"RSR_BATTERY_THREADS": "2"}}
    got = battery_env.stamp(row, {"RSR_GIT": "/merge"}, {"RSR_GIT": "/merge"})
    assert got["rsr_env"] == {"RSR_BATTERY_THREADS": "2"}
    assert got["rsr_env_suite"] == {"RSR_GIT": "/merge"}
    assert row == {"mutation": "m", "rsr_env": {"RSR_BATTERY_THREADS": "2"}}


# ----------------------------------------------------------------- the inventory


def test_inventory_every_rsr_name_the_tree_reads_is_declared():
    """The declared list is measured, not copied: a name that appears in `SCANNED`
    and is not declared fails here, naming where."""
    found = battery_env.inventory(_REPO)
    # 12 was the red team's count (m-2); a scan that finds fewer is broken
    assert len(found) > 12, found
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
    (tmp_path / "tests" / "test_t.py").write_text('"RSR_ONLY_IN_A_TEST"\n')
    (tmp_path / "tests" / "_helper.py").write_text('"RSR_IN_A_HELPER"\n')
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "x.env").write_text("# c\nRSR_IN_OPS=1\n")
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "settings.local.json").write_text('"RSR_LOCAL"\n')
    assert battery_env.inventory(tmp_path) == {
        "RSR_IN_A_HELPER": ["tests/_helper.py:1"],
        "RSR_IN_OPS": ["ops/x.env:2"],
        "RSR_NEW": ["scripts/a.py:2"],
    }


# ------------------------------------------ what the battery and the slot set


def test_launch_what_the_battery_sets_per_shard_is_declared(tmp_path, monkeypatch):
    """A battery nested in a shard's suite (``tests/test_battery_isolation.py``)
    inherits the shard env; it must not be refused by its own parent's variables.

    Only "declared" is asserted. WHICH names the battery sets is gated elsewhere
    (see the module docstring). No mutation makes this red on its own: a battery-set
    name that is undeclared refuses every in-process ``main()`` in the suite too.
    """
    monkeypatch.setenv("RSR_BATTERY_THREADS", "2")  # so the thread variables are set
    shard = iso.Shard(tmp_path, "0" * 40, 0, tmp_path / "probe.json")
    suite = iso.shard_env(shard, mb._suite_env())
    assert battery_env.undeclared(suite) == []


def test_launch_through_the_slot_wrapper_is_not_refused(tmp_path):
    """The integration battery is launched with ``orchestrator.slot run --lane
    battery``; whatever the wrapper puts in its child's environment is declared."""
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
    child = [sys.executable, "-c", code, str(_REPO / "scripts")]
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "orchestrator.slot",
            "run",
            "--lane",
            "battery",
            "--",
            *child,
        ],
        env=env,
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert isinstance(json.loads(r.stdout), dict)
