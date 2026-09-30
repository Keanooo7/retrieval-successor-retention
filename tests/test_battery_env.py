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

import contextlib
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
    """Defence in depth: a driver that bypasses ``main()`` and calls
    ``_run_isolated`` directly is refused before the baseline suite too. (No driver
    in the tree does that today; I2's ``run_shard`` calls ``mb.main()`` -- second
    review, MINOR-4.)"""
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
    # what pytest received, with the shard's paths as tokens; built by the same
    # function run_suite uses, so this holds whatever it sets (its own gates test
    # which names)
    suite = battery_env.portable(
        battery_env.present(mb.suite_env(shard)), shard.root, shard.probe
    )
    assert row["rsr_env_suite"] == suite
    assert suite["RSR_BATTERY_THREADS"] == "4"
    assert row["verdict"] == "PROVEN"


def test_declared_variable_is_recorded_as_the_env_the_suite_got(tmp_path, monkeypatch):
    """Second review, MINOR-3: every suite gets the one environment object
    ``_run_isolated`` built and recorded -- not a rebuild that can drift (an
    ``ops/lanes.json`` edited mid-run changed the mutation's thread count while the
    record kept the baseline's)."""
    shard = _one_mutation(tmp_path, monkeypatch)
    got = []

    def fake_run_suite(shard_, env=None):
        got.append(env)
        return {} if len(got) == 1 else {"tests/t.py::test_gate": None}

    monkeypatch.setattr(mb, "run_suite", fake_run_suite)
    out = tmp_path / "v.json"
    monkeypatch.setattr(sys, "argv", ["mutation_battery.py", "--json", str(out)])
    assert mb.main() == mb.Exit.OK
    assert len(got) == 2 and got[0] is not None and got[1] is got[0]
    (row,) = json.loads(out.read_text())
    assert row["rsr_env_suite"] == battery_env.portable(
        battery_env.present(got[1]), shard.root, shard.probe
    )


def test_rows_do_not_depend_on_where_the_shard_lives(tmp_path, monkeypatch, capsys):
    """Second review, MAJOR-2 (the manager's decision): two batteries at one SHA
    whose shards live at different paths produce equal rows for the same failing
    set.

    Third review, MAJOR-1: the shard is laid out as `battery_isolation.make_shard`
    lays it out -- root ``<pool>/shard-0``, probe ``<pool>/probe-0.json``, OUTSIDE
    the root. ``use_fake_shard`` puts the probe inside the root, where the
    ``<shard>`` token swallows it and the ``<probe>`` token goes untested."""
    rows = []
    for name in ("pool-a", "pool-b"):
        pool = tmp_path / name / "pool"
        root = pool / "shard-0"
        root.mkdir(parents=True)
        _one_mutation(root, monkeypatch)
        shard = iso.Shard(root, "0" * 40, 0, pool / "probe-0.json")
        monkeypatch.setattr(
            mb,
            "open_workspace",
            lambda args, shard=shard: contextlib.nullcontext([shard]),
        )
        runs = iter([{}, {"tests/t.py::test_gate": "AssertionError: gate"}])
        monkeypatch.setattr(mb, "run_suite", lambda *a, runs=runs: next(runs))
        out = root / "v.json"
        monkeypatch.setattr(sys, "argv", ["mutation_battery.py", "--json", str(out)])
        assert mb.main() == mb.Exit.OK
        rows.append(json.loads(out.read_text()))
    assert rows[0] == rows[1]
    text = json.dumps(rows[0])
    assert str(tmp_path) not in text
    # the real paths are still in stdout, for the reader of this run
    assert str(tmp_path / "pool-b") in capsys.readouterr().out


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
    # Both keys pre-set, so this does not also test how a fresh stamp is written
    # (that is test_declared_variable_is_recorded*'s gate; fix round 1 simulation).
    shard_env = {"rsr_env": {"RSR_GIT": "/s"}, "rsr_env_suite": {"RSR_GIT": "/s"}}
    row = {"mutation": "m", **shard_env}
    got = battery_env.stamp(row, {"RSR_GIT": "/merge"}, {"RSR_GIT": "/merge"})
    assert got == row
    assert row == {"mutation": "m", **shard_env}


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


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        [
            iso._git_bin(),
            "-C",
            str(cwd),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _repo(root: Path) -> Path:
    files = {
        "scripts/a.py": 'import os\nos.environ["RSR_NEW"]\n',
        "tests/test_t.py": '"RSR_ONLY_IN_A_TEST"\n',
        "tests/_helper.py": '"RSR_IN_A_HELPER"\n',
        "ops/x.env": "# c\nRSR_IN_OPS=1\n",
        ".claude/settings.json": '"RSR_IN_SETTINGS"\n',
        # the repository's own ignore rules, verbatim: no rule invented for the test
        # (fourth review, MINOR-1)
        ".gitignore": (_REPO / ".gitignore").read_text(),
    }
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fixture")
    return root


_TRACKED = {
    "RSR_IN_A_HELPER": ["tests/_helper.py:1"],
    "RSR_IN_OPS": ["ops/x.env:2"],
    "RSR_IN_SETTINGS": [".claude/settings.json:1"],
    "RSR_NEW": ["scripts/a.py:2"],
}


def test_inventory_finds_a_name_by_file_and_line(tmp_path):
    assert battery_env.inventory(_repo(tmp_path / "repo")) == _TRACKED


def _checkout_and_worktree(tmp_path: Path) -> tuple[Path, Path]:
    repo = _repo(tmp_path / "repo")
    _git(repo, "worktree", "add", "-q", "--detach", str(tmp_path / "shard"), "HEAD")
    assert (tmp_path / "shard" / ".git").is_file()
    return repo, tmp_path / "shard"


def _plant(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)


_AGENT_COPY = ".claude/worktrees/agent-x/scripts/battery_env.py"


def test_inventory_skips_ignored_files_in_a_checkout_and_a_worktree(tmp_path):
    """Second review, MAJOR-1: files the repository's ``.gitignore`` ignores are
    never read -- an agent checkout under ``.claude/worktrees/`` (Claude Code's
    ``isolation: worktree``), and, in the code directories where untracked files ARE
    read, a stray ``test-count.json`` and a ``__pycache__`` -- in a main checkout,
    and in a worktree whose ``.git`` is a file, as a battery shard is."""
    for root in _checkout_and_worktree(tmp_path):
        _plant(
            root,
            {
                _AGENT_COPY: '"RSR_BATTERY_THREAD"\n',
                "scripts/test-count.json": '"RSR_IGNORED_CENSUS"\n',
                "src/__pycache__/a.py": '"RSR_IGNORED_CACHE"\n',
            },
        )
        assert battery_env.inventory(root) == _TRACKED, root


def test_inventory_skips_untracked_claude_files_without_global_excludes(
    tmp_path, monkeypatch
):
    """Fourth review, MINOR-1: an untracked ``.claude/settings.local.json`` holding
    an ``RSR_*`` typo is not read even on a machine whose git has no global excludes
    file -- the repository does not ignore it, so reading untracked ``.claude/``
    files made the verdict depend on the machine."""
    empty = tmp_path / "empty-home"
    empty.mkdir()
    monkeypatch.setenv("HOME", str(empty))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(empty))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for root in _checkout_and_worktree(tmp_path):
        local = ".claude/settings.local.json"
        _plant(root, {local: '{"note": "RSR_BATTERY_THREAD"}\n'})
        # not vacuous: in this environment git does NOT ignore the file
        probe = subprocess.run(
            [iso._git_bin(), "-C", str(root), "check-ignore", "-q", local],
            capture_output=True,
        )
        assert probe.returncode == 1, probe
        assert battery_env.inventory(root) == _TRACKED, root


@pytest.mark.skipif(
    not os.path.exists(str(Path(__file__).resolve()).swapcase()),
    reason="case-sensitive filesystem: a differently-cased root is another path "
    "here, so this APFS-only check cannot run (NOT a pass)",
)
def test_tracked_accepts_a_differently_cased_root(tmp_path):
    """Fourth review, MINOR-2: on a case-insensitive filesystem (the Studio's APFS)
    a root spelled with different case is the same checkout."""
    repo = _repo(tmp_path / "repo")
    other = Path(str(repo).swapcase())
    assert str(other) != str(repo) and other.exists()
    assert battery_env.tracked(other) == battery_env.tracked(repo)


def test_inventory_reads_a_new_untracked_file(tmp_path):
    """Third review, MINOR-3: a new file that reads an ``RSR_*`` name is inventoried
    before it is ``git add``ed, so the author's own pre-commit run goes red."""
    for root in _checkout_and_worktree(tmp_path):
        _plant(root, {"scripts/untracked.py": '"RSR_UNTRACKED"\n'})
        assert battery_env.inventory(root) == {
            **_TRACKED,
            "RSR_UNTRACKED": ["scripts/untracked.py:1"],
        }, root


def test_inventory_refuses_a_tree_that_is_not_a_checkout(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "a.py").write_text('"RSR_NEW"\n')
    with pytest.raises(RuntimeError, match="git rev-parse"):
        battery_env.inventory(tmp_path)


def test_inventory_refuses_a_directory_inside_a_checkout(tmp_path):
    """Third review, MINOR-2: inside a checkout, ``ls-files`` exits 0 and lists
    nothing -- for a subdirectory, and for a copy of the tree under an ignored
    path. Both are refused, not inventoried as empty."""
    repo = _repo(tmp_path / "repo")
    copy = repo / ".claude" / "worktrees" / "copy"
    _plant(copy, {"scripts/a.py": '"RSR_NEW"\n'})
    for root in (repo / "scripts", copy):
        with pytest.raises(RuntimeError, match="not the top of a checkout"):
            battery_env.inventory(root)


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
    battery``; whatever the wrapper puts in its child's environment is declared.

    No mutation in the table reddens this test (fix round 1 simulation; second
    review, MINOR-5): the names the slot sets appear literally in the tree, so the
    inventory test already implies it. It is kept as the named diagnosis for a
    refused slot launch."""
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
