"""S0-05: the exit-code protocol, converted from a lesson into a gate.

`docs/gates.md` specifies five codes -- `0` ok, `1` real failure, `2` nothing to
compare, `3` did not run, `4` unbanked rise -- and says *"`2` and `3` are separate
deliberately, and `3` is where this design can be silently defeated."* Until S0-05
the protocol was stated in six files and enforced by none (`docs/ROADMAP.md` §6):

* `scripts/canary.py` mapped a first reading to `3`. It ran; it had nothing to
  compare. That is `2`. The test that pinned `3` was written by the same hand and
  mutated only against the older `3 -> 0` defect, so it could not see the value
  actually written.
* the same script wrote its ledger row as the literal `exit_code=0` *before* the
  verdict existed, so a MOVED canary exiting 1 filed a ledger saying 0.
* seven `experiments/e0*/run.py` stubs raised `NotImplementedError`: exit `1`.
* `scripts/mutation_battery.py` reported a stale anchor and a red baseline -- the
  battery did not run -- as `1`, via bare `raise SystemExit("...")`.
* `scripts/extract_golden_tensors.py` exits `1` on the ImportError it is guaranteed
  to hit in the project venv, where JAX is correctly absent.

Every test here has a mutation in `scripts/mutation_battery.py` that reddens it
and nothing else.
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

import canary  # noqa: E402
import ledger as ledger_mod  # noqa: E402
import mutation_battery  # noqa: E402
import render_scoreboard  # noqa: E402

from rsr.exit_codes import Exit, status  # noqa: E402

PY = sys.executable


# --------------------------------------------------------------------------- #
# the enum and the result helper
# --------------------------------------------------------------------------- #


def test_the_enum_is_the_five_code_protocol():
    """Five, not four: `canary.py`'s docstring listed four and dropped `4`."""
    assert {e.name: int(e) for e in Exit} == {
        "OK": 0,
        "FAIL": 1,
        "UNKNOWN": 2,
        "DID_NOT_RUN": 3,
        "UNBANKED_RISE": 4,
    }
    # `floors.py`'s name on macbook-local-2026-09-18, so the port can reuse this.
    assert Exit.DROP is Exit.FAIL


def test_status_refuses_a_bool():
    """`True` is `1` and `False` is `0`: a checker returning `ok` exits 1 on
    success, and one returning `not failed` has two states where there are five."""
    for b in (True, False):
        with pytest.raises(TypeError, match="bool"):
            status(b)


def test_status_refuses_none_and_codes_outside_the_protocol():
    """`sys.exit(None)` is 0: a `main()` that falls off its end reads as a pass."""
    with pytest.raises(TypeError, match="None"):
        status(None)
    for bad in (-1, 5, 127):
        with pytest.raises(ValueError, match="outside the protocol"):
            status(bad)
    assert status(3) is Exit.DID_NOT_RUN


# --------------------------------------------------------------------------- #
# scripts/canary.py -- bar item 6: the ledger's exit code is READ, not typed
# --------------------------------------------------------------------------- #


def test_the_canary_ledger_row_records_the_exit_code_it_returns(tmp_path, monkeypatch):
    """The row used to be written above the verdict as the literal `exit_code=0`.

    Asserts only *agreement* -- ledger row == returned code == the verdict's code
    -- across all three verdicts, deliberately not the values. The values are
    `test_a_first_canary_reading_exits_2_nothing_to_compare`'s job; asserting
    them here too would make every mapping mutation redden two tests.
    """
    import rsr.train.loop as loop

    readings = iter([[1.0, 0.9, 0.8], [1.0, 0.9, 0.8], [1.0, 0.5, 0.8]])

    def fake_train(*, out_dir, **_cfg):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        beats = next(readings)
        (out_dir / "heartbeat.jsonl").write_text(
            "".join(json.dumps({"kind": "beat", "loss": x}) + "\n" for x in beats)
        )

    monkeypatch.setattr(loop, "train", fake_train)
    monkeypatch.setattr(canary, "_REPO", tmp_path)
    monkeypatch.setattr(canary, "BASELINE", tmp_path / "baseline.json")
    monkeypatch.setattr(
        canary, "Ledger", functools.partial(ledger_mod.Ledger, runs_root=tmp_path / "L")
    )
    monkeypatch.setattr(ledger_mod, "_dirty_source_paths", lambda: [])

    verdicts = []
    for cycle in (1, 2, 3):
        r = canary.run(cycle)
        verdicts.append(r["verdict"])
        doc = json.loads(Path(r["ledger"]).read_text())
        (row,) = doc["commands"]
        assert row["exit_code"] == int(r["exit_code"]), (r["verdict"], row)
        assert r["exit_code"] == canary.exit_code_for(r["verdict"])
    assert verdicts == ["baseline", "held", "MOVED"]


# --------------------------------------------------------------------------- #
# the seven experiment stubs -- bar item 5
# --------------------------------------------------------------------------- #

STUBS = [f"experiments/e0{x}/run.py" for x in "adefghi"]


@pytest.mark.parametrize("stub", STUBS)
def test_an_unimplemented_experiment_exits_3(stub):
    """Not implemented is the definition of *did not run*. An uncaught
    `NotImplementedError` exits 1, which is *real failure*. The process status is
    read from the process, not from `main()`'s return value."""
    proc = subprocess.run([PY, stub], cwd=_REPO, capture_output=True, text=True)
    assert proc.returncode == 3, (stub, proc.returncode, proc.stderr[-400:])
    assert "DID NOT RUN" in proc.stderr


# --------------------------------------------------------------------------- #
# scripts/extract_golden_tensors.py
# --------------------------------------------------------------------------- #


def test_extract_golden_tensors_without_jax_exits_3():
    """In the project venv JAX is absent by design (ADR-0001), so this is the
    one environment where the tool is guaranteed not to run. It exited 1."""
    if importlib.util.find_spec("jax") is not None:
        pytest.skip("jax is importable here, so the did-not-run path is not reachable")
    proc = subprocess.run(
        [PY, "scripts/extract_golden_tensors.py", "--out", "unused.npz"],
        cwd=_REPO,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 3, (proc.returncode, proc.stderr[-400:])
    assert not (_REPO / "unused.npz").exists()


# --------------------------------------------------------------------------- #
# scripts/mutation_battery.py
# --------------------------------------------------------------------------- #


def test_a_stale_battery_anchor_exits_3():
    """Nothing was mutated, so nothing was tested: the battery did not run."""
    m = mutation_battery.Mutation(
        "stale", "test_x", "scripts/canary.py", "<<< no such anchor >>>", "y", "why"
    )
    with pytest.raises(SystemExit) as exc:
        mutation_battery.apply(m)
    assert exc.value.code == 3


def test_a_red_baseline_exits_3(monkeypatch):
    """A suite red before mutating means no mutation ran."""
    monkeypatch.setattr(mutation_battery, "run_suite", lambda: {"tests/t.py::t"})
    monkeypatch.setattr(sys, "argv", ["mutation_battery.py", "--check"])
    with pytest.raises(SystemExit) as exc:
        mutation_battery.main()
    assert exc.value.code == 3


def test_a_battery_with_no_mutations_exits_2(monkeypatch):
    """ "0/0 proven" contains no unproven gate and used to exit 0. It is nothing
    to compare."""

    def must_not_run():
        raise AssertionError("the suite ran with no mutations to apply")

    monkeypatch.setattr(mutation_battery, "MUTATIONS", ())
    monkeypatch.setattr(mutation_battery, "run_suite", must_not_run)
    monkeypatch.setattr(sys, "argv", ["mutation_battery.py", "--check"])
    assert mutation_battery.main() == 2


# --------------------------------------------------------------------------- #
# scripts/render_scoreboard.py -- the ambiguous `2`, decided
# --------------------------------------------------------------------------- #


def test_a_usage_error_exits_3_not_2():
    """argparse exits 2 on bad arguments; in this protocol 2 is *nothing to
    compare*, which render_scoreboard also returns for an empty board. One code,
    two meanings. A usage error is *did not run*."""
    with pytest.raises(SystemExit) as exc:
        render_scoreboard.main(["--no-such-flag"])
    assert exc.value.code == 3


def test_an_empty_board_is_2(tmp_path):
    """The repo's one correct `2` before S0-05, pinned now that it is the only
    meaning `2` has in this script."""
    assert render_scoreboard.main(["--over", str(tmp_path)]) == 2


# --------------------------------------------------------------------------- #
# static: every entry point, by AST
# --------------------------------------------------------------------------- #


def _entry_points() -> list[Path]:
    out = []
    for top in ("scripts", "experiments", "src/rsr"):
        for p in sorted((_REPO / top).rglob("*.py")):
            if '__name__ == "__main__"' in p.read_text():
                out.append(p)
    return out


def _main_def(tree: ast.Module) -> ast.FunctionDef | None:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node
    return None


def _returns(fn: ast.FunctionDef):
    """Return statements of `fn` itself, not of functions nested inside it."""
    stack = list(fn.body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            continue
        if isinstance(node, ast.Return):
            yield node
        stack.extend(ast.iter_child_nodes(node))


#: Helpers that return an `Exit`.
_EXIT_HELPERS = {"did_not_run", "not_implemented", "status"}


def _is_protocol_value(node: ast.AST | None) -> bool:
    """An expression that is statically a protocol code: `Exit.X`, an int literal
    0-4, a helper call, or a conditional between two of those. A bare name is NOT
    -- `return ok` is exactly how a bool reaches `sys.exit`."""
    if node is None:
        return False  # bare `return` -> None -> exit 0
    if isinstance(node, ast.Attribute):
        return isinstance(node.value, ast.Name) and node.value.id == "Exit"
    if isinstance(node, ast.Constant):
        return type(node.value) is int and 0 <= node.value <= 4
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return node.func.id in _EXIT_HELPERS
    if isinstance(node, ast.IfExp):
        return _is_protocol_value(node.body) and _is_protocol_value(node.orelse)
    return False


def test_no_checker_returns_a_bare_boolean():
    """`docs/ROADMAP.md` §6's conversion row: *"a test that no checker returns a
    bare boolean."* Covers EVERY entry point under scripts/, experiments/ and
    src/rsr/, including the ones S0-05 did not convert: a bool has two states and
    the protocol has five, and `True` exits 1."""
    bad = []
    for p in _entry_points():
        rel = p.relative_to(_REPO).as_posix()
        main = _main_def(ast.parse(p.read_text()))
        if main is None:
            continue
        for ret in _returns(main):
            if not _is_protocol_value(ret.value):
                src = ast.unparse(ret.value) if ret.value is not None else "None"
                bad.append(f"{rel}:{ret.lineno}: return {src}")
    assert not bad, "main() returns something that is not a protocol code:\n" + (
        "\n".join(bad)
    )


#: Entry points S0-05 did not convert, each with the reason. The test below
#: refuses a stale entry, so a converted or deleted file cannot stay exempt.
NOT_CONVERTED = {
    "src/rsr/train/loop.py": "S0-05 brief: do not touch; S0-03 is editing it",
    "src/rsr/cli.py": (
        "not in S0-05's files in scope. Reports UnmeasuredConstant as 1 where the "
        "protocol says 2 (no recorded value for this key) -- filed as a brief error"
    ),
    "experiments/s0-02/measure_capture_cost.py": "not in S0-05's files in scope",
    "experiments/s0-02/measure_qtok_collapse.py": (
        "not in S0-05's files in scope; three precondition refusals exit 1 via bare "
        "`raise SystemExit(msg)` -- filed as a brief error"
    ),
}


def _protocol_violations(p: Path) -> list[str]:
    text = p.read_text()
    tree = ast.parse(text)
    out = []
    if "argparse.ArgumentParser(" in text:
        out.append("uses argparse.ArgumentParser, whose usage error exits 2")
    main_block = [
        n
        for n in tree.body
        if isinstance(n, ast.If) and ast.unparse(n.test) == "__name__ == '__main__'"
    ]
    if not any("run_main(" in ast.unparse(n) for n in main_block):
        out.append("the __main__ block does not call run_main")
    main = _main_def(tree)
    if main is not None:
        for ret in _returns(main):
            if isinstance(ret.value, ast.Constant):
                out.append(f":{ret.lineno} returns the literal {ret.value.value!r}")
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "SystemExit"
            and node.args
            and isinstance(node.args[0], ast.Constant | ast.JoinedStr)
        ):
            out.append(
                f":{node.lineno} raise SystemExit({ast.unparse(node.args[0])[:40]}...) "
                f"-- a message exits 1 whatever it says"
            )
    return out


def test_every_converted_checker_exits_through_the_protocol():
    """The enum is the deliverable only if the checkers use it: `run_main` at the
    entry point, `Exit` members instead of literals, no message-only
    `SystemExit`, and a parser whose usage error is `3`."""
    eps = {p.relative_to(_REPO).as_posix(): p for p in _entry_points()}
    stale = sorted(set(NOT_CONVERTED) - set(eps))
    assert not stale, f"NOT_CONVERTED names files that are not entry points: {stale}"
    bad = {
        rel: v
        for rel, p in eps.items()
        if rel not in NOT_CONVERTED and (v := _protocol_violations(p))
    }
    assert not bad, json.dumps(bad, indent=2)
