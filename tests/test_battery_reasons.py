"""The battery records a one-line failure reason per failing node (PLAN-v4 I5).

On 2026-09-26 a battery shard went red on
`test_orch_dispatch.py::test_submit_launches_the_slot_detached_at_the_pinned_sha`,
and nothing survives of it but the node id: `run_suite` ran pytest with `--tb=no`,
so W4-fix.md:68 could not say whether it was a torn read or a timeout, and I5's 30
reproduction attempts found nothing (docs/lab-notes/i5-flake.md). The next
occurrence has to be capturable from the verdict JSON alone.

🔴 **Reasons are recorded; they never enter a verdict.** The node set is parsed
exactly as before -- the reason is extra data on the row -- and
`docs/lab-notes/i5-flake.md` records a 3-mutation run of the battery's own `main()`
giving identical verdicts before and after this change.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

import mutation_battery as mb  # noqa: E402

from _battery_fakes import use_fake_shard  # noqa: E402

# Real pytest 9.1 output under `-q --no-header -rfE --tb=line`, with the path
# shortened. Includes a parametrised id with a space in it, a failure whose
# message pytest dropped, a multi-line message, and a setup error.
OUTPUT = """\
.FFEFF                                                                   [100%]
==================================== ERRORS ====================================
__________________________ ERROR at setup of test_err __________________________
E   RuntimeError: fixture broke
=================================== FAILURES ===================================
E   AssertionError: job j1 did not finish in 20.0s
/x/tests/test_x.py:8: AssertionError: job j1 did not finish in 20.0s
E   ValueError: line one
    ERROR two
/x/tests/test_x.py:11: ValueError: line one
=========================== short test summary info ============================
FAILED tests/test_x.py::test_wait - AssertionError: job j1 did not finish in 20.0s
FAILED tests/test_x.py::test_p[a b] - AssertionError: x
FAILED tests/test_x.py::test_multi - ValueError: line one
FAILED tests/test_x.py::test_silent
ERROR tests/test_x.py::test_err - RuntimeError: fixture broke
4 failed, 1 passed, 1 error in 0.01s
"""


def _legacy_nodes(text: str, returncode: int) -> set[str]:
    """`run_suite`'s node parse as it stood at 9436c05, verbatim, for comparison."""
    failing = set()
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("FAILED ") or line.startswith("ERROR "):
            failing.add(line.split(" ", 1)[1].split(" ")[0])
    if not failing and returncode not in (0, 5):
        failing.add(f"<collection/exit {returncode}>")
    return failing


def test_parse_failures_records_a_one_line_reason_per_failing_node():
    got = mb.parse_failures(OUTPUT, 1)
    assert got == {
        "tests/test_x.py::test_wait": "AssertionError: job j1 did not finish in 20.0s",
        # the node id is cut at the first space, exactly as before; the reason
        # still lands on it
        "tests/test_x.py::test_p[a": "AssertionError: x",
        "tests/test_x.py::test_multi": "ValueError: line one",
        # pytest printed no message: recorded as None, never invented
        "tests/test_x.py::test_silent": None,
        "tests/test_x.py::test_err": "RuntimeError: fixture broke",
    }


# The same session as `--tb=no` printed it: no ERRORS/FAILURES bodies.
_LINES = OUTPUT.splitlines()
_SUMMARY = next(i for i, ln in enumerate(_LINES) if "short test summary" in ln)
OUTPUT_TB_NO = "\n".join(_LINES[:1] + _LINES[_SUMMARY:])


@pytest.mark.parametrize(
    ("text", "tb_no_text", "rc"),
    [
        (OUTPUT, OUTPUT_TB_NO, 1),
        ("....                          [100%]\n4 passed in 0.01s\n", None, 0),
        ("no tests ran in 0.01s\n", None, 5),
        ("INTERNALERROR> boom\n", None, 3),
    ],
    ids=["red", "green", "empty", "internal-error"],
)
def test_the_failing_node_set_is_unchanged(text, tb_no_text, rc):
    """The reasons must not move a verdict, so the node SET must be what the
    9436c05 parse gives on the `--tb=no` output of the same session."""
    legacy = _legacy_nodes(tb_no_text if tb_no_text is not None else text, rc)
    assert set(mb.parse_failures(text, rc)) == legacy


def test_the_tb_line_body_cannot_add_a_node():
    """`--tb=line` prints message continuation lines, and a message can contain
    'ERROR '. Only the short summary section names failing nodes."""
    assert "tests/two" not in mb.parse_failures(
        OUTPUT.replace("    ERROR two", "    ERROR tests/two - y"), 1
    )


def test_a_nonzero_exit_with_no_node_records_the_last_line():
    got = mb.parse_failures("collecting ...\nINTERNALERROR> boom\n", 3)
    assert got == {"<collection/exit 3>": "INTERNALERROR> boom"}


def test_run_suite_asks_pytest_for_the_reasons(monkeypatch):
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"], seen["env"] = argv, kw["env"]
        return subprocess.CompletedProcess(argv, 1, OUTPUT, "")

    monkeypatch.setattr(mb.subprocess, "run", fake_run)
    got = mb.run_suite()
    assert "--tb=no" not in seen["argv"]
    assert "--tb=line" in seen["argv"] and "-rfE" in seen["argv"]
    # pytest trims the summary message to the terminal width, and drops it
    # entirely when the node id alone fills 80 columns
    assert int(seen["env"]["COLUMNS"]) >= 1000
    assert got["tests/test_x.py::test_wait"].startswith("AssertionError: job j1")


def test_a_mutation_row_carries_the_failure_reasons(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    monkeypatch.setattr(mb, "ROOT", tmp_path)
    monkeypatch.setattr(
        mb,
        "MUTATIONS",
        (mb.Mutation("m", "test_gate", "src/a.py", "x = 1", "x = 2", "w"),),
    )
    red = {
        "tests/t.py::test_gate": "AssertionError: gate",
        "tests/t.py::test_other": "JSONDecodeError: Expecting value: line 1 column 1",
    }
    runs = iter([{}, red])
    use_fake_shard(monkeypatch, mb, tmp_path)
    monkeypatch.setattr(mb, "run_suite", lambda *a: next(runs))
    out = tmp_path / "v.json"
    monkeypatch.setattr(sys, "argv", ["mutation_battery.py", "--json", str(out)])
    assert mb.main() == mb.Exit.OK
    (row,) = json.loads(out.read_text())
    assert row["failure_reasons"] == red
    # and the verdict is computed from the node ids exactly as before
    assert row["verdict"] == "LEAKS"
    assert row["off_gate_undeclared"] == ["tests/t.py::test_other"]
    assert (tmp_path / "src" / "a.py").read_text() == "x = 1\n"
