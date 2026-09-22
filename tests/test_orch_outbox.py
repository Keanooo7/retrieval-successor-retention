"""`orchestrator.outbox`: one report file per run, and the generated index.

The per-run file is the blinding mechanism (a researcher's hook lets it read only
its own); the index replaces the prepend-only `researcher.md` without destroying it.
The validation here is the one the researcher's Stop hook runs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from orchestrator import outbox  # noqa: E402

ROLE = (_REPO / ".claude/agents/rsr-researcher.md").read_text()

FILLED = """status: RETURNED
run_id: r1
updated: 2026-09-22T03:00:00Z
BRIEF ERRORS: none
UNANSWERED BY THE BRIEF: none
BELIEVED, NOT VERIFIED: none
NEXT (proposed, not decided): none
"""


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / ".claude/agents").mkdir(parents=True)
    (tmp_path / ".claude/agents/rsr-researcher.md").write_text(ROLE)
    return tmp_path


def test_the_template_is_read_from_the_role_file():
    t = outbox.template_from_role(ROLE)
    for name in ("status", "run_id", *outbox.REQUIRED_FIELDS):
        assert f"\n{name}:" in "\n" + t, name


def test_new_writes_the_template_with_the_run_id(root):
    assert int(outbox.cmd_new(root, "s1-02")) == 0
    text = outbox.report_path(root, "s1-02").read_text()
    assert outbox.parse_fields(text)["run_id"] == "s1-02"
    # A fresh template is not a report: every required field is still a placeholder.
    problems = outbox.validate_report(text, "s1-02")
    assert any("status" in p for p in problems)
    assert sum("placeholder" in p for p in problems) == len(outbox.REQUIRED_FIELDS)


@pytest.mark.parametrize("bad", ["researcher", "_legacy-researcher", "../x", "a b", ""])
def test_new_refuses_a_bad_or_reserved_run_id(root, bad):
    assert int(outbox.cmd_new(root, bad)) == 3


def test_new_refuses_to_overwrite(root):
    assert int(outbox.cmd_new(root, "r1")) == 0
    assert int(outbox.cmd_new(root, "r1")) == 3


def test_a_filled_report_validates():
    assert outbox.validate_report(FILLED, "r1") == []


def test_template_placeholder_is_not_an_answer():
    text = FILLED.replace(
        "BELIEVED, NOT VERIFIED: none", "BELIEVED, NOT VERIFIED: <list>"
    )
    problems = outbox.validate_report(text, "r1")
    assert problems == ["`BELIEVED, NOT VERIFIED:` is still the template placeholder"]


def test_multiline_field_bodies_count():
    text = FILLED.replace(
        "NEXT (proposed, not decided): none",
        "NEXT (proposed, not decided):\n  - rerun at d=128, because the ratio moved",
    )
    assert outbox.validate_report(text, "r1") == []


def test_an_empty_field_and_a_wrong_run_id_are_reported():
    text = FILLED.replace("UNANSWERED BY THE BRIEF: none", "UNANSWERED BY THE BRIEF:")
    problems = outbox.validate_report(text, "r2")
    assert any("run_id" in p for p in problems)
    assert any("UNANSWERED BY THE BRIEF" in p and "empty" in p for p in problems)


def test_an_unchosen_status_line_is_refused():
    text = FILLED.replace("status: RETURNED", "status: RETURNED | BLOCKED")
    assert outbox.validate_report(text, "r1")


def test_check_exit_codes(root):
    assert int(outbox.cmd_check(root, "r1")) == 3  # no report: did not run
    outbox.report_path(root, "r1").parent.mkdir(parents=True)
    outbox.report_path(root, "r1").write_text(FILLED.replace("status: RETURNED", ""))
    assert int(outbox.cmd_check(root, "r1")) == 1
    outbox.report_path(root, "r1").write_text(FILLED)
    assert int(outbox.cmd_check(root, "r1")) == 0


def test_index_moves_the_legacy_outbox_aside_and_lists_runs(root):
    d = outbox.outbox_dir(root)
    d.mkdir(parents=True)
    legacy = "status: RETURNED\nrun_id: decisive-shuffle\n(the old prepend-only file)\n"
    (d / "researcher.md").write_text(legacy)
    (d / "r1.md").write_text(FILLED)
    (d / "r2.md").write_text(
        FILLED.replace("r1", "r2").replace("2026-09-22T03", "2026-09-22T05")
    )
    assert int(outbox.cmd_index(root)) == 0
    assert (d / "_legacy-researcher.md").read_text() == legacy
    index = (d / "researcher.md").read_text()
    assert index.startswith(outbox.GENERATED_MARKER)
    assert index.index("| r2 |") < index.index("| r1 |")  # newest first
    assert "_legacy-researcher.md" in index
    # regenerating is idempotent and never touches the legacy file again
    assert int(outbox.cmd_index(root)) == 0
    assert (d / "researcher.md").read_text() == index


def test_index_refuses_to_clobber_two_hand_written_files(root):
    d = outbox.outbox_dir(root)
    d.mkdir(parents=True)
    (d / "researcher.md").write_text("hand written\n")
    (d / "_legacy-researcher.md").write_text("also hand written\n")
    assert int(outbox.cmd_index(root)) == 3
    assert (d / "researcher.md").read_text() == "hand written\n"


def test_the_committed_index_is_current():
    """`.orchestrator/outbox/researcher.md` is generated; a hand edit, or a per-run
    report added without regenerating, shows up here."""
    committed = (_REPO / ".orchestrator/outbox/researcher.md").read_text()
    assert committed == outbox.render_index(_REPO)


def test_the_cli_usage_error_exits_3():
    proc = subprocess.run(
        [sys.executable, "-m", "orchestrator.outbox", "frobnicate"],
        capture_output=True,
        text=True,
        cwd=_REPO,
        env={"PYTHONPATH": str(_REPO / "scripts"), "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 3, proc.stderr
