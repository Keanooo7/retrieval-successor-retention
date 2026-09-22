"""`scripts/orchestrator/lint_brief.py` -- a brief is checked against its base.

Brief errors are this project's dominant failure (7 of 11 briefs wrong on
2026-09-18; every brief on 2026-09-21 had one). Each test plants one of the
recurring classes in a brief over a tiny throwaway repository and asserts the lint
names it. Every gate here has a mutation in `scripts/mutation_battery.py`
(`# --- orchestrator: lint_brief ---`) that reddens it and nothing else.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from orchestrator import lint_brief as lb  # noqa: E402

from rsr.exit_codes import Exit  # noqa: E402

GIT = lb.git_exe()

SOURCE = (
    "\n".join(
        [
            "def f():",  # 1
            "    x = 1",  # 2
            "    return x",  # 3
            "",  # 4
            "def g():",  # 5
            "    return 2",  # 6
        ]
    )
    + "\n"
)

BODY = """
# Brief: a test brief

## Why

Because `src/a.py:5` defines `g`.

## Falsifier

"g does not return 2."

## Files in scope

`src/a.py`.

## Bar

- `g()` returns 2.

## Done when

The bar holds.

## Do NOT

Touch anything else.
"""


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        [
            GIT,
            "-C",
            str(repo),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    (r / "src").mkdir()
    (r / "src" / "a.py").write_text(SOURCE)
    (r / "runs" / "x").mkdir(parents=True)
    (r / "runs" / "x" / "ledger.json").write_text(
        json.dumps({"verdict": {"outcome": "falsified"}, "rows": [{"v": 3}]})
    )
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "base")
    monkeypatch.setenv("RSR_ORCH_ROOT", str(r))
    return r


def _front(repo: Path, **over) -> dict:
    front = {
        "id": "t",
        "item": "Q-1",
        "baseline_sha": _git(repo, "rev-parse", "HEAD"),
        "falsifier": "g does not return 2.",
        "anchors": [{"path": "src/a.py", "line": 5, "expect": "def g():"}],
        "premises": [
            {
                "claim": "g returns 2 at base",
                "check": "grep -n 'return 2' src/a.py",
                "expect_rc": 0,
                "expect_stdout_contains": "6:",
            },
            {
                "claim": "the run falsified",
                "ledger": "runs/x/ledger.json",
                "key": "verdict.outcome",
                "expect": "falsified",
            },
        ],
        "files_in_scope": ["src/a.py", {"path": "tests/test_new.py", "new": True}],
        "bar": ["g() returns 2"],
        "done_when": ["the bar holds"],
        "do_not": ["touch anything else"],
    }
    front.update(over)
    return front


def _write(tmp_path: Path, front: dict, body: str = BODY) -> Path:
    p = tmp_path / "dispatch-t.md"
    p.write_text("---\n" + yaml.safe_dump(front, sort_keys=False) + "---\n" + body)
    return p


def _lint(repo: Path, brief: Path, base: str | None = None):
    base = base or _git(repo, "rev-parse", "HEAD")
    return lb.lint(brief, repo, lb.resolve_commit(repo, base))


def _kinds(findings) -> list[str]:
    return [f.kind for f in findings]


# --------------------------------------------------------------------------- #


def test_a_clean_brief_exits_0(repo, tmp_path, capsys):
    brief = _write(tmp_path, _front(repo))
    assert _lint(repo, brief) == []
    assert lb.main([str(brief), "--base", "HEAD"]) is Exit.OK
    assert "0 finding(s)" in capsys.readouterr().out


def test_a_drifted_anchor_is_a_finding(repo, tmp_path):
    """The rsr.py:433-vs-:448 class: the line cited no longer holds the text."""
    front = _front(repo, anchors=[{"path": "src/a.py", "line": 4, "expect": "def g():"}])
    body = BODY.replace("src/a.py:5", "src/a.py:4")
    f = _lint(repo, _write(tmp_path, front, body))
    assert _kinds(f) == ["anchor"], f
    assert "src/a.py:4 drifted" in f[0].detail
    assert "found at :5" in f[0].detail


def test_an_undeclared_body_citation_is_a_finding(repo, tmp_path):
    """A `path:N` in prose that no anchor checks is a line number nobody verified."""
    body = BODY.replace("Touch anything else.", "Touch `a.py:3` or anything else.")
    f = _lint(repo, _write(tmp_path, _front(repo), body))
    assert _kinds(f) == ["anchor"], f
    assert "a.py:3" in f[0].detail and "not a declared anchor" in f[0].detail


def test_a_failing_premise_is_a_finding(repo, tmp_path):
    """The carried-forward premise: the brief believes something the base denies."""
    front = _front(repo)
    front["premises"] = [
        {"claim": "h exists", "check": "grep -q 'def h' src/a.py", "expect_rc": 0},
        {"claim": "stdout", "check": "echo hello", "expect_stdout_contains": "bye"},
        {
            "claim": "the run survived",
            "ledger": "runs/x/ledger.json",
            "key": "verdict.outcome",
            "expect": "survived",
        },
        {
            "claim": "no such key",
            "ledger": "runs/x/ledger.json",
            "key": "rows.7.v",
            "expect": 3,
        },
    ]
    f = _lint(repo, _write(tmp_path, front))
    assert _kinds(f) == ["premise"] * 4, f
    assert "exited 1, expected 0" in f[0].detail
    assert "lacks 'bye'" in f[1].detail
    assert "is 'falsified'" in f[2].detail
    assert "no key `rows.7.v`" in f[3].detail


def test_a_ledger_premise_resolves_list_indices(repo, tmp_path):
    front = _front(repo)
    front["premises"] = [
        {"claim": "row", "ledger": "runs/x/ledger.json", "key": "rows.0.v", "expect": 3}
    ]
    assert _lint(repo, _write(tmp_path, front)) == []


@pytest.mark.parametrize(
    "cmd",
    [
        "echo pwned > PWNED",
        "grep x src/a.py | tee PWNED",
        "rm src/a.py",
        "mv src/a.py PWNED",
        "sed -i '' s/g/h/ src/a.py",
        "git commit --allow-empty -m pwned",
        "git push origin main",
    ],
)
def test_a_writer_command_is_rejected_and_not_run(repo, tmp_path, cmd):
    front = _front(repo)
    front["premises"] = [{"claim": "writes", "check": cmd, "expect_rc": 0}]
    head = _git(repo, "rev-parse", "HEAD")
    f = _lint(repo, _write(tmp_path, front))
    assert _kinds(f) == ["premise"], f
    assert "rejected, not run" in f[0].detail
    assert not (repo / "PWNED").exists()
    assert (repo / "src" / "a.py").read_text() == SOURCE
    assert _git(repo, "rev-parse", "HEAD") == head


def test_premises_run_in_a_throwaway_worktree_at_base(repo, tmp_path):
    """Not the caller's checkout: an uncommitted edit is invisible, and the
    throwaway worktree is gone afterwards."""
    (repo / "src" / "a.py").write_text(SOURCE.replace("return 2", "return 3"))
    front = _front(repo)
    front["premises"] = [
        {"claim": "base has 2", "check": "grep -q 'return 2' src/a.py", "expect_rc": 0},
        {"claim": "a worktree", "check": "test -f .git", "expect_rc": 0},
    ]
    assert _lint(repo, _write(tmp_path, front)) == []
    assert _git(repo, "worktree", "list", "--porcelain").count("worktree ") == 1


def test_a_wrong_baseline_sha_is_a_finding(repo, tmp_path):
    """Off by the brief's own commit: written at A, committed as B, checked at B."""
    first = _git(repo, "rev-parse", "HEAD")
    (repo / "notes.md").write_text("the brief's own commit\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "brief")
    f = _lint(repo, _write(tmp_path, _front(repo, baseline_sha=first)))
    assert _kinds(f) == ["baseline"], f
    assert first[:12] in f[0].detail


def test_a_missing_heading_is_a_finding(repo, tmp_path):
    body = BODY.replace("## Done when\n\nThe bar holds.\n", "")
    f = _lint(repo, _write(tmp_path, _front(repo), body))
    assert _kinds(f) == ["heading"], f
    assert "Done when" in f[0].detail


def test_an_empty_falsifier_is_a_finding(repo, tmp_path):
    body = BODY.replace('"g does not return 2."', "")
    f = _lint(repo, _write(tmp_path, _front(repo, falsifier=" "), body))
    assert sorted(_kinds(f)) == ["front-matter", "heading"], f


def test_files_in_scope_must_exist_at_base(repo, tmp_path):
    front = _front(
        repo, files_in_scope=["src/missing.py", {"path": "src/a.py", "new": True}]
    )
    f = _lint(repo, _write(tmp_path, front))
    assert _kinds(f) == ["scope", "scope"], f
    assert "src/missing.py does not exist" in f[0].detail
    assert "declared new but exists" in f[1].detail


def test_no_front_matter_is_a_finding_not_a_refusal(repo, tmp_path):
    p = tmp_path / "dispatch-t.md"
    p.write_text(BODY)
    assert lb.main([str(p), "--base", "HEAD"]) is Exit.FAIL


def test_no_base_exits_3(repo, tmp_path, capsys):
    """No --base and no night.json: the lint did not run, and says so."""
    brief = _write(tmp_path, _front(repo))
    assert lb.main([str(brief)]) is Exit.DID_NOT_RUN
    assert "no base" in capsys.readouterr().err
    assert lb.main([str(brief), "--base", "no-such-rev"]) is Exit.DID_NOT_RUN


def test_the_night_base_sha_is_the_default_base(repo, tmp_path):
    state = repo / ".orchestrator" / "state"
    state.mkdir(parents=True)
    (state / "night.json").write_text(
        json.dumps({"base_sha": _git(repo, "rev-parse", "HEAD")})
    )
    assert lb.main([str(_write(tmp_path, _front(repo)))]) is Exit.OK


def test_the_cli_exits_through_the_protocol_with_json(repo, tmp_path):
    front = _front(repo, anchors=[{"path": "src/a.py", "line": 5, "expect": "def h():"}])
    brief = _write(tmp_path, front)
    env = {**os.environ, "PYTHONPATH": str(_REPO / "scripts"), "RSR_ORCH_ROOT": str(repo)}
    cmd = [sys.executable, "-m", "orchestrator.lint_brief", str(brief), "--json"]
    r = subprocess.run([*cmd, "--base", "HEAD"], capture_output=True, text=True, env=env)
    assert r.returncode == 1, r.stderr
    out = json.loads(r.stdout)
    assert out["exit"] == 1 and [f["kind"] for f in out["findings"]] == ["anchor"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert r.returncode == 3, r.stderr
    assert json.loads(r.stdout)["exit"] == 3
