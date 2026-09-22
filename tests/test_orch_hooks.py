"""The orchestrator's guardrail hooks: hook JSON in, allow/deny out.

Each test feeds `orchestrator.hooks` the payload Claude Code sends on stdin
(https://code.claude.com/docs/en/hooks: `tool_name`, `tool_input`, `cwd`,
`agent_type`) and asserts the decision. Deny cases include the evasions the parser
is meant to see through -- `./`, `..`, absolute paths, symlinks, `bash -c`, `eval`,
`$(...)`, variables, wrappers, globs. The module docstring lists the ones it cannot.

Test-name prefixes are mutation gates (`scripts/mutation_battery.py`, section
`# --- orchestrator: hooks ---`): `test_owner_only_`, `test_push_to_main_`,
`test_blinding_`, `test_unwrapped_compute_`, `test_fail_closed_`.
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

from orchestrator import hooks  # noqa: E402

PROFILES = ("manager", "researcher", "verifier")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    for d in (
        ".git",
        ".orchestrator/outbox",
        "docs/owner/rulings",
        "docs/spec",
        "preregistration",
        ".claude/agents",
        ".claude/worktrees/agent-x/src",
        ".worktrees/job1",
        "src/rsr",
        "runs/r1",
        "runs/canary",
        "measurements",
        "experiments/e0c",
        "ops",
    ):
        (r / d).mkdir(parents=True, exist_ok=True)
    for f in ("r1.md", "r2.md", "researcher.md"):
        (r / ".orchestrator/outbox" / f).write_text("status: RETURNED\n")
    (r / "docs/owner/rulings/R-x.md").write_text("ruling\n")
    (r / "experiments/e0c/run.py").write_text("")
    return r


def _payload(root: Path, tool: str, tool_input: dict, **kw) -> dict:
    p = {
        "session_id": "s",
        "cwd": str(kw.pop("cwd", root)),
        "hook_event_name": "PreToolUse",
        "tool_name": tool,
        "tool_input": tool_input,
    }
    p.update(kw)
    return p


def _env(root: Path, profile: str | None = "researcher", run_id: str | None = "r1"):
    env = {"CLAUDE_PROJECT_DIR": str(root), "HOME": os.environ.get("HOME", "/")}
    if profile is not None:
        env["RSR_PROFILE"] = profile
    if run_id is not None:
        env["RSR_RUN_ID"] = run_id
    return env


def decide(root, tool, tool_input, profile="researcher", run_id="r1", **kw):
    agent_type = kw.pop("agent_type", None)
    payload = _payload(root, tool, tool_input, **kw)
    if agent_type:
        payload["agent_type"] = agent_type
    return hooks.decide_pretooluse(payload, _env(root, profile, run_id))


def bash(root, command, profile="researcher", **kw):
    return decide(root, "Bash", {"command": command}, profile, **kw)


def _profile_kw(profile: str) -> dict:
    """The verifier shares the researcher settings; `--agent` marks it."""
    if profile == "verifier":
        return {"profile": "researcher", "agent_type": "rsr-verifier"}
    return {"profile": profile}


# --------------------------------------------------------------------------- #
# Profiles
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("env_profile", "agent_type", "want"),
    [
        ("manager", None, "manager"),
        ("researcher", None, "researcher"),
        (None, None, "researcher"),
        ("bogus", None, "researcher"),
        ("researcher", "rsr-verifier", "verifier"),
        ("manager", "rsr-researcher", "researcher"),
        ("researcher", "rsr-manager", "researcher"),  # never looser
        ("verifier", "rsr-manager", "verifier"),
    ],
)
def test_profile_resolution_only_tightens(env_profile, agent_type, want):
    assert hooks.resolve_profile(env_profile, agent_type) == want


# --------------------------------------------------------------------------- #
# Owner-only paths (every profile)
# --------------------------------------------------------------------------- #

OWNER_PATHS = [
    "docs/owner/rulings/R-new.md",
    "preregistration/prereg.md",
    "docs/spec/rsr_model_spec_v0.5.md",
    ".claude/agents/rsr-manager.md",
    ".claude/settings.orchestrator.json",
    ".claude/settings.local.json",
    "ops/orchestrator.env",
    "runs/canary/baseline.json",
    "measurements/ledger.json",
    ".worktrees/job1/docs/owner/rulings/R-x.md",
    ".claude/worktrees/agent-x/.claude/agents/rsr-researcher.md",
]


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("rel", OWNER_PATHS)
@pytest.mark.parametrize("tool", ["Edit", "Write", "NotebookEdit"])
def test_owner_only_file_tools_denied(root, profile, rel, tool):
    key = "notebook_path" if tool == "NotebookEdit" else "file_path"
    why = decide(root, tool, {key: str(root / rel)}, **_profile_kw(profile))
    assert why is not None


@pytest.mark.parametrize(
    "spelling",
    [
        "{root}/src/../docs/owner/x.md",
        "{root}/./docs/owner/x.md",
        "{root}/docs//owner/x.md",
        "{root}/.claude/worktrees/../agents/x.md",
    ],
)
def test_owner_only_path_spellings_are_normalised(root, spelling):
    assert decide(root, "Write", {"file_path": spelling.format(root=root)}) is not None


def test_owner_only_through_a_symlink(root):
    (root / "innocent").symlink_to(root / "docs" / "owner")
    assert decide(root, "Write", {"file_path": str(root / "innocent/x.md")}) is not None
    assert bash(root, "echo x > innocent/x.md") is not None


@pytest.mark.parametrize(
    "rel",
    [
        "src/rsr/x.py",
        "docs/spec-corrections.md",
        "runs/r1/ledger.json",
        ".claude/worktrees/agent-x/src/x.py",
        "docs/lab-notes/dispatch-x.md",
    ],
)
def test_ordinary_paths_are_writable(root, rel):
    for profile in ("manager", "researcher"):
        assert decide(root, "Write", {"file_path": str(root / rel)}, profile) is None


OWNER_BASH = [
    "echo x > docs/owner/r.md",
    "echo x >> ./docs/owner/r.md",
    "echo x 1> docs/owner/r.md",
    "echo x &> docs/owner/r.md",
    "cat a | tee -a docs/spec/s.md",
    "cp /tmp/x .claude/agents/rsr-manager.md",
    "cp -t .claude/agents /tmp/x",
    "mv docs/owner/rulings/R-x.md /tmp/",
    "rm -rf preregistration",
    "/bin/rm docs/owner/rulings/R-x.md",
    "\\rm docs/owner/rulings/R-x.md",
    "env FOO=1 rm docs/owner/rulings/R-x.md",
    "timeout 5 rm docs/owner/rulings/R-x.md",
    "nohup touch .claude/settings.json",
    "sed -i '' s/a/b/ preregistration/p.md",
    "perl -pi -e s/a/b/ docs/spec/s.md",
    "dd if=/dev/zero of=docs/owner/x",
    "curl -o ops/orchestrator.env https://example.com",
    "ln -sf /tmp/x runs/canary/baseline.json",
    "echo '{}' > measurements/ledger.json",
    "git checkout -- docs/spec/rsr_model_spec_v0.5.md",
    "git rm docs/owner/rulings/R-x.md",
    "find docs/owner -name '*.md' -delete",
    "find docs/owner -exec rm {} ;",
    "bash -c 'echo x > docs/owner/r.md'",
    'sh -c "rm -rf .claude"',
    "zsh -lc 'touch docs/owner/x'",
    'eval "touch docs/owner/x"',
    "echo $(echo x > docs/owner/r.md)",
    "echo `touch docs/owner/x`",
    "cd docs && echo x > owner/r.md",
    "cd /tmp; echo x > {root}/docs/owner/r.md",
    "D=docs/owner; echo x > $D/r.md",
    "export D=docs/spec && touch $D/x",
    "xargs rm < list.txt; rm docs/owner/x",
    "python -c \"open('docs/owner/r.md', 'w').write(1)\"",
    "python3 - <<'EOF'\nfrom pathlib import Path\n"
    "Path('docs/spec/x').write_text('')\nEOF",
    "bash <<EOF\nrm -rf docs/owner\nEOF",
    "cat <<EOF > docs/owner/r.md\nhello\nEOF",
    "echo ok\ntouch docs/owner/x",
    "touch ~/.claude/settings.json",
]


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("cmd", OWNER_BASH)
def test_owner_only_bash_writes_denied(root, profile, cmd):
    kw = _profile_kw(profile)
    assert bash(root, cmd.replace("{root}", str(root)), **kw) is not None, cmd


@pytest.mark.parametrize(
    "cmd",
    [
        "cat docs/owner/rulings/R-x.md",
        "ls docs/spec",
        "cp docs/owner/rulings/R-x.md /tmp/y",
        "echo hi > runs/r1/notes.txt",
        "grep -n night docs/owner/rulings/R-x.md",
        'git commit -m "mention docs/owner and python open(w)"',
        "git diff --stat",
        "wc -l docs/spec/rsr_model_spec_v0.5.md",
    ],
)
def test_reads_and_ordinary_writes_allowed(root, cmd):
    assert bash(root, cmd, "manager") is None, cmd
    assert bash(root, cmd, "researcher") is None, cmd


# --------------------------------------------------------------------------- #
# git: main never moves in a night; no force; no --no-verify
# --------------------------------------------------------------------------- #

PUSH_MAIN = [
    "git push origin main",
    "git push origin HEAD:main",
    "git push origin feat:refs/heads/main",
    "git push origin :main",
    "git -C . push origin main",
    "git -c push.default=current push origin main",
    "'git' 'push' origin main",
    "/opt/homebrew/bin/git push origin main",
    "bash -c 'git push origin main'",
    "cd /tmp && git push -u origin main",
]


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("cmd", PUSH_MAIN)
def test_push_to_main_denied(root, profile, cmd):
    assert bash(root, cmd, **_profile_kw(profile)) is not None, cmd


def test_push_to_main_is_also_refused_via_update_ref(root):
    """Moving main without a push; shares `_is_main_ref` with the push rule."""
    for profile in ("manager", "researcher"):
        assert bash(root, "git update-ref refs/heads/main HEAD", profile) is not None


@pytest.mark.parametrize(
    "cmd",
    [
        "git push -f origin feat",
        "git push --force origin feat",
        "git push --force-with-lease origin feat",
        "git push origin +feat",
        "git push --mirror origin",
        "git push",
        "git push origin",
        "git push origin --delete feat",
        "git commit --no-verify -m x",
        "git commit -nm x",
        "git commit -a -n -m x",
        "git merge --no-verify feat",
        "gh pr merge 36 --squash",
        "git branch -f main HEAD",
        "git -c alias.p=push p origin feat",
        "git config alias.p push",
        "git config core.hooksPath /dev/null",
    ],
)
def test_git_force_noverify_and_merge_denied(root, cmd):
    for profile in ("manager", "researcher"):
        assert bash(root, cmd, profile) is not None, (profile, cmd)


@pytest.mark.parametrize(
    "cmd",
    [
        "git push origin night/2026-09-22",
        "git push -u origin s0-06-branch",
        'git commit -m "-n is not a flag here"',
        "git commit -am wip",
    ],
)
def test_ordinary_git_allowed_for_manager(root, cmd):
    assert bash(root, cmd, "manager") is None, cmd


def test_reset_hard_only_inside_a_job_worktree(root):
    assert bash(root, "git reset --hard HEAD") is not None
    assert bash(root, "git clean -fdx") is not None
    assert bash(root, "cd .worktrees/job1 && git reset --hard HEAD") is None
    assert bash(root, "git -C .worktrees/job1 reset --hard origin/main") is None
    assert bash(root, "git reset --hard", cwd=root / ".worktrees/job1") is None
    assert bash(root, "git reset --soft HEAD~1") is None


# --------------------------------------------------------------------------- #
# Compute goes through lane slots
# --------------------------------------------------------------------------- #

SLOT = (
    "PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run --lane cpu --slots 2 --"
)


@pytest.mark.parametrize(
    "cmd",
    [
        "pytest",
        "pytest -rs --tb=no",
        ".venv/bin/pytest tests/test_x.py",
        "uv run pytest -rs",
        "uv run --frozen pytest",
        ".venv/bin/python -m pytest tests",
        "python experiments/e0c/run.py",
        "uv run python experiments/decisive-shuffle/run.py --seed 0",
        "./experiments/e0c/run.py",
        "python -m experiments.e0c.run",
        "bash -c 'pytest -q'",
        f"{SLOT} true; pytest",
        "nice -n 10 pytest",
    ],
)
def test_unwrapped_compute_denied(root, cmd):
    for profile in ("manager", "researcher"):
        assert bash(root, cmd, profile) is not None, (profile, cmd)


@pytest.mark.parametrize(
    "cmd",
    [
        f"{SLOT} .venv/bin/pytest -rs --tb=no",
        f"{SLOT} uv run pytest",
        f"{SLOT} .venv/bin/python experiments/e0c/run.py",
        "cat experiments/e0c/run.py",
        "ruff check src/",
    ],
)
def test_slot_wrapped_compute_allowed(root, cmd):
    assert bash(root, cmd) is None, cmd


def test_slot_inner_command_is_still_checked(root):
    assert bash(root, f"{SLOT} launchctl list") is not None
    assert bash(root, f"{SLOT} rm -rf docs/owner") is not None


@pytest.mark.parametrize(
    "cmd", ["launchctl load x.plist", "caffeinate -i sleep 5", "nohup caffeinate -d &"]
)
def test_launchctl_and_caffeinate_denied(root, cmd):
    for profile in ("manager", "researcher"):
        assert bash(root, cmd, profile) is not None


# --------------------------------------------------------------------------- #
# Researcher / verifier: night branches and other runs' history
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "cmd",
    [
        "git checkout night/2026-09-22",
        "git switch night/2026-09-22",
        "git checkout origin/night/2026-09-22",
        "git worktree add ../x night/2026-09-22",
        "git log --all --oneline",
        "git log --branches",
        "git log --remotes=origin",
        "git show s0-03-branch",
        "git show s0-03-branch:runs/s0-03/ledger.json",
        "git diff night/2026-09-22",
        "git log origin/night/2026-09-22",
        "git cat-file -p other:README.md",
    ],
)
def test_researcher_git_history_limits(root, cmd):
    assert bash(root, cmd, "researcher") is not None, cmd
    assert bash(root, cmd, **_profile_kw("verifier")) is not None, cmd
    assert bash(root, cmd, "manager") is None, cmd


@pytest.mark.parametrize(
    "cmd",
    [
        "git show HEAD",
        "git show HEAD~2:src/rsr/x.py",
        "git log --oneline -n 5",
        "git log --oneline -5",
        "git diff main...HEAD",
        "git diff HEAD~1 -- src/",
        "git show 7871580",
        "git log --format=%H main..HEAD",
    ],
)
def test_researcher_git_on_own_history_allowed(root, cmd):
    assert bash(root, cmd, "researcher") is None, cmd


# --------------------------------------------------------------------------- #
# Blinding: the outbox
# --------------------------------------------------------------------------- #


def _outbox(root: Path, name: str = "") -> str:
    return str(root / ".orchestrator" / "outbox" / name)


@pytest.mark.parametrize("name", ["r2.md", "researcher.md", "_legacy-researcher.md"])
def test_blinding_researcher_cannot_read_other_reports(root, name):
    assert decide(root, "Read", {"file_path": _outbox(root, name)}) is not None


def test_blinding_researcher_reads_and_writes_its_own_report(root):
    assert decide(root, "Read", {"file_path": _outbox(root, "r1.md")}) is None
    assert decide(root, "Write", {"file_path": _outbox(root, "r1.md")}) is None
    assert bash(root, "cat .orchestrator/outbox/r1.md") is None
    assert bash(root, "cat .orchestrator/outbox/$RSR_RUN_ID.md") is None


def test_blinding_researcher_cannot_write_other_reports(root):
    assert decide(root, "Write", {"file_path": _outbox(root, "researcher.md")})
    assert decide(root, "Edit", {"file_path": _outbox(root, "r2.md")})
    assert bash(root, "echo x >> .orchestrator/outbox/researcher.md")


def test_blinding_without_a_run_id_reads_nothing(root):
    why = decide(root, "Read", {"file_path": _outbox(root, "r1.md")}, run_id=None)
    assert why is not None


@pytest.mark.parametrize(
    "cmd",
    [
        "cat .orchestrator/outbox/r2.md",
        "cat .orchestrator/outbox/*.md",
        "cat .orch*/outbox/r2.md",
        "head -5 ./.orchestrator/../.orchestrator/outbox/r2.md",
        "cat {root}/.orchestrator/outbox/r2.md",
        "cd .orchestrator/outbox && cat r2.md",
        "cd .orch* && cat outbox/r2.md",
        "ls .orchestrator",
        "grep -r threshold .",
        "grep -rn threshold",
        "rg threshold",
        "find . -name '*.md'",
        "bash -c 'cat .orchestrator/outbox/r2.md'",
        "python -c \"print(open('.orchestrator/outbox/r2.md').read())\"",
        "git show HEAD:.orchestrator/outbox/r2.md",
        "git log -p -- .orchestrator/outbox/researcher.md",
        "git grep threshold",
        "O=.orchestrator; cat $O/outbox/r2.md",
    ],
)
def test_blinding_bash_reads_of_the_outbox_denied(root, cmd):
    assert bash(root, cmd.replace("{root}", str(root))) is not None, cmd
    assert bash(root, cmd.replace("{root}", str(root)), "manager") is None, cmd


@pytest.mark.parametrize(
    "cmd",
    [
        "grep -rn threshold src tests",
        "rg threshold src/",
        "grep -r --exclude-dir=.orchestrator threshold .",
        "find src -name '*.py'",
        "git grep threshold -- src tests",
    ],
)
def test_blinding_narrow_searches_allowed(root, cmd):
    assert bash(root, cmd) is None, cmd


def test_blinding_grep_tool(root):
    assert decide(root, "Grep", {"pattern": "x"}) is not None
    assert decide(root, "Grep", {"pattern": "x", "path": str(root)}) is not None
    assert decide(root, "Grep", {"pattern": "x", "path": _outbox(root)}) is not None
    assert decide(root, "Grep", {"pattern": "x", "glob": "**/*.md"}) is not None
    assert decide(root, "Grep", {"pattern": "x", "path": str(root / "src")}) is None
    assert decide(root, "Grep", {"pattern": "x", "glob": "*.py"}) is None
    assert decide(root, "Grep", {"pattern": "x", "type": "py"}) is None
    assert decide(root, "Grep", {"pattern": "x"}, "manager") is None


def test_blinding_verifier_reads_no_report_at_all(root):
    kw = _profile_kw("verifier")
    assert decide(root, "Read", {"file_path": _outbox(root, "r1.md")}, **kw) is not None
    assert bash(root, "cat .orchestrator/outbox/r1.md", **kw) is not None
    assert (
        decide(root, "Read", {"file_path": str(root / "runs/r1/ledger.json")}, **kw)
        is None
    )


def test_blinding_manager_reads_every_report(root):
    for name in ("r1.md", "r2.md", "researcher.md"):
        assert decide(root, "Read", {"file_path": _outbox(root, name)}, "manager") is None


def test_verifier_never_edits_code(root):
    kw = _profile_kw("verifier")
    assert decide(root, "Write", {"file_path": str(root / "src/rsr/x.py")}, **kw)
    assert decide(root, "Edit", {"file_path": str(root / "tests/test_x.py")}, **kw)
    record = str(root / "runs/r1/verification.json")
    assert decide(root, "Write", {"file_path": record}, **kw) is None


def test_unknown_tools_pass_through(root):
    assert decide(root, "WebFetch", {"url": "https://example.com"}) is None
    assert decide(root, "Glob", {"pattern": "**/*.md"}) is None


# --------------------------------------------------------------------------- #
# Fail closed
# --------------------------------------------------------------------------- #


def _run(event: str, raw: str, env: dict, capsys) -> tuple[int, str]:
    rc = hooks.run_hook(event, raw, env)
    return int(rc), capsys.readouterr().out


def test_fail_closed_on_an_internal_error(root, monkeypatch, capsys):
    def boom(*_a, **_k):
        raise RuntimeError("parser bug")

    monkeypatch.setattr(hooks, "decide_pretooluse", boom)
    payload = _payload(root, "Bash", {"command": "ls"})
    rc, out = _run("pretooluse", json.dumps(payload), _env(root), capsys)
    assert rc == 0
    d = json.loads(out)["hookSpecificOutput"]
    assert d["permissionDecision"] == "deny"
    assert "parser bug" in d["permissionDecisionReason"]


def test_fail_closed_on_unparseable_payload(root, capsys):
    rc, out = _run("pretooluse", "{not json", _env(root), capsys)
    assert rc == 0
    assert json.loads(out)["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_pretooluse_json_shape_on_deny_and_silence_on_allow(root, capsys):
    deny = _payload(root, "Bash", {"command": "launchctl list"})
    rc, out = _run("pretooluse", json.dumps(deny), _env(root), capsys)
    assert rc == 0
    assert json.loads(out) == {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": json.loads(out)["hookSpecificOutput"][
                "permissionDecisionReason"
            ],
        }
    }
    allow = _payload(root, "Bash", {"command": "ls src"})
    rc, out = _run("pretooluse", json.dumps(allow), _env(root), capsys)
    assert (rc, out) == (0, "")


def test_the_cli_reads_stdin_and_exits_0_with_a_deny(root):
    payload = _payload(root, "Bash", {"command": "caffeinate -i sleep 1"})
    env = {**os.environ, **_env(root), "PYTHONPATH": str(_REPO / "scripts")}
    proc = subprocess.run(
        [sys.executable, "-m", "orchestrator.hooks", "pretooluse"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        cwd=_REPO,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"


# --------------------------------------------------------------------------- #
# Stop
# --------------------------------------------------------------------------- #

COMPLETE = """status: RETURNED
run_id: r1
updated: 2026-09-22T03:00:00Z
provenance: abc · cpu · h · [0]
BRIEF ERRORS: none
UNANSWERED BY THE BRIEF: none
BELIEVED, NOT VERIFIED:
  - the second seed's order
NEXT (proposed, not decided): rerun at d=128
"""


def _stop(root, env) -> str | None:
    return hooks.decide_stop({"hook_event_name": "Stop", "cwd": str(root)}, env)


def test_stop_blocks_without_a_report(root):
    (root / ".orchestrator/outbox/r1.md").unlink()
    why = _stop(root, _env(root))
    assert why is not None and "orchestrator.outbox new r1" in why


def test_stop_blocks_an_unfilled_template(root):
    (root / ".orchestrator/outbox/r1.md").unlink()
    (root / ".claude/agents/rsr-researcher.md").write_text(
        (_REPO / ".claude/agents/rsr-researcher.md").read_text()
    )
    from orchestrator import outbox

    assert int(outbox.cmd_new(root, "r1")) == 0
    assert _stop(root, _env(root)) is not None


@pytest.mark.parametrize(
    "field",
    [
        "BRIEF ERRORS",
        "UNANSWERED BY THE BRIEF",
        "BELIEVED, NOT VERIFIED",
        "NEXT (proposed, not decided)",
    ],
)
def test_stop_blocks_on_each_missing_field(root, field):
    text = "\n".join(
        line for line in COMPLETE.splitlines() if not line.startswith(field + ":")
    )
    if field == "BELIEVED, NOT VERIFIED":
        text = text.replace("  - the second seed's order\n", "")
    (root / ".orchestrator/outbox/r1.md").write_text(text + "\n")
    why = _stop(root, _env(root))
    assert why is not None and field in why


def test_stop_blocks_on_a_bad_status(root):
    (root / ".orchestrator/outbox/r1.md").write_text(
        COMPLETE.replace("status: RETURNED", "status: DONE")
    )
    assert _stop(root, _env(root)) is not None


def test_stop_allows_a_complete_report(root):
    (root / ".orchestrator/outbox/r1.md").write_text(COMPLETE)
    assert _stop(root, _env(root)) is None


def test_stop_blocks_without_a_run_id(root):
    assert _stop(root, _env(root, run_id=None)) is not None


def test_stop_does_not_hold_the_manager(root):
    assert _stop(root, _env(root, "manager", None)) is None


def test_stop_holds_the_verifier_until_its_record_is_valid(root):
    from orchestrator import verify

    env = _env(root)
    payload = {"hook_event_name": "Stop", "cwd": str(root), "agent_type": "rsr-verifier"}
    assert hooks.decide_stop(payload, env) is not None
    rec = {
        "status": "ok",
        "seed": verify.seed_for("r1"),
        "claims": [
            {
                "claim": "c",
                "command": "x",
                "expected": "1",
                "observed": "1",
                "match": True,
            }
        ],
        "verifier_sha": "a" * 40,
    }
    (root / "runs/r1/verification.json").write_text(json.dumps(rec))
    assert hooks.decide_stop(payload, env) is None
    rec["seed"] = "0" * 64
    (root / "runs/r1/verification.json").write_text(json.dumps(rec))
    assert hooks.decide_stop(payload, env) is not None


def test_stop_output_is_a_top_level_block(root, capsys):
    (root / ".orchestrator/outbox/r1.md").unlink()
    rc, out = _run("stop", json.dumps({"cwd": str(root)}), _env(root), capsys)
    assert rc == 0
    d = json.loads(out)
    assert d["decision"] == "block" and d["reason"]


# --------------------------------------------------------------------------- #
# SessionStart / SessionEnd
# --------------------------------------------------------------------------- #


def test_sessionstart_injects_night_json(root, capsys):
    # The contract path (loopcore.night_path): .orchestrator/state/night.json.
    (root / ".orchestrator/state").mkdir(parents=True, exist_ok=True)
    (root / ".orchestrator/state/night.json").write_text('{"branch": "night/2026-09-22"}')
    rc, out = _run("sessionstart", json.dumps({"cwd": str(root)}), _env(root), capsys)
    assert rc == 0
    ctx = json.loads(out)["hookSpecificOutput"]
    assert ctx["hookEventName"] == "SessionStart"
    assert "night/2026-09-22" in ctx["additionalContext"]
    assert "workqueue" not in ctx["additionalContext"]  # researchers do not see it


def test_sessionstart_manager_reports_the_workqueue_or_its_absence(root, capsys):
    rc, out = _run(
        "sessionstart", json.dumps({"cwd": str(root)}), _env(root, "manager"), capsys
    )
    assert rc == 0
    text = json.loads(out)["hookSpecificOutput"]["additionalContext"]
    assert "night.json: absent" in text
    assert "workqueue ready: unavailable" in text


def test_sessionend_appends_a_cycle_record(root, capsys):
    for reason in ("other", "prompt_input_exit"):
        payload = {"cwd": str(root), "session_id": "s1", "reason": reason}
        rc, _ = _run("sessionend", json.dumps(payload), _env(root), capsys)
        assert rc == 0
    lines = (root / ".orchestrator/cycles.jsonl").read_text().splitlines()
    recs = [json.loads(x) for x in lines]
    assert [r["reason"] for r in recs] == ["other", "prompt_input_exit"]
    assert recs[0]["profile"] == "researcher" and recs[0]["run_id"] == "r1"


# --------------------------------------------------------------------------- #
# The settings files that wire it
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "profile"),
    [
        ("settings.orchestrator.json", "manager"),
        ("settings.researcher.json", "researcher"),
    ],
)
def test_settings_wire_every_hook_with_its_profile(name, profile):
    s = json.loads((_REPO / ".claude" / name).read_text())
    h = s["hooks"]
    for event, sub in (
        ("PreToolUse", "pretooluse"),
        ("Stop", "stop"),
        ("SessionStart", "sessionstart"),
        ("SessionEnd", "sessionend"),
    ):
        cmds = [x["command"] for grp in h[event] for x in grp["hooks"]]
        assert any(f"RSR_PROFILE={profile}" in c and f" {sub}" in c for c in cmds), event
    pre = h["PreToolUse"][0]
    for tool in ("Bash", "Edit", "Write", "NotebookEdit", "Read", "Grep"):
        assert tool in pre["matcher"].split("|")
    # a hook that cannot start must block, not pass (exit 2, not 1)
    for event in ("PreToolUse", "Stop"):
        for x in h[event][0]["hooks"]:
            assert "exit 2" in x["command"]
    deny = s["permissions"]["deny"]
    for rule in (
        "Edit(docs/owner/**)",
        "Edit(preregistration/**)",
        "Edit(docs/spec/**)",
        "Edit(.claude/agents/**)",
        "Edit(ops/orchestrator.env)",
        "Edit(measurements/ledger.json)",
        "Bash(launchctl *)",
        "Bash(caffeinate *)",
    ):
        assert rule in deny, rule
