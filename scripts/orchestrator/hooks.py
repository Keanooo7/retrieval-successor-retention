"""Claude Code hooks for the headless orchestrator: one entry point, one subcommand
per hook event. Reads the hook's JSON payload on stdin.

    RSR_PROFILE=researcher PYTHONPATH=scripts .venv/bin/python \\
        -m orchestrator.hooks pretooluse|stop|sessionstart|sessionend

Wired by `.claude/settings.orchestrator.json` (manager) and
`.claude/settings.researcher.json` (researcher, verifier). The owner's interactive
sessions do not load either file and are not affected.

Schema -- from https://code.claude.com/docs/en/hooks (read 2026-09-22), not guessed:

* stdin carries `session_id`, `cwd`, `hook_event_name`, `permission_mode`, and
  `agent_type` "when the session uses `--agent` or the hook fires inside a
  subagent". PreToolUse adds `tool_name` and `tool_input`; "for the file tools
  `Write`, `Edit`, and `Read`, `tool_input.file_path` is always absolute". Bash
  carries `tool_input.command`; NotebookEdit `notebook_path`; Grep `path`.
* PreToolUse decides through `hookSpecificOutput.permissionDecision` =
  `"deny"` with `permissionDecisionReason` ("For "deny", shown to Claude").
* Stop blocks with top-level `{"decision": "block", "reason": ...}`; "Claude
  Code overrides the hook and ends the turn after 8 consecutive blocks".
* SessionStart adds context with `hookSpecificOutput.additionalContext`
  (capped at 10,000 characters). SessionEnd has no decision control.
* Exit 0 = JSON on stdout is honoured. Exit 2 blocks. **Any other non-zero exit
  is a non-blocking error: the tool call proceeds.** So this module never lets
  an exception escape `pretooluse`/`stop` -- an internal error becomes a deny /
  a block -- and the settings files add `|| exit 2` so a hook that cannot even
  start (no `.venv`) fails closed too.

Profiles. `RSR_PROFILE` is set in the hook command (`manager` | `researcher`); the
verifier shares the researcher settings file and is recognised by
`agent_type == "rsr-verifier"`. A profile can only be made STRICTER by
`agent_type` (manager -> researcher -> verifier), never looser, so a researcher
cannot un-blind itself by spawning a subagent named `rsr-manager`. An unknown or
unset `RSR_PROFILE` is treated as `researcher`.

What `pretooluse` denies (every profile unless marked):

* Edit/Write/NotebookEdit to an owner-only path: `docs/owner/**`,
  `preregistration/**`, `runs/canary/baseline*`, `docs/spec/**` (§15 must never be
  filled), `.claude/**` except `.claude/worktrees/**`, `ops/orchestrator.env`,
  and `measurements/ledger.json` (written only through the registry API).
  Matched on path SEGMENTS anywhere in the normalised absolute path, so the same
  rule holds in the main checkout, in `.worktrees/<job>/` and under `~/.claude`.
* Bash that writes one of those paths: a redirection target, the operands of
  rm/mv/touch/tee/truncate/chmod/..., the destination of cp/ln/rsync/install,
  `sed -i`/`perl -i`, `dd of=`, `curl -o`, `git rm/mv/restore/checkout <path>`, `find
  <root> -delete|-exec`, and an inline interpreter (`python -c`, heredoc) whose
  text names one of them next to a write call.
* `git push` of `main` (any refspec form), `--force`/`-f`/`+ref`/`--mirror`/
  `--delete`, and a bare `git push` with no refspec; `gh pr merge`; `git branch
  -f|-D|-M main`; `git update-ref refs/heads/main` (main never moves in a night,
  R-2026-09-22-night-branch).
* `--no-verify` anywhere, and `git commit -n`.
* `git -c alias.*=...` and `git config alias.*|core.hooksPath` (both defeat this list).
* `git reset --hard` and `git clean -f` unless the directory they act on is inside
  a `.worktrees/` job dir.
* bare `pytest` (any spelling: `pytest`, `python -m pytest`, `uv run pytest`) and
  `experiments/*/run.py`, unless the segment is `python -m orchestrator.slot run ...
  -- <cmd>` (the inner command is still checked against every other rule).
* `launchctl`, `caffeinate` (only the tick may).
* researcher/verifier: `git checkout|switch|worktree add` of `night/*`; `git log
  --all|--branches|--remotes|--glob|--reflog`; `git show|log|diff|cat-file|...`
  naming any ref other than HEAD-forms, a sha, `main`, `origin/main` or the
  current branch.
* researcher/verifier (blinding): any read of `.orchestrator/outbox/` other than
  the researcher's own `<RSR_RUN_ID>.md` -- Read, Grep over a tree containing it,
  a Bash token naming it, a glob that expands into it, a recursive grep/rg/find
  rooted above it. The verifier may not read even that one: it sees the ledger,
  manifest and claims list, never the prose.
* verifier: Edit/Write anywhere except `runs/<id>/verification.json` and the
  scratch/tmp dirs. It never edits code.

🔴 **A Bash denylist is best-effort, and this one is no exception.** It parses the
command text Claude writes; it is not a sandbox. The Claude Code docs say the same
of their own Bash rules: "a deny or ask rule covers the invocation Claude usually
produces and isn't a security boundary around the program"
(https://code.claude.com/docs/en/permissions). Known ways past it:

* a script file: `bash x.sh`, `python x.py`, `make`, a git hook, a `uv run`
  entry point -- anything whose effect is not in the command text;
* indirection the parser cannot evaluate: `$(printf ...)` building a path,
  arrays, `eval` of a variable set in an earlier Bash call, `IFS` games, aliases
  and functions defined in the shell profile;
* interpreters reading other files (`python -c "exec(open('x').read())"`);
* a `git` alias already present in `.git/config` or `~/.gitconfig`;
* reading the outbox by a route that never names it: `grep -r x .` from a
  directory ABOVE the repo root the hook knows about, `git archive`/`git bundle`
  then reading the archive, `gh api` against another branch, `tar` + reading
  inside the tarball under another name;
* a quoted operator (`";"`) is indistinguishable from a real one after shlex
  and splits a segment where the shell would not.

The second layer is the `permissions.deny` rules in the same settings files, which
Claude Code enforces "in every mode, including `bypassPermissions`"
(https://code.claude.com/docs/en/permission-modes). The PreToolUse docs do not say
whether a hook's deny holds under `bypassPermissions`; the deny rules do. The only
real boundary is OS-level (the Claude Code sandbox), which this does not configure.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import fnmatch
import glob as _glob
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from orchestrator import outbox as _outbox  # noqa: E402
from orchestrator import verify as _verify  # noqa: E402
from rsr.exit_codes import ArgumentParser, Exit, run_main, status  # noqa: E402

GIT = "/opt/homebrew/bin/git"
PROFILES = ("manager", "researcher", "verifier")
_RANK = {"manager": 0, "researcher": 1, "verifier": 2}
AGENT_PROFILE = {
    "rsr-manager": "manager",
    "rsr-researcher": "researcher",
    "rsr-verifier": "verifier",
}
SLOT_HINT = (
    "wrap it: PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run "
    "--lane <L> --slots <k> -- <cmd>"
)
_CONTEXT_CAP = 9000  # the docs cap additionalContext at 10,000 characters


def resolve_profile(env_profile: str | None, agent_type: str | None) -> str:
    """The effective profile: `RSR_PROFILE`, made stricter (never looser) by the
    `--agent` / subagent type. Unknown or unset -> researcher."""
    base = env_profile if env_profile in _RANK else "researcher"
    from_agent = AGENT_PROFILE.get(agent_type or "")
    if from_agent is not None and _RANK[from_agent] > _RANK[base]:
        return from_agent
    return base


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


def norm(path: str, cwd: str) -> str:
    p = os.path.expanduser(path)
    if not os.path.isabs(p):
        p = os.path.join(cwd, p)
    return os.path.normpath(p)


def _variants(p: str) -> list[str]:
    out = [p]
    with contextlib.suppress(OSError):
        r = os.path.realpath(p)
        if r != p:
            out.append(r)
    return out


def _segments(p: str) -> list[str]:
    return [s for s in p.split("/") if s]


def protected_reason(abs_path: str) -> str | None:
    """Why no headless agent may write `abs_path`, or None. Segment-matched."""
    for p in _variants(abs_path):
        parts = _segments(p)
        for i, s in enumerate(parts):
            nxt = parts[i + 1] if i + 1 < len(parts) else None
            if s == "docs" and nxt == "owner":
                return "docs/owner/** is owner-only (rulings are written by Brendan)"
            if s == "docs" and nxt == "spec":
                return "docs/spec/** is owner-only (§15 must never be filled by a model)"
            if s == "preregistration":
                return "preregistration/** is owner-only (signed, not re-openable)"
            if s == ".claude" and nxt != "worktrees":
                return ".claude/** (role and settings files) is owner-only"
            if s == "ops" and nxt == "orchestrator.env":
                return "ops/orchestrator.env is owner-only"
            if (
                s == "runs"
                and nxt == "canary"
                and i + 2 < len(parts)
                and parts[i + 2].startswith("baseline")
            ):
                return "runs/canary/baseline* is owner-only (canary re-baseline)"
            if s == "measurements" and nxt == "ledger.json":
                return "measurements/ledger.json is written only through the registry API"
    return None


def outbox_kind(abs_path: str) -> str | None:
    """None if `abs_path` is not in an outbox; "" for the outbox directory (or its
    `.orchestrator` parent); else the file name inside it."""
    for p in _variants(abs_path):
        parts = _segments(p)
        for i, s in enumerate(parts):
            if s != ".orchestrator":
                continue
            if i + 1 == len(parts):
                return ""
            if parts[i + 1] == "outbox":
                return parts[i + 2] if i + 2 < len(parts) else ""
    return None


@dataclass
class Ctx:
    profile: str
    run_id: str | None
    cwd: str
    env: dict[str, str]
    roots: list[str] = field(default_factory=list)
    vars: dict[str, str] = field(default_factory=dict)
    branch: str | None = None

    def outbox_dirs(self) -> list[str]:
        return [os.path.join(r, ".orchestrator", "outbox") for r in self.roots]


def blinded_reason(ctx: Ctx, abs_path: str) -> str | None:
    """Researcher/verifier: why reading `abs_path` would break blinding, or None."""
    if ctx.profile == "manager":
        return None
    kind = outbox_kind(abs_path)
    if kind is None:
        return None
    if ctx.profile == "researcher" and ctx.run_id and kind == f"{ctx.run_id}.md":
        return None
    who = (
        "the verifier reads the ledger, manifest and claims list, never report prose"
        if ctx.profile == "verifier"
        else f"a researcher reads only its own .orchestrator/outbox/{ctx.run_id}.md"
    )
    return f"blinding: {who} (the 09-21 decisive run saw S0-03's numbers this way)"


def above_outbox(ctx: Ctx, abs_dir: str) -> bool:
    """True if a recursive walk from `abs_dir` would enter an outbox."""
    if outbox_kind(abs_dir) is not None:
        return True
    d = abs_dir.rstrip("/") or "/"
    for o in ctx.outbox_dirs():
        if o == d or o.startswith(d.rstrip("/") + "/"):
            return True
    return os.path.isdir(os.path.join(d, ".orchestrator", "outbox"))


def _expand(ctx: Ctx, tok: str) -> list[str]:
    """Absolute paths a token may denote: itself, and its glob expansion."""
    p = norm(tok, ctx.cwd)
    out = [p]
    if any(ch in tok for ch in "*?["):
        with contextlib.suppress(OSError, re.error):
            out += _glob.glob(p, recursive=True)[:200]
    return out


# --------------------------------------------------------------------------- #
# Bash parsing
# --------------------------------------------------------------------------- #

_SEPARATORS = {";", "&&", "||", "|", "&", "|&", "(", ")", ";;", "{", "}"}
_REDIRECTS = {">", ">>", ">|", "&>", "&>>", ">&", "<>"}
_SHELLS = {"bash", "sh", "zsh", "dash", "ksh", "fish"}
_INTERPRETER_PROG = re.compile(r"python[0-9.]*|perl|ruby|node|osascript|php")
_WRITE_CALL = re.compile(
    r"open\([^)]*['\"][^'\"]*[wax+]|write_text|write_bytes|\.write\(|dump\(|"
    r"unlink|rename|replace\(|rmtree|shutil\.|os\.remove|touch\(|mkdir|>\s*\S"
)
_PROTECTED_TEXT = re.compile(
    r"docs/owner|docs/spec(/|\b)|preregistration/|\.claude/(?!worktrees)|"
    r"\.claude['\"]|ops/orchestrator\.env|runs/canary/baseline|measurements/ledger\.json"
)
_OUTBOX_TEXT = re.compile(r"\.orchestrator/outbox(?:/([^\s'\"`;|&)]*))?")
_RUN_PY = re.compile(r"(?:^|.*/)experiments/[^/]+/run\.py")
_ANY_ARG_WRITERS = {
    "rm", "rmdir", "unlink", "touch", "truncate", "chmod", "chown", "chflags",
    "chgrp", "xattr", "tee", "mkdir", "shred", "srm", "trash", "mv", "ed",
}  # fmt: skip
_DEST_WRITERS = {"cp", "ln", "install", "rsync", "scp", "ditto", "gcp"}
_WRAPPERS = {"command", "builtin", "nohup", "time", "noglob", "exec", "sudo", "stdbuf"}
_SEARCHERS = {"grep", "egrep", "fgrep", "rgrep", "rg", "ag", "ack", "find", "fd"}
_GIT_REV_CMDS = {
    "show", "log", "diff", "cat-file", "whatchanged", "difftool", "archive",
    "format-patch", "rev-list", "shortlog", "ls-tree", "blame", "bundle",
}  # fmt: skip
_GIT_VALUE_OPTS = {
    "-n", "--max-count", "--since", "--until", "--after", "--before", "--author",
    "--committer", "--grep", "--format", "--pretty", "-U", "-S", "-G", "-L",
    "--skip", "--date", "-o", "--output", "-C", "-M", "--diff-filter",
}  # fmt: skip


def _strip_heredocs(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Remove heredoc bodies. Returns the text and [(introducing line, body)]."""
    bodies = []
    out_lines: list[str] = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        out_lines.append(line)
        delims = re.findall(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1", line)
        i += 1
        for _q, delim in delims:
            body = []
            while i < len(lines) and lines[i].strip() != delim:
                body.append(lines[i])
                i += 1
            i += 1  # the delimiter line
            bodies.append((line, "\n".join(body)))
    return "\n".join(out_lines), bodies


def _extract_substitutions(text: str) -> tuple[str, list[str]]:
    """Pull out `$(...)` (nested) and backtick bodies; replace them with a word."""
    subs: list[str] = []
    out = []
    i = 0
    in_single = False
    while i < len(text):
        ch = text[i]
        if ch == "'" and not in_single:
            j = text.find("'", i + 1)
            if j == -1:
                out.append(text[i:])
                break
            out.append(text[i : j + 1])
            i = j + 1
            continue
        if text.startswith("$(", i):
            depth, j = 1, i + 2
            while j < len(text) and depth:
                if text[j] == "(":
                    depth += 1
                elif text[j] == ")":
                    depth -= 1
                j += 1
            subs.append(text[i + 2 : j - 1])
            out.append(" __SUBST__ ")
            i = j
            continue
        if ch == "`":
            j = text.find("`", i + 1)
            if j == -1:
                j = len(text)
            subs.append(text[i + 1 : j])
            out.append(" __SUBST__ ")
            i = j + 1
            continue
        out.append(ch)
        i += 1
    return "".join(out), subs


def _newlines_to_semicolons(text: str) -> str:
    """Unquoted newlines separate commands; shlex would treat them as blanks."""
    text = text.replace("\\\n", " ")
    out = []
    q: str | None = None
    for ch in text:
        if q:
            if ch == q:
                q = None
        elif ch in "'\"":
            q = ch
        elif ch == "\n":
            out.append(" ; ")
            continue
        out.append(ch)
    return "".join(out)


def tokenize(text: str) -> list[str]:
    try:
        lex = shlex.shlex(text, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        lex.commenters = ""
        return list(lex)
    except ValueError:  # unbalanced quotes: fall back to blanks
        return text.split()


def segments(tokens: list[str]) -> list[list[str]]:
    segs: list[list[str]] = [[]]
    for t in tokens:
        if t in _SEPARATORS:
            segs.append([])
        else:
            segs[-1].append(t)
    return [s for s in segs if s]


def _expand_vars(ctx: Ctx, tok: str) -> str:
    def sub(m: re.Match) -> str:
        name = m.group(1) or m.group(2)
        return ctx.vars.get(name, ctx.env.get(name, m.group(0)))

    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)", sub, tok)


_ASSIGN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*", re.S)


def _strip_wrappers(seg: list[str]) -> list[str]:
    """Drop leading assignments and exec wrappers (env, nohup, timeout, uv run...)."""
    s = list(seg)
    changed = True
    while s and changed:
        changed = False
        while s and _ASSIGN.fullmatch(s[0]):
            s = s[1:]
            changed = True
        if not s:
            break
        prog = os.path.basename(s[0].lstrip("\\"))
        if prog == "env":
            s = s[1:]
            while s and (s[0].startswith("-") or _ASSIGN.fullmatch(s[0])):
                s = s[2:] if s[0] in ("-u", "-C", "-S") else s[1:]
            changed = True
        elif prog in _WRAPPERS:
            s = s[1:]
            while s and s[0].startswith("-"):
                s = s[1:]
            changed = True
        elif prog in ("nice", "ionice"):
            s = s[1:]
            while s and s[0].startswith("-"):
                s = s[2:] if s[0] in ("-n", "-c") else s[1:]
            changed = True
        elif prog in ("timeout", "gtimeout"):
            s = s[1:]
            while s and s[0].startswith("-"):
                s = s[2:] if s[0] in ("-s", "-k") else s[1:]
            s = s[1:]  # the duration
            changed = True
        elif prog == "xargs":
            s = s[1:]
            while s and s[0].startswith("-"):
                takes = s[0] in ("-n", "-I", "-P", "-L", "-d", "-s", "-E", "-J", "-R")
                s = s[2:] if takes else s[1:]
            changed = True
        elif prog in ("uv", "uvx") and (prog == "uvx" or s[1:2] == ["run"]):
            s = s[1:] if prog == "uvx" else s[2:]
            while s and s[0].startswith("-"):
                takes = s[0] in (
                    "--with", "--extra", "--python", "-p", "--project", "--directory",
                    "--group", "--env-file", "--package", "--from",
                )  # fmt: skip
                s = s[2:] if takes and "=" not in s[0] else s[1:]
            changed = True
    return s


def _non_flags(args: list[str]) -> list[str]:
    out, after_dd = [], False
    for a in args:
        if a == "--" and not after_dd:
            after_dd = True
            continue
        if after_dd or not a.startswith("-"):
            out.append(a)
    return out


def _redirect_targets(seg: list[str]) -> tuple[list[str], list[str]]:
    """(tokens with redirections removed, redirection targets)."""
    rest, targets = [], []
    i = 0
    while i < len(seg):
        t = seg[i]
        if t in _REDIRECTS:
            if i + 1 < len(seg):
                tgt = seg[i + 1]
                if not (t == ">&" and tgt.isdigit()):
                    targets.append(tgt)
            i += 2
            continue
        if t == "<":
            i += 2
            continue
        if t.isdigit() and i + 1 < len(seg) and seg[i + 1] in _REDIRECTS:
            i += 1
            continue
        rest.append(t)
        i += 1
    return rest, targets


# --------------------------------------------------------------------------- #
# Bash rules
# --------------------------------------------------------------------------- #


def _git_split(args: list[str]) -> tuple[str | None, list[str], list[str], str | None]:
    """(subcommand, subcommand args, `-c` values, `-C` dir) for `git <args>`."""
    configs: list[str] = []
    cdir = None
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path"):
            if a == "-c" and i + 1 < len(args):
                configs.append(args[i + 1])
            if a == "-C" and i + 1 < len(args):
                cdir = args[i + 1]
            i += 2
            continue
        if a.startswith("-"):
            if a.startswith("--config-env") or a.startswith("-c"):
                configs.append(a.split("=", 1)[-1])
            i += 1
            continue
        return a, args[i + 1 :], configs, cdir
    return None, [], configs, cdir


def _short_cluster_has(arg: str, letter: str) -> bool:
    return arg.startswith("-") and not arg.startswith("--") and letter in arg[1:]


def _is_main_ref(ref: str) -> bool:
    ref = ref.lstrip("+")
    dst = ref.split(":", 1)[1] if ":" in ref else ref
    return dst in ("main", "refs/heads/main") or dst.endswith("/refs/heads/main")


def _rev_allowed(ctx: Ctx, rev: str) -> bool:
    if rev == "" or rev.isdigit():
        return True
    for side in re.split(r"\.\.\.?", rev):
        if side == "":
            continue
        base = re.split(r"[~^@{]", side, maxsplit=1)[0]
        if base in ("", "HEAD", "main", "origin/main", "FETCH_HEAD", "ORIG_HEAD"):
            continue
        if re.fullmatch(r"[0-9a-f]{7,40}", base):
            continue
        if ctx.branch and base == ctx.branch:
            continue
        return False
    return True


def _current_branch(cwd: str) -> str | None:
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        r = subprocess.run(
            [GIT, "-C", cwd, "symbolic-ref", "--short", "-q", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0:
            return r.stdout.strip() or None
    return None


def _in_job_worktree(path: str) -> bool:
    return ".worktrees" in _segments(path)


def _check_git(ctx: Ctx, args: list[str]) -> str | None:
    sub, sargs, configs, cdir = _git_split(args)
    for c in configs:
        if c.startswith("alias.") or c.lower().startswith("core.hookspath"):
            return f"`git -c {c}` defeats the orchestrator's git rules"
    if sub is None:
        return None
    workdir = norm(cdir, ctx.cwd) if cdir else ctx.cwd
    blinded = ctx.profile in ("researcher", "verifier")

    if sub == "push":
        pos = _non_flags(sargs)
        for a in sargs:
            if a in ("-f", "--mirror", "--all", "--delete", "-d", "--prune") or (
                a.startswith("--force")
            ):
                return f"`git push {a}` is refused (force/mirror/delete)"
            if _short_cluster_has(a, "f") or _short_cluster_has(a, "d"):
                return f"`git push {a}` is refused (force/delete)"
        refspecs = pos[1:]
        if not refspecs:
            return (
                "a `git push` with no explicit refspec is refused: name `origin "
                "<branch>` (main never moves in a night, R-2026-09-22-night-branch)"
            )
        for r in refspecs:
            if r.startswith("+"):
                return f"`git push {r}` is a force push"
            dst = r.split(":", 1)[-1]
            on_main = dst in ("HEAD", "@") and _current_branch(workdir) == "main"
            if _is_main_ref(r) or on_main:
                return (
                    "pushing main is refused: main never moves in a night "
                    "(R-2026-09-22-night-branch)"
                )
        return None
    if sub == "commit" and any(_short_cluster_has(a, "n") for a in _cmd_flags(sargs)):
        return "`git commit -n` is --no-verify"
    if sub == "config":
        for a in sargs:
            if a.startswith("alias.") or a.lower().startswith("core.hookspath"):
                return f"`git config {a}` defeats the orchestrator's git rules"
    moves = ("-f", "--force", "-D", "-M", "-m", "-d", "-C", "-c")
    if sub == "branch" and "main" in sargs and any(a in moves for a in sargs):
        return "moving or deleting main is refused (R-2026-09-22-night-branch)"
    if sub == "update-ref" and any(_is_main_ref(a) for a in _non_flags(sargs)):
        return "moving main is refused (R-2026-09-22-night-branch)"
    destroys = (sub == "reset" and "--hard" in sargs) or (
        sub == "clean"
        and any(a == "--force" or _short_cluster_has(a, "f") for a in sargs)
    )
    if destroys and not _in_job_worktree(workdir):
        return (
            f"`git {sub}` destroying work is allowed only inside a .worktrees/ "
            f"job dir; this runs in {workdir}"
        )
    if sub in ("rm", "mv", "restore", "checkout", "apply", "stash"):
        for a in _non_flags(sargs):
            for p in _expand(Ctx(ctx.profile, ctx.run_id, workdir, ctx.env), a):
                if (why := protected_reason(p)) is not None:
                    return f"`git {sub}` would change a protected path: {why}"
    if blinded and sub in ("checkout", "switch", "worktree"):
        for a in sargs:
            if re.match(r"(refs/heads/|refs/remotes/[^/]+/|[^/]+/)?night/", a):
                return (
                    f"a researcher does not check out a night branch ({a}); work on "
                    f"your own branch in your .worktrees/ dir"
                )
    if blinded and sub == "log":
        for a in sargs:
            if a in ("--all", "--reflog", "-g", "--walk-reflogs") or a.startswith(
                ("--branches", "--remotes", "--glob", "--exclude", "--tags")
            ):
                return (
                    f"blinding: `git log {a}` reads other runs' branches; log HEAD, "
                    f"main or your own branch"
                )
    if blinded and sub in _GIT_REV_CMDS:
        if ctx.branch is None:
            ctx.branch = _current_branch(workdir)
        skip = False
        after_dd = False
        for a in sargs:
            if skip:
                skip = False
                continue
            if a == "--":
                after_dd = True
                continue
            if a.startswith("-"):
                skip = a in _GIT_VALUE_OPTS
                continue
            rev, _, path = a.partition(":") if not after_dd else ("", "", a)
            if path and (why := blinded_reason(ctx, norm(path, workdir))):
                return why
            if after_dd or os.path.exists(norm(a, workdir)):
                continue
            if not _rev_allowed(ctx, rev):
                return (
                    f"blinding: `git {sub} {a}` names another branch; a researcher "
                    f"may name HEAD, main, a sha or its own branch (use --opt=value "
                    f"for option values)"
                )
    if blinded and sub == "grep":
        if "--" not in sargs:
            return (
                "blinding: `git grep` needs an explicit pathspec after `--` that "
                "excludes .orchestrator/outbox"
            )
        specs = sargs[sargs.index("--") + 1 :]
        for s in specs:
            p = norm(s, workdir)
            if above_outbox(ctx, p) or blinded_reason(ctx, p):
                return "blinding: that `git grep` pathspec covers .orchestrator/outbox"
    return None


def _cmd_flags(args: list[str]) -> list[str]:
    """Flags of a `git commit`, skipping the value after -m/-F/-c/-C/--author..."""
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
            continue
        if a in ("-m", "-F", "-c", "-C", "--author", "--date", "-t", "--fixup"):
            skip = True
            continue
        if a.startswith("-"):
            out.append(a)
    return out


def _write_targets(prog: str, args: list[str]) -> list[str]:
    nf = _non_flags(args)
    if prog in _ANY_ARG_WRITERS:
        return nf
    if prog in _DEST_WRITERS:
        tgts = nf[-1:] if len(nf) >= 2 else []
        for i, a in enumerate(args):
            if a == "-t" and i + 1 < len(args):
                tgts.append(args[i + 1])
            if a.startswith("--target-directory="):
                tgts.append(a.split("=", 1)[1])
        return tgts
    if prog in ("sed", "gsed", "perl", "ruby") and any(
        a == "-i" or a.startswith(("-i", "--in-place")) or _short_cluster_has(a, "i")
        for a in args
    ):
        return nf
    if prog == "dd":
        return [a[3:] for a in args if a.startswith("of=")]
    if prog == "patch":
        return nf
    if prog in ("curl", "wget"):
        out = []
        for i, a in enumerate(args):
            if a in ("-o", "-O", "--output", "--output-document") and i + 1 < len(args):
                out.append(args[i + 1])
            elif a.startswith(("--output=", "--output-document=")):
                out.append(a.split("=", 1)[1])
        return out
    if prog in ("tar", "unzip", "ditto"):
        out = []
        for i, a in enumerate(args):
            if a in ("-C", "-d", "--directory") and i + 1 < len(args):
                out.append(args[i + 1])
        return out
    return []


def _search_roots(prog: str, args: list[str]) -> list[str] | None:
    """Roots a recursive search starts from; None if the command does not recurse."""
    if prog == "find":
        roots = []
        for a in args:
            if a.startswith(("-", "(", "!")):
                break
            roots.append(a)
        return roots
    if prog in ("grep", "egrep", "fgrep") and not any(
        a in ("-r", "-R", "--recursive", "--dereference-recursive")
        or _short_cluster_has(a, "r")
        or _short_cluster_has(a, "R")
        for a in args
    ):
        return None
    nf = _non_flags(args)
    has_pattern_flag = any(a in ("-e", "-f", "--regexp", "--file") for a in args)
    return nf if has_pattern_flag or prog == "fd" else nf[1:]


def _excludes_outbox(args: list[str]) -> bool:
    joined = " ".join(args)
    return bool(
        re.search(r"(exclude-dir|--exclude|--ignore|-g\s*!|--glob=!|-prune)", joined)
        and (".orchestrator" in joined or "outbox" in joined)
    )


def _runs_an_experiment(prog: str, cmd: list[str]) -> bool:
    """`experiments/<x>/run.py` executed -- directly, or as python's script/module."""
    if _RUN_PY.fullmatch(cmd[0]):
        return True
    if not prog.startswith("python"):
        return False
    args = cmd[1:]
    for i, a in enumerate(args):
        if a == "-m" and i + 1 < len(args):
            return bool(re.fullmatch(r"experiments\.[^.]+\.run", args[i + 1]))
        if a in ("-c",):
            return False
        if not a.startswith("-"):
            return bool(_RUN_PY.fullmatch(a))
    return False


def _code_writes_protected(code: str) -> str | None:
    """Interpreter code that names a protected path next to a write call."""
    m = _PROTECTED_TEXT.search(code)
    if m and _WRITE_CALL.search(code):
        return (
            f"interpreter code names a protected path ({m.group(0)}) next to a write call"
        )
    return None


def check_segment(
    ctx: Ctx, seg: list[str], depth: int, in_slot: bool = False
) -> str | None:
    # Assignment-only segment: remember it for later `$VAR` expansion.
    if all(_ASSIGN.fullmatch(t) for t in seg) or (seg[0] == "export" and len(seg) > 1):
        for t in seg:
            if _ASSIGN.fullmatch(t):
                k, v = t.split("=", 1)
                ctx.vars[k] = _expand_vars(ctx, v)
        return None
    seg = [_expand_vars(ctx, t) for t in seg]
    rest, redirs = _redirect_targets(seg)
    for tgt in redirs:
        for p in _expand(ctx, tgt):
            if (why := protected_reason(p)) is not None:
                return f"redirect into a protected path: {why}"
            if (why := blinded_reason(ctx, p)) is not None:
                return why
    cmd = _strip_wrappers(rest)
    if not cmd:
        return None
    prog = os.path.basename(cmd[0].lstrip("\\"))
    args = cmd[1:]

    # Blinding: any token that names the outbox (researcher: other than its own).
    if ctx.profile in ("researcher", "verifier"):
        for tok in cmd:
            if re.match(r"--(exclude|ignore)", tok) or tok.startswith("!"):
                continue
            for cand in (tok, tok.split("=", 1)[-1]):
                if not cand or cand.startswith("-") or " " in cand:
                    continue
                for p in _expand(ctx, cand):
                    if (why := blinded_reason(ctx, p)) is not None:
                        return why
        roots = _search_roots(prog, args) if prog in _SEARCHERS else None
        if roots is not None and not _excludes_outbox(args):
            for r in roots or ["."]:
                if above_outbox(ctx, norm(r, ctx.cwd)):
                    return (
                        f"blinding: `{prog}` would walk .orchestrator/outbox; search a "
                        f"narrower path (src/, tests/, runs/<id>/) or exclude it"
                    )

    if prog in ("cd", "pushd"):
        if args and args[0] != "-":
            ctx.cwd = norm(args[0], ctx.cwd)
        return None
    if prog in ("launchctl", "caffeinate"):
        return f"`{prog}` is refused: only the tick manages power and launchd"
    if "--no-verify" in cmd:
        return "`--no-verify` is refused: the hooks it skips are the gates"

    # Recursion into command strings.
    if prog in _SHELLS:
        for i, a in enumerate(args):
            is_c = a == "-c" or _short_cluster_has(a, "c")
            if is_c and i + 1 < len(args):
                why = analyze_bash(ctx, args[i + 1], depth + 1, in_slot)
                if why is not None:
                    return why
    if prog == "eval":
        return analyze_bash(ctx, " ".join(args), depth + 1, in_slot)
    if prog == "find":
        for i, a in enumerate(args):
            if a in ("-exec", "-execdir", "-ok", "-okdir"):
                inner = []
                for b in args[i + 1 :]:
                    if b in (";", "+", "\\;"):
                        break
                    inner.append(b)
                if (why := check_segment(ctx, inner, depth + 1, in_slot)) is not None:
                    return why
        if any(a in ("-delete", "-exec", "-execdir") for a in args):
            for r in _search_roots("find", args) or ["."]:
                for p in _expand(ctx, r):
                    if (why := protected_reason(p)) is not None:
                        return f"`find -delete/-exec` under a protected path: {why}"

    # The slot wrapper: `python -m orchestrator.slot run ... -- <cmd>`.
    if prog.startswith("python") and "-m" in args:
        mi = args.index("-m")
        if args[mi + 1 : mi + 2] == ["orchestrator.slot"]:
            if "--" in args:
                inner = args[args.index("--") + 1 :]
                return (
                    check_segment(ctx, inner, depth + 1, in_slot=True) if inner else None
                )
            return None
        if args[mi + 1 : mi + 2] == ["pytest"] and not in_slot:
            return f"bare `python -m pytest` is refused: {SLOT_HINT}"
    if prog in ("pytest", "py.test") and not in_slot:
        return f"bare `pytest` is refused: {SLOT_HINT}"
    if not in_slot and _runs_an_experiment(prog, cmd):
        return f"an unwrapped experiments/*/run.py is refused: {SLOT_HINT}"
    if _INTERPRETER_PROG.fullmatch(prog):
        for i, a in enumerate(args[:-1]):
            code_flag = a in ("-c", "-e", "-E", "--eval", "-p")
            if code_flag and (why := _code_writes_protected(args[i + 1])) is not None:
                return why

    if prog == "git" and (why := _check_git(ctx, args)) is not None:
        return why
    if prog == "gh" and args[:2] == ["pr", "merge"]:
        return (
            "`gh pr merge` is refused: main never moves in a night "
            "(R-2026-09-22-night-branch)"
        )

    for tgt in _write_targets(prog, args):
        for p in _expand(ctx, tgt):
            if (why := protected_reason(p)) is not None:
                return f"`{prog}` would write a protected path: {why}"
    return None


def analyze_bash(
    ctx: Ctx, command: str, depth: int = 0, in_slot: bool = False
) -> str | None:
    """The first reason `command` is refused, or None. Best-effort: see module doc."""
    if depth > 6:
        return "command nesting is too deep to check; flatten it"
    text, bodies = _strip_heredocs(command)
    if ctx.profile in ("researcher", "verifier"):
        for m in _OUTBOX_TEXT.finditer(command):
            name = m.group(1) or ""
            if ctx.profile == "researcher" and ctx.run_id:
                own = {f"{ctx.run_id}.md", "${RSR_RUN_ID}.md", "$RSR_RUN_ID.md"}
                if name in own:
                    continue
            return blinded_reason(ctx, "/.orchestrator/outbox/" + name) or "blinding"
    for intro, body in bodies:
        head = intro.split("<<")[0]
        why = None
        if re.search(r"(^|[\s|;&(/])(ba|z|da|k)?sh\b", head):
            why = analyze_bash(ctx, body, depth + 1, in_slot)
        elif re.search(r"(^|[\s|;&(/])(python[0-9.]*|perl|ruby|node|php)\b", head):
            why = _code_writes_protected(body)
        if why is not None:
            return why
    text, subs = _extract_substitutions(text)
    for s in subs:
        if (why := analyze_bash(ctx, s, depth + 1, in_slot)) is not None:
            return why
    for seg in segments(tokenize(_newlines_to_semicolons(text))):
        if (why := check_segment(ctx, seg, depth, in_slot)) is not None:
            return why
    return None


# --------------------------------------------------------------------------- #
# Tool dispatch
# --------------------------------------------------------------------------- #

_EDIT_TOOLS = {"Edit", "Write", "NotebookEdit", "MultiEdit"}


def _repo_roots(payload: dict, env: dict[str, str]) -> list[str]:
    roots = []
    for r in (
        env.get("CLAUDE_PROJECT_DIR"),
        env.get("RSR_REPO_ROOT"),
        payload.get("cwd"),
    ):
        if r:
            roots.append(os.path.normpath(r))
    # the repo holding cwd, found by walking up to a `.git`
    d = payload.get("cwd") or ""
    while d and d != "/":
        if os.path.exists(os.path.join(d, ".git")):
            roots.append(d)
            break
        d = os.path.dirname(d)
    return list(dict.fromkeys(roots))


def make_ctx(payload: dict, env: dict[str, str]) -> Ctx:
    profile = resolve_profile(env.get("RSR_PROFILE"), payload.get("agent_type"))
    cwd = payload.get("cwd") or env.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    return Ctx(
        profile=profile,
        run_id=env.get("RSR_RUN_ID") or None,
        cwd=os.path.normpath(cwd),
        env=env,
        roots=_repo_roots(payload, env),
    )


def _verifier_may_write(ctx: Ctx, p: str, payload: dict) -> bool:
    parts = _segments(p)
    if len(parts) >= 3 and parts[-1] == "verification.json" and parts[-3] == "runs":
        return True
    scratch = [payload.get("scratchpad_dir"), "/tmp", "/private/tmp"]
    return any(s and (p == s or p.startswith(s.rstrip("/") + "/")) for s in scratch)


def decide_pretooluse(payload: dict, env: dict[str, str]) -> str | None:
    """The deny reason for this tool call, or None to let it through."""
    ctx = make_ctx(payload, env)
    tool = payload.get("tool_name", "")
    ti = payload.get("tool_input") or {}
    if tool in _EDIT_TOOLS:
        raw = ti.get("file_path") or ti.get("notebook_path") or ""
        if not raw:
            return None
        p = norm(raw, ctx.cwd)
        if (why := protected_reason(p)) is not None:
            return why
        if ctx.profile == "verifier" and not _verifier_may_write(ctx, p, payload):
            return (
                "the verifier never edits code; it writes only "
                "runs/<id>/verification.json"
            )
        return blinded_reason(ctx, p)
    if tool == "Read":
        raw = ti.get("file_path") or ""
        return blinded_reason(ctx, norm(raw, ctx.cwd)) if raw else None
    if tool == "Grep" and ctx.profile in ("researcher", "verifier"):
        p = norm(ti.get("path") or ".", ctx.cwd)
        if (why := blinded_reason(ctx, p)) is not None:
            return why
        if above_outbox(ctx, p):
            g, t = ti.get("glob") or "", ti.get("type") or ""
            if (
                g
                and "md" not in g
                and not any(fnmatch.fnmatch(s, g) for s in ("x.md", "a/x.md", "a/b/x.md"))
            ):
                return None
            if t and t not in ("md", "markdown"):
                return None
            return (
                "blinding: this Grep walks .orchestrator/outbox; give a narrower path "
                "(src/, tests/, runs/<id>/) or a non-markdown glob/type"
            )
        return None
    if tool == "Bash":
        return analyze_bash(ctx, ti.get("command") or "")
    return None


# --------------------------------------------------------------------------- #
# Stop / SessionStart / SessionEnd
# --------------------------------------------------------------------------- #


def _project_dir(payload: dict, env: dict[str, str]) -> Path:
    return Path(env.get("CLAUDE_PROJECT_DIR") or payload.get("cwd") or os.getcwd())


def decide_stop(payload: dict, env: dict[str, str]) -> str | None:
    """The reason the session may not stop yet, or None."""
    profile = resolve_profile(env.get("RSR_PROFILE"), payload.get("agent_type"))
    if profile == "manager":
        return None
    run_id = env.get("RSR_RUN_ID")
    if not run_id:
        return (
            "RSR_RUN_ID is not set, so there is no report to check. The launcher sets "
            "it; say so in your final message and stop."
        )
    root = _project_dir(payload, env)
    if profile == "verifier":
        path = root / "runs" / run_id / "verification.json"
        if not path.is_file():
            return f"write {path} before stopping (see rsr-verifier.md)"
        try:
            rec = json.loads(path.read_text())
        except (OSError, ValueError) as e:
            return f"{path} is not valid JSON: {e}"
        problems = _verify.validate_record(rec, run_id)
        return f"{path} is invalid: " + "; ".join(problems) if problems else None
    path = _outbox.report_path(root, run_id)
    if not path.is_file():
        return (
            f"no report at {path}. Create it with `PYTHONPATH=scripts .venv/bin/python "
            f"-m orchestrator.outbox new {run_id}` and fill every field."
        )
    problems = _outbox.validate_report(path.read_text(), run_id)
    if problems:
        return f"{path} is incomplete: " + "; ".join(problems)
    return None


def sessionstart_context(payload: dict, env: dict[str, str]) -> str:
    profile = resolve_profile(env.get("RSR_PROFILE"), payload.get("agent_type"))
    root = _project_dir(payload, env)
    lines = [f"[orchestrator] profile={profile} run_id={env.get('RSR_RUN_ID') or '-'}"]
    # night.json lives in the MAIN checkout's state dir (loopcore.night_path);
    # RSR_ORCH_ROOT is exported by tick/dispatch, the project dir is the fallback.
    orch_root = Path(env.get("RSR_ORCH_ROOT") or root)
    night = Path(
        env.get("RSR_NIGHT_JSON") or orch_root / ".orchestrator" / "state" / "night.json"
    )
    if night.is_file():
        lines.append(f"night.json ({night}):\n{night.read_text().strip()}")
    else:
        lines.append(f"night.json: absent at {night} (no night is open)")
    if profile == "manager":
        mod = root / "scripts" / "orchestrator" / "workqueue.py"
        if not mod.is_file():
            lines.append(f"workqueue ready: unavailable ({mod} does not exist)")
        else:
            try:
                r = subprocess.run(
                    [sys.executable, "-m", "orchestrator.workqueue", "ready"],
                    cwd=root,
                    env={**os.environ, "PYTHONPATH": str(root / "scripts")},
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                lines.append(f"workqueue ready (rc={r.returncode}):\n{r.stdout.strip()}")
                if r.returncode != 0:
                    lines.append(r.stderr.strip()[-800:])
            except (OSError, subprocess.SubprocessError) as e:
                lines.append(f"workqueue ready: failed to run ({e})")
    text = "\n".join(lines)
    return (
        text if len(text) <= _CONTEXT_CAP else text[: _CONTEXT_CAP - 20] + "\n[truncated]"
    )


def sessionend_record(payload: dict, env: dict[str, str]) -> dict:
    root = _project_dir(payload, env)
    head = None
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        r = subprocess.run(
            [GIT, "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        head = r.stdout.strip() if r.returncode == 0 else None
    return {
        "utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "event": "SessionEnd",
        "reason": payload.get("reason"),
        "session_id": payload.get("session_id"),
        "profile": resolve_profile(env.get("RSR_PROFILE"), payload.get("agent_type")),
        "agent_type": payload.get("agent_type"),
        "run_id": env.get("RSR_RUN_ID"),
        "cwd": payload.get("cwd"),
        "head": head,
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def _emit(obj: dict) -> None:
    print(json.dumps(obj))


def run_hook(event: str, raw: str, env: dict[str, str]) -> Exit:
    try:
        payload = json.loads(raw) if raw.strip() else {}
        if not isinstance(payload, dict):
            raise ValueError("hook payload is not a JSON object")
    except ValueError as e:
        payload, parse_error = {}, e
    else:
        parse_error = None

    if event == "pretooluse":
        try:
            if parse_error is not None:
                raise parse_error
            reason = decide_pretooluse(payload, env)
        except Exception as e:  # fail closed: an unchecked call is a denied call
            reason = f"orchestrator hook error, failing closed: {type(e).__name__}: {e}"
        if reason is not None:
            _emit(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": reason,
                    }
                }
            )
        return Exit.OK
    if event == "stop":
        try:
            if parse_error is not None:
                raise parse_error
            reason = decide_stop(payload, env)
        except Exception as e:
            reason = (
                f"orchestrator stop hook error, failing closed: {type(e).__name__}: {e}"
            )
        if reason is not None:
            _emit({"decision": "block", "reason": reason})
        return Exit.OK
    if event == "sessionstart":
        _emit(
            {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": sessionstart_context(payload, env),
                }
            }
        )
        return Exit.OK
    # sessionend
    rec = sessionend_record(payload, env)
    path = _project_dir(payload, env) / ".orchestrator" / "cycles.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return Exit.OK


def main() -> Exit:
    ap = ArgumentParser(prog="orchestrator.hooks")
    ap.add_argument("event", choices=("pretooluse", "stop", "sessionstart", "sessionend"))
    args = ap.parse_args()
    return status(run_hook(args.event, sys.stdin.read(), dict(os.environ)))


if __name__ == "__main__":
    run_main(main)
