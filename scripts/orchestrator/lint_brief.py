"""Lint a dispatch brief against the commit it claims to be written at.

Brief errors are this project's dominant failure. On 2026-09-18, 7 of 11 briefs
contained an error a researcher caught, and on 2026-09-21 every brief did
(`docs/lab-notes/overnight-2026-09-21.md` §6 *BRIEF ERRORS*). The recurring
classes, each with the check below that catches it:

* **drifted line anchors** (`canary.py:196` was `return {`; the battery anchors
  were +1 at `0da72e6`) -> every ``anchors`` entry is read with
  ``git show <base>:<path>`` and its ``expect`` substring must be on that line;
  and every ``path.ext:N`` citation in the body must be a declared anchor, so a
  line number cannot be cited without being checked.
* **premises carried forward from an earlier cycle** (PREREG-s003's "may change
  the gap distribution") -> every ``premises`` entry is executed: a read-only
  shell check in a THROWAWAY ``git worktree add --detach`` at base, or a ledger
  key read at base.
* **baseline sha off by the brief's own commit** -> ``baseline_sha`` must resolve
  to the same commit as ``--base``.
* **files in scope that do not exist** (S0-03: two missing) -> each must exist at
  base, or be declared ``new: true`` and NOT exist.
* **missing sections** -> the six required headings, each non-empty, and a
  non-empty ``falsifier``.

Not mechanically covered, and stated so nobody believes otherwise: *scope
contradicting the bar* and *a question that is a theorem, not a measurement*.
Both need a reader; a premise with a check is the nearest mechanical proxy.

Format: `docs/lab-notes/BRIEF-TEMPLATE.md`.

Exit codes (`rsr.exit_codes`, `docs/gates.md`): ``0`` clean · ``1`` at least one
finding · ``3`` the lint could not run (no base, unresolvable base, brief file
missing, throwaway worktree could not be created). A brief with no front matter
is a *finding* (``1``), not a refusal: the brief is wrong, the lint ran.

Run::

    PYTHONPATH=scripts .venv/bin/python -m orchestrator.lint_brief \\
        docs/lab-notes/dispatch-<id>.md [--base <sha>] [--json]

``--base`` defaults to ``base_sha`` in ``.orchestrator/state/night.json`` under the
root (env ``RSR_ORCH_ROOT``, else the git toplevel of the cwd).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from rsr.exit_codes import ArgumentParser, Exit, did_not_run, run_main  # noqa: E402

REQUIRED_HEADINGS = ("Why", "Falsifier", "Files in scope", "Bar", "Done when", "Do NOT")
"""Body sections every brief carries (`.claude/agents/rsr-manager.md`, brief rules)."""

REQUIRED_KEYS = (
    "id",
    "item",
    "baseline_sha",
    "falsifier",
    "anchors",
    "premises",
    "files_in_scope",
    "bar",
    "done_when",
    "do_not",
)

#: Obvious writers. A premise is a READ of the base tree; anything matching one of
#: these is rejected unrun. Deliberately crude and over-inclusive: `>` anywhere
#: (redirects, `2>&1`, and yes, a `>` inside a grep pattern -- use `-F` on a
#: literal without it, or `≥`), and the listed verbs as words.
WRITER_PATTERNS: tuple[tuple[str, str], ...] = (
    (r">", "a `>` redirect"),
    (r"(^|[\s;|&(`])rm\s", "`rm`"),
    (r"(^|[\s;|&(`])mv\s", "`mv`"),
    (r"(^|[\s;|&(`])cp\s", "`cp`"),
    (r"(^|[\s;|&(`])tee(\s|$)", "`tee`"),
    (r"(^|[\s;|&(`])touch\s", "`touch`"),
    (r"(^|[\s;|&(`])mkdir\s", "`mkdir`"),
    (r"sed\s+(-[a-zA-Z]*\s+)*-[a-zA-Z]*i", "`sed -i`"),
    (r"git\s+commit", "`git commit`"),
    (r"git\s+push", "`git push`"),
    (
        r"git\s+(checkout|reset|stash|worktree|merge|rebase|switch|restore)",
        "a git writer",
    ),
)

PREMISE_TIMEOUT_S = 120

#: `path.ext:N` citations in the body. Each must be a declared anchor.
_CITATION = re.compile(
    r"(?<![\w/.-])([\w./-]*[\w-]\.(?:py|md|json|toml|ya?ml|txt|sh|cfg|ini)):(\d+)"
)
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


@dataclass(frozen=True)
class Finding:
    kind: str
    detail: str


class CannotRun(Exception):
    """The lint could not run: exit 3, never 0 and never 1."""


# --------------------------------------------------------------------------- #
# git
# --------------------------------------------------------------------------- #


def git_exe() -> str:
    """`RSR_GIT`, else Homebrew's git on the Studio, else whatever is on PATH."""
    env = os.environ.get("RSR_GIT")
    if env:
        return env
    if Path("/opt/homebrew/bin/git").exists():
        return "/opt/homebrew/bin/git"
    found = shutil.which("git")
    if not found:
        raise CannotRun("no git executable")
    return found


def _git(root: Path, *args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        [git_exe(), "-C", str(root), *args], capture_output=True, text=True, check=check
    )


def resolve_root() -> Path:
    env = os.environ.get("RSR_ORCH_ROOT")
    if env:
        return Path(env).resolve()
    p = subprocess.run(
        [git_exe(), "rev-parse", "--show-toplevel"], capture_output=True, text=True
    )
    if p.returncode != 0:
        raise CannotRun("not inside a git repository and RSR_ORCH_ROOT is unset")
    return Path(p.stdout.strip())


def resolve_commit(root: Path, rev: str) -> str | None:
    p = _git(root, "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}")
    return p.stdout.strip() if p.returncode == 0 else None


def resolve_base(root: Path, arg: str | None) -> str:
    """The commit the brief is checked against. No base is exit 3, not a guess."""
    rev = arg
    if rev is None:
        night = root / ".orchestrator" / "state" / "night.json"
        if night.exists():
            try:
                rev = json.loads(night.read_text()).get("base_sha")
            except (json.JSONDecodeError, AttributeError) as e:
                raise CannotRun(f"{night} is unreadable: {e}") from None
    if not rev:
        raise CannotRun(
            "no base: pass --base <sha>, or open a night "
            "(.orchestrator/state/night.json with base_sha)"
        )
    sha = resolve_commit(root, rev)
    if sha is None:
        raise CannotRun(f"base {rev!r} does not resolve to a commit in {root}")
    return sha


def show_at(root: Path, base: str, path: str) -> str | None:
    p = _git(root, "show", f"{base}:{path}")
    return p.stdout if p.returncode == 0 else None


def exists_at(root: Path, base: str, path: str) -> bool:
    return _git(root, "cat-file", "-e", f"{base}:{path.rstrip('/')}").returncode == 0


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #


def split_front_matter(text: str) -> tuple[dict | None, str, str | None]:
    """`(front, body, error)`. Front matter is a leading `---` ... `---` block."""
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return None, text, "no YAML front matter (the file must start with `---`)"
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            raw, body = "".join(lines[1:i]), "".join(lines[i + 1 :])
            try:
                front = yaml.safe_load(raw)
            except yaml.YAMLError as e:
                return None, body, f"front matter is not valid YAML: {e}"
            if not isinstance(front, dict):
                return None, body, "front matter is not a mapping"
            return front, body, None
    return None, text, "front matter is not closed by a second `---`"


def sections(body: str) -> list[tuple[str, str]]:
    """`(heading text, section text)` for every heading outside a code fence."""
    out: list[tuple[str, list[str]]] = []
    fence = False
    for line in body.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            fence = not fence
        m = None if fence else _HEADING.match(line)
        if m:
            out.append((m.group(2), []))
        elif out:
            out[-1][1].append(line)
    return [(h, "\n".join(ls).strip()) for h, ls in out]


def _nonempty(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return bool(v.strip())
    if isinstance(v, list | dict):
        return bool(v)
    return True


def resolve_key(data: Any, dotted: str) -> tuple[bool, Any]:
    cur = data
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and re.fullmatch(r"-?\d+", part):
            i = int(part)
            if not -len(cur) <= i < len(cur):
                return False, None
            cur = cur[i]
        else:
            return False, None
    return True, cur


def writer_in(cmd: str) -> str | None:
    for pat, label in WRITER_PATTERNS:
        if re.search(pat, cmd):
            return label
    return None


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #


def check_structure(front: dict, body: str, brief: Path) -> list[Finding]:
    out: list[Finding] = []
    for k in REQUIRED_KEYS:
        if k not in front:
            out.append(Finding("front-matter", f"missing key `{k}`"))
    for k in ("id", "item", "falsifier", "files_in_scope", "bar", "done_when", "do_not"):
        if k in front and not _nonempty(front[k]):
            out.append(Finding("front-matter", f"`{k}` is empty"))
    for k in ("anchors", "premises", "files_in_scope"):
        if k in front and front[k] is not None and not isinstance(front[k], list):
            out.append(Finding("front-matter", f"`{k}` must be a list"))
    bid = front.get("id")
    if isinstance(bid, str) and bid and brief.name != f"dispatch-{bid}.md":
        out.append(
            Finding(
                "front-matter",
                f"id {bid!r} but the file is {brief.name!r}; "
                f"a brief lives at docs/lab-notes/dispatch-<id>.md",
            )
        )

    secs = sections(body)
    for want in REQUIRED_HEADINGS:
        hits = [t for h, t in secs if h.lower().startswith(want.lower())]
        if not hits:
            out.append(Finding("heading", f"missing required heading `{want}`"))
        elif not any(hits):
            out.append(Finding("heading", f"section `{want}` is empty"))
    return out


def check_baseline(front: dict, root: Path, base: str) -> list[Finding]:
    claimed = front.get("baseline_sha")
    if not isinstance(claimed, str) or not claimed.strip():
        return [Finding("baseline", "`baseline_sha` is missing or empty")]
    sha = resolve_commit(root, claimed.strip())
    if sha is None:
        return [Finding("baseline", f"baseline_sha {claimed} does not resolve here")]
    if sha != base:
        return [
            Finding(
                "baseline",
                f"baseline_sha {claimed} is {sha[:12]}, the base is {base[:12]}. "
                f"A brief's baseline is HEAD when it was written, never its own commit.",
            )
        ]
    return []


def check_anchors(front: dict, body: str, root: Path, base: str) -> list[Finding]:
    out: list[Finding] = []
    declared: list[tuple[str, int]] = []
    cache: dict[str, str | None] = {}
    for i, a in enumerate(front.get("anchors") or []):
        if not isinstance(a, dict) or not {"path", "line", "expect"} <= set(a):
            out.append(Finding("anchor", f"anchors[{i}] needs path, line and expect"))
            continue
        path, line, expect = str(a["path"]), a["line"], str(a["expect"])
        if not isinstance(line, int) or isinstance(line, bool) or not expect:
            out.append(
                Finding("anchor", f"anchors[{i}] {path}: bad line or empty expect")
            )
            continue
        declared.append((path, line))
        if path not in cache:
            cache[path] = show_at(root, base, path)
        text = cache[path]
        if text is None:
            out.append(Finding("anchor", f"{path}:{line} -- no such file at base"))
            continue
        lines = text.splitlines()
        actual = lines[line - 1] if 1 <= line <= len(lines) else None
        if actual is not None and expect in actual:
            continue
        found = [str(n) for n, ln in enumerate(lines, 1) if expect in ln]
        where = f"found at :{', :'.join(found)}" if found else "not found in the file"
        got = "past end of file" if actual is None else repr(actual.strip())
        out.append(
            Finding(
                "anchor",
                f"{path}:{line} drifted -- expected {expect!r}, line reads {got}; "
                f"{where}",
            )
        )

    # Every `path.ext:N` in the body must be declared, or it is a line number
    # nobody checked -- the drift class, cited in prose.
    fence = False
    for n, line in enumerate(body.splitlines(), 1):
        if line.lstrip().startswith(("```", "~~~")):
            fence = not fence
        for m in _CITATION.finditer(line):
            cited, ln = m.group(1), int(m.group(2))
            if not any(
                ln == dl and (dp == cited or dp.endswith("/" + cited.lstrip("./")))
                for dp, dl in declared
            ):
                out.append(
                    Finding(
                        "anchor",
                        f"body line {n} cites {cited}:{ln}, which is not a declared "
                        f"anchor (declare it with `expect`, or drop the line number)",
                    )
                )
    return out


def check_files_in_scope(front: dict, root: Path, base: str) -> list[Finding]:
    out: list[Finding] = []
    for i, f in enumerate(front.get("files_in_scope") or []):
        if isinstance(f, str):
            path, new = f, False
        elif isinstance(f, dict) and "path" in f:
            path, new = str(f["path"]), bool(f.get("new", False))
        else:
            out.append(Finding("scope", f"files_in_scope[{i}] is not a path"))
            continue
        there = exists_at(root, base, path)
        if new and there:
            out.append(Finding("scope", f"{path} is declared new but exists at base"))
        elif not new and not there:
            out.append(
                Finding("scope", f"{path} does not exist at base (mark it `new: true`?)")
            )
    return out


def _check_ledger(p: dict, root: Path, base: str, label: str) -> list[Finding]:
    ledger, key = str(p["ledger"]), p.get("key")
    if not isinstance(key, str) or "expect" not in p:
        return [Finding("premise", f"{label}: a ledger premise needs `key` and `expect`")]
    text = show_at(root, base, ledger)
    if text is None:
        return [Finding("premise", f"{label}: {ledger} does not exist at base")]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return [Finding("premise", f"{label}: {ledger} is not JSON: {e}")]
    ok, got = resolve_key(data, key)
    if not ok:
        return [Finding("premise", f"{label}: {ledger} has no key `{key}`")]
    if got != p["expect"]:
        return [
            Finding(
                "premise",
                f"{label}: {ledger}:{key} is {got!r}, the brief expects {p['expect']!r}",
            )
        ]
    return []


def _run_check(p: dict, tree: Path, label: str) -> list[Finding]:
    cmd = str(p["check"])
    want_rc = p.get("expect_rc", 0)
    want_out = p.get("expect_stdout_contains")
    try:
        r = subprocess.run(
            ["/bin/bash", "-c", cmd],
            cwd=tree,
            capture_output=True,
            text=True,
            timeout=PREMISE_TIMEOUT_S,
            env={**os.environ, "RSR_ORCH_ROOT": str(tree)},
        )
    except subprocess.TimeoutExpired:
        return [Finding("premise", f"{label}: `{cmd}` timed out ({PREMISE_TIMEOUT_S}s)")]
    out: list[Finding] = []
    if r.returncode != want_rc:
        out.append(
            Finding(
                "premise",
                f"{label}: `{cmd}` exited {r.returncode}, expected {want_rc}"
                + (f"; stderr: {r.stderr.strip()[:200]}" if r.stderr.strip() else ""),
            )
        )
    if want_out is not None and str(want_out) not in r.stdout:
        out.append(
            Finding(
                "premise",
                f"{label}: stdout of `{cmd}` lacks {str(want_out)!r} "
                f"(got {r.stdout.strip()[:200]!r})",
            )
        )
    return out


class ThrowawayWorktree:
    """`git worktree add --detach <tmp> <base>`, removed on exit however we leave.

    Premise checks run HERE, never in the caller's checkout: the tree the brief is
    checked against is the base, not whatever happens to be on disk.
    """

    def __init__(self, root: Path, base: str):
        self.root, self.base = root, base
        self.path: Path | None = None

    def __enter__(self) -> Path:
        self.path = Path(tempfile.mkdtemp(prefix="lint-brief-"))
        p = _git(self.root, "worktree", "add", "--detach", str(self.path), self.base)
        if p.returncode != 0:
            shutil.rmtree(self.path, ignore_errors=True)
            raise CannotRun(f"could not create a throwaway worktree: {p.stderr.strip()}")
        return self.path

    def __exit__(self, *exc: object) -> None:
        if self.path is None:
            return
        _git(self.root, "worktree", "remove", "--force", str(self.path))
        shutil.rmtree(self.path, ignore_errors=True)
        _git(self.root, "worktree", "prune")


def check_premises(front: dict, root: Path, base: str) -> list[Finding]:
    """Findings in premise order. Shell checks share ONE throwaway worktree."""
    per: dict[int, list[Finding]] = {}
    runnable: list[tuple[int, str, dict]] = []
    for i, p in enumerate(front.get("premises") or []):
        claim = p.get("claim") if isinstance(p, dict) else None
        label = f"premises[{i}] ({claim!r})" if claim else f"premises[{i}]"
        if not isinstance(p, dict) or not _nonempty(claim):
            per[i] = [Finding("premise", f"{label}: every premise needs a `claim`")]
            continue
        has_check, has_ledger = "check" in p, "ledger" in p
        if has_check == has_ledger:
            per[i] = [Finding("premise", f"{label}: exactly one of `check` or `ledger`")]
            continue
        if has_ledger:
            per[i] = _check_ledger(p, root, base, label)
            continue
        writer = writer_in(str(p["check"]))
        if writer:
            per[i] = [
                Finding(
                    "premise",
                    f"{label}: rejected, not run -- `{p['check']}` contains {writer}. "
                    f"A premise check must be read-only.",
                )
            ]
            continue
        runnable.append((i, label, p))
    if runnable:
        with ThrowawayWorktree(root, base) as tree:
            for i, label, p in runnable:
                per[i] = _run_check(p, tree, label)
    return [f for i in sorted(per) for f in per[i]]


def lint(brief: Path, root: Path, base: str) -> list[Finding]:
    """Every finding in `brief` against commit `base`. Raises `CannotRun`."""
    if not brief.is_file():
        raise CannotRun(f"no brief at {brief}")
    front, body, err = split_front_matter(brief.read_text())
    if front is None:
        return [Finding("front-matter", err or "no front matter")]
    out = check_structure(front, body, brief)
    out += check_baseline(front, root, base)
    out += check_anchors(front, body, root, base)
    out += check_files_in_scope(front, root, base)
    out += check_premises(front, root, base)
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> Exit:
    ap = ArgumentParser(prog="lint_brief", description=__doc__.split("\n\n")[0])
    ap.add_argument("brief", type=Path)
    ap.add_argument("--base", default=None, help="default: night.json base_sha")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    a = ap.parse_args(argv)

    try:
        root = resolve_root()
        base = resolve_base(root, a.base)
        findings = lint(a.brief, root, base)
    except CannotRun as e:
        if a.json:
            print(json.dumps({"brief": str(a.brief), "exit": 3, "reason": str(e)}))
        return did_not_run(str(e))

    if a.json:
        print(
            json.dumps(
                {
                    "brief": str(a.brief),
                    "base": base,
                    "findings": [asdict(f) for f in findings],
                    "exit": 1 if findings else 0,
                },
                indent=2,
            )
        )
    else:
        for f in findings:
            print(f"FINDING [{f.kind}] {f.detail}")
        print(f"{a.brief}: {len(findings)} finding(s) at base {base[:12]}")
    return Exit.FAIL if findings else Exit.OK


if __name__ == "__main__":
    run_main(main)
