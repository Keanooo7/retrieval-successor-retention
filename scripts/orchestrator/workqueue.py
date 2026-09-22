"""The work queue: committed item files, and a deterministic readiness rule.

    export PYTHONPATH=scripts
    .venv/bin/python -m orchestrator.workqueue validate
    .venv/bin/python -m orchestrator.workqueue ready [--json] [--base SHA]
    .venv/bin/python -m orchestrator.workqueue render [--base SHA]
    .venv/bin/python -m orchestrator.workqueue set-status ID STATUS [--attempt]
    .venv/bin/python -m orchestrator.workqueue digest [--base SHA]

**The LLM never decides readiness.** An item is ready or not by a rule over git
objects at one commit (``base``), evaluated here and nowhere else:

* ``owner`` items are **never** ready. They are decisions only Brendan makes
  (RESEARCH-CONTEXT §12, CLAUDE.md "Prohibitions"); the queue batches them into the
  morning digest and never works around them.
* ``eng`` items are ready when every ``requires`` predicate holds at ``base``.
* ``exp`` items additionally need a pre-registration that exists **at base**, was
  introduced by a commit touching **only that file**, that commit an ancestor of
  base, and whose front matter carries ``falsifier``, ``decision_rule``, ``seeds``
  and ``thresholds``. CLAUDE.md: *"Pre-registration commits land before the
  experiment they govern, in their own commit, ordered ahead in git history. A
  threshold registered after seeing the data is not a threshold."*

Predicates (``requires``, each a one-key mapping):

``{ruling: R-id}``
    ``docs/owner/rulings/<R-id>.md`` exists **at base** and is *signed*: its front
    matter has ``id`` equal to the file stem, a ``date`` and a ``stated_in``
    (``docs/owner/rulings/README.md``). A file present only in the working tree is
    not at base, so it does not satisfy. The value may also be ``R-*-<slug>``: a
    ruling not yet made has no date, so ``*`` stands for exactly one ISO date.
``{item_done: id}``
    the item ``id`` has status ``done``.
``{sprint_gate: S<n>}``
    a signed ruling named ``R-<date>-sprint<n>-gate*`` exists at base. ROADMAP
    line 56: *"No sprint begins before its predecessor's gate answers"*; only the
    owner answers a gate.
``{prereg: path}``
    the pre-registration rule above.
``{path_exists_at_base: path}``
    ``path`` is in the tree of ``base``.

🔴 **§16: only weeks 1-4 are approved.** An item with ``sprint > 4`` is invalid;
``validate`` fails on it (exit 1) and ``ready``/``render`` refuse to run (exit 3)
while any invalid item exists -- a queue that skipped a bad item silently would be
a queue that could be made to schedule one by a typo in the check.

``set-status`` is the only programmatic writer of item files. It refuses to move an
owner item to anything but ``blocked``/``parked`` (an owner item is closed by a
ruling, not a status write) and refuses ``claimed`` for an item this module does
not compute as ready.

Exit codes are ``rsr.exit_codes`` (docs/gates.md): 0 ok, 1 invalid items, 2 the
queue is empty (nothing to compare), 3 refused / did not run.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import yaml

from rsr.exit_codes import ArgumentParser, Exit, refuse, run_main, status

__all__ = [
    "MAX_SPRINT",
    "Item",
    "Queue",
    "Readiness",
    "load_items",
    "main",
    "repo_root",
    "validate_item",
]

# §16: weeks 1-4 are approved; sprints 0..4 map onto them (ROADMAP "Sprint 0" ..
# "Sprint 4"). Everything past this is a projection, not a permission.
MAX_SPRINT = 4

CLASSES = ("owner", "eng", "exp")
WORKSTREAMS = ("W0", "W1", "W2", "W3", "W4")
LANES = ("none", "cpu-det", "mps", "battery", "agent-only")
STATUSES = (
    "blocked",
    "ready",
    "claimed",
    "running",
    "collecting",
    "verifying",
    "done",
    "parked",
)
PREDICATES = ("ruling", "item_done", "sprint_gate", "prereg", "path_exists_at_base")
PREREG_FIELDS = ("falsifier", "decision_rule", "seeds", "thresholds")
RULINGS_DIR = "docs/owner/rulings"
ITEMS_DIR = "docs/queue/items"

REQUIRED = (
    "id",
    "class",
    "sprint",
    "workstream",
    "lane",
    "slots",
    "requires",
    "status",
    "attempts",
    "source",
)
OPTIONAL = ("title", "brief", "owner_question", "answered_by")

_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_RULING_EXACT = re.compile(r"^R-\d{4}-\d{2}-\d{2}-[a-z0-9][a-z0-9-]*$")
_RULING_PATTERN = re.compile(r"^R-\*-[a-z0-9][a-z0-9*-]*$")
_DATED = re.compile(r"^R-(\d{4}-\d{2}-\d{2})-(.+)$")
_SPRINT_GATE = re.compile(r"^S([0-9]+)$")


# --------------------------------------------------------------------------- #
# root and git
# --------------------------------------------------------------------------- #


def _git_bin() -> str:
    for cand in ("/opt/homebrew/bin/git", shutil.which("git")):
        if cand and Path(cand).exists():
            return cand
    refuse(Exit.DID_NOT_RUN, "no git binary found")


def repo_root() -> Path:
    """``RSR_ORCH_ROOT`` if set, else the MAIN checkout (parent of the common git dir).

    Not the toplevel: inside ``.worktrees/<item>`` that is the worktree, and the
    queue and night.json would fork per worktree (same rule as ``loopcore.root``).
    """
    env = os.environ.get("RSR_ORCH_ROOT")
    if env:
        return Path(env).resolve()
    proc = subprocess.run(
        [_git_bin(), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        refuse(Exit.DID_NOT_RUN, "not in a git repository and RSR_ORCH_ROOT is unset")
    return Path(proc.stdout.strip()).resolve().parent


class Git:
    """Read-only git queries against one root, cached per (base, question)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.git = _git_bin()
        self._cache: dict[tuple[str, ...], Any] = {}

    def run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.git, "-C", str(self.root), *args],
            capture_output=True,
            text=True,
        )

    def resolve(self, rev: str) -> str | None:
        p = self.run("rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}")
        return p.stdout.strip() if p.returncode == 0 and p.stdout.strip() else None

    def exists(self, base: str, path: str) -> bool:
        key = ("exists", base, path)
        if key not in self._cache:
            self._cache[key] = (
                self.run("cat-file", "-e", f"{base}:{path}").returncode == 0
            )
        return self._cache[key]

    def show(self, base: str, path: str) -> str | None:
        key = ("show", base, path)
        if key not in self._cache:
            p = self.run("show", f"{base}:{path}")
            self._cache[key] = p.stdout if p.returncode == 0 else None
        return self._cache[key]

    def ls(self, base: str, directory: str) -> list[str]:
        key = ("ls", base, directory)
        if key not in self._cache:
            p = self.run("ls-tree", "--name-only", base, f"{directory.rstrip('/')}/")
            names = p.stdout.split() if p.returncode == 0 else []
            self._cache[key] = [Path(n).name for n in names]
        return self._cache[key]

    def introducing_commit(self, base: str, path: str) -> str | None:
        """The most recent commit reachable from base that added ``path``."""
        p = self.run("log", "--format=%H", "--diff-filter=A", base, "--", path)
        lines = p.stdout.split() if p.returncode == 0 else []
        return lines[0] if lines else None

    def touched(self, commit: str) -> set[str]:
        """Paths a commit changed against its first parent (or the empty tree).

        A merge commit reports nothing here, so it can never satisfy "touches only
        the PREREG file" -- the safe direction.
        """
        p = self.run("diff-tree", "--no-commit-id", "--name-only", "-r", "--root", commit)
        return set(p.stdout.split()) if p.returncode == 0 else set()

    def is_ancestor(self, a: str, b: str) -> bool:
        return self.run("merge-base", "--is-ancestor", a, b).returncode == 0


# --------------------------------------------------------------------------- #
# items
# --------------------------------------------------------------------------- #


def split_front_matter(text: str) -> tuple[dict[str, Any] | None, str]:
    """``(front matter, body)``; ``None`` if the text has no parseable front matter."""
    if not text.startswith("---\n"):
        return None, text
    end = text.find("\n---\n", 4)
    if end < 0:
        if text.endswith("\n---"):
            end = len(text) - 4
        else:
            return None, text
    try:
        meta = yaml.safe_load(text[4:end])
    except yaml.YAMLError:
        return None, text
    if not isinstance(meta, dict):
        return None, text
    return meta, text[end + 5 :]


@dataclass
class Item:
    path: Path
    meta: dict[str, Any]
    body: str
    errors: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return str(self.meta.get("id", self.path.stem))

    @property
    def klass(self) -> str:
        return str(self.meta.get("class"))

    @property
    def status(self) -> str:
        return str(self.meta.get("status"))

    @property
    def requires(self) -> list[dict[str, str]]:
        req = self.meta.get("requires") or []
        return req if isinstance(req, list) else []


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _valid_ruling_ref(v: str) -> bool:
    return bool(_RULING_EXACT.match(v) or _RULING_PATTERN.match(v))


def validate_item(item: Item) -> list[str]:
    """Schema errors for one item; ``[]`` if valid. Cross-item checks are in
    ``Queue.validate``."""
    m = item.meta
    errs: list[str] = []
    for k in REQUIRED:
        if k not in m:
            errs.append(f"missing field {k!r}")
    unknown = sorted(set(m) - set(REQUIRED) - set(OPTIONAL))
    if unknown:
        # A misspelt `requires` would otherwise read as "no predicates" -> ready.
        errs.append(f"unknown field(s) {unknown}")
    if errs:
        return errs

    if not (isinstance(m["id"], str) and _ID.match(m["id"])):
        errs.append(f"id {m['id']!r} is not a lowercase slug")
    elif m["id"] != item.path.stem:
        errs.append(f"id {m['id']!r} does not match file name {item.path.name!r}")
    if m["class"] not in CLASSES:
        errs.append(f"class {m['class']!r} not in {CLASSES}")
    if not _is_int(m["sprint"]) or m["sprint"] < 0:
        errs.append(f"sprint {m['sprint']!r} is not an integer >= 0")
    elif m["sprint"] > MAX_SPRINT:
        errs.append(
            f"sprint {m['sprint']} > {MAX_SPRINT}: §16 approves weeks 1-4 only; "
            f"everything downstream is a projection, not a permission"
        )
    if m["workstream"] not in WORKSTREAMS:
        errs.append(f"workstream {m['workstream']!r} not in {WORKSTREAMS}")
    if m["lane"] not in LANES:
        errs.append(f"lane {m['lane']!r} not in {LANES}")
    if not _is_int(m["slots"]) or m["slots"] < 0:
        errs.append(f"slots {m['slots']!r} is not an integer >= 0")
    elif m["lane"] == "none" and m["slots"] != 0:
        errs.append("lane 'none' takes slots 0")
    elif m["lane"] in LANES and m["lane"] != "none" and m["slots"] < 1:
        errs.append(f"lane {m['lane']!r} needs slots >= 1")
    if m["status"] not in STATUSES:
        errs.append(f"status {m['status']!r} not in {STATUSES}")
    if not _is_int(m["attempts"]) or m["attempts"] < 0:
        errs.append(f"attempts {m['attempts']!r} is not an integer >= 0")
    if not (isinstance(m["source"], str) and m["source"].strip()):
        errs.append("source is empty: every item cites the file:line it came from")
    if "brief" in m and not (isinstance(m["brief"], str) and m["brief"].strip()):
        errs.append("brief, when given, is a path")

    req = m["requires"]
    if not isinstance(req, list):
        errs.append("requires is not a list")
        req = []
    for p in req:
        if not (isinstance(p, dict) and len(p) == 1):
            errs.append(f"predicate {p!r} is not a one-key mapping")
            continue
        ((k, v),) = p.items()
        if k not in PREDICATES:
            errs.append(f"predicate {k!r} not in {PREDICATES}")
        elif not (isinstance(v, str) and v.strip()):
            errs.append(f"predicate {k!r} has no value")
        elif k == "ruling" and not _valid_ruling_ref(v):
            errs.append(f"ruling {v!r} is neither R-<date>-<slug> nor R-*-<slug>")
        elif k == "sprint_gate":
            g = _SPRINT_GATE.match(v)
            if not g or int(g.group(1)) > MAX_SPRINT:
                errs.append(f"sprint_gate {v!r} is not S0..S{MAX_SPRINT}")

    if m["class"] == "owner":
        q = m.get("owner_question")
        if not (isinstance(q, str) and q.strip()):
            errs.append("an owner item states owner_question: the exact decision needed")
        a = m.get("answered_by")
        if not (isinstance(a, str) and _valid_ruling_ref(a)):
            errs.append("an owner item names answered_by: the ruling that closes it")
        if m["lane"] != "none":
            errs.append("an owner item runs on no lane")
    else:
        for k in ("owner_question", "answered_by"):
            if k in m:
                errs.append(f"{k} is for owner items only")
    if m["class"] == "exp" and not any(
        isinstance(p, dict) and "prereg" in p for p in req
    ):
        errs.append("an exp item requires a prereg (CLAUDE.md: pre-registration first)")
    return errs


def load_items(root: Path) -> list[Item]:
    items: list[Item] = []
    d = root / ITEMS_DIR
    if not d.is_dir():
        return items
    for p in sorted(d.glob("*.md")):
        if p.name.lower() == "readme.md":
            continue
        meta, body = split_front_matter(p.read_text())
        if meta is None:
            items.append(Item(p, {}, "", ["no parseable YAML front matter"]))
            continue
        it = Item(p, meta, body)
        it.errors = validate_item(it)
        items.append(it)
    return items


# --------------------------------------------------------------------------- #
# readiness
# --------------------------------------------------------------------------- #


@dataclass
class Readiness:
    ready: bool
    unmet: list[str]


class Queue:
    def __init__(self, root: Path, base: str | None = None) -> None:
        self.root = root
        self.items = load_items(root)
        self.by_id = {i.id: i for i in self.items}
        self.git = Git(root)
        self._base_arg = base

    # ---- validation -------------------------------------------------------- #

    def validate(self) -> dict[str, list[str]]:
        bad: dict[str, list[str]] = {}
        seen: dict[str, Path] = {}
        for it in self.items:
            errs = list(it.errors)
            if it.id in seen:
                errs.append(f"duplicate id (also {seen[it.id].name})")
            seen[it.id] = it.path
            for p in it.requires:
                if (
                    isinstance(p, dict)
                    and "item_done" in p
                    and p["item_done"] not in {i.id for i in self.items}
                ):
                    errs.append(f"item_done {p['item_done']!r} names no item")
            if errs:
                bad[it.path.name] = errs
        return bad

    # ---- base -------------------------------------------------------------- #

    def base(self) -> str:
        rev = self._base_arg
        if rev is None:
            night = self.root / ".orchestrator" / "state" / "night.json"
            if night.exists():
                try:
                    rev = json.loads(night.read_text()).get("base_sha")
                except (OSError, json.JSONDecodeError):
                    refuse(Exit.DID_NOT_RUN, f"{night} is not readable JSON")
            if not rev:
                rev = "HEAD"
        sha = self.git.resolve(rev)
        if sha is None:
            refuse(Exit.DID_NOT_RUN, f"base {rev!r} does not resolve to a commit")
        self._base_arg = sha
        return sha

    # ---- predicates -------------------------------------------------------- #

    def _signed(self, base: str, name: str) -> str | None:
        """``None`` if the ruling file is signed at base, else why not."""
        text = self.git.show(base, f"{RULINGS_DIR}/{name}")
        if text is None:
            return "absent at base"
        meta, _ = split_front_matter(text)
        if meta is None:
            return "present at base but has no front matter"
        stem = name[:-3]
        if meta.get("id") != stem:
            return f"present at base but front-matter id {meta.get('id')!r} != {stem!r}"
        for k in ("date", "stated_in"):
            if not meta.get(k):
                return f"present at base but unsigned (no {k})"
        return None

    def _ruling_matches(self, base: str, ref: str) -> tuple[bool, str]:
        names = [n for n in self.git.ls(base, RULINGS_DIR) if n.endswith(".md")]
        if "*" not in ref:
            if f"{ref}.md" not in names:
                return False, f"ruling {ref}: absent at {base[:9]}"
            why = self._signed(base, f"{ref}.md")
            return (why is None), f"ruling {ref}: {why}"
        # `R-*-slug`: `*` is exactly one ISO date, then the rest is an fnmatch.
        tail = ref[len("R-*-") :]
        candidates = []
        for n in names:
            m = _DATED.match(n[:-3])
            if m and fnmatchcase(m.group(2), tail):
                candidates.append(n)
        if not candidates:
            return False, f"ruling {ref}: no matching ruling at {base[:9]}"
        reasons = []
        for n in sorted(candidates):
            why = self._signed(base, n)
            if why is None:
                return True, ""
            reasons.append(f"{n[:-3]} {why}")
        return False, f"ruling {ref}: " + "; ".join(reasons)

    def _prereg(self, base: str, path: str) -> tuple[bool, str]:
        if not self.git.exists(base, path):
            return False, f"prereg {path}: absent at {base[:9]}"
        c = self.git.introducing_commit(base, path)
        if c is None:
            return False, f"prereg {path}: no introducing commit found"
        touched = self.git.touched(c)
        if touched != {path}:
            others = sorted(touched - {path})
            return False, (
                f"prereg {path}: introduced by {c[:9]} together with {others or '?'}; "
                f"a pre-registration lands in its own commit"
            )
        if not self.git.is_ancestor(c, base):
            return False, f"prereg {path}: {c[:9]} is not an ancestor of {base[:9]}"
        meta, _ = split_front_matter(self.git.show(base, path) or "")
        if meta is None:
            return False, f"prereg {path}: no front matter"
        missing = [k for k in PREREG_FIELDS if meta.get(k) in (None, "", [], {})]
        if missing:
            return False, f"prereg {path}: front matter lacks {missing}"
        return True, ""

    def predicate(self, base: str, p: dict[str, str]) -> tuple[bool, str]:
        ((k, v),) = p.items()
        if k == "ruling":
            return self._ruling_matches(base, v)
        if k == "sprint_gate":
            n = int(v[1:])
            ok, why = self._ruling_matches(base, f"R-*-sprint{n}-gate*")
            return ok, "" if ok else f"sprint_gate {v}: {why}"
        if k == "item_done":
            dep = self.by_id.get(v)
            if dep is None:
                return False, f"item_done {v}: no such item"
            if dep.status != "done":
                return False, f"item_done {v}: status is {dep.status}"
            return True, ""
        if k == "prereg":
            return self._prereg(base, v)
        if k == "path_exists_at_base":
            ok = self.git.exists(base, v)
            return ok, "" if ok else f"path {v}: absent at {base[:9]}"
        return False, f"unknown predicate {k}"  # validate() refuses first

    def readiness(self, item: Item) -> Readiness:
        base = self.base()
        unmet: list[str] = []
        if item.klass == "owner":
            # Never ready: an owner decision is batched to the digest, never run.
            return Readiness(False, ["owner decision: never dispatched"])
        if item.status not in ("blocked", "ready"):
            unmet.append(f"status is {item.status}")
        for p in item.requires:
            ok, why = self.predicate(base, p)
            if not ok:
                unmet.append(why)
        if item.klass == "exp" and not any("prereg" in p for p in item.requires):
            unmet.append("exp item without a prereg predicate")
        return Readiness(not unmet, unmet)

    def answered(self, item: Item) -> tuple[bool, str]:
        """Whether an owner item's ``answered_by`` ruling is signed at base."""
        return self._ruling_matches(self.base(), str(item.meta.get("answered_by")))

    def ready_items(self) -> list[Item]:
        out = [i for i in self.items if self.readiness(i).ready]
        return sorted(out, key=lambda i: (i.meta["sprint"], i.meta["workstream"], i.id))


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #


def _set_fields(text: str, updates: dict[str, Any]) -> str:
    """Rewrite scalar front-matter lines in place, keeping every other byte."""
    end = text.find("\n---\n", 4)
    head, rest = text[: end + 1], text[end + 1 :]
    for k, v in updates.items():
        line = f"{k}: {v}"
        pat = re.compile(rf"^{re.escape(k)}:.*$", re.MULTILINE)
        head, n = pat.subn(line, head, count=1)
        if n == 0:
            head = head + line + "\n"
    return head + rest


def _locked(root: Path):
    locks = root / ".orchestrator" / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    fh = open(locks / "workqueue.lock", "w")  # noqa: SIM115 -- held for the write
    fcntl.flock(fh, fcntl.LOCK_EX)
    return fh


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _refuse_if_invalid(q: Queue) -> None:
    bad = q.validate()
    if bad:
        for name, errs in bad.items():
            for e in errs:
                print(f"INVALID {name}: {e}", file=sys.stderr)
        refuse(
            Exit.DID_NOT_RUN,
            f"{len(bad)} invalid item(s); run `validate`. The queue does not "
            f"schedule around an invalid item.",
        )


def cmd_validate(q: Queue, args: Any) -> Exit:
    if not q.items:
        print("UNKNOWN: no items under docs/queue/items/", file=sys.stderr)
        return Exit.UNKNOWN
    bad = q.validate()
    for name, errs in bad.items():
        for e in errs:
            print(f"INVALID {name}: {e}")
    print(f"{len(q.items) - len(bad)}/{len(q.items)} items valid")
    return Exit.FAIL if bad else Exit.OK


def _row(q: Queue, it: Item) -> dict[str, Any]:
    r = q.readiness(it)
    return {
        "id": it.id,
        "class": it.klass,
        "sprint": it.meta["sprint"],
        "workstream": it.meta["workstream"],
        "lane": it.meta["lane"],
        "slots": it.meta["slots"],
        "status": it.status,
        "attempts": it.meta["attempts"],
        "brief": it.meta.get("brief"),
        "ready": r.ready,
        "unmet": r.unmet,
    }


def cmd_ready(q: Queue, args: Any) -> Exit:
    if not q.items:
        print("UNKNOWN: no items under docs/queue/items/", file=sys.stderr)
        return Exit.UNKNOWN
    _refuse_if_invalid(q)
    rows = [_row(q, i) for i in q.ready_items()]
    if args.json:
        print(json.dumps({"base_sha": q.base(), "ready": rows}, indent=2))
    else:
        for r in rows:
            print(r["id"])
    return Exit.OK


def cmd_render(q: Queue, args: Any) -> Exit:
    if not q.items:
        print("UNKNOWN: no items under docs/queue/items/", file=sys.stderr)
        return Exit.UNKNOWN
    _refuse_if_invalid(q)
    rows = [_row(q, i) for i in sorted(q.items, key=lambda i: i.id)]
    out = q.root / ".orchestrator" / "queue.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(out, json.dumps({"base_sha": q.base(), "items": rows}, indent=2) + "\n")
    print(f"wrote {out} ({sum(r['ready'] for r in rows)} ready of {len(rows)})")
    return Exit.OK


def cmd_set_status(q: Queue, args: Any) -> Exit:
    it = q.by_id.get(args.id)
    if it is None:
        refuse(Exit.DID_NOT_RUN, f"no item {args.id!r}")
    if it.errors:
        refuse(Exit.DID_NOT_RUN, f"item {args.id} is invalid: {it.errors}")
    if it.klass == "owner" and args.status not in ("blocked", "parked"):
        refuse(
            Exit.DID_NOT_RUN,
            f"{args.id} is an owner item: it is closed by a ruling in "
            f"{RULINGS_DIR}/ ({it.meta.get('answered_by')}), not by a status write",
        )
    if args.status == "claimed":
        r = q.readiness(it)
        if not r.ready:
            refuse(Exit.DID_NOT_RUN, f"{args.id} is not ready: {r.unmet}")
    fh = _locked(q.root)
    try:
        text = it.path.read_text()
        meta, _ = split_front_matter(text)
        updates: dict[str, Any] = {"status": args.status}
        if args.attempt:
            updates["attempts"] = int((meta or {}).get("attempts", 0)) + 1
        _atomic_write(it.path, _set_fields(text, updates))
    finally:
        fh.close()
    print(
        f"{args.id}: {it.status} -> {args.status}"
        + (f" (attempts {updates['attempts']})" if args.attempt else "")
    )
    return Exit.OK


def digest_markdown(q: Queue) -> str:
    base = q.base()
    lines = [f"## Work queue at `{base[:9]}`", ""]
    owner = [i for i in q.items if i.klass == "owner"]
    open_owner = []
    for i in owner:
        ok, _ = q.answered(i)
        if not ok:
            open_owner.append(i)
    lines += [f"### Decisions only Brendan can make ({len(open_owner)} open)", ""]
    for i in sorted(open_owner, key=lambda i: (i.meta["sprint"], i.id)):
        q_text = " ".join(str(i.meta.get("owner_question", "")).split())
        name = str(i.meta["answered_by"]).replace("R-*-", "R-<YYYY-MM-DD>-", 1)
        lines.append(
            f"- **{i.id}** (S{i.meta['sprint']}, {i.meta['workstream']}): {q_text} "
            f"-- answer as `{RULINGS_DIR}/{name.rstrip('*')}.md`; "
            f"source `{i.meta['source']}`"
        )
    lines.append("")
    blocked = []
    for i in q.items:
        if i.klass == "owner" or i.status in ("done", "parked"):
            continue
        r = q.readiness(i)
        if not r.ready and i.status in ("blocked", "ready"):
            blocked.append((i, r.unmet))
    lines += [f"### Blocked ({len(blocked)})", ""]
    for i, unmet in sorted(blocked, key=lambda t: (t[0].meta["sprint"], t[0].id)):
        lines.append(
            f"- **{i.id}** ({i.klass}, S{i.meta['sprint']}): " + "; ".join(unmet)
        )
    lines.append("")
    parked = [i for i in q.items if i.status == "parked"]
    if parked:
        lines += [f"### Parked ({len(parked)})", ""]
        for i in sorted(parked, key=lambda i: i.id):
            lines.append(f"- **{i.id}** (attempts {i.meta.get('attempts')})")
        lines.append("")
    return "\n".join(lines)


def cmd_digest(q: Queue, args: Any) -> Exit:
    if not q.items:
        print("UNKNOWN: no items under docs/queue/items/", file=sys.stderr)
        return Exit.UNKNOWN
    _refuse_if_invalid(q)
    print(digest_markdown(q))
    return Exit.OK


def main(argv: list[str] | None = None) -> Exit:
    ap = ArgumentParser(prog="orchestrator.workqueue")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate")
    p = sub.add_parser("ready")
    p.add_argument("--json", action="store_true")
    p.add_argument("--base", default=None)
    p = sub.add_parser("render")
    p.add_argument("--base", default=None)
    p = sub.add_parser("set-status")
    p.add_argument("id")
    p.add_argument("status", choices=STATUSES)
    p.add_argument("--attempt", action="store_true", help="increment attempts")
    p.add_argument("--base", default=None)
    p = sub.add_parser("digest")
    p.add_argument("--base", default=None)
    args = ap.parse_args(argv)
    q = Queue(repo_root(), getattr(args, "base", None))
    handler = {
        "validate": cmd_validate,
        "ready": cmd_ready,
        "render": cmd_render,
        "set-status": cmd_set_status,
        "digest": cmd_digest,
    }[args.cmd]
    return status(handler(q, args))


if __name__ == "__main__":
    run_main(main)
