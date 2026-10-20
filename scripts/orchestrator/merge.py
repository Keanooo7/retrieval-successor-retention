"""Merge a branch into `night/<date>` -- and nowhere else -- behind three gates.

    PYTHONPATH=scripts .venv/bin/python -m orchestrator.merge <item_id>
    PYTHONPATH=scripts .venv/bin/python -m orchestrator.merge <prefix>/<name>

A bare `<item_id>` means `run/<item_id>` (the original CLI, unchanged). A full
branch name is accepted for the prefixes `run/`, `fix/`, `eng/`, `feat/` and
`docs/` (PLAN-v4 §4 T5(b)); any other prefix is refused (3).

R-2026-09-22-night-branch: unattended work merges only into the night branch, with
a merge commit, in the manager's worktree (`.worktrees/_mgr`). **This module never
names `main`.** The owner merges the night branch once each morning.

Gate 1 -- **verification** (`run/` only): `runs/<item_id>/verification.json` with
`status: ok`, committed on the night branch (where the manager's cycle commits it)
or on the run branch. Missing or `inconclusive` -> refused (3); `failed` -> 1.

Gate 2 -- **the frozen-diff guard**, over `git diff <base_sha>..<branch>`, every
prefix.
Trips on any change to:

* `preregistration/**` -- signed, not re-openable;
* an **existing** `runs/<id>/` directory (one present at `base_sha`) -- new run
  directories are allowed; an old record is never edited;
* `runs/canary/baseline*` -- the canary's reference;
* `docs/owner/**` and `docs/spec/**`;
* a **FROZEN** definition in `src/rsr/constants.py` -- compared definition by
  definition (AST source segments), so a changed line inside a FROZEN entry, a
  FROZEN entry removed, one reclassified, or a new one added all trip it; an
  unparseable file trips it too (fail closed).

A trip is the most serious rejection there is (rsr-manager.md, "Frozen things":
*"it stops the night"*): `.orchestrator/HALT` is written with the reason, the
owner is notified, and the exit is 1.

Gate 3 -- **the review gate** (PLAN-v4 §4 T5(b)), every prefix: a committed
adversarial-review record `docs/reviews/<branch-slug>/<reviewed_head[:12]>.md` on
the branch. Format and rule: `docs/review-records.md`. Missing, stale (code changed
since the reviewed head), malformed, self-reviewed, or MERGE WITH FIXES (never
mergeable on its own; the fixes need a MERGE re-review) -> refused (3, naming what
is missing);
DO NOT MERGE -> 1. It runs after the guard, so a frozen edit HALTs the night
whether or not anyone reviewed it.

Exit: 0 merged · 1 guard trip, failed verification, or a merge conflict (aborted)
· 3 a precondition (no night, no branch, an unknown prefix, no verification, no
valid review record, a dirty manager worktree, a branch not descended from the
base). DO NOT MERGE is 1.
"""

from __future__ import annotations

import ast
import fnmatch
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from orchestrator import loopcore as lc
from orchestrator import notify
from orchestrator.night import mgr_worktree
from rsr.exit_codes import ArgumentParser, Exit, run_main, status

CONSTANTS = "src/rsr/constants.py"

#: Paths no unattended branch may change at all.
FROZEN_GLOBS = (
    "preregistration/*",
    "docs/owner/*",
    "docs/spec/*",
    "runs/canary/baseline*",
)

#: Branch prefixes this module merges (T5(b)). Only `run/` needs verification.json.
PREFIXES = ("run", "fix", "eng", "feat", "docs")

#: T5(b) review records live here, one directory per branch slug.
REVIEW_DIR = "docs/reviews"
VERDICTS = ("MERGE", "MERGE WITH FIXES", "DO NOT MERGE")
REVIEW_FIELDS = ("branch", "reviewed_head", "author", "reviewer", "verdict")


def frozen_definitions(source: str | None) -> dict[str, str] | None:
    """{name: source segment} of every `Definition(..., klass=Klass.FROZEN, ...)`.
    None when the source does not parse -- the caller treats that as a trip."""
    if source is None:
        return {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Definition"
        ):
            continue
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        klass = kw.get("klass")
        if not (isinstance(klass, ast.Attribute) and klass.attr == "FROZEN"):
            continue
        name = kw.get("name")
        key = name.value if isinstance(name, ast.Constant) else ast.unparse(node)[:60]
        out[str(key)] = ast.get_source_segment(source, node) or ast.unparse(node)
    return out


def _show(root: Path, rev: str, path: str) -> str | None:
    proc = lc.git(root, "show", f"{rev}:{path}")
    return proc.stdout if proc.returncode == 0 else None


def frozen_violations(root: Path, base: str, head: str) -> list[str]:
    diff = lc.git(root, "diff", "--name-status", "--no-renames", f"{base}..{head}")
    if diff.returncode != 0:
        return [f"git diff {base[:12]}..{head} failed -- cannot clear the guard"]
    base_runs = {
        line.split("/")[1]
        for line in lc.git(
            root, "ls-tree", "-r", "--name-only", base, "--", "runs"
        ).stdout.splitlines()
        if line.count("/") >= 2
    }
    out = []
    for line in diff.stdout.splitlines():
        if not line.strip():
            continue
        st, path = line.split("\t", 1)
        if any(fnmatch.fnmatch(path, g) for g in FROZEN_GLOBS):
            out.append(f"{st} {path}: a frozen path")
        elif path.startswith("runs/") and path.count("/") >= 2:
            run = path.split("/")[1]
            if run in base_runs:
                out.append(
                    f"{st} {path}: an existing run record (runs/{run}/ is at base)"
                )
        elif path == CONSTANTS:
            before = frozen_definitions(_show(root, base, CONSTANTS))
            after = frozen_definitions(_show(root, head, CONSTANTS))
            if before is None or after is None:
                out.append(f"{st} {path}: does not parse -- FROZEN entries unverifiable")
                continue
            for name in sorted(set(before) | set(after)):
                if name not in after:
                    out.append(f"{path}: FROZEN {name!r} removed or reclassified")
                elif name not in before:
                    out.append(f"{path}: FROZEN {name!r} added")
                elif before[name] != after[name]:
                    out.append(f"{path}: FROZEN {name!r} changed")
    return out


def read_verification(root: Path, item: str) -> tuple[dict | None, str]:
    """(record, where) from the night branch first, then the run branch."""
    night = lc.read_night(root) or {}
    rel = f"runs/{item}/verification.json"
    for rev in (night.get("branch"), f"run/{item}"):
        if not rev:
            continue
        text = _show(root, rev, rel)
        if text is None:
            continue
        try:
            return json.loads(text), f"{rev}:{rel}"
        except json.JSONDecodeError:
            return {"status": "unparseable"}, f"{rev}:{rel}"
    return None, rel


def trip(
    root: Path, label: str, found: list[str], *, dry_run: bool, branch: str | None = None
) -> tuple[Exit, str]:
    branch = branch or f"run/{label}"
    reason = f"frozen-diff guard tripped merging {label}: " + "; ".join(found)
    if not dry_run:
        lc.halt(root, reason)
        body = "\n".join(
            [f"The merge of `{branch}` into the night branch was refused.", ""]
            + [f"- {v}" for v in found]
            + [
                "",
                "`.orchestrator/HALT` is written; the loop stays stopped until you",
                "remove it.",
            ]
        )
        notify.post(root, f"HALT: frozen-diff guard tripped ({label})", body)
    return Exit.FAIL, reason


# --- T5(b): the review gate ------------------------------------------------- #


def resolve_target(target: str) -> tuple[str | None, str | None, str]:
    """(branch, run item or None, error). A bare id is `run/<id>` (the old CLI)."""
    if "/" not in target:
        return f"run/{target}", target, ""
    prefix, rest = target.split("/", 1)
    if prefix not in PREFIXES or not rest:
        accepted = ", ".join(f"{p}/" for p in PREFIXES)
        return None, None, f"branch prefix of {target!r} is not one of {accepted}"
    return target, (rest if prefix == "run" else None), ""


def review_exempt(path: str, run_item: str | None = None) -> bool:
    """Paths that may change after the reviewed head without staling the record.
    Only the merging `run/<item>`'s OWN verification.json is exempt (review
    MINOR-2): another run's record would ride into the night branch unreviewed and
    then satisfy gate 1 for that run. No verification.json is exempt off `run/`."""
    if path.startswith((f"{REVIEW_DIR}/", ".orchestrator/outbox/")):
        return True
    return run_item is not None and path == f"runs/{run_item}/verification.json"


def _full_sha(value: str) -> bool:
    return len(value) == 40 and all(c in "0123456789abcdef" for c in value)


def front_matter(text: str) -> dict[str, str] | None:
    """`key: value` lines between a leading `---` and the next `---`; None if absent,
    unterminated, or a line is not `key: value`."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    out: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return out
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            return None
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip().strip("\"'")
    return None


def _commit(root: Path, rev: str) -> str | None:
    proc = lc.git(root, "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}")
    return proc.stdout.strip() if proc.returncode == 0 else None


def _ancestor(root: Path, a: str, b: str) -> bool:
    """a is b or an ancestor of b."""
    return lc.git(root, "merge-base", "--is-ancestor", a, b).returncode == 0


def review_gate(
    root: Path, branch: str, base: str, head: str, run_item: str | None = None
) -> tuple[Exit, str]:
    """(OK, record path) or a refusal naming what is missing. docs/review-records.md."""
    rdir = f"{REVIEW_DIR}/{branch.replace('/', '-')}"
    want = f"{rdir}/{head[:12]}.md"
    listing = lc.git(root, "ls-tree", "--name-only", f"{head}:{rdir}")
    names = sorted(
        n
        for n in listing.stdout.splitlines()
        if listing.returncode == 0 and n.endswith(".md")
    )
    if not names:
        return Exit.DID_NOT_RUN, (
            f"no review record for {branch}@{head[:12]}: expected {want} (or a record "
            f"of an ancestor with only exempt paths changed since) committed on "
            f"{branch} -- see docs/review-records.md; not merged"
        )
    log = lc.git(root, "log", "--format=%an%n%ae", f"{base}..{head}")
    git_authors = {a.strip().casefold() for a in log.stdout.splitlines() if a.strip()}
    malformed: list[str] = []
    elsewhere: list[str] = []
    usable: list[tuple[int, str, dict[str, str]]] = []
    for name in names:
        path = f"{rdir}/{name}"
        fm = front_matter(_show(root, head, path) or "")
        if fm is None:
            malformed.append(f"{path}: no `---` front matter")
            continue
        missing = [f for f in REVIEW_FIELDS if not fm.get(f)]
        if missing:
            malformed.append(f"{path}: missing {', '.join(missing)}")
            continue
        rh, reviewer = fm["reviewed_head"], fm["reviewer"].strip()
        bad = []
        if fm["branch"] != branch:
            bad.append(f"branch is {fm['branch']!r}, not {branch!r}")
        if not _full_sha(rh):
            bad.append(f"reviewed_head {rh!r} is not a full sha")
        elif name != f"{rh[:12]}.md":
            bad.append(f"file name {name} is not reviewed_head[:12].md ({rh[:12]}.md)")
        # Review MAJOR-1: informational only, but a present value is held to the
        # same standard as reviewed_head -- a ref name (`fix/x`, `HEAD`) moves.
        fixes = fm.get("fixes_verified_at")
        if fixes is not None and not (_full_sha(fixes) and _commit(root, fixes)):
            bad.append(f"fixes_verified_at {fixes!r} is not the full sha of a commit")
        if fm["verdict"] not in VERDICTS:
            bad.append(f"verdict {fm['verdict']!r} is not one of {', '.join(VERDICTS)}")
        if reviewer.casefold() == fm["author"].strip().casefold():
            bad.append(f"reviewer {reviewer!r} is the author")
        elif reviewer.casefold() in git_authors:
            bad.append(f"reviewer {reviewer!r} is a git author of {branch}")
        if bad:
            malformed.append(f"{path}: " + "; ".join(bad))
            continue
        full = _commit(root, rh)
        if full is None or not _ancestor(root, full, head):
            elsewhere.append(
                f"{path}: reviewed_head {rh[:12]} is not an ancestor of {head[:12]}"
            )
            continue
        dist = int(lc.git_out(root, "rev-list", "--count", f"{full}..{head}"))
        usable.append((dist, path, fm))
    if malformed:
        # Fail closed: a record the gate cannot read is not skipped past.
        return Exit.DID_NOT_RUN, "malformed review record -- " + "; ".join(malformed)
    if not usable:
        return Exit.DID_NOT_RUN, (
            f"no review record for {branch}@{head[:12]} covers its history: "
            + "; ".join(elsewhere)
            + f" -- expected {want}; not merged"
        )
    _, path, fm = min(usable)  # the record nearest head governs
    who = f"{path} (reviewer {fm['reviewer']})"
    if fm["verdict"] == "DO NOT MERGE":
        return Exit.FAIL, f"review verdict DO NOT MERGE: {who} -- not merged"
    if fm["verdict"] == "MERGE WITH FIXES":
        # Review MAJOR-2, PM decision 2026-09-30: never mergeable on its own. The
        # fixes are code no record covers, and `fixes_verified_at` is writable by
        # anyone (docs/reviews/ is exempt), so it is an attestation the gate cannot
        # tell from the author's own. A MERGE record at or after the fixes governs
        # instead (the nearest record wins).
        return Exit.DID_NOT_RUN, (
            f"review verdict MERGE WITH FIXES is not mergeable on its own: {who} -- "
            f"the fixes need a re-review, a MERGE record at or after them "
            f"(expected {want}); not merged"
        )
    anchor = fm["reviewed_head"]
    diff = lc.git(root, "diff", "--name-only", "--no-renames", f"{anchor}..{head}")
    if diff.returncode != 0:
        return Exit.DID_NOT_RUN, f"git diff {anchor[:12]}..{head[:12]} failed: {who}"
    unreviewed = [
        p for p in diff.stdout.splitlines() if p and not review_exempt(p, run_item)
    ]
    if unreviewed:
        return Exit.DID_NOT_RUN, (
            f"review is stale: {who} covers {anchor[:12]}, but {anchor[:12]}.."
            f"{head[:12]} changes non-exempt paths: {', '.join(unreviewed)} -- "
            f"re-review at {head[:12]}; not merged"
        )
    return Exit.OK, path


def _verification_gate(root: Path, item: str) -> tuple[Exit, str]:
    rec, where = read_verification(root, item)
    if rec is None:
        return Exit.DID_NOT_RUN, f"no verification record ({where}) -- not merged"
    vstatus = rec.get("status")
    if vstatus == "failed":
        return Exit.FAIL, f"verification failed ({where}) -- not merged"
    if vstatus != "ok":
        return Exit.DID_NOT_RUN, f"verification is {vstatus!r} ({where}) -- not merged"
    return Exit.OK, where


def merge_item(root: Path, item: str, *, dry_run: bool = False) -> tuple[Exit, str]:
    """`item` is a run id (-> `run/<item>`) or a full `<prefix>/<name>` branch."""
    night = lc.read_night(root)
    if not night or night.get("closed"):
        return Exit.DID_NOT_RUN, "no open night"
    base, target = night["base_sha"], night["branch"]
    branch, run_item, err = resolve_target(item)
    if branch is None:
        return Exit.DID_NOT_RUN, err
    head = lc.git(root, "rev-parse", "--verify", f"refs/heads/{branch}")
    if head.returncode != 0:
        return Exit.DID_NOT_RUN, f"no branch {branch}"
    head_sha = head.stdout.strip()
    if lc.git(root, "merge-base", "--is-ancestor", base, head_sha).returncode != 0:
        return Exit.DID_NOT_RUN, f"{branch} does not descend from base {base[:12]}"
    passed = []
    if run_item is not None:
        code, where = _verification_gate(root, run_item)
        if code != Exit.OK:
            return code, where
        passed.append(f"verified ({where})")
    found = frozen_violations(root, base, head_sha)
    if found:
        label = run_item if run_item is not None else branch
        return trip(root, label, found, dry_run=dry_run, branch=branch)
    code, record = review_gate(root, branch, base, head_sha, run_item)
    if code != Exit.OK:
        return code, record
    passed.append(f"reviewed ({record})")
    if dry_run:
        return Exit.OK, f"DRY_RUN: would merge {branch} ({head_sha[:12]}) into {target}"
    wt = mgr_worktree(root)
    if not wt.exists():
        return Exit.DID_NOT_RUN, f"no manager worktree at {wt}"
    on = lc.git(wt, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if on != target:
        return Exit.DID_NOT_RUN, f"{wt} is on {on!r}, not {target}"
    if lc.git(wt, "status", "--porcelain", "--untracked-files=no").stdout.strip():
        return Exit.DID_NOT_RUN, f"{wt} has uncommitted changes"
    msg = f"merge(night {night['date']}): {branch} -- " + ", ".join(passed)
    proc = lc.git(wt, "merge", "--no-ff", "--no-edit", "-m", msg, head_sha)
    if proc.returncode != 0:
        lc.git(wt, "merge", "--abort")
        return Exit.FAIL, f"merge of {branch} conflicted and was aborted: {proc.stdout}"
    return Exit.OK, f"merged {branch} ({head_sha[:12]}) into {target}"


def main() -> Exit:
    ap = ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "item_id",
        help="a run id (merges run/<id>), or a full fix/, eng/, feat/, docs/ or run/ "
        "branch name",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dry = args.dry_run or os.environ.get("DRY_RUN") == "1"
    code, msg = merge_item(lc.root(), args.item_id, dry_run=dry)
    print(msg, file=sys.stdout if code == Exit.OK else sys.stderr)
    return status(code)


if __name__ == "__main__":
    run_main(main)
