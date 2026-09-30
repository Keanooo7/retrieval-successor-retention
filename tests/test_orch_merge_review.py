"""`orchestrator.merge` -- the T5(b) review gate and the non-`run/` prefixes.

PLAN-v4 §4 T5(b): the merge refuses a branch that has no committed adversarial-
review record. The record format and the gate are specified in
`docs/review-records.md`; in short, merging `branch@head` needs
`docs/reviews/<branch-slug>/<reviewed_head[:12]>.md` on the branch, whose
`reviewed_head` is `head` or an ancestor of it with only exempt paths
(`docs/reviews/`, `runs/<id>/verification.json`, `.orchestrator/outbox/`) changed
since, a reviewer who is not the author, and a verdict of MERGE (or MERGE WITH
FIXES with `fixes_verified_at` an ancestor of head). Missing / stale / malformed
-> 3 naming what is missing; DO NOT MERGE -> 1.

`fix/`, `eng/`, `feat/`, `docs/` branches go through the same merge; only `run/`
still needs `runs/<id>/verification.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "tests"))

from orchestrator import loopcore as lc  # noqa: E402
from orchestrator import night  # noqa: E402

from _orch_loop_helpers import NIGHT, Orch  # noqa: E402
from rsr.exit_codes import Exit  # noqa: E402


@pytest.fixture
def orch(tmp_path, monkeypatch):
    o = Orch(tmp_path, monkeypatch)
    code, msg = night.open_night(o.root, NIGHT, lc.Window())
    assert code == Exit.OK, msg
    o.mgr = night.mgr_worktree(o.root)
    return o


def _branch(orch: Orch, branch: str, files: dict[str, str]):
    wt = orch.root / ".worktrees" / branch.replace("/", "-")
    lc.git(orch.root, "worktree", "add", "-b", branch, str(wt), orch.base, check=True)
    return wt, orch.commit(wt, files, f"work on {branch}")


def _verify(orch: Orch, item: str = "a") -> None:
    orch.commit(
        orch.mgr,
        {f"runs/{item}/verification.json": json.dumps({"status": "ok"})},
        f"verify {item}",
    )


def _night_head(orch: Orch) -> str:
    return orch.git("rev-parse", f"refs/heads/night/{NIGHT}")


def _refused(orch: Orch, target: str, rc: int, *needles: str) -> str:
    before = _night_head(orch)
    proc = orch.run("merge", target)
    assert proc.returncode == rc, (proc.returncode, proc.stdout, proc.stderr)
    for n in needles:
        assert n in proc.stderr, (n, proc.stderr)
    assert _night_head(orch) == before, "a refused merge moved the night branch"
    assert lc.halted(orch.root) is None, "a review refusal is not a guard trip"
    return proc.stderr


# --- the review gate, on run/ (verification present) ------------------------ #


def test_merge_is_refused_without_a_review_record(orch):
    _branch(orch, "run/a", {"runs/a/ledger.json": "{}\n"})
    _verify(orch)
    _refused(orch, "a", 3, "no review record", "docs/reviews/run-a/")


def test_a_record_for_a_different_head_is_refused(orch):
    wt, _head = _branch(orch, "run/a", {"runs/a/ledger.json": "{}\n"})
    _, other = _branch(orch, "eng/other", {"src/other.py": "y = 2\n"})
    orch.review(wt, "run/a", other)  # reviews a commit not in run/a's history
    _verify(orch)
    _refused(orch, "a", 3, "not an ancestor of", other[:12])


def test_a_record_older_than_head_with_a_code_change_since_is_refused(orch):
    wt, head = _branch(orch, "run/a", {"runs/a/ledger.json": "{}\n"})
    orch.review(wt, "run/a", head)
    orch.commit(wt, {"src/late.py": "z = 3\n"}, "an unreviewed change")
    _verify(orch)
    _refused(orch, "a", 3, "src/late.py", head[:12])


def test_exempt_paths_after_the_reviewed_head_do_not_stale_the_record(orch):
    wt, head = _branch(orch, "run/a", {"runs/a/ledger.json": "{}\n"})
    orch.review(wt, "run/a", head)
    orch.commit(
        wt,
        {
            "runs/a/verification.json": json.dumps({"status": "ok"}),
            ".orchestrator/outbox/a.md": "status: RETURNED\n",
            "docs/reviews/run-a/notes.txt": "follow-up\n",
        },
        "exempt follow-ups",
    )
    proc = orch.run("merge", "a")
    assert proc.returncode == 0, proc.stderr


def test_do_not_merge_is_refused_as_a_failure(orch):
    wt, head = _branch(orch, "run/a", {"runs/a/ledger.json": "{}\n"})
    orch.review(wt, "run/a", head, "DO NOT MERGE")
    _verify(orch)
    before = _night_head(orch)
    proc = orch.run("merge", "a")
    assert proc.returncode == 1, proc.stderr
    assert "DO NOT MERGE" in proc.stderr
    assert _night_head(orch) == before
    assert lc.halted(orch.root) is None


def test_merge_with_fixes_without_fixes_verified_at_is_refused(orch):
    wt, head = _branch(orch, "run/a", {"runs/a/ledger.json": "{}\n"})
    orch.review(wt, "run/a", head, "MERGE WITH FIXES")
    _verify(orch)
    _refused(orch, "a", 3, "MERGE WITH FIXES", "re-review")


def test_merge_with_fixes_is_never_mergeable_on_its_own(orch):
    """Review MAJOR-2 (probe P3), PM decision 2026-09-30: MERGE WITH FIXES is
    self-attested -- the author can add `fixes_verified_at` to the reviewer's record
    after committing a "fix" -- so it never merges. Only a MERGE record at or after
    the fixes does."""
    wt, head = _branch(orch, "run/a", {"runs/a/ledger.json": "{}\n"})
    orch.review(wt, "run/a", head, "MERGE WITH FIXES")  # the reviewer
    fixed = orch.commit(wt, {"src/fix.py": "fixed = True\n"}, "the author's 'fix'")
    # the author edits the reviewer's record to attest the fix
    orch.review(wt, "run/a", head, "MERGE WITH FIXES", fixes_verified_at=fixed)
    _verify(orch)
    _refused(orch, "a", 3, "MERGE WITH FIXES", "re-review")


def test_a_fresh_merge_record_after_the_fixes_merges(orch):
    wt, head = _branch(orch, "run/a", {"runs/a/ledger.json": "{}\n"})
    orch.review(wt, "run/a", head, "MERGE WITH FIXES")
    fixed = orch.commit(wt, {"src/fix.py": "fixed = True\n"}, "the review's fixes")
    orch.review(wt, "run/a", fixed, "MERGE", reviewer="second-reviewer")
    _verify(orch)
    proc = orch.run("merge", "a")
    assert proc.returncode == 0, proc.stderr


def test_merge_with_fixes_with_code_after_the_fix_check_is_refused(orch):
    wt, head = _branch(orch, "run/a", {"runs/a/ledger.json": "{}\n"})
    fixed = orch.commit(wt, {"src/fix.py": "fixed = True\n"}, "the review's fixes")
    orch.review(wt, "run/a", head, "MERGE WITH FIXES", fixes_verified_at=fixed)
    orch.commit(wt, {"src/after.py": "late = 1\n"}, "after the fix check")
    _verify(orch)
    _refused(orch, "a", 3, "MERGE WITH FIXES")


def test_merge_with_fixes_verified_at_not_an_ancestor_is_refused(orch):
    wt, head = _branch(orch, "run/a", {"runs/a/ledger.json": "{}\n"})
    _, other = _branch(orch, "eng/other", {"src/other.py": "y = 2\n"})
    orch.review(wt, "run/a", head, "MERGE WITH FIXES", fixes_verified_at=other)
    _verify(orch)
    _refused(orch, "a", 3, "MERGE WITH FIXES")


@pytest.mark.parametrize("rev", ["run/a", "HEAD", "0" * 40])
def test_fixes_verified_at_must_be_a_full_sha_of_a_real_commit(orch, rev):
    """Review MAJOR-1 (probe P1): `fixes_verified_at: <branch>` resolved to the
    branch tip and waved every later commit through. The field is now informational,
    but a present one is validated exactly as `reviewed_head` is -- a record that
    names a moving ref is malformed, and malformed fails closed."""
    wt, head = _branch(orch, "run/a", {"runs/a/ledger.json": "{}\n"})
    orch.review(wt, "run/a", head, "MERGE", fixes_verified_at=rev)
    _verify(orch)
    _refused(orch, "a", 3, "fixes_verified_at", rev)


def test_only_the_merging_runs_own_verification_is_exempt(orch):
    """Review MINOR-2 (probe P4): a verification.json for ANOTHER run id, committed
    after review, rode in unreviewed and then satisfied gate 1 for that run from
    the night branch."""
    wt, head = _branch(orch, "run/a", {"runs/a/ledger.json": "{}\n"})
    orch.review(wt, "run/a", head)
    orch.commit(
        wt, {"runs/other/verification.json": json.dumps({"status": "ok"})}, "forged"
    )
    _verify(orch)
    _refused(orch, "a", 3, "runs/other/verification.json")


def test_no_verification_json_is_exempt_on_other_prefixes(orch):
    wt, head = _branch(orch, "eng/x", {"notes/x.md": "x\n"})
    orch.review(wt, "eng/x", head)
    orch.commit(wt, {"runs/x/verification.json": json.dumps({"status": "ok"})}, "forged")
    _refused(orch, "eng/x", 3, "runs/x/verification.json")


def test_a_valid_merge_record_merges(orch):
    wt, head = _branch(orch, "run/a", {"runs/a/ledger.json": "{}\n"})
    reviewed = orch.review(wt, "run/a", head)
    _verify(orch)
    proc = orch.run("merge", "a")
    assert proc.returncode == 0, proc.stderr
    parents = orch.git("rev-list", "--parents", "-n", "1", _night_head(orch)).split()
    assert reviewed in parents


def test_a_self_review_is_refused(orch):
    wt, head = _branch(orch, "run/a", {"runs/a/ledger.json": "{}\n"})
    orch.review(wt, "run/a", head, reviewer="same-session", author="same-session")
    _verify(orch)
    _refused(orch, "a", 3, "reviewer")


def test_a_reviewer_who_is_a_git_author_of_the_branch_is_refused(orch):
    wt, head = _branch(orch, "run/a", {"runs/a/ledger.json": "{}\n"})
    orch.review(wt, "run/a", head, reviewer="Test")  # the helper repo's git author
    _verify(orch)
    _refused(orch, "a", 3, "reviewer")


def test_a_record_for_another_branch_is_refused(orch):
    wt, head = _branch(orch, "run/a", {"runs/a/ledger.json": "{}\n"})
    orch.review(wt, "run/a", head, record_branch="run/b")
    _verify(orch)
    _refused(orch, "a", 3, "branch")


def test_a_record_whose_filename_is_not_its_reviewed_head_is_refused(orch):
    wt, head = _branch(orch, "run/a", {"runs/a/ledger.json": "{}\n"})
    orch.review(wt, "run/a", head, filename="review.md")
    _verify(orch)
    _refused(orch, "a", 3, "review.md")


def test_an_unknown_verdict_is_refused(orch):
    wt, head = _branch(orch, "run/a", {"runs/a/ledger.json": "{}\n"})
    orch.review(wt, "run/a", head, "LGTM")
    _verify(orch)
    _refused(orch, "a", 3, "verdict")


# --- run/ still needs verification ------------------------------------------ #


def test_run_prefix_still_needs_verification_even_with_a_review(orch):
    wt, head = _branch(orch, "run/a", {"runs/a/ledger.json": "{}\n"})
    orch.review(wt, "run/a", head)
    _refused(orch, "a", 3, "no verification record")
    _refused(orch, "run/a", 3, "no verification record")


def test_run_prefix_accepts_the_full_branch_name(orch):
    wt, head = _branch(orch, "run/a", {"runs/a/ledger.json": "{}\n"})
    orch.review(wt, "run/a", head)
    _verify(orch)
    proc = orch.run("merge", "run/a")
    assert proc.returncode == 0, proc.stderr


# --- the other prefixes ------------------------------------------------------ #


@pytest.mark.parametrize("prefix", ["eng", "docs", "fix", "feat"])
def test_other_prefixes_merge_with_a_review_and_no_verification(orch, prefix):
    branch = f"{prefix}/x"
    wt, head = _branch(orch, branch, {f"{prefix}-notes/x.md": "x\n"})
    reviewed = orch.review(wt, branch, head)
    main_before = orch.main_refs()
    proc = orch.run("merge", branch)
    assert proc.returncode == 0, proc.stderr
    parents = orch.git("rev-list", "--parents", "-n", "1", _night_head(orch)).split()
    assert len(parents) == 3 and reviewed in parents
    assert orch.main_refs() == main_before


@pytest.mark.parametrize("branch", ["eng/x", "docs/x"])
def test_other_prefixes_are_refused_without_a_review(orch, branch):
    _branch(orch, branch, {"notes/x.md": "x\n"})
    _refused(orch, branch, 3, "no review record")


def test_other_prefixes_must_descend_from_base(orch):
    empty_tree = orch.git("hash-object", "-t", "tree", "/dev/null")
    rootless = orch.git("commit-tree", empty_tree, "-m", "rootless")
    orch.git("branch", "eng/x", rootless)
    _refused(orch, "eng/x", 3, "does not descend from base")


def test_an_unknown_prefix_is_refused(orch):
    _branch(orch, "wip/x", {"notes/x.md": "x\n"})
    _refused(orch, "wip/x", 3, "prefix")


def test_the_frozen_guard_applies_to_other_prefixes(orch):
    wt, head = _branch(orch, "eng/x", {"docs/owner/ruling.md": "edited\n"})
    orch.review(wt, "eng/x", head)
    before = _night_head(orch)
    proc = orch.run("merge", "eng/x")
    assert proc.returncode == 1
    assert "docs/owner/ruling.md" in (lc.halted(orch.root) or "")
    assert _night_head(orch) == before


def test_the_frozen_guard_trips_even_without_a_review(orch):
    """The guard is not weakened by the review gate: a frozen edit still HALTs the
    night whether or not anyone reviewed it."""
    _branch(orch, "run/a", {"preregistration/e3.md": "threshold: 0.50\n"})
    _verify(orch)
    assert orch.run("merge", "a").returncode == 1
    assert "preregistration/e3.md" in (lc.halted(orch.root) or "")


def test_a_dirty_manager_worktree_refuses_other_prefixes(orch):
    wt, head = _branch(orch, "eng/x", {"notes/x.md": "x\n"})
    orch.review(wt, "eng/x", head)
    (orch.mgr / "docs" / "spec" / "spec.md").write_text("dirty\n")
    _refused(orch, "eng/x", 3, "uncommitted changes")
