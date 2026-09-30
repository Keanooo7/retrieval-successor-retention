---
branch: eng/t5b-review-gate
reviewed_head: bbfa38c9bade832dd9ef48134f1c821db979c4da
author: rsr-researcher 2026-09-29/30
reviewer: general-purpose adversarial reviewer 2026-09-30
verdict: MERGE WITH FIXES
---

# Adversarial review: eng/t5b-review-gate @ bbfa38c

Base: main f7a6b10. I read the code before the author's report
(`~/Documents/RSR-2026-09-29-day/reports/T5B.md`).

**Findings:** 0 BLOCKER, 2 MAJOR, 3 MINOR.

The gate is ordered correctly and the frozen guard is unchanged. The two MAJOR findings
both concern the MERGE WITH FIXES path. With either one, code that nobody reviewed can
merge. The fix for MAJOR-1 is one line.

## What was checked, and held

- **The frozen-diff guard is byte-unchanged.** At f7a6b10 and at bbfa38c I hashed the
  AST source segments of `frozen_definitions`, `frozen_violations`, `_show` and
  `read_verification`, and read `FROZEN_GLOBS` and `CONSTANTS`. All are identical
  (`frozen_violations` 86ab9d077f, `frozen_definitions` 0f74d74917). `trip()` changed
  only in its label and branch text.
- **Gate order is right.** At `merge.py:370-381` the order is verification (`run/`
  only), then the guard, then review. Two probes confirmed that a frozen edit HALTs
  with exit 1 even when a valid MERGE record is present:
  - P5: a `docs/spec` edit.
  - P6: a `preregistration/` edit made *after* the reviewed head.
- **A bare `<id>` stays backward compatible.** At `merge.py:199-200` a bare id still
  resolves to `run/<id>`. P7b: `merge zz` gives `no branch run/zz`, exit 3. An unknown
  prefix is refused with exit 3 (P7a, `exp/x`). The only change is that a bare id
  containing `/` is no longer read as `run/<a>/<b>`. No run id has that shape.
- **Stale records are refused.** A code change after `reviewed_head` is refused with
  exit 3. The staleness diff is a tree diff, `anchor..head` (`merge.py:329`), so
  merging main into the branch also counts as stale. That is conservative, which is
  correct.
- **Malformed records fail closed.** One bad record blocks the merge
  (`merge.py:301-303`).
- **`docs/review-records.md` matches the code,** with one exception, covered in
  MAJOR-1: the document says `fixes_verified_at` is "a sha", and the code does not
  enforce that.

## MAJOR-1: `fixes_verified_at` accepts any revision, so a branch name bypasses staleness permanently

`merge.py:316-328` resolves `fixes_verified_at` with `rev-parse --verify <rev>^{commit}`
(`_commit`, `merge.py:235-237`). Unlike `reviewed_head` (`merge.py:280`), it is never
checked to be 40 hex characters. So `fixes_verified_at: fix/x` resolves to the
branch's current tip, and three things follow:

- it is always an ancestor of head (it *is* head);
- it always descends from `reviewed_head`;
- the anchor becomes head, so the staleness diff is empty.

Every later code commit merges unreviewed. The failure can happen by accident: an
agent writing "the fixes are verified on fix/x" is enough. `HEAD`, `HEAD~0`, tags and
`@{...}` behave the same way.

**Evidence (probe P1, scratch):**

1. Commit `src/a.py` on `fix/x`.
2. Add a record: `MERGE WITH FIXES`, `fixes_verified_at: fix/x`.
3. Commit `x = 'UNREVIEWED'`.
4. Run `merge fix/x`. The result is `P1 rc 0 merged fix/x (125b5cfebc64) into
   night/2026-09-22`.

**Fix:**

- Apply the same `len == 40 and all hex` check to `fixes_verified_at`, and add it to
  the malformed list.
- Add a test that passes a branch name or `HEAD` and expects exit 3.

## MAJOR-2: MERGE WITH FIXES is self-attested, so fixes can merge without anyone verifying them

A reviewer who writes MERGE WITH FIXES at `reviewed_head` cannot yet know the sha of
the fix commit. Someone must therefore add `fixes_verified_at` later. The record lives
under `docs/reviews/**`, which `review_exempt` (`merge.py:210`) treats as exempt, so
anyone can add the field, including the author.

Nothing ties the field to a second look by the reviewer. The document's own
consequence note ("the anchor for MERGE WITH FIXES") makes the fixes the one thing no
record covers.

**Evidence (probe P3):**

1. The reviewer writes MERGE WITH FIXES.
2. The author commits a "fix".
3. The author edits the same record to add `fixes_verified_at: <fix sha>`.
4. The merge exits 0.

**Fix (simplest, and consistent with "the record nearest head governs"):**

- Make MERGE WITH FIXES non-mergeable (exit 3, "fixes not yet re-reviewed").
- Require a **new** record at the post-fix head with verdict MERGE. The
  nearest-record rule already lets that record govern.
- Drop `fixes_verified_at`.

If the field is kept, `docs/review-records.md` must state plainly that it is an
attestation the gate cannot distinguish from an author's own.

## MINOR-1: records are mutable after the fact (a verdict can be flipped)

`docs/reviews/**` is exempt from staleness, so editing an existing record does not
stale it.

**Evidence (probe P2):** a record with `verdict: DO NOT MERGE` was edited to `MERGE`
in a later commit, and the merge exited 0.

This has the same root as the undetectable forgery of a whole new record. All sessions
commit as `Brendan Keane` (the `git_authors` check at `merge.py:261-289` only catches a
reviewer who signs with that identity), and `author` is declared by the record itself.
The documentation covers this ("declared, not authenticated"). I grade it MINOR on
that basis.

**Fix:** refuse a record whose blob at head differs from its blob in the commit that
added it (`git log --diff-filter=A` / `--follow`). A record is then append-only. This
is cheap, and it closes the accidental case.

## MINOR-2: every `runs/<id>/verification.json` is exempt, not only this branch's own

`merge.py:212-213` exempts `runs/*/verification.json` for **any** id, on **any**
prefix. A commit after review can add `runs/other/verification.json` with
`{"status":"ok"}` for a run id that is not at base, so the guard allows it. That file
merges into the night branch unreviewed. `read_verification` (`merge.py:159-170`)
reads the night branch **first**, so the forged record then satisfies gate 1 for
`run/other`.

**Evidence (probe P4):** the merge exited 0, and `night:runs/other/verification.json`
contained `ok`.

**Fix:** exempt `runs/<run_item>/verification.json` only when the target is
`run/<run_item>`. Exempt none for `fix/`, `eng/`, `feat/` or `docs/`.

## MINOR-3: every automatic tick merge now parks until a review record exists (an owner or PM decision, flagged but not made)

`tick.py:197-211` calls `merge_item(root, item)`. Every verified run now parks with
"merge refused: no review record". The document states this ("Consequences worth
knowing"). It is not a defect. It is a behaviour change to the unattended loop that
someone has to decide on before this lands on a night branch the loop uses.

## Tests run (literal)

The branch's own tests, at bbfa38c, using the worktree `.venv`:

```
pytest -q tests/test_orch_merge_review.py tests/test_orch_merge.py "tests/test_orch_tick.py::test_a_whole_night_never_moves_main"
rc=0
passed=39 failed=0 skipped=0 errors=0
```

The adversarial probes are in a scratch file, `test_t5b_probe.py`, not committed. A
pass means the attack succeeded (P1-P4) or the defence held (P5-P7):

```
P1 rc 0 merged fix/x (125b5cfebc64) into night/2026-09-22
P7b 3 no branch run/zz
7 passed in 4.03s
```

## UNVERIFIED

- I did not run the two new battery mutations, because of the CPU constraint. The
  author's hand proof is in the report and I did not reproduce it.
- I ran only one test of `tests/test_orch_tick.py`, not the whole file.
- I did not run the full suite.
