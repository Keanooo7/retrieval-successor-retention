# Review records (PLAN-v4 §4 T5(b))

`orchestrator.merge` refuses to merge a branch into `night/<date>` unless the branch
carries a **committed adversarial-review record**. This file is the format and the
rule. The code is `review_gate()` in `scripts/orchestrator/merge.py`; the tests are
`tests/test_orch_merge_review.py`. The battery has five mutations whose names start
with "merge:" and cover the review gate.

These are tech decisions made under the PM's delegation on 2026-09-30 (T5B). They were
revised on the same day after the adversarial review
`docs/reviews/eng-t5b-review-gate/bbfa38c9bade.md`. The MERGE WITH FIXES rule is a
PM decision. None of this is an owner ruling.

> **This gate is a process check, not a security boundary.** It makes sure that a
> named, separate review is committed and that it covers the code being merged. It
> cannot authenticate who wrote a record, and it cannot stop a record from being
> edited. See "Known limits" below.

## Where the record lives

```
docs/reviews/<branch-slug>/<reviewed_head[:12]>.md
```

committed **on the branch being merged**. `<branch-slug>` is the branch name with every
`/` replaced by `-` (`eng/t5b-review-gate` -> `eng-t5b-review-gate`). The file name
must be the first 12 hex characters of `reviewed_head`. A record on `main`, on the
night branch, or only in `~/Documents` does not count.

## Format

```markdown
---
branch: eng/t5b-review-gate
reviewed_head: 0123456789abcdef0123456789abcdef01234567
author: <the session or agent that wrote the branch>
reviewer: <the session or agent that wrote this review>
verdict: MERGE
---

The review itself: findings, severities, and what was checked.
```

| field | required | rule |
|---|---|---|
| `branch` | yes | the full branch name being merged, exactly |
| `reviewed_head` | yes | a full 40-hex sha, reachable from the head being merged |
| `author` | yes | who wrote the branch. It is not in the brief's list; it was added so that "reviewer is not the author" is something the gate can check |
| `reviewer` | yes | who wrote the review. It must differ from `author` (case-insensitive), and it must not be the name or email of any git author of `base..head` |
| `verdict` | yes | exactly one of `MERGE`, `MERGE WITH FIXES`, `DO NOT MERGE`. **Only `MERGE` can merge** |
| `fixes_verified_at` | no | **informational only**. The gate never uses it to pass a merge. If present, it must be the full 40-hex sha of a commit that exists. A ref name such as `fix/x` or `HEAD` makes the record malformed |

The front matter is plain `key: value` lines between two `---` lines. It is not YAML;
quotes around values are stripped.

## The gate

To merge `branch@head`:

1. **Records are read** from `head:docs/reviews/<branch-slug>/*.md`. If there are
   none, the exit is **3** with the message "no review record", naming the expected
   path.
2. **Fail closed on malformed records.** If any record has no front matter, a missing
   field, a `branch` other than this branch, a `reviewed_head` that is not a full sha,
   a file name that does not match `reviewed_head[:12]`, a `fixes_verified_at` that is
   not the full sha of a commit, an unknown verdict, or a self-review, the exit is
   **3** and each problem is named. A malformed record is
   never skipped in favour of a good one.
3. Records whose `reviewed_head` is **not an ancestor of head**, for example a
   rebased-away commit, are ignored. If no record is left, the exit is **3**.
4. **The record nearest head governs**, meaning the one with the fewest commits in
   `reviewed_head..head`.
5. A verdict of `DO NOT MERGE` exits **1**.
6. A verdict of `MERGE WITH FIXES` exits **3**: **it is never mergeable on its own**
   (PM decision, 2026-09-30). The fixes are code that no record covers, and
   `fixes_verified_at` can be written by anyone, so the gate cannot tell it apart
   from the author's own claim. To merge, a reviewer commits a **new `MERGE` record
   whose `reviewed_head` is at or after the fixes**. Under step 4 that newer record
   governs.
7. **Staleness.** `git diff --name-only reviewed_head..head` may change only exempt
   paths:
   - `docs/reviews/**`
   - `.orchestrator/outbox/**`
   - when the target is `run/<item>`, **that run's own**
     `runs/<item>/verification.json`, and nothing else under `runs/`. On any other
     prefix, no `verification.json` is exempt.

   Any other path exits **3**, and the message names the paths and asks for a
   re-review at head.

Because the record is committed after the commit it reviews, head is normally
`reviewed_head` plus one commit that touches only `docs/reviews/`, and step 7 passes.

## Where the gate sits in `orchestrator.merge`

The gates run in this order:

1. The night is open.
2. The branch exists and has an accepted prefix.
3. The branch descends from `base_sha`.
4. **Verification**, for `run/` only.
5. **The frozen-diff guard**, unchanged, on every prefix.
6. **The review gate**, on every prefix.
7. `DRY_RUN`, then the manager worktree must be clean and on the night branch.
8. `merge --no-ff`.

The review gate runs **after** the guard. A frozen-path edit therefore still HALTs the
night whether or not anyone reviewed it, and the guard is not weakened.

Accepted targets:

- a bare `<item_id>`, meaning `run/<item_id>`. This is the original CLI, and `tick.py`
  still calls `merge_item(root, item)` this way;
- a full branch name with one of the prefixes `run/`, `fix/`, `eng/`, `feat/`, `docs/`.

Any other prefix exits 3.

## Consequences worth knowing

- **`run/` branches need a review too.** Tick's automatic merge of a verified run now
  parks the item with "merge refused: no review record ..." until a review record is
  committed on `run/<id>`. That is the T5(b) requirement applied as written. If
  unattended runs should instead merge on verification alone, that is a PM or owner
  decision; this change does not make it.
- **MERGE WITH FIXES means two reviews.** The first review names the fixes. The
  second review, a `MERGE` at the post-fix head, is what the gate accepts. The PM
  asked for the second review to come from a different reviewer. The gate does not
  enforce that.

## Known limits (process check, not a security boundary)

- **Identity is declared, not authenticated.** Every agent session commits under the
  same git identity. `author` and `reviewer` are strings that the record states about
  itself. The gate refuses a record where they are equal, or where the reviewer
  matches a git author of `base..head`. It cannot prove that a different session, or
  a different mind, wrote the review. Anyone who can commit to the branch can commit
  a forged record.
- **Records can be edited after the fact (review MINOR-1).** `docs/reviews/**` is
  exempt from staleness, so a later commit can edit an existing record, for example
  flipping `DO NOT MERGE` to `MERGE`, and the gate will accept it. Making records
  append-only would close the accidental case, but it was left as a documented limit
  by decision. Whoever merges should read the record's history
  (`git log -p -- docs/reviews/<slug>/`) whenever it has more than one commit.
- **The gate checks that a review exists, not what it says.** A `MERGE` record with
  an empty body passes.
