---
branch: eng/t5b-review-gate
reviewed_head: defe2dd7c48989b9a364cc64bc9a54b1924d1ca1
author: rsr-researcher 2026-09-29/30
reviewer: general-purpose adversarial reviewer 2026-09-30 (round 2)
verdict: MERGE
---

# Adversarial re-review (round 2): eng/t5b-review-gate @ defe2dd

The previous review was `bbfa38c9bade.md` (MERGE WITH FIXES). This review covers
c31e4c2..defe2dd, which is four commits: 5f07908, 25ca55c, 59b9f81 and defe2dd. I read the
code before reading the author's round-2 notes.

**Findings:** 0 BLOCKER, 0 MAJOR, 3 MINOR. All three MINORs are hardening items and can
land after the merge. None of them lets code merge past a review in a way that "Known
limits" does not already concede, since anyone who can commit can also forge a record.

## The PM decision is implemented

The PM decision reads: *MERGE WITH FIXES never merges on its own; only a MERGE record at
or after the fixes does.*

- When the governing record says `MERGE WITH FIXES`, the gate returns exit 3 with no
  other path (`merge.py:332-342`). `fixes_verified_at` no longer feeds any pass
  decision. The staleness anchor is always `reviewed_head` (`merge.py:343`).
- A newer `MERGE` record governs under the nearest-record rule, and the test
  `test_a_fresh_merge_record_after_the_fixes_merges` covers that case.
- `docs/review-records.md` step 6 and the field table say the same thing.

## The prior findings are re-probed and closed

I re-ran the prior reviewer's probe file, `rev-infra/test_t5b_probe.py`, unchanged,
against defe2dd. Under that file's convention, a PASS means the attack succeeded. P1,
P3 and P4 now FAIL, which means the attacks are refused.

| prior finding | probe | at bbfa38c | at defe2dd |
|---|---|---|---|
| MAJOR-1 `fixes_verified_at: fix/x` | P1 | rc 0, merged | **rc 3** `fixes_verified_at 'fix/x' is not the full sha of a commit` |
| MAJOR-2 self-attested MERGE WITH FIXES | P3 | rc 0, merged | **rc 3** `review verdict MERGE WITH FIXES is not mergeable on its own` |
| MINOR-2 another run's `verification.json` | P4 | rc 0, merged | **rc 3** `review is stale ... runs/other/verification.json` |
| MINOR-1 verdict edited after review | P2 | rc 0 | rc 0. Still open, and documented as a known limit (see below) |
| frozen edits still HALT | P5, P6 | rc 1 | rc 1 (held) |
| prefixes and bare ids | P7 | rc 3 | rc 3 (held) |

The prior review filed MINOR-1 as MINOR, and fixing it was optional. The author left it
open on purpose and wrote it into "Known limits" in plain terms, together with a
statement that the gate is a process check, not a security boundary. That is
acceptable.

MINOR-3, which parks every tick merge until a review record exists, is unchanged and is
still documented. It remains a PM or owner decision, not a code defect.

## New bypass hunt (scratch probes, `rev-round2/test_t5b_probe_r2.py`)

| probe | what it tries | result |
|---|---|---|
| N1 | a record whose `reviewed_head` descends from head (the branch was reset below it) | rc 3, "not an ancestor". Held |
| N2 | a record for `fix/q` placed in `fix/q2`'s directory | rc 3, malformed (`branch is 'fix/q'`). Held |
| N3 | a correct record for `fix/r` placed under `docs/reviews/fix-other/` | rc 3, no record. Held |
| N4 | after review, merge in a side branch that carries code | rc 3, stale (`src/side.py`). Held: staleness is a tree diff, so a merge counts |
| N5 | after review, an `-s ours` merge of a code branch | rc 0, and `src/side.py` is **not** in the night tree. This changes history, not content. Held |
| N6 | two `verdict:` lines in one record | rc 0. The last line wins silently (MINOR-2 below) |
| N7 | MERGE WITH FIXES at H, then a MERGE whose `reviewed_head` is the docs-only commit that added it, with no fixes made | rc 0. This is by design: a later reviewer's MERGE governs. The gate cannot check that fixes exist ("checks that a review exists, not what it says") |
| N8, N8b | DO NOT MERGE and MERGE records at equal distance on two parallel lines with identical code | the winner is **the lexically smaller file name**. Across 4 runs each verdict won at least once (MINOR-1 below) |
| N9 | after review, a commit adds `docs/reviews/eng-victim/<x>.md` and `docs/reviews/payload.py` | rc 0. Both ride in exempt (MINOR-3 below) |

## MINOR-1: when records conflict at equal distance, a sha prefix decides the outcome, not the verdict

`merge.py:324`: `_, path, fm = min(usable)`. When two usable records have the same
`rev-list --count` to head, `min` breaks the tie on the path, and the path is
`<reviewed_head[:12]>.md`. So a DO NOT MERGE and a MERGE on the same code resolve by
sha order.

**Evidence.** N8 and N8b over 4 runs gave `A<B 1 DO NOT MERGE`, `B<A 0 merged`,
`A<B 0 merged` and `B<A 1 DO NOT MERGE`.

Reaching this needs a merge topology. It cannot happen in a linear history, because
equal distance there means the same `reviewed_head` and therefore the same file.

**Fix.** Among the records at the minimum distance, fail closed with exit 3 ("conflicting
review records") when their verdicts differ. Alternatively, take the most severe
verdict. Add a test for the tie.

## MINOR-2: duplicate front-matter keys are accepted, and the last one wins

`merge.py:238`: `out[k.strip()] = v...` overwrites earlier keys. A record containing
`verdict: DO NOT MERGE` followed by `verdict: MERGE` merges (N6, rc 0). A person
reading the top of the record sees DO NOT MERGE.

**Fix.** In `front_matter`, return None, or report the record as malformed, when a key
repeats. The gate already fails closed on malformed records.

## MINOR-3: the `docs/reviews/**` exemption covers every slug and every file type

`merge.py:214` exempts any path under `docs/reviews/`. After review, a branch can
therefore do two things without staling its record:

- add or edit **another branch's** record;
- add a non-record file, such as `docs/reviews/payload.py`.

Both then ride into the night branch (N9, rc 0).

The other branch's gate reads records only from its own tree (`head:rdir`,
`merge.py:258`). A forged record therefore matters only if that branch later takes in
the night branch. Even then, taking in the night branch changes code, and the staleness
diff catches it. So the risk is low. It is still wider than the exemption needs to be.

**Fix.** Exempt only `docs/reviews/<this branch's slug>/*.md`. Pass the slug into
`review_exempt` the same way `run_item` is passed now. Add one test.

## Other checks

- **Mutation anchors.** The re-anchored mutations match the new source text.
  - `--check-anchors`: 256/256.
  - "MERGE WITH FIXES merges on its own" declares its three off-gate tests with a
    correct reason: each MERGE WITH FIXES test falls through to the MERGE path.
  - The 40-zero parametrisation of `test_fixes_verified_at_...` is listed in
    `_REVIEW_GATE_REFUSALS` with an id that matches pytest's.
- **Document accuracy.**
  - "Five mutations whose names start with `merge:` and cover the review gate" is
    correct. There are 8 `merge:` mutations in all, and 5 of them are for the gate.
  - The staleness paths listed in step 7 match `review_exempt`.

## Tests run (literal)

These ran at defe2dd using the worktree `.venv`, and the output was captured to a file
before the exit code was read.

```
pytest -q tests/test_orch_merge_review.py tests/test_orch_merge.py tests/test_orch_tick.py
rc=0
passed=55 failed=0 skipped=0 errors=0

scripts/mutation_battery.py --check-anchors
anchors_rc=0
256/256 anchors occur exactly once
```

Prior probes (`rev-infra/test_t5b_probe.py`): `3 failed, 4 passed`. The failures are
P1, P3 and P4, which means those attacks were refused, as intended.

New probes (`rev-round2/test_t5b_probe_r2.py`): `10 passed`. Each probe asserts the
behaviour recorded in the table above.

## UNVERIFIED

- I did not execute the three new battery mutations or the two re-anchored ones, because
  B5 holds the CPU. The author's hand proofs are in the T5B report, round 2, and I did not
  reproduce them. The integration battery must prove them before `main` moves.
- I did not run the full suite or the battery.
