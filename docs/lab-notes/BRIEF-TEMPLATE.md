---
# Copy this file to docs/lab-notes/dispatch-<id>.md. Everything above the second
# `---` is checked by scripts/orchestrator/lint_brief.py; the body below it is read
# by a researcher. The lint must exit 0 before the brief is committed:
#
#   PYTHONPATH=scripts .venv/bin/python -m orchestrator.lint_brief \
#       docs/lab-notes/dispatch-<id>.md --base "$(git rev-parse HEAD)"; rc=$?
#
# Read rc directly (CLAUDE.md, "Reading an exit code"). 0 clean, 1 findings, 3 did
# not run. This template itself is NOT lint clean: its values are placeholders.

id: <id>                      # the file must be docs/lab-notes/dispatch-<id>.md
item: <queue item id>         # the orchestrator queue item this brief discharges

# `git rev-parse HEAD` at the moment of writing, NEVER recalled. The brief's own
# commit comes after it; lint against this sha, not against the brief's commit.
# (2026-09-21: "the baseline sha was off by the brief's own commit", twice.)
baseline_sha: <40-hex sha>

# One sentence a result could make false. A question whose answer follows from the
# code by reading it is a theorem, not a measurement (PREREG-s003, 2026-09-21):
# write a premise below that would expose that, or do not dispatch.
falsifier: "<the claim a run could falsify>"

# Every line number the body cites, as `path:line`, must appear here. `expect` is a
# literal substring that must be ON that line at baseline_sha. Re-derive with
# `grep -n` at the base; never carry an anchor over from an earlier brief.
anchors:
  - path: src/rsr/example.py
    line: 123
    expect: "def the_function("

# What the brief believes about the tree or the record. Each is executed at base.
#   check: a READ-ONLY shell command, run by bash in a throwaway worktree at base.
#          Rejected unrun if it contains `>`, rm, mv, cp, tee, touch, mkdir,
#          `sed -i`, or a git writer. There is no .venv in that worktree: use git
#          grep / grep / test / python3 (stdlib).
#          expect_rc defaults to 0. `git grep` exits 1 on no match -- that is how
#          "X does not exist yet" is stated.
#   ledger: a committed ledger, read at base; `key` is dotted (list indices ok).
premises:
  - claim: "<what the brief assumes, in words>"
    check: "git grep -n 'the_function' -- src/"
    expect_rc: 0
    expect_stdout_contains: "example.py"
  - claim: "<a recorded result the brief builds on>"
    ledger: runs/<run-id>/ledger.json
    key: verdict.outcome
    expect: falsified

# Must exist at base, unless declared `new: true` (then it must NOT exist).
# Everything the bar requires changing must be in here: "scope contradicts the
# bar" (E-feas-s003, 2026-09-21) is not linted -- check it by reading.
files_in_scope:
  - src/rsr/example.py
  - path: tests/test_example.py
    new: true

bar:
  - "<a measurable condition, with the command that measures it>"
done_when:
  - "<what the report must contain for the manager to accept it>"
do_not:
  - "<the prohibitions that apply to this brief>"
---

# Brief: <title>

**Status:** <written / started / done>. **Written at:** `<baseline_sha>`.
**Worktree / branch:** <where it runs>. **Lane:** <researcher / builder>.

## Why

What prompted this, with sources (spec sections, run ids, PR numbers). Cite a line
as `path:line` only if it is declared in `anchors`.

## Falsifier

The front-matter `falsifier`, restated with the decision rule that reads it.

## Files in scope

The `files_in_scope` list, each with what may change in it.

## Bar

Numbered, each measurable, each with its command. Gates: `uv run pytest -rs
--tb=no` census and `$?`; ruff; the mutation battery with its N/N line; and for
each new gate, the mutation that must redden it.

## Done when

What the report must contain: literal census lines, exit codes, ledger keys, and a
`BRIEF ERRORS` field (write `none` out).

## Do NOT

The prohibitions. Include the `CLAUDE.md` ones this brief could plausibly breach.
