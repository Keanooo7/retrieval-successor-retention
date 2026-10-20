---
branch: eng/i1-isolation
reviewed_head: eca9f2ee10dda299fc1ca7b927b89fc1673074fc
author: rsr-researcher 2026-09-29/30
reviewer: general-purpose adversarial reviewer 2026-09-30 (round 2)
verdict: MERGE
---

# Adversarial re-review (round 2): eng/i1-isolation @ eca9f2e

**Prior review.** `~/Documents/RSR-2026-09-29-day/reviews/REVIEW-I1.md`, at ff6dccc. Its
verdict was MERGE WITH FIXES, with 3 MAJOR and 8 MINOR findings.

**Scope.** Everything in ff6dccc..eca9f2e. That is two commits:
- e8e61b7: the fixes, the tests and 12 mutations;
- eca9f2e: the coupling declarations.

I read the code first and the author's notes (`reports/P1-I1.md`) after it.

**Findings:** 0 BLOCKER, 0 MAJOR, 5 MINOR. None of the MINORs has to change before
merging.

## The three MAJOR findings are fixed and re-demonstrated

I re-ran the prior reviewer's two demos at eca9f2e. They are in
`scratchpad/rev-round2/sigint_child.py` and `ignored_persist.py`. I made one edit to
`sigint_child.py`: the stub's marker now holds `"<pid> <grandchild pid>"`, so the
script now reads the first field.

| MAJOR | fix | demo at ff6dccc | demo at eca9f2e |
|---|---|---|---|
| 1: a pytest interrupted by SIGINT, or ending with rc 3 or 4, was scored | `suite_problem` (`battery_isolation.py:226`) refuses `returncode < 0` and any rc outside `COMPLETED_EXIT = (0, 1)` (`:79`). `probe_problem` also requires the probe's own `exitstatus` to be 0 or 1. `conftest.py` passes `exitstatus` to `write_probe` | `slow ADDS NOTHING ['<collection/exit 2>']`, battery rc 0 | `DID_NOT_RUN slow ... pytest exited 2 ... the suite did not complete`, **battery rc = 3** |
| 2: the reset kept ignored files, and `verify_clean` could not see them | `CLEAN_ARGV = ("clean", "-ffdxq", "-e", "/.venv/")` (`:82`) is used by both `reset_shard` and the reuse path in `make_shard`. `verify_clean` runs `status --porcelain --ignored` and allows only `!! .venv/` (`:306`) | `LEAKS other`, `LEAKS proven` (PROVEN when run alone) | `LEAKS other` (it poisons its own suite, as designed), **`PROVEN proven`**. The carry-over is gone |
| 3: `--prune-shards` took no lock | `prune_shards` takes `pool_lock` (`:384`), the same flock that `open_pool` takes (`:416`). The teardown inside `open_pool` calls `_prune_unlocked` because it already holds the lock. `main` maps `Unisolated` to exit 3 with "nothing removed" | n/a | `test_prune_refuses_while_a_battery_holds_the_pool` passes. Its mutation "--prune-shards takes no lock" is anchored |

**rc 2 decision.** The prior review suggested keeping rc 2 from collection errors as a
verdict input. The author made **every** rc 2 DID_NOT_RUN instead. That is the more
conservative choice, and it is correct: by default pytest aborts the whole session on a
collection error, so no test ran. I checked whether any existing mutation depends on
the old behaviour. None of the rows in the 251-row night battery
(`RSR-2026-09-27-plan/battery/night-battery.json`) or the 245-row battery
(`battery-final2.json`) has a collection-style or non-`::` failing node. So no current
mutation turns into DID_NOT_RUN because of this change.

## The adopted MINOR findings

| prior MINOR | status |
|---|---|
| 1: the probe checks only `rsr`, and `PYTHONPATH` is inherited | **Partly done.** `_SUITE_STRIP` drops `PYTHONPATH` and `PYTHONPYCACHEPREFIX` (`:73`). The probe still records only `rsr`. See MINOR-A |
| 2: grandchildren outlive pytest | **Done.** `_spawn` gives the suite its own session (`start_new_session=True`), and `_kill_group` sends SIGKILL to the group on every exit (`mutation_battery.py:4677-4710`). The stub test starts a grandchild and asserts it is gone. See MINOR-B for a nit |
| 3: SIGINT on the battery | **Done.** SIGINT is in `_HANDLED_SIGNALS`, and the parametrised kill test has a `[SIGINT]` case |
| 4: `--markdown` records DID_NOT_RUN rows | **Done.** The table is not written, and the run says so on stderr |
| 5: bytecode guards proven only by unit tests | **Partly done.** An end-to-end stub test now exists (`test_bytecode_from_one_mutation_cannot_run_under_the_next`, with the mtime pinned). No mutation disables all three layers together, so no battery row gates that test. See MINOR-C |
| 6: an I1 node declared by more than 3 mutations | **Not adopted, and now worse.** `test_a_killed_pytest_child_is_did_not_run_and_the_tree_stays_clean` has 6 declarers (it had 4), and `test_a_sigint_to_the_pytest_child_alone_is_did_not_run` has 5. "The battery mutates the invoking tree again" declares 13 off-gate nodes. See MINOR-D |
| 7: pool policy | **Done.** It is documented in the `battery_isolation` docstring ("Pool policy"), and `battery_subset.py` passes `--shard-dir` through. See MINOR-E for a nit |
| 8: no mutation for "baseline did not run" | **Done.** "a baseline that did not run exits 1" is gated by a `no-probe` stub mode |

## MINOR findings (none has to change before merging)

**MINOR-A: the probe still asserts only `rsr`.** `tests/_battery_probe.py:51`. Stripping
`PYTHONPATH` removes the inherited route. What remains is the convention that every
`scripts/` and `experiments/` import uses a `sys.path.insert` relative to `__file__`.
Nothing asserts it. **Fix, when I2 lands:** record every `sys.modules[*].__file__` that
lies outside both the shard and `sys.prefix`/`base_prefix`, and refuse the run if any
exist.

**MINOR-B: `_kill_group` signals the group after pytest has been reaped.**
`mutation_battery.py:4687`. `communicate()` has already waited on pytest, so the pid is
free when `killpg` runs.
- While any grandchild is alive, the pgid still exists and cannot be reused, so the
  kill is safe.
- When no grandchild is alive, a new process that took the pid and made itself a group
  leader would receive the SIGKILL.

The window is microseconds and the risk is theoretical. **Fix:** call
`os.killpg(proc.pid, SIGKILL)` before the final wait only when the group still exists.
Alternatively, record the pgid while pytest is alive and skip the kill if
`os.getpgid` fails. The simplest option is to accept the risk and note it in the
docstring.

**MINOR-C: the three-layer bytecode defence has an end-to-end test but no mutation.**
Each layer is disabled alone by its own mutation, and any one of the others masks it.
**Fix:** add one mutation that disables `PYTHONDONTWRITEBYTECODE`, `purge_bytecode` and
`-x` together, gated by the end-to-end test. The author showed this red by hand in a
scratch copy.

**MINOR-D: coupling density on the I1 end-to-end nodes (prior MINOR-6, now worse).**
Each declaration names a causal reason, and the author measured them from the scan. The
cost is lost power: under those mutations, these nodes cannot expose an unrelated
regression. **Fix:** split the end-to-end tests by guard, as the prior review said, or
record in `docs/mutation-battery.md` that the coupling is structural.

**MINOR-E: `battery_subset.py` parses `--shard-dir` only in position 3, and crashes if
it has no value.** `battery_subset.py:19` reads `rest[1]` without a bound check.
- A trailing `--shard-dir` raises `IndexError`, which exits 1 rather than the refusal
  exit 3.
- A `--shard-dir` anywhere else is read as a mutation name, and the script then exits
  with "subset does not resolve". That fails loudly, which is fine.

**Fix:** use `argparse`, or check the length.

**Observation (not graded).**
- The non-shard branch of `run_suite` (`mutation_battery.py:4730`) still scores rc 2,
  3 and 4 through `parse_failures`. `_run_isolated` always passes a shard, so the
  branch is unreachable from the battery and is reached only by stubbed unit tests.
  Consider deleting it, so the MAJOR-1 gap cannot come back through it.
- `.venv/` survives the reset by design. A suite that wrote into the shard venv (a
  `.pth` file, for example) would persist, because `check_interpreter` runs only in
  `make_shard`. No current test does this: a grep for pip, uv sync or ensurepip into the
  root venv found nothing.

## The 20 mutations not yet proven alone

Only 10 of the 30 I1 mutations have been proven alone through the real battery at
eca9f2e. The owner stopped the run at mutation 11. The other 20 were scanned against
the 6 battery test files: each reddens its own gate, and every off-gate red is
declared. None has run against the full suite.

**This does not block the merge into the night branch.**
- The plan's merge rule requires a full battery at RC 0 on the night head before `main`
  moves. These 20 mutations are rows in `MUTATIONS`, so that battery runs them
  automatically.
- A mutation that turns out LEAKS or ADDS NOTHING makes the night battery non-zero. That
  holds `main` and is fixed forward, usually with a declaration. No wrong verdict
  reaches `main`.
- The mutations add evidence rows. They do not change what the battery does at run
  time.
- The scan already rules out the worst outcome, a mutation that does not redden its own
  gate.

What I verified myself:
- All 281 anchors resolve exactly once.
- The two MAJOR demos run through the real battery code on the stub.

**Accept the risk, but record it.** The night-battery report should list these 20 by
name, so that a LEAKS on any of them is read as "an I1 declaration was missing" and
not as an I1 defect.

## Tests run (literal)

At eca9f2e, in `.worktrees/i1-isolation`, with `git status --porcelain` showing 0 lines
before and after:

```
RSR_TEST_COUNT=<scratch> .venv/bin/python -m pytest tests/test_battery_isolation.py -p no:cacheprovider -q -rs
rc=0
passed=49 failed=0 skipped=0 errors=0

.venv/bin/python scripts/mutation_battery.py --check-anchors
anchors_rc=0
281/281 anchors occur exactly once
```

The demos, run through the real battery on the stub:

```
sigint_child.py: battery rc = 3; slow DID_NOT_RUN ... pytest exited 2 ...; proven PROVEN
ignored_persist.py: other LEAKS (own poison, expected); proven PROVEN off_gate= []
```

## UNVERIFIED

- The full suite at eca9f2e. I rely on the author's result:
  `passed=1646 failed=0 skipped=0 errors=0`.
- The 20 I1 mutations have not been proven alone against the full suite. The integration
  battery must prove them.
- The full battery at eca9f2e.
- The rc 3 and rc 4 paths through a real pytest. They are covered only by the
  `suite_problem` unit parametrisation.
- Behaviour on Linux or CUDA hosts (`flock`, `killpg`, `start_new_session`).
