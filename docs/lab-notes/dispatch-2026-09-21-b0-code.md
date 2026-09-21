# Brief 0 (2026-09-21) — three small code fixes the manager must not make itself

**Baseline:** `402d328f497c34498ad18e5e371ab9612692425e` (read from `git rev-parse HEAD` when this was written).
**Branch / worktree:** `s0/w0-code` at `.worktrees/w0-code`, created from `origin/main`. One worktree, one agent.
**Parent:** `docs/lab-notes/dispatch-2026-09-21-overnight.md`, Brief 0. Task C is added by the manager (see *Why*).

## Falsifier

- **A/B:** "The shuffle-control tests catch the two failure modes that #17's battery entries do not model: a replay that perturbs the memory it hands over, and a replay that also swaps the bos-copy path." This is refuted if either new mutation leaves every `test_shuffle_control.py` node green.
- **C:** "A historical-fact test can pin history without pinning the live tree." This is refuted if the rewritten test still reddens when a new `runs/canary/cycle-N/ledger.json` is added.

## Tasks

**A. Two new battery entries** in `scripts/mutation_battery.py`, next to the two existing `test_shuffle_control.py` entries ("the shuffle control hands each row its own memory" and "the shuffle control never applies its permutation"). Both use gate `"test_shuffle_control.py::"` and both mutate `src/rsr/metrics/memory_liveness.py:131`, whose line currently reads `out = model(ids_t, mask_t, kv[perm], valid[perm], bc, bv)`:
- `"the shuffle replay perturbs the memory it replays"`: `kv[perm]` → `kv[perm] + 1e-3`
- `"the shuffle replay hands over the bos gestalt too"`: `bc, bv` → `bc[perm], bv[perm]`

Each needs a `why` string, like its neighbours. The MacBook measured which node reddens under each one by hand: `test_replaying_each_rows_own_memory_reads_exactly_zero` for the first, and `test_disabled_memory_reads_exactly_zero` for the second. Confirm or refute those premises; do not copy them.

**B. The producer's citation.** `experiments/s0-02/write_ledger.py:225` writes `"NotImplementedError from rsr.py:433"`. Re-derive the raise with `grep -n` (it was at `src/rsr/retention/rsr.py:448`, and `def observe` at `:442`, at the baseline) and fix **the producer only**.

**C. The canary tally test.** `tests/test_evidence_machinery.py::test_the_scoreboard_reproduces_the_true_canary_tally` asserts `len(canaries) == 3` over the **live** `runs/` tree. The manager committed a legitimate new canary reading tonight and this test reddened (`passed=384 failed=1`). The docstring's claim is about three specific historical ledgers, `canary/cycle-04`, `cycle-08` and `cycle-12`. Make it assert on those three `run_id`s and their outcomes (`inconclusive`, `survived`, `survived`), so that it still guards the 09-18 tally without forbidding future readings. Add a test proving that an extra canary ledger does not redden it: build a scoreboard over a `tmp_path` copy of the tree with one added. Also prove that the test **does** redden if `cycle-04`'s outcome is changed in that copy.

## Files in scope

`scripts/mutation_battery.py` · `experiments/s0-02/write_ledger.py` · `tests/test_evidence_machinery.py` · `docs/mutation-battery.md` (only if `--markdown` regenerates it) · `.orchestrator/outbox/researcher.md`. Nothing else.

## Bar

- `uv run python scripts/mutation_battery.py --check` exits `0`, and the final line reads **N+2/N+2**. N is the count you **measure** on the baseline before any edit; do not carry one from any brief.
- Each new entry is **PROVEN**. Report which node ids redden under each, with the off-gate count.
- Task C: under the `tmp_path` mutation, the rewritten test reddens; with the extra ledger, it stays green. Quote both, with node ids.
- `uv run pytest -rs --tb=no` → census line and `$?`, both **before** and **after** your edits. `uv run ruff check src/ tests/ scripts/` and `uv run ruff format --check src/ tests/ scripts/` both exit `0`.
- `/opt/homebrew/bin/git diff --stat origin/main -- runs/ preregistration/` is empty.

## Done when

The PR from `s0/w0-code` is open against `main` with every Bar item quoted literally, and your report has been prepended to `.orchestrator/outbox/researcher.md` with a **`BRIEF ERRORS`** field (write `none` out if there are none).

## Do NOT

- Touch anything under `runs/`. `git grep -l 'rsr\.py:433' -- runs/` lists files that were true when written.
- Touch `src/`, except transiently through the battery (which must restore it).
- Merge. The manager verifies and merges.
- Read `$?` after a pipe. In zsh that is `$pipestatus[1]`.
- Run in the main checkout. Use `.worktrees/w0-code` only (H3: the battery mutates source in place).

## Why Task C was added (manager)

The overnight brief did not list it. Without it, every canary reading tonight that is committed under `runs/` reddens the suite, and a canary reading after every training brief is a Done-when item. It is a test fix, so by the brief's own separation rule it is not the manager's to make.
