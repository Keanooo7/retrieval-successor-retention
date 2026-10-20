---
branch: eng/i6-env-record
reviewed_head: ac20d28bf6faf26c6fab7664a7223cd540db0d58
author: rsr-researcher 2026-09-30 (I6 build + fix rounds 1-4)
reviewer: general-purpose adversarial reviewer 2026-09-30 (I6, fifth review)
verdict: MERGE
---

# Fifth review: `eng/i6-env-record` @ `ac20d28` (PLAN-v4 §4 I6)

- **Diff read in full:** `1c32b57..ac20d28` (one commit; `scripts/battery_env.py`, `scripts/mutation_battery.py` (table only), `tests/test_battery_env.py`).
- **Earlier rounds:** read the fourth review (`e1434501e4ba.md`) closely, skimmed the other three. Touched only where `ac20d28` interacts with them.
- **Requirement:** *"Record **all** `RSR_*` variables, and refuse to run on undeclared overrides. Proof: a stray override is refused."*
- **Manager decision, not re-litigated:** untracked files scanned only in `src scripts experiments tests ops`; `.claude/` tracked-only; `.gitignore` not edited; `os.path.samefile` for the top-of-checkout check.

## Summary

The fourth review's required MINOR-1 is fixed, and its optional MINOR-2 is taken.

- **`.claude/` is now machine-independent.** Only tracked files are read there. In a clone at this head, with an untracked `.claude/settings.local.json` holding an `RSR_*` typo and *no* excludes of any kind (`HOME`/`XDG_CONFIG_HOME` empty, `GIT_CONFIG_GLOBAL=/dev/null`, `GIT_CONFIG_NOSYSTEM=1`, `check-ignore` rc 1), the real inventory test is green: rc 0, `passed=1`.
- **Mutations:** all five new or re-anchored mutations anchor exactly once, and each gate names exactly one `def`. I proved two by hand; one of them also went red against all 27 battery-referencing files, with no off-gate red.
- **`samefile`:** correct for every path variant I tried.
- **Ruff:** clean.

**Nothing must change before merging.** Four optional MINORs follow. Every one fails loud (a false red) or can only fail to catch a future regression; none can turn a stray override green in the tree, and none touches the refusal. `check()` reads the environment, not the inventory.

## Findings

### MINOR-1 (optional): the new test's "no global excludes" setup can be bypassed through git's environment-level config, which turns the test red for a machine-local reason

**Where:** `tests/test_battery_env.py:408-411`. The test neutralises `HOME`, `XDG_CONFIG_HOME` and system config, but not `GIT_CONFIG_GLOBAL` or `GIT_CONFIG_COUNT`/`GIT_CONFIG_KEY_n`/`GIT_CONFIG_VALUE_n`.

**Measured (What I ran, 7):** I pointed `core.excludesFile` at a file holding `**/.claude/settings.local.json` through the environment and ran the new test.
- Via `GIT_CONFIG_GLOBAL`: rc 1, `passed=0 failed=1`.
- Via `GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.excludesFile GIT_CONFIG_VALUE_0=<file>`: rc 1, `passed=0 failed=1`.
- Both fail the test's non-vacuity guard, with the same line:

  ```
  E   AssertionError: CompletedProcess(args=['/opt/homebrew/bin/git', '-C', '…/repo', 'check-ignore', '-q', '.claude/settings.local.json'], returncode=0, stdout=b'', stderr=b'')
  ```
- **Control:** without those variables: rc 0, `passed=1`.

**Consequence:** the test added to remove machine dependence is itself red on a machine that exports either variable.
- The red is loud and explains itself: the probe shows `returncode=0`.
- The inventory's verdict is not affected.
- No machine known to this project sets these variables. My shell's only `GIT_*` variable is `GIT_EDITOR`, and nothing in `scripts src tests ops .github` sets them.
- If the Studio ever exported one, a battery's baseline would go red: the battery would refuse to run, not score wrongly.

**Fix:** add `monkeypatch.delenv("GIT_CONFIG_GLOBAL", raising=False)` and `monkeypatch.delenv("GIT_CONFIG_COUNT", raising=False)`. Or set `GIT_CONFIG_GLOBAL` to `os.devnull`, which also covers `~/.gitconfig` whatever `HOME` is.

### MINOR-2 (optional): `CODE_DIRS` is a new constant, and 4 of its 5 entries are untested

**Where:** `scripts/battery_env.py:164` (`CODE_DIRS = ("src", "scripts", "experiments", "tests", "ops")`). It hand-duplicates `SCANNED` minus `.claude` (`:147`).

**Measured (What I ran, 5):** delete-a-line probes against `tests/test_battery_env.py`:

| Probe | Result |
|---|---|
| `CODE_DIRS = ("scripts",)` | rc 0, `passed=26` |
| drop `"ops"` | rc 0, `passed=26` |
| drop `"tests"` | rc 0, `passed=26` |
| drop `"scripts"` | rc 1, `FAILED …::test_inventory_reads_a_new_untracked_file` |

**Consequence:**
- If a later edit drops an entry, or adds a directory to `SCANNED` but not to `CODE_DIRS`, the third review's MINOR-3 property is silently lost for that directory: a new, not-yet-added file there is no longer inventoried before it is committed.
- Once the file is committed, `--cached` reads it, so the author's run is green and CI red. That is the round-3 failure mode, but only for pre-commit runs.
- The tracked-file inventory and the refusal are unaffected.

**Fix:** derive the constant, `CODE_DIRS = tuple(d for d in SCANNED if d != ".claude")`, or assert `set(SCANNED) - set(CODE_DIRS) == {".claude"}` in a test. Either way, plant the new-file case in more than `scripts/`.

### MINOR-3 (optional): the verbatim `.gitignore` couples six fixture tests to unrelated `.gitignore` edits (false red only, never vacuous)

**Where:**
- `tests/test_battery_env.py:343`: `".gitignore": (_REPO / ".gitignore").read_text()`;
- `:349`: `git add -A`, so which fixture files become *tracked* now depends on the repository's ignore rules.

**Measured (What I ran, 5):**
- Fixture `.gitignore` plus `*.env`: rc 1, `passed=22 failed=4`. The red tests are `test_inventory_finds_a_name_by_file_and_line`, `…skips_ignored…`, `…skips_untracked_claude_files…` and `…reads_a_new_untracked_file`: `ops/x.env` is no longer committed.
  - A `*.env` rule is a plausible hygiene edit. The repository tracks `ops/orchestrator.env` and would keep tracking it, so this red would be about the fixture, not the tree.
- Fixture `.gitignore` = `""`: rc 1, only `…skips_ignored…` red. That is correct: the test does rest on the real rules.

**Vacuity direction:** I found none.
- A rule removed from `.gitignore` makes `…skips_ignored…` red: the planted `test-count.json` and `__pycache__` files are then read.
- A rule added that ignores `.claude/settings.local.json` trips the probe guard in the MINOR-1 test.
- No mutation in the table touches `.gitignore`, which I checked in-process.

**Fix:** `git add -f -- <the fixture's own files>` in `_repo`. The files the fixture calls "tracked" are then tracked whatever `.gitignore` says, and only the *planted* files test the real ignore rules. That is the half the coupling was introduced for.

### MINOR-4 (optional): the case test's skip condition checks the repository's filesystem, but the test runs on `tmp_path`'s

**Where:** `tests/test_battery_env.py:424-433`. The condition is `os.path.exists(str(Path(__file__).resolve()).swapcase())`; the body uses `_repo(tmp_path / "repo")`.

**Consequence:**
- **Repository case-insensitive, `TMPDIR` on a case-sensitive volume:** the test runs and fails `assert … other.exists()`. That is a loud false red.
- **The reverse:** it skips when it could have run.
- **On CI** (ubuntu, `/home/runner/…`): it skips, and `-rs` plus `conftest.py` print it as SKIPPED, never counted as passed. So there it is correct.
- **On the Studio,** both paths are on the same APFS volume, and it ran (not skipped) in all my runs.

**Fix:** decide inside the test on `tmp_path`: `if not Path(str(tmp_path).swapcase()).exists(): pytest.skip(...)`. Keep the same "(NOT a pass)" reason.

## Attack results (the brief's five points)

### 1. Machine-local git config

**`.claude/`: independent.** Tracked-only means no ignore source is consulted there.
- **Verified, real inventory test** in a clone, no excludes at all, untracked `.claude/settings.local.json` = `{"note": "RSR_BATTERY_THREAD"}`: rc 0, `passed=1` (What I ran, 8).
- **Still machine-local, all loud or negligible and outside this commit's surface:**
  - uncommitted edits to *tracked* `.claude/settings.json` are read, because the working tree is read. That is loud, and it is the author's own change.
  - a sparse checkout would hide tracked files (`is_file()` false). I did not test this, and nothing here uses sparse checkout.

**Code directories: yes, a per-machine exclude can hide an `RSR_*` typo in an untracked file.**
- **Measured (What I ran, 9):** in a clone I planted `scripts/new_tool.py` reading `RSR_BATTERY_THREAD`.
  - Default env: rc 1, `{'RSR_BATTERY_THREAD': ['scripts/new_tool.py:1']}`.
  - With `core.excludesFile` covering it, set via `GIT_CONFIG_COUNT`: rc 0, `passed=1`. **Silent.**
  - `git add scripts/new_tool.py` under the same excludes: rc 1, `The following paths are ignored by one of your .gitignore files:`.
- **Consequence:** no finding.
  - The silent green covers only a file that git on that machine refuses to add. It cannot reach the tree without `add -f`, and once tracked, `--cached` reads it on every machine.
  - The runtime refusal (`check()`, `battery_env.py:194`) reads the environment, so a stray set by such a local file is still refused.
  - The other direction, a code-dir file ignored on one machine and not on another, is a loud false red.
  - `SCANNED`'s docstring exempts "the author's own not-yet-added files in `CODE_DIRS`", which honestly covers this.

### 2. The fixture's verbatim `.gitignore`

See MINOR-3: false red on unrelated edits such as `*.env`; I found no vacuous direction.

### 3. `os.path.samefile`

Measured with `tracked()` called directly (What I ran, 6):

| Root | Result |
|---|---|
| the clone, as `Path` or `str` | OK, n=264 |
| trailing slash | OK |
| relative | OK |
| symlink to the clone | OK |
| `swapcase` of the whole path | OK |
| `/PRIVATE/` spelling | OK |
| nonexistent dir | `RuntimeError: git rev-parse --show-toplevel in … -> rc=…`, from `_git_out` before `samefile` runs |
| a file | `RuntimeError` from `_git_out` |
| the `.git` dir | `RuntimeError` from `_git_out` |
| a subdirectory | `RuntimeError: … is not the top of a checkout` |

- `inventory(swapcase root) == inventory(root)`: `True`.
- **The trailing newline:** handled by `.strip()`. Removing it turns 7 tests red (probe p3), so it is covered.
- **`top` nonexistent:** only reachable through a broken git config. `samefile` would then raise `FileNotFoundError`, which is loud; I did not construct that case.
- **Case-sensitive filesystem:** the test skips with its "(NOT a pass)" reason. CI prints skips through `-rs` and `conftest.py`, and CI's census step fails only on `failed`/`errors`/`passed<100`, so a skip is never counted as a pass. The mutation "the checkout-top check compares path strings" would then have a skipped gate, which the battery would report as unproven (loud); the battery runs on the Studio. MINOR-4 covers the filesystem-mismatch corner.

### 4. Mutations

**Anchors, in-process (What I ran, 1):** `ROOT` is this worktree; `n 368 problems []`; no duplicate names; all 15 `battery-env:` entries have old-count 1.

| Mutation | old count | gate → defs matched (all `tests/*.py`) |
|---|---|---|
| reads ignored files | 1 | `test_inventory_skips_ignored` → 1 |
| misses a new untracked file | 1 | `test_inventory_reads_a_new_untracked` → 1 |
| accepts a directory inside a checkout | 1 | `test_inventory_refuses_a_directory_inside` → 1 |
| reads untracked .claude files | 1 | `test_inventory_skips_untracked_claude_files` → 1 |
| checkout-top check compares path strings | 1 | `test_tracked_accepts_a_differently_cased_root` → 1 |

- Three mutations share the anchor `"--others", "--exclude-standard", "--", *CODE_DIRS`, and two share `    if not os.path.samefile(top, root):`. Each anchor occurs once, and each `new` produces valid Python.

**By-hand proofs (mine, in the worktree; apply with count asserted 1, restore with `git checkout --`):**

| Mutation | Tests | Result |
|---|---|---|
| checkout-top check compares path strings | `tests/test_battery_env.py` | rc 1, `passed=25 failed=1`, `FAILED tests/test_battery_env.py::test_tracked_accepts_a_differently_cased_root`; `E   RuntimeError: git ls-files: /PRIVATE/VAR/FOLDERS/…/REPO is not the top of a checkout (its checkout's top is /private/var/folders/…` |
| same | 27 battery-referencing `tests/test_*.py` | rc 1, `passed=1179 failed=1`; only that gate red |
| reads ignored files | `tests/test_battery_env.py` | rc 1, `passed=25 failed=1`, `FAILED tests/test_battery_env.py::test_inventory_skips_ignored_files_in_a_checkout_and_a_worktree` |

- Every proof: apply rc 0, restore rc 0, porcelain `[]` before and after.

### 5. Lines in `ac20d28` that no test covers

Delete-a-line probes, all in a scratch clone at `ac20d28`, against `tests/test_battery_env.py`; each restored, porcelain `[]`:

| Probe | Result | Meaning |
|---|---|---|
| p1 `--cached -- *SCANNED` → `*CODE_DIRS` | rc 1, 4 red | covered |
| p2 drop `-z` from the `--others` call | rc 1, `…reads_a_new_untracked_file` | covered |
| p3 drop `.strip()` | rc 1, 7 red | covered |
| p4 drop `"--cached"` | rc 0, 26 passed | not a finding: `ls-files` defaults to `--cached` |
| p5/p12/p13 drop `ops` / `tests` / all but `scripts` from `CODE_DIRS` | rc 0, 26 passed | **MINOR-2** |
| p6 drop `scripts` from `CODE_DIRS` | rc 1, `…reads_a_new_untracked_file` | covered |
| p7 fixture `.gitignore` = `""` | rc 1, `…skips_ignored…` | covered |
| p8 fixture `.gitignore` + `*.env` | rc 1, 4 red | **MINOR-3** |
| p9 drop `setenv("HOME")` | rc 0 | not a finding: this machine's excludes live under XDG. The line is for machines with `core.excludesFile` in `~/.gitconfig` |
| p10 drop the `check-ignore` guard | rc 0 | not a finding: it is a non-vacuity guard, and the mutation still reddens the inventory assertion |
| p11 drop `XDG_CONFIG_HOME` + `NOSYSTEM` | rc 0 | not a finding: `HOME` alone covers this machine |

- **Scope of `mutation_battery.py` changes:** only the table, as the author said (diff read).
- **`_git_out` rc check:** unchanged this round.
- **The `.claude/worktrees/` agent-copy plant** in `…skips_ignored…` no longer exercises `--exclude-standard`, since untracked `.claude/` is never listed. It is harmless and still true, but that half of the test now proves the `CODE_DIRS` restriction, not the ignore rule.

## What I ran (literal)

**Setup**
- **Worktree:** `.worktrees/i6-env-record` at `ac20d28bf6faf26c6fab7664a7223cd540db0d58`; porcelain empty at start.
- **Interpreter:** `.venv/bin/python -c "import rsr; print(rsr.__file__)"` → `/Users/keanooo7/retrieval-successor-retention/.worktrees/i6-env-record/src/rsr/__init__.py`.
- **Remote:** `origin/eng/i6-env-record` = `41db09da327d12a1ea117812272b83f502baedb1` (stale; untouched).
- **Shell:** `env | grep -c '^RSR_'` → `0`. Only `GIT_*` variable: `GIT_EDITOR`.
- **Git and exit codes:** `/opt/homebrew/bin/git` throughout; rc captured as `cmd > log 2>&1; rc=$?`.
- **`$S`:** `…/scratchpad/review5-i6/`.
- **Scratch clone:** `$S/clone` = `git clone -q --no-checkout <worktree>` + `checkout --detach ac20d28…`, run with `env -u VIRTUAL_ENV PYTHONPATH=$S/clone/src PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 <worktree>/.venv/bin/python -m pytest -q -p no:cacheprovider -rfEs --tb=line`. Its baseline was rc 0, `passed=26 failed=0 skipped=0 errors=0`.
- **Never done:**
  - nothing in `.worktrees/_mgr`;
  - no battery, subset, union or shards script run as a script;
  - nothing pushed, no GitHub;
  - no full suite, timing, or parallel run;
  - no classifier denial.

**Commands**

1. `$S/anchors.py` (in-process `mutation_battery.anchor_problems()`, per-entry old/new counts, gate-to-`def` match over `tests/*.py`) → rc 0. Output as in attack 4.
2. Worktree baseline: `pytest tests/test_battery_env.py` → rc 0, `passed=26 failed=0 skipped=0 errors=0`, no SKIPPED line.
3. `$S/byhand.zsh` ×3 → the attack 4 table.
   - The 27-file list is `grep -l 'mutation_battery\|battery_env' tests/test_*.py`. The brief's `tests/*.py` form also picks up `tests/_battery_fakes.py`, which does not collect standalone: `ERROR tests/_battery_fakes.py … ModuleNotFoundError: No module named 'battery_isolation'`, rc 2. I discarded that run as a list error, not a result.
4. `ruff check` + `ruff format --check` on the 3 files → rc 0/0, `All checks passed!`, `3 files already formatted`.
5. `$S/probe.zsh` × 13 probes (p1-p13) in the clone → the attack 5 table.
6. `$S/paths.py` (`tracked()` on path variants, in the clone) → rc 0. Output as in attack 3.
7. MINOR-1, in the clone, `…::test_inventory_skips_untracked_claude_files_without_global_excludes`:
   - with `GIT_CONFIG_GLOBAL=$S/gcg/config` (`[core] excludesFile = $S/gcg/ignore`; `ignore` = `**/.claude/settings.local.json`) → rc 1, `passed=0 failed=1`;
   - with `GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.excludesFile GIT_CONFIG_VALUE_0=$S/gcg/ignore` → rc 1, `passed=0 failed=1`;
   - the same command in the worktree → rc 1, `passed=0 failed=1`; control without the variables → rc 0, `passed=1`.
8. Clone plus untracked `.claude/settings.local.json` = `{"note": "RSR_BATTERY_THREAD"}`:
   - `HOME=XDG_CONFIG_HOME=$S/gcg GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1` → `…::test_inventory_every_rsr_name_the_tree_reads_is_declared` rc 0, `passed=1`;
   - `git check-ignore -v` in the same env → rc 1;
   - file moved out afterwards; clone porcelain empty.
9. Clone plus untracked `scripts/new_tool.py` = `import os; os.environ.get("RSR_BATTERY_THREAD")`:
   - default env → rc 1, `AssertionError: RSR_* names read in the tree but not declared in scripts/battery_env.py: {'RSR_BATTERY_THREAD': ['scripts/new_tool.py:1']}`;
   - with `GIT_CONFIG_COUNT` excludes covering it → rc 0, `passed=1`;
   - `git add scripts/new_tool.py` → rc 1, `The following paths are ignored by one of your .gitignore files:`;
   - file moved out afterwards; porcelain empty.
10. In-process: no mutation has a non-`.py` path, and none touches `.gitignore` (`Counter()`, `[]`). `git ls-files '*.env'` → `ops/orchestrator.env`. `git grep` callers of `inventory`/`tracked`: `tests/test_battery_env.py` only.
11. Read-only: `.gitignore`, the common dir's `info/exclude` (the `claude-code-runtime` block, all `**/.claude/…`), `~/.config/git/ignore` (`**/.claude/settings.local.json`), `.github/workflows/ci.yml` (census step), `tests/conftest.py`, the Fix round 4 section of `I6-build.md` and `I6-fix4.minor1-rerun.zsh`.
12. The worktree's `git status --porcelain` was empty before this record was added.

## Not verified

- **No case-sensitive filesystem run.** The skip on ubuntu/CI and the unproven-gate behaviour there are argued from the code (`swapcase` of `/home/runner/…` does not exist; `ci.yml` census logic), not observed. I did not mount a case-sensitive volume.
- **No real battery, full suite, or whole-table simulation.**
  - The author's whole-table simulation (`I6-fix4.61`) is not re-run.
  - Of the five changed or new mutations, I proved two by hand, and only one of them against all 27 files.
  - The other three rest on the author's logs (`I6-fix4.mut-*.log`) and the manager's re-run.
- **`battery_union` not re-checked** this round. The table change is inside the I6 block only (diff read), but I did not re-run the 6-order union with B5/I2.
- **Not constructed:** a `rev-parse --show-toplevel` that prints a nonexistent path, and a sparse checkout.
- **Other machines' excludes:** whether any real machine's excludes ignore files in the code directories, or whether any machine exports `GIT_CONFIG_GLOBAL`/`GIT_CONFIG_COUNT`, is unknown. MINOR-1 and attack 1 are argued from constructed configs.
