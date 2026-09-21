# S0-05 — the exit-code protocol, converted from a lesson into a gate

**Brief:** `docs/lab-notes/dispatch-S0-05-exit-code-enum.md` (baseline `a802521`; this branch is cut
from `0da72e6`, which contains it). **run_id:** none — this is code, not an experiment, so there is no
ledger; every count below is the literal output of a command whose exit status was captured with
`cmd > log 2>&1; rc=$?`, never after a pipe.

**Hardware / interpreter:** Mac Studio, CPU only. `uv run` resolves **Python 3.14.6** and pytest
9.1.1 in this worktree's `.venv` (CLAUDE.md says 3.12; recorded, not changed).

## What changed

| Site (anchor at `0da72e6`) | Before | After |
|---|---|---|
| `src/rsr/exit_codes.py` | did not exist | `class Exit(IntEnum)` — `OK 0 · FAIL 1 (alias DROP) · UNKNOWN 2 · DID_NOT_RUN 3 · UNBANKED_RISE 4` — plus `status()` (refuses `bool`, `None`, and ints outside 0–4), `did_not_run()`, `not_implemented()`, `refuse()`, `run_main()`, and an `ArgumentParser` whose usage error is `3` |
| `scripts/canary.py:69` | `"baseline": 3` | `"baseline": Exit.UNKNOWN` (**2**) |
| `scripts/canary.py:151` | `exit_code=0` literal, written before the verdict | row moved below the verdict; `exit_code=int(exit_code_for(verdict))` — the value `run()` returns |
| `scripts/canary.py:34-35` docstring | four codes, `4` absent | five codes; says which three this script emits and why `4` is unreachable here |
| `scripts/mutation_battery.py:915` stale anchor | `raise SystemExit(msg)` → **1** | `refuse(Exit.DID_NOT_RUN, …)` → **3** |
| `scripts/mutation_battery.py:939` red baseline | `raise SystemExit(msg)` → **1** | → **3** |
| `scripts/mutation_battery.py` zero mutations | `0/0 proven`, exit **0** | **2** |
| `experiments/e0{a,d,e,f,g,h,i}/run.py:7` | uncaught `NotImplementedError` → **1** | `return not_implemented("E0X")` → **3** |
| `scripts/extract_golden_tensors.py:96-100` (module-level `import jax`) | uncaught `ImportError` → **1** | **3** (see brief errors: this site was outside the audited `main()`) |
| `scripts/render_scoreboard.py` usage error | argparse **2** | **3**; `2` now means only "no rows" |
| `scripts/render_scoreboard.py`, `experiments/e0c/{run,measure}.py`, `experiments/efeas/run.py`, `experiments/cycle-01-masked-loss/run.py`, `experiments/shuffle-control/run.py`, `scripts/extract_golden_tensors.py` | int literals, `sys.exit(main())` | `Exit` members, `run_main(main)`; codes unchanged except as listed above |

### Why the canary test changed from `3` to `2`, and why that is not moving a goalpost

The brief: *"changing it is not moving a goalpost, it is correcting a value that was never measured
against anything."* The `3` in `test_evidence_machinery.py` was written by the person who wrote the
`3` in `canary.py`, and the only mutation over it (`baseline: 3 → 0`) proved the **old** defect stayed
dead. It could not see the value actually written. `docs/gates.md:96`, read at source:

> **`2` and `3` are separate deliberately, and `3` is where this design can be silently defeated.**
> "The gate did not run" and "the gate found nothing" are different facts.

and `docs/gates.md:91` defines `2` as *"floor UNKNOWN — no recorded floor for this key"*, which is
precisely and only a first reading. A first reading **ran**: `run()` calls `train()`, reads the beats
out of `heartbeat.jsonl` and writes `runs/canary/baseline.json`. The battery now mutates `baseline` to
**both** wrong values, `0` and `3`.

## The re-executed audit

Command, re-executed at `0da72e6` (glob quoted; `git grep` so ignored dirs are excluded):

```
/opt/homebrew/bin/git grep -n 'sys.exit\|EXIT_CODES\|exit_code\|raise SystemExit\|returncode\|NotImplementedError\|return [0-4]$' -- '*.py'
```

exit `0`. Every exit-status site it returned is classified below. `src/rsr/**` `NotImplementedError`
raises inside library stubs (`baselines/`, `metrics/`, `retention/`, `data/`) are not exit sites — no
entry point reaches them except through a `main()` already listed — and `tests/`/`third_party/` hits
are not checkers. `ledger.py`'s `returncode` hits are git subprocess checks inside the provenance code,
not process exits.

### 🔴 Wrong — fixed here

| Site | Was | Now | Why |
|---|---|---|---|
| `scripts/canary.py:69` | `baseline → 3` | `2` | ran; nothing to compare |
| `scripts/canary.py:151` | `exit_code=0` literal | read from the verdict | a typed 0 looks measured |
| `scripts/canary.py:34-35` | 4 codes | 5 | narrowed the protocol it claims to follow |
| `scripts/mutation_battery.py:915` | 1 | 3 | stale anchor: battery did not run |
| `scripts/mutation_battery.py:939` | 1 | 3 | red baseline: battery did not run |
| `scripts/mutation_battery.py:1003` (0 mutations) | 0 | 2 | nothing to compare |
| `experiments/e0{a,d,e,f,g,h,i}/run.py:7` (seven) | 1 | 3 | not implemented is did-not-run |
| `scripts/extract_golden_tensors.py:96-100` | 1 | 3 | JAX absent is environment/did-not-run; this is guaranteed in the project venv |
| `scripts/render_scoreboard.py:407,428` (argparse) | 2 | 3 | `2` had two meanings in one script |

### ⚠️ Ambiguous — decided and recorded

| Site | Decision |
|---|---|
| `scripts/render_scoreboard.py:437` malformed `--claim` → 3 | **Kept 3.** It did not run. With argparse moved to 3, all usage-shaped failures in this script are 3 and `2` means only an empty board. |
| `scripts/mutation_battery.py:1003` without `--check` | **Kept 0 when the table rendered.** The module docstring documents `--check` as the gate and the bare run as the table renderer; changing the renderer's status would change a documented CLI. With `--check`, unproven → 1. |
| `experiments/cycle-01-masked-loss/run.py:271`, `experiments/shuffle-control/run.py:152` — an arm's child exits non-zero or writes no `result.json` | **1, now chosen explicitly** (`refuse(Exit.FAIL, …)`) instead of defaulted by `raise SystemExit(msg)`. The child is committed code on a fixed config; its crash is a defect, not an absent environment. Not in the brief's audit. |
| `scripts/canary.py` — `train()` raising | **Uncaught → 1, kept.** A fixed-seed fixed-config run that crashes is an environment that moved. Documented in the docstring. |
| `scripts/mutation_battery.py:906` — `run_suite()` treats pytest exit **5** (no tests collected) as clean | **Not changed; flagged.** A baseline of "no tests collected" would pass the red-baseline guard. Not reachable on a whole-suite run as far as I can tell (`tests/conftest.py`'s floor), but not verified. |

### ✅ Correct — checked, and named because "checked and correct" is a result

| Site | Code | Verdict |
|---|---|---|
| `experiments/e0c/run.py:206` | `--point` printed → 0 | correct |
| `experiments/e0c/run.py:210` | MPS unavailable → 3 | correct — the template the enum generalises |
| `experiments/e0c/measure.py:232` | MPS unavailable → 3 | correct |
| `experiments/e0c/run.py:264`, `measure.py:317` | finished → 0 | correct |
| `scripts/render_scoreboard.py:432` | `--over` missing → 3 | correct |
| `scripts/render_scoreboard.py:441` | claim check `0 if ok else 1` | correct (now `Exit.OK if ok else Exit.FAIL`) |
| `scripts/render_scoreboard.py:446` | `--audit` file missing → 3 | correct |
| `scripts/render_scoreboard.py:450` / `:455` | all numbers resolve → 0 / unbacked → 1 | correct |
| `scripts/render_scoreboard.py:480` | `2 if not board.rows else 0` | correct — the repo's one right `2` before this change |
| `scripts/extract_golden_tensors.py:274`, `:332`, `:370` | three `ABORT` self-check failures → 1 | correct: each is the check finding the thing it exists to catch (capture ≠ reference loop; positive control not a control; wrong layer count) |
| `scripts/extract_golden_tensors.py:447` | fixtures written → 0 | correct |
| `experiments/efeas/run.py:198` | `exit_code=0` literal in a ledger row | **correct, disclosed**: the row's note says *"SELF-REPORTED … 0 means 'reached write()', not an observed exit status"*, and `main()` returns 0 on that path. Same shape as `canary.py:151`, different verdict because it says what it is |
| `experiments/cycle-01-masked-loss/run.py:276,547,605-614`; `experiments/shuffle-control/run.py:157,378` | ledger `exit_code` from `proc.returncode` | correct — read, not typed |
| `experiments/{cycle-01,shuffle-control,efeas}/run.py` `--single`/final `return 0` | → 0 | correct |
| `tests/conftest.py:85` | suite floor breached → `SystemExit(1)` | correct: a collapsed suite is a real failure |
| `experiments/e0*/run.py:11` | `raise SystemExit(main())` | the pass-through was correct; now `run_main(main)` |

### Not converted — out of this brief's files in scope, each named in `tests/test_exit_codes.py::NOT_CONVERTED`

| Site | Finding |
|---|---|
| `src/rsr/cli.py:31` | 🔴 **Wrong, and the brief marked it correct.** `rsr constants beta` exits **1** with `UnmeasuredConstant: 'beta' is MEASURED and has no logged value` (measured: `uv run rsr constants beta; echo rc=$?` → `rc=1`). "No recorded value for this key" is `docs/gates.md`'s definition of `2`. An unknown constant name (`rsr constants nope` → `rc=1`) is a usage error → 3. Not in scope; not changed. |
| `experiments/s0-02/measure_qtok_collapse.py:131,139,141` | three precondition refusals (fixture drift, not eval-mode, wrong profile shape) via `raise SystemExit(msg)` → **1**; they are did-not-run (3). Not in scope. |
| `experiments/s0-02/measure_capture_cost.py:230` | MPS unavailable → 3 — correct. Not converted. |
| `experiments/s0-02/write_ledger.py:48-101` | eight **typed** `exit_code=0` rows transcribing commands run earlier (one reads `sys.argv[1]`). Same class as `canary.py:151`; historical transcription with notes saying the value was read from `$?`. Not an entry point with a `__main__` guard, so the static tests do not see it. Not in scope. |
| `experiments/s0-02/write_ledger.py:435` | `raise SystemExit(msg)` → 1 on a type mismatch in its own dict — a real defect in the file, 1 is defensible. |
| `src/rsr/train/loop.py:506` | `return 0`. **Not audited beyond that line**: the brief forbids touching it (S0-03 is editing it tonight). `--device mps` with MPS absent was not probed. |

### `4` — UNBANKED RISE

`git grep -n 'UNBANKED\|return 4' -- '*.py'` on this branch returns only `src/rsr/exit_codes.py` and
`tests/test_exit_codes.py`. `git merge-base --is-ancestor origin/macbook-local-2026-09-18 origin/main`
→ exit `1` (not an ancestor); `git show origin/macbook-local-2026-09-18:src/rsr/gates/floors.py`
emits `Exit.UNBANKED_RISE` at `:118` and `:129`. So, as the brief's boundary section already
concluded: **specified on trunk, implemented on a branch trunk cannot reach.** This change defines
`4` in the enum and emits it nowhere; no checker on trunk has a floor to bank. The enum's member names
match `floors.py`'s (`DROP` is an alias of `FAIL`), so porting `src/rsr/gates/` can import it.

## Gates

All at `1b69d24` (the code commit; this file and the regenerated battery record land after it).

| Gate | Literal result | Exit |
|---|---|---|
| `uv run pytest -rsx -p no:cacheprovider` | `passed=393 failed=0 skipped=0 errors=0`, plus **1 xfailed** (pre-existing, strict: `test_train_loop.py::test_the_value_head_lands_in_its_own_mup_group`) | `0` |
| baseline, same command at `0da72e6` | `passed=373 failed=0 skipped=0 errors=0`, 1 xfailed | `0` |
| `uv run python scripts/mutation_battery.py --check --json … --markdown docs/mutation-battery.md` | `62/62 gates proven by mutation` (was 51/51 per Brief 0b) | `0` |
| `uv run ruff check .` / `uv run ruff format --check .` | All checks passed / already formatted | `0` / `0` |

Test delta, by diffing sorted node ids at `0da72e6` vs now: +19 in `tests/test_exit_codes.py`, +1
parametrised case `test_no_hardcoded_constants.py::…[exit_codes.py]` (picks up the new module), and
one rename (`test_a_first_canary_reading_exits_3_not_0` → `…_exits_2_nothing_to_compare`). 373 + 20 = 393.

The 11 new mutations, each reddening only its gate (0 off-gate, 0 declared couplings):

| Mutation | Gate | On gate |
|---|---|---|
| a first canary reading exits 3 again (**the right axis**) | `test_a_first_canary_reading_exits_2_nothing_to_compare` | 1 |
| a canary ledger row typed again | `test_the_canary_ledger_row_records_the_exit_code_it_returns` | 1 |
| unimplemented experiments raise again (one mutation, the class) | `test_an_unimplemented_experiment_exits_3` | 7 |
| a checker returns a bare boolean | `test_no_checker_returns_a_bare_boolean` | 1 |
| status() accepts a bool | `test_status_refuses_a_bool` | 1 |
| a stale battery anchor exits 1 again | `test_a_stale_battery_anchor_exits_3` | 1 |
| a red baseline exits 1 again | `test_a_red_baseline_exits_3` | 1 |
| an empty battery passes | `test_a_battery_with_no_mutations_exits_2` | 1 |
| a usage error exits 2 again | `test_a_usage_error_exits_3_not_2` | 1 |
| extract_golden_tensors exits 1 without JAX again | `test_extract_golden_tensors_without_jax_exits_3` | 1 |
| a checker bypasses run_main | `test_every_converted_checker_exits_through_the_protocol` | 1 |

The existing "a first canary reading exits 0 again" mutation was re-anchored onto the enum and still
reddens the corrected test (1 on gate, 0 off). The Brief 0b SIGKILL row, flagged as timing-dependent
in this brief, was PROVEN on this run (2 on gate).

Process-level probes (each `cmd >/dev/null 2>&1; echo rc=$?`): `experiments/e0{a,d,i}/run.py` → 3;
`extract_golden_tensors.py --out /dev/null` (project venv) → 3; `render_scoreboard.py --bogus` → 3;
`--claim nocolon` → 3; `--over /nonexistent` → 3; `--over <empty dir>` → 2; default → 0.

⚠️ **Not re-run:** `scripts/canary.py` itself (it trains on MPS and writes `runs/canary/`). Its row
ordering and codes are tested with `train` replaced by a fake that writes a heartbeat.
