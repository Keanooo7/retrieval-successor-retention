# S0-05 — the exit-code conflation, converted from a lesson into a gate

**Baseline:** `6a775b74` — read from `git rev-parse HEAD` at the moment of writing, 2026-09-20.
**Lane:** researcher · CPU, no device required.
**Sprint:** 0 — make the substrate honest.
**Opened by:** `docs/lab-notes/dispatch-2026-09-20b-cycle-1.md` Task B2.

---

## The finding is not the mapping. It is the shape.

`scripts/canary.py` had a defect: a first reading trained, produced beats, wrote
`baseline.json`, returned verdict `"baseline"`, and **exited 0**. That is `3` collapsing to `0` —
*did not run* reported as *pass*. It was found and fixed in cycle 0.

The fix is at `scripts/canary.py:69`:

```python
EXIT_CODES = {"held": 0, "MOVED": 1, "baseline": 3}
```

🔴 **It is wrong, and it is wrong in the same class, inverted, by the person fixing it.**

The script's own protocol, stated twice — in the module docstring at `:35` and again as the
comment directly above the mapping at `:67` — is:

> `0` pass · `1` real failure · **`2` nothing to compare** · `3` did not run

A first reading **ran**. `run()` calls `train(...)`, reads six beats out of
`heartbeat.jsonl`, and writes `runs/canary/baseline.json`. What it could not do is *compare*,
because there was no baseline to compare against. That is `2`, stated in the script's own words.
`docs/gates.md` defines `2` as *"floor UNKNOWN — no recorded floor for this key"*, which is
precisely and only a first reading.

**The original defect collapsed 3 → 0. The fix collapsed 2 → 3.** Same disease, inverted.

### The gate that was supposed to catch this is aimed at the wrong axis

This is the part worth the brief. The mapping is **already under test and already mutated**:

- `tests/test_evidence_machinery.py:346` asserts `canary.exit_code_for("baseline") == 3`
- `scripts/mutation_battery.py:529-530` mutates `"baseline": 3` → `"baseline": 0` and confirms
  that test reddens

So the battery reports this gate **PROVEN**. And it is — it proves the *old* defect cannot
return. It cannot see that the current value is wrong, because every artefact that could have
caught it was written by the same reading of the protocol. **A test written by the fixer, mutated
against the defect the fixer was fixing, cannot detect the fixer's error.** That is the finding.

### Do this the way the v0.5 self-audit did it

`docs/ROADMAP.md` §6 already names the rule and the precedent: *"`K = M` is too small… Same
disease as D-1, found by applying D-1's test to the other frozen constants."* One fix, applied as
a class, caught a second live defect for free. §6's conversion queue lists **"Exit 3 collapsing to
0 → one shared `Exit` enum and a result helper every checker uses"** as its highest-value row, and
it is still unconverted. That is why this is a brief and not an inline fix.

---

## The audit — every exit-code site in the repo, at `6a775b74`

Manager's reading. **Re-execute it; do not inherit it.** `grep -rn 'sys.exit\|EXIT_CODES\|exit_code\|raise SystemExit\|returncode' --include='*.py' .`
(quote the `--include` glob — zsh expands it otherwise and the grep returns nothing, which reads
as "no sites").

### 🔴 Wrong

| Site | Now | Should be | Why |
|---|---|---|---|
| `scripts/canary.py:69` | `"baseline": 3` | `2` | It ran. Only the comparison had nothing to compare. The script says so at `:35` and `:67`. |
| `scripts/canary.py:151` | `led.command(..., exit_code=0, ...)` | the code actually returned | A **literal**, written before the verdict is computed at `:196`. The process may exit 1 (MOVED) or 3 (baseline) and the ledger still records 0. `scripts/ledger.py:42` made `exit_code` required because `null` hid exactly this — **a hardcoded `0` is worse than `null`, because it looks measured.** |
| `scripts/mutation_battery.py:709` | `raise SystemExit("the suite is not green before mutating: …")` → exits **1** | `3` | Nothing was mutated. The battery **did not run**. |
| `scripts/mutation_battery.py:685` | `raise SystemExit("mutation …: anchor not found … The battery is stale")` → exits **1** | `3` | Same: the battery did not run. A stale anchor is not a failed mutation. |
| `experiments/e0a/run.py:7` and siblings `e0d`, `e0e`, `e0f`, `e0g`, `e0h`, `e0i` — all at `:7` | `raise NotImplementedError(...)` → uncaught, exits **1** | `3` | 🔑 **Seven of them.** "E0A is not implemented yet" is the definition of *did not run*, and every one reports it as *real failure*. This is the sibling sweep paying for itself — none of these were in the original finding. |

### ⚠️ Ambiguous, decide and record

| Site | Reading |
|---|---|
| `scripts/render_scoreboard.py:437` | A malformed `--claim` (no `:`) returns `3`. Defensible — it did not run. But `argparse` itself exits **2** on a usage error at `:428`, so this one script gives `2` **two** meanings: "bad arguments" (argparse's) and "no rows" (`:480`'s). Pick one and make the other explicit. |
| `scripts/mutation_battery.py:773` | `return 1 if (args.check and bad) else 0`. There is no `2` and no `3` in the mapping at all. Without `--check`, unproven gates return `0`. A battery with zero mutations to run is *nothing to compare*, not *pass*. |

### ✅ Correct — checked, and named here because "checked and correct" is a result

| Site | Code | Verdict |
|---|---|---|
| `experiments/e0c/run.py:210` | MPS unavailable → `3`, with the comment `# did not run -- never report as 0` | Correct. **This is the template the enum should generalise.** |
| `experiments/e0c/measure.py:232` | MPS unavailable → `3` | Correct. |
| `experiments/e0c/run.py:206` | `--point` printed → `0` | Correct. |
| `scripts/render_scoreboard.py:432` | `--over` directory missing → `3` | Correct. |
| `scripts/render_scoreboard.py:446` | `--audit` file missing → `3` | Correct. |
| `scripts/render_scoreboard.py:441` | `0 if ok else 1` on a claim check | Correct. |
| `scripts/render_scoreboard.py:450` | every number resolves → `0` | Correct. |
| `scripts/render_scoreboard.py:455` | unbacked numbers → `1` | Correct. A real failure. |
| `scripts/render_scoreboard.py:480` | `2 if not board.rows else 0` | Correct — **the repo's one right use of `2`**, and the reference reading for canary `:69`. |
| `src/rsr/cli.py:26,:31,:32` | `0` / `1` only | Correct. No "nothing to compare" or "did not run" state exists on this path, so there is nothing to map. |
| `tests/conftest.py:85` | suite floor breached → `SystemExit(1)` | Correct. A collapsed suite is a real failure, not an absence. |
| `experiments/e0d,e0e,e0f,e0g,e0h,e0i,e0a/run.py:11` | `raise SystemExit(main())` | The pass-through is correct. What `main()` returns is not — see the 🔴 table. |

**Not audited, and say so:** `scripts/extract_golden_tensors.py:451` (`sys.exit(main())`) — its
`main()` was not read. It is a one-time throwaway-venv tool per `third_party/PINS.md`. **Read it
and put it in one of the three tables.** An unaudited site in an audit is the hole the audit
exists to close.

---

## Files in scope

- `scripts/canary.py`
- `scripts/mutation_battery.py`
- `scripts/render_scoreboard.py`
- `experiments/*/run.py`, `experiments/e0c/measure.py`, `scripts/extract_golden_tensors.py`
- `src/rsr/exit_codes.py` — new, or wherever the enum belongs
- `tests/` — new tests
- **Not** `src/rsr/train/`. Cycle 1 owns `loop.py` and a concurrent edit there collides.

## The bar

1. **One shared `Exit` enum**, `0/1/2/3`, with the protocol as its docstring, and a result helper
   every checker uses. §6's conversion queue names this and it is the deliverable.
2. **`canary.py:69` maps `"baseline"` to `2`.** The existing test at
   `test_evidence_machinery.py:346` asserts `3` and **must be changed** — changing it is not
   moving a goalpost, it is correcting a value that was never measured against anything. Say so
   explicitly in `RESULTS.md` and quote this brief.
3. 🔴 **The new mutation must be on the right axis.** `mutation_battery.py:529-530` mutates
   `baseline: 3 → 0`. That only proves the old defect stays dead. **Add a mutation that maps
   `baseline` to `3`** — the current, wrong value — and prove it reddens the corrected test and
   only it. A gate that cannot see the error actually made is the thing this brief is about.
4. **A test that no checker returns a bare boolean**, per §6's conversion row.
5. **Seven experiment stubs return `3`**, not an uncaught `NotImplementedError`. One mutation
   covering the class, not seven.
6. `canary.py:151`'s `exit_code` is **read, not asserted**. If the verdict is not known at the
   point the row is written, move the row.
7. `uv run pytest -rs` and `uv run python scripts/mutation_battery.py --check` both green, counts
   reported separately from skips. **Read `$?` directly, never after a pipe** — this is zsh, so it
   is `$pipestatus[1]`, and `grep -c` exits 1 when the count is 0.

## Done when

One PR to `main` with the enum, the corrected mappings, the tests, the mutations, and a
`RESULTS.md` carrying the re-executed audit — **including every site that turned out correct.**

## Do NOT

- 🔴 **Do not touch `src/rsr/train/loop.py`.** Cycle 1 is live in that file.
- Do not change a frozen constant, bucket edge, threshold or arm to make anything pass.
- Do not treat `docs/gates.md` as a description of this tree. It is a specification of intent; it
  is quoted here for the *definition* of `2`, nothing else.
- Do not delete a mutation because its anchor went stale. Fix the anchor — `:685` says so itself.

## BRIEF ERRORS

Required, and `none` must be written out. **Assume this brief has some.** Its audit was produced
by one manager reading each site once; that is exactly the process that produced the defect it
describes.
