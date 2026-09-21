# S0-05 — the exit-code conflation, converted from a lesson into a gate

**Baseline:** `a802521f06be561f0457447f0ae706cd8630d000` (re-baselined 2026-09-21 by the manager; was `6a775b74`).
**Worktree:** `.worktrees/s0-05` on branch `s0/s0-05` from `origin/main`. One agent, one worktree. **run_id:** none (this is code, not an experiment); the re-executed audit goes in `RESULTS.md`.

> **Re-baseline, 2026-09-21. Every anchor below was re-derived with `grep -n` at `a802521`. Anchors further down this brief that are not listed here are stale; use these instead.**
>
> | brief says | at `a802521` | what is there |
> |---|---|---|
> | `canary.py:35`, `:67`, `:69`, `:151` | **unchanged** (`:34-35` docstring, `:67` comment, `:69` `EXIT_CODES = {"held": 0, "MOVED": 1, "baseline": 3}`, `:151` `exit_code=0,`) | |
> | `canary.py:196` (verdict computed) | not re-derived. Find it yourself | |
> | `mutation_battery.py:685` (stale anchor → exit 1) | **`:914`** | `raise SystemExit(` |
> | `mutation_battery.py:709` (suite not green → exit 1) | **`:938`** | `raise SystemExit(f"the suite is not green before mutating: ...")` |
> | `mutation_battery.py:773` (`return 1 if (args.check and bad) else 0`) | **`:1002`** | same |
> | `mutation_battery.py:529-530` (mutation `baseline: 3 → 0`) | **`:565-571`** | `Mutation("a first canary reading exits 0 again", "test_a_first_canary_reading_exits_3_not_0", ...)` |
> | `test_evidence_machinery.py:346` (`exit_code_for("baseline") == 3`) | **`:395`** | same |
> | `experiments/e0{a,d,e,f,g,h,i}/run.py:7` | **unchanged**; all seven still `raise NotImplementedError` (`grep -ln`) | |
> | `render_scoreboard.py:432/437/441/446/450/455/480` | **unchanged** | |
> | `e0c/run.py:206`, `:210`; `e0c/measure.py:232` | **unchanged** | |
> | `extract_golden_tensors.py:451`; `tests/conftest.py:85` | **unchanged** | |
> | `docs/gates.md:83-90`, "the same five everywhere" | **heading rewritten 2026-09-20** to "Exit codes — the protocol, which trunk does not yet implement". The five-code block is now at **`:89-93`**, and the "`2` and `3` are separate deliberately" sentence at **`:96`** | |
>
> **Two notes that the original brief could not have made:**
> - *"Do not touch `src/rsr/train/loop.py`. Cycle 1 is live in that file"*: cycle 1 has landed. The prohibition **stands for a new reason**: **S0-03 is editing `loop.py` tonight** in `.worktrees/s0-03`.
> - `tests/test_evidence_machinery.py` was edited tonight by Brief 0 (#27): the canary-tally tests. Rebase your mental map of that file on `a802521`.
> - The Brief 0 researcher found `scripts/mutation_battery.py --check` at **48/49, exit `1`** on the baseline, because the "checkpoints written straight to the final path" row is timing-dependent (caught 1 in 7). **Do not fix that here**; it is a separate brief. It does mean that "`--check` green" in this brief's bar may fail for a reason that is not yours. If so, report the flaky row by name, show that your own rows are PROVEN, and do not call the battery green.
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

1. **One shared `Exit` enum**, `0/1/2/3/4` — **five, not four; see the correction below** — with the
   protocol as its docstring, and a result helper every checker uses. §6's conversion queue names
   this and it is the deliverable.
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

One PR **opened** against `main` (the manager verifies and merges; do not merge), with a `BRIEF ERRORS` field in your report prepended to `.orchestrator/outbox/researcher.md`, and with the enum, the corrected mappings, the tests, the mutations, and a
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

---

## 🔴 CORRECTION, same day, before dispatch — the protocol has FIVE codes

The bar above originally said `0/1/2/3`. **That is wrong, and it is the same error the brief
describes: I read the protocol off `canary.py`, which is the file under suspicion.**

`docs/gates.md:83-90` — read at source, not quoted from a dispatch:

```
0  at or above the floor
1  FLOOR DROP              — the thing this exists to catch
2  floor UNKNOWN           — no recorded floor for this key
3  ENVIRONMENT / DID NOT RUN   <- NOT A PASS
4  UNBANKED RISE           — measured above the floor, floor not updated
```

under the heading **"Exit codes — the same five everywhere."** And the two lines after it are the
sharpest statement of this brief's whole subject, from the project's own hand:

> **`2` and `3` are separate deliberately, and `3` is where this design can be silently defeated.**
> "The gate did not run" and "the gate found nothing" are different facts.

**That sentence condemns `canary.py:69` directly.** Cite it in your `RESULTS.md`; it is better
evidence than anything I wrote above it.

Two further things fall out, and neither was in the original audit:

1. **`canary.py`'s own docstring at `:35` lists only four codes** and omits `4` entirely. The
   file that got `2` wrong also silently narrowed the protocol it claims to follow. **The
   docstring is a third defect in that file, alongside `:69` and `:151`.**
2. 🔑 **No site in this repo returns `4`.** `UNBANKED RISE` — measured above the floor, floor not
   updated — is specified in `gates.md` and implemented nowhere. `grep -rn 'return 4' --include='*.py' .`
   ⚠️ **Do not treat that as a defect on sight.** `gates.md` is a specification of intent, not a
   description of this tree, and an unimplemented code may simply be unreached. **Determine which
   it is and say so** — "specified and deliberately unimplemented" and "specified and forgotten"
   are different facts, in exactly the way the quotation above means.

*Found by re-executing a quotation I had published without checking: the `2` definition in this
brief came from the dispatch that commissioned it, not from `gates.md`. The definition turned out
correct. The table around it did not. Record this as a `BRIEF ERRORS` entry against me.*

---

## 🔴 THE SWEEP'S BOUNDARY — added 2026-09-20 by the manager, re-executed at `8ad64a2`

The audit above says *"every exit-code site in the repo."* **That is true of trunk and false of the
project.** A sweep that does not name its own frame invites the reader to promote "not found" to
"not there", which is the same promotion this brief exists to prevent.

🔑 **A gate that cannot name what it does not catch is not a gate.**

### Trees the sweep covered

| Tree | Covered | How |
|---|---|---|
| `origin/main` @ `8ad64a2` (= the audit's `6a775b74` for these files) | ✅ | `git grep` over the working tree |
| the Studio working tree @ `37ca1c5` | ✅ | same, plus `e76c42e`'s battery re-anchor |

### Trees the sweep structurally could not cover

**1. 🔴 `src/rsr/gates/` is not on trunk, and it holds the only implementation of the protocol.**

```
git ls-tree origin/main src/rsr/                          -> no `gates` entry
git merge-base --is-ancestor origin/macbook-local-2026-09-18 origin/main ; echo $?
1                                                          -- NOT an ancestor
```

`origin/macbook-local-2026-09-18` (`df508ee`) carries `src/rsr/gates/__init__.py` and
`src/rsr/gates/floors.py`. That file's `class Exit(IntEnum)` at `:32-37` is

```
OK = 0   DROP = 1   UNKNOWN = 2   DID_NOT_RUN = 3   UNBANKED_RISE = 4
```

and it **emits `4` at `floors.py:118` and `:129`** — the rise case for a
may-rise-never-fall floor and for a may-fall-never-rise ceiling respectively. Both verified by
`git show origin/macbook-local-2026-09-18:src/rsr/gates/floors.py`.

**This resolves the open question in item 2 of the correction above**, and the answer is neither of
the two offered. `4` is not *"specified and deliberately unimplemented"* and not *"specified and
forgotten."* It is **specified on trunk, implemented on a branch trunk cannot reach.** The finding
*"no site in the repo returns `4`"* is exactly right about trunk, and the reason is larger than a
missing `return`.

**2. Two trunk documents describe that implementation running.**

- `docs/gates.md:83` heads the block **"Exit codes — the same five everywhere."**
- `docs/decision-review.md:82`: *"the ratchet correctly reported `UNBANKED_RISE` before it was
  banked."*

But `git grep -iln ratchet origin/main -- '*.py'` returns **nothing** (exit `1`). The ratchet is
named on trunk only in prose — including in **both** `.claude/agents/rsr-manager.md` and
`.claude/agents/rsr-researcher.md`. ⚠️ **Both Studio roles are instructed about an instrument that
is absent from the tree they work in.** Recorded, not fixed: the `src/rsr/gates/` split is a
reconciliation decision and not the researcher's or the manager's to take unasked.

**3. The vault repository is out of frame entirely.** The `''`-on-failure git helper attributed to
this repo by the 2026-09-20b dispatch is not here —
`git grep -nE "return ''|return \"\"" origin/main` over all file types returns nothing. It lives in
a different repository. Corrected at source by Brendan; **not swept here, and not to be.**

### ⚠️ The audit's line anchors are trunk-relative and have already drifted

Confirmed at `8ad64a2`, at the exact lines the audit names:

- `scripts/mutation_battery.py:685` → `raise SystemExit(`
- `scripts/mutation_battery.py:709` → `raise SystemExit(f"the suite is not green before mutating: ...")`
- `scripts/canary.py:151` → `exit_code=0,`

On the Studio branch at `37ca1c5`, `e76c42e` shifted the battery by **+21 lines**: those two sites
are now `:706` and `:730`. `canary.py:151` is unmoved. Cite anchors with the sha, or the next reader
re-reads a line number off a file that has moved under it — which is brief error 2 of the
2026-09-20c dispatch, in its other direction.

### What still stands

Everything else in this brief is unaffected and re-confirmed at `8ad64a2`: the seven experiment
stubs returning `3`, the two `mutation_battery.py` sites, `canary.py:151`, and the three defects in
`canary.py` (`:35` docstring, `:69`, `:151`).
