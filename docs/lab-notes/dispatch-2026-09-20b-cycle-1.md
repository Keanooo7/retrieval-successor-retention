# Studio dispatch — 2026-09-20b · clear the decks, then run cycle 1

**Baseline: `4ece42958abf3ccb6514095a22186f9819ca3fc6`** (read from `git rev-parse HEAD` at the moment of writing).
You are the **RSR manager**. Your role file is `.claude/agents/rsr-manager.md`.

## 🔴 Launch condition — check this first and stop if it fails

```
cd ~/retrieval-successor-retention   # ← the session must START here, not in ~
```

Your last session ran from `/Users/keanooo7`, so the repo-scoped `.claude/agents/` never loaded
and you wrote that report from the dispatch text alone — the 2026-09-18 failure in a second form.
**Confirm `rsr-manager` and `rsr-researcher` are in your agent roster before doing anything.** If
they are not, say so and stop. Do not work around it a third time.

---

## You were right four times. I checked all four by a different path.

| Your correction | Verdict |
|---|---|
| Trunk had 7 commits, not 5 | ✅ #9 and #10 landed after I drafted |
| `uv.lock` is **not** on trunk | ✅ it is on `macbook-local-2026-09-18`, which is not an ancestor of `main`. My instruction to "take trunk's" named a file trunk does not have |
| Six conflicts, not two | ✅ and your `-w` note is the useful part: **the line count would have talked you out of my correct instruction.** AST-identical was the right check |
| The canary item is nowhere in `ROADMAP.md` | ✅ `grep -i canary docs/ROADMAP.md` → nothing. I told Brendan it was in §6. It is not |

**PR #11 is merged.** Trunk is `4ece429`, one history, your 13 paths all present,
`origin/studio-cycle-0-1` = `45892dd` preserved untouched.

---

## TASK A — the brief contradiction. I am deciding it, not asking you.

You found that S0-01's `Do NOT` forbids the measurement cycle-1 makes its falsifier. **That is my
error and it is the worst kind this project has.** I wrote *"Defect (a) makes them uninterpretable;
saying so is the deliverable"* — asserting a belief as settled **and then barring the test of it.**
A claim carried forward as fact because someone wrote it once is the exact failure this project's
own record is about, and I put it in a brief.

**Resolution, in this order:**

1. **`dispatch-cycle-01-masked-loss.md` owns defect (a) and supersedes S0-01 on it.** Its
   falsifier — the run-to-run floor at identical seed and config — is the discriminator. Without it
   "the loss scores padding" can be true and inert.
2. **Amend `dispatch-S0-01-loop-defects.md`:**
   - **Delete the `Do NOT` clause about re-running ledgers entirely.** Do not soften it.
   - **Remove row (a).** Cross-reference cycle-1.
   - **Demote (d).** You verified it is not independently fixable — `loop.py:165` is
     `policy = FIFOPolicy()` unconditional, so renaming at `:236` leaves `attr` still `None`.
     It is downstream of (b). S0-01 calls it independent and "not silent"; both are wrong.
   - **Keep (b), (c), (e) and the two flagged-uncertain items.**
   - 🔴 **Demote every number in S0-01 from premise to prediction.** "93.4% of 193,536" is something
     to measure, not something to assume. Write it as an expectation with the command that tests it.
3. Commit the amendment **before** dispatching anything, in its own commit.

---

## TASK B — three small things, then the real work

**B1. Correct `ROADMAP.md` §6.** The `off_gate` row describes the pre-cycle-0 tree. Your fix is
better than what I proposed — `off_gate_allowed` with a named reason per coupling, plus
`test_an_off_gate_failure_makes_a_mutation_unproven` and
`test_a_declared_coupling_does_not_make_a_mutation_unproven`. **Record the fix, do not re-apply the
diff**, and say plainly that the roadmap was stale.

**B2. Write the canary finding as a brief** — do not fix it inline. You are right:
`scripts/canary.py:13` states *"`2` nothing to compare · `3` did not run"* and
`EXIT_CODES` then maps `"baseline": 3`. A first reading trained, produced six beats and wrote
`baseline.json`. **It ran. Only the comparison could not.** `docs/gates.md` defines `2` as
*"floor UNKNOWN — no recorded floor for this key"*, which is precisely a first reading.

🔑 **The finding is not the mapping, it is the shape.** The original defect collapsed **3 → 0**. The
fix collapsed **2 → 3**. Same class, inverted, by the person fixing it. So the brief must do what
the v0.5 self-audit did with `K = M` — *"same disease as D-1, found by applying D-1's test to the
other frozen constants."* **Apply this test to every exit-code mapping in the repo before closing
it**, naming each one checked, including the ones that turn out correct.

**B3. Escalations** to `docs/lab-notes/for-brendan-2026-09-20.md`: the launch-directory
requirement, the canary finding, and anything from Task A you disagree with.

---

## TASK C — dispatch cycle 1. This is the session's real output.

The first experiment on a substrate anyone has reason to trust. **You do not run it.** Spawn a fresh
`rsr-researcher`, retire it after one brief.

**Before dispatching, re-verify the brief's premises against `4ece429`.** It was written against a
pre-merge tree. *Rebasing a brief does not revalidate it.*

**Bar — the discriminator, and it is the whole point:** the run-to-run floor at identical seed and
config. Two runs differing only in a label must be compared against that floor before any
padding-effect claim is read as signal. On MPS the previously measured floor was
`3.1789143850602386e-07`; **treat that as a number to reproduce, not to assume.**

**Report spread. An sd of exactly 0.0000 across seeds is a broken result, not a clean one** — that
has already happened here and a reported 0.0000 is what exposed it.

---

## Standing, non-negotiable

- **Literal output, or it did not happen.** Read `$?` directly; **never after a pipe**
  (`$pipestatus[1]` in zsh). I made that exact mistake today verifying my own gate.
- **Exit 3 is not exit 0, and 2 is not 3.** That distinction is Task B2's entire subject.
- **Every number comes from a ledger.** Join on `run_id`, never `cycle`.
- **A new gate is believed only when a mutation reddens it and reddens only it.** If nothing reddens
  it, the gate adds nothing and that is the finding.
- **Commit the brief before dispatching**, baseline sha read at write time.
- **`BRIEF ERRORS` is required back.** You returned four on the last dispatch and all four were
  right. **Assume this brief has some too** — I have now been wrong about `uv.lock`, the conflict
  count, the trunk commit count, the roadmap's contents, and the S0-01 `Do NOT`. Five in two
  dispatches. Check me.
- **Do not paraphrase this into the spawn prompt.** Send a verb and a path.

## Out of scope — unchanged

E0i and its signature · overnight cycles · `γ_b`, `ν`, E1 · §15 · the capture bridge (S0-02) ·
the rewardable corpus (S0-03).

## Report

Task A as a diff summary. B1–B3 as landed commits. **Task C as a verified cycle**: the researcher's
report, which claim you re-executed yourself and how, and your verdict — `survived`, `falsified`
or `inconclusive`, the three words and no others.

🔴 **A cycle with zero rejections and zero re-executions is not a good cycle, it is an unexamined
one.**
