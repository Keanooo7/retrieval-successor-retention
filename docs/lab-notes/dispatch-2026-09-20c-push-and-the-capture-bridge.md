# Studio dispatch — 2026-09-20c · make last night reachable, then build the blocker

**Baseline: `25732a2`** (read from `git rev-parse HEAD` at the moment of writing).
You are the **RSR manager**. Your role file is `.claude/agents/rsr-manager.md`.

## 🔴 Launch condition — check first, stop if it fails

```
cd ~/retrieval-successor-retention   # the session must START here, not in ~
```

Confirm `rsr-manager` and `rsr-researcher` are in your agent roster before doing anything.

---

## 🔴 TASK 0 — PUSH. Blocking. Nothing below is verifiable until this is done.

Your return names `6a775b7`, `5440290`, `e40fc05`, `dab4f08` and `37ca1c5`. On the MacBook, at
14:43 PDT, **all five resolve to nothing**:

```
6a775b7 -> fatal: Not a valid object name 6a775b7      (same for the other four)
origin/main                   = 8ad64a2   (unmoved since 13:17)
origin/merge-studio-trunk     = 2f939f8   (2026-09-20 12:50 — the PRE-merge commit)
```

I verified Task A, B1, B2, B3 and cycle 1 **by no path at all.** Every claim in that return is
currently unfalsifiable from here, and a verdict nobody else can check is not a result. This is §3's
trap in its other direction: *committed is not reachable.*

### 🔴 Two destructive hazards. Read both before you type.

1. **DO NOT run `git reset --hard origin/main`.** It is step 3 of the standing launch block and it
   will **destroy nine unpushed commits.** It is deliberately absent from this dispatch's launch
   block. If you have already run it, stop and say so — `git reflog` still has them, and recovering
   them is then the whole task.
2. **Do not push to `merge-studio-trunk`.** `origin/merge-studio-trunk` is `2f939f8`, the *pre-merge*
   commit that PR #11 **squashed** into `4ece429`. Pushing there revives a history main has already
   absorbed, and squash-merge has no merge-base to tell them apart — that is what produced six
   conflicts last time. **Push to a new name: `studio-2026-09-20c`.**

```
git push -u origin studio-2026-09-20c
git ls-remote origin studio-2026-09-20c
git log --oneline origin/main..HEAD          # expect the nine
git log --oneline HEAD..origin/main          # what you do not have
git merge-base --is-ancestor origin/main HEAD; echo $?    # 0 = built on trunk, 1 = NOT
```

🔑 **That last exit code is the real question and it decides the next week's shape.** If it is `1`,
your nine commits are stacked on the pre-squash branch rather than on trunk, and reconciliation is
another conflicted merge rather than a rebase. **Report the literal `0` or `1`. Do not fix it yet.**

**Report Task 0 before starting Task 2.** Everything below can wait; this cannot.

---

## TASK 1 — two things that must exist as artefacts, not as claims to me

**1a. Your rejection must be in the file.** You rejected the report's claim that cycle 1 *"retires
the third of the three candidate causes"*, on the grounds that **the shuffle control has no positive
control and has never been shown to read large on a live memory.** That is upheld here without
qualification, and it is the same argument as S0-04's own Bar item 1: *"a gate that has never been
observed failing is not known to work."* A rejection that lives only in a return to me is a lesson,
and **a defect filed as a lesson recurs.** Put it in the cycle's `RESULTS.md` and the scoreboard, in
the voice of the record, next to the claim it rejects.

**1b. S0-05's sweep has a boundary and must name it.** Your brief says *"every exit-code site named,
including the twelve correct ones."* That is true of trunk and **false of the project**, in two
directions, neither of which your sweep could have seen:

- 🔴 **`src/rsr/gates/` is not on trunk.** `git ls-tree origin/main src/rsr/` has no `gates`.
  It exists only on `origin/macbook-local-2026-09-18`, and
  `git merge-base --is-ancestor origin/macbook-local-2026-09-18 origin/main` → **`1`, not an
  ancestor.** That tree holds `src/rsr/gates/floors.py`, whose `class Exit(IntEnum)` is
  `OK=0 DROP=1 UNKNOWN=2 DID_NOT_RUN=3 UNBANKED_RISE=4` and which **emits `4` at `:118` and `:129`**.
  So your finding *"`4 UNBANKED RISE` … no site in the repo returns it"* is exactly right about
  trunk, and the reason is bigger than a missing return: **the only implementation of the five-code
  protocol is in a tree main cannot reach.** Two documents on trunk then describe it running —
  `docs/gates.md:83` heads the block *"the same five everywhere"*, and
  `docs/decision-review.md:82` states *"the ratchet correctly reported `UNBANKED_RISE` before it was
  banked."* `git grep -iln ratchet origin/main -- '*.py'` returns **nothing**. Both of your role
  files mention the ratchet, so **both Studio roles are instructed about an instrument absent from
  the tree they work in.**
- **The vault half is unreachable to you and is mine.** The `''`-on-failure git helper that the
  2026-09-20b dispatch attributed to this repo is not here at all —
  `git grep -nE "return ''|return \"\"" origin/main` over every file type returns **nothing**. It is
  `const git = … catch { return ''; }` in the vault's `.claude/orchestrator/brief-status.cjs:65` and
  `route.cjs:99`. My error, corrected at source; **do not go looking for it.**

**What S0-05 must gain:** a section stating which trees the sweep covered and which it structurally
could not, and that `4` is unreachable on trunk because its implementation is. 🔑 **A gate that
cannot name what it does not catch is not a gate.** Everything else in S0-05 stands — the seven
stubs, `mutation_battery.py:685,:709`, and `canary.py:151` are all confirmed here at `8ad64a2`, at
those exact lines.

---

## TASK 2 — S0-02, the capture bridge. The session's real output.

**Read `docs/lab-notes/dispatch-S0-02-capture-bridge.md` and execute it.** I am not restating it
here; *a paraphrase is a second, divergent copy.* Spawn a fresh `rsr-researcher` and retire it after
the one brief.

**Why this and not S0-04.** S0-04 is the sprint gate and its positive control is now the live
question — but S0-04 declares *"Depends on: S0-01, S0-02, S0-03 all landed"*, and S0-02 is not.
Dispatching it now would be me overriding a written constraint because I wanted the result, which is
the specific failure of the 2026-09-20b dispatch. S0-04 follows this one.

### Its premises, revalidated at `8ad64a2` — its own baseline `c647af4` is stale

*Rebasing a brief does not revalidate it, so I ran each one rather than trusting it.*

| S0-02 says | Checked |
|---|---|
| `reward.py` 160 lines, 11 passing tests, zero callers in `src/` | ✅ 160 lines; 11 `def test_`; **zero** importers of `rsr.retention.reward` anywhere in `src/` |
| `git grep '\.observe(' -- src/` returns zero hits | ✅ zero |
| `last_attention` at `model.py:335` | ✅ exact |
| `memory_gate` an `nn.Parameter` at `model.py:388` | ✅ exact |
| `attn_out_proj` applied to the α-weighted output at `:337-338` | ✅ the call is `:337`; `:338` is a comment |
| `policy_loop.py` exists | ✅ |

⚠️ **One I could not confirm, and it is Bar item 2's yardstick.** `310 ± 2 sent/s` appears in
**eight documents** and in **no ledger** — `git grep` over `runs/**/ledger.json` finds no row for it.
It traces to `experiments/e0c/RESULTS.md`, which exists and was really run, so this is a provenance
gap, not a fiction. 🔑 **But read `docs/RESEARCH-CONTEXT.md:741-744` before you use it as a
baseline:** the E0c figure is random tokens, one data shape per row, **no optimizer step**, MPS only
— *"a measurement of the model's throughput, not of a training loop's."* **The capture-on vs
capture-off delta is the number the bar wants, measured in one harness. Do not report it as a
percentage of training throughput, and do not quote `310` as a training rate.**

---

## MY BRIEF ERRORS — all three confirmed here, by a different path

1. 🔴 **The floor. You were right and it was load-bearing.** `runs/cycle-arm-identity/ledger.json`,
   key `loss_max_abs_diff.fifo_mps_a__vs__fifo_mps_b`, value `3.1789143850602386e-07`, **`how`:
   `"max_t |loss_x[t] - loss_y[t]| over the 10 beats"`** — a trajectory sup-norm, where cycle 1's
   falsifier is about final loss. And the same run records `final_loss_cpu_two_runs: n=2, sd 0.0`
   with `fifo_cpu vs rsr_cpu` bit-identical. **I handed you a discriminator that degrades to "moves
   by more than zero" on the device I did not pin.** Your `inconclusive` on CPU was correct against
   my brief, not despite it.
2. **`scripts/canary.py:13` is wrong.** Line 13 is *"with whatever else is on the GPU — a second
   run, a browser, Spotlight"*. The protocol is `:34-35`, the comment `:67-68`, the mapping `:69`.
   I read a line number off a file I had not opened at that line.
3. **Task A contradicted itself.** I told you to demote `"93.4% of 193,536"` to a prediction and to
   delete row (a), which is where that number lived. Resolving it by subtraction was right.

**Nine of mine in three dispatches now.** Two more you did not catch because you could not: I put
the exit-code example in the wrong repo, and I told you `ROADMAP.md` §6 listed the canary item when
`grep -i canary` returned nothing. **Assume this brief has some too, and return `BRIEF ERRORS`.**

## Predictions, with the command that tests each

Not premises. If one is wrong, that is the finding, and say so.

| Prediction | Command |
|---|---|
| `merge-base --is-ancestor origin/main HEAD` → `0` | the Task 0 block |
| `pytest -rs` at your tip → `passed=323 failed=0 skipped=0` | `.venv/bin/python -m pytest -rs`; read `$?` before anything else |
| Capture-off throughput is within noise of capture-off before the change | E0c's harness at `d=128, S=80, batch=16`, capture off, 3 repeats, spread reported |
| A `W_O`-zeroing mutation changes `r_i` | S0-02 Bar 3 |

## Standing, non-negotiable

- **Literal output, or it did not happen.** Read `$?` **directly, never after a pipe**
  (`$pipestatus[1]` in zsh, captured before any other command or it is clobbered).
- **Exit 3 is not exit 0, and 2 is not 3.**
- **Every number comes from a ledger.** Join on `run_id`, never `cycle`.
- **Report spread. An sd of exactly 0.0000 across seeds is broken, not clean.**
- **A new gate is believed only when a mutation reddens it and reddens only it.** If nothing reddens
  it, the gate adds nothing and that is the finding.
- **Commit the brief before dispatching**, baseline sha read at write time, in its own commit.
- **The manager never runs an experiment.** It re-executes a sample to verify.
- **Do not paraphrase this into the spawn prompt.** A verb and a path.
- **Push before you report.** Task 0 is the whole reason this dispatch exists.

## Out of scope

E0i and its signature · overnight cycles · `γ_b`, `ν`, E1 · §15 · S0-03 · S0-04 (it follows this
one) · the vault half of S0-05 · fixing the `src/rsr/gates/` split, which is a reconciliation
decision and not yours to take unasked.
