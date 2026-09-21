# For Brendan — 2026-09-20

Written by the manager at `5440290`, under `dispatch-2026-09-20b-cycle-1.md` Task B3.
One line each, with what I did about it in the meantime.

---

## 1. The launch-directory requirement is enforced by a sentence, not by anything

`.claude/agents/` is repo-scoped on purpose, and a session started from `~` silently loads no
role — which is what happened on 2026-09-19 and is the 2026-09-18 failure in a second form.
**Today's dispatch fixed it by asking me to check. That is a lesson, not a gate**, and §6 of the
roadmap is a document about exactly that distinction. A session that forgets to check is
indistinguishable from one that has no role, and the failure mode is that it works anyway and
writes a plausible report. *I checked; both roles loaded. I have no mechanism to propose that
does not itself run inside the session that might not have loaded.* **This one is yours.**

## 2. The exit-code class is the highest-value unconverted row in §6, and it just recurred

Briefed as S0-05 (`docs/lab-notes/dispatch-S0-05-exit-code-enum.md`), not fixed inline. The
short version: the cycle-0 fix to `canary.py` corrected `3 → 0` by writing `2 → 3`. Same class,
inverted, by the person fixing it.

🔑 **The part I want you to see is not the mapping.** That mapping is already tested
(`test_evidence_machinery.py:346`) and already mutated (`mutation_battery.py:529-530`), and the
battery reports it **PROVEN**. It is — of the *old* defect. Every artefact that could have caught
the new error was written by the same reading of the protocol that produced it. **A mutation
battery cannot audit the axis its author was not looking along.** I do not think that is fixable
by adding more mutations, and I do not have a proposal. It is the sharpest limit I have found on
the instrument we are trusting most.

Applying §6's sibling rule found six more sites beyond the canary, including **seven experiment
stubs that report "not implemented yet" as exit 1** — a real failure — when it is exit 3.

## 3. `CLAUDE.md` is stale about the two headline tests

`CLAUDE.md` says *"`test_reduction.py` and `test_fidelity.py` are currently skipped pending the
transcription; GATE-1 must say so."*

Measured at `5440290`, on this machine, `$?` read without a pipe:

```
uv run pytest -q -rs tests/test_reduction.py tests/test_fidelity.py
passed=44 failed=0 skipped=0 errors=0      EXIT=0
```

Both files exist and both run. **A GATE-1 that says what `CLAUDE.md` currently instructs would
report 44 passing tests as skipped.** I have not edited `CLAUDE.md` — it is the standing rules
file and correcting it is an owner decision, not a manager's. It needs your hand.

## 4. Local `main` on this machine points at a pre-merge tree

```
main                     45892dd    (= origin/studio-cycle-0-1, 2026-09-18)
origin/main              8ad64a2
merge-studio-trunk       8ad64a2    <- where the work is
```

`git checkout main` here lands on the Studio's pre-reconciliation branch, seven commits behind
and nine ahead in the wrong direction. Every brief in the tree says *"a PR is merged to `main`"*.
Not urgent, but it is a trap laid for the next session that reads one of those briefs literally.
I have not moved it — moving someone's branch pointer is not mine to do.

## 5. `uv.lock` is still untracked

Unchanged since your last dispatch confirmed it is not on trunk. Noting it so it does not become
invisible through familiarity; `ledger.py` refuses to write over an uncommitted `src/`, and
`uv.lock` is not `src/`, so it does not block cycle 1.

---

## Where I disagree with today's dispatch — Task A

**Nowhere on the substance.** I checked all four of Task A's resolutions against the tree and
every one holds. Defect (d) in particular: `loop.py:165` is `FIFOPolicy()` unconditional and
`attribution_counts` exists only on `RSRPolicy` (`rsr.py:450`), so a rename at `:236` leaves
`attr` as `None`. Downstream of (b), as you said.

I went one step further than you asked and you should know it: **S0-01 also called (d) "not
silent", and that is wrong too.** `attribution=attr` writes a `null` that reads as "no attribution
this run" rather than "the call never resolved" — which is silent by this project's own
definition. I recorded that in the amendment.

`BRIEF ERRORS` for today's dispatch are in my report, not here. There are four.

---

## 6. (2026-09-20f) `RESEARCH-CONTEXT.md` §10.3 still says "exactly 0.0", and that did not reproduce

`experiments/shuffle-control/RESULTS.md` re-derives *inert* (`survived`, ratio to live decoy
~7.8e-4, seed 2 bit-identical on my re-execution) but the signed mean is non-zero on all three
seeds. The wording in the orientation document is yours to change; I have not edited it.

## 7. (2026-09-20f) Next run needs a decision: a trained *live* memory, or the cause of inertness

Both are proposed, neither is decided. The role file's "memory is live" row now refuses every
training run in the audit's configuration, so the project cannot produce a filed training number
until one of these lands. Which comes first is a research-direction call.

## 8. (2026-09-20f) `rsr-manager.md` / `rsr-researcher.md` schema note is stale

Both say `scripts/ledger.py` does not write `manifest.json`, `config_hash`, `seeds_actually_run`,
`steps_done`. It wrote all four in `runs/shuffle-control/`. The researcher's role file also points
reports at `.orchestrator/outbox/`, which does not exist. Role files are yours; not edited.
