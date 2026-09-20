# Studio launch prompt — 2026-09-20

> 📌 **This file lives in the repo on purpose.** The first version was written to the MacBook vault
> at `~/Desktop/Claude Memory Palace/Projects/RSR/`, which **does not exist on the Studio** — the
> exact failure `docs/RESEARCH-CONTEXT.md` already names: *a document that guides agents has to
> arrive with `git pull`.* There are two directories called `Projects/RSR/`, one in the vault and
> one in this repo, and a bare relative path resolves differently in each. Reach for this file as
> `docs/lab-notes/dispatch-2026-09-20-studio-launch.md`.

You are the **RSR manager** on the Mac Studio. Your role file is in the repo at
`.claude/agents/rsr-manager.md` — read it before anything else. `~/.claude/agents/` does not exist
on this machine, which is why the role files were moved into the repo: a document that guides agents
has to arrive with `git pull`.

Working directory: `~/retrieval-successor-retention`
🔴 **Use `/opt/homebrew/bin/git`, never bare `git`.** Apple's blocks on a licence prompt and fails
as an *empty answer rather than an error*. That has already produced two false clean bills of health
on this project.

---

# TASK 0 — reconcile two histories. Nothing else starts until this lands.

**This machine has 7 commits nobody else has seen. Trunk has 5 commits you have not seen. They
diverged at `d1c221f` and both directions contain real work.**

```
this machine   d1c221f → … → 45892dd   (7 commits, UNPUSHED)
origin/main    d1c221f → … → c83fde1   (5 commits, not here)
```

🔴 **Do not pull. Do not reset. Do not rebase onto trunk before pushing.** Your 7 commits are the
only copy of ~2,537 insertions.

### What is on this machine and not on trunk

`scripts/render_scoreboard.py` (451 new) · `tests/test_evidence_machinery.py` (366 new) ·
`scripts/mutation_battery.py` (+511) · `scripts/ledger.py` (+403) · `scripts/canary.py` (+110) ·
`docs/mutation-battery.md` (regenerated, +340) · `runs/cycle-00-evidence-machinery/` ·
`Projects/RSR/{dispatch/cycle-01-masked-loss.md, for-brendan-2026-09-19.md, morning-2026-09-19.md,
overnight-2026-09-19.md}`

### What is on trunk and not here

`docs/ROADMAP.md`; five sprint briefs under `docs/lab-notes/dispatch-S*.md`; E0f pass 2 with
corrections 25–30 (**[P11], the cognitive claim, is wrong in the spec in three places**);
`docs/cognitive-grounding.md`; an Apache-2.0 LICENSE and the repo is now **public**; `Projects/RSR/*`
**moved to `docs/lab-notes/`**; ruff green.

### Do this

1. **Push your work first, to its own branch.** `/opt/homebrew/bin/git push -u origin
   HEAD:studio-cycle-0-1`. Verify it is on the remote before touching anything else.
2. `git fetch origin`, then merge `origin/main` into a merge branch — **not** into `main` directly.
3. **Expect exactly two real conflicts, both in `scripts/`.** Trunk's only change to
   `scripts/canary.py` and `scripts/ledger.py` was `ruff format` reflowing — verified, no logic.
   **Resolution: take YOUR version of both files wholesale, then run `ruff format`.**
4. **`Projects/RSR/*` is a location conflict, not a content one.** Trunk moved that directory to
   `docs/lab-notes/`. Move your four new files there too. Leave references *inside* them pointing at
   the old path — they are a historical record.
5. `uv.lock` is untracked here and tracked on trunk. Take trunk's.
6. Gate the merge: `.venv/bin/ruff check` exit 0 · `.venv/bin/ruff format --check` · `pytest -rs`
   with **no failures and no new skips** (trunk measures **287 passed, 0 skipped**; your
   `test_evidence_machinery.py` adds more — report the new count, measured, never typed).
7. Open a PR and merge it. Then `main` here fast-forwards to trunk and the two histories are one.

⚠️ **Report what the merge cost.** If anything of yours was dropped, say which file and why.

---

# TASK 1 — reconcile the two brief sets, then tell me which is right

You already wrote `Projects/RSR/dispatch/cycle-01-masked-loss.md` — loop.py defect 1, the loss
scoring padding. Trunk now carries `docs/lab-notes/dispatch-S0-01-loop-defects.md`, which covers
**that same defect plus four more** and was written without knowledge of your cycle-1 work.

**I planned against a tree that was not trunk. Say so plainly if the overlap is worse than I think.**

Read both. Then report which of these is true:
- your cycle-1 brief supersedes S0-01, or
- S0-01 supersedes it, or
- they compose — cycle-1 is defect (a) and S0-01 adds (b)–(e).

Also check the same way: **did cycle 0 already fix the things trunk's roadmap proposes?**
`docs/ROADMAP.md` §6 lists `mutation_battery.py`'s `off_gate` never entering the `unproven` filter,
and `canary.py` exiting 0 on a first reading. Your `+511` and `+110` may have closed both. **If they
are already fixed, the roadmap is stale and I want the correction, not the diff re-applied.**

---

# TASK 2 — verify the workspace, and report each line as measured

Do not assume any of this. I checked from the MacBook; check it yourself and report literal output.

| Check | What I saw from here |
|---|---|
| `.venv/bin/python -V` | Python 3.12.13 |
| torch + MPS | torch 2.14.0, `mps.is_available()` **True** |
| `.claude/agents/` resolves as project agents | `rsr-manager.md`, `rsr-researcher.md` present in-repo. 🔴 **Confirm the agent types actually register in your session** — on 2026-09-18 they did not exist at all and the manager ran from its launch prompt alone |
| `measurements/ledger.json` | **MISSING — expected.** Every `MEASURED` constant raises. That is the registry working, not a defect |
| `.rsr/` | **MISSING — expected.** `docs/gates.md` specifies ratchet machinery this tree does not implement |
| `data/` | **MISSING.** Needed only for E0i |
| fastcoref / wtpsplit | **NOT installed.** `uv pip install -e ".[coref]"` — only when S1-01 starts |
| disk | 709 GB free |

---

# What comes next, and what does NOT

After Tasks 0–2, the order is in `docs/ROADMAP.md`. **Sprint 0 is the substrate; Sprint 1 (E0i) is
CPU-only and runs beside it.**

🔴 **S1-01 is blocked.** `preregistration/e0i_threshold.md` is final and **unsigned**, and says *"it
must be signed by a person, and that person is not a model."* **Do not sign it. Do not start E0i.**

🔴 **No unattended overnight cycles yet.** Owner directive: they resume only when Sprint 0's gate —
the shuffle control — is green. The last 13 cycles ran above a model whose working memory moves the
loss by **exactly 0.0**.

🔴 **Do not start E1, do not tune `γ_b` or `ν`, do not touch §15.** All of it sits above a memory
that does nothing.

## The rules that are not negotiable

- **Literal output, or it did not happen.** `287 passed` is a result; "tests pass" is not.
- **Exit 3 is not exit 0.** "Did not run" and "found nothing" are different facts, and **never read
  `$?` after a pipe** — zsh spells it `$pipestatus[1]`.
- **Every number comes from a ledger. None is typed.** Join on `run_id`, never on `cycle`.
- **Always report spread.** An sd of exactly 0.0000 across seeds is a broken result, not a clean one.
- **A new gate is not believed green until a mutation has shown it red — and reddened only it.**
  If nothing reddens it, the gate adds nothing and **that is the finding.**
- **Commit the brief before dispatching**, in its own commit, baseline sha read from
  `git rev-parse HEAD` at the moment of writing. `BRIEF ERRORS` is a required field coming back.
- **You never run an experiment yourself.** You re-execute a sample to verify. *An agent that both
  produces and accepts a result can launder its own work.*

## First message back

Task 0's outcome with the literal gate output and the new test count; Task 1's verdict on the two
brief sets and on whether cycle 0 already closed the roadmap's items; Task 2 as a table of measured
lines. Then stop and wait — **do not begin Sprint 0 work in the same session.**
