# For Brendan — 2026-09-22: every open decision, with its sources

Written in the interactive session of 2026-09-22 at Brendan's request: *"For all of the decisions
that are needed I want you to create a md document I will review … ensure that you are properly
documenting each point so that an agent can clearly see where you got your questions from."*

## How to read this

- **Citations.** Unless marked otherwise, every `file:line` is pinned to commit **`1ce2999`** on
  branch `infra/2026-09-22`. To check one:
  `git show 1ce2999:<file> | sed -n '<line>p'`. Lines move after a commit; the sha does not.
- **Every entry has the same fields:**
  - **Decide**: the question.
  - **Why it is yours**: the rule that reserves it.
  - **Source**: where the question came from.
  - **Facts**: what is measured, each with its source.
  - **Options**.
  - **Recommendation**: marked as the assistant's, or "none" where the project's rules say agents
    must not propose one.
  - **Blocks**: what cannot proceed without the decision.
  - **Queue**: the `docs/queue/items/` item and the ruling filename that closes it.
- **How an answer takes effect.** A decision takes effect when a ruling file
  `docs/owner/rulings/R-<date>-<slug>.md` exists at the night's base. Its front matter needs
  `id`, `date` and `stated_in` (`docs/owner/rulings/README.md`). The queue matches the ruling by
  the pattern in the item's `answered_by`. Headless agents cannot write under `docs/owner/`
  (`scripts/orchestrator/hooks.py`), so rulings are written in an interactive session in which
  you state them.
- **Two warnings for any agent reading this:**
  - Several queue items cite their source by a line number that has since **drifted**. Agent G
    added dated notes to `docs/RESEARCH-CONTEXT.md` and `docs/ROADMAP.md` after agent A seeded the
    queue. Each such item is marked **(queue source stale)** below, and the line given here is the
    current one.
  - **Spec §15 is not on this list and must never be.** `CLAUDE.md` ("Do not fill §15");
    `docs/RESEARCH-CONTEXT.md:1094-1097`.

## Summary

| # | Decision | Kind | Blocks |
|---|---|---|---|
| A1 | Nightly spend caps | system | the unattended loop starting at all |
| A2 | Merge `infra/2026-09-22` | system | everything built today reaching `main` |
| A3 | Install the launchd tick | system | unattended operation |
| A4 | `mlx_lm` servers overnight | system | C0's memory reserve; the MPS lane |
| A5 | Branch protection on `main` | system | nothing; hardening |
| A6 | UPS / FileVault | system | recovery after a power cut |
| A7 | Night window opens at 21:00 | system | nothing; an assumption to confirm |
| A8 | `disableBypassPermissionsMode` in the headless settings | system | nothing; hardening |
| B1 | Sprint 0 gate: is the decisive result "green"? | research gate | Sprint 2 (E0d, E0e, E0a re-run) |
| B2 | May Sprint 2 start? | research gate | E0d, E0e, E0a re-run |
| B3 | Hinge on or off (arm C vs B) | research | the default training config |
| B4 | The inconclusive band exits `1` | research | the liveness-wiring brief's exit table |
| B5 | Canary re-baseline (MPS and CPU) | research integrity | the canary as a drift check |
| B6 | E0i: the signature, and whether the signer was you | pre-registration | Sprint 1 (E0i) |
| B7 | E0i: 100 human-annotated reintroductions | pre-registration | Sprint 1 (E0i) |
| B8 | RESEARCH-CONTEXT §13.3 "Do not start E1": its stated reason is now out of date | prohibition | E1 (Sprint 4) |
| B9 | The seed-0 cross-machine re-run never landed | reproducibility | nothing; an open verification |
| B10 | Three runs ran on Python 3.14.6: re-run on 3.12? | reproducibility | nothing; provenance |
| C1–C6 | RESEARCH-CONTEXT §12, items 1–6 | owner-only design | E0e, E1, `bias.py`, the §16 record |
| D1 | ADR-0005 sign-off, and confirming the `c_t` pin | owner-only | E0a re-run, E7 |
| E1 | Revoke the GitHub tokens in `~/.claude/settings.local.json` | security | nothing; do it anyway |
| E2 | Delete about 600 MB of 09-18 scratch runs | housekeeping | nothing |

---

## A. The unattended system

### A1. Nightly spend caps
- **Decide:** `RSR_CYCLE_CAP`, the USD limit per Claude session, and `RSR_NIGHT_CAP`, the USD
  limit per night.
- **Why it is yours:** spend is the owner's. `ops/orchestrator.env` marks both caps
  "OWNER DECISION", and ruling R-2026-09-22-owner-out-of-loop keeps budgets with you.
- **Source:**
  - `ops/orchestrator.env:5-12`: both caps read `UNSET`.
  - `scripts/orchestrator/tick.py` refuses with exit `3` while either is `UNSET`. Observed today
    in the end-to-end dry run: `DID NOT RUN: … RSR_CYCLE_CAP=UNSET RSR_NIGHT_CAP=UNSET`.
  - Queue: `owner-nightly-spend-cap`. Its source field says "no repo line yet"; the line above
    is the source now.
- **Facts:**
  - `RSR_CYCLE_CAP` is passed to every session as `--max-budget-usd`.
  - A session whose cost cannot be read is charged the whole cycle cap (`ops/orchestrator.env`,
    comment above the caps).
  - An idle night starts no session at all.
  - This is not compute spend. `CLAUDE.md`'s "Compute spend for weeks 1–4 is zero" is about GPU
    hardware and still holds.
- **Options:** set both values, or leave them `UNSET` (the loop never starts).
- **Recommendation:** none on the amount; that is your budget.
- **Blocks:** the loop starting at all.
- **Queue:** `owner-nightly-spend-cap`, answered by `R-*-spend-cap`.

### A2. Merge `infra/2026-09-22`
- **Decide:** whether to merge today's branch into `main`. It will arrive as a PR once the final
  mutation battery reports.
- **Why it is yours:** R-2026-09-22-night-branch says Brendan performs the merges to `main`.
- **Source:** the branch itself, `git log origin/main..infra/2026-09-22`.
- **Facts:**
  - Test suite at `c57d9e4`: `passed=1049 failed=0 skipped=0 errors=0`, rc 0.
  - `ruff check` and `ruff format --check` both rc 0.
  - `render_status.py --check` rc 0.
  - Mutation battery: the first full run at `c57d9e4` exited **3** on two stale anchors. Both
    are fixed in `3eee8af`, and the re-run at `1ce2999` is in progress. Its result goes in the PR.
- **Options:** merge; merge after review; ask for changes.
- **Recommendation:** merge once the battery reports N/N with rc 0. Do not merge on any other
  battery result.
- **Blocks:** A3, and everything the system does.

### A3. Install the launchd tick
- **Decide:** when to install `ops/launchd/com.keanooo7.rsr.tick.plist`. It is written but not
  loaded. The steps are in `ops/README.md`.
- **Why it is yours:** it starts unattended sessions on your account.
- **Facts:** the plist runs `tick.zsh` every 600 s. Preconditions that the code enforces:
  - A1 is set, or tick exits 3;
  - `ops/lanes.json` exists, which means C0 has run (`scripts/orchestrator/lanes.py:57`, every
    lane refuses with exit 3 until then);
  - A2 is merged.
- **Recommendation:** install after C0 (run attended) and A1.
- **Expect this on night 1:** briefs, not runs. A brief written during a night becomes runnable
  only once it exists at a night's base, which is after your morning merge. Observed in today's
  dry run: all 16 ready items were skipped with "has no brief path".

### A4. `mlx_lm` servers overnight
- **Decide:** may the `mlx_lm.server` processes on :8090 and :8091 stay up during nights?
- **Why it is yours:** they are your services, not the project's.
- **Source:**
  - `docs/decision-review.md:305`: "`mlx_lm.server` routinely holding ~25 GB of the Studio's 64".
  - `experiments/e0c/PROCUREMENT.md:177`.
  - Observed resident today (`ps`).
- **Facts:**
  - E0c measured a 32 GB peak at d=128, S=80, batch=16 on MPS (`experiments/e0c/RESULTS.md`).
  - 25 + 32 GB leaves about 7 GB of 64 for the OS, agents and CPU lanes.
  - The MPS lane admits a job only if free memory minus its declared peak stays above
    `reserve_gb` (`scripts/orchestrator/lanes.py`).
- **Options:**
  - stop them at night-open;
  - keep them and let the MPS lane refuse jobs that do not fit;
  - keep them and exclude MPS work at night.
- **Recommendation:** stop them for the night, or C0's memory measurements will describe a
  machine the experiments do not run on.
- **Queue:** `owner-mlx-servers-overnight`, answered by `R-*-mlx-servers`.

### A5. Branch protection on `main`
- **Decide:** enable GitHub branch protection (no direct pushes; merge through PRs)?
- **Why it is yours:** repository settings on your GitHub account.
- **Facts:**
  - The Studio's `gh` is logged in as `Keanooo7`, so an agent's push is indistinguishable from
    yours.
  - Today's hook denies pushes to `main` (`scripts/orchestrator/hooks.py`; observed denying
    `git push origin HEAD:main` in the dry run). That is a client-side guard, and branch
    protection would be the server-side one.
- **Recommendation:** enable it. It costs you nothing, because you merge through PRs already.
- **Queue:** `owner-branch-protection`, answered by `R-*-branch-protection`. Source "no repo line
  yet".

### A6. UPS / FileVault
- **Decide:** buy a UPS, or accept that after a power cut the Studio waits at the FileVault
  password screen until someone types the password.
- **Source:** `docs/RESEARCH-CONTEXT.md:1224-1225` ("One risk nothing can protect against").
- **Facts:**
  - `pmset` shows `autorestart 1`, but FileVault still blocks an unattended boot.
  - After unlock, the tick's reconcile marks orphaned jobs `crashed`.
- **Queue:** `owner-ups`, answered by `R-*-ups`.

### A7. Night window opens at 21:00
- **Decide:** confirm or move `RSR_WINDOW_OPENS=21:00`.
- **Source:** `ops/orchestrator.env:16-19`. The file's own comment reads: "not in the stop
  rules; an assumption, the owner may move it". Build agent D made the assumption.
- **Facts:** the other three times (last dispatch 06:30, park by 07:30, digest by 08:00) come
  from `docs/lab-notes/dispatch-2026-09-21-overnight.md`, in its "Wall clock" stop rules.
- **Recommendation:** none; it is your schedule.

### A8. `disableBypassPermissionsMode` in the headless settings
- **Decide:** set it in `.claude/settings.orchestrator.json` and `.claude/settings.researcher.json`?
- **Source:**
  - `scripts/orchestrator/hooks.py:95-97`, the docstring's list of known limits.
  - Build agent E's report: the docs do not say whether a hook's deny holds under
    `bypassPermissions`; the `permissions.deny` rules do hold.
- **Facts:**
  - The launcher uses `--permission-mode dontAsk`, not bypass (`scripts/orchestrator/loopcore.py`,
    `claude_argv`).
  - Setting it removes one way around the guards, and could break a launcher that ever asks for
    bypass.
- **Recommendation:** set it. No part of the design needs bypass.

---

## B. Research gates and integrity

### B1. Sprint 0 gate: is the decisive result "green"?
- **Decide:** does the decisive run answer ROADMAP's Sprint 0 gate?
- **Why it is yours:** `docs/ROADMAP.md:117-119` says "Whether this gate is now 'green' is the
  owner's reading … this note does not decide it."
- **Source:**
  - `docs/ROADMAP.md:104-109`, the gate text: "the shuffle control, as a permanent instrument,
    green … A run where memory contributes ≈0 is refused, not filed".
  - `docs/ROADMAP.md:111-119`, the 2026-09-22 update.
  - `docs/lab-notes/overnight-2026-09-21.md:120`.
- **Facts** (`runs/decisive-shuffle/ledger.json`, ledger sha `90438f3`):

  | Key | Value | Reading |
  |---|---|---|
  | `armA.ratio` | 0.0030 ± 0.0023 | inert |
  | `armB.ratio` | 9.33 ± 4.68 | live |
  | `armC.ratio` | 7.74 ± 6.06 | live |
  | `armB.heldout.answer.answer_all.honest_nll` | 2.904 | at or above `ln 16` = 2.773 |
  | `armC.heldout.answer.answer_all.honest_nll` | 2.852 | at or above `ln 16` = 2.773 |
  | `armB.heldout.answer.gap_1.honest_nll` | 2.725 | below chance |
  | `armC.heldout.answer.gap_1.honest_nll` | 2.609 | below chance |

  - The gap-1 values are below chance, which the RESULTS prose missed. A dated reading note was
    added, and the ledger wins.
  - The second half of the gate, "refused, not filed", does not exist yet. It is the
    liveness-wiring brief (`docs/lab-notes/dispatch-liveness-wiring.md`), which waits on C0.
- **Options:**
  - green now (memory is live on B and C);
  - green once liveness wiring lands (the gate's refusal half);
  - not green until memory helps answers (retrieval shown).
- **Recommendation:** green **when liveness wiring lands**. The gate text asks for a live
  shuffle, which B and C meet, and for a refusal of inert runs, which does not exist yet.
  Retrieval is a stronger claim than the gate makes.
- **Blocks:** B2 and all of Sprint 2.
- **Queue:** `owner-sprint0-gate`, answered by `R-*-sprint0-gate*` **(queue source stale:** it
  says `docs/ROADMAP.md:75`, the gate is now at `:104`**)**.

### B2. May Sprint 2 start?
- **Decide:** may E0d, E0e and the E0a re-run begin, given memory is live but answers are at
  chance?
- **Source:**
  - `docs/lab-notes/overnight-2026-09-21.md:120`.
  - The researcher's open question, `.orchestrator/outbox/_legacy-researcher.md:62`: "Does
    'live' satisfy the Sprint 2 precondition when held-out answer NLL … is ~2.9 nats in every gap
    bucket (at or above ln16)?" (the gap-1 exception is in B1).
  - `docs/ROADMAP.md:85`: "No sprint begins before its predecessor's gate answers."
- **Facts:**
  - The Sprint 2 instruments (`loo.py`, the E0e FIFO+EMA measurement) measure retention signal
    on a model. If memory does not help answers, `r_i` (the retention reward) may carry little
    signal.
  - Everything in Sprint 2 is gated on this ruling in the queue: `docs/queue/items/e0d.md:11`,
    `e0e.md:11`, `e0a-rerun.md:11`.
- **Options:**
  - start after B1;
  - start only the instruments whose validity does not depend on retrieval (for example E0e's
    `ū` distribution on FIFO);
  - hold until memory helps answers.
- **Recommendation:** none. This is the scientific judgement the gate exists for.
- **Queue:** `owner-sprint2-start`, answered by `R-*-sprint2-start`.

### B3. Hinge on or off (arm C vs arm B)
- **Decide:** should the srep-norm hinge (0.01) be on by default?
- **Source:**
  - `docs/lab-notes/overnight-2026-09-21.md:97` ("3 seeds cannot separate them, and the hinge
    stays the owner's decision").
  - `:121`.
  - `experiments/decisive-shuffle/PREREG.md:21` ("the hinge (an open owner decision …)").
- **Facts:** B is 9.33 ± 4.68 and C is 7.74 ± 6.06; the sds overlap almost entirely.
- **Options:** on, off, or measure first (more seeds on B vs C, with a PREREG committed first).
- **Recommendation:** none on the value. If you want data first, the measurement option can be
  queued.
- **Queue:** `owner-hinge-b-vs-c`, answered by `R-*-hinge`.

### B4. The inconclusive band exits `1` (confirm or revise)
- **Decide:** confirm that a liveness ratio in `0.01 < ratio < 0.1` exits `1 FAIL`.
- **Why it is yours:** it amends a ruling you approved.
- **Source:**
  - `docs/owner/rulings/R-2026-09-22-inert-exit-5.md`, its erratum section.
  - `experiments/decisive-shuffle/PREREG.md:35-38` for the bands, and `:45` for "reported as such,
    not rounded to either outcome".
- **Facts:**
  - The plan you approved said "ratio < 0.1 → 5". That was the assistant's transcription error:
    it folds the PREREG's `inconclusive` band into `inert`. Build agent C caught it.
  - Corrected mapping:

    | Verdict | Exit |
    |---|---|
    | live | 0 |
    | inert (≤ 0.01) | 5 |
    | inconclusive | 1 |
    | invalid, including `ratio == 1.0` exactly (decoy aliasing, PREREG `:35`, `:61`) | 3 |

- **Options:**
  - `1` (fails the run's premise, "live memory was not demonstrated");
  - `2` (nothing to compare; arguably wrong, since the check ran);
  - a new code.
- **Recommendation:** `1`, already applied as the default in the brief.
- **Blocks:** nothing now; the liveness brief uses `1` unless you revise it.

### B5. Canary re-baseline
- **Decide:** re-baseline the MPS canary, and write the first CPU baseline.
- **Why it is yours:** baselines are owner-only. The hooks deny writes to
  `runs/canary/baseline*` (`scripts/orchestrator/hooks.py`).
- **Source:**
  - `docs/lab-notes/for-brendan-2026-09-21.md:7`: trunk MOVED on 6 of 6 beats from code, not
    environment. The old sha `7254080` reproduces bit-exact.
- **Facts (from today's canary change, `31b4dfc`):**
  - Baselines now carry `loss_path_hash`. The MPS baseline predates it, so every MPS reading now
    exits `2` (not comparable), never `1`.
  - `runs/canary/baseline-cpu.json` does not exist yet.
- **Recommendation:** re-baseline both at the merged `main` after A2, in one attended session.
- **Blocks:** the canary as a drift check; the `cpu-canary-baseline` queue item
  (`docs/queue/items/cpu-canary-baseline.md:10`).
- **Queue:** `owner-canary-rebaseline`, answered by `R-*-canary-rebaseline`.

### B6. E0i: the signature, and whether the signer was you
- **Decide:**
  1. Did you personally make commit `fcd79d4`?
  2. Which of the file's contradictory texts is true?
- **Why it is yours:** `docs/RESEARCH-CONTEXT.md:1112-1113` says "Do not sign
  `preregistration/e0i_threshold.md` … it must be signed by a person, and that person is not a
  model." `preregistration/` is owner-only.
- **Source:**
  - `docs/lab-notes/for-brendan-2026-09-21.md:5`.
  - `preregistration/e0i_threshold.md:3` ("AWAITING SIGNATURE … NOT IN FORCE UNTIL SIGNED").
  - `:291` ("Unsigned. It must be signed by a person, and that person is not a model.").
  - `:300-302`, filled: "Threshold set by: Brendan Keane  Date: 2026-09-17".
- **Facts, checked today:**
  - `fcd79d4`: author and committer are both `Brendan Keane <bkbrohon795@gmail.com>`,
    2026-09-17 23:29:18 -0700. Subject: "Sign the E0i pre-registration (threshold in force as of
    this commit)". It changes 2 lines of that one file.
  - It has **no** `Co-Authored-By` trailer.
  - ⚠️ `bkbrohon795@gmail.com` is **this Mac Studio's configured git identity**
    (`git config user.email`). Every agent commit made on this machine uses it, including all of
    today's: 221 commits in history carry it.
  - **The author field therefore cannot distinguish you from an agent.** The missing trailer is
    weak evidence only; not every agent adds one.
- **Options:**
  - (a) You signed it. Then remove the stale "unsigned" text at `:3` and `:291`, which only you
    may do.
  - (b) You did not. Then the signature is void under §13.2, E0i is **not in force**, and
    `fcd79d4` should be reverted by you.
- **Recommendation:** none. Only you know. Separately: give agents a distinct git identity (see
  the note at the end) so this cannot happen again.
- **Queue:** `owner-e0i-signature`, answered by `R-*-e0i-signature` **(queue source stale:** it
  says `for-brendan-2026-09-21.md:3`, the line is `:5`**)**.

### B7. E0i: 100 human-annotated reintroductions
- **Decide:** who annotates, and when.
- **Why it is yours:** it is human work, which no agent may do. T3 in the E0i PREREG needs it.
- **Source:** `docs/lab-notes/for-brendan-2026-09-21.md:5` ("T3 needs 100 human-annotated
  reintroductions").
- **Facts:** `src/rsr/data/coref.py` and `pg19.py` are still stubs. Their queue items
  (`eng-coref`, `eng-pg19`) are ready as code work.
- **Blocks:** Sprint 1, the data kill gate.
- **Queue:** `owner-e0i-annotations`, answered by `R-*-e0i-annotations` **(queue source stale:**
  `:3` → `:5`**)**.

### B8. RESEARCH-CONTEXT §13.3 "Do not start E1": its reason is now out of date
- **Decide:** keep, reword, or lift the prohibition.
- **Why it is yours:** it is a standing prohibition, and agents may not relax one.
- **Source:**
  - `docs/RESEARCH-CONTEXT.md:1138`: "Do not start E1, and do not tune `γ_b` or `ν`. All of it
    sits above a memory that does nothing."
  - Flagged by build agent G, who deliberately did not edit it.
- **Facts:**
  - "A memory that does nothing" is false for arms B and C since #34 (B1).
  - E1 is Sprint 4 and is also blocked by the sprint order, so lifting the prohibition changes
    nothing today.
- **Recommendation:** reword its reason to the current one ("memory is live but retrieval is not
  shown"), and keep the prohibition until Sprint 3's gate.

### B9. The seed-0 cross-machine re-run never landed
- **Decide:** re-run it, or drop the claim.
- **Source:** `docs/lab-notes/dispatch-2026-09-21-overnight.md:68`: "Seed-0 re-execution on a
  second machine: RUNNING at the time of this commit, NOT a premise." No later commit on any
  branch records its result (checked with `git log --all`).
- **Recommendation:** queue it as a verification item once the MacBook is available, or record
  it as abandoned. Either way, do not cite it.

### B10. Three runs ran on Python 3.14.6: re-run them on 3.12?
- **Decide:** re-run S0-03, E-feas-s003 and decisive on the pinned 3.12, or accept the
  provenance notes.
- **Source:**
  - Their ledgers record `provenance.python` 3.14.6.
  - `experiments/efeas/RESULTS-s003.md:104-110` has a dated provenance note.
  - Ruling R-2026-09-22-python-312.
- **Facts:**
  - E-feas matched bit-for-bit across 3.12 and 3.14, **but E-feas trains no model**. Build agent
    G noted that it therefore says nothing about training numerics.
  - The decisive run **does** train, so its 3.14 numbers are untested on 3.12.
- **Recommendation:** re-run the decisive run on 3.12 as a verification, a later queue item.
  It is the result B1 and B2 rest on.

---

## C. RESEARCH-CONTEXT §12: the six owner-only design decisions

Source for all six: `docs/RESEARCH-CONTEXT.md:1042-1082`, headed "Decisions only the owner can
make — do not make them, do not work around them … Measure around them; do not resolve them"
(`:1044`). Per that instruction, **no recommendation is given for any of them.** The facts below
are quoted or summarised from the cited lines.

- **C1. The `b_max` invariant** (`:1046-1060`).
  - `b_max` is FROZEN at 1.0, which is about one sd of `ψ̂`.
  - It is about 3.25× the median top-2 margin at M=40 and about 1.9× at M=8, and 92.3% ± 0.5% of
    decisions are crossable.
  - "All options need one sentence naming the invariant … The missing input is the definition of
    'comparable', and no further measurement supplies it."
  - Queue: `owner-bmax-invariant`, answered by `R-*-bmax-invariant` **(queue source stale:
    `:996` → `:1046`)**.
- **C2. `γ_b` scoping** (`:1061-1067`).
  - `γ_b = b_max/(0.25·E[lifetime])` gives 0.5 at M=8, which is 5× over its stated range of
    0.05–0.1.
  - The audit reads this as a scoping bug. `E[lifetime]` is also policy-dependent.
  - Queue: `owner-gamma-b-scope`, answered by `R-*-gamma-b-scope` **(queue source stale:
    `:1011` → `:1061`)**.
- **C3. `ProtectionBias` / `_score` interface** (`:1068-1073`).
  - `_score` calls `self.bias.b(slots)`, but `ProtectionBias` declares no `b`.
  - "Evidence favours `ProtectionBias` … but the spec names no accessor."
  - Blocks `eng-bias` (`docs/queue/items/eng-bias.md:10`).
  - Queue: `owner-bias-interface`, answered by `R-*-bias-interface` **(queue source stale:
    `:1018` → `:1068`)**.
- **C4. Clamping ν** (`:1074-1077`).
  - At `n_live ≥ 3`, a negative best peer-cosine gives a protective bonus that does flip victims.
  - "Decide in E1's ν sweep whether the term should be `− ν · max(0, cos)`."
  - Queue: `owner-nu-clamp`, answered by `R-*-nu-clamp` **(queue source stale: `:1024` →
    `:1074`)**.
- **C5. Per-step vs global normalizer** (`:1078-1079`).
  - The two differ on 5.79% ± 0.93% of victims. "Neither is wrong."
  - Queue: `owner-score-normalizer`, answered by `R-*-score-normalizer` **(queue source stale:
    `:1028` → `:1078`)**.
- **C6. Confirm S-7 was actually sent** (`:1080-1082`).
  - It is the only §16 release-gate edit an unattended loop made (`docs/release-conditions.md:17`,
    condition 4, commit `f1ea1a0`).
  - "Nothing in the repo records the authorisation beyond the loop's own prose."
  - Queue: `owner-s7-confirm`, answered by `R-*-s7-confirm` **(queue source stale: `:1030` →
    `:1080`)**.

## D. ADR-0005 and the `c_t` pin

### D1. ADR-0005 sign-off, and confirming the `c_t` pin
- **Decide:**
  1. Sign ADR-0005, which departs from the spec's ranked E7 stimulus list.
  2. Confirm that `c_t` is the current sentence gestalt.
- **Source:**
  - `docs/decisions/ADR-0005-e7-stimulus-set.md:3`: "Status: proposed — pending Brendan's sign-off".
  - `docs/RESEARCH-CONTEXT.md:1084-1086`: "the decision is made as D-C, the confirmation is not".
  - `docs/RESEARCH-CONTEXT.md:282` ("`c_t` is the CURRENT SENTENCE GESTALT (correction 19 / D-C)").
- **Blocks:** the E0a re-run (`docs/queue/items/e0a-rerun.md:13`) and E7, which is out of scope
  for weeks 1–4 anyway.
- **Queue:** `owner-adr0005-ct-pin`, answered by `R-*-adr0005-ct-pin` **(queue source stale:
  `:1034` → `:1084`)**.

## E. Security and housekeeping

### E1. Revoke the GitHub tokens in `~/.claude/settings.local.json`
- **Source:** `~/.claude/settings.local.json` lines 8, 10 and 12. These are user-level config,
  outside the repo. Each is a `Bash(curl … -H 'Authorization: Bearer github_pat_…')` allow rule.
  The token values are deliberately not reproduced here.
- **Facts:** the tokens sit in plaintext in a settings file that every Claude session on this
  machine reads.
- **Recommendation:** revoke them on GitHub, then delete the three rules. I have not touched
  user-level config.

### E2. Delete about 600 MB of 2026-09-18 scratch runs
- **Source:** Phase 0 of today's plan (`~/.claude/plans/memoized-popping-blum.md`, step 8: "Check
  before deleting").
- **Facts:**
  - `runs/res`, `runs/smoke` and `runs/smoke2` are 203 MB each. `runs/crash` is 4 KB.
  - They are untracked, with 0 files in git (`git ls-files`).
  - No tracked file outside `docs/lab-notes/` references them (`git grep`, today).
  - `runs/crash/hb.jsonl` is a hand-made heartbeat crash smoke from 09-18 that no test uses.
- **Recommendation:** delete them.

---

## One structural note, not a decision you have to make now

Agents and you share one git identity on this machine (B6). Every other guard built today
(rulings under `docs/owner/`, the hooks, the queue) works around that. Giving headless agents
their own `user.email`, for example `rsr-agent@localhost`, set in the job worktrees, would make
authorship evidence again. It would also have answered B6 by itself. If you want it, say so and it
becomes a queue item.
