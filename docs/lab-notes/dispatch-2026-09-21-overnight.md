# Dispatch 2026-09-21 — the §10.3 decisive experiment, overnight and unattended

**Baseline:** `5d8b0d02e9a31e52af1350978b7bbae7e68ee133` — read from `git rev-parse HEAD` at the moment
of writing. If `origin/main` has moved when you start, re-derive every line citation below before you
act on it.
**Lane:** manager (`claude --agent rsr-manager`). **Sprint:** 0. **Mode:** unattended until the closing
gate or a stop rule.
**Written by:** the MacBook guiding window. **Owner rulings in force:** Sprint 0 gate-work is permitted
overnight (§7 forbids publishing numbered research cycles, not building the gate) · the manager merges
each brief once its gates pass, **merge commit only** · **3 seeds** · one morning document for an ML
reader and a cognitive-science reader.

You are `rsr-manager`. Read `.claude/agents/rsr-manager.md` in full before anything else. **You do not
run an experiment.** You write briefs, spawn one `rsr-researcher` per brief, verify each return by a
different path than the researcher used, merge, and keep the record.

🔑 **I expect this brief to be wrong somewhere.** My dispatches for this project have carried errors,
and the Studio caught most of them. Two of the line numbers in my own plan for this
brief were wrong and were caught only by re-deriving them. **A refuted premise is a successful return.** `BRIEF ERRORS`
is required, and an empty field is read as a check that was not run.

---

## Why tonight matters — the one question

`docs/RESEARCH-CONTEXT.md:879`: *"The decisive experiment, and both outcomes are decisive. Repair the
loop, make the corpus rewardable, re-run the ablation with the shuffle control as the primary
readout."*

- The loop is repaired: cycle 1 (masked loss) and S0-01 (defects b, c, e).
- The instrument now exists and has been verified (#17, below).
- **The corpus is not yet rewardable** (S0-03 is unlanded).

Tonight lands S0-03 and then asks the question. Every later E1/E3/E7 comparison means something only
if the working memory does something, so this is the question the project is blocked on.

**For the cognitive reader:** nothing tonight tests whether RSR discovers Kintsch & van Dijk's
leading-edge strategy. Tonight establishes whether the model has a working memory that any retention
policy could act on. **The morning document must say this plainly.**

---

## Verified premises — re-executed on the MacBook, not read

**#17, the shuffle control — ACCEPTED, WITH TWO DEFECTS TO FIX IN WAVE 0.**

Worktree at `5d8b0d0`, `uv sync --extra dev`, Python `3.12.13`, torch `2.14.0`.

Baseline suite:
```
.venv/bin/pytest -rs -p no:cacheprovider --tb=no
passed=385 failed=0 skipped=0 errors=0 · 385 passed, 1 xfailed in 65.17s · exit 0
```

**Hand mutations of `src/rsr/metrics/memory_liveness.py`**, one per failure mode, each restored before
the next (`git status --porcelain -- src tests scripts experiments` empty after every one):

| mutation | failure mode it models | census | reddened | off-gate |
|---|---|---|---|---|
| `:53` roll by `0` (identity, not a derangement) | the harness sees nothing | `passed=383 failed=2` | `test_derangement_gives_every_row_another_rows_memory`, `test_live_memory_moves_the_loss` | 0 |
| `:131` `kv[perm] + 1e-3` | the harness invents a delta | `passed=384 failed=1` | **only** `test_replaying_each_rows_own_memory_reads_exactly_zero` | 0 |
| `:131` also `bc[perm], bv[perm]` | measures the bos-copy path, not cross-attention — the wrong implementation someone would write without reading the docstring | `passed=384 failed=1` | **only** `test_disabled_memory_reads_exactly_zero` | 0 |

Each failure mode is caught by a different control, and each of the last two is caught by that control
alone. **The last two mutations are not in `scripts/mutation_battery.py`**, whose two entries for this
gate (`:705`, `:716`) both model "sees nothing". Wave 0 adds them.

**Seed-0 re-execution on a second machine: RUNNING at the time of this commit, NOT a premise.**
- The Studio re-ran seed 2. I am re-running seed 0 on MacBook CPU with the ledger's exact argv.
- It started at `2026-09-21T08:30:15Z` and runs at ~12 s/step, finishing at about 09:30Z.
- Its row-by-row comparison lands as a **separate later commit** that nothing tonight depends on.
- **Do not cite it until that commit exists.**

**Threshold provenance, checked by commit order:**
- The prereg was committed at `39d0011`, 03:19Z.
- **Amendment 1** (`ba71e86`, 03:23Z) switched to the ratio of `mean|Δ|` to the same-seed live decoy,
  falsified at `≥ 0.1`. It was prompted by a 2-iteration run on the *untrained* decoy, because the signed
  mean cancels and would have passed a live memory.
- The manifest was written at 04:10Z, **47 minutes after the amendment**. The threshold was not set
  against the trained data.
- Observed ratios (the verdict `detail` in `runs/shuffle-control/ledger.json`) are `0.000991` / `0.000877` /
  `0.000471` for seeds 0/1/2, which is **101×–213× inside** the `0.1` cutoff, so
  no reasonable threshold changes the verdict.

**Defects found in #17:**
1. 🔴 **`render_scoreboard.py --over runs/ --audit experiments/shuffle-control/RESULTS.md` → exit `1`,
   with 16 numbers resolving to no ledger key.** They are mostly rounded renderings of real keys
   (the prose's rounded ratio stands for the `ratio_to_live_decoy` statistic, mean `0.0007794603145151641`), plus some derived figures with no key at all
   (`~1580` tokens moved, `364`). The project's rule is that a number with no ledger key is a claim,
   not a measurement. **It was accepted without its own audit gate passing.**
2. `provenance.dirty: true`. Almost certainly the untracked `uv.lock` that `uv sync` creates, because
   **`uv.lock` is not on main**. The gate that matters did pass: `Ledger.write()` refuses a dirty
   `src/ scripts/ experiments/` (`scripts/ledger.py:84`, `:527-533`) and it wrote. But no artefact
   records the **torch version**, so a cross-machine difference could not be attributed to it.

**What #17 does NOT establish.** It measured the audit's model: **unmasked, hinge off, old corpus.** It
cannot distinguish two explanations:
- the readout ignores memory;
- memory carries no row-specific content. A participation ratio of `1.0001/128` means near-collinear
  memories, so a swap is a near no-op *even if memory is read*.

Its own `experiments/shuffle-control/PREREG.md:90` says a trained-live negative control is still owed.

**#19, E-feas — ACCEPTED, WITH ONE DEFECT AND ONE CONSEQUENCE.**
- Prereg `663d829` (06:34:04Z) precedes the manifest (06:34:48Z). All commands are reproducible,
  `dirty:false`, status `ok`. Survived: headroom per seed `0.195 / 0.174 / 0.174` at M=16.
- 🔴 **`--audit experiments/efeas/RESULTS.md` → exit `1`**, with one unbacked number (`64`,
  documents per seed).
- ⚠️ **E-feas is Sprint 3 and ran before the Sprint 0 gate** (`docs/ROADMAP.md:56`, *"No sprint begins
  before its predecessor's gate answers"*). It is model-free, so the result stands. **But it reads
  `SyntheticConfig` / `generate`, which S0-03 changes.** It measured a corpus that is about to be
  superseded, and it is re-run tonight.

---

## Files in scope

**Manager:**
- `docs/lab-notes/`
- `.claude/agents/rsr-manager.md` and `.claude/agents/rsr-researcher.md` (wave 0 only)
- `.orchestrator/outbox/`
- `experiments/shuffle-control/RESULTS.md` and `experiments/efeas/RESULTS.md` (prose only)

**Each researcher:** exactly its brief's *Files in scope*. Brief 0 owns `scripts/mutation_battery.py`
(two entries) and `experiments/s0-02/write_ledger.py`.

🔴 **No agent touches:**
- `preregistration/`
- any existing `runs/*/` file (`git diff --stat -- runs/` over pre-existing paths must be empty at
  every merge)
- `docs/ROADMAP.md` §15
- `ADR-0008`
- the MATS submission

---

## Hypotheses (not instructions) — refute any of them

- **H1 — S0-03 is the only unmet dependency of the decisive run.** Refute it by naming a line of
  `dispatch-S0-03-rewardable-corpus.md` or of this brief that needs `RSRPolicy.observe`
  (`src/rsr/retention/rsr.py:442`, raises at `:448`). The shuffle harness measures a forward-pass loss
  delta and does not call `observe`.
- **H2 — CPU is the right device for this instrument.**
  - On CPU the within-seed replicate floor is exactly `0.0`.
  - On MPS it is about `2.5e-6` (cycle 1), the same order as #17's readout (`mean|Δ| ≈ 6e-6`), so MPS
    noise would swamp the signal.
  - Refute it by measuring the MPS within-seed replicate floor at this config and showing it is at least
    10× below `reading.mean_abs_token_delta` in `runs/shuffle-control/ledger.json`. The 10× is my
    choice, and I say so; the comparison class is a ledger key, not a number I typed.
- **H3 — two agents in one checkout corrupt each other.** `Ledger.write()` refuses dirty source, and
  `mutation_battery.py` mutates source in place. Hence one `git worktree` per brief, always.
- **H4 — S0-03's bar is vacuous on its own.** "Answer loss ≈ chance with memory zeroed" is predicted
  both by a working corpus and by an inert memory. It means something only beside answer loss with
  memory *live*.
- **H5 — a positive result tonight would be unattributable without a single-factor arm.** #17 → the
  decisive run changes corpus, masking and hinge at once. Arm A below changes only the corpus.

---

## Wave 0 — you, serial, before any spawn

**Setup:**

```bash
cd ~/retrieval-successor-retention
/opt/homebrew/bin/git fetch origin && /opt/homebrew/bin/git reset --hard origin/main
/opt/homebrew/bin/git rev-parse HEAD          # record it; this is the night's base
```

Confirm `rsr-researcher` is in your agent roster. **If it is not, stop and say so.**

**0.1 — repair your own instrument.** On branch `w0-manager-instrument`, one PR. Every item below was
verified still open at `5d8b0d0`:
- `rsr-manager.md:17` and `rsr-researcher.md:17` say five ledger fields are *"not yet written by
  `scripts/ledger.py`"*. They are written (`Ledger.__init__`, `manifest()`). **Left as is, you would
  skip five of your own rejection checks all night.** Correct both files.
- `rsr-manager.md:115` gives the verify command as `uv run pytest -q -rs --tb=no`. `tests/conftest.py:16`:
  *"`pytest -q` on top of `addopts = "-q"` is `-qq`, which prints no count at all."* Change it to
  `uv run pytest -rs --tb=no`.
- `rsr-researcher.md:33` says there is *"no `.orchestrator/` in this repo"*, but `:134` says to *"Append
  to `.orchestrator/outbox/researcher.md`"*. Resolve it: the outbox exists, **you read it**, and nothing
  reads it mechanically.
- `rsr-manager.md` cites `rsr floor --check`, which does not exist on trunk. Mark it illustrative.

**0.2 — make the two accepted results pass their own gate.** Edit prose only; ledgers are untouched.
- **Also append a dated correction to the lab notes that cite `rsr.py:433`.** Do not rewrite them; the
  raise is at `:448`. They are `overnight-2026-09-18.md:58`, `overnight-2026-09-20.md:104` and
  `dispatch-2026-09-20d-s0-01.md:26`. Give `dispatch-S0-01-loop-defects.md:122`'s `(:144)` the same
  treatment: the derived vocab read is `src/rsr/train/loop.py:244`.
- Rewrite each unbacked number in `experiments/shuffle-control/RESULTS.md` and
  `experiments/efeas/RESULTS.md` so that it quotes a ledger key's value, or delete it. Those are the only
  two options the auditor gives.
- `render_scoreboard.py --over runs/ --audit <file>` must exit `0` for both. Read `$?` directly.

**0.3 — canary.** Run `uv run python scripts/canary.py <cycle>` and read `$?` directly (`0` held · `1`
MOVED · `3` first reading). Repeat it after every training brief and once at the end. **If it moves,
every result since the last good reading is suspect. Say so; do not explain it away.**

**0.4 — escalate, do not edit.** Write to `docs/lab-notes/for-brendan-2026-09-21.md`, one line each:
- **E0i is signed but its own text says unsigned.** Commit `fcd79d4` (2026-09-17) fills in the signature
  at `preregistration/e0i_threshold.md:300-302`, while `:3` and `:291` still say unsigned. So do
  `docs/ROADMAP.md:87` and `dispatch-S1-01-e0i.md:6`. The signing commit's author email is not the ucsd
  address; the owner confirms it is theirs. **S1-01 stays out tonight regardless**, because T3 needs 100
  human-annotated reintroductions.
- E-feas ran out of sprint order (above).

Merge wave 0 (gates below) before spawning anyone.

🔑 **Why the code items are not yours:** an agent that adds a gate and then accepts it has verified
its own work. The battery entries and the producer fix are Brief 0, done by a researcher, and verified
by you.

---

## The CPU lane — serial, one researcher per brief, one worktree per brief

```bash
/opt/homebrew/bin/git worktree add .worktrees/<slug> -b s0/<slug> origin/main
cd .worktrees/<slug> && uv sync --extra dev
```

Spawn each as `rsr-researcher` with **a verb and a path**, never a paraphrase:
`"Execute docs/lab-notes/<brief>.md"`.

### Brief 0 — two code fixes the manager must not make itself (write it, short, then spawn)

Branch `s0/w0-code`. Six headings, like every brief. Verify it by applying Task A's two mutations by
hand yourself and reading which node ids redden. The **battery's** PROVEN is not your check, because
it is the researcher's instrument.

**Task A — add the two new mutations to the battery.** In `scripts/mutation_battery.py`, next to `:705` and
`:716`, add:
- `"the shuffle replay perturbs the memory it replays"`: `:131` `kv[perm]` → `kv[perm] + 1e-3`
- `"the shuffle replay hands over the bos gestalt too"`: `:131` `bc, bv` → `bc[perm], bv[perm]`

Both use gate `"test_shuffle_control.py::"`.

`uv run python scripts/mutation_battery.py --check` must exit `0`. The final line must read **N+2/N+2**,
where N is today's count. Measure N; do not carry a number from this brief.

**Task B — stop the producer minting a wrong citation.** `experiments/s0-02/write_ledger.py:225` writes
`"NotImplementedError from rsr.py:433"` into every ledger it emits. The raise is at **`:448`**
(`def observe` is at `:442`). Fix the producer, and only the producer.
- **Do not touch anything under `runs/`.** `git grep -l 'rsr\.py:433' -- runs/` lists the files that
  carry the old citation: 10 at `5d8b0d0`, 5 of them `ledger.json`. They were true when written.
- The dated lab notes are the manager's, handled in wave 0.2.

### Brief 1 — S0-03, the rewardable corpus

Path: `docs/lab-notes/dispatch-S0-03-rewardable-corpus.md`. Its dependency, S0-01, landed in #15. Before
spawning, **re-baseline it in its own commit**: its baseline `c647af4` is stale. Re-derive every line it
cites (`src/rsr/data/synthetic.py:227`, `:238`) with `grep -n`.

Add this to its bar, in the same commit, because H4 says its bar is otherwise vacuous:
> Report answer-token loss with memory **live** and with all `M` slots **zeroed**, the gap between them,
> and the same split at `gap > M` vs `gap < M` under FIFO. **If live and zeroed are both at chance, the
> corpus is built but retrieval is not shown.** That is a valid return that routes to the decisive run,
> **not a pass of this bar.**

Required in the return: every figure as a ledger key with `how`, 3 seeds, the sd, and the `gap > M`
fraction. The determinism digest test `test_byte_identical_across_two_separate_processes`
(`tests/test_synthetic.py:176`) stays green.

Mutation bar: the corpus code is new, so a code mutation is trivially "only this test". **Mutate the
fixture**: make the generator emit the answer out of band again, and require a test to redden.

### Brief 2 — E-feas re-run on the S0-03 corpus (write this brief; it is short)

- **Only after S0-03 is on main.** Use a **new `run_id`**, `efeas-synthetic-s003`. `runs/efeas-synthetic/`
  is untouched.
- Same prereg rule as `experiments/efeas/PREREG.md`, committed ahead of the run.
- Report headroom at M=16 and M=32 against `0.181 ± 0.012` and `0.034`.
- This is a sibling check. It is required because S0-03 changes the object E-feas measured.

### Brief 3 — S0-05, the exit-code enum

Path: `docs/lab-notes/dispatch-S0-05-exit-code-enum.md`.
- Its baseline `6a775b74` is stale. **Re-baseline it in its own commit, and re-derive every line it
  cites with a command.** My own re-baselining pass of S0-01 introduced a false correction, so verify
  each correction before publishing it.
- `scripts/canary.py:69` is still `{"held": 0, "MOVED": 1, "baseline": 3}` on `5d8b0d0`.

---

## The decisive run — only after S0-03 is on main, and after a canary reading

Write `docs/lab-notes/dispatch-2026-09-21-decisive.md`. **Commit it, together with its pre-registration
`experiments/decisive-shuffle/PREREG.md`, in commits that precede any run.** Spawn a **fresh**
`rsr-researcher` that has seen none of the S0-03 numbers.

**Instrument:** `src/rsr/metrics/memory_liveness.py`, reused. **Do not write a third implementation.**
The experiment script follows `experiments/shuffle-control/run.py` (its `CONFIG` is at `:51-61`) and
changes only the arm settings.

**Decision rule: #17's Amendment 1, unchanged** (`experiments/shuffle-control/PREREG.md:96`, `:118`):
- `ratio = mean|Δ|_trained / mean|Δ|_decoy`, at the same seed, where the decoy is the untrained model at
  the same config.
- Memory is **live** on a seed if `ratio ≥ 0.1`.
- Any failed control (memory-disabled ≠ 0, own-memory ≠ 0, decoy = 0) makes that seed `inconclusive`.
- It was fixed before any of tonight's data existed. **No threshold is chosen tonight.**

**Arms.** All arms use the new corpus and the #17 `CONFIG` otherwise (300 iters, batch 16, 16 slots,
48 steps/stream), seeds `0, 1, 2`, **CPU** (H2):

| arm | `masked_loss` | `srep_norm_reg_weight` | isolates |
|---|---|---|---|
| **A** | `False` | `0.0` | **the corpus alone** — exactly #17's config except the corpus |
| **B** | `True` | `0.0` | the repaired objective |
| **C** | `True` | `None` → TGConfig `0.01` | the hinge, which is **the owner's open decision**. Running both informs it without taking it |

**Readouts,** every one a ledger row with `how`:
1. **Primary:** the Amendment 1 ratio on all real tokens, per seed and per arm, with the sd. This is
   comparable with #17.
2. **Secondary:** the same rule on **answer tokens only**, split `gap > M` vs `gap < M`.
3. **Rival-hypothesis discriminators.** These must be pre-registered as secondary, and they are what
   #17 could not tell apart:
   - **Cross-row cosine** of the trained memory contents. Near-1 means a swap is a near no-op whatever
     the readout does.
   - **Matched-norm random replacement** of the trained memory. If loss does not move, the readout
     ignores memory. If it moves while the shuffle does not, memory is read but carries no row-specific
     content.

**Say which outcome occurred.** Both are decisive:
- **Inert on B and C** means the memory path itself is broken: a wiring or gate problem.
  - Escalate.
  - **Sprint 2 does not start.**
  - The discriminators say which part is broken.
- **Live on any arm** means the cause was the objective or the corpus. Arm A's verdict says whether the
  corpus alone was enough.
  - S0-04's owed trained-live negative control now exists (`PREREG.md:90`).
  - Write the per-run wiring brief: liveness in `src/rsr/train/loop.py` and the heartbeat, refusal exit
    codes that keep "did not run" apart from "ran and inert". `git grep memory_liveness --
    src/rsr/train/` is empty today.
  - **Write that brief; do not start it.** The owner reads the result first.

Mutation bar: the experiment script is new. **Mutate the arm settings.** Swap two arms' configs and
require the manifest-hash check to catch it. Point the decoy at the trained checkpoint and require
`decoy = trained`, so that `ratio == 1` flags it.

---

## Bar — yours, on every return

- **Verify by a different path than the claimant used.** Re-executing a seed the researcher did not
  single out, applying a mutation by hand and reading which node ids redden, and recomputing a statistic
  from `raw.json` all count. **Rereading the researcher's own artefact does not count.** If that was
  your only check, write *"reproduced, not independently verified"*.
- **The code under test is new, so mutate the fixture or harness.** Require one mutation per failure
  mode, plus the most plausible *wrong implementation*.
- **A citation counts as re-measured only when a command re-derives it.** A correction is itself a
  change and needs the same check.
- **Report spread.** An sd of exactly `0.0000` across seeds means the run is broken, not clean.
- **Every number traces to a ledger.** `render_scoreboard.py --audit` exits `0` on every RESULTS.md you
  accept. **#17 and #19 were accepted without that.**
- **Exit codes:** read `$?` directly. In zsh the pipe status is `$pipestatus[1]`; `${PIPESTATUS[0]}`
  expands to the empty string. `3` is not `0`, and `2` is not `3`.
- **Record the torch version** in every new manifest.

---

## Done when

1. Wave 0 is merged, and both RESULTS.md files pass `--audit`. Brief 0 is merged: the battery reads
   N+2/N+2 and the producer is fixed.
2. S0-03 is merged, with live and zeroed answer loss, the gap, 3 seeds and the sd.
3. The E-feas re-run is merged as a new `run_id`, alongside the old one.
4. The decisive run is merged, or refused in writing for an unmet dependency. It states which §10.3
   outcome occurred and what the discriminators say.
5. S0-05 is merged, or parked with a reason.
6. The closing gate is green on main at the final sha.
7. The morning document exists and passes `--audit`.

---

## Stop rules — each predicate points at a source you can check

- **A frozen constant, a pre-registered threshold, or an arm changed to make a gate pass.** This stops
  the night (`rsr-manager.md`, the Frozen-things row).
- **The canary moves.** Mark every result since the last good reading suspect. **Two moves in a row
  escalate and stop the device-free lane too.**
- **A declared dependency is unmet.** Refuse that brief in writing and continue with the others.
- **Two failures with the same cause.** Retire the researcher and do not re-spawn for that brief.
- **`gh pr merge` is refused** by the session's permissions. Leave the PR open with its gates green,
  record the refusal, and stop anything that needs that merge on main.
- **Wall clock.** Start no new brief after **06:30** local. Everything is merged or parked by **07:30**,
  and the morning document is final by **08:00**.

---

## Do NOT

- **Git history:** do not squash, force-push, or **rebase a branch that holds a ledger**. A rebase
  orphans the recorded `git_sha`. Merge `origin/main` into the branch and re-gate before merging.
- **Records:** do not edit an existing `runs/*/` file, `preregistration/`, §15, ADR-0008, or the MATS
  submission.
- **Out-of-scope work:** do not start E1, E0d, E0h, S1-01 or any numbered cycle, and do not tune `γ_b`
  or `ν`.
- **Non-evidence:** do not report KL-from-uniform as evidence about memory. It is non-diagnostic and
  backwards (`RESEARCH-CONTEXT.md` §11).
- **Language:** do not use CLS language for slot memory or eviction (`docs/cognitive-grounding.md` §3).
- **Checkouts:** do not run two agents in one checkout (H3).
- **Spawn prompts:** do not paraphrase a brief into one.

---

## Closing gate — on main, at the final sha, every exit code literal

```
uv run pytest -rs                                        # census line + exit
uv run ruff check src/ tests/ scripts/                   # exit
uv run ruff format --check src/ tests/ scripts/          # exit
uv run python scripts/mutation_battery.py --check        # "N/N gates proven" + exit
uv run python scripts/canary.py <cycle>                  # exit: 0 held, 1 MOVED, 3 first reading
uv run python scripts/render_scoreboard.py --over runs/ --audit docs/lab-notes/overnight-2026-09-21.md
```

---

## Report — the morning document

`docs/lab-notes/overnight-2026-09-21.md`, one file, **concise**. The ledgers hold the detail and the
document points to them by `run_id`.

1. **Headline, three sentences, for the cognitive reader.** Say which of §10.3's two outcomes occurred
   and what it licenses next. State plainly that nothing tonight bears on whether RSR discovers the
   leading-edge strategy, and why that still matters.
2. **Scoreboard,** generated by `render_scoreboard.py --over runs/`, never typed. One row per
   `run_id`: falsifier (or "none — instrument"), expected, observed, verdict, and what you re-executed.
3. **One short section per brief, for the ML reader:** question → method, and why this method →
   controls → numbers (ledger keys only) → which mutation reddened what → **what this does not
   establish.**
4. **Decisions you took, and why.** Every judgment call.
5. **For the owner:** each escalation, one line.
6. **BRIEF ERRORS:** every researcher's field verbatim, then yours about this brief.

Also prepend the standard header block to `.orchestrator/outbox/researcher.md`.
