# Decisive run (2026-09-21): is the working memory live on the rewardable corpus?

**Written at:** `b6db595e0543d0a31221ee92642043218decd25e` (`git rev-parse HEAD` at the moment of writing). This is **before S0-03 was spawned**, so no S0-03 number existed when it was written.
**Depends on:** S0-03 merged to `main`, and a canary reading after it. **Do not start until the manager's spawn message arrives; the manager spawns only after both.**
**Worktree:** `.worktrees/decisive` on branch `s0/decisive`, from `origin/main` **after** S0-03 has merged.
**Researcher:** fresh. It must not have seen any S0-03 number, and it must not read `runs/s0-03-*` or S0-03's RESULTS.md.
**Pre-registration:** `experiments/decisive-shuffle/PREREG.md`, committed with this brief and ahead of any run. **It is the contract: condition, arms, decision rule and secondary readouts.** Read it first. Do not edit it; append a dated amendment if you must, naming what prompted it.

## Falsifier

"The TG working memory is inert on a corpus whose objective requires retrieval." Decision rule: #17's Amendment 1, unchanged (`experiments/shuffle-control/PREREG.md`, *Amendment 1*), applied per arm as in the PREREG.

## Files in scope

- `experiments/decisive-shuffle/run.py` (new). It follows `experiments/shuffle-control/run.py` and changes only the arm settings, the corpus, and the added readouts.
- `experiments/decisive-shuffle/RESULTS.md` (new).
- `tests/test_decisive_shuffle.py` (new).
- `src/rsr/metrics/memory_liveness.py`: **additive only**, for the random-replacement discriminator and an answer-token mask argument. Every existing test and battery entry must stay green and unchanged.
- `tests/test_shuffle_control.py`: additive only.
- `scripts/mutation_battery.py`: entries for your new gates.
- `runs/decisive-shuffle/` (new).
- `.orchestrator/outbox/researcher.md`.

## Bar

1. **Primary:** `ratio` per seed per arm, with mean ± sd per arm and `sd_exactly_zero`. Every figure is a ledger key with `how`, and there is a per-arm verdict row plus an overall verdict naming which §10.3 outcome occurred (PREREG *Which outcome occurred*).
2. **Controls on every seed of every arm:** memory-disabled `== 0.0`, own-memory `== 0.0`, decoy `> 0`, and random-replacement-with-self `== 0.0`.
3. **Secondary:** the answer-token split (`gap > M` / `gap < M`), the cross-row cosine, and the matched-norm random replacement, all as the PREREG defines them.
4. **Mutation bar (PREREG):**
   - An arm-config swap must be refused by the manifest-hash check.
   - Pointing the decoy at the trained checkpoint must give `ratio == 1.0` and `inconclusive`.
   - Each of these must redden a named test. Quote the node ids.
   - `mutation_battery.py --check` exits `0`; quote the N/N line.
5. **Gates:**
   - `uv run pytest -rs --tb=no`: census line and `$?`.
   - `ruff check` / `ruff format --check` over `src/ tests/ scripts/` both exit `0`.
   - `render_scoreboard.py --over runs/ --audit experiments/decisive-shuffle/RESULTS.md` exits `0`.
6. **Provenance:**
   - The manifest records the **torch version** and the torch thread count per process, and it is frozen before the first seed.
   - `git diff --stat origin/main -- runs/` shows only `runs/decisive-shuffle/`.
   - Checkpoints stay untracked. Do not commit `.pt` files.
   - **CPU only.**

**Wall clock.** 9 trainings, each at #17's config; #17 took ~15 min per seed on this machine's CPU. You may run seeds or arms as **parallel subprocesses** with a fixed `torch.set_num_threads` per process, which you record. **Stop starting new trainings at 06:30 local.** A partial run is reported as partial: `seeds_actually_run` and `steps_done` must say so, and a verdict on fewer than 3 seeds is written `inconclusive (partial)`, never rounded.

## Done when

A PR from `s0/decisive` is open with every Bar item quoted literally, and your report has been prepended to `.orchestrator/outbox/researcher.md` with a **`BRIEF ERRORS`** field (write `none` out if there are none).

## Do NOT

- Choose, move, or reinterpret a threshold. Change an arm to make a verdict come out.
- Report KL-from-uniform as evidence about memory (`RESEARCH-CONTEXT.md` §11).
- Use CLS language for slot memory.
- Read S0-03's numbers.
- Wire liveness into `src/rsr/train/`. That is a later brief, written only if the memory is live.
- Merge. The manager verifies and merges.
- Read `$?` after a pipe.
- Run in the main checkout.
