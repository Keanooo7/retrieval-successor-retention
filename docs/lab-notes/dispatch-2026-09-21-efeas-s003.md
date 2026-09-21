# Brief 2 (2026-09-21): the E-feas re-run on the S0-03 corpus

**Written at:** `5e6bfe5268047ddd4adc7466fa9b4141c582f81a`, before S0-03 was spawned. **Depends on:** S0-03 merged to `main`. The manager spawns this only after that merge.
**Worktree:** `.worktrees/efeas-s003` on branch `s0/efeas-s003`, from `origin/main` after S0-03 has merged.
**Pre-registration:** `experiments/efeas/PREREG-s003.md`, committed ahead of this run. It adopts `experiments/efeas/PREREG.md`'s rule unchanged.

## Falsifier

"The synthetic corpus, as S0-03 leaves it, has headroom for a retention rule over FIFO at `M = 16`." Rule: `experiments/efeas/PREREG.md`, *Decision rule*.

## Files in scope

- `runs/efeas-synthetic-s003/` (new).
- `experiments/efeas/RESULTS-s003.md` (new).
- `.orchestrator/outbox/researcher.md`.
- `experiments/efeas/run.py` **only** if it cannot run on the new corpus unchanged. If you change it, say why in `BRIEF ERRORS`, and keep `runs/efeas-synthetic/` reproducible from its recorded sha.

## Bar

- Run `uv run python experiments/efeas/run.py --run-id efeas-synthetic-s003` and read `$?` directly.
- Report headroom at `M = 16` (with the sd across seeds) and at `M = 32`, beside the old run's `headroom_oracle_minus_fifo` and `secondary.M32.headroom`. Every number is a ledger key.
- Report both controls per seed, and the verdict.
- Name every `SyntheticConfig` field whose default differs from the old run's manifest.
- The manifest records the torch version.
- `render_scoreboard.py --over runs/ --audit experiments/efeas/RESULTS-s003.md` exits `0`.
- `git diff --stat origin/main -- runs/` shows only `runs/efeas-synthetic-s003/`.
- `uv run pytest -rs --tb=no`: census line and `$?`.

## Done when

A PR from `s0/efeas-s003` is open with every Bar item quoted literally, and your report has been prepended to `.orchestrator/outbox/researcher.md` with a **`BRIEF ERRORS`** field (write `none` out if there are none).

## Do NOT

- Touch `runs/efeas-synthetic/` or `experiments/efeas/PREREG.md`.
- Change the `0.05` threshold, `M`, or the arms.
- Train a model.
- Merge.
- Run in the main checkout.
