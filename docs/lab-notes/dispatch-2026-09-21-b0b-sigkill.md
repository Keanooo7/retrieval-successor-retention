# Brief 0b (2026-09-21): make the SIGKILL checkpoint gate deterministic

**Written at:** `a802521f06be561f0457447f0ae706cd8630d000`. **Worktree:** `.worktrees/b0b-sigkill` on branch `s0/b0b-sigkill` from `origin/main`. One agent, one worktree.
**Opened by:** the Brief 0 researcher. `scripts/mutation_battery.py --check` was **48/49, exit `1`** at `3a458ad`: the row "checkpoints written straight to the final path" (`scripts/mutation_battery.py:759`, gate `tests/test_checkpoint.py::test_a_sigkill_mid_save_never_leaves_a_corrupt_checkpoint`, `:217`) was caught in **1 of 7** observations. The closing gate needs `--check` green, and green-by-timing is not green.

## Falsifier

"The SIGKILL test detects a non-atomic checkpoint write every time." It is refuted by any run in which the battery's mutation (`tmp = path`) leaves that test green.

## Files in scope

`tests/test_checkpoint.py` · `src/rsr/train/checkpoint.py` **only** for a test seam that is inert in production (for example, an optional hook called between the write and the rename, defaulting to a no-op) · `scripts/mutation_battery.py` (that row's `why`, if it changes) · `.orchestrator/outbox/researcher.md`.

## Bar

- The test kills the child **deterministically, inside the window** between the write and the atomic rename. That means a synchronisation point (a pipe, file or event the child signals), **not a wall-clock delay**.
- Under the battery's mutation it reddens **20 of 20** isolated runs. Run the test alone 20 times with the mutation applied by hand, and quote the count. Unmutated, it passes 20 of 20.
- Any production seam is a no-op by default. A test proves that `save()` without the hook produces byte-identical output to before.
- `uv run python scripts/mutation_battery.py --check` exits `0`. Quote the N/N line, and run it **twice**, quoting both.
- `uv run pytest -rs --tb=no`: census line and `$?`. `ruff check` / `ruff format --check` both exit `0`.

## Done when

A PR from `s0/b0b-sigkill` is open with the Bar quoted literally, and your report has been prepended to the outbox with a **`BRIEF ERRORS`** field.

## Do NOT

- Delete the battery row or weaken the gate.
- Add retries, or loop until red.
- Touch `src/rsr/train/loop.py` (S0-03 is live in it), `runs/`, or `preregistration/`.
- Merge.
- Run in the main checkout.
