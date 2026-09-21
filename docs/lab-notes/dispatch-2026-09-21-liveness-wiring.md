# Brief (written 2026-09-21, NOT STARTED): wire memory liveness into every training run

**Status: written, not started.** The overnight dispatch says: *"Write that brief; do not start it. The owner reads the result first."* **Nothing below runs until Brendan has read `runs/decisive-shuffle/` and says go.**
**Written at:** `8ef702474a7712e3c789c83a61161ba808470b4e`. Every anchor must be re-derived at spawn time. `git grep -l memory_liveness -- src/rsr/train/` returned 0 files at this sha.

## Why

`runs/decisive-shuffle/` (verdict `falsified`, meaning memory is live on arms B and C) makes S0-04's trained-live negative control exist (`experiments/shuffle-control/PREREG.md`, *What this does not do*). It also shows that a run can train with a **row-agnostic** memory: in arm A, `armA.train.cosine_trained` has a mean near 1, while random replacement still moves the loss. From the outside, that looks like a working model. At present, liveness is measured only by a separate script after training.

## Falsifier

"A training run whose memory is inert, or whose memory carries no row-specific content, is indistinguishable at exit from one whose memory is live."

## Files in scope (re-derive at spawn)

`src/rsr/train/loop.py` (heartbeat and end of run) · `src/rsr/metrics/memory_liveness.py` (reuse only; **no new implementation**) · `src/rsr/exit_codes.py` (S0-05's enum) · tests.

## Bar

- At the end of every training run, the heartbeat and the run's ledger record the Amendment 1 `ratio`, the cross-row cosine, and the matched-norm random-replacement ratio, plus the three controls.
- Exit codes keep **did not run** (`3`), **ran and inert** (a distinct refusal, to be decided with the owner against `docs/gates.md`), and **ran and live** (`0`) apart. A run whose liveness measurement itself fails is `3`, never `0`.
- Mutations, and the test each must redden:
  - the liveness hook skipped
  - an inert result reported as `0`
  - the decoy pointed at the trained model

## Open for the owner before spawn

Which exit code "ran and inert" gets (`1`, or a new code), and whether an inert run should refuse to write a checkpoint at all.

## Do NOT

Change the Amendment 1 thresholds. Write a third liveness implementation. Start before the owner reads the decisive result.
