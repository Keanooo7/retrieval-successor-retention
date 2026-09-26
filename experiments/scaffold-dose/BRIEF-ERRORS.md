Pre-run implementation notes, written by the implementing session, were moved verbatim to IMPLEMENTATION-NOTES.md in this directory so that the audited RESULTS page carries only post-run text.

## Post-run (written by the session that wrote the PREREG, after reading the ledger)

- The author's expectation named the next higher dose than the ledger's `k_star`. The expectation was wrong, in the direction of less memorisation being needed.
- The unlock is graded, not a switch: compare the `R_quantity` keys at each arm's end checkpoint in the table below. The lowest unlocking dose retrieves weakly; larger doses retrieve more after the same number of stream steps.
- The dose below k* moved on one seed only (the `k200` arm, on a single seed), which is why U fails there.
- Evicted-bucket check read before this note: `gap_gt_M` live accuracy at each arm's end checkpoint was looked at for every seed; no arm shows an excess as large as the one fresh-stream's arm B showed on one seed, though some seeds sit modestly above chance. Stated qualitatively; the keys are the end-checkpoint `heldout.live.gap_gt_M.answer_acc` key of each arm.

## Corrections after the independent verifier's report (verdict CONFIRMED on every number)

- The second post-run bullet says "in the table below"; there is no such table. The per-arm values are in the per-arm sections above it.
- "Larger doses retrieve more" holds only as: every larger dose retrieves more than the lowest unlocking dose on every seed. It is not a strict per-seed dose-response: on one seed the `k400` arm's end `R_quantity` is slightly above the `k600` arm's. Only the per-arm means increase with dose.
- `manifest.json` carries an `iters` field inherited from S0-03's CONFIG that matches nothing in this run; the children's heartbeat headers record the iterations actually run. No number is affected.
- The control-2 re-measured values are not stored in `raw.json` (only its summary), so that control can be re-checked only by re-measuring; the verifier did so for one seed and matched exactly.
