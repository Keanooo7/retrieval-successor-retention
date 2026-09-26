Pre-run implementation notes, written by the implementing session, were moved verbatim to IMPLEMENTATION-NOTES.md in this directory so that the audited RESULTS page carries only post-run text.

## Post-run (written by the session that wrote the PREREG, after reading the ledger)

- The author's expectation named the next higher dose than the ledger's `k_star`. The expectation was wrong, in the direction of less memorisation being needed.
- The unlock is graded, not a switch: compare the `R_quantity` keys at each arm's end checkpoint in the table below. The lowest unlocking dose retrieves weakly; larger doses retrieve more after the same number of stream steps.
- The dose below k* moved on one seed only (the `k200` arm, on a single seed), which is why U fails there.
- Evicted-bucket check read before this note: `gap_gt_M` live accuracy at each arm's end checkpoint was looked at for every seed; no arm shows an excess as large as the one fresh-stream's arm B showed on one seed, though some seeds sit modestly above chance. Stated qualitatively; the keys are the end-checkpoint `heldout.live.gap_gt_M.answer_acc` key of each arm.
