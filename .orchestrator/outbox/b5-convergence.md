status: RETURNED
run_id: b5-convergence
updated: 2026-09-30T08:31:00-07:00
provenance: a84f78d (run code; PREREG dfda530) · cpu, 3 seeds x 4 threads (cpu-det --slots 12) · stream documents [4160+16*3000, 4160+16*9000), start = fresh-stream B/seed<s>/ckpt-003000.pt copies (T0 sha256 verified) · seeds actually run [0, 1, 2]
manifest: runs/b5-convergence/manifest.json  (config hash 69fc307d24d03aba7bfb080f049f725971d8e6e6feab51d40c82b6c716fcefa3)
falsifier: PLAN-v4 B5 robustness secondary: "arm B's stream answer loss has plateaued by some step <= 9000" (the PREREG plateau rule, one common step); falsified by NO_PLATEAU by 9000. It does not touch the primary substrate (ckpt3000 stays primary; D2 is owner-only).
expected: NO_PLATEAU by 9000 about 55 %; PLATEAU about 40 % (most likely late, E* >= 8000); PLATEAU_WITHOUT_RETRIEVAL about 5 %.
observed: NO_PLATEAU by 9000 (run exit 0, ledger status ok, steps_done 9000 on every seed). The rule failed on every seed at every E from 7000 to 9000. At E = 9000 the relative slope and 95 % CI, % per 1000 steps, are -6.18 [-7.04, -5.32] / -6.09 [-6.77, -5.42] / -5.40 [-5.95, -4.85], against a bound of ±1. The decline is slowing (from about -9..-12 at E = 7000) but is still 5-7x the bound. Retrieval R holds on every seed at every checkpoint, 4000-9000. R at 9000 is 0.8683 / 0.8546 / 0.8573 (live 0.9320 / 0.9141 / 0.9379 against slots_zeroed 0.0637 / 0.0596 / 0.0805), flat at 0.84-0.89 while the stream loss fell from about 0.83 to about 0.49. Fallback applies: ckpt3000 stays primary; D2 is owner-only.

gates:
  T0 start: shasum -a 256 -c MANIFEST.sha256, rc=0 ok=435 notok=0
  tests/test_b5_convergence.py: passed=21 failed=0 skipped=0 errors=0, rc 0
  mutations (5, each alone): PROVEN 5/5, rc 0, baseline passed=21 (one declared coupling)
  full suite (once, at 3d3e38b): passed=1617 failed=0 skipped=0 errors=0, rc 0
  control 6 resume proof: (a) equal=True (b) equal=False (c) equal=True, rc 0
  control 3 measurement path: passed, max_abs_diff [0.0,0.0,0.0], 262 keys per seed
  control 4 stream/closure: ok; control 5 resume exact: True on every seed and segment
  run rc: launch 2 rc 0 (reports/b5/run.rc), job b5-run-20260930T024129; launch 1 rc 143 (stopped by me, see B5.md)
  T0 end: rc=0 ok=435 notok=0
ledger:    runs/b5-convergence/ledger.json (status ok)
numbers:   all in experiments/b5-convergence/RESULTS.md, generated from the ledger. Keys: rule_trace, stream_answer_loss_window_means, B.ckpt<c>.R_quantity / heldout.{live,slots_zeroed}.gap_2_to_M.answer_acc / C_quantity, seed<s>.s_per_step, timing, proof.
BRIEF ERRORS:
  - "Checkpoints are written atomically at the window cadence": train() already writes atomically. The cadence (250) is outside the frozen config, so it does not change the trajectory. Not an error, a confirmation.
  - "Resume must be bit-exact ... prove continuing from ckpt3000 equals the original trajectory where checkable": the original arm B stops at 3000, so continuation past 3000 has no original to compare. The checkable piece is 2900->3000. At 5 threads (the original's) it is exact. At 4 threads (the brief's) it is NOT: 205/206 model tensors differ, max abs diff 7.07e-4. The brief's 12-thread layout therefore makes B5 a numerically distinct continuation of arm B, not "the" continuation. Kill-and-resume at 4 threads is exact (6c).
  - "B1's last mutation proofs may still be going (17 entries)": at 02:12, 15 entries were done and m16 was running; all 17 finished by 02:23, before any B5 compute.
  - PLAN-v4 says "Cap: step 9000 (or later if 16 windows need it)". From 3000, 16 own 250-step windows first exist at 7000, so the clause never binds. This is stated in the PREREG.
  - At the measured 3.45 s/step, 3000->9000 is about 5.75 h. From a 02:41 launch it ends at about 08:30, exactly the session's window end. The brief's "launch by about 03:00" left no margin.
UNANSWERED BY THE BRIEF:
  - Whether a 4-thread (thread-count-distinct) continuation is acceptable as "arm B continued". I ran it, as the brief directs, and state the difference.
  - Whether the full suite must rerun after a B5-only fix. I did not rerun it: the fix at a84f78d touched only B5 files, and the B5 tests and mutations were re-proven.
BELIEVED, NOT VERIFIED:
  - The 4-vs-5-thread divergence is float reduction order, not a code-path difference. The RNG state is equal and (a) is exact, but this was not proven further.
  - The window residuals are independent enough for the OLS CI. The pre-run lag-1 is about 0, and the post-run residuals were not tested. Irrelevant to this outcome, since the point slopes are 5-12x the bound.
  - Launch 1's orphan children (killed at step 3002) appended beats for steps 3000-3002 that launch 2 overwrote by step key. The windows read the last beat per step, and the launch-2 beats come later in the file.
NEXT (proposed, not decided):
  - If the owner wants a plateau checkpoint for D2: a longer continuation (e.g. to 20000), resumable from ckpt-009000 (new PREREG, new run id). The decline slows about 1 %/1000 per 1000 steps, which crudely puts ±1 % beyond 12000-15000. That is a projection, not a measurement.
  - Note for D2: held-out retrieval R is flat (0.84-0.88) from 4000 to 8000 while the training loss keeps falling, so for the retrieval readout ckpt3000 already looks converged.
