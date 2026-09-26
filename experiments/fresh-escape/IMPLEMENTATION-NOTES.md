# fresh-escape — implementation notes (pre-run)

Written before the run, from the adversarial review of `run.py`. These are notes on
how the implementation reads the PREREG (`PREREG.md`, d5b9a23); they are not brief
errors (`BRIEF-ERRORS.md` is the post-run audit's, and is written after the run).

- **The deadline is applied literally.** P is the largest listed checkpoint whose
  measurement *completed* at or before 2026-09-26T08:30:00-07:00 on every seed.
  Every checkpoint measurement goes through `deadline_stamped`, applied at the
  call site in `_run`. It stamps `measured_at` (ISO, tz-aware, completion time),
  and it does not start a measurement after the deadline; such a record reads
  "not measured before deadline". A measurement that completes after the deadline
  stays in the ledger (`post_deadline_measurements`, `post_deadline: true`) and
  never feeds P, the primary `A.ckpt<c>.*` rows or the classification. The parent
  log lines are timestamped.
- **STIRRING reads windows from steps ≥ 3000 only**, from this run's heartbeats.
  That is equivalent to reading fresh-stream's windows too, because the reference
  ledger's `A.stream_answer_loss_first_window_below` is `None` on every seed
  (`runs/fresh-stream/ledger.json`). A window `[w0, w0 + 100)` counts as "at or
  before P" only if `w0 + 100 <= P`, because ckpt-P holds steps `[0, P)`.
- **`run.sh` writes `run.rc` even after a refusal.** A parent that refuses (exit
  3: dirty source, a start checkpoint refused, a deadline already passed) still
  leaves `runs/fresh-escape/run.rc` holding `3`. The one exception is
  `runs/fresh-escape` already existing: then `run.sh` refuses without touching it.
- **`run_arm` (fresh-stream's, imported) stops every seed if one exits early.**
  The surviving seeds are terminated, so a single crashed child ends training for
  all three. P then reflects what was measured by that point, and the P ≥ 6000 rule
  applies.
- **Thread oversubscription costs speed only.** The children use 3 × 5 threads,
  the measuring parent uses 12 (fresh-stream's `parent_torch_num_threads`), and
  the mutation battery runs alongside. Contention slows wall time and so can lower
  P at the deadline. It does not change any computed number: control 1 re-measured
  all 262 `A.ckpt3000.*` keys at 12 threads with max abs diff 0.0 on every seed.
