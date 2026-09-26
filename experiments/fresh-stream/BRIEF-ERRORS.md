### Pre-run, found while implementing `run.py` (2026-09-25, before any training or measurement)

Written by the implementing session (a model). The PREREG was found implementable as
written; nothing in it was worked around, and it was not edited. These are the places
where the implementation had to choose something the PREREG does not spell out.

- **The PREREG classifies; it names no falsifier.** As in scaffold-timing, the outcome is
  a classification (DATA_SUFFICES / SCAFFOLD / TRAP / NEITHER / inconclusive), and
  `scripts/ledger.py`'s `Ledger.verdict` takes only `survived` / `falsified` /
  `inconclusive` with a `falsifier` string. `run.py` does **not** call `Ledger.verdict`:
  the ledger's `verdict` is null and the result is in the rows `classification`,
  `classification_detail`, `A.R3000`, `B.R3000`, `A.usable` and `B.usable`.
- **How `control_2` (resume is exact) is checked "in the run".** `train()` gains only the stream argument
  (PREREG: the only change to the training code), so it has no hook between the load and
  the first step. Inside each arm-B child, `ResumeCheck` wraps
  `rsr.train.checkpoint.load` (to capture the model and optimizer it restores into) and
  `rsr.train.loop.run_policy_loop` (to compare at its first call, after the load and
  before the first forward and optimizer step). The comparison reads `ckpt-001000.pt`
  again from disk and is exact (`torch.equal`, dtype and shape; every non-tensor field by
  equality). The result is written to `runs/fresh-stream/B/seed<s>/resume_check.json` at
  once and copied into the ledger (`control_2.*`); a mismatch stops that child.
- **Arm A's call is the `n64` arm's call, not a retyping of it.** `single()` calls the
  corpus-size curve's own `single()` for its control arm with `train` bound to
  `functools.partial(train, stream=STREAM)` (plus `resume=` for arm B). Every other
  argument is therefore the `n64` arm's. `tests/test_fresh_stream.py` checks the two
  calls differ by exactly the stream, and that the parameters and the global RNG state at
  the first step are identical with and without the stream.
- **`control_1` (default path unchanged) trains through the corpus-size curve's child**,
  imported: its ledger commands name `experiments/corpus-size-curve/run.py --single` (the
  manifest's `control_1` gives the whole command), which calls `train()` without the
  stream. The parent measures at scaffold-timing's thread count
  (its manifest's `torch_threads`), because that is how the reference rows were produced.
  Whether another thread count would reproduce them was not measured.
- **The imported `measure_checkpoint` measures a little more than the PREREG names.** At a
  checkpoint with that curve's control label (`A.ckpt300`, arm A's first) it also runs S0-03's `measure()` on S0-03's own
  document sets. That table is kept in `raw.json` only; no rule and no ledger row reads it.
- **`stream_loss_threshold` is `chance_ln16` minus the PREREG's margin, exactly.** The PREREG writes the
  rounded natural log of the answer-symbol count; `run.py` uses S0-03's `CHANCE`,
  imported. The two differ below the PREREG's printed precision.
- **Arm B's start checkpoints are refused unless they are the ones scaffold-timing
  measured.** Before the manifest, each `ckpt-001000.pt` must exist, store its step, and
  have the sha256 recorded in `runs/scaffold-timing/manifest.json`. The three hashes are
  frozen in this run's manifest (`arm_B_start_checkpoint_sha256`).
- **The data-order generator is not in a checkpoint** (a property of `train()` before this
  change: `torch.randint` draws from a local generator that `ck.save` does not capture).
  It does not matter here: in stream mode nothing is drawn from it.
- **`run.sh` and `run.rc`.** The parent refuses a run directory that exists, so its log goes
  to `runs/fresh-stream.parent.log`, and `run.sh` writes `runs/fresh-stream/run.rc` after
  the parent exits. If the parent refused before creating the directory, `run.sh` still
  creates it to hold `run.rc`, and it must be moved aside before a retry.

## Post-run (written 2026-09-25 ~17:55 PDT by the session that wrote the PREREG, after reading the ledger)

- The author's expectation was wrong in the favourable direction for arm B: it predicted R would survive without growing and C3000(B) to fail more often than not. The ledger has `B.growth_R_quantity` and `B.C3000` true; see those keys.
- Leak check read before this note: in the held-out `gap_gt_M` bucket, which FIFO has evicted, arm B's live accuracy stays near chance (`B.ckpt3000.heldout.live.gap_gt_M.answer_acc`), and `slots_zeroed` returns every bucket to chance (`B.ckpt3000.heldout.slots_zeroed.gap_2_to_M.answer_acc`).
- Arm A never left the plateau: `A.stream_answer_loss_first_window_below` is null on every seed.
- The author saw the arm-B heartbeats (stream answer loss) shortly before the run ended, before reading any measured key.
