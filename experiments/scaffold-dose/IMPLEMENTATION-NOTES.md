### Pre-run, found while implementing `run.py` (2026-09-25, before any training or measurement)

Written by the implementing session (a model). The PREREG (4249567) was found implementable
as written; nothing in it was worked around, and it was not edited. These are the places
where the implementation had to choose something the PREREG does not spell out.

- **The PREREG classifies; it names no falsifier.** As in scaffold-timing and fresh-stream,
  `run.py` does **not** call `Ledger.verdict`: the ledger's `verdict` is null and the result
  is in the rows `classification`, `classification_detail`, `k_star`, `U`, `monotone`,
  `monotonicity_violations` and `<arm>.U`.
- **The PREREG numbers its controls differently from fresh-stream.** Here control 1 is
  "resume is exact" (fresh-stream's control 2, imported: `FS.ResumeCheck` /
  `FS.compare_to_checkpoint`), control 2 is "measurement path unchanged", control 3 is
  disjointness + closure. The ledger keys follow THIS PREREG: `<arm>.control_1.*` and
  `control_2.*`.
- **Order.** The brief says control 2 FIRST; the PREREG says "before the arms". `run.py`
  runs control 2, then control 3 (disjointness + closure), then the arms. Before either,
  and before the manifest is frozen, every start checkpoint (5 k x 3 seeds) must exist,
  store step k, and have the sha256 recorded in `runs/scaffold-timing/manifest.json`; any
  problem refuses the run (exit 3) and names every offending file. `--dry-run` runs the
  same check read-only and printed "every sha256 == runs/scaffold-timing/manifest.json,
  every step == k" on 2026-09-25.
- **How fresh-stream's arm-B child is reused for k != 1000.** `single(k, ...)` calls
  fresh-stream's own `single("B", seed, out_dir, iters=k + 500)` with fresh-stream's
  module constant `RESUME_STEP` set to k for the duration of the call
  (`fresh_stream_with(RESUME_STEP=k)`, restored after; a name that does not exist there
  raises). Every other argument is fresh-stream's arm B's, i.e. the corpus-size N = 64
  call plus `stream=` and `resume=`, inside its `ResumeCheck`. The same mechanism with
  `CHECKPOINTS={"k<k>": (k + 500,)}` lets fresh-stream's `arm_rows` write this run's
  per-arm ledger rows, so the naming and derived quantities are fresh-stream's.
  `tests/test_scaffold_dose.py::test_single_is_fresh_streams_arm_B_from_ckpt_k` checks
  that the child's `train()` call is the N = 64 call plus exactly the stream and the
  resume from `ckpt-{k:06d}.pt`, and that `RESUME_STEP` is restored.
- **The vocabulary closure covers a superset of the ids the arms use.** Control 3's
  disjointness is checked per arm, over exactly `[4160 + 16 k, 4160 + 16 (k + 500))`.
  The closure is fresh-stream's `vocabulary_closure(seed, iters)`, imported, which takes
  the ids of steps `[0, iters)`; with `iters = max(k) + 500 = 1100` it covers
  `[4160, 21760)`, which contains every arm's ids and also the ids of steps 0-99
  (`[4160, 5760)`), which no arm trains on. A word outside the map in those extra
  documents would refuse the run although no arm uses them; fresh-stream's closure over
  `[4160, 52160)` passed on all three seeds, so this is not expected to bite.
- **A raised measurement stops every later arm** (fresh-stream's `run_all` rule). The
  PREREG makes that arm's cell unreadable and the result inconclusive; the later arms are
  recorded under `arms_not_run` rather than trained into an already-inconclusive result.
  Other unreadable causes (a failed or missing resume check, a missing end checkpoint)
  do not stop later arms.
- **U(k) of an unreadable arm is null**, not false, and takes no part in k* or the
  monotonicity note; its R and C values are still in the ledger
  (`<arm>.ckpt<k+500>.R_quantity`).
- **Monotonicity is every pair**, not only adjacent arms: `monotonicity_violations` lists
  every `(k1, k2)` with `k1 < k2`, U(k1) and not U(k2).
- **Control 2's thread count.** The parent measures at scaffold-timing's thread count (its
  manifest's `torch_threads`), as fresh-stream did, because that is how the reference rows
  were produced; the parent keeps that count while the children train.
- **The imported `measure_checkpoint` also runs S0-03's default sets at corpus-size's control
  label 300.** No label this run measures is 300 (it measures 600 for control 2 and
  600-1100 as end checkpoints), so that extra table never appears.
- **`run.sh` and `run.rc`** as fresh-stream's: the parent refuses a run directory that exists,
  so its log goes to `runs/scaffold-dose.parent.log`, and `run.sh` writes
  `runs/scaffold-dose/run.rc` after the parent exits.

