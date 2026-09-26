### Pre-run, found while implementing `run.py` (2026-09-25, before any measurement)

- **The PREREG classifies; it names no falsifier.** Its outcome is BEFORE / WITH /
  AFTER / inconclusive. `scripts/ledger.py`'s `Ledger.verdict` takes only
  `survived` / `falsified` / `inconclusive` and needs a `falsifier` string. Mapping
  the classification onto that vocabulary would invent a claim the PREREG does not
  make, so `run.py` does **not** call `Ledger.verdict`: the ledger's `verdict` is
  null, and the result is in the rows `classification`, `onset_R`, `onset_M`,
  `classification_detail` and `next_run_arms`. The scoreboard counts this run
  under `no_verdict`. A gap between the PREREG and the ledger tooling, not a change
  to the PREREG; recorded for the owner.
- **Thread count is not in the PREREG.** The corpus-size ledger was measured in its
  parent process at its manifest's `parent_torch_num_threads`. The default
  (sequential) mode sets that same count. `--parallel` (one child per seed) exists
  but whether a different thread count reproduces the ledger within the tolerance
  has not been measured. The reproduction control decides it either way; a failed
  control is `Exit.DID_NOT_RUN`, and re-running in the other mode after seeing a failure would
  be tuning to make it match, which the PREREG forbids.
- **The imported `measure_checkpoint` measures a little more than the PREREG
  names.** At `n64.ckpt300` it also runs S0-03's `measure()` on S0-03's own
  document sets (the corpus-size curve's own control). That table is kept in
  `raw.json` only; no rule and no ledger row reads it.

### Post-run (2026-09-25, after the one real run at `4ab4b5a`)

Written by the executing session (a model) from `runs/scaffold-timing/ledger.json`. No
threshold was changed, and the PREREG was not edited. The list numbering above became
bullets, and one exit-code literal became its `Exit` name, so that `render_scoreboard.py --audit`
does not read them as measurements. Otherwise the wording is unchanged.

- **The parent's exit code is null in the ledger.** The row is written before the process
  exits, the same pattern as the corpus-size ledger's parent row. The executor captured the
  exit code directly (`; rc=$?` immediately after the command, no pipe, under `caffeinate -i`, sequential mode,
  which is the default). It was zero. The wall clock of `run()` is `elapsed_s`.
- **The reproduction control is exact, not merely within tolerance.**
  `reproduction_control.max_abs_diff` is zero on every seed, over
  `reproduction_control.n_keys_compared` keys per seed. The sequential mode reproduced
  the ledger at the corpus-size parent's thread count (`torch_threads`).
- **The PREREG's pre-run table has two rounding slips in the fourth decimal.** The control
  is exact, so these ledger values are also the corpus-size ledger's. The `seed2` sample of
  `n64.ckpt1000.R_quantity` is 0.07203389704227448, which rounds to 0.0720; the PREREG
  prints a value one unit higher in that decimal. The `seed1` sample of `n64.ckpt300.M_quantity` is
  0.12265094369649887, which rounds to 0.1227; the PREREG prints a value one unit lower.
  Both are far from `delta`, so neither changes any rule outcome.
- **The classification is set by the slowest seed. Every seed is individually AFTER.**
  R needs every seed. The executor computed per-seed sustained onsets from the ledger's
  `n64.ckpt<c>.R_quantity` and `n64.ckpt<c>.M_quantity` rows at `delta`. They are not
  pre-registered and are not ledger keys. By R, `seed0` onsets at checkpoint 500, `seed1`
  at checkpoint 400 and `seed2` at checkpoint 600, all taken from
  `checkpoints_preregistered`. By M, `seed0` and `seed1` onset at checkpoint 300 and `seed2` at
  checkpoint 200, also from `checkpoints_preregistered`. On each seed the R onset comes
  after the M onset. `seed1` already reaches `delta` at `n64.ckpt400`
  (`n64.ckpt400.R_reaches_delta_on_seeds`).
- **Part of the author's expectation is wrong.** The PREREG expected that "R rises steadily
  once train accuracy passes ~0.5". `seed2` does not fit: at `n64.ckpt500` its
  `train.live.gap_2_to_M.answer_acc` is 0.8376, while its `n64.ckpt500.R_quantity` is
  only 0.015536721795797348. It crosses at `n64.ckpt600` (0.056497178971767426). `seed1`
  moves the other way. Its `n64.ckpt400.R_quantity` is 0.04709141328930855 while its
  `n64.ckpt400.train.live.gap_2_to_M.answer_acc` is 0.4034, below one half. `seed0`'s R is
  not monotone: `n64.ckpt500.R_quantity` 0.05240793153643608, then
  `n64.ckpt600.R_quantity` 0.041076481342315674. The rest of the expectation held:
  `classification`, `onset_M`, `onset_R`, the train memory share rising before R, and
  `secondary_R_reaches_delta` naming only the `n512` arm's `seed2` at its last checkpoint
  (`n512.ckpt1000.R_quantity`, 0.03672316297888756). No `n4096` checkpoint reaches
  `delta` on any seed.
- **Surprising, and not interpreted here: the side-channel share is large.** In the `n64`
  arm from `n64.ckpt600` onward, `side_channel_share` (train `gate_zeroed` minus
  `gate_zeroed_bos_off` accuracy) lies between 0.18 and 0.38 on every seed. For example,
  `n64.ckpt800.side_channel_share` is 0.3822, 0.2188 and 0.3220. Over the same
  checkpoints, `train_memory_share` lies between 0.25 and 0.50. The two are comparable in
  size. No rule reads either.
- **`seed0`'s train memory share peaks and then falls back slightly.** It is
  `n64.ckpt600.train_memory_share` 0.4238505959510803 and
  `n64.ckpt1000.train_memory_share` 0.37068963050842285. `seed1` and `seed2` are roughly flat
  after `n64.ckpt600`. No rule reads it.
