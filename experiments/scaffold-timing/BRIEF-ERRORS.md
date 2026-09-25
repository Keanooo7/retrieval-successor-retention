### Pre-run, found while implementing `run.py` (2026-09-25, before any measurement)

1. **The PREREG classifies; it names no falsifier.** Its outcome is BEFORE / WITH /
   AFTER / inconclusive. `scripts/ledger.py`'s `Ledger.verdict` takes only
   `survived` / `falsified` / `inconclusive` and needs a `falsifier` string. Mapping
   the classification onto that vocabulary would invent a claim the PREREG does not
   make, so `run.py` does **not** call `Ledger.verdict`: the ledger's `verdict` is
   null, and the result is in the rows `classification`, `onset_R`, `onset_M`,
   `classification_detail` and `next_run_arms`. The scoreboard counts this run
   under `no_verdict`. A gap between the PREREG and the ledger tooling, not a change
   to the PREREG; recorded for the owner.
2. **Thread count is not in the PREREG.** The corpus-size ledger was measured in its
   parent process at its manifest's `parent_torch_num_threads`. The default
   (sequential) mode sets that same count. `--parallel` (one child per seed) exists
   but whether a different thread count reproduces the ledger within the tolerance
   has not been measured. The reproduction control decides it either way; a failed
   control is exit 3, and re-running in the other mode after seeing a failure would
   be tuning to make it match, which the PREREG forbids.
3. **The imported `measure_checkpoint` measures a little more than the PREREG
   names.** At `n64.ckpt300` it also runs S0-03's `measure()` on S0-03's own
   document sets (the corpus-size curve's own control). That table is kept in
   `raw.json` only; no rule and no ledger row reads it.
