# E-feas on the S0-03 corpus: results

**Verdict: `survived`** (ledger `runs/efeas-synthetic-s003/ledger.json`, key `verdict.outcome`).
Both controls held on every seed. **Every figure below is identical, to the last bit, to the
pre-S0-03 run `runs/efeas-synthetic/`**, as the manifest's `expected` field predicted before the
run. That identity is a property of the design, not a finding about the corpus: see *What this run
could and could not show*.

- Pre-registration: `experiments/efeas/PREREG-s003.md` (commit `626b293`), which adopts
  `experiments/efeas/PREREG.md`'s rule and threshold unchanged.
- Command (the caller's exit status, read directly: `rc=0`):

  ```
  uv run python experiments/efeas/run.py --run-id efeas-synthetic-s003 \
      --prereg experiments/efeas/PREREG-s003.md --expect "<the pre-registered expectation>"
  ```

  The full argv, expectation included, is the ledger's `commands[0].argv`.
- Git sha: `a2452196508e6732e034406736d96285c7e4677d` (ledger `provenance.git_sha`).
  `provenance.dirty` is `true`: the only untracked path at run time was `uv.lock`, which no
  code reads.
- Hardware: Mac Studio, `macOS-26.6.2-arm64-arm-64bit-Mach-O`, CPU only (no model is trained).
  Python and torch versions: ledger `provenance.python`, manifest `torch_version`.
- Manifest: `runs/efeas-synthetic-s003/manifest.json`, config hash
  `f1bc3829d10cd9230b570d4d6f66b15a0989963921e1a963ada0fd906688cfb5`. It records
  `synthetic_config_defaults` verbatim and a `corpus_sha256` per seed.
- Seeds actually run: `[0, 1, 2]` (ledger `seeds_actually_run`).

## Changed `SyntheticConfig` defaults

The old manifest does not record the generator defaults, so the comparison is against
`SyntheticConfig` at the old run's recorded sha `1fc8199`:

| field | old run (`1fc8199`) | this run |
|---|---|---|
| `answer_in_stream` | *did not exist* (behaviour equals `False`) | `True` |

No other field changed. `n_documents`, `sentences_per_document`, `max_gap`, `geometric_p`,
`heavy_tail_weight`, `heavy_tail_min` are unchanged (`git diff 1fc8199 HEAD -- src/rsr/data/synthetic.py`).
S0-03 added no generator invariant on the `gap > M` fraction; it added the helper
`fraction_of_pairs_beyond`, which reads gaps and changes none.

## Primary: headroom at M = 16

Ledger key `headroom_oracle_minus_fifo` (`hit(oracle) - hit(FIFO)`):

| seed | this run | old run |
|---|---|---|
| 0 | 0.19491525423728817 | 0.19491525423728817 |
| 1 | 0.17365771812080533 | 0.17365771812080533 |
| 2 | 0.1736227045075125 | 0.1736227045075125 |
| mean ± sd | 0.18073189228853534 ± 0.012283164234645585 | 0.18073189228853534 ± 0.012283164234645585 |

Every seed is at or above the threshold, so the rule returns `survived`.

## Secondary rows (not judged)

| key | this run: samples | mean ± sd | old run mean |
|---|---|---|---|
| `secondary.M32.headroom` | 0.041525423728813515, 0.03523489932885904, 0.024207011686143587 | 0.033655778247938715 ± 0.008766531039189842 | 0.033655778247938715 |
| `secondary.M8.headroom` | 0.2923728813559322, 0.25838926174496635, 0.26544240400667785 | 0.2720681823691921 ± 0.017934528280598227 | 0.2720681823691921 |

At M = 32 every seed is below the threshold. That is the same as before and it is not judged.

## Hit rates at M = 16

| arm (ledger key) | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|
| `hit_rate.fifo` | 0.8050847457627118 | 0.8263422818791947 | 0.8263772954924875 | 0.8192681077114647 ± 0.012283164234645585 |
| `hit_rate.oracle` | 1.0 | 1.0 | 1.0 | 1.0 ± 0.0 |
| `hit_rate.random` | 0.7898305084745763 | 0.8120805369127517 | 0.8055091819699499 | 0.8024734091190927 ± 0.011431442558855202 |

`hit_rate.oracle` has sd exactly `0.0` (`sd_exactly_zero: true`). Here that is not a broken seed
loop: an oracle with the true future demand retrieves every query at M = 16 on each of three
corpora with different gaps (`seed{s}.n_queries` are 1180, 1192 and 1198). The FIFO and random
arms vary across the same seeds.

## Controls, per seed

| seed | FIFO hit iff `gap <= M` | oracle >= FIFO on every document | ledger key |
|---|---|---|---|
| 0 | held | held | `seed0.controls_failed = []` |
| 1 | held | held | `seed1.controls_failed = []` |
| 2 | held | held | `seed2.controls_failed = []` |

## What this run could and could not show

S0-03's only generator change appends the answer token to the query text. It draws nothing from
the RNG, so facts, gaps and query positions are unchanged, and `rsr.metrics.headroom.simulate`
reads only positions and gaps, never text. So E-feas could not have come out differently: the
identity with the old run is a consistency check, confirming that nothing besides the appended
token changed. It is not new evidence about headroom.

What S0-03 *did* change, that the text is different, is confirmed separately. The manifest's
`corpus_sha256` hashes the S0-03 corpus. For each seed, `generate(SyntheticConfig(seed=s))` and
`generate(SyntheticConfig(seed=s, answer_in_stream=False))` have equal `gaps` and differ in text.
That check was run at the command line and is not in a ledger.

Every row of the new ledger has the same key set and values as `runs/efeas-synthetic/ledger.json`,
`by_gap` included.
