# E0E - Distribution of `u_bar` on a FIFO run

**Status: RUN, 2026-09-26. Values measured; NOT recorded in the registry** (owner
decision D2: which substrate E1's constants come from). Pre-registration:
`experiments/e0e/PREREG.md`, committed alone in `cf7a573`, ahead of the code
(`fdb2411`). Every number below is a key of `runs/e0e/ledger.json`, the ledger of
run `e0e`.

| | |
|---|---|
| Prediction | Sets `tau`. **Also produces `E[lifetime]` -> `gamma_b` and the EMA half-life** (spec-corrections §5). The registry refuses `gamma_b` until this lands. |
| Kill gate? | No |
| Sprint | ROADMAP's Sprint two |

## Reproduction

| | |
|---|---|
| Command | `uv run python experiments/e0e/run.py` (run `e0e`, its `commands` row, with the exit code the process returned: OK) |
| Git SHA | run `e0e` `provenance.git_sha` `fdb241167cf283e32acbf1ec44261198632dd4e1`, `provenance.dirty` false |
| Hardware | run `e0e` `provenance.platform` `macOS-26.6.2-arm64-arm-64bit` (the Mac Studio), `device` cpu, one thread |
| Seeds | run `e0e` `seeds_actually_run` [0, 1, 2]; `live_seeds` [0, 1, 2] |
| Config | `runs/e0e/manifest.json`, run `e0e` `config_hash` `4e3e2c7025e53427406a811adf233b5c6cc4bc2568f6be1a2cdfe6d7bdc1f7cb` |

## What was run (decided in the PREREG, not here)

FIFO eviction, eval mode (spec-corrections §20), over the held-out documents the PREREG
names, one stream at a time, on fresh-stream arm B's frozen final checkpoints (read
only, sha256-checked against the PREREG). `r_i` is §3.2.1's gated, fill-rescaled
target (`rsr.retention.reward.retrieval_demand`). No training.

**Liveness precondition** (`rsr.metrics.memory_liveness.measure_liveness`, the decisive
PREREG's bands): every seed is `live`. The `seed0.liveness`, `seed1.liveness` and
`seed2.liveness` ratios are 55.564, 98.968 and 96.840, with decoy `A` 0.010212,
0.005432 and 0.006189.

## Result

| key (run `e0e`) | value |
|---|---|
| `E_lifetime` (pooled, all written sentences) | 13.1667 |
| `per_seed.E_lifetime` mean ± sd | 13.1667 ± 0.0 (`sd_exactly_zero`: analytic, pre-registered) |
| `evicted_only_mean` | 16.0 |
| `write_to_boundary_mean` | 13.5 |
| `n_written` / `n_evicted` | 9216 / 6144 |
| `half_life` (a quarter of `E_lifetime`, §3.5) | 3.2917 steps |
| `ema_alpha` | 0.18988 |
| **`tau`** (q0.75 of the relative deviation of `M·ū` from the band centre, pooled) | **0.27555** |
| `per_seed.tau` samples | 0.24097, 0.27573, 0.31805 (mean 0.27825, sd 0.03860) |
| `fraction_inside_tau` | 0.75 (by construction of the rule) |
| `fraction_inside_v04_025` (the ±0.25 band) | 0.70403 |
| `tau_at_50pct_firing` | 0.15896 |
| `tau_init_first_obs` (ū started at its first observation) | 0.39788 |
| `tau_full_memory_only` | 0.29115 |
| `n_obs` / `n_obs_full_memory` | 121344 / 98304 |
| `ubar_mean` / `ubar_min` / `ubar_max` | 0.062949 / 0.024222 / 0.18433 |
| `M_ubar_quantiles` (q0.01, q0.5, q0.99 of `M·ū`) | 0.53216, 0.97910, 1.76478 |
| `b_max` (registry, FROZEN) | 1.0 |
| **`gamma_b`** = `b_max / (0.25 · E_lifetime)` | **0.30380** |
| `consistency_problems` | [] |

## Against the pre-registered expectation

- **Right:** `E_lifetime` is 13.1667 on every seed. The sd across seeds is exactly zero,
  as the PREREG predicted: under FIFO with every sentence written, lifetimes are fixed
  by `S` and `M`. The `half_life` is 3.2917 and `gamma_b` is 0.30380. Every seed is
  `live`.
- **Wrong:** the PREREG predicted a heavy-tailed `ū`, with less than half of the
  observations inside ±0.25. The measured `fraction_inside_v04_025` is **0.70403**.
  §3.5's worry that `ū` "may essentially never sit inside" the ±0.25 band **does not
  hold** on this substrate at this half-life. The companion prediction, `tau` above
  0.25, held narrowly: `tau` is 0.27555, but the first seed's is 0.24097.
- **Outside the spec's stated range:** `gamma_b` = 0.30380 against §3.5's "order
  0.05–0.1". This is RESEARCH-CONTEXT §12's second item, the scoping problem, seen at the
  synthetic scope. It is not resolved here.

## The `record()` calls this run did NOT make

The run writes them to its ledger row `would_be_record_calls_NOT_MADE` and executes
none of them. `measurements/ledger.json` is untouched.

- Run `e0e` `would_be_record_calls_NOT_MADE`: `rsr.constants.record("E_lifetime", 13.166666666666666, experiment="E0e", run_id="e0e", scope="synthetic", note="FIFO, fresh-stream arm B ckpt 3000, held-out docs 64..127, live seeds")`
- Run `e0e` `would_be_record_calls_NOT_MADE`, its second entry: `rsr.constants.record("tau", 0.2755544238890353, experiment="E0e", run_id="e0e", scope="synthetic", note=...)`, with the `note` string the ledger holds verbatim (it names the rule, the half-life and the EMA start).

`gamma_b` is `DERIVED`: `record()` refuses it by design. `get("gamma_b",
scope="synthetic")` would compute 0.30380 from the two rows above. The EMA half-life
has no registry entry; it would be a quarter of `E_lifetime`, the `half_life` row.

## Limitations

- **One substrate, one policy.** These are FIFO's values on fresh-stream arm B. The
  learned head's lifetime is not FIFO's (§12.2). `E_lifetime` here is a property of
  `(S, M)` and the stream-end truncation, not of the model.
- **The `tau` rule is the researcher's.** The spec gives none. It is pre-registered,
  and the owner can revise it. Under it, `fraction_inside_tau` is 0.75 by
  construction.
- **`tau` depends on the EMA's start.** Started at the band centre it is 0.27555; started at the
  first observation it is 0.39788. The PREREG fixed the first.
- The seeds' `tau` values range from 0.24097 to 0.31805. The pooled value is not a
  consensus.
- No scaling claim. Nothing here involves RSR, `ψ̂` or Kintsch & van Dijk.
