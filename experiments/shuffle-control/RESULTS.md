# Shuffle control — the §10.3 claim, re-derived from committed code

Brief: `docs/lab-notes/dispatch-2026-09-20f-shuffle-control.md`, section "TASK — run
it, once, on CPU". Pre-registration: `experiments/shuffle-control/PREREG.md`
(original rule + **Amendment 1**, both committed before this run). Neither
`PREREG.md`, `decide()` nor the thresholds were touched after the result.

Every number below is quoted from `runs/shuffle-control/ledger.json` (row key given)
unless it is marked as prose from §10.3.

---

## Provenance

| | |
|---|---|
| git sha | `8e60df1dad078044d9e03f2d4b67125da1c7dcd8` (`provenance.git_sha`, stamped by `scripts/ledger.py` from inside the tree). Descendant of the brief's baseline `7fdb9b2`, the last commit touching `run.py` |
| `provenance.dirty` | **`true`** — see the note under this table |
| device | `cpu` (`device`) · `platform` `macOS-26.6.2-arm64-arm-64bit`, Python `3.12.13` · Mac Studio |
| manifest | `runs/shuffle-control/manifest.json`, `config_hash` `e8faebd62694f79f1a350483e283c87d2199dd89189632aa0cc059d00bc4608e` |
| manifest frozen at | `2026-09-21T04:10:03Z` (`manifest_written_utc`), equal to `started_utc`, before the first seed started |
| seeds actually run | `[0, 1, 2]` (`seeds_actually_run`), one subprocess each |
| steps requested / done | `300 / 300` (`steps_requested`, `steps_done` = min over seeds) |
| per-seed runs | 3 `commands[]`, each `exit_code: 0`, each `reproducible: true`; per-seed trainer config hashes `dd6a92f615ddb22b` · `211126cfe52fd4f9` · `50e3f6891cf289e0` |
| wall clock | `started_utc 2026-09-21T04:10:03Z` → `finished_utc 2026-09-21T04:55:53Z`; seeds finished at `04:25:12Z`, `04:40:34Z`, `04:55:47Z` |
| status | `ok` |
| gates | `uv run pytest -rs` → **exit 0**, `passed=364 failed=0 skipped=0 errors=0`; pytest's own line `364 passed, 1 xfailed, 1 warning`. The xfail is `tests/test_train_loop.py`'s documented `xfail(strict=True)`; it is not a pass and the census line does not print it. **The mutation battery was not run this cycle.** |

Command:

```
uv run python experiments/shuffle-control/run.py --device cpu > runs/shuffle-control.log 2>&1; rc=$?
```

`rc=0`, read directly.

⚠️ **`provenance.dirty: true`.** The only change in the tree when the run began was an
untracked `uv.lock` at the repo root. It is not this run's and was left uncommitted
on instruction. `git status --short` just before launch printed only `?? uv.lock`.
`Ledger.write()`'s own check, which refuses uncommitted `src/`, `scripts/` or
`experiments/`, passed. So the measured code is exactly `8e60df1`. The whole-tree
flag is `true` because of the lockfile, not because of the source.

---

## Verdict — `survived`

`verdict.outcome` = **`survived`**. `verdict.detail`:

> every seed's mean |token delta| is <= 0.01 of the live decoy's ({0:
> 0.0009913241306859642, 1: 0.0008765567882922645, 2: 0.0004705000245672635}), and
> all three controls read as required on every seed

Under PREREG Amendment 1's rule, the trained model's tokens move at under 1% of the
rate a live random memory produces at the same shape and seed, on all three seeds.
Both controls that must read exactly `0.0` did so, and the decoy read non-zero.
**§10.3's claim that the trained memory is inert is re-derived** from committed code
by an instrument that was seen reading both ways in this same run.

---

## 1 — The primary statistic: ratio to the live decoy

`ratio_to_live_decoy` = `mean_abs_token_delta(trained) / mean_abs_token_delta(decoy, same seed)`.

| seed | `A_trained` (`reading.mean_abs_token_delta`) | `A_decoy` (`seedN.control_live_decoy.mean_abs_token_delta`) | **ratio** |
|---|---|---|---|
| 0 | `5.7338859433025566e-06` | `0.005784067759285646` | **`0.0009913241306859642`** |
| 1 | `9.085651261575595e-06` | `0.01036515988801655` | **`0.0008765567882922645`** |
| 2 | `3.671396278113374e-06` | `0.007803179779831244` | **`0.0004705000245672635`** |
| mean ± sd | `6.1636444943305084e-06 ± 2.732591913481156e-06` | | **`7.794603145151641e-04 ± 2.7365166049842687e-04`** |

The spread is non-zero (`sd_exactly_zero: false`). The largest seed, seed 0, is still
about 10× under the `0.01` inert threshold.

📌 **Consistency check.** The decoy's `A` per seed matches the values Amendment 1
quotes from the 2-iteration smoke run, to the digits that file gives (`5.8e-3`,
`1.04e-2`, `7.8e-3`). That is expected. The decoy is the untrained model initialised
at the seed, so it does not depend on `iters`. This is a determinism result, not a
coincidence.

---

## 2 — The controls, every seed

| seed | memory disabled: `delta_exactly_zero` / `n_tokens_moved` / `mean_abs_token_delta` | own memory replayed: same three | live decoy: `n_tokens_moved` / `delta_nats_per_token` |
|---|---|---|---|
| 0 | `true` / `0` / `0.0` | `true` / `0` / `0.0` | `1551` / `0.00021674290712248023` |
| 1 | `true` / `0` / `0.0` | `true` / `0` / `0.0` | `1556` / `0.00023981622585989015` |
| 2 | `true` / `0` / `0.0` | `true` / `0` / `0.0` | `1533` / `0.0003071628054795994` |

Every control passed on every seed. For the decoy, `delta_exactly_zero: false` is the
passing value.

---

## 3 — The reading, in full

| quantity (row) | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|
| honest real-token NLL (`reading.loss_real_tokens_honest_memory`) | `1.5561023927341826` | `1.6258142544373317` | `1.564667227170939` | `1.5821946247808178 ± 0.0380176691258267` |
| signed mean Δ, nats/token (`reading.delta_nats_per_token`) | `-1.464733645484273e-08` | `3.5269958353900677e-07` | `-4.609053716464473e-08` | `9.732056997317311e-08 ± 2.2172279798232956e-07` |
| signed Δ relative (`reading.delta_relative`) | `-9.41283589257023e-09` | `2.1693719474803747e-07` | `-2.945708605911087e-08` | `5.935575759878546e-08 ± 1.3683703808436684e-07` |
| max per-token Δ (`reading.max_token_delta`) | `0.00011730194091796875` | `0.00023126602172851562` | `0.00006008148193359375` | `0.00013621648152669272 ± 0.00008714560546022443` |
| min per-token Δ (`reading.min_token_delta`) | `-0.00019109249114990234` | `-0.00017136335372924805` | `-0.00015807151794433594` | `-0.0001735091209411621 ± 0.000016614734591815715` |
| `delta_exactly_zero` (`seedN.reading.…`) | `false` | `false` | `false` | |
| tokens moved / real tokens | `1122 / 1588` | `1254 / 1596` | `1100 / 1568` | |
| `n_sentences_with_eos` / `n_sentences` | `384 / 384` | `384 / 384` | `384 / 384` | |
| `final_slots_filled_per_row` | all 8 rows `16` | all 8 rows `16` | all 8 rows `16` | |
| `memory_gates` (6 layers) | `0.9486`–`0.9886` | `0.9823`–`0.9948` | `0.9681`–`0.9981` | |

(The gate ranges are the minimum and maximum of each seed's six-element
`seedN.reading.memory_gates` row, rounded to 4 d.p. for display. The full values are
in the ledger.)

---

## 4 — Beside §10.3, which is prose from a run nobody can reproduce

The row `audit_2026_09_18_for_comparison` carries the §10.3 digits. Its `how` reads:
*"PROSE, from an uncommitted script on uncommitted checkpoints. Reported beside the
reading, NOT the bar."*

| | §10.3 (prose, unreproducible) | this run (ledger) |
|---|---|---|
| signed mean Δ | "exactly 0.0" | **not exactly zero on any seed**: `-1.46e-08`, `+3.53e-07`, `-4.61e-08` nats/token |
| max per-token Δ | `1.2e-4` | `1.17e-4` · `2.31e-4` · `6.01e-5` |
| min per-token Δ | `-1.9e-4` | `-1.91e-4` · `-1.71e-4` · `-1.58e-4` |
| sentences | 384 | 384 per seed, all with EOS |
| `memory_gate` range | `0.949`–`0.989` | seed 0 `0.9486`–`0.9886` |

🔴 **Amendment 1 weakens §10.3's own headline, and this run shows how.** The claim
that the loss "moves by exactly 0.0" is about the **signed mean**. Amendment 1 showed,
before this run, that a memory known to be live moves the signed mean by only ~5e-5
relative, because tokens move both ways and cancel. The signed mean is therefore weak
evidence that a memory is inert, whatever its value. In this run it is also **not
exactly 0.0**: `delta_exactly_zero` is `false` on all three seeds, and 1100–1254 of
~1580 real tokens move. The headline should not be repeated as "exactly 0.0". What
carries the claim is the **per-token** magnitude measured against a live decoy, and
that is what survived: about `7.8e-4` of a live memory's rate on average.

Seed 0's per-token extremes and gate range come close to the §10.3 digits. Bit-identity
was not expected: PREREG lists the known differences from the audit's trainer.

---

## What this run does not establish

- **A random-init decoy is not a trained live memory.** PREREG says so itself. The
  decoy shows the instrument can read non-zero at this shape. It does not show what a
  trained model that uses its memory would read. S0-04's negative control on a
  trained live memory still needs such a model.
- **Why the memory is inert.** The run re-derives the absence. It does not separate
  the hinge-off, corpus-memorisability and unmasked-loss explanations.
- **One configuration.** FIFO, `d=128`, `M=16`, synthetic corpus, 300 iterations,
  CPU. §13: no cross-corpus comparison of absolute numbers is valid.
- **The audit's own checkpoints.** They were not committed and were not measured
  here. This is a re-derivation of the claim, not of the digits.
- **Checkpoints are not committed.** `runs/shuffle-control/seed*/ckpt-000300.pt` stay
  local, per the brief.
