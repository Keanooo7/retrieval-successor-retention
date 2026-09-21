# Studio dispatch — 2026-09-20f · the shuffle control, re-derived from committed code

**Baseline: `7fdb9b2`**, the commit that last touched `experiments/shuffle-control/run.py`
(`git log -1 --format=%h -- experiments/shuffle-control/run.py`, read at the moment of writing).
You are the **RSR manager**. Your role file is `.claude/agents/rsr-manager.md`.

## 🔴 Launch condition

```
cd ~/retrieval-successor-retention   # the session must START here, not in ~
git pull --ff-only
git log -1 --format=%h -- experiments/shuffle-control/run.py   # must print 7fdb9b2 or a descendant's
```

Confirm `rsr-manager` and `rsr-researcher` are in your roster before doing anything.

---

## Why this exists

§10.3 is the project's one finding about what a trained model does, and it is **prose only**.
The 2026-09-18 audit measured it with a scratch `measure.py` on `runs/cycle-srep-hinge/i300-*`
checkpoints. Neither was committed. `docs/ROADMAP.md` §5 item 1 says so.

The MacBook has now committed four things:

| | |
|---|---|
| the instrument | `src/rsr/metrics/memory_liveness.py` — per-token deltas plus the has_eos / slot-fill / `memory_gate` facts |
| its test | `tests/test_shuffle_control.py`. Live random-init memory reads non-zero; gates zeroed reads exactly `0.0`; own memory replayed reads exactly `0.0` |
| two mutations | in `scripts/mutation_battery.py`. Each reddens only `tests/test_shuffle_control.py` |
| the experiment | `experiments/shuffle-control/PREREG.md` (Amendment 1 included, both committed before any 300-iter run) + `run.py` |

## What was verified on the MacBook, at `7fdb9b2`

| Claim | Result |
|---|---|
| suite | `uv run pytest -rs` → `passed=364 failed=0 skipped=0 errors=0`, exit `0` |
| floor | base `42f54a7` measured in its own venv → `358`. **+6**: 5 new tests + 1 parametrized `test_no_hardcoded_constants[metrics/memory_liveness.py]` |
| battery | `mutation_battery.py --check` → **exit 1, `45/47`**. Both new mutations `PROVEN` (`2 on gate, 0 off` · `1 on gate, 0 off`). The 2 unproven are unrelated mutations, and each was reddened only by the SIGKILL flake below. Re-run in isolation, neither reddened it. **Read the battery's exit code as the flake's, not this branch's** |
| smoke | `run.py --device cpu --iters 2 --run-id shuffle-control-smoke` → exit `0`, ledger `dirty: false`, all 3 commands `reproducible: true`, verdict `falsified` (ratios `0.105 / 0.052 / 0.107`). **Plumbing only; the ledger was deleted, not committed** |

## 🔑 Amendment 1 — read it before you read the result

The original decision rule was on the **signed mean** Δ. The smoke run showed that the
**live decoy passes it**: a random memory moves the signed mean by ~5e-5 relative while single
tokens move ~0.1 nats, and the two cancel. The rule now uses **mean |per-token Δ|** as a ratio to
the live decoy at the same seed: `≤ 0.01` inert, `≥ 0.1` live. ⚠️ **It also weakens §10.3's own
headline**: *"moves by exactly 0.0"* is a signed-mean statement. **Say so in `RESULTS.md` whatever
the outcome.**

---

## TASK — run it, once, on CPU

Spawn a fresh `rsr-researcher`, and give it this verb and path: **run
`experiments/shuffle-control/run.py` per `experiments/shuffle-control/PREREG.md`.**

```
uv run python experiments/shuffle-control/run.py --device cpu > runs/shuffle-control.log 2>&1; rc=$?
```

- **CPU, not MPS.** The run-to-run floor there is exactly `0.0`, so every delta is real. PREREG
  pins it.
- **~20 min per seed** at `iters=300` (the audit's `i300-h0-seed0` took `1141.5 s`). Three seeds
  run serially.
- Then write `experiments/shuffle-control/RESULTS.md` **from `runs/shuffle-control/ledger.json`
  only**, following `experiments/cycle-01-masked-loss/RESULTS.md`'s shape: provenance table,
  verdict, the ratio per seed with spread, all three controls per seed, and §10.3's digits
  **beside** the reading, labelled as prose from an unreproducible run.
- Commit `runs/shuffle-control/{ledger.json,manifest.json,raw.json}` and `RESULTS.md`.
  **Not** the `.pt` checkpoints. Push.

## Predictions, with the command that tests each

| Prediction | Command |
|---|---|
| suite still `passed=364`, exit 0 | `uv run pytest -rs; echo $?` |
| every control passes on every seed | `jq '.rows[] \| select(.key \| test("control_.*delta_exactly_zero"))' runs/shuffle-control/ledger.json` |
| `n_sentences_with_eos` = `384` per seed | `jq '.rows[] \| select(.key \| test("n_sentences_with_eos"))'` |
| the memory is inert, i.e. `survived` | `jq .verdict runs/shuffle-control/ledger.json`. **Any outcome is a result. Do not rerun for a different one** |

## Standing, non-negotiable

- **Literal output, or it did not happen.** Read `$?` directly, never after a pipe.
- **Every number comes from the ledger.** `RESULTS.md` quotes rows; it types no number the ledger lacks.
- **Report spread. An sd of exactly 0.0000 across seeds is broken, not clean.**
- 🔴 **Do not edit PREREG.md, `decide()` or the thresholds after seeing the result.** A correction
  goes in `RESULTS.md` and names the file.
- **Push before you report.**

## Out of scope

The rest of S0-04: wiring the control into `train()` and the heartbeat, and refusing runs.
**S0-04's negative control on a *trained* live memory**: the random-init decoy is not that, and
PREREG says so. E1, E0d, E0h, `γ_b`, `ν`. §15.

## Also found — not yours to fix, recorded so it is not lost

`tests/test_checkpoint.py::test_a_sigkill_mid_save_never_leaves_a_corrupt_checkpoint[0.010]` failed
**three times** tonight: once at `42f54a7`, on the first run in a fresh venv: *"SIGKILL during the write replaced a good
checkpoint with a partial one"*, at byte 105. It failed twice more inside the mutation battery at `7fdb9b2`, under two unrelated mutations. It passed 5/5 × 4 params on a quiet re-run. It is
intermittent, but the assertion describes a real atomicity failure, not harness noise.
