---
id: liveness-wiring
item: liveness-wiring
baseline_sha: 42332ee135fb6193b9797b9f2e51ec307b0ca220
falsifier: >-
  A training run whose memory is inert is indistinguishable at exit from one whose
  memory is live.

anchors:
  # the exit-code protocol
  - {path: src/rsr/exit_codes.py, line: 61, expect: "class Exit(IntEnum):"}
  - {path: src/rsr/exit_codes.py, line: 70, expect: "UNBANKED_RISE = 4"}
  - {path: src/rsr/exit_codes.py, line: 73, expect: "def status(code: object) -> Exit:"}
  - {path: src/rsr/exit_codes.py, line: 103, expect: "outside the protocol (0-4, docs/gates.md)"}
  - {path: tests/test_exit_codes.py, line: 54, expect: "def test_the_enum_is_the_five_code_protocol():"}
  - {path: tests/test_exit_codes.py, line: 61, expect: '"UNBANKED_RISE": 4,'}
  - {path: tests/test_exit_codes.py, line: 79, expect: "for bad in (-1, 5, 127):"}
  - {path: tests/test_exit_codes.py, line: 269, expect: "0 <= node.value <= 4"}
  - {path: tests/test_exit_codes.py, line: 300, expect: '"src/rsr/train/loop.py": "S0-05 brief: do not touch'}
  - {path: docs/gates.md, line: 83, expect: "### Exit codes"}
  - {path: docs/gates.md, line: 93, expect: "4  UNBANKED RISE"}
  # the training loop
  - {path: src/rsr/train/loop.py, line: 268, expect: "def train("}
  - {path: src/rsr/train/loop.py, line: 288, expect: "out = Path(out_dir)"}
  - {path: src/rsr/train/loop.py, line: 366, expect: "hb = Heartbeat("}
  - {path: src/rsr/train/loop.py, line: 498, expect: "ck.save("}
  - {path: src/rsr/train/loop.py, line: 509, expect: 'hb.footer("completed", final_step=iters)'}
  - {path: src/rsr/train/loop.py, line: 527, expect: "def main(argv: list[str] | None = None) -> int:"}
  - {path: src/rsr/train/loop.py, line: 580, expect: "return 0"}
  - {path: src/rsr/train/loop.py, line: 584, expect: "sys.exit(main())"}
  # the checkpoint loader
  - {path: src/rsr/train/checkpoint.py, line: 282, expect: "def load("}
  - {path: src/rsr/train/checkpoint.py, line: 301, expect: "torch.load(path"}
  # the instrument (reuse only)
  - {path: src/rsr/metrics/memory_liveness.py, line: 73, expect: "def shuffle_control("}
  - {path: src/rsr/metrics/memory_liveness.py, line: 221, expect: "def random_replacement("}
  - {path: src/rsr/metrics/memory_liveness.py, line: 268, expect: "def cross_row_cosine("}
  # the decision rule (read only; do not edit)
  - {path: experiments/decisive-shuffle/PREREG.md, line: 36, expect: "every seed `ratio ≤ 0.01` | **inert**"}
  - {path: experiments/decisive-shuffle/PREREG.md, line: 37, expect: "any seed `ratio ≥ 0.1` | **live**"}
  - {path: experiments/decisive-shuffle/PREREG.md, line: 35, expect: "or** `ratio == 1.0` exactly on any seed"}
  - {path: experiments/decisive-shuffle/PREREG.md, line: 38, expect: "| otherwise | `inconclusive` |"}
  - {path: experiments/decisive-shuffle/PREREG.md, line: 45, expect: "is reported as such, **not** rounded to either outcome"}
  - {path: experiments/decisive-shuffle/PREREG.md, line: 61, expect: "**Decoy aliasing:**"}
  - {path: experiments/shuffle-control/PREREG.md, line: 96, expect: "## Amendment 1"}

premises:
  - claim: the training loop does not call the liveness metric yet
    check: "git grep -l memory_liveness -- src/rsr/train/"
    expect_rc: 1
  - claim: no INERT exit code exists yet
    check: "git grep -n INERT -- src/rsr/exit_codes.py tests/test_exit_codes.py docs/gates.md"
    expect_rc: 1
  - claim: the protocol today is exactly the codes 0-4 (5 is refused by status())
    check: >-
      python3 -c "import importlib.util as u;
      s = u.spec_from_file_location('e', 'src/rsr/exit_codes.py');
      m = u.module_from_spec(s); s.loader.exec_module(m);
      print(sorted({int(e) for e in m.Exit}))"
    expect_rc: 0
    expect_stdout_contains: "[0, 1, 2, 3, 4]"
  - claim: nothing under src/ quarantines or refuses a checkpoint today
    check: "git grep -n -i quarantine -- src/"
    expect_rc: 1
  - claim: the decisive run found memory live on at least one arm
    ledger: runs/decisive-shuffle/ledger.json
    key: verdict.outcome
    expect: falsified
  - claim: the decisive run completed
    ledger: runs/decisive-shuffle/ledger.json
    key: status
    expect: ok

files_in_scope:
  - src/rsr/exit_codes.py
  - tests/test_exit_codes.py
  - docs/gates.md
  - src/rsr/train/loop.py
  - src/rsr/train/checkpoint.py
  - src/rsr/metrics/memory_liveness.py
  - scripts/mutation_battery.py
  - tests/test_train_loop.py
  - path: tests/test_liveness_wiring.py
    new: true
  - path: .orchestrator/outbox/liveness-wiring.md
    new: true

bar:
  - "Exit.INERT = 5 exists; status() accepts 0-5 and refuses 6; test_exit_codes pins the six codes; docs/gates.md's table has a row for 5."
  - "Every training run ends with a liveness measurement recorded in the heartbeat footer and the returned dict: ratio, A_trained, A_decoy, the three controls, cross-row cosine, matched-norm random-replacement ratio, and the Amendment 1 band label."
  - "loop.main exits through run_main on the pre-registered bands (decisive PREREG.md:35-38): 0 live (ratio >= 0.1); 5 INERT (ratio <= 0.01); 1 FAIL for the inconclusive band (0.01 < ratio < 0.1), never rounded to 0 or 5; 3 if the measurement is not valid (a control failed, A_decoy == 0, ratio == 1.0 exactly, or the measurement raised)."
  - "An INERT run's checkpoints are under <out_dir>/quarantine/ with an INERT marker file; checkpoint.load refuses them unless explicitly overridden."
  - "Mutations, each reddening only its named test: liveness hook skipped; inert reported as 0; decoy pointed at the trained model; quarantine skipped; loader override defaulting to allow; status() refusing 5."
  - "Gates: uv run pytest -rs --tb=no census and rc; ruff check and ruff format --check over src/ tests/ scripts/ both rc 0; mutation_battery.py --check rc 0 with its N/N line."
done_when:
  - "The report quotes the pytest census line and rc, the battery N/N line and rc, each new mutation with the node id it reddened, and one smoke run's footer showing the liveness keys."
  - "The report has BRIEF ERRORS, UNANSWERED BY THE BRIEF and BELIEVED, NOT VERIFIED fields, each written out (none if none)."
do_not:
  - "Change the Amendment 1 thresholds (0.01, 0.1) or the decisive PREREG."
  - "Put a threshold on cross-row cosine. It is recorded only; a threshold needs its own PREREG committed first."
  - "Write a third liveness implementation. Reuse memory_liveness.py; add beside it only."
  - "Round an inconclusive-band run to 0 or 5, treat an INERT run as 1, or a failed measurement as 0, 1 or 5."
  - "Rent or price a GPU, or run anything but CPU smoke runs."
---

# Brief: wire memory liveness into every training run (exit 5 INERT, quarantined checkpoints)

**Status:** written, not started. This supersedes `dispatch-2026-09-21-liveness-wiring.md`,
which is kept unchanged as history. It incorporates the owner rulings
`R-2026-09-22-liveness-go`, `R-2026-09-22-inert-exit-5` and
`R-2026-09-22-inert-checkpoint-quarantine`.
**Written at:** the front matter's `baseline_sha`. Every anchor in the front matter
was re-derived with `grep -n` at that sha, and `orchestrator.lint_brief` exits 0
against it. Re-run the lint at spawn if the base has moved.
**Lane:** researcher. CPU only; smoke runs only.

## Why

`runs/decisive-shuffle/` returned `falsified` (memory is live on arms B and C), which
makes a trained-live negative control exist. It also showed that a run can train with
a row-agnostic memory: in arm A the cross-row cosine is near 1 while random
replacement still moves the loss. From the outside that looks like a working model.
Today liveness is measured only by a separate script after training, so an inert run
exits `0` and its checkpoint is consumed like any other.

The owner ruled on the two questions the old brief left open:

- **"Ran and inert" is a new exit code, `5 INERT`, not `1`**
  (`R-2026-09-22-inert-exit-5`). `1` means the check caught a failure and routes to a
  regression or debug item. `5` means the model trained but its memory carries nothing
  an eviction rule can act on, and routes to a model or training-config investigation;
  retention experiments on that checkpoint are refused.
- **An inert run still writes its checkpoint, quarantined**
  (`R-2026-09-22-inert-checkpoint-quarantine`): under `<out_dir>/quarantine/` with an
  `INERT` marker file. Loaders refuse a quarantined checkpoint unless explicitly
  overridden. Forensics are kept, and nothing downstream reads it silently.

## Falsifier

"A training run whose memory is inert is indistinguishable at exit from one whose
memory is live."

**Decision rule, per run (one seed):** the Amendment 1 statistic, unchanged
(`experiments/shuffle-control/PREREG.md`, *Amendment 1*, as applied by
`experiments/decisive-shuffle/PREREG.md`): `A` = mean |per-token Δ| over real tokens,
`ratio = A_trained / A_decoy`, the decoy being the untrained model at the same seed
and config.

| condition | exit |
|---|---|
| any control fails, `A_decoy == 0`, `ratio == 1.0` exactly (decoy aliasing, PREREG `:35`, `:61`), or the measurement raises | `3` DID NOT RUN |
| `ratio ≥ 0.1` | `0` OK (live) |
| `ratio ≤ 0.01` | `5` INERT |
| `0.01 < ratio < 0.1` (Amendment 1's `inconclusive`) | `1` FAIL — liveness not demonstrated |

The mapping follows the pre-registered bands exactly
(`experiments/decisive-shuffle/PREREG.md:35-38`). ⚠️ **The inconclusive band is
"reported as such, not rounded to either outcome"** (`:45`): it is neither `0` nor `5`.
It gets `1` because the run fails its premise — live memory was not demonstrated — and
the ledger records the band label (`inert` / `inconclusive` / `live`) beside the exit
code. (An earlier draft of this brief and of the ruling said `ratio < 0.1` → `5`; that
folded `inconclusive` into `inert` and is corrected in the ruling's erratum.) **Do not
move either threshold.** The cross-row cosine and the matched-norm random-replacement
ratio are recorded and have no threshold and no effect on the exit code.

## Files in scope

- `src/rsr/exit_codes.py`: add `Exit.INERT = 5`; `status()` accepts 0–5, and its
  refusal message says so; update the module docstring table.
- `tests/test_exit_codes.py`: the enum test pins six codes; the out-of-protocol
  examples replace `5` with `6`; `_is_protocol_value` accepts 0–5; remove
  `src/rsr/train/loop.py` from `NOT_CONVERTED` once `loop.main` is converted.
- `docs/gates.md`: add `5  INERT` to the exit-code table, with one line on why it is
  not `1`.
- `src/rsr/train/loop.py`: after the training loop, measure liveness with
  `memory_liveness.py`'s existing functions; write it to the heartbeat footer and the
  returned dict; quarantine on INERT; `main()` returns an `Exit` through `run_main`,
  using `rsr.exit_codes.ArgumentParser`.
- `src/rsr/train/checkpoint.py`: `load()` refuses a checkpoint under a `quarantine/`
  directory or beside an `INERT` marker unless an explicit override argument is set.
- `src/rsr/metrics/memory_liveness.py`: reuse only. An additive helper is allowed if
  the loop needs one; no new implementation.
- `tests/test_train_loop.py`, `tests/test_liveness_wiring.py` (new): the gates.
- `scripts/mutation_battery.py`: the new entries.
- `.orchestrator/outbox/liveness-wiring.md`: the report (one file per run;
  `orchestrator.outbox new liveness-wiring` creates it, `outbox index` regenerates
  `researcher.md`).

## Bar

1. `Exit.INERT == 5`; `status(5) is Exit.INERT`; `status(6)` raises; the enum test pins
   `{OK: 0, FAIL: 1, UNKNOWN: 2, DID_NOT_RUN: 3, UNBANKED_RISE: 4, INERT: 5}`;
   `docs/gates.md` has the row.
2. Every `train()` call ends with the liveness measurement in the heartbeat footer and
   in the returned dict: `ratio`, `A_trained`, `A_decoy`, the three controls
   (memory-disabled `== 0.0`, own-memory `== 0.0`, decoy `> 0`), the cross-row cosine,
   the matched-norm random-replacement ratio, and the Amendment 1 band label.
3. `loop.main` exits `0` on live, `5` on INERT, `1` on the inconclusive band, `3` on a
   measurement that is not valid, through `run_main` — one test per row of the table.
4. An INERT run writes its checkpoints under `<out_dir>/quarantine/` with an `INERT`
   marker, and `checkpoint.load` refuses them without the explicit override.
5. Mutations, each reddening only its named test (quote the node ids):
   - the liveness hook skipped;
   - an inert result reported as `0`;
   - the decoy pointed at the trained model (reads `ratio == 1.0` exactly, which the
     PREREG makes not-valid, so the run must exit `3`, not `0`);
   - an inconclusive-band ratio rounded to `0` or `5`;
   - the quarantine skipped for an INERT run;
   - the loader override defaulting to allow;
   - `status()` refusing `5`.
6. Gates: `uv run pytest -rs --tb=no` census line and `$?`; `ruff check` and
   `ruff format --check` over `src/ tests/ scripts/` both `0`;
   `mutation_battery.py --check` `0` with its N/N line.

## Done when

The report in `.orchestrator/outbox/liveness-wiring.md` quotes: the census line and rc; the
battery N/N line and rc; each new mutation and the node id it reddened; and the
footer of one CPU smoke run showing the liveness keys and the exit code it produced.
It carries `BRIEF ERRORS`, `UNANSWERED BY THE BRIEF` and `BELIEVED, NOT VERIFIED`,
each written out.

## Do NOT

- Change the Amendment 1 thresholds (`0.01`, `0.1`) or edit either PREREG.
- Put a threshold on the cross-row cosine. A threshold needs its own PREREG,
  committed first (`R-2026-09-22-inert-exit-5`).
- Write a third liveness implementation.
- Report INERT as `1`, or a failed measurement as `0` or `5`.
- Run anything but CPU smoke runs. No GPU is rented or priced (`CLAUDE.md`, ADR-0007).
- Backpropagate anything new into the transformer: the liveness measurement is
  `torch.no_grad()` evaluation only.
