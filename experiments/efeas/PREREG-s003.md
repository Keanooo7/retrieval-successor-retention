# E-feas re-run on the S0-03 corpus: pre-registration

🔒 **Written 2026-09-21 by the manager, before S0-03 was spawned, and so before the S0-03 corpus existed or had been measured by anyone.** It is committed ahead of the run it governs.

## Why a re-run

`runs/efeas-synthetic/` measured `generate(SyntheticConfig(seed=s))` **before** S0-03. S0-03 changes that generator: the answer enters the token stream, and a generator invariant on the `gap > M` fraction may change the gap distribution. E-feas is model-free, so the old result stands for the old corpus. The object it measured is being superseded, though, so it is re-measured on the new one. **`runs/efeas-synthetic/` is not touched.**

## Condition, controls, statistic, decision rule

**Identical to `experiments/efeas/PREREG.md`, and adopted by reference without change:** corpus seeds `0, 1, 2` with the committed `SyntheticConfig` defaults *as they stand on `main` after S0-03 merges*; `M = 16` primary, with `M = 8` and `M = 32` secondary; arms FIFO / Oracle (γ = 0.97) / Random; `H = hit_rate(oracle) − hit_rate(FIFO)`; the two controls (FIFO hits iff `gap ≤ M`; oracle ≥ FIFO on every document); and the rule:
- every seed `H ≥ 0.05` → `survived`;
- every seed `H < 0.05` → `falsified`;
- a control fails → `inconclusive`;
- otherwise → `inconclusive`.

**The `0.05` threshold is the one fixed in `experiments/efeas/PREREG.md`. None is chosen here.**

If S0-03 changed a `SyntheticConfig` default (for example, the gap distribution), the ledger's manifest records the new defaults verbatim, and `RESULTS-s003.md` names every changed field.

## Comparison (reported, not the bar)

The headroom at `M = 16` and at `M = 32`, beside the old run's ledger keys `headroom_oracle_minus_fifo` and `secondary.M32.headroom` in `runs/efeas-synthetic/ledger.json`. They are quoted from that ledger, never typed.

**run_id:** `efeas-synthetic-s003`.
