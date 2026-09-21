# E-feas — the oracle against FIFO, synthetic corpus — results

Pre-registration: `experiments/efeas/PREREG.md` (`663d829`), committed ahead of `run.py`
(`1fc8199`). Every number below is read from `runs/efeas-synthetic/ledger.json` or its
`raw.json`. None was typed from memory.

## Provenance

| | |
|---|---|
| git sha | `1fc8199` · `dirty: false` (measured by `scripts/ledger.py`, not typed) |
| machine | MacBook Pro, M1 Pro, CPU. **No model is trained**, so ADR-0007's all-training-on-the-Studio rule does not apply |
| manifest | `runs/efeas-synthetic/manifest.json`, config hash `2e591c0b6795f7a9`, frozen `2026-09-21T06:34:48Z` at the start of the run |
| seeds | corpus seeds `[0, 1, 2]` (documents per seed as set in `experiments/efeas/run.py`; no ledger key carries the count), **1180 / 1192 / 1198** queries |
| command | `uv run python experiments/efeas/run.py > log 2>&1; rc=$?` → **`rc=0`**, read directly. The ledger's own `exit_code: 0` is self-reported and says so |

## Verdict: **`survived`**. The synthetic corpus has headroom at M = 16

| | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|
| **headroom `H` = oracle − FIFO** | **0.1949** | **0.1737** | **0.1736** | **0.1807 ± 0.0123** |
| hit rate, oracle | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| hit rate, FIFO | 0.8051 | 0.8263 | 0.8264 | 0.8193 ± 0.0123 |
| hit rate, random (reference) | 0.7898 | 0.8121 | 0.8055 | 0.8025 ± 0.0114 |

Both controls held on every seed (`seed{0,1,2}.controls_failed = []`): FIFO hit a query
iff its gap was ≤ 16, and the oracle never fell below FIFO on any document.

## What the number is, and what it is not

🔑 **The headroom is exactly the share of facts FIFO cannot hold.** Per gap bucket (seed 0):

| gap | n | FIFO | oracle | random |
|---|---|---|---|---|
| 1–4 | 688 | 1.000 | 1.000 | 0.971 |
| 5–8 | 131 | 1.000 | 1.000 | 0.840 |
| 9–12 | 48 | 1.000 | 1.000 | 0.833 |
| 13–16 | 83 | 1.000 | 1.000 | 0.506 |
| 17–20 | 48 | **0.000** | 1.000 | 0.458 |
| 21–24 | 47 | **0.000** | 1.000 | 0.383 |
| 25–28 | 53 | **0.000** | 1.000 | 0.245 |
| 29–32 | 33 | **0.000** | 1.000 | 0.242 |
| 33–36 | 26 | **0.000** | 1.000 | 0.231 |
| 37–40 | 23 | **0.000** | 1.000 | 0.217 |

FIFO is a step function at `gap = M`, as its control requires. The oracle keeps
everything, so **at M = 16 capacity never cost a rule that knows the future a single
query**. (M = 8 is where it starts to: the oracle drops to 0.986–0.993.) The headroom is the generator's
heavy tail showing through FIFO, which is defect D-9, stated in PREREG before the run:
*the corpus builds its gaps*. `survived` was the expected outcome, and the pre-registration
said the informative result would have been `falsified`.

⚠️ **The answer depends on M.** The secondary rows were not judged, but they show it:

| M | headroom, mean ± sd | per seed |
|---|---|---|
| 8 | 0.2721 ± 0.0179 | 0.2924 / 0.2584 / 0.2654 |
| **16** | **0.1807 ± 0.0123** | 0.1949 / 0.1737 / 0.1736 |
| 32 | **0.0337 ± 0.0088** | 0.0415 / 0.0352 / 0.0242 |

**At M = 32 the same rule would have returned `falsified`**, with every seed below 0.05.
This corpus supports a retention study at the project's M = 16 and would not at twice
that capacity.

⚠️ **The oracle's `sd = 0.0000` is a ceiling, not a broken seed.** The project's standing
rule treats an exactly-zero sd across seeds as broken until shown otherwise, and the
ledger flags the row `sd_exactly_zero: true`. The discriminator is on the same page: the
seeds reached the RNG, because the query counts differ (1180 / 1192 / 1198) and FIFO's and
random's sd are 0.0123 and 0.0114. The oracle is 1.0 on every seed because it misses
nothing, and 1.0 has no spread to show.

📌 **Random ≈ FIFO in aggregate** (0.8025 vs 0.8193) **but not by gap**: random loses
short-gap facts FIFO keeps and keeps long-gap facts FIFO loses. An aggregate hit rate
cannot tell those two apart, so **E1 should report hits by gap bucket, not only in total**.

## What this does not establish

- **That TG would use what the oracle keeps.** `runs/shuffle-control/ledger.json`
  (`survived`) says the trained memory is inert. Headroom is what retention *could* buy.
- **That `ψ̂` can learn the oracle's ordering.** That is E1.
- **Anything about PG-19.** `OraclePolicy` refuses to run without ground-truth demand,
  and PG-19 has none until the shadow buffer exists (Sprint 2).


---

*2026-09-21 (manager, wave 0.2): prose-only edit so this file passes `render_scoreboard.py --audit`; the one unbacked literal (documents per seed) was deleted. The ledger is untouched.*
