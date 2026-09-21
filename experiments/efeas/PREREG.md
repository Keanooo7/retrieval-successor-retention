# E-feas — the oracle against FIFO, synthetic corpus — pre-registration

🔒 **Committed before `run.py` exists, and before any corpus-level hit rate has been
computed.** The only numbers seen so far come from `tests/test_oracle.py`: two hand-built
five- and six-sentence streams, and the property checks below, which assert relations,
not rates. A correction goes in `RESULTS.md` and names this file.

## The question

`src/rsr/baselines/oracle.py`, since the repo's first week:

> **Run the oracle first on each corpus as a feasibility probe (E-feas).** If oracle ≈ FIFO,
> that corpus cannot exhibit the effect. One day, saves six weeks.

**Does the synthetic corpus leave room for a retention rule to beat FIFO?** If an
optimal rule with perfect knowledge of future demand keeps barely more of what queries
need than FIFO does, then no learned rule can do better, and E1 on this corpus cannot
show RSR beating TG.

## Why without a model

The measurement is **model-free**: did the policy keep the sentence a query needs? §10.3
and `runs/shuffle-control/ledger.json` (verdict `survived`) show that this repo's trained
memory is inert. A headroom read through the model's loss would say "oracle ≈ FIFO"
because the model ignores its memory, not because of the corpus, and that is exactly
the confound this measurement has to avoid. The cost of this choice is stated below.

## Condition

| | |
|---|---|
| corpus | `generate(SyntheticConfig(seed=s))`, the committed defaults: 64 documents × 48 sentences, `max_gap 40`, `geometric_p 0.30`, `heavy_tail_weight 0.35`, `heavy_tail_min 12` |
| corpus seeds | `0, 1, 2` |
| memory | **`M = 16`** (the project's `memory_slots`). `M = 8` and `M = 32` are reported as secondary rows, not judged |
| mechanics | `src/rsr/metrics/headroom.py::simulate`, using the real `init_memory` / `write_at` / `memory_state`. A query at step `j` **hits** if its asserting sentence is resident before step `j`'s write |
| arms | `FIFOPolicy` · `OraclePolicy(discounted_demand(doc, γ=0.97))`, which is argmin of future discounted demand and Belady's MIN on this corpus for any γ in (0, 1) · `RandomPolicy`, seeded, reported for reference only |
| statistic | **headroom `H = hit_rate(oracle) − hit_rate(FIFO)`**, pooled over the 64 documents of one corpus seed. Spread is across seeds |

## Controls, per seed. If either fails, the verdict is `inconclusive`

1. **Simulator timing.** Under FIFO, every query hits **iff** `gap ≤ M`. FIFO keeps exactly
   the last `M` sentences, so an off-by-one in "reads before the write" breaks this.
2. **Oracle optimality floor.** On every document, `hit_rate(oracle) ≥ hit_rate(FIFO)`.

## Decision rule — `run.py::decide`

| condition | verdict |
|---|---|
| a control fails on any seed | `inconclusive` |
| every seed `H < 0.05` | `falsified`: oracle ≈ FIFO, so **the synthetic corpus cannot exhibit the effect**, and E1 on it is uninformative |
| every seed `H ≥ 0.05` | `survived`: the corpus has headroom, so a retention rule has something to win |
| otherwise | `inconclusive` |

`0.05` means five percentage points of all queries. It is a judgment call, not a
derived number: nothing in the repo yet says how small a hit-rate gap E1 could resolve.
What makes it a threshold rather than a fit is that it is fixed here, before any rate is
seen. The per-gap-bucket hit rates are reported and are not the bar.

## What this cannot show, stated before the result

- **Headroom is an upper bound, not an effect.** `survived` says retention *could* matter
  on this corpus. It says nothing about whether TG uses what it keeps (§10.3 says it
  does not) or whether `ψ̂` can learn the oracle's ordering (that is E1).
- **The corpus builds its gaps.** A `survived` here is partly by design (defect D-9): the
  generator's heavy tail puts facts past `M` on purpose. The informative outcome is
  therefore `falsified`, or a headroom much smaller than the gap structure suggests.
- **Synthetic only.** PG-19 needs the shadow buffer (Sprint 2), and this says nothing
  about it.
