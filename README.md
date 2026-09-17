# Retrieval-Successor Retention (RSR)

A learned memory-retention policy for the **Thought Gestalt** (TG) architecture.

TG is a recurrent transformer that compresses each sentence to one vector (a *gestalt*) and writes
it into a fixed-size working memory of `M` slots; later sentences reach earlier ones only through
cross-attention to that memory. When memory is full, **TG evicts the oldest slot.**

RSR replaces that one line: evict the slot with the lowest *predicted future retrieval demand*,
given where the discourse currently is.

The scientific question is older than the architecture. Kintsch & van Dijk (1978) hand-specified a
retention rule for a capacity-limited discourse buffer — the leading-edge strategy — and it
reproduced the human narrative-recall gradient. **Does a system trained only to predict the next
sentence discover that policy on its own?**

## Status

| | |
|---|---|
| Spec | `docs/spec/rsr_model_spec_v0.5.md` |
| Corrections that override the spec body | `docs/spec-corrections.md` — **read before implementing** |
| Citation audit (E0f) | `docs/citation-audit.md` |
| Decisions | `docs/decisions/` |
| Release conditions (§16) | `docs/release-conditions.md` |
| Approved scope | **Weeks 1–4 only.** Everything downstream is a projection, not a permission |

## Quickstart

```bash
uv sync
uv run pytest
uv run ruff check
uv run rsr --help
```

## Layout

```
src/rsr/
  model/        TG base, memory, gestalt write
  retention/    policy, value head, reward, shadow buffer, bias loop
  baselines/    fifo lru h2o expire_span leading_edge random oracle
  data/         synthetic generator, pg19, coref
  metrics/      reintroduction, loo, vacuity, gini
  mup/          parameter groups, coordinate check
  constants.py  the registry — raises on an unmeasured read
configs/        base + model/ + data/ + experiment/
experiments/    one directory per E0*/E-feas/E1/E2, each with run.py + RESULTS.md
preregistration/  committed BEFORE the experiment it governs
```

## The three load-bearing invariants

1. **§3.7 reduction to exact TG is reachable by configuration alone**, never by a separate code
   path. Every term added to the eviction score has a documented off-switch. `tests/test_reduction.py`
   asserts a bit-exact loss curve and runs on every commit touching `src/rsr/`.
2. **`src/rsr/constants.py` refuses unmeasured reads.** A `MEASURED` or `DERIVED` constant read
   before its source experiment has logged a value raises, naming the experiment.
3. **Every baseline and RSR implement one `RetentionPolicy` protocol**, so they are interchangeable
   by config and no comparison depends on which code path ran.
