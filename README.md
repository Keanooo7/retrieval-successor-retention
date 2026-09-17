# Retrieval-Successor Retention (RSR)

Thought Gestalt (TG) is a recurrent transformer that compresses each sentence to
one vector and writes it to a working memory of `M` slots; later sentences reach
earlier ones only through cross-attention to that memory. When memory is full, TG
evicts the oldest slot.

**RSR replaces that one line** with a learned policy that evicts the slot with the
lowest predicted *future* retrieval demand given the current discourse state.

> Kintsch & van Dijk (1978) specified retention by structural importance over a
> capacity-limited discourse buffer — the leading-edge strategy — and it
> reproduced the human recall gradient. Does a system trained only to predict the
> next sentence *discover* that policy? RSR is the test.

## Status

Sprint 1, week of 2026-09-17. **Only weeks 1–4 are approved** (spec §16);
everything downstream is a projection, not a permission.

## Read these first, in this order

| Document | Role |
|---|---|
| [`docs/spec-corrections.md`](docs/spec-corrections.md) | **Read before the spec.** 14 corrections that win over the spec body |
| [`docs/spec/rsr_model_spec_v0.5.md`](docs/spec/rsr_model_spec_v0.5.md) | The model specification, verbatim |
| [`CLAUDE.md`](CLAUDE.md) | Standing rules for every agent session |
| [`docs/decisions/`](docs/decisions/) | ADRs for anything expensive to reverse |
| [`docs/release-conditions.md`](docs/release-conditions.md) | §16's eight conditions, each linking its evidence |

The spec is on its fifth revision and carries three changelogs. Several passages
were superseded by a correction and never rewritten. `docs/spec-corrections.md` is
authoritative where the two disagree.

## Setup

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/pytest            # fast loop
.venv/bin/ruff check
```

PyTorch is the only project stack. **JAX is never installed here** — the TG
reference under `third_party/` is a pinned source reference plus a one-time tensor
extraction that runs on rented hardware. See [ADR-0001](docs/decisions/ADR-0001-tg-base.md).

## The two tests that matter

- **`tests/test_reduction.py`** — spec §3.7's reduction to exact TG, bit-exact,
  on every commit touching `src/rsr/`. Intra-repo by design: it guarantees that
  E3's FIFO and RSR arms differ only in the eviction rule.
- **`tests/test_fidelity.py`** — the PyTorch TG against golden tensors extracted
  once from the pinned JAX reference, to a tolerance committed in ADR-0002
  *before* the fixtures were generated. Gradients included, not just forwards.

## Layout

```
src/rsr/
  model/        tg base, memory, gestalt write
  retention/    policy, value_head, reward, shadow, bias
  baselines/    fifo, lru, h2o, expire_span, leading_edge, random, oracle
  data/         synthetic generator, pg19, coref
  metrics/      reintroduction, loo, vacuity, gini
  mup/          param groups, coordinate check
  constants.py  the registry that refuses unmeasured reads
configs/        base + model/ + data/ + experiment/
experiments/    e0a … e0i, each with run.py + RESULTS.md
preregistration/  committed BEFORE the experiment they govern
```

## Licence

Apache-2.0. `third_party/ThoughtGestaltCode` is Apache-2.0 upstream; see
`third_party/PINS.md`.
