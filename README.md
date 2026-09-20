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
| [`docs/RESEARCH-CONTEXT.md`](docs/RESEARCH-CONTEXT.md) | **Start here.** The single orientation document — claim, sources, decisions, measured numbers, what is broken, what no agent may touch |
| [`docs/spec-corrections.md`](docs/spec-corrections.md) | **Read before the spec.** 30 corrections that win over the spec body |
| [`docs/spec/rsr_model_spec_v0.5.md`](docs/spec/rsr_model_spec_v0.5.md) | The model specification, verbatim |
| [`CLAUDE.md`](CLAUDE.md) | Standing rules for every agent session |
| [`docs/decisions/`](docs/decisions/) | ADRs for anything expensive to reverse |
| [`docs/release-conditions.md`](docs/release-conditions.md) | §16's eight conditions, each linking its evidence |
| [`docs/cognitive-grounding.md`](docs/cognitive-grounding.md) | What the cognitive literature licenses, and what it does not. Read before writing anything that cites McClelland or Kintsch |
| [`docs/citation-audit.md`](docs/citation-audit.md) | E0f. 13 of 14 primary sources checked; **[P11], the cognitive claim, did not survive** |

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

PyTorch is the only project stack. **JAX is not a project dependency and never
enters `pyproject.toml`** — the TG reference under `third_party/` is a pinned,
read-only source reference. The one-time golden-tensor extraction has run, on this
machine's CPU, in a throwaway venv; the exact command is in
[`third_party/PINS.md`](third_party/PINS.md). See
[ADR-0001](docs/decisions/ADR-0001-tg-base.md) and
[ADR-0007](docs/decisions/ADR-0007-all-training-on-the-mac-studio.md).

**No GPU is rented.** Everything runs on the Mac Studio (ADR-0007).

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
  retention/    policy, value_head, reward           [shadow, bias = STUBS, raise]
  baselines/    fifo, lru, random                     [h2o, expire_span,
                                                       leading_edge, oracle = STUBS]
  data/         synthetic generator                   [pg19, coref = STUBS]
  metrics/      -- ALL FOUR ARE STUBS: reintroduction, loo, vacuity, gini
  mup/          param groups                          [coord_check = STUB]
  constants.py  the registry that refuses unmeasured reads
configs/        base + model/ + data/ + experiment/
experiments/    e0a … e0i, each with run.py + RESULTS.md
preregistration/  committed BEFORE the experiment they govern
```

**A stub raises `NotImplementedError`.** They are marked above rather than omitted
because the layout is also the work plan. Four of the eight baselines, both
anti-collapse mechanisms, and every metric -- including leave-one-out, which
§3.2.1 makes the arbiter of truth -- are not implemented. See
`docs/RESEARCH-CONTEXT.md` §10 for the full register, and read it before reading
any result in `runs/`.

## Licence

Apache-2.0. `third_party/ThoughtGestaltCode` is Apache-2.0 upstream; see
`third_party/PINS.md`.
