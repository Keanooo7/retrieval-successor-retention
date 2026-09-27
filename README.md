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

Sprint 2, as of 2026-09-27 (`main` at `941a68e`). **Only weeks 1–4 are approved**
(spec §16); everything downstream is a projection, not a permission.

**No RSR policy has been trained yet.** Every result below was measured under FIFO
eviction or is model-free, on the synthetic corpus at `M = 16`. None is evidence
for or against the hypothesis; they establish that the instrument is sound and
that the question has room.

| Finding | Number | Source |
|---|---|---|
| PyTorch TG matches the pinned JAX reference, gradients included | total loss bit-identical; worst gradient uses 0.006 of a tolerance committed before the fixtures | `tests/test_fidelity.py`, ADR-0002 |
| RSR reduces to TG when its terms are off (E0b, §3.7) | bit-exact | `tests/test_reduction.py` |
| An oracle beats FIFO on the synthetic corpus (E-feas) | headroom 0.1807 ± 0.0123 | `runs/efeas-synthetic/ledger.json` |
| The memory is live under the masked objective, but training from scratch does not retrieve | shuffle ratio 9.33 ± 4.68; held-out answer NLL 2.90 against chance 2.77 | `runs/decisive-shuffle/ledger.json` |
| Memorise-then-stream retrieves on unseen documents; a fresh start does not by step 9000 | classification SCAFFOLD; NO_ESCAPE | `experiments/fresh-stream/`, `experiments/fresh-escape/` |
| The retention target `r_i` barely separates pending facts from filler | demand ≈ 0.07 vs ≈ 0.06; a `γ = 0` rule on it hits 0.80–0.85 against FIFO's 0.81 and a causal rule's 0.97 | `runs/lookahead-room-r2/ledger.json` |

The last row is the open question: the headroom exists, and the signal RSR learns
from points at it only weakly. E0d (`r_i` against leave-one-out Δloss) tests it
next. The full register, including what is broken and what was retracted, is
[`docs/RESEARCH-CONTEXT.md`](docs/RESEARCH-CONTEXT.md) §6, §9–§11.

## Read these first, in this order

| Document | Role |
|---|---|
| [`docs/RESEARCH-CONTEXT.md`](docs/RESEARCH-CONTEXT.md) | **Start here.** The single orientation document — claim, sources, decisions, measured numbers, what is broken, what no agent may touch |
| [`docs/spec-corrections.md`](docs/spec-corrections.md) | **Read before the spec.** 31 corrections that win over the spec body |
| [`docs/spec/rsr_model_spec_v0.5.md`](docs/spec/rsr_model_spec_v0.5.md) | The model specification, verbatim |
| [`CLAUDE.md`](CLAUDE.md) | Standing rules for every agent session |
| [`docs/decisions/`](docs/decisions/) | ADRs for anything expensive to reverse |
| [`docs/release-conditions.md`](docs/release-conditions.md) | §16's eight conditions, each linking its evidence |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | **What happens next and in what order.** Supersedes §8's week numbering. Names the blocker that gates four experiments at once |
| [`docs/cognitive-grounding.md`](docs/cognitive-grounding.md) | What the cognitive literature licenses, and what it does not. Read before writing anything that cites McClelland or Kintsch |
| [`docs/citation-audit.md`](docs/citation-audit.md) | E0f. 13 of 14 primary sources checked. **The spec misstates [P11], Kintsch & van Dijk, in three places** (corrections 25–27); the hypothesis is untested, the spec's reading of the source is what failed |

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
  baselines/    fifo, lru, random, oracle             [h2o, expire_span,
                (oracle: synthetic only)                leading_edge = STUBS]
  data/         synthetic generator                   [pg19, coref = STUBS]
  metrics/      memory_liveness (shuffle control),    [reintroduction,
                headroom (E-feas), loo (E0d)            vacuity, gini = STUBS]
  mup/          param groups                          [coord_check = STUB]
  constants.py  the registry that refuses unmeasured reads
configs/        base + model/ + data/ + experiment/
experiments/    e0a … e0i, each with run.py + RESULTS.md
preregistration/  committed BEFORE the experiment they govern
```

**A stub raises `NotImplementedError`.** They are marked above rather than omitted
because the layout is also the work plan. Three baselines, both anti-collapse
mechanisms, and three of the spec's four metrics are not implemented. The fourth,
leave-one-out, which §3.2.1 makes the arbiter of truth, landed in #48 as E0d's
instrument. Two further instruments were added since: the shuffle control
(`runs/shuffle-control/`) and E-feas's retention headroom (`runs/efeas-synthetic/`). See
`docs/RESEARCH-CONTEXT.md` §10 for the full register, and read it before reading
any result in `runs/`.

## Provenance

Developed with AI coding agents under my design and review. Of the 112 non-merge
commits on `main` as of 2026-09-21, 111 carry a `Co-Authored-By` trailer naming one
(`git log --no-merges --format=%B | grep -i '^co-authored-by'`). Research direction,
pre-registrations and the decision to accept or reject a result were human calls; the
agents wrote most of the code and ran most of the experiments. A return is accepted only
after part of it is re-executed by a separate session (`.claude/agents/rsr-manager.md`).
The development history is complete in this repository.

## Licence

Apache-2.0. `third_party/ThoughtGestaltCode` is Apache-2.0 upstream; see
`third_party/PINS.md`.
