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

Sprint 2, as of 2026-09-30 (`main` at `f7a6b10`; the newest results sit on
`night/2026-09-30` at `898628a`, and E0d and B5 on their own run branches). **Only
weeks 1–4 are approved** (spec §16); everything downstream is a projection, not a
permission.

**No RSR policy has been trained yet: no ψ̂ has been trained in the loop (by `L_MC`
in `train()`).** Most results below were measured under FIFO eviction or are
model-free. The two newest (B2 and the newcomer bake-off) run *offline*, closed-form
ψ̂ fits as fixed eviction rules through the model. All are on the synthetic corpus at
`M = 16`, and none is a test of the hypothesis: they establish whether the instrument
is sound, whether the question has room, and what an untrained-in-the-loop ψ̂ does.

**Standing** means a committed test that runs on every commit. **Preliminary** means
three seeds on the synthetic corpus, read from a verified ledger, and not yet
replicated or written up. Each number is the ledger's own key, with the run's code
commit and the date the ledger landed.

| Finding | Number (3 seeds unless stated) | Status | Source · code commit · date |
|---|---|---|---|
| PyTorch TG matches the pinned JAX reference, gradients included | total loss bit-identical (`125.3105468750`); worst gradient uses 0.006 of a tolerance committed before the fixtures | Standing | `tests/test_fidelity.py` · `d03b37b`; tolerance in [ADR-0002](docs/decisions/ADR-0002-rented-hardware.md) Part B · `083d5e2` · 2026-09-17 |
| RSR reduces to TG when its terms are off (E0b, spec §3.7) | bit-exact | Standing | `tests/test_reduction.py` · `3e19dcb` · 2026-09-17 |
| An oracle beats FIFO on the synthetic corpus (E-feas, model-free) | `headroom_oracle_minus_fifo` 0.1807 ± 0.0123 at `M = 16` | Preliminary | run `efeas-synthetic` · `1fc8199` · 2026-09-20 |
| The memory is live under the masked objective, but training from scratch does not retrieve | `armB.ratio` 9.33 ± 4.68 (unmasked: 0.0030); held-out answer NLL 2.90 ± 0.06 against chance ln 16 ≈ 2.77 | Preliminary | run `decisive-shuffle` · `90438f3` · 2026-09-21 |
| Memorise-then-stream retrieves on unseen documents; a fresh start does not by step 9000 | `classification` SCAFFOLD; `classification_reported_as` NO_ESCAPE by 9000 | Preliminary | runs `fresh-stream` · `e45e4fe` · 2026-09-25; `fresh-escape` · `cdaa871` · 2026-09-26 |
| The retention target `r_i` barely separates pending facts from filler (FIFO-world, hindsight proxy; no RSR head trained) | mean demand 0.071 / 0.069 / 0.079 (pending) vs 0.057 / 0.061 / 0.054 (filler); a `γ = 0` rule on it hits 0.79–0.85 against FIFO's 0.81 and a causal fact/filler rule's 0.97 | Preliminary | run `lookahead-room-r2`, keys `B.ckpt3000.*` · `a0cc85a` · 2026-09-26 |
| `r_i` agrees with leave-one-out Δloss **at the moment of retrieval** (E0d, spec §3.2.1). It does not show that `r_i` predicts later demand | `e0d.class` AGREE; `primary.AUROC_strat_pct` 0.9423 / 0.9313 / 0.9433 against A\* = 0.85 | Preliminary | run `e0d` · `2fd9368` · 2026-09-29 |
| An offline ψ̂ eviction rule is harmful through the model (B2; closed-form ridge fit, not a trained head) | `classification` row 1, HARMFUL: ψ̂-U LOSS / LOSS / EQUIV, ψ̂-C EQUIV ×3 | Preliminary | run `b2-psi-probe` · `80c1cd7` · 2026-09-29 |
| Hard grace does not rescue it; the harm travels with the counterfactual target (newcomer bake-off, offline) | `decision.DQ1` NO; `decision.DQ2` INTERMEDIATE; `decision.DQ3` YES | Preliminary | run `newcomer-bakeoff` · `16f867a` · 2026-09-30 |
| `β` is not inert under AdamW at `ε = 1e-8` with `φ`-only gradient (B1, open-loop) | `verdict.outcome` falsified: decoupled_wd LIVE at 1e-8, INERT at 1e-12 | Preliminary | run `b1-beta-inertness` · `412f136` · 2026-09-30 |
| Arm B's stream loss has not plateaued by step 9000, and retrieval holds throughout (B5; FIFO only) | `classification_reported_as` NO_PLATEAU by 9000; `B.ckpt9000.R_quantity` 0.8683 / 0.8546 / 0.8573 | Preliminary | run `b5-convergence` · `b9595e0` (the ledger records `dirty: true`; the code ran at `a84f78d`, and `a84f78d`→`b9595e0` changes no code) · 2026-09-30 |

Ledgers are at `runs/<run id>/ledger.json`, beside the verifier's
`verification.json`. The rows dated 2026-09-29/30 are generated, with their keys,
verifier verdicts and review records, in
[`docs/results/2026-09-30-verified.md`](docs/results/2026-09-30-verified.md)
(`scripts/verified_results.py`; a test fails if that file and the ledgers disagree).
The open question has narrowed but not closed. The headroom exists; `r_i` ranks the
slot being retrieved *now* above its same-age peers (E0d); and an offline ψ̂ eviction
rule was harmful through the model (B2). Nothing here measures what connects those
facts. Whether a ψ̂ trained in the loop behaves differently has not been measured.

**Where to find the audit trail**

- **Shuffle control and its live-decoy amendment:** [`experiments/shuffle-control/PREREG.md`](experiments/shuffle-control/PREREG.md), Amendment 1. A known-live memory passed the original "inert" rule, so the rule was amended before the real run.
- **Audit of the 2026-09-18 overnight run**, including the arm stamped `rsr` that ran FIFO: [`docs/RESEARCH-CONTEXT.md`](docs/RESEARCH-CONTEXT.md) §10.1 and §10.5.
- **Retracted numbers:** [`docs/RESEARCH-CONTEXT.md`](docs/RESEARCH-CONTEXT.md) §11.
- **Oracle before any policy:** [`experiments/efeas/`](experiments/efeas/). Headroom was measured before any retention policy was trained.
- **Everything else:** what is measured, broken and open, in [`docs/RESEARCH-CONTEXT.md`](docs/RESEARCH-CONTEXT.md) §6 and §9–§11.

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
