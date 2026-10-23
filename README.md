# Retrieval-Successor Retention (RSR)

Thought Gestalt (TG) is a recurrent transformer that compresses each sentence to
one vector and writes it to a working memory of `M` slots; later sentences reach
earlier ones only through cross-attention to that memory. When memory is full, TG
evicts the oldest slot.

**RSR replaces that one line** with a learned policy that evicts the slot with the
lowest predicted *future* retrieval demand given the current discourse state.

> Kintsch & van Dijk (1978) implemented and validated a retention rule for a
> capacity-limited discourse buffer, the leading-edge strategy, originally proposed
> by Kintsch & Vipond (1978). It walks the microstructure coherence graph built from
> argument overlap and "emphasizes recency and frequency"
> ([`docs/spec-corrections.md`](docs/spec-corrections.md), corrections 25–27). RSR
> asked whether a system trained only to predict the next sentence *discovers* that
> policy. **That question has not been tested** (see Status): the learned heads did
> not beat FIFO, and the leading-edge baseline is still a stub.

## Status

As of 2026-10-23. **Only weeks 1–4 are approved** (spec §16); everything downstream
is a projection, not a permission. The 2026-10-19 plan,
[`docs/PLAN-2026-10-19.md`](docs/PLAN-2026-10-19.md), is superseded by the owner's
rulings of 2026-10-20 and 2026-10-22, which are not yet committed here. On 2026-10-20
the owner upheld the kill rule for the original thesis: no further attempts on the
run 1 / run 2 / 2a configuration. The remaining work is a diagnostic paper. Every
number the paper may cite is re-derived from the run records, one row per number with
its ledger key, per-seed CI, seed count and PREREG status, in
[`docs/claims-register.md`](docs/claims-register.md).

**Four content-only value heads have been deployed frozen against FIFO, and none won
on any seed.** Two were fitted offline: offline-psi's head (labels EQUIV / EQUIV / LOSS,
deploy class FAILS; register row `ofpsi.deploy.label`) and the 2a ranking head (labels
LOSS / EQUIV / LOSS; [`experiments/rsr-run3-ranking/RESULTS-2a.md`](experiments/rsr-run3-ranking/RESULTS-2a.md)).
The other two were trained in the loop, below. Diagnostics of why (a sign check, a decodability probe, imitation-trained heads) are
on unmerged branches and are not reported here until they reach `main`.

**Two RSR runs have trained ψ̂ in the loop, and both lose to FIFO on every seed.** Each
trains the content-only head ψC online by `L_MC` (gradient to `φ` only) on fresh-stream
arm B's `ckpt3000` with the transformer frozen, on the synthetic corpus at `M = 16`,
3 seeds (0, 1, 2), and measures held-out all-query answer accuracy on 2048 EVAL
documents per seed. **Run 2's pre-registered class is FAILS** (LOSS on all 3 seeds).
Run 1 was descriptive, with no gate and so no class; its CI is below 0 on all 3 seeds.
The first two rows below give ψC − FIFO per seed with a paired per-document bootstrap
95% CI (2000 resamples); no interval is pooled across seeds. The other results were
measured under FIFO eviction, are model-free, or run offline closed-form ψ̂ fits as
fixed eviction rules (B2, the newcomer bake-off). None is a test of the hypothesis.

**Standing** means a committed test that runs on every commit. **Preliminary** means
three seeds on the synthetic corpus, read from a verified ledger, and not yet
replicated or written up. Each number is the ledger's own key, or a
[claims-register](docs/claims-register.md) row id, with the run's code commit and the
date the ledger landed.

| Finding | Number (3 seeds unless stated) | Status | Source · code commit · date |
|---|---|---|---|
| RSR run 1: ψC trained in the loop on the `r_i` target loses to FIFO | ψC − FIFO all-query accuracy, per seed [95% CI]: s0 -0.2053 [-0.2096, -0.2007]; s1 -0.2279 [-0.2323, -0.2236]; s2 -0.1683 [-0.1726, -0.1639]. Descriptive, no gate | Preliminary | run `rsr-run1`, row `run1.acc.psiC-fifo` · `ed6302a`; verifier `157f1fb` · 2026-10-06 |
| RSR run 2: ψC trained in the loop on the on-policy LOO return (ruling C3) loses to FIFO; class **FAILS** | ψC − FIFO all-query accuracy, per seed [95% CI]: s0 -0.0723 [-0.0761, -0.0685]; s1 -0.0453 [-0.0488, -0.0418]; s2 -0.0834 [-0.0872, -0.0794]. Labels LOSS / LOSS / LOSS | Preliminary | run `rsr-run2`, rows `run2.acc.psiC-fifo`, `run2.label` · `728a42f`; verifier `9c432b6` · 2026-10-08 |
| PyTorch TG matches the pinned JAX reference, gradients included | total loss bit-identical (`125.3105468750`); worst gradient uses 0.006 of a tolerance committed before the fixtures | Standing | `tests/test_fidelity.py` · `d03b37b`; tolerance in [ADR-0002](docs/decisions/ADR-0002-rented-hardware.md) Part B · `083d5e2` · 2026-09-17 |
| RSR reduces to TG when its terms are off (E0b, spec §3.7) | bit-exact | Standing | `tests/test_reduction.py` · `3e19dcb` · 2026-09-17 |
| An oracle beats FIFO on the synthetic corpus (E-feas, model-free) | `headroom_oracle_minus_fifo` 0.1807 ± 0.0123 at `M = 16` | Preliminary | run `efeas-synthetic` · `1fc8199` · 2026-09-20 |
| The memory is live under the masked objective, but training from scratch does not retrieve | `armB.ratio` per seed 6.7423 / 14.7307 / 6.5024 (row `ds.ratio.B`; unmasked arm A 4.66e-04 / 0.0036 / 0.0049, `ds.ratio.A`); arm B held-out answer NLL per seed 2.8697 / 2.8732 / 2.9702 nats against chance ln 16 ≈ 2.77 (`ds.answer_nll.B.answer_all`) | Preliminary | run `decisive-shuffle` · `90438f3` · 2026-09-21 |
| Memorise-then-stream retrieves on unseen documents; a fresh start does not by step 9000 | `classification` SCAFFOLD: R(3000) per seed, arm B 0.8598 / 0.8532 / 0.8799, arm A 0.0085 / 0.0000 / -0.0014, against δ 0.03 on every seed (rows `fs.R3000.B`, `fs.R3000.A`); `classification_reported_as` NO_ESCAPE by 9000: arm A R(9000) 0.0014 / 0.0014 / 0.0014 (`fe.R_at_P`) | Preliminary | runs `fresh-stream` · `e45e4fe` · 2026-09-25; `fresh-escape` · `cdaa871` · 2026-09-26 |
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
The headroom exists; `r_i` ranks the slot being retrieved *now* above its same-age
peers (E0d); an offline ψ̂ eviction rule was harmful through the model (B2); ψ̂
trained in the loop lost to FIFO in both RSR runs; and no deployed content-only value
head won against FIFO on any seed. Why the heads fail is not established on `main`, and
nothing here measures what connects those facts.

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
| [`docs/spec-corrections.md`](docs/spec-corrections.md) | **Read before the spec.** 32 corrections that win over the spec body |
| [`docs/spec/rsr_model_spec_v0.5.md`](docs/spec/rsr_model_spec_v0.5.md) | The model specification, verbatim |
| [`CLAUDE.md`](CLAUDE.md) | Standing rules for every agent session |
| [`docs/decisions/`](docs/decisions/) | ADRs for anything expensive to reverse |
| [`docs/release-conditions.md`](docs/release-conditions.md) | §16's eight conditions, each linking its evidence |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | The 2026-09-20 sequencing, read through the 2026-10-19 plan. Both are superseded (see Status); kept for the record |
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
  baselines/    fifo, lru, random, oracle,            [h2o, leading_edge = STUBS]
                expire_span (oracle: synthetic only)
  data/         synthetic generator                   [pg19, coref = STUBS]
  metrics/      memory_liveness (shuffle control),    [reintroduction,
                headroom (E-feas), loo (E0d), vacuity   gini = STUBS]
  mup/          param groups                          [coord_check = STUB]
  constants.py  the registry that refuses unmeasured reads
configs/        base + model/ + data/ + experiment/
experiments/    e0a … e0i, each with run.py + RESULTS.md
preregistration/  committed BEFORE the experiment they govern
```

**A stub raises `NotImplementedError`.** They are marked above rather than omitted
because the layout is also the work plan. Two baselines (H2O and the leading-edge
strategy), both anti-collapse mechanisms, and two of the spec's four metrics
(reintroduction and Gini) are not implemented. Leave-one-out, which §3.2.1 makes the
arbiter of truth, landed in #48 as E0d's instrument; the vacuity statistic and the
Expire-Span baseline are also implemented. Two further instruments were added since:
the shuffle control (`runs/shuffle-control/`) and E-feas's retention headroom
(`runs/efeas-synthetic/`). See
`docs/RESEARCH-CONTEXT.md` §10 for the full register, and read it before reading
any result in `runs/`.

## Provenance

Developed with AI coding agents under my design and review. Of the 112 non-merge
commits on `main` as of 2026-09-21, 111 carry a `Co-Authored-By` trailer naming one
(`git log --no-merges --format=%B | grep -i '^co-authored-by'`). Research direction,
pre-registrations and the decision to accept or reject a result were human calls; the
agents wrote most of the code and ran most of the experiments. A return is accepted only
after part of it is re-executed by a separate session (`.claude/agents/rsr-verifier.md`).
The development history is complete in this repository.

## Licence

Apache-2.0. `third_party/ThoughtGestaltCode` is Apache-2.0 upstream; see
`third_party/PINS.md`.
