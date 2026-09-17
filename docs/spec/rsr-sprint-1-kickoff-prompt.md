# RSR — Sprint 1 kickoff prompt

*Paste everything below the rule into a fresh Claude Code session in an empty directory. Scope is repo scaffolding + Sprint 1 (week 1) only.*

---

You are scaffolding a research repository and executing its first week of work. I am the sole owner; you and other agents are the implementation layer. I direct, you build.

## Repo

Create a git repo named `retrieval-successor-retention` (directory `retrieval-successor-retention/`, initial branch `main`). Python 3.11+, PyTorch, `uv` for dependency management, `ruff` + `pytest` in CI. Target hardware for this sprint is a 64 GB Mac Studio on MPS; all code must also run on CUDA without modification.

## The project, in one paragraph

Thought Gestalt (TG) is a recurrent transformer that compresses each sentence to one vector and writes it to a working memory of `M` slots; later sentences reach earlier ones only through cross-attention to that memory. When memory is full, TG evicts the oldest slot. **Retrieval-Successor Retention (RSR) replaces that one line** with a learned policy that evicts the slot with the lowest predicted *future* retrieval demand given the current discourse state. The scientific question is whether a system trained only to predict the next sentence rediscovers Kintsch & van Dijk's 1978 leading-edge strategy. The full specification is `rsr_model_spec_v0.5` — I will place it at `docs/spec/rsr_model_spec_v0.5.md`. Read it end to end before writing code.

## Authority — read this before the spec

The spec is on its fifth revision and carries three changelogs. Several passages were superseded by a correction and never rewritten. **These corrections win over the spec body. Do not "resolve" them yourself — they are already resolved here:**

1. **`K = 64` on PG-19, `K = 40` on synthetic.** §3.4's closing line still says `K = M`; the self-audit three paragraphs above it and the §4.5 table both override that. `K` is the shadow-buffer depth and must be ≥ the max target gap.
2. **Monte-Carlo (`λ = 1`) is the default return; TD(0) is ablation A8.** §3.3 still composes `L = L_NTP + β·L_TD` and forbids backprop of `L_TD`. Read that as `L_MC` on the default path. The no-backprop rule applies to whichever retention loss is active.
3. **No EMA target copy of `φ` on the default path.** §3.4's "if the TD residual misbehaves, add an EMA target copy" is live only inside A8. MC has no bootstrap.
4. **§3.5 is headed "Three corrections" and lists four, numbered 1, 2, 4, 3.** All four apply. Item "4" (`γ_b` as a timescale) is the one that matters: `γ_b ≈ b_max / (0.25 · E[lifetime])`, order 0.05–0.1, **derived in E0e, never frozen at 0.001**.
5. **§0's "weeks 1–4 were approved unchanged; E0a–E2 are unaffected" is false under v0.5.** E0h and E0i are new and both are pre-gate. E0e now also produces `γ_b` and the EMA half-life. E1 now includes a `ν` sweep.
6. **§10.1's "that cost is not in the §4.1 budget" is superseded by D-5**, which funded the E7 small-`M` model.

Record all six in `docs/spec-corrections.md` on the first commit, with the section each corrects. If you find another passage the changelog superseded, add it there and tell me — do not silently pick a reading.

## Scope boundary — hard

**Only weeks 1–4 are approved (§16). Everything downstream is a projection, not a permission.** This prompt covers Sprint 1 (week 1) and the scaffolding that serves weeks 1–4.

You will **not**, in this sprint:

- Implement §3.2–3.5 beyond the minimum E0a needs (that is Sprint 2). Stub the rest against the interfaces below.
- Train on PG-19 or WikiText, or run anything at `S = 80` beyond E0c's profiling steps.
- Reserve or pay for GPU capacity. Week 1 gets a **named provider, a written quote, and a cancellable hold with a movable start date** — nothing billable. The reservation *is* the four-figure commitment §16 declines to approve; a hold satisfies §16 condition 8 without pre-empting the gate.
- Spend more than ~$100 of compute total. That is the approved envelope for weeks 1–4.

## Task 0 — before anything else

**Determine whether [P2]'s Thought Gestalt implementation is publicly available.** The spec treats TG as given ("Base model (unmodified TG) **[E]**") and budgets zero hours for building it. [P2] is arXiv:2512.25026v2, under review. Check the arXiv listing, the authors' GitHub, and any linked artifact.

Write the answer to `docs/decisions/ADR-0001-tg-base.md` with evidence, and **stop and report before proceeding** if the answer is no. If TG's code is not available, reproducing it to the point where E0b's bit-exactness test means anything is 3–6 weeks that exist in no schedule, and I need to decide how to handle that before you write another line.

If TG's code *is* available: vendor it under `src/rsr/model/tg/`, pin the commit, record the pin in the ADR, and reproduce its reported numbers (29.8 test PPL, 21 sentence-steps/sec on one A40 at `S ≈ 30`) as a smoke test before trusting anything built on it.

## Scaffolding

```
retrieval-successor-retention/
├── CLAUDE.md                     # standing rules for every agent session
├── README.md
├── pyproject.toml
├── src/rsr/
│   ├── model/                    # tg base, memory, gestalt write
│   ├── retention/                # policy, value_head, reward, shadow, bias  (mostly stubs)
│   ├── baselines/                # fifo, lru, h2o, expire_span, leading_edge, random, oracle (stubs)
│   ├── data/                     # synthetic generator, pg19, coref
│   ├── metrics/                  # reintroduction, loo, vacuity, gini
│   ├── mup/                      # param groups, coordinate check
│   ├── constants.py              # the registry described below
│   └── cli.py
├── configs/                      # base + model/ + data/ + experiment/ (one per E0*, E-feas, E1, E2)
├── experiments/e0a … e0i/        # run.py + RESULTS.md per experiment
├── preregistration/              # committed BEFORE the experiment they govern
├── docs/
│   ├── spec/                     # the v0.5 spec, verbatim
│   ├── spec-corrections.md
│   ├── constants.md
│   ├── release-conditions.md     # §16, eight checkboxes, each linking its evidence
│   └── decisions/                # ADRs
├── tests/
└── .github/workflows/ci.yml
```

Three scaffolding decisions are load-bearing. Get them right now and the rest of the project is cheap.

**1. One `RetentionPolicy` protocol.** Every baseline and RSR implement the same interface:

```python
class RetentionPolicy(Protocol):
    def select_eviction(self, slots: MemoryState, context: Tensor, step: int) -> int: ...
    def observe(self, slots: MemoryState, attn: AttentionTrace, step: int) -> None: ...
    def reset(self) -> None: ...
```

FIFO, LRU, H2O, Expire-Span, leading-edge, random, oracle and RSR are then interchangeable by config. **§3.7's reduction to exact TG must be reachable by configuration alone** — `ψ̂ ≡ −a_i, b ≡ 0, ν = 0, β = 0, A_max = M, T_warm = ∞`, shadow buffer off — not by a separate code path. Every term added to the eviction score needs a documented off-switch or §3.7 stops being a reduction and E0b stops testing what it claims to test.

**2. A constants registry that refuses unmeasured reads.** `src/rsr/constants.py` classifies every constant as `FROZEN`, `MEASURED`, or `DERIVED`, per §4.5. Reading a `MEASURED` or `DERIVED` constant before its source experiment has logged a value **raises**, with a message naming the experiment. This makes "never freeze an unmeasured constant" a mechanical guarantee instead of a discipline. Seed it with:

| Constant | Class | Source |
|---|---|---|
| `M` | FROZEN | 16 synthetic · 40 corpora · 8 for E7 (§5.1) |
| `S` | FROZEN per experiment | 30 / 48 / 80 / 30→+12 (§5.2) |
| `K` | DERIVED | `≥ max target gap` → 64 PG-19, 40 synthetic |
| `b_max` | FROZEN | 1.0 |
| `λ` | FROZEN | 1.0 (Monte Carlo) |
| `λ_shadow` | FROZEN | 0.5 |
| `T_warm` | FROZEN | one epoch, reported as a fraction of total epochs |
| `τ` | **MEASURED** | E0e |
| `E[lifetime]` | **MEASURED** | E0e |
| `γ_b` | **DERIVED** | `b_max / (0.25 · E[lifetime])`, from E0e |
| `γ`, `β`, `ν` | **MEASURED** | frozen at the week-4 gate from E1 |
| `A_max` | CONDITIONAL | `A_max ≤ S`; swept only where `S` makes it bind |

**3. E0b is a CI test, not a script.** `tests/test_reduction.py` asserts a bit-exact loss curve against stock TG under the §3.7 settings, and runs on every commit that touches `src/rsr/`. Construct `φ` **after** the model, or seed it from a separate RNG stream — otherwise instantiating the value head consumes draws, shifts data order and dropout masks, and the test fails for a reason unrelated to the mechanism. Write that reason into the test's docstring so nobody debugs it as a mechanism failure.

## Sprint 1 tasks — week of 2026-09-17

Sprint goal: **kill the project cheaply this week if it deserves killing.** Four of the nine kill gates land here.

| # | Task | Exit criterion |
|---|---|---|
| **T1** | Scaffold the repo, CLAUDE.md, CI, constants registry, policy protocol | CI green; `pytest` passes; `docs/spec-corrections.md` committed |
| **T2** | **Pre-register E0i's threshold** | `preregistration/e0i_threshold.md` committed **before** any histogram is computed, stating the minimum events-per-bucket in `(40, 64]` that clears the gate, and the reasoning. The spec calls this pre-registered and never gives the number. Registering it after seeing the data makes the gate unfalsifiable, and this gate protects 576 GPU-hours. |
| **T3** | **E0i** — coref/mention pipeline over the PG-19 30M-token subset, CPU. Histogram of reintroduction gaps and events-per-bucket at `S = 80` | Histogram vs the T2 threshold. **Plus: hand-annotate 100 reintroductions and report the coref system's precision/recall.** The spec's D-2 names coref as "an unnamed dependency with its own error rate feeding straight into the dependent measure," then prescribes a *population* check that cannot see precision. Half a day; closes the hole. |
| **T4** | **E0g** — name and obtain the E7 stimulus set | Candidates in spec order of preference: naturalistic free-recall corpora with event-level annotations (Chen et al. 2017 *Sherlock* recall data; the *Narratives* collection), then Thorndyke (1977), then Kintsch & van Dijk protocols. Report availability, licensing, and **whether importance is scored independently of serial position**. Treat "identified" and "in hand" as two separate exits — start any data-use agreement on day 1, because that is calendar time nobody controls. If nothing suitable exists, E7 is cut and falsifier 4 is withdrawn (§16 condition 4). **Report what was found regardless of outcome.** |
| **T5** | **E0c** — peak resident memory and throughput at `S ∈ {30, 50, 80}` × `d ∈ {128, 256}`, `M = 40` | A committed `(S, d, batch)` triple that fits 64 GB. **If `S = 80` does not fit, `S` wins and `d` is cut** — the experiment is defined by `S`; width is only hygiene. Fallback if it cannot be made to fit: `d = 128`, then `A_max = 40` targeting gaps `(16, 40]` at `M = 16`. Decide now, not in week 6. |
| **T6** | **E0d** — validate `r_i` against leave-one-out Δloss on a held-out subsample | `r_i(t) = Σ_{l,h} ‖α · W_O v‖₂`, normalized across live slots. **`W_O` is not optional** — it is where head-specific rescaling lives. **Report the per-layer profile once before collapsing to a scalar**; memory gates and cross-attention gradient share are depth-stratified, so the layer sum is not obviously right. Report Spearman ρ(`r_i`, LOO Δloss). **If they disagree, LOO is truth and `r_i` is a confound**, and §3.4's redundancy term is the specified response. Build the LOO harness as the shared oracle — E-feas and the shadow buffer both reuse it. |
| **T7** | **E0a** — μP coordinate check, run twice: bare TG, then TG + value head | Activation RMS width-invariant through the recurrence across `d ∈ {128, 192, 256, 384}`. **Check `ψ̂`'s scalar output coordinate scale specifically**, not only transformer activations — this is the failure that otherwise surfaces in week 9 with no error message. The bilinear head needs an explicit **`1/d` multiplier** on its output and its own μP parameter group: `W` (`d×d`) init `Var = 1/d`, hidden-matrix LR rule; `u` (`2d→1`) init `Var = 1/fan_in`, output rule. Do **not** justify the multiplier with "the form sums `d²` terms and scales as `Θ(d)` under standard init" — at init it is `Θ(√d)`; `Θ(d)` is the *correlated* regime. Same prescription, wrong argument, and the wrong argument gives the wrong multiplier for a differently-shaped head. |
| **T8** | GPU procurement | A named provider, a written quote, and a **cancellable hold with a movable start date** for ≥4 GPUs from week 6. Not week 5 — see the schedule note below. Nothing billable. |
| **T9** | **GATE-1 report** | `experiments/GATE-1.md`: every exit criterion above marked pass/fail with its evidence link, plus the §16 condition-4 decision on E7. |

T7 needs a value head to exist in week 1 while §3.2–3.5's implementation is week 2 work. Build **only** the bilinear estimator `ψ̂_φ(s_i, c_t) = s_iᵀWc_t + uᵀ[s_i; c_t]` with its μP group and multiplier. Leave reward, return, shadow buffer, warmup, redundancy and bias loop as typed stubs that raise `NotImplementedError`. T6 needs a briefly-trained TG at `S = 30` — a short run, not a corpus run.

## Prohibitions — honor these literally

- **Do not fill §15.** It reads: "*This section is a placeholder and must not be filled by a reviewer, an advisor, or a model.*" You are a model. If you find yourself drafting a candidate for §15.2, stop. You may schedule the work; you may not do it. Do not restate the pointer §15 leaves open, and do not suggest answers to it.
- **Do not backpropagate the retention loss into the transformer or `W_sent`.** Only `φ` receives gradient. `c_t` enters `ψ̂` with a stop-gradient on the transformer side.
- **Age is excluded from `ψ̂`.** Supplying it invites collapse onto recency and makes the vacuity failure mode invisible rather than merely possible. An age-only head and a content+age head exist as separate baseline arms.
- **`b` never enters `ψ̂`** or any differentiable path — eviction argmin only. Balance is a zero-gradient control loop, not a competing objective.
- **Make no scaling claim anywhere.** E5 is hygiene; a 3× width range at ≤21M parameters cannot resolve slope from intercept.
- **Do not call a truncated-BPTT window "consolidation."** The CLS mapping was inverted and the claim is deleted.
- **Do not down-weight underfull steps as the bias correction.** Rescale the target: `r_i(t) = share_i(t) · |memory_t| / M`. Down-weighting a biased target reduces how hard the estimator fits it; it does not remove the bias. And do not call the rescaled target "unbiased" — it credits absent competitors and biases stream-initial slots downward. That is the honest description and "unbiased" is the word a reviewer will test.
- **Delete §1's [P5]–[P14] verification note from the spec copy once E0f is logged** (Sprint 2). It cannot still be there at week 12.

## Two schedule facts to build against

**The week-4 gate is the milestone whose slip slips everything**, and not because dates move. Weeks 5–7 are exactly 504 wall-clock hours and must hold 842 GPU-hours of work — E3's 576, A2/A4's 192, the E7 `M = 8` model's 72, A1's 2. At 65% utilization with a 25% rerun margin that is 3.21 GPUs, which breaks the spec's stated floor of 3; hold the floor at 4. A one-week slip leaves 336 hours for the same 842 and pushes the requirement to 4.82 GPUs, or drops E3 from three seeds to two. That is why T8 buys a hold with a movable start date.

**Week 4 in the spec's timeline is two weeks of work.** It carries the E1/E2 gate *and* §15.1's rederivation and §15.2 — roughly 54 hours I cannot delegate to you, on top of a full experimental week. Plan for the gate to fall at the end of week 5, and set the T8 hold's start date to week 6 accordingly. Do not compress §15 to protect the calendar; it is release condition 7 and it is the part that decides whether the project is worth finishing.

## How to work

Small commits, conventional messages, one logical change each. Open an ADR in `docs/decisions/` for any choice that would be expensive to reverse — vendoring vs reimplementing TG, the coref tool, the config framework. Every experiment writes a `RESULTS.md` next to its `run.py` with the command that produced it, the git SHA, the hardware, and the numbers — including the ones that came out wrong.

Report the result of **Task 0** before you do anything else. Then scaffold, then work the table in order, and stop at T9 with the GATE-1 report. Do not start Sprint 2.
