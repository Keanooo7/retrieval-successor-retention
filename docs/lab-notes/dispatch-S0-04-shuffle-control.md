# S0-04 — the shuffle control, as a permanent instrument

**Baseline:** `c647af4290f7a45be2edf947b7973816dcc2af8b`. **Lane:** researcher · device. **Sprint:** 0 — **this is the sprint gate.**
**Depends on:** S0-01, S0-02, S0-03 all landed.

## Hypothesis (not instruction)

The strongest instrument this project has produced exists **in prose only**. `git grep shuffle`
over `src/`, `tests/`, `scripts/` and `experiments/` returns **nothing**. It was run by hand
once, during the 2026-09-18 audit, and it found the largest problem in the project: handing a row
**another document's entire 16-slot memory** moved the loss by **exactly 0.0** — max per-token delta
1.2e-4 nats, min −1.9e-4, across 384 sentences. Not a mean hiding a few tokens; **no single token
moved.** Against [P2]'s own ablation, removing working memory costs 29.8 → 45.8 PPL (+54%).

A thing that found that, and that is cheap, should not be a thing someone remembers to do.

## Files in scope

`src/rsr/train/loop.py` · `src/rsr/train/heartbeat.py` · `src/rsr/metrics/` (a new
`memory_liveness.py` is acceptable) · `tests/test_shuffle_control.py` (new).

## What to build

A control that runs **on every training run** and reports one number: the change in loss when each
row is given another row's memory. Report it in the heartbeat and in the ledger, with spread across
seeds.

🔴 **A run where memory contributes ≈0 is REFUSED, not filed.** Non-zero exit, and the exit code
must distinguish *"the control did not run"* from *"the control ran and memory is inert."* Use the
project's five-code taxonomy and **do not let 3 become 0** — `scripts/canary.py` already makes
exactly that mistake and is the cautionary example.

## Bar

Three, and the third is the one that matters:

1. **A positive control.** On a model with memory deliberately disabled, the shuffle delta must be
   ≈0 and the gate must go red. *A gate that has never been observed failing is not known to work.*
2. **A negative control.** On a model where memory is live, the delta must be non-zero and the gate
   green.
3. 🔑 **A mutation that reddens ONLY this gate.** If an existing test already catches what this
   catches, this gate adds nothing and **that is the finding** — report it and do not ship the gate.

## Done when — this is the Sprint 0 gate

**The shuffle control is green on a real training run**, with a stated non-zero delta and a spread.
🔴 **An sd of exactly 0.0000 across seeds is not a clean result, it is a broken one** — that failure
has already happened on this project and a reported 0.0000 is what exposed it.

If the control is wired correctly and the delta is **still** ≈0, that is not a failed brief. It is
the decisive result §10.3 asked for, it means the memory path itself is broken rather than the
objective, and **it stops Sprint 2 until it is understood.** Report it as such.

## Do NOT

- Do not start E1, E0d or E0h. Do not tune `γ_b` or `ν`. All of it sits above this gate.
- Do not weaken the threshold to make the gate pass.
- Do not report a KL-from-uniform figure as evidence about memory. It is **non-diagnostic and
  directionally backwards** — trained is 4.4× *farther* from uniform than untrained, monotone across
  every seed, and cycle 13's own ledger carries a row named `KL_IS_A_WEAK_DISCRIMINATOR`.

## Report

Standard block, `BRIEF ERRORS`, the delta with spread, both control outcomes, and which mutation
turns only this gate red.
