# CLAUDE.md — Retrieval-Successor Retention

Standing rules for every agent session in this repo. Read this before the spec.

## Read order — do not skip 2

1. `docs/spec-corrections.md` — **corrections that OVERRIDE the spec body.** The spec is on its
   fifth revision; several passages were superseded by a correction and never rewritten. Reading the
   spec without this file produces code that is wrong in ways the spec itself already knows about.
2. `docs/citation-audit.md` — what has and has not been verified against primary sources.
3. `docs/spec/rsr_model_spec_v0.5.md` — the spec.
4. `docs/decisions/` — ADRs. `ADR-0001` blocks three of week 1's gates; read it first.
5. `docs/gates.md` — the ratchets and what their exit codes mean.

## Scope — hard

**Only weeks 1–4 are approved (§16). Everything downstream is a projection, not a permission.**

Do not, without the owner saying so in this session:
- train on PG-19 or WikiText, or run anything at `S = 80` beyond E0c's profiling steps
- reserve or pay for GPU capacity — week 1 gets a **cancellable hold**, nothing billable
- spend more than **~$100 of compute total**, which is the whole envelope for weeks 1–4

## Prohibitions — honor these literally

- 🔴 **Do not fill §15.** It reads: *"This section is a placeholder and must not be filled by a
  reviewer, an advisor, or a model."* You are a model. If you find yourself drafting a candidate for
  §15.2, **stop.** You may schedule the work; you may not do it. Do not restate the pointer §15
  leaves open, and do not suggest answers to it. This is not a style preference — the section exists
  because every correction in this document arrived from outside, and one supplied by an agent would
  reproduce that failure one level up.
- **Do not backpropagate the retention loss into the transformer or `W_sent`.** Only `φ` receives
  gradient. `c_t` enters `ψ̂` with a stop-gradient on the transformer side.
- **Age is excluded from `ψ̂`.** Supplying it invites collapse onto recency and makes the vacuity
  failure mode invisible rather than merely possible. Age-only and content+age heads are separate
  baseline arms (A3), never this one.
- **`b` never enters `ψ̂`** or any differentiable path — eviction argmin only.
- **Make no scaling claim anywhere.** E5 is hygiene.
- **Do not call a truncated-BPTT window "consolidation."** The CLS mapping was inverted and the
  claim is deleted.
- **Do not down-weight underfull steps as the bias correction.** Rescale the target:
  `r_i(t) = share_i(t) · |memory_t| / M`. And do not call the rescaled target "unbiased" — it
  credits absent competitors and biases stream-initial slots downward.
- **Do not justify the μP multiplier with the `Θ(d)`-at-init argument.** At init the bilinear form
  is `Θ(√d)`. Measured: `experiments/e0a/RESULTS.md`.

## How work is verified here

- 🔴 **Literal command output, or it did not happen.** "Tests pass" is not a result;
  `41 passed, 1 skipped` is. A gate you skipped is a gate that failed.
- 🔴 **A new gate is not believed green until a mutation has shown it red** — and the mutation must
  redden *only* it. If nothing reddens it, the gate adds nothing, and **that is the finding.**
  Evidence lives in `docs/gates.md`.
- 🔴 **"Did not run" is not "found nothing."** Exit 3 is not a pass. See `docs/gates.md`.
- 🔴 **Never report a deferred gate as a pass.** Deferred, with a named blocker, is an honest and
  useful result. A pass that was not measured is neither.
- **A negative result must say where it searched.** "Found nothing" is unfalsifiable;
  "`grep -iE 'github|we release' <file>` over 20 pages returned 1 hit, and it was the word
  'reproduce' in an ablation" is checkable.
- **Numbers are measured, never typed.** `rsr floor` runs the suite and parses pytest's own summary.
  A human retyping a count into a command is the step where one digit changes.
- **Every experiment writes `RESULTS.md`** next to its `run.py`: the command, the git SHA, the
  hardware, the seeds, and the numbers — **including the ones that came out wrong.**

## The three invariants

1. **§3.7's reduction to exact TG is reachable by configuration alone**, never by a separate code
   path. Every term added to the eviction score needs an off-switch in `RetentionConfig`, and
   `tests/test_reduction.py::test_every_score_term_has_an_off_switch` fails if you add one without.
2. **`src/rsr/constants.py` refuses unmeasured reads.** A `MEASURED`/`DERIVED`/`CONDITIONAL`
   constant read before its source experiment logged a value raises, naming the experiment.
   `rsr constants` shows the table; the unset count is a ratchet.
3. **One `RetentionPolicy` protocol for every arm**, so no comparison depends on which code path ran.

## Commands

```bash
uv sync                      # set up
uv run pytest -q             # the suite
uv run ruff check src tests experiments
uv run rsr constants         # the registry, and what is still unset
uv run rsr floor --check     # measure the suite and check the ratchet
uv run python experiments/e0a/run.py
```

## Conventions

- Small commits, conventional messages, one logical change each.
- An **ADR** in `docs/decisions/` for any choice expensive to reverse — vendoring vs reimplementing
  TG, the coref tool, the config framework. ADRs are **superseded, never edited**.
- **Pre-registrations go in `preregistration/` and are committed BEFORE the experiment they govern.**
  `git log` is what makes that checkable afterwards, which is the entire point.
- Mention unrelated problems, don't fix them. Surface them in the return; don't fold cleanup into an
  unrelated change.
