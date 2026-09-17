# CLAUDE.md — standing rules for every agent session

## What this is

**Retrieval-Successor Retention (RSR).** Thought Gestalt (TG) is a recurrent
transformer that compresses each sentence to one vector and writes it to a working
memory of `M` slots; when memory is full it evicts the oldest. RSR replaces that
one line with a learned policy that evicts the slot with the lowest predicted
*future* retrieval demand given the current discourse state.

The scientific question is whether a system trained only to predict the next
sentence rediscovers Kintsch & van Dijk's 1978 leading-edge strategy.

Brendan is the sole owner and directs. Agents are the implementation layer.

## Read in this order

1. **`docs/spec-corrections.md`** — read this **before** the spec.
2. `docs/spec/rsr_model_spec_v0.5.md` — the specification, verbatim.
3. `docs/decisions/` — ADRs.
4. `docs/release-conditions.md` — §16's eight conditions.

**`docs/spec-corrections.md` wins over the spec body, always.** The spec is on its
fifth revision with three changelogs, and several passages were superseded by a
correction and never rewritten. If the two disagree, the corrections file is
authoritative. **Do not resolve a conflict by re-reading the spec.** If you find a
passage a changelog superseded that is not yet in that file, add it there **and
tell Brendan** — do not silently pick a reading.

## Scope boundary — hard

**Only weeks 1–4 are approved (§16). Everything downstream is a projection, not a
permission.** Total compute envelope for weeks 1–4 is **~$100**. No billable GPU
commitment: week 1 gets a named provider, a written quote, and a **cancellable
hold with a movable start date**. The reservation *is* the four-figure commitment
§16 declines to approve.

## Prohibitions — honor these literally

- **Do not fill §15.** It reads: *"This section is a placeholder and must not be
  filled by a reviewer, an advisor, or a model."* You are a model. If you find
  yourself drafting a candidate for §15.2, stop. You may schedule the work; you
  may not do it. **Do not restate the pointer §15 leaves open, and do not suggest
  answers to it.**
- **Do not backpropagate the retention loss into the transformer or `W_sent`.**
  Only `φ` receives gradient. `c_t` enters `ψ̂` with a stop-gradient on the
  transformer side. This holds for whichever retention loss is active.
- **Age is excluded from `ψ̂`.** Supplying it invites collapse onto recency and
  makes the vacuity failure mode invisible rather than merely possible. An
  age-only head and a content+age head exist as separate baseline arms.
- **`b` never enters `ψ̂`** or any differentiable path — eviction argmin only.
  Balance is a zero-gradient control loop, not a competing objective.
- **Make no scaling claim anywhere.** E5 is hygiene; a 3× width range at ≤21M
  parameters cannot resolve slope from intercept.
- **Do not call a truncated-BPTT window "consolidation."** The CLS mapping was
  inverted and the claim is deleted. In CLS, consolidation is how an experience
  *acquires* the ability to shape slow-learning weights; `detach` does the
  opposite.
- **Do not down-weight underfull steps as the bias correction.** Rescale the
  target: `r_i(t) = share_i(t) · |memory_t| / M`. Down-weighting a biased target
  reduces how hard the estimator fits it; it does not remove the bias. **And do
  not call the rescaled target "unbiased"** — it credits absent competitors and
  biases stream-initial slots downward. "Unbiased" is the word a reviewer will
  test.
- **Delete §1's [P5]–[P14] verification note from the spec copy once E0f is
  logged** (Sprint 2). It cannot still be there at week 12.

## The constants registry

`src/rsr/constants.py` classifies every constant `FROZEN`, `MEASURED`, `DERIVED`
or `CONDITIONAL`. Reading a `MEASURED`/`DERIVED` constant before its source
experiment has logged a value **raises**, naming the experiment.

**Never hardcode a `MEASURED` or `DERIVED` value.** That is not style — it is the
mechanism that prevents defect D-1 recurring, where a frozen `γ_b = 0.001` could
not move the argmin it governed and would have been reported as "no effect" in
week 7. If the registry refuses your read, run the experiment; do not route around
it.

`M` and `S` have **no global value** — they are per scope (`synthetic`, `corpora`,
`e7`). §13: no cross-corpus comparison of absolute numbers is valid.

## The two headline tests

- **`tests/test_reduction.py`** (E0b) — RSR under the §3.7 switches vs the PyTorch
  TG, **bit-exact**, on CPU, in CI. Intra-repo by design: it guarantees E3's FIFO
  and RSR arms differ only in the eviction rule.
- **`tests/test_fidelity.py`** — the PyTorch TG vs golden tensors from the pinned
  JAX reference, to a tolerance committed in ADR-0002 *before* the fixtures were
  generated. **Gradients included, not just forwards.**

**Every term added to the eviction score needs a documented off-switch**, set in
`RSRConfig.reduction_to_tg()`. Without one, §3.7 stops being a reduction and E0b
stops testing what it claims to test.

**If `test_reduction.py` fails, read its docstring before debugging.** The most
likely cause is RNG ordering, not the mechanism: constructing `φ` from the global
generator consumes draws, shifts data order and dropout masks, and fails the test
for a reason unrelated to retention.

## Stack

PyTorch only. Python 3.12, `uv`, `ruff`, `pytest`. **JAX is never installed here**
— `third_party/ThoughtGestaltCode` is a pinned read-only source reference plus a
one-time tensor extraction that runs on rented hardware (ADR-0001).

MPS is **best-effort, not required**. §4.4 gives the development machine only the
coordinate check, the reduction test, the synthetic corpus and debugging; CPU
serves all four and is better for E0b's determinism. All code must also run on
CUDA without modification.

## How to work

- Small commits, conventional messages, one logical change each.
- **Cite spec sections** (`§4.8`, `§12.3 item 2`) in code comments and commit
  messages. The whole document set is cross-referenced that way, and it is how a
  future session finds the rationale.
- Open an **ADR** in `docs/decisions/` for any choice expensive to reverse.
- **Every experiment writes a `RESULTS.md`** next to its `run.py` with the command
  that produced it, the git SHA, the hardware, and the numbers — **including the
  ones that came out wrong.**
- **Pre-registration commits land before the experiment they govern**, in their
  own commit, ordered ahead in git history. A threshold registered after seeing
  the data is not a threshold.
- **Measure, don't extrapolate** (§12.4). Documented configuration is not evidence
  of what was actually run.
- **Never report a skipped or unrun test as passing.** `test_reduction.py` and
  `test_fidelity.py` are currently skipped pending the transcription; GATE-1 must
  say so.
