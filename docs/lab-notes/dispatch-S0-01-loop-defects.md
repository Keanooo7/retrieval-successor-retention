# S0-01 — the five training-loop defects, tests first

**Baseline:** `c647af4290f7a45be2edf947b7973816dcc2af8b` (read from `git rev-parse HEAD` at the moment of writing).
**Lane:** researcher · device not required (CPU is fine and is better for determinism).
**Sprint:** 0 — make the substrate honest.

## Hypothesis (not instruction)

`src/rsr/train/loop.py` has 307 lines, **no test imports it**, and it has already shipped five
defects. Three of them are silent: they produce a plausible loss curve. I believe every number in
every `runs/*/ledger.json` produced through this loop is uninterpretable, and that the fix is
small in each case. **Test the belief; if a defect is not real, say so and leave it.**

## Files in scope

- `src/rsr/train/loop.py`
- `src/rsr/retention/rsr.py` — the `from_registry` eager read only, lines ~236-243
- `tests/test_train_loop.py` — new
- Nothing else in `src/`.

## The five, with what I think is wrong

| # | Line | Claim | Silent? |
|---|---|---|---|
| a | `:210-216` | `F.cross_entropy` with no `ignore_index`; `mask_t` and `row_valid` are parameters that are never read. `pad_id=0` is a real vocabulary row. **93.4% of 193,536 scored targets are PAD.** | yes |
| b | `:164-166` | `policy = FIFOPolicy()` unconditional, while `policy_name` is stamped into the frozen config (`:175`), the `run_id` (`:178`) and the heartbeat (`:207`). `main()` has **no `--policy` flag**. | yes |
| c | — | The `srep_norm` hinge is absent from the objective. `StepOutput.srep_norm_penalty` is computed at `model.py:424` and discarded. `grep -c srep_norm loop.py` → **0**. | yes |
| d | `:236` | `policy.attribution()`; the real method is `attribution_counts()` (`rsr.py:450`). `hasattr` is always False, so attribution is permanently `null`. | no |
| e | `:274` | `--vocab` defaults to 50257 against a corpus of **156 unique words**, so the derived path (`:144`) is unreachable from the CLI. | no |

Plus two I am less sure of, which is why they are listed separately:
- `:161` passes `value_head=None` into `build_param_groups`, so ψ̂ would train outside its μP
  group. The comment at `:158-160` says it must be passed. **Check whether anything yet constructs
  a value head here; if not, leave a failing test rather than a speculative fix.**
- `rsr.py:236-243` builds `kw` eagerly, so `from_registry` raises on an empty ledger even for
  overridden fields. **Verify by calling it with `nu=0.0, gamma=0.0` on an empty registry.**

## Bar

For each defect: **a test that fails before the fix and passes after**, and a mutation that reverts
only that fix and reddens **only** that test. 🔴 **If a mutation reddens nothing, the test is
vacuous and that is the finding — report it rather than keeping the test.**

Defect (b)'s test is four lines: construct with `policy_name="rsr"`, assert the constructed policy
is not a `FIFOPolicy`.

## Done when

`ruff check` exit 0 · `pytest -rs` prints a count with 0 failures and 0 new skips · the test floor
has risen and is recorded from a **measured** run · a PR is merged to `main`.

## Do NOT

- Do not fix the inert memory or the corpus in this brief. Those are S0-02 and S0-03.
- Do not change any frozen constant, bucket edge or threshold to make a gate pass.
- Do not re-run or re-interpret any existing `runs/*/ledger.json`. Defect (a) makes them
  uninterpretable; **saying so is the deliverable, re-deriving them is not.**
- Do not add a default to get past a `constants.py` refusal.

## Report

The standard block, plus `BRIEF ERRORS` — including any of the seven claims above that turns out
to be wrong. **For each fix, state which mutation turns only its test red.**
