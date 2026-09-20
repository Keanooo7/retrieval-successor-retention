# S0-01 — the training-loop defects, tests first

**Baseline:** `c647af4290f7a45be2edf947b7973816dcc2af8b` (read from `git rev-parse HEAD` at the moment of writing).
**Lane:** researcher · device not required (CPU is fine and is better for determinism).
**Sprint:** 0 — make the substrate honest.

---

## ⚠️ AMENDED 2026-09-20 at `8ad64a27ab130908071de9048325525ba8e1364e`

Amended by the manager under `docs/lab-notes/dispatch-2026-09-20b-cycle-1.md` Task A.
**The original brief contained a contradiction, and it was the owner's, stated as such.**

It asserted *"Defect (a) makes them uninterpretable; saying so is the deliverable, re-deriving
them is not"* — asserting a belief as settled **and in the same clause barring the measurement
that would test it.** A claim carried forward as fact because someone wrote it once is the exact
failure this project's record is about.

Four changes, all of them subtractions:

1. **Row (a) is gone from this brief.** `docs/lab-notes/dispatch-cycle-01-masked-loss.md` owns
   defect (a) and **supersedes this brief on it.** Its falsifier — the run-to-run floor at
   identical seed and config — is the discriminator. Without that floor, "the loss scores
   padding" can be true *and inert*.
2. **The `Do NOT` clause about re-running ledgers is deleted, not softened.** It forbade the
   measurement that cycle 1 exists to make.
3. **Defect (d) is demoted.** It is not independently fixable and it is not silent-free; this
   brief said both and both are wrong. See "Downstream, not independent" below.
4. 🔴 **Every number below is a prediction, not a premise**, and each carries the command that
   tests it. "93.4% of 193,536" is something to measure. Where the manager's own re-execution at
   `8ad64a2` already disagrees with the original text, the disagreement is recorded inline —
   that is a correction, not a new premise, and you still re-run the command.

---

## ⚠️ AMENDED (2) 2026-09-20 at `058e712b0f421d7533c96f30a0a7a0942d7df6b5`

**Cycle 1 landed in `loop.py` between the first amendment and this one, and it moved everything.**
`src/rsr/train/loop.py` gained 94 lines. Every line citation below was re-measured at `058e712`;
the numbers in the tables are the new ones, and the stale ones are named so nobody re-derives them.

🔴 **One premise of this brief is now FALSE, and cycle 1 falsified it.** The hypothesis opened
*"a training loop that no test imports."* `tests/test_train_loss.py:31` reads
`from rsr.train.loop import build_vocab, encode, lm_loss`. **A test imports it.** That does not
retire the defects — all three were re-measured and all three stand — but the framing "nothing
touches this file" is gone, and any argument that leaned on it is void.

| Citation as written | Measured at `058e712` |
|---|---|
| `loop.py` is 307 lines | **401** |
| no test imports it | **false** — `tests/test_train_loss.py:31` |
| (b) `FIFOPolicy()` at `:164-166` | **`:213-215`**, still unconditional |
| (b) `policy_name` stamped at `:175` / `:178` / `:207` | parameter `:184`, frozen config `:224`, `run_id` `:228` |
| (b) no `--policy` flag | ✅ holds — the flag table is `:364-375` and has none |
| (c) `grep -c srep_norm` → 0 | ✅ **0**, unchanged |
| (e) `--vocab` at `:274`, then `:275` | **`:369`** |
| (d) `policy.attribution()` at `:236` | **`:315`**; `attribution_counts` at `rsr.py:450` holds |
| `value_head=None` at `:161` | **`:210`**, `build_param_groups(model, None, …)`, comment `:207-209` |
| `from_registry` eager read at `rsr.py:236-243` | **`rsr.py:215`** |

*Three briefs in a row have now carried a wrong line number. **The line number is not the claim** —
re-run the grep, and if it disagrees with this table, this table is what is wrong.*

---

## Hypothesis (not instruction)

`src/rsr/train/loop.py` is a training loop that **no test imports**, and it has already shipped
several defects. Some of them are silent: they produce a plausible loss curve.

**Whether the numbers in `runs/*/ledger.json` are interpretable is an open question owned by
cycle 1, not an assumption owned by this brief.** Test the beliefs below; if a defect is not
real, say so and leave it.

| Prediction | Command that tests it | Manager's reading at `8ad64a2` |
|---|---|---|
| `loop.py` is 401 lines | `wc -l src/rsr/train/loop.py` | **401** at `058e712`; it was 307 before cycle 1 |
| 🔴 ~~No test imports it~~ **FALSIFIED** | `grep -rn 'rsr\.train\.loop\|from rsr.train import loop' tests/` | `tests/test_train_loss.py:31` **imports it** (`build_vocab, encode, lm_loss`), landed by cycle 1. `test_evidence_machinery.py` still only names the path as a string literal |

## Files in scope

- `src/rsr/train/loop.py`
- `src/rsr/retention/rsr.py` — the `from_registry` eager read only, around `:236-243`
- `tests/test_train_loop.py` — new
- Nothing else in `src/`.

## The defects, as predictions

| # | Predicted line | Claim to test | Predicted silent? |
|---|---|---|---|
| b | `:213-215` | `policy = (FIFOPolicy())` is constructed unconditionally, while `policy_name` (a parameter, `:184`, default `"fifo"`) is stamped into the frozen config (`:224`) and the `run_id` (`:228`). `main()`'s flag table (`:364-375`) has **no `--policy`**. | yes |
| c | — | The `srep_norm` hinge is absent from the objective. `StepOutput.srep_norm_penalty` is computed at `model.py:424` and discarded. Predicted count: **0**. | yes |
| e | `:369` | `--vocab` defaults to 50257 against a corpus predicted to hold **156 unique words**, so the derived path (`:144`) is predicted unreachable from the CLI. | no |

**The commands:**

```bash
# (b)
grep -n 'FIFOPolicy\|policy_name\|--policy' src/rsr/train/loop.py
# (c) -- grep -c exits 1 when the count is 0. Read the COUNT, not $?.
grep -c srep_norm src/rsr/train/loop.py
# (e)
grep -n '"--vocab"' src/rsr/train/loop.py
python -c "from rsr.train.loop import build_vocab; ..."   # the 156 is yours to measure
```

⚠️ **Two corrections the manager already found at `8ad64a2`, which you re-check anyway:**

- **(e)'s line number was wrong twice.** `("--vocab", int, 50257)` was `:275` at `8ad64a2`, not `:274`; at `058e712` it is **`:369`**.
- **(e)'s "unreachable from the CLI" is too strong.** the derived read (`V = vocab if vocab else …`) reads
  `V = vocab if vocab else 4 + len(build_vocab(probe))`, and `main()` passes `vocab=a.vocab`.
  `--vocab 0` is falsy and **does** reach the derived path. The defect is that the *default* is
  50257 and the derived path is unreachable **at the default**, which is a smaller claim. State
  which one you fixed.
- The **156** is unverified by the manager. It is a prediction.

## Downstream, not independent

**(d) `:315` `policy.attribution()` vs `attribution_counts()` (`rsr.py:450`).** The original brief
called this independent and "not silent". Both are wrong:

- It is **not independently fixable.** `loop.py:214` constructs `FIFOPolicy()` unconditionally and
  `attribution_counts` exists only on `RSRPolicy`. Renaming at `:315` leaves `attr` still `None`.
  It is **downstream of (b)** and cannot be closed before (b) is.
- It is **not loud.** `hasattr(policy, "attribution")` is False, the expression evaluates to
  `None`, and the ledger records `attribution=attr` — a `null` that looks like "no attribution
  this run" rather than "the call never resolved." That is silent by this project's own
  definition.

Do not fix (d) in this brief. Cycle 4 owns it, after (b).

## Less certain, listed separately on purpose

- `:210` passes `value_head=None` into `build_param_groups`, so ψ̂ would train outside its μP
  group. The comment at `:207-209` says it must be passed. **Check whether anything yet constructs
  a value head here; if not, leave a failing test rather than a speculative fix.**
- `rsr.py:215`'s `from_registry` builds `kw` eagerly, so `from_registry` is predicted to raise on an empty
  ledger even for overridden fields. **Verify by calling it with `nu=0.0, gamma=0.0` on an empty
  registry.**

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
- Do not fix defect (a). `dispatch-cycle-01-masked-loss.md` owns it and supersedes this brief there.
- Do not fix defect (d). It is downstream of (b); cycle 4 owns it.
- Do not change any frozen constant, bucket edge or threshold to make a gate pass.
- Do not add a default to get past a `constants.py` refusal.

## Report

The standard block, plus `BRIEF ERRORS` — including any prediction above that turns out to be
wrong. **For each fix, state which mutation turns only its test red.**
