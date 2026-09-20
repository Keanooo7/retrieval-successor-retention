# Cycle 1 — `loop.py` defect 1: the loss scores padding

**Baseline sha:** `46c208eef403e25a9c4452b689e6fca918496749` — read from `git rev-parse HEAD` at 2026-09-18 12:58:05 PDT, not recalled.
**Suite floor:** `passed=318 failed=0 skipped=0 errors=0`, measured at that sha. It may
rise. It may not fall, and a skip is not a pass.

```bash
cd ~/retrieval-successor-retention          # /opt/homebrew/bin/git, never bare git
```

🔴 **This shell is zsh.** `$?` after a pipe is the *last* command's status and
`PIPESTATUS` is undefined — it is `$pipestatus[1]`. Write output to a file and read
`$?` directly. Your manager made this mistake three times in cycle 0; it is the single
most productive error in this project.

---

## The falsifier

> The reading dies if, on the committed synthetic corpus at `seed=0, steps_per_stream=48,
> batch=8`, **either**: the fraction of scored targets that are PAD is **below 0.50**;
> **or** masking the loss to real tokens moves the final training loss by **no more than
> the run-to-run floor**, where that floor is measured as the spread between two runs at
> identical seed and config differing in nothing at all.

Both halves must be measured. The second is the one that matters: if masking changes
nothing, "the loss scores padding" is true and inert, and the handoff's claim that **every
perplexity figure in every ledger is uninterpretable** does not follow.

## The premise, verified at source — check it anyway

`src/rsr/train/loop.py`'s `step_fn` is:

```python
def step_fn(t, o, ids_t, mask_t, row_valid):
    lg = o.logits
    return F.cross_entropy(
        lg[:, :-1].reshape(-1, lg.shape[-1]), ids_t[:, 1:].reshape(-1), reduction="mean"
    )
```

No `ignore_index`, no mask. 📌 **`mask_t` and `row_valid` are already passed in and
already unused** — `src/rsr/model/tg/policy_loop.py:140` calls
`step_fn(t, out, ids_t, mask_t, row_valid)`. The fix does not need a new argument.
`TGConfig` sets `pad_id=0`.

⚠️ **Do not take the 93.4% on trust.** The handoff states "93.4% of the 193,536 scored
targets are PAD". That number is not in any ledger. **Measure it and write it to a ledger
row.** If it disagrees with the handoff, the handoff is wrong and you say so.

## Files in scope

- `src/rsr/train/loop.py` — `step_fn` only.
- `tests/` — a new test file for `rsr.train`. **365 lines of that package have zero
  tests**; nothing in `tests/` imports it except `checkpoint`.
- `scripts/mutation_battery.py` — one new `Mutation` entry.

Nothing else. In particular **not** `policy_loop.py`, `model.py`, or the corpus.

## The bar

1. **A test that fails before and passes after.** Write it first, watch it fail, then fix.
2. **A mutation that reddens ONLY that test**, added to `scripts/mutation_battery.py` and
   `--check` green. An off-gate failure now makes a mutation *unproven*; if it couples,
   declare the coupling in `off_gate_allowed` with a reason rather than shrugging at it.
3. **A ledger** at `runs/cycle-01-masked-loss/ledger.json`, written with
   `scripts/ledger.py`. It will refuse a command whose entry point is not committed at the
   recorded sha, and it will refuse to write at all over an uncommitted `src/`. **That is
   not a bug to route around** — commit your code, then measure.
4. Ledger rows for: the measured PAD fraction; the two-run floor; masked and unmasked final
   loss; `seeds_actually_run`; `steps_done`.
5. `uv run pytest -rs` and `uv run python scripts/mutation_battery.py --check` both
   green, counts reported separately from skips.

## State your expected result before you run

Write it in `RESULTS.md` **before** the measurement, and do not edit it afterwards:
masking should **raise** the reported per-token loss, because PAD is trivially predictable
and currently dominates the mean. A masked loss that is *lower* is a finding, not a bug to
tune away, and it would falsify the reading above.

If you cannot name what would change your mind, the run is not science and you should say so
rather than run it.

## Do NOT

- 🔴 **Do not touch the other four defects.** Policy selection is cycle 2, the `srep_norm`
  hinge is cycle 3, `attribution_counts()` is cycle 4, `--vocab` is cycle 5. One defect,
  one cycle, one commit.
- 🔴 **Do not change a frozen constant, bucket edge, threshold or arm** to make anything
  pass. That is the most serious thing you can do here and it stops the night.
- Do not tune `γ_b` or `ν`, do not start E1, do not make any §7 decision. All of it sits
  above a memory that does nothing.
- Do not invent a constant. `beta`, `nu`, `gamma`, `tau` come from the registry, which
  raises and names the experiment that owes the value. If it raises, the answer is "E0e has
  not run", not a default.
- Do not report a skipped test as a passing one.

## Done when

One commit on `main` containing: the test, the fix, the mutation entry, `RESULTS.md`, and
the ledger — with the suite and the battery both green and the ledger's
`verdict.outcome` one of `survived` / `falsified` / `inconclusive`. **A cycle is
closed by a commit, not by a conclusion.**

## BRIEF ERRORS

🔴 **Correcting this brief is part of the job, not a courtesy.** Open a `## BRIEF ERRORS`
section in your `RESULTS.md` and put every false premise, wrong line number, invented
threshold or logical contradiction you find in it there — including ones you worked around.

Last night **7 of 11 briefs contained an error a researcher caught**, and none of those
briefs survives, because they were in-session prompts to retired subagents. This one is in
git before you are dispatched, so your corrections have something durable to attach to.

**One known defect in the framing, declared up front:** defect 4 (`policy.attribution()`
vs `attribution_counts()`) cannot be fixed independently of defect 2.
`attribution_counts` exists only on `RSRPolicy` (`src/rsr/retention/rsr.py:450`), and
`loop.py` constructs `FIFOPolicy()` unconditionally — so `attr` would still be `None`
after a correct rename. That is cycle 4's problem, not yours; it is stated here so nobody
rediscovers it as a surprise.

---

*🔴 The primary claim of this project is cognitive, not engineering: does a model trained
only to predict the next sentence rediscover Kintsch & van Dijk's 1978 leading-edge
strategy? The engineering delta is secondary and the spec says so. This cycle is instrument
repair — fix the instrument, then measure. Drifting toward the engineering framing is drift,
even when every number is honest.*
