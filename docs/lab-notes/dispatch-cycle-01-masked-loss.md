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

---

# ⚠️ REVALIDATED 2026-09-20 at `dab4f0848d96ebab6466a8646c1426387e3a3140`

This brief was written against `46c208e`, on a branch, **before** the Studio/trunk reconciliation
(PR #11, `4ece429`). *Rebasing a brief does not revalidate it.* The manager re-executed its
premises against the merged tree. **Everything below supersedes the body above where they
disagree.** Read this block first.

**Baseline sha for this run:** `dab4f0848d96ebab6466a8646c1426387e3a3140` — read from
`git rev-parse HEAD` at the moment of writing, not recalled.

## Premises that held

| Premise | Re-executed how | Result |
|---|---|---|
| Suite floor `passed=318 failed=0 skipped=0 errors=0` | `uv run pytest -q -rs --tb=no`, output to a file, `$?` read on the next line with **no pipe** | ✅ **318 / 0 / 0 / 0, exit 0** at `5440290`. The merge did not move it. It may rise; it may not fall; a skip is not a pass. |
| `step_fn` has no `ignore_index` and no mask | `sed -n '210,216p' src/rsr/train/loop.py` | ✅ `:210-216`. The quoted source is semantically exact; the real code wraps the arguments onto separate lines. |
| `mask_t` and `row_valid` are already passed and already unused | `grep -n step_fn src/rsr/model/tg/policy_loop.py` | ✅ `:140` is `contrib = step_fn(t, out, ids_t, mask_t, row_valid)`. **The fix needs no new argument.** |
| The run is fresh | `ls runs/cycle-01-masked-loss` | ✅ does not exist. |
| `off_gate_allowed` exists for bar item 2 | `grep -n off_gate_allowed scripts/mutation_battery.py` | ✅ `:61`, with `unproven()` at `:624` and `leaked` at `:722`. Bar item 2 is executable as written. |

## 🔴 Premises that did NOT hold — three corrections

### 1. `TGConfig` does **not** set `pad_id=0`. `loop.py` does.

The brief says, under a heading that reads *"The premise, verified at source"*:

> `TGConfig` sets `pad_id=0`.

`src/rsr/model/tg/config.py:96` reads **`pad_id: int = 50257`**. The `0` comes from
`src/rsr/train/loop.py:144`, which passes `pad_id=0` explicitly into the `TGConfig(...)` it
builds, overriding the default.

**The conclusion survives and the provenance does not.** A researcher who does what the brief
says — check `TGConfig` — finds `50257`, and may reasonably conclude the premise is falsified and
stop. Check `loop.py:144`.

**Two things the brief does not say and you will need:**

- `encode()` at `loop.py:108` allocates `ids = torch.zeros(n, steps, L, dtype=torch.long)`. Pad
  positions are literally id `0` in the data, independent of any config field.
- Therefore there are **two** sources of PAD, not one: trailing positions inside a real sentence,
  **and every position of every sentence slot for a document with fewer than `steps` sentences**,
  which is an entire row of zeros. Count them separately in your ledger. A single PAD fraction
  hides which one dominates, and the masked-vs-unmasked delta depends on it.

### 2. The falsifier's config is under-specified — `iters` is missing, and the 93.4% implies `iters=8`

The falsifier names `seed=0, steps_per_stream=48, batch=8` and **does not name `iters`**. The run
is not reproducible without it, and the two-run floor is a comparison of two runs at "identical
config".

The handoff's denominator is checkable and it pins the answer:

```
batch × steps_per_stream × (max_tokens − 1) × iters
  8   ×        48        ×      (64 − 1)    ×   8    = 193,536   ← exactly the handoff's figure
  8   ×        48        ×      (64 − 1)             =  24,192   per iteration
```

So the handoff measured at **`iters=8`**, a value this brief never states. *(For contrast,
`scripts/canary.py`'s frozen `CONFIG` uses `iters=6`.)* 🔴 **Pin `iters` explicitly in your
manifest and say which value you used.** If you use anything but 8, the 93.4% is not the number
you are checking and you must say so. The `(max_tokens − 1)` factor is the `[:, :-1]` / `[:, 1:]`
shift in `step_fn` — targets, not tokens.

The reconstruction above is the manager's arithmetic, not a ledger row. **It is a prediction.**
Measure the denominator; do not adopt it.

### 3. "365 lines of that package have zero tests" does not reconcile

`wc -l src/rsr/train/*.py` at this sha:

```
 25 __init__.py   361 checkpoint.py   159 heartbeat.py   307 loop.py   852 total
```

`checkpoint.py` is tested. No subset gives 365: `852 − 361 = 491`, `loop + heartbeat = 466`,
`loop + __init__ = 332`. The number is either stale or was derived some other way. **Demoted to a
prediction; re-derive it or drop it.** It is not load-bearing for the falsifier — it is the
motivation — so do not spend the cycle on it.

## The discriminator — this is the whole point of the cycle

> **The run-to-run floor at identical seed and config.**

Two runs differing in **nothing at all** — same seed, same config, same code — bound how much of
any masked-vs-unmasked delta is signal. Measure the floor **first**, and measure it from two
actual runs, not from a recalled number.

⚠️ On MPS a floor of **`3.1789143850602386e-07`** was previously measured. 🔴 **That is a number
to reproduce, not to assume.** If yours differs, yours is the measurement and the old one is
prose. Say so plainly.

**Report spread.** An sd of exactly `0.0000` across seeds is a **broken result, not a clean one** —
it means the seed never reached the randomness. That has already happened in this project, and a
reported `0.0000` is what exposed it. A run with no sd, or an sd of exactly zero, is rejected.

## Also required, and not in the body above

- **The shuffle control.** A training run that does not report it, or reports it at ≈0, is refused
  rather than filed: a memory that contributes nothing makes every number in the run about a
  different model than the one under study.
- **`seeds_actually_run` and `steps_done`** as ledger rows, matching the prose. A multi-seed mean
  from a one-seed run is what those fields exist to catch.
- **Every command's `exit_code` read, never asserted.** This is zsh: `$?` after a pipe is the last
  command's status and `PIPESTATUS` is undefined — it is `$pipestatus[1]`. Redirect to a file and
  read `$?` on the next line. `grep -c` exits `1` when the count is `0`, which is a count, not a
  failure. 📌 See `docs/lab-notes/dispatch-S0-05-exit-code-enum.md` — **`scripts/canary.py:151`
  hardcodes `exit_code=0` into a ledger row.** Do not copy that shape.

## BRIEF ERRORS — still required, and this block has its own

The body's `## BRIEF ERRORS` section stands. **This revalidation block is also fair game**: its
`iters=8` reconstruction is arithmetic on a number that is itself unverified, and its claim that
`318` still holds was measured at `5440290`, one commit before this brief's baseline — the
difference is documentation only, which is itself a claim you may check with `git diff --stat`.
