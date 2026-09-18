# Overnight run — 2026-09-18

**Manager:** Claude Opus 5 (1M), unattended. **Started** 00:25 PDT. **Stops** 07:00 PDT.
**Repo:** `~/retrieval-successor-retention` @ `dce75c1` at start, clean tree, `287 passed, 0 skipped`.

Protocol per cycle: dispatch a **fresh** researcher with a brief naming one falsifier →
researcher runs **one** experiment → report → manager reviews, **re-executes one claim**,
records which → retire the researcher → dispatch the next. A researcher is never given a
second brief. Canary every 4th cycle: one fixed config, one fixed seed, re-run; if the
number moves, the environment moved and everything since is suspect.

**"We learned nothing this cycle" is a writable row.** So is "the falsifier could not be
reached." An empty *verified by re-execution* field is a failed cycle and is labelled one.

---

## Cycle 0 — manager setup (no experiment)

| | |
|---|---|
| **Falsifier** | none — infrastructure |
| **Dispatched** | nobody |
| **Result** | `scripts/ledger.py` written: the `runs/<id>/ledger.json` contract the protocol requires did not exist. |
| **Verified by re-execution** | Ran the helper's own refusals. `stat()` over 1 sample raises `SingleSample`; `sd` over 1 sample is `None`, never `0.0`; `sd` over three identical samples returns `0.0` **with `sd_exactly_zero: true`** set, so the flat statistic is visible rather than silent; `git_sha` is stamped from inside the tree (`dce75c1`) and `git_dirty` recorded. |
| **Learned** | The protocol's "every number traces to `runs/<id>/ledger.json`" was unenforceable — no such file or convention existed anywhere in the repo. `measurements/ledger.json` is a *different* artefact (frozen constants, §4.5) and is currently **absent**, i.e. the constants registry is empty. That last fact turns out to be the reason the training loop cannot instantiate RSR at all (see cycle 1). |

---

## Cycle 1 — is the RSR arm an arm?

| | |
|---|---|
| **Falsifier** | *"the training loop's RSR arm and its FIFO arm are different arms."* |
| **Dispatched** | fresh researcher, retired on hand-back |
| **Verdict** | **falsified — they are one arm.** `policy_name` is a label with no mechanism behind it. |
| **Verified by re-execution** | ✅ **Their headline claim, re-run by me at `4245b7d`.** Two CPU runs, seed 0, identical config, differing only in `policy_name`. Printed `policy fields: fifo rsr`, `run_ids differ: True`, `loss sequences bit-identical: True`, `max abs diff: 0.0`, 10 beats, and every `attribution` field `None`. Matches their ledger rows exactly. I also audited the ledger directly: all 60 rows and 5 command rows present, and every number in their report resolves to a row. |
| **Ledger** | `runs/cycle-arm-identity/ledger.json` — 60 rows, 2 statistics, 5 commands, verdict populated. Now committable (see below). |

**What we learned.** `src/rsr/train/loop.py:138` constructs `FIFOPolicy()` unconditionally
while stamping `policy_name` into the `run_id` and into the frozen config the heartbeat
header records. So a run whose heartbeat reads `"policy": "rsr"` **is FIFO** — corpus-sheet
defect 2 relocated out of `RSRConfig`'s defaults and into the training loop. On CPU the two
arms are **bit-identical** (`max abs diff = 0.0`). On MPS they differ by `3.18e-07` — and the
researcher's own design decision makes that readable: they ran a **`fifo`-vs-`fifo` replicate
at the same seed** and it differs by *exactly the same* `3.1789143850602386e-07`. That is the
nondeterminism floor, not policy signal. Without the replicate the MPS number would have been
uninterpretable, and it is the thing that turns a suggestive result into a clean one.

**The eviction path itself is fine.** Bypassing the walls in throwaway scratch: 256 evictions,
`attribution_counts = {"psi": 256}` — **zero `fifo_warmup`**, so explicitly not gauntlet 0.2 —
and the RSR victim disagreed with FIFO on the identical `MemoryState` in **239 of 256** cases
(0.934). The absence of any difference in the training loop is the loop's doing, not RSR's.

**Five walls, each with a real traceback** (their `walls.json`): (1) `constants.py:829`
`UnmeasuredConstant` on `nu` — root cause is that `measurements/ledger.json` does not exist at
all; (2) `rsr.py:300` `ValueError`, `b_enabled=True` needs a bias object; (3) `bias.py:60`
`NotImplementedError`, `ProtectionBias.__init__` is a stub; (4) `rsr.py:377` `AttributeError`;
(5) `rsr.py:433` `NotImplementedError` in `observe()`.

🔴 **Wall 4 is the one that goes beyond my reading, and it is a contradiction rather than an
absence.** `_score()` calls `self.bias.b(slots)`; `ProtectionBias` declares
`update`/`values`/`reset` and **no `b`**. The two surfaces have drifted apart and the
`# type: ignore[attr-defined]` on that line is exactly what let them — it gags the one check
that would have caught it. **Completing `ProtectionBias` to its own docstring would still not
satisfy `_score()`**, and `bias: object | None` in the constructor pins no protocol, so nothing
will catch it later either.

🔴 **Three further walls are *absences*, so they raise nothing and would each ship a
plausible-looking run.** `run_policy_loop` **never calls `policy.observe(...)`** and captures no
`AttentionTrace` to pass it — so wall 5 would never fire from a training run, `ψ̂` would simply
never receive gradient, and the arm would report itself as RSR while running a **frozen random
head**. `build_param_groups(model, None, …)` passes a literal `None` for the value head, so `φ`
gets no muP group. And `main()`'s argparse has **no `--policy` flag**, so the one knob that
names an arm is unreachable from the command line. Wall 5 is therefore *less* of a blocker than
it looks and *more* of a problem.

**Two process defects, one of them mine.**
- 🔴 **`/runs/` was in `.gitignore`, so the ledger the whole protocol rests on could not be
  committed.** The researcher caught this. Worse, the fix is not obvious: `/runs/` ignores the
  *directory*, and git does not descend into an excluded directory, so a `!` re-include under
  it can never match. Rewritten to ignore the **contents** (`/runs/**`) and re-include
  directories plus `ledger.json` / `RESULTS.md` / `walls.json` / `baseline.json`. Verified: 3
  evidence files staged, **0 `.pt` and 0 heartbeat files** staged.
- **My error:** the ledger records `git_dirty = true` at dispatch, caused by my own untracked
  `scripts/canary.py`. My "clean tree at dispatch" baseline in the brief was stale the moment I
  wrote it. **Corrected for the rest of the night: manager infrastructure is committed before
  the brief is written, not after.**
- Minor: their verdict string attributes the hypothesis to *"Brendan's reading"*. It was mine.
  Left as-is in the ledger (it is an append-only record) and corrected here.

**Boundary — what this does NOT say.** Nothing about whether RSR beats FIFO. The part-C 0.08
loss gap is an **untrained random `φ`**, one seed, forward-only, one batch; its sign is
meaningless and it must not be cited as an arm comparison. Nothing multi-seed — both `stat`
rows are `n`=2 and `n`=3 *run-to-run* repeats at a **fixed** seed, not seed variation, and both
carry `sd = 0.0` correctly `sd_exactly_zero`-flagged (the final beat coincides; the 3.2e-07
lives in intermediate beats). The `gamma = 0.97` in part C was supplied by the researcher as a
visible bypass and is **not** a measurement of `gamma`. Nothing at approved-scope widths
(`d=384`, `S=80`, `M=40`), nothing on PG-19, and the suite was not run.

---

## Owner correction landed mid-cycle — S-7

Brendan sent correction **S-7** (sentence length as an uncontrolled confound in E7) during
cycle 1. Filed as **correction 24**, not 16: **16 is already taken** by the `T_warm = ∞` /
E0b-vacuity entry that gauntlet 0.1 rests on, and 16–23 are cited by number elsewhere, so
overwriting would have destroyed a live correction and renumbering would have broken existing
citations. `docs/release-conditions.md` condition 4 now carries the analysis requirement in the
table plus a note, and the row's source reads `D-5 · S-7`.

**I re-measured the two claims he marked as measured**, since they were about to enter an
authoritative document. Mean **4.14** words/sentence ✅, max **6** ✅, **zero** sentences over 8
✅ — all three reproduce. But **n = 3072, not 1536**, at
`SyntheticConfig(sentences_per_document=48, seed=0)` (64 documents); full distribution
`{2: 91, 3: 244, 4: 1914, 5: 784, 6: 39}`. The 1536 is a different `sentences_per_document`.
The shape claims — which are the ones that carry his argument — hold either way. I recorded my
own number with its config rather than transcribing his, per §12.4.

The API readability percentages are written down **once**, labelled correlational, with an
explicit "do not cite as a constant" and an instruction that neither number enters
`constants.py`. No multiplier, no `srep_norm_target`, and no frozen constant was touched.

---

## Cycle 2 — do `ProtectionBias` and `_score()` describe the same object?

| | |
|---|---|
| **Falsifier** | *"`ProtectionBias`'s declared interface and `RSRPolicy._score()`'s call site describe the same object."* |
| **Dispatched** | fresh researcher, retired on hand-back |
| **Verdict** | **falsified.** They reconcile only if one side changes shape. |
| **Verified by re-execution** | ✅ **Their headline claim, re-derived by me in my own code** (not their harness). A class exposing *exactly* `ProtectionBias`'s four declared members — `__init__`/`update`/`values`/`reset` — fails one eviction with `AttributeError: 'Bias' object has no attribute 'b'` at **`src/rsr/retention/rsr.py:377`**, and adding **only** `b(slots)` to the identical class makes the same call succeed with `attribution = psi+b`, victim 0. (My first attempt errored on my own wrong `MemoryState` signature — mine, not theirs.) I also audited all 58 ledger rows: every number in their report resolves to a row, all five-seed statistics carry real sds, and every `sd = 0.0000` row is itemized with a reason I could check independently. |
| **Ledger** | `runs/cycle-bias-interface/ledger.json` — 58 rows, 5 seeds × 480 evictions. |

**The interface.** Their smallest faithful `ProtectionBias` works on its own terms (20 updates
drive a concentrated slot to `b = −1.0` while the rest rise to `+0.90`) and still cannot take one
eviction. **Which side is self-consistent: `ProtectionBias`'s.** §3.5 verbatim — *"A scalar bias
`b_i`, added inside the eviction argmin only"* — updated from `ū_i` and `b_i`'s previous value.
**Nothing in §3.5 makes `b` a function of the memory state**, so a zero-argument `values()` is the
faithful surface, and correction 4 touches `γ_b`'s magnitude and not the signature. The one
engineering argument for passing `slots` (device/liveness) does not hold: `_score` masks dead
slots itself one line later at `rsr.py:382` and the policy already has `_device_of(slots)`.
**The spec names no accessor**, so the code-shape choice is Brendan's — correctly left to him.

**Two further incompletenesses in the same interface**, found by building it: `update(u_bar)`'s
signature puts the EMA in the caller, which makes §3.5 item 2's half-life requirement
unenforceable from inside the class, and `__init__` has no half-life parameter at all. And
**`ProtectionBias` declares no per-slot invalidation hook**, though `RetentionPolicy.on_write`'s
own docstring names *"the anti-collapse loop's `ū`"* as having exactly gauntlet 0.4's shape —
without one, **a new occupant inherits the previous tenant's `b`.**

🔴 **They corrected my brief, and they were right.** I wrote *"the `ū` EMA with half-life `τ`"*.
`τ` is the **dead band** (§3.5 item 3); the EMA half-life is `E[lifetime]/4` (item 2), a separate
derived quantity. My error, recorded so it does not propagate into a later brief.

### The `γ_b` result — correction 4's arithmetic confirmed, its operational claim refined

Controlled so the comparison is clean: one `b_enabled=False` driver owns the memory, so the
`MemoryState` sequence is **identical** across every `γ_b`, and every arm shares one value-head
object so `ψ̂` is bit-identical. `M=40`, `S=80` (both FROZEN registry reads), `d=384`, 5 seeds ×
480 evictions, `attribution` all `psi` / `psi+b`, **zero `fifo_warmup`**.

| `γ_b` | fraction of evictions changed | max \|b\| |
|---|---|---|
| 0.0 (control) | 0.0 ± 0.0 | 0.0 |
| **0.001** | **0.02125 ± 0.00539** | 0.079 |
| 0.01 | 0.25083 ± 0.00731 | 0.79 |
| **0.05** | 0.56417 ± 0.01313 | 1.0 (clip) |
| 0.075 | 0.58667 ± 0.01264 | 1.0 (clip) |
| **0.1** | 0.59500 ± 0.01289 | 1.0 (clip) |

**Correction 4's arithmetic is confirmed to the digit**: predicted max \|b\| = `0.001 × 80 = 0.08`,
measured **0.079** = `0.001 × 79`. **Its operational claim is not exactly true.** Correction 4 says
that at `γ_b = 0.001` the control loop *"cannot move the argmin, and is operationally identical to
`b ≡ 0`"*. Measured, it **does** move it — on **2.1% ± 0.5%** of evictions, against **0.0% ± 0.0%**
for `b ≡ 0`. The mechanism is visible: 15.0% ± 1.3% of driver evictions have a top-2 score margin
below 0.079, and the median margin *of the evictions 0.001 actually flipped* is **0.0164 ± 0.0072**
versus **0.271 ± 0.025** at `γ_b = 0.1`. So at 0.001 the loop is **not inert — it is near-tie
noise**, perturbing exactly the decisions `ψ̂` had no opinion about. Arguably worse than inert,
because A5 would have measured a small non-zero effect of the wrong kind and had something to
report.

🔴 **And correction 4's own recommended range makes `b` the policy.** At 0.05–0.1, `b` decides
**56–60%** of evictions with `b` saturated at `±b_max` simultaneously at essentially every
eviction. §3.4's *"if `b` flips a large share of decisions, the balance controller is the policy"*
and §3.5 item 1's *"if `ψ̂ ≪ b`, the policy is the balance controller, not the value estimate"* are
both **realized at the range correction 4 prescribes**. The curve is **flat from 0.05 to 0.1**
(0.564 → 0.595) because both ends clip, so **A5's sweep cannot separate 0.05 from 0.1 on this
`ū`** — the discriminating variation lies *below* correction 4's range. This is not an argument
for 0.001. It is an argument that correction 4 fixed the magnitude and reopened the takeover
question that item 1's z-scoring was introduced to close.

Two side measurements: updates fired on **0.4926 ± 0.0106** of slot-steps at `τ = 0.25`, not the
~0.25 the `γ_b` formula divides by; and realized mean slot lifetime under the *learned-head*
driver is **26.49 ± 0.53**, not `M = 40` — so **`E[lifetime]` is policy-dependent, and a `γ_b`
derived from a FIFO E0e run is not the one the RSR arm's own lifetimes imply.**

**Boundary — and it is the whole caveat.** ⚠️ **`ū` is synthetic** (log-normal per-slot salience,
softmaxed). `observe()` still raises, no `AttentionTrace` capture exists, and
`rsr.retention.reward`'s `r_i` never entered this loop, so **every fraction above is conditional
on that generator and on `τ = 0.25`.** Nothing here measures `γ_b`/`τ`/`b_max`/`E[lifetime]` as
constants — that is E0e's job and the supplied values are labelled bypasses. Nothing about loss
or whether `b` helps (`φ` random, no LM ran). Collapse was never induced — imbalance was imposed
rather than arising from the closed loop. CPU only, so `_score`'s missing device alignment for
the bias is untested and remains a live suspect. The other four cycle-1 walls all stand.

**I did not touch correction 4.** A synthetic-`ū` measurement is not grounds for editing an
authoritative document; it is grounds for escalating. Both findings are in `for-brendan`.

---
