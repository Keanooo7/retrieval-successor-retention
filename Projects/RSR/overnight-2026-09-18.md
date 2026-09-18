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

## Cycle 3 — can `attribution` tell an eviction `b` decided from one it didn't?

| | |
|---|---|
| **Falsifier** | *"`EvictionRecord.attribution` distinguishes evictions that `b` decided from evictions that `b` did not."* |
| **Dispatched** | fresh researcher, retired on hand-back |
| **Verdict** | **falsified, decisively.** `attribution` is a config echo with a timestamp. |
| **Verified by re-execution** | ✅ **Their headline claim, re-run by me.** Their construction is sharper than my brief asked for: a bias whose `b()` returns `torch.zeros(M)` is **provably** incapable of changing any argmin. Printed `same victim (b provably inert): True`, `attribution, b disabled: psi`, `attribution, b enabled but 0: psi+b`. So the tag is wrong even when the term is mathematically inert. I audited all 254 ledger rows; every reported number resolves, and `sd_zero_rows_unclassified` is `[]` — all 87 zero-sd rows classified in-process. |
| **Ledger** | `runs/cycle-attribution-instrument/ledger.json` — 254 rows, 16 arms × 5 seeds × 480 evictions. |

**The measurement.** `attribution` carries `"+b"` on **1.0000 ± 0.0000** of evictions in *every*
`b_enabled=True` arm — including `γ_b = 0.0`, where `b` is identically zero and changed **0 of
2400** victims. Two of the four confusion cells are empty in every arm and every seed.
`I(attribution ; b-changed-victim) = 0.0000 ± 0.0000 bits` against a ground truth carrying
**0.97340 ± 0.00683 bits** at correction 4's own `γ_b = 0.1`. `H(changed | attribution) =
H(changed)` to every digit: conditioning on `attribution` removes **nothing**. And the mechanism
is measured rather than argued — `n_distinct_attribution_strings_max_over_arms = 1.0000 ±
0.0000` over all 16 runs: **no arm ever emitted a second attribution string across 480
evictions.** At `γ_b = 0.001`, `"psi+b"` on 2400/2400 while 51 victims changed → **97.875% false
labels.**

🔴 **The consequence is release-relevant.** §3.4's *"if `b` flips a large share of decisions, the
balance controller is the policy"* and §3.5 item 1's `ψ̂ ≪ b` are the two tests that would catch
the takeover cycle 2 documented. **Neither is computable from a run's log today**, and the field
that exists to compute them returns a constant.

**`ν` has the same defect with an instructive asymmetry.** `ν`'s guard is `config.nu != 0.0`, and
`0.0` *is* §3.7's off value, so the `ν = 0` control correctly emits no `"-nu"`. `b`'s guard is
`config.b_enabled` — **a different knob from `γ_b`** — so `b_enabled=True, γ_b=0` is a provably
inert bias reported as active, and **no value of `b_enabled` can express that state.** `ν`'s tag
is wrong about *effect*; `b`'s tag is wrong about *presence*. `b` is the worse one, and it is the
one §3.4 depends on.

**`score_margin` is not a sufficient substitute — four ways, all measured.** No threshold
separates the classes (0.947–0.996 of flipped evictions have a margin inside the unflipped
range). In correction 4's range the best possible margin rule still misreads ~30% of decisions
(0.295–0.320 error vs a 0.405–0.436 base rate). The high AUC at small `γ_b` is a **base-rate
illusion** — AUC 0.972 at `γ_b = 0.001` yet threshold error 0.01875 barely beats "assume nothing
flipped" (0.02125). The one sound use is a one-sided bound: `margin > 2·b_max` ⇒ `b` did not
decide it (`0.0000 ± 0.0000` violations across all six arms), and the converse — the direction
§3.4 needs — is worthless. **And they pre-empted the obvious fix**: logging the *pre-bias* margin
is **worse** (AUC 0.643 ± 0.031 vs the arm's own 0.686 ± 0.029 at `γ_b = 0.05`). Margins are the
wrong object; the counterfactual **victim** is the right one.

**The specification they produced is the actionable deliverable.** One field does almost all the
work: **`victim_without: dict[str, int]`** — the argmin with each optional term omitted, on the
same state. Then `b` decided it **iff** `victim != victim_without["b"]`. Cost is one extra
`argmin` over a `[M]` tensor already in registers inside `_score`, **not** a second forward pass.
Plus: `attribution` becomes effect-derived (a `γ_b = 0` run then reports `"psi"`) with the
configured set kept as a separate `terms_enabled` field — *the defect is the name, not the data*;
six post-z-scoring per-term scalars so `ψ̂ ≪ b` becomes computable at all; `b_spread_live`; and
`runner_up` beside `score_margin`, since a margin without the identity of the slot it is a margin
*to* cannot be joined to anything. `warm` is the one existing field that is already effect-true,
because the warmup branch genuinely bypasses the score.

**Unplanned cross-cycle replication.** Their `iid` regime reproduced cycle 2's numbers to every
reported digit — `0.02125 ± 0.00539`, `0.56417 ± 0.01313`, `0.59500 ± 0.01289` at `γ_b =
0.001/0.05/0.1` — from a separately written harness. That is stronger evidence the cycle-2 result
is real than either cycle alone, and it is an unplanned canary on the `γ_b` finding.

**Secondary finding, and it bears on E1's design.** In `iid`, `ν = 1.0` changes only **2.46% ±
0.68%** of victims, because 40 iid unit gestalts in `d = 384` have `mean_max_cosine 0.16514 ±
0.00478`; in a `redundant` regime (0.38087 ± 0.00467) the same `ν` is **4.6× more active**. So
**an E1 `ν` sweep on a corpus without near-duplicate gestalts will measure `ν ≈ inert`, and that
will be a property of the corpus rather than of `ν`.**

**Boundary.** ⚠️ `ū` is still synthetic and `observe()` still raises, so every rate is conditional
on that generator and `τ = 0.25`. Nothing about `γ_b`/`ν`/`τ`/`b_max`/`E[lifetime]` as constants.
Nothing about whether `b` or `ν` *helps* — "changed the victim" ≠ "made it worse", `φ` is random,
no LM ran. `"fifo_warmup"` and `"neg_age"` were never exercised, so gauntlet 0.2's mode is
untouched. One definition of "decided" (counterfactual victim change — the notion §3.4's own
sentence uses); a weaker "reinforced" notion would score higher and was not measured. CPU only.

**Two process notes.**
- 🔴 **My error, second of the same class.** My brief named baseline `4866e7b`; `HEAD` was already
  `026bef1` — I quoted the sha from an earlier `git log` instead of re-reading after committing
  the timestamp fix. They caught it and verified the delta was `docs`/`src`/`tests`-empty. Cycle 1
  was a stale *dirty-tree* claim, this was a stale *sha*. **Fix applied: the brief's baseline is
  now read from `git rev-parse HEAD` at the moment of writing, not recalled.**
- Minor, theirs: the report says the ledger stamped `git_dirty: false` at process start; the
  ledger row says `true`. Cause is their own untracked `runs/` directory, so immaterial — but the
  artifact and the prose disagree and the artifact wins.

---

## Cycle 4 — CANARY

| | |
|---|---|
| **Falsifier** | *"the environment has not moved."* |
| **Dispatched** | nobody — the canary is mine to run, not a researcher's |
| **Verdict** | **survived, and with a real comparison rather than a bare first reading.** |
| **Verified by re-execution** | ✅ **This cycle *is* a re-execution.** `runs/smoke2/heartbeat.jsonl` is a seed-0 run at `d=128, S=48, batch=8` from commit `d03b737`, made **before tonight started**. Re-running that exact config at `HEAD = 9362c68` — six commits later — reproduces it **bit-identically**: step 0 `10.817072550456` vs `10.817072550456` (rel diff `0.000e+00`), step 4 `8.130104700724` vs `8.130104700724`, and `grad_norm[0] = 507.4735412597656` in both. The environment has not moved, and tonight's six commits did not perturb the training path. |
| **Ledger** | `runs/canary/cycle-04/ledger.json`; baseline frozen at `runs/canary/baseline.json`. |

**A nuance, not a contradiction.** Cycle 1 measured `3.18e-07` run-to-run variation on MPS over 10
beats. These first two beats are bit-identical, so that variation accumulates later in a run
rather than being present from step 0. **MPS is not thereby deterministic** — the canary's
committed tolerance stays at `1e-4` relative, and the retrospective check happened to land inside
the exactly-equal regime.

### 🔴 And the canary caught something I was told as a premise

The canary's own first loss is **4.965**, not smoke2's **10.817**. That is **not drift** — it is a
different `V`, and chasing it down invalidates the framing of the run I was handed as "proven to
learn".

| quantity | value |
|---|---|
| distinct words in the synthetic corpus | **156** |
| `V` when `train()` derives it from the corpus | **160** → `ln V` = **5.0752** |
| `V` the CLI default supplies | **50257** → `ln V` = **10.8249** |
| smoke2's `V`, and its loss[0] | 50257, and **10.8171** |
| fraction of that vocab that can **never** occur | **99.682%** |

So the "loss 10.82 → 1.11 **from chance**" run started at chance **for a 50,257-token vocabulary
of which 50,097 tokens can never appear.** Chance for the *actual* corpus is `ln(160) = 5.075`.
The first ~5.75 nats of that 9.71-nat drop is the model learning that most of its output layer is
dead — which is real optimization but is **not** learning the language.

**The claim "the training loop learns" survives**, and it is worth being precise about why: final
loss **1.107** is well below `ln(160) = 5.075`, and final perplexity **3.024** against a
true-chance perplexity of **160**. That is genuine learning with a factor of ~53 in perplexity
behind it. But the honest baseline is **5.075, not 10.825**, and the interval that represents
learning the corpus is **5.075 → 1.107**, not 10.82 → 1.11.

**The canary's own baseline is internally valid**: its 4.965 at step 0 sits just below
`ln(160) = 5.075`, exactly where derived-vocab chance should be. Frozen as the reference for
cycles 8, 12, 16.

---

## Cycle 5 — is `b_max = 1.0` really one SD of `ψ̂`?

| | |
|---|---|
| **Falsifier** | *"`_score`'s z-scoring puts `b` on the scale §3.5 item 1 claims it does — `b_max = 1.0` really is one standard deviation of `ψ̂` as the policy sees it."* |
| **Dispatched** | fresh researcher, retired on hand-back |
| **Verdict** | **survived.** The normalizer is **0.98038 ± 0.01012** of `ψ̂`'s true sd, against a normal-theory prediction of **0.98111**. Item 1's sentence is off by 2.0%. |
| **Verified by re-execution** | ✅ **Their headline claim, re-run by me.** Printed `n_live= 2 measured 0.56298 ± 0.01165 theory 0.56419` and `n_live=40 measured 0.98104 ± 0.00232 theory 0.98111` — within sd of their 5-seed rows, with independent RNG ordering. I also read `policy_loop.py:145` myself and confirmed the premise-killer below. 229 ledger rows audited, `sd_zero_rows_unclassified` is `[]`. |
| **Ledger** | `runs/cycle-zscore-denominator/ledger.json` — 229 rows, CPU, 11.5 s. |

🔴 **My brief's premise was false, and that is the load-bearing finding.**
`src/rsr/model/tg/policy_loop.py:145` reads `if bool(mem.valid[row].all()): victim =
policy.select_eviction(...)` — **the policy is consulted only when the row's memory is full.**
Measured through the *real* `run_policy_loop` with a real `TGModel`:
`part0_distinct_n_live_over_real_evictions = [8]` — one value, `= M`;
`part0_frac_real_evictions_at_n_live_eq_M = 1.0000 ± 0.0000`. Memory fills 0→M monotonically and
never un-fills, so `n = n_live` is **never** "small early in a stream" *at a decision*. The
small-`n` bias is real in the estimator and **unreachable by the argmin.** I wrote that premise
into the brief from reading `_score` in isolation without checking its one call site.

**Side finding: `EvictionRecord.n_live` is a third config-echo field**, alongside cycle 3's
`attribution` and the `b_enabled`/`γ_b` mismatch. It cannot vary.

**They also declined to let me have the conclusion I'd set up for them.** The per-step-vs-pooled
ratio *is* systematically ≠ 1 — `0.93260 ± 0.00665` — which is what my item 1 was fishing for. But
`part1_frac_pooled_variance_from_between_step_means = 0.11833 ± 0.01237` and `1 − 0.9326² =
0.1303`: **the entire gap is variance of the per-step *mean*, a constant added to all live slots,
which `_score` subtracts one operation earlier** at `rsr.py:371`. A common additive offset cannot
move an argmin. Pooling across steps measures a quantity the argmin is blind to. Variance
decomposition residual `0.000000 ± 0.000000`.

### The real explanation of cycle 2's 56–60% takeover — and it is not a bug

**Two numbers, and conflating them is how item 1 reads as a protection when it is not one.**

- **In `ψ̂`-sd units, item 1's literal claim is right**: `b_max` = **0.98280 ± 0.00360** true SDs.
- **In decision units it is `3.24786 ± 0.04523`** — `b_max` is **3.25× the median top-2 z-margin**,
  and **92.278% ± 0.453%** of decisions have a margin `b_max` alone could cross.

🔴 **§3.5 item 1's argument is a true statement that does no protective work.** The quantity the
argmin turns on is not `ψ̂`'s spread — it is the *gap between the two smallest of `M` draws*, which
for `M = 40` is ~0.31 SD. One SD of protection against a 0.31 SD gap is not protection.

**So correction 4's `γ_b` range was derived against the right denominator.** Forcing the "correct"
`unbiased=True` denominator moves **0.4%** of victims (`0.00333 ± 0.00186` at `γ_b = 0.1`), and
0.08 SD of a denominator 2% low is still 0.08 SD. **Cycle 2's takeover is order statistics, not a
scaling error** — correcting the normalizer in either direction moves it ~1%, not by the factor
that would explain 56%.

**Honest residue they refused to over-claim.** Swapping the per-step sd for a *fixed global*
normalizer changes **5.792% ± 0.925%** of victims at `γ_b = 0.05`. Not the between-step mean (that
cancels) but the per-step sd's own wobble, `CV = 0.11698 ± 0.00479` — of which `σ_true`'s genuine
drift with `c_t` is only `CV = 0.03574 ± 0.00043`, the rest being sampling noise in a 40-sample
sd. Per-step normalization makes `b` stationary against the *local* spread; a global one would
make it stationary in raw units. **Neither is mis-scaled; they differ on ~5% of victims. A design
question for Brendan, and they explicitly declined to call it.**

**Methodological quality worth noting.** `σ_true` is not Monte-Carlo'd: `ψ̂` is a *linear
functional* of `s`, and `E[s sᵀ] = I/d` for the unit-norm gestalts of correction 15, so
`σ_true = ‖w_eff‖/√d` **exactly** — MC-checked at N=200k to `0.00107 ± 0.00073` against a 0.00158
noise floor. And `part3_selftest_hand_score_matches_policy_victim_* = 1.0000 ± 0.0000` on all 2400
evictions: the hand-recomputed score **is** `_score`, which is what licenses the counterfactuals.
Cross-cycle replication again: driver top-2 margin median `0.33388 ± 0.02135` vs cycle 2's
`0.3326 ± 0.0207`.

### 🔴 A correction to cycle 3, found in passing

Cycle 3's "redundant" regime used `gest[idx] + 0.15*randn(d)`, whose **noise norm is
`0.15·√384 = 2.94` — it dominates the unit-norm gestalt**, giving pair cosine ≈0.32 rather than
≈1. So that construction does not build near-duplicates the way it was described. Cycle 5
reproduced it (`loose` arm, mean-max-cosine `0.18332 ± 0.00026`) and confirms it perturbs nothing.
**Cycle 3's measured `0.38087` still stands** — it arose from duplicates *accumulating over a
stream*, which a per-group construction does not model — but the mechanism was mis-described.
**Cycle 3's directional conclusion survives on cycle 5's own evidence**: a genuinely tight arm
(mean-max-cosine `0.42423 ± 0.00098`) shows median margin **−26%** and `b_max`/margin **+36%**. So
corpus redundancy does change `b`'s strength; the number attached to it in cycle 3 came from a
different mechanism than claimed.

**Boundary.** `φ` is untrained — but they bounded what training *provably cannot* change while the
head keeps this form: `ψ̂` is linear in `s`, so uniform scaling of `φ` is **exactly** invariant
(`part4_alpha_invariance_max_abs_deviation = 0.0`, by degree-1 homogeneity), and rank-1 drift at
parity with the base term moves the conversion only 0.98144→0.97691. What training *could* change
is the **gestalt distribution**, which they showed is the real lever. `ū` is synthetic, so only
part 3's victim-change fractions are conditional; parts 0, 1, 2, 4 are unconditional. One gestalt
family (iid spherical + two duplicate arms) — **real PG-19 gestalts are neither iid nor isotropic,
and that is the axis that matters.** CPU only. And: **if a future change ever lets slots die
mid-stream or queries the policy on an underfull memory, the `n_live` curve becomes live and
`b_max` drops toward 0.56 true SD.** Today it cannot.

---

## Cycle 6 — can one FROZEN `b_max` serve every `M` the project trains?

| | |
|---|---|
| **Falsifier** | *"a single FROZEN `b_max` gives `b` comparable strength at every `M` the project trains."* |
| **Dispatched** | fresh researcher, retired on hand-back |
| **Verdict** | **falsified, twice over, on two independent measurements.** |
| **Verified by re-execution** | ✅ **Their headline claim, re-run by me from the registry's own values.** `b_max = 1.0` FROZEN; `M` by scope = 8 / 16 / 40. Printed `M= 8 → 1.9329 ± 0.0075`, `M=16 → 2.5404 ± 0.0127`, `M=40 → 3.1824 ± 0.0279` — a 1.65× swing **from a single `torch.randn`**, no head, no policy, no gestalt. I separately confirmed their correction-4 arithmetic and the `S`-scope claim (see below). 186 rows audited, `sd_zero_rows_unclassified` is `[]`, `git_dirty: false`. |
| **Ledger** | `runs/cycle-bmax-scope-dependence/ledger.json` — 186 rows, 7.6 s, CPU. |

**Order statistic:** `b_max / median-top-2-z-margin` = **1.88772 ± 0.05003** at `M = 8`, **2.59245 ±
0.05256** at 16, **3.17490 ± 0.08201** at 40 — swing **1.68244 ± 0.04855** over a 13-point sweep,
with exact-iid-normal theory reproducing the whole curve to 2–3% (`|ψ̂ − normal| ≤ 0.0994` at every
`n`). **Realized victim-flip rate** with `γ_b` from correction 4's own formula at each `M`:
**0.46713 ± 0.01702 → 0.53359 ± 0.02150 → 0.59500 ± 0.01289**, a **12.8 ± 2.4 pp** spread (≈5 sd).

**The finding I did not anticipate, and it is the good one.** Correction 4's `γ_b = 4/M` is **doing
real work against this defect even though it was not derived to.** Hold `γ_b` fixed at the `M = 40`
value of 0.1 and the flip-rate swing nearly **doubles to 1.84636 ± 0.08705** (0.323 → 0.485 →
0.595). So the formula's `1/M` closes roughly half the gap the order statistic opens — it
*over*-corrects relative to the ~1/1.68 the order statistic wants, and lands at a residual 1.28×.

### 🔴 Correction 4's formula and its own stated range disagree, at the `M` D-5 funds

Verified by me directly from the registry: `γ_b = b_max/(0.25·E[lifetime])` with `E[lt] = M` gives

| `M` | `γ_b` | correction 4's stated range |
|---|---|---|
| 40 | **0.1000** | top of *"order 0.05–0.1"* ✅ |
| 16 | 0.2500 | 2.5× over |
| **8** | **0.5000** | **5× over the top of its own range** |

**Both the range and the formula were written for `M = 40`.** At `M = 8` — exactly the E7 small-`M`
model that D-5 / correction 6 funds and that release condition 4 rests falsifier 4 on — they
disagree by 5×. This is a defect in an authoritative document, not in code.

**A third `M`-dependence, not in my brief.** `realized / FIFO` lifetime is **0.922 → 0.832 →
0.656** across `M` = 8/16/40, so the *two parameterizations* of correction 4's formula diverge
**more** at larger `M`. "Which `E[lifetime]`" is itself scope-dependent, not a global 0.66 factor.
Realized lifetimes under the learned-head driver: **7.37708 ± 0.07148 / 13.31693 ± 0.32914 /
26.23583 ± 0.47394** against FIFO's exactly 8/16/40. The `M = 40` figure replicates cycle 2's
26.49 ± 0.53.

**A direction warning worth keeping.** At `M = 8, γ_b = 0.5`, `b` saturates **more** (64.7% vs
54.3%) and `mean |b|` is **higher** (0.719 vs 0.641), yet flips **fewer** victims. **Any future
defence of a frozen `b_max` that argues from `|b|` reaching `b_max` is arguing about the wrong
quantity.** And `S` is not driving any of it: `M = 16` re-run at its own registry `S = 48` differs
from `S = 80` by −0.004 / −0.000 / −0.012, all within 1 sd of zero.

**The three registry options, costed, none picked** — correctly left to Brendan. **(A) `b_max`
per-scope:** mechanically cheapest, but forfeits §4.5's one-line interpretation *"one standard
deviation because `ψ̂` is z-scored"* — which cycle 5 measured to be **correct** — and moves `b_max`
out of FROZEN into an experiment-owned constant, creating a second knob A5 must cross with.
Equalizing the *ratio* puts `b_max ≈ 0.59` at `M = 8`; equalizing the *flip rate* is a different
number, so **the option needs a stated invariant before it has a value.** **(B) `γ_b`'s formula
absorbs `M`:** smallest diff and already half-realized by accident, but changes a formula §3.5 item
4 states and correction 4 pins, and **conflates two jobs in one number** — a *timescale* (reach
`O(b_max)` in a lifetime, what correction 4 derives) and a *decision authority* (how much of the
argmin `b` owns, what the order statistic sets). **(C) The spec accepts it:** zero code, but must
be written into §13 *and* the E7 writeup, because the anti-collapse loop has ~13 pp less authority
in the model the human-correlation claim rests on. **(C) is free iff `b` is not load-bearing for
the E7 claim** — which nobody has established.

### Two more corrections to my brief, and one self-correction of theirs

- 🔴 **`M = 16` does have a registry scope** — `get("M","synthetic")` = 16, FROZEN, §5.1. My brief
  said it might not. All three `M` were registry reads; none invented. What *is* missing is **`S`
  for the `M = 8` scope**: I confirmed `get("S","e7")` raises `ScopeRequired` and
  `get("S","corpora")` raises `UnknownScope`, so the small-`M` model's stream length had to be
  supplied and was, labelled.
- **Theirs, self-reported:** they mislabelled a row as "FIFO lifetime" when it was the oldest live
  slot in the *learned* driver's memory (24.3 at `M = 8`, not 8), caught it, re-measured with a
  real `FIFOPolicy` on its own memory, and **kept the wrong row under its true name** because
  §12.4 means recording that too. That is the right call.
- Cosmetic: they numbered their report "Cycle 5" and refer to my cycle 5 as "cycle 4" — they have
  no visibility of my cycle counter. Their internal cross-references are self-consistent.

**Boundary.** `φ` untrained, `ū` synthetic — parts 1–2 (the order statistic) involve **neither**
and are unconditional; parts 3–4 (flip rates) are wholly conditional on the synthetic `ū`.
**Whether the `M`-dependence survives training is not established** — only that it is present,
~1.68× wide, and a property of `n` under the near-Gaussian law now measured to hold. "Changed the
victim" is still not "made it worse": no LM, no loss, no `r_i`, no LOO Δloss, so whether the
`M = 8` model's weaker `b` is a *problem* needs E0e/E3. One gestalt family (iid spherical); `ν = 0`
and §3.7's reduction path untouched, and the `−ν·max cos` term is the other thing in `_score` that
could interact with `M`. No `S`-control at `M = 8`. CPU only.

---

## Cycle 7 — is the fix even `M`-shaped?

| | |
|---|---|
| **Falsifier** | *"`b`'s realized decision strength is a function of `M` alone — so any `M`-indexed fix is a complete fix."* |
| **Dispatched** | fresh researcher, retired on hand-back |
| **Verdict** | **falsified, on three counts, one of which I did not anticipate.** |
| **Verified by re-execution** | ✅ **Their part-G check, re-run by me**, reproduced **bit-exactly**: `M effect 1.6420 ± 0.0714`, `redundancy effect 2.5928 ± 0.1074`, `redundancy / M = 1.5790`. **And I independently verified their correction of my own brief** by reading the two prior ledgers directly: cycle 5's tight ratio `4.4075 / 3.2479 = 1.357` against cycle 6's same-readout `M` swing of `1.682`. 838 rows, `sd_zero_rows_unclassified = []`, every mean±sd in their writeup machine-checked against the ledger (0 unmatched), and prior-cycle numbers **read at runtime from those ledgers rather than retyped.** |
| **Ledger** | `runs/cycle-redundancy-vs-M/ledger.json` — 838 rows, 76 s, CPU. |

**The three counts.** (1) At fixed `M = 40`, redundancy moves `b_max/margin` by **2.37102 ± 0.21026**
(live memory) or **4.03253 ± 0.11747** (static groups), against a whole-`M`-range effect of
**1.65205 ± 0.17053** — crossing at achieved mean-max-cosine **0.541**, found at the same place by
two independent constructions. (2) 🔴 **The `M` gap itself moves with redundancy**: the flip-rate gap
an `M`-keyed `b_max` would be tuned to close is **1.19266 ± 0.05566** at the control and **1.49851 ±
0.11214** at high redundancy — **one `M`-keyed constant cannot close a gap that is a function of the
other variable.** (3) 🔴 **The sign flips with `M`**: redundancy *weakens* `b` at `M = 8` (**0.88401 ±
0.05446**) and *strengthens* it at `M = 40` (**1.10803 ± 0.03838**), straddling 1.0 about 3 sd apart.

**They bounded their own headline honestly.** On the *flip-rate* readout, redundancy (1.108×) stays
**below** the `M` effect (1.314×) everywhere reachable — 0 of 5 seeds crossed. The margin readout is
where redundancy wins. The verdict rests on counts 2 and 3, which hold on **either** readout. That
is the right way to carry a split result.

### 🔴 Two errors in my brief, both real

- **I claimed cycle 5's +36% redundancy swing was "larger than the entire `M = 8 → 40` swing." It
  is not** — 1.357 vs 1.684, about half. I compared a *ratio* swing against a *flip-rate* swing
  (1.275), which is the only reading on which my sentence is true. **Verified against the prior
  ledgers myself.** This is the fifth brief of seven with an error, and it is the first where the
  error was in the quantitative premise rather than a mechanism claim.
- **I asked for redundancy to be measured as achieved mean-max-cosine. That is not a sufficient
  statistic.** Two constructions at the *same* achieved cosine `0.51621 ± 0.00064` differ by
  **1.48406 ± 0.04645×**. What shrinks the margin is a near-tie between **two** slots — the *upper
  tail* of the pairwise-cosine distribution, not its mean. **So a cosine-keyed fix would be
  miscalibrated for the same reason an `M`-keyed one is**, and even a measured corpus mean-max-cosine
  would not settle the question.

### The mechanism, and it ties back to cycle 5

**Broad correlation is nearly flat, and that is a finding rather than a null.** At `M = 40`, going
from achieved cosine 0.111 to 0.624 moves the ratio only **1.17951 ± 0.10620** and the flip rate
**1.0107 ± 0.0383**. Why: a component shared by *every* live slot is a **common additive offset** in
`ψ̂`, and `_score` subtracts the per-step mean one operation before z-scoring — **the argmin is blind
to it.** That is the same mechanism cycle 5 used to dismiss the per-step-vs-pooled gap, arriving from
a different direction. **Anisotropy behaves the same way**: collapsing effective dimension from
**384.03 ± 0.11** to **4.11 ± 0.04** (achieved participation ratio, finite-`N` bias removed exactly)
moves the ratio only ~1.43× — at a *higher* achieved cosine (0.771) than the near-duplicate cell
(0.731) that moved it 2.37×. **Three constructions, same insufficiency conclusion.**

So my brief's premise — *"the margin distribution already moved +36% at fixed `M`"* — does not
generalize to redundancy as such. It was a property of cycle 5's **near-duplicate** structure
specifically.

**Another cross-cycle replication:** the `α = 0` live flip rate is `0.59500 ± 0.01289`, identical in
every digit to cycle 6's `M = 40` row, from a differently written generator.

### The registry question, re-costed — and a fourth option

**(A) per-scope `b_max` keyed on `M`** now costs *more* than cycle 6 thought: it closes the gap at
one redundancy and opens it at another (1.193 → 1.499), and redundancy's sign differs between `M = 8`
and `M = 40`. **(B) `M` factor in `γ_b`**: smallest diff, unchanged by this cycle, but `γ_b` has no
redundancy dependence to exploit — and the flip-rate readout it acts on is a **low-sensitivity
instrument** (a −17% margin buys +6.6% flips, because `b` is saturated most of the time). **(C)
`b_max` normalized by a *measured* margin** — `κ ·` trailing median `score_margin`, which the policy
**already logs** — is the only option that compensates for **both** axes at once, because the margin
is the variable both act *through*, and it sidesteps the sufficiency problem entirely since no
geometry statistic has to predict anything. Costs: `b_max` stops being FROZEN and becomes a second
online feedback path with untested stability, `b` stops being comparable across steps, it needs a
warm-up interacting with `T_warm`, and §3.5 item 1 gets **replaced rather than amended**. **(D)
`b_max` should not be frozen at all** — stated plainly because I asked for it: on cycles 5–7 it is a
FROZEN constant whose only justified reading does no protective work and whose decision-relevant
value ranges **1.88–12.99×** the margin depending on `M` and geometry, i.e. **a free parameter that
was never swept.**

🔴 **What all four need and none has: one sentence naming the invariant.** Equal flip rate? Equal
`b_max`/margin? Equal fraction of decisions `b` could cross? **The missing measurement is the
invariant, not more numbers** — and this harness supplies all three for any candidate in ~80 s.

**Boundary, and they were scrupulous about the important one.** **Nothing about PG-19** — no PG-19
gestalts exist in this repo, none were used, `φ` was never trained on text. **Whether a real corpus
sits above or below the crossing at 0.541 is not measured and not measurable from anything here.**
Marked as conjecture only: cycle 5's tight arm reached 0.424 and loose 0.183, both *below* 0.541, so
*if* a real corpus sits there the `M` axis dominates in practice — but §13 forbids reading a
synthetic number as a corpus number. `ν = 0` throughout, so §3.4's own redundancy term was never
exercised. Under high redundancy a flip is often between two near-equivalent slots, so **the flip
rate may over-state harm exactly where redundancy is high** (unquantified). `f = 0.75` is a stress
test, not a plausible corpus. `φ` untrained and `ū` synthetic; the order-statistic halves involve
`ū` not at all.

---
