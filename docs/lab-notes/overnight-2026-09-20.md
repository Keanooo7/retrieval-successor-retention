# Scoreboard — 2026-09-20

Manager's record. One row per cycle. **"We learned nothing this cycle" is a writable row**, and a
cycle with zero rejections and zero re-executions is an unexamined one, not a good one.

---

## Cycle 1 — `cycle-01-masked-loss` · verdict **`survived`** (mps) · `inconclusive` (cpu)

| | |
|---|---|
| **run_id** | `cycle-01-masked-loss` (mps, headline) · `cycle-01-masked-loss-cpu` (replication) |
| **brief** | `docs/lab-notes/dispatch-cycle-01-masked-loss.md` @ `3beb2b7`, corrected at `8a8a116` and `eedc0c0` |
| **falsifier** | PAD fraction `< 0.50`, **or** masking moves the final training loss by no more than the run-to-run floor |
| **expected** | Registered in `5a33b9d`, ahead of the run: masking **raises** the reported per-token loss; a lower masked loss would be a finding, not a bug |
| **observed** | PAD `0.9343`. Real-token NLL `3.967117 ± 0.015492` → `2.010772 ± 0.132726`, a **`1.956` nat cut, `66,184×`** the yardstick floor `2.956e-05` |
| **verdict** | **`survived`** on mps. `inconclusive` on cpu, where the floor is exactly `0.0` and half 2 discriminates nothing |
| **commits** | `5a33b9d` (pre-registration + fix + test + mutation) · `e76c42e` (anchor) · `471650a` (ledgers + results) |

### What I re-executed myself, and how

🔴 **By a different path than the claimant used. Reading their artefact reproduces their work; it
does not verify it.**

| Claim | My path | Result |
|---|---|---|
| PAD fraction `0.9342551256613757`, `180,812 / 193,536` | 🔑 **Blind, before they reported.** Re-encoded the corpus through `encode()` and counted over the `[:, 1:]` target slice. Never imported `step_fn`, never trained. The number was deliberately withheld from the brief so this comparison would mean something. | ✅ **Exact to every printed digit**, including `pad_source_B = 0` |
| `passed=323 failed=0 skipped=0 errors=0` | `uv run pytest -q -rs --tb=no` to a file, `$?` on the next line, **no pipe** | ✅ `EXIT 0`. Floor rose `318 → 323`, nothing skipped |
| `37/37 gates proven`, new gate isolated | `uv run python scripts/mutation_battery.py --check` to a file, `$?` on the next line | ✅ `EXIT 0`, `37 PROVEN / 0 UNPROVEN`; *"the training loss scores padding again"* → 1 on gate, **0 off, 0 undeclared** |
| The stale anchor blocked 12 mutations since `4ece429` | `git show <sha>:scripts/ledger.py` at five shas; counted `Mutation` entries at or after index 24 | ✅ present at `46c208e`, **absent from `4ece429` onward**; **12**, which confirms their own erratum against `e76c42e`'s message saying 13 |
| The pre-registered expectation was not edited after the run | `git diff 5a33b9d 471650a -- RESULTS.md`, removals only | ✅ the sole deletion was the `<!-- RESULTS PENDING -->` placeholder |
| Every prose number is ledger-backed with spread | Opened each row rather than trusting the key list | ✅ `kind: statistic` with `n`, `samples`, `mean`, `sd`, `sd_exactly_zero: false`. **No sd is zero** |
| The `3.1789143850602386e-07` floor | Opened `runs/cycle-arm-identity/ledger.json` instead of quoting the prose | ✅ ledger-backed, but its `how` is a **trajectory sup-norm at 10 beats on MPS** — the wrong comparator, which is why the brief was corrected at `eedc0c0` |
| Instrument sanity | `math.log(160)` | ✅ `5.075174` vs measured first beat `5.074075` |

### 🔴 Rejected — one claim, and it is the downstream one

📌 **Also filed in the record itself**, at
`experiments/cycle-01-masked-loss/RESULTS.md` §4, next to the claim it rejects, and the NEXT list
there is re-ordered behind it. *A defect filed as a lesson recurs; this one is now in the file a
future session reads instead of the return it was born in.*

The report states this **"retires the third of the three candidate causes"** for the inert memory.
**Rejected.** The researcher supplied the grounds themselves under *BELIEVED, NOT VERIFIED*: the
shuffle control **has no positive control** and has never been shown to read large on a
deliberately live memory. **A hypothesis cannot be retired by an instrument that has not been
shown to register a positive.**

That is the *same disease this cycle just repaired* — a number that looked like a measurement and
was an artifact of the instrument. Applying the defect's own test to its sibling, per ROADMAP §6,
is what turns it up.

The CPU arm sharpens it: the shuffle delta there is **negative**, `-7.651746273040771e-06`,
against a floor of exactly `0.0`. On that device it is **not noise**, and *"memory helps less than
nothing"* is not a reading anyone should bank. **Until S0-04 gives the control a positive, every
`≈0` in this project is uninterpretable in exactly the way the perplexities were.**

**What survives is the instrument repair, and nothing about memory.** Per the manager's standing
rule, every training number in this cycle is about a model whose memory contributes nothing —
recorded prominently rather than filed silently.

### What this cycle cost me, which is the useful part

🔴 **My revalidation declared bar item 2 "executable as written" on the strength of a `grep` that
`off_gate_allowed` existed. The battery was red at that moment**, and had been since the merge,
while `docs/mutation-battery.md` went on asserting 36/36. **I checked that a flag existed, not
that the gate ran** — in a brief whose entire subject is instruments that look green. The
researcher caught it and it is the single most valuable line in their report.

My correction at `eedc0c0` earned its keep in the other direction: the CPU arm returned
`inconclusive` under the vacuity rule rather than a comfortable `survived` off a `0.0` floor.

**Two of the night's corrections were mine against my own brief**, both found by re-executing
rather than re-reading: the retracted `iters=8` arithmetic (`8a8a116`) and the floor's statistic
and device (`eedc0c0`).

---

# S0-02 — the capture bridge. Manager's review.

**Verdict: ACCEPTED.** Gates green and re-executed, not read. One claim of my own is retracted
below, and one defect is filed against the cycle.

## What I re-executed rather than believed

| Claim | How I checked it | Result |
|---|---|---|
| `passed=343 failed=0 skipped=0 errors=0` | `.venv/bin/python -m pytest -rs -q`, `$?` read unpiped | ✅ exit `0`, census matches. Baseline was 323; **+20** is `tests/test_capture_bridge.py` |
| The two headline tests are not skipped | ran `test_reduction.py` and `test_fidelity.py` alone | ✅ `passed=44 failed=0 skipped=0`. **They genuinely run** |
| `W_O` mutation reddens **only** the new tests | applied the mutation by hand, ran the **full** suite, restored | ✅ exactly **4 distinct** node ids, all in `test_capture_bridge.py`, **0 off-gate** |
| Throughput rows | opened `runs/s0-02-capture-bridge/ledger.json`, not the prose | ✅ `388.797±1.130 / 360.460±0.396 / 329.963±0.935` — prose matches JSON |
| The transcribed number is flagged | opened row 15's `how` | ✅ names 🔴 *"Transcribed from a terminal, not from a JSON file"* in the row itself |
| `.gitignore` fix does not over-track | `git ls-files runs/` | ✅ 70 files, no `.pt`, `.git` 26 MB |

⚠️ **What I did not re-execute:** the other 37 mutations. `--check` is 39 full-suite runs; I
verified the one the brief's Bar item 3 rests on and the researcher's own `mutations.json` for the
rest. Stated so nobody reads "39/39" here as mine.

## 🔴 RETRACTED — my amendment's bar-4 argument was wrong, and it is the same error I reviewed

I wrote in `dispatch-S0-02-capture-bridge.md` that Bar item 4 was safe because *"`observe()` is on
the protocol, on `FIFOPolicy` and on `RSRPolicy` (`rsr.py:427`)."*

**`RSRPolicy.observe` raises `NotImplementedError` at `rsr.py:433`.** An unconditional `observe()`
call — the plain reading of bar 4 — would have turned `test_checkpoint.py` red on contact, via the
very call sites I cited as reassurance. **I checked that a method existed, not that it ran.**

That is verbatim the failure recorded three sections above in this same file, against Brendan's own
brief: *"I checked that a flag existed, not that the gate ran."* I reviewed that sentence and
reproduced it the next day, in an amendment whose subject was whether a dependency was real.
**Reached ≠ usable**, and the distinction is the one this project keeps paying for.

The researcher resolved it without routing around it — `observe=False`, opt-in, which Bar item 2
independently wanted — and drew the honest consequence: **`rsr.retention.reward` still has zero
real importers in `src/`.** The producer exists; the consumer is a stub. *Anyone reading this cycle
as "the bridge is wired end to end" is reading it wrong.*

## 🔴 FILED AGAINST THE CYCLE — ADR-0008's three figures are prose-only

`1.49 × 10⁻⁷`, `2.98 × 10⁻⁸` and `0.0383` carry the ADR's whole argument: the first that the
collapse is algebra, the second that mean-vs-sum is not a fork, the third that EOS-only is. None has
a ledger row (the ledger has 20, none of them the collapse comparison), no raw artefact, and the ADR
says only *"Measured on a real forward pass"* — **no command.**

📌 **This is the `310 ± 2 sent/s` disease, in the cycle that diagnosed it.** A real figure from a
real run that a reader cannot regenerate. Not fatal and not a retraction: the ADR's *claims* are
pinned by tests that run in CI — `test_mean_and_sum_collapse_give_the_same_r_i`,
`test_eos_only_collapse_is_a_different_measurement`, `test_the_collapse_reproduces_the_real_increment`
— which is in one way stronger than a ledger row. But the tests assert **thresholds**, not these
values. **What is missing is a producer script and a row; write them before ADR-0008 is cited
again.**

## What the cycle got right that is worth naming

🔑 **Bar item 3's own test was vacuous first, and the battery is what caught it.** The original
perturbed `attn_out_proj` on every cross block and stayed green under its own mutation, because
layer `l`'s output moves layer `l+1`'s attention — it detected *"the model changed."* Three other
tests were silently carrying the gate. This is the project's own rule paying out on the cycle that
invoked it, and it is a better result than the bridge.

📌 **The pre-registered expectation that `r_i` is cheaper than the bridge was falsified** and
reported as such: `−30.5` vs `−28.3 sent/s`. Launch-bound, not FLOP-bound — the same finding E0c
made about the dispatch loop. A cycle that reports its own failed prediction is the point.

⚠️ **`${PIPESTATUS[0]}` returned empty in zsh** (it is `$pipestatus[1]`). Visible only because it
was blank rather than `0`. **An idiom that silently returned the pipe's status would have read `0`
and meant nothing** — this belongs in the standing rules, not in one report.

## Numbers, for the record

| arm | mean ± sd (3 repeats) | Δ vs off |
|---|---|---|
| `capture_off` | **388.80 ± 1.13** sent/s | — |
| `capture_on` | **360.46 ± 0.40** | **−28.34 (−7.29 %)** |
| `capture_on_ri` | **329.96 ± 0.94** | **−58.83 (−15.13 %)** |

Free-when-off measured, not asserted: unmodified `e0c/measure.py`, same machine, same sitting —
`369 ± 11` pre-bridge vs `370 ± 9` with the bridge present. ⚠️ The pre-bridge figure is the one
transcribed from a terminal. **`310 ± 2` is correctly not used as a denominator anywhere.**

Three `peak_gb` rows carry sd `0.0` / `4.35e-15` and the ledger flags them. Accepted as **not** the
broken-seed shape: 2-dp-rounded deterministic allocation, and the throughput rows *from the same
three trials* carry sd `1.13 / 0.40 / 0.94`. That contrast is the discriminator, and it is the
reason the standing rule says to report spread rather than to fear a zero.

## Still open, and not this cycle's fault

**S0-01 has not landed** — `tests/test_train_loop.py` absent, no `--policy` flag, `srep_norm` out of
the objective, `--vocab` still 50257. **S0-04 is not unblocked**: it has a live *producer* now, but
`RSRPolicy.observe` is a stub, so the positive control still has no live memory to read large on —
which is the same rejection filed against cycle 1 at the top of this file, still standing.

---

# B1 — ADR-0008's provenance debt. Manager's review.

**Verdict: the three headline numbers reproduce exactly. The ADR is still not citable**, for a
reason B1 found that the debt did not predict.

## The debt is paid, and I checked it from the artefact rather than the report

`runs/s0-02-capture-bridge/qtok_collapse.json`, read directly:

| ADR prose | artefact | |
|---|---|---|
| `≤ 1.49 × 10⁻⁷`, values up to `8.23 × 10⁻¹` | `1.4901161193847656e-07`, `0.8234491348266602` | ✅ |
| `r_i` max abs difference `2.98 × 10⁻⁸` | `2.9802322387695312e-08` | ✅ |
| EOS-only max abs difference `0.0383` | `0.03832480311393738` | ✅ |
| *"differs by exactly `Q_real = 5`"* | `contribution_ratio = [5.0, 5.0, 5.0, 5.000000476837158]` | ✅ to float32 |
| `contribution()` sum slot 2 = `5.8777` | `5.877645…` → **`5.8776`** | ⚠️ 4th-decimal slip, corrected |

**8 of 9 published values agree exactly**, and `grep -c` over the generated ledger goes `0 → 17`.
`n_seeds: 1` and `spread: NONE` are written into the artefact **with the reason** — deterministic
float32 identity checks on a fixed fixture, not sampled estimates. That is the right way to record
an absent spread, and it is **not** the sd-of-0.0000 shape the standing rule forbids.

## 🔴 The finding that matters — ADR-0008's EOS table compares two collapses that share no query position

I re-derived this from the fixture's own definitions, without running the model:

```
TGConfig.L = 1 + max_sentence_tokens + sentence_tail_len = 1 + 8 + 1 = 10
_sentence: ids[:, -1] = eos_id      -> [EOS] is at query index 9
_sentence: mask[0, 5:] = 0          -> row 0's real query positions are 0..4
```

**On row 0 — the row every number in ADR-0008 comes from — `[EOS]` sits at a PAD position.** The
sum-collapse runs over positions 0–4; the EOS-collapse reads position 9. **They overlap nowhere.**
So `0.0383` is a divergence *guaranteed by the fixture's construction*, not one found in the
attention, and the ADR presents it as evidence that EOS-only "does not cancel."

The researcher measured row 1 alongside — full mask, `[EOS]` a real query token, `q_real = 10` —
and got **`0.00286`, 1.38%** against row 0's **13.78%**. 🔑 **The decision survives: EOS-only still
does not cancel, so sum-over-real-tokens stands.** What changed is the quality of the evidence, by
an order of magnitude. The artefact now carries `eos_is_a_real_query_token` and
`eos_overlaps_the_summed_positions` as explicit machine-readable fields, so this cannot be
re-published silently.

📌 **B1 was filed as a provenance debt and returned a substantive defect.** The three CI tests
pinning ADR-0008 assert thresholds and pass on *both* rows at `atol=1e-3`; no threshold test could
have caught a wrong description of the fixture. *That is the argument for producers over tests,
made by the case rather than in the abstract.* ADR-0008's fixture line was also wrong — it said
`Q_tok = 8` with a 3-token tail; it is **10** with a **5**-token tail — and the producer's own drift
guard caught it on first run.

**Owner's call, deliberately not taken by the researcher or by me:** whether ADR-0008 publishes row
1, both rows, or keeps row 0 with the caveat.

## 🔴 MY DEFECT — I put two researchers in one working tree, and two of this project's tools are mutually exclusive there

B1 is **BLOCKED on one command**, and the block is mine.

`ledger.write()`'s `DirtyTree` gate is **repo-wide** over `SOURCE_ROOTS = ("src/", "scripts/",
"experiments/")` (`scripts/ledger.py:84`). `scripts/mutation_battery.py` works by mutating a source
file, running the suite, restoring, and repeating — so a battery run keeps `scripts/` dirty
essentially continuously. **Any researcher running the battery blocks every other researcher's
ledger write for the whole run.** That is not a narrow window; it blocked B1 for its entire session.

🔴 **And it is worse than a scheduling collision. The battery mutates `scripts/ledger.py` itself, at
eight sites** (verified at `3973d1a`) — the same file that holds the gate. So a ledger written
during a battery run may be written **under deliberately broken gate logic.**

🔑 **The researcher stopped retrying on purpose, and that judgement is the best thing in the
return.** One refusal listed `runs/s0-02-capture-bridge/ledger.json` as a *source* path, which is
impossible under `SOURCE_ROOTS`; the battery had `scripts/ledger.py` mutated at that instant, its
diff showing `if self.doc["status"] is None:` replaced by `if False:`. **A blind retry loop that
eventually succeeds is a loop that may succeed inside a mutation window — a gate made to pass.**
Killing the loop and returning BLOCKED was correct, and a report that says *"did not run"* rather
than *"found nothing"* is the standard.

**I also restored `runs/s0-02-capture-bridge/ledger.json` to `HEAD`.** The generated copy sitting in
the tree was produced at `c49d80f`, before the `str`-vs-`float` fix at `bbb3b9d`, and carried
`headline_three_reproduce: false` — a **false** *"the ADR's headline figures do not reproduce"*,
aimed straight at the row future work joins on. That is B2's failure mode inverted, and with a
concurrent agent committing in the same tree, one `git add -A` would have landed it. The file is
regenerable from committed sources; the risk was not worth keeping it.

**Remaining: one command, once `scripts/` is genuinely clean and `scripts/ledger.py` unmutated** —
`.venv/bin/python experiments/s0-02/write_ledger.py 0`, then commit the ledger.

**This deserves its own brief, and the cheapest fix is the one I got wrong: do not put two
researchers in one working tree.** The alternatives — narrowing `DirtyTree` to the paths a run's
`commands[]` actually name, or making the battery take a lock — are real options, but they are
changes to the evidence machinery and should not be made to unblock a session.

### ✅ B1 unblocked and closed at `88257b1` — ADR-0008 is citable

The section above recorded B1 as BLOCKED. That was true when written and is now closed. The
researcher waited for the battery to finish rather than letting a retry win, and **checked the
specific hazard rather than the general one** before acting:

```
git status --porcelain                                    -> `?? uv.lock` only
git diff --stat scripts/ledger.py scripts/mutation_battery.py  -> EMPTY, i.e. ledger.py unmutated
.venv/bin/python experiments/s0-02/write_ledger.py 0      -> WRITE_EXIT=0
```

🔑 *That second command is the one that matters.* A clean `git status` alone would have been the
blind check; confirming `scripts/ledger.py` itself is unmutated is the check that addresses why the
loop was killed. **Verified at `HEAD` by the manager:**

| | |
|---|---|
| `grep -c "1.49\|2.98\|0.0383"` over the ledger | **`17`** (was `0`) |
| rows / commands / status | `20 → 27` · `7 → 8` · `ok` |
| `headline_three_reproduce` | **`True`**, `n_agree` 8 of 9 |
| sole disagreement | `mean_vs_sum.contribution_sum` — the genuine `5.8777` → `5.8776` ADR slip |
| `verdict` block | **untouched** — still `survived`, `capped_from: null`. Correct: these rows are a provenance debt, not a falsifier, and B1 had no business moving F-capture-free's outcome |

📌 **The flawed ledger never entered history.** `git log -- runs/s0-02-capture-bridge/ledger.json`
ends at three commits — `68bab05`, `3973d1a`, `88257b1`. The restore-to-`HEAD` above closed the
window rather than undoing a landing; nothing had to be reverted.

⚠️ **Neither finding above is retired by this.** The EOS-at-a-PAD-position defect stands and the
row-to-publish is still the owner's call. And **the `ledger.write()` / `mutation_battery.py`
exclusion stands** — B1 got through on timing, not because the hazard went away. *A hazard survived
by scheduling is not a hazard fixed.*
