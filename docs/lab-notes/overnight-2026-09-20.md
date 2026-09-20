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
