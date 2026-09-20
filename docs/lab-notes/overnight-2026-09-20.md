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
