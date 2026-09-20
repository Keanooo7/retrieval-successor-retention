# Cycle 1 — the training loss scores padding

Brief: `docs/lab-notes/dispatch-cycle-01-masked-loss.md` (body at `46c208e`,
revalidation at `3beb2b7`, first correction at `8a8a116`, second correction at
`eedc0c0`).

---

## 🔒 Pre-registration — written before the measurement, and not edited afterwards

> **Expected result.** Masking the loss to real tokens should **raise** the
> reported per-token loss, because PAD is trivially predictable and currently
> dominates the mean. **A masked loss that is *lower* is a finding, not a bug to
> tune away**, and it would falsify the reading.

**Falsifier.** The reading dies if **either**:

1. the fraction of scored targets that are PAD is **below 0.50**; or
2. masking the loss moves the **final training loss** by **no more than the
   run-to-run floor** at identical seed and config.

**The floor statistic, pinned.** `|final-beat difference|` between two runs
differing in nothing — same seed, same config, same code, separate processes.
This is deliberately **not** the `3.1789143850602386e-07` of
`runs/cycle-arm-identity/`, whose `how` field says it is
`max_t |loss_x[t] - loss_y[t]| over the 10 beats` — a **trajectory sup-norm**,
which is `≥` the final-beat difference by construction. The sup-norm is reported
beside the floor for continuity and is **not** the bar.

**Device, pinned: `mps`.** On CPU the run-to-run floor is `0.0`
(`runs/cycle-arm-identity/RESULTS.md`), which reduces the falsifier's half 2 to
*"the delta exceeds zero"* — satisfied by one float ulp. That is a vacuous
discriminator, and `decide()` in `run.py` returns **`inconclusive`**, not
`survived`, whenever the measured floor is exactly `0.0`. That rule is in the
commit that precedes the run.

**`iters = 50`, stated because the falsifier does not name it.** It is `train()`'s
own committed default. The manager's `iters = 8` reconstruction was retracted
before this run (`8a8a116`) as a coincidence of factorisation, so adopting it
would have been adopting arithmetic.

**Decision rule, pre-registered** — `experiments/cycle-01-masked-loss/run.py::decide`:

| condition | verdict |
|---|---|
| `pad_fraction ≤ 0.50` | `falsified` |
| floor `== 0.0` | `inconclusive` (half 2 is vacuous on this device) |
| `delta ≤ floor` | `falsified` (the defect is real and inert) |
| otherwise | `survived` |

---

## Results

*Filled after the run. Every number below is a `runs/cycle-01-masked-loss/ledger.json`
key; none is typed.*

<!-- RESULTS PENDING -->
