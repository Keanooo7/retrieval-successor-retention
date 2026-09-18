# GATE-1 — Sprint 1 report

**Status: IN PROGRESS.** This is the shape the report will take, not the report.

Every criterion carries pass / fail / not-run **with an evidence link**. Per
`CLAUDE.md`: **never report a skipped or unrun test as passing.**

| # | Task | Exit criterion | Status | Evidence |
|---|---|---|---|---|
| T1 | Scaffold, CLAUDE.md, CI, constants registry, policy protocol | CI green; `pytest` passes; `docs/spec-corrections.md` committed | ☐ | |
| T2 | Pre-register E0i's threshold | committed **before** any histogram is computed | ☐ **drafted, unsigned** | `preregistration/e0i_threshold.md` |
| T3 | E0i — coref histogram + 100-item precision/recall | histogram vs the T2 threshold | ☐ | `experiments/e0i/RESULTS.md` |
| T4 | E0g — name and obtain the E7 stimulus set | availability, licensing, position-independence; **report regardless of outcome** | ☐ | [ADR-0005](../docs/decisions/ADR-0005-e7-stimulus-set.md) |
| T5 | E0c — memory and throughput **on the Mac Studio** | a committed `(S, d, batch)` triple, measured on the machine that will run it | ☐ | `experiments/e0c/RESULTS.md` |
| T6 | E0d — `r_i` vs LOO Δloss | per-layer profile, then Spearman ρ | ☐ | `experiments/e0d/RESULTS.md` |
| T7 | E0a — μP coordinate check ×2 | activation RMS width-invariant; `ψ̂` output scale checked specifically | ☐ | `experiments/e0a/RESULTS.md` |
| ~~T8~~ | ~~GPU procurement~~ | **DROPPED** — no GPU is rented; everything runs on the Mac Studio | ✔ n/a | [ADR-0007](../docs/decisions/ADR-0007-all-training-on-the-mac-studio.md) |
| T9 | This report | | ☐ | |

## Task 0

**Answered: yes, with three qualifications that reshaped the sprint.** TG's
reference implementation is public but is JAX/Flax, is self-declared as differing
from the paper's model, and ships a non-runnable corpus path. See
[ADR-0001](../docs/decisions/ADR-0001-tg-base.md).

## §16 condition 4 decision (E7)

☐ Pending [ADR-0005](../docs/decisions/ADR-0005-e7-stimulus-set.md) sign-off and
the "in hand" exit. Current finding: **a usable set exists** — Raccah et al. 2024,
CC0, with position-independent per-event importance — so the expected decision is
that E7 proceeds and falsifier 4 stands. All three of the spec's own named
candidates fail the spec's own criterion.

## Findings that changed the plan, for the record

1. **Task 0** — above.
2. **Correction 15** — the reference L2-normalizes every gestalt to unit norm, so
   §4.3's μP premise ("two `d`-dimensional Θ(1) vectors") does not hold and the
   bilinear multiplier needs rederiving. **Flagged, not resolved.** E0a decides.
3. **Correction 9** — §3.6(a)'s `A_max ∈ {16,32,64}` is an illustration, not a
   prescription; `A_max = S = 48` on synthetic keeps `γ = 0.97` legal
   ([ADR-0004](../docs/decisions/ADR-0004-a-max-on-synthetic.md)).
4. **Correction 10** — E0c was budgeted against 64 GB (the Mac) while E3 runs on
   48 GB rentals. **Superseded by ADR-0007: nothing is rented, so E0c measures the ceiling on the Mac Studio — the machine that sizes the run is the machine that runs it.**
5. **Correction 8** — §4.1's weeks-5–7 total is 768 GPU-h; §8 also schedules the
   E7 model and A1 into that window → 842, which breaks the stated floor of 3.

## Schedule honesty

T5, T6 and T7 sit behind the PyTorch TG transcription. The golden-tensor
extraction is **done** (gauntlet 2.3); T8 is **dropped** (ADR-0007). If they have not completed they are
reported here as **in progress with a dated forecast**, not passed on a forecast.
The kickoff's own note is that the week-4 gate is the milestone whose slip slips
everything, so a slip is stated rather than absorbed.

## Tests currently skipped, and therefore NOT passing

Current suite, literal, at `33f5ad6`:

```
passed=199 failed=0 skipped=7 errors=0
```

**Seven skips, both reasons true for their tests:**

- `tests/test_fidelity.py` (6) — blocked on the **golden-tensor extraction**, which
  is behind T8. ADR-0002's tolerance half is now **closed** (committed `0c200d3`,
  before any fixture exists — gauntlet 2.2), so the remaining blocker is the
  extraction and the transcription, not the tolerance.
- `tests/test_reduction.py::test_loss_curve_is_bit_exact_against_stock_tg` (1) —
  blocked on the PyTorch TG transcription.

**The rest of `test_reduction.py` now runs.** It was skipped at module level under a
reason false for most of it (gauntlet 0.6): the §3.7 off-switch contract and the
eviction-rule reduction need no TG, and were therefore enforced by nothing. Twelve
tests there are live, including the mutation that reddens the reduction.

## Exit codes — `3` is not `0`

`did not run` must never be reported as `found nothing`. In ML a run that produced
no metric looks exactly like a run that produced a bad one.

| Item | Code | Meaning here |
|---|---|---|
| E0i threshold | **3** | `p` unmeasured until T3; the gate **cannot be evaluated**, and that is not a pass. See `preregistration/e0i_threshold.md` §1. |
| `test_fidelity.py` | **3** | unrun, pending the extraction |
| E0b loss curve | **3** | unrun, pending the transcription |
| E0b reduction + off-switch contract | **0** | passing, and mutation-proven |

## Mutation battery

`docs/mutation-battery.md`, **17/17 gates proven**. Two came back UNPROVEN on the
first run and are recorded there rather than fixed away — a gate no mutation reddens
adds nothing, and saying so is the finding.
