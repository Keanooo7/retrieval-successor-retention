# S0-02 — the AttentionTrace capture bridge

**Baseline:** `c647af4290f7a45be2edf947b7973816dcc2af8b`. **Lane:** researcher · device optional. **Sprint:** 0.
**Depends on:** S0-01 landed.

## Hypothesis (not instruction)

`src/rsr/retention/reward.py` is 160 lines, real, with 11 passing tests, and **zero callers in
`src/`**. It computes `r_i` — §3.2.1's reward, the quantity the whole mechanism rests on. I
believe it has no callers because the model physically cannot produce its input, and that this one
gap is why E0d, E0e, E0h, `RSRPolicy.observe`, H2O, the oracle and the shadow buffer are all
blocked at once.

## What `AttentionTrace` wants, and what exists

| Field | Wanted | Present |
|---|---|---|
| `alpha` | `[L, H, M]` | `[B, H, Q_tok, M]` — `CrossAttention.forward` stores `self.last_attention` at `model/tg/model.py:335`, collected into `StepOutput.cross_attention` |
| `wo_v` | `[L, H, M, d_model]` | **nothing.** `attn_out_proj` is applied to the already-α-weighted output at `model.py:337-338`, so per-slot per-head `W_O v` never exists as a tensor |
| `gate` | `[L]` | `block.memory_gate`, an `nn.Parameter` at `model.py:388`, not surfaced |

## 🔴 The decision this brief must make, and record

**How does `Q_tok` collapse to `[L, H, M]`?** Mean over real (non-PAD) tokens? The EOS position
only? A sum? **It changes `r_i`, and therefore it changes E0d's answer about whether `r_i` is a
confound.** This is not an implementation detail and it must not be decided in a commit message.

**Open `docs/decisions/ADR-0008-qtok-collapse.md` and state the choice, the alternatives, and what
would distinguish them.** §3.2.1 already requires reporting the per-layer profile before collapsing
across layers — the same argument applies one axis over, and the spec does not address it. If you
think the spec implies an answer, quote the passage; if it does not, say so.

## Files in scope

`src/rsr/model/tg/model.py` · `src/rsr/retention/policy.py` · `src/rsr/model/tg/policy_loop.py` ·
`tests/test_capture_bridge.py` (new) · `docs/decisions/ADR-0008-qtok-collapse.md` (new).

Also in scope, because they are the same gap: **call `policy.observe()` in `run_policy_loop`**
(`git grep '\.observe(' -- src/` returns **zero hits** today) and **call `policy.reset()` at
stream boundaries** ([P2] fact 2 — memory is reset at each boundary, and §3.5's `b` resets with it).

## Bar

1. `reward.retrieval_demand` runs **end to end on a real forward pass**, not on a hand-built
   fixture. Its 11 existing tests must still pass unchanged.
2. Capturing must be **off by default and free when off.** Measure and report the throughput delta
   with capture on versus off at `d=128, S=80, batch=16`. E0c's measured baseline is **310 ± 2
   sent/s** on the RSR arm.
3. 🔴 **A mutation that zeroes `W_O` in the capture path must change `r_i`.** §3.2.1: *"`W_O`
   is not optional… dropping it reintroduces the confound the norm-weighting was adopted to
   remove."* If `r_i` is unchanged, the capture is wrong and the test is vacuous.
4. `observe()` is reached: a counter, or a test that fails if the call site is removed.

## Done when

Gates green, ADR-0008 committed, the throughput delta reported as a number, and a PR merged.

## Do NOT

- **Do not modify `third_party/`.** It is a pinned read-only reference and `test_fidelity.py`
  compares against tensors from exactly that tree.
- **Do not change the forward pass's numerics.** `test_fidelity.py` must stay green to its
  committed tolerances. Capture is observation; if it moves a number, it is a defect.
- Do not implement the shadow buffer, `ProtectionBias`, or any metric here.
- Do not decide the `Q_tok` collapse silently.

## Report

Standard block, plus `BRIEF ERRORS`, the throughput delta with spread, and **which mutation turns
only the new tests red**.

---

## ⚠️ AMENDED 2026-09-20 by the manager, at `d537cc88cdff31b5bec6d4822a3a53b5bb23578a`

Re-baselined from `c647af4`, which is stale. Commissioned by
`docs/lab-notes/dispatch-2026-09-20c-push-and-the-capture-bridge.md` Task 2. Every claim below was
re-executed on this tree, not quoted.

### 🔴 The dependency header is wrong. S0-01 has **not** landed.

This brief's header says **"Depends on: S0-01 landed."** It is not. Measured at this sha:

| S0-01 deliverable | State |
|---|---|
| `tests/test_train_loop.py` (its named new file) | **absent** |
| defect (b) — `policy = FIFOPolicy()` unconditional, no `--policy` flag | **open**: `loop.py:213-215`, argparse `:365-379` has no `--policy` |
| defect (c) — `srep_norm` in the objective | **open**: `grep -c srep_norm src/rsr/train/loop.py` → `0` |
| defect (e) — `--vocab` default 50257 | **open**: `loop.py:369` |
| an S0-01 `RESULTS.md` | **none anywhere** |

⚠️ Note `loop.py` is now **401 lines, not 307**, and `tests/test_train_loss.py:31` *does* import it
— both are cycle 1's doing, for defect (a) only, which S0-01 explicitly disowns.

### The dependency is nominal, and that is a measurement, not a preference

Proceeding anyway is a decision, so here is what it rests on and where it stops.

1. **Zero file overlap.** S0-01's scope is `src/rsr/train/loop.py`, `src/rsr/retention/rsr.py`
   `from_registry` only, and `tests/test_train_loop.py`. This brief's scope is
   `src/rsr/model/tg/model.py`, `src/rsr/retention/policy.py`, `src/rsr/model/tg/policy_loop.py`,
   `tests/test_capture_bridge.py`, `docs/decisions/ADR-0008-qtok-collapse.md`. Disjoint.
2. 🔑 **Bar item 4 does not need defect (b) fixed.** `observe()` is on the *protocol*
   (`policy.py:216`), on `FIFOPolicy` (`fifo.py:35`) **and** on `RSRPolicy` (`rsr.py:427`). A call
   to `policy.observe()` in `run_policy_loop` is reached whichever policy is passed. And
   `run_policy_loop` takes `policy` as an argument — it constructs nothing — so the unconditional
   `FIFOPolicy()` in the *trainer* is not on this path. `tests/test_checkpoint.py:90,390-393`
   already drive `run_policy_loop` with a real `RSRPolicy`.
3. **Bars 1, 2 and 3** are a forward pass, a throughput delta and a `W_O` mutation. None touches
   S0-01's files.

⚠️ **Where this stops.** I checked file overlap and the `observe()` protocol. I did **not** prove no
dependency exists. **If you hit one, stop and report it — do not route around it.** That is the
finding, and it is worth more than the bridge.

📌 **Do not treat this as licence for S0-04.** S0-04's dependency on *this* brief is substantive,
not nominal: `reward.py` has zero callers, which is precisely the live memory S0-04's positive
control needs. That one still waits.

### 🔴 Bar item 2's yardstick is corrected. Do not quote `310` as a training rate.

Bar item 2 says *"E0c's measured baseline is **310 ± 2 sent/s** on the RSR arm."* That figure
appears in eight documents and traces to `experiments/e0c/RESULTS.md:58,67`, which was really run —
so this is a provenance gap, not a fiction. But:

- **It has no ledger row.** Checked by parsing every `runs/*/ledger.json` for a value near 310:
  **0 rows.** Stronger than expected — **no ledger in this project holds a throughput key at all**
  (`sent_per_s`, `sent/s`, `throughput`: no file matches). Cite `e0c/RESULTS.md`, never a ledger.
- **`docs/RESEARCH-CONTEXT.md:736-744` disqualifies it as a training baseline**: random tokens, one
  data shape per row, **no optimizer step**, MPS only — *"a measurement of the model's throughput,
  not of a training loop's."*

🔑 **The number this bar wants is the capture-on versus capture-off delta, measured in one harness,
in one sitting, on one device.** Report it as an absolute rate for each arm plus the delta, with
the spread over 3 repeats. **Do not report it as a percentage of training throughput, and do not
quote `310` as a training rate.** Prediction under test: capture-off after the change is within
noise of capture-off before it.

### Everything else in this brief was revalidated and holds

`reward.py` 160 lines · 11 `def test_` · **zero** importers of `rsr.retention.reward` in `src/` ·
`git grep '\.observe(' -- src/` → zero · `last_attention` at `model.py:335` · `memory_gate` an
`nn.Parameter` at `model.py:388` · `policy_loop.py` exists. ⚠️ One correction: `attn_out_proj` is
applied at **`model.py:337`**; `:338` is a comment, so the brief's `:337-338` overstates by a line.

**The 🔴 `Q_tok` collapse decision and ADR-0008 are unchanged and remain the centre of this brief.**
