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
