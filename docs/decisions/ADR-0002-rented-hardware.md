# ADR-0002 — Rented hardware: provider, capacity budget, and the fidelity tolerance

- **Status:** Part A draft (pending T8); **Part B accepted — the tolerance is committed, 2026-09-17, and NO fixture exists yet**
- **Date:** 2026-09-17
- **Relates to:** kickoff T5 (E0c), T8 (procurement); corrections 7, 8, 10, 14

One ADR covers both because both concern the same rented box: what we rent, what
we extract on it, and what we measure on it.

## Part A — the capacity budget (correction 10)

§4.2 fixes the maximum feasible `(S, d, batch)` triple from E0c "on 64 GB". **64 GB
is the Mac Studio.** But E3, A2/A4 and the E7 model all run on rented commodity
cards, and §4.1 prices "A40/A6000 rates" — both of which are **48 GB**. The fit
decision would be made against ~33% more memory than the hardware that has to run
it, and §7.6's fallback ladder would fire in week 6 — the exact outcome §7.6 exists
to prevent.

**Therefore E0c runs on the card type T8 holds, not on the Mac.** It doubles as the
provider smoke test, run *before* the hold is taken.

| Field | Value |
|---|---|
| Provider | ☐ TBD (T8) |
| Exact SKU | ☐ TBD |
| **VRAM, the binding budget** | ☐ TBD |
| On-demand rate | ☐ TBD |
| Hold: ≥4 GPUs, movable start, week 6 | ☐ TBD |

§4.2's tie-break — *"if `S = 80` does not fit, `S` wins and `d` is cut"* — is
evaluated against **that** number, not 64.

**Log per-trial free memory alongside peak-resident**, so a pass is not an artifact
of a warm cache.

> Recorded for the record: on the development machine, ~25 GB of the 64 GB is
> routinely held by unrelated local model servers. That alone would have made a
> Mac-side E0c unreliable, independently of the 48-vs-64 problem.

## Part B — the golden-tensor fidelity tolerance

Replaces the kickoff's smoke test against 29.8 test PPL / 21 sentence-steps/sec,
which is unreachable because the released code self-declares as differing from the
paper's model (correction 14, ADR-0001).

### Extraction protocol

- Pinned JAX TG at `f220b109`, on the rented box. **JAX is never installed on the
  development machine.**
- Fixed seed, fixed batch, **`d = 128`**, ~20 sentence steps, `M = 40`, `S = 30`.
- Committed to `tests/fixtures/tg_d128_seed0.npz`.

### What is dumped

| Tensor | Why |
|---|---|
| Per-layer activations | The transcription's backbone |
| Gestalt vectors `s_t` | **Note they are L2-normalized to 1.0** — correction 15 |
| Cross-attention weights `alpha` | Feeds `r_i`; also the E0h collinearity check |
| Logits | End-to-end forward |
| **Gradients w.r.t. `W_sent`** | **Mandatory — see below** |
| **Gradients w.r.t. transformer params** | **Mandatory — see below** |

### Why the gradient fixtures are mandatory

Gestalts are appended **without detaching the computation graph** (§3.1), and
backward depth is bounded by stream length `S` rather than by memory capacity `M`
— evicting a slot does not free its graph (§3.6, [P2] App. A).

**JAX functional autodiff and PyTorch retained-graph semantics diverge exactly
there.** A transcription that matches on every forward quantity while retaining the
wrong graph passes a forward-only check and survives to week 7, where it surfaces
as a gradient-flow difference in E3 that looks like a finding. Forward agreement is
necessary and not sufficient.

### The tolerance — committed 2026-09-17, before any fixture exists (D-H, gauntlet 2.2)

A tolerance chosen after seeing the mismatch is not a tolerance — the same
discipline as the E0i pre-registration. **`git log` must show this commit preceding
the fixture commit**, and that ordering is the evidence, not this sentence.

Both sides run in **float32**, with matched dtype and matched dropout
(deterministic, `srep_dropout` disabled), before any tolerance is argued about.

| Quantity | rtol | atol | Rationale |
|---|---|---|---|
| Per-layer activations | `1e-4` | `1e-5` | the forward default |
| Gestalt vectors | `1e-4` | `1e-5` | unit-norm by construction (`srep_norm_target = 1.0`), so `atol` is directly interpretable as a fraction of the vector's own length |
| Cross-attention weights | `1e-4` | `1e-5` | a simplex; `atol` dominates, and at `M = 40` a uniform row sits at 0.025, so `1e-5` is ~0.04% of a typical entry |
| Logits | `1e-4` | `1e-5` | the forward default |
| **Gradients** | **`1e-3`** | **`1e-4`** | **one order looser, and here is why, in advance** |

### Why gradients get exactly one order of magnitude more room

Two reasons, and both are properties of the comparison rather than of the
transcription:

1. **Gradients accumulate error.** A gradient at layer 0 is a product of per-layer
   Jacobians over 12 blocks *and* over `S` sentence steps, so float32 rounding
   compounds along a path the forward never traverses. One order is the
   conventional allowance and it is not tuned to anything observed.
2. **JAX and PyTorch reduce in different orders.** Summation order is not specified
   by either framework and differs by backend, and floating-point addition is not
   associative. The backward pass performs far more reductions than the forward,
   over longer axes, so the discrepancy is systematically larger — and it is a
   discrepancy between two correct implementations, not an error in either.

**One order, not two.** A looser allowance would start absorbing the very failure
the gradient fixtures exist to catch: the retained-graph divergence, which shows up
as a *structural* difference in which paths carry gradient at all, not as accumulated
rounding. A wrong graph is off by a large factor or by everything, not by `1e-3`.

### If the transcription cannot meet these

🔴 **Record the achieved value here, with the reason. Do not silently relax.** D-H.

The entry takes this form, and the ADR keeps both numbers:

> **MISS — <quantity>.** Committed `rtol = X, atol = Y`. Achieved `rtol = X', atol =
> Y'`. Cause: … . Why the achieved value is nonetheless sufficient for the claim
> `test_fidelity.py` makes: … .

A miss that cannot be explained mechanically is a transcription bug, not a tolerance
problem, and the response is to fix the transcription.

### Where the numbers live

`src/rsr/model/tg/tolerances.py`, as a frozen dataclass, read by **both** the
fixture generator and `tests/test_fidelity.py`. One definition, so the extraction
cannot be generated against one number and asserted against another.

## Consequences

- `tests/test_fidelity.py` is **skipped** until the fixtures exist. The tolerance
  half of the blocker is now closed; the extraction half is not. It must not be
  reported as passing in GATE-1.
- The transcription is on the Sprint 1 critical path behind the extraction, which
  is behind T8.
- §13 gains an entry: the PyTorch TG is a transcription of a JAX reference that is
  itself not the paper's model. **Two hops from the published numbers.**
