# ADR-0002 — Rented hardware: provider, capacity budget, and the fidelity tolerance

- **Status:** **DRAFT — the tolerance must be committed BEFORE the golden tensors are generated**
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

### The tolerance

**☐ TO BE FILLED AND COMMITTED BEFORE GENERATING THE FIXTURES.**

A tolerance chosen after seeing the mismatch is not a tolerance — the same
discipline as the E0i pre-registration.

| Quantity | rtol | atol | Rationale |
|---|---|---|---|
| Per-layer activations | ☐ | ☐ | |
| Gestalt vectors | ☐ | ☐ | unit-norm, so atol is directly interpretable |
| Cross-attention weights | ☐ | ☐ | a simplex; atol dominates |
| Logits | ☐ | ☐ | |
| Gradients | ☐ | ☐ | expect looser than forwards; **say how much looser and why, in advance** |

Both sides run in float32 with matched dtype and matched dropout (deterministic,
`srep_dropout` disabled) before any tolerance is argued about.

## Consequences

- `tests/test_fidelity.py` is **skipped** until this ADR is filled and the fixtures
  exist. It must not be reported as passing in GATE-1.
- The transcription is on the Sprint 1 critical path behind the extraction, which
  is behind T8.
- §13 gains an entry: the PyTorch TG is a transcription of a JAX reference that is
  itself not the paper's model. **Two hops from the published numbers.**
