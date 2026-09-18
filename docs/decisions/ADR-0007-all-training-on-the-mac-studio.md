# ADR-0007 — Everything trains on the Mac Studio; procurement is dropped

- **Status:** accepted
- **Date:** 2026-09-17
- **Supersedes:** ADR-0002 **Part A** (the capacity budget), and corrections **7**,
  **8** and **10**, all of which reason about a rented card
- **Leaves standing:** ADR-0002 **Part B** — the fidelity tolerance is unaffected
- **Relates to:** spec §4.1, §4.2, §7.6, §16; gauntlet Phase 2

## Decision

**No outsourced compute, at all.** No GPU is rented, no provider is named, no quote
is requested, no hold is taken, and nothing is priced. Every experiment the approved
scope contains runs on the Mac Studio (M4 Max, 64 GB).

**Procurement is out of scope, not deferred within it.** It is not a blocker, not a
dependency, and not a line item. It is reconsidered only *after* the model has
demonstrated its effect locally — a later conversation that nobody is having yet,
and one this ADR does not prepare for.

ADR-0002 Part A, `GATE-1`'s T8, and corrections 7, 8 and 10 are superseded by this
ADR and should be read as historical.

**Nothing is lost.** §16 approves weeks 1–4 — E0a–E0i, E1, E2 — and **E1/E2 are
synthetic at `M = 16, S = 48`**, which fits on this machine several times over. The
rental was only ever attached to the block §16 does *not* approve.

## Why this is a simplification and not a compromise

The rented card entered the plan for three things. Two are gone and the third was
never load-bearing.

1. **The golden-tensor extraction (2.3).** Thought to need a GPU because
   `jax-metal` is dead. It does not: **`jax-metal` is the Metal *GPU* backend**, and
   `jaxlib` ships `macosx_11_0_arm64` **CPU** wheels. The extraction is seconds of
   CPU work on a 2.48M-parameter model, and it is **done** — 23.48 MB of fixtures,
   `sha256 79edafe9…`, byte-reproducible across processes.

   CPU is not a fallback here, it is the correct target: fixtures must be
   byte-reproducible, and that is ADR-0001 D3's own argument for running E0b on CPU
   rather than MPS.

2. **The capacity budget (old E0c / 2.7).** Correction 10's point was real — §4.2
   sized against 64 GB while E3 was going to run on a 48 GB rental, so the fit
   decision would have been made against ~33% more memory than the hardware that
   had to honour it. **With no rental, the asymmetry disappears.** The machine that
   sizes the run is the machine that runs it, which is the condition §7.6's fallback
   ladder exists to protect and never had.

3. **Parallel capacity for weeks 5–7 (correction 8's 842 GPU-h).** Weeks 5–7 are
   **outside the approved scope.** CLAUDE.md is explicit: *"Only weeks 1–4 are
   approved (§16). Everything downstream is a projection, not a permission."*
   Pricing capacity for unapproved work was the thing §16 declined to approve, and
   the honest response to that is not a cancellable hold — it is to stop pricing it.

## What replaces old 2.7

The question worth answering is not "what fits on a card we might rent" but **"what
fits on this machine, at the widths we will actually run."** That is the new
Phase-2 item, and it is strictly cheaper: it is a measurement on hardware already
in hand, with no procurement in front of it.

> **Measure the `(S, d, batch)` ceiling on the Mac Studio** for the small-width
> training the approved scope contains — `d ∈ {128, 256}`, `S ∈ {30, 48, 80}`,
> `M` per scope — and commit the triple.

§4.2's rule is unchanged and now applies to a machine we control: **if `S = 80` does
not fit, `S` wins and `d` is cut.** Gaps longer than `S` do not exist in the data, so
`S` caps the headline experiment; width does not.

⚠️ **The inherited warning inverts.** The old note was *"the Studio has more memory
than the rental card, so do not let a Mac-side batch size leak into the E3 config."*
There is no rental card, so there is nothing to leak into — but the underlying
discipline survives in a different form: **the measured triple is a property of this
machine and must be reported as one.** §13's no-cross-corpus rule has a hardware
analogue, and a number measured here is not a claim about anything else.

## What does not change

- **Do not raise `iogpu.wired_limit_mb`.** It is 0, the system default.
- **`21 sent/sec` is not a plan input.** It was measured at `d_model = 768` /
  85.6M parameters. The new E0c measures the real number on this machine at the
  widths that will run — §12.4: measure, don't extrapolate.
- **MPS stays best-effort, not required** (ADR-0001 D3). CPU serves E0b's
  determinism, the coordinate check, the synthetic corpus and debugging. All code
  must still run on CUDA without modification — that is a portability property, not
  a procurement plan.
- **The scope boundary.** This ADR does not expand what is approved. It removes a
  purchase from weeks 1–4; it says nothing about weeks 5–7, which remain a
  projection.

## Measured, 2026-09-17 (E0C)

`experiments/e0c/RESULTS.md` carries the table. Two things it changed about how the
question is asked:

1. **`ru_maxrss` is a process high-water mark.** Measuring a grid in one process
   makes every row inherit the peak of the row before it. One subprocess per point.
2. **RSS does not measure MPS allocations.** The first MPS run reported 0.44 GB for
   a configuration that needs ~26 GB on CPU, because Metal memory does not appear in
   the process's resident size. On unified memory it is the same 64 GB either way,
   so `torch.mps.driver_allocated_memory()` is the figure that competes with
   everything else. A number that low should have been read as a broken metric, not
   as good news, and it nearly was not.

## Consequence for §13

§13 gains an entry, replacing the rental one: **every throughput and capacity number
in this project was measured on a single M4 Max.** Nothing here supports a claim
about training cost or feasibility on other hardware, and the paper must not imply
one.
