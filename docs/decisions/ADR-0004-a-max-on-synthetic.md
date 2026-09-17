# ADR-0004 — `A_max = S` on the synthetic corpus

- **Status:** accepted
- **Date:** 2026-09-17
- **Relates to:** spec §3.6(a), §4.5, §5.2; correction 9

## Context

The spec never states `A_max` for the synthetic corpus, and §3.6(a) contains a
string that reads like a prescription and is not one:

> "With `S = 30`, sweeping `A_max ∈ {16, 32, 64}` leaves two values inert and
> produces a flat curve that is an artifact, not a finding."

That is an **illustration of a failure mode at `S = 30`**. §4.5's `A_max` row says
only *"Swept only where `S` makes it bind."*

Read as a prescription, `{16, 32, 64}` makes `γ = 0.97` look **illegal** on
synthetic: at `S = 48`, §3.6(a)'s `A_max ≤ S` admits only `{16, 32}`, so
`min(S, A_max) ≤ 32 < 33 = 1/(1−0.97)`. Since §4.5 freezes `γ` from the synthetic
sweep **before** PG-19, that would make 0.97 unreachable at the point it must be
chosen.

## Decision

**`A_max = S = 48` on synthetic. Non-binding by construction.**

Checked against the actual constraint `1/(1−γ) ≤ min(S, A_max)`:

| Configuration | `S` | `A_max` | `min` | Horizon at γ=0.97 | Legal? |
|---|---|---|---|---|---|
| Synthetic (E1, E2) | 48 | **48** | 48 | 33 | **yes** |
| PG-19 E3 headline | 80 | 64 (forced — §3.6 ties measurable gap `k` to `A_max`, and E3 targets `k` to 64) | 64 | 33 | **yes** |
| A2 ablation (`A_max = M`) | 80 | 40 | 40 | 33 | **yes** |

`γ = 0.97` is legal everywhere it is used. `γ = 0.99` stays dropped: horizon 100
exceeds `S = 80` itself.

**This is a reading of §3.6(a), not a change to it.** Enforced in
`src/rsr/constants.py`, where `A_max` is `CONDITIONAL` and a read without `S`, or
with an `A_max` above `S`, raises.

## What this does NOT decide

**It does not pre-empt E1.** `γ` is swept `{0.5, 0.9, 0.97}` on synthetic and
frozen at whichever wins **on merit** (§4.5). This ADR establishes only that the
constraint does not exclude 0.97.

## Pre-registered consequence, recorded before E1 runs

E3's target window is gaps `(40, 64]`. The weight the return gives an event at that
distance:

```
γ = 0.90    0.90^40 = 0.015     0.90^64 = 0.0012     →  0.1–1.5% of the return
γ = 0.97    0.97^40 = 0.296     0.97^64 = 0.142      →  14–30% of the return
```

At `γ = 0.90` a slot needed 64 steps out contributes **one tenth of one percent**
of its own discounted return; the value function cannot express the claim E3 is
built to test. §4.5 says as much in prose — *"v0.4's `γ ≤ 0.9` left an order of
magnitude of horizon unused at `S = 80`"* — and this is the arithmetic behind it.

**If E1 freezes `γ` below 0.97, E3's `(40, 64]` window is known in advance to be
under-weighted in the return, and that must be stated in the E3 writeup as a
limitation rather than discovered as a null.** Written here, before E1 runs, so it
cannot be reconstructed afterwards.

## Caveat to state in the synthetic config

At `γ = 0.97` the horizon (33) is **shorter than the generator's max gap (40)**, so
the longest synthetic gaps are attenuated to ~0.30 weight. **Attenuated, not
invisible.** Acceptable; state it.
