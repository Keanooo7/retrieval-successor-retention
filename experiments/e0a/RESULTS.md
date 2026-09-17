# E0a — μP coordinate check

| | |
|---|---|
| Kill gate | **Yes** (§6) |
| Status | ⚠️ **PARTIAL — value-head half complete, bare-TG half BLOCKED** |
| Blocker | `docs/decisions/ADR-0001-tg-base.md` — no TG implementation available |
| Command | `uv run python experiments/e0a/run.py` |
| Hardware | Apple M-series, CPU tensors (width-scaling check; no accelerator needed) |
| Env | Python 3.13.5, torch 2.14.0 |
| Seeds | 0–4 (five) |

## What ran

The value-head half. `ψ̂_φ(s_i, c_t) = s_iᵀWc_t + uᵀ[s_i;c_t]` at `d ∈ {128, 192, 256, 384}`,
`n_live = 40`, fitted to a Θ(1) per-slot regression target under MSE for 8 AdamW steps — the shape
of `L_MC` (§3.3), so the coordinate scales are the ones training will actually produce.

The bare-TG half did not run. It cannot until there is a TG.

## Result

```
psi (the scalar output the spec names)
  step  d=128       d=192       d=256       d=384      drift
  0     0.07452     0.07490     0.06903     0.04835    0.351   <- init
  1     0.17687     0.13813     0.15422     0.16644    0.219
  2     0.31616     0.27051     0.28773     0.30638    0.144
  4     0.57928     0.52099     0.54921     0.56628    0.101
  5     0.69929     0.62979     0.66973     0.68202    0.099   <- minimum
  8     1.00271     0.88380     0.97178     0.96553    0.119
```

`drift` = `max |rms/rms(d=128) − 1|` across widths. **0 is perfect width-invariance.**

### Across five seeds

| quantity | mean | sd | range |
|---|---|---|---|
| drift at init | **0.419** | 0.107 | 0.323 – 0.541 |
| drift at step 8 | **0.120** | 0.034 | 0.075 – 0.156 |

And the decisive number:

| | |
|---|---|
| `ψ̂` RMS ratio `d=384 / d=128` at init, **predicted** by `Θ(1/√d)` | `1/√3 = 0.577` |
| **measured**, mean of 5 seeds | **0.581** |

A 0.7% match. The init-regime prediction in §4.3 is confirmed quantitatively, not just in sign.

## The finding, and it changes how E0a must be read

**At init the output is not width-invariant, and that is correct.**

`0.0484 / 0.0745 = 0.649` against the `1/√3 = 0.577` predicted by `Θ(1/√d)`. §4.3's footnote says
the bilinear form is `Θ(√d)` at init and `Θ(d)` once `s`, `W` and `c` are correlated; with the `1/d`
multiplier that is `Θ(1/√d)` at init and `Θ(1)` trained. **The measurement matches the footnote.**

Within one optimizer step the drift falls 0.351 → 0.219, and by step 4–5 it sits near 0.10 and
stops falling. That is the correlated regime arriving, which is the regime μP is governed by.

> 🔴 **Therefore the E0a verdict is read at `t ≥ 1`. Step 0 is recorded and never gated on.**
>
> A coordinate check run at init would show this head's output falling as `1/√d` and read as a μP
> violation. Anyone who "fixed" it by switching the multiplier to `1/√d` would have tuned the head
> for the init regime and broken it for the trained one — which is exactly the failure §4.3's
> footnote warns about, met from the other side. The prescription is the same either way; **the
> wrong argument gives the wrong multiplier**, and here the wrong argument is reachable from a
> perfectly ordinary measurement.
>
> This is written into `src/rsr/mup/coord_check.py`'s docstring so it cannot be rediscovered the
> expensive way.

## Limitations — what this does NOT establish

- **The residual 0.12 drift is noise, not a defect — but it is not zero either.** On seed 0 the
  `d = 192` column looked systematically ~10% low; across five seeds the low column moves (seed 2's
  is `d=128`, seed 4's is `d=256`), so there is no width trend. At `n_live = 40` the RMS of 40
  samples carries roughly `1/√80 ≈ 11%` sampling error, which is the size of what is left. **Raising
  `n_live` is how you would shrink it**, not changing the parameterization.
- **The head is isolated.** §4.3: *"Recurrence through a retained graph is untested territory for
  μP. E0a is a check, not a proof."* This runs the head standalone, so it is narrower still — it
  says nothing about the recurrence, and the recurrence is the untested part.
- **Inputs are i.i.d. Gaussian, not gestalts.** Real `s_i` and `c_t` come from the same transformer
  and are correlated with each other from the start. The correlated regime arrives here only
  through `W`; in the real system it arrives sooner and differently.

## Gate status

**DEFERRED, not passed.** The exit criterion is "activation RMS width-invariant through the
recurrence, `ψ̂` output included." There is no recurrence to run it through yet. What is established
is the harness, the multiplier, the parameter grouping, and the reading rule above.

## Next

1. ~~≥3 seeds on this half~~ — **done, 5 seeds, above.**
2. The bare-TG half, once `src/rsr/model/tg/` runs.
3. Then the attached run, which is the one the spec actually asks for — and the only one that
   touches the recurrence, which is the part [P4] does not cover.
