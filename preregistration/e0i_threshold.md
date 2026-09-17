# Pre-registration — E0i events-per-bucket threshold

| | |
|---|---|
| **Registered** | 2026-09-17 |
| **Governs** | E0i, the week-1 CPU kill gate (§6), and §16 condition 1 |
| **Protects** | E3's 576 GPU-hours |
| **State of the data at registration** | 🔴 **No histogram has been computed. No PG-19 subset has been loaded. No coref pipeline has been run.** `git log` is the check on that claim, and it is why this file is committed before `experiments/e0i/` contains anything. |

The spec calls this threshold "pre-registered" and never gives the number (D-2). Registering it
after seeing the histogram makes the gate unfalsifiable — you can always find a floor the data
clears. So it is written down first, with its derivation, and with what would make it wrong.

---

## 1. What counts as a reintroduction event

Fixed before measurement, because the definition is a researcher degree of freedom and a generous
one would inflate the count this gate is about.

An event is a sentence at stream position `j` that mentions entity `e`, where:

1. `e`'s most recent prior mention in the same stream is at position `i`, and the **gap** is
   `k = j − i`;
2. `i` and `j` are in the **same sentence stream** at `S = 80`. Cross-stream pairs do not count —
   memory is reset at stream boundaries ([P2]), so no policy could have retained anything;
3. `e` is a **named entity or a definite nominal**. Bare pronouns are excluded as the *reintroducing*
   mention: a pronoun at gap 50 is far more likely to be a coref error than a genuine long-range
   reintroduction, and including them is the easiest way to manufacture events;
4. `e` has ≥2 mentions in the stream (trivially implied, stated so the counting script is unambiguous).

**Structural ceiling, already known and not a finding:** at `S = 80`, a gap of exactly 64 can be
hosted only at positions `i ∈ [1, 16]` — **16 positions per stream**. The spec derives this and it is
correct. It is a property of the design, not of the corpus, and it is the reason this gate exists.

## 2. Buckets

Three bins of width 8 across the target window, fixed now:

```
(40, 48]    (48, 56]    (56, 64]
```

Reported alongside, not gated on: `(0,8] (8,16] (16,24] (24,32] (32,40]` and `(64, ∞)`.

## 3. The threshold

> ### 🔴 **PASS requires ≥ 150 precision-adjusted events in EVERY ONE of the three target bins, and ≥ 600 across `(40, 64]` in total.**

"Precision-adjusted" means `n_adjusted = n_raw × p`, where `p` is the coref precision measured on
the 100 hand-annotated reintroductions required by T3. **The raw count is not the gate.** D-2 names
coref as *"an unnamed dependency with its own error rate feeding straight into the dependent
measure"* and then prescribes a population check — which cannot see precision. This closes that.

If `p` is not measured, E0i returns **exit 3 — did not run.** Not a pass.

### Derivation

The headline comparison is paired: the same reintroduction sentences scored under RSR, FIFO, H2O
and RSR(γ=0). Pairing removes sentence-level difficulty variance, so the relevant quantity is the
SD of the **per-sentence paired loss difference**, `σ_d`.

For a paired two-sided test at α = 0.05 with 80% power:

```
n  ≈  (1.96 + 0.84)² · σ_d² / δ²  =  7.85 · σ_d² / δ²
```

Taking `σ_d = 0.30` nats — a deliberately conservative figure for paired NLL differences between two
arms of the same architecture at the same scale — and `n = 150`:

```
δ_min  =  sqrt(7.85 · 0.09 / 150)  =  0.069 nats
```

Against a baseline of ≈3.40 nats ([P2]'s 29.8 test PPL is ln 29.8 ≈ 3.39), that is a **minimum
detectable effect of ≈2.0%**.

That number is chosen against the one benchmark available: **[P2]'s own headline gain over GPT-2 is
2–4% perplexity.** A gate that could not resolve an effect the size of the base model's entire
published advantage would not be protecting anything.

600 across the window gives ≈1.0% MDE on the pooled `(40, 64]` contrast, which is the secondary
form of the result.

## 4. What this threshold does NOT establish, stated now

- **`σ_d = 0.30` is an assumption, not a measurement.** It cannot be measured before a model exists.
  → When the first paired run produces a real `σ_d`, recompute `δ_min` and **report it beside the
  headline**. If `σ_d` turns out to be 0.5, the true MDE at n=150 is 3.3%, not 2.0%, and the
  headline must say so. **The event threshold does not move retroactively** — that would be
  un-registering it. What moves is the honesty of the claim about power.
- **This gate counts events; it does not establish they are the right events.** Precision handles
  false positives. **Recall is reported but not gated** — a coref system that misses events shrinks
  `n` (which this gate already sees) rather than biasing the measure.
- **Passing says the corpus can host the measurement. It says nothing about the hypothesis.**

## 5. Outcomes, fixed in advance

| Outcome | Action |
|---|---|
| All three bins ≥ 150 adjusted, total ≥ 600 | **PASS.** E3 is measurable on this corpus. |
| Any bin short, but ≥ 80 adjusted in all three | **CONDITIONAL.** Either widen the window to `(32, 64]` — and **restate the claim to match**, because `(40, 64]` was chosen to exceed `M = 40` — or enlarge the PG-19 subset beyond 30M tokens and re-run. Not a quiet re-bucket after the fact. |
| Any bin < 80 adjusted | 🔴 **FAIL.** The primary metric has no events at the gaps the claim is about. E3 does not run on PG-19 as specified. §7.6's fallback applies: gaps `(16, 40]` at `M = 16`, **with the claim restated**, or a different corpus. |
| `p` unmeasured | **Exit 3 — did not run.** Not a pass, not a fail. |

## 6. Why this is a cheap gate and must stay one

One CPU-day against 576 GPU-hours. If `(40, 64]` is underpopulated, the headline curve is noise
dressed as a flat line and "RSR is flatter than FIFO" is unfalsifiable **in either direction** —
which is worse than a null, because a null is informative.

---

*Registered by the implementation lane before any E0i data existed. The threshold is a commitment,
not a target: it is not adjusted to whatever the histogram turns out to show.*

---

# ERRATA — 2026-09-17

> 🔴 **The threshold above is NOT being silently changed.** Re-registering a number after the fact is
> the exact failure this file exists to prevent. The error is stated, the corrected form is derived,
> and the number stands until the owner re-signs. Both versions remain in `git log`.

**Source:** session `d23e3dcd` handoff §8.1.

## The error: the attenuation exponent is wrong

§3 above defines the gate in **precision-adjusted** events as `n_adjusted = n_raw × p`.

**That is linear and it should be quadratic.** The correct relation is `n_raw × p² ≥ n_required`.

### Derivation

Let `p` be coref precision, so a fraction `(1 − p)` of counted "reintroduction events" are not
reintroductions. Those contribute no treatment effect, only noise. The measured effect is therefore
attenuated toward zero:

```
δ_observed  ≈  p · δ_true
```

Statistical power for a paired test depends on `δ² · n`, so the `n` required to detect `δ_true` at
fixed power is:

```
n_required(δ_observed)  =  7.85 · σ_d² / δ_observed²
                        =  7.85 · σ_d² / (p · δ_true)²
                        =  n_ideal / p²
```

⇒ **the usable event count is `n_raw × p²`, not `n_raw × p`.**

### What it costs

The linear form **understates the requirement**, and it does so worse as precision falls:

| coref precision `p` | raw events needed for 150 usable — linear (wrong) | quadratic (correct) |
|---|---|---|
| 0.95 | 158 | **166** |
| 0.90 | 167 | **185** |
| 0.80 | 188 | **234** |
| 0.70 | 214 | **306** |

At `p = 0.7` the original form asks for **30% fewer events than are actually needed** — and it asks
for least exactly where measurement error is worst. A gate that relaxes as its dependency degrades is
the wrong shape.

## Standing, until re-signed

- **The pre-registered threshold remains `≥ 150` per bin and `≥ 600` across `(40, 64]`.**
- **The adjustment formula is corrected to `n_raw × p²`.** This makes the gate *stricter*, never
  looser — a correction that only ever raises the bar cannot be a post-hoc convenience.
- `p` is still measured on 100 hand-annotated reintroductions. **Unmeasured `p` still returns
  exit 3 — did not run. Not a pass.** The correction makes that measurement *more* load-bearing:
  the gate now depends on `p²`.

## Two things the owner still decides

1. **Re-sign, or hold the linear form as registered.** The quadratic form is correct; adopting it
   after registration is still a change to a pre-registered rule and should be an explicit signature,
   not an edit absorbed by whoever noticed.
2. `d23e3dcd` proposes the gate as **`N_observed × p² ≥ 400`**. That is a different absolute target
   from the 150/bin derived in §3 above, which came from a stated `σ_d = 0.30` and a 2.0% MDE against
   [P2]'s own 2–4% headline effect. **Reconcile the two derivations before signing** — they disagree
   on the number, not on the exponent, and only one of them shows its σ assumption.

📌 `σ_d = 0.30` remains an **assumption**, flagged as one in §4 above. When the first paired run
produces a real `σ_d`, recompute the MDE and **report it beside the headline**. The event threshold
does not move retroactively.
