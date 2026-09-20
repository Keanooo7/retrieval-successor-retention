# Spec corrections

`docs/spec/rsr_model_spec_v0.5.md` is on its fifth revision and carries three
changelogs. Several passages were superseded by a correction and never rewritten.

**Everything in this file wins over the spec body.** Where the two disagree, this
file is authoritative and the spec body is stale text. Do not "resolve" a conflict
by re-reading the spec — read the entry here.

Corrections 1–6 were supplied with the Sprint 1 kickoff and are already resolved.
Corrections 7–14 were found by audit during Sprint 1 planning (2026-09-17) and are
reported here per the kickoff's instruction: *"If you find another passage the
changelog superseded, add it there and tell me — do not silently pick a reading."*

---

## 1 — `K = 64` on PG-19, `K = 40` on synthetic

**Supersedes:** §3.4 closing line, "`K = M`, `λ_shadow = 0.5`, both swept once in A7."

`K` is the shadow-buffer depth and must be **≥ the max target gap**. The
self-audit three paragraphs above that line, and the §4.5 table, both override it:
`K = 64` on PG-19, `K = 40` on synthetic.

At `M = 40` with E3's target window `(40, 64]`, a slot evicted shortly after write
needs ~64 steps of shadow coverage. `K = 40` censors exactly the events E3 exists
to measure.

## 2 — Monte-Carlo (`λ = 1`) is the default return; TD(0) is ablation A8

**Supersedes:** §3.3's composition `L = L_NTP + β · L_TD`.

Read that as `L_MC` on the default path. The no-backprop rule — only `φ` receives
gradient, never the transformer or `W_sent` — applies to **whichever retention
loss is active**.

## 3 — No EMA target copy of `φ` on the default path

**Supersedes:** §3.4's *"If the TD residual (§7.7) misbehaves after handover, add
an EMA target copy of `φ` before touching `β`."*

That sentence is live **only inside ablation A8**. Monte-Carlo has no bootstrap,
so there is no moving target to stabilize.

## 4 — §3.5 is headed "Three corrections" and lists four, numbered 1, 2, 4, 3

**Supersedes:** the heading, and any reading that drops one.

All four apply. Item "4" is the one that matters:

```
γ_b  ≈  b_max / (0.25 · E[lifetime])        order 0.05–0.1
```

**Derived in E0e. Never frozen at 0.001.** At `γ_b = 0.001` with a stream-bounded
lifetime, the maximum attainable |`b`| over a slot's entire life is 0.08 SD of a
z-scored `ψ̂` — the control loop cannot move the argmin, and is operationally
identical to `b ≡ 0`, which is also the §3.7 reduction condition.

## 5 — §0's "weeks 1–4 were approved unchanged; E0a–E2 are unaffected" is false under v0.5

**Supersedes:** §0, v0.2 → v0.4 changelog preamble.

- **E0h and E0i are new and both are pre-gate.**
- E0e now also produces `γ_b` and the EMA half-life, not only `τ`.
- E1 now includes a `ν` sweep.

## 6 — §10.1's "that cost is not in the §4.1 budget" is superseded by D-5

**Supersedes:** §10.1, precondition 1, closing clause.

D-5 funded the E7 small-`M` model: a second PG-19 model trained at `M = 8,
d = 128`, priced in §4.1 at 6 runs × ~12 h = ~72 GPU-h.

---

# Found by audit, 2026-09-17

## 7 — GPU procurement is a cancellable hold, not a reservation

> 🔴 **SUPERSEDED by [ADR-0007](decisions/ADR-0007-all-training-on-the-mac-studio.md), 2026-09-17: no GPU is rented and procurement is dropped. Everything runs on the Mac Studio. Kept for the record; do not act on it.**

**Supersedes:** §4.1 closing line (*"Name the provider and reserve in week 1"*,
*"Reserve 4, floor 3, continuously, from week 5"*) and §8's week-1 row
(*"Reserve 4 GPUs against §4.1, not just hours"*).

Week 1 gets a **named provider, a written quote, and a cancellable hold with a
movable start date**. Nothing billable. The reservation *is* the four-figure
commitment §16 declines to approve; a hold satisfies §16 condition 8 without
pre-empting the gate.

Start date is **week 6, not week 5** — the week-4 gate is planned to fall at the
end of week 5, because spec week 4 carries the E1/E2 gate *and* §15.1's
rederivation *and* §15.2, which is roughly 54 hours that cannot be delegated.

Floor is **4, not 3** — see correction 8.

## 8 — §4.1's 768 GPU-h for weeks 5–7 is an undercount

> 🔴 **SUPERSEDED by [ADR-0007](decisions/ADR-0007-all-training-on-the-mac-studio.md), 2026-09-17: no GPU is rented and procurement is dropped. Everything runs on the Mac Studio. Kept for the record; do not act on it.**

**Supersedes:** §4.1's parallel-capacity table and the "Reserve 4, floor 3"
conclusion drawn from it.

§4.1 computes weeks 5–7 as E3 (576) + A2/A4 (192) = 768 GPU-h against 504
wall-clock hours. But §8 also schedules into weeks 5–7:

| Item | GPU-h |
|---|---|
| E3, four arms × 3 seeds | 576 |
| A2, A4 ablations on PG-19 | 192 |
| E7 model at `M = 8, d = 128` | 72 |
| A1 | 2 |
| **Total** | **842** |

At 65% utilization with a 25% rerun margin that is **3.21 GPUs**, which breaks the
spec's own stated floor of 3. The 1.52 / 2.23 / 2.93 figures in §4.1's table are
stale. **Hold the floor at 4.**

A one-week slip leaves 336 hours for the same 842 and pushes the requirement to
4.82 GPUs, or drops E3 from three seeds to two. That is why correction 7 buys a
hold with a *movable* start date.

## 9 — The spec never states `A_max` for the synthetic corpus

**Supersedes:** nothing in the spec — it fills a gap. But it corrects a *misreading*
that the spec's own text invites, and the misreading is expensive.

§3.6(a) contains the string `A_max ∈ {16, 32, 64}`. **That is an illustration of a
failure mode at `S = 30`**, not a prescribed sweep set:

> "With `S = 30`, sweeping `A_max ∈ {16, 32, 64}` leaves two values inert and
> produces a flat curve that is an artifact, not a finding."

§4.5's `A_max` row says only *"Swept only where `S` makes it bind."* Read as a
prescription, the `{16, 32, 64}` set makes `γ = 0.97` look **illegal** on
synthetic: at `S = 48` only `{16, 32}` satisfy `A_max ≤ S`, so
`min(S, A_max) ≤ 32 < 33 = 1/(1−0.97)`.

**It is not illegal.** Checked against the actual constraint
`1/(1−γ) ≤ min(S, A_max)`:

| Configuration | `S` | `A_max` | `min` | Horizon at γ=0.97 | Legal? |
|---|---|---|---|---|---|
| Synthetic (E1, E2) | 48 | **`S` = 48, non-binding** | 48 | 33 | **yes** |
| PG-19 E3 headline | 80 | 64 (forced — §3.6 ties measurable gap `k` to `A_max`, and E3 targets `k` to 64) | 64 | 33 | **yes** |
| A2 ablation (`A_max = M`) | 80 | 40 | 40 | 33 | **yes** |

`γ = 0.97` is legal everywhere it is used. `γ = 0.99` stays dropped: horizon 100 >
`S` = 80.

**Resolution:** `A_max = S = 48` on synthetic. See `docs/decisions/ADR-0004`.
This is a reading of §3.6(a), not a change to it.

**This does not pre-empt E1.** `γ` is swept `{0.5, 0.9, 0.97}` on synthetic and
frozen at whichever wins on merit (§4.5). Correction 9 establishes only that the
constraint does not exclude 0.97.

**Pre-registered consequence, recorded before E1 runs.** E3's target window is
gaps `(40, 64]`. The weight the return gives an event at that distance:

```
γ = 0.90    0.90^40 = 0.015     0.90^64 = 0.0012     →  0.1–1.5% of the return
γ = 0.97    0.97^40 = 0.296     0.97^64 = 0.142      →  14–30% of the return
```

At `γ = 0.90` a slot needed 64 steps out contributes one tenth of one percent of
its own discounted return; the value function cannot express the claim E3 is built
to test. **If E1 freezes `γ` below 0.97, E3's `(40, 64]` window is known in
advance to be under-weighted in the return, and that must be stated in the E3
writeup as a limitation rather than discovered as a null.**

**Synthetic caveat to state in the config:** at `γ = 0.97` the horizon (33) is
shorter than the generator's max gap (40), so the longest synthetic gaps are
attenuated to ~0.30 weight. Attenuated, not invisible. Acceptable; state it.

## 10 — Activation memory is budgeted against the wrong card

> 🔴 **SUPERSEDED by [ADR-0007](decisions/ADR-0007-all-training-on-the-mac-studio.md), 2026-09-17: no GPU is rented and procurement is dropped. Everything runs on the Mac Studio. Kept for the record; do not act on it.**

**Supersedes:** §4.2's E0c budgeting protocol, and the 64 GB framing repeated in
§5.2 and §7.6.

> §4.2 fixes the maximum feasible `(S, d, batch)` triple from E0c "on 64 GB", and
> §5.2 and §7.6 repeat the 64 GB framing. 64 GB is the Mac Studio. But E3, A2/A4
> and the E7 model all run on rented commodity cards — §4.1 prices "A40/A6000
> rates", both of which are **48 GB**. The fit decision is therefore made against
> ~33% more memory than the hardware that has to run it.
>
> **Consequence.** E0c can pass at 64 GB and E3 can OOM in week 6, at which point
> §7.6's fallback ladder (cut to `d = 128`, then `A_max` to 40, target gaps
> `(16, 40]` at `M = 16`) fires in week 6 — the exact outcome §7.6 exists to
> prevent: "Decide in week 1 from E0c, not in week 6."
>
> **Resolution.** E0c runs on the card type T8 holds, not on the Mac. The binding
> budget is that card's VRAM, recorded in ADR-0002 alongside the provider and the
> exact SKU. §4.2's tie-break — "if `S = 80` does not fit, `S` wins and `d` is
> cut" — is evaluated against that number, not 64. Log per-trial free-memory
> alongside peak-resident so a pass is not an artifact of a warm cache.
>
> Status: new finding, not in the spec's changelogs.

## 11 — §2's falsifiers are misnumbered

**Supersedes:** nothing substantive. Recorded as a citation hazard.

The list runs **1, 2, 3, 3b, 3c, 5, 6, 4** — falsifier 4 ("No behavioural
signature", E7) appears last, after 6. §16 condition 4 cites "falsifier 4" by
number. Cite falsifiers by name as well as number in the writeup.

## 12 — [P8] is misdated

**Supersedes:** §1's source table, *"Rae et al. (2020), Compressive Transformer
(introduced PG-19)"*.

The paper is **arXiv:1911.05507, posted 2019-11-13** (ICLR 2020). Cite the 2019
arXiv posting or the ICLR 2020 proceedings explicitly; "Rae et al. (2020)" is
ambiguous and a reviewer will check. Feeds E0f.

## 13 — [P2] facts, verified against the source

**Supersedes:** §1's source table and the framing in §5.3 and §13.

1. **First author is Nasim Borazjanizadeh** (arXiv:2512.25026, v1 2025-12-31,
   v2 2026-01-12, with James McClelland; "authors contributed equally").
2. **[P2] trains on WikiText.** The string "PG-19" appears **zero** times in the
   paper. So §5.3's *"PG-19 — where the claim lives"* is RSR's own choice, not
   something inherited from TG.
3. Consequently §13's *"`S = 80` for E3 exceeds [P2]'s trained curriculum"*
   **understates the gap**: E3 is at 2.7× the stream depth *and* on a different
   corpus. TG's behaviour there is doubly uncharacterized. The FIFO baseline must
   be re-run at `S = 80` on PG-19 and may itself degrade.
4. [P2] does **not** cite H2O, Expire-Span, Scissorhands, StreamingLLM, Tensor
   Programs V, or Kobayashi et al. 2020. They are RSR's baselines, not TG's. Only
   the Compressive Transformer is shared. [P2]'s own comparators are GPT-2, a
   boundary-marker decoder, and gist-masking (Mu et al., arXiv:2304.08467).

## 14 — The kickoff's Task-0 premise fails, usefully

**Supersedes:** the kickoff's *"reproduce its reported numbers (29.8 test PPL, 21
sentence-steps/sec on one A40 at `S ≈ 30`) as a smoke test before trusting
anything built on it."*

TG's reference implementation **is** public — `jlmcc94303/ThoughtGestaltCode`,
Apache-2.0 — but its README states:

> "This version differs slightly from the version of the model described in
> ([arXiv:2512.25026]). It achieves comparable results to those reported in the
> paper when trained with 12M text tokens."

So the published numbers are **not reproducible from the released code by the
authors' own statement**, and a smoke test against them would be testing a
difference nobody has characterized.

**Replaced by** the golden-tensor fidelity harness: `tests/test_fidelity.py`
against fixtures extracted once from the pinned JAX reference, to a tolerance
committed in ADR-0002 *before* the fixtures are generated. See ADR-0001 and
ADR-0002.

Two further facts about the release, recorded because they will cost time
otherwise:

- `EXPERIMENTS.md` states the paper's gist baseline *"indexes the gist flag on the
  query axis instead of the key axis and so grants no cross-sentence access at
  all."* That is an admission of a bug in a published baseline.
- `tg/Rough/preprocess_corpus.py` imports `src_recurrent.pipelines.data.io`,
  `src_recurrent.core.tokenizer_setup` and `src_recurrent.core.data.sentence_splitter`.
  **`src_recurrent` does not exist anywhere in the repo tree.** The corpus
  preprocessing path is not runnable as published; RSR writes its own.

## 15 — Gestalts are L2-normalized, which breaks a premise of section 4.3's muP derivation

**Supersedes:** section 3.1's `s_t = W_sent . H^(7)[EOS]`, and the *premise* (not
necessarily the conclusion) of section 4.3's bilinear muP argument.

**Status: FLAGGED, NOT RESOLVED.** The resolution is a change to a muP
prescription that section 15.3 identifies as the author's own derivation. It is
Brendan's to make. E0a measures it.

### What the reference actually computes

`tg/models/tg_srep_head.py`, pinned commit `f220b109`:

```
s = normalize( Linear( Dropout_0.15( LayerNorm( h[EOS] ) ) ) ) * srep_norm_target
```

with `srep_norm_target = 1.0`, plus a **squared hinge penalty** on the
*pre-normalization* norm (`srep_norm_margin = 0.1`, `srep_norm_reg_weight = 0.01`)
that holds the raw norm in `[0.9, 1.1]`.

So `||s_i||_2 = 1.0` exactly, for every slot, at every step.

Section 3.1's one-liner is a simplification. `W_sent` is the `proj` Dense, but it
arrives wrapped in a LayerNorm, a dropout and an L2 normalization, and it carries a
bias.

*(Resolved while checking this: section 5.1's "gestalt layer 7" and the reference's
`srep_extraction_layer = 6` **agree**. `tg_model.py` uses
`for i, block in enumerate(self.blocks)`, so the index is 0-based and block 6 is
the 7th. No discrepancy.)*

### Why it is load-bearing

Section 4.3 derives the bilinear head's `1/d` multiplier like this:

> Read `psi_hat = s_i^T W c_t` as two stages: `h = W c_t` is a hidden matrix
> producing Theta(1) coordinates, and `s_i^T h` is **a dot product of two
> `d`-dimensional Theta(1) vectors**. That dot product is Theta(sqrt(d)) at init
> and Theta(d) after training.

**A unit-norm `d`-vector does not have Theta(1) coordinates. It has coordinates of
order `1/sqrt(d)`.** Both `s_i` and `c_t` are gestalts, so both are unit-norm.
Redoing the count with `W` at `Var = 1/d`:

| Quantity | Order |
|---|---|
| `s_i`, `c_t` coordinates | `1/sqrt(d)` |
| `h = W c_t` coordinates | `1/sqrt(d)` |
| `s_i^T h`, uncorrelated (init) | `1/sqrt(d)` |
| `s_i^T h`, **correlated (trained)** | **`Theta(1)`** |
| output after the prescribed `1/d` multiplier | **`Theta(1/d)`** |

Under the prescription as written, `psi_hat`'s output **decays with width in the
regime muP governs**, rather than being width-invariant. That is exactly the
failure section 4.3 says "surfaces in week 9 with no error message", arriving
through the input normalization rather than through the head's shape.

It is also a precise instance of the warning section 4.3 attaches to its own
derivation:

> the wrong argument gives the wrong multiplier for a head of slightly different
> shape, which is exactly how this failure mode propagates.

The arithmetic above is a count of orders, and it carries its own assumptions
(uncorrelated init, `Var = 1/d` on `W`, `c_t` a raw gestalt rather than a running
context vector). **It is not a substitute for the measurement.** Section 12.4:
measure, don't extrapolate.

### What happens now

1. **E0a is the decision.** It already checks `psi_hat`'s scalar output coordinate
   scale specifically. It now does so across candidate multipliers, and it must
   read the scale **after real optimizer steps**, not at initialization -- see
   `src/rsr/mup/coord_check.py` for why the init-time reading is misleading in its
   own right.
2. **Section 3.2.2's `c_t` must be pinned before E0a runs.** Section 3.2.2 says
   "the current sentence gestalt (**or a running context vector**)". Those two have
   different norms, and therefore different multipliers. Pick one and write it down.
3. **Section 3.4's redundancy term is unaffected and slightly simplified.**
   `cos(s_i, s_j)` on unit vectors is a plain dot product.
4. **Do not "fix" this by removing the L2 normalization from the transcription.**
   The normalization is TG's, the hinge penalty is in TG's loss, and section 3.1
   requires the base model be unmodified. Changing it would make E0b's reduction
   test a comparison against something that is not TG.

## 16 — §3.7's `T_warm = ∞` makes E0b vacuous; the reduction uses `T_warm = 0`

**Supersedes:** §3.7's *"warmup irrelevant (`T_warm = ∞`)"*.

Eviction dispatch is `t < T_warm → FIFO`. At `∞` that branch is taken forever, so
**under the §3.7 reduction no eviction ever reaches the score** and E0b certifies
FIFO against FIFO — the gate that exists to prove RSR reduces to TG passes without
executing the thing being reduced.

The spec's word "irrelevant" is right about the *outcome* and wrong about the
*path*: with `ψ̂ ≡ −a_i` the score already selects the oldest slot, so the warmup
changes nothing — which is exactly why the reduction should run through the score.

**`T_warm = 0.0` in `reduction_to_tg()`.** Gauntlet 0.1. Verified: 100 evictions,
all attributed to `neg_age`, victim sequence identical to `FIFOPolicy`, and
switching `psi_override` to the learned head breaks it.

## 17 — §3.2.1's `r_i` omits TG's memory gate

**Supersedes:** `r_i(t) = Σ_{l,h} ‖ α_{l,h,i} · W_O^{(l,h)} v_{l,h,i} ‖₂`.

```
r_i(t) = Σ_{l,h} ‖ g_mem^(l) · α_{l,h,i} · W_O^(l,h) v_{l,h,i} ‖₂
```

[P2] puts a learnable scalar `g_mem` on each cross-attention layer, scaling the
increment **before** the residual add. D-6's argument for keeping `W_O` —
*"precisely where head-specific rescaling lives"* — applies verbatim to `g_mem`,
which is where **layer**-specific rescaling lives.

App. C measures the gates **growing over training and larger in deeper layers**, so
the weighting is **non-stationary**: `r_i` at epoch 1 and at epoch 12 are not the
same measurement. **Compute both ways (gated and raw) and report both against LOO
Δloss in E0d.** §3.2.1's truth rule is unchanged: if they disagree, **LOO is
truth.** D-E; brief finding B-1.

## 18 — the per-layer `r_i` profile has six entries, not twelve

**Clarifies:** §3.2.1's *"report the per-layer profile once before collapsing to a
scalar."*

TG alternates self/cross blocks `S,C,S,C,…` over 12 layers, so cross-attention
lives at `ℓ ∈ {2,4,6,8,10,12}` — **six layers.** A twelve-entry profile is six real
rows and six zeros, and the zeros would be read as a depth finding. D-E.

## 19 — `c_t` is the current sentence gestalt; the running-context variant is an ablation

**Resolves:** §3.2.2's *"where `c_t` is the current sentence gestalt (or a running
context vector)"* — a parenthesis that reads as a second specification.

**Decided: the current sentence gestalt** (D-C). Three reasons: it is the simplest
reading of §3.2.2; it makes **both arguments to the bilinear form unit-norm** (with
correction 15, the reference L2-normalizes every gestalt), so the μP analysis is
single-valued instead of forked; and it keeps `ψ̂` positionally blind, which
ADR-0006 arrives at independently.

Written into the config as a named enum with one value implemented and the other
raising. The running-context version is a clean ablation, not a branch in the
critical path.

## 20 — `r_i` is collected in eval mode

**Adds to:** §3.2.1, which does not say.

`attn_dropout: float = 0.2` in the reference `tg_config.py`, so during training `α`
is stochastically zeroed and a slot can score zero demand because a mask fell on
it — noise in a *policy-relevant* direction. **Eval mode for the retention target;
train mode for the LM loss**, as an explicit mode switch in the code rather than an
ambient default. D-F. Enforced by `AttentionTrace.eval_mode` being a required
field, and by `rsr.retention.reward` refusing a train-mode trace.

## 21 — `T_warm` is a float number of steps, everywhere

**Supersedes:** the registry's `"one_epoch"` string.

§3.4 states the warmup as one epoch and requires it be *reported* as a fraction of
total epochs; the dispatch compares it against `t`. Those are two representations
with nothing converting between them, which is how the registry came to hold a
string while `RSRConfig` held a float.

**One representation: a float number of steps.** `T_warm` is CONDITIONAL on
`steps_per_epoch` and returns steps; `T_warm_epochs` (FROZEN, 1.0) stays for the
report. The string is deleted. D-I.

## 22 — No GPU is rented; everything runs on the Mac Studio

**Supersedes:** corrections 7, 8 and 10, §4.1's procurement lines, and §4.2's
"budget against the rental card" framing. See
[ADR-0007](decisions/ADR-0007-all-training-on-the-mac-studio.md).

The rental entered the plan for the golden-tensor extraction, the capacity budget,
and weeks 5–7 parallel capacity. The first turned out not to need a GPU at all
(`jax-metal` is the Metal *GPU* backend; `jaxlib` ships arm64 **CPU** wheels, and
CPU is the right target for byte-reproducible fixtures anyway). The second
dissolves once the sizing machine and the running machine are the same. The third
prices work **outside the approved scope**, which is what §16 declined.

**The replacement measurement:** the `(S, d, batch)` ceiling **on this machine**, at
the widths the approved scope actually runs. §4.2's rule is unchanged — *if `S = 80`
does not fit, `S` wins and `d` is cut*.

§13 gains an entry: every throughput and capacity number in this project was
measured on a single M4 Max, and nothing here supports a claim about other hardware.

## 23 — §4.1's compute budget assumed ≤7 sent/sec; measured is 171–392

**Supersedes:** §4.1's *"assume **≤ 7 sent./sec** pending E0c... ≈48 h per run"*, the
per-line-item hour estimates, and the 1,070 GPU-h total.

§4.1 reasoned from [P2]'s 21 sent/sec on an A40 at `S ≈ 30`, divided by ~3 for a
backward chain 2.7× deeper at `S = 80`. **E0c measured it instead** (§12.4: measure,
don't extrapolate), on this machine, with the memory path live and both policies:

| | measured | §4.1 assumed |
|---|---|---|
| RSR, `d=128, S=80, batch=16` | **310 sent/s** | ≤7 |
| RSR, `d=384, S=80, batch=8` | 171 sent/s | ≤7 |
| Hours per 1.2M-step run | **1.1 h** | 48 h |

The assumption is off by **~44×** in our favour, and the arithmetic behind it was
sound — it was anchored to a *rented A40 at a different width*. The A40 figure was
never measured here and should not have been scaled; the error is the anchoring, not
the division.

**Consequence for the budget.** §4.1's 912 GPU-h for E3 + A2/A4 + E4 + E5 becomes
roughly **21 hours of Mac time** at the recommended config. Combined with
[ADR-0007](decisions/ADR-0007-all-training-on-the-mac-studio.md) — no outsourced
compute — the whole §4.1 budget section, its parallel-capacity table and its cut
order are **superseded**: there is nothing to price, nothing to schedule across
GPUs, and no four-figure spend to stage.

⚠️ **What the measurement does not cover**, and what would move it:

* **random tokens, not real text.** Real batching uses uniform token-budget
  bucketing (20,000 supervised tokens/step in the reference), so sentence lengths
  vary and the effective batch does too. These are fixed-shape, fully-packed
  sentences — the easy case.
* **no optimizer step** in the timing. AdamW over 22M parameters adds two state
  tensors and a step.
* **one data shape per row**, and MPS only.

So this is a measurement of the model's throughput, not of a training loop's. The
cut order in §4.1 stays on the page as a contingency; it is simply not binding at
these numbers.

## 24 — S-7: sentence length is an uncontrolled confound in E7

> **Numbering note.** Brendan supplied this as "correction 16" and as **S-7** in his
> own scheme. **16 is already taken** — it is the `T_warm = ∞` / E0b-vacuity entry
> that gauntlet 0.1 rests on — and corrections 16–23 are all cited by number
> elsewhere. Filed as **24** to avoid both overwriting a live correction and a
> renumber that would break existing citations. Cite it as **correction 24 (S-7)**.

**Adds to:** §10.1's E7 correlation protocol. **Supersedes** nothing; it is an
analysis requirement and a §13 limitation, not a change to any prescription.

🔴 **Do not change the μP multiplier, `srep_norm_target`, or any frozen constant on
the strength of this.** It is a regressor in one analysis.

### The confound

TG normalizes every gestalt to `srep_norm_target = 1.0` (correction 15), so **every
sentence enters memory at identical strength regardless of how much was compressed
into it.** A 4-word sentence and a 43-word sentence produce equally loud entries.

Humans do not work that way: encoding quality falls with sentence length. So the two
systems being correlated in E7 forget for different reasons:

| | how it encodes | why it "forgets" |
|---|---|---|
| human | long sentences **poorly** | partly **encoding** |
| TG / RSR | all lengths **equally** | **retention** only |

§10.1's correlation therefore assumes both systems encoded comparably, and they do
not. **A correlation between what RSR drops and what people fail to recall can be
produced by sentence length alone, with no retention policy involved.** That is a
confound in the **primary claim**, not in a secondary measure.

The source for the human side is American Press Institute readability research
across 410 newspapers — reported shape: near-complete comprehension around 8 words
per sentence, falling below ~10% by ~43 words. ⚠️ **Correlational, not a controlled
experiment. Use the shape; do not cite the percentages as a constant** and do not
put either number in `constants.py`.

### The regime is not hypothetical

[P2]'s own corpus averages **~25 words/sentence** — the paper states this directly,
deriving ~25 × 30 ≈ 750 tokens per stream. That sits in the 50–60% comprehension
band, i.e. squarely inside the range where the human curve is steep.

### Fix

**Partial sentence length out of the E7 correlation, exactly as §10.1 already
partials out serial position.** One extra regressor.

- If the correlation **survives**, it is *stronger* than it would otherwise have
  been — the confound was carrying none of it.
- If it **does not**, RSR is tracking **sentence length rather than structural
  importance**. That is a vacuity failure **no falsifier currently names**, and the
  age-based §7.1 gate ("RSR must beat LRU") **structurally cannot see it**: length
  and age are independent, so a length-tracking policy beats LRU comfortably while
  having rediscovered nothing.

### Invisible in current work — do not go looking for it tonight

Both measured, not assumed:

- The synthetic corpus is **mean 4.14 words/sentence, max 6, min 2, zero sentences
  over 8** — re-measured 2026-09-18 at `SyntheticConfig(sentences_per_document=48,
  seed=0)`, **n = 3072** (64 documents), full distribution
  `{2: 91, 3: 244, 4: 1914, 5: 784, 6: 39}`. The entire corpus sits at the flat top
  of the human curve, so the confound has **no variance to act through**.

  *(The brief reported n = 1536 for the same shape; that is a different
  `sentences_per_document`, and the mean/max/ceiling claims reproduce either way.
  §12.4: the config is part of the measurement, so the config is recorded with it.)*
- `max_sentence_tokens = 64` **truncates rather than degrades** — a cliff, where the
  human curve is a gradient. Truncation is not a model of poor encoding.

So this cannot be studied on the synthetic corpus and does not bite until E7 has
real stimuli (ADR-0005).

### For the authors, not for the code

Normalizing magnitude away is a **departure from the Sentence Gestalt lineage**
(St. John & McClelland 1990) that TG names as its own ancestor, where magnitude
carried strength. Recorded as a question for the authors. **Not a code change.**

### Where this lands

- **§13 (limitations):** state that every E7 correlation is conditional on sentence
  length having been partialled out, and that TG's unit-norm gestalts make the two
  systems' forgetting mechanisms non-comparable without it.
- **`docs/release-conditions.md`:** condition 4's E7 line.

---

## 25 — §10.1's "largely independently of recency" is false, and E7's design depends on it

**Source:** Kintsch & van Dijk 1978, p. 379 and footnote 6, read in full 2026-09-20 (E0f pass 2).

Spec line 641 says the levels effect holds *"largely independently of recency."* The leading-edge
strategy **"emphasizes recency and frequency"** — recency is one of its two selection criteria.
Footnote 6 reports the comparison on the immediate-summary data: against leading-edge, a
**recency-only** strategy raises the minimum chi-square by **43%**, and a **levels-plus-primacy**
strategy by **23%** (both highly significant); random is rejected at **χ²(34) = 113.77**.

### Why this is not cosmetic

§10.1 requires E7 to report a partial correlation **controlling for serial position**, on the
grounds that *"the levels effect is not recency, and neither should the result be."* But the theory
being tested **predicts a recency component**. A partial correlation that zeroes recency out is
testing a strawman of the leading-edge strategy, and RSR could reproduce 1978 exactly and still
score near zero on it.

### Fix — E7 reports three numbers, not two

1. Raw rank correlation between survival time and normed human recall.
2. Partial, controlling serial position — **kept**, because it separates RSR from FIFO, whose
   survival time is the deterministic `min(M, S−i)`.
3. 🔑 **New and primary: the agreement between RSR's survival ordering and the leading-edge
   strategy's own ordering** on the same passages. That is the actual comparator, and it is the one
   the claim in §2 is about.

Reading (3) as the headline also makes the published ranking usable as a yardstick: leading-edge >
levels+primacy > recency-only > random. RSR's position in that ordering is the result.

---

## 26 — the leading-edge strategy runs on the MICROstructure, not the macrostructure

**Source:** same, p. 379.

Spec lines 517 and 686 say the rule keeps propositions *"highest in the macrostructure."* It does
not. It walks the **microstructure coherence graph**, built from argument overlap, cycle by cycle.
KvD disclaim the other reading in the same paragraph:

> *"Note that the procedure is strictly formal: There is no claim that topmost propositions are
> always most important or relevant in a more intuitive sense. That will be taken care of with the
> macro-operations described below."*

The macro-operators — deletion, generalization, construction, under schema control — are a
**separate mechanism** applied later.

🔴 **This is an implementation bug waiting to happen.** §11 makes the leading-edge strategy an
*implemented baseline* and calls it "an afternoon." A baseline that builds a macrostructure is not
the leading-edge strategy and will not reproduce the 1978 fit. `src/rsr/baselines/leading_edge.py`
is still a stub; the brief that implements it must specify the coherence graph.

**Two more parameters the implementation needs, both published:** the buffer is `s`, and KvD report
that `s = 4` fits, `s = 1–3` fit nearly as well, and **`s = 10` fails outright** — *"the minimum
chi-square increases drastically, and the model no longer can fit the data."* Recall probability is
**`1 − (1 − p)^k`** for `k` cycles survived, so **survival time is the model's own recall
predictor** and E7's dependent measure is already the right one.

📌 The fitted human range `s ≈ 1–4` is a better argument for evaluating E7 at `M = 8` than §10.1's
current one from eviction pressure. Use it in §5.1; it also shrinks precondition 1's problem.

Also minor, same source: the reinstatement search triggers on **absent argument overlap between the
input set and the buffer** — a cycle-level coherence failure — not on "a needed proposition has been
dropped" (spec line 686).

---

## 27 — attribution: the leading-edge strategy is Kintsch & Vipond (1978)

KvD 1978, p. 379: *"originally proposed by **Kintsch and Vipond (1978)**."* The spec calls it
"Kintsch & van Dijk's leading-edge strategy" throughout. KvD 1978 **implements and validates** it;
it does not propose it. Cite both. ⚠️ Kintsch & Vipond has been seen only as an in-text citation and
is itself unverified.

**Related, and the reason this matters more than a footnote:** [P11] bundles Kintsch & van Dijk with
**Thorndyke (1977)**, and they are different hierarchies — an argument-overlap coherence graph over
propositions versus a **story grammar** (setting / theme / plot / resolution). They produce different
rankings. E0g's stimulus selection must pick one, say which, and report the correlation between them.

🔴 **And TG does not cite Kintsch & van Dijk at all.** Checked against arXiv:2512.25026v2: TG's
discourse-memory citations are the situation-model tradition (Radvansky & Zacks, Jarvella, Zwaan)
plus the Sentence Gestalt model. **The spec may not describe the KvD connection as inherited from
TG.** Making it is a contribution; inheriting it is a false claim.

---

## 28 — §15.1 cites [P4] for an argument that is in [P2]

**For the author. Not a code change, and deliberately not edited in place.**

§15.1 asks whether the author can derive, *"from **[P4]** and from what `detach` does to the
gradient,"* why the CLS mapping in §3.6 is inverted. [P4] is Tensor Programs V — μP, zero-shot
hyperparameter transfer across width. It says nothing about gradient flow through a detached memory
write. That argument is in **[P2] Appendix A**.

Recorded here rather than fixed in the spec because §15 states it *"must not be filled by a
reviewer, an advisor, or a model,"* and swapping a citation inside it is still editing it. The
author should make the change.

---

## 29 — §10.2 omits the boundary conditions on its own biological evidence

**Source:** Dunsmoor, Murty, Davachi & Phelps (2015), *Nature* 520:345–348, abstract read
2026-09-20; Braun, Wimmer & Shohamy (2018), *Nat Commun* 9:4886.

§10.2 cites [P12] as a *"biological existence proof"* for retroactive prioritization. The gloss is
accurate; the abstract's next clause is not in the spec:

> *"Retroactive enhancements as a result of emotional learning were observed **following a period of
> consolidation**, but were **not observed in an immediate memory test** or for items strongly
> encoded before fear conditioning."*

🔴 RSR recomputes `ψ̂(s_i, c_t)` at **every step** — the immediate condition, which is exactly where
the human effect is **absent**. Braun et al. has the same delay dependence (24 hours).

**Fix — §13, not §10.2.** The claim survives in its computational form: retention value is
*revisable by later context*, and no accumulated-past-attention method has that property. The
**timescale does not transfer**, and §13 must say so rather than leaving a reviewer to find it.
Braun additionally reports prioritization **graded by distance from reward**, which is an
SR-discount-shaped result §10.2 could use and currently does not.

Same section, [P13]: the synaptic tag **decays in under three hours** and capture is a competition
for a **limited diffusible protein pool** triggered at another input. "The gestalt is the tag,
re-scoring is the capture" is an analogy about revisability. Label it as one.

---

## 30 — §11 is missing the nearest prior work

**Immertreu, Schilling, Kinfe & Krauss (2026)**, *Word Class Representations Spontaneously Emerge
from Successor Representations Trained on Natural Language*, arXiv:2605.24585. A network trained on
WikiText-103 to predict future word distributions across multiple horizons — rather than the next
token — develops word-class structure *"separable and recoverable through unsupervised clustering,"*
with shorter horizons more syntactic and longer horizons more semantic.

**This is the closest published work to RSR's premise that a successor representation is the right
object over discourse**, and §11 does not have it. Add it.

Two notes attached to the same search:

- 🔴 **"Successor heads"** (Gould, Ong, Ogden & Conmy, ICLR 2024) are attention heads that
  increment ordinal tokens. The paper does not mention successor representations or Dayan. **Do not
  let the name collision into §11.**
- 📌 **No paper applying an SR specifically to transformer KV-cache or working-memory eviction was
  found**, across roughly six query formulations. That is a real novelty claim — state it as "to our
  knowledge" and re-check before submission.

For general SR claims, cite **Carvalho, Tomov, de Cothi, Barry & Gershman (2024), Neural Computation
36(11)** alongside Dayan; it is the current review of record.
