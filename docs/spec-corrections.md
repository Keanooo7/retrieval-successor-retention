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
