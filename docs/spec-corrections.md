# Spec corrections — these override the spec body

**Read this before `docs/spec/rsr_model_spec_v0.5.md`, and before writing any code.**

The spec is on its fifth revision and carries three changelogs. Several passages were superseded by
a correction and never rewritten. **Where a correction and the body disagree, the correction wins.**
Do not resolve any of these yourself — they are resolved here.

Two groups:

- **C-1 … C-6** — supplied by the owner in the Sprint 1 kickoff prompt. Internal to the spec: a
  changelog entry that corrected a passage the body still carries.
- **B-1 … B-5, S-1 … S-6** — found 2026-09-17 by reading the primary sources ([P2], [P5], [P6],
  [P7]) against the spec. These are **not** internal inconsistencies; they are places the spec
  states something about a cited work that the cited work does not support, or omits something it
  does. Evidence and quotes: `docs/citation-audit.md`.

Nothing here edits the spec or the kickoff prompt. Both stand as the record of what was true on
their date.

---

## C — corrections internal to the spec

### C-1 · `K = 64` on PG-19, `K = 40` on synthetic. Not `M`.
§3.4's **closing line** still reads *"`K = M`, `λ_shadow = 0.5`, both swept once in A7."* The
self-audit three paragraphs above it and the §4.5 table both override it. `K` is shadow-buffer depth
and must be **≥ the max target gap**; `K = M = 40` censors exactly the `(40, 64]` events E3 exists to
measure.
📌 This is the single line most likely to be copied into an implementation.

### C-2 · Monte-Carlo (`λ = 1`) is the default return. TD(0) is ablation A8.
§3.3 still composes `L = L_NTP + β·L_TD` and forbids backpropagating `L_TD`. Read the composed term
as **`L_MC`** on the default path. The no-backprop rule applies to whichever retention loss is active.

### C-3 · No EMA target copy of `φ` on the default path.
§3.4's *"if the TD residual misbehaves after handover, add an EMA target copy"* is live **only inside
A8**. MC has no bootstrap, no moving target, and nothing to stabilise.

### C-4 · §3.5 is headed "Three corrections" and lists four, numbered 1, 2, 4, 3.
All four apply. Item "4" is the one that matters: `γ_b ≈ b_max / (0.25 · E[lifetime])`, order
**0.05–0.1**, **derived in E0e, never frozen at 0.001**. At 0.001 the mechanism is inert — 0.08 SD
over a full stream.

### C-5 · §0's "weeks 1–4 were approved unchanged" is false under v0.5.
E0h and E0i are new and both are pre-gate. E0e now also produces `γ_b` and the EMA half-life. E1 now
includes a `ν` sweep. A reader trusting that line will not re-read week 1's scope.

### C-6 · §10.1's "that cost is not in the §4.1 budget" is superseded by D-5.
D-5 funded the E7 small-`M` model; §4.1 now carries the line.

---

## B — blocking. Touches code, a frozen constant, or the budget.

### B-1 · `r_i` must be computed both with and without TG's memory gate.
**Spec:** §3.2.1 defines `r_i(t) = Σ_{l,h} ‖α_{l,h,i} · W_O^{(l,h)} v_{l,h,i}‖₂`.

**Problem:** [P2] places a **scalar learnable memory gate `g_mem`** on every cross-attention layer,
which *"scales the cross-attention increment before it is added back via the residual path."* [P2]
App. C measures that these gates **grow over training** and are **larger in deeper layers**. `r_i`
measures upstream of the gate, so its layer sum mis-weights layers by a factor that is both
depth-stratified and moving during training. E0d is a kill gate.

Independently, [P5]'s own authors published the correction (Kobayashi et al. 2021, EMNLP,
arXiv:2109.07152): extending the norm analysis through the residual and LayerNorm changes the 2020
method's conclusions.

**Do:** compute `r_i` **both ways** — raw, and gated by `g_mem^(l)` — and report both against LOO
Δloss in E0d. The spec already requires the per-layer profile before collapsing to a scalar; `g_mem`
is what to weight it by. If the two disagree, **LOO is still truth** (§3.2.1), and the gated variant
is the better candidate, not the automatic winner.

**Affects:** `src/rsr/metrics/reward.py`, E0d, and the shadow buffer (which reuses the same scorer).

### B-2 · The GPU budget descends from an off-configuration anchor. Re-derive it in E0c.
**Spec:** §4.1 scales the whole budget from *"[P2]: 21 sent./sec on one A40 at `S ≈ 30`"*, then
assumes ≤7 sent./sec at `S = 80`, giving ≈48 h/run and a 1,070 GPU-h total.

**Problem:** that measurement is at **`d_model = 768`, 85.6M non-embedding parameters**. RSR trains
at `d ≤ 384` — **≤21.3M, roughly 4× narrower**. Every hour and every dollar inherits an anchor
measured on a model the project never trains. The direction is probably favourable; the magnitude is
unknown, and "probably favourable" is not a budget.

**Do:** E0c reports sent./sec **at the widths that will actually run** (`d ∈ {128, 256}` × `S ∈ {30,
50, 80}`, `M = 40`), replicating [P2]'s uniform token-budget bucketing (App. B.1) or declaring that
it did not. Re-derive the 1,070 total from that measurement. Do not scale from 21.

**Affects:** E0c, `docs/release-conditions.md` #8, T8's quote, and every dollar figure downstream.

### B-3 · The baselines get a tuning budget, or falsifier 6 is not a referendum.
**Spec:** §5.4 calls Expire-Span *"the referendum, not a comparison"* and §6 makes
`Expire-Span ≥ RSR` a **stop condition**. It is run once, untuned, at the week-4 gate — against
RSR's `γ`×`β` grid (9 configs) plus a `ν` sweep.

**Problem:** Expire-Span has a **ramp length `R`** and a **loss coefficient `α`**, and *requires*
structured dropout, because without regularising memory size during training it overfits easily.
H2O's split ratio is likewise a budget-allocation choice. Google's Deep Learning Tuning Playbook:
nuisance hyperparameters must be **optimized over**, not fixed, to compare scientific ones fairly —
*"the more a given nuisance hyperparameter interacts with the scientific hyperparameters, the more
damaging it is to fix its value."* Dodge et al. (EMNLP 2019) show published comparisons that reverse
under a different search budget.

An untuned Expire-Span losing is uninformative. A tuned one winning ends the project. The asymmetry
runs in the direction that protects the project.

**Do:** run E1 under an **equal-budget protocol**. Every arm with tunable hyperparameters gets the
same number of search trials, drawn the same way (quasi-random), and **the per-arm trial count is
reported in the paper**. Expire-Span gets `R` and `α` swept and structured dropout enabled. If the
budget will not stretch, say so explicitly and state falsifier 6's verdict as *provisional under an
unequal budget* — never as a clean pass.

**Affects:** E1's design, `configs/experiment/e1.yaml`, §5.4, §13.

### B-4 · §5.1's parameter range is wrong. The set is ≈2.4M–21.3M.
**Spec:** pairs `d_model ∈ {128, 192, 256, 384}` with *"≈0.3M–21M non-embedding params."*

**Problem:** 0.34M is [P2]'s `d = 48`, not `d = 128`. [P2]'s sweep is `d ∈ {48, 96, 192, 384}` →
0.34M–21.3M. At 12 layers the spec's own set spans **≈2.4M – 21.3M**.

**Do:** correct the table. And note the consequence for **E5 (S-4)**: [P2] has already published the
width sweep, over an **8× range** against E5's 3×, with fitted exponents and the conclusion
"consistent with an intercept shift" — which is E5's own stated conclusion. Either extend down to
`d ∈ {48, 96}` (cheap runs) to buy a real lever, or cut E5 and cite [P2] Fig 2(b).

### B-5 · `A_max` on synthetic must be set before any `γ` sweep.
**Spec:** §4.5 constrains `1/(1−γ) ≤ min(S, A_max)` and sweeps `γ ∈ {0.5, 0.9, 0.97}` **on synthetic
only**. §5.2 sets synthetic `S = 48`; §5.1 sets synthetic `M = 16`. `A_max` on synthetic is never set.

**Problem:** if `A_max` defaults to `M = 16` (as §3.7's reduction does), then `γ = 0.97` has horizon
33 > min(48, 16) = 16 — **the top value of the sweep is illegal on the only corpus it is swept on.**
`γ = 0.9` (horizon 10) is legal.

📌 The *plan artifact* raises this question and attaches it to the wrong instance: it claims A2's
PG-19 sweep is over `A_max ∈ {16, 32}`, but §6 defines A2 as `A_max = M` (= 40 on corpora), where
horizon 33 ≤ min(80, 40) is **legal**. The conflict is real; it lives on synthetic, not in A2.

**Do:** set synthetic `A_max` explicitly — `A_max ≥ 33` to keep `γ = 0.97` legal, or drop `γ = 0.97`
from the sweep and say so. **Decide before E1 runs**, and record it here. The constants registry
treats synthetic `A_max` as unset until then.

---

## S — scientific. Does not block code; does block the writeup.

### S-1 · §11 must engage H2O's submodular formulation.
H2O *"formulate[s] the KV cache eviction as a dynamic submodular problem and prove[s] (under mild
assumptions) a theoretical guarantee"* — Lemma 3.1 (greedy is near-optimal), Theorem 4.4 (with cache
limitation), proofs in App. D. §3.4's D-3 argues *against* greedy independent scoring on
submodularity grounds without engaging it; §11's H2O entry reads only *"evicts by accumulated
attention."* This is S-1's failure class repeated on a second paper. Falsifier 5's *"regardless of
how good ψ̂ is"* is stated more strongly than the published theory supports.

### S-2 · Report age partial-ρ for every arm, not just RSR's.
H2O's App. B.2 documents that accumulated attention is biased toward the **least recent** tokens,
and that replacing it with an averaged score **degraded performance** for them. §3.5's EMA-rate fix
therefore has a published counter-datapoint to address. And because H2O's bias is *anti*-recency
while FIFO/LRU are pro-recency, E2's one-sided vacuity gate is uninterpretable without the other
arms' numbers. Costs nothing.

### S-3 · E3's "no curriculum" is [P2]'s worst measured ablation, and is not required.
[P2]: *"removing the stream curriculum produces the largest drop (PPL 30.5)"* against a 29.8
baseline. And `30 → +12/5ep` reaches 78 then 90, so a curriculum **terminating at ≥80** supplies the
same `(40,64]` gaps in late epochs while keeping the measured stability benefit. Either carry a
curriculum arm or state why not. §13 already says the FIFO baseline may degrade at `S = 80`; this
quantifies the nearest measurement.

### S-4 · E5 partially re-runs a published experiment. See B-4.

### S-5 · §3.3's stop-gradient keeps one of its two reasons.
*"[P2] §4 documents that auxiliary sentence-level objectives are brittle here"* — [P2] says it in
**related work**, citing the NSP literature, and TG has **no** auxiliary loss, so it never measured
one. The rule stands on its other reason (it preserves §3.7's reduction, which E0b tests). Do not
quote it as [P2]'s finding.

### S-6 · H2O is an inference-time method used here as a training-time policy.
H2O scores at each decoding step over a KV cache. TG evicts gestalt slots during training, where
evicted slots retain their computation graphs. That regime crossing has no analogue in the source
and belongs in §13 as a stated limitation of the comparison.

---

## Process corrections — not about the spec's content

### P-1 · §15.1 is scheduled after the deadline the spec sets for it.
§15.1: *"Work through §3.3, §3.6, §7.1 and §10 against [P1]–[P4] directly **before week 4**."*
The plan artifact schedules it **inside** week 4. Unflagged divergence. [P4] is also still
unverified (`docs/citation-audit.md`), and §15.1 requires deriving §4.3 from it directly.

### P-2 · §16 condition 7 is unsatisfiable as written.
It requires §15.2 *"defended under push-back **in person**."* There is one person on this project.
That clause has no referent and **is not quietly reinterpreted here.**
The nearest executable substitute — an adversarial refuter pass over a §15.2 the **owner** has
written — is **weaker than the condition** and must be recorded as a substitute, never as the
condition met. §15.2 itself remains the owner's, unfilled, and no agent may draft a candidate.

### P-3 · §1's verification note is NOT discharged.
E0f is partially run: [P2], [P5], [P6], [P7] checked. [P1], [P3], [P4], [P8]–[P14] outstanding.
The note stays in the spec until that list is empty. [P11] is highest value (it is now an
*implemented baseline*), [P4] second (§15.1 depends on it).
