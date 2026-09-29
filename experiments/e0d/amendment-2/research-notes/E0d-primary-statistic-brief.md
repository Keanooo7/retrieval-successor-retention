# Research brief: E0d's primary statistic

- **For:** Brendan's own research into the primary statistic that validates `r_i`.
- **Prepared:** 2026-09-27 by the RSR PM session.
- **Sources:** every fact carries its source. Numbers marked **[L]** come from a committed ledger. **[V]** means verified by the PM or by `rsr-verifier` in this session. **[S]** means a scratch computation not yet in any ledger, which should be treated as provisional.
- **What this is not:** this brief proposes no statistic. It gives the problem, the constraints, the data properties and the process, so you can choose.

---

## 1. What E0d is for, and what hangs on it

- **The question (spec §3.2.1, `docs/spec/rsr_model_spec_v0.5.md:189`):** does `r_i`, the retrieval signal RSR trains its retention head ψ̂ on, measure what each memory slot is actually *worth*?
- **The spec's operationalisation:** "On a held-out subsample, ablate slot *i* and measure Δ next-sentence loss (leave-one-out). Report Spearman ρ(`r_i`, LOO Δloss). **If they disagree, LOO is truth and `r_i` is a confound** — and §3.4's redundancy term is the specified response."
- **Its status:** a **kill gate** (spec table `:548`: "High ρ. If not, LOO is truth and `r_i` is a confound"). The spec gives **no numeric threshold**.
- **Downstream consumers:**
  - B2, the learned-ψ̂ probe, whose reading is gated on E0d (PLAN-v4);
  - the D2 substrate decision;
  - ADR-0009 (the learning path);
  - falsifier 5 (redundancy, spec `:150`).

### Two different levers. Keep them separate.

| Lever | What changes | Whose call |
|---|---|---|
| **(a) The comparison statistic** | How agreement between `r_i` and LOO is scored: Spearman, stratified, per-step, AUC, … | Pre-data amendment to the E0d PREREG. A technical decision, with the threshold ratified by you. |
| **(b) The ground truth itself** | Replacing LOO as "truth", e.g. with a multi-slot or trajectory-level causal measure | **A spec change**: §3.2.1 names LOO as truth. That needs a correction in `docs/spec-corrections.md`, which only you write. |

Your phrase "alternatives to leave-one-out" could mean either. Both are open to you; they just go through different doors.

---

## 2. The objects being compared

### 2.1 `r_i(t)`, the retrieval signal (RESEARCH-CONTEXT §4.1; `src/rsr/retention/reward.py`)

```
r_i(t) = Σ_{l,h} ‖ g_mem^(l) · α_{l,h,i} · W_O^(l,h) v_{l,h,i} ‖₂    (correction 17: g_mem included)
share_i(t) = r_i / Σ_j r_j ;   r_i(t) = share_i(t) · |memory_t| / M
```

- **It is a share.** At full memory, **Σ_i r_i(t) = 1 on every step, by construction.** So `r_i` can never be "zero everywhere", and it always ranks the slots.
- **Gated vs raw (correction 17):** the memory gates g_mem grow over training, so the two forms differ. **Both must be reported against LOO.**
- **Per-layer profile (correction 18):** reported once, before summing over layers.
- **Eval mode only (correction 20).**
- **Underfull steps are rescaled, not masked (D-7).** That biases stream-initial slots downward, which is why E0d's primary uses full-memory steps only.

### 2.2 LOO Δloss, the causal "truth" (`src/rsr/metrics/loo.py`)

- **Single-step intervention.** At step t, slot i's content is replaced; sentence t's mean real-target NLL is compared with the live NLL. The memory at t+1 is the *live* one, so **Δ is the slot's effect on sentence t alone, not on the trajectory.**
- **Knockout kinds:**
  - **resample (primary):** a same-kind gestalt from another document, at the same rank and step. It stays in distribution. The donor never carries the queried key or answer object.
  - **zero (secondary):** out of distribution. The key becomes position-only, and the value becomes the projection bias.
- **The validity mask is never changed.** A knocked-out slot is still attended to; it just carries different content.
- **bos-copy leak.** TG copies the previous sentence's gestalt into token 0, so at gap 1 the fact survives any memory knockout. There are `*_bos_off` twins for this case.
- **Cost.** One extra forward per (cell × knockout), so it is "affordable only on a subsample" (spec §13).
- **Single-slot LOO cannot see redundancy (D-3, spec `:62`).** Set value under LOO is *submodular*. Two redundant slots each look evictable, because knocking out one leaves the other. **Any statistic built on single-slot LOO inherits this blind spot.**

---

## 3. The data properties that decide what can work

Substrate: fresh-stream arm B, ckpt3000, 3 seeds, frozen. Corpus: S0-03 synthetic, **S = 48 sentences, M = 16 slots.**

### 3.1 Signal is extremely sparse

Only query sentences depend on memory. Figures are per seed on the already-inspected set E `[64,1088)`, 1024 documents, **[V]** (the reviewer's script, re-run by the PM with rc 0):

| Quantity | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Full-memory steps | 32768 | 32768 | 32768 |
| Full-memory cells (step × slot) | 524288 | 524288 | 524288 |
| Query steps (share of steps) | 13877 (0.423) | 13841 (0.422) | 13749 (0.420) |
| Query steps with the asked fact still resident under FIFO | 10192 | 10203 | 10139 |
| **Answer-bearing cells (share of all cells)** | **0.0194** | **0.0195** | **0.0193** |
| Answer-bearing cells as a share of query-step cells | ≈ 0.046 | ≈ 0.046 | ≈ 0.046 |

About 58% of steps have Δ ≈ 0 for every slot (the reviewer's count). **[S]**

### 3.2 When the signal is present, it is large

Answer NLL on held-out gap 2..16, **[L]** `runs/fresh-stream/ledger.json`:

| | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Live | 0.29087 | 0.26138 | 0.22961 |
| All memory zeroed | 2.81859 | 2.82067 | 2.84604 |

So retrieval is worth about **2.5 nats per answer token.** The effect is big, but it sits in about 2% of cells.

### 3.3 Why the two statistics tried so far fail

1. **Version 0, pooled Spearman over all full-memory cells.** It ranks about 98% near-zero noise against `r_i`, so "CONFOUND" is close to built in, whatever `r_i`'s quality. *Blocker in the pre-data review.*
2. **Version 1 (Amendment 1), Spearman over query-step cells (ρ_Q), with a positive control (C10: `r_i` replaced by true demand, 1 on the answer cell and 0 elsewhere).**
   - With answer-cell share p ≈ 0.046, a perfect 0/1 proxy correlates with a continuous Δ by at most **√(3p(1−p)) ≈ 0.36** (Spearman with a binary variable).
   - **[V]** simulation, re-run by the PM (rc 0): ρ_Q(PC) = 0.3630, CI [0.3614, 0.3647] when off-answer Δ is continuous.
   - If off-answer Δ is *exactly* 0 (ties), ρ_Q(PC) = 1.0.
   - **The ceiling therefore depends on an unmeasured fact: are off-answer LOO deltas exact ties or continuous noise?** That is being measured now (§6).

### 3.4 What `r_i` does by slot age and kind

- **Demand is U-shaped in age/rank.** It is highest at the newest and oldest slots. **[S]** REDTEAM-v1 §1a; the same numbers were reproduced by B0 **[V]**.
- **Pending-fact vs filler immediate demand** is large in young slots and decays with age:

  | Age | Seed 0 (pend / fill) | Seed 1 (pend / fill) | Seed 2 (pend / fill) |
  |---|---|---|---|
  | 1 | 0.0930 / 0.0580 | 0.0999 / 0.0831 | 0.1047 / 0.0615 |
  | 8 | 0.0580 / 0.0503 | 0.0578 / 0.0507 | 0.0591 / 0.0422 |
  | 14 | 0.0577 / **0.0632** | 0.0489 / **0.0598** | 0.0711 / 0.0652 |

  The gap reverses at ages ≥ 13–14 on seeds 0 and 1.
- **Under FIFO at full memory, rank = age exactly.** ψ̂ is **forbidden** to see age (CLAUDE.md). So any agreement between `r_i` and LOO that is carried only by age is **useless to ψ̂**. The primary statistic must remove, or at least separate, the age channel. That is why the current PREREG has a rank-stratified companion (ρ_Q,rank).
- **Expected discounted demand by sentence kind** (γ = 0.9, ckpt3000), **[L][V]** `runs/b0-ceilings/ledger.json`, verifier CONFIRMED:

  | Kind | Seed 0 | Seed 1 | Seed 2 |
  |---|---|---|---|
  | assert (fact) | 0.573 | **0.405** | 0.586 |
  | query | 0.534 | **0.510** | 0.477 |
  | filler | 0.459 | 0.342 | 0.318 |

  On seed 1, facts rank below their own questions, so the seeds are not interchangeable.
- **The inferential unit is 3 seeds of one training lineage.** Document-level cluster bootstraps capture within-seed variance only.

### 3.5 The decision the statistic ultimately serves

Eviction is an **argmin over the 16 live slots at each full-memory step**. What matters for RSR is whether `r_i` (and so ψ̂) ranks the *lowest-value* slot correctly **within a step**. It matters less whether `r_i` correlates with Δ across steps. The current PREREG already reports, as secondaries, ρ_step (within-step Spearman) and bottom-1 agreement (argmin r_i = argmin Δ; chance 1/16).

---

## 4. Constraints any candidate must meet

These come from the spec, CLAUDE.md, and the two failed versions.

| # | Requirement | Source |
|---|---|---|
| R1 | **Attainable ceiling well above the threshold** under a perfect proxy, given about 2–5% signal-bearing cells, whether off-signal Δ is tied or continuous | §3.3 |
| R2 | **A known null value**, e.g. `r_i` shuffled within a step, and **power** at 1024 docs × 3 seeds with the per-document cluster bootstrap | PREREG §3.1 |
| R3 | **The age channel is removed or separated.** Agreement only through age must not pass. | CLAUDE.md; PREREG §3.1 item 2 |
| R4 | **Strata fixed by document structure before data,** never by Δ or `r_i` | Amendment 1 |
| R5 | **Relevant to within-step argmin decisions** | §3.5 |
| R6 | **Faithful to §3.2.1's intent** ("does r_i track causal value?"). If it departs from "Spearman ρ(r_i, LOO)", say so. A departure from **LOO-as-truth** is lever (b), a spec correction. | spec `:189` |
| R7 | **Report gated and raw `r_i`, and resample and zero knockouts** | corrections 17, 18 |
| R8 | **A threshold with a stated meaning** on the chosen statistic, ratified by you, and named in the ruling | Amendment 1, A1.8 |
| R9 | **Ties handled explicitly.** Most Δ may be exactly 0. | §3.3 |
| R10 | **Optional:** says something about **redundancy (D-3)**, or states that it cannot | spec `:62`, `:150`, `:290` |

---

## 5. The process for adopting your choice

1. **You choose** the statistic and a threshold rationale. Nothing here pre-empts you: no E0d data exists, and nothing will be committed without your pick.
2. **Amendment 2.** A researcher writes it: append-only, pre-data, committed alone on `run/e0d`, and citing the ceiling finding.
3. **Independent adversarial review** of the amendment. Any blocker produces a further pre-data fix.
4. **Runner update.** A researcher updates `experiments/e0d/run.py`: tests first, mutations proven against the full suite.
5. **Your rulings.** You commit the three rulings the runner checks (C8):
   - `R-*-retrieval-shown*`;
   - `R-*-sprint0-gate*`;
   - `R-*-rho-star*`, containing `RHO_STAR: <value>`. It must **name the statistic** it ratifies the threshold on. If the threshold is not a ρ, the file and key name can be changed in Amendment 2.
6. **The run.** It must also pass:
   - a T0 record at `runs/t0-substrate/manifest.json`;
   - the disjointness and closure controls.

   Documents come from the fresh range `[262144, 263168)`.
7. **Then:** verifier, review, merge to `night/<date>`, and you move `main`.

**If you choose lever (b)** (a different ground truth): you write the correction first, then steps 2–7.

---

## 6. In flight, and how it relates to your research

- **A researcher is running E0D-A2.** It is **draft-only; nothing will be committed.** Its step 1 **measures the missing fact in §3.3**: on set E, seed 0, at most 64 docs, it records the fraction of off-answer LOO Δ that is exactly 0, and the quantiles of |Δ|. That result will be appended to this brief as §7 when it lands.
- **Its step 2** drafts candidate statistics, in `reviews/E0D-AMENDMENT-2-DRAFT.md`. **Treat that as one input to your research, not a recommendation.** The PM will not adopt it over your choice. If you would rather it did not draft candidates at all, say so and it will be stopped after step 1.

## 7. Where everything is

| What | Where |
|---|---|
| E0d PREREG, base + Amendment 1 + erratum | `experiments/e0d/PREREG.md` on branch `run/e0d` (`268b947`) |
| E0d runner | `experiments/e0d/run.py` on `run/e0d` (`4918f2d`) |
| The pre-data review that found the first flaw | `~/Documents/RSR-2026-09-27-plan/reviews/PREREG-REVIEW-e0d-b2.md`, with `e0d-cell-structure.py` |
| Ceiling simulation | `/private/tmp/claude-501/-Users-keanooo7-retrieval-successor-retention/66de2324-e06a-4f6e-ac11-cbd445d2063a/scratchpad/b2/ceiling_sim.py` |
| LOO instrument | `src/rsr/metrics/loo.py` (main) |
| `r_i` | `src/rsr/retention/reward.py` (main) |
| Spec text | `docs/spec/rsr_model_spec_v0.5.md:62, :139, :150, :189, :290, :548` |
| Corrections 17, 18, 20 | `docs/spec-corrections.md` |
| B0 ledger (kind means) | `runs/b0-ceilings/ledger.json` on branch `run/b0-ceilings` |
| Age tables (scratch) | `~/Documents/RSR-2026-09-27-plan/REDTEAM-v1.md` §1a |

---

## 8. The missing fact, now measured (added 2026-09-27, PM-verified)

- **Setup:** set E `[64,128)` (64 docs, already inspected), seed 0, arm B ckpt3000, resample LOO.
- **What was not done:** no `r_i` was computed, and no document in E0d's range was touched.
- **Files:** scripts, logs and cell data are in `~/Documents/RSR-2026-09-27-plan/reviews/e0d-a2/`. `step1_cells.npz` has sha256 `1338de4b…`.
- **Check:** the PM recomputed the headline figures from the saved cells, and they match.

### Off-answer LOO Δ is continuous, not tied, so the §3.3 ceiling binds

Exact zeros: **0 of 31,846** off-answer full-memory cells. 31,835 of the values are distinct.

Quantiles of |Δ| (nats per sentence, resample knockout):

| Population | q10 | q25 | q50 | q75 | q90 | q99 | max |
|---|---|---|---|---|---|---|---|
| All full-memory, off-answer | 0.000631 | 0.00173 | 0.00456 | 0.0124 | 0.0364 | 0.207 | 1.30 |
| Query-step, off-answer | 0.000994 | 0.00285 | 0.00943 | 0.0315 | 0.0812 | 0.311 | 1.30 |
| Non-query steps | 0.000498 | 0.00135 | 0.00323 | 0.00688 | 0.0132 | 0.0371 | 0.145 |
| **Answer cells** (652 at full memory) | 0.620 | 0.927 | **1.20** | 1.40 | 1.58 | 1.93 | 2.05 |

- Every full-memory answer-cell Δ is **positive**.
- The answer cell is LOO's **argmax within its step in 640/652 = 0.982** of answer steps.
- The answer-cell share of query-step cells is 663/14144 = 0.0469, so √(3p(1−p)) = 0.366.

### The small off-answer Δ is reproducible, not noise

Spearman correlation of LOO with itself, re-run with donor seeds 1 and 2:

| Cells | Donor seed 1 | Donor seed 2 |
|---|---|---|
| Query-step off-answer | 0.569 | 0.570 |
| All query-step cells | 0.626 | 0.627 |
| Answer cells | 0.926 | 0.919 |

This changes two things:
- **The ρ-family (Spearman over many cells)** is dominated by how a score orders the ~95% of cells with small but **real** Δ. It is not a pure noise floor. A second independent LOO run tops out at ρ_Q ≈ 0.63.
- **§3.3's ceiling belongs to the binary positive control, not to ρ_Q itself.** On real Δ the binary control scores ρ_Q = 0.3633. An LOO replicate used as the control scores 0.626.

### Corrections to this brief's earlier text

1. §3.3 said E0d "cannot return AGREE or CONFOUND". That holds for **C10 as written**. Changing the *control* is also a lever, not only changing the statistic.
2. The brief's example "Spearman restricted to the answer-step slots" is **still capped** with a binary control, at 0.416.

### A neutral candidate table (simulation on the real Δ above, synthetic r_i)

It is input to your research, **not a recommendation**. It is also **not independently reviewed**. The per-candidate details, the draft amendment text, and threshold *options* (flagged as model-derived) are in `reviews/E0D-AMENDMENT-2-DRAFT.md`.

| Candidate | Ceiling: binary control / LOO-replicate control | Null (shuffled within step) | Age-only | Recency | Note |
|---|---|---|---|---|---|
| K0: ρ_Q as committed | 0.363 / 0.626 | ≈ 0 | 0.162 | 0.186 | Every seed reads CEILING at 0.5 |
| K0′: ρ_Q, with the control replaced by an LOO replicate | – / 0.626 | ≈ 0 | – | – | No synthetic r_i reached 0.5 (best 0.330); a near-perfect answer-slot picker scores 0.166 |
| K1a: ρ_Q / ρ_Q(binary control) | 1 by construction; unbounded above (replicate 1.72) | ≈ 0 | 0.445 | **0.512** | **Recency alone clears 0.5** |
| K1b: ρ_Q / ρ_Q(LOO replicate) | binary control 0.581 / 1 | ≈ 0 | 0.258 | 0.297 | Read as "fraction of replicate agreement" |
| K4: Spearman over answer-step cells | 0.416 / 0.635 | ≈ 0 | – | – | Still capped with a binary control |
| K6: mean within-step Spearman, answer steps | 0.420 / 0.619 | ≈ 0 | – | – | Capped with a binary control |
| K2: top-1, argmax r_i = argmax Δ on answer steps | 0.965 / 0.983 | 0.047–0.056 (chance 1/16) | 0.243 | 0.243 | Not a ρ; needs a new bar |
| K3: AUC of LOO's top slot, answer steps | 0.982 / 0.996 | 0.49 | 0.770 | 0.752 | 0.5 is chance, so it needs a new bar |

- **Power** is about 1 for every candidate (bootstrap sd ≤ 0.005 at n = 1024, assuming iid documents). The choice therefore turns on **where the bar sits and what the statistic means**, not on power.
- **Believed, not verified:**
  - seed-to-seed spread, since only seed 0 was measured;
  - that the synthetic r_i family spans the realistic shapes;
  - the √n scaling.

### The trade-off, stated neutrally

- **ρ-family (K0, K0′, K1a, K1b, K4, K6):**
  - scores how `r_i` orders *all* query-step cells;
  - mostly the ~95% with small but real Δ;
  - is closest to the spec's literal "Spearman ρ(r_i, LOO)".
- **K2 / K3:**
  - score only whether `r_i` picks the slot LOO says matters most at each answer step;
  - keep "LOO is truth";
  - are closest to the eviction decision (§3.5);
  - but depart from the spec's named statistic (requirement R6);
  - and need a new bar.
- **Age-only and recency baselines** are the §4 R3 check. Any bar must sit clearly above them.
