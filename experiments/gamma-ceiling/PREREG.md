---
run_id: gamma-ceiling
item: "B0-gamma: the hindsight target-ceiling sweep over the discount gamma. Dispatched 2026-10-05 by the PM (session a3d27993) under Brendan's go for the night."
question: >-
  B0 measured the hindsight eviction rule (evict the resident slot with the lowest realised
  discounted return G_gamma of the r_i demand in D.pt) only at gamma in {0.9, 0}, and even a
  perfect predictor of G_0.9 captured only a fraction of the fact/filler headroom
  (ckpt3000 cap +0.184 / +0.134 / +0.537). Is that a discount-horizon problem (some gamma in
  the family points at the headroom) or a signal problem (no gamma in the family does)?
class: DESCRIPTIVE, with a pre-registered reading rule (section 6). No gate. Input to an owner decision only.
substrate: "lookahead-room-r2 D.pt tensors (B-ckpt{2500,3000}-seed{0,1,2}.D.pt), read-only, sha256-checked against ~/rsr-substrate/2026-09-27/MANIFEST.sha256 at start and end"
seeds: [0, 1, 2]
checkpoints: [3000 (headline), 2500 (secondary)]
gammas: [0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0]
already_seen: "gamma = 0 and gamma = 0.9 (B0 ledger runs/b0-ceilings/ledger.json). gamma in {0.5, 0.8, 0.95, 0.99, 1.0} NOT seen by anyone."
documents: "U = [64, 1088) ∪ [4096, 4160) of each seed's generator (1088 documents) — ALREADY INSPECTED (as B0)"
bootstrap: "2000 per-document resamples, paired across every gamma and checkpoint of a seed; percentile 95% CI; B0's boot_weights, torch.Generator seeded 20260927 + seed (identical replicates to B0)"
falsifier: NONE (descriptive; the reading rule of section 6 classifies, it does not falsify a claim)
---

# PREREG: B0-γ, the hindsight target-ceiling sweep

**Written 2026-10-05 by a model** (Claude Opus 5.5, an `rsr-researcher` session dispatched by the
PM, session a3d27993). **Not written by Brendan.** The reading rule in §6 is the PM's decision
under Brendan's delegation for tonight, transcribed verbatim.

This file is **committed alone, before any B0-γ computation.** At this commit this session has not
opened any `D.pt`, has not simulated any rule at any γ, and no B0-γ number exists. The γ = 0 and
γ = 0.9 rows **have been seen** (B0, `runs/b0-ceilings/ledger.json`, branch `run/b0-ceilings`
@ 6ac4351, code on `night/2026-09-30` @ 94a7ceb, which this branch is cut from). **The other five
γ values (0.5, 0.8, 0.95, 0.99, 1.0) have not been seen by anyone.**

## 1. What this is, and what it is not

- **Descriptive measurement, no training.** No checkpoint is loaded, no forward pass is run, no
  `rsr.constants.record()` is called. Model-free residency only.
- **It changes nothing.** Not a constant, not γ's ruled value (0.9 for run 1, Amendment 4), not the
  spec, not any gate, not §15. It is an input to an owner decision only.
- **Not evidence on the cognitive claim** (Kintsch & van Dijk; E7, unapproved). One corpus
  (S0-03), `M = 16`, `S = 48`, one width: **no scaling claim**.
- **A hindsight rule is not a learnable policy.** `G_γ` is read from the future of `D`. The cap of
  the hindsight rule is what a *perfect* predictor of `G_γ` would reach under model-free residency
  — an upper-bound-style proxy, not RSR's result.

## 2. How `D.pt` was produced (established before this PREREG, from the code that wrote it)

The six tensors were written by `experiments/lookahead-room/run.py::child`
(`torch.save(raw_D, ...)`, line 941), from `measure_doc` (line 863) → `run_fifo` (line 607) →
`FifoProbeRecorder` (line 473). Line numbers are at 94a7ceb.

**Answer: neither "every past sentence resident" nor "censored". `D` is uncensored, but it is a
one-at-a-time demand-if-resident probe inside FIFO's world.**

- **The rollout is FIFO's.** Each document runs alone under FIFO through the real
  `run_policy_loop` (`run_fifo`, lines 607–626). At every full-memory step the recorder asserts the
  memory is exactly FIFO's: `if written != list(range(step - self.m, step)): raise AssertionError`
  (lines 506–507).
- **A resident sentence** gets FIFO's own gated, fill-rescaled §3.2.1 share at that step:
  `r = r_of(attn, n_live, self.m)` (line 500), `self.D[step, w] = float(r[j])` (line 502). At
  underfull steps `t < M` every past sentence is resident, so every row is of this kind.
- **An evicted sentence is not censored to zero.** At every full-memory step
  (`if n_live == self.m: ... self._probe(...)`, lines 504–508) every sentence FIFO has already
  evicted is probed: `cand = [step - self.m, *range(step - self.m)]` (line 513) — row 0 is the
  identity probe of the oldest resident `t − M`, rows 1.. are every evicted sentence
  `0 … t − M − 1`. Each probe is the same step's forward on FIFO's memory with **only slot 0**
  (FIFO's oldest resident, sentence `t − M`) replaced by the evicted sentence's gestalt:
  `kvP[p, 0] = self.gest[i]` (line 517); its demand is read at slot 0 and stored:
  `self.D[step, i] = float(rp[0])` (line 534). The probe is a side computation and never feeds the
  rollout (lines 522–526).
- So at every step `t ≥ M`, `D[t][i]` is finite for **every** `i < t` (B0's control C2 checked
  this on all 6 × 1088 documents), and NaN for `i ≥ t`.

**Consequence for how the γ → 1 rows can be read (stated before the data):**

1. **No eviction-censoring artefact.** Because an evicted sentence keeps a probed demand, large γ
   does **not** mechanically favour sentences FIFO happened to keep: a sentence's future demand is
   not zeroed by FIFO having evicted it. A γ → 1 rule is therefore not trivially pulled toward
   recency by the instrument.
2. **But every future term is a FIFO-world counterfactual, not demand in the hindsight rule's own
   world.** `G_γ[t][i] = Σ_k γ^k D[t+k][i]` sums demands each measured against FIFO's other 15
   residents at step `t + k` (and FIFO's rollout context), one sentence substituted at a time —
   never jointly with the other sentences the hindsight rule would actually keep. The larger γ,
   the more of `G` comes from steps far from the decision, where the rule's memory and FIFO's
   memory have diverged most. A γ → 1 row describes "total future demand-if-resident in FIFO's
   world", not "total future demand under this rule".
3. **Two reading types are mixed, in an age-correlated proportion.** A candidate's future rows are
   resident-type (FIFO's share at its own slot) until FIFO evicts it at step `i + M`, and
   probe-type (slot 0, displacing sentence `t − M`) afterwards. Within one eviction decision a
   younger candidate has more resident-type rows ahead of it. Any level offset between the two
   reading types therefore enters `G_γ` as an age-correlated term whose weight grows with γ.
4. **Truncation at document end is uniform within a decision.** Every candidate at step `t` sums
   over the same `k = 0 … S − 1 − t`, so γ = 1.0 (the undiscounted sum to the end of the
   document) introduces no within-decision horizon asymmetry; the argmin at each step compares
   like-length sums.
5. **The hindsight advantage grows with γ** (lookahead-room's own caveat: "the hindsight is
   asymmetric in favour of gamma > 0"). A higher cap at larger γ is partly more future
   information, not only a better horizon.

## 3. Inputs, declared already inspected

- **Tensors.** Exactly the six files
  `.worktrees/lookahead-room/runs/lookahead-room-r2/raw/B-ckpt{2500,3000}-seed{0,1,2}.D.pt`
  (relative to the main checkout), read-only, `dict[doc_id → Tensor[48, 48]]`, step-major.
- **Documents.** The U range `[64, 1088) ∪ [4096, 4160)`, 1088 per seed, rebuilt model-free by
  `experiments/lookahead-room/run.py::documents(seed)` (imported through B0's module, not copied).
- **🔴 Declared: the U range is ALREADY INSPECTED,** and γ ∈ {0, 0.9} on it have been seen.

## 4. Definitions (fixed now)

- `M = 16`, `S = 48`, from lookahead-room's module (as B0).
- **Returns.** `G_γ[t][i] = Σ_{k=0}^{S−1−t} γ^k D[t+k][i]`, truncated at document end, `k = 0`
  included, computed by lookahead-room's `discounted_returns` (imported). γ ∈ {0, 0.5, 0.8, 0.9,
  0.95, 0.99, 1.0}. At γ = 0, `G = D`; at γ = 1.0, `G` is the undiscounted sum of future demand to
  the end of the document.
- **Hindsight rule** `rule_γ`: lookahead-room's `TargetPolicy` on `G_γ` — evict the live slot with
  the smallest `G_γ[t][written_at]`, ties to the lowest slot index (the oldest). Exactly B0's
  `rule_g0` / `rule_g09`, which are `TargetPolicy` on `D` and on `discounted_returns(D, 0.9)`.
- **Residency.** `rsr.metrics.headroom.simulate`, `M = 16` (imported, not re-implemented).
- **References** (as B0, recomputed): FIFO (`FIFOPolicy`); factfiller (lookahead-room's
  `FactFiller`, `random.Random(f"ff:{seed}:{doc_id}")`).
- **Hit rate:** queries whose assert is resident at the query step, pooled over U.
  Also by bucket `gap_2_to_M` (`2 ≤ gap ≤ M`) and `gap_gt_M`.
- **Cap:** `(hit − hit_FIFO) / (hit_factfiller − hit_FIFO)`, all-query.
- **CI:** B0's `boot_rule` with B0's `boot_weights(1088, seed)` — 2000 per-document resamples,
  percentile 95%, paired across every γ and checkpoint of a seed (the identical replicate matrix
  B0 used, so the γ ∈ {0, 0.9} CIs reproduce B0's exactly). A replicate whose denominator is ≤ 0
  is dropped from the cap CI and counted.

## 5. Outputs

Per seed × checkpoint × γ: hit (all, `gap_2_to_M`, `gap_gt_M`), `hit − FIFO` point and CI, cap
point and CI, cap-dropped count. FIFO and factfiller hit per seed. At ckpt3000, each γ's label and
the sweep class (§6). ckpt2500 numbers are reported **unlabelled** (the rule is defined on
ckpt3000 only). Seed is the unit; nothing is averaged into a single claim.

## 6. Pre-registered reading rule (PM decision under delegation — verbatim)

Per γ, at ckpt3000, using each seed's cap 95% CI:
- POINTS: CI lower ≥ 0.80 on all 3 seeds.
- PARTIAL: not POINTS, and point estimate ≥ 0.50 on all 3 seeds.
- MISSES: otherwise.
Sweep class: HORIZON if any γ is POINTS; SIGNAL if every γ is MISSES; MIXED otherwise. State in the PREREG that γ=0 and γ=0.9 were already seen (B0) and the other five were not. Nothing here changes a constant, γ's ruled value (0.9 for run 1, Amendment 4), the spec, or any gate. It is an input to an owner decision only.

Implementation notes (mine, fixed now): a NaN CI bound or point (every replicate dropped, or a
non-positive point denominator) does not satisfy a `≥` comparison. γ = 0 and γ = 0.9 are labelled
like the others; they are flagged "already seen" beside their label.

## 7. Controls (any failure → exit 3, and no numbers are reported as results)

- **C1 manifest.** sha256 of each of the six `D.pt` equals its line in
  `~/rsr-substrate/2026-09-27/MANIFEST.sha256`, before the first read and after the last
  computation (B0's `verify_manifest`, imported).
- **C2 shape.** Each `D.pt` holds exactly the 1088 U doc ids of its seed, each `[48, 48]`, finite
  at `i < t`, NaN at `i ≥ t` (B0's `check_shape`).
- **C3 reproduction of B0 (the control the PM set).** At both checkpoints and every seed, γ = 0
  and γ = 0.9 reproduce B0's ledger **exactly** (`==` on the float):
  `B.ckpt{c}.hit.rule_g0`, `B.ckpt{c}.cap.rule_g0.{point,lo,hi}`, and the same for `rule_g09`;
  plus `U.hit.fifo.all` and `U.hit.factfiller.all`. If they do not, the run stops (exit 3) and
  that is the report.
- **C4 seeds and sizes.** Three seeds, two checkpoints, seven γ, 1088 documents each ran.

## 8. Exit codes (`rsr.exit_codes`)

| Code | Meaning |
|---|---|
| **0** | Ran; every control passed. Whatever the numbers and the class are. |
| **1** | A defect after the controls passed: a NaN hit rate, or a raise inside the computation. |
| **3** | Did not run, or a control (C1–C4) failed. |

Exit 2 and 5 are not used.

## 9. Author's expectation (before any B0-γ number; allowed to be wrong)

- From B0, raising γ from 0 to 0.9 raised the ckpt3000 cap on every seed (−0.039 → +0.184,
  −0.136 → +0.134, +0.234 → +0.537). I expect the cap to keep **rising from 0.9 to 0.95–0.99**
  on every seed, and to be roughly flat or slightly lower at 1.0 than at 0.99. I expect 0.5 and
  0.8 to lie between the γ = 0 and γ = 0.9 values (monotone in γ below 0.9).
- I do **not** expect any γ to reach POINTS (CI lower ≥ 0.80 on all three seeds); seed 1 is the
  likeliest to fall short. I expect seed 2 may pass 0.50 at high γ and seeds 0 and 1 not to.
- **Predicted sweep class: SIGNAL** (low confidence; MIXED is the next most likely).
- C3 is expected to pass exactly (same code, same tensors, same documents, same replicates).

## 10. Artefacts

- `experiments/gamma-ceiling/run.py` (imports B0's `run.py` and lookahead-room's `run.py`; does not
  re-implement `simulate`, the references, the returns or the bootstrap), tests in
  `tests/test_gamma_ceiling.py`.
- `runs/gamma-ceiling/{manifest.json, ledger.json, claims.json}` via `scripts/ledger.py`.
- `experiments/gamma-ceiling/RESULTS.md` — command, git SHA, hardware, wall time, every number
  including any that came out wrong, the control outcomes, the labels and the class.

---

## Amendment 1 (2026-10-06): age-shape of the target (DIAGNOSTIC)

**Requested by the PM (session a3d27993) after the main run, and written after it.** The PM's
message asked for this amendment "BEFORE any data run". **That was not possible: it arrived after
the main run had completed (`runs/gamma-ceiling/ledger.json`, committed in c186ac6), and after its
cap numbers, labels and class had been seen by this session.** This amendment is therefore **not a
pre-registration of the main result**, and it changes nothing in §§1–9: not the arms, not the
reading rule, not a label, not the class (SIGNAL stands as computed). What it does precede is the
diagnostic itself: **at this commit no age-shape number below has been computed by anyone.**

**The PM's text (verbatim):** "Secondary, descriptive, never gating: for each γ × ckpt × seed,
report the within-step Spearman correlation between G_γ and slot age (t − written_at) across the
resident slots at full-memory eviction steps, pooled across steps as the median of per-step ρ with
a per-document bootstrap 95% CI. Reason: a long-horizon target can become age-shaped (spec
falsifier 1, vacuity). A γ that POINTS at the headroom but whose target is strongly age-ranked is a
different finding from one that is not. Label the column "age-shape of the target (DIAGNOSTIC)". It
does not change the class rule. Age is used here only as an analysis variable, never as a scorer
input."

**Operational definitions (mine, fixed now, before any value):**

- **Resident slots** = the 16 live slots of the hindsight rule's **own** memory at each of its
  eviction decisions in `simulate` (steps `t ∈ [M, S)`, memory full) — the exact set its argmin is
  taken over. Recorded by a pass-through wrapper around lookahead-room's `TargetPolicy` that reads
  `slots.written_at` and returns the wrapped policy's choice unchanged. Age and `G` are read; age is
  never passed to the rule.
- **Per-step ρ** = Spearman's ρ between `G_γ[t][written_at]` and `age = t − written_at` over those
  16 slots: Pearson correlation of average ranks (ties in `G` get the mean of their ranks). A step
  whose 16 `G` values are all equal has undefined ρ; it is dropped and counted (`n_steps_undefined`).
- **Sign:** ρ > 0 means older slots carry a larger `G` (the target ranks old slots as worth
  keeping); ρ < 0 means newer slots carry a larger `G` (recency-shaped, FIFO-like ranking). No
  threshold is set on ρ (none exists in the rulings; none is invented).
- **Pooling:** the **lower median** of the per-step ρ over every defined step of the 1088 documents
  (the smallest value whose cumulative count reaches half the total).
- **CI:** per-document bootstrap, 2000 replicates, B0's `boot_weights(1088, seed)` (the same
  replicate matrix as the cap CIs); in a replicate each step carries its document's multiplicity,
  and the statistic is the weighted lower median; percentile 95%.
- **Control A1-C (pass-through):** the wrapped rule's pooled all-query hit rate equals the main
  ledger's `B.ckpt{c}.<g>.hit.all` for that seed **exactly**; any difference is exit 3. Plus C1
  (T0 manifest sha256 at start and end) as in §7.
- **Output:** a separate run, `runs/gamma-ceiling-ageshape/` (script
  `experiments/gamma-ceiling/ageshape.py`), so the main ledger is not rewritten. Reported in
  `RESULTS.md` in a column labelled **"age-shape of the target (DIAGNOSTIC)"**, beside the caps, for
  every γ × ckpt × seed. It is never gating and does not enter the class rule.
- **No expectation is registered for the values** (the cap numbers were already seen, so any
  expectation written now would not be blind).
