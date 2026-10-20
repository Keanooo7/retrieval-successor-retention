---
run_id: cf-ceiling
item: "B0-CF: the counterfactual-return ceiling. Dispatched 2026-10-06 by the PM (session a3d27993) under Brendan's delegation; the run that follows RSR run 1. Tier D (review policy v2): descriptive, PREREG committed alone, no pre-data review."
question: >-
  The gamma sweep (run/gamma-ceiling, class SIGNAL, verified 7457f43) found that a hindsight rule on
  the r_i return G_gamma = sum_k gamma^k r_i(t+k) MISSES the fact/filler headroom at every gamma in
  {0, 0.5, 0.8, 0.9, 0.95, 0.99, 1}. Does a hindsight rule on the counterfactual return
  G^Delta_gamma(t) = sum_k gamma^k Delta_i(t+k), with Delta the resample-knockout leave-one-out
  Delta-loss (src/rsr/metrics/loo.py, E0d's instrument), point at the headroom where the r_i return
  does not?
class: DESCRIPTIVE, with a pre-registered reading rule (section 7, PM decision under delegation). No gate. Input to an owner decision only.
substrate: "fresh-stream arm B ckpt-00{3000,2500}.pt, seeds 0,1,2 (READ-ONLY; sha256 = T0 manifest ~/rsr-substrate/2026-09-27/MANIFEST.sha256 and lookahead-room's CKPT_SHA256), plus the six lookahead-room-r2 D.pt (for the r_i control), checked at start and end"
seeds: [0, 1, 2]
checkpoints: [3000 (headline), 2500 (secondary, unlabelled)]
gammas: [0.0, 0.5, 0.8, 0.9]
documents: "U = [64, 1088) ∪ [4096, 4160) of each seed's generator (1088 documents), the gamma sweep's set, FULL — no subsample (section 2). ALREADY INSPECTED for r_i; no Delta on U exists at this commit."
batch: "32 documents per loo batch, U in its order (34 batches); donors come from the batch"
bootstrap: "2000 per-document resamples, paired across every arm and checkpoint of a seed; B0's boot_weights (generator 20260927 + seed) — the gamma sweep's identical replicate matrix"
falsifier: NONE (descriptive; section 7 classifies, it does not falsify a claim)
---

# PREREG: B0-CF, the counterfactual-return ceiling

**Written 2026-10-06 by a model** (Claude Opus 5.5, an `rsr-researcher` session dispatched by the
PM, session a3d27993). **Not written by Brendan.** The reading rule in §7 is the PM's decision
under delegation, transcribed verbatim.

**This file is committed alone, before any B0-CF data.** At this commit **no Δ has been computed on
any U document**, no hindsight rule on any Δ return has been simulated, and no B0-CF hit rate or
cap exists. The only Δ computed by anyone for this run is the feasibility probe's (§2), on 20
documents **outside U** (vocabulary documents 0–19 of seed 0), and it computed **no residency, no
hit rate and no cap**. The r_i rows γ ∈ {0, 0.9} that this run reproduces as a control (§6, C3)
**have been seen** (B0, γ sweep).

## 1. What this is, and what it is not

- **Descriptive measurement, no training.** Checkpoints are loaded read-only for forward passes in
  eval mode under `torch.no_grad()`. `rsr.constants.record()` is never called. Nothing is
  backpropagated anywhere.
- **It changes nothing:** not a constant, not γ's ruled value, not the retention target (C3 / any
  LOO-derived target is option C3 of spec §3.2.1 and **owner-only**), not the spec, not a gate, not
  §15. It is an input to an owner decision only.
- **Not evidence on the cognitive claim** (Kintsch & van Dijk; E7, unapproved). One corpus
  (S0-03), `M = 16`, `S = 48`, one width: **no scaling claim**.
- **A hindsight rule is not a learnable policy.** `G^Δ_γ` is read from the future. Its cap is what a
  *perfect* predictor of `G^Δ_γ` would reach under model-free residency: an upper-bound-style proxy,
  not RSR's result. The hindsight advantage grows with γ.

## 2. Feasibility (Part 1 of the brief)

**Q1. Does E0d's saved data already hold Δ for every resident slot at every step on the B0 / γ-sweep
documents?** **No.** E0d (`run/e0d`, PREREG front matter) measured
`D_E0d = [262144, 263168)` per seed — 1024 documents **disjoint from U** — and only for resident
slots under FIFO (no evicted-sentence probe). Nothing on U is reusable. `loo.py` is byte-identical
on this branch's base f31cb8b and on `run/e0d` (blob `66678130`), so **no merge of `run/e0d` is
needed**: the instrument is imported from `src/rsr/metrics/loo.py`.

**Q2. What does computing it take?** Per document (`S = 48`, `M = 16`): 632 resident-knockout rows
(Σ_t min(t, 16)), 528 evicted-probe rows (Σ_{t=16}^{47} (t − 16 + 1), identity row included) and
96 live rows (loo's live pass and the probe loop's), ≈ 1256 sentence-forward rows. Full set:
1088 × 3 seeds ≈ 4.10 M rows per checkpoint.

**Measured** (commit 2767496, `experiments/cf-ceiling/feasibility-probe.log`, rc = 0; seed 0
ckpt3000, vocabulary documents 0–19 in **one batch of 20**, `torch.set_num_threads(1)`, through
`orchestrator.slot --lane cpu-det --slots 1`, **while RSR run 1's training held 9 of the 12 cpu-det
slots**):

| quantity | value |
|---|---|
| wall time, 20 documents | 34.93 s |
| loo resident pass (632 knockout forwards of 20 rows + 48 live, incl. donor selection) | 18.94 s → **27.9 ms per batched forward** |
| evicted probe (32 forwards, 10 560 rows) | 14.61 s → **1.38 ms per row** |
| **per document** | **1.747 s** |
| donor coverage at batch 20 | assert 0.9995 (3/5475 missing), query 0.9965 (16/4599), filler 0.9501 (128/2566) |
| checks | live loss == loo's (bit-exact), residency == loo's, FIFO memory ok, identity probe worst 0.0 |

**Arithmetic.** 1.747 s × 1088 × 3 = 5 701 s ≈ **1.6 h per checkpoint, ≈ 3.2 h for both**. This run
uses batch 32, not 20 (§3): the forward cost per row does not grow with the batch, but donor
selection is a Python loop over the batch, so I bound it by scaling the **whole** loo pass by
32/20: 0.947 × 1.6 + 0.797 = 2.31 s/doc → **≈ 2.1 h per checkpoint, ≈ 4.2 h for both** (an
extrapolation, not a measurement). Both are under the brief's ~6 h of single-slot CPU.

**Decision: the full U set, both checkpoints, no subsample.** Order: ckpt3000 seeds 0, 1, 2, then
ckpt2500 seeds 0, 1, 2. Δ per (checkpoint, seed) is saved to `runs/cf-ceiling/raw/` (gitignored,
sha256 in the ledger) as it completes, and a re-launch reuses a saved file only if its recorded
config hash equals the run's.

## 3. Definition of Δ, matched to D.pt's FIFO-world rollout (Part 2)

**The world.** Each document is rolled out under FIFO (`write_at`, victim 0; slot 0 is the oldest),
in eval mode, exactly as `loo_delta_loss` rolls out — the same world as D.pt
(`experiments/lookahead-room/run.py:473-534`: FIFO's rollout, every row a one-at-a-time
counterfactual inside it; its recorder asserts the memory is `range(t − M, t)` at every full step,
and so does this run, control C4). `L_t(memory)` = sentence `t`'s mean real-target NLL
(`loo_delta_loss`'s `sent_loss`). The intervention is single-step: it changes only the forward that
reads sentence `t`.

- **Resident sentence `i`** (every `i < t` while `t < M`; `t − M ≤ i < t` when full): **Δ[t][i] =
  `loo_delta_loss(mode="resample")`'s `delta` at the slot holding `i`** = `L_t(slot := donor) −
  L_t(FIFO memory)`, the donor a same-kind, same-rank, same-step gestalt from another document of
  the batch, never asking `i`'s own question (`pick_donor`; seed = the run seed). The loss
  **increase** when `i` is knocked out. Imported, not re-implemented.
- **Evicted sentence `i`** (`i < t − M`, memory full): **Δ[t][i] = `L_t(FIFO memory) − L_t(slot 0 :=
  gest_i)`** = the loss **decrease** when `i`'s gestalt is reinstated in slot 0, displacing FIFO's
  oldest resident `t − M` — the exact memory D.pt's probe reads (`kvP[p, 0] = self.gest[i]`). Both
  terms come from one batched forward whose identity row (slot 0 := `gest_{t−M}`) is checked
  against the live loss (C4).
- Sign: **positive = the sentence helps sentence `t`**, for both kinds. `Δ[t][i]` is finite for
  every `i < t` and NaN for `i ≥ t` (C2), the layout of D.pt.
- **Missing donor.** A resident cell for which `pick_donor` finds no donor in the batch
  (loo's NaN) is **imputed 0** and counted, by kind. Batch **32** (not E0d's 16) to raise
  coverage: the probe's batch-20 coverage was 0.95 for filler. If the full-memory resident
  coverage of any (checkpoint, seed) is below **0.80** (E0d's `RESAMPLE_MIN_COVERAGE`, its PREREG),
  the run stops, exit 3.

**The asymmetry, stated plainly.** The two kinds of reading have **different baselines**:

1. A **resident** reading compares `i`'s content with a **generic same-kind donor** at the same
   rank, all of FIFO's other residents present.
2. An **evicted** reading compares `i`'s content with **the content of the sentence it displaces**,
   FIFO's oldest `t − M`. It is net of that sentence's value: `≈ v_t(i) − v_t(t − M)` (only
   approximately; knockouts are not additive). It is **0 by construction** for `i = t − M` itself.

**Consequences for reading (stated before the data):**

- At a step where FIFO's oldest is itself valuable (e.g. an assert whose query is sentence `t`),
  **every evicted sentence's reading is pushed down** at that step; where the oldest is filler, the
  evicted reading is close to "`i` versus filler". The offset is common to all evicted sentences at
  a step but not to residents.
- A candidate's future rows are resident-type until FIFO evicts it (at step `i + M`) and
  evicted-type afterwards. At one decision a **younger** candidate has more resident-type rows ahead
  of it, so any level offset between the two kinds enters `G^Δ_γ` as an **age-correlated** term
  whose weight grows with γ — the same structure the γ sweep's PREREG §2 item 3 described for D.pt.
- Every future term is a FIFO-world counterfactual, never demand in the hindsight rule's own memory
  (γ sweep PREREG §2 item 2).
- **Sensitivity variant (DIAGNOSTIC, unlabelled, never in the class).** `Δshift[t][i] = Δ[t][i] +
  Δ_res[t][slot 0]` for evicted `i` (residents unchanged) = `L_t(slot 0 := donor of t − M) −
  L_t(slot 0 := gest_i)`: the evicted reading re-baselined to a same-rank donor, as a resident
  reading is (the donor matches `t − M`'s kind, not `i`'s). It costs no extra forward. Its rule is
  reported beside the primary so the asymmetry's effect is visible; it does not enter §7.

## 4. Arms (Part 3)

- **Primary:** the hindsight rule on `G^Δ_γ[t][i] = Σ_{k=0}^{S−1−t} γ^k Δ[t+k][i]`
  (lookahead-room's `discounted_returns`, imported; `k = 0` included, truncated at document end) for
  **γ ∈ {0, 0.5, 0.8, 0.9}**: lookahead-room's `TargetPolicy` (evict the live slot with the smallest
  `G^Δ_γ[t][written_at]`, ties to the oldest), wrapped by the γ sweep's pass-through
  `AgeShapeRecorder` (imported) for the diagnostic. **ckpt3000 is the headline; ckpt2500 is
  secondary and unlabelled.**
- **Sensitivity (DIAGNOSTIC):** the same rule on `G^{Δshift}_γ`, same γ.
- **r_i control arms:** `TargetPolicy(discounted_returns(D.pt, γ))`, γ ∈ {0, 0.9} (the γ sweep's
  `hindsight_target`, imported).
- **References** (recomputed, as B0 / the γ sweep): FIFO (`FIFOPolicy`), factfiller
  (`FactFiller`, `random.Random(f"ff:{seed}:{doc_id}")`), oracle (`OraclePolicy(discounted_demand(doc,
  0.97))`, B0's; reported only).
- Residency `rsr.metrics.headroom.simulate`, `M = 16`, by B0's `pairs_of`. Same seeds, same 1088
  documents, same cap, same paired per-document bootstrap (B0's `boot_rule` on `boot_weights(1088,
  seed)`). Code reused by import from `experiments/gamma-ceiling/{run,ageshape}.py`,
  `experiments/b0-ceilings/run.py`, `experiments/lookahead-room/run.py`.
- **Hit rate:** queries whose assert is resident at the query step, pooled over U; also by bucket
  `gap_2_to_M` and `gap_gt_M`. **Cap** = `(hit − hit_FIFO) / (hit_factfiller − hit_FIFO)`,
  all-query; CI percentile 95 %; a replicate with a non-positive denominator is dropped and counted.

## 5. Diagnostics (never gating)

- **Age-shape of the target (DIAGNOSTIC):** per arm × γ × checkpoint × seed, the within-step
  Spearman ρ between `G^Δ_γ[t][written_at]` and slot age `t − written_at` over the 16 live slots of
  the hindsight rule's **own** memory at each of its eviction decisions, pooled as the lower median
  with the per-document bootstrap CI — exactly the γ sweep's Amendment 1 statistic (`spearman_rows`,
  `pooled_median_ci`, imported). ρ > 0: older slots carry larger `G`. No threshold.
- **Δ by class:** mean, sd and count of `Δ[t][i]` over full-memory steps by lookahead-room's
  `slot_class` (pending / querying / answered / filler) × kind of reading (resident / evicted), and
  the missing-donor counts by kind. Descriptive.

## 6. Controls (any failure → exit 3, and no number is reported as a result)

- **C1 manifest:** sha256 of the six checkpoints and the six D.pt equal their T0 manifest lines (and
  the checkpoints equal lookahead-room's `CKPT_SHA256`), before the first read and after the last
  computation (B0's `verify_manifest`).
- **C2 shape:** each D.pt holds the 1088 U ids of its seed, `[48, 48]` (B0's `check_shape`); each
  Δ and Δshift passes the same `check_shape` (finite at `i < t`, NaN at `i ≥ t`).
- **C3 reproduction of the γ sweep (the PM's control):** at both checkpoints and every seed, the r_i
  arms γ = 0 and γ = 0.9 reproduce `runs/gamma-ceiling/ledger.json` **exactly** (`==` on the
  float): `B.ckpt{c}.g0/g09.hit.{all,gap_2_to_M,gap_gt_M}` and `.cap.{point,lo,hi}`, plus
  `U.hit.fifo.all` and `U.hit.factfiller.all`. Since the set is the full U (no subsample), the
  γ sweep's own aggregates are the comparison; the γ sweep's ledger holds no per-document vector, so
  "per-document exactly" is met by exact equality of every aggregate the per-document vectors and
  the identical replicate matrix produce.
- **C4 instrument, per batch:** the probe loop's live loss equals loo's bit-for-bit; its residency
  equals loo's; FIFO memory is `range(t − M, t)` at every full step; the identity probe's loss is
  within `NLL_TOL = 1e-4` nats (lookahead-room's, imported) of the live loss.
- **C5 coverage:** full-memory resident donor coverage ≥ 0.80 per (checkpoint, seed) (§3).
- **C6 sizes:** three seeds, two checkpoints, four γ, 1088 documents each ran; the age-shape
  pass-through's hit equals the plain `TargetPolicy` hit (it wraps, it does not decide).

## 7. Pre-registered reading rule (PM decision under delegation — verbatim)

Per γ, at ckpt3000:
- POINTS: cap CI lower ≥ 0.80 on all 3 seeds.
- PARTIAL: not POINTS, and the point estimate ≥ 0.50 on all 3 seeds.
- MISSES: otherwise.

Class:
- CF-POINTS if any γ POINTS;
- CF-MISSES if every γ MISSES;
- CF-MIXED otherwise.

The consequence is stated in advance, as an input to an owner decision, not a decision:
- CF-POINTS puts option C3 (a LOO-derived target, spec §3.2.1, owner-only) in front of Brendan for RSR run 2.
- CF-MISSES means no target in either family points at the headroom on this substrate, and the PM recommends writing up the negative result.

Implementation notes (mine, fixed now): the rule applies to the **primary** arm (`G^Δ_γ`) only;
the Δshift arm, the r_i arms, ckpt2500 and every diagnostic are unlabelled. A NaN point or CI bound
does not satisfy a `≥`. The class is computed from the four primary γ at ckpt3000.

## 8. Exit codes (`rsr.exit_codes`)

| Code | Meaning |
|---|---|
| **0** | Ran; every control passed. Whatever the numbers and the class are. |
| **1** | A defect after the controls passed: a NaN hit rate, or a raise inside the computation. |
| **3** | Did not run, or a control (C1–C6) failed. |

Exit 2 and 5 are not used.

## 9. Author's expectation (before any B0-CF number; allowed to be wrong)

- Δ is ≈ 0 for filler and spikes for an assert at its query's step (the answer token is one of the
  query sentence's targets). I expect the spike to survive discounting better than r_i's because
  filler carries no 0.06-per-step background, so I expect the Δ caps to be **higher than the r_i
  caps at the same γ** on every seed at γ ≥ 0.5.
- I expect γ = 0 to MISS (a pending fact has no spike until its query) and the cap to rise with γ
  to 0.8–0.9. I do **not** expect POINTS on all three seeds at any γ: long-gap facts
  (`gap_gt_M`) carry a spike discounted by γ^k against summed filler noise, and the evicted
  reading's baseline asymmetry adds an age-correlated offset.
- **Predicted class: CF-MIXED** (PARTIAL at 0.8 or 0.9), low confidence; CF-POINTS next.
- C3 is expected to pass exactly (same code, tensors, documents and replicates as the γ sweep).

## 10. Artefacts

- `experiments/cf-ceiling/{cfdelta.py, probe.py, run.py}`, tests `tests/test_cf_ceiling.py`.
- `runs/cf-ceiling/{manifest.json, ledger.json, claims.json}` via `scripts/ledger.py`;
  `runs/cf-ceiling/raw/B-ckpt{c}-seed{s}.delta.pt` (gitignored; sha256 in the ledger).
- `experiments/cf-ceiling/RESULTS.md`: command, git SHA, hardware, wall time, every number
  including those that came out wrong, the controls, the labels and the class.
- Launch: only after RSR run 1's held-out measurement has finished
  (`.worktrees/rsr-run1-run/runs/rsr-run1/finalize.rc` exists; PM's ordering change), through
  `orchestrator.slot --lane cpu-det --slots 1`.
