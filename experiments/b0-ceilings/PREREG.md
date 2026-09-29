---
run_id: b0-ceilings
item: "PLAN-v4 §2B B0, zero-cost ceilings. Governing plan: ~/Documents/RSR-2026-09-27-plan/PLAN-v4.md"
question: >-
  On the already-inspected U range, what do three label-driven eviction rules built from the
  existing lookahead-room-r2 demand tensors D.pt reach, as model-free residency: (i) the
  kind-oracle argmin E[G_gamma | kind], (ii) the kind x age-band reference (NOT a legal psi_hat),
  (iii) the age-only rule argmin E[G_gamma | age]? And the F1/F2 tables put into a ledger.
class: DESCRIPTIVE. No gate. No decision reads it. It feeds only B2's "already seen" addendum.
substrate: "lookahead-room-r2 D.pt tensors (B-ckpt{2500,3000}-seed{0,1,2}.D.pt), read-only, sha256-checked against the T0 manifest at start and end"
seeds: [0, 1, 2]
checkpoints: [2500, 3000]
gammas: [0.9, 0.0]
documents: "U = [64, 1088) ∪ [4096, 4160) of each seed's generator (1088 documents) — ALREADY INSPECTED"
bootstrap: "2000 per-document resamples, paired across every rule, checkpoint and gamma of a seed; percentile 95% CI; torch.Generator seeded 20260927 + seed"
falsifier: NONE (descriptive; see §1)
---

# PREREG: B0, zero-cost ceilings

**Written 2026-09-27 by a model** (Claude Opus 5.5, an `rsr-researcher` session, item B0 of
PLAN-v4). **Not written by Brendan.**

This file is **committed alone, before any B0 computation.** At this commit no `D.pt` has been
opened by this session, no B0 rule has been simulated, and no B0 number exists.

## 1. What this is, and what it is not

- **Descriptive only.** B0 has **no gate** and **no decision rule**. Nothing reads a B0 number
  to decide anything. Its exit code says only whether it ran with every control passing.
- **Its only reach:** B2's "already seen" addendum (`experiments/b2-psi-probe/PREREG.md`, the
  §A addendum slot, PLAN-v4 §2B B2 "Commit order" step 4). B0's numbers are listed there, with
  this run's ledger and SHA, so that B2's reader knows what its authors had seen.
- **It may never change B2's decision rules, thresholds, arms, ranges or targets,** and it may
  never supply **Q2's comparator.** B2's kind-oracle is **re-estimated on B2's own `FIT_TRAIN`**
  (B2 §6, §9.7: "never taken from B0's U-range numbers"). B0's kind-oracle is a U-range
  description, and nothing more.
- **Falsifier: NONE.** It is run because PLAN-v4 §1 cites F1/F2 numbers from `/private/tmp`
  scripts with no ledger, and B2's addendum must cite ledgered numbers.
- **Not evidence on the cognitive claim** (Kintsch & van Dijk; E7, unapproved). Not E1, not B2.
  One corpus (S0-03), `M = 16`, one width: **no scaling claim**.
- **Nothing trained; nothing recorded.** No model is loaded, no forward pass is run, no
  `rsr.constants.record()` is called. No frozen constant, spec text, corpus or §15 is touched.

## 2. Inputs, declared already inspected

- **Tensors.** Exactly the six files
  `.worktrees/lookahead-room/runs/lookahead-room-r2/raw/B-ckpt{2500,3000}-seed{0,1,2}.D.pt`
  (paths relative to the main checkout). Each is a `dict[doc_id → Tensor[S, S]]`, step-major:
  `D[t][i]` is W10's **demand-if-resident** of sentence `i` at step `t` (FIFO's own gated,
  fill-rescaled `r_i(t)` for a resident; the rank-0 probe for a sentence FIFO has evicted), NaN
  for `i ≥ t` (lookahead-room `run.py::child`, `measure_doc`, `FifoProbeRecorder`).
- **Documents.** The U range, **`[64, 1088) ∪ [4096, 4160)`**, 1088 documents per seed, rebuilt
  model-free by `experiments/lookahead-room/run.py::documents(seed)` (imported, not copied).
- **The reference ledger.** `runs/lookahead-room-r2/ledger.json` (committed), used only by the
  reproduction control C3.
- **🔴 Declared: the U range is ALREADY INSPECTED.** Every `U.hit.*` rule, the class `D` means by
  age, and the F1/F2 scratch numbers of PLAN-v4 §1 and B2 §A have been seen by the PM, the red
  teams and B2's author. B0 describes data whose summaries are already known. It is **not** a
  fresh-range measurement, and nothing it shows may be read as out-of-sample.
- **🔴 Declared: in-sample.** Every class mean below is estimated on the same U documents its
  rule is then scored on (per seed, per checkpoint, per γ). A ceiling estimated in-sample is
  optimistic by construction. That is accepted: B0 is descriptive.

## 3. Definitions (fixed now)

- `M = 16`, `S = 48`, read from S0-03's config via lookahead-room's module and asserted.
- **Full-memory steps:** `t ∈ [M, S)`. **Rows:** every `(t, i)` with `t ∈ [M, S)` and `i < t`
  (the U rows of B2 §5 restricted to full-memory steps, as B2 §6's kind-oracle class means).
- **Returns.** `G_γ[t][i] = Σ_{k=0}^{S−1−t} γ^k D[t+k][i]`, truncated at document end,
  `k = 0` included, computed by lookahead-room's `discounted_returns` (imported). γ ∈ {0.9, 0};
  at γ = 0, `G = D`.
- **Kind** of sentence `i`: the generator's `doc.sentences[i].kind` ∈ {assert, query, filler}.
- **Class** of sentence `i` at step `t` (F1 only): lookahead-room's `slot_class` ∈ {pending,
  querying, answered, filler} (an assert is pending before its query step, querying at it,
  answered after; a query sentence counts as filler there, as in W10).
- **Age** of sentence `i` at step `t`: `t − i` ∈ {1, …, 47}.
- **Age bands (fixed now):** `A1 = [1, 4]`, `A2 = [5, 8]`, `A3 = [9, 12]`, `A4 = [13, 16]`,
  `A5 = [17, 24]`, `A6 = [25, 47]`. The first four tile FIFO's resident ages; A5 and A6 are
  sentences FIFO has already evicted.
- **Class means** (per seed × checkpoint × γ), pooled over rows (row-weighted):
  - `m_kind(k) = mean G_γ[t][i]` over rows with `kind(i) = k`;
  - `m_kb(k, b) = mean G_γ[t][i]` over rows with `kind(i) = k` and `t − i ∈ b`;
  - `m_age(a) = mean G_γ[t][i]` over rows with `t − i = a`.
  - **An empty `(kind, band)` cell** falls back to `m_kind(k)`; the number of empty cells and of
    decisions that used a fallback is ledgered.

## 4. The rules (outputs i–iii)

Each is an eviction rule scored by **model-free residency** through
`rsr.metrics.headroom.simulate` (the instrument of every `U.hit.*` key in lookahead-room-r2).
At a full-memory step `t`, every live slot `k` (holding sentence `i = written_at[k]`) is scored,
and the victim is the argmin.

| Rule | Score of slot holding `i` at step `t` | Legal ψ̂? |
|---|---|---|
| **(i) kind-oracle** `ko` | `m_kind(kind(i))` | kind is a label ψ̂ never sees; the rule itself has no age |
| **(i′) kind-oracle, oldest tie** `ko_oldest` (secondary) | as (i) | as (i) |
| **(ii) kind × age-band** `kb` | `m_kb(kind(i), band(t − i))` | **NOT a legal ψ̂** (it reads age) — a reference only |
| **(iii) age-only** `age` | `m_age(t − i)` | an age rule by definition |

- **Tie rule, (i), (ii), (iii): exactly B2's A1.10.** Ties within the lowest score are broken
  **uniformly at random** with `random.Random(f"ko:{seed}:{doc_id}:{t}")`, choosing among the
  tied slot indices in ascending slot order (`rng.choice(sorted(tied))`). A fresh generator per
  (seed, document, step). The same string is used for all three rules (B2 A1.10 defines one).
- **(i′)** breaks ties to the lowest slot index (the oldest), as `OraclePolicy`; reported as a
  secondary, as B2 A1.10 does.
- **Reference rules**, recomputed here for the cap and for control C3: **FIFO**
  (`FIFOPolicy`), **factfiller** (lookahead-room's `FactFiller`, `random.Random(f"ff:{seed}:{doc_id}")`),
  **oracle** (`OraclePolicy(discounted_demand(doc, 0.97))`, lookahead-room's), and the
  **hindsight rule** `rule_g09` (lookahead-room's `TargetPolicy` on `G_0.9` from `D.pt`), and
  `rule_g0` (`TargetPolicy` on `D`).

## 5. Outputs (per seed × checkpoint × γ ∈ {0.9, 0})

1. **Hit rate** of every rule: queries whose assert is resident at the query step, pooled over
   U (sum hits / sum queries). Also on `gap_2_to_M` (`2 ≤ gap ≤ M`) and `gap_gt_M`.
2. **Cap** of every non-reference rule (ko, ko_oldest, kb, age, rule_g09, rule_g0, oracle):
   `cap = (hit − hit_FIFO) / (hit_factfiller − hit_FIFO)`, all-query.
3. **Per-document bootstrap CI** (2000 resamples, percentile 95%, paired: one resampled document
   multiset per replicate shared by every rule, checkpoint and γ of a seed;
   `torch.Generator().manual_seed(20260927 + seed)`): for `hit − hit_FIFO` and for `cap`
   (a replicate whose denominator is ≤ 0 is dropped from the cap's CI and counted).
4. **F1 table: demand by age × class.** Mean `D[t][i]` and row count over rows, per age
   `a ∈ {1, …, 47}` × class {pending, querying, answered, filler}, plus the pooled
   `age ≤ M` (FIFO-resident) and `age > M` (probe at rank 0) rows. (`D` does not depend on γ;
   reported once per seed × checkpoint.)
5. **F2 table.** `E[G_γ | kind]` for kind ∈ {assert, query, filler} and `E[G_γ | class]` for the
   four classes, over rows; and **`E[G_γ | assert] − E[G_γ | filler]`** with its per-document
   bootstrap CI (per-document sums, ratio of pooled sums per replicate). Also the class-mean
   tables `m_kb` and `m_age` that rules (ii) and (iii) use.

Headline rows in RESULTS.md: ckpt3000 × γ = 0.9, per seed, `hit` and `cap` for ko, kb, age.
Every other combination is reported beside it. Seed is the unit; nothing is averaged into a
single claim, and the ledger carries all three per-seed values with their sd.

## 6. Controls (any failure → exit 3, and no numbers are reported as results)

- **C1 manifest.** The sha256 of each of the six `D.pt` files equals its line in
  `~/rsr-substrate/2026-09-27/MANIFEST.sha256`, checked **before the first read and again after
  the last computation**. A missing file, a missing manifest line or a mismatch is exit 3.
- **C2 shape.** Each `D.pt` holds exactly the 1088 U doc ids of its seed, each `[48, 48]`,
  finite at every `i < t` and NaN at every `i ≥ t`.
- **C3 reproduction.** The recomputed all-query U hit rates of `fifo`, `factfiller`, `oracle`,
  `rule_g0` and `rule_g09`, and their `gap_gt_M` rates, equal `runs/lookahead-room-r2/ledger.json`
  `B.ckpt{c}.U.hit.<rule>[.gap_gt_M]` sample `seed` **exactly** (`==` on the float). This is what
  ties this run's reading of `D.pt` and of the documents to the instrument that wrote them.
- **C4 seeds and sizes.** Three seeds, two checkpoints, two γ, 1088 documents each were run
  (`seeds_actually_run` is written from what ran).

## 7. Exit codes (`rsr.exit_codes`)

| Code | Meaning |
|---|---|
| **0** | Ran; every control passed. Whatever the numbers are. |
| **1** | A defect after the controls passed: a NaN in a class mean or a hit rate, or a raise inside the computation. |
| **3** | Did not run, or a control (C1–C4) failed. |

Exit 2 and 5 are not used. `$?` is read directly.

## 8. Author's expectation (before any B0 number; allowed to be wrong)

- **Kind-oracle, ckpt3000, γ = 0.9.** From F2 as already seen (assert − filler +0.0633 /
  −0.0507 / +0.1601): on seeds 0 and 2 the lowest class is filler or query, so `ko` behaves like
  factfiller and its **cap is ≈ 0.8–1.1**; on **seed 1** asserts have the lowest mean, `ko` evicts
  asserts first, and its **cap is negative** (below FIFO, possibly far below). At γ = 0 the same
  sign pattern (F2 at γ = 0: +0.0084 / −0.0081 / +0.0205).
- Whether `query` has a lower mean than `filler` is not known to me; if it does, `ko` evicts
  query sentences first, which is harmless to residency.
- **Age-only.** F1's U-shape in age suggests `m_age` is not monotone, so `age` evicts from the
  middle of the age range. Expected cap between −0.5 and +0.5 on every seed, sign unknown.
- **Kind × age-band.** It can express F1's reversal at old ages. Expected to be ≥ `ko` on seed 1
  and ≈ `ko` on seeds 0 and 2.
- **C3** is expected to pass exactly (same code, same tensors, same documents).

## 9. Artefacts

- `experiments/b0-ceilings/run.py` (tests first: `tests/test_b0_ceilings.py`; ≥ 2 mutations in
  `scripts/mutation_battery.py`, each proven against the full suite).
- `runs/b0-ceilings/{manifest.json, ledger.json, claims.json}` — written by `scripts/ledger.py`.
- `experiments/b0-ceilings/RESULTS.md` — command, git SHA, hardware, every number, including any
  that came out wrong.
