---
run_id: b2-psi-probe
item: "PLAN-v4 §2B B2 (with its E0h section). Governing plan: ~/Documents/RSR-2026-09-27-plan/PLAN-v4.md"
question: >-
  Can the spec's context-conditioned value head psi_hat(s_i, c_t) = s_i^T W c_t + u^T [s_i ; c_t]
  (§3.2.2, §3.3), fitted with no age input to the Monte-Carlo return G (§3.4, correction 2),
  produce an eviction rule that beats the best age-based reference through the model? Q1: does
  it recover the kind prior? Q2: does it use c_t beyond kind?
class: MEASURE-THEN-DECIDE (PLAN-v4 §0.1). Every outcome is a RECOMMENDATION to Brendan. No outcome triggers a build.
substrate: "fresh-stream arm B, FROZEN, read-only. ckpt3000 PRIMARY; ckpt2500 reported, never gating."
seeds: [0, 1, 2]
gammas: {primary: 0.9, secondary: 0.0}
documents:
  FIT_TRAIN: "[920000, 936000) of each seed's generator; the first N_F ids are used (N_F = TBD-1)"
  FIT_VAL:   "[936000, 937024) of each seed's generator; all 1024 used"
  EVAL:      "[940000, 980000) of each seed's generator; the first N_E ids are used (N_E = TBD-3)"
  E0D_RESERVED_NOT_USED_HERE: "[900000, 920000)"
thresholds:
  delta: "0.25 x (acc_oracle(all) - acc_FIFO(all)), per seed, per gamma, on FIT_VAL, model-read"
  bootstrap: "2000 per-document resamples, paired across every arm, percentile 95% CI, torch.Generator seeded 20260927 + seed"
  E0h_proposed: "within-step R^2 >= 0.90 on >= 2 of 3 seeds -> COLLINEAR; < 0.49 on all 3 -> NOT_COLLINEAR; else INTERMEDIATE. PROPOSED; not read until Brendan ratifies."
tbd_fields: [N_F / ridge n, capture cost, N_E]
---

# PREREG: B2, the ψ̂ probe (with E0h)

**Written 2026-09-27 by a model** (Claude Opus 5.5, an `rsr-researcher` session dispatched by the
2026-09-27 PM session under PLAN-v4). **Not written by Brendan.**

This file is **committed alone**, before three things: the B0 PREREG, any B0 computation and any
B2 code. It holds the **entire** B2 section: framing, substrate, ranges, probe policy, targets,
fit, arms, age controls, metrics, decision rule, truth table, Q2, the γ = 0 tests and E0h
(PLAN-v4 §2B "Commit order", step 1; REDTEAM-v3 edit 2).

**The fence.** After this commit, only three things may be added to this file, each in its own
commit that shows its `git diff` against this one:
1. **n**: the TBD fields `N_F` (and the ridge row counts it implies) and `N_E`;
2. **the capture cost**;
3. **an "already seen" addendum** (§15), listing what B0 showed.

**Nothing else in this file may change.** That covers every threshold, arm, range, target,
control and rule. A defect found later is reported as a defect; it is not fixed here. An edit
outside those three is itself a finding, and it voids the classification.

---

## 1. Framing

- **Q1 and Q2.**
  - **Q1 (the kind prior):** does ψ̂, with no age input, recover the prior that the kind of
    sentence (assert / query / filler) carries?
  - **Q2 (beyond kind):** does ψ̂ use `c_t` to do better than the kind prior? Q2 is where B2
    earns its cost (REDTEAM-v2 §3).
- **What this is, and what it is not.**
  - It is an **engineering precondition and kill-gate evidence** (PLAN-v4 §0.2).
  - It is **not evidence on the cognitive claim.** Kintsch & van Dijk leading-edge rediscovery
    is tested only in E7, which is unapproved.
  - **E1 is a kill gate, never a green light (D-9)**, and B2 is not E1.
- **Inference is asymmetric** (REDTEAM-v1 M-4).
  - The fit is offline, full-batch, closed form, and drawn from the FIFO world.
  - An offline **NOT LEARNABLE** is strong evidence that the target is the problem.
  - An offline **WIN** is **necessary, not sufficient**, for online MC at μP rates, where
    ADR-0009 §3 predicts underfit and the data are non-stationary.
- **Not precedent for ADR-0009.** The probe fixes these choices only to make this measurement
  possible. **None of them is precedent for ADR-0009**, and none of them decides an L-item:
  - **L2** (which `c_t`): the whole harness runs in eval mode under `no_grad`, so the regression
    input and the decision input are both the eval-forward `out.srep` of step `t`.
  - **L4** (the probe form): single insertion at write-order rank 0, i.e. L4(a)+(i), with a rank
    1–3 sensitivity arm.
  - **L9** (the reduction): the rows are unweighted, with no `w_t`.
  - **L3, L5, L6:** the U-arm resembles a shadow target (L3–L5) and the C-arm resembles slice 1's
    `shadow=off` target (L6(a)). Resembling them decides nothing about them.
- **Stop-grad; no gradient into the transformer.** Nothing is trained by gradient. The fit is a
  closed-form ridge on captured tensors, and the transformer and `W_sent` are read only.
- **Structurally absent:** age as a ψ̂ input, `b`, ν, and shadow (T3). **`RSRPolicy` is not
  used.** `RSRConfig.from_registry(..., t_warm=0, …)` is not called; it constructs with
  `b_enabled=True, shadow_enabled=True` (REDTEAM-v2 M-4).
- **Records nothing.** No `rsr.constants.record()` call. No frozen constant, corpus, `M`, spec
  or §15 is touched.
- **Reading order.** No B2-derived recommendation goes to Brendan before E0d is read. If E0d finds
  `r_i` to be a confound (LOO is truth, §3.2.1), B2's report says that B2 describes the wrong
  target. It says so beside the classification; it does not replace it.

## 2. Substrate

- **Checkpoints:** fresh-stream arm B, `B/seed{0,1,2}/ckpt-003000.pt` (**primary**) and
  `ckpt-002500.pt` (**reported, never gating**).
  - The source is `.worktrees/fresh-stream/runs/fresh-stream/`, or the T0 copy
    `~/rsr-substrate/2026-09-27/` by its restore procedure.
  - They are **frozen**: loaded with `restore_rng=False`, eval mode, `no_grad`, and never written.
- **sha256 of every checkpoint.** These are the pins that lookahead-room pinned, and a mismatch is
  exit 3:
  - **2500:** `26c4162e7dc6ef1f80dffa451e4ee75c56360bc94227533b38a45b62003e3176`,
    `327c3facde61ef9a7f4b9fcaeb31efcf40ab36d65a5d584ec97637db334bdadd`,
    `74b026f1dd3fe59eb473e501c37ed9bb7d836276270ec875939e5654d8e86ddd`
  - **3000:** `0ee3f8a69b507d927c631eb85116e1cd9739ee4472c44eb7f145d739a118da60`,
    `dadd1e08a3849c1211c8479394df4060f91cd2a6e188b97701b783f2068a3da8`,
    `b507ebc573a54316f9d3c20337f49e664bb9c414388af925c5a8e87efd6b63b8`
- **T0 check.** T0's substrate manifest (`~/rsr-substrate/2026-09-27/MANIFEST.sha256`) is
  re-verified **when the run starts and when it finishes**. If T0 is incomplete or a check
  fails, the result is exit 3.
- **Model.** S0-03's `_cfg(V)`, with `V` and the token map from `doc_sets(seed, 64)` (the `[0, 64)`
  vocabulary).
  - `M = 16` and `S = 48` are read from the checkpoint's config and asserted. They are not typed.
  - `d` is read from `runs/fresh-stream/manifest.json` (key `d`; 128 at this writing) and
    asserted equal to the checkpoint's `D`.
- **Seeds:** 3 (0, 1, 2). **The unit of analysis is the seed.** Per-document CIs describe
  within-seed variance only.
- **Device and batching.** CPU. **B = 1 per document**, with a **fresh policy per document**,
  through the real `rsr.model.tg.policy_loop.run_policy_loop`, as in W6 and W10. The wrapper
  asserts that the document id it was built for is the id being run.
- **No overlap with B5** (PLAN-v4 §2B B5; REDTEAM-v3 N-7). B2's capture, fit and runs do not
  overlap B5's 12-thread job.

## 3. Document ranges (T4)

Every range is **fresh**: no document in it has been generated, run or inspected by any
experiment. The ranges are **mutually disjoint**. Ids follow the seed's generator,
`_generate_document(doc_id, cfg)` with generator seed `seed · 1_000_003 + doc_id`
(`synthetic.py:237`).

| Name | Ids (per seed) | Size | Use |
|---|---|---|---|
| `FIT_TRAIN` | `[920000, 936000)` | 16,000 reserved; the **first `N_F`** are used (`N_F` is TBD-1) | the ridge fits, the kind-oracle's class means, and the rank-sensitivity fits |
| `FIT_VAL` | `[936000, 937024)` | 1024, all used | λ; `ref`; δ; the power calculation; the age-decodability R² |
| `EVAL` | `[940000, 980000)` | 40,000 reserved; the **first `N_E`** are used (`N_E` is TBD-3, from §9.4) | every arm's in-loop evaluation; E0h |
| *(E0d, reserved, not used by B2)* | `[900000, 920000)` | 20,000 | set aside so that E0d, whose PREREG does not exist yet, can take a fresh range with no negotiation |

**Ranges already used, all of which B2 is asserted disjoint from** (exit 3 otherwise, checked
before any model is loaded):

| Range | Used by |
|---|---|
| `[0, 64)` | probe / vocabulary / S0-03 training; corpus-size n64; arm B steps 0–999 |
| `[64, 128)` | S0-03's held-out |
| `[0, 512)`, `[0, 4096)` | corpus-size-curve arms N = 512 and N = 4096 |
| `[64, 1088)` | E / EXT: retention-readability, lookahead-room, carry-forward (in four chunks) |
| `[4096, 4160)` | P / H64: fresh-stream's held-out, used by nearly everything |
| `U = [64, 1088) ∪ [4096, 4160)` | W6, W10; B0's inputs |
| `[4160, 52160)` | fresh-stream's stream, steps 0–2999 (including scaffold-dose `[4160, 21760)` and scaffold-timing `[4160, 5760)`) |
| `[52160, 148160)` | fresh-escape's stream, steps 3000–8999 |
| `[4160 + 16·3000, 4160 + 16·t_cap)` | B5's continuation of the stream, for any cap `t_cap` up to step 55,000 (stream id 884,160). B2's lowest id, 900,000, clears that. |
| `experiments/e0d/PREREG.md` | if that file exists on the run's branch, every range it declares |

**Two further assertions, each exit 3 if it fails:**
- **No cross-seed aliasing.** A generator seed is `s · 1_000_003 + id`. B2's generator seeds, for
  `s ∈ {0, 1, 2}`, must be disjoint from the generator seeds of every used range above, for every
  `s' ∈ {0, 1, 2}`. All of B2's ids are below 1,000,003, so this holds by construction; it is
  still asserted.
- **Vocabulary closure.** Every word of every B2 document is in the seed's `[0, 64)` map
  (`check_vocabulary_closure`).

**Coordination with E0d.**
- At this writing, no branch `run/e0d` exists (checked locally and on `origin`), and
  `experiments/e0d/` holds only its `NOT RUN` stub.
- `[900000, 920000)` is therefore reserved for E0d and **left unused by B2**.
- **B2 claims its ranges by this commit.** Any later PREREG, E0d's included, must avoid them.

## 4. The probe policy: `ProbeArgminPolicy`

- **Where it lives.** `ProbeArgminPolicy` is local to `experiments/b2-psi-probe/`. It implements
  the `RetentionPolicy` protocol (`select_eviction`, `observe`, `on_write`, `reset`).
- **What it is not.** It is **not `RSRPolicy`**, it imports nothing from `rsr.retention.rsr`, and
  it **sets no precedent for ADR-0009**.
- **Its only parameters** are a fitted weight vector `w` (fp64) and a feature map `f`. The
  feature map is either:
  - the bilinear map, `f(s_i, c_t) = [vec(s_i c_tᵀ); s_i; c_t]`; or
  - for the age-only arm, a one-hot age vector.
- **The rule.** At a full-memory step `t`, it scores every live slot `k` with
  `ψ̂_k = f(s_k, c_t)·w`, where:
  - `s_k` is `MemoryState.gestalts[k]`;
  - `c_t` is the `context` argument, i.e. `out.srep[row].detach()`.

  It returns `argmin_k ψ̂_k`, with ties going to the lowest slot index (the oldest), as in
  `OraclePolicy`.
- **What is structurally absent:**
  - age as a ψ̂ input (the bilinear arms);
  - `b`, ν, shadow, warmup;
  - any `observe` state (its `observe` is a no-op);
  - any RNG.

  The feature builder's signature takes `(s_i, c_t)` and nothing else. A test asserts this
  before the run.
- **The per-eviction log.** At each eviction, it logs:
  - document id, step `t`, model seed, arm;
  - the victim's slot, its `written_at`, its **age** `t − written_at`, and its **rank**;
  - the victim's kind, and for an assert, whether it is **pending**, **querying** or
    **answered** (a descriptive label from the document, for attribution only; never an input);
  - the **ψ̂ margin**: the second-lowest ψ̂ minus the lowest ψ̂ over live slots;
  - the full live ψ̂ vector;
  - the **ADR-0006 rank shift**, `(victim_rank, displacement)` from the pure function
    `rsr.retention.instrumentation.rank_shift(slots, victim)`.

  This log is the **attribution source for truth-table row 1** (HARMFUL).
- **The other arms** are logged with the same fields: victim, age, rank and rank shift, with no
  ψ̂ where the arm has none. This log is the source of the eviction-age histograms (§8).

## 5. Targets

**The capture: a fresh FIFO rollout of every `FIT_TRAIN` and `FIT_VAL` document**, per
(checkpoint, seed). It uses W10's harness definitions (lookahead-room PREREG, "Definitions"),
with `D.pt`'s construction, recomputed; **`D.pt` holds no gestalts and is not reused**.

- **What is captured, under `no_grad` and in eval mode.** These are stop-grad by construction:
  - `s_i`: the gestalt the FIFO rollout wrote for sentence `i` (`srep_mem`);
  - `c_t`: step `t`'s `out.srep`.
- **`r_i(t)`.** `retrieval_demand(gated=True)`: norm-weighted, gated (correction 17),
  fill-level rescaled (`share · |memory_t| / M`, §3.2.1), from an eval-mode trace
  (correction 20). It is exactly as in E0e and W10.
- **Demand-if-resident, `D[i][t]`.**
  - For a sentence FIFO holds at `t`: FIFO's own `r_i(t)`.
  - For a sentence FIFO has already evicted: a **probe**, in the same step's forward on FIFO's
    memory. The probe **replaces the rank-0 (oldest) slot's gestalt with sentence `i`'s gestalt,
    alone**, and reads the probe's `r` at that slot.
  - **Why rank 0.** This is L4(a)+(i): a sentence FIFO evicted is older than every FIFO resident,
    so write-order rank 0 is the rank it would hold if it were the only extra slot a policy kept
    (REDTEAM-v1 §1a).
  - **Declared optimistic** for states in which a policy keeps several old slots, where the
    correct rank depends on the policy.
  - Probes never alter the FIFO rollout.
- **Returns.** `G_γ[i][t] = Σ_{k=0}^{S−1−t} γ^k X[i][t+k]`, for `γ ∈ {0.9, 0}`.
  - **γ = 0.9 is primary.** γ = 0 is secondary, and it is **E0h's input**.
  - **Declared:** G is **truncated at document end** (`discounted_returns`, within the stream).
    At γ = 0.9, this matters for about the last 10 sentences.
  - **Declared:** the sum starts at `k = 0` (§3.4; correction 2; W10), so `G_γ[i][t]` includes
    step `t`'s own demand. At γ = 0, the target is `X[i][t]` alone.

**The target arms.**

| Arm | `X` | Rows `(i, t)` | Declared |
|---|---|---|---|
| **(U) uncensored** | `D` (demand-if-resident) | every `t ∈ [1, S)` and every sentence `i < t` | optimistic for multi-kept states (above) |
| **(C) censored** | the FIFO rollout's literal `r` (0 once FIFO has evicted `i`) | every `t ∈ [1, S)` and every FIFO-resident `i < t` | see below |
| **(U-r) rank sensitivity**, r ∈ {1, 2, 3} | `D` with the probe placed at rank `r` instead of rank 0 (for evicted `i`); resident `i` is unchanged | as U | **reported, never gating** |

- **Declared for (C).** This target is what ADR-0009 slice 1 (L6(a), `shadow=off`) regresses on
  **only during FIFO warmup**. After warmup, slice 1's censoring is by its own policy
  (on-policy), which this offline arm does not reproduce (REDTEAM-v2 m-6).
- **Row set.** Both arms include the under-full steps `t < M`, where no eviction is decided but
  slice 1 would still train.
  - On those steps, `r` is the rescaled share.
  - The rows are unweighted (not precedent for L9).
  - Per document, that gives 1128 U rows and 632 C rows.

## 6. The fit

- **Form.** The spec's bilinear head (§3.2.2 :227):
  `ψ̂ = s_iᵀ W c_t + uᵀ[s_i; c_t]`.
  - That is **`p = d² + 2d` features**: 16,640 at `d = 128`.
  - There is **no separate intercept** (the spec form has none).
  - `d` comes from the manifest and is asserted; `p` is computed, never typed.
- **Method.** Closed-form ridge in **fp64**: `w = (XᵀX + λ' I)⁻¹ Xᵀy`.
  - `XᵀX` (about 2.2 GB) and `Xᵀy` are accumulated over rows.
  - The solve uses a Cholesky, or an eigendecomposition if the Cholesky fails.
  - The solve residual `‖(XᵀX + λ'I)w − Xᵀy‖ / ‖Xᵀy‖` must be ≤ 1e-8. Otherwise the result is
    exit 1.
- **Inputs.** Stop-grad `s_i` and `c_t` from the fresh capture (§5). **No age, no rank, no
  kind label, no position.**
- **λ.** `λ' = λ · tr(XᵀX)/p`, for `λ ∈ {10⁻⁶, 10⁻⁵, …, 10⁴}`.
  - λ is chosen per (checkpoint, seed, arm, γ) as the minimum of the **validation MSE on
    `FIT_VAL`'s rows of the same target**.
  - Ties go to the larger λ.
  - The final `w` is refit on `FIT_TRAIN` alone at the chosen λ; `FIT_VAL` is never folded in.
- **Fits.** One fit for each combination of:
  - arm: U, C, U-1, U-2, U-3;
  - γ: 0.9, 0 (the U-r arms at γ = 0.9 only);
  - checkpoint: 3000, 2500;
  - seed: 0, 1, 2.
- **The age-only head, fitted the same way.**
  - The features are a **one-hot of age** `t − i ∈ {1, …, S−1}`. That is the most flexible
    age-only function, and it is still a function of age alone.
  - The rows, targets, ridge, λ grid, λ rule and precision are all identical to the bilinear fit.
  - There is one age-only head per (arm U/C, γ, checkpoint, seed).
- **The age-decodability predictor.** The same bilinear features and the same ridge procedure,
  but the target is **age** `t − i`, on the U rows and on the C rows. Its R² is reported on
  `FIT_VAL` (§8).
- **The kind-oracle (Q2's comparator), re-estimated here and never taken from B0.**
  - **Class means.** `m_γ(kind) = mean of G_U,γ[i][t]` over `FIT_TRAIN` U rows at full-memory
    steps, grouped by the generator's kind of sentence `i` (assert / query / filler).
  - **Policy.** Evict the live slot with the smallest `m_γ(kind(i))`, with ties going to the
    oldest.
  - It is estimated per (checkpoint, seed, γ).
  - It uses the kind label, which ψ̂ never sees; that is the point of the comparator.
- **TBD, filled only after the build, by diff** (the fence, above):
  - **TBD-1** `N_F`: the number of `FIT_TRAIN` documents used. The ridge row counts are
    `n_U = 1128·N_F` and `n_C = 632·N_F`. The build sets `N_F` so that `n_C ≥ 10·p` at minimum.
  - **TBD-2:** the measured capture cost (wall time and peak memory, per checkpoint and seed),
    and the Gram-accumulation and solve cost.

## 7. Arms (on `EVAL`, in the loop, one document at a time, model-read)

For every (checkpoint, seed, γ):

| Arm | Definition |
|---|---|
| **ψ̂-U** | `ProbeArgminPolicy` with the U fit |
| **ψ̂-C** | `ProbeArgminPolicy` with the C fit |
| **ψ̂-U-r**, r = 1, 2, 3 | `ProbeArgminPolicy` with the U-r fits (γ = 0.9); **reported, not gating** |
| **FIFO** | `rsr.baselines.fifo.FIFOPolicy` |
| **age-only (U), age-only (C)** | `ProbeArgminPolicy` with the one-hot-age fits of §6. It reads age from `MemoryState` (it is an age arm by definition) |
| **random ×5** | `rsr.baselines.random_policy.RandomPolicy`, uniform over **all** live slots (the spec's E1 arm). A `torch.Generator` per (random seed `k ∈ {0,…,4}`, model seed, document) is seeded from the first 8 bytes of `sha256(f"b2rand:{k}:{seed}:{doc_id}")` |
| **fact/filler** | evict uniformly at random among live non-assert slots; if every live slot is an assert, uniformly among all live slots. `random.Random(f"ff:{seed}:{doc_id}")`, fresh per document, exactly as in W6 |
| **oracle** | `rsr.baselines.oracle.OraclePolicy(discounted_demand(doc, γ))` (`src/rsr/baselines/oracle.py:36`), one fresh policy per document, **the same γ as the arm**. The wrapper asserts that its demand matrix equals `discounted_demand(doc, γ)` recomputed from the document being run |
| **kind-oracle** | §6, the U class means at this γ |

- **Declared.** For γ ∈ (0, 1), `OraclePolicy` is Belady's MIN (its docstring). Its decisions at
  γ = 0.9 are therefore identical to W6's and W10's oracle at γ = 0.97.
- **At γ = 0,** the "oracle" is only a one-step-ahead rule, not Belady's MIN. Its headroom, and
  with it the γ = 0 δ, may be small. That is accepted: γ = 0 is secondary.

## 8. Age controls (reported; they are not gates)

1. **Age decodability:** the R² on `FIT_VAL` of the bilinear age predictor (§6), for U rows and
   C rows, per (checkpoint, seed).
2. **corr(ψ̂, age):** Pearson and Spearman, over live slots at full-memory steps.
   - They are computed for every ψ̂ arm, both on `FIT_VAL`'s capture rows and in the arm's own
     in-loop `EVAL` rollout.
   - The same statistics are reported for the age-only head, beside them.
3. **The eviction-age histogram:** the victim's age (1 to S−1), per arm, per (checkpoint, seed,
   γ), from the per-eviction logs. It is reported beside FIFO (identically `M`) and the age-only
   head.

## 9. Metrics and the decision rule

### 9.1 Metrics (S0-03's definitions, per answer target, model-read)

- **`correct`:** the argmax over the full vocabulary equals the answer token (S0-03's
  `answer_acc`).
- **Primary: all-query answer accuracy (model-read).** The sum of correct answers over the sum
  of queries, pooled over the documents of the set.
- **The non-inferiority set:** the S0-03 bucket `gap_2_to_M` (`2 ≤ gap ≤ M`). This is
  "within gap M".
- **Secondary:**
  - accuracy on `gap_gt_M`;
  - **residency**: whether the queried assert is live in the memory the query step reads. It
    must equal a model-free `rsr.metrics.headroom.simulate` replay of the same victim sequence
    (a control).
- **Also reported:** `gap_eq_M` accuracy; NLL and Brier16 (S0-03's definitions) for every
  bucket; the ψ̂-margin distribution; the rank-shift distribution.

### 9.2 Bootstrap

- 2000 per-document resamples, **paired**: one resampled multiset of documents per replicate,
  and every arm, bucket and contrast computed on it.
- Percentile 95% CI.
- `torch.Generator` seeded `20260927 + seed`.
- Accuracy is pooled over the answers of the resampled documents.
- **The random arm's accuracy** in each replicate is the mean, over its 5 random seeds, of their
  pooled accuracies on that replicate's documents.

### 9.3 Reference and minimum effect (on `FIT_VAL`, never on `EVAL`)

- **`ref`,** per seed, per (arm U/C), per γ: whichever of **FIFO** and **that arm's age-only
  head** has the higher all-query accuracy on `FIT_VAL`. On a tie, FIFO.
- **`δ`,** per seed and per γ: `δ = 0.25 × (acc_oracle(all) − acc_FIFO(all))` on `FIT_VAL`, with
  the oracle at that γ, read through the model.
  - For scale, the U-range numbers already seen put this headroom at about 0.14, i.e. δ ≈ 0.035
    at γ = 0.9. That is context only; **the formula is fixed, and its value comes from
    `FIT_VAL`.**
  - **If `δ ≤ 0`** for a seed, that seed's contrasts at that γ are **UNRESOLVED (δ undefined)**
    and are logged as such.

### 9.4 Evaluation size (power; the procedure is fixed now, and `N_E` is TBD-3)

1. On `FIT_VAL` (`N_V = 1024`), per seed at ckpt3000 and γ = 0.9, take each gating contrast `c`:
   - ψ̂-U − ref_U and ψ̂-C − ref_C (all-query);
   - the same two on `gap_2_to_M`;
   - ψ̂-U − random-mean and ψ̂-C − random-mean;
   - ψ̂-U − kind-oracle.
2. For each, take its paired-bootstrap SD, `σ_{c,s}`.
3. The required size is `N_{c,s} = ⌈N_V · (1.96 · σ_{c,s} / (δ_s / 2))²⌉`, so that the expected
   95% CI half-width is ≤ δ/2.
4. `N_E = max_{c,s} N_{c,s}`, clipped to `[1024, 40000]`.
5. **If the cap binds,** `N_E = 40000`, and the run is labelled **UNDERPOWERED** in the report.
   **The rule below is read unchanged.**
6. `N_E` is written into this file by diff (TBD-3) **before any `EVAL` document is run.** The
   same `N_E` serves ckpt2500 and γ = 0.

### 9.5 Per-seed outcome (per arm ψ̂-U, ψ̂-C; per γ; per checkpoint)

**Δ** = all-query accuracy(ψ̂) − all-query accuracy(ref), on `EVAL`, with its paired 95% CI
`[lo, hi]`.

| Outcome | Condition |
|---|---|
| **WIN** | **all four** hold: <br>(1) `lo > 0`; <br>(2) `Δ ≥ δ`; <br>(3) **non-inferiority:** the paired 95% CI lower bound of the `gap_2_to_M` Δ (ψ̂ − ref) is `> −δ`; <br>(4) **beats random:** all-query accuracy(ψ̂) − the random arm's 5-seed mean has a paired CI lower bound `> 0`. |
| **EQUIV** | `−δ < lo` and `hi < δ` |
| **LOSS** | `hi < −δ` |
| **UNRESOLVED** | anything else, including δ undefined |

- The outcomes are evaluated in the order WIN, then LOSS, then EQUIV, then UNRESOLVED.
- WIN and LOSS cannot both hold.

### 9.6 Classification: the 6-row precedence truth table (ckpt3000, γ = 0.9)

The **first matching row wins.** The columns count, out of 3 seeds, the γ = 0.9 seeds with each
outcome.

| Precedence | ψ̂-U (γ = 0.9) | ψ̂-C (γ = 0.9) | Class → recommendation to Brendan |
|---|---|---|---|
| 1 | any seed LOSS | **or** any seed LOSS | **HARMFUL.** Report which arm, which seed, and the **attribution**, taken from `ProbeArgminPolicy`'s per-eviction log: the victim age/rank distribution, the ψ̂-margin distribution, the rank shift, and the kind/pending labels of the victims, against FIFO's. **No build recommendation.** |
| 2 | 3 WIN | 3 WIN | **NOT RULED OUT, CENSORED.** Slice-1 (L6(a)) learnability is not ruled out offline. This is **necessary, not sufficient**. |
| 3 | 3 WIN | ≤ 2 WIN | **NOT RULED OUT, UNCENSORED ONLY.** The shadow buffer (L3–L5) is on the critical path; revisit L6. |
| 4 | ≤ 2 WIN | 3 WIN | **ANOMALY (C > U).** Report it, and audit the probe instrument before any reading. |
| 5 | 3 EQUIV | 3 EQUIV | **NOT LEARNABLE, offline.** The target is the problem. Strong evidence, under the asymmetry. |
| 6 | otherwise | otherwise | **MIXED / UNRESOLVED.** A per-seed report. Seed variance or low power is the finding, and **no NOT-LEARNABLE claim is made.** |

- **ckpt2500:** the same outcomes and table are computed and **reported, never gating.** If the
  two checkpoints classify differently, both are reported and the result is flagged **moving**
  (F14: arm B has not converged).
- **The rank-sensitivity arms (U-1, U-2, U-3):** their per-seed outcomes against the same ref and
  δ are reported, **never gating**, beside ψ̂-U (rank 0).

### 9.7 Q2: beyond kind

- **Δ_Q2** = all-query accuracy(ψ̂-U) − all-query accuracy(kind-oracle), on `EVAL`, per seed,
  γ = 0.9, ckpt3000, with the paired 95% CI.
- The kind-oracle is **re-estimated on `FIT_TRAIN`** (§6). **It is never taken from B0's U-range
  numbers.**
- **The outcomes:**
  - **Q2-WIN** (ψ̂ exceeds the kind prior): the CI lower bound `> 0` **and** `Δ_Q2 ≥ δ`.
  - **Q2-EQUIV:** the CI lies inside `(−δ, δ)`.
  - **Q2-LOSS:** the CI upper bound `< −δ`.
  - **Q2-UNRESOLVED:** anything else.
- These are reported per seed. **Q2 is outside the truth table** and does not change the
  classification.
- ψ̂-C versus the kind-oracle is reported the same way, as a secondary.

### 9.8 γ = 0 (secondary)

- **What is computed.** Every test of §9.5–9.7, with the γ = 0 fits, the γ = 0 oracle and δ, and
  the same ref rule and `N_E`.
- **How it is reported.** Its truth-table class is reported, labelled **secondary**. It never
  overrides the γ = 0.9 class.
- **What it feeds.** Its capture and fits are **E0h's input** (§10).
- **What it does not test.** It is not falsifier 3b: that is RSR(γ = 0.9) against RSR(γ = 0)
  trained online, in E1.

## 10. E0h (falsifier 3c), pre-registered here as its own section

**The question** (spec §2 :139, falsifier 3c :149, kill-gate table :552): regress ψ̂(γ = 0) on the
model's current-step cross-attention logits. If the two are near-collinear, the estimator is
re-deriving the forward pass.

**🔴 This is an offline-ridge proxy, declared.** The ψ̂ here is B2's closed-form ridge fit, not
a head trained by `L_MC` in `train()`. An E0h result on the proxy says what the **target and
functional form** admit. It does not measure the trained head.

**The regression.**
- **Heads:** ψ̂-U(γ = 0) is the **headline**. ψ̂-C(γ = 0) is reported beside it.
  - Checkpoint: ckpt3000 is the headline; ckpt2500 is reported.
  - Seeds: all 3.
- **Rows:** every FIFO-resident slot `i` at every full-memory step `t`, in the **FIFO arm's
  rollout of `EVAL`**.
- **Regressors `x_{i,t}` ∈ ℝ^{6·H}:** for each of the six cross-attention layers
  (correction 18) and each head, the **pre-softmax** cross-attention logit from sentence `t`'s
  query tokens to slot `i`.
  - They are collapsed over query tokens by the same rule as `α` (`Q_TOK_COLLAPSE`, ADR-0008).
  - They are **recomputed in the harness** from the block's inputs, as `cross_capture`
    recomputes `v`, with **no change to the forward pass**.
  - A control asserts that the recomputed logits, softmaxed, reproduce the captured `α` to
    ≤ 1e-5.
  - **If the build cannot produce exact logits, E0h is exit 3 (DID NOT RUN).** No substitute
    (for example `log α`) is silently used.
- **Statistics:** OLS in fp64, in-sample (with `6H + 1 ≪ n`).
  - **Primary:** the **within-step R²**. Both ψ̂ and each logit feature are demeaned within each
    (document, t) group, and the regression has no intercept.
    - **Why within-step:** a softmax logit is defined only up to a per-query shift, and eviction
      is an argmin within a step. A per-step offset is invisible to both.
  - **Secondary:** the **pooled R²** (raw values, with an intercept).
  - **Reference, never gating:** the within-step R² of the **γ = 0 target itself** (`D[i][t]`)
    on the same logits.
    - **Why:** at γ = 0 the target includes step `t`'s own demand (§5), and `r_i(t)` is a
      function of step `t`'s attention.
    - **How it is read:** if the target is itself near-collinear with the logits, a collinear ψ̂
      is inherited from the target's definition. The report must say so. **It does not change
      the rule below.**

**The threshold.** The spec gives no number: it says only "Not collinear" (:552). **This is a
PROPOSAL. No kill is read until Brendan ratifies it**, and the E0h numbers are reported
unclassified until then.

| Proposed class | Condition (within-step R², ψ̂-U(γ = 0), ckpt3000) |
|---|---|
| **COLLINEAR**, the falsifier-3c kill signal | R² ≥ **0.90** on ≥ 2 of 3 seeds |
| **NOT COLLINEAR** | R² < **0.49** on all 3 seeds |
| **INTERMEDIATE** | otherwise |
| **UNINFORMATIVE** (per seed; that seed counts toward neither of the first two) | ψ̂-U(γ = 0)'s own validation R² on its target is ≤ 0 (the head learned nothing, so collinearity is meaningless), or ψ̂'s within-step variance is 0 in fp64 |

**The rationale for the proposed numbers.**
- **0.90:** the conventional near-collinearity line. It is a variance-inflation factor of 10,
  i.e. `R² = 1 − 1/VIF`. Below it, more than 10% of ψ̂'s within-step variance is something the
  current logits do not linearly carry.
- **0.49:** `|r| = 0.7`. That is the **spec's own line** for "the eviction score is not a
  function of X", in the E2 vacuity gate (§7.1; kill table :558, "Partial ρ < 0.7"). This
  reuses a number the spec already committed to, rather than inventing one.
- **The asymmetry is deliberate.** A kill needs a majority of seeds (2 of 3). "Not collinear"
  needs every seed.
- **The multivariate regressor set (6·H logits) is the conservative choice for this claim.**
  More regressors can only raise R², so a NOT COLLINEAR reading is hard to earn.

## 11. Controls (any failure → exit 3, and no classification)

1. **Checkpoint sha256.** Every checkpoint matches §2, and T0's manifest re-verifies at start and
   at end.
2. **Ranges.** The ranges are disjoint and the vocabulary is closed (§3), checked before any
   model is loaded.
3. **The harness's FIFO is exact.** On P = `[4096, 4160)`, used as a control only, the harness's
   B = 1 FIFO rollout reproduces `runs/fresh-stream/ledger.json`
   `B.ckpt{c}.heldout.live.<bucket>.answer_acc` for every S0-03 bucket, under W10 Amendment 1's
   rule:
   - argmax identity against `answer_readout(cond="live")`, with zero mismatches;
   - float32 accuracy exactly equal (`==`).
4. **The identity probe.** At every full-memory step of the capture, a probe that re-inserts
   FIFO's own rank-0 sentence reproduces FIFO's `r` at slot 0 to ≤ 1e-6 (W10's C4). The same
   holds at ranks 1–3 for the U-r probes.
5. **`Σ r_i` at full memory.** It is within 1e-5 of 1 on every full-memory step: in the FIFO
   rollout, in every probe row, and in every ψ̂ arm's rollout (E0e's `SUM_TOL`).
6. **One document, its own policy.** The per-document id assertion holds for every arm. The
   oracle's demand matches the document. `ProbeArgminPolicy` is asked only at full-memory steps
   and returns a live slot.
7. **In-loop residency equals the model-free replay** of the logged victim sequence, for every
   arm and document.
8. **Determinism.** Re-running the first 8 `FIT_VAL` documents for every arm reproduces every
   victim and every `correct` flag.
9. **No age in ψ̂.** A test shows that the bilinear feature builder's output is unchanged when
   `written_at` and step indices are permuted with the gestalts held fixed.
10. **The ridge solve.** Its residual is ≤ 1e-8 (§6). A failure here is exit **1**, not 3: the
    measurement itself is defective.

## 12. Exit codes (`src/rsr/exit_codes.py`)

| Code | Meaning |
|---|---|
| **0** | Ran, and every control passed. A classification was reached. **This includes HARMFUL, NOT LEARNABLE, ANOMALY, MIXED/UNRESOLVED and UNDERPOWERED.** A scientific outcome is never a non-zero exit. |
| **1** | Failed: a real defect in the measurement after its preconditions passed. That covers a failed ridge-residual check, a NaN or inf in ψ̂ or in a target, or a raise inside a measurement. |
| **3** | Did not run, or a control failed: a checkpoint missing or mismatched, T0 incomplete, a range overlap, vocabulary closure, any control in §11 (1–9), or E0h's exact logits unavailable (that one affects E0h only; B2 may still exit 0 with E0h marked DID NOT RUN). |

- **Exit 2 and exit 5 are not used.**
- Read `$?` directly (`cmd > log 2>&1; rc=$?`).

## 13. Author's expectation (before any B2 number; allowed to be wrong)

- **Given F2** (seed 1's reversed E[G | assert] − E[G | filler] at γ = 0.9), a content head that
  fits class means would evict asserts first on seed 1.
  - Expected: **ψ̂-U LOSS or UNRESOLVED on seed 1**, and so **HARMFUL (row 1) or MIXED (row 6)**
    as the most likely classes.
  - Rough odds: HARMFUL about 40%; MIXED about 40%; NOT LEARNABLE about 10%; NOT RULED OUT (rows
    2 and 3) about 10% together.
- **ψ̂-C:** closer to FIFO than ψ̂-U. The censored G is strongly age-shaped (REDTEAM-v1 B-2), and
  ψ̂-C should lean on whatever age the gestalts carry.
- **Age decodability:** R² is expected to be substantial. That is a guess, not a finding.
- **Q2-WIN:** unlikely on any seed.
- **E0h:** within-step R² for ψ̂-U(γ = 0) above 0.49 is likely, because the γ = 0 target includes
  step t's own attention readout. Whether it reaches 0.90 is genuinely open.

## 14. What this does not establish

- Nothing about Kintsch & van Dijk, E7 or the cognitive claim.
- Nothing about online MC training, μP rates, β, `b` or ν.
- It is not an E1 result, and not a D2 substrate ruling.
- It uses one corpus (S0-03), `M = 16`, one width, and FIFO-trained checkpoints. **No scaling
  claim is made.**
- A WIN says only that the target and functional form admit an eviction rule that beats `ref`
  through this model, offline. It says nothing about whether training would find it.

## 15. "Already seen" declaration

The author's team (the PM session, the red-team sessions and this author) **has seen**, before
this commit:

- **The 09-26 lookahead-room ledger:** all 613 keys of `runs/lookahead-room-r2/ledger.json` and
  run 1's `runs/lookahead-room/ledger.json`, on U = `[64, 1088) ∪ [4096, 4160)` and
  P. In particular:
  - `room_3b` at ckpt3000: 0.0355 / 0.0420 / 0.0478 (every CI lower bound > 0); at ckpt2500:
    0.0205 / 0.0367 / 0.0215.
  - Hit rates: the literal (censored) hindsight rule `B.ckpt3000.U.hit.lit_g09` is 0.8080 /
    0.8104 / 0.8123, against FIFO's 0.8059 / 0.8093 / 0.8088. Every `U.hit.*` and `P.hit.*` rule
    (fifo, oracle, pending_fifo, factfiller, rule_g0/g09/g097/next, lit_*, online_g0) is
    declared seen.
  - **Class `D` means by age** (REDTEAM-v1 §1a, `rt/age.py`). Pending vs filler at age 1:
    0.0930/0.0580, 0.0999/0.0831, 0.1047/0.0615. At age 16: 0.0783/0.0928, 0.0508/0.0683,
    0.0758/0.0731. On the probe at rank 0: 0.0843/0.0962, 0.0544/0.0817, 0.0829/0.0757.
    **The U-shape in age, and the reversal from age 13 on seeds 0 and 1.**
- **The red-team scratch numbers F1/F2 (PLAN-v4 §1).** They have **no ledger** yet; B0 gives
  them one.
  - **F1:** the age-stratified finding above; for example, the seed-2 pending − filler gap is
    +0.0432 at age 1 and +0.0027 at age 16.
  - **F2:** E[G | assert] − E[G | filler] at γ = 0.9 is +0.0633 / **−0.0507** / +0.1601 at
    ckpt3000 and +0.0558 / **−0.0867** / +0.1352 at ckpt2500. At γ = 0 (ckpt3000) it is +0.0084
    / **−0.0081** / +0.0205. Pooled `G_0.9`: pending 0.6956 / 0.5285 / 0.7200, filler 0.5095
    / 0.4562 / 0.4260. The hindsight cap of a perfect realised-`G_0.9` rule is 0.18 / 0.13 /
    0.54 of the fact/filler gain.
  - Censored `G_0.9` by kind at ages 9–16 (assert/filler): 0.212/0.223, 0.168/0.191,
    0.232/0.209. At ages 1–8 it is ≈ 0.41–0.46 for asserts, against ≈ 0.17–0.23 at ages 9–16
    (REDTEAM-v1 B-2).
- **Related ledgers, declared seen.**
  - `runs/retention-readability/ledger.json` (F3). Rescued-fact accuracy: fact/filler 0.5969 /
    0.6267 / 0.6032 and oracle 0.8551 / 0.8411 / 0.8770. `H_model` 0.1512 / 0.1459 / 0.1244.
    `H_ff_model` 0.0691 / 0.0689 / 0.0441. The U-range all-query oracle − FIFO ≈ 0.14 quoted in
    §9.3 comes from here.
  - `runs/fresh-stream/ledger.json`. P, ckpt3000, FIFO live: all 0.7761 / 0.7824 / 0.7978;
    `gap_gt_M` 0.0591 / 0.0966 / 0.1549; with memory wiped, `gap_gt_M` 0.0591 / 0.0725 / 0.0752.
- **F-X1** (REDTEAM-v2 `randres.py`, simulated, S0-03 seed 0, docs 64..1088). Residency past
  gap M: FIFO 0.0000 against random (never evicting the newest write) 0.3305. Within gap M:
  random 0.897 against FIFO 1.0.

**Not seen by anyone:** any document of `FIT_TRAIN`, `FIT_VAL`, `EVAL` or `[900000, 920000)`,
and any ψ̂ fit on any data.

**After B0,** only the "already seen" addendum below, `n` (TBD-1, TBD-3) and the capture cost
(TBD-2) may be added, each by a diff (the fence at the top). **Nothing else may change.**

### Addendum slots (empty at this commit)

- **TBD-1** `N_F` and the ridge row counts: _not yet written._
- **TBD-2** the capture cost: _not yet written._
- **TBD-3** `N_E`: _not yet written; computed by §9.4 before any `EVAL` document is run._
- **The "already seen" addendum after B0:** _not yet written._

## Amendment 1 (2026-09-27, pre-data)

**Written 2026-09-27 by a model** (Claude Opus 5.5, an RSR researcher session, item AMD). **Not
written by Brendan.** **Append-only.** Nothing above this heading has been edited; it is
byte-identical to `e5ab398`. Where this section and the text above disagree, **this section
governs**, including over the front matter.

- **Source.** Every paragraph below answers one finding of
  `~/Documents/RSR-2026-09-27-plan/reviews/PREREG-REVIEW-e0d-b2.md` (the review), and applies the
  review's amendment text, quoted. Where the author of this amendment disagrees with the review or
  has to fill a gap it leaves, the paragraph carries a marked **author's note**. No author's note
  changes a gating rule the review wrote.
- **Authority.** PLAN-v4 B2 "Commit order" step 4 restricts edits only **after B0**, and
  `~/Documents/RSR-2026-09-27-plan/LOOP.md` ("Hard stops", clarified in cycle 3) allows an
  append-only amendment committed alone before any data exists, citing the finding it answers.
- **This amendment is itself pre-data and pre-B0.** At this commit:
  - no document of `FIT_TRAIN`, `FIT_VAL`, `EVAL` or `[900000, 920000)` has been generated, run or
    inspected, and no ψ̂ has been fitted on any data;
  - no B0 PREREG and no B0 branch exists (`git branch -a` shows none; the only `b0` name,
    `origin/s0/b0b-sigkill`, is Sprint 0's unrelated "Brief 0");
  - no E0d data exists (E0d's own Amendment 1, `run/e0d@84e21c5`, is also pre-data).
- **The self-reference, stated.** The fence as committed (lines 35–43 above) forbids this very
  amendment. A1.1 replaces that fence with PLAN-v4's, as the review's blocker B2-B1 requires. The
  amendment is made under PLAN-v4 and LOOP.md, not under the committed fence; a reader who holds
  the committed fence authoritative should read this amendment as the reported defect the fence
  asks for, and B2's classification as void until Brendan rules. That is the owner's call.

### A1.1 The fence (answers B2-B1, BLOCKER)

The fence paragraph (lines 35–43, "**The fence.** After this commit, …" through "… it voids the
classification.") and the matching sentence under the "already seen" section ("**After B0,** only
…") are superseded by:

> **The fence.** Until the B0 PREREG is committed, this file may be amended only by commits that:
> (a) each cite the adversarial-review finding they apply (`reviews/PREREG-REVIEW-e0d-b2.md`);
> (b) show their `git diff` against `e5ab398`; and
> (c) are ordered before the B0 PREREG commit.
> No such amendment may use any B2, B0 or E0d data. **After the B0 PREREG is committed,** only
> three things may be added: … *(the existing three items, unchanged)*.

The existing three items, unchanged, are: (1) **n**: the TBD fields `N_F` (and the ridge row
counts it implies) and `N_E`; (2) **the capture cost**; (3) **an "already seen" addendum** (§A,
formerly "§15"; see A1.14), listing what B0 showed. Each is added in its own commit showing its
`git diff` against the commit before it. After the B0 PREREG is committed, an edit outside those
three is itself a finding and voids the classification, as the committed fence said.

Condition (a) is satisfied here by the finding map, A1.16. Condition (b) is satisfied by the
commit that lands this section, whose `git diff e5ab398` is additions only.

### A1.2 One-step-shifted companion targets, non-gating (answers B2-M1, MAJOR; and R1)

**Why.** `run_policy_loop` calls `select_eviction` *after* step `t`'s forward has read memory,
using that forward's `out.srep` as `c_t` (`policy_loop.py`). `OraclePolicy` scores demand from
`step + 1` ("this step's query has already read memory… so it is not the victim's to lose",
`oracle.py`). B2's `G_γ[i][t]` starts at `k = 0` (§5), so it includes `X[i][t]`, demand the
eviction at `t` can no longer affect. At γ = 0 the whole target is decision-irrelevant; at
γ = 0.9 the k = 0 term pulls ψ̂ toward protecting the slot just read (at a query step, the
just-answered assert, whose future value is 0 since each fact is queried once). This is
spec-faithful (§3.2.2, §3.3, correction 2), so **the gating arms are not changed**.

**§5 "The target arms" gains a row; §7 and §9.6 gain non-gating arms:**

> | **(U⁺, C⁺) one-step-shifted, non-gating** | as U and C, but `G⁺_γ[i][t] = Σ_{k=1}^{S−1−t} γ^{k−1} X[i][t+k]` (demand from the next step on, the quantity `OraclePolicy` scores) | as U and C | **Reported beside ψ̂-U and ψ̂-C with the same outcome rule; never gating.** It exists so that a LOSS or EQUIV can be attributed to the k = 0 term or to the target. |

- **γ values:** U⁺ and C⁺ are fitted and run at **γ = 0.9 and γ = 0**, per checkpoint and seed.
  At γ = 0, `G⁺_0[i][t] = X[i][t+1]` (with `0⁰ = 1`).
- **Rows:** exactly U's and C's rows; at `t = S−1` the sum is empty and `G⁺ = 0`. Declared.
- **Fit:** the §6 bilinear ridge with every §6 rule as amended here (A1.6, A1.7); the design
  matrix is U's (resp. C's), only `y` differs.
- **Outcome rule:** §9.5, against **the same `ref` and δ as the unshifted arm** (U⁺ against
  `ref_U`, C⁺ against `ref_C`).
- **Reporting:** a side-by-side table, per seed, of ψ̂-U, ψ̂-U⁺, ψ̂-C and ψ̂-C⁺ at each γ, with Δ,
  CI and outcome. The truth table (§9.6) is computed on the unshifted arms only. A second
  truth-table class computed on (U⁺, C⁺) is reported, labelled **non-gating, companion**.

**E0h companion (R1), added to §10 "Reference, never gating":**

> **k ≥ 1 companion, never gating.** The within-step R² of ψ̂-U⁺(γ = 0) (B2-M1: the target
> `X[i][t+1]`, the one-step-ahead demand the γ = 0 oracle scores) on the same logits and rows. If
> ψ̂-U(γ = 0) is COLLINEAR and ψ̂-U⁺(γ = 0) is not, the report states: "collinearity is inherited
> from the k = 0 term of the §3.2.2 target". The class is unchanged. The interpretation is
> Brendan's.

> **🔴 Flag to Brendan (not a correction; owner's call):** spec §3.2.2 and §3.3 include `r_i(t)` in
> the return, but the eviction at `t` is decided after step `t`'s read. Whether this is a spec
> defect for `spec-corrections.md` is the owner's decision.

This amendment does **not** edit `spec-corrections.md`, the spec, or the gating targets.

### A1.3 E0h's own entry point and exit code (answers B2-M2, MAJOR)

**Why.** E0h is a spec kill gate (spec :552, "Kills it? Yes"). As committed, COLLINEAR exits 0
and "exact logits unavailable" gives B2 exit 0 with E0h marked DID NOT RUN, so a kill gate that
did not run and one that fired both leave rc 0.

**§12's E0h clause** ("or E0h's exact logits unavailable (that one affects E0h only; B2 may still
exit 0 with E0h marked DID NOT RUN)") **is replaced, and §10.1 is added:**

> **§10.1 E0h's own status.** E0h is run by its own entry point (`run.py e0h`), writes its own
> ledger keys (`e0h.*`) and returns its own rc:
> - **0:** NOT_COLLINEAR;
> - **1:** COLLINEAR, emitted only once a ratifying ruling exists;
> - **2:** INTERMEDIATE, UNINFORMATIVE, or unratified;
> - **3:** DID NOT RUN, which includes logits being unavailable.
>
> B2's rc covers B2 alone. No report may give B2's rc as E0h's.

- "Unratified" means: while no ruling ratifying (or replacing) the E0h thresholds exists, E0h
  exits **2 whatever the R² values are**, and its numbers are reported unclassified (§10, "The
  threshold"). Only after ratification can E0h exit 0 or 1.
- `run.py e0h` needs B2's ψ̂-U(γ = 0) and ψ̂-U⁺(γ = 0) fits and the hooked FIFO rollout of `EVAL`
  (A1.5). If either is absent, E0h exits 3.
- An exception inside `run.py e0h` exits **3** (not 1), with the traceback in the ledger; exit 1
  is emitted only after `e0h.class = COLLINEAR` has been written under a ratifying ruling. (This
  is E0d-M3's rule, applied to E0h's kill gate; the review's M2 text implies it by making 1
  mean COLLINEAR only.)
- B2's own §12 table is otherwise unchanged: 0 = ran, every control passed, classification
  reached; 1 = a measurement defect; 3 = did not run or a control failed.

### A1.4 A compute ceiling on EVAL (answers B2-M3, MAJOR)

**Why.** About 25 arm-runs per EVAL document, each 48 B = 1 forwards, at `N_E = 40000` × 2
checkpoints × 3 seeds is about 2.9×10⁸ forwards: about 80 h even at 1 ms per forward, with no rule
for an unaffordable `N_E`, while PLAN-v4 §5 schedules B2 in Day 2–3.

**§9.4 gains item 7:**

> 7. **Compute ceiling, fixed now.**
>    - **Tier 1** (ckpt3000 × γ = 0.9, the arms in §9.5–9.7 plus FIFO, age-only, random, oracle
>      and kind-oracle) runs on `N_E` documents.
>    - **Tier 2** (ckpt2500, γ = 0, U-r, U⁺ and C⁺) runs on the first `min(N_E, 4096)` of the same
>      documents and is labelled so.
>    - Before any EVAL document is run, TBD-2 records the measured cost per arm-document. If Tier 1's
>      projected wall time exceeds **72 h**, `N_E` is reduced to the largest value that fits, and
>      the run is labelled **UNDERPOWERED**. The decision rule is read unchanged.

**The forward count at the cap** (B = 1 forwards; 48 per arm-document; 3 seeds):

| Tier | arm-runs per (document, seed) | documents at the cap | B = 1 forwards at the cap |
|---|---|---|---|
| 1 | 12: ψ̂-U, ψ̂-C, FIFO, age-only (U), age-only (C), random ×5, oracle, kind-oracle (random tie-break, A1.10) — all ckpt3000, γ = 0.9 | 40,000 | 12 · 48 · 40,000 · 3 = **69,120,000 ≈ 6.9×10⁷** |
| 2 | 44: at ckpt3000, 16 (the γ = 0 set of ψ̂-U, ψ̂-C, age-only ×2, oracle, kind-oracle, kind-oracle oldest-tie, U⁺, C⁺; plus U-1..3, U⁺, C⁺, kind-oracle oldest-tie and fact/filler at γ = 0.9); at ckpt2500, all 28 arms | 4,096 | 44 · 48 · 4,096 · 3 = **25,952,256 ≈ 2.6×10⁷** |
| total | | | **≈ 9.5×10⁷**, against ≈ 2.9×10⁸ uncapped (review) |

- **Wall-time budget.** Tier 1: **72 h** (the review's rule, above), i.e. the Tier 1 cap is
  affordable only at ≤ 3.75 ms per B = 1 forward including the policy's overhead
  (72 · 3600 s / 6.912×10⁷). TBD-2 measures that cost; `N_E` follows from it.
- *Author's note (Tier 2 budget; an addition the review does not contain).* The review gives no
  Tier 2 budget. I add one so that Tier 2 cannot silently extend B2 past PLAN-v4's schedule:
  **Tier 2's projected wall time, from the same TBD-2 cost, must be ≤ 24 h; if it is not, Tier 2's
  document count is reduced to the largest value that fits and labelled `TIER2_REDUCED`.** Tier 2
  is never gating, so this can change no classification.
- *Author's note (arms the review's tiers do not name).* fact/filler and the kind-oracle
  oldest-tie variant (A1.10) are reported, never gating, so I place them in Tier 2. E0h needs no
  arm-run of its own: its rows are the Tier 1 FIFO rollout (ckpt3000, `N_E` documents), captured
  with the A1.5 hook; the ckpt2500 E0h report uses the Tier 2 FIFO rollout.
- **The oracle** at γ = 0.9 is kept in Tier 1 as the review lists it (its headroom is reported on
  `EVAL`; δ still comes from `FIT_VAL`, §9.3).

### A1.5 E0h logits: an exact pre-hook, mean collapse, the right control (answers B2-M5, MAJOR)

**Why.** `Q_TOK_COLLAPSE` sums **post-softmax** α over query tokens (`policy_loop.py`,
`cross_capture`). The committed control collapses logits "by the same rule", then checks that
softmaxed logits reproduce captured α; since `softmax(Σ_q logit) ≠ Σ_q softmax(logit)`, that
control fails spuriously and marks E0h DID NOT RUN. Collapsing logits by sum also multiplies them
by `Q_real`, which varies by step, and breaks a pooled constant-β regression.

**§10 "Regressors": the three sub-bullets (collapse "by the same rule as α"; "recomputed in the
harness"; the softmax control; and "If the build cannot produce exact logits, E0h is exit 3 … No
substitute (for example `log α`) is silently used") are replaced by:**

> - Logits are captured per query token by a forward pre-hook on each `cross_attn` (the forward
>   pass is unchanged) and **collapsed by the mean over real query tokens**.
> - **Control:** per query token, *before* collapse, `softmax(recomputed logits)` reproduces
>   `out.cross_attention` to ≤ 1e-5.
> - If the hook cannot be built, the within-step primary may use the per-token `log α` (exactly
>   equal after within-step demeaning, since the full memory has no masked slot). Only the pooled
>   secondary is then DID NOT RUN.

- **The hook.** A `register_forward_pre_hook` on each block's `cross_attn` captures its input
  (`ln_mem(x)`, `mem_kv`, `mem_valid`); the logit is `query(x) · scale · key(mem_kv + PE)`,
  per head. `test_fidelity.py` and the forward pass are untouched. A failed control is E0h exit 3.
- **Why `log α` is exact for the primary.** Per query token `q`,
  `log α_{q,i} = logit_{q,i} − logsumexp_i(logit_{q,·})`; the second term is constant over slots
  `i`, so it vanishes under within-step demeaning, and the mean over `q` of demeaned values equals
  the demeaned mean. E0h rows are full-memory FIFO-resident slots only, so no slot is masked.
- If `log α` is used, the report says so, and E0h's ledger records `e0h.regressor_source =
  "log_alpha"`; otherwise `"prehook_logit"`.

### A1.6 The ridge residual check applies to the selected λ (answers B2-M4, MAJOR)

**Why.** At `λ = 1e-6 · tr/p` the condition number can reach 1e8–1e9 for Kronecker features of
low effective rank; a backward-stable solve's relative residual (≈ `c(p) · ε · κ`) can then exceed
1e-8 at p = 16,640, and a grid point that will never be selected would end the run at exit 1.

**§6 "Method": the Cholesky and residual bullets are replaced by:**

> - **Method:** eigendecompose `XᵀX` once per Gram (fp64), and solve every λ spectrally.
> - **Grid points:** any grid point whose relative residual exceeds `1e-6` is ineligible and
>   logged.
> - **The selected λ:** it must have a relative residual ≤ 1e-8. Otherwise the run exits 1.
> - **If no grid point is eligible,** the run exits 1.

The relative residual is §6's `‖(XᵀX + λ'I)w − Xᵀy‖ / ‖Xᵀy‖`. §11 control 10 now refers to the
selected λ's check and the "no eligible grid point" check. λ is selected among eligible grid
points only (A1.7).

### A1.7 λ selection on the within-step demeaned MSE (answers B2-m3, R7)

§6 "λ", first sub-bullet, is replaced by:

> λ minimises the **within-step demeaned** validation MSE over full-memory rows (argmin is
> shift-invariant within a step). The raw MSE is reported.

Ties still go to the larger λ, and the refit on `FIT_TRAIN` alone is unchanged.

*Author's note (scope).* This applies to every fit whose output is an eviction score: the
bilinear heads (U, C, U-r, U⁺, C⁺) and the age-only heads. The age-decodability predictor (§6,
§8.1) is not an eviction score, and its reported statistic is the R² of raw age, so it keeps λ by
raw validation MSE. The review does not address that predictor.

### A1.8 E0h thresholds: seed wording and the 0.49 provenance (answers B2-m9 and the 0.90/0.49 assessment)

**§10's NOT COLLINEAR row is replaced by:**

| Proposed class | Condition (within-step R², ψ̂-U(γ = 0), ckpt3000) |
|---|---|
| **NOT COLLINEAR** | R² < **0.49** on **every informative seed, with at least 2 informative** |

COLLINEAR (≥ 0.90 on ≥ 2 of 3 seeds), INTERMEDIATE and UNINFORMATIVE are unchanged. A seed is
informative when it is not UNINFORMATIVE. The front matter's `E0h_proposed` "< 0.49 on all 3" is
superseded by this row.

**Provenance note for 0.49.** 0.49 is `0.7²`, and 0.7 is the spec's line in E2's vacuity gate
(§7.1; kill table :558, "Partial ρ < 0.7"). **That is a different statistic:** E2's 0.7 is a
univariate *partial rank* correlation, while E0h's R² is a multiple R² over 12 regressors (6
cross-attention layers × H = 2 heads). The borrowing is a convention, not a derivation, and a
reader should not read E0h's 0.49 as the spec having set it.

Both numbers (0.90, 0.49) remain **PROPOSED**; the review recommends ratifying them only together
with A1.3 (E0h's own rc) and the k ≥ 1 companion (A1.2).

### A1.9 Declared departure: E0h's rows (answers B2-m7)

Added to §10:

> **Declared departure from PLAN-v4:** E0h's rows come from EVAL's FIFO rollout, not from the
> FIT capture, so that ψ̂ is scored out of sample.

### A1.10 The kind-oracle's tie-break (answers B2-m1, R4)

§6, the kind-oracle policy ("with ties going to the oldest"), is replaced by:

> Ties within the lowest class are broken **uniformly at random**, with
> `random.Random(f"ko:{seed}:{doc_id}:{t}")`. The oldest-tie-break variant is reported as a
> secondary. Q2 uses the random-tie-break comparator.

Why: an age tie-break gives Q2's comparator age information that ψ̂ is barred from. The
random-tie-break kind-oracle is also the kind-oracle of the §9.4 power list, §9.6 and every
Tier 1 use; the oldest-tie variant is Tier 2 (A1.4).

### A1.11 δ at γ = 0 (answers B2-m2, R8)

§9.3 is amended:

> At γ = 0, δ is the γ = 0.9 δ of the same seed. The γ = 0 oracle's headroom is reported, not
> used.

Why: with the one-step γ = 0 oracle ≈ FIFO, a γ = 0 δ near 0 makes EQUIV unreachable and LOSS a
sign test. This applies to every γ = 0 contrast, including U⁺ and C⁺ at γ = 0.

### A1.12 Power: only gating contrasts; and what the power is sized for (answers B2-m4, R9)

§9.4.1 is amended:

> Only the gating contrasts (the first four bullets) enter the max. ψ̂-U − kind-oracle is reported
> with the N_E that results.

*Author's note (a miscount in the review's text; the intent is unambiguous).* §9.4 item 1 has
**four** bullets, and the fourth is "ψ̂-U − kind-oracle", which the same sentence then excludes.
I read "the first four bullets" as **the contrasts of the first three bullets** (six contrasts:
ψ̂-U − ref_U and ψ̂-C − ref_C, all-query and on `gap_2_to_M`; ψ̂-U and ψ̂-C against the random
mean). The kind-oracle contrast (Q2, outside the truth table) does not enter the max.

**Stated, per R9.** The power is sized for EQUIV (CI half-width ≤ δ/2), not for WIN. At a true
Δ = δ, per-seed WIN is ≈ 50 %, so 3 of 3 WIN is ≈ 12.5 %. PLAN-v4 accepted this.

### A1.13 WIN/EQUIV overlap (answers B2-m8, R10)

§9.5 is amended:

> If `Δ̂ ∉ [lo, hi]`, the contrast is UNRESOLVED and flagged `CI_EXCLUDES_ESTIMATE`.

This check runs before the WIN/LOSS/EQUIV order. It removes the only WIN ∧ EQUIV case (`Δ̂ ≥ δ >
hi`); LOSS/EQUIV and WIN/LOSS were already disjoint. It applies to Q2 (§9.7) and to every
companion arm too.

### A1.14 "§15" is renamed "§A" (answers B2-m12)

The section headed "## 15. "Already seen" declaration" above is hereafter **§A, "Already seen"**.
Every reference in this file to "§15" in that sense (the fence's item 3; the addendum slots)
means §A. **This file has no content for, and makes no reference to, spec §15**, which no model
may fill. The heading text above is left as committed only because this file is append-only.

### A1.15 The remaining minors and R-items

- **Ranges (B2-m5).** §3 is amended:
  > The table of used ranges also asserts `[262144, 263168)` (E0d, per `run/e0d@5357ad2`).
  > `[900000, 920000)` stays unused by B2.

  Why: the conditional row ("if `experiments/e0d/PREREG.md` exists on the run's branch") is false
  on `run/b2-psi-probe`, so it would silently skip E0d's range. There is no collision today
  (review §1). The "Coordination with E0d" paragraph of §3 is stale: `run/e0d` now exists and E0d
  took `[262144, 263168)`, not the reserved block.
- **N_F (B2-m6).** TBD-1 is amended:
  > `N_F = ⌈10·p / 632⌉`, or a larger value fixed in the TBD-1 commit **before any fit is
  > computed**. It is never revised after a fit.

  At `d = 128`, `p = 16,640`, this is `⌈166,400 / 632⌉ = 264` (computed, to be asserted from `p`).
- **T0 manifest path (B2-m11, R14).** §2's "T0 check" and §11 control 1 read the T0 manifest path
  from the T0 record (`runs/t0-substrate/manifest.json` or its equivalent), not the literal
  `~/rsr-substrate/2026-09-27/MANIFEST.sha256`. No T0 record at start is exit 3.
- **`ref` (B2-m10, R3). NOT adopted by this amendment.** The review marks it optional and says it
  changes PLAN-v4's "whichever scores higher", so it needs the PM's or Brendan's sign-off. The
  committed §9.3 rule stands. The proposal, carried to them verbatim: "`ref` = age-only only if
  its paired FIT_VAL CI against FIFO has lower bound > 0; otherwise FIFO." If adopted, it is by a
  separate cited amendment ordered before the B0 PREREG (A1.1).
- **R3 reporting.** The `ref` chosen for each (seed, arm, γ) is reported beside that arm's Δ,
  since ψ̂-U and ψ̂-C may be compared against different refs.
- **R2, declared.** The age-only head is a one-hot ridge. The C rows see only ages ≤ 16, so the
  coefficients for ages never seen in its rows shrink to 0 under ridge; since the targets are
  non-negative, an older slot kept by an age-only head then scores ≈ 0 and is expected to be
  evicted next.
- **R5, declared.** F-X1's random (0.3305 residency past gap M) never evicted the newest write, so
  it is **not** this file's random arm (uniform over all live slots). F-X1 is not quoted as this
  arm's prior.

### A1.16 Finding → paragraph map

| Review finding | Severity | Answered in |
|---|---|---|
| B2-B1 (the fence) | BLOCKER | A1.1; pre-data/pre-B0 statement in the preamble |
| B2-M1 (k = 0 term; U⁺/C⁺ at γ = 0 and 0.9; flag to Brendan) | MAJOR | A1.2 |
| R1 (E0h k ≥ 1 companion) | assessment → amendment | A1.2, "E0h companion" |
| B2-M2 (E0h entry point and rc) | MAJOR | A1.3 |
| B2-M3 (EVAL compute ceiling; forward count; wall time) | MAJOR | A1.4 |
| B2-M4 (residual check at the selected λ) | MAJOR | A1.6 |
| B2-M5 (pre-hook logits, mean collapse, `log α`, control) | MAJOR | A1.5 |
| B2-m1 (R4, kind-oracle tie-break) | MINOR | A1.10 |
| B2-m2 (R8, γ = 0 δ) | MINOR | A1.11 |
| B2-m3 (R7, λ selection) | MINOR | A1.7 |
| B2-m4 (R9, gating contrasts only drive `N_E`) | MINOR | A1.12 |
| B2-m5 (E0d's range asserted) | MINOR | A1.15 |
| B2-m6 (`N_F` rule) | MINOR | A1.15 |
| B2-m7 (declared departure, E0h rows) | MINOR | A1.9 |
| B2-m8 (R10, WIN/EQUIV overlap) | MINOR | A1.13 |
| B2-m9 (E0h seed wording) | MINOR | A1.8 |
| B2-m10 (R3, `ref` rule) | MINOR, optional | A1.15 — **not adopted; needs sign-off** |
| B2-m11 (R14, T0 path) | MINOR | A1.15 |
| B2-m12 ("§15" → "§A") | MINOR | A1.14 |
| E0h 0.90/0.49 assessment (0.49 provenance) | assessment | A1.8 |
| R2, R3, R5, R9 notes | assessment | A1.15, A1.12 |

## Addendum: already seen (B0)

**Written 2026-09-27 by a model** (Claude Opus 5.5, an `rsr-researcher` session, item B0). **Not
written by Brendan.** **Append-only.** This fills fence item 3 (A1.1): the "already seen"
addendum, listing what B0 showed. Nothing above this heading changes. **It changes no rule,
threshold, arm, range or target of B2, and it is not Q2's comparator.** B2's kind-oracle is
re-estimated on `FIT_TRAIN` (§6, §9.7).

- **Source.** `runs/b0-ceilings/ledger.json` on `run/b0-ceilings`.
  - Run SHA **`bb40c40557d11bb01c0907a988ba76f86d3aa2ac`** (the code that ran; clean tree). Ledger committed at `b4f3661`.
  - PREREG `fc26dbf`. Config hash `3df1a646…2433`. `status: ok`, rc 0, seeds [0, 1, 2], 1088 documents each.
- **Scope.** Model-free residency on the U range, which was **already inspected**, with every
  class mean estimated **in-sample**. Every number is written seed 0 / 1 / 2.
- **Controls.**
  - C3: B0 reproduced lookahead-room-r2's `U.hit.{fifo,factfiller,oracle,rule_g0,rule_g09}` exactly.
  - T0 manifest: rechecked at start and at end, and 435/435 OK after the run.

**Caps at ckpt3000, γ = 0.9.** Cap = (hit − FIFO) / (factfiller − FIFO); FIFO 0.8059 / 0.8093 / 0.8088; factfiller 0.9655 / 0.9653 / 0.9666.

| Rule | Hit | Cap [paired per-doc 95% CI] |
|---|---|---|
| kind-oracle, the A1.10 random tie (`B.ckpt3000.g09.cap.ko`) | 0.9649 / 0.6613 / 0.9669 | +0.996 [+0.978, +1.016] / **−0.949** [−1.014, −0.891] / +1.002 [+0.982, +1.021] |
| kind-oracle, the oldest tie | 0.9873 / 0.7086 / 0.9862 | +1.137 / −0.645 / +1.124 |
| kind × age-band, **not a legal ψ̂** | 0.9491 / 0.7645 / 0.9770 | +0.898 / −0.287 / +1.066 |
| age-only, argmin E[G \| age] | 0.8124 / 0.8256 / 0.8105 | +0.041 [+0.004, +0.074] / +0.105 [+0.077, +0.133] / +0.010 [−0.028, +0.044] |
| hindsight `rule_g09` | 0.8352 / 0.8301 / 0.8935 | +0.184 / +0.134 / +0.537 |

**Other combinations** (kind-oracle cap):
- It is identical at γ = 0 and γ = 0.9 within each checkpoint, because the class-mean order does not change.
- At ckpt2500 it is **−0.933 / −0.949 / +1.002**: seed 0 is negative too.

**Why the kind-oracle is negative.** `E[G_0.9 | kind]` at ckpt3000 (assert / query / filler):
- seed 0: 0.5728 / 0.5340 / 0.4586;
- seed 1: 0.4055 / **0.5098** / 0.3425;
- seed 2: 0.5862 / 0.4772 / 0.3181.

Filler is the lowest class on every seed. On seed 1, however, assert < query, so once the fillers
are gone the rule evicts asserts before query sentences. At ckpt2500, seed 0 has the same order
(assert 0.5894 < query 0.5911).

**🔴 A correction to F2 as quoted in §A and PLAN-v4 §1.**
- Under the kind definition, `E[G_0.9 | assert] − E[G_0.9 | filler]` at ckpt3000 is **+0.114 / +0.063 / +0.268**. The CIs exclude 0, and at γ = 0 it is +0.015 / +0.007 / +0.036.
- The scratch F2 (+0.0633 / **−0.0507** / +0.1601) is `E[G | kind = assert] − E[G | slot_class = filler]`, and W10's `slot_class` counts **query sentences as filler**.
- From the same ledger: 0.5728 − 0.5095, 0.4055 − 0.4562, 0.5862 − 0.4260 = +0.0633 / −0.0507 / +0.1601.
- So the "seed-1 reversal" is asserts against a pool that contains high-G query sentences, not against true fillers.
- §13's expectation cites F2. **§13 is not edited:** it is the author's pre-data expectation, and it stands as written.

**Reproduced exactly from §A** (now ledgered, `B.ckpt3000.F1.D.<age>|<class>.mean`, and the class means):
- pending/filler `D` at age 1: 0.0930/0.0580, 0.0999/0.0831, 0.1047/0.0615;
- at age 16: 0.0783/0.0928, 0.0508/0.0683, 0.0758/0.0731;
- probe at rank 0: 0.0843/0.0962, 0.0544/0.0817, 0.0829/0.0757;
- pooled `G_0.9` pending 0.6956 / 0.5285 / 0.7200, filler 0.5095 / 0.4562 / 0.4260;
- hindsight cap 0.18 / 0.13 / 0.54.

## Addendum: TBD-1, n (fence item 1)

**Written 2026-09-27 by a model** (Claude Opus 5.5, an `rsr-researcher` session, item B2-B). **Not
written by Brendan.** **Append-only.** It fills fence item 1 (A1.1) with `N_F` and the ridge row counts,
and nothing else. It changes no rule, threshold, arm, range or target. `N_E` (TBD-3) is **not**
written here: §9.4 computes it from the phase-A fits on `FIT_VAL`, which have not run.

- **`N_F = 264`**, by A1.15's rule `N_F = ⌈10·p / 632⌉`, with no larger value chosen.
  - `d = 128` is read from `runs/fresh-stream/manifest.json` (key `d`) and asserted equal to each
    checkpoint's embedding width (sizing ledger `ckpt{2500,3000}.D` = 128 on all six).
  - `p = d² + 2d = 16,640` (sizing ledger `gram.p`).
  - `FIT_TRAIN` uses ids `[920000, 920264)` of each seed's generator.
- **The row counts per document were measured, on `FIT_VAL` only**, and match §5. They are 1128 U rows
  and 632 C rows on every one of the 32 documents × 3 seeds × 2 checkpoints (sizing ledger
  `ckpt{c}.rows_U_per_doc`, `ckpt{c}.rows_C_per_doc`).
- **The ridge row counts** are `n_U = 1128 · 264 = 297,792` and `n_C = 632 · 264 = 166,848`.
  `n_C / p = 10.03`.
- **Source.** `runs/b2-psi-probe-sizing/ledger.json` on `run/b2-psi-probe`, run at `9b76963`
  (status `ok`, rc 0). Its only documents are `FIT_VAL` ids `[936000, 936032)` per seed.
