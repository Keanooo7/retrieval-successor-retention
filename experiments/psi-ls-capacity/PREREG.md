---
run_id: psi-ls-capacity
item: "After diagnostic A: the exact least-squares optimum of psi_hat's function class on U, with no optimizer"
question: >-
  Diagnostic A trained offline-psi's head to convergence under run 2's optimizer (lr 1e-3, phi weight
  decay 0.01, muP multipliers) and it still MISSED (cap 0.27/0.41/0.15). Is that the function class,
  or the optimizer that searched it? The L_MC-optimal member of psi_hat's class is an ordinary
  least-squares problem. Solved exactly, does it reach a useful ranking (offline-psi PREREG §5's fit
  cap), on U-val and on the U-train documents it was fit to?
class: "DIAGNOSTIC (Tier D). U documents only. No EVAL read, no deployment, no spec falsifier, no gradient step. Two labels (U-val fit, U-train in-sample fit); together they route an OWNER decision (§6)."
not: "Not a re-run of offline-psi or diagnostic A, and it changes none of their labels (FIT-MISSES, FAILS, CONVERGED) or run 2's class. Changes no learning rate and no muP multiplier (correction 15 is Brendan's): the closed form has neither. No scaling claim. No cognitive claim. §15 untouched. Does NOT bound the best RANKING the class can express: L_MC is a squared error over all cells, and a ranking-targeted fit of the same class is not tested here (§3 item 5)."
seeds: "[0, 1, 2]"
status: "IN FORCE as of the commit that adds this file, ALONE, before any code. Owner go 2026-10-08: \"go ahead with the closed-form least-squares check\"."
---

# PREREG: the least-squares optimum of ψ̂'s class (capacity check)

**Drafted 2026-10-08 by a model** (Claude Opus 5.5, an interactive session with Brendan). **Not
written by Brendan.** He approved it in these words: *"go ahead with the closed-form least-squares
check"*, in reply to the PM's report proposing "an exact closed-form least-squares fit … no lr or μP
involved … the last check before any change to the thesis". From this commit on, amendments are
appended, dated and committed alone.

## 0. Why

Diagnostic A (`diag/offline-psi-converge`, PREREG `58783a3`, RESULTS `f7ac595`, rsr-verifier ok
`9c998d3` on `verify/offline-psi-converge`) trained offline-psi's head on U until U-val L_MC
stopped improving. All seeds converged (cf_all at 95.5k / 117k / 103.5k steps) and the cap at the
selected φ was 0.27 / 0.41 / 0.15: **FIT-MISSES**. Train and U-val L_MC were close, so the fit
was not overfitting.

That leaves two explanations it cannot separate:

- **the function class** cannot express G's ranking;
- **the optimizer** did not reach the class's best member. Run 2's AdamW holds φ weight decay
  0.01, and the μP multipliers (`1/d` on the bilinear form, `1/(2d)` on the linear term) set the
  effective step size per coordinate. Either can stop the fit short of the loss minimum, and a
  plateau under them is not the minimum.

The class's L_MC minimum needs no optimizer. ψ̂ is linear in its parameters:

    psi(s, c) = s^T W c / d + u_s·s / (2d) + u_c·c / (2d)
              = <A, s c^T> + a·s + b·c          with A = W/d, a = u_s/(2d), b = u_c/(2d)

Run 2's L_MC (ADR-0009 L9) is the squared error summed over cells and divided by a count that
does not depend on φ. Minimising it over (A, a, b) is ordinary least squares on 16,640 features
(`vec(s c^T)`: 16,384; `s`: 128; `c`: 128). The solution is exact, and every (A, a, b) is reachable
by (W, u), so the parametrisation (μP or not) does not change the optimum.

## 1. Data: offline-psi's, read, never recomputed

- **Gestalts:** `.worktrees/offline-psi/runs/offline-psi/seed{s}/gestalts.pt`, through diagnostic
  A's `load_inputs`. They carry the same sha256 pins as diagnostic A PREREG §1:
  - seed 0: `211244f1469b90972e3b1d3b6635b3ee9eb095e78b932dc396bba64df3f5562b`
  - seed 1: `3a18e2917b385a6339ef82e92535953105877cf295bbaf9f6495a4749a6a0c68`
  - seed 2: `74e3bec7e5366d631480f9e3ac9a07916926da21ffd60be57568acb3b3723706`
- **Δ:** B0-CF's `delta.pt`, with offline-psi PREREG §1's sha256 pins.
- **Targets, cells and split:** offline-psi's, imported, never re-implemented: `target_of`,
  `cell_mask`, `fit_tensors`, `u_documents` and `u_split`. G is `discounted_returns(Δ, 0.9)`; the
  arms are cf_all (primary) and cf_live (secondary); U-train / U-val is positions `p % 7`
  (932 / 156).
- **Diagnostic A's results**, read for comparison and for control C3(b). These are
  `.worktrees/offline-psi-converge/runs/offline-psi-converge/seed{s}/result.json`, pinned by sha256:
  - seed 0: `8c88bd05f89a3bfaa192852bf1cd5251586bb1381a6ca3770f8596a2d7b2d965`
  - seed 1: `0bb3ee8c15bf95ea53262215008ef29561569188fcab4339cb2691794c421204`
  - seed 2: `496beeb4e98c64dc9f80305317fc9dc5177d85750f9ae05c69b09956826edfc6`
- **EVAL is not read.** There is no deployment.

## 2. The fit: the exact least-squares solution, per seed and arm

- **Fit set:** U-train only. The arm's cells are the ones where the mask is set, with every cell
  weighted 1.
- **Normal equations,** assembled in float64 from per-(document, t) sufficient statistics: for
  each group, `Σ_i s_i s_iᵀ`, `Σ_i s_i`, `Σ_i G_ti s_i`, `Σ_i G_ti`, the cell count, and `c_t`. Nothing
  is subsampled.
- **Solve:** the symmetric eigendecomposition of `XᵀX` (`torch.linalg.eigh`, float64). Eigenvalues
  `≤ p · eps₆₄ · λ_max` (`p = 16,640`) are dropped. The solution is the minimum-norm least-squares
  solution, `w = V diag(1/λ) Vᵀ Xᵀy` over the kept eigenvalues.
- **Map to the head:** `W = d·A`, `u = 2d·[a; b]`, loaded into `BilinearValueHead` (built by
  offline-psi's `new_policy` at run 2's psiC φ seed) and cast to float64. Every reading below goes
  through that head's own forward pass and offline-psi's functions.
- **No hyperparameter is selected on U-val.** The ridge path in §3 item 4 is descriptive only.

## 3. Readings

1. **U-val fit cap** of the least-squares head, by offline-psi PREREG §5's definition and code:
   - the model-free replay `replay_rates` against `reference_rates`;
   - the cap from `B0.boot_rule`;
   - the paired per-document bootstrap over the 156 U-val documents (`boot_weights(156, seed,
     2000)`).
2. **U-train in-sample fit cap.** The same, over the 932 U-train documents, with
   `boot_weights(932, seed, 2000)`. It asks whether the class's L_MC optimum ranks well even on
   the documents it was fit to.
3. **Labels:** offline-psi PREREG §5's rule, verbatim, applied to cf_all.
   - **U-val fit label:** FIT-POINTS if the cap CI lower is ≥ 0.80 on all 3 seeds; FIT-PARTIAL if
     not, and the point is ≥ 0.50 on all 3 seeds; FIT-MISSES otherwise.
   - **In-sample label:** the same rule on reading 2, written IN-SAMPLE-POINTS / -PARTIAL /
     -MISSES.
4. **Descriptive, never labelled:**
   - U-val and U-train L_MC of the least-squares head, beside diagnostic A's selected values and
     beside the constant-mean baseline (diagnostic A's `constant_baseline`);
   - the share of diagnostic A's U-val loss that the optimizer left above the class's U-train
     optimum;
   - the within-step ρ(ψ̂, G) (offline-psi's `rho_psi_g`) on U-val;
   - B0-CF's own G replayed on U-val and on U-train (the reference rows);
   - cf_live's caps and labels, reported and never routed on;
   - the spectrum: numerical rank, λ_max, the smallest kept λ, and the count dropped;
   - a **ridge path** from the same eigendecomposition, `λ_ridge = ρ · mean(λ)` for
     `ρ ∈ {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1}`, with U-val L_MC and U-val cap at each ρ. It is
     descriptive only and is not used to select anything.
5. **Not measured:** the best ranking any member of the class can reach. The least-squares
   optimum minimises squared error over all of the arm's cells. The cap rewards only the bottom
   of each step's live ranking. Only `sᵀ(A c_t + a)` affects that ranking, since the `b·c_t`
   term is constant across slots within a step. A MISSES here means the class's **L_MC-optimal**
   member does not rank well. It does not mean that no member does.

## 4. Controls (any failure: inconclusive, exit 3; numbers still reported)

| id | control | expected |
|---|---|---|
| C1 | Gestalt and Δ files hash to §1's pins (diagnostic A's `load_inputs`), and diagnostic A's `result.json` hashes to §1's pins | true |
| C2 | **Assembly:** 64 entries of `XᵀX` and 32 of `Xᵀy`, drawn with `torch.Generator().manual_seed(seed)` and stratified across all blocks (bilinear, s, c, and their crosses), equal the direct sums over U-train cells. The test is `\|structured − direct\| ≤ 1e-9 · Σ\|terms\|` | true |
| C3 | **Optimality:** (a) the normal-equation residual computed directly from the cells satisfies `‖Xᵀ(Xw − y)‖₂ ≤ 1e-6 · ‖Xᵀy‖₂`; (b) the least-squares head's U-train L_MC is ≤ (1 + 1e-6) × diagnostic A's `train_l_mc_at_selected` for the same seed and arm (a feasible point of the same class; the slack covers diagnostic A's float32 evaluation); (c) the head-evaluated U-train L_MC equals the solver's quadratic-form value to 1e-6 relative | true |
| C4 | Every reported loss, cap and eigenvalue is finite | true |
| C5 | No EVAL document id (995000–997047) is generated or read; every deploy path raises (diagnostic A's `_seal_eval`) | true |

## 5. Compute

- **Measured 2026-10-08 on this machine, before this file:** float64 GEMM about 700 GFLOPS
  (Accelerate), and `eigh` of 4096×4096 at 4.4 s.
- **Per fit (seed × arm):** assembly about 2.4e13 FLOP, under 1 min. `eigh` of 16,640×16,640 is
  about 5 min, extrapolated as n³. Replays are seconds.
- **Total:** 6 fits, about 40 min serial. Seeds run as three parallel cpu-det slot jobs. Peak
  memory is about 12 GB per job.
- **Stop line:** 2 h wall. Past it, the builder stops and reports.

## 6. Readout (routes an OWNER decision)

| U-val fit label | In-sample label | Reading | Goes to Brendan with |
|---|---|---|---|
| POINTS or PARTIAL | any | The class's L_MC optimum ranks usefully. Run 2's optimizer (weight decay and/or the μP step sizes) stopped short of it, in diagnostic A and so in run 2. | Diagnostic B (correction 15: lr and μP multiplier), or an optimizer that reaches this optimum, before any thesis change. |
| MISSES | MISSES | Even exactly fit, and even on its own training documents, the content-only bilinear class's L_MC optimum does not rank G well. Capacity under L_MC is the limit. Diagnostic A's §6 row 2 stands. | The content+age arm or a richer head (both change the thesis), with §3 item 5's caveat: a ranking-targeted fit of this class was not tested. |
| MISSES | POINTS or PARTIAL | The class can express the ranking on documents it has seen, but the fit does not carry to new ones from 932 training documents. | More U data (more B0-CF Δ documents) or a regularised fit. It is the owner's choice. |

## 7. Seen already (declared)

- offline-psi's RESULTS and verification. Diagnostic A's RESULTS, `result.json` files and
  verification, including these selected values:
  - cf_all U-val L_MC 1.517 / 1.457 / 1.675;
  - cf_all U-train L_MC 1.564 / 1.469 / 1.649;
  - constant-mean baseline 1.761 / 1.756 / 1.927;
  - caps 0.274 / 0.406 / 0.148 (cf_all) and 0.523 / 0.547 / 0.316 (cf_live);
  - B0-CF's G on U-val: 0.938 / 0.948 / 0.858.
- **The compute probe in §5:** timing only. No fit and no RSR data were involved.
- No least-squares fit of ψ̂ exists.

## 8. Drafter's expectation (allowed to be wrong)

- **U-val label: FIT-MISSES, more likely than not.** Diagnostic A's train and U-val losses were
  close and had flattened. That suggests the optimizer was not far from the L_MC optimum, so the
  exact solution should lower the loss only modestly.
- **In-sample label: MISSES too**, for the same reason. A cap above diagnostic A's on some seeds
  would not surprise me. A cap reaching 0.8 would.

## Amendment 1 (2026-10-08, before launch, committed alone)

**What the pre-launch smoke showed.** The smoke ran seed 0, cf_all, U-train only, twice. It
computed assembly, C2, the solve and C3, and nothing else. **No cap and no U-val number was
computed.**

- **C2 passed:** worst relative error 2.8e-16 over 96 entries.
- **The solve is deterministic:** identical `w` sha256 both times.
- **Cost:** assembly 30 s, `eigh` 253 s, peak RSS 10.3 GB.
- **C3(b) and C3(c) passed.**
- **C3(a) failed:** `‖Xᵀ(Xw − y)‖ / ‖Xᵀy‖ = 1.06e-6`, against the 1e-6 tolerance.

A follow-up diagnosis on the same data (seed 0, cf_all, U-train only; no cap, no U-val
number) found why:

- **The gestalts are full rank, not low-rank.** U-train's `s` and `c` both have rank 128. 35
  singular values are above 1e-2 × the largest and 123 above 1e-4. So `XᵀX`'s spectrum decays
  smoothly over about 16 decades with no gap. The registered cutoff `p·eps·λ_max` kept 9,897
  of 16,640 directions.
- **The solve is exact on what it keeps.** Projected onto the kept eigenvectors, the residual is
  1.3e-10. All of the 1.06e-6 lies in the dropped directions, where the minimum-norm truncated
  solution leaves the gradient at `−V_dropᵀXᵀy` **by construction**. C3(a) as written tested
  stationarity in the full space, which a truncated solution cannot meet. That was a
  specification error in the control, not a numerical one.
- **Training L_MC depends on the cutoff (seen, declared):**

  | cutoff multiplier | rank | U-train L_MC |
  |---|---|---|
  | ×0.01 | 14,378 | 1.192 |
  | ×0.1 | 12,449 | 1.203 |
  | ×1 (registered) | 9,897 | 1.221 |
  | ×10 | 7,354 | 1.245 |
  | ×100 | 5,415 | 1.274 |
  | ×1e4 | 2,721 | 1.337 |
  | ×1e6 | 793 | 1.450 |

  Diagnostic A's selected φ had a U-train L_MC of 1.564. So "the exact optimum" in §0 and §2
  means the optimum on the directions fp64 normal equations resolve. The directions with the
  smallest variance are left unfit, and they would lower the U-train loss further.

**Changes. None touches a label threshold, the label rule, the readout table, or the
registered cutoff.**

1. **C3(a) is replaced.** The new test is `‖V_keepᵀ Xᵀ(Xw − y)‖₂ ≤ 1e-6 · ‖Xᵀy‖₂`, where
   `Xᵀ(Xw − y)` is still computed directly from the cells. It asks whether the solve is exact
   on the subspace it claims to fit. The dropped-subspace residual `‖V_dropᵀ Xᵀ(Xw − y)‖₂ /
   ‖Xᵀy‖₂` and the full-space value are reported, never gated.
2. **Cutoff sensitivity is added (descriptive, §3 item 4).** For cutoff multipliers ×0.1 and
   ×0.01 (still at least 166 × `eps·λ_max`), report the rank, U-train L_MC, U-val L_MC, U-val
   cap and U-train cap, for both arms.
3. **A caveat rule.** If the U-val label or the in-sample label computed at ×0.01 differs from
   the one at the registered cutoff, RESULTS states both. The route then carries the words
   "cutoff-sensitive". The label and route stay those of the registered cutoff.
4. **A correction to the PM's report to Brendan.** The earlier statement that the gestalts span
   "about 99 dimensions" was wrong. It inferred rank from the truncation count. They are full
   rank, with a steep spectrum.
