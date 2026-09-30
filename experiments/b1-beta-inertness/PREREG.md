# B1: β inertness at real scale — PREREG

**Written 2026-09-30 by a model** (Claude Opus 5.5, an `rsr-researcher` session). **Not written by
Brendan.** Committed **alone**, before any B1 code or data. At the time of writing no `runs/b1*`
exists in the main checkout, in any worktree, or in this branch's tree (the check and its literal
output are in this commit's message).

**Class: MEASURE-THEN-DECIDE.** The output is evidence for ADR-0009 **L7 and L8 together**
(PLAN-v4 §2B, L156–166). It is never a build, a correction entry, a registry change or an ADR edit.
No agent edits ADR-0009. The registry entry `beta` is not read and not touched: β here is a sweep
value named by PLAN-v4, not a registry constant.

## 1. The question

ADR-0009 (Context, "β is inert") argues that with `L = L_NTP + β·L_MC` and the isolation rule
(only `φ = {W, u}` receives `L_MC`'s gradient, and `φ` receives nothing from `L_NTP`), β scales the
whole of `φ`'s gradient by a constant, and AdamW's update `m̂/(√v̂ + ε)` is invariant to a constant
gradient scale **except through ε** (and through anything else that is not scale-invariant). Its
evidence is a d = 32 scratch script with unit-norm random inputs.

**B1 asks:** at the real width, with real gestalts `s_i`, real contexts `c_t` and the real scale of
the MC return `G`, does changing β ∈ {0.01, 0.1, 1} change `φ`, or the eviction argmin `φ` produces,
under each of four optimizer treatments of `φ` and at two values of ε? Where it does, through which
mechanism (ε, clip, coupled L2)?

**Falsifier addressed.** Not a spec falsifier (§1's list). B1 tests the premise of ADR-0009 L7
("β is inert under AdamW + isolation"), which is itself a finding that contradicts spec :280 and
:607. A result of LIVE in arm 1 at ε = 1e-8 would falsify L7's premise at real scale.

## 2. Settings (real scale)

| setting | value | source |
|---|---|---|
| substrate | fresh-stream **arm B, ckpt3000**, seeds 0, 1, 2 | the headline checkpoint (PLAN-v4; B2 §2); loaded read-only by B2's `load_checked` (sha256-pinned `CKPT_SHA256`) |
| `d_model` | 128 (read from `runs/fresh-stream/manifest.json` key `d`, asserted against the checkpoint) | B2 `load_checked` |
| `M`, `S`, `L` | 16, 48, 64 | B2 `M_STEPS`, `S_STEPS`, `L_TOKENS` (S0-03 config) |
| capture | **B2's `capture_doc`, imported, not copied**, FIFO, eval mode, `no_grad`, `ranks=(0,)` | `experiments/b2-psi-probe/run.py` |
| rows | B2 rowset **C**: at each step `t ∈ [1, S)`, every FIFO-resident sentence `i < t` | `row_index(cap, "C", m)` — the slots that are live, i.e. what online `L_MC` sums over |
| target | B2 arm **C at γ = 0.9**: `G = returns(literal(D[0], resident), 0.9)`, with `r` = `LR.r_of` (norm-weighted, gated, fill-rescaled) | `target_matrix(cap, "C", 0.9)` |
| training docs | seed `s`'s generator documents **920000 … 920255** (256 docs; the head of B2's FIT_TRAIN) | declared reuse, below |
| argmin docs | seed `s`'s documents **936000 … 936063** (64 docs; the head of B2's FIT_VAL) | declared reuse, below |

**Declared reuse.** Both ranges are B2's. Documents 920000–920255 are among the 264 B2 fitted on;
936000–936063 are among B2's FIT_VAL. B1 evaluates no retrieval quantity and makes no claim about
ψ̂'s accuracy, so reusing them contaminates nothing B1 reports; it also makes "gestalt statistics and
G scale from B2's capture" literally B2's documents through B2's code. The E0d-reserved range
(900000–920000) and B2's EVAL range are not touched.

**`ψ̂` is the real head**, `rsr.retention.value_head.BilinearValueHead(d_model=128, base_width=128)`
(multipliers `1/d` bilinear, `1/(2d)` linear), fp32, initialised from a **dedicated CPU generator
seeded `1000 + s`** (ADR-0009 L11's pattern), so the init is identical across every β, arm and ε of
seed `s`. Its inputs are the captured `s_i`, `c_t` (detached clones; B2's `_stopgrad`), so no
gradient can reach the transformer; the runner asserts `requires_grad` is False on both.

## 3. The loss and the optimizer steps

- `L_MC` per optimizer step: over a minibatch of **16 documents** (fresh-stream's `batch`), for every
  `(doc, t)` with `t ∈ [1, S)`, **sum over resident slots** of `(ψ̂(s_i, c_t) − G[t, i])²`, then
  **mean over `(doc, t)`**. This is ADR-0009 L9's proposed reduction (sum over live slots, mean over
  (row, t)), unweighted. `φ`'s loss is `β · L_MC`.
- **Steps: K = 3000 optimizer steps** per run (the substrate's own run length, 3000 steps). Minibatch
  order: 16 documents drawn per step without replacement within an epoch, from a **dedicated CPU
  generator seeded `2000 + s`**, so **every β, arm and ε of seed `s` sees the identical minibatch
  sequence**. Only β, the arm and ε differ between the runs being compared.
- Snapshots of `φ` (and the argmin read) at steps **100, 300, 1000, 3000**.

## 4. φ's optimizer settings — what is real, what is chosen

From `src/rsr/train/loop.py` and `src/rsr/mup/param_groups.py` at this branch:

| setting | value | status |
|---|---|---|
| optimizer | `torch.optim.AdamW` | real (`loop.py:434`) |
| betas | (0.9, 0.95) | real (`loop.py:434`) |
| weight decay | 0.01, decoupled | real: **global** 0.01 (`loop.py:434`); `φ` is not yet in `build_param_groups` (`loop.py:433` passes `value_head=None`), so a `φ` group would inherit it. ADR-0009 L8 proposes 0.0; that is an option, not what the code does |
| `φ` LR | `base_lr / m` = 1e-3 / (128/128) = **1e-3** | real rule (`param_groups.py`, `VALUE_HEAD_GROUP`), real `base_lr` (fresh-stream `lr` = 0.001) |
| ε | swept, {1e-8, 1e-12}; 1e-8 is torch's default and what `loop.py` uses | PLAN-v4 |
| transformer clip | `clip_grad_norm_(model.parameters(), 1.0)` | real (`loop.py:578`) |
| **`φ` clip threshold** | **1.0** | **NOT DEFINED for `φ` today. Chosen, labelled as chosen:** the transformer's threshold, the only clip value in the code |
| **coupled-L2 coefficient λ** | **0.01** | **NOT DEFINED today. Chosen:** the same number as the decoupled wd, so arm 4 differs from arm 1 only in coupling |
| transformer grad norm (joint clip only) | fresh-stream **arm B seed `s`**'s logged pre-clip `grad_norm` (`B/seed{s}/heartbeat.jsonl`), step `2000 + (k mod 1000)` at B1 step `k` | real, read-only. Already inspected before this PREREG (median ≈ 12, every step of 2000–2999 > 1): declared |

## 5. The arms (PLAN-v4), each at ε ∈ {1e-8, 1e-12}, each at β ∈ {0.01, 0.1, 1}

1. **`decoupled_wd`** — {decoupled wd, no φ clip}: AdamW(lr 1e-3, betas (0.9, 0.95), wd 0.01, ε), gradient
   `β·∇L_MC`, no clip.
2. **`phi_clip`** — as 1, plus `clip_grad_norm_(φ, 1.0)` on `φ` alone, after `β·∇L_MC` is formed.
3. **`joint_clip`** — as 1, but `φ`'s gradient is scaled by the **joint** clip coefficient
   `min(1, 1.0 / √(g_T,k² + ‖β∇L_MC‖²))`, where `g_T,k` is the logged transformer norm (§4). This is
   exactly what `clip_grad_norm_` over `model.parameters() ∪ φ` would apply, because `φ`'s gradient does
   not depend on the transformer's and vice versa. The same coefficient would also scale the
   transformer's gradient; B1 records it (§6 m4) as the size of the `L_MC → transformer` channel a
   joint clip opens.
4. **`coupled_l2`** — `torch.optim.Adam(lr 1e-3, betas (0.9, 0.95), weight_decay=0.01, ε)`: the L2
   term is added to the gradient (`β·∇L_MC + 0.01·φ`) before the moments, no clip.

**Controls (not arms):**

- **P (positive control), L7(a)'s reading:** arm 1 with β applied as `φ`'s **LR multiplier**
  (lr = β·1e-3, gradient unscaled), at both ε. PLAN-v4's note applies: under L7(a) β also scales the
  per-step decoupled decay `lr·wd`, and P includes that. **P must classify LIVE** (§7); if it does not,
  the metrics cannot see a live knob and B1 is **VOID**.
- **N (negative control, determinism):** arm 1, ε = 1e-8, β = 1, run a second time for each seed.
  `max|Δφ|` must be **exactly 0** and argmin agreement **exactly 1**. If not, B1 is **VOID** (every
  Δ below would include run-to-run noise).
- **I (context only, not gating):** arm 1, ε = 1e-8, β = 1, `φ` initialised from seed `1100 + s`.
  Its `max|Δφ|` and agreement against the seed-`1000 + s` run give the scale of "a different `φ`",
  for reading the thresholds. No rule reads it.

Total runs: 4 arms × 2 ε × 3 β × 3 seeds = 72; P adds 2 ε × 2 β × 3 seeds = 12 (its β = 1 is arm 1's
β = 1 run, not re-run); N adds 3; I adds 3. **90 runs.**

## 6. Metrics — all against **β = 1 of the same seed, arm and ε**

For each (arm, ε, β ∈ {0.01, 0.1}, seed):

- **m1 `max|Δφ|`**: `max |φ_β − φ_1|` over every entry of `W` and `u`, at K = 3000, and its
  **relative** form `max|Δφ| / max|φ_1|`. Also recorded at 100, 300, 1000 (descriptive).
- **m2 argmin agreement**: over the 64 argmin documents, at every step `t` where FIFO memory is full
  (`|resident[t]| = M`, i.e. `t ∈ [16, 48)`: 32 decisions per document, **2048 per seed**), the fraction
  of `(doc, t)` at which `argmin_i ψ̂_β(s_i, c_t)` over resident `i` equals `argmin_i ψ̂_1(s_i, c_t)`
  (ties to the lowest `i`). This is **open-loop** on the FIFO trajectory: the decision each `φ` would
  make in the same memory state. A closed-loop rollout would compound any disagreement; open-loop
  does not, and is declared as the measure. `b` is not applied (it never enters `ψ̂`; with `b` off,
  the z-score is monotone and leaves the argmin unchanged).
- **m3 `ε/√v̂` distribution**: at K = 3000, for every `φ` coordinate, `ε / √(v / (1 − 0.95^K))` from
  the optimizer's `exp_avg_sq`, per β. Reported: min, p1, p50, p99, max, and the fractions > 0.01
  and > 0.1. (For arm 4, `v` is of the L2-augmented gradient; that is what Adam divides by.)
- **m4 (arms 2 and 3 only)**: arm 2 — the fraction of steps where `‖β∇L_MC‖ > 1.0` (the clip binds),
  per β. Arm 3 — per β, the joint coefficient's relative deviation from the transformer-only
  coefficient `min(1, 1/g_T,k)`, max and median over steps: `|c_joint − c_T| / c_T`. Nonzero means the
  transformer's update depends on `L_MC` through the clip.
- Per-run: `‖∇L_MC‖` (unscaled) at steps 1, 100, 1000, 3000, and `L_MC` on the training minibatch at
  the same steps, so the G scale and fit are on record.

## 7. "Inert", defined before the data

Per cell (arm, ε), reading **both** β ∈ {0.01, 0.1} against β = 1, on **every** seed:

- **INERT**: min argmin agreement **≥ 0.99** AND max relative `max|Δφ|` **≤ 0.01**.
- **LIVE**: min argmin agreement **< 0.95** OR max relative `max|Δφ|` **> 0.10**.
- **INTERMEDIATE**: otherwise.

The same rule classifies P.

**Why these numbers.**

- **relΔ ≤ 0.01.** ADR-0009's own "inert" evidence is `max|φ(0.01) − φ(1)| = 5.8e-4` against
  `max|φ| = 0.674`, a relative 8.6e-4 at ε = 1e-8 (ADR L84–90). 0.01 is an order of magnitude above
  that, so the d = 32 result would be INERT under this rule, and a real-scale result an order of
  magnitude worse still would be too. relΔ > 0.10 is a change in `φ` of the same order as the change
  that decides a close argmin.
- **agreement ≥ 0.99.** The decision is what a β sweep would move. B2's decision-level contrasts
  (FIT_VAL accuracy differences between arms) are of order 0.01–0.08. If ≥ 99 % of open-loop
  decisions are identical, β can move at most 1 % of evictions in the same state, which is below the
  smallest contrast B2 resolved. At 2048 decisions per seed, 0.99 is ≤ 20 differing decisions.
  < 0.95 (> 102 differing decisions per seed) is a sweep that visibly changes the policy.
- **"every seed", "both β"**: β = 0.01 is where ε matters most, so a cell is INERT only if the extreme
  is.
- The I control gives the scale of an unrelated `φ` for reading these numbers; it does not set them.

## 8. VOID and exit codes

B1 is **VOID** (exit 3, "did not measure") if any of: the T0 manifest check at start or end is not
rc 0 with 435 OK; a checkpoint sha256 mismatches (B2's `load_checked` raises); a capture fails B2's
controls (identity ≤ 1e-6, sum ≤ 1e-5, every sentence written); vocabulary closure fails; any loss or
`φ` entry is non-finite; N is not bit-identical; P is not LIVE. A crash is `CRASHED`. Otherwise
exit 0 and the per-cell table is the result, **whatever it says**. No threshold, arm, ε, β or step
count changes after data exists; if one would need to, the run HALTs and reports.

## 9. Expected (before the data)

A rough gradient-scale estimate (unit-norm `s`, `c`; `1/d` multiplier; up to 16 slots; `G` of order
0.5) puts `√v̂` for `φ` at roughly 1e-5–1e-3 at β = 1, so 1e-7–1e-5 at β = 0.01: within two or three
decades of ε = 1e-8. This is arithmetic, not measured.

- **Arm 1 (decoupled wd):** ε = 1e-12 **INERT**. ε = 1e-8: **INERT more likely than not (~60 %)**;
  otherwise INTERMEDIATE driven by β = 0.01's low-`v̂` coordinates. LIVE unlikely (~10 %).
- **Arm 2 (φ clip at 1.0):** identical to arm 1, because `‖∇L_MC‖` is expected to stay well below 1
  (clip never binds). If it binds at β = 1 but not at 0.01, LIVE.
- **Arm 3 (joint clip):** `g_T ≈ 12 ≫ ‖∇L_MC‖`, so the joint coefficient ≈ `1/g_T,k`, a time-varying
  but β-independent rescaling, to first order. `φ`: **INERT** at ε = 1e-12, and like arm 1 or somewhat
  worse at 1e-8 (the extra 1/12 shrinks `√v̂` toward ε). m4: the coefficient deviation is nonzero but
  tiny (≲ 1e-4 relative). **Any nonzero value is the channel L8 forbids.**
- **Arm 4 (coupled L2):** **LIVE** at both ε: β sets the ratio of the data gradient to the L2 term.
- **P:** LIVE. **N:** bit-identical.

## 10. What B1 does not do

- It does not train the transformer and does not measure any retrieval or LM quantity.
- It does not choose between L7 (a)/(b)/(c) or L8's options; it reports which optimizer treatments
  leave β inert and why.
- It does not measure closed-loop policy divergence, `T_warm`, `b`, or the online (non-repeating)
  stream: the 256-document pool repeats (≈ 187 epochs over 3000 steps), which **lets `φ` fit further
  and shrinks gradients late**, i.e. it makes ε matter more than a non-repeating stream would. The
  direction of that bias is toward finding β live; declared.
