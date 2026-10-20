# RESULTS: B0-γ, the hindsight target-ceiling sweep

**Descriptive.** No training, no forward pass, no `rsr.constants.record()`. Model-free residency of
the hindsight rule `TargetPolicy(G_γ)` on the lookahead-room-r2 `D.pt` tensors, U range (already
inspected), `M = 16`, `S = 48`. Pre-registration: `experiments/gamma-ceiling/PREREG.md` (daabdb2,
committed alone before any computation). Written by a model (Claude Opus 5.5, `rsr-researcher`),
2026-10-05/06.

**Class (PREREG §6 reading rule): SIGNAL.** Every γ ∈ {0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0} is
**MISSES** at ckpt3000. γ = 0 and γ = 0.9 had already been seen (B0); the other five had not.
Nothing here changes a constant, γ's ruled value, the spec, or any gate.

## Provenance

| | |
|---|---|
| Command | `PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run --lane cpu-det --slots 1 --wait 3600 -- .venv/bin/python experiments/gamma-ceiling/run.py > <log> 2>&1; rc=$?` |
| Exit code | `rc=0` (read directly; ledger `commands[0].exit_code = 0`) |
| Git SHA (ledger provenance) | `02a7eeb49fd4cb1a1a5546a16a489d26bf6b770d`, `dirty: False`, branch `run/gamma-ceiling` cut from `94a7ceb` |
| Config hash | `6d6d512d1f2b8555d4c8e7da614674018f2f4075ee0a9a2576163c1cdeb53b06` (`runs/gamma-ceiling/manifest.json`) |
| Hardware | Mac Studio, Apple M4 Max, CPU only, `cpu-det` lane, 1 slot; macOS 26.6.2 arm64, Python 3.12.13, torch 2.14.0 |
| Wall time | ledger 2026-10-06T06:47:29Z → 06:50:00Z (151 s); compute per seed 52.4 / 49.5 / 47.4 s (`seconds_per_seed`) |
| Inputs | six `B-ckpt{2500,3000}-seed{0,1,2}.D.pt`, sha256 = T0 manifest lines at start and end; T0 manifest sha256 `33dd493d…565f`; B0 ledger sha256 `39b62832…c2e3` |
| Seeds actually run | `[0, 1, 2]`, 1088 documents each (`n_documents`) |
| Ledger | `runs/gamma-ceiling/ledger.json`; claims `runs/gamma-ceiling/claims.json` |

## Controls (PREREG §7) — all passed

- **C1** manifest: `C1.start.ok = True`, `C1.end.ok = True`.
- **C2** shape / doc ids: no failure (`controls_failed = []`).
- **C3** reproduction of B0, `==` on the float: γ = 0 and γ = 0.9 `hit` and `cap.{point,lo,hi}` at
  **both** checkpoints and all seeds equal `runs/b0-ceilings/ledger.json`
  `B.ckpt{c}.hit.rule_g0/g09` and `B.ckpt{c}.cap.rule_g0/g09.{point,lo,hi}` (16 key pairs, all
  equal); `U.hit.fifo.all` and `U.hit.factfiller.all` equal B0's. **Exact.**
- **C4** three seeds, two checkpoints, seven γ, 1088 documents: passed.

## Headline: ckpt3000, cap = (hit − FIFO) / (factfiller − FIFO), point [95% CI], per seed

References: FIFO hit 0.8059 / 0.8093 / 0.8088; factfiller hit 0.9655 / 0.9653 / 0.9666
(`U.hit.fifo.all`, `U.hit.factfiller.all`). No bootstrap replicate was dropped for a non-positive
denominator in any cell (`cap.dropped = 0` everywhere).

| γ | seed 0 | seed 1 | seed 2 | label | seen before? |
|---|---|---|---|---|---|
| 0    | −0.039 [−0.073, −0.006] | −0.136 [−0.164, −0.107] | +0.234 [+0.204, +0.260] | MISSES | yes (B0) |
| 0.5  | +0.331 [+0.304, +0.358] | +0.116 [+0.092, +0.141] | +0.512 [+0.488, +0.538] | MISSES | no |
| 0.8  | +0.406 [+0.379, +0.434] | +0.210 [+0.186, +0.236] | +0.609 [+0.584, +0.633] | MISSES | no |
| 0.9  | +0.184 [+0.147, +0.220] | +0.134 [+0.103, +0.165] | +0.537 [+0.508, +0.564] | MISSES | yes (B0) |
| 0.95 | −0.077 [−0.125, −0.033] | −0.032 [−0.073, +0.006] | +0.433 [+0.401, +0.463] | MISSES | no |
| 0.99 | −0.308 [−0.361, −0.259] | −0.265 [−0.311, −0.220] | +0.322 [+0.286, +0.357] | MISSES | no |
| 1.0  | −0.359 [−0.413, −0.307] | −0.329 [−0.378, −0.284] | +0.300 [+0.263, +0.335] | MISSES | no |

Keys: `B.ckpt3000.<g>.cap.{point,lo,hi}`, `g ∈ {g0, g05, g08, g09, g095, g099, g1}`; labels
`B.ckpt3000.label`; class `sweep_class`. No γ has a CI lower ≥ 0.80 on any seed; no γ has point
≥ 0.50 on all three seeds (seeds 0 and 1 never reach 0.50 at any γ).

Spread over seeds of the cap point (ledger `sd`): 0.192, 0.198, 0.200, 0.220, 0.282, 0.352, 0.372
for γ = 0 … 1.0. Seed 2 is the highest at every γ.

### ckpt3000 hit rates (all / `gap_2_to_M` / `gap_gt_M`), per seed

FIFO: `gap_2_to_M` 1.0 / 1.0 / 1.0, `gap_gt_M` 0.0 / 0.0 / 0.0. Factfiller: `gap_2_to_M`
0.9737 / 0.9734 / 0.9754, `gap_gt_M` 0.9027 / 0.9009 / 0.9020.

| γ | hit all | hit gap 2..M | hit gap > M |
|---|---|---|---|
| 0    | 0.7997 / 0.7881 / 0.8457 | 0.8794 / 0.9085 / 0.9226 | 0.3380 / 0.1745 / 0.4346 |
| 0.5  | 0.8588 / 0.8274 / 0.8897 | 0.9250 / 0.9437 / 0.9409 | 0.5027 / 0.2705 / 0.6074 |
| 0.8  | 0.8706 / 0.8420 / 0.9049 | 0.9093 / 0.9389 / 0.9291 | 0.6118 / 0.3620 / 0.7239 |
| 0.9  | 0.8352 / 0.8301 / 0.8935 | 0.8422 / 0.9108 / 0.8944 | 0.6353 / 0.3875 / 0.7727 |
| 0.95 | 0.7936 / 0.8043 / 0.8771 | 0.7720 / 0.8685 / 0.8598 | 0.6364 / 0.3839 / 0.7948 |
| 0.99 | 0.7567 / 0.7680 / 0.8597 | 0.7134 / 0.8132 / 0.8298 | 0.6259 / 0.3662 / 0.7972 |
| 1.0  | 0.7487 / 0.7579 / 0.8562 | 0.7015 / 0.7972 / 0.8243 | 0.6210 / 0.3636 / 0.7964 |

Keys: `B.ckpt3000.<g>.hit.{all,gap_2_to_M,gap_gt_M}`. Recorded, not interpreted: above γ = 0.9
the `gap_gt_M` rate is roughly flat while the `gap_2_to_M` rate falls on every seed.

## Secondary: ckpt2500 (unlabelled; the reading rule is defined on ckpt3000 only)

| γ | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| 0    | −0.020 [−0.053, +0.013] | −0.208 [−0.239, −0.178] | +0.219 [+0.189, +0.246] |
| 0.5  | +0.317 [+0.290, +0.343] | +0.074 [+0.048, +0.100] | +0.492 [+0.465, +0.516] |
| 0.8  | +0.350 [+0.320, +0.379] | +0.146 [+0.117, +0.174] | +0.504 [+0.476, +0.531] |
| 0.9  | +0.109 [+0.069, +0.147] | +0.027 [−0.007, +0.062] | +0.355 [+0.323, +0.386] |
| 0.95 | −0.147 [−0.197, −0.101] | −0.142 [−0.185, −0.102] | +0.175 [+0.137, +0.210] |
| 0.99 | −0.349 [−0.406, −0.296] | −0.324 [−0.375, −0.278] | +0.020 [−0.025, +0.060] |
| 1.0  | −0.402 [−0.461, −0.346] | −0.376 [−0.426, −0.329] | −0.014 [−0.058, +0.027] |

Keys: `B.ckpt2500.<g>.cap.{point,lo,hi}`. Hit rates by bucket are in the ledger under
`B.ckpt2500.<g>.hit.*`.

## What came out wrong (against PREREG §9, the author's expectation)

- **Wrong:** I expected the cap to keep rising from γ = 0.9 to 0.95–0.99 on every seed. It does
  the opposite. At ckpt3000 the cap **peaks at γ = 0.8 on all three seeds** (+0.406 / +0.210 /
  +0.609) and **falls monotonically above it**; at γ ≥ 0.95 its point estimate is **below FIFO**
  (negative) on seeds 0 and 1 (seed 1 at γ = 0.95: −0.032, CI upper +0.006). The same shape holds at ckpt2500. γ = 0.9 (B0's value, and run 1's ruled value)
  is past the peak on every seed at both checkpoints.
- **Wrong:** I expected 0.5 and 0.8 to lie between γ = 0 and γ = 0.9. At ckpt3000 γ = 0.8 is
  **above** γ = 0.9 on every seed; γ = 0.5 is above it on seed 0 only (+0.331 vs +0.184; seed 1
  +0.116 vs +0.134, seed 2 +0.512 vs +0.537).
- **Partly wrong:** I expected seed 1 to be the lowest. It is at γ ≤ 0.9; at γ ≥ 0.95 seed 0 is.
- **Right:** no γ is POINTS; predicted class SIGNAL; C3 exact.
- **Disclosure:** before the full run, a 20-document smoke (`--limit 20`, ledger to the session
  scratchpad, not committed) printed all-MISSES / SIGNAL on those 20 documents. It changed nothing:
  the PREREG and code were committed before it.

## How the rows can be read (PREREG §2, stated before the data)

`D.pt` is uncensored (an evicted sentence carries a probed demand, `lookahead-room/run.py:513,
517, 534`), but every `D[t][i]` is a one-at-a-time demand-if-resident in FIFO's rollout, not demand
in the hindsight rule's own memory. Large-γ rows therefore sum FIFO-world counterfactuals furthest
from the decision, and mix resident-type and probe-type readings in an age-correlated proportion.
This file does not go beyond the reading rule.

## Amendment 1: age-shape of the target (DIAGNOSTIC)

**Never gating; does not change any label or the class.** Amendment 1 (c4c1a25) was requested by
the PM and committed **after** the main run above had been seen, and **before** any age-shape
number was computed. No expectation was registered for these values.

| | |
|---|---|
| Command | `PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run --lane cpu-det --slots 1 --wait 3600 -- .venv/bin/python experiments/gamma-ceiling/ageshape.py > <log> 2>&1; rc=$?` |
| Exit code | `rc=0` (ledger `commands[0].exit_code = 0`) |
| Git SHA | `a0547a61be4d445b8f42e93a7e32b01176e4dc6f`, `dirty: False` |
| Config hash | `78138ff9154254e75c45d090d802df50b564a8191f186920f84cf1ec1280f4fa` |
| Wall time | 2026-10-06T06:53:50Z → 06:56:41Z; 61 / 56 / 55 s per seed |
| Ledger | `runs/gamma-ceiling-ageshape/ledger.json` (claims beside it) |
| Controls | C1 ok at start and end; **A1-C pass-through exact**: the recorded rule's all-query hit equals the main ledger's `B.ckpt{c}.<g>.hit.all` for every γ × ckpt × seed (`controls_failed = []`) |

Statistic: per eviction step of the hindsight rule's own memory (16 slots), Spearman ρ of `G_γ`
with slot age `t − written_at`; lower median over the 34 816 steps (1088 documents × 32 steps) per
seed; per-document bootstrap 95% CI (B0's replicates). **ρ > 0: older slots carry larger `G`;
ρ < 0: newer slots do.** `n_steps_undefined = 0` in every cell. No threshold exists on ρ.

### ckpt3000 — cap (from above) beside the age-shape of the target (DIAGNOSTIC)

| γ | label | age-shape of the target (DIAGNOSTIC), median ρ [95% CI]: seed 0 · seed 1 · seed 2 |
|---|---|---|
| 0    | MISSES | +0.318 [+0.312, +0.326] · −0.076 [−0.088, −0.068] · +0.100 [+0.094, +0.109] |
| 0.5  | MISSES | +0.338 [+0.329, +0.344] · +0.009 [−0.003, +0.018] · +0.126 [+0.118, +0.135] |
| 0.8  | MISSES | +0.338 [+0.329, +0.347] · +0.165 [+0.156, +0.174] · +0.168 [+0.159, +0.176] |
| 0.9  | MISSES | +0.321 [+0.312, +0.332] · +0.224 [+0.215, +0.232] · +0.176 [+0.168, +0.188] |
| 0.95 | MISSES | +0.294 [+0.282, +0.306] · +0.247 [+0.238, +0.256] · +0.174 [+0.165, +0.185] |
| 0.99 | MISSES | +0.268 [+0.253, +0.279] · +0.247 [+0.238, +0.256] · +0.171 [+0.159, +0.182] |
| 1.0  | MISSES | +0.262 [+0.247, +0.274] · +0.247 [+0.238, +0.256] · +0.171 [+0.159, +0.182] |

### ckpt2500 (secondary)

| γ | median ρ [95% CI]: seed 0 · seed 1 · seed 2 |
|---|---|
| 0    | +0.362 [+0.356, +0.371] · +0.018 [+0.012, +0.026] · +0.088 [+0.079, +0.094] |
| 0.5  | +0.382 [+0.374, +0.388] · +0.074 [+0.065, +0.082] · +0.106 [+0.100, +0.115] |
| 0.8  | +0.391 [+0.385, +0.400] · +0.203 [+0.197, +0.212] · +0.162 [+0.150, +0.171] |
| 0.9  | +0.371 [+0.362, +0.379] · +0.256 [+0.250, +0.265] · +0.176 [+0.168, +0.188] |
| 0.95 | +0.341 [+0.329, +0.353] · +0.268 [+0.259, +0.276] · +0.176 [+0.165, +0.188] |
| 0.99 | +0.318 [+0.306, +0.329] · +0.265 [+0.256, +0.274] · +0.174 [+0.162, +0.185] |
| 1.0  | +0.312 [+0.300, +0.324] · +0.265 [+0.256, +0.271] · +0.168 [+0.156, +0.182] |

Keys: `B.ckpt{c}.<g>.age_rho.{median,lo,hi,mean,n_steps,n_steps_undefined}`.

**Looks wrong, is not (checked):** several cells repeat exactly (ckpt3000 seed 1 at γ = 0.95,
0.99 and 1.0 all +0.247 [+0.238, +0.256]; seed 2 at 0.99 and 1.0). Spearman ρ over 16 slots takes a
discrete set of values, so a median and its bootstrap quantiles land on the same lattice point; the
per-cell **means** differ (`age_rho.mean`, e.g. seed 1: 0.218 / 0.229 / 0.229 — the last two
also equal to three decimals; seed 0: 0.277 / 0.251 / 0.246). Recorded, not interpreted: the
residents are the rule's own survivors, so ρ is measured on a set the rule selected by `G`.
