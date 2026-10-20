# RESULTS: B0-CF, the counterfactual-return ceiling

**Descriptive** (Tier D). No training, no gradient, no `rsr.constants.record()`. Forward passes in
eval mode on the frozen fresh-stream arm B checkpoints measure the counterfactual demand Δ under
FIFO on the U range (already inspected for r_i). A hindsight rule `TargetPolicy(G^Δ_γ)` is then
scored by model-free residency (`M = 16`, `S = 48`). Pre-registration:
`experiments/cf-ceiling/PREREG.md` (**0c5320e**, committed alone before any Δ on U). Written by a
model (Claude Opus 5.5, `rsr-researcher`), 2026-10-06.

**Class (PREREG §7 reading rule): CF-POINTS.** At ckpt3000, γ = 0.9 is **POINTS**, γ = 0.8 is
**PARTIAL**, and γ = 0 and γ = 0.5 are **MISSES**. Under the pre-registered consequence, CF-POINTS
puts option C3 (a LOO-derived target, spec §3.2.1, owner-only) in front of Brendan for RSR run 2.
That is an input to an owner decision, not a decision. Nothing here changes a constant, γ's ruled
value, the retention target, the spec or any gate.

## Provenance

| | |
|---|---|
| Command | `PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run --lane cpu-det --slots 1 --wait 3600 --job-id cf-ceiling-run-a3 -- .venv/bin/python experiments/cf-ceiling/run.py > <log> 2>&1; rc=$?` |
| Exit code | `rc=0` (read directly; ledger `commands[0].exit_code = 0`) |
| Git SHA (ledger provenance) | `03859e2b7709db6f466809bfa45bb9910dbeb54d`, `dirty: False`, branch `run/cf-ceiling` cut from f31cb8b |
| Full suite at that exact head | `passed=2109 failed=0 skipped=0 errors=0`, rc 0 (run by the launcher immediately before the run) |
| Config hash | `d8b05c1e0e4b45566d35955b07ba8d9d343dbb4aac6335eaa7447c14eebee506` (`runs/cf-ceiling/manifest.json`) |
| Hardware | Mac Studio, Apple M4 Max, CPU only, `torch.set_num_threads(1)`, `cpu-det` lane, 1 slot; macOS 26.6.2 arm64, Python 3.12.13, torch 2.14.0 |
| Launch trigger | RSR run 1's `finalize.rc` (PM's ordering change), seen 2026-10-06T10:17:32Z |
| Wall time | ledger 10:26:54Z → 12:33:09Z (**2 h 06 min**); Δ measurement 1209–1210 s per (checkpoint, seed) (`B.ckpt{c}.measure_seconds`); scoring 104 s per seed |
| Inputs | six `ckpt-00{3000,2500}.pt` and six lookahead-room-r2 `D.pt`; sha256 = T0 manifest lines at start and end (C1) |
| Δ files | `runs/cf-ceiling/raw/B-ckpt{c}-seed{s}.delta.pt` (gitignored; sha256 in ledger notes `B.ckpt{c}.seed{s}.raw`, none reused) |
| Seeds actually run | `[0, 1, 2]`, 1088 documents each, both checkpoints |
| Ledger | `runs/cf-ceiling/ledger.json`; claims `runs/cf-ceiling/claims.json` |

## Feasibility (PREREG §2), and how it turned out

- E0d's saved Δ covers `[262144, 263168)`, not U, so it was not reused. `loo.py` on f31cb8b is
  byte-identical to `run/e0d`'s, so no merge was needed.
- Probe (2767496; 20 non-U docs, 1 thread, under run 1's load): **1.747 s/doc**. That gave ≈ 1.6 h
  per checkpoint, ≈ 3.2 h for both, and a batch-32 bound of ≈ 4.2 h. **Full U, both checkpoints, no
  subsample.**
- **Measured in the run** (idle machine, batch 32): 1210 s per 1088 docs = **1.11 s/doc**, 2 h 06 min
  in all. That is faster than both the probe and the bound.

## Controls (PREREG §6): all passed (`controls_failed = []`)

- **C1** manifest over 6 checkpoints + 6 D.pt: `C1.start.ok = True`, `C1.end.ok = True`.
- **C2** D.pt ids/shape and Δ / Δshift shape: no failure.
- **C3 (the PM's control): exact.** The r_i arms γ = 0 and γ = 0.9 reproduce
  `runs/gamma-ceiling/ledger.json` with `==` on the float: hit (all, `gap_2_to_M`, `gap_gt_M`) and
  cap point/lo/hi at both checkpoints, all seeds (24 key pairs), plus `U.hit.fifo.all` and
  `U.hit.factfiller.all`. For example, ckpt3000 γ = 0.9 gives +0.184 / +0.134 / +0.537, as in the γ sweep.
- **C4** instrument, every batch: probe-loop live loss == loo's (bit-exact), residency == loo's,
  FIFO memory `range(t − M, t)`, identity-probe worst |Δloss| = **0.0** on every seed and checkpoint
  (`B.ckpt{c}.identity_worst`).
- **C5** full-memory resident donor coverage: 0.99905 / 0.99916 / 0.99900 (both checkpoints, the
  same donors). Missing cells (imputed 0): 529 / 468 / 558 at full memory. By kind over all steps:
  assert 11 / 26 / 24, query 0, filler 1344 / 1024 / 1216 (`B.ckpt{c}.missing.*`).
- **C6** sizes, and the age-shape pass-through's hit equals the plain `TargetPolicy` hit: passed.

## Headline: ckpt3000, primary arm `G^Δ_γ`, cap = (hit − FIFO) / (factfiller − FIFO), point [95% CI]

References: FIFO hit 0.8059 / 0.8093 / 0.8088; factfiller 0.9655 / 0.9653 / 0.9666; oracle
(`OraclePolicy(discounted_demand, 0.97)`, reported only) 1.0 / 1.0 / 1.0, cap +1.217 / +1.223 /
+1.212. `cap.dropped = 0` in every cell.

| γ | seed 0 | seed 1 | seed 2 | label |
|---|---|---|---|---|
| 0   | −0.780 [−0.837, −0.727] | −0.653 [−0.712, −0.599] | −0.835 [−0.898, −0.776] | MISSES |
| 0.5 | +0.045 [+0.018, +0.073] | +0.109 [+0.080, +0.137] | +0.070 [+0.037, +0.100] | MISSES |
| 0.8 | +0.620 [+0.598, +0.641] | +0.653 [+0.633, +0.675] | +0.587 [+0.566, +0.610] | PARTIAL |
| 0.9 | +0.916 [+0.895, +0.937] | +0.955 [+0.935, +0.976] | +0.874 [+0.852, +0.896] | **POINTS** |

Keys: `B.ckpt3000.cf.<g>.cap.{point,lo,hi}`; labels `B.ckpt3000.label`; class `cf_class`.
Spread over seeds of the cap point (ledger `sd`): 0.094 / 0.032 / 0.033 / 0.041 for γ = 0 / 0.5 / 0.8 / 0.9.

Side by side with the r_i return at the same γ (γ sweep, reproduced here as C3 for γ ∈ {0, 0.9}):
γ = 0 r_i −0.039 / −0.136 / +0.234 against Δ −0.780 / −0.653 / −0.835; γ = 0.9 r_i +0.184 / +0.134 /
+0.537 against Δ +0.916 / +0.955 / +0.874.

### ckpt3000 hit rates, primary (all / `gap_2_to_M` / `gap_gt_M`), per seed

FIFO: `gap_2_to_M` 1.0, `gap_gt_M` 0.0. Factfiller: `gap_2_to_M` 0.9737 / 0.9734 / 0.9754, `gap_gt_M`
0.9027 / 0.9009 / 0.9020.

| γ | hit all | hit gap 2..M | hit gap > M |
|---|---|---|---|
| 0   | 0.6815 / 0.7075 / 0.6770 | 0.6901 / 0.7080 / 0.6644 | 0.3099 / 0.3771 / 0.3579 |
| 0.5 | 0.8131 / 0.8263 / 0.8198 | 0.8941 / 0.8972 / 0.8906 | 0.3621 / 0.4099 / 0.3991 |
| 0.8 | 0.9049 / 0.9112 / 0.9014 | 0.9684 / 0.9687 / 0.9670 | 0.6067 / 0.6320 / 0.5873 |
| 0.9 | 0.9521 / 0.9583 / 0.9467 | 0.9887 / 0.9862 / 0.9853 | 0.7877 / 0.8244 / 0.7669 |

### ckpt3000 age-shape of the target (DIAGNOSTIC), median within-step Spearman ρ(G, age) [95% CI]

ρ > 0: older slots carry larger G. ρ < 0: newer slots do. No threshold exists. 34 816 steps per
seed, none undefined.

| γ | seed 0 · seed 1 · seed 2 | mean |
|---|---|---|
| 0   | −0.162 [−0.168, −0.156] · −0.150 [−0.156, −0.141] · −0.185 [−0.191, −0.179] | −0.139 / −0.125 / −0.164 |
| 0.5 | −0.279 [−0.285, −0.271] · −0.256 [−0.262, −0.247] · −0.309 [−0.315, −0.303] | −0.251 / −0.226 / −0.278 |
| 0.8 | −0.247 [−0.256, −0.241] · −0.212 [−0.221, −0.203] · −0.297 [−0.303, −0.288] | −0.229 / −0.194 / −0.272 |
| 0.9 | −0.197 [−0.206, −0.188] · −0.156 [−0.168, −0.147] · −0.250 [−0.259, −0.241] | −0.186 / −0.145 / −0.231 |

Recorded, not interpreted: the sign is opposite to the r_i target's (γ sweep Amendment 1, +0.10 to
+0.34). As there, ρ is measured on the rule's own survivors.

### Sensitivity arm `G^{Δshift}_γ` (DIAGNOSTIC, unlabelled): ckpt3000 cap

| γ | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| 0   | −0.779 [−0.837, −0.726] | −0.649 [−0.709, −0.595] | −0.848 [−0.910, −0.787] |
| 0.5 | +0.035 [+0.007, +0.065] | +0.122 [+0.091, +0.151] | +0.081 [+0.049, +0.112] |
| 0.8 | +0.635 [+0.611, +0.657] | +0.680 [+0.659, +0.702] | +0.600 [+0.579, +0.622] |
| 0.9 | +0.962 [+0.940, +0.983] | +0.999 [+0.980, +1.020] | +0.912 [+0.890, +0.933] |

Its age-shape at γ = 0.9: −0.165 / −0.124 / −0.212. Its hits and the rest are in the ledger under
`B.ckpt{c}.cfshift.*`.

### Δ by class (PREREG §5), ckpt3000, full-memory steps, mean (sd) [n], seed 0 · 1 · 2

| class \| reading | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| querying \| resident | 1.1712 (0.3725) [10 838] | 1.1404 (0.3555) [10 866] | 1.1940 (0.4211) [10 762] |
| querying \| evicted  | 0.6305 (0.2054) [3 905] | 0.6576 (0.2073) [3 845] | 0.5841 (0.2810) [3 836] |
| pending \| resident  | −0.0001 (0.0752) | 0.0001 (0.0739) | 0.0002 (0.0777) |
| pending \| evicted   | −0.0086 (0.0734) | −0.0073 (0.0719) | −0.0083 (0.0721) |
| answered \| resident | 0.0104 (0.0556) | 0.0075 (0.0527) | 0.0104 (0.0580) |
| answered \| evicted  | −0.0088 (0.0719) | −0.0076 (0.0705) | −0.0086 (0.0713) |
| filler \| resident   | 0.0044 (0.0385) | 0.0033 (0.0324) | 0.0058 (0.0465) |
| filler \| evicted    | −0.0046 (0.0619) | −0.0089 (0.0706) | −0.0035 (0.0641) |

Key: `B.ckpt3000.seed{s}.delta_by_class` (counts for every row are there). Recorded: away from the
query step, evicted readings average slightly **below 0** and resident readings slightly **above
0**. This is the level offset between the two reading kinds that PREREG §3 named as a possible
age-correlated term.

## Secondary: ckpt2500 (unlabelled; the rule is defined on ckpt3000 only)

| γ | primary seed 0 | seed 1 | seed 2 | sensitivity seed 0 · 1 · 2 (point) |
|---|---|---|---|---|
| 0   | −0.785 [−0.842, −0.731] | −0.698 [−0.758, −0.643] | −0.831 [−0.898, −0.770] | −0.794 · −0.694 · −0.827 |
| 0.5 | +0.031 [+0.003, +0.059] | +0.079 [+0.048, +0.108] | +0.074 [+0.043, +0.104] | +0.036 · +0.091 · +0.075 |
| 0.8 | +0.607 [+0.585, +0.628] | +0.633 [+0.611, +0.655] | +0.602 [+0.580, +0.625] | +0.619 · +0.655 · +0.612 |
| 0.9 | +0.908 [+0.888, +0.929] | +0.931 [+0.912, +0.951] | +0.885 [+0.863, +0.909] | +0.952 · +0.969 · +0.909 |

The r_i control at ckpt2500 reproduces the γ sweep exactly (γ = 0.9: +0.109 / +0.027 / +0.355).
Age-shape at γ = 0.9, primary: −0.135 / −0.156 / −0.212.

## What came out wrong (against PREREG §9, the author's expectation)

- **Wrong class.** I predicted CF-MIXED (low confidence, CF-POINTS next). It is **CF-POINTS**:
  γ = 0.9 clears CI lower ≥ 0.80 on all three seeds (lowest lower bound 0.852, seed 2).
- **Wrong:** I expected the Δ caps to be above the r_i caps at the same γ on every seed at γ ≥ 0.5.
  At γ = 0.5 they are far **below** (+0.045 / +0.109 / +0.070 against r_i +0.331 / +0.116 / +0.512).
  At γ = 0.8 they are above on seeds 0 and 1 but **below on seed 2** (+0.587 against +0.609). Only
  at γ = 0.9 are they above on every seed.
- **Wrong in size:** I expected γ = 0 to MISS. It does, but its cap is **far below FIFO**
  (−0.78 / −0.65 / −0.84), much lower than the r_i γ = 0 cap.
- **Right:** the cap rises with γ, monotone over the four γ on every seed and both checkpoints, and
  C3 was exact. The γ set stops at 0.9, and the cap is still rising there. Nothing above 0.9 was
  measured, and no claim about it is made.
- **Process errors, before the run (none touched a number):**
  1. The first full-suite gate at b9bc032 failed 2 tests. One was mine:
     `test_every_converted_checker_exits_through_the_protocol`, because `probe.py` exited with
     literals. The other was inherited: `test_the_committed_index_is_current`, because f31cb8b
     added an outbox report without regenerating the index. Fixed in 1e982ee (index, generated by
     the tool) and 03859e2 (probe exit plumbing only; the measurement is unchanged). The suite
     then passed at 03859e2 and the run launched from that head.
  2. The second launch attempt reused job id `cf-ceiling-suite` and the slot refused it (exit 3,
     "job record already exists"). Relaunched with new job ids. The old record was not touched.
- **Disclosure:** a 32-document smoke (`--limit 32`, ledger in the session scratchpad, not
  committed) ran before launch, at b9bc032's parent code (6ee8551). I read only its rc, controls and
  coverage, not its caps or labels. Its final JSON line (the smoke's labels) was printed to its log
  and filtered out of what I read. The PREREG and code were committed before it.

**Looks wrong, is not (checked):** at ckpt2500 γ = 0 the age-shape median and CI are identical on
all three seeds (−0.144 [−0.150, −0.138]). Spearman ρ over 16 slots takes a discrete set of values,
so the lower median and its quantiles land on the same lattice point (the γ sweep's RESULTS records
the same effect). The per-seed **means** differ: −0.122 / −0.124 / −0.129.

## How the rows can be read (PREREG §3, stated before the data)

Δ is measured in FIFO's rollout. A resident reading compares a sentence with a same-kind donor at
its rank. An evicted reading compares it with FIFO's oldest resident, the sentence it displaces.
Every future term is a FIFO-world counterfactual, never demand in the hindsight rule's own memory.
The hindsight rule is not a learnable policy: its cap is what a perfect predictor of `G^Δ_γ` would
reach, and the hindsight advantage grows with γ. This file does not go beyond the reading rule.
