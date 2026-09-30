# RESULTS: B0, zero-cost ceilings

**Descriptive only. No gate.** Everything here comes from the already-inspected U range and was
computed in-sample (PREREG §2). It feeds only B2's "already seen" addendum. It is **not** Q2's
comparator: B2 re-estimates its kind-oracle on `FIT_TRAIN`.

- **Command:** `PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run --lane cpu-det --slots 1 -- .venv/bin/python experiments/b0-ceilings/run.py`
  - rc **0**, `status: ok`.
- **Git SHA:** `bb40c40557d11bb01c0907a988ba76f86d3aa2ac`, clean (`provenance.dirty: false`).
  - PREREG `fc26dbf`, tests `3fbeeb1`, code `6b14028`, mutations `bb40c40`.
- **Hardware:** Mac Studio, Apple M4 Max, macOS 26.6.2 arm64, CPU only, 1 slot, Python 3.12.13.
  - Wall time 71.7 / 69.9 / 71.6 s per seed (`seconds_per_seed`).
  - Started 2026-09-27T21:21:59Z; finished 21:25:33Z.
- **Ledger:** `runs/b0-ceilings/ledger.json` (1306 rows). **Manifest:** `runs/b0-ceilings/manifest.json`,
  config hash `3df1a6465b75f92849f3ac6f88bff07625439468998f7ad9cc75a9f40f0f2433`.
- **Seeds actually run:** [0, 1, 2]. 1088 documents per seed (`n_documents`). Checkpoints 2500 and 3000. γ ∈ {0.9, 0}.
- **Metric:** model-free residency (`simulate`), **not** model-read accuracy.

## Controls, all passed (`controls_failed: []`)

- **C1 (the T0 manifest):**
  - At start and at end, all six `D.pt` sha256 values equal `~/rsr-substrate/2026-09-27/MANIFEST.sha256` (`C1.start.ok`, `C1.end.ok` = true).
  - Separately, the full manifest was re-checked after the run: `shasum -a 256 -c`, rc 0, **435/435 OK**, 0 not OK.
- **C2 (shape):** every `D.pt` holds exactly U's 1088 ids. Each `[48, 48]` tensor is finite at `i < t` and NaN at `i ≥ t`.
- **C3 (reproduction):** the recomputed U hit rates of `fifo`, `factfiller`, `oracle`, `rule_g0` and `rule_g09`, plus their `gap_gt_M` rates, **equal** `runs/lookahead-room-r2/ledger.json` exactly, at both checkpoints and on all three seeds.
- **C4:** 3 seeds × 1088 documents.

## Headline: ckpt3000, γ = 0.9, all-query residency (seed 0 / 1 / 2)

The references are the same at every checkpoint and γ:

| Reference | FIFO | factfiller | oracle |
|---|---|---|---|
| Hit rate | 0.8059 / 0.8093 / 0.8088 | 0.9655 / 0.9653 / 0.9666 | 1.0000 / 1.0000 / 1.0000 |

The oracle's cap is +1.217 / +1.223 / +1.212.

In the table below, cap = (hit − FIFO) / (factfiller − FIFO), with paired per-document bootstrap 95% CIs (2000 resamples). No cap replicate was dropped anywhere (`cap.*.dropped` = 0).

| Rule | Seed | Hit | Hit − FIFO [CI] | Cap [CI] |
|---|---|---|---|---|
| **(i) kind-oracle** `ko` (A1.10 random tie) | 0 | 0.9649 | +0.159 [+0.153, +0.165] | **+0.996** [+0.978, +1.016] |
| | 1 | 0.6613 | −0.148 [−0.154, −0.142] | **−0.949** [−1.014, −0.891] |
| | 2 | 0.9669 | +0.158 [+0.153, +0.163] | **+1.002** [+0.982, +1.021] |
| (i′) `ko_oldest` (secondary) | 0 | 0.9873 | +0.181 [+0.176, +0.187] | +1.137 [+1.120, +1.155] |
| | 1 | 0.7086 | −0.101 [−0.105, −0.096] | −0.645 [−0.693, −0.601] |
| | 2 | 0.9862 | +0.177 [+0.172, +0.183] | +1.124 [+1.106, +1.143] |
| **(ii) kind × age-band** `kb`: **NOT a legal ψ̂** | 0 | 0.9491 | +0.143 [+0.138, +0.148] | **+0.898** [+0.878, +0.917] |
| | 1 | 0.7645 | −0.045 [−0.049, −0.040] | **−0.287** [−0.323, −0.254] |
| | 2 | 0.9770 | +0.168 [+0.163, +0.173] | **+1.066** [+1.045, +1.086] |
| **(iii) age-only** `age` | 0 | 0.8124 | +0.007 [+0.001, +0.012] | **+0.041** [+0.004, +0.074] |
| | 1 | 0.8256 | +0.016 [+0.012, +0.021] | **+0.105** [+0.077, +0.133] |
| | 2 | 0.8105 | +0.002 [−0.004, +0.007] | **+0.010** [−0.028, +0.044] |
| hindsight `rule_g09` (lookahead-room) | 0 / 1 / 2 | 0.8352 / 0.8301 / 0.8935 | | +0.184 [+0.147, +0.220] / +0.134 [+0.103, +0.165] / +0.537 [+0.508, +0.564] |
| hindsight `rule_g0` | 0 / 1 / 2 | 0.7997 / 0.7881 / 0.8457 | | −0.039 [−0.073, −0.006] / −0.136 [−0.164, −0.107] / +0.234 [+0.204, +0.260] |

**Why the kind-oracle breaks on seed 1** (read off the class means; not a new measurement):
- `E[G_0.9 | kind]` for assert / query / filler:
  - seed 0: 0.5728 / 0.5340 / 0.4586;
  - seed 1: **0.4055 / 0.5098** / 0.3425;
  - seed 2: 0.5862 / 0.4772 / 0.3181.
- Filler is the lowest class on every seed. On seed 1, however, **asserts rank below query sentences**.
- Once the fillers are gone, the kind-oracle evicts asserts before queries. Fact/filler, by contrast, treats a query as a non-assert and evicts it.
- At ckpt2500, seed 0 has the same order (assert 0.5894 < query 0.5911), and there `ko` also goes negative (cap −0.933).
- The kind-oracle's hit rates are **identical at γ = 0.9 and γ = 0** within each checkpoint, because the three class means keep the same order at both γ.

## The other combinations (all-query cap, point estimate; the CIs are in the ledger)

| ckpt, γ | ko | ko_oldest | kb | age |
|---|---|---|---|---|
| 3000, 0 | +0.996 / −0.949 / +1.002 | +1.137 / −0.645 / +1.124 | +0.456 / −0.001 / +0.593 | +0.061 / −0.018 / +0.092 |
| 2500, 0.9 | −0.933 / −0.949 / +1.002 | −0.625 / −0.645 / +1.124 | +0.659 / −0.287 / +0.990 | +0.059 / +0.104 / +0.021 |
| 2500, 0 | −0.933 / −0.949 / +1.002 | −0.625 / −0.645 / +1.124 | +0.444 / +0.011 / +0.589 | +0.014 / −0.018 / +0.092 |

For ckpt2500, the hindsight caps are:
- `rule_g09`: +0.109 / +0.027 / +0.355;
- `rule_g0`: −0.020 / −0.208 / +0.219.

The `gap_2_to_M` and `gap_gt_M` hit rates of every rule are in the ledger (`B.ckpt{c}.{g}.hit.<rule>.<bucket>`).

For reference:
- FIFO is 1.0000 on `gap_2_to_M` and 0.0000 on `gap_gt_M`, on every seed.
- Fact/filler is 0.9737 / 0.9734 / 0.9754 on `gap_2_to_M` and 0.9027 / 0.9009 / 0.9020 on `gap_gt_M`.

## F2: E[G | assert] − E[G | filler] (ckpt3000)

| γ | kind assert − kind filler [CI] (the PREREG's definition) |
|---|---|
| 0.9 | +0.114 [+0.110, +0.118] / +0.063 [+0.061, +0.065] / +0.268 [+0.264, +0.272] |
| 0 | +0.015 [+0.015, +0.016] / +0.007 [+0.007, +0.008] / +0.036 [+0.036, +0.037] |

**🔴 This does not reproduce the "already seen" F2 (+0.0633 / −0.0507 / +0.1601), and here is why:**
- The scratch number is `E[G | kind = assert] − E[G | class = filler]`. W10's `slot_class` counts **query sentences as filler**.
- From this ledger, `E_G.kind.assert − E_G.class.filler` at γ = 0.9 gives:
  - 0.5728 − 0.5095 = **+0.0633**;
  - 0.4055 − 0.4562 = **−0.0507**;
  - 0.5862 − 0.4260 = **+0.1601** (from the unrounded ledger values).
- Those match the scratch numbers to 4 decimal places.
- So the "seed-1 reversal" is **not** asserts against true fillers. It is asserts against a pool that includes query sentences, whose G is high.
- Under the kind definition, assert > filler on every seed and at both γ, with CIs excluding 0.
- The PREREG fixed the kind definition before the run; the class-based reconciliation above is derived arithmetic on two ledger keys, not a pre-registered statistic.

`E[G_γ | class]` at ckpt3000, for pending / querying / answered / filler:
- γ = 0.9:
  - seed 0: 0.6956 / 0.6436 / 0.5372 / 0.5095;
  - seed 1: 0.5285 / 0.5390 / 0.3678 / 0.4562;
  - seed 2: 0.7200 / 0.6655 / 0.5479 / 0.4260.
- γ = 0:
  - seed 0: 0.0757 / 0.1992 / 0.0787 / 0.0732;
  - seed 1: 0.0639 / 0.1803 / 0.0561 / 0.0693;
  - seed 2: 0.0802 / 0.2127 / 0.0797 / 0.0631.

## F1: demand D by age × class (ckpt3000; the full table for 47 ages is in the ledger, `B.ckpt3000.F1.D.<age>|<class>.mean`)

| Age | pending | filler (slot_class; includes queries) |
|---|---|---|
| 1 | 0.0930 / 0.0999 / 0.1047 | 0.0580 / 0.0831 / 0.0615 |
| 8 | 0.0580 / 0.0578 / 0.0591 | 0.0503 / 0.0507 / 0.0422 |
| 13 | 0.0562 / 0.0472 / 0.0682 | 0.0579 / 0.0523 / 0.0601 |
| 16 | 0.0783 / 0.0508 / 0.0758 | 0.0928 / 0.0683 / 0.0731 |
| ≤ M (FIFO-resident) | 0.0711 / 0.0688 / 0.0787 | 0.0573 / 0.0606 / 0.0544 |
| > M (rank-0 probe) | 0.0843 / 0.0544 / 0.0829 | 0.0962 / 0.0817 / 0.0757 |

- These reproduce the already-seen F1 numbers (B2 §A) exactly: age 1, age 16, and the probe.
- Pending falls below filler from **age 13 on seeds 0 and 1**. On seed 2, pending stays above filler at every age from 1 to 16 (for example 0.0758 against 0.0731 at age 16).
- `m_age` at γ = 0.9 is U-shaped over ages 1–16 (minimum at age 8 on seed 1 and at ages 4–5 on seeds 0 and 2). It then falls from about age 18 to 47 (to 0.122 / 0.065 / 0.114 at age 47), the region where G's truncation at document end dominates.

## Expectations (PREREG §8) against what happened

- **ko:** a cap of ≈ 0.8–1.1 on seeds 0 and 2 and a negative one on seed 1 was expected, and that is what happened at ckpt3000.
  - **The mechanism I expected was wrong:** the break is assert < query, not assert < filler.
  - At ckpt2500, seed 0 also goes negative, which I did not predict.
- **age:** a cap in [−0.5, 0.5] was expected; it fell there (+0.041 / +0.105 / +0.010).
- **kb ≥ ko on seed 1:** yes (−0.287 against −0.949). **kb ≈ ko on seeds 0 and 2:** roughly (0.898 and 1.066).
- **C3 exact:** yes.
