# RESULTS: Expire-Span referendum on frozen arm B (Amendment 2 re-run)

**Class (§7.2): SUPERSEDES. falsifier_6 = True.**

δ per seed (A1.1): {'0': 0.038022, '1': 0.036757, '2': 0.031858}

## 1. HEADLINE: `acc.all.es_minus_psiC` (ES − run 1 ψC, all-query EVAL answer accuracy)

Within-seed paired per-document bootstrap 95% CI, run 1's resample weights `boot_weights(2048, s, 2000)`.

- seed 0: +0.2111 [+0.2066, +0.2158] CI above 0
- seed 1: +0.1819 [+0.1776, +0.1863] CI above 0
- seed 2: +0.1648 [+0.1602, +0.1692] CI above 0

## 2. SECONDARY (reported, never classified; unadjusted for multiplicity)

### Accuracy levels, all queries

| seed | ES | ψC (run 1) | FIFO (run 1) | n queries |
|---|---|---|---|---|
| 0 | 0.7808 | 0.5697 | 0.7750 | 37853 |
| 1 | 0.7205 | 0.5386 | 0.7665 | 37870 |
| 2 | 0.7996 | 0.6348 | 0.8031 | 37836 |

### ES − FIFO and ES − ψC per bucket (accuracy)

| contrast | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| `acc.all.es_minus_fifo` | +0.0058 [+0.0038, +0.0078] CI above 0 | -0.0460 [-0.0485, -0.0434] CI below 0 | -0.0035 [-0.0063, -0.0007] CI below 0 |
| `acc.all.es_minus_psiC` | +0.2111 [+0.2066, +0.2158] CI above 0 | +0.1819 [+0.1776, +0.1863] CI above 0 | +0.1648 [+0.1602, +0.1692] CI above 0 |
| `acc.gap_eq_1.es_minus_fifo` | -0.0001 [-0.0009, +0.0006] CI includes 0 | -0.0005 [-0.0015, +0.0004] CI includes 0 | +0.0006 [-0.0003, +0.0016] CI includes 0 |
| `acc.gap_eq_1.es_minus_psiC` | -0.0010 [-0.0019, -0.0002] CI below 0 | -0.0030 [-0.0045, -0.0017] CI below 0 | -0.0005 [-0.0017, +0.0007] CI includes 0 |
| `acc.gap_ge_2.es_minus_fifo` | +0.0074 [+0.0049, +0.0100] CI above 0 | -0.0581 [-0.0613, -0.0549] CI below 0 | -0.0046 [-0.0082, -0.0011] CI below 0 |
| `acc.gap_ge_2.es_minus_psiC` | +0.2685 [+0.2628, +0.2741] CI above 0 | +0.2313 [+0.2260, +0.2366] CI above 0 | +0.2089 [+0.2031, +0.2144] CI above 0 |
| `acc.gap_2_to_M.es_minus_fifo` | -0.0174 [-0.0196, -0.0152] CI below 0 | -0.0791 [-0.0829, -0.0754] CI below 0 | -0.0214 [-0.0240, -0.0189] CI below 0 |
| `acc.gap_2_to_M.es_minus_psiC` | +0.3555 [+0.3497, +0.3612] CI above 0 | +0.3218 [+0.3158, +0.3276] CI above 0 | +0.2773 [+0.2714, +0.2828] CI above 0 |
| `acc.gap_gt_M.es_minus_fifo` | +0.0847 [+0.0773, +0.0925] CI above 0 | +0.0079 [+0.0022, +0.0138] CI above 0 | +0.0484 [+0.0371, +0.0601] CI above 0 |
| `acc.gap_gt_M.es_minus_psiC` | -0.0019 [-0.0121, +0.0093] CI includes 0 | -0.0532 [-0.0619, -0.0446] CI below 0 | -0.0060 [-0.0192, +0.0071] CI includes 0 |

### NLL (positive = ES worse)

| contrast | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| `nll.real.es_minus_fifo` | +0.0038 [+0.0033, +0.0044] CI above 0 | +0.0159 [+0.0151, +0.0168] CI above 0 | +0.0037 [+0.0029, +0.0045] CI above 0 |
| `nll.real.es_minus_psiC` | -0.1650 [-0.1676, -0.1625] CI below 0 | -0.1926 [-0.1951, -0.1900] CI below 0 | -0.1696 [-0.1724, -0.1667] CI below 0 |
| `nll.answer.es_minus_fifo` | +0.0118 [+0.0068, +0.0169] CI above 0 | +0.1302 [+0.1231, +0.1374] CI above 0 | +0.0315 [+0.0250, +0.0391] CI above 0 |
| `nll.answer.es_minus_psiC` | -0.6124 [-0.6273, -0.5972] CI below 0 | -0.5224 [-0.5362, -0.5090] CI below 0 | -0.5490 [-0.5635, -0.5334] CI below 0 |

## 3. Selection (§6.3, VAL)

Selected **L48-k1e-2**; tied at max ['L48-k1e-2']; rule: max mean VAL all-query accuracy over seeds 0-2; exact; ties -> lowest §4.1 row.

| config | mean VAL acc | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|
| L16-k1e-4 | 0.70538 | 0.6968 | 0.6860 | 0.7334 |
| L16-k1e-3 | 0.70112 | 0.6934 | 0.6803 | 0.7296 |
| L16-k1e-2 | 0.68282 | 0.6782 | 0.6585 | 0.7118 |
| L32-k1e-4 | 0.73960 | 0.7357 | 0.6981 | 0.7850 |
| L32-k1e-3 | 0.74111 | 0.7408 | 0.6949 | 0.7876 |
| L32-k1e-2 | 0.74084 | 0.7486 | 0.6958 | 0.7780 |
| L48-k1e-4 | 0.73840 | 0.7418 | 0.6917 | 0.7817 |
| L48-k1e-3 | 0.73452 | 0.7376 | 0.6764 | 0.7895 |
| L48-k1e-2 | 0.76496 | 0.7796 | 0.7142 | 0.8011 |

## 4. Controls and provenance

- Run head `99646cb` (branch `run/expire-span-referendum-a2`), clean tree at launch (`launch-git.json` dirty=false), launched 2026-10-07T00:14:07Z UTC (17:14 PDT 10-06), CPU, OMP_NUM_THREADS=1, orchestrator cpu-det.
- Training: launcher rc 0; 27/27 job rc 0; 27 `ckpt-003000.pt`; 0 non-finite `loss`/`loss_lm`/`es_grad_norm` heartbeat values.
- `validate --seed 0/1/2` rc 0/0/0, `ok: true` (§8 controls). `select` rc 0. `evaluate --seed 0/1/2` rc 0/0/0 (EVAL read once, selected config only). `finalize` rc 0.
- Control A2-C (DIAGNOSTIC): A2-C: 18/18 jobs BIT-IDENTICAL — seeds 0/2, all 9 configs, steps 0–2999 on loss, loss_lm, es_grad_norm, es_weight_norm, es_bias, plus step-3000 policy weight/bias `torch.equal`. Script `runs/es-ref-a2c_control.py`.
- seed 0: n_documents 2048; sha256 {'es': 'f6a3d97f25c520c84091562002e719c489d5824ba53b484f2ad892ff12d194e1', 'psiC': 'abd3a6214a1a7ac9084dfa802c086714cb88bf3309bc4026fc7439dc4ce03c08', 'fifo': 'b05387509be57d12587b4577ca49fd3443f07ebaec735db52715d2335d8caa1e'}
- seed 1: n_documents 2048; sha256 {'es': 'd647767927b7c3655a8d473fdef990c534315ed461271c1a109d466dcd3f8db2', 'psiC': '8f3accc91c38b211ae30cbf392cd3cae754422642200329bd72f95080997d2c2', 'fifo': '819bd8af12a70c194fc17ddc58d9d3967a0e2c8d6bff4bee5128d9e578420413'}
- seed 2: n_documents 2048; sha256 {'es': '6ded09f44abdefd90d68c9dfcd2c8d904f7f1e8dc07c6f934078b4425ab5a74a', 'psiC': 'a6bf4b1054a6e49020f3240545a87806bc4e3045ef766875d3d17507dcaf0203', 'fifo': '3a7ad2cecb55e1096bf981aa42d72dd2eac1758a6a0d75a039c3537fa60a2cb5'}
- Commands: `env -u RSR_ORCH_ROOT -u OUT -u JOB_SUFFIX -u DRY_RUN -u CKPT OMP_NUM_THREADS=1 .venv/bin/python experiments/expire-span-referendum/run.py {validate --seed s | select | evaluate --seed s | finalize}`; logs and rc files `runs/es-ref-*.{log,rc}`.
- Hardware: Apple M4 Max (Mac Studio), CPU, torch 2.14.0, Python 3.12.13.
- Original launch `c59d1af` outputs kept at `.worktrees/es-referendum-run/runs/expire-span-referendum/` (A2.5); none of their VAL numbers entered selection.
