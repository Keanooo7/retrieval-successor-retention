# E0D - `r_i` validation against leave-one-out

**Status: DONE. Class `AGREE`, exit 0 (run 2026-09-29, 20:11:43Z → 22:20:09Z). Read "Scope of an AGREE" below before citing it.**

| | |
|---|---|
| Prediction | High Spearman rho. **If not, LOO is truth and `r_i` is a confound**, and section 3.4's redundancy term is the specified response. **Report the per-layer profile once before collapsing to a scalar.** |
| Kill gate? | **Yes** |
| Sprint 1 task | T6 |

## Result

**Class `AGREE`, exit 0.** All three seeds are labelled `AGREE` under A2.8. Seed labels:
`['AGREE', 'AGREE', 'AGREE']` (`e0d.labels`). The row-5b booleans are all false, and there is no HALT.
The class is the same with all cells included, with ungated `r_i`, with the zero knockout, with raw
`AUROC_strat` substituted in, and at τ q0.99 and q0.999.

| seed | `AUROC_strat,pct` [95% CI] (gate: CI lower ≥ A\* = 0.85) | label |
|---|---|---|
| 0 | **0.9423** [0.9391, 0.9455] | AGREE |
| 1 | **0.9313** [0.9278, 0.9346] | AGREE |
| 2 | **0.9433** [0.9402, 0.9462] | AGREE |
| across seeds (`primary.AUROC_strat_pct`) | mean 0.9390, sd 0.0066 (n = 3; `sd_exactly_zero` false) | – |

> **Scope of an AGREE (A2.4, review M-3). Read this before citing the result.** AGREE certifies
> that `r_i` and leave-one-out agree **at the moment of retrieval**. It shows that `r_i` ranks the
> slot being retrieved *now* above its same-age peers.
>
> On set E, about **92%** of the positives are the queried assert at its own query step. Only about
> **1.9%** are needed later (`m3.log`: 10 of 517 on seed 0). This run's ledger does not re-measure
> that composition on `D_E0d`, so the percentages are set-E figures.
>
> **E0d does NOT show that `r_i` predicts later demand.** A2.6's limit applies. Eviction consequence
> is B2's question, not E0d's.

**The A1 Spearman ρ readings point the other way, and they are reported as such.** They never gate
(A2.1). ρ_Q is only 0.17–0.19, and ρ_pool is only 0.065–0.083. Bottom-1 agreement is **below**
chance (1/16) on every seed. Under A1's rule (`label_A1`, ρ\* = 0.5), every seed reads `CEILING`,
because the `true_demand` control's own ρ_Q,rank CI is about [0.32, 0.33], below 0.5. So
`class_A1` = `MIXED_UNRESOLVED`. At ρ\* = 0.3 every seed reads `DISAGREE`, and `class_A1` would be
`CONFOUND`.

This is the regime A2.12 anticipated. With about 5% signal-bearing cells, Spearman ρ has a low
ceiling. It is not evidence against the AUROC gate. The two families do, however, answer
different questions, and only the AUROC one was pre-registered to gate.

## Runner choices the PREREG left open (`PREREG-OPEN:` in `run.py`; no data)

These are implementation readings of `PREREG.md` + Amendment 1. None changes a
threshold, a stratum, a label or an exit. Each is marked in `run.py`.

| # | where | choice |
|---|---|---|
| 1 | C8 | ~~the rho* ruling is found as `R-*-rho-star*`~~ **superseded by A2.13**: every `R-*-e0d-statistic*` ruling (first and `-amended`, both required), one `AUROC_STAR` by the unanchored key |
| 2 | C1 / A1.7 | the T0 record is `runs/t0-substrate/manifest.json`, key `substrate_manifest` (required, must exist) and `repo_root` (optional, default the main checkout). **No T0 record exists at this commit, so C1 refuses even once the rulings exist.** |
| 3 | C2 / A1.5 | every committed manifest with `stream: {offset, stride}` is re-checked with its own `stream_documents` and/or `[offset + stride·resume_step, offset + stride·end_step)`; a stream manifest with neither is a refusal; any stream range reaching step > 16123 is a refusal (the review's bound, applied to every stream manifest, not only one named B5) |
| 4 | C5 | underfull fraction counted over (document, step) pairs with t ≥ M |
| 5 | §3.1 item 6 | an assert is *pending* at t while its query is at or after t |
| 6 | ρ_Q,rank | a rank whose within-rank ρ is undefined carries no weight |
| 7 | ρ_step, bottom-1 | within a step, over its measured slots: ρ needs ≥ 3, bottom-1 needs ≥ 2; a slot with no Δ (no donor) or excluded by A1.3 is left out of its step |
| 8 | R_FLAT | CV = population sd (ddof 0) / mean over the step's slots with a value; under A1.3 rank M−1 is left out of the step |
| 9 | A1.3 | "every decision-bearing statistic" is read as everything a §7 label reads: ρ_Q, ρ_Q,rank, C10's ρ_Q, the A-cell LOO median and R_FLAT, all recomputed on the excluded cells |
| 10 | §7 row 0 | a positive-control CI that is undefined (NaN) reads CEILING; an undefined `r_i` CI falls through to UNRESOLVED |
| 11 | zero knockout | its (reported-only) table reads LOO flatness as the A-cell median of Δ_zero |
| 12 | A1.1 author's note | `degenerate_under_ceiling` is reported when ≥ 2 seeds meet the LOO-flat condition, at least one of them is labelled CEILING, and the class is not DEGENERATE_UNINFORMATIVE |
| 13 | ledger status | exit 0 → `ok`, 1 → `failed`, 2 → `partial` (the ledger has no word for "ran, no decision"); an exception → `crashed`, a failed control → `did_not_run`; both exit 3 |
| 14 | §5 undefined | no A-cell with a Δ, or no step with two `r_i` values, is exit 3 (the rule cannot be read) |

### Amendment 2 (8c63ecd) choices, runner at 90a74b5 and later (no data)

| # | where | choice |
|---|---|---|
| 15 | A2.4 "Undefined" | an undefined primary statistic on a LOO-flat seed is labelled by rows 1-2 directly (the `true_demand` control shares the labels, so it is undefined too and row 0 cannot be read); not LOO-flat is exit 3 |
| 16 | A2.5 STEP_SENSITIVE | the raw-substituted label also substitutes the control's raw `AUROC_strat` in row 0; row 5b is skipped |
| 17 | C11 in the population | every Q-step of a population must hold the same n_t, and it must be M (all cells) or M - 1 (A1.3); else exit 3 |
| 18 | A2.3 bins | a registered bin with no labelled cell (bin {1} under A1.3) is logged as excluded with 0 / 0 |
| 19 | secondary grid cells | an undefined statistic in a non-primary cell is labelled `UNDEFINED` (reported; section 8 puts it in row 6) |
| 20 | `label_A1` | computed at the PREREG's proposed rho* = 0.5 (A2.1 retires RHO_STAR; none is ratified), with the {0.3, 0.7} sensitivities |
| 21 | `H_age-oracle` | min over ages of the mean over Q_crit of y(t, a), the v1 draft's A2.7 definition (v3 names it without one) |
| 22 | pooled (reported) | its label is read with LOO-flat and r-flat false; it never gates |
| 23 | sink probe | assert/query pairs at full-memory steps of the population, query index < t, both resident |

**Blocked at 0914f26:** C8 exits 3 on this branch until `R-2026-09-27-e0d-statistic-amended`
(4252408, night/2026-09-27) is merged in, and C1 exits 3 until a T0 record
(`runs/t0-substrate/manifest.json`, A1.7) exists. *(Both were resolved before launch: C8 passed with both rulings present, and C1 passed on the T0 record `2fd9368`. See the C1 and C8 rows under Numbers.)*

## Reproduction

| | |
|---|---|
| Command | `PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run --lane cpu-det --slots 3 --job-id e0d-2026-09-29 -- .venv/bin/python experiments/e0d/run.py` (cwd `.worktrees/e0d`, detached with nohup) |
| Git SHA | `2fd93683a33d98b8540ab572f39848dcc0dbb6f5` (PREREG base + A1 + A2 + A3 `f5a8752`; T0 record `2fd9368`) |
| Hardware | Mac Studio, Apple M4 Max, 64 GB, macOS 26.6.2; device `cpu`, 3 threads (cpu-det lane, 3 slots) |
| Date | started 2026-09-29, UTC 2026-09-29T20:11:43Z |
| Inputs | arm B `ckpt-003000.pt`, seeds 0, 1, 2; documents `[262144, 263168)` |

## Numbers

Every figure below is read from `runs/e0d/ledger.json`: status `ok`, exit 0,
`seeds_actually_run` [0, 1, 2], config hash
`08b9eb6f71ffade6f502e6c3279b8749c79c7d2721750a852581b9cb1c22c5ee`, git SHA `2fd9368` (clean). The
tables were generated from the ledger by script and were not typed. Per-seed values come from
`seed<s>.analysis` → `populations.bos_excluded` (the A1.3 population, which is primary) or
`populations.all`. Cross-seed spread comes from `primary.*` / `secondary.*`, and the class from
`e0d.*`. Seed-level CIs are the per-document cluster bootstrap: 2000 resamples, generator seed
20260927, 95% percentile (A3.3). They capture **within-seed variance only** (A2.10).

### Headline (primary cell: gated r_i x resample LOO, A1.3 population; A* = 0.85 from C8)

| seed | `AUROC_strat,pct` [95% CI] | raw `AUROC_strat` [CI] | `AUROC_unstrat,pct` [CI] | `C_ws` [CI] | H [CI] | row 5b | label |
|---|---|---|---|---|---|---|---|
| 0 | **0.9423** [0.9391, 0.9455] | 0.9585 [0.9555, 0.9615] | 0.9479 [0.9448, 0.9510] | 0.3990 [0.3951, 0.4028] | 0.0046 [0.0032, 0.0063] | false | AGREE |
| 1 | **0.9313** [0.9278, 0.9346] | 0.9605 [0.9575, 0.9635] | 0.9491 [0.9459, 0.9520] | 0.3320 [0.3275, 0.3362] | 0.0040 [0.0025, 0.0056] | false | AGREE |
| 2 | **0.9433** [0.9402, 0.9462] | 0.9593 [0.9564, 0.9621] | 0.9502 [0.9471, 0.9530] | 0.4005 [0.3967, 0.4037] | 0.0021 [0.0009, 0.0035] | false | AGREE |
| pooled (reported, never gates) | 0.9383 [0.9365, 0.9401] | 0.9581 [0.9564, 0.9598] | 0.9490 [0.9474, 0.9507] | – | – | – | AGREE |

### Positive control (`true_demand`, A2.8 row 0) and counts, primary population

| seed | control pct [CI] | control raw [CI] | Q-steps | cells | labelled | positives / negatives | n_t | Q_crit (dropped for NaN donor) | undefined resamples (pct) | excluded bins |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.9596 [0.9565, 0.9628] | 0.9467 [0.9434, 0.9501] | 11446 | 171690 | 170174 | 8293 / 161881 | [15] | 6903 (1025) | 0 | [{'bin': 1, 'positives': 0, 'negatives': 0}] |
| 1 | 0.9723 [0.9696, 0.9748] | 0.9624 [0.9595, 0.9652] | 11450 | 171750 | 170278 | 7927 / 162351 | [15] | 6755 (950) | 0 | [{'bin': 1, 'positives': 0, 'negatives': 0}] |
| 2 | 0.9658 [0.9628, 0.9684] | 0.9488 [0.9453, 0.9522] | 11365 | 170475 | 169098 | 7852 / 161246 | [15] | 6670 (906) | 0 | [{'bin': 1, 'positives': 0, 'negatives': 0}] |
| pooled | 0.9658 [0.9640, 0.9673] | – | | | | | | | | |

### H and its baselines (A2.6, reported only), primary cell

| seed | H [CI] | H_random | H_age-random | R_H [CI] | R_H,uniform | H_FIFO | H_age-oracle |
|---|---|---|---|---|---|---|---|
| 0 | 0.0046 [0.0032, 0.0063] | 0.0698 | 0.0524 | 0.0885 [0.0614, 0.1204] | 0.0665 | 0.0326 | 0.0000 (see defect below) |
| 1 | 0.0040 [0.0025, 0.0056] | 0.0686 | 0.0289 | 0.1385 [0.0884, 0.1934] | 0.0583 | 0.0311 | 0.0000 (see defect below) |
| 2 | 0.0021 [0.0009, 0.0035] | 0.0690 | 0.0592 | 0.0354 [0.0156, 0.0586] | 0.0304 | 0.0220 | 0.0000 (see defect below) |

### The 2x2 grid and the all-cells population (A2.7 items 1-2): `AUROC_strat,pct` [CI] and label

| seed | population | gated x resample | ungated x resample | gated x zero | ungated x zero |
|---|---|---|---|---|---|
| 0 | A1.3 (primary) | 0.9423 [0.9391, 0.9455] AGREE | 0.9419 [0.9386, 0.9451] AGREE | 0.9727 [0.9714, 0.9740] AGREE | 0.9725 [0.9712, 0.9738] AGREE |
| 0 | all cells | 0.9488 [0.9463, 0.9516] AGREE | 0.9487 [0.9462, 0.9514] AGREE | 0.9757 [0.9746, 0.9768] AGREE | 0.9757 [0.9746, 0.9768] AGREE |
| 1 | A1.3 (primary) | 0.9313 [0.9278, 0.9346] AGREE | 0.9299 [0.9265, 0.9332] AGREE | 0.9549 [0.9530, 0.9567] AGREE | 0.9527 [0.9507, 0.9545] AGREE |
| 1 | all cells | 0.9396 [0.9368, 0.9423] AGREE | 0.9387 [0.9360, 0.9414] AGREE | 0.9600 [0.9582, 0.9616] AGREE | 0.9584 [0.9567, 0.9601] AGREE |
| 2 | A1.3 (primary) | 0.9433 [0.9402, 0.9462] AGREE | 0.9433 [0.9403, 0.9463] AGREE | 0.9649 [0.9633, 0.9665] AGREE | 0.9651 [0.9636, 0.9667] AGREE |
| 2 | all cells | 0.9503 [0.9478, 0.9527] AGREE | 0.9505 [0.9480, 0.9529] AGREE | 0.9686 [0.9673, 0.9700] AGREE | 0.9690 [0.9677, 0.9703] AGREE |

### Flatness (A1.1, §5), primary population

| seed | LOO A-cell median Δ_resample (n) | LOO flat | Δ_zero A-cell median (n) | `r_i` median CV gated / ungated | r flat |
|---|---|---|---|---|---|
| 0 | 1.1517 (7744) | false | 1.0847 (7801) | 0.4410 / 0.4457 | false |
| 1 | 1.1216 (7683) | false | 1.0606 (7736) | 0.4623 / 0.4553 | false |
| 2 | 1.1565 (7642) | false | 1.0986 (7690) | 0.5359 / 0.5331 | false |

### A1 ρ secondaries (A2.7 item 5), gated r_i x resample

| seed | population | ρ_Q [CI] | ρ_Q,rank [CI] | ρ_pool [CI] | ρ_rank (all full cells) [CI] | bottom-1, all full-memory steps [CI] | C10 control ρ_Q,rank CI | `label_A1` (ρ*=0.5) | ρ*=0.3 | ρ*=0.7 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | A1.3 | 0.1683 [0.1637, 0.1730] | 0.1618 [0.1570, 0.1666] | 0.0653 [0.0624, 0.0682] | 0.0645 [0.0615, 0.0675] | 0.0381 [0.0360, 0.0403] | [0.3219, 0.3274] | CEILING | DISAGREE | CEILING |
| 0 | all cells | 0.1755 [0.1712, 0.1799] | 0.1619 [0.1577, 0.1663] | 0.0763 [0.0736, 0.0792] | 0.0731 [0.0702, 0.0759] | 0.0332 [0.0312, 0.0352] | [0.3166, 0.3214] | CEILING | DISAGREE | CEILING |
| 1 | A1.3 | 0.1764 [0.1715, 0.1815] | 0.1482 [0.1435, 0.1529] | 0.0829 [0.0798, 0.0862] | 0.0638 [0.0610, 0.0668] | 0.0233 [0.0216, 0.0252] | [0.3202, 0.3258] | CEILING | DISAGREE | CEILING |
| 1 | all cells | 0.1828 [0.1783, 0.1874] | 0.1467 [0.1423, 0.1512] | 0.0904 [0.0871, 0.0936] | 0.0695 [0.0666, 0.0723] | 0.0196 [0.0181, 0.0211] | [0.3151, 0.3199] | CEILING | DISAGREE | CEILING |
| 2 | A1.3 | 0.1853 [0.1806, 0.1898] | 0.1769 [0.1723, 0.1814] | 0.0731 [0.0698, 0.0760] | 0.0742 [0.0709, 0.0773] | 0.0303 [0.0283, 0.0323] | [0.3198, 0.3254] | CEILING | DISAGREE | CEILING |
| 2 | all cells | 0.1855 [0.1811, 0.1895] | 0.1727 [0.1683, 0.1768] | 0.0833 [0.0800, 0.0862] | 0.0814 [0.0781, 0.0843] | 0.0238 [0.0222, 0.0255] | [0.3145, 0.3195] | CEILING | DISAGREE | CEILING |

### Sink / inversion probe (A2.7 item 8), gated r_i, A1.3 population (all-cells in parentheses)

| seed | mean r: pending assert | spent assert | query | filler | frac r_assert > r_query (pairs) |
|---|---|---|---|---|---|
| 0 | 0.1041 (0.1069) | 0.0513 (0.0503) | 0.0588 (0.0587) | 0.0504 (0.0494) | 0.4995 (105183) (0.4774, 124152) |
| 1 | 0.0975 (0.1014) | 0.0453 (0.0441) | 0.0658 (0.0673) | 0.0437 (0.0448) | 0.2458 (106125) (0.2240, 125232) |
| 2 | 0.1153 (0.1187) | 0.0573 (0.0561) | 0.0600 (0.0603) | 0.0334 (0.0328) | 0.4719 (105467) (0.4505, 124540) |

### Per-layer ρ profile and ρ by sentence kind (all-cells population, gated × resample; reported)

| seed | per-layer ρ, cross-attention layers 0-5 | ρ pending assert | spent assert | query | filler |
|---|---|---|---|---|---|
| 0 | 0.1167, 0.0634, 0.0425, 0.0177, 0.0584, 0.0805 | 0.3015 (n 75187) | 0.0094 (n 124152) | 0.0603 (n 207393) | -0.0073 (n 117556) |
| 1 | 0.1182, 0.0579, 0.0388, 0.0752, 0.0609, 0.0991 | 0.3028 (n 75472) | 0.0510 (n 125232) | 0.0762 (n 207607) | -0.0026 (n 115977) |
| 2 | 0.0949, 0.0818, 0.0487, 0.0772, 0.0771, 0.0377 | 0.2629 (n 75481) | -0.0102 (n 124540) | 0.0743 (n 206610) | -0.0083 (n 117657) |


### Classification, sensitivities and flags (`e0d.*` rows)

| reading | per-seed labels | class |
|---|---|---|
| **primary** (gated × resample, A1.3) | AGREE, AGREE, AGREE | **AGREE → exit 0** |
| all cells (no A1.3 exclusions) | AGREE, AGREE, AGREE | AGREE (`BOS_SENSITIVE` false) |
| ungated `r_i` | AGREE, AGREE, AGREE | AGREE (`GATE_SENSITIVE` false) |
| ungated, all cells | – | AGREE |
| zero knockout (reported only) | AGREE, AGREE, AGREE | AGREE |
| zero knockout, all cells | – | AGREE |
| raw `AUROC_strat` substituted (A2.5) | AGREE, AGREE, AGREE | – (`STEP_SENSITIVE` false on all seeds) |
| τ at q = 0.99 | AGREE, AGREE, AGREE | AGREE |
| τ at q = 0.999 | AGREE, AGREE, AGREE | AGREE |
| row 5b boolean (A2.8) | false, false, false | HALT false, halt seeds [] |
| flatness | LOO-flat false ×3; `r`-flat false ×3 | `degenerate_under_ceiling` false; `rank_ceiling_seeds` [] |
| **`label_A1` (ρ\* = 0.5, A3.4)** | CEILING, CEILING, CEILING | **`class_A1` = MIXED_UNRESOLVED** |
| `label_A1` at ρ\* = 0.3 | DISAGREE, DISAGREE, DISAGREE | CONFOUND |
| `label_A1` at ρ\* = 0.7 | CEILING, CEILING, CEILING | MIXED_UNRESOLVED |

Frozen τ used (`c12_tau_tables`; A2.2 parsed at run time): resample 0.4157 / 0.4316 / 0.4885 and
zero 0.3448 / 0.3300 / 0.3865 on seeds 0 / 1 / 2. Sensitivity τ values are in `manifest.json`
`tau_sensitivity`.

Cross-seed spread (`primary.*`, n = 3): pct mean 0.9390, sd 0.0066. Raw: 0.9594, sd 0.0011.
Unstratified pct: 0.9491, sd 0.0011. `C_ws`: 0.3772, sd 0.0391. H: 0.0036, sd 0.0013. The
seed-to-seed sd of the primary statistic (0.0066) is about **2× the width of any single seed's
bootstrap CI half-width**, so seed variance is not negligible next to the document bootstrap. Seed
1 is the low seed on pct (0.9313) and on `C_ws` (0.3320), yet it clears the bar by 0.078 at its CI
lower bound.

### §9's registered prediction, scored per A2.14 / A1.8 / A3.4

| part | prediction | observed | scored |
|---|---|---|---|
| class | CONFOUND (second most likely MIXED), scored against `class_A1` only | `class_A1` = MIXED_UNRESOLVED (all three seeds CEILING at ρ\* = 0.5) | **first choice wrong; second choice right.** MIXED arises only through C10's ceiling, and at ρ\* = 0.3 `class_A1` is CONFOUND |
| ρ_pool range | pooled gated ρ_resample 0.1–0.4 | 0.0653 / 0.0829 / 0.0731 (A1.3); 0.0763 / 0.0904 / 0.0833 (all cells) | **wrong**: below the range on every seed, in both populations |
| ρ_pool CI | CI upper < 0.5 on every seed | largest CI upper 0.0936 | right |
| recency | ρ_rank below ρ_pool | A1.3: 0.0645 < 0.0653; 0.0638 < 0.0829; **0.0742 > 0.0731**. All cells: below on all three | **2 of 3 in A1.3, 3 of 3 in all cells**. These are point estimates; the CIs overlap on seeds 0 and 2 |
| bottom-1 | above 1/16 but under 0.3 | 0.0381 / 0.0233 / 0.0303 (A1.3); 0.0332 / 0.0196 / 0.0238 (all cells); every CI upper < 0.0625 | **wrong**: below chance on every seed. The `true_demand` control reads 0.0631 / 0.0619 / 0.0382, not above chance either |

No class prediction was registered for the A2.8 classification (A2.14). The AGREE is therefore not
scored as a hit or a miss against §9.

### C1 and the other controls (ledger rows)

- **C1 start** (`c1_substrate_start`, sha256): arm B `ckpt-003000.pt` hashes are seed 0
  `0ee3f8a6…a118da60`, seed 1 `dadd1e08…068a3da8`, seed 2 `b507ebc5…fd6b63b8`. These equal
  PREREG's pinned values. The T0 record is `runs/t0-substrate/manifest.json` → `substrate_manifest`
  `/Users/keanooo7/rsr-substrate/2026-09-27/MANIFEST.sha256`.
- **C1 end** (`c1_substrate_end`, `shasum -a 256 -c`): rc 0, 435 files OK.
- **C2** (`c2_disjointness`): 3072 generator keys, 0 collisions over 42 committed ranges, max used
  148160.
- **C8** (`c8_authority`): both rulings were present, `R-2026-09-27-e0d-statistic.md` and
  `R-2026-09-27-e0d-statistic-amended.md`, plus retrieval-shown and sprint0-gate-green.
  `AUROC_STAR` 0.85 was read from the rulings.
- **C12**: the τ tables equal A2.2.
- **Per-seed controls**, primary population:


| seed | C4 worst live loss diff | C5 underfull | C6 donor coverage | C7 bit-identical | C9 FIFO | C11 |
|---|---|---|---|---|---|---|
| 0 | 0.0 | 0 / 32768 | 0.9909 (519512 / 524288) | delta_resample, delta_zero, r_gated, r_ungated | True (64 batches) | ok=True, 32768 steps, 524288 cells |
| 1 | 0.0 | 0 / 32768 | 0.9913 (519736 / 524288) | delta_resample, delta_zero, r_gated, r_ungated | True (64 batches) | ok=True, 32768 steps, 524288 cells |
| 2 | 0.0 | 0 / 32768 | 0.9916 (519867 / 524288) | delta_resample, delta_zero, r_gated, r_ungated | True (64 batches) | ok=True, 32768 steps, 524288 cells |

### Numbers that look wrong, or are, reported as they came out

- **`H_age-oracle` = 0.0000 on every seed in the A1.3 population is a runner artifact, not a
  measurement.** `ArgminHit` (`run.py` about line 1458) builds its age axis as `1..M` and takes
  `np.min` over every age. A1.3 removes rank M−1 (age 1), so that column is identically zero and
  wins the min. In the all-cells population the values are real: 0.0108 / 0.0099 / 0.0110. The
  statistic is reported only and never gated. `H`, `H_random`, `H_age-random`, `R_H` and `H_FIFO`
  do not use this min and are unaffected.
- **Bottom-1 agreement is below chance, and ρ_pool is under 0.1, on every seed** (above). The
  gated AUROC reading does not dissolve this; it measures something else (A2.12).
- **H is far below H_random, H_age-random and H_FIFO.** `argmin r_i` almost never lands on a
  critical slot: H 0.0046 / 0.0040 / 0.0021 against H_FIFO 0.0326 / 0.0311 / 0.0220. This is a
  contemporaneous statistic (A2.6). It says nothing about eviction cost later.
- **The sink/inversion probe is seed-dependent.** The fraction of resident (assert, already-read
  query) pairs with `r_assert > r_query` is 0.4995 / **0.2458** / 0.4719. So on seed 1 the read
  query out-scores its own assert three times in four. Pending asserts carry at least 2× the mean
  share of spent asserts or filler on every seed.
- **The zero knockout agrees more strongly than resample** (pct 0.955–0.973 against 0.931–0.943).
  Zero knockout is the less realistic intervention. It never gates.
- **The `true_demand` control's ρ_Q,rank CI is about [0.32, 0.33] on every seed.** A1's ρ\* = 0.5
  was therefore unreachable even by the control on this corpus. That is why `label_A1` is CEILING
  and not a verdict.

### Provenance notes

- **`prereg_amendment` omits A3.** The manifest/runner string reads "Amendment 1 (84e21c5), erratum
  (268b947), Amendment 2 (8c63ecd)". Amendment 3 (`f5a8752`) is an ancestor of the recorded SHA
  `2fd9368` (`git merge-base --is-ancestor f5a8752 2fd9368` rc 0), so the run *did* execute under
  A3. The string is stale, not the code. It was left unedited because the runner is PREREG-governed.
- **The pre-data `find ~ -name e0d.class` check was incomplete.** It returned 0 hits with rc 1,
  because 135 directories were unreadable (e.g. `~/.Trash`; the launch record
  `~/Documents/RSR-2026-09-29-day/reports/E0D-RUN.md`). `find . -path '*runs/e0d*'` returned 0
  lines, rc 0. "No prior E0d data existed" is therefore believed for the unreadable directories,
  not verified.
- **The ledger's `commands[0].argv` records `uv run python experiments/e0d/run.py`.** The command
  actually launched is the slot-wrapped `.venv/bin/python` one under Reproduction. The entry point
  and exit code (0) agree.
- **Runtime:** launched 2026-09-29T20:11:43Z (ledger `started_utc` 20:11:44Z), `finished_utc`
  22:20:09Z. That is about **2 h 8 m** at 3 threads (cpu-det lane, 3 slots). The rc file
  `~/Documents/RSR-2026-09-29-day/reports/e0d-run.rc` reads `0`. The log's final line is
  `class AGREE -> exit 0; labels ['AGREE', 'AGREE', 'AGREE']`.
- `steps_requested` / `steps_done` are `null`. This is an evaluation run with no training steps.
  `cycle` is `null`; join on `run_id`.

> Per `CLAUDE.md`, this section records the numbers that came out **wrong** as well
> as the ones that came out right, and section 12.4 applies: documented
> configuration is not evidence of what was actually run.
