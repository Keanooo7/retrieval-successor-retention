# E0D - `r_i` validation against leave-one-out

**Status: NOT RUN.**

| | |
|---|---|
| Prediction | High Spearman rho. **If not, LOO is truth and `r_i` is a confound**, and section 3.4's redundancy term is the specified response. **Report the per-layer profile once before collapsing to a scalar.** |
| Kill gate? | **Yes** |
| Sprint 1 task | T6 |

## Result

_Not run._

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
(`runs/t0-substrate/manifest.json`, A1.7) exists.

## Reproduction

| | |
|---|---|
| Command | _tbd_ |
| Git SHA | _tbd_ |
| Hardware | _tbd_ |
| Date | _tbd_ |

## Numbers

_Not run._

> Per `CLAUDE.md`, this section records the numbers that came out **wrong** as well
> as the ones that came out right, and section 12.4 applies: documented
> configuration is not evidence of what was actually run.
