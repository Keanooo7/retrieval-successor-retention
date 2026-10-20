# E0d Amendment 2: vendored supporting material

**Vendored 2026-09-29** (P2 step 2, a model session: Claude Opus 5.5) so that every path
PREREG Amendment 2 (`experiments/e0d/PREREG.md`, from `## Amendment 2`, commit `8c63ecd`)
cites resolves inside the repository. **Nothing here is authoritative over the PREREG.**
Where a file here and the committed PREREG differ, the PREREG governs.

Every file is a byte copy (`cp -p`) of the file named in the "copied from" column. None
was edited. The scripts are the record of what was run, so they are excluded from ruff
(`pyproject.toml`), like `experiments/e0c/sweep.py`. They are not re-run by anything in
the repository.

**Data hygiene.** Every measurement here is on set E, docs `[64, 128)`, fresh-stream
arm B ckpt3000. No file here holds any document of, or any measurement on,
`D_E0d = [262144, 263168)`. `m5_tau_seeds.py` asserts `HI = 128 <= 262144` and calls
`e0d_documents(..., cleared=False)`, which refuses the reserved range.

## Cited path → vendored copy

`V` = `experiments/e0d/amendment-2`. `~P` = `~/Documents/RSR-2026-09-27-plan`,
`~D` = `~/Documents/RSR-2026-09-29-day`.

| cited in the PREREG or v3 as | where the citation is | vendored copy | copied from |
|---|---|---|---|
| Part B (T1a–T19, mutations, §6A/B.1 calibration fixture), Part C, D.1–D.3 (incl. "D.2 item 2/3"), Part E | PREREG A2.4, A2.8, A2.11–A2.13 | `V/E0D-AMENDMENT-2-FINAL-v3.md` | `~D/E0D-AMENDMENT-2-FINAL-v3.md` |
| `reviews/e0d-a2/step1_cells.npz` (τ seed 0; sha256 `1338de4b…`) | A2.2 | `V/reviews/e0d-a2/step1_cells.npz` | `~P/reviews/e0d-a2/` |
| `reviews/e0d-a2-final/tau_cells_seed1.npz` (sha256 `b683dd1b…`) | A2.2 | `V/reviews/e0d-a2-final/tau_cells_seed1.npz` | `~P/reviews/e0d-a2-final/` |
| `reviews/e0d-a2-final/tau_cells_seed2.npz` (sha256 `4c55cc60…`) | A2.2 | `V/reviews/e0d-a2-final/tau_cells_seed2.npz` | `~P/reviews/e0d-a2-final/` |
| `reviews/e0d-a2-final/m5_out.json`, `m5.rc` | A2.2 | `V/reviews/e0d-a2-final/m5_out.json`, `m5.rc`, `m5.log`, `m5_tau_seeds.py` | `~P/reviews/e0d-a2-final/` |
| `m4.log` (choice of q = 0.995) | A2.2 | `V/reviews/e0d-a2-final/m4.log`, `m4_tau.py` | `~P/reviews/e0d-a2-final/` |
| `m3.log` (10 / 517 pending) | A2.4, A2.6 | `V/reviews/e0d-a2-final/m3.log`, `m3_oi.py` (also `m3b*`, `m3c*`) | `~P/reviews/e0d-a2-final/` |
| `reviews/e0d-a2-final/m6_out_012.json`, `m6.rc` | A2.2, A2.4 | `V/reviews/e0d-a2-final/m6_out_012.json`, `m6.rc`, `m6.log` | `~P/reviews/e0d-a2-final/` |
| `m6_pct_calib_power.py` | A2.11 | `V/reviews/e0d-a2-final/m6_pct_calib_power.py` | `~P/reviews/e0d-a2-final/` |
| `m7.log` (v3 D.3) | A2.11 figures | `V/reviews/e0d-a2-final/m7.log`, `m7.rc`, `m7_power_summary.py` | `~P/reviews/e0d-a2-final/` |
| `m1.log` (fixture matching, Part B.1), `m2` | v3 Part B.1, D.3 | `V/reviews/e0d-a2-final/m1*`, `m2*` | `~P/reviews/e0d-a2-final/` |
| `reviews/e0d-a2*/` (the grep for real-`r_i` AUROC) | A2.0 | `V/reviews/e0d-a2/` (step1, step1b, step3, step3b) and `V/reviews/e0d-a2-final/` (incl. `superseded/` slot logs) | `~P/reviews/` |
| "brief §3.4" (arm-B `r_i` carries age) | A2.8 row 5b | `V/research-notes/E0d-primary-statistic-brief.md` | `~P/research-notes/` |
| "REVIEW-E0D-A2v2 edits 1–12", "REVIEW-E0D-A2v3's edits", review M-2 | A2.0 preamble, A2.8 | `V/reviews/REVIEW-E0D-A2v2.md`, `V/reviews/REVIEW-E0D-A2v3.md` | `~D/reviews/` |
| "DIGEST 2026-09-29 cycles 2–3" (PM decisions) | A2.0 preamble | `V/DIGEST-2026-09-29.md` | `~D/DIGEST.md` |
| `R-2026-09-27-e0d-statistic-amended` (the amended ruling's text) | A2.0, A2.13 | `V/owner-drafts/R-2026-09-27-e0d-statistic-amended.md` | `~P/owner-drafts/` |
| v2, v1 drafts, the v2→v3 diff, the P0.3b edit map | v3 header | `V/history/` | `~P/reviews/`, `~D/`, `~D/reports/` |

## 🔴 The owner draft is not a ruling

`V/owner-drafts/R-2026-09-27-e0d-statistic-amended.md` is a **draft**, copied so that
`tests/test_e0d.py` (T14) can test C8 on the real text. It is **not** in
`docs/owner/rulings/`, it was **not** committed by Brendan, and C8 does not read this
directory. Until Brendan commits it under `docs/owner/rulings/`, C8 exits 3 (A2.13).

## Not vendored

- `~P/reviews/superseded/E0D-AMENDMENT-2-FINAL-v2.partial-2026-09-27T1912.md` and
  `~D/superseded/…` (earlier partial drafts, named only as housekeeping in v3's header).
- `~/Downloads/E0d_RoundTable_Synthesis.md` and the R-STAT / R-EVICT / R-JEV context
  reports. The PREREG does not cite them by path.
- `lookahead-room-r2` `D.pt` (A2.0). It is a run artefact that the PREREG names only to
  disclose that it exists.
