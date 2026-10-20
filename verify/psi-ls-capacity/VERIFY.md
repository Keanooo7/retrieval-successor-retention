# VERIFY: psi-ls-capacity

Verifier: rsr-verifier (Claude Opus 5.5, a model; not Brendan). 2026-10-08.
Run under verification: `psi-ls-capacity`, branch `diag/psi-ls-capacity`, run head `97a20ce`,
results commit `90c2720`. PREREG `87dc2ec` + Amendment 1 `f99f6ce` read in full. No RESULTS.md
was read (neither `experiments/psi-ls-capacity/RESULTS.md` nor `runs/psi-ls-capacity/RESULTS.md`).
Commands were executed from `.worktrees/psi-ls-capacity` at HEAD `90c27203ea5c01fdbe5993b0025a0a9f8fe93c5c`.
The only commit after the run head, `90c2720`, touches only `experiments/psi-ls-capacity/RESULTS.md`
(see S5), so the code that ran is the code at `97a20ce`.

## Overall status: INCONCLUSIVE

One drawn claim is CONFIRMED. Two drawn claims were re-executed, but **their outputs were never
observed**. The permission system refused the verifier's read of the captured stdout files
(auto-mode classifier, reason "Interfere With Workloads"). The verifier did not route around that
refusal, so those two claims are UNVERIFIED. They are not refuted. All seven supplementary checks
pass.

Machine record: `verification.json` (this directory; also at
`.worktrees/psi-ls-capacity/runs/psi-ls-capacity/verification.json`). `orchestrator.verify check`
returned `OK ... is a valid record` with exit 0. The schema requires a boolean `match`, so the two
unobserved claims carry `match: false`, and their `observed` field says they were not observed.

## Draw

```
PYTHONPATH=scripts .venv/bin/python -m orchestrator.verify --root <psi-ls-capacity worktree> draw psi-ls-capacity
rc=0
"seed": "fe00b510ad84ba72aeaee37ca372499bbb53705c7cf4b16d7e9aa41f7c822556"   (== sha256("psi-ls-capacity"), checked independently)
"n_claims": 6, "k": 3, chosen indices: 0, 1, 2
```

The tool read the claims file, so the hand fallback was not used.

## Drawn claims

Each was run as
`PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run --lane cpu-det --slots 1 --job-id verify-psi-ls-<n> -- <command> > out 2> err; rc=$?`

| # | claim | command | expected | observed | verdict |
|---|---|---|---|---|---|
| 0 | seed 0 cf_all U-val and U-train caps of the saved LS phi | `.venv/bin/python experiments/psi-ls-capacity/run.py recompute cap --seed 0` | `0.38589211618257263 0.409457478005865` | not observed (read refused) | UNVERIFIED (inconclusive) |
| 1 | seed 0 cf_all LS re-solved from the inputs: train and U-val L_MC, U-val cap | `.venv/bin/python experiments/psi-ls-capacity/run.py recompute solve --seed 0` | `1.2209696423420429 1.2468908153534253 0.38589211618257263` | `1.2209696423420429 1.2468908153534253 0.38589211618257263` (rc=0, stderr empty) | CONFIRMED (string `==`) |
| 2 | seed 1 cf_all U-val and U-train caps of the saved LS phi | `.venv/bin/python experiments/psi-ls-capacity/run.py recompute cap --seed 1` | `0.527114967462039 0.5488077496274218` | not observed (read refused) | UNVERIFIED (inconclusive) |

Claim 1 re-solves the normal equations from the pinned inputs, independently of the saved phi.
It reproduced the seed 0 U-val cap `0.38589211618257263` bit-exactly. That is the same number as
claim 0's first field, which partly corroborates claim 0, but claim 0's own output (from the saved
`phi-cf_all-ls.pt`) was not observed.

## Supplementary checks (read-only)

**S1. Labels re-derived.** The rule was written independently, not through `fit_label`:
POINTS if every lo ≥ 0.80, else PARTIAL if every point ≥ 0.50, else MISSES.
- cf_all `cap_val`, point/lo: s0 0.38589/0.33755, s1 0.52711/0.47139, s2 0.34513/0.29032. This is **FIT-MISSES**: no lo reaches 0.80, and s0 and s2 have points below 0.50.
- cf_all `cap_train`, point/lo: s0 0.40946/0.38743, s1 0.54881/0.52575, s2 0.38504/0.36277. This is **IN-SAMPLE-MISSES**.
- `cutoff_sweep` at ×0.01 (ranks 14378/13839/15257): `cap_val` points 0.38382/0.51844/0.35398 give FIT-MISSES. `cap_train` points 0.42815/0.54583/0.39130 give IN-SAMPLE-MISSES. At ×0.1 the labels are also MISSES/MISSES.
- cf_live gives FIT-MISSES / IN-SAMPLE-MISSES at the registered cutoff, at ×0.1 and at ×0.01.
- `summary.json` says `FIT-MISSES`, `IN-SAMPLE-MISSES`, `labels_at_x0.01` [FIT-MISSES, IN-SAMPLE-MISSES], `cutoff_sensitive` false, `cf_live` [FIT-MISSES, IN-SAMPLE-MISSES] and `controls_ok` true. **Consistent. PASS.**

**S2. Controls.** Every seed has `controls_ok=True` and `exit=0`.
- C1: the gestalt sha256 equals the PREREG §1 pins (211244f1…, 3a18e291…, 74e3bec7…). The Δ sha256 equals the offline-psi PREREG §1 pins (1c047a36…, bf2753ce…, c880a241…). The diagnostic A `result.json` sha256 equals the PREREG §1 pins (8c88bd05…, 0bb3ee8c…, 496beeb4…).
- C2: `ok` for every seed and arm, with 96 entries checked each time. The worst relative error is 4.05e-16.
- C3(a) is gated on `kkt_rel_kept` (Amendment 1), and the code gates `kkt["kept"] <= 1e-6` (run.py L400-403). Measured cf_all values are 1.27e-10, 1.00e-10 and 9.43e-11. cf_live values are 7.37e-11, 7.60e-11 and 5.97e-11. The ungated `kkt_rel_full` values are 5.3e-7 to 1.06e-6, which is what Amendment 1 predicts.
- C3(b) was re-checked as `train_l_mc_head <= (1+1e-6)*diag_a_train_l_mc`:

  | arm | s0 | s1 | s2 |
  |---|---|---|---|
  | cf_all | 1.22097 ≤ 1.56352 | 1.19493 ≤ 1.46936 | 1.32429 ≤ 1.64922 |
  | cf_live | 0.82139 ≤ 1.10443 | 0.78376 ≤ 1.03015 | 0.91189 ≤ 1.20327 |

  All true.
- C3(c): `quad_rel` is at most 1.27e-10. C4 and C5 are ok.
- **PASS.**

**S3. EVAL isolation.** The ids were read directly from the files:
- the gestalts (`.worktrees/offline-psi/runs/offline-psi/seed{s}/gestalts.pt`);
- Δ (`.worktrees/cf-ceiling/runs/cf-ceiling/raw/B-ckpt3000-seed{s}.delta.pt`);
- `u_documents`.

For every seed and every source there are 1088 ids, ranging from 64 to 4159, with **0** in
995000–997047. The gestalt ids equal the U ids. In code, `CV._seal_eval()` runs before
`load_inputs` in `run_seed` (run.py L441-442) and in `recompute` (L747-748). C5 records
`sealed: [deploy_pass, deploy_policy, docs_by_id]`. **PASS.**

**S4. phi sha256.** All six `phi-{cf_all,cf_live}-ls.pt` files hash to `result.json` `phi_sha256`.
The cf_all hashes are 3a798de2…, 3b3fb670… and 70736cb3…. **PASS.**

**S5. Run head clean.** `result.json` `tree` for all 3 seeds is
`{'sha': '97a20ce20c4a750e902f4810252ec00f1fcb0f45', 'dirty': False, 'porcelain': ''}`.
`git log 97a20ce..90c2720` lists only `90c2720`, and `git diff --stat 97a20ce 90c2720` shows
`experiments/psi-ls-capacity/RESULTS.md | 114 +` only. **PASS.**

**S6. Launch log.** `runs/psi-ls-capacity.launch.log` reads, literally:
`seed 0 rc=0 / seed 1 rc=0 / seed 2 rc=0 / finalize rc=0`. The per-seed logs say
`seed s exit 0 controls_ok True`. **PASS.**

**S7. finalize reproduces.** `runs/psi-ls-capacity` was copied with `cp -Rp` to a scratch dir.
`run.py finalize --out-root <scratch copy>` then ran under the cpu-det slot, with rc=0. `cmp` on
`summary.json` and on `claims.json` both gave rc=0. sha256: summary `d95aaede…a5d3`, claims
`b68ebf19…bbbe`, and both are equal to the originals. The original run dir was not written.
**PASS.**

## Incidents (declared)

- A stray bare `.venv/bin/python` (no script) was started by mistake in one of my background
  shells and sat idle waiting on stdin. My attempt to kill it was refused by the permission system.
  It does no compute and holds no lane slot, and the harness stops it at its 30 min background
  limit.
- The refusal of the follow-up read of claims 0 and 2's captured outputs is why those claims are
  unverified.

## Not done

No code was edited, nothing was fixed, and no recommendation is made about what runs next.
§15 is untouched.

## Addendum: claims 0 and 2 re-run

2026-10-08, rsr-verifier (Claude Opus 5.5, a model; not Brendan). This supersedes the
**Overall status: INCONCLUSIVE** above. The new overall status over drawn claims 0, 1 and 2 is
**OK**. The text above is left unedited as the record of attempt 1.

Commands were executed from `.worktrees/psi-ls-capacity` at HEAD
`90c27203ea5c01fdbe5993b0025a0a9f8fe93c5c`, the same tree as attempt 1. No RESULTS.md was read.
The draw was re-run first:
`PYTHONPATH=scripts .venv/bin/python -m orchestrator.verify draw psi-ls-capacity`, with rc=0, seed
`fe00b510ad84ba72aeaee37ca372499bbb53705c7cf4b16d7e9aa41f7c822556`, n_claims 6, k 3 and chosen
indices 0, 1, 2. That is identical to attempt 1.

Comparison: stdout's last line was compared with `expected` using an exact string `==` in zsh
(`[[ "$(tail -n 1 out)" == "<expected>" ]]`).

### Claim 0

```
PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run --lane cpu-det --slots 1 --job-id verify2-psi-ls-0-seq -- .venv/bin/python experiments/psi-ls-capacity/run.py recompute cap --seed 0 > claim0.out 2> claim0.err; rc=$?
rc=0
stdout (1 line): 0.38589211618257263 0.409457478005865
stderr: (empty, 0 bytes)
expected:        0.38589211618257263 0.409457478005865
```

**Verdict: CONFIRMED** (string `==`).

### Claim 2

```
PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run --lane cpu-det --slots 1 --job-id verify2-psi-ls-2-seq -- .venv/bin/python experiments/psi-ls-capacity/run.py recompute cap --seed 1 > claim2.out 2> claim2.err; rc=$?
rc=0
stdout (1 line): 0.527114967462039 0.5488077496274218
stderr: (empty, 0 bytes)
expected:        0.527114967462039 0.5488077496274218
```

**Verdict: CONFIRMED** (string `==`).

### Claim 1 (carried from attempt 1)

`recompute solve --seed 0` gave observed `1.2209696423420429 1.2468908153534253 0.38589211618257263`
with rc=0. That was **CONFIRMED** in attempt 1 and was not re-run.

### Incidents (declared)

- The first execution of claims 0 and 2 (job ids `verify2-psi-ls-0` and `verify2-psi-ls-2`) was
  issued as two concurrent tool calls. The instruction was one at a time. Both went through the
  cpu-det slot wrapper with `--slots 1`, and both returned rc=0 with empty stderr and stdout
  identical to the values above. They are **not** the recorded observations.
- I then ran each claim again, alone and in sequence. Reusing job id `verify2-psi-ls-0` returned
  rc=3, with stderr `DID NOT RUN: job record .../.orchestrator/jobs/verify2-psi-ls-0.json already
  exists`. That is a did-not-run, not a result.
- The recorded runs are the sequential re-runs under fresh job ids, `verify2-psi-ls-0-seq` and
  `verify2-psi-ls-2-seq`, shown above. All three executions of each claim printed the same string.

### Record

- `runs/psi-ls-capacity/verification.json` from attempt 1 was moved, not deleted, to
  `runs/psi-ls-capacity/verification.attempt1.json` (in `.worktrees/psi-ls-capacity`).
- The new `runs/psi-ls-capacity/verification.json` has status `ok`, 3 claims, every `match: true`,
  and verifier_sha `90c27203ea5c01fdbe5993b0025a0a9f8fe93c5c`. It is copied to this directory as
  `verification.json`, replacing attempt 1's copy, which is preserved in git history at `7373528`
  and as `verification.attempt1.json` in the run dir.
- `PYTHONPATH=scripts .venv/bin/python -m orchestrator.verify check psi-ls-capacity > check.out 2>&1; rc=$?`
  gave rc=0 and `OK: .../.worktrees/psi-ls-capacity/runs/psi-ls-capacity/verification.json is a valid record`.

No code was edited, nothing was fixed, and no recommendation is made about what runs next.
§15 is untouched.
