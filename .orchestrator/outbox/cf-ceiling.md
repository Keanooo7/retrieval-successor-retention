status: RETURNED
run_id: cf-ceiling
updated: 2026-10-06T12:50:00Z
provenance: 03859e2b7709db6f466809bfa45bb9910dbeb54d (dirty False) · cpu, 1 thread, cpu-det 1 slot · inputs sha256 = T0 manifest (6 ckpts + 6 D.pt, C1 start and end) · seeds [0, 1, 2], 1088 U docs each, ckpt3000 and ckpt2500
manifest: runs/cf-ceiling/manifest.json  (config hash d8b05c1e0e4b45566d35955b07ba8d9d343dbb4aac6335eaa7447c14eebee506)
falsifier: NONE. Descriptive Tier D run; the PM's pre-registered reading rule (PREREG §7, 0c5320e) classifies the run as an input to an owner decision on option C3 (spec §3.2.1).
expected: PREREG §9: predicted CF-MIXED (low confidence; CF-POINTS next); Δ caps above the r_i caps at γ ≥ 0.5 on every seed; γ = 0 MISSES; C3 exact.
observed: CF-POINTS. At ckpt3000, γ 0 MISSES, 0.5 MISSES, 0.8 PARTIAL, 0.9 POINTS (cap +0.916 / +0.955 / +0.874, CI lower 0.895 / 0.935 / 0.852). The cap rises monotonically with γ on every seed at both checkpoints. C3 is exact. The age-shape diagnostic is negative (newer slots carry larger G), the opposite sign to r_i.
gates:     full suite at launch head 03859e2: passed=2109 failed=0 skipped=0 errors=0, rc 0. Run: rc=0 (ledger commands[0].exit_code 0), controls_failed []. First suite attempt at b9bc032: passed=2107 failed=2 skipped=0, so the launcher stopped and did not run (see BRIEF ERRORS 3; fixed in 1e982ee and 03859e2). tests/test_cf_ceiling.py: passed=24 failed=0 skipped=0.
ledger:    runs/cf-ceiling/ledger.json (claims runs/cf-ceiling/claims.json, 10 claims; claims 0-4 and 6-8 re-executed by me and matched; the batch-0 recheck of ckpt3000 seed 1 reproduced 639.7262362596285 / 700.846475776684 exactly)
numbers:   ckpt3000 primary cap point [lo, hi], seeds 0/1/2 (B.ckpt3000.cf.<g>.cap.*):
           g0 −0.780 [−0.837, −0.727] / −0.653 [−0.712, −0.599] / −0.835 [−0.898, −0.776] MISSES
           g05 +0.045 [+0.018, +0.073] / +0.109 [+0.080, +0.137] / +0.070 [+0.037, +0.100] MISSES
           g08 +0.620 [+0.598, +0.641] / +0.653 [+0.633, +0.675] / +0.587 [+0.566, +0.610] PARTIAL
           g09 +0.916 [+0.895, +0.937] / +0.955 [+0.935, +0.976] / +0.874 [+0.852, +0.896] POINTS
           sd of cap point over seeds: 0.094 / 0.032 / 0.033 / 0.041.
           refs (U.hit.*.all): FIFO 0.8059/0.8093/0.8088, factfiller 0.9655/0.9653/0.9666, oracle 1.0.
           r_i control (B.ckpt{c}.ri.*): equals runs/gamma-ceiling/ledger.json exactly (24 pairs + 2 refs).
           Sensitivity Δshift (DIAGNOSTIC) ckpt3000 g09: +0.962 / +0.999 / +0.912.
           ckpt2500 primary g09 (unlabelled): +0.908 / +0.931 / +0.885.
           Age-shape median ρ, ckpt3000 primary g09: −0.197 / −0.156 / −0.250 (B.ckpt3000.cf.g09.age_rho.median).
           Coverage, full memory: 0.99905 / 0.99916 / 0.99900; identity probe worst 0.0.
           Wall time 2 h 06 min (1210 s per checkpoint × seed).
BRIEF ERRORS:
  1. The control "reproduce the γ sweep's per-document values exactly" names an artefact that does not exist. runs/gamma-ceiling/ledger.json holds aggregates only (hit by bucket, cap point/lo/hi, refs), not per-document vectors. With the full U set (no subsample was needed), I met it by exact equality of every aggregate, which is computed from the per-document vectors and the identical replicate matrix (PREREG §6 C3 says so).
  2. The feasibility formula "M=16 knockouts × steps" overcounts the resident pass: it is Σ_t min(t, 16) = 632 forwards per document, not 16 × 48 = 768. It also omits the evicted probe: 528 rows per document, Σ (t − 15) over t = 16…47, identity rows included. That is immaterial to the decision.
  3. Carried from the previous cycle: the brief's base f31cb8b (the γ sweep's outbox commit) left .orchestrator/outbox/researcher.md stale. test_the_committed_index_is_current therefore fails on any branch cut from it, and it failed my first launch-head suite. Fixed by regenerating the index with the tool (1e982ee). The γ-sweep branch still has the defect.
  4. The launch trigger in the brief (rsr-run1-launch.rc) would have overlapped run 1's held-out measurement. The PM corrected it mid-task to finalize.rc, and I followed the correction.
UNANSWERED BY THE BRIEF:
  - The evicted Δ's sign was not specified. I chose "loss decrease on reinstatement" so that positive means "helps" for both kinds of reading.
  - Neither a missing-donor policy nor a batch / donor pool was specified. I chose to impute 0, use batch 32, and stop below 0.80 coverage (E0d's value). Coverage came out at 0.999.
  - The γ set stops at 0.9, and the cap is still rising there. Whether it turns over above 0.9, as the r_i cap did above 0.8, is unmeasured.
  - Whether the oracle reference should be reported. I reported it but did not use it.
BELIEVED, NOT VERIFIED:
  - That the negative age-shape and the Δshift arm's slightly higher caps come from the evicted-reading offset that PREREG §3 predicted (evicted readings average about −0.008 and resident about +0.005 away from the query step). This was not isolated.
  - That the caps are insensitive to donor batch composition (batch 32 against E0d's 16). Only one batch size was run.
  - That the roughly 0.1 % imputed-zero cells (mostly filler) do not move any cap.
  - That POINTS reflects a usable target, as opposed to γ = 0.9's larger hindsight advantage. A hindsight cap is not a learnability result.
  - That the probe, smoke and test runs I made while run 1 trained (1 slot each) did not perturb run 1.
NEXT (proposed, not decided):
  1. rsr-verifier on run/cf-ceiling @ 651fe63 (claims.json), then a Tier D review of the code and RESULTS.
  2. Owner decision: CF-POINTS puts option C3 (a LOO-derived target, spec §3.2.1) in front of Brendan for RSR run 2. That is his call.
  3. If the owner wants it: a Tier D amendment extending γ to {0.95, 0.99} on the saved Δ files. It is model-free and takes about 5 min, because the raw Δ is reused by key.
  4. If the owner wants it: a donor batch-size sensitivity check (16 against 32) on ckpt3000, about 1 h.
