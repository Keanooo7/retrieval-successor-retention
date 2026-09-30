status: RETURNED
run_id: newcomer-bakeoff
updated: 2026-09-30T06:40:00Z
provenance: 16f867a3d530e87fe4c59d85b03b9626d32682e6 · cpu (1 thread/eval child, 4/fit child) · EVAL_NB [990000, 992048) per seed, generator seed s·1_000_003+id · seeds [0, 1, 2]
manifest: runs/newcomer-bakeoff/manifest.json  (config hash dc615c9ee5152a7f7143857ddbfc4df9898192546a09dcd62b7600847f72fa59)
falsifier: NONE directly. MEASURE-THEN-DECIDE evidence for ADR-0009 L3/L4 and the pack §4 newcomer options; §7.1 vacuity (falsifier 1) is read on the effective rule as a report, not a gate.
expected: PREREG §9: replication ~70%; DQ1 NO 60 / MIXED 30 / YES 10; DQ2 INTERMEDIATE 40 / HARMFUL 35 / SAFE 25; DQ3 YES 55 / NO 25 / other 20.
observed: DQ1 NO; DQ2 INTERMEDIATE; DQ3 YES; B2 replication HARMFUL (psi-U LOSS/LOSS/EQUIV, psi-C EQUIV x3). Offline only; no build recommendation; grace acceptability OWNER-ONLY.

gates:     run.py run rc 0 (reports/nb-run.rc); T0 shasum -c rc 0, 435 OK at start and end;
           fit controls C4-C9 pass all seeds, C5 C@0.9 refit bit-identical to B2 (argmin 32768/32768);
           C6 fifo/psiC/psiU FIT_VAL acc == B2; full suite at f0fba46 rc 1 passed=1728 failed=2
           (both pre-existing at b1c34a6, fixed in 623166b/e4a6261); battery_subset x6 at e4a6261 all PROVEN.
ledger:    runs/newcomer-bakeoff/ledger.json
numbers:   see experiments/newcomer-bakeoff/RESULTS.md (generated from ledger keys decision.*, seed{s}.*, fit.seed{s}).
BRIEF ERRORS: (1) pack §4.2B/§4.3 grace "age < g" contradicts its own "g=1 blocks 36.8% (age-1 victims)"; under B2's convention age<1 protects nothing -- PREREG uses age <= g. (2) "K-truncated" R1: K=40 is inert at S=48 under FIFO (0 truncated cells, measured). (3) union.py refuses one-sided tail edits (main's I5), so "use union.py, never by hand" could not be followed literally; used union.py + one rule (reports/union_tail.py). (4) B2's committed tree had 2 red tests, which blocks the mutation battery; brief did not anticipate.
UNANSWERED BY THE BRIEF: whether a non-gating λ=1 companion (d') was wanted (added, declared); N_E for the fresh range (fixed at 2048 pre-data); which "content" the partial rho conditions on (generator kind/status label, declared).
BELIEVED, NOT VERIFIED:  transfer of offline harm to online MC; that pending-assert under-prediction (not age) is the mechanism; the seed-1 R1 result is not driven by ref_R1L1=age on that seed only.
NEXT (proposed, not decided): owner ruling on grace (OWNER-ONLY); a λ_shadow dose sweep (0.25, 0.75) or a rank-1..3 R1 to see whether any counterfactual weight is non-harmful on seed 0; nothing is proposed as a build.
