status: RETURNED
run_id: b2-psi-probe
updated: 2026-09-29T17:40:00-07:00
provenance: 80c1cd78e4f0151aa4878e526c4acfbbf4b2b3bf (dirty False) · cpu (Mac Studio M4 Max, 6 children x 1 thread) · no dataset hash in the manifest: synthetic generator, EVAL ids [940000, 941024) per seed (N_E = N_tier2 = 1024) · seeds [0, 1, 2]
manifest: runs/b2-psi-probe/manifest.json  (config hash 27b112406ec1d601a3d460c109f4df347d069c9230704df2bf5981aee598f056)
falsifier: B2 is kill-gate evidence (PLAN-v4 §0.2), not a spec falsifier. E0h is falsifier 3c, with its own entry point and rc (A1.3).
expected: PREREG §13: HARMFUL ~40%, MIXED ~40%, NOT LEARNABLE ~10%, NOT RULED OUT ~10%. psi-C closer to FIFO than psi-U. Substantial age decodability. Q2-WIN unlikely. E0h within-step R^2 > 0.49 likely.
observed: §9.6 row 1, HARMFUL (ckpt3000, gamma 0.9, gating). psiU = LOSS / LOSS / EQUIV; psiC = EQUIV x3. The companion (U+, C+) is HARMFUL; ckpt2500 is HARMFUL; moving = false; gamma 0 (secondary) is HARMFUL at both checkpoints. Row 1 says: no build recommendation, and none is made. E0h returned rc 2 (unratified) at both checkpoints. The ckpt3000 headline within-step R^2 is 0.4452 / 0.4756 / 0.5454; the unratified reading is INTERMEDIATE.
gates:     B2 EVAL parent rc 0 (`b2-eval.rc` = 0, manager's launch). E0h: `run.py e0h` rc=2, `run.py e0h --checkpoint 2500` rc=2. Both were slot-wrapped, printed nothing to stdout or stderr, and wrote phaseB/e0h-ckpt{3000,2500}.json. results_report.py rc=0, and its checks were {'worst_abs_diff': 0.0, 'labels_ok': True, 'class_ok': True}.
ledger:    runs/b2-psi-probe/ledger.json
numbers:   Everything is in experiments/b2-psi-probe/RESULTS.md, which is generated from ledger.json, the phaseB/*.pt payloads, the *.logs.jsonl.gz files and the e0h-*.json files. Gating Δ (arm − ref, all-query) at ckpt3000 gamma 0.9: psiU −0.0784 [−0.0847, −0.0716] / −0.0811 [−0.0865, −0.0753] / −0.0202 [−0.0254, −0.0147]; psiC +0.0063 / +0.0085 / +0.0094. δ = 0.0380 / 0.0368 / 0.0319. Ref U: age / fifo / fifo; ref C: fifo x3.
BRIEF ERRORS:
  - "PREREG §11 L535 says HARMFUL is exit 0": that line is §12 (exit codes), file line 535. §11 is the controls section. The substance is correct.
  - The brief asks for "hit rates" on the gap strata. The PREREG's metric (§9.1) is model-read answer accuracy, and "residency" is the model-free hit. RESULTS reports both: accuracy in §2 and replayed residency in §2.2.
  - The brief's E0d AUROC figures 0.942 / 0.931 / 0.943 are the bos-excluded population (`primary.AUROC_strat_pct`), not the "all" population (0.949 / 0.940 / 0.950). The sink value 0.25 is also bos-excluded (0.2458; "all" is 0.2240). RESULTS says which population it uses.
UNANSWERED BY THE BRIEF:
  - Whether Q2's seed-1 Q2-WIN should be read at all. The kind-oracle scores below FIFO on seed 1 (0.6354 vs 0.7685), as B0 predicted (assert < query class mean), so that WIN is against a broken comparator. I reported it as not holding §13 and flagged it; I did not reclassify it.
  - §9.5 EQUIV does not check non-inferiority. psiU seed 2 is EQUIV while its gap_2_to_M Δ lower bound is −0.1264 against −δ = −0.0319, and psiC seed 2 is EQUIV with −0.0357. I reported this and did not reclassify. It is the owner's call whether that is a PREREG defect.
BELIEVED, NOT VERIFIED:
  - The A1.5 logit control worst is exactly 0.0 on every child. That is plausible, since the pre-hook recomputes the same ops in the same order on CPU. I did not independently check that the control can fail.
  - Phase A's control C3 records max_abs_acc_diff 2.9e-8 and ok=True, while PREREG §11.3 asks for float32 "==". I did not re-derive whether lookahead-room's control1 rule is exactly W10 A1's rule.
  - Phase A's ledger provenance says dirty=True at a94b228. I did not check what was dirty. The phase-A fits are inputs to this run.
  - The scratch disclosure (EVAL ids 940000–940031 and 940000–940001, and R^2 on 4 seed-0 docs, "no outcome read") is taken from the brief. I have no record of my own for it.
  - The code defect behind the per-eviction jsonl's `arm` field: it carries the inner policy label. "kind_oracle" is used for both kind and kind_oldest (ambiguous at gamma 0 tier 2), and "random" for all five random arms. The payload attribution is keyed correctly. RESULTS re-reads the log only for arms that are unique in Tier 1.
  - The interpretive note linking the attribution to E0d's poor bottom-1 is inference and is labelled that way.
NEXT (proposed, not decided): Brendan reads the HARMFUL class together with E0d's AGREE, and rules on the E0h thresholds (0.90 / 0.49) so that E0h can exit 0 or 1. Optionally, before any reading of Q2, fix the jsonl arm-label ambiguity in run.py (the log's `arm` field should use the tier arm name). No further compute is proposed.
