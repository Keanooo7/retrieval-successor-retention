status: RETURNED
run_id: gamma-ceiling
updated: 2026-10-06T07:00:00Z
provenance: 02a7eeb (main run) / a0547a6 (Amendment 1 run) · cpu (Mac Studio M4 Max, cpu-det lane) · lookahead-room-r2 D.pt, sha256 = ~/rsr-substrate/2026-09-27/MANIFEST.sha256 (T0 manifest sha256 33dd493d…565f) · seeds [0, 1, 2]
manifest: runs/gamma-ceiling/manifest.json  (config hash 6d6d512d1f2b8555d4c8e7da614674018f2f4075ee0a9a2576163c1cdeb53b06); Amendment 1: runs/gamma-ceiling-ageshape/manifest.json (config hash 78138ff9154254e75c45d090d802df50b564a8191f186920f84cf1ec1280f4fa)
falsifier: NONE — descriptive. The PM's pre-registered reading rule (PREREG §6) classifies each gamma and the sweep, as an input to an owner decision only.
expected: PREREG §9 (daabdb2): cap rises with gamma to 0.95-0.99, flat or lower at 1.0; no gamma POINTS; predicted class SIGNAL (low confidence); C3 exact.
observed: Sweep class SIGNAL: every gamma MISSES at ckpt3000. Cap peaks at gamma = 0.8 on every seed (+0.406 / +0.210 / +0.609) and falls monotonically above it, going negative (point estimate) on seeds 0 and 1 at gamma >= 0.95. The same shape holds at ckpt2500. The expected rise past 0.9 was wrong. C3 reproduced B0 exactly.
gates:     main run rc=0 (ledger status ok; controls_failed []; C1 start/end ok; C3 exact over 16 B0 key pairs + FIFO/factfiller); Amendment 1 run rc=0 (status ok; C1 ok; A1-C pass-through exact); tests/test_gamma_ceiling.py passed=24 failed=0 skipped=0 errors=0 (rc 0); tests/test_b0_ceilings.py run alongside earlier: combined passed=35 failed=0 skipped=0 errors=0
ledger:    runs/gamma-ceiling/ledger.json; runs/gamma-ceiling-ageshape/ledger.json
numbers:   see experiments/gamma-ceiling/RESULTS.md; ckpt3000 cap point [lo,hi] per seed from B.ckpt3000.<g>.cap.{point,lo,hi}:
           g0   -0.039 [-0.073,-0.006] | -0.136 [-0.164,-0.107] | +0.234 [+0.204,+0.260]
           g05  +0.331 [+0.304,+0.358] | +0.116 [+0.092,+0.141] | +0.512 [+0.488,+0.538]
           g08  +0.406 [+0.379,+0.434] | +0.210 [+0.186,+0.236] | +0.609 [+0.584,+0.633]
           g09  +0.184 [+0.147,+0.220] | +0.134 [+0.103,+0.165] | +0.537 [+0.508,+0.564]
           g095 -0.077 [-0.125,-0.033] | -0.032 [-0.073,+0.006] | +0.433 [+0.401,+0.463]
           g099 -0.308 [-0.361,-0.259] | -0.265 [-0.311,-0.220] | +0.322 [+0.286,+0.357]
           g1   -0.359 [-0.413,-0.307] | -0.329 [-0.378,-0.284] | +0.300 [+0.263,+0.335]
           labels B.ckpt3000.label all MISSES; sweep_class SIGNAL. Age-shape (DIAGNOSTIC), B.ckpt3000.<g>.age_rho.median: positive on all cells except g0 seed 1 (-0.076) and g05 seed 1 (+0.009, CI spans 0).
BRIEF ERRORS: (1) The PM's Amendment 1 instruction ("append ... BEFORE any data run") arrived after the main run had completed and been seen (c186ac6). It is committed alone as c4c1a25 and says so explicitly; it precedes only the age-shape numbers. (2) "-n 2 max": pytest-xdist is not installed at 94a7ceb (`-n` → usage error, rc 4); tests ran serially. (3) The censoring question offered two options (uncensored with every past sentence resident, or FIFO-censored). D.pt is neither: it is uncensored, but each value is a single-substitution probe in FIFO's rollout (lookahead-room/run.py:500-502, 504-508, 513, 517, 534).
UNANSWERED BY THE BRIEF: Which world's resident set the age-shape rho should use. I chose the hindsight rule's own 16 live slots at its eviction steps (fixed in Amendment 1 before any value).
BELIEVED, NOT VERIFIED:  "Amendment 4" (gamma = 0.9 for run 1) is not in docs/owner/rulings at 94a7ceb; I believe it is on main (f143ade) and did not check. The age-correlated mix of resident-type and probe-type readings in G (PREREG §2 item 3) is an argument from the code, not a measurement.
NEXT (proposed, not decided): Measure demand in the rule's own world, not FIFO's, because every G here is a FIFO-world counterfactual, so these rows cannot separate "no gamma has the signal" from "the instrument measures the wrong world". Alternatively, run the same sweep on a fresh, uninspected document range to get an out-of-sample read of the gamma = 0.8 peak.
