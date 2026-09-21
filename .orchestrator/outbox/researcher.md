status: RETURNED
run_id: decisive-shuffle
updated: 2026-09-21T13:36:39Z
provenance: 90438f3d2a9c1998cca37069a7b4f7ded649a4b9 (ledger git_sha; dirty=true from the untracked uv.lock only) · cpu, torch 2.14.0, py 3.14.6, 1 thread per process, 9 concurrent training subprocesses · corpus = SyntheticConfig defaults (answer_in_stream=True), train batch = first 8 training docs, held-out = S0-03 held-out docs 64..71 · seeds actually run [0,1,2] in every arm, steps 300/300, status ok
manifest: runs/decisive-shuffle/manifest.json  (config hash 2a299e4afc2dc074ff375d4a38ee8684ae9a7cc3b5bd638e166185bbda487c8d; manifest_written_utc 2026-09-21T11:28:59Z, before any training step)
falsifier: "the TG working memory is inert on a corpus whose objective requires retrieval" -- PREREG rule (#17 Amendment 1) per arm
expected: "B and C live (some seed ratio >= 0.1) ... A uncertain, most likely inconclusive or inert ... If B and C read inert, the memory path is broken" (manifest `expected`, written before the run)
observed: A INERT, B LIVE, C LIVE. Overall (PREREG "Which outcome"): LIVE on B and C -> the memory path is not broken; the corpus alone (A) was not enough. Ledger verdict: falsified.
  BUT (secondary, descriptive): A's inert shuffle coincides with cross-row cosine 0.9995 and a matched-norm random replacement >= 0.1 on every seed -> A's memory is READ but ROW-AGNOSTIC (the PREREG's second rival), not ignored. And B/C held-out answer NLL with own memory is ~2.9 nats in every gap bucket, at/above ln16: "live" means memory moves the tokens, not that it retrieves the right answer.

gates (final tree, PR head b1b461a + outbox commit, after merging origin/main ed8bdb0):
  uv run pytest -rs --tb=no  -> passed=420 failed=0 skipped=0 errors=0 (420 passed, 1 xfailed) ; rc=0   (the 1 xfailed is the strict xfail in tests/test_train_loop.py:310; 0 skipped)
  uv run ruff check src/ tests/ scripts/          -> All checks passed! ; rc=0
  uv run ruff format --check src/ tests/ scripts/ -> 75 files already formatted ; rc=0
  uv run python scripts/mutation_battery.py --check -> 71/71 gates proven by mutation ; rc=0
  uv run python scripts/render_scoreboard.py --over runs/ --audit experiments/decisive-shuffle/RESULTS.md -> OK: every number in experiments/decisive-shuffle/RESULTS.md resolves to a ledger key ; rc=0
  git diff --stat origin/main -- runs/ -> only runs/decisive-shuffle/ (12 files: ledger, manifest, raw, 9 result.json); no .pt tracked (9 checkpoints stay ignored on disk)
mutation bar (node ids that redden, measured by applying each battery entry and running tests/test_decisive_shuffle.py):
  arm swap in ARMS            -> FAILED tests/test_decisive_shuffle.py::test_arms_are_the_preregistered_table
  manifest-hash check no-op   -> FAILED ...::test_an_arm_swap_is_refused_by_the_manifest_check[A-B] / [B-C] / [C-A]
                                 (+ declared off-gate ...::test_a_config_that_does_not_hash_to_its_own_stamp_is_refused)
  decoy aliasing clause gone  -> FAILED ...::test_ratio_exactly_one_is_inconclusive_not_live
                                 FAILED ...::test_the_decoy_pointed_at_the_trained_checkpoint_reads_ratio_one_and_inconclusive
  measure() ignores decoy_ckpt-> FAILED ...::test_the_decoy_pointed_at_the_trained_checkpoint_reads_ratio_one_and_inconclusive
  (that test drives measure(ckpt, decoy_ckpt=ckpt) end to end: A_decoy == A_trained exactly, ratio == 1.0, verdict inconclusive)
  + 4 liveness entries (random-self perturbed, random not norm-matched, answer mask ignored, cosine constant): each PROVEN, 1 on gate, 0 off.
ledger:    runs/decisive-shuffle/ledger.json  (0 statistic rows sd_exactly_zero)
numbers (train batch = PRIMARY; ledger keys armX.ratio etc.; seeds 0/1/2; mean +/- sd):
  armA.ratio   0.0004656 / 0.003647 / 0.004921   0.003011 +/- 0.002295  -> inert (every seed <= 0.01)
  armB.ratio   6.742 / 14.73 / 6.502             9.325 +/- 4.683        -> live
  armC.ratio   2.989 / 14.56 / 5.672             7.741 +/- 6.058        -> live
  controls, all 9 (arm,seed) x both batches: memory-disabled A = 0.0, own-memory A = 0.0, random-self A = 0.0, decoy A > 0 (train: 0.01021 / 0.005432 / 0.006189)
  heldout.ratio (descriptive): A 0.002704 +/- 0.001779 · B 8.877 +/- 3.869 · C 8.444 +/- 6.478
  answer tokens, held-out, shuffle ratio mean +/- sd (n targets per seed 34/26/18 gap1, 85/87/98 gap2..M, 31/32/31 gap>M):
     A: gap1 0.001635 +/- 0.001502 · gap2..M 0.002864 +/- 0.002172 · gap>M 0.002764 +/- 0.001985
     B: gap1 18.35 +/- 6.648 · gap2..M 29.31 +/- 24.36 · gap>M 30.84 +/- 22.51
     C: gap1 20.28 +/- 16.38 · gap2..M 34.74 +/- 40.16 · gap>M 37.44 +/- 45.84
  answer NLL, own memory, held-out: A 2.803 +/- 0.06157 · B 2.904 +/- 0.05702 · C 2.852 +/- 0.08828 (answer_all)
  cross-row cosine (train): trained A 0.9995 +/- 0.0005338 · B 0.4833 +/- 0.06134 · C 0.4357 +/- 0.07591 · decoy 0.565 +/- 0.2042
  random_ratio (train): A 2.098 +/- 1.111 · B 14.08 +/- 5.199 · C 13.23 +/- 8.124  (every seed >= 0.1 in every arm)
  honest real-token NLL (train): A 1.681 +/- 0.02436 · B 1.573 +/- 0.03259 · C 1.567 +/- 0.02932
BRIEF ERRORS:
  1. The brief forbids the researcher from seeing any S0-03 number. But the role requires prepending to .orchestrator/outbox/researcher.md, and the top of that file WAS S0-03's report, numbers included. I read its head to copy the header format and saw them. By then the manifest (11:28:59Z), the expectation, run.py, the thresholds and all 9 trainings were already frozen or running, so nothing in the run could have been conditioned on them. It is still a breach of the brief's blinding, and the brief's own report destination caused it.
  2. Bar 6, "`git diff --stat origin/main -- runs/` shows only runs/decisive-shuffle/", cannot hold when main moves during the run, and it did (E-feas s003 #33 and S0-05 #32 landed). The two-dot diff then shows main's new runs as deletions. I merged origin/main (ed8bdb0) into s0/decisive, and after that the check holds. A three-dot diff would have been the right instrument.
  3. "Every existing ... battery entry must stay unchanged" is not literally satisfiable. The existing S0-03 mutation (answer out of band) reddens ANY new test that builds the answer-token mask, and PREREG Secondary 1 requires that mask. I added one declared refusal coupling to that entry (tests/test_decisive_shuffle.py::test_the_decoy_pointed_at_the_trained_checkpoint_reads_ratio_one_and_inconclusive), which follows the pattern its 10 existing couplings already use. No other existing entry changed. My first draft of memory_liveness.py did change the anchor line `out = model(ids_t, mask_t, kv[perm], valid[perm], bc, bv)`, which three battery entries target. I caught that before the counted run, restored the line verbatim, and relaunched (the 04:28 launch was killed at step ~3 and its directory deleted).
  4. Wall clock: the brief's "~15 min per seed" does not transfer to 9 concurrent 1-thread trainings. I measured 4.6 s/iter at 3 concurrent before launch. Under 9-way parallelism (plus the battery) it ran ~14.5 s/iter, about 75 min per training. All 9 were started at 04:29, well before 06:30, and completed.
  5. Files in scope omitted the S0-05 exit-code protocol, which landed on main mid-run. Its tests/test_exit_codes.py::test_every_converted_checker_exits_through_the_protocol reddened on the new run.py after the merge. I converted the exit plumbing only, after the run (commit 942ee4c). The ledger's sha 90438f3 is the pre-conversion script. The measurement and decision code is byte-identical.
  6. Bar 2 lists random-replacement-with-self as a control "on every seed of every arm", but the PREREG decision rule's control list is memory-disabled, own-memory and decoy only. I kept the rule as registered: random-self gates only the random-replacement readout. It read exactly 0.0 everywhere, so the ambiguity did not bite.
UNANSWERED BY THE BRIEF:
  - Does "live" satisfy the Sprint 2 precondition when held-out answer NLL with the model's own memory is ~2.9 nats in every gap bucket (at or above ln16)? The memory moves answer tokens a lot (shuffle ratio 18-37x on answers), but nothing here shows it moves them toward the right answer. That is a retrieval-accuracy question this PREREG did not ask.
  - How #17 should now be read. Arm A reproduces #17's inert shuffle on the new corpus, and the discriminators say why: cross-row cosine 0.9995 and random replacement >= 0.1 on every seed, i.e. the readout uses memory but every row's memory is nearly the same vector. The PREREG calls this rival (b). Whether RESEARCH-CONTEXT §10.3's wording changes is the manager's/owner's call.
  - The PREREG bands read ratio as "a fraction of a live memory's rate", but B/C ratios are 3-15 (the trained model is more memory-sensitive than the random-init decoy). No verdict changes (the bands are one-sided), but the decoy is not an upper bound, and the unit deserves a sentence in the next PREREG.
  - The hinge (arm C) remains an open owner decision. B and C are both live and not distinguishable at n=3 (9.325 +/- 4.683 vs 7.741 +/- 6.058). This informs the decision and does not take it.
BELIEVED, NOT VERIFIED:
  - That arm A's near-1 cross-row cosine comes from the unmasked objective (padding-dominated loss) collapsing the gestalts. Not tested. srep norms and per-step cosines were not read.
  - That B/C ratios exceed 1 because a confident trained model moves more per memory change than a near-uniform untrained one. This is a plausible mechanism, not a measurement.
  - That ln16 is the right chance reference for answer NLL (uniform over 16 answer symbols). The model's softmax is over the full vocab, so chance over the vocab is higher.
  - That torch thread count (1) does not change any reported value versus #17's default threads. The within-process controls read exactly 0.0, but cross-thread-count bit-equality was not checked.
  - That `dirty: true` in provenance is due only to the untracked uv.lock. Ledger.write() refused nothing under src/scripts/experiments, which is consistent with this, but porcelain output was not captured at launch.
NEXT (proposed, not decided):
  1. Retrieval accuracy under the masked objective. On arm B's (or C's) checkpoints, run held-out answer NLL live vs slots-zeroed vs gate-zeroed by gap bucket (S0-03's instrument), and add answer-symbol top-1 accuracy. "Live" says memory matters; this would say whether it helps. No retraining is needed, because the 9 checkpoints are on disk (untracked) under runs/decisive-shuffle/arm*/seed*/.
  2. Diagnose arm A's collinear memory: srep norm and cross-row cosine per step, from the arm A checkpoints. This is cheap, reuses cross_row_cosine, and would settle whether #17's null was a representation collapse caused by the unmasked objective.
  3. Only if 1 shows no retrieval gain: a longer-training arm B (more iters) as its own PREREG. Do not extend this run.

---

status: RETURNED
run_id: efeas-synthetic-s003
updated: 2026-09-21T11:22:12Z
provenance: a2452196508e6732e034406736d96285c7e4677d (dirty=true: untracked uv.lock only) · cpu (no model), Mac Studio macOS-26.6.2-arm64 · corpus_sha256 per seed in manifest (s0 60003e49..., s1 18b9e1e2..., s2 48a7ca7c...) · seeds_actually_run [0, 1, 2]
manifest: runs/efeas-synthetic-s003/manifest.json  (config hash f1bc3829d10cd9230b570d4d6f66b15a0989963921e1a963ada0fd906688cfb5)
falsifier: PREREG-s003.md (= PREREG.md rule): every seed H < 0.05 at M=16 -> "no headroom over FIFO on the S0-03 corpus"
expected: (manifest.expected, frozen before the run) every headroom and hit-rate row bit-identical to runs/efeas-synthetic, verdict survived, both controls hold, because answer_in_stream draws no RNG and simulate() reads only gaps/positions.
observed: exactly that. verdict.outcome = survived. All 15 ledger rows have the same keys and values as runs/efeas-synthetic/ledger.json, by_gap included.

gates:
  run:     uv run python experiments/efeas/run.py --run-id efeas-synthetic-s003 --prereg experiments/efeas/PREREG-s003.md --expect "<...>"  -> rc=0 (read directly as `cmd > log 2>&1; rc=$?`)
  audit:   uv run python scripts/render_scoreboard.py --over runs/ --audit experiments/efeas/RESULTS-s003.md -> rc=0
           "OK: every number in experiments/efeas/RESULTS-s003.md resolves to a ledger key"
  diff:    git diff --stat origin/main -- runs/  -> only runs/efeas-synthetic-s003/{ledger.json,manifest.json,raw.json}, 3 files changed, 1438 insertions(+)
  pytest:  uv run pytest -rs --tb=no -q -> rc=0; census "passed=382 failed=0 skipped=0 errors=0" (one UserWarning in tests/test_capture_bridge.py:516, not a failure)
ledger:    runs/efeas-synthetic-s003/ledger.json
numbers:
  headroom_oracle_minus_fifo (M=16) samples [0.19491525423728817, 0.17365771812080533, 0.1736227045075125], mean 0.18073189228853534, sd 0.012283164234645585. Old run: the same values.
  secondary.M32.headroom samples [0.041525423728813515, 0.03523489932885904, 0.024207011686143587], mean 0.033655778247938715, sd 0.008766531039189842. Old run: the same. Every seed is below 0.05. This row is not judged.
  secondary.M8.headroom mean 0.2720681823691921, sd 0.017934528280598227 (same as the old run)
  hit_rate.fifo mean 0.8192681077114647, sd 0.012283164234645585. hit_rate.oracle 1.0 on every seed, sd 0.0: the ledger flags sd_exactly_zero, but that is a perfect oracle, not a broken seed loop, because FIFO and random vary across the same seeds. hit_rate.random mean 0.8024734091190927, sd 0.011431442558855202.
  controls: seed0/seed1/seed2.controls_failed = [] (both controls held on every seed)
  seed{0,1,2}.n_queries = 1180, 1192, 1198
  changed SyntheticConfig default vs old run's sha 1fc8199: answer_in_stream (new field, True; the old behaviour equals False). No other field changed.
BRIEF ERRORS:
  1. The question is a theorem, not a measurement. The PREREG-s003 premise says S0-03 "may change the gap distribution" through a gap>M invariant. It did not. S0-03 added only answer_in_stream (no RNG draw) and the read-only helper fraction_of_pairs_beyond. simulate() never reads text, so the re-run could only reproduce the old numbers. It did, bit for bit. That makes it a consistency check, not new evidence about headroom.
  2. The brief's scope rules contradict its Bar. The scope rule is "run.py only if it cannot run unchanged", but the Bar ("manifest records the torch version") and PREREG-s003 ("manifest records the new defaults verbatim") need content that run.py did not write. The old manifest has neither. I changed run.py in commit a245219. The manifest now also records synthetic_config_defaults, corpus_sha256, torch_version, prereg (--prereg) and expected (--expect), and the ledger records the real argv, not a hardcoded string. The measurement is unchanged, and runs/efeas-synthetic stays reproducible from 1fc8199.
  3. "Name every field whose default differs from the old run's manifest" cannot be done as written, because the old manifest records no SyntheticConfig fields. I compared against SyntheticConfig at the old run's provenance sha 1fc8199 instead.
  4. The Bar's literal command has no way to record which prereg governs the run, and the old run.py hardcoded PREREG.md. I added --prereg and --expect. The command therefore differs from the Bar's literal one only by those two flags.
UNANSWERED BY THE BRIEF:
  - Python version. uv picked Python 3.14.6 in this worktree: no .python-version, and requires-python is ">=3.11". The old run used 3.12.13, and CLAUDE.md says 3.12. The numbers are identical anyway. Pinning the interpreter is an owner/manager decision; I did not change it.
  - uv.lock is untracked in every checkout, so every ledger records dirty=true. Should it be committed or gitignored?
BELIEVED, NOT VERIFIED:
  - The claim that facts, gaps and texts are equal apart from the appended token was checked at the command line, not in a ledger. I compared gaps with answer_in_stream True vs False for seeds 0-2: gaps equal, texts differ.
  - The prose audit backs a literal with ANY number in ANY ledger under runs/. Small integers such as 32 and 8 pass because some ledger contains them, not because this ledger does. The audit passing is weaker than "every number traces to this run".
  - Every value is bit-identical across Python 3.12 and 3.14 and between two torch installs. I did not check the old run's torch version, because its manifest does not record it.
  - Side observation, outside scope: query text reads "What does Hal-6 measures?" (the predicate is not uninflected). This is harmless to E-feas. Whether it matters for the LM objective is unexamined.
NEXT (proposed, not decided): E-feas has nothing further to say about S0-03; it is blind to token content by construction. The informative follow-up is the one S0-03 exists for: answer-token loss bucketed by gap (gap<=16 vs gap>16) for a FIFO-trained model on the new corpus, against the chance line ln 16. Separately, pin the Python interpreter (.python-version 3.12) so ledgers stop drifting to 3.14.
PR: https://github.com/Keanooo7/retrieval-successor-retention/pull/33 (open, not merged)

---

status: RETURNED
run_id: s0-03-rewardable-corpus
updated: 2026-09-21T11:17:32Z
provenance: 851103efd3d1c676b6ae844a923813bd9b1c1da9 · cpu (torch 2.14.0, py 3.14.6) · corpus = SyntheticConfig defaults (answer_in_stream=True), held-out = docs 64..127 of each seed · seeds actually run [0,1,2], steps 300/300
manifest: runs/s0-03-rewardable-corpus/manifest.json  (config hash 662c339b74acb1abaaf4d78846bb6bf1deb37c3137325dd02dbe215c5f4a78da)
falsifier: "the S0-03 answer tokens cannot be predicted without the memory" (bar 1) + "FIFO eviction raises answer loss at gap > M" (bar 2); H4 live-vs-zeroed per the brief addendum
expected: bar 1 passes; gap 1 falls below chance live via the bos-copy path; at 2<=gap<=M live at most modestly below zeroed; most likely "corpus built, retrieval not shown beyond gap 1"
observed: VERDICT inconclusive -- "retrieval not shown (not both at chance)". Bar 1 PASS, H4 FAIL, bar 2 PASS as registered but confounded. NOT a pass of the brief's bar; routes to the decisive run.

gates (re-run after merging origin/main incl. #29, at merge f5de37a):
  uv run pytest -rs --tb=no             -> passed=382 failed=0 skipped=0 errors=0 (1 xfailed) ; rc=0
     (396 -> 382 is entirely tests/test_checkpoint.py 30 -> 16 collected, #29's deterministic gate replacing the delay sweep)
  uv run ruff check .                   -> All checks passed! ; rc=0
  uv run ruff format --check .          -> 176 files already formatted ; rc=0
  uv run python scripts/mutation_battery.py --check -> 52/52 gates proven by mutation ; rc=0
     S0-03 entry: PROVEN the synthetic answer goes back out of band -> 1 on gate, 10 off (0 undeclared)
     gate node id: tests/test_synthetic.py::test_every_query_carries_its_answer_as_its_final_token
  uv run python scripts/render_scoreboard.py --over runs/ --audit experiments/s0-03-rewardable-corpus/RESULTS.md -> OK ; rc=0
ledger:    runs/s0-03-rewardable-corpus/ledger.json  (0 of 234 statistic rows sd_exactly_zero)
numbers (held-out, answer NLL, nats/answer token, mean +/- sd over seeds 0,1,2; chance ln16 = 2.7726):
  heldout.live.gap_2_to_M.answer_nll                 2.881 +/- 0.096  [2.833, 2.818, 2.992]
  heldout.slots_zeroed.gap_2_to_M.answer_nll         2.950 +/- 0.081  [2.856, 2.992, 3.000]
  heldout.slots_zeroed_minus_live.gap_2_to_M         0.068 +/- 0.092  [0.023, 0.174, 0.008]   H4 needs >= 0.10 every seed -> FAIL
  heldout.slots_zeroed.gap_ge_2.answer_nll           2.959 +/- 0.080  (bar 1 needs >= 2.673) -> PASS
  heldout.gate_zeroed.gap_ge_2.answer_nll            2.959 +/- 0.078  -> PASS
  heldout.live.gap_gt_M.answer_nll                   3.011 +/- 0.087
  heldout.live.gap_lt_M.answer_nll                   2.847 +/- 0.106
  heldout.live.gap_gt_M_minus_gap_lt_M               0.164 +/- 0.144  [0.099, 0.329, 0.064]  bar 2 PASS
  heldout.slots_zeroed.gap_gt_M_minus_gap_lt_M       0.073 +/- 0.034  <- same contrast with NO memory: bar 2 is partly not memory
  heldout.slots_zeroed_minus_live.gap_gt_M          -0.025 +/- 0.023  (live worse than zeroed once evicted, all 3 seeds)
  heldout.live.gap_eq_1 / slots_zeroed / gate_zeroed_bos_off   2.744 / 2.804 / 2.927
  heldout.live.gap_2_to_M.answer_acc 0.082 +/- 0.021 vs zeroed 0.076 +/- 0.009 (1/16 = 0.0625)
  train.slots_zeroed.gap_ge_2.answer_nll             2.636 +/- 0.104 [2.699, 2.692, 2.516]  memorisation: below chance with NO memory on trained docs
  heldout.fraction_of_pairs_gap_gt_M                 0.192 +/- 0.008 ; train 0.181 +/- 0.012
  seed*.train_final.answer_target_fraction           0.085 / 0.085 / 0.086
BRIEF ERRORS:
  1. "the answer tokens are ~6 of 193,536 targets" -- 193,536 = 64 docs x 48 x 63 (ALL targets incl. PAD, whole corpus). The same corpus has 1,180 answer targets (seed 0), 8.5% of 13,904 real targets. They do not "drown" at ~6; the mask is still needed to READ their loss, not to rescue them.
  2. The brief says the zeroed ablation with "all M slots zeroed" is the bar, but bos_replacement_mode="copy" carries the previous sentence's gestalt OUTSIDE the memory, so at gap 1 (21-24% of pairs) zeroing the memory does not remove the assert. Measured: gap-1 live 2.744 vs gate_zeroed 2.803 vs gate_zeroed_bos_off 2.927. Gap 1 was set aside in the pre-registered bars.
  3. The brief states no evaluation set. Scoring on the trained documents (as #17/decisive do with the first 8 train docs) is contaminated: 300 iters x 16 = 75 passes over 64 docs, and zeroed-memory answer NLL on train docs is 2.636 (seed 2: 2.516, 0.26 below chance) -- bar 1 would FAIL on train docs. Held-out was made primary (PREREG). The decisive-shuffle PREREG's "answer tokens only" secondary on the first 8 training docs inherits this contamination.
  4. "gap > M vs gap < M" leaves gap == M unassigned; under FIFO gap == M is still in memory. Reported both lt_M and le_M; gap==M has only 20/6/9 targets.
  5. Bar 2 as worded ("gap>M worse than gap<M under FIFO") is not a memory test on its own: the contrast is 0.073 +/- 0.034 with the memory zeroed (position/gap-1 composition confound).
  6. Files in scope omitted tests/test_train_loop.py (pins V=160 / 156 words, necessarily changes: now V=176) and scripts/mutation_battery.py (where the fixture mutation must live to be checked by --check).
  7. (Withdrawn after #29 merged: battery --check now 52/52, rc=0.) Was: SIGKILL gate timing-dependent.
  8. Baseline says 3a458ad; the worktree branch was cut at a802521 (origin/main). Scoped files identical between the two (git diff --stat empty).
UNANSWERED BY THE BRIEF:
  - "near chance" tolerance: none given; I pre-registered MARGIN=0.10 nats for all three clauses. A manager may prefer another.
  - Whether answer tokens should be up-weighted in the objective: not done (the objective is unchanged; the decisive PREREG trains train() arms by masked_loss/hinge only). answer_targets makes a weight trivially addable.
  - Which zeroing is "all M slots zeroed": reported both slots-kv-zeroed (primary) and memory_gate=0; they agree to <=0.01.
BELIEVED, NOT VERIFIED:
  - That the bar-2 residual under zeroing is a position effect (late-in-document queries), not measured.
  - That more iterations would widen live-vs-zeroed at 2<=gap<=M (seed 1 already at 0.174); not run.
  - That parallel training (5 threads/seed) does not change numbers vs sequential beyond float noise.
NEXT (proposed, not decided):
  - The decisive run as pre-registered, but its answer-token secondary should be scored on held-out docs (see brief error 3) -- a PREREG amendment decision for the manager.
  - A cheap follow-up: same condition at longer training (e.g. 1000-3000 iters) to see whether the seed-spread (0.008..0.174) resolves; plus a position-matched bar-2 contrast.
PR: see below (opened, not merged).

---

status: RETURNED
run_id: none (S0-05 is code, not an experiment. The audit is in experiments/s0-05/RESULTS.md)
updated: 2026-09-21T11:11:13Z
provenance: base 0da72e6 (the brief was written at a802521) · head 7abacdf · cpu · no dataset · no seeds
manifest: none. No training run and no runs/ ledger
falsifier: NONE in the experimental sense. The claim tested is "every checker's exit status matches docs/gates.md's five codes", and the battery settles it
expected: canary baseline 3->2; seven stubs 1->3; battery stale anchor and red baseline 1->3; a new baseline->3 mutation reddens only the corrected test
observed: all as expected, plus 3 sites the brief missed (extract_golden_tensors ImportError, argparse's 2, empty battery). PR #32 is open and not merged

gates:
  uv run pytest -rsx -> passed=393 failed=0 skipped=0 errors=0, 1 xfailed (pre-existing, strict), exit 0. Baseline at 0da72e6: passed=373, 1 xfailed, exit 0
  uv run python scripts/mutation_battery.py --check -> "62/62 gates proven by mutation", exit 0 (was 51/51)
  ruff check . / ruff format --check . -> exit 0 / 0
  the SIGKILL row this brief warned about was PROVEN on this run (2 on gate)
ledger: none (no experiment). The counts above are literal command output, with each exit status captured as 'cmd > log 2>&1; rc=$?'
numbers: +20 tests = 19 in tests/test_exit_codes.py + 1 new parametrised case in test_no_hardcoded_constants[exit_codes.py], from a diff of node ids. 11 new mutations, all with 0 off-gate failures. The stub-class mutation reddens 7/7 cases
BRIEF ERRORS:
  1. src/rsr/cli.py is listed as "Correct -- no nothing-to-compare state exists on this path". That is false. 'uv run rsr constants beta' raises UnmeasuredConstant and exits rc=1 (measured). "No logged value for this key" is exactly gates.md's definition of 2. cli.py is outside the files in scope, so it is unchanged and named in NOT_CONVERTED.
  2. "Every exit-code site in the repo" is false of trunk too. The brief's own grep also returns: experiments/s0-02/measure_qtok_collapse.py:131,139,141 (precondition refusals via SystemExit(msg) exit 1, which should be 3); s0-02/write_ledger.py (8 typed exit_code=0 rows, the same class as canary:151); cycle-01:271 and shuffle-control:152 (child failure via SystemExit(msg)); efeas/run.py:198 (a literal exit_code=0, but disclosed as SELF-REPORTED, so correct).
  3. extract_golden_tensors: the brief says to read main(). The actual defect is at module level: 'import jax' raises ImportError and exits 1 in the project venv (measured). The three ABORT returns in main() are correct 1s.
  4. The anchor table says "canary.py:196 (verdict computed)". At a802521 the verdict is computed at :175 and :178, and the exit code at :202. :196 is 'return {'.
  5. The battery anchors are +1 at 0da72e6 (:915/:939/:1003), not :914/:938/:1002. Brief 0b landed between them. This is the same drift class the brief itself warns about.
  6. Bar 5, "one mutation covering the class": the battery mutates one file per mutation, so this is only possible if the seven stubs share a helper. I built not_implemented() for that. The brief did not say so.
  7. (Brief's own item, carried over as asked) the protocol was read off the suspect file, canary.py.
UNANSWERED BY THE BRIEF:
  - Whether cli.py and experiments/s0-02 should be converted. They are listed in NOT_CONVERTED with reasons, and the test refuses a stale exemption.
  - No exit code exists for a crash. I kept a crash as 1 in canary (train raising) and in cycle-01/shuffle-control child failures, and chose it explicitly. A reviewer may prefer 3.
  - docs/lab-notes/dispatch-2026-09-21-overnight.md:413 tells the overnight run that canary exits "3 first reading". That is stale once this merges. It is a manager document, so I did not edit it.
BELIEVED, NOT VERIFIED:
  - mutation_battery.run_suite treats pytest exit 5 (no tests collected) as clean. I believe conftest's floor makes this unreachable on a whole-suite run. Not tested, not changed.
  - src/rsr/train/loop.py main: only 'return 0' was read (the brief forbids touching the file). Its behaviour with MPS absent was not probed.
  - canary.py was not run for real (MPS training, and it writes runs/canary/). The ordering is tested with a fake train().
  - uv resolves Python 3.14.6 in this venv, but CLAUDE.md says 3.12. Recorded, not investigated.
NEXT (proposed, not decided): a small follow-up converting src/rsr/cli.py (UnmeasuredConstant -> 2, unknown name -> 3) and experiments/s0-02/measure_qtok_collapse.py (-> 3), then removing both from NOT_CONVERTED. After that, port src/rsr/gates/ onto this enum, which is the only producer of 4.

status: RETURNED
run_id: b0b-sigkill (Brief 0b, 2026-09-21: test determinism. No training run, no runs/ ledger written)
updated: 2026-09-21T10:42:46Z
provenance: base 6e31a40 (brief written at a802521) · head 34ed0ac · cpu (pytest/battery) · no dataset · no seeds (no training)
manifest: none. No experiment ran; the falsifier is about a test and the battery settles it
falsifier: "The SIGKILL test detects a non-atomic checkpoint write every time." Refuted by any run where `tmp = path` leaves the test green.
expected: mutated 20/20 red, unmutated 20/20 green, battery --check rc 0 twice
observed: all as expected. Falsifier NOT fired: mutated 0/20 green.

change:
  src/rsr/train/checkpoint.py: `_write_hook(phase)` test seam, default `_no_op`, called at
    "mid_write" (half the buffer written and flushed), "before_replace" and "after_replace".
    The single fh.write(data) is now two writes of one memoryview (same bytes).
  tests/test_checkpoint.py: the child parks in the hook at the chosen phase and writes
    "AT <phase>" to its stdout pipe. The parent blocks on readline, then SIGKILLs it.
    No wall-clock delay. Parametrised over the 3 phases, replacing the 18-point kill-delay sweep.
    mid_write/before_replace: old checkpoint must be byte-identical. after_replace: must be the complete new one.
    New test_the_write_hook_is_inert_in_production: hook is _no_op by default; atomic_write and
    save(capture_rng=False) output == one torch.save of the same payload, byte for byte.
  scripts/mutation_battery.py: that row's `why` rewritten (was "TIMING-DEPENDENT").

gates:
  20x isolated, UNMUTATED  (pytest tests/test_checkpoint.py::test_a_sigkill_mid_save_never_leaves_a_corrupt_checkpoint):
    passed_runs=20 failed_runs=0 of 20;   20 x "passed=3 failed=0 skipped=0 errors=0"
  20x isolated, MUTATED by hand (`tmp = path`, then git checkout):
    passed_runs=0 failed_runs=20 of 20;   20 x "passed=1 failed=2 skipped=0 errors=0"
    20 FAILED ...[before_replace]  "SIGKILL at 'before_replace' replaced the old checkpoint (step=1) before the atomic rename"
    20 FAILED ...[mid_write]       "left an unloadable checkpoint (RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central directory ...)"
    [after_replace] stays green under the mutation. That is correct: a complete file is in place by then.
  $ uv run python scripts/mutation_battery.py --check     # run 1
    PROVEN        checkpoints written straight to the final path -> 2 on gate, 0 off (0 undeclared)
    51/51 gates proven by mutation            rc=0
  $ uv run python scripts/mutation_battery.py --check     # run 2
    PROVEN        checkpoints written straight to the final path -> 2 on gate, 0 off (0 undeclared)
    51/51 gates proven by mutation            rc=0
  $ uv run pytest -rs --tb=no
    passed=373 failed=0 skipped=0 errors=0    rc=0   (pytest line: 373 passed, 1 xfailed, 1 warning)
    373 = 387 - 15 (sweep went from 18 params to 3) + 1 (new test)
  $ uv run ruff check           rc=0  All checks passed!
  $ uv run ruff format --check  rc=0  173 files already formatted
ledger: none (no run)
numbers: every figure above is literal command output. None comes from a ledger because no run happened.

BRIEF ERRORS:
  1. The suggested seam, "a hook called between the write and the atomic rename", cannot catch the
     battery's mutation on its own. With `tmp = path` the whole payload is written and fsynced to the
     final path before that point, so a kill there leaves a COMPLETE new file. The old test's assertion
     accepts "complete new", so it would stay green 20/20. The kill has to land MID-write (partial
     bytes on disk). I split the write into two writes of one buffer and put the hook between them.
     The before-rename hook is kept as well, with a stricter "old bytes must survive" assertion.
  2. "N/N" for the battery: at 3a458ad the battery showed 48/49. This base shows 51 gates (51/51).
     The brief does not give an expected N; I am flagging it so nobody compares 51 to 49.
  3. Files in scope omit docs/mutation-battery.md, the generated record (`--markdown`). The `why` for
     this row changed, so that record is now stale for this row until it is regenerated. I did not
     touch it because it is out of scope.
  4. "Run the test alone 20 times" is ambiguous for a parametrised test. I ran the whole parametrised
     node 20 times and gave per-parametrisation counts.
UNANSWERED BY THE BRIEF:
  - Whether to keep a "kill after the rename" case. The old sweep had one (30/60 ms). I kept it as the
    deterministic [after_replace] phase.
  - The census line does not count the 1 xfailed test (tests/test_train_loop.py). It is not mine; it is
    reported here so it is not silently dropped.
BELIEVED, NOT VERIFIED:
  - That no other production caller assigns `_write_hook` (git grep shows only the test child).
  - torch.save byte-determinism beyond this torch version. The inert-hook test depends on it and passes here.
  - That fh.flush() before the mid_write hook puts the partial bytes into the file visible to another
    process. The 20/20 torn-file reading under the mutation is evidence for it on APFS, not proof on other filesystems.
NEXT (proposed, not decided): regenerate docs/mutation-battery.md (`--check --markdown docs/mutation-battery.md`)
  in whichever brief owns it, so the committed record matches the row's new `why`.

---

status: RETURNED
run_id: b0-code (Brief 0, 2026-09-21 -- code fixes; no training run, no runs/ ledger written)
updated: 2026-09-21T10:09:17Z
provenance: base 3a458ad (brief says 402d328; the only diff 402d328..3a458ad is the brief itself) · cpu (pytest/battery) · no dataset · no seeds (no training)
manifest: none -- no experiment ran; the brief's falsifiers are about tests, settled by the battery and by tmp_path trials
falsifier: A/B "the shuffle-control tests catch a replay that perturbs its memory, and one that swaps the bos-copy path" -- refuted if either mutation leaves every test_shuffle_control.py node green. C "a historical-fact test can pin history without pinning the live tree" -- refuted if the rewritten test reddens on an added canary ledger.
expected: (from the brief, checked not copied) +1e-3 reddens test_replaying_each_rows_own_memory_reads_exactly_zero; bos swap reddens test_disabled_memory_reads_exactly_zero.
observed: both premises CONFIRMED, each with exactly one on-gate node and 0 off-gate. Neither falsifier fired. Task C: extra ledger green, cycle-04 flip red.

gates:
  $ uv run pytest -rs --tb=no            # BEFORE, at 3a458ad
    passed=385 failed=0 skipped=0 errors=0        rc=0
  $ uv run pytest -rs --tb=no            # AFTER, at b61f2a2 (3 new tests)
    passed=387 failed=0 skipped=0 errors=0        rc=0
  $ uv run python scripts/mutation_battery.py --check      # BEFORE, at 3a458ad
    48/49 gates proven by mutation
    UNPROVEN: test_a_sigkill_mid_save_never_leaves_a_corrupt_checkpoint (checkpoints written straight to the final path) -- reddens nothing
    rc=1
  $ uv run python scripts/mutation_battery.py --check --markdown docs/mutation-battery.md   # AFTER
    PROVEN the shuffle replay perturbs the memory it replays -> 1 on gate, 0 off (0 undeclared)
    PROVEN the shuffle replay hands over the bos gestalt too -> 1 on gate, 0 off (0 undeclared)
    PROVEN checkpoints written straight to the final path    -> 1 on gate, 0 off (0 undeclared)
    51/51 gates proven by mutation
    rc=0
  $ uv run ruff check src/ tests/ scripts/          All checks passed!   rc=0
  $ uv run ruff format --check src/ tests/ scripts/ 72 files already formatted  rc=0
  $ git diff --stat origin/main...HEAD -- runs/ preregistration/     (empty) rc=0

ledger: none (no run). Evidence: battery JSON (scratch), the node ids below.
numbers:
  N (measured on baseline) = 49 entries; baseline --check = 48/49 exit 1, NOT 49/49.
  after = 51/51 exit 0  -> N+2/N+2 literally, but see the checkpoint caveat.
  A1 "perturbs the memory": FAILED tests/test_shuffle_control.py::test_replaying_each_rows_own_memory_reads_exactly_zero
     (AssertionError: 3.5040586897849835e-05 at :110); off-gate 0.
  A2 "bos gestalt too": FAILED tests/test_shuffle_control.py::test_disabled_memory_reads_exactly_zero
     (AssertionError: 0.005718388603728641 at :99); off-gate 0.
  C, tmp_path copy of runs/ + canary/cycle-99:
     old assertion -> RED: len(canaries)=4 ; new assertion -> GREEN
     cycle-04 outcome -> "survived": new assertion -> RED: {'canary/cycle-04': 'survived', 'canary/cycle-08': 'survived', 'canary/cycle-12': 'survived'}
     nodes: tests/test_evidence_machinery.py::test_the_scoreboard_reproduces_the_true_canary_tally PASSED
            tests/test_evidence_machinery.py::test_a_later_canary_reading_does_not_redden_the_09_18_tally PASSED
            tests/test_evidence_machinery.py::test_the_09_18_tally_reddens_when_cycle_04_changes_its_outcome PASSED
  B: raise re-derived: `grep -n` -> def observe :442, raise NotImplementedError :448 at 3a458ad;
     at 18557e7 (the sha the entry's own `how` names) def observe :427, raise :433. So ":433" was TRUE for the
     measurement it describes. Producer now writes "NotImplementedError from RSRPolicy.observe
     (src/rsr/retention/rsr.py:433 at 18557e7)" -- symbol + line@sha, which cannot go stale.

BRIEF ERRORS:
  1. "Both existing test_shuffle_control.py entries use gate test_shuffle_control.py::" -- false. "the shuffle control never
     applies its permutation" is gated on `test_live_memory_moves_the_loss`. (New entries use the brief's gate as instructed.)
  2. Task C premise "this test reddened (passed=384 failed=1)" does not hold on this branch: at 3a458ad runs/canary has only
     cycle-04/08/12 (e26900a kept the new reading out of runs/), and the baseline is passed=385 failed=0. The fix is still
     correct and needed; the red it describes is not reproducible here.
  3. Bar "final line reads N+2/N+2" assumes the baseline is N/N. It was 48/49, exit 1: the checkpoint mutation
     (SIGKILL test) is timing-dependent. Baseline run: ADDS NOTHING. Five isolated probes (-k that test, mutation applied):
     all 5 green, i.e. mutation NOT caught. Post-edit battery run: caught (PROVEN). So "51/51" is one lucky run on that row,
     not a stable property; my two entries are stable.
  4. Task B framing "rsr.py:433" as an error: it was correct at the measurement sha 18557e7; only the live line moved.
     Fixed the producer anyway (by pinning to the sha), since a bare line number reads as live.
  5. Baseline sha in the brief (402d328) is not the branch base (3a458ad); harmless, diff is the brief file only.
UNANSWERED BY THE BRIEF:
  - Whether docs/mutation-battery.md should be committed from a run whose checkpoint row is a timing flake. I committed it
    (it is script output and the brief allows it) and said so in the commit message; manager may prefer to drop that commit.
  - Should the SIGKILL checkpoint entry be excluded from --check, or its test made deterministic? Out of scope; not touched.
BELIEVED, NOT VERIFIED:
  - That the checkpoint flake rate on the Studio is roughly "rarely caught": 1 catch in 7 observations (1/2 battery runs,
    0/5 isolated probes). Small sample.
  - That the regenerated s0-02 ledger would differ only in that string; I did not re-run write_ledger.py (runs/ is off-limits).
NEXT (proposed, not decided):
  - A brief to make test_a_sigkill_mid_save_never_leaves_a_corrupt_checkpoint deterministic on the M4 Max (e.g. inject the
    kill between write and rename rather than by wall-clock delay), so --check stops being a coin toss.

---

status: RETURNED
run_id: s0-01-loop-defects
updated: 2026-09-20T17:05:00Z
provenance: 201e2d6 (defects measured at c39b498; HEAD moved to 3973d1a mid-cycle — a concurrent researcher landed B2 on this branch, and `git diff --name-only c39b498 3973d1a -- src/rsr/train/loop.py src/rsr/retention/rsr.py` is empty, so nothing in scope moved) · cpu · dataset 849f94007cc533f14b3ece035fe0941167231bcf65f906e00445a9a04b8dab27 · seeds actually run [0,1,2,3,4]
manifest: runs/s0-01-loop-defects/manifest.json  (config hash cfacc035d54946c4891b1b5d445ddf315d6e01213ba3dba7951869cdbe4f9775; pre-registration committed alone at 5133135, ordered ahead of the fix, where the same content hashed to d98be1a2… before `Ledger.manifest()` re-serialised it with sorted keys — verified semantically identical, `old == new` is True and no key differs)
falsifier: the brief's own Bar. If reverting a fix reddens NOTHING the test is vacuous and that is the finding; if it reddens tests other than its own the defects were not the separable things the brief claims. Secondary, for (c): if TGConfig's hinge weight were already 0, "absent from the objective" would be inert rather than a defect.
expected: all three predicted defects real at this sha; corpus 156 unique words; `from_registry` raising even through a full override; nothing constructing a value head, so the muP item stays a recorded xfail.
observed: all three real, all three fixed, all six mutations PROVEN with ZERO off-gate. Corpus 156 → V=160, exactly as predicted. `from_registry` raises on `nu` even when `nu` is supplied. 🔴 The falsifier's first clause FIRED, on my own work: `test_the_hinge_is_on_by_default` survived its mutation and was vacuous.

gates:
  $ pytest -rs                          # BEFORE any edit, at c39b498
    passed=343 failed=0 skipped=0 errors=0        exit 0
  $ pytest -rs                          # after, at 201e2d6
    passed=358 failed=0 skipped=0 errors=0
    358 passed, 1 xfailed, 1 warning in 10.51s    exit 0
  $ ruff check src/ tests/ scripts/
    All checks passed!                            exit 0
  $ python scripts/mutation_battery.py --check --markdown docs/mutation-battery.md
    45/45 gates proven by mutation                exit 0
  $ grep -c srep_norm src/rsr/train/loop.py       # 0 before, 9 after; COUNT read, not $? (=1)
ledger:  runs/s0-01-loop-defects/ledger.json   (status ok · verdict survived, not capped · 4 commands, all reproducible)

numbers: every figure below is a ledger key in runs/s0-01-loop-defects/ledger.json.
  loop_py_lines 401 · grep_c_srep_norm_prefix 0 · corpus_unique_words 156 · derived_V 160
  cli_vocab_default_prefix 50257 · vocab_over_allocation_factor 314.1
  tests_passed_before 343 → tests_passed_after 358 (xfailed 1, skipped 0)
  hinge_contribution_at_default_weight 6.235937e-4 vs float32_accumulation_noise 2.086e-7
  loss_lm_final         4.926144 ± 2.670e-02  (n=5, sd_exactly_zero False)
  loss_final            4.926171 ± 2.671e-02  (n=5, sd_exactly_zero False)
  loss_srep_hinge_final 0.002719 ± 1.648e-03  (n=5, sd_exactly_zero False)

mutations — each reverts exactly one fix; node ids collected from a full suite run, tree restored:
  policy built unconditionally again   → test_train_does_not_stamp_a_policy_it_did_not_build      1 on, 0 off
  --policy stops reaching train()      → test_the_policy_is_selectable_from_the_command_line      1 on, 0 off
  srep-norm hinge back out             → test_the_hinge_{reaches_the_objective,has_a_documented_off_switch,is_on_by_default}  3 on, 0 off
  ppl computed from the penalised loss → test_perplexity_is_a_perplexity_and_not_a_penalised_loss 1 on, 0 off
  --vocab default back to 50257        → test_the_cli_vocab_default_reaches_the_derived_path      1 on, 0 off
  from_registry reads eagerly again    → test_from_registry_{honours_an_override…,still_raises…}  2 on, 0 off

BRIEF ERRORS:
  1. 🔴 A FOURTH wrong line number, and it is inside the amendment that says every
     citation was re-measured. The defect table gives (e)'s derived path as `:144`.
     At 058e712 — the sha the amendment names — and at c39b498, `V = vocab if vocab
     else 4 + len(build_vocab(probe))` is at **:193**. Line 144 is a blank line inside
     `lm_loss`'s docstring at both shas. The brief warns "three briefs in a row have
     carried a wrong line number"; this is the fourth, and it survived a pass whose
     entire purpose was to catch it. Every OTHER citation I re-ran was exact,
     including `model.py:424` and `rsr.py:236-243`.
  2. 🔴 Defect (b)'s "four-line test" is not runnable as written. It says: "construct
     with policy_name='rsr', assert the constructed policy is not a FIFOPolicy."
     On an empty ledger nothing is constructed — `from_registry` raises
     `UnmeasuredConstant`, so the assertion is never reached and the test errors
     rather than passing. The claim that IS testable, and what I wrote, is the
     disjunction: asking for "rsr" yields a non-FIFO policy **or** a loud refusal
     naming E1, and never a FIFOPolicy.
  3. 🔴 "Done when" contradicts "Less certain". "Done when" requires "0 failures and
     0 new skips"; "Less certain" instructs "leave a failing test rather than a
     speculative fix". Those cannot both hold. Resolved with `xfail(strict=True)`,
     which `conftest.py` counts as neither passed nor skipped — so the record exists,
     the suite stays green, and `strict=True` means it cannot go green unnoticed. The
     brief should say which of the two it means.
  4. "Done when … a PR is merged to main" is not a thing this lane can do. The
     researcher role forbids choosing what happens next, this branch carries a second
     researcher's commits, and the merge is the manager's call. NOT DONE, and not
     reported as done.
  5. Minor, and the manager already half-corrected it: (e)'s "unreachable from the
     CLI" is the larger claim. I fixed and pinned the smaller true one — unreachable
     **at the default**. `--vocab 0` always reached the derived path and still does.
  6. Standing-file error, not the brief's: `.claude/agents/rsr-researcher.md`'s schema
     note says `manifest.json`, `config_hash`, `seeds_actually_run`, `steps_done` and
     `status` are "not yet written by scripts/ledger.py" and that reconciling them is
     cycle 0's job. **Cycle 0 landed it.** `Ledger.__init__` writes all five and
     `Ledger.manifest()` exists. The warning now instructs a researcher not to expect
     fields the tree does produce. Same file says there is "no `.orchestrator/` in
     this repo" while also telling me to report into `.orchestrator/outbox/` — the
     directory exists, untracked.

  Confirmed correct, re-run rather than read: 401 lines · tests/test_train_loss.py:31
  imports it · FIFOPolicy() unconditional at :214 · policy_name param :184, frozen
  config :224, run_id :228 · no --policy in the flag table · grep -c srep_norm = 0 ·
  --vocab at :369 · policy.attribution() at :315 · build_param_groups(model, None, …)
  at :210 · rsr.py:236-243 is the eager kw build, exactly as the manager revalidated ·
  model.py:424 is the penalty assignment · 156 unique words.

THE VACUITY FINDING — my own test, caught by the rule that exists for it:
  `test_the_hinge_is_on_by_default` asserted `loss > loss_lm`. It SURVIVES the hinge
  mutation. With `loss = lm` the two still differ by ~2.1e-7, because `loss` is a
  float32 tensor accumulated across 48 sentence steps inside `run_policy_loop` while
  `loss_lm` is a Python-float sum of the same 48 terms. A strict `>` against
  accumulation noise is not an assertion about the hinge. Rewritten with a **measured**
  margin: the hinge contributes 6.236e-4 at the default weight, three thousand times
  the noise, so 1e-4 sits between them with room on both sides. The mutation now
  reddens 3 of 3. Reported rather than quietly repaired — S0-02 had the same thing
  happen to Bar 3, and it is now twice in two cycles.

A DEFECT THE FIX FOR (c) WOULD HAVE INTRODUCED, had I not pinned it:
  `ppl` was `float(torch.exp(loss.detach() / steps_per_stream))`. That is a perplexity
  only while `loss` IS the LM loss. The moment a regulariser enters the objective the
  field keeps its name and stops being the thing the name says — a new silent defect
  shipped inside the fix for an old one. It now reads `loss_lm`, is identical to the
  old expression at weight 0, and has its own mutation. `loss_real_tokens`, cycle 1's
  common yardstick, is untouched and remains hinge-free.

⚠️ CONSEQUENCE FOR EXISTING LEDGERS — the manager should decide, not me:
  (c) changes the training objective, so `loss` and `ppl` from runs before 3b02ee2 are
  not comparable with runs after it. `loss_real_tokens` is unaffected and remains the
  cross-arm yardstick. `srep_norm_reg_weight` is now in `frozen`, so every config hash
  and every run_id changes — loudly, which is the point: two different objectives
  cannot share a hash. Cycle 1's `experiments/cycle-01-masked-loss/` results were
  produced under the old objective.

UNANSWERED BY THE BRIEF:
  1. What `steps_per_epoch` should be when building an RSR policy in this loop. It
     derives `T_warm`. I pass `float(iters)` — a choice, not a measurement, documented
     in `build_policy`'s docstring rather than hidden. It is inert today (the registry
     refuses first) and stops being inert the day E1 logs.
  2. Whether the hinge should default ON. I read `docs/spec-corrections.md` correction
     15 item 4 — "the hinge penalty is in TG's loss" — plus §3.1's "the base model be
     unmodified", and defaulted it on. If the intent was to keep the loop's objective
     frozen against cycle 1's runs, that is the owner's call and flipping the default
     is a one-line change with a test either way.
  3. Nothing in the brief says which scope `build_policy` should read the registry in.
     I hardcoded `"synthetic"` as a keyword-default, matching what this loop's corpus
     is. `M` and `S` have no global value (CLAUDE.md), so this will need revisiting
     for `corpora` and `e7`.
  4. The brief never says whether `masked_loss` should also get a CLI flag. It is a
     `train()` parameter with no `--masked-loss`, which is structurally the same defect
     as (b) — the unmasked control arm is unreachable from the command line. It sits
     against defect (a), which cycle 1 owns, so I did not touch it. Flagging, not fixing.

BELIEVED, NOT VERIFIED:
  1. That adding the hinge does not break `test_reduction.py` or `test_fidelity.py`.
     Both currently pass in the suite, but neither exercises `train()` — they are not
     evidence about the objective. The fidelity fixtures are the real check and I did
     not regenerate them.
  2. That `build_policy("rsr", …)` returns a working `RSRPolicy` once E1 logs. I could
     not execute that path: the registry refuses, correctly, and I did not fabricate a
     populated registry to find out. The construction arguments are believed right by
     reading `RSRPolicy.__init__`, not by running it.
  3. That `value_head=None` into `build_param_groups` is genuinely a latent defect
     rather than a non-issue. I verified only that nothing constructs a head today.
     The strict xfail records the belief; it does not test the μP consequence.
  4. That my 5-seed spread generalises. It is 3 iterations at d=32 on CPU — enough to
     show the seeds reach the RNG and the sd is not 0.0000, and nothing more.
  5. That regenerating `docs/mutation-battery.md` did not collide with the concurrent
     researcher. HEAD moved under me once during this cycle; I re-verified my files
     were untouched, pushed cleanly, and `git status` is clean but for a pre-existing
     untracked `uv.lock` that is not mine.

NEXT (proposed, not decided):
  1. **Defect (d) is now unblocked.** The brief says it is downstream of (b) and cycle
     4 owns it. (b) is closed: `build_policy` exists and a non-FIFO policy is now
     constructible in principle, so `policy.attribution()` vs `attribution_counts()`
     at `:315` can be fixed against a real policy rather than against `None`.
  2. **Decide the ledger-comparability question above** before any further training
     run, so nobody joins a pre-3b02ee2 `loss` against a post one.
  3. **Fix the brief's `:144` citation and the "Done when"/"Less certain"
     contradiction** in the file itself, since briefs here outlive their cycle.
  4. `--masked-loss` as a flag, if the owner wants the unmasked control arm drivable
     from the CLI — smallest possible follow-up, same shape as (b)'s CLI half.
status: RETURNED
run_id: s0-02-capture-bridge
updated: 2026-09-20T00:00:00Z
provenance: f85be35 (measurements taken at c81ac35) · mps · Apple M4 Max 64 GB · torch 2.14.0 · random-token synthetic batch, seeds 0,1,2
manifest: runs/s0-02-capture-bridge/manifest.json  (config hash 56baf95a8a4048585733f6496c771e4d2e489fecac6532afbfb36323e8ae8e80)
falsifier: F-capture-free — "capture is off by default and free when off." Secondary: bar item 3, "the capture actually contains `W_O`."
expected: capture_off near E0c's FIFO 374±8 (not the RSR 310); capture_on visibly slower; capture_off unchanged by the bridge's presence; and `r_i` cheaper than the bridge itself.
observed: capture_off 388.80 ± 1.13 sent/s · capture_on 360.46 ± 0.40 (−28.34, −7.29%) · capture_on+r_i 329.96 ± 0.94 (−58.83, −15.13%). Free-when-off survives: the unmodified e0c harness gives 369 ± 11 pre-bridge and 370 ± 9 with it. 🔴 The fourth expectation was FALSIFIED — `r_i` costs MORE than the bridge (−30.5 vs −28.3), because it is 1264 Python-level calls launching small MPS kernels, not a FLOP bill.

---

# S0-02 — the AttentionTrace capture bridge

## gates

```
$ .venv/bin/ruff check .
All checks passed!                                          exit 0

$ .venv/bin/ruff format --check .
148 files already formatted                                 exit 0

$ .venv/bin/pytest -q --no-header
passed=343 failed=0 skipped=0 errors=0                      exit 0
   (20 of those are new: tests/test_capture_bridge.py. Baseline before this
    cycle was passed=323 failed=0 skipped=0.)

$ .venv/bin/python scripts/mutation_battery.py --check
39/39 gates proven by mutation                              exit 0
```

🔴 **Exit codes read from `$?` directly, never after a pipe.** The one place I did
pipe to `tail` and then read `${PIPESTATUS[0]}`, it came back **empty** — this
shell is zsh, where the array is `$pipestatus[1]`. I noticed because the value was
blank rather than `0`; had the idiom silently returned the pipe's status it would
have read `0` and meant nothing. All gate exit codes above were re-taken
unpiped.

## ledger

`runs/s0-02-capture-bridge/ledger.json` — status `ok`, verdict **survived**,
uncapped (all 7 commands resolve to entry points committed at the recorded sha).
Raw artefacts beside it: `capture_cost.json`, `e0c_fifo_with_bridge.json`,
`mutations.json`. Writeup: `experiments/s0-02/RESULTS.md`. Producer:
`experiments/s0-02/write_ledger.py`, committed so the ledger is regenerable.

## numbers

Every figure traces to a ledger key.

| ledger key | value |
|---|---|
| `sent_per_s.capture_off` | 390.08, 387.95, 388.36 → **388.80 ± 1.13** (n=3) |
| `sent_per_s.capture_on` | 360.89, 360.11, 360.38 → **360.46 ± 0.40** (n=3) |
| `sent_per_s.capture_on_ri` | 330.75, 330.21, 328.93 → **329.96 ± 0.94** (n=3) |
| `delta_sent_per_s.capture_on_minus_off` | **−28.34** |
| `delta_pct.capture_on_minus_off` | **−7.29 %** |
| `delta_sent_per_s.capture_on_ri_minus_off` | **−58.83** |
| `delta_pct.capture_on_ri_minus_off` | **−15.13 %** |
| `peak_gb.capture_off / on / on_ri` | 28.75 / 28.79 / 28.80 |
| `observe_calls.capture_off / on / on_ri` | 0 / 1280 / 1280 |
| `r_i_computed.capture_on_ri` | 1264 ( = 1280 − 16; each row's first step has `n_live == 0`) |
| `e0c_fifo_sent_per_s.pre_bridge` | 369 ± 11 (n=3) |
| `e0c_fifo_sent_per_s.with_bridge` | 370 ± 9 (n=3) |
| `mutation.W_O_dropped_from_the_capture_path` | PROVEN — 4 on-gate, **0 off-gate** |
| `mutation.observe()_is_never_reached` | PROVEN — 2 on-gate, **0 off-gate** |
| `mutations.total_proven` | 39 / 39 |

**Config:** `d=128, S=80, batch=16, M=40, V=50257`, 64 tokens/sentence, model in
**eval** mode (correction 20 / D-F), forward + backward, **no optimizer step**,
three arms interleaved round-robin within each repeat, one discarded warm-up per
arm.

### Spread, stated properly

`peak_gb.capture_on` and `peak_gb.capture_on_ri` carry **sd exactly 0.0**, and the
ledger flags them (`sd_exactly_zero`). That is *not* the broken-seed shape: peak GB
is rounded to two decimals over a deterministic allocation, and the **throughput
rows from the same three trials** carry sd 1.13 / 0.40 / 0.94, which is what
proves the repeats genuinely differed. Reported rather than dropped, because a
flagged zero a reader can dismiss is worth more than a zero nobody mentions.

### Which mutation turns only the new tests red

**`W_O dropped from the capture path`** — replace
`wo_vs.append(torch.einsum("bmhk,hkd->bhmd", v, wo))` with
`wo_vs.append(v.permute(0, 2, 1, 3))` in `src/rsr/model/tg/policy_loop.py`.
Four tests red, **all four in the new file**, nothing else in the 343-test suite:

```
tests/test_capture_bridge.py::test_the_trace_has_the_shapes_the_reward_module_declares
tests/test_capture_bridge.py::test_W_O_is_load_bearing_in_the_capture
tests/test_capture_bridge.py::test_wo_v_is_the_projected_value_not_the_raw_value
tests/test_capture_bridge.py::test_the_collapse_reproduces_the_real_increment
```

It needs a mutation rather than a code review because **dropping `W_O` is
shape-compatible**: `reward.contribution` norms over the last axis, and
`[L, H, M, Dh]` norms just as happily as `[L, H, M, D]`. Nothing raises; `r_i`
quietly becomes norm-weighted raw attention, which is v0.1's rejected definition
wearing the new one's name.

🔴 **Bar item 3's own test was vacuous first, in the way bar item 3 warns about.**
The original `test_W_O_is_load_bearing_in_the_capture` perturbed `attn_out_proj`
on *every* cross-attention block and asserted `r_i` moved. Run against the
mutation it exists for, it **stayed green**: layer `l`'s output enters the residual
stream and moves layer `l+1`'s *attention*, so `r_i` changes whether or not `W_O`
is in the capture. The brief says *"if `r_i` is unchanged, the capture is wrong
and the test is vacuous"*; here `r_i` changed, for the wrong reason, which is the
same vacuity from the other side. Caught only because the battery names which
tests a mutation reddens — three *other* tests were silently carrying the gate.
Now the perturbation is confined to the **last** cross-attention block (which is
the last block, so nothing downstream attends to memory) and the test asserts
`alpha` is bit-identical across the intervention, leaving `wo_v` as the only route.

Second mutation, `observe() is never reached` (`if observe:` → `if False:`):
PROVEN, 2 on-gate, 0 off-gate. That is bar item 4.

**Three pre-existing `reward.py` mutations now also redden one new test each**, so
they are **declared with a reason** rather than tolerated (clause 2). The reason is
this cycle's whole point: `test_reward.py` asserts a property on a hand-built
`AttentionTrace`; `test_capture_bridge.py` asserts the same property end to end on
a real forward pass. Those were two disconnected claims until the bridge existed.
A `reward.py` mutation that reddened *only* the fixture test would now mean the
bridge does not reach the reward.

## ADR-0008 — the decision the brief required

`docs/decisions/ADR-0008-qtok-collapse.md`. **`Q_tok` collapses by summing `α` over
query positions where `mask != 0`.**

- **The sum is algebra, not a choice.** `W_O v_i` does not depend on `q`, so slot
  `i`'s total cross-attention increment over the sentence is exactly
  `(Σ_q α_q) · W_O v_i`. Checked, not asserted: ≤ **1.49 × 10⁻⁷** on values up to
  8.23 × 10⁻¹, all six layers (`test_the_collapse_reproduces_the_real_increment`).
- **Does the spec imply an answer? No**, and the ADR says so explicitly. §3.2.1
  writes `α_{l,h,i}` with three indices and never introduces a token index, so the
  notation presupposes the collapse. The nearest guidance is §3.2.1's own
  "report the per-layer profile before collapsing" — an argument that aggregating
  over a heterogeneous axis is a claim. It motivates the ADR; it does not pick a
  value.
- **PAD query positions excluded** — only PAD *keys* are masked, so each pad
  position still emits a full distribution over slots. **`attn_out_proj.bias`
  excluded** — one vector per position, shared by every head and slot.
- **Alternatives and what distinguishes them:** mean is *the same decision* (see
  BRIEF ERRORS 1); EOS-only is the real fork (max Δ`r_i` **0.0383**) and is left
  unimplemented and scheduled as an E0d row rather than as a dead branch.
  `cross_capture(collapse=...)` raises on anything else.

## BRIEF ERRORS

1. 🔴 **"How does `Q_tok` collapse to `[L,H,M]`? … It changes `r_i`, and therefore
   it changes E0d's answer" — half false, and the false half is the one the brief
   leans on.** Of the three candidates the brief names (mean over real tokens, EOS
   only, a sum), **mean and sum cannot change `r_i` at all.** `mean = sum/Q_real`
   with a single `Q_real` shared by every `(l,h,i)`, and §3.2.1's rescale is
   `share_i = raw_i / Σ_j raw_j`, so it cancels exactly. Measured on a real forward
   pass: max |Δ`r_i`| = **2.98 × 10⁻⁸**, while the *unnormalised* `contribution()`
   differs by exactly the factor 5. Only **EOS-only** is a genuine fork (max
   Δ`r_i` 0.0383). The decision is still real and still needed — the brief was
   right to demand an ADR — but two of its three options are the same option, and
   a brief that had been believed as written would have reported a resolved fork
   where there was none.

2. 🔴 **The amendment's bar-4 argument is wrong where it matters.** It says
   *"`observe()` is on the protocol, on `FIFOPolicy` **and** on `RSRPolicy`. A call
   to `policy.observe()` in `run_policy_loop` is reached whichever policy is
   passed."* `RSRPolicy.observe` (`src/rsr/retention/rsr.py:427`) **raises
   `NotImplementedError`** — Sprint 2. And the amendment itself notes that
   `tests/test_checkpoint.py:90,390-393` drive `run_policy_loop` with a real
   `RSRPolicy`. So an *unconditional* `observe()` call — the plain reading of bar
   item 4 — would have turned `test_checkpoint.py` red on contact. Reached is not
   the same as usable. Resolved without routing around it: `observe` is an opt-in
   parameter, `observe=False` by default, which bar item 2 independently requires
   anyway. The two bars are consistent; the amendment's justification for bar 4 is
   not. **`rsr.retention.reward` therefore still has zero real importers in
   `src/`** — the model can now produce its input, but the consumer is still a stub.

3. **`## Files in scope` omits two files the bars cannot be met without.** Bar item
   3 requires a mutation, and mutations live in `scripts/mutation_battery.py`; bar
   item 2 requires a measurement, and CLAUDE.md requires every experiment to write
   a `RESULTS.md` next to its runner. I used `scripts/mutation_battery.py` and
   created `experiments/s0-02/`. Flagged rather than silently widened.

4. **The amendment's `fifo.py:35` has no directory and the obvious one is wrong.**
   `src/rsr/retention/fifo.py` does not exist; `FIFOPolicy` is
   `src/rsr/baselines/fifo.py`, where line 35 is indeed `def observe`. Trivial, but
   the brief's own §"Files in scope" uses full paths and this one does not.

5. **"Done when: … and a PR merged."** Not done, and not mine to do. I was
   instructed not to push to `merge-studio-trunk`, and a merge is an owner
   decision. Eight commits sit on `studio-2026-09-20c`, ready.

6. **`model.py:337-338` → `:337`** — the amendment already caught this and it is
   correct as amended. Confirmed at HEAD: `last_attention` at `:335`,
   `attn_out_proj` at `:337`, `memory_gate` at `:388`, `reward.py` 160 lines,
   11 `def test_`, zero importers.

### Defects found in the machinery, not in the brief

7. 🔴 **`.gitignore`'s `/runs/**` silently ate the artefacts my own ledger cites.**
   I committed the ledger with five `how` fields naming `capture_cost.json`,
   `e0c_fifo_with_bridge.json` and `mutations.json` — **none of which a fresh clone
   had**, because `git add -A` skips ignored files without a word. That is the
   exact failure the `/data/` comment twelve lines above documents at length,
   recurring inside the rule written to fix it. `runs/cycle-01-masked-loss/`'s
   `gates.json` and `raw.json` had already been lost the same way. Fixed with
   `!/runs/**/*.json` (heartbeat `.jsonl`, `.pt` and `.log` stay ignored, verified),
   scratch counters re-ignored by name. 3 MB now tracked under `runs/`, zero `.pt`.

8. **`.orchestrator/outbox/` did not exist.** My own role file says to append the
   report there, and also says — correctly — that there is no `.orchestrator/` in
   this repo. I created the path. That contradiction is in a repo-scoped file and
   should be resolved rather than re-discovered next cycle.

9. **`ledger.command()` refuses `.venv/bin/pytest`.** `TOOL_ENTRY_POINTS`
   whitelists the console-script *name*, but the venv path resolves to a
   repo-relative file that is (correctly) not committed, so the literal command I
   ran is unrecordable. Recorded as `pytest …` with the literal invocation in the
   `note`. Cosmetic, but it means the argv in a ledger is not always the argv that
   ran, which is the property that field exists for.

10. **The rsr-researcher role file's schema warning is stale.** It says
    `manifest.json`, `config_hash`, `seeds_actually_run`, `steps_done` and `status`
    are "not yet written by `scripts/ledger.py`". They all are, and I used them.
    Likewise CLAUDE.md's *"`test_reduction.py` and `test_fidelity.py` are currently
    skipped"*: the suite reports **0 skipped**, and `experiments/GATE-1.md` already
    records both as closed. CLAUDE.md is the stale one.

## UNANSWERED BY THE BRIEF

1. **Whether `observe()` should be on in training.** The brief asks for the call
   site and for capture to be off by default; it does not say who turns it on.
   `src/rsr/train/loop.py:292` still calls `run_policy_loop` without it, so the
   bridge is reachable but unreached in the trainer. That is a Sprint 2 question
   (the MC return) and I did not answer it.
2. **Eval-mode capture inside a train-mode step.** Correction 20 requires `r_i` in
   eval; the LM loss needs train. The bridge reports `eval_mode` honestly and
   `reward` refuses a train-mode trace, so today a training run cannot collect
   `r_i` at all without a second forward pass. Nobody has costed that second pass,
   and it is not the number I measured.
3. **Which `r_i` the policy should store.** `gated` and `raw` are both computable
   from one trace (correction 17 wants both against LOO in E0d); the brief does not
   say which the policy trains on before E0d reports.
4. **Whether the `-7.3 %` is acceptable.** I measured it; the brief did not set a
   budget, and I am not the one who decides what the schedule can absorb.

## BELIEVED, NOT VERIFIED

1. **That `369 ± 11` is the right pre-bridge control.** It is the one number in the
   ledger **transcribed from a terminal rather than read from a JSON file** — the
   worktree was removed and I did not pass `--out`. Flagged as such in its own
   `how` field. Re-derivable in ~40 s: `git worktree add <dir> a1c9e857` and rerun.
2. **That the bridge is correct on CUDA.** Everything here is MPS or CPU. The
   einsums and `no_grad` are device-agnostic by construction, but nobody ran them.
3. **That `sum_over_real_query_tokens` is the right collapse.** The *algebra* is
   verified; the *choice against EOS-only* rests on an argument, not a measurement,
   and E0d is the thing that can settle it. ADR-0008 says so in its own text.
4. **That capture stays free when off under `torch.compile` or on a larger `M`.**
   Measured at one shape only, `d=128 S=80 batch=16 M=40`.
5. **That `RSRPolicy` will be able to consume these traces unchanged.** Its
   `observe` raises today, and the shape it eventually wants is a Sprint 2
   decision. The protocol type-checks; nothing has run through it.
6. **That the three declared couplings are the only ones.** I declared the three
   the battery surfaced. A fourth would only appear if a future `reward.py`
   mutation were added.

## NEXT (proposed, not decided)

1. 🔴 **A batched `retrieval_demand`, before E0d — and it is now measured, not
   guessed.** `r_i` costs −30.5 sent/s against the bridge's −28.3 because it runs
   1264 times from Python. `CrossCapture` already holds `[L, B, H, M]` and
   `[L, B, H, M, D]`, so the batched form is a reshape away and would collapse 1264
   call sites into ~80. Not done here because bar item 1 required `reward.py`'s 11
   tests to pass **unchanged**, and they pin the per-row signature. E0d will call
   this function on every held-out step; paying 15 % there is a schedule decision
   somebody should make deliberately.
2. **E0d, which this brief unblocked** — and per ADR-0008 it should report **four**
   rows, not one: `{gated, raw} × {sum, eos_only}` against LOO Δloss. Both
   collapses come from the *same* captured `[B, H, Q, M]` tensor, so it is two
   reductions of one forward pass, not two runs. If EOS-only correlates better,
   ADR-0008 is wrong and is superseded by the measurement, which is what it says
   should happen.
3. **S0-01, still not landed.** The amendment established that this brief did not
   need it; that is not the same as it being done. `tests/test_train_loop.py` is
   still absent, `--policy` is still missing from `loop.py`'s argparse, `srep_norm`
   is still out of the objective, and `--vocab` still defaults to 50257.
4. **S0-04 is now unblocked in the sense that mattered** — its positive control
   needed `reward.py` to have a live caller. It has a live *producer*; the consumer
   (`RSRPolicy.observe`) is still a stub, so I would not call it unblocked yet.
5. **A second forward pass, costed.** If `r_i` must be collected in eval while the
   LM loss is computed in train (correction 20), somebody has to measure what the
   extra eval pass costs before E3's schedule is believed. Nobody has.
