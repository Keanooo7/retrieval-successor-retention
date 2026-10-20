VERDICT: LAUNCH WITH FIXES -- 2a: LAUNCH as is, under operator precondition O-1 (PM-1: only after the _mgr-1019 battery has exited); 2b: fixes LB-1 and LB-2 required before 2b's measure (neither touches 2a's path).

---
branch: build/rsr-run3-ranking
reviewed_head: 66352afbbcde5bf6cd0424cda6cee775c78c7783
diff_base: 32475b5 (PREREG body, committed alone)
reviewer: Tier G reviewer, review policy v2 (one review of PREREG + code; at most one confirming round)
written_by: a model (Claude Opus 5.5); not by Brendan
---

# Review: RSR run 3, the ranking-loss test (phase 2a + 2b) @ `66352af`

Scope: `experiments/rsr-run3-ranking/PREREG.md` (body `32475b5`, Amendments 2a-S `8ca001b`,
PM-1 `c7f63ea`, PM-2 `f9eeb48`) and the code diff `32475b5..66352af`: `src/rsr/retention/rank_loss.py`,
the `loss` switch in `src/rsr/retention/rsr.py`, `experiments/rsr-run3-ranking/{run.py, phase2a.py,
phase2b.py, stage_seal.py, launch_2a.zsh, launch_2b.zsh}`, `tests/test_rsr_run3_ranking.py`, the
`tests/test_reduction.py` edit. Context read: PLAN §2 "Phase 2" / "Fixed now", run 2's and run 1's
harnesses (`experiments/rsr-run2/run.py`, `experiments/rsr-run1/run.py`), offline-psi
(`experiments/offline-psi/run.py`), diagnostic A (`experiments/offline-psi-converge/run.py`), phase
1's `common.py`. Read-only on code; this file is the only change on the review branch.

Labels: **LAUNCH-BLOCKING** (would change a number/label, touch sealed data or read TUNE/EVAL2 at the
wrong stage, break run-2 bit-equality, leak gradient/age, misimplement the registered rule, or violate
CLAUDE.md) and **FOLLOW-UP** (everything else). Each blocking item says whether it blocks 2a or only 2b.

## Executed (literal)

All in a scratch worktree detached at `66352af` (`git worktree add --detach`), `uv sync --frozen --extra dev`.

- **Full suite**, `PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run --lane cpu-det --slots 2 --wait 3600 -- .venv/bin/python -m pytest -q > log 2>&1; rc=$?`
  -> `rc=0`, census `passed=2787 failed=0 skipped=0 errors=0` (T1(b) `@slow` and `tests/test_reduction.py` included; nothing skipped).
- **Hand check of one group** (script against `rank_loss_groups`): `G = [0.5, 0.2, 0.2, 0.2, 0.1]`,
  `s = [1.0, 0.0, 0.5, -0.5, 0.2]`, k = 3, ω = 8. k-th smallest = 0.2, one strictly below, three tied,
  so `b = [0, 2/3, 2/3, 2/3, 1]`. Pairs `G_i > G_j`: (0,1) (0,2) (0,3) (0,4) (1,4) (2,4) (3,4); the three
  tied pairs form none. Weights by hand `[17/3, 17/3, 17/3, 8, 8, 8, 8]`.
  `ℓ_g = Σ w·softplus(s_j − s_i) / Σ w` = 0.5758607009 by hand, 0.5758606791 from the code (float32).
  A second group with all-equal G forms no pair and is excluded and counted (`n_groups_no_pair = 1`);
  a third with n = 2 ≤ k has b = 1, 1 and one pair weighted 8: 2.1269280 both ways. Batch loss = mean of
  the two groups with a pair: 1.3513944 both ways.
- **2a stand-in pipeline smoke, seed 0, U documents only**, scratch out-root, one cpu-det slot:
  `fit --rc2` (seeds 0, 1, 2), `fit --omega {2,8,32} --smoke-cap 500`, `select --smoke-standin 32`, `deploy --smoke-standin 32`. **R-C2 `==` on every seed and both arms**: seed 0 cf_all 1.6307810744736677 / cf_live 1.2519301986195617; seed 1 1.5766100367590767 / 1.1455279376524217; seed 2 1.790370677418465 / 1.381025548716497 (each equal to offline-psi's recorded step-3000 value; rc2 exit 0 x3). Fits, select and deploy exit 3 by design (smoke), `controls_failed = []`, no error; R-SEAL census ok with 0 sealed ids at every stage; R-C7 0 mismatches on the TUNE stand-in passes and the EVAL stand-in pass; R-C8 and R-C9 ok. The smoke's caps and label are stand-in numbers from 500-step fits on U-val and are not results.
- `git show --stat` / `--numstat` / `-U0` on the four PREREG commits: each touches only
  `PREREG.md`; body 547+/0−; 2a-S 22+/0−; PM-1 29+/0−; PM-2 27+/0−; every amendment hunk is
  `@@ -N,0 +N+1,…` at the file's end (appended, insertions only). No other commit in
  `32475b5..66352af` touches `PREREG.md`. Amendment 2a-S's record
  `runs/rsr-run3-2a-smoke/time-seed0.json` hashes to the pinned `579f8c07…297b8b` and was committed
  (`17897a0`) before the amendment (`8ca001b`).
- PM-1 wall projection by granted slots, from `phase2a.wall_for_slots` on the 2a-S record:
  1 slot 10.48 h (refused), 2 slots 5.71 h, 3–4 slots 3.49 h, 5–8 slots 2.37 h, ≥ 9 slots 1.27 h (all allowed).

Never generated: TUNE `[997200, 997712)`, EVAL2 `[984000, 986048)`, EVAL `[995000, 997048)`. Nothing launched.

## 1. The loss (§2.1) -- matches

`rank_loss.py` implements §2.1 exactly:
- `P = {(i, j) : G_i > G_j}` over valid cells, strict `>` on the stored float32 G, so exact ties form no pair
  (`pair_weights`, line 107).
- `b` is the exact fractional bottom-k membership (`bottom_k_membership`): 1 strictly below the k-th
  smallest valid value, `(k − #below)/#tied` at it, 0 above, and all 1 when `n ≤ k`. No RNG.
- `w_ij = 1 + (ω − 1)·max(b_i, b_j)` on pairs, 0 elsewhere.
- `sp[g, i, j] = softplus(s_j − s_i)` with `i` the larger-G slot (index check: `scores.unsqueeze(1)` is
  `s_j`, `unsqueeze(2)` is `s_i`).
- `ℓ_g = Σ w·sp / Σ w` per group, then the mean over groups with `|P| ≥ 1`. Groups with no pair are
  counted and excluded.
- No temperature and no secondary loss.

**2a pair set = `cf_all`.** `as_groups` reshapes offline-psi's `P[b, t, i] = ψ̂(s_i, c_t)`, `G[b, t, i]`
and `cell_mask("cf_all") & ~isnan(G)` to one group per `(document, t)` over `i < t` (§12 item 2). The
bottom-k is taken over that same set. **2b pair set** = `mc_targets`' rows of one `(row, t)`, which are
contiguous because each `_Obs` is appended whole, so `unique_consecutive` groups them correctly. `G` is
the same detached `G_i(t)` that L9 regresses. T4 checks the padded ragged path against a brute-force
loop (`test_t4_policy_rank_loss_equals_the_padded_definition`).

J (§4.2) is the same batch loss over every U-train group (`full_objective`: per-group sum in float64
over groups with a pair, `no_grad`, no RNG). `converged()` is §4.2's rule: the first check `k ≥ 2000`
with `best(k) ≥ (1 − 10⁻³)·best(k − 2000)`, where `best` takes the minimum over finite J at or before
the step and J is recorded at step 0. The fit stops there and `K_conv = k`; otherwise the label is
CAP-HIT at 200,000. φ is saved at `K_conv`, and at 3000 and 10,000 where reached. The constants are
machine-checked against the committed text (`prereg_check`).

## 2. Switch off = run 2 -- holds

- The `"mc"` branch of `retention_loss` is the unchanged line
  `(psi − target.detach()).pow(2).sum() / n_groups`. The new imports (`time`, `rank_loss`) draw no RNG.
  With `rank_stats is None`, `last_mc_stats` is byte-identical.
- **T1(a)** loads `rsr.py` from `728a42f` as its own module and asserts that its dependencies are
  unchanged since then. It drives both versions on the same stream, both RSR arms, and checks bitwise
  equality of the buffers, the loss, every φ gradient and the `last_mc_stats` JSON.
- **T1(b)** is a real end-to-end test. It runs run 2's psiC seed-0 command for 20 steps (16 warm + 4
  on-policy) against `728a42f`'s `src` (via `git archive`) and against HEAD's `src`. `PYTHONPATH=src`
  precedes the editable install's `.pth`, and `-m` from the repo root puts the root (which has no `rsr/`)
  first. A probe asserts that each side imports the tree it claims and that only HEAD has `rank_loss`.
  It then compares heartbeats (excluding `VOLATILE_KEYS`), diagnostics, evictions, checkpoint tensors
  (model and policy) and run_id. The harness (`experiments/rsr-run{1,2}/run.py`) is HEAD's on both
  sides, as §3 T1(b) states ("behavioural comparison of outputs"). It is `@slow`, but `addopts` does not
  deselect `slow`, so it runs in the full suite.
- **T1(c)**: `reduction_to_tg()` is built from `REDUCTION_SWITCHES`. The three new fields have off values
  `("mc", None, None)`, and `test_t1c_reduction_to_tg_is_unchanged` asserts that every old field is
  equal to its `728a42f` value. The `tests/test_reduction.py` edit **only adds** flip cases for `loss`,
  `rank_k` and `rank_omega` to the existing single-flip test (6+/0−). The `set(flips) == set(REDUCTION_SWITCHES)`
  assertion would otherwise fail on the new fields, so this is a legitimate extension, not a
  weakening. No other assertion in the file changed.
- **T1(d) / R-C2**: `fit_loop` with `loss="mc"` reproduces offline-psi's `fit` op for op on the
  fixture. On real data, R-C2 passed `==` on seed 0 for both arms (see Executed).

## 3. Isolation -- holds

Under `"rank"`, ψ̂ is computed at the same detach site (`s.detach()`, `c.detach()`), and G is
detached inside `rank_loss`. Only `psi` carries a graph. T2 runs the model in `train()` with
`requires_grad=True` on every transformer parameter, and after `backward()` every model grad is
`None` on both arms. The content arm's rank path never reads `t − key`; `age_of` is read only for the
no-grad diagnostics. T3 permutes `written_at` and leaves the content arm's loss unchanged, and the
age arm's loss changes, so the test can see a leak. In 2a, ψ̂ is computed on precomputed tensors,
`optimizer_holds_only_phi` holds, and `no_grads(model)` is checked before and after. `b` appears
nowhere in the loss. CLAUDE.md: no "unbiased", "consolidation", scaling or §15 text in the diff.

## 4. 2a stages and seals -- correct (one hardening, F0)

- **Seals.** `stage_seal.install` narrows phase 1's process-wide guard (imported, never copied) to the
  §1 table. It can only remove the range the stage opens, and one stage per process is enforced. In
  2a, TUNE is open only at `select` and EVAL2 is never open. T6 checks every stage in its own
  subprocess (stubbed generator, early `from … import` copy rebound), and a census asserts 0 sealed
  ids generated. No sealed literal appears in phase-2 code or tests.
- **Order.** In `fit → select → deploy`, `select` requires all three ω fits to have exit 0 (R-C6) and
  then runs the TUNE closure (run 1's `check_c0` with `eval_range = TUNE`, plus disjointness from EVAL
  and EVAL2). Both happen **before** any model loads; a failure is exit 3 and TUNE is not read
  further. The selection rule is: highest TUNE cap point, ties to the smaller ω (ascending scan with
  strict `>`), and non-finite or `None` points skipped. If every point is `None`, the seed is exit 3.
  TUNE records are written without `answers` / `real_sum` / `real_n`, and only `hits` feed the cap, so no
  accuracy statistic is computed on TUNE. The cap is run 2's `ceiling` with
  `R1.boot_weights(512, seed, 2000)`, paired across the three ω, FIFO (model-free `simulate`) and
  FactFiller (`ff:{seed}:{doc_id}`). The selection is written to `selection.json` and hashed before
  `deploy`. `deploy` refuses unless the file matches its hash, the frozen φ hashes to the selection,
  and the EVAL output does not yet exist. EVAL is therefore read once per seed, by the frozen selected
  φ (R-C6 also checks this from events).
- **EVAL label.** offline-psi's `score_seed` → `headline` → `R2.label_contrast` → run 2's `seed_label`
  (imported), with δ from `R2.read_delta(seed)` (R-C9 `==` the registered table), FIFO from run 2's
  records, and `R1.boot_weights(2048, seed, 2000)`. R-C8 pairs all 2048 documents against run 2's fifo.
- **Gate.** `summarize` reads labels only when every control and stage exit is 0.
  `gate_to_2b = all(labels[s] == "WIN" for s in registered seeds (0, 1, 2))`. The PREREG seeds are machine-read.
  `make_plan` independently refuses unless 2a exit 0, `gate_to_2b is True` and every label is WIN. This
  is "WIN on all 3 seeds" exactly.
- **R-C2** compares `OP.val_loss` at step 3000 against offline-psi's recorded history value with `==`,
  on cf_all and cf_live.
- **PM-1 slot check**: granted slots = probed free + the probe's own slot. With ≥ 9 slots, 2a launches.
  With fewer, the wall time is recomputed by an event-driven list schedule from 2a-S's measured times
  and must be ≤ 8 h. The 2a-S record is machine-read from the amendment text and sha-checked. The
  launcher fits nothing unless both launch-checks and the slot check exit 0. This is correct as
  PM-1's second bullet states it. See O-1 for its first bullet.

### F0 (FOLLOW-UP, recommended before the 2a launch; was a blocking candidate): a failed R-C2 does not stop TUNE's first read

`select_seed` (phase2a.py:833-849) gates TUNE on `R-C6.fits_done` over the three ω fits only. `rc2.json`
is never read before `tune_closure` / `R1.docs_by_id(TUNE)`. R-C2 is the harness-identity control. If it
fails, 2a is exit 3 whatever else happens (finalize: `R-C2` and `stage_exits`). Even so, `select` would
still generate all 512 TUNE documents per seed, deploy three heads on them and write the caps. That
spends the sealed TUNE split on a run whose numbers cannot be read, and a repaired re-run would then
select on an already-read TUNE. PM-2 applies the same principle to EVAL2 ("refuses before EVAL2 is
opened … the sealed range is not spent"). R-C2 finishes in about 15 s, long before any fit, and the
launcher's per-seed `wait` already orders it ahead of `select`.
**Fix (about 5 lines and a test):** in `select_seed`, before `tune_closure` and for the non-stand-in
path, require `seed_dir/rc2.json` to exist with `exit == 0` and `controls["R-C2"]["ok"]`, and fold it into
`R-C6.fits_done`. Better still, require it for all registered seeds, which costs nothing because the
three R-C2 processes start first. Add a fixture test: a failed or missing `rc2.json` makes `select` exit 3
with 0 TUNE ids generated.
**Why FOLLOW-UP, not blocking:** R-C2 is deterministic (1 thread, pinned inputs) and passes `==` on all three seeds and
both arms at this head (Executed), so at `66352af` it would not fire and TUNE would not be spent this way. The guard is
still worth the 5 lines if any commit lands before launch.

### O-1 (operator precondition for 2a, from PM-1 bullet 1; not a code change)

PM-1 binds: "2a launches only when the cpu-det lane can grant the layout's slots, i.e. **after the
integration battery (8 slots) has exited**." `slot-check` does not enforce this. It counts free
slots, and with the battery holding 8 of 12, 3 or 4 granted slots project 3.49 h, which is ≤ 8 h, so 2a
would launch beside the battery. At review time the battery in `_mgr-1019` holds cpu-det slots 1–8
(job `i2-10817167e270-…`). **Launch 2a only after that battery has exited.** The launcher's
`launch-slots.json` then records the grant.

## 5. 2b

- **K_2b rule**: `k_2b = ceil(max(3000, max_s K_conv(s, ω_s)) / 1500)·1500` (§5.3). `make_plan` takes
  K_conv from the selected rows of 2a's ledger, and C-K re-derives it and checks the launch record and
  `steps_done` for both RSR arms. Correct.
- **PM-1**: fifo is run 1's command at `fifo_iters()` (R-1, machine-read from run 1's owner table)
  with its own cadence. C10 runs run 1's function unchanged on a fifo view (iters = R-1) and an RSR
  view (registered iters replaced by K_2b). C-F checks no trainable parameter, no optimizer, and model
  tensors equal to the substrate's via run 1's `c3_frozen`, tested on run 2's real fifo checkpoint.
  Correct as ruled.
- **PM-2's five reads**: `checkpoint_path` maps fifo at K_2b to R-1; C6 per arm; C2 gives the RSR arms
  `[0, K_2b)` and each RSR arm plus fifo `[0, R-1)` with fifo's cfg; C4 uses fifo's cfg; Δloss_lm is
  matched steps ≤ R-1 and labelled. Each is the identity at K_2b = R-1 (tested). I read run 1's
  `c2_pairing` and `c4_measure`: both are arm-generic (`arms = list(beats)`) and iters-driven, so the
  views are well-formed. The logic is right.
- **EVAL2** opens only at `measure`, after `measure_precheck`, the training footers and `pm2_precheck`.
  Run 1's `measure_seed` applies C0 closure to `eval_range = EVAL2` before any held-out pass.

### LB-1 (LAUNCH-BLOCKING, blocks 2b only): C-ω, C-K, C11′ and C-F are evaluated after EVAL2 is read, and C-ω's parser rejects this file's own heading convention

`measure_stage` (phase2b.py:732-758) calls `R2.measure_seed` (EVAL2 generated and read) **before**
computing C11′, C-K, C-ω and C-F. None of these needs an EVAL2 document. Separately,
`amendment_2bS()` matches only `^## AMENDMENT 2b-S\b` (case-sensitive) plus exact
`- 2a ledger sha256: \`…\`` and `- K_2b: N` lines. Every amendment in this PREREG is headed
`## Amendment …`, and §5.3 registers no format. An Amendment 2b-S written in the file's own convention
therefore makes C-ω fail **after** EVAL2 is spent, and 2b is exit 3 with the fresh sealed split gone.
**Fix:** compute C-ω (including the 2b-S parse), C-K's plan/launch half, C11′'s label/launch half and
C-F before `measure_seed`, and refuse before EVAL2 on any failure, as `pm2_precheck` already does.
Match the heading case-insensitively or as `## Amendment 2b-S`. Add a test using the conventional
heading.

### LB-2 (LAUNCH-BLOCKING, blocks 2b only): the K_2b ≠ R-1 control paths are first exercised on real artifacts after EVAL2 is opened

The builder notes that the C2 and C4 wrappers are tested only with run 1's function stubbed. More
broadly: S-2b sets `FIFO_STEPS=$STEPS` (launch_2b.zsh:72), so in the smoke K = R-1 = 48 and every PM-1/PM-2
wrapper takes its identity branch. The per-arm C10, C6, C2, C4 and C-F views therefore never run
against real run-1 functions on real training artifacts before the real `measure`. In run 1's
`measure_seed`, a C10 or C2 failure does **not** stop the seed before the held-out passes (A3b.3:
"numbers written, marked not read"), and C4 runs **after** them. A wrapper or view defect, such as a
key the real C10 reads from `launch-git.json`'s `fifo_*` fields, would surface only after EVAL2 has been
read, giving exit 3 with EVAL2 spent. That is "touch sealed data" for 2b.
**Fix (2b only, before Amendment 2b-S or with it):** exercise K_2b ≠ R-1 end to end on real artifacts.
For example, run an S-2b variant with fifo at a smaller multiple of 3 than the RSR arms (such as 48 vs
96), then run measure's pre-held-out controls (C0–C10, C2, C3, C8, C-F, C4) through the wrappers with
the held-out range replaced by an already-used stand-in, never EVAL2. Alternatively, make a C2/C10
failure stop the seed before the held-out passes in 2b's measure. **Not blocking for 2a**, which uses
none of this.

## 6. FOLLOW-UP (8: F0 in §4, F1-F7 below)

F1. `deploy` does not require `select` exit 0. `selection_guard` checks only the selection file, its
    hash and the φ hash. `select` writes `selection.json` before its own R-C7, R-C4, census and pins
    are judged, so a `select` that failed a control still leads to the single EVAL read. EVAL is
    already used and finalize marks everything NOT READ, so no label changes. Suggest that `deploy`
    refuse unless `select.json` has `exit == 0`. This could share F0's patch.
F2. `fit_done`'s smoke clause is reachable from a non-stand-in `select`: a `--smoke-cap` fit file in
    the real out-root would count as done. Today this is blocked only because `launch_2a.zsh` refuses
    when `events.jsonl` exists. Suggest that the non-stand-in `select` require `exit == 0 and not smoke`.
F3. `cell_mask & ~isnan(G)` admits ±inf, while §2.1 says "finite G". Δ presumably holds no inf. A
    `torch.isfinite` assertion in R-C1 or R-C4 would make this explicit.
F4. A rank batch with no pair returns `None`. In 2a, `lt.backward()` then raises, giving exit 3. In 2b,
    `retention_loss` returns `None`, so that step makes no φ update (AdamW also skips weight decay on
    `None` grads), unlike L9, which always steps. C11′ would then fail that beat (`l_rank` is `None`).
    This is unreachable in practice (16 documents × 47 groups), but worth a counted note.
F5. §4.3 says the selection is "written to the ledger and hashed before deploy". It is written to
    `selection.json` + `selection.sha256` and to `select.json` before deploy, and `ledger.json` only at
    finalize. The spirit holds (hash file and event order). The wording or the code should agree.
F6. 2b `measure`: `c_es` loads the ES model before `measure_seed`'s EVAL2 C0 closure. §5.7 says
    closure comes "before any model loads". This is harmless (no EVAL2 read) but out of order.
F7. `test_t1c_test_reduction_is_in_the_suite` only asserts that the file exists. The real guarantee is
    that the full suite runs it (it does; see the census).
