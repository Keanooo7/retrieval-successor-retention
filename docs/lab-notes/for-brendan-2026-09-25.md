# For Brendan — 2026-09-25 morning (overnight 2026-09-24 → 25)

Written by a model: the Studio overnight session, executing `~/research-corpus/handoffs/2026-09-24-overnight-rsr.md` under your go. **Nothing is merged.** Plan (Phase 0): `~/research-corpus/handoffs/2026-09-24-overnight-PLAN.md` (corpus commit `1e3ebff`).

## Headline

1. **Track T: the corpus-size curve ran to completion. Verdict `survived`, but it is the PREREG's own named risk, not a finding about memory.** At N=512 and N=4096, 1000 iterations did not teach the model the answer task **even on its training documents**: answer NLL sits at `chance_ln16` on both the memorisation probe and the held-out set. So "no memory benefit at N=4096" is uninformative. It does show that removing memorisation also removed every other source of learning at this budget. The N=64 reproduction control was exact on every seed (`abs_diff` 0.0).
2. **Track R: 316 new component cards in the research corpus (317 in total), `validate` exit 0.** There is also a corpus-to-RSR note mapping them onto the three measured problems, D-3, and recommendations (a)–(g). The corpus is thin on memory eviction. Belady's MIN and Parrot are **not** in it.

## Track T — corpus-size curve

- **PR:** #44, `exp/corpus-size-curve` (do not merge yet).
- **Ledger:** `runs/corpus-size-curve/ledger.json`, committed in `26c590b`. The run was at `ec9332d` (`provenance.git_sha`), clean tree, `caffeinate -i`, exit code `0`, ledger `status` `ok`. It ran 2026-09-24 23:35:50 → 2026-09-25 02:38 PDT.
- **Arms:** 64 → 4096 → 512, one at a time, 3 seeds × 5 threads, 1000 iterations each, checkpoints 300 and 1000. `arms_not_run` is empty.
- **Commits, in order:** PREREG `0c67042` (alone, first), then brief `f532b34`, implementation `58c793c`, review fixes `ec9332d` (the run SHA), renderer fix `1190c72`, and the run record `26c590b`.
- **`verdict.outcome`:** **survived**. `verdict.detail`: "N=4096 ckpt1000 reached; bar 1 holds and B fails at every checkpoint: no memory benefit on held-out answers at N=4096 by ckpt1000".
- **Primary readout:** Brier over the 16 answer symbols, slots_zeroed − live, at gap 2..M, on common held-out docs [4096, 4160). `B` means every seed ≥ `brier_margin` 0.05. The PREREG justifies the Brier choice from research-corpus `03_Calibration_and_Uncertainty.md:132-134, :144, :148, :162`.
- **Reproduction control** (N=64, ckpt300, S0-03's own held-out): `reproduction_control.measured` = 2.833300079112343, 2.8183460763738126, 2.991847649975033. `abs_diff` = 0.0, 0.0, 0.0 (tolerance 1e-6).

### Per arm and checkpoint (ledger keys `n<N>.ckpt<c>.<column key>`; held-out unless marked; per-seed values in seed order 0, 1, 2)

| arm.ckpt | bar1 | B | Brier16 zeroed − live | Brier16 live | acc live | acc zeroed | NLL live | train (probe) NLL live | train (probe) acc live |
|---|---|---|---|---|---|---|---|---|---|
| `n64.ckpt300` | yes | no | 0.0042, 0.0117, -0.0040 | 0.9441, 0.9537, 0.9762 | 0.0779, 0.0748, 0.0466 | 0.0637, 0.0637, 0.0381 | 2.8410, 2.8831, 3.0314 | 2.6563, 2.4989, 2.4969 | 0.1279, 0.1974, 0.1977 |
| `n64.ckpt1000` | yes | yes | 0.1057, 0.1868, 0.1627 | 1.4416, 1.3828, 1.4601 | 0.1232, 0.1662, 0.1201 | 0.0623, 0.0540, 0.0480 | 6.4673, 6.1437, 6.5424 | 0.0028, 0.0024, 0.0022 | 1.0000, 1.0000, 1.0000 |
| `n4096.ckpt300` | yes | no | -0.0001, 0.0002, 0.0000 | 0.9394, 0.9377, 0.9388 | 0.0510, 0.0679, 0.0367 | 0.0510, 0.0596, 0.0367 | 2.7916, 2.7772, 2.7868 | 2.7838, 2.7810, 2.7920 | 0.0704, 0.0526, 0.0494 |
| `n4096.ckpt1000` | yes | no | -0.0002, -0.0002, 0.0001 | 0.9381, 0.9395, 0.9384 | 0.0694, 0.0554, 0.0777 | 0.0694, 0.0568, 0.0777 | 2.7790, 2.7885, 2.7814 | 2.7789, 2.7870, 2.7800 | 0.0517, 0.0511, 0.0636 |
| `n512.ckpt300` | yes | no | 0.0000, -0.0002, 0.0001 | 0.9381, 0.9391, 0.9403 | 0.0722, 0.0540, 0.0607 | 0.0779, 0.0540, 0.0607 | 2.7822, 2.7896, 2.7979 | 2.7870, 2.7830, 2.7892 | 0.0503, 0.0653, 0.0621 |
| `n512.ckpt1000` | yes | no | -0.0000, 0.0005, 0.0160 | 0.9383, 0.9385, 0.9323 | 0.0722, 0.0554, 0.0763 | 0.0722, 0.0568, 0.0395 | 2.7803, 2.7823, 2.7253 | 2.7860, 2.7713, 2.7138 | 0.0575, 0.0582, 0.0918 |

Column keys: `bar1`, `B`, `heldout.slots_zeroed_minus_live.gap_2_to_M.answer_brier_over_16`, `heldout.live.gap_2_to_M.answer_brier_over_16`, `heldout.{live,slots_zeroed}.gap_2_to_M.answer_acc`, `heldout.live.gap_2_to_M.answer_nll`, `train.live.gap_2_to_M.{answer_nll,answer_acc}`. References: `brier16_uniform` 0.9375; `chance_ln16` 2.7726; chance accuracy 1/16.

**How to read it:**
- **N=64:** at ckpt1000 it memorises its 64 training documents (probe accuracy 1.0). Memory helps held-out answers by Brier on every seed, and the live model is still worse than the uniform forecast. This is the retrieval curve's pattern again, now seen through a bounded score.
- **N=512 and N=4096:** the answer task was never learned. The training heartbeats (`loss_answer_tokens`, not committed) stay near ln 16 from about step 250 to step 1000 on every seed. N=512 seed 2 alone shows a first movement (`n512.ckpt1000.heldout.slots_zeroed_minus_live…brier` 0.0160).
- So the run separates memorisation from retrieval only in the sense that it shows memorisation was the **only** thing learned at N=64 by ckpt1000. That the N=512 and N=4096 models learned nothing at all in 1000 iterations is itself a finding. It is **inference** that more iterations would change it. Nothing was run to test that, because no PREREG covers it.
- The full hand-written account is in `experiments/corpus-size-curve/BRIEF-ERRORS.md`. `RESULTS.md` is rendered from the ledger, and `render_scoreboard.py --audit` exits 0.

### Gates

Before the run, at `ec9332d`:
- `uv run --extra dev pytest -rs --tb=no`: census `passed=1182 failed=0 skipped=0 errors=0`, `$?` = 0.
- `ruff check`: exit 0. `ruff format --check`: exit 0.
- `mutation_battery.py --check-anchors`: exit 0, "143/143 anchors occur exactly once".
- The 7 new mutations were each applied by hand against `tests/test_corpus_size_curve.py`, `test_retrieval_curve.py`, `test_train_loop.py` and `test_train_loss.py`. All 7 are PROVEN: each reddens its gate and no undeclared test.
- An independent read-only reviewer agent checked the implementation against the PREREG and the brief. It found nothing blocking. It ran an end-to-end `execute()` with the real `Ledger` over 5 failure paths. Findings N2, N3 and N6 were fixed before the run.
- `lint_brief` on the brief at base `0c67042`: 0 findings.

After the run:
- `render_scoreboard.py --audit experiments/corpus-size-curve/RESULTS.md`: exit 0.
- The renderer change after the run (`1190c72`) touches rendering only, and `tests/test_corpus_size_curve.py` passed=39 after it.

### Full mutation battery

- **Run:** `RSR_BATTERY_THREADS=12 uv run --extra dev python scripts/mutation_battery.py --check`, at `ec9332d` in `.worktrees/battery-csc`, after training. It ran 02:39 → 06:18 PDT. **Exit 1: "139/143 gates proven by mutation".**
- **The 4 unproven were undeclared couplings, not defects:**
  - S0-03's "answer goes back out of band" mutation now also reddens two new tests that run `train()` to its header.
  - The retrieval curve's `REPRO_TOL` mutation reddens a corpus-size-curve test. **This matters for reading the code:** the corpus-size curve imports the retrieval curve's `reproduction_control`, so the tolerance the run actually applied is the retrieval curve's `REPRO_TOL`. That value is 1e-6, the same as the PREREG's. The curve's own `REPRO_TOL` constant is only asserted equal to it.
  - Two of my declared couplings used substrings, but the battery matches exact node ids.
- **Fix:** `846291c` (on `exp/corpus-size-curve`, pushed) declares all four with reasons. Each of the 4 was re-proven by applying it over the **full** suite with the battery's exact-id rule: PROVEN, 0 undeclared.
- **Not done by 08:00:** a full battery re-run at `846291c`, which is needed before merging #44. It was **started at 06:28 PDT** in `.worktrees/battery-csc` (detached at `846291c`, `RSR_BATTERY_THREADS=12`). At ~3.6 h it should finish around 10:00. Its output goes to `battery2.log` in this session's scratchpad (`/private/tmp/claude-501/-Users-keanooo7-retrieval-successor-retention/b22d7401-2128-46c5-a7df-54573416ecb9/scratchpad/`). Read the `BATTERY_RC=` line: an exit I have not seen is not a pass.

## Track R — research corpus

**R1 (cards).** Corpus commits `b723a44` (cards, taxonomy, INDEX) and `5141530` (applications).
- **Counts:**
  - 316 new cards; 317 including `muon-optimizer`.
  - By status: source-verified 148, source-claimed-unverified 123, open-question 46.
  - `coverage`: all 15 sources COVERED.
  - `python3 tools/corpus.py validate`: "317 cards, 0 errors", exit 0. `index` regenerated.
- **Pipeline:**
  - 13 reader agents, one per source or a pair.
  - 4 adversarial verifiers, one per batch. Verdicts over 326 candidate cards: 291 keep, 27 fix, 8 drop.
  - 13 fix verdicts came back without a corrected card. Fix-up agents rewrote those from the verifier's reason, and none were dropped. The fix-up agents' rewrites were **not** re-verified by a second adversary.
  - 9 cards had empty `scales` and were given scales, and one too-short quote was lengthened.
  - 6 `related` links to dropped cards were removed. 1 duplicate id (`tokenizer-fingerprinting`) was merged.
- **Taxonomy:** readers proposed 50 ids, and 49 distinct ones were **added** to `taxonomy.json`. Cards named in each proposal's `needed_by` were retagged. This is a vocabulary change: see Decisions.

**R2 (mapping).**
- The note is `docs/lab-notes/research-2026-09-25-corpus-to-rsr.md` on branch `docs/research-2026-09-25` (`d7e8fcc`, pushed, no PR).
- It was produced by one synthesis agent plus one adversarial citation verifier. The verifier's verdict was pass-with-corrections: 11 corrections and 3 unsupported claims. All were applied or relabelled `UNVERIFIED -- inference`, and I re-checked the §11 names myself.
- `applications[]` with `project: "rsr"` were appended to 30 cards (24 proposed, 6 rejected). `validate` exits 0.
- **Verdicts on (a)–(g), from the note:**
  - (a) set scoring: partly supported in principle. Not addressed for memory eviction.
  - (b) Belady/Parrot: not addressed. Neither the corpus nor the spec mentions either.
  - (c) batched LOO: not addressed, analogy only. No spec change.
  - (d) E2 skill score: evaluation hygiene supported. The term itself is not addressed.
  - (e) fix the substrate first: supported in direction. The remedy is not addressed. Tonight's Track T result bears on it directly.
  - (f) accuracy instead of NLL: supported as an added readout, contradicted as a replacement for a proper score.
  - (g) both claims hold about different objects. The argmin-invariance claim is contradicted as worded, because `z(ψ̂) + b − ν·max cos` is not invariant to non-affine rescaling (spec `:268`, `:314`).

## Decisions that are yours (batched)

1. **Corpus-size curve, what next.** `survived` came from under-training at N ≥ 512. Options, each needing its own PREREG before any run:
   - (i) the same arms trained longer, e.g. N=4096 to 3000 iterations (~3 h, predicted);
   - (ii) intermediate N (128, 256) to find where learning stops;
   - (iii) stop here and take the retrieval curve's N=64 result as it stands.
   Nothing beyond the committed PREREG was trained.
2. **PR #44.** It changes shared code: `train()` gains `n_documents`; S0-03's `answer_readout` gains Brier16, and `measure()` gains `sets=`. The reproduction control shows S0-03's default path is unchanged. Merge once you accept the reading and the battery (below).
3. **Reading the retrieval curve (PR #43, BRIEF-ERRORS item 4)**, and whether Brier / accuracy / temperature-fitted NLL become co-reported readouts in future PREREGs. The R2 note's decisions 1–2.
4. **R2 PROPOSALS** (the note's "Decisions for Brendan" 3–7): E0d truth under multiple scores; set scoring alongside or instead of ν, after E0d measures D-3; §11 additions via primary-source check; a numeric threshold for E2's "≫"; gate order (substrate before policy arms). All are labelled **PROPOSAL — owner decision**. None is acted on.
5. **Corpus taxonomy:** keep or prune the 49 reader-proposed ids added tonight (`~/research-corpus/taxonomy.json`, `b723a44`).
6. **Housekeeping:**
   - Worktrees `.worktrees/battery-csc` (detached at `ec9332d`), `.worktrees/research-0925` and `.worktrees/for-brendan-0925`, left for you.
   - The run's checkpoints (~2.5 GB, gitignored) are in `.worktrees/corpus-curve/runs/corpus-size-curve/`.

## What failed, literally

- **Clock:** the handoff said "about 22:30"; `date` read 23:05 PDT at session start.
- **The PREREG commit was amended once**, before any code, data or push. An arithmetic slip ("~19 epochs at N = 256") was corrected to "~75 passes over N = 64". The original sha `99ab3af` was never pushed. The amended `0c67042` is the base the brief was linted at.
- **The first R1 fix-up workflow was launched with a placeholder argument** and failed at once with `TypeError: items.map is not a function`, before starting any agents. It was relaunched correctly.
- **R1's first `validate` returned 34 errors:** 13 fix verdicts without a card, 9 empty `scales`, 1 short quote, and dangling `related` links. All were resolved. Final: 0 errors.
- **The first full pytest run failed 1 test,** in my code: `tests/test_exit_codes.py::test_every_converted_checker_exits_through_the_protocol` (a `SystemExit` with a message in `doc_sets`). Census `passed=1181 failed=1`, `PYTEST_RC=1`. Fixed in `ec9332d`.
- **The first proof of the "control read on the common set" mutation was UNPROVEN.** It reddened 3 undeclared tests because of a test fixture. The fixture was fixed, and the re-proof passed.
- **The brief's first lint:** 1 finding, a body line citing an undeclared anchor. Fixed.
- **`render_scoreboard --audit` failed twice** on RESULTS.md: 9 unbacked numbers, then 1. The renderer and BRIEF-ERRORS were reworded. Now exit 0.
- **The R2 note had 11 citation errors and 3 unsupported claims** caught by its verifier: wrong line numbers, one misquote, and one per-seed range overstated. All were corrected before the push.
- **The run commit message was first written with a placeholder sha,** then amended before the push.
- **The full mutation battery exited 1 (139/143).** My pre-run hand proofs had said all 7 new mutations were PROVEN. They were wrong in scope: they ran only 4 test files, not the full suite, and matched declared couplings by substring. The battery matches exact node ids. The fix and the full-suite re-proof are in `846291c`, described above.
- **The experiment itself:** at N=512 and N=4096 nothing was learned in 1000 iterations, which the PREREG named as its main risk.
