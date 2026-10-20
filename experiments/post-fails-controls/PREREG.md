---
run_id: "post-fails-controls (one run id per item: post-fails-controls-1a ... -1e)"
item: "PLAN-2026-10-19 phase 1: controls from data on disk, after RSR run 2's FAILS (items 1a-1e; 1f is a reading-only note)"
question: >-
  (1a) Can a legal content-only fixed rule beat FIFO through this frozen reader on run 2's EVAL?
  (1b) Is sentence kind, and is an assert's pending/answered status, linearly decodable from what
  psi_hat sees, and where do E[G | kind] and E[G | status] rank? (1c) Does "attention ranks the top
  of the ranking well and the bottom badly" survive once demand is measured strictly after the
  decision step? (1d) Is FIFO's model-free strength a quirk of the question-spacing mix? (1e) Does
  single leave-one-out mis-value slots whose value is shared?
class: "DIAGNOSTIC (Tier D, review policy v2): no review before data, one review after; rsr-verifier once per item that has numbers. Each item has its own registered reading rule (PLAN §2). Only 1a feeds a gate (Gate P2, applied by the PM); 1b-1e gate nothing."
not: "Not E1, not E2, not a re-run of run 1, run 2, offline-psi, diagnostic A or the LS check; changes none of their labels. Trains no policy (1b fits linear probes on precomputed tensors only). No gradient into the transformer or W_sent. No age in psi_hat. No scaling claim. No cognitive claim. §15 and release condition 7 untouched."
attempt_context: "Written after RSR run 2's verified FAILS (9c432b6) and the four post-FAILS diagnostics (run2-vs-random 6ed6583, offline-psi dc6d1bd, diagnostic A 9c998d3, LS check 320c58a). Learned content-only policies attempted so far: run 1, run 2 (on-policy); offline-psi, diagnostic A and the LS fit are offline fits of the same class. Phase 1 fits no policy. PLAN §2 counts phase 2 as attempt 3."
seeds: "[0, 1, 2]"
sealed:
  TUNE: "[997200, 997712)"
  EVAL2: "[984000, 986048)"
status: "COMMITTED ALONE by the PM on 2026-10-20, before any phase-1 code or data (PM review: §11). Amendments are appended, dated, committed alone; the text above them is never edited."
---

# PREREG: post-FAILS controls (phase 1)

**Drafted 2026-10-19 by a model** (Claude Opus 5.5, an `rsr-researcher` session dispatched by the
PM). **Not written by Brendan.** Authority: PLAN-2026-10-19 §0 (phases 0, 1 and 2 GO) and §2
(phase 1 table, Gate P2). Where this file and PLAN §2 differ, every difference is listed in §9
and each one is **tighter**, never looser.

Base: `integ/2026-10-19` @ `654e70c` (run 1, run 2, B0, CF, γ-ceiling, E0d, ES, offline-psi,
diagnostic A, LS and their verifier records). Committed on `22662bc`, which differs from `654e70c`
only in `scripts/battery_env.py` (§11).

## 0. Conventions used by every item

- **Seeds** 0, 1, 2. The seed is the unit; nothing is averaged across seeds into a claim, and
  every per-seed value is ledgered with its spread. An sd of exactly 0 across seeds is flagged
  (`sd_exactly_zero`), never reported as clean.
- **Substrate:** fresh-stream arm B `ckpt-003000.pt`, run 1's SUBSTRATE pins
  (`experiments/rsr-run1/run.py:186-200`), transformer and `W_sent` frozen, eval mode, CPU,
  `cpu-det` lane via `orchestrator.slot`, 1 thread per process.
- **Bootstrap:** run 1's, imported, never re-implemented: `R1.boot_weights(n_docs, seed, 2000)`
  (per-document resampling, paired across every arm or scorer of a seed; `torch.Generator`
  seeded `20260927 + seed`, `experiments/rsr-run1/run.py:203,2345-2352`); percentile 95% CI;
  a ratio statistic is the ratio of pooled per-document sums per replicate
  (`R1.pooled_replicates`).
- **Every number in a RESULTS file is generated from a ledger** written by `scripts/ledger.py`
  (`runs/post-fails-controls-<item>/{manifest.json, ledger.json, claims.json}`). No number is typed.
- **Exit codes** (`rsr.exit_codes`), per item: **0** ran and every control passed, whatever the
  numbers are; **1** a defect after the controls passed (a NaN in a reported statistic, a raise in
  the computation); **3** did not run, or a control failed (numbers still written, marked NOT
  READ, and the item's reading rule is not applied). `$?` is read directly. Exit 5 is not used.
- **Tests first.** The harness `experiments/post-fails-controls/run.py` (subcommands `1a` … `1e`,
  `finalize`) is built against this committed file, with tests in
  `tests/test_post_fails_controls.py`, and the full suite must pass at the exact launch head.
- **Timing.** Every "timing" line below is a **placeholder**. Before launch, the builder runs a
  timing smoke per item (on non-sealed, already-used documents only), and writes the measured
  numbers and the projection in a dated **smoke amendment**, committed alone, before any item
  launches. A projected wall time above 8 h for any one item stops the launch and goes back to
  the PM.

## 1. Sealed ranges (registered here, never read in phase 1)

| name | ids (per seed's generator) | size | first read |
|---|---|---|---|
| **TUNE** (phase 2a selection) | `[997200, 997712)` | 512 per seed | phase 2a |
| **EVAL2** (phase 2b headline) | `[984000, 986048)` | 2048 per seed | phase 2b |

- Both lie below `GEN_SEED_K = 1_000_003` (`experiments/rsr-run1/run.py:155`), so `seed·K + id`
  cannot collide across seeds.
- **Disjointness.** Both are disjoint from every document-id range any PREREG, RESULTS, `run.py`,
  test or review on any of the 107 local branches names (inventory: §1.1). Both are disjoint from
  each other.
- **EVAL2 is not the PM's candidate `[997952, 1000000)`.** That candidate contains ids
  `998000`-`998003`. These are the run-1 harness test fixture's held-out range
  (`tests/test_rsr_run1_harness.py:53`, `EVAL = (998000, 998002)`) and the hand cases of review
  `docs/reviews/feat-rsr-run1-harness/6267fd645698.md:267` ("non-EVAL documents 998000–998003"),
  so those documents have been generated and read by earlier sessions. It also overlaps the
  range-checker unit-test window `(999000, 1001048)` (`tests/test_newcomer_bakeoff.py:125`; it
  generates no documents, per `docs/reviews/prereg-rsr-run1/6a4c5a8c6785.md:148`). Removing
  `998000`-`998003` and `[999000, 1000003)` leaves no 2048-id stretch in `[997048, 1000003)`.
  `[984000, 986048)` lies in the gap between B2's claimed EVAL `[940000, 980000)` and the
  bake-off's `[990000, 992048)`; a search of every local branch for any integer in
  `[980000, 990000)` (and `98x_xxx` forms) hit nothing inside `[981048, 990000)` (§1.1).
- **TUNE is the PM's candidate unchanged**: no integer in `[997048, 998000)` occurs on any local
  branch except `997048` itself, the EVAL upper bound (§1.1).
- **Closure under each seed's vocabulary is NOT checked now.** Checking it would generate the
  documents. It is checked at phase 2's first read of each range, as run 1's C0 does
  (`R1.check_c0`), and a failure there is phase 2's exit 3.

### 1.1 Inventory (searched 2026-10-19 over `git for-each-ref refs/heads`, 107 branches)

Document-id ranges named on any branch with an upper bound ≥ 4000 (unit-test fixtures marked):
`[64, 1088) ∪ [4096, 4160)` (U); run 1's `USED_RANGES` (`experiments/rsr-run1/run.py:157-173`:
`[0,64)`, `[20160,52160)`, `[4160,52160)`, `[4096,4160)`, `[64,128)`, `[0,4096)`, `[64,1088)`,
`[52160,148160)`, `[52160,884160)`, `[262144,263168)`, `[900000,920000)`, `[920000,936000)`,
`[936000,937024)`, `[940000,980000)`, `[990000,992048)`); run 1 POOL `[993000,993256)` and EVAL
`[995000,997048)` (also run 2, ES, offline-psi, run2-vs-random); ES VAL `[994000,994512)`;
offline-psi, diagnostic A, LS, CF, γ-ceiling: U only (EVAL sealed in diagnostic A and LS, their
C5); E0d `[262144,263168)`. Sub-ranges and fixtures: `[512,4096)`, `[1024,40000)`, `[3000,9000)`,
`[4160,5760)`, `[4160,21760)`, `[5760,21760)`, `[935000,937024)`, `[940000,941024)`,
`[993000,993256)` prefixes, `[995000,995016)`; test-only `(100,60000)`, `[4000,…)` fixtures,
`(52160,262128)`, `(262000,263000)`, `(900000,902048)`, `(910000,936000)`, `(936000,940000)`,
`(979000,981048)`, `(990000,992047)`, `(998000,998002)`, `(999000,1001048)`,
`(1000000,1000008)`, `(1000000,1000010)`.

Every integer in `[997048, 1000003)` on any local branch (regex over decimal and `_`-grouped
forms; hits by content, de-duplicated across branches):

| hit | file:line | what it is |
|---|---|---|
| `997048` | many (e.g. `experiments/rsr-run1/run.py:150`, `runs/offline-psi/manifest.json:12`) | EVAL's exclusive upper bound |
| `(998000, 998002)`, `998000–998003` | `tests/test_rsr_run1_harness.py:15,53,503,505,519`; `docs/reviews/feat-rsr-run1-harness/{549a6066015c.md:25, 6267fd645698.md:236,267,319, 63f930d8e6ce.md:29, d86af310fdb7.md:33,67}` | test fixture held-out ids; review hand-case documents |
| `(999000, 1001048)` | `tests/test_newcomer_bakeoff.py:125`; `docs/reviews/prereg-rsr-run1/{6a4c5a8c6785.md:148, 7c58777ab5ad.md:132,379}`; `experiments/rsr-run1/PREREG.md:1279` | range-checker unit test; no documents |
| `(1_000_000, 1_000_008)`, `(1_000_000, 1_000_010)` | `tests/test_rsr_run1_harness.py:708,710,724`; `tests/test_b2_psi_probe.py:153` | refusal-test ranges (above both sealed ranges) |
| `999999` | `tests/test_checkpoint.py:367` | a process id, not a document |
| `1000000.` | `tests/test_orch_lanes.py:298` | a memory-page count, not a document |
| `-998788.759…` | `runs/b2-psi-probe-fit/phaseA/ckpt3000-seed1.r2-corrected.json:148` | a float, not a document |
| `1_000_000` | `third_party/ThoughtGestaltCode/tg/Rough/subset_dataset.py:476,536` | a token count |

The literal command and its full output are in the drafter's return to the PM; the builder
re-runs the search at the launch head and ledgers it (G-C1).

## 2. Global controls (every item; failure = that item's exit 3)

| id | control | expected |
|---|---|---|
| **G-C1 sealed** | (a) Every document constructor phase 1 can reach (`rsr.data.synthetic._generate_document`, `generate`, and through them `R1.docs_by_id`, lookahead-room's `documents`) is wrapped for the whole process by a guard that raises `SealedRange` for any id in TUNE ∪ EVAL2, at any seed and any `SyntheticConfig`. (b) Each item ledgers the ids it generated, by range (count, min, max), and asserts 0 in TUNE ∪ EVAL2. (c) A test proves the guard raises on a sealed id and that 1d's modified configs go through it. (d) The sealed bounds are machine-read from this file's front matter, never retyped in code. (e) At launch, the §1.1 search is re-run on every local branch and its output ledgered; a new hit inside TUNE or EVAL2 is exit 3 and goes to the PM. (Pattern: psi-ls-capacity C5, `_seal_eval`.) | true |
| **G-C2 substrate** | `shasum -a 256 -c ~/rsr-substrate/2026-09-27/MANIFEST.sha256` rc 0 at the start and end of every item that reads it; that manifest's own sha256 equals `MANIFEST_SHA256` (`experiments/rsr-run1/run.py:201`); each checkpoint read equals its SUBSTRATE pin before and after | true |
| **G-C3 suite** | The full suite at the exact launch head: rc 0 read directly; `passed / failed / skipped / errors` ledgered; any skipped test is named and never counted as a pass | rc 0 |
| **G-C4 frozen** | No transformer or `W_sent` parameter is in any optimizer or receives a gradient; `rsr.constants.record()` is never called; no frozen constant is read as a default | true |
| **G-C5 seeds** | `seeds_actually_run` is written from what ran, and equals `[0, 1, 2]` | true |

## 3. Item 1a: FactFiller and the kind-oracle through the frozen reader

**Inputs.**
- Harness: run 2's, imported, never copied: `experiments/rsr-run2/run.py` and
  `experiments/rsr-run1/run.py`, byte-identical to `728a42f` at `654e70c` (checked 2026-10-19:
  `git diff --quiet 728a42f HEAD -- experiments/rsr-run1/run.py experiments/rsr-run2/run.py
  experiments/lookahead-room/run.py` rc 0). `src/` differs from `728a42f` in
  `src/rsr/model/tg/policy_loop.py` and `src/rsr/train/loop.py`; A-C2 is the behavioural check.
- Documents: **EVAL `[995000, 997048)`, 2048 per seed, ALREADY USED** (run 1, run 2, Expire-Span,
  offline-psi, run2-vs-random). Not a fresh range; every 1a number is labelled "EVAL, already used".
- Reader: the fifo arm's checkpoint, payload key `"fifo"`, as run 1's `oracle`, `lru` and `random*`
  reference passes use it (`experiments/rsr-run1/run.py:1611-1622`); `eval_forward` decision context;
  blocks of 16 documents (`R1.run_heldout`, `run_batch`).
- Run 2's records, read, not recomputed:
  `.worktrees/rsr-run2-run/runs/rsr-run2/heldout/seed{s}/{fifo,oracle}.jsonl.gz` (sha256 ledgered
  at first read). Run 2's ledger `runs/rsr-run2/ledger.json` (byte-equal to the run worktree's,
  `cmp` rc 0 on 2026-10-19).
- δ_seed: machine-read from `.worktrees/b2-psi-probe/runs/b2-psi-probe-fit/ledger.json`, key
  `ckpt3000.seed{s}.decisions.delta["0.9"]`, checked against run 2 PREREG §2.1's table
  (`experiments/rsr-run2/PREREG.md:97-99`) by run 2's `read_delta` (control Cdelta).
- Kind-oracle class means: `runs/b0-ceilings/ledger.json`, keys
  `B.ckpt3000.g09.E_G.kind.{assert,query,filler}`, sample `seed` (B0's `m_kind`, ckpt3000,
  γ = 0.9). **SEEN.**

**Procedure.** Two new held-out passes per seed, each run 1's record format (`doc`, `answers` =
`[t, gap, ok, nll]`, `hits`, `victims`, `evictions`), written to
`runs/post-fails-controls-1a/heldout/seed{s}/{factfiller,kindoracle}.jsonl.gz`:
- **factfiller**: `R1.PerRow` over lookahead-room's `FactFiller(doc, random.Random(f"ff:{seed}:{doc_id}"))`,
  one per document, built per block exactly as `R1.reference_policy` builds the oracle's (a random
  non-assert live slot; any live slot if all are asserts).
- **kindoracle**: B0's rule (i) `ko`, verbatim (`experiments/b0-ceilings/PREREG.md:95-105`): at a
  full-memory step the score of the slot holding sentence `i` is `m_kind(kind(i))`; the victim is
  the argmin; ties are broken uniformly at random with `random.Random(f"ko:{seed}:{doc_id}:{t}")`,
  `rng.choice(sorted(tied))` over slot indices. No age enters the rule.
- Plus two **reproduction passes** (controls only, never read as results): `fifo` and `oracle`,
  through the same new code path.

**Metric.** All-query held-out answer accuracy (argmax over the vocabulary equals the answer token,
pooled per answer target; run 1 §7.1, A3.2-A3.4), per arm and seed.

**Headline per seed and arm** (arm ∈ {factfiller, kindoracle}): `acc.all(arm) − acc.all(fifo)`,
FIFO from run 2's records; CI from `R1.contrast` with `R1.boot_weights(2048, seed, 2000)`, the
weights run 2's headline used. Labels by run 2's `seed_label`, imported (`experiments/rsr-run2/run.py:206-219`):
- **WIN** if CI lower > 0 and point ≥ +δ_seed;
- **LOSS** if CI upper < 0 and point ≤ −δ_seed;
- **EQUIV** if `−δ_seed < lo` and `hi < δ_seed`;
- **UNRESOLVED** otherwise (and whenever a number or δ is missing).

**Secondary (descriptive).** Per arm: accuracy and contrast on `gap_2_to_M` and `gap_gt_M`;
model-free hit rate (residency of the queried assert at the query step) and `arm − fifo` hit
contrast; age-1 eviction share (run 2's `age1_share`).

**Registered reading rule (PLAN §2, verbatim, then tightened).** *"WIN / LOSS / EQUIV / UNRESOLVED
per seed, using run 2's δ_seed and rule. **Win on all seeds:** the failure is in learning. **Any
LOSS:** reader-side (D2), an owner item."* The rule is applied to **FactFiller**. Tightening: any
other pattern (no LOSS, not WIN on all seeds) reads **INDETERMINATE**: neither learning-side nor
reader-side is supported. The kind-oracle's labels are reported and read the same way, and never
gate.

**Gate P2 (PLAN §2, verbatim; applied mechanically by the PM after 1a is verified).**
> Applied mechanically by the PM after phase 1's 1a is verified. No judgment is involved.
> - **If FactFiller's 1a label is LOSS on any seed:** phase 2 does NOT run. It is reader-side: a
>   content-only ranker cannot be expected to win through this reader. Go straight to Gate A, with
>   the frozen-reader (D2) question for Brendan.
> - **Otherwise:** phase 2a runs.
> - FactFiller is the gate, not the kind-oracle, because the kind-oracle fails seed 1 by
>   construction (assert < query).
> - 1b and 1c do not gate phase 2. They are reported at Gate A whatever they show.

If 1a exits 3, FactFiller has no label, and Gate P2 cannot be applied: phase 2 waits, and the PM
takes it to Gate A as a 1a failure, not as "no LOSS".

**Controls (failure: exit 3, the labels are not read).**

| id | control | expected |
|---|---|---|
| A-C1 | Cdelta: δ_seed read from the B2 ledger equals run 2 PREREG §2.1's table, `==` | true |
| A-C2 | **FIFO reproduced, bit-exact**: the new `fifo` pass's records equal run 2's `fifo.jsonl.gz` record for record (`doc`, every `answers` entry including `ok` and the float `nll`, `hits`, `victims`, `evictions`), all 2048 documents, every seed; and its pooled all-query accuracy equals `runs/rsr-run2/ledger.json` `seed{s}.heldout.accuracy.all.fifo` `point` with `==`. **Checked before any new arm is read**: the harness computes no factfiller or kindoracle statistic until A-C2 has passed for that seed. Tolerance zero. | equal |
| A-C3 | **Reference path reproduced**: the new `oracle` pass equals run 2's `oracle.jsonl.gz` record for record, as A-C2 (it exercises the same `PerRow` reference-policy path the two new arms use); accuracy equals `seed{s}.heldout.accuracy.all.oracle` `point`, `==` | equal |
| A-C4 | **Pairing** (offline-psi C6): each new pass's `(doc, [(t, gap)])` sequence equals run 2's fifo pass for all 2048 documents (length asserted, not a prefix) | equal |
| A-C5 | **Residency** (run 1 C7, `R1.residency_ok`): every new pass's in-loop residency equals the model-free `headroom.simulate` replay of its own victims | 0 mismatched |
| A-C6 | **FactFiller is B0-CF's FactFiller**: the in-loop factfiller hits equal run 2's model-free `factfiller_hits` per document, and the pooled hit rate equals `runs/rsr-run2/ledger.json` `seed{s}.run2.heldout.hit_rate.all.hit` (its `factfiller` point), `==` | equal |
| A-C7 | **Kind means pinned**: the three `m_kind` values read equal the B0 ledger keys above, `==`; their order per seed is ledgered | true |
| A-C8 | **Run 2 sources pinned**: sha256 of every run-2 file read is ledgered at first read and re-checked at the end | unchanged |

**Timing:** placeholder (smoke amendment). Four passes × 2048 documents per seed.

**What 1a does NOT show.**
- Not whether a *learned* head can reach FactFiller; a WIN locates the failure on the learning side
  only in the sense of PLAN's reading rule. It does not show which part of learning (objective,
  censoring, optimizer) failed.
- Not out-of-sample in documents: EVAL is already used. **And its outcome is largely SEEN by
  proxy** (§8): FactFiller through this reader on U, and the kind-oracle through this reader on the
  bake-off's EVAL, have both been measured.
- The kind-oracle reads the generator's kind label and B0's in-sample class means; it is a fixed
  rule, not a member of ψ̂'s class.
- No scaling or cognitive claim.

## 4. Item 1b: linear probes and class-mean rank caps on U

**Inputs.**
- U = `[64, 1088) ∪ [4096, 4160)`, 1088 documents per seed, in lookahead-room's order
  (offline-psi `u_documents`). **ALREADY INSPECTED** (B0, CF, offline-psi, diagnostic A, LS).
- FIFO-world gestalts `s_i`, `c_t`: `.worktrees/offline-psi/runs/offline-psi/seed{s}/gestalts.pt`
  via diagnostic A's `load_inputs`, sha256 pinned as psi-ls-capacity PREREG §1
  (`experiments/psi-ls-capacity/PREREG.md:53-56`).
- Δ: `.worktrees/cf-ceiling/runs/cf-ceiling/raw/B-ckpt3000-seed{s}.delta.pt`, key `Delta`, sha256
  pinned as offline-psi PREREG §1 (`experiments/offline-psi/PREREG.md:44-50`). D.pt files and the
  checkpoints they came from are checked against `~/rsr-substrate/2026-09-27/MANIFEST.sha256`
  (G-C2) at start and end.
- G = `discounted_returns(Δ, 0.9)` (offline-psi `target_of`, imported): C3's γ, uncensored,
  FIFO-world, `k = 0` included, truncated at document end.
- Labels from the generator only (`doc.sentences[i].kind`, `doc.pairs`), never from Δ or a model.

**The split. U-val is spent (psi-ls-capacity F10, `experiments/psi-ls-capacity/RESULTS.md:140`),
so 1b never reads U-val.** U-train = offline-psi's `u_split` train positions (932 documents). In
U-train order, position `j`: **P-test** = `j % 5 == 0`, **P-fit** = the rest. The builder asserts
the sizes and that P-fit, P-test and U-val are pairwise disjoint. Probes are fit on P-fit and
scored on P-test only. No hyperparameter is selected on any split (below).

**Probes** (float64; features standardised per coordinate with P-fit means and sds; class-balanced
cross-entropy; L2 penalty `λ = 1e-4` on weights, not on the bias; full-batch L-BFGS with
strong-Wolfe line search, at most 1000 iterations; **no selection**: `λ ∈ {1e-5, 1e-3}` are
reported descriptively and never replace the registered value):
1. **kind from `s`**: one cell per sentence `i` of each document; label ∈ {assert, query, filler};
   multinomial logistic regression on `s_i` (128 features).
2. **status from `s`**, **from `c`**, and **from `[s; c; s⊗c]`**: cells `(t, i)` with
   `t ∈ [16, 48)` (full-memory decision steps), `i ∈ [t−16, t)` (FIFO-resident), `kind(i) = assert`,
   and query index `q(i) ≠ t`. Label **pending** if `q(i) > t`, **answered** if `q(i) < t`. The
   `q(i) = t` (querying) cells are excluded and counted. Features: `s_i`; `c_t`;
   `[s̃_i; c̃_t; vec(s̃_i c̃_tᵀ)]` (16,640 features, ψ̂'s feature map over the standardised gestalts,
   which spans the same function class as ψ̂ plus a bias), fit without materialising the outer
   product. No age feature anywhere.

**Metric.** Held-out balanced accuracy on P-test (mean over classes of per-class recall). CI: the
per-document bootstrap over P-test documents, `R1.boot_weights(n_Ptest, seed, 2000)`, recomputing
the weighted per-class recalls per replicate.

**Rank caps (per seed, with CIs).**
- Rules (scored by model-free residency, `headroom.simulate`, `M = 16`, B0's procedure):
  - **kind rule** `ko_Δ`: score = `E[G | kind(i)]`;
  - **status rule** `st_Δ`: score = `E[G | class(i, t)]`, class ∈ {assert-pending (`q > t`),
    assert-querying (`q = t`), assert-answered (`q < t`), query, filler}.
  - Class means pooled over B0's rows (`t ∈ [16, 48)`, every `i < t`), row-weighted. Ties: B0's
    A1.10 rule, `random.Random(f"ko:{seed}:{doc_id}:{t}")`.
- **Primary: out-of-sample.** Means estimated on P-fit, rules replayed on P-test.
  `cap = (hit − hit_FIFO) / (hit_factfiller − hit_FIFO)`, all-query; FactFiller seeded
  `random.Random(f"ff:{seed}:{doc_id}")`. Paired CI over P-test, `R1.boot_weights(n_Ptest, seed, 2000)`;
  a replicate with a denominator ≤ 0 is dropped and counted (B0's rule).
- **Secondary: in-sample on all U**, for comparison with B0's `ko` caps on the r_i return,
  printed beside them as **SEEN**: `runs/b0-ceilings/ledger.json` `B.ckpt3000.g09.cap.ko.{point,lo,hi}`.
  The class means themselves (`E[G | kind]`, `E[G | class]` on the Δ return) are reported.

**Registered reading rule (PLAN §2, verbatim, then tightened).** *"Held-out probe balanced accuracy
against a registered threshold (PM proposal: kind ≥ 0.95, status ≥ 0.80). Status caps per seed
with CIs. The `ko` caps are SEEN; label them so."* Thresholds kept at the PM's defaults (no
documented reason was found to change them). Tightened to the CI:
- **kind probe** (from `s`): **DECODABLE** if the CI lower bound ≥ 0.95 on all 3 seeds;
  **NOT-DECODABLE** if the CI upper bound < 0.95 on any seed; **UNRESOLVED** otherwise.
- **status probe**: the same rule at 0.80, applied to the `[s; c; s⊗c]` probe only (ψ̂'s reach).
  The `s`-only and `c`-only status probes are descriptive.
- Caps: reported per seed with CIs, no label. 1b gates nothing; it is reported at Gate A.

**Controls (failure: exit 3).**

| id | control | expected |
|---|---|---|
| B-C1 | Gestalt and Δ files hash to their pins; D.pt/checkpoint manifest per G-C2 | true |
| B-C2 | Split: P-fit, P-test, U-val pairwise disjoint by doc id; no U-val document's cell enters any probe fit or score (asserted on the ids each tensor was built from) | true |
| B-C3 | **G reproduces CF**: `discounted_returns(Δ, 0.9)` replayed by lookahead-room's `TargetPolicy` on the full U gives `runs/cf-ceiling/ledger.json` `B.ckpt3000.cf.g09.cap.point` sample `seed`, `==` (offline-psi C2) | equal |
| B-C4 | **Labels from the generator**: cell counts by kind over U rows equal B0's `B.ckpt3000.g09.E_G.kind.{assert,query,filler}.n` sample `seed`, `==` | equal |
| B-C5 | **Probe fit sanity**: on a planted fixture with a known linear rule, each probe reaches balanced accuracy 1.0 (test); on a label-shuffled copy of P-fit (`torch.Generator().manual_seed(seed)`), the 16,640-feature status probe's P-test balanced accuracy CI contains 0.5 | true |
| B-C6 | Finite: every probe loss, gradient and score is finite; L-BFGS terminates by its tolerance, not by the iteration cap (else the probe is reported "not converged" and its label is UNRESOLVED) | true |

**Timing:** placeholder (smoke amendment).

**What 1b does NOT show.**
- Linear decodability is not use: a decodable status does not show ψ̂ trained under L_MC uses it.
- The caps are model-free residency of hindsight class-mean rules, not accuracy through the reader
  and not a learnable policy's reach.
- Gestalts and Δ are FIFO-world; status decodability under a learned policy's own memory is not
  measured.
- U is already inspected; only P-test is held out from the probe fit, and from nothing else.

## 5. Item 1c: top-k and bottom-k agreement with demand strictly after the decision

**Decision set (primary).** U, 1088 documents per seed (ALREADY INSPECTED), **FIFO's world**: at
step `t` the live slots are the 16 sentences `[t−16, t)`. Decisions: every full-memory step
`t ∈ [16, 47)` (31 per document; `t = 47` has no future step and is excluded). Population
**all-live** (`n_t = 16`). Secondary population **bos-excluded**: the age-1 slot (sentence `t−1`,
E0d's A1.3 exclusion) removed (`n_t = 15`). Secondary step subset: **Q-steps** (a query reads at
`t`), the subset E0d's `bottom1_Q` used.

**Reference: future demand, strictly after `t`.** `F_i(t) = G[t+1][i] = Σ_{k≥0} 0.9^k Δ[t+1+k][i]`,
with G as in 1b (B0-CF's Δ, resample knockout, uncensored, FIFO-world; offline-psi `target_of`).
`Δ[t][i]`, the same-step read, never enters `F_i(t)`.

**Scorers** (each scores the live slots at each decision; higher = more future demand; bottom = what
the scorer would evict):

| scorer | score of slot `i` at `t` | source | role |
|---|---|---|---|
| **ceiling** | `discounted_demand(doc, 0.9)[t+1][i]` (by-construction demand strictly after `t`) | `rsr.data.synthetic` | PLAN's ceiling control ("true_demand's own agreement", shifted to the strictly-future definition) |
| **r_i** | `D[t][i]`: FIFO's gated r_i at step `t` (fill rescale is 1 at full memory) | `.worktrees/lookahead-room/runs/lookahead-room-r2/raw/B-ckpt3000-seed{s}.D.pt`, sha256 per `~/rsr-substrate/2026-09-27/MANIFEST.sha256` | the scorer finding 1 is about |
| **ψC (run 2)** | `ψ̂(s_i, c_t)`, float32 | `.worktrees/rsr-run2-run/runs/rsr-run2/psiC/seed{s}/ckpt-003000.pt` (φ only; sha256 ledgered at first read) | descriptive; trained on POOL, so out-of-fit on all U |
| **offline-psi** | `ψ̂(s_i, c_t)`, float32 | `.worktrees/offline-psi/runs/offline-psi/seed{s}/phi-cf_all.pt`; sha256 = `runs/offline-psi/ledger.json` `seed{s}.selection.cf_all.phi_sha256` | descriptive; reported on U-val (out-of-fit) and U-train (in-sample) separately |
| **LS fit** | `ψ̂(s_i, c_t)`, **float64 only** (F11, `‖W‖ ≈ 2e6`) | `.worktrees/psi-ls-capacity/runs/psi-ls-capacity/seed{s}/phi-cf_all-ls.pt`; sha256 = that dir's `result.json` `arms.cf_all.phi_sha256` | descriptive; U-val and U-train separately |
| **Expire-Span** | `max_l (e_li − age_i)`, the pinned `L48-k1e-2` policy's own eviction score (`ExpireSpanPolicy.select_eviction`), on the FIFO-world gestalts | `~/rsr-substrate/2026-10-07-es-pinned/L48-k1e-2-seed{s}-ckpt-003000.pt` | descriptive; a baseline, so it may read age |
| **age** | `−age_i` (FIFO's order: oldest = bottom) | — | descriptive reference |
| **uniform** | constant | — | chance control |

**Agreement (exact, no RNG).** For a score vector over the `n_t` live slots, let `p_j` be slot `j`'s
probability of being in the bottom-k set under uniform random tie-breaking: 1 if its score is
strictly below the k-th smallest value, `(k − #strictly below)/#tied` if equal to it, 0 otherwise.
Top-k is the same on the negated scores. With `q_j` defined the same way from `F`, the step's
**agreement** is `Σ_j p_j q_j / k`, the expected overlap fraction under independent tie-breaking.
**Chance** at that step is `k / n_t` exactly (a constant scorer reaches it; this resolves e0d
follow-up m-5: 1/16 for all-live, 1/15 for bos-excluded, per step from that step's slot count).
`k ∈ {1, 3}`.

**Pooling and CI.** Per document: the sums over its decisions of agreement and of chance, and the
decision count. Per seed: pooled agreement = Σ agreement / Σ decisions; pooled chance likewise.
CI: `R1.boot_weights(n_docs, seed, 2000)` over the documents of the set, paired across every
scorer, k and population; the statistic read against chance is the CI of **(agreement − chance)**,
pooled per replicate. "CI lower bound above chance" means this CI's lower bound > 0; "CI upper
bound ≤ chance" means its upper bound ≤ 0. (With constant `n_t` this is identical to comparing the
agreement CI with `k/n_t`.)

**Bridge to E0d (descriptive).** On the same U decisions, the E0d-style same-step bottom-1
(`argmin r_i = argmin Δ[t]`, ties as above) is reported for r_i and for `true_demand` beside E0d's
SEEN `bottom1_full` / `bottom1_Q` keys (`runs/e0d/ledger.json`
`seed{s}.analysis.populations.{all,bos_excluded}.stats.{gated,pc}.resample.bottom1_{full,Q}`).
Different documents: printed beside, never contrasted.

**Registered reading rule (PLAN §2, verbatim, then tightened).** *"**Survives** iff, on all 3
seeds: the ceiling control's bottom-k CI lower bound is above chance; `r_i`'s top-k CI lower bound
is above chance; and `r_i`'s bottom-k CI upper bound is ≤ chance. Otherwise it is DROPPED from the
paper."* Tightened: read on the primary set (all-live, all decisions `t ∈ [16, 47)`, γ = 0.9), and
**SURVIVES only if the three conditions hold at k = 1 and at k = 3** on all 3 seeds; otherwise
**DROPPED**. A 1c exit 3 is **DROPPED** (the finding is not supported by a run that did not pass
its controls). The bos-excluded population, the Q-step subset and every non-r_i scorer are reported
and never change the label.

**Controls (failure: exit 3, except where scoped).**

| id | control | expected |
|---|---|---|
| C-C1 | Inputs pinned: gestalts, Δ, D.pt, the three ψ̂ files and the ES checkpoints hash to their pins (ES: `shasum -a 256 -c ~/rsr-substrate/2026-10-07-es-pinned/MANIFEST.sha256` rc 0, start and end) | true |
| C-C2 | FIFO's world: the live set at every decision is `range(t−16, t)`; D.pt and Δ are finite at every `i < t` and NaN at `i ≥ t` (B0 C2, offline-psi C1); 1c reads D.pt on the live cells only | true |
| C-C3 | **r_i is E0d's r_i**: on U's first 32 documents per seed, the gated r_i recomputed with E0d's own measurement path (`experiments/e0d/run.py`) equals `D[t][i]` on the live cells, max |diff| ≤ 1e-6 · max |r| (bit-exact expected) | true |
| C-C4 | **Exactness of the metric**: the uniform scorer's pooled agreement equals pooled chance exactly at every k, population and seed; the reference scored against itself gives 1 | true |
| C-C5 | **ψ̂ heads load into the class**: each φ loads into `BilinearValueHead(d_model=128)` with no age input, and its scores on 64 cells equal the run that produced it (run 2's `ProbePolicy`; offline-psi's and LS's own forward, LS in float64) to `==` | true |
| C-C6 (scoped to the ES row) | The ES checkpoint's transformer and `W_sent` state equal the arm-B substrate bit for bit (so FIFO-world gestalts are its own model's gestalts), and its score's argmin equals `ExpireSpanPolicy.select_eviction` on 64 sampled decisions. On failure the ES row is omitted, printed, and the item's label stands | true |
| C-C7 | Seal: U only; no EVAL, TUNE or EVAL2 id is generated (G-C1) | 0 |

**Timing:** placeholder (smoke amendment). No forward pass except C-C3.

**What 1c does NOT show.**
- `F` is FIFO-world LOO Δ: it is demand under FIFO's memory, with the evicted-reading level offset
  B0-CF reports (`experiments/cf-ceiling/RESULTS.md:131-134`). It is not demand under any scorer's
  own memory.
- Agreement at the top or bottom is not eviction quality; no scorer is deployed here.
- A DROPPED finding is not evidence that attention ranks well; a SURVIVES is one corpus, one width,
  in-sample in documents, and not a claim about real models (phase 3 is not approved).
- The ψ̂ rows are descriptive and on SEEN documents; offline-psi and LS were fit on U-train.

## 6. Item 1d: model-free residency over question-spacing mixes

**Grid.** `geometric_p ∈ {0.10, 0.20, 0.30, 0.50, 0.80}` × `heavy_tail_weight ∈ {0.0, 0.10, 0.20,
0.35, 0.50, 0.70}` (30 points). Every other `SyntheticConfig` field as run 1's corpus
(`SyntheticConfig(sentences_per_document=48, seed=s)`, `experiments/rsr-run1/run.py:1100`); the
base point `(0.30, 0.35)` is the default (`src/rsr/data/synthetic.py:153,156`).

**Documents.** ids `[995000, 997048)` per seed (EVAL, already used), generated under each grid
config. At non-base points they are different documents with the same ids.

**Policies** (model-free, `rsr.metrics.headroom.simulate`, `M = 16`): FIFO; oracle
(`OraclePolicy(discounted_demand(doc, 0.9))`, run 1's γ_o); FactFiller
(`random.Random(f"ff:{seed}:{doc_id}")`).

**Metrics.** Hit rate (all, `gap_2_to_M`, `gap_gt_M`), query count, `fraction_of_pairs_beyond(M)`;
`factfiller − fifo` and `oracle − fifo` hit contrasts with paired CIs (`R1.boot_weights(2048, seed, 2000)`).

**Reading rule (PLAN §2):** *"Descriptive sensitivity table; it gates nothing."*

**Controls.** D-C1 base point reproduces run 2: FIFO and oracle per-document hits equal run 2's
`fifo.jsonl.gz` / `oracle.jsonl.gz` `hits`, and FactFiller's pooled hit equals
`runs/rsr-run2/ledger.json` `seed{s}.run2.heldout.hit_rate.all.hit` (factfiller), `==`. D-C2 every
grid config passes `SyntheticConfig.__post_init__`. D-C3 seal (G-C1), including the modified configs.

**Timing:** placeholder.

**What 1d does NOT show.** Residency, not accuracy: the reader was trained on the base mix only, and
nothing here says what it would read under another mix. Not a claim about natural text.

## 7. Item 1e: group knockout against the sum of single leave-one-out

**Subset (registered).** U documents at U positions `[0, 64)`, i.e. ids `[64, 128)`: exactly
B0-CF's batches 0 and 1 (`BATCH = 32`, consecutive in U order, `experiments/cf-ceiling/run.py:86,135-137`),
so the donors drawn are B0-CF's. Seeds 0, 1, 2. Full-memory steps `t ∈ [16, 48)`, FIFO memory.

**Groups per (document, t)** (resident slots only):
- **G1 duplicate fillers:** every pair of filler slots with identical text;
- **G2 fact pairs:** every (assert, its query sentence) pair with both resident and `q < t` (the
  query sentence carries the answer token: E0d's sink-probe pairs, A2.7 item 8);
- **G3 random pairs:** 8, uniform over unordered pairs;
- **G4 random sets:** 2 of size 4, then 1 of size 8;
- G3 and G4 drawn, in that order, from `random.Random(f"grp:{seed}:{doc_id}:{t}")`.

**Procedure.** `rsr.metrics.loo`, resample mode, single-step, eval mode, `no_grad`. Each member's
replacement is its single-knockout donor (`pick_donor`, `_ROLE_E0D + rank`, B0-CF's seed); a
group's knockout applies `knockout_kv` for every member with those same donors, so
`Δ_G − Σ_{i∈G} Δ_i` contains no donor noise. A group with a member lacking a donor is skipped and
counted.

**Statistics, per group type × {Q-steps, other steps} × seed.** Mean `Δ_G`, mean `Σ Δ_i`, mean
interaction `I = Δ_G − Σ Δ_i`, and the **interaction share** `Σ |I| / Σ |Δ_G|` (ratio of pooled
per-document sums), each with `R1.boot_weights(64, seed, 2000)` CIs.

**Reading rule (PLAN §2):** *"Descriptive. Report the interaction share with CIs."* It gates nothing.

**Controls (failure: exit 3).** E-C1 singles reproduce B0-CF: every recomputed single `Δ_i(t)`
equals `B-ckpt3000-seed{s}.delta.pt` `Delta` on the resident cells, `==`. E-C2 a group of one equals
its single, `==`; untouched slots pass through bit-exactly (`knockout_kv`'s contract). E-C3 every
reported Δ is finite. E-C4 seal (G-C1).

**Timing:** placeholder (expected order: B0-CF measured 1.11 s/doc for singles,
`experiments/cf-ceiling/RESULTS.md:40`; groups add a comparable number of forwards).

**What 1e does NOT show.** Δ is single-step: the effect on sentence `t` only, not on the
trajectory. The generator asserts each fact once, so redundancy among *demanded* slots is limited
to what it creates (G2 pairs carry no future demand). Not a claim about natural text.

## 8. SEEN / UNSEEN

**SEEN (every number this file cites).**
- δ_seed (`experiments/rsr-run2/PREREG.md:97-99`); run 2's headline, FIFO and oracle accuracy, hit
  rates (`experiments/rsr-run2/RESULTS.md`, `runs/rsr-run2/ledger.json`).
- B0: `m_kind`, `ko` caps and every B0 number (`runs/b0-ceilings/ledger.json`).
- B0-CF caps and Δ by class; offline-psi, diagnostic A and LS results (incl. F10, F11); E0d's
  results incl. `bottom1_*` keys; the ES referendum's results.
- **1a's outcome is largely seen by proxy.** FactFiller through this reader on U:
  `experiments/retention-readability/RESULTS.md:88` (`B.ckpt3000.U.H_ff_model` 0.0691 [0.0641,
  0.0739] / 0.0689 [0.0641, 0.0739] / 0.0441 [0.0395, 0.0485]), `:97` (FIFO 0.7703 / 0.7664 /
  0.8037), `:111` (factfiller 0.8394 / 0.8353 / 0.8478). The kind-oracle (random tie) through this
  reader on the bake-off's `[990000, 992048)`: `experiments/newcomer-bakeoff/RESULTS.md:100,123,146`
  (all-query 0.8460 / 0.6314 / 0.8455) against FIFO `:93` (seed 0 0.7746).
- The §1.1 search output (ids only; no document in TUNE or EVAL2 was generated).

**UNSEEN (produced by phase 1).** Every 1a accuracy and label for factfiller and kindoracle on
EVAL; every 1b probe accuracy, class mean of the Δ return and cap; every 1c agreement, chance and
label, and the 1c bridge values on U; every 1d grid value except the base point (which reproduces
SEEN run-2 values by D-C1); every 1e group statistic (singles reproduce SEEN B0-CF cells by E-C1).

## 9. Where this file departs from PLAN §2 or the dispatch (each tighter, or forced)

1. **EVAL2 moved** to `[984000, 986048)`: the PM's candidate collides with generated ids (§1).
2. **1c's r_i source is lookahead-room-r2's D.pt on U, not "E0d's records".** E0d persisted no
   cells: its erratum says the seed cells "were never persisted"
   (`experiments/e0d/RESULTS.md`, Erratum items 1 and 3), and `.worktrees/e0d/runs/e0d/` holds only
   `claims.json`, `ledger.json`, `manifest.json`, `verification.json`. U is the one set where
   r_i, uncensored Δ, and the FIFO-world gestalts every ψ̂ scorer needs all exist and are pinned.
   C-C3 ties this r_i to E0d's instrument; the bridge ties the regime to E0d's SEEN values.
3. **1c tightened:** both k = 1 and k = 3 must hold; exit 3 reads DROPPED.
4. **1b tightened:** probe thresholds read on the CI lower bound, not the point.
5. **1a tightened:** non-WIN, non-LOSS patterns read INDETERMINATE; a 1a exit 3 is not "no LOSS".
6. **1b split:** a registered split of U-train (P-fit / P-test, `j % 5`), because U-val is spent.

## 10. Drafter's expectation (before any phase-1 number; allowed to be wrong)

- 1a: FactFiller WIN on all seeds (the SEEN U numbers above make this the likely outcome), so Gate
  P2 lets phase 2a run; the kind-oracle WIN on seeds 0 and 2 and LOSS on seed 1.
- 1b: kind DECODABLE; status from `[s; c; s⊗c]` UNRESOLVED or NOT-DECODABLE; `st_Δ` cap above
  `ko_Δ` cap on seed 1.
- 1c: DROPPED, most likely because the ceiling control's bottom-k does not clear chance (E0d's
  `true_demand` bottom-1 sat at or below chance).
- 1d: FIFO's hit rate falls as `heavy_tail_weight` rises; FactFiller's gain over FIFO rises.
- 1e: interaction share small for G3/G4; G2 not distinguishable from G3.

## 11. PM review and decisions at commit (2026-10-20, before any phase-1 code or data)

Reviewed in full by the PM (Claude Opus 5.5, the session executing PLAN-2026-10-19). Decisions on
the drafter's open questions. Each one keeps or tightens PLAN §2; none loosens it.

1. **Gate P2 stands as written**, applied mechanically to 1a's FactFiller label on EVAL. Its outcome
   is largely SEEN by proxy (§8: FactFiller through this reader on U; the kind-oracle on the
   bake-off's range). This corrects PLAN §1 / memory `review-2026-10-19`, which said no legal
   content-only rule had been run through the reader. That holds only for EVAL. 1a on EVAL is
   reported as a replication of a SEEN effect on new documents, not as a discovery. The
   correction goes to Brendan in the Gate A packet. It does not change the gate, which he approved
   before any phase-1 data.
2. **1c needs both k = 1 and k = 3**, as drafted (tighter).
3. **EVAL2 = `[984000, 986048)` is accepted.** Beyond the drafter's branch search, the PM also
   searched the gitignored run artefacts on 2026-10-20 (`~/rsr-runs-archive`, sha256-manifested):
   - plain-text files (json/jsonl/log/md/txt/csv) for any integer in `[984000, 986048)` or
     `[997200, 997712)`: no hit;
   - all 21,715 `*.gz` record files for `"doc": <id>` in either range: `hits=0`.
   It is not above every used range. It lies in a gap no artefact or branch has touched, which is
   what sealing requires.
4. **1b probe settings (λ = 1e-4, no selection) and the 1e group definitions are accepted.**
5. **PLAN's bottom-1 figures** (0.043 / 0.028 / 0.024) are E0d's
   `bos_excluded…gated.resample.bottom1_Q`. That ledger states chance as 0.0625 even where 15 slots
   are scored (m-5). Those figures are SEEN and superseded for the paper by 1c's exact per-step
   chance. The Gate A packet says so.
6. **Base head.** Committed on `22662bc` (integration head, full suite
   `passed=2670 failed=0 skipped=0 errors=0` rc 0, read directly). `git diff --stat 654e70c 22662bc`
   touches `scripts/battery_env.py` only, so no input this file names changed.
