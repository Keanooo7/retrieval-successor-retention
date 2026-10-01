---
run_id: rsr-run1
item: "RSR run 1: the first full-model run with a trained psi_hat. Scope: R-2026-09-30-rsr-run-1-scope (N0, D2(a), D2(b), 31(b)) and R-2026-09-30-adr0009-signoff (L1, L2, L6, L8 clip, L9-L13)"
question: >-
  On fresh-stream arm B's ckpt-003000 with the transformer frozen, when psi_hat is trained online by
  L_MC on a fixed pool of 256 documents arm B never saw, how do its evictions and the frozen
  model's held-out answer accuracy and LM loss compare with FIFO and with a learned age-only head,
  arms that differ from it only in the eviction rule?
class: DESCRIPTIVE. No gate and no pass/fail threshold. Controls decide only whether the run is readable.
not: "Not E1 and not E2 (ADR-0009 L6: no slice-1 run is E1). Not evidence on the cognitive claim. No scaling claim."
substrate: "fresh-stream arm B seed 0 ckpt-003000.pt, sha256 0ee3f8a69b507d927c631eb85116e1cd9739ee4472c44eb7f145d739a118da60, weights only, transformer and W_sent frozen"
seeds: "[0] as written; OWNER-OPEN O2 (1 vs 3 seeds)"
arms: {psiC: "RSR, content-only bilinear psi_hat", fifo: "FIFOPolicy", age: "RSR, learned age-only head (psi_override=age_only)"}
pool: "[993000, 993256) of the seed's generator, 256 documents, vocabulary of [0, 64)"
eval: "[995000, 997048) of the seed's generator, 2048 documents, never trained on by any arm"
steps: "3000 optimizer steps per arm; T_warm = 16 steps = 1 epoch = 1/187.5 of total epochs"
overrides: "b_enabled=false, shadow_enabled=false, nu=0, beta=1, gamma=OWNER-OPEN (O1)"
bootstrap: "2000 per-document resamples, paired across every arm, percentile 95% CI, torch.Generator seeded 20260927 + seed (B2 PREREG §9.2's convention, reused)"
falsifier: "NONE gated. Falsifier 1 (vacuity, §2 / §7.1) is described, never tested: no threshold applies to this run"
---

# PREREG: RSR run 1, the first trained ψ̂ on a frozen arm-B substrate

**Written 2026-10-01 by a model** (Claude Opus 5.5, a PREREG-drafting session dispatched by the PM).
**Not written by Brendan.** This file is committed **alone**, before any `run.py` and before any
run-1 step. Amendments are appended, dated, and name what prompted them. The text above them is
never edited.

The base of this commit is `211d6da` (`feat/rsr-run1-ready`, review-fix round in progress). Every
`file:line` below is read at `211d6da`. Items marked **TO-CONFIRM** must be re-checked against the
final head of that branch before the run. Items marked **OWNER-OPEN** are Brendan's. Until they
are ruled, this run does not start.

## 0. Authority, and what this run is not

- **The scope is ruled.** `docs/owner/rulings/R-2026-09-30-rsr-run-1-scope.md` (on `main` at
  `f143ade`). The substrate is arm B `ckpt3000` (D2(a)). The transformer is frozen for run 1
  (D2(b)). One epoch is one pass over the document pool, and `T_warm` is reported as a fraction of
  total epochs (31(b)). `ν` and `β` are set by labelled override, declared here, and nothing is
  recorded into the registry.
- **The build is ruled.** `docs/owner/rulings/R-2026-09-30-adr0009-signoff.md` signs L1, L2, L6,
  the L8 clip clause, and L9–L13. Not signed: L3, L4, L5, L7, the rest of L8, N1, N3, N4.
- **Not E1** (ADR-0009 L6(a): "No slice-1 run is E1"). `RESEARCH-CONTEXT.md` §13.3 ("Do not start
  E1") stands. The E1 baselines (H2O, Expire-Span, leading-edge, LRU) are stubs or absent
  (`src/rsr/baselines/`, `POLICIES = ("fifo", "rsr")` at `loop.py:584`).
- **Not E2.** The spec's E2 gate is "partial ρ < 0.7; content+age ≫ age-only" (spec §6, §7.1).
  Run 1 does not apply it. Its partial ρ is a different operational statistic (§7.4 below). The
  content+age head (spec §6 A3) does not exist.
- **Not evidence on the cognitive claim.** One corpus (synthetic S0-03), `M = 16`, one width
  (`d = 128`). **No scaling claim.**
- **"Verified results, row N"** means a row of `docs/results/2026-09-30-verified.md` on
  `night/2026-09-30` at `94a7ceb` (generated from ledgers; each row names its ledger key).
- **Spec §15 is untouched,** and so are the spec, `docs/spec-corrections.md`, every frozen constant and
  every ruling. No `rsr.constants.record()` call is made.

## 1. Substrate (fixed by D2(a), D2(b))

| item | value | evidence |
|---|---|---|
| checkpoint | `/Users/keanooo7/retrieval-successor-retention/.worktrees/fresh-stream/runs/fresh-stream/B/seed0/ckpt-003000.pt` (mode `-r--r--r--`, 29,550,411 bytes) | `ls -la`, this session |
| sha256 | **`0ee3f8a69b507d927c631eb85116e1cd9739ee4472c44eb7f145d739a118da60`** | computed by this session with `shasum -a 256`, rc 0; equals T0 manifest line 110 |
| T0 manifest | `/Users/keanooo7/rsr-substrate/2026-09-27/MANIFEST.sha256`, sha256 `33dd493d38067e88663b76e9b009e30211c99f75a8ed896a3808486f02fa565f` | line 110: `0ee3f8a6…da60  .worktrees/fresh-stream/runs/fresh-stream/B/seed0/ckpt-003000.pt` |
| manifest root | `/Users/keanooo7/retrieval-successor-retention` | the manifest's paths are relative to it |
| source config | `.worktrees/fresh-stream/runs/fresh-stream/B/seed0/result.json`, sha256 `3e06c0596a5586f605c724278795e66e7c2bfd129478b921a42b02bf124ec3b1` | arm B's checkpoints predate `meta["config"]`. Its config hashes to `71f630b895c717bf`, the suffix of the checkpoint's own `meta.run_id` `fifo-d128-s48-b16-71f630b895c717bf` (bound, `loop.py:416-443`) |
| loaded | `model.state_dict` only (transformer and `W_sent`), strict, from the hashed bytes | `loop.py:446-518`; optimizer, RNG, policy and step are **not** loaded |
| frozen | `requires_grad` off on every model parameter; the model is not in the optimizer; `L_NTP` is reported, not optimised; the model **stays in train mode** (dropout live) in the training loop | `loop.py:1010-1011`, `:1239`, `:1065-1068` (m-4) |
| step counter | `k = 0` at the warm start; §3.4's `T_warm` runs at relative steps `0 … T_warm − 1` | `loop.py:1218-1223` |

## 2. The pool: 256 documents arm B never saw (item 1)

**Choice: pool = ids `[993000, 993256)` of seed 0's generator** (`DocumentPool(offset=993000,
length=256, vocab_documents=64)`). Every id is generated per document with generator seed
`seed · 1_000_003 + id` (`synthetic.py`, prefix-stable). The vocabulary is that of documents
`[0, 64)`.

**What arm B trained on.** Read from arm B's own records (`pool_proof.py` in this session's
scratchpad, rc 0; the output is quoted literally):

```
manifest['stream'] = {"offset": 4160, "stride": 16, "vocab_documents": 64}
manifest['resume_step'] = 1000
manifest['iters'] = 3000
manifest['batch'] = 16
manifest['stream_documents'] = [4160, 52160]
manifest['heldout_documents'] = [4096, 4160]
manifest['probe_documents'] = [0, 64]
manifest['vocab_documents'] = [0, 64]
B/seed0/result.json run_id = fifo-d128-s48-b16-71f630b895c717bf  steps_done = 3000  seed = 0  stream = {"offset": 4160, "stride": 16, "vocab_documents": 64}
B/seed0/result.json resume_check.checkpoint = /Users/keanooo7/retrieval-successor-retention/.worktrees/corpus-curve/runs/corpus-size-curve/n64/seed0/ckpt-001000.pt  stored_step = 1000  ok = True
corpus-size n64/seed0 result.json run_id = fifo-d128-s48-b16-84521ba48ed36e13  n_documents = 64  steps_done = 1000  seed = 0
ledger status = ok  steps_done = 3000  seeds_actually_run = [0, 1, 2]
ledger row stream_disjointness.how = ids of STREAM.window(t, 16), t in [0, 3000), vs PROBE, HELDOUT
ledger row stream_disjointness.value = {"ok": true, "n_ids": 48000, "first_id": 4160, "last_id": 52159, "ids_in_probe": [], "ids_in_vocab_documents": [], "ids_in_heldout": [], "repeats": 0, "first_window_of_B": [20160, 20161, 20162, 20163, 20164, 20165, 20166, 20167, 20168, 20169, 20170, 20171, 20172, 20173, 20174, 20175]}
arm B trained on: [(0, 64), (20160, 52160)]  measured (not trained) on: [(4096, 4160), (0, 64)]
```

So arm B seed 0 trained on two sets of documents:

- **`[0, 64)`** in steps 0–999. This is corpus-size-curve `n64/seed0`: the default path,
  `n_documents = 64`, with-replacement sampling.
- **`[20160, 52160)`** in steps 1000–2999. This is the stream window `4160 + 16·t` for
  `t ∈ [1000, 3000)`. The first window is `first_window_of_B`.

It was measured, but not trained, on `[4096, 4160)` (held-out) and on `[0, 64)` (probe).

**Disjointness.** The pool is disjoint from those ranges. It is also disjoint from every range any
other experiment has used or reserved (`b2-psi-probe/PREREG.md` §3's used-range table, extended
with the later claims):

```
POOL (993000, 993256): disjoint from all used ranges: True  overlaps: []
EVAL (995000, 997048): disjoint from all used ranges: True  overlaps: []
POOL vs EVAL disjoint: True
max id < 1_000_003 (no cross-seed aliasing): True
POOL closure: True  documents=256 first_id=993000 last_id=993255 sentences=12288
EVAL closure: True  documents=2048 first_id=995000 last_id=997047 sentences=98304
```

The ranges checked were:

- arm B's two training ranges;
- fresh-stream's whole stream `[4160, 52160)`;
- H64 `[4096, 4160)`;
- S0-03 held-out `[64, 128)`;
- corpus-size `[0, 4096)`;
- U `[64, 1088)`;
- fresh-escape `[52160, 148160)`;
- B5's continuation cap `[52160, 884160)`;
- E0d `[262144, 263168)` and `[900000, 920000)`;
- B2 `FIT_TRAIN` `[920000, 936000)` (B1's pool `[920000, 920256)` lies inside it);
- B2 `FIT_VAL` `[936000, 937024)` and `EVAL` `[940000, 980000)`;
- the bake-off's `EVAL_NB` `[990000, 992048)`.

A search of every local branch for ids in `[992048, 1000003)` found no claim; the only hits were
float digits in RESULTS tables. `run.py` **re-asserts** both disjointness and closure before any
model is loaded, and either failure is exit 3.

**Why there and not elsewhere.** The ids are fresh to every experiment, not only to arm B, so no
run-1 number is about a document another run has inspected. They are also below `1_000_003`, so a
later seed-1 or seed-2 run cannot alias them (OWNER-OPEN O2).

**Sampling.** Batches are drawn by `torch.randint` over the pool, **with replacement** (m-3,
`loop.py:1083-1086`). One epoch (16 steps) therefore covers about `1 − 1/e` of the pool, not each
document exactly once.

## 3. Vocabulary: matched by the map's hash (item 2)

Arm B records `vocab_documents: 64`. Run 1 matches the **token→id map itself**, as the fixed code
checks it: `vocab_sha256` (`loop.py:338-341`) is the sha256 of the sorted `(token, id)` pairs.
`_load_warm_start` (`loop.py:462-487`) refuses unless the source's map hash equals the run's.

- **The run's map:** `build_vocab` of documents `[0, 64)` at seed 0, `S = 48`
  (`DocumentPool.vocab_documents = 64`).
- **The source's map:** arm B's checkpoint records no `vocab_sha256` (legacy). So it is rebuilt
  from arm B's **hash-bound** config (`result.json`, bound to the checkpoint's `run_id`). This is
  the "legacy source" branch at `loop.py:471-481`.

Measured by this session:

```
run vocab: words = 172  V = 4 + words = 176  sha256 = afbaf994e9d1f617fdc2e68579bf5f55622ee17aee056d33d8667d218d245031
source config hash = 71f630b895c717bf  run_id names 71f630b895c717bf  bound: True
source vocab docs = 64  sha256 = afbaf994e9d1f617fdc2e68579bf5f55622ee17aee056d33d8667d218d245031  equal to run's: True
source tg.V = 176
```

`train()`'s own warm-start record, from this session's dry run of the exact command lines (§9,
`--iters 0`), agrees: `source_vocab_sha256` = `run_vocab_sha256` =
`afbaf994e9d1f617fdc2e68579bf5f55622ee17aee056d33d8667d218d245031`, `vocab_checked: true`,
`seed_matches: true`.

**Expected map hash: `afbaf994…5031`.** Any other value is a refusal and exit 3.
`--allow-unrecorded-vocabulary` and `--allow-seed-mismatch` are **forbidden** in run 1.

## 4. Seed (item 3)

**Seed 0**, arm B's seed 0. The CLI passes `--seed 0`, and `train()` refuses a source whose seed
differs (`loop.py:495-500`; the dry run recorded `source_seed: 0`, `seed_matches: true`). φ's
generator is `phi_generator(0)` (L11, `loop.py:652-663`), with `phi_seed` `4778511090664290076` in
the dry run's frozen config. The data-order generator is `torch.Generator(device).manual_seed(0)`
(`loop.py:919`). It is the same object state in every arm, so all three arms draw the **same
batches at every step**.

How many seeds is **OWNER-OPEN O2** (§12).

## 5. Arms (item 4)

All three arms share:

- the same warm start;
- the same frozen transformer;
- the same pool, seed, data order, dropout masks, optimizer settings and thread count.

They differ **only in the eviction rule** after `T_warm`.

| arm | label (as `train()` stamps it) | eviction | what learns |
|---|---|---|---|
| **psiC** (i) | `rsr[b=off,beta=1,gamma=<O1>,nu=0,shadow=off]+frozen+warm[ckpt-003000.pt@0ee3f8a69b50,src_seed=0]` | `k < 16`: FIFO; `k ≥ 16`: `argmin z(ψ̂_φ(s_i, c_t))` | `φ = {W, u}` (`BilinearValueHead`, content only: `rsr.py:466-473` reads `(s_i, c_t)` and never age) |
| **fifo** (ii) | `fifo+frozen+warm[ckpt-003000.pt@0ee3f8a69b50,src_seed=0]` | FIFO always | nothing (no optimizer is built, `loop.py:1018-1023`) |
| **age** (iii) | `rsr[b=off,beta=1,gamma=<O1>,nu=0,psi=age_only,shadow=off]+frozen+warm[…]` | `k < 16`: FIFO; `k ≥ 16`: `argmin z(θ[min(age, A_max)])` | the age table `θ ∈ ℝ^{49}` (`AgeOnlyValueHead`, zero-initialised, draws no RNG; `A_max = 48` from the registry) |

- **(ii) is §3.7's reduction in behaviour.** It is `FIFOPolicy`, which
  `tests/test_reduction.py::test_fifo_policy_also_reproduces_stock_tg` shows equal to RSR under
  `reduction_to_tg()`.
- **(iii) requires `ν = 0`.** The code refuses `ν ≠ 0` with `psi_override="age_only"`
  (`rsr.py:378-386`).
- **(iii) is not spec §7.1 item 2's test.** That test compares content+age against age-only. Run 1
  compares **content-only** against age-only. The **content+age head (spec §6 A3) is not built**,
  and run 1 does not add it (`value_head.py:176` docstring).
- **Age is excluded from ψ̂** in arm (i), as CLAUDE.md and spec §3.2.2 require. Arm (iii) is a
  separate, labelled arm whose head reads age and nothing else.

## 6. Constants and settings (item 5)

Every number has a source, or is labelled a choice with its reason. **No MEASURED or DERIVED
value is hardcoded.** `ν`, `β` and `γ` are MEASURED, owed by E1 (`constants.py:483-519`). Run 1
reaches them **only** through labelled `--policy-override`s. These are stamped into `frozen`, the
config hash and the arm label (ADR-0009 L12, signed). Fields not overridden are read from the
registry, which raises if they are unmeasured.

| name | value | class / source |
|---|---|---|
| `M` | 16 | FROZEN, scope `synthetic` (spec §5.1; `constants.py:315-335`); `--memory-slots 16`; equals arm B's `max_sentences_in_short_term` |
| `S` | 48 | FROZEN, scope `synthetic` (spec §5.2; `constants.py:336-362`); `--steps-per-stream 48`; arm B's |
| `A_max` | 48 | CONDITIONAL, synthetic, `A_max = S`, non-binding (correction 9; ADR-0004); read from the registry |
| `T_warm` | **16.0 steps** = 1 epoch | `T_warm_epochs` FROZEN 1.0 (spec §3.4; `constants.py:418-428`). Converted to steps by `steps_per_epoch = 256 / 16 = 16` (31(b); `loop.py:623-639`). No `t_warm` override |
| `T_warm` as a fraction of total epochs | **1 / 187.5 = 0.00533** | `total_epochs = 3000 / 16 = 187.5` (31(b); stamped as `frozen["epoch"]`, `loop.py:1079-1092`) |
| `ν` | **0.0** (labelled override, all RSR arms) | Delegated to this PREREG by R-2026-09-30-rsr-run-1-scope. The reasons are below |
| `β` | **1.0** (labelled override) | Delegated likewise. The reasons are below |
| `γ` | **OWNER-OPEN O1** | **Not** delegated by the ruling. See §12 |
| `b` | off: `b_enabled=false`, required | ADR-0009 L12; N0 scope; `ProtectionBias` does not exist (RESEARCH-CONTEXT §12 item 3). E0e's `τ`, `γ_b` are **not read** |
| shadow buffer | off: `shadow_enabled=false`, required | ADR-0009 L6(a) (signed); the code refuses `true` (`loop.py:830-834`). `K` and `λ_shadow` are not read |
| `λ` (return) | 1.0, Monte Carlo | FROZEN (correction 2; `constants.py:395-405`) |
| `L_MC` | unweighted, sum over live slots, then mean over `(row, t)` | ADR-0009 L9 (signed); `rsr.py:685-747` |
| `r_i` | gated (correction 17), share-of-M rescaled (§3.2.1), eval mode (correction 20) | `rsr.py:621-625`; eval capture per L1(a) (signed) |
| `c_t` | the eval-forward `srep`, the same tensor for decision and regression | ADR-0009 L2 (signed); correction 19 |
| batch | 16 | arm B's (fresh-stream manifest `batch`); 31(b)'s own example ("16 at batch 16") |
| total steps | **3000 per arm** | **Choice.** See below |
| `d` | 128 | arm B's (`tg.D`) |
| `max_tokens` | 64 | arm B's (`tg.max_sentence_tokens`) |
| `V` | 176, derived | `4 + 172` words (§3) |
| base learning rate | `1e-3` | arm B's (`lr`, fresh-stream PREREG *Condition*); `train()`'s default |
| φ learning rate | `base_lr / m = 1e-3`, with `m = d / base_width = 128/128 = 1` | μP `value_head` group (spec §4.3; `param_groups.py:101-109`; ADR-0009 T9) |
| age-table learning rate | `base_lr / m = 1e-3` | the same `value_head` group (review P-2 of feat/rsr-run1-ready). This is a code choice, not a spec rule: the table is not a width-dependent parameter |
| optimizer | AdamW, one optimizer, φ's group only (the model is frozen) | as built (`loop.py:1019-1023`). L8's "one AdamW" clause is unsigned, as is its decay clause (O4) |
| AdamW `ε` | `1e-8`, pinned | review P-2; B1's measured regime (`β` LIVE at `ε = 1e-8`, verified results row 25); `ADAMW_EPS`, `loop.py:594` |
| AdamW betas | `(0.9, 0.95)` | `ADAMW_BETAS`, `loop.py:601`, as built |
| φ weight decay | `0.01`, **decoupled** | the brief; `ADAMW_WEIGHT_DECAY`, `loop.py:602-604`, which matches B1's `decoupled_wd` arm. ⚠️ The decay part of ADR-0009 L8 is **unsigned** (§13, BRIEF ERRORS) |
| φ clip | none; φ is out of the transformer clip | ADR-0009 L8 clip clause (signed). The transformer clip is moot under freeze; φ's gradient norm is logged |
| `srep_norm_reg_weight` | `0.01` (the CLI has no flag; arm B trained at `0.0`) | Under freeze it changes **no update** (`objective = None if freeze_transformer`, `loop.py:1239`). It does enter the `loss`/`loss_sum` fields, so **RESULTS reads `loss_lm` and `loss_real_tokens`, never `loss`** (§13) |
| `masked_loss` | true | arm B's; CLI default |
| device | CPU | the reasons are below |
| threads | 1 per arm, identical across arms | measured at this setting (§10); concurrency is **OWNER-OPEN O3** |
| `--beat-every` | 1 | every step on the heartbeat (review MINOR-3; `diagnostics.jsonl` is written every step regardless) |
| `--ckpt-every` | 3000 | one checkpoint, at the end, holding the trained φ (`__head__`, `checkpoint.py:444-445`). The reasons are below |

**Why `ν = 0`.**
1. The age-only arm requires it (`rsr.py:378-386`). With `ν = 0` in both RSR arms, they differ in
   ψ̂ alone.
2. `ν = 0` is the §3.7 reduction value.
3. `RESEARCH-CONTEXT.md` §10.4 settles that `ν` cannot arbitrate within a near-duplicate pair at
   any `ν`. §13.3 says "do not tune `ν`".
4. Whether the term should be clamped is §12 item 4, for E1's `ν` sweep, not here.

**Why `β = 1`.**
1. `β = 0` means `L_MC` is not computed and φ never learns (ADR-0009 L13(a), signed;
   `rsr.py:421-429`).
2. B1 measured `β` LIVE but weak at `ε = 1e-8`. Against `β = 1`, argmin agreement at
   `β ∈ {0.01, 0.1}` was at least 0.9707 at step 3000 on a 256-document pool (verified results
   row 25). So the value moves at most about 3% of decisions in that open-loop regime.
3. `1.0` is the value in §4.5's sweep set `{0.01, 0.1, 1.0}` at which `L_MC`'s gradient is
   unscaled. φ's step is then set by AdamW and the learning rate alone, which is what L7 argues
   is the real knob.

**Why 3000 steps.** This is a choice; the spec and rulings fix none.
1. It matches B1, the only measured φ-training regime on this substrate and pool size (3000 steps,
   256-document pool, about 187 epochs; B1 PREREG).
2. ADR-0009 §3's order-of-magnitude arithmetic says φ must grow `σ_max(W)` from about 2 at init
   toward `G·d` to fit the target scale. At `lr = 1e-3` that takes O(10³) steps, so a run of
   O(10²) steps would test an underfit head (correction 15 is still open).
3. It is arm B's own training length.

The consequence is stated, as §3.4 requires: **`T_warm` is 1 epoch of 187.5.** Under freeze,
§3.4's reason for the warmup ("early sentence representations are largely uninformative") does not
apply. The warmup here only gives ψ̂ 16 passive, FIFO-collected steps.

**Why CPU.**
1. Arm B and every substrate measurement (fresh-stream, B0–B5, E0d, E0e) ran on CPU.
2. B5's verifier showed results are bit-exact only at an equal thread count (5 threads reproduced
   arm B and 4 did not). The paired controls in §8 need bitwise equality across arms.
3. MPS is best-effort (CLAUDE.md).

**Why `--ckpt-every 3000`.**
1. `RSRPolicy.records` keeps every `EvictionRecord` for the whole run: 16 rows × 32 full-memory
   evictions × 3000 steps = 1,536,000 per RSR arm. Each is pickled into every checkpoint
   (`checkpoint.py:422-446`). The default of 25 would write 120 ever-growing checkpoints.
2. A checkpoint is useless for resuming a paired arm anyway (§11.3).

## 7. Outcomes, registered before any data (item 6)

**No threshold exists in the spec or the rulings for this run, so none is invented.** Every
comparison is a paired difference with a 95% CI. RESULTS describes each contrast only as "CI above
0", "CI below 0" or "CI includes 0". It makes no superiority, equivalence or non-inferiority
claim, and it uses no δ.

### 7.1 Primary (held-out, after training)

Measured on **EVAL = `[995000, 997048)`** (2048 documents, fresh, closure proven, §2). The frozen
model is in **eval mode** under `no_grad`. Each arm runs its **final** policy:

- for psiC and age, φ (or θ) is loaded from the arm's `ckpt-003000.pt`;
- `set_train_step(k)` is called with `k = 3000 ≥ T_warm`, so the scored rule runs
  (`rsr.py:508-518`);
- no `observe`, no `L_MC` and no optimizer step are run.

1. **All-query answer accuracy, model-read**, with S0-03's definitions: argmax over the full
   vocabulary equals the answer token, pooled over the queries of the resampled documents. It is
   reported per arm, and per S0-03 bucket: `all`, `gap_eq_1`, `gap_ge_2`, `gap_2_to_M`,
   `gap_gt_M`.
2. **Held-out next-token loss**: real-token NLL (`lm_loss`, masked), per arm. Answer-token NLL
   beside it.

**Contrasts** (per-document paired bootstrap, front matter):

- psiC − fifo;
- age − fifo;
- psiC − age.

Each is reported for accuracy per bucket and for both losses.

### 7.2 Secondary (held-out)

- **R per arm**, fresh-stream's statistic:
  `acc_live(arm, gap_2_to_M) − acc_slots_zeroed(gap_2_to_M)`. The slots-zeroed floor is one
  measurement, because zeroed slots carry no content for any rule to use. The floor is also
  measured under each arm, and any difference is reported.
- **Evaluation-only references.** These are not arms; nothing is trained for them.
  - **oracle**: `OraclePolicy(discounted_demand(doc, γ_o))`, Belady's MIN for any
    `γ_o ∈ (0, 1)` (B2 PREREG §7, "Declared"; TO-CONFIRM T7).
  - **random ×5**: `RandomPolicy`, seeded per (k, seed, doc) as in B2 §7.
- **Oracle regret** (spec §5.5, primary on synthetic): `acc_oracle − acc_arm`. Also the headroom
  share `(acc_arm − acc_fifo) / (acc_oracle − acc_fifo)`, as point and CI.
- **Residency control:** in-loop residency of each queried assert equals a model-free
  `headroom.simulate` replay of the same victim sequence (B2 §9.1).

### 7.3 Training-time, in-loop (the pool; paired per step)

All three arms see identical batches and dropout masks at every step (§4, §8 C2). So per-step
differences are pure eviction effects on the training pool:

- `Δloss_lm(arm, k) = loss_lm(arm, k) − loss_lm(fifo, k)` for psiC and age, over the post-warmup
  steps `k ∈ [16, 3000)`. Reported as the mean, the sd and 500-step block means. **In-sample for
  φ; descriptive only.**
- `L_MC` per step, split at `T_warm` (warm vs post-warm; ADR-0009 §4).
- `mc.g_mean_by_t`: the truncation profile.
- `corr_psi_age` (Pearson, over the regression rows).
- φ's gradient norm.
- `eval_forward_s` (L1's measured cost).

### 7.4 §7.1 diagnostics and recency statistics (logged every step; never a gate)

From `diagnostics.jsonl` (one line per step, `loop.py:1310-1322`) and the run's cumulative
`diagnostics`. Every summary is **split by the `warm` flag**, because the cumulative histogram mixes
warmup (FIFO) and scored evictions.

1. **Victim-age histogram**, per arm, warm vs scored. FIFO's is identically age `M` at full memory.
2. **ADR-0006 displacement** `= victim_rank − 1`: the distribution (not the mean), with
   `fraction_shifting` and `mean_shift`. It is identically 0 under FIFO.
3. **Recency statistics** (spec §7.1, as far as this tree computes them):
   - the fraction of scored evictions that take the oldest slot (`displacement = 0`);
   - raw Spearman ρ(score, age) (`spearman_rho_score_age`);
   - the age-only arm as an explicit arm (item 2's form, with the caveat in §5).
4. **Partial ρ**, `partial_rho_score_age_given_content`, labelled **a diagnostic, never a gate**:
   - the operational statistic is `rsr.metrics.vacuity.partial_correlation_with_age`;
   - Spearman, on ranks, with the **full eviction score** (`z(ψ̂)`, since `b` is off and `ν = 0`);
   - it covers every live slot at every decision of the step;
   - "content" is the **slot gestalts `s_i` only** (128 columns plus an intercept), **linear**
     residualisation (review P-1).
   - It does not partial out `c_t`, and the fit is linear while ψ̂ is bilinear. A score that never
     reads age can still keep a residual age correlation (the `vacuity.py` docstring).
   - Reported per step and summarised over the scored steps (median, IQR).
   - The spec's E2 gate value (0.7, §7.1) concerns a different object, the full score with `b`.
     **No verdict is drawn from any comparison with it.** `None` is reported as `None` (undefined,
     e.g. the age-only head at init, where every score ties).
5. **Leading-edge statistic: not computed.** The spec defines leading-edge as a baseline (§5.4),
   not a statistic. Correction 26 places it on the microstructure coherence graph.
   `src/rsr/baselines/leading_edge.py` is a four-`raise` stub, and ADR-0011 is **PROPOSED**, not
   ratified. RESULTS says so.
6. **LRU (§7.1 item 3): not run.** It is not implemented. RESULTS says so.
7. **Not available, and said so:**
   - attention-share Gini (spec §7.4; the heartbeat's `gini` is `None`);
   - decision attribution in the heartbeat (`attribution` is `None`, §13). With `b` off and
     `ν = 0`, every scored decision is ψ̂'s, and `warm` records which steps were FIFO.

### 7.5 Reporting that spec §3.2.1, §3.4 and N0 require

- `T_warm = 16` steps, 1 epoch of 187.5 (`frozen["epoch"]`).
- The fill fraction `mean(|memory_t| / M)` (§3.2.1, §5.2), computed, not typed.
- **The censored-target caveat**, verbatim in RESULTS (L6, N0):
  > The shadow buffer is off (ADR-0009 L6(a)). Every regression target `G_i(t)` is cut at the
  > slot's eviction. Before `T_warm` the cut is FIFO's, at age `M`, so the target itself is
  > age-shaped although ψ̂ has no age input. After `T_warm` the cut is the learned policy's own,
  > and self-confirming (§3.4, §7.2). Spec §3.4's description of pre-`T_warm` data (spec :280)
  > holds for the policy, not for the target. The rescaled `r_i` is described as rescaled, never
  > as anything stronger (§3.2.1).
- Every number is reported, including the ones that come out wrong (CLAUDE.md).

### 7.6 Author's expectation (before any run-1 number; allowed to be wrong)

These are written for `γ` in `(0, 1)`. If O1 rules `γ = 0`, an amendment restates them before any
run.

- **psiC.** It mostly evicts the oldest slot, though not always:
  - `fraction_shifting` on scored steps is between 0.05 and 0.5;
  - raw ρ(score, age) is clearly negative (below −0.5);
  - partial ρ is small in magnitude (`|ρ| < 0.3`);
  - held-out all-query accuracy is within ±0.01 of fifo, more likely above than below, with the
    CI possibly including 0.

  The reason is the nearest offline analogue, B2's ψ̂-C (the censored target). At γ = 0.9 it was
  EQUIV to FIFO with Δ +0.0035 / +0.0082 / +0.0071 (verified results row 20). Online,
  self-censoring after `T_warm` should push the head further toward recency, not away from it.
- **age.** The table learns `G` decreasing with age, because under FIFO censoring a slot of age
  `a` has at most `M − a` observable terms. So it evicts the oldest slot: `displacement = 0` on at
  least 95% of scored evictions, and accuracy within ±0.005 of fifo.
- **Losses.** Held-out real-token NLL differs across arms by less than 0.01 nats.
- **Controls.** Every control in §8 passes.

## 8. Controls (each failure makes the run inconclusive, exit 3; the numbers are still reported)

| id | control | expected |
|---|---|---|
| C0 | pool and EVAL disjoint from every used range, below `1_000_003`, closed under the `[0, 64)` vocabulary; asserted before any model loads | true (§2) |
| C1 | warm start: sha256 `0ee3f8a6…da60` equals T0 line 110; source config bound to `run_id`; map hash `afbaf994…5031` on both sides; source seed 0 | `train()` refuses otherwise (`WarmStartRefused`, exit 3) |
| C2 | **pairing.** For every step `k`, `n_targets_scored` and `n_answer_targets` are identical across all three arms (same batches). For `k < 16`, `loss_lm` is **bitwise equal** across all three arms (same evictions, same dropout masks) | equal. Already observed at `k = 0` in this session's timing runs: `loss_lm` `1.369111989935239` for fifo, psiC (warmup) and age (zero-init table, every score tied) |
| C3 | **frozen means frozen.** Each arm's final `model.state_dict()` equals `ckpt-003000.pt`'s `model` tensors, `torch.equal` on all 206 | equal |
| C4 | **substrate reproduction.** The FIFO arm's frozen model, measured on H64 `[4096, 4160)` with fresh-stream's own instrument (corpus-size-curve `measure_checkpoint`), reproduces `B.ckpt3000.R_quantity` seed 0 = `0.8597733527421951` (`runs/fresh-stream/ledger.json`) | within 1e-6 (fresh-stream's tolerance) |
| C5 | liveness: `train()`'s own measurement (R-2026-09-22-inert-exit-5) | `live` in every arm (identical, since the model is frozen). The dry run gave `band live`, ratio `50.306607642343124`. `inert` → exit 5 and quarantine per R-2026-09-22-inert-checkpoint-quarantine, and **no** retention number is reported |
| C6 | `steps_done = 3000` in every arm; `status` completed | true |
| C7 | residency equals the model-free replay, every arm, every EVAL document (§7.2) | true |
| C8 | φ is finite: `loss_mc` and `phi_grad_norm` finite at every step | true |

## 9. Commands (item 7)

Run from the worktree of the commit that runs. That commit contains this PREREG and the final
`feat/rsr-run1-ready` fixes. Each arm runs through a lane slot at **1 thread**. Flag names were
**read at `211d6da` and dry-run there** (`--iters 0`, all three rc 0, labels as in §5). They are
**TO-CONFIRM T1** against the final head.

```zsh
R=/Users/keanooo7/retrieval-successor-retention
COMMON=(--d 128 --batch 16 --iters 3000 --steps-per-stream 48 --memory-slots 16
        --lr 0.001 --seed 0 --beat-every 1 --ckpt-every 3000 --device cpu
        --freeze-transformer
        --pool-offset 993000 --pool-length 256 --pool-vocab-documents 64
        --warm-start $R/.worktrees/fresh-stream/runs/fresh-stream/B/seed0/ckpt-003000.pt
        --warm-start-sha256 0ee3f8a69b507d927c631eb85116e1cd9739ee4472c44eb7f145d739a118da60
        --warm-start-manifest /Users/keanooo7/rsr-substrate/2026-09-27/MANIFEST.sha256
        --warm-start-root $R
        --warm-start-source-config $R/.worktrees/fresh-stream/runs/fresh-stream/B/seed0/result.json)
RSR=(--policy rsr --policy-override b_enabled=false --policy-override shadow_enabled=false
     --policy-override nu=0 --policy-override beta=1 --policy-override gamma=<O1>)
SLOT=(env PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run --lane cpu-det --slots 1 --)

# (i) psiC
"${SLOT[@]}" .venv/bin/python -m rsr.train.loop "${COMMON[@]}" "${RSR[@]}" \
    --out-dir runs/rsr-run1/psiC/seed0 > runs/rsr-run1/psiC.log 2>&1; rc=$?
# (ii) fifo
"${SLOT[@]}" .venv/bin/python -m rsr.train.loop "${COMMON[@]}" --policy fifo \
    --out-dir runs/rsr-run1/fifo/seed0 > runs/rsr-run1/fifo.log 2>&1; rc=$?
# (iii) age
"${SLOT[@]}" .venv/bin/python -m rsr.train.loop "${COMMON[@]}" "${RSR[@]}" \
    --policy-override psi_override=age_only \
    --out-dir runs/rsr-run1/age/seed0 > runs/rsr-run1/age.log 2>&1; rc=$?
```

- `<O1>` is replaced by the ruled `γ`. `parse_overrides` reads `nu=0` and `beta=1` as floats
  (`loop.py:1538-1540`).
- Each `rc` is captured directly, never after a pipe.
- **Exit codes:** 0 live, 5 inert, 1 inconclusive liveness, 3 did not run (`loop.py:1429-1442`).
  A non-zero `rc` is reported as itself.
- **Measurement** (§7.1, §7.2, C3, C4, C7) and the ledger come from `experiments/rsr-run1/run.py`.
  This commit does not write it. Its interface, and which existing functions it imports, are
  **TO-CONFIRM T3**. It is reviewed before the run.

## 10. Compute (item 8): measured, then projected

**Measured by this session** at `211d6da`:

- conditions:
  - CPU, `torch.set_num_threads(1)` and `OMP_NUM_THREADS=1`;
  - Apple M4 Max (12P + 4E cores, 64 GB);
  - the step-2 mutation battery (pid 27178, 1 thread) was running, with load average about 4;
  - the real config (§6), `iters = 1`, no checkpoint, output in the scratchpad.
- `γ = 0.9` was a timing placeholder only. `γ` does not change the cost, and it is **not** a
  registered value.
- Scored-step cost was measured by declaring the `t_warm = 0` ablation for one step, timing only.
- `step_s` is from the heartbeat. `wall` is the whole `train()` call, including the warm-start
  hash, vocabulary build, encoding and the end-of-run liveness measurement.

| case | `step_s` | `wall` | log |
|---|---|---|---|
| psiC, warmup step (`k = 0 < 16`) | 2.63 s | 14.19 s (first call, cold) | `time_rsr_warm.log` |
| psiC, scored step | 2.47 s (twice: 2.47, 2.47) | 8.86 s / 9.71 s | `time_rsr_scored.log`, `time_rsr_scored_mem.log` |
| age, scored step | 2.68 s | 9.06 s | `time_age_scored.log` |
| fifo step | 1.53 s | 7.56 s | `time_fifo.log` |

- **The eval forward** (L1(a), measured): 0.77–0.85 s per step, inside `step_s`.
- **Peak RSS**, one psiC scored step: 493,371,392 bytes (`/usr/bin/time -l`).
- **`EvictionRecord` cost**: about 219 bytes each, from 200,000 constructed (`records_mem.log`).
- The logs are in `/private/tmp/claude-501/-Users-keanooo7-retrieval-successor-retention/f5f7e6f8-2469-4fc5-a810-e91d91e81a95/scratchpad/run1-prereg/`.

**Projected (an extrapolation from one step, labelled so):**

| | psiC | age | fifo |
|---|---|---|---|
| time per arm at 3000 steps, 1 thread | ≈ 7,400–7,900 s (≈ 2.1–2.2 h) | ≈ 8,000 s (≈ 2.2 h) | ≈ 4,600 s (≈ 1.3 h) |
| memory | ≈ 0.5 GB plus ≈ 0.34 GB of records by step 3000 | ≈ 0.5 GB plus ≈ 0.34 GB of records by step 3000 | ≈ 0.5 GB |

- With the three arms concurrent at 1 thread each, wall-clock is about 2.2 h of training.
- **Not measured:**
  - whether `step_s` drifts over 3000 steps;
  - the end checkpoint's size and its save time (it pickles about 1.5M records);
  - the cost of the §7 held-out measurement (EVAL × {3 arms, oracle, random ×5, slots-zeroed},
    plus H64 for C4).

  `run.py` times one EVAL batch per arm before the full measurement and reports it (TO-CONFIRM
  T5).

**Coexistence with the battery** is OWNER-OPEN O3 (§12):

- The step-2 battery runs in `.worktrees/_mgr` until about **04:00 Fri 10-02**, at 1 thread
  (`ops/lanes.json` `battery_cpu_slots: 1`).
- Run 1 would use 3 of the `cpu-det` lane's 12 slots alongside it.
- Run 1 never touches `_mgr`.
- The thread count is fixed at 1 per arm in either case, so waiting changes wall time and nothing
  numeric.

## 11. Failure modes, what invalidates the run, and stop rules (item 9)

1. **Collapse onto recency.** psiC evicts the oldest slot almost always (`displacement = 0`
   everywhere, a victim-age histogram equal to FIFO's).
   - **This is a result, not an invalidation and not a stop rule.** It is the most likely failure
     (spec §7.1), and the censored target (§7.5) feeds it.
   - Reported as "psiC behaved as FIFO on N% of scored decisions".
   - Nothing is re-tuned, re-run or relabelled because of it.
2. **Vacuity** (falsifier 1, spec §2 and §7.1; not §12).
   - Described by §7.4's statistics.
   - **No vacuity verdict is issued.** Run 1 is not E2, its partial ρ is not E2's statistic, and
     no ruling supplies a threshold. RESEARCH-CONTEXT §4.6 adds that the target is itself partly
     positional, so a non-zero age correlation is expected.
3. **Crash mid-run.** The heartbeat records the traceback. That arm is `CRASHED`, with its last
   good step.
   - **No resume.** The data-order generator is not checkpointed (fresh-stream BRIEF-ERRORS;
     `checkpoint.py` captures only the global RNG), so a resumed arm would not be paired with
     the others.
   - A re-run starts that arm from step 0. It runs only under an amendment committed before the
     re-run, with the crashed attempt kept on record.
4. **Warm-start mismatch.** Any `WarmStartRefused` (sha, manifest entry, config binding,
   vocabulary map, seed, non-FIFO source), a closure `ValueError` or an override `ValueError`
   means the run **did not run** (exit 3).
   - Never retried with an `--allow-*` flag.
   - Never retried with a different checkpoint, pool, seed or vocabulary.
5. **Non-finite φ** (C8). That arm is stopped and reported as crashed. It does not resume.
6. **Inert liveness** (C5, exit 5): quarantine per ruling, and no retention number is reported.
7. **A control fails** (C0–C8): inconclusive, exit 3. Every number is still in the ledger and in
   RESULTS, marked as not read.
8. **Never, on any outcome:**
   - change `γ`, `β`, `ν`, the step count, the pool, EVAL, the arms or the bootstrap;
   - stop early on an outcome;
   - drop an arm;
   - read a CI as a pass or a fail.

## 12. OWNER-OPEN (nothing runs until these are ruled)

### O1. `γ` for both RSR arms

`γ` is MEASURED, owed by E1 (`constants.py:483-496`). The run-1 scope ruling licenses labelled
overrides for **`ν` and `β` only** ("ν and β: still owed by E1. Run 1 sets them by labelled
override"). RESEARCH-CONTEXT §13.3 lists `γ` among the constants never to be invented. So this
PREREG does not choose it. The evidence:

| option | evidence |
|---|---|
| `γ = 0.9` | The offline evidence on this substrate was measured here: B2's primary γ (ψ̂-C EQUIV to FIFO, ψ̂-U HARMFUL, rows 5–7, 20); B0's `g09` caps (rows 39–40); falsifier 3b names `RSR(γ = 0.9)`. Horizon 10 < `M = 16`. ADR-0009 §3: target scale `~1/(M(1−γ)) ≈ 0.6`, needing `σ_max(W) ≈ 80` against about 2 at init (underfit risk; correction 15 open) |
| `γ = 0` | §4.5 calls it a control arm, "not a sweep value". The target is the present `r_i(t)`, which E0d (AGREE, row 1) certifies against LOO only **at the moment of retrieval** (E0d RESULTS, quoted in the verified results' "What this does NOT show"). B0: an eviction rule on the γ = 0 target sits near FIFO (hit 0.7997 / 0.7881 / 0.8457 against FIFO's 0.8059 / 0.8093 / 0.8088, RESEARCH-CONTEXT §9). E0h's R² of ψ̂(γ=0) on the attention logit is 0.45–0.55, unratified (row 9) |
| `γ = 0.5` or `0.97` | Both are in §4.5's sweep set. Neither has a measurement on this substrate. 0.97 is legal at `A_max = 48` (correction 9), but its target scale is about 2, which is a larger underfit risk |
| context for all options | lookahead-room r2: `room_3b` 0.0355 / 0.0420 / 0.0478, below its 0.05 bar, so lookahead has little room on S0-03 at `M = 16` |

### O2. Number of seeds

| option | consequence |
|---|---|
| 1 (seed 0, as written) | Spread is within-seed only (per-document bootstrap). There is no between-seed sd, and seed 0 alone stands for the substrate |
| 3 (seeds 0, 1, 2) | Arm B's seed-1 and seed-2 `ckpt-003000.pt` are T0-listed (lines 130 and 150: `dadd1e08…3da8`, `b507ebc5…63b8`), and B-series precedent makes "the unit of analysis the seed" (B2 §2). The cost is 3× (§10). The pool and EVAL proofs, closure and map hashes for seeds 1 and 2 would have to be re-run and amended in before the run (the same ids; the reviewer measured the map identical across seeds 0 and 1 at `S = 48`) |

The spec fixes no seed count for this kind of run.

### O3. Run alongside the step-2 battery, or wait for it

| option | consequence |
|---|---|
| alongside | Start once O1 and O2 are ruled. 3 `cpu-det` slots at 1 thread each, next to the battery's 1 thread. The §10 timings were measured under exactly that load |
| wait | Start after `_mgr`'s `battery.rc` exists (about 04:00 Fri 10-02). The same 1-thread setting, so the numbers are identical and only wall time differs |

### O4. φ weight decay

The brief fixes `0.01` decoupled, and the code pins it (`loop.py:602`). But ADR-0009 L8's decay
clause was **not signed** (the sign-off ruling's "Not signed" table: "L7, rest of L8 | wait for B1 (PLAN-v4 §2C)"). ADR-0009
recommended `0.0`. B1 has since reported (`decoupled_wd` LIVE at `ε = 1e-8`), and nobody has ruled.

- The tree offers only `0.01`.
- `0.0` would need a code change, which needs its own review.
- Run 1 is written for `0.01`.

## 13. BRIEF ERRORS and conflicts found while drafting

- **"vacuity (§12)".** Vacuity is falsifier 1 (spec §2), tested in §7.1. Spec §12 is the provenance
  note on the mnemonic-traditions material.
- **"leading-edge statistics as the spec defines them".** The spec defines no such statistic. It
  defines a leading-edge baseline (§5.4), corrected by correction 26 and still a stub, with
  ADR-0011 only proposed. It is not computed (§7.4).
- **"ADR-0009 (on this branch)".** It is not on `feat/rsr-run1-ready` or `211d6da`. It is on
  `docs/adr-learning-path` at `238663a`, still marked PROPOSED in the file. Its sign-off lives in
  the ruling.
- **Files not in the tree.** `reports/RUN1-READY-build.md` is not in the repo. It is at
  `~/Documents/RSR-2026-09-30-day/reports/RUN1-READY-build.md`. `docs/results/2026-09-30-verified.md`
  is on `night/2026-09-30` and `docs/verified-results-0930`, not on this base or `main`. The two
  09-30 rulings are on `main` only.
- **`γ` "(if read)".** It is read (the MC return), and the ruling does not delegate it → O1.
- **"φ weight decay 0.01 decoupled" is given as settled.** Its ADR clause is unsigned → O4.
- **"FIFO (the §3.7 reduction)".** The arm is `FIFOPolicy`, which E0b proves behaviourally equal
  to the reduction. It is not `RSRPolicy` under `reduction_to_tg()`.
- **New spec/corrections conflict found:** none beyond those already recorded (§3.4 :280 vs the
  censored target, and L7 vs :280/:607, both flagged in ADR-0009 and not corrections).

**Code observations at `211d6da`** (for the PM; not fixed here):

- **Heartbeat decision attribution is still `None` for RSR arms.** `loop.py:1324` calls
  `policy.attribution()`, but `RSRPolicy` defines only `attribution_counts()` (`rsr.py:805`). This
  is RESEARCH-CONTEXT §10.1 defect 4, still present. §3.4 requires attribution "from the first
  policy run", and this is that run. Under run 1's terms (`b` off, `ν = 0`) attribution follows
  from `warm`, which `diagnostics.jsonl` records, so run 1 is covered. The defect stands.
- **`RSRPolicy.records` grows without bound**, and the whole list is pickled into every checkpoint
  (`_POLICY_STATE_ATTRS`, `checkpoint.py:422`). That is about 1.5M records per arm here (§6,
  `--ckpt-every`).
- **The data-order generator is not checkpointed**, so `--resume` of a paired arm silently breaks
  the pairing (§11.3).
- **The CLI has no `--srep-norm-reg-weight`.** A warm-started run records `0.01` against the
  source's `0.0`. Under freeze it changes no update, but `loss`/`loss_sum` include
  `0.01 × hinge`. Arm B's last `loss_srep_hinge` was 222.9, i.e. about +2.2 nats in that field.
  So RESULTS reads `loss_lm`.
- **The cumulative `diagnostics` histogram mixes warmup (FIFO) and scored evictions.** The split
  must come from `diagnostics.jsonl` (§7.4).

## 14. TO-CONFIRM against the final head of `feat/rsr-run1-ready`, before the run

- **T1.** The flag names and semantics in §9:
  - `--pool-offset`, `--pool-length`, `--pool-vocab-documents`;
  - `--warm-start`, `--warm-start-sha256`, `--warm-start-manifest`, `--warm-start-root`,
    `--warm-start-source-config`;
  - `--freeze-transformer`, `--policy-override`.

  They are present and dry-run clean at `211d6da`. The fix round is still open (the run1-ready
  worktree had an uncommitted `src/rsr/train/checkpoint.py` edit when this was drafted).
- **T2.** The expected map hash `afbaf994…5031` and the warm-start record are unchanged at the
  final head. The config hash will differ from the dry-run hashes (they were taken at
  `--iters 0`), and it is quoted only from the run.
- **T3.** `experiments/rsr-run1/run.py`, to be written and reviewed. It must provide:
  - the held-out harness: eval mode, `no_grad`, final φ loaded from `__head__`,
    `set_train_step(3000)`, `observe` off;
  - per-document answer vectors for the paired bootstrap;
  - C0, C2, C3, C4 and C7.

  Which existing functions it imports needs confirming, e.g. B2's `run_arm`, `counts` and
  `paired_bootstrap` (`experiments/b2-psi-probe/run.py` on `night/2026-09-30`), and S0-03's
  `BUCKETS`. B2's `run_arm` calls `run_policy_loop` without `eval_capture`. An RSR policy needs
  the eval context, so it cannot be reused unchanged.
- **T4.** The slot wrapper (`orchestrator.slot run --lane cpu-det --slots 1`) sets the job to 1
  thread (`job_threads`, `slot.py:67-75`). Confirm that it is honoured inside `train()`.
- **T5.** The cost of the held-out measurement, timed on one EVAL batch per arm by `run.py`
  before the full measurement.
- **T6.** The end checkpoint (`ckpt-003000.pt` of each RSR arm) holds `__head__`, and it saves
  within memory and disk with about 1.5M records.
- **T7.** `OraclePolicy` decisions are identical for every `γ_o ∈ (0, 1)`, as B2 §7 declares. A
  test in `run.py` checks this before the oracle reference is used.
- **T8.** C4's instrument: corpus-size-curve `measure_checkpoint` accepts the substrate (or the
  FIFO arm's end checkpoint) and is imported unchanged.

## 14a. What this author has already seen (declared)

- **Pool documents.** In the timing runs (§10), one batch of 16 drawn from the pool went through
  the frozen model at `k = 0`, five times. The liveness batch `[993000, 993008)` was run once per
  call. The CLI dry runs (`--iters 0`) ran liveness only.
- **Numbers read:**
  - `loss_lm` 1.369111989935239 (FIFO, and RSR in warmup);
  - 1.3949098984400432 (psiC, scored at an untrained φ);
  - `loss_mc` 1.6278 / 2.0800 / 1.6265 at the `γ = 0.9` placeholder;
  - partial ρ 0.0036 (warmup score) and −0.0695 (scored) at init;
  - liveness `live`, ratio 50.306607642343124.

  None of these is a run-1 result. None is about a trained φ.
- **EVAL documents** were generated as text for the closure check. **No EVAL document has been
  run through any model**, and no number about one exists.

## 14b. What RESULTS.md will contain (item 10)

1. The exact command lines (§9, with `γ` filled), each arm's `rc`, `git_sha`, `run_id`,
   `config_hash` and arm label.
2. Hardware: Apple M4 Max, CPU, threads per arm, torch 2.14.0, Python 3.12.13 (as recorded in the
   heartbeat). Wall-clock and `step_s` per arm.
3. Every control C0–C8 with its literal value.
4. `seeds_actually_run`, `steps_requested`, `steps_done` and status, per arm.
5. Every §7 number per arm, with CIs, **including the ones that come out wrong or against
   expectation**. An sd of exactly zero is flagged as broken, not clean.
6. The §7.5 caveat verbatim; `T_warm` as 1 epoch of 187.5; the fill fraction.
7. A "what this does not show" section: not E1, not E2, not the cognitive claim, no scaling claim,
   one corpus, frozen substrate, shadow off, `b` off. Nothing in it is called "unbiased".
8. The author's expectation (§7.6), against the outcome, without editing either.
