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


---

## AMENDMENT 1 (2026-10-01): adversarial review fixes; three seeds; eval-mode substrate; permutation pass

**Written 2026-10-01 by a model** (Claude Opus 5.5, the PREREG author, on the PM's instruction).
**Not written by Brendan.** Committed alone, before any run-1 step. The text above this heading is
the original PREREG at `7c58777` and is unedited.

**What prompted it.** The review `docs/reviews/prereg-rsr-run1/7c58777ab5ad.md` (commit
`787875b`) found five MAJOR and ten MINOR defects; its verdict was MERGE WITH FIXES. The PM decided
O2 (three seeds), O3 (alongside the battery) and the MAJOR-3 preference. The PM is requesting the
permutation sampler (R-2) from the run1-ready author. The PM's record is that the review found O2
and O3 not owner-reserved, which is the review's *Owner items* (C). The owner is asked only for the
three slots in A1.9.

**Precedence.** Where this amendment and §0–§14b disagree, this amendment governs. Superseded:

- §2's with-replacement sampling paragraph → A1.7;
- §4 and the `seeds` front matter → A1.4;
- §5's "same dropout masks" → A1.3;
- in §6, the rows for total steps (→ A1.9 R-1), `--ckpt-every` (→ A1.8 m-9) and threads (→ A1.6),
  and reason 3 of "Why 3000 steps", which is withdrawn;
- §7.4 items 6 and 7 → A1.1 and A1.8 m-6;
- §7.6's B2 citation → A1.8 m-1;
- §8 C1, C2, C4 and C5 → A1.2–A1.4;
- §9 → A1.2;
- §10's concurrency and threads → A1.6;
- §11 items 3–4 → A1.2;
- §12 O2 and O3 → decided (A1.4, A1.6); O1 and O4 → A1.9;
- §14 → A1.10;
- §14b item 8 → A1.8 m-10.

**Launch preconditions.** All of the following must hold before any run-1 training command:

1. The three owner slots (A1.9) are filled.
2. Code changes C-1 (A1.3) and C-2 (A1.7) are on a reviewed head of `feat/rsr-run1-ready`.
3. `experiments/rsr-run1/run.py` exists, has been reviewed, and its preflight passes (A1.10 T3).
4. The launch is at a clean, committed, pinned head (A1.5).

### A1.1 MAJOR-1: LRU is implemented; it is added as an evaluation-only reference

§7.4 item 6 ("LRU … not run. It is not implemented") is **false and withdrawn**. It must not
reach RESULTS.

- `src/rsr/baselines/lru.py` is present at `211d6da`: 101 lines, `LRUPolicy`, keyed by
  `(row, written_at)` (fix `1ccfc40`). It was run as an evaluation-only arm in the newcomer
  bake-off.
- **What it is in run 1.** LRU is a **no-training reference**, like oracle and random (§7.2). Per
  seed, on that seed's EVAL documents, the frozen model in eval mode runs with `LRUPolicy`.
  - It reads `attn.alpha` in `observe`, so the evaluation loop must run with `observe` on for it
    (A1.10 T11).
  - It gets the residency control C7 like every other arm.
- **Its contrasts are secondary** (A1.4): psiC − LRU and age − LRU, all-query accuracy and per
  bucket. They are reported as spec §7.1 item 3 ("RSR must beat LRU") **descriptively**: no
  threshold and no verdict, the same as every other run-1 contrast.
- **Caveat stated in RESULTS.** `LRUPolicy` falls back to the write step for a never-attended
  slot, so it degrades toward FIFO by design (its docstring). Its distance from fifo is reported
  per seed.

### A1.2 MAJOR-2: exit codes (each one measured) and the launch commands

#### What each exit code means, as measured

Measured by this session at `787875b`:

- CPU, 1 thread, zsh 5.9;
- slot probes ran against an **isolated** orchestrator root (`RSR_ORCH_ROOT` = a scratch copy of
  `ops/lanes.json`), so no real lane lock or job record was touched;
- logs are in `scratchpad/run1-prereg/exitprobe/`.

| id | what was done | rc | heartbeat | slot job record |
|---|---|---|---|---|
| E1 | slot runs a child that exits 5 | **5** | — | written |
| E2 | slot runs a child that exits 1 | **1** | — | written |
| E3 | slot `--pinned-sha 211d6da` from a worktree whose HEAD is `787875b` | **3** (`DID NOT RUN: --pinned-sha …`) | — | **none** |
| E4 | slot `--pinned-sha <HEAD>`, clean tree | **0** | — | written |
| E5 | slot `--pinned-sha <HEAD>`, untracked file outside `runs/` (throwaway repo) | **3** (`… has changes outside runs/`) | — | none |
| E6 | slot on a busy lane, `--wait` at its default 0 | **3** (`not admitted within 0.0s (slots busy)`) | — | none |
| E7 | slot on a busy lane, `--wait 60` | **0**, admitted after 9 s | — | written |
| E8 | zsh `> missing_dir/log` redirection | **1** (zsh: no such file or directory); the command never ran | — | — |
| E9 | slot job exiting 5, backgrounded, collected with `wait $p` | **5** | — | written |
| E10 | `train()` with a wrong `--warm-start-sha256` (`WarmStartRefused`) | **1** | **absent** | (direct run) |
| E11 | `train()` `--policy rsr` without `shadow_enabled=false` (override `ValueError`) | **1** | **absent** | (direct run) |
| E12 | CLI usage error (`--pool-offset` without `--pool-length`), direct | **3** (`DID NOT RUN: rsr-train: …`) | absent | — |
| E12b | the same usage error, through the slot | **3** | absent | **written** (the child ran and exited 3) |
| E13 | a completed `--iters 0` FIFO run through the slot | **0** | footer `completed`, liveness `live` | written, `done` |
| E14 | SIGTERM to the slot wrapper mid-loop (forwarded to the child) | **143** | last line a `beat`; no footer, no crash line | written, `crashed`, rc 143, signal 15 |
| E16 | an exception inside the training loop (a checkpoint path that is a directory) | **1** | `header, beat, crash` (`exc_type: IsADirectoryError`) | (direct run) |
| E17b | zsh: 3 background jobs (exit 5, 0, 7), collected afterwards in reverse order with `wait <pid>` | 7, 0, 5 (each its own) | — | — |
| E17 / E17c | the same with `pid[a]=$!` (E17), and with `pid[$k]=$!` where `k=psiC-s0` (E17c) | **127** "job not found: $!"; the element holds the literal `$!` | — | — |
| E18 | `torch.get_num_threads()` in a child of the slot (`--lane cpu-det --slots 1`), no thread env set by the caller | **1** thread (`OMP/MKL/VECLIB_*` = 1); 12 without the slot | — | — |
| E19 | the launch script below with the O1 slot unfilled | **1** (`GAMMA: O1 unfilled`) | none | none |
| E20 | the launch script from a tree with an untracked file | **3** (`DID NOT RUN: not a clean committed head`) | none | none |

🔴 **E17 is a real trap.** In zsh 5.9, `pid[$k]=$!` stores the **literal string `$!`**. Every
later `wait` then returns 127. The launch script copies `$!` into a scalar first (`p=$!;
pid[$k]=$p`), which is E17b's form.

(E15 was an attempted SIGINT probe. A backgrounded process in a non-interactive zsh ignores
SIGINT, so it ran to completion, 50 frozen FIFO steps, rc 0. It measured nothing about crashes and
is listed only because it ran.)

**The reading rule** for each of the nine training processes. It replaces §9's and §11's exit
mapping.

| rc | job record | heartbeat | meaning |
|---|---|---|---|
| 0 | written, `done` | footer `completed`, liveness `live` | ran, live |
| 1 | written, `done` | **absent** | an exception before the heartbeat: a warm-start refusal, an override or closure `ValueError` (E10, E11). **Did not run** |
| 1 | written, `done` | ends in a `crash` line | an exception mid-loop (E16). **Crashed** |
| 1 | written, `done` | footer `completed`, liveness `inconclusive` | the liveness band was inconclusive (`loop.py:1429-1442`). ⚠️ **Not measured**: it cannot be provoked on a live substrate |
| 5 | written, `done` | footer `completed`, liveness `inert` | INERT (R-2026-09-22-inert-exit-5). ⚠️ **Not measured**, for the same reason |
| 3 | **none** | absent | the slot refused: lane busy past `--wait`, pinned sha mismatch, dirty tree (E3, E5, E6) |
| 3 | written | absent | a CLI usage error inside the child (E12b) |
| 128 + N | written, `crashed`, signal N | no footer | killed by signal N (E14) |

Any `rc ≠ 0` stops that arm's reading. It is reported with its rc, its job record and the last
line of its heartbeat.

#### The launch

The launch script below is the exact text of
`scratchpad/run1-prereg/launch_run1.zsh`.

**Dry run.** It was dry-run at `787875b` against the isolated root, with
`GAMMA=0.9` (a placeholder), `STEPS=0` and `CKPT=1000`:

- rc 0, 9 s wall;
- all nine `rc[…]=0`;
- nine job records `done`, `threads 1`, `pinned_sha 787875b9de92`;
- every log `live`, vocab `afbaf994e9d1`, the labels as in §5.

`C1_FLAGS` and `C2_FLAGS` were empty in the dry run, because those flags do not exist yet
(A1.3, A1.7). Their spellings are TO-CONFIRM T9 and T10.

**The procedure:**

1. `cd /Users/keanooo7/retrieval-successor-retention/.worktrees/rsr-run1-run`. This is a dedicated
   worktree at the final reviewed commit (A1.5), with its own `.venv` (`uv sync --frozen --extra dev`,
   rc 0; `rsr.__file__` under the worktree).
2. **Preflight (C0, review MINOR-3).** Run this and require rc 0:
   `PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run --lane cpu-det --slots 1 --wait 3600 --pinned-sha $(/opt/homebrew/bin/git rev-parse HEAD) -- .venv/bin/python experiments/rsr-run1/run.py --preflight > runs/rsr-run1-preflight.log 2>&1; rc_preflight=$?`
   It re-asserts C0 for all three seeds before any model loads (TO-CONFIRM T3).
3. Launch training. The slots are filled from A1.9 and the C-1/C-2 flags from T9/T10:
   `GAMMA=<O1> STEPS=<R-1> CKPT=<A1.8 m-9> C1_FLAGS='<T9>' C2_FLAGS='<T10>' /bin/zsh launch_run1.zsh > runs/rsr-run1-launch.log 2>&1; rc_launch=$?`
   The script is to be committed with `run.py` (T3), byte-identical to the text below.
4. Measurement (`run.py`, through the slot, 1 thread, A1.6) runs only after every training rc
   is read.

```zsh
#!/bin/zsh
# RSR run 1 launch (PREREG Amendment 1, A1.2). Run with /bin/zsh from the dedicated run
# worktree at the final reviewed commit. Slots: GAMMA (O1), STEPS (R-1), CKPT (derived),
# C1_FLAGS / C2_FLAGS (the flag spellings of code changes C-1 / C-2 at the fixed head).
# OUT defaults to runs/rsr-run1. Every rc is captured directly, one per arm x seed.
R=/Users/keanooo7/retrieval-successor-retention
GIT=/opt/homebrew/bin/git
: ${GAMMA:?O1 unfilled} ${STEPS:?R-1 unfilled} ${CKPT:?ckpt cadence unfilled}
: ${OUT:=runs/rsr-run1}
C1=(${=C1_FLAGS:-}) ; C2=(${=C2_FLAGS:-})

RUN_SHA=$($GIT rev-parse HEAD); rc_sha=$?
PORCELAIN=$($GIT status --porcelain); rc_status=$?
if (( rc_sha != 0 || rc_status != 0 )) || [[ -n $PORCELAIN ]]; then
  print -u2 "DID NOT RUN: not a clean committed head (rc_sha=$rc_sha rc_status=$rc_status)"
  exit 3
fi
mkdir -p $OUT/logs; rc_mkdir=$?
(( rc_mkdir == 0 )) || { print -u2 "DID NOT RUN: mkdir $OUT/logs rc=$rc_mkdir"; exit 3 }
print -r -- "{\"run_sha\": \"$RUN_SHA\", \"dirty\": false, \"porcelain_lines\": 0, \"launched_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" > $OUT/launch-git.json

typeset -A SHA
SHA[0]=0ee3f8a69b507d927c631eb85116e1cd9739ee4472c44eb7f145d739a118da60
SHA[1]=dadd1e08a3849c1211c8479394df4060f91cd2a6e188b97701b783f2068a3da8
SHA[2]=b507ebc573a54316f9d3c20337f49e664bb9c414388af925c5a8e87efd6b63b8
SLOT=(env PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run
      --lane cpu-det --slots 1 --wait 3600 --pinned-sha $RUN_SHA)
RSR=(--policy rsr --policy-override b_enabled=false --policy-override shadow_enabled=false
     --policy-override nu=0 --policy-override beta=1 --policy-override gamma=$GAMMA)

typeset -A pid rcs
for s in 0 1 2; do
  B=$R/.worktrees/fresh-stream/runs/fresh-stream/B/seed$s
  for a in psiC fifo age; do
    case $a in
      psiC) P=("${RSR[@]}") ;;
      fifo) P=(--policy fifo) ;;
      age)  P=("${RSR[@]}" --policy-override psi_override=age_only) ;;
    esac
    k=$a-s$s
    "${SLOT[@]}" --job-id rsr-run1-$k${JOB_SUFFIX:-} -- \
      .venv/bin/python -m rsr.train.loop \
        --d 128 --batch 16 --iters $STEPS --steps-per-stream 48 --memory-slots 16 \
        --lr 0.001 --seed $s --beat-every 1 --ckpt-every $CKPT --device cpu \
        --freeze-transformer "${C1[@]}" \
        --pool-offset 993000 --pool-length 256 --pool-vocab-documents 64 "${C2[@]}" \
        --warm-start $B/ckpt-003000.pt --warm-start-sha256 ${SHA[$s]} \
        --warm-start-manifest /Users/keanooo7/rsr-substrate/2026-09-27/MANIFEST.sha256 \
        --warm-start-root $R --warm-start-source-config $B/result.json \
        "${P[@]}" --out-dir $OUT/$a/seed$s \
      > $OUT/logs/$k.log 2>&1 &
    p=$!; pid[$k]=$p      # measured: pid[$k]=$! stores the literal string "$!" in zsh 5.9
  done
done
for k in ${(ok)pid}; do
  wait ${pid[$k]}; rcs[$k]=$?
  print -r -- ${rcs[$k]} > $OUT/logs/$k.rc
  print -r -- "rc[$k]=${rcs[$k]}"
done
```

At launch, `JOB_SUFFIX` (a dry-run hook) and `OUT` must be **unset**, so the job ids are `rsr-run1-<arm>-s<seed>` and the output goes to `runs/rsr-run1`.

**How the script handles each launch defect:**

- `mkdir -p` runs before any redirection into `$OUT/logs` (E8).
- `--wait 3600` is set (E6, E7).
- The job ids are fixed, `rsr-run1-<arm>-s<seed>`. A second launch with the same ids is refused
  (rc 3, no record) rather than silently overwriting a record.
- There is one rc per arm × seed: `rcs[psiC-s0]` … `rcs[age-s2]`, each also written to
  `logs/<arm>-s<seed>.rc`.
- The nine processes run **concurrently** (review MINOR-5).

### A1.3 MAJOR-3: the train/eval memory shift is ELIMINATED, which needs code change C-1

**The choice: eliminate it.** The frozen transformer runs in **eval mode (dropout off) in every
arm**, during training and during evaluation alike.

**Why eliminate rather than declare:**

1. The shift falls on psiC alone. psiC is the only arm whose rule reads `s_i`, so a declared
   confound would sit inside the headline contrast (A1.4). Removing it is cheaper than
   interpreting it.
2. A frozen transformer gains nothing from dropout. No transformer weight is updated (D2(b)), so
   dropout's regularising role has nothing to act on. All it adds is noise in the gestalts the
   policy reads, regresses on and decides on. That is "noise in a policy-relevant direction",
   correction 20's and ADR-0009 L2's own argument, applied to `s_i`.
3. Every measurement that selected this substrate read it in eval mode. That includes
   fresh-stream's R (C4), B2, the bake-off and E0e. Training-time memory then matches both the
   held-out evaluation and the substrate's measured regime.
4. Pairing becomes trivial. No dropout mask is drawn, so the arms share no RNG stream that could
   diverge.

**A reading of correction 20, flagged for the PM.** Correction 20 says "eval mode for the
retention target; train mode for the LM loss". This amendment reads "train mode for the LM loss"
as applying to a loss that is **optimised**. Under D2(b), `L_NTP` is reported and not optimised.

If the PM or the owner reads correction 20 as requiring train mode for the reported `L_NTP` even
under freeze, this choice is void. The fallback would be the declared-confound route, by a further
amendment. It is **not** pre-authorised here. This is not a new spec/corrections conflict; it is a
reading of how a correction applies to a configuration it did not contemplate.

**The code cannot do this today.** I measured it at `211d6da`:

- `train()` builds the model in its default train mode;
- `--freeze-transformer` removes only the update (`loop.py:1010-1011`), and records
  `frozen_model_mode = "train (dropout live)"` (`:1065-1068`, review m-4);
- there is no flag for the mode, and no CLI flag for any dropout rate;
- a grep of `loop.py`, `policy_loop.py` and `model.py` finds no other `.train(` call. The one
  mode switch is `eval_forward`'s, which restores the **prior** mode (`policy_loop.py:363-374`).

So a model put in eval mode once would stay in eval mode for every training forward.

**Code change C-1, for the run1-ready author:**

1. Add `train(..., frozen_model_mode: str = "train")` with the values `{"train", "eval"}`, and the
   CLI `--frozen-model-mode {train,eval}`.
   - `"eval"` without `freeze_transformer=True` is **refused**: it would silently remove dropout
     from an optimised `L_NTP`.
2. After `model.requires_grad_(False)` (`loop.py:1011`), when `"eval"`: call `model.eval()`.
   - It must come **after** the liveness decoy copy and the warm-start load, and before the loop.
   - Neither reads the mode, so the order is for clarity only.
3. Record it in `frozen["frozen_model_mode"]`: `"eval (dropout off)"` or `"train (dropout live)"`.
   That puts it in the config hash. The arm label must show it too (e.g. `+frozen[eval]`; spelling
   TO-CONFIRM).
4. Do **not** remove or change L1(a)'s eval forward. It becomes numerically redundant, but it stays
   the signed path and the r_i source.
5. Tests:
   - (a) frozen eval mode: at a step, the training forward's `srep` equals the eval forward's
     `srep` bitwise;
   - (b) the global RNG state is unchanged across one full training step;
   - (c) `model.training` is False at loop end;
   - (d) `"eval"` without freeze is refused;
   - (e) the mode is in `frozen`, the hash and the label;
   - (f) m-4's `test_frozen_mode_records_that_dropout_stays_on` is updated to cover both values.

   Each test needs a battery mutation that reddens it.

**Consequences for this PREREG:**

- §5's "same dropout masks" becomes "no dropout".
- C2's warmup clause is unchanged: `loss_lm` is bitwise equal for `k < 16`.
- In-loop `loss_lm` levels are eval-mode values. They are not comparable to arm B's train-mode
  heartbeat losses, and RESULTS says so.
- C5's liveness: the band is expected `live`. The ratio is reported per arm and is not a control.
  (Measured: liveness consumes the global RNG stream. A FIFO run gave ratio 50.306607642343124 at
  `--iters 0` and 50.30659070700227 after 50 train-mode steps. Under C-1 no step draws RNG.)

### A1.4 MAJOR-4: seeds 0, 1 and 2; one headline contrast; scoped CIs

**Seeds: 0, 1 and 2** (the PM's decision). B2 showed between-seed spread larger than within-seed
CIs (verified results row 6: −0.0784 [−0.0847, −0.0716] on seed 0 against −0.0202
[−0.0254, −0.0147] on seed 2).

- Each seed `s` uses arm B seed `s`'s `ckpt-003000.pt` and its own `--seed s`.
- The pool ids `[993000, 993256)` and the EVAL ids `[995000, 997048)` are the same for every
  seed. They are drawn from **seed s's generator** (generator seed `s·1_000_003 + id`), so each
  seed has its own documents.
- No (seed, id) pair here can share a generator seed with another seed's documents:
  `s·1_000_003 + x = s'·1_000_003 + y` with `x, y < 1_000_003` forces `s = s'` and `x = y`. The
  review measured this as well.

**The substrate for each seed.** `shasum -a 256` gave rc 0. The T0 manifest lines and file sizes
are from `ls -la`.

| seed | checkpoint sha256 | T0 line | size (bytes) | `result.json` sha256 | config hash = `run_id` suffix |
|---|---|---|---|---|---|
| 0 | `0ee3f8a69b507d927c631eb85116e1cd9739ee4472c44eb7f145d739a118da60` | 110 | 29,550,411 | `3e06c0596a5586f605c724278795e66e7c2bfd129478b921a42b02bf124ec3b1` | `71f630b895c717bf` |
| 1 | `dadd1e08a3849c1211c8479394df4060f91cd2a6e188b97701b783f2068a3da8` | 130 | 29,550,539 | `9c545bb125a2fa3142ac7426b8e2bc2ca35c1c94aabb69a0ba18d78ce1777c2f` | `9982c6a01218e069` |
| 2 | `b507ebc573a54316f9d3c20337f49e664bb9c414388af925c5a8e87efd6b63b8` | 150 | 29,550,411 | `83383192089ec63d9f275e2ea2507f5fc114cfbcf8cc5ca129b13417f3bc32b4` | `8bdc9e0715c147d9` |

**The pool-unseen, closure and vocabulary proofs for seeds 1 and 2.**
`pool_proof_seed.py <s>` is §2's script, parameterised by seed. It ran with rc 0 for each seed.
Its seed-0 output is identical to §2's (`diff` rc 0). The output, literally:

```
SEED = 1
manifest['stream'] = {"offset": 4160, "stride": 16, "vocab_documents": 64}
manifest['resume_step'] = 1000
manifest['iters'] = 3000
manifest['batch'] = 16
manifest['seeds'] = [0, 1, 2]
manifest['stream_documents'] = [4160, 52160]
manifest['heldout_documents'] = [4096, 4160]
manifest['probe_documents'] = [0, 64]
manifest['vocab_documents'] = [0, 64]
B/seed1/result.json run_id = fifo-d128-s48-b16-9982c6a01218e069  steps_done = 3000  seed = 1  stream = {"offset": 4160, "stride": 16, "vocab_documents": 64}
B/seed1/result.json resume_check.checkpoint = /Users/keanooo7/retrieval-successor-retention/.worktrees/corpus-curve/runs/corpus-size-curve/n64/seed1/ckpt-001000.pt  stored_step = 1000  ok = True
corpus-size n64/seed1 result.json run_id = fifo-d128-s48-b16-f4803595bee3bc51  n_documents = 64  steps_done = 1000  seed = 1
ledger status = ok  steps_done = 3000  seeds_actually_run = [0, 1, 2]
ledger row stream_disjointness.how = ids of STREAM.window(t, 16), t in [0, 3000), vs PROBE, HELDOUT
ledger row stream_disjointness.value = {"ok": true, "n_ids": 48000, "first_id": 4160, "last_id": 52159, "ids_in_probe": [], "ids_in_vocab_documents": [], "ids_in_heldout": [], "repeats": 0, "first_window_of_B": [20160, 20161, 20162, 20163, 20164, 20165, 20166, 20167, 20168, 20169, 20170, 20171, 20172, 20173, 20174, 20175]}
arm B trained on: [(0, 64), (20160, 52160)]  measured (not trained) on: [(4096, 4160), (0, 64)]
POOL (993000, 993256): disjoint from all used ranges: True  overlaps: []
EVAL (995000, 997048): disjoint from all used ranges: True  overlaps: []
POOL vs EVAL disjoint: True
max id < 1_000_003 (no cross-seed aliasing): True
run vocab: words = 172  V = 4 + words = 176  sha256 = afbaf994e9d1f617fdc2e68579bf5f55622ee17aee056d33d8667d218d245031
source config hash = 9982c6a01218e069  run_id names 9982c6a01218e069  bound: True
source vocab docs = 64  sha256 = afbaf994e9d1f617fdc2e68579bf5f55622ee17aee056d33d8667d218d245031  equal to run's: True
source tg.V = 176
POOL closure: True  documents=256 first_id=993000 last_id=993255 sentences=12288
EVAL closure: True  documents=2048 first_id=995000 last_id=997047 sentences=98304

SEED = 2
(manifest lines identical to seed 1's)
B/seed2/result.json run_id = fifo-d128-s48-b16-8bdc9e0715c147d9  steps_done = 3000  seed = 2  stream = {"offset": 4160, "stride": 16, "vocab_documents": 64}
B/seed2/result.json resume_check.checkpoint = /Users/keanooo7/retrieval-successor-retention/.worktrees/corpus-curve/runs/corpus-size-curve/n64/seed2/ckpt-001000.pt  stored_step = 1000  ok = True
corpus-size n64/seed2 result.json run_id = fifo-d128-s48-b16-6d107783fdf56a84  n_documents = 64  steps_done = 1000  seed = 2
ledger status = ok  steps_done = 3000  seeds_actually_run = [0, 1, 2]
(ledger stream_disjointness rows identical to seed 1's)
arm B trained on: [(0, 64), (20160, 52160)]  measured (not trained) on: [(4096, 4160), (0, 64)]
POOL (993000, 993256): disjoint from all used ranges: True  overlaps: []
EVAL (995000, 997048): disjoint from all used ranges: True  overlaps: []
POOL vs EVAL disjoint: True
max id < 1_000_003 (no cross-seed aliasing): True
run vocab: words = 172  V = 4 + words = 176  sha256 = afbaf994e9d1f617fdc2e68579bf5f55622ee17aee056d33d8667d218d245031
source config hash = 8bdc9e0715c147d9  run_id names 8bdc9e0715c147d9  bound: True
source vocab docs = 64  sha256 = afbaf994e9d1f617fdc2e68579bf5f55622ee17aee056d33d8667d218d245031  equal to run's: True
source tg.V = 176
POOL closure: True  documents=256 first_id=993000 last_id=993255 sentences=12288
EVAL closure: True  documents=2048 first_id=995000 last_id=997047 sentences=98304
```

(Seed 2's elided lines are byte-identical to seed 1's. The full log is
`scratchpad/run1-prereg/pool_proof_seed2.log`.)

**`train()`'s own warm-start check for seeds 1 and 2.** Each was run through the CLI at
`--iters 0`, for fifo and psiC, with rc 0 for all four. The seed-1 and seed-2 lines of the
nine-arm launch dry run (A1.2) agree.

```
seed 1 fifo fifo+frozen+warm[ckpt-003000.pt@dadd1e08a384,src_seed=1]
   manifest_entry .worktrees/fresh-stream/runs/fresh-stream/B/seed1/ckpt-003000.pt  sha256 dadd1e08a3849c1211c8479394df4060f91cd2a6e188b97701b783f2068a3da8
   source_run_id fifo-d128-s48-b16-9982c6a01218e069  source_seed 1  source_vocab_sha256 = run_vocab_sha256 = afbaf994e9d1f617fdc2e68579bf5f55622ee17aee056d33d8667d218d245031  vocab_checked True  seed_matches True
   liveness live 86.69945157035805 exit 0; psiC phi_seed 4643668418649829614
seed 2 fifo fifo+frozen+warm[ckpt-003000.pt@b507ebc573a5,src_seed=2]
   manifest_entry .worktrees/fresh-stream/runs/fresh-stream/B/seed2/ckpt-003000.pt  sha256 b507ebc573a54316f9d3c20337f49e664bb9c414388af925c5a8e87efd6b63b8
   source_run_id fifo-d128-s48-b16-8bdc9e0715c147d9  source_seed 2  source_vocab_sha256 = run_vocab_sha256 = afbaf994e9d1f617fdc2e68579bf5f55622ee17aee056d33d8667d218d245031  vocab_checked True  seed_matches True
   liveness live 77.85591251012764 exit 0; psiC phi_seed 4250115863317958758
```

**C4, per seed.** The reference is read from the ledger, not typed. The instrument is
corpus-size-curve's `S003.measure` with `doc_sets(s, 64)`, run at **1 thread**.

| seed | reference `B.ckpt3000.R_quantity` | measured R | abs diff | who measured |
|---|---|---|---|---|
| 0 | 0.8597733527421951 | 0.8597733527421951 | 0.0 | the reviewer, then the PM, 22.5 s, rc 0 |
| 1 | 0.8531856164336205 | 0.8531856164336205 | 0.0 | this session (`c4_probe_seed.py 1`), 21.9 s, rc 0 |
| 2 | 0.879943523555994 | 0.879943523555994 | 0.0 | this session (`c4_probe_seed.py 2`), 22.9 s, rc 0 |

These numbers were measured on the **source** checkpoints. During the run, C4 is re-measured on
each seed's FIFO arm at its end checkpoint (T8), with tolerance 1e-6.

**THE headline contrast.** There is exactly one: **psiC − fifo, held-out all-query answer
accuracy** (model-read, S0-03's definition, §7.1 item 1). It is read on EVAL at the final step,
**per seed**.

Why this contrast:

1. Run 1 asks whether a trained content ψ̂ changes what the frozen model retrieves, relative to
   the rule it replaces. FIFO is that rule (CLAUDE.md: RSR "replaces that one line").
2. The substrate was chosen for retrieval under FIFO (D2(a)), so answer accuracy is the quantity
   it is known to carry.
3. All-query nets the two directions an eviction rule can move. Departing from FIFO can lose a
   fact FIFO would have kept (`gap ≤ M`), or keep one FIFO would have dropped (`gap > M`).
   Choosing a single bucket would select a direction.
4. It is the measure B2's offline ψ̂-C used (row 7), so the online result can be read beside the
   offline one.
5. psiC − age is the §7.1-item-2-style comparison, and psiC − LRU is item 3's. They answer
   different questions, and the run cannot have three headlines.

**How the headline is reported:**

- **Per seed:** the point estimate, with a paired per-document bootstrap 95% CI (2000 resamples,
  generator seeded `20260927 + seed`).
- **Across seeds:** the three per-seed point estimates listed, their mean and sd (n = 3), and the
  number of seeds whose point estimate is above 0, below 0 or exactly 0.
- There is no CI across seeds, no test, and no pass or fail.

**Mandatory scope sentence, verbatim in RESULTS, beside every within-seed CI:**

> This interval resamples EVAL documents only. It holds fixed the trained φ (one φ
> initialisation, one data order), the substrate seed and the pool. It is a within-seed interval
> and does not estimate run-to-run or seed-to-seed variation, which only the three per-seed
> values show.

**Everything else is labelled, in RESULTS, as one of:**

- **SECONDARY.** Every other §7.1 and §7.2 contrast and bucket: the losses, age − fifo, psiC −
  age, psiC − LRU, age − LRU, R, oracle regret, headroom share, and the §7.3 in-loop Δloss.
- **DIAGNOSTIC.** The §7.4 statistics, the intermediate-snapshot readings (A1.8 m-9), `L_MC`,
  `g_mean_by_t`, `corr_psi_age`, φ's gradient norm and `eval_forward_s`.

Secondary and diagnostic intervals are **unadjusted for multiplicity**, and RESULTS says so in one
sentence. The C2 batch check stays a control.

### A1.5 MAJOR-5: launch only at a clean, committed, pinned head

- **Where.** The run launches from a **dedicated** worktree, `.worktrees/rsr-run1-run`. Its HEAD
  is the final reviewed commit, which contains this PREREG, C-1, C-2, `run.py` and the launch
  script. It is never launched from `.worktrees/rsr-run1-ready` or any worktree where proofs or
  mutations run.
- **Checks.** The launch script refuses unless `git rev-parse HEAD` and `git status --porcelain`
  both succeed **and** the porcelain is empty (exit 3, E20).
  - Every slot call carries `--pinned-sha $RUN_SHA`.
  - The slot refuses on a HEAD mismatch (E3) and on any change outside `runs/` (E5). It exits 3
    and writes no job record.
- **RESULTS records:**
  - `run_sha` and `dirty: false` (from `runs/rsr-run1/launch-git.json`, written before any slot
    call);
  - each job record's `pinned_sha`;
  - each heartbeat's `git_sha`.

  All three must agree.
- **Known gap.** `_sha()` (`loop.py:81-95`) records HEAD only and has no dirty flag. The
  pinned-sha refusal is what makes `git_sha` trustworthy here.

### A1.6 O3 (PM's decision): alongside the step-2 battery; the thread budget

Run 1 runs **alongside** the step-2 battery. It never touches `.worktrees/_mgr`.

**Thread budget, all at 1 thread per process:**

- **Training:** 9 processes (3 arms × 3 seeds), each holding one `cpu-det` slot. The slot forces
  1 thread (E18).
- **The battery:** 1 thread, holding the `battery` lane plus 1 `cpu-det` slot (`ops/lanes.json`:
  `battery_cpu_slots: 1`).
- **Total during training:** 10 single-threaded processes on the M4 Max's 12P + 4E cores. That is
  10 of the 12 `cpu-det` slots in use.
- **Measurement:** after every training rc is read, `run.py` measures in **3 processes**, one per
  seed, through the slot at 1 thread each (TO-CONFIRM T3). Each covers that seed's arms and the
  references: oracle, random ×5, **LRU**, slots-zeroed, and C4 on H64.
- **Memory, extrapolated:** about 6 × 0.85 GB (RSR, records included) plus 3 × 0.5 GB (FIFO),
  which is about 6.6 GB. Each one-third snapshot adds about 35–105 MB of pickled records
  (the review measured 20.6 MB for 300,000 records and extrapolated about 105 MB at 1.536M).

**Timing caveat**, stated in RESULTS. Every `step_s` in §10 was measured with **one** run-1 process
next to the battery (load average about 4). Per-step cost with 10 concurrent single-threaded
processes, some of which may land on efficiency cores, is **not measured**. The §10 projections
are a lower bound on wall time. RESULTS reports each process's measured `step_s` and wall-clock.

### A1.7 R-2: one pass = every pool document exactly once per epoch; needs code change C-2

The ruling reads "One pass over the document pool: 256 documents seen". The built sampler draws
with replacement, so one epoch sees about **162.0** distinct documents
(`256·(1 − (255/256)^256)` = 162.007).

**This amendment registers a permutation pass:**

- At the start of every epoch `e` (when `k mod 16 == 0`), the data-order generator `gen` draws
  `π_e = randperm(256)`. `gen` is seeded with `seed` (`loop.py:919`) and drawn from by nothing
  else, so the order is shared across arms.
- Step `k` trains on pool positions `π_e[16·(k mod 16) : 16·(k mod 16) + 16]`.
- So every full epoch visits all 256 documents exactly once, in an order identical across the
  three arms of a seed.
- `T_warm` (16 steps) is exactly epoch 0, which visits every pool document once under FIFO.
- If R-1 is not a multiple of 16, the last epoch is partial, and RESULTS states how many documents
  it visited. (At 3000 steps that is 187 full epochs plus 8 steps covering 128 documents.)

**Code change C-2, for the run1-ready author (TO-CONFIRM T10):**

1. In pool mode, replace `torch.randint` with the per-epoch permutation above. Refuse when
   `pool.length % batch != 0`.
2. Record `frozen["epoch"]["sampling"] = "per-epoch permutation (each document once per epoch)"`,
   so the hash changes.
3. **Log the step's selected pool ids every step**, or their sha256, in the heartbeat or
   `diagnostics.jsonl`.
4. Tests:
   - each epoch's ids equal the pool exactly once;
   - two arms at one seed draw identical ids at every step;
   - an RSR arm and a FIFO arm still see the same first batch (the existing L11 test);
   - with-replacement remains reachable only outside pool mode.

**If C-2 is not delivered, the run does not start.** Reverting to with-replacement sampling needs
a further amendment that states the 162-of-256 reading.

### A1.8 The ten MINORs

- **m-1 (B2 citation).** §7.6 cited "verified results row 20" for B2's ψ̂-C. Row 20 is the
  bake-off's. B2's ψ̂-C against FIFO is **row 7**: +0.0063 [+0.0028, +0.0098] / +0.0085
  [+0.0057, +0.0114] / +0.0094 [+0.0050, +0.0136], each EQUIV. The expectation's reasoning is
  unchanged.
- **m-2 (the handover tension).** Reason 3 for 3000 steps is withdrawn (A1.9 R-1). Handover is at
  `k = 16`, while ADR-0009 §3 says φ needs O(10³) steps to fit. So the first stretch of scored
  evictions runs on a nearly untrained bilinear head. RESULTS reports, per seed, the first 500
  post-warmup steps' displacement distribution beside that of uniform random eviction over 16
  live slots (displacement uniform on 0…15), as a DIAGNOSTIC.
- **m-3 (C0 placement).** C0 is the preflight in A1.2 step 2, run through the slot before any
  training command, with rc 0 required. `train()` checks closure but not disjointness.
- **m-4 (C2 can detect a mismatch).** The old batch check used `n_targets_scored`, which is the
  constant `16 × 63 × 48`, so it could not detect anything. C2 is now:
  - **(a) Primary.** C-2's logged per-step pool ids are identical across the three arms of a seed
    at every step, and every full epoch's ids equal the pool exactly once.
  - **(b) Independent fingerprint.** The per-step pair (`pad_target_fraction`,
    `n_answer_targets`), both on every beat, is identical across arms. I measured its power with
    `c2_fingerprint.py <s>` (rc 0, each seed): over 20,000 random pairs of different 16-document
    pool batches, the pair collided **23 / 9 / 11** times (rates 0.00115 / 0.00045 / 0.00055).
    It missed **2 / 2 / 3 of 240** single-document swaps. A diverged data order therefore shows
    on almost every step. (b) alone cannot certify that one step's batch is identical; (a) can.
  - **(c)** `loss_lm` is bitwise equal across arms for `k < 16` (§8, unchanged).
- **m-5 (concurrency).** Fixed: the launch is concurrent (A1.2).
- **m-6 (attribution).** The heartbeat's `attribution` is `None` in every arm, because neither
  `RSRPolicy` nor `FIFOPolicy` defines `attribution()`. With `b` off and `ν = 0`, §3.4's
  decomposition has a single term, so it is **vacuous** in run 1, not "covered".
  RESEARCH-CONTEXT §10.4 already records the field as NOT IMPLEMENTED. RESULTS says exactly this.
- **m-7 ("one pass").** Fixed by A1.7.
- **m-8 (branch search).** One more hit exists: the unit-test range `(999000, 1001048)` in
  `tests/test_newcomer_bakeoff.py`. It is passed to a range checker and generates no document. It
  does not overlap the pool or EVAL.
- **m-9 (intermediate φ snapshots).** Adopted: `CKPT = STEPS / 3` when `STEPS` is divisible by 3,
  otherwise `CKPT = STEPS` (the end only).
  - At 3000 steps the snapshots are at 1000, 2000 and 3000.
  - At the snapshots before the end, held-out all-query accuracy for psiC and age is measured per
    seed. It is DIAGNOSTIC: does φ's held-out behaviour settle?
  - The headline and secondaries are read at the final step only.
- **m-10 (expectations).** Expectation against outcome (§7.6) is reported as text, never as "met"
  or "missed", and never tallied.

**Also from the review (not a PREREG defect).** Spec §5.2 :493 calls the rescaled target's signal
"unbiased", which contradicts §3.2.1 :208. When RESULTS reports the fill fraction (§7.5), it does
not paraphrase §5.2. Routing this to the corrections file is the PM's and Brendan's.

### A1.9 Owner slots: one value each

| slot | value |
|---|---|
| **O1** γ (both RSR arms; also the oracle reference's construction) | `______` |
| **O4** φ weight decay (decoupled) | `______` |
| **R-1** total optimizer steps per arm | `______` |

**Derived mechanically from R-1, never typed separately:**

- `T_warm` as a fraction of total epochs = `16 / R-1`;
- `CKPT` (A1.8 m-9);
- the compute projection = measured `step_s × R-1`, an extrapolation, labelled.

**Evidence for O1** (§12 O1 stands; the review added the second row):

| | γ = 0.9 | γ = 0.97 | γ = 0.5 | γ = 0 |
|---|---|---|---|---|
| observable share of the discounted weight under FIFO censoring, fresh slot, ≤ 15 future terms (`1 − γ^15`; arithmetic re-run here) | **0.794** | **0.367** | 0.99997 | 1 (present term only) |
| measured on this substrate | B2 rows 5–7, B0 rows 39–40, bake-off rows 15–23 | none | none | B0's rule near FIFO; E0h R² 0.45–0.55 (unratified) |
| target scale (ADR-0009 §3, arithmetic) | ≈ 0.625 | ≈ 2.1 | ≈ 0.125 | ≈ 1/M |

The lookahead-room context is unchanged (`room_3b` < 0.05 on every seed).

**Evidence for O4:**

- With group lr 1e-3, decoupled decay multiplies φ by `(1 − 1e-5)^R-1` absent gradient. At 3000
  steps that is **0.970** (arithmetic re-run here: 0.97045).
- B1 **never measured wd = 0**. Its arms were `decoupled_wd`, `phi_clip`, `joint_clip` and
  `coupled_l2`, all at `WD = 0.01`.
- `0.0` would need code change C-3 (a flag, or a change to `ADAMW_WEIGHT_DECAY`, stamped into the
  optimizer record) and its own review.
- `0.01` needs no change.

**Evidence for R-1:**

- No ruling or spec passage fixes the length.
- Correction 31(b) notes that §3.4's "fraction of total epochs" presumes few epochs.
- §3.4 says "one epoch of three is a different experiment from one of twelve".

| R-1 | `T_warm` share | for | against |
|---|---|---|---|
| 3000 | 1/187.5 | B1 parity (3000 steps, a 256-document pool, about 187 epochs). ADR-0009 §3's arithmetic says φ needs O(10³) steps | far from §3.4's few-epoch presumption |
| 48–192 (3–12 epochs) | 1/3–1/12 | §3.4's own examples | ADR-0009 §3's underfit risk. The head would be mostly untrained when read |
| a larger pool at 3000 steps | — | fewer epochs at the same φ budget | reopens the ruling's 256-document pool. Not available to this PREREG |

(The original §6's third reason, "It is arm B's own training length", is **withdrawn**. Arm B was
1000 memorisation steps on `[0, 64)` plus 2000 stream steps. That is not a φ-only run of this
kind.)

### A1.10 TO-CONFIRM, updated (adds to §14)

- **T1–T8** stand, with these changes:
  - T3 now also covers the preflight, LRU, the per-seed processes, the snapshot measurement and
    the launch script committed byte-identical to A1.2;
  - T4 is **confirmed** (E18);
  - T8 covers all three seeds.
- **T9.** Code change C-1 lands on a reviewed head. Its flag spelling fills `C1_FLAGS`, and its
  label spelling is recorded.
- **T10.** Code change C-2 lands on a reviewed head. Its flag (if any) fills `C2_FLAGS`, and the
  logged-ids field name is recorded for C2(a).
- **T11.** `run.py` runs `LRUPolicy` through an observe-capable eval loop (it reads `attn.alpha`),
  one fresh policy per document or a row-correct batched policy.
- **T12.** The final config hash, labels and arm list are quoted from the run. The dry-run hashes
  (`--iters 0`, no C-1/C-2) are not the run's.

### A1.11 What this author has additionally seen (declared)

- **Seeds 1 and 2 at `--iters 0`:** the warm-start records and the liveness ratios (A1.4).
- **The nine-arm dry run:** each log's label, band, ratio and vocabulary hash (A1.2).
- **C4 R on seeds 1 and 2:** equal to their ledger values (A1.4).
- **E15:** 50 frozen, train-mode FIFO steps on seed 0's pool. Only rc, heartbeat kinds and the
  liveness ratio were read.
- **E16:** one frozen FIFO step on seed 0's pool.
- **C2 fingerprint counts:** computed from document text only, with no model.
- **Not seen:** no EVAL document has been run through any model; no φ has been trained beyond the
  single-step timing runs (§14a); no number about a trained φ exists.


---

## AMENDMENT 2 (2026-10-01): correction-20 slot O5, oracle decoupled from O1, launch under tmux

**Written 2026-10-01 by a model** (Claude Opus 5.5, the PREREG author, on the PM's instruction).
**Not written by Brendan.** It is committed alone, before any run-1 step. Everything above this
heading is unedited: the original PREREG at `7c58777` and Amendment 1 at `6a4c5a8`.

**What prompted it.** The second review, `docs/reviews/prereg-rsr-run1/6a4c5a8c6785.md` (commit
`c02cae7`), returned MERGE WITH FIXES. It raised two new MAJOR findings (A1-1, A1-2) and four new
MINOR findings (A1-1 to A1-4). The PM decided the oracle's γ (A2.2). The owner is asked for the
four slots in A2.8.

**Precedence.** Where this amendment and anything above it disagree, this amendment governs.
Superseded:

- A1.2's launch script and step 3 → A2.4 and A2.7;
- A1.2's E19 row → A2.5;
- the A1.4 scope sentence → A2.6;
- A1.3's C5 sentence → A2.3;
- A1.9's slot table and the oracle clause of its O1 row → A2.8 and A2.2;
- A1.10 T7, T9 and T10 → A2.9.

**Launch preconditions, restated.** All of these must hold before any run-1 training command:

1. **All four owner slots** (A2.8: O1, O4, R-1, O5) are filled.
2. C-1 (needed only if O5 = eval) and C-2 are on a reviewed head of `feat/rsr-run1-ready`.
3. `experiments/rsr-run1/run.py` and the launcher are committed and reviewed, and the preflight
   passes.
4. The launch is from a clean pinned head (A1.5), **inside tmux** (A2.4).

### A2.1 MAJOR-A1-1: the correction-20 reading becomes owner slot O5

**What O5 asks.** For run 1's frozen transformer, which mode is the substrate run in, during
training and during evaluation?

- **`eval`** is this PREREG's proposed reading (A1.3): dropout off in every arm. It needs code
  change C-1.
- **`train`** is the literal text of correction 20 for the LM loss, even though under D2(b) the
  LM loss is reported and not optimised. It is the as-built `frozen_model_mode = "train (dropout
  live)"`.

**Nothing launches until Brendan fills O5.** The launcher refuses an unfilled or invalid O5, with
exit 3 (A2.5).

**Evidence that `eval` is consistent with correction 20** (review §c, items 1–6):

1. The retention clause is satisfied: `r_i` stays eval-mode, from L1(a)'s eval forward.
2. The correction's rationale ("noise in a policy-relevant direction") extends to `s_i`, the input
   the policy reads. The signed ADR-0009 L2 already extended it to `c_t`.
3. There is no optimised LM loss to protect: under D2(b), `objective = None`
   (`loop.py:1239`). The correction predates any frozen configuration.
4. It is an explicit, recorded mode switch: a flag, stamped into `frozen`, the hash and the label,
   and refused without freeze.
5. Every measurement that selected or characterised the substrate ran it in `eval()`: S0-03's
   `answer_readout` (C4's instrument) and B2.
6. Pairing is unaffected, because every arm uses the same mode. This is unlike ADR-0009's
   rejected L1(b).

**Evidence that `eval` departs from correction 20, or needs the owner** (review §c, items 1–5):

1. The literal text has no "if optimised" condition. Under `eval`, the reported `L_NTP` is
   computed in eval mode.
2. It moves run 1 away from its planned successors. The "training continues" arm and E1/E3 run the
   training forward in train mode, so a run-1 φ differs from theirs in mode as well as in
   freezing.
3. It changes the setting a signed item assumed. L1(a)'s second forward becomes numerically
   redundant.
4. It reverses a reviewed build decision (m-4: "train (dropout live)").
5. The substrate *learned* to use memory while writing train-mode gestalts. `eval` matches how it
   was *measured*, not how it was *trained*.

**If O5 = `train`, exactly this changes:**

1. **The C-1 flag is dropped.** `C1_FLAGS` is empty. The launcher enforces this: O5 = train with a
   C-1 flag refuses (E21). Each arm runs `frozen_model_mode = "train (dropout live)"`, the
   as-built default, which is recorded in `frozen` and the hash. C-1 itself is then not a launch
   precondition.
2. **A named confound is declared on every ψ̂-C contrast.** Name: **TRAIN/EVAL MEMORY SHIFT
   (psiC only)**. During training, psiC regresses on and decides over dropout-noised train-mode
   slot gestalts `s_i`. During held-out evaluation it reads clean eval-mode gestalts. The fifo, age
   and LRU arms read no gestalt, so the shift enters only contrasts involving psiC.
   - RESULTS prints the name and this description beside the headline (psiC − fifo, per seed and
     across seeds).
   - It also prints them beside psiC − age, psiC − LRU, and every psiC entry in §7.2–§7.4.
   - The headline remains the headline, read with the confound named. No other headline is
     substituted.
3. **Unchanged under `train`:**
   - C2(c): `loss_lm` is bitwise equal across arms for `k < 16`. Dropout masks are drawn in the
     same order in every arm, and this was measured at `k = 0` in §8's timing runs.
   - A1.3's statement that liveness consumes the global RNG stream stays a note; C5 is per A2.3.
   - Every other control and outcome.

### A2.2 MAJOR-A1-2: the oracle reference uses γ_o = 0.9, a construction parameter decoupled from O1

**PM's decision: the oracle reference is built as `OraclePolicy(discounted_demand(doc, 0.9))`
whatever O1 is.**

- γ_o is a construction parameter of a reference, not a ψ̂ constant. It is not read from or
  written to the registry, and it is recorded in RESULTS as `oracle_gamma: 0.9`.
- A1.9's clause "also the oracle reference's construction" is withdrawn. O1 sets only the RSR
  arms' γ.

**Why this is not result-sensitive.** I measured T7 at `c02cae7`.

- Setup: model-free (`rsr.metrics.headroom.simulate`, M = 16), on the **pool** documents
  `[993000, 993256)` of each seed, never EVAL. The script was `oracle_gamma.py <seed> 256`, 1
  thread, rc 0 for each seed.
- Output, literally:

```
seed 0 docs 256: gamma_o=0.5 vs 0.9: decisions 8192, differing 0, documents differing 0
seed 0 docs 256: gamma_o=0.97 vs 0.9: decisions 8192, differing 0, documents differing 0
seed 0 docs 256: gamma_o=0.0 vs 0.9: decisions 8192, differing 5562, documents differing 250
seed 0: gamma_o=0.5 model-free oracle hit rate 4717/4717 = 1.0000
seed 0: gamma_o=0.9 model-free oracle hit rate 4717/4717 = 1.0000
seed 0: gamma_o=0.97 model-free oracle hit rate 4717/4717 = 1.0000
seed 0: gamma_o=0.0 model-free oracle hit rate 3884/4717 = 0.8234
seed 1 docs 256: gamma_o=0.5 vs 0.9: decisions 8192, differing 0, documents differing 0
seed 1 docs 256: gamma_o=0.97 vs 0.9: decisions 8192, differing 0, documents differing 0
seed 1 docs 256: gamma_o=0.0 vs 0.9: decisions 8192, differing 5656, documents differing 249
seed 1: gamma_o=0.5 model-free oracle hit rate 4775/4775 = 1.0000
seed 1: gamma_o=0.9 model-free oracle hit rate 4775/4775 = 1.0000
seed 1: gamma_o=0.97 model-free oracle hit rate 4775/4775 = 1.0000
seed 1: gamma_o=0.0 model-free oracle hit rate 3911/4775 = 0.8191
seed 2 docs 256: gamma_o=0.5 vs 0.9: decisions 8192, differing 0, documents differing 0
seed 2 docs 256: gamma_o=0.97 vs 0.9: decisions 8192, differing 0, documents differing 0
seed 2 docs 256: gamma_o=0.0 vs 0.9: decisions 8192, differing 5539, documents differing 247
seed 2: gamma_o=0.5 model-free oracle hit rate 4793/4793 = 1.0000
seed 2: gamma_o=0.9 model-free oracle hit rate 4793/4793 = 1.0000
seed 2: gamma_o=0.97 model-free oracle hit rate 4793/4793 = 1.0000
seed 2: gamma_o=0.0 model-free oracle hit rate 3956/4793 = 0.8254
```

**What this shows:**

- **At 0.5, 0.9 and 0.97, the decisions are identical:** 0 of 8192 differ per seed, and the
  model-free hit rate is 1.0. That is Belady's MIN.
- **At γ_o = 0 the rule is something else.** Its decisions differ on 5539–5656 of 8192, and its
  hit rate is 0.82. That is the one-step-lookahead rule the review describes.
- So coupling the oracle to O1 would have changed the reference's meaning exactly when O1 = 0.

**New control C9 (in run, per seed).** On one batch, the first 16 EVAL documents of the seed
(`[995000, 995016)`), `run.py` runs the oracle reference through the **same model-read evaluation
loop** at γ_o = 0.5 and at γ_o = 0.9. It asserts that the two victim sequences are identical,
document by document.

- If it fails, the run is inconclusive (exit 3). Oracle regret and headroom share are not read.
- It runs before the full oracle measurement (TO-CONFIRM T13).
- It checks in the model loop what the model-free measurement above shows: there, decisions
  could differ only if the loop's memory differed from `simulate`'s, for example a sentence with
  no EOS write.

### A2.3 MINOR-A1-4: C5 restated

**The liveness band stays a control in every arm and seed.**

- `live` is required.
- `inert` → exit 5, quarantine per R-2026-09-22-inert-checkpoint-quarantine, and **no** retention
  number is reported for that seed (the reading rule's rc 5 row).
- `inconclusive` → rc 1 with a completed footer. The arm is not read.

**Only the liveness *ratio* is reported.** It is not compared across arms or seeds, and it is not a
control.

### A2.4 MINOR-A1-2: launch inside tmux (measured); `nohup` does NOT protect the jobs

**Hang-up probe.** SIGHUP was sent to the launch's process group 4 s after start, as a terminal
hang-up does. Each launch form ran the launcher's own backgrounding pattern, with two slot jobs
that sleep 20 s.

- Isolated orchestrator root; zsh 5.9; tmux 3.7b on dedicated sockets.
- Logs: `scratchpad/run1-prereg/exitprobe/hup/`; probes `hup_probe.py` and `hup_tmux2.py`.

| id | launch form | result |
|---|---|---|
| H1 | `/bin/zsh launcher` | launcher rc 1; **both jobs `crashed`, rc 129, signal 1** |
| H2 | `nohup /bin/zsh launcher` | launcher rc 0; **both jobs `crashed`, rc 129, signal 1** |
| H3 | `tmux -L … new-session -d … "/bin/zsh launcher …"`, then SIGHUP to the launching group | launching group killed (rc −1). The tmux pane is a **different session** (pane sid 79630 vs launcher sid 79618). **Both jobs `done`, rc 0**; the inner launcher printed `rc[j1]=0`, `rc[j2]=0` |

**Why `nohup` fails.** `nohup` sets SIGHUP to ignored, but `orchestrator/slot.py` installs its own
SIGHUP handler (`_FORWARDED` includes SIGHUP). That handler overrides the inherited ignore and
forwards the signal to the child. So a slot-wrapped job under `nohup` still dies on a hang-up.

**The procedure, replacing A1.2's step 3.** Launch **inside a detached tmux session on a
dedicated socket**. The dedicated socket keeps it apart from any existing tmux server.

```zsh
W=/Users/keanooo7/retrieval-successor-retention/.worktrees/rsr-run1-run
/opt/homebrew/bin/tmux -L rsr-run1 new-session -d -s rsr-run1 \
  "cd $W && GAMMA=<O1> WD=<O4> STEPS=<R-1> CKPT=<derived> O5=<O5> C1_FLAGS='<T9>' C2_FLAGS='<T10>' \
   /bin/zsh experiments/rsr-run1/launch_run1.zsh > runs/rsr-run1-launch.log 2>&1; \
   echo \$? > runs/rsr-run1-launch.rc"
rc_tmux=$?          # 0 = the session started; the launcher's own rc is runs/rsr-run1-launch.rc
```

To watch it: `tmux -L rsr-run1 attach -t rsr-run1`.

**Measured: launcher v2 dry run inside tmux (E22).**

- At `c02cae7`: `GAMMA=0.9` (a placeholder), `WD=0.01`, `STEPS=0`, `CKPT=1000`, `O5=train`. O5
  was `train` only because C-1 is absent at this base. Isolated root, `JOB_SUFFIX=-dry2`.
- `tmux new-session` rc 0. Launcher rc (from its rc file) 0. 11 s wall.
- All nine `rc[…]=0`.
- Nine job records: `done 0 threads 1 pinned c02cae749d79`.
- `launch-git.json`:

```
{"run_sha": "c02cae749d79f0184bffb0ec3e1a19d14a178be2", "dirty": false, "porcelain_lines": 0, "gamma": "0.9", "wd": "0.01", "steps": "0", "ckpt": "1000", "o5": "train", "c1_flags": "", "c2_flags": "", "c3_flags": "", "launched_utc": "2026-10-01T10:42:09Z"}
```

### A2.5 MINOR-A1-3: an unfilled or inconsistent slot exits 3

Launcher v2 (A2.7) checks every slot before doing anything else. Measured (E21 and E20v2), each
with **no output directory created**:

| case | rc | message |
|---|---|---|
| no slots set | **3** | `DID NOT RUN: slot GAMMA unfilled` |
| `O5=maybe` | **3** | `O5 must be eval or train, got 'maybe'` |
| `O5=eval` without `C1_FLAGS` | **3** | `O5=eval needs C1_FLAGS (C-1's flag)` |
| `O5=train` with `C1_FLAGS='--frozen-model-mode eval'` | **3** | `O5=train: C1_FLAGS must be empty` |
| `WD=0.0` without `C3_FLAGS` | **3** | `O4=0.0 needs C3_FLAGS (code change C-3)` |
| dirty tree (scratch repo with an untracked file) | **3** | `not a clean committed head (rc_sha=0 rc_status=0)` |

**Reading rule for the launcher's own rc** (`runs/rsr-run1-launch.rc`):

- **0**: every training process was started and collected. Each one's own rc is in
  `logs/<arm>-s<seed>.rc`, read by A1.2's per-job rule.
- **3**: did not run. Nothing was started.

Any other value is reported as itself. The A1.2 E19 row (rc 1) described the old launcher and is
superseded.

### A2.6 MINOR-A1-1: the scope sentence, replaced

Verbatim in RESULTS, beside every within-seed CI:

> This interval resamples EVAL documents only. It holds fixed the trained φ (one φ
> initialisation, one data order), the substrate seed and the pool. It is a within-seed interval.
> The three per-seed values show the **combined** variation of substrate seed, φ initialisation,
> data order and EVAL documents, which change together across seeds. They cannot separate
> run-to-run variation from substrate-seed variation.

### A2.7 Launcher, version 2 (exact text; supersedes A1.2's)

`experiments/rsr-run1/launch_run1.zsh` is committed **byte-identical** to the text below
(TO-CONFIRM T3). This is the text that ran E20v2, E21 and E22.

**Changes from A1.2:**

- slot checks with exit 3;
- O5 and WD as slots, with consistency checks against `C1_FLAGS` and `C3_FLAGS`;
- every slot value is recorded in `launch-git.json`.

At launch, `JOB_SUFFIX` (a dry-run hook) and `OUT` must be **unset**.

```zsh
#!/bin/zsh
# RSR run 1 launch, version 2 (PREREG Amendment 2, A2.4; supersedes A1.2's text).
# Run with /bin/zsh INSIDE tmux (A2.4), from the dedicated run worktree at the final
# reviewed commit. Owner slots: GAMMA (O1), WD (O4), STEPS (R-1), O5 (eval|train).
# Derived: CKPT (A1.8 m-9). Code-change flags: C1_FLAGS (C-1), C2_FLAGS (C-2), C3_FLAGS
# (C-3, only if O4 != 0.01). OUT defaults to runs/rsr-run1. Any unfilled or inconsistent
# slot exits 3 ("did not run"). Every rc is captured directly, one per arm x seed.
R=/Users/keanooo7/retrieval-successor-retention
GIT=/opt/homebrew/bin/git
refuse() { print -u2 "DID NOT RUN: $1"; exit 3 }
for v in GAMMA WD STEPS CKPT O5; do [[ -n ${(P)v} ]] || refuse "slot $v unfilled"; done
[[ $O5 == eval || $O5 == train ]] || refuse "O5 must be eval or train, got '$O5'"
[[ $O5 == eval && -z ${C1_FLAGS:-} ]] && refuse "O5=eval needs C1_FLAGS (C-1's flag)"
[[ $O5 == train && -n ${C1_FLAGS:-} ]] && refuse "O5=train: C1_FLAGS must be empty"
[[ $WD != 0.01 && -z ${C3_FLAGS:-} ]] && refuse "O4=$WD needs C3_FLAGS (code change C-3)"
[[ $WD == 0.01 && -n ${C3_FLAGS:-} ]] && refuse "O4=0.01 is the built value: C3_FLAGS must be empty"
: ${OUT:=runs/rsr-run1}
C1=(${=C1_FLAGS:-}) ; C2=(${=C2_FLAGS:-}) ; C3=(${=C3_FLAGS:-})

RUN_SHA=$($GIT rev-parse HEAD); rc_sha=$?
PORCELAIN=$($GIT status --porcelain); rc_status=$?
if (( rc_sha != 0 || rc_status != 0 )) || [[ -n $PORCELAIN ]]; then
  refuse "not a clean committed head (rc_sha=$rc_sha rc_status=$rc_status)"
fi
mkdir -p $OUT/logs; rc_mkdir=$?
(( rc_mkdir == 0 )) || refuse "mkdir $OUT/logs rc=$rc_mkdir"
print -r -- "{\"run_sha\": \"$RUN_SHA\", \"dirty\": false, \"porcelain_lines\": 0, \"gamma\": \"$GAMMA\", \"wd\": \"$WD\", \"steps\": \"$STEPS\", \"ckpt\": \"$CKPT\", \"o5\": \"$O5\", \"c1_flags\": \"${C1_FLAGS:-}\", \"c2_flags\": \"${C2_FLAGS:-}\", \"c3_flags\": \"${C3_FLAGS:-}\", \"launched_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" > $OUT/launch-git.json

typeset -A SHA
SHA[0]=0ee3f8a69b507d927c631eb85116e1cd9739ee4472c44eb7f145d739a118da60
SHA[1]=dadd1e08a3849c1211c8479394df4060f91cd2a6e188b97701b783f2068a3da8
SHA[2]=b507ebc573a54316f9d3c20337f49e664bb9c414388af925c5a8e87efd6b63b8
SLOT=(env PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run
      --lane cpu-det --slots 1 --wait 3600 --pinned-sha $RUN_SHA)
RSR=(--policy rsr --policy-override b_enabled=false --policy-override shadow_enabled=false
     --policy-override nu=0 --policy-override beta=1 --policy-override gamma=$GAMMA)

typeset -A pid rcs
for s in 0 1 2; do
  B=$R/.worktrees/fresh-stream/runs/fresh-stream/B/seed$s
  for a in psiC fifo age; do
    case $a in
      psiC) P=("${RSR[@]}") ;;
      fifo) P=(--policy fifo) ;;
      age)  P=("${RSR[@]}" --policy-override psi_override=age_only) ;;
    esac
    k=$a-s$s
    "${SLOT[@]}" --job-id rsr-run1-$k${JOB_SUFFIX:-} -- \
      .venv/bin/python -m rsr.train.loop \
        --d 128 --batch 16 --iters $STEPS --steps-per-stream 48 --memory-slots 16 \
        --lr 0.001 --seed $s --beat-every 1 --ckpt-every $CKPT --device cpu \
        --freeze-transformer "${C1[@]}" "${C3[@]}" \
        --pool-offset 993000 --pool-length 256 --pool-vocab-documents 64 "${C2[@]}" \
        --warm-start $B/ckpt-003000.pt --warm-start-sha256 ${SHA[$s]} \
        --warm-start-manifest /Users/keanooo7/rsr-substrate/2026-09-27/MANIFEST.sha256 \
        --warm-start-root $R --warm-start-source-config $B/result.json \
        "${P[@]}" --out-dir $OUT/$a/seed$s \
      > $OUT/logs/$k.log 2>&1 &
    p=$!; pid[$k]=$p      # measured: pid[$k]=$! stores the literal string "$!" in zsh 5.9
  done
done
for k in ${(ok)pid}; do
  wait ${pid[$k]}; rcs[$k]=$?
  print -r -- ${rcs[$k]} > $OUT/logs/$k.rc
  print -r -- "rc[$k]=${rcs[$k]}"
done
```

### A2.8 Owner slots: one value each (supersedes A1.9's table)

| slot | value |
|---|---|
| **O1** γ, both RSR arms only (the oracle uses γ_o = 0.9, A2.2) | `______` |
| **O4** φ weight decay (decoupled) | `______` |
| **R-1** total optimizer steps per arm | `______` |
| **O5** substrate mode under freeze: `eval` (this PREREG's proposed reading of correction 20, needs C-1) or `train` (the literal reading; adds the named confound of A2.1) | `______` |

- The evidence for O1, O4 and R-1 is A1.9's. The O1 table's "also the oracle" clause no longer
  applies.
- The evidence for O5 is A2.1.
- R-1 is fixed before the run and **not revisited after any snapshot reading** (review §d).
- The derived values are unchanged: the `T_warm` fraction `16 / R-1`, and `CKPT` per A1.8 m-9.

### A2.9 TO-CONFIRM, updated

- **T7: measured model-free** (A2.2). It is also enforced in the run by C9.
- **T9: `C1_FLAGS`.** Filled from the run1-ready author's final hand-back, once the PM sends it.
  Until then, TO-CONFIRM. Observed spellings, read only, not chosen:
  - at `67c2450`, `--frozen-eval-mode`;
  - at the later head `c02b16d`, after `6d89ac3` ("C-1 frozen_model_mode {train,eval}
    (--frozen-model-mode)"), the flag is `--frozen-model-mode {train,eval}`, so eval is
    `--frozen-model-mode eval`.
  - The `.worktrees/rsr-run1-ready` worktree showed an uncommitted `src/rsr/train/loop.py` edit
    when this was checked, so the head may still move.
  - Under O5 = train, `C1_FLAGS` is empty whatever the spelling.
- **T10: `C2_FLAGS`.** At `c02b16d`, a pool run samples permutation epochs with no flag, and each
  step record carries `pool_doc_ids` (C2(a)'s field). If the final head confirms this, `C2_FLAGS`
  is empty. TO-CONFIRM from the same hand-back.
- **T13: C9.** The oracle at γ_o = 0.5 and 0.9 on the first EVAL batch, in the model-read loop,
  implemented in `run.py` before the full oracle measurement.
- **T3:** now covers launcher v2, committed byte-identical (A2.7).

### A2.10 What this author has additionally seen (declared)

- **The oracle's model-free decisions and hit rates** on the **pool** documents of seeds 0–2
  (A2.2). These are document-structure facts with no model. No EVAL document was used.
- **The hang-up probes**: sleep jobs only.
- **Launcher v2's dry run at `--iters 0`**: only the rcs, job records and `launch-git.json` were
  read.
- No EVAL document has been run through any model, and no φ has been trained.


---

## AMENDMENT 3 (2026-10-01): definitions the harness had to choose; launcher v3

**Written 2026-10-01 by a model** (Claude Opus 5.5, the PREREG author, on the PM's instruction).
**Not written by Brendan.** It is committed alone, before any run-1 step and before any EVAL
document has gone through any model. Everything above this heading is unedited.

**What prompted it.**

- The run-1 harness, `feat/rsr-run1-harness` at `58d25cc`, had to choose definitions that the
  PREREG did not fix. Its build report, `~/Documents/RSR-2026-09-30-day/reports/RUN1-HARNESS-build.md`,
  lists these as "PREREG ambiguities" A1–A14. The PM re-ran its test file: `passed` 68 of 68, rc 0,
  133 s.
- The third review, `docs/reviews/prereg-rsr-run1/7fcd3f91ca8f.md` (commit `7f7e2ad`), returned
  MERGE with five MINOR findings. MINOR-A2-3 and MINOR-A2-4 ask for launcher changes, which are
  made in A3.7.

This amendment registers each definition **as an analysis or protocol definition made before
data**. A3.1–A3.14 follow the harness's A-numbers; the `ambiguity` column of the A3.0 table maps
each one to its subsection.

**🔴 The owner slots stay BLANK.** O1, O4, R-1 and O5 (A2.8) are not filled here. Brendan's values
come in a separate Amendment 4. Nothing in this amendment depends on any of their values.

**Precedence.** Where this amendment and anything above it disagree, this amendment governs.
Superseded:

- A2.7's launcher → A3.7;
- A2.4's tmux command line → A3.8;
- A2.2's consequence of a C9 failure ("inconclusive (exit 3)") → A3.6;
- the placement wording "beside every within-seed CI" in A1.4 and A2.6 → A3.5;
- A2.9's T7 wording → A3.15.

### A3.0 What the harness chose, now registered

| ambiguity | registered definition | section |
|---|---|---|
| A1 | `measure` / `finalize` fix the seeds at (0, 1, 2), with no seed flag. This is A1.4's decision as built | (recorded) |
| A2 | held-out RSR passes take their decision context from `eval_forward` | A3.2 |
| A3 | held-out batch size 16, for every pass | A3.3 |
| A4 | held-out NLL is pooled per target | A3.4 |
| A5 | where the scope sentence is printed | A3.5 |
| A6 | a C9 or C9b failure scopes to the oracle-derived numbers | A3.6 |
| A7 | **C10**, named | A3.7a |
| A8 | B2's random seeding, verbatim | A3.9 |
| A9 | snapshot "minus final fifo" is DIAGNOSTIC | A3.10 |
| A10 | the pre-run T5 timing used a **pool** batch, so EVAL stayed unseen. The run's own T5 times the first EVAL batch of every pass inside `measure-seed`. Recorded as built | (recorded) |
| A11 | ledger status for a failed control | A3.11 |
| A12 | `RSR_ORCH_ROOT`, `OUT` and `JOB_SUFFIX`: recorded and refused; CKPT derived | A3.12, A3.7 |
| A13 | the `finalize` step | A3.13 |
| A14 | LRU's distance from fifo | A3.14 |

### A3.2 (A2) Held-out decision context: `eval_forward`

- **The definition.** In every held-out pass whose policy needs an evaluation context (psiC, age,
  their slots-zeroed passes, and their snapshots), the decision reads the srep from
  `eval_forward`. That is `run_policy_loop(eval_capture=True)`, the training path's ADR-0009 L2
  route. The fifo, oracle, LRU and random passes read no context.
- **Why.** It is the same code path the policy was trained through. Under held-out eval mode, the
  harness measured it as **identical** to `step_forward` in victims and answers, on its fixture.
  So the choice is not result-sensitive there, and it keeps one decision path for training and
  evaluation.
- **Cost.**
  - The harness report's A2 states about **1.8 CPU-h** over three seeds.
  - Its own T5 table implies less. Dropping `eval_forward` saves about 1,090 s per seed, about
    0.9 CPU-h over three seeds.
  - Both figures are extrapolations from one batch. I could not reconcile them from the report,
    so both are stated.
  - Either way the choice was not made on cost.

### A3.3 (A3) Held-out batch size: 16

- Every held-out pass, all references, slots-zeroed passes, snapshots and C9/C9b included, runs
  EVAL in consecutive blocks of **16 documents**, in id order (`[995000 + 16i, 995000 + 16i + 16)`).
- Policies are row-correct, with one state per row.
- All passes share the blocking, so any numerical effect of the batch shape is common to every
  arm.

### A3.4 (A4) Held-out NLL: pooled per target (exact definition)

For one document `d`, run under one pass, in eval mode:

- **Real-token NLL.** Take every real next-token target of every sentence step `t ∈ [0, 48)`:
  positions `j ≥ 1` of the sentence where `mask_t[j]` is true.
  - `n_d` is the number of these targets.
  - `s_d = Σ −log p(target)`, using `loop.lm_token_losses` (natural log), accumulated in float64.
- **Answer-token NLL.** The same, over S0-03's answer targets only (`answer_targets`' mask). Each
  term is that target's `−log p`.
- **Level over a document set D:** `Σ_{d∈D} s_d / Σ_{d∈D} n_d`, pooled per **target**.
- **Bootstrap replicate:** the same ratio with document multiplicities from the paired
  resampling, `W`, a 2000 × 2048 matrix with generator `20260927 + seed`.
- **Contrast:** `level(a) − level(b)` on the same replicate.

This is **not** training's `loss_real_tokens`, which is a mean over sentence steps of per-step
means (`lm_loss`). The held-out and in-loop loss numbers are therefore not directly comparable,
and RESULTS says so.

Accuracy is pooled per answer target in the same way (S0-03's definition, §7.1).

### A3.5 (A5) Where the scope sentence appears

A2.6's sentence, verbatim, is printed in two places:

1. **beside the headline**, per seed and across seeds;
2. **at the head of each seed's held-out table**, where it governs every within-seed CI in that
   table.

It is not repeated after every contrast line. This replaces the placement wording "beside every
within-seed CI" in A1.4 and A2.6. The sentence itself is unchanged.

### A3.6 (A6) A C9 or C9b failure invalidates ONLY the oracle-derived numbers

**This supersedes A2.2's "If it fails, the run is inconclusive (exit 3). Oracle regret and
headroom share are not read."**

The controls, on the first 16 EVAL documents per seed, in the model-read loop:

- **C9.** The oracle's victims at γ_o = 0.5 and at γ_o = 0.9 are identical, document by document.
- **C9b**, added for review MINOR-A2-2. At γ_o = 0.9, the loop's victims equal a model-free
  replay built from each document's **own** `discounted_demand`. This catches a batched wrapper
  handing a row another document's demand, which C9 and C7 cannot.

If either fails:

- **every oracle regret** (each bucket, each arm) and **every headroom share** is marked
  **NOT READ (C9/C9b failed)** in RESULTS and the ledger;
- the oracle reference's own accuracy row carries the same tag;
- the run's exit code and status do **not** change;
- **no other number changes:** not the headline, any non-oracle contrast, any diagnostic, or any
  control.

The headline does not depend on the oracle. A1.2's reading rule is otherwise unchanged.

### A3.7 Launcher, version 3 (exact text; supersedes A2.7's)

**Coordination.** `experiments/rsr-run1/launch_run1.zsh` must be **byte-identical** to the text
below.

- The harness builder copies it from here.
- The harness's byte-identity check (`launcher_text_from_prereg`, which at `58d25cc` reads under
  the heading "### A2.7 Launcher, version 2") must be re-pointed to **this** heading,
  "### A3.7 Launcher, version 3" (TO-CONFIRM T14).
- sha256 of the launcher text: **`89817f42798ed401aca3b0eac26a50a5280fdd6f07ddfefc642ad7521ec228c3`**.

**Changes from v2** (review MINOR-A2-3 and MINOR-A2-4):

- **The environment.**
  - `RSR_ORCH_ROOT` must be unset, or resolve to the shared root
    `/Users/keanooo7/retrieval-successor-retention`.
  - `OUT` and `JOB_SUFFIX` must be unset.
  - Otherwise the launcher refuses with exit 3, unless `DRY_RUN=1`.
  - `DRY_RUN` must be unset or exactly `1`.
  - All four values are recorded in `launch-git.json`: `rsr_orch_root`, `out`, `job_suffix` and
    `dry_run`. An empty string means unset.
- **CKPT is derived, never input.** `CKPT = STEPS / 3` if `STEPS > 0` and `STEPS % 3 == 0`,
  otherwise `CKPT = STEPS` (A1.8 m-9). A `CKPT` given in the environment is refused. The derived
  value and the rule are recorded in `launch-git.json` (`ckpt`, `ckpt_rule`).
- **`STEPS`** must be a non-negative integer, and `≥ 1` unless `DRY_RUN=1`.
- **Unchanged:** the four owner slots and their consistency checks; the clean-head check;
  `--pinned-sha`; `--wait 3600`; one rc per arm × seed; the `p=$!; pid[$k]=$p` form.
- **Not addressed:** review MINOR-A2-1's presence-only checks. C10 checks content (A3.7a).

**Measured at `7f7e2ad`.** CPU, zsh 5.9, isolated orchestrator root. Logs are in
`scratchpad/run1-prereg/exitprobe/v3/`.

| case | rc | message |
|---|---|---|
| leftover `OUT` | **3** | `OUT=… is set (a dry-run hook)` |
| leftover `JOB_SUFFIX` | **3** | `JOB_SUFFIX=-x is set (a dry-run hook)` |
| foreign `RSR_ORCH_ROOT` | **3** | `RSR_ORCH_ROOT=… is not the shared root /Users/keanooo7/retrieval-successor-retention` |
| `CKPT=1000` given | **3** | `CKPT is derived from STEPS (A1.8 m-9); do not set it` |
| `STEPS=abc` | **3** | `STEPS must be a non-negative integer, got 'abc'` |
| `STEPS=0` without `DRY_RUN` | **3** | `STEPS must be >= 1` |
| `DRY_RUN=2` | **3** | `DRY_RUN must be unset or 1, got '2'` |
| nothing set | **3** | `slot GAMMA unfilled` |
| `RSR_ORCH_ROOT` = the shared root (in a dirty scratch repo) | **3** at the *clean-head* check | the environment guard passed |
| `DRY_RUN=1` with an isolated root, `OUT` and `JOB_SUFFIX` (dirty scratch repo) | **3** at the clean-head check | the guard passed |

None of these cases created an output directory.

**CKPT derivation, measured.** The launcher ran in a **clean** scratch git repo with `DRY_RUN=1`.
That repo has no `.venv`, so every slot command failed at once with rc 127
(`env: .venv/bin/python: No such file or directory`) and nothing ran. Launcher rc 0 each time.

| `STEPS` | 3000 | 200 | 192 | 1 |
|---|---|---|---|---|
| recorded `ckpt` | **1000** | **200** | **64** | **1** |

The `launch-git.json` for `STEPS=3000` (scratch paths shortened to `<scratch>`, the timestamp elided as `...`; otherwise literal):

```
{"run_sha": "66e32f5cb1234948d27dc9a39d2806250bddb009", "dirty": false, "porcelain_lines": 0, "gamma": "0.9", "wd": "0.01", "steps": "3000", "ckpt": "1000", "ckpt_rule": "STEPS/3 if STEPS>0 and STEPS%3==0 else STEPS (A1.8 m-9)", "o5": "train", "c1_flags": "", "c2_flags": "", "c3_flags": "", "rsr_orch_root": "<scratch>/exitprobe/root12", "out": "<scratch>/v3_ckpt_3000", "job_suffix": "-ckpt3000", "dry_run": "1", ...}
```

`GAMMA=0.9` and `O5=train` in every probe are placeholders, not registered values.

**Nine-job dry run, inside tmux, at `7f7e2ad`.**

- Settings: `DRY_RUN=1`, isolated root, `STEPS=0` (so `ckpt 0`), `O5=train`.
- `tmux new-session` rc 0. Launcher rc 0. 10 s wall.
- All nine `rc[…]=0`.
- Nine job records: `done 0 threads 1 pinned 7f7e2ad4d24d`, each with `--ckpt-every 0` taken from
  the derived value.
- `launch-git.json` recorded `rsr_orch_root`, `out`, `job_suffix` and `dry_run: "1"`.

```zsh
#!/bin/zsh
# RSR run 1 launch, version 3 (PREREG Amendment 3, A3.7; supersedes A2.7's text).
# Run with /bin/zsh INSIDE tmux (A2.4), from the dedicated run worktree at the final
# reviewed commit. Owner slots: GAMMA (O1), WD (O4), STEPS (R-1), O5 (eval|train).
# CKPT is DERIVED here from STEPS (A1.8 m-9), never read from input. Code-change flags:
# C1_FLAGS (C-1), C2_FLAGS (C-2), C3_FLAGS (C-3, only if O4 != 0.01).
# RSR_ORCH_ROOT must be unset or the shared root, and OUT / JOB_SUFFIX unset, unless
# DRY_RUN=1; all four are recorded in launch-git.json. OUT defaults to runs/rsr-run1.
# Any unfilled or inconsistent slot exits 3 ("did not run"). One rc per arm x seed.
R=/Users/keanooo7/retrieval-successor-retention
GIT=/opt/homebrew/bin/git
refuse() { print -u2 "DID NOT RUN: $1"; exit 3 }
[[ -z ${DRY_RUN:-} || $DRY_RUN == 1 ]] || refuse "DRY_RUN must be unset or 1, got '$DRY_RUN'"
if [[ -z ${DRY_RUN:-} ]]; then
  [[ -z ${RSR_ORCH_ROOT:-} || ${RSR_ORCH_ROOT:A} == ${R:A} ]] \
    || refuse "RSR_ORCH_ROOT=$RSR_ORCH_ROOT is not the shared root $R"
  [[ -z ${OUT:-} ]] || refuse "OUT=$OUT is set (a dry-run hook)"
  [[ -z ${JOB_SUFFIX:-} ]] || refuse "JOB_SUFFIX=$JOB_SUFFIX is set (a dry-run hook)"
fi
[[ -z ${CKPT:-} ]] || refuse "CKPT is derived from STEPS (A1.8 m-9); do not set it"
for v in GAMMA WD STEPS O5; do [[ -n ${(P)v} ]] || refuse "slot $v unfilled"; done
[[ $STEPS == <-> ]] || refuse "STEPS must be a non-negative integer, got '$STEPS'"
(( STEPS >= 1 )) || [[ -n ${DRY_RUN:-} ]] || refuse "STEPS must be >= 1"
[[ $O5 == eval || $O5 == train ]] || refuse "O5 must be eval or train, got '$O5'"
[[ $O5 == eval && -z ${C1_FLAGS:-} ]] && refuse "O5=eval needs C1_FLAGS (C-1's flag)"
[[ $O5 == train && -n ${C1_FLAGS:-} ]] && refuse "O5=train: C1_FLAGS must be empty"
[[ $WD != 0.01 && -z ${C3_FLAGS:-} ]] && refuse "O4=$WD needs C3_FLAGS (code change C-3)"
[[ $WD == 0.01 && -n ${C3_FLAGS:-} ]] && refuse "O4=0.01 is the built value: C3_FLAGS must be empty"
if (( STEPS > 0 && STEPS % 3 == 0 )); then CKPT=$(( STEPS / 3 )); else CKPT=$STEPS; fi
ORCH_REC=${RSR_ORCH_ROOT:-} ; OUT_REC=${OUT:-} ; SUFFIX_REC=${JOB_SUFFIX:-}
: ${OUT:=runs/rsr-run1}
C1=(${=C1_FLAGS:-}) ; C2=(${=C2_FLAGS:-}) ; C3=(${=C3_FLAGS:-})

RUN_SHA=$($GIT rev-parse HEAD); rc_sha=$?
PORCELAIN=$($GIT status --porcelain); rc_status=$?
if (( rc_sha != 0 || rc_status != 0 )) || [[ -n $PORCELAIN ]]; then
  refuse "not a clean committed head (rc_sha=$rc_sha rc_status=$rc_status)"
fi
mkdir -p $OUT/logs; rc_mkdir=$?
(( rc_mkdir == 0 )) || refuse "mkdir $OUT/logs rc=$rc_mkdir"
print -r -- "{\"run_sha\": \"$RUN_SHA\", \"dirty\": false, \"porcelain_lines\": 0, \"gamma\": \"$GAMMA\", \"wd\": \"$WD\", \"steps\": \"$STEPS\", \"ckpt\": \"$CKPT\", \"ckpt_rule\": \"STEPS/3 if STEPS>0 and STEPS%3==0 else STEPS (A1.8 m-9)\", \"o5\": \"$O5\", \"c1_flags\": \"${C1_FLAGS:-}\", \"c2_flags\": \"${C2_FLAGS:-}\", \"c3_flags\": \"${C3_FLAGS:-}\", \"rsr_orch_root\": \"$ORCH_REC\", \"out\": \"$OUT_REC\", \"job_suffix\": \"$SUFFIX_REC\", \"dry_run\": \"${DRY_RUN:-}\", \"launched_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" > $OUT/launch-git.json

typeset -A SHA
SHA[0]=0ee3f8a69b507d927c631eb85116e1cd9739ee4472c44eb7f145d739a118da60
SHA[1]=dadd1e08a3849c1211c8479394df4060f91cd2a6e188b97701b783f2068a3da8
SHA[2]=b507ebc573a54316f9d3c20337f49e664bb9c414388af925c5a8e87efd6b63b8
SLOT=(env PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run
      --lane cpu-det --slots 1 --wait 3600 --pinned-sha $RUN_SHA)
RSR=(--policy rsr --policy-override b_enabled=false --policy-override shadow_enabled=false
     --policy-override nu=0 --policy-override beta=1 --policy-override gamma=$GAMMA)

typeset -A pid rcs
for s in 0 1 2; do
  B=$R/.worktrees/fresh-stream/runs/fresh-stream/B/seed$s
  for a in psiC fifo age; do
    case $a in
      psiC) P=("${RSR[@]}") ;;
      fifo) P=(--policy fifo) ;;
      age)  P=("${RSR[@]}" --policy-override psi_override=age_only) ;;
    esac
    k=$a-s$s
    "${SLOT[@]}" --job-id rsr-run1-$k${JOB_SUFFIX:-} -- \
      .venv/bin/python -m rsr.train.loop \
        --d 128 --batch 16 --iters $STEPS --steps-per-stream 48 --memory-slots 16 \
        --lr 0.001 --seed $s --beat-every 1 --ckpt-every $CKPT --device cpu \
        --freeze-transformer "${C1[@]}" "${C3[@]}" \
        --pool-offset 993000 --pool-length 256 --pool-vocab-documents 64 "${C2[@]}" \
        --warm-start $B/ckpt-003000.pt --warm-start-sha256 ${SHA[$s]} \
        --warm-start-manifest /Users/keanooo7/rsr-substrate/2026-09-27/MANIFEST.sha256 \
        --warm-start-root $R --warm-start-source-config $B/result.json \
        "${P[@]}" --out-dir $OUT/$a/seed$s \
      > $OUT/logs/$k.log 2>&1 &
    p=$!; pid[$k]=$p      # measured: pid[$k]=$! stores the literal string "$!" in zsh 5.9
  done
done
for k in ${(ok)pid}; do
  wait ${pid[$k]}; rcs[$k]=$?
  print -r -- ${rcs[$k]} > $OUT/logs/$k.rc
  print -r -- "rc[$k]=${rcs[$k]}"
done
```

### A3.7a (A7) C10, named

**C10 is the per-job recorded-value check.** It runs in `measure-seed`, after training and before
any held-out pass. For every arm × seed job, each of these must hold:

- `frozen["frozen_model_mode"]` equals the record for O5: `"eval (dropout off)"` for `eval`,
  `"train (dropout live)"` for `train`;
- the optimizer record's `weight_decay` equals O4;
- for the RSR arms, the label's `gamma=` and `policy_overrides.gamma` equal O1;
- `iters` equals R-1;
- the checkpoint cadence equals the derived CKPT. This is read from the job's `--ckpt-every` and
  from the checkpoint files on disk, and checked against `launch-git.json`'s `ckpt` and the A3.7
  rule applied to `steps`;
- **`run_sha` (from `launch-git.json`) = the job record's `pinned_sha` = the heartbeat's
  `provenance.git_sha`**, and `dirty` is false;
- the job records are in the **shared** orchestrator root, under their exact ids
  `rsr-run1-<arm>-s<seed>`;
- `launch-git.json` shows `rsr_orch_root` empty or equal to the shared root, and `out`,
  `job_suffix` and `dry_run` empty. **A `DRY_RUN` launch can never be read as the run.**

Each owner value is compared both with `launch-git.json` and with the value the owner registered
in Amendment 4.

**Any mismatch is a control failure: exit 3, ledger status per A3.11.** The run is not read.

### A3.8 The tmux launch command (replaces A2.4's, which passed CKPT)

```zsh
W=/Users/keanooo7/retrieval-successor-retention/.worktrees/rsr-run1-run
/opt/homebrew/bin/tmux -L rsr-run1 new-session -d -s rsr-run1 \
  "cd $W && GAMMA=<O1> WD=<O4> STEPS=<R-1> O5=<O5> C1_FLAGS='<T9>' C2_FLAGS='<T10>' \
   /bin/zsh experiments/rsr-run1/launch_run1.zsh > runs/rsr-run1-launch.log 2>&1; \
   echo \$? > runs/rsr-run1-launch.rc"
rc_tmux=$?
```

- `OUT`, `JOB_SUFFIX`, `RSR_ORCH_ROOT` (unless it is the shared root), `CKPT` and `DRY_RUN` must
  not be set. If any leaks in from the client environment (review MINOR-A2-3), the launcher
  refuses with rc 3 and nothing starts.
- Review MINOR-A2-5 (the pane's `-c` shell sources `~/.zshenv`) is **not** addressed. The slot
  forces the thread variables (E18), and C10 checks every recorded value afterwards.

### A3.9 (A8) Random reference seeding: B2's, verbatim

- The reference is `RandomPolicy`, uniform over all live slots, ×5.
- Each (k ∈ {0, …, 4}, model seed, document) gets its own `torch.Generator`, seeded from the first
  8 bytes of `sha256(f"b2rand:{k}:{seed}:{doc_id}")`. That is B2 §7's construction, **including
  the `"b2rand:"` prefix**, called through B2's own `random_seed_int`, not re-typed.
- The prefix names B2, not run 1. It is kept so that the construction is B2's byte for byte.
- The random reference's accuracy in a bootstrap replicate is the mean, over its 5 seeds, of
  their pooled accuracies (B2 §9.2).

### A3.10 (A9) The snapshot contrast is DIAGNOSTIC

At the snapshots CKPT and 2·CKPT (A1.8 m-9), the held-out passes `psiC@k` and `age@k` report:

- their all-query accuracy level;
- **"minus final fifo"**: `acc(arm@k) − acc(fifo at the final step)`, paired.

Both are **DIAGNOSTIC**, not SECONDARY. Neither is the headline, and neither is read against
anything. R-1 is never revisited after reading them (review of A1, §d).

### A3.11 (A11) Ledger status for a failed control: never "failed"

`scripts/ledger.py:105` allows these statuses:

```
STATUSES = ("ok", "partial", "crashed", "did_not_run", "failed")
```

Its comment at `:101-104` defines **"failed"** as a completed run whose pre-registered check
failed, **exit 1**. There is no `control_failed`. Adding one would change the shared ledger
vocabulary, which this PREREG does not do. Run 1 therefore uses only existing names:

| situation | exit | ledger status |
|---|---|---|
| everything passes | 0 | `ok` |
| a refusal before any measurement: environment (A3.12), C0, missing `launch-git.json` | 3 | `did_not_run` |
| an arm's training rc is not 0 (A1.2's rule), or C6 fails | 3 | `crashed` |
| **a control fails after training** (C1–C4, C7, C8, **C10**) | **3** | **`partial`** (precedent: B2 phase A writes `partial` when its controls are not ok, `experiments/b2-psi-probe/run.py:2220`) |
| a C5 band of `inert` | **5** | **`partial`**, with the ledger note `INERT` |
| C9/C9b fails | unchanged | unchanged (A3.6) |

**Run 1 never writes `failed`.** It has no pre-registered pass/fail check that exits 1.

The harness at `58d25cc` writes `"failed"` with exit 3 for a failed control (`run.py:2265`), and
`"failed"` with exit 5 for inert (`:2185`, `:2200`). Both become `partial` (TO-CONFIRM T15). The
inert case is the same clash as the control case, so it is fixed the same way.

### A3.12 (A12) RSR_ORCH_ROOT, OUT and JOB_SUFFIX

They are **recorded** in three places:

- by `--preflight`;
- in each `measure-seed<s>.json`;
- and now in `launch-git.json`, by launcher v3 (A3.7).

They are **refused** unless `RSR_ORCH_ROOT` is unset or the shared root, and `OUT` and
`JOB_SUFFIX` are unset. The refusals are:

- by the launcher, rc 3 (A3.7);
- by `--preflight` and `measure-seed` (`env_problems`, exit 3, status `did_not_run`);
- by C10, through the recorded values.

`--preflight --dry-run`, `measure-seed --dry-run` and `DRY_RUN=1` exist for tests and dry runs
only. Their outputs are never the run (C10).

### A3.13 (A13) The `finalize` step

There are **four** measurement steps:

- three `measure-seed --seed s` processes, one per seed (A1.6), followed by
- **`finalize`**, a single process through the slot at 1 thread.

`finalize` reads the three `measure-seed<s>.json` documents and the per-document vectors under
`heldout/seed<s>/`. It **runs no model**. It produces:

1. **the headline:** psiC − fifo, all-query, per seed with its within-seed CI, and across seeds
   (the three points, their mean and sd, and the sign counts; no CI);
2. **RESULTS.md**, with SECONDARY and DIAGNOSTIC labels. These texts are **quoted from this file**:
   - the A2.6 scope sentence, placed per A3.5;
   - the A2.1 confound, if O5 = train;
   - the §7.5 caveat;
   - the §7.6 expectation, reported as text per m-10;
3. **the ledger.** Its `config_hash` is the sha256 of `launch-git.json`.

Its exit is the worst seed's: inert gives 5, any other non-OK gives 3. Its status is that seed's,
per A3.11.

### A3.14 (A14) LRU's distance from fifo

It is reported in two ways:

- **`acc.<bucket>.lru_minus_fifo`**: held-out LRU − fifo accuracy, paired, for each bucket;
- LRU's held-out **`fraction_oldest`**: the fraction of LRU's evictions whose displacement is 0,
  i.e. it took the oldest slot. That is how often LRU behaved as FIFO.

Both are SECONDARY, under A1.1's caveat that LRU degrades toward FIFO by design.

### A3.15 T7, restated accurately (supersedes A2.9's wording)

**In exact arithmetic,** on S0-03 every fact is queried once at a distinct position. So
`discounted_demand[t+1][i]` is `γ^(q−t−1)` for a pending assert and 0 otherwise, and the oracle's
argmin is the same for every γ in (0, 1).

**In float64 that is false for tiny γ:** `γ^k` underflows.

- **The harness measured it** (`run.py probe-oracle`, pool `[993000, 993256)`). γ = 0.5 against
  1e-6, 1e-8, 0.999999 and 1 − 1e-12 differs on 0 documents. Against **1e-12 it differs on 1/256**
  (document 993080), and against **1e-300 on 247/256**.
- **I re-measured seed 0** against γ_o = 0.9 (`oracle_gamma_tiny.py 0 256`, rc 0):

```
seed 0 docs 256: gamma_o=0.5 vs 0.9: decisions 8192, differing 0, documents differing 0
seed 0 docs 256: gamma_o=1e-06 vs 0.9: decisions 8192, differing 0, documents differing 0
seed 0 docs 256: gamma_o=1e-08 vs 0.9: decisions 8192, differing 0, documents differing 0
seed 0 docs 256: gamma_o=1e-12 vs 0.9: decisions 8192, differing 24, documents differing 1
seed 0 docs 256: gamma_o=1e-300 vs 0.9: decisions 8192, differing 5487, documents differing 247
seed 0 docs 256: gamma_o=0.999999 vs 0.9: decisions 8192, differing 0, documents differing 0
seed 0 docs 256: gamma_o=0.999999999999 vs 0.9: decisions 8192, differing 0, documents differing 0
```

**T7, restated.** OraclePolicy's decisions are γ-invariant for γ in (0, 1) **as long as the
powers `γ^k`, k ≤ 46, do not underflow in float64**. They were identical at every γ measured
between 1e-8 and 1 − 1e-12: 1e-8, 1e-6, 0.01, 0.1, 0.5, 0.9, 0.97, 0.999, 0.999999 and 1 − 1e-12.
They were not identical at γ = 1e-12 or 1e-300.

**γ_o = 0.9 is unaffected**, and so is C9's 0.5. The phrase "every γ in (0, 1)" above (§10 T7,
A2.2) is to be read as this restatement.

### A3.16 TO-CONFIRM, added

- **T14.** The harness's launcher byte-identity check reads **A3.7**. The committed
  `launch_run1.zsh` has sha256 `89817f42…28c3`.
- **T15.** The harness writes the A3.11 statuses (`partial`, never `failed`), and tags the oracle
  row per A3.6.
- **T16.** C10 compares the new `launch-git.json` fields: `ckpt`, `ckpt_rule`, `rsr_orch_root`,
  `out`, `job_suffix` and `dry_run` (A3.7a).
- **Standing:** T9 (`C1_FLAGS`) and T10 (`C2_FLAGS`), from the run1-ready author's final
  hand-back, and the four owner slots, in Amendment 4.

### A3.17 What this author has additionally seen (declared)

- **Model-free oracle decisions** on the seed-0 pool at extreme γ: document structure, with no
  model involved.
- **Launcher v3 probes:** refusals, a CKPT derivation in which no job could start, and the
  `--iters 0` dry run (rcs and records only).
- **The harness report and code:** read, not run. The test result quoted above is the PM's.
- **No EVAL document has gone through any model.** No φ has been trained.


---

## AMENDMENT 3b (2026-10-01): DRY_RUN can no longer train; the tmux command strips leaked variables; launcher v4

**Written 2026-10-01 by a model** (Claude Opus 5.5, the PREREG author, on the PM's instruction).
**Not written by Brendan.** It is committed alone, before any run-1 step and before any EVAL
document has gone through any model. Everything above this heading is unedited.

**What prompted it.** The review of Amendment 3, `docs/reviews/prereg-rsr-run1/58a9216e51bf.md`
(commit `3ef97c8`), returned MERGE WITH FIXES with one MAJOR finding and six MINOR findings.

**🔴 The owner slots stay BLANK.** O1, O4, R-1 and O5 are not filled here; Amendment 4 fills them.
The values `GAMMA=0.9`, `WD=0.01` and `O5=train` in the probes below are placeholders, not
registered values.

**Precedence.** Where this amendment and anything above it disagree, this amendment governs.
Superseded:

- A3.7's launcher → A3b.1;
- A3.8's tmux command and its claim about leaked variables → A3b.2;
- A3.11's status table → A3b.3;
- A3.6's tagging scope → A3b.4;
- A3.13's status and exit rule → A3b.5;
- A3.15's bound → A3b.6.

### A3b.1 Launcher, version 4

**MAJOR-A3-1: `DRY_RUN=1` now requires `STEPS=0`.**

**The defect, confirmed.** In launcher v3, the environment guard runs only
`if [[ -z ${DRY_RUN:-} ]]`, and nothing ties `DRY_RUN` to `STEPS`. So `DRY_RUN=1` with
`STEPS=3000` skips the root, `OUT` and `JOB_SUFFIX` checks, and trains. The review measured this
leaking through tmux.

**Changes from v3:**

1. `DRY_RUN=1` with any `STEPS ≠ 0` is refused: exit 3, `DRY_RUN=1 needs STEPS=0 (a dry run never
   trains)`. **A dry run can no longer train**, so the guard bypass can no longer start a training
   job.
2. Review MINOR-A3-6: `launch-git.json` is written by Python's `json.dump`, run by the worktree's
   `.venv/bin/python`, not by string interpolation.
   - Values arrive as argv pairs; `dirty: false` and `porcelain_lines: 0` stay typed.
   - A failed write refuses with exit 3.
   - Key order differs from v3. The set of keys is v3's.

Everything else is v3's, byte for byte: the slot checks, the environment guard when `DRY_RUN` is
unset, the CKPT derivation, the clean-head check, `--pinned-sha`, `--wait 3600`, one rc per
arm × seed, and the `p=$!; pid[$k]=$p` form.

**Coordination.** `experiments/rsr-run1/launch_run1.zsh` must be **byte-identical** to the text
below.

- The harness extracts it with its own rule: `t.index(heading)`, then the next zsh fence, up to
  the closing fence, plus a trailing newline.
- The heading to use is **"### A3b.1 Launcher, version 4"**, this subsection's heading. TO-CONFIRM
  T14, updated.
- sha256 under that rule: **`5cd6e5fd712ffab036d87ea6481b3220a3669f071ba955255511fc840b3f7946`**.

```zsh
#!/bin/zsh
# RSR run 1 launch, version 4 (PREREG Amendment 3b, A3b.1; supersedes A3.7's text).
# Run with /bin/zsh INSIDE tmux (A2.4), from the dedicated run worktree at the final
# reviewed commit. Owner slots: GAMMA (O1), WD (O4), STEPS (R-1), O5 (eval|train).
# CKPT is DERIVED here from STEPS (A1.8 m-9), never read from input. Code-change flags:
# C1_FLAGS (C-1), C2_FLAGS (C-2), C3_FLAGS (C-3, only if O4 != 0.01).
# RSR_ORCH_ROOT must be unset or the shared root, and OUT / JOB_SUFFIX unset, unless
# DRY_RUN=1, and DRY_RUN=1 requires STEPS=0 (a dry run never trains). All four are
# recorded in launch-git.json, written by a JSON encoder. OUT defaults to runs/rsr-run1.
# Any unfilled or inconsistent slot exits 3 ("did not run"). One rc per arm x seed.
R=/Users/keanooo7/retrieval-successor-retention
GIT=/opt/homebrew/bin/git
refuse() { print -u2 "DID NOT RUN: $1"; exit 3 }
[[ -z ${DRY_RUN:-} || $DRY_RUN == 1 ]] || refuse "DRY_RUN must be unset or 1, got '$DRY_RUN'"
if [[ -z ${DRY_RUN:-} ]]; then
  [[ -z ${RSR_ORCH_ROOT:-} || ${RSR_ORCH_ROOT:A} == ${R:A} ]] \
    || refuse "RSR_ORCH_ROOT=$RSR_ORCH_ROOT is not the shared root $R"
  [[ -z ${OUT:-} ]] || refuse "OUT=$OUT is set (a dry-run hook)"
  [[ -z ${JOB_SUFFIX:-} ]] || refuse "JOB_SUFFIX=$JOB_SUFFIX is set (a dry-run hook)"
fi
[[ -z ${CKPT:-} ]] || refuse "CKPT is derived from STEPS (A1.8 m-9); do not set it"
for v in GAMMA WD STEPS O5; do [[ -n ${(P)v} ]] || refuse "slot $v unfilled"; done
[[ $STEPS == <-> ]] || refuse "STEPS must be a non-negative integer, got '$STEPS'"
[[ -n ${DRY_RUN:-} ]] && (( STEPS != 0 )) && refuse "DRY_RUN=1 needs STEPS=0 (a dry run never trains), got STEPS=$STEPS"
(( STEPS >= 1 )) || [[ -n ${DRY_RUN:-} ]] || refuse "STEPS must be >= 1"
[[ $O5 == eval || $O5 == train ]] || refuse "O5 must be eval or train, got '$O5'"
[[ $O5 == eval && -z ${C1_FLAGS:-} ]] && refuse "O5=eval needs C1_FLAGS (C-1's flag)"
[[ $O5 == train && -n ${C1_FLAGS:-} ]] && refuse "O5=train: C1_FLAGS must be empty"
[[ $WD != 0.01 && -z ${C3_FLAGS:-} ]] && refuse "O4=$WD needs C3_FLAGS (code change C-3)"
[[ $WD == 0.01 && -n ${C3_FLAGS:-} ]] && refuse "O4=0.01 is the built value: C3_FLAGS must be empty"
if (( STEPS > 0 && STEPS % 3 == 0 )); then CKPT=$(( STEPS / 3 )); else CKPT=$STEPS; fi
ORCH_REC=${RSR_ORCH_ROOT:-} ; OUT_REC=${OUT:-} ; SUFFIX_REC=${JOB_SUFFIX:-}
: ${OUT:=runs/rsr-run1}
C1=(${=C1_FLAGS:-}) ; C2=(${=C2_FLAGS:-}) ; C3=(${=C3_FLAGS:-})

RUN_SHA=$($GIT rev-parse HEAD); rc_sha=$?
PORCELAIN=$($GIT status --porcelain); rc_status=$?
if (( rc_sha != 0 || rc_status != 0 )) || [[ -n $PORCELAIN ]]; then
  refuse "not a clean committed head (rc_sha=$rc_sha rc_status=$rc_status)"
fi
mkdir -p $OUT/logs; rc_mkdir=$?
(( rc_mkdir == 0 )) || refuse "mkdir $OUT/logs rc=$rc_mkdir"
LAUNCHED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
.venv/bin/python -c 'import json, sys
a = sys.argv[2:]
d = dict(zip(a[0::2], a[1::2]))
d["dirty"] = False
d["porcelain_lines"] = 0
with open(sys.argv[1], "w") as f:
    json.dump(d, f)
    f.write("\n")' $OUT/launch-git.json \
  run_sha "$RUN_SHA" gamma "$GAMMA" wd "$WD" steps "$STEPS" ckpt "$CKPT" \
  ckpt_rule "STEPS/3 if STEPS>0 and STEPS%3==0 else STEPS (A1.8 m-9)" o5 "$O5" \
  c1_flags "${C1_FLAGS:-}" c2_flags "${C2_FLAGS:-}" c3_flags "${C3_FLAGS:-}" \
  rsr_orch_root "$ORCH_REC" out "$OUT_REC" job_suffix "$SUFFIX_REC" \
  dry_run "${DRY_RUN:-}" launched_utc "$LAUNCHED_UTC"; rc_json=$?
(( rc_json == 0 )) || refuse "writing $OUT/launch-git.json rc=$rc_json"

typeset -A SHA
SHA[0]=0ee3f8a69b507d927c631eb85116e1cd9739ee4472c44eb7f145d739a118da60
SHA[1]=dadd1e08a3849c1211c8479394df4060f91cd2a6e188b97701b783f2068a3da8
SHA[2]=b507ebc573a54316f9d3c20337f49e664bb9c414388af925c5a8e87efd6b63b8
SLOT=(env PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run
      --lane cpu-det --slots 1 --wait 3600 --pinned-sha $RUN_SHA)
RSR=(--policy rsr --policy-override b_enabled=false --policy-override shadow_enabled=false
     --policy-override nu=0 --policy-override beta=1 --policy-override gamma=$GAMMA)

typeset -A pid rcs
for s in 0 1 2; do
  B=$R/.worktrees/fresh-stream/runs/fresh-stream/B/seed$s
  for a in psiC fifo age; do
    case $a in
      psiC) P=("${RSR[@]}") ;;
      fifo) P=(--policy fifo) ;;
      age)  P=("${RSR[@]}" --policy-override psi_override=age_only) ;;
    esac
    k=$a-s$s
    "${SLOT[@]}" --job-id rsr-run1-$k${JOB_SUFFIX:-} -- \
      .venv/bin/python -m rsr.train.loop \
        --d 128 --batch 16 --iters $STEPS --steps-per-stream 48 --memory-slots 16 \
        --lr 0.001 --seed $s --beat-every 1 --ckpt-every $CKPT --device cpu \
        --freeze-transformer "${C1[@]}" "${C3[@]}" \
        --pool-offset 993000 --pool-length 256 --pool-vocab-documents 64 "${C2[@]}" \
        --warm-start $B/ckpt-003000.pt --warm-start-sha256 ${SHA[$s]} \
        --warm-start-manifest /Users/keanooo7/rsr-substrate/2026-09-27/MANIFEST.sha256 \
        --warm-start-root $R --warm-start-source-config $B/result.json \
        "${P[@]}" --out-dir $OUT/$a/seed$s \
      > $OUT/logs/$k.log 2>&1 &
    p=$!; pid[$k]=$p      # measured: pid[$k]=$! stores the literal string "$!" in zsh 5.9
  done
done
for k in ${(ok)pid}; do
  wait ${pid[$k]}; rcs[$k]=$?
  print -r -- ${rcs[$k]} > $OUT/logs/$k.rc
  print -r -- "rc[$k]=${rcs[$k]}"
done
```

**Measured: refusals** (`exitprobe/v4/`; `env -i HOME PATH` plus the listed variables; `GAMMA=0.9
WD=0.01 O5=train` placeholders):

| case | rc | message |
|---|---|---|
| `DRY_RUN=1 STEPS=3000` | **3** | `DRY_RUN=1 needs STEPS=0 (a dry run never trains), got STEPS=3000` |
| `DRY_RUN=1 STEPS=3000` plus a foreign `RSR_ORCH_ROOT`, `OUT`, `JOB_SUFFIX` (the review's case) | **3** | the same |
| `DRY_RUN=1 STEPS=1` | **3** | the same, `got STEPS=1` |
| `DRY_RUN=1 STEPS=0` plus an isolated root, `OUT`, `JOB_SUFFIX` (dirty scratch repo) | **3**, at the clean-head check | the guards passed (a dry run is allowed) |
| leftover `OUT` / `JOB_SUFFIX` / foreign root / `CKPT` given / `STEPS=abc` / `STEPS=0` without `DRY_RUN` / `DRY_RUN=2` / nothing set | **3** each | exactly v3's messages |
| `RSR_ORCH_ROOT` = the shared root (dirty scratch repo) | **3**, at the clean-head check | the guard passed |

No case created an output directory. `runs/rsr-run1` was absent in the worktree afterwards.

**Measured: the JSON encoder** (review MINOR-A3-6).

- The launch ran in a **clean** scratch git repo with a committed `.gitignore` (`.venv`) and
  `.venv` symlinked to the worktree's.
- Settings: `DRY_RUN=1`, `STEPS=0`, and `GAMMA='0.9"q\b'` (a double quote and a backslash).
- Nothing could train. Every slot call failed with rc 1 (`No module named 'orchestrator'`).

| launcher | rc | `launch-git.json` parses? | `gamma` read back |
|---|---|---|---|
| **v4** | 0 | **yes** | `'0.9"q\\b'`, exact; `dirty` `False`, `porcelain_lines` `0` |
| v3 (counterfactual) | 0 | **no**: `JSONDecodeError: Expecting ',' delimiter: line 1 column 109` | — |

**Measured: the nine-job dry run, inside tmux, with the A3b.2 command shape, at `3ef97c8`.**

- Settings: `DRY_RUN=1`, isolated root, `STEPS=0`, `O5=train`, set **after** the `env -u` in the
  pane.
- `rc_tmux` 0, launcher rc 0, 10 s wall.
- All nine `rc[…]=0`.
- Nine job records: `done 0 threads 1 pinned 3ef97c887c98`, `--ckpt-every 0`.
- `launch-git.json` parsed, with `dirty: False`, `porcelain_lines: 0` and `dry_run: '1'`.

The CKPT derivation is unchanged from v3 and is not re-measured here; A3.7's table stands. Under
v4 a dry run can only have `STEPS=0`, so CKPT is 0.

### A3b.2 The tmux launch command (replaces A3.8's), and A3.8's claim corrected

**A3.8's sentence was false.** It said: "If any leaks in from the client environment … the
launcher refuses with rc 3 and nothing starts". That was false for a leaked `DRY_RUN=1`. With
launcher v4 the leaked-`DRY_RUN` case now refuses (A3b.1). In addition, the command below strips
all five variables in two places:

- **on the client**, before tmux starts;
- **inside the pane**, before the launcher.

```zsh
W=/Users/keanooo7/retrieval-successor-retention/.worktrees/rsr-run1-run
env -u RSR_ORCH_ROOT -u OUT -u JOB_SUFFIX -u DRY_RUN -u CKPT \
  /opt/homebrew/bin/tmux -L rsr-run1 new-session -d -s rsr-run1 \
  "cd $W && env -u RSR_ORCH_ROOT -u OUT -u JOB_SUFFIX -u DRY_RUN -u CKPT \
     GAMMA=<O1> WD=<O4> STEPS=<R-1> O5=<O5> C1_FLAGS='<T9>' C2_FLAGS='<T10>' \
     /bin/zsh experiments/rsr-run1/launch_run1.zsh > runs/rsr-run1-launch.log 2>&1; \
   echo \$? > runs/rsr-run1-launch.rc"
rc_tmux=$?
```

**Why both places, measured.** An earlier session on the same tmux socket leaves its environment
in the **server**, which copies it into later sessions even when the client is clean. So the
client-side `env -u` alone is not enough.

The probe was `exitprobe/v4leak/`. The leaked client values were `DRY_RUN=1`, a foreign
`RSR_ORCH_ROOT`, `OUT`, `JOB_SUFFIX=-leak` and `CKPT=999`. The pane printed its own environment:

```
LEAK old form (A3.8): tmux rc=0 :: RSR_ORCH_ROOT=<foreign> OUT=<scratch> JOB_SUFFIX=-leak DRY_RUN=1 CKPT=999
LEAK new form (A3b.2): tmux rc=0 :: RSR_ORCH_ROOT=<unset> OUT=<unset> JOB_SUFFIX=<unset> DRY_RUN=<unset> CKPT=<unset>
LEAK new form, polluted server: tmux rc=0 :: RSR_ORCH_ROOT=<unset> OUT=<unset> JOB_SUFFIX=<unset> DRY_RUN=<unset> CKPT=<unset>
LEAK no env -u, clean client, polluted server: :: RSR_ORCH_ROOT=<foreign> OUT=<scratch> JOB_SUFFIX=-leak DRY_RUN=1 CKPT=999
```

(Paths are shortened to `<foreign>` and `<scratch>`. They are literal in `exitprobe/v4leak/*.out`.)

**The real launcher v4, through tmux, with a leaked client environment, in a dirty scratch repo,
at `STEPS=3000`:**

| form | leaked | launcher rc | message |
|---|---|---|---|
| old A3.8 form | `DRY_RUN`, root, `OUT`, `JOB_SUFFIX` (the review's case) | **3** | `DRY_RUN=1 needs STEPS=0 (a dry run never trains), got STEPS=3000` |
| old A3.8 form | the same plus `CKPT=999` | **3** | `CKPT is derived from STEPS (A1.8 m-9); do not set it` |
| **new A3b.2 form** | the same five | **3**, at the clean-head check | the leaked variables were stripped; only the dirty scratch repo stopped it |

No output directory was created in any case.

**The corrected claim.**

- A leaked `RSR_ORCH_ROOT`, `OUT`, `JOB_SUFFIX`, `DRY_RUN` or `CKPT` cannot reach the launcher
  through the A3b.2 command.
- If the command is not used, launcher v4 still refuses a leaked `DRY_RUN` that would train, and
  the leaked root, `OUT`, `JOB_SUFFIX` and `CKPT` when `DRY_RUN` is unset. In each case it exits 3
  and nothing starts.
- C10 rejects any recorded `dry_run`, foreign root, `OUT` or `JOB_SUFFIX` afterwards (A3.7a, T16).

Review MINOR-A2-5 (`~/.zshenv`) remains not addressed, as A3.8 said.

### A3b.3 Ledger status and exit (replaces A3.11's table; review MINOR-A3-1 and MINOR-A3-2)

**First, A3.11's B2 citation was wrong.** `experiments/b2-psi-probe/run.py:2220` (at
`night/2026-09-30`) is in **`run_size`**, the sizing step (`def` at `:2108`), not phase A. B2 is
mixed:

- phase A (`run_fit`, `:3547`) and phase B (`run_eval`, `:3637` and `:3658`) write **`"failed"`**
  on a non-OK exit;
- only sizing writes `"partial"`.

So B2 is not a precedent for `partial` beyond sizing. I checked each line at that commit.

**What the rules actually say:**

- `scripts/ledger.py:101-105`. The statuses are `ok`, `partial`, `crashed`, `did_not_run` and
  `failed`. Only `failed` is defined: a completed run whose own pre-registered check failed,
  **exit 1**.
- `R-2026-09-22-inert-exit-5`, mapping table:
  - liveness **inconclusive → `1 FAIL`**, "liveness not demonstrated; the ledger records the band
    label";
  - **"a control fails / … / measurement raised → `3 DID_NOT_RUN` — the measurement is not
    valid"**;
  - inert → `5`.
- `src/rsr/exit_codes.py`: exit 3 is named `DID_NOT_RUN`.

**The mapping, now aligned with the ruling and with the exit codes' own names.** This proposes a
better mapping than A3.11's within `ledger.py`'s existing vocabulary, as the PM allowed.

| situation | exit | ledger status | required ledger note |
|---|---|---|---|
| everything passes | 0 | `ok` | — |
| refusal before any measurement (environment, C0, missing `launch-git.json`) | 3 | `did_not_run` | the refusal |
| an arm's rc 1 with **no heartbeat** (an exception before training) | 3 | `did_not_run` | `ARM_DID_NOT_RUN: <arm>` |
| an arm's rc 3 (slot refusal or usage error) | 3 | `did_not_run` | `ARM_DID_NOT_RUN: <arm>` |
| an arm's rc 1 with a **`crash`** line, or 128 + N; C6 fails | 3 | `crashed` | `ARM_CRASHED: <arm>` |
| **an arm's liveness band is `inconclusive`** (rc 1, `completed` footer; or C5 reads `inconclusive`) | **1** | **`failed`** | `LIVENESS_INCONCLUSIVE: <arm>` |
| a C5 band of `inert` (rc 5) | **5** | `partial` | `INERT: <arm>` (quarantine per ruling) |
| **a control fails after training** (C1–C4, C7, C8, C10) | **3** | **`did_not_run`** | `CONTROL_FAILED: <list>` plus `steps_done` as recorded |
| C9/C9b fails | unchanged | unchanged | A3b.4's tags only |

**Why these choices:**

- **A failed control → `did_not_run`, exit 3.**
  - The ruling names this case `3 DID_NOT_RUN`, "the measurement is not valid".
  - `did_not_run` is the one status whose name is the exit code's own (`Exit.DID_NOT_RUN`).
  - The review's objection is that it hides that training ran. It is answered by **two required
    fields**: the `CONTROL_FAILED` note, and `steps_done` / `seeds_actually_run` as actually
    recorded. Neither may be cleared.
  - `partial` (A3.11) is withdrawn for this case. The ledger has no exit-3 status meaning "ran, but
    not valid". `did_not_run` is the ruling's word, and `partial` was only the nearest word.
  - Nothing gates on the difference. `scripts/orchestrator/lanes.py:291` treats every non-`ok`
    status alike.
- **Liveness inconclusive → `failed`, exit 1. This departs from the two options the PM offered**
  (`did_not_run` or `partial`), because an owner ruling already fixes it. R-2026-09-22-inert-exit-5
  maps the inconclusive band to `1 FAIL`, "liveness not demonstrated". `ledger.py` defines
  `failed` as exactly exit 1, a completed run whose pre-registered check failed. Liveness (C5) is
  such a check. Mapping it to 3 would contradict the ruling's code. The PM may object; the ruling
  is the owner's, and only the owner can change it.
- **Inert → `partial`, exit 5** (unchanged from A3.11).
  - No status names inert. The ruling says inert is *not* "the check caught a failure", so not
    `failed`. It is not invalid either, so not `did_not_run`. The run ran, and its results exist
    but are not readable.
  - The routing is carried by the exit code, 5, which the ruling defines.
- **An arm that crashed → `crashed`.** This is the ledger's own word for it.

`measure-seed` takes the worst arm by this table, using the A3b.5 order. The harness changes are
TO-CONFIRM T15.

### A3b.4 Review MINOR-A3-3: C9/C9b tagging covers the oracle accuracy column and the ledger

When C9 or C9b fails, these all carry the tag **NOT READ (C9/C9b failed)**:

- **in RESULTS:**
  - the oracle's column in every held-out accuracy table, every bucket;
  - every `oracle_regret.*` line;
  - every `headroom_share.*` line;
- **in the ledger:** every row that is the oracle's accuracy level (`acc.<bucket>.oracle`), every
  `oracle_regret.*` row and every `headroom_share.*` row. Each carries the fields
  `"not_read": true` and `"not_read_reason": "C9/C9b failed"`. The values stay recorded; they are
  not deleted.

Everything else in A3.6 stands: no exit or status change, and no other number touched.
TO-CONFIRM T15, which covers the ledger as well as RESULTS.

### A3b.5 Review MINOR-A3-4: `finalize` takes its status and its exit from the same worst seed

- **The worst seed** is the seed whose exit ranks highest in the order **5 > 3 > 1 > 0**.
  - 5 (inert) dominates, because the ruling quarantines it.
  - 3 (a measurement not valid) ranks above 1 (a valid measurement whose check failed).
  - Ties go to the **lowest seed number**.
- **`finalize`'s exit and status are both that seed's.**
- **Example:** seed 0 `crashed` (3) and seed 1 inert (5) give exit 5 with status `partial`, both
  from seed 1.

This replaces A3.13's "its exit is the worst seed's … its status is that seed's". At `58d25cc` the
harness takes the *first* non-OK seed's status (`run.py:2317-2319`). TO-CONFIRM T15.

### A3b.6 Review MINOR-A3-5: T7's bound, stated exactly (replaces A3.15's "k ≤ 46")

The oracle's argmin uses `discounted_demand[t+1][i] = γ^(q−t−1)` for an assert `i` still pending
at a decision step `t`, with query at `q`. The exponent `q − t − 1` is bounded twice by
construction:

- **≤ 38 by the gap:** `max_gap = 40` (`SyntheticConfig.max_gap`), and a decision on a slot comes
  at `t ≥ a + 1`, so `q − t − 1 ≤ (a + 40) − (a + 1) − 1 = 38`;
- **≤ 30 by the stream:** decisions happen only once memory is full, at `t ≥ M = 16`, and
  `q ≤ S − 1 = 47`, so `q − t − 1 ≤ 30`.

Measured on the pool (`t7_exponent.py`, model-free, rc 0):

```
seed 0: pool max gap 40 (cfg.max_gap 40); max exponent q-t-1 at a full-memory decision 30; bound max_gap-2 = 38
seed 1: pool max gap 40 (cfg.max_gap 40); max exponent q-t-1 at a full-memory decision 30; bound max_gap-2 = 38
seed 2: pool max gap 40 (cfg.max_gap 40); max exponent q-t-1 at a full-memory decision 30; bound max_gap-2 = 38
```

**T7, restated.**

- **A sufficient condition** for γ-invariance in float64: `γ^30` is a normal float64. Then every
  pending demand is a distinct, correctly ordered nonzero number, which holds for
  **γ ≥ 5.56e-11** (`(2.2250738585072014e-308)^(1/30)`).
- **Observed:** decisions are identical down to γ = 1e-11. The review measured 1e-10 and 1e-11,
  and A3.15 covers 1e-8 and above.
- **The failure point:** they differ at 1e-12 (24 decisions, 1 document on the seed-0 pool) and at
  1e-300. Underflow changes a decision only when an underflowed pending demand ties at 0 with a
  lower-index zero-demand slot.

γ_o = 0.9 and C9's 0.5 are far inside the condition.

### A3b.7 TO-CONFIRM, updated

- **T14:** the harness's byte-identity check reads **A3b.1's** heading. The committed launcher's
  sha256 is `5cd6e5fd712ffab036d87ea6481b3220a3669f071ba955255511fc840b3f7946`.
- **T15:**
  - the A3b.3 statuses, exits and required notes;
  - the A3b.4 tags in RESULTS **and** the ledger;
  - the A3b.5 worst-seed rule;
  - the C9 result string re-pointed from "(A2.2)" to A3.6 (review nit).
- **T16:** C10 reads `launch-git.json`'s `dry_run`, `rsr_orch_root`, `out`, `job_suffix`, `ckpt`
  and `ckpt_rule`, and **rejects any `dry_run`**.
- **Standing:** T9, T10, and the four owner slots, in Amendment 4.

### A3b.8 What this author has additionally seen (declared)

- launcher v4 refusal probes;
- tmux environment dumps;
- a JSON probe in which no job could start;
- the `--iters 0` nine-job dry run (rcs and records only);
- the pool's assert/query structure, with no model.

No EVAL document has gone through any model, and no φ has been trained.


---

## AMENDMENT 3c (2026-10-01): srep_norm_reg_weight 0.0 in every arm (launcher v5); tie rule; cost and C5 corrections

**Written 2026-10-01 by a model** (Claude Opus 5.5, the PREREG author, on the PM's instruction).
**Not written by Brendan.** It is committed alone, before any run-1 step and before any EVAL
document has gone through any model. Everything above this heading is unedited.

**What prompted it.** The harness rebase hand-back: `feat/rsr-run1-harness` at `6267fd6`, which
merges `feat/rsr-run1-ready` at `c02b16d`. Its addendum is in
`~/Documents/RSR-2026-09-30-day/reports/RUN1-HARNESS-build.md`, under "srep_norm_reg_weight",
"Cost reconciliation (A3.2)" and "Still open". The decision in A3c.1 is the PM's.

**🔴 The owner slots stay BLANK.** O1, O4, R-1 and O5 are not filled here; Amendment 4 fills them.
In the probes below, `GAMMA=0.9`, `WD=0.01` and both O5 values are placeholders.

**Precedence.** Where this amendment and anything above it disagree, this amendment governs.
Superseded:

- §6's `srep_norm_reg_weight` row and §13's fourth code observation ("The CLI has no
  `--srep-norm-reg-weight`") → A3c.1;
- A3b.1's launcher → A3c.1;
- A3.2's cost figures → A3c.3;
- A3b.3's per-seed rule, which had no tie rule for arms → A3c.2;
- A3b.3's liveness row → A3c.4, which completes it;
- A3.7a's C10 list → A3c.5, which extends it.

**Also corrected, for accuracy.** A1.8 m-6 says the heartbeat's `attribution` is `None` in every
arm. That described `211d6da`. At `c02b16d` the beat's attribution is set for RSR arms (harness
addendum, adapter 11). With `b` off and ν = 0, §3.4's decomposition still has a single term.

### A3c.1 Launcher, version 5

**`srep_norm_reg_weight` = 0.0 in every arm (the PM's decision).**

**§6's premise is false at the run's code.**

- §6 said: "`0.01` (the CLI has no flag; arm B trained at `0.0`)".
- `c02b16d` has the flag: `--srep-norm-reg-weight` (`loop.py:1833`, passed at `:1881`). `train()`
  validates it as finite and ≥ 0 (`:1062-1065`) and stamps it into `frozen` and the config hash.
- Without the flag, `train()` resolves `None` to TGConfig's **0.01**. The harness e2e recorded
  0.01 in all three arms.
- `loop.py` is byte-identical at `c02b16d` and `6267fd6` (`git diff --quiet`, rc 0).

**Why 0.0:**

1. **It changes no update.** Under freeze, `objective = None if freeze_transformer else loss`
   (`loop.py:1398`), so the hinge reaches no parameter at any value. It enters only the `loss` and
   `loss_sum` fields: `loss = (lm + w_srep * hinge) if w_srep else lm` (`:1325`). The analysis
   reads `loss_lm` (§13), which does not contain it.
2. **It matches the substrate.** Arm B trained at `srep_norm_reg_weight` 0.0 (its `result.json`
   config; fresh-stream manifest). At 0.0, the run-1 `loss` field equals `loss_lm` and is
   interpretable. At 0.01 it would carry about +2.2 nats of hinge (arm B's last
   `loss_srep_hinge` was 222.9).
3. **No silent 0.01 for a successor.** A later "training continues" arm (D2(b)) built from this
   command line would otherwise inherit 0.01 against the substrate's 0.0, and optimise a different
   objective from the one arm B trained under, without anyone choosing it.

**The change from v4** is one argument in the training command, `--srep-norm-reg-weight 0.0`, on
its own line after `--freeze-transformer …`, plus two header comment lines. Nothing else changed.

**Coordination.** `experiments/rsr-run1/launch_run1.zsh` must be **byte-identical** to the text
below. The harness's `LAUNCHER_HEADING` becomes **"### A3c.1 Launcher, version 5"**, this
subsection's heading. `cli_argv` must include the new argument at the same position, token for
token. Both are TO-CONFIRM T14 and T17.

**sha256 by the harness extraction rule** (`t.index(heading)`, then
`t.index("```zsh\n", i) + len("```zsh\n")` up to `"\n```\n"`, plus `"\n"`):
**`5cbaa14c26c72fc69fb5a96cd95fb950fdd9285862afc76cb790d226d5c22152`**.

```zsh
#!/bin/zsh
# RSR run 1 launch, version 5 (PREREG Amendment 3c, A3c.1; supersedes A3b.1's text).
# Run with /bin/zsh INSIDE tmux (A2.4), from the dedicated run worktree at the final
# reviewed commit. Owner slots: GAMMA (O1), WD (O4), STEPS (R-1), O5 (eval|train).
# CKPT is DERIVED here from STEPS (A1.8 m-9), never read from input. Code-change flags:
# C1_FLAGS (C-1), C2_FLAGS (C-2), C3_FLAGS (C-3, only if O4 != 0.01). Every arm passes
# --srep-norm-reg-weight 0.0, arm B's training value (A3c.1).
# RSR_ORCH_ROOT must be unset or the shared root, and OUT / JOB_SUFFIX unset, unless
# DRY_RUN=1, and DRY_RUN=1 requires STEPS=0 (a dry run never trains). All four are
# recorded in launch-git.json, written by a JSON encoder. OUT defaults to runs/rsr-run1.
# Any unfilled or inconsistent slot exits 3 ("did not run"). One rc per arm x seed.
R=/Users/keanooo7/retrieval-successor-retention
GIT=/opt/homebrew/bin/git
refuse() { print -u2 "DID NOT RUN: $1"; exit 3 }
[[ -z ${DRY_RUN:-} || $DRY_RUN == 1 ]] || refuse "DRY_RUN must be unset or 1, got '$DRY_RUN'"
if [[ -z ${DRY_RUN:-} ]]; then
  [[ -z ${RSR_ORCH_ROOT:-} || ${RSR_ORCH_ROOT:A} == ${R:A} ]] \
    || refuse "RSR_ORCH_ROOT=$RSR_ORCH_ROOT is not the shared root $R"
  [[ -z ${OUT:-} ]] || refuse "OUT=$OUT is set (a dry-run hook)"
  [[ -z ${JOB_SUFFIX:-} ]] || refuse "JOB_SUFFIX=$JOB_SUFFIX is set (a dry-run hook)"
fi
[[ -z ${CKPT:-} ]] || refuse "CKPT is derived from STEPS (A1.8 m-9); do not set it"
for v in GAMMA WD STEPS O5; do [[ -n ${(P)v} ]] || refuse "slot $v unfilled"; done
[[ $STEPS == <-> ]] || refuse "STEPS must be a non-negative integer, got '$STEPS'"
[[ -n ${DRY_RUN:-} ]] && (( STEPS != 0 )) && refuse "DRY_RUN=1 needs STEPS=0 (a dry run never trains), got STEPS=$STEPS"
(( STEPS >= 1 )) || [[ -n ${DRY_RUN:-} ]] || refuse "STEPS must be >= 1"
[[ $O5 == eval || $O5 == train ]] || refuse "O5 must be eval or train, got '$O5'"
[[ $O5 == eval && -z ${C1_FLAGS:-} ]] && refuse "O5=eval needs C1_FLAGS (C-1's flag)"
[[ $O5 == train && -n ${C1_FLAGS:-} ]] && refuse "O5=train: C1_FLAGS must be empty"
[[ $WD != 0.01 && -z ${C3_FLAGS:-} ]] && refuse "O4=$WD needs C3_FLAGS (code change C-3)"
[[ $WD == 0.01 && -n ${C3_FLAGS:-} ]] && refuse "O4=0.01 is the built value: C3_FLAGS must be empty"
if (( STEPS > 0 && STEPS % 3 == 0 )); then CKPT=$(( STEPS / 3 )); else CKPT=$STEPS; fi
ORCH_REC=${RSR_ORCH_ROOT:-} ; OUT_REC=${OUT:-} ; SUFFIX_REC=${JOB_SUFFIX:-}
: ${OUT:=runs/rsr-run1}
C1=(${=C1_FLAGS:-}) ; C2=(${=C2_FLAGS:-}) ; C3=(${=C3_FLAGS:-})

RUN_SHA=$($GIT rev-parse HEAD); rc_sha=$?
PORCELAIN=$($GIT status --porcelain); rc_status=$?
if (( rc_sha != 0 || rc_status != 0 )) || [[ -n $PORCELAIN ]]; then
  refuse "not a clean committed head (rc_sha=$rc_sha rc_status=$rc_status)"
fi
mkdir -p $OUT/logs; rc_mkdir=$?
(( rc_mkdir == 0 )) || refuse "mkdir $OUT/logs rc=$rc_mkdir"
LAUNCHED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
.venv/bin/python -c 'import json, sys
a = sys.argv[2:]
d = dict(zip(a[0::2], a[1::2]))
d["dirty"] = False
d["porcelain_lines"] = 0
with open(sys.argv[1], "w") as f:
    json.dump(d, f)
    f.write("\n")' $OUT/launch-git.json \
  run_sha "$RUN_SHA" gamma "$GAMMA" wd "$WD" steps "$STEPS" ckpt "$CKPT" \
  ckpt_rule "STEPS/3 if STEPS>0 and STEPS%3==0 else STEPS (A1.8 m-9)" o5 "$O5" \
  c1_flags "${C1_FLAGS:-}" c2_flags "${C2_FLAGS:-}" c3_flags "${C3_FLAGS:-}" \
  rsr_orch_root "$ORCH_REC" out "$OUT_REC" job_suffix "$SUFFIX_REC" \
  dry_run "${DRY_RUN:-}" launched_utc "$LAUNCHED_UTC"; rc_json=$?
(( rc_json == 0 )) || refuse "writing $OUT/launch-git.json rc=$rc_json"

typeset -A SHA
SHA[0]=0ee3f8a69b507d927c631eb85116e1cd9739ee4472c44eb7f145d739a118da60
SHA[1]=dadd1e08a3849c1211c8479394df4060f91cd2a6e188b97701b783f2068a3da8
SHA[2]=b507ebc573a54316f9d3c20337f49e664bb9c414388af925c5a8e87efd6b63b8
SLOT=(env PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run
      --lane cpu-det --slots 1 --wait 3600 --pinned-sha $RUN_SHA)
RSR=(--policy rsr --policy-override b_enabled=false --policy-override shadow_enabled=false
     --policy-override nu=0 --policy-override beta=1 --policy-override gamma=$GAMMA)

typeset -A pid rcs
for s in 0 1 2; do
  B=$R/.worktrees/fresh-stream/runs/fresh-stream/B/seed$s
  for a in psiC fifo age; do
    case $a in
      psiC) P=("${RSR[@]}") ;;
      fifo) P=(--policy fifo) ;;
      age)  P=("${RSR[@]}" --policy-override psi_override=age_only) ;;
    esac
    k=$a-s$s
    "${SLOT[@]}" --job-id rsr-run1-$k${JOB_SUFFIX:-} -- \
      .venv/bin/python -m rsr.train.loop \
        --d 128 --batch 16 --iters $STEPS --steps-per-stream 48 --memory-slots 16 \
        --lr 0.001 --seed $s --beat-every 1 --ckpt-every $CKPT --device cpu \
        --freeze-transformer "${C1[@]}" "${C3[@]}" \
        --srep-norm-reg-weight 0.0 \
        --pool-offset 993000 --pool-length 256 --pool-vocab-documents 64 "${C2[@]}" \
        --warm-start $B/ckpt-003000.pt --warm-start-sha256 ${SHA[$s]} \
        --warm-start-manifest /Users/keanooo7/rsr-substrate/2026-09-27/MANIFEST.sha256 \
        --warm-start-root $R --warm-start-source-config $B/result.json \
        "${P[@]}" --out-dir $OUT/$a/seed$s \
      > $OUT/logs/$k.log 2>&1 &
    p=$!; pid[$k]=$p      # measured: pid[$k]=$! stores the literal string "$!" in zsh 5.9
  done
done
for k in ${(ok)pid}; do
  wait ${pid[$k]}; rcs[$k]=$?
  print -r -- ${rcs[$k]} > $OUT/logs/$k.rc
  print -r -- "rc[$k]=${rcs[$k]}"
done
```

**Measured: the refusal matrix, re-run against v5.**

- Setup: `env -i HOME PATH` plus the listed variables; `GAMMA=0.9 WD=0.01 STEPS=3000 O5=train`
  unless stated. Logs in `exitprobe/v5/`.
- No case created an output directory, and `runs/rsr-run1` stayed absent.

| case | rc | message |
|---|---|---|
| `DRY_RUN=1 STEPS=3000` | 3 | `DRY_RUN=1 needs STEPS=0 (a dry run never trains), got STEPS=3000` |
| the same, plus a foreign root, `OUT` and `JOB_SUFFIX` | 3 | the same |
| `DRY_RUN=1 STEPS=1` | 3 | the same, `got STEPS=1` |
| `DRY_RUN=1 STEPS=0` with an isolated root, `OUT` and `JOB_SUFFIX`, in a dirty scratch repo | 3 | `not a clean committed head`. The guards passed |
| leftover `OUT` | 3 | `OUT=… is set (a dry-run hook)` |
| leftover `JOB_SUFFIX` | 3 | `JOB_SUFFIX=-x is set (a dry-run hook)` |
| foreign `RSR_ORCH_ROOT` | 3 | `… is not the shared root /Users/keanooo7/retrieval-successor-retention` |
| `CKPT=1000` given | 3 | `CKPT is derived from STEPS (A1.8 m-9); do not set it` |
| `STEPS=abc` | 3 | `STEPS must be a non-negative integer, got 'abc'` |
| `STEPS=0` without `DRY_RUN` | 3 | `STEPS must be >= 1` |
| `DRY_RUN=2` | 3 | `DRY_RUN must be unset or 1, got '2'` |
| nothing set | 3 | `slot GAMMA unfilled` |
| `O5=maybe` | 3 | `O5 must be eval or train, got 'maybe'` |
| `O5=eval` without `C1_FLAGS` | 3 | `O5=eval needs C1_FLAGS (C-1's flag)` |
| `O5=train` with `C1_FLAGS='--frozen-model-mode eval'` | 3 | `O5=train: C1_FLAGS must be empty` |
| `WD=0.0` without `C3_FLAGS` | 3 | `O4=0.0 needs C3_FLAGS (code change C-3)` |
| `WD=0.01` with `C3_FLAGS` | 3 | `O4=0.01 is the built value: C3_FLAGS must be empty` |
| `RSR_ORCH_ROOT` = the shared root, in a dirty scratch repo | 3 | `not a clean committed head`. The guard passed |
| a dirty tree | 3 | `not a clean committed head (rc_sha=0 rc_status=0)` |

**Measured: the nine-job dry run inside tmux, with the A3b.2 command shape, at the code the run
uses.**

- **Where.** My `.worktrees/rsr-run1-prereg` is based on `211d6da`, which has no `--srep-norm-reg-weight`.
  So the dry run ran from a new detached worktree, `.worktrees/rsr-run1-prereg-v5dry`, at the
  harness head `6267fd6`, with its own `.venv` (`uv sync --frozen --extra dev`, rc 0; `rsr.__file__`
  under it).
- **The launcher** was the scratch v5 file, byte-identical to the text above.
- **The command** was `env -u RSR_ORCH_ROOT -u OUT -u JOB_SUFFIX -u DRY_RUN -u CKPT` on the client
  and in the pane, then `DRY_RUN=1`, an isolated root, `OUT` and `JOB_SUFFIX` set after the strip,
  `STEPS=0`.
- **Clean before and after:** the worktree's `git status --porcelain` was empty.
- **It was run twice**, once for each O5 value. C-1 exists at `6267fd6`, so `O5=eval` could run
  with `C1_FLAGS='--frozen-model-mode eval'`.

| | O5 = train | O5 = eval |
|---|---|---|
| `rc_tmux` / launcher rc | 0 / 0 (14 s) | 0 / 0 (10 s) |
| `rc[…]=0` lines | 9 of 9 | 9 of 9 |
| job records | 9 × `done 0`, `threads 1`, `pinned 6267fd645698` | the same |
| **job command `--srep-norm-reg-weight`** | **`0.0` in all 9** | **`0.0` in all 9** |
| **recorded `config.srep_norm_reg_weight`** | **`0.0` in all 9** | **`0.0` in all 9** |
| recorded `frozen_model_mode` | `train (dropout live)` × 9 | `eval (dropout off)` × 9; labels `+frozen[eval]` |
| `launch-git.json` | `run_sha 6267fd64…`, `o5 train`, `c1_flags ""`, `dry_run "1"`, `dirty False` | `o5 eval`, `c1_flags "--frozen-model-mode eval"`, otherwise the same |

The per-arm lines are literal in `launch_dry_v5_{train,eval}.log` and the job records under
`exitprobe/root12/.orchestrator/jobs/rsr-run1-*-dry5{train,eval}.json`.

### A3c.2 Tie rule between arms in a seed (completes A3b.3)

A3b.3 ranks a seed's arms by exit, in the order 5 > 3 > 1 > 0 (A3b.5's order).

- **If two or more arms share the seed's worst exit, the seed's status and note are those of the
  first such arm in the order psiC, fifo, age.**
- Example: psiC `crashed` (3) and fifo `did_not_run` (3) give the seed `crashed`. The note lists
  every non-OK arm anyway (`ARM_CRASHED: psiC`, `ARM_DID_NOT_RUN: fifo`), so nothing is hidden.
- This is what the harness does at `6267fd6` (its addendum, "Still open"). The rule across seeds
  is unchanged: ties go to the lowest seed (A3b.5).

### A3c.3 A3.2's cost, corrected

The harness report's "about 1.8 CPU-h" was an arithmetic error, which its author confirmed. The
corrected figure, `eval_forward` against `step_forward` for the held-out RSR decision context:

- per seed ≈ 109.3 (psiC) + 109.9 (age) + 216.7 (sz_psiC) + 216.7 (sz_age, assumed equal to
  sz_psiC, not measured) + 438.4 (four snapshot passes) ≈ **1,091 s**;
- over three seeds ≈ 3,273 s ≈ **0.91 CPU-h**.

This is extrapolated from **one batch per pass** (the T5 table: psiC 1.7108 s, fifo 0.8571 s, age
1.7156 s, sz_psiC 2.5410 s, sz_fifo 0.8483 s per 16-document batch). The choice in A3.2 was not
made on cost and stands.

### A3c.4 Inconclusive liveness read only by C5, with rc 0 (completes A3b.3)

R-2026-09-22-inert-exit-5, line 21: "inconclusive — otherwise (the 0.01–0.1 band) | `1 FAIL` —
liveness not demonstrated; the ledger records the band label. Never `0`, never `5`."

- **If C5 reads an `inconclusive` band for an arm whose training rc is 0**, that arm is
  **`failed`, exit 1**, with the note `LIVENESS_INCONCLUSIVE: <arm>`. This is the same as the rc 1
  case in A3b.3.
- `train()` maps an inconclusive band to rc 1 itself (`liveness_exit`), so this case should not
  arise. It is registered so that a mismatch between the two readings cannot fall through to `ok`.
- This is the harness's behaviour at `6267fd6`.

### A3c.5 C10 records `srep_norm_reg_weight` and requires 0.0 (extends A3.7a)

- For every arm × seed job, C10 records `config.srep_norm_reg_weight` from the job's result, and
  the token after `--srep-norm-reg-weight` in the job record's `cmd`.
- It **requires both to equal 0.0**. Any other value, or a missing value or token, is a C10
  failure: exit 3, `did_not_run`, `CONTROL_FAILED: [C10]` (A3b.3).
- 0.0 is this PREREG's registered value (A3c.1), not an owner slot. C10 compares it with the
  constant registered here.

### A3c.6 TO-CONFIRM, updated

- **T14:** `LAUNCHER_HEADING` = "### A3c.1 Launcher, version 5". The committed launcher's sha256 is
  `5cbaa14c26c72fc69fb5a96cd95fb950fdd9285862afc76cb790d226d5c22152`.
- **T17:** `cli_argv` includes `--srep-norm-reg-weight 0.0` at the launcher's position, for 3 seeds
  × 3 arms × both O5 values. C10 implements A3c.5. The e2e then records 0.0, not 0.01. A
  seed's notes list **every** non-OK arm, not only the arm that sets the status (A3c.2).
- **Standing:** T9 is confirmed by the harness at `c02b16d` (`--frozen-model-mode eval`), and T10
  likewise (`C2_FLAGS` empty). Both are filled at launch per A3b.2. The four owner slots come in
  Amendment 4.

### A3c.7 What this author has additionally seen (declared)

- The v5 refusal probes.
- Two `--iters 0` nine-job dry runs at `6267fd6`. Only rcs, job records, recorded configs and
  labels were read; the liveness numbers were not.
- **Left in place, not deleted:** the new detached worktree `.worktrees/rsr-run1-prereg-v5dry`, at
  `6267fd6`, clean.
- No EVAL document has gone through any model, and no φ has been trained.
