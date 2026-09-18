# cycle-trained-alpha-pair — training the TG on the synthetic corpus **collapses its gestalt space to one direction**, so real cross-attention is uniform to 1.3e-3 nats and neither `r_i` nor LOO can separate the planted pair. The falsifier dies, and not for the reason it proposes

- **Falsifier.** *"`r_i`'s within-pair blindness is an artifact of an untrained attention
  head. In a **trained** TG, two gestalts at high cosine receive measurably different
  cross-attention when the query needs one of them, so §3.2.1's target does discriminate."*
- **Verdict: FALSIFIED**, on three independent limbs.
  1. **The trained model separates nothing, because every gestalt is the same vector.**
     After 300 iterations of `rsr.train.loop.train` at its documented defaults, all-pairs
     gestalt cosine on real corpus sentences is **0.999972 ± 0.000004** (`REALCORPUS_trained_allpairs_cos_mean`),
     participation-ratio effective dimension **2.62 ± 1.20** out of `d = 128`, per-dimension
     sd **4.25e-4**, and **100 %** of sentence pairs sit at `cos ≥ 0.9`. The **untrained**
     model on the same sentences is **0.505 ± 0.083** with effective dimension **10.55 ± 2.18**
     and only **6.92 % ± 3.78 %** of pairs at `cos ≥ 0.9`. Training *destroyed* the
     distinction the falsifier needed.
  2. **Real cross-attention is therefore uniform by arithmetic.** KL from uniform over the
     16 slots is **0.001301 ± 0.000381** nats against a maximum of `ln 16 = 2.7726`;
     `α(needed) = 0.062479 ± 0.000068` vs `α(other) = 0.062469 ± 0.000034` vs
     `α(distractor) = 0.062504 ± 0.000003`, all at `1/M = 0.0625`. `r_i` inherits it:
     `r_i(needed)/r_i(other) = 1.000637 ± 0.001470`.
  3. **Training did not move the discrimination rate off chance — paired, null 0.**
     `PAIRED_trained_minus_untrained_specific_disc_ri_gated` = **−0.007500 ± 0.040927**;
     the topic arm **+0.009167 ± 0.012829**. Trained `r_i` discrimination is
     **0.510833 ± 0.025166** (specific) and **0.513333 ± 0.012829** (topic) against
     **chance 0.500**.
- **LOO does not rescue it, which is the strong new statement.** In a trained TG, `r_i`
  and LOO **agree — both are blind**. On real corpus query events LOO picks the paired
  assert's slot over a random slot at **0.456296 ± 0.008981** (chance 0.5), and the LOO
  Δ next-sentence NLL is **−1.20e-07 ± 2.68e-08 nats** (`REALCORPUS_trained_loo_delta_assert`). Cycle 10's proxy/LOO *disagreement* was not a
  property of the architecture: cycle 10's LOO was measured on a readout that had oracle
  access to the needed detail, and `α` never did. §7 below.
- **Ledger:** `runs/cycle-trained-alpha-pair/ledger.json` — **493 rows; every number in
  this file is one of them.** 19 statistic rows have `sd == 0.0`, all 19 classified by a
  classifier running in the same process; **`sd_zero_rows_unclassified` is `[]`** (§9).
- **Provenance stamped by the ledger itself:** `git_sha 2420eb2` (the brief's baseline),
  `git_dirty: true` — that is **this untracked run directory only**.
  `/opt/homebrew/bin/git status --porcelain` shows nothing but
  `?? runs/cycle-trained-alpha-pair/`; `git diff --stat HEAD` is empty. `src/`, `tests/`,
  `docs/` untouched; nothing committed. Python 3.12.13, macOS-26.6.2-arm64, **CPU
  throughout** — **MPS was not used anywhere in this cycle**, so its 3.18e-07 run-to-run
  loss floor is not in play. The measurement process is single-threaded
  (`torch.set_num_threads(1)`, row `device`); the twelve training runs used torch's default
  CPU threading and were run in parallel, which affects wall time only.
- **Brief premises re-verified, not trusted.** `pytest tests/test_fidelity.py
  tests/test_reduction.py -q` → **44 passed, 0 failed, 0 skipped**. `TGModel.forward`
  takes `capture: bool = False` and returns `StepOutput.cross_attention`, a tuple of
  **six** `[B, H, L_tok, M]` tensors (correction 18 holds in the code). It does **not**
  return `W_O v` or `g_mem`; both are reconstructed from the live parameters —
  `blk.cross_attn.value(mem_kv)` contracted with `blk.cross_attn.attn_out_proj.kernel`
  (`[H, dh, D]`, bias excluded because it is not part of `W_O^(l,h) v`), and
  `blk.memory_gate`. `train(..., vocab=None)` derives **V = 160**, not 50257.
- **Labelled bypasses, all in the ledger** (`bypasses_labelled`): `d = 128`, `H = 2`,
  `N = 12`, six cross layers, `M = 16`, `iters = 300`, `lr = 1e-3`, `batch = 16`,
  `steps_per_stream = 48`. These are `TGConfig` / `rsr.train.loop.train` defaults, not
  registry constants. **No `ν`, `γ_b`, `b_max` or `τ` is used or reported anywhere in this
  cycle** — no RSR eviction rule is exercised, so the registry is never consulted for one.
- **§13.** One synthetic generator (`rsr.data.synthetic`, V = 160, mean 4.14 words/sentence).
  **No PG-19, no real text, no corpus of any kind.** Every rate here is dimensionless and
  internal to this generator; anything about real text is **conjecture this experiment does
  not establish**. Row `section_13_scope_statement`, written before any number.

---

## 1 — The brief contains an error, and it is the central one

The brief asks me to *"run cycle 10's planted pair through it and measure whether `α` on
the two members separates when the query needs one"*, anchored on cycle 10's headline
`r_i = 0.50300 ± 0.03493` at `wq = 1.0`, a **topic-level** query.

**At a topic-level query that number is 0.5 by an independence argument, for any function
of the input, trained or not.** It is cycle 10's own leak proof (its §5): the identity of
the needed member is drawn from a coin independent of `(memory, c_t)`, so
`P(α_needed > α_other) = 0.5` exactly, before any code runs. Training cannot move it. The
falsifier as posed is therefore **not decidable in the condition it cites** — 0.5 is a
theorem there, not a measurement.

And cycle 10 already reported **untrained** `r_i` discriminating at **1.00000 ± 0.00000**
when the query carried the detail (`wq = 0.9`). So "the untrained head" was **never the
binding constraint**: the untrained head discriminates perfectly when the input identifies
the need, and nothing discriminates when it does not.

Row `BRIEF_ERROR_topic_level_is_an_independence_theorem`.

**How I resolved it.** I ran **two** query conditions, both built from the synthetic
generator's own vocabularies and sentence templates:

| condition | pair | query | is the need identifiable from the input? |
|---|---|---|---|
| **specific** | `Bo recalls the north road.` / `Eli recalls the north road.` — differ in **exactly one token** | `What does Bo recalls?` | **yes** — the query names A's entity |
| **topic** | `Bo recalls the ferry.` / `Bo recalls a brass key.` — share entity **and** predicate | `What does Bo recalls?` | **no** — independent coin (cycle 10's `wq = 1.0`) |

`specific` is the only condition in which training *could* matter. It is the one the
verdict rests on. `topic` is run as the stated control, and comes out at chance exactly as
the theorem requires (`trained_topic_disc_ri_gated` = 0.513333 ± 0.012829).

---

## 2 — Did the model train?

`rsr.train.loop.train(d=128, steps_per_stream=48, batch=16, vocab=None, memory_slots=16,
iters=300, lr=1e-3, device='cpu', seed=s)`, seeds 0/1/2, ~25 min each on CPU.

| | value |
|---|---|
| `V` derived from the corpus (`vocab=None`) | **160** (`V_used`) |
| **true chance** | **`ln(160) = 5.07517`** (`true_chance_ln_V`) — *not* `ln(50257) = 10.8248` |
| loss at step 0 | **4.977347 ± 0.017786** (`train_loss_step0`) |
| loss at step 299 | **0.104307 ± 0.004785** (`train_loss_final`) |
| trajectory, seed 0 (`loss_trajectory_seed0_every25`) | 4.96008 (0) → 0.51682 (25) → 0.25028 (50) → 0.19731 (75) → 0.17076 (100) → 0.14948 (125) → 0.13975 (150) → 0.13080 (175) → 0.12231 (200) → 0.11072 (225) → 0.10482 (250) → 0.10291 (275) → **0.10207 (299)** |

**Yes, it learned — but read the unit.** `rsr.train.loop`'s `step_fn` is
`cross_entropy` over **all 63 next-token positions including PAD targets**, and a
~5-token sentence sits in a length-64 window, so ~58 of 63 positions are trivially
predictable padding. Row `train_loss_unit_caveat`. The masked, real-token-only NLL of a
held-in query sentence is **1.955239 ± 0.007022** for the trained model against
**5.091753 ± 0.044610** untrained (`{trained,untrained}_specific_nll_query`) — a genuine
3.14-nat improvement, so the model did learn the corpus. It simply did not learn to use
the memory (§4).

---

## 3 — The finding: training collapses the gestalt space

Measured on 400 real corpus sentences, per model, 3 seeds (`gestalt_geometry`,
`srep_collapse`):

| | **trained (300 it)** | **untrained** |
|---|---|---|
| all-pairs gestalt cosine, mean | **0.999972 ± 0.000004** | 0.505000 ± 0.083230 |
| … sd within the sample | 0.000028 ± 0.000009 | 0.213390 ± 0.054168 |
| … p50 / p90 / p99 | 0.99998 / 1.0 / 1.0 | 0.480 / **0.876** / 0.972 |
| fraction of pairs at `cos ≥ 0.9` | **1.00000 ± 0.00000** | **0.069169 ± 0.037754** |
| `‖mean gestalt‖` (1.0 ⇒ all identical) | **0.999986 ± 0.000002** | 0.709832 ± 0.059699 |
| per-dimension sd | **0.000425 ± 0.000046** | 0.060833 ± 0.004751 |
| effective dimension (participation ratio, of 128) | **2.62 ± 1.20** | 10.55 ± 2.18 |
| total variance | **2.8e-5** | 0.495 |

Every memory key is the same unit vector. Softmax over identical keys is uniform — that is
arithmetic, not a modelling choice.

**When does it happen? Within the first ten iterations.** A re-run of the identical
trajectory with `ckpt_every=2` (`train-fine-seed*`) and `ckpt_every=20`
(`train-ck-seed*`), 3 seeds, rows `TRAJ_iter*`:

| iter | 2 | 4 | 6 | 10 | 20 | 40 | 80 | 160 | 300 |
|---|---|---|---|---|---|---|---|---|---|
| all-pairs cos | 0.93722 | 0.97810 | 0.98899 | **0.99756** | 0.99953 | 0.99992 | 0.99994 | 0.99997 | 0.99997 |
| effective dim | 9.85 | 10.28 | 9.64 | 9.05 | 6.85 | 4.03 | 1.84 | 1.54 | 2.62 |
| `disc_ri` (chance 0.5) | 0.5044 | 0.5089 | 0.4911 | 0.4956 | 0.4867 | 0.5333 | 0.4956 | 0.5467 | 0.5044 |

Cosine collapses first (by iteration ~10), the effective dimension follows over the next
~80. **`disc_ri` is at chance at every one of the 24 checkpoints measured**, sds in the
ledger; it never rises above 0.56 at any seed-mean.

**It is not a learning-rate artifact.** A `lr = 3e-4` arm, 3 seeds, 300 iterations, final
loss **0.110770 ± 0.004814**: all-pairs cosine **0.999973 ± 0.000008**, effective dimension
**4.35 ± 0.43**, KL from uniform **0.000221 ± 0.000129**, `disc_ri` **0.482222 ± 0.016777**
(rows `LR3e4_*`). Same collapse, same chance.

### A defect in `rsr.train.loop` that this surfaced

`SrepHead` returns `srep_norm_penalty`, the reference's squared hinge on the
**pre-normalisation** norm (`srep_norm_margin = 0.1` around `srep_norm_target = 1.0`), and
the reference adds it to the loss with `srep_norm_reg_weight`. **`rsr.train.loop`'s
`step_fn` is `cross_entropy(logits)` only and drops it.** Measured consequence:
`srep_raw_norm_mean` rises monotonically **2.99 → 4.86 (it 20) → 12.86 (it 80) → 21.32
(it 300)** against the reference's `[0.9, 1.1]` band — **19× outside it**. Row
`REPO_FINDING_train_loop_omits_srep_norm_hinge`. `test_fidelity.py` cannot see this: it
**loads** reference weights and compares the forward pass and gradients, never the training
objective. I am **not** claiming the missing hinge causes the collapse — that is a separate
experiment (§10) — only that the objective actually optimised is not TG's.

---

## 4 — The trained model does not use its memory

| | trained | untrained |
|---|---|---|
| query-sentence NLL, memory on | 1.955239 ± 0.007022 | 5.091753 ± 0.044610 |
| **Δ NLL when the whole memory is switched off** | **0.005581 ± 0.000501** | 0.000769 ± 0.012982 |
| same, on **real** corpus query sentences | 0.005418 ± 0.000378 | 0.000091 ± 0.011579 |
| same, on real corpus **filler** sentences (control) | −0.001440 ± 0.003930 | 0.010916 ± 0.009863 |

Deleting all 16 slots costs **0.0056 nats out of 1.955** — 0.29 % of the loss. The
memory path is live (`use_memory=True`, gestalts are written every step, and the six
learned `g_mem` are 0.98863 / 0.94859 / 0.95867 / 0.97796 / 0.97026 / 0.97511 at seed 0,
the other two seeds in row `REALCORPUS_trained_g_mem_per_cross_layer`) and contributes
essentially nothing. The gates did **not** grow with depth, contrary to [P2] App. C — they
barely moved off `memory_gate_init = 1.0`, which is itself a symptom of the memory path
carrying no gradient worth taking. Correction 17's non-stationarity warning does not bite
here for the same reason.

**In-distribution retrieval control, on 32 real documents × 48 sentences, 450 query events
per seed.** At a real query sentence whose paired assert is still in memory, is the
argmax of `α` that assert's slot?

| | trained | untrained | **chance** |
|---|---|---|---|
| `P(argmax α = the paired assert's slot)` | **0.084444 ± 0.048737** | 0.087407 ± 0.012239 | **0.095499** |
| `P(argmax r_i = …)` | 0.088148 ± 0.035440 | 0.084444 ± 0.008012 | 0.095499 |
| `α` on the assert / on the mean other slot | 0.095419 ± 0.000247 / 0.095508 ± 0.000016 | 0.095345 / 0.095468 | — |
| `P(LOO(assert) > LOO(random other slot))` | **0.456296 ± 0.008981** | 0.472593 ± 0.024579 | **0.5** |

The trained model's attention lands on the sentence it needs **at or slightly below
chance**, and so does LOO. This is the control that makes the rest interpretable: the
model's cross-attention is not retrieval-competent on the task it was trained on.

---

## 5 — Trained `α` on the pair, and `r_i` discrimination vs chance 0.5

400 trials × 3 seeds per cell. `chance = 0.500` for every `disc_*` row; `1/M = 0.0625` for
every argmax row.

**Condition `specific` — the need IS identifiable from the query.**

| | **trained** | **untrained** | **paired Δ (null 0)** |
|---|---|---|---|
| pair gestalt cosine | **0.999989 ± 0.000010** | 0.665829 ± 0.147984 | +0.334161 ± 0.147974 |
| `α(needed)` | **0.062479 ± 0.000068** | 0.062481 ± 0.000031 | — |
| `α(other)` | **0.062469 ± 0.000034** | 0.062427 ± 0.000080 | — |
| `α(distractor)` | 0.062504 ± 0.000003 | 0.062507 ± 0.000008 | — |
| `α` KL from uniform (max `ln 16` = 2.7726) | 0.001301 ± 0.000381 | 0.002067 ± 0.000611 | −0.000766 ± 0.000883 |
| **`disc_α`** | **0.510000 ± 0.029475** | 0.519167 ± 0.020052 | **−0.009167 ± 0.049392** |
| **`disc_r_i` (gated)** | **0.510833 ± 0.025166** | 0.518333 ± 0.018930 | **−0.007500 ± 0.040927** |
| `disc_r_i` (ungated) | 0.510000 ± 0.025372 | 0.518333 ± 0.018930 | −0.008333 ± 0.040646 |
| **`r_i(needed)/r_i(other)`** | **1.000637 ± 0.001470** | 1.003216 ± 0.001101 | −0.002579 ± 0.002551 |
| `α(needed)/α(other)` | 1.000388 ± 0.001597 | 1.001210 ± 0.000826 | −0.000822 ± 0.002383 |
| `P(needed = global r_i argmax)`, chance 0.0625 | 0.054167 ± 0.011815 | 0.058333 ± 0.021262 | — |

**Condition `topic` — cycle 10's `wq = 1.0`, where 0.5 is a theorem.**

| | trained | untrained |
|---|---|---|
| pair gestalt cosine | 0.999973 ± 0.000015 | 0.569850 ± 0.174757 |
| `disc_α` | 0.496667 ± 0.033572 | 0.505833 ± 0.030035 |
| **`disc_r_i` (gated)** | **0.513333 ± 0.012829** | 0.504167 ± 0.012829 |
| `r_i(needed)/r_i(other)` | 1.000698 ± 0.001232 | 1.001420 ± 0.001542 |

**Gated vs raw, as correction 17 requires:** both reported everywhere, and they agree to
within 0.001 in every cell. In the *untrained* arm they are **identical by construction** —
`memory_gate_init = 1.0` on all six layers, so the gate is a uniform scale that `r_i`'s
share normalisation divides out. In the trained arm the gates differ per layer
(0.9886 … 0.9751) and the two forms separate only in the fifth decimal.

**The `eos` reduction of the query-token axis** (§3.2.1's formula has no token axis; TG's
cross-attention does) is reported alongside the token-mean: `trained_specific_disc_alpha_eos`
= 0.500833 ± 0.007638, `trained_topic_disc_alpha_eos` = 0.521667 ± 0.018428. Same answer.

---

## 6 — Can the synthetic corpus express the question? Partly — and training removes the part that could

Row `CORPUS_CANNOT_EXPRESS_THE_QUESTION`, with evidence rather than an opinion.

- **At initialisation, yes.** The untrained gestalt space is non-degenerate: cosine
  0.505 ± 0.083, p90 **0.876**, **6.92 %** of *arbitrary* sentence pairs already at
  `cos ≥ 0.9`, effective dimension 10.55. Cycle 10's premise — a near-duplicate pair at
  `cos ≥ 0.9` — is constructible here.
- **After training, no.** Every pair is at `cos ≥ 0.9` (100 %), so the near-duplicate pair
  stops being a *distinguished* configuration: there is nothing for it to be
  near-duplicate **relative to**.
- **And the attention is degenerate for reasons unrelated to the pair** — the brief's
  item 6 asked exactly this. `α` is uniform to 1.3e-3 nats of KL over 16 slots; `α` on a
  *distractor* (0.062504) is as large as `α` on the needed member (0.062479); the memory is
  worth 0.29 % of the loss; and on real in-distribution query events the `α` argmax finds
  the needed assert at **0.0844 ± 0.0487** against chance **0.0955**.

**This is a real constraint on E0d.** E0d validates `r_i` by correlating it against LOO
Δ next-sentence loss. On this corpus, at this training budget, **both sides of that
correlation are ≈ 0**: `r_i` is uniform and the LOO Δ is order 1e-7 nats (`trained_specific_loo_q_needed` 2.7e-08 ± 5.2e-08). E0d cannot be run
here as a *validation* — it would report a null that is a property of the model's
degeneracy, not of the target's fidelity.

---

## 7 — `r_i` vs LOO in the trained model, and §3.2.1's verdict

The repo's `rsr.metrics.loo.loo_delta_loss` is still a `NotImplementedError` stub (cycle 10
row `repo_loo_harness_status`), but a trained TG now exists, so LOO could be computed in
**its spec definition for the first time**: ablate slot `i` from the memory and measure the
change in the **next sentence's NLL**. Two variants, both on the same trials:

* **`LOO_query`** — Δ NLL of the naturally occurring query sentence. This is §3.2.1's LOO.
* **`LOO_restate`** — Δ NLL of a restatement sentence carrying the needed detail. This is
  the analogue of cycle 10's 2AFC readout, and it is labelled off-distribution: it is the
  variant that has **oracle access to the answer**.

| (trained, `specific`) | value | chance |
|---|---|---|
| `disc_LOO_query`, strict `>` | 0.399167 ± 0.015069 | 0.5 |
| **exact float-tie rate on the LOO Δ** | **0.221667 ± 0.025166** | — |
| **`disc_LOO_query`, tie-split `P(>) + ½P(=)`** | **0.510000 ± 0.027500** | **0.5** |
| `disc_LOO_restate`, strict `>` | 0.422500 ± 0.017500 | 0.5 |
| `LOO_query` Δ, needed / other | **2.7e-08 ± 5.2e-08** / **1.1e-08 ± 2.2e-08** | — |
| sign agreement, `r_i` vs `LOO_query` | 0.510000 ± 0.015612 | 0.5 |

**The strict-`>` rate reading 0.40 is an artifact and I am flagging it rather than banking
it** — the same degenerate-tie artifact cycle 10 flagged on its max-cosine rule. With a
collapsed memory the ablation changes the loss by less than float32 resolution on 22 % of
trials, and a strict `>` scores every tie as a loss for both sides. The tie-split rate is
the honest 2AFC statistic and it is **0.510000 ± 0.027500**. In the untrained arm the tie
rate is **exactly 0** and the strict rate is already at chance (0.506667 ± 0.015069).

**§3.2.1's verdict, applied.** In a trained TG on this corpus the proxy and LOO **do not
disagree — they agree, and both are uninformative.** §3.2.1's rule ("if they disagree, LOO
is truth") therefore does not fire, and the rule is *silent* about the situation that
actually obtains: both sides at chance because the model is degenerate. That is a gap in
the rule worth recording.

**And it reframes cycle 10's §6.** Cycle 10 found LOO at 1.00000 and `r_i` at 0.50300 at a
topic-level query, and read that as `r_i` being blind to a distinction LOO could see. The
correct reading is an **information asymmetry, not an architecture defect**: cycle 10's LOO
was evaluated on a readout that *contained the needed detail*, while `α` is a function of
the pre-query state only, in which the need was an independent coin. A counterfactual on
the downstream outcome can see what a function of the present input provably cannot. No
amount of training closes that gap, and §3.4's redundancy term is not the right response to
it because it is not a defect of the target.

---

## 8 — Re-execution

One command, self-contained, reads the snippet **out of the ledger it wrote** and prints
its own numbers (~35 s, CPU, 200 trials × 3 seeds):

```bash
cd /Users/keanooo7/retrieval-successor-retention && .venv/bin/python -c "
import json
rows = json.load(open('runs/cycle-trained-alpha-pair/ledger.json'))['rows']
exec(next(r['value'] for r in rows if r['key'] == 'reexec_snippet'))
"
```

It imports nothing but `rsr.*` and the stdlib. Its first line is
`CHANCE FOR EVERY disc_* RATE BELOW IS 0.500.  V=160, ln(V)=5.0752, M=16, trials=200 x 3 seeds`.

**It loads `runs/cycle-trained-alpha-pair/train-seed{0,1,2}/ckpt-000300.pt`. `*.pt` is
gitignored, so if the checkpoints are absent it prints
`[checkpoint … ABSENT (*.pt is gitignored) -> training it now, iters=300, cpu, ~17 min]`
and trains them itself** with `rsr.train.loop.train(d=128, iters=300, device='cpu',
vocab=None, seed=s)` before measuring. Those three checkpoints total 88 MB; the whole run
directory is 2.2 GB because of the trajectory arms, all of it gitignored and all of it
regenerable from the four commands in §11.

**What it should print** (rows `REEXEC_*`, written by exec'ing that same text inside the
harness process — so these are ledger rows, not transcription):

| (chance 0.500) | trained | untrained | paired Δ (null 0) |
|---|---|---|---|
| `specific` pair gestalt cosine | 0.99999 ± 0.00001 | 0.66866 ± 0.15331 | — |
| `specific` `disc_ri` | **0.51333 ± 0.02466** | 0.46667 ± 0.02466 | +0.046667 ± 0.049329 |
| `specific` `disc_alpha` | 0.53167 ± 0.05204 | 0.51167 ± 0.04252 | +0.020000 ± 0.093408 |
| `specific` tie-split `disc_LOO` | 0.51917 ± 0.04693 | 0.51000 ± 0.02500 | +0.009167 ± 0.036429 |
| `specific` exact-tie rate, LOO | 0.20500 ± 0.03500 | 0.00000 ± 0.00000 | — |
| `specific` `r_i(needed)/r_i(other)` | 1.00074 ± 0.00256 | 1.00087 ± 0.00281 | −0.000128 ± 0.004703 |
| `topic` `disc_ri` | 0.53333 ± 0.03175 | 0.48333 ± 0.03686 | +0.050000 ± 0.062650 |
| `topic` `α` KL from uniform | 0.00130 ± 0.00038 | 0.00209 ± 0.00068 | — |

These are at 200 trials × 3 seeds and so are noisier than the 400-trial rows in §5 —
e.g. `specific disc_ri` 0.51333 ± 0.02466 here versus 0.510833 ± 0.025166 there, both at
chance. The full stdout is stored verbatim in row `reexec_snippet_stdout`.

---

## 9 — `sd == 0.0` rows

**19** statistic rows carry `sd` exactly 0.0. All 19 are classified in the same process
that wrote them (`sd_zero_rows_classified`); **`sd_zero_rows_unclassified` is `[]`**.

| category | rows | why zero |
|---|---|---|
| `REALCORPUS_{trained,untrained}_chance_argmax` | 2 | deterministic by construction — the same 32 documents and the same 48 steps in every seed, so the mean `1/n_live` over query events is identical |
| `REALCORPUS_*_{n_query_events, n_events, n_sentences}` | 6 | **counts**, not measurements — same corpus, same event filter (450 / 450 / 400) |
| `REALCORPUS_trained_allpairs_frac_ge_0p9` | 1 | a saturated order statistic: **1.0 in every seed**, i.e. 100 % of sentence pairs at `cos ≥ 0.9`. **This row IS the collapse measurement.** Its untrained counterpart is 0.069169 ± 0.037754 and is not in this table |
| `{trained,untrained}_{specific,topic}_TIE_rate_{alpha,ri}` | 8 | exact-tie rates that are **0.0** in every seed — `α` and `r_i` never tie between the two members, in either model |
| `untrained_specific_TIE_rate_loo_query`, `REEXEC_untrained_specific_tie_loo` | 2 | the *untrained* LOO Δ never ties either — which is exactly what isolates the **trained** arm's 0.221667 ± 0.025166 tie rate as a collapse symptom rather than a harness bug |

None is a seed that failed to reach the RNG: the three seeds vary wherever they can —
`trained_specific_disc_ri_gated` sd 0.025166, `REALCORPUS_trained_argmax_alpha` sd 0.048737,
`untrained_specific_pair_cosine` sd 0.147984.

**Three rows print as `0.000000 ± 0.000000` in §5/§7 and are *not* in this table**, because
their sd is small but not exactly zero: the trained LOO Δ rows. Their true magnitudes are
`trained_specific_loo_q_needed` **2.7e-08 ± 5.2e-08**, `…_loo_q_other` **1.1e-08 ± 2.2e-08**,
`REALCORPUS_trained_loo_delta_assert` **−1.20e-07 ± 2.68e-08**. They are float32 noise
around zero, which is the point.

---

## 10 — What this does **not** establish

- **It does not show that a well-trained TG is blind.** It shows that *this* TG, trained by
  *this* repo's training entry point on *this* corpus at *this* budget, is degenerate. The
  falsifier's claim is dead as stated — training did not rescue `r_i` — but the stronger
  claim "no trained TG can separate a near-duplicate pair" is **not** established and I am
  not making it.
- **It does not identify the cause of the collapse.** Three candidates are visible and
  unseparated: (a) the missing `srep_norm_penalty` in `step_fn` (§3), (b) a corpus of
  3072 sentences / ~15k tokens that is memorisable without memory — the reference trains on
  **12M** tokens — and (c) the loss being dominated by ~58 pad positions per sentence.
  Only (a) is a code defect; (b) and (c) are scale. **Attributing it needs the next
  experiment, not this one.**
- **The pair is a construction, not a corpus observation.** `specific` and `topic` pairs
  are assembled from the generator's tuples; the corpus does not naturally emit two asserts
  differing in one token adjacent in memory. §13: nothing here is a claim about real text.
- **`LOO_restate` is off-distribution** and labelled as such — the corpus has no sentence
  after a query that carries the answer, so the oracle-informed readout had to be
  synthesised.
- **`α` is reduced over the query-token axis** (token-mean, with an `eos` variant reported).
  §3.2.1's formula has no token axis and does not say which. That choice is mine and is
  stated; both give the same answer here, but only because everything is uniform.
- **Gestalts in `gestalt_geometry`/`srep_collapse` are computed with an empty memory**, one
  sentence at a time. The in-trial cosines, computed through the real recurrence with the
  memory live, agree (`trained_specific_offdiag_cosine_mean` = 0.999972 ± 0.000015), so the
  simplification does not carry the result — but it is a simplification.
- **No claim about scaling.** 300 iterations at `d = 128` is a budget, not a curve.
- **CPU only, one machine.** Correction 22 / §13: nothing here supports a claim about other
  hardware, and MPS was not used, so this cycle says nothing about its loss floor either.

---

## 11 — The commands

```bash
# the three headline checkpoints (25 min each, run in parallel)
.venv/bin/python -c "from rsr.train.loop import train; train(d=128, iters=300, device='cpu', seed=S, out_dir='runs/cycle-trained-alpha-pair/train-seed$S', ckpt_every=300, beat_every=1)"
# the collapse trajectory: the SAME trajectory, checkpointed finely
.venv/bin/python -c "from rsr.train.loop import train; train(d=128, iters=20,  device='cpu', seed=S, out_dir='runs/cycle-trained-alpha-pair/train-fine-seed$S', ckpt_every=2,  beat_every=1)"
.venv/bin/python -c "from rsr.train.loop import train; train(d=128, iters=300, device='cpu', seed=S, out_dir='runs/cycle-trained-alpha-pair/train-ck-seed$S',   ckpt_every=20, beat_every=10)"
# the learning-rate robustness arm
.venv/bin/python -c "from rsr.train.loop import train; train(d=128, iters=300, device='cpu', seed=S, lr=3e-4, out_dir='runs/cycle-trained-alpha-pair/train-lr3e4-seed$S', ckpt_every=300, beat_every=10)"
# the measurement (writes every row of ledger.json, including the re-execution rows)
.venv/bin/python <scratch>/run_trained_alpha.py --trials 400
```

Scratch harness (throwaway, per the rules):
`…/scratchpad/{lib_trained_alpha.py, run_trained_alpha.py, reexec11.py}`. `reexec11.py` is
stored verbatim in the ledger as `reexec_snippet`.

---

## 12 — The next falsifier I would name

> **"The gestalt collapse is caused by the missing `srep_norm_penalty` in
> `rsr.train.loop`'s `step_fn`. Restore the reference's hinge term with
> `srep_norm_reg_weight`, retrain identically, and the gestalt space stays
> non-degenerate — all-pairs cosine stays below 0.9 and cross-attention KL from uniform
> rises above 0.1 nats."**

Why it is the right one, and why it is reachable:

- **It is the only one of the three candidate causes that is a code defect** rather than a
  budget. If it is the cause, every downstream experiment that trains a TG in this repo —
  E0b's reduction curve, E0d's `r_i`-vs-LOO validation, E1's `γ` sweep, E3's arms — is
  currently training a model whose memory is inert. That is not a small blast radius.
- **It is cheap and fully local.** One term added to a loss in a scratch copy of the loop,
  three 25-minute CPU runs, and the *same* measurement harness — the collapse rows
  (`allpairs_cos_mean`, `srep_eff_dim_participation_ratio`, `alpha_kl_from_uniform`,
  `memuse_delta_query`) are already written and already have their untrained and trained
  values, so the comparison is a re-run, not a new instrument.
- **It has a clean negative arm.** If restoring the hinge does *not* prevent the collapse,
  the cause is scale (3072 sentences against the reference's 12M tokens), which is a
  different and much more expensive finding — and one that must be known **before** E1 is
  read as evidence about anything.
- **It is prerequisite to re-asking this cycle's question honestly.** The falsifier I was
  given cannot be decided on a degenerate model. Fixing the degeneracy is what makes
  "does a trained TG separate a near-duplicate pair" a measurement rather than a tautology.
