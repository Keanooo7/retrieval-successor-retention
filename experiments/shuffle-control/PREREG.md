# Shuffle control — pre-registration

🔒 **Committed before `run.py` exists and before any measurement.** Not edited afterwards;
a correction goes in `RESULTS.md` and names this file.

## Why this experiment exists

`docs/RESEARCH-CONTEXT.md` §10.3 reports the project's one finding about what a trained
model does: hand a document's row **another document's entire 16-slot memory** and the
loss moves by *exactly 0.0*, max per-token delta 1.2e-4 nats, min −1.9e-4, across 384
sentences. `has_eos` 384/384, all rows 16/16, `memory_gate` 0.949–0.989.

🔴 **Those numbers cannot be re-derived from any commit.** They were measured on
2026-09-18 by a scratch `measure.py` against `runs/cycle-srep-hinge/i300-h0-seed*/final.pt`.
Neither the script nor the checkpoints were committed (the ledger's own `commands[]` cites
`<scratch>/train_arm.py`). `docs/ROADMAP.md` §5 item 1 says so.

`experiments/cycle-01-masked-loss/` added a shuffle control on 2026-09-20. It measures a
**different model**: 50 iterations, masked loss, hinge on. And the manager rejected its
reading because **the control had no positive control**: an instrument never seen to
read non-zero cannot report an absence.

This experiment **re-derives the §10.3 claim, not its digits**. It trains the audit's
model from committed code, then measures it with a committed instrument that has been
seen reading both ways.

## Condition — the audit's model, pinned from its ledger

From `runs/cycle-srep-hinge/i300-h0-seed0/result.json` (`iters 300`, `V 160`,
`srep_norm_reg_weight_used 0.0`) and `train()`'s defaults at `991983f`, the sha the audit's
trainer was copied from:

| | |
|---|---|
| trainer | `rsr.train.loop.train`, committed, unmodified |
| objective | `masked_loss=False` (the pre-cycle-1 loss that scores PAD) · `srep_norm_reg_weight=0.0` (hinge off) |
| shape | `d=128`, `memory_slots=16`, `max_tokens=64`, `steps_per_stream=48`, `batch=16`, `lr=1e-3`, vocab derived from the corpus |
| iterations | `300` |
| policy | FIFO |
| corpus | synthetic, `SyntheticConfig(sentences_per_document=48, seed=<seed>)` |
| seeds | `0, 1, 2`, each in its own process |
| device | **`cpu`**. The run-to-run floor there is exactly `0.0` (`runs/cycle-arm-identity/`), so any non-zero delta is real and not device noise |
| measured on | the first **8 documents × 48 sentences = 384 sentences** of the seed's corpus, `model.eval()`, `no_grad` |

⚠️ **Known differences from the audit, stated up front.** The audit's device is not
recorded, and its trainer was a scratch copy of `train()` at `991983f`, not today's
`train()`. Today's `train()` has since gained real-token masking (off here) and the
hinge (weight 0 here). So bit-identity with the audit is **not** expected. What is
tested is the claim.

## Instrument

`src/rsr/metrics/memory_liveness.py::shuffle_control`, permutation = roll-by-1 derangement,
memory contents replayed from the honest pass so only ownership changes, `bos_ctx` left
honest (§10.3's corrected ablation). Pinned by `tests/test_shuffle_control.py`; its
mutations are in `scripts/mutation_battery.py`.

## Controls, run on every seed

1. **Memory disabled.** The trained checkpoint with every `memory_gate` zeroed. **Must
   read exactly `0.0`.**
2. **Own memory replayed.** The trained checkpoint, identity permutation. **Must read
   exactly `0.0`.** This proves the replay is faithful.
3. **Live decoy.** The same config, **untrained**, initialised at the seed. **Must read
   non-zero** (`n_tokens_moved > 0`). This proves the instrument can register memory at
   this scale.

## Expectation

The memory is inert. On every seed, `|Δ| / honest loss` is at or below `1e-3`, against the
base paper's +54% for removing working memory.

## Decision rule — `run.py::decide`

| condition | verdict |
|---|---|
| any control fails on any seed | `inconclusive`: the instrument is not shown to work, and nothing is read from the trained delta |
| every seed `|Δ_rel| ≤ 1e-3` | `survived`: the memory is inert, and §10.3's claim is re-derived |
| any seed `|Δ_rel| > 1e-2` | `falsified`: memory contributes ≥1% on that seed, so §10.3's claim does not hold for this model |
| otherwise | `inconclusive` |

The thresholds are relative to the honest real-token NLL. §10.3's `exactly 0.0`, `1.2e-4`
and `−1.9e-4` are **reported beside the new numbers and are not the bar**. They are
digits from an unreproducible run.

## What this does not do

It does not wire the control into `train()` or the heartbeat, and it does not refuse runs.
That is the rest of S0-04 (`docs/lab-notes/dispatch-S0-04-shuffle-control.md`). A
randomly initialised model is a **live decoy, not a trained live memory**: it shows the
instrument can read non-zero, not what a trained, memory-using TG would read. S0-04's
negative control on a trained live memory still needs a model that has one.

---

## Amendment 1 — 2026-09-20, before the 300-iteration run, prompted by the smoke run

🔴 **The decision rule above cannot discriminate, and a plumbing run showed it.** Written
here before the real measurement, and naming the data that prompted it. The original
text above is left as committed.

**What was seen.** `run.py --iters 2 --run-id shuffle-control-smoke` (plumbing only, CPU,
not committed as evidence). The **live decoy**, an untrained model whose memory reaches
the logits by construction, moved the *signed mean* real-token loss by only
`|Δ|/honest` = **4.3e-5, 4.6e-5, 6.1e-5** (seeds 0–1–2), while single tokens moved by up to
**±0.06–0.11 nats**. Handing a row another document's memory pushes tokens both ways, and
the signed mean cancels them. **A memory known to be live therefore passes the original
"inert" test (`≤ 1e-3`) by a factor of ~20.** The original rule would have returned
`survived` whether or not the memory is used.

⚠️ **This also bears on §10.3's own headline.** *"The loss moves by exactly 0.0"* is a
statement about the signed mean, and the signed mean is this weak. The **per-token**
figures (max 1.2e-4, min −1.9e-4) carry the claim. The mean does not.

**The amended rule. It supersedes the table above for this experiment.**

The primary statistic is **`A` = mean |per-token Δ|** (`mean_abs_token_delta`), read on the
trained checkpoint and on the live decoy at the same seed. **`ratio = A_trained / A_decoy`**
states the trained model's sensitivity to whose memory it holds, in units of what a live
random memory produces at this exact shape.

| condition | verdict |
|---|---|
| any control fails on any seed (unchanged), **or** the decoy's `A` is 0 | `inconclusive` |
| every seed `ratio ≤ 0.01` | `survived`: the memory is inert; tokens move at ≤1% of a live memory's rate |
| any seed `ratio ≥ 0.1` | `falsified`: the trained model uses whose memory it holds at ≥10% of a live memory's rate |
| otherwise | `inconclusive` |

The signed mean Δ, its relative form and §10.3's digits are still reported, and none of
them is the bar. For scale, the smoke run's 2-iteration checkpoints, re-read with this
statistic, give `ratio` = **0.105, 0.052, 0.107** (seeds 0–1–2; `A_decoy` 5.8e-3, 1.04e-2,
7.8e-3 nats). A barely trained model is not inert by this measure, which is the
discriminating power the original rule lacked.
