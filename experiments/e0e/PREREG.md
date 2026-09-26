# PREREG — E0e: the `ū` distribution and `E[lifetime]` on a FIFO run

**Written 2026-09-26 by the W7 researcher, before any E0e code existed and before any
E0e number was computed.** This file is committed alone, ahead of the code it governs.
Amendments are appended below, dated, with the data that prompted them named. The text
above an amendment is never edited.

**run_id:** `e0e` · **device:** CPU · **seeds:** `0, 1, 2` (one checkpoint per seed) ·
**scope:** `synthetic` (`M = 16`, `S = 48`, `rsr.constants` scoped values).

## What E0e is, from the spec and the corrections

- §3.5 item 3: *"Measure the distribution of `ū` on a FIFO run (E0e) before freezing
  `τ`."* §3.5 item 2: `ū` is an EMA of the per-step share with **half-life
  `E[lifetime]/4`, measured in E0e**. §3.5 item 4: `γ_b ≈ b_max / (0.25 · E[lifetime])`,
  and *"`E[lifetime]` comes free from the same E0e FIFO run."*
- Correction 4: all four §3.5 items apply; `γ_b` is derived in E0e, never frozen at
  0.001. Correction 5: E0e produces `τ`, **and** `γ_b` and the EMA half-life.
- `docs/ROADMAP.md` Sprint 2: *"a FIFO run plus EMA bookkeeping."*
- `src/rsr/constants.py`: `tau` and `E_lifetime` are `MEASURED` with
  `source_experiment="E0e"`; `gamma_b` is `DERIVED` from `b_max` (`FROZEN`, 1.0) and
  `E_lifetime`. The EMA half-life is not a registry constant.
- `docs/queue/items/e0e.md` (from RESEARCH-CONTEXT §12 item 2): `E[lifetime]` is
  policy-dependent, so this file must say which policy's lifetime is measured.
  **It is FIFO's.**

## Model and protocol (the decision this file makes)

**"A FIFO run" is read as: FIFO eviction over complete streams of a TG model whose
memory is live, with `r_i` read in eval mode.** The spec does not require that run to
train. `τ`, `E[lifetime]` and the half-life are properties of the eviction rule and of
the model's retrieval shares, and correction 20 requires `r_i` be collected in eval
mode (`rsr.retention.reward` refuses a train-mode trace). So E0e is **eval-only on
frozen checkpoints** (Stage-1 allowance). No training run, no FIFO continuation.

- **Substrate:** fresh-stream arm B, `ckpt-003000`, seeds 0, 1, 2 — the only substrate
  on which retrieval was demonstrated (`experiments/fresh-stream/RESULTS.md`:
  classification `SCAFFOLD`, `R3000(B)` holds). Read by absolute path from
  `.worktrees/fresh-stream/runs/fresh-stream/B/seed<s>/ckpt-003000.pt`, never written.
  Their sha256 values, taken before this file was committed, go in the manifest:
  - seed 0 `0ee3f8a69b507d927c631eb85116e1cd9739ee4472c44eb7f145d739a118da60`
  - seed 1 `dadd1e08a3849c1211c8479394df4060f91cd2a6e188b97701b783f2068a3da8`
  - seed 2 `b507ebc573a54316f9d3c20337f49e664bb9c414388af925c5a8e87efd6b63b8`

  A hash mismatch is exit `3`. The model config comes from that checkpoint's own
  heartbeat header, not retyped: `D = 128`, `M = 16`, `S = 48`, `max_tokens = 64`, which
  match the registry's `synthetic` scope for `M` and `S`.
- **Streams:** held-out documents 64..127 of each seed's generator
  (`generate(SyntheticConfig(n_documents=128, sentences_per_document=48,
  seed=s))[64:]`), all 64 of them, encoded with the vocabulary of documents 0..63
  (the model's `V`). They are disjoint from arm B's stream and from its vocabulary
  documents (fresh-stream `stream_disjointness`). A word outside the vocabulary means
  exit `3`.
- **Precondition, liveness:** each checkpoint is measured with
  `rsr.metrics.memory_liveness.measure_liveness` (liveness-wiring) against the untrained
  decoy at the same seed and config, on `rsr.train.loop.liveness_batch(config)`, under
  Amendment 1's bands, unchanged. A quarantined checkpoint is refused (exit `3`). **Only
  seeds whose band is `live` contribute to the would-be values.** A seed that is not live
  is reported and excluded. With zero live seeds the run exits `3` and reports no values.
- **Instrument:** `rsr.model.tg.policy_loop.run_policy_loop(..., observe=True)` under
  `model.eval()` and `torch.no_grad()`. The policy is `FIFOPolicy`, eviction unchanged,
  subclassed only to record in `observe()`. `r_i(t) =
  rsr.retention.reward.retrieval_demand(trace, n_live, capacity=M, gated=True)`, which
  is §3.2.1's norm-weighted, gated (correction 17), fill-rescaled target. A sentence's
  identity in memory is its `written_at` step. Slots shift under `write_at`, and index
  is not identity.

## Definitions, fixed before data

1. **Lifetime** of a written sentence = the number of steps at which it is live in the
   memory a forward pass reads. That is the number of `observe()` calls it receives,
   and the number of steps on which `ū` and `b` could update. The stream end truncates
   it: memory and `b` reset at stream boundaries ([P2] fact 2, §3.5, §3.6). Every
   written sentence counts, including one written at the last step (lifetime 0).
   **`E_lifetime` = the mean over every written sentence of every stream of the live
   seeds, pooled.** Also reported:
   - the per-seed means, with spread;
   - the evicted-only mean;
   - the write-to-boundary variant (`e − w`, with `e = S` for survivors).
2. **Half-life** `h = E_lifetime / 4` steps (§3.5 item 2). Per-step EMA weight
   `α = 1 − 2^(−1/h)`.
3. **`ū`:** per sentence, initialised to `1/M` at its write, the centre of the band,
   matching `b_i = 0` at write. At each observation, `ū ← (1 − α)·ū + α·r_i(t)`. **The
   distribution is every post-update `ū` over every (stream, step, live slot).** Also
   reported, descriptively:
   - `ū` initialised at its first observation instead;
   - full-memory steps only (the steps at which an eviction is decided).
4. **`τ` rule:** `τ` = the **75th percentile of `|M·ū − 1|`** over that distribution.
   The band `[(1−τ)/M, (1+τ)/M]` then holds 75% of observations, so `b` updates fire on
   25% of them, which is the fraction §3.5 item 4 assumes when it writes
   `γ_b ≈ b_max / (0.25 · E[lifetime])`. **The spec gives no rule from the distribution
   to `τ`. This one is the researcher's, and is pre-registered here so it cannot be
   chosen after the fact. The owner can revise it.** Also reported, descriptively:
   - the fraction of observations inside `±25%` (v0.4's band, §3.5 item 3's worry);
   - `τ` at 50% firing (the median of `|M·ū − 1|`).
5. **`γ_b`** = `b_max / (0.25 · E_lifetime)`, `b_max = rsr.constants.get("b_max")`.
   This is the registry's own `_derive_gamma_b` formula, applied to the would-be
   `E_lifetime`. The scope question is open (RESEARCH-CONTEXT §12 item 2) and **this run
   does not resolve it**. It reports the `synthetic`-scope value on this substrate.

## Consistency checks (each failure is exit `1`, a real defect)

- At every full-memory step, `Σ_live r_i` = 1 within `1e-5`. Under the rescale, the sum
  is `n_live / M`.
- Per stream, the number of observations equals the sum of the lifetimes.
- No sentence is observed after its eviction, and none is observed before its write.

## Expected (written before any number)

- `encode()` puts an EOS on every sentence (S0-01), so every sentence writes, and FIFO
  lifetimes are **fixed by `(S, M)`**: the 32 sentences written at steps 0..31 live 16
  steps each, and those written at steps 32..47 live 15..0 steps. **`E_lifetime = 632/48
  = 13.1667` exactly, on every seed.** The across-seed sd of exactly 0 is the expected
  result here, not a broken one: it is analytic, and the measurement confirms that
  every sentence wrote. Any deviation means some sentence did not write.
- Half-life `3.2917` steps. `γ_b = 1 / (0.25 · 13.1667) = 0.3038`. That is outside
  §3.5's "order 0.05–0.1", which assumed lifetimes of 20–80. The audit's `E[lt] = M`
  gives 0.25 at `M = 16` (§12.2). The difference is the stream-end truncation.
- `ū` is heavy-tailed: the fraction inside `±25%` is `< 0.5`, and `τ > 0.25`. Low
  confidence; this is the only non-analytic prediction.
- All three seeds are `live` (fresh-stream arm B retrieves at `R3000`).

## Falsifier

**NONE in §2's sense.** E0e is a Sprint-2 instrument measurement that sets constants;
it is not a kill gate (`experiments/e0e/RESULTS.md`, "Kill gate? No"). It can
contradict two spec sentences, which are read off descriptively:
- §3.5 item 3's *"`ū` may essentially never sit inside a ±25% band"* (the fraction
  inside);
- §3.5 item 4's *"order 0.05–0.1"* at this scope.

## Exit codes

- `0`: every stage completed, at least one seed live, values computed.
- `3`: a precondition refused. That covers a checkpoint missing, a hash mismatch, a
  quarantined checkpoint, a vocabulary-closure failure, a measurement that raised, or
  zero live seeds.
- `1`: a consistency check above failed.

No other code.

## Not done here

- **No `rsr.constants.record()` call and no write to `measurements/ledger.json`.** Which
  substrate E1's constants come from is the owner's open decision D2. The run writes
  the would-be `record()` calls into its own ledger as a note, verbatim, and executes
  none of them.
- No threshold on the cross-row cosine. No change to Amendment 1's bands.

## Mutation bar (the script is new)

At least four mutations, each reddening a named test on hand-built inputs with known
answers:
- `r_i` read ungated, or as the raw share instead of the rescaled one;
- the half-life not `E_lifetime/4` (e.g. v0.4's frozen 16);
- lifetime counted over evicted sentences only;
- a non-live seed admitted to the values;
- the `τ` percentile moved off 0.75.

## What this does not establish

The values are FIFO's, on one substrate (fresh-stream arm B), at the `synthetic` scope.
Another substrate or policy gives other values; the learned head's lifetime is not
FIFO's (§12.2). Nothing here involves RSR, `ψ̂` or Kintsch & van Dijk. No scaling claim.
