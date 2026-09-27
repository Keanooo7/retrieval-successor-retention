# PREREG — lookahead-room (loop item W10)

**Committed alone, before any code or data of this experiment.** Everything below is
fixed now. Changing a threshold, an arm, a document set or a control after this
commit is changing the pre-registration.

## The question

Can falsifier 3b — RSR at `gamma = 0.9` versus RSR at `gamma = 0` — be tested on the
current S0-03 corpus at `M = 16`?

This is a **read-only measurement** on frozen fresh-stream arm B checkpoints. It
trains nothing, changes no corpus, calls no `rsr.constants.record()`.

## What the manager has already seen (stated, so it cannot be un-seen)

- At `M = 16` the Belady oracle is **decision-identical** to the causal rule "evict
  the oldest slot that is not a pending (asserted, not yet queried) fact": 0 of 6144
  victim decisions differ on `[4096, 4160)`, seeds 0–2, model-free
  (`~/Documents/RSR-2026-09-26-day/redteam-scripts/ident.py`, `id.log`).
- Model-free on `[4096, 4160)`: oracle − FIFO = 0.1852 / 0.1739 / 0.1896; the
  fact/filler rule (random non-assert victim) − FIFO = 0.1498 / 0.1403 / 0.1577
  (`kindrand.py`, `kr.log`).
- W6 (`retention-readability`): arm B reads kept facts at shifted ranks
  (READABLE_AT_SHIFTED_RANK), so residency is a meaningful proxy on this substrate.
- E0e: under FIFO on docs 64..127, full-memory `sum r_i` is 1 to within 1.49e-07.

The manager's conclusion "3b is untestable here" rests on the oracle/pending-rule
identity. That is incomplete: RSR(`gamma = 0`) trains `psi_hat` on the **immediate**
realized demand `r_i(t)` (§3.2.1, `retrieval_demand(gated=True)`), RSR(`gamma = 0.9`)
on the discounted MC return `G_i(t) = sum_k gamma^k r_i(t + k)` (§3.4, correction 2).
If pending asserts draw little `r_i` until their query, a `gamma = 0` target cannot
separate them from filler and lookahead is needed — 3b has room. If pending asserts
already draw high `r_i` before their query, `gamma = 0` suffices and 3b has little.

**Falsifier addressed:** none of §2 directly. This measures whether §2's 3b *can*
discriminate on this corpus — a precondition for running it. It is a D1 input.

## Substrate (read only, sha256-checked; a mismatch is exit 3)

Fresh-stream arm B, `.worktrees/fresh-stream/runs/fresh-stream/B/seed{0,1,2}/`:

Full hashes (the code pins these):
- 2500: `26c4162e7dc6ef1f80dffa451e4ee75c56360bc94227533b38a45b62003e3176`,
  `327c3facde61ef9a7f4b9fcaeb31efcf40ab36d65a5d584ec97637db334bdadd`,
  `74b026f1dd3fe59eb473e501c37ed9bb7d836276270ec875939e5654d8e86ddd`
- 3000: `0ee3f8a69b507d927c631eb85116e1cd9739ee4472c44eb7f145d739a118da60`,
  `dadd1e08a3849c1211c8479394df4060f91cd2a6e188b97701b783f2068a3da8`,
  `b507ebc573a54316f9d3c20337f49e664bb9c414388af925c5a8e87efd6b63b8`

Model config from S0-03 (`_cfg(V)`), `M = 16`, `S = 48`. Eval mode, `no_grad`,
**B = 1 per document**, one fresh policy per document, via the real
`rsr.model.tg.policy_loop.run_policy_loop`. `r_i = retrieval_demand(gated=True)`
exactly as E0e (`experiments/e0e/run.py: r_of`).

## Documents

`U = [64, 1088) ∪ [4096, 4160)` per seed, vocabulary from `[0, 64)`
(`CSC.doc_sets(seed, 64)`), as W6. `P = [4096, 4160)` is the control set.
Disjointness from the probe `[0, 64)`, `P`, fresh-stream's stream
`[4160, 4160 + 16·3000)` and fresh-escape's `[4160 + 16·3000, 4160 + 16·9000)`, and
vocabulary closure, are asserted (exit 3 otherwise).

**Pre-declared fallback, compute only:** if the measured throughput (on one probe
document, timing only, before any `r_i` value is read) projects a per-child wall time
above 3 h, the extended range is truncated to `[64, 64 + N)` with the largest `N`
that fits, `P` always included. The chosen `N` and the timing go in RESULTS.md.

## Definitions

Step `t`: the forward pass on sentence `t` reads the pre-write memory; `r_i(t)` is
collected; then, if memory is full, a victim is chosen; then sentence `t` is written.

**Slot classes at step `t`** (for a sentence `i < t`):
- `pending` — an assert whose query `q` satisfies `q > t`; `k = q − t ≥ 1`;
- `querying` — an assert whose query is `t` itself (`k = 0`), reported separately;
- `answered` — an assert whose query `q < t`;
- `filler` — every non-assert sentence (filler and already-written queries).

**Demand-if-resident, `D[i][t]`** (the primary target source). The FIFO rollout's
own `r_i(t)` for every sentence FIFO holds at `t`. For every sentence FIFO has
already evicted (`i < t − 16`), a **probe**: the same step's forward on FIFO's memory
with rank-0 slot's gestalt replaced by sentence `i`'s FIFO-rollout gestalt; `D[i][t]`
is the probe's `r` at slot 0. Probes are side computations in the same batch; they
never alter the FIFO rollout. Rationale: in a FIFO rollout, realized demand of an
evicted sentence is zero *because FIFO evicted it*, so targets read literally from
the rollout would push any target rule back to FIFO. `D` asks what each sentence
would draw if kept, in the FIFO world.

Targets, per sentence `i`, step `t`, within the stream:
- `T0[i][t] = D[i][t]` — the `gamma = 0` target;
- `G_g[i][t] = sum_{k = 0}^{S-1-t} g^k D[i][t + k]` for `g ∈ {0.9, 0.97}`;
- descriptive: `Tnext[i][t] = D[i][t + 1]` (one-step hindsight, 0 at `t = S − 1`).

**Target rules.** At a full-memory step `t`, evict the live slot with the smallest
target at `t`; ties to the lowest slot index (the oldest). Hit = the queried assert
is resident when its query's forward reads memory. Hit rates are **model-free
residency** through `rsr.metrics.headroom.simulate` with the rule as a policy object.

- `rule_g0` (`T0`), `rule_g09` (`G_0.9`), `rule_g097` (`G_0.97`), `rule_next`
  (`Tnext`, descriptive);
- references: `fifo`, `oracle` (`OraclePolicy(discounted_demand(doc, 0.97))`),
  `pending_fifo` (the red team's causal rule), `factfiller` (random non-assert victim,
  per-document RNG `random.Random(f"ff:{seed}:{doc_id}")`, as W6).

**Secondary variants (never the headline):**
- `literal` — the same rules with targets read literally from the FIFO rollout
  (`r = 0` for any sentence FIFO does not hold). Predicted ≈ FIFO for both, by the
  construction above.
- `online_g0` — the `gamma = 0` rule **with the model in the loop**: the policy
  observes `r_i(t)` from the forward under its *own* memory and evicts the argmin.
  Causal and exact for `gamma = 0`. There is **no causal online `gamma > 0` rule**:
  `G` needs the future under the policy's own evictions, so any `gamma > 0` rule
  here is a hindsight proxy. `online_g0` is compared with `rule_g0` to size the
  FIFO-world caveat on the `gamma = 0` side only.

## 🔴 The caveat, stated before data

The targets come from a FIFO rollout; any other policy changes future memory
contents, future gestalts and future attention. `rule_g*` hit rates are an
**upper-bound-style proxy of what each target could teach**, not RSR's result. The
hindsight is asymmetric: `T0` is causal (observable at decision time — a perfect
`psi_hat` could match it), `G_g` contains future realizations (a perfect `psi_hat`
predicts only its conditional mean). So `room_3b` is **optimistic** for lookahead:
NO_ROOM is the robust direction, TESTABLE_HERE means "room exists in principle".

## Measurements

1. **`r_i` by class and by `k`**: mean, sd, quantiles (10/50/90) of `D` for
   pending / querying / answered / filler, over resident slots at full-memory steps;
   pending by `k = 1 … 32` (bins `k ≥ 17` pooled for display); the same for the
   literal FIFO `r_i`.
2. **AUCs**, per full-memory step with ≥ 1 pending and ≥ 1 filler candidate, then
   averaged over steps: ROC-AUC (ties = 0.5) of `T0`, `G_0.9`, `G_0.97`, `Tnext` at
   ranking pending above filler; over (a) FIFO-resident slots, (b) all sentences
   `i < t` (resident + probed). Also pending vs answered, same way.
3. **Hit rates** per rule per seed on `U` and on `P`.
4. **Headline:** `room_3b = hit(rule_g09) − hit(rule_g0)` on `U`, per seed, with a
   paired per-document percentile bootstrap 95 % CI (2000 replicates, generator seed
   `20260926 + seed`, one resampled document multiset per replicate for every arm;
   hit rate = sum of hits / sum of queries). Compared, never gated, with
   `oracle − fifo` and `factfiller − fifo`, `pending_fifo − fifo`, and
   `room_3b_097 = hit(rule_g097) − hit(rule_g0)`.

## 🔒 Decision rule (primary: ckpt3000; ckpt2500 classified beside it, same rule)

Over seeds 0, 1, 2, on `U`, primary (`D`-based) `room_3b`:

- **TESTABLE_HERE** if `room_3b ≥ 0.05` on every seed **and** CI lower bound `> 0`
  on every seed (lookahead buys at least about a quarter of the ≈0.18 headroom);
- **NO_ROOM** if the CI upper bound is `< 0.02` on every seed (a negative `room_3b`
  counts as no room);
- **PARTIAL** otherwise;
- **INCONCLUSIVE** if any control below fails (exit 3).

Thresholds are the brief's. I keep them: 0.05 ≈ 27 % of the 0.18 oracle headroom is
the smallest effect a 3b run at three seeds could plausibly resolve; 0.02 is about
the spread of FIFO's model-free hit rate across seeds on `P` (0.810–0.826), i.e. below
anything a three-seed 3b comparison could separate from seed noise.

Ledger verdict mapping: TESTABLE_HERE → `survived` (the claim "3b has room here"),
NO_ROOM → `falsified`, PARTIAL / INCONCLUSIVE → `inconclusive`.

## Controls (any failure → INCONCLUSIVE, exit 3)

- **C1** FIFO rollout's live `answer_acc` on `P`, per S0-03 bucket, equals the
  fresh-stream ledger's `B.ckpt{c}.heldout.live.<bucket>.answer_acc` sample for the
  seed to `|diff| ≤ 1e-12` (count ratios; W6 found 0 argmax flips B = 1 vs batched).
  `answer_nll` compared too, tolerance 1e-4, reported.
- **C2** Model-free on `P`: `oracle − fifo` rounds (4 dp) to 0.1852 / 0.1739 /
  0.1896; the `kindrand.py` fact/filler convention (`random.Random(seed)`, shared over
  `P` in document order) rounds to 0.1498 / 0.1403 / 0.1577; `pending_fifo` and
  `oracle` victims differ on 0 decisions.
- **C3** Every full-memory step's `sum r_i` is within 1e-5 of 1 (E0e's `SUM_TOL`):
  the FIFO rollout, every probe row, and `online_g0`.
- **C4** Identity probe: at every full-memory step a probe re-inserting FIFO's own
  rank-0 sentence reproduces FIFO's `r` at slot 0 to `≤ 1e-6`.
- **C5** `online_g0`'s in-loop residency equals a model-free replay of its victim
  sequence; `rule_*` residencies computed by `simulate` are deterministic (re-run
  equal on `P`).
- **C6** Disjointness and vocabulary closure (above); sha256 of every checkpoint.

## Prediction (honest, before data)

I do not know whether this model's attention anticipates facts. Two considerations
pull opposite ways: filler sentences have no reason to retrieve, so pending asserts
should draw ~baseline share until queried (lookahead needed); but `r_i` is
norm-weighted by `||g · W_O v||`, and assert gestalts may carry larger value norms,
lifting asserts regardless of query timing (`gamma = 0` separates asserts from
filler, but not pending from answered — which the pending-vs-kind gap, 1.000 vs
≈0.985 model-free, bounds at ≈0.015).

- pending-vs-filler AUC: `T0` ≈ 0.6 (resident), `G_0.9` ≈ 0.75;
- `rule_g0` hit near FIFO or random (0.79–0.82); `rule_g09` ≈ 0.90;
- headline ckpt3000: **TESTABLE_HERE ≈ 45 %, PARTIAL ≈ 30 %, NO_ROOM ≈ 25 %**;
- `literal` variants: both ≈ FIFO, `room ≈ 0` — by construction, not evidence;
- `online_g0` within ±0.03 of `rule_g0`.

## Exit codes

0 a classification was reached (PARTIAL included) · 3 inconclusive / did not run
(a control failed, a measurement raised, a checkpoint missing or mismatched).

---

## Amendment 1 (2026-09-26), written AFTER run 1's data was seen

**Status.** This amendment is written **after** the first run (`runs/lookahead-room/`,
run sha `95a96fc`, verdict INCONCLUSIVE, exit 3) and after every number it produced
had been read. Run 1 stays on the record unchanged. The re-run it governs uses a new
run id, `lookahead-room-r2`, and a run sha at or after this commit.

**The only change: C1's tolerance.** C1 now passes when both of these hold:
1. **argmax identity.** On `P`, every answer's argmax-correct flag from the harness's
   B = 1 FIFO rollout equals the flag S0-03's own instrument produces
   (`answer_readout(cond="live")`, batched over `P` as the fresh-stream ledger was
   measured). Zero mismatches are allowed.
2. **float32 accuracy, exact.** Per S0-03 bucket, the harness's accuracy recomputed
   as `float(ok.float().mean())` (a float32 mean, which is how the reference ledger
   computed it) equals the ledger's `B.ckpt{c}.heldout.live.<bucket>.answer_acc`
   sample for the seed **exactly** (`==`).

The `answer_nll` comparison stays descriptive, with its 1e-4 tolerance.

**Why.** The reference ledger stores float32 means (`_summarise`:
`float(r["ok"][m].float().mean())`). A float64 recomputation differs from them by
float32 rounding (run 1: up to about 3e-08) even when every argmax matches. So the
original "`|diff| ≤ 1e-12`" could never hold, whatever the rollout did. The defect was
in this PREREG's transcription of "exactly", not in the rollout.

**Unchanged.** No threshold (0.05 / 0.02 / CI rule), arm, rule, target definition
(`D`, `T0`, `G_0.9`, `G_0.97`, `Tnext`, literal), document set, seed, checkpoint,
bootstrap, or other control changes.

**Expected outcome, stated honestly.** Run 1's numbers have been seen:
`room_3b` at ckpt3000 was 0.0355 / 0.0420 / 0.0478, every CI lower bound above 0 and
every point below 0.05. The pipeline is deterministic on CPU, so r2 is expected to
reproduce run 1 bit-for-bit and classify **PARTIAL** at both checkpoints. Any
difference from run 1 is itself a finding and will be reported.
