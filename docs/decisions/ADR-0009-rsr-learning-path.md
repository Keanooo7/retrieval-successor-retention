# ADR-0009 — The ψ̂ learning path: `observe` → realized `r_i` → Monte-Carlo return → `L_MC` on `φ` only

- **Status:** **proposed** — pending Brendan's sign-off. Design only; **no implementation
  code lands with this ADR.** Every choice the spec does not fix is marked
  **DECISION FOR BRENDAN**, with options and a recommendation. None of them is
  settled here.
- **Date:** 2026-09-26
- **Decision:** day-loop item W8 (Track 3 of `~/Documents/RSR-2026-09-26-day/ROADMAP.md`),
  "write the ψ̂ learning-path ADR, as PROPOSED, and stop."
- **Base:** `integ/2026-09-26` at `cd41d4e`, which includes correction 31. All `file:line`
  anchors below are read at that commit.
- **Relates to:** spec §3.2–§3.7, §4.3, §4.5, §7.1, §7.2, §7.7; `docs/spec-corrections.md`
  corrections 1, 2, 3, 15, 16, 19, 20, 21 and 31; ADR-0006 (rank-indexed `P^(sent)`);
  ADR-0008 (the `Q_tok` collapse); `docs/RESEARCH-CONTEXT.md` §12 (owner decisions);
  `docs/ROADMAP.md` Sprint 2 ordering trap.
- **Written before:** any code that trains `φ`. Today `RSRPolicy.observe` raises
  `NotImplementedError("Sprint 2: the MC return and the shadow buffer ...")`
  (`src/rsr/retention/rsr.py:474-483`), `ShadowBuffer` raises (`src/rsr/retention/shadow.py:39-46`),
  and there is no `L_MC` anywhere in `src/`.

---

## Context

§3.3 makes the Monte-Carlo return the default learning rule for `ψ̂`, and correction 2
makes it authoritative:

```
G_i(t) = Σ_{k=0}^{S−t} γ^k · r_i(t+k)                  (spec :249)
L_MC   = Σ_{i ∈ M} ( ψ̂_φ(s_i, c_t) − sg[ G_i(t) ] )²   (spec :250)
L      = L_NTP + β · L_MC                               (spec :262, read per correction 2)
```

Every component of that loop exists except the loop itself:

| piece | where it is | state |
|---|---|---|
| `r_i` (gated, share-of-M rescaled, eval-mode only) | `src/rsr/retention/reward.py:154-160` | implemented; E0e used it (W7) |
| capture bridge: `α [L,H,M]`, `W_O v`, `g_mem` from a real forward | `src/rsr/model/tg/policy_loop.py:170-259` | implemented (ADR-0008) |
| `observe` call site, before eviction and write | `policy_loop.py:342-351` | implemented, **off by default** (`observe=False`, `:288`) |
| `ψ̂_φ` with the §4.3 multipliers | `src/rsr/retention/value_head.py:44-166` | implemented |
| scored eviction with stop-grad on inputs | `rsr.py:371-407`, detach at `:384` | implemented |
| warmup on the optimizer step | `rsr.py:418-440`; `loop.py:566-567` | implemented (correction 31(a)) |
| `RSRPolicy.observe` | `rsr.py:474-483` | **raises** |
| shadow buffer | `shadow.py:36-46` | **raises** |
| `ProtectionBias` | `src/rsr/retention/bias.py:56-72` | **raises**; blocked on the owner item |
| value head in the optimizer | `loop.py:426` passes `value_head=None` | **strict xfail** `tests/test_train_loop.py:316-345` |
| `train()` building an RSR arm | `loop.py:296-348`, `:435` | **refuses** (correction 31(b), then `UnmeasuredConstant`) |

This ADR specifies how those pieces join. It does not pick the substrate (D2), the epoch
(31(b)), the corpus (D1), or the bias interface. It says what each of those blocks.

### Two latent defects found while reading for this ADR

Both are on a path that is unreachable today, because `build_policy("rsr")` refuses
(`loop.py:332-339`). Both would fire the day it stops refusing.

1. **`train()` builds `φ` from the data-order generator.** `gen` is seeded at `loop.py:385`,
   passed into `build_policy(..., generator=gen)` at `:435`, handed to
   `BilinearValueHead(d_model, generator=generator)` (`rsr.py:333`), and consumed by
   `W.normal_` and `u.normal_` (`value_head.py:129-130`). The same `gen` then draws the stream
   indices (`loop.py:547`). So an RSR arm and a FIFO arm at the same seed would train on
   **different stream sequences**, which is exactly §3.7's warning (spec :348). E0b cannot see
   it: it never goes through `train()`.
   *Checked in a scratch script:* first-batch indices from a CPU generator with and without
   head construction differ (`data order identical with/without head: False`).
2. **On MPS the same line raises.** `gen` is `torch.Generator(device=device)` (`loop.py:385`),
   and the head's parameters are created on CPU.
   *Checked:* `BilinearValueHead(32, generator=torch.Generator(device="mps"))` raises
   `RuntimeError: Expected a 'cpu' device type for generator but found 'mps'`.

**DECISION L11** below fixes both with a dedicated CPU generator for `φ`.

### A finding that changes what β means: under AdamW and the isolation rule, β is inert

§3.3 writes `L = L_NTP + β·L_MC`, and the registry makes β MEASURED by E1, swept over
`{0.01, 0.1, 1.0}` (`src/rsr/constants.py:497-506`; spec §4.5 :437). But the no-backprop
rule means `L_MC`'s gradient reaches **only `φ`**, and `φ` gets **no gradient from `L_NTP`**.
So β multiplies the whole of `φ`'s gradient and nothing else. AdamW's update
`m̂ / (√v̂ + ε)` is invariant to a constant gradient scale, apart from ε. The transformer's
update does not depend on `φ`'s gradient at all, provided `φ` is kept out of the transformer's
gradient clip (see **Gradient isolation** below).

*Checked in a scratch script* (d = 32, unit-norm inputs, 200 AdamW steps, betas
(0.9, 0.95), wd 0.01, lr 1e-3/m, max |φ| = 0.674):

| ε | max \|φ(β=0.1) − φ(β=1)\| | max \|φ(β=0.01) − φ(β=1)\| |
|---|---|---|
| 1e-8 (torch default) | 6.2e-05 | 5.8e-04 |
| 1e-30 | 1.2e-07 | 8.9e-08 (float rounding) |

So **E1's β sweep would return "no effect" for a reason that has nothing to do with the
hypothesis.** A frozen constant that cannot move what it governs is defect D-1's shape
(correction 4). β works as an on/off switch (β = 0 is §3.7's reduction value), and the knob
that actually sets how hard `φ` fits is `φ`'s learning rate. The same holds for the reduction
over samples in `L_MC` (sum vs mean): its overall scale is inert. **DECISION L7.**

§3.4 (:280, "before touching `β`") and §7.7 (:607, "move `λ` below 1 before touching `β`")
both presume β is a live knob. I did **not** add a correction entry; that file is Brendan's.
This is flagged for one.

---

## Proposal

### 1. The data path

```
per sentence step t, per live row b          (run_policy_loop, policy_loop.py:328-379)
 ├─ training forward (train mode, graph kept)          → L_NTP contribution, out.srep
 ├─ [NEW] eval-mode capture forward, no_grad            → CrossCapture(eval_mode=True), c_t^eval
 ├─ observe(memory_state(mem,t,b), trace_for_row(...))   (pre-write memory, :342-351)
 │     └─ RSRPolicy.observe:  r(t) = retrieval_demand(trace, n_live, M, gated=True)
 │                            append (row b, t, key=written_at_i, r_i(t), s_i, c_t) for live i
 ├─ select_eviction (full rows only)  — FIFO if k < T_warm, else argmin score
 └─ write_at
after the stream (the call returns)
 ├─ G_i(t) = Σ_{k=0}^{T_b−1−t} γ^k · r_i(t+k)   per (row, key), reverse discounted cumsum
 ├─ ψ̂ = head(s_i, c_t)  batched over every buffered (row, t, i)  — inputs detached
 └─ L_MC = mean_{(b,t)} Σ_{i live at t} ( ψ̂ − G_i(t) )²     returned by policy.retention_loss()
train():  loss_total = L_NTP + β · L_MC ;  backward once ;  clip(model) ;  opt.step()
```

**Slot identity is the write step, not the slot index.** `write_at` compacts the prefix when
it evicts a middle slot (`policy_loop.py:87-98`), so index `i` does not name the same gestalt
across steps. `mem.step` carries each slot's write step, surfaced as `MemoryState.written_at`
(`policy_loop.py:125`). At most one write happens per step, so `(row, written_at)` is unique
within a stream. `G` is keyed by it. A test keys `G` through a middle-slot eviction
(**Tests**, T5).

**`r_i` is the correction-20 eval-mode quantity.** The capture reads `last_attention` **after**
`attn_drop` (`src/rsr/model/tg/model.py:334-335`), and `cross_capture` stamps
`eval_mode=(not model.training) or attn_dropout == 0.0` (`policy_loop.py:257`). With the
reference's `attn_dropout = 0.2` (`src/rsr/model/tg/config.py:63`), a capture taken from the
training forward is train-mode, and `reward._require_eval` refuses it (`reward.py:88-96`). This
is correct behaviour. So the training loop needs an eval-mode source for `α`.
**DECISION L1.**

**The return is truncated at the stream end, as the spec writes it.** One call of
`run_policy_loop` is one stream: memory is initialised at the top and `policy.reset()` is
called on entry (`policy_loop.py:315, 326`). The spec's upper limit `S−t` (1-based) is
`T_b − 1 − t` for 0-based `t` and row length `T_b = lengths[b]`. Late samples therefore carry
systematically shorter returns. That is spec-fixed, not a choice. The implementation must
**report mean `G` by `t`** so the truncation profile is visible. `γ = 0` gives
`G_i(t) = r_i(t)` exactly; implement that case explicitly rather than relying on `0**0`.

**The shadow buffer's role, and what happens without it.** `r_i` exists only for slots in
memory (§3.4 :282), so a slot evicted at step `t_e` contributes `r_i = 0` to every later term.
Without a shadow buffer, every buffered `(s_i, c_t)` with `t < t_e` regresses onto a return
that is **cut at the policy's own eviction**. Two consequences, both of which must be written
on any shadow-off run:

- **Under FIFO (every pre-`T_warm` step), the cut is at age `M`.** A slot at age `a` has at
  most `M − a` observable terms. The target itself is then age-shaped, even though `ψ̂` has no
  age input (§3.2.2). Any content-age correlation lets `ψ̂` learn recency from the target. That
  is §7.1's failure mode, fed through the label rather than the features. Spec :280's
  "everything before `T_warm` is unbiased FIFO-collected data" is right about the *policy*
  (not self-selected) and wrong about the *target*, which is FIFO-censored.
- **After `T_warm`, the cut is the learned policy's**, so the censoring is self-confirming
  (§3.4 :282, §7.2).

With the shadow buffer (`K` from the registry: 40 synthetic, 64 PG-19; correction 1;
`constants.py:363-380`), an evicted gestalt keeps producing a would-be `r̃_i` for `K` steps.
Beyond `K` the return is still censored, and that is stated. How `λ_shadow` (FROZEN 0.5,
`constants.py:406-415`) enters the loss is not fixed by the spec. **DECISION L3.** How the
would-be `α` is formed is not fixed either. **DECISION L4.** The query tensors it needs are
not exposed by the forward. **DECISION L5.** Whether slice 1 ships without it is
**DECISION L6**.

**What `observe` needs that the protocol does not carry.** `observe(slots, attn, step)`
(`src/rsr/retention/policy.py:216`) has no `c_t` and no row identity. The policy is one object
serving `B` rows (`policy_loop.py:345-351, 358-364`), and the day roadmap has already caught a
batched policy applying row 0's state to every row (`OraclePolicy`, Track 4 note).
**DECISION L10.**

**`L_MC` is formed after the stream, from detached buffers.** `ψ̂` is recomputed in one batched
call at stream end. It is not kept with a graph from each step. This costs one extra `O(S·M·d)`
forward of a `d×d` bilinear per stream, which is negligible. It holds no autograd graph across
the loop, and it makes the stop-gradient a property of the buffer. Buffer size at S0-03 shapes
(`B = 16, S = 48, M = 16`) is ≤ 12,288 regression rows per optimizer step.

### 2. Gradient isolation — where the `detach` lives, and the test that proves it

CLAUDE.md, spec §3.3 :259, and correction 2 ("whichever retention loss is active") allow
`L_MC`'s gradient into `φ = {W, u}` only.

| input | where it is cut | line |
|---|---|---|
| `s_i` | `memory_state` returns `mem.kv[row].detach()` | `policy_loop.py:124` |
| `c_t` (eviction) | call site passes `out.srep[row].detach()` | `policy_loop.py:362` |
| `α`, `W_O v`, `g_mem` → `r_i` → `G` | the whole capture is under `torch.no_grad()` | `policy_loop.py:231` |
| `c_t` (eval capture, proposed) | the eval forward runs under `torch.no_grad()` | new |
| **`ψ̂` regression inputs (proposed)** | **`RSRPolicy.retention_loss` detaches `s` and `c` itself, as `_psi` does** | mirrors `rsr.py:384` |

**One owning site.** The policy owns the stop-gradient for its regression, as it already does
for scoring (`rsr.py:380-384`). The head deliberately does not detach
(`value_head.py:139-144`), so that a caller who meant to backpropagate cannot do it silently.
The upstream detaches (`:124`, `:362`, no_grad at `:231`) stay. The unit test must therefore
feed `retention_loss` **attached** inputs directly, so that removing the policy's own detach
reddens it. Otherwise the upstream detach masks the mutation (an equivalent mutant).

**Isolation is also an optimizer property, not only an autograd one.** The transformer's clip
is `clip_grad_norm_(model.parameters(), 1.0)` (`loop.py:571`). `φ` is not in
`model.parameters()` today, and it **must not be added to that clip**. A joint norm would make
the transformer's clip factor depend on `L_MC`. That is a gradient-free but real channel from
the retention loss into transformer updates. AdamW is elementwise, so sharing one optimizer
object is safe; sharing a clip norm is not.

**The test that proves it** (T1–T4 below). On a real `TGModel` with a real `RSRPolicy` and
`observe` on:
- `L_MC.backward()` leaves **every** `model.parameters()` gradient `None`, including
  `W_sent`/`srep_head`, `cross_attn.value`, `attn_out_proj` and `memory_gate`. Every `φ`
  gradient is non-`None` and nonzero.
- After one full optimizer step, **transformer weights are bitwise identical** with β = 1
  and with `L_MC` not computed. This is the strongest form, because it also covers the clip
  and the optimizer.

### 3. Optimizer and μP

- **Group.** The head goes into `build_param_groups(model, policy.head, ...)`. It gets its own
  `value_head` group at `base_lr / m` (`src/rsr/mup/param_groups.py:101-109`). This lifts the
  strict xfail at `tests/test_train_loop.py:316-345`. **Its body must be rewritten**: today it
  calls `build_param_groups(model, None, ...)` itself (`:344`), so it cannot turn green from a
  change to `train()`. It should assert the group through `train()`'s own construction path.
- **Order.** The policy (and head) must be built **before** the optimizer. Today the optimizer
  is built first (`loop.py:426-427`, then `:435`). With **L11**'s dedicated generator, this
  reordering does not move the data RNG.
- **Width transfer.** Unchanged from `value_head.py:115-116` (`1/d` bilinear, `1/(2d)` linear)
  and `param_groups.py` (`base_lr/m` for both). **Correction 15 is still open.** Gestalts are
  unit-norm, so with the `1/d` multiplier, fitting a target of order `G ~ 1/(M(1−γ))` needs
  `σ_max(W) ~ G·d`. At `M = 16, γ = 0.9, d = 128` that is ~80, against ~2 at init. This is
  arithmetic, not measured: `φ` may underfit the target scale inside a few-thousand-step run
  at μP LR. The eviction z-scores `ψ̂`, so scale alone does not change a decision, but an
  underfit regression does. **The ported E0a re-run with unit-norm gestalts (ROADMAP Sprint 2)
  is the gate for this, run twice (bare TG, then with the head), per §4.3 bill 3.**
- **One optimizer or two.** **DECISION L8.**
- **β.** **DECISION L7** (see Context).

### 4. Warmup and eval semantics after correction 31

- `train()` already calls `policy.set_train_step(it)` before every stream batch
  (`loop.py:566-567`), with `it` absolute, so a resume keeps its place.
- **During `k < T_warm`:** eviction is FIFO and recorded `warm=True, attribution="fifo_warmup"`
  (`rsr.py:440-443`). `observe` and `L_MC` run exactly as after warmup. This is §3.4's "ψ̂ trains
  passively on realized `r_i` (β active, policy inert)" (spec :276). Nothing in the learning
  path branches on `warm`.
- **Every buffered sample is tagged with `k` and `warm`,** so `L_MC` can be reported split at
  the on/off-policy boundary §3.4 names (and §7.2 needs). The FIFO-censored-target caveat
  (§1) goes on the pre-`T_warm` half.
- **Eval.** At evaluation, `observe` is off unless a diagnostic asks for it, and
  `retention_loss()` is never consumed. `set_train_step` must still be called with the
  trained step or any `k ≥ T_warm`, and it raises if unset (`rsr.py:433-439`). This ADR does
  not change that.
- **What `T_warm` is in steps is blocked on correction 31(b)** (below). The learning path is
  correct for any positive `T_warm`, and for `T_warm = 0` (correction 16's reduction value).
  It does not need the ruling to be *built*, only to be *run as the spec's arm*.

### 5. `train()` overrides — building the arms E1 needs before E1 has run

`RSRConfig.from_registry(..., **overrides)` already reads lazily and never reads an
overridden field (`rsr.py:242-268`; S0-01). `train()` cannot reach it:
`build_policy("rsr")` calls `from_registry(scope, steps_per_epoch=...)` with no overrides
(`loop.py:339`), and `train()` passes `steps_per_epoch=None` (`:435`). **DECISION L12**
proposes a `policy_overrides` mapping on `train()`/`build_policy`/the CLI with these rules:

- The override dict is **stamped into `frozen`**, so into the config hash and the `run_id`.
  The **arm label** is derived from it (e.g. `rsr[gamma=0,b=off,shadow=off]`). Two arms that
  differ in an override cannot share a hash, and an unlabelled pre-E0e arm (ROADMAP ordering
  trap) cannot exist.
- **`b_enabled=False` is required, not optional, until `ProtectionBias` exists.**
  `RSRPolicy.__init__` refuses `b_enabled=True` without a bias object (`rsr.py:321-327`).
  Every such arm is partially reduced toward TG (§3.7's `b ≡ 0`), and says so in the label.
- **`shadow_enabled=False` must be passed explicitly while the buffer raises**, and
  `RSRPolicy` must **refuse** `shadow_enabled=True` until slice 2. That follows the rule
  already written at `rsr.py:305`: "a switch that is set and does nothing is worse than no
  switch." Today the dataclass default is `True` (`rsr.py:205`) and nothing consumes it.
- **`gamma=0.0`** is the control arm (`constants.py:487-491`; `check_gamma_horizon` returns
  early at `:227-228`).
- **`t_warm` override is refused** unless the arm is declared as the no-warmup ablation.
  Otherwise an override would decide correction 31(b) by the back door.
- **Fields not overridden still read the registry and still raise** (asserted today at
  `tests/test_train_loop.py:305-313`). Overrides are visible departures, not defaults.

**Every new score term needs an off-switch in `REDUCTION_SWITCHES`** (`rsr.py:162-173`;
§3.7; E0b's `test_every_config_field_has_an_off_switch`, `tests/test_reduction.py:123`).
**This ADR adds no score term and proposes no new `RSRConfig` field for slice 1.** The
learning path changes `φ`, not the form of `argmin[z(ψ̂) + b − ν·max cos]`. Its knobs are
existing switches (`psi_override`, `beta`, `gamma`, `shadow_enabled`) or registry constants
read at construction (`K`, `lambda_shadow`, `lambda_return`). So `REDUCTION_SWITCHES` and
`test_reduction.py` stay byte-identical. If A8 (TD) or a `λ < 1` return is later made
configurable, **that** field needs a reduction entry and a flip case at
`test_reduction.py:163-184`.

### 6. Tests and mutations the implementation must ship

| # | test | guards |
|---|---|---|
| T1 | `L_MC.backward()` on a real tiny `TGModel`: every model grad `None`; every `φ` grad nonzero | the prohibition |
| T2 | `L_NTP.backward()` alone: every `φ` grad `None` | `φ` is not in the LM path |
| T3 | `retention_loss` fed **attached** `s`, `c` directly: model grads still `None` | the policy's own detach (non-equivalent mutant) |
| T4 | one optimizer step, β = 1 vs `L_MC` not computed: transformer weights **bitwise equal**; `φ` absent from the clip set | clip/optimizer channel |
| T5 | hand-built `r` sequence → `G` equals the closed form; `γ = 0` → `G = r(t)`; truncation at `lengths[b]`; keys follow a slot through a middle-slot eviction | the return |
| T6 | in `train()`, every trace reaching `observe` has `eval_mode=True`; `model.training` is `True` after the loop; global and data RNG states are unchanged by the eval forward | correction 20; RNG |
| T7 | two rows with different streams produce different buffers and different `G` | the row-0 bug class |
| T8 | `k < T_warm`: evictions `fifo_warmup` **and** `φ` receives nonzero grad; `k ≥ T_warm`: scored | §3.4 passive training |
| T9 | the value head is in its own μP group via `train()`'s path (the xfail at `test_train_loop.py:316` removed, body rewritten) | §4.3 |
| T10 | same seed: first-batch stream indices identical for the FIFO arm and the RSR arm in `train()`; RSR arm constructs on MPS | L11's two defects |
| T11 | override arms build on an empty ledger; a non-overridden field still raises; a `t_warm` override without the ablation flag raises; the label and hash differ per override set; `shadow_enabled=True` raises in slice 1 | §5 rules |
| T12 | β = 0: `φ` bitwise unchanged after a step | L13 |
| T13 | the §3.7 reduction through `train()`'s construction: no head, `observe` off, `retention_loss()` is `None` | E0b stays a reduction |

**Mutations for `scripts/mutation_battery.py`, each proven to redden its own gate:** drop the
policy's regression detach (T3); drop the eval toggle (T6: `TrainModeTrace`); key `G` by slot
index (T5); add `φ` to the transformer clip (T4); build the head from `gen` (T10); start the
sum at `k = 1` or run it to `T_b` (T5); pass `value_head=None` to `build_param_groups` (T9);
accept `shadow_enabled=True` silently (T11).

**Why E0b and fidelity stay untouched.**
- `test_reduction.py` runs `run_policy_loop` with the default `observe=False` and
  `psi_override="neg_age"` (no head). Slice 1 changes neither default and adds no
  `RSRConfig` field, so the bit-exact loss curve, the switch enumeration and the flip list are
  not edited.
- `test_fidelity.py` compares the transcription (`model.py` `run_sentence_loop`) against JAX
  golden tensors. Slice 1 does not touch `model.py`. The eval-capture forward is a second call
  of the unchanged forward.
- **The one exception is slice 2's query stash (L5)**, which touches `CrossAttention.forward`.
  It adds an attribute assignment and no arithmetic, and `test_fidelity.py` plus
  `test_reduction.py` are its gate.

### 7. What stays blocked, and on what

These are **not decided here**, and nothing in this ADR routes around them.

| blocked | on | why it binds the learning path |
|---|---|---|
| **E1 itself** | **D2**, the substrate ADR (from scratch on the repeating corpus as the spec does, or resume from arm B; transformer frozen or `L_NTP` continuing) | The path is written for the spec's joint `L_NTP + β·L_MC`. A frozen-transformer substrate would make `L_NTP` a constant, and T4's bitwise-equality becomes vacuous there. The path runs either way; D2 decides which run is E1. |
| **The spec's warmup** | **Correction 31(b)** / RESEARCH-CONTEXT §12 item 7: what an epoch is on `train()`'s loop | `T_warm = epochs × steps_per_epoch` (`constants.py:258`, `:428-443`); `build_policy` refuses `None` (`loop.py:332-338`). L12 refuses a `t_warm` override that would decide it by the back door. |
| **Falsifier 3b on S0-03** | **D1**: at `M = 16` the oracle is decision-identical to a causal fact/filler rule (0 of 6144 victims differ, per the day roadmap's red-team re-run; not re-verified here) | `RSR(γ = 0.9)` vs `RSR(γ = 0)` cannot fail for an interesting reason on this corpus. The `γ = 0` arm is still built (L12); what it can test is D1's. |
| **Recording E0e's constants** | the owner's D2 / τ-rule ruling (W7) | W7 measured, **unrecorded**: `E_lifetime = 13.1667` (analytic, 632/48), `τ = 0.27555`, `γ_b = 0.3038` (derived; outside §3.5's "order 0.05–0.1", which is RESEARCH-CONTEXT §12 item 2). The learning path does not read them; `ProtectionBias` does. |
| **`ProtectionBias` / `b_enabled=True`** | `docs/queue/items/owner-bias-interface.md` (§12 item 3: `_score` calls `self.bias.b(slots)` at `rsr.py:402`, and the stub declares no `b`), plus §12 items 1–2 | Every learning-path arm is `b_enabled=False` and labelled so (L12). |
| **`ν`, `β`, `γ` values** | E1 (MEASURED, `constants.py:483-519`) | Arms reach them only through L12 overrides. β additionally needs L7 before its sweep means anything. |
| **The μP claim for `ψ̂`** | the ported E0a re-run with unit-norm gestalts (correction 15; RESEARCH-CONTEXT §9 E0a, "DEFERRED, not passed") | See §3 above. |
| **`r_i` as the target at all** | E0d: ρ(`r_i`, LOO Δloss), gated and raw (correction 17). **LOO is truth.** | If E0d finds `r_i` a confound, the path is fine and its target is wrong. |

### 8. Alternatives considered

| alternative | cost | status |
|---|---|---|
| **TD(0)** (spec :241; ablation A8 per correction 2) | Online, lower variance, no end-of-stream truncation. But it bootstraps on a representation that is itself training, and v0.4 had to list that as a failure mode (§7.7). An EMA target copy is live **only** inside A8 (correction 3). | Ablation A8, later. Adding it later needs a `REDUCTION_SWITCHES` entry (§5). |
| **λ-return, `λ < 1`** (§3.3) | It interpolates toward TD and brings back a bootstrap. The spec permits it only "if MC variance is limiting", on synthetic (§7.7 :607). | Not in slice 1. `lambda_return` is FROZEN 1.0 (`constants.py:396-405`). |
| Keep `ψ̂` attached per step, sum `L_MC` online | It holds `S` small graphs for the whole stream, and it spreads the stop-grad over many call sites. | Rejected in favour of detached buffers + one batched recompute (§1). |
| Take `r_i` from the training forward | Refused by correction 20 and `reward.py:88-96`. | Rejected. |
| `attn_dropout = 0` on RSR arms only | The arms would then differ in more than the eviction rule, and E0b's claim (`policy_loop.py:13-16`) would no longer hold between them. | L1 option (b), not recommended. |
| Two optimizers | Equivalent elementwise to one AdamW with an extra group. It adds a second state dict to checkpoint and a second place for betas to drift. | L8. |
| Keep β as the E1 knob | It is inert (Context). E1 would report "no effect". | L7 option (b), not recommended. |
| Ship slice 1 with the shadow buffer | The buffer needs the queries (L5), a softmax design (L4) and a model-side stash, and it is on the fidelity-tested file. | L6. |

---

## DECISIONS FOR BRENDAN

Each is a choice the spec does not fix. **The recommendation is a recommendation.**

| id | decision | options | recommendation |
|---|---|---|---|
| **L1** | Eval-mode source for `α` during training (correction 20) | (a) a second `no_grad` forward per sentence step in `model.eval()`, same pre-write memory and `bos_ctx`, mode restored after; (b) `attn_dropout = 0` on RSR arms; (c) stash pre-dropout `α` from the training forward | **(a).** (c) is still not eval mode, because the queries saw residual and srep dropout. (b) breaks "differ only in the eviction rule." (a)'s cost is unmeasured. S0-02 measured capture + `r_i` at −15.13% sent/s *without* a second forward (`experiments/s0-02/RESULTS.md:31`). Every dropout in `model.py` is `nn.Dropout`, so an eval forward should draw no RNG (believed; T6 checks it). |
| **L2** | Which `c_t` is the regression input **and** the decision input | train-mode `out.srep` (today's decision input, `policy_loop.py:362`, with srep dropout 0.15) vs the eval forward's srep | **Eval srep for both.** They must be the same tensor, and correction 20's argument (noise in a policy-relevant direction) applies to `c_t` as much as to `α`. Under `neg_age` the context is unused, so E0b is unaffected. |
| **L3** | How `λ_shadow` enters (§3.4 :284 "counterfactual targets, down-weighted") | (R1) shadow `r̃` continues the return past eviction, `G = Σ_observed γ^k r + λ_shadow Σ_shadow γ^k r̃`; (R2) shadow slots add extra regression rows `(s_i, c_t)` at post-eviction `t`, loss-weighted by `λ_shadow`; (R3) both | **R1.** The decision at `t_e` reads `ψ̂(s_i, c_{t_e})`, which is trained on pre-eviction samples, and only R1 un-censors *those*. R2 alone teaches `ψ̂(s_i, c_40)` and leaves `ψ̂(s_i, c_5)` biased low. |
| **L4** | Would-be `α` for a shadow slot | (a) insert each shadow slot **singly** into the live softmax (`K` softmaxes over `M+1`); (b) all `K` jointly (one softmax over `M+K`); and the shadow key's `P^(sent)` rank (ADR-0006): (i) the rank it would hold by write order, (ii) its rank at eviction, (iii) none | **(a)+(i).** (a) is the single-slot counterfactual "had it been kept". (b) splits mass among shadows and understates each. (i) matches what the key would carry if the slot were resident. Record the choice in the trace. |
| **L5** | Queries for the shadow score: the forward contracts them away | (a) stash `q` as `CrossAttention.last_query` beside `last_attention` (`model.py:335`), capture-gated; (b) stash the block input and recompute `q` | **(a),** gated by `test_fidelity.py` and `test_reduction.py`. It is an attribute assignment with no arithmetic. It is still a touch to the transcribed forward, so it is Brendan's to allow. |
| **L6** | Slice 1 scope | (a) MC + observe + μP + overrides with `shadow_enabled=False` refused-if-true, the buffer in slice 2; (b) block the whole path on the buffer | **(a).** Every slice-1 run is labelled `shadow=off`, and §1's FIFO-censored-target caveat is written into its RESULTS. No slice-1 run is E1. |
| **L7** | **β is inert under AdamW + isolation** (Context; spec-level) | (a) re-read E1's β sweep as a sweep of `φ`'s LR multiplier (`lr_φ = β · base_lr / m`), by a correction entry; (b) keep β as a loss weight and accept a null sweep; (c) give `φ` an optimizer where scale matters (SGD), which departs from §4.3's μP + AdamW | **(a),** via a correction Brendan writes. It keeps E1's 3-point sweep (and the 27→3 collapse) meaningful. **The registry entry's semantics would change; I have not touched it.** |
| **L8** | Optimizer details for `φ` | one AdamW with a `value_head` group vs two optimizers; weight decay on `φ` (today's global 0.01, `loop.py:427`) vs 0.0; a separate `φ` clip vs none | **One AdamW; `φ` weight decay 0.0 until E0a answers; no `φ` clip, but log `φ`'s grad norm.** Decoupled decay fights the `O(d)` growth §3's arithmetic predicts, and MC has no divergence to clip (§7.7). The transformer clip **excluding `φ`** is not optional (§2). |
| **L9** | `L_MC` reduction and weighting | sum over live slots, then mean over `(row, t)`; optional `w_t = \|memory_t\|/M` as variance reduction (spec :206) | **Unweighted, sum-then-mean.** The overall scale is inert (L7). CLAUDE.md forbids down-weighting as the bias correction, and a `w_t` that is "only variance reduction" is hard to audit apart from that. |
| **L10** | Protocol plumbing for `c_t` and row identity | (a) add `context` (eval `c_t`) to `AttentionTrace` and `row: int = 0` to `MemoryState`, set in `trace_for_row` / `memory_state`; (b) change the `observe`/`select_eviction` signatures on every policy | **(a).** It is additive with defaults, so FIFO/LRU and the test `RecordingPolicy` are untouched. It pairs `r_i(t)` with the `c_t` of the same forward by construction, and it gives `ProtectionBias` its per-row key later. |
| **L11** | `φ`'s RNG | a dedicated **CPU** `torch.Generator` seeded from `(seed, "phi")`; head built on CPU, then `.to(device)` | **Adopt.** It fixes both latent defects (data-order shift; MPS raise) and satisfies §3.7's "seed it from a separate RNG stream". T10 guards it. |
| **L12** | The `train()` overrides path | as specified in §5 (stamped into `frozen`/hash/label; `b_enabled=False` required; `shadow_enabled=True` refused in slice 1; `t_warm` override refused unless declared as the no-warmup ablation) | **Adopt as specified.** |
| **L13** | What β = 0 means with a learned head | (a) `L_MC` not computed: `φ` grads stay `None`, AdamW skips `φ` (including decay), `φ` frozen at init, i.e. a random-content eviction arm; (b) computed ×0: decay still shrinks `φ` | **(a).** It is deterministic and makes "β = 0" mean "no retention learning". T12 guards it. |

### Observations for Brendan (spec-fixed; flagged, not decided)

- **The `k = 0` term is already realized at decision time.** Eviction at step `t` runs after
  that step's forward (`policy_loop.py:352-368`), so `r_i(t)` has been paid and `c_t` was
  produced by a forward that attended to `s_i`. The spec's `G_i(t)` includes `k = 0`
  (spec :249). With `γ = 0` the target is exactly the present retrieval. That is consistent
  with the spec's reading of `γ = 0` as "context relevance" (§2, C4), but it means part of
  every target is visible through `c_t` itself. I propose no change.
- **The FIFO-censored target** (§1) qualifies spec :280's "unbiased FIFO-collected data". It
  may merit a correction entry. I did not write one.
- **β's inertness** (L7) contradicts the premises of spec :280 and :607. It is a correction
  candidate. I did not write one.

## Consequences

- If accepted, slice 1 is implementable without any owner ruling except L1–L13 here. It
  produces **labelled engineering runs**, not E1: `b=off`, `shadow=off`, and `t_warm`
  unresolved.
- Two latent `train()` defects are fixed as a side effect (L11).
- E1 cannot start from this ADR alone. It needs D2, 31(b), D1's framing, L7, E0a, E0d, and
  (for `b`) the bias interface.

## Status of the alternatives

| Alternative | Status |
|---|---|
| TD(0) | ablation A8, not default (correction 2) |
| λ-return | frozen at 1.0; sweep only on measured MC variance |
| train-mode `r_i` | rejected (correction 20) |
| `attn_dropout = 0` on RSR arms | not recommended (L1) |
| β as the E1 knob | not recommended (L7) |
| shadow buffer in slice 1 | not recommended (L6) |
