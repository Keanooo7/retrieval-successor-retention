# S0-02 — what the §3.2.1 capture bridge costs

**Brief:** `docs/lab-notes/dispatch-S0-02-capture-bridge.md`, AMENDED 2026-09-20 at
`d537cc88cdff31b5bec6d4822a3a53b5bb23578a`.
**Run id:** `s0-02-capture-bridge` · ledger `runs/s0-02-capture-bridge/ledger.json`
**Manifest:** `runs/s0-02-capture-bridge/manifest.json`, config hash
`56baf95a8a4048585733f6496c771e4d2e489fecac6532afbfb36323e8ae8e80`, frozen
**before** the run.
**Git sha:** `c81ac35abb513c6864be684ecc6e39e684653e28` · branch
`studio-2026-09-20c`
**Hardware:** Apple M4 Max, 64 GB, Darwin arm64, MPS · torch 2.14.0

## Command

```
.venv/bin/python experiments/s0-02/measure_capture_cost.py \
    --device mps --repeats 3 \
    --out runs/s0-02-capture-bridge/capture_cost.json
```

`d = 128`, `S = 80`, `batch = 16`, `M = 40`, `V = 50257`, 64 tokens/sentence,
model in **eval** mode, forward + backward, **no optimizer step**, three arms
interleaved round-robin within each repeat, one discarded warm-up per arm.

## Result — bar item 2

| arm | sent/s (3 repeats) | mean ± sd | Δ vs off | peak GB | `observe()` calls | `r_i` computed |
|---|---|---|---|---|---|---|
| `capture_off` | 390.08, 387.95, 388.36 | **388.80 ± 1.13** | — | 28.75 | 0 | 0 |
| `capture_on` | 360.89, 360.11, 360.38 | **360.46 ± 0.40** | **−28.34 sent/s (−7.29%)** | 28.79 | 1280 | 0 |
| `capture_on_ri` | 330.75, 330.21, 328.93 | **329.96 ± 0.94** | **−58.83 sent/s (−15.13%)** | 28.80 | 1280 | 1264 |

**The bridge alone costs 28.3 sent/s. `r_i` on top of it costs a further 30.5
sent/s.** Memory is flat across all three arms to within 0.05 GB — capture adds
compute, not memory, which is the same shape E0c found for the policy itself.

`r_i` is computed 1264 times rather than 1280 because each of the 16 rows' first
step attends over an **empty** memory (`n_live == 0`), and a step with no live
slot has no demand to apportion. 1280 − 16 = 1264. Stated because a count that
does not match the obvious number is exactly where a silent skip hides.

### Off by default, and free when off — the falsifier, run

Timing `capture_off` against itself proves nothing: both arms are in the same
binary. The real question is whether the *presence* of the bridge slowed the
default path. Measured with `experiments/e0c/measure.py` — **an untouched harness,
identical on both sides** — same machine, same sitting, same config:

| tree | command | sent/s |
|---|---|---|
| **pre-bridge**, `a1c9e857` (git worktree) | `e0c/measure.py --device mps --policies fifo --configs 128:80:16 --repeats 3` | **369 ± 11** |
| **with the bridge**, `c81ac35` | the same command | **370 ± 9** |

**Within noise. `F-capture-free` survives.** Together with
`test_observe_is_off_by_default` (a policy that raises inside `observe` survives a
default run) and the harness's own assertion that `observe()` ran 0 times in the
control arm, "off by default and free when off" is a measured property here, not a
design intention.

### 🔴 `310 ± 2 sent/s` is not used as a denominator, and here is what it *is*

The brief's bar item 2 offers E0c's **310 ± 2 sent/s (RSR arm)** as the yardstick.
Per the amendment, it is cited to `experiments/e0c/RESULTS.md:58,67` and **to no
ledger row** — no ledger in this project holds a throughput key at all. It is used
here as a *prediction under test* and for nothing else. Three reasons the delta
above is reported in absolute sent/s rather than as a fraction of it:

1. **It is the RSR arm; this run's control is FIFO.** E0c's FIFO at the same
   config is **374 ± 8**, and that is the comparable number.
2. **E0c ran in train mode; this runs in eval.** `attn_dropout = 0.2` is live in
   the first and off in the second, and correction 20 / D-F *requires* eval for
   the retention target — so the control had to run in eval too, or the arms would
   differ in two things. 388.8 against 374 ± 8 is +1.9 sd, in the direction
   removing dropout predicts.
3. `docs/RESEARCH-CONTEXT.md:736-744` disqualifies 310 as a **training** rate
   anyway: random tokens, one data shape, no optimizer step.

## Pre-registered expectations, scored

Written into the manifest before the run, and scored honestly, including the one
that failed.

| # | Expectation | Outcome |
|---|---|---|
| 1 | `capture_off` lands near E0c's FIFO 374 ± 8, not near the RSR 310 | ✅ 388.8 ± 1.1, +1.9 sd, direction explained by eval mode |
| 2 | `capture_on` is slower by a visible margin | ✅ −28.3 sent/s, −7.3%, and sd 0.4 |
| 3 | `capture_off` is unchanged by the bridge's presence | ✅ 369 ± 11 → 370 ± 9 |
| 4 | **`r_i` costs LESS than the bridge itself** | 🔴 **FALSIFIED.** It costs *more*: −30.5 vs −28.3 sent/s |

**Expectation 4 was wrong and the reason is worth carrying forward.** The
arithmetic argument — six `[H, M, D]` norms are cheaper than six
`[B, M, H, Dh] × [H, Dh, D]` projections — is correct about FLOPs and irrelevant
to the measurement. `retrieval_demand` is called **1264 times from Python**, once
per live row per step, each call launching a handful of small MPS kernels on
`[6, 2, 40, 128]` tensors. The bridge's einsums are 6 launches per *step* on
batch-shaped tensors. This is launch-bound, not FLOP-bound, and it is the same
finding E0c already reported one level up: *"most of the cost is the Python
dispatch loop, not the value head."*

The actionable form: **`r_i` should be computed batch-wise, not per row.**
`CrossCapture` already holds `[L, B, H, M]` and `[L, B, H, M, D]`, so a batched
`retrieval_demand` is a reshape away and would collapse 1264 call sites into ~80.
Not done here — `reward.py`'s 11 tests pin the per-row signature and bar item 1
requires them to pass **unchanged**.

## What this does not measure

- **Training throughput.** No optimizer step, random tokens, one data shape. Same
  caveats as E0c, inherited on purpose so the two are comparable.
- **CPU.** MPS only (ADR-0001 D3: MPS is best-effort; CPU is what E0b needs for
  determinism, not what a throughput number needs).
- **Whether `r_i` is a good target.** That is E0d, and E0d was blocked on exactly
  this bridge.
- **The `Q_tok` collapse's effect on E0d.** ADR-0008 measures that mean-vs-sum
  cannot change `r_i` and that EOS-only can; which one correlates better with
  leave-one-out Δloss is an E0d row and is not answered here.

## Mutations — gauntlet 1.7

Regenerated record: `docs/mutation-battery.md`; raw
`runs/s0-02-capture-bridge/mutations.json`.

**39/39 proven, `--check` exit 0.**

| Mutation | Gate | Verdict | on-gate | off-gate |
|---|---|---|---|---|
| `W_O` dropped from the capture path | `test_capture_bridge` | **PROVEN** | 4 | **0** |
| `observe()` is never reached | `test_observe` | **PROVEN** | 2 | **0** |

**The `W_O` mutation turns only the new tests red**, and it is bar item 3. It is
needed *because dropping `W_O` is shape-compatible*: `reward.contribution` norms
over the last axis, and `[L, H, M, Dh]` norms just as happily as `[L, H, M, D]`.
Nothing raises, no shape assertion fires, and `r_i` silently becomes
norm-weighted raw attention — v0.1's rejected definition wearing the new one's
name. The four tests it reddens:

```
tests/test_capture_bridge.py::test_the_trace_has_the_shapes_the_reward_module_declares
tests/test_capture_bridge.py::test_W_O_is_load_bearing_in_the_capture
tests/test_capture_bridge.py::test_wo_v_is_the_projected_value_not_the_raw_value
tests/test_capture_bridge.py::test_the_collapse_reproduces_the_real_increment
```

### 🔴 The first version of bar item 3's own test was vacuous

`test_W_O_is_load_bearing_in_the_capture` originally perturbed `attn_out_proj` on
**every** cross-attention block and asserted `r_i` moved. Against the mutation it
was written for, it **stayed green**: layer `l`'s output enters the residual
stream and moves layer `l+1`'s *attention*, so `r_i` changes whether or not `W_O`
is in the capture. It was detecting "the model changed."

That is the brief's own warning landing from the other side — bar item 3 says *"if
`r_i` is unchanged, the capture is wrong and the test is vacuous"*; here `r_i`
changed, for the wrong reason. It was caught only because the battery names which
tests a mutation reddens, and three *other* tests were carrying the gate. The
perturbation is now confined to the **last** cross-attention block (which is the
last block, so nothing downstream attends to memory), and the test asserts `alpha`
is bit-identical before and after — leaving `wo_v` as the only route by which
`r_i` can move.

### Three pre-existing couplings, now declared

Three `reward.py` mutations now also redden one new test each. Declared with a
reason rather than tolerated, and the reason is this cycle's whole point:
`test_reward.py` asserts a property on a hand-built `AttentionTrace`,
`test_capture_bridge.py` asserts the same property end to end on a real forward
pass. Those were two disconnected claims until the bridge existed. **A `reward.py`
mutation that reddened only the fixture test would now mean the bridge does not
reach the reward.**
