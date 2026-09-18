# cycle-arm-identity — is `train()`'s RSR arm a different arm from its FIFO arm?

Every number below is a row in `runs/cycle-arm-identity/ledger.json`. Nothing here
is typed from memory.

| | |
|---|---|
| git sha | `4245b7daa744a3872e23694ed0ad5801f8f06655` |
| tree | **dirty** — one untracked file, `scripts/canary.py`, left by a prior cycle. No code path in this experiment imports it. |
| hardware | Mac Studio M4 Max, 64 GB (ADR-0007). CPU and MPS. |
| python / torch | 3.12.13 / 2.14.0 |
| ledger | `runs/cycle-arm-identity/ledger.json` |
| raw series | `runs/cycle-arm-identity/partA_series.json`, `runs/cycle-arm-identity/walls.json` |

## Falsifier

> Brendan's reading: `src/rsr/train/loop.py`'s `policy_name='rsr'` and
> `policy_name='fifo'` runs are the **same arm**. `train()` stamps `policy_name`
> into `run_id` and into the frozen config the heartbeat header records, then
> constructs `FIFOPolicy()` unconditionally, so a heartbeat reading
> `"policy": "rsr"` is a FIFO run.
>
> The reading dies if two runs at identical seed and config, differing only in
> `policy_name`, have trajectories that differ by more than the run-to-run floor.

**Verdict: `survived`.** The reading could not be falsified. The arms are one arm.

## A — the identity, measured

Config for all five runs: `d=128, steps_per_stream=48, batch=8, iters=10, seed=0,
memory_slots=16, lr=1e-3`. Separate `out_dir` per run. `policy_name` is reachable
only from the Python API — `main()` has no `--policy` flag (`cli_has_policy_flag =
False`).

A bare fifo-vs-rsr comparison on MPS cannot separate "same code path" from
"different code path, difference hidden by float noise", so the experiment carries
a **replicate**: `fifo` run twice at the same seed into different directories. The
replicate *is* the noise floor.

| pair | `loss` bit-identical | max&nbsp;|Δloss| over 10 beats | `config_hash` differs |
|---|---|---|---|
| `fifo_cpu` vs `rsr_cpu` | **True** | **0.0** | True |
| `fifo_mps_a` vs `rsr_mps` | False | 3.1789143850602386e-07 | True |
| `fifo_mps_a` vs `fifo_mps_b` *(replicate — the floor)* | False | **3.1789143850602386e-07** | False |
| `rsr_mps` vs `fifo_mps_b` | False | 3.1789143850602386e-07 | True |

The cross-arm MPS difference is **the same number** as the same-arm replicate
difference (`mps_nondeterminism_floor_equals_cross_arm_diff = True`). There is no
policy signal in it. On CPU, which is deterministic, the two arms are bit-identical.

`final_loss_cpu_two_runs`: n=2, mean 1.7659316062927246, **sd 0.0**
(`sd_exactly_zero` flagged). `final_loss_mps_three_runs`: n=3, mean
1.7557916641235352, sd 0.0 — the final beat coincides across all three MPS runs;
the 3.2e-07 lives in intermediate beats.

What the run nevertheless reports about itself:

- `heartbeat_policy_field.rsr_mps = "rsr"`, `heartbeat_policy_field.rsr_cpu = "rsr"`.
- `run_id` differs (`rsr-d128-s48-b8-…` vs `fifo-d128-s48-b8-…`), because the
  frozen config the hash is taken over contains `policy_name`.
- `attribution.rsr_mps = None` on every beat — `FIFOPolicy` has no
  `attribution()` method, so `hasattr` is False. **That null is the tell**, and it
  is the only one visible in the heartbeat.

This is corpus-sheet defect 2 (experimental arm and control silently the same arm)
relocated from `RSRConfig`'s defaults into `src/rsr/train/loop.py:138`.

## B — the wall list

Ordered as a run trying to build a real RSR arm hits them. Each was worked around
in throwaway scratch only far enough to reach the next.

1. **`src/rsr/constants.py:829` — `UnmeasuredConstant`.** From
   `RSRConfig.from_registry('synthetic', steps_per_epoch=100.0)`, the only
   supported constructor. `'nu' is MEASURED and has no logged value in scope
   'synthetic'. It is produced by **E1** …`. Root cause:
   `measurements/ledger.json` **does not exist** (`measurements_ledger_exists =
   False`). `nu` is only the first of `nu, beta, gamma, T_warm, A_max`.
2. **`src/rsr/retention/rsr.py:300` — `ValueError`.** Hand-building
   `RSRConfig(...)` to bypass wall 1 leaves `b_enabled=True` (the dataclass
   default): `b_enabled=True needs the anti-collapse bias (section 3.5), whose
   gamma_b and tau are measured by E0e. …`
3. **`src/rsr/retention/bias.py:60` — `NotImplementedError`.** Supplying the real
   object wall 2 demands: `ProtectionBias(capacity=16, b_max=1.0, gamma_b=0.07,
   tau=0.25)` → `Sprint 2. Spec section 3.5; gamma_b and tau from rsr.constants
   (E0e).` The constructor is a stub; so are `update`, `values`, `reset`.
4. **`src/rsr/retention/rsr.py:377` — `AttributeError`.** Passing a stand-in with
   `ProtectionBias`'s **entire declared surface** (`update`/`values`/`reset`) and
   driving one eviction: `'BiasShim' object has no attribute 'b'`. The offending
   line is `score = score + self.bias.b(slots)  # type: ignore[attr-defined]`.
   `_score()` calls a method `ProtectionBias` does not declare, and the
   `type: ignore` is what let the two surfaces drift apart. **Even a completed
   `ProtectionBias` would not satisfy `_score()`.**
5. **`src/rsr/retention/rsr.py:433` — `NotImplementedError`.** With
   `b_enabled=False` the policy constructs and evicts. `observe()` — the only path
   by which `φ` receives gradient — raises: `Sprint 2: the MC return and the shadow
   buffer (sections 3.3-3.4). r_i itself is implemented in rsr.retention.reward.`

Three further walls are **absences**, so they raise nothing and would be silent:

- `policy_loop_calls_observe = False`. `run_policy_loop` never calls
  `policy.observe(...)`, so wall 5 would never even fire from a training run —
  `ψ̂` would simply never train. There is also no `AttentionTrace` capture
  feeding it.
- `loop_py_value_head_in_param_groups = True` — `build_param_groups(model, None,
  …)` passes a literal `None` for the value head, so `φ` would get no muP
  parameter group.
- `cli_has_policy_flag = False` — the CLI cannot select an arm at all.

## C — the eviction path itself, with the walls bypassed

`psi_override=None` (learned head), `b_enabled=False`, `nu=0.0`, `t_warm=0.0`,
`shadow_enabled=False`, `d=128`, `M=16`, `S=48`, batch 8, CPU, forward-only, no
optimizer. A shim asks `FIFOPolicy` the same question on the **identical**
`MemoryState` at every call.

| row | value |
|---|---|
| `partC_n_evictions` | 256 |
| `partC_attribution_counts` | `{"psi": 256}` — **zero `fifo_warmup`** |
| `partC_victim_disagreements_same_state` | 239 |
| `partC_victim_disagreement_fraction` | 0.93359375 |
| `partC_loss_rsr_forward_only` | 237.985595703125 |
| `partC_loss_fifo_forward_only` | 237.9049530029297 |
| `partC_loss_abs_diff` | 0.0806427001953125 |

So the eviction path works, is not a disguised warmup, and moves the measured
loss by ~4 orders of magnitude more than the MPS noise floor. **A real RSR arm
would be visibly different from FIFO in exactly the quantity part A compared.**
The loop is not producing one.

## Re-execution

```bash
cd /Users/keanooo7/retrieval-successor-retention && rm -rf /tmp/chk-fifo /tmp/chk-rsr && .venv/bin/python -c "
import json
from rsr.train.loop import train
k=dict(d=128,steps_per_stream=48,batch=8,iters=10,seed=0,device='cpu',ckpt_every=0)
a=train(out_dir='/tmp/chk-fifo',policy_name='fifo',**k)
b=train(out_dir='/tmp/chk-rsr', policy_name='rsr', **k)
R=lambda p:[json.loads(l) for l in open(p)]
L=lambda r:[x['loss'] for x in r if x['kind']=='beat']
ra,rb=R(a['heartbeat']),R(b['heartbeat'])
print('policy fields:',ra[0]['config']['policy'],rb[0]['config']['policy'])
print('run_ids differ:',a['run_id']!=b['run_id'])
print('loss sequences bit-identical:',L(ra)==L(rb))
print('max abs diff:',max(abs(x-y) for x,y in zip(L(ra),L(rb))))
"
```

Prints:

```
policy fields: fifo rsr
run_ids differ: True
loss sequences bit-identical: True
max abs diff: 0.0
```

## What this does NOT establish

- Nothing about **whether RSR beats FIFO**. Part C's 0.08 loss gap is an
  **untrained random** `φ` at one seed, forward-only, one batch, 10 iterations of
  nothing. Its sign is meaningless and it is not an arm comparison.
- Nothing about whether `observe()`/the MC return/the shadow buffer are *correct*
  once written — only that they raise or are never called.
- Nothing multi-seed. Every part-A comparison is seed 0. The `sd`s reported are
  run-to-run nondeterminism at a fixed seed, not seed variation.
- Nothing at the widths the approved scope runs (`d=384`, `S=80`, `M=40`), and
  nothing on PG-19.
- `test_reduction.py` and `test_fidelity.py` were **not** run; the full suite was
  not run (one experiment per cycle). This says nothing about E0b.
- Part C used a hand-built `RSRConfig` with `gamma=0.97` supplied by me. That is a
  deliberate, visible bypass of wall 1 in throwaway scratch, not a frozen value,
  and it must not be read as a measurement of `gamma`.
