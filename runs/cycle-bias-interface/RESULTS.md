# cycle-bias-interface — the `ProtectionBias` / `_score()` interface, and what `b` does

- **Falsifier.** *"`ProtectionBias`'s declared interface and `RSRPolicy._score()`'s
  call site describe the same object."*
- **Verdict: FALSIFIED.**
- Ledger: `runs/cycle-bias-interface/ledger.json`. Every number below is a row there.
- Baseline `f1ea1a0`. `git_dirty: true` in the ledger is this untracked run
  directory only; `src/`, `tests/` and `docs/` were not touched. All work in
  throwaway scratch at
  `/private/tmp/claude-501/-Users-keanooo7-second-brain/1bdd6660-ee4c-49c5-99a7-28bf139ef2c9/scratchpad/`
  (`scratch_bias.py`, `run_bias_interface.py`).
- CPU, 1 thread, ~2 s wall. Python 3.12.13, macOS-26.6.2-arm64.
- Geometry is registry-read, not supplied: `M = 40`, `S = 80` (`pg19_e3`, FROZEN),
  `b_max = 1.0` (FROZEN). `d_model = 384` (correction 23's E3 width).
- **Deliberate visible bypasses**, labelled in the ledger as such and *not*
  measurements of those constants: `gamma_b` (DERIVED, E0e), `tau = 0.25`
  (MEASURED, E0e), EMA half-life `= 10.0` steps (from `E_lifetime` assumed `= M`),
  and `RSRConfig` hand-built rather than `from_registry` (cycle-1 wall 1:
  `measurements/ledger.json` still does not exist — re-checked, `False`).

## A — the falsifier

| row | value |
|---|---|
| `part1a_real_ProtectionBias` | `NotImplementedError` at `src/rsr/retention/bias.py:60` |
| `part1b_declared_surface_members_present` | `["reset", "update", "values"]` |
| `part1b_declared_surface_has_b` | `False` |
| `part1b_selftest_values_after_20_updates` | slot 0 (over-utilised) → `b = -1.0`; slot 1 → `+0.90` |
| `part1b_eviction_with_declared_surface` | **raised** `AttributeError: 'DeclaredSurfaceBias' object has no attribute 'b'` at **`src/rsr/retention/rsr.py:377`** |
| `part1c_eviction_after_adding_b_accessor` | **succeeded**, attribution `psi+b` |

A `ProtectionBias` written to satisfy its own docstring and §3.5 — the `± γ_b`
per-slot update, the `[-b_max, +b_max]` clip, the `ū` EMA — **works on its own
terms** (the self-test row: an over-utilised slot is driven to `-b_max`, the others
rise) and **still cannot take one eviction** through `RSRPolicy`. Adding one member,
`b(slots)`, to the identical class makes the same call succeed. The two surfaces are
reconcilable only by one side changing shape, which is the falsifier's death
condition.

## B — which side is self-consistent

**`ProtectionBias`'s declared surface. The spec settles the semantics; it does not
name the accessor.**

§3.5 verbatim: *"A scalar bias `b_i`, added **inside the eviction argmin only**"*,
updated by

```
b_i ← b_i − γ_b   if ū_i > (1+τ)/M
b_i ← b_i + γ_b   if ū_i < (1−τ)/M
b_i ← clip(b_i, −b_max, +b_max)
```

The update reads exactly two things: `ū_i` and `b_i`'s own previous value. Nothing
in §3.5 makes `b` a function of the memory state, so `b` is **state the controller
owns and publishes** — a zero-argument accessor (`values()`) is the faithful
surface. A `b(slots)` accessor asserts that `b` is *computed from* `MemoryState`,
which §3.5 does not say, and correction 4 — which is the only correction that
touches §3.5's mechanism — changes only `γ_b`'s magnitude, not the signature.

The one engineering argument for `slots` is device alignment. It does not hold here:
`_score` already masks dead slots itself one line later (`rsr.py:382`,
`torch.where(live, score, +inf)`, row `part1d`), and the policy already has
`_device_of(slots)`.

**What the spec does not settle:** it names no method, so the final choice of
*code shape* is Brendan's. The two candidates are (i) rename the call site to
`self.bias.values()` and align device in `_score`, or (ii) add `b(slots)` to
`ProtectionBias`. The evidence favours (i).

**Two further under-specifications of the same interface, found while building it:**

1. `update(u_bar)`'s parameter name says the caller owns the EMA, but item 2's
   requirement (*"an EMA rate, not a cumulative sum ... half-life =
   `E[lifetime]/4`"*) is then unenforceable from inside the class, and `__init__`
   has no half-life parameter. The scratch implementation owns the EMA and takes
   the per-step share.
2. **`ProtectionBias` declares no per-slot invalidation hook**, yet
   `RetentionPolicy.on_write`'s own docstring names *"the anti-collapse loop's
   `u_bar`"* as having exactly gauntlet 0.4's shape. Without one, a new occupant
   inherits the previous tenant's `b`. Scratch added `on_write(slot)`.

## C — what `b` actually does to decisions

Controlled design: one driver arm (`b_enabled=False`) owns the memory, so the
`MemoryState` sequence is **identical** across all `γ_b`; every `b_enabled=True`
arm is asked the same question on the same state and **shares the same value-head
object**, so `ψ̂` is bit-identical. `psi_override=None`, `nu=0.0`, `t_warm=0.0`,
`shadow_enabled=False`. 5 seeds × 12 streams × (80 − 40) = 480 evictions/seed.
`ū` is synthetic: a per-slot log-normal salience assigned at write time plus noise,
softmaxed over live slots (heavy-tailed, per item 3).

`attribution_driver = {"psi": 480}`, `attribution_b_arm = {"psi+b": 480}` — zero
`fifo_warmup`, so the score path really ran.
`driver_top2_score_margin_median = 0.3326 ± 0.0207` (z-scored `ψ̂` SD units).

| `γ_b` | frac. evictions **changed** | max \|b\| attained | mean `b` spread over live slots |
|---|---|---|---|
| 0.0 (control) | **0.0000 ± 0.0000** | 0.0 | 0.0 |
| 0.001 | **0.02125 ± 0.00539** | 0.079 | 0.0953 ± 0.0020 |
| 0.01 | 0.25083 ± 0.00731 | 0.79 | 0.9527 ± 0.0203 |
| 0.05 | 0.56417 ± 0.01313 | 1.0 (clip) | 1.9954 ± 0.0042 |
| 0.075 | 0.58667 ± 0.01264 | 1.0 (clip) | 1.9997 ± 0.0005 |
| 0.1 | **0.59500 ± 0.01289** | 1.0 (clip) | 2.0 |

### C1 — correction 4's magnitude claim is confirmed; its "cannot move the argmin" is not

Correction 4 predicts max \|b\| `= 0.001 × 80 = 0.08` SD. Measured: **0.079**
(`= 0.001 × 79`, the longest realized in-stream slot life). The arithmetic is right.

But the operational claim is **not exactly true**: at `γ_b = 0.001` the loop changes
**2.1% ± 0.5%** of evictions, against **0.0% ± 0.0%** for `b ≡ 0` (the harness
self-test). Mechanism, from the ledger: 15.0% ± 1.3% of driver evictions have a
top-2 margin below 0.079 and 28.9% ± 1.9% below 0.158, and the median margin *of the
evictions `γ_b = 0.001` flipped* is **0.0164 ± 0.0072** versus 0.271 ± 0.025 at
`γ_b = 0.1`. So a 0.08-SD bias moves the argmin **only where the argmin was very
nearly tied** — about 1 eviction in 47.

**Reading.** "operationally identical to `b ≡ 0`" is the right *engineering* call
and the wrong *literal* statement, and the difference matters for how D-1 is
described: at `γ_b = 0.001` the loop is not inert, it is **near-tie noise** — it
perturbs decisions the value estimate had no opinion about, which is worse than
inert, because A5 would have measured a small non-zero effect of the wrong kind and
had something to report.

### C2 — correction 4's own range makes `b` the policy

At `γ_b ∈ [0.05, 0.1]` the bias changes **56–60%** of evictions, with `b` saturated
at `±b_max` simultaneously (`mean_b_spread ≈ 2.0 = 2·b_max`) at essentially every
eviction. That is precisely §3.5 item 1's *"if `ψ̂ ≪ b`, **the policy is the balance
controller, not the value estimate** — and it might still beat FIFO for reasons
unrelated to the hypothesis"*, and §3.4's *"if `b` flips a large share of decisions,
the balance controller is the policy."* It is also item 3's own warning realized:
`b` saturated for most slots most of the time.

The curve is steeply concave in `γ_b` and **flat from 0.05 to 0.1** (0.564 → 0.595),
because both ends are already clipped. So on this synthetic `ū`, A5's sweep over
`γ_b` cannot separate 0.05 from 0.1 at all — the interesting variation is **below**
correction 4's range (0.01 → 0.25 of decisions).

**This is not an argument for going back to 0.001.** It is an argument that
correction 4 fixed the magnitude and reopened the takeover question the z-scoring in
item 1 was introduced to close, exactly as `bias.py`'s docstring says it would, and
that §3.4's decision-attribution decomposition is the thing that has to adjudicate
it. Note that `attribution` as currently implemented cannot: it records the string
`"psi+b"` for every eviction whether `b` mattered or not.

### C3 — the `0.25` in the `γ_b` formula

`γ_b ≈ b_max / (0.25 · E[lifetime])` divides by 4 because updates fire *"on order a
quarter of steps"*. Measured on this synthetic `ū` at `τ = 0.25`:
`frac_slot_steps_update_fired = 0.4926 ± 0.0106` — **~half**, not a quarter.
⚠️ This measures the synthetic generator, not reality; the real number is E0e's.
Recorded because it is a factor of 2 in a formula treated as settled.
Also measured: realized mean slot lifetime under the *learned-head* driver is
**26.49 ± 0.53** steps, not `M = 40` — so the formula's `E[lifetime]` is policy
dependent, and `γ_b` derived from a FIFO E0e run is not the `γ_b` the RSR arm's own
lifetimes imply.

## D — the `sd = 0.0000` rows

Every zero-sd row is a ceiling, a clip or a structural constant, and the
`sd_zero_rows_investigated` ledger row says which for each:
`n_evictions_per_seed` is `12 × (80 − 40)`; `max_abs_b` at `γ_b = 0.001/0.01` is
pinned at `γ_b × (S − 1)` because some slot survives a whole stream in every seed;
at `γ_b ≥ 0.05` it is the `b_max` clip; `mean_b_spread` at `γ_b = 0.1` is `2·b_max`
by double saturation; `frac_evictions_b_uniform` is 1.0 at `γ_b = 0` and 0.0 above
it by construction.
**The seed does reach the RNG:** `driver_top2_score_margin_median` sd 0.021,
`realized_mean_slot_lifetime_steps` sd 0.53, `frac_slot_steps_update_fired` sd 0.011,
and every `frac_evictions_changed` row at `γ_b > 0` has sd > 0. `φ` init, gestalts,
salience and the `ū` noise all move with the seed.

## Re-execution

Headline claim (the falsifier), deterministic:

```bash
cd /Users/keanooo7/retrieval-successor-retention && .venv/bin/python -c "
import torch, traceback
from rsr.retention.policy import MemoryState
from rsr.retention.rsr import RSRConfig, RSRPolicy
M, d = 40, 384
class Bias:  # exactly ProtectionBias's declared surface
    def __init__(s,*a,**k): s.b_=torch.zeros(M)
    def update(s,u_bar): pass
    def values(s): return s.b_
    def reset(s): pass
g = torch.Generator().manual_seed(0)
gest = torch.randn(M,d,generator=g); gest/=gest.norm(dim=-1,keepdim=True)
ctx = torch.randn(d,generator=g); ctx/=ctx.norm()
slots = MemoryState(gestalts=gest, written_at=torch.arange(M), live=torch.ones(M,dtype=torch.bool), step=M)
cfg = RSRConfig(nu=0.0,beta=0.0,gamma=0.0,t_warm=0.0,a_max=M,psi_override=None,b_enabled=True,shadow_enabled=False)
try:
    RSRPolicy(cfg, d, bias=Bias()).select_eviction(slots, ctx, M)
    print('NO ERROR')
except Exception as e:
    tb = traceback.extract_tb(e.__traceback__)[-1]
    print(type(e).__name__, '|', e, '|', tb.filename.split('retrieval-successor-retention/')[-1]+':'+str(tb.lineno))
print('declared surface has b():', hasattr(Bias(),'b'))
"
```

Prints:

```
AttributeError | 'Bias' object has no attribute 'b' | src/rsr/retention/rsr.py:377
declared surface has b(): False
```

The `γ_b` sweep re-runs in ~2 s with
`.venv/bin/python <scratchpad>/run_bias_interface.py`, which rewrites this
directory's `ledger.json` (the per-seed fractions are seeded and reproduce exactly;
the `part1b`/`part1c` *victim indices* do not, because those two policies build
their value head from the unseeded global generator — the exception-vs-success
outcome is what the falsifier turns on and that is deterministic).

## What this does NOT establish

- **Nothing about real `ū`.** The utilisation signal is synthetic. `observe()` still
  raises (cycle-1 wall 5), there is no `AttentionTrace` capture, and
  `rsr.retention.reward`'s `r_i` was never in this loop. Every fraction above is
  conditional on a log-normal-salience softmax and on `τ = 0.25`.
- **Nothing about `γ_b`, `τ`, `b_max` or `E_lifetime` as constants.** Those are E0e's.
- **Nothing about loss, perplexity, or whether the bias helps.** `φ` is random and
  untrained; `select_eviction` never fed a language model here.
- **Nothing about collapse.** The failure §3.5 exists to prevent was never induced —
  the synthetic `ū` imposes imbalance by construction rather than letting retention
  and attention close the loop.
- **Nothing on device placement.** CPU only. `_score` adds `self.bias.b(slots)` with
  no device alignment of its own; whether a CPU-resident bias against MPS memory
  raises was not tested.
- **Nothing about the other four cycle-1 walls**, all of which still stand.
