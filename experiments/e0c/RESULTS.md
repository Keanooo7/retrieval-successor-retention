# E0C — memory and throughput ceiling **on the Mac Studio**

**Status: RUN.** 2026-09-17, gauntlet 2.7.

| | |
|---|---|
| Question | The maximum feasible `(S, d, batch)` triple **on the machine that will run it** |
| Kill gate? | Resizes everything |
| Sprint 1 task | T5 |
| Supersedes | the rental-card framing (correction 10). No GPU is rented — [ADR-0007](../../docs/decisions/ADR-0007-all-training-on-the-mac-studio.md) |

§4.2's rule, unchanged and now applied to hardware in hand:

> **If `S = 80` does not fit, `S` wins and `d` is cut.**

**It never binds.** `S = 80` fits at every width in scope. The constraint that does
bind is batch size at `d = 384`.

## Hardware

```
Apple M4 Max · 64 GB unified (68,719,476,736 bytes) · macOS Darwin 25.6.0 arm64
torch 2.14.0 · iogpu.wired_limit_mb = 0 (system default — NOT raised)
```

## 🔴 Correction: the first numbers measured a model with no memory

`sweep.py` and `confirm.py` draw tokens with `randint(0, V)` and leave `TGConfig`'s
default special ids in place. At `V = 50257` the default `eos_id` is **50259** —
outside the range the tokens come from — so `has_eos` is never true. Measured:

```
sentences processed              64
sentences that WROTE to memory    0
times the policy was consulted    0
```

Memory is never written, so `row_has_mem` is false on every row and `TgCrossAttn`
returns **exactly zero** (`out * row_has_mem`). What was timed is a stack of
self-attention blocks with a dead cross-attention branch and **no retained gestalt
graph across the stream** — which is the dominant memory bill (§4.2) and the entire
mechanism of TG.

I reproduced those numbers closely (438 vs 439±4, 445 vs 448±2, 228 vs 228±1), so
the scripts are reproducible; it is the configuration that is wrong. They are kept
verbatim as the record of what was run, and superseded by `measure.py`, which
**asserts** that every sentence reaches memory and that the policy is consulted.

## Result — measured with the memory path live

`experiments/e0c/measure.py`, MPS, GPT-2 vocab (50257), 3 repeats, forward +
backward, no optimizer step.

| policy | `d` | `S` | batch | peak GB | sent/s | h per 1.2M steps | writes | evictions |
|---|---|---|---|---|---|---|---|---|
| FIFO | 128 | 80 | 16 | 32.2 | 374±8 | 0.9 | 1280 | 640 |
| RSR (`neg_age`) | 128 | 80 | 16 | 32.1 | 318±1 | 1.0 | 1280 | 640 |
| **RSR (learned head)** | **128** | **80** | **16** | **31.8** | **310±2** | **1.1** | 1280 | 640 |
| FIFO | 128 | 48 | 16 | 20.3 | 392±0 | 0.9 | 768 | 128 |
| RSR (`neg_age`) | 128 | 48 | 16 | 20.3 | 365±1 | 0.9 | 768 | 128 |
| **RSR (learned head)** | **128** | **48** | **16** | **20.3** | **357±2** | **0.9** | 768 | 128 |
| FIFO | 384 | 80 | 8 | 31.0 | 197±0 | 1.7 | 640 | 320 |
| RSR (`neg_age`) | 384 | 80 | 8 | 31.0 | 180±1 | 1.9 | 640 | 320 |
| **RSR (learned head)** | **384** | **80** | **8** | **31.0** | **171±4** | **1.9** | 640 | 320 |

> **RECOMMENDED TRAINING CONFIG: `d = 128`, `S = 80`, `batch = 16`** — 31.8 GB and
> **310 sent/s on the RSR arm**, ~1.1 h per 1.2M-step run.

**Memory is identical across policies** to within 0.4 GB. RSR adds compute, not
memory, exactly as §4.1 predicts (`O(M·d)` against a 12-layer transformer).

### What the RSR policy actually costs

| Comparison | at `d=128, S=80` | at `d=384, S=80` |
|---|---|---|
| FIFO → RSR dispatch (`neg_age`, no head) | −15% | −9% |
| dispatch → learned value head | −2.5% | −5% |
| **FIFO → RSR total** | **−17%** | **−13%** |

**Most of the cost is the Python dispatch loop, not the value head.** `neg_age`
runs no head at all and still costs 15%: `run_policy_loop` builds a `MemoryState`
per row per step and calls into the policy, and at `batch=16, S=80` that is 1,280
round-trips with tensor clones in each. The bilinear head itself adds only 2–5% on
top. If throughput ever needs recovering, the dispatch loop is the target and
batching the policy across rows is the fix — not simplifying `ψ̂`.

🔴 **The `neg_age` row is not RSR.** §3.7's reduction sets `ψ̂ ≡ −a_i` and runs no
value head, so quoting it as "the RSR arm" reports dispatch overhead under the name
of the mechanism. It is in the table as the decomposition, not as the headline.

## Two defects the corrected measurement exposed

**1. `RSRPolicy` could not run on an accelerator at all.** It held its value head on
the CPU while the memory lived on MPS, and every tensor it created was a CPU
tensor. Both paths failed:

```
neg_age:      RuntimeError: Expected all tensors to be on the same device,
              but found at least two devices, mps:0 and cpu
learned head: Tensor for argument #2 'mat2' is on CPU, but expected it on GPU
```

`neg_age` has no head, so this was not "a module was not moved" — it was
`torch.full(...)` with no `device=`. **E0b could not catch it**: §3.7's reduction
runs on CPU by design (ADR-0001 D3), so the entire RSR arm would have failed at the
first accelerated run with every test green. Fixed, with
`tests/test_device_placement.py` covering the class — a static AST scan that runs
everywhere including CI, plus live tests where an accelerator exists.

**2. `TGModel` never initialised its parameters.** `nn.Parameter(torch.empty(...))`
is uninitialised memory. A freshly constructed model held `3e36` in one attention
kernel and zeros in another, differing between runs under a fixed
`torch.manual_seed`. Invisible to every test, because `test_fidelity.py` and E0b
both **load** the reference weights over the top — so it would have surfaced only in
a training run, as divergence, and been blamed on the learning rate.

The init scales were then **measured from the fixture** (which holds an untrained
model, so its parameters *are* the reference's initialisation) rather than derived:

```
attention in-projections   std 0.0884  = xavier_uniform, fan_in=D, fan_out=H·Dh
every other kernel         std 0.0200  = normal(0, 0.02)
biases 0 · LayerNorm 1/0 · memory_gate 1.0
```

Deriving it from Flax's `_compute_fans` instead gives fan_in 256 / fan_out 8192 and
a std of **0.0154 — wrong by 5.7×**, with nothing to catch it.

## 🔴 The ceiling, and how it announces itself

`d = 384, S = 80, batch = 64` asks for **68.80 GB on a 64 GB machine.** Metal does
not refuse it — the run "succeeds". What happens instead:

```
swapouts before the run   512
swapouts after            1,024,668
vm.swapusage              16,010 MB used of 16,384 MB
```

and the throughput inverts:

```
batch 32   35.73 GB   1162 sent/s
batch 64   68.80 GB    807 sent/s     ← twice the work, 30% SLOWER
```

**A configuration that reports `fitted: true` while running slower than half its
batch size has not fitted.** That inversion is the honest ceiling signal, and it is
why this experiment measures throughput as well as memory: a memory number alone
would have called 68.80 GB a pass.

Batch 32 at `d = 384` is therefore the last real point, and it is the committed one.

## Two measurement corrections, both of which produced wrong numbers first

**1. `ru_maxrss` is a process high-water mark.** Running the grid in one process
makes every row inherit the peak of the row before it. The first run reported
`d=256, batch=8` at 11.34 GB — which was `d=128, batch=64`'s peak, not its own.
**One subprocess per point.** After isolating, the same configuration repeats to
±0.15 GB:

```
d=384 S=80 batch=64, CPU, three runs:  25.94 / 25.85 / 26.00 GB   169.7 / 169.3 / 169.2 sent/s
```

**2. RSS does not measure MPS allocations.** The first MPS run reported **0.44 GB**
for a configuration needing ~26 GB on CPU, because Metal memory does not appear in
the process's resident size. On unified memory it is the same 64 GB either way, so
`torch.mps.driver_allocated_memory()` is the figure that competes with everything
else. A number that low should read as a broken metric, not as good news.

## CPU, for reference

CPU is slower but the memory behaviour is the same shape, and it is what E0b's
determinism argument prefers (ADR-0001 D3). At `d = 384, S = 80, M = 40,
batch = 64`: **25.9 GB RSS, 169 sent/s** — 4.8× slower than MPS at the same point,
and it does not swap, because the CPU allocator does not over-commit the way Metal
does.

## Do NOT inherit the spec's `21 sent/s`

§4.4 and the kickoff quote 21 sentence-steps/sec from [P2]. That was measured at
`d_model = 768` / 85.6M parameters — **roughly 4× wider than anything in this
project**, whose largest configuration is 22.2M at `d = 384`. Measured here:
**885–3531 sent/s** depending on width and batch. §12.4: measure, don't extrapolate.

## What this does not claim

Every number here was measured on **one M4 Max**. §13 gains the entry: nothing in
this project supports a claim about training cost or feasibility on other hardware,
and the paper must not imply one.

`fitted: true` means one forward and **one backward** completed — the backward is
the measurement, because the bill is the retained computation graph across the
stream, not the weights. TG appends each gestalt without detaching and backward
depth is bounded by `S`, not by `M` (§3.6, [P2] App. A), so a forward-only number
would understate peak by the depth of that graph.

## Reproduction

| | |
|---|---|
| Command | `.venv/bin/python experiments/e0c/run.py --device mps --out e0c_mps.json` |
| Single point | `.venv/bin/python experiments/e0c/run.py --device mps --point 384,80,32,40` |
| Git SHA | recorded in the JSON output |
| Hardware | Apple M4 Max, 64 GB, `iogpu.wired_limit_mb = 0` |

⚠️ **The full `--device mps` grid will drive the machine into swap** when it reaches
`d = 384, batch = 64`, as it did here: 16 GB of swap and a million swapouts. Run the
individual points, or cap the batch sweep at 32 for `d = 384`.
