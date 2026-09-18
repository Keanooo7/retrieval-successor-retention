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

## Result — the committed triples

**E3 shape (`S = 80`, `M = 40`), MPS:**

| `d` | batch | driver memory | sent/s | params |
|---|---|---|---|---|
| 128 | 64 | 24.63 GB | 3457 | 2.68M |
| 256 | 32 | 23.57 GB | 1673 | 10.05M |
| 384 | 16 | 17.49 GB | 885 | 22.25M |
| **384** | **32** | **35.73 GB** | **1162** | 22.25M |
| 384 | 64 | 🔴 **68.80 GB** | 807 | 22.25M |

**Synthetic shape (`S = 48`, `M = 16`), MPS** — E1/E2, the approved scope:

| `d` | batch | driver memory | sent/s |
|---|---|---|---|
| 128 | 64 | 13.95 GB | 3531 |
| 256 | 64 | 26.28 GB | 2217 |
| 384 | 32 | 19.38 GB | 1243 |
| **384** | **64** | **38.38 GB** | **1466** |

> **COMMITTED: `S = 80`, `d = 384`, `batch = 32` for the E3 shape; `S = 48`,
> `d = 384`, `batch = 64` for synthetic.** Both on MPS, both with roughly 25 GB of
> the 64 GB left over.

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
