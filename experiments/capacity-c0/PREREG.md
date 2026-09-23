---
run_id: capacity-c0
falsifier: >-
  A deterministic CPU training job on the Mac Studio produces bit-identical output no matter
  how many other such jobs run beside it, at a fixed per-job thread count (1 and 5). The lane
  split in scripts/orchestrator/lanes.py rests on "CPU runs are bit-exact".
decision_rule: >-
  Falsifier first: if any determinism-sweep job differs in output hash from the solo run at the
  same thread count, the verdict is falsified, none of the five c0.* capacity keys is written
  (so `lanes generate` refuses on missing keys), and the run exits 1. Otherwise survived, and each
  capacity key is computed by its rule in the thresholds block, exactly as written. Any stop
  before completion (deadline, exception, a killed job) writes ledger status "partial", which
  `lanes generate` refuses.
seeds: [0]
thresholds:
  quiet_machine: "at start, no process outside this run with RSS > 2 GB (for example mlx_lm.server, llama-server) -> exit 3 'machine not quiet'. The only exception is a signed R-*-mlx-servers ruling whose text says the servers STAY UP; the executor quotes that text in the report, measures with them up, and records their RSS in c0.foreign_rss_gb. Under a signed ruling that says they YIELD (R-2026-09-22-mlx-servers, as proposed), the run stops the owner's LLM servers, identified by command line only, BEFORE this check. It records their PIDs, command lines and RSS freed (c0.stopped_servers), then restarts them when C0 completes. It never stops any other process."
  output_hash: "repr(final loss) + sha256 over the model's state_dict in sorted key order, feeding each key's name, dtype, shape and t.detach().contiguous().cpu().numpy().tobytes(). The checkpoint FILE is never hashed: it carries unseeded python/numpy RNG state (src/rsr/train/checkpoint.py RngState.capture) and a zip record name derived from the file name, so identical runs give different file bytes."
  job_isolation: "every job has its own out_dir: runs/capacity-c0/sweep/<part>/t<threads>/k<k>/rep<r>/job<j>"
  training_workload: "S0-03's CONFIG (imported), seed 0, iters 10, CPU. This is the real cpu-det workload: its RSS (~7 GB per job, measured by the pre-push review) is what limits concurrency"
  memory_cap: "All memory in GiB (bytes / 2^30). For each workload w, R_w = the solo 1-thread job's peak RSS, read EXACTLY as ru_maxrss from os.wait4 on the child (not ps sampling, which misses the per-iteration peak). k_mem_w = floor((hw.memsize/2^30 - 8) / R_w). No sweep level may exceed its workload's k_mem. Recorded as c0.r_job_gib.<w> and c0.k_mem_<w> for w in {training, core}. At R_training ~ 6.8 GiB the determinism sweep's k=8 level sits on the edge and may be dropped (to k=4 as the top level); that is the rule working, not a failure"
  determinism_sweep: "training workload. 1-thread jobs at k in {1, 2, 4, 8} up to k_mem_training, 2 repetitions; plus 5-thread jobs: solo, then 3 concurrent if 3 <= k_mem_training (the retrieval-curve layout)"
  core_workload: "the same CONFIG with batch 4 (per-sample cost ~0.45 s vs training's ~0.5 s; ~2.5 GiB peak, so 16 fit), seed 0, iters 30 (>= ~60 s per job, so process start-up does not dominate), CPU, 1 thread. It measures cores without memory confounding the answer; it is not a training result. Batch 1 was rejected in pre-push review: it is dominated by per-step Python overhead that scales perfectly and would overstate scaling"
  core_sweep_k: [1, 2, 4, 8, 10, 12, 14, 15, 16]
  repetitions: "2 per k. A job's time = its train-loop time from its own heartbeat (first beat to last beat), excluding process start-up; makespan(k) = the longest job time at that level, and the SLOWER of the two repetitions is used"
  cpu_det_slots: "core workload. T(0) = 0; T(k) = k / makespan(k). g_i = (T(k_i) - T(k_{i-1})) / (k_i - k_{i-1}). cpu_det_slots = the largest k_i with g_j >= 0.5 * T(1) for every sweep step j <= i. The first step always passes, so the minimum is 1. Slots count THREADS (scripts/orchestrator/slot.py job_threads)"
  battery_cpu_slots: "suite wall time W(n) = `pytest -rs --tb=no` at OMP/MKL threads n in {1, 2, 4, 8}, one run each; the smallest n with W(n) <= 1.15 * min W. Clamped to cpu_det_slots (raw value recorded as c0.battery_cpu_slots_raw)"
  mps_reserves_cpu_slots: "the training workload on device mps, iters 50, alone, thread count UNCAPPED; sample the process tree's `ps %cpu`/100 at 1 Hz (a decaying average on macOS, so it understates bursts; stated, not corrected). ceil(p95), minimum 1, clamped to cpu_det_slots (raw as c0.mps_reserves_cpu_slots_raw)"
  agent_sessions: "POLICY value 2: the manager verifies serially, so more concurrent agents produce work faster than it can be verified. Feasibility check only, with no new API spend: the RSS of the process tree of the claude process running C0, sampled at 1 Hz; if 2 * p95 > 0.10 * physical RAM the value is 1"
  reserve_gb: "agent_sessions * p95 agent RSS + 4.0 GB OS margin (+ c0.foreign_rss_gb if servers stay up by ruling), rounded UP to 0.5 GB. It deliberately EXCLUDES cpu-det job memory, see conflicts"
  conflicts: "reported in RESULTS.md, never resolved by changing a rule: (a) cpu_det_slots < 15 blocks retrieval-curve; (b) k_mem_training < cpu_det_slots means the cpu-det lane, which admits by threads only, can admit more training jobs than memory holds; (c) reserve_gb + c0.mps_job_peak_gb > physical RAM (GiB) means no MPS job of this workload is ever admitted. c0.mps_job_peak_gb = the max heartbeat mem_gb (torch.mps.driver_allocated_memory) of C0's own MPS run -- a C0 ledger key, measured on the S0-03 workload, not E0c's S=80 figure"
  deadline_hours: 4
---

# PREREG — C0: measure this machine's lanes

**Written 2026-09-22 by the MacBook owner-proxy window, before any measurement on the Studio.** The
pre-push review measured one solo job's RSS on the MacBook (about 7 GB), and that is why memory
appears below. No capacity number exists. This file is committed alone, ahead of the brief and of
any code. The text above an amendment is never edited.

**run_id:** `capacity-c0` · **machine:** the Mac Studio (M4 Max, 16 cores, 64 GB) · **seed:** `0`.

## Why

Every lane refuses with exit 3 until `ops/lanes.json` exists, and only `lanes generate` writes it,
from this run's ledger (`scripts/orchestrator/lanes.py`, `C0_LEDGER_KEYS`). Liveness wiring, the
retrieval curve and therefore the Sprint 0 gate all wait on C0. **Capacity is measured on this
machine, never typed** (`CLAUDE.md`, ADR-0007).

## The falsifier, and why it comes first

Lanes treat CPU results as bit-exact, which makes a result's lane part of its provenance. If a CPU
job's output depended on how busy the machine was, no `cpu-det` result could be reproduced, and
every capacity below would be admitting jobs into a lane whose premise is false. So the determinism
sweep runs the **real training workload**, at the thread counts real jobs use: 1, and 5 for
retrieval-curve's layout. A single mismatch writes no capacity. **Expectation: survived.**

## Two workloads, because memory and cores are different limits

A training job holds about 7 GB. So about 8 fit in 64 GB, while the machine has 16 cores. With
training jobs, a core sweep past about 8 would measure swapping and jetsam, not cores. The core
ceiling is therefore measured on a batch-4 copy of the same computation (about 2.5 GiB, and a
per-sample cost close to training's). That is a measurement of cores, not a training result. Batch 1
was rejected because it measures Python overhead. Memory gets its own number
(`k_mem`) and its own conflict line.

## The five capacities

| Key | Rule | Why |
|---|---|---|
| `c0.cpu_det_slots` | the largest thread count at which each added thread still adds at least half a solo job's throughput | "concurrent-job ceiling" (`lanes.py`); slots are threads (`slot.py`) |
| `c0.battery_cpu_slots` | the smallest thread count within 15 % of the fastest suite time, clamped | threads beyond that are taken from `cpu-det` for almost nothing |
| `c0.mps_reserves_cpu_slots` | `ceil(p95)` cores used by an uncapped MPS job, clamped | an MPS job still runs Python and data work on CPU |
| `c0.agent_sessions` | 2, a policy value; the measurement only checks feasibility | verification is serial |
| `c0.reserve_gb` | agents + 4 GB | covers what can start after an MPS job is admitted, **except** cpu-det jobs (conflict b) |

## Conflicts the author expects, stated before the data

- **(a) Below 15.** Retrieval-curve needs 15 thread-slots, and the Studio has 12 performance and 4
  efficiency cores. The rule is not tuned to 15.
- **(b) Memory.** The `cpu-det` lane admits by threads alone, so it can admit sixteen 1-thread
  training jobs, about 112 GB. **This is a scheduler gap, not a C0 result.** It is filed as queue
  item `eng-cpu-det-memory-admission`: `cpu-det` admission should check `free − declared peak`, as
  MPS does. Until that lands, the conflict line in RESULTS.md is the record.
- **(c) MPS starvation.** Checked against C0's own measured MPS peak (`c0.mps_job_peak_gb`), and reported.

## What this does not establish

Anything about RSR or memory retrieval. It does not establish bit-exactness at thread counts other
than 1 and 5, or on filesystems or OS versions other than the ones recorded in provenance.
