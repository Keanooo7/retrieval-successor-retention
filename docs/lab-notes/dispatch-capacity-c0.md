---
id: capacity-c0
item: capacity-c0
baseline_sha: 6027bb4cf1edc8d5f793bced778acb66d792befe
falsifier: >-
  A deterministic CPU training job on the Mac Studio produces bit-identical output no matter
  how many other such jobs run beside it, at a fixed per-job thread count (1 and 5).

anchors:
  - {path: scripts/orchestrator/lanes.py, line: 3, expect: "M4 Max, 16 cores, 64 GB unified"}
  - {path: scripts/orchestrator/lanes.py, line: 118, expect: "C0_LEDGER_KEYS: dict[str, str] = {"}
  - {path: scripts/orchestrator/lanes.py, line: 222, expect: 'return [("mps", k), ("cpu-det", self.mps_reserves_cpu_slots)]'}
  - {path: scripts/orchestrator/lanes.py, line: 224, expect: 'return [("battery", k), ("cpu-det", self.battery_cpu_slots)]'}
  - {path: scripts/orchestrator/lanes.py, line: 284, expect: "def generate(ledger_path: Path) -> dict:"}
  - {path: scripts/orchestrator/lanes.py, line: 336, expect: "def free_memory_bytes("}
  - {path: scripts/orchestrator/lanes.py, line: 349, expect: "def thread_env(n: int) -> dict[str, str]:"}
  - {path: scripts/orchestrator/lanes.py, line: 450, expect: "if free_gb - peak_gb >= cfg.reserve_gb:"}
  - {path: scripts/mutation_battery.py, line: 1976, expect: "def suite_threads() -> tuple[int | None, str]:"}
  - {path: experiments/s0-03-rewardable-corpus/run.py, line: 82, expect: "CONFIG = {"}
  - {path: experiments/s0-03-rewardable-corpus/run.py, line: 135, expect: "def single(seed: int, iters: int, device: str, out_dir: Path) -> dict:"}

premises:
  - claim: C0 has not run, so no lanes file exists
    check: "test -e ops/lanes.json"
    expect_rc: 1
  - claim: no C0 implementation or ledger exists
    check: "test -e experiments/capacity-c0/run.py -o -e runs/capacity-c0"
    expect_rc: 1
  - claim: C0_LEDGER_KEYS is exactly the five keys this PREREG writes
    check: >-
      python3 -c "import re; s = open('scripts/orchestrator/lanes.py').read();
      b = s[s.index('C0_LEDGER_KEYS: dict'):]; b = b[:b.index('}')];
      print(sorted(re.findall(r'\"(c0[.][a-z_]+)\"', b)))"
    expect_rc: 0
    expect_stdout_contains: "['c0.agent_sessions', 'c0.battery_cpu_slots', 'c0.cpu_det_slots', 'c0.mps_reserves_cpu_slots', 'c0.reserve_gb']"
  - claim: lanes generate refuses a ledger missing any key (so a falsified C0 that writes no keys yields no lanes file)
    check: "git grep -n 'no row for ledger key' -- scripts/orchestrator/lanes.py"
    expect_rc: 0

files_in_scope:
  - path: experiments/capacity-c0/run.py
    new: true
  - path: experiments/capacity-c0/RESULTS.md
    new: true
  - path: tests/test_capacity_c0.py
    new: true
  - scripts/mutation_battery.py

bar:
  - "Quiet machine: exit 3 'machine not quiet' when a foreign process has RSS > 2 GB, unless a signed R-*-mlx-servers ruling says the servers STAY UP. The report then quotes that ruling's text; a ruling saying 'stop them' does not exempt a machine where they run. A test drives both paths with a stubbed process list."
  - "Output hash exactly as the PREREG defines it (sorted state_dict name/dtype/shape/bytes + repr(final loss); the checkpoint file is never hashed). A test shows two identical solo runs hash equal while their checkpoint files differ, using iters=1 so the suite (and the battery) does not grow by minutes."
  - "Every job writes to its own out_dir, runs/capacity-c0/sweep/<part>/t<threads>/k<k>/rep<r>/job<j>. A test fails if two jobs share one."
  - "Determinism sweep on the training workload, capped at k_mem_training: 1-thread k in {1,2,4,8}, 2 reps; 5-thread solo then 3 concurrent. The ledger records the match count per level. One mismatch: verdict falsified, no c0.* capacity key written, exit 1."
  - "On survived, the ledger holds exactly the five C0_LEDGER_KEYS rows, each computed in code by its PREREG rule (never typed), plus k_mem_*, the *_raw values and every secondary. `PYTHONPATH=scripts .venv/bin/python -m orchestrator.lanes generate --ledger runs/capacity-c0/ledger.json` exits 0 and `lanes status` prints the five values; quote both literally."
  - "A stop before completion (the 4 h deadline, an exception, a killed job) writes ledger status 'partial'. A test shows `lanes generate` refuses that ledger."
  - "uv run pytest -rs --tb=no census line and $? quoted; ruff check and ruff format --check exit 0; mutation battery N/N line quoted."
  - "Four mutations, each reddening only its declared test: (1) the determinism comparison skipped (every job reported as matching); (2) cpu_det_slots taken from the faster repetition; (3) the checkpoint file hashed instead of the state_dict; (4) two jobs given the same out_dir."
done_when:
  - "RESULTS.md: the verdict; the five values with their ledger keys; k_mem_training and k_mem_core; and the three PREREG conflict lines (a) cpu_det_slots >= 15?, (b) k_mem_training < cpu_det_slots?, (c) reserve_gb + c0.mps_job_peak_gb > physical?, each stated yes/no with its numbers."
  - "provenance: <git sha> · Mac Studio (sysctl model + perflevel core counts) · workload corpus sha256 · seed 0, stamped from the tree."
  - "Every number in RESULTS.md is a ledger key; render_scoreboard.py --audit exits 0."
  - "BRIEF ERRORS written out, 'none' if none."
do_not:
  - "Type any capacity number, or edit ops/lanes.json by hand: lanes generate writes it."
  - "Change any threshold, the sweep, or the workload. They are in the PREREG."
  - "Stop any foreign process other than the owner's LLM servers. Under R-2026-09-22-mlx-servers (proposed; recorded by the Studio before this brief runs), stop mlx_lm.server / llama-server identified by command line, record PIDs, command lines and RSS freed in the ledger, run C0, then restart them when C0 completes. Anything else over 2 GB: refuse with exit 3."
  - "Run unattended beside anything else: no battery, no other brief. C0 measures a quiet machine."
  - "Lower retrieval-curve's thread count, or re-tune any rule, to make a conflict disappear. Report conflicts."
  - "Run any sweep level above its workload's k_mem, or run the core sweep with the training workload."
---

# Brief: C0, measure this machine's lanes

**Status:** written, not started. **Written at:** `6027bb4cf1edc8d5f793bced778acb66d792befe`
(the PREREG commit comes first and becomes the base). **Lane:** `agent-only`, run **attended**, on
a quiet machine (`for-brendan-2026-09-22.md` A3: "install after C0 (run attended)").
**Governing PREREG:** `experiments/capacity-c0/PREREG.md`.

## Why

Every lane refuses until `ops/lanes.json` exists, and only `lanes generate`
(`scripts/orchestrator/lanes.py:284`) writes it, from this run's ledger under the five
`C0_LEDGER_KEYS` (`:118`). `liveness-wiring` and `retrieval-curve` both wait on `capacity-c0`, and
so therefore does the Sprint 0 gate. This is the one item on the critical path with no brief.

## Falsifier

In the front matter. The decision rule and every capacity rule are in the PREREG, verbatim. The
determinism check is the one gate here that fails where nothing else does. Every other number is
a timing and will always produce *some* value.

## Files in scope

- `experiments/capacity-c0/run.py` (**new**). Two workloads, both S0-03's `CONFIG` (`:82`) through
  the `train()` call `single()` makes (`:135`), seed 0, 10 iterations, CPU:
  - the **training workload**, unchanged, for the determinism sweep and the memory cap;
  - the **core workload**, the same with batch 4 and 30 iterations, for the core sweep. A
    training job holds about 7 GB, so a 16-job sweep of it would measure swapping rather than
    cores. Batch 1 would measure Python overhead instead of training scaling.

  Job time comes from each job's own heartbeat, first beat to last beat. Peak RSS is read exactly
  from `os.wait4`'s `ru_maxrss`, never from `ps` samples.

  The child's thread env comes from `lanes.thread_env(n)` (`lanes.py:349`). The solo training job
  runs first; its RSS sets `k_mem`. CPU and RSS are sampled via `ps` at 1 Hz. The suite timing sets threads the
  way `suite_threads()` does (`scripts/mutation_battery.py:1976`). The MPS job is the same workload
  on `mps`. Free memory comes from `lanes.free_memory_bytes()` (`:336`), the same measure MPS
  admission uses (`:450`). Exit through `rsr.exit_codes`.
- `experiments/capacity-c0/RESULTS.md` (**new**), from the ledger.
- `tests/test_capacity_c0.py` (**new**): the quiet-machine refusal, the rule functions (each fed a
  synthetic sweep table), and the no-keys-on-falsified path.
- `scripts/mutation_battery.py`: three entries.

## Bar

In the front matter.

## Budget

**Prediction, not premise:** about 4.6–4.9 s per iteration, from the decisive run, so a 10-iteration
training job takes about 1 min. The core-workload jobs are faster. Determinism sweep: 15 jobs × 2
reps + 4 five-thread jobs. Core sweep: 82 jobs × 2 reps in 18 waves, about 1 min each at batch 4 × 30 iterations. Suite timings: about 4 × 80 s.
Expect under 1.5 h. At the 4 h deadline, write status `partial` and exit 3; `generate` refuses a
partial ledger.

## Done when

In the front matter.

## Do NOT

In the front matter. Also (`CLAUDE.md`): read `$?` directly, never after a pipe; stage by explicit
path.
