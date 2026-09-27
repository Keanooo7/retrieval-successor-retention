# I5: the flaky dispatch test. Capture attempt, 2026-09-27

PLAN-v4 §4 I5. The test is
`tests/test_orch_dispatch.py::test_submit_launches_the_slot_detached_at_the_pinned_sha`.

**Result: it did not fail in 30 attempts, so no traceback was captured and nothing was
fixed.** The brief says: "If it never fails in 30 attempts, report that and stop. Do not
fix a cause you have not seen."

## What ran

- Tree: `run/i5-flake` at `941a68e` (branched from `night/2026-09-27`), with its own venv
  (`uv sync --frozen --extra dev`).
- Harness: `scripts/i5_stress.sh OUT 30 1`. It starts 3 full suites in a loop as load, each
  one run as `orchestrator.slot run --lane cpu-det --slots 3`. It waits 20 s, then runs the
  test alone 30 times, each as `orchestrator.slot run --lane cpu-det --slots 1 --
  .venv/bin/pytest --tb=long`. It stops on the first failure. Each rc is captured directly.
- Logs: `~/Documents/RSR-2026-09-27-plan/i5-logs/capture/` (attempt-1..30, load-{1,2,3}-1)
  and `~/Documents/RSR-2026-09-27-plan/i5-logs/capture.out`.

## Literal output (capture.out)

```
attempt 1 rc=0
...                (attempts 2-29 identical: rc=0)
attempt 30 rc=0
attempts_failed=0
load stopped
rc=0
```

Every attempt log reads `passed=1 failed=0 skipped=0 errors=0`. Each of the three load
suites reads `passed=1571 failed=0 skipped=0 errors=0` / `rc=0`.

Timing, from file mtimes: attempts ran 15:36:39 to 15:37:08. The load suites ran until
about 15:39:30, so all 30 attempts ran while the 3 suites were running. A fourth suite
from another session (the b2-psi-probe worktree) was also running at the start.

## What this does and does not show

- **No traceback exists for the 09-26 red.** The battery runs pytest with `--tb=no`
  (`scripts/mutation_battery.py` `run_suite`), so W4-fix.md:68 records only the node id.
  Neither the torn-read cause (F9 / M-14) nor the 20 s timeout has been observed.
- The conditions differ from 09-26 in ways that could matter:
  - On 09-26 the test ran inside a battery shard, in a detached temporary worktree, with
    mutated code and battery thread caps.
  - Here the load suites were capped at 3 threads each by the cpu-det lane.
  - Here each attempt was about 1 s, which gives about 60 polls per attempt. A torn read
    needs one poll to land inside the stub's truncate-then-write window.
- The torn-read defect is still real by construction. `tests/_orch_loop_helpers.py`'s
  stub `slot` rewrites the job record with `path.write_text(...)`, which truncates and then
  writes, while `wait_job` does `json.loads(read_text())` every 50 ms. It is still
  unobserved as the cause of *this* flake.
