---
id: eng-cpu-det-memory-admission
title: 'cpu-det admission checks memory (free - declared peak), as mps does'
class: eng
sprint: 0
workstream: W0
lane: agent-only
slots: 1
requires: []
status: parked
attempts: 0
source: scripts/orchestrator/lanes.py:450; experiments/capacity-c0/PREREG.md
---
# cpu-det admission must check memory

Found in the C0 pre-push review (owner proxy, 2026-09-22). One S0-03-config training job peaks at
about 7 GB RSS (`/usr/bin/time -l`: 7309426688, 6876348416 and 7032832000 bytes on three solo runs).
`cpu-det` admits by **threads only** (`scripts/orchestrator/slot.py` `job_threads`), so a full lane
of 1-thread training jobs can be admitted into roughly 112 GB on a 64 GB machine. Only `mps`
checks `free − peak ≥ reserve` (`lanes.py:450`).

The fix is to require a declared `--peak-gb` for `cpu-det` too, and to admit only while
`free − peak ≥ reserve_gb`, reusing `free_memory_bytes()`. Tests first. Add a battery entry in which
removing the check reddens only the new test. **This is a hidden failure: jetsam kills a job, or
swap silently slows every job, and nothing reports it.** Until this lands, C0's conflict line (b) is
the record.
