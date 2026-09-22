---
id: cpu-canary-baseline
title: Re-baseline the canary on CPU at trunk
class: exp
sprint: 0
workstream: W0
lane: cpu-det
slots: 1
requires:
- ruling: R-*-canary-rebaseline
- prereg: experiments/cpu-canary-baseline/PREREG.md
status: blocked
attempts: 0
source: docs/lab-notes/for-brendan-2026-09-21.md:7
---
# Re-baseline the canary on CPU at trunk

`baseline.json` reads MOVED on every trunk reading because code between `7254080` and `945b501` changed the canary's loss path; the machine did not move. Re-baselining is the owner's call. **No agent writes `runs/canary/baseline*`**: the job writes a candidate elsewhere and the owner promotes it.
