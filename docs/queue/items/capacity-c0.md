---
id: capacity-c0
title: 'C0: measure this machine''s lanes (defines ops/lanes.json)'
class: exp
sprint: 0
workstream: W0
lane: agent-only
slots: 1
requires:
- prereg: experiments/capacity-c0/PREREG.md
status: blocked
attempts: 0
source: CLAUDE.md:47; docs/decisions/ADR-0007-all-training-on-the-mac-studio.md
---
# C0: measure this machine's lanes (defines ops/lanes.json)

Capacity is answered by measuring this machine (CLAUDE.md:47, ADR-0007): the concurrent-job ceiling per lane, from which `ops/lanes.json` is generated. Lane `agent-only` because the lanes do not exist until this runs (the `cpu-det` lane refuses with exit 3 while `ops/lanes.json` is absent). No training.

**Acceptance check on the result (owner proxy, 2026-09-22):** once `ops/lanes.json` is
generated, report whether `cpu_det_slots >= 15`. `retrieval-curve` needs 3 seeds × 5 threads
(`docs/queue/items/retrieval-curve.md`; the thread count is fixed by
`experiments/retrieval-curve/PREREG.md` so its ckpt-300 control can reproduce S0-03). **If
C0 allows fewer, report it — do not lower the thread count**; changing threads changes CPU
float reductions and would fail the reproduction control for a reason unrelated to the question.
