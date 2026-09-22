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
