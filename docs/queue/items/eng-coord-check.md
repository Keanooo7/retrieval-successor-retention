---
id: eng-coord-check
title: Port mup/coord_check.py from macbook-local-2026-09-18
class: eng
sprint: 2
workstream: W2
lane: agent-only
slots: 1
requires: []
status: blocked
attempts: 0
source: src/rsr/mup/coord_check.py:42; docs/ROADMAP.md:97
---
# Port mup/coord_check.py from macbook-local-2026-09-18

ROADMAP:97-99: port `mup/coord_check.py` from branch `macbook-local-2026-09-18` (API differs: `coord_check` / `width_invariance` -> `run_coord_check`). Keep the trunk docstring's E0a reading rule (RESEARCH-CONTEXT §10.6: it is load-bearing).

**Done when:** `run_coord_check` runs on unit-norm gestalts under test.
