---
id: eng-port-gates-floors
title: Port src/rsr/gates/ from macbook-local-2026-09-18
class: eng
sprint: 0
workstream: W0
lane: agent-only
slots: 1
requires:
- path_exists_at_base: src/rsr/exit_codes.py
status: blocked
attempts: 0
source: docs/lab-notes/for-brendan-2026-09-21.md:16; origin/macbook-local-2026-09-18:src/rsr/gates/floors.py
---
# Port src/rsr/gates/ from macbook-local-2026-09-18

`src/rsr/gates/` (`__init__.py`, `floors.py`) is the only producer of exit `4` (`UNBANKED_RISE`) and is still off trunk (`src/rsr/exit_codes.py` docstring). Port it, importing `rsr.exit_codes.Exit` rather than defining a second enum.

**Done when:** `floors.py` is on the branch with tests, and exit `4` has a producer under test.
