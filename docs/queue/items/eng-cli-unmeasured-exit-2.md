---
id: eng-cli-unmeasured-exit-2
title: rsr.cli exits 2, not 1, on an unmeasured constant
class: eng
sprint: 0
workstream: W0
lane: agent-only
slots: 1
requires: []
status: blocked
attempts: 0
source: docs/lab-notes/for-brendan-2026-09-21.md:12; src/rsr/cli.py:31
---
# rsr.cli exits 2, not 1, on an unmeasured constant

`src/rsr/cli.py` returns `1` when `REGISTRY.get` raises a `ConstantError` (`UnmeasuredConstant` included). An unmeasured constant is *nothing to compare against* -- `Exit.UNKNOWN` (2), docs/gates.md. Convert `main()` to `rsr.exit_codes.run_main` and return `Exit` members.

**Done when:** a test pins exit `2` for an unmeasured `MEASURED` constant, and a mutation back to `1` reddens only that test.
