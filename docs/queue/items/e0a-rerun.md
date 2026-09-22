---
id: e0a-rerun
title: E0a re-run with unit-norm gestalts
class: exp
sprint: 2
workstream: W2
lane: cpu-det
slots: 1
requires:
- sprint_gate: S0
- ruling: R-*-sprint2-start
- item_done: eng-coord-check
- ruling: R-*-adr0005-ct-pin
- prereg: experiments/e0a/PREREG-rerun.md
status: blocked
attempts: 0
source: docs/ROADMAP.md:99; experiments/e0a/run.py
---
# E0a re-run with unit-norm gestalts

ROADMAP:99: the existing E0a numbers fed `torch.randn` gestalts, the wrong premise for real TG. Kill gate (ROADMAP:102). `c_t` must be pinned by the owner before E0a runs (RESEARCH-CONTEXT:1035).
