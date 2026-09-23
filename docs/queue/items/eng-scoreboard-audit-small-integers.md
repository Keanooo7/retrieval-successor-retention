---
id: eng-scoreboard-audit-small-integers
title: 'render_scoreboard --audit: small integers are backed by accident'
class: eng
sprint: 0
workstream: W0
lane: agent-only
slots: 1
requires: []
status: done
attempts: 0
source: docs/lab-notes/overnight-2026-09-21.md:30; scripts/render_scoreboard.py:289 at 7871580 (rewritten by 46fcb35)
---
# render_scoreboard --audit: small integers are backed by accident

`audit_prose` backs an integer literal when *any* ledger holds that value (`:289`). Small integers occur in some ledger almost always, so a typed count passes the audit whatever it is -- the overnight manager and the E-feas-s003 researcher both noted it (`overnight-2026-09-21.md:30`). Design a check that can fail on a fabricated small count (e.g. require a ledger key reference for integers below a bound) without drowning in section numbers.

**Done when:** a test with a fabricated small count reddens the audit, and a battery entry proves it.
