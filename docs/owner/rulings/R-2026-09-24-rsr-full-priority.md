---
id: R-2026-09-24-rsr-full-priority
date: 2026-09-24
stated_in: 'Studio interactive session 2026-09-24 — Brendan, verbatim: "I do not need the seccond brain this RSR project will take full priority and control 100%. ensure that nothing is blocking the RSR progress"'
---
**The Mac Studio is 100 % RSR's.** The owner's Local Second Brain servers are not needed.

This **amends `R-2026-09-22-mlx-servers`**, whose restart clause ("Restart them only when no RSR
work holds the machine: after C0 completes, and at night close") no longer applies:

- **Do not restart them** — not after C0, not at night close, not after any run. No agent and no
  orchestrator step brings them back. Only Brendan, in person, reverses this.
- They are launchd daemons (`/Library/LaunchDaemons/com.vanta.{main,fast,embed}.plist`,
  `KeepAlive` and `RunAtLoad` true), so a kill does not stop them. They were unloaded on
  2026-09-24 with `sudo launchctl bootout system/com.vanta.<job>`; `launchctl disable` keeps them
  off across a reboot.
- `R-2026-09-22-mlx-servers` stays in force for everything else — C0 still parses it as `yield`,
  and a run that finds the servers up still stops them and records it.

⚠️ Not widened: this covers the owner's LLM servers only. It does not license stopping any other
process, and it relaxes no project prohibition (`CLAUDE.md`, RESEARCH-CONTEXT §13).
