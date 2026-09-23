---
id: R-2026-09-22-mlx-servers
date: 2026-09-22
stated_in: 'MacBook owner-proxy session 2026-09-22 — Brendan, verbatim: "This model training takes full priority over the mac studio" -- recorded in the Studio interactive session 2026-09-22 under R-2026-09-22-proxy-go'
---
**RSR training has full priority on the Mac Studio.** The owner's Local Second Brain servers
(`mlx_lm.server` / `llama-server`, ~25 GB) **yield to it**.

- **Stop them** before C0 runs, and whenever an RSR night opens. A run or an orchestrator step may
  stop them itself; this ruling authorises it. The stop must be recorded: the PIDs, the command
  lines and the RSS freed go in the run's ledger or the night log.
- **Restart them** only when no RSR work holds the machine: after C0 completes, and at night
  close. They are a daytime tool.
- **Measure capacity on the machine RSR actually gets.** C0 runs with the servers stopped, and
  `reserve_gb` does not budget for them.
- ⚠️ This does not license killing arbitrary processes. It covers the owner's LLM servers only,
  identified by command line. Anything else over 2 GB at C0 start still makes C0 refuse with
  exit 3.
