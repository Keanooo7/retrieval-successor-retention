# For Brendan — 2026-09-24: what ran, literal results, what is open

This note was written by the Studio lead session on 2026-09-24. `main` was at `340dd11` when it
was written; the retrieval-curve run is on PR #43, which is not merged. Every number below is
quoted from the ledger or log it names.

## What ran

### Phase A — remote access

- **SSH:** `ssh keanooo7@100.81.77.20`. The Studio's local user is `keanooo7`, not `brendankeane`.
  The MacBook's key is in `authorized_keys`.
- **Key-only:** `/etc/ssh/sshd_config.d/100-keyonly.conf` sets `PasswordAuthentication`,
  `KbdInteractiveAuthentication` and `PermitRootLogin` to `no`. `sshd -t` rc 0. The MacBook tested
  it before and after. A password-only attempt gets `Permission denied (publickey)`.
- **Already in place:** Remote Login was already on, limited to the `com.apple.access_ssh` group.
  `pmset` was already `sleep 0 disksleep 0 womp 1 autorestart 1`.
- **Persistent session:** `tmux new-session -d -s rsr -c ~/retrieval-successor-retention
  'claude --remote-control rsr-studio'`.

### Phase B — inventory

- `origin/main` was `fee5f04`, as expected. No local commit was missing from the remote, and no
  stash existed.
- Two agent worktrees have an untracked `.python-version` and `uv.lock`, byte-identical to
  `main`'s.
- **The 132/132 battery record from the 7a9c2d7 run was lost before it was committed.**
  - The run was real. Its pin records rc 0 and a regenerated file.
  - Every copy on disk read 71/71.
  - Its log and pin are preserved in `runs/battery-logs/` (gitignored).
  - It was superseded by the 136/136 record below.

### Merged today

| PR | What | Gate |
|---|---|---|
| #40 | C0 brief: `suite_threads` anchor `:1956 → :1976` | lint 4 → 3 findings; the 3 left are post-#38 scope/premise |
| #41 | `R-2026-09-24-rsr-full-priority`, your words verbatim | CI pass |
| #42 | C0 run + `ops/lanes.json` | CI pass |
| #39 | retrieval-curve implementation + Studio review fixes | pytest `passed=1143 failed=0 skipped=0 errors=0`; ruff rc 0; battery `136/136 gates proven by mutation`, rc 0 |

### C0 (#42)

It took three attempts, and the ledgers of all three are committed:

1. **Attempt 1, exit 3.** The Second Brain servers are launchd `KeepAlive` daemons and respawned.
   run.py's respawn check refused, as it was designed to.
2. **Attempt 2, exit 3.** A fresh worktree's venv had no pytest, which is in the `dev` extra.
3. **Attempt 3, rc 0**, 14:08:48 → 14:25:37. Verdict **survived**: all 18 determinism-sweep jobs
   match the solo hash.

**The five values:**
- `cpu_det_slots` 12
- `battery_cpu_slots` 1
- `agent_sessions` 2
- `reserve_gb` 5.5
- `mps_reserves_cpu_slots` 2

**Conflicts:**
- (a) **no**, 12 vs 15.
- (b) **yes**, `k_mem_training` 4 < 12.
- (c) no, 13.49 vs 64.0.

The brief errors are in `experiments/capacity-c0/BRIEF-ERRORS.md`.

### Retrieval curve (#43, open, for you)

- **The run:** rc 0, 17:30:47 → 20:33:02, at `340dd11`, 3×5 threads as preregistered. It ran
  outside the lane scheduler on your go.
- **The ckpt-300 reproduction control:** `abs_diff` **0.0** on every seed.
- **The verdict, from the unchanged PREREG rule: `falsified`** at ckpt1000 and ckpt3000.

| ckpt | held-out zeroed − live | held-out live NLL (chance 2.7726) | train live NLL | held-out acc (chance 0.0625) |
|---|---|---|---|---|
| 300 | 0.0228, 0.1739, 0.0083 | 2.8333, 2.8183, 2.9918 | 2.6563, 2.4989, 2.4969 | 0.0707, 0.1055, 0.0688 |
| 1000 | 1.1755, 1.7128, 1.3298 | 6.2774, 6.2781, 6.5387 | 0.0028, 0.0024, 0.0022 | 0.1485, 0.1646, 0.1431 |
| 3000 | 1.9049, 2.4994, 2.3815 | 8.1453, 7.8372, 8.4254 | 0.0001, 0.0000, 0.0001 | 0.1499, 0.1899, 0.1527 |

**Not a clean "memory helps".**
- Held-out live NLL rises to about 3× chance while train NLL goes to about 0. The model memorises
  and is confidently wrong on held-out documents.
- The rule's difference measures zeroing the memory making that worse.
- Bar 1 can't bind when every loss is above chance.
- Held-out accuracy does rise to about 2.4× chance.

Recorded in `experiments/retrieval-curve/BRIEF-ERRORS.md` item 4. No threshold was changed. **How
to read it is your decision.**

## What is open

1. **#43: read and merge the curve**, and decide what the result means (above).
2. **Keep the servers off across reboots**, once, on the Studio. `bootout` alone lasts only until
   the next reboot:
   `for j in main fast embed; do sudo launchctl disable system/com.vanta.$j; done`
3. **C0 conflict (b):** the `cpu-det` lane admits by threads alone, 12, but memory holds 4 training
   jobs. Queue item `eng-cpu-det-memory-admission`, per C0's PREREG.
4. **C0 brief errors:**
   - Preflight for `.venv/bin/pytest`.
   - `lanes generate` writes to the main checkout from a worktree.
   - The KeepAlive servers can't be stopped without sudo.
5. **Tailscale:** turn on Launch at login (`TailscaleStartOnLogin = 0`).
6. **FileVault is On.** After a power cut the Studio waits at the unlock screen (A6/UPS).
7. **B10:** plain `python` is 3.14.6. Every project run goes through `uv`, which uses 3.12.13.
8. **Housekeeping:** the 10 old worktrees and several local branches are fully pushed. They can be
   removed whenever you like; nothing was deleted today.
