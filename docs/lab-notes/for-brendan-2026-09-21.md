# For Brendan — 2026-09-21 overnight escalations

One line each. The manager did not act on any of these; work that does not depend on them continued.

- **E0i is signed but its own text says unsigned.** `fcd79d4` (2026-09-17, author `Brendan Keane <bkbrohon795@gmail.com>` — not the ucsd address; please confirm it is yours) fills `preregistration/e0i_threshold.md:300-302`, while `:3` and `:291` still say unsigned, as do `docs/ROADMAP.md:87` and `docs/lab-notes/dispatch-S1-01-e0i.md:6`. Not edited (`preregistration/` is not the manager's to touch). S1-01 stays out tonight regardless: T3 needs 100 human-annotated reintroductions.
- **E-feas (#19) ran out of sprint order.** It is Sprint 3 and ran before the Sprint 0 gate (`docs/ROADMAP.md:56`, "No sprint begins before its predecessor's gate answers"). It is model-free, so the result stands, but it measured the pre-S0-03 corpus; it is re-run tonight as a new `run_id` after S0-03.
- **The canary baseline is stale against trunk.** `scripts/canary.py 16` at `945b501` exits `1` (MOVED, 6 of 6 beats; ledger moved to `docs/lab-notes/canary-2026-09-21/cycle-16-trunk-945b501/` because `tests/test_evidence_machinery.py::test_the_scoreboard_reproduces_the_true_canary_tally` pins the live `runs/canary/` count at 3 and reddens on any new canary ledger — a test defect routed to Brief 0). The same script at the last good sha `7254080`, same machine, same torch `2.14.0`, exits `0` and reproduces `runs/canary/baseline.json` exactly. So the machine did not move; code between `7254080` and `945b501` changed the canary's loss path, and `baseline.json` (a `runs/` file, not editable tonight) now reads MOVED on every trunk reading. Tonight's canary = old-sha env reading (must hold) + trunk readings compared to `cycle-16`'s losses (must match). Re-baselining is your call.
- **Python is not pinned, so tonight's runs used two interpreters.** There is no `.python-version`, and `requires-python = ">=3.11"`. `uv` built every `.worktrees/*` venv on Homebrew **3.14.6**, while the main checkout's `.venv` is 3.12 and `CLAUDE.md` says 3.12. The S0-03, E-feas-s003 and decisive ledgers record `python: 3.14.6`. E-feas reproduced bit-for-bit across the two, and the old-sha canary held bit-exact under 3.14. Pin it, or not: your call.
- **`uv.lock` is untracked, so every ledger records `dirty: true`.** `Ledger.write()` still refuses dirty `src/ scripts/ experiments/`, so the gate that matters holds. But the flag is now always true and carries no information. Commit the lockfile or gitignore it: your call.
- **The canary exit protocol changed tonight.** S0-05 (#32) makes a first reading exit `2` (nothing to compare), no longer `3`. `docs/lab-notes/dispatch-2026-09-21-overnight.md:413` ("3 first reading") is now stale. It is a dispatch; I have not edited it.
- **Follow-ups found tonight, not started (each needs a brief):**
  - `src/rsr/cli.py` exits `1` on `UnmeasuredConstant` (should be `2`).
  - Not converted to the exit-code protocol: `experiments/s0-03-rewardable-corpus/run.py` and `experiments/s0-02/measure_qtok_collapse.py`, whose precondition refusals exit `1`.
  - `experiments/s0-02/write_ledger.py` has typed `exit_code=0` rows.
  - `tests/test_evidence_machinery.py::test_a_later_canary_reading_does_not_redden_the_09_18_tally` hard-codes `cycle-99`, and it errors if a real `runs/canary/cycle-99/` ever exists.
  - `src/rsr/gates/` (the only emitter of exit `4`) is still off trunk.

## Resolved 2026-09-22

Appended, not edited. The items above stay as written on 2026-09-21.

- **Python is not pinned. Resolved.** `5563b62` (`build: pin Python 3.12 (.python-version,
  requires-python ==3.12)`), owner ruling R2 of 2026-09-22 per that commit's message. The three
  3.14.6 ledgers (S0-03, E-feas-s003, decisive) are **not re-run**. Each RESULTS now carries a
  dated provenance note saying so (`experiments/s0-03-rewardable-corpus/RESULTS.md`,
  `experiments/efeas/RESULTS-s003.md`, `experiments/decisive-shuffle/RESULTS.md`).
- **`uv.lock` is untracked. Resolved.** `9a763df` (`build: commit uv.lock (resolved on 3.12)`),
  owner ruling R1 of 2026-09-22 per that commit's message. It affects ledgers written from now on.
  Ledgers already written still record `dirty: true`.
- **Also resolved, not listed above: local `main` reset.** Local `main` now equals `origin/main`
  (`a1304c9` when this note was written). The original local-main SHAs are kept in the annotated
  local tag `archive/local-main-2026-09-18` (→ `39302cb`). The tag is local, so check it exists
  before relying on it from another clone.

Not resolved by the above, and still open: the E0i signed/unsigned mismatch, the stale canary
baseline, the dispatch line on the canary exit protocol, and the follow-up list.
