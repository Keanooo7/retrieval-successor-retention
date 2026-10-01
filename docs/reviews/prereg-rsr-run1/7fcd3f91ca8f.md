---
branch: prereg/rsr-run1
reviewed_head: 7fcd3f91ca8f57de1ace22a5460be63048383aa9
author: Claude Opus 5.5 PREREG-drafting session (PM-dispatched), Amendment 2, 2026-10-01
reviewer: independent adversarial reviewer (general-purpose agent), 2026-10-01, third pass (narrow)
verdict: MERGE
---
# Review: PREREG Amendment 2 @ `7fcd3f9`

This review answers `6a4c5a8c6785.md` (commit `c02cae7`).

Written by a model (Claude Opus 5.5, the same independent reviewer as the two earlier records).
**Not written by Brendan.** Nothing here is a ruling.

- `PREREG:N` is a line of `experiments/rsr-run1/PREREG.md` at `7fcd3f9`. Amendment 2 is lines
  1374–1741.
- `git diff --numstat 6a4c5a8 7fcd3f9 -- experiments/rsr-run1/PREREG.md` gives `370 0`: nothing
  above the amendment changed.

## Verdict

**MERGE.**

Amendment 2 resolves both MAJOR findings and all four MINOR findings of the second review.
Every claim I was asked to check, I re-ran with my own scripts, and each held:

- the hang-up result;
- launcher v2's six-case refusal matrix;
- the launcher v2 dry run inside a detached tmux session;
- the oracle's γ_o invariance, including on C9's own documents.

The owner slots O1, O4, R-1 and O5 are single blanks.

**Five new MINOR findings** (below) concern launcher robustness and C9's power. None blocks merging
the PREREG. Fixing them changes the launcher text that A2.7 pins byte-identical, so they need a
dated amendment **before launch**, at the latest at the T3 review of `run.py` and the launcher.
The launch gates stand unchanged (`PREREG:1397-1403`):

- the four owner slots are filled;
- C-1 (if O5 = eval) and C-2 are on a reviewed head;
- `run.py` and the launcher are reviewed;
- the launch is from a clean pinned head, inside tmux.

---

## (a) The second review's findings

| finding | resolved? | evidence |
|---|---|---|
| **MAJOR-A1-1**: the correction-20 reading was not a precondition | **Yes** | It is now owner slot **O5** (`PREREG:1405-1417`, `:1707`). Precondition 1 requires all four slots (`PREREG:1399`). The launcher refuses an unfilled or invalid O5 with rc 3 (re-run by me, §c). The evidence both ways is copied from review §c without a choice (`PREREG:1419-1444`). The `train` branch is fully specified: C-1 is dropped, and **TRAIN/EVAL MEMORY SHIFT (psiC only)** is printed beside every psiC contrast, the headline included (`PREREG:1446-1465`) |
| **MAJOR-A1-2**: the oracle was coupled to O1 | **Yes** | γ_o = 0.9 is a recorded construction parameter, decoupled from O1 (`PREREG:1467-1475`, `:1704`). The author's model-free T7 output (`PREREG:1484-1506`) reproduced **exactly** with my own loop (§d). New control C9 (`PREREG:1516-1525`). The PM made this decision, and it is not owner-reserved (§e) |
| **MINOR-A1-1**: scope sentence | **Yes** | The sentence now says the three per-seed values show the **combined** variation, and that run-to-run cannot be separated from substrate-seed variation (`PREREG:1612-1616`) |
| **MINOR-A1-2**: launch shell's hang-up | **Yes, measured** (§b) | The tmux procedure (`PREREG:1558-1570`). See MINOR-A2-3 for an environment hazard the procedure introduces |
| **MINOR-A1-3**: unfilled slot exited 1 | **Yes** | Launcher v2 gives rc 3 in all six cases (§c). There is now a launcher-level reading rule, and E19 is superseded (`PREREG:1599-1606`) |
| **MINOR-A1-4**: C5 | **Yes** | The band stays a control in every arm and seed: `live` is required, `inert` means exit 5, quarantine and no retention number, and `inconclusive` means the arm is not read. Only the ratio is non-control (`PREREG:1527-1537`) |

## (b) The hang-up finding: re-run with my own driver

**slot.py's handler**, read at `scripts/orchestrator/slot.py`:

- `_FORWARDED = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)` (`:64`).
- `previous = {s: signal.signal(s, forward) for s in _FORWARDED}` (`:180`). This installs a
  Python handler **unconditionally**, so an inherited `SIG_IGN` (what `nohup` sets) is replaced.
- `forward` sends the signal on to the child (`:175-178`).
- The child is started by `subprocess.Popen` (`:183`). It gets the default disposition, because
  handlers do not survive `exec` and the ignore was replaced.
- The record is then `status = "crashed"`, with `rc = 128 - raw` when the child dies by signal
  (`:215-221`).

The author's explanation (`PREREG:1554-1556`) is correct.

**Probe.** `a2/hup_driver.py` and `a2/hup_launcher.zsh` are mine, not the author's
`hup_probe.py`.

- The mini-launcher reuses launcher v2's backgrounding pattern (`p=$!; pid[$k]=$p`, `wait`) with
  two slot-wrapped `/bin/sleep 20` jobs.
- They run in an isolated root `a2/orch2` (a copy of `ops/lanes.json`) with `--wait 60` and
  `--pinned-sha HEAD`.
- Each launch form is started in a **new session and process group**, as a terminal's foreground
  job would be. `os.killpg(pgid, SIGHUP)` is sent 4 s later. The job records are read about 22 s
  after that.
- For tmux, the "terminal" is a zsh that runs `tmux -L rvhup new-session -d …` and then stays open
  (`sleep 40`) until it is hung up.

The run exited with rc 0. Literal output:

```
plain: launching group pgid 91823 hung up; launcher/driver rc 1; jobs {'j1': ('crashed', 129, 1), 'j2': ('crashed', 129, 1)}
nohup: launching group pgid 92249 hung up; launcher/driver rc 0; jobs {'j1': ('crashed', 129, 1), 'j2': ('crashed', 129, 1)}
tmux: launching group pgid 92965 hung up; launcher/driver rc -1; jobs {'j1': ('done', 0, None), 'j2': ('done', 0, None)}
tmux inner launcher rc file: 0
tmux inner launcher log: rc[j1]=0
rc[j2]=0
tmux -L rvhup ls after: 1 no server running on /private/tmp/tmux-501/rvhup
```

H1, H2 and H3 (`PREREG:1548-1552`) are reproduced exactly:

- **Plain zsh and `nohup` both lose the jobs** (`crashed`, rc 129, signal 1).
- **The detached tmux session keeps them** (`done`, rc 0). The inner launcher's own rc is 0.
- The dedicated-socket server exits once its session ends, so nothing is left running.

## (c) Launcher v2

**Text.** I extracted `PREREG:1631-1698`. `diff` against the author's
`scratchpad/run1-prereg/launch_run1_v2.zsh` gives rc 0 (66 lines, byte-identical).

**Refusal matrix.** It is run with `env -i HOME PATH` plus only the listed variables, and `OUT`
set to a scratch path per case. "outdir" is whether `OUT` exists afterwards.

```
noslots rc=3 outdir=absent :: DID NOT RUN: slot GAMMA unfilled
o5maybe rc=3 outdir=absent :: DID NOT RUN: O5 must be eval or train, got 'maybe'
o5eval_noC1 rc=3 outdir=absent :: DID NOT RUN: O5=eval needs C1_FLAGS (C-1's flag)
o5train_C1 rc=3 outdir=absent :: DID NOT RUN: O5=train: C1_FLAGS must be empty
wd0_noC3 rc=3 outdir=absent :: DID NOT RUN: O4=0.0 needs C3_FLAGS (code change C-3)
dirty rc=3 outdir=absent :: DID NOT RUN: not a clean committed head (rc_sha=0 rc_status=0)
wd001_C3 rc=3 outdir=absent :: DID NOT RUN: O4=0.01 is the built value: C3_FLAGS must be empty
```

All six of A2.5's cases are confirmed (rc 3, no output directory). So is a seventh: WD = 0.01 with
`C3_FLAGS` set. The dirty case ran in a scratch git repo holding an untracked file.

**Consistency checks O5 ↔ C1_FLAGS and O4 ↔ C3_FLAGS.** They work as written. They check
**presence only, not content.** The probes below ran in the dirty scratch repo, so nothing could
launch. Reaching the dirty-tree refusal means every slot check passed:

```
gap_eval_C1train rc=3 :: DID NOT RUN: not a clean committed head (rc_sha=0 rc_status=0)    # O5=eval, C1_FLAGS='--frozen-model-mode train'
gap_wd0_C3wrong  rc=3 :: DID NOT RUN: not a clean committed head (rc_sha=0 rc_status=0)    # WD=0.0, C3_FLAGS='--phi-weight-decay 0.05'
gap_wd_spelling  rc=3 :: DID NOT RUN: O4=0.010 needs C3_FLAGS (code change C-3)            # string, not numeric, compare
```

See MINOR-A2-1.

**Dry run inside tmux, at `7fcd3f9`.** The command was:

`tmux -L rvdry2 new-session -d -s rvdry2 "cd <worktree> && RSR_ORCH_ROOT=<my isolated root, 1 cpu-det slot> OUT=<scratch> JOB_SUFFIX=-rvdry2 GAMMA=0.9 WD=0.01 STEPS=0 CKPT=1000 O5=train /bin/zsh launch_v2 > log; echo \$? > rc"`

`GAMMA=0.9` is a placeholder, not a registered value. `O5=train` was used because C-1 is absent at
this base. The nine jobs ran one at a time under `--wait 3600`.

```
rc_tmux=0
launcher rc: 0
rc[age-s0]=0 … rc[psiC-s2]=0        (all nine 0)
{"run_sha": "7fcd3f91ca8f57de1ace22a5460be63048383aa9", "dirty": false, "porcelain_lines": 0, "gamma": "0.9", "wd": "0.01", "steps": "0", "ckpt": "1000", "o5": "train", "c1_flags": "", "c2_flags": "", "c3_flags": "", "launched_utc": "2026-10-01T10:48:19Z"}
rsr-run1-<arm>-s<seed>-rvdry2 done 0 threads 1 pinned 7fcd3f91ca8f     (x9)
no server running on /private/tmp/tmux-501/rvdry2
```

**The author's E22 artefacts agree.** `scratchpad/run1-prereg/launch_dry_v2/launch-git.json`
matches `PREREG:1582` exactly, `launch_dry_v2.rc` reads `0`, and all nine `*-dry2` job records read
`done 0 1 c02cae749d79`.

**The worktree stayed clean throughout.** `runs/**` is git-ignored (`.gitignore:38`), so the
tmux procedure's `runs/rsr-run1-launch.log` and `.rc` cannot dirty the tree before the porcelain
check. I checked this with `git check-ignore -v`.

## (d) Control C9

**Is it well-defined?** Mostly. It names:

- the documents: the first 16 EVAL documents per seed, `[995000, 995016)`;
- the γ values: 0.5 against 0.9;
- the loop: the same model-read evaluation loop;
- the assertion: identical victim sequences, document by document;
- the consequence of failure.

`OraclePolicy` holds **one** document's demand matrix (`oracle.py`). Running it on "one batch" of 16
therefore needs either a fresh policy per document or a row-correct batched wrapper. This is
TO-CONFIRM T13, the same issue as the first review's oracle note.

**Is it runnable?** Yes, cheaply: one batch of 16, twice, per seed. It is not yet implemented
(`run.py` does not exist).

**Is it a real test at 0.5 vs 0.9?** It is a real test of the *implementation*, and a weak test of
anything else. Two points.

**First, its outcome is fixed by construction for a correct implementation.** In S0-03 every fact
is queried exactly once and query positions are distinct. So `discounted_demand[t+1][i]` is
γ^(q−t−1) for a pending assert and 0 otherwise. The argmin is then γ-invariant on (0, 1), and the
only ties (zero-demand slots) are broken by index in both. My own loop calls
`OraclePolicy.select_eviction` directly (`a2/oracle_review.py`, not the author's `oracle_gamma.py`)
and exited with rc 0. Excerpt:

```
seed 0 EVAL[995000,995016): gamma_o 0.5 vs 0.9: decisions 512, differing 0, documents differing 0
seed 0 EVAL[995000,995016): gamma_o 0.97 vs 0.9: decisions 512, differing 0, documents differing 0
seed 0 EVAL[995000,995016): gamma_o 0.0 vs 0.9: decisions 512, differing 350, documents differing 15
seed 0 EVAL[995000,995016): gamma_o 0.9 hit 290/290 = 1.0000
seed 0 EVAL[995000,995016): gamma_o 0.0 hit 235/290 = 0.8103
seed 0 POOL: gamma_o 0.0 vs 0.9: decisions 8192, differing 5562, documents differing 250
seed 0 POOL: gamma_o 0.0 hit 3884/4717 = 0.8234
seed 1 EVAL[995000,995016): gamma_o 0.5 vs 0.9: … differing 0 …;  0.0 vs 0.9: differing 303, documents 16;  hit 0.9 294/294, 0.0 252/294 = 0.8571
seed 2 EVAL[995000,995016): gamma_o 0.5 vs 0.9: … differing 0 …;  0.0 vs 0.9: differing 340, documents 16;  hit 0.9 299/299, 0.0 254/299 = 0.8495
seed 1 POOL: 0.0 vs 0.9 differing 5656 (249 docs), hit 0.0 3911/4775 = 0.8191;  seed 2 POOL: 5539 (247), 3956/4793 = 0.8254
(0.5 vs 0.9 and 0.97 vs 0.9: 0 differing on POOL for every seed)
```

My pool numbers equal A2.2's line for line (`PREREG:1485-1505`) and the coordinator's seed-0 re-run.
On C9's own 16 documents, 0.5 against 0.9 differs on 0 of 512 decisions for every seed.

**Second, C9 cannot see the failure most likely in a batched oracle wrapper.** That failure is
using the wrong document's demand for a row: the "row-0 bug" class ADR-0009 cites. Such a bug
gives identical (wrong) victims at both γ values, so C9 passes. C7 does not catch it either: C7
replays the loop's own victim sequence model-free, so it checks residency, not the demand
matrix. See MINOR-A2-2.

## (e) The owner boundary

The owner slots are single blanks: `PREREG:1704-1707` give O1, O4, R-1 and O5 as `______` each.

- The O1 row now reads "both RSR arms only".
- The evidence for O5 copies the review's §c on both sides, without a recommendation beyond
  labelling `eval` "this PREREG's proposed reading". That label is a proposal, not a fill.
- The launcher refuses each blank (§c).
- T9 lists observed `C1_FLAGS` spellings as "read only, not chosen", and `C1_FLAGS` itself stays
  TO-CONFIRM (`PREREG:1718-1726`).

**γ_o = 0.9 is not an owner value.** It is a construction parameter of a reference whose
decisions are γ-invariant on (0, 1) (measured twice, §d). It is not the registry's ψ̂ γ: it is
never read from or written to the registry (`PREREG:1472-1473`). Precedent: lookahead-room used
`ORACLE_GAMMA = 0.97` on the same grounds.

**Nothing else in Amendment 2 fills an owner-reserved item.** The `train` branch's named confound
is a pre-registered consequence of an owner choice, not a choice.

---

## New findings

### BLOCKER

None.

### MAJOR

None.

### MINOR

**MINOR-A2-1. The launcher's consistency checks are presence-only.**
`PREREG:1644-1647` check that `C1_FLAGS` and `C3_FLAGS` are set or empty, not what they say.

- `O5=eval` with `C1_FLAGS='--frozen-model-mode train'` passes.
- `WD=0.0` with a C-3 flag carrying another value passes.
- `WD` is compared as a string, so `0.010` is treated as a departure from 0.01. That one is
  conservative: it refuses.

The run's own `frozen` config records the actual mode and decay, so a mismatch is detectable after
the fact. *Fix:* the preflight or `run.py` asserts, per job, that `frozen["frozen_model_mode"]`
matches O5, the optimizer record's `weight_decay` matches WD, the label's `gamma=` matches GAMMA,
and `ckpt_every` matches the derived CKPT (STEPS/3 or STEPS, A1.8 m-9). Any mismatch is a control
failure.

**MINOR-A2-2. C9 cannot detect a wrong-document demand in a batched oracle.** See §d.

*Fix (cheap):* add C9b. For the same 16 documents, assert the model-read loop's oracle victim
sequence equals a model-free oracle replay (`headroom.simulate`, or a per-document loop like
`a2/oracle_review.py`) built from each document's **own** `discounted_demand`. Also consider
scoping C9's failure to the oracle-derived numbers (regret, headroom share) rather than "the run is
inconclusive" (`PREREG:1521`). The headline does not depend on the oracle.

**MINOR-A2-3. The tmux procedure passes on leftover environment variables, and the launcher
neither refuses nor records them.** A tmux server copies the launching client's environment into
the pane. Measured:

```
$ env OUT=/some/dry/out JOB_SUFFIX=-leftover RSR_ORCH_ROOT=/some/isolated/root tmux -L rvenv new-session -d -s rvenv "echo OUT=\$OUT JOB_SUFFIX=\$JOB_SUFFIX RSR_ORCH_ROOT=\$RSR_ORCH_ROOT > …"
OUT=/some/dry/out JOB_SUFFIX=-leftover RSR_ORCH_ROOT=/some/isolated/root
```

`PREREG:1629` says `JOB_SUFFIX` and `OUT` must be unset. The launcher does not enforce this, and
`RSR_ORCH_ROOT` is not mentioned at all. A leftover `RSR_ORCH_ROOT` from a dry run would place all
nine real jobs in an isolated lane root. The battery's thread budget would then be exceeded
silently, and the job records would land outside the main root. `launch-git.json` records none of
the three variables.

*Fix:* the launcher records `OUT`, `JOB_SUFFIX` and `RSR_ORCH_ROOT` in `launch-git.json`, and
refuses (rc 3) when any is set unless an explicit `DRY_RUN=1` is given. Alternatively, the tmux
command wraps the launch in `env -u OUT -u JOB_SUFFIX -u RSR_ORCH_ROOT`.

**MINOR-A2-4. CKPT is an operator-typed "derived" slot.** The launcher requires it to be non-empty
but does not check `CKPT == STEPS/3` (or `STEPS` when not divisible). This is covered by
MINOR-A2-1's fix.

**MINOR-A2-5. The pane's shell is the tmux default-shell, run with `-c`.** That sources the user's
`~/.zshenv` (if any) before `cd $W`, and that file can set environment variables such as threads
or `PYTHONPATH`. The slot forces the thread variables, so this is low risk. Not measured.

---

## Not verified

- **C-1, C-2 and C-3 code**, and the final `feat/rsr-run1-ready` head: the T9 and T10 spellings
  and the `pool_doc_ids` field. They were not read or run.
- **`run.py`**, which does not exist: C9, the batched oracle wrapper (T13), the preflight, and
  every other run-time control.
- **The `O5=eval` launch path end to end.** It cannot run without C-1. Only its refusals were
  tested.
- **A real terminal hang-up and a macOS logout or sleep.** My probe sent SIGHUP to a new session's
  process group. That models a terminal hang-up but is not one. Whether a detached tmux server
  survives a user logout was not tested.
- **The author's `hup_tmux2.py`, `oracle_gamma.py` and E21/E20v2 logs** were not audited line by
  line. I re-ran their claims with my own scripts instead.
- **What I have seen, declared:** model-free oracle victims and hit rates on EVAL documents
  `[995000, 995016)` of seeds 0–2. These are document-structure facts with no model. No EVAL
  document went through any model, and no φ was trained. The dry run ran `--iters 0` only.

Scratchpad (private, not committed):
`/private/tmp/claude-501/-Users-keanooo7-retrieval-successor-retention/f5f7e6f8-2469-4fc5-a810-e91d91e81a95/scratchpad/review-run1-prereg/a2/`.
It holds:

- the hang-up probe: `hup_driver.py`, `hup_launcher.zsh`, `hup_driver.out` and `hup_out/`;
- the launcher and its refusals: `launch_v2_from_prereg.zsh`, `ref_*.log` and `dirty2/`;
- the tmux dry run: `dry2.launcher.log`, `dry2.launcher.rc` and `dry2_out/`;
- the oracle check: `oracle_review.py` and `oracle_review.out`;
- the environment probe: `tmux_env.out`;
- the isolated root `orch2/`.

The dry run used `../a1/orch1` as its isolated root. Nothing was deleted.
