---
id: retrieval-curve
item: retrieval-curve
baseline_sha: 22d66bb216e08b3ff68cef08228d5c57a46b0b9b
falsifier: >-
  Training S0-03's configuration for longer does not make the working memory help answers
  at 2 <= gap <= M: at no checkpoint up to 3000 iterations does every seed show
  slots_zeroed - live answer NLL >= 0.10 on held-out documents with bar 1 holding.

anchors:
  # the instrument and the rule, reused unchanged
  - {path: experiments/s0-03-rewardable-corpus/run.py, line: 100, expect: "MARGIN = 0.10"}
  - {path: experiments/s0-03-rewardable-corpus/run.py, line: 135, expect: "def single(seed: int, iters: int, device: str, out_dir: Path) -> dict:"}
  - {path: experiments/s0-03-rewardable-corpus/run.py, line: 260, expect: "def measure(ckpt: Path, seed: int, device: str) -> dict:"}
  - {path: experiments/s0-03-rewardable-corpus/run.py, line: 296, expect: "def decide(per: dict[int, dict]) -> dict:"}
  - {path: experiments/s0-03-rewardable-corpus/run.py, line: 311, expect: "# H4: retrieval shown iff live beats slots-zeroed by MARGIN at 2 <= gap <= M."}
  - {path: experiments/s0-03-rewardable-corpus/PREREG.md, line: 52, expect: "**H4 (retrieval shown):**"}
  - {path: experiments/decisive-shuffle/run.py, line: 328, expect: "def measure(ckpt: Path, seed: int, *, decoy_ckpt: Path | None = None) -> dict:"}
  - {path: experiments/decisive-shuffle/PREREG.md, line: 73, expect: "carries the previous sentence outside the memory"}
  # the training loop's checkpointing
  - {path: src/rsr/train/loop.py, line: 268, expect: "def train("}
  - {path: src/rsr/train/loop.py, line: 282, expect: "ckpt_every: int = 25,"}
  - {path: src/rsr/train/loop.py, line: 497, expect: "if ckpt_every and (it + 1) % ckpt_every == 0:"}

premises:
  - claim: no retrieval-curve implementation exists yet
    check: "test -e experiments/retrieval-curve/run.py"
    expect_rc: 1
  - claim: no longer-than-300-iteration run of this configuration has a ledger
    check: "test -e runs/retrieval-curve"
    expect_rc: 1
  - claim: S0-03 returned retrieval not shown
    ledger: runs/s0-03-rewardable-corpus/ledger.json
    key: verdict.outcome
    expect: inconclusive
  - claim: S0-03 ran on CPU (the reproduction control needs the same device)
    ledger: runs/s0-03-rewardable-corpus/ledger.json
    key: device
    expect: cpu
  - claim: the decisive run found memory live on at least one arm
    ledger: runs/decisive-shuffle/ledger.json
    key: verdict.outcome
    expect: falsified
  - claim: the loop never deletes a checkpoint, so one run can keep 300/1000/3000
    check: "git grep -n -i -E 'keep_last|max_to_keep' -- src/rsr/train/"
    expect_rc: 1

files_in_scope:
  - path: experiments/retrieval-curve/run.py
    new: true
  - path: experiments/retrieval-curve/RESULTS.md
    new: true
  - path: tests/test_retrieval_curve.py
    new: true
  - scripts/mutation_battery.py

bar:
  - "The ckpt-300 reproduction control is computed per seed as soon as ckpt-000300.pt exists, against runs/s0-03-rewardable-corpus/ledger.json heldout.live.gap_2_to_M.answer_nll (rows are a list, looked up by key; samples are in seed order 0,1,2); tolerance 1e-6. On failure: terminate all training children, verdict inconclusive, exit 3, without waiting for 3000."
  - "S0-03's measure() and decide() are imported, not copied: git grep -n 'def decide' -- experiments/retrieval-curve/ returns nothing."
  - "runs/retrieval-curve/ledger.json holds, per checkpoint in {300,1000,3000} and per seed, every S0-03 bucket for live and slots_zeroed on heldout and train, plus the decisive measure()'s ratio, random_ratio and cross-row cosine; verdict.outcome is one of survived/falsified/inconclusive per the PREREG table, and S0-03's own per-checkpoint label is recorded under a separate key s003_outcome (its 'survived' is this PREREG's 'falsified')."
  - "uv run pytest -rs --tb=no census line and $? quoted; ruff check and ruff format --check exit 0; scripts/mutation_battery.py N/N line quoted."
  - "Three mutations, each reddening only its declared test: (1) decide() fed the train population instead of heldout; (2) the reproduction control's tolerance widened or its comparison ledger swapped; (3) every checkpoint label measured from the final checkpoint (the step stored in each checkpoint must equal its label)."
done_when:
  - "RESULTS.md names the falsifier, the verdict, the checkpoint that carried it (if falsified), and whether the cognitive claim moved (it cannot move here: say so)."
  - "provenance: <git sha> · cpu · <corpus sha256 per seed> · seeds 0,1,2 — stamped from the tree, never typed."
  - "Every number in RESULTS.md is a ledger key; render_scoreboard.py --audit exits 0 on it."
  - "BRIEF ERRORS written out, 'none' if none."
do_not:
  - "Edit experiments/retrieval-curve/PREREG.md, S0-03's run.py, or the decisive run's run.py. Import them."
  - "Choose, move or add any threshold. MARGIN, CHANCE, the checkpoints and the tolerance are in the PREREG."
  - "Read gap_eq_1 as retrieval (bos copy, decisive PREREG Amendment 1)."
  - "Change the corpus, the objective, the hinge, n_documents or bos_replacement_mode. One factor: iters."
  - "Run on MPS. The reproduction control needs CPU."
  - "Change the thread count: 5 per seed, 3 seeds as parallel children, exactly S0-03's (manifest threads_per_seed). Bit-exactness holds per thread count only."
  - "Run any measurement inside a training process: the decisive measure() calls torch.manual_seed. Measure in the parent, after that seed's training process has exited or on a checkpoint file already on disk."
---

# Brief: the retrieval curve — does longer training make memory help answers?

**Status:** written, not started. **Written at:** `22d66bb216e08b3ff68cef08228d5c57a46b0b9b`
(re-derive at spawn; the PREREG commit comes first and becomes the base).
**Lane:** researcher. CPU. **Governing PREREG:** `experiments/retrieval-curve/PREREG.md`.

## Why

The decisive run (`runs/decisive-shuffle/`, #34) and S0-03 (`runs/s0-03-rewardable-corpus/`,
#30) together say the working memory is **used** but does **not help the model answer** at
`2 ≤ gap ≤ M`. Every cognitive experiment downstream (E0d, E1, E3, E7) needs a memory whose
contents change answers, or no retention policy has anything to win. Both runs stopped at 300
iterations, and "behaviour past 300 iterations" is listed as not established. This brief
measures exactly that and nothing else.

The gap-1 bucket is below chance in both runs, but not through memory: at `gap = 1` the bos copy
carries the previous sentence (`experiments/decisive-shuffle/PREREG.md:73`).

## Falsifier

In the front matter. Decision rule, thresholds and checkpoints: the PREREG, verbatim. The rule is
S0-03's `decide()` (`experiments/s0-03-rewardable-corpus/run.py:296`, H4 at `:311`, `MARGIN` at
`:100`) applied at each checkpoint.

## Files in scope

- `experiments/retrieval-curve/run.py` (**new**). One training per seed to 3000 iterations via
  `rsr.train.loop.train` (`src/rsr/train/loop.py:268`) with S0-03's `CONFIG` imported and
  `ckpt_every=100`. Launch the three seeds as parallel child processes at `OMP_NUM_THREADS=5` /
  `MKL_NUM_THREADS=5` each, the way S0-03's `main()` does, and record the thread count in the
  manifest. The loop writes `ckpt-{step:06d}.pt` at every multiple (`:497`) and never
  prunes, so 300, 1000 and 3000 all survive. At each of the three: S0-03's `measure()` (`:260`),
  then S0-03's `decide()`; then the decisive run's `measure()`
  (`experiments/decisive-shuffle/run.py:328`) with `decoy_ckpt=None`: it builds the untrained same-seed decoy itself. Passing a checkpoint there is the aliasing mutation. Exit through
  `rsr.exit_codes` (`run_main`/`Exit`), like S0-05's conversions: 0 verdict reached, 3 did not
  run or reproduction control failed.
- `experiments/retrieval-curve/RESULTS.md` (**new**). Written from the ledger.
- `tests/test_retrieval_curve.py` (**new**). Tests for the three mutations in the bar, and the
  per-checkpoint step check.
- `scripts/mutation_battery.py`. Three entries, each declaring any off-gate coupling with a
  reason.

## Bar

In the front matter, numbered there. The **reproduction control is the one gate that fails where
nothing else does**: a curve that silently isn't S0-03's configuration would pass every other
check here.

## Budget

S0-03 took 20.5 min wall for 3 seeds × 300 iterations, measurement included
(`runs/s0-03-rewardable-corpus/ledger.json` `started_utc`→`finished_utc`). **Prediction, not
premise:** about 3.5 h for 3000 iterations, if cost is linear. The PREREG's 8 h wall deadline
caps it. `train()` has no deadline hook, so the **parent** enforces it: it polls the checkpoint
files, measures each pre-registered checkpoint as it appears (ckpt-300 first, which is the
control), and at the deadline terminates the children and measures whatever pre-registered
checkpoints exist on disk. Report the checkpoints reached; the verdict table covers that case.

⚠️ **Anchor hazard.** Queue item `eng-s003-run-exit-protocol` edits S0-03's `run.py`. If it lands
first, the line anchors above move and lint refuses dispatch. Re-derive them; do not dispatch around
the refusal.

## Done when

In the front matter. The report also carries the manifest's pre-registered expectation line,
written before the run.

## Do NOT

In the front matter. Also (`CLAUDE.md`): stage by explicit path, never `git add -A`; literal
output or it did not happen; an sd of exactly 0.0000 across seeds is broken, not clean.
