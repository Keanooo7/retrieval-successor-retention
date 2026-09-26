---
id: corpus-size-curve
item: corpus-size-curve
baseline_sha: 0c6704245c20dd53d685c98d7d15b717ea97712c
falsifier: >-
  With a training set too large to memorise (N = 4096 documents; S0-03's configuration
  otherwise), the working memory does not help held-out answers at 2 <= gap <= M: at
  neither checkpoint (300, 1000) does every seed show Brier16(slots_zeroed) -
  Brier16(live) >= 0.05 on the common held-out set with bar 1 holding.

anchors:
  # the corpus size is hardcoded today
  - {path: src/rsr/train/loop.py, line: 268, expect: "def train("}
  - {path: src/rsr/train/loop.py, line: 293, expect: "probe = generate(SyntheticConfig(sentences_per_document=steps_per_stream, seed=seed))"}
  - {path: src/rsr/train/loop.py, line: 352, expect: "docs = generate(SyntheticConfig(sentences_per_document=steps_per_stream, seed=seed))"}
  - {path: src/rsr/data/synthetic.py, line: 149, expect: "n_documents: int = 64"}
  # prefix stability: one RNG per document
  - {path: src/rsr/data/synthetic.py, line: 237, expect: "cfg.seed * 1_000_003 + doc_id"}
  # the instrument and the rule, reused
  - {path: experiments/s0-03-rewardable-corpus/run.py, line: 100, expect: "MARGIN = 0.10"}
  - {path: experiments/s0-03-rewardable-corpus/run.py, line: 226, expect: "out_r16.append(-lp16.gather(1, pos.unsqueeze(1)).squeeze(1))"}
  - {path: experiments/s0-03-rewardable-corpus/run.py, line: 260, expect: "def measure(ckpt: Path, seed: int, device: str) -> dict:"}
  - {path: experiments/s0-03-rewardable-corpus/run.py, line: 296, expect: "def decide(per: dict[int, dict]) -> dict:"}
  - {path: experiments/retrieval-curve/run.py, line: 306, expect: "def reproduction_control(seed: int, s003_measured: dict, reference: dict) -> dict:"}

premises:
  - claim: no corpus-size-curve implementation exists yet
    check: "test -e experiments/corpus-size-curve/run.py"
    expect_rc: 1
  - claim: no corpus-size-curve run has a ledger
    check: "test -e runs/corpus-size-curve"
    expect_rc: 1
  - claim: train() has no n_documents argument yet
    check: "git grep -n 'n_documents' -- src/rsr/train/loop.py"
    expect_rc: 1
  - claim: S0-03 ran on CPU (the reproduction control needs the same device)
    ledger: runs/s0-03-rewardable-corpus/ledger.json
    key: device
    expect: cpu
  - claim: S0-03 returned retrieval not shown at 300 iterations
    ledger: runs/s0-03-rewardable-corpus/ledger.json
    key: verdict.outcome
    expect: inconclusive

files_in_scope:
  - path: experiments/corpus-size-curve/run.py
    new: true
  - path: experiments/corpus-size-curve/RESULTS.md
    new: true
  - path: experiments/corpus-size-curve/BRIEF-ERRORS.md
    new: true
  - path: tests/test_corpus_size_curve.py
    new: true
  - src/rsr/train/loop.py
  - experiments/s0-03-rewardable-corpus/run.py
  - scripts/mutation_battery.py

bar:
  - "(1) train() gains n_documents: int | None = None. None is today's call byte for byte: both generate() calls get SyntheticConfig(sentences_per_document=S, seed=seed), and frozen gets no n_documents key, so the default config_hash equals S0-03's per-seed hash in its ledger. The N=64 arm omits the argument."
  - "(2) Held-out is documents [4096, 4160) of each seed's generator in every arm: disjoint from [0, N) by index and content, identical across arms; a test proves it, and proves generate() prefix-stable."
  - "(3) Brier16 per answer target in S0-03's answer_readout: sum over the 16 symbols of (p_k - 1[k=y])^2 with p renormalised over the symbols; its own test on a known distribution; S0-03's existing numbers unchanged (the reproduction control proves it)."
  - "(4) The verdict reads the N=4096 arm only, per the PREREG table row for row; S0-03's decide() is imported and fed held-out only."
  - "(5) The N=64 ckpt-300 reproduction control uses S0-03's measure() with NO override against runs/s0-03-rewardable-corpus/ledger.json within 1e-6; failure stops every arm, exit 3."
  - "uv run --extra dev pytest -rs --tb=no census line and $? quoted; ruff check and ruff format --check exit 0; mutation_battery.py --check-anchors exit 0; each new mutation shown to redden its gate."
done_when:
  - "RESULTS.md names the falsifier, the verdict, the checkpoint that carried it (if falsified), and says the cognitive claim cannot move here."
  - "provenance: git sha, cpu, corpus sha256 per arm and seed, seeds 0,1,2 -- stamped from the tree, never typed."
  - "Every number in RESULTS.md is a ledger key; render_scoreboard.py --audit exits 0 on it."
  - "BRIEF-ERRORS.md written, 'none' if none."
do_not:
  - "Edit experiments/corpus-size-curve/PREREG.md above an appended amendment, or move any threshold."
  - "Change anything but n_documents: not the objective, hinge, bos_replacement_mode, thread count, iters per arm, or device."
  - "Read gap_eq_1 as retrieval (bos copy, decisive PREREG Amendment 1)."
  - "Run any measurement inside a training process."
  - "Touch psi_hat, the retention loss, age or b; make a scaling claim; fill section 15."
  - "Run arms in parallel, or any training the PREREG does not cover."
---

# Brief: the corpus-size curve — does memory's held-out benefit survive a training set too large to memorise?

**Status:** written, implementation in progress. **Written at:**
`0c6704245c20dd53d685c98d7d15b717ea97712c` (the PREREG commit, which is the base).
**Lane:** researcher. CPU. **Governing PREREG:** `experiments/corpus-size-curve/PREREG.md`.
Written by a model (the Studio overnight session, 2026-09-24) under Brendan's go.

## Why

The retrieval curve (PR #43, `runs/retrieval-curve/ledger.json` on
`run/retrieval-curve-2026-09-24`) could not separate retrieval from memorisation. It trained on
64 documents, ~750 passes by step 3000. The training-set answer NLL went to ≈ 0 while held-out
NLL rose to ~8 nats against chance 2.77. Held-out accuracy nonetheless depended on memory at
ckpt1000. Every eviction-policy arm downstream needs a memory that helps on data the model has
not memorised. This brief changes the training-set size and nothing else.

## Falsifier

In the front matter. The decision rule, thresholds, arms, checkpoints and deadline are in the
PREREG, verbatim. The primary readout is Brier16, justified there from the research corpus
(its section *Why Brier is primary*, which cites `03_Calibration_and_Uncertainty.md` by line).

## Files in scope

- `src/rsr/train/loop.py`: `n_documents` in `train()` (`:268`), fed to both `generate()` calls
  (`:293`, `:352`), stamped into `frozen` only when set.
- `experiments/s0-03-rewardable-corpus/run.py`: `answer_readout` gains `brier16` (beside `:226`),
  `_summarise` gains `answer_brier_over_16`, and `measure()` (`:260`) gains an optional `sets=`
  whose default is `_sets(seed)`.
- `experiments/corpus-size-curve/run.py` (**new**): arms 64 → 4096 → 512 in one parent.
  The retrieval curve's `reproduction_control` (`experiments/retrieval-curve/run.py:306`),
  `child_env`, `Child`, `ckpt_path` and `stored_step` are imported.
- `tests/test_corpus_size_curve.py` (**new**), and seven mutation-battery entries.

## Bar

In the front matter. The **reproduction control is the one gate that fails where nothing else
does**. It is also what proves that bar (1)'s default path and bar (3)'s readout change left
S0-03's numbers alone.

## Budget

The retrieval curve took `elapsed_s` 10934.9 s for 3000 iterations at 3 seeds × 5 threads,
measurement included. **Prediction, not premise:** ≈ 1 h per 1000-iteration arm, so ≈ 3–3.5 h
for three arms. The PREREG's absolute deadline, 2026-09-25 07:30 PDT, is enforced by the parent.

## Done when

In the front matter. The report also carries the manifest's pre-registered expectation.

## Do NOT

In the front matter. Also (`CLAUDE.md`): stage by explicit path, never `git add -A`; literal
output or it did not happen; capture `$?` directly.
