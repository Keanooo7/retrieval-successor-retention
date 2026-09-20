status: RETURNED
run_id: s0-02-capture-bridge
updated: 2026-09-20T00:00:00Z
provenance: f85be35 (measurements taken at c81ac35) · mps · Apple M4 Max 64 GB · torch 2.14.0 · random-token synthetic batch, seeds 0,1,2
manifest: runs/s0-02-capture-bridge/manifest.json  (config hash 56baf95a8a4048585733f6496c771e4d2e489fecac6532afbfb36323e8ae8e80)
falsifier: F-capture-free — "capture is off by default and free when off." Secondary: bar item 3, "the capture actually contains `W_O`."
expected: capture_off near E0c's FIFO 374±8 (not the RSR 310); capture_on visibly slower; capture_off unchanged by the bridge's presence; and `r_i` cheaper than the bridge itself.
observed: capture_off 388.80 ± 1.13 sent/s · capture_on 360.46 ± 0.40 (−28.34, −7.29%) · capture_on+r_i 329.96 ± 0.94 (−58.83, −15.13%). Free-when-off survives: the unmodified e0c harness gives 369 ± 11 pre-bridge and 370 ± 9 with it. 🔴 The fourth expectation was FALSIFIED — `r_i` costs MORE than the bridge (−30.5 vs −28.3), because it is 1264 Python-level calls launching small MPS kernels, not a FLOP bill.

---

# S0-02 — the AttentionTrace capture bridge

## gates

```
$ .venv/bin/ruff check .
All checks passed!                                          exit 0

$ .venv/bin/ruff format --check .
148 files already formatted                                 exit 0

$ .venv/bin/pytest -q --no-header
passed=343 failed=0 skipped=0 errors=0                      exit 0
   (20 of those are new: tests/test_capture_bridge.py. Baseline before this
    cycle was passed=323 failed=0 skipped=0.)

$ .venv/bin/python scripts/mutation_battery.py --check
39/39 gates proven by mutation                              exit 0
```

🔴 **Exit codes read from `$?` directly, never after a pipe.** The one place I did
pipe to `tail` and then read `${PIPESTATUS[0]}`, it came back **empty** — this
shell is zsh, where the array is `$pipestatus[1]`. I noticed because the value was
blank rather than `0`; had the idiom silently returned the pipe's status it would
have read `0` and meant nothing. All gate exit codes above were re-taken
unpiped.

## ledger

`runs/s0-02-capture-bridge/ledger.json` — status `ok`, verdict **survived**,
uncapped (all 7 commands resolve to entry points committed at the recorded sha).
Raw artefacts beside it: `capture_cost.json`, `e0c_fifo_with_bridge.json`,
`mutations.json`. Writeup: `experiments/s0-02/RESULTS.md`. Producer:
`experiments/s0-02/write_ledger.py`, committed so the ledger is regenerable.

## numbers

Every figure traces to a ledger key.

| ledger key | value |
|---|---|
| `sent_per_s.capture_off` | 390.08, 387.95, 388.36 → **388.80 ± 1.13** (n=3) |
| `sent_per_s.capture_on` | 360.89, 360.11, 360.38 → **360.46 ± 0.40** (n=3) |
| `sent_per_s.capture_on_ri` | 330.75, 330.21, 328.93 → **329.96 ± 0.94** (n=3) |
| `delta_sent_per_s.capture_on_minus_off` | **−28.34** |
| `delta_pct.capture_on_minus_off` | **−7.29 %** |
| `delta_sent_per_s.capture_on_ri_minus_off` | **−58.83** |
| `delta_pct.capture_on_ri_minus_off` | **−15.13 %** |
| `peak_gb.capture_off / on / on_ri` | 28.75 / 28.79 / 28.80 |
| `observe_calls.capture_off / on / on_ri` | 0 / 1280 / 1280 |
| `r_i_computed.capture_on_ri` | 1264 ( = 1280 − 16; each row's first step has `n_live == 0`) |
| `e0c_fifo_sent_per_s.pre_bridge` | 369 ± 11 (n=3) |
| `e0c_fifo_sent_per_s.with_bridge` | 370 ± 9 (n=3) |
| `mutation.W_O_dropped_from_the_capture_path` | PROVEN — 4 on-gate, **0 off-gate** |
| `mutation.observe()_is_never_reached` | PROVEN — 2 on-gate, **0 off-gate** |
| `mutations.total_proven` | 39 / 39 |

**Config:** `d=128, S=80, batch=16, M=40, V=50257`, 64 tokens/sentence, model in
**eval** mode (correction 20 / D-F), forward + backward, **no optimizer step**,
three arms interleaved round-robin within each repeat, one discarded warm-up per
arm.

### Spread, stated properly

`peak_gb.capture_on` and `peak_gb.capture_on_ri` carry **sd exactly 0.0**, and the
ledger flags them (`sd_exactly_zero`). That is *not* the broken-seed shape: peak GB
is rounded to two decimals over a deterministic allocation, and the **throughput
rows from the same three trials** carry sd 1.13 / 0.40 / 0.94, which is what
proves the repeats genuinely differed. Reported rather than dropped, because a
flagged zero a reader can dismiss is worth more than a zero nobody mentions.

### Which mutation turns only the new tests red

**`W_O dropped from the capture path`** — replace
`wo_vs.append(torch.einsum("bmhk,hkd->bhmd", v, wo))` with
`wo_vs.append(v.permute(0, 2, 1, 3))` in `src/rsr/model/tg/policy_loop.py`.
Four tests red, **all four in the new file**, nothing else in the 343-test suite:

```
tests/test_capture_bridge.py::test_the_trace_has_the_shapes_the_reward_module_declares
tests/test_capture_bridge.py::test_W_O_is_load_bearing_in_the_capture
tests/test_capture_bridge.py::test_wo_v_is_the_projected_value_not_the_raw_value
tests/test_capture_bridge.py::test_the_collapse_reproduces_the_real_increment
```

It needs a mutation rather than a code review because **dropping `W_O` is
shape-compatible**: `reward.contribution` norms over the last axis, and
`[L, H, M, Dh]` norms just as happily as `[L, H, M, D]`. Nothing raises; `r_i`
quietly becomes norm-weighted raw attention, which is v0.1's rejected definition
wearing the new one's name.

🔴 **Bar item 3's own test was vacuous first, in the way bar item 3 warns about.**
The original `test_W_O_is_load_bearing_in_the_capture` perturbed `attn_out_proj`
on *every* cross-attention block and asserted `r_i` moved. Run against the
mutation it exists for, it **stayed green**: layer `l`'s output enters the residual
stream and moves layer `l+1`'s *attention*, so `r_i` changes whether or not `W_O`
is in the capture. The brief says *"if `r_i` is unchanged, the capture is wrong
and the test is vacuous"*; here `r_i` changed, for the wrong reason, which is the
same vacuity from the other side. Caught only because the battery names which
tests a mutation reddens — three *other* tests were silently carrying the gate.
Now the perturbation is confined to the **last** cross-attention block (which is
the last block, so nothing downstream attends to memory) and the test asserts
`alpha` is bit-identical across the intervention, leaving `wo_v` as the only route.

Second mutation, `observe() is never reached` (`if observe:` → `if False:`):
PROVEN, 2 on-gate, 0 off-gate. That is bar item 4.

**Three pre-existing `reward.py` mutations now also redden one new test each**, so
they are **declared with a reason** rather than tolerated (clause 2). The reason is
this cycle's whole point: `test_reward.py` asserts a property on a hand-built
`AttentionTrace`; `test_capture_bridge.py` asserts the same property end to end on
a real forward pass. Those were two disconnected claims until the bridge existed.
A `reward.py` mutation that reddened *only* the fixture test would now mean the
bridge does not reach the reward.

## ADR-0008 — the decision the brief required

`docs/decisions/ADR-0008-qtok-collapse.md`. **`Q_tok` collapses by summing `α` over
query positions where `mask != 0`.**

- **The sum is algebra, not a choice.** `W_O v_i` does not depend on `q`, so slot
  `i`'s total cross-attention increment over the sentence is exactly
  `(Σ_q α_q) · W_O v_i`. Checked, not asserted: ≤ **1.49 × 10⁻⁷** on values up to
  8.23 × 10⁻¹, all six layers (`test_the_collapse_reproduces_the_real_increment`).
- **Does the spec imply an answer? No**, and the ADR says so explicitly. §3.2.1
  writes `α_{l,h,i}` with three indices and never introduces a token index, so the
  notation presupposes the collapse. The nearest guidance is §3.2.1's own
  "report the per-layer profile before collapsing" — an argument that aggregating
  over a heterogeneous axis is a claim. It motivates the ADR; it does not pick a
  value.
- **PAD query positions excluded** — only PAD *keys* are masked, so each pad
  position still emits a full distribution over slots. **`attn_out_proj.bias`
  excluded** — one vector per position, shared by every head and slot.
- **Alternatives and what distinguishes them:** mean is *the same decision* (see
  BRIEF ERRORS 1); EOS-only is the real fork (max Δ`r_i` **0.0383**) and is left
  unimplemented and scheduled as an E0d row rather than as a dead branch.
  `cross_capture(collapse=...)` raises on anything else.

## BRIEF ERRORS

1. 🔴 **"How does `Q_tok` collapse to `[L,H,M]`? … It changes `r_i`, and therefore
   it changes E0d's answer" — half false, and the false half is the one the brief
   leans on.** Of the three candidates the brief names (mean over real tokens, EOS
   only, a sum), **mean and sum cannot change `r_i` at all.** `mean = sum/Q_real`
   with a single `Q_real` shared by every `(l,h,i)`, and §3.2.1's rescale is
   `share_i = raw_i / Σ_j raw_j`, so it cancels exactly. Measured on a real forward
   pass: max |Δ`r_i`| = **2.98 × 10⁻⁸**, while the *unnormalised* `contribution()`
   differs by exactly the factor 5. Only **EOS-only** is a genuine fork (max
   Δ`r_i` 0.0383). The decision is still real and still needed — the brief was
   right to demand an ADR — but two of its three options are the same option, and
   a brief that had been believed as written would have reported a resolved fork
   where there was none.

2. 🔴 **The amendment's bar-4 argument is wrong where it matters.** It says
   *"`observe()` is on the protocol, on `FIFOPolicy` **and** on `RSRPolicy`. A call
   to `policy.observe()` in `run_policy_loop` is reached whichever policy is
   passed."* `RSRPolicy.observe` (`src/rsr/retention/rsr.py:427`) **raises
   `NotImplementedError`** — Sprint 2. And the amendment itself notes that
   `tests/test_checkpoint.py:90,390-393` drive `run_policy_loop` with a real
   `RSRPolicy`. So an *unconditional* `observe()` call — the plain reading of bar
   item 4 — would have turned `test_checkpoint.py` red on contact. Reached is not
   the same as usable. Resolved without routing around it: `observe` is an opt-in
   parameter, `observe=False` by default, which bar item 2 independently requires
   anyway. The two bars are consistent; the amendment's justification for bar 4 is
   not. **`rsr.retention.reward` therefore still has zero real importers in
   `src/`** — the model can now produce its input, but the consumer is still a stub.

3. **`## Files in scope` omits two files the bars cannot be met without.** Bar item
   3 requires a mutation, and mutations live in `scripts/mutation_battery.py`; bar
   item 2 requires a measurement, and CLAUDE.md requires every experiment to write
   a `RESULTS.md` next to its runner. I used `scripts/mutation_battery.py` and
   created `experiments/s0-02/`. Flagged rather than silently widened.

4. **The amendment's `fifo.py:35` has no directory and the obvious one is wrong.**
   `src/rsr/retention/fifo.py` does not exist; `FIFOPolicy` is
   `src/rsr/baselines/fifo.py`, where line 35 is indeed `def observe`. Trivial, but
   the brief's own §"Files in scope" uses full paths and this one does not.

5. **"Done when: … and a PR merged."** Not done, and not mine to do. I was
   instructed not to push to `merge-studio-trunk`, and a merge is an owner
   decision. Eight commits sit on `studio-2026-09-20c`, ready.

6. **`model.py:337-338` → `:337`** — the amendment already caught this and it is
   correct as amended. Confirmed at HEAD: `last_attention` at `:335`,
   `attn_out_proj` at `:337`, `memory_gate` at `:388`, `reward.py` 160 lines,
   11 `def test_`, zero importers.

### Defects found in the machinery, not in the brief

7. 🔴 **`.gitignore`'s `/runs/**` silently ate the artefacts my own ledger cites.**
   I committed the ledger with five `how` fields naming `capture_cost.json`,
   `e0c_fifo_with_bridge.json` and `mutations.json` — **none of which a fresh clone
   had**, because `git add -A` skips ignored files without a word. That is the
   exact failure the `/data/` comment twelve lines above documents at length,
   recurring inside the rule written to fix it. `runs/cycle-01-masked-loss/`'s
   `gates.json` and `raw.json` had already been lost the same way. Fixed with
   `!/runs/**/*.json` (heartbeat `.jsonl`, `.pt` and `.log` stay ignored, verified),
   scratch counters re-ignored by name. 3 MB now tracked under `runs/`, zero `.pt`.

8. **`.orchestrator/outbox/` did not exist.** My own role file says to append the
   report there, and also says — correctly — that there is no `.orchestrator/` in
   this repo. I created the path. That contradiction is in a repo-scoped file and
   should be resolved rather than re-discovered next cycle.

9. **`ledger.command()` refuses `.venv/bin/pytest`.** `TOOL_ENTRY_POINTS`
   whitelists the console-script *name*, but the venv path resolves to a
   repo-relative file that is (correctly) not committed, so the literal command I
   ran is unrecordable. Recorded as `pytest …` with the literal invocation in the
   `note`. Cosmetic, but it means the argv in a ledger is not always the argv that
   ran, which is the property that field exists for.

10. **The rsr-researcher role file's schema warning is stale.** It says
    `manifest.json`, `config_hash`, `seeds_actually_run`, `steps_done` and `status`
    are "not yet written by `scripts/ledger.py`". They all are, and I used them.
    Likewise CLAUDE.md's *"`test_reduction.py` and `test_fidelity.py` are currently
    skipped"*: the suite reports **0 skipped**, and `experiments/GATE-1.md` already
    records both as closed. CLAUDE.md is the stale one.

## UNANSWERED BY THE BRIEF

1. **Whether `observe()` should be on in training.** The brief asks for the call
   site and for capture to be off by default; it does not say who turns it on.
   `src/rsr/train/loop.py:292` still calls `run_policy_loop` without it, so the
   bridge is reachable but unreached in the trainer. That is a Sprint 2 question
   (the MC return) and I did not answer it.
2. **Eval-mode capture inside a train-mode step.** Correction 20 requires `r_i` in
   eval; the LM loss needs train. The bridge reports `eval_mode` honestly and
   `reward` refuses a train-mode trace, so today a training run cannot collect
   `r_i` at all without a second forward pass. Nobody has costed that second pass,
   and it is not the number I measured.
3. **Which `r_i` the policy should store.** `gated` and `raw` are both computable
   from one trace (correction 17 wants both against LOO in E0d); the brief does not
   say which the policy trains on before E0d reports.
4. **Whether the `-7.3 %` is acceptable.** I measured it; the brief did not set a
   budget, and I am not the one who decides what the schedule can absorb.

## BELIEVED, NOT VERIFIED

1. **That `369 ± 11` is the right pre-bridge control.** It is the one number in the
   ledger **transcribed from a terminal rather than read from a JSON file** — the
   worktree was removed and I did not pass `--out`. Flagged as such in its own
   `how` field. Re-derivable in ~40 s: `git worktree add <dir> a1c9e857` and rerun.
2. **That the bridge is correct on CUDA.** Everything here is MPS or CPU. The
   einsums and `no_grad` are device-agnostic by construction, but nobody ran them.
3. **That `sum_over_real_query_tokens` is the right collapse.** The *algebra* is
   verified; the *choice against EOS-only* rests on an argument, not a measurement,
   and E0d is the thing that can settle it. ADR-0008 says so in its own text.
4. **That capture stays free when off under `torch.compile` or on a larger `M`.**
   Measured at one shape only, `d=128 S=80 batch=16 M=40`.
5. **That `RSRPolicy` will be able to consume these traces unchanged.** Its
   `observe` raises today, and the shape it eventually wants is a Sprint 2
   decision. The protocol type-checks; nothing has run through it.
6. **That the three declared couplings are the only ones.** I declared the three
   the battery surfaced. A fourth would only appear if a future `reward.py`
   mutation were added.

## NEXT (proposed, not decided)

1. 🔴 **A batched `retrieval_demand`, before E0d — and it is now measured, not
   guessed.** `r_i` costs −30.5 sent/s against the bridge's −28.3 because it runs
   1264 times from Python. `CrossCapture` already holds `[L, B, H, M]` and
   `[L, B, H, M, D]`, so the batched form is a reshape away and would collapse 1264
   call sites into ~80. Not done here because bar item 1 required `reward.py`'s 11
   tests to pass **unchanged**, and they pin the per-row signature. E0d will call
   this function on every held-out step; paying 15 % there is a schedule decision
   somebody should make deliberately.
2. **E0d, which this brief unblocked** — and per ADR-0008 it should report **four**
   rows, not one: `{gated, raw} × {sum, eos_only}` against LOO Δloss. Both
   collapses come from the *same* captured `[B, H, Q, M]` tensor, so it is two
   reductions of one forward pass, not two runs. If EOS-only correlates better,
   ADR-0008 is wrong and is superseded by the measurement, which is what it says
   should happen.
3. **S0-01, still not landed.** The amendment established that this brief did not
   need it; that is not the same as it being done. `tests/test_train_loop.py` is
   still absent, `--policy` is still missing from `loop.py`'s argparse, `srep_norm`
   is still out of the objective, and `--vocab` still defaults to 50257.
4. **S0-04 is now unblocked in the sense that mattered** — its positive control
   needed `reward.py` to have a live caller. It has a live *producer*; the consumer
   (`RSRPolicy.observe`) is still a stub, so I would not call it unblocked yet.
5. **A second forward pass, costed.** If `r_i` must be collected in eval while the
   LM loss is computed in train (correction 20), somebody has to measure what the
   extra eval pass costs before E3's schedule is believed. Nobody has.
